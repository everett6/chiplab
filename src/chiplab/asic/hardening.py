"""Import real Tiny Tapeout hardening results (LibreLane on GitHub Actions) into ChipLab.

The TT IHP template builds the GDS in CI. This module reads a finished run's job log through
the GitHub CLI (fixed argument lists, no shell) and pulls out what the flow itself reported:
DRC/LVS checker verdicts, timing-violation corners, antenna counts, utilization and, when the
workflow prints them, LibreLane's final metrics. Nothing is estimated here; a value the log does
not contain stays None.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from chiplab.asic import verify

REPO_DIR = Path(__file__).resolve().parents[3] / "projects" / "ttihp-chiplab-pem"
RTL_IN_REPO = ["pem_core.v", "pem_lane.v", "tt_um_chiplab_pem.v"]

_TS = re.compile(r"^\S+\t[^\t]+\t\d{4}-\d\d-\d\dT[\d:.]+Z ?")   # "job<TAB>step<TAB>timestamp "
_METRIC = re.compile(r"^([a-z][a-z0-9_]*(?:__[a-z0-9_:]+)+),\s*(-?[\d.]+(?:[eE][-+]?\d+)?|inf|-inf|nan)\s*$")
_CORNER = re.compile(r"^\s*\*\s+(nom|min|max)_\w+\s*$")


def _num(s: str) -> float | None:
    try:
        v = float(s)
    except ValueError:
        return None
    return v if v == v and abs(v) != float("inf") else None


def _int(v: float | None) -> int | None:
    return None if v is None else int(v)


def _corners_after(lines: list[str], marker: str) -> list[str]:
    """Corner names listed (one '* corner' per line) after a LibreLane 'X violations found' message."""
    for i, ln in enumerate(lines):
        if marker in ln:
            out = []
            for nxt in lines[i + 1:i + 12]:
                if _CORNER.match(nxt):
                    out.append(nxt.strip().lstrip("*").strip())
                elif out:
                    break
            return out
    return []


def parse_log(text: str) -> dict[str, Any]:
    """Parse a TT gds job log (as printed by `gh run view --log`) into flow-reported facts."""
    lines = [_TS.sub("", ln) for ln in text.splitlines()]
    body = "\n".join(lines)

    def count(label: str) -> int | None:
        m = re.search(rf"^{label} violations:\s*(\d+)", body, re.M)
        return int(m.group(1)) if m else None

    util = re.search(r"IFP-0104\] Effective utilization:\s*([\d.]+)", body)
    metrics: dict[str, float] = {}
    for ln in lines:
        m = _METRIC.match(ln.strip())
        if m and (v := _num(m.group(2))) is not None:
            metrics[m.group(1)] = v
    setup = _corners_after(lines, "Setup violations found in the following")
    hold = _corners_after(lines, "Hold violations found in the following")
    failed = "harden failed" in body or "LibreLane will now quit" in body
    return {
        "flow_completed": not failed,
        "routed": "Check for LVS errors clear" in body,        # LVS only passes on a routed layout
        "drc_clear": "Check for Magic DRC errors clear" in body,
        "lvs_clear": "Check for LVS errors clear" in body,
        "setup_violation_corners": setup,
        "hold_violation_corners": hold,
        "max_slew_violations": "Max Slew violations found" in body,
        "max_cap_violations": "Max Cap violations found" in body,
        "antenna_pins": count("Pin"),
        "antenna_nets": count("Net"),
        "effective_utilization": float(util.group(1)) if util else None,
        "metrics": metrics,
    }


def _gh(args: list[str], cwd: Path) -> str:
    r = subprocess.run(["gh", *args], cwd=cwd, capture_output=True, text=True, timeout=300)
    if r.returncode:
        raise RuntimeError(r.stderr.strip()[-400:] or f"gh {args[0]} failed")
    return r.stdout


def rtl_matches_repo(repo_dir: Path = REPO_DIR) -> bool:
    return all((repo_dir / "src" / f).exists() and (repo_dir / "src" / f).read_bytes() == (verify.RTL / f).read_bytes()
               for f in RTL_IN_REPO)


def _clock_period(repo_dir: Path, sha: str | None) -> float | None:
    """CLOCK_PERIOD from src/config.json at the commit the run built."""
    if not sha or not re.fullmatch(r"[0-9a-f]{7,40}", sha):
        return None
    subprocess.run(["git", "fetch", "-q"], cwd=repo_dir, capture_output=True, timeout=60)
    r = subprocess.run(["git", "show", f"{sha}:src/config.json"], cwd=repo_dir, capture_output=True, text=True,
                       timeout=30)
    try:
        return float(json.loads(r.stdout)["CLOCK_PERIOD"]) if r.returncode == 0 else None
    except (ValueError, KeyError, TypeError):
        return None


def import_run(run_id: int | None = None, repo_dir: Path = REPO_DIR) -> dict[str, Any]:
    """Read the latest (or given) completed `gds` workflow run and return its parsed results."""
    if not repo_dir.is_dir():
        return {"ok": False, "error": f"TT repo clone not found at {repo_dir}"}
    if run_id is not None and (not isinstance(run_id, int) or run_id <= 0):
        raise ValueError("run_id must be a positive integer")
    try:
        if run_id is None:
            runs = json.loads(_gh(["run", "list", "--workflow", "gds", "--status", "completed", "-L", "1",
                                   "--json", "databaseId"], repo_dir))
            if not runs:
                return {"ok": False, "error": "no completed gds run yet"}
            run_id = int(runs[0]["databaseId"])
        info = json.loads(_gh(["run", "view", str(run_id), "--json",
                               "conclusion,headSha,createdAt,updatedAt,url,jobs"], repo_dir))
        jobs = {j["name"]: j for j in info.get("jobs", [])}
        if "gds" not in jobs:
            return {"ok": False, "error": f"run {run_id} has no gds job"}
        log = _gh(["run", "view", str(run_id), "--log", "--job", str(jobs["gds"]["databaseId"])], repo_dir)
    except (RuntimeError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        return {"ok": False, "error": str(e)}
    p = parse_log(log)
    period = _clock_period(repo_dir, info.get("headSha"))
    m = p["metrics"]
    return {
        "ok": True, "run_id": run_id, "url": info.get("url"), "head_sha": info.get("headSha"),
        "finished": info.get("updatedAt"), "conclusion": info.get("conclusion"),
        "jobs": {k: v.get("conclusion") for k, v in jobs.items()},
        "rtl_matches_repo": rtl_matches_repo(repo_dir),
        "clock_period_ns": period, "clock_mhz": round(1000 / period, 3) if period else None,
        **p,
        "antenna_pins": p["antenna_pins"] if p["antenna_pins"] is not None else _int(m.get("antenna__violating__pins")),
        "antenna_nets": p["antenna_nets"] if p["antenna_nets"] is not None else _int(m.get("antenna__violating__nets")),
        "setup_wns_ns": m.get("timing__setup__ws"), "setup_tns_ns": m.get("timing__setup__tns"),
        "hold_wns_ns": m.get("timing__hold__ws"),
        "hardened": all(jobs.get(k, {}).get("conclusion") == "success" for k in ("gds", "precheck", "gl_test")),
        "utilization": m.get("design__instance__utilization", p["effective_utilization"]),
        "utilization_source": "final metrics" if "design__instance__utilization" in m else
                              "floorplan effective utilization (IFP-0104)" if p["effective_utilization"] else None,
        "instance_area_um2": m.get("design__instance__area__stdcell"),   # placed cells, without fill
        "core_area_um2": m.get("design__core__area"), "stdcell_count": m.get("design__instance__count__stdcell"),
        "power_w": m.get("power__total"),
        "route_drc_errors": m.get("route__drc_errors"),
        "fidelity": "LibreLane post-route results from the official TT IHP GitHub Actions flow",
    }
