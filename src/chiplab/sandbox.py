"""Agent modification sandbox.

An agent that finds a missing capability may propose new code, but never changes ChipLab
directly:

  1. propose()  records the files and tests; paths must be under src/chiplab/ or tests/.
                Files that decide whether experiments succeed (physics models, solvers, the
                review gate, the learning loop, credibility checks) are PROTECTED: a proposal
                touching them needs an explicit physics review at approval.
  2. prepare()  copies the current tree into an isolated sandbox directory and applies the
                proposal there (works with uncommitted trees, unlike a git worktree).
  3. run()      runs the regression suite and the proposal's own tests inside the sandbox,
                and records the counts next to the baseline (the same suite on the main tree).
  4. approve()  is for a human. It requires i_am_human=True, a name, and physics_review=True
                for protected files. It does not apply anything.
  5. apply()    copies approved files into the main tree, refusing if any target changed
                since the proposal was made.

This module is Python API only. Nothing here is exposed to MCP or web clients except
read-only listing, so a remote agent cannot run code or approve its own changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SANDBOX = Path(os.environ.get("CHIPLAB_SANDBOX_DIR", str(ROOT / ".chiplab_sandbox")))
ALLOWED = (re.compile(r"^src/chiplab/[\w/]+\.py$"), re.compile(r"^tests/[\w/]+\.py$"),
           re.compile(r"^docs/[\w/\-]+\.md$"))
PROTECTED = (re.compile(r"^src/chiplab/models/"), re.compile(r"^src/chiplab/engines?/"),
             re.compile(r"^src/chiplab/solvers/"), re.compile(r"^src/chiplab/multiphysics/"),
             re.compile(r"^src/chiplab/science/(review|fidelity|falsify)\.py$"),
             re.compile(r"^src/chiplab/learning/"), re.compile(r"^src/chiplab/credibility\.py$"),
             re.compile(r"^src/chiplab/model_registry\.py$"),
             re.compile(r"^src/chiplab/pim/dram\.py$"), re.compile(r"^src/chiplab/sandbox\.py$"),
             re.compile(r"^tests/conftest\.py$"))


class SandboxError(RuntimeError):
    pass


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def _path(pid: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{10}", pid):
        raise SandboxError("bad proposal id")
    return SANDBOX / "proposals" / f"{pid}.json"


def _load(pid: str) -> dict[str, Any]:
    p = _path(pid)
    if not p.is_file():
        raise SandboxError(f"no proposal {pid}")
    return json.loads(p.read_text())


def _save(rec: dict[str, Any]) -> None:
    p = _path(rec["id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2))


def propose(title: str, rationale: str, files: dict[str, str], tests: list[str],
            author: str, root: Path | None = None) -> dict[str, Any]:
    root = root or ROOT
    if not files:
        raise SandboxError("no files")
    bad = [f for f in files if ".." in f or not any(a.match(f) for a in ALLOWED)]
    if bad:
        raise SandboxError(f"paths not allowed: {bad}")
    missing = [t for t in tests if t not in files]
    if not tests or missing:
        raise SandboxError("every proposal must bring its own tests (listed in files)")
    protected = [f for f in files if any(p.match(f) for p in PROTECTED)]
    rec = {"id": uuid.uuid4().hex[:10], "title": title, "rationale": rationale,
           "author": author, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "files": files, "tests": tests, "protected_files": protected,
           "base_hashes": {f: _sha(root / f) for f in files}, "status": "proposed",
           "root": str(root), "runs": [], "approval": None}
    _save(rec)
    return {k: rec[k] for k in ("id", "status", "protected_files")} | {
        "note": "protected files need a human physics review before approval"
        if protected else ""}


def prepare(pid: str) -> Path:
    rec = _load(pid)
    root = Path(rec["root"])
    dest = SANDBOX / "trees" / pid
    if dest.exists():
        shutil.rmtree(dest)
    for sub in ("src", "tests", "docs"):
        if (root / sub).is_dir():
            shutil.copytree(root / sub, dest / sub,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.db"))
    for f in ("pyproject.toml", "pytest.ini", "setup.cfg"):
        if (root / f).is_file():
            shutil.copy2(root / f, dest / f)
    for rel, text in rec["files"].items():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_text(text)
    rec["status"] = "prepared"
    rec["tree"] = str(dest)
    _save(rec)
    return dest


def _pytest(cwd: Path, args: list[str], timeout: float) -> dict[str, Any]:
    env = os.environ | {"PYTHONPATH": str(cwd / "src"), "CHIPLAB_SANDBOX_RUN": "1"}
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            *args], cwd=cwd, env=env, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "summary": "timeout"}
    tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|error|errors|skipped)",
                                               tail[0])}
    return {"ok": r.returncode == 0, "summary": tail[0], "counts": counts}


def run(pid: str, regression: list[str] | None = None, timeout: float = 1800) -> dict[str, Any]:
    """Regression + new tests inside the sandbox; baseline = same regression on main tree."""
    rec = _load(pid)
    tree = Path(rec.get("tree") or prepare(pid))
    regression = regression or ["tests"]
    probe = subprocess.run([sys.executable, "-c", "import chiplab; print(chiplab.__file__)"],
                           cwd=tree, env=os.environ | {"PYTHONPATH": str(tree / "src")},
                           capture_output=True, text=True, timeout=60)
    if not probe.stdout.strip().startswith(str(tree)):
        raise SandboxError("sandbox would import chiplab from outside the sandbox tree "
                           f"({probe.stdout.strip() or probe.stderr.strip()[-200:]})")
    base = _pytest(Path(rec["root"]), regression + ["--ignore=" + t for t in rec["tests"]
                                                    if (Path(rec["root"]) / t).exists()],
                   timeout)
    reg = _pytest(tree, regression + ["--ignore=" + t for t in rec["tests"]], timeout)
    new = _pytest(tree, rec["tests"], timeout)
    regressed = base["ok"] and not reg["ok"]
    result = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "baseline": base, "regression": reg,
              "new_tests": new, "regressed": regressed,
              "passed": reg["ok"] and new["ok"]}
    rec["runs"].append(result)
    rec["status"] = "awaiting_approval" if result["passed"] else "failed"
    _save(rec)
    return result | {"status": rec["status"]}


def approve(pid: str, approver: str, *, i_am_human: bool, physics_review: bool = False,
            notes: str = "") -> dict[str, Any]:
    rec = _load(pid)
    if not i_am_human or not approver.strip():
        raise SandboxError("approval must come from a named human")
    if approver.strip().lower() == str(rec["author"]).strip().lower():
        raise SandboxError("the author cannot approve their own proposal")
    if rec["status"] != "awaiting_approval":
        raise SandboxError(f"status is {rec['status']}; run the tests first")
    if rec["protected_files"] and not physics_review:
        raise SandboxError(f"protected files {rec['protected_files']} need physics_review=True")
    rec["approval"] = {"by": approver, "physics_review": physics_review, "notes": notes,
                       "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rec["status"] = "approved"
    _save(rec)
    return {"id": pid, "status": "approved"}


def apply(pid: str) -> dict[str, Any]:
    rec = _load(pid)
    if rec["status"] != "approved" or not rec.get("approval"):
        raise SandboxError("only approved proposals can be applied")
    root = Path(rec["root"])
    changed = [f for f, h in rec["base_hashes"].items() if _sha(root / f) != h]
    if changed:
        raise SandboxError(f"files changed since the proposal: {changed}; re-propose")
    for rel, text in rec["files"].items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    rec["status"] = "applied"
    _save(rec)
    return {"id": pid, "status": "applied", "files": list(rec["files"])}


def listing() -> list[dict[str, Any]]:
    d = SANDBOX / "proposals"
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        r = json.loads(p.read_text())
        out.append({k: r.get(k) for k in ("id", "title", "author", "status", "created_at",
                                          "protected_files")} | {"files": list(r["files"])})
    return out
