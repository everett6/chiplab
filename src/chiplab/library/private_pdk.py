"""Private (NDA) PDK profiles: compiled locally, git-ignored, redacted everywhere else.

A foundry Liberty/LEF set is parsed once into a *compiled profile*: numeric summaries and
calibration factors only (FO4, leakage density, pin cap, pitches, site height, macro sizes).
Raw tables, cell / macro / layer / library names and file paths are never stored in the
profile, returned to MCP clients, written to the activity log or the experiment DB.

Profiles live in ``.chiplab_private_pdks/`` (override with CHIPLAB_PRIVATE_PDK_DIR). The
directory is created 0700 with its own ``.gitignore`` ("*"), and the entry is appended to the
enclosing project's .gitignore when there is one. Profiles are addressed by an opaque id
``pdk_<hash>`` derived from the source file hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

DIR_NAME = ".chiplab_private_pdks"
REDACTED = "<redacted>"
_ID = re.compile(r"^pdk_[0-9a-f]{12}$")
_LABEL = re.compile(r"[^A-Za-z0-9 _.-]+")
_MIN_SECRET_LEN = 3


def private_root() -> Path:
    env = os.environ.get("CHIPLAB_PRIVATE_PDK_DIR")
    return Path(env).expanduser().resolve() if env else (Path.cwd() / DIR_NAME).resolve()


def ensure_private_root() -> Path:
    """Create the store (0700) and make sure git ignores it."""
    root = private_root()
    root.mkdir(parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except OSError:  # pragma: no cover - e.g. foreign filesystem
        pass
    inner = root / ".gitignore"
    if not inner.exists():
        inner.write_text("# ChipLab private PDK profiles (NDA): never commit.\n*\n")
    parent = root.parent
    outer = parent / ".gitignore"
    entry = f"{root.name}/"
    if outer.exists() or (parent / ".git").exists():
        lines = outer.read_text().splitlines() if outer.exists() else []
        if entry not in lines and root.name not in lines:
            with outer.open("a") as fh:
                if lines and lines[-1] != "":
                    fh.write("\n")
                fh.write(f"# ChipLab private PDK profiles (NDA)\n{entry}\n")
    return root


# --- redaction -----------------------------------------------------------------------------

def collect_secrets(liberty: dict[str, Any] | None, lef: dict[str, Any] | None,
                    sources: list[dict[str, Any]] | None) -> set[str]:
    """Every proprietary identifier found in parsed inputs: library, cell, macro, pin, layer
    and site names plus file paths/basenames."""
    out: set[str] = set()
    lib = liberty or {}
    for k in ("library", "fo4_reference_cell"):
        if lib.get(k):
            out.add(str(lib[k]))
    out.update(str(c) for c in lib.get("cell_names") or [])
    for key in ("layers", "sites", "macros"):
        for item in (lef or {}).get(key) or []:
            if item.get("name"):
                out.add(str(item["name"]))
            for pin in item.get("pins") or []:
                if pin.get("name"):
                    out.add(str(pin["name"]))
    for s in sources or []:
        if s.get("path"):
            p = Path(str(s["path"]))
            out.update({str(p), p.name, p.stem})
    return {s for s in out if len(s) >= _MIN_SECRET_LEN}


def scrub(obj: Any, secrets: set[str]) -> Any:
    """Recursively replace secret identifiers inside every string (keys included)."""
    if not secrets:
        return obj
    pat = re.compile("|".join(re.escape(s) for s in sorted(secrets, key=len, reverse=True)))

    def _s(v: Any) -> Any:
        if isinstance(v, str):
            return pat.sub(REDACTED, v)
        if isinstance(v, dict):
            return {_s(k): _s(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_s(x) for x in v]
        return v

    return _s(obj)


def redact_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep type, size and a short content hash; drop paths."""
    return [{"type": s.get("type"), "kind": s.get("kind"), "bytes": s.get("bytes"),
             "sha256_prefix": str(s.get("sha256", ""))[:12]} for s in sources]


def redact_args(args: dict[str, Any]) -> dict[str, Any]:
    """Activity-log view of tool arguments: file contents and paths never get logged."""
    out: dict[str, Any] = {}
    for k, v in args.items():
        if v is None:
            out[k] = v
        elif k.endswith("_path") or k.endswith("_paths") or k in (
                "liberty", "lef", "spef", "timing_report", "content"):
            out[k] = REDACTED
        else:
            out[k] = v
    return out


# --- compiled profiles ---------------------------------------------------------------------

def _numeric_liberty(lib: dict[str, Any] | None) -> dict[str, Any]:
    if not lib:
        return {}
    keep = ("nom_voltage", "nom_temperature", "nom_process", "time_unit_ps", "cells",
            "fo4_ps", "leakage_mw_per_mm2_median", "input_cap_ff_median",
            "cell_delay_ps_median", "cell_area_um2_median")
    return {k: lib[k] for k in keep if k in lib}


def _numeric_lef(lef: dict[str, Any] | None) -> dict[str, Any]:
    if not lef:
        return {}
    routing = [ly for ly in lef.get("layers") or [] if ly.get("type") == "ROUTING"]
    blocks = [m for m in lef.get("macros") or [] if str(m.get("class", "")).startswith("BLOCK")]
    return {
        "routing_layers": len(routing),
        "pitches_um": [ly.get("pitch") for ly in routing],      # by metal index, no names
        "wire_r_ohm_per_um": [ly.get("r_ohm_per_um") for ly in routing],
        "wire_c_ff_per_um": [ly.get("c_ff_per_um") for ly in routing],
        "site_height_um": (lef.get("summary") or {}).get("site_height_um"),
        "macros": len(lef.get("macros") or []),
        "block_macros": [{"id": f"macro_{i + 1}", "width_um": m.get("width_um"),
                          "height_um": m.get("height_um"),
                          "signal_pins": sum(p.get("use", "SIGNAL") == "SIGNAL"
                                             for p in m.get("pins") or [])}
                         for i, m in enumerate(blocks)],
    }


def profile_id(node: str, sources: list[dict[str, Any]]) -> str:
    h = hashlib.sha256(node.upper().encode())
    for s in sorted(str(x.get("sha256", "")) for x in sources):
        h.update(s.encode())
    return f"pdk_{h.hexdigest()[:12]}"


def save_profile(node: str, liberty: dict[str, Any] | None, lef: dict[str, Any] | None,
                 calibration: dict[str, Any], sources: list[dict[str, Any]],
                 label: str | None = None) -> dict[str, Any]:
    """Write the compiled (already redacted) profile and return it."""
    secrets = collect_secrets(liberty, lef, sources)
    pid = profile_id(node, sources)
    prof = scrub({
        "pdk_id": pid, "node": node.upper(),
        "label": (_LABEL.sub("", label)[:60] if label else None),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "calibration_id": calibration.get("calibration_id"),
        "factors": {f["name"]: f["value"] for f in calibration.get("factors", [])},
        "liberty": _numeric_liberty(liberty), "lef": _numeric_lef(lef),
        "sources": redact_sources(sources),
        "nda": True,
    }, secrets)
    root = ensure_private_root()
    path = root / f"{pid}.json"
    path.write_text(json.dumps(prof, indent=2, default=str))
    try:
        path.chmod(0o600)
    except OSError:  # pragma: no cover
        pass
    return prof


def load_profile(pdk_id: str) -> dict[str, Any]:
    if not _ID.match(pdk_id):
        raise ValueError("pdk_id must look like 'pdk_<12 hex>'")
    path = private_root() / f"{pdk_id}.json"
    if not path.is_file():
        raise KeyError(f"No private PDK profile {pdk_id}")
    return json.loads(path.read_text())


def list_profiles() -> list[dict[str, Any]]:
    root = private_root()
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.glob("pdk_*.json")):
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append({k: d.get(k) for k in ("pdk_id", "node", "label", "created_at",
                                           "factors")})
    return out
