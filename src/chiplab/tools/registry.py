"""Find the optional external tools and run them safely.

Lookup order per tool: ``CHIPLAB_<TOOL>`` env var (path to the binary), ``PATH``, then a
user-local unpacked install under ``~/.local/opt/<tool>/usr`` (Debian packages extracted with
``dpkg -x``, which is how this machine has them without root). Local installs get the env
vars they need (library paths, data dirs) set per call.

:func:`run` is the only way ChipLab starts a tool: the program must be one of the known tools
resolved here, arguments are a list (never a shell string), the working directory is a fresh
temporary directory, and there is a timeout. Nothing a web or MCP client sends becomes part
of a command line except validated numbers and ChipLab-generated file names.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

OPT = Path(os.environ.get("CHIPLAB_TOOLS_OPT", Path.home() / ".local" / "opt"))
MULTIARCH = "x86_64-linux-gnu"

# tool key -> (binary name, user-local package dir, purpose)
TOOLS: dict[str, tuple[str, str, str]] = {
    "ngspice": ("ngspice", "ngspice", "SPICE transient simulation for Circuit Studio"),
    "xschem": ("xschem", "xschem", "schematic capture; exports the Circuit Studio schematic SVG"),
    "kicad": ("kicad-cli", "kicad", "PCB export, DRC and 3D/SVG rendering for Board Studio"),
    "openfoam": ("blockMesh", "openfoam", "CFD conjugate heat transfer for the heatsink"),
}


@dataclass
class Tool:
    key: str
    path: str | None
    version: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    note: str = ""

    @property
    def available(self) -> bool:
        return self.path is not None

    def public(self) -> dict[str, Any]:
        return {"available": self.available, "version": self.version,
                "purpose": TOOLS[self.key][2] if self.key in TOOLS else "",
                "source": ("local install (~/.local/opt)" if self.path and str(OPT) in self.path
                           else "system" if self.path else None),
                "note": self.note}


def _local_env(key: str, root: Path) -> dict[str, str]:
    lib = root / "usr" / "lib" / MULTIARCH
    env: dict[str, str] = {}
    if lib.is_dir():
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(lib)] + ([os.environ["LD_LIBRARY_PATH"]] if os.environ.get("LD_LIBRARY_PATH")
                          else []))
    if key == "ngspice":
        share = root / "usr" / "share" / "ngspice"
        env["SPICE_LIB_DIR"] = str(share)
        env["SPICE_SCRIPTS"] = str(share / "scripts")
    elif key == "xschem":
        env["XSCHEM_SHAREDIR"] = str(root / "usr" / "share" / "xschem")
    elif key == "kicad":
        share = root / "usr" / "share" / "kicad"
        env["KICAD9_TEMPLATE_DIR"] = str(share / "template")
    return env


def _find(key: str) -> Tool:
    binary, pkg, _ = TOOLS[key]
    override = os.environ.get(f"CHIPLAB_{key.upper()}")
    if override:
        return Tool(key, override if Path(override).is_file() else None)
    found = shutil.which(binary)
    if found:
        return Tool(key, found)
    root = OPT / pkg
    cand = root / "usr" / "bin" / binary
    if cand.is_file() and os.access(cand, os.X_OK):
        return Tool(key, str(cand), env=_local_env(key, root))
    return Tool(key, None, note=f"{binary} not installed")


def _version(t: Tool) -> str | None:
    args = {"ngspice": ["-v"], "xschem": ["-v"], "kicad": ["version"], "openfoam": ["-help"]}
    try:
        out = run(t, args[t.key], timeout=20).stdout
    except (OSError, subprocess.SubprocessError, ToolError):
        return None
    m = re.search(r"(\d+\.\d+(?:\.\d+)?)", out)
    return m.group(1) if m else None


@functools.lru_cache(maxsize=None)
def get(key: str) -> Tool:
    t = _find(key)
    if t.available:
        t.version = _version(t)
    return t


def klayout() -> dict[str, Any]:
    try:
        import klayout.db as kdb  # noqa: F401
        import klayout.lay  # noqa: F401
    except ImportError:
        return {"available": False, "version": None, "source": None,
                "purpose": "read-only GDS/OASIS layout rendering for Layout Studio",
                "note": "klayout Python package not installed (pip install klayout)"}
    import klayout
    return {"available": True, "version": getattr(klayout, "__version__", None),
            "source": "python package", "note": "",
            "purpose": "read-only GDS/OASIS layout rendering for Layout Studio"}


def status() -> dict[str, Any]:
    out = {k: get(k).public() for k in TOOLS}
    out["klayout"] = klayout()
    return out


class ToolError(RuntimeError):
    pass


def run(tool: Tool, args: list[str], *, cwd: str | Path | None = None, timeout: float = 60,
        extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run a resolved tool with a fixed argument list. Never uses a shell."""
    if not tool.available or tool.key not in TOOLS:
        raise ToolError(f"{tool.key} is not installed")
    if not all(isinstance(a, str) for a in args):
        raise ToolError("tool arguments must be strings")
    env = dict(os.environ)
    env.update(tool.env)
    env.update(extra_env or {})
    if tool.key != "kicad":       # batch modes must never open windows
        env.pop("DISPLAY", None)
    return subprocess.run([str(tool.path), *args], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=timeout, shell=False, check=False,
                          stdin=subprocess.DEVNULL)
