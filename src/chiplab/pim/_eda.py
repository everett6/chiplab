"""Shared Icarus Verilog / Yosys helpers for the PIM lab (no shell; fixed argv only)."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from chiplab.asic import flow
from chiplab.imc_lab.digital import LIBERTY


class EdaError(RuntimeError):
    pass


def tool(name: str) -> str:
    p = flow._which(name)
    if not p:
        raise EdaError(f"{name} not installed")
    return p


def have(*names: str) -> bool:
    return all(flow._which(n) for n in names)


def iverilog_run(sources: dict[str, str], timeout: float = 300) -> str:
    """Compile and run sources (filename -> text) with Icarus; return stdout."""
    with tempfile.TemporaryDirectory(prefix="chiplab-pim-") as tmp:
        d = Path(tmp)
        for name, text in sources.items():
            (d / name).write_text(text)
        comp = subprocess.run([tool("iverilog"), "-g2005", "-o", "sim", *sources], cwd=d,
                              capture_output=True, text=True, timeout=120)
        if comp.returncode:
            raise EdaError("iverilog failed: " + comp.stderr[-1200:])
        run = subprocess.run([tool("vvp"), "-n", "sim"], cwd=d, capture_output=True, text=True,
                             timeout=timeout)
        if run.returncode:
            raise EdaError("vvp failed: " + (run.stderr or run.stdout)[-800:])
        return run.stdout


def yosys_synth(verilog: str, top: str, timeout: float = 900) -> dict[str, Any]:
    """Synthesize to IHP sg13g2 standard cells; report cells, flops and liberty area (no P&R)."""
    if not LIBERTY.is_file():
        raise EdaError(f"IHP sg13g2 liberty not found at {LIBERTY}")
    with tempfile.TemporaryDirectory(prefix="chiplab-pim-syn-") as tmp:
        d = Path(tmp)
        (d / "top.v").write_text(verilog)
        (d / "s.ys").write_text(
            f"read_verilog top.v\nsynth -top {top} -flatten\nmemory_map\nopt\n"
            f"dfflibmap -liberty {LIBERTY}\nabc -liberty {LIBERTY}\nopt_clean\n"
            f"tee -o stat.txt stat -liberty {LIBERTY}\n")
        r = subprocess.run([tool("yosys"), "-q", "-s", "s.ys"], cwd=d, capture_output=True,
                           text=True, timeout=timeout)
        if r.returncode:
            raise EdaError("yosys failed: " + (r.stderr or r.stdout)[-1200:])
        stat = (d / "stat.txt").read_text()
    cells = (re.search(r"Number of cells:\s+(\d+)", stat)
             or re.search(r"^\s*(\d+)\s+\S+\s+cells\s*$", stat, re.M))
    area = re.search(r"Chip area for (?:top )?module.*?:\s*([\d.]+)", stat)
    flops = sum(int(n) for n in re.findall(r"^\s+(\d+)\s+\S+\s+sg13g2_(?:df|sdf|dl)\w*", stat,
                                           re.M))
    return {"tool": "yosys + abc", "liberty": LIBERTY.name,
            "cells": int(cells.group(1)) if cells else None, "flip_flops": flops or None,
            "cell_area_um2": float(area.group(1)) if area else None}
