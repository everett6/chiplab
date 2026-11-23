"""Elaborate, lint and simulate the generated memory-subsystem RTL with Verilator 5.

Skipped when Verilator is not installed. Besides the passing configurations, a deliberately
broken ECC decoder must make the self-checking testbench fail (the bench is not vacuous).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from chiplab.memory_gen.rtl_generator import generate
from chiplab.schemas.digital_twin import OnChipMemoryConfig

VERILATOR = shutil.which("verilator") or (
    str(Path.home() / ".local/bin/verilator")
    if (Path.home() / ".local/bin/verilator").exists() else None)
pytestmark = pytest.mark.skipif(VERILATOR is None, reason="Verilator not installed")

SOURCES = ["onchip_mem_subsystem_top.sv", "behavioral_bitcell_macro.sv"]


def _write(cfg: dict, d: Path) -> None:
    g = generate(OnChipMemoryConfig.model_validate(cfg), tb_ops=800)
    for name, text in g["files"].items():
        (d / name).write_text(text)


def _run(args: list[str], cwd: Path, timeout: int = 240) -> subprocess.CompletedProcess:
    env = {**os.environ, "PATH": f"{Path(VERILATOR).parent}:{os.environ.get('PATH', '')}"}
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                          env=env)


def _lint(d: Path) -> subprocess.CompletedProcess:
    return _run([VERILATOR, "--lint-only", "-Wall", "-Wno-DECLFILENAME", "-Wno-UNUSEDSIGNAL",
                 "-Wno-UNUSEDPARAM", "--top-module", "onchip_mem_subsystem_top", *SOURCES], d)


def _sim(d: Path) -> subprocess.CompletedProcess:
    build = _run([VERILATOR, "--binary", "--timing", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                  "-j", "4", "--top-module", "tb_onchip_mem", "tb_onchip_mem.sv", *SOURCES], d)
    assert build.returncode == 0, build.stderr[-3000:]
    return _run([str(d / "obj_dir" / "Vtb_onchip_mem")], d)


@pytest.mark.parametrize("cfg", [
    {"technology": "sram6t", "pdk": "sky130", "capacity_kib": 16, "banks": 2, "word_bits": 32,
     "ecc": "none", "num_requesters": 2},
    {"technology": "gc2t", "pdk": "ihp_sg13g2", "capacity_kib": 32, "banks": 4,
     "word_bits": 64, "ecc": "secded", "interleave": "xor", "arbiter": "priority",
     "pim": {"enabled": True, "lanes": 4, "mac_bits": 4}, "num_requesters": 6,
     "axi4_lite_bridge": True},
    {"technology": "gc3t", "pdk": "n28", "capacity_kib": 64, "banks": 16, "word_bits": 64,
     "ecc": "secded", "interleave": "block", "pim": {"enabled": True, "lanes": 4},
     "num_requesters": 8},
], ids=["sram_2bank", "gc2t_ecc_pim_axil", "gc3t_16bank_block"])
def test_generated_rtl_lints_clean_and_testbench_passes(cfg, tmp_path):
    _write(cfg, tmp_path)
    lint = _lint(tmp_path)
    assert lint.returncode == 0, lint.stderr[-3000:]
    run = _sim(tmp_path)
    assert run.returncode == 0 and "TB PASS" in run.stdout, (run.stdout + run.stderr)[-3000:]
    if cfg["ecc"] == "secded":
        n = int(run.stdout.split("responses checked, ")[1].split()[0])
        assert n > 0, "ECC injection phase must produce corrected reads"


@pytest.mark.parametrize("bug,find,replace", [
    ("ecc_decoder_never_corrects", "fixed[syn] = ~code[syn];", "fixed[syn] = code[syn];"),
    ("refresh_disabled", "assign do_ref = GAIN_CELL && ref_pending;", "assign do_ref = 1'b0;"),
    ("pim_accumulate_drops_sum",
     "s1_op[b] == 2'd3) acc <= acc + dot;", "s1_op[b] == 2'd3) acc <= dot;"),
])
def test_testbench_catches_injected_rtl_bugs(bug, find, replace, tmp_path):
    _write({"technology": "gc2t", "pdk": "sky130", "capacity_kib": 16, "banks": 4,
            "word_bits": 32, "ecc": "secded", "pim": {"enabled": True, "lanes": 2},
            "num_requesters": 3}, tmp_path)
    top = tmp_path / "onchip_mem_subsystem_top.sv"
    text = top.read_text()
    assert find in text, bug
    top.write_text(text.replace(find, replace))
    run = _sim(tmp_path)
    assert "TB PASS" not in run.stdout and run.returncode != 0, f"{bug} was not detected"
