"""Open-source EDA runs on the PEM RTL (skip when OSS CAD Suite is not installed)."""

from __future__ import annotations

import pytest

from chiplab.asic import eda, flow

HAVE_YOSYS = flow._which("yosys") is not None
HAVE_SBY = flow._which("sby") is not None


@pytest.mark.skipif(not HAVE_YOSYS, reason="yosys not installed")
def test_generic_synthesis_matches_register_estimate():
    r = eda.synthesize_generic()
    assert r["ok"] and r["cells"] > 1000
    est = flow.register_bits()["total"]
    assert abs(r["flops"] - est) / est < 0.1        # synthesis may prune a few unused bits


@pytest.mark.skipif(not HAVE_YOSYS, reason="yosys not installed")
def test_gate_level_netlist_matches_model():
    r = eda.gate_level_regression(seeds=4, cycles=2000)
    assert r["passed"] == 4
    assert r["protocols"]["passed"] == r["protocols"]["total"]


@pytest.mark.skipif(not HAVE_SBY, reason="sby not installed")
def test_formal_bmc_short_depth_passes():
    r = eda.formal_check(depth=6, mode="bmc")
    assert r["status"] == "PASS", r["log_tail"]


def test_formal_rejects_bad_arguments():
    with pytest.raises(ValueError):
        eda.formal_check(depth=0)
    with pytest.raises(ValueError):
        eda.formal_check(mode="sim")


@pytest.mark.skipif(not HAVE_YOSYS or eda.ihp_liberty("slow") is None, reason="yosys or IHP liberty missing")
def test_ihp_mapping_reports_area_delay_and_tile_fit():
    r = eda.synthesize_ihp("slow", 50)
    assert r["ok"] and r["cell_area_um2"] > 0 and r["logic_delay_ns"] > 0
    assert r["flops"] == eda.synthesize_generic()["flops"]
    w, h = eda.TT_IHP_TILES["8x4"]
    assert r["tiles"]["8x4"]["cell_density"] == round(r["cell_area_um2"] / (w * h), 3)
    assert r["tiles"]["1x1"]["verdict"] == "too small"


def test_tile_table_is_the_official_outline():
    assert eda.TT_IHP_TILES["1x1"] == (202.08, 154.98) and eda.TT_IHP_TILES["8x4"] == (1724.16, 710.64)


@pytest.mark.skipif(flow._which("iverilog") is None, reason="iverilog not installed")
def test_exported_tt_test_sequence_passes_in_icarus():
    r = eda.icarus_tt_test()
    assert r["ok"], r


def test_exported_cocotb_test_uses_current_program():
    src = flow._cocotb_test()
    compile(src, "test.py", "exec")
    a, d = flow.tt_test_writes()[0]
    assert f"(0x{a:02x}, 0x{d:04x})" in src and 'unit="ns"' in src
