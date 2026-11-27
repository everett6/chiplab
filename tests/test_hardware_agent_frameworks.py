"""Tests for ChipRO EDA frameworks, MAGE RTL pipeline, CHIA co-design loop, and MCP tools."""

from __future__ import annotations

from pathlib import Path

from chiplab.agent.eda_frameworks import (
    chipro_open_klayout,
    chipro_rtl_lint,
    chipro_run_pnr,
    chipro_tweak_constraints,
)
from chiplab.agent.rtl_co_design import (
    chia_codesign_loop,
    mage_generate_textual_testbench,
    mage_rtl_pipeline,
    mage_static_ast_check,
    mage_waveform_debug,
)
from chiplab.mcp_experimental_tools import (
    chiplab_chia_codesign_loop,
    chiplab_chipro_eda_flow,
    chiplab_local_llm_models,
    chiplab_mage_rtl_pipeline,
)
from chiplab.mcp_server import create_mcp_server


PASSING_ADDER_RTL = """\
module adder4 (
    input  [3:0] a,
    input  [3:0] b,
    output [3:0] sum
);
    assign sum = a + b;
endmodule
"""

BUGGY_ADDER_RTL = """\
module adder4 (
    input  [3:0] a,
    input  [3:0] b,
    output [3:0] sum
);
    assign sum = a - b;
endmodule
"""

SEQ_REG_RTL = """\
module dff_reg (
    input        clk,
    input        rst,
    input  [7:0] d,
    output reg [7:0] q
);
    always @(posedge clk) begin
        if (rst)
            q <= 8'd0;
        else
            q <= d;
    end
endmodule
"""


# ---------------------------------------------------------------------------
# 1. ChipRO EDA Framework Tests
# ---------------------------------------------------------------------------


def test_chipro_rtl_lint_clean_and_broken():
    res_ok = chipro_rtl_lint(PASSING_ADDER_RTL, top_module="adder4")
    assert res_ok["ok"] is True
    assert res_ok["error_count"] == 0
    assert res_ok["top_module"] == "adder4"

    broken_rtl = "module bad (\n  input a,\n  output y\n);\n  always @(posedge clk) begin\n    y = a;\n"
    res_bad = chipro_rtl_lint(broken_rtl, top_module="bad")
    assert res_bad["ok"] is False
    assert res_bad["error_count"] >= 1


def test_chipro_run_pnr_scales_with_node():
    res_130 = chipro_run_pnr(SEQ_REG_RTL, top_module="dff_reg", pdk="sky130", node_nm=130.0)
    res_7 = chipro_run_pnr(SEQ_REG_RTL, top_module="dff_reg", pdk="asap7", node_nm=7.0)

    assert res_130["ok"] is True
    assert res_7["ok"] is True
    assert res_7["core_area_um2"] < res_130["core_area_um2"]
    assert res_7["max_freq_mhz"] > res_130["max_freq_mhz"]


def test_chipro_tweak_constraints():
    sdc_in = (
        "create_clock -name clk -period 10.0 [get_ports clk]\n"
        "set_input_delay 1.0 -clock clk [all_inputs]\n"
        "set_output_delay 1.0 -clock clk [all_outputs]\n"
    )
    res = chipro_tweak_constraints(sdc_in, target_freq_mhz=250.0, input_delay_ns=0.4, output_delay_ns=0.6)
    assert res["ok"] is True
    assert res["period_ns"] == 4.0
    assert "-period 4.0" in res["sdc_text"]
    assert res["clocks_updated"] == 1


def test_chipro_open_klayout(tmp_path: Path):
    gds_file = tmp_path / "design.gds"
    gds_file.write_bytes(b"GDS2")
    res = chipro_open_klayout(str(gds_file))
    assert "klayout_available" in res
    assert res["gds_exists"] is True
    assert "klayout" in res["command"]


# ---------------------------------------------------------------------------
# 2. MAGE / VLSI-AGENT Multi-Agent RTL + Waveform Pipeline Tests
# ---------------------------------------------------------------------------


def test_mage_static_ast_and_testbench_generation():
    ast = mage_static_ast_check(SEQ_REG_RTL)
    assert ast["passed"] is True
    assert ast["module_name"] == "dff_reg"
    port_names = {p["name"] for p in ast["ports"]}
    assert {"clk", "rst", "d", "q"} <= port_names
    assert len(ast["always_blocks"]) == 1
    assert ast["always_blocks"][0]["kind"] == "ff"

    tb = mage_generate_textual_testbench("dff_reg", ast["ports"])
    assert "module tb_dff_reg;" in tb
    assert "WAVE t=%0t" in tb
    assert "$finish;" in tb


def test_mage_rtl_pipeline_passing_rtl():
    vectors = [
        {"a": 2, "b": 3, "expected": {"sum": 5}},
        {"a": 7, "b": 4, "expected": {"sum": 11}},
    ]
    res = mage_rtl_pipeline(PASSING_ADDER_RTL, top_module="adder4", test_vectors=vectors)
    assert res["passed"] is True
    assert res["rounds"] == 1
    assert "WAVE t=" in res["waveform_log"]
    assert "MISMATCH" not in res["waveform_log"]


def test_mage_rtl_pipeline_fixes_buggy_rtl_via_waveform():
    vectors = [
        {"a": 6, "b": 2, "expected": {"sum": 8}},
        {"a": 5, "b": 3, "expected": {"sum": 8}},
    ]
    res = mage_rtl_pipeline(BUGGY_ADDER_RTL, top_module="adder4", test_vectors=vectors, max_fix_rounds=3)
    assert res["passed"] is True
    assert res["rounds"] >= 2
    assert "a + b" in res["final_rtl"]
    assert len(res["agent_trace"]) >= 2

    # Direct check on mage_waveform_debug
    dbg = mage_waveform_debug(
        BUGGY_ADDER_RTL,
        "WAVE t=5 a=6 b=2 sum=4\nMISMATCH t=5 signal=sum expected=8 actual=4\n",
    )
    assert dbg["has_mismatches"] is True
    assert "a + b" in dbg["patched_rtl"]


# ---------------------------------------------------------------------------
# 3. CHIA Cyclic Hardware/Software Co-Design Loop Tests
# ---------------------------------------------------------------------------


def test_chia_codesign_loop_converges():
    res = chia_codesign_loop(
        design_spec={"gate_count": 20000, "pipeline_stages": 1, "logic_depth_fo4": 64.0},
        node_nm=28.0,
        target_freq_mhz=2500.0,
        max_area_um2=50000.0,
        max_power_mw=50.0,
        max_iters=6,
    )
    assert res["converged"] is True
    assert res["iterations"] >= 2
    assert len(res["history"]) == res["iterations"]
    final_m = res["final_metrics"]
    assert final_m["achieved_freq_mhz"] >= 2500.0
    assert final_m["area_um2"] <= 50000.0
    assert final_m["power_mw"] <= 50.0


# ---------------------------------------------------------------------------
# 4. New MCP Tools & Server Registration Tests
# ---------------------------------------------------------------------------


def test_mcp_experimental_framework_tools():
    lint_res = chiplab_chipro_eda_flow(action="lint", rtl_code=PASSING_ADDER_RTL, top_module="adder4")
    assert lint_res["ok"] is True

    pnr_res = chiplab_chipro_eda_flow(action="pnr", rtl_code=SEQ_REG_RTL, top_module="dff_reg", node_nm=28.0)
    assert pnr_res["ok"] is True

    sdc_res = chiplab_chipro_eda_flow(action="tweak_constraints", sdc_text="", target_freq_mhz=200.0)
    assert sdc_res["ok"] is True

    mage_res = chiplab_mage_rtl_pipeline(
        rtl_code=BUGGY_ADDER_RTL,
        top_module="adder4",
        test_vectors=[{"a": 4, "b": 3, "expected": {"sum": 7}}],
    )
    assert mage_res["passed"] is True

    chia_res = chiplab_chia_codesign_loop(
        node_nm=28.0,
        target_freq_mhz=800.0,
        max_area_um2=50000.0,
        max_power_mw=50.0,
        max_iters=5,
    )
    assert chia_res["converged"] is True

    llm_res = chiplab_local_llm_models()
    assert "local_models" in llm_res
    assert "openrouter_free_models" in llm_res
    assert llm_res["openrouter_count"] >= 1


def test_mcp_server_registers_new_tools():
    mcp = create_mcp_server()
    registered = set(mcp._tool_manager._tools.keys())
    expected = {
        "chiplab_chipro_eda_flow",
        "chiplab_mage_rtl_pipeline",
        "chiplab_chia_codesign_loop",
        "chiplab_local_llm_models",
    }
    assert expected <= registered
