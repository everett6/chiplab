"""BYO-PDK parsers, calibration, IP catalog, black-box macros and handoff bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chiplab.backends import handoff
from chiplab.calibration import pdk_ingestor as pi
from chiplab.library import ip_catalog
from chiplab.schemas.common import Provenance
from chiplab.schemas.digital_twin import BlackBoxMacro

FIX = Path(__file__).parent / "fixtures" / "eda"


def _t(name: str) -> str:
    return (FIX / name).read_text()


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("CHIPLAB_PRICE_FILE", str(tmp_path / "prices.json"))
    monkeypatch.setenv("CHIPLAB_EXPORT_DIR", str(tmp_path / "exports"))


# --- Liberty ---------------------------------------------------------------------------------

def test_liberty_units_cells_and_fo4():
    lib = pi.parse_liberty(_t("mini.lib"))
    assert lib["library"] == "mini7" and lib["cells"] == 2
    assert lib["nom_voltage"] == 0.75 and lib["time_unit_ps"] == 1000.0
    # FO4 load = 4 x 0.5 fF = 2.0 fF (index_2 middle); middle transition row:
    # rise 0.0075 ns, fall 0.0065 ns -> 7.0 ps average.
    assert lib["fo4_ps"] == pytest.approx(7.0)
    assert lib["fo4_reference_cell"] == "INV_X1"
    # leakage density median of 0.6/0.05 = 12 and 1.0/0.08 = 12.5 nW/um2 (= mW/mm2)
    assert lib["leakage_mw_per_mm2_median"] == pytest.approx(12.25)


def test_liberty_interpolates_between_load_points():
    text = _t("mini.lib").replace("capacitance : 0.5 ;", "capacitance : 0.3125 ;", 1)
    lib = pi.parse_liberty(text)  # load 1.25 fF: halfway between 0.5 and 2.0
    rise = 0.005 + (0.0075 - 0.005) * 0.5
    fall = 0.0045 + (0.0065 - 0.0045) * 0.5
    assert lib["fo4_ps"] == pytest.approx((rise + fall) / 2 * 1000)


def test_liberty_rejects_non_liberty_and_never_executes_content():
    with pytest.raises(ValueError):
        pi.parse_liberty("__import__('os').system('echo pwned')")
    payload = 'library (x) { cell (c) { area : __import__("os") ; } }'
    lib = pi.parse_liberty(payload)
    assert lib["cells"] == 1 and lib["cell_area_um2_median"] == 0.0


# --- LEF -------------------------------------------------------------------------------------

def test_lef_layers_sites_macros():
    lef = pi.parse_lef(_t("mini.lef"))
    s = lef["summary"]
    assert lef["dbu_per_micron"] == 2000
    assert s["routing_layers"] == 2 and s["min_pitch_um"] == 0.036
    assert s["site_height_um"] == 0.27 and s["block_macros"] == ["UCIE_PHY_X16"]
    m1 = lef["layers"][0]
    assert m1["r_ohm_per_um"] == pytest.approx(38.0 / 0.018)
    assert m1["c_ff_per_um"] == pytest.approx((0.0002 * 0.018 + 2 * 0.00005) * 1000)
    macro = lef["macros"][0]
    assert (macro["width_um"], macro["height_um"]) == (800.0, 400.0)
    assert [p["name"] for p in macro["pins"]] == ["TX_DATA", "RX_DATA", "VDD"]
    assert macro["obs"] is True


def test_lef_macro_becomes_blackbox_without_power_pins():
    macro = pi.parse_lef(_t("mini.lef"))["macros"][0]
    bb = pi.lef_macro_to_blackbox(macro, keepout_um=10)
    assert bb.area_mm2 == pytest.approx(0.32)
    assert bb.area_with_keepout_mm2 == pytest.approx(0.820 * 0.420)
    assert {p.name: p.direction for p in bb.pins} == {"TX_DATA": "input", "RX_DATA": "output"}
    assert bb.provenance == Provenance.MEASURED


# --- Timing reports and SPEF -----------------------------------------------------------------

def test_opensta_report():
    t = pi.parse_timing_report(_t("opensta.rpt"))
    w = t["worst"]
    assert (w["startpoint"], w["endpoint"]) == ("core/reg_a", "core/reg_b")
    assert w["arrival_ps"] == pytest.approx(130) and w["required_ps"] == pytest.approx(480)
    assert w["slack_ps"] == pytest.approx(350) and w["status"] == "MET"
    assert w["period_ps"] == pytest.approx(500)
    assert w["cell_delay_ps"] == pytest.approx(90) and w["wire_delay_ps"] == pytest.approx(40)
    assert w["logic_stages"] == 3


def test_primetime_report_with_net_lines():
    w = pi.parse_timing_report(_t("primetime.rpt"))["worst"]
    assert w["arrival_ps"] == pytest.approx(210) and w["slack_ps"] == pytest.approx(160)
    assert w["period_ps"] == pytest.approx(400)
    assert w["wire_delay_ps"] == pytest.approx(60)  # 2 net lines + D-pin arrival
    assert w["cell_delay_ps"] == pytest.approx(150)


def test_violated_paths_and_worst_selection():
    rpt = _t("opensta.rpt")
    bad = rpt.replace("0.35   slack (MET)", "-0.05   slack (VIOLATED)")
    t = pi.parse_timing_report(rpt + "\n" + bad)
    assert t["path_count"] == 2 and t["violated"] == 1
    assert t["worst"]["slack_ps"] == pytest.approx(-50)
    with pytest.raises(ValueError):
        pi.parse_timing_report("nothing here")


def test_spef_parasitics():
    s = pi.parse_spef(_t("mini.spef"))
    assert s["nets"] == 2 and s["total_cap_ff"] == pytest.approx(4.0)
    assert s["avg_res_ohm"] == pytest.approx(45.0) and s["max_cap_net"] == "n2"
    with pytest.raises(ValueError):
        pi.parse_spef("*SPEF")


# --- Calibration -----------------------------------------------------------------------------

def test_calibration_factors_from_liberty_and_timing():
    lib = pi.parse_liberty(_t("mini.lib"))
    timing = pi.parse_timing_report(_t("primetime.rpt"))
    rep = pi.calibrate("N7", None, lib, None, timing, pi.parse_spef(_t("mini.spef")))
    f = rep.factor_map()
    assert f["gate_delay"] == pytest.approx(1.0)       # 7 ps FO4 == model N7 FO4 at 0.75 V
    assert f["leakage"] == pytest.approx(12.25 / 10.0)
    assert 0.25 <= f["wire_rc"] <= 4.0
    assert all(x.provenance == Provenance.MEASURED for x in rep.factors)
    assert rep.summary["timing"]["wire_ps"] == pytest.approx(60)


def test_calibration_from_timing_only_and_clamping():
    timing = pi.parse_timing_report(_t("opensta.rpt"))
    rep = pi.calibrate("N5", None, None, None, timing, None)
    assert {"gate_delay", "wire_rc"} <= set(rep.factor_map())
    huge = _t("opensta.rpt").replace("0.04    0.04 ^ core/reg_a/Q", "9.00    9.00 ^ core/reg_a/Q")
    rep2 = pi.calibrate("N5", None, None, None, pi.parse_timing_report(huge), None)
    assert rep2.factor_map()["gate_delay"] == 4.0 and any("clamped" in w for w in rep2.warnings)
    with pytest.raises(ValueError):
        pi.calibrate("N9")


def test_read_input_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("CHIPLAB_DATA_DIRS", str(tmp_path))
    ok = tmp_path / "a.lib"
    ok.write_text("library (x) { }")
    text, src = pi.read_input(path=str(ok))
    assert text.startswith("library") and src["sha256"]
    with pytest.raises(PermissionError):
        pi.read_input(path=str(FIX / "mini.lib"))  # outside the allowed root
    bad = tmp_path / "a.py"
    bad.write_text("x")
    with pytest.raises(ValueError):
        pi.read_input(path=str(bad))
    with pytest.raises(ValueError):
        pi.read_input()
    assert pi.read_input(content="abc")[1]["kind"] == "inline"


# --- IP catalog ------------------------------------------------------------------------------

def test_ip_catalog_entries_are_traceable():
    names = {i["name"] for i in ip_catalog.list_ip()}
    assert {"cva6", "rocket", "xiangshan", "openram", "basejump_stl", "pulp_axi", "tilelink",
            "ucie_open_adapter"} <= names
    for ip in ip_catalog.list_ip():
        assert ip["license"] and "silicon_proven" in ip
        for m in ip["metrics"].values():
            assert m["provenance"] in ("published", "estimated") and m["source"]
    assert ip_catalog.get_ip("ucie_open_adapter")["silicon_proven"] is False
    assert [i["name"] for i in ip_catalog.list_ip("vector_unit")] == ["ara"]
    with pytest.raises(KeyError):
        ip_catalog.get_ip("nope")


def test_blackbox_presets_and_custom():
    phy = ip_catalog.get_blackbox("example_ucie_phy_x16")
    assert phy.provenance == Provenance.ASSUMED and phy.power_mw == 260.0
    custom = ip_catalog.get_blackbox({"name": "pll", "width_um": 100, "height_um": 100,
                                      "kind": "pll"})
    assert isinstance(custom, BlackBoxMacro) and custom.area_mm2 == pytest.approx(0.01)
    with pytest.raises(KeyError):
        ip_catalog.get_blackbox("missing")


# --- Handoff ---------------------------------------------------------------------------------

POINT = {"params": {"cores": 4, "vlen": 256, "dlen": 128, "l1_kb": 32, "l2_kb": 512,
                    "llc_mb": 4.0, "vdd": 0.7, "freq_ghz": 2.0, "node": "N5", "ucie_lanes": 16,
                    "utilization": 0.7},
         "metrics": {"slack_ps": 50.0}, "robustness": {"robustness_score": 97.0}}


def test_handoff_bundle_contents_and_manifest(tmp_path):
    phy = ip_catalog.get_blackbox("example_ucie_phy_x16")
    res = handoff.build_bundle(POINT, "twin_x", 1, [phy])
    out = Path(res["directory"])
    assert out.parent == handoff.export_root()
    expected = {"soc_top.sv", "constraints.sdc", "power.upf", "openroad_flow.tcl",
                "synopsys_dc.tcl", "lint_verilator.sh", "equiv_yosys.ys", "README.md",
                "manifest.json", "structural_lint.txt"}
    assert expected == set(res["files"])
    assert res["structural_lint"] == ["PASS"]
    manifest = json.loads((out / "manifest.json").read_text())
    for name, sha in manifest["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == sha
    sdc = (out / "constraints.sdc").read_text()
    assert "create_clock -name clk -period 0.5000" in sdc
    assert "u_example_ucie_phy_x16/tx_data" in sdc
    upf = (out / "power.upf").read_text()
    assert "PD_EXAMPLE_UCIE_PHY_X16" in upf and "{ON 0.700}" in upf
    sv = (out / "soc_top.sv").read_text()
    assert "(* black_box *) module example_ucie_phy_x16" in sv
    assert "parameter int NUM_CORES = 4" in sv and "ucie_tx" in sv
    assert "verilator --lint-only" in (out / "lint_verilator.sh").read_text()
    assert "equiv_status -assert" in (out / "equiv_yosys.ys").read_text()


def test_structural_lint_catches_errors():
    good = handoff.gen_systemverilog(POINT["params"], [], "top")
    assert handoff.structural_lint(good) == []
    assert any("endmodule" in i for i in handoff.structural_lint(good.replace("endmodule", "", 1)))
    assert any("unbalanced" in i for i in handoff.structural_lint(good + "\n("))
    dup = "module m (input wire a, input wire a);\nendmodule\n"
    assert any("duplicate port" in i for i in handoff.structural_lint(dup))
    assert any("initial" in i for i in handoff.structural_lint(good + "\ninitial x = 1;"))


def test_handoff_output_name_cannot_escape():
    res = handoff.build_bundle(POINT, "run", 1, [], output_name="../../etc/evil")
    assert Path(res["directory"]).parent == handoff.export_root()


def test_no_eval_exec_or_shell_in_new_modules():
    root = Path(__file__).parents[1] / "src" / "chiplab"
    files = [root / "engine" / "digital_twin.py", root / "calibration" / "pdk_ingestor.py",
             root / "backends" / "handoff.py", root / "library" / "ip_catalog.py",
             root / "schemas" / "digital_twin.py"]
    for f in files:
        src = f.read_text()
        for bad in ("eval(", "exec(", "subprocess", "os.system", "os.popen", "pickle"):
            assert bad not in src, f"{bad} in {f.name}"
