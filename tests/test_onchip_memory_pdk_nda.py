"""PDK ladder, on-chip memory model + simulator, RTL generator, twin extensions, private-PDK
NDA redaction, the design_onchip_memory_subsystem MCP tool and the app's Lab endpoints."""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

import chiplab.mcp_server as srv
from chiplab.activity import ActivityLog, ActivityMiddleware
from chiplab.app.server import create_app, summarize
from chiplab.backends.handoff import structural_lint
from chiplab.cost import prices
from chiplab.cost.reference import node_for_nm
from chiplab.engine import digital_twin as dt
from chiplab.engine.validator import validate_design
from chiplab.experiments.store import ExperimentStore
from chiplab.library import pdk_library as pl
from chiplab.library import private_pdk as pp
from chiplab.memory_gen import rtl_generator as rg
from chiplab.models import onchip_memory as om
from chiplab.schemas.common import Provenance
from chiplab.schemas.design import DesignSpec
from chiplab.schemas.digital_twin import (
    DigitalTwinConfig,
    MemWorkload,
    OnChipMemoryConfig,
    default_search_space,
)

FIX = Path(__file__).parent / "fixtures" / "eda"
ROOT = Path(__file__).resolve().parents[1]
SECRETS = ("mini7", "INV_X1", "NAND2_X1", "UCIE_PHY_X16", "TX_DATA", "RX_DATA", "mini.lib",
           "mini.lef", "core7", str(FIX))


@pytest.fixture()
def db(monkeypatch, tmp_path):
    monkeypatch.setenv("CHIPLAB_PRICE_FILE", str(tmp_path / "prices.json"))
    monkeypatch.setenv("CHIPLAB_EXPORT_DIR", str(tmp_path / "exports"))
    monkeypatch.setenv("CHIPLAB_DATA_DIRS", str(FIX))
    monkeypatch.setenv("CHIPLAB_PRIVATE_PDK_DIR", str(tmp_path / "proj" / ".chiplab_private_pdks"))
    (tmp_path / "proj").mkdir()
    fd, path = tempfile.mkstemp(suffix=".db", dir=tmp_path)
    os.close(fd)
    monkeypatch.setattr(srv, "store", ExperimentStore(db_path=path))
    return path


# --- PDK ladder ------------------------------------------------------------------------------

REQUIRED_MARGIN = {"gf180": 5, "sky130": 5, "ihp_sg13g2": 5, "n65": 10, "n28": 15,
                   "freepdk15": 15, "asap7": 15, "n3": 20, "n2": 25}


def test_ladder_has_eight_node_rungs_with_required_margins_and_licences():
    assert {p.node_nm for p in pl.PDK_LADDER.values()} == {180, 130, 65, 28, 15, 7, 3, 2}
    for key, margin in REQUIRED_MARGIN.items():
        p = pl.get_pdk(key)
        assert p.accuracy_margin_pct == margin, key
        assert p.citation_source and p.license_type
        assert p.tapeout_ready == (key in ("gf180", "sky130", "ihp_sg13g2"))
    gf = pl.get_pdk("gf180")
    assert gf.license_type == "Apache-2.0" and gf.commercial_use_allowed
    for f in pl.NUMERIC_FIELDS:
        for p in pl.PDK_LADDER.values():
            fld = getattr(p, f)
            assert fld.value > 0 and isinstance(fld.provenance, Provenance), (p.key, f)
    # Smaller nodes are faster and denser.
    order = sorted(pl.PDK_LADDER.values(), key=lambda p: -p.node_nm)
    fo4 = [p.scalar("fo4_delay_ps") for p in order]
    assert fo4[0] > fo4[-1] and order[-1].scalar("gate_density_kgates_mm2") > \
        order[0].scalar("gate_density_kgates_mm2")


def test_sources_doc_is_regenerated_from_module():
    doc = (ROOT / "docs" / "sources_and_accuracy.md").read_text()
    assert doc == pl.generate_sources_doc()
    assert "ASAP7" in doc and "Apache-2.0" in doc


def test_validator_licence_and_predictive_pdk_warnings():
    base = {"name": "x", "chiplets": [{"id": "c", "type": "compute", "process_nm": 7}]}
    research = validate_design(DesignSpec.model_validate({**base, "metadata": {"pdk": "asap7"}}))
    commercial = validate_design(DesignSpec.model_validate(
        {**base, "metadata": {"pdk": "asap7", "intended_use": "commercial",
                              "ip": ["ucie_open_adapter", "cva6"]}}))
    assert any("not tape-out ready" in w for w in commercial.warnings)
    assert any("ucie_open_adapter" in w for w in commercial.warnings)
    assert not any("cva6" in w for w in commercial.warnings)
    assert not any("not tape-out ready" in w for w in research.warnings)
    implicit = validate_design(DesignSpec.model_validate(
        {**base, "metadata": {"intended_use": "commercial"}}))
    assert any("asap7" in w for w in implicit.warnings)
    open_ok = pl.commercial_readiness("sky130", "commercial")
    assert not any("not tape-out ready" in w for w in open_ok)


def test_mature_nodes_in_cost_reference():
    assert node_for_nm(180) == "N180" and node_for_nm(130) == "N130" and node_for_nm(65) == "N65"
    assert node_for_nm(7) == "N7" and node_for_nm(3) == "N3"
    assert 0 < prices.value("wafer.tsmc.N180") < prices.value("wafer.tsmc.N65") < \
        prices.value("wafer.tsmc.N7")


# --- on-chip memory model ---------------------------------------------------------------------

def test_sram_vs_gain_cell_physics():
    sram = om.analyze(OnChipMemoryConfig(technology="sram6t", pdk="n28"))
    gc2 = om.analyze(OnChipMemoryConfig(technology="gc2t", pdk="n28"))
    gc3 = om.analyze(OnChipMemoryConfig(technology="gc3t", pdk="n28"))
    assert sram["metrics"]["refresh_overhead"] == 0 and sram["metrics"]["refresh_power_w"] == 0
    assert gc2["metrics"]["array_area_mm2"] < gc3["metrics"]["array_area_mm2"] < \
        sram["metrics"]["array_area_mm2"]
    assert gc2["metrics"]["refresh_overhead"] > 0
    assert gc3["metrics"]["retention_us"] > gc2["metrics"]["retention_us"]
    cool = om.analyze(OnChipMemoryConfig(technology="gc2t", pdk="n28", temp_c=25))
    hot = om.analyze(OnChipMemoryConfig(technology="gc2t", pdk="n28", temp_c=125))
    assert cool["metrics"]["retention_us"] / hot["metrics"]["retention_us"] == \
        pytest.approx(2 ** (100 / om.RET_HALVING_C), rel=1e-3)
    assert hot["metrics"]["refresh_overhead"] > cool["metrics"]["refresh_overhead"]
    assert gc2["fidelity_disclaimer"] and gc2["provenance"]


def test_secded_and_bank_conflicts():
    assert om.secded_check_bits(64) == 8 and om.secded_check_bits(32) == 7
    secded = om.analyze(OnChipMemoryConfig(ecc="secded", word_bits=64))
    assert secded["ecc_overhead_pct"] == pytest.approx(12.5)
    few = om.analyze(OnChipMemoryConfig(banks=2, num_requesters=8),
                     MemWorkload(requests_per_cycle=4.0))
    many = om.analyze(OnChipMemoryConfig(banks=32, num_requesters=8),
                      MemWorkload(requests_per_cycle=4.0))
    assert few["metrics"]["conflict_rate"] > many["metrics"]["conflict_rate"]
    dual = om.analyze(OnChipMemoryConfig(banks=2, num_requesters=8, ports="2rw"),
                      MemWorkload(requests_per_cycle=4.0))
    assert dual["metrics"]["conflict_rate"] < few["metrics"]["conflict_rate"]
    pim = om.analyze(OnChipMemoryConfig(pim={"enabled": True, "lanes": 8}))
    assert pim["metrics"]["pim_gmacs"] > 0 and pim["pim"]["lanes_per_bank"] == 8


def test_refresh_cannot_keep_up_warning():
    r = om.analyze(OnChipMemoryConfig(technology="gc2t", pdk="n2", capacity_kib=1024, banks=2,
                                      temp_c=125, freq_mhz=200))
    assert r["metrics"]["refresh_overhead"] > 0.2
    assert any("Refresh" in w for w in r["warnings"])


@pytest.mark.parametrize("tech,banks,load,ports", [
    ("sram6t", 8, 1.0, "1rw"), ("gc2t", 8, 2.0, "1rw"), ("gc3t", 16, 3.0, "1rw"),
    ("gc2t", 4, 2.0, "2rw"), ("sram6t", 16, 0.5, "1rw")])
def test_analytical_model_tracks_cycle_simulation(tech, banks, load, ports):
    cfg = OnChipMemoryConfig(technology=tech, pdk="sky130", banks=banks, ports=ports,
                             num_requesters=8)
    wl = MemWorkload(requests_per_cycle=load)
    a = om.analyze(cfg, wl)["metrics"]
    s = om.simulate(cfg, wl, cycles=20_000, seed=3)
    assert abs(a["conflict_rate"] - s["conflict_rate"]) < 0.06
    assert a["served_per_cycle"] == pytest.approx(s["throughput_per_cycle"], rel=0.12)
    assert a["refresh_overhead"] == pytest.approx(s["refresh_overhead"], abs=0.01)
    assert s["missed_refresh_deadlines"] == 0


def test_simulation_is_deterministic_and_trace_is_side_effect_free():
    cfg = OnChipMemoryConfig(technology="gc2t", pdk="n28", banks=4, temp_c=105)
    a = om.simulate(cfg, cycles=3000, seed=7)
    b = om.simulate(cfg, cycles=3000, seed=7, trace_cycles=50)
    assert {k: v for k, v in b.items() if k != "frames"} == a
    assert len(b["frames"]) == 50
    for f in b["frames"]:
        assert len(f["banks"]) == 4 and set(f["banks"]) <= set(".RWFrw")
        assert len(f["req"]) == cfg.num_requesters and set(f["req"]) <= set(".gsf")
        for st, t in zip(f["req"], f["target"]):
            assert (t >= 0) == (st != ".")
    assert "F" in "".join(f["banks"] for f in b["frames"]) or b["refresh_overhead"] == 0
    assert sum(r[1] for r in a["timeline"]["rows"]) == a["served"]
    other = om.simulate(cfg, cycles=3000, seed=8)
    assert (other["served"], other["conflict_rate"]) != (a["served"], a["conflict_rate"])


# --- RTL generator -----------------------------------------------------------------------------

@pytest.mark.parametrize("k", [8, 16, 32, 64, 128])
def test_secded_reference_codec_corrects_single_and_detects_double(k):
    lay = rg.secded_layout(k)
    for d in (0, 1, (1 << k) - 1, 0x5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A & ((1 << k) - 1)):
        c = rg.secded_encode(d, k)
        assert rg.secded_decode(c, k) == (d, "ok")
        for b in range(lay["width"]):
            assert rg.secded_decode(c ^ (1 << b), k) == (d, "corrected")
        for b1, b2 in ((1, 2), (0, lay["width"] - 1), (3, lay["width"] // 2)):
            assert rg.secded_decode(c ^ (1 << b1) ^ (1 << b2), k)[1] == "uncorrectable"


def test_generated_ecc_xor_trees_match_python_layout():
    k = 64
    lay = rg.secded_layout(k)
    sv = rg.gen_ecc_modules(k)
    enc = dict(re.findall(r"c\[(\d+)\] = ((?:c\[\d+\](?: \^ )?)+);", sv))
    for p, cov in lay["cover"].items():
        assert sorted(int(x) for x in re.findall(r"c\[(\d+)\]", enc[str(p)])) == sorted(cov)
    data_map = dict(re.findall(r"c\[(\d+)\] = data\[(\d+)\];", sv))
    assert {int(v): int(c) for c, v in data_map.items()} == dict(enumerate(lay["data_pos"]))
    assert f"[{lay['width'] - 1}:0] code" in sv


@pytest.mark.parametrize("cfg", [
    {"technology": "sram6t", "pdk": "sky130", "banks": 4, "word_bits": 32},
    {"technology": "gc2t", "pdk": "n28", "banks": 8, "ecc": "secded",
     "pim": {"enabled": True, "lanes": 8}, "axi4_lite_bridge": True},
    {"technology": "gc3t", "pdk": "asap7", "banks": 16, "interleave": "block",
     "arbiter": "priority", "capacity_kib": 128},
    {"technology": "gc2t", "pdk": "gf180", "banks": 2, "interleave": "low_order",
     "num_requesters": 1, "word_bits": 16, "capacity_kib": 16},
])
def test_rtl_generation_lints_clean_and_is_deterministic(cfg):
    c = OnChipMemoryConfig.model_validate(cfg)
    g1, g2 = rg.generate(c), rg.generate(c)
    assert g1["files"] == g2["files"]
    assert all(v == [] for v in g1["lint"].values()), g1["lint"]
    top = g1["files"]["onchip_mem_subsystem_top.sv"]
    assert "module onchip_mem_subsystem_top" in top and "behavioral_bitcell_macro" in top
    assert ("secded_enc_" in top) == (c.ecc == "secded")
    assert ("module axil_to_mem_req" in top) == c.axi4_lite_bridge
    assert f"NUM_BANKS        = {c.banks}" in top
    gain = c.technology != "sram6t"
    assert (g1["params"]["REFRESH_INTERVAL"] > 0) == gain
    assert "$fatal" in g1["files"]["behavioral_bitcell_macro.sv"]
    tb = g1["files"]["tb_onchip_mem.sv"]
    assert "TB PASS" in tb and "module tb_onchip_mem" in tb
    sdc = g1["files"]["onchip_mem.sdc"]
    assert re.search(r"create_clock -name clk -period [\d.]+", sdc)


def test_rtl_rejects_unrefreshable_and_bad_configs():
    with pytest.raises(ValueError):
        rg.rtl_params(OnChipMemoryConfig(technology="gc2t", pdk="n2", capacity_kib=1024,
                                         banks=2, temp_c=125), 2000.0)
    with pytest.raises(ValueError):
        rg.rtl_params(OnChipMemoryConfig(banks=1), 100.0)


def test_write_bundle_manifest_and_path_safety(db):
    b = rg.write_bundle(OnChipMemoryConfig(technology="gc2t"), output_name="../../evil")
    out = Path(b["directory"])
    assert out.parent == Path(os.environ["CHIPLAB_EXPORT_DIR"]).resolve()
    man = json.loads((out / "manifest.json").read_text())
    assert set(man["files"]) >= {"onchip_mem_subsystem_top.sv", "tb_onchip_mem.sv",
                                 "onchip_mem.sdc", "behavioral_bitcell_macro.sv"}
    assert all(v == ["PASS"] for v in b["lint"].values())


def test_structural_lint_catches_new_mismatches():
    assert "case/endcase mismatch" in structural_lint("module m; always_comb case (a) endmodule")
    assert any("initial" in i for i in structural_lint("module m; initial x = 1; endmodule"))
    assert structural_lint("module m;\n`ifndef SYNTHESIS\ninitial x = 1;\n`endif\nendmodule") \
        == []
    assert structural_lint("module t; initial x = 1; endmodule", testbench=True) == []
    assert structural_lint('module m; $display("(((");\nendmodule') == []


# --- digital twin extensions -------------------------------------------------------------------

def _ladder_cfg(**kw):
    base = dict(search_space=default_search_space("ladder", memory=True),
                memory={"capacity_kib": 1024}, population=24, generations=8, mc_trials=64,
                seed=5)
    return DigitalTwinConfig(**{**base, **kw})


def test_twin_ladder_memory_search_reports_refresh_and_margins():
    s, trace = dt.run_digital_twin(_ladder_cfg())
    assert s.top and s.evaluated == 24 * (8 + 1)          # NSGA-II: population x (gens + 1)
    assert len(_ladder_cfg().search_space) == 15
    for p in s.top:
        assert p.params["node"] in dt.NODE_NAMES and p.params["mem_tech"] in om.TECHS
        assert "refresh_overhead" in p.metrics and "pim_gmacs" in p.metrics
        rob = p.robustness
        assert rob is not None and 0 <= rob.refresh_violation_prob <= 1
        assert rob.accuracy_margin_pct == dt.NODE_MARGIN_PCT[p.params["node"].upper()]
        assert p.params["vdd"] == pytest.approx(
            p.params["vdd_rel"] * dt._NODE["vnom"][dt._node_index(p.params["node"])], abs=1e-3)
    assert {"retention", "mem_traffic"} <= set(s.unknown_parameters)
    s2, _ = dt.run_digital_twin(_ladder_cfg())
    assert [p.params for p in s2.top] == [p.params for p in s.top]


def test_node_priors_apply_pdk_margin_except_calibrated():
    pri = dt.default_priors({"leakage"}, memory=True)
    n2 = {p.name: p for p in dt.node_priors(pri, "n2")}
    sky = {p.name: p for p in dt.node_priors(pri, "sky130")}
    assert n2["gate_delay"].spread_pct == 25 and sky["gate_delay"].spread_pct == 15
    assert n2["leakage"].provenance == Provenance.MEASURED
    assert n2["retention"].spread_pct == 25 and n2["mem_traffic"].spread_pct == 20
    assert n2["theta_ja"].spread_pct == 20  # non-technology factor untouched


def test_wider_pdk_margin_widens_monte_carlo_spread(monkeypatch):
    cfg = DigitalTwinConfig(fixed={"node": "N7", "freq_ghz": 2.0}, search_space=[],
                            mc_trials=400)
    prob = dt._Problem(cfg)
    d = prob.decode([])
    nom = prob.metrics([d])[0]

    def rep(margin):
        monkeypatch.setitem(dt.NODE_MARGIN_PCT, "N7", margin)
        f = dt._sample_factors(dt.node_priors(dt.default_priors(), "N7"), 400, 1, prob.numpy)
        return dt.monte_carlo(prob, d, nom, f, 400, margin)[0]

    narrow, wide = rep(15.0), rep(40.0)
    assert wide.accuracy_margin_pct == 40.0
    assert wide.power_p95_ratio > narrow.power_p95_ratio
    assert wide.worst_slack_ps_p5 < narrow.worst_slack_ps_p5


def test_memory_knobs_need_memory_config_and_validate():
    with pytest.raises(ValueError, match="memory"):
        dt._Problem(DigitalTwinConfig(search_space=default_search_space(memory=True)))
    with pytest.raises(ValueError, match="mem_tech"):
        dt.evaluate_point({"mem_tech": "flash"}, memory=_ladder_cfg().memory)
    m = dt.evaluate_point({"node": "sky130", "freq_ghz": 0.1, "mem_tech": "gc2t"},
                          memory=_ladder_cfg().memory)
    assert m["refresh_overhead"] > 0 and m["mem_area_mm2"] > 0
    assert dt.evaluate_point({"node": "sky130"})["fmax_ghz"] < dt.evaluate_point(
        {"node": "n2"})["fmax_ghz"]


def test_twin_numpy_and_pure_python_agree_with_memory():
    cfg = _ladder_cfg()
    a, b = dt._Problem(cfg), dt._Problem(cfg, force_pure_python=True)
    import random
    rng = random.Random(2)
    ds = [a.decode([rng.random() for _ in a.space]) for _ in range(40)]
    for x, y in zip(a.metrics(ds), b.metrics(ds)):
        for k in x:
            assert x[k] == pytest.approx(y[k], rel=1e-9, abs=1e-12)


def test_pareto_front_improves_over_generations():
    s, trace = dt.run_digital_twin(_ladder_cfg(generations=15, mc_trials=0))
    rows = trace["rows"]
    feas = [r for r in rows if r["violation"] == 0]
    first = max(r["metrics"]["perf"] for r in feas if r["gen"] == 0)
    best = max(r["metrics"]["perf"] for r in feas)
    assert best >= first
    cheap0 = min(r["metrics"]["cost_usd"] for r in feas if r["gen"] == 0)
    assert min(r["metrics"]["cost_usd"] for r in feas) <= cheap0


# --- private PDK store / NDA redaction --------------------------------------------------------

def _no_secrets(obj):
    text = json.dumps(obj, default=str)
    leaked = [s for s in SECRETS if s in text]
    assert not leaked, leaked


def test_ingest_private_pdk_redacts_everything_and_is_gitignored(db, tmp_path):
    (tmp_path / "proj" / ".git").mkdir()
    r = srv.ingest_byo_pdk("N7", liberty_path=str(FIX / "mini.lib"),
                           lef_path=str(FIX / "mini.lef"), label="Foundry X 7nm")
    _no_secrets(r)
    assert re.fullmatch(r"pdk_[0-9a-f]{12}", r["pdk_id"]) and r["nda_redacted"]
    assert r["blackbox_candidates"][0]["name"] == "macro_1"
    assert r["liberty"]["fo4_ps"] == pytest.approx(7.0)
    assert {f["name"] for f in r["factors"]} == {"gate_delay", "leakage"}
    _no_secrets(srv.store.get_calibration(r["calibration_id"]))
    root = Path(os.environ["CHIPLAB_PRIVATE_PDK_DIR"])
    prof_file = root / f"{r['pdk_id']}.json"
    _no_secrets(json.loads(prof_file.read_text()))
    assert (root / ".gitignore").read_text().strip().endswith("*")
    assert ".chiplab_private_pdks/" in (tmp_path / "proj" / ".gitignore").read_text()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(prof_file.stat().st_mode) == 0o600
    assert srv.list_private_pdks()["profiles"][0]["pdk_id"] == r["pdk_id"]
    assert pp.load_profile(r["pdk_id"])["node"] == "N7"
    with pytest.raises(ValueError):
        pp.load_profile("../etc/passwd")
    # calibrated twin still works from the redacted calibration
    t = srv.run_digital_twin_search(calibration_id=r["calibration_id"], population=16,
                                    generations=2, mc_trials=16)
    assert t["calibrated_parameters"] == ["gate_delay", "leakage"]


def test_private_errors_and_timing_reports_hide_names_and_paths(db):
    with pytest.raises(PermissionError) as ei:
        srv.ingest_byo_pdk("N7", liberty_path="/opt/foundry/secret_n3_tt.lib")
    assert "secret_n3" not in str(ei.value) and "/opt" not in str(ei.value)
    inline = (FIX / "mini.lib").read_text().replace("mini7", "acme_secret_lib") \
        .replace("INV_X1", "ACME_INVD1")
    r = srv.ingest_byo_pdk("N7", liberty=inline)
    assert "acme_secret_lib" not in json.dumps(r) and "ACME_INVD1" not in json.dumps(r)
    rep = srv.calibrate_from_eda_report("N7", timing_report=(FIX / "opensta.rpt").read_text(),
                                        liberty_path=str(FIX / "mini.lib"))
    text = json.dumps(rep)
    assert "core/reg_a" not in text and "mini7" not in text and str(FIX) not in text
    assert rep["timing_paths"][0]["startpoint"] == "path_1_start"
    open_rep = srv.calibrate_from_eda_report(
        "N7", timing_report=(FIX / "opensta.rpt").read_text(), private=False)
    assert open_rep["timing_paths"][0]["startpoint"] == "core/reg_a"


def test_activity_log_never_stores_paths_or_file_contents(db):
    log = ActivityLog(db)
    mw = ActivityMiddleware(log)

    async def ok(ctx):
        return {"content": [{"type": "text", "text": json.dumps({"ok": True})}]}

    asyncio.run(mw(SimpleNamespace(method="tools/call", session_id=None, params={
        "name": "ingest_byo_pdk", "arguments": {
            "node": "N7", "liberty_path": "/nda/foundry_n3.lib",
            "lef": "MACRO SECRET_SRAM ...", "keepout_um": 5}}), ok))
    ev = log.recent()[-1]
    args = ev["args"]
    assert args["liberty_path"] == "<redacted>" and args["lef"] == "<redacted>"
    assert args["node"] == "N7" and args["keepout_um"] == 5
    assert "foundry_n3" not in json.dumps(ev) and "SECRET_SRAM" not in json.dumps(ev)


def test_scrub_and_collect_secrets_helpers():
    secrets = pp.collect_secrets({"library": "acme_n3", "cell_names": ["INVD1", "ND2"]},
                                 {"macros": [{"name": "SRAM_HD_2K", "pins": [{"name": "Q"}]}]},
                                 [{"path": "/x/acme_n3_tt.lib"}])
    out = pp.scrub({"note": "FO4 from INVD1 in acme_n3_tt.lib", "SRAM_HD_2K": 1}, secrets)
    assert "INVD1" not in json.dumps(out) and "acme" not in json.dumps(out)
    assert "ND2" in secrets and "Q" not in secrets   # names under 3 chars are not scrubbed
    assert pp.redact_args({"spef_path": "/a", "x": 1}) == {"spef_path": "<redacted>", "x": 1}


# --- MCP tool + app Lab --------------------------------------------------------------------------

def test_design_onchip_memory_subsystem_tool(db):
    r = srv.design_onchip_memory_subsystem(
        {"technology": "gc2t", "pdk": "sky130", "banks": 8, "ecc": "secded",
         "pim": {"enabled": True, "lanes": 8}}, {"requests_per_cycle": 2.0},
        simulate_cycles=10_000, generate_rtl=True, intended_use="commercial",
        ip=["ucie_open_adapter"], trace_cycles=20)
    assert r["metrics"]["refresh_overhead"] > 0 and r["simulation"]["cycles"] == 10_000
    assert len(r["simulation"]["frames"]) == 20
    cc = r["cross_check"]
    assert abs(cc["conflict_rate"]["analytical"] - cc["conflict_rate"]["simulated"]) < 0.06
    assert all(v == ["PASS"] for v in r["rtl"]["lint"].values())
    assert Path(r["rtl"]["directory"], "onchip_mem_subsystem_top.sv").is_file()
    assert any("ucie_open_adapter" in w for w in r["warnings"])
    assert r["fidelity_disclaimer"]
    tools = {t.name for t in asyncio.run(srv.create_mcp_server().list_tools())}
    assert {"design_onchip_memory_subsystem", "list_pdk_ladder", "list_private_pdks"} <= tools
    assert len(srv.list_pdk_ladder()["nodes"]) == len(pl.PDK_LADDER)
    assert "KiB on sky130" in summarize({"ok": True, "tool": "design_onchip_memory_subsystem",
                                         "args": {}, "result": r})


def test_mcp_twin_tool_ladder_and_memory(db):
    r = srv.run_digital_twin_search(node_set="ladder", memory={"capacity_kib": 512},
                                    population=16, generations=3, mc_trials=24)
    km = r["top"][0]["key_metrics"]
    assert "refresh_overhead" in km and "pim_gmacs" in km
    assert "refresh_violation_prob" in r["top"][0]["robustness"]
    with pytest.raises(ValueError):
        srv.run_digital_twin_search(node_set="moon")


def test_app_lab_endpoints(db):
    srv.chip_design_from_template("edge_npu", name="npu")
    c = TestClient(create_app(db))
    meta = c.get("/api/lab/meta").json()
    assert len(meta["ladder"]) == len(pl.PDK_LADDER) and "SKY130" in meta["twin_nodes"]
    m = c.post("/api/lab/memory", json={"config": {"technology": "gc2t", "pdk": "n28"},
                                        "cycles": 2000, "trace_cycles": 999, "rtl": True}).json()
    assert len(m["simulation"]["frames"]) == 400 and "onchip_mem_subsystem_top.sv" in \
        m["rtl"]["files"]
    p = c.post("/api/lab/point", json={"params": {"node": "ASAP7", "vdd_rel": 1.0,
                                                  "mem_tech": "gc3t"},
                                       "memory": {"capacity_kib": 512}}).json()
    assert p["accuracy_margin_pct"] == 15 and p["metrics"]["mem_area_mm2"] > 0
    ds = c.post("/api/lab/design_sim", json={"design_id": "npu_0.1", "fidelity": "analytical",
                                             "workload": {"type": "llm_decode",
                                                          "model_params_b": 1}}).json()
    assert ds["metrics"]["peak_compute_tops"] > 0
    t = c.post("/api/lab/twin", json={"population": 8, "generations": 2, "mc_trials": 8}).json()
    assert c.get(f"/api/twin_runs/{t['run_id']}").status_code == 200
    assert c.post("/api/lab/point", json={}, headers={"origin": "http://evil.test"}).status_code \
        == 403
    assert c.post("/api/lab/point", content=b"{}",
                  headers={"content-type": "text/plain"}).status_code == 403
    bad = c.post("/api/lab/memory", json={"config": {"banks": 3}})
    assert bad.status_code == 400 and "power of two" in bad.json()["error"]
    assert c.get("/api/lab/memory").status_code == 405
