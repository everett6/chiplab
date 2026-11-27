"""Genealogy, falsification, DFT, technology plugins and the agent sandbox."""

from __future__ import annotations

import pytest

from chiplab import dft, sandbox
from chiplab.brain import Brain
from chiplab.pim import _eda
from chiplab.plugins import technology as tech
from chiplab.science import falsify
from chiplab.science.genealogy import Genealogy


@pytest.fixture()
def gen(tmp_path):
    return Genealogy(Brain(tmp_path / "b.db"))


def test_genealogy_tree_lineage_and_novelty(gen):
    gen.create("PIM-v1", {"mem": "any"})
    d = gen.derive("PIM-v1", "DRAM-PIM", {"mem": "dram", "op": "and"}, why="bulk bitwise",
                   improved={"energy": "-90%"}, evidence=["run_1"])
    assert d["diff"]["mem"] == {"from": "any", "to": "dram"} and not d["warnings"]
    gen.derive("PIM-v1", "SRAM-PIM", {"mem": "sram"}, why="digital shuttle")
    gen.reject("DRAM-PIM", "needs a DRAM process; TRA error rate too high")
    t = gen.tree()
    assert "DRAM-PIM [rejected]" in t["text"] and t["count"] == 3
    nov = gen.check_novelty({"mem": "dram", "op": "and"})
    assert not nov["novel"] and "TRA" in nov["warning"]
    lin = gen.lineage("DRAM-PIM")
    assert [n["name"] for n in lin] == ["PIM-v1", "DRAM-PIM"]
    assert gen.brain.recall("dram pim")["rejected"]


def test_derive_flags_missing_evidence(gen):
    gen.create("A", {"x": 1})
    w = gen.derive("A", "B", {"x": 2}, why="try")["warnings"]
    assert len(w) == 2
    with pytest.raises(ValueError):
        gen.derive("A", "C", {"x": 3}, why=" ")


def test_falsification_plan_and_judgement(tmp_path):
    b = Brain(tmp_path / "b.db")
    b.remember("project", "DRAM needs custom process", "no shuttle", tags=("dram",),
               status="limitation")
    p = falsify.plan("TRA AND cuts energy", "dram", "energy_per_op_j", min_effect=0.5, brain=b)
    assert p["required_evidence"]["min_fidelity"] == "circuit_spice"
    assert any("sense margin" in r for r in p["reasons_it_might_fail"])
    assert any("[limitation]" in r for r in p["reasons_it_might_fail"])
    assert falsify.judge(p, 0.1, 1.0, fidelity="analytical")["outcome"] == "inconclusive"
    assert falsify.judge(p, 0.1, 1.0, fidelity="measured")["outcome"] == "supported"
    assert falsify.judge(p, 0.9, 1.0, fidelity="measured")["outcome"] == "refuted"
    assert falsify.judge(p, 0.1, 1.0, fidelity="measured",
                         review_verdict="rejected")["outcome"] == "inconclusive"
    assert falsify.judge(p, 0.4, 1.0, fidelity="measured",
                         ratio_ci95=(0.3, 1.2))["outcome"] == "inconclusive"


def test_march_coverage_matches_theory():
    c = dft.coverage("March C-", 32, 100)["coverage"]
    assert all(v == 1.0 for v in c.values())
    m = dft.coverage("MATS+", 32, 100)["coverage"]
    assert m["SAF"] == 1.0 and m["AF"] == 1.0 and m["TF"] < 1.0
    assert dft.march_length("March C-", 1) == 10


@pytest.mark.skipif(not _eda.have("iverilog", "vvp"), reason="iverilog missing")
def test_mbist_detects_stuck_at():
    assert not dft.mbist_check(8, 4)["fail_flag"]
    assert dft.mbist_check(8, 4, (3, 2, 1))["fail_flag"]
    assert dft.mbist_check(8, 4, (7, 0, 0))["fail_flag"]


def test_yield_and_scan_and_observability():
    y = dft.yield_models(100, 0.1)
    assert y["poisson"] == pytest.approx(0.9048, rel=1e-3)
    assert y["poisson"] <= y["murphy"] <= y["negative_binomial"]
    o = dft.observability(4096, 8, 8)
    assert o["cycles_to_dump_state"] == 512 and "scan" in o["advice"]
    try:
        s = dft.scan_overhead(100, 10_000)
    except (ValueError, FileNotFoundError):
        pytest.skip("liberty missing")
    assert s["scan_dff_um2"] > s["dff_um2"] and s["added_area_um2"] > 0


def test_seu_parity_catches_single_flips():
    r = dft.seu_injection({"rows": 4, "lanes": 2, "lane_bits": 4}, trials=300)
    assert 0 < r["result_corrupted"] < 1 and r["corruptions_detected"] == 1.0


def test_plugin_admission():
    p = tech.example_resistive_plugin()
    res = tech.check(p)
    assert res["admissible"], res["problems"]
    p.material_properties["r_on"]["unit"] = "furlongs"
    p.validation_tests.append(lambda: ("bad", False, "nope"))
    res = tech.check(p)
    assert not res["admissible"] and len(res["problems"]) >= 2


def test_plugin_hard_coded_model_rejected():
    p = tech.example_resistive_plugin()
    p.device_model = lambda x: {"read_current_a": 2e-5}
    assert not tech.check(p)["admissible"]


def test_sandbox_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "SANDBOX", tmp_path / "sb")
    with pytest.raises(sandbox.SandboxError):
        sandbox.propose("x", "y", {"/etc/passwd": ""}, [], "agent")
    with pytest.raises(sandbox.SandboxError):
        sandbox.propose("x", "y", {"src/chiplab/new.py": "X = 1\n"}, [], "agent")
    r = sandbox.propose("touch physics", "y",
                        {"src/chiplab/models/physics/x.py": "X=1\n",
                         "tests/test_x.py": "def test_x():\n    assert True\n"},
                        ["tests/test_x.py"], "agent", root=tmp_path)
    assert r["protected_files"] == ["src/chiplab/models/physics/x.py"]
    with pytest.raises(sandbox.SandboxError):
        sandbox.approve(r["id"], "everett", i_am_human=True)      # tests not run yet
    rec = sandbox._load(r["id"])
    rec["status"] = "awaiting_approval"
    sandbox._save(rec)
    with pytest.raises(sandbox.SandboxError):
        sandbox.approve(r["id"], "agent", i_am_human=True, physics_review=True)  # self
    with pytest.raises(sandbox.SandboxError):
        sandbox.approve(r["id"], "everett", i_am_human=True)      # needs physics review
    with pytest.raises(sandbox.SandboxError):
        sandbox.apply(r["id"])
    sandbox.approve(r["id"], "everett", i_am_human=True, physics_review=True)
    assert sandbox.apply(r["id"])["status"] == "applied"
    assert (tmp_path / "src/chiplab/models/physics/x.py").read_text() == "X=1\n"


def test_sandbox_runs_tests_in_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "SANDBOX", tmp_path / "sb")
    root = tmp_path / "repo"
    (root / "src/chiplab").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src/chiplab/__init__.py").write_text("")
    (root / "tests/test_base.py").write_text("def test_base():\n    assert True\n")
    r = sandbox.propose("feature", "z", {
        "src/chiplab/feat.py": "def f():\n    return 2\n",
        "tests/test_feat.py": "from chiplab.feat import f\n\ndef test_f():\n    assert f() == 2\n"},
        ["tests/test_feat.py"], "agent", root=root)
    res = sandbox.run(r["id"], ["tests"], timeout=120)
    assert res["passed"] and res["status"] == "awaiting_approval"
    assert not (root / "src/chiplab/feat.py").exists()             # main tree untouched


def test_hil_report_and_import(tmp_path):
    from chiplab import hil
    from chiplab.learning.loop import LearningLoop
    rep = hil.parse_fpga_report("Info: Max frequency for clock 'clk': 61.35 MHz (PASS at 12.00 MHz)\n"
                                "Info:     ICESTORM_LC:  1234/ 7680    16%\n")
    assert rep["fmax_mhz"] == 61.35 and rep["luts"] == 1234 and not rep["is_measurement"]
    loop = LearningLoop(tmp_path / "l.db")
    pid = loop.record_prediction("sram_nmc", "fmax_mhz", 55.0, "MHz")
    csv_text = ("prediction_id,value,unit,method,measured_on,repeats\n"
                f"{pid},58.0,MHz,freq sweep,2026-11-20,57.5;58.2;58.1\n"
                "pred_nope,1,MHz,x,2026-11-20,\n")
    res = hil.import_measurements(csv_text, loop, submitted_by="everett", board="iCEBreaker",
                                  attested_real=True)
    assert res["accepted"] == 1 and res["not_accepted"] == 1
    no = hil.import_measurements(csv_text, loop, submitted_by="e", board="b",
                                 attested_real=False)
    assert no["accepted"] == 0
