"""MCP research tools: each tool returns ok and respects its guard rails."""

from __future__ import annotations

import pytest

from chiplab import mcp_research_tools as t
from chiplab.pim import _eda
from chiplab.plugins import technology as tech


@pytest.fixture(autouse=True)
def fresh(monkeypatch, tmp_path):
    t._brain_at.cache_clear()
    monkeypatch.setattr("chiplab.sandbox.SANDBOX", tmp_path / "sb")
    tech.register(tech.example_resistive_plugin(), replace=True)
    yield
    t._brain_at.cache_clear()


def test_brain_tools():
    assert t.brain_remember("science", "x", "y")["ok"] is False      # needs a source
    assert t.brain_remember("project", "DRAM needs custom fab", "no shuttle",
                            status="limitation", tags=["dram"])["ok"]
    assert t.brain_decide("Use SRAM first", "digital shuttle", tags=["sram"])["ok"]
    r = t.brain_recall("dram fab")
    assert r["ok"] and r["limitations"]
    assert t.brain_stats()["total"] >= 2


def test_research_investigate_records_and_stays_honest():
    r = t.research_investigate("Does near-bank DRAM cut matmul energy?", "dram",
                               {"kind": "matmul", "m": 64, "n": 64, "k": 64, "dtype": "int8"})
    assert r["ok"], r
    assert r["implementation"] == "dram_near_bank"
    assert r["judgement"]["outcome"] == "inconclusive"               # analytical < spice
    assert "spice" in r["next_experiment"].lower()
    assert r["ledger"]["run_id"] and r["memory_id"]
    assert t.science_review_run(r["ledger"]["run_id"])["ok"]


def test_workload_and_falsify_tools():
    w = t.workload_compile({"kind": "bitwise", "n_bytes": 1 << 20, "op": "and"})
    assert w["ok"] and w["ranking"]
    p = t.falsification_plan("SRAM NMC halves energy", "sram", "energy_j", min_effect=0.5)
    j = t.falsification_judge(p, 0.4, 1.0, "rtl_sim")
    assert j["outcome"] == "inconclusive"


def test_genealogy_tool():
    assert t.architecture_genealogy("create", name="root", spec={"a": 1})["ok"]
    d = t.architecture_genealogy("derive", parent="root", name="kid", spec={"a": 2},
                                 why="test", improved={"energy": "-10%"}, evidence=["r1"])
    assert d["ok"] and not d["warnings"]
    assert "kid" in t.architecture_genealogy("tree")["text"]
    assert t.architecture_genealogy("bogus")["ok"] is False


def test_dram_and_dft_and_plugin_tools():
    assert t.pim_dram("margins")["ok"]
    mc = t.pim_dram("monte_carlo", op="and", trials=500)
    assert mc["ok"] and "ber" in str(mc).lower()
    assert t.pim_dram("compare", workload="bitwise_and", n_bytes=1 << 20)["ok"]
    assert t.pim_dram("nope")["ok"] is False
    assert t.dft_analyze("coverage", words=16)["ok"]
    assert t.dft_analyze("yield", die_area_mm2=50)["ok"]
    assert t.pim_sram("predict", {"rows": 8, "lanes": 4, "lane_bits": 8})["ok"]
    assert t.pim_sram("compare", {"rows": 8, "lanes": 4, "lane_bits": 8})["ok"]
    assert t.pim_sram("predict", {"rows": 999, "lanes": 4, "lane_bits": 8})["ok"] is False
    d = t.technology_plugins("describe")
    assert any(p["key"].startswith("example_resistive") for p in d["plugins"])
    assert t.technology_plugins("simulate", "example_resistive", {"state": 0})["ok"]
    assert t.sandbox_proposals() == {"ok": True, "proposals": []}


@pytest.mark.skipif(not _eda.have("iverilog", "vvp"), reason="iverilog missing")
def test_sram_simulate_tool():
    r = t.pim_sram("simulate", {"rows": 4, "lanes": 2, "lane_bits": 4}, n_ops=10)
    assert r["ok"] and r.get("mismatches", 0) == 0


def test_no_code_execution_or_approval_exposed():
    names = {f.__name__ for f in t.READ_ONLY + t.WRITES}
    assert not any(k in n for n in names for k in ("approve", "apply", "exec", "eval",
                                                  "shell", "propose"))
