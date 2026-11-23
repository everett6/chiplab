"""Tests for the cost model, price overrides, templates, LLM workloads and easy MCP tools."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

import chiplab.mcp_server as srv
from chiplab import templates
from chiplab.cost import model, prices
from chiplab.cost.reference import memory_tech_for, node_for_nm, package_key
from chiplab.experiments.store import ExperimentStore
from chiplab.schemas.design import DesignSpec
from chiplab.schemas.workload import WorkloadSpec, expand_workload_operators, llm_param_counts


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Temp price file and temp experiment DB for every test."""
    monkeypatch.setenv("CHIPLAB_PRICE_FILE", str(tmp_path / "prices.json"))
    monkeypatch.delenv("CHIPLAB_PRICE_FEED_URL", raising=False)
    fd, db = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setattr(srv, "store", ExperimentStore(db_path=db))
    yield
    os.unlink(db)


def _design(**over) -> DesignSpec:
    return DesignSpec.model_validate(templates.build_from_template("llm_accelerator_hbm3e", over))


# --- reference lookups ---------------------------------------------------------------------

def test_node_memory_and_package_mapping():
    assert node_for_nm(3) == "N3"
    assert node_for_nm(5) == "N5"
    assert node_for_nm(14) == "N12" or node_for_nm(14) == "N16"
    assert node_for_nm("N3E") == "N3"
    assert node_for_nm("A16") == "A16"
    assert memory_tech_for("hbm3e") == "HBM3E"
    assert memory_tech_for("HBM3") == "HBM3"
    assert memory_tech_for("LPDDR5X") == "LPDDR5X"
    assert memory_tech_for("digital_compute") is None
    assert package_key("2.5D", "silicon") == "cowos_s"
    assert package_key("2D", "silicon") == "fcbga"  # 2D ignores the defaulted interposer
    assert package_key("3D", "silicon") == "soic"
    assert package_key("2.5D", "cowos_l") == "cowos_l"


# --- physics of cost -----------------------------------------------------------------------

def test_dies_per_wafer_and_yield():
    # Reticle-class square die on 300 mm: exact placement gives 68 (the old closed-form
    # approximation gave 60); tests/test_validation_golden.py checks it against brute force.
    assert 60 <= model.dies_per_wafer(814) <= 70
    assert model.dies_per_wafer(100) > 8 * model.dies_per_wafer(814)
    assert model.die_yield(1, 0.1) > 0.99
    assert model.die_yield(800, 0.1) < model.die_yield(100, 0.1)
    with pytest.raises(ValueError):
        model.dies_per_wafer(0)


def test_sram_area_shrinks_with_node():
    assert model.sram_area_mm2(64, "N3") < model.sram_area_mm2(64, "N7")
    assert 20 < model.sram_area_mm2(64, "N7") < 50  # 64 MB V-Cache class


def test_scenarios_are_ordered_and_breakdown_sums():
    d = _design()
    lo, typ, hi = (model.estimate_unit_cost(d, 100_000, s) for s in ("low", "value", "high"))
    assert lo["unit_cost_usd"] < typ["unit_cost_usd"] < hi["unit_cost_usd"]
    assert sum(typ["breakdown_usd"].values()) == pytest.approx(typ["unit_cost_usd"], abs=0.1)
    # 8 x 24 GB HBM3E at the built-in $/GB
    assert typ["breakdown_usd"]["memory"] == pytest.approx(192 * prices.value("memory.HBM3E"))


def test_nre_amortizes_with_volume():
    d = _design()
    small = model.estimate_unit_cost(d, 10_000)["unit_cost_usd"]
    big = model.estimate_unit_cost(d, 10_000_000)["unit_cost_usd"]
    assert small - big > 100


def test_realism_warnings():
    huge = _design(chiplets=[{"id": "compute", "area_mm2": 1000.0}])
    assert any("reticle" in w for w in model.estimate_unit_cost(huge)["warnings"])
    hbm_on_bga = _design(package={"type": "2D", "interposer": "organic"})
    assert any("interposer" in w for w in model.estimate_unit_cost(hbm_on_bga)["warnings"])
    sram_overflow = _design(chiplets=[{"id": "compute", "area_mm2": 50.0,
                                        "compute": {"sram_mb": 512}}])
    assert any("SRAM" in w for w in model.estimate_unit_cost(sram_overflow)["warnings"])


def test_board_cost_added():
    d = _design()
    no_board = model.estimate_unit_cost(d)["unit_cost_usd"]
    with_board = model.estimate_unit_cost(d, board={"area_cm2": 400, "layers": 16})
    assert with_board["breakdown_usd"]["board"] > 0
    assert with_board["unit_cost_usd"] > no_board


# --- prices / live overrides ---------------------------------------------------------------

def test_override_wins_and_is_traceable():
    builtin = prices.value("memory.HBM3E")
    entry = prices.set_override("memory.HBM3E", builtin * 2, "https://example.com/hbm",
                                "2026-11-01")
    assert entry["origin"] == "override" and entry["value"] == builtin * 2
    assert entry["low"] < entry["value"] < entry["high"]
    assert json.loads(prices.price_file().read_text())["memory.HBM3E"]["source"]
    assert prices.clear_override("memory.HBM3E")
    assert prices.value("memory.HBM3E") == builtin


def test_override_validation():
    with pytest.raises(ValueError):
        prices.set_override("memory.HBM3E", 20, "")
    with pytest.raises(ValueError):
        prices.set_override("memory.HBM3E", -1, "src")
    with pytest.raises(ValueError):
        prices.set_override("made.up.key", 1, "src")
    with pytest.raises(KeyError):
        prices.get("made.up.key")
    assert prices.set_override("custom.item", 3, "src", unit="USD")["value"] == 3


def test_stale_flags_and_lookup_plan():
    assert prices.get("wafer.tsmc.N3")["stale"]  # built-ins are dated 2025-06
    plan = prices.lookup_plan("wafer.")
    assert any(p["key"] == "wafer.tsmc.N3" and "N3" in p["search_query"] for p in plan)
    prices.set_override("wafer.tsmc.N3", 21_000, "https://example.com/n3")
    assert not prices.get("wafer.tsmc.N3")["stale"]
    assert all(p["key"] != "wafer.tsmc.N3" for p in prices.lookup_plan("wafer."))


def test_feed_requires_config_and_https():
    assert prices.refresh_from_feed()["error"]
    assert "https" in prices.refresh_from_feed("http://insecure.example")["error"]


# --- templates -----------------------------------------------------------------------------

def test_template_merge_edit_add_remove():
    spec = templates.build_from_template("llm_accelerator_hbm3e", {
        "chiplets": [{"id": "compute", "process_nm": 2}, {"id": "hbm", "remove": True},
                     {"id": "io", "type": "io", "count": 1, "area_mm2": 100}]})
    ids = {c["id"]: c for c in spec["chiplets"]}
    assert ids["compute"]["process_nm"] == 2 and ids["compute"]["area_mm2"] == 780.0
    assert "hbm" not in ids and "io" in ids
    assert templates.TEMPLATES["llm_accelerator_hbm3e"]["spec"]["chiplets"][0]["process_nm"] == 3
    with pytest.raises(KeyError):
        templates.build_from_template("nope")


# --- LLM workload --------------------------------------------------------------------------

def test_llm_params_and_decode_traffic():
    w = WorkloadSpec(type="llm_decode", model_params_b=70, hidden_dim=8192, num_layers=80,
                     precision="FP8", batch_size=1, sequence_length=1)
    total, active = llm_param_counts(w)
    assert total == pytest.approx(70e9) and active == pytest.approx(70e9)
    op = expand_workload_operators(w)[0]
    assert op.bytes_read == pytest.approx(70e9, rel=0.01)  # weights dominate decode
    moe = WorkloadSpec(type="moe_llm_decode", num_experts=8, active_experts=2)
    t, a = llm_param_counts(moe)
    assert a < t


# --- MCP tools end to end ------------------------------------------------------------------

LLM_70B = {"name": "70b", "type": "llm_decode", "model_params_b": 70, "hidden_dim": 8192,
           "num_layers": 80, "precision": "FP8", "batch_size": 8, "sequence_length": 4096,
           "kv_cache_fraction": 0.125}


def test_evaluate_template_end_to_end():
    created = srv.chip_design_from_template("llm_accelerator_hbm3e", name="t1")
    assert created["validation"]["valid"]
    r = srv.chip_evaluate("t1", workload=LLM_70B)
    # 70 GB weights + ~11 GB KV over 9.6 TB/s HBM3E -> ~8 ms per step -> ~1000 tok/s.
    assert 600 < r["performance"]["throughput_tokens_per_sec"] < 1500
    assert r["power_thermal"]["total_power_watts"] == pytest.approx(2 * 450 + 8 * 15)
    assert r["cost"]["range_usd"]["optimistic"] < r["cost"]["unit_cost_usd"]
    assert r["efficiency"]["usd_per_peak_tops"] > 0
    assert r["price_data"]["stale_keys"]  # built-in prices are old -> lookup advised


def test_evaluate_spec_without_workload_omits_throughput():
    spec = templates.build_from_template("chiplet_cpu")
    r = srv.chip_evaluate(spec=spec)
    assert r["performance"]["throughput_tokens_per_sec"] is None
    assert r["cost"]["volume"] == 1_000_000


def test_price_tools_and_search_fetch():
    assert srv.chip_cost_references("wafer.")["count"] >= 10
    srv.chip_set_price("memory.HBM3E", 25, "https://example.com/x", "2026-11-01")
    assert srv.chip_cost_references("memory.HBM3E")["entries"]["memory.HBM3E"]["value"] == 25
    assert srv.chip_clear_price("memory.HBM3E")["removed"]
    srv.chip_design_from_template("edge_npu")
    hits = srv.search("edge")["results"]
    assert any(h["id"].startswith("design:") for h in hits)
    doc = srv.fetch(hits[0]["id"])
    assert doc["text"] and doc["title"]
    assert srv.fetch("template:edge_npu")["text"]


def test_quickstart_and_server_build():
    q = srv.chip_quickstart()
    assert "N3" in q["tsmc_nodes"] and "HBM3E" in q["memory_technologies"]
    assert set(q["templates"]) == set(templates.TEMPLATES)
    assert srv.create_mcp_server() is not None
