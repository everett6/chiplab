"""MCP prompt library: prompts register, name only real tools, and their tool chains run."""

from __future__ import annotations

import asyncio
import re

import pytest

from chiplab import mcp_server as srv
from chiplab.prompts import PROMPTS
from chiplab.experiments.store import ExperimentStore


@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "store", ExperimentStore(db_path=str(tmp_path / "p.db")))
    monkeypatch.setenv("CHIPLAB_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.chdir(tmp_path)  # handoff bundles are written under the working directory
    return srv.store


def _server():
    return srv.create_mcp_server()


def test_prompts_registered():
    names = {p.name for p in asyncio.run(_server().list_prompts())}
    assert {f.__name__ for f in PROMPTS} | {"explore_architecture"} <= names


@pytest.mark.parametrize("fn", PROMPTS, ids=lambda f: f.__name__)
def test_prompt_mentions_only_registered_tools(fn):
    tools = {t.name for t in asyncio.run(_server().list_tools())}
    text = fn()
    called = set(re.findall(r"\b([a-z][a-z0-9_]+)\(", text))
    assert called, text
    assert called <= tools, called - tools


def test_prompt_renders_through_mcp():
    res = asyncio.run(_server().get_prompt("moe_decode_chiplet_balance",
                                           {"model_params_b": "47"}))
    assert "chip_sweep_parameters" in str(res)


def test_riscv_prompt_arg_validation():
    with pytest.raises(ValueError):
        PROMPTS[1](node_set="bogus")


def test_moe_chain_runs(tmp_store):
    wl = {"type": "moe_llm_decode", "model_params_b": 47.0, "num_experts": 8,
          "active_experts": 2, "batch_size": 8}
    srv.chip_design_from_template(template="llm_accelerator_hbm3e", name="moe_balance",
                                  overrides={"constraints": {"max_cost_usd": 8000.0}})
    ev = srv.chip_evaluate(design_id_or_name="moe_balance", workload=wl)
    assert ev["valid"]
    sw = srv.chip_sweep_parameters(
        design_id_or_name="moe_balance",
        parameters={"chiplets[0].count": [1, 2], "chiplets[1].count": [4, 6, 8]},
        workload=wl, objective="throughput_tokens_per_sec")
    assert len(sw["points"]) == 6 and sw["best_point"]
    assert srv.chip_analyze_bottleneck(ev["run_id"])


def test_riscv_chain_runs(tmp_store):
    r = srv.run_digital_twin_search(
        node_set="tsmc", strategy="nsga2", population=64, generations=20, mc_trials=128,
        top_k=5, seed=1,
        config={"constraints": {"max_power_w": 5.0, "max_area_mm2": 60.0},
                "objectives": [{"name": "perf", "goal": "max"},
                               {"name": "power_w", "goal": "min"},
                               {"name": "area_mm2", "goal": "min"}]})
    assert r["feasible"] > 0 and r["top"]
    assert srv.get_digital_twin_run(run_id=r["run_id"], include_rows=5)["summary"]
    assert srv.export_pareto_handoff_bundle(run_id=r["run_id"], rank=1)


def test_hbm_chain_runs(tmp_store):
    wl = {"type": "llm_decode", "model_params_b": 70.0, "batch_size": 8}
    assert srv.chip_quickstart()
    srv.chip_design_from_template(template="llm_accelerator_hbm3e", name="hbm_cpt",
                                  overrides={"constraints": {"max_power_watts": 1200.0}})
    assert srv.chip_evaluate(design_id_or_name="hbm_cpt", workload=wl)["valid"]
    sw = srv.chip_sweep_parameters(
        design_id_or_name="hbm_cpt",
        parameters={"chiplets[1].count": [4, 6, 8], "workload.batch_size": [1, 8, 32]},
        workload=wl, objective="throughput_tokens_per_sec")
    assert len(sw["points"]) == 9
    assert srv.chip_estimate_cost(design_id_or_name="hbm_cpt")["unit_cost_usd"] > 0


def test_analog_pim_chain_runs(tmp_store):
    from chiplab.backends import EcoreBackend
    if not EcoreBackend().is_available():
        pytest.skip("Echo Lab eCore not available")
    r = srv.simulate_ecore_array(rows=256, cols=256, precision="int8", adc_bits=8,
                                 organization="org_b")
    assert r["accuracy"]["method"]
    spec = {"name": "pim_check", "chiplets": [
        {"id": "pim", "type": "PIM", "count": 1, "area_mm2": 100.0, "process_nm": 28,
         "compute": {"architecture": "processing_in_memory",
                     "ecore": {"rows": 256, "cols": 256, "precision": "int8",
                               "adc_bits": 8, "organization": "org_b"}}}]}
    ev = srv.chip_evaluate(spec=spec)
    assert ev["performance"]["peak_compute_tops"] > 0


def test_sweep_progress_and_cap(tmp_store):
    srv.chip_design_from_template(template="edge_npu", name="prog")
    seen = []
    srv.chip_sweep_parameters("prog", {"workload.batch_size": [1, 2, 4]},
                              workload={"type": "llm_decode", "model_params_b": 8},
                              progress=lambda d, t: seen.append((d, t)))
    assert seen == [(1, 3), (2, 3), (3, 3)]
    with pytest.raises(ValueError, match="run_digital_twin_search"):
        srv.chip_sweep_parameters("prog", {"workload.batch_size": list(range(1, 101)),
                                           "chiplets[0].count": list(range(1, 101))})


def test_sweep_tool_streams_progress_over_mcp(tmp_store):
    srv.chip_design_from_template(template="edge_npu", name="prog2")

    class Ctx:
        def __init__(self):
            self.calls = []

        async def report_progress(self, progress, total=None, message=None):
            self.calls.append((progress, total))

    ctx = Ctx()
    out = asyncio.run(srv._chip_sweep_parameters_tool(
        "prog2", {"workload.batch_size": [1, 2]},
        workload={"type": "llm_decode", "model_params_b": 8}, ctx=ctx))
    assert len(out["points"]) == 2 and ctx.calls[-1] == (2, 2)
