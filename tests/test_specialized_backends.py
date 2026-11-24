"""Level 3 backends: Ramulator 2 and SCALE-Sim adapters and the specialized engine.
Tool-dependent tests skip when the tool is not installed (e.g. on CI)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from chiplab import mcp_server as srv
from chiplab.backends import RamulatorBackend, ScaleSimBackend
from chiplab.backends.ramulator import standard_for
from chiplab.backends.scale_sim import array_geometry, is_systolic, transformer_gemms
from chiplab.experiments.store import ExperimentStore
from chiplab.schemas.design import DesignSpec
from chiplab.schemas.simulation import SimulationConfig
from chiplab.schemas.workload import WorkloadSpec

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ROOT / "src" / "chiplab" / "backends"
needs_ram = pytest.mark.skipif(not RamulatorBackend().is_available(),
                               reason="Ramulator 2 not installed")
needs_ss = pytest.mark.skipif(not ScaleSimBackend().is_available(),
                              reason="SCALE-Sim not installed")


# --- always run --------------------------------------------------------------------------

@pytest.mark.parametrize("fname", ["base.py", "ramulator.py", "scale_sim.py",
                                   "_ramulator_runner.py", "_scalesim_runner.py"])
def test_no_shell_eval_exec(fname):
    tree = ast.parse((BACKENDS / fname).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")
            assert name not in ("eval", "exec", "system", "popen", "Popen"), fname
            for kw in node.keywords:
                if kw.arg == "shell":
                    assert isinstance(kw.value, ast.Constant) and kw.value.value is False


def test_mappings():
    assert standard_for("HBM3E") == "HBM3" and standard_for("lpddr5x") == "LPDDR5"
    assert standard_for("GDDR6") is None and standard_for("DRAM") is None
    assert array_geometry(16384) == (128, 1)
    assert array_geometry(262144) == (128, 16)
    assert array_geometry(262144, 256) == (256, 4)
    assert is_systolic("tensor_engine") and is_systolic("systolic_array")
    assert not is_systolic("simd") and not is_systolic("cpu_vector")
    g = transformer_gemms(4096, 8, 128, True, 32)
    assert [x["M"] for x in g] == [8] * 4 and g[2]["N"] == 4 * 4096
    assert transformer_gemms(4096, 1, 2048, False, 32)[0]["M"] == 2048


def test_unsupported_memory_technology_rejected():
    with pytest.raises(ValueError, match="no supported preset"):
        RamulatorBackend().measure("GDDR6")


def test_specialized_falls_back_when_nothing_is_covered(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "store", ExperimentStore(db_path=str(tmp_path / "s.db")))
    c = srv.chip_create_design({"name": "simd_only", "chiplets": [
        {"type": "compute", "compute": {"architecture": "simd", "peak_tops": 10}}]})
    res = srv.chip_run_simulation(c["design_id"], fidelity="specialized")
    assert res["fidelity"] == "analytical"
    assert any("'specialized' is not available" in w for w in res["warnings"])


# --- Ramulator -------------------------------------------------------------------------------

@needs_ram
def test_ramulator_hbm3_stream():
    r = RamulatorBackend().measure("HBM3", "sequential", 0.0, n_requests=50_000)
    assert r["simulated_channel_peak_gbps"] == pytest.approx(51.2)  # 64 bit x 6.4 Gb/s
    assert 0.5 < r["efficiency"] <= 1.0
    assert 20 < r["avg_read_latency_ns"] < 500
    assert r["config"]["timing_preset"] == "HBM3_6400Mbps"


@needs_ram
def test_ramulator_random_slower_than_sequential_and_deterministic():
    b = RamulatorBackend()
    seq = b.measure("DDR5", "sequential", n_requests=20_000)
    rnd = b.measure("DDR5", "random", n_requests=20_000)
    assert rnd["efficiency"] < seq["efficiency"]
    assert rnd["row_hit_rate"] < seq["row_hit_rate"]
    from chiplab.backends.ramulator import _measure_cached
    _measure_cached.cache_clear()
    assert b.measure("DDR5", "random", n_requests=20_000) == rnd


@needs_ram
def test_ramulator_runner_rejects_bad_request():
    with pytest.raises(RuntimeError, match="failed"):
        RamulatorBackend().call_runner({"standard": "NOPE", "pattern": "sequential",
                                        "n_requests": 1000, "write_fraction": 0, "seed": 1})


# --- SCALE-Sim -------------------------------------------------------------------------------

@needs_ss
def test_scalesim_prefill_beats_decode_and_scales():
    b = ScaleSimBackend()
    dec = b.simulate_gemms([{"name": "d", "M": 8, "N": 512, "K": 512}], 16384, 8.0, "ws")
    pre = b.simulate_gemms([{"name": "p", "M": 512, "N": 512, "K": 512}], 16384, 8.0, "ws")
    assert 0 < dec["utilization"] < pre["utilization"] <= 1.0
    one = b.simulate_gemms([{"name": "a", "M": 8, "N": 512, "K": 512}], 16384, 8.0, "ws")
    two = b.simulate_gemms([{"name": "a", "M": 8, "N": 1024, "K": 512}], 16384, 8.0, "ws")
    assert two["cycles_per_array_total"] == pytest.approx(2 * one["cycles_per_array_total"],
                                                          rel=0.05)


@needs_ss
def test_scalesim_extrapolation_is_flagged():
    r = ScaleSimBackend().simulate_gemms([{"name": "big", "M": 8, "N": 8192, "K": 8192}],
                                         16384, 8.0, "ws")
    assert r["extrapolated"] and r["gemms"][0]["extrapolation_factor"] > 1


# --- specialized engine ----------------------------------------------------------------------

@needs_ram
@needs_ss
def test_specialized_hbm_accelerator(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "store", ExperimentStore(db_path=str(tmp_path / "s.db")))
    text = (ROOT / "examples" / "hbm_ai_accelerator.yaml").read_text()
    data = yaml.safe_load(text)["design"]
    data["chiplets"][0]["compute"]["dataflow"] = "os"      # one dataflow keeps the test fast
    c = srv.chip_create_design({"design": data})
    wl = data["metadata"]["workload_hint"]
    a = srv.chip_run_simulation(c["design_id"], workload=wl, fidelity="analytical")
    s = srv.chip_run_simulation(c["design_id"], workload=wl, fidelity="specialized")
    assert s["fidelity"] == "specialized" and "Ramulator" in s["fidelity_disclaimer"]
    am, sm = a["metrics"], s["metrics"]
    assert sm["aggregate_memory_bandwidth_tbps"] < am["aggregate_memory_bandwidth_tbps"]
    assert sm["effective_compute_tops"] < am["effective_compute_tops"]
    assert sm["throughput_tokens_per_sec"] <= am["throughput_tokens_per_sec"]
    assert sm["total_power_watts"] == am["total_power_watts"]   # power stays analytical
    srcs = {x["source"] for x in s["assumptions"] if x["source"]}
    assert any(x.startswith("ramulator:HBM3") for x in srcs)
    assert any(x.startswith("scale_sim:") for x in srcs)
    raw = srv.store.get_simulation(s["run_id"]).raw_details["specialized"]
    assert raw["memory"][0]["chiplet"] == "hbm" and raw["compute"][0]["chiplet"] == "tensor"
    assert raw["compute"][0]["dataflow_choice"] == "specified"


@needs_ram
def test_specialized_backend_run_and_sweep():
    from chiplab.engine.sweep import run_parameter_sweep
    spec = {"name": "lp", "chiplets": [
        {"id": "c", "type": "compute", "count": 1, "area_mm2": 30, "tdp_watts": 2,
         "process_nm": 7, "compute": {"architecture": "simd", "peak_tops": 4}},
        {"id": "d", "type": "LPDDR5", "count": 1, "memory": {
            "technology": "LPDDR5", "capacity_per_chiplet_gb": 4, "channels": 2,
            "bandwidth_per_channel_gbps": 12.8}}]}
    d = DesignSpec.model_validate(spec)
    wl = WorkloadSpec.model_validate({"type": "llm_decode", "model_params_b": 1})
    cfg = SimulationConfig(fidelity="specialized")
    r = RamulatorBackend().run(d, wl, cfg)
    assert r.fidelity.value == "specialized"
    assert r.metrics.aggregate_memory_bandwidth_tbps < 0.0256
    sw = run_parameter_sweep(d, wl, cfg, {"chiplets[1].memory.channels": [2, 4]})
    tps = [p.metrics.throughput_tokens_per_sec for p in sw.points]
    assert tps[1] > tps[0]
