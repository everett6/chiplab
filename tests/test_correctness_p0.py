"""Regression tests for the correctness review: packaging yield, capacity aggregation,
honest external backends and fidelity labels, MoE operator totals, ECHO configs, and a
discrete-event engine that really simulates queues and contention. Synthetic data only."""

from __future__ import annotations

import math

import pytest

import chiplab.mcp_server
from chiplab.backends import AccelSimBackend, RamulatorBackend, ScaleSimBackend, backend_status
from chiplab.engine.analytical import run_analytical_simulation
from chiplab.engine.discrete_event import run_discrete_event_simulation
from chiplab.engine.validator import validate_design
from chiplab.models.memory import estimate_memory_chiplet
from chiplab.models.packaging import estimate_packaging
from chiplab.schemas.design import (
    ChipletSpec, ComputeSpec, DesignSpec, InterconnectSpec, MemorySpec, PackageSpec)
from chiplab.schemas.simulation import FidelityLevel, SimulationConfig
from chiplab.schemas.workload import WorkloadSpec, expand_workload_operators

ROOT = "/home/everett/Chip Lab"
ECHO = [f"{ROOT}/projects/echo/echo_v0_1.yaml", f"{ROOT}/projects/echo/echo_v0_2.yaml"]
DE = SimulationConfig(fidelity=FidelityLevel.DISCRETE_EVENT, num_requests=1)


def _load(path: str) -> DesignSpec:
    with open(path) as f:
        return DesignSpec.from_yaml(f.read())


# --- packaging yield -------------------------------------------------------------------------

def test_packaging_yield_is_poisson_zero_defect_probability():
    d = DesignSpec(name="y", package=PackageSpec(type="monolithic",
                                                 yield_defect_density_per_cm2=0.1),
                   chiplets=[ChipletSpec(type="compute", area_mm2=400.0)])
    res, _, _ = estimate_packaging(d, [400.0])
    assert res["yield_pct"] == pytest.approx(math.exp(-0.1 * 4.0) * 100.0)   # 67.0 %, not 33 %
    worse = d.model_copy(deep=True)
    worse.package.yield_defect_density_per_cm2 = 0.5
    assert estimate_packaging(worse, [400.0])[0]["yield_pct"] < res["yield_pct"]


# --- capacity aggregation --------------------------------------------------------------------

@pytest.mark.parametrize("memory", [None, MemorySpec(technology="HBM", capacity_per_chiplet_gb=24.0,
                                                     channels=8, bandwidth_per_channel_gbps=100.0)])
def test_memory_capacity_and_channels_scale_with_count(memory):
    one, _, _ = estimate_memory_chiplet(ChipletSpec(type="HBM", count=1, memory=memory))
    four, _, _ = estimate_memory_chiplet(ChipletSpec(type="HBM", count=4, memory=memory))
    assert one["capacity_gb"] > 0
    assert four["capacity_gb"] == pytest.approx(4 * one["capacity_gb"])
    assert four["bandwidth_tbps"] == pytest.approx(4 * one["bandwidth_tbps"])
    assert four["banks_per_channel"] == one["banks_per_channel"]      # per channel: not scaled


def test_des_capacity_and_bisection_match_analytical_models():
    d = _load(ECHO[0])                       # 4 DRAM chiplets x 8 GB
    wl = WorkloadSpec(type="llm_decode", model_params_b=8)
    de = run_discrete_event_simulation(d, wl, DE).metrics
    an = run_analytical_simulation(d, wl, SimulationConfig()).metrics
    assert de.total_memory_capacity_gb == pytest.approx(32.0)
    assert de.interconnect_bisection_bw_tbps == pytest.approx(an.interconnect_bisection_bw_tbps)
    assert de.interconnect_bisection_bw_tbps != pytest.approx(de.aggregate_memory_bandwidth_tbps * 0.1)


# --- external backends never fake success ----------------------------------------------------

@pytest.mark.parametrize("cls", [RamulatorBackend, ScaleSimBackend, AccelSimBackend])
def test_external_backend_raises_even_when_installed(cls, monkeypatch):
    b = cls()
    with pytest.raises(NotImplementedError, match="not installed"):
        b.run(None, None, None)
    monkeypatch.setattr("chiplab.backends.base.shutil.which", lambda name: f"/usr/bin/{name}")
    assert b.is_available()
    with pytest.raises(NotImplementedError, match="no adapter"):
        b.run(None, None, None)
    assert b.status() == {"name": b.name, "installed": True, "integrated": False,
                          "usable": False}


def test_backend_status_lists_all_three_unusable():
    st = {s["name"]: s for s in backend_status()}
    assert set(st) == {"ramulator", "scale_sim", "accel_sim", "ecore"}
    assert not any(st[n]["usable"] for n in ("ramulator", "scale_sim", "accel_sim"))
    assert st["ecore"]["integrated"] is True  # usable depends on Echo Lab being present


@pytest.mark.parametrize("requested", ["specialized", "rtl_circuit"])
def test_unavailable_fidelity_is_labeled_analytical(requested, tmp_path, monkeypatch):
    from chiplab.experiments.store import ExperimentStore
    monkeypatch.setattr(chiplab.mcp_server, "store", ExperimentStore(db_path=str(tmp_path / "x.db")))
    created = chiplab.mcp_server.chip_create_design(
        {"name": "fid", "chiplets": [{"type": "compute", "compute": {"architecture": "simd",
                                                                    "peak_tops": 10}}]})
    res = chiplab.mcp_server.chip_run_simulation(created["design_id"], fidelity=requested)
    assert res["fidelity"] == "analytical"
    assert any(f"'{requested}' is not available" in w for w in res["warnings"])
    info = chiplab.mcp_server.get_models_assumptions()
    assert "NOT INTEGRATED" in info["fidelity_levels"]["specialized"]["description"]
    assert "(SimPy)" not in info["fidelity_levels"]["discrete_event"]["description"]


# --- workload operators keep the declared totals ---------------------------------------------

def test_moe_expert_operator_keeps_total_flops_and_bytes():
    wl = WorkloadSpec(type="moe_decode", active_experts=2, flops=1e12, bytes_read=4e9,
                      bytes_write=1e9)
    ops = {o.name: o for o in expand_workload_operators(wl)}
    e = ops["moe_expert_compute"]
    assert (e.flops, e.bytes_read, e.bytes_write) == (1e12, 4e9, 1e9)


# --- ECHO configs ----------------------------------------------------------------------------

@pytest.mark.parametrize("path", ECHO)
def test_echo_configs_validate_and_simulate(path):
    d = _load(path)
    assert validate_design(d, True).valid
    assert validate_design(d, False).valid
    r = run_discrete_event_simulation(d, WorkloadSpec(type="llm_decode", model_params_b=8), DE)
    assert r.success and r.metrics.estimated_latency_ms > 0


# --- the discrete-event engine is event-driven -----------------------------------------------

def _two_node(topology: str, link_gbps: float = 50.0, ch_gbps: float = 100.0) -> DesignSpec:
    """Memory on one chiplet, compute on another: every byte crosses the interconnect."""
    return DesignSpec(
        name=f"split_{topology}", package=PackageSpec(type="2.5D"),
        interconnect=InterconnectSpec(topology=topology, link_bandwidth_gbps=link_gbps,
                                      latency_ns=5.0),
        chiplets=[
            ChipletSpec(id="mem", type="HBM", memory=MemorySpec(
                technology="HBM", capacity_per_chiplet_gb=16, channels=4,
                bandwidth_per_channel_gbps=ch_gbps, latency_ns=40.0)),
            ChipletSpec(id="cpu", type="compute", compute=ComputeSpec(
                architecture="simd", peak_tops=1000.0, efficiency_factor=1.0)),
        ])


def _memory_bound() -> WorkloadSpec:
    return WorkloadSpec(type="llm_decode", model_params_b=1, batch_size=1)


def test_des_detects_interconnect_bottleneck_and_scales_with_link_bandwidth():
    slow = run_discrete_event_simulation(_two_node("crossbar", 50.0), _memory_bound(), DE)
    fast = run_discrete_event_simulation(_two_node("crossbar", 200.0), _memory_bound(), DE)
    assert slow.bottleneck.bound_type == "interconnect"
    assert slow.bottleneck.utilization_breakdown["interconnect_link_max_pct"] > 90
    # 4x the link bandwidth (200 GB/s, still under the 4 x 100 GB/s of channels): ~4x faster
    ratio = slow.metrics.estimated_latency_ms / fast.metrics.estimated_latency_ms
    assert 3.5 < ratio < 4.5
    channel_bound = run_discrete_event_simulation(_two_node("crossbar", 1600.0), _memory_bound(), DE)
    assert channel_bound.bottleneck.bound_type == "memory_bandwidth"


def test_des_memory_bound_decode_agrees_with_roofline():
    """Validation: with memory next to compute, a weight-streaming decode is set by channel
    bandwidth, so the event simulation must land on the analytical roofline."""
    d = _load(f"{ROOT}/examples/transformer_accelerator.yaml")
    wl = WorkloadSpec(type="llm_decode", model_params_b=70, batch_size=8)
    de = run_discrete_event_simulation(d, wl, DE)
    an = run_analytical_simulation(d, wl, SimulationConfig())
    assert de.bottleneck.bound_type == "memory_bandwidth"
    assert de.metrics.estimated_latency_ms == pytest.approx(an.metrics.estimated_latency_ms,
                                                            rel=0.02)


def test_des_queueing_grows_with_load_but_throughput_saturates():
    d = _load(f"{ROOT}/examples/transformer_accelerator.yaml")
    wl = WorkloadSpec(type="llm_decode", model_params_b=70, batch_size=8)
    r1 = run_discrete_event_simulation(d, wl, DE)
    r16 = run_discrete_event_simulation(d, wl, DE.model_copy(update={"num_requests": 16}))
    assert r16.raw_details["concurrency"] == 4
    assert r16.metrics.estimated_latency_ms > 3.5 * r1.metrics.estimated_latency_ms   # queues
    assert r16.bottleneck.utilization_breakdown["memory_channel_wait_us"] > \
        r1.bottleneck.utilization_breakdown["memory_channel_wait_us"]
    assert r16.metrics.throughput_tokens_per_sec == pytest.approx(
        r1.metrics.throughput_tokens_per_sec, rel=0.05)                      # saturated


def test_des_topology_matters_bus_is_slower_than_crossbar():
    def lat(topo: str) -> float:
        d = _two_node(topo, 100.0)
        d.chiplets.append(ChipletSpec(id="cpu2", type="compute", compute=ComputeSpec(
            architecture="simd", peak_tops=1000.0, efficiency_factor=1.0)))
        d.chiplets.append(ChipletSpec(id="mem2", type="HBM", memory=MemorySpec(
            technology="HBM", capacity_per_chiplet_gb=16, channels=4,
            bandwidth_per_channel_gbps=100.0, latency_ns=40.0)))
        return run_discrete_event_simulation(d, _memory_bound(), DE).metrics.estimated_latency_ms

    assert lat("bus") > 1.5 * lat("crossbar")


def test_des_is_deterministic_and_reports_events():
    d = _load(ECHO[1])
    wl = WorkloadSpec(type="llm_decode", model_params_b=8, batch_size=4)
    cfg = DE.model_copy(update={"num_requests": 8, "seed": 7})
    a, b = (run_discrete_event_simulation(d, wl, cfg) for _ in range(2))
    assert a.metrics == b.metrics and a.raw_details == b.raw_details
    assert a.raw_details["engine"] == "tile_level_discrete_event"
    assert a.raw_details["events"] > 1000
    assert "discrete-event" in a.fidelity_disclaimer and "not cycle-accurate" in a.fidelity_disclaimer


def test_des_fails_honestly_without_compute():
    d = DesignSpec(name="mem_only", package=PackageSpec(type="chiplets"), chiplets=[
        ChipletSpec(type="HBM", memory=MemorySpec(technology="HBM", capacity_per_chiplet_gb=8,
                                                  channels=2, bandwidth_per_channel_gbps=50))])
    r = run_discrete_event_simulation(d, WorkloadSpec(type="llm_decode", model_params_b=1), DE)
    assert not r.success and "no chiplet has compute" in r.error_message
