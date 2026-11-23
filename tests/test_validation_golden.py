"""Golden-reference validation: ChipLab's models against published data, standards and
independent first-principles computations. Each case states its source and tolerance;
docs/validation.md lists them. These are reference checks, not silicon correlation."""

from __future__ import annotations

import math

import pytest

from chiplab.cost import model, prices
from chiplab.engine import run_analytical_simulation
from chiplab.schemas.design import DesignSpec
from chiplab.schemas.simulation import SimulationConfig
from chiplab.schemas.workload import WorkloadSpec


# 1. SRAM high-density 6T bitcell areas (um2), as published by TSMC at IEDM.
PUBLISHED_BITCELL = {
    "N28": (0.127, "TSMC 28 nm HD bitcell"),
    "N16": (0.070, "TSMC 16FF, IEDM 2013"),
    "N7": (0.027, "TSMC 7 nm, IEDM 2016"),
    "N5": (0.021, "TSMC 5 nm, IEDM 2019"),
    "N3": (0.0199, "TSMC N3, IEDM 2022"),
}


@pytest.mark.parametrize("node", sorted(PUBLISHED_BITCELL))
def test_sram_bitcell_matches_published(node):
    published, _src = PUBLISHED_BITCELL[node]
    assert prices.value(f"sram_bitcell.tsmc.{node}") == pytest.approx(published, rel=0.10)


# 2. Dies per wafer vs an independent brute-force placement count.
def _brute_force_dpw(area: float, steps: int = 20) -> int:
    s = math.sqrt(area) + model.SCRIBE_MM
    r = model.WAFER_DIAMETER_MM / 2 - model.EDGE_EXCLUSION_MM
    best = 0
    for ox in range(steps):
        for oy in range(steps):
            n, i = 0, 0
            while -r + ox * s / steps + (i + 1) * s <= r:
                j = 0
                while -r + oy * s / steps + (j + 1) * s <= r:
                    x0 = -r + ox * s / steps + i * s
                    y0 = -r + oy * s / steps + j * s
                    if max(x0 * x0, (x0 + s) ** 2) + max(y0 * y0, (y0 + s) ** 2) <= r * r:
                        n += 1
                    j += 1
                i += 1
            best = max(best, n)
    return best


@pytest.mark.parametrize("area", [25.0, 100.0, 400.0, 814.0])
def test_dies_per_wafer_matches_exact_placement(area):
    assert model.dies_per_wafer(area) == pytest.approx(_brute_force_dpw(area), rel=0.03)


# 3. Negative-binomial yield converges to Poisson exp(-A*D0) as alpha -> infinity.
@pytest.mark.parametrize("area,d0", [(100.0, 0.07), (400.0, 0.1), (800.0, 0.09)])
def test_yield_poisson_limit(area, d0):
    assert model.die_yield(area, d0, alpha=1e7) == pytest.approx(math.exp(-area / 100 * d0),
                                                                 rel=1e-5)
    # Clustering (alpha=3) always yields at least the Poisson value.
    assert model.die_yield(area, d0) >= math.exp(-area / 100 * d0)


def _sim(spec: dict, workload: dict | None = None):
    wl = WorkloadSpec.model_validate(workload) if workload else None
    return run_analytical_simulation(DesignSpec.model_validate(spec), wl, SimulationConfig())


# 4. JEDEC memory bandwidth: HBM3 (JESD238) 1024 bits x 6.4 Gb/s = 819.2 GB/s per stack;
#    LPDDR5-6400 x16 channel = 12.8 GB/s.
def test_hbm3_stack_bandwidth_jedec():
    spec = {"name": "hbm3", "chiplets": [
        {"id": "c", "type": "compute", "count": 1, "area_mm2": 600, "tdp_watts": 300,
         "process_nm": 5, "compute": {"architecture": "x", "peak_tops": 1000}},
        {"id": "hbm", "type": "HBM3", "count": 6, "memory": {
            "technology": "HBM3", "capacity_per_chiplet_gb": 16, "channels": 16,
            "bandwidth_per_channel_gbps": 1024 / 16 * 6.4 / 8}}]}
    assert _sim(spec).metrics.aggregate_memory_bandwidth_tbps == pytest.approx(6 * 0.8192,
                                                                                rel=0.01)


def test_lpddr5_channel_bandwidth_jedec():
    spec = {"name": "lp5", "chiplets": [
        {"id": "c", "type": "compute", "count": 1, "area_mm2": 30, "tdp_watts": 2,
         "process_nm": 7, "compute": {"architecture": "x", "peak_tops": 4}},
        {"id": "d", "type": "LPDDR5", "count": 1, "memory": {
            "technology": "LPDDR5", "capacity_per_chiplet_gb": 4, "channels": 2,
            "bandwidth_per_channel_gbps": 6400e6 * 16 / 8 / 1e9}}]}
    assert _sim(spec).metrics.aggregate_memory_bandwidth_tbps == pytest.approx(0.0256, rel=0.01)


# 5. Bandwidth-bound decode (H100 SXM5 published 3.35 TB/s, 70B params in FP8 = 70 GB of
#    weights per token at batch 1): tokens/s must sit just under 3350/70 = 47.9.
H100_LIKE = {
    "name": "h100_like", "package": {"type": "2.5D", "interposer": "cowos_s"},
    "chiplets": [
        {"id": "gpu", "type": "compute", "count": 1, "area_mm2": 814, "tdp_watts": 700,
         "process_nm": 4, "compute": {"architecture": "tensor_engine", "precision": "FP8",
                                      "peak_tops": 1979, "efficiency_factor": 1.0}},
        {"id": "hbm", "type": "HBM3", "count": 5, "memory": {
            "technology": "HBM3", "capacity_per_chiplet_gb": 16, "channels": 16,
            "bandwidth_per_channel_gbps": 3350 / 5 / 16}}],
    "constraints": {"thermal_resistance_c_per_w": 0.05, "ambient_temp_c": 25.0},
}


@pytest.mark.parametrize("seq", [128, 2048])
def test_decode_matches_bandwidth_bound(seq):
    m = _sim(H100_LIKE, {"type": "llm_decode", "model_params_b": 70, "batch_size": 1,
                         "sequence_length": seq, "precision": "FP8"}).metrics
    bound = 3350 / 70
    assert m.throughput_tokens_per_sec <= bound * 1.001
    assert m.throughput_tokens_per_sec == pytest.approx(bound, rel=0.02)


# 6. Lumped thermal model identity: Tj = Tambient + P * theta_ja.
def test_thermal_identity():
    m = _sim(H100_LIKE).metrics
    assert m.estimated_temp_c == pytest.approx(25.0 + m.total_power_watts * 0.05, rel=1e-6)


# 7. Uncertainty bands bracket the nominal and are reproducible.
def test_uncertainty_bands_bracket_nominal_and_are_seeded():
    from chiplab.engine.uncertainty import propagate_uncertainty
    d = DesignSpec.model_validate(H100_LIKE)
    wl = WorkloadSpec.model_validate({"type": "llm_decode", "model_params_b": 70,
                                      "batch_size": 1, "precision": "FP8"})
    a = propagate_uncertainty(d, wl, samples=100, seed=3)
    b = propagate_uncertainty(d, wl, samples=100, seed=3)
    assert a == b
    for k, band in a["bands"].items():
        assert band["p5"] <= band["p50"] <= band["p95"], k
        assert band["p5"] <= band["nominal"] * 1.02 and band["p95"] >= band["nominal"] * 0.98, k
    sens = a["sensitivity_top3"]["throughput_tokens_per_sec"]
    assert sens and sens[0]["parameter"] == "chiplets[1].memory.bandwidth_per_channel_gbps"


def test_tracked_uncertainty_pct_is_used():
    from chiplab.engine.uncertainty import propagate_uncertainty
    spec = {**H100_LIKE, "chiplets": [dict(H100_LIKE["chiplets"][0]),
                                      H100_LIKE["chiplets"][1]]}
    spec["chiplets"][0] = {**spec["chiplets"][0], "compute": {
        **spec["chiplets"][0]["compute"],
        "peak_tops": {"value": 1979, "provenance": "published", "uncertainty_pct": 2.0}}}
    r = propagate_uncertainty(DesignSpec.model_validate(spec), samples=50, seed=1,
                              include_cost=False)
    peak = next(p for p in r["varied_inputs"] if p["parameter"].endswith("peak_tops"))
    assert peak["uncertainty_pct"] == 2.0 and peak["provenance"] == "published"
    assert r["bands"]["peak_compute_tops"]["spread_pct"] <= 2.0


def test_tracked_values_evaluate_like_plain_numbers():
    from chiplab.schemas.design import plain_design
    tracked = {**H100_LIKE, "chiplets": [
        {**H100_LIKE["chiplets"][0],
         "area_mm2": {"value": 814, "provenance": "published", "source": "NVIDIA"}},
        H100_LIKE["chiplets"][1]]}
    d = DesignSpec.model_validate(tracked)
    plain, recs = plain_design(d)
    assert recs[0]["parameter"] == "chiplets[0].area_mm2" and recs[0]["source"] == "NVIDIA"
    r_t = run_analytical_simulation(d, None, SimulationConfig())
    r_p = run_analytical_simulation(DesignSpec.model_validate(H100_LIKE), None,
                                    SimulationConfig())
    assert r_t.metrics == r_p.metrics
    assert any(a.parameter == "chiplets[0].area_mm2" and a.source == "NVIDIA"
               for a in r_t.assumptions)
    assert model.estimate_unit_cost(d)["unit_cost_usd"] == \
        model.estimate_unit_cost(DesignSpec.model_validate(H100_LIKE))["unit_cost_usd"]


def test_uncertainty_tool_registered_read_only():
    import asyncio

    from chiplab.mcp_server import create_mcp_server
    tools = {t.name: t for t in asyncio.run(create_mcp_server().list_tools())}
    ann = tools["chip_uncertainty"].annotations
    assert getattr(ann, "read_only_hint", getattr(ann, "readOnlyHint", None)) is True
