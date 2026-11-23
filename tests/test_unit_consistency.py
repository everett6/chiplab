"""Regression tests for unit and formula consistency in the simulation engines.

Synthetic workloads only; the ECHO design YAML is the same fixture the existing tests use.
"""

import yaml
import pytest

from chiplab.schemas.design import DesignSpec
from chiplab.schemas.workload import WorkloadSpec
from chiplab.schemas.simulation import SimulationConfig, FidelityLevel
from chiplab.engine.discrete_event import run_discrete_event_simulation
from chiplab.engine.analytical import run_analytical_simulation

ECHO_YAML = "/home/everett/Chip Lab/projects/echo/echo_v0_1.yaml"
# The ECHO design fails DES validation (pim_0 has zero capacity), so DES tests use the example.
DES_YAML = "/home/everett/Chip Lab/examples/transformer_accelerator.yaml"


def _load(path: str) -> DesignSpec:
    with open(path) as f:
        return DesignSpec.from_yaml(yaml.dump(yaml.safe_load(f)))


def _design() -> DesignSpec:
    return _load(ECHO_YAML)


def _des_design() -> DesignSpec:
    return _load(DES_YAML)


def _workload(flops: float = 1.0e12, bytes_read: float = 1.0e9, bytes_write: float = 0.0) -> WorkloadSpec:
    return WorkloadSpec(
        name="synthetic",
        type="moe_decode",
        batch_size=1,
        sequence_length=128,
        hidden_dim=4096,
        num_layers=32,
        num_experts=8,
        active_experts=2,
        precision="INT8",
        flops=flops,
        bytes_read=bytes_read,
        bytes_write=bytes_write,
    )


def test_des_compute_time_unit_conversion():
    """Compute time must be FLOPs / (TOPS * 1e12) seconds, reported in ms."""
    design = _des_design()
    wl = _workload(flops=1.0e12, bytes_read=0.0, bytes_write=0.0)
    cfg = SimulationConfig(fidelity=FidelityLevel.DISCRETE_EVENT, num_requests=1)
    result = run_discrete_event_simulation(design, wl, cfg, design_id="transformer")
    peak_tops = result.metrics.peak_compute_tops
    assert peak_tops > 0
    assert result.success, result.error_message
    expected_ms = 1.0e12 / (peak_tops * 1e12) * 1e3
    assert result.metrics.estimated_latency_ms == pytest.approx(expected_ms, rel=1e-2)


def test_des_energy_is_power_times_latency_in_mj():
    """W * ms = mJ, so energy must not be divided by 1000."""
    design = _des_design()
    wl = _workload()
    cfg = SimulationConfig(fidelity=FidelityLevel.DISCRETE_EVENT, num_requests=1)
    m = run_discrete_event_simulation(design, wl, cfg, design_id="transformer").metrics
    assert m.total_power_watts > 0 and m.estimated_latency_ms > 0
    assert m.energy_per_inference_mj == pytest.approx(
        m.total_power_watts * m.estimated_latency_ms, rel=1e-2
    )


def test_des_ridge_point_is_peak_over_bandwidth():
    """Ridge point (ops/byte) = peak TOPS / bandwidth TB/s."""
    design = _des_design()
    cfg = SimulationConfig(fidelity=FidelityLevel.DISCRETE_EVENT, num_requests=1)
    m = run_discrete_event_simulation(design, _workload(), cfg, design_id="transformer").metrics
    assert m.aggregate_memory_bandwidth_tbps > 0
    expected = m.peak_compute_tops / m.aggregate_memory_bandwidth_tbps
    assert m.ridge_point_ops_per_byte == pytest.approx(expected, rel=1e-2)


def test_des_utilization_is_compute_fraction_of_request_time():
    """Utilization is compute time over total request time, capped at 100 percent."""
    design = _des_design()
    cfg = SimulationConfig(fidelity=FidelityLevel.DISCRETE_EVENT, num_requests=1)
    # Memory-heavy workload: compute is a small fraction of total time.
    m = run_discrete_event_simulation(
        design, _workload(flops=1.0e9, bytes_read=1.0e12), cfg, design_id="transformer"
    ).metrics
    assert 0.0 <= m.utilization_pct < 100.0


def test_analytical_energy_is_power_times_latency_in_mj():
    design = _design()
    wl = _workload()
    cfg = SimulationConfig(fidelity=FidelityLevel.ANALYTICAL)
    m = run_analytical_simulation(design, wl, cfg, design_id="echo_v0_1").metrics
    assert m.estimated_latency_ms > 0
    assert m.energy_per_inference_mj == pytest.approx(
        m.total_power_watts * m.estimated_latency_ms, rel=1e-3
    )


def test_analytical_ridge_point_matches_peak_over_bandwidth():
    design = _design()
    wl = _workload()
    cfg = SimulationConfig(fidelity=FidelityLevel.ANALYTICAL)
    m = run_analytical_simulation(design, wl, cfg, design_id="echo_v0_1").metrics
    assert m.aggregate_memory_bandwidth_tbps > 0
    expected = m.effective_compute_tops / m.aggregate_memory_bandwidth_tbps
    assert m.ridge_point_ops_per_byte == pytest.approx(expected, rel=1e-3)
