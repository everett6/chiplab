"""Tests for uncertainty propagation in analytical simulation."""

import pytest
from chiplab.schemas.design import DesignSpec, ChipletSpec
from chiplab.schemas.simulation import SimulationConfig, FidelityLevel
from chiplab.schemas.workload import WorkloadSpec
from chiplab.engine.analytical import run_analytical_simulation
from chiplab.schemas.common import TrackedValue


def test_uncertainty_backwards_compatible():
    """Designs without uncertainty_pct should work identically (backwards-compatible)."""
    design = DesignSpec(
        name="test_design",
        version="0.1",
        package={},  # empty package defaults
        chiplets=[
            ChipletSpec(
                type="compute",
                count=1,
                compute={
                    "architecture": "test",
                    "precision": "INT8",
                    "peak_tops": 500.0,
                },
                memory=None,
            )
        ],
        interconnect={},
        constraints={},
    )

    workload = WorkloadSpec(
        name="test_workload",
        type="compute",
        batch_size=1,
        sequence_length=64,
    )

    config = SimulationConfig(fidelity=FidelityLevel.ANALYTICAL)
    result = run_analytical_simulation(design, workload, config, design_id="test_backward")

    # Should work without errors
    assert result.success == True
    assert result.fidelity == FidelityLevel.ANALYTICAL
    # uncertainty_bounds should be empty/default when no uncertainty_pct is present
    assert result.uncertainty_bounds == {}


def test_uncertainty_with_tracked_value():
    """Designs with TrackedValue(uncertainty_pct=15.0) produce uncertainty_bounds where low < nominal < high."""
    design = DesignSpec(
        name="test_design_uncertainty",
        version="0.1",
        package={},
        chiplets=[
            ChipletSpec(
                type="compute",
                count=1,
                area_mm2=TrackedValue(value=100.0, uncertainty_pct=15.0),
                tdp_watts=TrackedValue(value=50.0, uncertainty_pct=10.0),
                compute={
                    "architecture": "test",
                    "precision": "INT8",
                    "peak_tops": TrackedValue(value=500.0, uncertainty_pct=5.0),
                },
                memory=None,
            )
        ],
        interconnect={},
        constraints={},
    )

    workload = WorkloadSpec(
        name="test_workload",
        type="compute",
        batch_size=1,
        sequence_length=64,
    )

    config = SimulationConfig(fidelity=FidelityLevel.ANALYTICAL)
    result = run_analytical_simulation(design, workload, config, design_id="test_uncertainty")

    # Should complete successfully
    assert result.success == True

    # uncertainty_bounds should be populated
    assert "total_area_mm2" in result.uncertainty_bounds, "total_area_mm2 should have uncertainty bounds"
    assert "total_power_watts" in result.uncertainty_bounds, "total_power_watts should have uncertainty bounds"
    # uncertainty_bounds should be populated with mapped metrics
    assert "total_area_mm2" in result.uncertainty_bounds, "total_area_mm2 should have uncertainty bounds"
    assert "total_power_watts" in result.uncertainty_bounds, "total_power_watts should have uncertainty bounds"

    # Check that low < nominal < high for each metric
    for metric_name, bounds in result.uncertainty_bounds.items():
        # nominal should be 1.0 (normalized), low < 1.0 < high
        assert bounds["low"] < bounds["nominal"] < bounds["high"], \
            f"For {metric_name}: low={bounds['low']} not < nominal={bounds['nominal']} < high={bounds['high']}"

    # Verify specific uncertainty_pct values are computed via RSS
    # With 15% area and 10% power, RSS should be sqrt(15^2 + 10^2) = sqrt(325) ≈ 18.03%
    assert result.uncertainty_bounds["total_area_mm2"]["uncertainty_pct"] > 0
    assert result.uncertainty_bounds["total_power_watts"]["uncertainty_pct"] > 0


def test_uncertainty_mixed_no_uncertainty():
    """Designs with some TrackedValues without uncertainty_pct should still work."""
    design = DesignSpec(
        name="test_design_mixed",
        version="0.1",
        package={},
        chiplets=[
            ChipletSpec(
                type="compute",
                count=1,
                area_mm2=TrackedValue(value=100.0, uncertainty_pct=0.0),  # 0% uncertainty
                tdp_watts=TrackedValue(value=50.0),  # No uncertainty_pct field
                compute={
                    "architecture": "test",
                    "precision": "INT8",
                    "peak_tops": TrackedValue(value=500.0, uncertainty_pct=5.0),
                },
                memory=None,
            )
        ],
        interconnect={},
        constraints={},
    )

    workload = WorkloadSpec(
        name="test_workload",
        type="compute",
        batch_size=1,
        sequence_length=64,
    )

    config = SimulationConfig(fidelity=FidelityLevel.ANALYTICAL)
    result = run_analytical_simulation(design, workload, config, design_id="test_mixed")

    assert result.success == True
    # Should have uncertainty_bounds for the metric with uncertainty_pct > 0
    assert "total_area_mm2" in result.uncertainty_bounds or "estimated_latency_ms" in result.uncertainty_bounds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])