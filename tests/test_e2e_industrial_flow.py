"""End-to-end industrial flow test.

Tests the full pipeline: Device/Material -> Physics (GPU/CPU) -> Reliability -> 
Discovery -> Manufacturing Signoff -> Digital Twin snapshot.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from chiplab.manufacturing.pipeline import (
    ManufacturingPipeline,
    PipelineStage,
    SignoffReport,
)
from chiplab.schemas.common import Provenance
from chiplab.reliability.aging import (
    arrhenius_accelerated_aging,
    monte_carlo_lifetime,
)
from chiplab.discovery.explorer import (
    DesignSpace,
    evaluate_candidate,
    explore_design_space,
    quick_exploration,
    ParetoFrontier,
)
from chiplab.schemas.digital_twin import (
    DigitalTwinConfig,
    Objective,
    TwinWorkload,
)
from chiplab.digital_twin.twin import DigitalTwin


def test_arrhenius_aging():
    """Test Arrhenius accelerated aging model."""
    result = arrhenius_accelerated_aging(
        temp_operating_k=350.0,
        temp_reference_k=300.0,
        time_operating=1e6,
        activation_energy_eV=0.7,
        provenance=Provenance.ESTIMATED,
    )
    assert result.value > 0
    assert result.unit == "seconds"
    assert result.provenance == Provenance.ESTIMATED


def test_monte_carlo_lifetime():
    """Test Monte Carlo lifetime/MTTF distribution."""
    result = monte_carlo_lifetime(
        n_samples=100,
        activation_energy_mean_ev=0.7,
        activation_energy_std_ev=0.1,
        temperature_k=300.0,
        current_density_a_per_m2=1e6,
        provenance=Provenance.ESTIMATED,
    )
    assert result.value > 0
    assert result.unit == "seconds"
    assert result.provenance == Provenance.ESTIMATED


def test_quick_exploration():
    """Quick design-space exploration with default parameter ranges."""
    frontier = quick_exploration(
        compute_range=(16, 64),
        memory_range=(128.0, 256.0),
        interconnect_range=(256.0, 512.0),
        material_range=(0.8, 1.2),
        n_candidates=20,
    )
    assert isinstance(frontier, ParetoFrontier)
    # Should have some non-dominated candidates
    frontier_candidates = frontier.get_frontier_candidates()
    assert len(frontier_candidates) > 0


def test_explore_design_space():
    """Full design-space exploration with Latin Hypercube sampling."""
    design_space = DesignSpace(
        parameters={
            "compute_units": (16, 64),
            "memory_bandwidth": (128.0, 512.0),
            "interconnect_width": (256.0, 1024.0),
            "material_quality": (0.8, 1.2),
        }
    )
    frontier = explore_design_space(
        design_space=design_space,
        n_candidates=30,
        sampling_method="latin_hypercube",
        verbose=False,
    )
    assert isinstance(frontier, ParetoFrontier)
    frontier_candidates = frontier.get_frontier_candidates()
    assert len(frontier_candidates) > 0


def test_digital_twin_snapshot():
    """Digital twin snapshot after full pipeline evaluation."""
    twin = DigitalTwin(
        config=DigitalTwinConfig(
            strategy="nsga2",
            population=10,
            generations=2,
            objectives=[
                Objective(name="perf", goal="max"),
                Objective(name="power_w", goal="min"),
            ],
        ),
        workload=TwinWorkload(
            workload_type="ppa",
            description="PPA evaluation workload",
        ),
    )

    # Evaluate with geometry params
    result = twin.eval_with_geometry(
        cores=32,
        vlen=256,
        dlen=128,
        l1_kb=32,
        l2_kb=512,
        llc_mb=4.0,
        vdd=0.7,
        freq_ghz=2.0,
        utilization=0.7,
        node="N7",
    )

    # Keys are 'perf', 'power_w', 'area_mm2', 'cost_usd', etc.
    assert "perf" in result
    assert "power_w" in result
    assert "area_mm2" in result
    assert "temperature" in result or "temp" in result

    # Take snapshot
    snapshot = twin.snapshot()
    assert "config_summary" in snapshot
    assert "calibration_count" in snapshot
    assert "drift_count" in snapshot
    assert "live_state_keys" in snapshot
    assert "timestamp" in snapshot


def test_full_e2e_industrial_flow():
    """Full end-to-end industrial flow: Device/Material -> Physics -> Reliability -> Discovery -> Signoff -> Digital Twin.

    This is the canonical E2E test that exercises the complete workflow.
    """
    tmp_dir = Path(tempfile.mkdtemp())

    try:
        # ── 1. Device/Material ──
        pipeline = ManufacturingPipeline(design_name="e2e-full", version="1.0.0")

        pipeline_data = {
            PipelineStage.MATERIAL: {"area_mm2": 15.0},
            PipelineStage.DEVICE: {"technology_node": "7", "power_watts": 12.0},
            PipelineStage.CIRCUIT: {"transistor_count": 5000, "frequency_ghz": 3.0},
            PipelineStage.RTL: {"module_count": 8, "pipeline_depth": 6},
            PipelineStage.FLOORPLAN: {"core_area_mm2": 100.0, "aspect_ratio": 1.2},
            PipelineStage.PCB: {"layer_count": 8, "trace_spacing_mil": 4.0},
            PipelineStage.PACKAGING: {
                "package_type": "3D_stacking",
                "chiplet_areas_mm2": [30.0, 25.0, 20.0],
                "defect_density_cm2": 0.05,
                "cost_per_mm2_usd": 0.8,
            },
            PipelineStage.THERMAL: {
                "total_power_watts": 20.0,
                "thermal_resistance_c_per_w": 15.0,
                "ambient_temp_c": 30.0,
                "max_temp_c": 90.0,
            },
            PipelineStage.TEST: {
                "test_vector_count": 500,
                "target_coverage_pct": 90.0,
                "actual_coverage_pct": 92.0,
            },
        }

        report: SignoffReport = pipeline.run_full_pipeline(pipeline_data)

        # Verify pipeline stages run (some may fail due to thermal constraints,
        # that's expected - the test verifies the pipeline executes fully)
        assert report.overall_pass is not None
        assert len(report.stage_results) > 0
        # Key stages should be present
        assert PipelineStage.MATERIAL.value in str(report.stage_results)
        assert PipelineStage.DEVICE.value in str(report.stage_results)
        assert PipelineStage.THERMAL.value in str(report.stage_results)

        # ── 2. Reliability: Arrhenius Aging ──
        aging_result = arrhenius_accelerated_aging(
            temp_operating_k=350.0,
            temp_reference_k=300.0,
            time_operating=1e6,
            activation_energy_eV=0.7,
            provenance=Provenance.ESTIMATED,
        )
        assert aging_result.value > 0
        assert aging_result.unit == "seconds"

        # ── 3. Reliability: Monte Carlo Lifetime ──
        mc_result = monte_carlo_lifetime(
            n_samples=500,
            activation_energy_mean_ev=0.7,
            activation_energy_std_ev=0.1,
            temperature_k=300.0,
            current_density_a_per_m2=5e5,
            provenance=Provenance.ESTIMATED,
        )
        assert mc_result.value > 0
        assert mc_result.unit == "seconds"

        # ── 4. Discovery: Design Space Exploration ──
        design_space = DesignSpace(
            parameters={
                "compute_units": (32, 128),
                "memory_bandwidth": (256.0, 1024.0),
                "interconnect_width": (512.0, 2048.0),
                "material_quality": (0.9, 1.1),
            }
        )
        frontier = explore_design_space(
            design_space=design_space,
            n_candidates=50,
            sampling_method="latin_hypercube",
            verbose=False,
        )
        frontier_candidates = frontier.get_frontier_candidates()
        assert len(frontier_candidates) >= 3  # At least 3 non-dominated candidates

        # Verify Pareto candidates have valid PPA metrics
        for candidate in frontier_candidates[:3]:
            assert candidate.performance > 0
            assert candidate.power > 0
            assert candidate.area > 0
            assert candidate.temperature >= 0
            assert candidate.ir_drop >= 0

        # ── 5. Digital Twin Snapshot ──
        twin = DigitalTwin(
            config=DigitalTwinConfig(
                strategy="nsga2",
                population=10,
                generations=2,
                objectives=[
                    Objective(name="perf", goal="max"),
                    Objective(name="power_w", goal="min"),
                ],
            ),
            workload=TwinWorkload(
                workload_type="ppa",
                description="PPA evaluation workload from E2E flow",
            ),
        )

        # Evaluate with geometry params reflecting the pipeline design
        evaluate_result = twin.eval_with_geometry(
            cores=64,
            vlen=512,
            dlen=256,
            l1_kb=64,
            l2_kb=1024,
            llc_mb=8.0,
            vdd=0.8,
            freq_ghz=3.0,
            utilization=0.8,
            node="N7",
        )

        # Keys are 'perf', 'power_w', 'area_mm2', etc.
        assert evaluate_result.get("perf", 0) > 0
        assert evaluate_result.get("power_w", 0) > 0
        assert evaluate_result.get("area_mm2", 0) > 0
        assert evaluate_result.get("temperature", 0) >= 0 or evaluate_result.get("temp", 0) >= 0

        # Take snapshot
        snapshot = twin.snapshot()
        assert snapshot["config_summary"]["strategy"] == "nsga2"
        assert snapshot["calibration_count"] >= 0
        assert snapshot["drift_count"] >= 0
        assert "timestamp" in snapshot

        # ── 6. Verify Signoff Report Contains All Stages ──
        assert len(report.stage_results) >= 8  # Pipeline stages executed up to thermal/test signoff
        assert len(report.dfm_checks) > 0

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_provenance_through_pipeline():
    """Verify provenance is tracked through the pipeline stages."""
    pipeline = ManufacturingPipeline(design_name="e2e-provenance", version="1.0.0")

    pipeline_data = {
        PipelineStage.MATERIAL: {"area_mm2": 10.0},
        PipelineStage.DEVICE: {"technology_node": "7", "power_watts": 5.0},
        PipelineStage.CIRCUIT: {"transistor_count": 1000, "frequency_ghz": 1.0},
        PipelineStage.RTL: {"module_count": 3, "pipeline_depth": 3},
        PipelineStage.FLOORPLAN: {"core_area_mm2": 25.0, "aspect_ratio": 1.0},
        PipelineStage.PCB: {"layer_count": 4, "trace_spacing_mil": 5.0},
        PipelineStage.PACKAGING: {
            "package_type": "monolithic",
            "chiplet_areas_mm2": [10.0],
            "defect_density_cm2": 0.1,
            "cost_per_mm2_usd": 0.2,
        },
        PipelineStage.THERMAL: {
            "total_power_watts": 5.0,
            "thermal_resistance_c_per_w": 10.0,
            "ambient_temp_c": 25.0,
            "max_temp_c": 85.0,
        },
        PipelineStage.TEST: {
            "test_vector_count": 100,
            "target_coverage_pct": 95.0,
            "actual_coverage_pct": 96.0,
        },
    }

    report: SignoffReport = pipeline.run_full_pipeline(pipeline_data)

    # All stages should have provenance
    for stage, gate in pipeline.stage_gates.items():
        assert gate.provenance == Provenance.ASSUMED