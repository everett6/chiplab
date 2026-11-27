"""Tests for the manufacturing pipeline.

Tests full pipeline execution, stage gate pass/fail, and signoff report export.
"""

import pytest

from chiplab.manufacturing import (
    ManufacturingPipeline,
    PipelineStage,
    StageGate,
    GateResult,
    DFMCheck,
    PVTVariation,
    SignoffReport,
    material_stage,
    device_stage,
    circuit_stage,
    rtl_stage,
    floorplan_stage,
    pcb_stage,
    packaging_stage,
    thermal_stage,
    test_stage as run_test_stage,
)


class TestManufacturingPipeline:
    """Test the full manufacturing pipeline."""

    def test_full_pipeline_pass(self):
        """Test a full pipeline run where all stages pass."""
        pipeline = ManufacturingPipeline(design_name="test_design", version="1.0")

        pipeline_data = {
            PipelineStage.MATERIAL: {"area_mm2": 500.0},
            PipelineStage.DEVICE: {
                "technology_node": "14",
                "power_watts": 100.0,
            },
            PipelineStage.CIRCUIT: {
                "transistor_count": 1000000,
            },
            PipelineStage.RTL: {
                "module_count": 8,
                "pipeline_depth": 5,
            },
            PipelineStage.FLOORPLAN: {
                "core_area_mm2": 200.0,
                "aspect_ratio": 1.5,
            },
            PipelineStage.PCB: {
                "layer_count": 8,
                "trace_spacing_mil": 5.0,
            },
            PipelineStage.PACKAGING: {
                "package_type": "2.5D",
                "chiplet_areas_mm2": [100.0, 150.0, 120.0],
                "defect_density_cm2": 0.1,
                "cost_per_mm2_usd": 0.5,
            },
            PipelineStage.THERMAL: {
                "total_power_watts": 150.0,
                "thermal_resistance_c_per_w": 0.1,
                "ambient_temp_c": 25.0,
                "max_temp_c": 85.0,
            },
            PipelineStage.TEST: {
                "test_vector_count": 1000,
                "target_coverage_pct": 95.0,
                "actual_coverage_pct": 97.0,
            },
        }

        report = pipeline.run_full_pipeline(pipeline_data)

        # Verify signoff report
        assert isinstance(report, SignoffReport)
        assert report.design_name == "test_design"
        assert report.version == "1.0"

        # Verify all stages passed
        for stage, result in report.stage_results.items():
            assert result == GateResult.PASS, f"Stage {stage.value} failed with result {result}"

        # Verify overall pass
        assert report.overall_pass is True

        # Verify DFM checks exist for each stage
        for stage in PipelineStage:
            assert stage in report.dfm_checks, f"DFM check missing for stage {stage.value}"

        # Verify yield was calculated
        assert report.yield_pct is not None
        assert 0 <= report.yield_pct <= 100

    def test_full_pipeline_with_failures(self):
        """Test a full pipeline run where some stages fail."""
        pipeline = ManufacturingPipeline(design_name="failing_design", version="1.0")

        pipeline_data = {
            PipelineStage.MATERIAL: {"area_mm2": -10.0},  # Invalid: negative area
            # Skip remaining stages since material will fail
        }

        report = pipeline.run_full_pipeline(pipeline_data)

        # Material stage should fail
        assert (
            PipelineStage.MATERIAL in report.stage_results
            and report.stage_results[PipelineStage.MATERIAL] == GateResult.FAIL
        )

        # Overall should fail
        assert report.overall_pass is False

    def test_pvt_summary_generated(self):
        """Test that PVT variation summary is generated."""
        pipeline = ManufacturingPipeline(design_name="pvt_test", version="1.0")

        pipeline_data = {
            PipelineStage.MATERIAL: {"area_mm2": 500.0},
            PipelineStage.DEVICE: {
                "technology_node": "14",
                "power_watts": 100.0,
            },
            PipelineStage.CIRCUIT: {"transistor_count": 1000000},
            PipelineStage.RTL: {"module_count": 8, "pipeline_depth": 5},
            PipelineStage.FLOORPLAN: {"core_area_mm2": 200.0, "aspect_ratio": 1.5},
            PipelineStage.PCB: {"layer_count": 8, "trace_spacing_mil": 5.0},
            PipelineStage.PACKAGING: {
                "package_type": "2.5D",
                "chiplet_areas_mm2": [100.0, 150.0],
                "defect_density_cm2": 0.1,
                "cost_per_mm2_usd": 0.5,
            },
            PipelineStage.THERMAL: {
                "total_power_watts": 150.0,
                "thermal_resistance_c_per_w": 0.1,
                "ambient_temp_c": 25.0,
                "max_temp_c": 85.0,
            },
            PipelineStage.TEST: {
                "test_vector_count": 1000,
                "target_coverage_pct": 95.0,
                "actual_coverage_pct": 97.0,
            },
        }

        report = pipeline.run_full_pipeline(pipeline_data)

        # PVT summary should be in the report
        assert report.pvt_summary is not None
        assert len(report.pvt_summary.process_corners) > 0
        assert len(report.pvt_summary.voltages) > 0
        assert len(report.pvt_summary.temperatures) > 0

    def test_signoff_report_export(self):
        """Test signoff report dictionary export."""
        pipeline = ManufacturingPipeline(design_name="export_test", version="0.5")

        pipeline_data = {
            PipelineStage.MATERIAL: {"area_mm2": 500.0},
            PipelineStage.DEVICE: {
                "technology_node": "14",
                "power_watts": 100.0,
            },
            PipelineStage.CIRCUIT: {"transistor_count": 1000000},
            PipelineStage.RTL: {"module_count": 8, "pipeline_depth": 5},
            PipelineStage.FLOORPLAN: {"core_area_mm2": 200.0, "aspect_ratio": 1.5},
            PipelineStage.PCB: {"layer_count": 8, "trace_spacing_mil": 5.0},
            PipelineStage.PACKAGING: {
                "package_type": "2.5D",
                "chiplet_areas_mm2": [100.0, 150.0],
                "defect_density_cm2": 0.1,
                "cost_per_mm2_usd": 0.5,
            },
            PipelineStage.THERMAL: {
                "total_power_watts": 150.0,
                "thermal_resistance_c_per_w": 0.1,
                "ambient_temp_c": 25.0,
                "max_temp_c": 85.0,
            },
            PipelineStage.TEST: {
                "test_vector_count": 1000,
                "target_coverage_pct": 95.0,
                "actual_coverage_pct": 97.0,
            },
        }

        report = pipeline.run_full_pipeline(pipeline_data)

        report_dict = report.to_dict()

        # Verify dictionary structure
        assert "design_name" in report_dict
        assert "version" in report_dict
        assert "overall_pass" in report_dict
        assert "stage_results" in report_dict
        assert "dfm_checks" in report_dict
        assert "pvt_summary" in report_dict
        assert "total_cost_usd" in report_dict
        assert "yield_pct" in report_dict
        assert "notes" in report_dict

        # Verify stage results structure
        assert PipelineStage.MATERIAL.value in report_dict["stage_results"]
        assert PipelineStage.TEST.value in report_dict["stage_results"]

        # Verify overall pass is True
        assert report_dict["overall_pass"] is True

    def test_stage_gates_creation(self):
        """Test that stage gates are created correctly."""
        pipeline = ManufacturingPipeline(design_name="gate_test", version="1.0")

        # Run just the material stage
        result, dfm_check = material_stage(
            pipeline,
            {"technology": "SRAM"},
            area_mm2=500.0,
        )

        # Verify gate was created
        assert PipelineStage.MATERIAL in pipeline.stage_gates
        gate = pipeline.stage_gates[PipelineStage.MATERIAL]
        assert gate.name == "material_gate"
        assert gate.stage == PipelineStage.MATERIAL

        # Verify result
        assert isinstance(result, GateResult)
        assert result in [GateResult.PASS, GateResult.FAIL, GateResult.WARNING]

        # Verify DFM check
        assert isinstance(dfm_check, DFMCheck)
        assert hasattr(dfm_check, "pass_flag")
        assert hasattr(dfm_check, "checks")

    def test_individual_stage_functions(self):
        """Test individual stage convenience functions."""
        pipeline = ManufacturingPipeline(design_name="stage_func_test", version="1.0")

        # Test material stage
        result, dfm_check = material_stage(
            pipeline,
            {"technology": "SRAM"},
            area_mm2=500.0,
        )
        assert PipelineStage.MATERIAL in pipeline.stage_results

        # Test device stage
        result, dfm_check = device_stage(
            pipeline,
            {"technology_node": "7", "power_watts": 50.0},
        )
        assert PipelineStage.DEVICE in pipeline.stage_results

        # Test circuit stage
        result, dfm_check = circuit_stage(
            pipeline,
            {"transistor_count": 500000},
        )
        assert PipelineStage.CIRCUIT in pipeline.stage_results

        # Test RTL stage
        result, dfm_check = rtl_stage(
            pipeline,
            {"module_count": 4, "pipeline_depth": 3},
        )
        assert PipelineStage.RTL in pipeline.stage_results

        # Test floorplan stage
        result, dfm_check = floorplan_stage(
            pipeline,
            {"core_area_mm2": 300.0, "aspect_ratio": 1.2},
        )
        assert PipelineStage.FLOORPLAN in pipeline.stage_results

        # Test PCB stage
        result, dfm_check = pcb_stage(
            pipeline,
            {"layer_count": 6, "trace_spacing_mil": 4.0},
        )
        assert PipelineStage.PCB in pipeline.stage_results

        # Test packaging stage
        result, dfm_check = packaging_stage(
            pipeline,
            {
                "package_type": "chiplets",
                "chiplet_areas_mm2": [50.0, 75.0],
                "defect_density_cm2": 0.05,
                "cost_per_mm2_usd": 0.3,
            },
        )
        assert PipelineStage.PACKAGING in pipeline.stage_results

        # Test thermal stage
        result, dfm_check = thermal_stage(
            pipeline,
            {
                "total_power_watts": 100.0,
                "thermal_resistance_c_per_w": 0.15,
                "ambient_temp_c": 30.0,
                "max_temp_c": 85.0,
            },
        )
        assert PipelineStage.THERMAL in pipeline.stage_results

        # Test test stage
        result, dfm_check = run_test_stage(
            pipeline,
            {
                "test_vector_count": 500,
                "target_coverage_pct": 90.0,
                "actual_coverage_pct": 92.0,
            },
        )
        assert PipelineStage.TEST in pipeline.stage_results

    def test_yield_calculation_with_poisson(self):
        """Test yield calculation using Poisson model from yield_model."""
        pipeline = ManufacturingPipeline(design_name="yield_test", version="1.0")

        pipeline_data = {
            PipelineStage.MATERIAL: {"area_mm2": 500.0},
            PipelineStage.DEVICE: {
                "technology_node": "28",
                "power_watts": 50.0,
            },
            PipelineStage.CIRCUIT: {"transistor_count": 100000},
            PipelineStage.RTL: {"module_count": 4, "pipeline_depth": 3},
            PipelineStage.FLOORPLAN: {"core_area_mm2": 100.0, "aspect_ratio": 1.0},
            PipelineStage.PCB: {"layer_count": 4, "trace_spacing_mil": 5.0},
            PipelineStage.PACKAGING: {
                "package_type": "monolithic",
                "chiplet_areas_mm2": [500.0],
                "defect_density_cm2": 0.5,
                "cost_per_mm2_usd": 0.2,
            },
            PipelineStage.THERMAL: {
                "total_power_watts": 30.0,
                "thermal_resistance_c_per_w": 0.2,
                "ambient_temp_c": 25.0,
                "max_temp_c": 85.0,
            },
            PipelineStage.TEST: {
                "test_vector_count": 500,
                "target_coverage_pct": 95.0,
                "actual_coverage_pct": 97.0,
            },
        }

        report = pipeline.run_full_pipeline(pipeline_data)

        # Yield should be calculated and be a reasonable percentage
        assert report.yield_pct is not None
        assert 0 <= report.yield_pct <= 100

        # DFM check for packaging should have yield
        pkg_dfm = report.dfm_checks.get(PipelineStage.PACKAGING)
        assert pkg_dfm is not None
        assert pkg_dfm.yield_pct is not None
        assert 0 <= pkg_dfm.yield_pct <= 100

    def test_cost_tracking(self):
        """Test cost tracking through the pipeline."""
        pipeline = ManufacturingPipeline(design_name="cost_test", version="1.0")

        pipeline_data = {
            PipelineStage.MATERIAL: {"area_mm2": 500.0},
            PipelineStage.DEVICE: {
                "technology_node": "14",
                "power_watts": 100.0,
            },
            PipelineStage.CIRCUIT: {"transistor_count": 1000000},
            PipelineStage.RTL: {"module_count": 8, "pipeline_depth": 5},
            PipelineStage.FLOORPLAN: {"core_area_mm2": 200.0, "aspect_ratio": 1.5},
            PipelineStage.PCB: {"layer_count": 8, "trace_spacing_mil": 5.0},
            PipelineStage.PACKAGING: {
                "package_type": "2.5D",
                "chiplet_areas_mm2": [100.0, 150.0, 120.0],
                "defect_density_cm2": 0.1,
                "cost_per_mm2_usd": 0.5,
            },
            PipelineStage.THERMAL: {
                "total_power_watts": 150.0,
                "thermal_resistance_c_per_w": 0.1,
                "ambient_temp_c": 25.0,
                "max_temp_c": 85.0,
            },
            PipelineStage.TEST: {
                "test_vector_count": 1000,
                "target_coverage_pct": 95.0,
                "actual_coverage_pct": 97.0,
            },
        }

        report = pipeline.run_full_pipeline(pipeline_data)

        # Cost should be tracked
        assert report.total_cost_usd is not None
        assert report.total_cost_usd > 0

        # Cost should be roughly: total_area * cost_per_mm2
        # Total area: 100 + 150 + 120 = 370 mm²
        # Cost: 370 * 0.5 = 185 USD
        assert abs(report.total_cost_usd - 185.0) < 10.0, f"Expected ~185 USD, got {report.total_cost_usd}"

    def test_warning_detection(self):
        """Test that warnings are properly detected and recorded."""
        pipeline = ManufacturingPipeline(design_name="warning_test", version="1.0")

        pipeline_data = {
            PipelineStage.MATERIAL: {"area_mm2": 500.0},
            PipelineStage.DEVICE: {
                "technology_node": "14",
                "power_watts": 600.0,  # High power - should generate warning
            },
            PipelineStage.CIRCUIT: {"transistor_count": 1000000},
            PipelineStage.RTL: {"module_count": 8, "pipeline_depth": 5},
            PipelineStage.FLOORPLAN: {"core_area_mm2": 200.0, "aspect_ratio": 1.5},
            PipelineStage.PCB: {"layer_count": 8, "trace_spacing_mil": 5.0},
            PipelineStage.PACKAGING: {
                "package_type": "2.5D",
                "chiplet_areas_mm2": [100.0, 150.0],
                "defect_density_cm2": 0.1,
                "cost_per_mm2_usd": 0.5,
            },
            PipelineStage.THERMAL: {
                "total_power_watts": 150.0,
                "thermal_resistance_c_per_w": 0.1,
                "ambient_temp_c": 25.0,
                "max_temp_c": 85.0,
            },
            PipelineStage.TEST: {
                "test_vector_count": 1000,
                "target_coverage_pct": 95.0,
                "actual_coverage_pct": 97.0,
            },
        }

        report = pipeline.run_full_pipeline(pipeline_data)

        # Should have warnings in notes
        assert len(report.notes) > 0
        # Check that at least one note mentions a warning
        warning_notes = [n for n in report.notes if "WARNING" in n or "warning" in n.lower()]
        assert len(warning_notes) > 0

    def test_missing_stage_data(self):
        """Test pipeline behavior when stage data is missing."""
        pipeline = ManufacturingPipeline(design_name="missing_data_test", version="1.0")

        pipeline_data = {
            PipelineStage.MATERIAL: {"area_mm2": 500.0},
            # Missing all other stages
        }

        report = pipeline.run_full_pipeline(pipeline_data)

        # Material should pass (valid data)
        assert (
            PipelineStage.MATERIAL in report.stage_results
            and report.stage_results[PipelineStage.MATERIAL] == GateResult.PASS
        )

        # Overall should be evaluated based on what ran
        # Since only material ran, overall depends on implementation
        assert isinstance(report.overall_pass, bool)


def test_stage_functions_standalone():
    """Test individual stage functions can be called standalone."""
    pipeline = ManufacturingPipeline(design_name="standalone_test", version="1.0")

    # Test material stage
    result, dfm_check = material_stage(
        pipeline,
        {"technology": "SRAM"},
        area_mm2=500.0,
    )
    assert PipelineStage.MATERIAL in pipeline.stage_results

    # Test device stage
    result, dfm_check = device_stage(
        pipeline,
        {"technology_node": "7", "power_watts": 50.0},
    )
    assert PipelineStage.DEVICE in pipeline.stage_results

    # Test circuit stage
    result, dfm_check = circuit_stage(
        pipeline,
        {"transistor_count": 500000},
    )
    assert PipelineStage.CIRCUIT in pipeline.stage_results

    # Test RTL stage
    result, dfm_check = rtl_stage(
        pipeline,
        {"module_count": 4, "pipeline_depth": 3},
    )
    assert PipelineStage.RTL in pipeline.stage_results

    # Test floorplan stage
    result, dfm_check = floorplan_stage(
        pipeline,
        {"core_area_mm2": 300.0, "aspect_ratio": 1.2},
    )
    assert PipelineStage.FLOORPLAN in pipeline.stage_results

    # Test PCB stage
    result, dfm_check = pcb_stage(
        pipeline,
        {"layer_count": 6, "trace_spacing_mil": 4.0},
    )
    assert PipelineStage.PCB in pipeline.stage_results

    # Test packaging stage
    result, dfm_check = packaging_stage(
        pipeline,
        {
            "package_type": "chiplets",
            "chiplet_areas_mm2": [50.0, 75.0],
            "defect_density_cm2": 0.05,
            "cost_per_mm2_usd": 0.3,
        },
    )
    assert PipelineStage.PACKAGING in pipeline.stage_results

    # Test thermal stage
    result, dfm_check = thermal_stage(
        pipeline,
        {
            "total_power_watts": 100.0,
            "thermal_resistance_c_per_w": 0.15,
            "ambient_temp_c": 30.0,
            "max_temp_c": 85.0,
        },
    )
    assert PipelineStage.THERMAL in pipeline.stage_results

    # Test test stage
    result, dfm_check = run_test_stage(
        pipeline,
        {
            "test_vector_count": 500,
            "target_coverage_pct": 90.0,
            "actual_coverage_pct": 92.0,
        },
    )
    assert PipelineStage.TEST in pipeline.stage_results
