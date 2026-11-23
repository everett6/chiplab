"""Tests for ExperimentStore SQLite persistence and reporter.py exports."""

import pytest
import os
import tempfile
import json
import csv
from chiplab.experiments.store import ExperimentStore, DesignSpec, SimulationResult, SweepResult, ComparisonResult
from chiplab.schemas.design import DesignSpec as DSDesignSpec, ChipletSpec
from chiplab.schemas.workload import WorkloadSpec
from chiplab.schemas.simulation import SimulationConfig, FidelityLevel, SimulationMetrics, BottleneckReport
from chiplab.engine.analytical import run_analytical_simulation
from chiplab.engine.discrete_event import run_discrete_event_simulation
from chiplab.results.reporter import export_simulation_report, export_comparison_report, export_sweep_report


class TestExperimentStoreDesignPersistence:
    """Test ExperimentStore SQLite persistence for designs."""

    def test_store_and_retrieve_design(self):
        """Test saving and retrieving a design from the store."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            store = ExperimentStore(db_path=db_path)

            # Create and save a design
            design = DSDesignSpec(
                name="test_design",
                version="0.1",
                description="Test design",
                architecture="heterogeneous_accelerator",
                chiplets=[
                    ChipletSpec(type="compute", architecture="SIMD", peak_tops=100.0),
                    ChipletSpec(type="memory", technology="DRAM", capacity_per_chiplet_gb=8.0),
                ],
                interconnect={"topology": "mesh", "link_bandwidth_gbps": 100.0},
            )

            design_id = store.save_design(design)
            assert design_id is not None

            # Retrieve the design using positional arg (design_id_or_name)
            retrieved_id, retrieved_design = store.get_design(design_id)
            assert retrieved_design.name == "test_design"
            assert retrieved_design.version == "0.1"
            assert len(retrieved_design.chiplets) == 2
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_design_versioning_0_1_to_0_2(self):
        """Test design versioning: 0.1 -> 0.2 with parent_version."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            store = ExperimentStore(db_path=db_path)

            # Save v0.1
            design_v01 = DSDesignSpec(
                name="versioned_design",
                version="0.1",
                description="Version 0.1",
                architecture="heterogeneous_accelerator",
                chiplets=[ChipletSpec(type="compute")],
            )
            store.save_design(design_v01)

            # Save v0.2 with parent_version
            design_v02 = DSDesignSpec(
                name="versioned_design",
                version="0.2",
                parent_version="0.1",
                description="Version 0.2",
                architecture="heterogeneous_accelerator",
                chiplets=[ChipletSpec(type="compute", peak_tops=200.0)],
            )
            store.save_design(design_v02)

            # Get design history
            history = store.get_design_history("versioned_design")
            assert len(history) == 2
            assert history[0]["version"] == "0.1"
            assert history[1]["version"] == "0.2"
            assert history[1]["parent_version"] == "0.1"
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_get_design_nonexistent(self):
        """Test getting a non-existent design raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            store = ExperimentStore(db_path=db_path)
            with pytest.raises(ValueError, match="Design not found"):
                store.get_design("nonexistent")
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestExperimentStoreSimulationPersistence:
    """Test ExperimentStore SQLite persistence for simulations."""

    def test_store_and_retrieve_simulation(self):
        """Test saving and retrieving a simulation result."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            store = ExperimentStore(db_path=db_path)

            # Create and save a design
            design = DSDesignSpec(
                name="sim_test_design",
                version="0.1",
                architecture="heterogeneous_accelerator",
                chiplets=[ChipletSpec(type="compute", architecture="SIMD", peak_tops=100.0)],
            )
            design_id = store.save_design(design)

            # Create a simulation result with metrics as dict
            metrics_dict = SimulationMetrics(
                total_memory_capacity_gb=8.0,
                aggregate_memory_bandwidth_tbps=64.0,
                peak_compute_tops=100.0,
                effective_compute_tops=80.0,
                interconnect_bisection_bw_tbps=12.5,
                estimated_latency_ms=5.5,
                throughput_tokens_per_sec=1000.0,
                throughput_ops_per_sec=80e12,
                total_area_mm2=500.0,
                total_power_watts=25.0,
                estimated_temp_c=65.0,
                estimated_cost_usd=10000.0,
                arithmetic_intensity=25.0,
                ridge_point_ops_per_byte=8000.0,
                utilization_pct=80.0,
                energy_per_inference_mj=2.5,
            ).model_dump()

            bottleneck_dict = BottleneckReport(
                primary_bottleneck="balanced",
                bound_type="balanced",
                severity_score=30.0,
                utilization_breakdown={},
                details=[],
                recommendations=[],
            ).model_dump()

            sim_result = SimulationResult(
                run_id="test_run_1",
                design_id=design_id,
                design_name="sim_test_design",
                design_version="0.1",
                workload_name="default_workload",
                fidelity=FidelityLevel.ANALYTICAL,
                fidelity_disclaimer="Level 1 analytical model",
                timestamp=1234567890.0,
                metrics=metrics_dict,
                bottleneck=bottleneck_dict,
                assumptions=[],
                warnings=[],
            )

            run_id = store.save_simulation(sim_result)
            assert run_id is not None

            # Retrieve the simulation
            retrieved = store.get_simulation(run_id="test_run_1")
            # retrieved.metrics is a dict now, access via key
            assert retrieved.run_id == "test_run_1"
            assert retrieved.design_name == "sim_test_design"
            assert retrieved.fidelity == FidelityLevel.ANALYTICAL
            assert retrieved.metrics.total_memory_capacity_gb == 8.0
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_simulation_save_get_with_versioning(self):
        """Test simulation save/get with design versioning."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            store = ExperimentStore(db_path=db_path)

            # Save v0.1 design and simulation
            design_v01 = DSDesignSpec(
                name="versioned_sim",
                version="0.1",
                architecture="heterogeneous_accelerator",
                chiplets=[ChipletSpec(type="compute", peak_tops=100.0)],
            )
            design_id_v01 = store.save_design(design_v01)

            sim_v01 = SimulationResult(
                run_id="sim_01",
                design_id=design_id_v01,
                design_name="versioned_sim",
                design_version="0.1",
                workload_name="default",
                fidelity=FidelityLevel.ANALYTICAL,
                fidelity_disclaimer="test",
                metrics=SimulationMetrics(),
                bottleneck=BottleneckReport(primary_bottleneck="balanced", bound_type="balanced"),
            )
            store.save_simulation(sim_v01)

            # Save v0.2 design and simulation
            design_v02 = DSDesignSpec(
                name="versioned_sim",
                version="0.2",
                parent_version="0.1",
                architecture="heterogeneous_accelerator",
                chiplets=[ChipletSpec(type="compute", peak_tops=200.0)],
            )
            design_id_v02 = store.save_design(design_v02)

            sim_v02 = SimulationResult(
                run_id="sim_02",
                design_id=design_id_v02,
                design_name="versioned_sim",
                design_version="0.2",
                workload_name="default",
                fidelity=FidelityLevel.ANALYTICAL,
                fidelity_disclaimer="test v2",
                metrics=SimulationMetrics(),
                bottleneck=BottleneckReport(primary_bottleneck="balanced", bound_type="balanced"),
            )
            store.save_simulation(sim_v02)

            # List simulations
            simulations = store.list_simulations()
            assert len(simulations) == 2

            # Get design history includes simulations
            history = store.get_design_history("versioned_sim")
            assert len(history) == 2
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestExperimentStoreSweepPersistence:
    """Test ExperimentStore SQLite persistence for sweeps."""

    def test_store_and_retrieve_sweep(self):
        """Test saving and retrieving a sweep result."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            store = ExperimentStore(db_path=db_path)

            # Create a design
            design = DSDesignSpec(
                name="sweep_test",
                version="0.1",
                architecture="heterogeneous_accelerator",
                chiplets=[ChipletSpec(type="compute", architecture="SIMD", peak_tops=100.0)],
            )
            design_id = store.save_design(design)

            # Create a sweep result
            from chiplab.schemas.simulation import SimulationMetrics, BottleneckReport

            sweep_result = SweepResult(
                sweep_id="sweep_test_1",
                design_name="sweep_test",
                design_version="0.1",
                fidelity=FidelityLevel.ANALYTICAL,
                fidelity_disclaimer="test sweep",
                swept_parameters=["chiplets[0].compute.peak_tops"],
                points=[],
                best_point=None,
                objective="throughput_tokens_per_sec",
            )

            sweep_id = store.save_sweep(sweep_result, design_id=design_id)
            assert sweep_id is not None

            # Retrieve the sweep
            retrieved = store.get_sweep(sweep_id="sweep_test_1")
            assert retrieved.sweep_id == "sweep_test_1"
            assert retrieved.design_name == "sweep_test"
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestExperimentStoreComparisonPersistence:
    """Test ExperimentStore SQLite persistence for comparisons."""

    def test_store_and_retrieve_comparison(self):
        """Test saving and retrieving a comparison result."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            store = ExperimentStore(db_path=db_path)

            # Create comparison result
            comparison = ComparisonResult(
                comparison_id="comp_test_1",
                fidelity=FidelityLevel.ANALYTICAL,
                fidelity_disclaimer="test comparison",
                designs=[
                    {"design_name": "design_a", "design_version": "0.1"},
                    {"design_name": "design_b", "design_version": "0.2"},
                ],
                metric_deltas={"throughput_tokens_per_sec": {"design_a": 500.0, "design_b": 800.0}},
                winner_by_metric={"throughput_tokens_per_sec": "design_b"},
                summary="design_b has higher throughput",
            )

            comp_id = store.save_comparison(comparison)
            assert comp_id is not None

            # Verify by listing designs
            designs = store.list_designs()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestReporterMarkdownExport:
    """Test reporter.py Markdown exports."""

    def test_export_simulation_markdown(self):
        """Test exporting simulation result as markdown."""
        from chiplab.results.reporter import export_simulation_report

        metrics = SimulationMetrics(
            total_memory_capacity_gb=8.0,
            aggregate_memory_bandwidth_tbps=64.0,
            peak_compute_tops=100.0,
            effective_compute_tops=80.0,
            interconnect_bisection_bw_tbps=12.5,
            estimated_latency_ms=5.5,
            throughput_tokens_per_sec=1000.0,
            throughput_ops_per_sec=80e12,
            total_area_mm2=500.0,
            total_power_watts=25.0,
            estimated_temp_c=65.0,
            estimated_cost_usd=10000.0,
            arithmetic_intensity=25.0,
            ridge_point_ops_per_byte=8000.0,
            utilization_pct=80.0,
            energy_per_inference_mj=2.5,
        )

        bottleneck = BottleneckReport(
            primary_bottleneck="memory_bandwidth",
            bound_type="memory_bandwidth",
            severity_score=75.0,
            utilization_breakdown={},
            details=["Arithmetic intensity below ridge point"],
            recommendations=["Increase memory bandwidth"],
        )

        result = SimulationResult(
            run_id="test_markdown",
            design_id="test_design",
            design_name="test_design",
            design_version="0.1",
            workload_name="default",
            fidelity=FidelityLevel.ANALYTICAL,
            fidelity_disclaimer="Level 1 analytical model: theoretical roofline and first-order estimates; not cycle-accurate.",
            timestamp=1234567890.0,
            metrics=metrics.model_dump(),
            bottleneck=bottleneck.model_dump(),
            assumptions=[],
            warnings=[],
            raw_details={},
            success=True,
        )

        # Test markdown export without output_path
        markdown_text = export_simulation_report(result, fmt="markdown")
        assert "Fidelity: Analytical" in markdown_text or "Fidelity: Analytical" in markdown_text
        assert "Total Memory Capacity" in markdown_text
        assert "Bottleneck" in markdown_text

    def test_export_simulation_json(self):
        """Test exporting simulation result as JSON."""
        from chiplab.results.reporter import export_simulation_report

        metrics = SimulationMetrics(
            total_memory_capacity_gb=8.0,
            aggregate_memory_bandwidth_tbps=64.0,
            peak_compute_tops=100.0,
            effective_compute_tops=80.0,
            interconnect_bisection_bw_tbps=12.5,
            estimated_latency_ms=5.5,
            throughput_tokens_per_sec=1000.0,
            throughput_ops_per_sec=80e12,
            total_area_mm2=500.0,
            total_power_watts=25.0,
            estimated_temp_c=65.0,
            estimated_cost_usd=10000.0,
            arithmetic_intensity=25.0,
            ridge_point_ops_per_byte=8000.0,
            utilization_pct=80.0,
            energy_per_inference_mj=2.5,
        )

        bottleneck = BottleneckReport(
            primary_bottleneck="balanced",
            bound_type="balanced",
            severity_score=30.0,
            utilization_breakdown={},
            details=[],
            recommendations=[],
        )

        result = SimulationResult(
            run_id="test_json",
            design_id="test_design",
            design_name="test_design",
            design_version="0.1",
            workload_name="default",
            fidelity=FidelityLevel.ANALYTICAL,
            fidelity_disclaimer="test disclaimer",
            timestamp=1234567890.0,
            metrics=metrics.model_dump(),
            bottleneck=bottleneck.model_dump(),
            assumptions=[],
            warnings=[],
            raw_details={},
            success=True,
        )

        # Test JSON export without output_path
        json_text = export_simulation_report(result, fmt="json")
        data = json.loads(json_text)
        assert data["fidelity"] == "analytical"
        assert data["fidelity_disclaimer"] == "test disclaimer"

    def test_export_simulation_csv(self):
        """Test exporting simulation result as CSV."""
        from chiplab.results.reporter import export_simulation_report

        metrics = SimulationMetrics(
            total_memory_capacity_gb=8.0,
            aggregate_memory_bandwidth_tbps=64.0,
            peak_compute_tops=100.0,
            effective_compute_tops=80.0,
            interconnect_bisection_bw_tbps=12.5,
            estimated_latency_ms=5.5,
            throughput_tokens_per_sec=1000.0,
            throughput_ops_per_sec=80e12,
            total_area_mm2=500.0,
            total_power_watts=25.0,
            estimated_temp_c=65.0,
            estimated_cost_usd=10000.0,
            arithmetic_intensity=25.0,
            ridge_point_ops_per_byte=8000.0,
            utilization_pct=80.0,
            energy_per_inference_mj=2.5,
        )

        bottleneck = BottleneckReport(
            primary_bottleneck="balanced",
            bound_type="balanced",
            severity_score=30.0,
            utilization_breakdown={},
            details=[],
            recommendations=[],
        )

        result = SimulationResult(
            run_id="test_csv",
            design_id="test_design",
            design_name="test_design",
            design_version="0.1",
            workload_name="default",
            fidelity=FidelityLevel.ANALYTICAL,
            fidelity_disclaimer="test",
            timestamp=1234567890.0,
            metrics=metrics.model_dump(),
            bottleneck=bottleneck.model_dump(),
            assumptions=[],
            warnings=[],
            raw_details={},
            success=True,
        )

        # Test CSV export without output_path
        csv_text = export_simulation_report(result, fmt="csv")
        reader = csv.reader(csv_text.splitlines())
        rows = list(reader)
        assert len(rows) > 0  # header + data rows


class TestReporterComparisonExport:
    """Test reporter.py comparison exports."""

    def test_export_comparison_markdown(self):
        """Test exporting comparison as markdown."""
        from chiplab.results.reporter import export_comparison_report

        comparison = ComparisonResult(
            comparison_id="comp_markdown_test",
            fidelity=FidelityLevel.ANALYTICAL,
            fidelity_disclaimer="test",
            designs=[
                {"design_name": "design_a", "design_version": "0.1"},
                {"design_name": "design_b", "design_version": "0.2"},
            ],
            metric_deltas={"throughput_tokens_per_sec": {"design_a": 500.0, "design_b": 800.0}},
            winner_by_metric={"throughput_tokens_per_sec": "design_b"},
            summary="design_b wins on throughput",
        )

        markdown_text = export_comparison_report(comparison, fmt="markdown")
        assert "Comparison Report" in markdown_text
        assert "design_a" in markdown_text
        assert "design_b" in markdown_text

    def test_export_comparison_json(self):
        """Test exporting comparison as JSON."""
        from chiplab.results.reporter import export_comparison_report

        comparison = ComparisonResult(
            comparison_id="comp_json_test",
            fidelity=FidelityLevel.DISCRETE_EVENT,
            fidelity_disclaimer="test disclaimer",
            designs=[
                {"design_name": "design_a", "design_version": "0.1"},
            ],
            metric_deltas={},
            winner_by_metric={},
            summary="",
        )

        json_text = export_comparison_report(comparison, fmt="json")
        data = json.loads(json_text)
        assert data["fidelity"] == "discrete_event"

    def test_export_comparison_csv(self):
        """Test exporting comparison as CSV."""
        from chiplab.results.reporter import export_comparison_report

        comparison = ComparisonResult(
            comparison_id="comp_csv_test",
            fidelity=FidelityLevel.ANALYTICAL,
            fidelity_disclaimer="test",
            designs=[
                {"design_name": "design_a", "design_version": "0.1"},
                {"design_name": "design_b", "design_version": "0.2"},
            ],
            metric_deltas={"throughput_tokens_per_sec": {"design_a": 500.0, "design_b": 800.0}},
            winner_by_metric={"throughput_tokens_per_sec": "design_b"},
            summary="design_b wins",
        )

        csv_text = export_comparison_report(comparison, fmt="csv")
        reader = csv.reader(csv_text.splitlines())
        rows = list(reader)
        assert len(rows) > 1  # header + data rows


class TestReporterSweepExport:
    """Test reporter.py sweep exports."""

    def test_export_sweep_markdown(self):
        """Test exporting sweep as markdown."""
        from chiplab.results.reporter import export_sweep_report

        # Create a simple sweep object
        class _Sweep:
            sweep_id = "sweep_markdown_test"
            design_name = "sweep_test"
            design_version = "0.1"
            fidelity = FidelityLevel.ANALYTICAL
            fidelity_disclaimer = "test sweep"
            swept_parameters = ["chiplets[0].compute.peak_tops"]
            points = []
            best_point = None
            objective = "throughput_tokens_per_sec"

        sweep = _Sweep()
        markdown_text = export_sweep_report(sweep, fmt="markdown")
        assert "Sweep ID:" in markdown_text or "Sweep ID" in markdown_text

    def test_export_sweep_json(self):
        """Test exporting sweep as JSON."""
        from chiplab.results.reporter import export_sweep_report

        class _Sweep:
            sweep_id = "sweep_json_test"
            design_name = "sweep_test"
            design_version = "0.1"
            fidelity = FidelityLevel.DISCRETE_EVENT
            fidelity_disclaimer = "test"
            swept_parameters = ["chiplets[0].compute.peak_tops"]
            points = []
            best_point = None
            objective = "throughput_tokens_per_sec"

        sweep = _Sweep()
        json_text = export_sweep_report(sweep, fmt="json")
        data = json.loads(json_text)
        assert data["fidelity"] == "discrete_event"

    def test_export_sweep_csv(self):
        """Test exporting sweep as CSV."""
        from chiplab.results.reporter import export_sweep_report

        class _Sweep:
            sweep_id = "sweep_csv_test"
            design_name = "sweep_test"
            design_version = "0.1"
            fidelity = FidelityLevel.ANALYTICAL
            fidelity_disclaimer = "test"
            swept_parameters = ["chiplets[0].compute.peak_tops"]
            points = []
            best_point = None
            objective = "throughput_tokens_per_sec"

        sweep = _Sweep()
        csv_text = export_sweep_report(sweep, fmt="csv")
        reader = csv.reader(csv_text.splitlines())
        rows = list(reader)
        assert len(rows) > 0
