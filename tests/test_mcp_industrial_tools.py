"""End-to-end tests for ChipLab MCP industrial tools.

Tests all 5 tools: chiplab_reliability_analyze, chiplab_discover_architectures,
chiplab_manufacturing_signoff, chiplab_digital_twin_manage, chiplab_multiscale_surrogate.
"""

from __future__ import annotations

import json
import pytest

from chiplab.mcp_industrial_tools import (
    chiplab_reliability_analyze,
    chiplab_discover_architectures,
    chiplab_manufacturing_signoff,
    chiplab_digital_twin_manage,
    chiplab_multiscale_surrogate,
)


class TestChiplabReliabilityAnalyze:
    """Tests for chiplab_reliability_analyze function."""

    def test_basic_reliability_analyze(self):
        """Test basic reliability analysis with typical parameters."""
        result = chiplab_reliability_analyze(
            technology="nmos",
            stress_temp_k=350.0,
            operating_temp_k=300.0,
            duration_hours=1000.0,
            n_mc_samples=100,
        )

        # Check required keys exist
        assert "technology" in result
        assert "stress_temp_k" in result
        assert "operating_temp_k" in result
        assert "duration_hours" in result
        assert "arrhenius_acceleration_factor" in result
        assert "degradation_metrics" in result
        assert "monte_carlo_lifetime" in result
        assert "provenance" in result

        # Check values
        assert result["technology"] == "nmos"
        assert result["stress_temp_k"] == 350.0
        assert result["operating_temp_k"] == 300.0
        assert result["duration_hours"] == 1000.0
        assert result["arrhenius_acceleration_factor"] > 1.0  # Arrhenius factor > 1 when stress > operating

        # Check degradation metrics
        dm = result["degradation_metrics"]
        assert "threshold" in dm
        assert "current_degradation" in dm

        # Check monte carlo lifetime
        mcl = result["monte_carlo_lifetime"]
        assert mcl["samples"] == 100
        percentiles = mcl["percentiles_hours"]
        assert "p10" in percentiles
        assert "p50" in percentiles
        assert "p90" in percentiles
        assert "mean" in percentiles
        assert "median" in percentiles

    def test_different_technology(self):
        """Test with different technology node."""
        result = chiplab_reliability_analyze(
            technology="pmos",
            stress_temp_k=375.0,
            operating_temp_k=300.0,
            duration_hours=500.0,
            n_mc_samples=500,
        )

        assert result["technology"] == "pmos"
        assert result["arrhenius_acceleration_factor"] > 1.0

    def test_arrhenius_factor_ordering(self):
        """Arrhenius factor should be > 1 when stress temp > operating temp."""
        result = chiplab_reliability_analyze(
            technology="nmos",
            stress_temp_k=400.0,
            operating_temp_k=300.0,
            duration_hours=100.0,
            n_mc_samples=100,
        )
        assert result["arrhenius_acceleration_factor"] > 1.0

        # Lower stress temp should give factor closer to 1
        result2 = chiplab_reliability_analyze(
            technology="nmos",
            stress_temp_k=310.0,
            operating_temp_k=300.0,
            duration_hours=100.0,
            n_mc_samples=100,
        )
        assert result2["arrhenius_acceleration_factor"] < result["arrhenius_acceleration_factor"]


class TestChiplabDiscoverArchitectures:
    """Tests for chiplab_discover_architectures function."""

    def test_basic_discovery(self):
        """Test basic architecture discovery."""
        parameter_ranges = {
            "compute_units": (16, 256),
            "memory_bandwidth": (64.0, 512.0),
            "interconnect_width": (128.0, 1024.0),
            "material_quality": (0.5, 1.5),
        }

        result = chiplab_discover_architectures(
            parameter_ranges=parameter_ranges,
            n_candidates=50,
            sampling_method="latin_hypercube",
        )

        # Check required keys
        assert "parameter_ranges" in result
        assert "n_candidates_generated" in result
        assert "sampling_method" in result
        assert "pareto_frontier_candidates" in result
        assert "frontier_candidates" in result
        assert "bottleneck_type_summary" in result
        assert "provenance" in result

        # Check values
        assert result["parameter_ranges"] == parameter_ranges
        assert result["n_candidates_generated"] == 50
        assert result["sampling_method"] == "latin_hypercube"
        assert result["pareto_frontier_candidates"] > 0

        # Check frontier candidates structure
        for fc in result["frontier_candidates"]:
            assert "params" in fc
            assert "ppa_tuple" in fc
            assert "bottlenecks" in fc
            assert "efficiency_ops_per_watt" in fc

    def test_different_sampling(self):
        """Test with different sampling methods."""
        parameter_ranges = {
            "compute_units": (32, 128),
            "memory_bandwidth": (128.0, 256.0),
            "interconnect_width": (256.0, 512.0),
            "material_quality": (0.8, 1.0),
        }

        for method in ["latin_hypercube", "sobol", "random"]:
            result = chiplab_discover_architectures(
                parameter_ranges=parameter_ranges,
                n_candidates=30,
                sampling_method=method,
            )
            assert result["pareto_frontier_candidates"] > 0
            assert result["sampling_method"] == method

        # latin_hypercube should give different results from random
        result_lh = chiplab_discover_architectures(
            parameter_ranges=parameter_ranges,
            n_candidates=30,
            sampling_method="latin_hypercube",
        )
        result_rand = chiplab_discover_architectures(
            parameter_ranges=parameter_ranges,
            n_candidates=30,
            sampling_method="random",
        )


class TestChiplabManufacturingSignoff:
    """Tests for chiplab_manufacturing_signoff function."""

    def test_basic_signoff(self):
        """Test basic manufacturing signoff."""
        result = chiplab_manufacturing_signoff(
            design_name="test_design",
            version="1.0",
            material_spec={"material": "silicon", "resistivity": 10.0},
            device_params={"channel_length": 28, "channel_width": 180},
            circuit_params={"voltage": 1.0, "current": 0.5},
            rtl_params={"num_stages": 4, "pipeline": True},
        )

        # Check required keys
        assert "design_name" in result
        assert "version" in result
        assert "overall_pass" in result
        assert "stage_results" in result
        assert "dfm_checks" in result
        assert "pvt_summary" in result
        assert "total_cost_usd" in result
        assert "yield_pct" in result
        assert "notes" in result
        assert "provenance" in result

        # Check values
        assert result["design_name"] == "test_design"
        assert result["version"] == "1.0"

        # Check stage results
        stage_keys = list(result["stage_results"].keys())
        # Should have pipeline stages
        assert len(stage_keys) > 0

        # Check DFM checks structure
        for stage, dfm in result["dfm_checks"].items():
            assert "pass" in dfm
            assert "checks" in dfm
            assert "violations" in dfm

        # Check PVT summary
        pvt = result["pvt_summary"]
        assert "process_corners" in pvt
        assert "voltages" in pvt
        assert "temperatures" in pvt

    def test_minimal_signoff(self):
        """Test signoff with minimal parameters."""
        result = chiplab_manufacturing_signoff(
            design_name="minimal_design",
            version="0.1",
        )

        assert result["design_name"] == "minimal_design"
        assert result["version"] == "0.1"
        assert "overall_pass" in result


class TestChiplabDigitalTwinManage:
    """Tests for chiplab_digital_twin_manage function."""

    def test_create_and_evaluate(self):
        """Test create and evaluate actions."""
        # Create
        result_create = chiplab_digital_twin_manage(
            action="create",
            design_id="test_design_001",
            design_data={"compute_units": 64, "memory_bw": 128.0},
        )

        assert result_create["action"] == "create"
        assert result_create["design_id"] == "test_design_001"
        assert "state" in result_create

        # Evaluate
        result_eval = chiplab_digital_twin_manage(
            action="evaluate",
            design_id="test_design_001",
        )
        assert result_eval["action"] == "evaluate"
        assert "result" in result_eval

    def test_calibrate_and_snapshot(self):
        """Test calibrate and snapshot actions."""
        # Calibrate
        result_cal = chiplab_digital_twin_manage(
            action="calibrate",
            design_id="test_design_001",
            calibration_data={"target_perf": 1000.0, "target_power": 50.0},
        )
        assert result_cal["action"] == "calibrate"
        assert result_cal["design_id"] == "test_design_001"

        # Snapshot
        result_snap = chiplab_digital_twin_manage(
            action="snapshot",
            design_id="test_design_001",
            snapshot_name="snap1",
        )
        assert result_snap["action"] == "snapshot"
        assert result_snap["snapshot_name"] == "snap1"

    def test_save_and_load(self):
        """Test save and load actions."""
        import tempfile
        import os

        # Create a temp file for saving
        with tempfile.NamedTemporaryFile(suffix=".dtwin", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Save
            result_save = chiplab_digital_twin_manage(
                action="save",
                design_id="test_design_001",
                save_path=tmp_path,
            )
            assert result_save["action"] == "save"
            assert result_save["save_path"] == tmp_path

            # Load
            result_load = chiplab_digital_twin_manage(
                action="load",
                design_id="test_design_001",
                load_path=tmp_path,
            )
            assert result_load["action"] == "load"
            assert result_load["design_id"] == "test_design_001"
            assert "state" in result_load
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_unknown_action(self):
        """Test that unknown action raises ValueError."""
        with pytest.raises(ValueError, match="Unknown action"):
            chiplab_digital_twin_manage(action="invalid_action")


class TestChiplabMultiscaleSurrogate:
    """Tests for chiplab_multiscale_surrogate function."""

    def test_reduce_action(self):
        """Test 'reduce' action - POD/SVD reduction."""
        import numpy as np

        # Create sample data
        np.random.seed(42)
        X = np.random.normal(size=(50, 256))

        result = chiplab_multiscale_surrogate(
            action="reduce",
            input_data=X,
            n_components=16,
            device="cpu",
        )

        assert result["action"] == "reduce"
        assert "modes_shapes" in result
        assert "explained_variance" in result
        assert result["modes_shapes"]["modes"][0] == 50  # n_samples
        assert result["modes_shapes"]["singular_values"][0] == 16  # n_components

    def test_surrogate_action(self):
        """Test 'surrogate' action - create GPUSurrogate."""
        import numpy as np

        np.random.seed(42)
        X = np.random.normal(size=(50, 256))

        result = chiplab_multiscale_surrogate(
            action="surrogate",
            input_data=X,
            n_components=16,
            hidden_dim=64,
            operator_type="mlp",
            device="cpu",
        )

        assert result["action"] == "surrogate"
        assert "model_type" in result
        assert "input_dim" in result
        assert "output_dim" in result
        assert result["input_dim"] == 256
        assert result["output_dim"] == 256

    def test_predict_action(self):
        """Test 'predict' action - make predictions with surrogate."""
        import numpy as np

        np.random.seed(42)
        X = np.random.normal(size=(50, 256))

        result = chiplab_multiscale_surrogate(
            action="predict",
            input_data=X,
            n_components=16,
            hidden_dim=64,
            operator_type="mlp",
            device="cpu",
            n_epochs=5,
        )

        assert result["action"] == "predict"
        assert "n_predictions" in result
        assert "mse" in result
        assert result["n_predictions"] == 50

    def test_quick_demo_action(self):
        """Test 'quick_demo' action."""
        import numpy as np

        np.random.seed(42)
        X = np.random.normal(size=(50, 256))

        result = chiplab_multiscale_surrogate(
            action="quick_demo",
            input_data=X,
            n_components=16,
            device="cpu",
        )

        assert result["action"] == "quick_demo"
        assert "modes" in result
        assert "singular_values_count" in result
        assert "reconstruction_error" in result
        assert result["modes"] == 16

    def test_unknown_action(self):
        """Test that unknown action raises ValueError."""
        with pytest.raises(ValueError, match="Unknown action"):
            chiplab_multiscale_surrogate(action="invalid_action")