"""Tests for Level 1 analytical engine simulation."""

import pytest
import yaml
from chiplab.schemas.design import DesignSpec, ChipletSpec
from chiplab.schemas.workload import WorkloadSpec
from chiplab.schemas.simulation import SimulationConfig, FidelityLevel
from chiplab.engine.analytical import run_analytical_simulation
from chiplab.engine.validator import validate_design


class TestAnalyticalEngineECHO:
    """Level 1 analytical simulation tests for ECHO designs."""

    def test_echo_v0_1_analytical_simulation(self):
        """Run analytical simulation on echo_v0_1 design, verify fidelity level."""
        with open("/home/everett/Chip Lab/projects/echo/echo_v0_1.yaml") as f:
            data = yaml.safe_load(f)
        design = DesignSpec.from_yaml(yaml.dump(data))

        workload = WorkloadSpec(
            name="moe_decode_workload",
            type="moe_decode",
            batch_size=1,
            sequence_length=128,
            hidden_dim=4096,
            num_layers=32,
            num_experts=8,
            active_experts=2,
            precision="INT8",
            flops=1.5e15,
            bytes_read=64e9,
            bytes_write=16e9,
        )

        config = SimulationConfig(fidelity=FidelityLevel.ANALYTICAL)
        result = run_analytical_simulation(design, workload, config, design_id="echo_v0_1")

        # Verify fidelity level
        assert result.fidelity == FidelityLevel.ANALYTICAL
        # Verify fidelity disclaimer is set
        assert result.fidelity_disclaimer != ""
        # Verify assumptions were recorded
        assert len(result.assumptions) > 0
        # Verify bottleneck detection
        assert result.bottleneck is not None
        assert result.bottleneck.primary_bottleneck is not None
        # Verify metrics are populated
        assert result.metrics is not None
        assert result.metrics.total_memory_capacity_gb > 0

    def test_echo_v0_2_analytical_simulation(self):
        """Run analytical simulation on echo_v0_2 design."""
        with open("/home/everett/Chip Lab/projects/echo/echo_v0_2.yaml") as f:
            data = yaml.safe_load(f)
        design = DesignSpec.from_yaml(yaml.dump(data))

        workload = WorkloadSpec(
            name="moe_decode_workload",
            type="moe_decode",
            batch_size=1,
            sequence_length=128,
            hidden_dim=4096,
            num_layers=32,
            num_experts=8,
            active_experts=2,
            precision="INT8",
            flops=1.5e15,
            bytes_read=64e9,
            bytes_write=16e9,
        )

        config = SimulationConfig(fidelity=FidelityLevel.ANALYTICAL)
        result = run_analytical_simulation(design, workload, config, design_id="echo_v0_2")

        assert result.fidelity == FidelityLevel.ANALYTICAL
        assert result.fidelity_disclaimer != ""
        assert len(result.assumptions) > 0
        assert result.bottleneck is not None

    def test_echo_v0_1_with_defaults_false(self):
        """Analytical simulation with allow_estimated_defaults=False."""
        with open("/home/everett/Chip Lab/projects/echo/echo_v0_1.yaml") as f:
            data = yaml.safe_load(f)
        design = DesignSpec.from_yaml(yaml.dump(data))

        workload = WorkloadSpec(
            name="moe_decode_workload",
            type="moe_decode",
            batch_size=1,
            sequence_length=128,
            hidden_dim=4096,
            num_layers=32,
            num_experts=8,
            active_experts=2,
            precision="INT8",
        )

        config = SimulationConfig(fidelity=FidelityLevel.ANALYTICAL, allow_estimated_defaults=False)
        result = run_analytical_simulation(design, workload, config, design_id="echo_v0_1_defaults_false")

        # Should still pass since MemorySpec is provided
        assert result.fidelity == FidelityLevel.ANALYTICAL
        assert result.success == True


class TestAnalyticalEngineNonECHO:
    """Level 1 analytical simulation tests for non-ECHO (transformer) designs."""

    def test_transformer_accelerator_analytical_simulation(self):
        """Run analytical simulation on transformer_accelerator design."""
        with open("/home/everett/Chip Lab/examples/transformer_accelerator.yaml") as f:
            data = yaml.safe_load(f)
        design = DesignSpec.from_yaml(yaml.dump(data))

        workload = WorkloadSpec(
            name="transformer_inference_workload",
            type="transformer_decode",
            batch_size=1,
            sequence_length=512,
            hidden_dim=8192,
            num_layers=48,
            num_experts=16,
            active_experts=8,
            precision="BF16",
            flops=3.0e16,
            bytes_read=256e9,
            bytes_write=64e9,
        )

        config = SimulationConfig(fidelity=FidelityLevel.ANALYTICAL)
        result = run_analytical_simulation(design, workload, config, design_id="transformer_accelerator")

        assert result.fidelity == FidelityLevel.ANALYTICAL
        assert result.fidelity_disclaimer != ""
        assert len(result.assumptions) > 0
        assert result.bottleneck is not None
        assert result.metrics is not None

    def test_transformer_with_parameter_sweep(self):
        """Test parameter sweep on transformer design."""
        import chiplab.engine.sweep as sweep_module

        with open("/home/everett/Chip Lab/examples/transformer_accelerator.yaml") as f:
            data = yaml.safe_load(f)
        design = DesignSpec.from_yaml(yaml.dump(data))

        workload = WorkloadSpec(
            name="transformer_inference_workload",
            type="transformer_decode",
            batch_size=1,
            sequence_length=512,
            hidden_dim=8192,
            num_layers=48,
            num_experts=16,
            active_experts=8,
            precision="BF16",
        )

        config = SimulationConfig(fidelity=FidelityLevel.ANALYTICAL)

        # Sweep over peak_tops
        parameters = {
            "chiplets[0].compute.peak_tops": [200.0, 500.0, 1000.0],
        }

        result = sweep_module.run_parameter_sweep(design, workload, config, parameters)

        assert result is not None
        assert len(result.points) > 0
        assert result.best_point is not None
        assert result.best_point.metrics is not None

    def test_compare_results(self):
        """Test comparison of two analytical results."""
        import chiplab.engine.analyzer as analyzer_module

        with open("/home/everett/Chip Lab/projects/echo/echo_v0_1.yaml") as f:
            data = yaml.safe_load(f)
        design = DesignSpec.from_yaml(yaml.dump(data))

        workload = WorkloadSpec(
            name="moe_decode_workload",
            type="moe_decode",
            batch_size=1,
            sequence_length=128,
            hidden_dim=4096,
            num_layers=32,
            num_experts=8,
            active_experts=2,
            precision="INT8",
            flops=1.5e15,
            bytes_read=64e9,
            bytes_write=16e9,
        )

        config = SimulationConfig(fidelity=FidelityLevel.ANALYTICAL)

        # Run two simulations with different configs
        result1 = run_analytical_simulation(design, workload, config, design_id="run1")
        result2 = run_analytical_simulation(design, workload, config, design_id="run2")

        comparison = analyzer_module.compare_results([result1, result2])

        assert comparison is not None
        assert comparison.designs is not None
        assert comparison.winner_by_metric is not None
        assert comparison.summary != ""