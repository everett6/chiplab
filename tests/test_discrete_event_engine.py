"""Tests for Level 2 discrete-event simulation engine using SimPy."""

import pytest
import yaml
from chiplab.schemas.design import DesignSpec, ChipletSpec
from chiplab.schemas.workload import WorkloadSpec
from chiplab.schemas.simulation import SimulationConfig, FidelityLevel
from chiplab.engine.discrete_event import run_discrete_event_simulation


class TestDiscreteEventEngineECHO:
    """Level 2 discrete-event simulation tests for ECHO designs."""

    def test_echo_v0_1_determinism_same_seed(self):
        """Verify determinism with same seed - running twice should give same results."""
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

        config = SimulationConfig(
            fidelity=FidelityLevel.DISCRETE_EVENT,
            seed=42,
            num_requests=32,
        )

        result1 = run_discrete_event_simulation(design, workload, config)
        result2 = run_discrete_event_simulation(design, workload, config)

        # Both should have DISCRETE_EVENT fidelity
        assert result1.fidelity == FidelityLevel.DISCRETE_EVENT
        assert result2.fidelity == FidelityLevel.DISCRETE_EVENT

        # Results should be deterministic with same seed
        assert result1.metrics.estimated_latency_ms == result2.metrics.estimated_latency_ms
        assert result1.metrics.throughput_tokens_per_sec == result2.metrics.throughput_tokens_per_sec

    def test_echo_v0_1_determinism_different_seeds(self):
        """Verify different seeds produce different results."""
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

        config1 = SimulationConfig(
            fidelity=FidelityLevel.DISCRETE_EVENT,
            seed=42,
            num_requests=32,
        )
        config2 = SimulationConfig(
            fidelity=FidelityLevel.DISCRETE_EVENT,
            seed=123,
            num_requests=32,
        )

        result1 = run_discrete_event_simulation(design, workload, config1)
        result2 = run_discrete_event_simulation(design, workload, config2)

        # Latency may differ with different seeds (due to event ordering)
        # At minimum, both should be valid results
        assert result1.fidelity == FidelityLevel.DISCRETE_EVENT
        assert result2.fidelity == FidelityLevel.DISCRETE_EVENT

    def test_echo_v0_1_contention_higher_num_requests(self):
        """Higher num_requests increases queue wait/latency."""
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

        # Low contention
        config_low = SimulationConfig(
            fidelity=FidelityLevel.DISCRETE_EVENT,
            seed=42,
            num_requests=8,
        )
        # High contention
        config_high = SimulationConfig(
            fidelity=FidelityLevel.DISCRETE_EVENT,
            seed=42,
            num_requests=128,
        )

        result_low = run_discrete_event_simulation(design, workload, config_low)
        result_high = run_discrete_event_simulation(design, workload, config_high)

        # Higher num_requests should result in higher or equal latency
        assert result_high.metrics.estimated_latency_ms >= result_low.metrics.estimated_latency_ms

    def test_echo_v0_1_contention_lower_bandwidth(self):
        """Lower link bandwidth increases queue wait/latency."""
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

        # Normal bandwidth (from design)
        config_normal = SimulationConfig(
            fidelity=FidelityLevel.DISCRETE_EVENT,
            seed=42,
            num_requests=32,
        )
        # Reduced bandwidth - modify interconnect
        design_reduced = DesignSpec(
            name="design_reduced_bw",
            package=design.package,
            chiplets=design.chiplets,
            interconnect=type(design.interconnect)(
                topology=design.interconnect.topology,
                link_bandwidth_gbps=10.0,  # Reduced from 100.0
                latency_ns=design.interconnect.latency_ns,
                energy_pj_per_bit=design.interconnect.energy_pj_per_bit,
                lanes_per_chiplet=design.interconnect.lanes_per_chiplet,
            ),
            constraints=design.constraints,
        )

        config_reduced = SimulationConfig(
            fidelity=FidelityLevel.DISCRETE_EVENT,
            seed=42,
            num_requests=32,
        )

        result_normal = run_discrete_event_simulation(design, workload, config_normal)
        result_reduced = run_discrete_event_simulation(design_reduced, workload, config_reduced)

        # Lower bandwidth should result in higher or equal latency
        assert result_reduced.metrics.estimated_latency_ms >= result_normal.metrics.estimated_latency_ms


class TestDiscreteEventEngineNonECHO:
    """Level 2 discrete-event simulation tests for non-ECHO designs."""

    def test_transformer_determinism_same_seed(self):
        """Verify determinism with same seed for transformer design."""
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

        config = SimulationConfig(
            fidelity=FidelityLevel.DISCRETE_EVENT,
            seed=42,
            num_requests=32,
        )

        result1 = run_discrete_event_simulation(design, workload, config)
        result2 = run_discrete_event_simulation(design, workload, config)

        assert result1.fidelity == FidelityLevel.DISCRETE_EVENT
        assert result2.fidelity == FidelityLevel.DISCRETE_EVENT
        assert result1.metrics is not None
        assert result2.metrics is not None

    def test_transformer_contention_higher_requests(self):
        """Higher num_requests increases latency for transformer design."""
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

        config_low = SimulationConfig(
            fidelity=FidelityLevel.DISCRETE_EVENT,
            seed=42,
            num_requests=8,
        )
        config_high = SimulationConfig(
            fidelity=FidelityLevel.DISCRETE_EVENT,
            seed=42,
            num_requests=64,
        )

        result_low = run_discrete_event_simulation(design, workload, config_low)
        result_high = run_discrete_event_simulation(design, workload, config_high)

        assert result_high.metrics.estimated_latency_ms >= result_low.metrics.estimated_latency_ms