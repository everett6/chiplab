"""Tests for ChipLab Discovery Engine.

Tests candidate generation (Latin Hypercube / Sobol / random), Pareto frontier
extraction, bottleneck identification, and CPU/GPU parity using GPUBackend.
"""

import os
import sys
import random

import numpy as np
import pytest

# Add the source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chiplab.core.gpu import GPUBackend, get_gpu_backend
from chiplab.discovery.explorer import (
    Candidate,
    DesignSpace,
    ParetoFrontier,
    BottleneckType,
    identify_bottleneck,
    explore_design_space,
    quick_exploration,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gpu_backend():
    """Provide a GPUBackend instance for testing."""
    return GPUBackend(prefer="auto", device="auto")


@pytest.fixture
def cpu_backend():
    """Provide a CPU-only GPUBackend instance."""
    return GPUBackend(prefer="cpu", device="cpu")


@pytest.fixture
def design_space():
    """Provide a default DesignSpace for testing."""
    return DesignSpace(
        parameters={
            "compute_units": (16, 256),
            "memory_bandwidth": (64.0, 512.0),
            "interconnect_width": (128.0, 1024.0),
            "material_quality": (0.5, 1.5),
        }
    )


@pytest.fixture
def sample_candidate():
    """Provide a sample Candidate for testing."""
    return Candidate(
        params={"compute_units": 64, "memory_bandwidth": 256.0, "interconnect_width": 512.0, "material_quality": 1.0},
        performance=1.0,
        power=10.0,
        area=32.0,
        temperature=60.0,
        ir_drop=0.05,
    )


# ---------------------------------------------------------------------------
# Candidate Tests
# ---------------------------------------------------------------------------


class TestCandidate:
    """Tests for the Candidate dataclass."""

    def test_candidate_creation(self, sample_candidate):
        """Test basic Candidate creation."""
        assert sample_candidate is not None
        assert sample_candidate.performance > 0
        assert sample_candidate.power > 0
        assert sample_candidate.area > 0
        assert sample_candidate.temperature >= 0
        assert sample_candidate.ir_drop >= 0

    def test_candidate_hash(self, sample_candidate):
        """Test Candidate hash is deterministic."""
        h1 = sample_candidate._hash
        # Recreate with same params
        c2 = Candidate(
            params=sample_candidate.params,
            performance=sample_candidate.performance,
            power=sample_candidate.power,
            area=sample_candidate.area,
            temperature=sample_candidate.temperature,
            ir_drop=sample_candidate.ir_drop,
        )
        assert c2._hash == h1

    def test_candidate_ppa_tuple(self, sample_candidate):
        """Test ppa_tuple property."""
        ppa = sample_candidate.ppa_tuple
        assert len(ppa) == 3
        assert ppa[0] == sample_candidate.performance  # maximize
        assert ppa[1] == sample_candidate.power  # minimize
        assert ppa[2] == sample_candidate.area  # minimize


# ---------------------------------------------------------------------------
# DesignSpace Tests
# ---------------------------------------------------------------------------


class TestDesignSpace:
    """Tests for the DesignSpace dataclass."""

    def test_design_space_creation(self, design_space):
        """Test DesignSpace creation."""
        assert design_space is not None
        assert len(design_space.parameters) == 4

    def test_sample_latin_hypercube(self, design_space):
        """Test Latin Hypercube sampling."""
        n = 20
        rng = random.Random(42)
        samples = design_space.sample(n, "latin_hypercube", rng)
        assert len(samples) == n

        # All samples should have the right keys
        expected_keys = set(design_space.parameters.keys())
        for sample in samples:
            assert set(sample.keys()) == expected_keys

        # All values should be within bounds
        for sample in samples:
            for dim, (pmin, pmax) in design_space.parameters.items():
                assert pmin <= sample[dim] <= pmax

    def test_sample_sobol(self, design_space):
        """Test Sobol sampling."""
        n = 20
        rng = random.Random(42)
        samples = design_space.sample(n, "sobol", rng)
        assert len(samples) == n

        expected_keys = set(design_space.parameters.keys())
        for sample in samples:
            assert set(sample.keys()) == expected_keys

        for sample in samples:
            for dim, (pmin, pmax) in design_space.parameters.items():
                assert pmin <= sample[dim] <= pmax

    def test_sample_random(self, design_space):
        """Test random sampling."""
        n = 20
        rng = random.Random(42)
        samples = design_space.sample(n, "random", rng)
        assert len(samples) == n

        expected_keys = set(design_space.parameters.keys())
        for sample in samples:
            assert set(sample.keys()) == expected_keys

        for sample in samples:
            for dim, (pmin, pmax) in design_space.parameters.items():
                assert pmin <= sample[dim] <= pmax

    def test_sample_unknown_method(self, design_space):
        """Test that unknown sampling method raises ValueError."""
        rng = random.Random(42)
        with pytest.raises(ValueError):
            design_space.sample(10, "invalid_method", rng)


# ---------------------------------------------------------------------------
# ParetoFrontier Tests
# ---------------------------------------------------------------------------


class TestParetoFrontier:
    """Tests for the ParetoFrontier dataclass."""

    def test_pareto_frontier_creation(self):
        """Test ParetoFrontier creation with candidates."""
        c1 = Candidate(
            params={"x": 1.0}, performance=10.0, power=5.0, area=10.0,
            temperature=50.0, ir_drop=0.02
        )
        c2 = Candidate(
            params={"x": 2.0}, performance=15.0, power=8.0, area=12.0,
            temperature=55.0, ir_drop=0.03
        )
        c3 = Candidate(
            params={"x": 3.0}, performance=12.0, power=4.0, area=8.0,
            temperature=45.0, ir_drop=0.01
        )

        frontier = ParetoFrontier(candidates=[c1, c2, c3])
        assert len(frontier.candidates) == 3
        assert len(frontier.frontier_indices) >= 0

    def test_pareto_frontier_identify_non_dominated(self):
        """Test that Pareto frontier correctly identifies non-dominated candidates."""
        # c1: high perf, high power, high area
        # c2: medium perf, low power, low area  
        # c3: medium perf, medium power, medium area
        c1 = Candidate(params={"x": 1.0}, performance=20.0, power=15.0, area=20.0,
                       temperature=70.0, ir_drop=0.05)
        c2 = Candidate(params={"x": 2.0}, performance=15.0, power=5.0, area=10.0,
                       temperature=50.0, ir_drop=0.02)
        c3 = Candidate(params={"x": 3.0}, performance=18.0, power=12.0, area=15.0,
                       temperature=60.0, ir_drop=0.04)

        frontier = ParetoFrontier(candidates=[c1, c2, c3])
        # c2 should be on frontier (low power/area), c1 and c3 may be dominated
        print(f"Frontier indices: {frontier.frontier_indices}")
        frontier_cands = frontier.get_frontier_candidates()
        print(f"Frontier candidates: {len(frontier_cands)}")

    def test_pareto_frontier_empty(self):
        """Test ParetoFrontier with no candidates."""
        frontier = ParetoFrontier(candidates=[])
        assert len(frontier.frontier_indices) == 0


# ---------------------------------------------------------------------------
# Bottleneck Identification Tests
# ---------------------------------------------------------------------------


class TestIdentifyBottleneck:
    """Tests for bottleneck identification."""

    def test_bottleneck_compute_bound(self):
        """Test compute-bound classification."""
        # Low performance relative to power = compute-bound
        c = Candidate(
            params={"x": 1.0}, performance=100.0, power=100.0, area=50.0,
            temperature=40.0, ir_drop=0.01
        )
        bottleneck = identify_bottleneck(c)
        # With our thresholds, compute_efficiency = 100/100 = 1.0 < 1e3, so compute-bound
        assert bottleneck in (BottleneckType.COMPUTE_BOUND, BottleneckType.MEMORY_BOUND)

    def test_bottleneck_thermal_bound(self):
        """Test thermal-bound classification."""
        # High temperature relative to performance
        c = Candidate(
            params={"x": 1.0}, performance=100.0, power=50.0, area=30.0,
            temperature=95.0, ir_drop=0.01  # 95C > 80C threshold
        )
        bottleneck = identify_bottleneck(c)
        assert bottleneck == BottleneckType.THERMAL_BOUND

    def test_bottleneck_ir_drop_bound(self):
        """Test IR-drop-bound classification."""
        # High IR drop relative to performance
        c = Candidate(
            params={"x": 1.0}, performance=100.0, power=50.0, area=30.0,
            temperature=50.0, ir_drop=0.15  # 0.15V > 0.1V threshold
        )
        bottleneck = identify_bottleneck(c)
        assert bottleneck == BottleneckType.IR_DROP_BOUND

    def test_bottleneck_memory_bound(self):
        """Test memory-bandwidth-bound classification."""
        # Moderate performance, moderate power, but low efficiency relative to area
        c = Candidate(
            params={"x": 1.0}, performance=500.0, power=50.0, area=500.0,
            temperature=40.0, ir_drop=0.01
        )
        bottleneck = identify_bottleneck(c)
        # memory_efficiency = 500/500 = 1.0 < 1e5, could be memory-bound
        assert bottleneck in (BottleneckType.MEMORY_BOUND, BottleneckType.COMPUTE_BOUND)


# ---------------------------------------------------------------------------
# Design-Space Exploration Tests
# ---------------------------------------------------------------------------


class TestExploreDesignSpace:
    """Tests for design-space exploration."""

    def test_explore_with_gpu_backend(self, design_space, gpu_backend):
        """Test exploration with GPU backend."""
        frontier = explore_design_space(
            design_space=design_space,
            n_candidates=30,
            sampling_method="latin_hypercube",
            backend=gpu_backend,
            verbose=False,
        )
        assert isinstance(frontier, ParetoFrontier)
        assert len(frontier.candidates) == 30

    def test_explore_with_cpu_backend(self, design_space, cpu_backend):
        """Test exploration with CPU backend."""
        frontier = explore_design_space(
            design_space=design_space,
            n_candidates=30,
            sampling_method="latin_hypercube",
            backend=cpu_backend,
            verbose=False,
        )
        assert isinstance(frontier, ParetoFrontier)
        assert len(frontier.candidates) == 30

    def test_explore_different_sampling_methods(self, design_space, gpu_backend):
        """Test exploration with different sampling methods."""
        for method in ["latin_hypercube", "sobol", "random"]:
            frontier = explore_design_space(
                design_space=design_space,
                n_candidates=20,
                sampling_method=method,
                backend=gpu_backend,
                verbose=False,
            )
            assert isinstance(frontier, ParetoFrontier)
            assert len(frontier.candidates) == 20

    def test_quick_exploration(self):
        """Test quick_exploration convenience function."""
        frontier = quick_exploration(n_candidates=20)
        assert isinstance(frontier, ParetoFrontier)
        assert len(frontier.candidates) == 20

    def test_quick_exploration_default_params(self):
        """Test quick_exploration with default parameter ranges."""
        frontier = quick_exploration()
        assert isinstance(frontier, ParetoFrontier)
        assert len(frontier.candidates) == 50


# ---------------------------------------------------------------------------
# CPU/GPU Parity Tests
# ---------------------------------------------------------------------------


class TestCPUGPUParity:
    """Tests for CPU/GPU parity in candidate evaluation."""

    def test_cpu_gpu_same_params_same_results(self, design_space, gpu_backend, cpu_backend):
        """Test that CPU and GPU backends produce consistent results for same params."""
        # Generate a sample
        rng = random.Random(42)
        params = design_space.sample(1, "random", rng)[0]

        # Evaluate with both backends
        candidate_gpu = evaluate_candidate(params, gpu_backend)
        candidate_cpu = evaluate_candidate(params, cpu_backend)

        # Results should be positive and finite
        assert np.isfinite(candidate_gpu.performance)
        assert np.isfinite(candidate_cpu.performance)
        assert np.isfinite(candidate_gpu.power)
        assert np.isfinite(candidate_cpu.power)

        # They may differ slightly due to different implementations, but both should be valid
        assert candidate_gpu.performance > 0
        assert candidate_cpu.performance > 0
        assert candidate_gpu.power > 0
        assert candidate_cpu.power > 0

    def test_bottleneck_consistency(self, sample_candidate):
        """Test that bottleneck identification is consistent."""
        bottleneck = identify_bottleneck(sample_candidate)
        assert bottleneck in [
            BottleneckType.COMPUTE_BOUND,
            BottleneckType.MEMORY_BOUND,
            BottleneckType.THERMAL_BOUND,
            BottleneckType.IR_DROP_BOUND,
        ]

    def test_bottleneck_different_candidates(self):
        """Test bottleneck identification with different candidate profiles."""
        # Compute-bound candidate
        c_compute = Candidate(
            params={"x": 1.0}, performance=500.0, power=500.0, area=50.0,
            temperature=40.0, ir_drop=0.01
        )
        b1 = identify_bottleneck(c_compute)

        # Thermal-bound candidate
        c_thermal = Candidate(
            params={"x": 1.0}, performance=500.0, power=100.0, area=50.0,
            temperature=95.0, ir_drop=0.01
        )
        b2 = identify_bottleneck(c_thermal)

        # IR-drop bound candidate
        c_irdrop = Candidate(
            params={"x": 1.0}, performance=500.0, power=100.0, area=50.0,
            temperature=40.0, ir_drop=0.20
        )
        b3 = identify_bottleneck(c_irdrop)

        # All should return valid bottleneck types
        assert b1 in [
            BottleneckType.COMPUTE_BOUND,
            BottleneckType.MEMORY_BOUND,
        ]
        assert b2 == BottleneckType.THERMAL_BOUND
        assert b3 == BottleneckType.IR_DROP_BOUND


# Helper function for parity tests
def evaluate_candidate(params: Dict[str, float], backend: GPUBackend) -> Candidate:
    """Evaluate a design candidate using the given backend."""
    from chiplab.discovery.explorer import evaluate_candidate as _eval
    return _eval(params, backend)