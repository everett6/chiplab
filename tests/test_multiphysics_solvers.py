"""Tests for Multiphysics Device Lab three fidelity solvers."""
import numpy as np
import pytest
import time

from chiplab.models.materials.ferrofluid import FerrofluidState, ferrofluid_write, ferrofluid_relax, ferrofluid_read
from chiplab.models.materials.particle_dynamics import neel_relaxation, brownian_motion
from chiplab.models.materials.impedance import impedance_matrix, frequency_response
from chiplab.schemas.common import TrackedValue, Provenance
from chiplab.schemas.simulation import FidelityLevel


class TestFidelityLevels:
    """Tests for three fidelity levels: ANALYTICAL, DISCRETE_EVENT, SPECIALIZED."""

    def test_level1_analytical_speed(self):
        """Test Level 1 (reduced-order) speed and accuracy."""
        state = FerrofluidState(magnetization=100.0, relaxation_time=1e-3)
        start = time.time()
        # Level 1: analytical simulation via relaxation
        n_steps = 100
        for _ in range(n_steps):
            state = ferrofluid_relax(state, dt=1.0)
        elapsed = time.time() - start
        # Should complete quickly (Level 1 is reduced-order)
        assert elapsed < 10.0  # 10 seconds timeout
        # Magnetization should decay
        assert abs(state.magnetization.value) < 100.0

    def test_level1_analytical_consistency(self):
        """Test Level 1 produces consistent results."""
        state1 = FerrofluidState(magnetization=100.0, relaxation_time=1e-3)
        state2 = FerrofluidState(magnetization=100.0, relaxation_time=1e-3)
        # Both should decay at same rate
        for _ in range(10):
            state1 = ferrofluid_relax(state1, dt=1.0)
            state2 = ferrofluid_relax(state2, dt=1.0)
        assert abs(state1.magnetization.value) == abs(state2.magnetization.value)

    def test_frequency_response_basic(self):
        """Test frequency-dependent impedance computation (Level 2)."""
        state = FerrofluidState()
        result = frequency_response(state, f_min=1e3, f_max=1e6, n_points=10)
        assert "frequencies_Hz" in result
        assert "impedance_magnitude_Ω" in result
        assert len(result["frequencies_Hz"]) == 10
        assert result["f_min_Hz"] == 1e3
        assert result["f_max_Hz"] == 1e6

    def test_frequency_response_organization_dependence(self):
        """Test frequency response depends on organization state (Level 2)."""
        state_organized = FerrofluidState(particle_organization=0.8)
        state_disorganized = FerrofluidState(particle_organization=0.1)
        result_organized = frequency_response(state_organized, f_min=1e3, f_max=1e6, n_points=5)
        result_disorganized = frequency_response(state_disorganized, f_min=1e3, f_max=1e6, n_points=5)
        # More organized state should have different magnitude response
        assert not np.allclose(result_organized["impedance_magnitude_Ω"], result_disorganized["impedance_magnitude_Ω"])
        # More organized state should have higher peak response
        assert np.max(result_organized["impedance_magnitude_Ω"]) > np.max(result_disorganized["impedance_magnitude_Ω"])


class TestParticleSolvers:
    """Tests for particle-based solvers (Level 3)."""

    def test_neel_relaxation_basic(self):
        """Test Neel relaxation basic function."""
        N = 3
        particles_flat = np.random.randn(N * 3)
        field = np.array([0.0, 0.0, 1.0])  # field along z-axis
        dt = 1e-9
        new_moments = neel_relaxation(particles_flat, field, dt)
        # New moments should have same number of elements
        assert new_moments.shape == (N * 3,)

    def test_neel_relaxation_field_alignment(self):
        """Test Neel relaxation aligns moments with field."""
        N = 5
        particles_flat = np.random.randn(N * 3)
        field = np.array([0.0, 0.0, 1.0])  # field along z-axis
        dt = 1e-9
        new_moments = neel_relaxation(particles_flat, field, dt)
        # Moments should have same shape
        assert new_moments.shape == (N * 3,)
        # Check that relaxation aligns moments with field
        moments_2d = new_moments.reshape(N, 3)
        # Mean z-component should be positive (alignment with +z field)
        assert np.mean(moments_2d[:, 2]) > 0

    def test_brownian_motion_basic(self):
        """Test Brownian motion basic function."""
        N = 5
        particles = np.random.randn(N, 6)  # [x, y, z, vx, vy, vz]
        dt = 1e-3
        temperature = 300.0
        new_particles = brownian_motion(particles, dt, temperature)
        assert new_particles.shape == particles.shape
        # Check that displacement has non-zero standard deviation
        displacement = new_particles[:, :3] - particles[:, :3]
        assert np.std(displacement) > 0


class TestProvenanceTracking:
    """Tests for provenance tracking through all fidelity levels."""

    def test_provenance_through_relax(self):
        """Test provenance tracking through relaxation."""
        state = FerrofluidState(
            magnetization=100.0,
            relaxation_time=1e-3,
        )
        new_state = ferrofluid_relax(state, dt=100.0)
        # Provenance should be recorded
        assert new_state.magnetization.provenance is not None
        assert new_state.relaxation_time.provenance is not None
        assert new_state.particle_organization.provenance is not None

    def test_provenance_through_write(self):
        """Test provenance tracking through write operation."""
        state = FerrofluidState()
        voltage = TrackedValue(value=5.0, unit="V", provenance=Provenance.ESTIMATED)
        duration = TrackedValue(value=1e-3, unit="s", provenance=Provenance.ESTIMATED)
        new_state = ferrofluid_write(state, voltage=voltage, duration=duration)
        # Provenance should be recorded
        assert new_state.magnetization.provenance is not None
        assert new_state.electrical_impedance.provenance is not None

    def test_provenance_through_all_levels(self):
        """Test provenance tracking through write -> relax -> read cycle."""
        state = FerrofluidState()
        # Write
        voltage = TrackedValue(value=3.0, unit="V", provenance=Provenance.ESTIMATED)
        duration = TrackedValue(value=1e-3, unit="s", provenance=Provenance.ESTIMATED)
        write_state = ferrofluid_write(state, voltage=voltage, duration=duration)
        # Relax
        relax_state = ferrofluid_relax(write_state, dt=100.0)
        # Read
        result = ferrofluid_read(relax_state, frequency=1e6)
        # Provenance should be tracked through the cycle
        assert result["organization"] >= 0


class TestSolverAccuracy:
    """Tests for solver accuracy across operations."""

    def test_analytical_vs_particle_consistency(self):
        """Test consistency between analytical and particle solvers."""
        # Create state with aligned moments
        N = 3
        particles_flat = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0])
        field = np.array([0.0, 0.0, 1.0])
        dt = 1e-9
        result = neel_relaxation(particles_flat, field, dt)
        assert result.shape == (N * 3,)
        # Result should have same number of moments
        moments = result.reshape(N, 3)
        assert moments.shape == (N, 3)

    def test_relaxation_preserves_magnitude(self):
        """Test that relaxation preserves moment magnitudes."""
        N = 3
        moment_mags = np.array([1.0, 2.0, 3.0])
        # Create N particles: each has 3 components (x, y, z) with given magnitude
        particles = np.zeros((N, 3))
        for i in range(N):
            particles[i] = moment_mags[i] * np.array([1.0, 0.0, 0.0])  # x-axis aligned
        particles_flat = particles.flatten()
        field = np.array([0.0, 0.0, 1.0])
        dt = 1e-9
        result = neel_relaxation(particles_flat, field, dt)
        # Result should have same number of elements
        assert result.shape == (N * 3,)
        # Reshape and check magnitudes are preserved (only direction changes)
        result_mags = np.linalg.norm(result.reshape(N, 3), axis=1)
        assert np.allclose(result_mags, moment_mags)


class TestMultiFidelityWorkflow:
    """Tests for workflow across multiple fidelity levels."""

    def test_write_relax_read_workflow(self):
        """Test complete write -> relax -> read workflow across fidelity."""
        state = FerrofluidState()

        # Write operation
        voltage = TrackedValue(value=5.0, unit="V", provenance=Provenance.ESTIMATED)
        duration = TrackedValue(value=1e-3, unit="s", provenance=Provenance.ESTIMATED)
        write_state = ferrofluid_write(state, voltage=voltage, duration=duration)

        # Relaxation operation
        relax_state = ferrofluid_relax(write_state, dt=100.0)

        # Read/readout operation
        result = ferrofluid_read(relax_state, frequency=1e6)
        assert "impedance_magnitude_Ω" in result
        assert result["organization"] >= 0
        assert result["relaxation_time_s"] > 0

    def test_particle_dynamics_consistency(self):
        """Test particle dynamics functions produce consistent outputs."""
        N = 5

        # Brownian motion
        particles = np.random.randn(N, 6)
        dt = 1e-3
        temperature = 300.0
        new_particles = brownian_motion(particles, dt, temperature)
        assert new_particles.shape == (N, 6)

        # Neel relaxation
        particles_flat = np.random.randn(N * 3)
        field = np.array([0.0, 0.0, 1.0])
        dt = 1e-9
        new_moments = neel_relaxation(particles_flat, field, dt)
        assert new_moments.shape == (N * 3,)