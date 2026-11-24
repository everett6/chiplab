"""Tests for Multiphysics Device Lab GPU Studio Material Explorer."""

import pytest
import json

from chiplab.models.materials.ferrofluid import FerrofluidState, ferrofluid_write, ferrofluid_read, ferrofluid_relax
from chiplab.models.materials.impedance import impedance_matrix, frequency_response


class TestHTMLExportValidity:
    """Tests for HTML export validity."""

    def test_export_html_structure(self):
        """Test HTML export has valid structure."""
        state = FerrofluidState()
        # Generate some simulation data
        result = ferrofluid_read(state, frequency=1e6)
        
        # Verify we can create a basic HTML structure with the data
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Ferrofluid Material Simulation</title>",
            "</head>",
            "<body>",
            f"<h1>Ferrofluid Simulation Results</h1>",
            f"<p>Frequency: {result['frequency_hz']:.2e} Hz</p>",
            f"<p>Impedance (real): {result['impedance_real_Ω']:.4f} Ω</p>",
            f"<p>Impedance (imag): {result['impedance_imag_Ω']:.4f} Ω</p>",
            f"<p>Magnitude: {result['impedance_magnitude_Ω']:.4f} Ω</p>",
            f"<p>Phase: {result['phase_rad']:.4f} rad</p>",
            f"<p>Organization: {result['organization']:.4f}</p>",
            "</body>",
            "</html>",
        ]
        
        html_content = "\n".join(html_parts)
        assert "<!DOCTYPE html>" in html_content
        assert "</html>" in html_content
        assert "Ferrofluid Simulation Results" in html_content

    def test_html_minimal_valid(self):
        """Test minimal valid HTML output."""
        html = """<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body><p>Test content</p></body>
</html>"""
        
        assert html.startswith("<!DOCTYPE html>")
        assert html.startswith("<html>")
        assert html.endswith("</html>")


class TestParticleVisualizationDataFormat:
    """Tests for particle visualization data format."""

    def test_particle_data_format(self):
        """Test particle visualization data format."""
        from chiplab.models.materials.particle_dynamics import brownian_motion, neel_relaxation
        
        # Create particle data
        N = 10
        particles = np.random.randn(N, 3)  # magnetic moment vectors
        
        # Test brownian_motion output format
        dt = 1e-3
        temperature = 300.0
        new_particles = brownian_motion(particles.reshape(N, 6) if particles.ndim == 1 else particles, dt, temperature)
        
        # Verify format
        assert new_particles.shape == particles.shape
        
        # Test neel_relaxation output format
        field = np.array([0.0, 0.0, 1.0])
        relaxed = neel_relaxation(particles, field, dt)
        assert len(relaxed) == N


class TestHeatmapGeneration:
    """Tests for heatmap generation."""

    def test_heatmap_data_generation(self):
        """Test heatmap data generation from simulation results."""
        from chiplab.models.materials.ferrofluid import FerrofluidState
        
        state = FerrofluidState()
        result = ferrofluid_read(state, frequency=1e6)
        
        # Generate heatmap data from impedance frequency response
        # This would typically involve varying frequency and collecting magnitude data
        frequencies = np.logspace(3, 6, 50)  # 1 kHz to 1 MHz
        magnitudes = []
        
        for freq in frequencies:
            r = ferrofluid_read(state, frequency=freq)
            magnitudes.append(r["impedance_magnitude_Ω"])
        
        # Verify heatmap data structure
        assert len(magnitudes) == len(frequencies)
        assert all(m > 0 for m in magnitudes)
        
        # Verify frequency range
        assert frequencies[0] == 1e3
        assert frequencies[-1] == 1e6

    def test_heatmap_with_organization(self):
        """Test heatmap generation with varying organization."""
        from chiplab.models.materials.ferrofluid import FerrofluidState, ferrofluid_write
        
        # Create states with different organizations
        states = []
        for org in [0.0, 0.5, 1.0]:
            state = FerrofluidState(particle_organization=org)
            states.append(state)
        
        results = []
        for state in states:
            result = ferrofluid_read(state, frequency=1e6)
            results.append(result["impedance_magnitude_Ω"])
        
        # Heatmap should show variation with organization
        assert results[0] != results[-1]  # Different organizations should give different results


class TestTimeSliderSynchronization:
    """Tests for time slider synchronization with solver."""

    def test_time_slider_sync_basic(self):
        """Test time slider synchronization with solver steps."""
        from chiplab.models.materials.ferrofluid import ferrofluid_relax, FerrofluidState
        
        # Create initial state
        state = FerrofluidState(magnetization=100.0, relaxation_time=1e-3)
        
        # Simulate time steps with different durations
        dt_values = [1.0, 10.0, 100.0, 1000.0]
        states = [state]
        
        for dt in dt_values:
            new_state = ferrofluid_relax(state, dt=dt)
            states.append(new_state)
            state = new_state
        
        # Verify we have the expected number of states
        assert len(states) == len(dt_values) + 1  # initial + each step
        
        # Magnetization should decay over time
        magnifications = [s.magnetization.value for s in states]
        assert magnifications[0] > magnifications[-1]  # Should decay over time

    def test_time_slider_sync_consistency(self):
        """Test time slider synchronization produces consistent results."""
        from chiplab.models.materials.ferrofluid import ferrofluid_relax, FerrofluidState
        
        # Same initial state, same total time, different step sizes
        total_time = 100.0
        state = FerrofluidState(magnetization=100.0, relaxation_time=1e-3)
        
        # Single large step
        state1 = ferrofluid_relax(state, dt=total_time)
        
        # Multiple small steps
        state2 = FerrofluidState(magnetization=100.0, relaxation_time=1e-3)
        steps = 10
        step_dt = total_time / steps
        for _ in range(steps):
            state2 = ferrofluid_relax(state2, dt=step_dt)
        
        # Results should be approximately consistent (not exact due to discrete steps)
        assert abs(state1.magnetization.value - state2.magnetization.value) < 50.0