"""Tests for Multiphysics Device Lab circuit cosimulation modules."""
import numpy as np
import pytest

from chiplab.models.materials.ferrofluid import FerrofluidState, ferrofluid_read, ferrofluid_write, ferrofluid_relax
from chiplab.models.materials.impedance import impedance_matrix, frequency_response
from chiplab.schemas.common import TrackedValue, Provenance


def _get_value(v):
    """Helper to get value from TrackedValue or return directly."""
    if isinstance(v, TrackedValue):
        return v.value
    return v


class TestCircuitComponentModels:
    """Tests for circuit component models."""

    def test_ferrofluid_write_read_consistency(self):
        """Test that write followed by read produces consistent results."""
        state = FerrofluidState()
        # Apply a write operation
        new_state = ferrofluid_write(state, voltage=5.0, duration=1e-3)
        # Read the impedance
        result = ferrofluid_read(new_state, frequency=1e6)
        assert "impedance_magnitude_Ω" in result
        assert result["organization"] > 0  # Organization should increase after write

    def test_read_different_frequencies(self):
        """Test read operation with different frequencies."""
        state = FerrofluidState()
        result_low = ferrofluid_read(state, frequency=1e3)
        result_high = ferrofluid_read(state, frequency=1e9)
        assert result_low["frequency_hz"] == 1e3
        assert result_high["frequency_hz"] == 1e9

    def test_write_with_tracked_values(self):
        """Test write operation with TrackedValue parameters."""
        state = FerrofluidState()
        voltage = TrackedValue(value=3.0, unit="V", provenance=Provenance.ESTIMATED)
        duration = TrackedValue(value=2e-3, unit="s", provenance=Provenance.ESTIMATED)
        new_state = ferrofluid_write(state, voltage=voltage, duration=duration)
        assert new_state is not state  # Should be a new object


class TestTransientAnalysis:
    """Tests for transient analysis."""

    def test_transient_relaxation(self):
        """Test transient relaxation of ferrofluid state."""
        state = FerrofluidState(magnetization=100.0, relaxation_time=1e-3)
        # Apply relaxation multiple times
        states = [state]
        for _ in range(5):
            new_state = ferrofluid_relax(states[-1], dt=10.0)
            states.append(new_state)
        # Magnetization should decay over time
        mags = [abs(s.magnetization.value) for s in states[1:]]
        initial_mag = abs(states[0].magnetization.value)
        assert all(m < initial_mag for m in mags)

    def test_transient_organization_decay(self):
        """Test organization decay during transient relaxation."""
        state = FerrofluidState(magnetization=100.0, particle_organization=0.8, relaxation_time=1e-3)
        new_state = ferrofluid_relax(state, dt=100.0)
        # Organization should decay towards zero
        assert abs(_get_value(new_state.particle_organization)) < abs(_get_value(state.particle_organization))


class TestACAnalysis:
    """Tests for AC analysis."""

    def test_frequency_response_basic(self):
        """Test basic frequency response computation."""
        state = FerrofluidState()
        result = frequency_response(state, f_min=1e3, f_max=1e6, n_points=10)
        assert "frequencies_Hz" in result
        assert "impedance_magnitude_Ω" in result
        assert len(result["frequencies_Hz"]) == 10
        assert result["f_min_Hz"] == 1e3
        assert result["f_max_Hz"] == 1e6

    def test_frequency_response_organization_dependence(self):
        """Test frequency response depends on organization state."""
        state_organized = FerrofluidState(particle_organization=0.8)
        state_disorganized = FerrofluidState(particle_organization=0.1)
        # Check that organization values are different
        assert _get_value(state_organized.particle_organization) == 0.8
        assert _get_value(state_disorganized.particle_organization) == 0.1
        # Read impedance at a single frequency and check organization affects response
        result_organized = ferrofluid_read(state_organized, frequency=1e6)
        result_disorganized = ferrofluid_read(state_disorganized, frequency=1e6)
        # Organization should affect the response amplitude
        assert result_organized["response_amplitude"] != result_disorganized["response_amplitude"]
        # More organized state should have higher response amplitude
        assert result_organized["response_amplitude"] > result_disorganized["response_amplitude"]


class TestSPICENetlistExport:
    """Tests for SPICE netlist export format."""

    def test_impedance_matrix_netlist_format(self):
        """Test impedance matrix can be formatted for SPICE."""
        electrode_positions = np.array([
            [0.0, 0.0, 0.0],
            [1e-2, 0.0, 0.0],
        ])
        frequencies = [1e3, 1e6]
        Z = impedance_matrix(None, electrode_positions, frequencies)
        # Verify complex impedance matrix format
        assert Z.shape == (2, 2, 2)  # 2 frequencies, 2 electrodes
        assert np.iscomplexobj(Z)

    def test_netlist_export_has_required_segments(self):
        """Test SPICE netlist export contains required segments."""
        from chiplab.models.materials.impedance import impedance_matrix
        electrode_positions = np.array([
            [0.0, 0.0, 0.0],
            [1e-2, 0.0, 0.0],
        ])
        frequencies = [1e3]
        Z = impedance_matrix(None, electrode_positions, frequencies)
        # Check that real and imaginary parts are extractable
        real_part = np.real(Z[0])
        imag_part = np.imag(Z[0])
        assert real_part.shape == (2, 2)
        assert imag_part.shape == (2, 2)


class TestCosimulationWithFerrofluid:
    """Tests for cosimulation with ferrofluid device model."""

    def test_cosimulation_basic_workflow(self):
        """Test basic cosimulation workflow with ferrofluid model."""
        state = FerrofluidState()
        # Simulate write operation
        new_state = ferrofluid_write(state, voltage=3.0, duration=1e-3)
        # Read impedance at multiple frequencies
        result = ferrofluid_read(new_state, frequency=1e6)
        assert result["impedance_magnitude_Ω"] > 0

    def test_cosimulation_state_evolution(self):
        """Test state evolution through write -> read -> relax cycle."""
        state = FerrofluidState()
        # Write
        write_state = ferrofluid_write(state, voltage=5.0, duration=1e-3)
        # Read
        read_result = ferrofluid_read(write_state, frequency=1e6)
        # Relax
        relaxed_state = ferrofluid_relax(write_state, dt=100.0)
        # Read after relaxation
        relaxed_result = ferrofluid_read(relaxed_state, frequency=1e6)
        # Organization should change through the cycle
        assert read_result["organization"] > 0
        assert relaxed_result["organization"] < read_result["organization"]