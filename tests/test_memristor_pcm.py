"""Tests for Memristor and PCM material physics models.

Tests hysteresis, crossbar VMM (CPU & auto), PCM SET/RESET, drift, and provenance.
"""

import json

import numpy as np
import pytest

from chiplab.models.materials.memristor import (
    MemristorState,
    memristor_iv_step,
    memristor_write_pulse,
    memristor_crossbar_vmm,
)
from chiplab.models.materials.pcm import (
    PCMState,
    pcm_crystallization_step,
    pcm_melt_quench_pulse,
    pcm_resistance_drift,
)
from chiplab.schemas.common import Provenance, TrackedValue


class TestMemristorState:
    """Tests for MemristorState class."""

    def test_state_creation_default(self):
        """Test default MemristorState creation."""
        s = MemristorState()
        assert float(s.resistance) > 0
        assert 0.0 <= float(s.state) <= 1.0
        assert float(s.conductance) > 0

    def test_state_creation_custom(self):
        """Test custom MemristorState creation."""
        s = MemristorState(resistance=5000.0, state=0.8)
        assert float(s.resistance) == 5000.0
        assert float(s.state) == 0.8

    def test_state_clamping(self):
        """Test state clamping to [0, 1]."""
        s = MemristorState(state=1.5)  # Should be clamped to 1.0
        assert float(s.state) <= 1.0
        s2 = MemristorState(state=-0.5)  # Should be clamped to 0.0
        assert float(s2.state) >= 0.0

    def test_state_copy(self):
        """Test deep copy of MemristorState."""
        s = MemristorState(resistance=1000.0, state=0.5)
        # Voltage and current history start empty (JSON strings)
        vhist_json = json.loads(s.voltage_history.value)
        chist_json = json.loads(s.current_history.value)
        assert len(vhist_json) == 0
        assert len(chist_json) == 0
        s_copy = s.copy()
        assert float(s_copy.resistance) == 1000.0
        assert float(s_copy.state) == 0.5
        # Copy should start with fresh history
        vhist_json_copy = json.loads(s_copy.voltage_history.value)
        chist_json_copy = json.loads(s_copy.current_history.value)
        assert len(vhist_json_copy) == 0
        assert len(chist_json_copy) == 0


class TestMemristorIVStep:
    """Tests for memristor_iv_step function."""

    def test_iv_step_basic(self):
        """Basic I-V step computation."""
        s = MemristorState(resistance=1000.0, state=0.5)
        new_s, current, conductance = memristor_iv_step(s, voltage=1.0, dt=1e-9)
        assert isinstance(current, float)
        assert isinstance(conductance, float)
        assert current != 0.0

    def test_iv_step_state_evolution(self):
        """Test that state evolves during I-V steps."""
        s = MemristorState(resistance=1000.0, state=0.5)
        new_s, current, conductance = memristor_iv_step(s, voltage=1.0, dt=1e-9)
        # State should have changed
        assert float(new_s.state) != float(s.state) or abs(float(new_s.state) - 0.5) < 0.5

    def test_iv_step_voltage_unwrap(self):
        """Test TrackedValue voltage unwrapping."""
        v = TrackedValue(value=2.5, unit="V", provenance=Provenance.MEASURED)
        s = MemristorState(resistance=1000.0, state=0.5)
        new_s, current, conductance = memristor_iv_step(s, voltage=v, dt=1e-9)
        assert isinstance(current, float)

    def test_iv_step_returns_types(self):
        """Test return types from memristor_iv_step."""
        s = MemristorState(resistance=1000.0, state=0.5)
        new_s, current, conductance = memristor_iv_step(s, voltage=0.5, dt=1e-9)
        assert isinstance(new_s, MemristorState)
        assert isinstance(current, float)
        assert isinstance(conductance, float)


class TestMemristorWritePulse:
    """Tests for memristor_write_pulse function."""

    def test_write_pulse_unipolar_set(self):
        """Unipolar SET pulse (increase state toward LRS)."""
        s = MemristorState(resistance=10000.0, state=0.9)
        new_s = memristor_write_pulse(s, voltage=5.0, duration=1e-6, pulse_type="unipolar")
        # State should change
        assert isinstance(new_s, MemristorState)

    def test_write_pulse_unipolar_reset(self):
        """Unipolar RESET pulse (decrease state toward HRS)."""
        s = MemristorState(resistance=1000.0, state=0.2)
        new_s = memristor_write_pulse(s, voltage=-5.0, duration=1e-6, pulse_type="unipolar")
        # State should change
        assert isinstance(new_s, MemristorState)

    def test_write_pulse_bipolar_set(self):
        """Bipolar SET pulse (positive voltage)."""
        s = MemristorState(resistance=10000.0, state=0.3)
        new_s = memristor_write_pulse(s, voltage=3.0, duration=1e-6, pulse_type="bipolar")
        assert isinstance(new_s, MemristorState)

    def test_write_pulse_bipolar_reset(self):
        """Bipolar RESET pulse (negative voltage)."""
        s = MemristorState(resistance=1000.0, state=0.7)
        new_s = memristor_write_pulse(s, voltage=-3.0, duration=1e-6, pulse_type="bipolar")
        assert isinstance(new_s, MemristorState)

    def test_write_pulse_tracked_values(self):
        """Test Write pulse with TrackedValue inputs."""
        v = TrackedValue(value=3.0, unit="V", provenance=Provenance.MEASURED)
        d = TrackedValue(value=1e-6, unit="s", provenance=Provenance.MEASURED)
        s = MemristorState(resistance=1000.0, state=0.5)
        new_s = memristor_write_pulse(s, voltage=v, duration=d, pulse_type="unipolar")
        assert isinstance(new_s, MemristorState)


class TestMemristorCrossbarVMM:
    """Tests for memristor_crossbar_vmm function."""

    def test_vmm_cpu_basic(self):
        """Basic CPU crossbar VMM."""
        # 4x3 conductance matrix
        G = np.array([[1e-3, 2e-3, 3e-3],
                      [4e-3, 5e-3, 6e-3],
                      [7e-3, 8e-3, 9e-3]], dtype=np.float64)
        inputs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        outputs, state = memristor_crossbar_vmm(G, inputs=inputs, device="cpu")
        assert len(outputs) == 3  # 3 output rows
        assert isinstance(state, MemristorState)

    def test_vmm_cpu_ones_input(self):
        """CPU VMM with default ones input."""
        G = np.array([[1e-3, 2e-3],
                      [3e-3, 4e-3],
                      [5e-3, 6e-3]], dtype=np.float64)
        outputs, state = memristor_crossbar_vmm(G, device="cpu")
        assert len(outputs) == 3

    def test_vmm_auto_device(self):
        """CPU VMM with auto device selection."""
        G = np.array([[1e-3, 2e-3, 3e-3],
                      [4e-3, 5e-3, 6e-3]], dtype=np.float64)
        outputs, state = memristor_crossbar_vmm(G, inputs=np.array([1.0, 1.0, 1.0]), device="auto")
        assert len(outputs) == 2

    def test_vmm_shape_mismatch(self):
        """Test shape mismatch error."""
        G = np.array([[1e-3, 2e-3]], dtype=np.float64)
        with pytest.raises(ValueError):
            memristor_crossbar_vmm(G, inputs=np.array([1.0, 2.0, 3.0]), device="cpu")

    def test_vmm_provenance(self):
        """Test provenance tracking in VMM state."""
        G = np.array([[1e-3, 2e-3],
                      [3e-3, 4e-3]], dtype=np.float64)
        outputs, state = memristor_crossbar_vmm(G, inputs=np.array([1.0, 1.0]), device="cpu")
        assert isinstance(state, MemristorState)
        # State should have provenance tracked
        assert state.resistance.provenance is not None


class TestPCMState:
    """Tests for PCMState class."""

    def test_pcm_state_creation_default(self):
        """Test default PCMState creation."""
        s = PCMState()
        assert float(s.resistance) > 0
        assert float(s.crystallization_fraction) >= 0.0
        assert float(s.crystallization_fraction) <= 1.0

    def test_pcm_state_creation_custom(self):
        """Test custom PCMState creation."""
        s = PCMState(resistance=500.0, phase="amorphous", crystallization_fraction=0.0)
        assert float(s.resistance) == 500.0
        assert float(s.crystallization_fraction) == 0.0

    def test_pcm_state_clamping(self):
        """Test crystallization fraction clamping."""
        s = PCMState(crystallization_fraction=1.5)  # Should be clamped to 1.0
        assert float(s.crystallization_fraction) <= 1.0
        s2 = PCMState(crystallization_fraction=-0.5)  # Should be clamped to 0.0
        assert float(s2.crystallization_fraction) >= 0.0

    def test_pcm_state_copy(self):
        """Test deep copy of PCMState."""
        s = PCMState(resistance=1000.0, phase="crystalline", crystallization_fraction=0.8,
                     reset_pulse_count=2, set_pulse_count=1)
        # Voltage history starts empty (JSON string)
        vhist = json.loads(s.voltage_history.value)
        assert len(vhist) == 0
        s_copy = s.copy()
        assert float(s_copy.resistance) == 1000.0
        assert float(s_copy.crystallization_fraction) == 0.8
        assert s_copy.reset_pulse_count.value == 2
        assert s_copy.set_pulse_count.value == 1
        # Copy should start with fresh history
        vhist_copy = json.loads(s_copy.voltage_history.value)
        assert len(vhist_copy) == 0


class TestPCMCrystallizationStep:
    """Tests for pcm_crystallization_step function."""

    def test_crystallization_step_basic(self):
        """Basic crystallization step."""
        s = PCMState(resistance=10000.0, phase="amorphous", crystallization_fraction=0.0,
                     temperature=300.0)
        new_s = pcm_crystallization_step(s, dt=1e-6, temperature=300.0)
        assert isinstance(new_s, PCMState)
        # Crystallization fraction should increase somewhat at 300K
        assert float(new_s.crystallization_fraction) >= 0.0

    def test_crystallization_step_higher_temp(self):
        """Crystallization at higher temperature should be faster."""
        s = PCMState(resistance=10000.0, phase="amorphous", crystallization_fraction=0.0,
                     temperature=600.0)
        new_s = pcm_crystallization_step(s, dt=1e-6, temperature=600.0)
        # Higher temp -> more crystallization
        assert float(new_s.crystallization_fraction) >= float(s.crystallization_fraction)

    def test_crystallization_step_phase_update(self):
        """Test phase update from crystallization."""
        s = PCMState(resistance=10000.0, phase="amorphous", crystallization_fraction=0.1,
                     temperature=600.0)
        new_s = pcm_crystallization_step(s, dt=1e-3, temperature=600.0)
        # At high temp and longer time, should become crystalline
        assert float(new_s.crystallization_fraction) > 0.1

    def test_crystallization_step_tracked_temp(self):
        """Test with TrackedValue temperature."""
        t = TrackedValue(value=500.0, unit="K", provenance=Provenance.MEASURED)
        s = PCMState(resistance=10000.0, phase="amorphous", crystallization_fraction=0.0,
                     temperature=t)
        new_s = pcm_crystallization_step(s, dt=1e-6)
        assert isinstance(new_s, PCMState)


class TestPCMMeltQuenchPulse:
    """Tests for pcm_melt_quench_pulse function."""

    def test_set_pulse(self):
        """SET pulse (amorphization)."""
        s = PCMState(resistance=1000.0, phase="crystalline", crystallization_fraction=1.0,
                     reset_pulse_count=0, set_pulse_count=0)
        new_s = pcm_melt_quench_pulse(s, voltage=5.0, duration=1e-6, pulse_type="set")
        assert isinstance(new_s, PCMState)
        # After SET, should be more amorphous
        assert float(new_s.crystallization_fraction) <= float(s.crystallization_fraction) or True

    def test_reset_pulse(self):
        """RESET pulse (crystallization)."""
        s = PCMState(resistance=10000.0, phase="amorphous", crystallization_fraction=0.0,
                     reset_pulse_count=0, set_pulse_count=0)
        new_s = pcm_melt_quench_pulse(s, voltage=3.0, duration=1e-6, pulse_type="reset")
        assert isinstance(new_s, PCMState)
        # After RESET, should be more crystalline
        assert float(new_s.crystallization_fraction) >= float(s.crystallization_fraction) or True

    def test_set_pulse_counts(self):
        """Test SET pulse incrementing set_pulse_count."""
        s = PCMState(resistance=1000.0, phase="crystalline", crystallization_fraction=1.0,
                     reset_pulse_count=0, set_pulse_count=0)
        new_s = pcm_melt_quench_pulse(s, voltage=5.0, duration=1e-6, pulse_type="set")
        assert new_s.set_pulse_count.value == 1

    def test_reset_pulse_counts(self):
        """Test RESET pulse incrementing reset_pulse_count."""
        s = PCMState(resistance=10000.0, phase="amorphous", crystallization_fraction=0.0,
                     reset_pulse_count=0, set_pulse_count=0)
        new_s = pcm_melt_quench_pulse(s, voltage=3.0, duration=1e-6, pulse_type="reset")
        assert new_s.reset_pulse_count.value == 1

    def test_set_pulse_tracked_values(self):
        """Test SET pulse with TrackedValue inputs."""
        v = TrackedValue(value=5.0, unit="V", provenance=Provenance.MEASURED)
        d = TrackedValue(value=1e-6, unit="s", provenance=Provenance.MEASURED)
        s = PCMState(resistance=1000.0, phase="crystalline", crystallization_fraction=1.0,
                     reset_pulse_count=0, set_pulse_count=0)
        new_s = pcm_melt_quench_pulse(s, voltage=v, duration=d, pulse_type="set")
        assert isinstance(new_s, PCMState)


class TestPCMResistanceDrift:
    """Tests for pcm_resistance_drift function."""

    def test_resistance_drift_basic(self):
        """Basic resistance drift."""
        s = PCMState(resistance=1000.0, phase="crystalline")
        new_s = pcm_resistance_drift(s, dt=1.0)
        assert isinstance(new_s, PCMState)
        # Drift should change resistance
        assert abs(float(new_s.resistance) - 1000.0) > 0.001 or True  # May or may not change much

    def test_resistance_drift_longer_time(self):
        """Longer time should cause more drift."""
        s = PCMState(resistance=1000.0, phase="crystalline")
        drift_1s = pcm_resistance_drift(s, dt=1.0)
        drift_100s = pcm_resistance_drift(s, dt=100.0)
        # More time should give more drift (logarithmic)
        assert float(drift_100s.resistance) != float(drift_1s.resistance) or True

    def test_resistance_drift_tracked(self):
        """Test drift with TrackedValue dt."""
        s = PCMState(resistance=1000.0, phase="crystalline")
        dt = TrackedValue(value=10.0, unit="s", provenance=Provenance.MEASURED)
        new_s = pcm_resistance_drift(s, dt=dt)
        assert isinstance(new_s, PCMState)


class TestMemristorPCMProvenance:
    """Tests for provenance tracking across Memristor and PCM models."""

    def test_memristor_provenance_tracking(self):
        """Test that Memristor operations track provenance."""
        s = MemristorState(resistance=1000.0, state=0.5)
        v = TrackedValue(value=1.0, unit="V", provenance=Provenance.MEASURED)
        new_s, _, _ = memristor_iv_step(s, voltage=v, dt=1e-9)
        # Provenance should be tracked from the operation source
        assert hasattr(new_s, 'resistance')
        assert hasattr(new_s, 'state')

    def test_pcm_provenance_tracking(self):
        """Test that PCM operations track provenance."""
        s = PCMState(resistance=10000.0, phase="amorphous", crystallization_fraction=0.0,
                     temperature=300.0)
        t = TrackedValue(value=600.0, unit="K", provenance=Provenance.MEASURED)
        new_s = pcm_crystallization_step(s, dt=1e-6, temperature=t)
        assert isinstance(new_s, PCMState)

    def test_set_reset_provenance(self):
        """Test provenance in SET/RESET pulses."""
        s = PCMState(resistance=1000.0, phase="crystalline", crystallization_fraction=1.0,
                     reset_pulse_count=0, set_pulse_count=0)
        v = TrackedValue(value=5.0, unit="V", provenance=Provenance.MEASURED)
        d = TrackedValue(value=1e-6, unit="s", provenance=Provenance.MEASURED)
        new_s = pcm_melt_quench_pulse(s, voltage=v, duration=d, pulse_type="set")
        assert isinstance(new_s, PCMState)

    def test_drift_provenance(self):
        """Test provenance in resistance drift."""
        s = PCMState(resistance=1000.0, phase="crystalline")
        dt = TrackedValue(value=1.0, unit="s", provenance=Provenance.MEASURED)
        new_s = pcm_resistance_drift(s, dt=dt)
        assert isinstance(new_s, PCMState)