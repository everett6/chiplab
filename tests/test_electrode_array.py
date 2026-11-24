"""Tests for Multiphysics Device Lab electrode array architecture."""

import numpy as np
import pytest

from chiplab.models.materials.electrode_array import (
    ElectrodeArray,
    Port,
    create_array,
    simultaneous_write,
    independent_read,
    write_read_interference,
    cross_port_coupling,
    frequency_dependent_impedance,
    state_retention,
    signal_noise,
)


class TestPortCreation:
    """Tests for Port dataclass."""

    def test_port_creation_basic(self):
        """Test basic Port creation."""
        port = Port(
            id="port_1",
            position=(1e-3, 0.0, 0.0),
            shape="circular",
            area=1e-6,
            write_voltage_range=(-5.0, 5.0),
            read_frequency_range=(1e3, 1e7),
            read_amplitude=0.1,
        )
        assert port.id == "port_1"
        assert port.position == (1e-3, 0.0, 0.0)
        assert port.shape == "circular"
        assert port.area == 1e-6

    def test_port_shapes(self):
        """Test different port shapes."""
        for shape in ["circular", "rectangular"]:
            port = Port(
                id="port_1",
                position=(1e-3, 0.0, 0.0),
                shape=shape,
                area=1e-6,
                write_voltage_range=(-5.0, 5.0),
                read_frequency_range=(1e3, 1e7),
                read_amplitude=0.1,
            )
            assert port.shape == shape


class TestElectrodeArrayCreation:
    """Tests for ElectrodeArray class construction."""

    def test_array_2port_linear(self):
        """Test creating a 2-port linear array."""
        array = create_array(n_ports=2, arrangement="linear", pitch=10e-3)
        assert len(array.ports) == 2
        assert array.geometry["positions"].shape == (2, 3)

    def test_array_4port_grid_2x2(self):
        """Test creating a 4-port 2x2 grid array."""
        array = create_array(n_ports=4, arrangement="grid_2x2", pitch=10e-3)
        assert len(array.ports) == 4

    def test_array_8port_grid_4x4(self):
        """Test creating an 8-port 4x4 grid array."""
        array = create_array(n_ports=8, arrangement="grid_4x4", pitch=10e-3)
        assert len(array.ports) == 8

    def test_array_16port_grid_4x4(self):
        """Test creating a 16-port 4x4 grid array."""
        array = create_array(n_ports=16, arrangement="grid_4x4", pitch=10e-3)
        assert len(array.ports) == 16

    def test_array_circular(self):
        """Test creating a circular arrangement."""
        array = create_array(n_ports=4, arrangement="circular", pitch=10e-3)
        assert len(array.ports) == 4

    def test_array_custom(self):
        """Test creating a custom arrangement."""
        array = create_array(
            n_ports=4,
            arrangement="custom",
            pitch=10e-3,
            port_config={"positions": [(1e-3, 0.0, 0.0), (2e-3, 0.0, 0.0), (3e-3, 0.0, 0.0), (4e-3, 0.0, 0.0)]},
        )
        assert len(array.ports) == 4


class TestArrayCreationCounts:
    """Tests for array creation with various port counts."""

    @pytest.mark.parametrize("n_ports,arrangement", [
        (2, "linear"),
        (4, "linear"),
        (8, "grid_4x4"),
        (16, "grid_4x4"),
        (4, "circular"),
    ])
    def test_various_arrangements(self, n_ports, arrangement):
        """Test array creation with various port counts and arrangements."""
        if arrangement == "grid_2x2" and n_ports > 4:
            with pytest.raises(ValueError):
                create_array(n_ports=n_ports, arrangement=arrangement, pitch=10e-3)
        else:
            array = create_array(n_ports=n_ports, arrangement=arrangement, pitch=10e-3)
            assert len(array.ports) == n_ports


class TestSimultaneousWrite:
    """Tests for simultaneous write operation."""

    def test_simultaneous_write_basic(self):
        """Test basic simultaneous write to all ports."""
        array = create_array(n_ports=4, arrangement="linear", pitch=10e-3)
        voltages = {port.id: 3.0 for port in array.ports}
        result = simultaneous_write(array, port_voltages=voltages, duration=1e-3)
        assert len(result) == 4
        for port_id, tracked_value in result.items():
            assert tracked_value.unit == "V"
            assert tracked_value.value == 3.0  # Clipped to range

    def test_simultaneous_write_clipping(self):
        """Test that voltages outside range are clipped."""
        array = create_array(n_ports=4, arrangement="linear", pitch=10e-3)
        voltages = {port.id: 100.0 for port in array.ports}  # Outside range (-5 to 5)
        result = simultaneous_write(array, port_voltages=voltages, duration=1e-3)
        for port_id, tracked_value in result.items():
            assert tracked_value.value == 5.0  # Clipped to max

    def test_simultaneous_write_retains_state(self):
        """Test that write updates material state retention."""
        array = create_array(n_ports=4, arrangement="linear", pitch=10e-3)
        initial_retention = {port_id: state["retention"] for port_id, state in array.material_state.items()}
        simultaneous_write(array, port_voltages={port_id: 3.0 for port_id in array.material_state.keys()}, duration=1e-3)
        for port_id in array.material_state.keys():
            assert array.material_state[port_id]["retention"] < initial_retention[port_id]


class TestIndependentRead:
    """Tests for independent read operation."""

    def test_independent_read_basic(self):
        """Test basic independent read from all ports."""
        array = create_array(n_ports=4, arrangement="linear", pitch=10e-3)
        frequencies = {port.id: 1e6 for port in array.ports}
        result = independent_read(array, port_frequencies=frequencies)
        assert len(result) == 4
        for port_id, tracked_value in result.items():
            assert tracked_value.unit == "Ω"

    def test_independent_read_different_frequencies(self):
        """Test read with different frequencies per port."""
        array = create_array(n_ports=4, arrangement="linear", pitch=10e-3)
        frequencies = {
            array.ports[0].id: 1e3,
            array.ports[1].id: 1e4,
            array.ports[2].id: 1e5,
            array.ports[3].id: 1e6,
        }
        result = independent_read(array, port_frequencies=frequencies)
        assert len(result) == 4


class TestCrosstalkAndCoupling:
    """Tests for crosstalk and coupling matrices."""

    def test_write_read_interference_basic(self):
        """Test basic crosstalk matrix computation."""
        array = create_array(n_ports=4, arrangement="linear", pitch=10e-3)
        write_ports = [array.ports[0].id, array.ports[1].id]
        read_ports = [array.ports[2].id, array.ports[3].id]
        coupling = write_read_interference(array, write_ports=write_ports, read_ports=read_ports)
        assert coupling.shape == (2, 2)  # 2 read ports x 2 write ports

    def test_cross_port_coupling(self):
        """Test cross-port coupling coefficient."""
        array = create_array(n_ports=4, arrangement="linear", pitch=10e-3)
        # Adjacent ports should have higher coupling
        coupling_adjacent = cross_port_coupling(array, array.ports[0].id, array.ports[1].id)
        # Far ports should have lower coupling
        coupling_far = cross_port_coupling(array, array.ports[0].id, array.ports[3].id)
        assert coupling_adjacent >= coupling_far
        assert 0.0 <= coupling_adjacent <= 1.0
        assert 0.0 <= coupling_far <= 1.0

    def test_cross_port_coupling_same_port(self):
        """Test coupling of port to itself."""
        array = create_array(n_ports=4, arrangement="linear", pitch=10e-3)
        coupling = cross_port_coupling(array, array.ports[0].id, array.ports[0].id)
        assert coupling > 0.5  # Self-coupling should be high


class TestFrequencyDependentImpedance:
    """Tests for frequency-dependent impedance."""

    def test_frequency_dependent_impedance_basic(self):
        """Test frequency-dependent impedance computation."""
        array = create_array(n_ports=4, arrangement="linear", pitch=10e-3)
        frequencies = np.array([1e3, 1e6, 1e9])
        result = frequency_dependent_impedance(array, frequencies)
        assert len(result) == 4  # All ports
        for port_id, Z in result.items():
            assert len(Z) == 3  # 3 frequency points

    def test_frequency_dependent_impedance_sweep(self):
        """Test impedance across frequency sweep."""
        array = create_array(n_ports=2, arrangement="linear", pitch=10e-3)
        frequencies = np.logspace(3, 7, 10)  # 1 kHz to 10 MHz
        result = frequency_dependent_impedance(array, frequencies)
        assert len(result) == 2


class TestStateRetentionAndNoise:
    """Tests for state retention and noise."""

    def test_state_retention_basic(self):
        """Test state retention decay over time."""
        array = create_array(n_ports=4, arrangement="linear", pitch=10e-3)
        retention = state_retention(array, time=1.0)
        assert len(retention) == 4
        for port_id, ret in retention.items():
            assert 0.0 <= ret <= 1.0
            # Longer retention time should have higher retention
        retention_long = state_retention(array, time=100.0)
        retention_short = state_retention(array, time=1.0)
        # All retentions should be <= 1 and decay over time
        for port_id in retention.keys():
            assert retention_long[port_id] <= retention_short[port_id]

    def test_signal_noise_basic(self):
        """Test noise computation per port."""
        array = create_array(n_ports=4, arrangement="linear", pitch=10e-3)
        noise = signal_noise(array, bandwidth=1e6)
        assert len(noise) == 4
        for port_id, tracked_value in noise.items():
            assert tracked_value.unit == "V"
            assert tracked_value.value >= 0


class TestThreeFidelities:
    """Tests consistency across fidelity levels in electrode array context."""

    def test_array_with_analytical_fidelity(self):
        """Test electrode array operations with analytical model."""
        array = create_array(n_ports=4, arrangement="linear", pitch=10e-3)
        # Write operation
        voltages = simultaneous_write(array, port_voltages={port.id: 3.0 for port in array.ports}, duration=1e-3)
        assert len(voltages) == 4
        # Read operation
        frequencies = {port.id: 1e6 for port in array.ports}
        result = independent_read(array, port_frequencies=frequencies)
        assert len(result) == 4