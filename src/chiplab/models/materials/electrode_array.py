"""Electrode array configuration and simulation for configurable multi-port ferrofluid devices.

HEURISTIC: the write, read, coupling, impedance, retention and noise rules here are
placeholder scalings, not physics. Port layout is usable; for physical multi-port results
use ``ferro_device`` (docs/ferrofluid_model.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from chiplab.schemas.common import TrackedValue, Provenance


@dataclass
class Port:
    """Dataclass representing a single electrode port.

    Attributes
    ----------
    id : str
        Unique port identifier.
    position : tuple[float, float, float]
        3D position (x, y, z) of the port in meters.
    shape : Literal["circular", "rectangular"]
        Shape of the electrode port.
    area : float
        Electrode area in square meters (m^2).
        For circular: area = π * (diameter/2)^2
        For rectangular: area = width * height
    write_voltage_range : tuple[float, float]
        Valid write voltage range (min, max) in volts.
    read_frequency_range : tuple[float, float]
        Valid read frequency range (min, max) in Hz.
    read_amplitude : float
        Read signal amplitude in volts.
    """

    id: str
    position: tuple[float, float, float]
    shape: Literal["circular", "rectangular"]
    area: float
    write_voltage_range: tuple[float, float]
    read_frequency_range: tuple[float, float]
    read_amplitude: float

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Port):
            return self.id == other.id
        return False


class ElectrodeArray:
    """Class representing a configurable multi-port electrode array for ferrofluid devices.

    Attributes
    ----------
    ports : list[Port]
        List of Port objects, each representing an electrode port.
    geometry : dict
        Dictionary containing electrode geometry information:
        - 'positions': numpy array of shape (n_ports, 3)
        - 'areas': numpy array of shape (n_ports,)
        - 'shapes': numpy array of str of shape (n_ports,)
    material_state : dict
        Reference to ferrofluid state tracking per port.
    """

    def __init__(self, ports: list[Port]):
        self.ports = ports
        self.geometry = {
            "positions": np.array([p.position for p in ports]),
            "areas": np.array([p.area for p in ports]),
            "shapes": np.array([p.shape for p in ports]),
        }
        self.material_state = {
            port: {"retention": 1.0, "impedance": None, "noise": None}
            for port in ports
        }

    def simultaneous_write(
        self, port_voltages: dict[str, float], duration: float
    ) -> dict[str, TrackedValue]:
        """Apply write voltages to ports and update material state.

        Parameters
        ----------
        port_voltages : dict[str, float]
            Mapping of port id to applied write voltage in volts.
            Voltages outside the port's write_voltage_range are clipped.
        duration : float
            Write duration in seconds.

        Returns
        -------
        dict[str, TrackedValue]
            Mapping of port id to tracked write voltage value with unit 'V'.
        """
        applied_voltages: dict[str, TrackedValue] = {}
        for port in self.ports:
            voltage = port_voltages.get(port.id, 0.0)
            vmin, vmax = port.write_voltage_range
            clipped = max(vmin, min(vmax, voltage))
            applied_voltages[port.id] = TrackedValue(
                value=clipped, unit="V", provenance=Provenance.ESTIMATED
            )
            # Update material state retention for this port
            if port in self.material_state:
                self.material_state[port]["retention"] *= (
                    1.0
                    - 0.001
                    * abs(clipped)
                    / max(1.0, (vmax - vmin))
                ) * max(0.0, 1.0 - duration / 1000.0)

        return applied_voltages

    def independent_read(
        self, port_frequencies: dict[str, float]
    ) -> dict[str, TrackedValue]:
        """Read impedance per port at specified frequencies.

        Parameters
        ----------
        port_frequencies : dict[str, float]
            Mapping of port id to read frequency in Hz.
            Frequencies outside the port's read_frequency_range are clipped.

        Returns
        -------
        dict[str, TrackedValue]
            Mapping of port id to tracked impedance value in ohms.
        """
        impedances: dict[str, TrackedValue] = {}
        for port in self.ports:
            freq = port_frequencies.get(port.id, 1e6)
            fmin, fmax = port.read_frequency_range
            clipped_freq = max(fmin, min(fmax, freq))

            # Impedance model: magnitude decreases with frequency for ferrofluid
            base_impedance = port.area * 1e3  # ohms, scaled by area
            freq_ratio = clipped_freq / max(1e-6, fmin)
            freq_dependent = base_impedance / np.sqrt(max(1e-12, freq_ratio))

            # Use magnitude of complex impedance, store phase in notes
            impedance_magnitude = abs(complex(freq_dependent, freq_dependent * 0.1))
            impedance_phase = np.angle(complex(freq_dependent, freq_dependent * 0.1))
            impedances[port.id] = TrackedValue(
                value=float(impedance_magnitude),
                unit="Ω",
                provenance=Provenance.ASSUMED,
                notes=f"phase:{impedance_phase:.4f}",
            )
            self.material_state[port]["impedance"] = float(freq_dependent)

        return impedances

    def write_read_interference(
        self, write_ports: list[str], read_ports: list[str]
    ) -> np.ndarray:
        """Compute crosstalk matrix between write and read ports.

        Parameters
        ----------
        write_ports : list[str]
            List of port IDs designated as write ports.
        read_ports : list[str]
            List of port IDs designated as read ports.

        Returns
        -------
        np.ndarray
            Crosstalk matrix of shape (n_read_ports, n_write_ports) where
            element [i, j] is the coupling coefficient from write port j
            to read port i.
        """
        n_write = len(write_ports)
        n_read = len(read_ports)
        if n_write == 0 or n_read == 0:
            return np.zeros((n_read, n_write))

        # Compute coupling based on distance between port positions
        coupling_matrix = np.zeros((n_read, n_write))

        for i, read_id in enumerate(read_ports):
            for j, write_id in enumerate(write_ports):
                # Find port objects
                read_port = next((p for p in self.ports if p.id == read_id), None)
                write_port = next(
                    (p for p in self.ports if p.id == write_id), None
                )
                if read_port is None or write_port is None:
                    continue

                # Distance-based coupling: inverse distance with exponential decay
                pos_diff = np.array(read_port.position) - np.array(
                    write_port.position
                )
                distance = np.linalg.norm(pos_diff)

                # Coupling coefficient: decreases with distance and depends on areas
                area_product = read_port.area * write_port.area
                # Reference distance of 1mm for normalization
                ref_distance = 1e-3

                # Exponential distance decay with area scaling
                coupling = (
                    area_product
                    / max(1e-12, distance**2 + ref_distance**2)
                ) * np.exp(-distance / max(1e-6, ref_distance))

                coupling_matrix[i, j] = coupling

        return coupling_matrix

    def cross_port_coupling(self, port_i: str, port_j: str) -> float:
        """Compute coupling coefficient between two ports.

        Parameters
        ----------
        port_i : str
            ID of the first port.
        port_j : str
            ID of the second port.

        Returns
        -------
        float
            Coupling coefficient (0 to 1) between port i and port j.
            Returns 0 if either port is not found.
            Returns 1.0 if port_i == port_j (same port).
        """
        # Same port: perfect coupling
        if port_i == port_j:
            return 1.0

        port_i_obj = next((p for p in self.ports if p.id == port_i), None)
        port_j_obj = next((p for p in self.ports if p.id == port_j), None)

        if port_i_obj is None or port_j_obj is None:
            return 0.0

        # Distance-based coupling
        pos_diff = np.array(port_i_obj.position) - np.array(
            port_j_obj.position
        )
        distance = np.linalg.norm(pos_diff)

        # Normalized coupling: decreases with distance
        # Reference: coupling = 1 at 0 distance, decaying with distance
        ref_distance = 1e-3  # 1 mm reference

        coupling = np.exp(-distance / max(1e-6, ref_distance))

        # Also factor in area similarity
        area_factor = (
            2 * port_i_obj.area * port_j_obj.area
        ) / max(1e-12, port_i_obj.area + port_j_obj.area) ** 2

        return float(coupling * area_factor)

    def frequency_dependent_impedance(
        self, frequencies: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Compute impedance per port across an array of frequencies.

        Parameters
        ----------
        frequencies : np.ndarray
            Array of frequencies in Hz.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of port id to complex impedance array Z(f) in ohms.
            Each array has the same shape as the input frequencies array.
        """
        impedance_dict: dict[str, np.ndarray] = {}

        for port in self.ports:
            # Impedance model for ferrofluid electrode
            # Z(f) = R / sqrt(1 + (f/f0)^2) + j * ω * L
            # Where R is resistance, f0 is characteristic frequency, L is inductance

            R = 1.0 / max(port.area, 1e-12)  # resistance scaled by area
            f0 = 1e6 / max(port.area, 1e-12)  # characteristic frequency
            L = port.area * 1e-9  # inductance scaled by area

            omega = 2 * np.pi * frequencies

            # Frequency-dependent magnitude
            magnitude = R / np.sqrt(1.0 + (frequencies / max(f0, 1e-6)) ** 2)

            # Inductive reactance
            reactance = omega * L

            # Complex impedance
            Z = magnitude + 1j * reactance
            impedance_dict[port.id] = Z

        return impedance_dict

    def state_retention(self, time: float) -> dict[Port, float]:
        """Compute retention decay per port over time.

        Parameters
        ----------
        time : float
            Retention time in seconds.

        Returns
        -------
        dict[Port, float]
            Mapping of Port object to retention value (0 to 1) after the given time.
            Retention decays exponentially with time constant dependent on port area.
        """
        retention: dict[Port, float] = {}
        for port in self.ports:
            # Retention time constant proportional to area (larger area = slower decay)
            tau = max(port.area, 1e-12) * 100.0  # time constant in seconds
            retention_value = np.exp(-time / max(tau, 1e-12))
            retention[port] = float(retention_value)
            self.material_state[port]["retention"] = retention_value

        return retention

    def signal_noise(self, bandwidth: float) -> dict[str, TrackedValue]:
        """Compute noise per port given a measurement bandwidth.

        Parameters
        ----------
        bandwidth : float
            Measurement bandwidth in Hz.

        Returns
        -------
        dict[str, TrackedValue]
            Mapping of port id to tracked noise voltage in volts.
            Noise includes thermal noise component scaled by port area and bandwidth.
        """
        noise: dict[str, TrackedValue] = {}
        for port in self.ports:
            # Thermal noise: v_n = sqrt(4 * k * T * R * Δf)
            # k = Boltzmann constant = 1.38e-23 J/K
            # T = temperature in Kelvin (assume 300 K)
            # R = resistance ≈ 1/area
            # For simplicity, use area-scaled model

            k = 1.38e-23
            T = 300.0
            R = 1.0 / max(port.area, 1e-12)

            thermal_noise = np.sqrt(4 * k * T * R * bandwidth)
            # Scale by read amplitude for relative noise
            noise_value = thermal_noise / max(port.read_amplitude, 1e-12)

            noise[port.id] = TrackedValue(
                value=noise_value,
                unit="V",
                provenance=Provenance.ASSUMED,
            )

        return noise


def create_array(
    n_ports: int,
    arrangement: Literal[
        "linear", "grid_2x2", "grid_4x4", "circular", "custom"
    ],
    pitch: float,
    port_config: dict | None = None,
) -> ElectrodeArray:
    """Create an ElectrodeArray with specified arrangement and port configuration.

    Parameters
    ----------
    n_ports : int
        Number of ports (2, 4, 8, 16, or arbitrary).
    arrangement : {"linear", "grid_2x2", "grid_4x4", "circular", "custom"}
        Spatial arrangement of ports.
    pitch : float
        Spacing between adjacent ports in meters.
    port_config : dict | None, optional
        Additional port configuration overrides. If None, default port configs
        are used. Expected keys: 'shape', 'area', 'voltage_range', 'frequency_range',
        'read_amplitude'. Values can be a single value applied to all ports or
        a list/tuple per port.

    Returns
    -------
    ElectrodeArray
        Configured electrode array with ports positioned according to the arrangement.

    Raises
    ------
    ValueError
        If n_ports is not supported for the given arrangement, or if port_config
        has invalid structure.
    """
    # Default port parameters
    default_shape = "circular"
    default_area = 1e-6  # 1 mm^2 in m^2
    default_voltage_range = (-5.0, 5.0)
    default_frequency_range = (1e3, 1e7)
    default_read_amplitude = 0.1  # 100 mV

    # Parse port_config if provided
    n_configured = 0
    if port_config is not None:
        if "shape" in port_config:
            if isinstance(port_config["shape"], (list, tuple)):
                n_configured = len(port_config["shape"])
            else:
                n_configured = n_ports
        if "area" in port_config:
            if isinstance(port_config["area"], (list, tuple)):
                n_configured = len(port_config["area"])
            else:
                n_configured = n_ports

    # Determine actual number of ports to configure
    actual_n_ports = min(n_configured, n_ports) if n_configured > 0 else n_ports

    # Generate port positions based on arrangement
    positions = []
    ids = []

    if arrangement == "linear":
        if actual_n_ports < 2:
            raise ValueError("Linear arrangement requires at least 2 ports.")
        for i in range(actual_n_ports):
            x = i * pitch
            positions.append((x, 0.0, 0.0))
            ids.append(f"port_{i+1}")

    elif arrangement == "grid_2x2":
        if actual_n_ports > 4:
            raise ValueError("grid_2x2 arrangement supports at most 4 ports.")
        # Fill grid row-major; unused slots omitted
        grid_size = 2
        for i in range(actual_n_ports):
            row = i // grid_size
            col = i % grid_size
            x = col * pitch
            y = row * pitch
            positions.append((x, y, 0.0))
            ids.append(f"port_{i+1}")

    elif arrangement == "grid_4x4":
        if actual_n_ports > 16:
            raise ValueError("grid_4x4 arrangement supports at most 16 ports.")
        grid_size = 4
        for i in range(actual_n_ports):
            row = i // grid_size
            col = i % grid_size
            x = col * pitch
            y = row * pitch
            positions.append((x, y, 0.0))
            ids.append(f"port_{i+1}")

    elif arrangement == "circular":
        if actual_n_ports < 2:
            raise ValueError("Circular arrangement requires at least 2 ports.")
        angle_step = 2 * np.pi / actual_n_ports
        for i in range(actual_n_ports):
            angle = i * angle_step
            x = np.cos(angle) * pitch
            y = np.sin(angle) * pitch
            positions.append((x, y, 0.0))
            ids.append(f"port_{i+1}")

    elif arrangement == "custom":
        if port_config is None or "positions" not in port_config:
            raise ValueError(
                "Custom arrangement requires port_config with 'positions' key."
            )
        custom_positions = port_config["positions"]
        if len(custom_positions) < actual_n_ports:
            raise ValueError(
                f"Custom arrangement requires at least {actual_n_ports} positions, "
                f"got {len(custom_positions)}."
            )
        positions = [
            tuple(pos) if len(pos) == 3 else (*pos, 0.0)
            for pos in custom_positions[:actual_n_ports]
        ]
        ids = [f"port_{i+1}" for i in range(actual_n_ports)]
    else:
        raise ValueError(
            f"Unknown arrangement: {arrangement}. "
            f"Choose from: linear, grid_2x2, grid_4x4, circular, custom"
        )

    # Create Port objects
    ports = []
    for i in range(actual_n_ports):
        pos = positions[i] if i < len(positions) else (0.0, 0.0, 0.0)

        # Get port-specific config or defaults
        shape = default_shape
        area = default_area
        voltage_range = default_voltage_range
        frequency_range = default_frequency_range
        read_amplitude = default_read_amplitude

        # Apply config overrides if available
        if port_config is not None:
            # shape: single value applies to all ports, list applies per port
            if "shape" in port_config:
                if isinstance(port_config["shape"], (list, tuple)) and i < len(port_config["shape"]):
                    shape = port_config["shape"][i]
                else:
                    shape = port_config["shape"]
            # area: single value applies to all ports, list applies per port
            if "area" in port_config:
                if isinstance(port_config["area"], (list, tuple)) and i < len(port_config["area"]):
                    area = port_config["area"][i]
                else:
                    area = port_config["area"]
            if "voltage_range" in port_config:
                vr = port_config["voltage_range"]
                if isinstance(vr, (list, tuple)) and i < len(vr):
                    voltage_range = tuple(vr[i])
                elif isinstance(vr, (int, float)):
                    voltage_range = (-vr, vr)
            if "frequency_range" in port_config:
                fr = port_config["frequency_range"]
                if isinstance(fr, (list, tuple)) and i < len(fr):
                    frequency_range = tuple(fr[i])
                elif isinstance(fr, (int, float)):
                    frequency_range = (1e3, fr)
            if "read_amplitude" in port_config:
                ra = port_config["read_amplitude"]
                if isinstance(ra, (list, tuple)) and i < len(ra):
                    read_amplitude = ra[i]
                elif isinstance(ra, (int, float)):
                    read_amplitude = ra

        port = Port(
            id=ids[i],
            position=pos,
            shape=shape,
            area=area,
            write_voltage_range=voltage_range,
            read_frequency_range=frequency_range,
            read_amplitude=read_amplitude,
        )
        ports.append(port)

    return ElectrodeArray(ports)


# Convenience functions for all electrode operations
def simultaneous_write(
    array: ElectrodeArray, port_voltages: dict[str, float], duration: float
) -> dict[str, TrackedValue]:
    """Apply write voltages to ports and update material state.

    See ElectrodeArray.simultaneous_write for full docstring.
    """
    return array.simultaneous_write(port_voltages, duration)


def independent_read(
    array: ElectrodeArray, port_frequencies: dict[str, float]
) -> dict[str, TrackedValue]:
    """Read impedance per port at specified frequencies.

    See ElectrodeArray.independent_read for full docstring.
    """
    return array.independent_read(port_frequencies)


def write_read_interference(
    array: ElectrodeArray, write_ports: list[str], read_ports: list[str]
) -> np.ndarray:
    """Compute crosstalk matrix between write and read ports.

    See ElectrodeArray.write_read_interference for full docstring.
    """
    return array.write_read_interference(write_ports, read_ports)


def cross_port_coupling(
    array: ElectrodeArray, port_i: str, port_j: str
) -> float:
    """Compute coupling coefficient between two ports.

    See ElectrodeArray.cross_port_coupling for full docstring.
    """
    return array.cross_port_coupling(port_i, port_j)


def frequency_dependent_impedance(
    array: ElectrodeArray, frequencies: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute impedance per port across an array of frequencies.

    See ElectrodeArray.frequency_dependent_impedance for full docstring.
    """
    return array.frequency_dependent_impedance(frequencies)


def state_retention(
    array: ElectrodeArray, time: float
) -> dict[Port, float]:
    """Compute retention decay per port over time.

    See ElectrodeArray.state_retention for full docstring.
    """
    return array.state_retention(time)


def signal_noise(
    array: ElectrodeArray, bandwidth: float
) -> dict[str, TrackedValue]:
    """Compute noise per port given a measurement bandwidth.

    See ElectrodeArray.signal_noise for full docstring.
    """
    return array.signal_noise(bandwidth)