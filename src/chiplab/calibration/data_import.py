"""Experimental data ingestion for chiplab calibration.

This module provides containers and import functions for multi-modal
experimental measurements used in calibration workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Measurement type enumeration
# ---------------------------------------------------------------------------

class MeasurementType:
    """Enumeration of supported measurement types."""
    VOLTAGE = "voltage"
    CURRENT = "current"
    IMPEDANCE = "impedance"
    FREQUENCY_RESPONSE = "frequency_response"
    TEMPERATURE = "temperature"
    STATE_RETENTION = "state_retention"
    WRITE_READ_TIMING = "write_read_timing"


# ---------------------------------------------------------------------------
# Core measurement containers
# ---------------------------------------------------------------------------

@dataclass
class VoltageMeasurement:
    """Time-series voltage measurement V(t) with metadata.

    Attributes:
        time: Time array in seconds.
        voltage: Voltage array in volts.
        setup: Description of the measurement setup.
        probe: Probe station or characterization setup details.
        bandwidth: Bandwidth limit in Hz (optional).
    """
    time: np.ndarray
    voltage: np.ndarray
    setup: str = ""
    probe: str = ""
    bandwidth: Optional[float] = None  # Hz

    @property
    def duration(self) -> float:
        """Total measurement duration in seconds."""
        return float(np.ptp(self.time))

    @property
    def dt(self) -> float:
        """Sampling interval in seconds."""
        return float(np.median(np.diff(self.time)))


@dataclass
class CurrentMeasurement:
    """Time-series current measurement I(t) with transimpedance amp calibration.

    Attributes:
        time: Time array in seconds.
        current: Current array in amps.
        transimpedance: Transimpedance amplifier gain in V/A.
        bandwidth: Measurement bandwidth in Hz (optional).
        feedback_resistance: Feedback resistor value in ohms (optional).
    """
    time: np.ndarray
    current: np.ndarray
    transimpedance: float = 1e6  # V/A, default 1 MΩ
    bandwidth: Optional[float] = None  # Hz
    feedback_resistance: Optional[float] = None  # Ω

    @property
    def voltage_equivalent(self) -> np.ndarray:
        """Current converted to equivalent voltage via transimpedance."""
        return self.current * self.transimpedance


@dataclass
class ImpedanceMeasurement:
    """Complex impedance Z(f) from frequency sweep metadata.

    Attributes:
        frequency: Frequency array in Hz.
        real_part: Real part of impedance in ohms.
        imag_part: Imaginary part of impedance in ohms.
        magnitude: Magnitude |Z| in ohms (computed).
        phase: Phase angle in radians (computed).
        setup: Description of the impedance measurement setup.
    """
    frequency: np.ndarray
    real_part: np.ndarray
    imag_part: np.ndarray
    setup: str = ""
    _magnitude: Optional[np.ndarray] = None
    _phase: Optional[np.ndarray] = None

    @property
    def magnitude(self) -> np.ndarray:
        """Magnitude |Z| in ohms."""
        if self._magnitude is None:
            self._magnitude = np.sqrt(self.real_part**2 + self.imag_part**2)
        return self._magnitude

    @property
    def phase(self) -> np.ndarray:
        """Phase angle in radians."""
        if self._phase is None:
            self._phase = np.arctan2(self.imag_part, self.real_part)
        return self._phase


@dataclass
class FrequencyResponse:
    """Magnitude/phase vs frequency for Bode-plot style data.

    Attributes:
        frequency: Frequency array in Hz.
        magnitude: Magnitude in dB or linear (context-dependent).
        phase: Phase in degrees or radians (context-dependent).
        reference: Optional reference frequency response for normalization.
        comment: Free-text comment.
    """
    frequency: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray
    reference: Optional[FrequencyResponse] = None
    comment: str = ""


@dataclass
class TemperatureMeasurement:
    """Temperature measurement T(t) or spatial T(x,y,z).

    Attributes:
        time: Time array in seconds (optional, None for spatial only).
        temperature: Temperature array.
        unit: Temperature unit, "C" or "K".
        location: Spatial coordinates if measurement is spatial, shape (N, 3).
        location_unit: Unit of spatial coordinates, "m" or "mm" (optional).
    """
    time: Optional[np.ndarray] = None
    temperature: np.ndarray = field(default_factory=lambda: np.array([]))
    unit: str = "C"
    location: Optional[np.ndarray] = None
    location_unit: Optional[str] = None  # "m" or "mm"

    @property
    def is_spatial(self) -> bool:
        """True if this is a spatial temperature map."""
        return self.location is not None and self.location.shape[1] == 3

    @property
    def t(self) -> Optional[np.ndarray]:
        """Return time array if available."""
        return self.time


@dataclass
class StateRetentionMeasurement:
    """State retention vs time and temperature.

    Attributes:
        time: Time array in seconds.
        state: State values (e.g., resistance, threshold voltage).
        temperature: Temperature at each time point in Kelvin or Celsius.
        temperature_unit: Unit of temperature, "C" or "K".
        init_state: Initial state value before retention measurement.
        final_state: Final state value after retention measurement.
    """
    time: np.ndarray
    state: np.ndarray
    temperature: np.ndarray
    temperature_unit: str = "C"
    init_state: Optional[float] = None
    final_state: Optional[float] = None

    @property
    def retention_ratio(self) -> np.ndarray:
        """Ratio of state change: (state - init) / (final - init)."""
        if self.init_state is None or self.final_state is None:
            return np.zeros_like(self.state)
        return (self.state - self.init_state) / (self.final_state - self.init_state)


@dataclass
class WriteReadTiming:
    """Write-read timing pulse parameters and measured response.

    Attributes:
        pulse_width: Pulse width in seconds.
        rise_time: Rise time in seconds (optional).
        fall_time: Fall time in seconds (optional).
        measured_response: Measured voltage or current response array.
        response_time: Time array for measured response in seconds.
        setup: Description of the write-read setup.
    """
    pulse_width: float
    rise_time: Optional[float] = None
    fall_time: Optional[float] = None
    measured_response: Optional[np.ndarray] = None
    response_time: Optional[np.ndarray] = None
    setup: str = ""


# ---------------------------------------------------------------------------
# DataValidator - check units, ranges, consistency, flag anomalies
# ---------------------------------------------------------------------------

class DataValidator:
    """Validate experimental measurement data for units, ranges, and consistency.

    This checker runs systematic validation on any measurement container,
    flagging anomalies and returning a validation report.

    Attributes:
        strict: If True, raise errors on anomalies; if False, return warnings.
    """

    def __init__(self, strict: bool = False):
        self.strict = strict
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def validate(self, measurement: Any) -> Dict[str, List[str]]:
        """Run validation on a measurement object.

        Args:
            measurement: Any measurement dataclass instance.

        Returns:
            Dict with 'warnings' and 'errors' lists.
        """
        self.warnings = []
        self.errors = []
        result = {"warnings": self.warnings, "errors": self.errors}

        if isinstance(measurement, VoltageMeasurement):
            self._validate_voltage(measurement)
        elif isinstance(measurement, CurrentMeasurement):
            self._validate_current(measurement)
        elif isinstance(measurement, ImpedanceMeasurement):
            self._validate_impedance(measurement)
        elif isinstance(measurement, FrequencyResponse):
            self._validate_frequency_response(measurement)
        elif isinstance(measurement, TemperatureMeasurement):
            self._validate_temperature(measurement)
        elif isinstance(measurement, StateRetentionMeasurement):
            self._validate_state_retention(measurement)
        elif isinstance(measurement, WriteReadTiming):
            self._validate_write_read_timing(measurement)
        else:
            self.errors.append(f"Unknown measurement type: {type(measurement)}")

        return result

    def _check_range(self, name: str, values: np.ndarray, lo: float, hi: float,
                     strict_msg: str, warning_msg: str) -> None:
        """Check that values fall within [lo, hi]."""
        bad = np.where((values < lo) | (values > hi))[0]
        if len(bad) > 0:
            if self.strict:
                raise ValueError(f"{name}[{bad[:3]}]: {strict_msg}")
            self.errors.append(f"{name}: {warning_msg} (found {len(bad)} out-of-range)")

    def _check_nan_inf(self, name: str, values: np.ndarray) -> None:
        """Check for NaN and Inf values."""
        if np.any(np.isnan(values)):
            n = int(np.sum(np.isnan(values)))
            if self.strict:
                raise ValueError(f"{name}: {n} NaN values found")
            self.errors.append(f"{name}: {n} NaN values detected")
        if np.any(np.isinf(values)):
            n = int(np.sum(np.isinf(values)))
            if self.strict:
                raise ValueError(f"{name}: {n} Inf values found")
            self.errors.append(f"{name}: {n} Inf values detected")

    def _check_monotonic(self, name: str, values: np.ndarray, increasing: bool = True) -> None:
        """Check that time or frequency values are monotonically increasing."""
        if len(values) < 2:
            return
        diffs = np.diff(values)
        if increasing:
            if np.any(diffs <= 0):
                n = int(np.sum(diffs <= 0))
                self.errors.append(f"{name}: {n} non-increasing intervals detected")
        else:
            if np.any(diffs >= 0):
                n = int(np.sum(diffs >= 0))
                self.errors.append(f"{name}: {n} non-decreasing intervals detected")

    # --- Validation methods per type ---

    def _validate_voltage(self, m: VoltageMeasurement) -> None:
        self._check_monotonic("time", m.time)
        self._check_range("voltage", m.voltage, -1e6, 1e6,
                          "Voltage out of reasonable range",
                          f"Voltage values out of range: min={m.voltage.min():.3g}, max={m.voltage.max():.3g}")
        self._check_nan_inf("voltage", m.voltage)

    def _validate_current(self, m: CurrentMeasurement) -> None:
        self._check_monotonic("time", m.time)
        self._check_nan_inf("current", m.current)
        # Check transimpedance is positive
        if m.transimpedance <= 0:
            self.errors.append(f"transimpedance must be positive, got {m.transimpedance}")

    def _validate_impedance(self, m: ImpedanceMeasurement) -> None:
        self._check_monotonic("frequency", m.frequency)
        self._check_nan_inf("real_part", m.real_part)
        self._check_nan_inf("imag_part", m.imag_part)
        # Check magnitude and phase are consistent
        expected_mag = np.sqrt(m.real_part**2 + m.imag_part**2)
        if not np.allclose(expected_mag, m.magnitude, rtol=1e-10):
            self.errors.append("Impedance magnitude inconsistent with real/imag parts")

    def _validate_frequency_response(self, m: FrequencyResponse) -> None:
        self._check_nan_inf("magnitude", m.magnitude)
        self._check_nan_inf("phase", m.phase)
        self._check_monotonic("frequency", m.frequency)

    def _validate_temperature(self, m: TemperatureMeasurement) -> None:
        self._check_nan_inf("temperature", m.temperature)
        if m.unit not in ("C", "K"):
            self.errors.append(f"Temperature unit must be 'C' or 'K', got '{m.unit}'")
        if m.is_spatial and m.location is not None:
            if m.location.shape[1] != 3:
                self.errors.append(f"Spatial location must have 3 columns, got {m.location.shape[1]}")

    def _validate_state_retention(self, m: StateRetentionMeasurement) -> None:
        self._check_monotonic("time", m.time)
        self._check_nan_inf("state", m.state)
        self._check_nan_inf("temperature", m.temperature)
        if len(m.state) != len(m.time):
            self.errors.append(f"state and time must have same length: {len(m.state)} vs {len(m.time)}")

    def _validate_write_read_timing(self, m: WriteReadTiming) -> None:
        if m.pulse_width <= 0:
            self.errors.append(f"pulse_width must be positive, got {m.pulse_width}")
        self._check_nan_inf("measured_response", m.measured_response if m.measured_response is not None else np.array([]))