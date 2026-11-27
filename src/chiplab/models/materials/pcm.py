"""Phase-Change Memory (PCM) material physics models.

Models for PCM crystallization, melt-quench programming, and resistance drift.
"""

from __future__ import annotations

from typing import Any

import json

import numpy as np

from chiplab.schemas.common import Provenance, TrackedValue
from chiplab.core.gpu import GPUBackend


class PCMState:
    """State of a Phase-Change Memory (PCM) device.

    Tracks phase, resistance, temperature, and switching history
    for phase-change RAM modeling. All scalar values use TrackedValue
    with Provenance tracking.
    """

    def __init__(
        self,
        resistance: TrackedValue | float = 10000.0,
        phase: TrackedValue | str = "crystalline",
        temperature: TrackedValue | float = 300.0,
        crystallization_fraction: TrackedValue | float = 1.0,
        reset_pulse_count: TrackedValue | int = 0,
        set_pulse_count: TrackedValue | int = 0,
    ):
        """Initialize PCMState with flexible input types (TrackedValue or float).

        Parameters
        ----------
        resistance : float or TrackedValue, optional
            Device resistance in ohms (default 10 kΩ, amorphous HRS).
        phase : str or TrackedValue, optional
            Current phase: "amorphous" or "crystalline" (default "crystalline").
        temperature : float or TrackedValue, optional
            Device temperature in Kelvin (default 300 K).
        crystallization_fraction : float or TrackedValue, optional
            Fraction of crystalline material in [0, 1] (default 1.0).
            0 = fully amorphous, 1 = fully crystalline.
        reset_pulse_count : int or TrackedValue, optional
            Number of reset pulses applied (default 0).
        set_pulse_count : int or TrackedValue, optional
            Number of set pulses applied (default 0).
        """
        # Handle resistance
        if isinstance(resistance, TrackedValue):
            self.resistance = resistance
        else:
            self.resistance = TrackedValue(
                value=float(resistance), unit="Ω", provenance=Provenance.ASSUMED
            )

        # Handle phase
        if isinstance(phase, TrackedValue):
            self.phase = phase
        else:
            self.phase = TrackedValue(
                value=str(phase), unit="phase", provenance=Provenance.ASSUMED
            )

        # Handle temperature
        if isinstance(temperature, TrackedValue):
            self.temperature = temperature
        else:
            self.temperature = TrackedValue(
                value=float(temperature), unit="K", provenance=Provenance.ASSUMED
            )

        # Handle crystallization fraction
        if isinstance(crystallization_fraction, TrackedValue):
            self.crystallization_fraction = crystallization_fraction
        else:
            self.crystallization_fraction = TrackedValue(
                value=float(crystallization_fraction),
                unit="fraction",
                provenance=Provenance.ASSUMED,
            )
        # Clamp to [0, 1]
        self.crystallization_fraction.value = max(0.0, min(1.0, self.crystallization_fraction.value))

        # Handle reset pulse count
        if isinstance(reset_pulse_count, TrackedValue):
            self.reset_pulse_count = reset_pulse_count
        else:
            self.reset_pulse_count = TrackedValue(
                value=float(reset_pulse_count),
                unit="pulses",
                provenance=Provenance.ASSUMED,
            )

        # Handle set pulse count
        if isinstance(set_pulse_count, TrackedValue):
            self.set_pulse_count = set_pulse_count
        else:
            self.set_pulse_count = TrackedValue(
                value=float(set_pulse_count),
                unit="pulses",
                provenance=Provenance.ASSUMED,
            )

        # Handle voltage history (stored as JSON string for TrackedValue compatibility)
        self.voltage_history = TrackedValue(
            value=json.dumps([]), unit="s", provenance=Provenance.ASSUMED
        )

    def copy(self) -> PCMState:
        """Create a deep copy of the state."""
        # Create new state - voltage_history will be set inside __init__
        new_state = PCMState(
            resistance=TrackedValue(
                value=self.resistance.value,
                unit=self.resistance.unit,
                provenance=self.resistance.provenance,
                source=self.resistance.source,
                uncertainty_pct=self.resistance.uncertainty_pct,
                notes=self.resistance.notes,
            ),
            phase=TrackedValue(
                value=self.phase.value,
                unit=self.phase.unit,
                provenance=self.phase.provenance,
                source=self.phase.source,
                uncertainty_pct=self.phase.uncertainty_pct,
                notes=self.phase.notes,
            ),
            temperature=TrackedValue(
                value=self.temperature.value,
                unit=self.temperature.unit,
                provenance=self.temperature.provenance,
                source=self.temperature.source,
                uncertainty_pct=self.temperature.uncertainty_pct,
                notes=self.temperature.notes,
            ),
            crystallization_fraction=TrackedValue(
                value=self.crystallization_fraction.value,
                unit=self.crystallization_fraction.unit,
                provenance=self.crystallization_fraction.provenance,
                source=self.crystallization_fraction.source,
                uncertainty_pct=self.crystallization_fraction.uncertainty_pct,
                notes=self.crystallization_fraction.notes,
            ),
            reset_pulse_count=TrackedValue(
                value=self.reset_pulse_count.value,
                unit=self.reset_pulse_count.unit,
                provenance=self.reset_pulse_count.provenance,
                source=self.reset_pulse_count.source,
                uncertainty_pct=self.reset_pulse_count.uncertainty_pct,
                notes=self.reset_pulse_count.notes,
            ),
            set_pulse_count=TrackedValue(
                value=self.set_pulse_count.value,
                unit=self.set_pulse_count.unit,
                provenance=self.set_pulse_count.provenance,
                source=self.set_pulse_count.source,
                uncertainty_pct=self.set_pulse_count.uncertainty_pct,
                notes=self.set_pulse_count.notes,
            ),
        )
        # Copy voltage history after init (it was set by __init__ to empty,
        # now we restore the original value)
        new_state.voltage_history = TrackedValue(
            value=self.voltage_history.value,
            unit=self.voltage_history.unit,
            provenance=self.voltage_history.provenance,
            source=self.voltage_history.source,
            uncertainty_pct=self.voltage_history.uncertainty_pct,
            notes=self.voltage_history.notes,
        )
        return new_state


def _update_voltage_history(state: PCMState, new_value) -> PCMState:
    """Helper to update voltage_history with a new value, stored as JSON string."""
    current_val = state.voltage_history.value
    if isinstance(current_val, str):
        current_list = json.loads(current_val)
    else:
        current_list = (
            json.loads(current_val) if isinstance(current_val, (list, np.ndarray)) else []
        )
    new_list = current_list + [new_value]
    state.voltage_history = TrackedValue(
        value=json.dumps(new_list),
        unit=state.voltage_history.unit,
        provenance=state.voltage_history.provenance,
        source=state.voltage_history.source,
        uncertainty_pct=state.voltage_history.uncertainty_pct,
        notes=state.voltage_history.notes,
    )
    return state


def pcm_crystallization_step(
    state: PCMState,
    dt: float | TrackedValue = 1e-6,
    temperature: float | TrackedValue | None = None,
) -> PCMState:
    """Compute one crystallization step using JMAK (Johnson-Mehl-Avrami-Kolmogorov) + Arrhenius.

    Models the temperature-dependent crystallization of amorphous GST material.
    The JMAK equation: X(t) = 1 - exp(-(k*t)^n)
    where k = k0 * exp(-Ea/(kB*T)) is the Arrhenius rate constant.

    Parameters
    ----------
    state : PCMState
        Current PCM state.
    dt : float or TrackedValue, optional
        Time step in seconds (default 1 μs).
    temperature : float or TrackedValue, optional
        Absolute temperature in K. If None, uses state.temperature.

    Returns
    -------
    new_state : PCMState
        Updated PCM state after crystallization step.
    """
    # Unwrap time step
    dt_val = float(dt) if not isinstance(dt, TrackedValue) else dt.value

    # Determine temperature
    if temperature is not None:
        t = float(temperature) if not isinstance(temperature, TrackedValue) else temperature.value
        t_prov = temperature.provenance if isinstance(temperature, TrackedValue) else Provenance.MEASURED
    else:
        t = float(state.temperature.value)
        t_prov = state.temperature.provenance

    new_state = state.copy()

    # JMAK parameters for GST (typical values)
    k0 = 1e12  # pre-exponential factor (1/s)
    Ea = 1.5  # activation energy (eV)
    n = 2.0  # Avrami exponent
    kB = 8.617333262145e-5  # Boltzmann constant in eV/K

    # Arrhenius rate constant
    k = k0 * np.exp(-Ea / (kB * max(t, 1.0)))

    # JMAK crystallization: fraction crystallized increases with time
    # X(t) = 1 - exp(-(k*t)^n)
    # For incremental step:
    kdt = k * dt_val
    # Crystallization increment (derivative approach)
    dX = n * (kdt ** (n - 1.0)) * np.exp(-(kdt ** n)) * dt_val

    # New crystalline fraction
    X_current = float(new_state.crystallization_fraction)
    X_new = X_current + dX * (1.0 - X_current)  # saturation as X -> 1
    X_new = max(0.0, min(1.0, X_new))
    new_state.crystallization_fraction = TrackedValue(
        value=X_new,
        unit="fraction",
        provenance=t_prov,
        source="pcm_crystallization_step",
    )

    # Update phase based on crystallization fraction
    if X_new > 0.5:
        new_phase = "crystalline"
    else:
        new_phase = "amorphous"
    new_state.phase = TrackedValue(
        value=new_phase,
        unit="phase",
        provenance=t_prov,
        source="pcm_crystallization_step",
    )

    # Update resistance based on phase and crystallization fraction
    # R = R_amorphous * (1 - X) + R_crystalline * X, with drift factor
    R_amorphous = 10000.0  # HRS amorphous
    R_crystalline = 1000.0  # LRS crystalline

    # Base resistance from phase mixture
    base_R = R_amorphous * (1.0 - X_new) + R_crystalline * X_new

    # Apply temperature-dependent factor
    temp_factor = t / max(state.temperature.value, 1.0) if state.temperature.value > 0 else 1.0
    new_R = base_R * temp_factor

    new_state.resistance = TrackedValue(
        value=new_R,
        unit="Ω",
        provenance=t_prov,
        source="pcm_crystallization_step",
    )

    # Record in voltage history (time step)
    _update_voltage_history(new_state, dt_val)

    return new_state


def pcm_melt_quench_pulse(
    state: PCMState,
    voltage: float | TrackedValue,
    duration: float | TrackedValue,
    pulse_type: str = "set",
) -> PCMState:
    """Apply a melt-quench pulse to a PCM device.

    Models SET (amorphization) and RESET (crystallization) operations
    via controlled melt and rapid quenching.

    Parameters
    ----------
    state : PCMState
        Current PCM state.
    voltage : float or TrackedValue
        Pulse voltage in volts.
    duration : float or TrackedValue
        Pulse duration in seconds.
    pulse_type : str, optional
        "set" for amorphization (fast quench) or "reset" for crystallization
        (slow quench). Default "set".

    Returns
    -------
    new_state : PCMState
        Updated PCM state after the melt-quench pulse.
    """
    # Unwrap values
    v = float(voltage) if not isinstance(voltage, TrackedValue) else voltage.value
    d = float(duration) if not isinstance(duration, TrackedValue) else duration.value

    new_state = state.copy()

    if pulse_type == "set":
        # SET operation: amorphization (melt + ultra-fast quench)
        # High voltage, short duration -> amorphous state (high resistance)
        # Crystallization fraction decreases -> more amorphous

        # Melt energy proportional to voltage and duration
        melt_energy = abs(v) * d

        # Crystallization fraction decreases (more amorphous)
        dX = -0.1 * melt_energy / 1e-6  # normalized

        new_X = float(new_state.crystallization_fraction) + dX
        new_X = max(0.0, min(1.0, new_X))
        new_state.crystallization_fraction = TrackedValue(
            value=new_X,
            unit="fraction",
            provenance=new_state.crystallization_fraction.provenance,
            source="pcm_melt_quench_pulse",
        )

        # Update phase to amorphous if mostly amorphous
        if new_X < 0.5:
            new_state.phase = TrackedValue(
                value="amorphous",
                unit="phase",
                provenance=new_state.phase.provenance,
                source="pcm_melt_quench_pulse",
            )

        # Resistance increases dramatically (amorphous HRS)
        R_amorphous = 10000.0
        new_state.resistance = TrackedValue(
            value=R_amorphous,
            unit="Ω",
            provenance=new_state.resistance.provenance,
            source="pcm_melt_quench_pulse",
        )

        # Pulse count
        new_state.set_pulse_count = TrackedValue(
            value=float(new_state.set_pulse_count.value) + 1,
            unit="pulses",
            provenance=new_state.set_pulse_count.provenance,
            source="pcm_melt_quench_pulse",
        )

    elif pulse_type == "reset":
        # RESET operation: crystallization (melt + slow quench)
        # Moderate voltage, longer duration -> crystalline state (low resistance)
        # Crystallization fraction increases -> more crystalline

        # Melt energy proportional to voltage and duration
        melt_energy = abs(v) * d

        # Crystallization fraction increases (more crystalline)
        dX = 0.05 * melt_energy / 1e-6  # normalized

        new_X = float(new_state.crystallization_fraction) + dX
        new_X = max(0.0, min(1.0, new_X))
        new_state.crystallization_fraction = TrackedValue(
            value=new_X,
            unit="fraction",
            provenance=new_state.crystallization_fraction.provenance,
            source="pcm_melt_quench_pulse",
        )

        # Update phase to crystalline if mostly crystalline
        if new_X > 0.5:
            new_state.phase = TrackedValue(
                value="crystalline",
                unit="phase",
                provenance=new_state.phase.provenance,
                source="pcm_melt_quench_pulse",
            )

        # Resistance decreases (crystalline LRS)
        R_crystalline = 1000.0
        new_state.resistance = TrackedValue(
            value=R_crystalline,
            unit="Ω",
            provenance=new_state.resistance.provenance,
            source="pcm_melt_quench_pulse",
        )

        # Pulse count
        new_state.reset_pulse_count = TrackedValue(
            value=float(new_state.reset_pulse_count.value) + 1,
            unit="pulses",
            provenance=new_state.reset_pulse_count.provenance,
            source="pcm_melt_quench_pulse",
        )

    # Record pulse (store as JSON strings)
    _update_voltage_history(new_state, v)

    return new_state


def pcm_resistance_drift(
    state: PCMState,
    dt: float | TrackedValue = 1.0,
) -> PCMState:
    """Compute resistance drift for a PCM device.

    Models the logarithmic resistance drift of the amorphous phase:
    R(t) = R0 * (1 + S * log(t/t0))

    where S is the drift coefficient and t0 is a reference time.

    Parameters
    ----------
    state : PCMState
        Current PCM state.
    dt : float or TrackedValue, optional
        Time step in seconds (default 1 s).

    Returns
    -------
    new_state : PCMState
        Updated PCM state with drifted resistance.
    """
    # Unwrap time step
    dt_val = float(dt) if not isinstance(dt, TrackedValue) else dt.value

    new_state = state.copy()

    # Drift parameters for GST
    S = 0.1  # drift coefficient
    t0 = 1.0  # reference time in seconds

    # Current resistance and time
    R_current = float(new_state.resistance)
    # Effective time since last reset
    t_effective = max(dt_val, t0)

    # Logarithmic drift: R(t) = R0 * (1 + S * log(t/t0))
    drift_factor = 1.0 + S * np.log(t_effective / t0)
    new_R = R_current * drift_factor

    new_state.resistance = TrackedValue(
        value=new_R,
        unit="Ω",
        provenance=new_state.resistance.provenance,
        source="pcm_resistance_drift",
    )

    # Record drift
    _update_voltage_history(new_state, dt_val)

    return new_state