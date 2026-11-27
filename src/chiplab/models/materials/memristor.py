"""Memristor (ReRAM) material physics models.

Models for memristor resistive switching, write/read pulses, and
crossbar vector-matrix multiplication with GPU acceleration.
"""

from __future__ import annotations

from typing import Any

import json

import numpy as np

from chiplab.schemas.common import Provenance, TrackedValue
from chiplab.core.gpu import GPUBackend


class MemristorState:
    """State of a memristor (ReRAM) device.

    Tracks resistance, internal state variable, voltage history,
    and switching dynamics for resistive RAM modeling.
    All scalar values use TrackedValue with Provenance tracking.
    """

    def __init__(
        self,
        resistance: TrackedValue | float = 10000.0,
        state: TrackedValue | float = 0.5,
        voltage_history: TrackedValue | list | None = None,
        current_history: TrackedValue | list | None = None,
        conductance: TrackedValue | float | None = None,
    ):
        """Initialize MemristorState with flexible input types (TrackedValue or float).

        Parameters
        ----------
        resistance : float or TrackedValue, optional
            Device resistance in ohms (default 10 kΩ).
        state : float or TrackedValue, optional
            Internal memristor state variable w in [0, 1] (default 0.5).
            w=0: high resistance state (HRS), w=1: low resistance state (LRS).
        voltage_history : list or TrackedValue, optional
            History of applied voltages. Stored as JSON string internally.
        current_history : list or TrackedValue, optional
            History of measured currents. Stored as JSON string internally.
        conductance : float or TrackedValue, optional
            Device conductance in siemens (auto-computed from resistance).
        """
        # Handle resistance
        if isinstance(resistance, TrackedValue):
            self.resistance = resistance
        else:
            self.resistance = TrackedValue(
                value=float(resistance), unit="Ω", provenance=Provenance.ASSUMED
            )

        # Handle state variable w in [0, 1]
        if isinstance(state, TrackedValue):
            self.state = state
        else:
            self.state = TrackedValue(
                value=float(state), unit="normalized", provenance=Provenance.ASSUMED
            )

        # Ensure state is clamped to [0, 1]
        self.state.value = max(0.0, min(1.0, self.state.value))

        # Handle voltage history
        # Store as JSON string internally for TrackedValue compatibility
        if voltage_history is None:
            self.voltage_history = TrackedValue(
                value=json.dumps([]), unit="V", provenance=Provenance.ASSUMED
            )
        elif isinstance(voltage_history, TrackedValue):
            self.voltage_history = voltage_history
        else:
            self.voltage_history = TrackedValue(
                value=json.dumps(list(voltage_history)), unit="V",
                provenance=Provenance.ASSUMED
            )

        # Handle current history
        # Store as JSON string internally for TrackedValue compatibility
        if current_history is None:
            self.current_history = TrackedValue(
                value=json.dumps([]), unit="A", provenance=Provenance.ASSUMED
            )
        elif isinstance(current_history, TrackedValue):
            self.current_history = current_history
        else:
            self.current_history = TrackedValue(
                value=json.dumps(list(current_history)), unit="A",
                provenance=Provenance.ASSUMED
            )

        # Handle conductance (auto-computed if not provided)
        if conductance is None:
            if isinstance(self.resistance.value, (int, float)) and self.resistance.value > 0:
                self.conductance = TrackedValue(
                    value=1.0 / self.resistance.value,
                    unit="S",
                    provenance=Provenance.ASSUMED,
                )
            else:
                self.conductance = TrackedValue(
                    value=0.0, unit="S", provenance=Provenance.ASSUMED
                )
        elif isinstance(conductance, TrackedValue):
            self.conductance = conductance
        else:
            self.conductance = TrackedValue(
                value=float(conductance), unit="S", provenance=Provenance.ASSUMED
            )

    def copy(self) -> MemristorState:
        """Create a deep copy of the state."""
        # Convert internal JSON strings back to lists for the copy
        vhist = list(json.loads(self.voltage_history.value)) if self.voltage_history.value else []
        chist = list(json.loads(self.current_history.value)) if self.current_history.value else []

        return MemristorState(
            resistance=TrackedValue(
                value=self.resistance.value,
                unit=self.resistance.unit,
                provenance=self.resistance.provenance,
                source=self.resistance.source,
                uncertainty_pct=self.resistance.uncertainty_pct,
                notes=self.resistance.notes,
            ),
            state=TrackedValue(
                value=self.state.value,
                unit=self.state.unit,
                provenance=self.state.provenance,
                source=self.state.source,
                uncertainty_pct=self.state.uncertainty_pct,
                notes=self.state.notes,
            ),
            voltage_history=TrackedValue(
                value=json.dumps(vhist),
                unit=self.voltage_history.unit,
                provenance=self.voltage_history.provenance,
                source=self.voltage_history.source,
                uncertainty_pct=self.voltage_history.uncertainty_pct,
                notes=self.voltage_history.notes,
            ),
            current_history=TrackedValue(
                value=json.dumps(chist),
                unit=self.current_history.unit,
                provenance=self.current_history.provenance,
                source=self.current_history.source,
                uncertainty_pct=self.current_history.uncertainty_pct,
                notes=self.current_history.notes,
            ),
            conductance=TrackedValue(
                value=self.conductance.value,
                unit=self.conductance.unit,
                provenance=self.conductance.provenance,
                source=self.conductance.source,
                uncertainty_pct=self.conductance.uncertainty_pct,
                notes=self.conductance.notes,
            ),
        )


def memristor_iv_step(
    state: MemristorState,
    voltage: float | TrackedValue,
    dt: float | TrackedValue = 1e-9,
) -> tuple[MemristorState, float, float]:
    """Compute one I-V step for a memristor device.

    Models the nonlinear I-V characteristic and internal state evolution
    of a memristor based on the HP/RST model.

    Parameters
    ----------
    state : MemristorState
        Current memristor state.
    voltage : float or TrackedValue
        Applied voltage in volts.
    dt : float or TrackedValue, optional
        Time step in seconds (default 1 ns).

    Returns
    -------
    new_state : MemristorState
        Updated memristor state after the I-V step.
    current : float
        Current in amps through the device.
    conductance : float
        Differential conductance in siemens.
    """
    # Unwrap voltage
    v = float(voltage) if not isinstance(voltage, TrackedValue) else voltage.value

    # Create new state from copy
    new_state = state.copy()

    # Memristor state evolution (HP model simplified)
    # dw/dt = A * I * (1 - w^p) - B * v * w^q
    # or simplified: dw/dt = (eta * I / (R_on * R_off)) * (1 - w) * f(v)

    # Get current conductance
    G = float(new_state.conductance)
    R_on = 100.0  # low resistance state in ohms
    R_off = 10000.0  # high resistance state in ohms

    # Current from Ohm's law: I = V / R(w)
    # R(w) = R_off - (R_off - R_on) * w
    R_w = R_off - (R_off - R_on) * float(new_state.state)
    I = v / max(R_w, 1e-12)  # current in amps

    # State evolution
    # Simplified: dw/dt proportional to |I| * (1 - |w|)
    eta = 0.1  # scaling factor
    dw_dt = eta * abs(I) * (1.0 - float(new_state.state))

    # Update state
    new_w = float(new_state.state) + dw_dt * float(dt)
    new_w = max(0.0, min(1.0, new_w))  # clamp to [0, 1]
    new_state.state = TrackedValue(
        value=new_w,
        unit="normalized",
        provenance=new_state.state.provenance,
        source="memristor_iv_step",
    )

    # Update resistance based on new state
    # R(w) = R_off - (R_off - R_on) * w
    new_R = R_off - (R_off - R_on) * new_w
    new_state.resistance = TrackedValue(
        value=new_R,
        unit="Ω",
        provenance=new_state.resistance.provenance,
        source="memristor_iv_step",
    )
    new_state.conductance = TrackedValue(
        value=1.0 / max(new_R, 1e-12),
        unit="S",
        provenance=new_state.conductance.provenance,
        source="memristor_iv_step",
    )

    # Record voltage and current (store as JSON strings)
    vhist_val = json.loads(new_state.voltage_history.value) if isinstance(new_state.voltage_history.value, str) else new_state.voltage_history.value
    chist_val = json.loads(new_state.current_history.value) if isinstance(new_state.current_history.value, str) else new_state.current_history.value

    new_state.voltage_history = TrackedValue(
        value=json.dumps(vhist_val + [v]) if vhist_val else json.dumps([v]),
        unit="V",
        provenance=new_state.voltage_history.provenance,
        source="memristor_iv_step",
    )
    new_state.current_history = TrackedValue(
        value=json.dumps(chist_val + [I]) if chist_val else json.dumps([I]),
        unit="A",
        provenance=new_state.current_history.provenance,
        source="memristor_iv_step",
    )

    # Compute current and conductance
    current = I
    conductance = 1.0 / max(new_R, 1e-12)

    return new_state, current, conductance


def memristor_write_pulse(
    state: MemristorState,
    voltage: float | TrackedValue,
    duration: float | TrackedValue,
    pulse_type: str = "unipolar",
) -> MemristorState:
    """Apply a write pulse to a memristor device.

    Models the programming of a memristor via voltage pulses,
    switching between HRS and LRS depending on pulse polarity and duration.

    Parameters
    ----------
    state : MemristorState
        Current memristor state.
    voltage : float or TrackedValue
        Pulse voltage in volts.
    duration : float or TrackedValue
        Pulse duration in seconds.
    pulse_type : str, optional
        Type of pulse: "unipolar" or "bipolar" (default "unipolar").

    Returns
    -------
    new_state : MemristorState
        Updated memristor state after the write pulse.
    """
    # Unwrap values
    v = float(voltage) if not isinstance(voltage, TrackedValue) else voltage.value
    d = float(duration) if not isinstance(duration, TrackedValue) else duration.value

    new_state = state.copy()

    if pulse_type == "unipolar":
        # Unipolar switching: magnitude of voltage determines switching
        # Higher magnitude / longer duration -> more switching
        switch_strength = abs(v) * d / 1e-6  # normalize to 1V·s

        # State change proportional to switch strength
        dw = 0.01 * switch_strength  # 0.01 per V·s normalized

        # New state
        new_w = float(new_state.state) + dw
        new_w = max(0.0, min(1.0, new_w))
        new_state.state = TrackedValue(
            value=new_w,
            unit="normalized",
            provenance=new_state.state.provenance,
            source="memristor_write_pulse",
        )

    elif pulse_type == "bipolar":
        # Bipolar switching: polarity determines direction
        if v > 0:
            # Positive voltage -> SET (decrease resistance, increase w)
            dw = 0.02 * abs(v) * d / 1e-6
        else:
            # Negative voltage -> RESET (increase resistance, decrease w)
            dw = -0.02 * abs(v) * d / 1e-6

        new_w = float(new_state.state) + dw
        new_w = max(0.0, min(1.0, new_w))
        new_state.state = TrackedValue(
            value=new_w,
            unit="normalized",
            provenance=new_state.state.provenance,
            source="memristor_write_pulse",
        )

    # Update resistance based on new state
    R_on = 100.0
    R_off = 10000.0
    new_R = R_off - (R_off - R_on) * float(new_state.state)
    new_state.resistance = TrackedValue(
        value=new_R,
        unit="Ω",
        provenance=new_state.resistance.provenance,
        source="memristor_write_pulse",
    )
    new_state.conductance = TrackedValue(
        value=1.0 / max(new_R, 1e-12),
        unit="S",
        provenance=new_state.conductance.provenance,
        source="memristor_write_pulse",
    )

    # Record pulse (store as JSON strings)
    vhist_val = json.loads(new_state.voltage_history.value) if isinstance(new_state.voltage_history.value, str) else new_state.voltage_history.value
    chist_val = json.loads(new_state.current_history.value) if isinstance(new_state.current_history.value, str) else new_state.current_history.value

    new_state.voltage_history = TrackedValue(
        value=json.dumps(vhist_val + [v]) if vhist_val else json.dumps([v]),
        unit="V",
        provenance=new_state.voltage_history.provenance,
        source="memristor_write_pulse",
    )
    new_state.current_history = TrackedValue(
        value=json.dumps(chist_val + [v / max(new_R, 1e-12)]) if chist_val else json.dumps([v / max(new_R, 1e-12)]),
        unit="A",
        provenance=new_state.current_history.provenance,
        source="memristor_write_pulse",
    )

    return new_state


def memristor_crossbar_vmm(
    conductances: np.ndarray,
    inputs: np.ndarray | None = None,
    device: str = "auto",
) -> tuple[np.ndarray, MemristorState]:
    """Perform crossbar vector-matrix multiply using memristor conductances.

    Models a crossbar array where conductances are stored in memristors.
    Performs Y = G @ X where G is the conductance matrix and X is the input vector.

    GPU acceleration is available via GPUBackend with automatic CPU fallback.

    Parameters
    ----------
    conductances : np.ndarray
        2D array of shape (n_outputs, n_inputs) with memristor conductances in S.
    inputs : np.ndarray, optional
        1D input vector of shape (n_inputs,). If None, uses ones vector.
    device : str, optional
        Device selection: "auto", "cuda", "gpu", or "cpu". Default "auto".

    Returns
    -------
    outputs : np.ndarray
        1D output vector of shape (n_outputs,) with currents in A.
    state : MemristorState
        Aggregate memristor state summarizing the crossbar condition.
    """
    # Determine device/backend
    backend = GPUBackend(prefer=device, device=device)

    # Set up inputs
    if inputs is None:
        inputs = np.ones(conductances.shape[1], dtype=np.float64)
    inputs = np.asarray(inputs, dtype=np.float64).flatten()

    # Validate shapes
    n_inputs = conductances.shape[1]
    n_outputs = conductances.shape[0]
    if len(inputs) != n_inputs:
        raise ValueError(
            f"Input vector size {len(inputs)} does not match conductance matrix columns {n_inputs}"
        )

    # Perform matrix multiply: outputs = conductances @ inputs
    # Use backend's batch_matmul or appropriate method
    if backend.backend_name in ("torch_cuda", "torch_cpu"):
        # PyTorch path
        import torch

        G_t = backend.to_device(conductances, dtype="float64")
        x_t = backend.to_device(inputs, dtype="float64")
        # Reshape inputs for batched matmul: (1, n_inputs) -> (n_outputs, n_inputs) @ (n_inputs, 1) -> (n_outputs,)
        x_reshaped = x_t.unsqueeze(-1)  # (n_inputs, 1)
        y_t = torch.matmul(G_t, x_reshaped)
        outputs = backend.to_numpy(y_t.squeeze(-1))  # (n_outputs,)
    elif backend.backend_name == "cupy":
        import cupy as cp

        G_c = backend.to_device(conductances)
        x_c = backend.to_device(inputs)
        y_c = cp.matmul(G_c, x_c)
        outputs = cp.asnumpy(y_c)
    else:
        # CPU NumPy path
        outputs = conductances @ inputs

    # Create aggregate memristor state
    # Summary: average conductance, resistance range, and total conductance
    avg_G = float(np.mean(conductances))
    total_G = float(np.sum(conductances))
    R_avg = 1.0 / max(avg_G, 1e-12)

    state = MemristorState(
        resistance=TrackedValue(value=R_avg, unit="Ω"),
        state=TrackedValue(value=0.5, unit="normalized"),
        conductance=TrackedValue(value=avg_G, unit="S"),
    )

    return outputs, state


# Helper function for average resistance from conductances
def R_avg_from_conductances(conductances: np.ndarray) -> float:
    """Compute average resistance from a conductance matrix."""
    G_avg = float(np.mean(conductances))
    return 1.0 / max(G_avg, 1e-12)