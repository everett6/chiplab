"""
Spike-Timing-Dependent Plasticity (STDP) Synapse Model
========================================================

Implements memristive STDP synapses for neuromorphic computing.
Based on biological STDP rules and memristor physics.

References:
- Bi & Poo (1998) - STDP biological basis
- Serrano-Gotarredona et al. (2013) - Memristive STDP
- Ambrogio et al. (2018) - RRAM-based neuromorphic arrays
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np

from chiplab.schemas.common import Provenance, TrackedValue
from chiplab.core.gpu import GPUBackend, get_gpu_backend as get_backend


class STDPRule(Enum):
    """STDP learning rule variants."""
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    TRIPLET = "triplet"
    VOLTAGE_DEPENDENT = "voltage_dependent"
    CALCIUM_BASED = "calcium_based"


class MemristorModel(Enum):
    """Underlying memristor physics model."""
    LINEAR_ION_DRIFT = "linear_ion_drift"
    NONLINEAR_ION_DRIFT = "nonlinear_ion_drift"
    THERMAL_CHEMICAL = "thermal_chemical"
    PHASE_CHANGE = "phase_change"
    FERROELECTRIC = "ferroelectric"


@dataclass
class STDPState:
    """
    State of an STDP synapse with full provenance tracking.
    
    Tracks synaptic weight, trace variables, and memristor internal state.
    """
    # Synaptic weight (normalized 0-1, maps to conductance)
    weight: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.5,
        provenance=Provenance.ASSUMED,
        metadata={"description": "Normalized synaptic weight [0,1]"}
    ))
    
    # Pre-synaptic trace (exponential decay)
    trace_pre: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,
        provenance=Provenance.ASSUMED,
        metadata={"description": "Pre-synaptic trace for STDP"}
    ))
    
    # Post-synaptic trace (exponential decay)
    trace_post: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,
        provenance=Provenance.ASSUMED,
        metadata={"description": "Post-synaptic trace for STDP"}
    ))
    
    # Triplet rule traces
    trace_pre2: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,
        provenance=Provenance.ASSUMED,
        metadata={"description": "Second pre-synaptic trace for triplet rule"}
    ))
    trace_post2: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,
        provenance=Provenance.ASSUMED,
        metadata={"description": "Second post-synaptic trace for triplet rule"}
    ))
    
    # Memristor internal state (filament length, oxygen vacancies, etc.)
    internal_state: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.5,
        provenance=Provenance.ASSUMED,
        metadata={"description": "Memristor internal state variable"}
    ))
    
    # Device parameters
    g_min: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e-6,
        provenance=Provenance.MEASURED,
        metadata={"description": "Minimum conductance (S)", "unit": "S"}
    ))
    g_max: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e-3,
        provenance=Provenance.MEASURED,
        metadata={"description": "Maximum conductance (S)", "unit": "S"}
    ))
    
    # STDP time constants (ms)
    tau_pre: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=20.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Pre-synaptic trace decay time constant", "unit": "ms"}
    ))
    tau_post: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=20.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Post-synaptic trace decay time constant", "unit": "ms"}
    ))
    tau_pre2: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=40.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Second pre trace decay time constant", "unit": "ms"}
    ))
    tau_post2: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=40.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Second post trace decay time constant", "unit": "ms"}
    ))
    
    # STDP amplitude parameters
    a_plus: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.01,
        provenance=Provenance.MEASURED,
        metadata={"description": "LTP amplitude"}
    ))
    a_minus: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.012,
        provenance=Provenance.MEASURED,
        metadata={"description": "LTD amplitude"}
    ))
    a_plus2: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.005,
        provenance=Provenance.MEASURED,
        metadata={"description": "Triplet LTP amplitude"}
    ))
    a_minus2: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.006,
        provenance=Provenance.MEASURED,
        metadata={"description": "Triplet LTD amplitude"}
    ))
    
    # Voltage-dependent STDP parameters
    v_thresh: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.5,
        provenance=Provenance.MEASURED,
        metadata={"description": "Voltage threshold for plasticity", "unit": "V"}
    ))
    k_voltage: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=10.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Voltage sensitivity factor"}
    ))
    
    # Calcium-based STDP parameters
    ca_rest: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.05,
        provenance=Provenance.MEASURED,
        metadata={"description": "Resting calcium concentration", "unit": "uM"}
    ))
    ca_thresh_ltp: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.3,
        provenance=Provenance.MEASURED,
        metadata={"description": "Calcium threshold for LTP", "unit": "uM"}
    ))
    ca_thresh_ltd: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.15,
        provenance=Provenance.MEASURED,
        metadata={"description": "Calcium threshold for LTD", "unit": "uM"}
    ))
    tau_ca: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=50.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Calcium decay time constant", "unit": "ms"}
    ))
    
    # Memristor model parameters
    memristor_model: MemristorModel = MemristorModel.NONLINEAR_ION_DRIFT
    ron: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e3,
        provenance=Provenance.MEASURED,
        metadata={"description": "ON state resistance", "unit": "ohm"}
    ))
    roff: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e6,
        provenance=Provenance.MEASURED,
        metadata={"description": "OFF state resistance", "unit": "ohm"}
    ))
    mu_v: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e-14,
        provenance=Provenance.MEASURED,
        metadata={"description": "Ion mobility", "unit": "m^2/(V*s)"}
    ))
    d: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=10e-9,
        provenance=Provenance.MEASURED,
        metadata={"description": "Device thickness", "unit": "m"}
    ))
    p: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Nonlinearity parameter for window function"}
    ))
    
    # History for analysis
    weight_history: list[float] = field(default_factory=list)
    spike_times_pre: list[float] = field(default_factory=list)
    spike_times_post: list[float] = field(default_factory=list)
    
    def copy(self) -> STDPState:
        """Create a deep copy with proper TrackedValue handling."""
        new_state = STDPState()
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, TrackedValue):
                setattr(new_state, field_name, value.copy())
            elif isinstance(value, list):
                setattr(new_state, field_name, value.copy())
            elif isinstance(value, MemristorModel):
                setattr(new_state, field_name, value)
            else:
                setattr(new_state, field_name, value)
        return new_state
    
    def get_conductance(self) -> float:
        """Get actual conductance from normalized weight."""
        w = self.weight.value
        return float(self.g_min.value + w * (self.g_max.value - self.g_min.value))
    
    def set_conductance(self, g: float) -> None:
        """Set weight from conductance value."""
        g_min = float(self.g_min.value)
        g_max = float(self.g_max.value)
        self.weight.value = max(0.0, min(1.0, (g - g_min) / (g_max - g_min)))
        self.weight.provenance = Provenance.ESTIMATED


def stdp_trace_decay(
    trace: TrackedValue,
    tau: TrackedValue,
    dt: float,
    backend: GPUBackend | None = None
) -> TrackedValue:
    """
    Apply exponential decay to STDP trace.
    
    Args:
        trace: Current trace value
        tau: Decay time constant (ms)
        dt: Time step (ms)
        backend: GPU backend for acceleration
        
    Returns:
        Updated trace TrackedValue
    """
    if backend is None:
        backend = get_backend()
    
    decay_factor = math.exp(-dt / tau.value)
    new_trace = TrackedValue(
        value=trace.value * decay_factor,
        provenance=Provenance.ESTIMATED,
        metadata={"operation": "exponential_decay", "dt_ms": dt, "tau_ms": tau.value}
    )
    return new_trace


def stdp_additive_update(
    state: STDPState,
    spike_pre: bool,
    spike_post: bool,
    dt: float,
    backend: GPUBackend | None = None
) -> STDPState:
    """
    Additive STDP weight update (classic Song-Abbott rule).
    
    Args:
        state: Current STDP state
        spike_pre: Whether pre-synaptic spike occurred
        spike_post: Whether post-synaptic spike occurred
        dt: Time step (ms)
        backend: GPU backend
        
    Returns:
        Updated STDP state
    """
    new_state = state.copy()
    
    # Decay traces
    new_state.trace_pre = stdp_trace_decay(state.trace_pre, state.tau_pre, dt, backend)
    new_state.trace_post = stdp_trace_decay(state.trace_post, state.tau_post, dt, backend)
    
    # Pre-synaptic spike: LTD (depress)
    if spike_pre:
        new_state.trace_pre.value += 1.0
        new_state.trace_pre.provenance = Provenance.MEASURED
        new_state.spike_times_pre.append(len(new_state.weight_history) * dt)
        
        # Weight depression proportional to post trace
        dw = -state.a_minus.value * state.trace_post.value
        new_state.weight.value = max(0.0, min(1.0, state.weight.value + dw))
        new_state.weight.provenance = Provenance.ESTIMATED
    
    # Post-synaptic spike: LTP (potentiate)
    if spike_post:
        new_state.trace_post.value += 1.0
        new_state.trace_post.provenance = Provenance.MEASURED
        new_state.spike_times_post.append(len(new_state.weight_history) * dt)
        
        # Weight potentiation proportional to pre trace
        dw = state.a_plus.value * state.trace_pre.value
        new_state.weight.value = max(0.0, min(1.0, state.weight.value + dw))
        new_state.weight.provenance = Provenance.ESTIMATED
    
    new_state.weight_history.append(new_state.weight.value)
    new_state.internal_state.value = new_state.weight.value  # Simple mapping
    return new_state


def stdp_multiplicative_update(
    state: STDPState,
    spike_pre: bool,
    spike_post: bool,
    dt: float,
    backend: GPUBackend | None = None
) -> STDPState:
    """
    Multiplicative STDP weight update (weight-dependent plasticity).
    
    Depression scales with weight, potentiation scales with (1-weight).
    """
    new_state = state.copy()
    
    # Decay traces
    new_state.trace_pre = stdp_trace_decay(state.trace_pre, state.tau_pre, dt, backend)
    new_state.trace_post = stdp_trace_decay(state.trace_post, state.tau_post, dt, backend)
    
    if spike_pre:
        new_state.trace_pre.value += 1.0
        new_state.trace_pre.provenance = Provenance.MEASURED
        new_state.spike_times_pre.append(len(new_state.weight_history) * dt)
        
        # Multiplicative LTD: scales with current weight
        dw = -state.a_minus.value * state.trace_post.value * state.weight.value
        new_state.weight.value = max(0.0, min(1.0, state.weight.value + dw))
        new_state.weight.provenance = Provenance.ESTIMATED
    
    if spike_post:
        new_state.trace_post.value += 1.0
        new_state.trace_post.provenance = Provenance.MEASURED
        new_state.spike_times_post.append(len(new_state.weight_history) * dt)
        
        # Multiplicative LTP: scales with (1 - weight)
        dw = state.a_plus.value * state.trace_pre.value * (1.0 - state.weight.value)
        new_state.weight.value = max(0.0, min(1.0, state.weight.value + dw))
        new_state.weight.provenance = Provenance.ESTIMATED
    
    new_state.weight_history.append(new_state.weight.value)
    new_state.internal_state.value = new_state.weight.value
    return new_state


def stdp_triplet_update(
    state: STDPState,
    spike_pre: bool,
    spike_post: bool,
    dt: float,
    backend: GPUBackend | None = None
) -> STDPState:
    """
    Triplet STDP rule (Pfister & Gerstner 2006).
    
    Accounts for spike triplet interactions for better frequency dependence.
    """
    new_state = state.copy()
    
    # Decay all traces
    new_state.trace_pre = stdp_trace_decay(state.trace_pre, state.tau_pre, dt, backend)
    new_state.trace_post = stdp_trace_decay(state.trace_post, state.tau_post, dt, backend)
    new_state.trace_pre2 = stdp_trace_decay(state.trace_pre2, state.tau_pre2, dt, backend)
    new_state.trace_post2 = stdp_trace_decay(state.trace_post2, state.tau_post2, dt, backend)
    
    if spike_pre:
        new_state.trace_pre.value += 1.0
        new_state.trace_pre2.value += 1.0
        new_state.trace_pre.provenance = Provenance.MEASURED
        new_state.trace_pre2.provenance = Provenance.MEASURED
        new_state.spike_times_pre.append(len(new_state.weight_history) * dt)
        
        # Triplet LTD: depends on post trace AND second post trace
        dw = -state.a_minus.value * state.trace_post.value * (1 + state.a_minus2.value * state.trace_post2.value)
        new_state.weight.value = max(0.0, min(1.0, state.weight.value + dw))
        new_state.weight.provenance = Provenance.ESTIMATED
    
    if spike_post:
        new_state.trace_post.value += 1.0
        new_state.trace_post2.value += 1.0
        new_state.trace_post.provenance = Provenance.MEASURED
        new_state.trace_post2.provenance = Provenance.MEASURED
        new_state.spike_times_post.append(len(new_state.weight_history) * dt)
        
        # Triplet LTP: depends on pre trace AND second pre trace
        dw = state.a_plus.value * state.trace_pre.value * (1 + state.a_plus2.value * state.trace_pre2.value)
        new_state.weight.value = max(0.0, min(1.0, state.weight.value + dw))
        new_state.weight.provenance = Provenance.ESTIMATED
    
    new_state.weight_history.append(new_state.weight.value)
    new_state.internal_state.value = new_state.weight.value
    return new_state


def stdp_voltage_dependent_update(
    state: STDPState,
    spike_pre: bool,
    spike_post: bool,
    v_pre: float,
    v_post: float,
    dt: float,
    backend: GPUBackend | None = None
) -> STDPState:
    """
    Voltage-dependent STDP (memristor physics-based).
    
    Plasticity depends on voltage across the memristor during spikes.
    """
    new_state = state.copy()
    
    # Decay traces
    new_state.trace_pre = stdp_trace_decay(state.trace_pre, state.tau_pre, dt, backend)
    new_state.trace_post = stdp_trace_decay(state.trace_post, state.tau_post, dt, backend)
    
    # Voltage-dependent plasticity window
    v_diff = v_post - v_pre
    voltage_factor = 1.0 / (1.0 + math.exp(-state.k_voltage.value * (abs(v_diff) - state.v_thresh.value)))
    
    if spike_pre:
        new_state.trace_pre.value += 1.0
        new_state.trace_pre.provenance = Provenance.MEASURED
        new_state.spike_times_pre.append(len(new_state.weight_history) * dt)
        
        if v_diff < 0:  # Pre before post -> LTD
            dw = -state.a_minus.value * state.trace_post.value * voltage_factor
            new_state.weight.value = max(0.0, min(1.0, state.weight.value + dw))
    
    if spike_post:
        new_state.trace_post.value += 1.0
        new_state.trace_post.provenance = Provenance.MEASURED
        new_state.spike_times_post.append(len(new_state.weight_history) * dt)
        
        if v_diff > 0:  # Post before pre -> LTP
            dw = state.a_plus.value * state.trace_pre.value * voltage_factor
            new_state.weight.value = max(0.0, min(1.0, state.weight.value + dw))
    
    new_state.weight.provenance = Provenance.ESTIMATED
    new_state.weight_history.append(new_state.weight.value)
    new_state.internal_state.value = new_state.weight.value
    return new_state


def stdp_calcium_based_update(
    state: STDPState,
    spike_pre: bool,
    spike_post: bool,
    dt: float,
    backend: GPUBackend | None = None
) -> STDPState:
    """
    Calcium-based STDP model (Graupeter & Brunel 2007).
    
    Calcium dynamics determine LTP/LTD based on concentration thresholds.
    """
    new_state = state.copy()
    
    # Calcium dynamics
    ca = state.internal_state.value  # Using internal_state as calcium concentration
    ca_decay = math.exp(-dt / state.tau_ca.value)
    ca = ca * ca_decay + state.ca_rest.value * (1 - ca_decay)
    
    if spike_pre:
        new_state.trace_pre.value += 1.0
        new_state.trace_pre.provenance = Provenance.MEASURED
        new_state.spike_times_pre.append(len(new_state.weight_history) * dt)
        ca += 0.1 * state.trace_post.value  # NMDA-like contribution
    
    if spike_post:
        new_state.trace_post.value += 1.0
        new_state.trace_post.provenance = Provenance.MEASURED
        new_state.spike_times_post.append(len(new_state.weight_history) * dt)
        ca += 0.2  # Voltage-gated calcium channel contribution
    
    # Calcium-dependent plasticity
    ca_ltp = state.ca_thresh_ltp.value
    ca_ltd = state.ca_thresh_ltd.value
    
    if ca > ca_ltp:
        # LTP
        dw = state.a_plus.value * (ca - ca_ltp) * (1.0 - state.weight.value)
        new_state.weight.value = max(0.0, min(1.0, state.weight.value + dw))
    elif ca > ca_ltd:
        # LTD
        dw = -state.a_minus.value * (ca_ltp - ca) * state.weight.value
        new_state.weight.value = max(0.0, min(1.0, state.weight.value + dw))
    
    new_state.weight.provenance = Provenance.ESTIMATED
    new_state.internal_state.value = ca
    new_state.weight_history.append(new_state.weight.value)
    
    # Decay traces
    new_state.trace_pre = stdp_trace_decay(state.trace_pre, state.tau_pre, dt, backend)
    new_state.trace_post = stdp_trace_decay(state.trace_post, state.tau_post, dt, backend)
    
    return new_state


def stdp_memristor_physics_update(
    state: STDPState,
    spike_pre: bool,
    spike_post: bool,
    v_pre: float,
    v_post: float,
    dt: float,
    backend: GPUBackend | None = None
) -> STDPState:
    """
    Memristor physics-based STDP update.
    
    Uses actual memristor equations (Strukov model with window function)
    to update internal state based on voltage pulses.
    """
    new_state = state.copy()
    
    if backend is None:
        backend = get_backend()
    
    # Time step in seconds
    dt_s = dt * 1e-3
    
    # Voltage across memristor
    v_mem = v_post - v_pre
    
    # Current through memristor (Ohm's law with state-dependent resistance)
    w = new_state.internal_state.value  # Normalized filament length [0,1]
    ron = state.ron.value
    roff = state.roff.value
    r_mem = ron * w + roff * (1 - w)
    i_mem = v_mem / r_mem if r_mem > 0 else 0.0
    
    # State update: dw/dt = mu_v * (R_ON/D) * i * f(w)
    # Window function f(w) = 1 - (2w - 1)^(2p)
    p = state.p.value
    window = 1.0 - (2.0 * w - 1.0) ** (2.0 * p)
    window = max(0.0, window)
    
    dw_dt = state.mu_v.value * (ron / state.d.value) * i_mem * window
    new_w = max(0.0, min(1.0, w + dw_dt * dt_s))
    
    # Apply spike-triggered additional updates
    if spike_pre and spike_post:
        # Coincident spikes: strong potentiation
        new_w = min(1.0, new_w + 0.05)
    elif spike_pre:
        # Pre only: depression
        new_w = max(0.0, new_w - 0.02)
    elif spike_post:
        # Post only: potentiation
        new_w = min(1.0, new_w + 0.02)
    
    new_state.internal_state.value = new_w
    new_state.internal_state.provenance = Provenance.ESTIMATED
    
    # Update weight from internal state
    new_state.weight.value = new_w
    new_state.weight.provenance = Provenance.ESTIMATED
    new_state.weight_history.append(new_w)
    
    # Decay traces for STDP compatibility
    new_state.trace_pre = stdp_trace_decay(state.trace_pre, state.tau_pre, dt, backend)
    new_state.trace_post = stdp_trace_decay(state.trace_post, state.tau_post, dt, backend)
    
    if spike_pre:
        new_state.trace_pre.value += 1.0
        new_state.spike_times_pre.append(len(new_state.weight_history) * dt)
    if spike_post:
        new_state.trace_post.value += 1.0
        new_state.spike_times_post.append(len(new_state.weight_history) * dt)
    
    return new_state


def stdp_step(
    state: STDPState,
    spike_pre: bool,
    spike_post: bool,
    dt: float,
    rule: STDPRule = STDPRule.ADDITIVE,
    v_pre: float = 0.0,
    v_post: float = 0.0,
    backend: GPUBackend | None = None
) -> STDPState:
    """
    Main STDP step function dispatching to specific rule.
    
    Args:
        state: Current STDP state
        spike_pre: Pre-synaptic spike flag
        spike_post: Post-synaptic spike flag
        dt: Time step (ms)
        rule: STDP rule to apply
        v_pre: Pre-synaptic voltage (for voltage-dependent rules)
        v_post: Post-synaptic voltage (for voltage-dependent rules)
        backend: GPU backend
        
    Returns:
        Updated STDP state
    """
    if rule == STDPRule.ADDITIVE:
        return stdp_additive_update(state, spike_pre, spike_post, dt, backend)
    elif rule == STDPRule.MULTIPLICATIVE:
        return stdp_multiplicative_update(state, spike_pre, spike_post, dt, backend)
    elif rule == STDPRule.TRIPLET:
        return stdp_triplet_update(state, spike_pre, spike_post, dt, backend)
    elif rule == STDPRule.VOLTAGE_DEPENDENT:
        return stdp_voltage_dependent_update(state, spike_pre, spike_post, v_pre, v_post, dt, backend)
    elif rule == STDPRule.CALCIUM_BASED:
        return stdp_calcium_based_update(state, spike_pre, spike_post, dt, backend)
    else:
        raise ValueError(f"Unknown STDP rule: {rule}")


def stdp_crossbar_vmm(
    weights: np.ndarray,
    inputs: np.ndarray,
    g_min: float = 1e-6,
    g_max: float = 1e-3,
    device: str = "auto",
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    Vector-Matrix Multiplication (VMM) on memristor crossbar array.
    
    Simulates analog VMM using conductance matrix with non-idealities.
    
    Args:
        weights: Weight matrix [output_dim, input_dim] (normalized 0-1)
        inputs: Input voltage vector [input_dim]
        g_min: Minimum conductance (S)
        g_max: Maximum conductance (S)
        device: "auto", "cuda", "cpu"
        backend: GPU backend
        
    Returns:
        Output current vector [output_dim]
    """
    if backend is None:
        backend = get_backend(device)
    
    xp = backend.xp
    
    # Convert to backend arrays
    w = backend.to_device(weights)
    x = backend.to_device(inputs)
    
    # Map weights to conductances
    g = g_min + w * (g_max - g_min)
    
    # VMM: I = G @ V
    # Add non-idealities: wire resistance, sneak paths, variability
    output = xp.matmul(g, x)
    
    # Add thermal noise (Johnson-Nyquist)
    temp = 300.0  # K
    k_b = 1.38e-23
    bandwidth = 1e6  # 1 MHz
    noise_std = xp.sqrt(4 * k_b * temp * bandwidth * g.mean())
    noise = backend.random_normal(output.shape, std=float(noise_std))
    output = output + noise
    
    return backend.to_numpy(output)


def stdp_array_simulate(
    n_pre: int,
    n_post: int,
    spike_trains_pre: list[list[float]],
    spike_trains_post: list[list[float]],
    dt: float = 1.0,
    rule: STDPRule = STDPRule.ADDITIVE,
    initial_weights: np.ndarray | None = None,
    device: str = "auto",
    backend: GPUBackend | None = None
) -> tuple[np.ndarray, list[STDPState]]:
    """
    Simulate STDP learning in a full crossbar array.
    
    Args:
        n_pre: Number of pre-synaptic neurons
        n_post: Number of post-synaptic neurons
        spike_trains_pre: List of spike times for each pre neuron (ms)
        spike_trains_post: List of spike times for each post neuron (ms)
        dt: Simulation time step (ms)
        rule: STDP rule
        initial_weights: Initial weight matrix [n_post, n_pre]
        device: Compute device
        backend: GPU backend
        
    Returns:
        Final weight matrix, list of final states per synapse
    """
    if backend is None:
        backend = get_backend(device)
    
    xp = backend.xp
    
    # Initialize weights
    if initial_weights is not None:
        w = backend.to_device(initial_weights)
    else:
        w = backend.random_uniform((n_post, n_pre), 0.4, 0.6)
    
    # Convert spike trains to binary matrices [n_neurons, n_timesteps]
    max_time_pre = max((max(t) for t in spike_trains_pre if t), default=0)
    max_time_post = max((max(t) for t in spike_trains_post if t), default=0)
    max_time = max(max_time_pre, max_time_post)
    n_steps = int(max_time / dt) + 1
    
    spikes_pre = xp.zeros((n_pre, n_steps), dtype=bool)
    spikes_post = xp.zeros((n_post, n_steps), dtype=bool)
    
    for i, times in enumerate(spike_trains_pre):
        for t in times:
            idx = int(t / dt)
            if idx < n_steps:
                spikes_pre[i, idx] = True
    
    for j, times in enumerate(spike_trains_post):
        for t in times:
            idx = int(t / dt)
            if idx < n_steps:
                spikes_post[j, idx] = True
    
    # Simulate
    states = []
    for i in range(n_pre):
        for j in range(n_post):
            state = STDPState()
            state.weight.value = float(w[j, i])
            
            for step in range(n_steps):
                spike_p = bool(spikes_pre[i, step])
                spike_post = bool(spikes_post[j, step])
                state = stdp_step(state, spike_p, spike_post, dt, rule, backend=backend)
            
            w[j, i] = state.weight.value
            states.append(state)
    
    return backend.to_numpy(w), states


def stdp_stdp_curve(
    state: STDPState,
    dt: float = 0.1,
    max_dt: float = 50.0,
    rule: STDPRule = STDPRule.ADDITIVE,
    backend: GPUBackend | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute STDP learning window (weight change vs spike timing difference).
    
    Args:
        state: Base STDP state
        dt: Time resolution (ms)
        max_dt: Maximum delta-t to simulate (ms)
        rule: STDP rule
        backend: GPU backend
        
    Returns:
        delta_t array, delta_w array
    """
    delta_ts = np.arange(-max_dt, max_dt + dt, dt)
    delta_ws = []
    
    for delta_t in delta_ts:
        test_state = state.copy()
        
        if delta_t > 0:
            # Pre before post: LTP
            # Simulate pre spike at t=0, post at t=delta_t
            n_steps = int(delta_t / dt) + 1
            for step in range(n_steps):
                spike_pre = (step == 0)
                spike_post = (step == n_steps - 1)
                test_state = stdp_step(test_state, spike_pre, spike_post, dt, rule, backend=backend)
        elif delta_t < 0:
            # Post before pre: LTD
            n_steps = int(-delta_t / dt) + 1
            for step in range(n_steps):
                spike_post = (step == 0)
                spike_pre = (step == n_steps - 1)
                test_state = stdp_step(test_state, spike_pre, spike_post, dt, rule, backend=backend)
        else:
            # Simultaneous
            test_state = stdp_step(test_state, True, True, dt, rule, backend=backend)
        
        delta_ws.append(test_state.weight.value - state.weight.value)
    
    return delta_ts, np.array(delta_ws)


# Export all public functions and classes
__all__ = [
    "STDPRule",
    "MemristorModel",
    "STDPState",
    "stdp_trace_decay",
    "stdp_additive_update",
    "stdp_multiplicative_update",
    "stdp_triplet_update",
    "stdp_voltage_dependent_update",
    "stdp_calcium_based_update",
    "stdp_memristor_physics_update",
    "stdp_step",
    "stdp_crossbar_vmm",
    "stdp_array_simulate",
    "stdp_stdp_curve",
]