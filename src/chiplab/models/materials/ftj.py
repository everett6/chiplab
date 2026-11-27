"""
Ferroelectric Tunnel Junction (FTJ) Model
==========================================

Implements FTJ physics for neuromorphic synapses and memory.
Based on tunneling electroresistance (TER) effect in ferroelectric barriers.

References:
- Garcia et al. (2009) - Giant tunneling electroresistance
- Boyn et al. (2017) - FTJ-based synapses
- Yin et al. (2019) - FTJ for neuromorphic computing
- Max Planck Institute FTJ reviews
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np

from chiplab.schemas.common import Provenance, TrackedValue
from chiplab.core.gpu import GPUBackend, get_gpu_backend as get_backend


class FTJMaterial(Enum):
    """Ferroelectric barrier materials."""
    BTO = "BaTiO3"           # Barium Titanate
    PZT = "PbZrTiO3"         # Lead Zirconate Titanate
    HZO = "HfZrO2"           # Hafnium Zirconium Oxide (CMOS-compatible)
    BFO = "BiFeO3"           # Bismuth Ferrite (multiferroic)
    SBT = "SrBi2Ta2O9"       # Strontium Bismuth Tantalate
    CUSTOM = "custom"


class ElectrodeMaterial(Enum):
    """Electrode materials."""
    PT = "Pt"                # Platinum
    SRO = "SrRuO3"           # Strontium Ruthenate
    LSMO = "LaSrMnO3"        # Lanthanum Strontium Manganite
    TI_N = "TiN"             # Titanium Nitride (CMOS)
    TA_N = "TaN"             # Tantalum Nitride
    GRAPHENE = "graphene"    # Graphene
    CUSTOM = "custom"


@dataclass
class FTJParams:
    """FTJ device parameters with provenance."""
    # Barrier properties
    material: FTJMaterial = FTJMaterial.HZO
    thickness: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=2e-9,
        provenance=Provenance.MEASURED,
        metadata={"description": "Ferroelectric barrier thickness", "unit": "m"}
    ))
    polarization_sat: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=30e-6,
        provenance=Provenance.MEASURED,
        metadata={"description": "Saturation polarization", "unit": "C/cm^2"}
    ))
    coercive_field: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e6,
        provenance=Provenance.MEASURED,
        metadata={"description": "Coercive field", "unit": "V/m"}
    ))
    dielectric_const: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=25.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Relative dielectric constant"}
    ))
    band_gap: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=3.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Band gap", "unit": "eV"}
    ))
    electron_affinity: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Electron affinity", "unit": "eV"}
    ))
    effective_mass: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.5,
        provenance=Provenance.MEASURED,
        metadata={"description": "Effective mass (m0)"}
    ))
    
    # Electrode properties
    top_electrode: ElectrodeMaterial = ElectrodeMaterial.TI_N
    bottom_electrode: ElectrodeMaterial = ElectrodeMaterial.TI_N
    work_function_top: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=4.7,
        provenance=Provenance.MEASURED,
        metadata={"description": "Top electrode work function", "unit": "eV"}
    ))
    work_function_bottom: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=4.7,
        provenance=Provenance.MEASURED,
        metadata={"description": "Bottom electrode work function", "unit": "eV"}
    ))
    
    # Interface properties
    interface_trap_density: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e12,
        provenance=Provenance.MEASURED,
        metadata={"description": "Interface trap density", "unit": "cm^-2 eV^-1"}
    ))
    screening_length: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.5e-9,
        provenance=Provenance.MEASURED,
        metadata={"description": "Screening length", "unit": "m"}
    ))
    
    # Switching dynamics
    switching_time: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e-9,
        provenance=Provenance.MEASURED,
        metadata={"description": "Polarization switching time", "unit": "s"}
    ))
    nucleation_field: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.2e6,
        provenance=Provenance.MEASURED,
        metadata={"description": "Nucleation field", "unit": "V/m"}
    ))
    growth_velocity: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=10.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Domain wall growth velocity", "unit": "m/s"}
    ))
    
    # Noise and variability
    cycle_to_cycle_var: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.05,
        provenance=Provenance.MEASURED,
        metadata={"description": "Cycle-to-cycle variability (std)"}
    ))
    device_to_device_var: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.1,
        provenance=Provenance.MEASURED,
        metadata={"description": "Device-to-device variability (std)"}
    ))
    retention_time: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e10,
        provenance=Provenance.MEASURED,
        metadata={"description": "Retention time at room temp", "unit": "s"}
    ))
    
    # Temperature
    temperature: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=300.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Operating temperature", "unit": "K"}
    ))


@dataclass
class FTJState:
    """
    FTJ device state with full provenance tracking.
    
    Tracks polarization, resistance states, and history.
    """
    # Device parameters
    params: FTJParams = field(default_factory=FTJParams)
    area: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=50e-9 * 50e-9,
        provenance=Provenance.MEASURED,
        metadata={"description": "Junction area", "unit": "m^2"}
    ))
    
    # Polarization state (normalized -1 to +1)
    polarization: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,
        provenance=Provenance.ASSUMED,
        metadata={"description": "Normalized polarization [-1, 1]"}
    ))
    
    # Domain configuration (for multi-domain)
    domain_fraction_up: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.5,
        provenance=Provenance.ASSUMED,
        metadata={"description": "Fraction of up domains"}
    ))
    
    # Resistance states
    r_on: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e3,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Low resistance state (parallel pol.)", "unit": "ohm"}
    ))
    r_off: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e6,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "High resistance state (antiparallel pol.)", "unit": "ohm"}
    ))
    ter_ratio: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1000.0,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Tunneling electroresistance ratio"}
    ))
    
    # Current state
    current_resistance: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e4,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Current resistance", "unit": "ohm"}
    ))
    current_voltage: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Applied voltage", "unit": "V"}
    ))
    current_current: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Current through device", "unit": "A"}
    ))
    
    # History
    polarization_history: list[float] = field(default_factory=list)
    resistance_history: list[float] = field(default_factory=list)
    voltage_history: list[float] = field(default_factory=list)
    switching_events: list[dict] = field(default_factory=list)
    
    def copy(self) -> FTJState:
        """Create a deep copy."""
        new_state = FTJState()
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, TrackedValue):
                setattr(new_state, field_name, value.copy())
            elif isinstance(value, (FTJParams, FTJMaterial, ElectrodeMaterial)):
                setattr(new_state, field_name, value)
            elif isinstance(value, (int, float)):
                setattr(new_state, field_name, value)
            elif isinstance(value, list):
                setattr(new_state, field_name, value.copy())
            else:
                setattr(new_state, field_name, value)
        return new_state
    
    def get_ter(self) -> float:
        """Get TER ratio."""
        return float(self.ter_ratio.value)
    
    def get_resistance(self) -> float:
        """Get current resistance."""
        return float(self.current_resistance.value)


def ftj_tunneling_current(
    state: FTJState,
    voltage: float,
    temperature: float = 300.0,
    backend: GPUBackend | None = None
) -> float:
    """
    Compute tunneling current using Simmons model (direct tunneling).
    
    J = (e^2 / (2π h d^2)) * (φ - eV/2) * exp(-4π d √(2m(φ - eV/2)) / h)
    for low bias, with ferroelectric barrier modification.
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    
    # Constants
    e = 1.602e-19  # C
    h = 6.626e-34  # J*s
    hbar = h / (2 * np.pi)
    m0 = 9.109e-31  # kg
    k_b = 1.381e-23  # J/K
    
    # Parameters
    d = state.params.thickness.value
    phi = state.params.band_gap.value  # Barrier height (eV)
    m_eff = state.params.effective_mass.value * m0
    area = state.area.value
    pol = state.polarization.value
    
    # Polarization modifies barrier height
    # Δφ = ± P_s * d / (ε_0 * ε_r)  (depolarization field)
    eps_0 = 8.854e-12
    eps_r = state.params.dielectric_const.value
    p_sat = state.params.polarization_sat.value * 1e4  # Convert to C/m^2
    
    # Depolarization field contribution
    delta_phi = pol * p_sat * d / (eps_0 * eps_r) / e  # in eV
    
    # Effective barrier heights for two spin channels (if magnetic electrodes)
    # For non-magnetic: average barrier
    phi_eff = phi - delta_phi / 2  # Simplified
    
    # Voltage division across barrier
    v = voltage
    
    # Simmons formula (low temperature, direct tunneling)
    # J = A * (φ - eV/2) * exp(-B * d * sqrt(φ - eV/2))
    # where A = e^2 / (2π h d^2), B = 4π sqrt(2m) / h
    
    phi_j = phi_eff * e  # Convert to Joules
    v_j = v * e
    
    if phi_j - v_j/2 <= 0:
        # Barrier suppressed - Fowler-Nordheim or thermionic
        return ftj_fowler_nordheim_current(state, voltage, backend)
    
    A = e**2 / (2 * np.pi * h * d**2)
    B = 4 * np.pi * np.sqrt(2 * m_eff) / h
    
    term = phi_j - v_j / 2
    exponent = -B * d * np.sqrt(term)
    
    # Prevent overflow
    exponent = max(exponent, -100)
    
    j = A * term * np.exp(exponent)  # A/m^2
    
    # Add thermal correction
    if temperature > 0:
        thermal_factor = 1 + (np.pi * k_b * temperature / term)**2 / 6
        j *= thermal_factor
    
    current = j * area
    return float(current)


def ftj_fowler_nordheim_current(
    state: FTJState,
    voltage: float,
    backend: GPUBackend | None = None
) -> float:
    """
    Fowler-Nordheim tunneling for high fields.
    
    J = A * E^2 * exp(-B / E)
    """
    if backend is None:
        backend = get_backend()
    
    e = 1.602e-19
    h = 6.626e-34
    m0 = 9.109e-31
    
    d = state.params.thickness.value
    phi = state.params.band_gap.value * e
    m_eff = state.params.effective_mass.value * m0
    area = state.area.value
    
    E = voltage / d  # V/m
    
    if E <= 0:
        return 0.0
    
    A_fn = e**3 / (8 * np.pi * h * phi)
    B_fn = 4 * np.pi * np.sqrt(2 * m_eff) * phi**(3/2) / (3 * e * h)
    
    j = A_fn * E**2 * np.exp(-B_fn / E)
    return float(j * area)


def ftj_polarization_dynamics(
    state: FTJState,
    voltage: float,
    dt: float,
    backend: GPUBackend | None = None
) -> FTJState:
    """
    Update polarization based on applied voltage.
    
    Uses Kolmogorov-Avrami-Ishibashi (KAI) model for switching:
    P(t) = P_s * [1 - exp(-(t/τ)^n)] for nucleation and growth
    
    With field-dependent switching time.
    """
    if backend is None:
        backend = get_backend()
    
    new_state = state.copy()
    
    # Electric field
    d = state.params.thickness.value
    E = voltage / d
    E_c = state.params.coercive_field.value
    E_nuc = state.params.nucleation_field.value
    
    # Current polarization
    P = new_state.polarization.value
    P_sat = 1.0  # Normalized
    
    # KAI model: switching time depends on field
    # τ = τ_0 * exp(E_a / E)  (Merz law)
    # Or τ = τ_0 * (E/E_c - 1)^-n
    
    if abs(E) > E_nuc:
        # Nucleation occurs
        # Switching time
        tau_0 = state.params.switching_time.value
        # Field acceleration
        if abs(E) > E_c:
            tau = tau_0 * (E_c / abs(E))**2
        else:
            tau = tau_0 * 100  # Very slow below coercive
        
        # Avrami exponent (n=2 for 2D growth, n=3 for 3D)
        n_avrami = 2.0
        
        # Fraction switched
        frac_switched = 1 - np.exp(-(dt / tau)**n_avrami)
        
        # Direction
        target_pol = 1.0 if E > 0 else -1.0
        
        # Update polarization
        new_P = P + (target_pol - P) * frac_switched
        
        # Add variability
        var = state.params.cycle_to_cycle_var.value
        noise = np.random.normal(0, var * frac_switched)
        new_P = np.clip(new_P + noise, -1.0, 1.0)
        
        new_state.polarization.value = new_P
        new_state.polarization.provenance = Provenance.ESTIMATED
        
        # Record switching event
        if abs(new_P - P) > 0.1:
            new_state.switching_events.append({
                "time": len(new_state.polarization_history) * dt,
                "voltage": voltage,
                "field": E,
                "polarization_before": P,
                "polarization_after": new_P,
                "frac_switched": frac_switched
            })
    
    # Update domain fraction
    new_state.domain_fraction_up.value = (new_state.polarization.value + 1) / 2
    
    # Update resistance based on polarization
    new_state = ftj_update_resistance(new_state, backend)
    
    return new_state


def ftj_update_resistance(
    state: FTJState,
    backend: GPUBackend | None = None
) -> FTJState:
    """
    Update resistance based on polarization state.
    
    TER effect: R = R_on for P=+1, R_off for P=-1
    Smooth interpolation for partial polarization.
    """
    new_state = state.copy()
    
    # Base resistance states
    r_on = new_state.r_on.value
    r_off = new_state.r_off.value
    
    # Polarization-dependent resistance
    # R(P) = R_on * exp(-TER * (P+1)/2)  or similar
    # Simplified: linear in TER ratio
    pol = new_state.polarization.value
    ter = new_state.ter_ratio.value
    
    # Log-linear interpolation
    log_r_on = np.log(r_on)
    log_r_off = np.log(r_off)
    log_r = log_r_on + (log_r_off - log_r_on) * (1 - pol) / 2
    
    new_R = np.exp(log_r)
    
    # Add device variability
    var = state.params.device_to_device_var.value
    new_R *= np.random.lognormal(0, var)
    
    new_state.current_resistance.value = new_R
    new_state.current_resistance.provenance = Provenance.ESTIMATED
    
    return new_state


def ftj_iv_curve(
    state: FTJState,
    voltages: np.ndarray,
    temperature: float = 300.0,
    backend: GPUBackend | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute I-V curve for given polarization state.
    
    Returns:
        voltages, currents
    """
    if backend is None:
        backend = get_backend()
    
    currents = []
    for v in voltages:
        # For I-V sweep, polarization may switch
        # Here we compute static I-V for fixed polarization
        i = ftj_tunneling_current(state, v, temperature, backend)
        currents.append(i)
    
    return voltages, np.array(currents)


def ftj_pv_loop(
    state: FTJState,
    v_max: float = 3.0,
    n_points: int = 200,
    freq: float = 1e3,
    backend: GPUBackend | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate P-V hysteresis loop (ferroelectric characterization).
    
    Applies triangular voltage waveform and tracks polarization.
    
    Returns:
        voltages, polarizations, currents
    """
    if backend is None:
        backend = get_backend()
    
    # Triangular wave
    t = np.linspace(0, 1/freq, n_points)
    v_tri = np.zeros_like(t)
    for i, ti in enumerate(t):
        phase = (ti * freq) % 1.0
        if phase < 0.25:
            v_tri[i] = v_max * phase * 4
        elif phase < 0.75:
            v_tri[i] = v_max * (1 - (phase - 0.25) * 4)
        else:
            v_tri[i] = -v_max * (1 - (phase - 0.75) * 4)
    
    dt = t[1] - t[0]
    run_state = state.copy()
    
    polarizations = []
    currents = []
    
    for v in v_tri:
        run_state = ftj_polarization_dynamics(run_state, v, dt, backend)
        i = ftj_tunneling_current(run_state, v, run_state.params.temperature.value, backend)
        polarizations.append(run_state.polarization.value)
        currents.append(i)
    
    return v_tri, np.array(polarizations), np.array(currents)


def ftj_write_pulse(
    state: FTJState,
    voltage: float,
    pulse_width: float,
    dt: float = 1e-12,
    backend: GPUBackend | None = None
) -> FTJState:
    """
    Apply write pulse to switch polarization.
    
    Args:
        state: Initial state
        voltage: Pulse amplitude (V)
        pulse_width: Pulse duration (s)
        dt: Simulation time step (s)
        backend: GPU backend
        
    Returns:
        State after pulse
    """
    if backend is None:
        backend = get_backend()
    
    n_steps = int(pulse_width / dt)
    run_state = state.copy()
    
    for _ in range(n_steps):
        run_state = ftj_polarization_dynamics(run_state, voltage, dt, backend)
        run_state.current_voltage.value = voltage
        run_state.current_current.value = ftj_tunneling_current(
            run_state, voltage, run_state.params.temperature.value, backend
        )
        run_state.polarization_history.append(run_state.polarization.value)
        run_state.resistance_history.append(run_state.current_resistance.value)
        run_state.voltage_history.append(voltage)
    
    # Post-pulse relaxation (zero voltage)
    for _ in range(100):
        run_state = ftj_polarization_dynamics(run_state, 0.0, dt, backend)
    
    return run_state


def ftj_read_pulse(
    state: FTJState,
    voltage: float = 0.1,
    dt: float = 1e-9,
    backend: GPUBackend | None = None
) -> tuple[float, float]:
    """
    Apply low-voltage read pulse (non-disturbive).
    
    Returns:
        resistance, current
    """
    if backend is None:
        backend = get_backend()
    
    # Read at low voltage to avoid switching
    i = ftj_tunneling_current(state, voltage, state.params.temperature.value, backend)
    r = voltage / i if i > 0 else float('inf')
    
    return r, i


def ftj_synaptic_weight(
    state: FTJState,
    w_min: float = 0.0,
    w_max: float = 1.0
) -> float:
    """
    Map FTJ resistance to synaptic weight [0, 1].
    
    Weight = (R - R_on) / (R_off - R_on)  (for depression = high R)
    or inverted depending on convention.
    """
    r = state.current_resistance.value
    r_on = state.r_on.value
    r_off = state.r_off.value
    
    # Normalize: 0 = R_on (potentiated), 1 = R_off (depressed)
    w = (r - r_on) / (r_off - r_on)
    return float(np.clip(w, w_min, w_max))


def ftj_stdp_update(
    state: FTJState,
    spike_pre: bool,
    spike_post: bool,
    v_pre: float,
    v_post: float,
    dt: float,
    a_plus: float = 0.1,
    a_minus: float = 0.12,
    tau_pre: float = 20e-3,
    tau_post: float = 20e-3,
    backend: GPUBackend | None = None
) -> FTJState:
    """
    STDP update using FTJ physics.
    
    Pre-before-post (LTP): Apply positive pulse -> polarization up -> R down
    Post-before-pre (LTD): Apply negative pulse -> polarization down -> R up
    """
    if backend is None:
        backend = get_backend()
    
    new_state = state.copy()
    
    # Voltage across FTJ
    v_ftj = v_post - v_pre
    
    # Trace-based STDP (simplified)
    # In practice, would need trace variables
    if spike_pre and spike_post:
        # Coincident spikes - strong potentiation
        pulse_v = 2.0  # Positive voltage for LTP
        pulse_w = 10e-9
        new_state = ftj_write_pulse(new_state, pulse_v, pulse_w, dt=1e-12, backend=backend)
    elif spike_pre:
        # Pre only - depression
        pulse_v = -1.5
        pulse_w = 10e-9
        new_state = ftj_write_pulse(new_state, pulse_v, pulse_w, dt=1e-12, backend=backend)
    elif spike_post:
        # Post only - potentiation
        pulse_v = 1.5
        pulse_w = 10e-9
        new_state = ftj_write_pulse(new_state, pulse_v, pulse_w, dt=1e-12, backend=backend)
    
    return new_state


def ftj_crossbar_vmm(
    states: list[FTJState],
    inputs: np.ndarray,  # [n_inputs] voltages
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    Vector-Matrix Multiplication using FTJ crossbar array.
    
    Each FTJ acts as a conductance: I_j = Σ_i G_ij * V_i
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    n_outputs = len(states)
    n_inputs = len(inputs)
    
    # Build conductance matrix
    G = xp.zeros((n_outputs, n_inputs), dtype=xp.float32)
    for j, ftj_state in enumerate(states):
        r = ftj_state.current_resistance.value
        g = 1.0 / r if r > 0 else 0.0
        G[j, :] = g
    
    V = backend.to_device(inputs.astype(np.float32))
    I = xp.matmul(G, V)
    
    return backend.to_numpy(I)


def ftj_retention_simulation(
    state: FTJState,
    time_points: np.ndarray,  # Time points in seconds
    temperature: float = 300.0,
    backend: GPUBackend | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate polarization retention over time.
    
    Uses thermal activation model: dP/dt = -P/τ_ret
    τ_ret = τ_0 * exp(E_a / kT)
    """
    if backend is None:
        backend = get_backend()
    
    # Activation energy for depolarization
    # E_a ≈ P_s * E_c * d * volume
    p_sat = state.params.polarization_sat.value * 1e4
    e_c = state.params.coercive_field.value
    d = state.params.thickness.value
    area = state.area.value
    volume = area * d
    
    E_a = p_sat * e_c * d * volume  # J
    k_b = 1.381e-23
    
    tau_0 = 1e-12  # Attempt time
    tau_ret = tau_0 * np.exp(E_a / (k_b * temperature))
    
    # Retention time from params
    tau_ret = state.params.retention_time.value
    
    pol_0 = state.polarization.value
    polarizations = pol_0 * np.exp(-time_points / tau_ret)
    resistances = []
    
    for p in polarizations:
        temp_state = state.copy()
        temp_state.polarization.value = p
        temp_state = ftj_update_resistance(temp_state, backend)
        resistances.append(temp_state.current_resistance.value)
    
    return polarizations, np.array(resistances)


def ftj_endurance_simulation(
    state: FTJState,
    n_cycles: int,
    write_voltage: float = 2.0,
    pulse_width: float = 10e-9,
    dt: float = 1e-12,
    backend: GPUBackend | None = None
) -> dict:
    """
    Simulate endurance cycling.
    
    Returns statistics on TER degradation, switching variability.
    """
    if backend is None:
        backend = get_backend()
    
    run_state = state.copy()
    
    ter_history = []
    r_on_history = []
    r_off_history = []
    switch_voltage_history = []
    
    for cycle in range(n_cycles):
        # Write '1' (positive polarization)
        run_state = ftj_write_pulse(run_state, write_voltage, pulse_width, dt, backend)
        r_on = run_state.current_resistance.value
        
        # Write '0' (negative polarization)
        run_state = ftj_write_pulse(run_state, -write_voltage, pulse_width, dt, backend)
        r_off = run_state.current_resistance.value
        
        ter = r_off / r_on if r_on > 0 else 0
        
        ter_history.append(ter)
        r_on_history.append(r_on)
        r_off_history.append(r_off)
        
        # Estimate switching voltage from last event
        if run_state.switching_events:
            switch_voltage_history.append(run_state.switching_events[-1]["voltage"])
    
    return {
        "ter_history": np.array(ter_history),
        "r_on_history": np.array(r_on_history),
        "r_off_history": np.array(r_off_history),
        "switch_voltage_history": np.array(switch_voltage_history),
        "ter_degradation": (ter_history[0] - ter_history[-1]) / ter_history[0] if ter_history[0] > 0 else 0,
        "cycles_to_failure": n_cycles  # Would need failure criterion
    }


# Export
__all__ = [
    "FTJMaterial",
    "ElectrodeMaterial",
    "FTJParams",
    "FTJState",
    "ftj_tunneling_current",
    "ftj_fowler_nordheim_current",
    "ftj_polarization_dynamics",
    "ftj_update_resistance",
    "ftj_iv_curve",
    "ftj_pv_loop",
    "ftj_write_pulse",
    "ftj_read_pulse",
    "ftj_synaptic_weight",
    "ftj_stdp_update",
    "ftj_crossbar_vmm",
    "ftj_retention_simulation",
    "ftj_endurance_simulation",
]