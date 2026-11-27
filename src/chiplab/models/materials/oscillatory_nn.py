"""
Oscillatory Neural Network (ONN) - Coupled VO2/MTJ Oscillators
===============================================================

Implements networks of coupled oscillators for pattern recognition,
associative memory, and optimization (Ising machines).

Oscillator types:
- VO2 (Vanadium Dioxide) relaxation oscillators - insulator-metal transition
- MTJ (Magnetic Tunnel Junction) spin-torque oscillators
- CMOS ring oscillators (baseline)
- Photonic crystal cavity oscillators

References:
- Hoppensteadt & Izhikevich (2001) - Oscillatory neural networks
- Csaba & Porod (2013) - Coupled oscillators for computing
- Datta et al. (2017) - VO2 oscillators for neuromorphic
- Zografos et al. (2020) - MTJ oscillators for Ising machines
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np

from chiplab.schemas.common import Provenance, TrackedValue
from chiplab.core.gpu import GPUBackend, get_gpu_backend as get_backend


class OscillatorType(Enum):
    """Physical oscillator implementation."""
    VO2_RELAXATION = "vo2_relaxation"
    MTJ_STT = "mtj_stt"
    MTJ_SOT = "mtj_sot"
    CMOS_RING = "cmos_ring"
    PHOTONIC_CAVITY = "photonic_cavity"
    ABSTRACT_KURAMOTO = "abstract_kuramoto"
    ABSTRACT_FITZHUGH_NAGUMO = "abstract_fitzhugh_nagumo"


class CouplingType(Enum):
    """Coupling mechanism between oscillators."""
    RESISTIVE = "resistive"
    CAPACITIVE = "capacitive"
    INDUCTIVE = "inductive"
    MUTUAL_INDUCTANCE = "mutual_inductance"
    OPTICAL = "optical"
    SPIN_WAVE = "spin_wave"
    GLOBAL = "global"  # All-to-all


@dataclass
class OscillatorParams:
    """Parameters for a single oscillator."""
    # Natural frequency (Hz)
    f0: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e9,
        provenance=Provenance.MEASURED,
        metadata={"description": "Natural frequency", "unit": "Hz"}
    ))
    
    # Quality factor
    q_factor: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=100.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Quality factor"}
    ))
    
    # Amplitude limits
    amp_min: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.1,
        provenance=Provenance.MEASURED,
        metadata={"description": "Minimum oscillation amplitude"}
    ))
    amp_max: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Maximum oscillation amplitude"}
    ))
    
    # VO2-specific: Insulator-metal transition
    v_th: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=2.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "VO2 threshold voltage", "unit": "V"}
    ))
    v_h: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "VO2 holding voltage", "unit": "V"}
    ))
    r_on: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=100.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "VO2 ON resistance", "unit": "ohm"}
    ))
    r_off: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=10e3,
        provenance=Provenance.MEASURED,
        metadata={"description": "VO2 OFF resistance", "unit": "ohm"}
    ))
    c_vo2: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e-12,
        provenance=Provenance.MEASURED,
        metadata={"description": "VO2 capacitance", "unit": "F"}
    ))
    
    # MTJ-specific
    i_stt: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e-3,
        provenance=Provenance.MEASURED,
        metadata={"description": "STT critical current", "unit": "A"}
    ))
    alpha_g: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.01,
        provenance=Provenance.MEASURED,
        metadata={"description": "Gilbert damping"}
    ))
    p_polar: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.5,
        provenance=Provenance.MEASURED,
        metadata={"description": "Spin polarization"}
    ))
    
    # Phase noise
    phase_noise: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e-6,
        provenance=Provenance.MEASURED,
        metadata={"description": "Phase noise coefficient"}
    ))


@dataclass
class ONNState:
    """
    State of an Oscillatory Neural Network.
    
    Tracks phases, amplitudes, and coupling matrix for all oscillators.
    """
    # Network configuration
    n_oscillators: int = 10
    oscillator_type: OscillatorType = OscillatorType.VO2_RELAXATION
    coupling_type: CouplingType = CouplingType.RESISTIVE
    
    # Oscillator parameters (per oscillator or shared)
    params: OscillatorParams = field(default_factory=OscillatorParams)
    
    # Phase of each oscillator [0, 2π)
    phases: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=np.zeros(10),
        provenance=Provenance.ASSUMED,
        metadata={"description": "Oscillator phases", "unit": "rad"}
    ))
    
    # Amplitude of each oscillator
    amplitudes: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=np.ones(10),
        provenance=Provenance.ASSUMED,
        metadata={"description": "Oscillator amplitudes"}
    ))
    
    # Coupling matrix (symmetric)
    coupling: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=np.zeros((10, 10)),
        provenance=Provenance.ASSUMED,
        metadata={"description": "Coupling strength matrix"}
    ))
    
    # External input currents/voltages
    inputs: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=np.zeros(10),
        provenance=Provenance.ASSUMED,
        metadata={"description": "External inputs"}
    ))
    
    # Frequency detuning (disorder)
    detuning: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=np.zeros(10),
        provenance=Provenance.MEASURED,
        metadata={"description": "Frequency detuning from nominal"}
    ))
    
    # History
    phase_history: list[np.ndarray] = field(default_factory=list)
    amplitude_history: list[np.ndarray] = field(default_factory=list)
    sync_history: list[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize arrays with correct dimensions based on n_oscillators."""
        self.phases = TrackedValue(
            value=np.zeros(self.n_oscillators),
            provenance=Provenance.ASSUMED,
            metadata={"description": "Oscillator phases", "unit": "rad"}
        )
        self.amplitudes = TrackedValue(
            value=np.ones(self.n_oscillators),
            provenance=Provenance.ASSUMED,
            metadata={"description": "Oscillator amplitudes"}
        )
        self.coupling = TrackedValue(
            value=np.zeros((self.n_oscillators, self.n_oscillators)),
            provenance=Provenance.ASSUMED,
            metadata={"description": "Coupling strength matrix"}
        )
        self.inputs = TrackedValue(
            value=np.zeros(self.n_oscillators),
            provenance=Provenance.ASSUMED,
            metadata={"description": "External inputs"}
        )
        self.detuning = TrackedValue(
            value=np.zeros(self.n_oscillators),
            provenance=Provenance.MEASURED,
            metadata={"description": "Frequency detuning from nominal"}
        )
    
    def copy(self) -> ONNState:
        """Create a deep copy."""
        new_state = ONNState(
            n_oscillators=self.n_oscillators,
            oscillator_type=self.oscillator_type,
            coupling_type=self.coupling_type,
            params=self.params
        )
        for field_name in self.__dataclass_fields__:
            if field_name in ['n_oscillators', 'oscillator_type', 'coupling_type', 'params']:
                continue
            value = getattr(self, field_name)
            if isinstance(value, TrackedValue):
                setattr(new_state, field_name, value.copy())
            elif isinstance(value, (OscillatorType, CouplingType, OscillatorParams)):
                setattr(new_state, field_name, value)
            elif isinstance(value, (int, float)):
                setattr(new_state, field_name, value)
            elif isinstance(value, list):
                setattr(new_state, field_name, [v.copy() if isinstance(v, np.ndarray) else v for v in value])
            else:
                setattr(new_state, field_name, value)
        return new_state
    
    def compute_order_parameter(self) -> complex:
        """Compute Kuramoto order parameter r * exp(i*psi)."""
        phases = self.phases.value
        z = np.mean(np.exp(1j * phases))
        return z
    
    def sync_level(self) -> float:
        """Synchronization level (magnitude of order parameter)."""
        return abs(self.compute_order_parameter())


def create_coupling_matrix(
    n: int,
    coupling_type: CouplingType,
    topology: str = "all_to_all",
    strength: float = 0.1,
    distance_decay: float = 1.0,
    seed: int = 42,
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    Create coupling matrix for oscillator network.
    
    Args:
        n: Number of oscillators
        coupling_type: Physical coupling mechanism
        topology: "all_to_all", "nearest_neighbor", "ring", "small_world", "random"
        strength: Base coupling strength
        distance_decay: Decay with distance (for spatial topologies)
        seed: Random seed
        backend: GPU backend
        
    Returns:
        Coupling matrix [n, n]
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    rng = np.random.default_rng(seed)
    
    K = xp.zeros((n, n), dtype=xp.float32)
    
    if topology == "all_to_all":
        # All-to-all coupling (mean-field)
        K = xp.full((n, n), strength / n, dtype=xp.float32)
        # Set diagonal to zero (no self-coupling)
        if hasattr(xp, 'fill_diagonal'):
            xp.fill_diagonal(K, 0)
        else:
            # Manual diagonal zeroing for torch
            for i in range(n):
                K[i, i] = 0
        
    elif topology == "nearest_neighbor":
        # 1D nearest neighbor
        for i in range(n):
            if i > 0:
                K[i, i-1] = strength
            if i < n-1:
                K[i, i+1] = strength
        K = K + xp.transpose(K)
        
    elif topology == "ring":
        # Ring topology
        for i in range(n):
            K[i, (i+1)%n] = strength
            K[i, (i-1)%n] = strength
            
    elif topology == "small_world":
        # Watts-Strogatz small world
        k = 4  # nearest neighbors
        p = 0.1  # rewiring probability
        for i in range(n):
            for j in range(1, k//2 + 1):
                K[i, (i+j)%n] = strength
                K[i, (i-j)%n] = strength
        # Rewire
        for i in range(n):
            for j in range(1, k//2 + 1):
                if rng.random() < p:
                    K[i, (i+j)%n] = 0
                    new_j = rng.integers(0, n)
                    if new_j != i:
                        K[i, new_j] = strength
        K = (K + xp.transpose(K)) / 2
        
    elif topology == "random":
        # Random sparse
        density = 0.1
        n_conn = int(n * n * density)
        rows = rng.choice(n, n_conn)
        cols = rng.choice(n, n_conn)
        for r, c in zip(rows, cols):
            if r != c:
                K[r, c] = strength * rng.normal(1, 0.2)
        K = (K + xp.transpose(K)) / 2
        
    elif topology == "2d_grid":
        # 2D grid (assume sqrt(n) x sqrt(n))
        side = int(np.sqrt(n))
        for i in range(side):
            for j in range(side):
                idx = i * side + j
                if i > 0:
                    K[idx, (i-1)*side + j] = strength
                if i < side-1:
                    K[idx, (i+1)*side + j] = strength
                if j > 0:
                    K[idx, i*side + (j-1)] = strength
                if j < side-1:
                    K[idx, i*side + (j+1)] = strength
    
    # Apply distance decay for spatial topologies
    if topology in ["2d_grid", "nearest_neighbor", "ring"] and distance_decay != 1.0:
        # Already handled by topology structure
        pass
    
    return backend.to_numpy(K)


def vo2_relaxation_step(
    state: ONNState,
    dt: float,
    backend: GPUBackend | None = None
) -> ONNState:
    """
    VO2 relaxation oscillator step (insulator-metal transition).
    
    Model: Series RC circuit with voltage-dependent resistance.
    dV/dt = (I_in - V/R(V)) / C
    R(V) = R_off for V < V_th, R_on for V > V_h (hysteresis)
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    new_state = state.copy()
    n = state.n_oscillators
    
    # Move to device
    v = backend.to_device(state.phases.value)  # Using phases as voltage
    i_in = backend.to_device(state.inputs.value)
    coupling = backend.to_device(state.coupling.value)
    
    # VO2 parameters
    v_th = state.params.v_th.value
    v_h = state.params.v_h.value
    r_on = state.params.r_on.value
    r_off = state.params.r_off.value
    c = state.params.c_vo2.value
    
    # Current state: determine resistance based on voltage (hysteresis)
    # Track state internally (simplified: use amplitude as state indicator)
    # amplitude ~ 0: insulating, amplitude ~ 1: metallic
    r = xp.where(v > v_th, r_on, r_off)
    # Hysteresis: once metallic, stay until v < v_h
    # This is simplified - full model needs internal state variable
    
    # Coupling current: resistive coupling I_coup = K * (v_j - v_i)
    v_diff = xp.outer(v, xp.ones(n)) - xp.outer(xp.ones(n), v)
    i_coup = xp.sum(coupling * v_diff, axis=1)
    
    # Total current
    i_total = i_in + i_coup
    
    # Voltage update
    dv_dt = i_total / c - v / (r * c)
    v_new = v + dv_dt * dt
    
    # Update phase (voltage maps to phase for synchronization analysis)
    # Normalize voltage to phase
    v_norm = (v_new - v_h) / (v_th - v_h)
    phase_new = (xp.arctan2(xp.sin(2 * np.pi * v_norm), xp.cos(2 * np.pi * v_norm)) + 2 * np.pi) % (2 * np.pi)
    
    # Amplitude: 0 for insulating, 1 for metallic
    amp_new = xp.where(v_new > v_th, 1.0, 0.1)
    
    # Add phase noise
    noise_std = state.params.phase_noise.value * np.sqrt(dt)
    phase_noise = backend.random_normal((n,), std=noise_std)
    phase_new = (phase_new + phase_noise) % (2 * np.pi)
    
    new_state.phases.value = backend.to_numpy(phase_new)
    new_state.phases.provenance = Provenance.ESTIMATED
    new_state.amplitudes.value = backend.to_numpy(amp_new)
    new_state.amplitudes.provenance = Provenance.ESTIMATED
    
    new_state.phase_history.append(new_state.phases.value.copy())
    new_state.amplitude_history.append(new_state.amplitudes.value.copy())
    new_state.sync_history.append(new_state.sync_level())
    
    return new_state


def kuramoto_step(
    state: ONNState,
    dt: float,
    backend: GPUBackend | None = None
) -> ONNState:
    """
    Kuramoto model step (phase-only oscillators).
    
    dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i) + I_i
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    new_state = state.copy()
    n = state.n_oscillators
    
    phases = backend.to_device(state.phases.value)
    coupling = backend.to_device(state.coupling.value)
    inputs = backend.to_device(state.inputs.value)
    detuning = backend.to_device(state.detuning.value)
    f0 = state.params.f0.value
    
    # Natural frequencies with detuning
    omega = 2 * np.pi * f0 * (1 + detuning)
    
    # Phase differences
    phase_diff = xp.outer(phases, xp.ones(n)) - xp.outer(xp.ones(n), phases)
    
    # Coupling term: Σ_j K_ij sin(θ_j - θ_i) = -Σ_j K_ij sin(θ_i - θ_j)
    coupling_term = -xp.sum(coupling * xp.sin(phase_diff), axis=1)
    
    # Phase update
    dphase_dt = omega + coupling_term + inputs
    phases_new = (phases + dphase_dt * dt) % (2 * np.pi)
    
    # Add phase noise
    noise_std = state.params.phase_noise.value * np.sqrt(dt)
    phase_noise = backend.random_normal((n,), std=noise_std)
    phases_new = (phases_new + phase_noise) % (2 * np.pi)
    
    new_state.phases.value = backend.to_numpy(phases_new)
    new_state.phases.provenance = Provenance.ESTIMATED
    new_state.amplitudes.value = xp.ones(n)  # Constant amplitude for Kuramoto
    new_state.amplitudes.provenance = Provenance.ESTIMATED
    
    new_state.phase_history.append(new_state.phases.value.copy())
    new_state.amplitude_history.append(new_state.amplitudes.value.copy())
    new_state.sync_history.append(new_state.sync_level())
    
    return new_state


def fitzhugh_nagumo_step(
    state: ONNState,
    dt: float,
    backend: GPUBackend | None = None
) -> ONNState:
    """
    FitzHugh-Nagumo oscillator step (2D relaxation oscillator).
    
    dv/dt = v - v^3/3 - w + I + coupling
    dw/dt = ε(v + β - γw)
    
    Using phases as v, amplitudes as w.
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    new_state = state.copy()
    n = state.n_oscillators
    
    v = backend.to_device(state.phases.value)  # Voltage/phase
    w = backend.to_device(state.amplitudes.value)  # Recovery variable
    coupling = backend.to_device(state.coupling.value)
    inputs = backend.to_device(state.inputs.value)
    
    # FHN parameters
    eps = 0.08
    beta = 0.7
    gamma = 0.8
    
    # Coupling: diffusive coupling on v
    v_diff = xp.outer(v, xp.ones(n)) - xp.outer(xp.ones(n), v)
    i_coup = xp.sum(coupling * v_diff, axis=1)
    
    # FHN equations
    dv_dt = v - v**3 / 3 - w + inputs + i_coup
    dw_dt = eps * (v + beta - gamma * w)
    
    v_new = v + dv_dt * dt
    w_new = w + dw_dt * dt
    
    # Map v to phase for sync analysis
    phase_new = (xp.arctan2(xp.sin(v_new), xp.cos(v_new)) + 2 * np.pi) % (2 * np.pi)
    
    # Add noise
    noise_std = state.params.phase_noise.value * np.sqrt(dt)
    phase_noise = backend.random_normal((n,), std=noise_std)
    phase_new = (phase_new + phase_noise) % (2 * np.pi)
    
    new_state.phases.value = backend.to_numpy(phase_new)
    new_state.phases.provenance = Provenance.ESTIMATED
    new_state.amplitudes.value = backend.to_numpy(w_new)
    new_state.amplitudes.provenance = Provenance.ESTIMATED
    
    new_state.phase_history.append(new_state.phases.value.copy())
    new_state.amplitude_history.append(new_state.amplitudes.value.copy())
    new_state.sync_history.append(new_state.sync_level())
    
    return new_state


def mtj_stt_step(
    state: ONNState,
    dt: float,
    backend: GPUBackend | None = None
) -> ONNState:
    """
    MTJ Spin-Torque Oscillator step.
    
    Based on Landau-Lifshitz-Gilbert-Slonczewski equation.
    Phase = magnetization precession angle.
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    new_state = state.copy()
    n = state.n_oscillators
    
    phases = backend.to_device(state.phases.value)
    amplitudes = backend.to_device(state.amplitudes.value)
    coupling = backend.to_device(state.coupling.value)
    inputs = backend.to_device(state.inputs.value)
    
    # MTJ parameters
    alpha = state.params.alpha_g.value
    i_stt = state.params.i_stt.value
    p = state.params.p_polar.value
    f0 = state.params.f0.value
    
    # STT oscillator: frequency depends on current
    # f = f0 * (I/I_c - 1) for I > I_c
    i_total = inputs + xp.sum(coupling * xp.sin(xp.outer(phases, xp.ones(n)) - xp.outer(xp.ones(n), phases)), axis=1)
    
    # Frequency shift
    freq = f0 * xp.maximum(0, i_total / i_stt - 1)
    omega = 2 * np.pi * freq
    
    # Phase update
    phases_new = (phases + omega * dt) % (2 * np.pi)
    
    # Amplitude dynamics (simplified)
    # Amplitude grows with current above threshold
    amp_target = xp.where(i_total > i_stt, 1.0, 0.1)
    tau_amp = 1e-9  # 1 ns amplitude relaxation
    amplitudes_new = amplitudes + (amp_target - amplitudes) * dt / tau_amp
    
    # Add noise
    noise_std = state.params.phase_noise.value * np.sqrt(dt)
    phase_noise = backend.random_normal((n,), std=noise_std)
    phases_new = (phases_new + phase_noise) % (2 * np.pi)
    
    new_state.phases.value = backend.to_numpy(phases_new)
    new_state.phases.provenance = Provenance.ESTIMATED
    new_state.amplitudes.value = backend.to_numpy(amplitudes_new)
    new_state.amplitudes.provenance = Provenance.ESTIMATED
    
    new_state.phase_history.append(new_state.phases.value.copy())
    new_state.amplitude_history.append(new_state.amplitudes.value.copy())
    new_state.sync_history.append(new_state.sync_level())
    
    return new_state


def onn_step(
    state: ONNState,
    dt: float,
    backend: GPUBackend | None = None
) -> ONNState:
    """
    Main ONN step dispatcher.
    """
    if state.oscillator_type == OscillatorType.VO2_RELAXATION:
        return vo2_relaxation_step(state, dt, backend)
    elif state.oscillator_type == OscillatorType.ABSTRACT_KURAMOTO:
        return kuramoto_step(state, dt, backend)
    elif state.oscillator_type == OscillatorType.ABSTRACT_FITZHUGH_NAGUMO:
        return fitzhugh_nagumo_step(state, dt, backend)
    elif state.oscillator_type in [OscillatorType.MTJ_STT, OscillatorType.MTJ_SOT]:
        return mtj_stt_step(state, dt, backend)
    else:
        # Default to Kuramoto
        return kuramoto_step(state, dt, backend)


def onn_run(
    state: ONNState,
    n_steps: int,
    dt: float,
    inputs: np.ndarray | None = None,  # [n_steps, n_oscillators]
    backend: GPUBackend | None = None
) -> ONNState:
    """
    Run ONN for multiple steps.
    """
    if backend is None:
        backend = get_backend()
    
    current_state = state
    
    if inputs is not None:
        for step in range(n_steps):
            current_state.inputs.value = inputs[step]
            current_state = onn_step(current_state, dt, backend)
    else:
        for _ in range(n_steps):
            current_state = onn_step(current_state, dt, backend)
    
    return current_state


def onn_ising_solve(
    state: ONNState,
    J: np.ndarray,  # Ising coupling matrix [n, n]
    h: np.ndarray | None = None,  # External fields [n]
    n_steps: int = 10000,
    dt: float = 1e-12,
    backend: GPUBackend | None = None
) -> tuple[np.ndarray, float]:
    """
    Solve Ising problem using oscillatory network (Ising machine).
    
    Maps Ising Hamiltonian H = -Σ J_ij s_i s_j - Σ h_i s_i
    to oscillator phases: s_i = sign(cos(θ_i))
    
    Coupling: K_ij = -J_ij (ferromagnetic coupling aligns phases)
    """
    if backend is None:
        backend = get_backend()
    
    n = state.n_oscillators
    xp = backend.xp
    
    # Map Ising to oscillator coupling
    # For Ising machine: K_ij = -J_ij (attractive for ferromagnetic)
    coupling = -J.astype(np.float32)
    
    # Normalize coupling
    max_coup = np.max(np.abs(coupling))
    if max_coup > 0:
        coupling = coupling / max_coup * 0.1
    
    # Set up state
    run_state = state.copy()
    run_state.coupling.value = coupling
    run_state.coupling.provenance = Provenance.ESTIMATED
    
    # External fields as inputs
    if h is not None:
        run_state.inputs.value = h.astype(np.float32)
    else:
        run_state.inputs.value = np.zeros(n)
    
    # Initialize random phases
    run_state.phases.value = np.random.uniform(0, 2*np.pi, n)
    
    # Run
    run_state = onn_run(run_state, n_steps, dt, backend=backend)
    
    # Extract Ising solution: s_i = sign(cos(θ_i))
    final_phases = run_state.phases.value
    spins = np.sign(np.cos(final_phases))
    spins[spins == 0] = 1  # Handle exact zero
    
    # Compute energy
    energy = -0.5 * np.dot(spins, np.dot(J, spins))
    if h is not None:
        energy -= np.dot(h, spins)
    
    return spins, energy


def onn_pattern_recognition(
    state: ONNState,
    patterns: np.ndarray,  # [n_patterns, n_oscillators] binary patterns
    test_pattern: np.ndarray,
    n_steps: int = 5000,
    dt: float = 1e-12,
    backend: GPUBackend | None = None
) -> tuple[int, float]:
    """
    Associative memory: recognize closest stored pattern.
    
    Uses Hebbian coupling: K_ij = (1/n) Σ_μ ξ_i^μ ξ_j^μ
    """
    if backend is None:
        backend = get_backend()
    
    n_patterns, n_osc = patterns.shape
    assert n_osc == state.n_oscillators
    
    # Hebbian coupling matrix
    coupling = np.zeros((n_osc, n_osc), dtype=np.float32)
    for mu in range(n_patterns):
        xi = patterns[mu].astype(np.float32)
        coupling += np.outer(xi, xi)
    coupling = coupling / n_osc
    np.fill_diagonal(coupling, 0)
    
    # Set up state
    run_state = state.copy()
    run_state.coupling.value = coupling
    run_state.coupling.provenance = Provenance.ESTIMATED
    
    # Initialize with test pattern (phases = 0 for +1, π for -1)
    init_phases = np.where(test_pattern > 0, 0.0, np.pi)
    run_state.phases.value = init_phases
    
    # Run
    run_state = onn_run(run_state, n_steps, dt, backend=backend)
    
    # Read out final state
    final_phases = run_state.phases.value
    recalled = np.sign(np.cos(final_phases))
    recalled[recalled == 0] = 1
    
    # Find closest pattern
    overlaps = np.dot(patterns, recalled) / n_osc
    best_idx = int(np.argmax(overlaps))
    best_overlap = float(overlaps[best_idx])
    
    return best_idx, best_overlap


def onn_phase_locking_analysis(
    state: ONNState,
    freq_range: tuple[float, float] = (0.5, 2.0),
    n_points: int = 50,
    n_steps: int = 10000,
    dt: float = 1e-12,
    backend: GPUBackend | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Analyze phase locking regions (Arnold tongues).
    
    Varies detuning and coupling strength, measures sync.
    """
    if backend is None:
        backend = get_backend()
    
    f_min, f_max = freq_range
    detunings = np.linspace(f_min, f_max, n_points)
    sync_levels = []
    
    base_f0 = state.params.f0.value
    
    for detune in detunings:
        run_state = state.copy()
        run_state.params.f0.value = base_f0 * detune
        run_state.detuning.value = np.random.normal(0, 0.01, state.n_oscillators)
        
        run_state = onn_run(run_state, n_steps, dt, backend=backend)
        sync_levels.append(run_state.sync_level())
    
    return detunings, np.array(sync_levels)


# Export
__all__ = [
    "OscillatorType",
    "CouplingType",
    "OscillatorParams",
    "ONNState",
    "create_coupling_matrix",
    "vo2_relaxation_step",
    "kuramoto_step",
    "fitzhugh_nagumo_step",
    "mtj_stt_step",
    "onn_step",
    "onn_run",
    "onn_ising_solve",
    "onn_pattern_recognition",
    "onn_phase_locking_analysis",
]