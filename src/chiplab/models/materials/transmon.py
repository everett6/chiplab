"""
Superconducting Transmon Qubit Model
=====================================

Implements transmon qubit physics for superconducting quantum computing.
Based on Josephson junction physics with capacitive shunting.

References:
- Koch et al. (2007) - Transmon qubit
- Blais et al. (2004) - Cavity QED with superconducting qubits
- Krantz et al. (2019) - Superconducting quantum computing review
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np

from chiplab.core.gpu import GPUBackend, get_gpu_backend as get_backend
from chiplab.schemas.common import Provenance, TrackedValue


class TransmonType(Enum):
    """Transmon qubit variants."""
    FIXED_FREQUENCY = "fixed_frequency"
    TUNABLE = "tunable"  # SQUID-based tunable
    FLUXONIUM = "fluxonium"  # Heavy fluxonium
    XMON = "xmon"  # Cross-shaped (Google)
    GMON = "gmon"  # Gmon (Rigetti)


@dataclass
class TransmonParams:
    """Transmon qubit parameters with provenance."""
    # Josephson junction
    ej: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=20e9 * 6.626e-34,  # 20 GHz * h
        provenance=Provenance.MEASURED,
        metadata={"description": "Josephson energy", "unit": "J"}
    ))
    ec: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=200e6 * 6.626e-34,  # 200 MHz * h
        provenance=Provenance.MEASURED,
        metadata={"description": "Charging energy", "unit": "J"}
    ))
    
    # Capacitances
    c_shunt: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=100e-15,
        provenance=Provenance.MEASURED,
        metadata={"description": "Shunt capacitance", "unit": "F"}
    ))
    c_junction: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=2e-15,
        provenance=Provenance.MEASURED,
        metadata={"description": "Junction capacitance", "unit": "F"}
    ))
    c_coupling: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e-15,
        provenance=Provenance.MEASURED,
        metadata={"description": "Coupling capacitance to resonator", "unit": "F"}
    ))
    
    # SQUID asymmetry (for tunable transmon)
    d: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "SQUID asymmetry parameter"}
    ))
    
    # Flux bias (for tunable)
    phi_ext: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "External flux bias", "unit": "Phi_0"}
    ))
    
    # Resonator parameters (for cQED)
    resonator_freq: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=6.5e9,
        provenance=Provenance.MEASURED,
        metadata={"description": "Resonator frequency", "unit": "Hz"}
    ))
    resonator_kappa: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e6,
        provenance=Provenance.MEASURED,
        metadata={"description": "Resonator decay rate", "unit": "Hz"}
    ))
    g_coupling: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=100e6,
        provenance=Provenance.MEASURED,
        metadata={"description": "Qubit-resonator coupling", "unit": "Hz"}
    ))
    
    # Noise and decoherence
    t1: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=50e-6,
        provenance=Provenance.MEASURED,
        metadata={"description": "Energy relaxation time", "unit": "s"}
    ))
    t2: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=30e-6,
        provenance=Provenance.MEASURED,
        metadata={"description": "Dephasing time", "unit": "s"}
    ))
    t_phi: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=60e-6,
        provenance=Provenance.MEASURED,
        metadata={"description": "Pure dephasing time", "unit": "s"}
    ))
    
    # Temperature
    temperature: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.015,
        provenance=Provenance.MEASURED,
        metadata={"description": "Operating temperature", "unit": "K"}
    ))
    
    # Derived (computed)
    ej_ec_ratio: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=100.0,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "E_J/E_C ratio"}
    ))
    transition_freq: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=5.0e9,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Qubit transition frequency (0->1)", "unit": "Hz"}
    ))
    anharmonicity: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=-200e6,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Anharmonicity (1->2 minus 0->1)", "unit": "Hz"}
    ))


@dataclass
class TransmonState:
    """
    Transmon qubit state with full provenance tracking.
    
    Tracks density matrix, populations, coherences.
    """
    # Parameters
    params: TransmonParams = field(default_factory=TransmonParams)
    transmon_type: TransmonType = TransmonType.FIXED_FREQUENCY
    
    # Hilbert space truncation
    n_levels: int = 4  # |0>, |1>, |2>, |3>
    
    # Density matrix (complex)
    rho: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=np.eye(4, dtype=complex),
        provenance=Provenance.ASSUMED,
        metadata={"description": "Density matrix"}
    ))
    
    # Drive parameters
    drive_freq: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=5.0e9,
        provenance=Provenance.MEASURED,
        metadata={"description": "Drive frequency", "unit": "Hz"}
    ))
    drive_amp: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Drive amplitude", "unit": "V"}
    ))
    drive_phase: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Drive phase", "unit": "rad"}
    ))
    
    # Flux bias (for tunable)
    flux_bias: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Flux bias", "unit": "Phi_0"}
    ))
    
    # History
    population_history: list[np.ndarray] = field(default_factory=list)
    coherence_history: list[np.ndarray] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize density matrix in ground state."""
        self.rho = TrackedValue(
            value=np.zeros((self.n_levels, self.n_levels), dtype=complex),
            provenance=Provenance.ASSUMED,
            metadata={"description": "Density matrix"}
        )
        self.rho.value[0, 0] = 1.0  # Ground state
    
    def copy(self) -> TransmonState:
        """Create a deep copy."""
        new_state = TransmonState(
            params=self.params,
            transmon_type=self.transmon_type,
            n_levels=self.n_levels
        )
        for field_name in self.__dataclass_fields__:
            if field_name in ['params', 'transmon_type', 'n_levels']:
                continue
            value = getattr(self, field_name)
            if isinstance(value, TrackedValue):
                setattr(new_state, field_name, value.copy())
            elif isinstance(value, (TransmonType, int, float)):
                setattr(new_state, field_name, value)
            elif isinstance(value, list):
                setattr(new_state, field_name, [v.copy() if isinstance(v, np.ndarray) else v for v in value])
            else:
                setattr(new_state, field_name, value)
        return new_state
    
    def populations(self) -> np.ndarray:
        """Get diagonal populations."""
        return np.real(np.diag(self.rho.value))
    
    def coherence(self, i: int, j: int) -> complex:
        """Get coherence rho_ij."""
        return self.rho.value[i, j]
    
    def excited_state_population(self) -> float:
        """Get |1> population."""
        return float(np.real(self.rho.value[1, 1]))
    
    def ground_state_population(self) -> float:
        """Get |0> population."""
        return float(np.real(self.rho.value[0, 0]))


def transmon_compute_spectrum(params: TransmonParams) -> tuple[np.ndarray, float, float]:
    """
    Compute transmon energy spectrum using perturbation theory.
    
    For transmon: E_m ≈ -E_J + sqrt(8 E_J E_C) (m + 1/2) - E_C/12 (6m^2 + 6m + 3)
    
    Returns:
        energies: Energy levels [n_levels]
        freq_01: 0->1 transition frequency
        anharm: Anharmonicity
    """
    h = 6.626e-34
    EJ = params.ej.value
    EC = params.ec.value
    
    # Plasma frequency
    omega_p = np.sqrt(8 * EJ * EC) / h  # Hz
    
    # Energy levels (perturbation theory)
    n_levels = 10
    energies = np.zeros(n_levels)
    for m in range(n_levels):
        energies[m] = -EJ + h * omega_p * (m + 0.5) - EC * (6*m**2 + 6*m + 3) / 12
    
    # Transition frequencies
    freq_01 = (energies[1] - energies[0]) / h
    freq_12 = (energies[2] - energies[1]) / h
    anharm = freq_12 - freq_01
    
    return energies, freq_01, anharm


def transmon_compute_wavefunctions(params: TransmonParams, n_levels: int = 4) -> np.ndarray:
    """
    Compute transmon wavefunctions in phase basis.
    
    Uses harmonic oscillator basis with anharmonic corrections.
    """
    EJ = params.ej.value
    EC = params.ec.value
    hbar = 1.055e-34
    
    # Harmonic oscillator length
    l_osc = (hbar**2 / (2 * EC * EJ))**0.25
    
    # Phase grid
    phi_max = 5 * l_osc
    n_grid = 512
    phi = np.linspace(-phi_max, phi_max, n_grid)
    dphi = phi[1] - phi[0]
    
    # Potential: -EJ * cos(phi) + phi^2 / (2 * L) for fluxonium
    # For transmon: -EJ * cos(phi) ≈ -EJ + EJ * phi^2 / 2 - EJ * phi^4 / 24
    V = -EJ * np.cos(phi)
    
    # Kinetic energy (finite difference)
    T = -4 * EC * (np.diag(-2*np.ones(n_grid)) + np.diag(np.ones(n_grid-1), 1) + np.diag(np.ones(n_grid-1), -1)) / dphi**2
    H = T + np.diag(V)
    
    # Diagonalize
    eigvals, eigvecs = np.linalg.eigh(H)
    
    # Return lowest n_levels wavefunctions
    return eigvecs[:, :n_levels], eigvals[:n_levels]


def transmon_hamiltonian(params: TransmonParams, n_levels: int = 4, flux: float = 0.0) -> np.ndarray:
    """
    Compute transmon Hamiltonian in charge/phase basis.
    
    H = 4 E_C n^2 - E_J cos(phi) for fixed frequency
    H = 4 E_C n^2 - E_J(Phi) cos(phi) for tunable (SQUID)
    """
    h = 6.626e-34
    EJ = params.ej.value
    EC = params.ec.value
    
    # Effective EJ for tunable transmon
    transmon_type = getattr(params, 'transmon_type', TransmonType.FIXED_FREQUENCY)
    if transmon_type == TransmonType.TUNABLE:
        # SQUID: EJ_eff = EJ * |cos(pi * Phi_ext / Phi_0)|
        # With asymmetry d: EJ_eff = EJ * sqrt(cos^2 + d^2 sin^2)
        phi_ext = flux * 2 * np.pi
        d = params.d.value
        EJ_eff = EJ * np.sqrt(np.cos(phi_ext)**2 + d**2 * np.sin(phi_ext)**2)
    else:
        EJ_eff = EJ
    
    # Charge basis Hamiltonian: H = 4 EC n^2 - EJ cos(phi)
    # In charge basis |n>, cos(phi) = 1/2 (|n><n+1| + |n+1><n|)
    n_max = n_levels * 2
    n_vals = np.arange(-n_max, n_max + 1)
    
    H = np.zeros((2*n_max + 1, 2*n_max + 1), dtype=complex)
    
    # Diagonal: charging energy
    for i, n in enumerate(n_vals):
        H[i, i] = 4 * EC * n**2
    
    # Off-diagonal: Josephson coupling
    for i in range(len(n_vals) - 1):
        H[i, i+1] = -EJ_eff / 2
        H[i+1, i] = -EJ_eff / 2
    
    # Diagonalize and return lowest n_levels
    eigvals, eigvecs = np.linalg.eigh(H)
    idx = np.argsort(eigvals)
    
    return eigvals[idx[:n_levels]], eigvecs[:, idx[:n_levels]]


def transmon_drive_hamiltonian(
    state: TransmonState,
    t: float,
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    Time-dependent drive Hamiltonian in qubit basis.
    
    H_d = Omega(t) * (sigma_x cos(omega_d t) + sigma_y sin(omega_d t))
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    n = state.n_levels
    
    # Drive parameters
    omega_d = 2 * np.pi * state.drive_freq.value
    amp = state.drive_amp.value
    phase = state.drive_phase.value
    
    # Rabi frequency (proportional to drive amplitude)
    # Omega = 2 * g * amp, where g is matrix element <0|n|1>
    g_matrix = np.sqrt(np.arange(1, n))  # Approximate charge matrix elements
    Omega = amp * g_matrix[0] * 1e6  # Scale factor
    
    # Rotating frame Hamiltonian
    H_drive = np.zeros((n, n), dtype=complex)
    
    for i in range(n-1):
        # Rotating wave approximation
        H_drive[i, i+1] = 0.5 * Omega * np.exp(-1j * phase)
        H_drive[i+1, i] = 0.5 * Omega * np.exp(1j * phase)
    
    return H_drive


def transmon_lindblad_operators(state: TransmonState) -> list[np.ndarray]:
    """
    Lindblad jump operators for decoherence.
    
    Returns list of jump operators: sqrt(gamma) * sigma_-, sqrt(gamma_phi) * sigma_z
    """
    n = state.n_levels
    ops = []
    
    # Relaxation: |1> -> |0>
    gamma_1 = 1.0 / state.params.t1.value
    if gamma_1 > 0:
        L1 = np.zeros((n, n), dtype=complex)
        L1[0, 1] = np.sqrt(gamma_1)
        ops.append(L1)
    
    # Higher level relaxation
    for i in range(1, n-1):
        gamma_i = 1.0 / state.params.t1.value * (i + 1)  # Approximate scaling
        Li = np.zeros((n, n), dtype=complex)
        Li[i, i+1] = np.sqrt(gamma_i)
        ops.append(Li)
    
    # Pure dephasing
    gamma_phi = 1.0 / state.params.t_phi.value - 0.5 / state.params.t1.value
    if gamma_phi > 0:
        L_phi = np.zeros((n, n), dtype=complex)
        for i in range(n):
            L_phi[i, i] = np.sqrt(gamma_phi) * i  # Phase proportional to level
        ops.append(L_phi)
    
    return ops


def transmon_master_equation_step(
    state: TransmonState,
    dt: float,
    backend: GPUBackend | None = None
) -> TransmonState:
    """
    Single step of Lindblad master equation.
    
    drho/dt = -i[H, rho] + sum_k (L_k rho L_k^dag - 1/2 {L_k^dag L_k, rho})
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    new_state = state.copy()
    
    n = state.n_levels
    
    # System Hamiltonian (in rotating frame)
    # H_sys = sum_m omega_m |m><m| + H_drive
    params = state.params
    energies, _, _ = transmon_compute_spectrum(params)
    H_sys = np.diag(energies[:n] / 6.626e-34)  # Convert to Hz
    
    # Add drive
    H_drive = transmon_drive_hamiltonian(state, 0.0, backend)
    H_total = H_sys + H_drive
    
    # Lindblad operators
    L_ops = transmon_lindblad_operators(state)
    
    # Current density matrix
    rho = state.rho.value
    
    # Commutator term: -i[H, rho]
    comm = -1j * (H_total @ rho - rho @ H_total)
    
    # Dissipator term
    dissip = np.zeros((n, n), dtype=complex)
    for L in L_ops:
        L_dag = L.conj().T
        dissip += L @ rho @ L_dag - 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L)
    
    # Total derivative
    drho_dt = comm + dissip
    
    # Update (Euler step)
    rho_new = rho + drho_dt * dt
    
    # Ensure trace = 1 and hermiticity
    rho_new = (rho_new + rho_new.conj().T) / 2
    rho_new = rho_new / np.trace(rho_new)
    
    new_state.rho.value = rho_new
    new_state.rho.provenance = Provenance.ESTIMATED
    
    # Record history
    new_state.population_history.append(np.real(np.diag(rho_new)).copy())
    new_state.coherence_history.append(rho_new.copy())
    
    return new_state


def transmon_rabi_oscillation(
    state: TransmonState,
    drive_amplitudes: np.ndarray,
    drive_duration: float,
    dt: float = 1e-9,
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    Simulate Rabi oscillations vs drive amplitude.
    
    Returns excited state population for each amplitude.
    """
    if backend is None:
        backend = get_backend()
    
    populations = []
    
    for amp in drive_amplitudes:
        run_state = state.copy()
        run_state.drive_amp.value = amp
        run_state.drive_freq.value = state.params.transition_freq.value
        
        n_steps = int(drive_duration / dt)
        for _ in range(n_steps):
            run_state = transmon_master_equation_step(run_state, dt, backend)
        
        populations.append(run_state.excited_state_population())
    
    return np.array(populations)


def transmon_ramsey(
    state: TransmonState,
    wait_times: np.ndarray,
    dt: float = 1e-9,
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    Ramsey interference experiment.
    
    pi/2 - wait - pi/2 sequence.
    """
    if backend is None:
        backend = get_backend()
    
    signals = []
    pi_half_duration = 20e-9  # 20 ns pi/2 pulse
    
    for wait_time in wait_times:
        run_state = state.copy()
        run_state.drive_freq.value = state.params.transition_freq.value
        run_state.drive_amp.value = np.pi / (2 * pi_half_duration)  # Calibrated pi/2
        
        # First pi/2 pulse
        n_pulse = int(pi_half_duration / dt)
        for _ in range(n_pulse):
            run_state = transmon_master_equation_step(run_state, dt, backend)
        
        # Wait (no drive)
        run_state.drive_amp.value = 0.0
        n_wait = int(wait_time / dt)
        for _ in range(n_wait):
            run_state = transmon_master_equation_step(run_state, dt, backend)
        
        # Second pi/2 pulse
        run_state.drive_amp.value = np.pi / (2 * pi_half_duration)
        for _ in range(n_pulse):
            run_state = transmon_master_equation_step(run_state, dt, backend)
        
        signals.append(run_state.excited_state_population())
    
    return np.array(signals)


def transmon_t1_measurement(
    state: TransmonState,
    wait_times: np.ndarray,
    dt: float = 1e-6,
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    T1 measurement: pi pulse - wait - measure.
    """
    if backend is None:
        backend = get_backend()
    
    signals = []
    pi_duration = 40e-9
    
    for wait_time in wait_times:
        run_state = state.copy()
        run_state.drive_freq.value = state.params.transition_freq.value
        run_state.drive_amp.value = np.pi / pi_duration
        
        # Pi pulse
        n_pulse = int(pi_duration / dt)
        for _ in range(n_pulse):
            run_state = transmon_master_equation_step(run_state, dt, backend)
        
        # Wait (no drive)
        run_state.drive_amp.value = 0.0
        n_wait = int(wait_time / dt)
        for _ in range(n_wait):
            run_state = transmon_master_equation_step(run_state, dt, backend)
        
        signals.append(run_state.excited_state_population())
    
    return np.array(signals)


def transmon_t2_echo(
    state: TransmonState,
    wait_times: np.ndarray,
    dt: float = 1e-9,
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    Hahn echo: pi/2 - wait/2 - pi - wait/2 - pi/2 - measure.
    """
    if backend is None:
        backend = get_backend()
    
    signals = []
    pi_half_duration = 20e-9
    pi_duration = 40e-9
    
    for wait_time in wait_times:
        run_state = state.copy()
        run_state.drive_freq.value = state.params.transition_freq.value
        run_state.drive_amp.value = np.pi / (2 * pi_half_duration)
        
        # First pi/2
        n_pulse = int(pi_half_duration / dt)
        for _ in range(n_pulse):
            run_state = transmon_master_equation_step(run_state, dt, backend)
        
        # Wait/2
        run_state.drive_amp.value = 0.0
        n_wait = int(wait_time / 2 / dt)
        for _ in range(n_wait):
            run_state = transmon_master_equation_step(run_state, dt, backend)
        
        # Pi pulse
        run_state.drive_amp.value = np.pi / pi_duration
        n_pi = int(pi_duration / dt)
        for _ in range(n_pi):
            run_state = transmon_master_equation_step(run_state, dt, backend)
        
        # Wait/2
        run_state.drive_amp.value = 0.0
        for _ in range(n_wait):
            run_state = transmon_master_equation_step(run_state, dt, backend)
        
        # Final pi/2
        run_state.drive_amp.value = np.pi / (2 * pi_half_duration)
        for _ in range(n_pulse):
            run_state = transmon_master_equation_step(run_state, dt, backend)
        
        signals.append(run_state.excited_state_population())
    
    return np.array(signals)


def transmon_spectroscopy(
    state: TransmonState,
    probe_freqs: np.ndarray,
    probe_duration: float = 10e-6,
    dt: float = 1e-9,
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    Qubit spectroscopy: measure excited population vs probe frequency.
    """
    if backend is None:
        backend = get_backend()
    
    signals = []
    
    for freq in probe_freqs:
        run_state = state.copy()
        run_state.drive_freq.value = freq
        run_state.drive_amp.value = 1e-3  # Weak probe
        
        n_steps = int(probe_duration / dt)
        for _ in range(n_steps):
            run_state = transmon_master_equation_step(run_state, dt, backend)
        
        signals.append(run_state.excited_state_population())
    
    return np.array(signals)


def transmon_two_qubit_gate(
    state1: TransmonState,
    state2: TransmonState,
    coupling_freq: float,
    gate_time: float,
    dt: float = 1e-9,
    backend: GPUBackend | None = None
) -> tuple[TransmonState, TransmonState]:
    """
    Simulate two-qubit gate (iSWAP, CZ, etc.) via cross-resonance or parametric coupling.
    
    Simplified: direct exchange coupling H = g (sigma1+ sigma2- + h.c.)
    """
    if backend is None:
        backend = get_backend()
    
    # This is a placeholder for full two-qubit simulation
    # Would need tensor product Hilbert space
    return state1.copy(), state2.copy()


# Export
__all__ = [
    "TransmonType",
    "TransmonParams",
    "TransmonState",
    "transmon_compute_spectrum",
    "transmon_compute_wavefunctions",
    "transmon_hamiltonian",
    "transmon_drive_hamiltonian",
    "transmon_lindblad_operators",
    "transmon_master_equation_step",
    "transmon_rabi_oscillation",
    "transmon_ramsey",
    "transmon_t1_measurement",
    "transmon_t2_echo",
    "transmon_spectroscopy",
    "transmon_two_qubit_gate",
]