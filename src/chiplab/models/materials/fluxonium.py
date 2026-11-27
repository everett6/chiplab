"""
Fluxonium Qubit Model
======================

Implements fluxonium qubit physics - a superconducting qubit with large inductance
(josephson junction array or high-kinetic-inductance element) for protected qubits.

References:
- Manucharyan et al. (2009) - Fluxonium qubit
- Pop et al. (2014) - Coherent suppression of electromagnetic dissipation
- Nguyen et al. (2019) - Fluxonium with long coherence times
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np

from chiplab.core.gpu import GPUBackend, get_gpu_backend as get_backend
from chiplab.schemas.common import Provenance, TrackedValue


class FluxoniumType(Enum):
    """Fluxonium variants."""
    STANDARD = "standard"           # Single junction + large inductor
    HEAVY = "heavy"                 # E_L << E_C << E_J (heavy fluxonium)
    DUAL = "dual"                   # Two-junction fluxonium
    ZERO_PI = "0_pi"                # 0-pi qubit (protected)
    BIFLUXONIUM = "bifluxonium"     # Two fluxoniums coupled


@dataclass
class FluxoniumParams:
    """Fluxonium qubit parameters."""
    # Josephson energy
    ej: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=5e9 * 6.626e-34,  # 5 GHz * h
        provenance=Provenance.MEASURED,
        metadata={"description": "Josephson energy", "unit": "J"}
    ))
    
    # Charging energy
    ec: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e9 * 6.626e-34,  # 1 GHz * h
        provenance=Provenance.MEASURED,
        metadata={"description": "Charging energy", "unit": "J"}
    ))
    
    # Inductive energy (large inductance)
    el: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.5e9 * 6.626e-34,  # 0.5 GHz * h
        provenance=Provenance.MEASURED,
        metadata={"description": "Inductive energy", "unit": "J"}
    ))
    
    # External flux bias
    phi_ext: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.5,  # Half flux quantum for sweet spot
        provenance=Provenance.MEASURED,
        metadata={"description": "External flux bias", "unit": "Phi_0"}
    ))
    
    # Junction asymmetry (for dual junction)
    d: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Junction asymmetry"}
    ))
    
    # Capacitances
    c_shunt: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=50e-15,
        provenance=Provenance.MEASURED,
        metadata={"description": "Shunt capacitance", "unit": "F"}
    ))
    c_junction: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=2e-15,
        provenance=Provenance.MEASURED,
        metadata={"description": "Junction capacitance", "unit": "F"}
    ))
    
    # Inductance implementation
    inductor_type: str = "junction_array"  # junction_array, kinetic_inductance, geometric
    n_junctions_array: int = 100  # For junction array inductor
    
    # Noise and decoherence
    t1: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=100e-6,
        provenance=Provenance.MEASURED,
        metadata={"description": "Energy relaxation time", "unit": "s"}
    ))
    t2: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=50e-6,
        provenance=Provenance.MEASURED,
        metadata={"description": "Dephasing time", "unit": "s"}
    ))
    
    # Temperature
    temperature: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.015,
        provenance=Provenance.MEASURED,
        metadata={"description": "Operating temperature", "unit": "K"}
    ))
    
    # Derived
    ej_ec_ratio: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=5.0,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "E_J/E_C ratio"}
    ))
    el_ec_ratio: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.5,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "E_L/E_C ratio"}
    ))
    transition_freq: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.5e9,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Qubit transition frequency", "unit": "Hz"}
    ))
    anharmonicity: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=-100e6,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Anharmonicity", "unit": "Hz"}
    ))


@dataclass
class FluxoniumState:
    """Fluxonium qubit state."""
    params: FluxoniumParams = field(default_factory=FluxoniumParams)
    fluxonium_type: FluxoniumType = FluxoniumType.STANDARD
    n_levels: int = 10  # Need more levels for fluxonium
    
    # Density matrix
    rho: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=np.zeros((10, 10), dtype=complex),
        provenance=Provenance.ASSUMED,
        metadata={"description": "Density matrix"}
    ))
    
    # Drive
    drive_freq: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.5e9,
        provenance=Provenance.MEASURED,
        metadata={"description": "Drive frequency", "unit": "Hz"}
    ))
    drive_amp: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Drive amplitude", "unit": "V"}
    ))
    
    # Flux bias
    flux_bias: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.5,
        provenance=Provenance.MEASURED,
        metadata={"description": "Flux bias", "unit": "Phi_0"}
    ))
    
    # History
    population_history: list[np.ndarray] = field(default_factory=list)
    
    def __post_init__(self):
        self.rho = TrackedValue(
            value=np.zeros((self.n_levels, self.n_levels), dtype=complex),
            provenance=Provenance.ASSUMED,
            metadata={"description": "Density matrix"}
        )
        self.rho.value[0, 0] = 1.0
    
    def copy(self) -> FluxoniumState:
        new_state = FluxoniumState(
            params=self.params,
            fluxonium_type=self.fluxonium_type,
            n_levels=self.n_levels
        )
        for field_name in self.__dataclass_fields__:
            if field_name in ['params', 'fluxonium_type', 'n_levels']:
                continue
            value = getattr(self, field_name)
            if isinstance(value, TrackedValue):
                setattr(new_state, field_name, value.copy())
            elif isinstance(value, (FluxoniumType, int, float)):
                setattr(new_state, field_name, value)
            elif isinstance(value, list):
                setattr(new_state, field_name, [v.copy() if isinstance(v, np.ndarray) else v for v in value])
            else:
                setattr(new_state, field_name, value)
        return new_state
    
    def populations(self) -> np.ndarray:
        return np.real(np.diag(self.rho.value))
    
    def excited_state_population(self) -> float:
        return float(np.real(self.rho.value[1, 1]))


def fluxonium_hamiltonian(params: FluxoniumParams, n_levels: int = 10, flux: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Fluxonium Hamiltonian in phase basis.
    
    H = 4 E_C n^2 - E_J cos(phi) + 0.5 E_L (phi - phi_ext)^2
    
    For heavy fluxonium: E_L << E_C << E_J
    """
    h = 6.626e-34
    EJ = params.ej.value
    EC = params.ec.value
    EL = params.el.value
    
    # Phase grid
    phi_max = 10 * np.pi
    n_grid = 1024
    phi = np.linspace(-phi_max, phi_max, n_grid)
    dphi = phi[1] - phi[0]
    
    # Potential: -EJ cos(phi) + 0.5 EL (phi - 2π*flux)^2
    phi_ext = 2 * np.pi * flux
    V = -EJ * np.cos(phi) + 0.5 * EL * (phi - phi_ext)**2
    
    # Kinetic energy
    T = -4 * EC * (np.diag(-2*np.ones(n_grid)) + np.diag(np.ones(n_grid-1), 1) + np.diag(np.ones(n_grid-1), -1)) / dphi**2
    H = T + np.diag(V)
    
    # Diagonalize
    eigvals, eigvecs = np.linalg.eigh(H)
    
    return eigvals[:n_levels], eigvecs[:, :n_levels]


def fluxonium_spectrum_vs_flux(params: FluxoniumParams, n_levels: int = 10, n_flux: int = 101) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute fluxonium spectrum as function of external flux.
    
    Returns:
        flux_values: External flux values [n_flux]
        energies: Energy levels [n_flux, n_levels]
    """
    flux_values = np.linspace(0, 1, n_flux)
    energies = np.zeros((n_flux, n_levels))
    
    for i, flux in enumerate(flux_values):
        evals, _ = fluxonium_hamiltonian(params, n_levels, flux)
        energies[i, :] = evals
    
    return flux_values, energies


def fluxonium_wavefunctions(params: FluxoniumParams, n_levels: int = 10, flux: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Get fluxonium wavefunctions in phase basis.
    """
    evals, evecs = fluxonium_hamiltonian(params, n_levels, flux)
    
    # Phase grid
    phi_max = 10 * np.pi
    n_grid = 1024
    phi = np.linspace(-phi_max, phi_max, n_grid)
    
    return phi, evecs


def fluxonium_matrix_elements(params: FluxoniumParams, n_levels: int = 10, flux: float = 0.5) -> dict:
    """
    Compute matrix elements for fluxonium.
    
    Returns dict with:
    - 'n': charge matrix elements <m|n|n'>
    - 'phi': phase matrix elements <m|phi|n'>
    - 'cos_phi': cos(phi) matrix elements
    - 'sin_phi': sin(phi) matrix elements
    """
    evals, evecs = fluxonium_hamiltonian(params, n_levels, flux)
    
    # Phase grid
    phi_max = 10 * np.pi
    n_grid = 1024
    phi = np.linspace(-phi_max, phi_max, n_grid)
    dphi = phi[1] - phi[0]
    
    # Position operator (phase)
    phi_op = np.diag(phi)
    
    # Momentum operator (charge) n = -i d/dphi
    n_op = -1j * (np.diag(np.ones(n_grid-1), 1) - np.diag(np.ones(n_grid-1), -1)) / (2 * dphi)
    
    # Transform to eigenbasis
    n_matrix = evecs[:, :n_levels].conj().T @ n_op @ evecs[:, :n_levels]
    phi_matrix = evecs[:, :n_levels].conj().T @ phi_op @ evecs[:, :n_levels]
    cos_phi_matrix = evecs[:, :n_levels].conj().T @ np.diag(np.cos(phi)) @ evecs[:, :n_levels]
    sin_phi_matrix = evecs[:, :n_levels].conj().T @ np.diag(np.sin(phi)) @ evecs[:, :n_levels]
    
    return {
        'n': n_matrix,
        'phi': phi_matrix,
        'cos_phi': cos_phi_matrix,
        'sin_phi': sin_phi_matrix,
        'energies': evals,
        'wavefunctions': evecs
    }


def fluxonium_transition_rates(params: FluxoniumParams, n_levels: int = 10, flux: float = 0.5) -> dict:
    """
    Compute transition rates for fluxonium.
    
    Uses matrix elements and noise spectral densities.
    """
    elements = fluxonium_matrix_elements(params, n_levels, flux)
    
    # Noise spectral densities (simplified)
    # Charge noise: S_n(omega) = A_n / omega
    # Flux noise: S_phi(omega) = A_phi / omega
    # Critical current noise: S_J(omega) = A_J / omega
    
    A_n = 1e-6  # Charge noise amplitude
    A_phi = 1e-6  # Flux noise amplitude
    A_J = 1e-6  # Critical current noise amplitude
    
    rates = {}
    energies = elements['energies']
    
    for i in range(n_levels):
        for j in range(i+1, n_levels):
            omega_ij = (energies[j] - energies[i]) / 6.626e-34  # Hz
            
            # Charge noise relaxation
            n_ij = elements['n'][i, j]
            gamma_n = 2 * np.pi * A_n * np.abs(n_ij)**2 * omega_ij
            
            # Flux noise relaxation
            phi_ij = elements['phi'][i, j]
            gamma_phi = 2 * np.pi * A_phi * np.abs(phi_ij)**2 * omega_ij
            
            # Total rate
            rates[(i, j)] = gamma_n + gamma_phi
            rates[(j, i)] = rates[(i, j)] * np.exp(-omega_ij * 6.626e-34 / (1.381e-23 * params.temperature.value))
    
    return rates


def fluxonium_parity_measurement(params: FluxoniumParams, n_levels: int = 10) -> dict:
    """
    Analyze parity protection in fluxonium.
    
    Heavy fluxonium has parity-protected states (even/odd parity).
    """
    # At half flux quantum (flux = 0.5), potential is symmetric double well
    # Ground state manifold: even and odd parity states
    flux = 0.5
    evals, evecs = fluxonium_hamiltonian(params, n_levels, flux)
    
    # Parity operator: P|phi> = |-phi>
    phi_max = 10 * np.pi
    n_grid = 1024
    phi = np.linspace(-phi_max, phi_max, n_grid)
    
    # Parity matrix in phase basis
    P = np.zeros((n_grid, n_grid))
    for i, ph in enumerate(phi):
        idx = np.argmin(np.abs(phi + ph))
        P[i, idx] = 1.0
    
    # Transform to eigenbasis
    P_eigen = evecs[:, :n_levels].conj().T @ P @ evecs[:, :n_levels]
    
    # Parity expectation values
    parity_expect = np.diag(P_eigen).real
    
    # Splitting between parity doublets
    splittings = []
    for i in range(0, n_levels-1, 2):
        if i+1 < n_levels:
            splitting = (evals[i+1] - evals[i]) / 6.626e-34  # Hz
            splittings.append(splitting)
    
    return {
        'parity_expectation': parity_expect,
        'parity_splittings': np.array(splittings),
        'energies': evals,
        'parity_operator': P_eigen
    }


def fluxonium_0_pi_analysis(params: FluxoniumParams, n_levels: int = 10) -> dict:
    """
    Analyze 0-pi qubit properties.
    
    0-pi qubit uses two fluxoniums with opposite flux bias for protected qubit.
    """
    # Two fluxoniums with flux bias +0.5 and -0.5
    # Logical |0> = |0>_+ |0>_-
    # Logical |1> = |1>_+ |1>_-
    
    # This is a simplified analysis
    evals, evecs = fluxonium_hamiltonian(params, n_levels, 0.5)
    
    # Matrix elements for logical operations
    elements = fluxonium_matrix_elements(params, n_levels, 0.5)
    
    # Protection: matrix elements between logical states
    # <0|n|1> should be small for protection
    n_01 = elements['n'][0, 1]
    n_02 = elements['n'][0, 2]
    
    return {
        'n_01': n_01,
        'n_02': n_02,
        'protection_ratio': np.abs(n_02 / n_01) if np.abs(n_01) > 0 else 0,
        'energies': evals
    }


def fluxonium_master_equation_step(
    state: FluxoniumState,
    dt: float,
    backend: GPUBackend | None = None
) -> FluxoniumState:
    """
    Lindblad master equation step for fluxonium.
    """
    if backend is None:
        backend = get_backend()
    
    new_state = state.copy()
    n = state.n_levels
    
    # Hamiltonian at current flux
    evals, _ = fluxonium_hamiltonian(state.params, n, state.flux_bias.value)
    H = np.diag(evals / 6.626e-34)  # Hz
    
    # Drive
    omega_d = 2 * np.pi * state.drive_freq.value
    amp = state.drive_amp.value
    # Charge coupling
    elements = fluxonium_matrix_elements(state.params, n, state.flux_bias.value)
    n_matrix = elements['n']
    
    H_drive = amp * n_matrix * np.cos(omega_d * 0)  # Simplified
    
    # Total Hamiltonian
    H_total = H + H_drive
    
    # Lindblad operators
    rates = fluxonium_transition_rates(state.params, n, state.flux_bias.value)
    L_ops = []
    
    for (i, j), rate in rates.items():
        if rate > 0:
            L = np.zeros((n, n), dtype=complex)
            L[i, j] = np.sqrt(rate)
            L_ops.append(L)
    
    # Pure dephasing
    gamma_phi = 1.0 / state.params.t2.value - 0.5 / state.params.t1.value
    if gamma_phi > 0:
        L_phi = np.zeros((n, n), dtype=complex)
        for i in range(n):
            L_phi[i, i] = np.sqrt(gamma_phi) * i
        L_ops.append(L_phi)
    
    # Master equation step
    rho = state.rho.value
    comm = -1j * (H_total @ rho - rho @ H_total)
    
    dissip = np.zeros((n, n), dtype=complex)
    for L in L_ops:
        L_dag = L.conj().T
        dissip += L @ rho @ L_dag - 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L)
    
    drho_dt = comm + dissip
    rho_new = rho + drho_dt * dt
    rho_new = (rho_new + rho_new.conj().T) / 2
    rho_new = rho_new / np.trace(rho_new)
    
    new_state.rho.value = rho_new
    new_state.rho.provenance = Provenance.ESTIMATED
    new_state.population_history.append(np.real(np.diag(rho_new)).copy())
    
    return new_state


# Export
__all__ = [
    "FluxoniumType",
    "FluxoniumParams",
    "FluxoniumState",
    "fluxonium_hamiltonian",
    "fluxonium_spectrum_vs_flux",
    "fluxonium_wavefunctions",
    "fluxonium_matrix_elements",
    "fluxonium_transition_rates",
    "fluxonium_parity_measurement",
    "fluxonium_0_pi_analysis",
    "fluxonium_master_equation_step",
]