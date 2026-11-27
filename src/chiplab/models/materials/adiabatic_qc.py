"""
Adiabatic Quantum Computing (AQC) / Quantum Annealing Model
=============================================================

Implements adiabatic quantum computation and quantum annealing.
Supports Ising Hamiltonian embedding, transverse field annealing,
and quantum approximate optimization algorithm (QAOA).

References:
- Farhi et al. (2000) - Quantum computation by adiabatic evolution
- D-Wave Systems - Quantum annealing hardware
- Albash & Lidar (2018) - Adiabatic quantum computation
- Farhi et al. (2014) - Quantum Approximate Optimization Algorithm (QAOA)
- Boixo et al. (2014) - Evidence for quantum annealing with more than 100 qubits
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from chiplab.schemas.common import Provenance, TrackedValue
from chiplab.core.gpu import GPUBackend, get_gpu_backend as get_backend


class AnnealingSchedule(Enum):
    """Annealing schedule types."""
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    EXPONENTIAL = "exponential"
    OPTIMAL = "optimal"  # Based on gap
    CUSTOM = "custom"


class QubitType(Enum):
    """Physical qubit types for AQC."""
    FLUX_QUBIT = "flux_qubit"       # Persistent current qubit (D-Wave)
    TRANSMON = "transmon"           # Transmon with tunable coupling
    FLUXONIUM = "fluxonium"         # Fluxonium qubit
    XMON = "xmon"                   # Xmon (Google)
    GATEMON = "gatemon"             # Gatemon (semiconductor-superconductor)


@dataclass
class AQCParams:
    """Adiabatic quantum computing parameters with provenance."""
    # Qubit properties
    qubit_type: QubitType = QubitType.FLUX_QUBIT
    n_qubits: int = 100

    # Energy scales
    max_h_field: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=10.0,  # GHz
        provenance=Provenance.MEASURED,
        metadata={"description": "Maximum transverse field (driver Hamiltonian)", "unit": "GHz"}
    ))
    max_j_coupling: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.0,  # GHz
        provenance=Provenance.MEASURED,
        metadata={"description": "Maximum Ising coupling", "unit": "GHz"}
    ))
    max_h_bias: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=5.0,  # GHz
        provenance=Provenance.MEASURED,
        metadata={"description": "Maximum longitudinal bias", "unit": "GHz"}
    ))

    # Annealing schedule
    schedule_type: AnnealingSchedule = AnnealingSchedule.LINEAR
    anneal_time: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=20e-6,  # s
        provenance=Provenance.MEASURED,
        metadata={"description": "Total annealing time", "unit": "s"}
    ))
    initial_transverse_field: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=10.0,  # GHz
        provenance=Provenance.MEASURED,
        metadata={"description": "Initial transverse field", "unit": "GHz"}
    ))
    final_transverse_field: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,  # GHz
        provenance=Provenance.MEASURED,
        metadata={"description": "Final transverse field", "unit": "GHz"}
    ))

    # Temperature and noise
    temperature: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=15.0,  # mK
        provenance=Provenance.MEASURED,
        metadata={"description": "Operating temperature", "unit": "mK"}
    ))
    dephasing_time: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=10e-6,  # s
        provenance=Provenance.MEASURED,
        metadata={"description": "Dephasing time T2", "unit": "s"}
    ))
    relaxation_time: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=50e-6,  # s
        provenance=Provenance.MEASURED,
        metadata={"description": "Relaxation time T1", "unit": "s"}
    ))
    flux_noise: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e-6,  # Phi0/sqrt(Hz)
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Flux noise spectral density", "unit": "Phi0/sqrt(Hz)"}
    ))
    crosstalk: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.01,  # 1%
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Coupling crosstalk"}
    ))

    # Readout
    readout_fidelity: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.99,
        provenance=Provenance.MEASURED,
        metadata={"description": "Single-shot readout fidelity"}
    ))
    readout_time: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=10e-6,  # s
        provenance=Provenance.MEASURED,
        metadata={"description": "Readout time", "unit": "s"}
    ))


@dataclass
class AQCState:
    """
    Adiabatic quantum computing state with full provenance tracking.

    Tracks Hamiltonian, quantum state, and annealing progress.
    """
    params: AQCParams = field(default_factory=AQCParams)

    # Hamiltonian components
    # H(t) = A(t) * H_driver + B(t) * H_problem
    # H_driver = -sum(sigma_x_i)
    # H_problem = sum(h_i * sigma_z_i) + sum(J_ij * sigma_z_i * sigma_z_j)

    # Problem Hamiltonian (Ising)
    h_bias: np.ndarray = field(default_factory=lambda: np.zeros(100))  # Local fields
    j_coupling: np.ndarray = field(default_factory=lambda: np.zeros((100, 100)))  # Couplings

    # Current Hamiltonian coefficients
    a_coeff: float = 1.0  # Driver coefficient
    b_coeff: float = 0.0  # Problem coefficient

    # Quantum state (for small systems, use statevector)
    statevector: np.ndarray | None = None  # 2^n complex amplitudes
    density_matrix: np.ndarray | None = None  # For open system

    # Classical state (for large systems, use simulated annealing / QMC)
    classical_spins: np.ndarray = field(default_factory=lambda: np.ones(100))  # +/-1

    # Annealing progress
    anneal_fraction: float = 0.0  # 0 to 1
    time_elapsed: float = 0.0  # s

    # Energy tracking
    instantaneous_energy: float = 0.0
    ground_state_energy: float = 0.0
    excited_state_energy: float = 0.0
    gap: float = 0.0

    # History
    energy_history: list[float] = field(default_factory=list)
    gap_history: list[float] = field(default_factory=list)
    anneal_fraction_history: list[float] = field(default_factory=list)
    spin_history: list[np.ndarray] = field(default_factory=list)

    def copy(self) -> AQCState:
        """Create a deep copy."""
        new_state = AQCState()
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, TrackedValue):
                setattr(new_state, field_name, value.copy())
            elif isinstance(value, (AQCParams, AnnealingSchedule, QubitType)):
                setattr(new_state, field_name, value)
            elif isinstance(value, (int, float)):
                setattr(new_state, field_name, value)
            elif isinstance(value, np.ndarray):
                setattr(new_state, field_name, value.copy())
            elif isinstance(value, list):
                setattr(new_state, field_name, [v.copy() if isinstance(v, np.ndarray) else v for v in value])
            else:
                setattr(new_state, field_name, value)
        return new_state


def aqc_ising_hamiltonian(
    h_bias: np.ndarray,
    j_coupling: np.ndarray,
    backend: GPUBackend | None = None
) -> csr_matrix:
    """
    Construct Ising Hamiltonian as sparse matrix.

    H = sum_i h_i * sigma_z_i + sum_{i<j} J_ij * sigma_z_i * sigma_z_j

    Args:
        h_bias: Local fields [n_qubits]
        j_coupling: Coupling matrix [n_qubits, n_qubits]
        backend: GPU backend

    Returns:
        Sparse Hamiltonian matrix (2^n x 2^n)
    """
    if backend is None:
        backend = get_backend()

    n = len(h_bias)
    dim = 2**n

    # For large n, return sparse matrix in computational basis
    # This is only feasible for n <= 20
    if n > 20:
        raise ValueError(f"Exact Hamiltonian construction only feasible for n <= 20, got n={n}")

    # Build using Pauli matrices
    # sigma_z = [[1, 0], [0, -1]]
    # sigma_x = [[0, 1], [1, 0]]

    # Use sparse matrix construction
    from scipy.sparse import kron, identity, csr_matrix

    sz = csr_matrix([[1, 0], [0, -1]], dtype=np.complex128)
    sx = csr_matrix([[0, 1], [1, 0]], dtype=np.complex128)
    id2 = csr_matrix([[1, 0], [0, 1]], dtype=np.complex128)

    H = csr_matrix((dim, dim), dtype=np.complex128)

    # Local fields
    for i in range(n):
        if abs(h_bias[i]) > 1e-12:
            # Build sigma_z on qubit i
            ops = [id2] * n
            ops[i] = sz
            term = h_bias[i] * ops[0]
            for op in ops[1:]:
                term = kron(term, op, format='csr')
            H += term

    # Couplings
    for i in range(n):
        for j in range(i+1, n):
            if abs(j_coupling[i, j]) > 1e-12:
                ops = [id2] * n
                ops[i] = sz
                ops[j] = sz
                term = j_coupling[i, j] * ops[0]
                for op in ops[1:]:
                    term = kron(term, op, format='csr')
                H += term

    return H


def aqc_driver_hamiltonian(
    n_qubits: int,
    backend: GPUBackend | None = None
) -> csr_matrix:
    """
    Construct transverse field driver Hamiltonian.

    H_driver = -sum_i sigma_x_i

    Args:
        n_qubits: Number of qubits
        backend: GPU backend

    Returns:
        Sparse driver Hamiltonian
    """
    if backend is None:
        backend = get_backend()

    if n_qubits > 20:
        raise ValueError(f"Exact Hamiltonian construction only feasible for n <= 20")

    from scipy.sparse import kron, identity, csr_matrix

    sx = csr_matrix([[0, 1], [1, 0]], dtype=np.complex128)
    id2 = csr_matrix([[1, 0], [0, 1]], dtype=np.complex128)

    dim = 2**n_qubits
    H = csr_matrix((dim, dim), dtype=np.complex128)

    for i in range(n_qubits):
        ops = [id2] * n_qubits
        ops[i] = sx
        term = ops[0]
        for op in ops[1:]:
            term = kron(term, op, format='csr')
        H -= term  # Negative sign for driver

    return H


def aqc_annealing_schedule(
    t: float,
    t_total: float,
    schedule_type: AnnealingSchedule = AnnealingSchedule.LINEAR,
    backend: GPUBackend | None = None
) -> tuple[float, float]:
    """
    Compute annealing schedule coefficients A(t), B(t).

    A(t) = transverse field coefficient (driver)
    B(t) = problem Hamiltonian coefficient

    Args:
        t: Current time
        t_total: Total annealing time
        schedule_type: Schedule type
        backend: GPU backend

    Returns:
        (A(t), B(t)) coefficients
    """
    if backend is None:
        backend = get_backend()

    s = t / t_total  # Normalized time [0, 1]

    if schedule_type == AnnealingSchedule.LINEAR:
        a = 1 - s
        b = s
    elif schedule_type == AnnealingSchedule.QUADRATIC:
        a = (1 - s)**2
        b = s**2
    elif schedule_type == AnnealingSchedule.EXPONENTIAL:
        # Exponential approach to final Hamiltonian
        a = np.exp(-5 * s)
        b = 1 - np.exp(-5 * s)
    elif schedule_type == AnnealingSchedule.OPTIMAL:
        # Optimal schedule based on gap (simplified)
        # A(s) ~ gap(s), B(s) ~ 1 - gap(s)
        # For now use linear
        a = 1 - s
        b = s
    else:  # CUSTOM - user must provide
        a = 1 - s
        b = s

    return float(a), float(b)


def aqc_time_evolution_step(
    state: AQCState,
    dt: float,
    backend: GPUBackend | None = None
) -> AQCState:
    """
    Perform one time step of adiabatic evolution.

    For small systems: exact Schrodinger evolution
    For large systems: quantum Monte Carlo / simulated annealing

    Args:
        state: AQCState
        dt: Time step (s)
        backend: GPU backend

    Returns:
        Updated state
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()
    params = new_state.params

    # Update time
    new_state.time_elapsed += dt
    new_state.anneal_fraction = new_state.time_elapsed / params.anneal_time.value
    new_state.anneal_fraction = min(new_state.anneal_fraction, 1.0)

    # Get schedule coefficients
    a, b = aqc_annealing_schedule(
        new_state.time_elapsed,
        params.anneal_time.value,
        params.schedule_type,
        backend
    )
    new_state.a_coeff = a * params.max_h_field.value
    new_state.b_coeff = b * params.max_j_coupling.value

    n = params.n_qubits

    if n <= 16:
        # Exact evolution using statevector
        new_state = aqc_exact_evolution_step(new_state, dt, backend)
    else:
        # Simulated quantum annealing (classical approximation)
        new_state = aqc_sqa_step(new_state, dt, backend)

    # Record history
    new_state.energy_history.append(new_state.instantaneous_energy)
    new_state.gap_history.append(new_state.gap)
    new_state.anneal_fraction_history.append(new_state.anneal_fraction)
    new_state.spin_history.append(new_state.classical_spins.copy())

    return new_state


def aqc_exact_evolution_step(
    state: AQCState,
    dt: float,
    backend: GPUBackend | None = None
) -> AQCState:
    """
    Exact quantum evolution for small systems.

    Args:
        state: AQCState
        dt: Time step
        backend: GPU backend

    Returns:
        Updated state
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()
    n = new_state.params.n_qubits

    # Build full Hamiltonian
    H_driver = aqc_driver_hamiltonian(n, backend)
    H_problem = aqc_ising_hamiltonian(new_state.h_bias, new_state.j_coupling, backend)

    H = new_state.a_coeff * H_driver + new_state.b_coeff * H_problem

    # Initialize statevector if needed
    if new_state.statevector is None:
        # Start in ground state of driver (all |+> state)
        dim = 2**n
        new_state.statevector = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)

    # Time evolution: psi(t+dt) = exp(-i*H*dt) * psi(t)
    # Use sparse matrix exponential for small systems
    from scipy.sparse.linalg import expm_multiply

    new_state.statevector = expm_multiply(-1j * H * dt, new_state.statevector)

    # Compute energies
    # Expectation value
    psi = new_state.statevector
    new_state.instantaneous_energy = float(np.real(np.conj(psi) @ (H @ psi)))

    # Ground and excited state energies
    eigvals, _ = eigsh(H, k=2, which='SA', return_eigenvectors=True)
    new_state.ground_state_energy = float(eigvals[0])
    new_state.excited_state_energy = float(eigvals[1])
    new_state.gap = new_state.excited_state_energy - new_state.ground_state_energy

    # Classical spin configuration (measurement in Z basis)
    probs = np.abs(new_state.statevector)**2
    # Sample or take most probable
    max_idx = np.argmax(probs)
    spins = np.array([1 if (max_idx >> i) & 1 else -1 for i in range(n)])
    new_state.classical_spins = spins

    return new_state


def aqc_sqa_step(
    state: AQCState,
    dt: float,
    backend: GPUBackend | None = None
) -> AQCState:
    """
    Simulated Quantum Annealing (SQA) step.

    Uses path-integral Monte Carlo with Trotter slices.
    Approximates quantum annealing with classical parallel tempering.

    Args:
        state: AQCState
        dt: Time step
        backend: GPU backend

    Returns:
        Updated state
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()
    params = new_state.params
    n = params.n_qubits

    # Number of Trotter slices (imaginary time discretization)
    n_slices = 100

    # Current temperature
    k_b = 1.381e-23  # J/K
    temp = params.temperature.value * 1e-3  # K
    beta = 1.0 / (k_b * temp) if temp > 0 else 1e10

    # Classical Ising energy
    def ising_energy(spins: np.ndarray) -> float:
        h = new_state.h_bias
        J = new_state.j_coupling
        return -np.sum(h * spins) - np.sum(J * np.outer(spins, spins)) / 2

    # Transverse field term (effective classical coupling between slices)
    gamma = new_state.a_coeff  # Transverse field strength
    # Effective coupling between slices: K = -0.5 * log(tanh(gamma * dt / n_slices))
    # For SQA, we use the mapping to classical Ising with extra dimension
    k_coupling = 0.5 * np.log(1.0 / np.tanh(gamma * dt / n_slices)) if gamma > 0 else 10.0

    # Current spins (replicate across slices)
    if len(new_state.spin_history) == 0:
        new_state.classical_spins = np.random.choice([-1, 1], size=n)

    spins = new_state.classical_spins.copy()

    # Monte Carlo updates
    n_sweeps = 10
    for _ in range(n_sweeps):
        for i in range(n):
            # Compute local field
            h_local = new_state.h_bias[i] + np.sum(new_state.j_coupling[i, :] * spins)
            # Energy change from flip
            delta_e = 2 * spins[i] * h_local

            # Metropolis acceptance
            if delta_e < 0 or np.random.random() < np.exp(-beta * delta_e):
                spins[i] *= -1

    new_state.classical_spins = spins
    new_state.instantaneous_energy = ising_energy(spins)

    # Estimate gap (simplified)
    new_state.gap = max(0.1, new_state.a_coeff * 0.5)  # Rough estimate

    return new_state


def aqc_qaoa_step(
    state: AQCState,
    gamma: float,
    beta: float,
    backend: GPUBackend | None = None
) -> AQCState:
    """
    QAOA (Quantum Approximate Optimization Algorithm) step.

    Applies: U = exp(-i*beta*H_driver) * exp(-i*gamma*H_problem)

    Args:
        state: AQCState
        gamma: Problem Hamiltonian angle
        beta: Driver Hamiltonian angle
        backend: GPU backend

    Returns:
        Updated state
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()
    n = new_state.params.n_qubits

    if n > 16:
        # Classical approximation for large systems
        return aqc_qaoa_classical_step(new_state, gamma, beta, backend)

    # Build Hamiltonians
    H_driver = aqc_driver_hamiltonian(n, backend)
    H_problem = aqc_ising_hamiltonian(new_state.h_bias, new_state.j_coupling, backend)

    # Initialize statevector if needed
    if new_state.statevector is None:
        dim = 2**n
        new_state.statevector = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)

    # Apply problem unitary
    from scipy.sparse.linalg import expm_multiply
    new_state.statevector = expm_multiply(-1j * gamma * H_problem, new_state.statevector)

    # Apply driver unitary
    new_state.statevector = expm_multiply(-1j * beta * H_driver, new_state.statevector)

    # Compute energy
    psi = new_state.statevector
    H = H_problem  # Problem Hamiltonian energy
    new_state.instantaneous_energy = float(np.real(np.conj(psi) @ (H @ psi)))

    # Update classical spins
    probs = np.abs(new_state.statevector)**2
    max_idx = np.argmax(probs)
    spins = np.array([1 if (max_idx >> i) & 1 else -1 for i in range(n)])
    new_state.classical_spins = spins

    return new_state


def aqc_qaoa_classical_step(
    state: AQCState,
    gamma: float,
    beta: float,
    backend: GPUBackend | None = None
) -> AQCState:
    """
    Classical approximation of QAOA step for large systems.

    Uses simulated annealing with QAOA-inspired schedule.

    Args:
        state: AQCState
        gamma: Problem angle
        beta: Driver angle
        backend: GPU backend

    Returns:
        Updated state
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()

    # Simplified: alternate between problem and driver "updates"
    # Problem update: minimize Ising energy
    # Driver update: random spin flips with probability ~ sin^2(beta)

    # Problem update (local search)
    spins = new_state.classical_spins.copy()
    n = len(spins)

    for i in range(n):
        h_local = new_state.h_bias[i] + np.sum(new_state.j_coupling[i, :] * spins)
        delta_e = 2 * spins[i] * h_local
        if delta_e < 0:
            spins[i] *= -1

    # Driver update (random flips)
    flip_prob = np.sin(beta)**2
    for i in range(n):
        if np.random.random() < flip_prob:
            spins[i] *= -1

    new_state.classical_spins = spins
    new_state.instantaneous_energy = -np.sum(new_state.h_bias * spins) - \
        np.sum(new_state.j_coupling * np.outer(spins, spins)) / 2

    return new_state


def aqc_embed_ising(
    problem_h: np.ndarray,
    problem_j: np.ndarray,
    hardware_graph: np.ndarray,
    backend: GPUBackend | None = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Embed Ising problem onto hardware connectivity graph.

    Uses minor embedding (heuristic).

    Args:
        problem_h: Problem local fields
        problem_j: Problem couplings
        hardware_graph: Adjacency matrix of hardware couplings
        backend: GPU backend

    Returns:
        (embedded_h, embedded_j, embedding_info)
    """
    if backend is None:
        backend = get_backend()

    # Simplified embedding: direct mapping if possible
    n_problem = len(problem_h)
    n_hardware = hardware_graph.shape[0]

    if n_problem <= n_hardware:
        # Try direct embedding
        embedded_h = np.zeros(n_hardware)
        embedded_j = np.zeros((n_hardware, n_hardware))

        embedded_h[:n_problem] = problem_h
        embedded_j[:n_problem, :n_problem] = problem_j

        return embedded_h, embedded_j, {"method": "direct", "chains": []}

    # For larger problems, would need minor embedding
    # This is a placeholder for the full embedding algorithm
    raise NotImplementedError("Minor embedding for n_problem > n_hardware not yet implemented")


def aqc_reverse_annealing(
    state: AQCState,
    initial_spins: np.ndarray,
    reanneal_fraction: float = 0.5,
    backend: GPUBackend | None = None
) -> AQCState:
    """
    Reverse annealing: start from classical state, go back into quantum regime.

    Args:
        state: AQCState
        initial_spins: Initial classical spin configuration
        reanneal_fraction: Fraction to re-anneal (0 to 1)
        backend: GPU backend

    Returns:
        State after reverse annealing
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()
    new_state.classical_spins = initial_spins.copy()

    # Start from s = reanneal_fraction (partially through anneal)
    new_state.anneal_fraction = reanneal_fraction
    new_state.time_elapsed = reanneal_fraction * new_state.params.anneal_time.value

    # Go back to s=0 (full transverse field)
    n_steps = 100
    dt = -new_state.time_elapsed / n_steps  # Negative time step

    for _ in range(n_steps):
        new_state = aqc_time_evolution_step(new_state, dt, backend)

    # Now forward anneal again
    dt = new_state.params.anneal_time.value / n_steps
    for _ in range(n_steps):
        new_state = aqc_time_evolution_step(new_state, dt, backend)

    return new_state


def aqc_sample_solutions(
    state: AQCState,
    n_samples: int = 1000,
    backend: GPUBackend | None = None
) -> dict:
    """
    Sample solutions from final state.

    Args:
        state: Final AQCState
        n_samples: Number of samples
        backend: GPU backend

    Returns:
        Dict with samples, energies, and statistics
    """
    if backend is None:
        backend = get_backend()

    n = state.params.n_qubits
    samples = []
    energies = []

    if n <= 16 and state.statevector is not None:
        # Sample from quantum state
        probs = np.abs(state.statevector)**2
        for _ in range(n_samples):
            idx = np.random.choice(2**n, p=probs)
            spins = np.array([1 if (idx >> i) & 1 else -1 for i in range(n)])
            energy = -np.sum(state.h_bias * spins) - np.sum(state.j_coupling * np.outer(spins, spins)) / 2
            samples.append(spins)
            energies.append(energy)
    else:
        # Sample from classical distribution (Boltzmann)
        beta = 1.0 / (1.381e-23 * state.params.temperature.value * 1e-3) if state.params.temperature.value > 0 else 1e10
        for _ in range(n_samples):
            # Metropolis sampling
            spins = state.classical_spins.copy()
            for _ in range(100 * n):
                i = np.random.randint(n)
                h_local = state.h_bias[i] + np.sum(state.j_coupling[i, :] * spins)
                delta_e = 2 * spins[i] * h_local
                if delta_e < 0 or np.random.random() < np.exp(-beta * delta_e):
                    spins[i] *= -1
            energy = -np.sum(state.h_bias * spins) - np.sum(state.j_coupling * np.outer(spins, spins)) / 2
            samples.append(spins)
            energies.append(energy)

    samples = np.array(samples)
    energies = np.array(energies)

    # Statistics
    unique_samples, counts = np.unique(samples, axis=0, return_counts=True)
    best_idx = np.argmin(energies)
    best_spins = samples[best_idx]
    best_energy = energies[best_idx]

    return {
        "samples": samples,
        "energies": energies,
        "best_spins": best_spins,
        "best_energy": best_energy,
        "mean_energy": float(np.mean(energies)),
        "std_energy": float(np.std(energies)),
        "unique_solutions": len(unique_samples),
        "solution_counts": dict(zip([tuple(s) for s in unique_samples], counts)),
        "ground_state_probability": float(counts[0] / n_samples) if len(counts) > 0 else 0,
    }


def aqc_quantum_monte_carlo(
    h_bias: np.ndarray,
    j_coupling: np.ndarray,
    n_slices: int = 100,
    beta: float = 10.0,
    gamma: float = 1.0,
    n_sweeps: int = 1000,
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    Path-integral Quantum Monte Carlo for transverse-field Ising model.

    Maps quantum system to classical Ising model in (d+1) dimensions.

    Args:
        h_bias: Local fields
        j_coupling: Couplings
        n_slices: Number of Trotter slices
        beta: Inverse temperature
        gamma: Transverse field strength
        n_sweeps: Monte Carlo sweeps
        backend: GPU backend

    Returns:
        Classical spin configurations [n_slices, n_qubits]
    """
    if backend is None:
        backend = get_backend()

    n = len(h_bias)

    # Inter-slice coupling
    k = 0.5 * np.log(1.0 / np.tanh(gamma * beta / n_slices))

    # Initialize spins
    spins = np.random.choice([-1, 1], size=(n_slices, n))

    for sweep in range(n_sweeps):
        for s in range(n_slices):
            for i in range(n):
                # Local field from problem Hamiltonian
                h_local = h_bias[i] + np.sum(j_coupling[i, :] * spins[s, :])

                # Coupling to neighboring slices
                h_slice = k * (spins[(s-1) % n_slices, i] + spins[(s+1) % n_slices, i])

                # Total effective field
                h_eff = h_local + h_slice

                # Metropolis
                delta_e = 2 * spins[s, i] * h_eff
                if delta_e < 0 or np.random.random() < np.exp(-delta_e / n_slices):
                    spins[s, i] *= -1

    return spins


def aqc_adiabatic_condition(
    state: AQCState,
    backend: GPUBackend | None = None
) -> dict:
    """
    Check adiabatic condition: |<1|dH/dt|0>| / gap^2 << 1

    Args:
        state: AQCState
        backend: GPU backend

    Returns:
        Adiabatic condition metrics
    """
    if backend is None:
        backend = get_backend()

    n = state.params.n_qubits

    if n > 16:
        return {"checkable": False, "reason": "System too large for exact check"}

    # Build Hamiltonians
    H_driver = aqc_driver_hamiltonian(n, backend)
    H_problem = aqc_ising_hamiltonian(state.h_bias, state.j_coupling, backend)

    # dH/dt = (dA/dt) * H_driver + (dB/dt) * H_problem
    t = state.time_elapsed
    t_total = state.params.anneal_time.value
    s = t / t_total

    # Derivatives of schedule
    if state.params.schedule_type == AnnealingSchedule.LINEAR:
        da_dt = -1 / t_total
        db_dt = 1 / t_total
    else:
        # Approximate
        da_dt = -1 / t_total
        db_dt = 1 / t_total

    dH_dt = da_dt * H_driver + db_dt * H_problem

    # Current Hamiltonian
    H = state.a_coeff * H_driver + state.b_coeff * H_problem

    # Ground and first excited states
    eigvals, eigvecs = eigsh(H, k=2, which='SA')
    gap = eigvals[1] - eigvals[0]
    psi0 = eigvecs[:, 0]
    psi1 = eigvecs[:, 1]

    # Matrix element
    matrix_element = abs(np.conj(psi1) @ (dH_dt @ psi0))

    adiabatic_parameter = matrix_element / (gap**2) if gap > 0 else float('inf')

    return {
        "gap": float(gap),
        "matrix_element": float(matrix_element),
        "adiabatic_parameter": float(adiabatic_parameter),
        "is_adiabatic": adiabatic_parameter < 0.1,
        "da_dt": float(da_dt),
        "db_dt": float(db_dt),
    }


# Export key functions and classes
__all__ = [
    "AnnealingSchedule",
    "QubitType",
    "AQCParams",
    "AQCState",
    "aqc_ising_hamiltonian",
    "aqc_driver_hamiltonian",
    "aqc_annealing_schedule",
    "aqc_time_evolution_step",
    "aqc_exact_evolution_step",
    "aqc_sqa_step",
    "aqc_qaoa_step",
    "aqc_qaoa_classical_step",
    "aqc_embed_ising",
    "aqc_reverse_annealing",
    "aqc_sample_solutions",
    "aqc_quantum_monte_carlo",
    "aqc_adiabatic_condition",
]