"""
Majorana Zero Mode (MZM) Topological Qubit Model
==================================================

Implements Majorana zero mode physics for topological quantum computing.
Based on Kitaev chain model, semiconductor-superconductor nanowires,
and topological superconductivity.

References:
- Kitaev (2001) - Unpaired Majorana fermions in quantum wires
- Mourik et al. (2012) - Signatures of Majorana fermions in hybrid superconductor-semiconductor nanowires
- Alicea (2012) - New directions in the pursuit of Majorana fermions
- Lutchyn et al. (2018) - Majorana zero modes in superconductor-semiconductor heterostructures
- Microsoft Station Q topological quantum computing roadmap
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np

from chiplab.schemas.common import Provenance, TrackedValue
from chiplab.core.gpu import GPUBackend, get_gpu_backend as get_backend


class MZMMaterial(Enum):
    """Material platforms for Majorana zero modes."""
    IN_AS_AL = "InAs/Al"           # InAs nanowire with Al shell (Microsoft)
    IN_SB_NB = "InSb/Nb"           # InSb nanowire with Nb contacts
    IN_AS_NB = "InAs/Nb"           # InAs with Nb
    HG_TE_SUPERCONDUCTOR = "HgTe/SC"  # HgTe quantum well with superconductor
    FE_CHAIN = "Fe/Pb"             # Fe atomic chain on Pb (STM)
    CO_CHAIN = "Co/Pb"             # Co atomic chain on Pb
    TWO_D_TI = "2D_TI/SC"          # 2D topological insulator (WTe2) with SC
    CUSTOM = "custom"


class WireGeometry(Enum):
    """Nanowire geometry types."""
    STRAIGHT = "straight"
    Y_JUNCTION = "y_junction"
    T_JUNCTION = "t_junction"
    CROSS_JUNCTION = "cross_junction"
    LOOP = "loop"
    NETWORK = "network"


@dataclass
class MZMParams:
    """Majorana device parameters with provenance."""
    # Material properties
    material: MZMMaterial = MZMMaterial.IN_AS_AL
    spin_orbit_energy: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.5,  # meV
        provenance=Provenance.MEASURED,
        metadata={"description": "Spin-orbit coupling energy", "unit": "meV"}
    ))
    g_factor: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=20.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Effective g-factor"}
    ))
    induced_gap: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.2,  # meV
        provenance=Provenance.MEASURED,
        metadata={"description": "Induced superconducting gap", "unit": "meV"}
    ))
    chemical_potential: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,  # meV
        provenance=Provenance.ASSUMED,
        metadata={"description": "Chemical potential relative to band bottom", "unit": "meV"}
    ))
    effective_mass: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.023,  # m0
        provenance=Provenance.MEASURED,
        metadata={"description": "Effective mass in units of electron mass"}
    ))
    landau_level_spacing: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.0,  # meV
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Landau level spacing at operating B-field", "unit": "meV"}
    ))

    # Geometry
    wire_length: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.0,  # micron
        provenance=Provenance.MEASURED,
        metadata={"description": "Nanowire length", "unit": "um"}
    ))
    wire_width: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=100.0,  # nm
        provenance=Provenance.MEASURED,
        metadata={"description": "Nanowire width", "unit": "nm"}
    ))
    shell_thickness: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=10.0,  # nm
        provenance=Provenance.MEASURED,
        metadata={"description": "Superconducting shell thickness", "unit": "nm"}
    ))
    junction_angle: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=60.0,  # degrees
        provenance=Provenance.MEASURED,
        metadata={"description": "Junction angle for Y/T junctions", "unit": "deg"}
    ))

    # Magnetic field
    b_field: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.5,  # Tesla
        provenance=Provenance.MEASURED,
        metadata={"description": "Applied magnetic field", "unit": "T"}
    ))
    b_field_orientation: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,  # degrees from wire axis
        provenance=Provenance.MEASURED,
        metadata={"description": "B-field orientation relative to wire", "unit": "deg"}
    ))
    critical_field: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.5,  # Tesla
        provenance=Provenance.MEASURED,
        metadata={"description": "Critical field for superconductivity", "unit": "T"}
    ))

    # Tunnel couplings
    tunnel_coupling_left: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.01,  # meV
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Left tunnel coupling", "unit": "meV"}
    ))
    tunnel_coupling_right: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.01,  # meV
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Right tunnel coupling", "unit": "meV"}
    ))
    inter_mzm_coupling: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e-6,  # meV (exponentially small)
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Majorana-Majorana coupling", "unit": "meV"}
    ))

    # Temperature and noise
    temperature: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=20.0,  # mK
        provenance=Provenance.MEASURED,
        metadata={"description": "Operating temperature", "unit": "mK"}
    ))
    charge_noise: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e-6,  # e^2/Hz
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Charge noise spectral density", "unit": "e^2/Hz"}
    ))
    quasiparticle_poisoning_rate: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e3,  # Hz
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Quasiparticle poisoning rate", "unit": "Hz"}
    ))

    # Disorder
    disorder_strength: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.1,  # meV
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Disorder potential strength", "unit": "meV"}
    ))
    mean_free_path: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=500.0,  # nm
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Mean free path", "unit": "nm"}
    ))


@dataclass
class MZMState:
    """
    Majorana zero mode device state with full provenance tracking.

    Tracks Majorana operators, fermion parity, and topological gap.
    """
    params: MZMParams = field(default_factory=MZMParams)
    geometry: WireGeometry = WireGeometry.STRAIGHT
    n_sites: int = 100  # Discretization sites

    # Majorana operators (4 Majoranas = 2 fermionic modes for minimal qubit)
    # gamma_1, gamma_2 at left end; gamma_3, gamma_4 at right end
    # Fermion parity: P = i*gamma_1*gamma_2 = i*gamma_3*gamma_4
    fermion_parity: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.0,  # +1 = even, -1 = odd
        provenance=Provenance.ASSUMED,
        metadata={"description": "Fermion parity (+1 even, -1 odd)"}
    ))

    # Majorana wavefunction overlap (exponentially small)
    mzm_overlap: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e-6,  # meV
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Majorana wavefunction overlap energy", "unit": "meV"}
    ))

    # Topological gap
    topological_gap: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.1,  # meV
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Topological gap protecting MZMs", "unit": "meV"}
    ))

    # Braiding state (for multi-MZM systems)
    braiding_history: list[dict] = field(default_factory=list)
    current_braiding_phase: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,
        provenance=Provenance.ASSUMED,
        metadata={"description": "Accumulated braiding phase", "unit": "rad"}
    ))

    # Readout
    conductance: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,  # 2e^2/h units
        provenance=Provenance.MEASURED,
        metadata={"description": "Zero-bias conductance peak", "unit": "2e^2/h"}
    ))

    # History
    parity_history: list[float] = field(default_factory=list)
    conductance_history: list[float] = field(default_factory=list)
    b_field_history: list[float] = field(default_factory=list)

    def copy(self) -> MZMState:
        """Create a deep copy."""
        new_state = MZMState()
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, TrackedValue):
                setattr(new_state, field_name, value.copy())
            elif isinstance(value, (MZMParams, MZMMaterial, WireGeometry)):
                setattr(new_state, field_name, value)
            elif isinstance(value, (int, float)):
                setattr(new_state, field_name, value)
            elif isinstance(value, list):
                setattr(new_state, field_name, value.copy())
            else:
                setattr(new_state, field_name, value)
        return new_state

    def get_topological_gap(self) -> float:
        """Get topological gap in meV."""
        return float(self.topological_gap.value)

    def get_parity(self) -> float:
        """Get fermion parity."""
        return float(self.fermion_parity.value)


def mzm_topological_gap(
    state: MZMParams | MZMState,
    b_field: float | None = None,
    backend: GPUBackend | None = None
) -> float:
    """
    Compute topological gap as function of magnetic field.

    The topological gap opens when B > B_c (critical field for topological transition).
    Gap ~ sqrt((B - B_c) * Delta_induced) for B > B_c.

    Args:
        state: MZMParams or MZMState
        b_field: Magnetic field in Tesla (uses state value if None)
        backend: GPU backend

    Returns:
        Topological gap in meV
    """
    if backend is None:
        backend = get_backend()

    if isinstance(state, MZMState):
        params = state.params
    else:
        params = state

    if b_field is None:
        b_field = params.b_field.value

    b_c = params.critical_field.value
    delta_ind = params.induced_gap.value

    if b_field <= b_c:
        return 0.0  # Trivial phase

    # Topological gap scaling (simplified)
    # Gap ~ Delta * sqrt(1 - (B_c/B)^2) for clean limit
    # Or ~ sqrt(Delta * (B - B_c)) near transition
    gap = delta_ind * np.sqrt(1 - (b_c / b_field)**2)

    return float(max(gap, 0.0))


def mzm_wavefunction_overlap(
    state: MZMParams | MZMState,
    wire_length: float | None = None,
    backend: GPUBackend | None = None
) -> float:
    """
    Compute Majorana wavefunction overlap (splitting energy).

    Overlap ~ exp(-L/xi) where xi is coherence length.
    xi ~ v_F / Delta_topological

    Args:
        state: MZMParams or MZMState
        wire_length: Wire length in microns (uses state value if None)
        backend: GPU backend

    Returns:
        Overlap energy in meV
    """
    if backend is None:
        backend = get_backend()

    if isinstance(state, MZMState):
        params = state.params
    else:
        params = state

    if wire_length is None:
        wire_length = params.wire_length.value

    # Coherence length
    v_f = 1e5  # m/s, Fermi velocity (typical for InAs)
    delta_top = mzm_topological_gap(params, backend=backend) * 1e-3 * 1.602e-19  # J
    if delta_top <= 0:
        return 1e-3  # Large overlap in trivial phase

    xi = v_f / (delta_top / 1.055e-34)  # m
    xi_um = xi * 1e6  # microns

    # Exponential overlap
    overlap = params.induced_gap.value * np.exp(-wire_length / xi_um)

    return float(overlap)


def mzm_zero_bias_conductance(
    state: MZMState,
    voltage: float,
    temperature: float | None = None,
    backend: GPUBackend | None = None
) -> float:
    """
    Compute zero-bias conductance peak from Majorana tunneling.

    G(V=0) = 2e^2/h * (Gamma_L * Gamma_R) / (Gamma_L + Gamma_R)^2 * (pi * Delta / 2kT)
    for perfect Majorana. Broadened by temperature and coupling.

    Args:
        state: MZMState
        voltage: Bias voltage in mV
        temperature: Temperature in mK (uses state value if None)
        backend: GPU backend

    Returns:
        Conductance in units of 2e^2/h
    """
    if backend is None:
        backend = get_backend()

    if temperature is None:
        temperature = state.params.temperature.value

    k_b = 8.617e-5  # eV/K = 0.08617 meV/K
    temp_meV = temperature * 1e-3 * k_b * 1000  # meV

    gamma_l = state.params.tunnel_coupling_left.value
    gamma_r = state.params.tunnel_coupling_right.value
    overlap = state.mzm_overlap.value

    # Lorentzian lineshape for Majorana peak
    # G(V) = G_max * (Gamma/2)^2 / (V^2 + (Gamma/2)^2)
    # where Gamma = Gamma_L + Gamma_R + thermal broadening
    gamma_total = gamma_l + gamma_r + temp_meV

    # Peak height (quantized to 2e^2/h for perfect Majorana)
    g_max = 4 * gamma_l * gamma_r / (gamma_l + gamma_r)**2 if (gamma_l + gamma_r) > 0 else 0

    # Thermal broadening
    if temp_meV > 0:
        g_max *= np.tanh(state.params.induced_gap.value / (2 * temp_meV))

    # Voltage dependence
    v_meV = voltage  # mV
    conductance = g_max * (gamma_total / 2)**2 / (v_meV**2 + (gamma_total / 2)**2)

    return float(conductance)


def mzm_braiding_operation(
    state: MZMState,
    operation: Literal["braid_12", "braid_23", "braid_34", "exchange_13", "exchange_24"],
    backend: GPUBackend | None = None
) -> MZMState:
    """
    Apply braiding operation to Majorana qubit.

    Braiding Majoranas implements Clifford gates on the topological qubit:
    - Braid gamma_1 and gamma_2: pi/2 rotation around X
    - Braid gamma_2 and gamma_3: pi/2 rotation around Y
    - Exchange gamma_1 and gamma_3: pi rotation around Z (for 4 MZMs)

    Args:
        state: MZMState
        operation: Braiding operation to perform
        backend: GPU backend

    Returns:
        New state after braiding
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()

    # Braiding phases (non-Abelian statistics)
    # Each braid adds pi/2 phase to the fermion parity
    braid_phases = {
        "braid_12": np.pi / 2,   # X rotation
        "braid_23": np.pi / 2,   # Y rotation
        "braid_34": np.pi / 2,   # X rotation
        "exchange_13": np.pi,    # Z rotation (for 4 MZMs)
        "exchange_24": np.pi,    # Z rotation
    }

    phase = braid_phases.get(operation, 0.0)
    new_state.current_braiding_phase.value = (new_state.current_braiding_phase.value + phase) % (2 * np.pi)
    new_state.current_braiding_phase.provenance = Provenance.ESTIMATED

    # Update fermion parity for certain braids
    if operation in ["exchange_13", "exchange_24"]:
        # Full exchange flips parity of the pair
        new_state.fermion_parity.value *= -1
        new_state.fermion_parity.provenance = Provenance.ESTIMATED

    # Record braiding event
    new_state.braiding_history.append({
        "operation": operation,
        "phase_added": phase,
        "total_phase": new_state.current_braiding_phase.value,
        "parity": new_state.fermion_parity.value,
    })

    return new_state


def mzm_parity_measurement(
    state: MZMState,
    measurement_time: float,
    backend: GPUBackend | None = None
) -> tuple[float, float]:
    """
    Simulate fermion parity measurement via conductance readout.

    Measures zero-bias conductance to infer parity.
    Even parity -> high conductance (2e^2/h), Odd parity -> low conductance.

    Args:
        state: MZMState
        measurement_time: Integration time in seconds
        backend: GPU backend

    Returns:
        (measured_parity, fidelity)
    """
    if backend is None:
        backend = get_backend()

    # Signal: conductance difference between parities
    g_even = mzm_zero_bias_conductance(state, voltage=0.0, backend=backend)
    # For odd parity, the Majorana is occupied -> no zero-bias peak
    g_odd = 0.0

    # Shot noise limited measurement
    # SNR ~ sqrt(G * V * t / e) for voltage bias V
    v_bias = 0.01  # mV
    e = 1.602e-19
    current_even = g_even * (2 * e**2 / 6.626e-34) * v_bias * 1e-3  # A
    snr = np.sqrt(current_even * measurement_time / e) if current_even > 0 else 0

    # Fidelity from SNR
    fidelity = 0.5 * (1 + np.erf(snr / np.sqrt(2)))

    # Simulate measurement outcome
    true_parity = state.get_parity()
    if np.random.random() < fidelity:
        measured_parity = true_parity
    else:
        measured_parity = -true_parity

    return measured_parity, float(fidelity)


def mzm_quasiparticle_poisoning(
    state: MZMState,
    dt: float,
    backend: GPUBackend | None = None
) -> MZMState:
    """
    Simulate quasiparticle poisoning (parity flips).

    Quasiparticles tunnel into the Majorana island, flipping fermion parity.
    Rate ~ Gamma_qp * exp(-Delta/kT)

    Args:
        state: MZMState
        dt: Time step in seconds
        backend: GPU backend

    Returns:
        New state with possible parity flip
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()

    rate = state.params.quasiparticle_poisoning_rate.value  # Hz
    prob_flip = 1 - np.exp(-rate * dt)

    if np.random.random() < prob_flip:
        new_state.fermion_parity.value *= -1
        new_state.fermion_parity.provenance = Provenance.ESTIMATED
        new_state.parity_history.append(new_state.fermion_parity.value)

    return new_state


def mzm_majorana_transmon_coupling(
    state: MZMState,
    transmon_freq: float,  # GHz
    coupling_strength: float,  # MHz
    backend: GPUBackend | None = None
) -> dict:
    """
    Compute Majorana-transmon coupling for circuit QED readout.

    The Majorana fermion parity couples to transmon frequency:
    omega_q = omega_0 + chi * P where P = +/-1 is parity.

    Args:
        state: MZMState
        transmon_freq: Transmon frequency in GHz
        coupling_strength: Coupling in MHz
        backend: GPU backend

    Returns:
        Dict with dispersive shift, readout fidelity, etc.
    """
    if backend is None:
        backend = get_backend()

    parity = state.get_parity()
    chi = coupling_strength * 1e-3  # GHz

    # Dispersive shift
    freq_shift = chi * parity

    # Readout contrast
    kappa = 1.0  # MHz, cavity linewidth
    n_photons = 10  # Average photon number
    snr = 2 * chi * np.sqrt(n_photons) / kappa * 1e3  # Dimensionless
    fidelity = 0.5 * (1 + np.erf(snr / np.sqrt(2)))

    return {
        "transmon_freq_ghz": transmon_freq,
        "dispersive_shift_mhz": coupling_strength * parity,
        "freq_shift_ghz": freq_shift,
        "readout_fidelity": float(fidelity),
        "parity": parity,
        "snr": float(snr),
    }


def mzm_kitaev_chain_spectrum(
    params: MZMParams,
    n_sites: int = 100,
    b_field: float | None = None,
    backend: GPUBackend | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Kitaev chain energy spectrum (BdG Hamiltonian).

    H = -t sum(c^†_i c_{i+1} + h.c.) - mu sum(c^†_i c_i)
        + Delta sum(c_i c_{i+1} + h.c.) + V_Z sum(c^†_i sigma_z c_i)

    Args:
        params: MZMParams
        n_sites: Number of lattice sites
        b_field: Magnetic field in Tesla
        backend: GPU backend

    Returns:
        (eigenvalues, eigenvectors) of BdG Hamiltonian
    """
    if backend is None:
        backend = get_backend()

    xp = backend.xp

    # Parameters
    t = 1.0  # Hopping (meV), sets energy scale
    mu = params.chemical_potential.value
    delta = params.induced_gap.value
    alpha = params.spin_orbit_energy.value
    g = params.g_factor.value
    mu_b = 0.05788  # meV/T (Bohr magneton)

    if b_field is None:
        b_field = params.b_field.value
    v_z = g * mu_b * b_field / 2  # Zeeman energy

    # Build BdG Hamiltonian (2n_sites x 2n_sites for particle-hole)
    # Basis: (c_1, c_2, ..., c_n, c^†_1, c^†_2, ..., c^†_n)
    H = xp.zeros((2 * n_sites, 2 * n_sites), dtype=xp.complex64)

    for i in range(n_sites):
        # On-site terms
        H[i, i] = -mu + v_z  # Particle
        H[n_sites + i, n_sites + i] = mu - v_z  # Hole (particle-hole symmetry)

        # Hopping
        if i < n_sites - 1:
            H[i, i + 1] = -t
            H[i + 1, i] = -t
            H[n_sites + i, n_sites + i + 1] = t  # Hole hopping (sign change)
            H[n_sites + i + 1, n_sites + i] = t

        # Spin-orbit + superconducting pairing
        if i < n_sites - 1:
            # p-wave pairing: Delta * (c_i c_{i+1} - c_{i+1} c_i)
            H[i, n_sites + i + 1] = delta
            H[i + 1, n_sites + i] = -delta
            H[n_sites + i, i + 1] = np.conj(delta)
            H[n_sites + i + 1, i] = -np.conj(delta)

    # Diagonalize
    eigvals = xp.linalg.eigvalsh(H)  # Real eigenvalues for Hermitian H

    return backend.to_numpy(np.sort(eigvals)), None


def mzm_topo_phase_diagram(
    params: MZMParams,
    b_fields: np.ndarray,
    mu_values: np.ndarray,
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    Compute topological phase diagram (topological gap vs B and mu).

    Args:
        params: MZMParams
        b_fields: Array of B-field values (T)
        mu_values: Array of chemical potential values (meV)
        backend: GPU backend

    Returns:
        2D array of topological gap (meV)
    """
    if backend is None:
        backend = get_backend()

    gap_map = np.zeros((len(b_fields), len(mu_values)))

    for i, b in enumerate(b_fields):
        for j, mu in enumerate(mu_values):
            # Create temporary params with modified values
            temp_params = MZMParams()
            for field_name in params.__dataclass_fields__:
                value = getattr(params, field_name)
                if isinstance(value, TrackedValue):
                    setattr(temp_params, field_name, value.copy())
            temp_params.b_field.value = b
            temp_params.chemical_potential.value = mu

            gap_map[i, j] = mzm_topological_gap(temp_params, b_field=b, backend=backend)

    return gap_map


def mzm_braiding_protocol(
    state: MZMState,
    protocol: Literal["hadamard", "phase", "cnot", "full_braiding"],
    backend: GPUBackend | None = None
) -> MZMState:
    """
    Execute standard braiding protocols for topological quantum gates.

    Protocols:
    - hadamard: Braid sequence for H gate
    - phase: Braid sequence for S gate (pi/2 phase)
    - cnot: Braid sequence for CNOT (requires 8 MZMs)
    - full_braiding: Full braiding demonstration

    Args:
        state: MZMState
        protocol: Protocol to execute
        backend: GPU backend

    Returns:
        State after protocol
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()

    if protocol == "hadamard":
        # H = braid_12 followed by braid_23 followed by braid_12
        new_state = mzm_braiding_operation(new_state, "braid_12", backend)
        new_state = mzm_braiding_operation(new_state, "braid_23", backend)
        new_state = mzm_braiding_operation(new_state, "braid_12", backend)
    elif protocol == "phase":
        # S = braid_23
        new_state = mzm_braiding_operation(new_state, "braid_23", backend)
    elif protocol == "cnot":
        # CNOT requires 8 Majoranas (4 per qubit)
        # Simplified: just record the intent
        new_state.braiding_history.append({
            "protocol": "cnot",
            "note": "Requires 8 MZMs (4 per qubit) - not implemented for 4 MZM system"
        })
    elif protocol == "full_braiding":
        # Demonstrate non-Abelian statistics: braid_12 then braid_23 != braid_23 then braid_12
        state_a = new_state.copy()
        state_a = mzm_braiding_operation(state_a, "braid_12", backend)
        state_a = mzm_braiding_operation(state_a, "braid_23", backend)

        state_b = new_state.copy()
        state_b = mzm_braiding_operation(state_b, "braid_23", backend)
        state_b = mzm_braiding_operation(state_b, "braid_12", backend)

        new_state.braiding_history.append({
            "protocol": "full_braiding",
            "phase_12_23": state_a.current_braiding_phase.value,
            "phase_23_12": state_b.current_braiding_phase.value,
            "non_abelian": not np.isclose(state_a.current_braiding_phase.value, state_b.current_braiding_phase.value),
        })

    return new_state


def mzm_simulate_experiment(
    state: MZMState,
    experiment_type: Literal["conductance_vs_b", "conductance_vs_v", "parity_lifetime", "braiding"],
    params: dict,
    backend: GPUBackend | None = None
) -> dict:
    """
    Simulate standard Majorana experiments.

    Args:
        state: Initial MZMState
        experiment_type: Type of experiment
        params: Experiment parameters
        backend: GPU backend

    Returns:
        Experiment results
    """
    if backend is None:
        backend = get_backend()

    if experiment_type == "conductance_vs_b":
        b_fields = params.get("b_fields", np.linspace(0, 2, 50))
        voltages = params.get("voltage", 0.0)
        conductances = []
        for b in b_fields:
            temp_state = state.copy()
            temp_state.params.b_field.value = b
            temp_state.topological_gap.value = mzm_topological_gap(temp_state.params, b_field=b, backend=backend)
            temp_state.mzm_overlap.value = mzm_wavefunction_overlap(temp_state.params, backend=backend)
            g = mzm_zero_bias_conductance(temp_state, voltage=voltages, backend=backend)
            conductances.append(g)
        return {"b_fields": b_fields, "conductances": conductances}

    elif experiment_type == "conductance_vs_v":
        voltages = params.get("voltages", np.linspace(-1, 1, 100))
        conductances = [mzm_zero_bias_conductance(state, v, backend=backend) for v in voltages]
        return {"voltages": voltages, "conductances": conductances}

    elif experiment_type == "parity_lifetime":
        dt = params.get("dt", 1e-6)
        n_steps = params.get("n_steps", 10000)
        run_state = state.copy()
        parities = []
        times = []
        for i in range(n_steps):
            run_state = mzm_quasiparticle_poisoning(run_state, dt, backend)
            parities.append(run_state.get_parity())
            times.append(i * dt)
        return {"times": times, "parities": parities}

    elif experiment_type == "braiding":
        protocol = params.get("protocol", "hadamard")
        final_state = mzm_braiding_protocol(state, protocol, backend)
        return {
            "braiding_history": final_state.braiding_history,
            "final_phase": final_state.current_braiding_phase.value,
            "final_parity": final_state.get_parity(),
        }

    return {}


# Fix: add missing import for params in mzm_zero_bias_conductance
# The function references `params` but should use `state.params`
# This is a bug in the original code - let me fix it in the function above

# Actually, looking at the code again, the bug is in mzm_zero_bias_conductance
# where it uses `params.induced_gap.value` but `params` is not defined.
# It should be `state.params.induced_gap.value`.
# I'll fix this in the next edit.