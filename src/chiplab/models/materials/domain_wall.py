"""
Magnetic Domain Wall (DW) Motion for Memory and Logic
=======================================================

Implements domain wall dynamics in ferromagnetic nanowires for:
- Racetrack memory (shift register)
- Domain wall logic gates
- Neuromorphic synapses (DW position as weight)
- Skyrmion-based devices (related)

References:
- Parkin et al. (2008) - Magnetic racetrack memory
- Hayashi et al. (2008) - Current-driven DW motion
- Allwood et al. (2005) - DW logic
- Fert et al. (2017) - Skyrmions for memory
- Iwasaki et al. (2014) - Current-driven skyrmion motion
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np

from chiplab.schemas.common import Provenance, TrackedValue
from chiplab.core.gpu import GPUBackend, get_gpu_backend as get_backend


class DWMaterial(Enum):
    """Ferromagnetic materials for DW devices."""
    PERMALLOY = "Ni80Fe20"      # Py - low damping, soft
    COFE = "CoFe"               # CoFe - high polarization
    COFEB = "CoFeB"             # CoFeB - PMA, CMOS compatible
    CO_NI = "Co/Ni"             # Multilayer - PMA, low current
    PT_CO = "Pt/Co"             # Heavy metal/ferromagnet - SOT
    TA_CO = "Ta/CoFeB"          # Ta/CoFeB - SOT
    HEUSLER = "Co2MnSi"         # Heusler - high spin polarization
    CUSTOM = "custom"


class DWType(Enum):
    """Domain wall type."""
    TRANSVERSE = "transverse"       # In-plane magnetization
    VORTEX = "vortex"               # Vortex wall
    NEEL = "neel"                   # Neel wall (PMA + DMI)
    BLOCH = "bloch"                 # Bloch wall
    SKYRMION = "skyrmion"           # Magnetic skyrmion


class DrivingMechanism(Enum):
    """Domain wall driving mechanism."""
    STT = "spin_transfer_torque"      # Current through wire (STT)
    SOT = "spin_orbit_torque"         # Spin Hall effect (heavy metal)
    FIELD = "magnetic_field"          # External field
    VCMA = "voltage_controlled"       # Voltage-controlled magnetic anisotropy
    THERMAL = "thermal_gradient"      # Spin Seebeck effect


@dataclass
class DWParams:
    """Domain wall device parameters."""
    # Material
    material: DWMaterial = DWMaterial.COFEB
    dw_type: DWType = DWType.NEEL
    driving: DrivingMechanism = DrivingMechanism.SOT
    
    # Geometry
    wire_width: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=50e-9,
        provenance=Provenance.MEASURED,
        metadata={"description": "Nanowire width", "unit": "m"}
    ))
    wire_thickness: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e-9,
        provenance=Provenance.MEASURED,
        metadata={"description": "Ferromagnetic layer thickness", "unit": "m"}
    ))
    wire_length: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=10e-6,
        provenance=Provenance.MEASURED,
        metadata={"description": "Nanowire length", "unit": "m"}
    ))
    
    # Magnetic properties
    ms: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.2e6,
        provenance=Provenance.MEASURED,
        metadata={"description": "Saturation magnetization", "unit": "A/m"}
    ))
    ku: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.8e6,
        provenance=Provenance.MEASURED,
        metadata={"description": "Perpendicular magnetic anisotropy", "unit": "J/m^3"}
    ))
    aex: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=15e-12,
        provenance=Provenance.MEASURED,
        metadata={"description": "Exchange stiffness", "unit": "J/m"}
    ))
    alpha: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.02,
        provenance=Provenance.MEASURED,
        metadata={"description": "Gilbert damping"}
    ))
    dmi: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.5e-3,
        provenance=Provenance.MEASURED,
        metadata={"description": "Dzyaloshinskii-Moriya interaction", "unit": "J/m^2"}
    ))
    
    # Spin transport
    spin_polarization: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.6,
        provenance=Provenance.MEASURED,
        metadata={"description": "Spin polarization (STT)"}
    ))
    spin_hall_angle: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.15,
        provenance=Provenance.MEASURED,
        metadata={"description": "Spin Hall angle (SOT)"}
    ))
    resistivity: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=200e-8,
        provenance=Provenance.MEASURED,
        metadata={"description": "Resistivity", "unit": "ohm*m"}
    ))
    spin_diffusion_length: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=5e-9,
        provenance=Provenance.MEASURED,
        metadata={"description": "Spin diffusion length", "unit": "m"}
    ))
    
    # Non-adiabatic torque
    beta: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.04,
        provenance=Provenance.MEASURED,
        metadata={"description": "Non-adiabatic STT parameter"}
    ))
    xi: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.02,
        provenance=Provenance.MEASURED,
        metadata={"description": "Field-like SOT parameter"}
    ))
    
    # Pinning
    pinning_sites: int = 10
    pinning_strength: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e-20,   # ~2.4 kT at 300 K; was 1e-12 J (~2e8 kT) mislabelled as measured
        provenance=Provenance.ASSUMED,
        metadata={"description": "Pinning potential depth U0", "unit": "J"}
    ))
    pinning_spacing: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=100e-9,
        provenance=Provenance.MEASURED,
        metadata={"description": "Pinning site spacing", "unit": "m"}
    ))
    
    # Temperature
    temperature: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=300.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Temperature", "unit": "K"}
    ))
    
    # Derived properties (computed)
    dw_width: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=10e-9,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Domain wall width (Δ)", "unit": "m"}
    ))
    walker_field: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=10e-3,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Walker breakdown field", "unit": "T"}
    ))
    walker_current: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e6,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Walker breakdown current density", "unit": "A/m^2"}
    ))


@dataclass
class DWState:
    """
    Domain wall state for single or multiple DWs.
    
    Tracks position, velocity, internal structure (tilt angle).
    """
    # Parameters
    params: DWParams = field(default_factory=DWParams)
    
    # Number of domain walls
    n_dws: int = 1
    
    # DW positions along wire [0, wire_length]
    positions: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=np.array([100e-9]),
        provenance=Provenance.ASSUMED,
        metadata={"description": "DW positions", "unit": "m"}
    ))
    
    # DW velocities
    velocities: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=np.array([0.0]),
        provenance=Provenance.ASSUMED,
        metadata={"description": "DW velocities", "unit": "m/s"}
    ))
    
    # DW internal angles (tilt for transverse, chirality for Neel)
    angles: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=np.array([0.0]),
        provenance=Provenance.ASSUMED,
        metadata={"description": "DW internal angles", "unit": "rad"}
    ))
    
    # DW types (for mixed configurations)
    dw_types: list[DWType] = field(default_factory=lambda: [DWType.NEEL])
    
    # Current/field inputs
    current_density: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Current density", "unit": "A/m^2"}
    ))
    applied_field: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Applied magnetic field", "unit": "T"}
    ))
    vcma_voltage: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "VCMA gate voltage", "unit": "V"}
    ))
    
    # Pinning sites positions
    pinning_positions: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=np.array([]),
        provenance=Provenance.ASSUMED,
        metadata={"description": "Pinning site positions", "unit": "m"}
    ))
    
    # History
    position_history: list[np.ndarray] = field(default_factory=list)
    velocity_history: list[np.ndarray] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize arrays with correct dimensions based on n_dws."""
        self.positions = TrackedValue(
            value=np.full(self.n_dws, 100e-9),
            provenance=Provenance.ASSUMED,
            metadata={"description": "DW positions", "unit": "m"}
        )
        self.velocities = TrackedValue(
            value=np.zeros(self.n_dws),
            provenance=Provenance.ASSUMED,
            metadata={"description": "DW velocities", "unit": "m/s"}
        )
        self.angles = TrackedValue(
            value=np.zeros(self.n_dws),
            provenance=Provenance.ASSUMED,
            metadata={"description": "DW internal angles", "unit": "rad"}
        )
        self.dw_types = [DWType.NEEL] * self.n_dws
        self.inputs = TrackedValue(
            value=np.zeros(self.n_dws),
            provenance=Provenance.ASSUMED,
            metadata={"description": "External inputs"}
        )
    
    def copy(self) -> DWState:
        """Create a deep copy."""
        return copy.deepcopy(self)
    
    def initialize_pinning(self, backend: GPUBackend | None = None) -> DWState:
        """Initialize random pinning sites."""
        new_state = self.copy()
        n_pins = self.params.pinning_sites
        spacing = self.params.pinning_spacing.value
        length = self.params.wire_length.value
        
        # Regular spacing with random offset
        positions = np.linspace(spacing, length - spacing, n_pins)
        positions += np.random.uniform(-spacing/4, spacing/4, n_pins)
        positions = np.clip(positions, 0, length)
        
        new_state.pinning_positions.value = positions
        new_state.pinning_positions.provenance = Provenance.ASSUMED
        return new_state


def dw_compute_derived_params(state: DWState) -> DWState:
    """Compute derived parameters (DW width, Walker breakdown)."""
    new_state = state.copy()
    p = new_state.params
    
    # DW width: Δ = sqrt(A_ex / K_eff)
    # K_eff = K_u - μ0 M_s^2 / 2 (for PMA)
    mu0 = 4 * np.pi * 1e-7
    ms = p.ms.value
    ku = p.ku.value
    aex = p.aex.value
    
    k_eff = ku - mu0 * ms**2 / 2
    if k_eff > 0:
        delta = np.sqrt(aex / k_eff)
    else:
        delta = np.sqrt(aex / ku)  # Approximate
    
    new_state.params.dw_width.value = delta
    new_state.params.dw_width.provenance = Provenance.ESTIMATED
    
    # Walker breakdown is set by the wall's hard-axis (Neel-Bloch) anisotropy, not K_eff, which is
    # negative for in-plane films and used to give a negative threshold (NaN velocities).
    # Thin-film wall demag: H_K_perp = N_x * Ms with N_x ~ t / (t + Delta) (Mougin et al., EPL 2007).
    t = p.wire_thickness.value
    h_k_perp = t / (t + delta) * ms
    h_w = p.alpha.value * h_k_perp / 2
    new_state.params.walker_field.value = h_w
    new_state.params.walker_field.provenance = Provenance.ESTIMATED

    # STT Walker threshold (Thiaville et al., EPL 2005): u_W = gamma0 Delta H_K_perp / 2 * alpha/|beta-alpha|,
    # converted to current density through u = P hbar J / (2 e Ms).
    hbar = 1.055e-34
    e = 1.602e-19
    gamma0 = 2.211e5
    P = p.spin_polarization.value
    nonadiabatic = abs(p.beta.value - p.alpha.value)
    u_w = gamma0 * delta * h_k_perp / 2 * (p.alpha.value / nonadiabatic if nonadiabatic else np.inf)
    j_w = u_w * 2 * e * ms / (2.0 * 9.274e-24 * P) if P > 0 else np.inf   # invert u(J)
    new_state.params.walker_current.value = j_w
    new_state.params.walker_current.provenance = Provenance.ESTIMATED
    new_state._derived = True

    return new_state


def dw_stt_velocity(
    state: DWState,
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    STT-driven DW velocity (1D model).
    
    v = (β/α) * u  (below Walker)
    u = g μB P J / (2 e M_s)  (spin drift velocity)
    """
    if backend is None:
        backend = get_backend()
    
    p = state.params
    J = state.current_density.value
    
    # Spin drift velocity
    hbar = 1.055e-34
    e = 1.602e-19
    ms = p.ms.value
    P = p.spin_polarization.value
    
    # u = g muB P J / (2 e Ms); the old P*hbar*J/(2 e Ms) was off by e/m_e (~1.8e11)
    mu_b = 9.274e-24
    u = (2.0 * mu_b * P / (2 * e * ms)) * J
    
    # Velocity (steady state below Walker)
    v = (p.beta.value / p.alpha.value) * u
    
    # Walker breakdown check
    j_w = p.walker_current.value
    if abs(J) > j_w:
        # Above Walker: precessional motion, reduced average velocity
        v = v * (j_w / abs(J))**0.5  # Approximate
    
    return np.full(state.n_dws, v)


def dw_sot_velocity(
    state: DWState,
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    SOT-driven DW velocity (heavy metal/ferromagnet bilayer).
    
    v = (θ_SH * ħ / (2e * M_s * t_FM)) * J_HM * (1/α) * (1 - ξ/β)
    """
    if backend is None:
        backend = get_backend()
    
    p = state.params
    J = state.current_density.value  # Current in heavy metal
    
    hbar = 1.055e-34
    e = 1.602e-19
    ms = p.ms.value
    t_fm = p.wire_thickness.value
    theta_sh = p.spin_hall_angle.value
    alpha = p.alpha.value
    xi = p.xi.value
    beta = p.beta.value
    
    # SOT efficiency
    tau_dl = theta_sh * hbar / (2 * e * ms * t_fm)  # Damping-like torque
    tau_fl = xi * tau_dl  # Field-like torque
    
    # Velocity
    v = tau_dl * J * (beta - xi) / (alpha * (1 + (xi/beta)**2))  # Simplified
    
    # More accurate: v = (γ Δ / α) * (H_DL - H_FL * α/β)
    # where H_DL = τ_DL * J / (γ M_s), H_FL = τ_FL * J / (γ M_s)
    gamma = 2.21e5  # m/(A*s) - gyromagnetic ratio
    delta = p.dw_width.value
    
    h_dl = tau_dl * J / (gamma * ms)
    h_fl = tau_fl * J / (gamma * ms)
    
    v = (gamma * delta / alpha) * (h_dl - h_fl * alpha / beta)
    
    return np.full(state.n_dws, v)


def dw_field_velocity(
    state: DWState,
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    Field-driven DW velocity.
    
    v = (γ Δ / α) * H  (below Walker)
    """
    if backend is None:
        backend = get_backend()
    
    p = state.params
    H = state.applied_field.value
    
    gamma = 2.21e5
    delta = p.dw_width.value
    alpha = p.alpha.value
    
    v = (gamma * delta / alpha) * H
    
    # Walker breakdown
    h_w = p.walker_field.value
    if abs(H) > h_w:
        v = v * (h_w / abs(H))**0.5
    
    return np.full(state.n_dws, v)


def dw_vcma_velocity(
    state: DWState,
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    VCMA-driven DW motion (voltage-controlled magnetic anisotropy).
    
    Modulates K_u with voltage, creating effective field gradient.
    """
    if backend is None:
        backend = get_backend()
    
    p = state.params
    V = state.vcma_voltage.value
    
    # VCMA coefficient (typical ~ 50 fJ/Vm)
    vcma_coeff = 50e-15  # J/Vm
    
    # Anisotropy change
    delta_ku = vcma_coeff * V / p.wire_thickness.value
    
    # Effective field from anisotropy gradient
    # H_vcma = 2 * delta_Ku / (μ0 * M_s)
    mu0 = 4 * np.pi * 1e-7
    ms = p.ms.value
    h_vcma = 2 * delta_ku / (mu0 * ms)
    
    # Velocity
    gamma = 2.21e5
    delta = p.dw_width.value
    alpha = p.alpha.value
    
    v = (gamma * delta / alpha) * h_vcma
    
    return np.full(state.n_dws, v)


def dw_pinning_force(
    state: DWState,
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    Compute pinning force on each DW.
    
    Models pinning as parabolic potential wells.
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    positions = backend.to_device(state.positions.value)
    pinning_pos = backend.to_device(state.pinning_positions.value)
    pinning_strength = state.params.pinning_strength.value
    delta = state.params.dw_width.value
    
    n_dws = state.n_dws
    n_pins = len(pinning_pos)
    
    forces = xp.zeros(n_dws, dtype=xp.float32)
    
    for i in range(n_dws):
        for j in range(n_pins):
            dx = positions[i] - pinning_pos[j]
            # Parabolic well: F = -dU/dx = -2*k*x for |x| < delta
            # where k = U_0 / delta^2
            if abs(dx) < delta:
                k = pinning_strength / delta**2
                force = -2 * k * dx
                forces[i] += force
    
    return backend.to_numpy(forces)


def dw_friction(state: DWState) -> float:
    """1D collective-coordinate wall friction eta = 2 alpha Ms S / (gamma Delta) (N*s/m).

    Thiele damping for a wall of cross-section S and width Delta; gamma in rad/(s*T).
    Used for both force->velocity (v = F/eta) and the Einstein relation D = kB*T/eta.
    """
    p = state.params
    gamma = 1.760859e11
    s = p.wire_width.value * p.wire_thickness.value
    return 2 * p.alpha.value * p.ms.value * s / (gamma * p.dw_width.value)


def dw_thermal_noise(
    state: DWState,
    dt: float,
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    Thermal noise on DW position (Brownian motion).
    
    <x^2> = 2 * D * dt, with D = k_B * T / eta (Einstein relation, eta from dw_friction)
    """
    if backend is None:
        backend = get_backend()
    
    k_b = 1.380649e-23
    D = k_b * state.params.temperature.value / dw_friction(state)

    # Position noise
    sigma = np.sqrt(2 * D * dt)
    noise = backend.random_normal((state.n_dws,), std=sigma)
    
    return noise


def dw_step(
    state: DWState,
    dt: float,
    backend: GPUBackend | None = None
) -> DWState:
    """
    Single time step for DW dynamics.
    
    Combines driving mechanism, pinning, and thermal noise.
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    new_state = state.copy()
    
    # Ensure derived params computed
    # (the old check on dw_width.provenance never fired: the default is already ESTIMATED, so the
    # Walker current stayed at its 1e6 A/m^2 placeholder)
    if not getattr(new_state, "_derived", False):
        new_state = dw_compute_derived_params(new_state)
    
    # Driving velocity based on mechanism
    driving = new_state.params.driving
    if driving == DrivingMechanism.STT:
        v_drive = dw_stt_velocity(new_state, backend)
    elif driving == DrivingMechanism.SOT:
        v_drive = dw_sot_velocity(new_state, backend)
    elif driving == DrivingMechanism.FIELD:
        v_drive = dw_field_velocity(new_state, backend)
    elif driving == DrivingMechanism.VCMA:
        v_drive = dw_vcma_velocity(new_state, backend)
    else:
        v_drive = np.zeros(new_state.n_dws)
    
    # Pinning force -> velocity reduction
    f_pin = dw_pinning_force(new_state, backend)
    v_pin = f_pin / dw_friction(new_state)
    delta = new_state.params.dw_width.value
    
    # Total velocity
    v_total = v_drive + v_pin
    
    # Thermal noise
    noise = dw_thermal_noise(new_state, dt, backend)
    
    # Update positions
    positions = new_state.positions.value + v_total * dt + noise
    
    # Boundary conditions (reflect at ends)
    length = new_state.params.wire_length.value
    positions = np.clip(positions, 0, length)
    
    # DW-DW repulsion (prevent crossing)
    for i in range(new_state.n_dws - 1):
        if positions[i] > positions[i+1]:
            # Swap or repel
            mid = (positions[i] + positions[i+1]) / 2
            positions[i] = mid - delta
            positions[i+1] = mid + delta
    
    new_state.positions.value = positions
    new_state.positions.provenance = Provenance.ESTIMATED
    new_state.velocities.value = v_total
    new_state.velocities.provenance = Provenance.ESTIMATED
    
    # Update angles (simplified: align with velocity)
    angles = new_state.angles.value
    for i in range(new_state.n_dws):
        if v_total[i] > 0:
            angles[i] = 0  # Right-moving
        elif v_total[i] < 0:
            angles[i] = np.pi  # Left-moving
    new_state.angles.value = angles
    
    # History
    new_state.position_history.append(positions.copy())
    new_state.velocity_history.append(v_total.copy())
    
    return new_state


def dw_run(
    state: DWState,
    n_steps: int,
    dt: float,
    current_profile: np.ndarray | None = None,  # [n_steps] current density
    field_profile: np.ndarray | None = None,
    backend: GPUBackend | None = None
) -> DWState:
    """Run DW simulation for multiple steps."""
    if backend is None:
        backend = get_backend()
    
    run_state = state.copy()
    
    for step in range(n_steps):
        if current_profile is not None:
            run_state.current_density.value = current_profile[step]
        if field_profile is not None:
            run_state.applied_field.value = field_profile[step]
        run_state = dw_step(run_state, dt, backend)
    
    return run_state


def dw_racetrack_shift(
    state: DWState,
    n_shifts: int,
    current_pulse: float = 5e11,
    pulse_width: float = 1e-9,
    dt: float = 1e-12,
    backend: GPUBackend | None = None
) -> DWState:
    """
    Shift domain walls in racetrack memory.
    
    Applies current pulses to move DWs by one pinning site each.
    """
    if backend is None:
        backend = get_backend()
    
    run_state = state.copy()
    spacing = run_state.params.pinning_spacing.value
    
    # Pulse profile
    n_pulse_steps = int(pulse_width / dt)
    n_relax_steps = int(1e-9 / dt)  # 1 ns relaxation
    
    for shift in range(n_shifts):
        # Apply pulse
        pulse_profile = np.full(n_pulse_steps, current_pulse)
        for j in range(n_pulse_steps):
            run_state.current_density.value = pulse_profile[j]
            run_state = dw_step(run_state, dt, backend)
        
        # Relaxation
        for _ in range(n_relax_steps):
            run_state.current_density.value = 0.0
            run_state = dw_step(run_state, dt, backend)
    
    return run_state


def dw_read_resistance(
    state: DWState,
    mtj_readout: bool = True,
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    Read DW positions via resistance measurement.
    
    - MTJ readout: DW at MTJ position changes resistance
    - AMR readout: Anisotropic magnetoresistance along wire
    """
    if backend is None:
        backend = get_backend()
    
    positions = state.positions.value
    length = state.params.wire_length.value
    n_dws = state.n_dws
    
    if mtj_readout:
        # Assume MTJ at fixed position (e.g., wire end)
        mtj_pos = length * 0.9
        resistances = []
        for pos in positions:
            # DW at MTJ: high resistance (antiparallel)
            # DW away: low resistance (parallel)
            dist = abs(pos - mtj_pos)
            dw_width = state.params.dw_width.value
            if dist < dw_width:
                r = 10e3  # High R
            else:
                r = 1e3   # Low R
            resistances.append(r)
        return np.array(resistances)
    else:
        # AMR: resistance proportional to number of DWs in wire
        base_r = state.params.resistivity.value * length / (state.params.wire_width.value * state.params.wire_thickness.value)
        amr_ratio = 0.02  # 2%
        r = base_r * (1 + amr_ratio * n_dws)
        return np.full(n_dws, r)


def dw_logic_gate(
    state: DWState,
    gate_type: str,
    inputs: list[bool],
    backend: GPUBackend | None = None
) -> tuple[bool, DWState]:
    """
    Domain wall logic gates (Allwood et al. 2005).
    
    Gates: NOT, AND, OR, NAND, NOR, XOR
    Uses DW propagation in Y-junctions.
    """
    if backend is None:
        backend = get_backend()
    
    # Simplified: simulate DW motion in logic geometry
    # This is a placeholder for full micromagnetic simulation
    
    run_state = state.copy()
    
    if gate_type == "NOT":
        # Single input, invert
        out = not inputs[0]
    elif gate_type == "AND":
        out = inputs[0] and inputs[1]
    elif gate_type == "OR":
        out = inputs[0] or inputs[1]
    elif gate_type == "NAND":
        out = not (inputs[0] and inputs[1])
    elif gate_type == "NOR":
        out = not (inputs[0] or inputs[1])
    elif gate_type == "XOR":
        out = inputs[0] != inputs[1]
    else:
        raise ValueError(f"Unknown gate: {gate_type}")
    
    # Simulate DW motion for the logic operation
    # (would need full geometry simulation)
    
    return out, run_state


def dw_synaptic_weight(
    state: DWState,
    w_min: float = 0.0,
    w_max: float = 1.0
) -> np.ndarray:
    """
    Map DW positions to synaptic weights.
    
    Weight = position / wire_length (normalized)
    """
    positions = state.positions.value
    length = state.params.wire_length.value
    weights = positions / length
    return np.clip(weights, w_min, w_max)


def dw_stdp_update(
    state: DWState,
    spike_pre: bool,
    spike_post: bool,
    dt: float,
    current_amplitude: float = 3e11,
    pulse_width: float = 1e-9,
    backend: GPUBackend | None = None
) -> DWState:
    """
    STDP update using DW motion.
    
    Pre-before-post: move DW one direction (potentiation)
    Post-before-pre: move DW opposite direction (depression)
    """
    if backend is None:
        backend = get_backend()
    
    new_state = state.copy()
    
    if spike_pre and spike_post:
        # Coincident: strong potentiation
        pulse = current_amplitude
    elif spike_pre:
        # Pre only: depression (move back)
        pulse = -current_amplitude
    elif spike_post:
        # Post only: potentiation (move forward)
        pulse = current_amplitude
    else:
        return new_state
    
    # Apply pulse
    n_steps = int(pulse_width / dt)
    for _ in range(n_steps):
        new_state.current_density.value = pulse
        new_state = dw_step(new_state, dt, backend)
    
    # Relax
    for _ in range(int(1e-9 / dt)):
        new_state.current_density.value = 0.0
        new_state = dw_step(new_state, dt, backend)
    
    return new_state


def skyrmion_dynamics(
    state: DWState,
    dt: float,
    current_density: float,
    backend: GPUBackend | None = None
) -> DWState:
    """
    Skyrmion dynamics (Thiele equation).
    
    G × v + α D v = F_STT + F_SOT + F_pinning + F_thermal
    
    Skyrmion has gyrotropic force (Magnus effect).
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    new_state = state.copy()
    
    # Skyrmion parameters
    # Topological charge Q = -1 (for skyrmion)
    Q = -1
    # Gyrotropic tensor G = 4π Q
    G = 4 * np.pi * Q
    
    # Dissipative tensor D ≈ 4π * (R/Δ) for radius R
    R = 50e-9  # Skyrmion radius
    delta = new_state.params.dw_width.value
    D = 4 * np.pi * R / delta
    
    # Forces
    # STT force: F_STT = (ħ P / (2e)) * J * π R^2
    hbar = 1.055e-34
    e = 1.602e-19
    P = new_state.params.spin_polarization.value
    f_stt = (hbar * P / (2 * e)) * current_density * np.pi * R**2
    
    # SOT force
    theta_sh = new_state.params.spin_hall_angle.value
    t_fm = new_state.params.wire_thickness.value
    ms = new_state.params.ms.value
    f_sot = theta_sh * hbar / (2 * e * ms * t_fm) * current_density * np.pi * R**2
    
    # Pinning force
    f_pin = dw_pinning_force(new_state, backend)
    
    # Thiele equation: G × v + α D v = F_total
    # In 2D: v_x = (α D F_x + G F_y) / (G^2 + (α D)^2)
    #       v_y = (α D F_y - G F_x) / (G^2 + (α D)^2)
    # For 1D wire: only v_x, F_y = 0
    # v_x = α D F_x / (G^2 + (α D)^2)  (with skyrmion Hall effect)
    
    alpha = new_state.params.alpha.value
    denom = G**2 + (alpha * D)**2
    
    v_x = alpha * D * f_stt / denom  # Longitudinal
    v_y = -G * f_stt / denom  # Transverse (skyrmion Hall)
    
    # Update position (1D wire: only x)
    positions = new_state.positions.value + v_x * dt
    new_state.positions.value = np.clip(positions, 0, new_state.params.wire_length.value)
    new_state.velocities.value = np.full(new_state.n_dws, v_x)
    
    new_state.position_history.append(new_state.positions.value.copy())
    new_state.velocity_history.append(new_state.velocities.value.copy())
    
    return new_state


# Export
__all__ = [
    "DWMaterial",
    "DWType",
    "DrivingMechanism",
    "DWParams",
    "DWState",
    "dw_compute_derived_params",
    "dw_stt_velocity",
    "dw_sot_velocity",
    "dw_field_velocity",
    "dw_vcma_velocity",
    "dw_pinning_force",
    "dw_friction",
    "dw_thermal_noise",
    "dw_step",
    "dw_run",
    "dw_racetrack_shift",
    "dw_read_resistance",
    "dw_logic_gate",
    "dw_synaptic_weight",
    "dw_stdp_update",
    "skyrmion_dynamics",
]