"""Multi-port ferrofluid device: conductor ports around one shared fluid region.

Write path: voltage -> conductor current -> Biot-Savart field -> particle magnetization
and redistribution (``ferro_particles``).  Read path: reciprocity gives the flux linked
by every port from that same magnetization and susceptibility:

    Z_ij(w) = R_i d_ij + j w [ L0_ij + mu0 Int h_i . chi*(r, w) . h_j dV ]
    flux_j  = mu0 Int h_j . M dV,        emf_j = -d flux_j / dt

h_i is the field of port i per ampere.  The integral is first order in the
susceptibility (the fluid does not distort the conductor field), so its relative error
is of the order of chi.  See docs/ferrofluid_model.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chiplab.models.materials.ferro_particles import (
    KB,
    MU0,
    FerrofluidParams,
    ParticleEnsemble,
)

COPPER_RESISTIVITY = 1.68e-8  # ohm m at 293 K (published handbook value)
EPS0 = 8.8541878128e-12
ELECTROMIGRATION_J = 1.0e10  # A/m^2, order of magnitude for thin-film copper
ROLES = ("write", "read", "both")

NOT_MODELLED = [
    "energy per bit, bit latency (no bit definition or error criterion exists for this device)",
    "endurance cycles",
    "storage density",
    "thermal sensitivity coefficient",
    "Joule heating and temperature gradients (temperature is a fixed input)",
    "direct electric-field forces on particles (electrophoresis, dielectrophoresis)",
    "magnetic-state dependence of electrical conductivity or permittivity (chain formation)",
    "aggregation, sedimentation, ferrohydrodynamic flow",
    "demagnetizing field of the fluid body and field distortion by the fluid",
    "eddy currents, skin effect, coil parasitic capacitance, ferromagnetic resonance",
    "field dependence of Neel relaxation and blocked-particle switching",
]


def _vec(value: Any, name: str) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 2:
        arr = np.append(arr, 0.0)
    if arr.size != 3 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be 2 or 3 finite numbers")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


@dataclass
class CoilPort:
    """One electrical port: a circular loop (default) or a straight conductor segment."""

    id: str
    role: str = "both"
    conductor: str = "loop"  # "loop" or "line"
    position_um: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius_um: float = 20.0
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    start_um: tuple[float, float, float] | None = None
    end_um: tuple[float, float, float] | None = None
    turns: int = 1
    wire_radius_um: float = 1.0
    series_resistance_ohm: float = 50.0
    electrode_radius_um: float | None = None
    assumed: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"port {self.id}: role must be one of {ROLES}")
        if self.conductor not in ("loop", "line"):
            raise ValueError(f"port {self.id}: conductor must be 'loop' or 'line'")
        self.position_um = _vec(self.position_um, "position_um")
        self.axis = _vec(self.axis, "axis")
        if self.turns < 1 or int(self.turns) != self.turns:
            raise ValueError(f"port {self.id}: turns must be a positive integer")
        if not self.wire_radius_um > 0 or not self.series_resistance_ohm >= 0:
            raise ValueError(f"port {self.id}: wire radius must be > 0, resistance >= 0")
        if self.conductor == "loop":
            if not self.radius_um >= 2 * self.wire_radius_um:
                raise ValueError(f"port {self.id}: loop radius must be at least twice the "
                                 "wire radius")
            if np.linalg.norm(self.axis) == 0:
                raise ValueError(f"port {self.id}: axis must be non-zero")
        else:
            if self.start_um is None or self.end_um is None:
                raise ValueError(f"port {self.id}: a line conductor needs start_um and end_um")
            self.start_um = _vec(self.start_um, "start_um")
            self.end_um = _vec(self.end_um, "end_um")
            if self.length_m <= 2 * self.wire_radius_um * 1e-6:
                raise ValueError(f"port {self.id}: line is shorter than the wire diameter")
            if self.turns != 1:
                raise ValueError(f"port {self.id}: a line conductor has exactly one turn")
        if self.electrode_radius_um is not None and not self.electrode_radius_um > 0:
            raise ValueError(f"port {self.id}: electrode_radius_um must be > 0")

    @property
    def length_m(self) -> float:
        if self.conductor == "loop":
            return 2.0 * np.pi * self.radius_um * 1e-6
        return float(np.linalg.norm(np.subtract(self.end_um, self.start_um))) * 1e-6

    @property
    def centre_um(self) -> np.ndarray:
        if self.conductor == "loop":
            return np.array(self.position_um)
        return (np.array(self.start_um) + np.array(self.end_um)) / 2.0

    def segments(self, n: int = 96) -> tuple[np.ndarray, np.ndarray]:
        """Straight segments (start, end) in metres describing one turn."""
        if self.conductor == "line":
            points = np.linspace(self.start_um, self.end_um, n + 1) * 1e-6
        else:
            axis = np.array(self.axis) / np.linalg.norm(self.axis)
            helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u = np.cross(helper, axis)
            u /= np.linalg.norm(u)
            v = np.cross(axis, u)  # right-handed about axis
            angle = np.linspace(0.0, 2.0 * np.pi, n + 1)
            # polygon with the same enclosed area as the circle
            radius = self.radius_um * np.sqrt((2 * np.pi / n) / np.sin(2 * np.pi / n))
            points = (np.array(self.position_um)
                      + radius * (np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v)) * 1e-6
        return points[:-1], points[1:]

    def resistance_ohm(self) -> float:
        area = np.pi * (self.wire_radius_um * 1e-6) ** 2
        return float(self.series_resistance_ohm
                     + COPPER_RESISTIVITY * self.turns * self.length_m / area)

    def self_inductance_h(self) -> float:
        a = self.wire_radius_um * 1e-6
        if self.conductor == "loop":
            # thin circular loop, uniform current: mu0 R (ln(8R/a) - 7/4), times N^2
            r = self.radius_um * 1e-6
            return float(MU0 * r * (np.log(8.0 * r / a) - 1.75) * self.turns**2)
        # Rosa partial self inductance of a straight round wire
        length = self.length_m
        return float(MU0 * length / (2.0 * np.pi) * (np.log(2.0 * length / a) - 0.75))


def segment_field(points: np.ndarray, start: np.ndarray, end: np.ndarray,
                  chunk: int = 4000) -> np.ndarray:
    """H per ampere at `points` from straight segments (exact Biot-Savart per segment)."""
    out = np.zeros_like(points, dtype=float)
    for first in range(0, len(points), chunk):
        p = points[first:first + chunk]
        r1 = p[:, None, :] - start[None, :, :]
        r2 = p[:, None, :] - end[None, :, :]
        n1 = np.linalg.norm(r1, axis=2)
        n2 = np.linalg.norm(r2, axis=2)
        denom = n1 * n2 * (n1 * n2 + np.sum(r1 * r2, axis=2))
        denom = np.maximum(denom, 1e-40)
        factor = (n1 + n2) / denom
        out[first:first + chunk] = np.sum(np.cross(r1, r2) * factor[..., None], axis=1) / (
            4.0 * np.pi)
    return out


def port_field(port: CoilPort, points: np.ndarray, n_segments: int = 96) -> np.ndarray:
    """H per ampere of drive current (including turns) at the given points [1/m]."""
    start, end = port.segments(n_segments)
    return port.turns * segment_field(points, start, end)


def _distance_to_port(port: CoilPort, points: np.ndarray) -> np.ndarray:
    start, end = port.segments(96)
    best = np.full(len(points), np.inf)
    for a, b in zip(start, end, strict=True):
        ab = b - a
        t = np.clip((points - a) @ ab / (ab @ ab), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(points - a - t[:, None] * ab, axis=1))
    return best


def mutual_inductance_h(a: CoilPort, b: CoilPort, n: int = 192) -> float:
    """Neumann double integral M = mu0/4pi N_a N_b oint oint dl_a . dl_b / r."""
    sa, ea = a.segments(n)
    sb, eb = b.segments(n)
    da, db = ea - sa, eb - sb
    ma, mb = (sa + ea) / 2.0, (sb + eb) / 2.0
    r = np.linalg.norm(ma[:, None, :] - mb[None, :, :], axis=2)
    r = np.maximum(r, (a.wire_radius_um + b.wire_radius_um) * 1e-6)
    return float(MU0 / (4.0 * np.pi) * a.turns * b.turns * np.sum((da @ db.T) / r))


@dataclass
class FluidRegion:
    """Axis-aligned box of ferrofluid, discretised on a regular grid."""

    center_um: tuple[float, float, float]
    size_um: tuple[float, float, float]
    grid: tuple[int, int, int]
    assumed: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.center_um = _vec(self.center_um, "center_um")
        self.size_um = _vec(self.size_um, "size_um")
        if min(self.size_um) <= 0:
            raise ValueError("fluid region size must be positive in every direction")
        grid = tuple(int(g) for g in self.grid)
        if len(grid) != 3 or min(grid) < 1 or int(np.prod(grid)) > 60000:
            raise ValueError("grid must be three positive integers with at most 60000 cells")
        self.grid = (grid[0], grid[1], grid[2])

    def cells(self) -> tuple[np.ndarray, np.ndarray]:
        axes = []
        for c, s, n in zip(self.center_um, self.size_um, self.grid, strict=True):
            axes.append((c - s / 2.0 + (np.arange(n) + 0.5) * s / n) * 1e-6)
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.stack([m.ravel() for m in mesh], axis=1)
        volume = np.prod([s * 1e-6 / n for s, n in zip(self.size_um, self.grid, strict=True)])
        return points, np.full(len(points), float(volume))

    @classmethod
    def around(cls, ports: list[CoilPort], height_um: float = 10.0, standoff_um: float = 1.0
               ) -> FluidRegion:
        """Default film above the conductors (an assumed geometry)."""
        low, high = np.full(3, np.inf), np.full(3, -np.inf)
        for port in ports:
            start, end = port.segments(48)
            pts = np.vstack([start, end]) * 1e6
            low, high = np.minimum(low, pts.min(axis=0)), np.maximum(high, pts.max(axis=0))
        sizes = [p.radius_um if p.conductor == "loop" else p.length_m * 1e6 / 4 for p in ports]
        margin = float(max(sizes))
        wire = max(p.wire_radius_um for p in ports)
        z0 = float(high[2]) + wire + standoff_um
        lx = float(high[0] - low[0]) + 2 * margin
        ly = float(high[1] - low[1]) + 2 * margin
        cell = max(min(sizes) / 4.0, np.sqrt(lx * ly / 2500.0))
        grid = (max(8, int(np.ceil(lx / cell))), max(8, int(np.ceil(ly / cell))), 8)
        center = (float(low[0] + high[0]) / 2, float(low[1] + high[1]) / 2, z0 + height_um / 2)
        return cls(center, (lx, ly, height_um), grid, assumed=[
            f"fluid film {height_um} um thick starting {standoff_um} um above the conductors",
            f"fluid film extends {margin:.1f} um beyond the conductors",
        ])


@dataclass
class DeviceState:
    time_s: float
    magnetization: np.ndarray  # cells x bins x 3 [A/m]
    concentration: np.ndarray  # cells x bins, c/c0 - 1

    def copy(self) -> DeviceState:
        return DeviceState(self.time_s, self.magnetization.copy(), self.concentration.copy())


class FerrofluidDevice:
    """Several conductor ports sharing one ferrofluid region."""

    def __init__(self, ports: list[CoilPort], params: FerrofluidParams | None = None,
                 region: FluidRegion | None = None, concentration_length_um: float | None = None,
                 conductivity_s_per_m: float | None = None,
                 relative_permittivity: float | None = None,
                 double_layer_f_per_m2: float | None = None):
        if not 1 <= len(ports) <= 32:
            raise ValueError("a device needs between 1 and 32 ports")
        ids = [p.id for p in ports]
        if len(set(ids)) != len(ids):
            raise ValueError(f"port ids must be unique, got {ids}")
        self.ports = ports
        self.params = params or FerrofluidParams()
        self.ensemble = ParticleEnsemble(self.params)
        self.region = region or FluidRegion.around(ports)
        points, volume = self.region.cells()
        inside = np.zeros(len(points), dtype=bool)
        for port in ports:
            inside |= _distance_to_port(port, points) < port.wire_radius_um * 1e-6
        if inside.all():
            raise ValueError("the fluid region lies entirely inside a conductor")
        self.excluded_cells = int(inside.sum())
        self.points, self.volume = points[~inside], volume[~inside]
        self.h = np.stack([port_field(p, self.points) for p in ports])  # P x C x 3
        self.resistance = np.array([p.resistance_ohm() for p in ports])
        self.inductance_air = np.diag([p.self_inductance_h() for p in ports])
        self.warnings: list[str] = []
        for i in range(len(ports)):
            for j in range(i + 1, len(ports)):
                gap = np.min(_distance_to_port(ports[i], ports[j].segments(96)[0]))
                if gap < (ports[i].wire_radius_um + ports[j].wire_radius_um) * 1e-6:
                    self.warnings.append(f"conductors of {ids[i]} and {ids[j]} touch or overlap: "
                                         "their coupling is unreliable")
                mutual = mutual_inductance_h(ports[i], ports[j])
                self.inductance_air[i, j] = self.inductance_air[j, i] = mutual
        if np.linalg.eigvalsh(self.inductance_air).min() <= 0:
            raise ValueError("port geometry gives a non-physical (non positive definite) "
                             "inductance matrix; separate the conductors")
        if concentration_length_um is None:
            sizes = [2 * p.radius_um if p.conductor == "loop" else p.length_m * 1e6
                     for p in ports]
            concentration_length_um = float(min(np.mean(sizes), max(self.region.size_um)))
        if not concentration_length_um > 0:
            raise ValueError("concentration_length_um must be positive")
        self.concentration_length_um = float(concentration_length_um)
        mean_d = float(np.sum(self.ensemble.volume_weight * self.ensemble.diffusivity))
        self.tau_concentration = (
            (self.concentration_length_um * 1e-6) ** 2 / (np.pi**2 * mean_d)
            if mean_d > 0 else np.inf)
        for name, value in (("conductivity_s_per_m", conductivity_s_per_m),
                            ("relative_permittivity", relative_permittivity),
                            ("double_layer_f_per_m2", double_layer_f_per_m2)):
            if value is not None and not value > 0:
                raise ValueError(f"{name} must be positive")
        self.conductivity = conductivity_s_per_m
        self.permittivity = relative_permittivity
        self.double_layer = double_layer_f_per_m2

    # -- helpers ---------------------------------------------------------
    def index(self, port: str | int) -> int:
        if isinstance(port, int) and not isinstance(port, bool):
            if 0 <= port < len(self.ports):
                return port
        else:
            for i, p in enumerate(self.ports):
                if p.id == port:
                    return i
        raise ValueError(f"unknown port {port!r}; ports are {[p.id for p in self.ports]}")

    def current(self, voltages: dict[str | int, float] | None = None) -> np.ndarray:
        """Quasi-static drive current I = V / R for every port."""
        out = np.zeros(len(self.ports))
        for port, voltage in (voltages or {}).items():
            i = self.index(port)
            if self.resistance[i] <= 0:
                raise ValueError("a driven port needs a non-zero resistance")
            out[i] = float(voltage) / self.resistance[i]
        return out

    def field(self, currents: np.ndarray) -> np.ndarray:
        """Total H in every fluid cell: vector superposition of all port fields."""
        return np.tensordot(currents, self.h, axes=1)

    def initial_state(self) -> DeviceState:
        shape = (len(self.points), self.ensemble.bins)
        return DeviceState(0.0, np.zeros(shape + (3,)), np.zeros(shape))

    # -- dynamics ----------------------------------------------------------
    def _relaxation_terms(self, state: DeviceState, currents: np.ndarray):
        h = self.field(currents)
        mag = np.linalg.norm(h, axis=1)
        unit = h / np.maximum(mag, 1e-30)[:, None]
        m_eq, xi = self.ensemble.equilibrium(mag, state.concentration)
        tau_par, tau_perp = self.ensemble.relaxation_times(xi)
        target = m_eq[:, :, None] * unit[:, None, :]
        delta = state.magnetization - target
        parallel = np.sum(delta * unit[:, None, :], axis=2)[:, :, None] * unit[:, None, :]
        return target, parallel, delta - parallel, tau_par[:, :, None], tau_perp[:, :, None]

    def evolve(self, state: DeviceState, currents: np.ndarray, duration_s: float
               ) -> DeviceState:
        """Advance the state under constant port currents.

        Magnetization follows the Shliomis relaxation equation toward the local
        equilibrium [S74], solved exactly for a constant field.  Concentration relaxes
        toward the magnetophoretic equilibrium with a single diffusive time constant
        L^2/(pi^2 D) (an order-of-magnitude, single-mode approximation).
        """
        if duration_s < 0 or not np.isfinite(duration_s):
            raise ValueError("duration must be finite and non-negative")
        new = state.copy()
        if duration_s == 0:
            return new
        steps = 1
        c_eq = None
        if np.isfinite(self.tau_concentration):
            mag = np.linalg.norm(self.field(currents), axis=1)
            c_eq = self.ensemble.concentration_equilibrium(mag, self.volume)
            steps = int(min(64, max(1, np.ceil(duration_s / (0.1 * self.tau_concentration)))))
        dt = duration_s / steps
        for _ in range(steps):
            if c_eq is not None:
                keep = np.exp(-dt / self.tau_concentration)
                new.concentration = c_eq + (new.concentration - c_eq) * keep
            target, par, perp, tau_par, tau_perp = self._relaxation_terms(new, currents)
            new.magnetization = target + par * np.exp(-dt / tau_par) + perp * np.exp(
                -dt / tau_perp)
        new.time_s = state.time_s + duration_s
        return new

    # -- observables -------------------------------------------------------
    def flux_linkage(self, state: DeviceState) -> np.ndarray:
        """Flux linked by each port from the fluid magnetization [Wb]."""
        total = state.magnetization.sum(axis=1)
        return MU0 * np.einsum("pcx,cx,c->p", self.h, total, self.volume)

    def emf(self, state: DeviceState, currents: np.ndarray) -> np.ndarray:
        """Voltage induced in each port by the changing magnetization at this instant."""
        _, par, perp, tau_par, tau_perp = self._relaxation_terms(state, currents)
        rate = -(par / tau_par + perp / tau_perp).sum(axis=1)
        return -MU0 * np.einsum("pcx,cx,c->p", self.h, rate, self.volume)

    def fluid_inductance(self, frequency_hz: np.ndarray, currents: np.ndarray,
                         concentration: np.ndarray | None = None) -> np.ndarray:
        """Complex fluid contribution to the inductance matrix, shape (F, P, P)."""
        freqs = np.atleast_1d(np.asarray(frequency_hz, dtype=float))
        if np.any(freqs < 0) or not np.all(np.isfinite(freqs)):
            raise ValueError("frequencies must be finite and non-negative")
        h = self.field(currents)
        mag = np.linalg.norm(h, axis=1)
        unit = h / np.maximum(mag, 1e-30)[:, None]
        par, perp, xi = self.ensemble.static_susceptibility_bins(mag, concentration)
        tau_par, tau_perp = self.ensemble.relaxation_times(xi)
        along = np.einsum("pcx,cx->pc", self.h, unit)
        components = [self.h[:, :, x] for x in range(3)]
        out = np.empty((len(freqs), len(self.ports), len(self.ports)), dtype=complex)
        for k, f in enumerate(freqs):
            w = 2j * np.pi * f
            chi_par = np.sum(par / (1.0 + w * tau_par), axis=1) * self.volume
            chi_perp = np.sum(perp / (1.0 + w * tau_perp), axis=1) * self.volume
            matrix = (along * (chi_par - chi_perp)) @ along.T
            for h_x in components:
                matrix = matrix + (h_x * chi_perp) @ h_x.T
            out[k] = MU0 * (matrix + matrix.T) / 2.0
        return out

    def impedance_matrix(self, frequency_hz: np.ndarray, currents: np.ndarray | None = None,
                         state: DeviceState | None = None) -> np.ndarray:
        """Small-signal port impedance matrix Z(f) [ohm] at a bias and particle state."""
        freqs = np.atleast_1d(np.asarray(frequency_hz, dtype=float))
        bias = np.zeros(len(self.ports)) if currents is None else np.asarray(currents, float)
        fluid = self.fluid_inductance(
            freqs, bias, None if state is None else state.concentration)
        w = 2j * np.pi * freqs[:, None, None]
        return np.diag(self.resistance)[None] + w * (self.inductance_air[None] + fluid)

    def electrical_pair_impedance(self, a: str | int, b: str | int, frequency_hz: np.ndarray,
                                  include_double_layer: bool = True) -> np.ndarray | None:
        """Conductive/dielectric impedance between two pillar electrodes, or None.

        Two cylindrical electrodes spanning the fluid height h with insulating floor and
        lid (2-D current flow): G = pi sigma h / arccosh(s / 2a'), C = G eps / sigma, with
        arccosh generalised to unequal radii.  The magnetic state does not enter.
        """
        pa, pb = self.ports[self.index(a)], self.ports[self.index(b)]
        if (self.conductivity is None or self.permittivity is None
                or pa.electrode_radius_um is None or pb.electrode_radius_um is None):
            return None
        r1, r2 = pa.electrode_radius_um * 1e-6, pb.electrode_radius_um * 1e-6
        s = float(np.linalg.norm((pa.centre_um - pb.centre_um)[:2])) * 1e-6
        if s <= r1 + r2:
            raise ValueError(f"electrodes of {pa.id} and {pb.id} overlap")
        shape = np.arccosh((s**2 - r1**2 - r2**2) / (2 * r1 * r2)) / (2 * np.pi)
        height = self.region.size_um[2] * 1e-6
        w = 2.0 * np.pi * np.atleast_1d(np.asarray(frequency_hz, dtype=float))
        admittance = (self.conductivity + 1j * w * EPS0 * self.permittivity) * height / shape
        z = 1.0 / admittance
        if include_double_layer and self.double_layer is not None:
            if np.any(w <= 0):
                raise ValueError("a double-layer capacitance needs frequencies above 0 Hz")
            for radius in (r1, r2):
                z = z + 1.0 / (1j * w * self.double_layer * 2 * np.pi * radius * height)
        return z

    def grid_convergence(self) -> float:
        """Relative change of the fluid self-inductances on a 1.5x finer grid."""
        grid = [int(np.ceil(1.5 * g)) for g in self.region.grid]
        while np.prod(grid) > 60000:
            grid = [max(1, g - 1) for g in grid]
        fine = FluidRegion(self.region.center_um, self.region.size_um, tuple(grid))
        points, volume = fine.cells()
        keep = np.ones(len(points), dtype=bool)
        for port in self.ports:
            keep &= _distance_to_port(port, points) >= port.wire_radius_um * 1e-6
        coarse_l = np.einsum("pcx,pcx,c->p", self.h, self.h, self.volume)
        fine_h = np.stack([port_field(p, points[keep]) for p in self.ports])
        fine_l = np.einsum("pcx,pcx,c->p", fine_h, fine_h, volume[keep])
        return float(np.max(np.abs(coarse_l - fine_l) / fine_l))


# ---------------------------------------------------------------------------
# Pulse sequences
# ---------------------------------------------------------------------------


@dataclass
class WritePulse:
    port: str | int
    voltage_v: float
    duration_s: float
    start_s: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.voltage_v):
            raise ValueError("write voltage must be finite")
        if not (np.isfinite(self.duration_s) and self.duration_s > 0):
            raise ValueError("write duration must be positive")
        if not (np.isfinite(self.start_s) and self.start_s >= 0):
            raise ValueError("write start time must be non-negative")

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


@dataclass
class ReadEvent:
    port: str | int
    frequency_hz: float = 1e6
    time_s: float | None = None  # absolute; default: delay_s after the last write ends
    delay_s: float = 0.0
    amplitude_v: float | None = None
    bandwidth_hz: float = 1e6

    def __post_init__(self) -> None:
        if not (np.isfinite(self.frequency_hz) and self.frequency_hz > 0):
            raise ValueError("read frequency must be positive")
        if self.time_s is not None and not self.time_s >= 0:
            raise ValueError("read time must be non-negative")
        if not self.delay_s >= 0 or not self.bandwidth_hz > 0:
            raise ValueError("read delay must be >= 0 and bandwidth > 0")
        if self.amplitude_v is not None and not self.amplitude_v > 0:
            raise ValueError("read amplitude must be positive")


def complex_dict(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _run_timeline(device: FerrofluidDevice, writes: list[WritePulse], reads: list[ReadEvent],
                  read_times: list[float], unwritten: dict[float, np.ndarray]
                  ) -> tuple[list[dict[str, Any]], DeviceState]:
    """Run the pulses; `unwritten` caches the at-rest fluid inductance per frequency."""
    boundaries = {0.0}
    for w in writes:
        boundaries.update((w.start_s, w.end_s))
    boundaries.update(read_times)
    rest = device.initial_state()
    state = rest.copy()
    zero = np.zeros(len(device.ports))
    results: list[dict[str, Any] | None] = [None] * len(reads)
    last_end = max((w.end_s for w in writes), default=0.0)
    end_state = state.copy() if last_end == 0.0 else None
    ordered = sorted(boundaries)
    for k, t in enumerate(ordered):
        active = {device.index(w.port): w.voltage_v for w in writes
                  if w.start_s <= t < w.end_s}
        currents = device.current(active)
        now: dict[float, np.ndarray] = {}  # reads at one instant share the fluid state
        flux = emf = None
        if t in read_times:
            flux, emf = device.flux_linkage(state), device.emf(state, currents)
        for n, (read, when) in enumerate(zip(reads, read_times, strict=True)):
            if when != t:
                continue
            j = device.index(read.port)
            f = np.array([read.frequency_hz])
            w = 2j * np.pi * read.frequency_hz
            if read.frequency_hz not in now:
                now[read.frequency_hz] = device.fluid_inductance(
                    f, currents, state.concentration)[0]
            if read.frequency_hz not in unwritten:
                unwritten[read.frequency_hz] = device.fluid_inductance(
                    f, zero, rest.concentration)[0]
            l_now, l_rest = now[read.frequency_hz], unwritten[read.frequency_hz]
            fluid = w * l_now
            z = np.diag(device.resistance) + w * device.inductance_air + fluid
            # taken from the fluid term alone: subtracting two full Z would lose precision
            change = w * (l_now - l_rest)
            signal = change[j, j]
            noise = float(np.sqrt(4 * KB * device.params.temperature_k * z[j, j].real
                                  * read.bandwidth_hz))
            entry: dict[str, Any] = {
                "port": device.ports[j].id,
                "time_s": t,
                "frequency_hz": read.frequency_hz,
                "bias_currents_a": {p.id: float(c) for p, c in
                                    zip(device.ports, currents, strict=True) if c != 0},
                "impedance_ohm": complex_dict(z[j, j]),
                "impedance_magnitude_ohm": float(abs(z[j, j])),
                "fluid_contribution_ohm": complex_dict(fluid[j, j]),
                "signal_vs_unwritten_ohm": complex_dict(signal),
                "transfer_impedance_ohm": {
                    p.id: complex_dict(z[j, i]) for i, p in enumerate(device.ports) if i != j},
                "transfer_signal_vs_unwritten_ohm": {
                    p.id: complex_dict(change[j, i])
                    for i, p in enumerate(device.ports) if i != j},
                "magnetization_flux_linkage_wb": float(flux[j]),
                "magnetization_emf_v": float(emf[j]),
                "thermal_noise_v_rms": noise,
                "noise_bandwidth_hz": read.bandwidth_hz,
            }
            if read.amplitude_v is not None:
                i_read = read.amplitude_v / abs(z[j, j])
                signal_v = float(abs(signal) * i_read)
                entry["read_current_a"] = float(i_read)
                entry["signal_v"] = signal_v
                entry["thermal_noise_limited_snr_db"] = (
                    float(20 * np.log10(signal_v / noise)) if signal_v > 0 and noise > 0 else None)
                h_read = i_read * np.linalg.norm(device.h[j], axis=1).max()
                if h_read * device.ensemble.typical_xi_per_field > 0.3:
                    entry["warning"] = ("read current is large enough to saturate particles: "
                                        "the small-signal impedance is not valid")
            results[n] = entry
        if t == last_end:
            end_state = state.copy()
        if k + 1 < len(ordered):
            state = device.evolve(state, currents, ordered[k + 1] - t)
    assert end_state is not None
    return [r for r in results if r is not None], end_state


def _retention(device: FerrofluidDevice, end_state: DeviceState,
               read_ports: list[int]) -> dict[str, Any]:
    """Zero-field decay after the last write: exact sum of exponentials per size bin."""
    tau = device.ensemble.tau_zero_field
    per_bin = MU0 * np.einsum("pcx,cbx,c->pb", device.h, end_state.magnetization,
                              device.volume, optimize=True)
    times = np.logspace(np.log10(tau.min()) - 2, np.log10(tau.max()) + 2, 600)
    times = times[np.isfinite(times)]
    decay = np.exp(-times[:, None] / tau)
    sample = np.concatenate([[0.0], np.logspace(-9, 3, 13)])
    out: dict[str, Any] = {
        "relaxation_time_range_s": [float(tau.min()), float(tau.max())],
        "per_port": {},
        "definition": "time after which |flux linkage from magnetization| stays below the "
                      "stated fraction of its value when the last write ended; this is a "
                      "physical decay time, not a memory retention specification",
    }
    for j in read_ports:
        start = float(per_bin[j].sum())
        entry: dict[str, Any] = {"flux_at_write_end_wb": start}
        if abs(start) > 0:
            ratio = np.abs(decay @ per_bin[j]) / abs(start)
            for label, level in (("time_to_1_over_e_s", np.exp(-1.0)),
                                 ("time_to_10_percent_s", 0.1)):
                above = np.nonzero(ratio > level)[0]
                if len(above) == 0:
                    entry[label] = float(times[0])
                elif above[-1] + 1 < len(times):
                    entry[label] = float(times[above[-1] + 1])
                else:
                    entry[label] = None
        else:
            entry["time_to_1_over_e_s"] = entry["time_to_10_percent_s"] = None
        entry["decay_samples"] = [
            {"delay_s": float(t), "flux_wb": float(np.exp(-t / tau) @ per_bin[j])}
            for t in sample]
        out["per_port"][device.ports[j].id] = entry
    peak = float(np.max(np.abs(end_state.concentration))) if end_state.concentration.size else 0.0
    out["concentration_memory"] = {
        "peak_relative_concentration_change": peak,
        "diffusive_time_constant_s": (float(device.tau_concentration)
                                      if np.isfinite(device.tau_concentration) else None),
        "note": "single-mode L^2/(pi^2 D) approximation, order of magnitude only",
    }
    return out


def simulate_sequence(device: FerrofluidDevice, writes: list[WritePulse],
                      reads: list[ReadEvent], interaction_analysis: bool = True
                      ) -> dict[str, Any]:
    """Drive any set of write ports and read any set of ports on one shared state."""
    for w in writes:
        i = device.index(w.port)
        if device.ports[i].role == "read":
            raise ValueError(f"port {device.ports[i].id} is read-only and cannot be written")
    for a_i, a in enumerate(writes):
        for b in writes[a_i + 1:]:
            if (device.index(a.port) == device.index(b.port)
                    and a.start_s < b.end_s and b.start_s < a.end_s):
                raise ValueError(f"overlapping write pulses on port {a.port!r}")
    for r in reads:
        i = device.index(r.port)
        if device.ports[i].role == "write":
            raise ValueError(f"port {device.ports[i].id} is write-only and cannot be read")
    last_end = max((w.end_s for w in writes), default=0.0)
    read_times = [r.time_s if r.time_s is not None else last_end + r.delay_s for r in reads]
    unwritten: dict[float, np.ndarray] = {}
    results, end_state = _run_timeline(device, writes, reads, read_times, unwritten)
    warnings = list(device.warnings)
    write_report = []
    max_field = 0.0
    for w in writes:
        i = device.index(w.port)
        port = device.ports[i]
        current = w.voltage_v / device.resistance[i]
        peak = float(abs(current) * np.linalg.norm(device.h[i], axis=1).max())
        density = float(abs(current) / (np.pi * (port.wire_radius_um * 1e-6) ** 2))
        time_constant = device.inductance_air[i, i] / device.resistance[i]
        if density > ELECTROMIGRATION_J:
            warnings.append(f"port {port.id}: current density {density:.1e} A/m^2 exceeds about "
                            f"{ELECTROMIGRATION_J:.0e} A/m^2 (conductor damage not modelled)")
        if w.duration_s < 5 * time_constant:
            warnings.append(f"port {port.id}: pulse shorter than 5 L/R; the instantaneous "
                            "current step assumed here is not valid")
        write_report.append({
            "port": port.id, "voltage_v": w.voltage_v, "start_s": w.start_s,
            "duration_s": w.duration_s, "current_a": float(current),
            "current_density_a_m2": density,
            "peak_field_in_fluid_a_m": peak,
            "peak_langevin_argument": peak * device.ensemble.typical_xi_per_field,
            "drive_energy_j": float(w.voltage_v * current * w.duration_s),
        })
        max_field = max(max_field, peak)
    # worst case field when pulses overlap
    for t in {w.start_s for w in writes}:
        currents = device.current({device.index(w.port): w.voltage_v for w in writes
                                   if w.start_s <= t < w.end_s})
        max_field = max(max_field, float(np.linalg.norm(device.field(currents), axis=1).max()))
    max_freq = max((r.frequency_hz for r in reads), default=0.0)
    warnings += device.ensemble.validity_warnings(max_field, max_freq)
    chi0 = device.ensemble.diagnostics()["initial_susceptibility"]
    if chi0 > 0.3:
        warnings.append(f"initial susceptibility {chi0:.2f} > 0.3: the first-order coupling "
                        "integral neglects field distortion and its error grows with chi")
    local = device.params.volume_fraction * (1.0 + end_state.concentration.max(initial=0.0))
    if local > 0.3:
        warnings.append("local particle concentration exceeds 0.3: steric limits not modelled")

    out: dict[str, Any] = {
        "model": "ferrofluid particle model (physics-based, uncalibrated unless parameters "
                 "are labelled fitted)",
        "writes": write_report,
        "reads": results,
        "retention": _retention(device, end_state,
                                sorted({device.index(r.port) for r in reads})),
    }

    written = sorted({device.index(w.port) for w in writes if w.voltage_v != 0})
    if interaction_analysis and len(written) > 1 and reads:
        alone = [
            _run_timeline(device, [w for w in writes if device.index(w.port) == i],
                          reads, read_times, unwritten)[0] for i in written]
        rows = []
        for n, combined in enumerate(results):
            row: dict[str, Any] = {"port": combined["port"], "time_s": combined["time_s"]}
            for key, label in (("signal_vs_unwritten_ohm", "impedance_signal"),
                               ("magnetization_flux_linkage_wb", "flux")):
                def value(entry, key=key):
                    v = entry[key]
                    return complex(v["real"], v["imag"]) if isinstance(v, dict) else v

                total = value(combined)
                summed = sum(value(single[n]) for single in alone)
                residual = total - summed
                row[label] = {
                    "all_ports_together": complex_dict(total) if isinstance(total, complex)
                    else total,
                    "sum_of_ports_alone": complex_dict(summed) if isinstance(summed, complex)
                    else summed,
                    "nonlinear_residual": complex_dict(residual)
                    if isinstance(residual, complex) else residual,
                    "relative_to_combined": (float(abs(residual) / abs(total))
                                             if abs(total) > 0 else None),
                }
            rows.append(row)
        out["port_interaction"] = {
            "definition": "response with all write ports driven minus the sum of responses "
                          "with each write port driven alone; non-zero values come from "
                          "Langevin saturation, particle mean-field interaction and "
                          "magnetophoresis in the shared fluid",
            "reads": rows,
        }

    out["port_properties"] = [{
        "id": p.id, "role": p.role, "conductor": p.conductor,
        "position_um": [float(v) for v in p.centre_um],
        "radius_um": p.radius_um if p.conductor == "loop" else None,
        "turns": p.turns, "wire_radius_um": p.wire_radius_um,
        "resistance_ohm": float(device.resistance[i]),
        "air_inductance_h": float(device.inductance_air[i, i]),
        "assumed": p.assumed,
    } for i, p in enumerate(device.ports)]
    ids = [p.id for p in device.ports]
    k = device.inductance_air / np.sqrt(np.outer(np.diag(device.inductance_air),
                                                 np.diag(device.inductance_air)))
    out["air_coupling_coefficient"] = {
        ids[i]: {ids[j]: float(k[i, j]) for j in range(len(ids)) if j != i}
        for i in range(len(ids))}
    pairs = []
    if reads:
        freqs = sorted({r.frequency_hz for r in reads})
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                z = device.electrical_pair_impedance(i, j, np.array(freqs))
                if z is not None:
                    pairs.append({"ports": [ids[i], ids[j]], "spectrum": [
                        {"frequency_hz": f, "impedance_ohm": complex_dict(v)}
                        for f, v in zip(freqs, z, strict=True)]})
    out["electrical_electrode_pairs"] = pairs or (
        "not modelled: supply conductivity_s_per_m, relative_permittivity and "
        "electrode_radius_um on at least two ports")
    out["fluid_region"] = {
        "center_um": list(device.region.center_um), "size_um": list(device.region.size_um),
        "grid": list(device.region.grid), "cells_used": len(device.points),
        "cells_inside_conductors": device.excluded_cells,
        "assumed": device.region.assumed,
    }
    out["parameters"] = device.params.describe()
    out["particle_diagnostics"] = device.ensemble.diagnostics()
    out["warnings"] = list(dict.fromkeys(warnings))
    out["not_modelled"] = NOT_MODELLED
    return out
