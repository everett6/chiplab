"""Reduced-order electrical state-space model of an N-port ferrofluid sample.

    S(t + dt) = F(S(t), V(t), T, theta)        Z(t) = G(S(t), omega, V(t), T, theta)

The fluid is one network shared by all ports: a branch from every port to the common
ground and a branch between port pairs.  Every branch has a state-dependent impedance;
writing any port changes the voltages, and therefore the states, of the branches it shares
with other ports.

This is a **phenomenological** model.  Its parameters are circuit and kinetic constants
to be fitted to measurements of a particular sample and fixture; they are not particle
size, concentration or viscosity, there are no defaults for them, and no mechanism
(chaining, electrophoresis, electrochemistry) is claimed.  It is unrelated to the magnetic
coil model in ``ferro_device``.  See docs/ferrofluid_state_model.md.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

MODEL_VERSION = "ferro-statespace-1"
KB_EV = 8.617333262e-5
GROUND = "gnd"
STATES = ("fast", "slow")
MAX_PORTS = 16
MAX_STEPS = 400_000
MAX_EXPONENT = 50.0

# name: (unit, lower, upper, default; None default means required)
_REQUIRED = object()
PARAMETERS: dict[str, tuple[str, float, float, Any]] = {
    "r_inf_ohm": ("ohm", 1e-6, 1e12, _REQUIRED),
    "delta_r_ohm": ("ohm", 0.0, 1e12, _REQUIRED),
    "tau_dielectric_s": ("s", 1e-15, 1e6, _REQUIRED),
    "alpha": ("1", 0.2, 1.0, _REQUIRED),
    "capacitance_f": ("F", 0.0, 1.0, _REQUIRED),
    "bias_linear_per_v": ("1/V", -20.0, 20.0, 0.0),
    "bias_quadratic_per_v2": ("1/V^2", -20.0, 20.0, 0.0),
    "reference_temperature_k": ("K", 1.0, 2000.0, None),
    "resistance_activation_ev": ("eV", -5.0, 5.0, 0.0),
    "fatigue_depth": ("1", 0.0, 1.0, 0.0),
    "fatigue_voltage_v": ("V", 1e-3, 1e4, None),
    "fatigue_time_s": ("s", 1e-9, 1e12, None),
    "fatigue_recovery_s": ("s", 1e-9, 1e12, None),
    "recovery_activation_ev": ("eV", -5.0, 5.0, 0.0),
}
for _state in STATES:
    _need = _REQUIRED if _state == "fast" else None
    PARAMETERS.update({
        f"{_state}_write_voltage_v": ("V", 1e-3, 1e4, _need),
        f"{_state}_write_time_s": ("s", 1e-12, 1e12, _need),
        f"{_state}_saturation_voltage_v": ("V", 1e-3, 1e4, _need),
        # "never" means the state does not relax (nonvolatile): claim it only from data
        f"{_state}_retention_s": ("s", 1e-12, 1e15, _need),
        f"{_state}_gain_r_inf": ("1", -20.0, 20.0, 0.0),
        f"{_state}_gain_delta_r": ("1", -20.0, 20.0, 0.0),
        f"{_state}_gain_capacitance": ("1", -20.0, 20.0, 0.0),
        f"{_state}_write_activation_ev": ("eV", -5.0, 5.0, 0.0),
        f"{_state}_retention_activation_ev": ("eV", -5.0, 5.0, 0.0),
    })
SYMMETRIES = ("odd", "even")  # branch key "target_symmetry"
ACTIVATIONS = [n for n in PARAMETERS if n.endswith("_activation_ev")]
_SLOW_GROUP = [f"slow_{n}" for n in
               ("write_voltage_v", "write_time_s", "saturation_voltage_v", "retention_s")]
_FATIGUE_GROUP = ["fatigue_voltage_v", "fatigue_time_s", "fatigue_recovery_s"]

NOT_MODELLED = [
    "any microscopic mechanism: particle size, concentration, surfactant, viscosity, "
    "chaining, electrophoresis, electrode reactions, gas evolution",
    "non-reciprocal response (Z12 differing from Z21)",
    "distributed (transmission-line) effects, fixture and cable parasitics beyond the "
    "fitted branches",
    "disturbance of the state by the RF read signal; duration of a frequency sweep",
    "driver output impedance, current limits and rise time",
    "self heating (temperature is an input)",
    "irreversible degradation, device-to-device variation",
]


def describe_parameters() -> list[dict[str, Any]]:
    return [{"name": n, "unit": u, "min": lo, "max": hi,
             "required": d is _REQUIRED,
             "default": None if d is _REQUIRED else d}
            for n, (u, lo, hi, d) in PARAMETERS.items()]


def _branch_parameters(name: str, values: dict) -> dict[str, Any]:
    unknown = sorted(set(values) - set(PARAMETERS) - {"target_symmetry"})
    if unknown:
        raise ValueError(f"branch {name}: unknown parameter(s) {unknown}")
    out: dict[str, Any] = {}
    for key, (unit, low, high, default) in PARAMETERS.items():
        if key not in values:
            if default is _REQUIRED:
                raise ValueError(f"branch {name}: parameter {key} [{unit}] is required; "
                                 "this model has no default values")
            out[key] = default
            continue
        value = values[key]
        if key.endswith("_retention_s") and value == "never":
            out[key] = np.inf
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"branch {name}: {key} must be a number")
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"branch {name}: {key}={value} outside [{low}, {high}] {unit}")
        out[key] = float(value)
    for group, label in ((_SLOW_GROUP, "slow state"), (_FATIGUE_GROUP, "fatigue")):
        given = [k for k in group if out[k] is not None]
        if given and len(given) != len(group):
            raise ValueError(f"branch {name}: the {label} needs all of {group}")
    if out["slow_write_voltage_v"] is None and any(
            out[f"slow_gain_{q}"] for q in ("r_inf", "delta_r", "capacitance")):
        raise ValueError(f"branch {name}: slow gains given without slow-state kinetics")
    if out["fatigue_depth"] and out["fatigue_voltage_v"] is None:
        raise ValueError(f"branch {name}: fatigue_depth needs {_FATIGUE_GROUP}")
    if any(out[k] for k in ACTIVATIONS) and out["reference_temperature_k"] is None:
        raise ValueError(f"branch {name}: activation energies need reference_temperature_k")
    symmetry = values.get("target_symmetry", "odd")
    if symmetry not in SYMMETRIES:
        raise ValueError(f"branch {name}: target_symmetry must be one of {SYMMETRIES}")
    out["target_symmetry"] = symmetry
    return out


@dataclass
class State:
    fast: np.ndarray
    slow: np.ndarray
    fatigue: np.ndarray

    def copy(self) -> State:
        return State(self.fast.copy(), self.slow.copy(), self.fatigue.copy())


class StateSpaceDevice:
    """N-port network with shared, voltage-programmable branch states.

    model = {"ports": ["p1", "p2"],
             "ground": {...branch parameters},
             "coupling": {...branch parameters} or None for no port-to-port branches,
             "couplings": [["p1", "p2"], ...]   (default: every pair),
             "branches": {"p1|gnd": {...overrides}, "p1|p2": {...}}}
    """

    KEYS = {"ports", "ground", "coupling", "couplings", "branches", "provenance"}

    def __init__(self, model: dict):
        if not isinstance(model, dict):
            raise ValueError("model must be a dictionary (see describe_parameters)")
        unknown = sorted(set(model) - self.KEYS)
        if unknown:
            raise ValueError(f"unknown model key(s) {unknown}")
        ports = model.get("ports")
        if (not isinstance(ports, list) or not 1 <= len(ports) <= MAX_PORTS
                or len(set(map(str, ports))) != len(ports)):
            raise ValueError(f"ports must be 1..{MAX_PORTS} distinct names")
        self.ports = [str(p) for p in ports]
        if any(p == GROUND or "|" in p or "." in p for p in self.ports):
            raise ValueError("port names cannot be 'gnd' or contain '|' or '.'")
        self.model = copy.deepcopy(model)
        n = len(self.ports)
        ground = model.get("ground")
        if not isinstance(ground, dict):
            raise ValueError("model.ground (parameters of each port-to-ground branch) is "
                             "required")
        coupling = model.get("coupling")
        pairs = model.get("couplings")
        if coupling is None:
            if pairs:
                raise ValueError("couplings listed but model.coupling is missing")
            if n > 1 and "coupling" not in model:
                raise ValueError("give model.coupling parameters, or coupling: null to "
                                 "state that ports are not connected to each other")
            index_pairs: list[tuple[int, int]] = []
        elif pairs is None:
            index_pairs = list(combinations(range(n), 2))
        else:
            index_pairs = []
            for pair in pairs:
                if len(pair) != 2 or pair[0] == pair[1]:
                    raise ValueError("each coupling is a pair of different ports")
                a, b = sorted(self.index(p) for p in pair)
                if (a, b) not in index_pairs:
                    index_pairs.append((a, b))
        ends = [(i, n) for i in range(n)] + index_pairs  # node n is ground
        self.branches = [f"{self.ports[a]}|{GROUND if b == n else self.ports[b]}"
                         for a, b in ends]
        overrides = model.get("branches") or {}
        stray = sorted(set(overrides) - set(self.branches))
        if stray:
            raise ValueError(f"unknown branch(es) {stray}; branches: {self.branches}")
        resolved = []
        for name, (_, b) in zip(self.branches, ends, strict=True):
            base = ground if b == n else coupling
            resolved.append(_branch_parameters(name, {**base, **overrides.get(name, {})}))
        self.shared_branches = [b for b in self.branches if b not in overrides]
        self.a = np.array([e[0] for e in ends])
        self.b = np.array([e[1] for e in ends])
        self.incidence = np.zeros((len(ends), n + 1))
        self.incidence[np.arange(len(ends)), self.a] = 1.0
        self.incidence[np.arange(len(ends)), self.b] = -1.0
        self.even = np.array([r["target_symmetry"] == "even" for r in resolved])
        self.p: dict[str, np.ndarray] = {
            key: np.array([np.nan if r[key] is None else r[key] for r in resolved])
            for key in PARAMETERS}
        self.needs_temperature = any(np.any(self.p[k] != 0) for k in ACTIVATIONS)
        self.has_slow = ~np.isnan(self.p["slow_write_voltage_v"])
        self.has_fatigue = ~np.isnan(self.p["fatigue_voltage_v"])

    # -- basics -----------------------------------------------------------
    def index(self, port: str | int) -> int:
        if isinstance(port, int) and not isinstance(port, bool):
            if 0 <= port < len(self.ports):
                return port
        elif str(port) in self.ports:
            return self.ports.index(str(port))
        raise ValueError(f"unknown port {port!r}; ports: {self.ports}")

    def initial_state(self, saved: dict | None = None) -> State:
        m = len(self.branches)
        if saved is None:
            return State(np.zeros(m), np.zeros(m), np.zeros(m))
        if list(saved.get("branches", [])) != self.branches:
            raise ValueError("initial_state belongs to a different network")
        arrays = [np.asarray(saved[k], dtype=float) for k in ("fast", "slow", "fatigue")]
        if any(a.shape != (m,) or not np.all(np.abs(a) <= 1) for a in arrays):
            raise ValueError("initial_state values must lie in [-1, 1], one per branch")
        return State(*arrays)

    def export_state(self, state: State) -> dict[str, Any]:
        return {"branches": list(self.branches), "fast": state.fast.tolist(),
                "slow": state.slow.tolist(), "fatigue": state.fatigue.tolist()}

    def _arrhenius(self, key: str, temperature_k: float | None) -> np.ndarray:
        """exp(Ea/kB (1/T - 1/Tref)); multiplies a time constant or a resistance."""
        energy = self.p[key]
        out = np.ones_like(energy)
        used = energy != 0
        if used.any():
            if temperature_k is None or not temperature_k > 0:
                raise ValueError("this model depends on temperature: give temperature_k")
            ref = self.p["reference_temperature_k"][used]
            exponent = energy[used] / KB_EV * (1.0 / temperature_k - 1.0 / ref)
            out[used] = np.exp(np.clip(exponent, -MAX_EXPONENT, MAX_EXPONENT))
        return out

    @staticmethod
    def _rate(v: np.ndarray, scale: np.ndarray, time: np.ndarray) -> np.ndarray:
        """Voltage-activated rate (cosh(v/scale) - 1) / time: zero with no voltage."""
        x = np.minimum(np.abs(v) / scale, MAX_EXPONENT)
        return (np.cosh(x) - 1.0) / time

    @staticmethod
    def _approach(value: np.ndarray, drive: np.ndarray, decay: np.ndarray,
                  target: np.ndarray, dt: float) -> np.ndarray:
        """Exact solution of dy/dt = drive (target - y) - decay y."""
        total = drive + decay
        final = np.divide(drive * target, total, out=np.zeros_like(total), where=total > 0)
        moved = final + (value - final) * np.exp(-np.minimum(total * dt, 700.0))
        return np.where(total > 0, moved, value)

    # -- F ------------------------------------------------------------------
    def _kinetics(self, state: State, v: np.ndarray, dt: float,
                  temperature_k: float | None) -> State:
        out = state.copy()
        for name in STATES:
            active = np.ones(len(v), bool) if name == "fast" else self.has_slow
            if not active.any():
                continue
            write_time = (self.p[f"{name}_write_time_s"]
                          * self._arrhenius(f"{name}_write_activation_ev", temperature_k))
            retention = (self.p[f"{name}_retention_s"]
                         * self._arrhenius(f"{name}_retention_activation_ev", temperature_k))
            scale = self.p[f"{name}_saturation_voltage_v"]
            target = np.tanh(np.where(self.even, np.abs(v), v)[active] / scale[active])
            values = getattr(state, name).copy()
            values[active] = self._approach(
                values[active],
                self._rate(v[active], self.p[f"{name}_write_voltage_v"][active],
                           write_time[active]),
                1.0 / retention[active], target, dt)
            setattr(out, name, values)
        f = self.has_fatigue
        if f.any():
            recovery = (self.p["fatigue_recovery_s"]
                        * self._arrhenius("recovery_activation_ev", temperature_k))
            fatigue = state.fatigue.copy()
            fatigue[f] = self._approach(
                fatigue[f],
                self._rate(v[f], self.p["fatigue_voltage_v"][f], self.p["fatigue_time_s"][f]),
                1.0 / recovery[f], np.ones(int(f.sum())), dt)
            out.fatigue = fatigue
        return out

    # -- G ------------------------------------------------------------------
    def branch_elements(self, state: State, v: np.ndarray, temperature_k: float | None
                        ) -> dict[str, np.ndarray]:
        """State-, bias- and temperature-dependent R_inf, delta R and C per branch."""
        usable = 1.0 - self.p["fatigue_depth"] * state.fatigue
        shift = {q: usable * (self.p[f"fast_gain_{q}"] * state.fast
                              + self.p[f"slow_gain_{q}"] * state.slow)
                 for q in ("r_inf", "delta_r", "capacitance")}
        bias = self.p["bias_linear_per_v"] * v + self.p["bias_quadratic_per_v2"] * v**2
        thermal = self._arrhenius("resistance_activation_ev", temperature_k)

        def scaled(key: str, exponent: np.ndarray) -> np.ndarray:
            return self.p[key] * np.exp(np.clip(exponent, -MAX_EXPONENT, MAX_EXPONENT))

        return {
            "r_inf_ohm": scaled("r_inf_ohm", shift["r_inf"]) * thermal,
            "delta_r_ohm": scaled("delta_r_ohm", shift["delta_r"] + bias) * thermal,
            "capacitance_f": scaled("capacitance_f", shift["capacitance"]),
        }

    def port_voltages(self, state: State, voltages: dict, temperature_k: float | None
                      ) -> np.ndarray:
        """All node voltages (ground last): driven ports fixed, floating ports solved (DC)."""
        if not isinstance(voltages, dict) or {str(k) for k in voltages} != set(self.ports):
            raise ValueError(f"give a voltage (or null for floating) for every port "
                             f"{self.ports}")
        n = len(self.ports)
        node = np.zeros(n + 1)
        floating = []
        for port, value in voltages.items():
            i = self.index(str(port))
            if value is None:
                floating.append(i)
            elif isinstance(value, bool) or not np.isfinite(value) or abs(value) > 1e4:
                raise ValueError("port voltages must be finite numbers of volts, or null")
            else:
                node[i] = float(value)
        if not floating:
            return node
        driven = [i for i in range(n + 1) if i not in floating]
        free = sorted(floating)
        for _ in range(200):
            v = self.incidence @ node
            e = self.branch_elements(state, v, temperature_k)
            g = 1.0 / (e["r_inf_ohm"] + e["delta_r_ohm"])
            laplacian = self.incidence.T @ (g[:, None] * self.incidence)
            solved = np.linalg.solve(laplacian[np.ix_(free, free)],
                                     -laplacian[np.ix_(free, driven)] @ node[driven])
            change = np.max(np.abs(solved - node[free]))
            node[free] = 0.5 * (node[free] + solved) if change > 0.1 else solved
            if change < 1e-9:
                return node
        raise ValueError("floating-port voltages did not converge (bias dependence too "
                         "strong)")

    def dc_current(self, state: State, node: np.ndarray, temperature_k: float | None
                   ) -> np.ndarray:
        """Direct current into every port."""
        v = self.incidence @ node
        e = self.branch_elements(state, v, temperature_k)
        current = v / (e["r_inf_ohm"] + e["delta_r_ohm"])
        return (self.incidence.T @ current)[:-1]

    def impedance_matrix(self, frequency_hz: np.ndarray, state: State,
                         node: np.ndarray | None = None,
                         temperature_k: float | None = None) -> np.ndarray:
        """Small-signal Z(omega) of shape (n_frequencies, N, N) about the bias `node`."""
        f = np.atleast_1d(np.asarray(frequency_hz, dtype=float))
        if f.ndim != 1 or not np.all(np.isfinite(f)) or np.any(f < 0):
            raise ValueError("frequencies must be finite and non-negative")
        n = len(self.ports)
        v = np.zeros(len(self.branches)) if node is None else self.incidence @ node
        e = self.branch_elements(state, v, temperature_k)
        jw = 2j * np.pi * f[:, None]
        relax = (jw * self.p["tau_dielectric_s"]) ** self.p["alpha"]
        y = (1.0 / (e["r_inf_ohm"] + e["delta_r_ohm"] / (1.0 + relax))
             + jw * e["capacitance_f"])
        incidence = self.incidence[:, :n]
        admittance = np.einsum("fb,bi,bj->fij", y, incidence, incidence, optimize=True)
        return np.linalg.inv(admittance)

    # -- time stepping ------------------------------------------------------
    def step(self, state: State, voltages: dict, duration_s: float,
             temperature_k: float | None = None, substeps: int = 20) -> State:
        """Hold the port voltages for duration_s.

        With every port driven the update is exact.  With floating ports their voltages
        follow the changing branch resistances, resolved in `substeps` steps.
        """
        if not (np.isfinite(duration_s) and duration_s >= 0):
            raise ValueError("duration_s must be finite and non-negative")
        floating = any(v is None for v in voltages.values())
        count = max(int(substeps), 1) if floating else 1
        for _ in range(count):
            node = self.port_voltages(state, voltages, temperature_k)
            if duration_s:
                state = self._kinetics(state, self.incidence @ node, duration_s / count,
                                       temperature_k)
        return state


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


def parse_protocol(device: StateSpaceDevice, protocol: dict) -> list[dict[str, Any]]:
    """{"segments": [{"duration_s", "voltages_v": {port: V or null}, "temperature_k"}],
    "repeat": n} -> flat list of segments."""
    if not isinstance(protocol, dict) or set(protocol) - {"segments", "repeat"}:
        raise ValueError("protocol is {'segments': [...], 'repeat': n}")
    segments = protocol.get("segments")
    repeat = protocol.get("repeat", 1)
    if not segments:
        raise ValueError("protocol has no segments")
    if not isinstance(repeat, int) or repeat < 1 or repeat * len(segments) > MAX_STEPS:
        raise ValueError(f"repeat must be a positive integer, at most {MAX_STEPS} segments")
    out = []
    for k, segment in enumerate(segments):
        extra = set(segment) - {"duration_s", "voltages_v", "temperature_k"}
        if extra or "duration_s" not in segment or "voltages_v" not in segment:
            raise ValueError(f"segment {k} needs duration_s and voltages_v (optional "
                             f"temperature_k); unexpected: {sorted(extra)}")
        duration = float(segment["duration_s"])
        if not (np.isfinite(duration) and duration > 0):
            raise ValueError(f"segment {k}: duration_s must be positive")
        temperature = segment.get("temperature_k")
        if temperature is None and device.needs_temperature:
            raise ValueError(f"segment {k}: this model depends on temperature_k")
        out.append({"duration_s": duration, "voltages_v": dict(segment["voltages_v"]),
                    "temperature_k": None if temperature is None else float(temperature)})
    return out * repeat


def run_protocol(device: StateSpaceDevice, protocol: dict, read_times_s: list[float],
                 initial_state: dict | None = None, substeps: int = 20
                 ) -> list[dict[str, Any]]:
    """Advance through the protocol and snapshot the state at each read time (sorted).

    A read on a segment boundary sees the segment that is ending (t = 0: the first).
    Each snapshot: time_s, state, node voltages, temperature_k, dc_current_a.
    """
    segments = parse_protocol(device, protocol)
    times = np.asarray(read_times_s, dtype=float)
    total = sum(s["duration_s"] for s in segments)
    slack = 1e-9 * max(total, 1.0)
    if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)):
        raise ValueError("read times must be a non-empty list of numbers")
    if times.min() < 0 or times.max() > total + slack:
        raise ValueError(f"read times must lie within the protocol (0 to {total:g} s)")
    if np.any(np.diff(times) < 0):
        raise ValueError("read times must be sorted")
    state = device.initial_state(initial_state)
    out: list[dict[str, Any]] = []
    clock, k = 0.0, 0
    for index, segment in enumerate(segments):
        end = clock + segment["duration_s"]
        volts, temperature = segment["voltages_v"], segment["temperature_k"]
        last = index == len(segments) - 1
        done = 0.0
        while k < len(times) and (times[k] <= end + slack or last):
            at = min(max(times[k] - clock, 0.0), segment["duration_s"])
            if at > done:
                share = max(1, int(np.ceil(substeps * (at - done) / segment["duration_s"])))
                state = device.step(state, volts, at - done, temperature, share)
                done = at
            node = device.port_voltages(state, volts, temperature)
            out.append({"time_s": float(times[k]), "state": state.copy(), "node_v": node,
                        "temperature_k": temperature,
                        "dc_current_a": device.dc_current(state, node, temperature)})
            k += 1
        if k >= len(times):
            break
        rest = segment["duration_s"] - done
        if rest > 0:
            share = max(1, int(np.ceil(substeps * rest / segment["duration_s"])))
            state = device.step(state, volts, rest, temperature, share)
        clock = end
    return out


def _matrix(z: np.ndarray) -> dict[str, Any]:
    return {"real_ohm": z.real.tolist(), "imag_ohm": z.imag.tolist()}


def _warnings(device: StateSpaceDevice) -> list[str]:
    out = ["phenomenological model: results are only as good as the fit of its parameters "
           "to measurements of this sample and fixture"]
    if len(device.ports) > 2 and len(device.shared_branches) > 0:
        out.append(
            f"{len(device.shared_branches)} of {len(device.branches)} branches use shared "
            "parameters: how coupling depends on electrode geometry is not modelled, so "
            "this many-port network is an extrapolation until fitted branch by branch")
    return out


def simulate(model: dict, protocol: dict, read_times_s: list[float],
             frequency_hz: list[float], initial_state: dict | None = None,
             substeps: int = 20) -> dict[str, Any]:
    """Z(omega, t) as an N x N matrix at each read time, with states and DC currents."""
    device = StateSpaceDevice(model)
    n, nf, nt = len(device.ports), len(frequency_hz), len(read_times_s)
    if nf == 0 or nf * nt * n * n > 2_000_000:
        raise ValueError("ask for fewer frequencies or read times (2,000,000 matrix "
                         "entries at most)")
    snapshots = run_protocol(device, protocol, read_times_s, initial_state, substeps)
    reads = []
    for s in snapshots:
        z = device.impedance_matrix(frequency_hz, s["state"], s["node_v"], s["temperature_k"])
        reads.append({
            "time_s": s["time_s"], "temperature_k": s["temperature_k"],
            "port_voltage_v": dict(zip(device.ports, s["node_v"][:-1].tolist(), strict=True)),
            "dc_current_a": dict(zip(device.ports, s["dc_current_a"].tolist(), strict=True)),
            "z": _matrix(z), "state": device.export_state(s["state"])})
    out: dict[str, Any] = {
        "model_version": MODEL_VERSION, "ports": device.ports, "branches": device.branches,
        "frequency_hz": [float(f) for f in frequency_hz],
        "z_index_order": "z[frequency][to_port][from_port]", "reads": reads,
        "calibration_status": "fitted" if (model.get("provenance") or {}).get("fit")
        else "uncalibrated: parameters supplied without a fit record",
        "warnings": _warnings(device), "not_modelled": NOT_MODELLED,
    }
    if any(v is None for s in protocol["segments"] for v in s["voltages_v"].values()):
        fine = run_protocol(device, protocol, read_times_s, initial_state, substeps * 2)
        out["substep_convergence"] = float(max(
            np.max(np.abs(a["state"].fast - b["state"].fast))
            + np.max(np.abs(a["state"].slow - b["state"].slow))
            for a, b in zip(snapshots, fine, strict=True)))
        if out["substep_convergence"] > 0.01:
            out["warnings"].append("floating-port time stepping is not converged: raise "
                                   "substeps")
    return out


# ---------------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------------


def sensitivity(model: dict, protocol: dict, frequency_hz: list[float],
                relative_step: float = 0.05, read_time_s: float | None = None
                ) -> dict[str, Any]:
    """Change in Z and state at one read time for a small increase of each given parameter."""
    if not 0 < relative_step <= 0.3:
        raise ValueError("relative_step must be in (0, 0.3]")
    device = StateSpaceDevice(model)
    segments = parse_protocol(device, protocol)
    total = sum(s["duration_s"] for s in segments)
    when = total if read_time_s is None else float(read_time_s)

    def outputs(candidate: dict) -> tuple[np.ndarray, np.ndarray]:
        d = StateSpaceDevice(candidate)
        s = run_protocol(d, protocol, [when])[0]
        z = d.impedance_matrix(frequency_hz, s["state"], s["node_v"], s["temperature_k"])
        return z, np.concatenate([s["state"].fast, s["state"].slow])

    z0, s0 = outputs(model)
    groups = [("ground", model["ground"])]
    if model.get("coupling"):
        groups.append(("coupling", model["coupling"]))
    groups += [(f"branches.{k}", v) for k, v in (model.get("branches") or {}).items()]
    table: dict[str, Any] = {}
    for label, values in groups:
        for key, value in values.items():
            path = f"{label}.{key}"
            if isinstance(value, str):
                continue
            if value == 0:
                table[path] = {"status": "not run: value is zero (effect switched off)"}
                continue
            candidate = copy.deepcopy(model)
            target = candidate
            for part in label.split("."):
                target = target[part]
            target[key] = value * (1 + relative_step)
            try:
                z, s = outputs(candidate)
            except ValueError as error:
                table[path] = {"status": f"not run: {error}"}
                continue
            dz = float(np.max(np.abs(z - z0) / np.abs(z0)))
            ds = float(np.max(np.abs(s - s0)))
            table[path] = {"status": "computed", "max_relative_z_change": dz,
                           "max_state_change": ds,
                           "affects_this_protocol": bool(dz > 1e-9 or ds > 1e-9)}
    return {
        "relative_step": relative_step, "read_time_s": when, "parameters": table,
        "no_effect": [p for p, r in table.items()
                      if r["status"] == "computed" and not r["affects_this_protocol"]],
        "how_to_read": "a parameter has no effect when its mechanism is idle in this "
                       "protocol, e.g. temperature terms at the reference temperature, a "
                       "gain on a state that was never written, or a shared value that "
                       "every branch overrides",
    }


# ---------------------------------------------------------------------------
# Multiport interaction and retrieval
# ---------------------------------------------------------------------------


def multiport_report(model: dict, write_voltages_v: dict, duration_s: float,
                     frequency_hz: list[float], idle_ports: str = "float",
                     temperature_k: float | None = None, hold_s: float = 0.0,
                     noise_sigma_ohm: float | None = None, snr_threshold: float = 3.0,
                     substeps: int = 20) -> dict[str, Any]:
    """Writes alone versus together, and whether written patterns can be told apart.

    A "bit" is port i either pulsed with write_voltages_v[i] or left idle.  The readout is
    every entry of Z at `frequency_hz` after `hold_s` with all ports at 0 V.
    """
    if idle_ports not in ("float", "ground"):
        raise ValueError("idle_ports must be 'float' or 'ground'")
    if not (duration_s > 0 and hold_s >= 0):
        raise ValueError("duration_s must be positive and hold_s non-negative")
    if noise_sigma_ohm is not None and not (noise_sigma_ohm > 0 and snr_threshold > 0):
        raise ValueError("noise_sigma_ohm and snr_threshold must be positive")
    device = StateSpaceDevice(model)
    written = {device.index(str(p)): float(v) for p, v in (write_voltages_v or {}).items()}
    if not written:
        raise ValueError("write_voltages_v needs at least one port")
    order = sorted(written)
    zero = {p: 0.0 for p in device.ports}
    cache: dict[frozenset, np.ndarray] = {}

    def readout(active: frozenset) -> np.ndarray:
        if active not in cache:
            state = device.initial_state()
            if active:
                volts = {p: written[i] if i in active else (
                    0.0 if idle_ports == "ground" else None)
                    for i, p in enumerate(device.ports)}
                state = device.step(state, volts, duration_s, temperature_k, substeps)
            if hold_s:
                state = device.step(state, zero, hold_s, temperature_k)
            cache[active] = device.impedance_matrix(
                frequency_hz, state, None, temperature_k)
        return cache[active]

    base = readout(frozenset())
    alone = {i: readout(frozenset([i])) - base for i in order}
    together = readout(frozenset(order)) - base
    summed = sum(alone.values())
    scale = float(np.linalg.norm(together))
    per_port = {}
    for i in order:
        self_change = float(np.linalg.norm(alone[i][:, i, i]))
        per_port[device.ports[i]] = {
            "self_change_ohm": self_change,
            "change_on_other_ports": {
                device.ports[j]: (float(np.linalg.norm(alone[i][:, j, j])) / self_change
                                  if self_change else None)
                for j in range(len(device.ports)) if j != i}}
    out: dict[str, Any] = {
        "ports": device.ports, "written_ports": [device.ports[i] for i in order],
        "idle_ports": idle_ports, "n_ports": len(device.ports),
        "write_alone": per_port,
        "change_when_written_together_ohm": scale,
        "interaction_residual_fraction": (
            float(np.linalg.norm(together - summed)) / scale if scale else None),
        "definition": "residual = Z change with all writes at once minus the sum of the "
                      "changes from each write alone; 0 means independent ports",
        "warnings": _warnings(device), "not_modelled": NOT_MODELLED,
    }
    if noise_sigma_ohm is None:
        out["retrieval"] = {"status": "not_computed",
                            "reason": "give noise_sigma_ohm, the measured noise of one "
                                      "impedance reading"}
        return out
    k = len(order)
    exhaustive = k <= 8
    if exhaustive:
        patterns = [frozenset(order[b] for b in range(k) if code >> b & 1)
                    for code in range(2**k)]
        pairs = list(combinations(range(len(patterns)), 2))
    else:  # neighbours of "nothing written" and "everything written" only
        everything = frozenset(order)
        patterns = [frozenset(), everything] + [frozenset([i]) for i in order] + [
            everything - {i} for i in order]
        pairs = [(0, 2 + b) for b in range(k)] + [(1, 2 + k + b) for b in range(k)]
    features = [np.concatenate([readout(p).real.ravel(), readout(p).imag.ravel()])
                for p in patterns]
    distance = [float(np.linalg.norm(features[a] - features[b])) / noise_sigma_ohm
                for a, b in pairs]
    worst = int(np.argmin(distance))
    resolved = distance[worst] >= snr_threshold
    out["retrieval"] = {
        "status": ("all_patterns_distinguishable" if resolved else "not_distinguishable")
        if exhaustive else
        ("partial_check_passed" if resolved else "not_distinguishable"),
        "patterns_checked": len(patterns), "possible_patterns": 2**k,
        "exhaustive": exhaustive,
        "smallest_separation_in_noise_units": distance[worst],
        "closest_pair": [sorted(device.ports[i] for i in patterns[p]) for p in pairs[worst]],
        "noise_sigma_ohm": noise_sigma_ohm, "threshold": snr_threshold,
        "assumptions": "ideal joint readout of all matrix entries with independent noise; "
                       "a distinct pattern is necessary for retrieval, not a decoder",
    }
    return out
