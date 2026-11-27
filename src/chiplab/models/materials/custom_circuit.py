"""
Custom Circuit Builder & GPU-Accelerated MNA Simulator (Multi-nm Support)
==========================================================================

Supports arbitrary user-defined mixed CMOS + emerging/experimental circuits:
- Standard elements: Resistor, Capacitor, Inductor, Diode, VSource (DC/Pulse/Sine), ISource
- Multi-nm Transistors: NMOS, PMOS (scaled via ProcessNodeParams for any nm size)
- Emerging/Experimental Devices: Memristor (ReRAM), FeFET, MTJ (MRAM), PCM, FTJ,
  NCFET, TFET, TMD_FET
- Solvers: Modified Nodal Analysis (MNA) DC operating point, Backward-Euler
  Transient simulation, GPU-accelerated linear solves & Monte Carlo variation
- SPICE netlist import/export (safe parser, no eval/exec)
- Pre-built customizable templates: CMOS Inverter, Ring Oscillator, 6T SRAM,
  1T1R ReRAM/FeFET/MRAM Cell, Hybrid Crossbar Column (with nm wire RC)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from chiplab.core.gpu import GPUBackend, get_gpu_backend as get_backend
from chiplab.models.materials.process_nodes import (
    ProcessCorner,
    ProcessNodeParams,
    get_process_node,
    mosfet_iv,
)
from chiplab.schemas.common import Provenance, TrackedValue


class ComponentType(Enum):
    """Supported circuit component types."""
    RESISTOR = "resistor"
    CAPACITOR = "capacitor"
    INDUCTOR = "inductor"
    DIODE = "diode"
    V_SOURCE = "v_source"
    I_SOURCE = "i_source"
    NMOS = "nmos"
    PMOS = "pmos"
    MEMRISTOR = "memristor"
    FEFET = "fefet"
    MTJ = "mtj"
    PCM = "pcm"
    FTJ = "ftj"
    NCFET = "ncfet"
    TFET = "tfet"
    TMD_FET = "tmd_fet"


@dataclass
class CircuitComponent:
    """A single component in a CustomCircuit."""
    name: str
    comp_type: ComponentType
    nodes: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    width_nm: float | None = None
    length_nm: float | None = None
    node_nm: float | None = None  # Optional per-device nm override


class CustomCircuit:
    """Custom mixed-technology circuit with multi-nm scaling and MNA simulation."""

    def __init__(
        self,
        name: str = "custom_circuit",
        node_nm: float = 28.0,
        corner: str | ProcessCorner = ProcessCorner.TT,
        temperature_k: float = 300.0,
    ) -> None:
        self.name = name
        self.process_node: ProcessNodeParams = get_process_node(
            node_nm=node_nm,
            corner=corner,
            temperature_k=temperature_k,
        )
        self.node_nm = TrackedValue(
            value=float(node_nm),
            provenance=self.process_node.node_nm.provenance,
            metadata={"description": "Circuit process node", "unit": "nm"},
        )
        self.components: list[CircuitComponent] = []
        self._node_cache: dict[float, ProcessNodeParams] = {
            float(node_nm): self.process_node
        }

    def _get_node_params(self, override_nm: float | None) -> ProcessNodeParams:
        if override_nm is None or abs(override_nm - self.node_nm.value) < 1e-9:
            return self.process_node
        key = float(override_nm)
        if key not in self._node_cache:
            self._node_cache[key] = get_process_node(
                node_nm=key,
                corner=self.process_node.corner,
                temperature_k=self.process_node.temperature_k.value,
            )
        return self._node_cache[key]

    def set_process_node(
        self,
        node_nm: float,
        corner: str | ProcessCorner | None = None,
        temperature_k: float | None = None,
    ) -> ProcessNodeParams:
        """Re-target the entire circuit to a different nm node / corner / temperature."""
        c = corner if corner is not None else self.process_node.corner
        t = temperature_k if temperature_k is not None else self.process_node.temperature_k.value
        self.process_node = get_process_node(node_nm=node_nm, corner=c, temperature_k=t)
        self.node_nm = TrackedValue(
            value=float(node_nm),
            provenance=self.process_node.node_nm.provenance,
            metadata={"description": "Circuit process node", "unit": "nm"},
        )
        self._node_cache = {float(node_nm): self.process_node}
        return self.process_node

    def add_component(
        self,
        name: str,
        comp_type: ComponentType | str,
        nodes: list[str],
        params: dict[str, Any] | None = None,
        width_nm: float | None = None,
        length_nm: float | None = None,
        node_nm: float | None = None,
    ) -> CircuitComponent:
        ct = comp_type if isinstance(comp_type, ComponentType) else ComponentType(comp_type.lower())
        comp = CircuitComponent(
            name=name,
            comp_type=ct,
            nodes=[str(n) for n in nodes],
            params=dict(params or {}),
            width_nm=width_nm,
            length_nm=length_nm,
            node_nm=node_nm,
        )
        self.components.append(comp)
        return comp

    # Convenience builders
    def add_resistor(self, name: str, n1: str, n2: str, resistance_ohm: float) -> CircuitComponent:
        return self.add_component(name, ComponentType.RESISTOR, [n1, n2], {"resistance": float(resistance_ohm)})

    def add_wire_segment(self, name: str, n1: str, n2: str, length_um: float) -> tuple[CircuitComponent, CircuitComponent]:
        """Add nm-scaled RC wire segment (series R + shunt C to GND) using process node parasitics."""
        r_val = max(0.01, self.process_node.wire_r_ohm_per_um.value * length_um)
        c_val = max(1e-18, self.process_node.wire_c_ff_per_um.value * 1e-15 * length_um)
        r = self.add_resistor(f"{name}_R", n1, n2, r_val)
        c = self.add_capacitor(f"{name}_C", n2, "0", c_val)
        return r, c

    def add_capacitor(self, name: str, n1: str, n2: str, capacitance_f: float) -> CircuitComponent:
        return self.add_component(name, ComponentType.CAPACITOR, [n1, n2], {"capacitance": float(capacitance_f)})

    def add_inductor(self, name: str, n1: str, n2: str, inductance_h: float) -> CircuitComponent:
        return self.add_component(name, ComponentType.INDUCTOR, [n1, n2], {"inductance": float(inductance_h)})

    def add_vsource(
        self,
        name: str,
        p_node: str,
        n_node: str = "0",
        dc: float = 0.0,
        waveform: str = "dc",
        v_low: float = 0.0,
        v_high: float | None = None,
        period: float = 1e-9,
        duty: float = 0.5,
        delay: float = 0.0,
        rise: float = 2e-11,
        freq: float = 1e9,
        amp: float = 0.5,
    ) -> CircuitComponent:
        return self.add_component(
            name,
            ComponentType.V_SOURCE,
            [p_node, n_node],
            {
                "dc": float(dc),
                "waveform": waveform.lower(),
                "v_low": float(v_low),
                "v_high": float(self.process_node.vdd.value if v_high is None else v_high),
                "period": float(period),
                "duty": float(duty),
                "delay": float(delay),
                "rise": float(rise),
                "freq": float(freq),
                "amp": float(amp),
            },
        )

    def add_isource(self, name: str, p_node: str, n_node: str = "0", dc: float = 0.0) -> CircuitComponent:
        return self.add_component(name, ComponentType.I_SOURCE, [p_node, n_node], {"dc": float(dc)})

    def add_nmos(
        self,
        name: str,
        drain: str,
        gate: str,
        source: str,
        width_nm: float | None = None,
        length_nm: float | None = None,
        vt_shift_v: float = 0.0,
        node_nm: float | None = None,
    ) -> CircuitComponent:
        return self.add_component(
            name, ComponentType.NMOS, [drain, gate, source],
            {"vt_shift": float(vt_shift_v)}, width_nm=width_nm, length_nm=length_nm, node_nm=node_nm,
        )

    def add_pmos(
        self,
        name: str,
        drain: str,
        gate: str,
        source: str,
        width_nm: float | None = None,
        length_nm: float | None = None,
        vt_shift_v: float = 0.0,
        node_nm: float | None = None,
    ) -> CircuitComponent:
        return self.add_component(
            name, ComponentType.PMOS, [drain, gate, source],
            {"vt_shift": float(vt_shift_v)}, width_nm=width_nm, length_nm=length_nm, node_nm=node_nm,
        )

    def add_memristor(
        self,
        name: str,
        top: str,
        bottom: str,
        r_on: float = 1e3,
        r_off: float = 1e5,
        state_x: float = 0.5,
        v_th: float = 0.4,
        tau_s: float = 1e-9,
    ) -> CircuitComponent:
        return self.add_component(
            name, ComponentType.MEMRISTOR, [top, bottom],
            {"r_on": float(r_on), "r_off": float(r_off), "x": float(state_x), "v_th": float(v_th), "tau": float(tau_s)},
        )

    def add_fefet(
        self,
        name: str,
        drain: str,
        gate: str,
        source: str,
        polarization: float = 0.0,
        memory_window_v: float = 0.8,
        width_nm: float | None = None,
        length_nm: float | None = None,
    ) -> CircuitComponent:
        return self.add_component(
            name, ComponentType.FEFET, [drain, gate, source],
            {"polarization": float(polarization), "memory_window": float(memory_window_v)},
            width_nm=width_nm, length_nm=length_nm,
        )

    def add_mtj(
        self,
        name: str,
        free_node: str,
        ref_node: str,
        r_p: float = 2e3,
        tmr: float = 1.5,
        mz: float = 1.0,
        i_c: float = 50e-6,
    ) -> CircuitComponent:
        return self.add_component(
            name, ComponentType.MTJ, [free_node, ref_node],
            {"r_p": float(r_p), "tmr": float(tmr), "mz": float(mz), "i_c": float(i_c)},
        )

    def add_emerging_fet(
        self,
        name: str,
        comp_type: ComponentType | str,
        drain: str,
        gate: str,
        source: str,
        width_nm: float | None = None,
        length_nm: float | None = None,
    ) -> CircuitComponent:
        """Add NCFET, TFET, or TMD_FET."""
        return self.add_component(
            name, comp_type, [drain, gate, source], {}, width_nm=width_nm, length_nm=length_nm
        )

    def get_node_names(self) -> list[str]:
        """Return sorted non-ground node names."""
        nodes: set[str] = set()
        for c in self.components:
            for n in c.nodes:
                if n not in ("0", "gnd", "GND"):
                    nodes.add(n)
        return sorted(nodes)

    @staticmethod
    def _eval_vsource(comp: CircuitComponent, t: float) -> float:
        p = comp.params
        wf = p.get("waveform", "dc")
        if wf == "dc":
            return float(p.get("dc", 0.0))
        if wf == "sine":
            return float(p.get("dc", 0.0) + p.get("amp", 0.5) * np.sin(2.0 * np.pi * p.get("freq", 1e9) * t))
        if wf == "pulse":
            delay = p.get("delay", 0.0)
            if t < delay:
                return float(p.get("v_low", 0.0))
            period = max(p.get("period", 1e-9), 1e-15)
            phase = (t - delay) % period
            on_time = p.get("duty", 0.5) * period
            rise = max(p.get("rise", 1e-12), 1e-15)
            v_lo = float(p.get("v_low", 0.0))
            v_hi = float(p.get("v_high", 0.9))
            if phase < rise:
                return v_lo + (v_hi - v_lo) * (phase / rise)
            if phase < on_time:
                return v_hi
            if phase < on_time + rise:
                return v_hi - (v_hi - v_lo) * ((phase - on_time) / rise)
            return v_lo
        return float(p.get("dc", 0.0))

    def _device_conductance_and_current(
        self,
        comp: CircuitComponent,
        v_nodes: np.ndarray,
        node_idx: dict[str, int],
        dev_state: dict[str, dict[str, float]],
        vt_jitter_map: dict[str, float] | None = None,
    ) -> tuple[float, float]:
        """Compute instantaneous current & small-signal conductance for nonlinear/2-terminal & 3-terminal devices."""
        def _v(n: str) -> float:
            if n in ("0", "gnd", "GND"):
                return 0.0
            return float(v_nodes[node_idx[n]])

        ct = comp.comp_type
        if ct == ComponentType.MEMRISTOR:
            st = dev_state[comp.name]
            x = np.clip(st["x"], 0.0, 1.0)
            r = comp.params.get("r_on", 1e3) * x + comp.params.get("r_off", 1e5) * (1.0 - x)
            g = 1.0 / max(r, 1.0)
            return g, 0.0

        if ct == ComponentType.PCM:
            st = dev_state[comp.name]
            c_frac = np.clip(st["x"], 0.0, 1.0)
            r = comp.params.get("r_set", 2e3) * c_frac + comp.params.get("r_reset", 2e5) * (1.0 - c_frac)
            return 1.0 / max(r, 1.0), 0.0

        if ct == ComponentType.MTJ:
            st = dev_state[comp.name]
            mz = np.clip(st["mz"], -1.0, 1.0)
            r_p = comp.params.get("r_p", 2e3)
            tmr = comp.params.get("tmr", 1.5)
            r_ap = r_p * (1.0 + tmr)
            r = 0.5 * ((1.0 + mz) * r_p + (1.0 - mz) * r_ap)
            return 1.0 / max(r, 1.0), 0.0

        if ct == ComponentType.FTJ:
            st = dev_state[comp.name]
            pol = np.clip(st["pol"], -1.0, 1.0)
            r_on = comp.params.get("r_on", 5e3)
            r_off = comp.params.get("r_off", 5e5)
            r = np.exp(np.log(r_on) + (np.log(r_off) - np.log(r_on)) * (1.0 - pol) * 0.5)
            return 1.0 / max(r, 1.0), 0.0

        if ct == ComponentType.DIODE:
            vd = _v(comp.nodes[0]) - _v(comp.nodes[1])
            is_val = comp.params.get("is", 1e-14)
            n_vt = 0.026 * comp.params.get("n", 1.2)
            arg = np.clip(vd / n_vt, -30.0, 35.0)
            i_d = is_val * (np.exp(arg) - 1.0)
            g_d = (is_val / n_vt) * np.exp(arg)
            return float(g_d), float(i_d - g_d * vd)

        return 0.0, 0.0

    def _fet_ids_gm_gds(
        self,
        comp: CircuitComponent,
        vd: float,
        vg: float,
        vs: float,
        dev_state: dict[str, dict[str, float]],
        vt_jitter: float = 0.0,
    ) -> tuple[float, float, float]:
        """Compute (Ids, gm, gds) for NMOS, PMOS, FEFET, NCFET, TFET, TMD_FET."""
        pnode = self._get_node_params(comp.node_nm)
        ct = comp.comp_type
        is_pmos = (ct == ComponentType.PMOS)

        # Effective Vth shift for FeFET / NCFET / TFET / TMD_FET
        vt_shift = float(comp.params.get("vt_shift", 0.0)) + vt_jitter
        if ct == ComponentType.FEFET:
            pol = dev_state[comp.name]["pol"]
            mw = comp.params.get("memory_window", 0.8)
            vt_shift -= 0.5 * mw * pol
        elif ct == ComponentType.NCFET:
            # Negative capacitance boosts effective gate overdrive (~1.25x) and lowers SS
            vg = vs + 1.25 * (vg - vs)
        elif ct == ComponentType.TFET:
            # Steep slope, lower saturation drive (~0.45x CMOS)
            pass

        if is_pmos:
            vsg = vs - vg
            vsd = vs - vd
            vgs_q, vds_q = vsg, vsd
        else:
            vgs_q = vg - vs
            vds_q = vd - vs

        # Apply vt_shift via effective gate voltage
        vgs_eff = vgs_q - vt_shift
        eps = 1e-3

        w = comp.width_nm or (6.0 * pnode.node_nm.value if is_pmos else 4.0 * pnode.node_nm.value)
        l = comp.length_nm or pnode.gate_length_nm.value

        vals = mosfet_iv(
            pnode,
            np.array([vgs_eff, vgs_eff + eps, vgs_eff]),
            np.array([vds_q, vds_q, vds_q + eps]),
            width_nm=w,
            length_nm=l,
            is_pmos=is_pmos,
            device="cpu",
        )
        scale = 1.0
        if ct == ComponentType.TFET:
            scale = 0.45
        elif ct == ComponentType.TMD_FET:
            scale = 1.15 * (7.0 / max(pnode.node_nm.value, 1.0)) ** 0.15

        ids = float(vals[0]) * scale
        gm = max(1e-12, float((vals[1] - vals[0]) / eps) * scale)
        gds = max(1e-12, float((vals[2] - vals[0]) / eps) * scale)

        if is_pmos:
            # Current flows from source -> drain, so I_ds (drain out) = -I_sd
            return -ids, gm, gds
        return ids, gm, gds

    def _solve_mna_step(
        self,
        t: float,
        dt: float | None,
        v_prev: np.ndarray,
        i_ind_prev: dict[str, float],
        dev_state: dict[str, dict[str, float]],
        vt_jitter_map: dict[str, float] | None = None,
        backend: GPUBackend | None = None,
        max_iter: int = 30,
        tol: float = 1e-5,
    ) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
        """Solve one DC (dt=None) or Backward-Euler transient step via Newton-Raphson MNA."""
        node_names = self.get_node_names()
        n_nodes = len(node_names)
        node_idx = {name: i for i, name in enumerate(node_names)}

        vsources = [c for c in self.components if c.comp_type == ComponentType.V_SOURCE]
        inductors = [c for c in self.components if c.comp_type == ComponentType.INDUCTOR]
        n_vs = len(vsources)
        n_ind = len(inductors)
        dim = n_nodes + n_vs + n_ind

        if dim == 0:
            return np.zeros(0), {}, {}

        x = np.zeros(dim, dtype=np.float64)
        x[:n_nodes] = v_prev[:n_nodes] if len(v_prev) >= n_nodes else 0.0

        def _idx(n: str) -> int:
            return -1 if n in ("0", "gnd", "GND") else node_idx[n]

        g_min = 1e-10
        fet_types = {
            ComponentType.NMOS, ComponentType.PMOS, ComponentType.FEFET,
            ComponentType.NCFET, ComponentType.TFET, ComponentType.TMD_FET,
        }

        for _ in range(max_iter):
            A = np.zeros((dim, dim), dtype=np.float64)
            b = np.zeros(dim, dtype=np.float64)

            # Gmin to ground for numerical stability
            for i in range(n_nodes):
                A[i, i] += g_min

            for comp in self.components:
                ct = comp.comp_type
                if ct == ComponentType.RESISTOR:
                    g = 1.0 / max(comp.params["resistance"], 1e-6)
                    i1, i2 = _idx(comp.nodes[0]), _idx(comp.nodes[1])
                    if i1 >= 0:
                        A[i1, i1] += g
                    if i2 >= 0:
                        A[i2, i2] += g
                    if i1 >= 0 and i2 >= 0:
                        A[i1, i2] -= g
                        A[i2, i1] -= g

                elif ct == ComponentType.CAPACITOR:
                    if dt is not None and dt > 0:
                        c_val = comp.params["capacitance"]
                        g_eq = c_val / dt
                        i1, i2 = _idx(comp.nodes[0]), _idx(comp.nodes[1])
                        v1p = v_prev[i1] if i1 >= 0 else 0.0
                        v2p = v_prev[i2] if i2 >= 0 else 0.0
                        i_eq = g_eq * (v1p - v2p)
                        if i1 >= 0:
                            A[i1, i1] += g_eq
                            b[i1] += i_eq
                        if i2 >= 0:
                            A[i2, i2] += g_eq
                            b[i2] -= i_eq
                        if i1 >= 0 and i2 >= 0:
                            A[i1, i2] -= g_eq
                            A[i2, i1] -= g_eq

                elif ct == ComponentType.I_SOURCE:
                    i_dc = comp.params.get("dc", 0.0)
                    i1, i2 = _idx(comp.nodes[0]), _idx(comp.nodes[1])
                    if i1 >= 0:
                        b[i1] -= i_dc
                    if i2 >= 0:
                        b[i2] += i_dc

                elif ct in (ComponentType.MEMRISTOR, ComponentType.PCM, ComponentType.MTJ, ComponentType.FTJ, ComponentType.DIODE):
                    g, i_off = self._device_conductance_and_current(
                        comp, x[:n_nodes], node_idx, dev_state, vt_jitter_map
                    )
                    i1, i2 = _idx(comp.nodes[0]), _idx(comp.nodes[1])
                    if i1 >= 0:
                        A[i1, i1] += g
                        b[i1] -= i_off
                    if i2 >= 0:
                        A[i2, i2] += g
                        b[i2] += i_off
                    if i1 >= 0 and i2 >= 0:
                        A[i1, i2] -= g
                        A[i2, i1] -= g

                elif ct in fet_types:
                    id_n, ig_n, is_n = _idx(comp.nodes[0]), _idx(comp.nodes[1]), _idx(comp.nodes[2])
                    vd = x[id_n] if id_n >= 0 else 0.0
                    vg = x[ig_n] if ig_n >= 0 else 0.0
                    vs = x[is_n] if is_n >= 0 else 0.0
                    jit = (vt_jitter_map or {}).get(comp.name, 0.0)
                    ids, gm, gds = self._fet_ids_gm_gds(comp, vd, vg, vs, dev_state, jit)
                    i_eq = ids - gm * (vg - vs) - gds * (vd - vs)
                    # Stamp into D and S rows
                    if id_n >= 0:
                        A[id_n, id_n] += gds
                        if ig_n >= 0:
                            A[id_n, ig_n] += gm
                        if is_n >= 0:
                            A[id_n, is_n] -= (gm + gds)
                        b[id_n] -= i_eq
                    if is_n >= 0:
                        A[is_n, is_n] += (gm + gds)
                        if id_n >= 0:
                            A[is_n, id_n] -= gds
                        if ig_n >= 0:
                            A[is_n, ig_n] -= gm
                        b[is_n] += i_eq

            # Stamp voltage sources
            for k, vs_comp in enumerate(vsources):
                row = n_nodes + k
                i1, i2 = _idx(vs_comp.nodes[0]), _idx(vs_comp.nodes[1])
                if i1 >= 0:
                    A[i1, row] += 1.0
                    A[row, i1] += 1.0
                if i2 >= 0:
                    A[i2, row] -= 1.0
                    A[row, i2] -= 1.0
                b[row] = self._eval_vsource(vs_comp, t)

            # Stamp inductors
            for k, ind_comp in enumerate(inductors):
                row = n_nodes + n_vs + k
                i1, i2 = _idx(ind_comp.nodes[0]), _idx(ind_comp.nodes[1])
                if i1 >= 0:
                    A[i1, row] += 1.0
                    A[row, i1] += 1.0
                if i2 >= 0:
                    A[i2, row] -= 1.0
                    A[row, i2] -= 1.0
                if dt is not None and dt > 0:
                    l_eq = ind_comp.params["inductance"] / dt
                    A[row, row] -= l_eq
                    b[row] = -l_eq * i_ind_prev.get(ind_comp.name, 0.0)
                else:
                    # DC short circuit
                    A[row, row] -= 1e-6

            # Solve linear system (use GPU backend for larger matrices >= 64)
            if backend is not None and backend.is_gpu and dim >= 64:
                xp = backend.xp
                x_new = backend.to_numpy(xp.linalg.solve(backend.to_device(A), backend.to_device(b)))
            else:
                x_new = np.linalg.solve(A, b)

            diff = float(np.max(np.abs(x_new - x)))
            # Damped update for smooth NR convergence
            if diff > 0.5:
                x = 0.5 * x + 0.5 * x_new
            else:
                x = x_new
            if diff < tol:
                break

        v_map = {name: float(x[i]) for i, name in enumerate(node_names)}
        v_map["0"] = 0.0
        vs_currents = {vs.name: float(x[n_nodes + k]) for k, vs in enumerate(vsources)}
        ind_currents = {ind.name: float(x[n_nodes + n_vs + k]) for k, ind in enumerate(inductors)}
        return x[:n_nodes], v_map, {"vs": vs_currents, "ind": ind_currents}

    def _init_device_states(self) -> dict[str, dict[str, float]]:
        states: dict[str, dict[str, float]] = {}
        for comp in self.components:
            ct = comp.comp_type
            if ct == ComponentType.MEMRISTOR:
                states[comp.name] = {"x": float(comp.params.get("x", 0.5))}
            elif ct == ComponentType.PCM:
                states[comp.name] = {"x": float(comp.params.get("x", 0.5))}
            elif ct == ComponentType.FEFET:
                states[comp.name] = {"pol": float(comp.params.get("polarization", 0.0))}
            elif ct == ComponentType.MTJ:
                states[comp.name] = {"mz": float(comp.params.get("mz", 1.0))}
            elif ct == ComponentType.FTJ:
                states[comp.name] = {"pol": float(comp.params.get("polarization", 0.0))}
        return states

    def _update_device_states(
        self,
        dt: float,
        v_map: dict[str, float],
        dev_state: dict[str, dict[str, float]],
    ) -> None:
        for comp in self.components:
            ct = comp.comp_type
            if ct == ComponentType.MEMRISTOR:
                v = v_map.get(comp.nodes[0], 0.0) - v_map.get(comp.nodes[1], 0.0)
                v_th = comp.params.get("v_th", 0.4)
                tau = max(comp.params.get("tau", 1e-9), 1e-13)
                if abs(v) > v_th:
                    dx = ((abs(v) - v_th) * np.sign(v) / tau) * dt
                    dev_state[comp.name]["x"] = float(np.clip(dev_state[comp.name]["x"] + dx, 0.0, 1.0))
            elif ct == ComponentType.FEFET:
                vg = v_map.get(comp.nodes[1], 0.0) - v_map.get(comp.nodes[2], 0.0)
                v_c = comp.params.get("v_coercive", 0.7)
                if abs(vg) > v_c:
                    target = 1.0 if vg > 0 else -1.0
                    tau = 2e-9
                    pol = dev_state[comp.name]["pol"]
                    dev_state[comp.name]["pol"] = float(np.clip(pol + (target - pol) * min(1.0, dt / tau), -1.0, 1.0))
            elif ct == ComponentType.MTJ:
                v = v_map.get(comp.nodes[0], 0.0) - v_map.get(comp.nodes[1], 0.0)
                r_p = comp.params.get("r_p", 2e3)
                i_dev = v / max(r_p, 1.0)
                i_c = comp.params.get("i_c", 50e-6)
                if abs(i_dev) > i_c:
                    target = -1.0 if i_dev > 0 else 1.0
                    mz = dev_state[comp.name]["mz"]
                    dev_state[comp.name]["mz"] = float(np.clip(mz + (target - mz) * min(1.0, dt / 2e-9), -1.0, 1.0))

    def simulate_dc(self, device: str = "auto") -> dict[str, Any]:
        """Compute DC operating point of the custom circuit."""
        backend = get_backend(device=device)
        node_names = self.get_node_names()
        dev_state = self._init_device_states()
        _, v_map, currents = self._solve_mna_step(
            t=0.0,
            dt=None,
            v_prev=np.zeros(len(node_names)),
            i_ind_prev={},
            dev_state=dev_state,
            backend=backend,
        )
        power_w = sum(
            abs(self._eval_vsource(vs, 0.0) * currents["vs"].get(vs.name, 0.0))
            for vs in self.components if vs.comp_type == ComponentType.V_SOURCE
        )
        return {
            "circuit_name": self.name,
            "node_nm": self.node_nm.value,
            "architecture": self.process_node.architecture.value,
            "corner": self.process_node.corner.value,
            "node_voltages": v_map,
            "vsource_currents": currents["vs"],
            "static_power_w": float(power_w),
        }

    def simulate_transient(
        self,
        t_stop: float = 2e-9,
        dt: float = 2e-11,
        vt_jitter_map: dict[str, float] | None = None,
        device: str = "auto",
    ) -> dict[str, Any]:
        """Run transient MNA simulation and extract waveforms + PPA metrics."""
        backend = get_backend(device=device)
        node_names = self.get_node_names()
        n_steps = max(2, int(np.ceil(t_stop / dt)))
        times = np.linspace(0.0, t_stop, n_steps)

        dev_state = self._init_device_states()
        v_prev = np.zeros(len(node_names), dtype=np.float64)
        # Initialize DC operating point at t=0
        v_prev, v_map0, cur0 = self._solve_mna_step(
            t=0.0, dt=None, v_prev=v_prev, i_ind_prev={}, dev_state=dev_state,
            vt_jitter_map=vt_jitter_map, backend=backend,
        )
        i_ind_prev = cur0["ind"]

        waveforms: dict[str, list[float]] = {n: [] for n in node_names}
        inst_power: list[float] = []

        for t in times:
            v_prev, v_map, currents = self._solve_mna_step(
                t=float(t), dt=dt, v_prev=v_prev, i_ind_prev=i_ind_prev,
                dev_state=dev_state, vt_jitter_map=vt_jitter_map, backend=backend,
            )
            i_ind_prev = currents["ind"]
            self._update_device_states(dt, v_map, dev_state)

            for n in node_names:
                waveforms[n].append(v_map[n])
            p_t = sum(
                abs(self._eval_vsource(vs, float(t)) * currents["vs"].get(vs.name, 0.0))
                for vs in self.components if vs.comp_type == ComponentType.V_SOURCE
            )
            inst_power.append(p_t)

        avg_power = float(np.mean(inst_power))
        peak_power = float(np.max(inst_power))
        total_energy = float(np.trapezoid(inst_power, times) if hasattr(np, "trapezoid") else np.trapz(inst_power, times))

        # Estimate propagation delay if 'in' and 'out' exist
        delay_ps = None
        if "in" in waveforms and "out" in waveforms:
            vdd = self.process_node.vdd.value
            v_in = np.asarray(waveforms["in"])
            v_out = np.asarray(waveforms["out"])
            idx_in = np.where(v_in >= 0.5 * vdd)[0]
            idx_out = np.where(np.abs(v_out - v_out[0]) >= 0.5 * max(abs(np.max(v_out) - np.min(v_out)), 0.1 * vdd))[0]
            if len(idx_in) > 0 and len(idx_out) > 0:
                delay_ps = float(abs(times[idx_out[0]] - times[idx_in[0]]) * 1e12)

        return {
            "circuit_name": self.name,
            "node_nm": self.node_nm.value,
            "architecture": self.process_node.architecture.value,
            "corner": self.process_node.corner.value,
            "vdd_v": self.process_node.vdd.value,
            "times_s": times.tolist(),
            "waveforms": waveforms,
            "avg_power_w": avg_power,
            "peak_power_w": peak_power,
            "total_energy_j": total_energy,
            "delay_ps": delay_ps,
            "final_device_states": dev_state,
        }

    def simulate_monte_carlo(
        self,
        n_runs: int = 50,
        t_stop: float = 1e-9,
        dt: float = 2e-11,
        seed: int = 42,
        device: str = "auto",
    ) -> dict[str, Any]:
        """Run Monte Carlo Pelgrom Vt variability analysis at the current nm node."""
        rng = np.random.default_rng(seed)
        sigma_vt_v = self.process_node.sigma_vt_mv.value * 1e-3
        fet_types = {
            ComponentType.NMOS, ComponentType.PMOS, ComponentType.FEFET,
            ComponentType.NCFET, ComponentType.TFET, ComponentType.TMD_FET,
        }
        fet_names = [c.name for c in self.components if c.comp_type in fet_types]

        powers: list[float] = []
        energies: list[float] = []
        delays: list[float] = []

        for _ in range(n_runs):
            jit = {fn: float(rng.normal(0.0, sigma_vt_v)) for fn in fet_names}
            res = self.simulate_transient(t_stop=t_stop, dt=dt, vt_jitter_map=jit, device=device)
            powers.append(res["avg_power_w"])
            energies.append(res["total_energy_j"])
            if res["delay_ps"] is not None:
                delays.append(res["delay_ps"])

        return {
            "circuit_name": self.name,
            "node_nm": self.node_nm.value,
            "n_runs": n_runs,
            "sigma_vt_mv": self.process_node.sigma_vt_mv.value,
            "power_mean_w": float(np.mean(powers)),
            "power_std_w": float(np.std(powers)),
            "energy_mean_j": float(np.mean(energies)),
            "energy_std_j": float(np.std(energies)),
            "delay_mean_ps": float(np.mean(delays)) if delays else None,
            "delay_std_ps": float(np.std(delays)) if delays else None,
        }

    def compare_across_nodes(
        self,
        nodes_nm: list[float],
        t_stop: float = 1e-9,
        dt: float = 2e-11,
        corner: str = "TT",
        device: str = "auto",
    ) -> dict[str, Any]:
        """Simulate this custom circuit across multiple nm nodes (e.g. [130, 65, 28, 7, 3, 2])."""
        orig_nm = self.node_nm.value
        orig_corner = self.process_node.corner
        results: list[dict[str, Any]] = []

        try:
            for nm in nodes_nm:
                self.set_process_node(nm, corner=corner)
                # Update pulsed/dc Vdd sources if named Vdd
                for comp in self.components:
                    if comp.comp_type == ComponentType.V_SOURCE and comp.name.lower().startswith("vdd"):
                        comp.params["dc"] = self.process_node.vdd.value
                    elif comp.comp_type == ComponentType.V_SOURCE and comp.params.get("waveform") == "pulse":
                        comp.params["v_high"] = self.process_node.vdd.value
                sim = self.simulate_transient(t_stop=t_stop, dt=dt, device=device)
                results.append({
                    "node_nm": float(nm),
                    "architecture": sim["architecture"],
                    "vdd_v": sim["vdd_v"],
                    "avg_power_w": sim["avg_power_w"],
                    "total_energy_j": sim["total_energy_j"],
                    "delay_ps": sim["delay_ps"],
                })
        finally:
            self.set_process_node(orig_nm, corner=orig_corner)

        return {"circuit_name": self.name, "corner": corner, "node_comparison": results}

    def to_spice_netlist(self) -> str:
        """Export circuit to a standard SPICE-compatible netlist string."""
        lines = [f"* ChipLab CustomCircuit: {self.name} ({self.node_nm.value:g} nm, {self.process_node.corner.value})"]
        for c in self.components:
            ct = c.comp_type
            ns = " ".join(c.nodes)
            if ct == ComponentType.RESISTOR:
                lines.append(f"R_{c.name} {ns} {c.params['resistance']:.6g}")
            elif ct == ComponentType.CAPACITOR:
                lines.append(f"C_{c.name} {ns} {c.params['capacitance']:.6g}")
            elif ct == ComponentType.INDUCTOR:
                lines.append(f"L_{c.name} {ns} {c.params['inductance']:.6g}")
            elif ct == ComponentType.V_SOURCE:
                lines.append(f"V_{c.name} {ns} DC {c.params.get('dc', 0.0):.6g}")
            elif ct == ComponentType.I_SOURCE:
                lines.append(f"I_{c.name} {ns} DC {c.params.get('dc', 0.0):.6g}")
            elif ct in (ComponentType.NMOS, ComponentType.PMOS):
                model = "nmos" if ct == ComponentType.NMOS else "pmos"
                w = c.width_nm or (4.0 * self.node_nm.value)
                l = c.length_nm or self.process_node.gate_length_nm.value
                b = "0" if ct == ComponentType.NMOS else "vdd"
                lines.append(f"M_{c.name} {ns} {b} {model} W={w:g}n L={l:g}n")
            else:
                lines.append(f"X_{c.name} {ns} {ct.value}")
        lines.append(".end")
        return "\n".join(lines)

    @classmethod
    def from_spice_netlist(cls, netlist: str, node_nm: float = 28.0, corner: str = "TT") -> CustomCircuit:
        """Safely parse basic SPICE netlist lines (R, C, L, V, I, M) into a CustomCircuit."""
        circ = cls(name="spice_import", node_nm=node_nm, corner=corner)
        for raw_line in netlist.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("*") or line.startswith("."):
                continue
            tokens = line.split()
            if len(tokens) < 4:
                continue
            prefix = tokens[0][0].upper()
            name = tokens[0]
            if prefix == "R":
                circ.add_resistor(name, tokens[1], tokens[2], float(tokens[3]))
            elif prefix == "C":
                circ.add_capacitor(name, tokens[1], tokens[2], float(tokens[3]))
            elif prefix == "L":
                circ.add_inductor(name, tokens[1], tokens[2], float(tokens[3]))
            elif prefix == "V":
                val = float(tokens[4]) if tokens[3].upper() == "DC" and len(tokens) >= 5 else float(tokens[3])
                circ.add_vsource(name, tokens[1], tokens[2], dc=val)
            elif prefix == "I":
                val = float(tokens[4]) if tokens[3].upper() == "DC" and len(tokens) >= 5 else float(tokens[3])
                circ.add_isource(name, tokens[1], tokens[2], dc=val)
            elif prefix == "M" and len(tokens) >= 6:
                model = tokens[5].lower()
                if "pmos" in model:
                    circ.add_pmos(name, tokens[1], tokens[2], tokens[3])
                else:
                    circ.add_nmos(name, tokens[1], tokens[2], tokens[3])
        return circ


# ---------------------------------------------------------------------------
# Pre-built Custom Circuit Templates (any nm node)
# ---------------------------------------------------------------------------

def build_cmos_inverter(node_nm: float = 28.0, corner: str = "TT", load_cap_ff: float = 5.0) -> CustomCircuit:
    """Build a CMOS inverter with pulsed input and load capacitance at any nm node."""
    circ = CustomCircuit(name=f"cmos_inverter_{node_nm:g}nm", node_nm=node_nm, corner=corner)
    vdd = circ.process_node.vdd.value
    circ.add_vsource("Vdd", "vdd", "0", dc=vdd)
    circ.add_vsource("Vin", "in", "0", waveform="pulse", v_low=0.0, v_high=vdd, period=1e-9, duty=0.5, delay=1e-10)
    circ.add_pmos("Mp", "out", "in", "vdd")
    circ.add_nmos("Mn", "out", "in", "0")
    circ.add_capacitor("Cload", "out", "0", load_cap_ff * 1e-15)
    return circ


def build_ring_oscillator(n_stages: int = 5, node_nm: float = 28.0, corner: str = "TT") -> CustomCircuit:
    """Build an N-stage CMOS ring oscillator (odd N >= 3) at any nm node."""
    if n_stages < 3 or n_stages % 2 == 0:
        raise ValueError("n_stages must be an odd integer >= 3")
    circ = CustomCircuit(name=f"rosc_{n_stages}_{node_nm:g}nm", node_nm=node_nm, corner=corner)
    vdd = circ.process_node.vdd.value
    circ.add_vsource("Vdd", "vdd", "0", dc=vdd)
    c_stage = max(0.2e-15, circ.process_node.wire_c_ff_per_um.value * 1e-15 * 2.0)
    for i in range(n_stages):
        n_in = f"n{i}"
        n_out = f"n{(i + 1) % n_stages}"
        circ.add_pmos(f"Mp{i}", n_out, n_in, "vdd")
        circ.add_nmos(f"Mn{i}", n_out, n_in, "0")
        circ.add_capacitor(f"C{i}", n_out, "0", c_stage)
    # Kickstart pulse on n0
    circ.add_isource("Ikick", "n0", "0", dc=1e-6)
    return circ


def build_6t_sram_cell(node_nm: float = 28.0, corner: str = "TT") -> CustomCircuit:
    """Build a 6T SRAM bitcell with access transistors and bitline caps at any nm node."""
    circ = CustomCircuit(name=f"sram_6t_{node_nm:g}nm", node_nm=node_nm, corner=corner)
    vdd = circ.process_node.vdd.value
    circ.add_vsource("Vdd", "vdd", "0", dc=vdd)
    circ.add_vsource("Vwl", "wl", "0", dc=vdd)
    circ.add_vsource("Vbl", "bl", "0", dc=vdd)
    circ.add_vsource("Vblb", "blb", "0", dc=0.0)
    # Cross-coupled inverters
    circ.add_pmos("Mpl", "q", "qb", "vdd")
    circ.add_nmos("Mnl", "q", "qb", "0")
    circ.add_pmos("Mpr", "qb", "q", "vdd")
    circ.add_nmos("Mnr", "qb", "q", "0")
    # Access transistors
    circ.add_nmos("Maxl", "q", "wl", "bl")
    circ.add_nmos("Maxr", "qb", "wl", "blb")
    circ.add_capacitor("Cq", "q", "0", 1e-15)
    circ.add_capacitor("Cqb", "qb", "0", 1e-15)
    return circ


def build_1t1r_cell(node_nm: float = 28.0, tech: str = "memristor", corner: str = "TT") -> CustomCircuit:
    """Build a 1T1R (1-Transistor + 1-Memristor/MTJ/PCM) memory cell at any nm node."""
    circ = CustomCircuit(name=f"1t1r_{tech}_{node_nm:g}nm", node_nm=node_nm, corner=corner)
    vdd = circ.process_node.vdd.value
    circ.add_vsource("Vwl", "wl", "0", dc=vdd)
    circ.add_vsource("Vbl", "bl", "0", waveform="pulse", v_low=0.1, v_high=vdd, period=1e-9, duty=0.5)
    circ.add_nmos("Macc", "mid", "wl", "0")
    t = tech.lower().strip()
    if t in ("mtj", "mram", "spintronic"):
        circ.add_mtj("Dev", "bl", "mid")
    elif t == "pcm":
        circ.add_component("Dev", ComponentType.PCM, ["bl", "mid"], {"r_set": 2e3, "r_reset": 2e5, "x": 0.5})
    else:
        circ.add_memristor("Dev", "bl", "mid")
    circ.add_wire_segment("bl_wire", "mid", "sl", length_um=5.0)
    circ.add_resistor("Rsl", "sl", "0", 10.0)
    return circ


def build_hybrid_crossbar_column(
    n_rows: int = 8,
    node_nm: float = 28.0,
    tech: str = "memristor",
    corner: str = "TT",
) -> CustomCircuit:
    """Build an N-row hybrid crossbar column with nm-scaled wire RC parasitics."""
    circ = CustomCircuit(name=f"xbar_{n_rows}x1_{tech}_{node_nm:g}nm", node_nm=node_nm, corner=corner)
    vdd = circ.process_node.vdd.value
    pitch_um = max(0.05, circ.process_node.metal1_pitch_nm.value * 2e-3)
    prev_bl = "bl_0"
    for r in range(n_rows):
        in_node = f"wl_{r}"
        circ.add_vsource(f"Vwl_{r}", in_node, "0", dc=0.25 * vdd * ((r % 3) + 1) / 3.0)
        cur_bl = f"bl_{r}"
        if r > 0:
            circ.add_wire_segment(f"w_{r}", prev_bl, cur_bl, length_um=pitch_um)
        if tech.lower().strip() in ("mtj", "mram"):
            circ.add_mtj(f"cell_{r}", in_node, cur_bl, mz=1.0 if r % 2 == 0 else -1.0)
        else:
            circ.add_memristor(f"cell_{r}", in_node, cur_bl, state_x=(r + 1) / (n_rows + 1))
        prev_bl = cur_bl
    circ.add_resistor("Rsense", prev_bl, "0", 50.0)
    return circ


__all__ = [
    "ComponentType",
    "CircuitComponent",
    "CustomCircuit",
    "build_cmos_inverter",
    "build_ring_oscillator",
    "build_6t_sram_cell",
    "build_1t1r_cell",
    "build_hybrid_crossbar_column",
]
