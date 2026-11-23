"""Zero-token digital twin: fast deterministic PPA model + multi-objective search + Monte Carlo.

No LLM in the loop: an MCP caller sends one config, the twin evaluates thousands of design
points locally and returns a compact summary (top-K Pareto designs, sensitivities, yield).

Model (per design point, vectorized over a batch with numpy; pure-Python scalar fallback):
- Area: RISC-V core + vector unit (DLEN) + VRF (VLEN) + L1/L2 per core + shared LLC + UCIe
  PHY modules, logic scaled by node density, SRAM from bitcell area; divided by placement
  utilization; black-box macros added with their keepout halo.
- Timing: critical path = logic depth x FO4(node, Vdd) via the alpha-power law + repeated
  wire RC over a length that grows with core size and routing congestion (utilization).
  Slack = clock period - path; black-box pin timing and macro fmax add their own slacks.
- Performance: per-core CPI from base CPI + cache-miss penalties (power-law miss curves,
  DRAM latency in cycles) x a memory-stall overlap factor; vector speedup from DLEN/VLEN with
  Amdahl's law; multi-core Amdahl plus a DRAM-bandwidth roofline cap. Unit: GIPS-equivalent.
- Power: activity x Ceff x V^2 x f + leakage(node, Vdd, T) with a thermal fixed point
  T = ambient + P x theta_JA.
- Cost: dies per wafer, negative-binomial yield and wafer price per node (cost.prices).
Unknown inputs (provenance 'estimated'/'assumed') are multiplicative factors that Monte
Carlo perturbs; calibrated factors become 'measured' and are held fixed.
"""

from __future__ import annotations

import math
import random
import time
import uuid
from typing import Any

from chiplab.cost import prices
from chiplab.library.pdk_library import PDK_LADDER
from chiplab.models.onchip_memory import TECHS, mem_core, secded_check_bits
from chiplab.schemas.common import Provenance
from chiplab.schemas.digital_twin import (
    BlackBoxMacro,
    DigitalTwinConfig,
    DigitalTwinSummary,
    ParetoPoint,
    RobustnessReport,
    SearchParameter,
    SensitivityEntry,
    TwinMemory,
    TwinWorkload,
    UncertaintyPrior,
)

try:  # numpy is a project dependency, but the twin still runs without it
    import numpy as _np
except ImportError:  # pragma: no cover - exercised via force_pure_python
    _np = None

MAX_TRACE_ROWS = 20_000

# --- Node table ---------------------------------------------------------------------------
# logic area vs N7, FO4 at Vnom (ps), Vnom, Vth, repeated-wire delay (ps/mm),
# switched-cap scale vs N7, leakage mW/mm2 at Vnom/25C. Density ratios follow TSMC's
# published logic-density claims; the rest are estimates (see provenance below).
TSMC_NODES = ["N16", "N7", "N5", "N3"]
_NODE: dict[str, list[Any]] = {
    "logic_area": [3.15, 1.00, 0.66, 0.46],
    "fo4_ps": [10.0, 7.0, 6.0, 5.3],
    "vnom": [0.80, 0.75, 0.75, 0.75],
    "vth": [0.35, 0.32, 0.30, 0.29],
    "wire_ps_mm": [60.0, 90.0, 110.0, 130.0],
    "cap_scale": [1.8, 1.0, 0.8, 0.68],
    "leak_mw_mm2": [5.0, 10.0, 12.0, 14.0],
    "cost_node": list(TSMC_NODES),
    "bitcell_um2": [None, None, None, None],        # from the cost reference (TSMC HD 6T)
    "retention_us": [40.0, 15.0, 10.0, 8.0],          # 2T gain cell at 25 C (estimated)
    "margin_pct": [15.0, 15.0, 20.0, 20.0],           # Monte Carlo accuracy margin
}
# PDK-ladder rows, derived from chiplab.library.pdk_library profiles (N3 reuses the TSMC row):
# logic area = 10k kgate/mm2 / density (ASAP7 = 1.0 = N7), wire ps/mm = 25 + 2000 / M1 pitch,
# switched-cap scale = (E_gate / Vdd^2) / 0.3 fF, leakage mW/mm2 = nW/gate x kgate/mm2 / 1000.
_LADDER_ROWS = {"GF180": "gf180", "SKY130": "sky130", "IHP_SG13G2": "ihp_sg13g2", "N65": "n65",
                "N28": "n28", "FREEPDK15": "freepdk15", "ASAP7": "asap7", "N2": "n2"}
for _name, _key in _LADDER_ROWS.items():
    _pr = PDK_LADDER[_key]
    _kg, _v = _pr.scalar("gate_density_kgates_mm2"), _pr.scalar("nominal_vdd_v")
    _NODE["logic_area"].append(10_000.0 / _kg)
    _NODE["fo4_ps"].append(_pr.scalar("fo4_delay_ps"))
    _NODE["vnom"].append(_v)
    _NODE["vth"].append(_pr.scalar("vth_v"))
    _NODE["wire_ps_mm"].append(25.0 + 2000.0 / _pr.scalar("metal1_pitch_nm"))
    _NODE["cap_scale"].append(_pr.scalar("dynamic_energy_fj_per_gate") / _v**2 / 0.3)
    _NODE["leak_mw_mm2"].append(_pr.scalar("leakage_nw_per_gate") * _kg / 1000.0)
    _NODE["cost_node"].append(_pr.cost_node)
    _NODE["bitcell_um2"].append(_pr.scalar("sram_bitcell_um2"))
    _NODE["retention_us"].append(_pr.scalar("gain_cell_retention_us_25c"))
    _NODE["margin_pct"].append(_pr.accuracy_margin_pct)
NODE_NAMES = TSMC_NODES + list(_LADDER_ROWS)
NODE_MARGIN_PCT = dict(zip(NODE_NAMES, _NODE["margin_pct"]))
TECH_FACTORS = ("gate_delay", "wire_rc", "leakage", "cdyn", "logic_area")
INTERLEAVES = ("low_order", "xor", "block")
ALPHA = 1.3
LOGIC_DEPTH_FO4 = 40.0
CORE_AREA_N7 = 0.30          # mm2, RV64GC application core + FPU
VEC_AREA_PER_128 = 0.25      # mm2 per 128-bit DLEN slice at N7
CORE_CEFF_NF = 0.36          # nF switched per core at N7 (activity applied separately)
VEC_CEFF_PER_128_NF = 0.30
UCIE_X16_AREA = 0.25         # mm2 per x16 module (estimated)
UCIE_X16_W = 0.26            # W per x16 module at 32 GT/s, 0.5 pJ/bit
DRAM_LAT_NS = 90.0
DRAM_BW_GBS = 100.0

# Unknown parameters and their default provenance (what Monte Carlo perturbs).
UNKNOWNS: dict[str, tuple[Provenance, str]] = {
    "gate_delay": (Provenance.ESTIMATED, "FO4 delay vs. node/Vdd"),
    "wire_rc": (Provenance.ASSUMED, "uncalibrated N5/N3 wire parasitics"),
    "leakage": (Provenance.ESTIMATED, "leakage density"),
    "cdyn": (Provenance.ESTIMATED, "switched capacitance"),
    "logic_area": (Provenance.ESTIMATED, "logic area per core"),
    "mem_stall": (Provenance.ASSUMED, "memory-stall overlap (workload)"),
    "dram_latency": (Provenance.ESTIMATED, "loaded DRAM latency"),
    "theta_ja": (Provenance.ASSUMED, "package/heatsink thermal resistance"),
    "wafer_price": (Provenance.ESTIMATED, "foundry wafer price"),
    "defect_density": (Provenance.ESTIMATED, "defect density D0"),
}
MEMORY_UNKNOWNS: dict[str, tuple[Provenance, str, float]] = {
    "retention": (Provenance.ASSUMED, "gain-cell retention time", 25.0),
    "mem_traffic": (Provenance.ASSUMED, "offered on-chip memory load (workload)", 20.0),
}


def default_priors(calibrated: set[str] | None = None,
                   memory: bool = False) -> list[UncertaintyPrior]:
    """Priors for the unknowns. Technology factors are widened per design to the PDK's
    accuracy margin in ``run_digital_twin`` (calibrated factors stay fixed)."""
    calibrated = calibrated or set()
    items = [(n, prov, desc, 20.0 if prov == Provenance.ASSUMED else 15.0)
             for n, (prov, desc) in UNKNOWNS.items()]
    if memory:
        items += [(n, prov, desc, sp) for n, (prov, desc, sp) in MEMORY_UNKNOWNS.items()]
    out = []
    for name, prov, desc, spread in items:
        if name in calibrated:
            prov = Provenance.MEASURED
        out.append(UncertaintyPrior(name=name, provenance=prov, spread_pct=spread,
                                    description=desc))
    return out


def node_priors(priors: list[UncertaintyPrior], node: Any) -> list[UncertaintyPrior]:
    """Widen technology-factor spreads to the node's PDK accuracy margin."""
    margin = NODE_MARGIN_PCT[NODE_NAMES[_node_index(node)]]
    return [p.model_copy(update={"spread_pct": max(p.spread_pct, margin)})
            if p.name in TECH_FACTORS
            and p.provenance not in (Provenance.PUBLISHED, Provenance.MEASURED) else p
            for p in priors]


# --- Array backend shim ---------------------------------------------------------------------

class _Scalar:
    """Elementwise ops on Python floats, mirroring the numpy calls the model uses."""

    exp = staticmethod(math.exp)
    sqrt = staticmethod(math.sqrt)
    log2 = staticmethod(math.log2)

    @staticmethod
    def minimum(a: float, b: float) -> float:
        return a if a < b else b

    @staticmethod
    def maximum(a: float, b: float) -> float:
        return a if a > b else b

    @staticmethod
    def clip(a: float, lo: float, hi: float) -> float:
        return lo if a < lo else hi if a > hi else a

    @staticmethod
    def take(table: list[float], idx: Any) -> float:
        return table[int(idx)]

    @staticmethod
    def floor(a: float) -> float:
        return float(math.floor(a))


class _Numpy:
    def __init__(self, np: Any) -> None:
        self.np = np
        self.exp, self.sqrt, self.log2 = np.exp, np.sqrt, np.log2
        self.minimum, self.maximum, self.clip, self.floor = (np.minimum, np.maximum, np.clip,
                                                             np.floor)

    def take(self, table: list[float], idx: Any) -> Any:
        return self.np.asarray(table, dtype=float)[self.np.asarray(idx, dtype=int)]


def _backend(force_pure_python: bool = False) -> Any:
    return _Scalar() if force_pure_python or _np is None else _Numpy(_np)


# --- PPA model ------------------------------------------------------------------------------

def _wafer_tables(calib: dict[str, float]) -> tuple[list[float], list[float]]:
    wafer = [prices.value(f"wafer.tsmc.{n}") for n in _NODE["cost_node"]]
    d0 = [prices.value(f"defect_density.tsmc.{n}") for n in _NODE["cost_node"]]
    return wafer, d0


def _bitcells() -> list[float]:
    return [b if b is not None else prices.value(f"sram_bitcell.tsmc.{n}")
            for b, n in zip(_NODE["bitcell_um2"], _NODE["cost_node"])]


def _sram_mm2_per_mb() -> list[float]:
    eff = prices.value("sram.array_efficiency")
    return [8 * 2**20 * b / 1e6 / eff for b in _bitcells()]


class TwinModel:
    """Deterministic analytical PPA model. ``evaluate`` takes arrays (numpy) or floats."""

    def __init__(self, workload: TwinWorkload | None = None,
                 calibration: dict[str, float] | None = None,
                 blackboxes: list[BlackBoxMacro] | None = None,
                 ambient_c: float = 25.0, theta_ja: float | None = None,
                 memory: TwinMemory | None = None) -> None:
        self.wl = workload or TwinWorkload()
        self.mem = memory
        self.bitcell = _bitcells()
        self.calib = dict(calibration or {})
        self.blackboxes = list(blackboxes or [])
        self.ambient = ambient_c
        self.theta_fixed = theta_ja
        self.wafer, self.d0 = _wafer_tables(self.calib)
        self.sram_per_mb = _sram_mm2_per_mb()
        self.bb_area = sum(b.area_with_keepout_mm2 * b.count for b in self.blackboxes)
        self.bb_power_w = sum(b.power_mw * b.count for b in self.blackboxes) / 1000.0

    def evaluate(self, p: dict[str, Any], u: dict[str, Any] | None, xp: Any) -> dict[str, Any]:
        u = u or {}
        c = self.calib

        def f(name: str) -> Any:
            return c.get(name, 1.0) * u.get(name, 1.0)

        wl = self.wl
        node = p["node"]
        cores, vlen, dlen = p["cores"], p["vlen"], p["dlen"]
        l1, l2, llc = p["l1_kb"], p["l2_kb"], p["llc_mb"]
        vdd, freq, util, lanes = p["vdd"], p["freq_ghz"], p["utilization"], p["ucie_lanes"]

        # Area
        la = xp.take(_NODE["logic_area"], node) * f("logic_area")
        sram = xp.take(self.sram_per_mb, node)
        core = CORE_AREA_N7 * la + VEC_AREA_PER_128 * (dlen / 128.0) * la
        vrf = 2.0 * sram * (vlen * 32 / 8 / 2**20)            # 32 vregs, RF ~2x SRAM
        caches = sram * ((2 * l1 + l2) / 1024.0)
        tile = (core + vrf + caches) * 1.10                     # +10% uncore/NoC per tile
        ucie_mods = lanes / 16.0
        cell_area = cores * tile + sram * llc + ucie_mods * UCIE_X16_AREA
        area = cell_area / util + self.bb_area
        vn = xp.take(_NODE["vnom"], node)

        # Timing (ps)
        vt = xp.take(_NODE["vth"], node)
        vscale = (vdd / (xp.maximum(vdd - vt, 0.02)) ** ALPHA) / (vn / (vn - vt) ** ALPHA)
        fo4 = xp.take(_NODE["fo4_ps"], node) * vscale * f("gate_delay")
        depth = LOGIC_DEPTH_FO4 + 2.0 * xp.log2(xp.maximum(dlen / 64.0, 1.0))
        congestion = 1.0 + 3.0 * xp.maximum(util - 0.70, 0.0)
        wire_len = 0.30 * xp.sqrt(tile) * congestion
        wire = xp.take(_NODE["wire_ps_mm"], node) * f("wire_rc") * wire_len
        path = depth * fo4 + wire
        period = 1000.0 / freq
        slack = period - path
        for b in self.blackboxes:
            slack = xp.minimum(slack, self._bb_slack(b, period, wire / xp.maximum(wire_len,
                                                                                   1e-6), xp))

        # Performance (GIPS-equivalent)
        ws = wl.working_set_mb
        l1_miss = xp.clip(0.05 * (32.0 / l1) ** 0.5, 0.005, 1.0)
        l2_miss = xp.clip(0.5 * ((ws * 1024.0 / xp.maximum(cores, 1.0)) / l2) ** 0.5 * 0.3,
                          0.01, 1.0)
        llc_miss = xp.clip((ws / xp.maximum(llc, 0.25)) ** 0.5 * 0.5, 0.02, 1.0)
        dram_cyc = DRAM_LAT_NS * f("dram_latency") * freq
        penalty = l1_miss * (12.0 + l2_miss * (40.0 + llc_miss * dram_cyc))
        cpi = wl.base_cpi + wl.mem_refs_per_inst * penalty * 0.5 * f("mem_stall")
        vec_speed = (dlen / 64.0) * (0.7 + 0.3 * xp.minimum(vlen / (2.0 * dlen), 1.0))
        per_core = freq / cpi / ((1 - wl.vector_fraction) + wl.vector_fraction / vec_speed)
        pf = wl.parallel_fraction
        speedup = 1.0 / ((1 - pf) + pf / cores)
        perf_cpu = per_core * speedup
        bytes_per_inst = wl.mem_refs_per_inst * l1_miss * l2_miss * llc_miss * 64.0
        perf = xp.minimum(perf_cpu, DRAM_BW_GBS / xp.maximum(bytes_per_inst, 1e-9))

        # Power (W) with thermal fixed point
        cap = xp.take(_NODE["cap_scale"], node) * f("cdyn")
        ceff = cores * (CORE_CEFF_NF + VEC_CEFF_PER_128_NF * dlen / 128.0
                        + 0.002 * 2 * l1 + 0.0005 * l2) + 0.02 * llc
        dyn = wl.activity_factor * ceff * cap * vdd * vdd * freq + ucie_mods * UCIE_X16_W * 0.5
        mem = None
        mem_w: Any = 0.0
        if self.mem is not None:
            mem = self._memory(p, f, xp, node, vdd, vn, fo4, freq, self.ambient + 30.0)
            area = area + 1.1 * mem["area_mm2"]          # macro + 10% halo
            mem_w = mem["power_w"]
        theta = (self.theta_fixed if self.theta_fixed is not None
                 else 0.4 + 60.0 / xp.maximum(area, 1.0)) * f("theta_ja")
        leak0 = xp.take(_NODE["leak_mw_mm2"], node) * f("leakage") * cell_area / 1000.0 \
            * xp.exp((vdd - vn) / 0.12)
        temp = self.ambient + (dyn + leak0 + mem_w) * theta
        for _ in range(3):
            leak = leak0 * 2.0 ** ((xp.minimum(temp, 150.0) - 25.0) / 30.0)
            power = dyn + leak + self.bb_power_w + mem_w
            temp = self.ambient + power * theta
        if mem is not None:  # re-evaluate retention / refresh at the operating temperature
            mem = self._memory(p, f, xp, node, vdd, vn, fo4, freq, temp)
            power = dyn + leak + self.bb_power_w + mem["power_w"]
            temp = self.ambient + power * theta
            slack = xp.minimum(slack, mem["slack_ps"])
            # Half of the memory-service shortfall shows up as lost throughput (overlap).
            perf = perf * (1.0 - 0.5 * (1.0 - mem["service"]))

        # Cost (USD): die + package + test
        side = xp.sqrt(area) + 0.1
        a = side * side
        d = 294.0
        dpw = xp.maximum(math.pi * (d / 2) ** 2 / a - math.pi * d / xp.sqrt(2 * a), 1.0)
        y = (1.0 + (area / 100.0) * xp.take(self.d0, node) * f("defect_density") / 3.0) ** -3.0
        wafer = xp.take(self.wafer, node) * f("wafer_price")
        cost = wafer * 1.03 / (dpw * y) + 5.0 + 0.01 * area

        out = {"perf": perf, "power_w": power, "area_mm2": area, "cost_usd": cost,
               "perf_per_watt": perf / power, "energy_pj_op": power / perf * 1000.0,
               "slack_ps": slack, "temp_c": temp, "fmax_ghz": 1000.0 / path,
               "yield": y, "leakage_w": leak, "dynamic_w": dyn}
        if mem is not None:
            out.update({"mem_area_mm2": mem["area_mm2"], "mem_power_w": mem["power_w"],
                        "refresh_overhead": mem["refresh_overhead"],
                        "retention_us": mem["retention_us"],
                        "mem_conflict_rate": mem["conflict_rate"],
                        "mem_bandwidth_gbs": mem["bandwidth_gbs"],
                        "pim_gmacs": mem["pim_gmacs"]})
        return out

    def _memory(self, p: dict[str, Any], f: Any, xp: Any, node: Any, vdd: Any, vn: Any,
                fo4: Any, freq: Any, temp: Any) -> dict[str, Any]:
        """On-chip memory subsystem (chiplab.models.onchip_memory.mem_core) at this node."""
        m = self.mem
        assert m is not None
        tech, banks, il, lanes = p["mem_tech"], p["mem_banks"], p["mem_interleave"], \
            p["pim_lanes"]
        R = m.num_requesters
        rows = xp.maximum(m.capacity_kib * 8192.0 / m.word_bits / banks / 4.0, 1.0)
        # Banks the traffic spreads over: random part uses all banks, strided part depends on
        # interleaving (low-order: banks / gcd(stride, banks); XOR folds power-of-two strides;
        # block: one bank per requester stream).
        p2 = float(m.stride & -m.stride)
        eff_low = banks / xp.minimum(banks, p2)
        eff_xor = banks if m.stride & (m.stride - 1) == 0 else eff_low
        eff_blk = xp.minimum(banks, float(R))
        w0 = xp.maximum(1.0 - il, 0.0)
        w2 = xp.maximum(il - 1.0, 0.0)
        eff_pat = w0 * eff_low + (1.0 - w0 - w2) * eff_xor + w2 * eff_blk
        sf = m.strided_fraction
        eff = 1.0 / ((1.0 - sf) / banks + sf / eff_pat)
        kg = 10_000.0 / xp.take(_NODE["logic_area"], node)
        pdk = {"gate_density_kgates_mm2": kg,
               "dynamic_energy_fj_per_gate": xp.take(_NODE["cap_scale"], node) * f("cdyn")
               * 0.3 * vdd * vdd,
               "leakage_nw_per_gate": xp.take(_NODE["leak_mw_mm2"], node) * f("leakage")
               * 1000.0 / kg * xp.exp((vdd - vn) / 0.12),
               "fo4_delay_ps": fo4,
               "sram_bitcell_um2": xp.take(self.bitcell, node),
               "gain_cell_retention_us_25c": xp.take(_NODE["retention_us"], node)}
        traffic = m.requests_per_cycle * f("mem_traffic")
        ecc = 1.0 if m.ecc == "secded" else 0.0
        r = mem_core(xp, tech, m.capacity_kib * 8192.0, banks, float(m.word_bits), rows,
                     traffic, eff, 1.0, freq * 1000.0, temp, pdk, lanes, float(m.mac_bits),
                     f("retention"), m.refresh_margin, ecc, R)
        code_bits = m.word_bits + (secded_check_bits(m.word_bits) if ecc else 0)
        e_pj = 0.7 * r["read_energy_pj_per_bit"] + 0.3 * r["write_energy_pj_per_bit"]
        dyn_w = r["served_per_cycle"] * freq * 1e9 * code_bits * e_pj * 1e-12
        pim_w = r["pim_gmacs"] * 1e9 * m.pim_fraction * r["pim_energy_pj_per_mac"] * 1e-12
        r["power_w"] = dyn_w + pim_w + r["refresh_power_w"] + r["leakage_w"]
        r["service"] = xp.minimum(r["served_per_cycle"] / traffic, 1.0)
        return r

    @staticmethod
    def _bb_slack(b: BlackBoxMacro, period: Any, wire_ps_mm: Any, xp: Any) -> Any:
        """Interface slack of a black-box macro: pin timing + 0.2 mm route + 60 ps logic."""
        route = 0.2 * wire_ps_mm + 60.0
        slack = period * 1.0 + 1e9  # start huge, reduce per constraint
        for pin in b.pins:
            if pin.direction in ("input", "inout") and pin.setup_ps is not None:
                slack = xp.minimum(slack, period - route - 40.0 - pin.setup_ps)
            if pin.direction in ("output", "inout") and pin.clk_to_q_ps is not None:
                slack = xp.minimum(slack, period - pin.clk_to_q_ps - route - 20.0)
        if b.max_freq_ghz:
            slack = xp.minimum(slack, period - 1000.0 / b.max_freq_ghz)
        return slack


# --- Encoding -------------------------------------------------------------------------------

def _decode_one(sp: SearchParameter, x: float) -> Any:
    x = min(max(x, 0.0), 1.0)
    if sp.kind == "choice":
        assert sp.choices is not None
        return sp.choices[min(int(x * len(sp.choices)), len(sp.choices) - 1)]
    lo, hi = float(sp.low), float(sp.high)  # type: ignore[arg-type]
    v = math.exp(math.log(lo) + x * (math.log(hi) - math.log(lo))) if sp.log_scale \
        else lo + x * (hi - lo)
    return int(round(v)) if sp.kind == "int" else v


def _node_index(v: Any) -> int:
    name = str(v).upper()
    if name not in NODE_NAMES:
        raise ValueError(f"Twin node must be one of {NODE_NAMES} (PDK ladder keys such as "
                         f"'sky130' or 'asap7' also work), got {v!r}")
    return NODE_NAMES.index(name)


REQUIRED = ("cores", "vlen", "dlen", "l1_kb", "l2_kb", "llc_mb", "vdd", "freq_ghz", "node",
            "ucie_lanes", "utilization")
MEMORY_KNOBS = ("mem_tech", "mem_banks", "mem_interleave", "pim_lanes")
OPTIONAL = ("vdd_rel",) + MEMORY_KNOBS
DEFAULTS = {"cores": 8, "vlen": 256, "dlen": 128, "l1_kb": 32, "l2_kb": 512, "llc_mb": 8.0,
            "vdd": 0.75, "freq_ghz": 1.5, "node": "N7", "ucie_lanes": 0, "utilization": 0.7}
MEMORY_DEFAULTS = {"mem_tech": "sram6t", "mem_banks": 8, "mem_interleave": "xor",
                   "pim_lanes": 0}
MEMORY_METRICS = ("pim_gmacs", "mem_bandwidth_gbs", "refresh_overhead")


def _encode(k: str, v: Any) -> Any:
    if k == "node":
        return _node_index(v)
    if k == "mem_tech":
        if v not in TECHS:
            raise ValueError(f"mem_tech must be one of {TECHS}, got {v!r}")
        return TECHS.index(v)
    if k == "mem_interleave":
        if v not in INTERLEAVES:
            raise ValueError(f"mem_interleave must be one of {INTERLEAVES}, got {v!r}")
        return float(INTERLEAVES.index(v))
    return float(v)


def _finish_design(d: dict[str, Any], vdd_auto: bool) -> dict[str, Any]:
    """Derived parameters: DLEN <= VLEN, vdd from vdd_rel, ladder nodes at nominal Vdd."""
    if d["dlen"] > d["vlen"]:  # RVV requires DLEN <= VLEN
        d["dlen"] = d["vlen"]
    vnom = _NODE["vnom"][_node_index(d["node"])]
    if "vdd_rel" in d:
        d["vdd"] = round(float(d["vdd_rel"]) * vnom, 4)
    elif vdd_auto and str(d["node"]).upper() not in TSMC_NODES:
        d["vdd"] = vnom
    return d


class _Problem:
    def __init__(self, cfg: DigitalTwinConfig, force_pure_python: bool = False) -> None:
        self.cfg = cfg
        self.space = [s for s in cfg.search_space if s.name not in cfg.fixed]
        names = {s.name for s in self.space} | set(cfg.fixed)
        unknown = names - set(REQUIRED) - set(OPTIONAL)
        if unknown:
            raise ValueError(f"Unknown search parameters {sorted(unknown)}; "
                             f"supported: {list(REQUIRED + OPTIONAL)}")
        mem_named = names & set(MEMORY_KNOBS)
        needs_mem = {o.name for o in cfg.objectives} & set(MEMORY_METRICS)
        if (mem_named or needs_mem) and cfg.memory is None:
            raise ValueError(f"{sorted(mem_named | needs_mem)} need the on-chip memory model: "
                             "set config.memory (e.g. {'capacity_kib': 2048}).")
        self.fixed = {**DEFAULTS, **(MEMORY_DEFAULTS if cfg.memory else {}), **cfg.fixed}
        self.keys = REQUIRED + (MEMORY_KNOBS if cfg.memory else ())
        self.vdd_auto = "vdd" not in names and "vdd_rel" not in names
        self.xp = _backend(force_pure_python)
        self.numpy = isinstance(self.xp, _Numpy)
        self.model = TwinModel(cfg.workload, cfg.calibration, cfg.blackboxes, cfg.ambient_c,
                               cfg.theta_ja_c_per_w, cfg.memory)
        self.evaluated = 0

    def decode(self, x: list[float]) -> dict[str, Any]:
        d = dict(self.fixed)
        for sp, xi in zip(self.space, x):
            d[sp.name] = _decode_one(sp, xi)
        return _finish_design(d, self.vdd_auto)

    def _arrays(self, designs: list[dict[str, Any]]) -> dict[str, Any]:
        cols: dict[str, Any] = {}
        for k in self.keys:
            vals = [_encode(k, dd[k]) for dd in designs]
            cols[k] = _np.asarray(vals, dtype=int if k in ("node", "mem_tech") else float)
        return cols

    def metrics(self, designs: list[dict[str, Any]],
                factors: dict[str, Any] | None = None) -> list[dict[str, float]]:
        """Evaluate designs (optionally with per-design factor arrays aligned to designs)."""
        self.evaluated += len(designs)
        if self.numpy:
            out = self.model.evaluate(self._arrays(designs), factors, self.xp)
            keys = list(out)
            cols = [_np.broadcast_to(_np.asarray(out[k], dtype=float), (len(designs),))
                    for k in keys]
            return [dict(zip(keys, map(float, row))) for row in zip(*cols)]
        res = []
        for i, dd in enumerate(designs):
            p = {k: _encode(k, dd[k]) for k in self.keys}
            fi = {k: v[i] for k, v in (factors or {}).items()}
            res.append(self.model.evaluate(p, fi, self.xp))
        return res

    def objectives(self, m: dict[str, float]) -> list[float]:
        """Objective vector in minimization form."""
        return [-m[o.name] if o.goal == "max" else m[o.name] for o in self.cfg.objectives]

    def violation(self, m: dict[str, float]) -> float:
        c = self.cfg.constraints
        v = max(0.0, c.min_slack_ps - m["slack_ps"]) / 100.0
        v += max(0.0, m["temp_c"] - c.max_temp_c) / 10.0
        for lim, key in ((c.max_power_w, "power_w"), (c.max_area_mm2, "area_mm2"),
                         (c.max_cost_usd, "cost_usd")):
            if lim is not None:
                v += max(0.0, m[key] - lim) / lim
        if "refresh_overhead" in m:
            v += max(0.0, m["refresh_overhead"] - c.max_refresh_overhead) \
                / c.max_refresh_overhead
        return v


# --- NSGA-II --------------------------------------------------------------------------------

def _dominates(a: tuple[list[float], float], b: tuple[list[float], float]) -> bool:
    (fa, va), (fb, vb) = a, b
    if va == 0 and vb > 0:
        return True
    if va > 0 and vb > 0:
        return va < vb
    if va > 0:
        return False
    return all(x <= y for x, y in zip(fa, fb)) and any(x < y for x, y in zip(fa, fb))


def nondominated_sort(items: list[tuple[list[float], float]]) -> list[list[int]]:
    """Deb's fast non-dominated sort with constrained domination. Returns fronts of indices."""
    n = len(items)
    dominated: list[list[int]] = [[] for _ in range(n)]
    count = [0] * n
    fronts: list[list[int]] = [[]]
    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(items[i], items[j]):
                dominated[i].append(j)
                count[j] += 1
            elif _dominates(items[j], items[i]):
                dominated[j].append(i)
                count[i] += 1
        if count[i] == 0:
            fronts[0].append(i)
    # count[i] for i processed later may change; recompute front 0 robustly
    fronts[0] = [i for i in range(n) if count[i] == 0]
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in dominated[i]:
                count[j] -= 1
                if count[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding(front: list[int], objs: list[list[float]]) -> dict[int, float]:
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: math.inf for i in front}
    for m in range(len(objs[front[0]])):
        srt = sorted(front, key=lambda i: objs[i][m])
        lo, hi = objs[srt[0]][m], objs[srt[-1]][m]
        dist[srt[0]] = dist[srt[-1]] = math.inf
        if hi == lo:
            continue
        for a, i, b in zip(srt, srt[1:], srt[2:]):
            dist[i] += (objs[b][m] - objs[a][m]) / (hi - lo)
    return dist


def _sbx(rng: random.Random, a: list[float], b: list[float], eta: float = 15.0,
         prob: float = 0.9) -> tuple[list[float], list[float]]:
    c1, c2 = a[:], b[:]
    if rng.random() > prob:
        return c1, c2
    for i in range(len(a)):
        if rng.random() < 0.5 and abs(a[i] - b[i]) > 1e-12:
            u = rng.random()
            beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            c1[i] = 0.5 * ((1 + beta) * a[i] + (1 - beta) * b[i])
            c2[i] = 0.5 * ((1 - beta) * a[i] + (1 + beta) * b[i])
    return [min(max(x, 0.0), 1.0) for x in c1], [min(max(x, 0.0), 1.0) for x in c2]


def _mutate(rng: random.Random, x: list[float], eta: float = 20.0) -> list[float]:
    pm = 1.0 / max(len(x), 1)
    y = x[:]
    for i in range(len(y)):
        if rng.random() < pm:
            u = rng.random()
            delta = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            y[i] = min(max(y[i] + delta, 0.0), 1.0)
    return y


def latin_hypercube(n: int, d: int, rng: random.Random) -> list[list[float]]:
    cols = []
    for _ in range(d):
        strata = [(k + rng.random()) / n for k in range(n)]
        rng.shuffle(strata)
        cols.append(strata)
    return [list(row) for row in zip(*cols)] if d else [[] for _ in range(n)]


# --- Search driver --------------------------------------------------------------------------

class _Archive:
    def __init__(self) -> None:
        self.x: list[list[float]] = []
        self.designs: list[dict[str, Any]] = []
        self.metrics: list[dict[str, float]] = []
        self.objs: list[list[float]] = []
        self.viol: list[float] = []
        self.gen: list[int] = []

    def add(self, prob: _Problem, xs: list[list[float]], gen: int) -> list[int]:
        designs = [prob.decode(x) for x in xs]
        ms = prob.metrics(designs)
        start = len(self.x)
        for x, d, m in zip(xs, designs, ms):
            self.x.append(x)
            self.designs.append(d)
            self.metrics.append(m)
            self.objs.append(prob.objectives(m))
            self.viol.append(prob.violation(m))
            self.gen.append(gen)
        return list(range(start, len(self.x)))


def _nsga2(prob: _Problem, arc: _Archive, rng: random.Random) -> list[int]:
    cfg = prob.cfg
    n, d = cfg.population, len(prob.space)
    pop = arc.add(prob, latin_hypercube(n, d, rng), 0)
    for g in range(1, cfg.generations + 1):
        items = [(arc.objs[i], arc.viol[i]) for i in pop]
        fronts = nondominated_sort(items)
        rank = {pop[i]: r for r, fr in enumerate(fronts) for i in fr}
        crowd: dict[int, float] = {}
        for fr in fronts:
            crowd.update({pop[k]: v for k, v in
                          crowding(fr, [arc.objs[i] for i in pop]).items()})

        def pick() -> int:
            a, b = rng.choice(pop), rng.choice(pop)
            return a if (rank[a], -crowd[a]) <= (rank[b], -crowd[b]) else b

        kids: list[list[float]] = []
        while len(kids) < n:
            c1, c2 = _sbx(rng, arc.x[pick()], arc.x[pick()])
            kids += [_mutate(rng, c1), _mutate(rng, c2)]
        merged = pop + arc.add(prob, kids[:n], g)
        fronts = nondominated_sort([(arc.objs[i], arc.viol[i]) for i in merged])
        nxt: list[int] = []
        for fr in fronts:
            ids = [merged[i] for i in fr]
            if len(nxt) + len(ids) <= n:
                nxt += ids
                continue
            cd = crowding(list(range(len(ids))), [arc.objs[i] for i in ids])
            order = sorted(range(len(ids)), key=lambda k: -cd[k])
            nxt += [ids[k] for k in order[: n - len(nxt)]]
            break
        pop = nxt
    return pop


def _scalarize(arc: _Archive, idx: list[int]) -> list[float]:
    m = len(arc.objs[idx[0]])
    lo = [min(arc.objs[i][k] for i in idx) for k in range(m)]
    hi = [max(arc.objs[i][k] for i in idx) for k in range(m)]
    return [sum((arc.objs[i][k] - lo[k]) / ((hi[k] - lo[k]) or 1.0) for k in range(m)) / m
            + 10.0 * arc.viol[i] for i in idx]


def _bayes(prob: _Problem, arc: _Archive, rng: random.Random, notes: list[str]) -> None:
    cfg = prob.cfg
    d = len(prob.space)
    arc.add(prob, latin_hypercube(max(16, 4 * d), d, rng), 0)
    if not prob.numpy:
        notes.append("numpy unavailable: 'bayes' fell back to Latin hypercube sampling.")
        arc.add(prob, latin_hypercube(cfg.bayes_iterations, d, rng), 1)
        return
    np = _np
    nrng = np.random.default_rng(cfg.seed)
    ls, noise = 0.25, 1e-6
    for it in range(cfg.bayes_iterations):
        idx = list(range(len(arc.x)))
        X = np.asarray(arc.x, dtype=float)
        y = np.asarray(_scalarize(arc, idx))
        mu_y, sd_y = y.mean(), y.std() or 1.0
        yn = (y - mu_y) / sd_y
        K = np.exp(-((X[:, None, :] - X[None, :, :]) ** 2).sum(-1) / (2 * ls * ls))
        L = np.linalg.cholesky(K + noise * np.eye(len(X)) + 1e-8 * np.eye(len(X)))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))
        cand = nrng.random((2048, d))
        best = X[np.argmin(yn)]
        cand[:256] = np.clip(best + nrng.normal(0, 0.08, (256, d)), 0, 1)  # local exploit
        Ks = np.exp(-((cand[:, None, :] - X[None, :, :]) ** 2).sum(-1) / (2 * ls * ls))
        mu = Ks @ alpha
        v = np.linalg.solve(L, Ks.T)
        sd = np.sqrt(np.maximum(1.0 - (v * v).sum(0), 1e-12))
        z = (yn.min() - mu) / sd
        pdf = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        cdf = 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
        ei = (yn.min() - mu) * cdf + sd * pdf
        arc.add(prob, [cand[int(np.argmax(ei))].tolist()], it + 1)


# --- Monte Carlo ----------------------------------------------------------------------------

def _sample_factors(priors: list[UncertaintyPrior], n: int, seed: int,
                    numpy_ok: bool) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    if numpy_ok:
        g = _np.random.default_rng(seed)
    else:
        r = random.Random(seed)
    for p in priors:
        if p.provenance in (Provenance.PUBLISHED, Provenance.MEASURED) or p.spread_pct == 0:
            continue
        s = p.spread_pct / 100.0
        if numpy_ok:
            if p.distribution == "normal":
                v = g.normal(1.0, s, n)
            elif p.distribution == "uniform":
                v = g.uniform(1 - s, 1 + s, n)
            else:
                v = g.triangular(1 - s, 1.0, 1 + s, n)
            out[p.name] = _np.clip(v, 0.3, 3.0).tolist()
        else:
            if p.distribution == "normal":
                v = [r.gauss(1.0, s) for _ in range(n)]
            elif p.distribution == "uniform":
                v = [r.uniform(1 - s, 1 + s) for _ in range(n)]
            else:
                v = [r.triangular(1 - s, 1 + s, 1.0) for _ in range(n)]
            out[p.name] = [min(max(x, 0.3), 3.0) for x in v]
    return out


def _ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=v.__getitem__)
    r = [0.0] * len(v)
    for k, i in enumerate(order):
        r[i] = float(k)
    return r


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 3:
        return 0.0
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va and vb else 0.0


def _quantile(v: list[float], q: float) -> float:
    s = sorted(v)
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def monte_carlo(prob: _Problem, design: dict[str, Any], nominal: dict[str, float],
                factors: dict[str, list[float]], trials: int,
                margin_pct: float | None = None) -> tuple[RobustnessReport,
                                                          list[dict[str, float]]]:
    ms = prob.metrics([design] * trials,
                      {k: (_np.asarray(v) if prob.numpy else v) for k, v in factors.items()})
    c = prob.cfg.constraints
    slack = [m["slack_ps"] for m in ms]
    slack_yield = sum(s >= c.min_slack_ps for s in slack) / trials
    p_therm = sum(m["temp_c"] > c.max_temp_c for m in ms) / trials
    perf_ratio = _quantile([m["perf"] for m in ms], 0.05) / nominal["perf"]
    power_ratio = _quantile([m["power_w"] for m in ms], 0.95) / nominal["power_w"]
    p_ref = (sum(m["refresh_overhead"] > c.max_refresh_overhead for m in ms) / trials
             if "refresh_overhead" in nominal else 0.0)
    score = 100.0 * slack_yield * (1 - p_therm) * (1 - p_ref) * min(1.0, perf_ratio)
    rep = RobustnessReport(trials=trials, slack_yield=round(slack_yield, 4),
                           thermal_violation_prob=round(p_therm, 4),
                           worst_slack_ps_p5=round(_quantile(slack, 0.05), 2),
                           perf_p5_ratio=round(perf_ratio, 4),
                           power_p95_ratio=round(power_ratio, 4),
                           refresh_violation_prob=round(p_ref, 4),
                           accuracy_margin_pct=margin_pct,
                           robustness_score=round(score, 2))
    return rep, ms


# --- Public API ----------------------------------------------------------------------------

def run_digital_twin(cfg: DigitalTwinConfig, force_pure_python: bool = False
                     ) -> tuple[DigitalTwinSummary, dict[str, Any]]:
    """Run the search + Monte Carlo. Returns (compact summary, full trace for persistence)."""
    t0 = time.perf_counter()
    prob = _Problem(cfg, force_pure_python)
    rng = random.Random(cfg.seed)
    arc = _Archive()
    notes: list[str] = []
    if not prob.space:
        arc.add(prob, [[]], 0)
    elif cfg.strategy == "nsga2":
        _nsga2(prob, arc, rng)
    elif cfg.strategy == "lhs":
        arc.add(prob, latin_hypercube(cfg.lhs_samples, len(prob.space), rng), 0)
    else:
        _bayes(prob, arc, rng, notes)

    all_idx = list(range(len(arc.x)))
    fronts = nondominated_sort([(arc.objs[i], arc.viol[i]) for i in all_idx]) \
        if len(all_idx) <= 6000 else _front_large(arc)
    front = fronts[0]
    feasible_front = [i for i in front if arc.viol[i] == 0]
    if not feasible_front:
        notes.append("No design met every constraint; showing least-violating designs.")
    pool = feasible_front or front
    # Thin a large front to diverse candidates for Monte Carlo.
    if len(pool) > 3 * cfg.top_k:
        cd = crowding(pool, arc.objs)
        pool = sorted(pool, key=lambda i: -cd[i])[: 3 * cfg.top_k]

    calibrated = set((cfg.calibration or {}).keys())
    priors = cfg.priors or default_priors(calibrated, memory=cfg.memory is not None)
    priors = [p.model_copy(update={"provenance": Provenance.MEASURED})
              if p.name in calibrated else p for p in priors]
    unknown = [p.name for p in priors
               if p.provenance not in (Provenance.PUBLISHED, Provenance.MEASURED)]
    mc_rows: list[dict[str, Any]] = []
    points: list[ParetoPoint] = []
    sens_acc: dict[str, dict[str, list[float]]] = {}
    # Common random numbers: every node samples with the same seed, so designs differ only
    # by their node's accuracy margin, not by sampling noise.
    node_factors: dict[int, dict[str, list[float]]] = {}
    for i in pool:
        rob = None
        ni = _node_index(arc.designs[i]["node"])
        if cfg.mc_trials and ni not in node_factors:
            node_factors[ni] = _sample_factors(node_priors(priors, NODE_NAMES[ni]),
                                               cfg.mc_trials, cfg.seed + 1, prob.numpy)
        factors = node_factors.get(ni, {})
        if cfg.mc_trials and factors:
            rob, ms = monte_carlo(prob, arc.designs[i], arc.metrics[i], factors, cfg.mc_trials,
                                  NODE_MARGIN_PCT[NODE_NAMES[ni]])
            for name, vals in factors.items():
                for metric in ("slack_ps", "perf", "power_w", "temp_c"):
                    rho = spearman(vals, [m[metric] for m in ms])
                    sens_acc.setdefault(name, {}).setdefault(metric, []).append(abs(rho))
            mc_rows.append({"index": i, "robustness": rob.model_dump()})
        points.append(ParetoPoint(rank=0, params=_clean(arc.designs[i]),
                                  objectives={o.name: round(arc.metrics[i][o.name], 4)
                                              for o in cfg.objectives},
                                  metrics={k: round(v, 4) for k, v in arc.metrics[i].items()},
                                  feasible=arc.viol[i] == 0, robustness=rob))
    first = cfg.objectives[0]
    points.sort(key=lambda p: (-(p.robustness.robustness_score if p.robustness else 0.0),
                               -p.objectives[first.name] if first.goal == "max"
                               else p.objectives[first.name]))
    top = points[: cfg.top_k]
    for k, p in enumerate(top):
        p.rank = k + 1

    prov = {p.name: p.provenance for p in priors}
    sensitivity = sorted(
        (SensitivityEntry(parameter=name, provenance=prov[name],
                          impact={m: round(sum(v) / len(v), 4) for m, v in metrics.items()},
                          score=round(max(sum(v) / len(v) for v in metrics.values()), 4))
         for name, metrics in sens_acc.items()), key=lambda s: -s.score)

    elapsed = time.perf_counter() - t0
    run_id = f"twin_{uuid.uuid4().hex[:12]}"
    summary = DigitalTwinSummary(
        run_id=run_id, strategy=cfg.strategy, seed=cfg.seed, evaluated=len(arc.x),
        mc_evaluations=prob.evaluated - len(arc.x),
        feasible=sum(v == 0 for v in arc.viol), elapsed_s=round(elapsed, 4),
        designs_per_sec=round(prob.evaluated / elapsed, 1) if elapsed else 0.0,
        pareto_size=len(front), top=top, sensitivity=sensitivity, unknown_parameters=unknown,
        calibrated_parameters=sorted(calibrated),
        notes=notes + ([] if prob.numpy else ["Pure-Python backend (numpy not used)."]))
    rows = all_idx if len(all_idx) <= MAX_TRACE_ROWS else all_idx[-MAX_TRACE_ROWS:]
    trace = {
        "config": cfg.model_dump(mode="json"),
        "front": front,
        "rows": [{"i": i, "gen": arc.gen[i], "params": _clean(arc.designs[i]),
                  "metrics": {k: round(v, 5) for k, v in arc.metrics[i].items()},
                  "violation": round(arc.viol[i], 5)} for i in rows],
        "trace_truncated": len(rows) < len(all_idx),
        "monte_carlo": mc_rows,
    }
    return summary, trace


def _front_large(arc: _Archive) -> list[list[int]]:
    """Front 0 for very large archives: feasible-first, O(n * front) sweep."""
    order = sorted(range(len(arc.x)), key=lambda i: (arc.viol[i], arc.objs[i]))
    front: list[int] = []
    for i in order:
        if not any(_dominates((arc.objs[j], arc.viol[j]), (arc.objs[i], arc.viol[i]))
                   for j in front):
            front = [j for j in front
                     if not _dominates((arc.objs[i], arc.viol[i]), (arc.objs[j], arc.viol[j]))]
            front.append(i)
    return [front]


def _clean(d: dict[str, Any]) -> dict[str, Any]:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def evaluate_point(params: dict[str, Any], workload: TwinWorkload | None = None,
                   calibration: dict[str, float] | None = None,
                   blackboxes: list[BlackBoxMacro] | None = None,
                   memory: TwinMemory | None = None) -> dict[str, float]:
    """Evaluate one design point (nominal, no Monte Carlo). ``memory`` enables the on-chip
    memory model (params may then set mem_tech / mem_banks / mem_interleave / pim_lanes)."""
    unknown = set(params) - set(REQUIRED) - set(OPTIONAL)
    if unknown:
        raise ValueError(f"Unknown parameters {sorted(unknown)}")
    full = _finish_design({**DEFAULTS, **(MEMORY_DEFAULTS if memory else {}), **params},
                          vdd_auto="vdd" not in params and "vdd_rel" not in params)
    keys = REQUIRED + (MEMORY_KNOBS if memory else ())
    p = {k: _encode(k, full[k]) for k in keys}
    return TwinModel(workload, calibration, blackboxes, memory=memory).evaluate(
        p, None, _Scalar())


def benchmark(n: int = 20_000, seed: int = 0, force_pure_python: bool = False) -> dict[str, float]:
    """Raw model throughput (designs/sec) on random design points."""
    cfg = DigitalTwinConfig(seed=seed)
    prob = _Problem(cfg, force_pure_python)
    rng = random.Random(seed)
    designs = [prob.decode(x) for x in latin_hypercube(n, len(prob.space), rng)]
    t0 = time.perf_counter()
    prob.metrics(designs)
    dt = time.perf_counter() - t0
    return {"designs": n, "seconds": round(dt, 4), "designs_per_sec": round(n / dt, 1),
            "backend": "pure_python" if force_pure_python or _np is None else "numpy"}
