"""On-chip memory model: multi-bank 6T SRAM and 2T/3T gain-cell eDRAM with refresh, bank
conflicts, port contention, SECDED ECC and near-bank PIM MAC engines.

Two views share the same constants:
- ``analyze``: closed-form analytical model (also used, vectorized, by the digital twin via
  ``mem_core``).
- ``simulate``: deterministic cycle-accurate simulation of requesters, interleaving,
  arbitration and the refresh controller, used to cross-check the analytical numbers.

Technology factors follow the gain-cell literature (Meinerzhagen et al., 'Gain-Cell
Embedded DRAMs for Low-Power VLSI Systems-on-Chip', Springer 2018): a 2T gain cell is about
2x denser than 6T SRAM with far lower static leakage, but data decays and must be refreshed.
They are 'estimated'; retention is 'assumed' and perturbed in Monte Carlo.
"""

from __future__ import annotations

import math
import random
from typing import Any

from chiplab.library.pdk_library import PdkProfile, commercial_readiness, get_pdk
from chiplab.schemas.common import FidelityLevel
from chiplab.schemas.digital_twin import MemWorkload, OnChipMemoryConfig

TECHS = ("sram6t", "gc2t", "gc3t")
AREA_FACTOR = (1.0, 0.45, 0.62)          # bitcell area vs 6T
READ_E_FACTOR = (1.0, 0.70, 0.80)
LEAK_FACTOR = (1.0, 0.15, 0.20)
RET_FACTOR = (0.0, 1.0, 1.5)             # x node gain-cell retention (0 = no refresh)
FO4_FACTOR = (1.0, 1.15, 1.05)           # gain-cell read is slower
ARRAY_EFF = (0.65, 0.60, 0.60)
RET_HALVING_C = 18.0             # ~10x retention loss from 25 to 85 C
PORTS = {"1rw": 1.0, "1r1w": 2.0, "2rw": 2.0}

DISCLAIMER = ("Analytical / cycle-level on-chip memory model with literature-derived "
              "technology factors. Bitcell timing, retention and energy must be verified with "
              "OpenRAM characterization or SPICE on the target PDK before tape-out.")


def secded_check_bits(k: int) -> int:
    """Hamming check bits r (2^r >= k + r + 1) plus one overall-parity bit."""
    r = 0
    while 2**r < k + r + 1:
        r += 1
    return r + 1


def interleave_bank(addr: int, banks: int, words_per_bank: int, mode: str) -> int:
    if banks == 1:
        return 0
    bw = banks.bit_length() - 1
    if mode == "low_order":
        return addr & (banks - 1)
    if mode == "xor":
        return (addr ^ (addr >> bw)) & (banks - 1)
    return (addr // words_per_bank) & (banks - 1)  # block


def effective_banks(cfg: OnChipMemoryConfig, wl: MemWorkload) -> float:
    """Banks a request stream spreads over (stride/pattern vs interleaving)."""
    b = cfg.banks
    if wl.pattern == "random":
        return float(b)
    stride = 1 if wl.pattern == "sequential" else wl.stride
    if cfg.interleave == "block":
        # Requesters walk independent regions; a burst stays inside one bank.
        return float(max(1, min(b, cfg.num_requesters)))
    if cfg.interleave == "xor" and stride & (stride - 1) == 0:
        return float(b)  # XOR folding spreads power-of-two strides
    return float(b // math.gcd(stride, b))


class _S:
    exp, sqrt, log2 = staticmethod(math.exp), staticmethod(math.sqrt), staticmethod(math.log2)

    @staticmethod
    def minimum(a: float, b: float) -> float:
        return a if a < b else b

    @staticmethod
    def maximum(a: float, b: float) -> float:
        return a if a > b else b

    @staticmethod
    def take(t: Any, i: Any) -> float:
        return t[int(i)]


def mem_core(xp: Any, tech: Any, bits: Any, banks: Any, word_bits: Any, rows: Any,
             req_rate: Any, eff_banks: Any, ports: Any, freq_mhz: Any, temp_c: Any,
             pdk: dict[str, Any], pim_lanes: Any = 0.0, mac_bits: Any = 8.0,
             retention_factor: Any = 1.0, refresh_margin: float = 0.8,
             ecc: Any = 0.0, num_req: Any = 4.0) -> dict[str, Any]:
    """Vector-friendly core. ``pdk`` maps field -> scalar or array (per design)."""
    density = pdk["gate_density_kgates_mm2"] * 1000.0          # gates/mm2
    dyn = pdk["dynamic_energy_fj_per_gate"]
    leak_g = pdk["leakage_nw_per_gate"]
    fo4 = pdk["fo4_delay_ps"]
    k_ecc = ecc * (xp.log2(word_bits) + 2.0)                    # ~SECDED check bits
    code_bits = word_bits + k_ecc
    tot_bits = bits * code_bits / word_bits

    # Area
    cell = pdk["sram_bitcell_um2"] * xp.take(AREA_FACTOR, tech)
    array_mm2 = tot_bits * cell / 1e6 / xp.take(ARRAY_EFF, tech)
    gc = xp.minimum(tech, 1.0)                                  # 1 for gain cells
    periph_gates = banks * (2000.0 + 40.0 * word_bits + 1500.0 * gc
                            + ecc * 30.0 * word_bits
                            + pim_lanes * (mac_bits * mac_bits * 6.0 + 40.0))
    xbar_gates = num_req * banks * word_bits * 2.0
    area = array_mm2 + (periph_gates + xbar_gates) / density

    # Timing
    path_fo4 = (25.0 + 3.0 * xp.log2(xp.maximum(rows, 2.0))
                + 2.0 * xp.log2(word_bits * 4.0)) * xp.take(FO4_FACTOR, tech)
    path_ps = fo4 * path_fo4
    fmax_mhz = 1e6 / path_ps
    slack_ps = 1e6 / freq_mhz - path_ps

    # Retention / refresh (gain cells only)
    ret_us = pdk["gain_cell_retention_us_25c"] * xp.take(RET_FACTOR, tech) * retention_factor \
        * 2.0 ** (-(temp_c - 25.0) / RET_HALVING_C)
    period_cycles = xp.maximum(ret_us * refresh_margin * freq_mhz, 1e-9)
    refresh_overhead = gc * rows / period_cycles

    # Conflicts / contention: closed-loop model of R requesters, each with one outstanding
    # request (a stalled requester retries every cycle, an idle one issues with probability
    # p = req_rate / R). For a tagged request, K other active requests hit the same bank with
    # K ~ Binomial(R-1, a / eff_banks); it wins a port with probability min(1, P/(K+1)).
    # Refresh occupies one port of the bank for a fraction 'refresh_overhead' of cycles.
    R = int(num_req) if not hasattr(num_req, "__len__") else int(max(num_req))
    p_issue = xp.minimum(req_rate / R, 1.0)
    ov = xp.minimum(refresh_overhead, 1.0)
    a = p_issue
    lose = p_issue * 0.0
    lose_conf = lose
    for _ in range(4):
        q = a / eff_banks
        win_free = 0.0
        win_ref = 0.0
        for k in range(R):
            pmf = math.comb(R - 1, k) * q ** k * (1.0 - q) ** (R - 1 - k)
            win_free = win_free + pmf * xp.minimum(ports / (k + 1.0), 1.0)
            win_ref = win_ref + pmf * xp.minimum(xp.maximum(ports - 1.0, 0.0) / (k + 1.0), 1.0)
        win = (1.0 - ov) * win_free + ov * win_ref
        lose = 1.0 - win
        lose_conf = 1.0 - win_free
        a = p_issue / xp.maximum(1.0 - lose + p_issue * lose, 1e-9)
    conflict = lose_conf
    served = R * a * (1.0 - lose)
    rho = xp.minimum(lose, 0.999)
    latency = 2.0 + ecc + lose / xp.maximum(1.0 - lose, 1e-3)

    # Energy / power
    e_read_fj = dyn * (20.0 + rows / 8.0) * xp.take(READ_E_FACTOR, tech)
    e_write_fj = e_read_fj * 1.2
    refresh_w = gc * tot_bits * e_read_fj * 1.1e-15 / xp.maximum(ret_us * refresh_margin * 1e-6,
                                                                1e-12)
    leak_w = (tot_bits * leak_g * 0.3 * xp.take(LEAK_FACTOR, tech)
              + (periph_gates + xbar_gates) * leak_g) * 1e-9
    bw_gbs = served * word_bits / 8.0 * freq_mhz / 1e3

    # Near-bank PIM
    avail = 1.0 - ov
    busy = xp.minimum(served / xp.maximum(eff_banks * ports, 1e-9), 1.0)
    pim_mac_cycle = banks * pim_lanes * avail * xp.maximum(1.0 - busy, 0.0)
    pim_gmacs = pim_mac_cycle * freq_mhz / 1e3
    e_mac_fj = dyn * mac_bits * mac_bits * 1.2
    return {"area_mm2": area, "array_area_mm2": array_mm2, "fmax_mhz": fmax_mhz,
            "slack_ps": slack_ps, "retention_us": ret_us, "refresh_overhead": refresh_overhead,
            "conflict_rate": conflict, "served_per_cycle": served, "bandwidth_gbs": bw_gbs,
            "avg_latency_cycles": latency, "read_energy_pj_per_bit": e_read_fj / 1000.0,
            "write_energy_pj_per_bit": e_write_fj / 1000.0, "refresh_power_w": refresh_w,
            "leakage_w": leak_w, "pim_gmacs": pim_gmacs, "pim_energy_pj_per_mac": e_mac_fj / 1e3,
            "utilization": busy, "stall_prob": rho}


def pdk_scalars(p: PdkProfile) -> dict[str, float]:
    return {f: p.scalar(f) for f in ("gate_density_kgates_mm2", "dynamic_energy_fj_per_gate",
                                     "leakage_nw_per_gate", "fo4_delay_ps", "sram_bitcell_um2",
                                     "gain_cell_retention_us_25c", "nominal_vdd_v")}


def analyze(cfg: OnChipMemoryConfig, wl: MemWorkload | None = None,
            retention_factor: float = 1.0) -> dict[str, Any]:
    """Analytical report for one configuration."""
    wl = wl or MemWorkload()
    pdk = get_pdk(cfg.pdk)
    ps = pdk_scalars(pdk)
    tech = TECHS.index(cfg.technology)
    rows = cfg.rows_per_bank
    bits = cfg.capacity_kib * 1024 * 8
    eff_b = effective_banks(cfg, wl)
    ports = PORTS[cfg.ports] if cfg.ports != "1r1w" else \
        1.0 / max(wl.read_fraction, 1.0 - wl.read_fraction, 0.5)
    probe = mem_core(_S, tech, bits, cfg.banks, cfg.word_bits, rows, wl.requests_per_cycle,
                     eff_b, ports, 1.0, cfg.temp_c, ps)
    freq = cfg.freq_mhz or round(probe["fmax_mhz"] * 0.8, 1)
    ecc_bits = secded_check_bits(cfg.word_bits) if cfg.ecc == "secded" else 0
    m = mem_core(_S, tech, bits, cfg.banks, cfg.word_bits, rows, wl.requests_per_cycle,
                 eff_b, ports, freq, cfg.temp_c, ps, cfg.pim_lanes, cfg.pim.mac_bits,
                 retention_factor, cfg.refresh_margin, 1.0 if cfg.ecc == "secded" else 0.0,
                 cfg.num_requesters)
    if cfg.interleave == "block" and wl.pattern != "random":
        # Each requester streams through its own bank: no request-request conflicts while
        # requesters <= banks; beyond that requesters share banks.
        m["conflict_rate"] = max(0.0, 1.0 - cfg.banks / cfg.num_requesters)
    # A request is also stalled when its bank is refreshing (refresh takes one port).
    m["refresh_stall_prob"] = min(1.0, m["refresh_overhead"]) / ports
    dyn_w = m["served_per_cycle"] * freq * 1e6 * (cfg.word_bits + ecc_bits) * 1e-12 * (
        wl.read_fraction * m["read_energy_pj_per_bit"]
        + (1 - wl.read_fraction) * m["write_energy_pj_per_bit"])
    pim_w = m["pim_gmacs"] * 1e9 * wl.pim_fraction * m["pim_energy_pj_per_mac"] * 1e-12
    warnings = []
    if m["slack_ps"] < 0:
        warnings.append(f"Clock {freq:g} MHz exceeds macro fmax {m['fmax_mhz']:.0f} MHz.")
    if cfg.technology != "sram6t":
        if m["refresh_overhead"] >= 1.0:
            warnings.append("Refresh cannot keep up: retention too short for the row count. "
                            "Use more banks, fewer rows per bank, lower temperature or SRAM.")
        elif m["refresh_overhead"] > 0.2:
            warnings.append(f"Refresh steals {m['refresh_overhead'] * 100:.0f}% of bank cycles.")
    if m["stall_prob"] >= 0.5:
        warnings.append(f"Requests stall {m['stall_prob'] * 100:.0f}% of attempts (bank "
                        "conflicts / refresh); add banks, ports or change interleaving.")
    if cfg.ports != "1rw":
        warnings.append("RTL generator emits single-port (1RW) banks; multi-port contention is "
                        "modeled analytically only.")
    warnings += commercial_readiness(cfg.pdk, "research")
    return {
        "config": cfg.model_dump(mode="json"),
        "pdk": {"key": pdk.key, "node_nm": pdk.node_nm, "tapeout_ready": pdk.tapeout_ready,
                "accuracy_margin_pct": pdk.accuracy_margin_pct},
        "freq_mhz": freq,
        "ecc_check_bits": ecc_bits,
        "ecc_overhead_pct": round(100.0 * ecc_bits / cfg.word_bits, 2),
        "effective_banks": eff_b,
        "metrics": {k: round(float(v), 6) for k, v in m.items()},
        "power_w": {"dynamic": round(dyn_w, 6), "refresh": round(m["refresh_power_w"], 6),
                    "leakage": round(m["leakage_w"], 6), "pim": round(pim_w, 6),
                    "total": round(dyn_w + m["refresh_power_w"] + m["leakage_w"] + pim_w, 6)},
        "pim": {"lanes_per_bank": cfg.pim_lanes,
                "peak_gmacs": round(cfg.banks * cfg.pim_lanes * freq / 1e3, 3),
                "effective_gmacs": round(m["pim_gmacs"], 3),
                "effective_tops": round(2 * m["pim_gmacs"] / 1e3, 4)},
        "provenance": {"technology_factors": "estimated", "retention": "assumed",
                       "pdk_fields": f"{pdk.key} ladder profile (see docs/sources_and_accuracy.md)"},
        "fidelity": FidelityLevel.ANALYTICAL.value,
        "fidelity_disclaimer": DISCLAIMER,
        "warnings": warnings,
    }


def simulate(cfg: OnChipMemoryConfig, wl: MemWorkload | None = None, cycles: int = 20_000,
             seed: int = 1, freq_mhz: float | None = None,
             trace_cycles: int = 0) -> dict[str, Any]:
    """Deterministic cycle-accurate simulation (pure Python).

    ``trace_cycles`` > 0 also returns per-cycle frames for the first N cycles (capped at
    2000) for visualization: per bank '.' idle / 'R' read / 'W' write / 'F' refresh
    (lower-case r/w = served while refreshing on a second port); per requester '.' idle,
    'g' granted, 's' bank-conflict stall, 'f' refresh stall, with its target bank. A
    ~100-bucket timeline of served / conflict / refresh-stall counts is always returned."""
    wl = wl or MemWorkload()
    cycles = int(min(max(cycles, 100), 500_000))
    rng = random.Random(seed)
    b_n, r_n, wpb = cfg.banks, cfg.num_requesters, cfg.words_per_bank
    words = b_n * wpb
    if freq_mhz is None:
        freq_mhz = analyze(cfg, wl)["freq_mhz"]
    pdk = get_pdk(cfg.pdk)
    tech = TECHS.index(cfg.technology)
    gain = tech > 0
    ret_us = pdk.scalar("gain_cell_retention_us_25c") * RET_FACTOR[tech] * \
        2.0 ** (-(cfg.temp_c - 25.0) / RET_HALVING_C)
    rows = cfg.rows_per_bank
    period = ret_us * cfg.refresh_margin * freq_mhz if gain else math.inf
    interval = max(1, int(period // rows)) if gain else 0
    ref_next = [interval + (b * interval) // max(b_n, 1) for b in range(b_n)] if gain else []
    ref_pending = [0] * b_n            # cycle the pending refresh became due (0 = none)
    ref_done = [0] * b_n
    missed = 0
    ports_rw = {"1rw": (1, 0, 0), "1r1w": (0, 1, 1), "2rw": (2, 0, 0)}[cfg.ports]
    rr = [0] * b_n
    p_req = min(1.0, wl.requests_per_cycle / r_n)
    ptr = [r * (words // r_n) for r in range(r_n)]
    step = 1 if wl.pattern == "sequential" else wl.stride
    waiting: list[tuple[int, int, bool, int] | None] = [None] * r_n  # bank, issue_cycle, is_read, first
    issued = served = conflicts = refresh_blocks = lat_sum = 0
    max_wait = 0
    trace_n = int(min(max(trace_cycles, 0), 2000))
    frames: list[dict[str, Any]] = []
    bucket = max(1, cycles // 100)
    timeline: list[list[int]] = []
    tl = [0, 0, 0, 0]  # served, conflicts, refresh stalls, issued
    for cyc in range(cycles):
        n_iss0, n_srv0, n_cf0, n_rb0 = issued, served, conflicts, refresh_blocks
        tracing = cyc < trace_n
        if tracing:
            bank_st = ["."] * b_n
            req_st = ["."] * r_n
            target = [-1] * r_n
        for r in range(r_n):
            if waiting[r] is None and rng.random() < p_req:
                if wl.pattern == "random":
                    a = rng.randrange(words)
                else:
                    a = ptr[r] % words
                    ptr[r] += step
                bank = interleave_bank(a, b_n, wpb, cfg.interleave)
                waiting[r] = (bank, cyc, rng.random() < wl.read_fraction, 1)
                issued += 1
        busy_ref = [False] * b_n
        if gain:
            for b in range(b_n):
                if cyc >= ref_next[b] and not ref_pending[b]:
                    ref_pending[b] = cyc + 1
                    ref_next[b] += interval
                if ref_pending[b]:
                    if cyc - (ref_pending[b] - 1) >= interval:
                        missed += 1
                    busy_ref[b] = True       # refresh has priority
                    ref_pending[b] = 0
                    ref_done[b] += 1
        by_bank: dict[int, list[int]] = {}
        for r in range(r_n):
            w = waiting[r]
            if w is not None:
                by_bank.setdefault(w[0], []).append(r)
        for b, cands in by_bank.items():
            rw_slots, r_slots, w_slots = ports_rw
            if busy_ref[b]:
                if rw_slots:
                    rw_slots -= 1
                elif r_slots:
                    r_slots -= 1
            if cfg.arbiter == "round_robin":
                order = sorted(cands, key=lambda r: (r - rr[b]) % r_n)
            else:
                order = sorted(cands)
            granted = []
            for r in order:
                is_read = waiting[r][2]  # type: ignore[index]
                if rw_slots:
                    rw_slots -= 1
                elif is_read and r_slots:
                    r_slots -= 1
                elif not is_read and w_slots:
                    w_slots -= 1
                else:
                    continue
                granted.append(r)
            if granted and cfg.arbiter == "round_robin":
                rr[b] = (granted[-1] + 1) % r_n
            total_ports = sum(ports_rw)
            for rank, r in enumerate(order):
                w = waiting[r]
                assert w is not None
                if tracing:
                    target[r] = b
                if r in granted:
                    if tracing:
                        req_st[r] = "g"
                        ch = "R" if w[2] else "W"
                        bank_st[b] = ch.lower() if busy_ref[b] else ch
                    lat = cyc - w[1] + 2 + (1 if cfg.ecc == "secded" else 0)
                    lat_sum += lat
                    max_wait = max(max_wait, cyc - w[1])
                    served += 1
                    waiting[r] = None
                else:
                    if w[3]:
                        # Would have won a port without the refresh -> refresh stall.
                        if busy_ref[b] and rank < total_ports:
                            refresh_blocks += 1
                        else:
                            conflicts += 1
                    if tracing:
                        req_st[r] = "f" if busy_ref[b] and rank < total_ports else "s"
                    waiting[r] = (w[0], w[1], w[2], 0)
        if tracing:
            for b in range(b_n):
                if busy_ref[b] and bank_st[b] == ".":
                    bank_st[b] = "F"
            for r in range(r_n):
                if waiting[r] is not None and req_st[r] == ".":
                    req_st[r] = "s"
                    target[r] = waiting[r][0]  # type: ignore[index]
            frames.append({"c": cyc, "banks": "".join(bank_st), "req": "".join(req_st),
                           "target": target})
        tl[0] += served - n_srv0
        tl[1] += conflicts - n_cf0
        tl[2] += refresh_blocks - n_rb0
        tl[3] += issued - n_iss0
        if (cyc + 1) % bucket == 0 or cyc == cycles - 1:
            timeline.append([cyc + 1, *tl])
            tl = [0, 0, 0, 0]
    return {
        "cycles": cycles, "seed": seed, "freq_mhz": freq_mhz, "issued": issued,
        "served": served, "throughput_per_cycle": round(served / cycles, 5),
        "conflict_rate": round(conflicts / max(issued, 1), 5),
        "refresh_stall_rate": round(refresh_blocks / max(issued, 1), 5),
        "refresh_overhead": round(sum(ref_done) / (cycles * b_n), 6) if gain else 0.0,
        "refresh_interval_cycles": interval, "missed_refresh_deadlines": missed,
        "avg_latency_cycles": round(lat_sum / max(served, 1), 4), "max_wait_cycles": max_wait,
        "fidelity": "cycle_accurate_python",
        "timeline": {"columns": ["cycle", "served", "conflicts", "refresh_stalls", "issued"],
                     "bucket_cycles": bucket, "rows": timeline},
        **({"frames": frames} if trace_n else {}),
    }
