"""Commodity DRAM PIM Designer.

Builds on chiplab.pim.dram (1T1C charge sharing, sense amp, multi-row activation, Monte
Carlo variation) and adds what a design study needs:

  retention      lognormal per-cell retention; failure fraction at a refresh interval, for a
                 normal read and for the smaller TRA margin
  refresh        tRFC / tREFI time lost and how it blocks compute
  variation      row yield for PIM: probability that every column of a row computes right
  scheduling     event-driven schedule of row operations across banks under tRRD, tFAW and
                 refresh blackouts
  area           reserved rows and per-bank logic as a fraction of the die
  comparison     in-array (Ambit-style) vs near-bank logic (HBM-PIM/UPMEM-style) vs a CPU
                 reading DRAM over the channel, on the same workload

Every constant carries a provenance string. Defaults are generic DDR4-like values, not one
vendor's part; results are design-space estimates (fidelity "analytical"/"behavioral").
"""

from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np

from chiplab.pim import dram

GEOMETRY: dict[str, tuple[float, str]] = {
    "banks": (16, "DDR4 x8: 16 banks (4 bank groups)"),
    "subarrays_per_bank": (64, "assumed"),
    "rows_per_subarray": (512, "assumed; Ambit evaluates 512-row subarrays"),
    "row_bytes_rank": (8192, "8 KB row across an x8 rank (8 chips x 1 KB)"),
    "t_rrd_s": (5e-9, "JEDEC DDR4 typical tRRD_S"),
    "t_faw_s": (21e-9, "JEDEC DDR4-3200 x8 typical tFAW"),
    "t_refi_s": (7.8e-6, "JEDEC DDR4 tREFI (normal temperature)"),
    "t_rfc_s": (350e-9, "JEDEC DDR4 8 Gb tRFC"),
    "channel_bytes_per_s": (25.6e9, "DDR4-3200 x64 channel peak"),
    "die_area_mm2": (60.0, "assumed: 8 Gb DDR4 die order of magnitude"),
    "cell_array_fraction": (0.55, "assumed: share of die area in cell arrays"),
    "reserved_rows": (8, "Ambit B-group: 4 temporary + 2 DCC + 2 control rows"),
    "decoder_overhead_frac": (0.002, "assumed: extra row-decoder logic for B-group"),
    "nb_pu_area_mm2_per_bank": (0.25, "assumed: SIMD unit in DRAM process per bank"),
    "nb_bits_per_cycle": (256, "assumed: bank I/O width feeding the near-bank unit"),
    "nb_clock_hz": (300e6, "assumed: logic clock in a DRAM process"),
    "nb_internal_pj_per_bit": (1.0, "assumed: column access inside the die, no I/O"),
    "alu_pj_per_8b_add": (0.03, "published: Horowitz ISSCC 2014, 45 nm"),
    "alu_pj_per_8b_mul": (0.2, "published: Horowitz ISSCC 2014, 45 nm"),
    "cpu_pj_per_8b_op": (0.5, "assumed: SIMD op incl. register file, per byte"),
    "retention_median_s": (2.0, "assumed: median cell retention at room temperature"),
    "retention_log_sigma": (1.0, "assumed: lognormal spread; weak-cell tail sets refresh"),
}

WORKLOADS = {
    "bitwise_and": {"in_array": "and", "bytes_in": 2, "bytes_out": 1, "alu": "add"},
    "bitwise_xor": {"in_array": "xor", "bytes_in": 2, "bytes_out": 1, "alu": "add"},
    "copy": {"in_array": "copy", "bytes_in": 1, "bytes_out": 1, "alu": None},
    "vector_add_int8": {"in_array": None, "bytes_in": 2, "bytes_out": 1, "alu": "add"},
    "dot_int8": {"in_array": None, "bytes_in": 2, "bytes_out": 0, "alu": "mul"},
}


def geometry(overrides: dict[str, Any] | None = None) -> dict[str, float]:
    g = {k: v for k, (v, _) in GEOMETRY.items()}
    for k, v in (overrides or {}).items():
        if k not in g:
            raise dram.DramError(f"unknown geometry parameter {k}")
        if not isinstance(v, (int, float)) or v <= 0:
            raise dram.DramError(f"{k} must be positive")
        g[k] = float(v)
    return g


# ---- retention & refresh -------------------------------------------------------------------
def retention(t_refresh_s: float = 0.064, overrides: dict[str, Any] | None = None,
              dram_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fraction of cells that fail at a refresh interval, for a normal read and for TRA.

    A stored 1 decays as exp(-t/tau_cell) with tau_cell lognormal. A cell fails when the
    remaining charge no longer gives the sense amp the required margin (3 sigma of offset).
    For TRA the decayed cell shares the bitline with two other cells, so the margin is smaller
    and weaker cells fail first."""
    g, p = geometry(overrides), dram.params(dram_overrides)
    need = 3 * p["sa_offset_sigma_v"]
    cc, cb, v = p["c_cell_f"], p["c_bitline_f"], p["vdd"]
    # single read: dV = cc (V - v/2)/(cb+cc) >= need  ->  V >= v/2 + need (cb+cc)/cc
    v_min_single = v / 2 + need * (cb + cc) / cc
    # TRA worst case for a 1 result: this cell (decayed) + one full 1 + one 0:
    # dV = cc (V + v - 3v/2)/(cb + 3cc) >= need -> V >= v/2 + need (cb+3cc)/cc
    v_min_tra = v / 2 + need * (cb + 3 * cc) / cc
    out = {}
    for name, vmin in (("single_read", v_min_single), ("tra", v_min_tra)):
        if vmin >= v:
            out[name] = {"fail_fraction": 1.0, "v_min": vmin}
            continue
        tau_needed = t_refresh_s / math.log(v / vmin)       # cells with smaller tau fail
        z = (math.log(tau_needed) - math.log(g["retention_median_s"])) / g[
            "retention_log_sigma"]
        frac = 0.5 * math.erfc(-z / math.sqrt(2))
        out[name] = {"fail_fraction": frac, "v_min": vmin, "tau_needed_s": tau_needed}
    refresh_busy = g["t_rfc_s"] / g["t_refi_s"]
    return {"t_refresh_s": t_refresh_s, **out, "refresh_time_fraction": refresh_busy,
            "note": "TRA needs fresher charge than a normal read; either refresh compute rows "
                    "more often or copy operands in (RowClone) right before computing, which "
                    "Ambit does"}


# ---- manufacturing variation -> row yield --------------------------------------------------
def row_yield(op: str = "and", trials: int = 200_000, seed: int = 0,
              dram_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Probability that all columns of one row compute correctly, from the per-column BER."""
    mc = dram.monte_carlo(op, trials, seed, dram_overrides)
    bits = dram.params(dram_overrides)["row_bits"]
    ber, up = mc["bit_error_rate"], mc["ber_95_upper"]
    return {"op": op, "ber": ber, "ber_95_upper": up,
            "row_yield": (1 - ber) ** bits, "row_yield_95_lower": (1 - up) ** bits,
            "trials": trials,
            "note": "with zero observed errors the bound is limited by the trial count"}


# ---- area ----------------------------------------------------------------------------------
def area_overhead(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    g = geometry(overrides)
    in_array = g["reserved_rows"] / g["rows_per_subarray"] * g["cell_array_fraction"] + g[
        "decoder_overhead_frac"]
    near_bank = g["banks"] * g["nb_pu_area_mm2_per_bank"] / g["die_area_mm2"]
    return {"in_array_fraction": in_array, "near_bank_fraction": near_bank,
            "cpu_fraction": 0.0,
            "in_array_detail": f"{int(g['reserved_rows'])} reserved rows per "
                               f"{int(g['rows_per_subarray'])}-row subarray + decoder",
            "note": "near-bank logic in a DRAM process is far less dense than in a logic "
                    "process; capacity is lost either way"}


# ---- scheduling ----------------------------------------------------------------------------
def schedule(n_row_ops: int, op: str = "and", overrides: dict[str, Any] | None = None,
             dram_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Event-driven schedule of `n_row_ops` independent in-array row operations over all banks.

    Each op is AAPS[op] activate-precharge pairs on one bank (serial within the bank). Across
    banks: at most one ACT per tRRD and four per tFAW (rank-wide), and no ACT during a
    refresh (tRFC every tREFI)."""
    if not 1 <= n_row_ops <= 200_000:
        raise dram.DramError("n_row_ops must be in [1, 200000]")
    g, p = geometry(overrides), dram.params(dram_overrides)
    aaps = dram.AAPS[op] or 1
    acts_per_op = {"copy": 2, "and": 8, "or": 8, "maj": 1, "not": 4, "xor": 18,
                   "copy_psm": 2}[op]
    act_gap = (p["t_ras_s"] + p["t_rp_s"]) * aaps / acts_per_op   # bank-local spacing
    banks = int(g["banks"])
    bank_free = [(0.0, b) for b in range(banks)]
    heapq.heapify(bank_free)
    recent: list[float] = []                 # last four ACT times (tFAW window)
    last_act = -1e9
    t_end = 0.0

    def legal(t: float) -> float:
        while True:
            t = max(t, last_act + g["t_rrd_s"])
            if len(recent) == 4:
                t = max(t, recent[0] + g["t_faw_s"])
            k = math.floor(t / g["t_refi_s"])
            r0 = k * g["t_refi_s"]
            if r0 <= t < r0 + g["t_rfc_s"]:
                t = r0 + g["t_rfc_s"]
                continue
            return t

    for _ in range(n_row_ops):
        t, b = heapq.heappop(bank_free)
        for _a in range(acts_per_op):
            t = legal(t)
            last_act = t
            recent = (recent + [t])[-4:]
            t += act_gap
        t_end = max(t_end, t)
        heapq.heappush(bank_free, (t, b))
    ideal = n_row_ops * acts_per_op * act_gap / banks
    bytes_done = n_row_ops * g["row_bytes_rank"]
    return {"op": op, "row_ops": n_row_ops, "makespan_s": t_end,
            "throughput_bytes_per_s": bytes_done / t_end,
            "bank_parallel_ideal_s": ideal, "efficiency": ideal / t_end,
            "binding_constraint": "tFAW/tRRD (rank activation rate)"
            if 4 / g["t_faw_s"] < banks / act_gap else "bank occupancy",
            "fidelity": "analytical"}


# ---- architecture comparison ---------------------------------------------------------------
def compare(workload: str = "bitwise_and", n_bytes: int = 64 * 2**20,
            overrides: dict[str, Any] | None = None,
            dram_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """In-array vs near-bank vs CPU on one workload of n_bytes per operand."""
    if workload not in WORKLOADS:
        raise dram.DramError(f"workload must be one of {list(WORKLOADS)}")
    if not 1 <= n_bytes <= 2**36:
        raise dram.DramError("n_bytes must be in [1, 64 GiB]")
    w = WORKLOADS[workload]
    g, p = geometry(overrides), dram.params(dram_overrides)
    alu_pj = {"add": g["alu_pj_per_8b_add"], "mul": g["alu_pj_per_8b_mul"] + g[
        "alu_pj_per_8b_add"], None: 0.0}[w["alu"]]
    moved = n_bytes * (w["bytes_in"] + w["bytes_out"])
    rows_touched = math.ceil(n_bytes / g["row_bytes_rank"]) * (w["bytes_in"] + max(
        w["bytes_out"], 0))
    act_j = g["row_bytes_rank"] * 8 * p["c_bitline_f"] * p["vdd"] ** 2 / 2

    # CPU
    cpu_t = moved / g["channel_bytes_per_s"]
    cpu_e = moved * 8 * p["io_pj_per_bit"] * 1e-12 + rows_touched * act_j + n_bytes * g[
        "cpu_pj_per_8b_op"] * 1e-12
    out = {"cpu": {"supported": True, "time_s": cpu_t, "energy_j": cpu_e,
                   "bytes_over_channel": moved}}

    # near-bank
    bank_bw = g["nb_bits_per_cycle"] / 8 * g["nb_clock_hz"]
    refresh = 1 - g["t_rfc_s"] / g["t_refi_s"]
    nb_t = moved / (bank_bw * g["banks"] * refresh)
    nb_e = moved * 8 * g["nb_internal_pj_per_bit"] * 1e-12 + rows_touched * act_j + \
        n_bytes * alu_pj * 1e-12
    out["near_bank"] = {"supported": True, "time_s": nb_t, "energy_j": nb_e,
                        "bytes_over_channel": 0}

    # in-array
    op = w["in_array"]
    if op is None:
        out["in_array"] = {"supported": False,
                           "why": "Ambit-style arrays do bitwise logic and copy; integer "
                                  "add/multiply need many bit-serial TRA steps (SIMDRAM-like) "
                                  "and are not modelled"}
    else:
        n_ops = math.ceil(n_bytes / g["row_bytes_rank"])
        sch = schedule(min(n_ops, 200_000), op, overrides, dram_overrides)
        scale = n_ops / min(n_ops, 200_000)
        te = dram.timing_energy(op, dram_overrides)
        chips = g["row_bytes_rank"] * 8 / p["row_bits"]
        out["in_array"] = {"supported": True, "time_s": sch["makespan_s"] * scale,
                           "energy_j": n_ops * te["energy_nj"] * 1e-9 * chips,
                           "bytes_over_channel": 0, "schedule_efficiency": sch["efficiency"],
                           "row_yield": row_yield(op, 50_000, 0, dram_overrides)["row_yield"]}
    best_e = min((k for k in out if out[k].get("supported")), key=lambda k: out[k]["energy_j"])
    best_t = min((k for k in out if out[k].get("supported")), key=lambda k: out[k]["time_s"])
    return {"workload": workload, "n_bytes": n_bytes, "architectures": out,
            "lowest_energy": best_e, "fastest": best_t, "area": area_overhead(overrides),
            "retention": retention(0.064, overrides, dram_overrides),
            "fidelity": "analytical",
            "caveats": ["The in-array energy model over-predicts the published Ambit/RowClone "
                        "gains (see dram.benchmark_review); treat its ratios as upper bounds.",
                        "Near-bank constants are assumptions, not a measured product.",
                        "CPU side ignores caches: right for streaming data larger than LLC."]}


def designer_report(workloads: list[str] | None = None,
                    n_bytes: int = 64 * 2**20) -> dict[str, Any]:
    """All ten designer checks in one call."""
    p = dram.params()
    return {
        "cell_and_charge_sharing": dram.margins(p),
        "sense_amp": {"single_row_sense_ns": dram.sense_time_ns(
            dram.margins(p)["single_row_mv"] / 1e3, p), "tra_sense_ns": dram.sense_time_ns(
            dram.margins(p)["tra_mv"] / 1e3, p)},
        "multi_row_activation": {op: dram.monte_carlo(op, 20000, 0) for op in ("and", "maj")},
        "bank_level_and_scheduling": schedule(4096, "and"),
        "refresh_and_retention": retention(),
        "manufacturing_variation": row_yield("and", 100_000),
        "area": area_overhead(),
        "energy_per_op": {op: dram.timing_energy(op) for op in ("copy", "and", "not", "xor")},
        "read_disturb": dram.read_disturb(1e6),
        "comparison": {wl: compare(wl, n_bytes) for wl in (workloads or list(WORKLOADS))},
    }


__all__ = ["GEOMETRY", "WORKLOADS", "area_overhead", "compare", "designer_report", "retention",
           "row_yield", "schedule"]

_ = np  # numpy kept for callers that pass arrays through dram
