"""DRAM charge-sharing simulator for in-DRAM bulk bitwise operations.

Mechanisms (Seshadri et al., Ambit, MICRO 2017; RowClone, MICRO 2013):

* Single-row activation: the cell (C_c, at V_cell) shares charge with the bitline (C_b,
  precharged to VDD/2). Bitline deviation
      dV = C_c (V_cell - VDD/2) / (C_b + C_c).
* Triple-row activation (TRA): three cells share one bitline,
      dV = C_c (V1 + V2 + V3 - 3 VDD/2) / (C_b + 3 C_c),
  so the sense amplifier resolves the majority. With a control row of 0s the result is
  A AND B, with 1s it is A OR B. The sense amp then restores the result into all three rows,
  so the sources are destroyed: Ambit copies operands into reserved rows first (RowClone).
* NOT: a dual-contact cell (DCC) row is connected to the negated bitline; after sensing A, the
  DCC captures NOT A, which is then copied out.
* RowClone FPM: ACT(src) then ACT(dst) in the same subarray; the sense amplifier overwrites
  dst with src. PSM copies across banks over the internal bus.

Non-idealities in the Monte Carlo: per-cell capacitance spread, per-bitline capacitance
spread, sense-amp input offset (static per column), kT/C noise, charge loss since the last
refresh, incomplete charge sharing for short activation, and read-disturb (RowHammer) risk
from repeated activation.

All device numbers are labelled. Defaults are typical public values, not a specific part;
change them before drawing conclusions about a real DRAM.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from chiplab.tools import registry
from chiplab.tools.spice import parse_raw

KB = 1.380649e-23

PARAMS: dict[str, tuple[float, str]] = {
    "vdd": (1.2, "JEDEC DDR4 VDD"),
    "c_cell_f": (22e-15, "assumed: typical published DRAM cell 10-30 fF"),
    "c_bitline_f": (85e-15, "assumed: bitline/cell ratio ~4"),
    "sigma_c_cell": (0.05, "assumed: 5 % cell capacitance spread (1 sigma)"),
    "sigma_c_bitline": (0.03, "assumed: 3 % bitline capacitance spread"),
    "sa_offset_sigma_v": (0.010, "assumed: 10 mV sense-amp offset (1 sigma)"),
    "retention_tau_s": (0.5, "assumed: exponential charge loss time constant"),
    "t_since_refresh_max_s": (0.064, "JEDEC 64 ms refresh window"),
    "r_access_ohm": (20e3, "assumed: access transistor on-resistance"),
    "t_share_s": (5e-9, "assumed: time from wordline rise to sense enable"),
    "tau_sense_s": (0.15e-9, "assumed: sense-amp regeneration time constant"),
    "t_ras_s": (35e-9, "JEDEC DDR4 typical tRAS"),
    "t_rp_s": (13.75e-9, "JEDEC DDR4 typical tRP"),
    "temperature_k": (300.0, "room temperature"),
    "row_bits": (8192, "8 Kb row per subarray (assumed)"),
    "hammer_threshold": (10_000, "assumed: activations of a neighbour before flips appear; "
                                 "published DDR4 values span thousands to ~100k"),
    "io_pj_per_bit": (20.0, "derived: Horowitz ISSCC 2014 DRAM access 1.3 nJ per 64 bits"),
    "wordline_pj": (5.0, "assumed: wordline drive energy per activation"),
    # Datasheet route (Micron TN-41-01 / TN-40-07 method). 0 = not supplied: the bitline-only
    # estimate is used. Fill these from the IDD table of the part being modelled.
    "idd0_a": (0.0, "datasheet IDD0 (one-bank ACT-PRE current) per device; not supplied"),
    "idd2n_a": (0.0, "datasheet IDD2N (precharge standby) per device; not supplied"),
    "idd3n_a": (0.0, "datasheet IDD3N (active standby) per device; not supplied"),
    "t_rc_s": (48.75e-9, "JEDEC DDR4 typical tRC = tRAS + tRP"),
    "devices_per_row": (1.0, "devices whose rows open together; row_bits is per device"),
}

# Activate-activate-precharge (AAP) sequences per operation (Ambit, Sec. 5).
AAPS = {"copy": 1, "and": 4, "or": 4, "maj": 1, "not": 2, "xor": 9, "copy_psm": 0}
OPERATIONS = tuple(AAPS)


class DramError(ValueError):
    pass


def params(overrides: dict[str, Any] | None = None) -> dict[str, float]:
    p = {k: v for k, (v, _) in PARAMS.items()}
    for k, v in (overrides or {}).items():
        if k not in p:
            raise DramError(f"unknown parameter {k}")
        if not isinstance(v, (int, float)) or v < 0 or not math.isfinite(v):
            raise DramError(f"{k} must be a finite non-negative number")
        p[k] = float(v)
    if p["c_cell_f"] <= 0 or p["c_bitline_f"] <= 0 or p["vdd"] <= 0:
        raise DramError("capacitances and vdd must be positive")
    return p


def activation_energy_j(p: dict[str, float]) -> tuple[float, str]:
    """Energy of one ACT+PRE of a row_bits row.

    With datasheet IDD values (TN-41-01):
        E = VDD [IDD0 tRC - (IDD3N tRAS + IDD2N (tRC - tRAS))]   per device,
    which includes sense amps, wordline, decoders and restore. Without them, only the bitline
    swing C_b VDD^2/2 per column plus a wordline term: a lower bound."""
    if p["idd0_a"] > 0:
        e = p["vdd"] * (p["idd0_a"] * p["t_rc_s"] - (p["idd3n_a"] * p["t_ras_s"] + p["idd2n_a"]
                                                     * (p["t_rc_s"] - p["t_ras_s"])))
        if e <= 0:
            raise DramError("IDD values give non-positive activation energy; check IDD0 > "
                            "IDD3N, IDD2N")
        return e * max(p["devices_per_row"], 1.0), "datasheet_idd"
    e = p["row_bits"] * p["c_bitline_f"] * p["vdd"] ** 2 / 2 + p["wordline_pj"] * 1e-12
    return e * max(p["devices_per_row"], 1.0), "bitline_lower_bound"


# ---- closed form --------------------------------------------------------------------------
def deviation(cell_volts: list[float], p: dict[str, float]) -> float:
    """Ideal bitline deviation from VDD/2 when len(cell_volts) cells share one bitline."""
    n = len(cell_volts)
    return p["c_cell_f"] * (sum(cell_volts) - n * p["vdd"] / 2) / (p["c_bitline_f"]
                                                                   + n * p["c_cell_f"])


def share_tau(n_cells: int, p: dict[str, float]) -> float:
    """RC time constant of charge sharing: access resistance (n in parallel) times the series
    combination of the cell capacitance and the bitline."""
    cc, cb = n_cells * p["c_cell_f"], p["c_bitline_f"]
    return (p["r_access_ohm"] / n_cells) * cc * cb / (cc + cb)


def margins(p: dict[str, float] | None = None) -> dict[str, Any]:
    """Worst-case ideal sense margins for a normal read and for TRA."""
    p = p or params()
    v1, v0 = p["vdd"], 0.0
    single = abs(deviation([v1], p))
    tra = abs(deviation([v1, v1, v0], p))
    return {"single_row_mv": single * 1e3, "tra_mv": tra * 1e3,
            "tra_over_single": tra / single,
            "single_tau_ns": share_tau(1, p) * 1e9, "tra_tau_ns": share_tau(3, p) * 1e9,
            "note": "TRA margin is smaller than a single read because the same signal charge "
                    "is spread over C_b + 3 C_c"}


# ---- Monte Carlo ---------------------------------------------------------------------------
def _cells(rng: np.random.Generator, bits: np.ndarray, p: dict[str, float]) -> tuple[
        np.ndarray, np.ndarray]:
    """Stored voltages after random time since refresh, and per-cell capacitances."""
    t = rng.uniform(0, p["t_since_refresh_max_s"], bits.shape)
    decay = np.exp(-t / p["retention_tau_s"]) if p["retention_tau_s"] > 0 else 1.0
    volts = np.where(bits == 1, p["vdd"] * decay, 0.0)
    caps = p["c_cell_f"] * (1 + p["sigma_c_cell"] * rng.standard_normal(bits.shape))
    return volts, np.clip(caps, 0.1 * p["c_cell_f"], None)


def _sense(rng: np.random.Generator, volts: np.ndarray, caps: np.ndarray,
           cb: np.ndarray, offset: np.ndarray, p: dict[str, float]) -> tuple[np.ndarray,
                                                                             np.ndarray]:
    """volts/caps shaped (n_cells, N). Returns sensed bits and the bitline deviation."""
    n = volts.shape[0]
    q = cb * p["vdd"] / 2 + np.sum(caps * volts, axis=0)
    ctot = cb + np.sum(caps, axis=0)
    v_final = q / ctot
    # incomplete sharing within t_share
    tau = (p["r_access_ohm"] / n) * (np.sum(caps, axis=0) * cb / ctot)
    settled = 1 - np.exp(-p["t_share_s"] / tau)
    dv = (v_final - p["vdd"] / 2) * settled
    dv = dv + rng.standard_normal(dv.shape) * np.sqrt(KB * p["temperature_k"] / ctot)
    return (dv + offset > 0).astype(np.int8), dv


def monte_carlo(op: str = "and", trials: int = 20_000, seed: int = 0,
                overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Bit error rate of one in-DRAM operation over `trials` random bit columns.

    Each trial is one bitline with its own capacitance and sense-amp offset, random operand
    bits and random time since refresh. Composite operations (and/or via copies + TRA, not
    via DCC, xor) chain the primitive steps so copy errors propagate."""
    if op not in OPERATIONS:
        raise DramError(f"op must be one of {OPERATIONS}")
    if not 100 <= trials <= 2_000_000:
        raise DramError("trials must be in [100, 2e6]")
    p = params(overrides)
    rng = np.random.default_rng(seed)
    cb = p["c_bitline_f"] * (1 + p["sigma_c_bitline"] * rng.standard_normal(trials))
    offset = p["sa_offset_sigma_v"] * rng.standard_normal(trials)
    a = rng.integers(0, 2, trials).astype(np.int8)
    b = rng.integers(0, 2, trials).astype(np.int8)
    c = rng.integers(0, 2, trials).astype(np.int8)
    margins_seen: list[np.ndarray] = []

    def read1(bits: np.ndarray) -> np.ndarray:
        v, cap = _cells(rng, bits[None, :], p)
        out, dv = _sense(rng, v, cap, cb, offset, p)
        margins_seen.append(np.abs(dv))
        return out

    def tra(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        bits = np.stack([x, y, z])
        v, cap = _cells(rng, bits, p)
        # operands were just written by RowClone, so they are freshly restored (no decay)
        v = np.where(bits == 1, p["vdd"], 0.0)
        out, dv = _sense(rng, v, cap, cb, offset, p)
        margins_seen.append(np.abs(dv))
        return out

    if op == "copy":
        got, want = read1(a), a
    elif op == "copy_psm":
        got, want = read1(a), a                      # sensed once, then moved digitally
    elif op == "maj":
        got, want = tra(a, b, c), ((a + b + c) >= 2).astype(np.int8)
    elif op in ("and", "or"):
        ctrl = np.zeros(trials, np.int8) if op == "and" else np.ones(trials, np.int8)
        ta, tb = read1(a), read1(b)                  # RowClone operands into T rows
        got = tra(ta, tb, ctrl)
        want = (a & b) if op == "and" else (a | b)
    elif op == "not":
        got, want = 1 - read1(a), 1 - a              # DCC captures negated bitline
        got = read1(got)                             # copy DCC out
    else:                                            # xor = (a & ~b) | (~a & b)
        na, nb = 1 - read1(a), 1 - read1(b)
        t1 = tra(read1(a), read1(nb), np.zeros(trials, np.int8))
        t2 = tra(read1(na), read1(b), np.zeros(trials, np.int8))
        got = tra(t1, t2, np.ones(trials, np.int8))
        want = a ^ b
    errors = int(np.sum(got != want))
    m = np.concatenate(margins_seen)
    ber = errors / trials
    return {
        "op": op, "trials": trials, "errors": errors, "bit_error_rate": ber,
        "ber_95_upper": _upper95(errors, trials),
        "row_errors_expected": ber * p["row_bits"],
        "min_margin_mv": float(m.min() * 1e3), "median_margin_mv": float(np.median(m) * 1e3),
        "fidelity": "behavioral", "model": "charge conservation + Gaussian variation",
    }


def _upper95(k: int, n: int) -> float:
    """One-sided 95 % upper bound on a rate (rule of three when k = 0, Wilson otherwise)."""
    if k == 0:
        return 3.0 / n
    phat, z = k / n, 1.645
    return (phat + z * z / (2 * n) + z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
            ) / (1 + z * z / n)


# ---- timing, energy, read disturb ----------------------------------------------------------
def timing_energy(op: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Latency and energy for one row-wide operation, and the CPU-side alternative (read the
    operand rows over the channel, compute, write back)."""
    if op not in OPERATIONS:
        raise DramError(f"op must be one of {OPERATIONS}")
    p = params(overrides)
    aap = p["t_ras_s"] + p["t_rp_s"]
    n_act = {"copy": 2, "and": 8, "or": 8, "maj": 1, "not": 4, "xor": 18, "copy_psm": 2}[op]
    rows_in = {"copy": 1, "copy_psm": 1, "not": 1, "maj": 3}.get(op, 2)
    bits = p["row_bits"] * max(p["devices_per_row"], 1.0)
    e_act, act_method = activation_energy_j(p)
    pim_j = n_act * e_act
    latency = AAPS[op] * aap
    if op == "copy_psm":
        latency = 2 * aap + bits / 64 * 5e-9                        # 64 b per tCCD-ish 5 ns
        pim_j += bits * p["io_pj_per_bit"] * 0.1e-12                # internal bus, assumed
    cpu_j = (rows_in + 1) * e_act + (rows_in + 1) * bits * p["io_pj_per_bit"] * 1e-12
    cpu_s = (rows_in + 1) * (aap + bits / 64 * 1.25e-9)             # 64 b per 1.25 ns burst
    return {"op": op, "aaps": AAPS[op], "activations": n_act, "latency_ns": latency * 1e9,
            "energy_nj": pim_j * 1e9, "cpu_energy_nj": cpu_j * 1e9,
            "cpu_latency_ns": cpu_s * 1e9, "energy_ratio_cpu_over_pim": cpu_j / pim_j,
            "row_bits": int(bits), "fidelity": "analytical",
            "activation_energy_nj": e_act * 1e9, "activation_energy_method": act_method,
            "assumptions": ("activation energy from datasheet IDD currents" if act_method ==
                            "datasheet_idd" else "activation energy = bitline swing only (lower "
                            "bound: no sense-amp, decoder or restore energy)") +
                           "; CPU path pays activations plus off-chip I/O per bit; no cache "
                           "reuse"}


def implied_activation_energy(op: str = "and") -> dict[str, Any]:
    """Diagnostic, not a fit: the activation energy at which timing_energy() would reproduce
    the published energy ratio, holding everything else fixed. Compare it with a datasheet
    IDD-derived value; it does not change the model."""
    ref = PUBLISHED[op]["energy_ratio_cpu_over_pim"]
    p = params()
    te = timing_energy(op)
    n, rows = te["activations"], {"copy": 1, "not": 1, "maj": 3}.get(op, 2) + 1
    io = rows * p["row_bits"] * p["io_pj_per_bit"] * 1e-12
    denom = ref * n - rows
    implied = io / denom if denom > 0 else math.inf
    return {"op": op, "published_ratio": ref, "source": PUBLISHED[op]["source"],
            "model_activation_nj": te["activation_energy_nj"],
            "implied_activation_nj": implied * 1e9,
            "factor": implied * 1e9 / te["activation_energy_nj"],
            "reading": "factor > 1: the model's activation energy is too low by about this "
                       "much if the rest of the model (I/O energy, op sequence) is right"}


def sense_time_ns(dv_v: float, p: dict[str, float] | None = None) -> float:
    """Regenerative latch time to amplify dv to VDD/2: tau ln(VDD/2 / |dv|)."""
    p = p or params()
    if dv_v == 0:
        return math.inf
    return p["tau_sense_s"] * math.log(p["vdd"] / 2 / abs(dv_v)) * 1e9


def read_disturb(ops_per_second: float, op: str = "and",
                 overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Activations a neighbour row sees per refresh window vs the hammer threshold."""
    p = params(overrides)
    acts = {"copy": 2, "and": 8, "or": 8, "maj": 1, "not": 4, "xor": 18, "copy_psm": 2}[op]
    # Ambit concentrates activations on the reserved T/C rows: assume half hit one victim.
    per_window = ops_per_second * acts * 0.5 * p["t_since_refresh_max_s"]
    return {"neighbour_activations_per_window": per_window,
            "hammer_threshold": p["hammer_threshold"],
            "at_risk": per_window >= p["hammer_threshold"],
            "max_safe_ops_per_s": p["hammer_threshold"] / (acts * 0.5
                                                           * p["t_since_refresh_max_s"]),
            "note": "Reserved compute rows are activated far more often than data rows; "
                    "mitigations (more refresh, row remapping) cost throughput."}


# Headline figures from the papers' abstracts. Their baselines (a full CPU + DDR3 system)
# differ from timing_energy()'s simple CPU path, so agreement within a factor of a few is
# the most that can be expected; a larger gap means the energy model is missing something.
PUBLISHED = {
    "and": {"energy_ratio_cpu_over_pim": 35.0, "rel_tol": 1.0,
            "source": "Seshadri et al., Ambit, MICRO 2017 (abstract, average over ops)"},
    "copy": {"energy_ratio_cpu_over_pim": 74.4, "rel_tol": 1.0,
             "source": "Seshadri et al., RowClone, MICRO 2013 (abstract, 4 KB copy)"},
}


def benchmark_review(op: str = "and") -> dict[str, Any]:
    """Run the review gate on timing_energy(op) against the published figure."""
    from chiplab.science import review
    ref = PUBLISHED[op]
    te = timing_energy(op)
    r = review(f"dram.{op}.energy_ratio", {"energy_ratio_cpu_over_pim": te[
        "energy_ratio_cpu_over_pim"]}, fidelity="analytical",
        provenance={"tool": "chiplab.pim.dram.timing_energy"},
        evaluate=lambda q: {"energy_ratio_cpu_over_pim": timing_energy(op, q)[
            "energy_ratio_cpu_over_pim"]},
        inputs={"c_bitline_f": PARAMS["c_bitline_f"][0], "io_pj_per_bit":
                PARAMS["io_pj_per_bit"][0]},
        benchmarks={"energy_ratio_cpu_over_pim": {
            "value": ref["energy_ratio_cpu_over_pim"], "rel_tol": ref["rel_tol"],
            "source": ref["source"]}})
    return r.as_dict() | {"model_value": te["energy_ratio_cpu_over_pim"]}


# ---- ngspice cross-check -------------------------------------------------------------------
def spice_check(cell_volts: list[float], overrides: dict[str, Any] | None = None,
                timeout: float = 60) -> dict[str, Any]:
    """Simulate the charge-sharing network (cells -> access R -> bitline) in ngspice and
    compare the settled bitline voltage and its time constant with the closed form."""
    p = params(overrides)
    if not 1 <= len(cell_volts) <= 4:
        raise DramError("1-4 cells")
    tool = registry.get("ngspice")
    if not tool.available:
        raise DramError("ngspice is not installed")
    n = len(cell_volts)
    tau = share_tau(n, p)
    t_end = 12 * tau
    lines = ["* chiplab dram charge sharing", f"Cb bl 0 {p['c_bitline_f']:.6e}"]
    for i, v in enumerate(cell_volts):
        lines += [f"Cc{i} c{i} 0 {p['c_cell_f']:.6e}", f"R{i} c{i} bl {p['r_access_ohm']:.6e}"]
    ics = " ".join([f"v(bl)={p['vdd'] / 2:.6f}"] + [f"v(c{i})={v:.6f}"
                                                     for i, v in enumerate(cell_volts)])
    lines += [f".ic {ics}", f".tran {t_end / 2000:.4e} {t_end:.4e} uic", ".end"]
    with tempfile.TemporaryDirectory(prefix="chiplab-dram-spice-") as tmp:
        (Path(tmp) / "x.cir").write_text("\n".join(lines) + "\n")
        proc = registry.run(tool, ["-b", "-r", "out.raw", "x.cir"], cwd=tmp, timeout=timeout)
        raw = Path(tmp) / "out.raw"
        if not raw.is_file() or raw.stat().st_size == 0:
            raise DramError("ngspice produced no output: " + (proc.stdout + proc.stderr)[-600:])
        sig = parse_raw(raw.read_bytes())
    t = np.real(sig["time"])
    vbl = np.real(sig.get("v(bl)", sig.get("bl")))
    v_end = float(vbl[-1])
    dv = deviation(cell_volts, p)
    target = p["vdd"] / 2 + dv
    # time to reach 63.2 % of the final deviation
    frac = (vbl - p["vdd"] / 2) / dv if dv else np.zeros_like(vbl)
    idx = int(np.argmax(frac >= 1 - math.exp(-1))) if dv else 0
    return {"cells_v": cell_volts, "spice_final_v": v_end, "closed_form_v": target,
            "abs_error_v": abs(v_end - target), "spice_tau_s": float(t[idx]) if dv else None,
            "closed_form_tau_s": tau,
            "agree": abs(v_end - target) < 1e-4 and (not dv or abs(t[idx] - tau) / tau < 0.05)}
