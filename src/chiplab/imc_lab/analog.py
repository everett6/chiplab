"""Resistive-crossbar analog MVM that can be built from catalogue parts on a breadboard or PCB.

Inputs x[c] in [0, 1] become row voltages x*Vread. Each signed weight w[r,c] in [-1, 1] is a
differential pair of resistors (G+ and G- columns). Each output column ends in an inverting
transimpedance amplifier (op-amp + Rf), so

    y_r = Vout(-)_r - Vout(+)_r = Rf * (Gmax - Gmin) * Vread * sum_c w[r,c] x[c]   (ideal)

What makes it buildable, and what is modelled:
- resistors snap to a standard E-series (E24 / E96) and get a Monte Carlo tolerance, which is the
  real weight-programming error of a hand-built array;
- wire/trace resistance between crosspoints (IR drop) and finite op-amp open-loop gain;
- the circuit is simulated by ngspice (``.op``) and, independently, by a NumPy nodal solver;
  the two must agree, so neither is trusted on its own;
- read energy uses a single-pole op-amp settling estimate (labelled as an estimate).
Not modelled: op-amp offset/bias current, noise, slew, temperature drift. A bench measurement
through the learning loop is how those enter.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from chiplab.tools import registry
from chiplab.tools.spice import parse_raw

E24 = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7,
       5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1]
E96 = [1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30, 1.33, 1.37, 1.40,
       1.43, 1.47, 1.50, 1.54, 1.58, 1.62, 1.65, 1.69, 1.74, 1.78, 1.82, 1.87, 1.91, 1.96, 2.00,
       2.05, 2.10, 2.15, 2.21, 2.26, 2.32, 2.37, 2.43, 2.49, 2.55, 2.61, 2.67, 2.74, 2.80, 2.87,
       2.94, 3.01, 3.09, 3.16, 3.24, 3.32, 3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12,
       4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99, 5.11, 5.23, 5.36, 5.49, 5.62, 5.76, 5.90,
       6.04, 6.19, 6.34, 6.49, 6.65, 6.81, 6.98, 7.15, 7.32, 7.50, 7.68, 7.87, 8.06, 8.25, 8.45,
       8.66, 8.87, 9.09, 9.31, 9.53, 9.76]
SERIES = {"E24": E24, "E96": E96, "ideal": None}

# name -> (default, min, max, unit)
PARAMS: dict[str, tuple[float, float, float, str]] = {
    "vread": (0.2, 0.01, 2.0, "V"),
    "r_on_ohm": (10e3, 100.0, 1e7, "ohm"),       # resistor for |w| = 1 (Gmax)
    "r_off_ohm": (1e6, 1e3, 1e9, "ohm"),         # resistor for w = 0 (Gmin); 0 means omit
    "rf_ohm": (0.0, 0.0, 1e8, "ohm"),            # 0 -> auto: full scale ~ 1 V
    "wire_ohm": (0.05, 0.0, 100.0, "ohm"),       # per crosspoint segment
    "opamp_gain": (1e5, 10.0, 1e8, "V/V"),
    "opamp_gbw_hz": (1e6, 1e3, 1e10, "Hz"),
    "tolerance_pct": (1.0, 0.0, 20.0, "%"),
    "supply_v": (5.0, 1.0, 36.0, "V"),            # op-amp rail (single-ended swing check)
}
LIMITS = {"rows": (1, 8), "cols": (1, 8)}


class AnalogCimError(ValueError):
    pass


def validate(spec: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, (lo, hi) in LIMITS.items():
        v = spec.get(k, 4)
        if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
            raise AnalogCimError(f"{k} must be an int in [{lo}, {hi}]")
        out[k] = v
    for k, (d, lo, hi, _u) in PARAMS.items():
        v = float(spec.get(k, d))
        if not (math.isfinite(v) and lo <= v <= hi):
            raise AnalogCimError(f"{k} must be in [{lo}, {hi}]")
        out[k] = v
    if out["r_off_ohm"] <= out["r_on_ohm"]:
        raise AnalogCimError("r_off_ohm must exceed r_on_ohm")
    series = spec.get("series", "E96")
    if series not in SERIES:
        raise AnalogCimError(f"series must be one of {sorted(SERIES)}")
    out["series"] = series
    if out["rf_ohm"] == 0:
        out["rf_ohm"] = 1.0 / (out["cols"] * (1 / out["r_on_ohm"] - 1 / out["r_off_ohm"])
                               * out["vread"])
    return out


def snap(r: float, series: str) -> float:
    """Nearest standard value (log distance) in the series; 'ideal' leaves it unchanged."""
    vals = SERIES[series]
    if vals is None:
        return r
    dec = 10 ** math.floor(math.log10(r))
    cands = [v * dec * k for v in vals for k in (0.1, 1, 10)]
    return min(cands, key=lambda c: abs(math.log(c / r)))


def resistors(p: dict[str, Any], w: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Nominal-then-snapped-then-toleranced resistances for the + and - arrays, (rows, cols)."""
    gmin, gmax = 1 / p["r_off_ohm"], 1 / p["r_on_ohm"]
    rng = np.random.default_rng(seed)
    out = []
    for sign in (1, -1):
        g = gmin + np.clip(sign * w, 0, 1) * (gmax - gmin)
        r = np.vectorize(lambda x: snap(1 / x, p["series"]))(g)
        r = r * (1 + rng.normal(0, p["tolerance_pct"] / 100 / 3, r.shape))   # tol = 3 sigma
        out.append(r)
    return out[0], out[1]


def ideal(p: dict[str, Any], w: np.ndarray, x: np.ndarray) -> np.ndarray:
    gmin, gmax = 1 / p["r_off_ohm"], 1 / p["r_on_ohm"]
    return p["rf_ohm"] * (gmax - gmin) * p["vread"] * (w @ x)


def _nodes(p: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    """Node names: input line i at crosspoint c of output line (r, s): a_{i}_{r}{s};
    output line (r, s) at input i: b_{r}{s}_{i}; TIA inverting input m_{r}{s}."""
    names = []
    for s in "pn":
        for r in range(p["rows"]):
            for i in range(p["cols"]):
                names += [f"a_{i}_{r}{s}", f"b_{r}{s}_{i}"]
            names.append(f"m_{r}{s}")
    return names, {n: k for k, n in enumerate(names)}


def _elements(p: dict[str, Any], rp: np.ndarray, rn: np.ndarray, x: np.ndarray):
    """Resistors (n1, n2, ohm) and sources, shared by the netlist and the nodal solver.

    Each input line runs past every output line in order (p rows then n rows), so its wire
    segments chain in_i -> a_i_0p -> a_i_1p ... ; each output line chains b_rs_0 -> ... -> m_rs.
    """
    res: list[tuple[str, str, float]] = []
    wire = max(p["wire_ohm"], 1e-6)
    order = [(r, s) for s in "pn" for r in range(p["rows"])]
    for i in range(p["cols"]):
        prev = f"in_{i}"
        for r, s in order:
            res.append((prev, f"a_{i}_{r}{s}", wire))
            prev = f"a_{i}_{r}{s}"
    for r, s in order:
        arr = rp if s == "p" else rn
        for i in range(p["cols"]):
            res.append((f"a_{i}_{r}{s}", f"b_{r}{s}_{i}", float(arr[r, i])))
            nxt = f"b_{r}{s}_{i + 1}" if i + 1 < p["cols"] else f"m_{r}{s}"
            res.append((f"b_{r}{s}_{i}", nxt, wire))
        res.append((f"m_{r}{s}", f"o_{r}{s}", p["rf_ohm"]))
    vin = {f"in_{i}": float(x[i]) * p["vread"] for i in range(p["cols"])}
    return res, vin, order


def solve_nodal(p: dict[str, Any], rp: np.ndarray, rn: np.ndarray, x: np.ndarray
                ) -> dict[str, Any]:
    """Modified nodal analysis in NumPy (op-amps as VCVS o = -A*m)."""
    res, vin, order = _elements(p, rp, rn, x)
    names, idx = _nodes(p)
    n = len(names)
    a = np.zeros((n, n))
    b = np.zeros(n)
    gain = p["opamp_gain"]

    def volt(node: str) -> tuple[int | None, float, float]:
        """(unknown index, coefficient, constant) such that V(node) = coef*x[idx] + const."""
        if node in vin:
            return None, 0.0, vin[node]
        if node.startswith("o_"):
            return idx["m_" + node[2:]], -gain, 0.0
        return idx[node], 1.0, 0.0

    for n1, n2, r in res:
        g = 1.0 / r
        for here, there in ((n1, n2), (n2, n1)):
            if here in vin or here.startswith("o_"):
                continue                        # source-driven nodes need no KCL
            k = idx[here]
            a[k, k] -= g
            j, coef, const = volt(there)
            if j is None:
                b[k] -= g * const
            else:
                a[k, j] += g * coef
    v = np.linalg.solve(a, b)
    vo = {f"{r}{s}": -gain * v[idx[f"m_{r}{s}"]] for r, s in order}
    power = sum((vin[n1] - (v[idx[n2]] if n2 in idx else 0.0)) * vin[n1] / r
                for n1, n2, r in res if n1 in vin)
    y = np.array([vo[f"{r}n"] - vo[f"{r}p"] for r in range(p["rows"])])
    return {"y": y, "vout": vo, "input_power_w": float(power)}


def netlist(p: dict[str, Any], rp: np.ndarray, rn: np.ndarray, x: np.ndarray) -> str:
    res, vin, order = _elements(p, rp, rn, x)
    lines = [f"* ChipLab IMC lab: {p['rows']}x{p['cols']} differential resistive crossbar"]
    lines += [f"V{n} {n} 0 DC {v:.9g}" for n, v in vin.items()]
    lines += [f"R{k} {n1} {n2} {r:.9g}" for k, (n1, n2, r) in enumerate(res)]
    lines += [f"E{r}{s} o_{r}{s} 0 0 m_{r}{s} {p['opamp_gain']:.9g}" for r, s in order]
    lines += [".op", ".end"]
    return "\n".join(lines) + "\n"


def run_ngspice(text: str, timeout: float = 60) -> dict[str, float]:
    tool = registry.get("ngspice")
    if not tool.available:
        raise AnalogCimError("ngspice is not installed")
    with tempfile.TemporaryDirectory(prefix="chiplab-imc-spice-") as tmp:
        (Path(tmp) / "x.cir").write_text(text)
        proc = registry.run(tool, ["-b", "-r", "out.raw", "x.cir"], cwd=tmp, timeout=timeout)
        raw = Path(tmp) / "out.raw"
        if not raw.is_file() or raw.stat().st_size == 0:
            raise AnalogCimError("ngspice produced no output: " + (proc.stdout + proc.stderr)[-600:])
        sig = parse_raw(raw.read_bytes())
    return {k: float(v[0]) for k, v in sig.items()}


def _settle_s(p: dict[str, Any], rp: np.ndarray, rn: np.ndarray) -> float:
    """0.1 % settling of a single-pole op-amp at noise gain 1 + Rf * sum(G): ~6.9 tau."""
    g_line = max(np.sum(1 / rp, axis=1).max(), np.sum(1 / rn, axis=1).max())
    noise_gain = 1 + p["rf_ohm"] * g_line
    return 6.9 * noise_gain / (2 * math.pi * p["opamp_gbw_hz"])


def bom(p: dict[str, Any], rp: np.ndarray, rn: np.ndarray) -> list[dict[str, Any]]:
    counts: dict[float, int] = {}
    for r in np.concatenate([rp.ravel(), rn.ravel()]):
        v = snap(float(r), p["series"]) if p["series"] != "ideal" else round(float(r), 1)
        counts[v] = counts.get(v, 0) + 1
    items = [{"part": f"resistor {v:.4g} ohm {p['series']} {p['tolerance_pct']:g}%",
              "qty": q} for v, q in sorted(counts.items())]
    channels = 2 * p["rows"]
    items += [{"part": f"Rf {snap(p['rf_ohm'], 'E96'):.4g} ohm 0.1%", "qty": channels},
              {"part": "quad op-amp (rail-to-rail, GBW >= "
                       f"{p['opamp_gbw_hz'] / 1e6:g} MHz)", "qty": math.ceil(channels / 4)},
              {"part": "differential ADC input or difference amplifier per output",
               "qty": p["rows"]},
              {"part": f"DAC or divider per input (0..{p['vread']:g} V)", "qty": p["cols"]}]
    return items


def evaluate(spec: dict[str, Any], w: Any = None, x: Any = None, seed: int = 0,
             use_ngspice: bool = True) -> dict[str, Any]:
    """Build one crossbar instance and compare ngspice, the nodal solver and the ideal MVM."""
    p = validate(spec)
    rng = np.random.default_rng(seed + 1000)
    w = np.asarray(w if w is not None else rng.uniform(-1, 1, (p["rows"], p["cols"])), float)
    x = np.asarray(x if x is not None else rng.uniform(0, 1, p["cols"]), float)
    if w.shape != (p["rows"], p["cols"]) or x.shape != (p["cols"],):
        raise AnalogCimError("w must be (rows, cols) and x (cols,)")
    if np.abs(w).max() > 1 or x.min() < 0 or x.max() > 1:
        raise AnalogCimError("w must be in [-1, 1] and x in [0, 1]")
    rp, rn = resistors(p, w, seed)
    y_ideal = ideal(p, w, x)
    nod = solve_nodal(p, rp, rn, x)
    full_scale = ideal(p, np.ones((1, p["cols"])), np.ones(p["cols"]))[0]
    out: dict[str, Any] = {
        "params": p, "w": w.round(6).tolist(), "x": x.round(6).tolist(),
        "y_ideal_v": y_ideal.round(9).tolist(), "y_nodal_v": nod["y"].round(9).tolist(),
        "full_scale_v": float(full_scale),
        "rms_error_pct_fs": float(np.sqrt(np.mean((nod["y"] - y_ideal) ** 2)) / full_scale * 100),
        "max_abs_output_v": float(max(abs(v) for v in nod["vout"].values())),
    }
    out["swing_ok"] = out["max_abs_output_v"] < p["supply_v"] / 2 - 0.1
    settle = _settle_s(p, rp, rn)
    out["static_power_w"] = nod["input_power_w"]
    out["settle_time_s"] = settle
    out["energy_per_mvm_j"] = nod["input_power_w"] * settle
    out["energy_note"] = ("input-side crossbar power x single-pole 0.1% settling time; op-amp "
                          "quiescent power excluded (estimate)")
    if use_ngspice:
        text = netlist(p, rp, rn, x)
        sig = run_ngspice(text)
        y_sp = np.array([sig[f"v(o_{r}n)"] - sig[f"v(o_{r}p)"] for r in range(p["rows"])])
        out["y_ngspice_v"] = y_sp.round(9).tolist()
        out["ngspice_vs_nodal_max_v"] = float(np.abs(y_sp - nod["y"]).max())
        out["solvers_agree"] = out["ngspice_vs_nodal_max_v"] <= 1e-6 * max(1.0, full_scale)
        out["netlist"] = text
    out["bom"] = bom(p, rp, rn)
    return out


def monte_carlo(spec: dict[str, Any], trials: int = 50, seed: int = 0) -> dict[str, Any]:
    """Error spread over resistor tolerance draws (nodal solver; ngspice-checked in evaluate)."""
    if not 1 <= trials <= 1000:
        raise AnalogCimError("trials must be in [1, 1000]")
    errs = [evaluate(spec, seed=seed + k, use_ngspice=False)["rms_error_pct_fs"]
            for k in range(trials)]
    return {"trials": trials, "rms_error_pct_fs_mean": float(np.mean(errs)),
            "rms_error_pct_fs_p95": float(np.percentile(errs, 95))}


def predict(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Fast numbers for ranking: one nodal solve plus a 20-draw tolerance spread."""
    e = evaluate(spec, seed=0, use_ngspice=False)
    mc = monte_carlo(spec, trials=20)
    return {
        "rms_error_pct_fs": {"value": mc["rms_error_pct_fs_mean"], "unit": "%FS",
                             "provenance": "nodal solve + tolerance Monte Carlo (20 draws)"},
        "static_power_w": {"value": e["static_power_w"], "unit": "W",
                           "provenance": "nodal solve, random w/x"},
        "energy_per_mvm_j": {"value": e["energy_per_mvm_j"], "unit": "J",
                             "provenance": "estimate: power x single-pole settling"},
        "settle_time_s": {"value": e["settle_time_s"], "unit": "s",
                          "provenance": "estimate: single-pole op-amp"},
        "macs_per_mvm": {"value": e["params"]["rows"] * e["params"]["cols"], "unit": "MAC",
                         "provenance": "definition"},
    }
