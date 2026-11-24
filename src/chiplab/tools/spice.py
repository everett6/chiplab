"""Circuit Studio: fixed SPICE templates simulated by the real ngspice.

Each template is a list of devices built from validated numeric parameters only; the same
device list produces the ngspice netlist and the schematic (Xschem or generated SVG), so the
picture always matches what was simulated. ngspice runs in batch mode (``-b -r out.raw``) in a
fresh temporary directory; its binary raw file is parsed with numpy.

Transistor models are generic level-1 MOSFET cards (roughly a 45-65 nm-class bulk process,
long-channel equations). They show circuit behaviour (charge sharing, latch regeneration,
gain-cell read disturb, charge-domain MAC linearity), not sign-off numbers for any PDK.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from chiplab.tools import registry

MODELS = """\
.model nch nmos level=1 vto=0.42 kp=220u lambda=0.08 gamma=0.35 phi=0.8 cgso=0.25n cgdo=0.25n
+ cj=0.8m
.model pch pmos level=1 vto=-0.42 kp=90u lambda=0.1 gamma=0.35 phi=0.8 cgso=0.25n cgdo=0.25n
+ cj=0.8m
"""

P = dict[str, float]

# ---------------------------------------------------------------- parameter specs
# name -> (default, min, max, unit, label)
SPECS: dict[str, dict[str, tuple[float, float, float, str, str]]] = {
    "dram_sense_amp": {
        "vdd": (1.1, 0.6, 1.8, "V", "Array supply VDD"),
        "wl_boost": (0.6, 0.0, 1.5, "V", "Word-line boost above VDD"),
        "cs_ff": (20.0, 2.0, 100.0, "fF", "Cell storage capacitance"),
        "cbl_ff": (80.0, 5.0, 500.0, "fF", "Bit-line capacitance"),
        "bit": (1, 0, 1, "", "Stored bit"),
        "t_sense_ns": (6.0, 3.5, 20.0, "ns", "Sense-amp enable time"),
    },
    "dram_1t1c": {
        "vdd": (1.1, 0.6, 1.8, "V", "Array supply VDD"),
        "wl_boost": (0.6, 0.0, 1.5, "V", "Word-line boost above VDD"),
        "cs_ff": (20.0, 2.0, 100.0, "fF", "Cell storage capacitance"),
        "cbl_ff": (80.0, 5.0, 500.0, "fF", "Bit-line capacitance"),
        "bit": (1, 0, 1, "", "Bit written"),
        "leak_gohm": (50.0, 0.1, 10000.0, "GΩ", "Storage-node leakage resistance"),
        "hold_ns": (20.0, 2.0, 200.0, "ns", "Hold time before read"),
    },
    "gain_cell_2t": {
        "vdd": (1.1, 0.6, 1.8, "V", "Supply VDD"),
        "wwl_boost": (0.5, 0.0, 1.5, "V", "Write word-line boost above VDD"),
        "csn_ff": (2.0, 0.2, 20.0, "fF", "Storage-node capacitance"),
        "crbl_ff": (15.0, 1.0, 200.0, "fF", "Read bit-line capacitance"),
        "bit": (1, 0, 1, "", "Bit written"),
    },
    "ecore_mac": {
        "vdd": (1.1, 0.6, 1.8, "V", "Array supply VDD"),
        "wl_boost": (0.7, 0.0, 1.5, "V", "Word-line boost above VDD"),
        "cs_ff": (10.0, 1.0, 100.0, "fF", "Cell capacitance per weight bit"),
        "cbl_ff": (40.0, 5.0, 500.0, "fF", "Column (bit-line) capacitance"),
        **{f"w{i}": (float(i % 2 == 0 or i == 3), 0, 1, "", f"Weight bit w{i}") for i in range(4)},
        **{f"x{i}": (float(i < 3), 0, 1, "", f"Input bit x{i}") for i in range(4)},
    },
}

TITLES = {
    "dram_sense_amp": ("DRAM sense amplifier",
                       "1T1C cell + cross-coupled latch sense amp with equalizer: charge "
                       "sharing, regeneration and cell restore."),
    "dram_1t1c": ("1T1C DRAM cell", "Write, hold with leakage, then read by charge sharing "
                  "onto a precharged bit line (no sense amp)."),
    "gain_cell_2t": ("2T gain cell", "Write transistor + read transistor; non-destructive read "
                     "discharges the read bit line when the stored node is high."),
    "ecore_mac": ("eCore charge-domain MAC column",
                  "Four DRAM-style cells hold weight bits; input bits open the word lines and "
                  "the shared column settles to a charge-weighted popcount of w AND x."),
}

BIT_KEYS = {"bit", "w0", "w1", "w2", "w3", "x0", "x1", "x2", "x3"}


class CircuitError(ValueError):
    pass


def templates() -> list[dict[str, Any]]:
    return [{"key": k, "title": TITLES[k][0], "description": TITLES[k][1],
             "params": [{"name": n, "default": d, "min": lo, "max": hi, "unit": u, "label": lab,
                         "bit": n in BIT_KEYS}
                        for n, (d, lo, hi, u, lab) in spec.items()]}
            for k, spec in SPECS.items()]


def validate(template: str, params: dict[str, Any] | None) -> P:
    if template not in SPECS:
        raise CircuitError(f"unknown circuit template {template!r}; "
                           f"choose one of {sorted(SPECS)}")
    out: P = {}
    params = params or {}
    unknown = set(params) - set(SPECS[template])
    if unknown:
        raise CircuitError(f"unknown parameters for {template}: {sorted(unknown)}")
    for name, (default, lo, hi, unit, _) in SPECS[template].items():
        raw = params.get(name, default)
        try:
            v = float(raw)
        except (TypeError, ValueError) as exc:
            raise CircuitError(f"{name} must be a number") from exc
        if not math.isfinite(v) or v < lo or v > hi:
            raise CircuitError(f"{name}={raw} outside {lo:g}..{hi:g} {unit}".rstrip())
        if name in BIT_KEYS:
            if v not in (0.0, 1.0):
                raise CircuitError(f"{name} must be 0 or 1")
        out[name] = v
    return out


# ---------------------------------------------------------------- device lists
def _m(name: str, kind: str, d: str, g: str, s: str, b: str, w: float, l: float,
       pos: tuple[float, float]) -> dict[str, Any]:
    return {"name": name, "kind": kind, "pins": {"d": d, "g": g, "s": s, "b": b},
            "w_um": w, "l_um": l, "pos": pos}


def _c(name: str, p: str, m: str, ff: float, pos: tuple[float, float]) -> dict[str, Any]:
    return {"name": name, "kind": "cap", "pins": {"p": p, "m": m}, "value": f"{ff:g}f",
            "pos": pos}


def _r(name: str, p: str, m: str, value: str, pos: tuple[float, float]) -> dict[str, Any]:
    return {"name": name, "kind": "res", "pins": {"p": p, "m": m}, "value": value, "pos": pos}


def _v(name: str, p: str, spec: str, pos: tuple[float, float], m: str = "0") -> dict[str, Any]:
    return {"name": name, "kind": "vsrc", "pins": {"p": p, "m": m}, "value": spec, "pos": pos}


def _pulse(v0: float, v1: float, t_on: float, t_off: float | None = None,
           edge: float = 0.1) -> str:
    pts = [(0.0, v0), (t_on, v0), (t_on + edge, v1)]
    if t_off is not None:
        pts += [(t_off, v1), (t_off + edge, v0)]
    return "PWL(" + " ".join(f"{t:g}n {v:.4g}" for t, v in pts) + ")"


def _sense_amp(p: P) -> dict[str, Any]:
    vdd, vpp, half = p["vdd"], p["vdd"] + p["wl_boost"], p["vdd"] / 2
    t_eq, t_wl, t_se = 1.0, 2.0, p["t_sense_ns"]
    t_end = t_se + 12.0
    dev = [
        _v("VDD", "vdd", f"DC {vdd:g}", (0, 6)), _v("VPRE", "vpre", f"DC {half:g}", (1, 6)),
        _v("VEQ", "eq", _pulse(vdd, 0, t_eq), (2, 6)),
        _v("VWL", "wl", _pulse(0, vpp, t_wl), (3, 6)),
        _v("VSE", "se", _pulse(0, vdd, t_se), (4, 6)),
        _v("VSEB", "seb", _pulse(vdd, 0, t_se), (5, 6)),
        _m("MA", "nmos", "bl", "wl", "sn", "0", 0.12, 0.06, (0, 1)),
        _c("CS", "sn", "0", p["cs_ff"], (0, 3)),
        _c("CBL", "bl", "0", p["cbl_ff"], (1, 3)),
        _c("CBLB", "blb", "0", p["cbl_ff"], (5, 3)),
        _m("MEQ1", "nmos", "bl", "eq", "vpre", "0", 0.3, 0.06, (2, 0)),
        _m("MEQ2", "nmos", "blb", "eq", "vpre", "0", 0.3, 0.06, (4, 0)),
        _m("MEQ3", "nmos", "bl", "eq", "blb", "0", 0.3, 0.06, (3, 0)),
        _m("MP1", "pmos", "bl", "blb", "sap", "vdd", 0.4, 0.06, (2, 2)),
        _m("MP2", "pmos", "blb", "bl", "sap", "vdd", 0.4, 0.06, (4, 2)),
        _m("MN1", "nmos", "bl", "blb", "san", "0", 0.3, 0.06, (2, 4)),
        _m("MN2", "nmos", "blb", "bl", "san", "0", 0.3, 0.06, (4, 4)),
        _m("MSP", "pmos", "sap", "seb", "vdd", "vdd", 1.2, 0.06, (3, 1)),
        _m("MSN", "nmos", "san", "se", "0", "0", 1.0, 0.06, (3, 5)),
        _m("MPS", "nmos", "sap", "eq", "vpre", "0", 0.3, 0.06, (1, 1)),
        _m("MNS", "nmos", "san", "eq", "vpre", "0", 0.3, 0.06, (1, 5)),
    ]
    ic = {"bl": half, "blb": half, "sn": vdd * p["bit"], "sap": half, "san": half}
    return {"devices": dev, "ic": ic, "t_end": t_end, "step": 0.01,
            "events": [("equalize off", t_eq), ("word line on", t_wl), ("sense enable", t_se)],
            "primary": ["bl", "blb", "sn", "wl", "se"],
            "currents": ["i(@ma[id])", "i(@msn[id])", "i(@msp[id])", "i(vdd)"],
            "t_sense": t_se}


def _one_t_one_c(p: P) -> dict[str, Any]:
    vdd, vpp, half = p["vdd"], p["vdd"] + p["wl_boost"], p["vdd"] / 2
    t_w1, t_w2 = 1.0, 4.0
    t_r = t_w2 + p["hold_ns"]
    t_pre_off = t_r - 1.0
    t_end = t_r + 6.0
    dev = [
        _v("VDD", "vdd", f"DC {vdd:g}", (0, 5)), _v("VPRE", "vpre", f"DC {half:g}", (1, 5)),
        _v("VDATA", "data", f"DC {vdd * p['bit']:g}", (2, 5)),
        _v("VWE", "we", _pulse(0, vpp, 0.5, t_w2 - 0.5), (3, 5)),
        _v("VPC", "pc", _pulse(0, vpp, t_w2, t_pre_off), (4, 5)),
        _v("VWL", "wl", "PWL(" + " ".join(f"{t:g}n {v:.4g}" for t, v in [
            (0, 0), (t_w1, 0), (t_w1 + 0.1, vpp), (t_w2, vpp), (t_w2 + 0.1, 0),
            (t_r, 0), (t_r + 0.1, vpp)]) + ")", (5, 5)),
        _m("MWR", "nmos", "bl", "we", "data", "0", 0.6, 0.06, (1, 0)),
        _m("MPC", "nmos", "bl", "pc", "vpre", "0", 0.4, 0.06, (3, 0)),
        _m("MA", "nmos", "bl", "wl", "sn", "0", 0.12, 0.06, (2, 1)),
        _c("CS", "sn", "0", p["cs_ff"], (2, 3)),
        _r("RLEAK", "sn", "0", f"{p['leak_gohm']:g}G", (3, 3)),
        _c("CBL", "bl", "0", p["cbl_ff"], (1, 3)),
    ]
    ic = {"bl": 0.0, "sn": vdd * (1 - p["bit"])}
    return {"devices": dev, "ic": ic, "t_end": t_end, "step": 0.02,
            "events": [("write", t_w1), ("hold", t_w2), ("precharge off", t_pre_off),
                       ("read", t_r)],
            "primary": ["bl", "sn", "wl"],
            "currents": ["i(@ma[id])", "i(vdd)"], "t_read": t_r, "vpre": half}


def _gain_cell(p: P) -> dict[str, Any]:
    vdd, vpp = p["vdd"], p["vdd"] + p["wwl_boost"]
    t_w, t_wo, t_pre, t_r = 1.0, 3.0, 4.0, 6.0
    t_end = 12.0
    dev = [
        _v("VDD", "vdd", f"DC {vdd:g}", (0, 5)),
        _v("VWBL", "wbl", f"DC {vdd * p['bit']:g}", (1, 5)),
        _v("VWWL", "wwl", _pulse(0, vpp, t_w, t_wo), (2, 5)),
        _v("VPREB", "preb", _pulse(vdd, 0, 0.2, t_pre), (3, 5)),
        _v("VRWL", "rwl", _pulse(vdd, 0, t_r), (4, 5)),
        _m("MW", "nmos", "wbl", "wwl", "sn", "0", 0.12, 0.06, (0, 1)),
        _m("MR", "nmos", "rbl", "sn", "rwl", "0", 0.2, 0.06, (2, 1)),
        _m("MPRE", "pmos", "rbl", "preb", "vdd", "vdd", 0.4, 0.06, (2, 0)),
        _c("CSN", "sn", "0", p["csn_ff"], (1, 3)),
        _c("CRBL", "rbl", "0", p["crbl_ff"], (3, 3)),
    ]
    ic = {"sn": vdd * (1 - p["bit"]), "rbl": 0.0}
    return {"devices": dev, "ic": ic, "t_end": t_end, "step": 0.01,
            "events": [("write", t_w), ("write off", t_wo), ("precharge off", t_pre),
                       ("read (RWL low)", t_r)],
            "primary": ["sn", "rbl", "wwl", "rwl"],
            "currents": ["i(@mr[id])", "i(@mw[id])", "i(vdd)"], "t_read": t_r}


def _ecore(p: P) -> dict[str, Any]:
    vdd, vpp = p["vdd"], p["vdd"] + p["wl_boost"]
    t_rst, t_wl = 1.0, 2.0
    t_end = 8.0
    dev = [
        _v("VRST", "rst", _pulse(vdd, 0, t_rst), (0, 5)),
        _m("MRST", "nmos", "col", "rst", "0", "0", 0.4, 0.06, (0, 0)),
        _c("CCOL", "col", "0", p["cbl_ff"], (0, 3)),
    ]
    ic = {"col": 0.0}
    for i in range(4):
        x = p[f"x{i}"]
        dev += [
            _v(f"VWL{i}", f"wl{i}", _pulse(0, vpp * x, t_wl), (i + 1, 5)),
            _m(f"MA{i}", "nmos", "col", f"wl{i}", f"sn{i}", "0", 0.2, 0.06, (i + 1, 1)),
            _c(f"CS{i}", f"sn{i}", "0", p["cs_ff"], (i + 1, 3)),
        ]
        ic[f"sn{i}"] = vdd * p[f"w{i}"]
    return {"devices": dev, "ic": ic, "t_end": t_end, "step": 0.01,
            "events": [("column reset off", t_rst), ("inputs drive word lines", t_wl)],
            "primary": ["col", "sn0", "sn1", "sn2", "sn3"],
            "currents": [f"i(@ma{i}[id])" for i in range(4)]}


BUILDERS = {"dram_sense_amp": _sense_amp, "dram_1t1c": _one_t_one_c,
            "gain_cell_2t": _gain_cell, "ecore_mac": _ecore}


def build(template: str, params: dict[str, Any] | None = None) -> tuple[P, dict[str, Any]]:
    p = validate(template, params)
    return p, BUILDERS[template](p)


def netlist(template: str, p: P, circ: dict[str, Any]) -> str:
    lines = [f"* ChipLab Circuit Studio: {TITLES[template][0]}",
             "* " + ", ".join(f"{k}={v:g}" for k, v in p.items()), MODELS.rstrip()]
    for d in circ["devices"]:
        pins = d["pins"]
        if d["kind"] in ("nmos", "pmos"):
            model = "nch" if d["kind"] == "nmos" else "pch"
            lines.append(f"{d['name']} {pins['d']} {pins['g']} {pins['s']} {pins['b']} {model} "
                         f"w={d['w_um']:g}u l={d['l_um']:g}u")
        else:
            lines.append(f"{d['name']} {pins['p']} {pins['m']} {d['value']}")
    lines.append(".ic " + " ".join(f"v({n})={v:.4g}" for n, v in circ["ic"].items()))
    lines.append(".options savecurrents")
    lines.append(f".tran {circ['step']:g}n {circ['t_end']:g}n uic")
    lines.append(".end")
    return "\n".join(lines) + "\n"


def nets(circ: dict[str, Any]) -> list[str]:
    seen: dict[str, None] = {}
    for d in circ["devices"]:
        for n in d["pins"].values():
            if n != "0":
                seen[n] = None
    return list(seen)


# ---------------------------------------------------------------- raw file
def parse_raw(data: bytes) -> dict[str, np.ndarray]:
    """Parse the first plot of an ngspice raw file (binary or ASCII, real or complex)."""
    pos = 0
    header: dict[str, str] = {}
    names: list[str] = []
    mode = None
    while pos < len(data):
        end = data.index(b"\n", pos)
        line = data[pos:end].decode("latin-1")
        pos = end + 1
        if line.startswith("Variables:"):
            nvar = int(header["no. variables"])
            for _ in range(nvar):
                end = data.index(b"\n", pos)
                parts = data[pos:end].decode("latin-1").split()
                pos = end + 1
                names.append(parts[1])
            continue
        if line.startswith("Binary:"):
            mode = "binary"
            break
        if line.startswith("Values:"):
            mode = "ascii"
            break
        if ":" in line:
            k, v = line.split(":", 1)
            header[k.strip().lower()] = v.strip()
    if mode is None or not names:
        raise CircuitError("ngspice raw file has no data")
    nvar, npts = len(names), int(header["no. points"])
    cplx = "complex" in header.get("flags", "")
    if mode == "binary":
        width = nvar * (2 if cplx else 1)
        arr = np.frombuffer(data, dtype="<f8", count=npts * width, offset=pos)
        arr = arr.reshape(npts, width)
        if cplx:
            arr = arr[:, 0::2]
    else:
        nums = []
        for tok in data[pos:].decode("latin-1").split():
            if "," in tok:
                tok = tok.split(",")[0]
            try:
                nums.append(float(tok))
            except ValueError:
                continue
        # ASCII rows are "index value" then nvar-1 values
        per = nvar + 1
        arr = np.array(nums[: npts * per]).reshape(npts, per)[:, 1:]
    return {n.lower(): arr[:, i].copy() for i, n in enumerate(names)}


# ---------------------------------------------------------------- metrics
def _at(t: np.ndarray, y: np.ndarray, when_ns: float) -> float:
    return float(np.interp(when_ns * 1e-9, t, y))


def metrics(template: str, p: P, circ: dict[str, Any], sig: dict[str, np.ndarray]
            ) -> dict[str, Any]:
    t = sig["time"]
    v = lambda n: sig[f"v({n})"]  # noqa: E731
    vdd = p["vdd"]
    out: dict[str, Any] = {}
    if template == "dram_sense_amp":
        ts = circ["t_sense"]
        dv = _at(t, v("bl") - v("blb"), ts)
        split = v("bl") - v("blb")
        after = t >= ts * 1e-9
        done = np.nonzero(after & (np.abs(split) >= 0.9 * vdd))[0]
        ideal = (p["bit"] - 0.5) * vdd * p["cs_ff"] / (p["cs_ff"] + p["cbl_ff"])
        out = {"signal_mv": round(dv * 1e3, 2),
               "ideal_charge_share_mv": round(ideal * 1e3, 2),
               "sense_time_ps": (round(float(t[done[0]] - ts * 1e-9) * 1e12, 1) if len(done)
                                 else None),
               "final_bl_v": round(float(v("bl")[-1]), 4),
               "final_blb_v": round(float(v("blb")[-1]), 4),
               "restored_cell_v": round(float(v("sn")[-1]), 4),
               "read_bit": int(v("bl")[-1] > v("blb")[-1]),
               "correct": int(v("bl")[-1] > v("blb")[-1]) == int(p["bit"])}
    elif template == "dram_1t1c":
        tr = circ["t_read"]
        before = _at(t, v("sn"), tr)
        dv = float(v("bl")[-1]) - circ["vpre"]
        out = {"stored_v_before_read": round(before, 4),
               "bitline_signal_mv": round(dv * 1e3, 2),
               "read_bit": int(dv > 0), "correct": int(dv > 0) == int(p["bit"]),
               "ideal_charge_share_mv": round((p["bit"] * vdd - circ["vpre"]) * p["cs_ff"]
                                              / (p["cs_ff"] + p["cbl_ff"]) * 1e3, 2)}
    elif template == "gain_cell_2t":
        tr = circ["t_read"]
        sn = _at(t, v("sn"), tr)
        drop = float(v("rbl")[np.searchsorted(t, tr * 1e-9)]) - float(v("rbl")[-1])
        out = {"stored_v": round(sn, 4), "rbl_drop_mv": round(drop * 1e3, 2),
               "read_bit": int(drop > 0.2 * vdd), "correct": int(drop > 0.2 * vdd) == int(p["bit"]),
               "write_vt_loss_mv": round((vdd * p["bit"] - sn) * 1e3, 1) if p["bit"] else 0.0}
    elif template == "ecore_mac":
        wx = sum(int(p[f"w{i}"]) * int(p[f"x{i}"]) for i in range(4))
        nx = sum(int(p[f"x{i}"]) for i in range(4))
        ideal = vdd * p["cs_ff"] * wx / (p["cbl_ff"] + p["cs_ff"] * nx)
        lsb = vdd * p["cs_ff"] / (p["cbl_ff"] + p["cs_ff"] * nx)
        sim = float(v("col")[-1])
        out = {"dot_product": wx, "active_inputs": nx, "column_v": round(sim, 4),
               "ideal_v": round(ideal, 4), "lsb_mv": round(lsb * 1e3, 2),
               "error_lsb": round((sim - ideal) / lsb, 3) if lsb > 0 else None,
               "decoded": int(round(sim / lsb)) if lsb > 0 else 0}
        out["correct"] = out["decoded"] == wx
    return out


def _activity(sig: dict[str, np.ndarray], node_names: list[str]) -> dict[str, float]:
    """Per-net swing (max - min) in volts: how strongly each node toggled."""
    act = {}
    for n in node_names:
        y = sig.get(f"v({n})")
        if y is not None:
            act[n] = round(float(y.max() - y.min()), 4)
    return act


def _decimate(t: np.ndarray, n_max: int = 1500) -> np.ndarray:
    if len(t) <= n_max:
        return np.arange(len(t))
    return np.unique(np.linspace(0, len(t) - 1, n_max).astype(int))


def simulate(template: str, params: dict[str, Any] | None = None,
             timeout: float = 60) -> dict[str, Any]:
    """Run ngspice on a template and return waveforms, metrics, netlist and log tail."""
    p, circ = build(template, params)
    tool = registry.get("ngspice")
    if not tool.available:
        raise registry.ToolError("ngspice is not installed")
    text = netlist(template, p, circ)
    with tempfile.TemporaryDirectory(prefix="chiplab-spice-") as tmp:
        cir, raw = Path(tmp) / "circuit.cir", Path(tmp) / "out.raw"
        cir.write_text(text)
        proc = registry.run(tool, ["-b", "-r", raw.name, cir.name], cwd=tmp, timeout=timeout)
        if not raw.is_file() or raw.stat().st_size == 0:
            raise CircuitError("ngspice produced no output: "
                               + (proc.stdout + proc.stderr)[-800:])
        sig = parse_raw(raw.read_bytes())
    t = sig["time"]
    idx = _decimate(t)
    node_names = nets(circ)
    volts = {n: sig[f"v({n})"][idx].round(5).tolist() for n in node_names if f"v({n})" in sig}
    currents = {}
    for c in circ["currents"]:
        if c in sig:
            currents[c] = (sig[c][idx] * 1e6).round(4).tolist()   # µA
    log = [ln for ln in (proc.stdout + proc.stderr).splitlines()
           if ln.strip() and "pages" not in ln.lower()]
    return {
        "template": template, "title": TITLES[template][0], "params": p,
        "simulator": {"name": "ngspice", "version": tool.version, "points": int(len(t)),
                      "returncode": proc.returncode},
        "time_ns": (t[idx] * 1e9).round(5).tolist(),
        "voltages": volts, "currents_ua": currents,
        "primary": circ["primary"], "events": circ["events"],
        "activity_v": _activity(sig, node_names),
        "metrics": metrics(template, p, circ, sig),
        "netlist": text, "log": log[-12:],
    }
