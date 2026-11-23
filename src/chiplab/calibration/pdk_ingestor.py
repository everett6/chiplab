"""Deterministic parsers for Liberty, LEF, OpenSTA/PrimeTime timing reports and SPEF, plus
calibration of the digital twin's scaling factors against real synthesis / P&R results.

Everything is a tokenizer or line state machine: no eval/exec, no shell, bounded input size.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import uuid
from pathlib import Path
from statistics import median
from typing import Any

from chiplab.schemas.common import Provenance
from chiplab.schemas.digital_twin import (
    BlackBoxMacro,
    CalibrationFactor,
    CalibrationReport,
    PinTiming,
)

MAX_BYTES = 50 * 1024 * 1024
ALLOWED_SUFFIXES = {".lib", ".lef", ".tlef", ".rpt", ".txt", ".spef", ".log", ".sta"}
FACTOR_RANGE = (0.25, 4.0)


# --- Safe input -----------------------------------------------------------------------------

def allowed_roots() -> list[Path]:
    env = os.environ.get("CHIPLAB_DATA_DIRS")
    roots = env.split(os.pathsep) if env else [os.getcwd(), str(Path.home())]
    return [Path(r).expanduser().resolve() for r in roots if r]


def read_input(content: str | None = None, path: str | None = None) -> tuple[str, dict[str, Any]]:
    """Return (text, source-info). Paths must sit under CHIPLAB_DATA_DIRS (default: cwd and
    home), use an EDA file suffix and be at most 50 MB."""
    if content is not None:
        if len(content.encode()) > MAX_BYTES:
            raise ValueError("Input exceeds 50 MB.")
        return content, {"kind": "inline", "sha256": hashlib.sha256(content.encode()).hexdigest(),
                         "bytes": len(content)}
    if not path:
        raise ValueError("Provide content or path.")
    p = Path(path).expanduser().resolve()
    if not any(p == r or r in p.parents for r in allowed_roots()):
        raise PermissionError(f"{p} is outside the allowed data directories "
                              "(set CHIPLAB_DATA_DIRS to add more).")
    if p.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported file type {p.suffix}; allowed: {sorted(ALLOWED_SUFFIXES)}")
    if not p.is_file():
        raise FileNotFoundError(str(p))
    if p.stat().st_size > MAX_BYTES:
        raise ValueError("File exceeds 50 MB.")
    data = p.read_bytes()
    return data.decode("utf-8", errors="replace"), {
        "kind": "file", "path": str(p), "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest()}


# --- Liberty --------------------------------------------------------------------------------

_LIB_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|[A-Za-z_!\[\]][\w.\[\]!:$/-]*|'
                        r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?[A-Za-z]*|[(){}:;,]')


def _lib_tokens(text: str) -> list[str]:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    text = text.replace("\\\n", " ")
    return _LIB_TOKEN.findall(text)


class _LibGroup(dict):
    pass


def parse_liberty_tree(text: str) -> dict[str, Any]:
    """Parse Liberty into nested groups: {type, args, attrs, complex, groups}."""
    toks = _lib_tokens(text)
    pos = 0

    def group(gtype: str, args: list[str]) -> dict[str, Any]:
        nonlocal pos
        g: dict[str, Any] = {"type": gtype, "args": args, "attrs": {}, "complex": {},
                             "groups": []}
        while pos < len(toks):
            t = toks[pos]
            if t == "}":
                pos += 1
                return g
            if t == ";":
                pos += 1
                continue
            name = t.strip('"')
            pos += 1
            if pos < len(toks) and toks[pos] == ":":
                pos += 1
                vals = []
                while pos < len(toks) and toks[pos] not in (";", "}"):
                    vals.append(toks[pos].strip('"'))
                    pos += 1
                if pos < len(toks) and toks[pos] == ";":
                    pos += 1
                g["attrs"][name] = " ".join(vals)
            elif pos < len(toks) and toks[pos] == "(":
                pos += 1
                a = []
                while pos < len(toks) and toks[pos] != ")":
                    if toks[pos] != ",":
                        a.append(toks[pos].strip('"'))
                    pos += 1
                pos += 1  # ')'
                if pos < len(toks) and toks[pos] == "{":
                    pos += 1
                    g["groups"].append(group(name, a))
                else:
                    g["complex"].setdefault(name, []).append(a)
            # anything else: skip stray token
        return g

    root = group("root", [])
    libs = [g for g in root["groups"] if g["type"] == "library"]
    if not libs:
        raise ValueError("No 'library (...) { }' group found; is this a Liberty file?")
    return libs[0]


def _num(s: str | None, default: float | None = None) -> float | None:
    if s is None:
        return default
    m = re.match(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)", str(s))
    return float(m.group(1)) if m else default


def _unit_scale(unit: str | None, base: str, scales: dict[str, float]) -> float:
    """'1ns' -> ps factor etc. Returns multiplier to the target unit."""
    if not unit:
        return 1.0
    m = re.match(r"\s*([\d.]+)?\s*([a-zA-Z]+)", unit)
    if not m:
        return 1.0
    mult = float(m.group(1) or 1.0)
    return mult * scales.get(m.group(2).lower(), 1.0)


_TIME_TO_PS = {"s": 1e12, "ms": 1e9, "us": 1e6, "ns": 1e3, "ps": 1.0, "fs": 1e-3}
_POWER_TO_NW = {"w": 1e9, "mw": 1e6, "uw": 1e3, "nw": 1.0, "pw": 1e-3, "fw": 1e-6}
_CAP_TO_FF = {"f": 1e15, "pf": 1e3, "ff": 1.0, "nf": 1e6}


def _table(values: list[list[str]]) -> list[list[float]]:
    rows = []
    for v in values:
        for row in v:
            rows.append([float(x) for x in re.split(r"[\s,]+", row.strip()) if x])
    return rows


def _index(g: dict[str, Any], name: str) -> list[float]:
    raw = g["complex"].get(name)
    if not raw:
        return []
    return [float(x) for x in re.split(r"[\s,]+", " ".join(raw[0]).strip()) if x]


def _interp(xs: list[float], ys: list[float], x: float) -> float:
    if not xs or len(xs) != len(ys):
        return median(ys) if ys else float("nan")
    if x <= xs[0]:
        return ys[0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / ((x1 - x0) or 1.0)
    return ys[-1]


def parse_liberty(text: str) -> dict[str, Any]:
    """Summarize a Liberty library: units, corner, cells (area, leakage, pin caps, delays)."""
    lib = parse_liberty_tree(text)
    a = lib["attrs"]
    t_ps = _unit_scale(a.get("time_unit", "1ns"), "ps", _TIME_TO_PS)
    p_nw = _unit_scale(a.get("leakage_power_unit", "1nW"), "nW", _POWER_TO_NW)
    cap_raw = lib["complex"].get("capacitive_load_unit")
    c_ff = 1.0
    if cap_raw:
        c_ff = float(cap_raw[0][0]) * _CAP_TO_FF.get(cap_raw[0][1].lower(), 1.0)
    templates = {g["args"][0]: g for g in lib["groups"]
                 if g["type"].endswith("_template") and g["args"]}
    cells = []
    for c in (g for g in lib["groups"] if g["type"] == "cell"):
        area = _num(c["attrs"].get("area"), 0.0) or 0.0
        leak = _num(c["attrs"].get("cell_leakage_power"))
        pins, delays = [], []
        for pin in (g for g in c["groups"] if g["type"] == "pin"):
            pins.append({"name": pin["args"][0] if pin["args"] else "?",
                         "direction": pin["attrs"].get("direction", ""),
                         "capacitance_ff": (_num(pin["attrs"].get("capacitance")) or 0.0) * c_ff})
            for tg in (g for g in pin["groups"] if g["type"] == "timing"):
                for arc in (g for g in tg["groups"] if g["type"] in ("cell_rise", "cell_fall")):
                    tbl = _table(arc["complex"].get("values", []))
                    tmpl = templates.get(arc["args"][0]) if arc["args"] else None
                    idx1 = _index(arc, "index_1") or (_index(tmpl, "index_1") if tmpl else [])
                    idx2 = _index(arc, "index_2") or (_index(tmpl, "index_2") if tmpl else [])
                    var1 = (tmpl or {}).get("attrs", {}).get("variable_1", "")
                    if tbl:
                        delays.append({"table": tbl, "index_1": idx1, "index_2": idx2,
                                       "variable_1": var1})
        cells.append({"name": c["args"][0] if c["args"] else "?", "area_um2": area,
                      "leakage_nw": None if leak is None else leak * p_nw,
                      "pins": pins, "delays": delays})
    if not cells:
        raise ValueError("Liberty library has no cells.")

    def in_cap(cell: dict[str, Any]) -> float:
        caps = [p["capacitance_ff"] for p in cell["pins"] if p["direction"] == "input"]
        return caps[0] if caps else 0.0

    invs = sorted((c for c in cells if re.match(r"(?i)^(inv|ckinv|inx|iv)", c["name"])
                   and c["delays"]), key=lambda c: c["area_um2"])
    fo4 = None
    if invs:
        inv = invs[0]
        load = 4.0 * in_cap(inv) / c_ff  # in library cap units
        vals = []
        for d in inv["delays"]:
            tbl = d["table"]
            loads, trans = (d["index_1"], d["index_2"]) if "capacitance" in d["variable_1"] \
                else (d["index_2"], d["index_1"])
            row = tbl[len(tbl) // 2] if "capacitance" not in d["variable_1"] else \
                [r[len(r) // 2] for r in tbl]
            vals.append(_interp(loads, row, load) if loads else median(row))
        fo4 = sum(vals) / len(vals) * t_ps
    dens = [c["leakage_nw"] / c["area_um2"] for c in cells
            if c["leakage_nw"] is not None and c["area_um2"] > 0]
    caps = [p["capacitance_ff"] for c in cells for p in c["pins"] if p["direction"] == "input"]
    all_delays = [v for c in cells for d in c["delays"] for r in d["table"] for v in r]
    return {
        "library": lib["args"][0] if lib["args"] else "?",
        "nom_voltage": _num(a.get("nom_voltage")),
        "nom_temperature": _num(a.get("nom_temperature")),
        "nom_process": _num(a.get("nom_process")),
        "time_unit_ps": t_ps, "cells": len(cells),
        "fo4_ps": None if fo4 is None else round(fo4, 3),
        "fo4_reference_cell": invs[0]["name"] if invs else None,
        # nW/um2 is numerically equal to mW/mm2
        "leakage_mw_per_mm2_median": round(median(dens), 6) if dens else None,
        "input_cap_ff_median": round(median(caps), 4) if caps else None,
        "cell_delay_ps_median": round(median(all_delays) * t_ps, 3) if all_delays else None,
        "cell_area_um2_median": round(median(c["area_um2"] for c in cells), 4),
        "cell_names": [c["name"] for c in cells][:200],
    }


# --- LEF ------------------------------------------------------------------------------------

def parse_lef(text: str) -> dict[str, Any]:
    """Line state machine over LEF: units, sites, routing layers (pitch/width/RC), macros."""
    text = re.sub(r"#[^\n]*", "", text)
    stmts = [s.strip() for s in re.split(r";|\n", text) if s.strip()]
    out: dict[str, Any] = {"dbu_per_micron": None, "layers": [], "sites": [], "macros": []}
    stack: list[tuple[str, dict[str, Any]]] = []
    for s in stmts:
        w = s.split()
        kw = w[0].upper()
        top = stack[-1] if stack else None
        if kw == "END":
            if len(w) > 1 and stack:
                # END <name> closes the nearest block with that name
                for k in range(len(stack) - 1, -1, -1):
                    if stack[k][1].get("name") == w[1] or stack[k][0] == w[1].upper():
                        del stack[k:]
                        break
            elif stack and stack[-1][0] in ("PORT", "OBS"):
                stack.pop()
            continue
        if kw in ("LAYER", "MACRO", "SITE", "PIN") and len(w) >= 2 and \
                not (top and top[0] in ("PORT", "OBS")):
            blk: dict[str, Any] = {"name": w[1]}
            if kw == "LAYER" and not stack:
                out["layers"].append(blk)
            elif kw == "SITE" and not stack:
                out["sites"].append(blk)
            elif kw == "MACRO":
                blk.update({"pins": [], "obs": False})
                out["macros"].append(blk)
            elif kw == "PIN" and top and top[0] == "MACRO":
                top[1]["pins"].append(blk)
            stack.append((kw, blk))
            continue
        if kw in ("PORT", "OBS"):
            if kw == "OBS" and top and top[0] == "MACRO":
                top[1]["obs"] = True
            stack.append((kw, {}))
            continue
        if kw == "UNITS":
            stack.append(("UNITS", {}))
            continue
        if kw == "DATABASE" and len(w) >= 3:
            out["dbu_per_micron"] = _num(w[2])
            continue
        if not top:
            continue
        kind, blk = top
        if kind == "LAYER":
            if kw == "TYPE":
                blk["type"] = w[1].upper()
            elif kw == "DIRECTION":
                blk["direction"] = w[1].upper()
            elif kw in ("PITCH", "WIDTH", "SPACING", "THICKNESS", "EDGECAPACITANCE"):
                blk[kw.lower()] = _num(w[1])
            elif kw == "RESISTANCE" and len(w) >= 3 and w[1].upper() == "RPERSQ":
                blk["r_per_sq"] = _num(w[2])
            elif kw == "CAPACITANCE" and len(w) >= 3 and w[1].upper() == "CPERSQDIST":
                blk["c_per_sq_dist"] = _num(w[2])
        elif kind in ("MACRO", "SITE"):
            if kw == "CLASS":
                blk["class"] = " ".join(w[1:]).upper()
            elif kw == "SIZE" and len(w) >= 4:
                blk["width_um"], blk["height_um"] = _num(w[1]), _num(w[3])
            elif kw == "ORIGIN" and len(w) >= 3:
                blk["origin"] = [_num(w[1]), _num(w[2])]
        elif kind == "PIN":
            if kw == "DIRECTION":
                blk["direction"] = w[1].upper()
            elif kw == "USE":
                blk["use"] = w[1].upper()
    routing = [lyr for lyr in out["layers"] if lyr.get("type") == "ROUTING"]
    for lyr in routing:
        r, c, wd = lyr.get("r_per_sq"), lyr.get("c_per_sq_dist"), lyr.get("width")
        if r and wd:
            lyr["r_ohm_per_um"] = r / wd
        if c and wd:
            # CPERSQDIST pF/um2 * width + 2 * EDGECAPACITANCE pF/um -> fF/um
            lyr["c_ff_per_um"] = (c * wd + 2 * (lyr.get("edgecapacitance") or 0.0)) * 1000.0
    pitches = [lyr["pitch"] for lyr in routing if lyr.get("pitch")]
    core_sites = [s for s in out["sites"] if s.get("height_um")]
    out["summary"] = {
        "routing_layers": len(routing),
        "min_pitch_um": min(pitches) if pitches else None,
        "site_height_um": core_sites[0]["height_um"] if core_sites else None,
        "macros": len(out["macros"]),
        "block_macros": [m["name"] for m in out["macros"]
                         if str(m.get("class", "")).startswith(("BLOCK", "PAD", "RING"))][:100],
    }
    return out


def lef_macro_to_blackbox(macro: dict[str, Any], keepout_um: float = 5.0) -> BlackBoxMacro:
    """Turn a LEF macro (size + pins) into a BlackBoxMacro; add timing from the datasheet."""
    pins = [PinTiming(name=p["name"], direction=str(p.get("direction", "INPUT")).lower()
                      if str(p.get("direction", "INPUT")).lower() in ("input", "output", "inout")
                      else "inout")
            for p in macro.get("pins", []) if p.get("use") not in ("POWER", "GROUND")]
    return BlackBoxMacro(name=macro["name"], width_um=macro.get("width_um") or 0.0,
                         height_um=macro.get("height_um") or 0.0, keepout_um=keepout_um,
                         pins=pins, provenance=Provenance.MEASURED, datasheet="LEF")


# --- Timing reports (OpenSTA / PrimeTime) ---------------------------------------------------

_PATH_LINE = re.compile(r"^\s*(?:(?P<fan>\d+)\s+)?(?:(?P<cap>[\d.]+)\s+)?(?:(?P<slew>[\d.]+)\s+)?"
                        r"(?P<incr>-?[\d.]+)\s+[*&]?\s*(?P<path>-?[\d.]+)\s*[\^v rf]*\s+"
                        r"(?P<pt>\S+)(?:\s+\((?P<ref>[^)]*)\))?")
_PT_LINE = re.compile(r"^\s*(?P<pt>[^\s(]+)\s+\((?P<ref>[^)]*)\)\s+(?:[\d.]+\s+)*?"
                      r"(?P<incr>-?[\d.]+)\s*[*&]?\s+(?P<path>-?[\d.]+)\s*[rf]?\s*$")
_INPUT_PIN = re.compile(r"/(A\d*|B\d*|C\d*|D\d*|I\d*|S\d*|SE|SI|CK|CLK|CP|EN|E|G|GN|RN|SN|R|"
                        r"IN\d*|A[1-9]|B[1-9])$")


def parse_timing_report(text: str) -> dict[str, Any]:
    """Parse OpenSTA `report_checks` / PrimeTime `report_timing` text."""
    unit_ps = 1000.0  # default ns
    m = re.search(r"(?im)^\s*time\s*(?:unit)?\s*[:=]?\s*1?\s*(ps|ns)\b", text)
    if m and m.group(1).lower() == "ps":
        unit_ps = 1.0
    blocks = re.split(r"(?m)^\s*Startpoint:\s*", text)[1:]
    paths = []
    for b in blocks:
        start = b.split()[0] if b.split() else "?"
        end = re.search(r"Endpoint:\s*(\S+)", b)
        arr = re.search(r"(?m)^[ \t]*(-?[\d.]+)[ \t]+data arrival time", b) or \
            re.search(r"data arrival time[ \t]+(-?[\d.]+)", b)
        req = re.search(r"(?m)^[ \t]*(-?[\d.]+)[ \t]+data required time", b) or \
            re.search(r"data required time[ \t]+(-?[\d.]+)", b)
        slack = re.search(r"(-?[\d.]+)\s+slack\s*\((MET|VIOLATED)[^)]*\)", b)
        if not slack:
            slack2 = re.search(r"slack\s*\((MET|VIOLATED)[^)]*\)\s+(-?[\d.]+)", b)
            slack_v = float(slack2.group(2)) if slack2 else None
            status = slack2.group(1) if slack2 else None
        else:
            slack_v, status = float(slack.group(1)), slack.group(2)
        # Clock edge lines: "<incr> <time> clock clk (rise edge)" (OpenSTA/PrimeTime layout).
        edges = [float(t) for t in re.findall(
            r"(?m)^\s*-?[\d.]+\s+(-?[\d.]+)\s+clock\s+\S+\s+\((?:rise|fall) edge\)", b)] or \
            [float(t) for t in re.findall(
                r"(?m)^\s*clock\s+\S+\s+\((?:rise|fall) edge\)\s+-?[\d.]+\s+(-?[\d.]+)", b)]
        period_ps = edges[1] * unit_ps if len(edges) > 1 and edges[1] > 0 else None
        arrival_section = b.split("data arrival time")[0]
        cell = wire = 0.0
        stages = 0
        for line in arrival_section.splitlines():
            if "clock" in line or "edge" in line:
                continue
            pm = _PATH_LINE.match(line) or _PT_LINE.match(line)
            if not pm:
                continue
            incr = float(pm.group("incr"))
            pt = pm.group("pt")
            ref = pm.group("ref")
            if ref == "net" or "(net)" in line:
                wire += incr
            elif _INPUT_PIN.search(pt):
                wire += incr
            elif ref:
                cell += incr
                stages += 1
        paths.append({
            "startpoint": start, "endpoint": end.group(1) if end else "?",
            "arrival_ps": float(arr.group(1)) * unit_ps if arr else None,
            "required_ps": float(req.group(1)) * unit_ps if req else None,
            "slack_ps": slack_v * unit_ps if slack_v is not None else None,
            "status": status,
            "period_ps": period_ps,
            "cell_delay_ps": round(cell * unit_ps, 3), "wire_delay_ps": round(wire * unit_ps, 3),
            "logic_stages": stages,
        })
    if not paths:
        raise ValueError("No timing paths found (expected 'Startpoint:' blocks).")
    worst = min(paths, key=lambda p: p["slack_ps"] if p["slack_ps"] is not None else math.inf)
    return {"paths": paths, "worst": worst, "path_count": len(paths),
            "violated": sum(p["status"] == "VIOLATED" for p in paths)}


# --- SPEF -----------------------------------------------------------------------------------

def parse_spef(text: str) -> dict[str, Any]:
    t_unit = re.search(r"\*T_UNIT\s+([\d.]+)\s+(\w+)", text)
    c_unit = re.search(r"\*C_UNIT\s+([\d.]+)\s+(\w+)", text)
    r_unit = re.search(r"\*R_UNIT\s+([\d.]+)\s+(\w+)", text)
    c_ff = float(c_unit.group(1)) * _CAP_TO_FF.get(c_unit.group(2).lower(), 1.0) if c_unit else 1.0
    r_ohm = float(r_unit.group(1)) * {"ohm": 1.0, "kohm": 1e3}.get(r_unit.group(2).lower(), 1.0) \
        if r_unit else 1.0
    nets = []
    section = None
    cur: dict[str, Any] | None = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("*D_NET"):
            w = s.split()
            cur = {"name": w[1], "total_cap_ff": (_num(w[2], 0.0) or 0.0) * c_ff, "res_ohm": 0.0,
                   "cap_segments": 0}
            nets.append(cur)
            section = None
        elif s.startswith("*CAP"):
            section = "cap"
        elif s.startswith("*RES"):
            section = "res"
        elif s.startswith(("*CONN", "*INDUC")):
            section = "other"
        elif s.startswith("*END"):
            cur, section = None, None
        elif cur is not None and section in ("cap", "res") and s and s[0].isdigit():
            w = s.split()
            if section == "res" and len(w) >= 4:
                cur["res_ohm"] += (_num(w[3], 0.0) or 0.0) * r_ohm
            elif section == "cap":
                cur["cap_segments"] += 1
    if not nets:
        raise ValueError("No *D_NET sections found in SPEF.")
    caps = [n["total_cap_ff"] for n in nets]
    elmore = [0.69 * n["res_ohm"] * n["total_cap_ff"] * 1e-3 / 2 for n in nets]  # ps
    return {"nets": len(nets), "time_unit": t_unit.group(2) if t_unit else None,
            "total_cap_ff": round(sum(caps), 4), "avg_cap_ff": round(sum(caps) / len(caps), 4),
            "avg_res_ohm": round(sum(n["res_ohm"] for n in nets) / len(nets), 4),
            "avg_elmore_ps": round(sum(elmore) / len(elmore), 4),
            "max_cap_net": max(nets, key=lambda n: n["total_cap_ff"])["name"]}


# --- Calibration ----------------------------------------------------------------------------

def _clamp(name: str, v: float, warnings: list[str]) -> float:
    lo, hi = FACTOR_RANGE
    if not math.isfinite(v) or v <= 0:
        warnings.append(f"{name}: computed factor {v} is invalid; ignored.")
        return float("nan")
    if v < lo or v > hi:
        warnings.append(f"{name}: factor {v:.3f} outside [{lo}, {hi}]; clamped. Check units "
                        "and that the design point matches the run.")
    return min(max(v, lo), hi)


def calibrate(node: str, design_point: dict[str, Any] | None = None,
              liberty: dict[str, Any] | None = None, lef: dict[str, Any] | None = None,
              timing: dict[str, Any] | None = None, spef: dict[str, Any] | None = None,
              sources: list[dict[str, Any]] | None = None) -> CalibrationReport:
    """Compare parsed EDA data with the twin model at the same design point and derive
    multiplicative calibration factors (gate_delay, wire_rc, leakage)."""
    from chiplab.engine import digital_twin as dt

    node = node.upper()
    if node not in dt.NODE_NAMES:
        raise ValueError(f"node must be one of {dt.NODE_NAMES}")
    ni = dt.NODE_NAMES.index(node)
    point = {**dt.DEFAULTS, **(design_point or {}), "node": node}
    warnings: list[str] = []
    factors: list[CalibrationFactor] = []
    summary: dict[str, Any] = {}
    src = ", ".join(s.get("path") or s.get("kind", "inline") for s in (sources or [])) or "inline"

    vn, vt = dt._NODE["vnom"][ni], dt._NODE["vth"][ni]
    if liberty:
        summary["liberty"] = {k: liberty.get(k) for k in (
            "library", "nom_voltage", "fo4_ps", "leakage_mw_per_mm2_median", "cells")}
        v = liberty.get("nom_voltage") or vn
        if liberty.get("fo4_ps"):
            vscale = (v / max(v - vt, 0.02) ** dt.ALPHA) / (vn / (vn - vt) ** dt.ALPHA)
            model_fo4 = dt._NODE["fo4_ps"][ni] * vscale
            f = _clamp("gate_delay", liberty["fo4_ps"] / model_fo4, warnings)
            if math.isfinite(f):
                factors.append(CalibrationFactor(
                    name="gate_delay", value=round(f, 4), source=src,
                    detail=f"Liberty FO4 {liberty['fo4_ps']:.2f} ps vs model {model_fo4:.2f} ps "
                           f"at {v} V ({liberty.get('fo4_reference_cell')})"))
        if liberty.get("leakage_mw_per_mm2_median"):
            f = _clamp("leakage", liberty["leakage_mw_per_mm2_median"]
                       / dt._NODE["leak_mw_mm2"][ni], warnings)
            if math.isfinite(f):
                factors.append(CalibrationFactor(
                    name="leakage", value=round(f, 4), source=src,
                    detail=f"Liberty median leakage {liberty['leakage_mw_per_mm2_median']:.3f} "
                           f"mW/mm2 vs model {dt._NODE['leak_mw_mm2'][ni]} mW/mm2"))
    if timing:
        w = timing["worst"]
        if w.get("period_ps") and "freq_ghz" not in (design_point or {}):
            point["freq_ghz"] = 1000.0 / w["period_ps"]
        m = dt.evaluate_point(point)
        model_path = 1000.0 / m["fmax_ghz"]
        nominal = dt.TwinModel()
        p = {k: (ni if k == "node" else float(point[k])) for k in dt.REQUIRED}
        parts = nominal.evaluate(p, {"wire_rc": 1e-9}, dt._Scalar())
        model_cell = 1000.0 / parts["fmax_ghz"]
        model_wire = max(model_path - model_cell, 1e-6)
        meas_cell, meas_wire = w.get("cell_delay_ps") or 0.0, w.get("wire_delay_ps") or 0.0
        summary["timing"] = {"worst_slack_ps": w.get("slack_ps"), "arrival_ps": w.get("arrival_ps"),
                             "cell_ps": meas_cell, "wire_ps": meas_wire,
                             "model_cell_ps": round(model_cell, 2),
                             "model_wire_ps": round(model_wire, 2),
                             "paths": timing["path_count"], "violated": timing["violated"]}
        if meas_cell > 0 and not any(f.name == "gate_delay" for f in factors):
            f = _clamp("gate_delay", meas_cell / model_cell, warnings)
            if math.isfinite(f):
                factors.append(CalibrationFactor(
                    name="gate_delay", value=round(f, 4), source=src,
                    detail=f"critical-path cell delay {meas_cell:.1f} ps vs model "
                           f"{model_cell:.1f} ps"))
        if meas_wire > 0:
            f = _clamp("wire_rc", meas_wire / model_wire, warnings)
            if math.isfinite(f):
                factors.append(CalibrationFactor(
                    name="wire_rc", value=round(f, 4), source=src,
                    detail=f"critical-path wire delay {meas_wire:.1f} ps vs model "
                           f"{model_wire:.1f} ps"))
        elif meas_cell == 0 and w.get("arrival_ps"):
            warnings.append("Timing report has no cell/wire breakdown; only the summary was "
                            "recorded. Use report_checks -fields {net} or report_timing -nets.")
    if spef:
        summary["spef"] = spef
        if not any(f.name == "wire_rc" for f in factors):
            warnings.append("SPEF parsed for parasitic statistics; wire_rc is only calibrated "
                            "from a timing report's wire-delay split.")
    if lef:
        summary["lef"] = lef.get("summary")
    if not factors:
        warnings.append("No calibration factor could be derived from the inputs.")
    return CalibrationReport(calibration_id=f"cal_{uuid.uuid4().hex[:10]}", node=node,
                             factors=factors, sources=sources or [], warnings=warnings,
                             summary=summary)
