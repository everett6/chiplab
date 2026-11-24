"""Component SDK: hierarchical, user-definable hardware components.

A component is data, not code: parameters (with units and provenance), ports (digital,
analog or power), model expressions (area, power, delay, transistor count), an optional
digital behaviour (combinational outputs, or registers with next-state expressions), child
instances with parameter bindings, connections, validation rules, a visual style and a
simulation backend name. Expressions use the sandboxed language in ``chiplab.sdk.expr``.

Users and agents add components by registering a definition (JSON/YAML); ChipLab's core is
not modified and nothing they write is executed as Python. A registered name@version is
immutable: changing a component means registering a new version.

Elaboration resolves parameters through the hierarchy, checks connections and port widths,
rolls up area/power/transistors (a parent without its own model sums its children), and
produces a block-diagram graph. ``simulate`` runs a cycle-based functional simulation of the
digital leaves. Fidelity: FUNCTIONAL for simulation, ANALYTICAL for the roll-ups.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chiplab.library.pdk_library import get_pdk
from chiplab.sdk import expr

CATEGORIES = ("transistor", "passive", "logic", "arithmetic", "memory", "control", "processor",
              "analog", "mixed_signal", "interconnect", "clock_reset", "io", "custom")
BACKENDS = ("functional", "analytical", "rtl", "spice", "memory_compiler", "external")
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class ComponentError(ValueError):
    pass


@dataclass
class Component:
    name: str
    version: str = "1.0"
    category: str = "custom"
    description: str = ""
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    ports: list[dict[str, Any]] = field(default_factory=list)
    models: dict[str, str | float] = field(default_factory=dict)
    behavior: dict[str, Any] | None = None
    instances: list[dict[str, Any]] = field(default_factory=list)
    connections: list[list[str]] = field(default_factory=list)
    rules: list[dict[str, str]] = field(default_factory=list)
    view: dict[str, Any] = field(default_factory=dict)
    backend: str = "analytical"
    builtin: bool = False

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        d = copy.deepcopy(self.__dict__)
        d.pop("builtin", None)
        return d


def _check_def(d: dict[str, Any]) -> Component:
    allowed = set(Component.__dataclass_fields__) - {"builtin"}
    unknown = set(d) - allowed
    if unknown:
        raise ComponentError(f"unknown component fields: {sorted(unknown)}")
    c = Component(**{k: copy.deepcopy(v) for k, v in d.items()})
    if not _NAME.match(c.name):
        raise ComponentError("name must be an identifier (letters, digits, _)")
    if c.category not in CATEGORIES:
        raise ComponentError(f"category must be one of {CATEGORIES}")
    if c.backend not in BACKENDS:
        raise ComponentError(f"backend must be one of {BACKENDS}")
    seen = set()
    for p in c.ports:
        if not _NAME.match(str(p.get("name", ""))) or p["name"] in seen:
            raise ComponentError(f"bad or duplicate port {p.get('name')!r}")
        seen.add(p["name"])
        if p.get("dir") not in ("in", "out", "inout"):
            raise ComponentError(f"port {p['name']}: dir must be in, out or inout")
        if p.get("domain", "digital") not in ("digital", "analog", "power"):
            raise ComponentError(f"port {p['name']}: domain must be digital, analog or power")
    for name, spec in c.params.items():
        if not _NAME.match(name) or "default" not in spec:
            raise ComponentError(f"parameter {name!r} needs an identifier name and a default")
        if spec.get("provenance", "assumed") not in ("measured", "published", "calibrated", "estimated", "assumed"):
            raise ComponentError(f"parameter {name}: provenance must be measured/published/calibrated/estimated/assumed")
    for k, v in c.models.items():
        if k not in ("area_um2", "power_uw", "delay_ns", "transistors", "energy_pj"):
            raise ComponentError(f"unknown model {k!r}")
        if isinstance(v, str):
            expr.parse(v)
    for r in c.rules:
        expr.parse(r["expr"])
    if c.behavior:
        t = c.behavior.get("type")
        if t not in ("comb", "seq"):
            raise ComponentError("behavior.type must be comb or seq")
        for e in list(c.behavior.get("outputs", {}).values()) + list(c.behavior.get("next", {}).values()):
            expr.parse(e)
    inames = set()
    for i in c.instances:
        if not _NAME.match(str(i.get("name", ""))) or i["name"] in inames or "component" not in i:
            raise ComponentError(f"bad instance {i!r}")
        inames.add(i["name"])
    return c


# ------------------------------------------------------------------ built-in library
def _leaf(name, cat, desc, params, ports, models, behavior=None, backend="analytical", color="#6ea0ff"):
    return {"name": name, "category": cat, "description": desc, "params": params, "ports": ports,
            "models": models, "behavior": behavior, "backend": backend, "view": {"color": color}}


def _p(default, unit="", prov="assumed", **kw):
    return {"default": default, "unit": unit, "provenance": prov, **kw}


D_IN = lambda n, w=1: {"name": n, "dir": "in", "width": w}            # noqa: E731
D_OUT = lambda n, w=1: {"name": n, "dir": "out", "width": w}          # noqa: E731
A_IN = lambda n: {"name": n, "dir": "inout", "width": 1, "domain": "analog"}  # noqa: E731

LIBRARY: list[dict[str, Any]] = [
    _leaf("nmos", "transistor", "n-channel MOSFET (analog leaf; simulated in ngspice when installed)",
          {"W": _p(0.5, "um"), "L": _p(0.13, "um")}, [A_IN("d"), A_IN("g"), A_IN("s"), A_IN("b")],
          {"area_um2": "W * L * 6 + 0.2", "transistors": 1}, backend="spice", color="#c8687a"),
    _leaf("pmos", "transistor", "p-channel MOSFET", {"W": _p(1.0, "um"), "L": _p(0.13, "um")},
          [A_IN("d"), A_IN("g"), A_IN("s"), A_IN("b")], {"area_um2": "W * L * 6 + 0.2", "transistors": 1},
          backend="spice", color="#c8687a"),
    _leaf("resistor", "passive", "poly resistor", {"R": _p(10e3, "ohm"), "sheet": _p(350.0, "ohm/sq")},
          [A_IN("p"), A_IN("n")], {"area_um2": "R / sheet * 0.5 * 0.5 + 1"}, backend="spice", color="#8b97a4"),
    _leaf("capacitor", "passive", "MIM capacitor", {"C": _p(100e-15, "F"), "density": _p(1.5, "fF/um2")},
          [A_IN("p"), A_IN("n")], {"area_um2": "C * 1e15 / density"}, backend="spice", color="#8b97a4"),
    _leaf("inductor", "passive", "spiral inductor", {"Lh": _p(1e-9, "H")}, [A_IN("p"), A_IN("n")],
          {"area_um2": "40000 * sqrt(Lh * 1e9)"}, backend="spice", color="#8b97a4"),
    _leaf("diode", "passive", "junction diode", {"A": _p(1.0, "um2")}, [A_IN("a"), A_IN("k")],
          {"area_um2": "A + 2"}, backend="spice", color="#8b97a4"),
    _leaf("inv", "logic", "inverter", {}, [D_IN("a"), D_OUT("y")],
          {"area_um2": "gate_um2 * 0.67", "transistors": 2, "delay_ns": "fo4_ns"},
          {"type": "comb", "outputs": {"y": "1 - (a & 1)"}}, "functional"),
    _leaf("nand2", "logic", "2-input NAND", {}, [D_IN("a"), D_IN("b"), D_OUT("y")],
          {"area_um2": "gate_um2", "transistors": 4, "delay_ns": "fo4_ns * 1.2"},
          {"type": "comb", "outputs": {"y": "1 - (a & b & 1)"}}, "functional"),
    _leaf("nor2", "logic", "2-input NOR", {}, [D_IN("a"), D_IN("b"), D_OUT("y")],
          {"area_um2": "gate_um2", "transistors": 4, "delay_ns": "fo4_ns * 1.4"},
          {"type": "comb", "outputs": {"y": "1 - ((a | b) & 1)"}}, "functional"),
    _leaf("xor2", "logic", "2-input XOR", {}, [D_IN("a"), D_IN("b"), D_OUT("y")],
          {"area_um2": "gate_um2 * 2", "transistors": 10, "delay_ns": "fo4_ns * 2"},
          {"type": "comb", "outputs": {"y": "(a ^ b) & 1"}}, "functional"),
    _leaf("mux2", "logic", "2:1 multiplexer (width W)", {"W": _p(1, "bit")},
          [D_IN("a", "W"), D_IN("b", "W"), D_IN("s"), D_OUT("y", "W")],
          {"area_um2": "gate_um2 * 2.5 * W", "transistors": "12 * W", "delay_ns": "fo4_ns * 2"},
          {"type": "comb", "outputs": {"y": "select(s & 1, b, a)"}}, "functional"),
    _leaf("dff", "control", "D flip-flop register (width W)", {"W": _p(1, "bit")},
          [D_IN("d", "W"), D_OUT("q", "W")],
          {"area_um2": "gate_um2 * 6 * W", "transistors": "24 * W", "power_uw": "W * 0.5"},
          {"type": "seq", "state": {"r": {"width": "W", "reset": 0}}, "next": {"r": "d"}, "outputs": {"q": "r"}},
          "functional", "#e6c35c"),
    _leaf("adder", "arithmetic", "ripple-carry adder (width W)", {"W": _p(8, "bit")},
          [D_IN("a", "W"), D_IN("b", "W"), D_OUT("y", "W"), D_OUT("co")],
          {"area_um2": "gate_um2 * 7 * W", "transistors": "28 * W", "delay_ns": "fo4_ns * 2 * W"},
          {"type": "comb", "outputs": {"y": "(a + b) & mask(W)", "co": "((a + b) >> W) & 1"}}, "functional", "#3fb960"),
    _leaf("multiplier", "arithmetic", "array multiplier (W x W -> 2W)", {"W": _p(8, "bit")},
          [D_IN("a", "W"), D_IN("b", "W"), D_OUT("y", "2 * W")],
          {"area_um2": "gate_um2 * 6 * W * W", "transistors": "24 * W * W", "delay_ns": "fo4_ns * 4 * W"},
          {"type": "comb", "outputs": {"y": "(a * b) & mask(2 * W)"}}, "functional", "#3fb960"),
    _leaf("mac", "arithmetic", "multiply-accumulate (acc += a*b), accumulator width A",
          {"W": _p(8, "bit"), "A": _p(24, "bit")},
          [D_IN("a", "W"), D_IN("b", "W"), D_IN("clr"), D_OUT("acc", "A")],
          {"area_um2": "gate_um2 * (6 * W * W + 13 * A)", "transistors": "24 * W * W + 52 * A"},
          {"type": "seq", "state": {"r": {"width": "A", "reset": 0}},
           "next": {"r": "select(clr & 1, 0, (r + a * b) & mask(A))"}, "outputs": {"acc": "r"}}, "functional", "#3fb960"),
    _leaf("counter", "control", "up counter with enable", {"W": _p(8, "bit")}, [D_IN("en"), D_OUT("q", "W")],
          {"area_um2": "gate_um2 * 10 * W"},
          {"type": "seq", "state": {"r": {"width": "W", "reset": 0}}, "next": {"r": "(r + (en & 1)) & mask(W)"},
           "outputs": {"q": "r"}}, "functional", "#e6c35c"),
    _leaf("sram_cell", "memory", "6T SRAM bit cell (circuit leaf)", {}, [A_IN("bl"), A_IN("blb"), A_IN("wl")],
          {"area_um2": "sram_bitcell_um2", "transistors": 6}, backend="spice", color="#9c5470"),
    _leaf("dram_cell", "memory", "1T1C DRAM cell (needs a DRAM capacitor process)", {"Cs": _p(20e-15, "F")},
          [A_IN("bl"), A_IN("wl")], {"area_um2": "sram_bitcell_um2 * 0.22", "transistors": 1},
          backend="spice", color="#9c5470"),
    _leaf("sense_amp", "analog", "latch-type sense amplifier", {"offset_mv": _p(20.0, "mV", "estimated")},
          [A_IN("bl"), A_IN("blb"), D_IN("en"), D_OUT("out")],
          {"area_um2": "gate_um2 * 12", "transistors": 11, "energy_pj": 0.01},
          {"type": "comb", "outputs": {"out": "select(en & 1, select(bl > blb, 1, 0), 0)"}}, "functional", "#d9a24a"),
    _leaf("adc", "mixed_signal", "N-bit ADC (ideal quantiser for functional simulation)",
          {"bits": _p(6, "bit"), "vref": _p(1.0, "V"), "fom_fj": _p(50.0, "fJ/conv-step", "estimated")},
          [{"name": "vin", "dir": "in", "width": 1, "domain": "analog"}, D_OUT("code", "bits")],
          {"area_um2": "gate_um2 * 40 * (1 << bits) / 8", "energy_pj": "fom_fj * (1 << bits) / 1000"},
          {"type": "comb", "outputs": {"code": "quantize(vin, bits, vref)"}}, "functional", "#d9a24a"),
    _leaf("dac", "mixed_signal", "N-bit DAC (ideal)", {"bits": _p(6, "bit"), "vref": _p(1.0, "V")},
          [D_IN("code", "bits"), {"name": "vout", "dir": "out", "width": 1, "domain": "analog"}],
          {"area_um2": "gate_um2 * 20 * bits"},
          {"type": "comb", "outputs": {"vout": "code * vref / mask(bits)"}}, "functional", "#d9a24a"),
    _leaf("memory_array", "memory", "memory macro sized by the ChipLab memory compiler",
          {"words": _p(1024, "words"), "width": _p(32, "bit"), "kind_code": _p(0, "", note="0 sram 1 rom 2 edram")},
          [D_IN("addr", "clog2(words)"), D_IN("wdata", "width"), D_IN("we"), D_OUT("rdata", "width")],
          {"area_um2": "words * width * sram_bitcell_um2 * 1.6"}, backend="memory_compiler", color="#9c5470"),
    _leaf("clock_divider", "clock_reset", "clock divider by N", {"N": _p(4, "")}, [D_OUT("tick")],
          {"area_um2": "gate_um2 * 10 * clog2(N)"},
          {"type": "seq", "state": {"c": {"width": "clog2(N) + 1", "reset": 0}},
           "next": {"c": "select(c + 1 >= N, 0, c + 1)"}, "outputs": {"tick": "select(c == 0, 1, 0)"}}, "functional", "#e6c35c"),
    _leaf("reset_sync", "clock_reset", "two-flop reset synchroniser", {}, [D_IN("rst_in"), D_OUT("rst_out")],
          {"area_um2": "gate_um2 * 12"},
          {"type": "seq", "state": {"s1": {"width": 1, "reset": 1}, "s2": {"width": 1, "reset": 1}},
           "next": {"s1": "rst_in", "s2": "s1"}, "outputs": {"rst_out": "s2"}}, "functional", "#e6c35c"),
    _leaf("io_pad", "io", "bidirectional I/O pad (area from the pad ring, assumed)", {"pitch_um": _p(80.0, "um")},
          [D_IN("o"), D_IN("oe"), D_OUT("i"), A_IN("pad")], {"area_um2": "pitch_um * 150"}, color="#8a6a3a"),
    _leaf("wire", "interconnect", "on-chip wire (Elmore delay, repeaterless)",
          {"length_um": _p(1000.0, "um"), "r_ohm_um": _p(0.1, "ohm/um", "estimated")}, [D_IN("a"), D_OUT("y")],
          {"delay_ns": "0.38 * r_ohm_um * length_um * wire_cap_ff_per_um * length_um * 1e-6", "area_um2": 0},
          {"type": "comb", "outputs": {"y": "a"}}, "functional", "#8b97a4"),
    _leaf("pem_core", "processor", "ChipLab PEM-16 protocol engine (4 lanes; RTL in chiplab/asic)",
          {"lanes": _p(4, ""), "imem_words": _p(32, "")}, [D_IN("bus", 24), D_OUT("pins", 14)],
          {"area_um2": "gate_um2 * 6 * (1024 + lanes * 162 + imem_words * 0)", "transistors": "24 * (1024 + lanes * 162)"},
          backend="rtl", color="#2563eb"),
]

TEMPLATES: list[dict[str, Any]] = [
    {"name": "dram_cim_bitcell", "category": "memory", "version": "1.0",
     "description": "Composite: access transistor + storage capacitor + sense amp -> a compute-in-memory bit cell",
     "params": {"Cs": _p(20e-15, "F"), "W": _p(0.3, "um")},
     "ports": [A_IN("bl"), A_IN("wl"), D_IN("sense_en"), D_OUT("out")],
     "instances": [{"name": "m_access", "component": "nmos", "params": {"W": "W", "L": "0.13"}},
                   {"name": "c_store", "component": "capacitor", "params": {"C": "Cs"}},
                   {"name": "sa", "component": "sense_amp", "params": {}}],
     "connections": [["bl", "m_access.d"], ["wl", "m_access.g"], ["m_access.s", "c_store.p"],
                     ["bl", "sa.bl"], ["sense_en", "sa.en"], ["sa.out", "out"]],
     "rules": [{"expr": "Cs >= 5e-15", "message": "storage cap below 5 fF gives too little read signal"}],
     "backend": "spice", "view": {"color": "#9c5470"}},
    {"name": "mac_unit_pipelined", "category": "arithmetic", "version": "1.0",
     "description": "Composite: input registers + MAC (a hierarchical digital block)",
     "params": {"W": _p(8, "bit"), "A": _p(24, "bit")},
     "ports": [D_IN("a", "W"), D_IN("b", "W"), D_IN("clr"), D_OUT("acc", "A")],
     "instances": [{"name": "ra", "component": "dff", "params": {"W": "W"}},
                   {"name": "rb", "component": "dff", "params": {"W": "W"}},
                   {"name": "m", "component": "mac", "params": {"W": "W", "A": "A"}}],
     "connections": [["a", "ra.d"], ["b", "rb.d"], ["ra.q", "m.a"], ["rb.q", "m.b"], ["clr", "m.clr"], ["m.acc", "acc"]],
     "backend": "functional", "view": {"color": "#3fb960"}},
]


class Registry:
    """Built-ins + templates + user components (stored as JSON files, one per name@version)."""

    def __init__(self, user_dir: str | Path | None = None) -> None:
        self.user_dir = Path(user_dir) if user_dir else Path.cwd() / ".chiplab_components"
        self.items: dict[str, Component] = {}
        for d in LIBRARY + TEMPLATES:
            c = _check_def(d)
            c.builtin = True
            self.items[c.key] = c
        if self.user_dir.exists():
            for f in sorted(self.user_dir.glob("*.json")):
                try:
                    c = _check_def(json.loads(f.read_text()))
                    self.items.setdefault(c.key, c)
                except (ComponentError, expr.ExprError, json.JSONDecodeError, TypeError):
                    continue

    def get(self, ref: str) -> Component:
        if "@" in ref:
            if ref not in self.items:
                raise ComponentError(f"unknown component {ref!r}")
            return self.items[ref]
        vs = [c for c in self.items.values() if c.name == ref]
        if not vs:
            raise ComponentError(f"unknown component {ref!r}")
        return max(vs, key=lambda c: [int(x) if x.isdigit() else x for x in re.split(r"[.\-]", c.version)])

    def list(self, category: str | None = None) -> list[dict[str, Any]]:
        return [{"key": c.key, "name": c.name, "version": c.version, "category": c.category,
                 "description": c.description, "builtin": c.builtin, "backend": c.backend,
                 "composite": bool(c.instances)} for c in self.items.values()
                if category is None or c.category == category]

    def register(self, d: dict[str, Any], persist: bool = True) -> Component:
        c = _check_def(d)
        if c.key in self.items:
            raise ComponentError(f"{c.key} already exists; versions are immutable, register a new version")
        for i in c.instances:                         # children must exist
            self.get(i["component"])
        elaborate(self, c)                            # must elaborate cleanly before it is accepted
        self.items[c.key] = c
        if persist:
            self.user_dir.mkdir(parents=True, exist_ok=True)
            (self.user_dir / f"{c.name}@{c.version}.json").write_text(json.dumps(c.to_dict(), indent=1))
        return c


# ------------------------------------------------------------------ elaboration
def _pdk_env(pdk: str) -> dict[str, float]:
    p = get_pdk(pdk)
    return {"gate_um2": 1000.0 / p.scalar("gate_density_kgates_mm2"), "fo4_ns": p.scalar("fo4_delay_ps") / 1000,
            "sram_bitcell_um2": p.scalar("sram_bitcell_um2"), "vdd": p.scalar("nominal_vdd_v"),
            "wire_cap_ff_per_um": p.scalar("wire_cap_ff_per_um"), "node_nm": p.node_nm}


def elaborate(reg: Registry, comp: Component | str, params: dict[str, Any] | None = None,
              pdk: str = "sky130", path: str = "top", depth: int = 0) -> dict[str, Any]:
    if depth > 16:
        raise ComponentError("hierarchy deeper than 16 levels (recursive component?)")
    c = reg.get(comp) if isinstance(comp, str) else comp
    env = _pdk_env(pdk)
    vals: dict[str, Any] = {}
    for k, spec in c.params.items():
        vals[k] = spec["default"]
    for k, v in (params or {}).items():
        if k not in c.params:
            raise ComponentError(f"{path}: {c.name} has no parameter {k!r}")
        vals[k] = v
    for k, spec in c.params.items():
        if "min" in spec and vals[k] < spec["min"] or "max" in spec and vals[k] > spec["max"]:
            raise ComponentError(f"{path}: {k}={vals[k]} outside [{spec.get('min')}, {spec.get('max')}]")
    scope = {**env, **vals}
    ports = []
    for p in c.ports:
        w = expr.evaluate(p.get("width", 1), scope)
        ports.append({**p, "width": int(w), "domain": p.get("domain", "digital")})
    children = []
    for inst in c.instances:
        child_c = reg.get(inst["component"])
        bound = {k: expr.evaluate(v, scope) for k, v in inst.get("params", {}).items()}
        children.append(elaborate(reg, child_c, bound, pdk, f"{path}.{inst['name']}", depth + 1) | {"inst": inst["name"]})
    errors, warnings = [], []
    for r in c.rules:
        try:
            if not expr.evaluate(r["expr"], scope):
                errors.append(f"{path}: rule failed: {r.get('message', r['expr'])}")
        except expr.ExprError as e:
            errors.append(f"{path}: rule error: {e}")
    # connections: endpoint "port" (own) or "inst.port"
    pmap = {p["name"]: p for p in ports}
    cmap = {ch["inst"]: {p["name"]: p for p in ch["ports"]} for ch in children}
    driven: dict[str, str] = {}
    used_in: set[str] = set()

    def ep(s: str) -> dict[str, Any]:
        if "." in s:
            i, p = s.split(".", 1)
            if i not in cmap or p not in cmap[i]:
                raise ComponentError(f"{path}: unknown endpoint {s!r}")
            return cmap[i][p]
        if s not in pmap:
            raise ComponentError(f"{path}: unknown port {s!r}")
        return pmap[s]

    for a, b in c.connections:
        pa, pb = ep(a), ep(b)
        if pa["width"] != pb["width"]:
            errors.append(f"{path}: width mismatch {a}[{pa['width']}] - {b}[{pb['width']}]")
        if pa["domain"] != pb["domain"] and "power" not in (pa["domain"], pb["domain"]):
            warnings.append(f"{path}: {a} ({pa['domain']}) connected to {b} ({pb['domain']}): needs a converter model")
        for s, p in ((a, pa), (b, pb)):
            is_driver = ("." in s and p["dir"] == "out") or ("." not in s and p["dir"] == "in")
            other = b if s == a else a
            if is_driver and p["domain"] == "digital":
                if other in driven and driven[other] != s:
                    errors.append(f"{path}: {other} driven by both {driven[other]} and {s}")
                driven[other] = s
            used_in.add(s)
    for ch in children:
        for p in ch["ports"]:
            s = f"{ch['inst']}.{p['name']}"
            if p["dir"] == "in" and s not in used_in:
                warnings.append(f"{path}: input {s} unconnected")
    for ch in children:
        errors += ch["errors"]
        warnings += ch["warnings"]
    roll: dict[str, float | None] = {}
    for m in ("area_um2", "power_uw", "transistors", "energy_pj", "delay_ns"):
        if m in c.models:
            roll[m] = float(expr.evaluate(c.models[m], scope))
        elif children and m != "delay_ns":
            vals_m = [ch["rollup"].get(m) for ch in children]
            roll[m] = sum(v for v in vals_m if v is not None) if any(v is not None for v in vals_m) else None
        else:
            roll[m] = None
    return {"path": path, "component": c.key, "name": c.name, "category": c.category, "params": vals,
            "ports": ports, "children": children, "connections": c.connections, "rollup": roll,
            "behavior": c.behavior, "backend": c.backend, "view": c.view, "errors": errors,
            "warnings": warnings, "fidelity": {"rollup": "analytical", "simulation": "functional"}}


# ------------------------------------------------------------------ functional simulation
def simulate(tree: dict[str, Any], stimulus: list[dict[str, int | float]], max_iter: int = 50) -> list[dict[str, Any]]:
    """Cycle-based functional simulation of the digital/behavioural leaves of an elaborated
    tree. ``stimulus`` is one dict of top-level input values per cycle. Returns the top-level
    outputs per cycle (sampled before the clock edge)."""
    leaves: list[dict[str, Any]] = []
    nets: dict[str, str] = {}                               # endpoint -> net representative

    def find(x: str) -> str:
        while nets.get(x, x) != x:
            x = nets[x]
        return x

    def walk(node: dict[str, Any], prefix: str) -> None:
        for a, b in node["connections"]:
            ra = find(prefix + a if "." in a else (prefix + a if prefix else a))
            rb = find(prefix + b if "." in b else (prefix + b if prefix else b))
            if ra != rb:
                nets[ra] = rb
        for ch in node["children"]:
            sub = prefix + ch["inst"] + "."
            # child's own ports are the same nets as "<inst>.<port>" seen from the parent
            if ch["children"]:
                walk(ch, sub)
            elif ch["behavior"]:
                leaves.append({"node": ch, "prefix": sub})
            else:
                raise ComponentError(f"{ch['path']} ({ch['component']}) has no functional behaviour; "
                                     f"backend {ch['backend']!r} is needed")

    if tree["children"]:
        walk(tree, "")
    elif tree["behavior"]:
        leaves.append({"node": tree, "prefix": ""})
    state: dict[str, dict[str, int]] = {}
    for lf in leaves:
        n = lf["node"]
        env = {**n["params"]}
        state[lf["prefix"]] = {k: int(expr.evaluate(v.get("reset", 0), env)) for k, v in n["behavior"].get("state", {}).items()}
    values: dict[str, Any] = {}
    out_ports = [p["name"] for p in tree["ports"] if p["dir"] == "out"]
    trace = []
    for stim in stimulus:
        for k, v in stim.items():
            values[find(k)] = v
        for _ in range(max_iter):                            # settle combinational logic
            changed = False
            for lf in leaves:
                n, pre = lf["node"], lf["prefix"]
                env = {**n["params"], **state[pre]}
                for p in n["ports"]:
                    if p["dir"] != "out":
                        env[p["name"]] = values.get(find(pre + p["name"]), 0)
                for port, e in n["behavior"].get("outputs", {}).items():
                    v = expr.evaluate(e, env)
                    key = find(pre + port)
                    if values.get(key) != v:
                        values[key] = v
                        changed = True
            if not changed:
                break
        else:
            raise ComponentError("combinational loop did not settle")
        trace.append({p: values.get(find(p), 0) for p in out_ports})
        for lf in leaves:                                    # clock edge
            n, pre = lf["node"], lf["prefix"]
            if n["behavior"]["type"] != "seq":
                continue
            env = {**n["params"], **state[pre]}
            for p in n["ports"]:
                if p["dir"] != "out":
                    env[p["name"]] = values.get(find(pre + p["name"]), 0)
            state[pre] = {k: int(expr.evaluate(e, env)) for k, e in n["behavior"]["next"].items()}
    return trace


# ------------------------------------------------------------------ views
def block_graph(tree: dict[str, Any]) -> dict[str, Any]:
    nodes = [{"id": ch["inst"], "label": ch["name"], "category": ch["category"], "color": ch["view"].get("color"),
              "area_um2": ch["rollup"].get("area_um2"), "composite": bool(ch["children"]),
              "ports": ch["ports"]} for ch in tree["children"]]
    edges = [{"from": a, "to": b} for a, b in tree["connections"]]
    return {"top": tree["name"], "ports": tree["ports"], "nodes": nodes, "edges": edges}


def block_svg(tree: dict[str, Any]) -> str:
    from html import escape
    g = block_graph(tree)
    n = max(1, len(g["nodes"]))
    cols = min(4, n)
    W = 160 + cols * 170
    H = 90 + ((n + cols - 1) // cols) * 110
    pos = {}
    body = [f'<text x="12" y="22" style="font:600 13px ui-monospace;fill:#cfd8e3">{escape(tree["name"])} · '
            f'{escape(tree["component"])} · area {tree["rollup"].get("area_um2") or 0:.1f} µm² (analytical)</text>']
    ins = [p for p in tree["ports"] if p["dir"] != "out"]
    outs = [p for p in tree["ports"] if p["dir"] == "out"]
    for k, p in enumerate(ins):
        pos[p["name"]] = (60, 60 + k * 26)
        body.append(f'<text x="8" y="{64 + k * 26}" style="font:11px ui-monospace;fill:#8b97a4">{escape(p["name"])}</text>')
    for k, p in enumerate(outs):
        pos[p["name"]] = (W - 60, 60 + k * 26)
        body.append(f'<text x="{W - 54}" y="{64 + k * 26}" style="font:11px ui-monospace;fill:#8b97a4">{escape(p["name"])}</text>')
    for i, nd in enumerate(g["nodes"]):
        x, y = 100 + (i % cols) * 170, 50 + (i // cols) * 110
        pos[nd["id"]] = (x + 65, y + 35)
        body.append(f'<rect x="{x}" y="{y}" width="130" height="70" rx="8" fill="#18202a" stroke="{nd["color"] or "#6ea0ff"}"'
                    f'{" stroke-dasharray=&quot;5 3&quot;" if nd["composite"] else ""}/>'
                    f'<text x="{x + 8}" y="{y + 20}" style="font:600 12px ui-monospace;fill:#e6edf3">{escape(nd["id"])}</text>'
                    f'<text x="{x + 8}" y="{y + 38}" style="font:11px ui-monospace;fill:#8b97a4">{escape(nd["label"])}</text>'
                    f'<text x="{x + 8}" y="{y + 56}" style="font:11px ui-monospace;fill:#8b97a4">{(nd["area_um2"] or 0):.1f} µm²</text>')
    for e in g["edges"]:
        a, b = (pos.get(e["from"].split(".")[0]), pos.get(e["to"].split(".")[0]))
        if a and b:
            body.append(f'<path d="M{a[0]} {a[1]} L{b[0]} {b[1]}" stroke="#6ea0ff" stroke-width="1.2" opacity=".6" fill="none"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'style="background:#0e141b">{"".join(body)}</svg>')


def to_verilog(reg: Registry, tree: dict[str, Any]) -> str:
    """Structural Verilog for a digital composite (children instantiated by component name).
    Leaf modules are emitted from their behaviour expressions where they translate directly."""
    if any(p["domain"] != "digital" for p in tree["ports"]):
        raise ComponentError("structural Verilog export needs an all-digital interface")
    lines = [f"// Generated by ChipLab Component SDK from {tree['component']} (functional fidelity)"]
    ports = ", ".join(f"{'input' if p['dir'] == 'in' else 'output'} wire [{p['width'] - 1}:0] {p['name']}"
                      for p in tree["ports"])
    lines.append(f"module {tree['name']} ({ports});")
    for ch in tree["children"]:
        for p in ch["ports"]:
            n = f"{ch['inst']}_{p['name']}"
            lines.append(f"    wire [{p['width'] - 1}:0] {n};")
    pdir = {p["name"]: p["dir"] for p in tree["ports"]}
    cdir = {f"{ch['inst']}.{p['name']}": p["dir"] for ch in tree["children"] for p in ch["ports"]}

    def drives(e: str) -> bool:          # endpoint supplies the value on this net
        return cdir.get(e) == "out" if "." in e else pdir.get(e) == "in"

    for a, b in tree["connections"]:
        src, dst = (a, b) if drives(a) else (b, a)
        lines.append(f"    assign {dst.replace('.', '_')} = {src.replace('.', '_')};")
    for ch in tree["children"]:
        conns = ", ".join(f".{p['name']}({ch['inst']}_{p['name']})" for p in ch["ports"])
        prm = ", ".join(f".{k}({v})" for k, v in ch["params"].items() if isinstance(v, int))
        lines.append(f"    {ch['name']} {'#(' + prm + ') ' if prm else ''}{ch['inst']} ({conns});")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"
