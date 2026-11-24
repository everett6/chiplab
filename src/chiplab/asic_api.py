"""Shared entry points for the ASIC workspaces (competition ASIC, memory compiler, component
SDK, multi-level simulation status). Used by the MCP server and the web app so both expose
the same behaviour, limits and fidelity labels.

Safety: no shell, no eval/exec. External tools run only through fixed argument lists inside
chiplab.asic / chiplab.memory_compiler. Expensive runs are explicit calls with bounded sizes.
Exports go only under the project's ``projects/`` directory.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LIMITS = {"cycles": 20000, "seeds": 100, "regression_cycles": 10000, "stimulus": 2000, "mem_ops": 2000,
          "source_chars": 20000}


def _bound(name: str, v: int, lo: int = 1) -> int:
    v = int(v)
    if not lo <= v <= LIMITS[name]:
        raise ValueError(f"{name} must be {lo}..{LIMITS[name]}")
    return v


# ------------------------------------------------------------------ competition ASIC
def asic_architecture() -> dict[str, Any]:
    from chiplab.asic import flow, isa, protocols, report
    return {"competition": flow.COMPETITION, "isa_markdown": isa.ISA_DOC,
            "architecture_markdown": report.architecture_md(), "register_bits": flow.register_bits(),
            "lanes": isa.N_LANES, "imem_words": isa.IMEM_WORDS, "fifo_depth": isa.FIFO_DEPTH,
            "programs": {"uart_tx": protocols.uart_tx()[0], "uart_rx": protocols.uart_rx()[0],
                         "spi_mode0": protocols.spi_controller(0)[0], "i2c": protocols.i2c_controller()[0],
                         "usb_ls_tx": protocols.usb_ls_tx()[0]},
            "examples": _examples(), "protocol_tests": list(protocols.TESTS), "planned_protocols": protocols.PLANNED}


def _examples() -> dict[str, dict[str, Any]]:
    """Protocol Lab presets: program, its lane registers, and a bench that exercises it."""
    from chiplab.asic import protocols as P
    uart_rx_bench = "S 4\nC 40\n" + P._uart_frame(0x5A, 2, 16) + "C 60\nR 40"
    ex = {"uart_tx": (P.uart_tx(pin=0), "W 40 55\nC 200\nW 40 a3"),
          "uart_rx": (P.uart_rx(pin=2), uart_rx_bench),
          "spi_mode0": (P.spi_controller(0), "LB 0 3\nW 40 9600\nC 200\nR 40"),
          "i2c": (P.i2c_controller(), f"I2C 5 4 42\nW 40 {P._i2c_word(0x42 << 1, start=True, stop=True):x}\nC 400\nR 40"),
          "usb_ls_tx": (P.usb_ls_tx(), "W 40 1e\nW 40 c380\nC 300\nW 40 ffff\nC 300\nW 40 8001")}
    return {k: {"source": src, "regs": regs, "bench": bench} for k, ((src, regs), bench) in ex.items()}


def asic_assemble(source: str) -> dict[str, Any]:
    from chiplab.asic import assembler
    if len(source) > LIMITS["source_chars"]:
        raise ValueError("source too long")
    try:
        p = assembler.assemble(source)
    except assembler.AsmError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "start": p.start, "wrap": p.wrap, "labels": p.labels,
            "words": [{"addr": a, "word": f"{w:04x}", "asm": assembler.disassemble(w)} for a, w in sorted(p.words.items())]}


def asic_simulate(source: str, lane_regs: dict[str, int] | None = None, cycles: int = 2000,
                  script: str = "", lane: int = 0, clkdiv: int = 1) -> dict[str, Any]:
    """Run a PEM-16 program on the reference model (fidelity: reference model, not RTL).
    `script` may add bench lines (W/R/C/S/LB/I2C) after the program is loaded; leading S/LB
    lines are applied before the lanes start, so an input line can idle high."""
    from chiplab.asic import assembler, iss, isa, protocols, verify
    cycles = _bound("cycles", cycles)
    if not 0 <= lane < isa.N_LANES:
        raise ValueError("lane must be 0-3")
    a = asic_assemble(source)
    if not a["ok"]:
        return a
    p = assembler.assemble(source)
    regs = {k: int(v) for k, v in (lane_regs or {}).items()}
    for k in regs:
        if k not in isa.LANE_REGS:
            raise ValueError(f"unknown lane register {k!r}")
    bench = verify.load_script([(lane, p, {"clkdiv": clkdiv, **regs})], 1 << lane)
    allowed = re.compile(r"^(W [0-9a-fA-F]+ [0-9a-fA-F]+|R [0-9a-fA-F]+|C \d+|S [0-9a-fA-F]+|LB \d+ \d+|I2C \d+ \d+ [0-9a-fA-F]+)$")
    extra = [ln.strip() for ln in script.splitlines() if ln.strip()]
    bad = [ln for ln in extra if not allowed.match(ln)]
    if bad:
        raise ValueError(f"bad bench lines: {bad[:3]}")
    total = sum(int(ln.split()[1]) for ln in extra if ln.startswith("C ")) + cycles
    if total > LIMITS["cycles"] * 2:
        raise ValueError("too many cycles")
    env = []                                         # leading S/LB lines set the pin environment
    while extra and extra[0].split()[0] in ("S", "LB"):   # before the lanes start (idle lines)
        env.append(extra.pop(0))
    m = iss.PEM()
    pre = "".join(ln + "\n" for ln in env)
    lines = iss.run_script(pre + bench + "\n".join(extra) + f"\nC {cycles}\n", m)
    return {"ok": True, "fidelity": "reference model (Python, cycle accurate)", "events": lines[:4000],
            "waveform": protocols.waveform(lines), "oe_waveform": protocols.waveform(lines, 1),
            "reads": protocols.reads(lines), "cycles": m.cycle, "assembled": a,
            "lane_state": {k: getattr(m.lanes[lane], k) for k in ("pc", "x", "y", "osr", "isr", "crc")}}


def asic_run(kind: str, seeds: int = 20, cycles: int = 3000, backend: str = "rtl") -> dict[str, Any]:
    """Explicit runs: protocols | regression | mutation | lint | tt_smoke | synthesis | formal | gate_sim |
    hardening_import.
    For formal, `cycles` (if <= 40) is the BMC depth; otherwise depth 20."""
    from chiplab.asic import report, verify
    if kind == "protocols":
        if backend not in ("rtl", "model"):
            raise ValueError("backend rtl or model")
        return report.run_protocols(backend)
    if kind == "regression":
        return report.run_regression(_bound("seeds", seeds), _bound("regression_cycles", cycles, 100))
    if kind == "mutation":
        return report.run_mutation(_bound("seeds", seeds), _bound("regression_cycles", cycles, 100))
    if kind == "lint":
        return report.run_lint()
    if kind == "tt_smoke":
        r = verify.tt_smoke()
        report._save("tt_smoke", r)
        return r
    if kind == "synthesis":
        return report.run_synthesis()
    if kind == "synthesis_ihp":
        return report.run_synthesis_ihp("slow" if backend not in ("typ", "slow") else backend)
    if kind == "icarus_tt":
        return report.run_icarus_tt()
    if kind == "formal":
        return report.run_formal(int(cycles) if 1 <= int(cycles) <= 40 else 20)
    if kind == "gate_sim":
        return report.run_gate_level(_bound("seeds", seeds), _bound("regression_cycles", cycles, 100))
    if kind == "hardening_import":   # reads the latest completed TT gds run of the local repo clone via gh
        return report.run_hardening_import()
    raise ValueError("kind must be protocols, regression, mutation, lint, tt_smoke, synthesis, synthesis_ihp, "
                     "icarus_tt, formal, gate_sim or hardening_import")


def asic_report() -> dict[str, Any]:
    from chiplab.asic import report
    return {"report": report.competition_report(), "checklist": report.checklist(), "history": report.history(30)}


def asic_export(name: str = "jane_street_pem") -> dict[str, Any]:
    from chiplab.asic import flow
    if not re.fullmatch(r"[a-z0-9_]{1,40}", name):
        raise ValueError("name: lowercase letters, digits, underscore")
    return flow.export_tt_project(ROOT / "projects" / name)


def asic_technical_report() -> str:
    from chiplab.asic import report
    return report.technical_report_md()


# ------------------------------------------------------------------ memory compiler
def memory_compile(spec: dict[str, Any], views: bool = False) -> dict[str, Any]:
    from chiplab.memory_compiler import compiler
    from chiplab.memory_compiler import views as V
    s = compiler.spec_from_dict(spec)
    r = compiler.compile_memory(s)
    if views:
        r["views"] = V.all_views(s, r)
    return r


def memory_compare(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from chiplab.memory_compiler import compiler
    if not 1 <= len(specs) <= 12:
        raise ValueError("1-12 specs")
    return compiler.compare([compiler.spec_from_dict(s) for s in specs])


def memory_rtl(spec: dict[str, Any], run_tests: bool = True, n_ops: int = 300,
               retention_cycles: int = 0, refresh_interval: int = 0, refresh: bool = True) -> dict[str, Any]:
    from chiplab.memory_compiler import compiler, rtl
    s = compiler.spec_from_dict(spec)
    if s.validate():
        raise ValueError("; ".join(s.validate()))
    if s.words > 65536:
        raise ValueError("RTL generation limited to 65536 words")
    if s.kind in ("dram", "edram") and retention_cycles and s.words > 1024:
        raise ValueError("retention model limited to 1024 words")
    if not run_tests:
        return {"ran": False, "rtl": rtl.gen_rtl(s, retention_cycles, refresh_interval)}
    return rtl.run_tests(s, _bound("mem_ops", n_ops), retention_cycles=retention_cycles,
                         refresh_interval=max(1, refresh_interval) if retention_cycles else 0, refresh=refresh)


# ------------------------------------------------------------------ component SDK
def _registry():
    from chiplab.sdk.components import Registry
    return Registry(ROOT / ".chiplab_components")


def component_list(category: str | None = None) -> list[dict[str, Any]]:
    return _registry().list(category)


def component_get(ref: str) -> dict[str, Any]:
    c = _registry().get(ref)
    return {**c.to_dict(), "key": c.key, "builtin": c.builtin}


def component_create(definition: dict[str, Any]) -> dict[str, Any]:
    c = _registry().register(definition)
    return {"created": c.key, "note": "versions are immutable; register a new version to change it"}


def architecture_elaborate(ref: str, params: dict[str, Any] | None = None, pdk: str = "sky130",
                           svg: bool = False) -> dict[str, Any]:
    from chiplab.sdk.components import block_graph, block_svg, elaborate
    t = elaborate(_registry(), ref, params or {}, pdk)
    out = {"component": t["component"], "rollup": t["rollup"], "ports": t["ports"], "errors": t["errors"],
           "warnings": t["warnings"], "block_graph": block_graph(t), "fidelity": t["fidelity"],
           "children": [{"inst": c["inst"], "component": c["component"], "rollup": c["rollup"]} for c in t["children"]]}
    if svg:
        out["svg"] = block_svg(t)
    return out


def component_simulate(ref: str, stimulus: list[dict[str, Any]], params: dict[str, Any] | None = None) -> dict[str, Any]:
    from chiplab.sdk.components import elaborate, simulate
    if not 1 <= len(stimulus) <= LIMITS["stimulus"]:
        raise ValueError(f"stimulus must have 1..{LIMITS['stimulus']} cycles")
    t = elaborate(_registry(), ref, params or {})
    if t["errors"]:
        return {"ok": False, "errors": t["errors"]}
    return {"ok": True, "fidelity": "functional (cycle-based, behavioural expressions)", "trace": simulate(t, stimulus)}


def architecture_verilog(ref: str, params: dict[str, Any] | None = None) -> str:
    from chiplab.sdk.components import elaborate, to_verilog
    reg = _registry()
    return to_verilog(reg, elaborate(reg, ref, params or {}))


# ------------------------------------------------------------------ multi-level simulation
def simulation_levels() -> list[dict[str, Any]]:
    """What each ASIC simulation level runs with here, and whether it is available."""
    from chiplab.asic import flow
    from chiplab.tools import registry as tools
    ts = flow.tool_status()["tools"]
    ext = tools.status()
    have = lambda k: bool(ts.get(k, {}).get("installed"))  # noqa: E731
    return [
        {"level": 1, "name": "Functional simulation", "engine": "Component SDK cycle simulator; PEM reference model",
         "available": True, "fidelity": "functional"},
        {"level": 2, "name": "RTL simulation", "engine": "Verilator", "available": have("verilator"), "fidelity": "RTL"},
        {"level": 3, "name": "Gate-level simulation", "engine": "Yosys netlist + Verilator/iverilog",
         "available": have("yosys"), "fidelity": "gate", "blocked_by": None if have("yosys") else "yosys not installed"},
        {"level": 4, "name": "Transistor-level simulation", "engine": "ngspice",
         "available": bool(ext.get("ngspice", {}).get("available")), "fidelity": "circuit (SPICE)"},
        {"level": 5, "name": "Synthesis", "engine": "Yosys", "available": have("yosys"), "fidelity": "netlist"},
        {"level": 6, "name": "Static timing analysis", "engine": "OpenROAD / OpenSTA", "available": have("openroad"),
         "fidelity": "signoff-style STA (with PDK liberty)"},
        {"level": 7, "name": "Physical implementation", "engine": "OpenROAD / LibreLane (Tiny Tapeout flow)",
         "available": have("openroad"), "fidelity": "place and route"},
        {"level": 8, "name": "DRC and LVS", "engine": "KLayout / Magic / Netgen",
         "available": bool(ext.get("klayout", {}).get("available")) or have("klayout"), "fidelity": "rule check"},
        {"level": 9, "name": "Power estimation", "engine": "analytical (memory compiler, SDK roll-up); OpenROAD after PnR",
         "available": True, "fidelity": "analytical"},
        {"level": 10, "name": "Thermal analysis", "engine": "ChipLab package thermal solver (finite volume)",
         "available": True, "fidelity": "physics model (2-layer FV)"},
    ]
