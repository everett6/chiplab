"""Competition evaluation report, requirement tracker and submission checklist for the PEM ASIC.

Every number here comes from a run (protocol tests, lockstep regression, mutation check, lint,
synthesis) made through chiplab.asic, either now or stored by an earlier run with its RTL hash.
Steps that did not run are reported as not run, never as passed.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

from chiplab.asic import flow, isa

RESULTS = Path.home() / ".cache" / "chiplab_asic" / "results"


def _save(kind: str, rec: dict[str, Any]) -> None:
    from chiplab.asic.verify import rtl_hash
    RESULTS.mkdir(parents=True, exist_ok=True)
    rec = {**rec, "rtl_hash": rtl_hash(), "timestamp": dt.datetime.now().isoformat(timespec="seconds")}
    (RESULTS / f"{kind}.json").write_text(json.dumps(rec, indent=1, default=str))
    with (RESULTS / "history.jsonl").open("a") as f:
        f.write(json.dumps({"kind": kind, **{k: v for k, v in rec.items() if k in (
            "passed", "total", "seeds", "killed", "mutants", "rtl_hash", "timestamp", "ok", "cells", "status")}}) + "\n")


def load(kind: str) -> dict[str, Any] | None:
    """Last stored result of a kind, marked stale if the RTL changed since."""
    p = RESULTS / f"{kind}.json"
    if not p.exists():
        return None
    from chiplab.asic.verify import rtl_hash
    rec = json.loads(p.read_text())
    rec["stale"] = rec.get("rtl_hash") != rtl_hash()
    return rec


def history(limit: int = 50) -> list[dict[str, Any]]:
    p = RESULTS / "history.jsonl"
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines()[-limit:]]


def run_protocols(backend: str = "rtl") -> dict[str, Any]:
    from chiplab.asic import protocols
    r = protocols.run_all(backend)
    _save(f"protocols_{backend}", r)
    return r


def run_regression(seeds: int = 40, cycles: int = 4000) -> dict[str, Any]:
    from chiplab.asic import verify
    r = verify.regression(seeds=seeds, cycles=cycles, coverage=True)
    r.pop("functional_hits", None)
    _save("regression", r)
    return r


def run_mutation(seeds: int = 20, cycles: int = 3000) -> dict[str, Any]:
    from chiplab.asic import verify
    r = verify.mutation_check(seeds=seeds, cycles=cycles)
    _save("mutation", r)
    return r


def run_lint() -> dict[str, Any]:
    r = flow.lint()
    _save("lint", r)
    return r


def run_synthesis() -> dict[str, Any]:
    from chiplab.asic import eda
    r = eda.synthesize_generic()
    _save("synthesis", r)
    return r


def run_synthesis_ihp(corner: str = "slow", clock_mhz: float = 50.0) -> dict[str, Any]:
    from chiplab.asic import eda
    r = eda.synthesize_ihp(corner, clock_mhz)
    _save("synthesis_ihp", r)
    return r


def run_icarus_tt() -> dict[str, Any]:
    from chiplab.asic import eda
    r = eda.icarus_tt_test()
    _save("icarus_tt", r)
    return r


def run_formal(depth: int = 20, mode: str = "bmc") -> dict[str, Any]:
    from chiplab.asic import eda
    r = eda.formal_check(depth, mode)
    _save("formal", r)
    return r


def run_gate_level(seeds: int = 20, cycles: int = 4000) -> dict[str, Any]:
    from chiplab.asic import eda
    r = eda.gate_level_regression(seeds, cycles)
    _save("gate_sim", r)
    return r


# ------------------------------------------------------------------ documents
def architecture_md() -> str:
    rb = flow.register_bits()
    return f"""# PEM: Protocol Emulation Machine

A reprogrammable pin processor for the {flow.COMPETITION['name']} ({flow.COMPETITION['process']}).

## How it works
- **Four lanes** (independent state machines) execute PEM-16 programs from one shared 32-word
  instruction memory. Each lane has its own clock divider, X/Y registers, 16-bit output and
  input shift registers (OSR/ISR), a two-entry TX and RX FIFO to the host, and pin mapping.
- **Shared resources**: instruction memory (lanes take address windows), 16-pin space, eight
  IRQ flags for lane-to-lane synchronisation (flag 7 drives the host interrupt pin).
- **Serial datapath per lane** (the distinctive part): NRZI line coding with optional
  differential drive, USB-style bit stuffing/destuffing with a sticky error flag and a fixed
  stuffed-bit length, autopull for gap-free bit streams, and a CRC engine (width 1-32,
  programmable polynomial/seed, MSB- or LSB-first) that snoops serial data bits.
- **Host port**: SPI mode 0 on three inputs; it loads programs, configures lanes, moves FIFO
  data and reads status/debug registers, so every protocol is defined after fabrication.

## Programs shipped
UART TX/RX (8N1), SPI controller modes 0-3, I2C controller (open drain via pin directions,
repeated start, ACK/NACK), low-speed USB packet transmitter (NRZI + stuffing + CRC16 + EOP).

## Size
{rb['total']} flip-flops from the RTL declarations (pre-synthesis; {rb['provenance']}). Mapped area,
timing and power need the IHP synthesis and hardening flow; see the evaluation report.

## Interface
| Pin | Use |
|---|---|
| ui_in[0..2] | host SCK, CS_n, MOSI |
| ui_in[3..7] | extra inputs (pins 8-12) |
| uo_out[0..1] | host MISO, IRQ |
| uo_out[2..7] | extra outputs (pins 8-13) |
| uio[0..7] | programmable bidirectional pins 0-7 |

{isa.ISA_DOC}
"""


def verification_md() -> str:
    return """# Verification methodology

1. **Independent reference model** (`chiplab/asic/iss.py`): a cycle-accurate Python model
   written from the ISA document, not from the RTL.
2. **Lockstep comparison**: the same bench script runs on the Verilator RTL bench and on the
   model; every pin change, register read and bench event must match exactly.
3. **Constrained-random programs**: legal random instruction streams and structured serial
   loops (line coder, stuffing, CRC, autopull) with random lane configuration, host traffic,
   input stimulus and pin loopback.
4. **Directed protocol tests with independent decoders**: UART, SPI and USB decoders sample
   the pin waveform; the I2C target is a separate bench model; none share code with the RTL
   or the model's datapath.
5. **Mutation testing**: deliberate bugs are injected into copies of the RTL; the suite must
   catch each one. This measures the tests, not the design.
6. **Assertions**: immediate assertions on FIFO bounds compiled in with `--assert`.
7. **Coverage**: functional bins (opcodes, jump conditions taken/not taken, stalls, line-coder
   modes, stuffing events) from the model; line/toggle coverage from Verilator.
8. **Formal (bounded)**: SymbiYosys BMC of assertions A1-A4 from reset with every bus and pin
   input unconstrained. A pass means no input sequence violates them within the checked depth;
   it is not an unbounded proof. A deliberately false property is found in a few cycles, so
   the setup can fail.
9. **Gate level**: the Yosys netlist (generic gates, zero delay) runs the same random scripts
   and directed protocol tests and must match the model event for event.

Passing self-generated tests is evidence, not proof: the model and RTL share one author and one
reading of the ISA. Not yet done: unbounded (k-induction) proofs, timing-annotated gate-level
simulation with IHP cells, cocotb run of the Tiny Tapeout top level, and directed tests for
JTAG, SWD, PS/2, CAN and USB receive.
"""


# ------------------------------------------------------------------ report
LIMITATIONS = [
    "10 Mbit Ethernet: no hardware Manchester coder; software Manchester cannot reach 10 Mb/s at plausible clocks (CRC-32 is supported).",
    "USB: a stuffed bit owed after the final CRC bit is not emitted before EOP (rare: CRC ending in six 1s).",
    "UART RX does not check the stop bit (no framing-error flag).",
    "I2C: no clock stretching or multi-controller arbitration.",
    "mov pins/pindirs writes 8 pins from out_base; programs sharing pins with other lanes must use set/out.",
    "Instruction memory is 32 words shared by four lanes; three protocols at once used 28.",
    "IHP area and delay come from Yosys+ABC mapping only: no placement, routing, clock tree or sign-off STA; power not estimated.",
    "The IHP-mapped netlist is not simulated (needs the sg13g2 cell Verilog models); gate-level runs use the generic netlist.",
    "Formal is a bounded check (BMC from reset), not an unbounded proof.",
]


def _r8_status() -> str:
    ihp = load("synthesis_ihp")
    if ihp and ihp.get("ok"):
        return f"{ihp['tiles']['8x4']['verdict']} (estimate)"
    return "unknown"


def _r8_evidence() -> str:
    ihp, sy = load("synthesis_ihp"), load("synthesis")
    if ihp and ihp.get("ok"):
        t = ihp["tiles"]["8x4"]
        return (f"IHP sg13g2 cell area {ihp['cell_area_um2']:.0f} um2 = {t['cell_density'] * 100:.0f}% of the 8x4 tile "
                f"({t['area_um2']} um2, official outline); placement and routing not run")
    if sy and sy.get("ok"):
        return f"generic Yosys synthesis: {sy['cells']} cells, {sy['flops']} flip-flops; IHP mapping not run"
    return "needs synthesis + placement with the IHP flow"


def requirements(protocols_rec: dict | None, regression: dict | None, flow_st: dict) -> list[dict[str, Any]]:
    def proto_ok(name_prefix: str) -> bool | None:
        if not protocols_rec:
            return None
        rs = [r for r in protocols_rec["results"] if r["name"].startswith(name_prefix)]
        return all(r["passed"] for r in rs) if rs else None

    gds_ready = any(s["step"] == "gds" and s["available"] for s in flow_st["steps"])
    return [
        {"id": "R1", "requirement": "Reprogrammable after fabrication", "status": "met",
         "evidence": "programs and lane configuration loaded over the SPI host port"},
        {"id": "R2", "requirement": "UART", "status": _st(proto_ok("uart")), "evidence": "uart_tx, uart_rx"},
        {"id": "R3", "requirement": "SPI", "status": _st(proto_ok("spi")), "evidence": "spi_mode0..3"},
        {"id": "R4", "requirement": "I2C", "status": _st(proto_ok("i2c")), "evidence": "i2c, i2c_nack"},
        {"id": "R5", "requirement": "Low-speed USB (stretch)", "status": "partial" if proto_ok("usb") else _st(proto_ok("usb")),
         "evidence": "transmit verified; receive datapath exists without a directed test"},
        {"id": "R6", "requirement": "10 Mbit Ethernet (stretch)", "status": "not met",
         "evidence": "no Manchester coder in hardware"},
        {"id": "R7", "requirement": "Not fixed-function peripherals", "status": "met",
         "evidence": "one programmable lane architecture runs every protocol"},
        {"id": "R8", "requirement": "Fits 8x4 tiles on IHP 130 nm", "status": _r8_status(),
         "evidence": _r8_evidence()},
        {"id": "R9", "requirement": "Tiny Tapeout flow / GDS / precheck", "status": "ready to run" if gds_ready else "not run",
         "evidence": "flow tools not installed here" if not gds_ready else "tools present"},
        {"id": "R10", "requirement": "Strong verification", "status": "in progress" if regression else "not run",
         "evidence": f"{regression['passed']}/{regression['seeds']} lockstep seeds" if regression else "run the regression"},
        {"id": "R11", "requirement": f"Deadline {flow.COMPETITION['deadline']}",
         "status": f"{(dt.date.fromisoformat(flow.COMPETITION['deadline']) - dt.date.today()).days} days left",
         "evidence": "calendar"},
    ]


def _st(ok: bool | None) -> str:
    return "not run" if ok is None else "met" if ok else "failing"


def competition_report() -> dict[str, Any]:
    prot = load("protocols_rtl")
    reg = load("regression")
    mut = load("mutation")
    lint = load("lint")
    syn, frm, gls, ihp = load("synthesis"), load("formal"), load("gate_sim"), load("synthesis_ihp")
    fst = flow.flow_status()
    rb = flow.register_bits()
    na = {"value": None, "status": "not run", "needs": "IHP synthesis / hardening flow"}
    return {
        "competition": flow.COMPETITION,
        "design": {"top": flow.TOP, "lanes": isa.N_LANES, "imem_words": isa.IMEM_WORDS,
                   "register_bits": rb},
        "synthesis": _brief(syn, ["ok", "cells", "flops", "mapping", "tool", "note"]),
        "synthesis_ihp": _brief(ihp, ["ok", "corner", "liberty", "cell_area_um2", "sequential_area_um2", "cells",
                                      "flops", "clock_mhz", "logic_delay_ns", "logic_slack_ns", "tiles",
                                      "tile_source", "pdk_source", "fidelity"]),
        **_ihp_metrics(ihp, na),
        "routing_congestion": {**na, "needs": "placement and routing (OpenROAD / TT flow)"},
        "power_mw": {**na, "needs": "switching activity + placed netlist (OpenROAD / TT flow)"},
        "memory_utilization": {"imem_words": isa.IMEM_WORDS,
                               "largest_concurrent_program_words": 28,
                               "note": "UART TX + UART RX + SPI mode 0 together (directed test)"},
        "protocol_tests": _brief(prot, ["passed", "total", "backend"]),
        "verification": {"regression": _brief(reg, ["passed", "seeds", "cycles_per_seed", "functional", "code", "assertions"]),
                         "mutation": _brief(mut, ["killed", "mutants", "results"]),
                         "lint": _brief(lint, ["ok", "warnings"]),
                         "formal": _brief(frm, ["ok", "status", "mode", "depth", "engine", "meaning"]),
                         "icarus_tt": _brief(load("icarus_tt"), ["ok", "detail", "simulator", "what"]),
                         "gate_sim": _brief(gls, ["ok", "passed", "seeds", "cycles_per_seed", "protocols", "fidelity"])},
        "flow": fst, "requirements": requirements(prot, reg, fst), "limitations": LIMITATIONS,
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
    }


def _ihp_metrics(ihp: dict | None, na: dict[str, Any]) -> dict[str, Any]:
    """Metrics backed by the IHP-mapped synthesis, labelled as estimates; otherwise not run."""
    if not ihp or not ihp.get("ok"):
        return {"tile_utilization": na, "mapped_cell_area_um2": na, "timing_margin_ns": na, "fmax_mhz": na}
    est = f"estimate: Yosys+ABC on IHP sg13g2 {ihp['corner']} corner" + (" (stale: RTL changed)" if ihp.get("stale") else "")
    t8 = ihp["tiles"].get("8x4", {})
    return {
        "tile_utilization": {"value": t8.get("cell_density"), "status": est,
                             "needs": "cell area / 8x4 tile area, before placement; placement decides the fit"},
        "mapped_cell_area_um2": {"value": ihp["cell_area_um2"], "status": est, "needs": "liberty cell area"},
        "timing_margin_ns": {"value": ihp["logic_slack_ns"], "status": est,
                             "needs": f"logic-only at {ihp['clock_mhz']} MHz; wires, clock tree, setup need STA"},
        "fmax_mhz": {"value": round(1000 / ihp["logic_delay_ns"]) if ihp.get("logic_delay_ns") else None,
                     "status": est, "needs": "upper bound from logic delay only"},
    }


def _brief(rec: dict | None, keys: list[str]) -> dict[str, Any]:
    if not rec:
        return {"status": "not run"}
    out = {k: rec.get(k) for k in keys}
    out.update(timestamp=rec.get("timestamp"), stale=rec.get("stale"), rtl_hash=rec.get("rtl_hash"))
    return out


def checklist(repo: Path | None = None) -> list[dict[str, Any]]:
    repo = repo or Path(__file__).resolve().parents[3]
    try:
        remote = subprocess.run(["git", "-C", str(repo), "remote", "get-url", "origin"], capture_output=True,
                                text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        remote = ""
    prot, reg = load("protocols_rtl"), load("regression")
    fst = flow.flow_status()
    ran = {s["step"]: s for s in fst["steps"]}

    def item(name: str, ok: bool | None, evidence: str) -> dict[str, Any]:
        return {"item": name, "status": "done" if ok else "missing" if ok is False else "not checked",
                "evidence": evidence}

    rtl = all((flow.RTL / f).exists() for f in flow.SOURCES)
    return [
        item("Public source repository", None if not remote else None,
             f"remote {remote or 'none'}; public visibility not checked (needs GitHub, not queried)"),
        item("RTL available", rtl, ", ".join(flow.SOURCES)),
        item("Passing tests", bool(prot and prot["passed"] == prot["total"] and not prot.get("stale")),
             f"protocol tests {prot['passed']}/{prot['total']}" if prot else "protocol tests not run"),
        item("Architecture documentation", True, "report.architecture_md() -> docs/info.md on export"),
        item("Instruction-set documentation", True, "isa.ISA_DOC -> docs/isa.md on export"),
        item("Assembler / toolchain", True, "chiplab.asic.assembler (two-pass, with disassembler)"),
        item("Example protocol programs", True, "UART, SPI 0-3, I2C, USB-LS -> programs/*.pem"),
        item("Verification methodology", bool(reg), "docs/verification.md; regression " +
             (f"{reg['passed']}/{reg['seeds']}" if reg else "not run")),
        item("Successful Tiny Tapeout flow", False if not ran["gds"]["available"] else None,
             "hardening not run: " + ran["gds"]["status"]),
        item("Passing GDS / precheck", False if not ran["tt_precheck"]["available"] else None,
             "precheck not run: " + ran["tt_precheck"]["status"]),
    ]


def technical_report_md() -> str:
    r = competition_report()
    lines = [f"# PEM technical report ({r['generated']})", "", architecture_md(), verification_md(),
             "## Requirement tracker", "", "| ID | Requirement | Status | Evidence |", "|---|---|---|---|"]
    lines += [f"| {q['id']} | {q['requirement']} | {q['status']} | {q['evidence']} |" for q in r["requirements"]]
    lines += ["", "## Results", "```json", json.dumps({k: r[k] for k in ("protocol_tests", "verification")},
                                                        indent=1, default=str)[:6000], "```", "",
              "## Known limitations", *[f"- {x}" for x in r["limitations"]]]
    return "\n".join(lines) + "\n"
