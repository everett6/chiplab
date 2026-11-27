"""Physical-design flow for the PEM ASIC on the Tiny Tapeout IHP 130 nm shuttle.

ChipLab does not reimplement synthesis, place and route or signoff. This module
  * exports a Tiny Tapeout project directory (info.yaml, src/, docs/, programs, cocotb test),
  * runs the open tools that are installed (Yosys synthesis/stat; the Tiny Tapeout /
    LibreLane hardening flow when its tooling is present) through fixed argument lists, and
  * reports which steps ran, with their real outputs, and which did not and why.
No area, timing, power, DRC or GDS number is produced unless a tool produced it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from chiplab.asic import assembler, isa, protocols

RTL = Path(__file__).parent / "rtl"
TOP = "tt_um_chiplab_pem"
SOURCES = ["tt_um_chiplab_pem.v", "pem_core.v", "pem_lane.v"]

COMPETITION = {
    "name": "Jane Street Protocol Emulator ASIC Competition",
    "process": "IHP 130 nm open PDK (sg13cmos5l variant per Tiny Tapeout IHP shuttles)",
    "flow": "Tiny Tapeout",
    "max_tiles": "8x4",
    "deadline": "2027-01-18",
    "baseline_protocols": ["UART", "SPI", "I2C"],
    "stretch_protocols": ["USB low speed", "10 Mbit Ethernet"],
    "reprogrammable": True,
    "source": "user-supplied competition brief (not independently verified by ChipLab)",
}

# Steps of the hardening flow, in order, with the tool that performs each.
FLOW_STEPS = [
    ("lint", "verilator", "RTL lint (-Wall)"),
    ("synthesis", "yosys", "RTL synthesis to generic gates"),
    ("synthesis_ihp", "yosys", "technology mapping to IHP sg13g2 standard cells (area, logic delay)"),
    ("formal", "sby", "SymbiYosys bounded model check of the core assertions"),
    ("gate_sim", "yosys", "gate-level lockstep simulation of the synthesized netlist"),
    ("floorplan", "openroad", "floorplan inside the Tiny Tapeout tile area"),
    ("placement", "openroad", "global and detailed placement"),
    ("cts", "openroad", "clock-tree synthesis"),
    ("routing", "openroad", "global and detailed routing"),
    ("sta", "openroad", "static timing analysis (OpenSTA)"),
    ("power", "openroad", "power estimation"),
    ("drc", "klayout", "design-rule check"),
    ("gds", "openroad", "GDS generation"),
    ("tt_precheck", "tt_precheck", "Tiny Tapeout precheck (tt-support-tools)"),
]


OSS_CAD = Path(os.environ.get("CHIPLAB_OSS_CAD", Path.home() / ".local" / "opt" / "oss-cad-suite"))


def _which(name: str) -> str | None:
    env = os.environ.get(f"CHIPLAB_{name.upper()}")
    if env and Path(env).exists():
        return env
    p = shutil.which(name)
    if p:
        return p
    for d in (Path.home() / ".local" / "bin", OSS_CAD / "bin"):   # user-local, then OSS CAD Suite
        if (d / name).exists():
            return str(d / name)
    return None


def tool_status() -> dict[str, Any]:
    tools = {k: _which(k) for k in ("verilator", "yosys", "openroad", "klayout", "iverilog", "sby",
                                    "docker")}
    tt = os.environ.get("CHIPLAB_TT_TOOLS")           # path to a tt-support-tools checkout
    pdk = os.environ.get("PDK_ROOT")
    return {
        "tools": {k: {"installed": v is not None, "path": v} for k, v in tools.items()},
        "tt_support_tools": {"configured": bool(tt and Path(tt).exists()), "path": tt},
        "pdk_root": {"configured": bool(pdk and Path(pdk).exists()), "path": pdk},
        "install_hint": ("Tiny Tapeout's own flow is the reference: create a repository from the "
                         "official IHP Verilog template, copy the exported project in, and its GitHub "
                         "Actions run LibreLane hardening, precheck and GDS. Locally: OSS CAD Suite "
                         "(Yosys, iverilog, sby) plus tt-support-tools with the IHP PDK. ChipLab "
                         "will not download these without your OK."),
    }


def register_bits() -> dict[str, Any]:
    """Flip-flop count from the RTL register declarations (pre-synthesis; synthesis may prune
    unused configuration bits). Not an area number."""
    core = {"instruction memory (32x16)": 512, "lane registers (4x8x16)": 512,
            "FIFOs (4 lanes x (2+2) x 16 + counts)": 4 * 4 * 16 + 16,
            "pins/irq/enable/err/sync": 14 + 8 + 8 + 4 + 4 + 26}
    lane = {"pc, delay, divider": 5 + 5 + 16, "x, y, osr, isr": 64, "shift counts": 10,
            "crc": 32, "line coder state": 6 + 2 + 6}
    host = 3 + 3 + 3 + 5 + 24 + 16 + 1 + 1 + 1 + 7 + 16
    total = sum(core.values()) + 4 * sum(lane.values()) + host
    return {"total": total, "core": core, "per_lane": lane, "lanes": isa.N_LANES, "spi_host": host,
            "provenance": "derived from RTL declarations", "is_area": False}


def lint() -> dict[str, Any]:
    exe = _which("verilator")
    if not exe:
        return {"step": "lint", "ran": False, "reason": "verilator not installed"}
    r = subprocess.run([exe, "--lint-only", "-Wall", "-Wno-DECLFILENAME", "-Wno-UNUSEDSIGNAL",
                        "--top-module", TOP, *SOURCES], cwd=RTL, capture_output=True, text=True,
                       timeout=120)
    warnings = [ln for ln in r.stderr.splitlines() if ln.startswith("%Warning") or ln.startswith("%Error")]
    return {"step": "lint", "ran": True, "ok": r.returncode == 0 and not warnings,
            "warnings": warnings[:50], "tool": "verilator --lint-only -Wall"}


def synthesize(liberty: str | None = None, workdir: Path | None = None) -> dict[str, Any]:
    """Yosys synthesis. Without a liberty file this is a generic (technology-independent) cell
    count; with the IHP standard-cell liberty it maps to real cells and reports their area."""
    exe = _which("yosys")
    if not exe:
        return {"step": "synthesis", "ran": False, "reason": "yosys not installed"}
    wd = Path(workdir or Path.home() / ".cache" / "chiplab_asic" / "synth")
    wd.mkdir(parents=True, exist_ok=True)
    for f in SOURCES:
        shutil.copy(RTL / f, wd / f)
    script = [f"read_verilog {' '.join(SOURCES)}", f"synth -top {TOP} -flatten"]
    if liberty:
        script += [f"dfflibmap -liberty {liberty}", f"abc -liberty {liberty}", f"stat -liberty {liberty}"]
    else:
        script += ["stat"]
    (wd / "synth.ys").write_text("\n".join(script) + "\n")
    r = subprocess.run([exe, "-q", "-l", "synth.log", "-s", "synth.ys"], cwd=wd, capture_output=True,
                       text=True, timeout=1800)
    log = (wd / "synth.log").read_text(errors="replace") if (wd / "synth.log").exists() else r.stdout
    cells = re.findall(r"Number of cells:\s+(\d+)", log)
    area = re.findall(r"Chip area for (?:top )?module.*?:\s+([\d.]+)", log)
    return {"step": "synthesis", "ran": True, "ok": r.returncode == 0,
            "cells": int(cells[-1]) if cells else None,
            "area_um2": float(area[-1]) if area else None,
            "mapped": bool(liberty), "log_tail": log[-2000:]}


def flow_status() -> dict[str, Any]:
    """Every flow step with whether it can run here and its last real result, if any."""
    ts = tool_status()
    have = {k: v["installed"] for k, v in ts["tools"].items()}
    have["tt_precheck"] = ts["tt_support_tools"]["configured"]
    steps = []
    from chiplab.asic import report                   # lazy: report imports flow
    for key, tool, desc in FLOW_STEPS:
        last = report.load(key) if key in ("synthesis", "synthesis_ihp", "formal", "gate_sim") else None
        status = "ready" if have.get(tool) else "blocked: tool not installed"
        if last:
            status = ("passed" if last.get("ok") else "failed") + (" (stale: RTL changed)" if last.get("stale") else "")
        steps.append({"step": key, "tool": tool, "description": desc, "available": bool(have.get(tool)),
                      "result": {k: last.get(k) for k in ("cells", "flops", "cell_area_um2", "logic_delay_ns", "status", "mode", "depth", "passed",
                                                         "seeds", "protocols", "timestamp")} if last else None,
                      "status": status})
    _apply_hardening(steps, report.load("hardening"))
    return {"steps": steps, "tools": ts}


def _apply_hardening(steps: list[dict[str, Any]], h: dict[str, Any] | None) -> None:
    """Steps the TT GitHub Actions run performed get that run's verdict (CI ran them, not this machine)."""
    if not h or not h.get("ok"):
        return
    tag = f" (CI run {h['run_id']}" + (", stale)" if h.get("stale") else ")")
    setup = h.get("setup_violation_corners") or []
    verdict = {
        "floorplan": h.get("routed"), "placement": h.get("routed"), "cts": h.get("routed"),
        "routing": h.get("routed"), "drc": h.get("drc_clear"),
        "sta": None if not h.get("routed") else not setup and not h.get("hold_violation_corners"),
        "gds": h.get("jobs", {}).get("gds") == "success",
        "power": True if h.get("power_w") is not None else None,
        "tt_precheck": {"success": True, "failure": False}.get(h.get("jobs", {}).get("precheck", "")),
    }
    for s in steps:
        if s["step"] not in verdict:
            continue
        v = verdict[s["step"]]
        if v is None and s["step"] == "tt_precheck":
            s["status"] = f"not run{tag}"
        elif v is not None:
            note = f": setup fails at {', '.join(setup)}" if s["step"] == "sta" and setup else ""
            s["status"] = ("passed" if v else "failed") + note + tag
        s["result"] = {"run_id": h["run_id"], "url": h.get("url"), "timestamp": h.get("timestamp")}


# ------------------------------------------------------------------ Tiny Tapeout export
def _info_yaml() -> str:
    pins = {
        "ui": ["host SCK", "host CS_n", "host MOSI", "in 8", "in 9", "in 10", "in 11", "in 12"],
        "uo": ["host MISO", "host IRQ", "out 8", "out 9", "out 10", "out 11", "out 12", "out 13"],
        "uio": [f"pin {k}" for k in range(8)],
    }
    lines = [
        "# Tiny Tapeout project information (generated by ChipLab; fill in the TODO fields).",
        "# Check field names and the accepted 'tiles' values against the official IHP template",
        "# before submitting; ChipLab did not verify them.",
        "project:",
        '  title: "PEM: programmable protocol emulation machine"',
        '  author: "TODO"',
        '  discord: ""',
        '  description: "Four-lane reprogrammable pin processor with hardware NRZI, bit stuffing and CRC"',
        '  language: "Verilog"',
        "  clock_hz: 50000000   # TODO: confirm against timing results",
        f'  tiles: "{COMPETITION["max_tiles"]}"   # competition maximum; shrink after synthesis',
        f'  top_module: "{TOP}"',
        "  source_files:",
        *[f'    - "{f}"' for f in SOURCES],
        "pinout:",
    ]
    for grp, names in pins.items():
        for k, n in enumerate(names):
            lines.append(f'  {grp}[{k}]: "{n}"')
    lines.append('yaml_version: 6')
    return "\n".join(lines) + "\n"


def export_tt_project(dest: str | Path) -> dict[str, Any]:
    """Write a Tiny Tapeout project tree. Overwrites only files it generates."""
    d = Path(dest)
    (d / "src").mkdir(parents=True, exist_ok=True)
    (d / "docs").mkdir(exist_ok=True)
    (d / "programs").mkdir(exist_ok=True)
    (d / "test").mkdir(exist_ok=True)
    written = []
    for f in SOURCES:
        shutil.copy(RTL / f, d / "src" / f)
        written.append(f"src/{f}")
    (d / "info.yaml").write_text(_info_yaml())
    written.append("info.yaml")
    from chiplab.asic import report
    (d / "docs" / "info.md").write_text(report.architecture_md())
    (d / "docs" / "isa.md").write_text(isa.ISA_DOC)
    (d / "docs" / "verification.md").write_text(report.verification_md())
    written += ["docs/info.md", "docs/isa.md", "docs/verification.md"]
    progs = {
        "uart_tx.pem": protocols.uart_tx()[0], "uart_rx.pem": protocols.uart_rx()[0],
        **{f"spi_mode{m}.pem": protocols.spi_controller(m)[0] for m in range(4)},
        "i2c_controller.pem": protocols.i2c_controller()[0], "usb_ls_tx.pem": protocols.usb_ls_tx()[0],
    }
    for name, src in progs.items():
        assembler.assemble(src)                       # every shipped program must assemble
        (d / "programs" / name).write_text(src.strip() + "\n")
        written.append(f"programs/{name}")
    (d / "test" / "test.py").write_text(_cocotb_test())
    (d / "test" / "tb.v").write_text(_TB_V)
    (d / "test" / "README.md").write_text(
        "cocotb test of the Tiny Tapeout top level through the SPI host port (template layout:\n"
        "test.py, tb.v; set PROJECT_SOURCES in the template's Makefile to the src files).\n"
        "ChipLab replays the same SPI writes in Icarus Verilog before export (eda.icarus_tt_test);\n"
        "the template's GitHub Actions run this cocotb test on RTL and on the gate-level netlist.\n")
    stale = d / "test" / "test_pem.py"                # older ChipLab exports used this name
    if stale.exists():
        stale.unlink()
    written += ["test/test.py", "test/tb.v", "test/README.md"]
    (d / "chiplab_export.json").write_text(json.dumps({"top": TOP, "files": written,
                                                        "competition": COMPETITION}, indent=2))
    return {"dest": str(d), "files": written}


def tt_test_writes(byte: int = 0xA5) -> list[tuple[int, int]]:
    """Host-port writes that load UART TX on lane 0 (pin 0, clkdiv 2) and push one byte. Built
    with the same load_script the regression uses, so the exported test cannot drift from it."""
    from chiplab.asic import verify
    src, regs = protocols.uart_tx(pin=0)
    p = assembler.assemble(src)
    lines = verify.load_script([(0, p, {"clkdiv": 2, **regs})], 1).split()
    w = [(int(lines[i + 1], 16), int(lines[i + 2], 16)) for i in range(0, len(lines), 3) if lines[i] == "W"]
    return w + [(isa.A_FIFO, byte)]


TT_UART_BIT_CYCLES = 16          # 8 instruction slots per bit x clkdiv 2


def _cocotb_test() -> str:
    writes = ",\n    ".join(f"(0x{a:02x}, 0x{d:04x})" for a, d in tt_test_writes())
    return f'''# SPDX-License-Identifier: Apache-2.0
"""cocotb test for tt_um_chiplab_pem (generated by ChipLab).

Loads the UART TX program over the SPI host port (mode 0: ui[0] SCK, ui[1] CS_n, ui[2] MOSI,
uo[0] MISO; frame = {{rw, addr[6:0]}} then 16 data bits), pushes 0xA5 into lane 0's TX FIFO and
decodes the 8N1 frame on uio[0]. Runs on RTL and, in the template's gl_test job, on the
hardened gate-level netlist."""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

WRITES = [
    {writes},
]
BIT = {TT_UART_BIT_CYCLES}


async def spi_write(dut, addr, data):
    bits = [(addr >> (7 - k)) & 1 for k in range(8)] + [(data >> (15 - k)) & 1 for k in range(16)]
    dut.ui_in.value = 0b000                      # CS_n low, SCK low
    await ClockCycles(dut.clk, 4)
    for b in bits:
        dut.ui_in.value = b << 2                 # MOSI with SCK low
        await ClockCycles(dut.clk, 4)
        dut.ui_in.value = (b << 2) | 1           # SCK rising edge samples MOSI
        await ClockCycles(dut.clk, 4)
    dut.ui_in.value = 0b000
    await ClockCycles(dut.clk, 4)
    dut.ui_in.value = 0b010                      # CS_n high
    await ClockCycles(dut.clk, 8)


async def uart_receive(dut):
    """Wait for the start bit on uio[0] (driven), then sample 8 data bits mid-bit, LSB first."""
    for _ in range(20000):
        await ClockCycles(dut.clk, 1)
        if int(dut.uio_oe.value) & 1 and not int(dut.uio_out.value) & 1:
            break
    else:
        return None
    await ClockCycles(dut.clk, BIT // 2)
    assert not int(dut.uio_out.value) & 1, "start bit too short"
    byte = 0
    for k in range(8):
        await ClockCycles(dut.clk, BIT)
        byte |= (int(dut.uio_out.value) & 1) << k
    await ClockCycles(dut.clk, BIT)
    assert int(dut.uio_out.value) & 1, "stop bit"
    return byte


@cocotb.test()
async def uart_tx_over_spi(dut):
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    dut.ena.value = 1
    dut.ui_in.value = 0b010
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    for addr, data in WRITES[:-1]:              # program, lane registers, entry point, enable
        await spi_write(dut, addr, data)
    rx = cocotb.start_soon(uart_receive(dut))   # line is idle high now; listen before the push
    await spi_write(dut, *WRITES[-1])           # push 0xA5 into lane 0's TX FIFO
    byte = await rx
    assert byte is not None, "no start bit on uio[0]"
    assert byte == 0xA5, hex(byte)
'''


_TB_V = """`default_nettype none
`timescale 1ns / 1ps
module tb ();
  initial begin
    $dumpfile("tb.fst");
    $dumpvars(0, tb);
    #1;
  end
  reg clk;
  reg rst_n;
  reg ena;
  reg [7:0] ui_in;
  reg [7:0] uio_in;
  wire [7:0] uo_out;
  wire [7:0] uio_out;
  wire [7:0] uio_oe;
  tt_um_chiplab_pem user_project (
      .ui_in  (ui_in),
      .uo_out (uo_out),
      .uio_in (uio_in),
      .uio_out(uio_out),
      .uio_oe (uio_oe),
      .ena    (ena),
      .clk    (clk),
      .rst_n  (rst_n)
  );
endmodule
"""
