"""Open-source EDA runs on the PEM RTL: Yosys synthesis, SymbiYosys formal checks, and a
gate-level lockstep regression (the Yosys netlist simulated by Verilator against the Python model).

All tools come from fixed argument lists (no shell). Results are what the tools printed; nothing is
estimated here. Without the IHP liberty, synthesis is technology-independent (generic gates), so
cell counts are not IHP area.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from chiplab.asic import flow, iss, verify

WORK = verify.BUILD_ROOT / "eda"


def _need(tool: str) -> str:
    exe = flow._which(tool)
    if not exe:
        raise verify.ToolMissing(f"{tool} not installed (OSS CAD Suite provides it)")
    return exe


def _copy(dest: Path, files: list[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy(flow.RTL / f, dest / f)


# ------------------------------------------------------------------ synthesis
def synthesize_generic() -> dict[str, Any]:
    """Yosys `synth` of the full TT top (flattened, generic gates): cell mix and flop count."""
    exe = _need("yosys")
    wd = WORK / "synth"
    _copy(wd, flow.SOURCES)
    script = [f"read_verilog {' '.join(flow.SOURCES)}", f"synth -top {flow.TOP} -flatten",
              "tee -o stat.txt stat", "check -assert"]
    (wd / "synth.ys").write_text("\n".join(script) + "\n")
    t0 = time.time()
    r = subprocess.run([exe, "-q", "-l", "synth.log", "-s", "synth.ys"], cwd=wd, capture_output=True,
                       text=True, timeout=1800)
    stat = (wd / "stat.txt").read_text(errors="replace") if (wd / "stat.txt").exists() else ""
    cells: dict[str, int] = {}
    for n, name in re.findall(r"^\s+(\d+)\s+(?:[\d.]+\s+)?(\$_\w+_)\s*$", stat, re.M):
        cells[name] = int(n)
    if not cells:                                    # older stat layout: "$_AND_   123"
        for name, n in re.findall(r"^\s+(\$_\w+_)\s+(\d+)\s*$", stat, re.M):
            cells[name] = int(n)
    flops = sum(v for k, v in cells.items() if "DFF" in k or "SDFF" in k or "DLATCH" in k)
    total = re.findall(r"(\d+)\s+cells\s*$", stat, re.M) or re.findall(r"Number of cells:\s+(\d+)", stat)
    return {"step": "synthesis", "ran": True, "ok": r.returncode == 0, "tool": "yosys " + _version(exe),
            "mapping": "generic gates (no IHP liberty)", "cells": int(total[-1]) if total else sum(cells.values()),
            "flops": flops, "cell_mix": dict(sorted(cells.items(), key=lambda kv: -kv[1])),
            "seconds": round(time.time() - t0, 1), "error": None if r.returncode == 0 else (r.stderr or r.stdout)[-1500:],
            "note": "cell counts are technology independent; IHP area needs the sg13g2 liberty"}


def _version(exe: str) -> str:
    try:
        return subprocess.run([exe, "-V"], capture_output=True, text=True, timeout=30).stdout.split()[1]
    except Exception:  # noqa: BLE001
        return "?"


# ------------------------------------------------------------------ formal
FORMAL_WRAPPER = """`default_nettype none
module pem_formal (input wire clk, input wire rst_n, input wire bus_we, input wire bus_re,
                   input wire [6:0] bus_addr, input wire [15:0] bus_wdata, input wire [12:0] raw_in);
    wire [15:0] rd; wire [13:0] ov; wire [7:0] oe, irq;
    pem_core dut (.clk(clk), .rst_n(rst_n), .bus_we(bus_we), .bus_re(bus_re), .bus_addr(bus_addr),
                  .bus_wdata(bus_wdata), .bus_rdata(rd), .raw_in(raw_in), .out_val(ov), .oe(oe), .irq(irq));
    reg init = 1'b1;
    always @(posedge clk) init <= 1'b0;
    always @(*) if (init) assume (!rst_n);          // start from reset; every other input is free
endmodule
"""


def formal_check(depth: int = 24, mode: str = "bmc") -> dict[str, Any]:
    """SymbiYosys on the core's assertions A1-A4 with every bus/pin input unconstrained.
    bmc: no assertion can fail within `depth` cycles of reset. prove: k-induction (unbounded)."""
    if mode not in ("bmc", "prove"):
        raise ValueError("mode bmc or prove")
    if not 1 <= depth <= 60:
        raise ValueError("depth 1..60")
    exe = _need("sby")
    _need("yosys")
    wd = WORK / f"formal_{mode}"
    if wd.exists():
        shutil.rmtree(wd)
    _copy(wd, verify.RTL_FILES)
    (wd / "pem_formal.v").write_text(FORMAL_WRAPPER)
    files = verify.RTL_FILES + ["pem_formal.v"]
    (wd / "pem.sby").write_text(
        f"[options]\nmode {mode}\ndepth {depth}\nmulticlock off\n\n[engines]\nsmtbmc yices\n\n[script]\n"
        f"read_verilog -formal -DPEM_ASSERT {' '.join(files)}\nprep -top pem_formal\n\n[files]\n"
        + "\n".join(files) + "\n")
    t0 = time.time()
    r = subprocess.run([exe, "-f", "pem.sby"], cwd=wd, capture_output=True, text=True, timeout=3600)
    out = r.stdout + r.stderr
    status = (re.findall(r"DONE \((\w+),", out) or ["ERROR"])[-1]
    cex = re.findall(r"Assert failed in .*", out)
    return {"step": "formal", "ran": True, "mode": mode, "depth": depth, "engine": "smtbmc yices",
            "status": status, "ok": status == "PASS", "assertions": [a["id"] for a in verify.ASSERTIONS],
            "failed_assertions": cex[:4], "seconds": round(time.time() - t0, 1),
            "meaning": (f"no reachable state within {depth} cycles of reset violates A1-A4 for any inputs"
                        if mode == "bmc" else "A1-A4 hold in every reachable state (k-induction)")
            if status == "PASS" else "see log", "log_tail": out[-1500:]}


# ------------------------------------------------------------------ gate-level simulation
def _netlist() -> Path:
    exe = _need("yosys")
    wd = WORK / f"gl_{verify.rtl_hash()}"
    net = wd / "pem_core_gl.v"
    if net.exists():
        return net
    _copy(wd, verify.RTL_FILES)
    (wd / "gl.ys").write_text(f"read_verilog {' '.join(verify.RTL_FILES)}\nsynth -top pem_core -flatten\n"
                              "opt_clean -purge\nwrite_verilog -noattr pem_core_gl.v\n")
    r = subprocess.run([exe, "-q", "-l", "gl.log", "-s", "gl.ys"], cwd=wd, capture_output=True, text=True,
                       timeout=1800)
    if r.returncode != 0 or not net.exists():
        raise RuntimeError("yosys netlist failed: " + (r.stderr or r.stdout)[-1500:])
    return net


def _build_gl() -> Path:
    net = _netlist()
    out = net.parent / "obj"
    binary = out / "Vpem_core"
    if binary.exists():
        return binary
    out.mkdir(exist_ok=True)
    shutil.copy(flow.RTL / "tb_core.cpp", net.parent / "tb_core.cpp")
    cmd = [verify.verilator(), "--cc", "--exe", "--build", "-O2", "-Wno-fatal", "-Wno-lint", "-Wno-style",
           "--top-module", "pem_core", "--Mdir", str(out), str(net), str(net.parent / "tb_core.cpp")]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not binary.exists():
        raise RuntimeError("verilator build of the gate netlist failed:\n" + (r.stderr or r.stdout)[-2000:])
    return binary


def run_gate(script: str) -> list[str]:
    r = subprocess.run([str(_build_gl())], input=script, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"gate-level bench failed: {(r.stderr or r.stdout)[-1500:]}")
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def gate_level_regression(seeds: int = 10, cycles: int = 3000) -> dict[str, Any]:
    """Run the constrained-random scripts on the synthesized netlist and on the reference model;
    every host read and pin event must match. Also replays the directed protocol scripts."""
    binary = _build_gl()
    t0 = time.time()
    results = []
    for s in range(seeds):
        sc = verify.random_script(s, cycles)
        ref = iss.run_script(sc, iss.PEM())
        r = subprocess.run([str(binary)], input=sc, capture_output=True, text=True, timeout=600)
        gl = [ln for ln in r.stdout.splitlines() if ln.strip()]
        first = next(({"index": k, "model": a, "gate": b} for k, (a, b) in
                      enumerate(zip(ref + ["<end>"] * len(gl), gl + ["<end>"] * len(ref))) if a != b), None)
        results.append({"seed": s, "match": first is None and r.returncode == 0, "events": len(ref),
                        "first_divergence": first})
    from chiplab.asic import protocols
    directed = protocols.run_all("gate")
    passed = sum(r["match"] for r in results)
    return {"step": "gate_sim", "ran": True, "ok": passed == seeds and directed["passed"] == directed["total"],
            "passed": passed, "seeds": seeds,
            "protocols": {"passed": directed["passed"], "total": directed["total"],
                          "failed": [r["name"] for r in directed["results"] if not r["passed"]]},
            "cycles_per_seed": cycles, "netlist": "yosys generic gates, flattened pem_core",
            "results": results, "seconds": round(time.time() - t0, 1),
            "fidelity": "gate-level, zero delay (no SDF timing)"}


# ------------------------------------------------------------------ IHP sg13g2 mapping
IHP_PDK = Path(__import__("os").environ.get("CHIPLAB_IHP_PDK", Path.home() / ".local" / "opt" / "ihp-sg13g2"))
IHP_LIBS = {"typ": "sg13g2_stdcell_typ_1p20V_25C.lib", "slow": "sg13g2_stdcell_slow_1p08V_125C.lib"}
# Official Tiny Tapeout IHP tile outlines (um), tt-support-tools tech/ihp-sg13g2/tile_sizes.yaml
# at commit 7d2d2ef5261e (2026-11-23).
TT_IHP_TILES = {"1x1": (202.08, 154.98), "1x2": (202.08, 313.74), "2x2": (419.52, 313.74),
                "3x2": (636.96, 313.74), "4x2": (854.40, 313.74), "3x4": (636.96, 710.64),
                "4x4": (854.40, 710.64), "5x4": (1071.84, 710.64), "6x2": (1289.28, 313.74),
                "6x4": (1289.28, 710.64), "8x2": (1724.16, 313.74), "8x4": (1724.16, 710.64)}
TILE_SOURCE = "TinyTapeout/tt-support-tools tech/ihp-sg13g2/tile_sizes.yaml @ 7d2d2ef5261e"


def ihp_liberty(corner: str = "slow") -> Path | None:
    p = IHP_PDK / "sg13g2_stdcell" / "lib" / IHP_LIBS[corner]
    return p if p.exists() else None


def synthesize_ihp(corner: str = "slow", clock_mhz: float = 50.0) -> dict[str, Any]:
    """Map the TT top onto IHP sg13g2 standard cells with Yosys + ABC (timing-driven at the
    target period). Reports liberty cell area and ABC's longest register-to-register logic delay.
    Not placement, routing or sign-off STA."""
    if corner not in IHP_LIBS:
        raise ValueError("corner typ or slow")
    if not 1 <= clock_mhz <= 500:
        raise ValueError("clock_mhz 1..500")
    exe = _need("yosys")
    lib = ihp_liberty(corner)
    if not lib:
        raise verify.ToolMissing(f"IHP liberty not found under {IHP_PDK} (sg13g2_stdcell/lib/{IHP_LIBS[corner]})")
    wd = WORK / f"ihp_{corner}"
    _copy(wd, flow.SOURCES)
    period_ps = 1e6 / clock_mhz
    abc = f"+strash;dch,-f;map,-D,{period_ps:.0f};buffer;upsize,-D,{period_ps:.0f};dnsize,-D,{period_ps:.0f};stime,-p"
    (wd / "ihp.ys").write_text(
        f"read_liberty -lib {lib}\nread_verilog {' '.join(flow.SOURCES)}\nsynth -top {flow.TOP} -flatten\n"
        f"dfflibmap -liberty {lib}\nabc -liberty {lib} -script {abc}\nopt_clean -purge\n"
        f"tee -o stat.txt stat -liberty {lib}\nwrite_verilog -noattr {flow.TOP}_ihp.v\n")
    t0 = time.time()
    r = subprocess.run([exe, "-l", "ihp.log", "-s", "ihp.ys"], cwd=wd, capture_output=True, text=True, timeout=3600)
    log = (wd / "ihp.log").read_text(errors="replace") if (wd / "ihp.log").exists() else ""
    stat = (wd / "stat.txt").read_text(errors="replace") if (wd / "stat.txt").exists() else ""
    area = re.findall(r"Chip area for module .*?:\s+([\d.]+)", stat)
    seq = re.findall(r"used for sequential elements:\s+([\d.]+)", stat)
    delays = [float(x) for x in re.findall(r"Delay =\s+([\d.]+) ps", log)]
    cells = {m[1]: int(m[0]) for m in re.findall(r"^\s+(\d+)\s+[\d.E+]+\s+(sg13g2_\w+)\s*$", stat, re.M)}
    a = float(area[-1]) if area else None
    tiles = {}
    if a:
        for name, (w, h) in TT_IHP_TILES.items():
            dens = a / (w * h)
            tiles[name] = {"area_um2": round(w * h), "cell_density": round(dens, 3),
                           "verdict": "too small" if dens >= 1 else "unlikely" if dens > 0.7
                           else "uncertain" if dens > 0.5 else "likely"}
    delay = max(delays) if delays else None
    return {"step": "synthesis_ihp", "ran": True, "ok": r.returncode == 0 and a is not None,
            "corner": corner, "liberty": lib.name, "tool": "yosys " + _version(exe) + " + abc",
            "cell_area_um2": round(a, 1) if a else None, "sequential_area_um2": round(float(seq[-1]), 1) if seq else None,
            "cells": sum(cells.values()), "flops": sum(v for k, v in cells.items() if "df" in k),
            "cell_mix": dict(sorted(cells.items(), key=lambda kv: -kv[1])),
            "clock_mhz": clock_mhz, "logic_delay_ns": round(delay / 1000, 3) if delay else None,
            "logic_slack_ns": round(period_ps / 1000 - delay / 1000, 3) if delay else None,
            "tiles": tiles, "tile_source": TILE_SOURCE, "seconds": round(time.time() - t0, 1),
            "pdk_source": (IHP_PDK / "SOURCE").read_text().strip() if (IHP_PDK / "SOURCE").exists() else None,
            "fidelity": ("liberty cell area and ABC unit-wire logic delay (no placement, wires, clock tree, "
                         "clk-to-q or setup); tile verdicts are a density heuristic, not a placement result"),
            "error": None if r.returncode == 0 else (r.stderr or r.stdout)[-1500:]}


# ------------------------------------------------------------------ Icarus replay of the TT test
def icarus_tt_test(netlist: Path | None = None) -> dict[str, Any]:
    """Replay the exported cocotb test's SPI writes in a plain Verilog bench under Icarus Verilog
    (a third simulator, independent of Verilator and the Python model) and decode the UART byte."""
    exe, vvp = _need("iverilog"), _need("vvp")
    wd = WORK / "icarus_tt"
    _copy(wd, flow.SOURCES)
    writes = flow.tt_test_writes(0xA5)
    seq = "\n".join(f"    spi_write(8'h{a:02x}, 16'h{d:04x});" for a, d in writes[:-1])
    a_last, d_last = writes[-1]
    bit = flow.TT_UART_BIT_CYCLES
    (wd / "tb_icarus.v").write_text(f"""`timescale 1ns/1ps
module tb_icarus;
  reg clk = 0, rst_n = 0, ena = 1; reg [7:0] ui_in = 8'b010, uio_in = 0;
  wire [7:0] uo_out, uio_out, uio_oe;
  integer k, n; reg [7:0] byte_;
  tt_um_chiplab_pem dut (.ui_in(ui_in), .uo_out(uo_out), .uio_in(uio_in), .uio_out(uio_out),
                         .uio_oe(uio_oe), .ena(ena), .clk(clk), .rst_n(rst_n));
  always #10 clk = ~clk;
  task cyc(input integer c); repeat (c) @(posedge clk); endtask
  task spi_write(input [7:0] a, input [15:0] d);
    reg [23:0] f; begin
      f = {{a, d}}; ui_in = 8'b000; cyc(4);
      for (k = 23; k >= 0; k = k - 1) begin ui_in = {{5'b0, f[k], 2'b00}}; cyc(4); ui_in = {{5'b0, f[k], 2'b01}}; cyc(4); end
      ui_in = 8'b000; cyc(4); ui_in = 8'b010; cyc(8);
    end
  endtask
  reg listen = 0; integer j;
  initial begin                                   // receiver: armed before the push
    wait (listen);
    n = 0;
    while (!(uio_oe[0] && !uio_out[0]) && n < 20000) begin cyc(1); n = n + 1; end
    if (n >= 20000) begin $display("RESULT FAIL no-start-bit"); $finish; end
    cyc({bit // 2}); byte_ = 0;
    for (j = 0; j < 8; j = j + 1) begin cyc({bit}); byte_[j] = uio_out[0]; end
    cyc({bit});
    if (byte_ == 8'hA5 && uio_out[0]) $display("RESULT PASS %h", byte_);
    else $display("RESULT FAIL %h stop=%b", byte_, uio_out[0]);
    $finish;
  end
  initial begin
    cyc(10); rst_n = 1; cyc(5);
{seq}
    listen = 1;
    spi_write(8'h{a_last:02x}, 16'h{d_last:04x});
    cyc(100000); $display("RESULT FAIL timeout"); $finish;
  end
endmodule
""")
    srcs = [str(netlist)] if netlist else flow.SOURCES
    t0 = time.time()
    c = subprocess.run([exe, "-g2012", "-o", "tb.vvp", "tb_icarus.v", *srcs], cwd=wd, capture_output=True,
                       text=True, timeout=600)
    if c.returncode != 0:
        return {"step": "icarus_tt", "ran": True, "ok": False, "error": (c.stderr or c.stdout)[-1500:]}
    r = subprocess.run([vvp, "-n", "tb.vvp"], cwd=wd, capture_output=True, text=True, timeout=600)
    res = re.findall(r"RESULT (PASS|FAIL)(.*)", r.stdout)
    ok = bool(res) and res[-1][0] == "PASS"
    return {"step": "icarus_tt", "ran": True, "ok": ok, "detail": (res[-1][1].strip() if res else r.stdout[-300:]),
            "writes": len(writes), "seconds": round(time.time() - t0, 1),
            "simulator": "Icarus Verilog", "what": "exported cocotb test's SPI writes replayed on the TT top"}
