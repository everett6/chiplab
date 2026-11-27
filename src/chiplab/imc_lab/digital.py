"""Bit-serial digital compute-in-memory macro: small enough for a Tiny Tapeout tile or an FPGA.

The array stores ROWS x COLS signed WB-bit weights in flip-flops (the "memory"). An MVM
y = W x with unsigned XB-bit inputs runs bit-serially: each cycle one input bit plane is broadcast
to every row, each row adds the weights whose input bit is 1 (an adder tree next to the storage),
and a shift-accumulator weights the plane by 2^b. ROWS dot products finish in XB cycles.

Everything here is checked against tools, not assumed:
- the Verilog is simulated with Icarus Verilog on random vectors and compared bit-exactly with
  ``golden()`` (NumPy integer maths);
- Yosys synthesises it to the IHP sg13g2 standard cells (the PDK the PEM ASIC hardened on) and
  reports cell count and cell area from the liberty file.
Not modelled here: timing closure, power, placement. Those come from a real P&R run or a bench
measurement, which the learning loop can take in.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from chiplab.asic import flow

LIMITS = {"rows": (1, 16), "cols": (1, 16), "wbits": (2, 8), "xbits": (1, 8)}
LIBERTY = Path(os.environ.get(
    "CHIPLAB_IHP_LIBERTY",
    Path.home() / ".local/opt/ihp-sg13g2/sg13g2_stdcell/lib/sg13g2_stdcell_typ_1p20V_25C.lib"))


class DigitalCimError(ValueError):
    pass


def validate(spec: dict[str, Any]) -> dict[str, int]:
    out = {}
    for k, (lo, hi) in LIMITS.items():
        v = spec.get(k, {"rows": 4, "cols": 4, "wbits": 4, "xbits": 4}[k])
        if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
            raise DigitalCimError(f"{k} must be an int in [{lo}, {hi}], got {v!r}")
        out[k] = v
    return out


def acc_bits(p: dict[str, int]) -> int:
    """Width that holds any result: |sum| <= cols * 2^(wb-1) * (2^xb - 1)."""
    worst = p["cols"] * (1 << (p["wbits"] - 1)) * ((1 << p["xbits"]) - 1)
    return int(worst).bit_length() + 1


def golden(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Reference MVM: y[r] = sum_c w[r, c] * x[c] (exact integers)."""
    return (w.astype(np.int64) @ x.astype(np.int64)).astype(np.int64)


def rtl(p: dict[str, int]) -> str:
    """Synthesisable Verilog-2005 for the macro (one module, parameters baked in)."""
    r, c, wb, xb, ab = p["rows"], p["cols"], p["wbits"], p["xbits"], acc_bits(p)
    ra = max(1, (r * c - 1).bit_length())
    oa = max(1, (r - 1).bit_length())
    xw = c * xb
    sb = (c * (1 << (wb - 1))).bit_length() + 1        # one bit plane's partial-sum width
    cb = max(1, xb.bit_length())
    return f"""// ChipLab IMC lab: bit-serial digital CIM, {r}x{c}, w{wb} signed, x{xb} unsigned
module cim_macro (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              w_we,            // write one weight
    input  wire [{ra - 1}:0]       w_addr,          // row*{c} + col
    input  wire [{wb - 1}:0]       w_data,
    input  wire              start,           // latch x and run {xb} bit planes
    input  wire [{xw - 1}:0]      x_in,            // x[col] at bits [col*{xb} +: {xb}]
    input  wire [{oa - 1}:0]       y_addr,
    output wire signed [{ab - 1}:0] y_out,
    output reg               done
);
    reg signed [{wb - 1}:0] w [0:{r * c - 1}];
    reg [{xw - 1}:0] x_q;
    reg [{cb - 1}:0] bitpos;
    reg busy;
    reg signed [{ab - 1}:0] acc [0:{r - 1}];
    integer i, j;

    // per-row partial sum of the weights selected by the current input bit plane
    reg signed [{sb - 1}:0] psum [0:{r - 1}];
    always @* begin
        for (i = 0; i < {r}; i = i + 1) begin
            psum[i] = 0;
            for (j = 0; j < {c}; j = j + 1)
                if (x_q[j*{xb} + bitpos]) psum[i] = psum[i] + w[i*{c} + j];
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            busy <= 1'b0; done <= 1'b0; bitpos <= 0;
            for (i = 0; i < {r}; i = i + 1) acc[i] <= 0;
        end else begin
            if (w_we && !busy) w[w_addr] <= w_data;
            if (start && !busy) begin
                x_q <= x_in; bitpos <= 0; busy <= 1'b1; done <= 1'b0;
                for (i = 0; i < {r}; i = i + 1) acc[i] <= 0;
            end else if (busy) begin
                for (i = 0; i < {r}; i = i + 1)
                    acc[i] <= acc[i] + (psum[i] <<< bitpos);
                if (bitpos == {xb - 1}) begin busy <= 1'b0; done <= 1'b1; end
                else bitpos <= bitpos + 1'b1;
            end
        end
    end
    assign y_out = acc[y_addr];
endmodule
"""


def testbench(p: dict[str, int], vectors: list[tuple[np.ndarray, np.ndarray]]) -> str:
    r, c, wb, xb = p["rows"], p["cols"], p["wbits"], p["xbits"]
    lines = []
    for w, x in vectors:
        for rr in range(r):
            for cc in range(c):
                lines.append(f"    write_w({rr * c + cc}, {int(w[rr, cc]) & ((1 << wb) - 1)});")
        xv = sum(int(x[cc]) << (cc * xb) for cc in range(c))
        lines.append(f"    run({c * xb}'d{xv});")
    body = "\n".join(lines)
    return f"""`timescale 1ns/1ps
module tb;
    reg clk = 0, rst_n = 0, w_we = 0, start = 0;
    reg [{max(1, (r * c - 1).bit_length()) - 1}:0] w_addr = 0;
    reg [{wb - 1}:0] w_data = 0;
    reg [{c * xb - 1}:0] x_in = 0;
    reg [{max(1, (r - 1).bit_length()) - 1}:0] y_addr = 0;
    wire signed [{acc_bits(p) - 1}:0] y_out;
    wire done;
    integer k, cycles;
    cim_macro dut(.clk(clk), .rst_n(rst_n), .w_we(w_we), .w_addr(w_addr), .w_data(w_data),
                  .start(start), .x_in(x_in), .y_addr(y_addr), .y_out(y_out), .done(done));
    always #5 clk = ~clk;
    task write_w(input integer a, input integer d);
        begin @(negedge clk); w_addr = a; w_data = d; w_we = 1; @(negedge clk); w_we = 0; end
    endtask
    task run(input [{c * xb - 1}:0] x);
        begin
            @(negedge clk); x_in = x; start = 1; @(negedge clk); start = 0; cycles = 1;
            while (!done) begin @(negedge clk); cycles = cycles + 1; end
            for (k = 0; k < {r}; k = k + 1) begin y_addr = k; #1 $display("Y %0d", y_out); end
            $display("C %0d", cycles);
        end
    endtask
    initial begin
        repeat (2) @(negedge clk); rst_n = 1;
{body}
        $finish;
    end
endmodule
"""


def _tool(name: str) -> str:
    exe = flow._which(name)
    if not exe:
        raise DigitalCimError(f"{name} not installed (OSS CAD Suite provides it)")
    return exe


def simulate(spec: dict[str, Any], n_vectors: int = 8, seed: int = 0) -> dict[str, Any]:
    """Run the generated RTL in Icarus Verilog on random vectors; compare with golden()."""
    p = validate(spec)
    if not 1 <= n_vectors <= 64:
        raise DigitalCimError("n_vectors must be in [1, 64]")
    rng = np.random.default_rng(seed)
    lo, hi = -(1 << (p["wbits"] - 1)), (1 << (p["wbits"] - 1)) - 1
    vectors = [(rng.integers(lo, hi + 1, (p["rows"], p["cols"])),
                rng.integers(0, 1 << p["xbits"], p["cols"])) for _ in range(n_vectors - 2)]
    # corner cases: most-negative weights with full-scale inputs, and all zeros
    vectors.append((np.full((p["rows"], p["cols"]), lo), np.full(p["cols"], (1 << p["xbits"]) - 1)))
    vectors.append((np.zeros((p["rows"], p["cols"]), int), np.zeros(p["cols"], int)))
    with tempfile.TemporaryDirectory(prefix="chiplab-imc-") as tmp:
        d = Path(tmp)
        (d / "cim.v").write_text(rtl(p))
        (d / "tb.v").write_text(testbench(p, vectors))
        comp = subprocess.run([_tool("iverilog"), "-g2005", "-o", "sim", "tb.v", "cim.v"], cwd=d,
                              capture_output=True, text=True, timeout=120)
        if comp.returncode:
            raise DigitalCimError("iverilog failed: " + comp.stderr[-800:])
        run = subprocess.run([_tool("vvp"), "-n", "sim"], cwd=d, capture_output=True, text=True,
                             timeout=300)
    ys = [int(m) for m in re.findall(r"^Y (-?\d+)", run.stdout, re.M)]
    cyc = [int(m) for m in re.findall(r"^C (\d+)", run.stdout, re.M)]
    expect = np.concatenate([golden(w, x) for w, x in vectors]).tolist()
    mismatches = sum(a != b for a, b in zip(ys, expect)) + abs(len(ys) - len(expect))
    return {"simulator": "icarus verilog", "vectors": len(vectors), "outputs_checked": len(expect),
            "mismatches": mismatches, "bit_exact": mismatches == 0 and len(ys) == len(expect),
            "cycles_per_mvm": max(cyc) if cyc else None, "log_tail": run.stdout[-400:]}


def synthesize(spec: dict[str, Any]) -> dict[str, Any]:
    """Yosys synthesis to IHP sg13g2 cells; cell count and liberty cell area (no P&R)."""
    p = validate(spec)
    if not LIBERTY.is_file():
        raise DigitalCimError(f"IHP sg13g2 liberty not found at {LIBERTY}")
    with tempfile.TemporaryDirectory(prefix="chiplab-imc-syn-") as tmp:
        d = Path(tmp)
        (d / "cim.v").write_text(rtl(p))
        script = (f"read_verilog cim.v\nsynth -top cim_macro -flatten\n"
                  f"dfflibmap -liberty {LIBERTY}\nabc -liberty {LIBERTY}\nopt_clean\n"
                  f"tee -o stat.txt stat -liberty {LIBERTY}\n")
        (d / "s.ys").write_text(script)
        r = subprocess.run([_tool("yosys"), "-q", "-s", "s.ys"], cwd=d, capture_output=True,
                           text=True, timeout=900)
        if r.returncode:
            raise DigitalCimError("yosys failed: " + (r.stderr or r.stdout)[-800:])
        stat = (d / "stat.txt").read_text()
    cells = (re.search(r"Number of cells:\s+(\d+)", stat)
             or re.search(r"^\s*(\d+)\s+\S+\s+cells\s*$", stat, re.M))
    area = re.search(r"Chip area for (?:top )?module.*?:\s*([\d.]+)", stat)
    flops = sum(int(n) for n in re.findall(r"^\s+(\d+)\s+\S+\s+sg13g2_(?:df|sdf|dl)\w*", stat, re.M))
    return {"tool": "yosys + abc", "liberty": LIBERTY.name,
            "cells": int(cells.group(1)) if cells else None,
            "flip_flops": flops or None,
            "cell_area_um2": float(area.group(1)) if area else None,
            "stat_tail": stat[-1200:]}


# Tiny Tapeout IHP tile: 202.08 x 154.98 um (TT IHP template); 8x4 tiles hardened the PEM ASIC.
TT_TILE_UM2 = 202.08 * 154.98


def predict(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Fast closed-form numbers (no tools), each tagged with how it was derived."""
    p = validate(spec)
    storage = p["rows"] * p["cols"] * p["wbits"]
    return {
        "cycles_per_mvm": {"value": p["xbits"] + 1, "unit": "cycles", "provenance": "rtl_structure",
                           "note": "1 load cycle + one cycle per input bit plane"},
        "macs_per_mvm": {"value": p["rows"] * p["cols"], "unit": "MAC", "provenance": "definition"},
        "weight_bits": {"value": storage, "unit": "bit", "provenance": "definition"},
        "flip_flops": {"value": storage + p["rows"] * acc_bits(p) + p["cols"] * p["xbits"]
                       + max(1, p["xbits"].bit_length()) + 2, "unit": "flop",
                       "provenance": "rtl_structure"},
    }
