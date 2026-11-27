"""Design-for-test lab.

  march tests     MATS+, March X, March C- on a memory with injected faults (stuck-at,
                  transition, inversion/idempotent coupling, address decoder); coverage per
                  fault class. Expected coverage follows van de Goor, "Testing Semiconductor
                  Memories" (1991): March C- detects all unlinked SAF, TF, AF, CFin, CFid.
  MBIST RTL       March C- controller + memory in Verilog, checked in Icarus with and without
                  an injected stuck-at bit
  scan            area cost of replacing every flop with a scan flop (IHP liberty areas) and
                  test time for a chain length
  observability   state bits vs output bits: cycles to dump all state
  timing          longest combinational path (Yosys ltp) as a logic-depth proxy
  yield           Poisson, Murphy and negative-binomial yield for a die area and defect density
  fault injection single-event upsets in the SRAM near-memory bank's golden model
"""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ---- memory fault simulation ---------------------------------------------------------------
FAULT_CLASSES = ("SAF", "TF", "AF", "CFin", "CFid")


@dataclass
class Fault:
    kind: str
    cell: int
    value: int = 0            # SAF value / TF direction (1 = cannot rise) / CFid forced value
    aggressor: int = -1       # coupling aggressor, or AF alias address
    trigger: int = 1          # coupling trigger transition: 1 = rising, 0 = falling


class FaultyMemory:
    """Bit-oriented memory (n cells) with one fault."""

    def __init__(self, n: int, fault: Fault | None):
        self.n, self.f = n, fault
        self.cells = [0] * n

    def _addr(self, a: int) -> int:
        f = self.f
        if f and f.kind == "AF" and a == f.aggressor:
            return f.cell                     # decoder maps aggressor onto victim
        return a

    def write(self, a: int, v: int) -> None:
        a = self._addr(a)
        f = self.f
        old = self.cells[a]
        if f and f.kind == "TF" and a == f.cell:
            if f.value == 1 and old == 0 and v == 1:
                return                         # cannot rise
            if f.value == 0 and old == 1 and v == 0:
                return                         # cannot fall
        self.cells[a] = v
        if f and f.kind in ("CFin", "CFid") and a == f.aggressor and old != v and \
                v == f.trigger:
            if f.kind == "CFin":
                self.cells[f.cell] ^= 1
            else:
                self.cells[f.cell] = f.value
        if f and f.kind == "SAF":
            self.cells[f.cell] = f.value

    def read(self, a: int) -> int:
        a = self._addr(a)
        if self.f and self.f.kind == "SAF" and a == self.f.cell:
            return self.f.value
        return self.cells[a]


# March elements: (order, [ops]) with order "up", "down" or "any"; ops "w0", "w1", "r0", "r1".
MARCH: dict[str, list[tuple[str, list[str]]]] = {
    "MATS+": [("any", ["w0"]), ("up", ["r0", "w1"]), ("down", ["r1", "w0"])],
    "March X": [("any", ["w0"]), ("up", ["r0", "w1"]), ("down", ["r1", "w0"]),
                ("any", ["r0"])],
    "March C-": [("any", ["w0"]), ("up", ["r0", "w1"]), ("up", ["r1", "w0"]),
                 ("down", ["r0", "w1"]), ("down", ["r1", "w0"]), ("any", ["r0"])],
}


def run_march(name: str, mem: FaultyMemory) -> bool:
    """True if the test detects a fault (some read mismatches)."""
    for order, ops in MARCH[name]:
        addrs = range(mem.n) if order in ("up", "any") else range(mem.n - 1, -1, -1)
        for a in addrs:
            for op in ops:
                v = int(op[1])
                if op[0] == "w":
                    mem.write(a, v)
                elif mem.read(a) != v:
                    return True
    return False


def march_length(name: str, n: int) -> int:
    return n * sum(len(ops) for _, ops in MARCH[name])


def _faults(n: int, rng: np.random.Generator, per_class: int) -> list[Fault]:
    out: list[Fault] = []
    for _ in range(per_class):
        v = int(rng.integers(n))
        a = int(rng.integers(n - 1))
        a = a if a < v else a + 1
        out += [Fault("SAF", v, int(rng.integers(2))), Fault("TF", v, int(rng.integers(2))),
                Fault("AF", v, aggressor=a),
                Fault("CFin", v, aggressor=a, trigger=int(rng.integers(2))),
                Fault("CFid", v, int(rng.integers(2)), aggressor=a,
                      trigger=int(rng.integers(2)))]
    return out


def coverage(name: str, n: int = 64, per_class: int = 100, seed: int = 0) -> dict[str, Any]:
    """Fault coverage of a march test per fault class, by exhaustive injection of random
    single faults."""
    if name not in MARCH:
        raise ValueError(f"test must be one of {list(MARCH)}")
    rng = np.random.default_rng(seed)
    hit: dict[str, list[int]] = {k: [0, 0] for k in FAULT_CLASSES}
    for f in _faults(n, rng, per_class):
        det = run_march(name, FaultyMemory(n, f))
        hit[f.kind][0] += det
        hit[f.kind][1] += 1
    return {"test": name, "cells": n, "ops": march_length(name, n),
            "complexity": f"{march_length(name, 1)}n",
            "coverage": {k: d / t for k, (d, t) in hit.items()}}


# ---- MBIST RTL ---------------------------------------------------------------------------
def mbist_rtl(words: int, width: int) -> str:
    """March C- BIST controller with an internal word-oriented memory (solid 0/1 backgrounds).
    `define FAULT_ADDR / FAULT_BIT / FAULT_VAL inject a stuck-at bit for testing."""
    aw = max(1, (words - 1).bit_length())
    return f"""// ChipLab March C- MBIST, {words} x {width}
module mbist(input clk, input rst_n, input go, output reg done, output reg fail);
  reg [{width - 1}:0] mem [0:{words - 1}];
  reg [2:0] elem;  reg ph;  reg [{aw}:0] i;
  wire [{aw - 1}:0] addr = (elem >= 3'd3 && elem <= 3'd4) ? {words - 1} - i[{aw - 1}:0]
                                                        : i[{aw - 1}:0];
  wire [{width - 1}:0] raw = mem[addr];
`ifdef FAULT_ADDR
  wire [{width - 1}:0] rd = (addr == `FAULT_ADDR) ?
      (`FAULT_VAL ? raw | ({{{{{width - 1}{{1'b0}}}}, 1'b1}} << `FAULT_BIT)
                  : raw & ~({{{{{width - 1}{{1'b0}}}}, 1'b1}} << `FAULT_BIT)) : raw;
`else
  wire [{width - 1}:0] rd = raw;
`endif
  // element: 0 w0 | 1 up(r0,w1) | 2 up(r1,w0) | 3 down(r0,w1) | 4 down(r1,w0) | 5 r0
  wire expect1 = (elem == 3'd2) || (elem == 3'd4);
  wire wval = (elem == 3'd1) || (elem == 3'd3);
  reg running;
  always @(posedge clk) begin
    if (!rst_n) begin
      elem <= 0; ph <= 0; i <= 0; done <= 0; fail <= 0; running <= 0;
    end else if (go && !running && !done) begin
      running <= 1;
    end else if (running) begin
      if (elem == 3'd0) begin
        mem[addr] <= {{{width}{{1'b0}}}};
        if (i == {words - 1}) begin i <= 0; elem <= 1; end else i <= i + 1;
      end else if (elem == 3'd5) begin
        if (rd != {{{width}{{1'b0}}}}) fail <= 1;
        if (i == {words - 1}) begin running <= 0; done <= 1; end else i <= i + 1;
      end else if (!ph) begin
        if (rd != {{{width}{{expect1}}}}) fail <= 1;
        ph <= 1;
      end else begin
        mem[addr] <= {{{width}{{wval}}}};
        ph <= 0;
        if (i == {words - 1}) begin i <= 0; elem <= elem + 1; end else i <= i + 1;
      end
    end
  end
endmodule
"""


def mbist_check(words: int = 16, width: int = 8, fault: tuple[int, int, int] | None = None
                ) -> dict[str, Any]:
    """Simulate the MBIST in Icarus; fault = (addr, bit, stuck_value)."""
    from chiplab.pim import _eda
    if not (2 <= words <= 1024 and 1 <= width <= 64):
        raise ValueError("words 2-1024, width 1-64")
    defs = "" if fault is None else (f"`define FAULT_ADDR {fault[0]}\n`define FAULT_BIT "
                                     f"{fault[1]}\n`define FAULT_VAL {fault[2]}\n")
    tb = f"""{defs}`timescale 1ns/1ps
module tb; reg clk=0, rst_n=0, go=0; wire done, fail;
  mbist dut(.clk(clk), .rst_n(rst_n), .go(go), .done(done), .fail(fail));
  always #5 clk=~clk;
  integer cyc = 0; always @(posedge clk) cyc = cyc + 1;
  initial begin #22 rst_n=1; go=1; wait(done); #1 $display("R %0d %0d", fail, cyc); $finish; end
endmodule
"""
    out = _eda.iverilog_run({"tb.v": tb, "m.v": mbist_rtl(words, width)})
    m = re.search(r"R (\d) (\d+)", out)
    if not m:
        raise RuntimeError("MBIST simulation produced no result: " + out[-300:])
    return {"words": words, "width": width, "fault": fault, "fail_flag": m.group(1) == "1",
            "cycles": int(m.group(2)), "expected_cycles_approx": 10 * words}


# ---- scan, observability, timing --------------------------------------------------------
def _liberty_area(cell: str) -> float:
    from chiplab.imc_lab.digital import LIBERTY
    text = LIBERTY.read_text(errors="replace")
    m = re.search(r"cell \(\"?" + re.escape(cell) + r"\"?\)\s*\{\s*area\s*:\s*([\d.]+)", text)
    if not m:
        raise ValueError(f"{cell} not in liberty")
    return float(m.group(1))


def scan_overhead(n_flops: int, base_area_um2: float, chains: int = 1,
                  shift_hz: float = 10e6, patterns: int = 500) -> dict[str, Any]:
    """Area and test time to replace every flop with a scan flop (IHP sg13g2 liberty)."""
    if n_flops < 1 or chains < 1:
        raise ValueError("n_flops and chains must be >= 1")
    dff, sdff = _liberty_area("sg13g2_dfrbp_1"), _liberty_area("sg13g2_sdfrbp_1")
    extra = n_flops * (sdff - dff)
    chain_len = math.ceil(n_flops / chains)
    return {"flops": n_flops, "dff_um2": dff, "scan_dff_um2": sdff,
            "added_area_um2": extra, "area_overhead": extra / base_area_um2,
            "chain_length": chain_len,
            "test_time_s": patterns * (chain_len + 1) / shift_hz,
            "note": "routing of scan_en/scan chains not included"}


def observability(state_bits: int, output_bits: int, input_bits: int) -> dict[str, Any]:
    if min(state_bits, output_bits, input_bits) < 1:
        raise ValueError("bit counts must be >= 1")
    return {"state_bits": state_bits, "cycles_to_dump_state": math.ceil(state_bits /
                                                                        output_bits),
            "cycles_to_load_state": math.ceil(state_bits / input_bits),
            "observability_ratio": output_bits / state_bits,
            "advice": "add scan or a debug read port" if state_bits / output_bits > 64 else
            "state is reachable through the functional ports"}


def logic_depth(verilog: str, top: str) -> dict[str, Any]:
    """Longest combinational path in cells after generic synthesis (Yosys `ltp -noff`)."""
    from chiplab.pim import _eda
    with tempfile.TemporaryDirectory(prefix="chiplab-dft-") as tmp:
        d = Path(tmp)
        (d / "t.v").write_text(verilog)
        (d / "s.ys").write_text(f"read_verilog t.v\nsynth -top {top} -flatten\nmemory_map\n"
                                "opt\nabc -g AND,NAND,OR,NOR,XOR,XNOR,MUX\nopt_clean\n"
                                "tee -o ltp.txt ltp -noff\n")
        r = subprocess.run([_eda.tool("yosys"), "-q", "-s", "s.ys"], cwd=d, capture_output=True,
                           text=True, timeout=600)
        if r.returncode:
            raise RuntimeError("yosys failed: " + (r.stderr or r.stdout)[-600:])
        txt = (d / "ltp.txt").read_text()
    m = re.search(r"Longest topological path.*?\(length=(\d+)\)", txt)
    depth = int(m.group(1)) if m else None
    gate_ps = 60.0
    return {"logic_depth_gates": depth, "gate_delay_ps_assumed": gate_ps,
            "fmax_mhz_rough": 1e6 / (depth * gate_ps) if depth else None,
            "note": "depth in 2-input gates; real timing needs STA with the liberty "
                    "(the TT CI flow runs it)"}


# ---- yield ---------------------------------------------------------------------------------
def yield_models(die_area_mm2: float, defect_density_per_cm2: float = 0.1,
                 cluster_alpha: float = 3.0) -> dict[str, Any]:
    """Classic yield models. D0 default 0.1/cm2 is an assumed mature-node value."""
    if die_area_mm2 <= 0 or defect_density_per_cm2 < 0 or cluster_alpha <= 0:
        raise ValueError("positive area, non-negative D0, positive alpha")
    ad = die_area_mm2 / 100 * defect_density_per_cm2
    murphy = ((1 - math.exp(-ad)) / ad) ** 2 if ad > 0 else 1.0
    return {"area_x_d0": ad, "poisson": math.exp(-ad), "murphy": murphy,
            "negative_binomial": (1 + ad / cluster_alpha) ** -cluster_alpha,
            "assumptions": {"d0_per_cm2": defect_density_per_cm2, "alpha": cluster_alpha}}


# ---- fault injection into the SRAM NMC model ---------------------------------------------
def seu_injection(spec: dict[str, Any] | None = None, trials: int = 500, seed: int = 0,
                  detector: Callable[[list[int]], bool] | None = None) -> dict[str, Any]:
    """Flip one random stored bit before a random op; how often is the op result corrupted,
    and how often does a per-row parity bit (default detector) catch it."""
    from chiplab.pim import sram
    p = sram.validate(spec or {})
    rng = np.random.default_rng(seed)
    corrupt = caught = 0
    names = [n for n in sram.OPS if n not in ("ACCCLR",)]
    for _ in range(trials):
        clean = sram.NmcModel(p)
        clean.mem = [int.from_bytes(rng.bytes((p["width"] + 7) // 8), "little")
                     & ((1 << p["width"]) - 1) for _ in range(p["rows"])]
        parity = [bin(v).count("1") & 1 for v in clean.mem]
        bad = sram.NmcModel(p)
        bad.mem = list(clean.mem)
        r, bit = int(rng.integers(p["rows"])), int(rng.integers(p["width"]))
        bad.mem[r] ^= 1 << bit
        n = names[int(rng.integers(len(names)))]
        a, b, d = (int(x) for x in rng.integers(0, p["rows"], 3))
        if detector is None:
            det = any((bin(v).count("1") & 1) != parity[i] for i, v in enumerate(bad.mem))
        else:
            det = detector(bad.mem)
        clean.op(n, a, b, d)
        bad.op(n, a, b, d)
        wrong = clean.mem[d] != bad.mem[d] or clean.acc != bad.acc
        corrupt += wrong
        caught += wrong and det
    return {"trials": trials, "result_corrupted": corrupt / trials,
            "corruptions_detected": caught / max(corrupt, 1),
            "detector": "per-row parity" if detector is None else "custom"}
