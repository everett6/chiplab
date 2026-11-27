"""SRAM near-memory compute bank: a small register-file "SRAM" with a bank-local vector ALU.

Each row holds `lanes` packed unsigned lanes of `lane_bits`. One command reads rows a and b,
computes, and writes row d (or the accumulator) in a single cycle, without the data crossing
a bus. Operations:

  AND OR XOR NOT        bitwise on whole rows
  ADD SUB               lane-wise, wrapping
  POPCNT                popcount of row a -> row d
  EQ LTU                lane-wise compare -> all-ones/all-zeros lane mask
  MAC / ACCCLR          acc += sum_i a_i * b_i  (dot product of lanes) / clear acc
  SEARCH                parallel compare of row a against every row -> row-match bitmask
  COPY                  row a -> row d (RowClone analogue)

The RTL is checked bit-exact against `NmcModel` in Icarus Verilog and synthesised with Yosys to
IHP sg13g2 cells. On silicon this maps to flip-flops/latches (what a digital shuttle can
build), not to a foundry SRAM macro with bitline computing.

`compare_with_cpu` contrasts the near-memory command with a conventional load-load-op-store
sequence. It is an analytical model; its energy constants are labelled with their sources.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chiplab.pim import _eda

OPS = {"AND": 0, "OR": 1, "XOR": 2, "NOT": 3, "ADD": 4, "SUB": 5, "POPCNT": 6, "EQ": 7,
       "LTU": 8, "MAC": 9, "ACCCLR": 10, "SEARCH": 11, "COPY": 12}
LIMITS = {"rows": (2, 32), "lanes": (1, 8), "lane_bits": (2, 16)}


class SramNmcError(ValueError):
    pass


def validate(spec: dict[str, Any]) -> dict[str, int]:
    p = {"rows": 8, "lanes": 4, "lane_bits": 8} | {k: v for k, v in spec.items() if k in LIMITS}
    for k, (lo, hi) in LIMITS.items():
        if not isinstance(p[k], int) or not lo <= p[k] <= hi:
            raise SramNmcError(f"{k} must be an integer in [{lo}, {hi}]")
    p["width"] = p["lanes"] * p["lane_bits"]
    if p["rows"] > p["width"]:
        raise SramNmcError("SEARCH writes one bit per row, so rows must be <= lanes*lane_bits")
    p["aw"] = max(1, (p["rows"] - 1).bit_length())
    p["accw"] = 2 * p["lane_bits"] + p["lanes"].bit_length() + 8
    return p


class NmcModel:
    """Golden reference (pure Python ints)."""

    def __init__(self, p: dict[str, int]):
        self.p = p
        self.mem = [0] * p["rows"]
        self.acc = 0

    def _lanes(self, v: int) -> list[int]:
        b = self.p["lane_bits"]
        return [(v >> (i * b)) & ((1 << b) - 1) for i in range(self.p["lanes"])]

    def _pack(self, lanes: list[int]) -> int:
        b = self.p["lane_bits"]
        return sum((x & ((1 << b) - 1)) << (i * b) for i, x in enumerate(lanes))

    def op(self, name: str, a: int, b: int, d: int) -> None:
        p, va, vb = self.p, self.mem[a], self.mem[b]
        mask = (1 << p["width"]) - 1
        full = (1 << p["lane_bits"]) - 1
        la, lb = self._lanes(va), self._lanes(vb)
        res: int | None
        if name == "AND":
            res = va & vb
        elif name == "OR":
            res = va | vb
        elif name == "XOR":
            res = va ^ vb
        elif name == "NOT":
            res = ~va & mask
        elif name == "ADD":
            res = self._pack([x + y for x, y in zip(la, lb)])
        elif name == "SUB":
            res = self._pack([x - y for x, y in zip(la, lb)])
        elif name == "POPCNT":
            res = bin(va).count("1")
        elif name == "EQ":
            res = self._pack([full if x == y else 0 for x, y in zip(la, lb)])
        elif name == "LTU":
            res = self._pack([full if x < y else 0 for x, y in zip(la, lb)])
        elif name == "MAC":
            self.acc = (self.acc + sum(x * y for x, y in zip(la, lb))) % (1 << p["accw"])
            res = None
        elif name == "ACCCLR":
            self.acc, res = 0, None
        elif name == "SEARCH":
            res = sum(1 << r for r in range(p["rows"]) if self.mem[r] == va)
        elif name == "COPY":
            res = va
        else:
            raise SramNmcError(f"unknown op {name}")
        if res is not None:
            self.mem[d] = res & mask


def rtl(spec: dict[str, Any]) -> str:
    p = validate(spec)
    R, L, B, W, AW, ACCW = p["rows"], p["lanes"], p["lane_bits"], p["width"], p["aw"], p["accw"]
    return f"""// ChipLab SRAM near-memory compute bank: {R} rows x {L} lanes x {B} bits
module sram_nmc(
  input clk, input rst_n,
  input h_we, input [{AW - 1}:0] h_addr, input [{W - 1}:0] h_wdata, output [{W - 1}:0] h_rdata,
  input start, input [3:0] op, input [{AW - 1}:0] a, input [{AW - 1}:0] b, input [{AW - 1}:0] d,
  output reg done, output reg [{ACCW - 1}:0] acc);
  reg [{W - 1}:0] mem [0:{R - 1}];
  assign h_rdata = mem[h_addr];
  wire [{W - 1}:0] va = mem[a];
  wire [{W - 1}:0] vb = mem[b];
  reg [{W - 1}:0] res;
  reg wr;
  reg [{ACCW - 1}:0] dot;
  integer i;
  always @* begin
    res = {W}'d0; wr = 1'b1; dot = {ACCW}'d0;
    case (op)
      4'd0: res = va & vb;
      4'd1: res = va | vb;
      4'd2: res = va ^ vb;
      4'd3: res = ~va;
      4'd4: for (i = 0; i < {L}; i = i + 1) res[i*{B} +: {B}] = va[i*{B} +: {B}] + vb[i*{B} +: {B}];
      4'd5: for (i = 0; i < {L}; i = i + 1) res[i*{B} +: {B}] = va[i*{B} +: {B}] - vb[i*{B} +: {B}];
      4'd6: for (i = 0; i < {W}; i = i + 1) res = res + va[i];
      4'd7: for (i = 0; i < {L}; i = i + 1)
              res[i*{B} +: {B}] = (va[i*{B} +: {B}] == vb[i*{B} +: {B}]) ? {{{B}{{1'b1}}}} : {B}'d0;
      4'd8: for (i = 0; i < {L}; i = i + 1)
              res[i*{B} +: {B}] = (va[i*{B} +: {B}] < vb[i*{B} +: {B}]) ? {{{B}{{1'b1}}}} : {B}'d0;
      4'd9: begin wr = 1'b0;
              for (i = 0; i < {L}; i = i + 1)
                dot = dot + va[i*{B} +: {B}] * vb[i*{B} +: {B}];
            end
      4'd10: wr = 1'b0;
      4'd11: for (i = 0; i < {R}; i = i + 1) res[i] = (mem[i] == va);
      4'd12: res = va;
      default: wr = 1'b0;
    endcase
  end
  always @(posedge clk) begin
    if (!rst_n) begin
      done <= 1'b0; acc <= {ACCW}'d0;
    end else begin
      done <= start;
      if (start && wr) mem[d] <= res;
      if (start && op == 4'd9) acc <= acc + dot;
      if (start && op == 4'd10) acc <= {ACCW}'d0;
      if (h_we && !start) mem[h_addr] <= h_wdata;
    end
  end
endmodule
"""


def _program(p: dict[str, int], n_ops: int, seed: int) -> tuple[list[int], list[tuple]]:
    rng = np.random.default_rng(seed)
    nbytes = (p["width"] + 7) // 8
    init = [int.from_bytes(rng.bytes(nbytes), "little") & ((1 << p["width"]) - 1)
            for _ in range(p["rows"])]
    if p["rows"] >= 3:
        init[2] = init[0]                         # guarantee a SEARCH hit beyond itself
    names = list(OPS)
    prog = [(n, 0, 0, 0) for n in names]          # each op once on fixed rows first
    prog = [(n, 0, 1 % p["rows"], (2 + i) % p["rows"]) for i, (n, *_) in enumerate(prog)]
    for _ in range(n_ops):
        prog.append((names[int(rng.integers(len(names)))], *(int(x) for x in
                                                            rng.integers(0, p["rows"], 3))))
    return init, prog


def testbench(p: dict[str, int], init: list[int], prog: list[tuple]) -> str:
    AW, W = p["aw"], p["width"]
    lines = [f"  hw({r}, {W}'d{v});" for r, v in enumerate(init)]
    lines += [f"  cmd(4'd{OPS[n]}, {a}, {b}, {d});" for n, a, b, d in prog]
    body = "\n".join(lines)
    return f"""`timescale 1ns/1ps
module tb;
  reg clk = 0, rst_n = 0, h_we = 0, start = 0;
  reg [{AW - 1}:0] h_addr = 0, a = 0, b = 0, d = 0;
  reg [{W - 1}:0] h_wdata = 0;
  reg [3:0] op = 0;
  wire [{W - 1}:0] h_rdata;
  wire done;
  wire [{p['accw'] - 1}:0] acc;
  sram_nmc dut(.clk(clk), .rst_n(rst_n), .h_we(h_we), .h_addr(h_addr), .h_wdata(h_wdata),
               .h_rdata(h_rdata), .start(start), .op(op), .a(a), .b(b), .d(d), .done(done),
               .acc(acc));
  always #5 clk = ~clk;
  task hw(input [{AW - 1}:0] ad, input [{W - 1}:0] v); begin
    @(negedge clk); h_we = 1; h_addr = ad; h_wdata = v; @(negedge clk); h_we = 0; end endtask
  task cmd(input [3:0] o, input [{AW - 1}:0] ia, input [{AW - 1}:0] ib, input [{AW - 1}:0] id);
  begin
    @(negedge clk); op = o; a = ia; b = ib; d = id; start = 1;
    @(negedge clk); start = 0; h_addr = id; #1;
    $display("S %0d %0d %0d %0d", o, h_rdata, acc, done);
  end endtask
  initial begin
    @(negedge clk); @(negedge clk); rst_n = 1;
{body}
    $finish;
  end
endmodule
"""


def simulate(spec: dict[str, Any], n_ops: int = 40, seed: int = 0) -> dict[str, Any]:
    """Run a random program (every op at least once) in Icarus; compare each step with
    NmcModel: the destination row and the accumulator after every command."""
    p = validate(spec)
    if not 0 <= n_ops <= 400:
        raise SramNmcError("n_ops must be in [0, 400]")
    init, prog = _program(p, n_ops, seed)
    out = _eda.iverilog_run({"tb.v": testbench(p, init, prog), "nmc.v": rtl(p)})
    got = [tuple(int(x) for x in line.split()[1:]) for line in out.splitlines()
           if line.startswith("S ")]
    m = NmcModel(p)
    m.mem = list(init)
    expect = []
    for n, a, b, d in prog:
        m.op(n, a, b, d)
        expect.append((OPS[n], m.mem[d], m.acc, 1))
    bad = [i for i, (g, e) in enumerate(zip(got, expect)) if g != e]
    return {"simulator": "icarus verilog", "commands": len(prog), "checked": len(got),
            "mismatches": len(bad) + abs(len(got) - len(expect)),
            "bit_exact": not bad and len(got) == len(expect),
            "first_mismatch": None if not bad else {"step": bad[0], "op": prog[bad[0]][0],
                                                    "got": got[bad[0]],
                                                    "expect": expect[bad[0]]},
            "ops_covered": sorted({n for n, *_ in prog}), "cycles_per_command": 1}


def synthesize(spec: dict[str, Any]) -> dict[str, Any]:
    return _eda.yosys_synth(rtl(spec), "sram_nmc")


def predict(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    p = validate(spec)
    return {
        "storage_bits": {"value": p["rows"] * p["width"], "unit": "bit",
                         "provenance": "definition"},
        "flip_flops": {"value": p["rows"] * p["width"] + p["accw"] + 1, "unit": "count",
                       "provenance": "rtl_structure", "note": "array + accumulator + done"},
        "cycles_per_command": {"value": 1, "unit": "cycles", "provenance": "rtl_structure"},
    }


# ---- data-movement comparison -------------------------------------------------------------
# Energy constants (45 nm) from Horowitz, ISSCC 2014: 64-bit read of an 8 KB SRAM ~10 pJ,
# 8-bit add 0.03 pJ, 32-bit add 0.1 pJ, 8-bit multiply 0.2 pJ, 32-bit multiply 3.1 pJ.
# Wire and per-instruction overheads are assumptions, labelled as such.
ENERGY = {
    "sram_access_pj_per_64b": (10.0, "published: Horowitz ISSCC 2014, 8 KB SRAM, 45 nm"),
    "add_pj_per_8b": (0.03, "published: Horowitz ISSCC 2014, 45 nm"),
    "mul_pj_per_8b": (0.2, "published: Horowitz ISSCC 2014, 45 nm"),
    "wire_pj_per_bit_mm": (0.1, "assumed: on-chip global wire, order of magnitude"),
    "bus_distance_mm": (1.0, "assumed: memory bank to core distance"),
    "instr_overhead_pj": (10.0, "assumed: fetch/decode/regfile per instruction, small core"),
    "load_latency_cycles": (3.0, "assumed: L1-like hit latency"),
}


def _op_counts(p: dict[str, int], name: str) -> dict[str, float]:
    """Work per command: rows read/written, ALU energy units, CPU instructions."""
    L, R = p["lanes"], p["rows"]
    scale = p["lane_bits"] / 8
    alu = {"AND": L / 3, "OR": L / 3, "XOR": L / 3, "NOT": L / 3,
           "ADD": L * 1.0, "SUB": L * 1.0, "POPCNT": p["width"] / 8, "EQ": L, "LTU": L,
           "MAC": L * (1 + 0.2 / 0.03), "ACCCLR": 0.0, "SEARCH": R * L, "COPY": 0.0}[name]
    reads = {"NOT": 1, "POPCNT": 1, "COPY": 1, "ACCCLR": 0, "SEARCH": R}.get(name, 2)
    writes = 0 if name in ("MAC", "ACCCLR") else 1
    # CPU with a word-wide SIMD unit: one instruction per op, except MAC (lane multiplies +
    # adds) and SEARCH (one compare + branch per row).
    instr = {"MAC": 2 * L, "SEARCH": 2 * R, "ACCCLR": 1}.get(name, 1)
    return {"reads": reads, "writes": writes, "alu_add8_units": alu * scale, "instr": instr}


def compare_with_cpu(spec: dict[str, Any], op_mix: dict[str, int] | None = None,
                     overrides: dict[str, float] | None = None) -> dict[str, Any]:
    """Energy, bits moved over the bus and cycles: near-memory command vs a CPU doing
    load(a), load(b), compute, store(d). Analytical; both sides pay the array accesses."""
    p = validate(spec)
    e = {k: v for k, (v, _) in ENERGY.items()} | (overrides or {})
    op_mix = op_mix or {n: 1 for n in OPS}
    for n in op_mix:
        if n not in OPS:
            raise SramNmcError(f"unknown op {n}")
    W = p["width"]
    access = e["sram_access_pj_per_64b"] * W / 64
    wire = e["wire_pj_per_bit_mm"] * e["bus_distance_mm"]
    cmd_bits = 4 + 3 * p["aw"]
    rows, tot = [], {"cpu_pj": 0.0, "nmc_pj": 0.0, "cpu_bits": 0, "nmc_bits": 0,
                     "cpu_cycles": 0.0, "nmc_cycles": 0.0}
    for n, count in op_mix.items():
        c = _op_counts(p, n)
        alu = c["alu_add8_units"] * e["add_pj_per_8b"]
        array = (c["reads"] + c["writes"]) * access
        cpu_bits = (c["reads"] + c["writes"]) * W
        cpu_pj = array + alu + cpu_bits * wire + c["instr"] * e["instr_overhead_pj"]
        nmc_pj = array + alu + cmd_bits * wire
        cpu_cyc = c["reads"] * e["load_latency_cycles"] + c["instr"] + c["writes"]
        nmc_cyc = 1 + 1                                # issue + execute
        rows.append({"op": n, "count": count, "cpu_pj": round(cpu_pj, 3),
                     "nmc_pj": round(nmc_pj, 3), "energy_ratio": round(cpu_pj / nmc_pj, 2),
                     "cpu_bus_bits": cpu_bits, "nmc_bus_bits": cmd_bits,
                     "cpu_cycles": cpu_cyc, "nmc_cycles": nmc_cyc})
        tot["cpu_pj"] += count * cpu_pj
        tot["nmc_pj"] += count * nmc_pj
        tot["cpu_bits"] += count * cpu_bits
        tot["nmc_bits"] += count * cmd_bits
        tot["cpu_cycles"] += count * cpu_cyc
        tot["nmc_cycles"] += count * nmc_cyc
    return {
        "fidelity": "analytical", "per_op": rows,
        "total": {k: round(v, 3) for k, v in tot.items()} | {
            "energy_ratio_cpu_over_nmc": round(tot["cpu_pj"] / tot["nmc_pj"], 2),
            "bus_bits_ratio": round(tot["cpu_bits"] / max(tot["nmc_bits"], 1), 2),
            "speedup": round(tot["cpu_cycles"] / tot["nmc_cycles"], 2)},
        "constants": {k: {"value": e[k], "source": ENERGY[k][1]} for k in ENERGY},
        "caveats": [
            "Both sides pay the same array reads/writes; the saving is bus traffic, "
            "instruction overhead and load latency.",
            "Bitwise ALU energy is a nominal 1/3 of an 8-bit add per byte (assumed).",
            "45 nm constants; scale for other nodes before comparing with silicon.",
        ],
    }
