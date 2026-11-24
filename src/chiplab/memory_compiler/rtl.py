"""RTL and verification tests for compiled memories.

Generates synthesizable Verilog for the logical memory (mask ROM as a case table, SRAM /
register file / eDRAM / DRAM behavioural arrays with their ports, a synchronous FIFO, and a
refresh controller with a retention model for dynamic memories), plus a self-checking
testbench whose expected values come from an independent Python model of the same spec.
Runs with Verilator (--binary --timing) when it is installed.

The RTL describes the logical behaviour (what the memory returns); it is not the
transistor-level circuit and not the physical array.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

from chiplab.memory_compiler.compiler import MemSpec

BUILD = Path(os.environ.get("CHIPLAB_MEM_BUILD", Path.home() / ".cache" / "chiplab_memc"))


def _hex(v: int, w: int) -> str:
    return f"{w}'h{v:x}"


def gen_rtl(spec: MemSpec, retention_cycles: int = 0, refresh_interval: int = 0) -> str:
    n, w, a = spec.name, spec.width, spec.addr_bits
    if spec.kind == "mask_rom":
        data = list(spec.contents or []) + [0] * (spec.words - len(spec.contents or []))
        cases = "\n".join(f"            {_hex(i, a)}: data <= {_hex(v, w)};" for i, v in enumerate(data) if v)
        return f"""// Mask ROM {n}: {spec.words} x {w}, registered output. Logical contents only;
// the physical array (NOR/NAND, programmed sites) is in the compiler's physical view.
module {n} (input wire clk, input wire [{a - 1}:0] addr, output reg [{w - 1}:0] data);
    always @(posedge clk) begin
        case (addr)
{cases}
            default: data <= {_hex(0, w)};
        endcase
    end
endmodule
"""
    if spec.kind == "fifo":
        return f"""// Synchronous FIFO {n}: depth {spec.words}, width {w}, show-ahead read.
module {n} (input wire clk, input wire rst_n, input wire push, input wire [{w - 1}:0] din,
            input wire pop, output wire [{w - 1}:0] dout, output wire full, output wire empty,
            output reg [{a}:0] count);
    reg [{w - 1}:0] mem [0:{spec.words - 1}];
    reg [{a - 1}:0] wp, rp;
    wire do_push = push && !full;
    wire do_pop  = pop && !empty;
    assign full  = count == {spec.words};
    assign empty = count == 0;
    assign dout  = mem[rp];
    always @(posedge clk) begin
        if (!rst_n) begin
            wp <= 0; rp <= 0; count <= 0;
        end else begin
            if (do_push) begin
                mem[wp] <= din;
                wp <= (wp == {spec.words - 1}) ? 0 : wp + 1;
            end
            if (do_pop) rp <= (rp == {spec.words - 1}) ? 0 : rp + 1;
            count <= count + (do_push ? 1 : 0) - (do_pop ? 1 : 0);
        end
    end
endmodule
"""
    # array memories: R read ports, W write ports (rw ports modelled as one R + one W sharing addr)
    R = spec.read_ports + spec.rw_ports
    W = spec.write_ports + spec.rw_ports
    ports = ["input wire clk", "input wire rst_n"]
    for i in range(W):
        ports += [f"input wire we{i}", f"input wire [{a - 1}:0] waddr{i}", f"input wire [{w - 1}:0] wdata{i}"]
    for j in range(R):
        ports += [f"input wire re{j}", f"input wire [{a - 1}:0] raddr{j}", f"output reg [{w - 1}:0] rdata{j}"]
    dyn = spec.kind in ("dram", "edram") and retention_cycles > 0
    rows = spec.words
    if dyn:
        ports += ["input wire refresh_en", "output wire busy", "output reg retention_error"]
    writes = "\n".join(f"            if (we{i}) begin mem[waddr{i}] <= wdata{i};{' age[waddr' + str(i) + '] <= 0;' if dyn else ''} end"
                       for i in range(W))
    reads = "\n".join(
        f"            if (re{j}) begin rdata{j} <= mem[raddr{j}];"
        + (f" if (age[raddr{j}] >= RET) retention_error <= 1'b1;" if dyn else "") + " end"
        for j in range(R))
    dyn_decl = dyn_logic = ""
    if dyn:
        dyn_decl = f"""
    // Retention model (simulation of the dynamic cell, per word): age counts cycles since the
    // last write or refresh; a read of a word older than RET flags a retention error.
    localparam integer RET = {retention_cycles};
    localparam integer REF_INT = {refresh_interval};
    integer age [0:{rows - 1}];
    reg [{a - 1}:0] ref_row;
    integer ref_cnt;
    assign busy = refresh_en && ref_cnt == 0;
    integer k;"""
        dyn_logic = f"""
            for (k = 0; k < {rows}; k = k + 1) if (age[k] < 32'h7fffffff) age[k] <= age[k] + 1;
            if (refresh_en) begin
                ref_cnt <= (ref_cnt == 0) ? REF_INT - 1 : ref_cnt - 1;
                if (ref_cnt == 0) begin
                    age[ref_row] <= 0;
                    ref_row <= (ref_row == {rows - 1}) ? 0 : ref_row + 1;
                end
            end"""
        reset_dyn = f"""
            retention_error <= 1'b0; ref_row <= 0; ref_cnt <= REF_INT - 1;
            for (k = 0; k < {rows}; k = k + 1) age[k] <= 0;"""
    else:
        reset_dyn = ""
    return f"""// {spec.kind.upper()} {n}: {spec.words} x {w}, {R} read / {W} write port(s), synchronous.
// Behavioural (logical) model{' with a refresh controller and retention model' if dyn else ''}.
// Read-during-write to the same word returns the old data.
module {n} ({', '.join(ports)});
    reg [{w - 1}:0] mem [0:{spec.words - 1}];{dyn_decl}
    always @(posedge clk) begin
        if (!rst_n) begin{reset_dyn}
        end else begin
{reads}
{writes}{dyn_logic}
        end
    end
endmodule
"""


# ------------------------------------------------------------------ golden model + testbench
def _ops(spec: MemSpec, n_ops: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if spec.kind == "mask_rom":
        return [{"addr": rng.randrange(spec.words)} for _ in range(n_ops)]
    if spec.kind == "fifo":
        return [{"push": rng.random() < 0.55, "pop": rng.random() < 0.45, "din": rng.randrange(1 << spec.width)}
                for _ in range(n_ops)]
    R, W = spec.read_ports + spec.rw_ports, spec.write_ports + spec.rw_ports
    hot = [rng.randrange(spec.words) for _ in range(8)]
    pick = lambda: rng.choice(hot) if rng.random() < 0.5 else rng.randrange(spec.words)  # noqa: E731
    return [{"we": [rng.random() < 0.5 for _ in range(W)], "wa": [pick() for _ in range(W)],
             "wd": [rng.randrange(1 << spec.width) for _ in range(W)],
             "re": [rng.random() < 0.7 for _ in range(R)], "ra": [pick() for _ in range(R)]}
            for _ in range(n_ops)]


def golden(spec: MemSpec, ops: list[dict[str, Any]]) -> list[list[int | None]]:
    """Independent Python model: expected outputs after each op (None = don't care)."""
    out: list[list[int | None]] = []
    if spec.kind == "mask_rom":
        data = list(spec.contents or []) + [0] * spec.words
        return [[data[o["addr"]]] for o in ops]
    if spec.kind == "fifo":
        q: list[int] = []
        for o in ops:
            full, empty = len(q) == spec.words, not q
            head = q[0] if q else None
            if o["pop"] and not empty:
                q.pop(0)
            if o["push"] and not full:
                q.append(o["din"])
            out.append([head, int(full), int(empty)])          # values seen before this edge
        return out
    mem: dict[int, int] = {}
    for o in ops:
        row = [mem.get(a) if r else None for r, a in zip(o["re"], o["ra"])]
        for we, a, d in zip(o["we"], o["wa"], o["wd"]):        # later port wins
            if we:
                mem[a] = d
        out.append(row)
    return out


def gen_testbench(spec: MemSpec, ops: list[dict[str, Any]], exp: list[list[int | None]],
                  dyn: bool = False, refresh: bool = True) -> str:
    n, w, a = spec.name, spec.width, spec.addr_bits
    lines = [f"module tb; reg clk = 0; reg rst_n = 0; integer errors = 0; integer checks = 0;"]
    if spec.kind == "mask_rom":
        lines += [f"reg [{a - 1}:0] addr = 0; wire [{w - 1}:0] data;", f"{n} dut(.clk(clk), .addr(addr), .data(data));"]
        body = []
        for i, (o, e) in enumerate(zip(ops, exp)):
            body.append(f"addr = {o['addr']}; @(posedge clk); #1; checks = checks + 1; "
                        f"if (data !== {_hex(e[0], w)}) begin errors = errors + 1; $display(\"MISMATCH op {i} addr {o['addr']}: %h\", data); end")
    elif spec.kind == "fifo":
        lines += [f"reg push = 0, pop = 0; reg [{w - 1}:0] din = 0; wire [{w - 1}:0] dout; wire full, empty; wire [{a}:0] count;",
                  f"{n} dut(.clk(clk), .rst_n(rst_n), .push(push), .din(din), .pop(pop), .dout(dout), .full(full), .empty(empty), .count(count));"]
        body = []
        for i, (o, e) in enumerate(zip(ops, exp)):
            chk = f"checks = checks + 1; if (full !== {e[1]} || empty !== {e[2]}"
            chk += f" || dout !== {_hex(e[0], w)}" if e[0] is not None else ""
            chk += f") begin errors = errors + 1; $display(\"MISMATCH op {i}\"); end"
            body.append(f"push = {int(o['push'])}; pop = {int(o['pop'])}; din = {_hex(o['din'], w)}; #1; {chk} @(posedge clk); #1;")
    else:
        R, W = spec.read_ports + spec.rw_ports, spec.write_ports + spec.rw_ports
        decl, conn = [], [".clk(clk)", ".rst_n(rst_n)"]
        for i in range(W):
            decl.append(f"reg we{i} = 0; reg [{a - 1}:0] waddr{i} = 0; reg [{w - 1}:0] wdata{i} = 0;")
            conn += [f".we{i}(we{i})", f".waddr{i}(waddr{i})", f".wdata{i}(wdata{i})"]
        for j in range(R):
            decl.append(f"reg re{j} = 0; reg [{a - 1}:0] raddr{j} = 0; wire [{w - 1}:0] rdata{j};")
            conn += [f".re{j}(re{j})", f".raddr{j}(raddr{j})", f".rdata{j}(rdata{j})"]
        if dyn:
            decl.append(f"reg refresh_en = {int(refresh)}; wire busy; wire retention_error;")
            conn += [".refresh_en(refresh_en)", ".busy(busy)", ".retention_error(retention_error)"]
        lines += decl + [f"{n} dut({', '.join(conn)});"]
        body = []
        for i, (o, e) in enumerate(zip(ops, exp)):
            s = "".join(f"we{k} = {int(o['we'][k])}; waddr{k} = {o['wa'][k]}; wdata{k} = {_hex(o['wd'][k], w)}; " for k in range(W))
            s += "".join(f"re{k} = {int(o['re'][k])}; raddr{k} = {o['ra'][k]}; " for k in range(R))
            if dyn:
                s = "while (busy) begin " + "".join(f"we{k} = 0; " for k in range(W)) + "".join(
                    f"re{k} = 0; " for k in range(R)) + "@(posedge clk); #1; end " + s
            s += "@(posedge clk); #1; "
            for k in range(R):
                if e[k] is not None:
                    s += (f"checks = checks + 1; if (rdata{k} !== {_hex(e[k], w)}) begin errors = errors + 1; "
                          f"$display(\"MISMATCH op {i} port {k}: %h\", rdata{k}); end ")
            body.append(s)
    lines += ["always #5 clk = ~clk;", "initial begin", "  repeat (2) @(posedge clk); rst_n = 1; #1;"]
    lines += ["  " + b for b in body]
    if dyn:
        lines.append('  $display("RETENTION_ERROR %0d", retention_error);')
    lines += ['  $display("CHECKS %0d ERRORS %0d", checks, errors);', "  $finish;", "end", "endmodule"]
    return "\n".join(lines) + "\n"


def verilator() -> str | None:
    p = shutil.which("verilator") or str(Path.home() / ".local" / "bin" / "verilator")
    return p if Path(p).exists() else None


def run_tests(spec: MemSpec, n_ops: int = 300, seed: int = 1, retention_cycles: int = 0,
              refresh_interval: int = 0, refresh: bool = True) -> dict[str, Any]:
    """Generate RTL + testbench, simulate with Verilator, compare with the Python model."""
    exe = verilator()
    rtl = gen_rtl(spec, retention_cycles, refresh_interval)
    ops = _ops(spec, n_ops, seed)
    dyn = spec.kind in ("dram", "edram") and retention_cycles > 0
    tb = gen_testbench(spec, ops, golden(spec, ops), dyn, refresh)
    if not exe:
        return {"ran": False, "reason": "verilator not installed", "rtl": rtl}
    wd = BUILD / f"{spec.name}_{abs(hash((rtl, tb))) % 10**12}"
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "mem.v").write_text(rtl)
    (wd / "tb.sv").write_text(tb)
    b = subprocess.run([exe, "--binary", "--timing", "-Wno-fatal", "-Wno-lint", "-Wno-style", "--top-module",
                        "tb", "-Mdir", "obj", "mem.v", "tb.sv"], cwd=wd, capture_output=True, text=True, timeout=900)
    if b.returncode != 0:
        return {"ran": False, "reason": "build failed", "log": (b.stderr or b.stdout)[-2000:], "rtl": rtl}
    r = subprocess.run([str(wd / "obj" / "Vtb")], capture_output=True, text=True, timeout=600)
    out = r.stdout
    m = [ln for ln in out.splitlines() if ln.startswith("CHECKS")]
    checks, errors = (int(m[-1].split()[1]), int(m[-1].split()[3])) if m else (0, -1)
    ret = [ln for ln in out.splitlines() if ln.startswith("RETENTION_ERROR")]
    return {"ran": True, "checks": checks, "errors": errors, "passed": errors == 0 and checks > 0,
            "retention_error": int(ret[-1].split()[1]) if ret else None,
            "mismatches": [ln for ln in out.splitlines() if ln.startswith("MISMATCH")][:10],
            "fidelity": "RTL simulation (Verilator) vs independent Python model", "rtl": rtl,
            "testbench_ops": len(ops)}
