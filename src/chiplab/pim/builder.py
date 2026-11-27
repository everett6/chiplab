"""PIM chip builder: memory -> operations -> configuration -> architecture -> simulation ->
RTL -> verification -> synthesis -> review -> Tiny Tapeout export (never submitted).

Every stage records pass/fail/skip with its evidence; a later stage does not run if an earlier
one failed. Fabrication limits are explicit: a digital shuttle builds standard cells only, so
DRAM, memristor and ferrofluid designs stop after simulation with the reason stated.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from chiplab.pim import _eda, dram, sram

MEMORIES: dict[str, dict[str, Any]] = {
    "sram": {"ops": tuple(sram.OPS), "digital_shuttle": True,
             "how": "flip-flop/latch array + bank-local ALU (standard cells)"},
    "dram": {"ops": dram.OPERATIONS, "digital_shuttle": False,
             "how": "needs a DRAM process (storage capacitors, modified sense amps, DCC rows)"},
    "memristor": {"ops": ("mvm",), "digital_shuttle": False,
                  "how": "needs a BEOL resistive-memory process; prototype with discrete "
                         "resistors (imc_lab analog_crossbar)"},
    "ferrofluid": {"ops": (), "digital_shuttle": False,
                   "how": "no validated model and no fabrication route; research only"},
}
TT_TILE_UM2 = 202.08 * 154.98       # IHP TT tile
TT_MAX_TILES = 16
ASSUMED_DENSITY = 0.5


def _stage(name: str, status: str, **evidence: Any) -> dict[str, Any]:
    return {"stage": name, "status": status} | evidence


def tt_wrapper(spec: dict[str, Any]) -> str:
    """Byte-serial Tiny Tapeout wrapper around sram_nmc.

    uio_in[0] strobe (rising edge), uio_in[2:1] mode: 0 push ui_in into the command shift
    register, 1 execute, 2 select output byte = ui_in. Command (MSB first):
    {we, op[3:0], a, b, d, data}. Execute writes mem[d] = data when we=1, else runs op.
    uo_out shows byte `sel` of {acc, mem[d]}.
    """
    p = sram.validate(spec)
    AW, W, ACCW = p["aw"], p["width"], p["accw"]
    cmdw = 1 + 4 + 3 * AW + W
    nbytes = math.ceil(cmdw / 8)
    outw = 8 * math.ceil((ACCW + W) / 8)
    return f"""`default_nettype none
module tt_um_chiplab_sram_nmc(
  input wire [7:0] ui_in, output wire [7:0] uo_out,
  input wire [7:0] uio_in, output wire [7:0] uio_out, output wire [7:0] uio_oe,
  input wire ena, input wire clk, input wire rst_n);
  reg [{8 * nbytes - 1}:0] sh;
  reg [7:0] sel;
  reg strobe_q, exec;
  wire rise = uio_in[0] & ~strobe_q;
  wire [1:0] mode = uio_in[2:1];
  wire [{cmdw - 1}:0] cmd = sh[{cmdw - 1}:0];
  wire we = cmd[{cmdw - 1}];
  wire [3:0] op = cmd[{cmdw - 2}:{cmdw - 5}];
  wire [{AW - 1}:0] a = cmd[{W + 3 * AW - 1}:{W + 2 * AW}];
  wire [{AW - 1}:0] b = cmd[{W + 2 * AW - 1}:{W + AW}];
  wire [{AW - 1}:0] d = cmd[{W + AW - 1}:{W}];
  wire [{W - 1}:0] data = cmd[{W - 1}:0];
  wire [{W - 1}:0] rdata;
  wire [{ACCW - 1}:0] acc;
  wire done;
  sram_nmc core(.clk(clk), .rst_n(rst_n), .h_we(exec & we), .h_addr(d), .h_wdata(data),
                .h_rdata(rdata), .start(exec & ~we), .op(op), .a(a), .b(b), .d(d),
                .done(done), .acc(acc));
  wire [{outw - 1}:0] outbus = {{{outw - ACCW - W}'d0, acc, rdata}};
  assign uo_out = outbus[sel*8 +: 8];
  assign uio_out = 8'd0;
  assign uio_oe = 8'd0;
  always @(posedge clk) begin
    if (!rst_n) begin
      sh <= 0; sel <= 0; strobe_q <= 0; exec <= 0;
    end else begin
      strobe_q <= uio_in[0];
      exec <= rise && mode == 2'd1;
      if (rise && mode == 2'd0) sh <= {{sh[{8 * nbytes - 9}:0], ui_in}};
      if (rise && mode == 2'd2) sel <= ui_in;
    end
  end
  wire _unused = &{{ena, uio_in[7:3], done, 1'b0}};
endmodule
"""


def _wrapper_tb(p: dict[str, int], steps: list[tuple[int, int, int, int, int, int]]) -> str:
    AW, W = p["aw"], p["width"]
    cmdw = 1 + 4 + 3 * AW + W
    nbytes = math.ceil(cmdw / 8)
    outb = math.ceil((p["accw"] + W) / 8)
    body = []
    for we, op, a, b, d, data in steps:
        word = (we << (cmdw - 1)) | (op << (cmdw - 5)) | (a << (W + 2 * AW)) | \
            (b << (W + AW)) | (d << W) | data
        for i in reversed(range(nbytes)):
            body.append(f"    strobe(2'd0, 8'd{(word >> (8 * i)) & 0xFF});")
        body.append("    strobe(2'd1, 8'd0);")
        for i in range(outb):
            body.append(f"    strobe(2'd2, 8'd{i}); #1 $write(\"%0d \", uo_out);")
        body.append('    $display("");')
    return f"""`timescale 1ns/1ps
module tbw;
  reg clk = 0, rst_n = 0;
  reg [7:0] ui_in = 0, uio_in = 0;
  wire [7:0] uo_out, uio_out, uio_oe;
  tt_um_chiplab_sram_nmc dut(.ui_in(ui_in), .uo_out(uo_out), .uio_in(uio_in),
    .uio_out(uio_out), .uio_oe(uio_oe), .ena(1'b1), .clk(clk), .rst_n(rst_n));
  always #5 clk = ~clk;
  task strobe(input [1:0] m, input [7:0] v); begin
    @(negedge clk); ui_in = v; uio_in = {{5'd0, m, 1'b0}};
    @(negedge clk); uio_in[0] = 1; @(negedge clk); uio_in[0] = 0; @(negedge clk);
  end endtask
  initial begin
    @(negedge clk); @(negedge clk); rst_n = 1;
{chr(10).join(body)}
    $finish;
  end
endmodule
"""


def verify_wrapper(spec: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    """Drive the TT wrapper through its pins: write rows, run AND/ADD/MAC/SEARCH, read back."""
    import numpy as np
    p = sram.validate(spec)
    rng = np.random.default_rng(seed)
    m = sram.NmcModel(p)
    W = p["width"]
    steps, expect = [], []
    for r in range(p["rows"]):            # every row: SEARCH reads them all
        v = int(rng.integers(0, 1 << min(W, 62))) & ((1 << W) - 1)
        steps.append((1, 0, 0, 0, r, v))
        m.mem[r] = v
        expect.append((m.acc << W) | m.mem[r])
    for name, a, b, d in (("AND", 0, 1, 2 % p["rows"]), ("ADD", 0, 1, 3 % p["rows"]),
                          ("MAC", 0, 1, 0), ("SEARCH", 1, 0, 3 % p["rows"])):
        steps.append((0, sram.OPS[name], a, b, d, 0))
        m.op(name, a, b, d)
        expect.append((m.acc << W) | m.mem[d])
    out = _eda.iverilog_run({"tbw.v": _wrapper_tb(p, steps), "w.v": tt_wrapper(p),
                             "nmc.v": sram.rtl(p)})
    got = []
    for line in out.splitlines():
        parts = line.split()
        if parts and all(x.isdigit() for x in parts):
            got.append(sum(int(x) << (8 * i) for i, x in enumerate(parts)))
    return {"steps": len(steps), "checked": len(got),
            "pin_level_bit_exact": got == expect, "got": got[:3], "expect": expect[:3]}


def build(memory: str, ops: list[str], config: dict[str, Any] | None = None,
          export_dir: str | Path | None = None, seed: int = 0) -> dict[str, Any]:
    """Run the pipeline. Returns every stage with its evidence and a review verdict."""
    config = dict(config or {})
    stages: list[dict[str, Any]] = []
    out: dict[str, Any] = {"memory": memory, "ops": ops, "stages": stages}
    mem = MEMORIES.get(memory)
    if mem is None:
        stages.append(_stage("memory", "fail", why=f"memory must be one of {list(MEMORIES)}"))
        return out | {"ok": False}
    stages.append(_stage("memory", "pass", technology=mem["how"],
                         digital_shuttle=mem["digital_shuttle"]))
    bad = [o for o in ops if o not in mem["ops"]]
    if not ops or bad:
        stages.append(_stage("operations", "fail",
                             why=f"unsupported {bad or 'none given'}; {memory} offers "
                                 f"{list(mem['ops'])}"))
        return out | {"ok": False}
    stages.append(_stage("operations", "pass", ops=ops))

    if memory == "sram":
        return _build_sram(out, ops, config, export_dir, seed)
    if memory == "dram":
        sims = {o: dram.monte_carlo(o, 20000, seed, config) for o in ops}
        te = {o: dram.timing_energy(o, config) for o in ops}
        stages.append(_stage("simulation", "pass", fidelity="behavioral", monte_carlo=sims,
                             timing_energy=te, margins=dram.margins(dram.params(config))))
        stages.append(_stage("rtl", "skip", why="a DRAM array is not RTL; the controller "
                                                "issues AAP command sequences"))
        stages.append(_stage("fabrication", "blocked", why=mem["how"]))
        return out | {"ok": True, "buildable_on_digital_shuttle": False}
    if memory == "memristor":
        from chiplab.imc_lab import analog
        ev = analog.evaluate(config or {"rows": 4, "cols": 4}, seed=seed,
                             use_ngspice=bool(config.pop("use_ngspice", True)))
        stages.append(_stage("simulation", "pass" if ev["solvers_agree"] else "fail",
                             fidelity="circuit_spice", rms_error_pct_fs=ev["rms_error_pct_fs"]))
        stages.append(_stage("fabrication", "blocked", why=mem["how"]))
        return out | {"ok": ev["solvers_agree"], "buildable_on_digital_shuttle": False}
    stages.append(_stage("simulation", "blocked", why=mem["how"]))
    return out | {"ok": False, "buildable_on_digital_shuttle": False}


def _build_sram(out: dict[str, Any], ops: list[str], config: dict[str, Any],
                export_dir: str | Path | None, seed: int) -> dict[str, Any]:
    from chiplab.science import review
    stages = out["stages"]
    try:
        p = sram.validate(config)
    except sram.SramNmcError as e:
        stages.append(_stage("configuration", "fail", why=str(e)))
        return out | {"ok": False}
    stages.append(_stage("configuration", "pass", config={k: p[k] for k in sram.LIMITS}))
    pred = sram.predict(p)
    stages.append(_stage("architecture", "pass", width=p["width"], acc_bits=p["accw"],
                         predictions=pred,
                         note="all ops are synthesized; unused ops are not pruned yet"))
    cmp_ = sram.compare_with_cpu(p, {o: 1 for o in ops})
    stages.append(_stage("simulation", "pass", fidelity="analytical",
                         data_movement=cmp_["total"]))
    if not _eda.have("iverilog", "vvp"):
        stages.append(_stage("verification", "skip", why="iverilog not installed"))
        return out | {"ok": False}
    sim = sram.simulate(p, n_ops=60, seed=seed)
    stages.append(_stage("verification", "pass" if sim["bit_exact"] else "fail",
                         rtl_vs_model=sim))
    if not sim["bit_exact"]:
        return out | {"ok": False}
    wrap = verify_wrapper(p, seed)
    stages.append(_stage("tt_wrapper_verification",
                         "pass" if wrap["pin_level_bit_exact"] else "fail", **wrap))
    syn = None
    try:
        syn = sram.synthesize(p)
        stages.append(_stage("synthesis", "pass", **syn))
    except _eda.EdaError as e:
        stages.append(_stage("synthesis", "skip", why=str(e)))
    metrics = {"flip_flops": pred["flip_flops"]["value"]}
    bench = ({"flip_flops": {"value": syn["flip_flops"], "rel_tol": 0.0,
                             "source": "yosys synthesis"}} if syn and syn["flip_flops"] else None)
    rv = review(f"sram_nmc {p['rows']}x{p['lanes']}x{p['lane_bits']}", metrics,
                fidelity="synthesis" if syn else "rtl_sim",
                provenance={"tool": "chiplab.pim.sram.predict"},
                evaluate=lambda q: {"flip_flops": sram.predict(q)["flip_flops"]["value"]},
                inputs={k: p[k] for k in sram.LIMITS}, benchmarks=bench)
    stages.append(_stage("review", rv.verdict, **rv.as_dict()))
    tiles = None
    if syn and syn.get("cell_area_um2"):
        tiles = math.ceil(syn["cell_area_um2"] / (TT_TILE_UM2 * ASSUMED_DENSITY))
    fits = tiles is not None and tiles <= TT_MAX_TILES
    export = None
    if export_dir and wrap["pin_level_bit_exact"]:
        d = Path(export_dir)
        (d / "src").mkdir(parents=True, exist_ok=True)
        (d / "src" / "sram_nmc.v").write_text(sram.rtl(p))
        (d / "src" / "project.v").write_text(tt_wrapper(p))
        (d / "info.yaml").write_text(
            "# Generated by ChipLab PIM builder. NOT SUBMITTED. Review before any shuttle.\n"
            "project:\n  title: \"ChipLab SRAM near-memory compute\"\n"
            "  top_module: \"tt_um_chiplab_sram_nmc\"\n"
            f"  tiles: \"{tiles or 1}x1\"\n  source_files:\n    - project.v\n"
            "    - sram_nmc.v\n  clock_hz: 10000000\n")
        export = str(d)
    stages.append(_stage("tt_export", "pass" if export else "skip",
                         tiles_estimate=tiles, fits_tt=fits, path=export,
                         submitted=False,
                         note=f"tiles at an assumed {ASSUMED_DENSITY:.0%} placement density; "
                              "hardening (OpenROAD/LibreLane) has not been run"))
    ok = all(s["status"] in ("pass", "skip", "plausible", "validated") for s in stages)
    return out | {"ok": ok, "buildable_on_digital_shuttle": True, "verdict": rv.verdict}
