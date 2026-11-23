"""Deterministic SystemVerilog-2012 generator for an on-chip memory subsystem.

Emits:
- onchip_mem_subsystem_top.sv: multi-requester crossbar, address interleaver, per-bank
  round-robin/priority arbiter, gain-cell auto-refresh controller, SECDED ECC
  encoder/decoder (explicit XOR trees generated here), near-bank PIM dot-product/MAC
  engine, and an optional AXI4-Lite -> request bridge.
- behavioral_bitcell_macro.sv: simulation model of one bank with retention assertions
  (requires OpenRAM / SPICE characterization for the physical macro).
- tb_onchip_mem.sv: self-checking testbench (Verilator 5 --binary --timing).
- onchip_mem.sdc, run_verilator.sh, README.md, manifest.json.

The same Hamming position tables drive the generated RTL and the Python reference codec
(``secded_encode`` / ``secded_decode``), so the ECC logic is verified in Python.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from chiplab.backends.handoff import export_root, structural_lint
from chiplab.library.pdk_library import get_pdk
from chiplab.models.onchip_memory import RET_FACTOR, RET_HALVING_C, TECHS, analyze
from chiplab.schemas.digital_twin import MemWorkload, OnChipMemoryConfig

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


# --- SECDED tables + Python reference codec -------------------------------------------------

def secded_layout(k: int) -> dict[str, Any]:
    """Hamming(n, k) + overall parity. Codeword bit 0 = overall parity, bits 1..n = Hamming
    positions (parity at powers of two)."""
    r = 0
    while 2**r < k + r + 1:
        r += 1
    n = k + r
    data_pos = [i for i in range(1, n + 1) if i & (i - 1)]
    parity_pos = [1 << j for j in range(r)]
    cover = {p: [i for i in range(1, n + 1) if i & p and i != p] for p in parity_pos}
    return {"k": k, "r": r, "n": n, "width": n + 1, "data_pos": data_pos[:k],
            "parity_pos": parity_pos, "cover": cover}


def secded_encode(data: int, k: int) -> int:
    lay = secded_layout(k)
    code = 0
    for bit, pos in enumerate(lay["data_pos"]):
        if data >> bit & 1:
            code |= 1 << pos
    for p, cov in lay["cover"].items():
        par = 0
        for i in cov:
            par ^= code >> i & 1
        code |= par << p
    overall = bin(code).count("1") & 1
    return code | overall


def secded_decode(code: int, k: int) -> tuple[int, str]:
    """Returns (data, status) with status in {'ok', 'corrected', 'uncorrectable'}."""
    lay = secded_layout(k)
    syn = 0
    for p in lay["parity_pos"]:
        par = code >> p & 1
        for i in lay["cover"][p]:
            par ^= code >> i & 1
        if par:
            syn |= p
    overall_err = bin(code).count("1") & 1
    status = "ok"
    if syn and overall_err:
        if syn <= lay["n"]:
            code ^= 1 << syn
        status = "corrected"
    elif syn and not overall_err:
        status = "uncorrectable"
    elif overall_err:
        status = "corrected"  # error in the overall parity bit only
    data = 0
    for bit, pos in enumerate(lay["data_pos"]):
        data |= (code >> pos & 1) << bit
    return data, status


def _xor(bits: list[str]) -> str:
    return " ^ ".join(bits) if bits else "1'b0"


def gen_ecc_modules(k: int) -> str:
    lay = secded_layout(k)
    w = lay["width"]
    enc = [f"module secded_enc_{k} (", f"  input  logic [{k - 1}:0] data,",
           f"  output logic [{w - 1}:0] code", ");", f"  logic [{w - 1}:0] c;",
           "  always_comb begin", "    c = '0;"]
    for bit, pos in enumerate(lay["data_pos"]):
        enc.append(f"    c[{pos}] = data[{bit}];")
    for p, cov in lay["cover"].items():
        enc.append(f"    c[{p}] = {_xor([f'c[{i}]' for i in cov])};")
    enc += ["    c[0] = ^c;", "  end", "  assign code = c;", "endmodule", ""]
    dec = [f"module secded_dec_{k} (", f"  input  logic [{w - 1}:0] code,",
           f"  output logic [{k - 1}:0] data,",
           "  output logic corrected,", "  output logic uncorrectable", ");",
           f"  logic [{lay['r'] - 1}:0] syn;", "  logic overall;", f"  logic [{w - 1}:0] fixed;",
           "  always_comb begin"]
    for j, p in enumerate(lay["parity_pos"]):
        dec.append(f"    syn[{j}] = {_xor([f'code[{p}]'] + [f'code[{i}]' for i in lay['cover'][p]])};")
    dec += ["    overall = ^code;", "    fixed = code;", "    corrected = 1'b0;",
            "    uncorrectable = 1'b0;",
            "    if (syn != '0 && overall) begin",
            f"      if (int'(syn) <= {lay['n']}) fixed[syn] = ~code[syn];",
            "      corrected = 1'b1;",
            "    end else if (syn != '0) begin", "      uncorrectable = 1'b1;",
            "    end else if (overall) begin", "      corrected = 1'b1;", "    end"]
    for bit, pos in enumerate(lay["data_pos"]):
        dec.append(f"    data[{bit}] = fixed[{pos}];")
    dec += ["  end", "endmodule", ""]
    return "\n".join(enc + dec)


# --- parameters ----------------------------------------------------------------------------

def rtl_params(cfg: OnChipMemoryConfig, freq_mhz: float) -> dict[str, Any]:
    if cfg.banks < 2:
        raise ValueError("RTL generator needs at least 2 banks")
    pdk = get_pdk(cfg.pdk)
    tech = TECHS.index(cfg.technology)
    gain = tech > 0
    ret_us = pdk.scalar("gain_cell_retention_us_25c") * RET_FACTOR[tech] * \
        2.0 ** (-(cfg.temp_c - 25.0) / RET_HALVING_C)
    ret_cycles = int(ret_us * freq_mhz) if gain else 0
    rows = cfg.rows_per_bank
    interval = max(1, int(ret_us * cfg.refresh_margin * freq_mhz // rows)) if gain else 0
    if gain and interval < 2:
        raise ValueError("Retention too short to refresh every row (refresh interval < 2 "
                         "cycles); use more banks, smaller banks, lower clock or SRAM.")
    ecc_w = secded_layout(cfg.word_bits)["width"] if cfg.ecc == "secded" else cfg.word_bits
    lanes = cfg.pim_lanes
    mac = cfg.pim.mac_bits
    acc_w = 2 * mac + max(1, math.ceil(math.log2(max(lanes, 2)))) + 8 if lanes else 1
    if lanes and acc_w > cfg.word_bits:
        raise ValueError("PIM accumulator wider than the data word; widen word_bits or reduce "
                         "lanes/mac_bits")
    words = cfg.banks * cfg.words_per_bank
    if cfg.interleave == "xor" and cfg.banks ** 2 > words:
        raise ValueError("xor interleaving needs words >= banks^2")
    return {
        "NUM_REQ": cfg.num_requesters, "NUM_BANKS": cfg.banks, "DATA_W": cfg.word_bits,
        "CODE_W": ecc_w, "ECC_EN": int(cfg.ecc == "secded"),
        "WORDS_PER_BANK": cfg.words_per_bank, "ADDR_W": int(math.log2(words)),
        "BANK_W": int(math.log2(cfg.banks)), "ROW_W": int(math.log2(cfg.words_per_bank)),
        "COL_MUX": cfg.col_mux, "ROWS": rows, "GAIN_CELL": int(gain),
        "REFRESH_INTERVAL": interval, "RETENTION_CYCLES": ret_cycles,
        "PIM_EN": int(lanes > 0), "PIM_LANES": max(lanes, 1), "MAC_W": mac, "ACC_W": acc_w,
        "RR_ARB": int(cfg.arbiter == "round_robin"),
        "INTERLEAVE": {"low_order": 0, "xor": 1, "block": 2}[cfg.interleave],
        "freq_mhz": freq_mhz, "retention_us": ret_us,
    }


# --- top-level subsystem -------------------------------------------------------------------

def gen_top(cfg: OnChipMemoryConfig, p: dict[str, Any]) -> str:
    k = p["DATA_W"]
    ecc = gen_ecc_modules(k) if p["ECC_EN"] else ""
    axil = gen_axil_bridge(p) if cfg.axi4_lite_bridge else ""
    enc = (f"secded_enc_{k} u_enc (.data(g_wdata), .code(enc_code));" if p["ECC_EN"]
           else "assign enc_code = g_wdata;")
    dec = (f"secded_dec_{k} u_dec (.code(mac_rdata), .data(dec_data), "
           ".corrected(dec_corr), .uncorrectable(dec_unc));" if p["ECC_EN"] else
           "assign dec_data = mac_rdata; assign dec_corr = 1'b0; assign dec_unc = 1'b0;")
    return f"""// onchip_mem_subsystem_top.sv - generated by ChipLab memory_gen (deterministic).
// Config: {cfg.technology} on {cfg.pdk}, {cfg.capacity_kib} KiB, {p['NUM_BANKS']} banks x
// {p['WORDS_PER_BANK']} words x {p['DATA_W']} bits, ECC={'SECDED' if p['ECC_EN'] else 'off'},
// PIM lanes={p['PIM_LANES'] if p['PIM_EN'] else 0}, {p['freq_mhz']:.1f} MHz target.
// Request ops: 0 = read, 1 = write, 2 = PIM dot product, 3 = PIM MAC-accumulate.
// Every accepted request returns exactly one response two cycles later, in order.
`default_nettype none

{ecc}
module onchip_mem_subsystem_top #(
  parameter int NUM_REQ          = {p['NUM_REQ']},
  parameter int NUM_BANKS        = {p['NUM_BANKS']},
  parameter int DATA_W           = {p['DATA_W']},
  parameter int CODE_W           = {p['CODE_W']},
  parameter int WORDS_PER_BANK   = {p['WORDS_PER_BANK']},
  parameter int ADDR_W           = {p['ADDR_W']},
  parameter int BANK_W           = {p['BANK_W']},
  parameter int ROW_W            = {p['ROW_W']},
  parameter int COL_MUX          = {p['COL_MUX']},
  parameter int ROWS             = {p['ROWS']},
  parameter bit GAIN_CELL        = 1'b{p['GAIN_CELL']},
  parameter int REFRESH_INTERVAL = {max(p['REFRESH_INTERVAL'], 1)},
  parameter int RETENTION_CYCLES = {p['RETENTION_CYCLES']},
  parameter bit PIM_EN           = 1'b{p['PIM_EN']},
  parameter int PIM_LANES        = {p['PIM_LANES']},
  parameter int MAC_W            = {p['MAC_W']},
  parameter int ACC_W            = {p['ACC_W']},
  parameter bit RR_ARB           = 1'b{p['RR_ARB']},
  parameter int INTERLEAVE       = {p['INTERLEAVE']}  // 0 low-order, 1 xor, 2 block
) (
  input  logic                              clk,
  input  logic                              rst_n,
  input  logic [NUM_REQ-1:0]                req_valid,
  output logic [NUM_REQ-1:0]                req_ready,
  input  logic [NUM_REQ-1:0][1:0]           req_op,
  input  logic [NUM_REQ-1:0][ADDR_W-1:0]    req_addr,
  input  logic [NUM_REQ-1:0][DATA_W-1:0]    req_wdata,
  output logic [NUM_REQ-1:0]                rsp_valid,
  output logic [NUM_REQ-1:0][DATA_W-1:0]    rsp_rdata,
  output logic [NUM_REQ-1:0][1:0]           rsp_ecc,     // [0] corrected, [1] uncorrectable
  output logic [NUM_BANKS-1:0]              refresh_active,
  input  logic                              dft_inject_err  // test hook: flip one codeword data bit on writes (ECC test)
);
  localparam int REQ_W = (NUM_REQ > 1) ? $clog2(NUM_REQ) : 1;
  // Test hook flips a *data* bit (Hamming position 3 with ECC) so a decoder that fails to
  // correct data is caught, not just one that fails to raise the flag.
  localparam int INJECT_BIT = (CODE_W > 3) ? 3 : 0;

  function automatic logic [BANK_W-1:0] bank_of(input logic [ADDR_W-1:0] a);
    if (INTERLEAVE == 0)      bank_of = a[BANK_W-1:0];
    else if (INTERLEAVE == 1) bank_of = a[BANK_W-1:0] ^ a[2*BANK_W-1:BANK_W];
    else                      bank_of = a[ADDR_W-1:ADDR_W-BANK_W];
  endfunction

  function automatic logic [ROW_W-1:0] word_of(input logic [ADDR_W-1:0] a);
    if (INTERLEAVE == 2) word_of = a[ROW_W-1:0];
    else                 word_of = a[ADDR_W-1:BANK_W];
  endfunction

  logic [NUM_BANKS-1:0][NUM_REQ-1:0] grant;

  always_comb begin
    req_ready = '0;
    for (int b = 0; b < NUM_BANKS; b++) req_ready |= grant[b];
  end

  // Per-requester response stage (each requester has at most one grant per cycle).
  logic [NUM_BANKS-1:0]                 s1_valid;
  logic [NUM_BANKS-1:0][REQ_W-1:0]      s1_req;
  logic [NUM_BANKS-1:0][1:0]            s1_op;
  logic [NUM_BANKS-1:0][DATA_W-1:0]     b_rsp_data;
  logic [NUM_BANKS-1:0][1:0]            b_rsp_ecc;

  for (genvar b = 0; b < NUM_BANKS; b++) begin : g_bank
    logic [NUM_REQ-1:0] hit;
    logic [REQ_W-1:0]   rr_ptr;
    logic               ref_pending;
    logic [$clog2(REFRESH_INTERVAL+1)-1:0] ref_cnt;
    logic [$clog2(ROWS+1)-1:0] ref_row;
    logic               do_ref;
    logic               any_grant;
    logic [REQ_W-1:0]   g_idx;

    always_comb begin
      for (int r = 0; r < NUM_REQ; r++)
        hit[r] = req_valid[r] && (bank_of(req_addr[r]) == BANK_W'(b));
    end

    // Auto-refresh controller: one row every REFRESH_INTERVAL cycles, priority over traffic.
    if (GAIN_CELL) begin : g_refresh
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          ref_cnt <= '0; ref_pending <= 1'b0; ref_row <= '0;
        end else begin
          if (int'(ref_cnt) == REFRESH_INTERVAL - 1) begin
            ref_cnt <= '0; ref_pending <= 1'b1;
          end else begin
            ref_cnt <= ref_cnt + 1'b1;
          end
          if (do_ref) begin
            ref_pending <= 1'b0;
            ref_row <= (int'(ref_row) == ROWS - 1) ? '0 : ref_row + 1'b1;
          end
        end
      end
`ifndef SYNTHESIS
      // A pending refresh must never wait past the next interval (deadline guarantee).
      always @(posedge clk) if (ref_pending && int'(ref_cnt) == REFRESH_INTERVAL - 1)
        $fatal(1, "bank %0d: refresh deadline missed", b);
`endif
    end else begin : g_no_refresh
      assign ref_pending = 1'b0;
      assign ref_cnt = '0;
      assign ref_row = '0;
    end
    assign do_ref = GAIN_CELL && ref_pending;
    assign refresh_active[b] = do_ref;

    // Arbiter: round-robin (rotating priority from rr_ptr) or fixed priority (lowest index).
    int r;
    always_comb begin
      any_grant = 1'b0;
      g_idx = '0;
      grant[b] = '0;
      r = 0;
      if (!do_ref) begin
        for (int k = 0; k < NUM_REQ; k++) begin
          r = RR_ARB ? ((int'(rr_ptr) + k) % NUM_REQ) : k;
          if (!any_grant && hit[r]) begin
            any_grant = 1'b1;
            g_idx = REQ_W'(r);
          end
        end
        if (any_grant) grant[b][g_idx] = 1'b1;
      end
    end

    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) rr_ptr <= '0;
      else if (any_grant) rr_ptr <= (int'(g_idx) == NUM_REQ - 1) ? '0 : g_idx + 1'b1;
    end

    // Bank datapath
    logic [1:0]        g_op;
    logic [ROW_W-1:0]  g_word;
    logic [DATA_W-1:0] g_wdata;
    logic [CODE_W-1:0] enc_code, wr_code, mac_rdata;
    logic [DATA_W-1:0] dec_data;
    logic              dec_corr, dec_unc;
    logic [DATA_W-1:0] s1_operand;
    assign g_op    = req_op[g_idx];
    assign g_word  = word_of(req_addr[g_idx]);
    assign g_wdata = req_wdata[g_idx];
    {enc}
    assign wr_code = enc_code ^ (CODE_W'(dft_inject_err) << INJECT_BIT);

    behavioral_bitcell_macro #(
      .WORDS(WORDS_PER_BANK), .WIDTH(CODE_W), .COL_MUX(COL_MUX), .ROWS(ROWS),
      .GAIN_CELL(GAIN_CELL), .RETENTION_CYCLES(RETENTION_CYCLES), .BANK_ID(b)
    ) u_macro (
      .clk(clk), .en(any_grant), .we(any_grant && g_op == 2'd1), .addr(g_word),
      .wdata(wr_code), .rdata(mac_rdata),
      .ref_en(do_ref), .ref_row(ref_row[$clog2(ROWS+1)-1:0])
    );

    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        s1_valid[b] <= 1'b0; s1_req[b] <= '0; s1_op[b] <= '0; s1_operand <= '0;
      end else begin
        s1_valid[b] <= any_grant; s1_req[b] <= g_idx; s1_op[b] <= g_op; s1_operand <= g_wdata;
      end
    end

    {dec}

    // Near-bank PIM: signed dot product of PIM_LANES x MAC_W lanes (op 2) and accumulate (op 3).
    logic signed [ACC_W-1:0] dot, acc;
    always_comb begin
      dot = '0;
      if (PIM_EN)
        for (int l = 0; l < PIM_LANES; l++)
          dot += ACC_W'($signed(dec_data[l*MAC_W +: MAC_W]) * $signed(s1_operand[l*MAC_W +: MAC_W]));
    end
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) acc <= '0;
      else if (PIM_EN && s1_valid[b] && s1_op[b] == 2'd3) acc <= acc + dot;
    end

    always_comb begin
      if (PIM_EN && s1_op[b] == 2'd2)      b_rsp_data[b] = DATA_W'($unsigned(dot));
      else if (PIM_EN && s1_op[b] == 2'd3) b_rsp_data[b] = DATA_W'($unsigned(acc + dot));
      else                                 b_rsp_data[b] = dec_data;
      b_rsp_ecc[b] = {{dec_unc, dec_corr}};
    end
  end

  // Response mux (registered): route each bank's stage-1 result to its requester.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rsp_valid <= '0; rsp_rdata <= '0; rsp_ecc <= '0;
    end else begin
      rsp_valid <= '0;
      for (int b = 0; b < NUM_BANKS; b++) begin
        if (s1_valid[b]) begin
          rsp_valid[s1_req[b]] <= 1'b1;
          rsp_rdata[s1_req[b]] <= b_rsp_data[b];
          rsp_ecc[s1_req[b]]   <= (s1_op[b] == 2'd1) ? 2'b00 : b_rsp_ecc[b];
        end
      end
    end
  end
endmodule

{axil}
`default_nettype wire
"""


def gen_axil_bridge(p: dict[str, Any]) -> str:
    return f"""// AXI4-Lite slave -> one requester port of onchip_mem_subsystem_top (single outstanding
// transaction; word-addressed: AXI byte address >> $clog2(DATA_W/8)). For full AXI4 bursts,
// put pulp-platform/axi (see the ChipLab IP catalog) in front of this bridge.
module axil_to_mem_req #(
  parameter int DATA_W = {p['DATA_W']},
  parameter int ADDR_W = {p['ADDR_W']}
) (
  input  logic              clk,
  input  logic              rst_n,
  input  logic [ADDR_W+$clog2(DATA_W/8)-1:0] s_awaddr,
  input  logic              s_awvalid,
  output logic              s_awready,
  input  logic [DATA_W-1:0] s_wdata,
  input  logic              s_wvalid,
  output logic              s_wready,
  output logic [1:0]        s_bresp,
  output logic              s_bvalid,
  input  logic              s_bready,
  input  logic [ADDR_W+$clog2(DATA_W/8)-1:0] s_araddr,
  input  logic              s_arvalid,
  output logic              s_arready,
  output logic [DATA_W-1:0] s_rdata,
  output logic [1:0]        s_rresp,
  output logic              s_rvalid,
  input  logic              s_rready,
  output logic              m_req_valid,
  input  logic              m_req_ready,
  output logic [1:0]        m_req_op,
  output logic [ADDR_W-1:0] m_req_addr,
  output logic [DATA_W-1:0] m_req_wdata,
  input  logic              m_rsp_valid,
  input  logic [DATA_W-1:0] m_rsp_rdata,
  input  logic [1:0]        m_rsp_ecc
);
  localparam int OFF = $clog2(DATA_W/8);
  typedef enum logic [2:0] {{IDLE, REQ, WAIT, BRESP, RRESP}} state_t;
  state_t state;
  logic is_write;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE; is_write <= 1'b0; m_req_addr <= '0; m_req_wdata <= '0; m_req_op <= '0;
      s_rdata <= '0; s_rresp <= '0; s_bresp <= '0;
    end else begin
      case (state)
        IDLE: if (s_awvalid && s_wvalid) begin
                is_write <= 1'b1; m_req_op <= 2'd1; m_req_addr <= s_awaddr[ADDR_W+OFF-1:OFF];
                m_req_wdata <= s_wdata; state <= REQ;
              end else if (s_arvalid) begin
                is_write <= 1'b0; m_req_op <= 2'd0; m_req_addr <= s_araddr[ADDR_W+OFF-1:OFF];
                state <= REQ;
              end
        REQ:  if (m_req_ready) state <= WAIT;
        WAIT: if (m_rsp_valid) begin
                s_rdata <= m_rsp_rdata;
                s_rresp <= m_rsp_ecc[1] ? 2'b10 : 2'b00;   // SLVERR on uncorrectable ECC
                s_bresp <= 2'b00;
                state <= is_write ? BRESP : RRESP;
              end
        BRESP: if (s_bready) state <= IDLE;
        RRESP: if (s_rready) state <= IDLE;
        default: state <= IDLE;
      endcase
    end
  end
  assign s_awready   = (state == IDLE) && s_awvalid && s_wvalid;
  assign s_wready    = (state == IDLE) && s_awvalid && s_wvalid;
  assign s_arready   = (state == IDLE) && !(s_awvalid && s_wvalid) && s_arvalid;
  assign m_req_valid = (state == REQ);
  assign s_bvalid    = (state == BRESP);
  assign s_rvalid    = (state == RRESP);
endmodule
"""


def gen_macro() -> str:
    return """// behavioral_bitcell_macro.sv - generated by ChipLab memory_gen.
// SIMULATION MODEL ONLY. Replace with an OpenRAM-compiled (SRAM) or SPICE-characterized
// gain-cell macro for physical design; timing, retention and power here are not signoff.
// 1-cycle registered read, write-through-free (read returns old data on same-cycle write).
// For gain cells, every word tracks the cycle it was last written or refreshed and a read
// older than RETENTION_CYCLES is a fatal error (data would have decayed).
`default_nettype none
module behavioral_bitcell_macro #(
  parameter int WORDS = 1024,
  parameter int WIDTH = 64,
  parameter int COL_MUX = 4,
  parameter int ROWS = 256,
  parameter bit GAIN_CELL = 1'b0,
  parameter int RETENTION_CYCLES = 0,
  parameter int BANK_ID = 0
) (
  input  logic                       clk,
  input  logic                       en,
  input  logic                       we,
  input  logic [$clog2(WORDS)-1:0]   addr,
  input  logic [WIDTH-1:0]           wdata,
  output logic [WIDTH-1:0]           rdata,
  input  logic                       ref_en,
  input  logic [$clog2(ROWS+1)-1:0]  ref_row
);
  logic [WIDTH-1:0] mem [WORDS];

  always_ff @(posedge clk) begin
    if (en) begin
      if (we) mem[addr] <= wdata;
      rdata <= mem[addr];
    end
  end

`ifndef SYNTHESIS
  longint unsigned cycle;
  longint unsigned last_touch [WORDS];
  initial begin
    cycle = 0;
    for (int i = 0; i < WORDS; i++) begin
      last_touch[i] = 0;
      mem[i] = '0;
    end
  end
  always @(posedge clk) begin
    cycle <= cycle + 1;
    if (GAIN_CELL) begin
      if (en && we) last_touch[addr] <= cycle;
      if (ref_en)
        for (int c = 0; c < COL_MUX; c++)
          if (int'(ref_row) * COL_MUX + c < WORDS) last_touch[int'(ref_row) * COL_MUX + c] <= cycle;
      if (en && !we && RETENTION_CYCLES > 0 && cycle - last_touch[addr] > longint'(RETENTION_CYCLES))
        $fatal(1, "bank %0d word %0d read %0d cycles after last write/refresh (retention %0d)",
               BANK_ID, addr, cycle - last_touch[addr], RETENTION_CYCLES);
    end
  end
`endif
endmodule
`default_nettype wire
"""


def gen_testbench(cfg: OnChipMemoryConfig, p: dict[str, Any], seed: int = 1,
                  ops: int = 4000) -> str:
    idle = 2 * p["RETENTION_CYCLES"] + 100 if p["GAIN_CELL"] else 0
    return f"""// tb_onchip_mem.sv - self-checking testbench generated by ChipLab memory_gen.
// Run: verilator --binary --timing -Wno-fatal --top-module tb_onchip_mem \\
//        tb_onchip_mem.sv onchip_mem_subsystem_top.sv behavioral_bitcell_macro.sv
// Each requester owns the addresses with addr % NUM_REQ == id, so a per-requester scoreboard
// (reference memory + in-order expected-response FIFO) checks every response exactly.
`timescale 1ns/1ps
`default_nettype none
module tb_onchip_mem;
  localparam int NUM_REQ = {p['NUM_REQ']};
  localparam int NUM_BANKS = {p['NUM_BANKS']};
  localparam int DATA_W = {p['DATA_W']};
  localparam int ADDR_W = {p['ADDR_W']};
  localparam int WORDS = {cfg.banks * cfg.words_per_bank};
  localparam bit ECC_EN = 1'b{p['ECC_EN']};
  localparam bit PIM_EN = 1'b{p['PIM_EN']};
  localparam int PIM_LANES = {p['PIM_LANES']};
  localparam int MAC_W = {p['MAC_W']};
  localparam int ACC_W = {p['ACC_W']};
  localparam int OPS = {ops};
  localparam int IDLE_CYCLES = {idle};
  localparam int FIFO = 8;

  logic clk = 1'b0, rst_n = 1'b0, inject = 1'b0;
  always #1 clk = ~clk;

  logic [NUM_REQ-1:0] req_valid, req_ready, rsp_valid;
  logic [NUM_REQ-1:0][1:0] req_op, rsp_ecc;
  logic [NUM_REQ-1:0][ADDR_W-1:0] req_addr;
  logic [NUM_REQ-1:0][DATA_W-1:0] req_wdata, rsp_rdata;
  logic [NUM_BANKS-1:0] refresh_active;

  onchip_mem_subsystem_top dut (
    .clk, .rst_n, .req_valid, .req_ready, .req_op, .req_addr, .req_wdata,
    .rsp_valid, .rsp_rdata, .rsp_ecc, .refresh_active, .dft_inject_err(inject)
  );

  logic [DATA_W-1:0] ref_mem [WORDS];
  logic signed [ACC_W-1:0] ref_acc [NUM_BANKS];
  logic [DATA_W-1:0] exp_data [NUM_REQ][FIFO];
  logic [1:0]        exp_op   [NUM_REQ][FIFO];
  logic              exp_corr [NUM_REQ][FIFO];
  int exp_head [NUM_REQ], exp_tail [NUM_REQ];
  logic injected [WORDS];
  int errors = 0, checked = 0, corrected_seen = 0, issued = 0;
  bit phase_inject = 1'b0;
  bit accepted [NUM_REQ];

  function automatic logic signed [ACC_W-1:0] dot(input logic [DATA_W-1:0] a,
                                                   input logic [DATA_W-1:0] b);
    logic signed [ACC_W-1:0] s = '0;
    for (int l = 0; l < PIM_LANES; l++)
      s += ACC_W'($signed(a[l*MAC_W +: MAC_W]) * $signed(b[l*MAC_W +: MAC_W]));
    return s;
  endfunction

  function automatic int bank_of(input int a);
    return {"a % NUM_BANKS" if p['INTERLEAVE'] == 0 else
            "(a % NUM_BANKS) ^ ((a / NUM_BANKS) % NUM_BANKS)" if p['INTERLEAVE'] == 1 else
            "a / (WORDS / NUM_BANKS)"};
  endfunction

  // Monitor: record accepted requests, check responses (sampled at posedge, pre-update).
  always @(posedge clk) if (rst_n) begin
    for (int r = 0; r < NUM_REQ; r++) begin
      if (req_valid[r] && req_ready[r]) begin
        automatic int a = int'(req_addr[r]);
        automatic int t = exp_tail[r] % FIFO;
        accepted[r] = 1'b1;
        exp_op[r][t] = req_op[r];
        exp_corr[r][t] = injected[a];
        case (req_op[r])
          2'd1: begin
            exp_data[r][t] = '0;
            ref_mem[a] = req_wdata[r];
            injected[a] = phase_inject && ECC_EN;
          end
          2'd2: exp_data[r][t] = PIM_EN ? DATA_W'($unsigned(dot(ref_mem[a], req_wdata[r])))
                                        : ref_mem[a];
          2'd3: begin
            if (PIM_EN) begin
              ref_acc[bank_of(a)] = ref_acc[bank_of(a)] + dot(ref_mem[a], req_wdata[r]);
              exp_data[r][t] = DATA_W'($unsigned(ref_acc[bank_of(a)]));
            end else exp_data[r][t] = ref_mem[a];
          end
          default: exp_data[r][t] = ref_mem[a];
        endcase
        exp_tail[r]++;
      end
      if (rsp_valid[r]) begin
        automatic int h = exp_head[r] % FIFO;
        if (exp_head[r] == exp_tail[r]) begin
          $error("req %0d: unexpected response", r); errors++;
        end else begin
          if (exp_op[r][h] != 2'd1 && rsp_rdata[r] !== exp_data[r][h]) begin
            $error("req %0d op %0d: got %h expected %h", r, exp_op[r][h], rsp_rdata[r],
                   exp_data[r][h]);
            errors++;
          end
          if (rsp_ecc[r][1]) begin $error("req %0d: uncorrectable ECC error", r); errors++; end
          if (exp_op[r][h] != 2'd1 && exp_corr[r][h]) begin
            if (!rsp_ecc[r][0]) begin
              $error("req %0d: injected error not reported as corrected", r); errors++;
            end else corrected_seen++;
          end else if (rsp_ecc[r][0]) begin
            $error("req %0d: spurious ECC correction", r); errors++;
          end
          checked++;
          exp_head[r]++;
        end
      end
    end
  end

  // Driver: random traffic per requester, one outstanding handshake at a time.
  // window > 0 confines each requester to its first `window` addresses so read-backs hit
  // recently written (e.g. error-injected) words.
  task automatic drive(input int n_ops, input bit allow_pim, input int window = 0);
    int done = 0;
    int span = (window > 0 && window < WORDS / NUM_REQ) ? window : WORDS / NUM_REQ;
    while (done < n_ops) begin
      @(negedge clk);
      for (int r = 0; r < NUM_REQ; r++) begin
        if (!req_valid[r] || accepted[r]) begin
          if (accepted[r]) begin done++; accepted[r] = 1'b0; end
          if ($urandom_range(0, 3) != 0 && (exp_tail[r] - exp_head[r]) < FIFO - 2) begin
            int a = ($urandom_range(0, span - 1) * NUM_REQ) + r;
            int sel = $urandom_range(0, 9);
            req_valid[r] = 1'b1;
            req_addr[r]  = ADDR_W'(a);
            req_wdata[r] = DATA_W'({{$urandom, $urandom, $urandom, $urandom, $urandom, $urandom,
                             $urandom, $urandom}});
            req_op[r]    = (sel < 4) ? 2'd1 : (allow_pim && PIM_EN && sel == 8) ? 2'd2 :
                           (allow_pim && PIM_EN && sel == 9) ? 2'd3 : 2'd0;
            issued++;
          end else req_valid[r] = 1'b0;
        end
      end
    end
    @(negedge clk);
    while (req_valid != '0) begin
      for (int r = 0; r < NUM_REQ; r++)
        if (accepted[r]) begin req_valid[r] = 1'b0; accepted[r] = 1'b0; end
      @(negedge clk);
    end
    repeat (8) @(negedge clk);
  endtask

  initial begin
    void'($urandom({seed}));
    req_valid = '0; req_op = '0; req_addr = '0; req_wdata = '0;
    for (int i = 0; i < WORDS; i++) begin ref_mem[i] = '0; injected[i] = 1'b0; end
    for (int b = 0; b < NUM_BANKS; b++) ref_acc[b] = '0;
    for (int r = 0; r < NUM_REQ; r++) begin
      exp_head[r] = 0; exp_tail[r] = 0; accepted[r] = 1'b0;
    end
    repeat (4) @(negedge clk);
    rst_n = 1'b1;
    drive(OPS, 1'b1);
    if (ECC_EN) begin                       // writes with a flipped codeword bit
      phase_inject = 1'b1; inject = 1'b1;
      drive(OPS / 4, 1'b0, 16);
      phase_inject = 1'b0; inject = 1'b0;
      drive(OPS / 4, 1'b0, 16);             // reads must come back corrected
      if (corrected_seen == 0) begin
        $error("ECC correction path was never exercised"); errors++;
      end
    end
    if (IDLE_CYCLES > 0) begin              // idle past retention: refresh must keep data
      repeat (IDLE_CYCLES) @(negedge clk);
      drive(OPS / 4, 1'b0);
    end
    for (int r = 0; r < NUM_REQ; r++)
      if (exp_head[r] != exp_tail[r]) begin $error("req %0d: missing responses", r); errors++; end
    if (errors == 0 && checked > 0) $display("TB PASS: %0d responses checked, %0d ECC corrections",
                                              checked, corrected_seen);
    else $fatal(1, "TB FAIL: %0d errors (%0d checked)", errors, checked);
    $finish;
  end
endmodule
`default_nettype wire
"""


def gen_sdc(p: dict[str, Any]) -> str:
    period = 1000.0 / p["freq_mhz"]
    return "\n".join([
        "# onchip_mem.sdc - generated by ChipLab memory_gen",
        f"create_clock -name clk -period {period:.4f} [get_ports clk]",
        f"set_clock_uncertainty {0.05 * period:.4f} [get_clocks clk]",
        f"set_input_delay {0.25 * period:.4f} -clock clk [remove_from_collection [all_inputs] "
        "[get_ports {clk rst_n dft_inject_err}]]",
        f"set_output_delay {0.25 * period:.4f} -clock clk [all_outputs]",
        "set_false_path -from [get_ports rst_n]",
        "set_false_path -from [get_ports dft_inject_err]",
        "# The bitcell macro is replaced by a characterized hard macro; its Liberty model "
        "supplies the array timing.",
        "set_dont_touch [get_cells -hier *u_macro*]", ""])


def generate(cfg: OnChipMemoryConfig, wl: MemWorkload | None = None, seed: int = 1,
             tb_ops: int = 4000) -> dict[str, Any]:
    """Return {files: {name: text}, params, report, lint}."""
    report = analyze(cfg, wl)
    p = rtl_params(cfg, report["freq_mhz"])
    files = {
        "onchip_mem_subsystem_top.sv": gen_top(cfg, p),
        "behavioral_bitcell_macro.sv": gen_macro(),
        "tb_onchip_mem.sv": gen_testbench(cfg, p, seed, tb_ops),
        "onchip_mem.sdc": gen_sdc(p),
        "run_verilator.sh": "\n".join([
            "#!/usr/bin/env bash", "# Generated by ChipLab (not executed by ChipLab).",
            "set -euo pipefail",
            "verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM "
            "--top-module onchip_mem_subsystem_top onchip_mem_subsystem_top.sv "
            "behavioral_bitcell_macro.sv",
            "verilator --binary --timing -Wno-fatal -Wno-lint -Wno-style --top-module tb_onchip_mem "
            "tb_onchip_mem.sv onchip_mem_subsystem_top.sv behavioral_bitcell_macro.sv",
            "./obj_dir/Vtb_onchip_mem", ""]),
    }
    lint = {name: structural_lint(text, testbench=name.startswith("tb_"))
            for name, text in files.items() if name.endswith(".sv")}
    return {"files": files, "params": p, "report": report, "lint": lint}


def write_bundle(cfg: OnChipMemoryConfig, wl: MemWorkload | None = None,
                 output_name: str | None = None, seed: int = 1) -> dict[str, Any]:
    gen = generate(cfg, wl, seed)
    root = export_root()
    name = _SAFE.sub("_", output_name or f"mem_{cfg.name}").strip("._") or "mem"
    out = (root / name).resolve()
    if root not in out.parents:
        raise PermissionError("output_name must stay inside the export directory")
    out.mkdir(parents=True, exist_ok=True)
    files = dict(gen["files"])
    files["README.md"] = (
        f"# {cfg.name}: generated on-chip memory subsystem\n\n"
        f"{cfg.technology} on `{cfg.pdk}`, {cfg.capacity_kib} KiB, {cfg.banks} banks, "
        f"{cfg.word_bits}-bit words, ECC {cfg.ecc}, PIM lanes {cfg.pim_lanes}.\n\n"
        "Run `bash run_verilator.sh` (Verilator 5) to lint and run the self-checking "
        "testbench. `behavioral_bitcell_macro.sv` is a simulation model: generate the "
        "physical macro with OpenRAM (SRAM) or characterize the gain cell in SPICE.\n")
    manifest = {"config": cfg.model_dump(mode="json"), "params": gen["params"],
                "metrics": gen["report"]["metrics"], "lint": gen["lint"], "files": {}}
    for fname, text in files.items():
        (out / fname).write_text(text)
        manifest["files"][fname] = hashlib.sha256(text.encode()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return {"directory": str(out), "files": sorted([*files, "manifest.json"]),
            "lint": {k: (v or ["PASS"]) for k, v in gen["lint"].items()},
            "params": gen["params"], "report": gen["report"]}
