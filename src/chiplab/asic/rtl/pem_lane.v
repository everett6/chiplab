// PEM-16 lane: one programmable protocol state machine with its serial datapath
// (NRZI coder, bit stuffing/destuffing, CRC snoop). Written from the ISA document in
// chiplab/asic/isa.py; checked cycle-for-cycle against the independent Python model (iss.py).
`default_nettype none

module pem_lane (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        en,
    input  wire        restart,
    input  wire        pc_we,
    input  wire [4:0]  pc_wdata,
    input  wire [15:0] instr,
    input  wire [15:0] cfg_clkdiv,
    input  wire [15:0] cfg_pins,
    input  wire [15:0] cfg_shift,
    input  wire [31:0] crc_poly,
    input  wire [31:0] crc_init,
    input  wire [5:0]  crc_cfg,
    input  wire [3:0]  stuff_pad,
    input  wire [15:0] pins,
    input  wire [7:0]  irq,
    input  wire [1:0]  tx_cnt,
    input  wire [15:0] tx_head,
    input  wire [1:0]  rx_cnt,
    input  wire        err_bit,
    output reg  [4:0]  pc,
    output reg  [13:0] pin_we,
    output reg  [13:0] pin_val,
    output reg  [7:0]  dir_we,
    output reg  [7:0]  dir_val,
    output reg  [7:0]  irq_set,
    output reg  [7:0]  irq_clr,
    output reg         tx_pop,
    output reg         rx_push,
    output reg  [15:0] rx_data,
    output reg         err_set,
    output wire [15:0] dbg_x,
    output wire [15:0] dbg_y
);
    // ------------------------------------------------------------------ state
    reg [4:0]  delay;
    reg [15:0] div;
    reg [15:0] x, y, osr, isr;
    reg [4:0]  osr_cnt, isr_cnt;
    reg [31:0] crc;
    reg        snoop, nrzi, stuff, nrzi_in, destuff, diff;
    reg        lvl_out, lvl_in;
    reg [2:0]  ones_out, ones_in;
    assign dbg_x = x;
    assign dbg_y = y;

    reg [4:0]  n_pc;
    reg [4:0]  n_delay;
    reg [15:0] n_div, n_x, n_y, n_osr, n_isr;
    reg [4:0]  n_osr_cnt, n_isr_cnt;
    reg [31:0] n_crc;
    reg        n_snoop, n_nrzi, n_stuff, n_nrzi_in, n_destuff, n_diff;
    reg        n_lvl_out, n_lvl_in;
    reg [2:0]  n_ones_out, n_ones_in;

    // ------------------------------------------------------------------ decode helpers
    wire [3:0] op    = instr[15:12];
    wire [3:0] dly   = instr[11:8];
    wire [7:0] arg   = instr[7:0];
    wire [3:0] out_base = cfg_pins[3:0];
    wire [3:0] set_base = cfg_pins[7:4];
    wire [3:0] in_base  = cfg_pins[11:8];
    wire [3:0] jmp_pin  = cfg_pins[15:12];
    wire       out_right = cfg_shift[0];
    wire       in_right  = cfg_shift[1];
    wire [4:0] start     = cfg_shift[6:2];
    wire [4:0] wrap      = cfg_shift[11:7];
    wire       autopull  = cfg_shift[12];
    wire [15:0] cd       = (cfg_clkdiv == 16'd0) ? 16'd1 : cfg_clkdiv;
    wire [31:0] pins2    = {pins, pins};
    wire [15:0] rot      = pins2[{1'b0, in_base} +: 16];
    wire [4:0]  nbits    = {1'b0, arg[3:0]} + 5'd1;

    function [15:0] mask16(input [4:0] n);
        mask16 = (n >= 5'd16) ? 16'hFFFF : ((16'd1 << n) - 16'd1);
    endfunction

    function [15:0] rev16(input [15:0] v);
        integer k;
        for (k = 0; k < 16; k = k + 1) rev16[k] = v[15 - k];
    endfunction

    function [31:0] crc_step(input [31:0] c, input b, input [31:0] poly, input [5:0] cc);
        reg [31:0] m, r;
        reg fb;
        reg [5:0] w;
        begin
            w = {1'b0, cc[4:0]} + 6'd1;
            m = (w == 6'd32) ? 32'hFFFFFFFF : ((32'd1 << w) - 32'd1);
            if (cc[5]) begin
                fb = b ^ c[cc[4:0]];
                r = (c << 1) & m;
            end else begin
                fb = b ^ c[0];
                r = c >> 1;
            end
            if (fb) r = r ^ (poly & m);
            crc_step = r;
        end
    endfunction

    // ------------------------------------------------------------------ next state
    reg        stall, rpt, b, rb;
    reg [4:0]  nxt;
    reg [15:0] data, v, src, osr_e;
    reg [4:0]  osr_cnt_e;
    reg [15:0] pw, pv;          // 16-bit internal pin write mask/value (pins 14,15 dropped)
    reg [31:0] srcw;
    reg [3:0]  idx;
    integer    k;

    always @* begin
        n_pc = pc; n_delay = delay; n_div = div; n_x = x; n_y = y; n_osr = osr; n_isr = isr;
        n_osr_cnt = osr_cnt; n_isr_cnt = isr_cnt; n_crc = crc; n_snoop = snoop;
        n_nrzi = nrzi; n_stuff = stuff; n_nrzi_in = nrzi_in; n_destuff = destuff; n_diff = diff;
        n_lvl_out = lvl_out; n_lvl_in = lvl_in; n_ones_out = ones_out; n_ones_in = ones_in;
        pw = 16'd0; pv = 16'd0; dir_we = 8'd0; dir_val = 8'd0; irq_set = 8'd0; irq_clr = 8'd0;
        tx_pop = 1'b0; rx_push = 1'b0; rx_data = isr; err_set = 1'b0;
        stall = 1'b0; rpt = 1'b0; b = 1'b0; rb = 1'b0; data = 16'd0; v = 16'd0; src = 16'd0;
        srcw = 32'd0; idx = 4'd0; osr_e = osr; osr_cnt_e = osr_cnt;
        nxt = (pc == wrap) ? start : pc + 5'd1;
        if (en) begin
            n_div = ((div + 16'd1) >= cd) ? 16'd0 : div + 16'd1;
            if (div == 16'd0) begin
                if (delay != 5'd0) begin
                    n_delay = delay - 5'd1;
                end else begin
                    case (op)
                    4'd1: begin // SET
                        for (k = 0; k < 4; k = k + 1) if (k <= arg[6:5]) begin
                            idx = set_base + k[3:0];
                            if (arg[7]) begin
                                if (idx < 4'd8) begin dir_we[idx[2:0]] = 1'b1; dir_val[idx[2:0]] = arg[k]; end
                            end else begin
                                pw[idx] = 1'b1; pv[idx] = arg[k];
                            end
                        end
                    end
                    4'd2: begin // OUT
                        if (arg[7:5] == 3'd0 && arg[3:0] == 4'd0 && stuff && ones_out == 3'd6) begin
                            // insert a stuffed 0, keep the OSR bit for the next bit time
                            b = 1'b0;
                            if (nrzi) begin n_lvl_out = ~lvl_out; b = ~lvl_out; end
                            pw[out_base] = 1'b1; pv[out_base] = b;
                            if (diff) begin pw[out_base + 4'd1] = 1'b1; pv[out_base + 4'd1] = ~b; end
                            n_ones_out = 3'd0;
                            rpt = 1'b1;
                        end else if (autopull && osr_cnt == 5'd16 && tx_cnt == 2'd0) begin
                            stall = 1'b1;                  // autopull: wait for data
                        end else begin
                            if (autopull && osr_cnt == 5'd16) begin
                                tx_pop = 1'b1; osr_e = tx_head; osr_cnt_e = 5'd0;
                            end
                            if (out_right) begin
                                data = osr_e & mask16(nbits);
                                n_osr = (nbits == 5'd16) ? 16'd0 : (osr_e >> nbits);
                            end else begin
                                data = (osr_e >> (5'd16 - nbits)) & mask16(nbits);
                                n_osr = (nbits == 5'd16) ? 16'd0 : (osr_e << nbits);
                            end
                            n_osr_cnt = ({1'b0, osr_cnt_e} + {1'b0, nbits} > 6'd16) ? 5'd16 : osr_cnt_e + nbits;
                            if (arg[7:5] == 3'd0 && arg[3:0] == 4'd0) begin
                                b = data[0];
                                if (nrzi) begin
                                    if (!data[0]) n_lvl_out = ~lvl_out;
                                    b = data[0] ? lvl_out : ~lvl_out;
                                end
                                pw[out_base] = 1'b1; pv[out_base] = b;
                                if (diff) begin pw[out_base + 4'd1] = 1'b1; pv[out_base + 4'd1] = ~b; end
                                n_ones_out = data[0] ? ((ones_out == 3'd7) ? 3'd7 : ones_out + 3'd1) : 3'd0;
                                if (snoop) n_crc = crc_step(crc, data[0], crc_poly, crc_cfg);
                            end else begin
                                case (arg[7:5])
                                3'd0: for (k = 0; k < 16; k = k + 1) if (k < nbits) begin
                                          idx = out_base + k[3:0]; pw[idx] = 1'b1; pv[idx] = data[k];
                                      end
                                3'd1: for (k = 0; k < 16; k = k + 1) if (k < nbits) begin
                                          idx = out_base + k[3:0];
                                          if (idx < 4'd8) begin dir_we[idx[2:0]] = 1'b1; dir_val[idx[2:0]] = data[k]; end
                                      end
                                3'd2: n_x = data;
                                3'd3: n_y = data;
                                3'd5: nxt = data[4:0];
                                3'd6: begin n_isr = data; n_isr_cnt = nbits; end
                                default: ;
                                endcase
                            end
                        end
                    end
                    4'd3: begin // IN
                        if (arg[7:5] == 3'd0 && arg[3:0] == 4'd0) begin
                            rb = rot[0];
                            b = nrzi_in ? (rb == lvl_in) : rb;
                            if (nrzi_in) n_lvl_in = rb;
                            if (destuff && ones_in == 3'd6) begin
                                if (b) err_set = 1'b1;
                                n_ones_in = 3'd0;
                                rpt = 1'b1;
                            end else begin
                                n_isr = in_right ? ((isr >> 1) | {b, 15'd0}) : {isr[14:0], b};
                                n_isr_cnt = (isr_cnt == 5'd16) ? 5'd16 : isr_cnt + 5'd1;
                                n_ones_in = b ? ((ones_in == 3'd7) ? 3'd7 : ones_in + 3'd1) : 3'd0;
                                if (snoop) n_crc = crc_step(crc, b, crc_poly, crc_cfg);
                            end
                        end else begin
                            case (arg[7:5])
                            3'd0: src = rot;
                            3'd1: src = x;
                            3'd2: src = y;
                            3'd4: src = 16'hFFFF;
                            3'd5: src = crc[15:0];
                            3'd6: src = osr;
                            default: src = 16'd0;
                            endcase
                            src = src & mask16(nbits);
                            if (nbits == 5'd16) n_isr = src;
                            else if (in_right) n_isr = (isr >> nbits) | (src << (5'd16 - nbits));
                            else n_isr = (isr << nbits) | src;
                            n_isr_cnt = ({1'b0, isr_cnt} + {1'b0, nbits} > 6'd16) ? 5'd16 : isr_cnt + nbits;
                        end
                    end
                    4'd4: begin // WAIT
                        case (arg[6:5])
                        2'd0: b = pins[arg[3:0]];
                        2'd1: b = rot[arg[3:0]];
                        2'd2: b = irq[arg[2:0]];
                        default: b = pins[jmp_pin];
                        endcase
                        if (b != arg[7]) stall = 1'b1;
                    end
                    4'd5: begin // JMP
                        case (arg[7:5])
                        3'd0: b = 1'b1;
                        3'd1: b = (x == 16'd0);
                        3'd2: b = (x != 16'd0);
                        3'd3: b = (y == 16'd0);
                        3'd4: b = (y != 16'd0);
                        3'd5: b = (x != y);
                        3'd6: b = pins[jmp_pin];
                        default: b = (osr_cnt < 5'd16);
                        endcase
                        if (arg[7:5] == 3'd2) n_x = x - 16'd1;
                        if (arg[7:5] == 3'd4) n_y = y - 16'd1;
                        if (b) nxt = arg[4:0];
                    end
                    4'd6: begin // PULL
                        if (tx_cnt != 2'd0) begin
                            tx_pop = 1'b1; n_osr = tx_head; n_osr_cnt = 5'd0;
                        end else if (arg[0]) stall = 1'b1;
                        else begin n_osr = x; n_osr_cnt = 5'd0; end
                    end
                    4'd7: begin // PUSH
                        if (rx_cnt < 2'd2) begin
                            rx_push = 1'b1; n_isr = 16'd0; n_isr_cnt = 5'd0;
                        end else if (arg[0]) stall = 1'b1;
                        else begin n_isr = 16'd0; n_isr_cnt = 5'd0; end
                    end
                    4'd8: begin // MOV
                        case (arg[2:0])
                        3'd0: srcw = {16'd0, rot};
                        3'd1: srcw = {16'd0, x};
                        3'd2: srcw = {16'd0, y};
                        3'd3: srcw = {16'd0, isr};
                        3'd4: srcw = {16'd0, osr};
                        3'd5: srcw = crc;
                        3'd6: srcw = 32'd0;
                        default: srcw = {27'd0, err_bit, tx_cnt, rx_cnt};
                        endcase
                        case (arg[4:3])
                        2'd0: v = srcw[15:0];
                        2'd1: v = ~srcw[15:0];
                        2'd2: v = rev16(srcw[15:0]);
                        default: v = srcw[31:16];
                        endcase
                        case (arg[7:5])
                        3'd0: for (k = 0; k < 8; k = k + 1) begin
                                  idx = out_base + k[3:0]; pw[idx] = 1'b1; pv[idx] = v[k];
                              end
                        3'd1: n_x = v;
                        3'd2: n_y = v;
                        3'd3: begin n_isr = v; n_isr_cnt = 5'd0; end
                        3'd4: begin n_osr = v; n_osr_cnt = 5'd0; end
                        3'd5: nxt = v[4:0];
                        3'd6: for (k = 0; k < 8; k = k + 1) begin
                                  idx = out_base + k[3:0];
                                  if (idx < 4'd8) begin dir_we[idx[2:0]] = 1'b1; dir_val[idx[2:0]] = v[k]; end
                              end
                        default: ;
                        endcase
                    end
                    4'd9: begin // IRQ
                        if (arg[3]) irq_clr[arg[2:0]] = 1'b1; else irq_set[arg[2:0]] = 1'b1;
                    end
                    4'd10: n_x = {8'd0, arg};
                    4'd11: n_y = {8'd0, arg};
                    4'd12: begin // LINE
                        n_nrzi = arg[0]; n_stuff = arg[1]; n_nrzi_in = arg[2]; n_destuff = arg[3];
                        n_diff = arg[5];
                        if (arg[4]) begin
                            n_ones_out = 3'd0; n_ones_in = 3'd0; n_lvl_out = 1'b1; n_lvl_in = 1'b1;
                        end
                    end
                    4'd13: begin // CRC
                        if (arg[1:0] == 2'd0 || arg[1:0] == 2'd3)
                            n_crc = crc_init & (({1'b0, crc_cfg[4:0]} == 6'd31) ? 32'hFFFFFFFF
                                    : ((32'd1 << ({1'b0, crc_cfg[4:0]} + 6'd1)) - 32'd1));
                        if (arg[1:0] == 2'd1 || arg[1:0] == 2'd3) n_snoop = 1'b1;
                        if (arg[1:0] == 2'd2) n_snoop = 1'b0;
                    end
                    4'd14: stall = 1'b1; // HALT
                    default: ;
                    endcase
                    if (!stall) begin
                        n_delay = rpt ? {1'b0, stuff_pad} : {1'b0, dly};
                        if (!rpt) n_pc = nxt;
                    end
                end
            end
        end
        pin_we = pw[13:0];
        pin_val = pv[13:0];
    end

    // ------------------------------------------------------------------ registers
    always @(posedge clk) begin
        if (!rst_n || restart) begin
            pc <= rst_n ? start : 5'd0;
            delay <= 5'd0; div <= 16'd0; x <= 16'd0; y <= 16'd0; osr <= 16'd0; isr <= 16'd0;
            osr_cnt <= 5'd16; isr_cnt <= 5'd0; crc <= 32'd0; snoop <= 1'b0;
            nrzi <= 1'b0; stuff <= 1'b0; nrzi_in <= 1'b0; destuff <= 1'b0; diff <= 1'b0;
            lvl_out <= 1'b1; lvl_in <= 1'b1; ones_out <= 3'd0; ones_in <= 3'd0;
        end else begin
            pc <= pc_we ? pc_wdata : n_pc; delay <= pc_we ? 5'd0 : n_delay; div <= n_div; x <= n_x; y <= n_y; osr <= n_osr;
            isr <= n_isr; osr_cnt <= n_osr_cnt; isr_cnt <= n_isr_cnt; crc <= n_crc;
            snoop <= n_snoop; nrzi <= n_nrzi; stuff <= n_stuff; nrzi_in <= n_nrzi_in;
            destuff <= n_destuff; diff <= n_diff; lvl_out <= n_lvl_out; lvl_in <= n_lvl_in;
            ones_out <= n_ones_out; ones_in <= n_ones_in;
        end
    end
endmodule
`default_nettype wire
