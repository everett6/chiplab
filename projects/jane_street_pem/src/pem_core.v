// PEM core: four lanes sharing a 32x16 instruction memory, pins, IRQ flags and a host bus.
`default_nettype none

module pem_core (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        bus_we,
    input  wire        bus_re,
    input  wire [6:0]  bus_addr,
    input  wire [15:0] bus_wdata,
    output reg  [15:0] bus_rdata,
    input  wire [12:0] raw_in,
    output reg  [13:0] out_val,
    output reg  [7:0]  oe,
    output reg  [7:0]  irq
);
    reg [15:0] imem [0:31];
    reg [15:0] cfg  [0:31];            // lane*8 + reg
    reg [3:0]  enable, err;
    reg [12:0] sync1, sync2;
    reg [15:0] tx0 [0:3], tx1 [0:3], rx0 [0:3], rx1 [0:3];
    reg [1:0]  txc [0:3], rxc [0:3];

    wire [15:0] pins = {3'b000, sync2};
    wire        is_fifo = (bus_addr[6:2] == 5'b10000);       // 0x40-0x43
    wire        restart_any = bus_we && bus_addr == 7'h48;

    wire [4:0]  pc      [0:3];
    wire [13:0] pin_we  [0:3];
    wire [13:0] pin_val [0:3];
    wire [7:0]  dir_we  [0:3];
    wire [7:0]  dir_val [0:3];
    wire [7:0]  irq_set [0:3];
    wire [7:0]  irq_clr [0:3];
    wire [3:0]  tx_pop, rx_push, err_set;
    wire [15:0] rx_data [0:3];
    wire [15:0] dbg_x [0:3];
    wire [15:0] dbg_y [0:3];

    genvar g;
    generate for (g = 0; g < 4; g = g + 1) begin : lane
        pem_lane u (
            .clk(clk), .rst_n(rst_n), .en(enable[g]),
            .restart(restart_any && bus_wdata[4 + g]),
            .pc_we(bus_we && bus_addr == 7'h50 + g), .pc_wdata(bus_wdata[4:0]),
            .instr(imem[pc[g]]),
            .cfg_clkdiv(cfg[g*8 + 0]), .cfg_pins(cfg[g*8 + 1]), .cfg_shift(cfg[g*8 + 2]),
            .crc_poly({cfg[g*8 + 4], cfg[g*8 + 3]}), .crc_init({cfg[g*8 + 6], cfg[g*8 + 5]}),
            .crc_cfg(cfg[g*8 + 7][5:0]),
            .stuff_pad(cfg[g*8 + 7][9:6]),
            .pins(pins), .irq(irq), .tx_cnt(txc[g]), .tx_head(tx0[g]), .rx_cnt(rxc[g]),
            .err_bit(err[g]),
            .pc(pc[g]), .pin_we(pin_we[g]), .pin_val(pin_val[g]), .dir_we(dir_we[g]),
            .dir_val(dir_val[g]), .irq_set(irq_set[g]), .irq_clr(irq_clr[g]),
            .tx_pop(tx_pop[g]), .rx_push(rx_push[g]), .rx_data(rx_data[g]), .err_set(err_set[g]),
            .dbg_x(dbg_x[g]), .dbg_y(dbg_y[g])
        );
    end endgenerate

    // ------------------------------------------------------------------ bus read (combinational)
    always @* begin
        bus_rdata = 16'd0;
        if (bus_addr < 7'h20) bus_rdata = imem[bus_addr[4:0]];
        else if (bus_addr < 7'h40) bus_rdata = cfg[bus_addr[4:0]];
        else if (is_fifo) bus_rdata = (rxc[bus_addr[1:0]] != 2'd0) ? rx0[bus_addr[1:0]] : 16'd0;
        else if (bus_addr == 7'h48) bus_rdata = {12'd0, enable};
        else if (bus_addr == 7'h49) bus_rdata = {irq,
            txc[3] != 2'd2, txc[2] != 2'd2, txc[1] != 2'd2, txc[0] != 2'd2,
            rxc[3] != 2'd0, rxc[2] != 2'd0, rxc[1] != 2'd0, rxc[0] != 2'd0};
        else if (bus_addr == 7'h4A) bus_rdata = {12'd0, err};
        else if (bus_addr[6:2] == 5'b10100) bus_rdata = {11'd0, pc[bus_addr[1:0]]};
        else if (bus_addr[6:2] == 5'b10101) bus_rdata = dbg_x[bus_addr[1:0]];
        else if (bus_addr[6:2] == 5'b10110) bus_rdata = dbg_y[bus_addr[1:0]];
    end

    // ------------------------------------------------------------------ shared state
    reg [13:0] ov;
    reg [7:0]  oev, s, c;
    integer i;
    always @* begin
        ov = out_val; oev = oe; s = 8'd0; c = 8'd0;
        for (i = 0; i < 4; i = i + 1) begin
            ov  = (ov & ~pin_we[i]) | (pin_val[i] & pin_we[i]);
            oev = (oev & ~dir_we[i]) | (dir_val[i] & dir_we[i]);
            s = s | irq_set[i];
            c = c | irq_clr[i];
        end
    end

    integer l;
    wire [3:0] push_ok, pop_ok;
    generate for (g = 0; g < 4; g = g + 1) begin : fifo_ok
        assign push_ok[g] = bus_we && is_fifo && bus_addr[1:0] == g && txc[g] != 2'd2;
        assign pop_ok[g]  = bus_re && is_fifo && bus_addr[1:0] == g && rxc[g] != 2'd0;
    end endgenerate
    always @(posedge clk) begin
        if (!rst_n) begin
            out_val <= 14'd0; oe <= 8'd0; irq <= 8'd0; enable <= 4'd0; err <= 4'd0;
            sync1 <= 13'd0; sync2 <= 13'd0;
            for (l = 0; l < 4; l = l + 1) begin txc[l] <= 2'd0; rxc[l] <= 2'd0; end
        end else begin
            out_val <= ov; oe <= oev; irq <= (irq & ~c) | s;
            err <= (err | err_set) & ~((bus_we && bus_addr == 7'h4A) ? bus_wdata[3:0] : 4'd0);
            sync2 <= sync1; sync1 <= raw_in;
            if (bus_we && bus_addr < 7'h20) imem[bus_addr[4:0]] <= bus_wdata;
            if (bus_we && bus_addr >= 7'h20 && bus_addr < 7'h40) cfg[bus_addr[4:0]] <= bus_wdata;
            if (restart_any) enable <= bus_wdata[3:0];
            for (l = 0; l < 4; l = l + 1) begin
                // TX FIFO: host pushes, lane pops (pre-edge occupancy decides)
                if (tx_pop[l] && push_ok[l]) begin
                    tx0[l] <= bus_wdata;               // count stays 1
                end else if (tx_pop[l]) begin
                    tx0[l] <= tx1[l]; txc[l] <= txc[l] - 2'd1;
                end else if (push_ok[l]) begin
                    if (txc[l] == 2'd0) tx0[l] <= bus_wdata; else tx1[l] <= bus_wdata;
                    txc[l] <= txc[l] + 2'd1;
                end
                // RX FIFO: lane pushes, host pops
                if (rx_push[l] && pop_ok[l]) begin
                    if (rxc[l] == 2'd1) rx0[l] <= rx_data[l];
                    else begin rx0[l] <= rx1[l]; rx1[l] <= rx_data[l]; end
                end else if (pop_ok[l]) begin
                    rx0[l] <= rx1[l]; rxc[l] <= rxc[l] - 2'd1;
                end else if (rx_push[l]) begin
                    if (rxc[l] == 2'd0) rx0[l] <= rx_data[l]; else rx1[l] <= rx_data[l];
                    rxc[l] <= rxc[l] + 2'd1;
                end
                if (restart_any && bus_wdata[4 + l]) begin txc[l] <= 2'd0; rxc[l] <= 2'd0; end
            end
        end
    end

`ifdef PEM_ASSERT
    integer a;
    always @(posedge clk) if (rst_n) begin
        for (a = 0; a < 4; a = a + 1) begin
            assert (txc[a] <= 2'd2);                       // A1 TX occupancy bounded
            assert (rxc[a] <= 2'd2);                       // A2 RX occupancy bounded
            assert (!(rx_push[a] && rxc[a] == 2'd2));      // A3 lane never overflows RX
            assert (!(tx_pop[a] && txc[a] == 2'd0));       // A4 lane never pops empty TX
        end
    end
`endif
endmodule
`default_nettype wire
