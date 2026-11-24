// Tiny Tapeout top level for the PEM protocol emulation machine.
// ui_in[0] SCK, ui_in[1] CS_n, ui_in[2] MOSI (host SPI, mode 0); ui_in[7:3] extra inputs.
// uo_out[0] MISO, uo_out[1] IRQ (flag 7), uo_out[7:2] output-only pins 8-13.
// uio[7:0] programmable bidirectional pins 0-7.
`default_nettype none

module pem_spi_host (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        sck,
    input  wire        cs_n,
    input  wire        mosi,
    output wire        miso,
    output reg         bus_we,
    output reg         bus_re,
    output reg  [6:0]  bus_addr,
    output reg  [15:0] bus_wdata,
    input  wire [15:0] bus_rdata
);
    reg [2:0] sck_s, cs_s, mosi_s;
    reg [4:0] nbit;
    reg [23:0] sh;
    reg [15:0] rd;
    reg        rd_mode;
    wire rise = sck_s[2:1] == 2'b01;
    wire fall = sck_s[2:1] == 2'b10;
    wire cs_active = !cs_s[1];
    assign miso = rd[15];

    always @(posedge clk) begin
        if (!rst_n) begin
            sck_s <= 3'd0; cs_s <= 3'b111; mosi_s <= 3'd0; nbit <= 5'd0; sh <= 24'd0;
            rd <= 16'd0; rd_mode <= 1'b0; bus_we <= 1'b0; bus_re <= 1'b0; bus_addr <= 7'd0;
            bus_wdata <= 16'd0;
        end else begin
            sck_s <= {sck_s[1:0], sck}; cs_s <= {cs_s[1:0], cs_n}; mosi_s <= {mosi_s[1:0], mosi};
            bus_we <= 1'b0;
            bus_re <= 1'b0;
            if (bus_re) rd <= bus_rdata;               // capture the addressed value
            if (!cs_active) begin
                nbit <= 5'd0; rd_mode <= 1'b0;
            end else if (rise) begin
                sh <= {sh[22:0], mosi_s[1]};
                nbit <= (nbit == 5'd31) ? nbit : nbit + 5'd1;
                if (nbit == 5'd7) begin
                    bus_addr <= {sh[5:0], mosi_s[1]};
                    rd_mode <= sh[6];
                    if (sh[6]) bus_re <= 1'b1;
                end
                if (nbit == 5'd23 && !rd_mode) begin
                    bus_wdata <= {sh[14:0], mosi_s[1]};
                    bus_we <= 1'b1;
                end
            end else if (fall && rd_mode && nbit > 5'd8) begin
                rd <= {rd[14:0], 1'b0};
            end
        end
    end
endmodule

module tt_um_chiplab_pem (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);
    wire        we, re;
    wire [6:0]  addr;
    wire [15:0] wdata, rdata;
    wire [13:0] out_val;
    wire [7:0]  oe, irq;
    wire        miso;

    pem_spi_host host (
        .clk(clk), .rst_n(rst_n), .sck(ui_in[0]), .cs_n(ui_in[1]), .mosi(ui_in[2]), .miso(miso),
        .bus_we(we), .bus_re(re), .bus_addr(addr), .bus_wdata(wdata), .bus_rdata(rdata)
    );
    pem_core core (
        .clk(clk), .rst_n(rst_n), .bus_we(we), .bus_re(re), .bus_addr(addr), .bus_wdata(wdata),
        .bus_rdata(rdata), .raw_in({ui_in[7:3], uio_in}), .out_val(out_val), .oe(oe), .irq(irq)
    );
    assign uio_out = out_val[7:0];
    assign uio_oe  = oe;
    assign uo_out  = {out_val[13:8], irq[7], miso};
    wire _unused = &{ena, 1'b0};
endmodule
`default_nettype wire
