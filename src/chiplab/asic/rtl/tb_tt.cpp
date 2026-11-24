// Verilator bench for the Tiny Tapeout top level: drives the SPI host port bit by bit.
// Script on stdin: "W addr data", "R addr" (prints "R addr value"), "C n", "U" (print uio_out bit0 trace as "U cycle level" on change).
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include "Vtt_um_chiplab_pem.h"
#include "verilated.h"

static Vtt_um_chiplab_pem* t;
static unsigned long cyc = 0;
static int last_u = -1, trace = 0;
static void clk(int n = 1) {
    for (int i = 0; i < n; i++) {
        t->clk = 0; t->eval(); t->clk = 1; t->eval(); cyc++;
        int u = t->uio_out & 1;
        if (trace && u != last_u) { printf("U %lx %d\n", cyc, u); last_u = u; }
    }
}
static const int HALF = 4;   // SPI half period in system clocks
static unsigned xfer(unsigned bits, int n) {
    unsigned in = 0;
    for (int k = n - 1; k >= 0; k--) {
        t->ui_in = (t->ui_in & ~0b101) | (((bits >> k) & 1) << 2); clk(HALF);
        t->ui_in |= 1; in = (in << 1) | (t->uo_out & 1); clk(HALF);
    }
    t->ui_in &= ~1;
    return in;
}
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    t = new Vtt_um_chiplab_pem;
    t->ena = 1; t->ui_in = 0b10; t->uio_in = 0; t->rst_n = 0; clk(5); t->rst_n = 1; clk(5);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream ss(line); std::string c; if (!(ss >> c)) continue;
        if (c == "W") { std::string a, d; ss >> a >> d;
            t->ui_in &= ~0b10; clk(HALF); xfer(std::stoul(a, 0, 16) & 0x7F, 8); xfer(std::stoul(d, 0, 16), 16);
            clk(HALF); t->ui_in |= 0b10; clk(4 * HALF); }
        else if (c == "R") { std::string a; ss >> a; unsigned ad = std::stoul(a, 0, 16) & 0x7F;
            t->ui_in &= ~0b10; clk(HALF); xfer(0x80 | ad, 8); unsigned v = xfer(0, 16);
            clk(HALF); t->ui_in |= 0b10; clk(4 * HALF); printf("R %x %x\n", ad, v); }
        else if (c == "C") { long n; ss >> n; clk(n); }
        else if (c == "U") { trace = 1; }
    }
    t->final(); delete t; return 0;
}
