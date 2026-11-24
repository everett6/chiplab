// Verilator bench for pem_core. Reads a bench script on stdin (same language as
// chiplab/asic/iss.py run_script) and prints the same event lines, so the RTL and the
// Python reference model can be compared line by line.
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "Vpem_core.h"
#include "verilated.h"
#if VM_COVERAGE
#include "verilated_cov.h"
#endif

static Vpem_core* top;
static unsigned long cycle = 0;
static unsigned ext = 0, raw = 0;
static std::vector<std::pair<int, int>> loops;

struct I2CTarget {
    bool on = false;
    int addr = 0, scl_p = 0, sda_p = 0;
    std::string state = "idle";
    int bit = 0, sh = 0, rw = 0, byte = 0xA5, pull = 0, pscl = 1, psda = 1;
    bool ack = false;
    void step(int scl, int sda) {
        int ps = pscl, pd = psda;
        if (scl && ps && pd && !sda) { state = "addr"; bit = 0; sh = 0; pull = 0; printf("TS\n"); }
        else if (scl && ps && !pd && sda) { state = "idle"; pull = 0; printf("TP\n"); }
        else if (!ps && scl) {
            if (state == "addr" || state == "wdata") { sh = ((sh << 1) | sda) & 0xFF; bit++; }
            else if (state == "rack") { printf("TR %x %d\n", byte, !sda); ack = !sda; }
        } else if (ps && !scl) {
            if ((state == "addr" || state == "wdata") && bit == 8) {
                if (state == "addr") {
                    if ((sh >> 1) == addr) { rw = sh & 1; printf("TA %x %d\n", addr, rw); pull = 1; state = "acka"; }
                    else state = "idle";
                } else { printf("TW %x\n", sh); pull = 1; state = "ackw"; }
            } else if (state == "acka") {
                if (rw) { state = "rdata"; bit = 0; pull = !((byte >> 7) & 1); }
                else { pull = 0; state = "wdata"; bit = 0; sh = 0; }
            } else if (state == "ackw") { pull = 0; state = "wdata"; bit = 0; sh = 0; }
            else if (state == "rdata") {
                bit++;
                if (bit < 8) pull = !((byte >> (7 - bit)) & 1); else { pull = 0; state = "rack"; }
            } else if (state == "rack") {
                if (ack) { byte = (byte + 1) & 0xFF; state = "rdata"; bit = 0; pull = !((byte >> 7) & 1); }
                else { state = "idle"; pull = 0; }
            }
        }
        pscl = scl; psda = sda;
    }
} i2c;

static unsigned env() {
    unsigned uo = top->out_val & 0xFF, oe = top->oe, r = ext;
    for (auto& p : loops) {
        int src = p.first, dst = p.second;
        unsigned drv = (src >= 8 || ((oe >> src) & 1)) ? ((top->out_val >> src) & 1) : 1;
        r = (r & ~(1u << dst)) | (drv << dst);
    }
    if (i2c.on) {
        int scl = (((oe >> i2c.scl_p) & 1) && !((uo >> i2c.scl_p) & 1)) ? 0 : 1;
        int drv_sda = ((oe >> i2c.sda_p) & 1) && !((uo >> i2c.sda_p) & 1);
        int sda = (drv_sda || i2c.pull) ? 0 : 1;
        i2c.step(scl, sda);
        sda = (drv_sda || i2c.pull) ? 0 : 1;
        r = (r & ~((1u << i2c.scl_p) | (1u << i2c.sda_p))) | (scl << i2c.scl_p) | (sda << i2c.sda_p);
    }
    return r;
}

static unsigned last_o[3];
static void outputs(unsigned o[3]) {
    o[0] = top->out_val & 0xFF;
    o[1] = top->oe;
    o[2] = (((top->out_val >> 8) & 0x3F) << 2) | (((top->irq >> 7) & 1) << 1);
}

static unsigned tick(int we, int re, unsigned addr, unsigned wdata) {
    top->bus_we = we; top->bus_re = re; top->bus_addr = addr; top->bus_wdata = wdata;
    top->raw_in = raw & 0x1FFF;
    top->clk = 0; top->eval();
    unsigned r = top->bus_rdata;
    top->clk = 1; top->eval();
    cycle++;
    top->clk = 0; top->bus_we = 0; top->bus_re = 0; top->eval();
    unsigned o[3];
    outputs(o);
    if (o[0] != last_o[0] || o[1] != last_o[1] || o[2] != last_o[2]) {
        printf("P %lx %x %x %x\n", cycle, o[0], o[1], o[2]);
        last_o[0] = o[0]; last_o[1] = o[1]; last_o[2] = o[2];
    }
    raw = env();
    return r;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    top = new Vpem_core;
    top->rst_n = 0;
    for (int i = 0; i < 3; i++) { top->clk = 0; top->eval(); top->clk = 1; top->eval(); }
    top->clk = 0; top->rst_n = 1; top->eval();
    outputs(last_o);
    raw = env();
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream ss(line);
        std::string t;
        if (!(ss >> t)) continue;
        if (t == "W") { std::string a, d; ss >> a >> d; tick(1, 0, std::stoul(a, 0, 16), std::stoul(d, 0, 16)); }
        else if (t == "R") { std::string a; ss >> a; unsigned ad = std::stoul(a, 0, 16); unsigned v = tick(0, 1, ad, 0); printf("R %x %x\n", ad, v); }
        else if (t == "C") { long n; ss >> n; for (long i = 0; i < n; i++) tick(0, 0, 0, 0); }
        else if (t == "S") { std::string v; ss >> v; ext = std::stoul(v, 0, 16); raw = env(); }
        else if (t == "LB") { int s, d; ss >> s >> d; loops.push_back({s, d}); raw = env(); }
        else if (t == "I2C") { std::string a; ss >> i2c.scl_p >> i2c.sda_p >> a; i2c.addr = std::stoul(a, 0, 16); i2c.on = true; raw = env(); }
        else if (t == "Q") break;
    }
    fflush(stdout);
#if VM_COVERAGE
    const char* cov = getenv("PEM_COVERAGE_FILE");
    VerilatedCov::write(cov ? cov : "coverage.dat");
#endif
    top->final();
    delete top;
    return 0;
}
