cocotb test of the Tiny Tapeout top level through the SPI host port (template layout:
test.py, tb.v; set PROJECT_SOURCES in the template's Makefile to the src files).
ChipLab replays the same SPI writes in Icarus Verilog before export (eda.icarus_tt_test);
the template's GitHub Actions run this cocotb test on RTL and on the gate-level netlist.
