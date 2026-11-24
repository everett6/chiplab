# Verification methodology

1. **Independent reference model** (`chiplab/asic/iss.py`): a cycle-accurate Python model
   written from the ISA document, not from the RTL.
2. **Lockstep comparison**: the same bench script runs on the Verilator RTL bench and on the
   model; every pin change, register read and bench event must match exactly.
3. **Constrained-random programs**: legal random instruction streams and structured serial
   loops (line coder, stuffing, CRC, autopull) with random lane configuration, host traffic,
   input stimulus and pin loopback.
4. **Directed protocol tests with independent decoders**: UART, SPI and USB decoders sample
   the pin waveform; the I2C target is a separate bench model; none share code with the RTL
   or the model's datapath.
5. **Mutation testing**: deliberate bugs are injected into copies of the RTL; the suite must
   catch each one. This measures the tests, not the design.
6. **Assertions**: immediate assertions on FIFO bounds compiled in with `--assert`.
7. **Coverage**: functional bins (opcodes, jump conditions taken/not taken, stalls, line-coder
   modes, stuffing events) from the model; line/toggle coverage from Verilator.
8. **Formal (bounded)**: SymbiYosys BMC of assertions A1-A4 from reset with every bus and pin
   input unconstrained. A pass means no input sequence violates them within the checked depth;
   it is not an unbounded proof. A deliberately false property is found in a few cycles, so
   the setup can fail.
9. **Gate level**: the Yosys netlist (generic gates, zero delay) runs the same random scripts
   and directed protocol tests and must match the model event for event.

Passing self-generated tests is evidence, not proof: the model and RTL share one author and one
reading of the ISA. Not yet done: unbounded (k-induction) proofs, timing-annotated gate-level
simulation with IHP cells, cocotb run of the Tiny Tapeout top level, and directed tests for
JTAG, SWD, PS/2, CAN and USB receive.
