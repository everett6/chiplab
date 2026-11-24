# Spec coverage matrix

This maps every item in the three briefs (the Jane Street protocol emulator competition, the ChipLab
universal platform vision, and the Universal ASIC and Memory Architecture addendum) to what exists.

Status key:
- **done**: built and covered by tests.
- **partial**: works, but with a stated gap.
- **missing**: not built yet.
- **blocked**: needs a tool or download that isn't installed (they need your OK).

Last updated 2026-11-23.

## A. Jane Street protocol emulator ASIC

| Item | Status | Evidence |
|---|---|---|
| IHP 130 nm, Tiny Tapeout flow, ≤ 8×4 tiles, deadline 2027-01-18 | partial | `asic/flow.py` `COMPETITION`. Process and deadline come from the brief and are marked unverified. Tile outlines are the official ones, and the tile fit is estimated (see R8 below). |
| Reprogrammable engine, not fixed peripherals | done | PEM-16 ISA in `asic/isa.py`: 4 lanes, shared 32-word imem, NRZI, bit stuffing and CRC datapath. RTL in `asic/rtl/*.v`. |
| UART | done | `asic/protocols.py` `uart_tx`/`uart_rx`. Directed tests run on the RTL and the model, with an independent decoder. |
| SPI | done | `spi_controller(mode)` for modes 0–3, with a loopback test. |
| I2C | done | `i2c_controller`, a bench target model, and an ACK/NACK test. |
| Low-speed USB (stretch) | partial | TX only: NRZI, bit stuffing, CRC16 and SYNC/EOP, decoded independently. RX is not written. |
| 10 Mbit Ethernet (stretch) | missing | Listed in `PLANNED`. A 20 MHz Manchester stream needs a faster clock than this design targets. |
| JTAG, SWD, PS/2, CAN | missing | Listed in `PLANNED` with notes. |
| Architecture studio | done | `/asic` Architecture tab: ISA document, lane datapath, register-bit count. |
| RTL verification studio | done | Lockstep constrained-random regression, functional and line/toggle coverage, assertions A1–A4, 6/6 mutants killed, lint, and an SPI host smoke test of the TT top. |
| Formal verification | partial | `asic/eda.py`: SymbiYosys BMC of A1–A4 at depth 20 passes with every input free. A deliberately false property fails, so the check can catch bugs. There is no unbounded (k-induction) proof yet. |
| Gate-level simulation | done | The Yosys netlist runs under Verilator. 40/40 random seeds and 11/11 protocol tests match the model. Zero delay, no SDF timing. |
| Synthesis | done | Generic Yosys: 40,340 cells, 2,014 flops. Mapped to IHP sg13g2 cells (slow corner): 337,861 µm² of cells, 30,319 cells, 7.97 ns longest logic path (126 MHz upper bound). |
| Tile fit (R8) | partial | 28% cell density in the official 8×4 outline (1724.16 × 710.64 µm, from tt-support-tools), so it is likely to fit. 5×4 and 6×4 are also likely. Placement and routing are not run, so this is an estimate. |
| Physical design studio | partial | Flow steps, tool status and the TT project export (`projects/jane_street_pem`) are in place. Hardening, GDS, timing, area and power are blocked (they need the TT IHP flow or the GitHub Actions template). |
| Protocol lab | done | `/asic` Protocol Lab: editor, assembler, reference-model simulation with a pin waveform, protocol tests on the model or RTL. |
| Competition evaluation | done | Metrics (physical metrics show "not run", never invented), requirements R1–R11, checklist, limitations, technical report. |
| AI assistance with versioning and no fabricated results | done | MCP tools `asic_*` (16 tools in this group). Results are stored with an RTL hash and marked stale when the RTL changes (`asic/report.py`). |
| Submission checklist and technical report | done | `report.checklist()`, `/api/asic/technical_report`. |
| Shares ChipLab infrastructure | done | Same app server, MCP server, PDK ladder (memory compiler) and tool registry. |

## B. ChipLab universal platform (sections 1–22)

| Item | Status | Evidence |
|---|---|---|
| Audit and keep existing work | done | No existing feature was removed. The full suite still passes. |
| GPU Studio modes | partial | 7 of the 10 modes exist (`studio.js` `MODES`): assembly, die, memory, thermal, power, interconnect, airflow. The Manufacturing Explorer and two others are missing. |
| Realistic die view | done | `layout3d.js`: cell texture, met1–met5 stack, straps and pads, plus the GDS view through KLayout. |
| React Three Fiber | partial | Vanilla three.js is used instead. The behaviour is the same and there is no build step. |
| Architecture Foundry | partial | Component Foundry: hierarchy, roll-up, functional simulation and Verilog export. There is no drag-and-drop canvas editor. |
| Component SDK with sandbox | done | `sdk/expr.py` uses an AST whitelist with length and size bounds, and has no eval/exec. Versions are immutable (`sdk/components.py`). |
| Multi-fidelity simulation | done | Roofline, discrete-event and eCore backends, plus `simulation_levels` (10 levels, each labelled with its fidelity). |
| Event-driven physical-to-logical engine | partial | Discrete-event engine plus eCore. Device physics is not coupled to logic events. |
| Memory Studio | done | Memory compiler plus `/asic` Memory Studio and the GPU Studio memory mode. |
| Circuit, PCB and physics studios | partial | ngspice, KiCad and CFD through `tools/` and `toolviews.js`. OpenFOAM is absent and there is no mechanical FEA. |
| Optimization and design-space search | done | Sweeps, digital twin search, Pareto handoff. |
| Workloads | done | Templates and workload models in the analytical and discrete-event engines. |
| Manufacturing (cost, yield) | done | `cost/`: wafer price, yield, packaging, test and NRE. |
| Digital twin | done | `engine/digital_twin.py`, `/live`. |
| Typed MCP tools | partial | chip_*, asic_*, memory_*, component_* and architecture_* exist. The spec's pcb_*, thermal_*, simulation_*, experiment_*, manufacturing_* and circuit_* names do not yet exist as separate tools; the same work runs through chip_evaluate and the tools API. |
| Experiment branching | partial | The design store keeps `parent_version`. There is no branch tree UI or merge. |
| Provenance | done | Every price, parameter and result carries a source, date, fidelity or "assumed" label. |
| Development rules (no silent edits, no fake results) | done | Registry versions are immutable. Steps that didn't run are reported as "not run". |

## C. Universal ASIC and memory addendum

| Item | Status | Evidence |
|---|---|---|
| A. Memory types: mask ROM, SRAM, DRAM, eDRAM, register file, FIFO | done | `memory_compiler/compiler.py` (11 cell types). |
| A. Custom arrays, controllers, processing-in-memory | partial | A CIM bitcell template exists in the SDK. There is no PIM compiler. |
| A. Configurable capacity, width, ports, banks, timing, power, area | done | `MemSpec` plus the candidate search over banks, column mux and decoder. |
| A. Mask ROM: contents, organization, decoder, cell, layout view | done | Logical, circuit and physical views (`views.py`). The site map is tested against the contents. |
| A. Keep logical, circuit and physical views separate | done | Three separate views, all generated from the same database. |
| B. Custom ASIC studio, hierarchical | done | Component SDK plus Foundry (see B above). |
| C. Compiler outputs: org, circuit, area, timing, power, RTL, tests, layout | partial | All are produced. The layout is a schematic floorplan, not GDS. |
| C. PDK validation | partial | Rule checks against the PDK ladder. DRC and LVS are reported as "not run" (they need a real layout). |
| D. 10 simulation levels with fidelity labels | partial | Levels 1–5, 9 and 10 run here. Levels 6–8 (STA, place and route, DRC/LVS) need OpenROAD, KLayout and the IHP PDK. |
| E. Architecture comparison | done | `memory_compare` and the Compare button. |
| F. Visual views tied to the data | done | All SVGs are generated from the compile result. |
| G. Extensibility | done | User components are registered as JSON with sandboxed models. |

## What unblocks the most

1. Done 2026-11-23: OSS CAD Suite is installed in `~/.local/opt/oss-cad-suite`.
2. Done 2026-11-23: the IHP sg13g2 liberty (typ and slow corners) is in `~/.local/opt/ihp-sg13g2`, pinned to IHP-Open-PDK commit e73adb40.
3. Run the TT IHP hardening flow. Either install it locally, or push `projects/jane_street_pem` to a repo made from the official Tiny Tapeout IHP template so GitHub Actions does it. This gives GDS, timing, area and precheck.
4. Write the USB RX and Ethernet protocol programs and their tests.
