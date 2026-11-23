# ChipLab: Heterogeneous Accelerator Design Space Exploration

ChipLab is a framework for designing and simulating heterogeneous accelerator architectures, including chiplet-based designs with memory, compute, and interconnect components.

## Architecture Overview

ChipLab consists of the following major components:

### Schema Layer (`src/chiplab/schemas/`)
- **DesignSpec**: Heterogeneous hardware design specification with support for chiplets, memory specs, compute specs, packages, and interconnects
- **Common**: Provenance tracking (`Provenance` enum: measured, published, estimated, assumed) and value unwrapping (`unwrap_value`, `get_provenance`)
- **Simulation**: Simulation configuration, metrics, bottleneck reports, and comparison results

### Engine Layer (`src/chiplab/engine/`)
- **Analytical**: Level 1 roofline model simulation with fidelity `FidelityLevel.ANALYTICAL`
- **Discrete-Event**: Level 2 SimPy-based simulation with fidelity `FidelityLevel.DISCRETE_EVENT`
- **Sweep**: Parameter sweep over design/workload parameters
- **Analyzer**: Bottleneck analysis and result comparison

### Models Layer (`src/chiplab/models/`)
- Physical models for memory, compute, interconnect, packaging, and power/thermal estimation
- Component library with default values and assumption tracking

### Results Layer (`src/chiplab/results/`)
- Report export in Markdown, JSON, and CSV formats
- Simulation result, comparison, and sweep reporting

### Experiments Layer (`src/chiplab/experiments/`)
- SQLite-backed persistence for designs, simulations, sweeps, and comparisons
- Design versioning support (parent_version tracking)
- Save/get operations for all result types

### Backends (`src/chiplab/backends/`)
- Scale simulation backend
- Ramulator memory simulator
- Accelerator simulator

### YAML Design Files
ChipLab uses YAML design specifications that can be loaded via `DesignSpec.from_yaml()`:
- `projects/echo/echo_v0_1.yaml`: Baseline DRAM-PIM + digital compute design
- `projects/echo/echo_v0_2.yaml`: Updated version with HBM3 memory option
- `examples/transformer_accelerator.yaml`: Generic HBM + systolic array architecture

## Installation

### Using uv (recommended)
```bash
uv venv
source .venv/bin/activate
uv pip install chiplab
```

### Using pip
```bash
pip install chiplab
```

Or install from source:
```bash
git clone https://github.com/your-repo/chilab.git
cd chilab
python -m pip install -e .
```

## MCP Setup (Claude Code, Codex, Gemini CLI, ChatGPT)

The same server works in every MCP client. Local clients start it over stdio; ChatGPT on the
web needs it over HTTP behind a public HTTPS URL. Set `CHIPLAB_DB_PATH` so every client
shares one design database (otherwise it is created in the client's working directory).

### Claude Code
```bash
claude mcp add --scope user chiplab -e CHIPLAB_DB_PATH="/home/everett/Chip Lab/chiplab_experiments.db" -- "/home/everett/Chip Lab/.venv/bin/chiplab-mcp"
```

### Codex CLI / Codex IDE extension
```bash
codex mcp add chiplab --env CHIPLAB_DB_PATH="/home/everett/Chip Lab/chiplab_experiments.db" -- "/home/everett/Chip Lab/.venv/bin/chiplab-mcp"
```
Or add this to `~/.codex/config.toml`:
```toml
[mcp_servers.chiplab]
command = "/home/everett/Chip Lab/.venv/bin/chiplab-mcp"
env = { CHIPLAB_DB_PATH = "/home/everett/Chip Lab/chiplab_experiments.db" }
```

### Gemini CLI
Add to `~/.gemini/settings.json`:
```json
{
  "mcpServers": {
    "chiplab": {
      "command": "/home/everett/Chip Lab/.venv/bin/chiplab-mcp",
      "env": { "CHIPLAB_DB_PATH": "/home/everett/Chip Lab/chiplab_experiments.db" }
    }
  }
}
```

### ChatGPT (web/desktop, Developer mode connector)
1. Run the server over HTTP:
   ```bash
   "/home/everett/Chip Lab/.venv/bin/chiplab-mcp" --transport streamable-http --port 8000
   ```
2. Expose it over HTTPS with a tunnel, e.g. `cloudflared tunnel --url http://localhost:8000`,
   then restart step 1 with `--allowed-host <your-tunnel-host>` so the host check accepts it.
3. In ChatGPT: Settings -> Apps & Connectors -> Advanced -> Developer mode, then create a
   connector with URL `https://<your-tunnel-host>/mcp`.

The HTTP server has no login. Anyone with the URL can use it, so only run the tunnel while
you are using it. `search` and `fetch` tools are included for ChatGPT's connector format.

### Codex cloud (chatgpt.com/codex)
Cloud tasks run in a container built from the repository and cannot reach a server on your
computer. Install the package in the environment's setup script (`pip install -e .`) and
call it from Python, e.g. `from chiplab.mcp_server import chip_evaluate`.

## Design and test a chip in three calls

1. `chip_quickstart`: the workflow, templates, TSMC nodes, memory and package types.
2. `chip_design_from_template("llm_accelerator_hbm3e", overrides={...})` or `chip_create_design(spec)`.
3. `chip_evaluate(design, workload={"type": "llm_decode", "model_params_b": 70, ...})`: the
   performance (roofline), power and thermal, a unit-cost range, $/TOPS, TOPS/W,
   tokens/s per $1k, pass/fail against constraints, and realism warnings (reticle limit,
   SRAM that doesn't fit, HBM on a package that can't host it, interposer too large).

Templates: `edge_npu` (N4 + LPDDR5X, FCBGA), `llm_accelerator_hbm3e` (2x N3 + 8x HBM3E,
CoWoS-L), `chiplet_cpu` (8x N3 CCD + N6 IOD, organic).

### Cost model (`src/chiplab/cost/`)
- **Logic dies**: TSMC wafer price per node (N40 down to A16), dies per wafer, and
  negative-binomial yield from the node's defect density. SRAM area comes from the
  published HD bitcell size and array efficiency. Mask-set NRE is spread over the volume.
- **Memory**: DDR4/5, LPDDR5/5X, GDDR6/7, HBM2E/3/3E/4 priced in $/GB.
- **Packaging**: FCBGA, organic, EMIB, CoWoS-S, CoWoS-L, SoIC. Cost is per mm² plus a
  per-die attach cost, and the assembly yield loss scraps every die in a failed package.
- **Test and board**: wafer sort, final test, and optionally the PCB (area, layers, HDI,
  high-speed material), VRM and cooling, which scale with watts.
- **Scenarios**: every number has a low/high range, so each result is a typical,
  optimistic and pessimistic cost.

### Live prices
Foundries and memory makers don't publish prices, so the built-in figures are dated
industry estimates. Each one shows its source and date in `chip_cost_references`, and any
figure older than 180 days is marked stale. To refresh them:
- **With web search** (Claude, ChatGPT, Gemini, Codex): call `chip_price_lookup_plan`
  to get one search query per stale price. Search each one, then call
  `chip_set_price(key, value, source_url, as_of)`.
- **From a feed**: set `CHIPLAB_PRICE_FEED_URL` to an https JSON file shaped like
  `{key: {value, source, as_of}}` and call `chip_refresh_prices`.

Overrides are stored in `~/.chiplab/prices.json` (or `CHIPLAB_PRICE_FILE`) and are shared by
every client. `chip_clear_price` restores the built-in value.

## ChipLab Live (watch agents design in real time)

```bash
cd "/home/everett/Chip Lab" && .venv/bin/chiplab-app --open
```
This opens http://127.0.0.1:8770. Every MCP tool call from every connected agent (Claude Code,
Codex, Gemini CLI, ChatGPT) is written to the shared database and streams into the page:
- **Agent activity**: a live feed with the client name, the tool, the design and a one-line
  result (cost, tok/s, TOPS, watts, failed constraints).
- **Designs**: the package floorplan (dies to scale, HBM stacks, interposer), headline
  numbers, the cost breakdown with its optimistic–pessimistic range, per-chiplet yield
  and cost, and warnings.
- **Digital twin**: a scatter of every explored design (any metric on x/y, coloured by
  node, Pareto front or generation), the top-K robust designs, and sensitivity bars for the
  unknown parameters.
- **Lab · simulate**: run things yourself, no agent needed.
  - *On-chip memory simulator*: pick SRAM or 2T/3T gain-cell eDRAM, a PDK node, banks,
    interleaving, ECC, requesters, temperature, PIM lanes and the traffic pattern. Run it
    and watch requesters and banks cycle by cycle (grants, bank-conflict stalls, refresh
    stalls), with play, pause and step controls. You also get model-vs-simulation numbers
    and the throughput/stall timeline. **Generate RTL** shows the SystemVerilog, testbench
    and SDC with lint status, and lets you download each file.
  - *Design-point simulator*: drag node (GF180 … 2 nm), cores, clock, supply, caches and
    the memory knobs. PPA, cost, slack, temperature, refresh and PIM throughput update
    live, with the node's accuracy band. "Search around this point" runs NSGA-II plus
    Monte Carlo and opens the result in the Digital twin tab.
  - *Simulate this design* (Designs tab): run an agent's design through the analytical
    or discrete-event engine with an LLM workload. Results are not saved.
"Follow live" jumps to whatever an agent is working on. Use `--db` to point at another
database (default: `$CHIPLAB_DB_PATH` or `./chiplab_experiments.db`).

## Zero-token digital twin

`run_digital_twin_search` explores thousands of RISC-V SoC design points locally (cores,
VLEN/DLEN, L1/L2/LLC, Vdd/frequency, TSMC N16–N3 or the full 180 nm–2 nm PDK ladder, UCIe lanes,
placement utilization, optional on-chip memory knobs) with
NSGA-II, Latin hypercube or Gaussian-process-guided sampling. It then runs Monte Carlo over
every parameter whose provenance is `estimated` or `assumed` (wire parasitics, gate delay,
leakage, switched capacitance, memory stalls, thermal resistance, wafer price, D0;
±15–25%). The agent gets back a few KB: the top-K Pareto designs with slack yield,
thermal-violation probability and a robustness score, plus the unknowns ranked by impact
(|Spearman ρ|). The full trace is saved in the database.

- Raw model speed: about 400–490k designs/s with numpy, about 210k/s in pure Python.
  A full NSGA-II run with Monte Carlo achieves about 5–7k model evaluations/s; Latin
  hypercube about 27k/s.
- **Calibration**: `ingest_byo_pdk` parses Liberty (FO4 from the smallest inverter's delay
  table, leakage density) and LEF (layers, pitches, macros → black-box candidates).
  `calibrate_from_eda_report` parses OpenSTA/PrimeTime timing reports (cell vs wire
  delay) and SPEF. The resulting factors are marked `measured` and are no longer
  perturbed. File inputs must sit under `CHIPLAB_DATA_DIRS` (default: cwd and home).
- **IP**: `list_ip_catalog` covers CVA6, Ara, Rocket, XiangShan, Ibex, OpenRAM, BaseJump
  STL, PULP AXI and TileLink, and notes that no silicon-proven open UCIe PHY exists.
  `BlackBoxMacro` plugs in commercial PHYs and controllers from datasheet numbers: pin
  timing, area, keep-out zones and power rails.
- **Handoff**: `export_pareto_handoff_bundle` writes a SystemVerilog wrapper (stub tiles +
  black boxes), SDC, UPF, OpenROAD and Synopsys DC TCL, Verilator-lint and Yosys-equivalence
  scripts and a sha256 manifest to `CHIPLAB_EXPORT_DIR`. Nothing is executed. MCP clients
  get no shell, no eval/exec, and seeds are deterministic.

## PDK ladder (180 nm to 2 nm)

`list_pdk_ladder` and `docs/sources_and_accuracy.md` give every number with its provenance
and citation (regenerate with `python -m chiplab.library.pdk_library`). The accuracy margin
is applied automatically to the technology factors in twin Monte Carlo.

| Node | Vdd | FO4 ps | kgate/mm² | 6T µm² | Tape-out | Margin | Licence | Commercial |
|---|---|---|---|---|---|---|---|---|
| GF180MCU | 3.3 | 95 | 60 | 5.0 | yes | ±5% | Apache-2.0 | yes |
| SkyWater sky130 | 1.8 | 45 | 160 | 2.0 | yes | ±5% | Apache-2.0 | yes |
| IHP SG13G2 | 1.2 | 40 | 150 | 2.2 | yes | ±5% | Apache-2.0 | yes |
| 65 nm (literature) | 1.0 | 17 | 600 | 0.52 | no | ±10% | literature | no |
| 28 nm (predictive) | 0.9 | 11 | 2,000 | 0.127 | no | ±15% | unverified | no |
| FreePDK15 / NanGate15 | 0.8 | 8 | 6,000 | 0.07 | no | ±15% | academic | no |
| ASAP7 | 0.7 | 6 | 10,000 | 0.027 | no | ±15% | BSD-3-Clause | yes |
| 3 nm (literature) | 0.72 | 4.8 | 28,000 | 0.0199 | no | ±20% | literature | no |
| 2 nm (literature) | 0.68 | 4.2 | 36,000 | 0.0175 | no | ±25% | literature | no |

Validation warns when a design targets a predictive PDK, or uses non-commercial IP or
black boxes, with `metadata.intended_use = "commercial"`.

## On-chip memory subsystems and RTL

`design_onchip_memory_subsystem` models multi-bank 6T SRAM and 2T/3T gain-cell eDRAM on any
ladder node. It reports:
- area, fmax and slack
- retention (halving every 18 °C), refresh overhead and stall %
- bank-conflict rate and port contention
- SECDED overhead, pJ/bit, and near-bank PIM GMAC/s

A deterministic cycle-accurate simulator cross-checks the analytical model: conflict rate
agrees within 0.06 absolute and throughput within 12% up to saturation, and the tests
enforce this. With `generate_rtl=true` it writes SystemVerilog-2012 to `CHIPLAB_EXPORT_DIR`:
- `onchip_mem_subsystem_top.sv`: crossbar, interleaver (low-order, XOR or block),
  round-robin or priority arbiter per bank, auto-refresh controller with a deadline
  assertion, SECDED encoder/decoder (XOR trees generated from the same tables as the
  Python reference codec), PIM dot-product/MAC, and an optional AXI4-Lite bridge
- `behavioral_bitcell_macro.sv`: a simulation model with retention assertions. Use
  OpenRAM or SPICE for the real macro.
- `tb_onchip_mem.sv`: a self-checking testbench (scoreboard, ECC error injection, PIM
  check, idle-past-retention refresh check)
- SDC constraints and `run_verilator.sh`

Verified with Verilator 5.032 (`tests/test_rtl_verilator.py`, skipped if Verilator is not
installed):
- `verilator --lint-only -Wall` is clean, and the self-checking testbench passes across
  SRAM and eDRAM, 2–16 banks, all interleave and arbiter modes, ECC on and off, and
  PIM 4/8-bit.
- Four deliberately injected bugs (ECC decoder that never corrects, disabled refresh,
  PIM accumulate, address aliasing) all make the testbench fail.
- The AXI4-Lite bridge is linted but not driven by the testbench.

Run `bash run_verilator.sh` in an exported bundle to reproduce.

The twin takes the same knobs: `run_digital_twin_search(node_set="ladder",
memory={"capacity_kib": 2048})` searches SRAM vs eDRAM, banks, interleaving and PIM lanes
across the ladder. It adds a refresh-violation probability to the robustness score and
±25% retention and ±20% traffic priors.

## Private (NDA) PDKs

`ingest_byo_pdk` and `calibrate_from_eda_report` default to `private=True`:
- The compiled profile (numbers and calibration factors only) is saved to
  `.chiplab_private_pdks/`, override with `CHIPLAB_PRIVATE_PDK_DIR`. The folder is
  created 0700 with its own `.gitignore` (`*`) and is appended to the project's
  `.gitignore`. Profiles are mode 0600 and named by an opaque `pdk_<hash>` id.
- Library, cell, macro, pin, layer and site names, timing-path instance names and file
  paths are redacted from MCP responses, the stored calibration and error messages.
  Macros become `macro_1`, `macro_2`, …
- The activity log never stores `*_path` arguments or file contents.

`list_private_pdks` lists the saved profiles. Pass `private=false` for open PDKs to keep
the names.

## Example Workflow

### 1. Load a Design Spec
```python
from chiplab.schemas.design import DesignSpec

with open("projects/echo/echo_v0_1.yaml") as f:
    data = yaml.safe_load(f)

design = DesignSpec.from_yaml(yaml.dump(data))
```

### 2. Validate the Design
```python
from chiplab.engine.validator import validate_design

result = validate_design(design, allow_estimated_defaults=True)
if result.valid:
    print("Design is valid!")
else:
    print(f"Errors: {result.errors}")
```

### 3. Run an Analytical Simulation
```python
from chiplab.schemas.workload import WorkloadSpec
from chiplab.schemas.simulation import SimulationConfig, FidelityLevel
from chiplab.engine.analytical import run_analytical_simulation

workload = WorkloadSpec(
    name="moe_decode_workload",
    type="moe_decode",
    batch_size=1,
    sequence_length=128,
    hidden_dim=4096,
    num_layers=32,
    num_experts=8,
    active_experts=2,
    precision="INT8",
    flops=1.5e15,
    bytes_read=64e9,
    bytes_write=16e9,
)

config = SimulationConfig(fidelity=FidelityLevel.ANALYTICAL)
result = run_analytical_simulation(design, workload, config, design_id="echo_v0_1")

print(f"Fidelity: {result.fidelity}")
print(f"Throughput: {result.metrics.throughput_tokens_per_sec:.1f} tokens/sec")
print(f"Bottleneck: {result.bottleneck.primary_bottleneck}")
```

### 4. Run a Discrete-Event Simulation
```python
from chiplab.engine.discrete_event import run_discrete_event_simulation

config = SimulationConfig(
    fidelity=FidelityLevel.DISCRETE_EVENT,
    seed=42,
    num_requests=32,
)

result = run_discrete_event_simulation(design, workload, config)
print(f"Fidelity: {result.fidelity}")
print(f"Latency: {result.metrics.estimated_latency_ms:.2f} ms")
```

### 5. Save Results to Experiment Store
```python
from chiplab.experiments.store import ExperimentStore
import tempfile

with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
    db_path = tmp.name

store = ExperimentStore(db_path=db_path)

# Save design
design_id = store.save_design(design)

# Save simulation
sim_id = store.save_simulation(result)

# Retrieve later
retrieved_design_id, retrieved_design = store.get_design(design_id)
retrieved_sim = store.get_simulation(sim_id)
```

### 6. Export Reports
```python
from chiplab.results.reporter import export_simulation_report

markdown_report = export_simulation_report(result, fmt="markdown")
with open("report.md", "w") as f:
    f.write(markdown_report)

json_report = export_simulation_report(result, fmt="json")
with open("report.json", "w") as f:
    f.write(json_report)

csv_report = export_simulation_report(result, fmt="csv")
with open("report.csv", "w") as f:
    f.write(csv_report)
```

## YAML Design Files

### echo_v0_1.yaml
Heterogeneous DRAM-PIM + digital compute chiplet design for MoE decode baseline version.

### echo_v0_2.yaml
Updated version with improved PIM efficiency and HBM3 memory option, parent_version: "0.1".

### transformer_accelerator.yaml
Generic architecture demonstrating ChipLab is not tied to ECHO designs: HBM3 + systolic array tensor engine.
