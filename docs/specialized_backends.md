# Specialized backends: Ramulator 2 and SCALE-Sim

`chip_run_simulation(..., fidelity="specialized")` (and sweeps at that fidelity) use two
third-party simulators when they are installed. Both are optional: without them the run
falls back to the analytical model and is labeled `analytical`, and their tests skip.

## How ChipLab calls them
Each tool lives in its own Python environment. ChipLab writes a small JSON request to a
temporary directory and runs

    <tool python> -I src/chiplab/backends/_<tool>_runner.py request.json result.json

as an argument list (`shell=False`). The runner validates every field (fixed tables and
bounded integers), runs the tool and writes JSON back. MCP clients cannot choose the
interpreter, the runner or any path: the interpreter comes from an environment variable set
by whoever runs the server.

| Tool | Env var | Default interpreter |
|---|---|---|
| Ramulator 2 | `CHIPLAB_RAMULATOR_PYTHON` | `~/.local/opt/ramulator-venv/bin/python` |
| SCALE-Sim 3 | `CHIPLAB_SCALESIM_PYTHON` | `~/.local/opt/scalesim-venv/bin/python` |

`backend_status()` (also in `chiplab://models/assumptions`) shows whether each is usable,
with the reason when not. Results are cached per process (same inputs, same numbers).

## Install
Ramulator 2 (C++ with Python bindings; needs CMake, a C++20 compiler):

```bash
git clone https://github.com/CMU-SAFARI/ramulator2 ~/.local/src/ramulator2
python3 -m venv ~/.local/opt/ramulator-venv
~/.local/opt/ramulator-venv/bin/pip install -e ~/.local/src/ramulator2
~/.local/opt/ramulator-venv/bin/python -c "import ramulator"
```

SCALE-Sim 3 (pure Python; pin numpy below 2.4, newer numpy breaks it):

```bash
python3 -m venv ~/.local/opt/scalesim-venv
~/.local/opt/scalesim-venv/bin/pip install scalesim "numpy<2.4"
~/.local/opt/scalesim-venv/bin/python -c "import scalesim"
```

## What each contributes
- **Ramulator 2**: one channel of the memory chiplet's DRAM standard (FR-FCFS, open row,
  MOP4CLXOR mapping, all-bank refresh) on a sequential 64 B stream with the workload's write
  share. ChipLab multiplies the design's spec bandwidth by achieved / channel peak and reports
  loaded read latency. Technologies: HBM2/2E, HBM3/3E, HBM4, LPDDR5/5X, DDR4, DDR5, GDDR7.
  GDDR6 is not supported (the preset lacks nRFCpb/nRFCab).
- **SCALE-Sim 3**: systolic / tensor-engine chiplets are mapped to square arrays
  (`compute.array_dim`, else the largest power of two <= sqrt(mac_units), capped at 128).
  The workload's QKV, attention-output and FFN-up/down GEMMs run on one array's share of N;
  utilization = MACs / (cycles x array MACs). Dataflow is `compute.dataflow` or the best of
  ws/os/is. GEMMs larger than M 512 or 4 folds of N/K are simulated at the cap and scaled by
  the fold ratio; results say `extrapolated`.

Every derated number is listed in the run's assumptions with source
`ramulator:<standard>/<timing preset>` or `scale_sim:<side>x<side>x<arrays>/<dataflow>`.
Power, area, thermal and cost stay analytical.

## Limits
Not cycle-accurate end to end: latency is still max(compute, memory). Attention score and
context GEMMs, caches, the network on chip and random-access memory patterns are not
simulated at this level.
