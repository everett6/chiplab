# External tools in GPU Studio

GPU Studio's **External tools** rail (Circuit, Board, Layout, CFD) drives real EDA tools.
ChipLab writes their input files and parses their output; it never reimplements or imitates
them. Code: `src/chiplab/tools/`, HTTP API `src/chiplab/app/tools_api.py`, UI
`src/chiplab/app/static/toolviews.js`.

| View | Tool | What it does |
|---|---|---|
| Circuit Studio | ngspice (+ Xschem) | Fixed SPICE templates (DRAM sense amp, 1T1C cell, 2T gain cell, eCore charge-domain MAC column) built from validated numbers, run with `ngspice -b -r`, raw file parsed with numpy, waveforms in Plotly. Schematic exported by Xschem (`xschem -x -q --svg`) from a generated `.sch`, or a generated SVG; net labels are coloured by the simulated voltage at the time cursor. |
| Board Studio | KiCad `kicad-cli` | `scene["pcb"]` written as a self-contained 4-layer `.kicad_pcb` (downloadable without KiCad); KiCad plots SVG, renders 3D PNG and runs DRC. DRC counts are KiCad's; the placement is coarse, not routed. |
| Layout Studio | KLayout (Python package) | Read-only GDSII/OASIS viewer: upload, layer list, KLayout LayoutView PNG render per layer selection. Also exports the design's package floorplan to GDS. |
| CFD | OpenFOAM | One fin-channel case (simpleFoam + scalarTransportFoam), outlet temperature → h via LMTD, compared with `studio/thermal.py`. Not installed on this machine: the view shows the case files and the airflow-model numbers only. |

## Safety
* `registry.run` is the only process launcher: known tools only, argument lists, `shell=False`,
  temp working dir, timeout. Clients send template keys, numbers, design ids and layout bytes.
* Layout uploads are checked by magic bytes, stored under a content hash, never written back.

## Installation (optional, per tool)
Lookup order: `CHIPLAB_<TOOL>` env var → `PATH` → `~/.local/opt/<tool>/usr/bin` (Debian
packages unpacked with `apt-get download` + `dpkg -x`, no root). On this machine:
ngspice 45.2, Xschem 3.4.4 and KiCad 9.0.8 are unpacked under `~/.local/opt`; the ngspice
`spinit` there was edited to point its code models at the local path. KLayout 0.30.14 is
pip-installed in `.venv`. OpenFOAM is not installed.

Tests: `tests/test_tools_integration.py` (tool-driven tests skip when a tool is missing).
