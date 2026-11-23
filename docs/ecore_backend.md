# eCore backend (Echo Lab analog DRAM CIM)

ChipLab calls Echo Lab's eCore simulator for **array-level** numbers and uses them in
package-level estimates for PIM / analog-CIM chiplets. Echo Lab stays a separate, read-only
codebase (`~/.gemini/antigravity/scratch/Echo Lab`, overridable by the operator with
`CHIPLAB_ECHO_LAB_PATH`).

## How it is wired
- `src/chiplab/backends/ecore.py`: `EcoreBackend.simulate(rows, cols, precision, adc_bits,
  organization, seed)`. Echo Lab is imported in-process (its deps, numpy and PyYAML, are in
  ChipLab's venv). No subprocess, shell, eval or exec; MCP arguments are validated enums and
  bounded ints. No bytecode is written into Echo Lab.
- If Echo Lab or its SPICE calibration JSON is missing, `is_available()` is False and every
  call raises `EcoreUnavailableError`. eCore's own silent uncalibrated fallback is refused.
- MCP tool `simulate_ecore_array` (read-only). `backend_status()` lists `ecore`.
- Chiplets opt in with `compute.ecore` (see `projects/echo/echo_v0_1_ecore.yaml`):
  `{rows, cols, precision, adc_bits, organization, arrays?, array_area_fraction=0.6, seed}`.
  `apply_ecore_to_design` (run by the analytical and discrete-event engines and by the MCP
  cost/evaluate paths) sets `peak_tops = arrays x array TOPS`, `tdp_watts = arrays x array
  power`, and `area_mm2 = arrays x array area / array_area_fraction` when `arrays` is given
  (otherwise the array count is derived from `area_mm2`). Every substituted number is an
  `AssumptionRecord` with source `ecore:<rows>x<cols>/<precision>/<org>/adc<bits>` and the
  calibration date and sha256; `raw_details["ecore"]` keeps the full provenance.

## What a result contains
- `accuracy`: SNR, rel L2, RMSE, max/mean abs error and Pearson r against exact `W@x`.
  - int4/int8: `ecore_bitlevel_signed_int_matvec`, the real bit-sliced charge-domain path.
  - fp8/fp16: `calibrated_statistical_fp_model`, Echo Lab's FP pipeline, which applies the
    calibrated noise statistics to the digital reference (`ecore_calls=0`). Treat it as an
    optimistic bound.
- `per_mac_snr_db`: eCore's single-MAC swing/noise ratio (the ~45 dB headline). It is **not**
  end-to-end accuracy.
- `latency_ns`, `energy_fj`, `area_um2`: totals plus per-term breakdowns, each tagged with the
  Echo Lab source (`ngspice_derived` -> provenance `estimated`, `analytical_assumption` ->
  `assumed`).
- `derived`: array TOPS, pJ/op, W, mm2, weight capacity.
- `calibration`: kind `spice` (ngspice-41, generic Level-1 MOSFET, not a PDK),
  `measured_silicon: false`, the file, its sha256, and its date (the file's mtime, since the
  JSON has no date field).
- `limitations`.

## Known finding (as of 2026-11-23)
The bit-level int path does not reproduce `W@x`: end-to-end SNR is ~0 dB (rel L2 ~1.0) at
every size tried. Echo Lab's own `results/stage3_ecore/dimension_sweep.csv` shows the same
(`snr_db_vs_exact` ~0). ChipLab reports this as a limitation and a warning on any design that
uses such an array. Its TOPS are not usable compute until Echo Lab fixes the recombination or
scaling.
