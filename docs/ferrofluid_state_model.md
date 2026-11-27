# Electrical ferrofluid state-space model and reference experiment

**Status: the published experiment is not reproduced. Its raw data are not available to
ChipLab.** What exists is a model that can be fitted to such data, and a harness that
compares the two.

This is a second ferrofluid model. The magnetic particle model
([ferrofluid_model.md](ferrofluid_model.md)) drives the fluid with coil fields and has no
electric-field effect, so it cannot represent an electrically programmed experiment.

| File (`src/chiplab/`) | Contents |
|---|---|
| `models/materials/ferro_statespace.py` | N-port network, state equations, sensitivity, multiport report |
| `models/materials/ferro_experiment.py` | CSV / drive trace / Touchstone import, comparison, fitting |
| `models/materials/ferro_reference.py` | the cited experiment, reproduction harness, data request |
| `model_registry.py` | scientific model registry |

MCP: `chiplab_ferrofluid_state_model(task, model, protocol, measurements, options)` and
`chiplab_model_registry`.

## Model

`S(t+dt) = F(S, V, T, θ)`, `Z(t) = G(S, ω, V, T, θ)`. The sample is one network: a branch
from each port to ground and a branch between port pairs.

| Part | Equation |
|---|---|
| fast and slow states, per branch | `dz/dt = a(v) (tanh(v/Vs) − z) − z/τr`, `a(v) = (cosh(v/Vw) − 1)/τw` |
| reversible fatigue | `dw/dt = a_f(v) (1 − w) − w/τrec` |
| temperature | `τ(T) = τ exp(Ea/kB (1/T − 1/Tref))`; resistances likewise |
| branch impedance | `Z_b = R∞ + ΔR/(1 + (jωτ)^α)`, in parallel with `C` |
| state to impedance | `R∞, ΔR, C` scale as `exp((1 − d·w)(g_fast z_fast + g_slow z_slow))`; optional bias terms on `ΔR` |
| ports | `Z(ω) = Y(ω)⁻¹` (reciprocal and passive) |

Updates are exact when every port is driven; with floating ports their voltages are solved
and the step is subdivided (`substep_convergence` is reported). `target_symmetry` is `odd`
(polarity is stored) or `even`; for a port-pair branch, positive voltage means the first
port in the port list is higher.

**These are empirical circuit and kinetic parameters.** They have no defaults. They are not
particle size, concentration or viscosity: changing the fluid has no effect here until the
model is refitted. `"never"` for a retention time declares a nonvolatile state; claim that
only if data show it. The equations are a modelling choice of ours, not taken from the
cited work, and may prove unable to fit it.

## The reference experiment

M. Crepaldi, C. Mohan, E. Garofalo, A. Adamatzky, K. Szaciłowski, A. Chiolerio,
*Evidence of In-Memory Computing in a Ferrofluid*,
[arXiv:2211.08152v1](https://arxiv.org/html/2211.08152v1) (preprint); published in
Adv. Mater. 2023, doi:10.1002/adma.202211406. Only the preprint's main text and captions
were read (through a web-fetch summary), not its supplement or the published version.
Check the values against the paper before relying on them.

| Reported | Value | Where |
|---|---|---|
| sample | water-based Fe3O4 ferrofluid | abstract |
| set-up | two-port VNA, DC bias generator (negative grounded), two bias tees | Fig. 1A text |
| sweep | −3.8 V to 3.8 V, 0.1 V steps of 1 s, 50 repeats | Fig. 1C |
| summed Z at VP = −0.85 V | Z11 14312, Z12 2320, Z21 2060, Z22 11795 Ω | Fig. 1C text |
| read band | 10 MHz – 6 GHz, impedance summed over frequencies | Fig. 1 |
| memory test | 3.3 V pulses of varying duration; reset to summed Z11 = 14338 Ω | Fig. 2 |
| behaviour | pinched hysteresis that shrinks with iterations; reversible shrinking of dynamics over days | Fig. 1C, text |
| data | available from the corresponding author on request | data statement |

Not stated in what was read: composition, geometry, size of the impedance changes, time
constants, temperature, pulse durations, reset voltage, port wiring, sweep order. The
reported Z12 ≠ Z21 cannot be represented by this reciprocal model. The classification
accuracies depend on a software readout and are outside the harness.

### Data to obtain

`.s2p` files or impedances with time stamps for the sweep, the pulse tests and the multi-day
run; the applied voltage on each port; VNA settings and calibration; the definition of the
summed impedance; repeat sweeps for uncertainty; sample and geometry; temperature.

### Harness verdict

Items `hysteresis`, `pulse_duration`, `long_term_drift` (tag datasets with
`reference_item`) each report `reference_data_missing`, `model_parameters_missing`,
`protocol_differs`, `insufficient_coverage`, `not_independent`, `disagrees` or `agrees`.
Overall `reproduced_on_supplied_data` needs all three to agree. Agreement: reduced χ² ≤ 2,
≥ 90 % within 2σ, ≥ 5 points, median σ ≤ 20 % of rms signal (our thresholds). The tests
reach a pass only with synthetic data from the model.

## Workflow

```python
from chiplab.models.materials import ferro_experiment as ex, ferro_reference as ref
c = ex.read_csv_columns("trace.csv")
protocol = ex.protocol_from_trace(c["time_s"], {"p1": c["v1"], "p2": c["v2"]},
                                  c["temperature_k"])
datasets = ex.touchstone_datasets(text, ["p1", "p2"], time_s=12.0, sigma_ohm=2.0,
                                  reference_item="hysteresis")
record = {"provenance": {"source": "..."}, "protocol": protocol, "datasets": datasets}
fit = ex.fit(start_model, calibration_record, ["ground.delta_r_ohm", ...])
ref.reproduce(fit["model"], held_out_records)
```

Dataset kinds: `impedance`,
`summed_impedance`, `current` (see the module docstring). Fit on one run, judge on another.

## Multiport

`task="multiport"` writes each port alone and all together, reports the non-additive part of
the impedance change, and, given your measured noise, the smallest separation between written
patterns (exhaustive up to 8 ports; single-bit neighbours of "none" and "all" above that).
2, 4, 8 and 16 ports run. Above 2 ports, shared branch parameters make it an extrapolation:
electrode geometry does not enter unless each branch is fitted.

## Audit of the third report

| Requested | Status | Still missing |
|---|---|---|
| voltage-dependent, bipolar, pulse-width writes | new (electrical); magnetic model had voltage and width | validation |
| hysteresis | new | validation; pinching is not enforced |
| relaxation, short- and long-term memory | new: two states, optional nonvolatile | more than two time scales |
| frequency-dependent impedance | new electrical; magnetic had inductive | fixture / distributed effects |
| temperature | new Arrhenius terms | heating |
| "a physical parameter explains the output" | met by the magnetic model for material parameters; here only for circuit/kinetic parameters | link from composition to these parameters |
| N-port Z(ω,t), simultaneous writes, interference, retrieval | extended / new | non-reciprocity, geometry, a decoder, any experiment |
| reproduce two-port reference | harness only | the data |
| state-space model | new | evidence that this form fits |
| import V/I, impedance, sweeps, temperature, retention, timing; compare | new (CSV, trace, Touchstone v1) | other instrument formats, sweep duration |
| shrinking response with prolonged use | new reversible fatigue state | irreversible degradation |
| P1 circuit co-simulation (drivers, DAC, RF, ADC, controller) | not started | all |
| P1 reliability | partial: temperature, relaxation, reversible drift | endurance, variation, read disturb here, degradation |
| P1 package scaling, 256 GB / 500 GB/s | not started, by design | a validated device model first |
| P2 visualization, optimization | not started | all |
| model registry | new, 6 records | other models; engines do not dispatch through it |

The state model cannot enter an architecture-level design: the validation gate only
accepts records of the magnetic model, and rejects everything else.
