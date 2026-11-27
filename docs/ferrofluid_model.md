# Ferrofluid particle model

Physics-based, reduced-order, **uncalibrated** model of a magnetic nanoparticle suspension
read and written by several conductor ports. It has been checked against analytic limits
and a particle-resolved stochastic simulation, **not against any experiment**.

This model drives the fluid magnetically and has no electric-field effect. For electrically
programmed experiments see [ferrofluid_state_model.md](ferrofluid_state_model.md).

| File (`src/chiplab/models/materials/`) | Contents |
|---|---|
| `ferro_particles.py` | particle statistics, relaxation, susceptibility, magnetophoresis, stochastic reference |
| `ferro_device.py` | conductor ports, fields, port impedance matrix, pulse sequences |
| `ferro_calibration.py` | fit to user measurements |
| `ferro_interface.py` | JSON adapter used by the MCP tools |
| `ferro_analysis.py` | sensitivity, read disturbance, state distinguishability, array scaling |
| `ferro_validation.py` | physics validation gate for use in architecture-level designs |

`src/chiplab/credibility.py` holds the name matching, refusals and the design-level gate.

`ferrofluid.py`, `impedance.py`, `electrode_array.py` and `particle_dynamics.neel_relaxation`
are older **heuristic** helpers, kept for existing callers and labelled as such. The MCP tools
do not use them for ferrofluid.

## Chain of physics

voltage → current `I = V/R` → field `H(r)` (Biot-Savart, summed over ports) → magnetization
and concentration of particles → flux and impedance at every port.

## Equations and validity

| Quantity | Equation | Valid when / limitation |
|---|---|---|
| Conductor field | exact finite-segment Biot-Savart; loops are 96-segment polygons | quasi-static; no skin effect, no field from the fluid |
| Wire resistance | `R = R_series + ρ_Cu N l/(π a²)` | 293 K copper; no heating |
| Self inductance | loop `μ0 R N² (ln(8R/a) − 7/4)`; line (Rosa, partial) `μ0 l/2π (ln(2l/a) − 3/4)` | thin wire (`a ≪ R`); N turns treated as one bundle |
| Mutual inductance | Neumann double integral | conductors that do not touch |
| Equilibrium magnetization [R] | `M = Σ_k φ_k M_d L(ξ_k)`, `ξ = μ0 m H_e/kT`, `L = coth ξ − 1/ξ` | superparamagnetic, dilute |
| Particle interaction [IK] | `H_e = H + M_L(H)/3` (first-order mean field) | φ ≲ 0.15, dipolar coupling λ ≲ 1–2; chaining not modelled |
| Size distribution | log-normal core diameter, Gauss-Hermite bins weighted by volume | single population |
| Brownian time [S74] | `τ_B = 3 η V_h / kT` | Newtonian carrier, viscosity given at the operating T |
| Néel time [N] | `τ_N = τ0 exp(K V/kT)` | KV/kT ≳ 2; no field dependence |
| Combined | `1/τ = 1/τ_B + 1/τ_N` [S74] | |
| Field-dependent Brownian [MRS] | `τ∥/τ_B = d ln L/d ln ξ`, `τ⊥/τ_B = 2L/(ξ − L)` | near equilibrium |
| Magnetization dynamics [S74] | `dM/dt = −(M − M_eq)_∥/τ∥ − (M − M_eq)_⊥/τ⊥` per cell and size | near equilibrium; no precession, no blocked-particle switching |
| Complex susceptibility | `χ*(ω) = Σ_k χ_k/(1 + jωτ_k)`, tensor about the bias field | small read signal; f ≲ 100 MHz |
| Magnetophoresis [R] | equilibrium `c ∝ sinh ξ/ξ`, particle number conserved; relaxes with `τ_D = ℓ²/(π² D)` | **single-mode, order of magnitude only**; no steric limit, no flow |
| Port impedance (reciprocity) | `Z_ij = R_i δ_ij + jω[L0_ij + μ0 ∫ h_i·χ*·h_j dV]` | first order in χ: relative error of order χ (warned above 0.3) |
| Flux and induced voltage | `λ_j = μ0 ∫ h_j·M dV`, `v_j = −dλ_j/dt` | same as above |
| Read noise | `v_n = sqrt(4 kT Re Z B)` (fluctuation-dissipation) | thermal only; amplifier noise only if supplied |
| Electrode pair (optional) | two pillars, insulating floor and lid: `G = 2πσh/acosh((s² − r1² − r2²)/2r1r2)`, `C = Gε/σ`, optional series double layer | needs user conductivity and permittivity; independent of magnetic state |

References: [R] Rosensweig, *Ferrohydrodynamics* (1985). [S74] Shliomis, Sov. Phys. Usp. 17,
153 (1974). [N] Néel, Ann. Geophys. 5, 99 (1949); Brown, Phys. Rev. 130, 1677 (1963).
[MRS] Martsenyuk, Raikher, Shliomis, Sov. Phys. JETP 38, 413 (1974). [IK] Ivanov,
Kuznetsova, Phys. Rev. E 64, 041405 (2001). Citations are given from general knowledge and
were not re-checked against the papers while writing this model.

## Default values (all assumed unless you supply them)

φ = 0.01, core diameter 10 nm (σ_ln 0.2), coating 2 nm, M_d 446 kA/m (bulk magnetite),
K = 2·10⁴ J/m³, τ0 = 1 ns, η = 2 mPa·s, T = 300 K. Port: 20 µm radius single-turn loop,
1 µm wire radius, 50 Ω series resistance, 100 µm pitch. Fluid: 10 µm film, 1 µm above the
conductors. Every result lists `assumed_inputs` and a provenance label per parameter.

## Using it through MCP

| Tool | Ferrofluid behaviour |
|---|---|
| `chiplab_simulate_experimental_tech` | one write and one read loop (or `electrodes` / `ports`); unmodelled metrics are `null` |
| `chiplab_custom_device_simulate` | any electrodes around one fluid region; reads with field on and off |
| `configure_material_array` → `run_multiphysics_simulation` | independent, overlapping write pulses; reads at chosen times and frequencies; `port_interaction` |
| `simulate_material_circuit` | dc / ac / noise / transient from the same model |
| `sweep_material_parameters` | objectives `write_energy`, `power`, `read_signal`, `read_flux`, `decay_time`, `peak_field` |
| `calibrate_material_model` | least-squares fit to supplied data |
| `chiplab_ferrofluid_analyze` | `sensitivity`, `read_disturbance`, `retention`, `scaling` |
| `chiplab_validate_material_device` | the gate report: every check pass, fail or not_run |
| `material_device_to_chiplet` | no chiplet unless the gate accepts; then conductor bounding-box area and drive power only |

A device counts as ferrofluid if any common spelling appears in `technology`, `material`,
`type`, `name`, `regions`, the array technology or a parameter block. A spec that names
both ferrofluid and another technology is refused. Generic paths that have no ferrofluid
physics refuse it: node scaling, reliability analysis, architecture discovery,
manufacturing sign-off, digital twin management, component creation, `get_solver`,
`ReducedOrderSolver`, `ParticleScaleSolver`, `SpatialMultiphysicsSolver` and
`FerrofluidCoupling`. The older helpers listed above stay callable and labelled heuristic.

`port_interaction.nonlinear_residual` is *all ports driven* minus *sum of each port alone*.
The impedance signal depends on H², so the cross term gives a residual even in weak fields;
the flux is linear in weak fields and departs only as particles saturate.

## What the default device actually predicts

Computed with all defaults (single-turn 20 µm loops 50 µm apart), not measured:

| Drive | Current | Peak field | Read impedance change | Read thermal noise (1 MHz) |
|---|---|---|---|---|
| 0.2 V | 3.9 mA | 248 A/m | 3.5·10⁻¹² Ω | 0.92 µV |
| 0.5 V | 9.9 mA | 619 A/m | 2.2·10⁻¹¹ Ω | 0.92 µV |
| 1.0 V | 19.7 mA | 1239 A/m | 8.7·10⁻¹¹ Ω | 0.92 µV |

* The read signal of such a device is far below thermal noise.
* The magnetization falls to 1/e in about **0.1 µs** once the field is removed. A liquid
  carrier cannot hold a magnetic state much beyond the Brownian times (µs, longer only for
  rare large particles). The old 1000 s "retention" default has no support in this model.
* Concentration changes last longer (seconds) but are about 10⁻⁶ relative at 1 V.

## Calibration

```json
{"provenance": {"source": "VSM run 12, sample A, 2026-11-01"},
 "datasets": [{"kind": "magnetization_curve", "H_a_per_m": [], "M_a_per_m": [],
               "sigma_M_a_per_m": []}],
 "fixed_parameters": {"temperature_k": 295}}
```

Dataset kinds: `magnetization_curve`, `susceptibility_spectrum` (`frequency_hz`,
`chi_real`, `chi_loss`, optional `bias_field_a_per_m`), `magnetization_decay` (`time_s`,
`m_normalized`, `initial_field_a_per_m`). A source is mandatory. Output: values, standard
errors and 95 % intervals (linearised), correlations, non-identifiable parameters (not
passed on), reduced χ², data SHA-256, and `independently_validated: false`. Pass the result
back as `calibration` to the simulation tools. Fitting port impedance data directly is
**not** implemented.

## Physics validation gate

A material chiplet enters an architecture-level design only with its device record (spec,
array, writes, reads, measurements). Every check is `pass`, `fail` or `not_run`, with
evidence; **accepted only if all pass**. Checks are recomputed from the record each time
(cached by content hash); a stored "passed" flag is never read.

| Check | Passes when |
|---|---|
| `zero_input` | zero drive gives zero field, magnetization and signal |
| `parameter_sensitivity` | changing each of the nine material parameters changes an output |
| `unit_consistency` | independent routes to the same quantity agree |
| `boundary_conditions` | weak-field, saturation, long-time and frequency limits hold |
| `conservation` | particle number conserved; port network passive and reciprocal |
| `numerical_convergence` | fluid inductance changes by at most 10 % on a finer grid |
| `model_validity` | no validity-range warning (a blocked-particle notice is excluded) |
| `inputs_established` | all nine parameters user-supplied or fitted; port geometry, fluid region and operating conditions given |
| `experiment_equilibrium` | your magnetization curve agrees and reaches the peak write field |
| `experiment_dynamics` | your susceptibility spectrum covers the read frequencies, or your decay data agree |
| `experiment_port_response` | your fluid-induced port impedance change agrees for every read port |

Measurement agreement requires: a stated source; at least 5 points; reduced χ² ≤ 2; at
least 90 % of points within 2σ; median σ at most 20 % of the rms signal; temperature
within 5 K. A dataset identical to one used in calibration fails. **These thresholds are
choices made in `ferro_validation.py`, not industry standards.** A record from which no
device can be built reports `device_definition: fail` and everything else `not_run`.

Port-response dataset: `{"kind": "port_impedance_change", "port", "to_port" (optional),
"frequency_hz", "delta_z_real_ohm", "delta_z_imag_ohm", "sigma_delta_z_real_ohm",
"sigma_delta_z_imag_ohm"}` (fluid filled minus empty, unwritten).

Where it is enforced: `material_device_to_chiplet`, design validation, the analytical,
discrete-event and specialized engines, the cost model, the design store, RTL generation,
design verification, full-project export and `integrate_into_design`. An accepted chiplet
has `type: "material"`, the computed `area_mm2` and `tdp_watts`, and no memory, compute or
process node; edits to those or to the record are caught. Performance, capacity and cost
are not modelled: engines count it as zero compute and the cost model lists it at $0 with
a `NOT MODELLED` warning.

What the gate cannot do:

* Tell whether measurements are authentic. Data generated from the model passes (the tests
  do exactly this, with a source string that says it is synthetic).
* Check written-state dynamics, retention or port interaction against experiment: the port
  data cover the unwritten fluid only.
* Detect calibration data reused with any value changed.
* Recognise a material chiplet that does not declare or name itself as one.
* Gate other experimental technologies a hand-written chiplet names (memristor, PCM,
  ferroelectric, spintronic, photonic and similar): the validator only warns.
* Re-judge simulation results saved before the gate existed.
* RTL generation says nothing about a material chiplet (no RTL exists for one).
* Cover `app/custom_memory_routes.py`, which does its own chiplet arithmetic.

`integrate_into_design` / `derive_chip_parameters` infer ferrofluid from 2 bits per cell,
so a 2-bit device described only by cell numbers is refused there.

## Analyses

* **Sensitivity**: central-difference `d ln(output) / d ln(parameter)` for the nine
  parameters and drive voltage. Local to the operating point: for a write shorter than the
  slow relaxation times, raising viscosity can *shorten* the decay time because slow
  particles are written less.
* **Read disturbance**: (written + read) − (unwritten + read) − (written, no read),
  relative to the stored state, averaged over the last read cycle; grows as amplitude².
  The residue a read leaves regardless of the state (linear in amplitude) is reported
  separately. At most 200 cycles; no heating, no switching of blocked particles.
* **Distinguishability** (`retention`): time for which the induced-voltage difference of
  two written states exceeds a chosen multiple of the port thermal noise in a chosen
  bandwidth. One ideal sample; no amplifier noise or averaging.
* **Scaling**: steady-state flux of a larger array predicted from port-to-port transfer in
  a small one. Reported only if it matches a **direct simulation of the same model** within
  10 %, the Langevin argument is at most 0.3 and the large grid has converged; otherwise,
  and above 32 ports, `not_modelled`. Impedance, capacity, bandwidth, energy, yield,
  routing and heating are not scaled. Not checked against measurements.

## Still missing

* Multi-port: mutual capacitance and electrical coupling between coil ports, eddy
  currents, source/driver impedance and crosstalk circuits, field distortion by the fluid.
* Retention: switching over anisotropy barriers under field, aggregation, sedimentation,
  flow, temperature drift.
* Calibration: fitting port impedance or pulse data, joint fits with geometry unknowns,
  non-linearised uncertainty, tracking which data were used across sessions.

## Not modelled

Energy per bit, bit latency, endurance, storage density, thermal sensitivity; Joule
heating; electric-field forces on particles; magnetic-state dependence of conductivity or
permittivity; aggregation, sedimentation, flow; demagnetizing field and field distortion by
the fluid; eddy currents, skin effect, coil capacitance, ferromagnetic resonance; field
dependence of Néel relaxation and blocked-particle switching; L/R current rise (quasi-static `I = V/R`).

## Cross-check of the field-dependent relaxation times

One-off run (2·10⁵ stochastic particles, small field step, not part of the test suite),
relaxation time over τ_B, stochastic / model:

| ξ | parallel | perpendicular |
|---|---|---|
| 1 | 0.84 / 0.88 | 0.87 / 0.91 |
| 3 | 0.46 / 0.45 | 0.57 / 0.58 |
| 6 | 0.20 / 0.20 | 0.31 / 0.32 |

## Tests

`tests/test_ferrofluid_physics.py`: Langevin limits and saturation, mean-field initial
susceptibility, Néel/Brownian values, field-dependent relaxation limits, Debye peak,
stochastic particles against `L(ξ)` and `exp(−t/τ_B)`, diffusion MSD, loop and wire fields,
dipole mutual inductance, passivity and reciprocity, `RC = ε/σ`, `emf = −dλ/dt`,
superposition at weak drive, fit recovery on **synthetic** data (which tests the fitting
code, not the physics), and the MCP tools. `tests/test_material_credibility.py`: gate
decisions, rejection and bypass attempts, name routing and the analyses, again with
synthetic "measurements". These tests were written alongside the model;
passing them is not experimental validation.
