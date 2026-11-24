# Validation: golden references

`tests/test_validation_golden.py` asserts that ChipLab's models land within a stated tolerance
of published data, JEDEC standards, or an independent first-principles computation. These
are **reference checks**: they show the models implement the physics and data they claim.
They are not correlation against measured silicon, and they do not bound the analytical
model's error on real workloads (caches, scheduling and software stack are not modeled).

| # | Model | Reference | Tolerance | Result (2026-11-23) |
|---|---|---|---|---|
| 1 | SRAM bitcell table (N28, N16, N7, N5, N3) | TSMC HD 6T bitcell areas published at IEDM: 0.127, 0.070 (2013), 0.027 (2016), 0.021 (2019), 0.0199 um2 (2022) | 10% | N16 table value 0.074 is +5.7%; others exact |
| 2 | Dies per wafer (25, 100, 400, 814 mm2) | Brute-force count of square dies fully inside the usable circle, best of 20x20 offsets | 3% | Matches. The old closed-form formula was -9% at 400 mm2 and -12% at 814 mm2; replaced by exact placement |
| 3 | Negative-binomial yield | Poisson limit exp(-A*D0) as alpha -> infinity; clustered yield >= Poisson | 1e-5 | Matches |
| 4 | HBM3 stack bandwidth | JEDEC JESD238: 1024 bits x 6.4 Gb/s = 819.2 GB/s per stack | 1% | Exact |
| 5 | LPDDR5 channel bandwidth | JEDEC LPDDR5-6400, x16 channel = 12.8 GB/s | 1% | Exact |
| 6 | Decode roofline | H100 SXM5 published HBM3 bandwidth 3.35 TB/s; 70B FP8 weights read per token at batch 1 -> 47.9 tokens/s upper bound | <= bound, within 2% | 47.8 (seq 128), 47.5 (seq 2048: KV cache reads) |
| 7 | Lumped thermal | Tj = Ta + P * theta_ja | 1e-6 | Identity holds |
| 8 | Uncertainty propagation | Bands bracket nominal; seeded runs are identical; HBM bandwidth ranks first for decode throughput | - | Holds |

## Adding a case
Give the reference value, its source (paper, standard or datasheet with year), and a
tolerance justified by that source. Never widen a tolerance to make a failing case pass:
fix the model or the data, and record the change here.

## Not yet covered
- End-to-end cycle accuracy. Ramulator 2 (DRAM timing, one channel, sequential stream) and
  SCALE-Sim 3 (systolic-array cycles on the workload's GEMMs) are integrated at
  `fidelity="specialized"`, but they only derate the roofline's two ideal inputs; latency
  is still max(compute, memory). No golden cases exist for the specialized level yet.
  Measured sequential-stream efficiency (one channel, Ramulator 2.1 presets): HBM2 0.83,
  HBM3 0.82, HBM4 0.64, LPDDR5 0.86, DDR4 0.94, DDR5 0.91, GDDR7 0.76.
- Measured-silicon correlation of power and performance.
