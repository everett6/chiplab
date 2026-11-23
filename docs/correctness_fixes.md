# ChipLab correctness fixes (2026-11-23)

Results from before this date may be wrong in these ways:

| Area | Was | Now |
|---|---|---|
| Package yield | `1 - exp(-D0*A)`: the chance of at least one defect, reported as yield | `exp(-D0*A)` (Poisson, zero defects) |
| Interconnect | the design's link bandwidth, latency and energy were replaced by library defaults whenever the topology was `mesh`, `crossbar`, `pcie` or `chiplet_link` | the design's values win; defaults fill only missing fields |
| Memory without a `memory:` block | capacity and channels from library defaults were not multiplied by `count`; banks per channel were multiplied by `count` | capacity and channels scale with `count`; banks per channel do not |
| MoE workloads | the expert operator divided the declared FLOPs and bytes by `active_experts`, so part of the work disappeared | it carries the declared totals |
| Discrete-event engine | a closed-form estimate: interconnect fixed at 15% of memory time, bisection at 10% of memory bandwidth, capacity not multiplied by `count`, utilization hard-coded to 100% | a real tile-level event simulation (see README) |
| External backends | returned `success=True` with empty metrics when installed | raise `NotImplementedError`; `specialized` and `rtl_circuit` requests run analytical and are labeled analytical |
| ECHO YAMLs | PIM chiplets declared 0 GB memory, which fails validation | `memory:` block removed (memoryless compute chiplet) |

The on-chip memory model (`design_onchip_memory_subsystem`) covers 6T SRAM and 2T/3T
gain-cell eDRAM macros only. It is not a model of 1T1C DRAM arrays, HBM stacks or analog
DRAM compute-in-memory; results carry `memory_model_class` to say so.
