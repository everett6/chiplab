# ChipLab PDK ladder: sources and accuracy

Generated from `src/chiplab/library/pdk_library.py`; regenerate with `python -m chiplab.library.pdk_library`. Each number carries a provenance: **published** (stated in the cited open PDK or paper) or **estimated** (typical value consistent with the sources). Monte Carlo in the digital twin perturbs process parameters by at least the node's accuracy margin.

| Node | PDK | Tape-out ready | Accuracy margin | Licence | Commercial use |
|---|---|---|---|---|---|
| 180 nm | GlobalFoundries GF180MCU (`gf180`) | yes | ±5% | Apache-2.0 | allowed |
| 130 nm | SkyWater SKY130 (`sky130`) | yes | ±5% | Apache-2.0 | allowed |
| 130 nm | IHP SG13G2 (`ihp_sg13g2`) | yes | ±5% | Apache-2.0 | allowed |
| 65 nm | 65 nm bulk CMOS (published TSMC/UMC 65GP data) (`n65`) | no (predictive) | ±10% | published literature (no PDK licence; real kit is foundry NDA) | not allowed |
| 28 nm | 28 nm planar HKMG (OpenRPDK28 / published 28HPC+ data) (`n28`) | no (predictive) | ±15% | unverified predictive kit + published literature | not allowed |
| 15 nm | NC State FreePDK15 / NanGate 15nm OCL (`freepdk15`) | no (predictive) | ±15% | academic predictive kit (see NC State licence terms) | not allowed |
| 7 nm | ASU/Arm ASAP7 (OpenROAD) (`asap7`) | no (predictive) | ±15% | BSD-3-Clause | allowed |
| 3 nm | 3 nm FinFET/GAA (published IMEC / ISSCC scaling data) (`n3`) | no (predictive) | ±20% | published literature | not allowed |
| 2 nm | 2 nm GAA nanosheet (IRDS / IMEC / ISSCC data) (`n2`) | no (predictive) | ±25% | published literature | not allowed |

## Parameters

| PDK | Vdd (V) | FO4 (ps) | Density (kgate/mm²) | 6T bitcell (µm²) | E_dyn (fJ/gate) | Leakage (nW/gate) | M1 pitch (nm) | Wire C (fF/µm) |
|---|---|---|---|---|---|---|---|---|
| `gf180` | 3.3 | 95* | 60* | 5* | 110* | 0.05* | 460 | 0.2* |
| `sky130` | 1.8 | 45* | 160* | 2* | 16* | 0.1* | 340 | 0.18* |
| `ihp_sg13g2` | 1.2 | 40* | 150* | 2.2* | 6* | 0.08* | 420* | 0.19* |
| `n65` | 1* | 17* | 600* | 0.52* | 2* | 1* | 180* | 0.2* |
| `n28` | 0.9* | 11* | 2000* | 0.127* | 0.73* | 1.5* | 90* | 0.2* |
| `freepdk15` | 0.8 | 8* | 6000* | 0.07* | 0.32* | 1* | 64* | 0.19* |
| `asap7` | 0.7 | 6* | 10000* | 0.027* | 0.147* | 0.8* | 36 | 0.18* |
| `n3` | 0.72* | 4.8* | 28000* | 0.0199* | 0.114* | 0.6* | 23* | 0.2* |
| `n2` | 0.68* | 4.2* | 36000* | 0.0175* | 0.088* | 0.5* | 21* | 0.21* |

`*` = estimated; unmarked = published.

## Sources per node

### 180 nm: GlobalFoundries GF180MCU

- **Source:** google/gf180mcu-pdk (github.com/google/gf180mcu-pdk), GF180MCU design-rule manual and gf180mcu_fd_sc_mcu7t5v0 library
- **Device:** bulk CMOS
- **Error bound:** ±5% on timing, power and area outputs derived from this profile
- **Tape-out ready:** True; **licence:** Apache-2.0; **commercial use:** True
- `nominal_vdd_v` = 3.3 V (published): 3.3 V core devices; 5 V/6 V also offered
- `fo4_delay_ps` = 95 ps (estimated): 3.3 V devices, TT 25 C
- `gate_density_kgates_mm2` = 60 kgate/mm2 (estimated): 7-track NAND2 ~8.8 um2 at ~60% util
- `dynamic_energy_fj_per_gate` = 110 fJ (estimated): ~10 fF switched x 3.3 V^2
- `metal1_pitch_nm` = 460 nm (published): Metal1 min width 0.23 um + space 0.23 um

### 130 nm: SkyWater SKY130

- **Source:** google/skywater-pdk (github.com/google/skywater-pdk), sky130A tech LEF and sky130_fd_sc_hd Liberty
- **Device:** bulk CMOS
- **Error bound:** ±5% on timing, power and area outputs derived from this profile
- **Tape-out ready:** True; **licence:** Apache-2.0; **commercial use:** True
- `nominal_vdd_v` = 1.8 V (published): sky130_fd_sc_hd nominal corner
- `fo4_delay_ps` = 45 ps (estimated): sky130_fd_sc_hd inverter, TT 1.8 V 25 C
- `gate_density_kgates_mm2` = 160 kgate/mm2 (estimated): nand2_1 = 3.75 um2 (published) at ~60% utilization
- `sram_bitcell_um2` = 2 um2 (estimated): foundry 6T SRAM cell used by OpenRAM, ~2 um2
- `dynamic_energy_fj_per_gate` = 16 fJ (estimated): ~5 fF x 1.8 V^2
- `metal1_pitch_nm` = 340 nm (published): met1 pitch 0.34 um (tech LEF)

### 130 nm: IHP SG13G2

- **Source:** IHP-GmbH/IHP-Open-PDK (github.com/IHP-GmbH/IHP-Open-PDK), SG13G2 process specification and sg13g2_stdcell library
- **Device:** bulk CMOS + SiGe HBT
- **Error bound:** ±5% on timing, power and area outputs derived from this profile
- **Tape-out ready:** True; **licence:** Apache-2.0; **commercial use:** True
- `nominal_vdd_v` = 1.2 V (published): 1.2 V core MOS devices
- `dynamic_energy_fj_per_gate` = 6 fJ (estimated): ~4 fF x 1.2 V^2

### 65 nm: 65 nm bulk CMOS (published TSMC/UMC 65GP data)

- **Source:** Published 65 nm GP technology data (TSMC/UMC ISSCC and IEDM papers, 2005-2008). Note: NC State's open predictive kits are FreePDK45 and FreePDK15; ChipLab found no 'FreePDK65', so this node is literature-derived
- **Device:** bulk CMOS
- **Error bound:** ±10% on timing, power and area outputs derived from this profile
- **Tape-out ready:** False; **licence:** published literature (no PDK licence; real kit is foundry NDA); **commercial use:** False
- `nominal_vdd_v` = 1 V (estimated): 65GP core Vdd 1.0 V (LP variants 1.2 V)
- `sram_bitcell_um2` = 0.52 um2 (estimated): reported 6T cells 0.499-0.57 um2

### 28 nm: 28 nm planar HKMG (OpenRPDK28 / published 28HPC+ data)

- **Source:** RIOSLaboratory OpenRPDK28 (as referenced; availability and licence not independently verified by ChipLab) cross-checked with published 28 nm HPC/HPC+ ISSCC data
- **Device:** planar HKMG
- **Error bound:** ±15% on timing, power and area outputs derived from this profile
- **Tape-out ready:** False; **licence:** unverified predictive kit + published literature; **commercial use:** False
- `sram_bitcell_um2` = 0.127 um2 (estimated): widely reported 28 nm HD 6T cell

### 15 nm: NC State FreePDK15 / NanGate 15nm OCL

- **Source:** NC State FreePDK15 (Bhanushali & Davis, ISPD 2015, 'FreePDK15: An Open-Source Predictive Process Design Kit for 15nm FinFET Technology') and the NanGate 15nm Open Cell Library
- **Device:** FinFET (predictive)
- **Error bound:** ±15% on timing, power and area outputs derived from this profile
- **Tape-out ready:** False; **licence:** academic predictive kit (see NC State licence terms); **commercial use:** False
- `nominal_vdd_v` = 0.8 V (published): FreePDK15 nominal supply

### 7 nm: ASU/Arm ASAP7 (OpenROAD)

- **Source:** L. T. Clark et al., 'ASAP7: A 7-nm finFET predictive process design kit', Microelectronics Journal 53 (2016); The-OpenROAD-Project/asap7
- **Device:** FinFET (predictive)
- **Error bound:** ±15% on timing, power and area outputs derived from this profile
- **Tape-out ready:** False; **licence:** BSD-3-Clause; **commercial use:** True
- `nominal_vdd_v` = 0.7 V (published): ASAP7 nominal supply
- `gate_density_kgates_mm2` = 10000 kgate/mm2 (estimated): 7.5-track, 270 nm cell height
- `sram_bitcell_um2` = 0.027 um2 (estimated): typical 7 nm HD 6T cell
- `dynamic_energy_fj_per_gate` = 0.147 fJ (estimated): ~0.3 fF x 0.7 V^2
- `metal1_pitch_nm` = 36 nm (published): M1 pitch 36 nm (Clark et al. 2016)
- **Note:** Licence permits commercial use of the kit, but it is predictive: no foundry manufactures ASAP7.

### 3 nm: 3 nm FinFET/GAA (published IMEC / ISSCC scaling data)

- **Source:** Published 3 nm scaling data: IMEC roadmap papers and ISSCC/IEDM 2022-2023 reports (e.g. 0.0199-0.021 um2 HD SRAM cells); open 3 nm predictive kits ('ASAP3') referenced but not verified by ChipLab
- **Device:** FinFET / GAA (predictive)
- **Error bound:** ±20% on timing, power and area outputs derived from this profile
- **Tape-out ready:** False; **licence:** published literature; **commercial use:** False

### 2 nm: 2 nm GAA nanosheet (IRDS / IMEC / ISSCC data)

- **Source:** IEEE IRDS More Moore roadmap (2022/2023 editions) and IMEC/ISSCC/IEDM nanosheet scaling reports (2023-2025)
- **Device:** GAA nanosheet (research)
- **Error bound:** ±25% on timing, power and area outputs derived from this profile
- **Tape-out ready:** False; **licence:** published literature; **commercial use:** False

## Gain-cell eDRAM retention

Retention at 25 °C per node (estimated) comes from: gain-cell eDRAM literature: P. Meinerzhagen, A. Teman, R. Giterman, N. Edri, A. Burg, A. Fish, 'Gain-Cell Embedded DRAMs for Low-Power VLSI Systems-on-Chip', Springer 2018; retention falls roughly 2x per 10-15 C and with scaling. The on-chip memory model halves retention every 18 °C (about 10x from 25 to 85 °C) and treats it as an 'assumed' parameter (±25% in Monte Carlo).

- `gf180`: 5000 µs
- `sky130`: 2000 µs
- `ihp_sg13g2`: 1500 µs
- `n65`: 500 µs
- `n28`: 100 µs
- `freepdk15`: 40 µs
- `asap7`: 15 µs
- `n3`: 8 µs
- `n2`: 5 µs
