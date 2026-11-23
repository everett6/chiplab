"""Multi-node PDK ladder (180 nm -> 2 nm) with per-field provenance, citations, error bounds
and licence / tape-out status, plus the commercial-readiness validator.

Provenance rules used here:
- 'published': the number is stated in the cited open PDK (design rules, tech LEF, Liberty)
  or the cited paper. Only used where ChipLab's authors are confident of the figure.
- 'estimated': derived or typical values consistent with the cited sources; treat with the
  node's accuracy_margin_pct.
Predictive kits (FreePDK, ASAP7) and literature-derived nodes are NOT manufacturable;
tapeout_ready=False and the validator warns for commercial production.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from chiplab.schemas.common import Provenance

P, E = Provenance.PUBLISHED, Provenance.ESTIMATED


class PdkField(BaseModel):
    value: float
    unit: str
    provenance: Provenance
    note: str = ""


class PdkProfile(BaseModel):
    key: str
    node_nm: float
    name: str
    device: str
    citation_source: str
    nominal_vdd_v: PdkField
    vth_v: PdkField
    fo4_delay_ps: PdkField
    gate_density_kgates_mm2: PdkField
    sram_bitcell_um2: PdkField
    dynamic_energy_fj_per_gate: PdkField
    leakage_nw_per_gate: PdkField
    metal1_pitch_nm: PdkField
    wire_cap_ff_per_um: PdkField
    tapeout_ready: bool
    accuracy_margin_pct: float
    license_type: str
    commercial_use_allowed: bool
    cost_node: str = Field(description="cost-reference node used for wafer price / D0")
    gain_cell_retention_us_25c: PdkField
    notes: str = ""

    def scalar(self, field: str) -> float:
        return float(getattr(self, field).value)


def _f(v: float, unit: str, prov: Provenance = E, note: str = "") -> PdkField:
    return PdkField(value=v, unit=unit, provenance=prov, note=note)


_GC_SRC = ("gain-cell eDRAM literature: P. Meinerzhagen, A. Teman, R. Giterman, N. Edri, "
           "A. Burg, A. Fish, 'Gain-Cell Embedded DRAMs for Low-Power VLSI Systems-on-Chip', "
           "Springer 2018; retention falls roughly 2x per 10-15 C and with scaling")

PDK_LADDER: dict[str, PdkProfile] = {p.key: p for p in [
    PdkProfile(
        key="gf180", node_nm=180, name="GlobalFoundries GF180MCU", device="bulk CMOS",
        citation_source="google/gf180mcu-pdk (github.com/google/gf180mcu-pdk), "
                        "GF180MCU design-rule manual and gf180mcu_fd_sc_mcu7t5v0 library",
        nominal_vdd_v=_f(3.3, "V", P, "3.3 V core devices; 5 V/6 V also offered"),
        vth_v=_f(0.70, "V", E),
        fo4_delay_ps=_f(95.0, "ps", E, "3.3 V devices, TT 25 C"),
        gate_density_kgates_mm2=_f(60.0, "kgate/mm2", E, "7-track NAND2 ~8.8 um2 at ~60% util"),
        sram_bitcell_um2=_f(5.0, "um2", E),
        dynamic_energy_fj_per_gate=_f(110.0, "fJ", E, "~10 fF switched x 3.3 V^2"),
        leakage_nw_per_gate=_f(0.05, "nW", E),
        metal1_pitch_nm=_f(460.0, "nm", P, "Metal1 min width 0.23 um + space 0.23 um"),
        wire_cap_ff_per_um=_f(0.20, "fF/um", E),
        tapeout_ready=True, accuracy_margin_pct=5.0, license_type="Apache-2.0",
        commercial_use_allowed=True, cost_node="N180",
        gain_cell_retention_us_25c=_f(5000.0, "us", E, _GC_SRC)),
    PdkProfile(
        key="sky130", node_nm=130, name="SkyWater SKY130", device="bulk CMOS",
        citation_source="google/skywater-pdk (github.com/google/skywater-pdk), sky130A tech "
                        "LEF and sky130_fd_sc_hd Liberty",
        nominal_vdd_v=_f(1.8, "V", P, "sky130_fd_sc_hd nominal corner"),
        vth_v=_f(0.45, "V", E),
        fo4_delay_ps=_f(45.0, "ps", E, "sky130_fd_sc_hd inverter, TT 1.8 V 25 C"),
        gate_density_kgates_mm2=_f(160.0, "kgate/mm2", E,
                                   "nand2_1 = 3.75 um2 (published) at ~60% utilization"),
        sram_bitcell_um2=_f(2.0, "um2", E, "foundry 6T SRAM cell used by OpenRAM, ~2 um2"),
        dynamic_energy_fj_per_gate=_f(16.0, "fJ", E, "~5 fF x 1.8 V^2"),
        leakage_nw_per_gate=_f(0.1, "nW", E),
        metal1_pitch_nm=_f(340.0, "nm", P, "met1 pitch 0.34 um (tech LEF)"),
        wire_cap_ff_per_um=_f(0.18, "fF/um", E),
        tapeout_ready=True, accuracy_margin_pct=5.0, license_type="Apache-2.0",
        commercial_use_allowed=True, cost_node="N130",
        gain_cell_retention_us_25c=_f(2000.0, "us", E, _GC_SRC)),
    PdkProfile(
        key="ihp_sg13g2", node_nm=130, name="IHP SG13G2", device="bulk CMOS + SiGe HBT",
        citation_source="IHP-GmbH/IHP-Open-PDK (github.com/IHP-GmbH/IHP-Open-PDK), SG13G2 "
                        "process specification and sg13g2_stdcell library",
        nominal_vdd_v=_f(1.2, "V", P, "1.2 V core MOS devices"),
        vth_v=_f(0.40, "V", E),
        fo4_delay_ps=_f(40.0, "ps", E),
        gate_density_kgates_mm2=_f(150.0, "kgate/mm2", E),
        sram_bitcell_um2=_f(2.2, "um2", E),
        dynamic_energy_fj_per_gate=_f(6.0, "fJ", E, "~4 fF x 1.2 V^2"),
        leakage_nw_per_gate=_f(0.08, "nW", E),
        metal1_pitch_nm=_f(420.0, "nm", E),
        wire_cap_ff_per_um=_f(0.19, "fF/um", E),
        tapeout_ready=True, accuracy_margin_pct=5.0, license_type="Apache-2.0",
        commercial_use_allowed=True, cost_node="N130",
        gain_cell_retention_us_25c=_f(1500.0, "us", E, _GC_SRC)),
    PdkProfile(
        key="n65", node_nm=65, name="65 nm bulk CMOS (published TSMC/UMC 65GP data)",
        device="bulk CMOS",
        citation_source="Published 65 nm GP technology data (TSMC/UMC ISSCC and IEDM papers, "
                        "2005-2008). Note: NC State's open predictive kits are FreePDK45 and "
                        "FreePDK15; ChipLab found no 'FreePDK65', so this node is "
                        "literature-derived",
        nominal_vdd_v=_f(1.0, "V", E, "65GP core Vdd 1.0 V (LP variants 1.2 V)"),
        vth_v=_f(0.35, "V", E),
        fo4_delay_ps=_f(17.0, "ps", E),
        gate_density_kgates_mm2=_f(600.0, "kgate/mm2", E),
        sram_bitcell_um2=_f(0.52, "um2", E, "reported 6T cells 0.499-0.57 um2"),
        dynamic_energy_fj_per_gate=_f(2.0, "fJ", E),
        leakage_nw_per_gate=_f(1.0, "nW", E),
        metal1_pitch_nm=_f(180.0, "nm", E),
        wire_cap_ff_per_um=_f(0.20, "fF/um", E),
        tapeout_ready=False, accuracy_margin_pct=10.0,
        license_type="published literature (no PDK licence; real kit is foundry NDA)",
        commercial_use_allowed=False, cost_node="N65",
        gain_cell_retention_us_25c=_f(500.0, "us", E, _GC_SRC)),
    PdkProfile(
        key="n28", node_nm=28, name="28 nm planar HKMG (OpenRPDK28 / published 28HPC+ data)",
        device="planar HKMG",
        citation_source="RIOSLaboratory OpenRPDK28 (as referenced; availability and licence "
                        "not independently verified by ChipLab) cross-checked with published "
                        "28 nm HPC/HPC+ ISSCC data",
        nominal_vdd_v=_f(0.9, "V", E),
        vth_v=_f(0.32, "V", E),
        fo4_delay_ps=_f(11.0, "ps", E),
        gate_density_kgates_mm2=_f(2000.0, "kgate/mm2", E),
        sram_bitcell_um2=_f(0.127, "um2", E, "widely reported 28 nm HD 6T cell"),
        dynamic_energy_fj_per_gate=_f(0.73, "fJ", E),
        leakage_nw_per_gate=_f(1.5, "nW", E),
        metal1_pitch_nm=_f(90.0, "nm", E),
        wire_cap_ff_per_um=_f(0.20, "fF/um", E),
        tapeout_ready=False, accuracy_margin_pct=15.0,
        license_type="unverified predictive kit + published literature",
        commercial_use_allowed=False, cost_node="N28",
        gain_cell_retention_us_25c=_f(100.0, "us", E, _GC_SRC)),
    PdkProfile(
        key="freepdk15", node_nm=15, name="NC State FreePDK15 / NanGate 15nm OCL",
        device="FinFET (predictive)",
        citation_source="NC State FreePDK15 (Bhanushali & Davis, ISPD 2015, 'FreePDK15: An "
                        "Open-Source Predictive Process Design Kit for 15nm FinFET "
                        "Technology') and the NanGate 15nm Open Cell Library",
        nominal_vdd_v=_f(0.8, "V", P, "FreePDK15 nominal supply"),
        vth_v=_f(0.30, "V", E),
        fo4_delay_ps=_f(8.0, "ps", E),
        gate_density_kgates_mm2=_f(6000.0, "kgate/mm2", E),
        sram_bitcell_um2=_f(0.07, "um2", E),
        dynamic_energy_fj_per_gate=_f(0.32, "fJ", E),
        leakage_nw_per_gate=_f(1.0, "nW", E),
        metal1_pitch_nm=_f(64.0, "nm", E),
        wire_cap_ff_per_um=_f(0.19, "fF/um", E),
        tapeout_ready=False, accuracy_margin_pct=15.0,
        license_type="academic predictive kit (see NC State licence terms)",
        commercial_use_allowed=False, cost_node="N16",
        gain_cell_retention_us_25c=_f(40.0, "us", E, _GC_SRC)),
    PdkProfile(
        key="asap7", node_nm=7, name="ASU/Arm ASAP7 (OpenROAD)", device="FinFET (predictive)",
        citation_source="L. T. Clark et al., 'ASAP7: A 7-nm finFET predictive process design "
                        "kit', Microelectronics Journal 53 (2016); The-OpenROAD-Project/asap7",
        nominal_vdd_v=_f(0.7, "V", P, "ASAP7 nominal supply"),
        vth_v=_f(0.25, "V", E),
        fo4_delay_ps=_f(6.0, "ps", E),
        gate_density_kgates_mm2=_f(10000.0, "kgate/mm2", E, "7.5-track, 270 nm cell height"),
        sram_bitcell_um2=_f(0.027, "um2", E, "typical 7 nm HD 6T cell"),
        dynamic_energy_fj_per_gate=_f(0.147, "fJ", E, "~0.3 fF x 0.7 V^2"),
        leakage_nw_per_gate=_f(0.8, "nW", E),
        metal1_pitch_nm=_f(36.0, "nm", P, "M1 pitch 36 nm (Clark et al. 2016)"),
        wire_cap_ff_per_um=_f(0.18, "fF/um", E),
        tapeout_ready=False, accuracy_margin_pct=15.0, license_type="BSD-3-Clause",
        commercial_use_allowed=True, cost_node="N7",
        notes="Licence permits commercial use of the kit, but it is predictive: no foundry "
              "manufactures ASAP7.",
        gain_cell_retention_us_25c=_f(15.0, "us", E, _GC_SRC)),
    PdkProfile(
        key="n3", node_nm=3, name="3 nm FinFET/GAA (published IMEC / ISSCC scaling data)",
        device="FinFET / GAA (predictive)",
        citation_source="Published 3 nm scaling data: IMEC roadmap papers and ISSCC/IEDM "
                        "2022-2023 reports (e.g. 0.0199-0.021 um2 HD SRAM cells); open 3 nm "
                        "predictive kits ('ASAP3') referenced but not verified by ChipLab",
        nominal_vdd_v=_f(0.72, "V", E),
        vth_v=_f(0.25, "V", E),
        fo4_delay_ps=_f(4.8, "ps", E),
        gate_density_kgates_mm2=_f(28000.0, "kgate/mm2", E),
        sram_bitcell_um2=_f(0.0199, "um2", E),
        dynamic_energy_fj_per_gate=_f(0.114, "fJ", E),
        leakage_nw_per_gate=_f(0.6, "nW", E),
        metal1_pitch_nm=_f(23.0, "nm", E),
        wire_cap_ff_per_um=_f(0.20, "fF/um", E),
        tapeout_ready=False, accuracy_margin_pct=20.0,
        license_type="published literature", commercial_use_allowed=False, cost_node="N3",
        gain_cell_retention_us_25c=_f(8.0, "us", E, _GC_SRC)),
    PdkProfile(
        key="n2", node_nm=2, name="2 nm GAA nanosheet (IRDS / IMEC / ISSCC data)",
        device="GAA nanosheet (research)",
        citation_source="IEEE IRDS More Moore roadmap (2022/2023 editions) and IMEC/ISSCC/IEDM "
                        "nanosheet scaling reports (2023-2025)",
        nominal_vdd_v=_f(0.68, "V", E),
        vth_v=_f(0.24, "V", E),
        fo4_delay_ps=_f(4.2, "ps", E),
        gate_density_kgates_mm2=_f(36000.0, "kgate/mm2", E),
        sram_bitcell_um2=_f(0.0175, "um2", E),
        dynamic_energy_fj_per_gate=_f(0.088, "fJ", E),
        leakage_nw_per_gate=_f(0.5, "nW", E),
        metal1_pitch_nm=_f(21.0, "nm", E),
        wire_cap_ff_per_um=_f(0.21, "fF/um", E),
        tapeout_ready=False, accuracy_margin_pct=25.0,
        license_type="published literature", commercial_use_allowed=False, cost_node="N2",
        gain_cell_retention_us_25c=_f(5.0, "us", E, _GC_SRC)),
]}

NUMERIC_FIELDS = ("nominal_vdd_v", "fo4_delay_ps", "gate_density_kgates_mm2",
                  "sram_bitcell_um2", "dynamic_energy_fj_per_gate", "leakage_nw_per_gate",
                  "metal1_pitch_nm", "wire_cap_ff_per_um")


def get_pdk(key: str) -> PdkProfile:
    k = key.lower()
    if k not in PDK_LADDER:
        raise KeyError(f"Unknown PDK '{key}'. Available: {sorted(PDK_LADDER)}")
    return PDK_LADDER[k]


def pdk_for_nm(nm: float | None) -> PdkProfile:
    """Nearest ladder profile for a process size (open, tape-out-ready kit preferred on ties)."""
    if nm is None:
        return PDK_LADDER["asap7"]
    order = sorted(PDK_LADDER.values(), key=lambda p: (abs(p.node_nm - float(nm)),
                                                       not p.tapeout_ready))
    return order[0]


def ladder_table() -> list[dict[str, Any]]:
    rows = []
    for p in PDK_LADDER.values():
        row: dict[str, Any] = {"key": p.key, "node_nm": p.node_nm, "name": p.name,
                               "tapeout_ready": p.tapeout_ready,
                               "accuracy_margin_pct": p.accuracy_margin_pct,
                               "license": p.license_type,
                               "commercial_use_allowed": p.commercial_use_allowed}
        for f in NUMERIC_FIELDS:
            row[f] = p.scalar(f)
        rows.append(row)
    return rows


def commercial_readiness(pdk_key: str | None, intended_use: str = "research",
                         ip_names: list[str] | None = None,
                         blackbox_licenses: dict[str, bool | None] | None = None) -> list[str]:
    """Warnings for using predictive PDKs or non-commercial IP in commercial production."""
    warnings: list[str] = []
    commercial = intended_use.lower() in ("commercial", "production", "commercial_production")
    if pdk_key:
        try:
            p = get_pdk(pdk_key)
        except KeyError as exc:
            return [str(exc)]
        if commercial and not p.tapeout_ready:
            warnings.append(f"PDK '{p.key}' ({p.name}) is predictive/literature-derived and not "
                            "tape-out ready: results carry +/-"
                            f"{p.accuracy_margin_pct:g}% error and cannot be manufactured. "
                            "Obtain the foundry PDK (NDA) before committing to production.")
        if commercial and not p.commercial_use_allowed:
            warnings.append(f"PDK '{p.key}' licence ('{p.license_type}') does not permit "
                            "commercial use.")
        if not commercial and not p.tapeout_ready:
            warnings.append(f"PDK '{p.key}' is predictive (accuracy +/-{p.accuracy_margin_pct:g}%)"
                            "; fine for exploration, not for tape-out.")
    if ip_names:
        from chiplab.library.ip_catalog import IP_CATALOG

        for name in ip_names:
            ip = IP_CATALOG.get(name)
            if ip is None:
                warnings.append(f"IP '{name}' is not in the catalog; check its licence manually.")
            elif commercial and not ip.get("commercial_use_allowed", False):
                warnings.append(f"IP '{name}' (licence {ip['license']}) is not cleared for "
                                "commercial use.")
            elif commercial and not ip.get("silicon_proven", False):
                warnings.append(f"IP '{name}' is not silicon-proven.")
    for name, ok in (blackbox_licenses or {}).items():
        if commercial and ok is not True:
            warnings.append(f"Black-box macro '{name}' has no confirmed commercial licence.")
    return warnings


def generate_sources_doc() -> str:
    """Markdown for docs/sources_and_accuracy.md (regenerated from this module)."""
    lines = [
        "# ChipLab PDK ladder: sources and accuracy",
        "",
        "Generated from `src/chiplab/library/pdk_library.py`; regenerate with "
        "`python -m chiplab.library.pdk_library`. Each number carries a provenance: "
        "**published** (stated in the cited open PDK or paper) or **estimated** (typical value "
        "consistent with the sources). Monte Carlo in the digital twin perturbs process "
        "parameters by at least the node's accuracy margin.",
        "",
        "| Node | PDK | Tape-out ready | Accuracy margin | Licence | Commercial use |",
        "|---|---|---|---|---|---|",
    ]
    for p in PDK_LADDER.values():
        lines.append(f"| {p.node_nm:g} nm | {p.name} (`{p.key}`) | "
                     f"{'yes' if p.tapeout_ready else 'no (predictive)'} | "
                     f"±{p.accuracy_margin_pct:g}% | {p.license_type} | "
                     f"{'allowed' if p.commercial_use_allowed else 'not allowed'} |")
    lines += ["", "## Parameters", "",
              "| PDK | Vdd (V) | FO4 (ps) | Density (kgate/mm²) | 6T bitcell (µm²) | "
              "E_dyn (fJ/gate) | Leakage (nW/gate) | M1 pitch (nm) | Wire C (fF/µm) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for p in PDK_LADDER.values():
        cells = []
        for f in NUMERIC_FIELDS:
            fld = getattr(p, f)
            mark = "" if fld.provenance == Provenance.PUBLISHED else "*"
            cells.append(f"{fld.value:g}{mark}")
        lines.append(f"| `{p.key}` | " + " | ".join(cells) + " |")
    lines += ["", "`*` = estimated; unmarked = published.", "", "## Sources per node", ""]
    for p in PDK_LADDER.values():
        lines += [f"### {p.node_nm:g} nm: {p.name}", "",
                  f"- **Source:** {p.citation_source}",
                  f"- **Device:** {p.device}",
                  f"- **Error bound:** ±{p.accuracy_margin_pct:g}% on timing, power and area "
                  "outputs derived from this profile",
                  f"- **Tape-out ready:** {p.tapeout_ready}; **licence:** {p.license_type}; "
                  f"**commercial use:** {p.commercial_use_allowed}"]
        for f in NUMERIC_FIELDS:
            fld = getattr(p, f)
            if fld.note:
                lines.append(f"- `{f}` = {fld.value:g} {fld.unit} ({fld.provenance.value}): "
                             f"{fld.note}")
        if p.notes:
            lines.append(f"- **Note:** {p.notes}")
        lines.append("")
    lines += ["## Gain-cell eDRAM retention", "",
              "Retention at 25 °C per node (estimated) comes from: " + _GC_SRC + ". "
              "The on-chip memory model halves retention every 18 °C (about 10x from 25 to 85 °C) and treats it as an "
              "'assumed' parameter (±25% in Monte Carlo).", ""]
    for p in PDK_LADDER.values():
        lines.append(f"- `{p.key}`: {p.gain_cell_retention_us_25c.value:g} µs")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    from pathlib import Path

    out = Path(__file__).resolve().parents[3] / "docs" / "sources_and_accuracy.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(generate_sources_doc())
    print(f"wrote {out}")
