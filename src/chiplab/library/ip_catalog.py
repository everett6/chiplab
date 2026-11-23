"""Open-source IP presets (cores, SRAM, interconnect) and black-box macro examples.

Provenance is explicit per number: 'published' only where the figure appears in the
project's own paper/announcement (cited in ``source``); everything else is 'estimated'.
Check the repository before relying on a number; versions and configurations vary.
"""

from __future__ import annotations

from typing import Any

from chiplab.schemas.common import Provenance
from chiplab.schemas.digital_twin import BlackBoxMacro, PinTiming, PowerRail


def _m(value: float, unit: str, provenance: Provenance, source: str,
       node: str | None = None) -> dict[str, Any]:
    return {"value": value, "unit": unit, "provenance": provenance.value, "source": source,
            "node": node}


P, E = Provenance.PUBLISHED, Provenance.ESTIMATED

IP_CATALOG: dict[str, dict[str, Any]] = {
    "cva6": {
        "commercial_use_allowed": True,
        "kind": "core", "isa": "RV64GC", "pipeline": "6-stage in-order, Linux-capable",
        "repo": "https://github.com/openhwgroup/cva6", "license": "Solderpad-2.1",
        "silicon_proven": True,
        "silicon_notes": "Taped out as Ariane in GlobalFoundries 22FDX (ETH Zurich/PULP).",
        "metrics": {
            "freq_ghz": _m(1.7, "GHz", P, "Zaruba & Benini, IEEE TVLSI 2019 ('Linux-ready "
                           "1.7GHz 64bit RISC-V core in 22nm FDSOI')", "GF22FDX"),
            "core_area_mm2": _m(0.5, "mm2", E, "estimate incl. L1 caches", "GF22FDX"),
        },
        "twin_defaults": {"vlen": 128, "dlen": 64, "l1_kb": 32},
    },
    "ara": {
        "commercial_use_allowed": True,
        "kind": "vector_unit", "isa": "RVV 1.0", "pipeline": "lane-based vector coprocessor "
        "for CVA6", "repo": "https://github.com/pulp-platform/ara", "license": "Solderpad-0.51",
        "silicon_proven": True, "silicon_notes": "Fabricated in GF22FDX by the PULP team.",
        "metrics": {"lanes": _m(4, "lanes", E, "typical configuration (2-16 supported)")},
        "twin_defaults": {"vlen": 4096, "dlen": 256},
    },
    "rocket": {
        "commercial_use_allowed": True,
        "kind": "core", "isa": "RV64GC", "pipeline": "5-stage in-order",
        "repo": "https://github.com/chipsalliance/rocket-chip", "license": "Apache-2.0 / BSD-3",
        "silicon_proven": True,
        "silicon_notes": "Many tapeouts (UC Berkeley Raven/Hurricane/BROOM, commercial SoCs).",
        "metrics": {
            "freq_ghz": _m(1.0, "GHz", E, "typical 28nm-class implementation", "28nm"),
            "core_area_mm2": _m(0.4, "mm2", E, "estimate incl. 16 KiB L1s", "28nm"),
        },
        "twin_defaults": {"vlen": 128, "dlen": 64, "l1_kb": 16},
    },
    "xiangshan": {
        "commercial_use_allowed": True,
        "kind": "core", "isa": "RV64GCBK(V)", "pipeline": "superscalar out-of-order",
        "repo": "https://github.com/OpenXiangShan/XiangShan", "license": "MulanPSL-2.0",
        "silicon_proven": True,
        "silicon_notes": "Nanhu generation taped out in a 14nm-class process (team reports).",
        "metrics": {
            "freq_ghz": _m(2.0, "GHz", E, "Nanhu target frequency per project announcements",
                           "14nm"),
            "core_area_mm2": _m(3.0, "mm2", E, "rough OoO core + L2 estimate", "14nm"),
        },
        "twin_defaults": {"l1_kb": 64, "l2_kb": 1024},
    },
    "ibex": {
        "commercial_use_allowed": True,
        "kind": "core", "isa": "RV32IMC", "pipeline": "2/3-stage in-order microcontroller",
        "repo": "https://github.com/lowRISC/ibex", "license": "Apache-2.0",
        "silicon_proven": True, "silicon_notes": "Used in OpenTitan silicon.",
        "metrics": {"core_area_kge": _m(25.0, "kGE", E, "small configuration estimate")},
        "twin_defaults": {"vlen": 128, "dlen": 64, "l1_kb": 16},
    },
    "openram": {
        "commercial_use_allowed": True,
        "kind": "sram_compiler", "isa": None, "pipeline": "SRAM macro compiler",
        "repo": "https://github.com/VLSIDA/OpenRAM", "license": "BSD-3-Clause",
        "silicon_proven": True,
        "silicon_notes": "Macros fabricated on SkyWater sky130 shuttles; FreePDK45 is "
                         "predictive only.",
        "metrics": {"nodes": _m(130, "nm", P, "sky130 support in OpenRAM docs", "sky130")},
    },
    "basejump_stl": {
        "commercial_use_allowed": True,
        "kind": "library", "isa": None, "pipeline": "SystemVerilog building blocks (FIFOs, "
        "NoC, caches, DRAM ctrl)", "repo": "https://github.com/bespoke-silicon-group/basejump_stl",
        "license": "BSD-style (see repository)", "silicon_proven": True,
        "silicon_notes": "Used in Celerity and other Bespoke Silicon Group tapeouts.",
        "metrics": {},
    },
    "pulp_axi": {
        "commercial_use_allowed": True,
        "kind": "interconnect", "isa": None, "pipeline": "AXI4/AXI4-Lite crossbars, muxes, "
        "CDC, DMA", "repo": "https://github.com/pulp-platform/axi", "license": "Solderpad-0.51",
        "silicon_proven": True, "silicon_notes": "Used across PULP silicon.",
        "metrics": {},
    },
    "tilelink": {
        "commercial_use_allowed": True,
        "kind": "interconnect", "isa": None, "pipeline": "TileLink cache-coherent fabric "
        "(rocket-chip diplomacy)", "repo": "https://github.com/chipsalliance/rocket-chip",
        "license": "Apache-2.0 / BSD-3", "silicon_proven": True,
        "silicon_notes": "Shipped in rocket-chip based SoCs.", "metrics": {},
    },
    "ucie_open_adapter": {
        "commercial_use_allowed": False,
        "kind": "interconnect", "isa": None,
        "pipeline": "UCIe die-to-die adapter (behavioral / FPGA models only)",
        "repo": None, "license": "varies", "silicon_proven": False,
        "silicon_notes": "No silicon-proven open-source UCIe PHY is known; model the PHY as a "
                         "BlackBoxMacro from a vendor datasheet (see 'example_ucie_phy_x16').",
        "metrics": {},
    },
}

BLACKBOX_EXAMPLES: dict[str, BlackBoxMacro] = {
    "example_ucie_phy_x16": BlackBoxMacro(
        name="example_ucie_phy_x16", vendor="example (illustrative numbers)", kind="phy",
        width_um=1100.0, height_um=400.0, keepout_um=20.0, max_freq_ghz=2.0,
        pins=[PinTiming(name="tx_data", direction="input", clock="lclk", setup_ps=80.0,
                        hold_ps=20.0, width=256),
              PinTiming(name="rx_data", direction="output", clock="lclk", clk_to_q_ps=150.0,
                        width=256)],
        rails=[PowerRail(name="VDDA", voltage_v=0.75, typical_mw=180.0),
               PowerRail(name="VDDQ", voltage_v=0.4, typical_mw=80.0)],
        provenance=Provenance.ASSUMED,
        datasheet="Illustrative values; replace with your vendor's datasheet."),
    "example_lpddr5x_phy": BlackBoxMacro(
        name="example_lpddr5x_phy", vendor="example (illustrative numbers)", kind="phy",
        width_um=2400.0, height_um=600.0, keepout_um=30.0, max_freq_ghz=1.1,
        pins=[PinTiming(name="dfi_wrdata", direction="input", clock="dfi_clk", setup_ps=120.0,
                        width=128),
              PinTiming(name="dfi_rddata", direction="output", clock="dfi_clk",
                        clk_to_q_ps=200.0, width=128)],
        rails=[PowerRail(name="VDD", voltage_v=0.75, typical_mw=250.0),
               PowerRail(name="VDDQ", voltage_v=0.5, typical_mw=150.0)],
        provenance=Provenance.ASSUMED,
        datasheet="Illustrative values; replace with your vendor's datasheet."),
}


def list_ip(kind: str | None = None) -> list[dict[str, Any]]:
    return [{"name": k, **v} for k, v in IP_CATALOG.items() if kind in (None, v["kind"])]


def get_ip(name: str) -> dict[str, Any]:
    if name not in IP_CATALOG:
        raise KeyError(f"Unknown IP '{name}'. Available: {sorted(IP_CATALOG)}")
    return {"name": name, **IP_CATALOG[name]}


def get_blackbox(name_or_spec: str | dict[str, Any] | BlackBoxMacro) -> BlackBoxMacro:
    if isinstance(name_or_spec, BlackBoxMacro):
        return name_or_spec
    if isinstance(name_or_spec, dict):
        return BlackBoxMacro.model_validate(name_or_spec)
    if name_or_spec not in BLACKBOX_EXAMPLES:
        raise KeyError(f"Unknown black-box preset '{name_or_spec}'. "
                       f"Available: {sorted(BLACKBOX_EXAMPLES)}")
    return BLACKBOX_EXAMPLES[name_or_spec]
