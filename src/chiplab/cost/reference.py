"""Built-in cost reference table for chip, memory, packaging and board estimates.

Every entry carries a typical value plus a low/high range, the unit, the date the figure
was compiled, and where it comes from. Foundries and memory makers do not publish list
prices, so most figures are industry estimates (analyst reports and press coverage).
Treat them as starting points: use ``chip_set_price`` (or ``prices.set_override``) to
replace any value with a fresher, sourced number.
"""

from __future__ import annotations

from typing import Any

BUILTIN_AS_OF = "2025-06"

_EST = "estimated"
_PUB = "published"


def _e(value: float, low: float, high: float, unit: str, source: str,
       provenance: str = _EST, as_of: str = BUILTIN_AS_OF, query: str = "") -> dict[str, Any]:
    return {
        "value": value, "low": low, "high": high, "unit": unit, "source": source,
        "provenance": provenance, "as_of": as_of, "lookup_query": query,
    }


_WAFER_SRC = ("Industry wafer-price estimates (CSET 2020 model, DigiTimes/TrendForce/"
              "Tom's Hardware reporting); TSMC does not publish list prices")
_D0_SRC = ("TSMC public defect-density disclosures at symposia (mature ramp values); "
           "early-ramp lots run higher")
_SRAM_SRC = "TSMC/IEDM/ISSCC published high-density 6T bitcell areas"
_MEM_SRC = "DRAM contract/spot price reporting (TrendForce/DRAMeXchange, analyst notes)"
_PKG_SRC = "Advanced-packaging cost estimates (SemiAnalysis, Yole, press coverage)"
_PCB_SRC = "Volume PCB fabrication rules of thumb from fab quote calculators"

# Nodes: wafer price (300 mm), mature defect density, HD SRAM bitcell, mask-set NRE.
_NODES: dict[str, tuple[tuple[float, float, float], float, float, tuple[float, float, float]]] = {
    #        wafer $ (typ, low, high)      D0/cm2  bitcell um2  mask set $M (typ, low, high)
    # Mature nodes (180/130/65 nm): estimates for 300 mm-equivalent wafers; MPW shuttles
    # (e.g. sky130 / GF180 / IHP open-PDK runs) are priced per seat instead.
    "N180": ((700, 500, 1_000), 0.03, 4.65, (0.12, 0.08, 0.2)),
    "N130": ((1_000, 750, 1_400), 0.04, 2.0, (0.25, 0.15, 0.4)),
    "N65": ((1_800, 1_400, 2_300), 0.05, 0.52, (0.6, 0.4, 0.8)),
    "N40": ((2_600, 2_000, 3_200), 0.05, 0.242, (0.9, 0.6, 1.2)),
    "N28": ((3_000, 2_500, 3_800), 0.05, 0.127, (1.5, 1.0, 2.0)),
    "N22": ((3_400, 2_800, 4_200), 0.05, 0.110, (2.0, 1.5, 2.5)),
    "N16": ((4_000, 3_500, 5_000), 0.06, 0.074, (4.0, 3.0, 5.0)),
    "N12": ((4_200, 3_600, 5_200), 0.06, 0.074, (4.5, 3.5, 5.5)),
    "N7":  ((9_500, 8_500, 10_500), 0.07, 0.027, (10.0, 8.0, 15.0)),
    "N6":  ((10_000, 9_000, 11_000), 0.07, 0.027, (10.0, 8.0, 15.0)),
    "N5":  ((16_000, 14_000, 18_000), 0.07, 0.021, (15.0, 12.0, 20.0)),
    "N4":  ((17_000, 15_000, 19_000), 0.07, 0.021, (16.0, 12.0, 22.0)),
    "N3":  ((20_000, 18_000, 23_000), 0.09, 0.0199, (25.0, 20.0, 35.0)),
    "N2":  ((30_000, 27_000, 33_000), 0.12, 0.0175, (35.0, 28.0, 45.0)),
    "A16": ((45_000, 38_000, 50_000), 0.15, 0.0175, (45.0, 35.0, 60.0)),
}

# Memory parts bought per GB (not fabbed by you).
_MEMORY: dict[str, tuple[float, float, float, float]] = {
    #            $/GB typ, low, high, GB/s per device or stack (typical)
    "DDR4":     (2.0, 1.2, 4.0, 25.6),
    "DDR5":     (4.0, 2.5, 10.0, 51.2),
    "LPDDR5":   (4.5, 3.0, 9.0, 51.2),
    "LPDDR5X":  (5.0, 3.5, 10.0, 68.0),
    "GDDR6":    (6.0, 3.5, 10.0, 64.0),
    "GDDR7":    (10.0, 7.0, 15.0, 128.0),
    "HBM2E":    (10.0, 7.0, 14.0, 460.0),
    "HBM3":     (15.0, 10.0, 20.0, 819.0),
    "HBM3E":    (20.0, 15.0, 28.0, 1_200.0),
    "HBM4":     (30.0, 22.0, 40.0, 2_000.0),
}

# Packaging: interposer/substrate $/mm2, fixed assembly $, per-die bond $, assembly yield.
_PACKAGES: dict[str, dict[str, tuple[float, float, float]]] = {
    "organic":  {"area_cost": (0.030, 0.020, 0.050), "base": (5.0, 3.0, 10.0),
                 "per_die": (1.0, 0.5, 2.0), "yield": (0.995, 0.99, 0.999)},
    "fcbga":    {"area_cost": (0.040, 0.025, 0.070), "base": (8.0, 5.0, 15.0),
                 "per_die": (1.5, 1.0, 3.0), "yield": (0.99, 0.985, 0.998)},
    "emib":     {"area_cost": (0.10, 0.07, 0.15), "base": (25.0, 15.0, 40.0),
                 "per_die": (4.0, 2.0, 6.0), "yield": (0.98, 0.96, 0.99)},
    "cowos_l":  {"area_cost": (0.18, 0.12, 0.25), "base": (60.0, 40.0, 90.0),
                 "per_die": (6.0, 4.0, 10.0), "yield": (0.97, 0.94, 0.985)},
    "cowos_s":  {"area_cost": (0.30, 0.20, 0.45), "base": (50.0, 30.0, 80.0),
                 "per_die": (6.0, 4.0, 10.0), "yield": (0.97, 0.94, 0.985)},
    "soic":     {"area_cost": (0.25, 0.15, 0.40), "base": (40.0, 25.0, 70.0),
                 "per_die": (10.0, 6.0, 15.0), "yield": (0.95, 0.90, 0.98)},
}

PACKAGE_ALIASES = {
    "2d": "fcbga", "flip_chip": "fcbga", "fcbga": "fcbga", "organic": "organic",
    "mcm": "organic", "2.5d": "cowos_s", "cowos": "cowos_s", "cowos-s": "cowos_s",
    "cowos_s": "cowos_s", "silicon": "cowos_s", "cowos-l": "cowos_l", "cowos_l": "cowos_l",
    "rdl": "cowos_l", "info": "cowos_l", "emib": "emib", "bridge": "emib",
    "3d": "soic", "soic": "soic", "hybrid_bonding": "soic",
}


def _build() -> dict[str, dict[str, Any]]:
    ref: dict[str, dict[str, Any]] = {}
    for node, (wafer, d0, bitcell, mask) in _NODES.items():
        ref[f"wafer.tsmc.{node}"] = _e(
            *wafer, "USD per 300mm wafer", _WAFER_SRC,
            query=f"TSMC {node} wafer price 2026")
        ref[f"defect_density.tsmc.{node}"] = _e(
            d0, d0 * 0.7, d0 * 2.0, "defects per cm2", _D0_SRC,
            query=f"TSMC {node} defect density D0")
        ref[f"sram_bitcell.tsmc.{node}"] = _e(
            bitcell, bitcell, bitcell * 1.2, "um2 per bit (HD 6T)", _SRAM_SRC, _PUB,
            query=f"TSMC {node} SRAM bitcell area")
        ref[f"mask_set.tsmc.{node}"] = _e(
            mask[0] * 1e6, mask[1] * 1e6, mask[2] * 1e6, "USD per full mask set", _WAFER_SRC,
            query=f"TSMC {node} mask set cost")
    for tech, (typ, low, high, bw) in _MEMORY.items():
        ref[f"memory.{tech}"] = _e(
            typ, low, high, "USD per GB", _MEM_SRC, query=f"{tech} price per GB contract 2026")
        ref[f"memory_bw.{tech}"] = _e(
            bw, bw, bw, "GB/s per device or stack", "JEDEC standard / vendor datasheets", _PUB)
    for pkg, vals in _PACKAGES.items():
        ref[f"package.{pkg}.area_cost"] = _e(
            *vals["area_cost"], "USD per mm2 interposer/substrate", _PKG_SRC,
            query=f"{pkg.replace('_', '-')} packaging cost per mm2")
        ref[f"package.{pkg}.base"] = _e(*vals["base"], "USD per package", _PKG_SRC)
        ref[f"package.{pkg}.per_die"] = _e(*vals["per_die"], "USD per die attached", _PKG_SRC)
        ref[f"package.{pkg}.assembly_yield"] = _e(*vals["yield"], "fraction", _PKG_SRC)
    ref.update({
        "sram.array_efficiency": _e(0.45, 0.35, 0.60, "bitcell area / cache area",
                                    "Compiled SRAM macro + tag/periphery overhead; e.g. 64 MB "
                                    "N7 V-Cache die is ~41 mm2", _EST),
        "test.wafer_sort_fraction": _e(0.03, 0.01, 0.06, "fraction of wafer cost",
                                       "ATE cost rules of thumb"),
        "test.final_per_package": _e(5.0, 2.0, 25.0, "USD per package",
                                     "ATE cost rules of thumb"),
        "pcb.per_cm2_per_layer": _e(0.008, 0.004, 0.02, "USD per cm2 per layer (volume)",
                                    _PCB_SRC, query="PCB price per square cm per layer volume"),
        "pcb.hdi_multiplier": _e(1.6, 1.3, 2.2, "x", _PCB_SRC),
        "pcb.high_speed_material_multiplier": _e(1.8, 1.4, 2.5, "x (Megtron/Tachyon class)",
                                                 _PCB_SRC),
        "pcb.assembly_per_board": _e(15.0, 5.0, 60.0, "USD per board (volume SMT)", _PCB_SRC),
        "pcb.vrm_per_watt": _e(0.10, 0.05, 0.20, "USD per watt delivered",
                               "Power-stage BOM rules of thumb"),
        "pcb.cooling_per_watt": _e(0.08, 0.03, 0.20, "USD per watt (heatsink/cold plate)",
                                   "Thermal solution BOM rules of thumb"),
    })
    return ref


BUILTIN_REFERENCE: dict[str, dict[str, Any]] = _build()

NODES = tuple(_NODES)
MEMORY_TECHS = tuple(_MEMORY)
PACKAGES = tuple(_PACKAGES)


def node_for_nm(process_nm: int | float | str | None) -> str:
    """Map a process size (nm int or a node name like 'N3E') to the nearest table node."""
    if process_nm is None:
        return "N7"
    if isinstance(process_nm, str):
        s = process_nm.strip().upper().replace("NM", "")
        if s.startswith("A16"):
            return "A16"
        s = s.lstrip("N").rstrip("PEXLA") or "7"
        process_nm = float(s)
    nm = float(process_nm)
    if nm < 2:
        return "A16"
    numeric = {"N180": 180, "N130": 130, "N65": 65, "N40": 40, "N28": 28, "N22": 22, "N16": 16, "N12": 12, "N7": 7, "N6": 6,
               "N5": 5, "N4": 4, "N3": 3, "N2": 2}
    return min(numeric, key=lambda k: (abs(numeric[k] - nm), numeric[k]))


def memory_tech_for(name: str | None) -> str | None:
    """Return the table key for a memory technology name, or None if it is not a bought part."""
    if not name:
        return None
    s = name.upper().replace("-", "").replace("_", "").replace(" ", "")
    for tech in sorted(_MEMORY, key=len, reverse=True):
        if s.startswith(tech):
            return tech
    return None


def package_key(package_type: str | None, interposer: str | None = None) -> str:
    """Map a design's package type/interposer to a packaging table key."""
    ptype = PACKAGE_ALIASES.get(str(package_type or "").strip().lower())
    if ptype in ("fcbga", "organic", "soic", "emib", "cowos_l"):
        return ptype  # package type is decisive; ignore the (defaulted) interposer field
    inter = PACKAGE_ALIASES.get(str(interposer or "").strip().lower())
    return inter or ptype or "fcbga"
