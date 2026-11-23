"""Unit-cost model: die yield and cost per node, bought memory, packaging, test, board, NRE.

Formulas
- Dies per wafer: exact grid placement of square dies (+ scribe) inside the usable
  diameter (edge exclusion), best of 8x8 placement offsets.
- Die yield: negative binomial, Y = (1 + A*D0/alpha)^-alpha, A in cm2, alpha = 3.
- Known-good-die cost: wafer cost * (1 + sort fraction) / (dies per wafer * yield).
- Package: base + interposer/substrate $/mm2 * area + per-die attach; divided by the
  assembly yield (a failed assembly scraps every die in it).
Scenarios: 'value' = typical, 'low' = optimistic, 'high' = pessimistic (every input at the
end of its range that raises cost).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from chiplab.schemas.common import unwrap_value
from chiplab.schemas.design import ChipletSpec, DesignSpec

from . import prices
from .reference import memory_tech_for, node_for_nm, package_key

WAFER_DIAMETER_MM = 300.0
EDGE_EXCLUSION_MM = 3.0
SCRIBE_MM = 0.1
RETICLE_LIMIT_MM2 = 858.0
YIELD_ALPHA = 3.0
DEFAULT_LOGIC_AREA_MM2 = 100.0
HBM_STACK_FOOTPRINT_MM2 = 110.0

# Inputs where a *higher* number means *lower* cost.
_INVERTED = ("assembly_yield", "array_efficiency")


def _num(v: Any, default: float | None = None) -> float | None:
    return default if v is None else float(unwrap_value(v))


def _pv(key: str, scenario: str) -> float:
    if scenario == "value":
        return prices.value(key)
    pessimistic = scenario == "high"
    if any(key.endswith(s) for s in _INVERTED):
        pessimistic = not pessimistic
    return prices.value(key, "high" if pessimistic else "low")


@lru_cache(maxsize=4096)
def dies_per_wafer(die_area_mm2: float, wafer_diameter_mm: float = WAFER_DIAMETER_MM) -> int:
    """Gross dies per wafer: exact count of square dies (plus scribe) that fit entirely inside
    the usable wafer circle, best of an 8x8 grid of placement offsets.

    The closed-form approximation pi*d^2/(4S) - pi*d/sqrt(2S) undercounts large dies (by ~9%
    at 400 mm2 and ~12% at 800 mm2), which overstated reticle-class die cost."""
    if die_area_mm2 <= 0:
        raise ValueError("die area must be positive")
    import numpy as np
    s = math.sqrt(die_area_mm2) + SCRIBE_MM
    r = wafer_diameter_mm / 2 - EDGE_EXCLUSION_MM
    n = int(2 * r // s) + 2
    steps = 8
    best = 0
    for ox in range(steps):
        lo_x = -r + ox * s / steps + s * np.arange(n)
        mx2 = np.maximum(lo_x ** 2, (lo_x + s) ** 2)
        mx2 = mx2[lo_x + s <= r]
        for oy in range(steps):
            lo_y = -r + oy * s / steps + s * np.arange(n)
            my2 = np.maximum(lo_y ** 2, (lo_y + s) ** 2)
            my2 = my2[lo_y + s <= r]
            best = max(best, int((mx2[:, None] + my2[None, :] <= r * r).sum()))
    return best


def die_yield(die_area_mm2: float, d0_per_cm2: float, alpha: float = YIELD_ALPHA) -> float:
    return (1.0 + (die_area_mm2 / 100.0) * d0_per_cm2 / alpha) ** (-alpha)


def sram_area_mm2(sram_mb: float, node: str, scenario: str = "value") -> float:
    bits = sram_mb * 8 * 2**20
    bitcell = _pv(f"sram_bitcell.tsmc.{node}", scenario)
    return bits * bitcell / 1e6 / _pv("sram.array_efficiency", scenario)


def _logic_die(ch: ChipletSpec, design: DesignSpec, scenario: str,
               notes: list[str], warnings: list[str]) -> dict[str, Any]:
    node = node_for_nm(_num(ch.process_nm))
    if ch.process_nm is None:
        notes.append(f"{ch.id or ch.type}: process_nm missing, assumed TSMC {node}.")
    sram_mb = _num(ch.compute.sram_mb, 0.0) if ch.compute else 0.0
    sram_mm2 = sram_area_mm2(sram_mb, node, scenario) if sram_mb else 0.0
    area = _num(ch.area_mm2)
    if area is None:
        area = DEFAULT_LOGIC_AREA_MM2 + sram_mm2
        notes.append(f"{ch.id or ch.type}: area_mm2 missing, assumed {area:.0f} mm2 "
                     f"({DEFAULT_LOGIC_AREA_MM2:.0f} logic + {sram_mm2:.0f} SRAM).")
    elif sram_mm2 > 0.9 * area:
        warnings.append(f"{ch.id or ch.type}: {sram_mb:g} MB SRAM needs ~{sram_mm2:.0f} mm2 on "
                        f"{node}, but the die is only {area:.0f} mm2. Grow the die or cut SRAM.")
    if area > RETICLE_LIMIT_MM2:
        warnings.append(f"{ch.id or ch.type}: {area:.0f} mm2 exceeds the ~{RETICLE_LIMIT_MM2:.0f} "
                        "mm2 reticle limit; it cannot be made as one die. Split it into chiplets.")
    d0 = _num(design.package.yield_defect_density_per_cm2)
    if d0 is None:
        d0 = _pv(f"defect_density.tsmc.{node}", scenario)
    wafer = _pv(f"wafer.tsmc.{node}", scenario)
    sort = _pv("test.wafer_sort_fraction", scenario)
    dpw = dies_per_wafer(area)
    y = die_yield(area, d0)
    good = dpw * y
    kgd = wafer * (1 + sort) / good if good > 0 else float("inf")
    return {"kind": "logic", "node": f"TSMC {node}", "area_mm2": round(area, 1),
            "sram_mb": sram_mb, "sram_area_mm2": round(sram_mm2, 1),
            "wafer_cost_usd": wafer, "defect_density_per_cm2": d0, "dies_per_wafer": dpw,
            "yield": round(y, 4), "good_dies_per_wafer": round(good, 1),
            "cost_per_good_die_usd": round(kgd, 2),
            "mask_set_usd": _pv(f"mask_set.tsmc.{node}", scenario)}


def _memory_part(ch: ChipletSpec, tech: str, scenario: str,
                 warnings: list[str]) -> dict[str, Any]:
    cap = _num(ch.memory.capacity_per_chiplet_gb) if ch.memory else None
    if not cap:
        warnings.append(f"{ch.id or ch.type}: {tech} part has no capacity_per_chiplet_gb; "
                        "costed as $0.")
        cap = 0.0
    per_gb = _pv(f"memory.{tech}", scenario)
    footprint = _num(ch.area_mm2) or (HBM_STACK_FOOTPRINT_MM2 if tech.startswith("HBM") else 0.0)
    return {"kind": "memory", "technology": tech, "capacity_gb": cap,
            "usd_per_gb": per_gb, "cost_per_part_usd": round(cap * per_gb, 2),
            "footprint_mm2": footprint}


def estimate_unit_cost(design: DesignSpec, volume: int = 100_000, scenario: str = "value",
                       board: dict[str, Any] | None = None) -> dict[str, Any]:
    """Estimate per-unit cost of a design. ``board`` adds a PCB/VRM/cooling estimate:
    {"area_cm2": 300, "layers": 12, "hdi": False, "high_speed": True, "units_per_board": 1}.
    """
    from ..schemas.design import plain_design
    design, _ = plain_design(design)
    if scenario not in ("value", "low", "high"):
        raise ValueError("scenario must be 'value', 'low' or 'high'")
    if volume <= 0:
        raise ValueError("volume must be a positive number of units")
    notes: list[str] = []
    warnings: list[str] = []
    lines: list[dict[str, Any]] = []
    silicon = memory = nre = 0.0
    n_dies = 0
    die_area_total = 0.0
    tdp_total = 0.0
    pkg = package_key(design.package.type, design.package.interposer)

    for ch in design.chiplets:
        count = int(_num(ch.count, 1) or 1)
        n_dies += count
        tdp_total += (_num(ch.tdp_watts, 0.0) or 0.0) * count
        tech = memory_tech_for(ch.memory.technology if ch.memory else None) or \
            memory_tech_for(ch.type)
        if tech:
            part = _memory_part(ch, tech, scenario, warnings)
            memory += part["cost_per_part_usd"] * count
            die_area_total += part["footprint_mm2"] * count
            if tech.startswith("HBM") and pkg in ("fcbga", "organic"):
                warnings.append(f"{ch.id or ch.type}: {tech} needs a 2.5D interposer or bridge "
                                f"(CoWoS/EMIB); package '{design.package.type}' cannot host it.")
            line = part
        else:
            die = _logic_die(ch, design, scenario, notes, warnings)
            silicon += die["cost_per_good_die_usd"] * count
            die_area_total += die["area_mm2"] * count
            nre += die["mask_set_usd"]
            line = die
        line.update({"id": ch.id or ch.type, "count": count,
                     "subtotal_usd": round((line.get("cost_per_good_die_usd")
                                            or line.get("cost_per_part_usd") or 0.0) * count, 2)})
        lines.append(line)

    area_factor = 1.2 if pkg in ("cowos_s", "cowos_l", "soic") else 3.0
    pkg_area = _num(design.package.max_area_mm2) or die_area_total * area_factor
    if pkg_area > 3 * RETICLE_LIMIT_MM2 and pkg == "cowos_s":
        warnings.append(f"Interposer ~{pkg_area:.0f} mm2 exceeds CoWoS-S (~3.3x reticle); "
                        "use CoWoS-L or EMIB.")
    package = (_pv(f"package.{pkg}.base", scenario)
               + _pv(f"package.{pkg}.area_cost", scenario) * pkg_area
               + _pv(f"package.{pkg}.per_die", scenario) * n_dies)
    asm_yield = _pv(f"package.{pkg}.assembly_yield", scenario) * 0.995 ** max(0, n_dies - 2)
    final_test = _pv("test.final_per_package", scenario)
    assembled = (silicon + memory + package) / asm_yield + final_test

    board_cost = 0.0
    board_detail: dict[str, Any] | None = None
    if board:
        area_cm2 = float(board.get("area_cm2", 200.0))
        layers = int(board.get("layers", 10))
        per_board = max(1, int(board.get("units_per_board", 1)))
        fab = area_cm2 * layers * _pv("pcb.per_cm2_per_layer", scenario)
        if board.get("hdi"):
            fab *= _pv("pcb.hdi_multiplier", scenario)
        if board.get("high_speed", True):
            fab *= _pv("pcb.high_speed_material_multiplier", scenario)
        vrm = tdp_total * per_board * _pv("pcb.vrm_per_watt", scenario)
        cooling = tdp_total * per_board * _pv("pcb.cooling_per_watt", scenario)
        asm = _pv("pcb.assembly_per_board", scenario)
        board_cost = (fab + vrm + cooling + asm) / per_board
        board_detail = {"pcb_fab_usd": round(fab, 2), "vrm_usd": round(vrm, 2),
                        "cooling_usd": round(cooling, 2), "assembly_usd": asm,
                        "per_unit_usd": round(board_cost, 2), "layers": layers,
                        "area_cm2": area_cm2}

    nre_per_unit = nre / volume
    unit = assembled + board_cost
    out = {
        "scenario": scenario,
        "volume": volume,
        "package_type": pkg,
        "package_area_mm2": round(pkg_area, 1),
        "chiplets": lines,
        "breakdown_usd": {
            "logic_silicon": round(silicon, 2),
            "memory": round(memory, 2),
            "package": round(package, 2),
            "assembly_yield_loss": round((silicon + memory + package) * (1 / asm_yield - 1), 2),
            "final_test": round(final_test, 2),
            "board": round(board_cost, 2),
            "nre_amortized": round(nre_per_unit, 2),
        },
        "assembly_yield": round(asm_yield, 4),
        "mask_nre_usd": round(nre, 0),
        "unit_cost_excl_nre_usd": round(unit, 2),
        "unit_cost_usd": round(unit + nre_per_unit, 2),
        "notes": notes,
        "warnings": warnings,
    }
    if design.constraints.max_cost_usd is not None:
        limit = _num(design.constraints.max_cost_usd) or 0.0
        if out["unit_cost_usd"] > limit:
            warnings.append(f"Unit cost ${out['unit_cost_usd']:,.0f} exceeds max_cost_usd "
                            f"${limit:,.0f}.")
    return out


def estimate_cost_range(design: DesignSpec, volume: int = 100_000,
                        board: dict[str, Any] | None = None) -> dict[str, Any]:
    """Typical estimate plus optimistic/pessimistic totals and price-data freshness."""
    typical = estimate_unit_cost(design, volume, "value", board)
    low = estimate_unit_cost(design, volume, "low", board)
    high = estimate_unit_cost(design, volume, "high", board)
    used = _keys_used(typical)
    stale = [k for k in used if prices.get(k)["stale"]]
    typical["range_usd"] = {"optimistic": low["unit_cost_usd"],
                            "typical": typical["unit_cost_usd"],
                            "pessimistic": high["unit_cost_usd"]}
    typical["price_data"] = {
        "keys_used": used,
        "stale_keys": stale,
        "advice": ("Some prices are older than "
                   f"{prices.STALE_DAYS} days. Call chip_price_lookup_plan, search the web, "
                   "then chip_set_price with the source URL.") if stale else "Prices are fresh.",
    }
    return typical


def _keys_used(result: dict[str, Any]) -> list[str]:
    keys = set()
    for line in result["chiplets"]:
        if line["kind"] == "memory":
            keys.add(f"memory.{line['technology']}")
        else:
            node = line["node"].split()[-1]
            keys.update({f"wafer.tsmc.{node}", f"defect_density.tsmc.{node}",
                         f"mask_set.tsmc.{node}"})
            if line["sram_mb"]:
                keys.add(f"sram_bitcell.tsmc.{node}")
    pkg = result["package_type"]
    keys.update({f"package.{pkg}.area_cost", f"package.{pkg}.base"})
    if result["breakdown_usd"]["board"]:
        keys.add("pcb.per_cm2_per_layer")
    return sorted(keys)
