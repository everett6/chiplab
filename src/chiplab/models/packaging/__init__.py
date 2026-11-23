from __future__ import annotations

import math

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from chiplab.schemas.simulation import AssumptionRecord
from chiplab.schemas.common import Provenance
from chiplab.schemas.design import DesignSpec


def estimate_packaging(
    design: DesignSpec,
    chiplet_areas_mm2: List[float],
    allow_defaults: bool = True,
) -> tuple[dict[str, float], list[AssumptionRecord], list[str]]:
    """Estimate packaging parameters based on design and chiplet areas.

    Handles monolithic, chiplets, 2.5D, 3D_stacking packaging types.

    Computes:
        - total_area_mm2: total package area in mm²
        - interposer_area_mm2: interposer area in mm²
        - yield_pct: manufacturing yield percentage
        - estimated_cost_usd: estimated manufacturing cost in USD

    Args:
        design: DesignSpec containing package configuration
        chiplet_areas_mm2: list of chiplet areas in mm²
        allow_defaults: whether to use component library defaults when params are missing

    Returns:
        tuple of (metrics_dict, assumption_records, warnings)
    """
    assumptions: list[AssumptionRecord] = []
    warnings: list[str] = []

    package_type = design.package.type

    # Get defect density from design constraints or package spec
    defect_density_per_cm2 = design.constraints.thermal_resistance_c_per_w  # placeholder, will use proper field
    # Actually look at package spec yield_defect_density_per_cm2
    pkg_yield_density = design.package.yield_defect_density_per_cm2

    if pkg_yield_density is not None:
        defect_density_per_cm2 = pkg_yield_density
        assumptions.append(
            AssumptionRecord(
                parameter="yield_defect_density_per_cm2",
                value=pkg_yield_density,
                unit="defects/cm²",
                provenance=Provenance.MEASURED,
                source="design_package_spec",
            )
        )
    elif allow_defaults:
        # Use defaults based on package type
        if package_type == "2.5D":
            defect_density_per_cm2 = 0.1  # 0.1 defects/cm² typical for silicon interposer
            assumptions.append(
                AssumptionRecord(
                    parameter="yield_defect_density_per_cm2",
                    value=0.1,
                    unit="defects/cm²",
                    provenance=Provenance.ASSUMED,
                    source="component_library_default",
                )
            )
        elif package_type == "3D_stacking":
            defect_density_per_cm2 = 0.15  # 0.15 defects/cm² typical for TSV
            assumptions.append(
                AssumptionRecord(
                    parameter="yield_defect_density_per_cm2",
                    value=0.15,
                    unit="defects/cm²",
                    provenance=Provenance.ASSUMED,
                    source="component_library_default",
                )
            )
        elif package_type == "chiplets":
            defect_density_per_cm2 = 0.05  # 0.05 defects/cm² for chiplet packaging
            assumptions.append(
                AssumptionRecord(
                    parameter="yield_defect_density_per_cm2",
                    value=0.05,
                    unit="defects/cm²",
                    provenance=Provenance.ASSUMED,
                    source="component_library_default",
                )
            )
        else:
            defect_density_per_cm2 = 0.02  # monolithic: very low defect density
            assumptions.append(
                AssumptionRecord(
                    parameter="yield_defect_density_per_cm2",
                    value=0.02,
                    unit="defects/cm²",
                    provenance=Provenance.ASSUMED,
                    source="component_library_default",
                )
            )
    else:
        # allow_defaults=False and no density specified
        if pkg_yield_density is None:
            warnings.append(
                "yield_defect_density_per_cm2 not specified and allow_defaults=False"
            )
        defect_density_per_cm2 = 0.0  # will compute yield as 100%

    # Compute total chiplet area
    total_chiplet_area_mm2 = sum(chiplet_areas_mm2) if chiplet_areas_mm2 else 0.0
    if not chiplet_areas_mm2 and allow_defaults:
        warnings.append("chiplet_areas_mm2 not provided, using 0 area (monolithic assumed)")

    # Compute total package area based on type
    if package_type == "monolithic":
        # Monolithic: single die, area = sum of chiplet areas (or just one large die)
        total_area_mm2 = total_chiplet_area_mm2 if total_chiplet_area_mm2 > 0 else 100.0
        interposer_area_mm2 = 0.0
        if total_chiplet_area_mm2 == 0 and allow_defaults:
            warnings.append("monolithic area estimated as 100 mm² default")
    elif package_type == "chiplets":
        # Chiplets on PCB: total area = chiplet areas + spacing
        total_area_mm2 = total_chiplet_area_mm2 * 1.5  # 50% spacing overhead
        interposer_area_mm2 = 0.0
    elif package_type == "2.5D":
        # 2.5D: silicon interposer with chiplets on top
        # Interposer area is typically 1.5-2x the total chiplet area
        interposer_area_mm2 = total_chiplet_area_mm2 * 1.5
        total_area_mm2 = total_chiplet_area_mm2 + interposer_area_mm2
        assumptions.append(
            AssumptionRecord(
                parameter="interposer_area_mm2",
                value=interposer_area_mm2,
                unit="mm²",
                provenance=Provenance.ASSUMED,
                source="package_type_default",
            )
        )
    elif package_type == "3D_stacking":
        # 3D stacking: TSV-packaged chiplets
        # Interposer (or substrate) area + stack overhead
        interposer_area_mm2 = total_chiplet_area_mm2 * 0.8
        total_area_mm2 = total_chiplet_area_mm2 + interposer_area_mm2
    else:
        # Unknown package type
        total_area_mm2 = total_chiplet_area_mm2 if total_chiplet_area_mm2 > 0 else 100.0
        interposer_area_mm2 = 0.0
        warnings.append(f"Unknown package type '{package_type}', using default area")

    # Compute yield using Poisson model: yield = exp(-defect_density * area_cm2)
    # area_cm2 = area_mm2 / 100
    area_cm2 = total_area_mm2 / 100.0
    if defect_density_per_cm2 is not None and defect_density_per_cm2 > 0:
        # P(zero defects) under Poisson; 1 - exp(...) would be P(at least one defect).
        yield_pct = math.exp(-defect_density_per_cm2 * area_cm2) * 100.0
    else:
        yield_pct = 100.0

    # Ensure yield is between 0 and 100
    yield_pct = max(0.0, min(100.0, yield_pct))

    # Estimated cost based on area and package type
    cost_per_mm2_usd = design.package.cost_per_mm2_usd

    if cost_per_mm2_usd is None and allow_defaults:
        # Use default costs per package type
        if package_type == "monolithic":
            cost_per_mm2_usd = 2.0
        elif package_type == "chiplets":
            cost_per_mm2_usd = 3.0
        elif package_type == "2.5D":
            cost_per_mm2_usd = 5.0
        elif package_type == "3D_stacking":
            cost_per_mm2_usd = 8.0
        else:
            cost_per_mm2_usd = 3.0
        warnings.append(
            f"cost_per_mm2_usd not specified, using default {cost_per_mm2_usd} for {package_type}"
        )

    estimated_cost_usd = total_area_mm2 * cost_per_mm2_usd

    result: dict[str, float] = {
        "total_area_mm2": float(total_area_mm2),
        "interposer_area_mm2": float(interposer_area_mm2),
        "yield_pct": float(yield_pct),
        "estimated_cost_usd": float(estimated_cost_usd),
    }

    return result, assumptions, warnings