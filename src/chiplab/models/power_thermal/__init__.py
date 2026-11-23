from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from chiplab.schemas.simulation import AssumptionRecord
from chiplab.schemas.common import Provenance
from chiplab.schemas.design import DesignSpec


def estimate_power_thermal(
    design: DesignSpec,
    component_power_watts: float,
    total_area_mm2: float,
    allow_defaults: bool = True,
) -> tuple[dict[str, float], list[AssumptionRecord], list[str]]:
    """Estimate power and thermal characteristics of the design.

    Computes:
        - total_power_watts: total power consumption in watts
        - power_density_w_per_mm2: power per unit area in W/mm²
        - estimated_temp_c: steady-state temperature in °C
        - thermal_throttling_factor: throttling factor (1.0 if no throttling)

    Args:
        design: DesignSpec containing thermal constraints
        component_power_watts: total component power in watts (from other model estimates)
        total_area_mm2: total die/package area in mm²
        allow_defaults: whether to use defaults when params are missing

    Returns:
        tuple of (metrics_dict, assumption_records, warnings)
    """
    assumptions: list[AssumptionRecord] = []
    warnings: list[str] = []

    # Get thermal constraints from design
    constraints = design.constraints
    max_temp_c = constraints.max_temp_c
    ambient_temp_c = constraints.ambient_temp_c
    thermal_resistance = constraints.thermal_resistance_c_per_w

    # Total power: use component_power_watts parameter, or sum from design
    total_power_watts = float(component_power_watts) if component_power_watts is not None else 0.0

    if component_power_watts is None and allow_defaults:
        warnings.append(
            "component_power_watts not provided, using 0W; "
            "ensure this is set from compute/memory model estimates"
        )
    elif component_power_watts is None and not allow_defaults:
        raise ValueError("component_power_watts is required when allow_defaults=False")

    # Power density
    if total_area_mm2 > 0:
        power_density_w_per_mm2 = total_power_watts / total_area_mm2
    else:
        power_density_w_per_mm2 = 0.0
        warnings.append("total_area_mm2 is 0, power_density set to 0")

    # Estimate steady-state temperature
    # T_steady = T_ambient + P * R_th
    if thermal_resistance is not None and thermal_resistance > 0:
        estimated_temp_c = ambient_temp_c + total_power_watts * thermal_resistance
    else:
        # Default thermal resistance: ~10°C/W for typical chips
        # Or estimate based on power density
        if allow_defaults:
            thermal_resistance = 10.0  # default 10°C/W
            estimated_temp_c = ambient_temp_c + total_power_watts * thermal_resistance
            assumptions.append(
                AssumptionRecord(
                    parameter="thermal_resistance_c_per_w",
                    value=10.0,
                    unit="°C/W",
                    provenance=Provenance.ASSUMED,
                    source="default_estimate",
                )
            )
            warnings.append(
                f"thermal_resistance not specified, using default {thermal_resistance}°C/W"
            )
        else:
            raise ValueError(
                "thermal_resistance_c_per_w is required when allow_defaults=False"
            )

    # Ensure temperature doesn't go below ambient
    if estimated_temp_c < ambient_temp_c:
        estimated_temp_c = ambient_temp_c

    # Thermal throttling factor
    # 1.0 if <= max_temp_c, else scaled down
    if estimated_temp_c <= max_temp_c:
        thermal_throttling_factor = 1.0
    else:
        # Linear scaling: factor = max_temp_c / estimated_temp_c
        # Could be more sophisticated, but keep simple
        thermal_throttling_factor = max_temp_c / estimated_temp_c
        if thermal_throttling_factor < 0:
            thermal_throttling_factor = 0.0
        warnings.append(
            f"Thermal throttling applied: "
            f"temp {estimated_temp_c:.1f}°C > max {max_temp_c}°C, "
            f"factor = {thermal_throttling_factor:.2f}"
        )

    # Record assumptions
    assumptions.append(
        AssumptionRecord(
            parameter="ambient_temp_c",
            value=ambient_temp_c,
            unit="°C",
            provenance=Provenance.MEASURED,
            source="design_constraints",
        )
    )
    assumptions.append(
        AssumptionRecord(
            parameter="max_temp_c",
            value=max_temp_c,
            unit="°C",
            provenance=Provenance.MEASURED,
            source="design_constraints",
        )
    )

    result: dict[str, float] = {
        "total_power_watts": total_power_watts,
        "power_density_w_per_mm2": power_density_w_per_mm2,
        "estimated_temp_c": estimated_temp_c,
        "thermal_throttling_factor": thermal_throttling_factor,
    }

    return result, assumptions, warnings