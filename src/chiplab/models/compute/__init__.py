from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from chiplab.schemas.simulation import AssumptionRecord
from chiplab.schemas.common import Provenance
from chiplab.schemas.design import ChipletSpec, DesignSpec
from chiplab.library.components import COMPONENT_LIBRARY, get_component_defaults


# Precision scaling factors relative to INT8
PRECISION_SCALING = {
    "INT4": 2.0,  # INT4 ~2x INT8
    "INT8": 1.0,
    "BF16": 0.5,
    "FP16": 0.5,
    "FP32": 0.25,
}


def estimate_compute_chiplet(
    chiplet: ChipletSpec,
    allow_defaults: bool = True,
) -> tuple[dict[str, float], list[AssumptionRecord], list[str]]:
    """Estimate compute chiplet parameters using component library.

    Handles SIMD, systolic_array, tensor_engine, vector_processor, pim, analog_pim
    uniformly via component library & precision scaling.

    Precision scaling:
        INT4 ~2x INT8, INT8 1x, FP16/BF16 0.5x, FP32 0.25x if derived from MACs/clock.

    Returns:
        tuple of (metrics_dict, assumption_records, warnings)
        - peak_tops: peak theoretical performance in TOPS
        - efficiency_factor: fractional efficiency (0-1)
        - effective_tops: effective TOPS after precision scaling
        - sram_mb: SRAM size in MB
        - area_mm2: chip area in mm²
        - power_watts: power in watts
        - clock_ghz: clock frequency in GHz
    """
    assumptions: list[AssumptionRecord] = []
    warnings: list[str] = []

    comp_spec = chiplet.compute

    if comp_spec is not None:
        # Use provided compute spec values
        peak_tops_raw = float(comp_spec.peak_tops) if comp_spec.peak_tops else None
        clock_ghz = float(comp_spec.clock_ghz) if comp_spec.clock_ghz else None
        sram_mb = float(comp_spec.sram_mb) if comp_spec.sram_mb else None
        efficiency_factor = (
            float(comp_spec.efficiency_factor) if comp_spec.efficiency_factor else 1.0
        )

        # Determine precision and scaling factor
        precision = comp_spec.precision or "INT8"
        scaling_factor = PRECISION_SCALING.get(precision, 1.0)
        if precision not in PRECISION_SCALING:
            warnings.append(
                f"Unknown precision '{precision}', defaulting to scaling factor 1.0 (INT8)"
            )

        # Apply precision scaling to get effective peak TOPS
        if peak_tops_raw is not None:
            peak_tops = peak_tops_raw
            effective_tops = peak_tops_raw * scaling_factor
        else:
            effective_tops = 0.0
            warnings.append("peak_tops not provided in compute spec")

        # Record provided values
        if peak_tops_raw is not None:
            assumptions.append(
                AssumptionRecord(
                    parameter="peak_tops",
                    value=peak_tops_raw,
                    unit="TOPS",
                    provenance=Provenance.MEASURED,
                    source="chiplet_spec",
                )
            )
        if clock_ghz is not None:
            assumptions.append(
                AssumptionRecord(
                    parameter="clock_ghz",
                    value=clock_ghz,
                    unit="GHz",
                    provenance=Provenance.MEASURED,
                    source="chiplet_spec",
                )
            )
        if sram_mb is not None:
            assumptions.append(
                AssumptionRecord(
                    parameter="sram_mb",
                    value=sram_mb,
                    unit="MB",
                    provenance=Provenance.MEASURED,
                    source="chiplet_spec",
                )
            )
        if efficiency_factor is not None:
            assumptions.append(
                AssumptionRecord(
                    parameter="efficiency_factor",
                    value=efficiency_factor,
                    unit="fraction",
                    provenance=Provenance.MEASURED,
                    source="chiplet_spec",
                )
            )
        # Area and power from chiplet TDP if available
        area_mm2 = float(chiplet.area_mm2) if chiplet.area_mm2 else None
        power_watts = float(chiplet.tdp_watts) if chiplet.tdp_watts else None

        if area_mm2 is not None and chiplet.area_mm2 is None:
            warnings.append(
                f"area_mm2 not in chiplet spec, using provided compute area"
            )
        if power_watts is not None and chiplet.tdp_watts is None:
            warnings.append(
                f"power_watts not in chiplet spec, using provided compute TDP"
            )
    else:
        # Fall back to component library defaults
        comp_name = chiplet.type or "SIMD"
        comp_defaults = get_component_defaults("compute", comp_name)

        if comp_defaults is not None and allow_defaults:
            peak_tops = comp_defaults["peak_tops"]
            clock_ghz = comp_defaults["clock_ghz"]
            sram_mb = comp_defaults["sram_mb"]
            efficiency_factor = comp_defaults["efficiency_factor"]

            precision = comp_defaults.get("precision", "INT8")
            scaling_factor = PRECISION_SCALING.get(precision, 1.0)

            effective_tops = peak_tops * scaling_factor

            assumptions.append(
                AssumptionRecord(
                    parameter="peak_tops",
                    value=peak_tops,
                    unit="TOPS",
                    provenance=Provenance.ASSUMED,
                    source="component_library",
                )
            )
            assumptions.append(
                AssumptionRecord(
                    parameter="clock_ghz",
                    value=clock_ghz,
                    unit="GHz",
                    provenance=Provenance.ASSUMED,
                    source="component_library",
                )
            )
            assumptions.append(
                AssumptionRecord(
                    parameter="sram_mb",
                    value=sram_mb,
                    unit="MB",
                    provenance=Provenance.ASSUMED,
                    source="component_library",
                )
            )
            assumptions.append(
                AssumptionRecord(
                    parameter="efficiency_factor",
                    value=efficiency_factor,
                    unit="fraction",
                    provenance=Provenance.ASSUMED,
                    source="component_library",
                )
            )
            assumptions.append(
                AssumptionRecord(
                    parameter="precision",
                    value=precision,
                    unit="str",
                    provenance=Provenance.ASSUMED,
                    source="component_library",
                )
            )
            assumptions.append(
                AssumptionRecord(
                    parameter="scaling_factor",
                    value=scaling_factor,
                    unit="factor",
                    provenance=Provenance.ASSUMED,
                    source="component_library",
                )
            )
        elif comp_defaults is not None and not allow_defaults:
            # User explicitly disallowed defaults, but no chiplet spec provided
            raise ValueError(
                f"No compute specification found for chiplet type '{chiplet.type}' and allow_defaults=False"
            )
            peak_tops = comp_defaults["peak_tops"]
            clock_ghz = comp_defaults["clock_ghz"]
            sram_mb = comp_defaults["sram_mb"]
            efficiency_factor = comp_defaults["efficiency_factor"]

            precision = comp_defaults.get("precision", "INT8")
            scaling_factor = PRECISION_SCALING.get(precision, 1.0)
            effective_tops = peak_tops * scaling_factor
        else:
            raise ValueError(
                f"No compute specification found for chiplet type '{chiplet.type}' and no component library defaults available"
            )

        # Set area and power - try chiplet spec first
        area_mm2 = float(chiplet.area_mm2) if chiplet.area_mm2 else None
        power_watts = float(chiplet.tdp_watts) if chiplet.tdp_watts else None

        if area_mm2 is None:
            # Rough estimate based on sram_mb: ~0.1 mm²/MB
            area_mm2 = sram_mb * 0.1 if sram_mb else 10.0
            warnings.append(
                f"area_mm2 not available, estimated as {area_mm2:.1f} mm² based on SRAM"
            )
        if power_watts is None:
            # Rough estimate: ~1W per TOPS (scaled)
            power_watts = peak_tops * 1.0 if peak_tops else 100.0
            warnings.append(
                f"power_watts not available, estimated as {power_watts:.1f} W based on peak TOPS"
            )

    # Compute effective TOPS with precision scaling
    effective_tops = effective_tops if 'effective_tops' in dir() else 0.0

    result: dict[str, float] = {
        "peak_tops": float(peak_tops) if 'peak_tops' in dir() and 'peak_tops' in locals() else 0.0,
        "efficiency_factor": float(efficiency_factor) if 'efficiency_factor' in dir() else 1.0,
        "effective_tops": float(effective_tops) if 'effective_tops' in dir() else 0.0,
        "sram_mb": float(sram_mb) if 'sram_mb' in dir() and sram_mb is not None else 0.0,
        "area_mm2": float(area_mm2) if 'area_mm2' in dir() and area_mm2 is not None else 0.0,
        "power_watts": float(power_watts) if 'power_watts' in dir() and power_watts is not None else 0.0,
        "clock_ghz": float(clock_ghz) if 'clock_ghz' in dir() and clock_ghz is not None else 0.0,
    }

    # Fill in missing keys with defaults
    for key in ["peak_tops", "efficiency_factor", "effective_tops", "sram_mb", "area_mm2", "power_watts", "clock_ghz"]:
        if key not in result:
            if key == "efficiency_factor":
                result[key] = 1.0
            elif key == "clock_ghz":
                result[key] = 1.0
            else:
                result[key] = 0.0

    return result, assumptions, warnings