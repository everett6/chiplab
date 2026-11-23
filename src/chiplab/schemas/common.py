from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class Provenance(str, Enum):
    MEASURED = "measured"
    PUBLISHED = "published"
    ESTIMATED = "estimated"
    ASSUMED = "assumed"


class FidelityLevel(str, Enum):
    ANALYTICAL = "analytical"
    DISCRETE_EVENT = "discrete_event"
    SPECIALIZED = "specialized"
    RTL_CIRCUIT = "rtl_circuit"


def unwrap_value(v: float | int | str | bool | BaseModel) -> float | int | str | bool:
    """Helper: return .value if v is a TrackedValue, else v."""
    if isinstance(v, TrackedValue):
        return v.value
    return v


def get_provenance(v: float | int | str | bool | BaseModel) -> Provenance:
    """Helper: return Provenance of v, defaulting to ASSUMED."""
    if isinstance(v, TrackedValue):
        return v.provenance
    return Provenance.ASSUMED


class TrackedValue(BaseModel):
    value: float | int | str | bool
    unit: str | None = None
    provenance: Provenance = Provenance.ASSUMED
    source: str | None = None
    uncertainty_pct: float | None = None
    notes: str | None = None


# Type aliases
ParamFloat = float | int | TrackedValue
ParamInt = int | TrackedValue