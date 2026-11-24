from enum import Enum
from typing import Any
import numpy as np
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
    """A value with optional unit, provenance tracking, and metadata."""
    value: float | int | str | bool | np.ndarray
    unit: str | None = None
    provenance: Provenance = Provenance.ASSUMED
    source: str | None = None
    uncertainty_pct: float | None = None
    notes: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    # Models call float()/int() and truth tests on spec fields; a TrackedValue must behave
    # like its number there, or any design that uses provenance/uncertainty crashes.
    def __float__(self) -> float:
        v = self.value
        if isinstance(v, np.ndarray):
            return float(v.mean())  # use mean of array
        return float(v)

    def __int__(self) -> int:
        v = self.value
        if isinstance(v, np.ndarray):
            return int(v.mean())
        return int(v)

    def __bool__(self) -> bool:
        v = self.value
        if isinstance(v, np.ndarray):
            return bool(v.mean())
        return bool(v)


# Type aliases
ParamFloat = float | int | TrackedValue
ParamInt = int | TrackedValue