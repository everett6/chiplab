from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from .common import FidelityLevel, ParamFloat, ParamInt, Provenance


class MemorySpec(BaseModel):
    technology: str
    capacity_per_chiplet_gb: ParamFloat
    channels: ParamInt = 1
    banks_per_channel: ParamInt = 8
    bandwidth_per_channel_gbps: ParamFloat | None = None
    latency_ns: ParamFloat | None = None
    energy_pj_per_bit: ParamFloat | None = None
    stack_layers: ParamInt = 1


class EcoreArraySpec(BaseModel):
    """Take a PIM / analog-CIM chiplet's compute rate, energy/op and power (and area when
    ``arrays`` is set) from Echo Lab's eCore array simulator instead of hand-entered numbers."""

    rows: int = Field(64, ge=8, le=1024)
    cols: int = Field(64, ge=8, le=1024)
    precision: Literal["int4", "int8", "fp8_e4m3", "fp8_e5m2", "fp16"] = "int8"
    adc_bits: Literal[4, 6, 8, 10, 12] = 8
    organization: Literal["org_a", "org_b"] = "org_b"
    arrays: int | None = Field(None, ge=1, description=(
        "Arrays per chiplet. If omitted, derived from area_mm2 x array_area_fraction."))
    array_area_fraction: float = Field(0.6, gt=0.0, le=1.0, description=(
        "Share of die area occupied by eCore arrays (rest: periphery, I/O). Assumed."))
    seed: int = Field(42, ge=0, lt=2**31)


class ComputeSpec(BaseModel):
    architecture: str
    precision: str = "INT8"
    peak_tops: ParamFloat | None = None
    clock_ghz: ParamFloat | None = None
    mac_units: ParamInt | None = None
    sram_mb: ParamFloat | None = None
    efficiency_factor: ParamFloat | None = None
    ecore: EcoreArraySpec | None = None
    # Systolic-array geometry for SCALE-Sim (fidelity="specialized"). If omitted: square
    # arrays of the largest power of two <= sqrt(mac_units), capped at 128, and the best of
    # the three dataflows.
    array_dim: int | None = Field(None, ge=4, le=1024)
    dataflow: Literal["ws", "os", "is"] | None = None


class ChipletSpec(BaseModel):
    id: str | None = None
    type: str
    count: ParamInt = 1
    area_mm2: ParamFloat | None = None
    tdp_watts: ParamFloat | None = None
    process_nm: ParamInt | None = None
    memory: MemorySpec | None = None
    compute: ComputeSpec | None = None
    material_device: dict[str, Any] | None = Field(None, description=(
        "Device record (geometry, parameters, operating conditions, user measurements) of a "
        "simulated material device. Engines re-run the physics validation gate on it; see "
        "chiplab.credibility."))


class PackageSpec(BaseModel):
    type: str = "2.5D"
    interposer: str | None = "silicon"
    max_area_mm2: ParamFloat | None = None
    cost_per_mm2_usd: ParamFloat | None = None
    yield_defect_density_per_cm2: ParamFloat | None = None


class InterconnectSpec(BaseModel):
    topology: str = "mesh"
    link_bandwidth_gbps: ParamFloat | None = None
    latency_ns: ParamFloat | None = None
    energy_pj_per_bit: ParamFloat | None = None
    lanes_per_chiplet: ParamInt = 1


class PhysicalConstraints(BaseModel):
    max_power_watts: ParamFloat | None = None
    max_area_mm2: ParamFloat | None = None
    max_temp_c: ParamFloat = 85.0
    ambient_temp_c: ParamFloat = 25.0
    thermal_resistance_c_per_w: ParamFloat | None = None
    max_cost_usd: ParamFloat | None = None


class DesignSpec(BaseModel):
    name: str
    version: str = "0.1"
    parent_version: str | None = None
    description: str = ""
    architecture: str = "heterogeneous_accelerator"
    package: PackageSpec = Field(default_factory=PackageSpec)
    chiplets: list[ChipletSpec]
    interconnect: InterconnectSpec = Field(default_factory=InterconnectSpec)
    constraints: PhysicalConstraints = Field(default_factory=PhysicalConstraints)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, text: str) -> DesignSpec:
        import yaml
        data = yaml.safe_load(text)
        # Handle optional top-level "design:" key
        if data and "design" in data and isinstance(data["design"], dict):
            data = data["design"]
        # Handle optional top-level "project:" key (used by some YAML designs)
        if data and "project" in data and isinstance(data["project"], dict):
            data = data["project"]
        return cls(**data)

    def to_yaml(self) -> str:
        import yaml
        d = self.model_dump(by_alias=False, exclude_none=True)
        # Remove default package values for cleaner output
        if d.get("package") == PackageSpec().model_dump(by_alias=False, exclude_none=True):
            d.pop("package")
        if d.get("interconnect") == InterconnectSpec().model_dump(by_alias=False, exclude_none=True):
            d.pop("interconnect")
        if d.get("constraints") == PhysicalConstraints().model_dump(by_alias=False, exclude_none=True):
            d.pop("constraints")
        return yaml.dump({"design": d}, default_flow_style=False, sort_keys=False)

def plain_design(design: DesignSpec) -> tuple[DesignSpec, list[dict[str, Any]]]:
    """Return a copy of ``design`` with every TrackedValue replaced by its plain value, plus
    one provenance record per replaced field ({parameter, value, unit, provenance, source,
    uncertainty_pct, notes}). Engines and the cost model work on plain numbers; the records
    keep where each number came from. A design without TrackedValues comes back as is."""
    from .common import TrackedValue

    records: list[dict[str, Any]] = []

    def strip(obj: Any, path: str) -> Any:
        if isinstance(obj, TrackedValue):
            records.append({"parameter": path, "value": obj.value, "unit": obj.unit,
                            "provenance": obj.provenance, "source": obj.source,
                            "uncertainty_pct": obj.uncertainty_pct, "notes": obj.notes})
            return obj.value
        if isinstance(obj, BaseModel):
            changed = {}
            for name in type(obj).model_fields:
                val = getattr(obj, name)
                new = strip(val, f"{path}.{name}" if path else name)
                if new is not val:
                    changed[name] = new
            return obj.model_copy(update=changed) if changed else obj
        if isinstance(obj, list):
            items = [strip(v, f"{path}[{i}]") for i, v in enumerate(obj)]
            return items if any(a is not b for a, b in zip(items, obj)) else obj
        return obj

    out = strip(design, "")
    return out, records
