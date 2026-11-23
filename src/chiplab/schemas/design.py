from __future__ import annotations

from typing import Any, Optional

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


class ComputeSpec(BaseModel):
    architecture: str
    precision: str = "INT8"
    peak_tops: ParamFloat | None = None
    clock_ghz: ParamFloat | None = None
    mac_units: ParamInt | None = None
    sram_mb: ParamFloat | None = None
    efficiency_factor: ParamFloat | None = None


class ChipletSpec(BaseModel):
    id: str | None = None
    type: str
    count: ParamInt = 1
    area_mm2: ParamFloat | None = None
    tdp_watts: ParamFloat | None = None
    process_nm: ParamInt | None = None
    memory: MemorySpec | None = None
    compute: ComputeSpec | None = None


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