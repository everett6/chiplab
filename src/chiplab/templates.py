"""Starter designs an AI client can copy and tweak instead of writing a spec from scratch."""

from __future__ import annotations

import copy
from typing import Any

TEMPLATES: dict[str, dict[str, Any]] = {
    "edge_npu": {
        "summary": "Phone/IoT NPU: one small TSMC N4 die + LPDDR5X, flip-chip BGA, ~5 W.",
        "spec": {
            "name": "edge_npu", "version": "0.1",
            "description": "Single-die edge inference NPU with on-board LPDDR5X.",
            "package": {"type": "2D", "interposer": "organic"},
            "chiplets": [
                {"id": "npu", "type": "compute", "count": 1, "area_mm2": 60.0,
                 "tdp_watts": 4.0, "process_nm": 4,
                 "compute": {"architecture": "systolic", "precision": "INT8",
                             "peak_tops": 40.0, "clock_ghz": 1.2, "mac_units": 16384,
                             "sram_mb": 8.0, "efficiency_factor": 0.6}},
                {"id": "dram", "type": "LPDDR5X", "count": 1, "tdp_watts": 1.0,
                 "memory": {"technology": "LPDDR5X", "capacity_per_chiplet_gb": 8.0,
                            "channels": 4, "bandwidth_per_channel_gbps": 17.0,
                            "latency_ns": 100.0, "energy_pj_per_bit": 4.0}},
            ],
            "interconnect": {"topology": "bus", "link_bandwidth_gbps": 128.0,
                             "latency_ns": 10.0, "energy_pj_per_bit": 1.0},
            "constraints": {"max_power_watts": 6.0, "max_temp_c": 95.0,
                            "thermal_resistance_c_per_w": 6.0, "max_cost_usd": 100.0},
            "metadata": {"suggested_volume": 5_000_000},
        },
    },
    "llm_accelerator_hbm3e": {
        "summary": "Datacenter LLM accelerator: 2 TSMC N3 compute dies + 8 HBM3E stacks "
                   "on CoWoS-L, ~1 kW.",
        "spec": {
            "name": "llm_accelerator_hbm3e", "version": "0.1",
            "description": "Two reticle-class compute dies with 8x24 GB HBM3E.",
            "package": {"type": "2.5D", "interposer": "cowos_l"},
            "chiplets": [
                {"id": "compute", "type": "compute", "count": 2, "area_mm2": 780.0,
                 "tdp_watts": 450.0, "process_nm": 3,
                 "compute": {"architecture": "tensor_engine", "precision": "FP8",
                             "peak_tops": 2250.0, "clock_ghz": 1.8, "mac_units": 327680,
                             "sram_mb": 64.0, "efficiency_factor": 0.7}},
                {"id": "hbm", "type": "HBM3E", "count": 8, "tdp_watts": 15.0,
                 "memory": {"technology": "HBM3E", "capacity_per_chiplet_gb": 24.0,
                            "channels": 16, "bandwidth_per_channel_gbps": 75.0,
                            "latency_ns": 100.0, "energy_pj_per_bit": 3.5,
                            "stack_layers": 8}},
            ],
            "interconnect": {"topology": "mesh", "link_bandwidth_gbps": 10000.0,
                             "latency_ns": 20.0, "energy_pj_per_bit": 0.3,
                             "lanes_per_chiplet": 4},
            "constraints": {"max_power_watts": 1200.0, "max_temp_c": 95.0,
                            "thermal_resistance_c_per_w": 0.05, "max_cost_usd": 8000.0},
            "metadata": {"suggested_volume": 200_000},
        },
    },
    "chiplet_cpu": {
        "summary": "Server CPU: 8 TSMC N3 core chiplets + 1 N6 I/O die on an organic "
                   "substrate (DDR5 sits on the board).",
        "spec": {
            "name": "chiplet_cpu", "version": "0.1",
            "description": "Chiplet server CPU with a central I/O die.",
            "package": {"type": "organic", "interposer": "organic"},
            "chiplets": [
                {"id": "ccd", "type": "compute", "count": 8, "area_mm2": 70.0,
                 "tdp_watts": 40.0, "process_nm": 3,
                 "compute": {"architecture": "cpu", "precision": "FP64", "peak_tops": 2.0,
                             "clock_ghz": 3.5, "mac_units": 256, "sram_mb": 32.0,
                             "efficiency_factor": 0.5}},
                {"id": "iod", "type": "io", "count": 1, "area_mm2": 400.0,
                 "tdp_watts": 60.0, "process_nm": 6},
            ],
            "interconnect": {"topology": "star", "link_bandwidth_gbps": 64.0,
                             "latency_ns": 15.0, "energy_pj_per_bit": 1.5},
            "constraints": {"max_power_watts": 400.0, "max_temp_c": 95.0,
                            "thermal_resistance_c_per_w": 0.12, "max_cost_usd": 1500.0},
            "metadata": {"suggested_volume": 1_000_000},
        },
    },
}


def list_templates() -> dict[str, str]:
    return {k: v["summary"] for k, v in TEMPLATES.items()}


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    for k, v in over.items():
        if k == "chiplets" and isinstance(v, list):
            by_id = {c.get("id"): c for c in base.get("chiplets", [])}
            for item in v:
                cid = item.get("id")
                if cid in by_id and not item.get("remove"):
                    _merge(by_id[cid], item)
                elif cid in by_id:
                    base["chiplets"].remove(by_id[cid])
                else:
                    base.setdefault("chiplets", []).append(item)
        elif isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def build_from_template(template: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a spec dict: the template deep-merged with overrides.

    Chiplets merge by ``id``: an override with an existing id edits it, a new id adds a
    chiplet, and ``{"id": "...", "remove": true}`` drops one.
    """
    if template not in TEMPLATES:
        raise KeyError(f"Unknown template '{template}'. Choose one of {sorted(TEMPLATES)}.")
    spec = copy.deepcopy(TEMPLATES[template]["spec"])
    return _merge(spec, copy.deepcopy(overrides or {}))
