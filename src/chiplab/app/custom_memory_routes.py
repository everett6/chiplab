"""Custom Memory Type Explorer backend routes and PPA/roofline evaluation."""

from __future__ import annotations

from typing import Any, Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from chiplab.experiments.store import ExperimentStore
from chiplab.models.materials.process_nodes import scale_experimental_device

VALID_TECH_CLASSES = (
    "dram",
    "sram",
    "reram",
    "fefet",
    "mram",
    "pcm",
    "optical",
    "custom",
)

# Mapping from custom memory tech_class to process_nodes.scale_experimental_device technology key
_TECH_SCALING_MAP: dict[str, str] = {
    "reram": "reram",
    "fefet": "fefet",
    "mram": "mram",
    "pcm": "pcm",
    "optical": "photonic",
    "memristor": "memristor",
    "ferroelectric": "ferroelectric",
    "spintronic": "spintronic",
    "ftj": "ftj",
    "photonic": "photonic",
    "ncfet": "ncfet",
    "tfet": "tfet",
    "tmd_fet": "tmd_fet",
}

_TECH_COST_FACTOR: dict[str, float] = {
    "dram": 1.0,
    "sram": 1.35,
    "reram": 0.90,
    "fefet": 1.05,
    "mram": 1.15,
    "pcm": 0.95,
    "optical": 1.65,
    "custom": 1.10,
}

MEMORY_PRESETS: list[dict[str, Any]] = [
    {
        "name": "HBM3e-36GB",
        "tech_class": "dram",
        "node_nm": 10.0,
        "stacks": 8,
        "dies_per_stack": 12,
        "capacity_gb_per_stack": 36.0,
        "channels_per_stack": 16,
        "io_gbps_per_pin": 9.6,
        "bus_width_bits": 1024,
        "energy_pj_per_bit": 3.5,
        "read_latency_ns": 38.0,
        "write_latency_ns": 40.0,
        "density_gb_per_mm2": 0.03,
        "retention_s": 0.064,
        "endurance_cycles": 1e16,
        "cim_capable": False,
        "cim_tops_per_w": 0.0,
    },
    {
        "name": "HBM4-2048b",
        "tech_class": "dram",
        "node_nm": 8.0,
        "stacks": 8,
        "dies_per_stack": 16,
        "capacity_gb_per_stack": 48.0,
        "channels_per_stack": 32,
        "io_gbps_per_pin": 8.0,
        "bus_width_bits": 2048,
        "energy_pj_per_bit": 2.8,
        "read_latency_ns": 34.0,
        "write_latency_ns": 36.0,
        "density_gb_per_mm2": 0.035,
        "retention_s": 0.064,
        "endurance_cycles": 1e16,
        "cim_capable": False,
        "cim_tops_per_w": 0.0,
    },
    {
        "name": "3D-ReRAM-Crossbar",
        "tech_class": "reram",
        "node_nm": 14.0,
        "stacks": 8,
        "dies_per_stack": 8,
        "capacity_gb_per_stack": 64.0,
        "channels_per_stack": 16,
        "io_gbps_per_pin": 6.4,
        "bus_width_bits": 1024,
        "energy_pj_per_bit": 1.8,
        "read_latency_ns": 18.0,
        "write_latency_ns": 65.0,
        "density_gb_per_mm2": 0.12,
        "retention_s": 1e6,
        "endurance_cycles": 1e10,
        "cim_capable": True,
        "cim_tops_per_w": 45.0,
    },
    {
        "name": "FeFET-CIM",
        "tech_class": "fefet",
        "node_nm": 7.0,
        "stacks": 6,
        "dies_per_stack": 8,
        "capacity_gb_per_stack": 32.0,
        "channels_per_stack": 16,
        "io_gbps_per_pin": 8.0,
        "bus_width_bits": 1024,
        "energy_pj_per_bit": 0.9,
        "read_latency_ns": 10.0,
        "write_latency_ns": 14.0,
        "density_gb_per_mm2": 0.15,
        "retention_s": 1e8,
        "endurance_cycles": 1e11,
        "cim_capable": True,
        "cim_tops_per_w": 80.0,
    },
    {
        "name": "STT-MRAM-LLC",
        "tech_class": "mram",
        "node_nm": 5.0,
        "stacks": 4,
        "dies_per_stack": 4,
        "capacity_gb_per_stack": 8.0,
        "channels_per_stack": 16,
        "io_gbps_per_pin": 12.8,
        "bus_width_bits": 1024,
        "energy_pj_per_bit": 1.4,
        "read_latency_ns": 6.5,
        "write_latency_ns": 12.0,
        "density_gb_per_mm2": 0.04,
        "retention_s": 1e9,
        "endurance_cycles": 1e13,
        "cim_capable": True,
        "cim_tops_per_w": 25.0,
    },
    {
        "name": "3D-V-SRAM",
        "tech_class": "sram",
        "node_nm": 3.0,
        "stacks": 8,
        "dies_per_stack": 4,
        "capacity_gb_per_stack": 2.0,
        "channels_per_stack": 32,
        "io_gbps_per_pin": 16.0,
        "bus_width_bits": 2048,
        "energy_pj_per_bit": 0.65,
        "read_latency_ns": 1.8,
        "write_latency_ns": 1.8,
        "density_gb_per_mm2": 0.012,
        "retention_s": None,
        "endurance_cycles": 1e16,
        "cim_capable": True,
        "cim_tops_per_w": 35.0,
    },
    {
        "name": "Optical-DRAM",
        "tech_class": "optical",
        "node_nm": 12.0,
        "stacks": 8,
        "dies_per_stack": 8,
        "capacity_gb_per_stack": 64.0,
        "channels_per_stack": 32,
        "io_gbps_per_pin": 32.0,
        "bus_width_bits": 1024,
        "energy_pj_per_bit": 0.45,
        "read_latency_ns": 15.0,
        "write_latency_ns": 18.0,
        "density_gb_per_mm2": 0.025,
        "retention_s": 0.064,
        "endurance_cycles": 1e16,
        "cim_capable": False,
        "cim_tops_per_w": 0.0,
    },
]

_PRESETS_BY_NAME: dict[str, dict[str, Any]] = {p["name"]: p for p in MEMORY_PRESETS}


def _extract_design_context(store: ExperimentStore, design_id: str | None) -> dict[str, Any]:
    """Load compute, thermal, and baseline cost context from store or baseline GPU fallback."""
    spec = None
    resolved_id: str | None = None

    if design_id:
        resolved_id, spec = store.get_design(str(design_id))
    else:
        designs = store.list_designs()
        if designs:
            resolved_id, spec = store.get_design(designs[0]["design_id"])

    if spec is None:
        return {
            "design_id": None,
            "design_name": "Baseline-AI-GPU",
            "source": "baseline_gpu",
            "base_compute_tops": 2000.0,
            "base_compute_power_w": 450.0,
            "ambient_temp_c": 25.0,
            "max_temp_c": 85.0,
            "thermal_resistance_c_per_w": 0.075,
            "max_power_watts": 750.0,
            "baseline_memory_cost_usd": 1200.0,
        }

    base_compute_tops = 0.0
    base_compute_power_w = 0.0
    baseline_memory_cost_usd = 0.0

    for chiplet in spec.chiplets:
        count = int(chiplet.count)
        if chiplet.compute is not None:
            tops = float(chiplet.compute.peak_tops) if chiplet.compute.peak_tops is not None else 500.0
            base_compute_tops += count * tops
            tdp = float(chiplet.tdp_watts) if chiplet.tdp_watts is not None else 150.0
            base_compute_power_w += count * tdp
        elif chiplet.memory is not None:
            cap = float(chiplet.memory.capacity_per_chiplet_gb)
            layers = int(chiplet.memory.stack_layers)
            baseline_memory_cost_usd += count * (60.0 + cap * 2.5 + layers * 4.0)
        else:
            tdp = float(chiplet.tdp_watts) if chiplet.tdp_watts is not None else 25.0
            base_compute_power_w += count * tdp

    if base_compute_tops <= 0.0:
        base_compute_tops = 2000.0
    if base_compute_power_w <= 0.0:
        base_compute_power_w = 450.0
    if baseline_memory_cost_usd <= 0.0:
        baseline_memory_cost_usd = 1200.0

    constraints = spec.constraints
    ambient_temp_c = float(constraints.ambient_temp_c)
    max_temp_c = float(constraints.max_temp_c)
    thermal_r = (
        float(constraints.thermal_resistance_c_per_w)
        if constraints.thermal_resistance_c_per_w is not None
        else 0.075
    )
    max_power_watts = (
        float(constraints.max_power_watts)
        if constraints.max_power_watts is not None
        else 750.0
    )

    return {
        "design_id": resolved_id,
        "design_name": spec.name,
        "source": "store",
        "base_compute_tops": base_compute_tops,
        "base_compute_power_w": base_compute_power_w,
        "ambient_temp_c": ambient_temp_c,
        "max_temp_c": max_temp_c,
        "thermal_resistance_c_per_w": thermal_r,
        "max_power_watts": max_power_watts,
        "baseline_memory_cost_usd": baseline_memory_cost_usd,
    }


def _resolve_memory_spec(data: dict[str, Any], ambient_temp_c: float) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Merge preset, user overrides, and scale_experimental_device physics."""
    raw_mem = data.get("memory")
    if isinstance(raw_mem, str):
        preset_name = raw_mem
        overrides: dict[str, Any] = {}
    elif isinstance(raw_mem, dict):
        preset_name = data.get("preset") or raw_mem.get("preset") or raw_mem.get("name")
        overrides = dict(raw_mem)
    else:
        preset_name = data.get("preset")
        ignored = {"design_id", "preset", "memories", "operating_ai", "scale_with_node"}
        overrides = {k: v for k, v in data.items() if k not in ignored}

    base = dict(_PRESETS_BY_NAME[preset_name]) if preset_name and preset_name in _PRESETS_BY_NAME else dict(MEMORY_PRESETS[0])
    if preset_name and preset_name not in _PRESETS_BY_NAME and not overrides:
        raise ValueError(f"Unknown memory preset: {preset_name}")

    merged = {**base, **overrides}
    tech_class = str(merged.get("tech_class", "custom")).lower().strip()
    if tech_class not in VALID_TECH_CLASSES:
        tech_class = "custom"
    merged["tech_class"] = tech_class

    node_nm = max(0.5, float(merged.get("node_nm", 10.0)))
    merged["node_nm"] = node_nm

    device_scaling: dict[str, Any] | None = None
    mapped_tech = _TECH_SCALING_MAP.get(tech_class)
    if mapped_tech is not None:
        device_scaling = scale_experimental_device(
            mapped_tech,
            node_nm,
            temperature_k=ambient_temp_c + 273.15,
        )
        # Apply node scaling when user did not explicitly override key device metrics
        # or when node_nm was changed on a preset without explicit metric overrides.
        preset_node = base.get("node_nm") if preset_name in _PRESETS_BY_NAME else None
        node_changed = preset_node is not None and abs(float(preset_node) - node_nm) > 1e-6
        if "energy_pj_per_bit" not in overrides and (preset_name not in _PRESETS_BY_NAME or node_changed):
            merged["energy_pj_per_bit"] = round(device_scaling["energy_per_bit_j"] * 1e12, 4)
        if "read_latency_ns" not in overrides and (preset_name not in _PRESETS_BY_NAME or node_changed):
            merged["read_latency_ns"] = round(device_scaling["latency_ns"], 3)
        if "density_gb_per_mm2" not in overrides and (preset_name not in _PRESETS_BY_NAME or node_changed):
            # Convert GB/cm2 -> GB/mm2 (1 cm2 = 100 mm2)
            merged["density_gb_per_mm2"] = round(device_scaling["area_density_gb_cm2"] / 100.0, 5)
        if "retention_s" not in overrides and preset_name not in _PRESETS_BY_NAME:
            merged["retention_s"] = device_scaling.get("retention_s")
        if "endurance_cycles" not in overrides and preset_name not in _PRESETS_BY_NAME:
            merged["endurance_cycles"] = device_scaling.get("endurance_cycles")

    cap_per_stack = merged.get("capacity_gb_per_stack")
    if cap_per_stack is None:
        cap_per_stack = merged.get("cap_per_stack_gb", 32.0)

    custom_memory = {
        "name": str(merged.get("name", "Custom-Memory")),
        "tech_class": tech_class,
        "node_nm": node_nm,
        "stacks": max(1, int(merged.get("stacks", 8))),
        "dies_per_stack": max(1, int(merged.get("dies_per_stack", 8))),
        "capacity_gb_per_stack": max(0.01, float(cap_per_stack)),
        "cap_per_stack_gb": max(0.01, float(cap_per_stack)),
        "channels_per_stack": max(1, int(merged.get("channels_per_stack", 16))),
        "io_gbps_per_pin": max(0.1, float(merged.get("io_gbps_per_pin", 8.0))),
        "bus_width_bits": max(8, int(merged.get("bus_width_bits", 1024))),
        "energy_pj_per_bit": max(0.01, float(merged.get("energy_pj_per_bit", 2.0))),
        "read_latency_ns": max(0.1, float(merged.get("read_latency_ns", 20.0))),
        "write_latency_ns": max(0.1, float(merged.get("write_latency_ns", 25.0))),
        "density_gb_per_mm2": max(0.001, float(merged.get("density_gb_per_mm2", 0.05))),
        "retention_s": float(merged["retention_s"]) if merged.get("retention_s") is not None else None,
        "endurance_cycles": float(merged["endurance_cycles"]) if merged.get("endurance_cycles") is not None else None,
        "cim_capable": bool(merged.get("cim_capable", False)),
        "cim_tops_per_w": max(0.0, float(merged.get("cim_tops_per_w", 0.0))),
        "provenance": "estimated",
    }
    return custom_memory, device_scaling


def evaluate_custom_memory(store: ExperimentStore, data: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a custom or preset memory architecture against a stored or baseline GPU design."""
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")

    ctx = _extract_design_context(store, data.get("design_id"))
    mem, device_scaling = _resolve_memory_spec(data, ctx["ambient_temp_c"])

    total_capacity_gb = mem["stacks"] * mem["capacity_gb_per_stack"]
    total_bandwidth_tb_s = mem["stacks"] * (mem["bus_width_bits"] * mem["io_gbps_per_pin"] / 8.0) / 1000.0
    stack_footprint_mm2 = max(
        12.0,
        mem["capacity_gb_per_stack"] / max(mem["density_gb_per_mm2"] * mem["dies_per_stack"], 0.05),
    )
    memory_power_w = (total_bandwidth_tb_s * 1e12 * 8.0) * (mem["energy_pj_per_bit"] * 1e-12) * 0.65
    cim_bonus_tops = (memory_power_w * mem["cim_tops_per_w"]) if mem["cim_capable"] else 0.0

    effective_compute_tops = ctx["base_compute_tops"] + cim_bonus_tops
    ridge_point_flops_per_byte = effective_compute_tops / max(total_bandwidth_tb_s, 1e-9)

    operating_ai = max(0.5, float(data.get("operating_ai", 64.0)))
    achieved_tops = min(effective_compute_tops, total_bandwidth_tb_s * operating_ai)
    roofline_ops_per_byte = operating_ai

    latency_factor = 1.0 / (1.0 + max(0.0, mem["read_latency_ns"] - 15.0) / 500.0)
    ops_per_token = 140e9
    throughput_tok_s = (achieved_tops * 1e12 / ops_per_token) * latency_factor

    total_power_w = ctx["base_compute_power_w"] + memory_power_w
    peak_temp_c = ctx["ambient_temp_c"] + total_power_w * ctx["thermal_resistance_c_per_w"]

    if peak_temp_c > ctx["max_temp_c"]:
        bottleneck = "thermal-limited"
    elif operating_ai < ridge_point_flops_per_byte * 0.85:
        bottleneck = "memory-bandwidth"
    elif operating_ai > ridge_point_flops_per_byte * 1.15:
        bottleneck = "compute-bound"
    else:
        bottleneck = "balanced"

    tech_factor = _TECH_COST_FACTOR.get(mem["tech_class"], 1.10)
    node_cost_mult = 1.0 + (28.0 / max(mem["node_nm"], 2.0)) * 0.08
    die_cost_per_stack = (stack_footprint_mm2 * 0.22 * mem["dies_per_stack"] ** 0.72) * tech_factor * node_cost_mult
    packaging_per_stack = 35.0 + 3.0 * mem["dies_per_stack"]
    estimated_memory_cost_usd = mem["stacks"] * (die_cost_per_stack + packaging_per_stack)
    estimated_cost_delta_usd = estimated_memory_cost_usd - ctx["baseline_memory_cost_usd"]

    metrics = {
        "total_capacity_gb": round(total_capacity_gb, 3),
        "total_bandwidth_tb_s": round(total_bandwidth_tb_s, 4),
        "stack_footprint_mm2": round(stack_footprint_mm2, 3),
        "memory_power_w": round(memory_power_w, 3),
        "cim_bonus_tops": round(cim_bonus_tops, 3),
        "effective_compute_tops": round(effective_compute_tops, 3),
        "ridge_point_flops_per_byte": round(ridge_point_flops_per_byte, 3),
        "roofline_ops_per_byte": round(roofline_ops_per_byte, 3),
        "throughput_tok_s": round(throughput_tok_s, 2),
        "total_power_w": round(total_power_w, 3),
        "peak_temp_c": round(peak_temp_c, 2),
        "bottleneck": bottleneck,
        "estimated_cost_delta_usd": round(estimated_cost_delta_usd, 2),
    }

    roofline = {
        "compute_ceiling_tops": round(effective_compute_tops, 3),
        "memory_bw_tb_s": round(total_bandwidth_tb_s, 4),
        "ridge_point": round(ridge_point_flops_per_byte, 3),
        "operating_ai": round(operating_ai, 3),
        "achieved_tops": round(achieved_tops, 3),
    }

    comparison_summary = (
        f"{mem['name']} ({mem['tech_class'].upper()} @ {mem['node_nm']:g}nm): "
        f"{metrics['total_capacity_gb']:g} GB, {metrics['total_bandwidth_tb_s']:.2f} TB/s, "
        f"{metrics['throughput_tok_s']:,.0f} tok/s, {metrics['total_power_w']:.1f} W "
        f"({metrics['peak_temp_c']:.1f}°C, {bottleneck})"
    )

    baseline_bw_tb_s = 8.0
    baseline_tops = ctx["base_compute_tops"]
    baseline_power_w = ctx["base_compute_power_w"] + 120.0
    baseline_temp_c = ctx["ambient_temp_c"] + baseline_power_w * ctx["thermal_resistance_c_per_w"]
    baseline_tok_s = (min(baseline_tops, baseline_bw_tb_s * operating_ai) * 1e12 / ops_per_token)
    baseline_ridge = baseline_tops / max(baseline_bw_tb_s, 1e-9)

    before_summary = {
        "name": "Baseline Interposer Memory",
        "bandwidth_tbps": round(baseline_bw_tb_s, 2),
        "tb_s": round(baseline_bw_tb_s, 2),
        "tops": round(baseline_tops, 1),
        "effective_tops": round(min(baseline_tops, baseline_bw_tb_s * operating_ai), 1),
        "tok_s": round(baseline_tok_s, 1),
        "throughput_tokens_per_sec": round(baseline_tok_s, 1),
        "total_power_w": round(baseline_power_w, 1),
        "power_w": round(baseline_power_w, 1),
        "peak_temp_c": round(baseline_temp_c, 1),
        "temp_c": round(baseline_temp_c, 1),
        "bottleneck": "memory-bandwidth" if operating_ai < baseline_ridge else "compute-bound",
    }

    after_summary = {
        "name": mem["name"],
        "bandwidth_tbps": metrics["total_bandwidth_tb_s"],
        "tb_s": metrics["total_bandwidth_tb_s"],
        "tops": metrics["effective_compute_tops"],
        "effective_tops": metrics["effective_compute_tops"],
        "tok_s": metrics["throughput_tok_s"],
        "throughput_tokens_per_sec": metrics["throughput_tok_s"],
        "total_power_w": metrics["total_power_w"],
        "power_w": metrics["total_power_w"],
        "peak_temp_c": metrics["peak_temp_c"],
        "temp_c": metrics["peak_temp_c"],
        "bottleneck": bottleneck,
    }

    result: dict[str, Any] = {
        "custom_memory": mem,
        "metrics": metrics,
        "before": before_summary,
        "after": after_summary,
        "bandwidth_tbps": metrics["total_bandwidth_tb_s"],
        "effective_tops": metrics["effective_compute_tops"],
        "throughput_tokens_per_sec": metrics["throughput_tok_s"],
        "total_power_w": metrics["total_power_w"],
        "peak_temp_c": metrics["peak_temp_c"],
        "bottleneck": bottleneck,
        "roofline": roofline,
        "comparison_summary": comparison_summary,
        "design_context": {
            "design_id": ctx["design_id"],
            "design_name": ctx["design_name"],
            "source": ctx["source"],
        },
    }
    if device_scaling is not None:
        result["device_scaling"] = device_scaling
    return result


def routes(
    lab_wrap: Callable[[Callable[[dict[str, Any]], Any]], Any],
    store: ExperimentStore,
) -> list[Route]:
    """Return Starlette routes for the Custom Memory Type Explorer API."""

    async def presets(_: Request) -> JSONResponse:
        return JSONResponse({"presets": MEMORY_PRESETS})

    return [
        Route("/api/lab/custom_memory/presets", presets, methods=["GET"]),
        Route(
            "/api/lab/custom_memory",
            lab_wrap(lambda d: evaluate_custom_memory(store, d)),
            methods=["POST"],
        ),
        Route(
            "/api/lab/custom_memory/compare",
            lab_wrap(
                lambda d: {
                    "comparisons": [
                        evaluate_custom_memory(store, {**d, "memory": m})
                        for m in d.get("memories", [])
                    ]
                }
            ),
            methods=["POST"],
        ),
    ]
