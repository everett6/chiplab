from typing import Any, List, Dict
from chiplab.schemas.design import DesignSpec, ChipletSpec
from chiplab.schemas.workload import WorkloadSpec, OperatorSpec, expand_workload_operators


def _count_chiplets_by_type(design: DesignSpec) -> Dict[str, int]:
    """Count total chiplets available by type."""
    counts: Dict[str, int] = {}
    for chiplet in design.chiplets:
        ctype = chiplet.type
        counts[ctype] = counts.get(ctype, 0) + chiplet.count
    return counts


def _map_operator_to_group(op: OperatorSpec, chiplet_counts: Dict[str, int]) -> Dict[str, Any]:
    """Map a single operator to the best-matching chiplet group."""
    preferred = op.preferred_chiplet_type

    # Determine target chiplet type based on operator preference and type
    if preferred:
        pref_lower = preferred.lower()
        # PIM-related targets
        if "pim" in pref_lower and "analog" in pref_lower:
            target_type = "analog_pim"
        elif "pim" in pref_lower:
            target_type = "pim"
        # Tensor/systolic engines → digital compute
        elif any(
            kw in pref_lower
            for kw in ["tensor_engine", "systolic_array", "tensor"]
        ):
            target_type = "digital_compute"
        # Vector processor → digital compute
        elif "vector_processor" in pref_lower:
            target_type = "digital_compute"
        # Memory-related types → memory_compute
        elif any(
            kw in pref_lower for kw in ["sram", "memory", "dramsim", "hbm"]
        ):
            target_type = "memory_compute"
        else:
            target_type = "digital_compute"
    else:
        # No explicit preference; decide based on operator type
        if op.op_type in {"matrix_multiply", "attention"}:
            target_type = "digital_compute"
        elif op.op_type in {"convolution"}:
            target_type = "digital_compute"
        elif op.op_type in {"dot_product", "gating"}:
            target_type = "digital_compute"
        else:
            target_type = "digital_compute"

    # Available chiplets of this type
    available = chiplet_counts.get(target_type, 0)
    used_count = max(1, min(op.parallelism if op.parallelism else 1, available)) if available > 0 else 1

    # Inter-chiplet bytes: data that must move between chiplets for this operator
    inter_chiplet_bytes = op.bytes_read + op.bytes_write

    return {
        "name": op.name,
        "op_type": op.op_type,
        "flops": op.flops,
        "bytes_read": op.bytes_read,
        "bytes_write": op.bytes_write,
        "target_chiplet_type": target_type,
        "chiplet_count": used_count,
        "inter_chiplet_bytes": inter_chiplet_bytes,
    }


def schedule_operators(design: DesignSpec, workload: WorkloadSpec) -> List[dict[str, Any]]:
    """Schedule operators from a workload onto the available chiplet groups.

    Uses ``expand_workload_operators`` to get the operator list, then maps each
    operator to the best-matching chiplet group:

    - ``memory_compute`` / ``pim`` / ``analog_pim`` for memory-bound or local ops
    - ``digital_compute`` / ``compute`` for dense matmuls / attention

    Returns a list of dicts with ``name``, ``op_type``, ``flops``, ``bytes_read``,
    ``bytes_write``, ``target_chiplet_type``, ``chiplet_count``, ``inter_chiplet_bytes``.
    """
    operators = expand_workload_operators(workload)
    chiplet_counts = _count_chiplets_by_type(design)

    results: List[dict[str, Any]] = []
    for op in operators:
        results.append(_map_operator_to_group(op, chiplet_counts))

    return results