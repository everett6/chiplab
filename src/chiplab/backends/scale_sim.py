"""SCALE-Sim backend: cycle-level systolic-array simulation of a workload's GEMMs.

A compute chiplet is mapped to ``n_arrays`` square systolic arrays (side: ``compute.array_dim``
if set, else the largest power of two <= sqrt(mac_units), capped at 128 like a TPU MXU).
Each GEMM's N dimension is split across the arrays and one array's share is simulated. To bound run time (a full 8x4096x4096 GEMM takes ~45 s and
3 GB in SCALE-Sim), M, N and K are capped at multiples of the array size and cycles are
scaled by the fold ratio; results say when this extrapolation was used. ChipLab takes the
array utilization (useful MACs / (cycles x array MACs)) as the chiplet's efficiency.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from ..schemas.common import FidelityLevel
from .base import RunnerBackend

SYSTOLIC_ARCHS = ("systolic", "systolic_array", "tensor_engine", "tensor", "npu", "matrix",
                  "tpu")
DATAFLOWS = ("ws", "os", "is")
MAX_ARRAY = 128
CAP_FOLDS = 4      # simulate at most 4 folds of N and K per GEMM
CAP_M = 512


def is_systolic(architecture: str | None) -> bool:
    a = str(architecture or "").lower()
    return any(k in a for k in SYSTOLIC_ARCHS)


def array_geometry(mac_units: int, side: int | None = None) -> tuple[int, int]:
    """(array side, number of arrays) for a chiplet with ``mac_units`` MACs."""
    if mac_units < 16:
        raise ValueError("mac_units must be >= 16")
    if side is None:
        side = min(MAX_ARRAY, 2 ** int(math.log2(math.isqrt(mac_units))))
    if not 4 <= side <= 1024:
        raise ValueError("array side must be in [4, 1024]")
    return side, max(1, mac_units // (side * side))


def transformer_gemms(hidden: int, batch: int, seq: int, decode: bool, num_layers: int,
                      kv_fraction: float = 1.0, active_experts: int = 1
                      ) -> list[dict[str, Any]]:
    """GEMMs of one forward pass (per token step for decode): QKV, attention output,
    FFN up and down per layer. Attention score/context GEMMs are not included."""
    m = batch if decode else batch * seq
    h = hidden
    kv = max(1, int(round(h * kv_fraction)))
    per_layer = [
        ("qkv", m, h + 2 * kv, h, 1),
        ("attn_out", m, h, h, 1),
        ("ffn_up", m, 4 * h, h, active_experts),
        ("ffn_down", m, h, 4 * h, active_experts),
    ]
    return [{"name": n, "M": mm, "N": nn, "K": kk, "count": c * num_layers}
            for n, mm, nn, kk, c in per_layer]


class ScaleSimBackend(RunnerBackend):
    name: str = "scale_sim"
    display_name: str = "SCALE-Sim"
    fidelity: FidelityLevel = FidelityLevel.SPECIALIZED
    runner: str = "_scalesim_runner.py"
    env_var: str = "CHIPLAB_SCALESIM_PYTHON"
    default_python: str = "~/.local/opt/scalesim-venv/bin/python"
    probe_module: str = "scalesim"
    timeout_s: float = 900.0

    def simulate_gemms(self, gemms: list[dict[str, Any]], mac_units: int,
                       sram_mb: float = 8.0, dataflow: str = "ws",
                       array_dim: int | None = None) -> dict[str, Any]:
        """Simulate ``gemms`` ([{name, M, N, K, count}]) on a chiplet with ``mac_units``
        MACs split into systolic arrays. Returns per-GEMM cycles and the time-weighted
        utilization of the chiplet over the whole list."""
        if dataflow not in DATAFLOWS:
            raise ValueError(f"dataflow must be one of {DATAFLOWS}")
        if not 1 <= len(gemms) <= 32:
            raise ValueError("1-32 GEMMs")
        side, n_arrays = array_geometry(int(mac_units), array_dim)
        per_kb = max(64, min(1 << 16, int(sram_mb * 1024 / n_arrays)))
        shapes = tuple((str(g.get("name", f"g{i}")), int(g["M"]), int(g["N"]), int(g["K"]),
                        int(g.get("count", 1))) for i, g in enumerate(gemms))
        return dict(_simulate_cached(self.python(), shapes, side, n_arrays, per_kb, dataflow))


@lru_cache(maxsize=128)
def _simulate_cached(python: str, shapes: tuple, side: int, n_arrays: int, per_kb: int,
                     dataflow: str) -> dict[str, Any]:
    sim_shapes, scales, fulls = [], [], []
    for _name, m, n, k, _count in shapes:
        n_arr = max(1, math.ceil(n / n_arrays))            # one array's share of N
        m_s = min(m, CAP_M)
        n_s = min(n_arr, CAP_FOLDS * side)
        k_s = min(k, CAP_FOLDS * side)
        scale = ((m / m_s) * (math.ceil(n_arr / side) / math.ceil(n_s / side))
                 * (math.ceil(k / side) / math.ceil(k_s / side)))
        sim_shapes.append({"M": m_s, "N": n_s, "K": k_s})
        scales.append(scale)
        fulls.append((m, n_arr, k))
    raw = ScaleSimBackend().call_runner({
        "array_rows": side, "array_cols": side, "dataflow": dataflow,
        "ifmap_kb": max(16, per_kb * 2 // 5), "filter_kb": max(16, per_kb * 2 // 5),
        "ofmap_kb": max(16, per_kb // 5), "gemms": sim_shapes})
    rows = []
    busy_cycles = 0.0
    useful_macs = 0.0
    for (name, m, n, k, count), layer, scale, (fm, fn, fk) in zip(shapes, raw["layers"],
                                                                  scales, fulls):
        cycles = layer["total_cycles"] * scale
        macs = fm * fn * fk
        rows.append({"name": name, "M": m, "N": n, "K": k, "count": count,
                     "array_share_N": fn, "simulated": {"M": layer["M"], "N": layer["N"],
                                                        "K": layer["K"]},
                     "extrapolation_factor": round(scale, 3),
                     "cycles_per_array": cycles,
                     "utilization": macs / (cycles * side * side) if cycles else 0.0,
                     "mapping_efficiency_pct": layer["mapping_efficiency_pct"]})
        busy_cycles += cycles * count
        useful_macs += macs * count
    return {
        "array_side": side, "n_arrays": n_arrays, "dataflow": dataflow,
        "sram_kb_per_array": per_kb, "gemms": rows,
        "cycles_per_array_total": busy_cycles,
        "utilization": useful_macs / (busy_cycles * side * side) if busy_cycles else 0.0,
        "extrapolated": any(s > 1.0 for s in scales),
        "tool": f"SCALE-Sim {raw.get('scalesim_version', '3.0.0')}",
        "note": ("Cycles exclude SCALE-Sim's initial prefetch (assumed overlapped with the "
                 "previous layer); interface bandwidth in CALC mode (not DRAM-limited)."),
    }
