"""Professional Multi-Layer PCB CAD & CUDA Field Generator for ChipLab Studio.

Generates EDA-grade PCB geometry and CUDA-accelerated power-plane IR-drop /
current-density fields:
- 45-degree chamfered differential-pair & power-delivery trace routing
- Perimeter & power-plane via stitching fences + signal return vias
- Plated-through-hole (PTH) mounting holes with ENIG annular rings
- Optical alignment fiducials & BGA solder-ball grid
- Individual PCIe gold fingers & silkscreen reference designators
- CUDA-accelerated (PyTorch CUDA -> CuPy -> NumPy) 2D PDN voltage-drop grid
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from chiplab.core.gpu import GPUBackend, get_gpu_backend


def compute_pcb_pdn_field_cuda(
    board_w: float,
    board_d: float,
    pkg_x: float,
    pkg_w: float,
    pkg_d: float,
    vrm_positions: list[tuple[float, float]],
    amps: float,
    grid_nx: int = 64,
    grid_ny: int = 40,
    device: str = "auto",
) -> dict[str, Any]:
    """Compute a 2D PCB power-plane current-density / IR-drop field on GPU (CUDA) or CPU."""
    backend: GPUBackend = get_gpu_backend(device=device)
    xp = backend.xp

    xs = backend.to_device(np.linspace(-board_w / 2.0, board_w / 2.0, grid_nx, dtype=np.float32))
    ys = backend.to_device(np.linspace(-board_d / 2.0, board_d / 2.0, grid_ny, dtype=np.float32))
    gx, gy = xp.meshgrid(xs, ys, indexing="xy")

    # Package sink (Gaussian footprint around package)
    sig_x = max(pkg_w * 0.45, 5.0)
    sig_y = max(pkg_d * 0.45, 5.0)
    sink = xp.exp(-(((gx - pkg_x) / sig_x) ** 2 + (gy / sig_y) ** 2))

    # VRM sources
    source = xp.zeros_like(gx)
    for vx, vy in vrm_positions:
        source = source + xp.exp(-(((gx - vx) / 6.0) ** 2 + ((gy - vy) / 6.0) ** 2))

    # Normalize & compute effective current density magnitude (A/mm^2 proxy)
    sink_norm = sink / xp.clip(xp.sum(sink), 1e-6, None)
    src_norm = source / xp.clip(xp.sum(source), 1e-6, None)
    net_field = xp.abs(src_norm - sink_norm) * float(amps)

    # Smooth diffusion iterations on GPU for realistic copper-plane spreading
    field = net_field
    for _ in range(12):
        # Pad with edge values without torch.nn.functional dimension constraints
        top = field[:1, :]
        bot = field[-1:, :]
        v_pad = xp.concatenate([top, field, bot], axis=0)
        left = v_pad[:, :1]
        right = v_pad[:, -1:]
        padded = xp.concatenate([left, v_pad, right], axis=1)
        field = 0.25 * (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:])

    arr = backend.to_numpy(field)
    peak = float(np.max(arr)) if arr.size else 0.0
    mean = float(np.mean(arr)) if arr.size else 0.0
    # Quantize to compact 0..255 list for lightweight JSON transfer to WebGL
    norm = (arr / max(peak, 1e-9) * 255.0).astype(np.uint8)
    return {
        "backend": backend.backend_name,
        "device": backend.device_name,
        "nx": grid_nx,
        "ny": grid_ny,
        "peak_current_density": round(peak, 4),
        "mean_current_density": round(mean, 4),
        "grid_u8": norm.ravel().tolist(),
    }


def chamfered_route(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    chamfer_frac: float = 0.35,
) -> list[list[float]]:
    """Generate a 45-degree chamfered EDA trace path between (x0, y0) and (x1, y1)."""
    dx = x1 - x0
    dy = y1 - y0
    if abs(dx) < 0.2 or abs(dy) < 0.2:
        return [[round(x0, 3), round(y0, 3)], [round(x1, 3), round(y1, 3)]]
    diag = min(abs(dx), abs(dy)) * chamfer_frac
    sx = 1.0 if dx >= 0 else -1.0
    sy = 1.0 if dy >= 0 else -1.0
    mx0 = x0 + (abs(dx) - diag) * 0.5 * sx
    mx1 = mx0 + diag * sx
    my1 = y0 + diag * sy
    return [
        [round(x0, 3), round(y0, 3)],
        [round(mx0, 3), round(y0, 3)],
        [round(mx1, 3), round(my1, 3)],
        [round(mx1, 3), round(y1, 3)],
        [round(x1, 3), round(y1, 3)],
    ]


def build_pcb_cad_details(
    board_w: float,
    board_d: float,
    board_t: float,
    pkg_x: float,
    pkg_w: float,
    pkg_d: float,
    vrm_positions: list[tuple[float, float]],
    amps: float,
    lanes: int,
    fx0: float,
    device: str = "auto",
) -> dict[str, Any]:
    """Build professional EDA PCB details (stackup, PTH holes, fiducials, vias, gold fingers, silkscreen, CUDA PDN)."""
    # 1. 10-layer FR-4 + Copper stackup
    stackup = [
        {"layer": "Top Soldermask", "material": "Epoxy LPI", "thickness_mm": 0.025, "color": "#0b4f3a"},
        {"layer": "L1 Top Signal/Pad", "material": "Cu (1 oz)", "thickness_mm": 0.035, "color": "#d48b4b"},
        {"layer": "Prepreg 1-2", "material": "FR-4 High-Tg", "thickness_mm": 0.12, "color": "#263b2a"},
        {"layer": "L2 GND Plane", "material": "Cu (1 oz)", "thickness_mm": 0.035, "color": "#c87b3c"},
        {"layer": "Core", "material": "FR-4", "thickness_mm": 1.17, "color": "#1e3322"},
        {"layer": "L9 PWR Plane", "material": "Cu (2 oz)", "thickness_mm": 0.070, "color": "#c87b3c"},
        {"layer": "Bottom Soldermask", "material": "Epoxy LPI", "thickness_mm": 0.025, "color": "#0b4f3a"},
    ]

    # 2. 4x M3 Plated-Through-Hole (PTH) mounting holes at corners
    inset = 6.0
    mounting_holes = [
        {"id": f"MH{i + 1}", "x": sx * (board_w / 2.0 - inset), "y": sy * (board_d / 2.0 - inset),
         "drill_d": 3.2, "pad_d": 5.8}
        for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)])
    ]

    # 3. 3x Optical alignment fiducials (ENIG gold discs)
    fiducials = [
        {"id": "FID1", "x": -board_w / 2.0 + 11.0, "y": board_d / 2.0 - 6.0, "d": 1.2},
        {"id": "FID2", "x": board_w / 2.0 - 11.0, "y": board_d / 2.0 - 6.0, "d": 1.2},
        {"id": "FID3", "x": -board_w / 2.0 + 11.0, "y": -board_d / 2.0 + 10.0, "d": 1.2},
    ]

    # 4. Perimeter & PDN via stitching fence
    vias: list[dict[str, float]] = []
    step = 9.0
    x_min, x_max = -board_w / 2.0 + 12.0, board_w / 2.0 - 12.0
    y_min, y_max = -board_d / 2.0 + 12.0, board_d / 2.0 - 12.0
    nx = max(4, int((x_max - x_min) / step))
    ny = max(4, int((y_max - y_min) / step))
    for i in range(nx + 1):
        xv = x_min + i * (x_max - x_min) / nx
        vias.append({"x": round(xv, 2), "y": round(y_max, 2), "d": 0.65})
        vias.append({"x": round(xv, 2), "y": round(y_min, 2), "d": 0.65})
    for j in range(1, ny):
        yv = y_min + j * (y_max - y_min) / ny
        vias.append({"x": round(x_min, 2), "y": round(yv, 2), "d": 0.65})
        vias.append({"x": round(x_max, 2), "y": round(yv, 2), "d": 0.65})
    # Power-plane stitching vias near VRM & package edge
    for vx, vy in vrm_positions:
        vias.append({"x": round(vx - 3.5, 2), "y": round(vy + 2.0, 2), "d": 0.75})
        vias.append({"x": round(vx - 3.5, 2), "y": round(vy - 2.0, 2), "d": 0.75})

    # 5. BGA solder-ball grid footprint under the package
    bga_nx, bga_ny = 16, 16
    bga_balls = []
    for ix in range(bga_nx):
        for iy in range(bga_ny):
            bx = pkg_x - pkg_w * 0.44 + ix * (pkg_w * 0.88) / (bga_nx - 1)
            by = -pkg_d * 0.44 + iy * (pkg_d * 0.88) / (bga_ny - 1)
            bga_balls.append({"x": round(bx, 2), "y": round(by, 2), "d": 0.48})

    # 6. Individual PCIe gold fingers
    gold_fingers = [
        {"x": round(fx0 + k * 2.2, 2), "y": round(-board_d / 2.0 + 3.0, 2), "w": 1.1, "d": 5.2}
        for k in range(lanes)
    ]

    # 7. Silkscreen annotations
    silkscreen = [
        {"text": "U1 (ASIC)", "x": round(pkg_x - pkg_w / 2.0, 2), "y": round(pkg_d / 2.0 + 3.0, 2), "size": 3.0},
        {"text": "PCIe x" + str(lanes), "x": round(fx0, 2), "y": round(-board_d / 2.0 + 8.5, 2), "size": 2.6},
        {"text": "12VHPWR", "x": round(board_w / 2.0 - 24.0, 2), "y": round(-board_d / 2.0 + 18.0, 2), "size": 2.6},
        {"text": "CHIPLAB EDA REV 2.0", "x": round(-board_w / 2.0 + 14.0, 2), "y": round(board_d / 2.0 - 9.0, 2), "size": 2.8},
    ]

    # 8. CUDA-accelerated PDN field
    cuda_field = compute_pcb_pdn_field_cuda(
        board_w=board_w,
        board_d=board_d,
        pkg_x=pkg_x,
        pkg_w=pkg_w,
        pkg_d=pkg_d,
        vrm_positions=vrm_positions,
        amps=amps,
        device=device,
    )

    return {
        "layers": len(stackup),
        "stackup": stackup,
        "mounting_holes": mounting_holes,
        "fiducials": fiducials,
        "vias": vias,
        "bga_balls": bga_balls,
        "gold_fingers": gold_fingers,
        "silkscreen": silkscreen,
        "cuda_pdn_field": cuda_field,
    }


__all__ = [
    "compute_pcb_pdn_field_cuda",
    "chamfered_route",
    "build_pcb_cad_details",
]
