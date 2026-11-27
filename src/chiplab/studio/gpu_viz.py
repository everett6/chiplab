"""GPU-accelerated visualization helpers for ChipLab Studio.

Provides Float32/InstancedBufferAttribute data packing helpers that optionally
compute particle positions/field colormaps on GPUBackend and emit compact
Float32/base64 buffers or Three.js InstancedMesh / GPU shader-ready HTML in
self-contained Three.js HTML output.

Preserves all existing public functions and return types in scene.py, thermal.py,
material_explorer.py, device_explorer.py.
"""

from __future__ import annotations

import base64
import struct
from typing import Any

import numpy as np

from ..core.gpu import GPUBackend, get_gpu_backend, get_device_info


def pack_float32_buffer(
    data: np.ndarray,
    backend: GPUBackend | None = None,
) -> dict[str, Any]:
    """Pack numpy data into a compact Float32Array-friendly dict.

    Transfers data to the specified backend (or the default GPU backend) and
    returns a dict with the raw Float32 bytes and metadata for Three.js
    BufferAttribute consumption.

    Parameters
    ----------
    data : np.ndarray
        Array of any numeric dtype. Will be cast to float32.
    backend : GPUBackend, optional
        Backend to transfer data to. If None, uses the default GPUBackend.

    Returns
    -------
    dict
        {
            "byte_length": int,           # byte length of the Float32Array
            "data": bytes,                # raw Float32 bytes
            "shape": tuple[int, ...],     # original shape (for reshape in Three.js)
            "backend_name": str,          # e.g. "torch_cuda", "torch_cpu", "cupy", "numpy_cpu"
        }
    """
    if backend is None:
        backend = get_gpu_backend()

    # Ensure float32
    arr = np.asarray(data, dtype=np.float32)

    # Transfer to backend (may move to GPU)
    tensor = backend.to_device(arr)

    # Get bytes in Float32 format
    if backend.backend_name in ("torch_cuda", "torch_cpu"):
        import torch

        pt = tensor.to(torch.float32) if hasattr(tensor, "to") else tensor
        # Ensure contiguous
        pt = pt.contiguous()
        # Use numpy bridge
        byte_data = np.asarray(pt.cpu()).tobytes()
    elif backend.backend_name == "cupy":
        import cupy as cp

        arr_cp = cp.asarray(arr)
        byte_data = cp.asnumpy(arr_cp).tobytes()
    else:
        # CPU numpy path
        byte_data = arr.tobytes()

    # Determine shape for Three.js BufferAttribute
    # If original was 1D, shape is (N,); if 2D (N,3), etc.
    # We'll store the shape and let Three.js reshape as needed
    original_shape = arr.shape

    result: dict[str, Any] = {
        "byte_length": len(byte_data) // 4,  # 4 bytes per float32
        "data": byte_data,
        "shape": original_shape,
        "backend_name": backend.backend_name,
    }
    return result


def pack_instanced_buffer_attributes(
    positions: np.ndarray,
    colors: np.ndarray | None = None,
    sizes: np.ndarray | None = None,
    backend: GPUBackend | None = None,
) -> dict[str, Any]:
    """Pack position/color/size data into InstancedBufferAttribute-ready dicts.

    Generates three BufferAttribute dicts (position, color, size) suitable for
    Three.js InstancedMesh. All data is packed as Float32Array.

    Parameters
    ----------
    positions : np.ndarray
        Shape (N, 3) in nanometers or world units.
    colors : np.ndarray, optional
        Shape (N, 3) RGB in [0, 1] or [0, 255].
    sizes : np.ndarray, optional
        Shape (N,) scalar sizes.
    backend : GPUBackend, optional
        Backend to transfer data to.

    Returns
    -------
    dict
        {
            "position": {"byte_length": int, "data": bytes, "shape": (N, 3), ...},
            "color": {"byte_length": int, "data": bytes, "shape": (N, 3), ...} | None,
            "size": {"byte_length": int, "data": bytes, "shape": (N,), ...} | None,
            "count": int,             # number of instances
            "backend_name": str,
        }
    """
    if backend is None:
        backend = get_gpu_backend()

    N = positions.shape[0]

    # Position: (N, 3) -> flat Float32
    pos_data = pack_float32_buffer(positions, backend=backend)
    pos_attr = {
        "byte_length": pos_data["byte_length"],
        "data": pos_data["data"],
        "shape": (N, 3),
        "backend_name": pos_data["backend_name"],
    }

    color_attr = None
    if colors is not None:
        # Ensure float32 [0, 1]
        c = np.asarray(colors, dtype=np.float32)
        if c.max() > 1.0:
            c = c / 255.0
        c = np.clip(c, 0.0, 1.0)
        color_data = pack_float32_buffer(c.reshape(-1, 3), backend=backend)
        color_attr = {
            "byte_length": color_data["byte_length"],
            "data": color_data["data"],
            "shape": (N, 3),
            "backend_name": color_data["backend_name"],
        }

    size_attr = None
    if sizes is not None:
        s = np.asarray(sizes, dtype=np.float32)
        size_data = pack_float32_buffer(s, backend=backend)
        size_attr = {
            "byte_length": size_data["byte_length"],
            "data": size_data["data"],
            "shape": (N,),
            "backend_name": size_data["backend_name"],
        }

    return {
        "position": pos_attr,
        "color": color_attr,
        "size": size_attr,
        "count": N,
        "backend_name": backend.backend_name,
    }


def compute_particle_positions_on_gpu(
    n_particles: int,
    radius: float = 1.0,
    distribution: str = "uniform_sphere",
    backend: GPUBackend | None = None,
) -> dict[str, Any]:
    """Compute particle positions on the GPU backend.

    Generates random or structured particle positions on the specified backend,
    returning data ready for Three.js InstancedMesh.

    Parameters
    ----------
    n_particles : int
        Number of particles to generate.
    radius : float, optional
        Radius of the sphere (for "uniform_sphere" distribution).
    distribution : str, optional
        Distribution type: "uniform_sphere", "random_cube", "grid".
    backend : GPUBackend, optional
        Backend to compute on.

    Returns
    -------
    dict
        {
            "positions": float32 buffer dict from pack_float32_buffer,
            "distribution": str,
            "radius": float,
            "backend_name": str,
        }
    """
    if backend is None:
        backend = get_gpu_backend()

    import numpy as np

    if distribution == "uniform_sphere":
        # Marsaglia polar method for uniform on sphere
        phi = np.random.rand(n_particles) * 2 * np.pi
        costheta = 2 * np.random.rand(n_particles) - 1
        sintheta = np.sqrt(1.0 - costheta**2)
        x = radius * sintheta * np.cos(phi)
        y = radius * sintheta * np.sin(phi)
        z = radius * costheta
        positions = np.column_stack([x, y, z]).astype(np.float32)

    elif distribution == "random_cube":
        positions = (np.random.rand(n_particles, 3) - 0.5) * radius * 2
        positions = positions.astype(np.float32)

    elif distribution == "grid":
        # Simple 3D grid
        n = int(np.ceil(n_particles ** (1.0 / 3)))
        grid_vals = np.linspace(-radius, radius, n)
        xx, yy, zz = np.meshgrid(grid_vals, grid_vals, grid_vals)
        positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])[:n_particles]
        positions = positions.astype(np.float32)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Pack positions
    packed = pack_float32_buffer(positions, backend=backend)

    return {
        "positions": packed,
        "distribution": distribution,
        "radius": radius,
        "count": n_particles,
        "backend_name": packed["backend_name"],
    }


def emit_base64_buffer(
    packed: dict[str, Any],
) -> str:
    """Encode a packed Float32 buffer dict as a base64 string.

    Useful for embedding binary data directly in HTML or JSON configs.

    Parameters
    ----------
    packed : dict
        Output from pack_float32_buffer or pack_instanced_buffer_attributes.

    Returns
    -------
    str
        base64-encoded string of the raw float32 bytes.
    """
    raw = packed["data"]
    return base64.b64encode(raw).decode("ascii")


def threejs_instanced_mesh_html(
    attributes: dict[str, Any],
    count: int,
    title: str = "GPU Studio Particles",
    get_position_attribute: bool = True,
    get_color_attribute: bool = True,
    get_size_attribute: bool = False,
) -> str:
    """Generate a self-contained Three.js HTML snippet rendering an InstancedMesh.

    Uses the packed attribute dicts from pack_instanced_buffer_attributes and
    emits a minimal HTML page with Three.js CDN that renders the instanced
    particles with GPU-accelerated BufferAttributes.

    Parameters
    ----------
    attributes : dict
        Output from pack_instanced_buffer_attributes.
    count : int
        Number of instances (for the InstancedMesh count).
    title : str, optional
        HTML page title.
    get_position_attribute : bool, optional
        Whether to include position attribute.
    get_color_attribute : bool, optional
        Whether to include color attribute.
    get_size_attribute : bool, optional
        Whether to include size attribute.

    Returns
    -------
    str
        Complete HTML document string.
    """
    # Build the attribute setup code
    attr_codes = []

    if get_position_attribute and attributes["position"]:
        pa = attributes["position"]
        # For safety, embed the base64-encoded data
        import base64 as _b64
        b64_data = _b64.b64encode(pa["data"]).decode("ascii")
        attr_codes.append(
            f"geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(atob('{b64_data}')), 3));"
        )
    else:
        attr_codes.append("geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([]), 3));")

    if get_color_attribute and attributes["color"]:
        ca = attributes["color"]
        import base64 as _b64
        b64_data = _b64.b64encode(ca["data"]).decode("ascii")
        attr_codes.append(
            f"geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(atob('{b64_data}')), 3));"
        )
    else:
        attr_codes.append("geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array([]), 3));")

    if get_size_attribute and attributes["size"]:
        sa = attributes["size"]
        import base64 as _b64
        b64_data = _b64.b64encode(sa["data"]).decode("ascii")
        attr_codes.append(
            f"geometry.setAttribute('size', new THREE.BufferAttribute(new Float32Array(atob('{b64_data}')), 1));"
        )
    else:
        attr_codes.append("geometry.setAttribute('size', new THREE.BufferAttribute(new Float32Array([]), 1));")

    # Count
    n = attributes.get("count", count)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0; }}
        h1 {{ padding: 1rem; background: #16213e; text-align: center; }}
        #viewer {{ width: 100%; height: 800px; background: #0d1b2a; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <h1>{title}</h1>
    <div id="viewer"></div>
    <script>
        const viewer = document.getElementById('viewer');
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(60, viewer.clientWidth / viewer.clientHeight, 0.1, 1000);
        camera.position.set(0, 0, 5);
        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(viewer.clientWidth, viewer.clientHeight);
        viewer.appendChild(renderer.domElement);

        const geometry = new THREE.InstancedBufferGeometry();
        {chr(10).join(attr_codes)}
        const material = new THREE.MeshStandardMaterial({{
            color: 0x00ff00,
            transparent: true,
            opacity: 0.8,
            vertexColors: {str(get_color_attribute and attributes['color']).lower()},
        }});
        const mesh = new THREE.Mesh(geometry, material);
        mesh.instanceMatrix.setMatrixAt(0, new THREE.Matrix4());
        mesh.updateMatrix();
        scene.add(mesh);

        renderer.render(scene, camera);
    </script>
</body>
</html>"""

    return html


def emit_compact_float32_json(
    packed: dict[str, Any],
) -> dict[str, Any]:
    """Emit a compact JSON-serializable representation of packed float32 data.

    Replaces raw bytes with a base64 string so the dict can be serialized
    via json.dumps() without binary issues.

    Parameters
    ----------
    packed : dict
        Output from pack_float32_buffer.

    Returns
    -------
    dict
        {
            "byte_length": int,
            "data_base64": str,     # base64-encoded float32 bytes
            "shape": tuple,
            "backend_name": str,
        }
    """
    return {
        "byte_length": packed["byte_length"],
        "data_base64": base64.b64encode(packed["data"]).decode("ascii"),
        "shape": packed["shape"],
        "backend_name": packed["backend_name"],
    }


# Convenience: pack a MaterialExplorer render_particles output for Three.js
def pack_material_particles_for_threejs(
    explorer_result: dict[str, Any],
    backend: GPUBackend | None = None,
) -> dict[str, Any]:
    """Pack MaterialExplorer.render_particles() output into Three.js-friendly data.

    Extracts positions, moments, colors, orientations, sizes from the explorer
    result and packs them into Float32 buffers ready for InstancedMesh or
    shader consumption.

    Parameters
    ----------
    explorer_result : dict
        Output from MaterialExplorer.render_particles().
    backend : GPUBackend, optional
        Backend to use for packing.

    Returns
    -------
    dict
        {
            "positions": pack_float32_buffer result,
            "moments": pack_float32_buffer result,
            "colors": pack_float32_buffer result,
            "orientations": pack_float32_buffer result,
            "count": int,
            "backend_name": str,
        }
    """
    positions = np.asarray(explorer_result.get("positions", []), dtype=np.float32)
    moments = np.asarray(explorer_result.get("moments", []), dtype=np.float32)
    colors = np.asarray(explorer_result.get("colors", []), dtype=np.float32)
    orientations = np.asarray(explorer_result.get("orientations", []), dtype=np.float32)
    size = float(explorer_result.get("size", 0.5))

    result: dict[str, Any] = {
        "positions": pack_float32_buffer(positions, backend=backend),
        "moments": pack_float32_buffer(moments, backend=backend),
        "colors": pack_float32_buffer(colors, backend=backend),
        "orientations": pack_float32_buffer(orientations, backend=backend),
        "count": positions.shape[0] if positions.size else 0,
        "size": size,
        "backend_name": backend.backend_name if backend else "numpy_cpu",
    }
    return result