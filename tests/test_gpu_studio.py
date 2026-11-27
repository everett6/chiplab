"""Tests for GPU/CPU buffer packing and self-contained HTML generation in GPU Studio."""

from __future__ import annotations

import json

import numpy as np
import pytest

from chiplab.core.gpu import GPUBackend, get_gpu_backend, get_device_info, is_cuda_available
from chiplab.studio.gpu_viz import (
    compute_particle_positions_on_gpu,
    emit_base64_buffer,
    emit_compact_float32_json,
    pack_float32_buffer,
    pack_instanced_buffer_attributes,
    pack_material_particles_for_threejs,
    threejs_instanced_mesh_html,
)


def _get_backend():
    """Return a GPUBackend, preferring GPU if available."""
    if is_cuda_available():
        return get_gpu_backend(prefer="torch", device="cuda")
    return get_gpu_backend(prefer="cpu")


# ===========================================================================
# pack_float32_buffer tests
# ===========================================================================


def test_pack_float32_buffer_byte_length_and_shape():
    """Pack float32 data; verify byte_length and shape are correct."""
    backend = get_gpu_backend(prefer="cpu")
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    packed = pack_float32_buffer(data, backend=backend)
    assert packed["byte_length"] == 5  # 5 floats * 4 bytes
    assert packed["shape"] == (5,)


def test_pack_float32_buffer_shape_2d():
    """Pack 2D data; shape is preserved."""
    backend = get_gpu_backend(prefer="cpu")
    data = np.arange(12.0, dtype=np.float64).reshape(4, 3)
    packed = pack_float32_buffer(data, backend=backend)
    assert packed["shape"] == (4, 3)
    assert packed["byte_length"] == 12  # 4*3 floats


def test_pack_float32_buffer_returns_bytes():
    """Packed data should be bytes."""
    backend = get_gpu_backend(prefer="cpu")
    data = np.array([1.0, 2.0], dtype=np.float64)
    packed = pack_float32_buffer(data, backend=backend)
    assert isinstance(packed["data"], bytes)


# ===========================================================================
# pack_instanced_buffer_attributes tests
# ===========================================================================


def test_pack_instanced_buffer_attributes_position_color():
    """Pack position + color for InstancedMesh; verify shapes and backend."""
    backend = _get_backend() or get_gpu_backend(prefer="cpu")
    # Use clipped colors in [0, 1]
    positions = np.random.rand(20, 3).astype(np.float32)
    colors = (np.random.rand(20, 3) * 255).astype(np.float32) / 255.0
    colors = np.clip(colors, 0.0, 1.0)
    attrs = pack_instanced_buffer_attributes(positions, colors=colors, backend=backend)
    assert attrs["count"] == 20
    assert attrs["position"]["shape"] == (20, 3)
    assert attrs["color"]["shape"] == (20, 3)
    assert attrs["position"]["backend_name"] == backend.backend_name
    assert attrs["color"]["backend_name"] == backend.backend_name


def test_pack_instanced_buffer_attributes_with_sizes():
    """Pack position + color + size."""
    backend = _get_backend() or get_gpu_backend(prefer="cpu")
    positions = np.random.rand(10, 3).astype(np.float32)
    sizes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)
    attrs = pack_instanced_buffer_attributes(positions, sizes=sizes, backend=backend)
    assert attrs["count"] == 10
    assert attrs["size"]["shape"] == (10,)
    assert attrs["size"] is not None


# ===========================================================================
# compute_particle_positions_on_gpu tests
# ===========================================================================


@pytest.mark.skipif(
    not is_cuda_available(),
    reason="CUDA not available; skipping GPU particle position test",
)
def test_compute_particle_positions_uniform_sphere():
    """Compute particle positions on GPU uniform sphere distribution."""
    backend = _get_backend()
    if backend is None:
        pytest.skip("No GPU available")
    result = compute_particle_positions_on_gpu(
        n_particles=100, radius=1.0, distribution="uniform_sphere", backend=backend
    )
    assert result["count"] == 100
    assert result["radius"] == 1.0


def test_compute_particle_positions_random_cube():
    """Compute particle positions in a cube distribution (CPU backend)."""
    backend = _get_backend() or get_gpu_backend(prefer="cpu")
    result = compute_particle_positions_on_gpu(
        n_particles=50, radius=2.0, distribution="random_cube", backend=backend
    )
    assert result["count"] == 50
    assert result["radius"] == 2.0


def test_compute_particle_positions_grid():
    """Compute particle positions on a grid."""
    backend = _get_backend() or get_gpu_backend(prefer="cpu")
    result = compute_particle_positions_on_gpu(
        n_particles=27, radius=1.0, distribution="grid", backend=backend
    )
    assert result["count"] == 27
    assert result["radius"] == 1.0


# ===========================================================================
# pack_material_particles_for_threejs tests
# ===========================================================================


def test_pack_material_particles_for_threejs():
    """Pack MaterialExplorer-style output for Three.js."""
    explorer_result = {
        "positions": np.random.rand(15, 3).astype(np.float32),
        "moments": np.random.rand(15, 3).astype(np.float32),
        "colors": np.random.rand(15, 3).astype(np.float32),
        "orientations": np.random.rand(15).astype(np.float32),
        "size": 0.5,
    }
    backend = _get_backend() or get_gpu_backend(prefer="cpu")
    packed = pack_material_particles_for_threejs(explorer_result, backend=backend)
    assert packed["count"] == 15
    assert packed["size"] == 0.5
    # All should have same backend_name
    bn = packed["backend_name"]
    assert packed["positions"]["backend_name"] == bn
    assert packed["moments"]["backend_name"] == bn
    assert packed["colors"]["backend_name"] == bn
    assert packed["orientations"]["backend_name"] == bn


# ===========================================================================
# threejs_instanced_mesh_html tests
# ===========================================================================


def test_threejs_instanced_mesh_html_basic():
    """Generate basic Three.js HTML with InstancedMesh."""
    backend = _get_backend() or get_gpu_backend(prefer="cpu")
    positions = np.random.rand(8, 3).astype(np.float32)
    colors = np.random.rand(8, 3).astype(np.float32)
    attrs = pack_instanced_buffer_attributes(positions, colors=colors, backend=backend)
    html = threejs_instanced_mesh_html(attrs, count=8, title="Test Particles")
    assert "<!DOCTYPE html>" in html
    assert "three.min.js" in html
    assert "InstancedBufferGeometry" in html
    assert "Test Particles" in html


def test_threejs_instanced_mesh_html_with_position_only():
    """Generate HTML with only position attribute (no colors)."""
    backend = _get_backend() or get_gpu_backend(prefer="cpu")
    positions = np.random.rand(5, 3).astype(np.float32)
    attrs = pack_instanced_buffer_attributes(positions, backend=backend)
    html = threejs_instanced_mesh_html(attrs, count=5, title="Position Only", get_color_attribute=False)
    assert "three.min.js" in html


def test_threejs_instanced_mesh_html_base64_embedding():
    """Test that HTML embeds base64-encoded buffer data."""
    backend = _get_backend() or get_gpu_backend(prefer="cpu")
    positions = np.random.rand(6, 3).astype(np.float32)
    colors = np.random.rand(6, 3).astype(np.float32)
    attrs = pack_instanced_buffer_attributes(positions, colors=colors, backend=backend)
    html = threejs_instanced_mesh_html(attrs, count=6, title="Base64 Test")
    # HTML should contain atob() for base64 decoding
    assert "atob(" in html


# ===========================================================================
# emit_base64_buffer tests
# ===========================================================================


def test_emit_base64_buffer():
    """Encode packed buffer as base64 string."""
    backend = get_gpu_backend(prefer="cpu")
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    packed = pack_float32_buffer(data, backend=backend)
    b64 = emit_base64_buffer(packed)
    assert isinstance(b64, str)
    import base64 as _b64
    raw = _b64.b64decode(b64)
    assert len(raw) == 12  # 3 floats * 4 bytes


# ===========================================================================
# Emit compact JSON tests
# ===========================================================================


def test_emit_compact_float32_json_roundtrip():
    """Test JSON round-trip with base64-encoded data."""
    backend = get_gpu_backend(prefer="cpu")
    data = np.array([0.5, -1.5, 2.75], dtype=np.float64)
    packed = pack_float32_buffer(data, backend=backend)
    compact = emit_compact_float32_json(packed)
    # Serialize and deserialize
    s = json.dumps(compact)
    restored = json.loads(s)
    assert "data_base64" in restored
    assert restored["byte_length"] == packed["byte_length"]
    import base64 as _b64
    struct = __import__("struct")
    raw = _b64.b64decode(restored["data_base64"])
    floats = [struct.unpack("f", raw[i:i+4])[0] for i in range(0, len(raw), 4)]
    assert len(floats) == 3


# ===========================================================================
# Device info test
# ===========================================================================


def test_get_device_info():
    """Test that get_device_info returns structured metadata."""
    info = get_device_info()
    assert info.is_gpu in (True, False)
    assert info.backend in ("torch_cuda", "torch_cpu", "cupy", "numpy_cpu")
    assert isinstance(info.device_name, str)
