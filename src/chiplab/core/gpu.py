"""Unified CUDA/GPU acceleration backend with automatic CPU fallback.

Supports PyTorch CUDA (primary, verified on RTX 5070 / CUDA 13.x), CuPy,
Numba CUDA detection, and transparent NumPy/SciPy CPU fallback so every
ChipLab physics, multiphysics, multiscale, calibration, and visualization
module can offload heavy tensor/PDE/FFT/Monte-Carlo workloads to GPU when
available without breaking CPU-only environments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np

# Optional GPU libraries (lazy / safe imports)
try:
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False

try:
    import cupy as cp  # type: ignore

    _HAS_CUPY = True
except Exception:
    cp = None  # type: ignore
    _HAS_CUPY = False

try:
    from numba import cuda as numba_cuda  # type: ignore

    _HAS_NUMBA_CUDA = bool(numba_cuda.is_available())
except Exception:
    numba_cuda = None  # type: ignore
    _HAS_NUMBA_CUDA = False


def is_cuda_available() -> bool:
    """Return True if any supported CUDA backend (PyTorch or CuPy) is usable."""
    if os.environ.get("CHIPLAB_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return False
    if _HAS_TORCH:
        try:
            if torch.cuda.is_available():
                return True
        except Exception:
            pass
    if _HAS_CUPY:
        try:
            if cp.cuda.is_available():
                return True
        except Exception:
            pass
    return False


@dataclass(frozen=True)
class GPUDeviceInfo:
    """Metadata describing the active compute device."""

    backend: str
    is_gpu: bool
    device: str
    device_name: str
    total_memory_mb: float
    torch_available: bool
    cupy_available: bool
    numba_cuda_available: bool


class GPUBackend:
    """Unified tensor/array compute backend with GPU (PyTorch/CuPy) and CPU (NumPy) support."""

    def __init__(self, prefer: str = "auto", device: str = "auto") -> None:
        self.prefer = prefer
        self.requested_device = device
        self.backend_name = "numpy_cpu"
        self.is_gpu = False
        self.device = "cpu"
        self.device_name = "CPU"
        self._torch_device = None

        force_cpu = (
            device == "cpu"
            or prefer == "cpu"
            or os.environ.get("CHIPLAB_FORCE_CPU", "").lower() in ("1", "true", "yes")
        )
        if not force_cpu:
            self._init_gpu(prefer=prefer, device=device)

    def _init_gpu(self, prefer: str, device: str) -> None:
        # 1. PyTorch CUDA (primary)
        if prefer in ("auto", "torch", "pytorch") and _HAS_TORCH:
            try:
                if torch.cuda.is_available():
                    dev_str = "cuda:0" if device in ("auto", "cuda", "gpu") else device
                    self._torch_device = torch.device(dev_str)
                    idx = self._torch_device.index or 0
                    self.backend_name = "torch_cuda"
                    self.is_gpu = True
                    self.device = str(self._torch_device)
                    self.device_name = torch.cuda.get_device_name(idx)
                    return
            except Exception:
                pass

        # 2. CuPy CUDA
        if prefer in ("auto", "cupy") and _HAS_CUPY:
            try:
                if cp.cuda.is_available():
                    props = cp.cuda.runtime.getDeviceProperties(0)
                    name = props["name"].decode("utf-8", errors="ignore")
                    self.backend_name = "cupy"
                    self.is_gpu = True
                    self.device = "cuda:0"
                    self.device_name = name
                    return
            except Exception:
                pass

        # 3. PyTorch CPU fallback if explicitly requested
        if prefer == "torch" and _HAS_TORCH:
            self._torch_device = torch.device("cpu")
            self.backend_name = "torch_cpu"
            self.is_gpu = False
            self.device = "cpu"
            self.device_name = "CPU (PyTorch)"

    @property
    def xp(self):
        """Return the array module for the active backend (numpy, torch, or cupy)."""
        if self.backend_name in ("torch_cuda", "torch_cpu"):
            return torch
        if self.backend_name == "cupy":
            return cp
        return np

    def random_normal(self, shape, mean=0.0, std=1.0):
        """Generate random normal array on the active backend."""
        xp = self.xp
        if self.backend_name in ("torch_cuda", "torch_cpu"):
            return torch.normal(mean, std, shape, device=self._torch_device, dtype=torch.float64)
        if self.backend_name == "cupy":
            return cp.random.normal(mean, std, shape)
        return np.random.normal(mean, std, shape)

    def random_uniform(self, shape, low=0.0, high=1.0):
        """Generate random uniform array on the active backend."""
        xp = self.xp
        if self.backend_name in ("torch_cuda", "torch_cpu"):
            return torch.rand(shape, device=self._torch_device, dtype=torch.float64) * (high - low) + low
        if self.backend_name == "cupy":
            return cp.random.uniform(low, high, shape)
        return np.random.uniform(low, high, shape)

    def random_integers(self, low, high, size):
        """Generate random integers on the active backend."""
        if self.backend_name in ("torch_cuda", "torch_cpu"):
            return torch.randint(low, high, (size,), device=self._torch_device, dtype=torch.int64)
        if self.backend_name == "cupy":
            return cp.random.randint(low, high, size)
        return np.random.randint(low, high, size)

    # ------------------------------------------------------------------
    # Array creation & transfers
    # ------------------------------------------------------------------
    def _torch_dtype(self, dtype: Any) -> Any:
        if dtype is None or not _HAS_TORCH:
            return torch.float64 if _HAS_TORCH else None
        if isinstance(dtype, torch.dtype):
            return dtype
        mapping = {
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.complex64: torch.complex64,
            np.complex128: torch.complex128,
            float: torch.float64,
            int: torch.int64,
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "complex64": torch.complex64,
            "complex128": torch.complex128,
        }
        return mapping.get(dtype, torch.float64)

    def to_device(self, arr: Any, dtype: Any = None) -> Any:
        """Move array/tensor/sequence onto the active backend device."""
        if self.backend_name in ("torch_cuda", "torch_cpu"):
            tdtype = self._torch_dtype(dtype) if dtype is not None else None
            if isinstance(arr, torch.Tensor):
                return arr.to(device=self._torch_device, dtype=tdtype)
            np_arr = np.asarray(arr)
            if tdtype is None:
                if np.issubdtype(np_arr.dtype, np.floating):
                    tdtype = torch.float64
                elif np.issubdtype(np_arr.dtype, np.complexfloating):
                    tdtype = torch.complex128
                elif np.issubdtype(np_arr.dtype, np.integer):
                    tdtype = torch.int64
            return torch.as_tensor(np_arr, dtype=tdtype, device=self._torch_device)
        if self.backend_name == "cupy":
            return cp.asarray(arr, dtype=dtype)
        return np.asarray(arr, dtype=dtype)

    def asarray(self, arr: Any, dtype: Any = None) -> Any:
        return self.to_device(arr, dtype=dtype)

    def to_numpy(self, arr: Any) -> np.ndarray:
        """Transfer array/tensor from GPU (or any backend) back to a NumPy ndarray."""
        if _HAS_TORCH and isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        if _HAS_CUPY and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)

    def zeros(self, shape: Union[int, Sequence[int]], dtype: Any = np.float64) -> Any:
        if self.backend_name in ("torch_cuda", "torch_cpu"):
            return torch.zeros(shape, dtype=self._torch_dtype(dtype), device=self._torch_device)
        if self.backend_name == "cupy":
            return cp.zeros(shape, dtype=dtype)
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: Union[int, Sequence[int]], dtype: Any = np.float64) -> Any:
        if self.backend_name in ("torch_cuda", "torch_cpu"):
            return torch.ones(shape, dtype=self._torch_dtype(dtype), device=self._torch_device)
        if self.backend_name == "cupy":
            return cp.ones(shape, dtype=dtype)
        return np.ones(shape, dtype=dtype)

    def linspace(self, start: float, stop: float, num: int, dtype: Any = np.float64) -> Any:
        if self.backend_name in ("torch_cuda", "torch_cpu"):
            return torch.linspace(
                float(start), float(stop), int(num), dtype=self._torch_dtype(dtype), device=self._torch_device
            )
        if self.backend_name == "cupy":
            return cp.linspace(start, stop, num, dtype=dtype)
        return np.linspace(start, stop, num, dtype=dtype)

    # ------------------------------------------------------------------
    # Linear algebra & PDE / FFT kernels
    # ------------------------------------------------------------------
    def solve_linear(self, A: Any, b: Any) -> np.ndarray:
        """Solve A x = b on the active device and return a NumPy array."""
        if self.backend_name in ("torch_cuda", "torch_cpu"):
            A_t = self.to_device(A, dtype="float64")
            b_t = self.to_device(b, dtype="float64")
            squeeze = b_t.ndim == 1
            if squeeze:
                b_t = b_t.unsqueeze(-1)
            x_t = torch.linalg.solve(A_t, b_t)
            if squeeze:
                x_t = x_t.squeeze(-1)
            return self.to_numpy(x_t)
        if self.backend_name == "cupy":
            A_c = cp.asarray(A, dtype=cp.float64)
            b_c = cp.asarray(b, dtype=cp.float64)
            return cp.asnumpy(cp.linalg.solve(A_c, b_c))
        return np.linalg.solve(np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64))

    def solve_cg(
        self,
        matvec_fn: Callable[[Any], Any],
        b: Any,
        x0: Optional[Any] = None,
        tol: float = 1e-6,
        maxiter: int = 500,
    ) -> Tuple[np.ndarray, int]:
        """Matrix-free Conjugate Gradient solver running on GPU or CPU."""
        b_d = self.to_device(b, dtype="float64")
        x = self.zeros(b_d.shape, dtype="float64") if x0 is None else self.to_device(x0, dtype="float64").clone() if hasattr(b_d, "clone") else self.to_device(x0, dtype="float64").copy()
        r = b_d - matvec_fn(x)
        p = r.clone() if hasattr(r, "clone") else r.copy()

        def _dot(u: Any, v: Any) -> float:
            if self.backend_name in ("torch_cuda", "torch_cpu"):
                return float(torch.sum(u * v).item())
            if self.backend_name == "cupy":
                return float(cp.sum(u * v).item())
            return float(np.sum(u * v))

        rsold = _dot(r, r)
        b_norm = max(np.sqrt(_dot(b_d, b_d)), 1e-30)
        iters = 0
        for iters in range(1, maxiter + 1):
            if np.sqrt(max(rsold, 0.0)) / b_norm < tol:
                break
            Ap = matvec_fn(p)
            denom = _dot(p, Ap)
            if abs(denom) < 1e-30:
                break
            alpha = rsold / denom
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = _dot(r, r)
            if np.sqrt(max(rsnew, 0.0)) / b_norm < tol:
                iters += 1
                break
            p = r + (rsnew / max(rsold, 1e-30)) * p
            rsold = rsnew
        return self.to_numpy(x), iters

    def jacobi_poisson_2d(
        self,
        rhs: np.ndarray,
        dx: float,
        dy: float,
        n_iters: int = 500,
        dirichlet_mask: Optional[np.ndarray] = None,
        dirichlet_values: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Vectorized 2D Poisson / Laplace relaxation on GPU (or CPU fallback)."""
        rhs_d = self.to_device(rhs, dtype="float64")
        u = self.zeros(rhs.shape, dtype="float64")
        mask_d = self.to_device(dirichlet_mask) if dirichlet_mask is not None else None
        val_d = self.to_device(dirichlet_values, dtype="float64") if dirichlet_values is not None else None

        dx2 = dx * dx
        dy2 = dy * dy
        denom = 2.0 * (1.0 / dx2 + 1.0 / dy2)

        for _ in range(n_iters):
            u_new = u.clone() if hasattr(u, "clone") else u.copy()
            u_new[1:-1, 1:-1] = (
                (u[2:, 1:-1] + u[:-2, 1:-1]) / dx2
                + (u[1:-1, 2:] + u[1:-1, :-2]) / dy2
                - rhs_d[1:-1, 1:-1]
            ) / denom
            if mask_d is not None and val_d is not None:
                if self.backend_name in ("torch_cuda", "torch_cpu"):
                    u_new = torch.where(mask_d.bool(), val_d, u_new)
                elif self.backend_name == "cupy":
                    u_new = cp.where(mask_d.astype(bool), val_d, u_new)
                else:
                    u_new = np.where(mask_d.astype(bool), val_d, u_new)
            u = u_new
        return self.to_numpy(u)

    def fft2(self, arr: Any) -> np.ndarray:
        if self.backend_name in ("torch_cuda", "torch_cpu"):
            t = self.to_device(arr, dtype="complex128")
            return self.to_numpy(torch.fft.fft2(t))
        if self.backend_name == "cupy":
            return cp.asnumpy(cp.fft.fft2(cp.asarray(arr)))
        return np.fft.fft2(np.asarray(arr))

    def ifft2(self, arr: Any) -> np.ndarray:
        if self.backend_name in ("torch_cuda", "torch_cpu"):
            t = self.to_device(arr, dtype="complex128")
            return self.to_numpy(torch.fft.ifft2(t))
        if self.backend_name == "cupy":
            return cp.asnumpy(cp.fft.ifft2(cp.asarray(arr)))
        return np.fft.ifft2(np.asarray(arr))

    def batch_matmul(self, a: Any, b: Any) -> np.ndarray:
        if self.backend_name in ("torch_cuda", "torch_cpu"):
            return self.to_numpy(torch.matmul(self.to_device(a), self.to_device(b)))
        if self.backend_name == "cupy":
            return cp.asnumpy(cp.matmul(cp.asarray(a), cp.asarray(b)))
        return np.matmul(np.asarray(a), np.asarray(b))

    # ------------------------------------------------------------------
    # Diagnostics & memory management
    # ------------------------------------------------------------------
    def memory_info(self) -> dict:
        info = {
            "backend": self.backend_name,
            "is_gpu": self.is_gpu,
            "device": self.device,
            "device_name": self.device_name,
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "total_mb": 0.0,
        }
        if self.backend_name == "torch_cuda" and _HAS_TORCH:
            idx = self._torch_device.index or 0
            props = torch.cuda.get_device_properties(idx)
            info["allocated_mb"] = round(torch.cuda.memory_allocated(idx) / (1024**2), 2)
            info["reserved_mb"] = round(torch.cuda.memory_reserved(idx) / (1024**2), 2)
            info["total_mb"] = round(props.total_memory / (1024**2), 2)
        elif self.backend_name == "cupy" and _HAS_CUPY:
            free_b, total_b = cp.cuda.runtime.memGetInfo()
            info["allocated_mb"] = round((total_b - free_b) / (1024**2), 2)
            info["total_mb"] = round(total_b / (1024**2), 2)
        return info

    def clear_cache(self) -> None:
        if self.backend_name == "torch_cuda" and _HAS_TORCH:
            torch.cuda.empty_cache()
        elif self.backend_name == "cupy" and _HAS_CUPY:
            cp.get_default_memory_pool().free_all_blocks()


_DEFAULT_BACKEND: Optional[GPUBackend] = None


def get_gpu_backend(prefer: str = "auto", device: str = "auto") -> GPUBackend:
    """Get a GPUBackend instance (cached singleton for default args)."""
    global _DEFAULT_BACKEND
    if prefer == "auto" and device == "auto":
        if _DEFAULT_BACKEND is None:
            _DEFAULT_BACKEND = GPUBackend(prefer=prefer, device=device)
        return _DEFAULT_BACKEND
    return GPUBackend(prefer=prefer, device=device)


def get_device_info() -> GPUDeviceInfo:
    """Return structured device/backend availability metadata."""
    backend = get_gpu_backend()
    mem = backend.memory_info()
    return GPUDeviceInfo(
        backend=backend.backend_name,
        is_gpu=backend.is_gpu,
        device=backend.device,
        device_name=backend.device_name,
        total_memory_mb=float(mem.get("total_mb", 0.0)),
        torch_available=_HAS_TORCH,
        cupy_available=_HAS_CUPY,
        numba_cuda_available=_HAS_NUMBA_CUDA,
    )
