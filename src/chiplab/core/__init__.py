"""Core utilities for ChipLab, including GPU/CUDA acceleration and backend detection."""

from chiplab.core.gpu import (
    GPUBackend,
    get_device_info,
    get_gpu_backend,
    is_cuda_available,
)

__all__ = [
    "GPUBackend",
    "get_device_info",
    "get_gpu_backend",
    "is_cuda_available",
]
