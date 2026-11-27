"""GPU benchmark utilities for ChipLab."""

from chiplab.core.gpu import GPUBackend, get_device_info, is_cuda_available


def get_gpu_status() -> dict:
    """Return structured GPU status information.

    Returns a dictionary with GPU availability, device info, and backend details.
    """
    if is_cuda_available():
        info = get_device_info()
        return {
            "gpu_available": True,
            "backend": info.backend,
            "is_gpu": info.is_gpu,
            "device": info.device,
            "device_name": info.device_name,
            "total_memory_mb": info.total_memory_mb,
            "torch_available": info.torch_available,
            "cupy_available": info.cupy_available,
            "numba_cuda_available": info.numba_cuda_available,
        }
    else:
        return {
            "gpu_available": False,
            "backend": "numpy_cpu",
            "is_gpu": False,
            "device": "cpu",
            "device_name": "CPU",
            "total_memory_mb": 0.0,
            "torch_available": False,
            "cupy_available": False,
            "numba_cuda_available": False,
        }


def run_gpu_benchmark(size: int = 128) -> dict:
    """Run a simple GPU benchmark matrix operation.

    Performs a matrix multiplication of size x size matrices on GPU if available,
    otherwise falls back to CPU. Returns timing and result information.

    Args:
        size: Size of square matrices to multiply (default 128).

    Returns:
        Dictionary with benchmark results including success, backend used,
        execution time, and result shape.
    """
    from chiplab.core.gpu import get_gpu_backend

    backend = get_gpu_backend()
    results = {
        "size": size,
        "backend_used": backend.backend_name,
        "is_gpu": backend.is_gpu,
        "success": False,
        "error": None,
    }

    try:
        import time
        import numpy as np

        # Create random matrices
        A = np.random.rand(size, size).astype(np.float64)
        B = np.random.rand(size, size).astype(np.float64)

        # Warmup run
        _ = backend.batch_matmul(A, B)

        # Benchmark run
        start = time.perf_counter()
        C = backend.batch_matmul(A, B)
        elapsed = time.perf_counter() - start

        results["success"] = True
        results["execution_time_ms"] = round(elapsed * 1000, 3)
        results["result_shape"] = list(C.shape)
        results["result_dtype"] = str(C.dtype)

    except Exception as e:
        results["error"] = str(e)
        results["execution_time_ms"] = None

    return results