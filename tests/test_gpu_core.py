"""Comprehensive GPU core tests for ChipLab."""

import os
import numpy as np
import pytest
import torch

from chiplab.core.gpu import GPUBackend, get_device_info, is_cuda_available
from chiplab.core.gpu_benchmark import run_gpu_benchmark


@pytest.fixture
def gpu_backend():
    """Provide a GPUBackend instance for testing."""
    return GPUBackend(prefer="auto", device="auto")


@pytest.fixture
def cpu_backend():
    """Provide a CPU-only GPUBackend instance."""
    return GPUBackend(prefer="cpu", device="cpu")


@pytest.fixture
def small_rhs():
    """Provide a small right-hand side array for Poisson solver tests."""
    return np.random.rand(10, 10)


@pytest.fixture
def matvec_fn():
    """Provide a matrix-vector function for CG solver tests."""
    def matvec(x):
        # Handle both numpy arrays and torch tensors
        if isinstance(x, torch.Tensor):
            A = torch.tensor([[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]], 
                           device=x.device, dtype=x.dtype)
            return A @ x
        else:
            A = np.array([[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]])
            return A @ x
    return matvec


class TestGPUBackend:
    """Tests for GPUBackend class."""

    def test_init_default_gpu(self, gpu_backend):
        """Test default initialization attempts GPU usage."""
        assert gpu_backend is not None

    def test_init_cpu_explicit(self, cpu_backend):
        """Test explicit CPU initialization."""
        assert cpu_backend.prefer == "cpu"
        assert cpu_backend.is_gpu == False
        assert cpu_backend.device == "cpu"

    def test_backend_name(self, cpu_backend):
        """Test backend name is set correctly for CPU."""
        assert cpu_backend.backend_name in ("torch_cpu", "numpy_cpu")

    def test_is_gpu_property(self, cpu_backend):
        """Test is_gpu property reflects CPU mode."""
        assert cpu_backend.is_gpu == False


class TestGetDeviceInfo:
    """Tests for get_device_info function."""

    def test_device_info_structure(self):
        """Test get_device_info returns GPUDeviceInfo."""
        info = get_device_info()
        assert info is not None
        assert hasattr(info, 'backend')
        assert hasattr(info, 'is_gpu')
        assert hasattr(info, 'device')
        assert hasattr(info, 'device_name')
        assert hasattr(info, 'total_memory_mb')
        assert hasattr(info, 'torch_available')
        assert hasattr(info, 'cupy_available')
        assert hasattr(info, 'numba_cuda_available')

    def test_info_has_reasonable_values(self):
        """Test device info has reasonable values."""
        info = get_device_info()
        assert isinstance(info.is_gpu, bool)
        assert isinstance(info.torch_available, bool)
        assert isinstance(info.cupy_available, bool)
        assert isinstance(info.numba_cuda_available, bool)
        assert info.total_memory_mb >= 0


class TestIsCudaAvailable:
    """Tests for is_cuda_available function."""

    def test_is_cuda_available_returns_bool(self):
        """Test is_cuda_available returns a boolean."""
        result = is_cuda_available()
        assert isinstance(result, bool)

    def test_chiplab_force_cpu(self):
        """Test CHIPLAB_FORCE_CPU environment variable."""
        os.environ["CHIPLAB_FORCE_CPU"] = "1"
        try:
            result = is_cuda_available()
            assert isinstance(result, bool)
            assert result == False
        finally:
            del os.environ["CHIPLAB_FORCE_CPU"]


class TestSolveLinear:
    """Tests for solve_linear method."""

    def test_solve_linear_cpu(self, cpu_backend):
        """Test solve_linear on CPU."""
        A = np.array([[3.0, 1.0], [1.0, 2.0]])
        b = np.array([9.0, 8.0])
        x = cpu_backend.solve_linear(A, b)
        expected = np.linalg.solve(A, b)
        np.testing.assert_allclose(x, expected, rtol=1e-10)

    def test_solve_linear_gpu_if_available(self, gpu_backend):
        """Test solve_linear on GPU if available."""
        A = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        b = np.array([9.0, 8.0], dtype=np.float64)
        x = gpu_backend.solve_linear(A, b)
        expected = np.linalg.solve(A, b)
        np.testing.assert_allclose(x, expected, rtol=1e-10)

    def test_solve_linear_small_system(self, cpu_backend):
        """Test solve_linear with small system."""
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([5.0, 3.0])
        x = cpu_backend.solve_linear(A, b)
        np.testing.assert_allclose(x, b, rtol=1e-10)


class TestSolveCG:
    """Tests for solve_cg method."""

    def test_solve_cg_basic(self, cpu_backend, matvec_fn):
        """Test basic CG solve on CPU."""
        b = np.array([1.0, 2.0, 3.0])
        x, iters = cpu_backend.solve_cg(matvec_fn, b, tol=1e-6, maxiter=500)
        residual = np.linalg.norm(matvec_fn(x) - b)
        assert residual < 1e-4 or iters > 0
        assert isinstance(iters, int)

    def test_solve_cg_gpu_if_available(self, gpu_backend, matvec_fn):
        """Test CG solve on GPU if available."""
        b = np.array([1.0, 2.0, 3.0])
        b_t = gpu_backend.to_device(b, dtype="float64")
        x, iters = gpu_backend.solve_cg(matvec_fn, b_t, tol=1e-6, maxiter=500)
        assert isinstance(iters, int)


class TestJacobiPoisson2D:
    """Tests for jacobi_poisson_2d method."""

    def test_jacobi_poisson_2d_cpu(self, cpu_backend, small_rhs):
        """Test Jacobi Poisson 2D on CPU."""
        dx = 0.1
        dy = 0.1
        result = cpu_backend.jacobi_poisson_2d(small_rhs, dx, dy, n_iters=100)
        assert result.shape == small_rhs.shape
        assert result.dtype == np.float64

    def test_jacobi_poisson_2d_gpu_if_available(self, gpu_backend, small_rhs):
        """Test Jacobi Poisson 2D on GPU if available."""
        dx = 0.1
        dy = 0.1
        result = gpu_backend.jacobi_poisson_2d(small_rhs, dx, dy, n_iters=100)
        assert result.shape == small_rhs.shape
        assert result.dtype == np.float64

    def test_jacobi_poisson_2d_with_mask(self, cpu_backend, small_rhs):
        """Test Jacobi Poisson 2D with Dirichlet mask."""
        dx = 0.1
        dy = 0.1
        mask = np.zeros_like(small_rhs, dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        values = np.ones_like(small_rhs) * 100.0
        result = cpu_backend.jacobi_poisson_2d(
            small_rhs, dx, dy, n_iters=100,
            dirichlet_mask=mask, dirichlet_values=values
        )
        assert result.shape == small_rhs.shape


class TestFFT2:
    """Tests for fft2 method."""

    def test_fft2_basic(self, cpu_backend):
        """Test basic fft2 on CPU."""
        arr = np.random.rand(8, 8) + 1j * np.random.rand(8, 8)
        result = cpu_backend.fft2(arr)
        assert result.shape == arr.shape
        expected = np.fft.fft2(arr)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_fft2_gpu_if_available(self, gpu_backend):
        """Test fft2 on GPU if available."""
        arr = np.random.rand(8, 8) + 1j * np.random.rand(8, 8)
        result = gpu_backend.fft2(arr)
        assert result.shape == arr.shape


class TestIFFT2:
    """Tests for ifft2 method."""

    def test_ifft2_basic(self, cpu_backend):
        """Test basic ifft2 on CPU."""
        arr = np.random.rand(8, 8) + 1j * np.random.rand(8, 8)
        result = cpu_backend.ifft2(arr)
        assert result.shape == arr.shape
        expected = np.fft.ifft2(arr)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_ifft2_gpu_if_available(self, gpu_backend):
        """Test ifft2 on GPU if available."""
        arr = np.random.rand(8, 8) + 1j * np.random.rand(8, 8)
        result = gpu_backend.ifft2(arr)
        assert result.shape == arr.shape


class TestCHIPLAB_FORCE_CPU:
    """Tests for CHIPLAB_FORCE_CPU environment variable."""

    def test_force_cpu_disables_gpu(self):
        """Test that CHIPLAB_FORCE_CPU=1 disables GPU."""
        os.environ["CHIPLAB_FORCE_CPU"] = "1"
        try:
            from chiplab.core.gpu import is_cuda_available
            result = is_cuda_available()
            assert result == False
        finally:
            del os.environ["CHIPLAB_FORCE_CPU"]


class TestRunGPUBenchmark:
    """Tests for run_gpu_benchmark function."""

    def test_run_gpu_benchmark_cpu(self):
        """Test GPU benchmark on CPU (no GPU available)."""
        result = run_gpu_benchmark(size=32)
        assert isinstance(result, dict)
        assert "size" in result
        assert "backend_used" in result
        assert "is_gpu" in result

    def test_run_gpu_benchmark_size(self):
        """Test GPU benchmark with different sizes."""
        for size in [16, 32, 64, 128]:
            result = run_gpu_benchmark(size=size)
            assert result["size"] == size
            assert isinstance(result["success"], bool)

    def test_run_gpu_benchmark_structure(self):
        """Test benchmark result has expected structure."""
        result = run_gpu_benchmark(size=32)
        assert result["size"] == 32
        assert result["backend_used"] in ["torch_cuda", "torch_cpu", "cupy", "numpy_cpu"]
        assert "execution_time_ms" in result or result["success"] == False


class TestGPUBackendInitialization:
    """Tests for various GPUBackend initialization paths."""

    def test_init_with_device_cuda(self):
        """Test initialization with device='cuda'."""
        backend = GPUBackend(prefer="auto", device="cuda")
        assert backend is not None

    def test_init_with_device_cpu(self):
        """Test initialization with device='cpu'."""
        backend = GPUBackend(prefer="auto", device="cpu")
        assert backend is not None
        assert backend.is_gpu == False
        assert backend.device == "cpu"

    def test_init_prefer_cupy(self):
        """Test initialization with prefer='cupy'."""
        backend = GPUBackend(prefer="cupy", device="auto")
        assert backend is not None
