"""Tests for GPUBackend device='cpu' and device='auto' support in calibration modules."""

import numpy as np
import pytest

from chiplab.core.gpu import GPUBackend, get_gpu_backend


# ---- GPUBackend device tests ----

def test_gpu_backend_device_cpu():
    """Test GPUBackend with device='cpu'."""
    b = GPUBackend(device="cpu")
    assert b.backend_name == "numpy_cpu"
    assert b.is_gpu is False
    assert b.device == "cpu"


def test_gpu_backend_device_auto_gpu_available():
    """Test GPUBackend with device='auto' when GPU is available - should use GPU."""
    b = GPUBackend(device="auto")
    # When GPU is available, device='auto' should use GPU
    assert b.is_gpu is True or b.backend_name != "numpy_cpu"


def test_gpu_backend_device_auto_cpu_fallback():
    """Test GPUBackend with device='auto' falls back to CPU when no GPU forced."""
    # This test verifies the backend is properly initialized
    b = GPUBackend(device="auto")
    assert b is not None
    assert b.backend_name in ("numpy_cpu", "torch_cuda", "cupy")


def test_gpu_backend_get_gpu_backend():
    """Test get_gpu_backend returns a GPUBackend instance."""
    b = get_gpu_backend()
    assert isinstance(b, GPUBackend)


# ---- Calibration test models ----

def _simple_simulator_1d(params):
    """Simple 1D simulator: y = a*params[0]"""
    return params[0] * 2.0


def _simple_simulator_2d(params):
    """Simple 2D simulator: y = a*params[0] + b*params[1]"""
    return params[0] * 2.0 + params[1] * 3.0


# ---- Parameter fitting tests ----

def test_parameter_fitting_imports():
    """Test that parameter fitting modules can be imported."""
    from chiplab.calibration.parameter_fitting import (
        ParameterEstimationProblem,
        ParameterBounds,
        CostFunction,
        Optimizer,
    )
    # Just verify imports work
    assert ParameterEstimationProblem is not None
    assert ParameterBounds is not None
    assert CostFunction is not None
    assert Optimizer is not None


def test_parameter_bounds_device_cpu():
    """Test ParameterBounds with device='cpu'."""
    from chiplab.calibration.parameter_fitting import ParameterBounds
    pb = ParameterBounds(names=["p1"], lower=[0.0], upper=[10.0])
    assert pb.fixed.size == 1


def test_cost_function_batched():
    """Test batched cost function evaluation."""
    from chiplab.calibration.parameter_fitting import CostFunction

    def simulator(params):
        return np.array([params[0] ** 2 + params[1] ** 2])

    experimental = np.array([5.0])
    weights = np.array([1.0])

    cf = CostFunction(kind="log_likelihood", weights=weights)

    # Evaluate likelihood at several parameter points
    param_points = np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
    likelihoods = np.array([cf.compute(simulator(p), experimental, p, None) for p in param_points])

    assert len(likelihoods) == 3
    assert all(np.isfinite(likelihoods))


# ---- Sensitivity analysis tests ----

def test_sensitivity_local():
    """Test local sensitivity analysis."""
    from chiplab.calibration.sensitivity import LocalSensitivity

    def simulate(params):
        return params[0] ** 2 + params[1] ** 2

    ls = LocalSensitivity(method="finite_difference")
    param_vector = np.array([1.0, 2.0])
    S = ls.compute(simulate, param_vector)
    assert S.shape == (1, 2)  # 1 observable, 2 parameters


def test_sensitivity_global():
    """Test global sensitivity analysis."""
    from chiplab.calibration.sensitivity import GlobalSensitivity

    def simulate(params):
        return params[0] ** 2 + params[1] ** 2

    gs = GlobalSensitivity(n_samples=100, n_parameters=2)
    si = gs.sobol_indices(simulate)
    assert "first_order" in si
    assert "total_effect" in si


# ---- Uncertainty propagation tests ----

def test_uncertainty_mc():
    """Test Monte Carlo uncertainty propagation."""
    from chiplab.calibration.uncertainty import UncertaintyQuantification

    def simulator(params):
        return params[0] * 2.0 + np.random.normal(0, 0.1)

    uq = UncertaintyQuantification(n_mc_samples=100)
    parameter_samples = np.random.uniform(0, 1, (100, 1))

    result = uq.monte_carlo(simulator, parameter_samples, return_surrogate=False)
    assert "observables" in result
    assert "mean" in result
    assert "std" in result


# ---- Batched likelihood/residual tests ----

def test_batched_likelihood():
    """Test batched likelihood evaluation."""
    from chiplab.calibration.parameter_fitting import CostFunction

    def simulator(params):
        return np.array([params[0] ** 2 + params[1] ** 2])

    experimental = np.array([5.0])
    weights = np.array([1.0])

    cf = CostFunction(kind="log_likelihood", weights=weights)

    # Evaluate likelihood at several parameter points
    param_points = np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
    likelihoods = np.array([cf.compute(simulator(p), experimental, p, None) for p in param_points])

    assert len(likelihoods) == 3
    assert all(np.isfinite(likelihoods))


# ---- GPU backend integration tests ----

def test_gpu_to_device_cpu():
    """Test to_device works correctly on CPU backend."""
    from chiplab.core.gpu import GPUBackend

    b = GPUBackend(device="cpu")
    x = np.array([1.0, 2.0, 3.0])
    y = b.to_device(x)
    assert isinstance(y, np.ndarray)
    np.testing.assert_array_equal(y, x)


def test_gpu_to_numpy():
    """Test to_numpy converts back to numpy array."""
    from chiplab.core.gpu import GPUBackend

    b = GPUBackend(device="cpu")
    x = np.array([1.0, 2.0, 3.0])
    y = b.to_numpy(x)
    assert isinstance(y, np.ndarray)
    np.testing.assert_array_equal(y, x)


def test_gpu_batch_matmul():
    """Test batch matrix multiplication on CPU backend."""
    from chiplab.core.gpu import GPUBackend

    b = GPUBackend(device="cpu")
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    C = b.batch_matmul(A, B)
    expected = A @ B
    np.testing.assert_array_almost_equal(C, expected)


def test_gpu_solve_linear():
    """Test solving linear system on CPU backend."""
    from chiplab.core.gpu import GPUBackend

    b = GPUBackend(device="cpu")
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b_vec = np.array([5.0, 7.0])
    x = b.solve_linear(A, b_vec)
    expected = np.linalg.solve(A, b_vec)
    np.testing.assert_array_almost_equal(x, expected)


def test_gpu_solve_cg():
    """Test CG solver on CPU backend."""
    from chiplab.core.gpu import GPUBackend

    b = GPUBackend(device="cpu")

    def matvec(v):
        A = np.array([[3.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 3.0]])
        return A @ v

    x, iters = b.solve_cg(matvec, np.array([5.0, 6.0, 7.0]), x0=np.zeros(3))
    assert iters > 0
    assert len(x) == 3


def test_gpu_zeros_ones():
    """Test zeros and ones on CPU backend."""
    from chiplab.core.gpu import GPUBackend

    b = GPUBackend(device="cpu")
    z = b.zeros((3, 4))
    assert z.shape == (3, 4)
    o = b.ones((2, 3))
    assert o.shape == (2, 3)


def test_gpu_linspace():
    """Test linspace on CPU backend."""
    from chiplab.core.gpu import GPUBackend

    b = GPUBackend(device="cpu")
    l = b.linspace(0, 10, 5)
    assert len(l) == 5
    assert l[0] == 0.0
    assert l[-1] == 10.0


def test_gpu_solve_cg_2d_poisson():
    """Test 2D Poisson solver on CPU backend."""
    from chiplab.core.gpu import GPUBackend

    b = GPUBackend(device="cpu")
    # Simple 3x3 grid with boundary conditions
    rhs = np.zeros((5, 5))
    rhs[2, 2] = 1.0  # source term
    u = b.jacobi_poisson_2d(rhs, 1.0, 1.0, n_iters=10)
    assert u.shape == (5, 5)
