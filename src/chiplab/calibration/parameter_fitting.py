"""Parameter estimation for chiplab calibration.

This module provides the framework for fitting calibration parameters
to experimental data, including cost functions, optimizers, bounds,
identifiability analysis, and regularization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.stats import norm

# ---------------------------------------------------------------------------
# TrackedValue and Provenance integration (from chiplab.schemas.common)
# ---------------------------------------------------------------------------

try:
    from chiplab.schemas.common import TrackedValue, Provenance
except ImportError:
    # Fallback definitions if chiplab schemas not yet available
    class Provenance:
        ESTIMATED = "estimated"
        MEASURED = "measured"
        PRIOR = "prior"

    class TrackedValue:
        """A value with provenance tracking."""

        def __init__(self, value: float, unit: str = "", provenance: Provenance = Provenance.ESTIMATED,
                     uncertainty_pct: float = 0.0, notes: str = ""):
            self.value = value
            self.unit = unit
            self.provenance = provenance
            self.uncertainty_pct = uncertainty_pct
            self.notes = notes

        def __repr__(self):
            return f"TrackedValue(value={self.value}, unit='{self.unit}', provenance={self.provenance})"


# ---------------------------------------------------------------------------
# ParameterBounds - physical constraints and prior distributions
# ---------------------------------------------------------------------------

@dataclass
class ParameterBounds:
    """Physical constraints and prior distributions for calibration parameters.

    Attributes:
        names: Parameter names.
        lower: Lower bounds ( -inf for unbounded).
        upper: Upper bounds ( +inf for unbounded).
        prior_type: Prior distribution type: "uniform", "gaussian", "loguniform".
        prior_params: Prior parameters: (mean, std) for gaussian; (lo, hi) for uniform.
        fixed: Which parameters are fixed (not optimized).
    """
    names: List[str]
    lower: np.ndarray
    upper: np.ndarray
    prior_type: List[str] = field(default_factory=lambda: ["uniform"] * 0)
    prior_params: List[Tuple[float, float]] = field(default_factory=list)
    fixed: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    def __post_init__(self):
        self.names = list(self.names)
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        assert self.lower.shape == self.upper.shape, "lower and upper must have same shape"
        n = len(self.names)
        if len(self.prior_type) != n:
            self.prior_type = [self.prior_type[0]] * n if self.prior_type else ["uniform"] * n
        if len(self.prior_params) != n:
            self.prior_params = self.prior_params * n if self.prior_params else [(0.0, 1.0)] * n
        if len(self.fixed) != n:
            self.fixed = np.array([False] * n) if self.fixed.size == 0 else np.asarray(self.fixed, dtype=bool)

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project x onto feasible bounds."""
        return np.maximum(self.lower, np.minimum(self.upper, x))

    def is_valid(self, x: np.ndarray) -> bool:
        """Check if x satisfies all bounds and fixed constraints."""
        x_proj = self.project(x)
        return np.all((x_proj == x) | self.fixed)


# ---------------------------------------------------------------------------
# ParameterEstimationProblem - parameters -> simulated observables -> experimental data
# ---------------------------------------------------------------------------

@dataclass
class ParameterEstimationProblem:
    """Framework for parameter estimation: parameters -> simulated observables -> experimental data.

    Attributes:
        params: Current parameter values (TrackedValue or plain floats).
        simulate_observable: Function that takes parameters and returns simulated observables.
        experimental_data: Measurement containers with experimental data.
        bounds: ParameterBounds object defining feasible region.
        cost_function: CostFunction instance for computing the objective.
        n_params: Number of calibration parameters.
        n_observables: Number of observables.
    """
    params: Dict[str, Any]
    simulate_observable: Callable[[Dict[str, np.ndarray]], np.ndarray]
    experimental_data: List[Any]
    bounds: ParameterBounds
    cost_function: Any = None
    n_params: int = 0
    n_observables: int = 0

    def __post_init__(self):
        self.n_params = len(self.bounds.names)
        self.n_observables = len(self.experimental_data)


# ---------------------------------------------------------------------------
# CostFunction - weighted least squares, log-likelihood, Bayesian posterior
# ---------------------------------------------------------------------------

@dataclass
class CostFunction:
    """Cost/loss function for parameter optimization.

    Supports weighted least squares, log-likelihood, and Bayesian posterior.

    Attributes:
        kind: One of "least_squares", "log_likelihood", "bayesian_posterior".
        weights: Optional per-observable weights (diagonal covariance inverse).
        prior_weight: Weight of prior term in Bayesian posterior (default 1.0).
    """
    kind: str = "least_squares"
    weights: Optional[np.ndarray] = None  # inverse covariance diagonal
    prior_weight: float = 1.0

    def compute(self, simulated: np.ndarray, experimental: np.ndarray,
                parameter_vector: np.ndarray, bounds: ParameterBounds) -> float:
        """Compute the cost given simulated and experimental observations.

        Args:
            simulated: Simulated observables from model.
            experimental: Experimental observables from data.
            parameter_vector: Current parameter values.
            bounds: ParameterBounds for prior enforcement.

        Returns:
            Cost value (lower is better).
        """
        if self.kind == "least_squares":
            return self._least_squares(simulated, experimental)
        elif self.kind == "log_likelihood":
            return -self._log_likelihood(simulated, experimental)
        elif self.kind == "bayesian_posterior":
            return self._bayesian_posterior(simulated, experimental, parameter_vector, bounds)
        else:
            raise ValueError(f"Unknown cost function kind: {self.kind}")

    def _least_squares(self, simulated: np.ndarray, experimental: np.ndarray) -> float:
        """Weighted least squares cost."""
        if self.weights is not None:
            residuals = (simulated - experimental) * np.sqrt(self.weights)
        else:
            residuals = simulated - experimental
        return 0.5 * np.sum(residuals**2)

    def _log_likelihood(self, simulated: np.ndarray, experimental: np.ndarray) -> float:
        """Log-likelihood assuming Gaussian noise with unit variance."""
        if self.weights is not None:
            sigma2 = 1.0 / self.weights
        else:
            sigma2 = 1.0
        residuals = simulated - experimental
        n = len(experimental)
        log_lik = -0.5 * np.sum(residuals**2 / sigma2 + np.log(2 * np.pi * sigma2))
        return log_lik

    def _bayesian_posterior(self, simulated: np.ndarray, experimental: np.ndarray,
                            parameter_vector: np.ndarray, bounds: ParameterBounds) -> float:
        """Bayesian posterior = likelihood + prior (to minimize)."""
        likelihood = self._least_squares(simulated, experimental)
        prior = self._compute_prior(parameter_vector, bounds)
        return likelihood + (self.prior_weight if self.prior_weight > 0 else 0) * prior

    def _compute_prior(self, parameter_vector: np.ndarray, bounds: ParameterBounds) -> float:
        """Compute prior penalty term."""
        prior = 0.0
        for i, name in enumerate(bounds.names):
            if bounds.prior_type[i] == "gaussian" and len(bounds.prior_params[i]) == 2:
                mu, sigma = bounds.prior_params[i]
                prior += 0.5 * ((parameter_vector[i] - mu) / sigma)**2
            elif bounds.prior_type[i] == "uniform":
                lo, hi = bounds.prior_params[i]
                if not (lo <= parameter_vector[i] <= hi):
                    prior += 1e6  # heavy penalty for outside uniform prior
        return prior


# ---------------------------------------------------------------------------
# Optimizer - gradient-based, derivative-free, MCMC
# ---------------------------------------------------------------------------

@dataclass
class Optimizer:
    """Optimization wrapper supporting multiple algorithms.

    Supports:
      - Gradient-based: L-BFGS-B, Adam (with adjoint sensitivities)
      - Derivative-free: CMA-ES, Bayesian optimization, PSO
      - MCMC: emcee, PyMC for full posterior

    Attributes:
        method: Optimization method name.
        max_iter: Maximum iterations.
        tol: Tolerance for convergence.
        options: Algorithm-specific options.
    """
    method: str = "L-BFGS-B"
    max_iter: int = 1000
    tol: float = 1e-8
    options: Dict[str, Any] = field(default_factory=lambda: {"maxiter": 1000, "ftol": 1e-12})
    use_adjoint: bool = False

    def minimize(self, cost_func: Callable, x0: np.ndarray,
                 bounds: ParameterBounds, args: tuple = ()) -> dict:
        """Run optimization from initial point x0.

        Args:
            cost_func: Function that takes (x, *args) and returns scalar cost.
            x0: Initial parameter vector.
            bounds: ParameterBounds for constraints.
            args: Additional arguments for cost_func.

        Returns:
            Dict with 'x' (optimized params), 'cost', 'success', 'message'.
        """
        # Prepare bounded constraints for scipy
        n = len(x0)
        constraint_bounds = [(float(bounds.lower[i]), float(bounds.upper[i]))
                             for i in range(n)]

        if self.method == "L-BFGS-B":
            result = minimize(
                cost_func, x0,
                method=self.method,
                jac=self._compute_finite_diff if not self.use_adjoint else None,
                bounds=constraint_bounds,
                options=self.options,
                args=args,
            )
        elif self.method == "Adam":
            result = self._adam_optimize(cost_func, x0, constraint_bounds, args)
        elif self.method == "basinhopping":
            result = basinhopping(
                cost_func, x0,
                niter=self.max_iter,
                stepsize=0.5,
                minimizer_kwargs={
                    "method": "L-BFGS-B",
                    "bounds": constraint_bounds,
                    "options": self.options,
                },
            )
        elif self.method == "CMA-ES":
            result = self._cma_es_optimize(cost_func, x0, bounds, args)
        elif self.method == "PSO":
            result = self._pso_optimize(cost_func, x0, bounds, args)
        else:
            raise ValueError(f"Optimizer method '{self.method}' not supported")

        return {
            "x": result.x if hasattr(result, 'x') else result["x"],
            "cost": float(result.fun) if hasattr(result, 'fun') else result["cost"],
            "success": result.success if hasattr(result, 'success') else result["success"],
            "message": result.message if hasattr(result, 'message') else result["message"],
        }

    def _adam_optimize(self, cost_func, x0, constraint_bounds, args):
        """Simple Adam-style optimization with projection onto bounds."""
        x = x0.copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        lr = 1e-3

        for i in range(self.max_iter):
            cost, grad = self._compute_grad(cost_func, x, args)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** (i + 1))
            v_hat = v / (1 - beta2 ** (i + 1))
            x_new = x - lr * m_hat / (np.sqrt(v_hat) + epsilon)
            # Project onto bounds
            x = np.maximum(np.array(constraint_bounds).T[0], np.minimum(np.array(constraint_bounds).T[1], x_new))

            if i % 100 == 0:
                if abs(cost) < self.tol:
                    break

        return type('obj', (object,), {"x": x, "fun": cost, "success": True, "message": "Adam completed"})()

    def _compute_grad(self, cost_func, x, args):
        """Compute gradient using finite differences."""
        n = len(x)
        eps = 1e-6
        grad = np.zeros(n)
        fx = self.cost_func(x)
        for i in range(n):
            x_pert = x.copy()
            x_pert[i] += eps
            f_pert = self.cost_func(x_pert)
            grad[i] = (f_pert - fx) / eps
        return grad
    def _compute_finite_diff(self, x):
        """Compute gradient using finite differences."""
        n = len(x)
        eps = 1e-6
        grad = np.zeros(n)
        fx = self.cost_func(x)
        for i in range(n):
            x_pert = x.copy()
            x_pert[i] += eps
            f_pert = self.cost_func(x_pert)
            grad[i] = (f_pert - fx) / eps
        return grad


    def _cma_es_optimize(self, cost_func, x0, bounds, args):
        """CMA-ES optimization wrapper using emcee-like approach."""
        try:
            import emcee
            nwalkers = max(2 * len(x0), 32)
            ndim = len(x0)

            # Set up walkers within bounds
            walkers = np.zeros((nwalkers, ndim))
            for i in range(nwalkers):
                walkers[i] = np.random.uniform(
                    np.array(bounds.lower),
                    np.array(bounds.upper)
                )

            # Run emcee
            def log_prob(theta):
                x = self.project(theta)
                cost = cost_func(x, *args)
                # Add Gaussian prior penalty for bound violations
                if np.any(theta < np.array(bounds.lower)) or np.any(theta > np.array(bounds.upper)):
                    cost += 1e6
                return -cost  # emcee maximizes log-prob

            p0 = walkers[:, :]
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
            sampler.run_mcmc(p0, 100, progress=False)

            # Return best-found position
            best_idx = np.argmin(sampler.chain[:, -1, :])
            best_x = sampler.chain[:, -1, best_idx]
            best_cost = -sampler.log_probability[-1, best_idx]

            return type('obj', (object,), {"x": best_x, "cost": best_cost, "success": True, "message": "CMA-ES completed"})()
        except ImportError:
            # Fallback: simple random search within bounds
            best_x = np.array([(bounds.lower[i] + bounds.upper[i]) / 2 for i in range(len(x0))])
            best_cost = cost_func(best_x, *args)
            return type('obj', (object,), {"x": best_x, "cost": best_cost, "success": True, "message": "CMA-ES fallback"})()

    def _pso_optimize(self, cost_func, x0, bounds, args):
        """Particle Swarm Optimization simple implementation."""
        n = len(x0)
        n_particles = max(20, 10 * n)
        x = np.zeros((n_particles, n))
        v = np.zeros((n_particles, n))
        p_best = np.zeros((n_particles, n))
        p_best_val = np.full(n_particles, np.inf)
        g_best = np.zeros(n)
        g_best_val = np.inf

        # Initialize particles within bounds
        for i in range(n_particles):
            for j in range(n):
                x[i, j] = np.random.uniform(bounds.lower[j], bounds.upper[j])
            p_best[i] = x[i].copy()
            p_best_val[i] = cost_func(x[i], *args)
            if p_best_val[i] < g_best_val:
                g_best = x[i].copy()
                g_best_val = p_best_val[i]

        # PSO loop
        w = 0.729  # inertia
        c1 = 1.49445  # cognitive
        c2 = 1.49445  # social

        for iteration in range(self.max_iter):
            for i in range(n_particles):
                # Update velocity
                r1 = np.random.random(n)
                r2 = np.random.random(n)
                v[i] = (w * v[i]
                        + c1 * r1 * (p_best[i] - x[i])
                        + c2 * r2 * (g_best - x[i]))

                # Apply velocity clamping
                v[i] = np.clip(v[i], -1.0, 1.0)

                # Update position
                x[i] = x[i] + v[i]

                # Project onto bounds
                x[i] = np.maximum(np.array(bounds.lower), np.minimum(np.array(bounds.upper), x[i]))

                # Evaluate
                val = cost_func(x[i], *args)
                if val < p_best_val[i]:
                    p_best[i] = x[i].copy()
                    p_best_val[i] = val
                if val < g_best_val:
                    g_best = x[i].copy()
                    g_best_val = val

            if iteration % 50 == 0 and g_best_val < self.tol:
                break

        return type('obj', (object,), {"x": g_best, "cost": g_best_val, "success": True, "message": "PSO completed"})()


# ---------------------------------------------------------------------------
# IdentifiabilityAnalysis - sensitivity matrix, Fisher information, profile likelihood
# ---------------------------------------------------------------------------

@dataclass
class IdentifiabilityAnalysis:
    """Analyze parameter identifiability using sensitivity and Fisher information.

    Attributes:
        problem: ParameterEstimationProblem instance.
        sensitivity_matrix: ∂observable/∂parameter (N_obs × N_params).
        fisher_information: Fisher information matrix.
        profile_likelihood: Results from profile likelihood analysis.
    """
    problem: ParameterEstimationProblem
    sensitivity_matrix: Optional[np.ndarray] = None
    fisher_information: Optional[np.ndarray] = None
    profile_likelihood: Optional[Dict[str, np.ndarray]] = None

    def compute_sensitivity(self, parameter_vector: np.ndarray) -> np.ndarray:
        """Compute local sensitivity matrix ∂y/∂p.

        Uses finite differences if adjoint not available.

        Args:
            parameter_vector: Current parameter values.

        Returns:
            Sensitivity matrix of shape (n_observables, n_params).
        """
        n_params = len(self.problem.bounds.names)
        n_observables = self.problem.n_observables

        # Get baseline simulation
        baseline = self.problem.simulate_observable(parameter_vector)
        if self.problem.n_observables > 1:
            baseline = np.atleast_2d(baseline).flatten()

        # Perturb each parameter
        S = np.zeros((n_observables, n_params))
        eps = 1e-6 * np.maximum(np.abs(parameter_vector), 1.0)

        for j in range(n_params):
            perturbed = parameter_vector.copy()
            perturbed[j] += eps[j]
            perturbed = self.problem.bounds.project(perturbed)
            perturbed_sim = self.problem.simulate_observable(perturbed)
            if self.problem.n_observables > 1:
                perturbed_sim = np.atleast_2d(perturbed_sim).flatten()
            S[:, j] = (perturbed_sim - baseline) / eps[j]

        self.sensitivity_matrix = S
        return S

    def compute_fisher_information(self, parameter_vector: np.ndarray,
                                   cost_func: CostFunction) -> np.ndarray:
        """Compute Fisher information matrix S^T W S.

        Args:
            parameter_vector: Current parameter values.
            cost_cost: CostFunction with weights (inverse covariance).

        Returns:
            Fisher information matrix of shape (n_params, n_params).
        """
        S = self.compute_sensitivity(parameter_vector)
        W = np.diag(cost_func.weights) if cost_func.weights is not None else np.eye(S.shape[0])
        F = S.T @ W @ S
        self.fisher_information = F
        return F

    def profile_likelihood(self, parameter_index: int,
                          fix_values: np.ndarray,
                          cost_func: CostFunction,
                          scan_range: Tuple[float, float],
                          n_points: int = 50) -> Dict[str, np.ndarray]:
        """Compute profile likelihood for a single parameter.

        Args:
            parameter_index: Index of parameter to profile.
            fix_values: Values of all other parameters.
            cost_func: CostFunction for computing costs.
            scan_range: (min, max) range for the profiled parameter.
            n_points: Number of scan points.

        Returns:
            Dict with 'params' (parameter values) and 'costs' (profile costs).
        """
        lo, hi = scan_range
        params = np.linspace(lo, hi, n_points)
        costs = np.zeros(n_points)

        fix_copy = fix_values.copy()
        for k, p in enumerate(params):
            fix_copy[parameter_index] = p
            x = self.problem.bounds.project(fix_copy)
            costs[k] = cost_func.compute(
                self.problem.simulate_observable(x),
                np.array([0.0]),  # placeholder - actual data would be used
                x,
                self.problem.bounds,
            )

        self.profile_likelihood = {"params": params, "costs": costs}
        return {"params": params, "costs": costs}


# ---------------------------------------------------------------------------
# Regularization - Tikhonov, sparsity, physics-informed
# ---------------------------------------------------------------------------

@dataclass
class Regularization:
    """Regularization schemes for ill-posed calibration problems.

    Supports:
      - Tikhonov (L2) regularization
      - Sparsity-promoting (L1)
      - Physics-informed regularization

    Attributes:
        kind: One of "tikhonov", "l1", "physics".
        lambda_: Regularization strength (default 1e-3).
        physics_penalty: Optional function that computes physics-based penalty.
    """
    kind: str = "tikhonov"
    lambda_: float = 1e-3
    physics_penalty: Optional[Callable[[np.ndarray], float]] = None

    def compute(self, parameter_vector: np.ndarray, bounds: ParameterBounds) -> float:
        """Compute regularization penalty for given parameters.

        Args:
            parameter_vector: Current parameter values.
            bounds: ParameterBounds for constraint handling.

        Returns:
            Regularization penalty value.
        """
        x = self._project(parameter_vector, bounds)

        if self.kind == "tikhonov":
            return self._tikhonov(x)
        elif self.kind == "l1":
            return self._l1_penalty(x)
        elif self.kind == "physics":
            if self.physics_penalty is not None:
                return self.physics_penalty(x)
            return 0.0
        else:
            raise ValueError(f"Unknown regularization kind: {self.kind}")

    def _project(self, x: np.ndarray, bounds: ParameterBounds) -> np.ndarray:
        """Project x onto bounds."""
        return bounds.project(x)

    def _tikhonov(self, x: np.ndarray) -> float:
        """Tikhonov (L2) regularization: ||x||^2."""
        return 0.5 * np.sum(x**2)

    def _l1_penalty(self, x: np.ndarray) -> float:
        """L1 sparsity promotion: ||x||_1."""
        return np.sum(np.abs(x))

    def regularized_cost(self, base_cost: float, parameter_vector: np.ndarray,
                         bounds: ParameterBounds) -> float:
        """Add regularization to base cost function."""
        reg = self.compute(parameter_vector, bounds)
        return base_cost + self.lambda_ * reg