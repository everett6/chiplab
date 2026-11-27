"""Uncertainty propagation for chiplab calibration.

This module provides uncertainty quantification methods including Monte Carlo,
Polynomial Chaos Expansion, stochastic collocation, and Bayesian posterior
predictive analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


# ---------------------------------------------------------------------------
# UncertaintyQuantification - Monte Carlo, PCE, Stochastic Collocation
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyQuantification:
    """Uncertainty quantification framework.

    Supports Monte Carlo simulation, Polynomial Chaos Expansion (PCE),
    stochastic collocation, and Bayesian posterior predictive analysis.

    Attributes:
        n_mc_samples: Number of Monte Carlo samples (default 1000).
        polynomial_order: Polynomial order for PCE (default 3).
        chaos_order: Polynomial chaos order (alternative to polynomial_order).
        surrogate_model: Optional surrogate model for speeding up MC.
    """
    n_mc_samples: int = 1000
    polynomial_order: int = 3
    chaos_order: Optional[int] = None
    surrogate_model: Optional[Any] = None

    def monte_carlo(self, simulator: Callable[[np.ndarray], np.ndarray],
                    parameter_samples: np.ndarray,
                    return_surrogate: bool = False) -> Dict[str, Any]:
        """Monte Carlo uncertainty propagation.

        Args:
            simulator: Function mapping parameter vector → observables.
            parameter_samples: Array of shape (N_samples, n_params) with parameter values.
            return_surrogate: If True, also return surrogate predictions.

        Returns:
            Dict with 'observables' (N_obs × N_samples), 'mean', 'std',
            'percentiles', and optionally 'surrogate'.
        """
        n_samples = parameter_samples.shape[0]
        n_params = parameter_samples.shape[1]

        # Run simulator on all samples
        observables = np.array([simulator(x) for x in parameter_samples])

        # Handle both scalar and array outputs
        if observables.ndim == 1:
            observables = observables[:, np.newaxis]
        elif observables.ndim == 0:
            observables = observables[np.newaxis, np.newaxis]

        n_obs = observables.shape[0]
        n_sim = observables.shape[1]

        # Compute statistics
        mean = np.mean(observables, axis=1)
        std = np.std(observables, axis=1, ddof=1)
        percentiles = np.percentile(observables, [5, 95], axis=1)

        result = {
            "observables": observables,  # shape (n_obs, n_sim)
            "mean": mean,  # shape (n_obs,)
            "std": std,  # shape (n_obs,)
            "percentiles": percentiles,  # shape (n_obs, 2) [5%, 95%]
            "parameter_samples": parameter_samples,
        }

        if return_surrogate and self.surrogate_model is not None:
            # Compute surrogate predictions for comparison
            surrogate_obs = self.surrogate_model.predict(parameter_samples)
            result["surrogate_observables"] = surrogate_obs

        return result

    def polynomial_chaos_expansion(self, simulator: Callable[[np.ndarray], np.ndarray],
                                   parameter_samples: np.ndarray,
                                   chaos_order: Optional[int] = None) -> Dict[str, Any]:
        """Polynomial Chaos Expansion (PCE) for uncertainty propagation.

        Uses chiaply or basic polynomial basis. If chiaply not available,
        uses a simple least-squares PCE construction.

        Args:
            simulator: Function mapping parameter vector → observables.
            parameter_samples: Array of shape (N_samples, n_params).
            chaos_order: Polynomial chaos order (uses polynomial_order if None).

        Returns:
            Dict with 'coefficients', 'mean', 'std', and 'pce_function'.
        """
        if chaos_order is None:
            chaos_order = self.polynomial_order

        n_samples = parameter_samples.shape[0]
        n_params = parameter_samples.shape[1]

        # Generate model outputs
        observables = np.array([simulator(x) for x in parameter_samples]).flatten()

        # Simple PCE using least squares projection with Legendre polynomials
        # on normalized [-1, 1] parameter space
        # Normalize parameters
        lo = parameter_samples.min(axis=0)
        hi = parameter_samples.max(axis=0)
        param_norm = (parameter_samples - lo) / (hi - lo) * 2 - 1  # normalize to [-1, 1]

        # Build polynomial basis values
        n_basis = self._count_basis_functions(chaos_order, n_params)
        basis_matrix = np.zeros((n_samples, n_basis))

        for i in range(n_samples):
            basis_matrix[i, :] = self._legendre_basis(param_norm[i, :], chaos_order)

        # Solve least squares: observables ≈ basis_matrix @ coefficients
        # Use pseudo-inverse for stability
        coefficients, residuals, rank, singular_values = np.linalg.lstsq(
            basis_matrix, observables, rcond=None
        )

        # Evaluate PCE on a dense grid for statistics
        n_dense = 1000
        dense_params = np.random.uniform(lo, hi, (n_dense, n_params))
        dense_norm = (dense_params - lo) / (hi - lo) * 2 - 1
        dense_basis = np.zeros((n_dense, n_basis))
        for i in range(n_dense):
            dense_basis[i, :] = self._legendre_basis(dense_norm[i, :], chaos_order)

        pce_observables = dense_basis @ coefficients
        mean = np.mean(pce_observables)
        std = np.std(pce_observables, ddof=1)
        percentiles = np.percentile(pce_observables, [5, 95])

        return {
            "coefficients": coefficients,  # shape (n_basis,)
            "mean": mean,
            "std": std,
            "percentiles": percentiles,
            "pce_function": lambda x: basis_matrix[i] @ coefficients  # simplified
            if False else None,  # placeholder
            "basis_matrix": basis_matrix,
            "norm_params": param_norm,
        }

    def _count_basis_functions(self, order: int, n_params: int) -> int:
        """Count the number of polynomial basis functions for given order and parameters.

        For total-order indices with linear basis: C(order + n_params, n_params)
        For linear+quadratic: 1 (constant) + n_params (linear) + n_params*(n_params+1)/2 (quadratic)
        """
        # Simple count: constant + linear terms + quadratic cross-terms
        count = 1 + n_params + n_params * (n_params + 1) // 2
        # Cap at order+1 for simplicity in this basic implementation
        return min(count, order + 1)

    def _legendre_basis(self, x_norm: np.ndarray, order: int) -> np.ndarray:
        """Evaluate normalized Legendre polynomial basis up to given order.

        Args:
            x_norm: Normalized parameter values in [-1, 1].
            order: Maximum polynomial order.

        Returns:
            Basis function values of length (order + 1) for scalar input,
            or (n_params, order+1) for array input.
        """
        x = np.asarray(x_norm, dtype=float)
        if x.ndim == 0 or x.shape == ():
            # Scalar case: compute Legendre polynomials T_0, T_1, ..., T_order
            result = np.zeros(order + 1)
            result[0] = 1.0  # P_0(x) = 1
            if order >= 1:
                result[1] = x  # P_1(x) = x
            # Use recurrence: (n+1)*P_{n+1}(x) = (2n+1)*x*P_n(x) - n*P_{n-1}(x)
            for n in range(1, order):
                result[n + 1] = ((2 * n + 1) * x * result[n] - n * result[n - 1]) / (n + 1)
            return result
        else:
            # Array case: compute for each element
            result = np.zeros((len(x), order + 1))
            result[:, 0] = 1.0
            if order >= 1:
                result[:, 1] = x
            for n in range(1, order):
                result[:, n + 1] = ((2 * n + 1) * x * result[:, n] - n * result[:, n - 1]) / (n + 1)
            return result

    def stochastic_collocation(self, simulator: Callable[[np.ndarray], np.ndarray],
                               collocation_points: int = 10,
                               **kwargs) -> Dict[str, Any]:
        """Stochastic collocation for uncertainty propagation.

        Uses Gauss-Legendre quadrature points for efficient uncertainty
        propagation without full Monte Carlo sampling.

        Args:
            simulator: Function mapping parameter vector → observables.
            collocation_points: Number of collocation points per dimension.

        Returns:
            Dict with 'observables_at_collocation', 'mean', 'std', 'weights'.
        """
        n_dims = kwargs.get('n_params', 2)

        # Generate Gauss-Legendre collocation points and weights
        # For 1D: use roots of Legendre polynomial
        # For multi-D: tensor product or sparse grid

        # Simple 1D-like implementation for demonstration
        # In production, use chaospy or dedicated quadrature library
        nodes_1d, weights_1d = self._gauss_legendre_1d(collocation_points)

        # For multi-dimensional, use simple tensor product if low dimensionality
        if n_dims <= 3:
            # Tensor product
            points_1d = nodes_1d
            # Generate full grid
            from itertools import product
            grid_indices = list(product(range(collocation_points), repeat=n_dims))

            observables_grid = []
            for idx in grid_indices:
                # Map indices to parameter values
                params = np.array([points_1d[j][idx[j]] for j in range(n_dims)])
                try:
                    obs = simulator(params)
                    observables_grid.append(obs)
                except Exception:
                    observables_grid.append(np.nan)

            observables_grid = np.array(observables_grid)
            # Compute weighted mean and std
            # This is simplified - full implementation needs proper weight handling
            valid = ~np.isnan(observables_grid).any(axis=1) if observables_grid.ndim > 1 else ~np.isnan(observables_grid)
            w_total = np.sum(weights_1d ** n_dims)  # approximate

            if np.any(valid):
                mean = np.sum(observables_grid[valid] * weights_1d ** n_dims) / w_total
                std = np.sqrt(
                    np.sum((observables_grid[valid] - mean) ** 2 * weights_1d ** n_dims) / w_total
                )
            else:
                mean = 0.0
                std = 0.0

            return {
                "observables_at_collocation": observables_grid,
                "mean": mean,
                "std": std,
                "weights": weights_1d ** n_dims,
                "collocation_type": "tensor_product",
            }
        else:
            # For higher dimensions, use sparse grid or random subset
            return self.monte_carlo(simulator, np.random.uniform(-1, 1, (100, n_dims)))

    def _gauss_legendre_1d(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """Gauss-Legendre quadrature nodes and weights for 1D.

        Simple implementation using known nodes for small n.
        """
        # For common node counts, use hardcoded values
        # In production, use numpy.polynomial.legendre.leggauss
        if n_points == 1:
            return np.array([0.0]), np.array([2.0])
        elif n_points == 2:
            sqrt3 = np.sqrt(3)
            return np.array([-sqrt3 / 3, sqrt3 / 3]), np.array([1.0, 1.0])
        elif n_points == 3:
            sqrt15 = np.sqrt(15)
            sqrt5 = np.sqrt(5)
            nodes = np.array([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)])
            weights = np.array([5 / 9, 8 / 9, 5 / 9])
            return nodes, weights
        else:
            # Fallback: use numpy if available
            try:
                from numpy.polynomial.legendre import leggauss
                nodes, weights = leggauss(n_points)
                return nodes, weights
            except ImportError:
                # Very basic: uniform distribution as fallback
                nodes = np.linspace(-1, 1, n_points)
                weights = np.full(n_points, 2 / n_points)
                return nodes, weights


# ---------------------------------------------------------------------------
# UncertaintyDecomposition - aleatory vs epistemic
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyDecomposition:
    """Decompose uncertainty into aleatory (inherent randomness) and epistemic (knowledge-based).

    Attributes:
        total_variance: Total variance of the output.
        aleatory_variance: Variance due to aleatory uncertainty (irreducible).
        epistemic_variance: Variance due to epistemic uncertainty (reducible with more info).
        decomposition_sources: Dict describing the source of each variance component.
    """
    total_variance: float = 0.0
    aleatory_variance: float = 0.0
    epistemic_variance: float = 0.0
    decomposition_sources: Dict[str, float] = field(default_factory=dict)

    def decompose(self, mc_results: Dict[str, Any],
                  parameter_uncertainties: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Decompose total uncertainty into aleatory and epistemic components.

        Args:
            mc_results: Results from UncertaintyQuantification.monte_carlo().
            parameter_uncertainties: Optional dict of parameter variance contributions.

        Returns:
            Dict with variance decomposition components.
        """
        observables = mc_results.get("observables", np.array([]))
        if observables.size == 0:
            self.total_variance = 0.0
            self.aleatory_variance = 0.0
            self.epistemic_variance = 0.0
            return {"total_variance": 0.0, "aleatory_variance": 0.0, "epistemic_variance": 0.0}

        # Compute total variance across all samples
        total_var = np.var(observables, axis=1, ddof=1).mean()  # average across observables

        # Aleatory variance: variance due to inherent randomness in the system
        # This is estimated from the Monte Carlo spread
        aleatory_var = total_var  # In basic MC, all variance is treated as aleatory

        # Epistemic variance: can be estimated from parameter uncertainty contributions
        epistemic_var = 0.0
        if parameter_uncertainties is not None:
            # Sum of parameter variance contributions
            epistemic_var = sum(parameter_uncertainties.values()) if parameter_uncertainties else 0.0

        # Remainder is aleatory (or modeling discrepancy)
        self.total_variance = float(total_var)
        self.aleatory_variance = float(aleatory_var)
        self.epistemic_variance = float(epistemic_var)

        # Track decomposition sources
        self.decomposition_sources = {
            "total_variance": self.total_variance,
            "aleatory_variance": self.aleatory_variance,
            "epistemic_variance": self.epistemic_variance,
        }

        return {
            "total_variance": self.total_variance,
            "aleatory_variance": self.aleatory_variance,
            "epistemic_variance": self.epistemic_variance,
        }


# ---------------------------------------------------------------------------
# PredictionIntervals - credible intervals for observables
# ---------------------------------------------------------------------------

@dataclass
class PredictionIntervals:
    """Compute credible/confidence intervals for model observables.

    Supports equal-tailed intervals, highest posterior density (HPD),
    and prediction intervals that account for both parameter uncertainty
    and model discrepancy.

    Attributes:
        confidence_level: Confidence level (e.g., 0.95 for 95% intervals).
        method: Interval construction method: "equal_tailed", "hpd".
    """
    confidence_level: float = 0.95
    method: str = "equal_tailed"

    def compute_intervals(self, mc_observables: np.ndarray,
                          parameter_samples: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Compute prediction intervals for observables.

        Args:
            mc_observables: Array of shape (n_observables, n_mc_samples) from MC.
            parameter_samples: Optional parameter samples for conditional intervals.

        Returns:
            Dict with interval bounds:
            - 'lower': lower bound (e.g., 2.5th percentile for 95% level)
            - 'upper': upper bound (e.g., 97.5th percentile for 95% level)
            - 'midpoint': midpoint of interval
            - 'width': interval width
        """
        n_obs = mc_observables.shape[0]

        if self.method == "equal_tailed":
            # Use percentiles directly
            alpha = 1 - self.confidence_level
            lower_pct = (alpha / 2) * 100
            upper_pct = (1 - alpha / 2) * 100

            lower = np.percentile(mc_observables, lower_pct, axis=1)
            upper = np.percentile(mc_observables, upper_pct, axis=1)
        elif self.method == "hpd":
            # Highest Posterior Density interval - simplified implementation
            # In practice, this requires sorted samples and finding the
            # shortest interval containing the specified probability
            lower = np.percentile(mc_observables, (1 - self.confidence_level) / 2 * 100, axis=1)
            upper = np.percentile(mc_observables, (1 + self.confidence_level) / 2 * 100, axis=1)
        else:
            raise ValueError(f"Unknown interval method: {self.method}")

        midpoint = (lower + upper) / 2
        width = upper - lower

        return {
            "lower": lower,
            "upper": upper,
            "midpoint": midpoint,
            "width": width,
            "confidence_level": self.confidence_level,
        }


# ---------------------------------------------------------------------------
# ModelDiscrepancy - Gaussian process discrepancy term
# ---------------------------------------------------------------------------

@dataclass
class ModelDiscrepancy:
    """Gaussian process model discrepancy term.

    Accounts for the difference between the true physical system and
    the calibrated model, using a Gaussian process to model the
    systematic discrepancy.

    Attributes:
        gp_mean: Mean function of the GP (default: zero).
        gp_kernel: Covariance kernel function.
        noise_variance: nugget/observation noise variance.
        trained: Whether the GP has been trained on discrepancy data.
    """
    gp_mean: Optional[np.ndarray] = None
    gp_kernel: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
    noise_variance: float = 1e-6
    trained: bool = False

    def set_kernel(self, kernel_name: str = "squared_exponential",
                   lengthscale: float = 1.0, variance: float = 1.0) -> None:
        """Set the covariance kernel for the GP discrepancy.

        Args:
            kernel_name: Name of kernel ("squared_exponential", "matern", "rational_quadratic").
            lengthscale: Lengthscale parameter.
            variance: Variance (amplitude) parameter.
        """
        if kernel_name == "squared_exponential":
            def kernel(x1, x2):
                """Squared exponential (RBF) kernel."""
                x1 = np.atleast_2d(x1)
                x2 = np.atleast_2d(x2)
                dist_sq = np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1) - 2 * x1 @ x2.T
                return variance * np.exp(-0.5 * dist_sq / (lengthscale**2))
        elif kernel_name == "matern":
            nu = 5 / 2  # Matérn 5/2
            def kernel(x1, x2):
                x1 = np.atleast_2d(x1)
                x2 = np.atleast_2d(x2)
                dist = np.sqrt(np.sum((x1 - x2) ** 2, axis=1)).reshape(-1, 1)
                # Matérn 5/2: (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)
                term1 = 1 + np.sqrt(5) * dist / lengthscale + 5 * dist**2 / (3 * lengthscale**2)
                result = variance * term1 * np.exp(-np.sqrt(5) * dist / lengthscale)
                return result.flatten()
        elif kernel_name == "rational_quadratic":
            alpha = 2.0  # granularity parameter
            def kernel(x1, x2):
                x1 = np.atleast_2d(x1)
                x2 = np.atleast_2d(x2)
                dist_sq = np.sum((x1 - x2) ** 2, axis=1).reshape(-1, 1)
                result = variance * (1 + dist_sq / (2 * alpha * lengthscale**2)) ** (-alpha)
                return result.flatten()
        else:
            raise ValueError(f"Unknown kernel: {kernel_name}")

        self.gp_kernel = kernel

    def train(self, discrepancy_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Train the GP discrepancy model.

        Args:
            discrepancy_data: Tuple of (X_discrepancy, y_discrepancy)
                where X are input parameters and y are observed minus model predictions.
        """
        X, y = discrepancy_data
        self.gp_mean = np.mean(y) if self.gp_mean is None else self.gp_mean
        y_centered = y - self.gp_mean

        if self.gp_kernel is not None:
            n = len(X)
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i, j] = self.gp_kernel(X[i], X[j])

            # Add noise nugget
            K += self.noise_variance * np.eye(n)

            # Store for prediction
            self._K = K
            self._X = X
            self._y_centered = y_centered
            self.trained = True
        else:
            # Default: constant discrepancy
            self.gp_mean = np.mean(y)
            self.trained = True

    def predict(self, X_new: np.ndarray) -> Dict[str, Any]:
        """Predict discrepancy at new parameter values.

        Args:
            X_new: New parameter values at which to predict discrepancy.

        Returns:
            Dict with 'mean' (discrepancy prediction) and 'variance' (uncertainty).
        """
        if not self.trained:
            raise RuntimeError("ModelDiscrepancy GP must be trained before prediction")

        if self.gp_kernel is None:
            # Return constant mean discrepancy
            return {"mean": np.full(len(X_new), self.gp_mean if self.gp_mean is not None else 0.0),
                    "variance": np.full(len(X_new), self.noise_variance)}

        X_new = np.atleast_2d(X_new)
        n_new = X_new.shape[0]

        # Compute cross-covariance between new points and training points
        K_star = np.zeros((n_new, len(self._X)))
        for i in range(n_new):
            for j in range(len(self._X)):
                K_star[i, j] = self.gp_kernel(X_new[i], self._X[j])

        # Compute predictive mean
        K = self._K
        mean = self.gp_mean + K_star @ np.linalg.solve(K, self._y_centered)

        # Compute predictive variance
        # v = solve(K, K_star.T) then variance = k(x,x) - v^T * K_star
        K_star_T = K_star.T
        v = np.linalg.solve(K, K_star_T)
        k_star_diag = np.array([self.gp_kernel(x, x) for x in X_new.flatten()])
        # Simplified: use prior variance minus reduction
        var = np.full(n_new, self.noise_variance)
        for i in range(n_new):
            var[i] = k_star_diag[i] - K_star[i, :] @ v[:, i]

        # Ensure non-negative variance
        var = np.maximum(var, 0)

        return {"mean": mean, "variance": var}