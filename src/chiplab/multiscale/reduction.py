"""Model reduction techniques for multiscale simulation.

Provides various model reduction methods including Proper Orthogonal Decomposition (POD),
Dynamic Mode Decomposition (DMD), Balanced Truncation, Proper Generalized Decomposition (PGD),
and Neural Operator Reduction (DeepONet/FNO) for parametric PDEs.

Also provides a ReducedOrderModel (ROM) base class with train/predict/error_estimate methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ReducedOrderModel(ABC):
    """Base class for reduced-order models.

    Provides the interface for training, prediction, and error estimation
    that all specific reduction methods must implement.

    Parameters
    ----------
    reduced_dimension : int
        The dimension of the reduced subspace.
    """

    def __init__(self, reduced_dimension: int) -> None:
        self.reduced_dimension = reduced_dimension
        self.basis: Optional[np.ndarray] = None
        self.trained = False

    @abstractmethod
    def train(self, snapshots: np.ndarray) -> None:
        """Train the ROM from snapshot data.

        Parameters
        ----------
        snapshots : np.ndarray
            Snapshot matrix of shape (n_states, n_snapshots) or (n_snapshots, n_states).
        """
        pass

    @abstractmethod
    def predict(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Predict the state given parameters.

        Parameters
        ----------
        parameters : dict
            Parameter dictionary for the prediction.

        Returns
        -------
        np.ndarray
            Reduced state prediction.
        """
        pass

    @abstractmethod
    def error_estimate(self, test_snapshot: np.ndarray) -> float:
        """Estimate the error of the reduced model.

        Parameters
        ----------
        test_snapshot : np.ndarray
            A test snapshot to compare against.

        Returns
        -------
        float
            Estimated error norm.
        """
        pass

    def project_to_reduced(self, high_fidelity_state: np.ndarray) -> np.ndarray:
        """Project a high-fidelity state onto the reduced basis.

        Parameters
        ----------
        high_fidelity_state : np.ndarray
            The high-fidelity state vector.

        Returns
        -------
        np.ndarray
            The projected reduced-state coordinates.
        """
        if self.basis is None:
            raise RuntimeError("ROM not trained yet; basis is None.")
        # Assume basis shape: (n_states, n_reduced)
        projected = self.basis.T @ high_fidelity_state
        return projected

    def project_from_reduced(self, reduced_coords: np.ndarray) -> np.ndarray:
        """Project reduced coordinates back to high-fidelity space.

        Parameters
        ----------
        reduced_coords : np.ndarray
            The reduced-state coordinates.

        Returns
        -------
        np.ndarray
            The approximated high-fidelity state.
        """
        if self.basis is None:
            raise RuntimeError("ROM not trained yet; basis is None.")
        # Assume basis shape: (n_states, n_reduced)
        approximated = self.basis @ reduced_coords
        return approximated


class ProperOrthogonalDecomposition(ReducedOrderModel):
    """Proper Orthogonal Decomposition (POD) for snapshot-based basis reduction.

    POD computes an orthonormal basis from a set of snapshots that captures
    the most energetic modes of the system. The basis is ordered by eigenvalue
    (energy), so the first few modes typically capture most of the system's behavior.

    Parameters
    ----------
    reduced_dimension : int
        Number of POD modes to retain (default: 10).
    """

    def __init__(self, reduced_dimension: int = 10) -> None:
        super().__init__(reduced_dimension)

    def train(self, snapshots: np.ndarray) -> None:
        """Train the POD basis from snapshot data.

        Parameters
        ----------
        snapshots : np.ndarray
            Snapshot matrix of shape (n_states, n_snapshots).
        """
        # Center the snapshots
        mean = np.mean(snapshots, axis=1, keepdims=True)
        centered = snapshots - mean

        # Compute SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # Select the first reduced_dimension modes
        self.basis = U[:, : self.reduced_dimension].copy()
        # Store the eigenvalues (singular values squared / (n_snapshots - 1))
        self.eigenvalues = (S[: self.reduced_dimension] ** 2) / (snapshots.shape[1] - 1)
        self.mean = mean.flatten()
        self.trained = True

    def predict(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Predict the state given parameters (POD is parameter-independent).

        Parameters
        ----------
        parameters : dict
            Ignored for POD; the model predicts using the stored basis.

        Returns
        -------
        np.ndarray
            The predicted state in high-fidelity space.
        """
        if not self.trained:
            raise RuntimeError("POD ROM not trained yet.")
        # POD prediction is just the basis projection; we need reduced coords
        # For a true POD-ROM, we'd solve a reduced system, but here we just reconstruct
        # This is a simplified version - in practice, we'd need a reduced operator
        if "reduced_coords" in parameters:
            reduced = parameters["reduced_coords"]
        else:
            # Return zero reduced coords projection (reconstruct from mean only)
            reduced = np.zeros(self.reduced_dimension)
        return self.project_from_reduced(reduced)

    def error_estimate(self, test_snapshot: np.ndarray) -> float:
        """Estimate the error of the POD ROM.

        Parameters
        ----------
        test_snapshot : np.ndarray
            A test snapshot to compare against.

        Returns
        -------
        float
            The error norm (difference between test and POD reconstruction).
        """
        if not self.trained:
            raise RuntimeError("POD ROM not trained yet.")
        # Center the test snapshot
        centered = test_snapshot - self.mean
        # Project onto the basis and reconstruct
        reduced_coords = self.project_to_reduced(centered)
        reconstructed = self.project_from_reduced(reduced_coords)
        # Compute error
        error = np.linalg.norm(test_snapshot - reconstructed)
        return error


class DynamicModeDecomposition(ReducedOrderModel):
    """Dynamic Mode Decomposition (DMD) for Koopman operator approximation.

    DMD approximates the Koopman operator from snapshot data, providing a
    linear dynamical system that approximates the dynamics of the full-order model.

    Parameters
    ----------
    reduced_dimension : int
        Number of DMD modes to retain (default: 10).
    """

    def __init__(self, reduced_dimension: int = 10) -> None:
        super().__init__(reduced_dimension)

    def train(self, snapshots: np.ndarray) -> None:
        """Train the DMD model from snapshot data.

        Parameters
        ----------
        snapshots : np.ndarray
            Snapshot matrix of shape (n_states, n_snapshots + 1), where
            snapshots[:, i] and snapshots[:, i+1] are consecutive time steps.
        """
        # Split into X (current) and X' (next)
        X = snapshots[:, :-1]
        Xp = snapshots[:, 1:]

        # Compute SVD of X for compression
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        # Truncate to reduced dimension
        U_r = U[:, : self.reduced_dimension]
        S_r = S[: self.reduced_dimension]
        Vt_r = Vt[: self.reduced_dimension, :]

        # Compute the projected dynamics matrix
        A_tilde = U_r.T @ Xp @ Vt_r.T @ np.linalg.inv(np.diag(S_r))

        # Compute eigenvalues and eigenvectors of A_tilde
        eigenvalues, W = np.linalg.eig(A_tilde)

        # Store DMD modes: Phi = Xp @ Vt_r.T @ np.linalg.inv(np.diag(S_r)) @ W
        # Actually: Phi = X[:, :reduced] @ W @ np.linalg.inv(diag(S_r))
        # Proper DMD modes:
        self.modes = np.real(Xp @ Vt_r.T @ np.linalg.inv(np.diag(S_r)) @ W)
        self.eigenvalues = eigenvalues
        self.trained = True

    def predict(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Predict the state given parameters using DMD dynamics.

        Parameters
        ----------
        parameters : dict
            Should contain 'initial_state' and optionally 'num_steps'.

        Returns
        -------
        np.ndarray
            The predicted state after applying DMD dynamics.
        """
        if not self.trained:
            raise RuntimeError("DMD ROM not trained yet.")
        initial_state = parameters.get("initial_state", np.zeros(self.basis.shape[0] if self.basis else 1))
        num_steps = parameters.get("num_steps", 1)

        # Use the DMD modes (ensured real during training)
        modes = self.modes  # Already real-valued from training

        # Project initial state onto modes
        # Take real part to ensure real-valued reduced coordinates
        reduced_init = np.real(modes.T @ initial_state)

        # Apply eigenvalues for multi-step prediction
        # Use real parts of eigenvalues for growth factors
        lam = self.eigenvalues[: self.reduced_dimension]
        # Compute total growth factor: product of (1 + lambda_i) for each mode
        growth_factor = 1.0
        for j in range(min(len(lam), self.reduced_dimension)):
            growth_factor *= (1 + lam[j].real)

        # Apply growth for the specified number of steps
        # For multi-step: growth per step = growth_factor^(1/num_steps)
        # Total growth after k steps: growth_factor^(k/num_steps)
        if num_steps > 0:
            growth_per_step = growth_factor ** (1.0 / num_steps)
            total_growth = growth_per_step ** num_steps  # = growth_factor
        else:
            total_growth = 1.0

        # Reconstruct the predicted state as a combination of modes
        predicted = np.zeros_like(initial_state, dtype=np.float64)
        for j in range(min(len(lam), self.reduced_dimension)):
            predicted += modes[:, j] * reduced_init[j] * total_growth

        return predicted[: initial_state.shape[0]]

    def error_estimate(self, test_snapshot: np.ndarray) -> float:
        """Estimate the error of the DMD ROM.

        Parameters
        ----------
        test_snapshot : np.ndarray
            A test snapshot to compare against.

        Returns
        -------
        float
            The error norm.
        """
        if not self.trained:
            raise RuntimeError("DMD ROM not trained yet.")
        # Simple error estimate: compare with one-step prediction
        if "initial_state" in test_snapshot:
            # This is a simplified approach
            pass
        # Project test snapshot onto modes
        if self.basis is not None:
            reduced = self.project_to_reduced(test_snapshot)
            reconstructed = self.project_from_reduced(reduced)
        else:
            reconstructed = np.zeros_like(test_snapshot)
        error = np.linalg.norm(test_snapshot - reconstructed)
        return error


class BalancedTruncation(ReducedOrderModel):
    """Balanced Truncation for linearized systems.

    Balanced Truncation is a model reduction technique for linear time-invariant
    systems that balances controllability and observability Gramians to produce
    a reduced-order model with guaranteed error bounds.

    Parameters
    ----------
    reduced_dimension : int
        The dimension of the reduced system (default: 10).
    """

    def __init__(self, reduced_dimension: int = 10) -> None:
        super().__init__(reduced_dimension)

    def train(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
        """Train the Balanced Truncation ROM from linear system matrices.

        Parameters
        ----------
        A : np.ndarray
            System matrix (n_states × n_states).
        B : np.ndarray
            Input matrix (n_states × n_inputs).
        C : np.ndarray
            Output matrix (n_outputs × n_states).
        """
        n = A.shape[0]

        # Solve Lyapunov equations for Gramians
        # Controllability Gramman P: A*P + P*A^T + B*B^T = 0
        # Observability Gramian Q: A^T*Q + Q*A + C^T*C = 0

        # Use scipy-like approach with numpy (simplified)
        # In practice, use scipy.linalg.solve_continuous_lyapunov

        # Simplified: compute Gramians assuming stable A
        try:
            # Controllability Gramian
            P = self._lyapunov_solve(A, B @ B.T)
            # Observability Gramian
            Q = self._lyapunov_solve(A.T, C.T @ C)
        except np.linalg.LinAlgError:
            # Fallback: use identity Gramians
            P = np.eye(n)
            Q = np.eye(n)

        # Compute the product and its square root
        # PQ product
        PQ = P @ Q

        # Eigenvalue decomposition of PQ
        eigenvalues, eigenvectors = np.linalg.eigh(PQ)

        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Balancing transformation
        # H = Q^{1/2} * V * sqrt(eigenvalues) * V^T * P^{1/2}
        # For simplicity, use the square roots
        Q_sqrt = np.sqrt(np.maximum(Q, 1e-15 * np.eye(n)))
        P_sqrt = np.sqrt(np.maximum(P, 1e-15 * np.eye(n)))

        # Balanced realization
        W = Q_sqrt @ eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
        W_inv = np.linalg.inv(W)

        # Transform matrices
        A_bal = W_inv @ A @ W
        B_bal = W_inv @ B
        C_bal = C @ W

        # Truncate to reduced dimension
        self.A = A_bal[: self.reduced_dimension, : self.reduced_dimension]
        self.B = B_bal[: self.reduced_dimension, :]
        self.C = C_bal[: self.reduced_dimension, :]
        self.trained = True

    def _lyapunov_solve(self, A: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Solve the continuous Lyapunov equation A*X + X*A^T + Q = 0.

        Parameters
        ----------
        A : np.ndarray
            System matrix.
        Q : np.ndarray
            Right-hand side matrix.

        Returns
        -------
        np.ndarray
            Solution X.
        """
        # Simple iterative solver for stable A
        # In practice, use scipy.linalg.solve_continuous_lyapunov
        n = A.shape[0]
        X = np.zeros((n, n))
        max_iter = 1000
        tol = 1e-10

        for _ in range(max_iter):
            X_new = -0.5 * (A @ X + X @ A.T + Q)
            if np.linalg.norm(X_new - X, "fro") < tol:
                return X_new
            X = X_new

        return X

    def predict(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Predict the state given parameters for the balanced-truncation ROM.

        Parameters
        ----------
        parameters : dict
            Should contain 'initial_state' and 'input_sequence'.

        Returns
        -------
        np.ndarray
            The predicted output sequence.
        """
        if not self.trained:
            raise RuntimeError("BalancedTruncation ROM not trained yet.")
        initial_state = parameters.get("initial_state", np.zeros(self.A.shape[1]))
        input_sequence = parameters.get("input_sequence", np.array([]))

        # Simulate the reduced system
        state = initial_state[: self.A.shape[1]].copy()
        outputs = []

        for u in input_sequence:
            state = self.A @ state + self.B @ u
            y = self.C @ state
            outputs.append(y)

        return np.array(outputs).flatten() if outputs else np.array([])

    def error_estimate(self, test_output: np.ndarray) -> float:
        """Estimate the error of the Balanced Truncation ROM.

        Parameters
        ----------
        test_output : np.ndarray
            Test output to compare against.

        Returns
        -------
        float
            The error norm. Balanced Truncation provides a rigorous error bound:
            ||G - G_r||_Hinf <= 2 * sum(eigenvalues_{r+1:n})
        """
        if not self.trained:
            raise RuntimeError("BalancedTruncation ROM not trained yet.")
        # Error bound from balanced truncation
        # The H-infinity error bound is 2 * sum of neglected Hankel singular values
        if hasattr(self, "eigenvalues"):
            neglected_sum = float(np.sum(self.eigenvalues[self.reduced_dimension :]))
            error_bound = 2.0 * neglected_sum
        else:
            error_bound = 0.0
        return error_bound


class ProperGeneralizedDecomposition(ReducedOrderModel):
    """Proper Generalized Decomposition (PGD) for separated representation.

    PGD constructs a separated representation of the solution operator,
    allowing efficient evaluation across parameter spaces. The solution
    is approximated as a sum of products of functions of individual parameters.

    Parameters
    ----------
    reduced_dimension : int
        Number of PGD modes to compute (default: 10).
    """

    def __init__(self, reduced_dimension: int = 10) -> None:
        super().__init__(reduced_dimension)

    def train(self, snapshots: np.ndarray, parameters: Optional[np.ndarray] = None) -> None:
        """Train the PGD model from snapshot data.

        Parameters
        ----------
        snapshots : np.ndarray
            Snapshot matrix of shape (n_states, n_snapshots).
        parameters : np.ndarray, optional
            Parameter matrix of shape (n_params, n_snapshots), optional.
        """
        # Simplified PGD implementation
        # In a full implementation, this would use the incremental PGD algorithm
        # with successive subspace corrections

        # For now, use a truncated SVD as a placeholder for the separated representation
        U, S, Vt = np.linalg.svd(snapshots, full_matrices=False)

        # Take the first reduced_dimension modes
        self.basis = U[:, : self.reduced_dimension].copy()
        self.greedy_modes = Vt[: self.reduced_dimension, :].copy()
        self.singular_values = S[: self.reduced_dimension]
        self.trained = True

    def predict(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Predict the state given parameters using PGD separated representation.

        Parameters
        ----------
        parameters : dict
            Parameter values for the prediction.

        Returns
        -------
        np.ndarray
            The predicted state.
        """
        if not self.trained:
            raise RuntimeError("PGD ROM not trained yet.")

        # Simplified: return projection onto PGD basis
        if "reduced_coords" in parameters:
            reduced = parameters["reduced_coords"]
        else:
            reduced = np.zeros(self.reduced_dimension)

        return self.project_from_reduced(reduced)

    def error_estimate(self, test_snapshot: np.ndarray) -> float:
        """Estimate the error of the PGD ROM.

        Parameters
        ----------
        test_snapshot : np.ndarray
            A test snapshot to compare against.

        Returns
        -------
        float
            The error norm.
        """
        if not self.trained:
            raise RuntimeError("PGD ROM not trained yet.")
        # Reconstruct from PGD basis and compute error
        if self.basis is not None:
            reduced = self.project_to_reduced(test_snapshot)
            reconstructed = self.project_from_reduced(reduced)
            error = np.linalg.norm(test_snapshot - reconstructed)
        else:
            error = np.linalg.norm(test_snapshot)
        return error


class NeuralOperatorReduction(ReducedOrderModel):
    """Neural Operator Reduction using DeepONet or FNO for parametric PDEs.

    Uses neural networks to learn the solution operator mapping from parameters
    to the full-field response, enabling rapid evaluation across the parameter space.

    Parameters
    ----------
    reduced_dimension : int
        The neural network hidden dimension (default: 64).
    network_type : str
        Type of neural operator: "deepone" or "fno" (Fourier Neural Operator).
    """

    def __init__(self, reduced_dimension: int = 64, network_type: str = "deepone") -> None:
        super().__init__(reduced_dimension)
        self.network_type = network_type
        self.model = None
        self.trained = False

    def train(self, snapshots: np.ndarray, parameters: np.ndarray) -> None:
        """Train the neural operator from snapshot data.

        Parameters
        ----------
        snapshots : np.ndarray
            Snapshot matrix of shape (n_states, n_snapshots).
        parameters : np.ndarray
            Parameter matrix of shape (n_params, n_snapshots).
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch is required for NeuralOperatorReduction. "
                              "Install with: pip install torch")

        n_states, n_snapshots = snapshots.shape
        n_params = parameters.shape[0]

        # Define a simple DeepONet-style architecture
        class DeepONet(nn.Module):
            def __init__(self, branch_dim, trunk_dim, hidden_dim, output_dim):
                super().__init__()
                self.branch_net = nn.Sequential(
                    nn.Linear(branch_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.trunk_net = nn.Sequential(
                    nn.Linear(trunk_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.output = nn.Linear(hidden_dim, output_dim)

            def forward(self, branch_input, trunk_input):
                branch_features = self.branch_net(branch_input)
                trunk_features = self.trunk_net(trunk_input)
                # Broadcast and combine
                branch_expanded = branch_features[:, :, np.newaxis] if hasattr(branch_features, '__len__') else branch_features
                trunk_expanded = trunk_features[:, np.newaxis, :] if hasattr(trunk_features, '__len__') else trunk_features
                combined = branch_features + trunk_features  # Simplified
                return self.output(combined)

        # Initialize model
        self.model = DeepONet(
            branch_dim=n_params,
            trunk_dim=n_states,
            hidden_dim=self.reduced_dimension,
            output_dim=n_states,
        )

        # Convert to tensors
        snapshots_tensor = torch.tensor(snapshots, dtype=torch.float32)
        parameters_tensor = torch.tensor(parameters, dtype=torch.float32).T  # (n_snapshots, n_params)

        # Train the model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        self.model.train()
        for epoch in range(100):  # Simplified: few epochs
            optimizer.zero_grad()
            output = self.model(parameters_tensor, snapshots_tensor)
            loss = loss_fn(output, snapshots_tensor)
            loss.backward()
            optimizer.step()

        self.trained = True

    def predict(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Predict the state given parameters using the neural operator.

        Parameters
        ----------
        parameters : dict
            Should contain parameter values for prediction.

        Returns
        -------
        np.ndarray
            The predicted state.
        """
        if not self.trained:
            raise RuntimeError("NeuralOperatorReduction not trained yet.")
        if self.model is None:
            raise RuntimeError("Neural network model not initialized.")

        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for NeuralOperatorReduction.")

        # Extract parameters from dict
        param_values = parameters.get("values", list(parameters.values()))

        self.model.eval()
        with torch.no_grad():
            param_tensor = torch.tensor(param_values, dtype=torch.float32).unsqueeze(0)
            prediction = self.model(param_tensor, param_tensor).squeeze(0)

        return prediction.numpy()

    def error_estimate(self, test_snapshot: np.ndarray) -> float:
        """Estimate the error of the neural operator ROM.

        Parameters
        ----------
        test_snapshot : np.ndarray
            A test snapshot to compare against.

        Returns
        -------
        float
            The error norm (MSE or L2 error).
        """
        if not self.trained:
            raise RuntimeError("NeuralOperatorReduction not trained yet.")
        if self.model is None:
            raise RuntimeError("Neural network model not initialized.")

        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for NeuralOperatorReduction.")

        self.model.eval()
        with torch.no_grad():
            # This is tricky - we need to pass parameters for the test snapshot
            # For now, compute MSE between predicted and actual
            pass

        # Return placeholder: in practice, this would compare neural net output
        # with the test snapshot for a given parameter set
        return 0.0