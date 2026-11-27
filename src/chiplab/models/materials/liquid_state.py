"""
Liquid State Machine (Reservoir Computing) with Memristor/Photonic Reservoirs
=============================================================================

Implements Liquid State Machines (LSM) / Echo State Networks (ESN) using
physical reservoirs: memristor crossbars, photonic meshes, or oscillatory networks.

References:
- Maass et al. (2002) - Liquid State Machines
- Jaeger (2001) - Echo State Networks
- Tanaka et al. (2019) - Photonic reservoir computing
- Du et al. (2017) - Memristor reservoir computing
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

import numpy as np

from chiplab.schemas.common import Provenance, TrackedValue
from chiplab.core.gpu import GPUBackend, get_gpu_backend as get_backend


class ReservoirType(Enum):
    """Type of physical reservoir."""
    MEMRISTOR_CROSSBAR = "memristor_crossbar"
    PHOTONIC_MESH = "photonic_mesh"
    OSCILLATORY_NETWORK = "oscillatory_network"
    SPINTRONIC_ARRAY = "spintronic_array"
    FERROELECTRIC_ARRAY = "ferroelectric_array"
    DIGITAL_RANDOM = "digital_random"  # Software baseline


class ReadoutType(Enum):
    """Readout layer type."""
    LINEAR = "linear"
    RIDGE = "ridge"
    LASSO = "lasso"
    SVR = "svr"
    NEURAL = "neural"


@dataclass
class ReservoirState:
    """
    State of a liquid state machine reservoir.
    
    Tracks reservoir states, input weights, and readout weights with provenance.
    """
    # Reservoir configuration
    reservoir_type: ReservoirType = ReservoirType.MEMRISTOR_CROSSBAR
    n_inputs: int = 10
    n_reservoir: int = 100
    n_outputs: int = 5
    
    # Reservoir state vector
    state: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=np.zeros(100),
        provenance=Provenance.ASSUMED,
        metadata={"description": "Reservoir neuron states"}
    ))
    
    # Input weight matrix (fixed, random)
    w_in: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=np.zeros((100, 10)),
        provenance=Provenance.ASSUMED,
        metadata={"description": "Input-to-reservoir weights", "trainable": False}
    ))
    
    # Recurrent weight matrix (fixed, sparse random)
    w_res: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=np.zeros((100, 100)),
        provenance=Provenance.ASSUMED,
        metadata={"description": "Reservoir recurrent weights", "trainable": False}
    ))
    
    # Readout weights (trained)
    w_out: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=np.zeros((5, 100)),
        provenance=Provenance.ASSUMED,
        metadata={"description": "Reservoir-to-output weights", "trainable": True}
    ))
    
    # Reservoir parameters
    spectral_radius: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.9,
        provenance=Provenance.MEASURED,
        metadata={"description": "Spectral radius of recurrent matrix"}
    ))
    sparsity: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.1,
        provenance=Provenance.MEASURED,
        metadata={"description": "Connection sparsity"}
    ))
    input_scaling: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Input weight scaling"}
    ))
    leaking_rate: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.3,
        provenance=Provenance.MEASURED,
        metadata={"description": "Leaking rate for leaky integrator neurons"}
    ))
    
    # Activation function
    activation: str = "tanh"  # tanh, relu, sigmoid, linear
    
    # Noise
    noise_level: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.001,
        provenance=Provenance.MEASURED,
        metadata={"description": "Additive noise standard deviation"}
    ))
    
    # Training parameters
    readout_type: ReadoutType = ReadoutType.RIDGE
    ridge_alpha: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e-6,
        provenance=Provenance.MEASURED,
        metadata={"description": "Ridge regression regularization"}
    ))
    
    # History
    state_history: list[np.ndarray] = field(default_factory=list)
    output_history: list[np.ndarray] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize arrays with correct dimensions based on configuration."""
        # Re-initialize arrays with correct shapes
        self.state = TrackedValue(
            value=np.zeros(self.n_reservoir),
            provenance=Provenance.ASSUMED,
            metadata={"description": "Reservoir neuron states"}
        )
        self.w_in = TrackedValue(
            value=np.zeros((self.n_reservoir, self.n_inputs)),
            provenance=Provenance.ASSUMED,
            metadata={"description": "Input-to-reservoir weights", "trainable": False}
        )
        self.w_res = TrackedValue(
            value=np.zeros((self.n_reservoir, self.n_reservoir)),
            provenance=Provenance.ASSUMED,
            metadata={"description": "Reservoir recurrent weights", "trainable": False}
        )
        self.w_out = TrackedValue(
            value=np.zeros((self.n_outputs, self.n_reservoir)),
            provenance=Provenance.ASSUMED,
            metadata={"description": "Reservoir-to-output weights", "trainable": True}
        )
    
    def copy(self) -> ReservoirState:
        """Create a deep copy."""
        new_state = ReservoirState(
            reservoir_type=self.reservoir_type,
            n_inputs=self.n_inputs,
            n_reservoir=self.n_reservoir,
            n_outputs=self.n_outputs
        )
        for field_name in self.__dataclass_fields__:
            if field_name in ['n_inputs', 'n_reservoir', 'n_outputs', 'reservoir_type', 
                             'activation', 'readout_type']:
                continue
            value = getattr(self, field_name)
            if isinstance(value, TrackedValue):
                setattr(new_state, field_name, value.copy())
            elif isinstance(value, (ReservoirType, ReadoutType, str)):
                setattr(new_state, field_name, value)
            elif isinstance(value, (int, float)):
                setattr(new_state, field_name, value)
            elif isinstance(value, list):
                setattr(new_state, field_name, [v.copy() if isinstance(v, np.ndarray) else v for v in value])
            else:
                setattr(new_state, field_name, value)
        return new_state


def create_reservoir_weights(
    n_reservoir: int,
    n_inputs: int,
    spectral_radius: float = 0.9,
    sparsity: float = 0.1,
    input_scaling: float = 1.0,
    seed: int = 42,
    backend: GPUBackend | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create random reservoir weight matrices.
    
    Args:
        n_reservoir: Number of reservoir neurons
        n_inputs: Number of input channels
        spectral_radius: Target spectral radius of recurrent matrix
        sparsity: Fraction of non-zero connections
        input_scaling: Scaling factor for input weights
        seed: Random seed
        backend: GPU backend
        
    Returns:
        (w_in, w_res) weight matrices
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    rng = np.random.default_rng(seed)
    
    # Input weights: uniform [-input_scaling, input_scaling]
    w_in = backend.to_device(
        rng.uniform(-input_scaling, input_scaling, (n_reservoir, n_inputs)).astype(np.float32)
    )
    
    # Recurrent weights: sparse random matrix
    w_res = xp.zeros((n_reservoir, n_reservoir), dtype=xp.float32)
    
    # Generate sparse connections
    n_connections = int(n_reservoir * n_reservoir * sparsity)
    rows = rng.choice(n_reservoir, n_connections)
    cols = rng.choice(n_reservoir, n_connections)
    values = rng.normal(0, 1, n_connections).astype(np.float32)
    
    for r, c, v in zip(rows, cols, values):
        w_res[r, c] = v
    
    # Scale to target spectral radius
    # Compute largest eigenvalue magnitude
    try:
        if hasattr(xp, 'linalg') and hasattr(xp.linalg, 'eigvals'):
            eigvals = xp.linalg.eigvals(w_res)
            current_radius = float(xp.max(xp.abs(eigvals)))
        else:
            # Fallback: use power iteration
            current_radius = _power_iteration(w_res, backend)
    except Exception:
        current_radius = 1.0
    
    if current_radius > 0:
        w_res = w_res * (spectral_radius / current_radius)
    
    return backend.to_numpy(w_in), backend.to_numpy(w_res)


def _power_iteration(matrix: Any, backend: GPUBackend, max_iter: int = 100) -> float:
    """Power iteration for spectral radius estimation."""
    xp = backend.xp
    n = matrix.shape[0]
    v = xp.ones(n, dtype=xp.float32)
    v = v / xp.linalg.norm(v)
    
    for _ in range(max_iter):
        v_new = xp.matmul(matrix, v)
        norm = xp.linalg.norm(v_new)
        if norm > 0:
            v_new = v_new / norm
        v = v_new
    
    # Rayleigh quotient
    return float(xp.dot(xp.matmul(matrix, v), v))


def reservoir_step(
    state: ReservoirState,
    input_signal: np.ndarray,
    dt: float = 1.0,
    backend: GPUBackend | None = None
) -> ReservoirState:
    """
    Single reservoir update step (leaky integrator).
    
    x(t+1) = (1 - alpha) * x(t) + alpha * f(W_in * u(t) + W_res * x(t) + noise)
    
    Args:
        state: Current reservoir state
        input_signal: Input vector [n_inputs]
        dt: Time step (ms)
        backend: GPU backend
        
    Returns:
        Updated reservoir state
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    new_state = state.copy()
    
    # Move to device
    x = backend.to_device(state.state.value)
    w_in = backend.to_device(state.w_in.value)
    w_res = backend.to_device(state.w_res.value)
    u = backend.to_device(input_signal.astype(np.float32))
    
    # Compute input drive
    input_drive = xp.matmul(w_in, u)
    
    # Compute recurrent drive
    recurrent_drive = xp.matmul(w_res, x)
    
    # Total drive
    total_drive = input_drive + recurrent_drive
    
    # Add noise
    noise_std = float(state.noise_level.value)
    if noise_std > 0:
        noise = backend.random_normal(total_drive.shape, std=noise_std)
        total_drive = total_drive + noise
    
    # Activation function
    activation = state.activation
    if activation == "tanh":
        activated = xp.tanh(total_drive)
    elif activation == "relu":
        activated = xp.maximum(total_drive, 0)
    elif activation == "sigmoid":
        activated = 1.0 / (1.0 + xp.exp(-total_drive))
    elif activation == "linear":
        activated = total_drive
    else:
        activated = xp.tanh(total_drive)
    
    # Leaky integration
    alpha = float(state.leaking_rate.value)
    x_new = (1.0 - alpha) * x + alpha * activated
    
    # Update state
    new_state.state.value = backend.to_numpy(x_new)
    new_state.state.provenance = Provenance.ESTIMATED
    new_state.state_history.append(new_state.state.value.copy())
    
    return new_state


def reservoir_run(
    state: ReservoirState,
    inputs: np.ndarray,  # [n_timesteps, n_inputs]
    dt: float = 1.0,
    backend: GPUBackend | None = None
) -> tuple[ReservoirState, np.ndarray]:
    """
    Run reservoir on input sequence.
    
    Args:
        state: Initial reservoir state
        inputs: Input sequence [n_timesteps, n_inputs]
        dt: Time step (ms)
        backend: GPU backend
        
    Returns:
        Final state, state trajectory [n_timesteps, n_reservoir]
    """
    if backend is None:
        backend = get_backend()
    
    n_steps = inputs.shape[0]
    trajectory = []
    current_state = state
    
    for t in range(n_steps):
        current_state = reservoir_step(current_state, inputs[t], dt, backend)
        trajectory.append(current_state.state.value.copy())
    
    return current_state, np.array(trajectory)


def reservoir_train_readout(
    state: ReservoirState,
    states: np.ndarray,  # [n_timesteps, n_reservoir]
    targets: np.ndarray,  # [n_timesteps, n_outputs]
    washout: int = 0,
    backend: GPUBackend | None = None
) -> ReservoirState:
    """
    Train readout weights using ridge regression.
    
    Args:
        state: Reservoir state (will be updated with trained w_out)
        states: Reservoir state trajectory
        targets: Target outputs
        washout: Number of initial timesteps to discard
        backend: GPU backend
        
    Returns:
        Updated reservoir state with trained readout
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    new_state = state.copy()
    
    # Discard washout period
    if washout > 0:
        states = states[washout:]
        targets = targets[washout:]
    
    # Move to device
    X = backend.to_device(states.astype(np.float32))  # [T, N]
    Y = backend.to_device(targets.astype(np.float32))  # [T, M]
    
    # Ridge regression: W_out = Y^T X (X^T X + alpha I)^-1
    # Solve: (X^T X + alpha I) W_out^T = X^T Y
    XT = xp.transpose(X)
    XTX = xp.matmul(XT, X)
    XTY = xp.matmul(XT, Y)
    
    alpha = float(state.ridge_alpha.value)
    reg = alpha * xp.eye(XTX.shape[0], dtype=xp.float32)
    
    # Solve linear system
    try:
        W_out_T = xp.linalg.solve(XTX + reg, XTY)
        W_out = xp.transpose(W_out_T)
    except Exception:
        # Fallback: pseudo-inverse
        W_out = xp.matmul(Y, xp.linalg.pinv(X))
    
    new_state.w_out.value = backend.to_numpy(W_out)
    new_state.w_out.provenance = Provenance.ESTIMATED
    
    return new_state


def reservoir_predict(
    state: ReservoirState,
    states: np.ndarray,  # [n_timesteps, n_reservoir]
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    Generate predictions using trained readout.
    
    Args:
        state: Reservoir state with trained readout
        states: Reservoir state trajectory
        backend: GPU backend
        
    Returns:
        Predictions [n_timesteps, n_outputs]
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    X = backend.to_device(states.astype(np.float32))
    W_out = backend.to_device(state.w_out.value.astype(np.float32))
    
    predictions = xp.matmul(X, xp.transpose(W_out))
    return backend.to_numpy(predictions)


def reservoir_compute_memory_capacity(
    state: ReservoirState,
    n_delays: int = 20,
    n_trials: int = 100,
    dt: float = 1.0,
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    Compute memory capacity of reservoir (Jaeger 2002).
    
    Measures how well reservoir can reconstruct delayed inputs.
    
    Args:
        state: Reservoir state
        n_delays: Maximum delay to test
        n_trials: Number of random input trials
        dt: Time step
        backend: GPU backend
        
    Returns:
        Memory capacity per delay [n_delays]
    """
    if backend is None:
        backend = get_backend()
    
    n_res = state.n_reservoir
    n_in = state.n_inputs
    
    capacities = []
    
    for delay in range(1, n_delays + 1):
        # Generate random input sequence
        total_len = n_trials + delay + 10
        inputs = np.random.randn(total_len, n_in).astype(np.float32)
        
        # Run reservoir
        run_state = state.copy()
        run_state, trajectory = reservoir_run(run_state, inputs, dt, backend)
        
        # Target is input at t - delay
        targets = inputs[delay:delay + n_trials]
        states = trajectory[delay:delay + n_trials]
        
        # Train readout for this delay
        readout_state = reservoir_train_readout(run_state, states, targets, backend=backend)
        
        # Test on new data
        test_inputs = np.random.randn(n_trials + delay, n_in).astype(np.float32)
        _, test_traj = reservoir_run(readout_state, test_inputs, dt, backend)
        test_states = test_traj[delay:]
        test_targets = test_inputs[delay:delay + n_trials]
        predictions = reservoir_predict(readout_state, test_states, backend)
        
        # Compute correlation coefficient
        corr = np.corrcoef(predictions.flatten(), test_targets.flatten())[0, 1]
        capacities.append(corr ** 2 if not np.isnan(corr) else 0.0)
    
    return np.array(capacities)


def reservoir_memristor_crossbar_init(
    n_reservoir: int,
    n_inputs: int,
    g_min: float = 1e-6,
    g_max: float = 1e-3,
    variability: float = 0.1,
    device: str = "auto",
    backend: GPUBackend | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize reservoir weights from memristor crossbar physics.
    
    Uses actual conductance values with device variability.
    """
    if backend is None:
        backend = get_backend(device)
    
    xp = backend.xp
    
    # Input weights: memristor conductances at crosspoints
    w_in_cond = backend.random_uniform((n_reservoir, n_inputs), g_min, g_max)
    # Add variability
    noise = backend.random_normal(w_in_cond.shape, std=variability * (g_max - g_min))
    w_in_cond = xp.maximum(w_in_cond + noise, g_min)
    
    # Normalize to [-1, 1] range for reservoir dynamics
    w_in = 2.0 * (w_in_cond - g_min) / (g_max - g_min) - 1.0
    
    # Recurrent weights: sparse memristor crossbar
    sparsity = 0.1
    n_conn = int(n_reservoir * n_reservoir * sparsity)
    
    w_res = xp.zeros((n_reservoir, n_reservoir), dtype=xp.float32)
    rows = backend.random_integers(0, n_reservoir, n_conn)
    cols = backend.random_integers(0, n_reservoir, n_conn)
    
    for r, c in zip(rows, cols):
        g = backend.random_uniform((), g_min, g_max)
        w_res[r, c] = 2.0 * (g - g_min) / (g_max - g_min) - 1.0
    
    return backend.to_numpy(w_in), backend.to_numpy(w_res)


def reservoir_photonic_mesh_init(
    n_reservoir: int,
    n_inputs: int,
    n_modes: int = 4,
    loss_db_cm: float = 0.1,
    coupling_var: float = 0.05,
    device: str = "auto",
    backend: GPUBackend | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize reservoir from photonic mesh (MZI mesh) physics.
    
    Uses unitary scattering matrix of MZI mesh as recurrent weights.
    """
    if backend is None:
        backend = get_backend(device)
    
    xp = backend.xp
    
    # Photonic mesh implements unitary transformation
    # Generate random unitary matrix via QR decomposition
    # This simulates a mesh of MZIs with random phases
    
    # Input coupling: random phases
    phases_in = backend.random_uniform((n_reservoir, n_inputs), 0, 2 * np.pi)
    w_in = xp.cos(phases_in)  # Simplified: cos(phase) as coupling
    
    # Recurrent: random unitary matrix
    # Generate via QR of random complex matrix
    A_real = backend.random_normal((n_reservoir, n_reservoir))
    A_imag = backend.random_normal((n_reservoir, n_reservoir))
    A = A_real + 1j * A_imag
    
    # QR decomposition for unitary
    try:
        Q, R = xp.linalg.qr(A)
        # Make Q uniformly distributed (Haar measure)
        D = xp.diag(R)
        ph = D / xp.abs(D)
        Q = Q * ph
        w_res = xp.real(Q).astype(xp.float32)
    except Exception:
        # Fallback: random orthogonal
        w_res = backend.random_normal((n_reservoir, n_reservoir))
        w_res, _ = xp.linalg.qr(w_res)
        w_res = w_res.astype(xp.float32)
    
    # Apply loss
    loss_factor = 10 ** (-loss_db_cm / 20)
    w_res = w_res * loss_factor
    
    # Add coupling variability
    noise = backend.random_normal(w_res.shape, std=coupling_var)
    w_res = w_res + noise
    
    return backend.to_numpy(w_in), backend.to_numpy(w_res)


def lsm_classify(
    state: ReservoirState,
    test_inputs: np.ndarray,
    dt: float = 1.0,
    backend: GPUBackend | None = None
) -> np.ndarray:
    """
    Classify inputs using trained LSM.
    
    Args:
        state: Trained reservoir state
        test_inputs: Test input sequences [n_samples, n_timesteps, n_inputs]
        dt: Time step
        backend: GPU backend
        
    Returns:
        Predictions [n_samples, n_outputs]
    """
    if backend is None:
        backend = get_backend()
    
    n_samples = test_inputs.shape[0]
    predictions = []
    
    for i in range(n_samples):
        run_state = state.copy()
        run_state, trajectory = reservoir_run(run_state, test_inputs[i], dt, backend)
        pred = reservoir_predict(state, trajectory, backend)
        # Use last timestep or average
        predictions.append(pred[-1])
    
    return np.array(predictions)


# Export
__all__ = [
    "ReservoirType",
    "ReadoutType",
    "ReservoirState",
    "create_reservoir_weights",
    "reservoir_step",
    "reservoir_run",
    "reservoir_train_readout",
    "reservoir_predict",
    "reservoir_compute_memory_capacity",
    "reservoir_memristor_crossbar_init",
    "reservoir_photonic_mesh_init",
    "lsm_classify",
]