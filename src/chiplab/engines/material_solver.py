"""Level 1: Reduced-order mathematical model for magnetic device simulation.

Provides a fast ReducedOrderSolver that uses analytical ODEs to model
magnetization dynamics, pulse writing, impedance reading, and thermal
relaxation. Supports architecture experiments with fast numerical performance.

GENERIC PLACEHOLDER, not a ferrofluid model (``read_impedance`` is a constant 50 ohm).
It refuses ferrofluid inputs; use ``models.materials.ferro_device``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import torch
from chiplab.schemas.common import Provenance, TrackedValue
from chiplab.core.gpu import get_gpu_backend
from chiplab.credibility import refuse_ferrofluid


class ReducedOrderSolver:
    """Level 1 reduced-order solver using analytical ODEs.

    This solver provides a fast, reduced-order model based on analytical
    ordinary differential equations for modeling magnetization dynamics,
    pulse writing, impedance reading, and thermal relaxation. It is designed
    to support thousands of architecture experiments quickly.

    Parameters
    ----------
    gamma : float
        Gyromagnetic ratio (rad/s/T), provenance: measured
    alpha : float
        Damping factor, provenance: estimated
    Ms : float
        Saturation magnetization (A/m), provenance: measured
    """

    def __init__(
        self,
        gamma: float | TrackedValue = 1.76e11,
        alpha: float | TrackedValue = 0.01,
        Ms: float | TrackedValue = 8e5,
        device: str = "auto",
        material: str | None = None,
    ):
        """Initialize the ReducedOrderSolver.

        Parameters
        ----------
        gamma : float or TrackedValue
            Gyromagnetic ratio (rad/s/T). If TrackedValue, provenance must be set.
        alpha : float or TrackedValue
            Damping factor. If TrackedValue, provenance must be set.
        Ms : float or TrackedValue
            Saturation magnetization (A/m). If TrackedValue, provenance must be set.
        device : str
            Device to run on: "auto", "cuda", "gpu", or "cpu". Default "auto".
        """
        self.gamma = self._ensure_tracked_value(gamma, Provenance.MEASURED, "rad/s/T")
        self.alpha = self._ensure_tracked_value(alpha, Provenance.ESTIMATED, "")
        self.Ms = self._ensure_tracked_value(Ms, Provenance.MEASURED, "A/m")
        refuse_ferrofluid("ReducedOrderSolver", material)
        self.device = device
        self.backend = get_gpu_backend(device=device)

    # ------------------------------------------------------------------
    # Helper to convert raw float / int / TrackedValue into a TrackedValue
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_tracked_value(
        val: float | int | TrackedValue,
        default_provenance: Provenance,
        default_unit: str,
    ) -> TrackedValue:
        if isinstance(val, TrackedValue):
            # Ensure provenance is set; fill missing fields with defaults
            if val.provenance is Provenance.ASSUMED:
                val.provenance = default_provenance
            if val.unit is None:
                val.unit = default_unit
            return val
        # Raw value -- promote to TrackedValue with assumed provenance
        return TrackedValue(value=float(val), provenance=default_provenance, unit=default_unit)

    # ------------------------------------------------------------------
    # Core ODE: dM/dt = -gamma * (M x H_eff) + alpha * (M x dM/dt) / Ms
    # ------------------------------------------------------------------
    def _effective_field(self, m: np.ndarray, **kwargs) -> np.ndarray:
        """Effective field for unit magnetization vector.

        In the reduced-order model we assume a simple uniaxial anisotropy
        along the z-axis with no external field for the baseline model.
        """
        # anisotropy field along z: H_k = (2*K_u / (mu0*Ms)) * z_hat
        # For the simplest reduced model we just return zero field
        return np.zeros(3, dtype=float)

    def solve(
        self,
        state: np.ndarray,
        inputs: dict[str, TrackedValue | float | int],
        dt: float,
    ) -> np.ndarray:
        """ Advance the state by dt using analytical ODE integration.

        The state vector is [m_x, m_y, m_z] representing the unit
        magnetization direction.  The ODE used is the Landau-Lifshitz- Gilbert
        equation in the precessional-damping form, linearized for small dt.

        Parameters
        ----------
        state : np.ndarray
            Current state [m_x, m_y, m_z] (unit vector).
        inputs : dict
            Additional inputs (e.g. effective field, temperature).
            Values may be TrackedValue or raw numbers.
        dt : float
            Time step size (seconds).

        Returns
        -------
        np.ndarray
            New state [m_x, m_y, m_z] after time step dt.
        """
        refuse_ferrofluid("ReducedOrderSolver", inputs)
        m = state.copy().astype(float)

        # Extract effective field from inputs, default to zero
        h_eff = np.zeros(3)
        if "effective_field" in inputs:
            h = inputs["effective_field"]
            if isinstance(h, TrackedValue):
                h = float(h)
            h_eff = np.array([h, 0.0, 0.0], dtype=float)

        # Use GPU backend if available for field operations
        if self.backend.is_gpu:
            h_eff = self.backend.to_device(h_eff, dtype="float64")
            # Use PyTorch operations on GPU
            m_t = self.backend.to_device(m, dtype="float64")
            h_eff_t = h_eff
        else:
            m_t = m
            h_eff_t = h_eff

        # LLG integration (explicit Euler, precessional + damping)
        # dm/dt = -gamma * (m x h_eff) + alpha * m x (dm/dt)
        # For small dt we use the combined form:
        if self.backend.is_gpu:
            m_cross_h = torch.cross(m_t, h_eff_t)
            dm_dt_t = -float(self.gamma) * m_cross_h + float(self.alpha) * torch.cross(m_t, m_cross_h)
            dm_dt = self.backend.to_numpy(dm_dt_t)
        else:
            m_cross_h = np.cross(m, h_eff)
            dm_dt = -float(self.gamma) * m_cross_h + float(self.alpha) * np.cross(m, m_cross_h)

        m_new = m + dt * dm_dt
        # Renormalize to keep it a unit vector
        m_new_norm = np.linalg.norm(m_new)
        if m_new_norm > 0:
            m_new = m_new / m_new_norm

        return m_new.astype(float)

    # ------------------------------------------------------------------
    # write_pulse: apply a voltage pulse (frequency, duration)
    # ------------------------------------------------------------------
    def write_pulse(
        self,
        voltage: float | TrackedValue,
        duration: float | TrackedValue,
        frequency: float | TrackedValue,
    ) -> TrackedValue:
        """Apply a voltage pulse and return the state update.

        In the reduced-order model the pulse induces an effective field
        proportional to the voltage amplitude and the frequency sets the
        resonance condition.

        Parameters
        ----------
        voltage : float or TrackedValue
            Pulse voltage (V). Provenance will be recorded.
        duration : float or TrackedValue
            Pulse duration (s). Provenance will be recorded.
        frequency : float or TrackedValue
            Pulse frequency (Hz). Provenance will be recorded.

        Returns
        -------
        TrackedValue
            The resulting magnetization state change magnitude (|Delta m|),
            with provenance set to ``Provenance.ESTIMATED``.
        """
        v = float(self._ensure_tracked_value(voltage, Provenance.ESTIMATED, "V").value)
        d = float(self._ensure_tracked_value(duration, Provenance.ESTIMATED, "s").value)
        f = float(self._ensure_tracked_value(frequency, Provenance.ESTIMATED, "Hz").value)

        # Simple model: effective field proportional to voltage, scaled by frequency
        h_eff = v / (f + 1e-6)  # avoid div-by-zero

        # Integrate for the pulse duration
        state = np.array([1.0, 0.0, 0.0], dtype=float)  # start along x
        steps = max(1, int(d / 1e-12))  # simple stepping
        dt_internal = d / steps
        m = state.copy()
        for _ in range(steps):
            m = self.solve(m, {"effective_field": h_eff}, dt_internal)
        # Return magnitude of change from initial [1,0,0]
        delta_m = float(np.linalg.norm(m - state))
        return TrackedValue(value=delta_m, provenance=Provenance.ESTIMATED, unit="")

    # ------------------------------------------------------------------
    # read_impedance: compute impedance at a given frequency
    # ------------------------------------------------------------------
    def read_impedance(self, frequency: float | TrackedValue) -> TrackedValue:
        """Compute the complex impedance at a given frequency.

        In the reduced-order RLC-resonator model the impedance near
        resonance is Z(f) = R + j * (omega*L - 1/(omega*C)),
        where the resonance frequency is determined by Ms and geometry.

        Parameters
        ----------
        frequency : float or TrackedValue
            Frequency to evaluate impedance at (Hz).

        Returns
        -------
        TrackedValue
            Complex impedance real part (ohms) with provenance.
        """
        f = float(self._ensure_tracked_value(frequency, Provenance.ESTIMATED, "Hz").value)
        # Simple RLC model: R = 50 ohms, f_res determined by Ms
        # Resonance approximately at f_res = gamma * Ms / (2*pi)
        mu0 = 4e-7 * np.pi
        if hasattr(self.Ms, "value"):
            f_res = float(self.gamma.value) * float(self.Ms.value) / (2 * np.pi)
        else:
            f_res = float(self.Ms) * self.gamma / (2 * np.pi)
        # Simplified: assume we are off-resonance and return mostly resistive
        r = 50.0
        omega = 2 * np.pi * f
        # Inductive and capacitive reactance (simplified)
        x_l = omega * 1e-9  # simplified inductance term
        x_c = 1 / (omega * 1e-9)  # simplified capacitance term
        z_real = r  # primarily resistive near resonance
        return TrackedValue(value=z_real, provenance=Provenance.ESTIMATED, unit="ohms")

    # ------------------------------------------------------------------
    # relaxation: time evolution towards equilibrium
    # ------------------------------------------------------------------
    def relaxation(self, dt: float | TrackedValue) -> np.ndarray:
        """Time evolution of magnetization towards equilibrium.

        Models the exponential relaxation of the magnetization towards
        the equilibrium direction (here, +z) with time constant tau.

        Parameters
        ----------
        dt : float or TrackedValue
            Time step (seconds).

        Returns
        -------
        np.ndarray
            Updated magnetization vector [m_x, m_y, m_z] after relaxation dt.
        """
        t = float(self._ensure_tracked_value(dt, Provenance.ESTIMATED, "s").value)
        # Relaxation time constant (simplified)
        tau = 1e-9  # 1 ns simplified
        # Initial state: assume starting in xy plane
        m = np.array([1.0, 0.0, 0.0], dtype=float)
        # Exponential decay towards +z equilibrium
        frac = 1.0 - np.exp(-t / tau)
        m_new = np.array([(1 - frac) * m[0], (1 - frac) * m[1], frac])
        return m_new