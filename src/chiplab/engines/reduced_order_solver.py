"""Level 2: Spatial multiphysics model with 2D/3D grid and coupled PDEs.

GENERIC PLACEHOLDER, not a ferrofluid model. It refuses ferrofluid inputs; use
``models.materials.ferro_device``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from chiplab.credibility import refuse_ferrofluid
from chiplab.schemas.common import Provenance, TrackedValue


class SpatialMultiphysicsSolver:
    """Level 2 spatial multiphysics solver.

    This solver resolves 2D/3D device geometry with coupled PDEs:
    - Magnetization diffusion
    - Charge transport
    - Heat equation

    It provides spatially resolved state updates for device-level
    multiphysics analysis.
    """

    def __init__(
        self,
        grid_size: tuple[int, int] | tuple[int, int, int] = (64, 64),
        precision: str = "double",
        material: str | None = None,
    ):
        """Initialize the SpatialMultiphysicsSolver.

        Parameters
        ----------
        grid_size : tuple of int
            Number of grid points in each dimension.
            (nx, ny) for 2D or (nx, ny, nz) for 3D.
        precision : str
            Floating point precision ("float" or "double").
        """
        refuse_ferrofluid("SpatialMultiphysicsSolver", material)
        self.grid_size = grid_size
        self.precision = precision
        self.dtype = np.float64 if precision == "double" else np.float32

        # Physical constants (with provenance)
        self._gamma = TrackedValue(value=1.76e11, provenance=Provenance.MEASURED, unit="rad/s/T")
        self._alpha = TrackedValue(value=0.01, provenance=Provenance.ESTIMATED, unit="")
        self._mu0 = TrackedValue(value=4e-7 * np.pi, provenance=Provenance.MEASURED, unit="H/m")

    # ------------------------------------------------------------------
    # Core solve: spatially resolved state update
    # ------------------------------------------------------------------
    def solve(
        self,
        state: np.ndarray,
        inputs: dict[str, TrackedValue | float | int],
        dt: float,
        grid: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        """Advance the spatially resolved state by dt using coupled PDEs.

        Parameters
        ----------
        state : np.ndarray
            Current spatially resolved state field.
        inputs : dict
            Additional inputs (fields, sources, etc.) as TrackedValue or raw numbers.
        dt : float
            Time step size (seconds).
        grid : dict, optional
            Grid configuration (spacing, boundaries, etc.).

        Returns
        -------
        np.ndarray
            New spatially resolved state after time step dt.
        """
        refuse_ferrofluid("SpatialMultiphysicsSolver", inputs, grid)
        # Ensure state has the right shape for the grid
        if state.ndim == 1:
            # Reshape to grid
            nx, ny = self.grid_size[:2]
            state = state.reshape(nx, ny)

        # Extract effective fields from inputs
        h_eff = self._extract_effective_field(inputs)

        # Magnetization diffusion (heat equation coupled)
        state = self._magnetization_diffusion(state, h_eff, dt)

        # Charge transport (simplified)
        state = self._charge_transport(state, inputs, dt)

        # Heat equation
        state = self._heat_equation(state, dt)

        # Renormalize if magnetization
        if state.ndim == 2 and state.shape[0] > 1:
            state = self._renormalize_magnetization(state)

        # Flatten back to 1D if original was 1D
        if state.ndim == 2 and self.grid_size[0] * self.grid_size[1] == state.size:
            state = state.flatten()

        return state

    # ------------------------------------------------------------------
    # Electrode boundary conditions
    # ------------------------------------------------------------------
    def electrode_boundary_conditions(self, electrode_array: Any) -> dict[str, Any]:
        """Generate port-specific boundary conditions for an electrode array.

        Parameters
        ----------
        electrode_array : Any
            Electrode array configuration specifying positions and properties.

        Returns
        -------
        dict
            Boundary condition specifications per electrode.
        """
        bc = {
            "type": "dirichlet",
            "electrodes": [],
        }
        if hasattr(electrode_array, "__iter__"):
            for i, electrode in enumerate(electrode_array):
                bc["electrodes"].append(
                    {
                        "index": i,
                        "potential": getattr(electrode, "potential", 0.0),
                        "current": getattr(electrode, "current", 0.0),
                    }
                )
        return bc

    # ------------------------------------------------------------------
    # Internal PDE solvers (simplified)
    # ------------------------------------------------------------------
    def _magnetization_diffusion(
        self, state: np.ndarray, h_eff: np.ndarray, dt: float
    ) -> np.ndarray:
        """Solve magnetization diffusion PDE."""
        # Simplified: explicit Euler update with damping
        m = state.copy()
        if m.ndim == 2 and m.shape[0] >= 3:
            # Assume first 3 channels are [mx, my, mz] or similar
            m_x = m[0, :]
            m_y = m[1, :]
            m_z = m[2, :]

            # Diffusion term (Laplacian)
            laplacian_mx = self._laplacian(m_x)
            laplacian_my = self._laplacian(m_y)
            laplacian_mz = self._laplacian(m_z)

            # LLG-inspired update with diffusion
            dm_dt = (
                -self._gamma.value * self._cross_3d(m, h_eff)
                + float(self._alpha.value) * self._cross_3d(m, self._cross_3d(m, h_eff))
            )

            # Add diffusion
            dm_dt += 0.01 * np.stack([laplacian_mx, laplacian_my, laplacian_mz], axis=0)

            m_new = m + dt * dm_dt
            # Renormalize
            m_norm = np.linalg.norm(m_new, axis=0, keepdims=True)
            m_new = m_new / np.maximum(m_norm, 1e-10)

            return m_new
        return state

    def _charge_transport(self, state: np.ndarray, inputs: dict, dt: float) -> np.ndarray:
        """Solve charge transport PDE (simplified)."""
        # Placeholder: minimal charge transport effect
        return state

    def _heat_equation(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Solve heat equation PDE (simplified)."""
        # Placeholder: minimal heat effect
        return state

    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute the Laplacian of a 2D field using finite differences."""
        # Pad for boundary conditions
        padded = np.pad(field, 1, mode="constant", constant_values=0)
        laplacian = (
            padded[2:, 1:-1]
            + padded[:-2, 1:-1]
            + padded[1:-1, 2:]
            + padded[1:-1, :-2]
            - 4 * field
        )
        return laplacian

    def _cross_3d(self, m: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Compute cross product for 3D magnetization/field arrays."""
        mx = m[0] if m.ndim == 2 else m[..., 0]
        my = m[1] if m.ndim == 2 else m[..., 1]
        mz = m[2] if m.ndim == 2 else m[..., 2]
        hx = h[0] if h.ndim == 1 else h[..., 0]
        hy = h[1] if h.ndim == 1 else h[..., 1]
        hz = h[2] if h.ndim == 1 else h[..., 2]
        return np.stack([my * hz - mz * hy, mz * hx - mx * hz, mx * hy - my * hx], axis=-1)

    def _renormalize_magnetization(self, m: np.ndarray) -> np.ndarray:
        """Renormalize magnetization vectors to unit length."""
        norms = np.linalg.norm(m, axis=0, keepdims=True)
        return m / np.maximum(norms, 1e-10)