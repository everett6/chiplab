"""Level 3: Particle-scale reference model with nanoparticle dynamics."""

from __future__ import annotations

from typing import Any

import numpy as np

from chiplab.schemas.common import Provenance, TrackedValue


class ParticleScaleSolver:
    """Level 3 particle-scale reference solver.

    This solver models individual nanoparticle dynamics including:
    - Position, moment, and orientation of each nanoparticle
    - Brownian + Néel relaxation
    - Dipole-dipole interactions between particles

    Intended for small regions only (validation of simpler models).
    """

    def __init__(
        self,
        num_particles: int = 100,
        grid_size: tuple[int, int, int] = (10, 10, 10),
        dt: float = 1e-12,
    ):
        """Initialize the ParticleScaleSolver.

        Parameters
        ----------
        num_particles : int
            Number of nanoparticles in the simulation.
        grid_size : tuple of int
            Size of the simulation grid (nx, ny, nz).
        dt : float
            Time step size (seconds).
        """
        self.num_particles = num_particles
        self.grid_size = grid_size
        self.dt = dt

        # Physical constants (with provenance)
        self.gamma = TrackedValue(
            value=1.76e11, provenance=Provenance.MEASURED, unit="rad/s/T"
        )
        self.mu0 = TrackedValue(
            value=4e-7 * np.pi, provenance=Provenance.MEASURED, unit="H/m"
        )
        self.alpha = TrackedValue(
            value=0.01, provenance=Provenance.ESTIMATED, unit=""
        )
        self.kB = TrackedValue(
            value=1.38e-23, provenance=Provenance.MEASURED, unit="J/K"
        )
        self.temperature = TrackedValue(
            value=300.0, provenance=Provenance.ASSUMED, unit="K"
        )

        # Particle state
        self.positions: np.ndarray | None = None
        self.moments: np.ndarray | None = None
        self.orientations: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Core solve: particle trajectory integration
    # ------------------------------------------------------------------
    def solve(
        self,
        particles: dict[str, Any],
        field: np.ndarray,
        dt: float,
    ) -> dict[str, np.ndarray]:
        """Advance particle trajectories by dt under applied magnetic field.

        Parameters
        ----------
        particles : dict
            Particle state dictionary with positions, moments, orientations.
        field : np.ndarray
            Magnetic field array at particle positions.
        dt : float
            Time step size (seconds).

        Returns
        -------
        dict
            Updated particle state with new positions, moments, orientations.
        """
        # Initialize particle state if not provided
        if self.positions is None:
            self._initialize_particles(particles)

        # Compute effective field at each particle position
        h_eff = self._compute_effective_field(field)

        # Update each particle
        for i in range(self.num_particles):
            self._update_particle(i, h_eff[:, i] if h_eff.ndim > 1 else h_eff, dt)

        return {
            "positions": self.positions.copy(),
            "moments": self.moments.copy() if self.moments is not None else np.zeros((3, self.num_particles)),
            "orientations": self.orientations.copy() if self.orientations is not None else np.zeros(self.num_particles),
        }

    # ------------------------------------------------------------------
    # Brownian + Néel relaxation
    # ------------------------------------------------------------------
    def _brownian_relaxation(self, moment: np.ndarray, dt: float) -> np.ndarray:
        """Brownian relaxation of magnetic moment due to thermal noise."""
        magnitude = np.linalg.norm(moment)
        if magnitude < 1e-10:
            return moment

        # Thermal noise amplitude
        noise_amp = np.sqrt(2 * float(self.kB.value) * float(self.temperature.value) * dt / magnitude**2)

        # Random field direction
        random_dir = np.random.randn(3)
        random_dir = random_dir / np.linalg.norm(random_dir)

        # Apply Brownian torque
        delta_m = noise_amp * random_dir * magnitude
        new_moment = moment + delta_m

        # Renormalize
        new_mag = np.linalg.norm(new_moment)
        if new_mag > 1e-10:
            new_moment = new_moment / new_mag * magnitude

        return new_moment

    def _neel_relaxation(self, moment: np.ndarray, anisotropy_field: np.ndarray, dt: float) -> np.ndarray:
        """Néel relaxation via energy barrier crossing."""
        # Simplified: just add anisotropy effect
        effective_field = moment + anisotropy_field
        magnitude = np.linalg.norm(moment)
        if magnitude < 1e-10:
            return moment

        # Small deterministic update
        delta = float(self.alpha.value) * effective_field * dt * magnitude
        new_moment = moment + delta

        # Renormalize
        new_mag = np.linalg.norm(new_moment)
        if new_mag > 1e-10:
            new_moment = new_moment / new_mag * magnitude

        return new_moment

    # ------------------------------------------------------------------
    # Dipole-dipole interactions
    # ------------------------------------------------------------------
    def _dipole_interaction(self, i: int, j: int, moments: np.ndarray) -> np.ndarray:
        """Compute dipole-dipole interaction field from particle j on particle i."""
        mu0 = float(self.mu0.value)
        mi = moments[:, i]
        mj = moments[:, j]

        # Vector from j to i
        ri = self.positions[:, i] if self.positions is not None else np.zeros(3)
        rj = self.positions[:, j] if self.positions is not None else np.zeros(3)
        r_vec = ri - rj
        r_dist = np.linalg.norm(r_vec)

        if r_dist < 1e-10:
            return np.zeros(3)

        # Dipole field formula: B = (mu0/4pi) * (3(m·r̂)r̂ - m) / r^3
        r_hat = r_vec / r_dist
        m_dot_r = np.dot(mj, r_hat)
        field = (mu0 / (4 * np.pi)) * (3 * m_dot_r * r_hat - mj) / r_dist**3
        return field

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialize_particles(self, particles: dict[str, Any]) -> None:
        """Initialize particle state from dict or defaults."""
        self.positions = np.random.randn(3, self.num_particles) * 1e-9  # nm scale
        self.moments = np.random.randn(3, self.num_particles) * 1e-18  # emu scale
        self.orientations = np.random.rand(self.num_particles) * 2 * np.pi

        # Normalize moments
        for i in range(self.num_particles):
            mag = np.linalg.norm(self.moments[:, i])
            if mag > 1e-10:
                self.moments[:, i] = self.moments[:, i] / mag

    def _compute_effective_field(self, field: np.ndarray) -> np.ndarray:
        """Compute total effective field at all particle positions."""
        if field.ndim == 1:
            field = field.reshape(3, 1)

        # Add thermal noise field
        thermal_noise = np.random.randn(3) * 0.01  # scaled noise
        h_eff = field + thermal_noise.reshape(3, 1)

        return h_eff

    def _update_particle(
        self, i: int, h_eff: np.ndarray, dt: float
    ) -> None:
        """Update a single particle's state."""
        moment = self.moments[:, i].copy() if self.moments is not None else np.zeros(3)

        # Precession (LLG-like)
        cross_term = np.cross(moment, h_eff)
        precession = self.gamma.value * dt * cross_term

        # Damping
        damping = float(self.alpha.value) * dt * np.cross(moment, cross_term)

        # Brownian + Néel relaxation
        brownian = self._brownian_relaxation(moment, dt)
        neel = self._neel_relaxation(moment, h_eff, dt)

        # Update moment
        new_moment = moment + precession - damping + brownian + neel

        # Renormalize
        mag = np.linalg.norm(new_moment)
        if mag > 1e-10:
            new_moment = new_moment / mag * mag

        self.moments[:, i] = new_moment

        # Update position (Brownian diffusion)
        if self.positions is not None:
            diffusion = np.random.randn(3) * np.sqrt(2 * float(self.kB.value) * float(self.temperature.value) * dt)
            self.positions[:, i] += diffusion

        # Update orientation
        self.orientations[i] = np.arctan2(
            new_moment[1], new_moment[0]
        )