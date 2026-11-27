"""Level 3: particle-scale reference model for single-domain magnetic nanoparticles.

Each particle is a macrospin (unit vector m, moment mu = Ms*V) fixed in a rigid matrix. Its
moment follows the stochastic Landau-Lifshitz-Gilbert equation in SI units (H in A/m):

    dm/dt = -gamma0/(1+alpha^2) * [ m x H + alpha * m x (m x H) ],   gamma0 = gamma * mu0

with H = applied + uniaxial anisotropy + dipole-dipole + thermal field. The thermal field is
Brown's (1963) white noise with per-component standard deviation
sqrt(2*alpha*kB*T / (gamma0*mu0*Ms*V*dt)), so Neel relaxation comes out of the dynamics instead
of being added by hand. It is integrated with the Heun (Stratonovich) scheme and m is
renormalised every step.

Not modelled: Brownian (mechanical) rotation, hydrodynamics, particle size distributions.
Translation is off unless ``viscosity_pa_s`` is given. Material parameters are assumptions
unless the caller supplies measured ones. Ferrofluid inputs are refused; use
``models.materials.ferro_device`` and ``ferro_particles.brownian_rotation_monte_carlo``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from chiplab.core.gpu import get_gpu_backend
from chiplab.credibility import refuse_ferrofluid
from chiplab.schemas.common import Provenance, TrackedValue

MU0 = 4e-7 * math.pi           # H/m
KB = 1.380649e-23              # J/K
GAMMA_E = 1.760859e11          # rad/(s*T), electron gyromagnetic ratio
MAX_SUBSTEPS = 2_000_000


def _as_3n(a: Any, n: int, name: str) -> np.ndarray:
    """Accept (3,), (3,N) or (N,3) and return a float (3,N) array."""
    arr = np.asarray(a, dtype=float)
    if arr.shape == (3,):
        return np.repeat(arr.reshape(3, 1), n, axis=1)
    if arr.shape == (3, n):
        return arr.copy()
    if arr.shape == (n, 3):
        return arr.T.copy()
    raise ValueError(f"{name} must have shape (3,), (3,{n}) or ({n},3); got {arr.shape}")


def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=0, keepdims=True)
    if np.any(norm == 0) or not np.all(np.isfinite(norm)):
        raise ValueError("moment/axis vectors must be finite and non-zero")
    return v / norm


class ParticleScaleSolver:
    """Stochastic-LLG macrospin solver for a small assembly of nanoparticles.

    Intended for small regions (tens to a few hundred particles) to check simpler models.
    Dipole coupling is a direct O(N^2) sum with no periodic images.
    """

    def __init__(
        self,
        num_particles: int = 100,
        grid_size: tuple[int, int, int] = (10, 10, 10),
        dt: float = 1e-12,
        device: str = "auto",
        material: str | None = None,
        *,
        ms_a_per_m: float = 4.8e5,
        diameter_m: float = 10e-9,
        anisotropy_j_per_m3: float = 1.0e4,
        alpha: float = 0.1,
        temperature_k: float = 300.0,
        spacing_m: float | None = None,
        dipolar: bool = True,
        viscosity_pa_s: float | None = None,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        num_particles : int
            Number of particles; placed on the first sites of a grid_size lattice.
        grid_size : (nx, ny, nz)
            Lattice used to place particles when no positions are given.
        dt : float
            Largest internal time step (s). ``solve`` sub-steps longer intervals.
        device : str
            Kept for API compatibility; the O(N^2) sums run in NumPy.
        material : str, optional
            Ferrofluid is refused.
        ms_a_per_m, diameter_m, anisotropy_j_per_m3, alpha, temperature_k :
            Saturation magnetisation, magnetic core diameter, uniaxial anisotropy constant,
            Gilbert damping and temperature. Defaults are magnetite-like assumptions.
        spacing_m : float, optional
            Lattice spacing (default 2 * diameter, i.e. particles not touching).
        dipolar : bool
            Include dipole-dipole fields.
        viscosity_pa_s : float, optional
            If given, particles translate by Stokes-Einstein diffusion in that medium.
        seed : int, optional
            Seed for reproducible noise and placement.
        """
        refuse_ferrofluid("ParticleScaleSolver", material)
        if num_particles < 1:
            raise ValueError("num_particles must be >= 1")
        if int(np.prod(grid_size)) < num_particles:
            raise ValueError("grid_size has fewer sites than num_particles")
        for name, v in (("dt", dt), ("ms_a_per_m", ms_a_per_m), ("diameter_m", diameter_m),
                        ("alpha", alpha)):
            if not (v > 0 and math.isfinite(v)):
                raise ValueError(f"{name} must be positive and finite")
        if temperature_k < 0 or anisotropy_j_per_m3 < 0:
            raise ValueError("temperature_k and anisotropy_j_per_m3 must be >= 0")

        self.num_particles = int(num_particles)
        self.grid_size = tuple(int(g) for g in grid_size)
        self.dt = float(dt)
        self.device = device
        self.backend = get_gpu_backend(device=device)
        self.dipolar = dipolar
        self.viscosity_pa_s = viscosity_pa_s
        self.spacing_m = spacing_m if spacing_m is not None else 2.0 * diameter_m
        self.rng = np.random.default_rng(seed)

        self.gamma = TrackedValue(value=GAMMA_E, provenance=Provenance.PUBLISHED, unit="rad/s/T")
        self.mu0 = TrackedValue(value=MU0, provenance=Provenance.PUBLISHED, unit="H/m")
        self.kB = TrackedValue(value=KB, provenance=Provenance.PUBLISHED, unit="J/K")
        self.alpha = TrackedValue(value=alpha, provenance=Provenance.ASSUMED, unit="")
        self.temperature = TrackedValue(value=temperature_k, provenance=Provenance.ASSUMED, unit="K")
        self.ms = TrackedValue(value=ms_a_per_m, provenance=Provenance.ASSUMED, unit="A/m")
        self.diameter = TrackedValue(value=diameter_m, provenance=Provenance.ASSUMED, unit="m")
        self.anisotropy = TrackedValue(value=anisotropy_j_per_m3, provenance=Provenance.ASSUMED,
                                       unit="J/m^3")

        self.positions: np.ndarray | None = None     # (3,N) m
        self.moments: np.ndarray | None = None       # (3,N) unit vectors
        self.easy_axes: np.ndarray | None = None     # (3,N) unit vectors
        self.orientations: np.ndarray | None = None  # (N,) azimuth of m, rad
        self.time_s = 0.0

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    @property
    def volume_m3(self) -> float:
        return math.pi / 6.0 * float(self.diameter.value) ** 3

    @property
    def moment_am2(self) -> float:
        """Particle magnetic moment mu = Ms * V (A*m^2)."""
        return float(self.ms.value) * self.volume_m3

    @property
    def gamma0(self) -> float:
        return float(self.gamma.value) * MU0

    @property
    def anisotropy_field_a_per_m(self) -> float:
        """H_K = 2K / (mu0 Ms); the T=0 easy-axis switching field (Stoner-Wohlfarth)."""
        return 2.0 * float(self.anisotropy.value) / (MU0 * float(self.ms.value))

    def thermal_field_std(self, dt: float) -> float:
        """Per-component std of Brown's thermal field for one step of length dt (A/m)."""
        t = float(self.temperature.value)
        if t == 0:
            return 0.0
        a = float(self.alpha.value)
        return math.sqrt(2.0 * a * KB * t / (self.gamma0 * MU0 * float(self.ms.value)
                                             * self.volume_m3 * dt))

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def _initialize_particles(self, particles: dict[str, Any] | None) -> None:
        p = particles or {}
        n = self.num_particles
        if p.get("positions") is not None:
            self.positions = _as_3n(p["positions"], n, "positions")
        else:
            nx, ny, nz = self.grid_size
            idx = np.arange(n)
            sites = np.stack([idx % nx, (idx // nx) % ny, idx // (nx * ny)]).astype(float)
            self.positions = sites * self.spacing_m
        if p.get("moments") is not None:
            self.moments = _unit(_as_3n(p["moments"], n, "moments"))
        else:
            self.moments = _unit(self.rng.standard_normal((3, n)))
        if p.get("easy_axes") is not None:
            self.easy_axes = _unit(_as_3n(p["easy_axes"], n, "easy_axes"))
        else:
            self.easy_axes = _unit(self.rng.standard_normal((3, n)))
        self._check_overlap()
        self.orientations = np.arctan2(self.moments[1], self.moments[0])

    def _check_overlap(self) -> None:
        assert self.positions is not None
        if self.num_particles < 2:
            return
        d = self.positions[:, :, None] - self.positions[:, None, :]
        r = np.sqrt((d ** 2).sum(axis=0))
        np.fill_diagonal(r, np.inf)
        if r.min() < float(self.diameter.value) * (1 - 1e-9):
            raise ValueError("particles overlap (centre spacing < diameter)")

    # ------------------------------------------------------------------
    # Fields (A/m)
    # ------------------------------------------------------------------
    def _dipole_fields(self, m: np.ndarray) -> np.ndarray:
        """Dipole field on every particle from all others: (mu/4pi) (3(m.r)r - m) / r^3."""
        assert self.positions is not None
        d = self.positions[:, :, None] - self.positions[:, None, :]   # r_i - r_j, (3,N,N)
        r = np.sqrt((d ** 2).sum(axis=0))
        np.fill_diagonal(r, np.inf)
        rhat = d / r
        mdotr = np.einsum("kj,kij->ij", m, rhat)                      # m_j . rhat_ij
        h = (3.0 * mdotr[None] * rhat - m[:, None, :]) / r[None] ** 3
        return self.moment_am2 / (4.0 * math.pi) * h.sum(axis=2)

    def _dipole_interaction(self, i: int, j: int, moments: np.ndarray) -> np.ndarray:
        """Dipole field (A/m) at particle i from particle j; moments are unit vectors."""
        mom = np.asarray(moments, dtype=float)
        if mom.shape[0] != 3:
            mom = mom.T
        if self.positions is None:
            self._initialize_particles({})
        assert self.positions is not None
        r_vec = self.positions[:, i] - self.positions[:, j]
        r = float(np.linalg.norm(r_vec))
        if i == j or r == 0:
            return np.zeros(3)
        rhat = r_vec / r
        mj = mom[:, j]
        return self.moment_am2 / (4.0 * math.pi) * (3.0 * np.dot(mj, rhat) * rhat - mj) / r ** 3

    def _deterministic_field(self, m: np.ndarray, h_app: np.ndarray) -> np.ndarray:
        assert self.easy_axes is not None
        h = h_app + self.anisotropy_field_a_per_m * (m * self.easy_axes).sum(0) * self.easy_axes
        if self.dipolar and self.num_particles > 1:
            h = h + self._dipole_fields(m)
        return h

    def _llg_rhs(self, m: np.ndarray, h: np.ndarray) -> np.ndarray:
        a = float(self.alpha.value)
        mxh = np.cross(m, h, axis=0)
        return -self.gamma0 / (1.0 + a * a) * (mxh + a * np.cross(m, mxh, axis=0))

    def _step(self, h_app: np.ndarray, dt: float) -> None:
        assert self.moments is not None
        sigma = self.thermal_field_std(dt)
        h_th = sigma * self.rng.standard_normal(self.moments.shape) if sigma else 0.0
        m0 = self.moments
        k1 = self._llg_rhs(m0, self._deterministic_field(m0, h_app) + h_th)
        m_pred = _unit(m0 + dt * k1)
        k2 = self._llg_rhs(m_pred, self._deterministic_field(m_pred, h_app) + h_th)
        self.moments = _unit(m0 + 0.5 * dt * (k1 + k2))
        if self.viscosity_pa_s:
            diff = KB * float(self.temperature.value) / (
                3.0 * math.pi * self.viscosity_pa_s * float(self.diameter.value))
            assert self.positions is not None
            self.positions = self.positions + math.sqrt(2.0 * diff * dt) * \
                self.rng.standard_normal(self.positions.shape)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(self, particles: dict[str, Any] | None, field: np.ndarray, dt: float
              ) -> dict[str, Any]:
        """Advance the assembly by ``dt`` seconds under an applied field (A/m).

        ``field`` is (3,) for a uniform field or (3,N)/(N,3) per particle. ``particles`` may
        give initial ``positions`` (m), ``moments`` and ``easy_axes``; it is read on the first
        call only. Intervals longer than the solver's ``dt`` are sub-stepped.
        """
        refuse_ferrofluid("ParticleScaleSolver", particles)
        if not (dt > 0 and math.isfinite(dt)):
            raise ValueError("dt must be positive and finite")
        if self.moments is None:
            self._initialize_particles(particles)
        h_app = _as_3n(field, self.num_particles, "field")
        if not np.all(np.isfinite(h_app)):
            raise ValueError("field must be finite")
        n_sub = max(1, math.ceil(dt / self.dt - 1e-9))
        if n_sub > MAX_SUBSTEPS:
            raise ValueError(f"dt/self.dt = {n_sub} sub-steps exceeds {MAX_SUBSTEPS}")
        h = dt / n_sub
        for _ in range(n_sub):
            self._step(h_app, h)
        self.time_s += dt
        assert self.moments is not None and self.positions is not None
        self.orientations = np.arctan2(self.moments[1], self.moments[0])
        return {
            "positions": self.positions.copy(),
            "moments": self.moments.copy(),
            "orientations": self.orientations.copy(),
            "mean_m": self.moments.mean(axis=1),
            "time_s": self.time_s,
            "substeps": n_sub,
            "units": {"positions": "m", "moments": "unit vector (moment = Ms*V)",
                      "field": "A/m", "time_s": "s"},
            "model": "stochastic LLG macrospins, Heun scheme, rigid matrix",
        }
