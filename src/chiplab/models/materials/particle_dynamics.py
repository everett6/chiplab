"""Nanoparticle dynamics models."""

from __future__ import annotations

from typing import Any

import numpy as np

from chiplab.schemas.common import Provenance, TrackedValue


def brownian_motion(
    particles: np.ndarray,
    dt: float | TrackedValue,
    temperature: float | TrackedValue,
) -> np.ndarray:
    """Brownian motion of nanoparticles in a fluid.

    Updates particle positions due to thermal random forces.

    Parameters
    ----------
    particles : np.ndarray
        Particle state array of shape (N, 6) where columns are [x, y, z, vx, vy, vz]
        in meters and meters/second.
    dt : float or TrackedValue
        Time step in seconds.
    temperature : float or TrackedValue
        Fluid temperature in Kelvin.

    Returns
    -------
    np.ndarray
        Updated particle positions with Brownian displacement.
    """
    N = particles.shape[0]

    # Unpack temperature
    T = float(temperature) if not isinstance(temperature, TrackedValue) else temperature.value

    # Brownian diffusion coefficient: D = kB * T / (3 * π * η * d)
    # Simplified: assume η (viscosity) = 1e-3 Pa·s (water-like), d = 10 nm particle diameter
    kB = 1.38e-23  # Boltzmann constant J/K
    eta = 1e-3  # dynamic viscosity Pa·s
    particle_diameter = 10e-9  # particle diameter 10 m

    D = kB * T / (3.0 * np.pi * eta * particle_diameter)  # diffusion coefficient m²/s

    # Time step displacement std dev: sqrt(6 * D * dt) for 3D
    displacement_std = np.sqrt(6.0 * D * float(dt) if not isinstance(dt, TrackedValue) else dt.value)

    # Generate random displacements
    displacements = np.random.normal(
        0.0, displacement_std, size=(N, 3)
    )

    # Update positions
    new_positions = particles[:, :3].copy() + displacements

    # Return updated particles (only positions changed for this function)
    new_particles = particles.copy()
    new_particles[:, :3] = new_positions

    return new_particles


def neel_relaxation(
    particles: np.ndarray,
    field: np.ndarray,
    dt: float | TrackedValue,
) -> np.ndarray:
    """Neel relaxation of magnetic moments.

    Models the thermal relaxation of magnetic moments in nanoparticles
    towards equilibrium alignment with an external field.

    Parameters
    ----------
    particles : np.ndarray
        Particle state array of shape (N, 3) with magnetic moment vectors.
        If 1D of length N*3, will be reshaped.
    field : np.ndarray
        External magnetic field vector in A/m.
    dt : float or TrackedValue
        Time step in seconds.

    Returns
    -------
    np.ndarray
        Updated magnetic moment vectors after Neel relaxation.
        Same format as input (flat if input was flat, 2D if input was 2D).
    """
    # Determine if input is flat (1D) or 2D (N, 3)
    flat_input = particles.ndim == 1
    if flat_input:
        particles = particles.reshape(-1, 3)

    # Ensure particles is 2D (N, 3)
    particles = np.atleast_2d(particles)
    N = particles.shape[0]

    # Unpack dt
    d = float(dt) if not isinstance(dt, TrackedValue) else dt.value

    # Unpack field
    field_mag = np.linalg.norm(field)
    field_hat = field / max(field_mag, 1e-12)

    # Neel relaxation time: τ_N = τ_0 * exp(KV / kB*T)
    # For simplicity, use typical values: τ_0 = 1e-9 s, K_V / kB ≈ 1000 K
    tau_0 = 1e-9  # attempt time s
    KV_kB = 1000.0  # effective energy barrier in K

    # Initial relaxation time - use magnitude of moments as energy scale
    moment_mags = np.linalg.norm(particles, axis=1)
    energy_scale = np.max(moment_mags) if N > 0 else 1.0
    tau_N = tau_0 * np.exp(KV_kB / max(1.0, energy_scale))

    # Relaxation factor
    decay = np.exp(-d / max(tau_N, 1e-12))

    # Current moment direction
    moment_hat = particles / max(moment_mags.max(), 1e-12)

    # Target alignment with field
    target_hat = field_hat[np.newaxis, :] if field_hat.ndim == 1 else field_hat

    # Relaxed moments: towards field alignment
    relaxed_hat = decay * moment_hat + (1.0 - decay) * target_hat
    relaxed_hat = relaxed_hat / np.maximum(np.linalg.norm(relaxed_hat, axis=1, keepdims=True), 1e-12)

    # Preserve magnitude, only change direction
    new_moments = moment_mags[:, np.newaxis] * relaxed_hat

    # Return in original format
    if flat_input:
        return new_moments.flatten()
    return new_moments


def brownian_relaxation(
    particles: np.ndarray,
    viscosity: float | TrackedValue,
    dt: float | TrackedValue,
) -> np.ndarray:
    """Physical rotation relaxation of particles (Stokes drag).

    Models the viscous drag on rotating magnetic nanoparticles.

    Parameters
    ----------
    particles : np.ndarray
        Particle state with angular velocity [ωx, ωy, ωz] in rad/s.
    viscosity : float or TrackedValue
        Fluid dynamic viscosity in Pa·s.
    dt : float or TrackedValue
        Time step in seconds.

    Returns
    -------
    np.ndarray
        Updated angular velocities after viscous relaxation.
    """
    N = particles.shape[0] if particles.ndim > 1 else 1

    # Unpack viscosity
    eta = float(viscosity) if not isinstance(viscosity, TrackedValue) else viscosity.value

    # Unpack dt
    d = float(dt) if not isinstance(dt, TrackedValue) else dt.value

    # Rotational drag coefficient: ζ = 8πηr³ for sphere of radius r
    # Simplified: assume particle radius r = 5 nm
    r = 5e-9  # radius in meters
    zeta = 8.0 * np.pi * eta * r**3  # rotational drag coefficient

    # Relaxation time for rotation: τ = ζ / (kB*T) - simplified without T
    # Use typical room temperature
    kB = 1.38e-23
    T = 300.0
    tau_rot = zeta / (kB * T)

    # Decay factor
    decay = np.exp(-d / max(tau_rot, 1e-12))

    # Update angular velocities
    new_omega = decay * particles

    return new_omega


def concentration_field(
    particles: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    """Compute particle concentration field on a grid.

    Projects particle positions onto a 3D grid to get a concentration
    distribution.

    Parameters
    ----------
    particles : np.ndarray
        Particle positions array of shape (N, 3) in meters.
    grid_size : int
        Number of grid cells per dimension.

    Returns
    -------
    np.ndarray
        3D concentration array of shape (grid_size, grid_size, grid_size).
    """
    N = particles.shape[0]

    # Grid boundaries (assume particles in [-1, 1] × [-1, 1] × [-1, 1] m cube)
    # Normalize particle positions to grid coordinates
    grid_coords = (particles + 1.0) / 2.0 * (grid_size - 1)  # map [0, grid_size-1]

    # Clip to grid bounds
    grid_coords = np.clip(grid_coords, 0, grid_size - 1)

    # Initialize concentration grid
    concentration = np.zeros((grid_size, grid_size, grid_size), dtype=float)

    # Assign particles to grid (simple voxel assignment)
    for i in range(N):
        xi, yi, zi = int(grid_coords[i, 0]), int(grid_coords[i, 1]), int(grid_coords[i, 2])
        concentration[xi, yi, zi] += 1.0

    # Normalize by total number of particles and grid volume
    concentration = concentration / N if N > 0 else concentration
    concentration = concentration / (grid_size ** 3) if grid_size > 0 else concentration

    return concentration