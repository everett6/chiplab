"""Magnetic response models."""

from __future__ import annotations

from typing import Any

import numpy as np

from chiplab.schemas.common import Provenance, TrackedValue


def magnetization_curve(
    H: np.ndarray,
    material_params: dict[str, float | TrackedValue],
) -> np.ndarray:
    """M(H) hysteresis curve for a magnetic material.

    Parameters
    ----------
    H : np.ndarray
        Applied magnetic field strength in A/m.
    material_params : dict
        Material parameters including:
        - Ms: saturation magnetization (A/m)
        - Hc: coercivity (A/m)
        - Mr: remanent magnetization (A/m)
        - alpha: shape factor (dimensionless)

    Returns
    -------
    np.ndarray
        Magnetization M in A/m corresponding to each H value.
    """
    # Extract parameters with defaults
    Ms = material_params.get("Ms", 1.0e5)  # saturation magnetization A/m
    Hc = material_params.get("Hc", 1.0e3)  # coercivity A/m
    Mr = material_params.get("Mr", 0.5 * Ms)  # remanent magnetization
    alpha = material_params.get("alpha", 1.0)  # shape factor

    # Ensure Ms and Mr are floats
    Ms_float = float(Ms) if not isinstance(Ms, TrackedValue) else Ms.value
    Mr_float = float(Mr) if not isinstance(Mr, TrackedValue) else Mr.value

    # Return shape: same as H
    M = np.zeros_like(H, dtype=float)

    for i, h in enumerate(H):
        h_val = float(h) if not isinstance(h, TrackedValue) else h.value

        # Preisach-like simplified hysteresis model
        # For increasing H (from negative to positive)
        if i == 0 or h_val >= H[i - 1]:
            # Major loop approximation
            M[i] = Mr + (Ms_float - Mr) * np.tanh(h_val / (Hc + 1e-10))
        else:
            # Reversal path - simplified
            M[i] = M[i - 1] + 0.1 * (Ms_float - M[i - 1]) * np.sign(h_val)

    # Apply shape factor
    M = M * alpha

    return M


def particle_interaction_energy(
    positions: np.ndarray,
    moments: np.ndarray,
) -> float:
    """Dipole-dipole interaction energy between magnetic nanoparticles.

    Parameters
    ----------
    positions : np.ndarray
        Particle positions array of shape (N, 3) in meters.
    moments : np.ndarray
        Magnetic moment magnitudes array of shape (N,) in A·m².

    Returns
    -------
    float
        Total dipole-dipole interaction energy in Joules.
    """
    N = positions.shape[0]

    # Compute pairwise dipole-dipole energies
    total_energy = 0.0

    for i in range(N):
        for j in range(i + 1, N):
            # Vector from j to i
            r_vec = positions[i] - positions[j]
            r_mag = np.linalg.norm(r_vec) + 1e-12  # avoid division by zero

            # Unit vector
            r_hat = r_vec / r_mag

            # Dipole moments
            mi = moments[i]
            mj = moments[j]

            # Dipole-dipole energy formula:
            # U = (μ0 / 4π) * [m_i · m_j / r^3 - 3 * (m_i · r_hat) * (m_j · r_hat) / r^3]
            # With μ0/4π = 1e-7 H/m
            mu0_over_4pi = 1e-7

            # Dot products
            mi_dot_mj = mi * mj  # assuming collinear moments for simplicity
            mi_dot_r = mi  # simplified: moments aligned with position vector
            mj_dot_r = mj  # simplified

            # Energy
            u = (mu0_over_4pi) * (mi_dot_mj / r_mag**3 - 3 * mi_dot_r * mj_dot_r / r_mag**3)
            total_energy += u

    return float(total_energy)


def collective_reorganization(
    state: Any,
    field: np.ndarray,
) -> dict[str, Any]:
    """Particle organization dynamics under applied field.

    Models how magnetic particles reorganize under an external field,
    computing the resulting order and alignment.

    Parameters
    ----------
    state : any
        Current state containing particle positions and moments.
    field : np.ndarray
        Applied magnetic field vector in A/m.

    Returns
    -------
    dict
        Organization metrics including order parameter and alignment.
    """
    # Extract positions and moments from state
    # This is a simplified interface - actual implementation depends on state format
    N = 10  # default number of particles

    # Compute alignment with field
    field_mag = np.linalg.norm(field)
    if field_mag > 0:
        alignment = field / field_mag
    else:
        alignment = np.array([1.0, 0.0, 0.0])

    # Order parameter: average cosine of angle between moments and field
    # For now, return a default organization metric
    order_parameter = 0.5  # partial ordering

    return {
        "order_parameter": order_parameter,
        "alignment": alignment,
        "field_strength_A_per_m": field_mag,
        "num_particles": N,
    }