"""Electrical impedance models."""

from __future__ import annotations

from typing import Any

import numpy as np

from chiplab.schemas.common import Provenance, TrackedValue


def impedance_matrix(
    state: Any,
    electrode_positions: np.ndarray,
    frequencies: np.ndarray | list[float],
) -> np.ndarray:
    """Electrical impedance matrix for a ferrofluid between electrodes.

    Computes the complex impedance Z(f) between electrode pairs
    based on the ferrofluid state (conductivity, permittivity, particle organization).

    Parameters
    ----------
    state : any
        Ferrofluid state containing conductivity and organization parameters.
    electrode_positions : np.ndarray
        Electrode positions array of shape (M, 3) in meters.
    frequencies : np.ndarray or list of float
        Frequency points in Hz.

    Returns
    -------
    np.ndarray
        Complex impedance matrix of shape (len(frequencies), M, M) where
        Z[k, i, j] is the impedance between electrodes i and j at frequency k.
    """
    freqs = np.array(frequencies, dtype=float)

    M = electrode_positions.shape[0]  # number of electrodes

    # Initialize impedance matrix
    Z = np.zeros((len(freqs), M, M), dtype=complex)

    # Get conductivity from state (simplified interface)
    # In a real implementation, this would come from state.electrical_impedance or similar
    sigma = 0.1  # baseline conductivity S/m (assumed)
    epsilon_r = 80  # relative permittivity (water-like)
    epsilon_0 = 8.854e-12  # vacuum permittivity F/m

    # Debye relaxation parameters
    tau = 1e-6  # relaxation time s

    for k, f in enumerate(freqs):
        omega = 2.0 * np.pi * f

        # Complex permittivity: ε*(ω) = ε_inf + (ε_s - ε_inf) / (1 + jωτ)
        # simplified: ε*(ω) = ε_0 * ε_r / (1 + jωτ)
        complex_permittivity = epsilon_0 * epsilon_r / (1.0 + 1j * omega * tau)

        # Complex conductivity: σ*(ω) = jωε*(ω)
        complex_conductivity = 1j * omega * complex_permittivity

        # For each electrode pair, compute impedance
        for i in range(M):
            for j in range(M):
                if i == j:
                    # Self-impedance: open circuit or reference
                    Z[k, i, j] = complex(1e6, 0)  # 1MΩ reference
                else:
                    # Vector between electrodes
                    r_ij = electrode_positions[i] - electrode_positions[j]
                    r_mag = np.linalg.norm(r_ij) + 1e-12

                    # Effective conductance through ferrofluid
                    # G = σ * A / L, simplified as conductance proportional to σ
                    # and inversely proportional to distance
                    A = 1e-4  # effective cross-section m² (assumed)
                    L = r_mag  # distance

                    # Impedance: Z = 1 / (σ* + jωε*) * L / A
                    # Using complex conductivity
                    if abs(complex_conductivity) > 1e-15:
                        Z_k_ij = L / (A * complex_conductivity)
                    else:
                        Z_k_ij = complex(1e6, 0)  # high impedance default

                    Z[k, i, j] = Z_k_ij

    return Z


def effective_medium_impedance(
    volume_fraction: float | TrackedValue,
    particle_impedance: float | TrackedValue,
    matrix_impedance: float | TrackedValue,
) -> float:
    """Effective medium impedance using Maxwell-Garnett or Bruggeman formula.

    Computes the effective complex impedance of a composite material
    with conductive particles in a matrix medium.

    Parameters
    ----------
    volume_fraction : float or TrackedValue
        Volume fraction of particles (0 to 1).
    particle_impedance : float or TrackedValue
        Complex impedance of the particle phase in Ω.
    matrix_impedance : float or TrackedValue
        Complex impedance of the matrix phase in Ω.

    Returns
    -------
    float
        Effective complex impedance in Ω (real part returned for simplicity,
        or could return complex).
    """
    # Unwrap values
    f = float(volume_fraction) if not isinstance(volume_fraction, TrackedValue) else volume_fraction.value
    Zp = float(particle_impedance) if not isinstance(particle_impedance, TrackedValue) else particle_impedance.value
    Zm = float(matrix_impedance) if not isinstance(matrix_impedance, TrackedValue) else matrix_impedance.value

    # Maxwell-Garnett effective medium approximation for spherical inclusions:
    # (ε_eff - ε_m) / (ε_eff + 2ε_m) = f * (ε_p - ε_m) / (ε_p + 2ε_m)
    # For impedance (inverse of permittivity-related quantities):
    # Z_eff = Z_m * (Z_p + 2Z_m - 2f(Z_m - Z_p)) / (Z_p + 2Z_m + f(Z_m - Z_p))

    # Avoid division by zero
    denom = Zp + 2.0 * Zm + f * (Zm - Zp)
    if abs(denom) < 1e-12:
        return Zm  # fallback to matrix impedance

    Z_eff = Zm * (Zp + 2.0 * Zm - 2.0 * f * (Zm - Zp)) / denom

    return Z_eff


def frequency_response(
    state: Any,
    f_min: float = 1e3,
    f_max: float = 1e9,
    n_points: int = 100,
) -> dict[str, Any]:
    """Impedance spectrum of the ferrofluid material.

    Computes the frequency-dependent impedance response across a range
    of frequencies, revealing relaxation processes and electrode effects.

    Parameters
    ----------
    state : any
        Ferrofluid state containing conductivity, permittivity, and geometry.
    f_min : float, optional
        Minimum frequency in Hz (default 1 kHz).
    f_max : float, optional
        Maximum frequency in Hz (default 1 GHz).
    n_points : int, optional
        Number of frequency points (default 100).

    Returns
    -------
    dict
        Impedance spectrum with frequency, real/imaginary impedance, and magnitude.
    """
    # Generate logarithmically spaced frequency points
    frequencies = np.logspace(np.log10(f_min), np.log10(f_max), n_points)

    # Electrode positions (simplified: 4 electrodes in a line)
    electrode_positions = np.array([
        [0.0, 0.0, 0.0],
        [1e-2, 0.0, 0.0],   # 1 cm spacing
        [2e-2, 0.0, 0.0],   # 2 cm spacing
        [3e-2, 0.0, 0.0],   # 3 cm spacing
    ])

    # Compute impedance matrix
    Z = impedance_matrix(state, electrode_positions, frequencies)

    # Extract S-parameters or Z-parameters metrics
    magnitudes = np.abs(Z).mean(axis=(1, 2))  # average magnitude across electrode pairs
    real_parts = np.real(Z).mean(axis=(1, 2))
    imag_parts = np.imag(Z).mean(axis=(1, 2))

    # Find characteristic frequencies
    # Peak in imaginary part indicates relaxation frequency
    peak_idx = np.argmax(np.abs(imag_parts))
    peak_freq = frequencies[peak_idx]

    return {
        "frequencies_Hz": frequencies.tolist(),
        "impedance_real_Ω": real_parts.tolist(),
        "impedance_imag_Ω": imag_parts.tolist(),
        "impedance_magnitude_Ω": magnitudes.tolist(),
        "peak_frequency_Hz": peak_freq,
        "frequency_points": n_points,
        "f_min_Hz": f_min,
        "f_max_Hz": f_max,
    }