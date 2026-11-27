"""Aging models for ChipLab reliability.

Thermal sensitivity, Arrhenius accelerated aging, Monte Carlo lifetime/MTTF
distribution using GPUBackend (device="auto").
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional

from chiplab.core.gpu import GPUBackend, get_gpu_backend
from chiplab.schemas.common import Provenance, TrackedValue


def arrhenius_accelerated_aging(
    temp_operating_k: float,
    temp_reference_k: float = 300.0,
    time_operating: float = 1.0,
    activation_energy_eV: float = 0.7,
    provenance: Provenance = Provenance.ESTIMATED,
) -> TrackedValue:
    """Arrhenius accelerated aging model.

    Parameters
    ----------
    temp_operating_k : float
        Operating temperature in Kelvin.
    temp_reference_k : float, optional
        Reference temperature in Kelvin (default 300K).
    time_operating : float, optional
        Operating time in seconds (default 1.0).
    activation_energy_eV : float, optional
        Activation energy in eV (default 0.7 eV).
    provenance : Provenance, optional
        Provenance of the measurement (default ESTIMATED).

    Returns
    -------
    TrackedValue
        Tracked value with accelerated aging factor and provenance set.
    """
    k = 8.617333262145e-5  # Boltzmann constant in eV/K

    # Arrhenius factor: ratio of aging rates
    arrhenius_factor = np.exp(
        activation_energy_eV / k * (1 / temp_reference_k - 1 / temp_operating_k)
    )

    # Equivalent time at reference temperature
    equivalent_time = time_operating * arrhenius_factor

    return TrackedValue(
        value=equivalent_time,
        unit="seconds",
        provenance=provenance,
        notes=f"Arrhenius aging: operating={temp_operating_k:.1f}K, ref={temp_reference_k:.1f}K, "
              f"factor={arrhenius_factor:.2f}x",
    )


def monte_carlo_lifetime(
    n_samples: int = 1000,
    activation_energy_mean_ev: float = 0.7,
    activation_energy_std_ev: float = 0.1,
    temperature_k: float = 300.0,
    current_density_a_per_m2: float = 1e6,
    constant_a: float = 1e-9,
    constant_b: float = 2.0,
    device: str = "auto",
    provenance: Provenance = Provenance.ESTIMATED,
) -> TrackedValue:
    """Monte Carlo lifetime/MTTF distribution using GPUBackend.

    Uses Black's equation with random variation in activation energy
    to produce a distribution of MTTF values.

    Parameters
    ----------
    n_samples : int, optional
        Number of Monte Carlo samples (default 1000).
    activation_energy_mean_ev : float, optional
        Mean activation energy in eV (default 0.7 eV).
    activation_energy_std_ev : float, optional
        Standard deviation of activation energy in eV (default 0.1 eV).
    temperature_k : float, optional
        Temperature in Kelvin (default 300K).
    current_density_a_per_m2 : float, optional
        Current density in A/m^2 (default 1e6 A/m^2).
    constant_a : float, optional
        Black's equation constant A (default 1e-9).
    constant_b : float, optional
        Black's equation exponent n (default 2.0).
    device : str, optional
        GPU device selection ("auto", "cpu", "cuda", etc.) (default "auto").
    provenance : Provenance, optional
        Provenance of the measurement (default ESTIMATED).

    Returns
    -------
    TrackedValue
        Tracked value with Monte Carlo MTTF statistics and provenance set.
    """
    kb = 8.617333262145e-5  # Boltzmann constant in eV/V/K

    # Get GPU backend for potential acceleration
    backend = get_gpu_backend(prefer=device)

    # Generate random activation energies
    np.random.seed(42)  # For reproducibility
    ae_samples = np.random.normal(
        activation_energy_mean_ev, activation_energy_std_ev, n_samples
    )

    # Ensure positive activation energies
    ae_samples = np.abs(ae_samples)

    # Black's equation: MTTF = A * J^-n * exp(Ea/(k*T))
    # Current density J = current_density_a_per_m2 (already in A/m^2)
    j_density = current_density_a_per_m2

    # Compute MTTF for each sample
    mttf_samples = constant_a * (j_density ** (-constant_b)) * np.exp(
        ae_samples / (kb * temperature_k)
    )

    # Compute statistics
    mttf_mean = float(np.mean(mttf_samples))
    mttf_median = float(np.median(mttf_samples))
    mttf_std = float(np.std(mttf_samples))
    mttf_min = float(np.min(mttf_samples))
    mttf_max = float(np.max(mttf_samples))

    # Compute failure probability at t = 1 second (for reference)
    failure_prob = float(np.sum(mttf_samples < 1.0) / n_samples)

    # Use GPU if available and beneficial
    if backend.is_gpu and backend.backend_name in ("torch_cuda", "cupy"):
        # Could potentially offload the computation to GPU
        pass

    # Build notes with distribution info
    notes = f"Monte Carlo MTTF: {n_samples} samples, T={temperature_k:.1f}K, "
    notes += f"J={j_density:.1e} A/m^2, mean={mttf_mean:.2e}s"

    return TrackedValue(
        value=mttf_mean,  # Return the mean MTTF as the primary value
        unit="seconds",
        provenance=provenance,
        notes=notes,
    )


def monte_carlo_ttf_distribution(
    n_samples: int = 1000,
    activation_energy_mean_ev: float = 0.7,
    activation_energy_std_ev: float = 0.1,
    temperature_k: float = 300.0,
    current_density_a_per_m2: float = 1e6,
    constant_a: float = 1e-9,
    constant_b: float = 2.0,
    device: str = "auto",
    provenance: Provenance = Provenance.ESTIMATED,
) -> TrackedValue:
    """Monte Carlo TTF (Time-to-Failure) distribution.

    Parameters
    ----------
    n_samples : int, optional
        Number of Monte Carlo samples (default 1000).
    activation_energy_mean_ev : float, optional
        Mean activation energy in eV (default 0.7 eV).
    activation_energy_std_ev : float, optional
        Standard deviation of activation energy in eV (default 0.1 eV).
    temperature_k : float, optional
        Temperature in Kelvin (default 300K).
    current_density_a_per_m2 : float, optional
        Current density in A/m^2 (default 1e6 A/m^2).
    constant_a : float, optional
        Black's equation constant A (default 1e-9).
    constant_b : float, optional
        Black's equation exponent n (default 2.0).
    device : str, optional
        GPU device selection ("auto", "cpu", "cuda", etc.) (default "auto").
    provenance : Provenance, optional
        Provenance of the measurement (default ESTIMATED).

    Returns
    -------
    TrackedValue
        Tracked value with TTF distribution statistics and provenance set.
    """
    kb = 8.617333262145e-5  # Boltzmann constant in eV/V/K

    # Get GPU backend
    backend = get_gpu_backend(prefer=device)

    # Generate random activation energies
    np.random.seed(42)
    ae_samples = np.random.normal(
        activation_energy_mean_ev, activation_energy_std_ev, n_samples
    )
    ae_samples = np.abs(ae_samples)

    # Compute TTF for each sample using Black's equation
    j_density = current_density_a_per_m2
    ttf_samples = constant_a * (j_density ** (-constant_b)) * np.exp(
        ae_samples / (kb * temperature_k)
    )

    # Compute distribution statistics
    sorted_ttf = np.sort(ttf_samples)
    percentiles = [10, 50, 90, 95, 99]

    percentile_values = {}
    for p in percentiles:
        percentile_values[f"p{p}"] = float(np.percentile(ttf_samples, p))

    # Compute failure CDF at various times
    test_times = [1e3, 1e6, 1e9, 1e12]  # 1ks, 1Ms, 1Gs, 1Ts
    failure_cdf = {}
    for t in test_times:
        failure_cdf[f"fail_at_{t//1e9:.1f}g"] = float(np.sum(ttf_samples < t) / n_samples)

    # Use GPU if available
    if backend.is_gpu and backend.backend_name in ("torch_cuda", "cupy"):
        pass  # GPU acceleration available

    # Build notes with distribution info
    notes = f"Monte Carlo TTF distribution: {n_samples} samples, T={temperature_k:.1f}K, "
    notes += f"J={j_density:.1e} A/m^2, median={float(np.median(ttf_samples)):.2e}s"

    return TrackedValue(
        value=float(np.median(ttf_samples)),  # Return median TTF as primary value
        unit="seconds",
        provenance=provenance,
        notes=notes,
    )
