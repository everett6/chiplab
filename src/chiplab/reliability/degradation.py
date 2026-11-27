"""Degradation models for ChipLab reliability.

State drift, cycling endurance degradation (PCM/memristor/ferrofluid/SRAM),
electromigration (Black's equation), NBTI/PBTI/HCI aging models.
All returning TrackedValue with Provenance tracking.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from chiplab.schemas.common import Provenance, TrackedValue
from chiplab.core.gpu import GPUBackend


def state_drift(
    initial_value: float,
    drift_rate: float = 1e-9,
    time_elapsed: float = 1.0,
    provenance: Provenance = Provenance.ESTIMATED,
) -> TrackedValue:
    """Model state drift over time.

    Parameters
    ----------
    initial_value : float
        Initial value before drift.
    drift_rate : float, optional
        Drift rate per second (default 1e-9).
    time_elapsed : float, optional
        Time elapsed in seconds (default 1.0).
    provenance : Provenance, optional
        Provenance of the measurement (default ESTIMATED).

    Returns
    -------
    TrackedValue
        Tracked value with drift applied and provenance set.
    """
    drifted = initial_value + drift_rate * time_elapsed
    return TrackedValue(
        value=drifted,
        provenance=provenance,
        notes="State drift model",
    )


def pcms_endurance(
    cycle_count: int,
    endurance_limit: int = 10**6,
    degradation_per_cycle: float = 1e-9,
    provenance: Provenance = Provenance.ESTIMATED,
) -> TrackedValue:
    """Model PCM (Phase-Change Memory) endurance degradation.

    Parameters
    ----------
    cycle_count : int
        Number of write cycles.
    endurance_limit : int, optional
        Endurance limit in cycles (default 1e6 for typical PCM).
    degradation_per_cycle : float, optional
        Degradation per write cycle (default 1e-9).

    Returns
    -------
    TrackedValue
        Tracked value with PCM degradation and provenance set.
    """
    degradation = degradation_per_cycle * cycle_count
    remaining_endurance = endurance_limit * (1 - degradation)
    # Clamp to reasonable bounds
    remaining_endurance = max(0, min(remaining_endurance, endurance_limit))
    return TrackedValue(
        value=remaining_endurance,
        unit="cycles",
        provenance=provenance,
        notes=f"PCM endurance after {cycle_count} cycles",
    )


def memristor_cycling_endurance(
    cycle_count: int,
    endurance_limit: int = 10**9,
    degradation_per_cycle: float = 1e-12,
    provenance: Provenance = Provenance.ESTIMATED,
) -> TrackedValue:
    """Model memristor cycling endurance degradation.

    Parameters
    ----------
    cycle_count : int
        Number of switching cycles.
    endurance_limit : int, optional
        Endurance limit in cycles (default 1e9 for typical memristor).
    degradation_per_cycle : float, optional
        Degradation per switching cycle (default 1e-12).

    Returns
    -------
    TrackedValue
        Tracked value with memristor degradation and provenance set.
    """
    degradation = degradation_per_cycle * cycle_count
    remaining_endurance = endurance_limit * (1 - degradation)
    remaining_endurance = max(0, min(remaining_endurance, endurance_limit))
    return TrackedValue(
        value=remaining_endurance,
        unit="cycles",
        provenance=provenance,
        notes=f"Memristor endurance after {cycle_count} cycles",
    )


def ferrofluid_retention(
    time_elapsed: float,
    retention_half_life: float = 3600.0,
    provenance: Provenance = Provenance.ESTIMATED,
) -> TrackedValue:
    """Model ferrofluid retention/degradation over time.

    Parameters
    ----------
    time_elapsed : float
        Time elapsed in seconds.
    retention_half_life : float, optional
        Retention half-life in seconds (default 3600s = 1 hour).
    provenance : Provenance, optional
        Provenance of the measurement (default ESTIMATED).

    Returns
    -------
    TrackedValue
        Tracked value with ferrofluid retention and provenance set.
    """
    # Exponential decay model for retention
    retention = np.exp(-np.log(2) * time_elapsed / retention_half_life)
    return TrackedValue(
        value=retention,
        unit="fraction",
        provenance=provenance,
        notes=f"Ferrofluid retention after {time_elapsed:.1f}s. HEURISTIC: the half-life is "
        "an unsupported input; see docs/ferrofluid_model.md",
    )


def sram_retention(
    temperature_k: float,
    time_elapsed: float,
    alpha: float = 1e-12,
    provenance: Provenance = Provenance.ESTIMATED,
) -> TrackedValue:
    """Model SRAM retention degradation.

    SRAM cell retention depends on temperature and time.

    Parameters
    ----------
    temperature_k : float
        Temperature in Kelvin.
    time_elapsed : float
        Time elapsed in seconds.
    alpha : float, optional
        Technology-dependent retention coefficient (default 1e-12).
    provenance : Provenance, optional
        Provenance of the measurement (default ESTIMATED).

    Returns
    -------
    TrackedValue
        Tracked value with SRAM retention and provenance set.
    """
    # Arrhenius-like retention model
    retention = np.exp(-alpha * np.exp(-6000 / temperature_k) * time_elapsed)
    return TrackedValue(
        value=retention,
        unit="fraction",
        provenance=provenance,
        notes=f"SRAM retention at {temperature_k:.1f}K for {time_elapsed:.1f}s",
    )


def blacks_equation(
    current: float,
    time_elapsed: float,
    activation_energy: float = 0.7,
    area: float = 1e-12,
    constant_a: float = 1e-9,
    constant_b: float = 2.0,
    provenance: Provenance = Provenance.ESTIMATED,
) -> TrackedValue:
    """Model electromigration using Black's equation.

    Black's equation: MTTF = A * J^-n * exp(Ea/(k*T))
    where J is current density.

    Parameters
    ----------
    current : float
        Current in Amperes.
    time_elapsed : float
        Time elapsed in seconds.
    activation_energy : float, optional
        Activation energy in eV (default 0.7 eV).
    area : float, optional
        Cross-sectional area in m^2 (default 1e-12 m^2).
    constant_a : float, optional
        Technology constant A (default 1e-9).
    constant_b : float, optional
        Current density exponent n (default 2.0).
    provenance : Provenance, optional
        Provenance of the measurement (default ESTIMATED).

    Returns
    -------
    TrackedValue
        Tracked value with electromigration lifetime and provenance set.
    """
    k_boltzmann = 8.617333262145e-5  # eV/K
    temperature = 300.0  # Assume room temperature

    # Current density J = I / A
    current_density = current / area if area > 0 else 0

    # Black's equation for MTTF
    mttf = constant_a * (current_density ** (-constant_b)) * np.exp(activation_energy / (k_boltzmann * temperature))

    # Degradation fraction after time_elapsed
    degradation = time_elapsed / mttf if mttf > 0 else 1.0
    degradation = min(degradation, 1.0)

    return TrackedValue(
        value=degradation,
        unit="fraction",
        provenance=provenance,
        notes=f"Electromigration Black's equation: MTTF={mttf:.1f}s, degradation={degradation:.4f}",
    )


def nbti_aging(
    stress_time: float,
    temperature_k: float = 300.0,
    voltage: float = 1.0,
    provenance: Provenance = Provenance.ESTIMATED,
) -> TrackedValue:
    """Model NBTI (Negative Bias Temperature Instability) aging.

    Parameters
    ----------
    stress_time : float
        Stress time in seconds.
    temperature_k : float, optional
        Temperature in Kelvin (default 300K).
    voltage : float, optional
        Gate voltage (default 1.0V).
    provenance : Provenance, optional
        Provenance of the measurement (default ESTIMATED).

    Returns
    -------
    TrackedValue
        Tracked value with NBTI threshold voltage shift and provenance set.
    """
    # NBTI model: Vth shift proportional to t_stress^n * exp(Ea/(k*T))
    k_boltzmann = 8.617333262145e-5
    Ea = 0.05  # eV, NBTI activation energy
    n = 0.16  # NBTI time exponent

    vth_shift = voltage ** 2 * (stress_time ** n) * np.exp(Ea / (k_boltzmann * temperature_k))
    return TrackedValue(
        value=vth_shift,
        unit="V",
        provenance=provenance,
        notes=f"NBTI Vth shift after {stress_time:.1f}s stress at {temperature_k:.1f}K",
    )


def pbti_aging(
    stress_time: float,
    temperature_k: float = 300.0,
    voltage: float = 1.0,
    provenance: Provenance = Provenance.ESTIMATED,
) -> TrackedValue:
    """Model PBTI (Positive Bias Temperature Instability) aging.

    Parameters
    ----------
    stress_time : float
        Stress time in seconds.
    temperature_k : float, optional
        Temperature in Kelvin (default 300K).
    voltage : float, optional
        Gate voltage (default 1.0V).
    provenance : Provenance, optional
        Provenance of the measurement (default ESTIMATED).

    Returns
    -------
    TrackedValue
        Tracked value with PBTI threshold voltage shift and provenance set.
    """
    # PBTI model: similar to NBTI but for PMOS
    k_boltzmann = 8.617333262145e-5
    Ea = 0.1  # eV, PBTI activation energy
    n = 0.08  # PBTI time exponent

    vth_shift = voltage ** 1.5 * (stress_time ** n) * np.exp(Ea / (k_boltzmann * temperature_k))
    return TrackedValue(
        value=vth_shift,
        unit="V",
        provenance=provenance,
        notes=f"PBTI Vth shift after {stress_time:.1f}s stress at {temperature_k:.1f}K",
    )


def hci_aging(
    drain_voltage: float,
    gate_voltage: float,
    temperature_k: float = 300.0,
    time_elapsed: float = 1.0,
    provenance: Provenance = Provenance.ESTIMATED,
) -> TrackedValue:
    """Model HCI (Hot Carrier Injection) aging.

    Parameters
    ----------
    drain_voltage : float
        Drain voltage in Volts.
    gate_voltage : float
        Gate voltage in Volts.
    temperature_k : float, optional
        Temperature in Kelvin (default 300K).
    time_elapsed : float, optional
        Time elapsed in seconds (default 1.0).
    provenance : Provenance, optional
        Provenance of the measurement (default ESTIMATED).

    Returns
    -------
    TrackedValue
        Tracked value with HCI degradation and provenance set.
    """
    # HCI model: degradation proportional to impact ionization rate
    k_boltzmann = 8.617333262145e-5
    Ea = 0.3  # eV, HCI activation energy

    # Simplified HCI model
    stress_factor = drain_voltage * gate_voltage
    hci_degradation = stress_factor * np.exp(-Ea / (k_boltzmann * temperature_k)) * time_elapsed

    return TrackedValue(
        value=hci_degradation,
        unit="arbitrary",
        provenance=provenance,
        notes=f"HCI degradation after {time_elapsed:.1f}s at {temperature_k:.1f}K",
    )
