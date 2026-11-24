"""SRAM yield & soft error rate models.

Provides SRAM yield estimation using Poisson model, soft error rate
(FIT) calculation, and chip yield using Murphy's model.
"""

from __future__ import annotations

from chiplab.schemas.common import TrackedValue, Provenance
import numpy as np


def sram_yield(
    bitcell_count: int,
    node: str,
    vdd: float,
    temperature: float,
    process_sigma: float,
) -> float:
    """Calculate SRAM yield using Poisson model.

    SRAM yield is modeled using the Poisson distribution for bitcell failures:
        Y = exp(-lambda) where lambda = bitcell_count * failure_rate

    The bitcell failure rate depends on:
    - Process variation (sigma)
    - Supply voltage (Vdd) margin
    - Temperature (higher T -> more failures)
    - Technology node (smaller nodes more sensitive)

    Parameters
    ----------
    bitcell_count : int
        Total number of bitcells in the SRAM array.
    node : str
        Technology node (e.g., "nm" for nm process, e.g., "7", "14", "28").
    vdd : float
        Supply voltage in volts.
    temperature : float
        Operating temperature in Celsius.
    process_sigma : float
        Process variation sigma (standard deviation) in percent.

    Returns
    -------
    float
        SRAM yield as a percentage (0 to 100).
    """
    # Reference Vdd for the technology node (typical values)
    # These are approximate nominal Vdd values per node
    vdd_references = {
        "28": 1.2,
        "14": 0.9,
        "7": 0.8,
        "5": 0.7,
        "nm": 1.0,  # generic
    }

    # Get reference Vdd for the node
    vdd_ref = vdd_references.get(node, 1.0)

    # Failure rate model:
    # lambda = bitcell_count * P(failure per bitcell)
    # P(failure) depends on Vdd margin, temperature, process sigma

    # Vdd margin: how much Vdd is above the minimum required
    # Vdd_margin = Vdd - Vdd_min
    # Vdd_min scales with node and temperature

    # Temperature derating: higher temperature increases failure rate
    # Reference at 25°C, derating factor of ~0.02/°C (empirical)
    temp_ref = 25.0
    temp_derating_factor = (temperature - temp_ref) * 0.02

    # Process sigma impact: larger sigma -> more failures
    # sigma impact is exponential in some models
    sigma_factor = process_sigma / 100.0  # convert to fraction

    # Minimum Vdd at reference conditions
    vdd_min_ref = 0.6  # V at 25°C, reference node

    # Vdd margin in volts
    vdd_margin = vdd - vdd_ref

    # Failure probability per bitcell (simplified model)
    # P_fail ∝ exp(-Vdd_margin / (kT/q)) * (1 + sigma_factor) * temp_derating
    # Using a simplified Arrhenius-like model

    # Thermal voltage kT/q ≈ 0.026 eV at 300K
    # We use a simplified form:
    vdd_effective = vdd_margin + vdd_min_ref

    # Failure rate component from voltage margin
    # Higher margin -> lower failure rate
    if vdd_effective > 0:
        voltage_factor = np.exp(-vdd_effective / 0.1)  # simplified
    else:
        voltage_factor = 1.0

    # Combined failure rate factor
    failure_rate_factor = voltage_factor * (1.0 + sigma_factor) * (1.0 + temp_derating_factor)

    # Lambda = bitcell_count * base_failure_rate * failure_rate_factor
    # Base failure rate per bitcell (very small for functional SRAM)
    base_failure_rate = 1e-6  # per bitcell at ideal conditions

    lambda_params = bitcell_count * base_failure_rate * failure_rate_factor

    # Poisson yield: Y = exp(-lambda)
    yield_fraction = np.exp(-lambda_params)

    # Return as percentage
    yield_pct = float(yield_fraction * 100.0)

    return max(0.0, min(100.0, yield_pct))


def soft_error_rate(
    bitcell_count: int,
    node: str,
    altitude_m: float,
    neutron_flux: float,
) -> float:
    """Calculate soft error rate (FIT) for SRAM.

    FIT (Failures In Time) = failures per billion hours.

    The soft error rate depends on:
    - Bitcell count (more bits -> more errors)
    - Technology node (smaller nodes more sensitive to SEU)
    - Altitude (higher altitude -> more neutron flux)
    - Neutron flux environment

    Parameters
    ----------
    bitcell_count : int
        Total number of bitcells in the SRAM array.
    node : str
        Technology node (e.g., "7", "14", "28").
    altitude_m : float
        Altitude in meters above sea level.
    neutron_flux : float
        Neutron flux in neutrons/cm^2/s at sea level.

    Returns
    -------
    float
        Soft error rate in FIT (Failures In Time, failures per billion hours).
    """
    # Sea level neutron flux reference
    # Typical: ~1 neutron/cm^2/s at sea level, mid-latitudes
    sea_level_neutron_flux = 1.0  # neutrons/cm^2/s

    # Altitude scaling: neutron flux increases with altitude
    # Exponential model: flux ~ exp(altitude / scale_height)
    # Scale height ~ 8000 m for neutron flux
    scale_height = 8000.0  # meters

    # Actual neutron flux at given altitude
    actual_neutron_flux = neutron_flux * np.exp(altitude_m / scale_height)

    # Technology node sensitivity: smaller nodes more susceptible
    # SEU cross-section scales with node size
    # Rough model: cross-section ∝ 1 / node (smaller = more sensitive)
    node_sensitivity = {
        "28": 1.0,
        "14": 2.0,
        "7": 4.0,
        "5": 8.0,
        "nm": 1.0,
    }

    sensitivity = node_sensitivity.get(node, 1.0)

    # FIT calculation:
    # FIT = bitcell_count * neutron_flux * cross_section * conversion_factor
    # Conversion: 1 FIT = 1 failure per 1e9 hours

    # Cross-section factor (arbitrary units, normalized)
    # Typical SEU cross-section for SRAM: ~1e-7 cm^2 at sea level
    # FIT per bitcell = flux * cross_section * conversion

    # FIT per bitcell at sea level, reference node
    fit_per_bitcell_sea_level = 0.1  # FIT per billion hours per bitcell

    # Adjust for node sensitivity and altitude
    fit_per_bitcell = fit_per_bitcell_sea_level * sensitivity * (actual_neutron_flux / sea_level_neutron_flux)

    # Total FIT for the SRAM array
    total_fit = bitcell_count * fit_per_bitcell

    return float(total_fit)


def chip_yield(
    die_area_mm2: float,
    defect_density: float,
    node: str,
) -> float:
    """Calculate chip yield using Murphy's model.

    Murphy's yield model:
        Y = 1 / (1 + A * D)

    Where:
        A = die area
        D = defect density (defects per cm^2)

    This model gives the probability that a die has zero defects.

    Parameters
    ----------
    die_area_mm2 : float
        Die area in square millimeters.
    defect_density : float
        Defect density in defects per cm^2.
    node : str
        Technology node (used for context, affects defect density interpretation).

    Returns
    -------
    float
        Chip yield as a percentage (0 to 100).
    """
    # Convert die area from mm^2 to cm^2
    # 1 mm^2 = 0.1 cm^2  (since 1 cm = 10 mm, 1 cm^2 = 100 mm^2)
    # Actually: 1 cm^2 = 100 mm^2, so 1 mm^2 = 0.01 cm^2
    die_area_cm2 = die_area_mm2 / 100.0

    # Murphy's model: Y = 1 / (1 + A * D)
    # A = die area in cm^2, D = defect density in defects/cm^2

    # Ensure positive values
    die_area_cm2 = max(0.001, die_area_cm2)  # minimum area
    defect_density = max(0.0, defect_density)  # non-negative

    # Murphy's yield equation
    yield_fraction = 1.0 / (1.0 + die_area_cm2 * defect_density)

    # Return as percentage
    yield_pct = float(yield_fraction * 100.0)

    # Cap at reasonable range
    return max(0.0, min(100.0, yield_pct))