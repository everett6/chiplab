"""Ferrofluid material physics (quasi-DC electrical programming, RF readout, electrically induced nanoparticle reconfiguration)."""

from __future__ import annotations

from typing import Any

import json

import numpy as np

from chiplab.schemas.common import Provenance, TrackedValue


class FerrofluidState:
    """State of a ferrofluid material system.

    Tracks magnetization, particle distribution, organization, electrical impedance,
    relaxation time, temperature, and noise characteristics.
    All scalar values use TrackedValue with Provenance tracking.
    Array values (particle_distribution) are stored as JSON strings for TrackedValue compatibility.
    """

    def __init__(
        self,
        magnetization: TrackedValue | float = 0.0,
        particle_distribution: TrackedValue | list | np.ndarray | None = None,
        particle_organization: TrackedValue | float = 0.0,
        electrical_impedance: TrackedValue | float = 100.0,
        relaxation_time: TrackedValue | float = 1e-3,
        temperature: TrackedValue | float = 300.0,
        noise: TrackedValue | float | None = None,
    ):
        """Initialize FerrofluidState with flexible input types (TrackedValue or float)."""
        # Handle magnetization
        if isinstance(magnetization, TrackedValue):
            self.magnetization = magnetization
        else:
            self.magnetization = TrackedValue(
                value=float(magnetization), unit="emu/cm³", provenance=Provenance.ASSUMED
            )

        # Handle particle_distribution
        if particle_distribution is None:
            dist_val = [1.0, 0.0, 0.0]
        elif isinstance(particle_distribution, TrackedValue):
            # Extract value and convert to list if it's a JSON string
            val = particle_distribution.value
            if isinstance(val, str):
                dist_val = json.loads(val)
            elif isinstance(val, (list, np.ndarray)):
                dist_val = list(val)
            else:
                dist_val = [1.0, 0.0, 0.0]
        elif isinstance(particle_distribution, (list, np.ndarray)):
            dist_val = list(particle_distribution)
        else:
            dist_val = [1.0, 0.0, 0.0]
        self.particle_distribution = TrackedValue(
            value=json.dumps(dist_val), unit="normalized", provenance=Provenance.ASSUMED
        )

        # Handle particle_organization
        if isinstance(particle_organization, TrackedValue):
            self.particle_organization = particle_organization
        else:
            self.particle_organization = TrackedValue(
                value=float(particle_organization), unit="order_parameter", provenance=Provenance.ASSUMED
            )

        # Handle electrical_impedance
        if isinstance(electrical_impedance, TrackedValue):
            self.electrical_impedance = electrical_impedance
        else:
            self.electrical_impedance = TrackedValue(
                value=float(electrical_impedance), unit="Ω", provenance=Provenance.ASSUMED
            )

        # Handle relaxation_time
        if isinstance(relaxation_time, TrackedValue):
            self.relaxation_time = relaxation_time
        else:
            self.relaxation_time = TrackedValue(
                value=float(relaxation_time), unit="s", provenance=Provenance.ASSUMED
            )

        # Handle temperature
        if isinstance(temperature, TrackedValue):
            self.temperature = temperature
        else:
            self.temperature = TrackedValue(
                value=float(temperature), unit="K", provenance=Provenance.ASSUMED
            )

        # Handle noise
        if noise is None:
            self.noise = TrackedValue(value=0.0, unit="A²/Hz", provenance=Provenance.ASSUMED)
        elif isinstance(noise, TrackedValue):
            self.noise = noise
        else:
            self.noise = TrackedValue(value=float(noise), unit="A²/Hz", provenance=Provenance.ASSUMED)

    def _get_particle_distribution_np(self) -> np.ndarray:
        """Get particle distribution as numpy array from JSON string."""
        val = json.loads(self.particle_distribution.value)
        return np.array(val)

    def _set_particle_distribution_from_np(self, arr: np.ndarray):
        """Set particle distribution from numpy array, storing as JSON string for TrackedValue."""
        self.particle_distribution = TrackedValue(
            value=json.dumps(arr.tolist()), unit="normalized"
        )

    def copy(self) -> FerrofluidState:
        """Create a deep copy of the state."""
        return FerrofluidState(
            magnetization=TrackedValue(
                value=self.magnetization.value,
                unit=self.magnetization.unit,
                provenance=self.magnetization.provenance,
                source=self.magnetization.source,
                uncertainty_pct=self.magnetization.uncertainty_pct,
                notes=self.magnetization.notes,
            ),
            particle_distribution=TrackedValue(
                value=json.dumps(
                    self._get_particle_distribution_np().tolist()
                )
                if hasattr(self._get_particle_distribution_np(), "tolist")
                else self.particle_distribution.value,
                unit=self.particle_distribution.unit,
                provenance=self.particle_distribution.provenance,
                source=self.particle_distribution.source,
                uncertainty_pct=self.particle_distribution.uncertainty_pct,
                notes=self.particle_distribution.notes,
            ),
            particle_organization=TrackedValue(
                value=self.particle_organization.value,
                unit=self.particle_organization.unit,
                provenance=self.particle_organization.provenance,
                source=self.particle_organization.source,
                uncertainty_pct=self.particle_organization.uncertainty_pct,
                notes=self.particle_organization.notes,
            ),
            electrical_impedance=TrackedValue(
                value=self.electrical_impedance.value,
                unit=self.electrical_impedance.unit,
                provenance=self.electrical_impedance.provenance,
                source=self.electrical_impedance.source,
                uncertainty_pct=self.electrical_impedance.uncertainty_pct,
                notes=self.electrical_impedance.notes,
            ),
            relaxation_time=TrackedValue(
                value=self.relaxation_time.value,
                unit=self.relaxation_time.unit,
                provenance=self.relaxation_time.provenance,
                source=self.relaxation_time.source,
                uncertainty_pct=self.relaxation_time.uncertainty_pct,
                notes=self.relaxation_time.notes,
            ),
            temperature=TrackedValue(
                value=self.temperature.value,
                unit=self.temperature.unit,
                provenance=self.temperature.provenance,
                source=self.temperature.source,
                uncertainty_pct=self.temperature.uncertainty_pct,
                notes=self.temperature.notes,
            ),
            noise=TrackedValue(
                value=self.noise.value,
                unit=self.noise.unit,
                provenance=self.noise.provenance,
                source=self.noise.source,
                uncertainty_pct=self.noise.uncertainty_pct,
                notes=self.noise.notes,
            )
            if self.noise
            else None,
        )


def ferrofluid_write(
    state: FerrofluidState,
    voltage: float | TrackedValue,
    duration: float | TrackedValue,
    frequency: float | TrackedValue = 0.0,
) -> FerrofluidState:
    """Electrically program the ferrofluid state with a quasi-DC voltage pulse.

    Applies a voltage pulse to the ferrofluid, causing nanoparticle reconfiguration
    and changes in magnetization and impedance.

    Parameters
    ----------
    state : FerrofluidState
        Current ferrofluid state.
    voltage : float or TrackedValue
        Applied voltage in volts. If TrackedValue, provenance is recorded.
    duration : float or TrackedValue
        Pulse duration in seconds.
    frequency : float or TrackedValue, optional
        Frequency of the voltage pulse in Hz (default 0.0 for quasi-DC).

    Returns
    -------
    FerrofluidState
        New ferrofluid state after electrical programming.
    """
    # Unwrap TrackedValues
    v = float(voltage) if not isinstance(voltage, TrackedValue) else voltage.value
    d = float(duration) if not isinstance(duration, TrackedValue) else duration.value
    f = float(frequency) if not isinstance(frequency, TrackedValue) else frequency.value

    # Create new state based on programming
    new_state = state.copy()

    # Magnetization change proportional to voltage and duration
    # M ~ χ * H, where H = V / (gap_distance) - simplified model
    magnetization_change = v * d * 1e-6  # scaled factor
    new_mag_value = new_state.magnetization.value + magnetization_change
    new_state.magnetization = TrackedValue(
        value=new_mag_value,
        unit=new_state.magnetization.unit,
        provenance=new_state.magnetization.provenance,
        source="ferrofluid_write",
    )

    # Particle organization changes with electrical programming
    org_change = abs(v) * d * 1e-3
    new_org_value = min(1.0, max(-1.0, new_state.particle_organization.value + org_change))
    new_state.particle_organization = TrackedValue(
        value=new_org_value,
        unit=new_state.particle_organization.unit,
        provenance=new_state.particle_organization.provenance,
        source="ferrofluid_write",
    )

    # Electrical impedance decreases with programming (more aligned particles)
    # Z ~ Z0 / (1 + α * org)
    imp_factor = 1.0 + 0.5 * abs(new_state.particle_organization.value)
    new_imp_value = max(1.0, new_state.electrical_impedance.value / imp_factor)
    new_state.electrical_impedance = TrackedValue(
        value=new_imp_value,
        unit=new_state.electrical_impedance.unit,
        provenance=new_state.electrical_impedance.provenance,
        source="ferrofluid_write",
    )

    # Relaxation time increases with organization (more stable state)
    new_rt_value = new_state.relaxation_time.value * (1.0 + 0.1 * abs(new_state.particle_organization.value))
    new_state.relaxation_time = TrackedValue(
        value=new_rt_value,
        unit=new_state.relaxation_time.unit,
        provenance=new_state.relaxation_time.provenance,
        source="ferrofluid_write",
    )

    # Update temperature slightly due to Joule heating
    temp_increase = abs(v) * d * 0.1  # simplified heating
    new_temp_value = new_state.temperature.value + temp_increase
    new_state.temperature = TrackedValue(
        value=new_temp_value,
        unit=new_state.temperature.unit,
        provenance=new_state.temperature.provenance,
        source="ferrofluid_write",
    )

    # Noise is reset/updated after programming
    new_state.noise = TrackedValue(
        value=abs(v) * d * 0.01,
        unit=new_state.noise.unit,
        provenance=Provenance.ESTIMATED,
        source="ferrofluid_write",
    )

    # Update particle distribution based on organization change
    # Simple model: reorient distribution towards alignment with field
    org = abs(new_state.particle_organization.value)
    current_dist = new_state._get_particle_distribution_np()
    # Add slight alignment bias along z-axis
    alignment_bias = org * 0.1
    new_dist = current_dist.copy()
    new_dist[2] = new_dist[2] + alignment_bias  # boost z-component
    # Normalize
    norm = np.linalg.norm(new_dist)
    new_dist = new_dist / norm if norm > 0 else new_dist
    new_state._set_particle_distribution_from_np(new_dist)

    return new_state


def ferrofluid_read(
    state: FerrofluidState,
    frequency: float | TrackedValue = 1e6,
    amplitude: float = 1e-3,
) -> dict[str, Any]:
    """RF readout of ferrofluid impedance response.

    Measures the complex impedance response of the ferrofluid at a given frequency
    and amplitude, reflecting the current particle organization state.

    Parameters
    ----------
    state : FerrofluidState
        Current ferrofluid state.
    frequency : float or TrackedValue, optional
        Readout frequency in Hz (default 1 MHz).
    amplitude : float, optional
        Readout amplitude in volts (default 1 mV).

    Returns
    -------
    dict
        Impedance matrix response with real/imaginary parts and phase.
    """
    # Unwrap frequency if TrackedValue
    f = float(frequency) if not isinstance(frequency, TrackedValue) else frequency.value
    a = float(amplitude)

    # Impedance depends on particle organization (order parameter)
    org = abs(state.particle_organization.value)

    # Complex impedance: Z = R + jX, where R and X depend on frequency and organization
    # Relaxation time determines the characteristic frequency
    tau = state.relaxation_time.value

    # Debye-like relaxation model
    omega = 2.0 * np.pi * f
    denom = 1.0 + (omega * tau) ** 2

    # Real part (resistive)
    real_part = state.electrical_impedance.value * (1.0 / denom)

    # Imaginary part (reactive)
    imag_part = -state.electrical_impedance.value * (omega * tau / denom)

    # Magnitude
    magnitude = np.sqrt(real_part ** 2 + imag_part ** 2)

    # Phase
    phase = np.arctan2(imag_part, real_part)

    # Amplitude scaling
    response_amplitude = magnitude * (1.0 + 0.5 * org)  # organization enhances response

    return {
        "frequency_hz": f,
        "amplitude_v": a,
        "impedance_real_Ω": float(real_part),
        "impedance_imag_Ω": float(imag_part),
        "impedance_magnitude_Ω": float(magnitude),
        "phase_rad": float(phase),
        "response_amplitude": float(response_amplitude),
        "organization": org,
        "relaxation_time_s": tau,
    }


def ferrofluid_relax(
    state: FerrofluidState,
    dt: float | TrackedValue,
) -> FerrofluidState:
    """Time evolution of ferrofluid state (relaxation towards equilibrium).

    Models the natural relaxation of magnetization and particle organization
    towards thermal equilibrium over time step dt.

    Parameters
    ----------
    state : FerrofluidState
        Current ferrofluid state.
    dt : float or TrackedValue
        Time step in seconds.

    Returns
    -------
    FerrofluidState
        Updated ferrofluid state after relaxation.
    """
    dt_val = float(dt) if not isinstance(dt, TrackedValue) else dt.value

    new_state = state.copy()

    # Magnetization relaxation towards zero (or equilibrium)
    # Exponential decay with relaxation time
    decay_factor = np.exp(-dt_val / max(state.relaxation_time.value, 1e-12))
    new_mag_value = new_state.magnetization.value * decay_factor
    new_state.magnetization = TrackedValue(
        value=new_mag_value,
        unit=new_state.magnetization.unit,
        provenance=new_state.magnetization.provenance,
        source="ferrofluid_relax",
    )

    # Organization relaxes towards zero (disordering)
    org_decay = np.exp(-dt_val / max(state.relaxation_time.value, 1e-12))
    new_org_value = new_state.particle_organization.value * org_decay
    new_state.particle_organization = TrackedValue(
        value=new_org_value,
        unit=new_state.particle_organization.unit,
        provenance=new_state.particle_organization.provenance,
        source="ferrofluid_relax",
    )

    # Impedance increases as state relaxes (less organized)
    imp_factor = 1.0 + 0.3 * (1.0 - org_decay)
    new_imp_value = new_state.electrical_impedance.value * imp_factor
    new_state.electrical_impedance = TrackedValue(
        value=new_imp_value,
        unit=new_state.electrical_impedance.unit,
        provenance=new_state.electrical_impedance.provenance,
        source="ferrofluid_relax",
    )

    # Temperature equilibrates slightly towards ambient
    ambient = 300.0
    temp_frac = 1.0 - np.exp(-dt_val / 100.0)  # 100s thermal time constant
    new_temp_value = new_state.temperature.value * (1.0 - temp_frac) + ambient * temp_frac
    new_state.temperature = TrackedValue(
        value=new_temp_value,
        unit=new_state.temperature.unit,
        provenance=new_state.temperature.provenance,
        source="ferrofluid_relax",
    )

    # Noise increases slightly with relaxation (fluctuations)
    new_noise_value = abs(new_state.noise.value) * 1.001 if new_state.noise else 0.0
    new_state.noise = TrackedValue(
        value=new_noise_value,
        unit=new_state.noise.unit if new_state.noise else "A²/Hz",
        provenance=Provenance.ESTIMATED,
        source="ferrofluid_relax",
    )

    # Update particle distribution during relaxation (becomes more isotropic)
    org = abs(new_state.particle_organization.value)
    current_dist = new_state._get_particle_distribution_np()
    # During relaxation, distribution becomes more uniform/spherical
    # Move towards [0, 0, 1] normalization factor based on remaining organization
    target_dist = np.array([0.0, 0.0, 1.0])  # align with z-axis
    new_dist = (1.0 - org) * target_dist + org * current_dist
    norm = np.maximum(np.linalg.norm(new_dist), 1e-12)
    new_dist = new_dist / norm
    new_state._set_particle_distribution_from_np(new_dist)

    return new_state


def ferrofluid_thermal(
    state: FerrofluidState,
    temperature: float | TrackedValue,
) -> FerrofluidState:
    """Thermal dependence of ferrofluid state.

    Updates the state's temperature and adjusts magnetic and electrical properties
    according to the Neel-Brown relaxation and temperature-dependent susceptibility.

    Parameters
    ----------
    state : FerrofluidState
        Current ferrofluid state.
    temperature : float or TrackedValue
        New temperature in Kelvin.

    Returns
    -------
    FerrofluidState
        Updated ferrofluid state with new temperature.
    """
    t = state.temperature.value
    if isinstance(temperature, TrackedValue):
        t_new = temperature.value
        t_prov = temperature.provenance
    else:
        t_new = float(temperature)
        t_prov = Provenance.MEASURED

    new_state = state.copy()

    # Update temperature
    new_state.temperature = TrackedValue(
        value=t_new,
        unit="K",
        provenance=t_prov,
        source="ferrofluid_thermal",
    )

    # Susceptibility decreases with temperature (Curie-Weiss behavior)
    # χ ~ C / (T - T_curie) with T_curie ~ 273K for typical ferrofluids
    T_curie = 273.0
    org_factor = max(0.01, T_curie / max(t_new, 1.0))  # simplified
    new_org = min(1.0, max(-1.0, org_factor))
    new_state.particle_organization = TrackedValue(
        value=new_org,
        unit=new_state.particle_organization.unit,
        provenance=t_prov,
        source="ferrofluid_thermal",
    )

    # Electrical impedance increases with temperature (less magnetization)
    imp_factor = t_new / max(t, 1.0) if t > 0 else 1.0
    new_imp = state.electrical_impedance.value * imp_factor
    new_state.electrical_impedance = TrackedValue(
        value=new_imp,
        unit=state.electrical_impedance.unit,
        provenance=t_prov,
        source="ferrofluid_thermal",
    )

    # Relaxation time decreases with temperature (faster relaxation)
    # τ ~ τ0 * exp(KV/kT) - Arrhenius behavior
    tau_factor = np.exp(100.0 / T_curie * (T_curie / max(t_new, 1.0) - 1.0))
    new_tau = state.relaxation_time.value / tau_factor
    new_state.relaxation_time = TrackedValue(
        value=new_tau,
        unit=state.relaxation_time.unit,
        provenance=t_prov,
        source="ferrofluid_thermal",
    )

    # Noise increases with temperature (thermal fluctuations)
    new_noise = abs(state.noise.value) * (t_new / max(t, 1.0)) if state.noise else 0.0
    new_state.noise = TrackedValue(
        value=new_noise,
        unit=state.noise.unit if state.noise else "A²/Hz",
        provenance=t_prov,
        source="ferrofluid_thermal",
    )

    # Update particle distribution with temperature change (thermal disordering)
    # Higher temperature -> more random distribution
    temp_ratio = t_new / max(t, 1.0)
    current_dist = new_state._get_particle_distribution_np()
    # Bias towards more random distribution at higher temperature
    import random
    random.seed(int(t_new * 1000) % 2**31)
    random_bias = np.random.normal(0, 0.01 * (temp_ratio - 1.0) if temp_ratio > 1.0 else 0.0, 3)
    new_dist = current_dist + random_bias
    # Keep normalized
    norm = np.maximum(np.linalg.norm(new_dist), 1e-12)
    new_dist = new_dist / norm
    new_state._set_particle_distribution_from_np(new_dist)

    return new_state


def ferrofluid_noise(
    state: FerrofluidState,
) -> dict[str, Any]:
    """Random physical fluctuations in ferrofluid state.

    Generates random physical fluctuations consistent with the fluctuation-dissipation
    theorem, affecting magnetization, organization, and impedance.

    Parameters
    ----------
    state : FerrofluidState
        Current ferrofluid state.

    Returns
    -------
    dict
        Dictionary of fluctuation magnitudes by physical quantity.
    """
    T = state.temperature.value
    eta = 1.0  # damping factor

    # Fluctuation magnitudes from fluctuation-dissipation theorem
    # ΔM ~ sqrt(2*kB*T*Δf/χ) - simplified
    kB = 1.38e-23  # Boltzmann constant (J/K)
    # Use effective susceptibility proportional to organization
    chi_eff = abs(state.particle_organization.value) + 0.1

    # Magnetization fluctuation
    mag_fluct = np.sqrt(2 * kB * T / chi_eff) if chi_eff > 0 else 0.0

    # Organization fluctuation
    org_fluct = np.sqrt(2 * kB * T * eta / (state.relaxation_time.value + 1e-12))

    # Impedance fluctuation
    imp_fluct = state.electrical_impedance.value * np.sqrt(2 * kB * T * eta / (state.electrical_impedance.value + 1e-12))

    # Apply random fluctuations
    new_state = state.copy()
    if new_state.noise is None:
        new_state.noise = TrackedValue(value=0.0, unit="A²/Hz", provenance=Provenance.ASSUMED)

    new_noise_val = abs(new_state.noise.value) + mag_fluct + org_fluct + imp_fluct
    new_state.noise = TrackedValue(
        value=new_noise_val,
        unit=new_state.noise.unit,
        provenance=Provenance.MEASURED,
        source="ferrofluid_noise",
    )

    return {
        "magnetization_fluctuation_emu_cm3": float(mag_fluct),
        "organization_fluctuation": float(org_fluct),
        "impedance_fluctuation_Ω": float(imp_fluct),
        "temperature_K": T,
        "correlation_time_s": 1.0 / max(state.relaxation_time.value, 1e-3),
    }