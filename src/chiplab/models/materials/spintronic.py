"""Spintronic (STT/MTJ-MRAM) material physics models.

Spin-transfer torque, spin-orbit torque, TMR ratio, and thermal stability
modeling for MRAM and spintronic devices.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chiplab.schemas.common import Provenance, TrackedValue


class MRAMState:
    """State of a magnetic tunnel junction (MTJ) MRAM cell.

    Tracks magnetization direction, resistance, and thermal parameters.
    All scalar values use TrackedValue with Provenance tracking.
    """

    def __init__(
        self,
        magnetization: TrackedValue | float | np.ndarray = 1.0,
        resistance: TrackedValue | float = 10000.0,
        parallel_resistance: TrackedValue | float = 10000.0,
        antiparallel_resistance: TrackedValue | float = 15000.0,
        tunnel_barrier_thickness: TrackedValue | float = 1.0,
        area: TrackedValue | float = 1e-12,
        temperature: TrackedValue | float = 300.0,
        thermal_noise: TrackedValue | float | None = None,
    ):
        """Initialize MRAMState with flexible input types (TrackedValue or float).

        Parameters
        ----------
        magnetization : TrackedValue, float, or np.ndarray
            Magnetization direction as a unit vector (3,) or scalar factor.
            Default 1.0 (parallel state).
        resistance : TrackedValue or float
            Current resistance in Ω. Default 10kΩ.
        parallel_resistance : TrackedValue or float
            Resistance in parallel (P) state in Ω. Default 10kΩ.
        antiparallel_resistance : TrackedValue or float
            Resistance in antiparallel (AP) state in Ω. Default 15kΩ.
        tunnel_barrier_thickness : TrackedValue or float
            Tunnel barrier thickness in nm. Default 1.0.
        area : TrackedValue or float
            MTJ area in m². Default 1e-12 (0.01 μm²).
        temperature : TrackedValue or float
            Temperature in K. Default 300.0.
        thermal_noise : TrackedValue or float or None
            Thermal noise in Ω/√Hz. Default None (auto-initialized).
        """
        # Handle magnetization
        if isinstance(magnetization, TrackedValue):
            self.magnetization = magnetization
        elif isinstance(magnetization, np.ndarray):
            self.magnetization = TrackedValue(
                value=magnetization,
                provenance=Provenance.ASSUMED,
            )
        else:
            # Scalar: 1.0 = parallel, -1.0 = antiparallel
            self.magnetization = TrackedValue(
                value=float(magnetization),
                provenance=Provenance.ASSUMED,
            )

        # Handle resistance
        if isinstance(resistance, TrackedValue):
            self.resistance = resistance
        else:
            self.resistance = TrackedValue(
                value=float(resistance),
                unit="Ω",
                provenance=Provenance.ASSUMED,
            )

        # Handle parallel_resistance
        if isinstance(parallel_resistance, TrackedValue):
            self.parallel_resistance = parallel_resistance
        else:
            self.parallel_resistance = TrackedValue(
                value=float(parallel_resistance),
                unit="Ω",
                provenance=Provenance.ASSUMED,
            )

        # Handle antiparallel_resistance
        if isinstance(antiparallel_resistance, TrackedValue):
            self.antiparallel_resistance = antiparallel_resistance
        else:
            self.antiparallel_resistance = TrackedValue(
                value=float(antiparallel_resistance),
                unit="Ω",
                provenance=Provenance.ASSUMED,
            )

        # Handle tunnel_barrier_thickness
        if isinstance(tunnel_barrier_thickness, TrackedValue):
            self.tunnel_barrier_thickness = tunnel_barrier_thickness
        else:
            self.tunnel_barrier_thickness = TrackedValue(
                value=float(tunnel_barrier_thickness),
                unit="nm",
                provenance=Provenance.ASSUMED,
            )

        # Handle area
        if isinstance(area, TrackedValue):
            self.area = area
        else:
            self.area = TrackedValue(
                value=float(area),
                unit="m²",
                provenance=Provenance.ASSUMED,
            )

        # Handle temperature
        if isinstance(temperature, TrackedValue):
            self.temperature = temperature
        else:
            self.temperature = TrackedValue(
                value=float(temperature),
                unit="K",
                provenance=Provenance.ASSUMED,
            )

        # Handle thermal_noise
        if thermal_noise is None:
            self.thermal_noise = TrackedValue(value=0.0, unit="Ω/√Hz", provenance=Provenance.ASSUMED)
        elif isinstance(thermal_noise, TrackedValue):
            self.thermal_noise = thermal_noise
        else:
            self.thermal_noise = TrackedValue(value=float(thermal_noise), unit="Ω/√Hz", provenance=Provenance.ASSUMED)

    def copy(self) -> MRAMState:
        """Create a deep copy of the state.

        Returns
        -------
        MRAMState
            New MRAMState with all values copied.
        """
        return MRAMState(
            magnetization=TrackedValue(
                value=self.magnetization.value,
                unit=getattr(self.magnetization, "unit", None),
                provenance=self.magnetization.provenance,
                source=self.magnetization.source,
                uncertainty_pct=self.magnetization.uncertainty_pct,
                notes=self.magnetization.notes,
            )
            if hasattr(self.magnetization, "value")
            else TrackedValue(value=self.magnetization),
            resistance=TrackedValue(
                value=self.resistance.value,
                unit=self.resistance.unit,
                provenance=self.resistance.provenance,
                source=self.resistance.source,
                uncertainty_pct=self.resistance.uncertainty_pct,
                notes=self.resistance.notes,
            ),
            parallel_resistance=TrackedValue(
                value=self.parallel_resistance.value,
                unit=self.parallel_resistance.unit,
                provenance=self.parallel_resistance.provenance,
                source=self.parallel_resistance.source,
                uncertainty_pct=self.parallel_resistance.uncertainty_pct,
                notes=self.parallel_resistance.notes,
            ),
            antiparallel_resistance=TrackedValue(
                value=self.antiparallel_resistance.value,
                unit=self.antiparallel_resistance.unit,
                provenance=self.antiparallel_resistance.provenance,
                source=self.antiparallel_resistance.source,
                uncertainty_pct=self.antiparallel_resistance.uncertainty_pct,
                notes=self.antiparallel_resistance.notes,
            ),
            tunnel_barrier_thickness=TrackedValue(
                value=self.tunnel_barrier_thickness.value,
                unit=self.tunnel_barrier_thickness.unit,
                provenance=self.tunnel_barrier_thickness.provenance,
                source=self.tunnel_barrier_thickness.source,
                uncertainty_pct=self.tunnel_barrier_thickness.uncertainty_pct,
                notes=self.tunnel_barrier_thickness.notes,
            ),
            area=TrackedValue(
                value=self.area.value,
                unit=self.area.unit,
                provenance=self.area.provenance,
                source=self.area.source,
                uncertainty_pct=self.area.uncertainty_pct,
                notes=self.area.notes,
            ),
            temperature=TrackedValue(
                value=self.temperature.value,
                unit=self.temperature.unit,
                provenance=self.temperature.provenance,
                source=self.temperature.source,
                uncertainty_pct=self.temperature.uncertainty_pct,
                notes=self.temperature.notes,
            ),
            thermal_noise=TrackedValue(
                value=self.thermal_noise.value,
                unit=self.thermal_noise.unit,
                provenance=self.thermal_noise.provenance,
                source=self.thermal_noise.source,
                uncertainty_pct=self.thermal_noise.uncertainty_pct,
                notes=self.thermal_noise.notes,
            ),
        )


def stt_switching_step(
    state: MRAMState,
    current_density: float | TrackedValue,
    pulse_duration: float | TrackedValue = 1e-9,
) -> MRAMState:
    """Perform a single spin-transfer torque (STT) switching step.

    Models the probability and dynamics of magnetization switching in an
    STT-MRAM cell under a write current. Uses the Slonczewski model for
    STT-induced switching.

    The switching probability per pulse is modeled as:

        P_switch = 1 / (1 + exp(-α * (J - Jc) / Jc))

    where J is the current density, Jc is the critical current density,
    and α is a steepness factor.

    Parameters
    ----------
    state : MRAMState
        Current MRAM state.
    current_density : float or TrackedValue
        Write current density in A/m². If TrackedValue, provenance is recorded.
    pulse_duration : float or TrackedValue
        Current pulse duration in seconds. Default 1 ns. If TrackedValue,
        provenance is recorded.

    Returns
    -------
    MRAMState
        Updated MRAM state after the STT switching step. If switching
        occurs, the magnetization direction is flipped.
    """
    # Unwrap TrackedValues
    j = float(current_density) if not isinstance(current_density, TrackedValue) else current_density.value
    dur = float(pulse_duration) if not isinstance(pulse_duration, TrackedValue) else pulse_duration.value

    new_state = state.copy()

    # Extract magnetization direction
    m = float(new_state.magnetization.value) if not isinstance(new_state.magnetization.value, np.ndarray) else float(np.mean(new_state.magnetization.value))

    # Critical current density for STT switching (Slonczewski model)
    # Jc = (2e * α * Ms * t * Hk) / (ħ * η * A)
    # Simplified: Jc depends on material parameters and device geometry

    # Material parameters (typical for CoFeB/MgO MTJ)
    ms = 1.0e6  # saturation magnetization A/m
    hk = 5.0e4  # anisotropy field A/m
    t = float(new_state.tunnel_barrier_thickness.value)  # barrier thickness nm
    eta = 0.6  # spin polarization efficiency
    a = 1.0  # damping parameter

    # Critical current density A/m²
    # Jc = (2 * e * α * Ms * t_nm * 1e-9 * Hk) / (ħ * η) * (1/A)
    # Using simplified form with area
    hbar = 1.054571817e-34  # J·s
    e = 1.602176634e-19  # C

    # Jc formula: Jc = (2 * e * α * Ms * t) / (ħ * η) where t is in meters
    t_m = t * 1e-9  # nm to meters
    jc = (2.0 * e * a * ms * t_m) / (hbar * eta)  # A/m²

    # Switching probability
    if jc > 0:
        prob = 1.0 / (1.0 + np.exp(-a * (j - jc) / max(jc, 1e-10)))
    else:
        prob = 0.0

    # Stochastic switching: generate random number
    import random
    random.seed(int(float(new_state.temperature.value) * 1000) % 2**31)
    r = random.random()

    # Apply switching if random number < probability
    if r < prob:
        # Switch magnetization direction
        new_m = -m
        new_state.magnetization = TrackedValue(
            value=new_m,
            provenance=new_state.magnetization.provenance,
            source="stt_switching_step",
        )

        # Resistance changes from P to AP or AP to P
        rp = float(new_state.parallel_resistance.value)
        rap = float(new_state.antiparallel_resistance.value)
        if m > 0:  # was parallel, now antiparallel
            new_state.resistance = TrackedValue(
                value=rap,
                unit="Ω",
                provenance=new_state.resistance.provenance,
                source="stt_switching_step",
            )
        else:  # was antiparallel, now parallel
            new_state.resistance = TrackedValue(
                value=rp,
                unit="Ω",
                provenance=new_state.resistance.provenance,
                source="stt_switching_step",
            )
    else:
        # No switching, resistance stays the same
        # Small thermal fluctuation in resistance
        thermal_fluct = abs(float(new_state.thermal_noise.value)) if new_state.thermal_noise else 0.0
        new_r = float(new_state.resistance.value) + np.random.normal(0, thermal_fluct * 0.1) if thermal_fluct > 0 else float(new_state.resistance.value)
        new_state.resistance = TrackedValue(
            value=max(0.1, new_r),
            unit="Ω",
            provenance=new_state.resistance.provenance,
            source="stt_switching_step",
        )

    # Update temperature slightly due to Joule heating
    joule_heat = j * dur * 1e-6  # simplified heating
    new_temp = float(new_state.temperature.value) + joule_heat
    new_state.temperature = TrackedValue(
        value=new_temp,
        unit="K",
        provenance=new_state.temperature.provenance,
        source="stt_switching_step",
    )

    # Update thermal noise
    new_noise_val = abs(float(new_state.thermal_noise.value) * (new_temp / max(float(new_state.temperature.value), 1))) if new_state.thermal_noise else 0.0
    new_state.thermal_noise = TrackedValue(
        value=new_noise_val,
        unit="Ω/√Hz",
        provenance=Provenance.ESTIMATED,
        source="stt_switching_step",
    )

    return new_state


def sot_switching_step(
    state: MRAMState,
    current_density: float | TrackedValue,
    pulse_duration: float | TrackedValue = 1e-9,
) -> MRAMState:
    """Perform a single spin-orbit torque (SOT) switching step.

    Models the probability and dynamics of magnetization switching in an
    SOT-MRAM cell under a charge current through a heavy metal layer
    (e.g., Ta, W, Pt). Uses the damping-like torque model.

    The switching probability per pulse is modeled as:

        P_switch = 1 / (1 + exp(-β * (J - Jc_sot) / Jc_sot))

    where J is the current density, Jc_sot is the critical current density
    for SOT, and β is a steepness factor. SOT switching typically has
    better endurance than STT.

    Parameters
    ----------
    state : MRAMState
        Current MRAM state.
    current_density : float or TrackedValue
        Write current density in A/m². If TrackedValue, provenance is recorded.
    pulse_duration : float or TrackedValue
        Current pulse duration in seconds. Default 1 ns. If TrackedValue,
        provenance is recorded.

    Returns
    -------
    MRAMState
        Updated MRAM state after the SOT switching step. If switching
        occurs, the magnetization direction is flipped.
    """
    # Unwrap TrackedValues
    j = float(current_density) if not isinstance(current_density, TrackedValue) else current_density.value
    dur = float(pulse_duration) if not isinstance(pulse_duration, TrackedValue) else pulse_duration.value

    new_state = state.copy()

    # Extract magnetization direction
    m = float(new_state.magnetization.value) if not isinstance(new_state.magnetization.value, np.ndarray) else float(np.mean(new_state.magnetization.value))

    # Critical current density for SOT switching (damping-like torque)
    # Jc_sot = (2e * α * Ms * t * Hk) / (ħ * θ_SH * jc_factor)
    # where θ_SH is the spin Hall angle

    # Material parameters (typical for Ta/CoFeB/MgO)
    ms = 1.0e6  # saturation magnetization A/m
    hk = 5.0e4  # anisotropy field A/m
    t = float(new_state.tunnel_barrier_thickness.value)  # barrier thickness nm
    theta_sh = 0.3  # spin Hall angle (for Ta)
    a = 1.0  # damping parameter

    # Critical current density A/m² for SOT
    hbar = 1.054571817e-34  # J·s
    e = 1.602176634e-19  # C
    t_m = t * 1e-9  # nm to meters

    jc_sot = (2.0 * e * a * ms * t_m) / (hbar * theta_sh)  # A/m²

    # Switching probability
    if jc_sot > 0:
        prob = 1.0 / (1.0 + np.exp(-a * (j - jc_sot) / max(jc_sot, 1e-10)))
    else:
        prob = 0.0

    # Stochastic switching
    import random
    random.seed(int(float(new_state.temperature.value) * 1000) % 2**31)
    r = random.random()

    # Apply switching if random number < probability
    if r < prob:
        # Switch magnetization direction (SOT typically switches with opposite polarity)
        new_m = -m
        new_state.magnetization = TrackedValue(
            value=new_m,
            provenance=new_state.magnetization.provenance,
            source="sot_switching_step",
        )

        # Resistance changes from P to AP or AP to P
        rp = float(new_state.parallel_resistance.value)
        rap = float(new_state.antiparallel_resistance.value)
        if m > 0:  # was parallel, now antiparallel
            new_state.resistance = TrackedValue(
                value=rap,
                unit="Ω",
                provenance=new_state.resistance.provenance,
                source="sot_switching_step",
            )
        else:  # was antiparallel, now parallel
            new_state.resistance = TrackedValue(
                value=rp,
                unit="Ω",
                provenance=new_state.resistance.provenance,
                source="sot_switching_step",
            )
    else:
        # No switching, resistance stays the same
        # Small thermal fluctuation in resistance
        thermal_fluct = abs(float(new_state.thermal_noise.value)) if new_state.thermal_noise else 0.0
        new_r = float(new_state.resistance.value) + np.random.normal(0, thermal_fluct * 0.1) if thermal_fluct > 0 else float(new_state.resistance.value)
        new_state.resistance = TrackedValue(
            value=max(0.1, new_r),
            unit="Ω",
            provenance=new_state.resistance.provenance,
            source="sot_switching_step",
        )

    # Update temperature slightly due to Joule heating
    joule_heat = j * dur * 1e-6  # simplified heating
    new_temp = float(new_state.temperature.value) + joule_heat
    new_state.temperature = TrackedValue(
        value=new_temp,
        unit="K",
        provenance=new_state.temperature.provenance,
        source="sot_switching_step",
    )

    # Update thermal noise
    new_noise_val = abs(float(new_state.thermal_noise.value) * (new_temp / max(float(new_state.temperature.value), 1))) if new_state.thermal_noise else 0.0
    new_state.thermal_noise = TrackedValue(
        value=new_noise_val,
        unit="Ω/√Hz",
        provenance=Provenance.ESTIMATED,
        source="sot_switching_step",
    )

    return new_state


def tmr_read_resistance(
    state: MRAMState,
) -> dict[str, Any]:
    """Read the TMR ( tunneling magnetoresistance) ratio and resistance state.

    Measures the TMR ratio and determines the current magnetic state of the
    MTJ based on the resistance. The TMR ratio is defined as:

        TMR = (RAP - RP) / RP * 100%

    where RAP is the antiparallel resistance and RP is the parallel resistance.

    Parameters
    ----------
    state : MRAMState
        Current MRAM state.

    Returns
    -------
    dict
        Dictionary containing:
        - 'resistance_Ω': current resistance in Ω
        - 'parallel_resistance_Ω': parallel state resistance in Ω
        - 'antiparallel_resistance_Ω': antiparallel state resistance in Ω
        - 'tmr_ratio_percent': TMR ratio in percent
        - 'magnetic_state': 'parallel' or 'antiparallel'
        - 'resistance_change_Ω': resistance change from baseline
    """
    r = float(state.resistance.value)
    rp = float(state.parallel_resistance.value)
    rap = float(state.antiparallel_resistance.value)

    # TMR ratio
    if rp > 0:
        tmr_ratio = (rap - rp) / rp * 100.0
    else:
        tmr_ratio = 0.0

    # Determine magnetic state based on resistance comparison
    if r <= (rp + rap) / 2.0:
        magnetic_state = "parallel"
    else:
        magnetic_state = "antiparallel"

    # Resistance change from midpoint
    baseline = (rp + rap) / 2.0
    resistance_change = r - baseline

    result = {
        "resistance_Ω": r,
        "parallel_resistance_Ω": rp,
        "antiparallel_resistance_Ω": rap,
        "tmr_ratio_percent": tmr_ratio,
        "magnetic_state": magnetic_state,
        "resistance_change_Ω": resistance_change,
    }

    # Add thermal noise summary
    thermal_noise_val = float(state.thermal_noise.value) if state.thermal_noise else 0.0
    result["thermal_noise_Ω_sqrt_Hz"] = thermal_noise_val

    return result


def thermal_stability_factor(
    state: MRAMState,
    temperature: float | TrackedValue | None = None,
) -> dict[str, Any]:
    """Calculate the thermal stability factor (Δ) for an MTJ MRAM cell.

    The thermal stability factor determines the retention time of the MRAM cell:

        Δ = (K * V) / (kB * T)

    where:
    - K is the magnetic anisotropy energy density (J/m³)
    - V is the magnetic volume (m³)
    - kB is Boltzmann's constant (1.38e-23 J/K)
    - T is the temperature in K

    For a perpendicular MTJ: V = A * t where A is the area and t is the free layer thickness.

    A stable MRAM cell typically requires Δ ≥ 60 for 10-year retention at 85°C.

    Parameters
    ----------
    state : MRAMState
        Current MRAM state.
    temperature : float or TrackedValue or None
        Temperature in K for the calculation. If None, uses state temperature.
        If TrackedValue, provenance is recorded.

    Returns
    -------
    dict
        Dictionary containing:
        - 'delta': thermal stability factor Δ (dimensionless)
        - 'energy_barrier_J': anisotropy energy barrier in J
        - 'retention_time_s': estimated retention time in seconds
        - 'critical_temperature_K': blocking temperature estimate
        - 'temperature_K': temperature used for calculation
    """
    # Use state temperature if not provided
    if temperature is None:
        t = float(state.temperature.value)
    elif isinstance(temperature, TrackedValue):
        t = temperature.value
        temp_prov = temperature.provenance
    else:
        t = float(temperature)
        temp_prov = Provenance.MEASURED

    # Material parameters for typical CoFeB/MgO MTJ
    kB = 1.380649e-23  # Boltzmann constant J/K

    # Anisotropy energy density J/m³ (typical for CoFeB)
    k = 5.0e5  # J/m³

    # Magnetic volume V = area * free_layer_thickness
    # Free layer thickness is typically ~1.5 nm for CoFeB
    free_layer_thickness_nm = 1.5
    area_m2 = float(state.area.value)
    thickness_m = free_layer_thickness_nm * 1e-9  # nm to meters
    volume_m3 = area_m2 * thickness_m

    # Energy barrier E = K * V
    energy_barrier = k * volume_m3  # J

    # Thermal stability factor Δ = E / (kB * T)
    delta = energy_barrier / (kB * max(t, 1.0))

    # Estimated retention time using Arrhenius model:
    # τ = τ0 * exp(Δ) where τ0 ≈ 1 ns
    tau0 = 1e-9  # attempt time ~1 ns
    retention_time = tau0 * np.exp(max(delta, 0))

    # Critical temperature (blocking temperature where Δ = ~25)
    # T_block = K * V / (kB * 25)
    t_block = energy_barrier / (kB * 25.0) if kB > 0 else float("inf")

    result = {
        "delta": delta,
        "energy_barrier_J": energy_barrier,
        "retention_time_s": retention_time,
        "critical_temperature_K": t_block,
        "temperature_K": t,
    }

    # Add temperature provenance if provided via TrackedValue
    if isinstance(temperature, TrackedValue):
        result["temperature_provenance"] = temperature.provenance

    return result