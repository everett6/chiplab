"""Ferroelectric (FeFET/FeRAM) material physics models.

Landau-Khalatnikov dynamics, P-V hysteresis, and FeFET threshold voltage shift
modeling for ferroelectric memory devices.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chiplab.schemas.common import Provenance, TrackedValue


class FerroelectricState:
    """State of a ferroelectric material system.

    Tracks polarization, electric field, voltage, and device parameters.
    All scalar values use TrackedValue with Provenance tracking.
    """

    def __init__(
        self,
        polarization: TrackedValue | float = 0.0,
        electric_field: TrackedValue | float = 0.0,
        voltage: TrackedValue | float = 0.0,
        remanent_polarization: TrackedValue | float = 0.0,
        coercive_field: TrackedValue | float = 0.0,
        dielectric_thickness: TrackedValue | float = 100.0,
        area: TrackedValue | float = 1e-4,
        temperature: TrackedValue | float = 300.0,
        fatigue_factor: TrackedValue | float = 1.0,
        noise: TrackedValue | float | None = None,
    ):
        """Initialize FerroelectricState with flexible input types (TrackedValue or float).

        Parameters
        ----------
        polarization : TrackedValue or float
            Polarization in μC/cm². Default 0.0 (down state).
        electric_field : TrackedValue or float
            Electric field in V/μm. Default 0.0.
        voltage : TrackedValue or float
            Applied voltage in V. Default 0.0.
        remanent_polarization : TrackedValue or float
            Remanent polarization in μC/cm². Default 0.0.
        coercive_field : TrackedValue or float
            Coercive field in V/μm. Default 0.0.
        dielectric_thickness : TrackedValue or float
            Dielectric thickness in nm. Default 100.0.
        area : TrackedValue or float
            Device area in cm². Default 1e-4 (100 μm × 100 μm).
        temperature : TrackedValue or float
            Temperature in K. Default 300.0.
        fatigue_factor : TrackedValue or float
            Fatigue factor (1.0 = fresh, < 1.0 = degraded). Default 1.0.
        noise : TrackedValue or float or None
            Noise in pA/√Hz. Default None (auto-initialized).
        """
        # Handle polarization
        if isinstance(polarization, TrackedValue):
            self.polarization = polarization
        else:
            self.polarization = TrackedValue(
                value=float(polarization),
                unit="μC/cm²",
                provenance=Provenance.ASSUMED,
            )

        # Handle electric_field
        if isinstance(electric_field, TrackedValue):
            self.electric_field = electric_field
        else:
            self.electric_field = TrackedValue(
                value=float(electric_field),
                unit="V/μm",
                provenance=Provenance.ASSUMED,
            )

        # Handle voltage
        if isinstance(voltage, TrackedValue):
            self.voltage = voltage
        else:
            self.voltage = TrackedValue(
                value=float(voltage),
                unit="V",
                provenance=Provenance.ASSUMED,
            )

        # Handle remanent_polarization
        if isinstance(remanent_polarization, TrackedValue):
            self.remanent_polarization = remanent_polarization
        else:
            self.remanent_polarization = TrackedValue(
                value=float(remanent_polarization),
                unit="μC/cm²",
                provenance=Provenance.ASSUMED,
            )

        # Handle coercive_field
        if isinstance(coercive_field, TrackedValue):
            self.coercive_field = coercive_field
        else:
            self.coercive_field = TrackedValue(
                value=float(coercive_field),
                unit="V/μm",
                provenance=Provenance.ASSUMED,
            )

        # Handle dielectric_thickness
        if isinstance(dielectric_thickness, TrackedValue):
            self.dielectric_thickness = dielectric_thickness
        else:
            self.dielectric_thickness = TrackedValue(
                value=float(dielectric_thickness),
                unit="nm",
                provenance=Provenance.ASSUMED,
            )

        # Handle area
        if isinstance(area, TrackedValue):
            self.area = area
        else:
            self.area = TrackedValue(
                value=float(area),
                unit="cm²",
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

        # Handle fatigue_factor
        if isinstance(fatigue_factor, TrackedValue):
            self.fatigue_factor = fatigue_factor
        else:
            self.fatigue_factor = TrackedValue(
                value=float(fatigue_factor),
                provenance=Provenance.ASSUMED,
            )

        # Handle noise
        if noise is None:
            self.noise = TrackedValue(value=0.0, unit="pA/√Hz", provenance=Provenance.ASSUMED)
        elif isinstance(noise, TrackedValue):
            self.noise = noise
        else:
            self.noise = TrackedValue(value=float(noise), unit="pA/√Hz", provenance=Provenance.ASSUMED)

    def copy(self) -> FerroelectricState:
        """Create a deep copy of the state.

        Returns
        -------
        FerroelectricState
            New FerroelectricState with all values copied.
        """
        return FerroelectricState(
            polarization=TrackedValue(
                value=self.polarization.value,
                unit=self.polarization.unit,
                provenance=self.polarization.provenance,
                source=self.polarization.source,
                uncertainty_pct=self.polarization.uncertainty_pct,
                notes=self.polarization.notes,
            ),
            electric_field=TrackedValue(
                value=self.electric_field.value,
                unit=self.electric_field.unit,
                provenance=self.electric_field.provenance,
                source=self.electric_field.source,
                uncertainty_pct=self.electric_field.uncertainty_pct,
                notes=self.electric_field.notes,
            ),
            voltage=TrackedValue(
                value=self.voltage.value,
                unit=self.voltage.unit,
                provenance=self.voltage.provenance,
                source=self.voltage.source,
                uncertainty_pct=self.voltage.uncertainty_pct,
                notes=self.voltage.notes,
            ),
            remanent_polarization=TrackedValue(
                value=self.remanent_polarization.value,
                unit=self.remanent_polarization.unit,
                provenance=self.remanent_polarization.provenance,
                source=self.remanent_polarization.source,
                uncertainty_pct=self.remanent_polarization.uncertainty_pct,
                notes=self.remanent_polarization.notes,
            ),
            coercive_field=TrackedValue(
                value=self.coercive_field.value,
                unit=self.coercive_field.unit,
                provenance=self.coercive_field.provenance,
                source=self.coercive_field.source,
                uncertainty_pct=self.coercive_field.uncertainty_pct,
                notes=self.coercive_field.notes,
            ),
            dielectric_thickness=TrackedValue(
                value=self.dielectric_thickness.value,
                unit=self.dielectric_thickness.unit,
                provenance=self.dielectric_thickness.provenance,
                source=self.dielectric_thickness.source,
                uncertainty_pct=self.dielectric_thickness.uncertainty_pct,
                notes=self.dielectric_thickness.notes,
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
            fatigue_factor=TrackedValue(
                value=self.fatigue_factor.value,
                provenance=self.fatigue_factor.provenance,
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


def landau_khalatnikov_step(
    state: FerroelectricState,
    voltage: float | TrackedValue,
    dt: float | TrackedValue,
    alpha: float | TrackedValue = 1.0,
    beta: float | TrackedValue = 1.0,
    gamma: float | TrackedValue = 0.1,
) -> FerroelectricState:
    """Perform a single Landau-Khalatnikov time step for polarization switching.

    The Landau-Khalatnikov equation describes the dynamics of ferroelectric
    polarization switching under an applied electric field:

        dP/dt = -γ * δF/δP = -γ * (α*P + β*P³ - E)

    where P is polarization, E is electric field, and α, β, γ are material
    parameters. The effective field includes contributions from the applied
    voltage and the intrinsic material response.

    Parameters
    ----------
    state : FerroelectricState
        Current ferroelectric state.
    voltage : float or TrackedValue
        Applied voltage in V. If TrackedValue, provenance is recorded.
    dt : float or TrackedValue
        Time step in seconds. If TrackedValue, provenance is recorded.
    alpha : float or TrackedValue
        Landau coefficient α (default 1.0). Controls the double-well depth.
        If TrackedValue, provenance is recorded.
    beta : float or TrackedValue
        Landau coefficient β (default 1.0). Controls the nonlinearity.
        If TrackedValue, provenance is recorded.
    gamma : float or TrackedValue
        Kinetic coefficient γ (default 0.1). Controls switching speed.
        If TrackedValue, provenance is recorded.

    Returns
    -------
    FerroelectricState
        Updated ferroelectric state after the LK step.
    """
    # Unwrap TrackedValues
    v = float(voltage) if not isinstance(voltage, TrackedValue) else voltage.value
    d = float(dt) if not isinstance(dt, TrackedValue) else dt.value
    a = float(alpha) if not isinstance(alpha, TrackedValue) else alpha.value
    b = float(beta) if not isinstance(beta, TrackedValue) else beta.value
    g = float(gamma) if not isinstance(gamma, TrackedValue) else gamma.value

    # Compute electric field from voltage (E = V / thickness, thickness in nm -> V/μm)
    # dielectric_thickness is in nm, convert to μm: 1 nm = 0.001 μm
    thickness_um = float(state.dielectric_thickness.value) * 0.001
    e = v / max(thickness_um, 1e-6)  # electric field in V/μm

    new_state = state.copy()

    # Landau-Khalatnikov: dP/dt = -γ * (α*P + β*P³ - E)
    # discretized: P_new = P - γ * dt * (α*P + β*P³ - E)
    p = float(state.polarization.value)
    ep = float(state.electric_field.value)

    # Effective field term
    effective_field = a * p + b * p**3 - e

    # Polarization update
    dp = -g * d * effective_field
    new_p = p + dp

    # Apply fatigue degradation
    ff = float(state.fatigue_factor.value)
    new_p = new_p * ff  # fatigue reduces switchable polarization

    # Clamp polarization to physical bounds [-Pr, Pr] approximately
    pr = float(state.remanent_polarization.value)
    new_p = max(-pr * 1.5, min(pr * 1.5, new_p))

    new_state.polarization = TrackedValue(
        value=new_p,
        unit=state.polarization.unit,
        provenance=state.polarization.provenance,
        source="landau_khalatnikov_step",
    )

    # Update electric field
    new_state.electric_field = TrackedValue(
        value=e,
        unit=state.electric_field.unit,
        provenance=state.electric_field.provenance,
        source="landau_khalatnikov_step",
    )

    # Update voltage
    new_state.voltage = TrackedValue(
        value=v,
        unit=state.voltage.unit,
        provenance=state.voltage.provenance,
        source="landau_khalatnikov_step",
    )

    # Slight heating due to switching
    temp_increase = abs(dp) * 0.01  # simplified heating
    new_temp = float(state.temperature.value) + temp_increase
    new_state.temperature = TrackedValue(
        value=new_temp,
        unit="K",
        provenance=state.temperature.provenance,
        source="landau_khalatnikov_step",
    )

    # Noise from switching
    new_noise_value = abs(dp) * 0.1 + (float(state.noise.value) if state.noise else 0.0)
    new_state.noise = TrackedValue(
        value=new_noise_value,
        unit=state.noise.unit if state.noise else "pA/√Hz",
        provenance=Provenance.ESTIMATED,
        source="landau_khalatnikov_step",
    )

    return new_state


def ferroelectric_pv_loop(
    state: FerroelectricState,
    voltage_sweep: list[float | TrackedValue],
    include_noise: bool = False,
) -> dict[str, Any]:
    """Simulate a P-V (polarization-voltage) hysteresis loop.

    Sweeps voltage across the ferroelectric capacitor and tracks the resulting
    polarization, producing a hysteresis loop characteristic of ferroelectric
    materials.

    Parameters
    ----------
    state : FerroelectricState
        Initial ferroelectric state.
    voltage_sweep : list of float or TrackedValue
        Sequence of voltage values to apply (in V). Typically a full cycle:
        -V_max -> +V_max -> +V_min -> -V_min -> -V_max.
    include_noise : bool, optional
        Whether to include thermal noise in the polarization response. Default False.

    Returns
    -------
    dict
        Dictionary containing:
        - 'voltage_V': list of applied voltages
        - 'polarization_ucm2': list of polarization values in μC/cm²
        - 'hysteresis': bool indicating if hysteresis was observed
        - 'coercive_voltage_V': coercive voltage from the loop
        - 'remanent_polarization_ucm2': remanent polarization from the loop
    """
    # Track the P-V loop
    voltages = []
    polarizations = []

    # Start from remanent state (assuming negative remanence)
    current_state = state.copy()

    for v_arg in voltage_sweep:
        # Unwrap voltage if TrackedValue
        v = float(v_arg) if not isinstance(v_arg, TrackedValue) else v_arg.value

        # Apply LK step
        current_state = landau_khalatnikov_step(
            state=current_state,
            voltage=v,
            dt=1e-9,  # 1 ns default time step
        )

        voltages.append(v)
        polarizations.append(float(current_state.polarization.value))

    # Analyze the hysteresis loop
    # Find coercive voltage (where P = 0)
    coercive_voltage = None
    remanent_polarization = None

    # Find points near P = 0 (crossings)
    zero_crossings = []
    for i in range(1, len(polarizations)):
        if polarizations[i - 1] * polarizations[i] <= 0:
            # Linear interpolation to find V where P = 0
            v1, v2 = voltages[i - 1], voltages[i]
            p1, p2 = polarizations[i - 1], polarizations[i]
            if abs(p2 - p1) > 1e-10:
                v_zero = v1 - p1 * (v2 - v1) / (p2 - p1)
                zero_crossings.append(v_zero)

    # Also find remanent polarization (P at E=0, i.e., V=0 points)
    # After positive saturation, find P at V=0 on the return sweep
    # Simple approach: look at the loop endpoints
    if len(polarizations) >= 4:
        # Positive remanence (after positive sweep)
        prem = polarizations[-1] if len(polarizations) > 0 else 0.0
        # Negative remanence
        nrem = polarizations[0] if len(polarizations) > 0 else 0.0
        remanent_polarization = abs(prem)  # use absolute value

    # Coercive voltage from zero crossings
    if zero_crossings:
        coercive_voltage = sum(zero_crossings) / len(zero_crossings)

    # Hysteresis detected if there's a significant loop
    hysteresis = (
        remanent_polarization is not None
        and remanent_polarization > 0.1
        and coercive_voltage is not None
        and abs(coercive_voltage) > 0.1
    )

    result = {
        "voltage_V": voltages,
        "polarization_ucm2": polarizations,
        "hysteresis": hysteresis,
        "coercive_voltage_V": coercive_voltage,
        "remanent_polarization_ucm2": remanent_polarization,
    }

    if include_noise:
        # Add noise summary
        noise_vals = []
        for i, v_arg in enumerate(voltage_sweep):
            v = float(v_arg) if not isinstance(v_arg, TrackedValue) else v_arg.value
            # Add slight noise based on voltage magnitude
            noise_component = abs(v) * 1e-3
            noise_vals.append(noise_component)
        result["noise_summary"] = {
            "voltage_noise_pA_sqrt_Hz": [
                float(n) if not isinstance(n, float) else n for n in noise_vals
            ]
        }

    return result


def fefet_threshold_shift(
    state: FerroelectricState,
    program_voltage: float | TrackedValue,
    program_duration: float | TrackedValue,
    n_cycles: int = 1,
) -> dict[str, Any]:
    """Calculate FeFET threshold voltage shift after ferroelectric polarization switching.

    In a FeFET (Ferroelectric FET), the threshold voltage shifts in response to
    polarization switching in the ferroelectric channel. The shift depends on the
    amount of polarization switched, the gate capacitance, and device parameters.

    The threshold voltage shift is modeled as:

        ΔVth = (2 * Q * P_sw) / (C_ge * W * L)

    where:
    - P_sw is the switched polarization (μC/cm²)
    - Q is the coupling efficiency (0 < Q ≤ 1)
    - C_ge is the gate oxide capacitance
    - W, L are gate width and length

    Parameters
    ----------
    state : FerroelectricState
        Current ferroelectric state.
    program_voltage : float or TrackedValue
        Programming voltage in V. If TrackedValue, provenance is recorded.
    program_duration : float or TrackedValue
        Programming pulse duration in s. If TrackedValue, provenance is recorded.
    n_cycles : int, optional
        Number of switching cycles (default 1). Multiple cycles can cause
        fatigue-dependent shifts.

    Returns
    -------
    dict
        Dictionary containing:
        - 'delta_vth_V': threshold voltage shift in V
        - 'switched_polarization_ucm2': polarization switched in μC/cm²
        - 'effective_coupling': effective coupling efficiency
        - 'fe_fet_voltage_V': effective FeFET gate voltage
    """
    # Unwrap program voltage if TrackedValue
    v = float(program_voltage) if not isinstance(program_voltage, TrackedValue) else program_voltage.value

    # Unwrap program duration if TrackedValue
    d = float(program_duration) if not isinstance(program_duration, TrackedValue) else program_duration.value

    n = int(n_cycles)

    # Perform a landau-khalatnikov step to simulate polarization switching
    switched_state = landau_khalatnikov_step(
        state=state,
        voltage=v,
        dt=d,
    )

    # Calculate switched polarization (change from initial)
    initial_p = float(state.polarization.value)
    final_p = float(switched_state.polarization.value)
    switched_p = final_p - initial_p  # positive = switched up, negative = switched down

    # Apply fatigue after multiple cycles
    ff = float(state.fatigue_factor.value)
    fatigue_factor = max(0.1, ff ** n)  # fatigue accumulates with cycles
    switched_p = switched_p * fatigue_factor

    # Calculate threshold voltage shift
    # Simplified model: ΔVth = (2 * Q * P_sw) / (ε₀ * ε_r / t)
    # where ε₀ = 8.854e-12 F/m, area in cm², thickness in nm

    # Gate area in m²
    area_cm2 = float(state.area.value)
    area_m2 = area_cm2 * 1e-4  # cm² to m²

    # Dielectric thickness in m
    thickness_nm = float(state.dielectric_thickness.value)
    thickness_m = thickness_nm * 1e-9  # nm to m

    # Relative permittivity of typical ferroelectric (e.g., HfO2)
    epsilon_r = 25.0

    # Gate capacitance per unit area: C/A = ε₀ * ε_r / t
    # ΔVth = (2 * Q * P_sw) / (C/A) where Q is coupling efficiency
    # Using Q ≈ 0.7 for typical HfO2-based FeFETs

    coupling_efficiency = 0.7  # typical value for HfO2 FeFETs

    # Polarization in C/m²: 1 μC/cm² = 1e-6 C / 1e-4 m² = 1e-2 C/m²
    switched_p_cms2 = abs(switched_p)
    switched_p_cm2_to_si = switched_p_cms2 * 1e-2  # μC/cm² to C/m²

    # Capacitance per unit area
    capacitance_per_area = epsilon_r * 8.854e-12 / max(thickness_m, 1e-12)

    # Threshold voltage shift
    delta_vth = (2.0 * coupling_efficiency * switched_p_cm2_to_si) / max(capacitance_per_area, 1e-12)

    # Scale by number of cycles and fatigue
    delta_vth = delta_vth * min(n, 10)  # saturate at 10 cycles

    # Effective FeFET voltage
    fe_fet_voltage = v * coupling_efficiency

    result = {
        "delta_vth_V": delta_vth,
        "switched_polarization_ucm2": abs(switched_p),
        "effective_coupling": coupling_efficiency,
        "fe_fet_voltage_V": fe_fet_voltage,
        "n_cycles": n,
        "fatigue_factor": fatigue_factor,
        "initial_polarization_ucm2": abs(initial_p),
        "final_polarization_ucm2": abs(final_p),
    }

    # Update fatigue factor in the state
    new_fatigue = max(0.1, ff ** n)
    switched_state.fatigue_factor = TrackedValue(
        value=new_fatigue,
        provenance=state.fatigue_factor.provenance,
        source="fefet_threshold_shift",
    )

    # Update noise
    new_noise = abs(switched_p) * 0.05 + (float(state.noise.value) if state.noise else 0.0)
    switched_state.noise = TrackedValue(
        value=new_noise,
        unit=state.noise.unit if state.noise else "pA/√Hz",
        provenance=Provenance.ESTIMATED,
        source="fefet_threshold_shift",
    )

    return result