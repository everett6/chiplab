"""Sensitivity, read disturbance, state distinguishability and array scaling.

All four are derived from the particle/port model in ``ferro_device``; none is
experimentally validated.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chiplab.models.materials.ferro_device import (
    NOT_MODELLED,
    CoilPort,
    DeviceState,
    FerrofluidDevice,
    ReadEvent,
    WritePulse,
    simulate_sequence,
)
from chiplab.models.materials.ferro_particles import (
    KB,
    MU0,
    PARAMETER_TABLE,
    FerrofluidParams,
)

MAX_READ_CYCLES = 200
SCALING_TOLERANCE = 0.10
LINEAR_XI_LIMIT = 0.3


def rebuild(device: FerrofluidDevice, **changes: float) -> FerrofluidDevice:
    """The same ports and fluid geometry with some material parameters changed."""
    values = {n: getattr(device.params, n) for n in FerrofluidParams.names()}
    values.update(changes)
    return FerrofluidDevice(
        device.ports, FerrofluidParams(**values, provenance=dict(device.params.provenance)),
        device.region, device.concentration_length_um, device.conductivity,
        device.permittivity, device.double_layer)


def state_after_writes(device: FerrofluidDevice, writes: list[WritePulse]) -> DeviceState:
    """State at the instant the last write pulse ends."""
    times = sorted({0.0} | {w.start_s for w in writes} | {w.end_s for w in writes})
    state = device.initial_state()
    for start, end in zip(times[:-1], times[1:], strict=True):
        active = {device.index(w.port): w.voltage_v for w in writes
                  if w.start_s <= start < w.end_s}
        state = device.evolve(state, device.current(active), end - start)
    return state


# ---------------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------------


def _outputs(device: FerrofluidDevice, writes: list[WritePulse], reads: list[ReadEvent]
             ) -> dict[str, float]:
    result = simulate_sequence(device, writes, reads, interaction_analysis=False)
    out = {"peak_write_field_a_m": max(
        (w["peak_field_in_fluid_a_m"] for w in result["writes"]), default=0.0)}
    static = device.fluid_inductance(np.array([0.0]), np.zeros(len(device.ports)))[0].real
    for n, row in enumerate(result["reads"]):
        key = f"read{n}_{row['port']}"
        signal = row["signal_vs_unwritten_ohm"]
        j = device.index(row["port"])
        out[f"{key}.flux_wb"] = row["magnetization_flux_linkage_wb"]
        out[f"{key}.impedance_change_ohm"] = float(np.hypot(signal["real"], signal["imag"]))
        out[f"{key}.static_fluid_inductance_h"] = float(static[j, j])
    for port, entry in result["retention"]["per_port"].items():
        value = entry.get("time_to_1_over_e_s")
        if value is not None:
            out[f"{port}.decay_1_over_e_s"] = value
    return out


def parameter_sensitivity(device: FerrofluidDevice, writes: list[WritePulse],
                          reads: list[ReadEvent], relative_step: float = 0.05,
                          parameters: list[str] | None = None) -> dict[str, Any]:
    """Normalised sensitivities d ln(y) / d ln(p) by central differences."""
    if not 0 < relative_step <= 0.3:
        raise ValueError("relative_step must be in (0, 0.3]")
    if not writes or not reads:
        raise ValueError("sensitivity needs at least one write and one read")
    names = parameters or [*PARAMETER_TABLE, "drive_voltage"]
    unknown = sorted(set(names) - set(PARAMETER_TABLE) - {"drive_voltage"})
    if unknown:
        raise ValueError(f"unknown parameter(s) {unknown}")
    base = _outputs(device, writes, reads)
    table: dict[str, Any] = {}
    for name in names:
        runs = []
        try:
            for sign in (1.0, -1.0):
                factor = 1.0 + sign * relative_step
                if name == "drive_voltage":
                    scaled = [WritePulse(w.port, w.voltage_v * factor, w.duration_s, w.start_s)
                              for w in writes]
                    runs.append(_outputs(device, scaled, reads))
                else:
                    value = getattr(device.params, name)
                    if value == 0:
                        raise ValueError("value is zero: a relative step is undefined")
                    runs.append(_outputs(rebuild(device, **{name: value * factor}),
                                         writes, reads))
        except ValueError as error:
            table[name] = {"status": f"not run: {error}"}
            continue
        row = {}
        for key, value in base.items():
            if value == 0 or key not in runs[0] or key not in runs[1]:
                row[key] = None
            else:
                row[key] = float((runs[0][key] - runs[1][key]) / (2 * relative_step * value))
        affected = sorted(k for k, v in row.items() if v is not None and abs(v) > 1e-6)
        table[name] = {"status": "computed", "sensitivity": row, "affects": affected}
    return {
        "definition": "d ln(output) / d ln(parameter), central difference "
                      f"with a {relative_step:.0%} step; null where the base output is zero",
        "base_outputs": base,
        "parameters": table,
        "no_effect_on_these_outputs": [
            n for n, r in table.items() if r["status"] == "computed" and not r["affects"]],
        "not_modelled": NOT_MODELLED,
    }


# ---------------------------------------------------------------------------
# Read disturbance
# ---------------------------------------------------------------------------


def read_disturbance(device: FerrofluidDevice, writes: list[WritePulse], port: str | int,
                     amplitude_v: float, frequency_hz: float, cycles: int = 10,
                     steps_per_cycle: int = 32) -> dict[str, Any]:
    """Does driving the read port change the stored particle state?

    Three runs of the same length: the written state with a sinusoidal read current (peak
    V/|Z|) on `port`, the written state with no read, and an unwritten state with the read.
    (written+read) - (unwritten+read) - (written, no read) is what the read does to the
    stored information; it vanishes as the square of a small amplitude.  The unwritten
    run alone is the residue the read leaves whatever was stored (linear in amplitude).
    Both are averaged over the last cycle to remove the steady oscillation.
    """
    if not (amplitude_v > 0 and frequency_hz > 0):
        raise ValueError("read amplitude and frequency must be positive")
    if not (1 <= cycles <= MAX_READ_CYCLES and 8 <= steps_per_cycle <= 256):
        raise ValueError(f"cycles must be 1..{MAX_READ_CYCLES} and steps_per_cycle 8..256; "
                         "longer reads are not modelled")
    j = device.index(port)
    impedance = device.impedance_matrix(np.array([frequency_hz]))[0][j, j]
    current = amplitude_v / abs(impedance)
    start = state_after_writes(device, writes)
    weight = device.volume[:, None, None]
    start_norm = float(np.sqrt(np.sum(weight * start.magnetization**2)))
    read, idle, blank = start.copy(), start.copy(), device.initial_state()
    dt = 1.0 / (frequency_hz * steps_per_cycle)
    mean_difference = np.zeros_like(start.magnetization)
    mean_residue = np.zeros_like(start.magnetization)
    drive = np.zeros(len(device.ports))
    total = cycles * steps_per_cycle
    for k in range(total):
        drive[j] = current * np.sin(2 * np.pi * (k + 0.5) / steps_per_cycle)
        read = device.evolve(read, drive, dt)
        idle = device.evolve(idle, np.zeros(len(device.ports)), dt)
        blank = device.evolve(blank, drive, dt)
        if k >= total - steps_per_cycle:
            mean_difference += (read.magnetization - blank.magnetization
                                - idle.magnetization) / steps_per_cycle
            mean_residue += blank.magnetization / steps_per_cycle
    field = float(current * np.linalg.norm(device.h[j], axis=1).max())
    xi = field * device.ensemble.typical_xi_per_field
    warnings = []
    if xi > LINEAR_XI_LIMIT:
        warnings.append("read field is not small: the near-equilibrium relaxation equation "
                        "is being used outside its comfortable range")
    out: dict[str, Any] = {
        "read_port": device.ports[j].id,
        "read_current_peak_a": float(current),
        "peak_read_field_a_m": field,
        "peak_read_langevin_argument": float(xi),
        "read_duration_s": cycles / frequency_hz,
        "stored_state_norm_at_read_start": start_norm,
        "concentration_change_caused_by_read": float(
            np.abs(read.concentration - idle.concentration).max()),
        "not_modelled": ["Joule heating by the read current",
                         "field-assisted switching of blocked particles",
                         f"reads longer than {MAX_READ_CYCLES} cycles"],
        "warnings": warnings,
    }
    if start_norm == 0:
        out.update(disturbance_fraction=None, read_residue_fraction=None,
                   remaining_without_read=None, note="nothing was stored when the read started")
    else:
        out["disturbance_fraction"] = float(
            np.sqrt(np.sum(weight * mean_difference**2)) / start_norm)
        out["read_residue_fraction"] = float(
            np.sqrt(np.sum(weight * mean_residue**2)) / start_norm)
        out["remaining_without_read"] = float(
            np.sqrt(np.sum(weight * idle.magnetization**2)) / start_norm)
        out["note"] = (
            "all fractions are relative to the stored state when the read starts: "
            "disturbance_fraction is the read's effect on that state; read_residue_fraction "
            "is magnetization the read adds regardless of the state; "
            "1 - remaining_without_read is lost to ordinary relaxation anyway")
    return out


# ---------------------------------------------------------------------------
# State distinguishability over time
# ---------------------------------------------------------------------------


def state_distinguishability(device: FerrofluidDevice, writes_a: list[WritePulse],
                             writes_b: list[WritePulse], port: str | int,
                             bandwidth_hz: float = 1e6, snr_threshold: float = 3.0
                             ) -> dict[str, Any]:
    """How long can two written states be told apart from the voltage induced in a port?

    Signal: difference of the magnetization emf, a sum of exponentials.  Noise: thermal
    voltage of the port resistance in `bandwidth_hz`.  A single sample, no averaging.
    """
    if not (bandwidth_hz > 0 and snr_threshold > 0):
        raise ValueError("bandwidth and snr_threshold must be positive")
    j = device.index(port)
    tau = device.ensemble.tau_zero_field

    def flux_by_bin(writes: list[WritePulse]) -> np.ndarray:
        state = state_after_writes(device, writes)
        return MU0 * np.einsum("cx,cbx,c->b", device.h[j], state.magnetization,
                                        device.volume, optimize=True)

    difference = flux_by_bin(writes_a) - flux_by_bin(writes_b)
    noise = float(np.sqrt(4 * KB * device.params.temperature_k * device.resistance[j]
                          * bandwidth_hz))
    earliest = 1.0 / (2.0 * bandwidth_hz)
    longest = max(earliest * 10, float(min(tau.max() * 100, 1e18)))
    times = np.logspace(np.log10(earliest), np.log10(longest), 1500)
    decay = np.exp(-times[:, None] / tau)
    signal = np.abs(decay @ (difference / tau))
    above = np.nonzero(signal >= snr_threshold * noise)[0]
    return {
        "read_port": device.ports[j].id,
        "thermal_noise_v_rms": noise,
        "bandwidth_hz": bandwidth_hz,
        "snr_threshold": snr_threshold,
        "earliest_sample_s": earliest,
        "snr_at_earliest_sample": float(signal[0] / noise),
        "distinguishable": bool(len(above) > 0),
        "distinguishable_until_s": float(times[above[-1]]) if len(above) else None,
        "samples": [{"delay_s": float(times[i]), "signal_v": float(signal[i]),
                     "snr": float(signal[i] / noise)} for i in range(0, 1500, 150)],
        "assumptions": "ideal sample of the induced voltage; amplifier noise, averaging, "
                       "flux-based detection and slow concentration differences are excluded",
    }


# ---------------------------------------------------------------------------
# Scaling to larger arrays
# ---------------------------------------------------------------------------


def _grid_ports(geometry: dict, shape: tuple[int, int], pitch_um: float) -> list[CoilPort]:
    return [CoilPort(id=f"p{ix}_{iy}", position_um=(ix * pitch_um, iy * pitch_um, 0.0),
                     **geometry)
            for iy in range(shape[1]) for ix in range(shape[0])]


def array_scaling(params: FerrofluidParams, geometry: dict, pitch_um: float,
                  small_shape: tuple[int, int], large_shape: tuple[int, int],
                  voltage_v: float) -> dict[str, Any]:
    """Predict the large-array steady flux from port-to-port transfer in a small array.

    Basis: in weak fields flux is linear in the currents and, away from the fluid edges,
    depends only on the offset between ports.  The prediction is accepted only after it
    agrees with a direct simulation of the large array within 10 percent; larger arrays
    than the solver can check directly are reported as not modelled.
    """
    small = (int(small_shape[0]), int(small_shape[1]))
    large = (int(large_shape[0]), int(large_shape[1]))
    if min(small + large) < 1 or small[0] * small[1] > 32 or small[0] * small[1] < 2:
        raise ValueError("small array needs 2..32 ports and positive dimensions")
    n_large = large[0] * large[1]
    not_modelled = ["read impedance change (quadratic in field, not superposable)",
                    "storage capacity, bandwidth, latency, energy per bit, yield, routing "
                    "parasitics, heating", "arrays with more than 32 ports"]
    base: dict[str, Any] = {
        "small_array": list(small), "large_array": list(large), "pitch_um": pitch_um,
        "voltage_v": voltage_v, "tolerance": SCALING_TOLERANCE, "not_modelled": not_modelled,
    }
    if n_large > 32:
        return {**base, "status": "not_modelled",
                "reason": f"{n_large} ports cannot be checked against a direct simulation"}
    source = FerrofluidDevice(_grid_ports(geometry, small, pitch_um), params)
    current = voltage_v / source.resistance[0]
    field = float(abs(current) * np.linalg.norm(source.field(np.ones(len(source.ports))),
                                                axis=1).max())
    xi = field * source.ensemble.typical_xi_per_field
    transfer = source.fluid_inductance(np.array([0.0]), np.zeros(len(source.ports)))[0].real
    table: dict[tuple[int, int], list[float]] = {}
    index = [(ix, iy) for iy in range(small[1]) for ix in range(small[0])]
    for j, (xj, yj) in enumerate(index):
        for i, (xi_, yi_) in enumerate(index):
            table.setdefault((abs(xj - xi_), abs(yj - yi_)), []).append(transfer[j, i])
    mean = {k: float(np.mean(v)) for k, v in table.items()}
    spread = max(float(np.ptp(v) / max(abs(np.mean(v)), 1e-300)) for k, v in table.items()
                 if k == (0, 0))
    target = FerrofluidDevice(_grid_ports(geometry, large, pitch_um), params)
    cells = [(ix, iy) for iy in range(large[1]) for ix in range(large[0])]
    currents = voltage_v / target.resistance
    predicted = np.zeros(n_large)
    unseen = 0
    for j, (xj, yj) in enumerate(cells):
        for i, (xi_, yi_) in enumerate(cells):
            key = (abs(xj - xi_), abs(yj - yi_))
            if key in mean:
                predicted[j] += mean[key] * currents[i]
            else:
                unseen += 1
    # direct check: steady state under the same drive
    longest = float(min(target.ensemble.tau_zero_field.max() * 50, 1e18))
    state = target.evolve(target.initial_state(), currents, longest)
    direct = target.flux_linkage(state)
    scale = np.max(np.abs(direct))
    error = np.abs(predicted - direct) / scale if scale > 0 else np.zeros(n_large)
    worst = float(error.max())
    reasons = []
    if xi > LINEAR_XI_LIMIT:
        reasons.append(f"drive is nonlinear (Langevin argument {xi:.2f} > {LINEAR_XI_LIMIT})")
    if worst > SCALING_TOLERANCE:
        reasons.append(f"prediction differs from direct simulation by {worst:.0%}")
    grid_error = target.grid_convergence()
    if grid_error > 0.1:
        reasons.append(f"direct large-array grid is not converged ({grid_error:.0%})")
    if scale == 0:
        reasons.append("zero response: nothing to compare")
    ok = not reasons
    out = {
        **base,
        "status": "validated_against_direct_model" if ok else "not_modelled",
        "reason": None if ok else "; ".join(reasons),
        "worst_relative_error": worst,
        "peak_langevin_argument": float(xi),
        "offsets_missing_from_small_array_pairs": unseen,
        "self_term_spread_in_small_array": spread,
        "direct_grid_convergence": grid_error,
        "validity_range": (
            f"only this geometry, pitch and material, drive up to {abs(voltage_v)} V, "
            f"arrays up to {large[0]} x {large[1]}; checked against this model, "
            "not against measurements") if ok else None,
        "drive_power_w": float(np.sum(voltage_v**2 / target.resistance)),
    }
    if ok:
        out["ports"] = [{"id": p.id, "predicted_flux_wb": float(a),
                         "direct_flux_wb": float(b)}
                        for p, a, b in zip(target.ports, predicted, direct, strict=True)]
    return out
