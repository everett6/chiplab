"""Physics validation gate for a ferrofluid device.

Every check returns pass, fail or not_run with its evidence.  A device is accepted only
if every check passes, so a device without the user's own measurements is rejected.

The physics checks compare independent code paths and analytic limits of the *model*;
passing them shows the computation is self-consistent, not that the model is true.  Only
the experiment checks compare against reality, and ChipLab cannot verify that supplied
measurements are authentic.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np

from chiplab.models.materials import ferro_interface as interface
from chiplab.models.materials.ferro_analysis import rebuild, state_after_writes
from chiplab.models.materials.ferro_calibration import (
    DATASET_COLUMNS,
    _column,
    _Dataset,
    dataset_fingerprint,
)
from chiplab.models.materials.ferro_device import (
    FerrofluidDevice,
    ReadEvent,
    WritePulse,
    simulate_sequence,
)
from chiplab.models.materials.ferro_particles import (
    MODEL_VERSION,
    PARAMETER_TABLE,
    ParticleEnsemble,
)

# Acceptance thresholds (choices made here, not standards)
MAX_REDUCED_CHI_SQUARE = 2.0
MIN_FRACTION_WITHIN_2_SIGMA = 0.9
MIN_POINTS = 5
MAX_RELATIVE_SIGMA = 0.2  # median uncertainty / rms signal
MAX_TEMPERATURE_MISMATCH_K = 5.0
MAX_GRID_CHANGE = 0.10

RECORD_KEYS = {"device_spec", "electrode_array", "write_sequence", "read_sequence",
               "validation_measurements"}
PHYSICS_IDS = ("zero_input", "parameter_sensitivity", "unit_consistency",
               "boundary_conditions", "conservation", "numerical_convergence",
               "model_validity")
EXPERIMENT_IDS = ("experiment_equilibrium", "experiment_dynamics",
                  "experiment_port_response")
CHECK_IDS = (*PHYSICS_IDS, "inputs_established", *EXPERIMENT_IDS)
PORT_DATA_COLUMNS = ["frequency_hz", "delta_z_real_ohm", "delta_z_imag_ohm",
                     "sigma_delta_z_real_ohm", "sigma_delta_z_imag_ohm"]


def _check(identifier: str, status: str, requirement: str, evidence: Any) -> dict[str, Any]:
    return {"id": identifier, "status": status, "requirement": requirement,
            "evidence": evidence}


def _passed(flag: Any) -> str:
    return "pass" if bool(flag) else "fail"


# ---------------------------------------------------------------------------
# Physics checks
# ---------------------------------------------------------------------------


def _zero_input(device: FerrofluidDevice, writes: list[WritePulse]) -> dict[str, Any]:
    quiet = [WritePulse(w.port, 0.0, w.duration_s, w.start_s) for w in writes]
    state = state_after_writes(device, quiet)
    zero = np.zeros(len(device.ports))
    evidence = {
        "max_magnetization_a_m": float(np.abs(state.magnetization).max()),
        "max_flux_wb": float(np.abs(device.flux_linkage(state)).max()),
        "max_emf_v": float(np.abs(device.emf(state, zero)).max()),
        "max_concentration_change": float(np.abs(state.concentration).max()),
    }
    ok = all(v <= 1e-12 for v in evidence.values())
    return _check("zero_input", _passed(ok),
                  "no drive gives no magnetization, flux, voltage or redistribution", evidence)


def _sensitivity(device: FerrofluidDevice, writes: list[WritePulse]) -> dict[str, Any]:
    ensemble = device.ensemble
    values = {n: getattr(device.params, n) for n in type(device.params).names()}

    def changed(**kw: float) -> ParticleEnsemble:
        return ParticleEnsemble(type(device.params)(**{**values, **kw}))

    chi = ensemble.diagnostics()["initial_susceptibility"]
    # perturbations go downward so that no parameter leaves its accepted range
    less = changed(volume_fraction=values["volume_fraction"] * 0.9)
    colder = changed(temperature_k=values["temperature_k"] * 0.9)
    evidence: dict[str, Any] = {
        "susceptibility_falls_with_lower_volume_fraction":
            bool(less.diagnostics()["initial_susceptibility"] < chi),
        "susceptibility_rises_with_lower_temperature":
            bool(colder.diagnostics()["initial_susceptibility"] > chi),
    }
    if device.params.carrier_mobile:
        thinner = changed(viscosity_pa_s=values["viscosity_pa_s"] * 0.9)
        evidence["brownian_time_falls_with_lower_viscosity"] = bool(
            np.all(thinner.tau_brown < ensemble.tau_brown))
    currents = device.current({device.index(w.port): w.voltage_v for w in writes})
    one = np.linalg.norm(device.field(currents), axis=1).max()
    two = np.linalg.norm(device.field(2 * currents), axis=1).max()
    evidence["field_doubles_with_voltage"] = bool(one > 0 and np.isclose(two, 2 * one))
    return _check("parameter_sensitivity", _passed(all(evidence.values())),
                  "changing a physical input moves the expected outputs the expected way",
                  evidence)


def _units(device: FerrofluidDevice, writes: list[WritePulse]) -> dict[str, Any]:
    port = device.index(writes[0].port)
    unit_field = float(np.linalg.norm(device.h[port], axis=1).max())
    current = np.zeros(len(device.ports))
    current[port] = 1e-4 / (unit_field * device.ensemble.typical_xi_per_field)
    longest = float(device.ensemble.tau_zero_field.max() * 60)
    # magnetic path: evolve the particles, integrate flux (Wb) and divide by current (A)
    still = rebuild(device, carrier_mobile=device.params.carrier_mobile)
    still.tau_concentration = np.inf
    state = still.evolve(still.initial_state(), current, longest)
    magnetic = still.flux_linkage(state) / current[port]
    # electrical path: low-frequency reactance of the impedance matrix (ohm / (rad/s) = H)
    frequency = 1e-9 / (2 * np.pi * longest)
    z = device.impedance_matrix(np.array([frequency]))[0]
    omega = 2 * np.pi * frequency
    electrical = device.fluid_inductance(np.array([frequency]),
                                         np.zeros(len(device.ports)))[0].real[:, port]
    scale = np.abs(electrical).max()
    flux_error = float(np.abs(magnetic - electrical).max() / scale) if scale > 0 else 0.0
    total = z.imag[:, port] / omega
    reference = device.inductance_air[:, port] + electrical
    reactance_error = float(np.abs(total - reference).max() / np.abs(reference).max())
    middle = device.evolve(state, np.zeros(len(device.ports)),
                           float(device.ensemble.tau_zero_field.min()))
    dt = float(device.ensemble.tau_zero_field.min()) * 1e-5
    later = device.evolve(middle, np.zeros(len(device.ports)), dt)
    numeric = -(device.flux_linkage(later) - device.flux_linkage(middle)) / dt
    emf = device.emf(middle, np.zeros(len(device.ports)))
    emf_scale = np.abs(emf).max()
    emf_error = float(np.abs(emf - numeric).max() / emf_scale) if emf_scale > 0 else 0.0
    evidence = {"flux_per_amp_vs_inductance_relative_error": flux_error,
                "reactance_over_omega_relative_error": reactance_error,
                "emf_vs_flux_derivative_relative_error": emf_error,
                "particle_concentration_held_fixed": True}
    ok = flux_error < 1e-2 and reactance_error < 1e-6 and emf_error < 1e-2
    return _check("unit_consistency", _passed(ok),
                  "flux/current equals the inductance seen electrically (H) and the induced "
                  "voltage equals -d(flux)/dt (V)", evidence)


def _boundaries(device: FerrofluidDevice) -> dict[str, Any]:
    ensemble = device.ensemble
    params = device.params
    saturation = params.volume_fraction * params.domain_magnetization_a_m
    values = {n: getattr(params, n) for n in type(params).names()}
    empty = ParticleEnsemble(type(params)(**{**values, "volume_fraction": 0.0}))
    chi0 = abs(ensemble.complex_susceptibility(0.0)[0][0])
    tau = ensemble.tau_zero_field
    evidence = {
        "saturation_relative_error": (
            float(abs(ensemble.magnetization(1e10) - saturation) / saturation)
            if saturation > 0 else None),
        "susceptibility_without_particles": float(
            abs(empty.complex_susceptibility(1e6)[0][0])),
        "high_frequency_susceptibility_ratio": (
            float(abs(ensemble.complex_susceptibility(1e6 / tau.min())[0][0]) / chi0)
            if chi0 > 0 else None),
        "weak_field_linearity_error": (
            float(abs(ensemble.magnetization(2e-2) / (2 * ensemble.magnetization(1e-2)) - 1))
            if saturation > 0 else None),
        "remanence_after_long_time": float(np.exp(-60.0)),
    }
    ok = (saturation > 0
          and evidence["saturation_relative_error"] < 1e-3
          and evidence["susceptibility_without_particles"] == 0.0
          and evidence["high_frequency_susceptibility_ratio"] < 1e-3
          and evidence["weak_field_linearity_error"] < 1e-6)
    if np.isfinite(tau.max()) and saturation > 0:
        evidence["remanence_after_long_time"] = float(
            ensemble.remanence(np.array([min(60 * tau.max(), 1e300)]), 1e3)[0])
        ok = ok and evidence["remanence_after_long_time"] < 1e-12
    return _check("boundary_conditions", _passed(ok),
                  "saturation, empty fluid, high frequency, weak field and long time limits",
                  evidence)


def _conservation(device: FerrofluidDevice, writes: list[WritePulse],
                  frequencies: list[float]) -> dict[str, Any]:
    currents = device.current({device.index(w.port): w.voltage_v for w in writes})
    duration = (float(5 * device.tau_concentration)
                if np.isfinite(device.tau_concentration) else 1e-6)
    state = device.evolve(device.initial_state(), currents, duration)
    drift = np.abs(device.volume @ state.concentration) / device.volume.sum()
    freqs = np.unique(np.concatenate([np.logspace(1, 8, 8), frequencies]))
    z = device.impedance_matrix(freqs, currents, state)
    smallest = min(float(np.linalg.eigvalsh((m + m.conj().T) / 2).min()) for m in z)
    fluid = device.fluid_inductance(freqs, currents, state.concentration)
    loss = min(float(np.linalg.eigvalsh(-f.imag).min() / max(np.abs(f).max(), 1e-300))
               for f in fluid)
    asymmetry = float(np.abs(z - np.swapaxes(z, 1, 2)).max() / np.abs(z).max())
    evidence = {"particle_number_drift": float(drift.max()),
                "smallest_hermitian_eigenvalue_ohm": smallest,
                "smallest_relative_fluid_loss_eigenvalue": loss,
                "reciprocity_asymmetry": asymmetry}
    ok = drift.max() < 1e-9 and smallest >= 0 and loss > -1e-9 and asymmetry < 1e-12
    return _check("conservation", _passed(ok),
                  "particle number conserved; impedance passive and reciprocal", evidence)


def _validity(result: dict[str, Any]) -> dict[str, Any]:
    problems = [w for w in result["warnings"] if not w.startswith("immobilised particles")]
    problems += [f"read {r['port']}: {r['warning']}" for r in result["reads"] if "warning" in r]
    return _check("model_validity", _passed(not problems),
                  "the device stays inside the stated validity range of every equation",
                  problems or "no validity warning")


def _established(device: FerrofluidDevice, assumed_operation: list[str]) -> dict[str, Any]:
    rows = {r["name"]: r["provenance"] for r in device.params.describe()}
    assumed = [n for n in PARAMETER_TABLE
               if not rows[n].startswith(("user_supplied", "fitted"))]
    geometry = [f"port {p.id}: {', '.join(p.assumed)}" for p in device.ports if p.assumed]
    geometry += device.region.assumed
    evidence = {"assumed_material_parameters": assumed, "assumed_geometry": geometry,
                "assumed_operating_conditions": assumed_operation}
    return _check("inputs_established", _passed(not (assumed or geometry or assumed_operation)),
                  "every material parameter, dimension and operating condition is supplied "
                  "or fitted, none assumed", evidence)


# ---------------------------------------------------------------------------
# Experimental agreement
# ---------------------------------------------------------------------------


def _agreement(observed: np.ndarray, predicted: np.ndarray, sigma: np.ndarray,
               points: int) -> tuple[bool, dict[str, Any]]:
    residual = (predicted - observed) / sigma
    chi = float(np.mean(residual**2))
    inside = float(np.mean(np.abs(residual) <= 2.0))
    rms = float(np.sqrt(np.mean(observed**2)))
    relative = float(np.median(sigma) / rms) if rms > 0 else float("inf")
    problems = []
    if points < MIN_POINTS:
        problems.append(f"fewer than {MIN_POINTS} points")
    if relative > MAX_RELATIVE_SIGMA:
        problems.append("uncertainties too large (or signal too small) to test the model")
    if chi > MAX_REDUCED_CHI_SQUARE:
        problems.append("model disagrees with the data beyond the stated uncertainty")
    if inside < MIN_FRACTION_WITHIN_2_SIGMA:
        problems.append("too few points within 2 sigma")
    return not problems, {
        "points": points, "reduced_chi_square": chi, "fraction_within_2_sigma": inside,
        "median_sigma_over_rms_signal": relative, "problems": problems}


def _material_dataset(data: dict, device: FerrofluidDevice
                      ) -> tuple[bool, dict[str, Any], _Dataset]:
    measured = _Dataset(data)
    if not measured.weighted:
        return False, {"problems": ["measurement uncertainties not supplied"]}, measured
    predicted = measured.predict(device.params)
    ok, evidence = _agreement(measured.observed, predicted, measured.sigma, len(measured.x))
    if (measured.temperature is not None and abs(
            float(measured.temperature) - device.params.temperature_k)
            > MAX_TEMPERATURE_MISMATCH_K):
        ok = False
        evidence["problems"].append("measured at a different temperature from the device")
    return ok, evidence, measured


def _port_dataset(data: dict, device: FerrofluidDevice) -> tuple[bool, dict[str, Any]]:
    interface.check_keys(data, set(PORT_DATA_COLUMNS) | {"kind", "port", "to_port",
                                                         "temperature_k"},
                         "port_impedance_change dataset")
    if "port" not in data:
        raise ValueError("port_impedance_change needs 'port'")
    i = device.index(data["port"])
    j = device.index(data.get("to_port", data["port"]))
    frequency = _column(data, "frequency_hz")
    n = len(frequency)
    missing = [c for c in PORT_DATA_COLUMNS[3:] if c not in data]
    if missing:
        return False, {"problems": ["measurement uncertainties not supplied"]}
    observed = np.concatenate([_column(data, c, n) for c in PORT_DATA_COLUMNS[1:3]])
    sigma = np.concatenate([_column(data, c, n) for c in PORT_DATA_COLUMNS[3:]])
    if np.any(sigma <= 0) or np.any(frequency <= 0):
        raise ValueError("frequencies and uncertainties must be positive")
    change = (2j * np.pi * frequency * device.fluid_inductance(
        frequency, np.zeros(len(device.ports)))[:, i, j])
    ok, evidence = _agreement(observed, np.concatenate([change.real, change.imag]), sigma, n)
    if (data.get("temperature_k") is not None and abs(
            float(data["temperature_k"]) - device.params.temperature_k)
            > MAX_TEMPERATURE_MISMATCH_K):
        ok = False
        evidence["problems"].append("measured at a different temperature from the device")
    return ok, evidence


def _experiments(device: FerrofluidDevice, measurements: dict | None, calibration: set[str],
                 peak_field: float, reads: list[ReadEvent]) -> list[dict[str, Any]]:
    groups = {
        "experiment_equilibrium": (
            "magnetization measurements agree within uncertainty and reach the device's "
            "peak field", ("magnetization_curve",)),
        "experiment_dynamics": (
            "susceptibility spectrum covering the read frequencies, or magnetization "
            "decay, agrees within uncertainty",
            ("susceptibility_spectrum", "magnetization_decay")),
        "experiment_port_response": (
            "measured fluid-induced port impedance (with minus without fluid) agrees within "
            "uncertainty for every read port and frequency range", ("port_impedance_change",)),
    }
    measurements = measurements or {}
    interface.check_keys(measurements, {"provenance", "datasets"}, "validation_measurements")
    datasets = measurements.get("datasets") or []
    source = str((measurements.get("provenance") or {}).get("source", "")).strip()
    known = set(DATASET_COLUMNS) | {"port_impedance_change"}
    for data in datasets:
        if data.get("kind") not in known:
            raise ValueError(f"dataset kind must be one of {sorted(known)}")
    out = []
    frequencies = [r.frequency_hz for r in reads]
    for identifier, (requirement, kinds) in groups.items():
        chosen = [d for d in datasets if d["kind"] in kinds]
        if not chosen or not source:
            reason = ("no user measurements of this kind supplied" if not chosen
                      else "measurements have no provenance source")
            out.append(_check(identifier, "not_run", requirement, reason))
            continue
        evidence: dict[str, Any] = {"source": source, "datasets": [],
                                    "authenticity": "not verifiable by ChipLab"}
        ok, covered, ports = True, False, set()
        for data in chosen:
            fingerprint = dataset_fingerprint(data)
            if data["kind"] == "port_impedance_change":
                good, detail = _port_dataset(data, device)
                f = np.asarray(data["frequency_hz"], dtype=float)
                if f.min() <= min(frequencies) and f.max() >= max(frequencies):
                    ports.add(device.ports[device.index(
                        data.get("to_port", data["port"]))].id)
                    ports.add(device.ports[device.index(data["port"])].id)
            else:
                good, detail, measured = _material_dataset(data, device)
                if data["kind"] == "magnetization_curve":
                    covered |= bool(np.abs(measured.x).max() >= peak_field)
                elif data["kind"] == "susceptibility_spectrum":
                    covered |= bool(measured.x.min() <= min(frequencies)
                                    and measured.x.max() >= max(frequencies))
                else:
                    covered = True
            if fingerprint in calibration:
                good = False
                detail.setdefault("problems", []).append(
                    "this dataset was used for calibration: not an independent test")
            ok &= good
            evidence["datasets"].append(
                {"kind": data["kind"], "sha256": fingerprint, "agrees": bool(good), **detail})
        if identifier == "experiment_port_response":
            missing = sorted({device.ports[device.index(r.port)].id for r in reads} - ports)
            covered = not missing
            evidence["read_ports_without_covering_data"] = missing
        evidence["covers_operating_range"] = bool(covered)
        out.append(_check(identifier, _passed(ok and covered), requirement, evidence))
    return out


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def _chiplet_numbers(device: FerrofluidDevice, writes: list[WritePulse]) -> dict[str, float]:
    """The only chiplet quantities the model supports: conductor area and drive power."""
    low, high = np.full(2, np.inf), np.full(2, -np.inf)
    for port in device.ports:
        start, end = port.segments(48)
        points = np.vstack([start, end])[:, :2] * 1e6
        low = np.minimum(low, points.min(axis=0) - port.wire_radius_um)
        high = np.maximum(high, points.max(axis=0) + port.wire_radius_um)
    power = 0.0
    for t in {w.start_s for w in writes}:
        active = sum(w.voltage_v**2 / device.resistance[device.index(w.port)]
                     for w in writes if w.start_s <= t < w.end_s)
        power = max(power, float(active))
    return {"area_mm2": float(np.prod((high - low) * 1e-3)), "tdp_watts": power}

_CACHE: dict[str, dict[str, Any]] = {}


def _device_from(record: dict) -> FerrofluidDevice:
    spec = record.get("device_spec") or {}
    array = record.get("electrode_array") or {}
    if not interface.routes_to_ferrofluid(spec, array):
        raise ValueError("only ferrofluid devices have physics checks")
    interface.validate_device_spec(spec)
    return interface.build_device(interface.ports_from_array(array), array, spec)


def _calibration_fingerprints(spec: dict) -> set[str]:
    out: set[str] = set()
    for source in (spec, spec.get("parameters"), spec.get("ferrofluid")):
        if isinstance(source, dict) and source.get("calibration"):
            out |= set(source["calibration"].get("provenance", {}).get("dataset_sha256", []))
    return out


def _undefined_report(key: str, error: str) -> dict[str, Any]:
    """Report for a record from which no device can be built: nothing else can be run."""
    reason = "the model cannot represent this device"
    checks = [_check("device_definition", "fail",
                     "the record defines a device inside the model's accepted ranges", error)]
    checks += [_check(i, "not_run", reason, reason) for i in CHECK_IDS]
    return {
        "accepted": False,
        "rule": "accepted only if every check passes; not_run is never a pass",
        "summary": {"pass": 0, "fail": 1, "not_run": len(CHECK_IDS)},
        "checks": checks, "record_sha256": key, "chiplet_numbers": None,
        "model_version": MODEL_VERSION, "limitations": [],
    }


def evaluate_record(record: dict) -> dict[str, Any]:
    """Run every check on a device record and decide whether it is accepted."""
    interface.check_keys(record, RECORD_KEYS, "device record")
    key = hashlib.sha256(
        (MODEL_VERSION + json.dumps(record, sort_keys=True, default=str)).encode()).hexdigest()
    if key in _CACHE:
        return copy.deepcopy(_CACHE[key])
    try:
        device = _device_from(record)
    except ValueError as error:
        return _undefined_report(key, str(error))
    assumed: list[str] = []
    ignored: list[str] = []
    writes = interface.parse_writes(record.get("write_sequence") or [], device, assumed,
                                    (record.get("device_spec") or {}).get("write_voltage_v"))
    reads = []
    if record.get("read_sequence"):
        reads = interface.parse_reads(record["read_sequence"], device, assumed, ignored)
    assumed = [a for a in assumed if "at the instant the last write ends" not in a
               and "bandwidth_hz" not in a]
    if not writes or not reads or all(w.voltage_v == 0 for w in writes):
        reason = "needs a write_sequence with a non-zero voltage and a read_sequence"
        checks = [_check(i, "not_run", reason, reason) for i in PHYSICS_IDS]
        checks.append(_established(device, assumed or [reason]))
        checks += [_check(i, "not_run", reason, reason) for i in EXPERIMENT_IDS]
        peak = 0.0
    else:
        writes = [w for w in writes if w.voltage_v != 0]
        result = simulate_sequence(device, writes, reads, interaction_analysis=False)
        peak = max(w["peak_field_in_fluid_a_m"] for w in result["writes"])
        grid = device.grid_convergence()
        checks = [
            _zero_input(device, writes),
            _sensitivity(device, writes),
            _units(device, writes),
            _boundaries(device),
            _conservation(device, writes, [r.frequency_hz for r in reads]),
            _check("numerical_convergence", _passed(grid <= MAX_GRID_CHANGE),
                   f"fluid inductance changes by at most {MAX_GRID_CHANGE:.0%} on a finer grid",
                   {"relative_change": grid}),
            _validity(result),
            _established(device, assumed),
        ]
        checks += _experiments(
            device, record.get("validation_measurements"),
            _calibration_fingerprints(record.get("device_spec") or {}), peak, reads)
    counts = {s: sum(c["status"] == s for c in checks) for s in ("pass", "fail", "not_run")}
    report = {
        "accepted": counts["fail"] == 0 and counts["not_run"] == 0,
        "rule": "accepted only if every check passes; not_run is never a pass",
        "summary": counts,
        "checks": checks,
        "record_sha256": key,
        "chiplet_numbers": _chiplet_numbers(device, writes),
        "model_version": MODEL_VERSION,
        "limitations": [
            "physics checks test the model's self-consistency, not its truth",
            "measurement authenticity cannot be verified",
            "port measurements test the unwritten (zero-bias) fluid only: written-state "
            "dynamics, retention and port interaction are not experimentally checked",
            "calibration independence is detected only for identical datasets",
            "thresholds are choices made in ferro_validation.py, not industry standards",
        ],
    }
    if len(_CACHE) >= 32:
        _CACHE.clear()
    _CACHE[key] = copy.deepcopy(report)
    return report


def chiplet_problems(chiplet: Any, record: dict) -> list[str]:
    """Why a chiplet carrying this record is not acceptable (empty list if it is)."""
    try:
        report = evaluate_record(record)
        if not report["accepted"]:
            return [f"check {c['id']} is {c['status']}" for c in report["checks"]
                    if c["status"] != "pass"]
        numbers = report["chiplet_numbers"]
    except (ValueError, KeyError, TypeError) as error:
        return [f"device record is invalid: {error}"]
    problems = []
    if str(chiplet.type) != "material":
        problems.append("a validated material device must have type 'material'")
    if chiplet.memory is not None or chiplet.compute is not None:
        problems.append("capacity, bandwidth, latency, energy and compute rate are not "
                        "modelled: a material chiplet cannot declare memory or compute specs")
    if chiplet.process_nm is not None:
        problems.append("process_nm is not modelled for a material device")
    for name, expected in numbers.items():
        declared = getattr(chiplet, name)
        if declared is None or not np.isclose(float(declared), expected, rtol=1e-6, atol=0):
            problems.append(f"{name} must equal the computed value {expected:.6g}")
    return problems
