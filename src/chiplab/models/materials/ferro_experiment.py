"""Measurements for the electrical state-space model: import, compare, fit.

Nothing here contains or generates experimental data.  A measurement record is

    {"provenance": {"source": ...},
     "protocol": {"segments": [...]},          what was applied, with timing and temperature
     "datasets": [ ... ]}

with datasets (every one needs the measurement uncertainty):

    {"kind": "impedance", "to_port", "from_port", "time_s", "frequency_hz",
     "z_real_ohm", "z_imag_ohm", "sigma_z_real_ohm", "sigma_z_imag_ohm"}     rows
    {"kind": "summed_impedance", "to_port", "from_port", "time_s", "zc_ohm",
     "sigma_zc_ohm", "frequency_grid_hz", "definition"}        sum over all frequencies
    {"kind": "current", "port", "time_s", "current_a", "sigma_current_a"}

Any dataset may carry "reference_item" (see ferro_reference).
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from chiplab.models.materials.ferro_statespace import (
    MODEL_VERSION,
    PARAMETERS,
    StateSpaceDevice,
    run_protocol,
)
from chiplab.models.materials.ferro_validation import (
    MAX_REDUCED_CHI_SQUARE,
    MAX_RELATIVE_SIGMA,
    MIN_FRACTION_WITHIN_2_SIGMA,
    MIN_POINTS,
)

COLUMNS = {
    "impedance": (["time_s", "frequency_hz", "z_real_ohm", "z_imag_ohm"],
                  ["sigma_z_real_ohm", "sigma_z_imag_ohm"]),
    "summed_impedance": (["time_s", "zc_ohm"], ["sigma_zc_ohm"]),
    "current": (["time_s", "current_a"], ["sigma_current_a"]),
}
ZC_DEFINITIONS = ("sum_of_magnitudes", "sum_of_real_parts")
META = {
    "impedance": ["to_port", "from_port"],
    "summed_impedance": ["to_port", "from_port", "frequency_grid_hz", "definition"],
    "current": ["port"],
}
MAX_ROWS = 200_000


# ---------------------------------------------------------------------------
# File import
# ---------------------------------------------------------------------------


def read_csv_columns(path: str | Path) -> dict[str, list[float]]:
    """A CSV file with a header row of column names and numeric values."""
    with open(path, newline="") as handle:
        rows = list(csv.reader(line for line in handle if not line.startswith("#")))
    if len(rows) < 2:
        raise ValueError(f"{path}: needs a header row and data rows")
    names = [n.strip() for n in rows[0]]
    columns: dict[str, list[float]] = {n: [] for n in names}
    for number, row in enumerate(rows[1:], start=2):
        if not row:
            continue
        if len(row) != len(names):
            raise ValueError(f"{path}: row {number} has {len(row)} values, expected "
                             f"{len(names)}")
        for name, value in zip(names, row, strict=True):
            try:
                columns[name].append(float(value))
            except ValueError as error:
                raise ValueError(f"{path}: row {number}, column {name}: not a number "
                                 f"{value!r}") from error
    return columns


def protocol_from_trace(time_s: list[float], voltages_v: dict[str, list[float | None]],
                        temperature_k: list[float] | float | None = None) -> dict[str, Any]:
    """Zero-order-hold protocol from a recorded drive trace.

    Sample k is held from time_s[k] to time_s[k+1]; the final sample only marks the end.
    The protocol clock starts at time_s[0].  None means the port was floating.
    """
    t = np.asarray(time_s, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.all(np.diff(t) > 0):
        raise ValueError("time_s must hold at least two strictly increasing times")
    for port, values in voltages_v.items():
        if len(values) != len(t):
            raise ValueError(f"voltage trace of {port} does not match time_s")
    if isinstance(temperature_k, (list, tuple)) and len(temperature_k) != len(t):
        raise ValueError("temperature_k trace does not match time_s")
    segments: list[dict[str, Any]] = []
    for k in range(len(t) - 1):
        volts = {p: (None if v[k] is None else float(v[k])) for p, v in voltages_v.items()}
        segment: dict[str, Any] = {"duration_s": float(t[k + 1] - t[k]), "voltages_v": volts}
        if temperature_k is not None:
            segment["temperature_k"] = float(
                temperature_k[k] if isinstance(temperature_k, (list, tuple))
                else temperature_k)
        previous = segments[-1] if segments else None
        if previous and previous["voltages_v"] == volts and previous.get(
                "temperature_k") == segment.get("temperature_k"):
            previous["duration_s"] += segment["duration_s"]
        else:
            segments.append(segment)
    return {"segments": segments}


def normalise_protocol(protocol: dict) -> dict[str, Any]:
    """Accept either {"segments": ...} or a drive trace {"time_s", "voltages_v", ...}."""
    if isinstance(protocol, dict) and "time_s" in protocol:
        extra = set(protocol) - {"time_s", "voltages_v", "temperature_k"}
        if extra or "voltages_v" not in protocol:
            raise ValueError("a drive trace has time_s, voltages_v and optionally "
                             "temperature_k")
        return protocol_from_trace(**protocol)
    return protocol


_UNITS = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}


def parse_touchstone(text: str, n_ports: int) -> dict[str, Any]:
    """Touchstone (version 1) S-parameter text -> frequencies, S[f, to, from], Z0."""
    if not 1 <= n_ports <= 16:
        raise ValueError("n_ports must be 1..16")
    unit, fmt, z0, seen_option = 1e9, "ma", 50.0, False
    numbers: list[float] = []
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            raise ValueError("Touchstone version 2 keywords are not supported")
        if line.startswith("#"):
            if seen_option:
                continue
            seen_option = True
            tokens = line[1:].lower().split()
            for i, token in enumerate(tokens):
                if token in _UNITS:
                    unit = _UNITS[token]
                elif token in ("ma", "db", "ri"):
                    fmt = token
                elif token == "r" and i + 1 < len(tokens):
                    z0 = float(tokens[i + 1])
                elif token in ("y", "z", "g", "h"):
                    raise ValueError("only S-parameter files are supported")
            continue
        numbers.extend(float(x) for x in line.split())
    width = 1 + 2 * n_ports**2
    if not numbers or len(numbers) % width:
        raise ValueError(f"data do not divide into {width} values per frequency: wrong "
                         "n_ports, noise data or a damaged file")
    table = np.array(numbers).reshape(-1, width)
    first, second = table[:, 1::2], table[:, 2::2]
    if fmt == "ri":
        values = first + 1j * second
    else:
        magnitude = 10 ** (first / 20) if fmt == "db" else first
        values = magnitude * np.exp(1j * np.deg2rad(second))
    s = values.reshape(-1, n_ports, n_ports)
    if n_ports == 2:  # two-port files list S11 S21 S12 S22
        s = s.transpose(0, 2, 1)
    f = table[:, 0] * unit
    if np.any(np.diff(f) <= 0):
        raise ValueError("frequencies must be strictly increasing")
    return {"frequency_hz": f, "s": s, "z0_ohm": z0}


def s_to_z(s: np.ndarray, z0_ohm: float) -> np.ndarray:
    """Z = Z0 (I + S)(I - S)^-1 for equal real reference impedances."""
    identity = np.eye(s.shape[-1])
    return z0_ohm * np.linalg.solve(identity - s, identity + s)


def touchstone_datasets(text: str, ports: list[str], time_s: float,
                        sigma_ohm: float | None = None, sigma_relative: float | None = None,
                        reference_item: str | None = None) -> list[dict[str, Any]]:
    """One VNA sweep taken at `time_s` as "impedance" datasets, one per matrix entry.

    The measurement uncertainty must be supplied (absolute ohms and/or a fraction of |Z|).
    """
    if sigma_ohm is None and sigma_relative is None:
        raise ValueError("state the uncertainty: sigma_ohm and/or sigma_relative")
    parsed = parse_touchstone(text, len(ports))
    z = s_to_z(parsed["s"], parsed["z0_ohm"])
    out = []
    for i, to in enumerate(ports):
        for j, source in enumerate(ports):
            entry = z[:, i, j]
            sigma = np.hypot(sigma_ohm or 0.0, (sigma_relative or 0.0) * np.abs(entry))
            data: dict[str, Any] = {
                "kind": "impedance", "to_port": to, "from_port": source,
                "time_s": [float(time_s)] * len(entry),
                "frequency_hz": parsed["frequency_hz"].tolist(),
                "z_real_ohm": entry.real.tolist(), "z_imag_ohm": entry.imag.tolist(),
                "sigma_z_real_ohm": sigma.tolist(), "sigma_z_imag_ohm": sigma.tolist()}
            if reference_item:
                data["reference_item"] = reference_item
            out.append(data)
    return out


# ---------------------------------------------------------------------------
# Records, prediction, comparison
# ---------------------------------------------------------------------------


def dataset_fingerprint(data: dict) -> str:
    """SHA-256 of what was measured (uncertainties and labels excluded)."""
    kind = data["kind"]
    keys = META[kind] + COLUMNS[kind][0]
    values = {k: ([float(x) for x in data[k]] if isinstance(data[k], list) else str(data[k]))
              for k in keys}
    return hashlib.sha256(
        json.dumps({"kind": kind, **values}, sort_keys=True).encode()).hexdigest()


def _array(data: dict, name: str, length: int | None = None) -> np.ndarray:
    if name not in data:
        raise ValueError(f"dataset {data.get('kind')!r} needs the column {name!r}")
    values = np.asarray(data[name], dtype=float)
    if values.ndim != 1 or not 0 < len(values) <= MAX_ROWS or not np.all(
            np.isfinite(values)):
        raise ValueError(f"column {name!r} must be a non-empty list of finite numbers")
    if length is not None and len(values) != length:
        raise ValueError(f"column {name!r} has {len(values)} values, expected {length}")
    return values


class _Record:
    """A validated measurement record that can be predicted by a device."""

    def __init__(self, record: dict):
        if not isinstance(record, dict) or set(record) - {"provenance", "protocol",
                                                          "datasets"}:
            raise ValueError("a measurement record has provenance, protocol and datasets")
        source = (record.get("provenance") or {}).get("source")
        if not str(source or "").strip():
            raise ValueError("measurements need provenance={'source': ...} naming where "
                             "they came from (instrument, file, publication)")
        if not record.get("datasets"):
            raise ValueError("no datasets supplied")
        if not record.get("protocol"):
            raise ValueError("the applied protocol (voltages, timing, temperature) is "
                             "required")
        self.source = str(source)
        self.protocol = normalise_protocol(record["protocol"])
        self.datasets = []
        for data in record["datasets"]:
            kind = data.get("kind")
            if kind not in COLUMNS:
                raise ValueError(f"dataset kind must be one of {sorted(COLUMNS)}")
            values, sigmas = COLUMNS[kind]
            allowed = set(values + sigmas + META[kind]) | {"kind", "reference_item"}
            if set(data) - allowed:
                raise ValueError(f"{kind}: unknown field(s) {sorted(set(data) - allowed)}")
            for name in META[kind]:
                if name not in data:
                    raise ValueError(f"{kind} dataset needs {name!r}")
            time = _array(data, "time_s")
            columns = {n: _array(data, n, len(time)) for n in values + sigmas}
            if any(np.any(columns[n] <= 0) for n in sigmas):
                raise ValueError(f"{kind}: uncertainties must be positive")
            if kind == "summed_impedance":
                if data["definition"] not in ZC_DEFINITIONS:
                    raise ValueError(f"definition must be one of {ZC_DEFINITIONS}")
                _array(data, "frequency_grid_hz")
            self.datasets.append((data, columns))

    def predict(self, device: StateSpaceDevice) -> list[dict[str, np.ndarray]]:
        times = np.unique(np.concatenate([c["time_s"] for _, c in self.datasets]))
        snapshots = run_protocol(device, self.protocol, times.tolist())
        out = []
        for data, c in self.datasets:
            index = np.searchsorted(times, c["time_s"])
            kind = data["kind"]
            observed, sigma = [], []
            predicted = np.zeros(len(index))
            if kind == "current":
                p = device.index(data["port"])
                predicted = np.array([snapshots[k]["dc_current_a"][p] for k in index])
                observed, sigma = [c["current_a"]], [c["sigma_current_a"]]
                parts = [predicted]
            else:
                i, j = device.index(data["to_port"]), device.index(data["from_port"])
                if kind == "impedance":
                    value = np.zeros(len(index), complex)
                    for k in np.unique(index):
                        rows = index == k
                        s = snapshots[k]
                        value[rows] = device.impedance_matrix(
                            c["frequency_hz"][rows], s["state"], s["node_v"],
                            s["temperature_k"])[:, i, j]
                    parts = [value.real, value.imag]
                    observed = [c["z_real_ohm"], c["z_imag_ohm"]]
                    sigma = [c["sigma_z_real_ohm"], c["sigma_z_imag_ohm"]]
                else:
                    grid = np.asarray(data["frequency_grid_hz"], dtype=float)
                    totals = {}
                    for k in np.unique(index):
                        s = snapshots[k]
                        z = device.impedance_matrix(grid, s["state"], s["node_v"],
                                                    s["temperature_k"])[:, i, j]
                        totals[k] = float(np.sum(
                            np.abs(z) if data["definition"] == "sum_of_magnitudes"
                            else z.real))
                    parts = [np.array([totals[k] for k in index])]
                    observed, sigma = [c["zc_ohm"]], [c["sigma_zc_ohm"]]
            out.append({"predicted": np.concatenate(parts),
                        "observed": np.concatenate(observed),
                        "sigma": np.concatenate(sigma)})
        return out


def parse_records(measurements: dict | list) -> list[_Record]:
    items = measurements if isinstance(measurements, list) else [measurements]
    if not items or not items[0]:
        raise ValueError("no measurements supplied")
    return [_Record(r) for r in items]


def agreement(observed: np.ndarray, predicted: np.ndarray, sigma: np.ndarray) -> dict:
    """The same agreement rule as the validation gate (thresholds are ChipLab choices)."""
    weighted = (predicted - observed) / sigma
    rms = float(np.sqrt(np.mean(observed**2)))
    detail = {
        "points": int(len(observed)),
        "reduced_chi_square": float(np.mean(weighted**2)),
        "fraction_within_2_sigma": float(np.mean(np.abs(weighted) <= 2)),
        "rms_residual": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "max_abs_residual": float(np.max(np.abs(predicted - observed))),
        "median_relative_sigma": float(np.median(sigma) / rms) if rms else None,
    }
    problems = []
    if detail["points"] < MIN_POINTS:
        problems.append(f"fewer than {MIN_POINTS} points")
    if detail["reduced_chi_square"] > MAX_REDUCED_CHI_SQUARE:
        problems.append(f"reduced chi-square above {MAX_REDUCED_CHI_SQUARE}")
    if detail["fraction_within_2_sigma"] < MIN_FRACTION_WITHIN_2_SIGMA:
        problems.append(f"less than {MIN_FRACTION_WITHIN_2_SIGMA:.0%} within 2 sigma")
    if not rms or detail["median_relative_sigma"] > MAX_RELATIVE_SIGMA:
        problems.append("uncertainty too large compared with the signal to test anything")
    detail["agrees"] = not problems
    detail["problems"] = problems
    return detail


def _calibration_hashes(model: dict) -> set[str]:
    fit = (model.get("provenance") or {}).get("fit") or {}
    return set(fit.get("dataset_sha256", []))


def compare(model: dict, measurements: dict | list, include_points: bool = False
            ) -> dict[str, Any]:
    """Simulated against measured, dataset by dataset."""
    device = StateSpaceDevice(model)
    used = _calibration_hashes(model)
    rows = []
    for r, record in enumerate(parse_records(measurements)):
        for (data, c), result in zip(record.datasets, record.predict(device), strict=True):
            fingerprint = dataset_fingerprint(data)
            row: dict[str, Any] = {
                "record": r, "kind": data["kind"], "source": record.source,
                "reference_item": data.get("reference_item"), "sha256": fingerprint,
                "independent_of_calibration": fingerprint not in used,
                **{k: data[k] for k in ("to_port", "from_port", "port") if k in data},
                **agreement(result["observed"], result["predicted"], result["sigma"])}
            if include_points:
                row["points_data"] = {"time_s": c["time_s"].tolist(),
                                      "observed": result["observed"].tolist(),
                                      "predicted": result["predicted"].tolist()}
            rows.append(row)
    independent = [r for r in rows if r["independent_of_calibration"]]
    return {
        "datasets": rows,
        "all_agree": all(r["agrees"] for r in rows),
        "independent_datasets": len(independent),
        "independent_agreement": (all(r["agrees"] for r in independent)
                                  if independent else None),
        "rule": f"reduced chi-square <= {MAX_REDUCED_CHI_SQUARE}, at least "
                f"{MIN_FRACTION_WITHIN_2_SIGMA:.0%} within 2 sigma, at least {MIN_POINTS} "
                f"points, median sigma <= {MAX_RELATIVE_SIGMA:.0%} of the rms signal",
        "limitations": ["independence is detected only for identical datasets",
                        "measurement authenticity cannot be verified"],
    }


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

_LOG_UNITS = ("ohm", "s", "F", "V", "K")
_PATH = re.compile(r"^(ground|coupling|branches\.[^.]+)\.([a-z0-9_]+)$")


def _locate(model: dict, path: str) -> tuple[dict, str]:
    match = _PATH.match(path)
    if not match:
        raise ValueError(f"parameter path {path!r} must look like 'ground.r_inf_ohm' or "
                         "'branches.p1|p2.delta_r_ohm'")
    group, key = match.groups()
    if key not in PARAMETERS or key == "reference_temperature_k":
        raise ValueError(f"cannot fit {key!r}")
    target: Any = model
    for part in group.split("."):
        target = target.get(part) if isinstance(target, dict) else None
    if not isinstance(target, dict) or key not in target or isinstance(target[key], str):
        raise ValueError(f"{path}: give a numeric starting value in the model")
    return target, key


def fit(model: dict, measurements: dict | list, parameters_to_fit: list,
        max_evaluations: int = 200) -> dict[str, Any]:
    """Weighted least squares of chosen model parameters to the user's measurements.

    parameters_to_fit: paths, or {"path", "lower", "upper"}.  Starting values come from
    the model.  Positive quantities are fitted in their logarithm.
    """
    if not parameters_to_fit:
        raise ValueError("parameters_to_fit is empty")
    records = parse_records(measurements)
    StateSpaceDevice(model)  # reject a bad starting model before fitting
    items = [{"path": p} if isinstance(p, str) else dict(p) for p in parameters_to_fit]
    if len({i["path"] for i in items}) != len(items):
        raise ValueError("a parameter is listed twice")
    start, low, high, logs = [], [], [], []
    for item in items:
        target, key = _locate(model, item["path"])
        unit, floor, ceiling, _ = PARAMETERS[key]
        value = float(target[key])
        lower = max(float(item.get("lower", floor)), floor)
        upper = min(float(item.get("upper", ceiling)), ceiling)
        log = unit in _LOG_UNITS and value > 0
        if log:
            lower = max(lower, 1e-30)
        if not lower < upper or not lower <= value <= upper:
            raise ValueError(f"{item['path']}: need lower <= starting value <= upper")
        convert = np.log if log else float
        start.append(float(convert(value)))
        low.append(float(convert(lower)))
        high.append(float(convert(upper)))
        logs.append(log)

    def build(theta: np.ndarray) -> dict:
        candidate = copy.deepcopy(model)
        for item, value, log in zip(items, theta, logs, strict=True):
            target, key = _locate(candidate, item["path"])
            target[key] = float(np.exp(value) if log else value)
        return candidate

    def residuals(theta: np.ndarray) -> np.ndarray:
        device = StateSpaceDevice(build(theta))
        parts = [(r["predicted"] - r["observed"]) / r["sigma"]
                 for record in records for r in record.predict(device)]
        return np.concatenate(parts)

    n_data = len(residuals(np.array(start)))
    if n_data < len(items) + 2:
        raise ValueError(f"{n_data} data values cannot constrain {len(items)} parameters")
    solution = least_squares(
        residuals, start, bounds=(low, high), method="trf", x_scale="jac",
        max_nfev=max_evaluations, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    theta, r, jac = solution.x, solution.fun, solution.jac
    chi = float(r @ r)
    dof = n_data - len(items)
    reduced = chi / dof

    norms = np.linalg.norm(jac, axis=0)
    insensitive = norms < 1e-6 * max(float(norms.max()), 1.0)
    scaled = jac / np.where(insensitive, 1.0, norms)
    scaled[:, insensitive] = 0.0
    _, singular, vt = np.linalg.svd(scaled, full_matrices=False)
    degenerate = singular < 1e-4 * max(float(singular.max()), 1e-300)
    unidentified = insensitive.copy()
    for vector in vt[degenerate]:
        unidentified |= np.abs(vector) > 0.1
    inverse = np.where(degenerate, 0.0, 1.0 / np.maximum(singular, 1e-300) ** 2)
    safe = np.where(insensitive, 1.0, norms)
    covariance = (vt.T * inverse) @ vt / np.outer(safe, safe) * max(reduced, 1.0)
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    results: dict[str, Any] = {}
    poor = []
    for k, item in enumerate(items):
        path, log = item["path"], logs[k]
        value = float(np.exp(theta[k]) if log else theta[k])
        at_bound = bool(solution.active_mask[k] != 0)
        limit = 1.0 if log else max(abs(value), 1.0)
        identifiable = bool(not unidentified[k] and not at_bound and errors[k] < limit)
        entry: dict[str, Any] = {"value": value, "unit": PARAMETERS[path.split(".")[-1]][0],
                                 "at_bound": at_bound, "identifiable": identifiable}
        if identifiable:
            interval = [theta[k] - 1.96 * errors[k], theta[k] + 1.96 * errors[k]]
            entry["standard_error"] = float(value * errors[k] if log else errors[k])
            entry["ci95"] = [float(np.exp(x) if log else x) for x in interval]
        else:
            entry.update(standard_error=None, ci95=None)
            poor.append(path)
        results[path] = entry
    denominator = np.outer(errors, errors)
    correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance),
                            where=denominator > 0)
    warnings = []
    if poor:
        warnings.append(f"the data do not determine {poor}")
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            if abs(correlation[a, b]) > 0.98:
                warnings.append(f"{items[a]['path']} and {items[b]['path']} are strongly "
                                f"correlated ({correlation[a, b]:+.3f})")
    if reduced > 3.0:
        warnings.append(f"reduced chi-square {reduced:.1f}: the model does not explain the "
                        "data within the stated uncertainties")
    converged = bool(solution.status > 0)
    if not converged:
        warnings.append("the optimiser did not converge")
    hashes = [dataset_fingerprint(d) for record in records for d, _ in record.datasets]
    fitted = build(theta)
    previous = _calibration_hashes(model)
    fitted["provenance"] = {**(model.get("provenance") or {}), "fit": {
        "sources": [record.source for record in records],
        "dataset_sha256": sorted(previous | set(hashes)),
        "parameters": [i["path"] for i in items], "reduced_chi_square": reduced,
        "model_version": MODEL_VERSION}}
    status = ("failed" if not converged else
              "fitted_with_reservations" if poor or reduced > 3.0 else
              "fitted_to_user_data")
    return {
        "status": status, "independently_validated": False,
        "parameters": results,
        "correlation": {items[a]["path"]: {items[b]["path"]: float(correlation[a, b])
                                           for b in range(len(items))}
                        for a in range(len(items))},
        "fit_quality": {"chi_square": chi, "degrees_of_freedom": dof,
                        "reduced_chi_square": reduced, "converged": converged},
        "model": fitted, "warnings": warnings,
        "method": "weighted trust-region least squares (scipy), linearised covariance",
        "validity": "only for this sample, fixture and the ranges covered by the data",
    }
