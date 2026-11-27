"""Fit ferrofluid particle parameters to measurements supplied by the user.

Nothing here contains or generates experimental data.  A fit needs data arrays and a
stated source; the result records both (with a SHA-256 of the data), the parameters held
fixed, standard errors from the linearised least-squares covariance, and which
parameters the data cannot determine.  A fit is not an independent validation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from chiplab.models.materials.ferro_particles import (
    MODEL_VERSION,
    PARAMETER_TABLE,
    FerrofluidParams,
    ParticleEnsemble,
)

FITTABLE = [n for n in PARAMETER_TABLE if n != "temperature_k"]

DATASET_COLUMNS = {
    "magnetization_curve": ["H_a_per_m", "M_a_per_m"],
    "susceptibility_spectrum": ["frequency_hz", "chi_real", "chi_loss"],
    "magnetization_decay": ["time_s", "m_normalized"],
}
SIGMA_COLUMNS = {
    "magnetization_curve": ["sigma_M_a_per_m"],
    "susceptibility_spectrum": ["sigma_chi_real", "sigma_chi_loss"],
    "magnetization_decay": ["sigma_m_normalized"],
}


def dataset_fingerprint(data: dict) -> str:
    """SHA-256 of a dataset's measured values (kind and non-uncertainty columns)."""
    values = {k: ([float(x) for x in v] if isinstance(v, list) else v)
              for k, v in data.items() if not k.startswith("sigma_")}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _column(data: dict, name: str, length: int | None = None) -> np.ndarray:
    if name not in data:
        raise ValueError(f"dataset {data.get('kind')!r} needs the column {name!r}")
    arr = np.asarray(data[name], dtype=float)
    if arr.ndim != 1 or len(arr) == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"column {name!r} must be a non-empty list of finite numbers")
    if length is not None and len(arr) != length:
        raise ValueError(f"column {name!r} has {len(arr)} values, expected {length}")
    return arr


class _Dataset:
    """One measured data set: observed values, weights and a model predictor."""

    def __init__(self, data: dict):
        kind = data.get("kind")
        if kind not in DATASET_COLUMNS:
            raise ValueError(f"dataset kind must be one of {list(DATASET_COLUMNS)}, got {kind!r}")
        allowed = set(DATASET_COLUMNS[kind]) | set(SIGMA_COLUMNS[kind]) | {"kind"}
        extras = {"temperature_k", "bias_field_a_per_m", "initial_field_a_per_m"}
        unknown = set(data) - allowed - extras
        if unknown:
            raise ValueError(f"unknown dataset key(s) {sorted(unknown)} for {kind}")
        self.kind = kind
        self.x = _column(data, DATASET_COLUMNS[kind][0])
        n = len(self.x)
        observed = [_column(data, c, n) for c in DATASET_COLUMNS[kind][1:]]
        self.temperature = data.get("temperature_k")
        self.bias = float(data.get("bias_field_a_per_m", 0.0))
        self.initial_field = None
        if kind == "magnetization_decay":
            if "initial_field_a_per_m" not in data:
                raise ValueError("magnetization_decay needs initial_field_a_per_m")
            self.initial_field = float(data["initial_field_a_per_m"])
        self.observed = np.concatenate(observed)
        sigmas = SIGMA_COLUMNS[kind]
        given = [c in data for c in sigmas]
        if any(given) and not all(given):
            raise ValueError(f"supply all of {sigmas} or none of them")
        if all(given):
            self.sigma = np.concatenate([_column(data, c, n) for c in sigmas])
            if np.any(self.sigma <= 0):
                raise ValueError("measurement uncertainties must be positive")
            self.weighted = True
        else:
            scale = float(np.sqrt(np.mean(self.observed**2)))
            if scale == 0:
                raise ValueError("observed data are all zero")
            self.sigma = np.full(len(self.observed), 0.01 * scale)
            self.weighted = False
        self.ranges = {DATASET_COLUMNS[kind][0]: [float(self.x.min()), float(self.x.max())],
                       "points": n}

    def predict(self, params: FerrofluidParams) -> np.ndarray:
        if self.temperature is not None:
            values = {n: getattr(params, n) for n in FerrofluidParams.names()}
            values["temperature_k"] = float(self.temperature)
            params = FerrofluidParams(**values)
        ensemble = ParticleEnsemble(params)
        if self.kind == "magnetization_curve":
            return np.sign(self.x) * ensemble.magnetization(np.abs(self.x))
        if self.kind == "susceptibility_spectrum":
            par, _ = ensemble.complex_susceptibility(self.x, abs(self.bias))
            return np.concatenate([par.real, -par.imag])
        return ensemble.remanence(self.x, abs(self.initial_field))


def fit_parameters(datasets: list[dict], parameters_to_fit: list[str],
                   provenance: dict[str, Any], fixed_parameters: dict[str, Any] | None = None,
                   max_iterations: int = 100) -> dict[str, Any]:
    """Weighted Levenberg-Marquardt fit of particle parameters (positive, fitted in log).

    Parameters
    ----------
    datasets : list of dicts, each with "kind" and its columns (see DATASET_COLUMNS).
    parameters_to_fit : names from FITTABLE.
    provenance : must contain a non-empty "source" describing where the data came from.
    fixed_parameters : values for parameters held constant (also used as starting values
        for fitted parameters).
    """
    if not isinstance(provenance, dict) or not str(provenance.get("source", "")).strip():
        raise ValueError("calibration needs provenance={'source': ...} naming where the "
                         "measurements came from (instrument, file, publication)")
    if not datasets:
        raise ValueError("no datasets supplied: calibration needs user measurements")
    if not parameters_to_fit:
        raise ValueError("parameters_to_fit is empty")
    unknown = sorted(set(parameters_to_fit) - set(FITTABLE))
    if unknown:
        raise ValueError(f"cannot fit {unknown}; fittable parameters: {FITTABLE}")
    names = list(dict.fromkeys(parameters_to_fit))
    sets = [_Dataset(d) for d in datasets]
    n_data = sum(len(s.observed) for s in sets)
    if n_data < len(names) + 2:
        raise ValueError(f"{n_data} data values cannot constrain {len(names)} parameters")
    base = FerrofluidParams.from_dict(fixed_parameters)
    base_values = {n: getattr(base, n) for n in FerrofluidParams.names()}
    low = np.array([max(PARAMETER_TABLE[n][4], 1e-4 if n in (
        "volume_fraction", "size_sigma_ln", "coating_nm", "anisotropy_j_m3") else 0.0)
        for n in names])
    high = np.array([PARAMETER_TABLE[n][5] for n in names])
    low = np.maximum(low, 1e-300)
    log_low, log_high = np.log(low), np.log(high)

    def residual(theta: np.ndarray) -> np.ndarray | None:
        values = dict(base_values)
        values.update({n: float(v) for n, v in zip(names, np.clip(np.exp(theta), low, high),
                                                   strict=True)})
        try:
            params = FerrofluidParams(**values)
        except ValueError:
            return None
        return np.concatenate([(s.predict(params) - s.observed) / s.sigma for s in sets])

    def jacobian(theta: np.ndarray, r0: np.ndarray) -> np.ndarray:
        jac = np.zeros((len(r0), len(theta)))
        for k in range(len(theta)):
            step = np.zeros(len(theta))
            step[k] = 1e-5
            up, down = residual(theta + step), residual(theta - step)
            if up is not None and down is not None:
                jac[:, k] = (up - down) / 2e-5
            elif up is not None:
                jac[:, k] = (up - r0) / 1e-5
            elif down is not None:
                jac[:, k] = (r0 - down) / 1e-5
        return jac

    start = np.log(np.clip([base_values[n] for n in names], low, high))
    best: tuple[float, np.ndarray, bool] | None = None
    for factor in (1.0, 0.6, 1.7):  # deterministic multi-start
        theta = np.clip(start + np.log(factor), log_low, log_high)
        r = residual(theta)
        if r is None:
            continue
        cost = float(r @ r)
        damping, converged = 1e-3, False
        for _ in range(max_iterations):
            jac = jacobian(theta, r)
            grad = jac.T @ r
            hess = jac.T @ jac
            accepted = False
            while damping < 1e12:
                scale = np.diag(np.maximum(np.diag(hess), 1e-12))
                step = np.linalg.solve(hess + damping * scale, -grad)
                trial = np.clip(theta + np.clip(step, -1.0, 1.0), log_low, log_high)
                r_trial = residual(trial)
                if r_trial is not None and float(r_trial @ r_trial) <= cost:
                    accepted = True
                    break
                damping *= 10.0
            if not accepted:
                converged = bool(np.max(np.abs(grad)) < 1e-3 * max(cost, 1.0))
                break
            new_cost = float(r_trial @ r_trial)
            small = (np.max(np.abs(trial - theta)) < 1e-7
                     or cost - new_cost <= 1e-10 * max(cost, 1e-12))
            theta, r, cost = trial, r_trial, new_cost
            damping = max(damping / 10.0, 1e-9)
            if small:
                converged = True
                break
        if best is None or cost < best[0]:
            best = (cost, theta, converged)
    if best is None:
        raise ValueError("no physically valid starting point; check fixed_parameters")
    cost, theta, converged = best
    r = residual(theta)
    assert r is not None
    jac = jacobian(theta, r)
    dof = n_data - len(names)
    reduced = cost / dof
    all_weighted = all(s.weighted for s in sets)

    # identifiability from the column-normalised Jacobian
    norms = np.linalg.norm(jac, axis=0)
    insensitive = norms < 1e-6 * max(norms.max(), 1.0)
    scaled = jac / np.where(insensitive, 1.0, norms)
    scaled[:, insensitive] = 0.0
    _, singular, vt = np.linalg.svd(scaled, full_matrices=False)
    degenerate = singular < 1e-4 * singular.max()
    unidentified = insensitive.copy()
    for vector in vt[degenerate]:
        unidentified |= np.abs(vector) > 0.1
    inverse = np.where(degenerate, 0.0, 1.0 / np.maximum(singular, 1e-300) ** 2)
    cov_scaled = (vt.T * inverse) @ vt
    safe = np.where(insensitive, 1.0, norms)
    covariance = cov_scaled / np.outer(safe, safe)
    # With stated measurement errors only inflate for lack of fit; without them the
    # noise level is estimated from the residuals.
    covariance *= max(reduced, 1.0) if all_weighted else reduced
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    values = np.clip(np.exp(theta), low, high)
    results = {}
    poorly = []
    for k, name in enumerate(names):
        at_bound = bool(np.isclose(theta[k], log_low[k]) or np.isclose(theta[k], log_high[k]))
        entry: dict[str, Any] = {
            "value": float(values[k]), "unit": PARAMETER_TABLE[name][1],
            "at_bound": at_bound}
        if unidentified[k] or at_bound:
            entry.update(identifiable=False, standard_error=None, ci95=None)
            poorly.append(name)
        else:
            entry.update(
                identifiable=bool(errors[k] < 1.0),
                standard_error=float(values[k] * errors[k]),
                ci95=[float(values[k] * np.exp(-1.96 * errors[k])),
                      float(values[k] * np.exp(1.96 * errors[k]))])
            if errors[k] >= 1.0:
                poorly.append(name)
        results[name] = entry
    denom = np.outer(errors, errors)
    correlation = np.divide(covariance, denom, out=np.zeros_like(covariance), where=denom > 0)

    warnings = []
    if poorly:
        warnings.append(f"the data do not determine {poorly}: fix them from independent "
                        "measurements or add data that is sensitive to them")
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            if abs(correlation[a, b]) > 0.98:
                warnings.append(f"{names[a]} and {names[b]} are strongly correlated "
                                f"({correlation[a, b]:+.3f})")
    if all_weighted and reduced > 3.0:
        warnings.append(f"reduced chi-square {reduced:.1f}: the model does not explain the "
                        "data within the stated uncertainties")
    if not all_weighted:
        warnings.append("measurement uncertainties were not supplied for every dataset; "
                        "errors assume scatter estimated from the residuals")
    if not converged:
        warnings.append("the optimiser did not converge")

    rms = []
    offset = 0
    for s in sets:
        n = len(s.observed)
        raw = r[offset:offset + n] * s.sigma
        rms.append({"kind": s.kind, "rms_residual": float(np.sqrt(np.mean(raw**2))),
                    "rms_relative": float(np.sqrt(np.mean(raw**2) / np.mean(s.observed**2))),
                    **s.ranges})
        offset += n

    if not converged:
        status = "failed"
    elif poorly or (all_weighted and reduced > 3.0):
        status = "fitted_with_reservations"
    else:
        status = "fitted_to_user_data"
    digest = hashlib.sha256(json.dumps(datasets, sort_keys=True).encode()).hexdigest()
    calibrated = dict(fixed_parameters or {})
    calibrated.update({n: results[n]["value"] for n in names
                       if converged and results[n]["identifiable"]})
    return {
        "status": status,
        "independently_validated": False,
        "parameters": results,
        "correlation": {names[a]: {names[b]: float(correlation[a, b])
                                   for b in range(len(names))} for a in range(len(names))},
        "fit_quality": {"chi_square": cost, "degrees_of_freedom": dof,
                        "reduced_chi_square": reduced if all_weighted else None,
                        "datasets": rms, "converged": converged},
        "parameters_for_simulation": calibrated,
        "held_fixed": [row for row in base.describe() if row["name"] not in names],
        "provenance": {**provenance, "data_sha256": digest, "model_version": MODEL_VERSION,
                       "dataset_sha256": [dataset_fingerprint(d) for d in datasets],
                       "method": "weighted Levenberg-Marquardt, linearised covariance",
                       "validity": "only within the measured ranges listed in fit_quality"},
        "warnings": warnings,
    }
