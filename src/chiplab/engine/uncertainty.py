"""Uncertainty propagation: Monte Carlo confidence bands on PPA and cost.

Every continuous design parameter that is not exact by construction (compute rate, clock,
efficiency, die area, TDP, memory bandwidth/latency/energy, link bandwidth/latency/energy,
thermal resistance) is drawn uniformly within +/- its uncertainty:

- a TrackedValue's own ``uncertainty_pct`` if set;
- else a default by its provenance: measured 5%, published 10%, estimated 20%, assumed 30%;
- plain numbers (no provenance given) use ``default_pct``.

Counts, capacities, process node and the workload are held fixed: they are design choices,
not uncertain inputs. Sampling is seeded (numpy default_rng), so results are reproducible.
Each sample runs the Level 1 analytical model (and the cost model), so the bands describe
input uncertainty only, not the model-form error of the analytical model itself.
"""

from __future__ import annotations

import copy
import re
from typing import Any

import numpy as np
from pydantic import BaseModel

from ..schemas.common import Provenance, TrackedValue
from ..schemas.design import DesignSpec, plain_design
from ..schemas.simulation import SimulationConfig
from ..schemas.workload import WorkloadSpec

DEFAULT_PCT_BY_PROVENANCE = {
    Provenance.MEASURED: 5.0,
    Provenance.PUBLISHED: 10.0,
    Provenance.ESTIMATED: 20.0,
    Provenance.ASSUMED: 30.0,
}

# Field-path regex -> upper clip (None = unbounded). Lower bound is always > 0.
UNCERTAIN_FIELDS: dict[str, float | None] = {
    r"chiplets\[\d+\]\.(area_mm2|tdp_watts)": None,
    r"chiplets\[\d+\]\.compute\.(peak_tops|clock_ghz|sram_mb)": None,
    r"chiplets\[\d+\]\.compute\.efficiency_factor": 1.0,
    r"chiplets\[\d+\]\.memory\.(bandwidth_per_channel_gbps|latency_ns|energy_pj_per_bit)": None,
    r"interconnect\.(link_bandwidth_gbps|latency_ns|energy_pj_per_bit)": None,
    r"constraints\.thermal_resistance_c_per_w": None,
}
_COMPILED = [(re.compile(f"^{k}$"), v) for k, v in UNCERTAIN_FIELDS.items()]

OUTPUTS = ("peak_compute_tops", "effective_compute_tops", "throughput_tokens_per_sec",
           "estimated_latency_ms", "total_power_watts", "estimated_temp_c",
           "energy_per_inference_mj", "unit_cost_usd")

MAX_SAMPLES = 5000


def _uncertain_params(design: DesignSpec, default_pct: float) -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = []

    def visit(obj: Any, path: str) -> None:
        if isinstance(obj, TrackedValue) or (isinstance(obj, (int, float))
                                             and not isinstance(obj, bool)):
            clip = next((c for rx, c in _COMPILED if rx.match(path)), "skip")
            if clip == "skip":
                return
            if isinstance(obj, TrackedValue):
                if not isinstance(obj.value, (int, float)) or isinstance(obj.value, bool):
                    return
                nominal = float(obj.value)
                pct = (obj.uncertainty_pct if obj.uncertainty_pct is not None
                       else DEFAULT_PCT_BY_PROVENANCE[obj.provenance])
                prov = obj.provenance.value
            else:
                nominal, pct, prov = float(obj), default_pct, "unspecified"
            if nominal > 0 and pct > 0:
                params.append({"parameter": path, "nominal": nominal,
                               "uncertainty_pct": float(pct), "provenance": prov,
                               "clip_max": clip})
            return
        if isinstance(obj, BaseModel):
            for name in type(obj).model_fields:
                visit(getattr(obj, name), f"{path}.{name}" if path else name)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                visit(v, f"{path}[{i}]")

    visit(design, "")
    return params


def _set(d: dict[str, Any], path: str, value: float) -> None:
    keys = re.findall(r"[^.\[\]]+", path)
    cur: Any = d
    for k in keys[:-1]:
        cur = cur[int(k)] if isinstance(cur, list) else cur[k]
    cur[keys[-1]] = value


def _evaluate(design: DesignSpec, workload: WorkloadSpec | None, include_cost: bool,
              volume: int) -> dict[str, float | None]:
    from ..cost.model import estimate_unit_cost
    from .analytical import run_analytical_simulation

    res = run_analytical_simulation(design, workload, SimulationConfig())
    m = res.metrics
    out: dict[str, float | None] = {k: getattr(m, k, None) for k in OUTPUTS
                                    if k != "unit_cost_usd"}
    if workload is None:
        out["throughput_tokens_per_sec"] = None
        out["estimated_latency_ms"] = None
    out["unit_cost_usd"] = (estimate_unit_cost(design, volume)["unit_cost_usd"]
                            if include_cost else None)
    return out


def propagate_uncertainty(design: DesignSpec, workload: WorkloadSpec | None = None,
                          samples: int = 200, seed: int = 42, default_pct: float = 15.0,
                          include_cost: bool = True, volume: int = 100_000) -> dict[str, Any]:
    """Monte Carlo PPA/cost bands. Returns nominal, p5/p50/p95 per output, the inputs that
    were varied, and up to 3 inputs per output whose |Spearman rank correlation| clears the
    noise floor 4/sqrt(samples-1)."""
    if not 10 <= samples <= MAX_SAMPLES:
        raise ValueError(f"samples must be in [10, {MAX_SAMPLES}]")
    if not 0 <= default_pct <= 100:
        raise ValueError("default_pct must be in [0, 100]")
    from ..backends.ecore import apply_ecore_to_design

    params = _uncertain_params(design, default_pct)
    base, _ = plain_design(design)
    # eCore-derived chiplet numbers come from the array simulator, not from the spec: they
    # are fixed here (their uncertainty is described by the eCore result's limitations).
    base, _, _ = apply_ecore_to_design(base)
    ecore_paths = {f"chiplets[{i}]" for i, c in enumerate(base.chiplets)
                   if c.compute is not None and c.compute.ecore is not None}
    params = [p for p in params if not any(p["parameter"].startswith(e + ".")
                                           for e in ecore_paths)
              or p["parameter"].endswith((".clock_ghz", ".efficiency_factor"))]
    base_dict = base.model_dump()

    nominal = _evaluate(base, workload, include_cost, volume)
    rng = np.random.default_rng(seed)
    draws = rng.uniform(-1.0, 1.0, size=(samples, len(params)))
    inputs = np.zeros_like(draws)
    rows: list[dict[str, float | None]] = []
    for s in range(samples):
        d = copy.deepcopy(base_dict)
        for j, p in enumerate(params):
            v = p["nominal"] * (1.0 + draws[s, j] * p["uncertainty_pct"] / 100.0)
            v = max(v, p["nominal"] * 1e-3)
            if p["clip_max"] is not None:
                v = min(v, p["clip_max"])
            inputs[s, j] = v
            _set(d, p["parameter"], v)
        rows.append(_evaluate(DesignSpec.model_validate(d), workload, include_cost, volume))

    bands: dict[str, Any] = {}
    sensitivity: dict[str, list[dict[str, Any]]] = {}
    ranks_in = np.argsort(np.argsort(inputs, axis=0), axis=0).astype(float)
    # Independent draws still correlate by chance (max |rho| ~0.25 over ~100 pairs at
    # n=200); only report rank correlations above a multiple-comparison noise floor.
    noise_floor = 4.0 / float(np.sqrt(samples - 1))
    for k in OUTPUTS:
        vals = np.array([r[k] for r in rows if r[k] is not None], dtype=float)
        if nominal.get(k) is None or len(vals) < len(rows):
            continue
        p5, p50, p95 = (float(x) for x in np.percentile(vals, [5, 50, 95]))
        nom = float(nominal[k])  # type: ignore[arg-type]
        bands[k] = {"nominal": nom, "p5": p5, "p50": p50, "p95": p95,
                    "spread_pct": (100.0 * (p95 - p5) / (2 * abs(nom))) if nom else None}
        if np.ptp(vals) == 0 or not params:
            continue
        r_out = np.argsort(np.argsort(vals)).astype(float)
        corr = []
        for j, p in enumerate(params):
            col = ranks_in[:, j]
            c = float(np.corrcoef(col, r_out)[0, 1]) if np.ptp(col) > 0 else 0.0
            corr.append({"parameter": p["parameter"], "spearman": round(c, 3)})
        corr = [c for c in corr if abs(c["spearman"]) >= noise_floor]
        corr.sort(key=lambda x: -abs(x["spearman"]))
        sensitivity[k] = corr[:3]

    return {
        "samples": samples, "seed": seed, "method": "monte_carlo_uniform",
        "bands": bands,
        "sensitivity_top3": sensitivity,
        "sensitivity_noise_floor": round(noise_floor, 3),
        "varied_inputs": [{k: v for k, v in p.items() if k != "clip_max"} for p in params],
        "default_pct_by_provenance": {k.value: v for k, v in DEFAULT_PCT_BY_PROVENANCE.items()},
        "default_pct_unspecified": default_pct,
        "disclaimer": ("Bands propagate input-parameter uncertainty through the Level 1 "
                       "analytical and cost models (uniform +/- uncertainty_pct, 5th-95th "
                       "percentile). They do not include the analytical model's own error."),
    }
