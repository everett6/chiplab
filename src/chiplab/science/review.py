"""Scientific review gate.

An agent must not be able to declare its own simulation successful just because it produced
numbers. This module computes the verdict from checks it runs itself; any status the claimant
supplies is ignored (and noted). Verdicts, weakest to strongest:

  rejected   at least one check failed
  unverified sane numbers, but nothing re-ran the model and nothing independent agrees
  plausible  sane, reproducible, and the outputs respond to their inputs; no benchmark
  validated  all of the above plus agreement with an independent benchmark (a reference whose
             source is not the tool that produced the claim)

"validated" never means "true": it means "agrees with an independent reference within the
stated tolerance". Benchmarks are only as good as their sources.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from chiplab.experiment_db.ledger import FIDELITY_LEVELS

KB = 1.380649e-23
VERDICTS = ("rejected", "unverified", "plausible", "validated")

# Units the gate understands. Prefixed forms (mW, ns, um2 ...) are accepted via _PREFIX.
_BASE_UNITS = {
    "J", "W", "s", "Hz", "V", "A", "ohm", "Ohm", "F", "H", "T", "K", "m", "m2", "m3", "um2",
    "mm2", "B", "bit", "bits", "op", "ops", "OPS", "TOPS", "FLOPS", "TFLOPS", "B/s", "GB/s",
    "TB/s", "J/bit", "J/op", "W/m2", "A/m", "A/m2", "Pa", "Pa*s", "C", "S", "%", "1",
    "dimensionless", "cycles", "cells", "count", "USD", "$", "degC", "J/m3", "kg/m3", "rad",
}
_PREFIX = ("f", "p", "n", "u", "µ", "m", "c", "k", "M", "G", "T")


def unit_ok(unit: str) -> bool:
    if not isinstance(unit, str) or not unit.strip():
        return False
    u = unit.strip()
    if u in _BASE_UNITS:
        return True
    return u[0] in _PREFIX and u[1:] in _BASE_UNITS


# Physical bounds keyed by metric-name pattern: (regex, lo, hi, why).
_BOUNDS: list[tuple[str, float, float, str]] = [
    (r"(^|_)(energy|e)(_per_\w+)?_j$|_j$", 0.0, math.inf, "energy cannot be negative"),
    (r"power(_\w+)?_w$|_w$", 0.0, math.inf, "power cannot be negative"),
    (r"(latency|delay|time|period)(_\w+)?_(s|ns|ps|us|ms)$", 0.0, math.inf,
     "time cannot be negative"),
    (r"area(_\w+)?_(m2|mm2|um2)$", 0.0, math.inf, "area cannot be negative"),
    (r"(efficiency|yield|utilization|fraction|probability|accuracy)$", 0.0, 1.0,
     "a fraction lies in [0, 1]"),
    (r"(error_rate|ber|p_fail)$", 0.0, 1.0, "a rate per trial lies in [0, 1]"),
    (r"temperature(_\w+)?_k$", 0.0, math.inf, "absolute temperature is positive"),
    (r"(count|cells|flip_flops|cycles)$", 0.0, math.inf, "a count cannot be negative"),
]


def landauer_j(temperature_k: float = 300.0) -> float:
    """Minimum energy to erase one bit, kT ln 2 (Landauer 1961)."""
    return KB * temperature_k * math.log(2)


@dataclass
class Check:
    name: str
    status: str           # pass | fail | warn | skip
    detail: str = ""


@dataclass
class Review:
    subject: str
    verdict: str
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject, "verdict": self.verdict,
            "passed": [c.name for c in self.checks if c.status == "pass"],
            "failed": [{"check": c.name, "why": c.detail} for c in self.checks
                       if c.status == "fail"],
            "warnings": [{"check": c.name, "why": c.detail} for c in self.checks
                         if c.status == "warn"],
            "skipped": [c.name for c in self.checks if c.status == "skip"],
            "notes": self.notes,
            "meaning": _MEANING[self.verdict],
        }


_MEANING = {
    "rejected": "At least one check failed; do not use these numbers.",
    "unverified": "Numbers are sane but nothing re-ran or independently confirmed them.",
    "plausible": "Sane, reproducible and input-sensitive; not yet compared to an independent "
                 "reference.",
    "validated": "Agrees with an independent reference within tolerance (not proof of truth).",
}


def _metric_values(metrics: dict[str, Any]) -> dict[str, tuple[Any, str | None]]:
    out: dict[str, tuple[Any, str | None]] = {}
    for k, v in (metrics or {}).items():
        if isinstance(v, dict) and "value" in v:
            out[k] = (v.get("value"), v.get("unit"))
        else:
            out[k] = (v, None)
    return out


def _numeric(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _flat_outputs(res: Any) -> dict[str, float]:
    if not isinstance(res, dict):
        return {}
    return {k: float(v) for k, (v, _) in _metric_values(res).items() if _numeric(v)}


def _same(a: dict[str, float], b: dict[str, float], rtol: float) -> bool:
    if a.keys() != b.keys():
        return False
    return all(math.isclose(a[k], b[k], rel_tol=rtol, abs_tol=1e-300) or
               (math.isnan(a[k]) and math.isnan(b[k])) for k in a)


def review(
    subject: str,
    metrics: dict[str, Any],
    *,
    fidelity: str | None,
    provenance: dict[str, Any] | None = None,
    evaluate: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    inputs: dict[str, Any] | None = None,
    benchmarks: dict[str, dict[str, Any]] | None = None,
    temperature_k: float = 300.0,
    claimed_status: str | None = None,
    repro_rtol: float = 1e-9,
    perturb: float = 0.1,
) -> Review:
    """Review one simulation claim.

    metrics: {name: value} or {name: {"value": v, "unit": u}}.
    evaluate/inputs: the model and its inputs, so the gate can re-run it (reproducibility) and
        perturb each numeric input (sensitivity). Without them the verdict is at most
        "unverified".
    benchmarks: {metric: {"value": v, "rel_tol": 0.1, "source": "doi or dataset"}}.
    """
    checks: list[Check] = []
    notes: list[str] = []
    prov = provenance or {}
    if claimed_status:
        notes.append(f"claimant said {claimed_status!r}; ignored, the verdict is computed here")

    # 1. fidelity
    if fidelity in FIDELITY_LEVELS:
        checks.append(Check("fidelity_known", "pass", str(fidelity)))
    else:
        checks.append(Check("fidelity_known", "fail",
                            f"fidelity {fidelity!r} is not one of the recorded levels"))

    # 2. provenance: which tool/model made the numbers
    tool = str(prov.get("tool") or prov.get("model") or "").strip()
    if tool:
        checks.append(Check("provenance", "pass", tool))
    else:
        checks.append(Check("provenance", "fail", "no tool/model named in provenance"))

    vals = _metric_values(metrics)
    if not vals:
        checks.append(Check("has_metrics", "fail", "no metrics to review"))

    # 3. finite numbers
    bad = [k for k, (v, _) in vals.items() if not _numeric(v) or not math.isfinite(float(v))]
    checks.append(Check("finite", "fail" if bad else "pass",
                        f"non-finite or non-numeric: {bad}" if bad else ""))

    # 4. units
    no_unit = [k for k, (_, u) in vals.items() if u is not None and not unit_ok(u)]
    missing = [k for k, (_, u) in vals.items() if u is None and not _suffix_unit(k)]
    if no_unit:
        checks.append(Check("units", "fail", f"unrecognised units: {no_unit}"))
    elif missing:
        checks.append(Check("units", "warn", f"no unit given or implied by name: {missing}"))
    else:
        checks.append(Check("units", "pass"))

    # 5. physical bounds + Landauer
    viol = []
    for k, (v, _) in vals.items():
        if not _numeric(v) or not math.isfinite(float(v)):
            continue
        for pat, lo, hi, why in _BOUNDS:
            if re.search(pat, k) and not lo <= float(v) <= hi:
                viol.append(f"{k}={v}: {why}")
                break
        if re.search(r"energy_per_(bit|op|mac)", k) and 0 < float(v) < landauer_j(temperature_k):
            viol.append(f"{k}={v:.3g} J is below the Landauer limit "
                        f"{landauer_j(temperature_k):.3g} J at {temperature_k} K")
    checks.append(Check("physical_bounds", "fail" if viol else "pass", "; ".join(viol)))

    # 6-7. reproducibility and input sensitivity (need the model)
    ran = False
    if evaluate is not None and inputs is not None:
        ran = True
        try:
            a = _flat_outputs(evaluate(dict(inputs)))
            b = _flat_outputs(evaluate(dict(inputs)))
        except Exception as e:  # noqa: BLE001 - a crashing model is a failed review
            checks.append(Check("reproducible", "fail", f"model raised {type(e).__name__}: {e}"))
            a = b = {}
        else:
            claimed = {k: float(v) for k, (v, _) in vals.items() if _numeric(v)}
            same_claim = all(k in a and math.isclose(a[k], claimed[k], rel_tol=max(repro_rtol,
                             1e-6)) for k in claimed if k in a)
            if not _same(a, b, repro_rtol):
                checks.append(Check("reproducible", "fail", "two runs with the same inputs "
                                    "disagree (unseeded randomness?)"))
            elif not same_claim:
                checks.append(Check("reproducible", "fail",
                                    "re-running the model does not give the claimed numbers"))
            else:
                checks.append(Check("reproducible", "pass"))
        if a:
            moved, dead = [], []
            for name, x in inputs.items():
                if not _numeric(x) or x == 0:
                    continue
                pert = dict(inputs)
                pert[name] = type(x)(x * (1 + perturb)) if isinstance(x, float) else x + max(
                    1, round(abs(x) * perturb))
                try:
                    out = _flat_outputs(evaluate(pert))
                except Exception:  # noqa: BLE001 - out-of-range perturbation; not evidence
                    continue
                (moved if not _same(a, out, 1e-12) else dead).append(name)
            if not moved and dead:
                checks.append(Check("input_sensitivity", "fail",
                                    f"no output changed when any of {dead} changed by "
                                    f"{perturb:.0%}: the numbers may be hard-coded"))
            elif dead:
                checks.append(Check("input_sensitivity", "warn",
                                    f"outputs ignore inputs {dead}"))
            elif moved:
                checks.append(Check("input_sensitivity", "pass", f"responds to {moved}"))
            else:
                checks.append(Check("input_sensitivity", "skip", "no numeric inputs to perturb"))
    else:
        checks.append(Check("reproducible", "skip", "no model/inputs given to re-run"))
        checks.append(Check("input_sensitivity", "skip", "no model/inputs given to perturb"))

    # 8. independent benchmarks
    bench_ok = False
    if benchmarks:
        fails, passes = [], []
        for k, ref in benchmarks.items():
            src = str(ref.get("source") or "").strip()
            if not src:
                fails.append(f"{k}: benchmark has no source")
                continue
            if tool and src.lower() == tool.lower():
                fails.append(f"{k}: benchmark source is the same tool that made the claim")
                continue
            if k not in vals or not _numeric(vals[k][0]):
                fails.append(f"{k}: not in the claimed metrics")
                continue
            rv, tol = float(ref["value"]), float(ref.get("rel_tol", 0.1))
            got = float(vals[k][0])
            err = abs(got - rv) / max(abs(rv), 1e-300)
            (passes if err <= tol else fails).append(
                f"{k}: {got:.4g} vs {rv:.4g} ({err:.1%}, tol {tol:.0%}, {src})")
        checks.append(Check("benchmark", "fail" if fails else "pass",
                            "; ".join(fails or passes)))
        bench_ok = not fails
    else:
        checks.append(Check("benchmark", "skip", "no independent reference given"))

    if any(c.status == "fail" for c in checks):
        verdict = "rejected"
    elif not ran:
        verdict = "validated" if bench_ok else "unverified"
        if bench_ok:
            notes.append("matches a benchmark but the model was not re-run here")
    elif bench_ok:
        verdict = "validated"
    else:
        verdict = "plausible"
    return Review(subject, verdict, checks, notes)


def _suffix_unit(name: str) -> bool:
    tail = name.rsplit("_", 1)[-1] if "_" in name else ""
    return unit_ok(tail) or tail in {"pct", "um2", "mm2", "ns", "ps", "us", "ms", "hz", "ohm"} \
        or bool(re.search(r"(efficiency|yield|utilization|fraction|probability|accuracy|"
                          r"count|cells|cycles|error_rate|ber)$", name))


def review_ledger_run(run: dict[str, Any],
                      benchmarks: dict[str, dict[str, Any]] | None = None) -> Review:
    """Review a stored ledger run. The model is not available, so the best possible verdict is
    "validated" via benchmark, otherwise "unverified"."""
    prov: dict[str, Any] = {"tool": run.get("logged_by")}
    if isinstance(run.get("provenance"), dict) and run["provenance"].get("tool"):
        prov = run["provenance"]
    r = review(run.get("run_id", "run"), run.get("metrics") or {},
               fidelity=run.get("fidelity"), provenance=prov, benchmarks=benchmarks)
    if run.get("duplicate_of"):
        r.notes.append(f"this row duplicates {run['duplicate_of']}")
    return r
