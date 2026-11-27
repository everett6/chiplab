"""Automatic scientific falsification.

For a proposed idea, `plan()` writes down — before any result exists — the hypothesis, the
measurable predictions, the control (a conventional architecture to compare against), the
quantitative result that would refute it, the evidence needed (and at what fidelity), and an
adversarial list of reasons it might NOT work, drawn from a failure-mode catalogue per
technology plus the Project Brain's limitations and rejected ideas.

`judge()` compares results against the frozen plan: supported, refuted or inconclusive. A
result below the plan's required fidelity is inconclusive no matter how good it looks.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from chiplab.experiment_db.ledger import FIDELITY_LEVELS

FAILURE_MODES: dict[str, list[str]] = {
    "dram": [
        "TRA sense margin is ~30 % smaller than a normal read; sense-amp offset and cell "
        "variation may push error rates above what ECC can correct",
        "Triple-row activation destroys its operands; the RowClone copies needed first may "
        "dominate latency and energy",
        "Operands must sit in the same subarray; real data layouts rarely cooperate",
        "Reserved compute rows are hammered; read disturb may force extra refresh",
        "Commodity DRAM vendors do not expose multi-row activation; it may need a custom part",
        "Arithmetic (add/multiply) needs long bit-serial majority sequences",
    ],
    "sram": [
        "Array accesses are paid either way; savings exist only when bus/instruction overhead "
        "dominates",
        "Bank-local ALUs cost area that could have been more cache",
        "Operands must be co-located in one bank; data placement and transposition cost time",
        "Multi-row bitline compute in a real SRAM macro risks read disturb and needs a custom "
        "macro; standard-cell versions are much less dense",
    ],
    "analog": [
        "ADCs may dominate energy and area, erasing the analog advantage",
        "Device variation, drift, IR drop and noise limit effective precision",
        "Weights must be stationary; reprogramming is slow and wears devices",
        "Peripheral DACs and calibration overheads are often left out of comparisons",
    ],
    "memristor": [
        "Endurance and retention of resistive cells limit training/reprogramming",
        "Conductance nonlinearity and cycle-to-cycle variation reduce accuracy",
        "No access to a BEOL resistive-memory process for prototyping",
    ],
    "ferrofluid": [
        "No validated model: any predicted gain may be an artefact of assumptions",
        "Fluid dynamics are slow compared with electronic switching",
        "No fabrication route; integration with CMOS is unproven",
    ],
    "generic": [
        "The baseline may be weak: compare against a well-tuned conventional design",
        "The workload may not be representative; gains may vanish on others",
        "Model parameters were assumed, not measured; results may follow the assumptions",
        "Benefit may disappear after counting control, data layout and host overheads",
    ],
}

# Minimum fidelity whose evidence can settle a claim about each metric family.
REQUIRED_FIDELITY = {"energy": "circuit_spice", "power": "circuit_spice", "area": "synthesis",
                     "latency": "rtl_sim", "throughput": "rtl_sim", "accuracy": "behavioral",
                     "error_rate": "circuit_spice", "yield": "measured", "cost": "published",
                     "data_movement": "discrete_event"}

# The fidelity levels are not a single ladder: RTL simulation is more detailed than SPICE in
# one sense, but says nothing about energy. So each metric family lists the kinds of evidence
# that can settle it.
_POWER = {"circuit_spice", "place_route", "signoff", "measured", "published"}
_TIME = {"rtl_sim", "synthesis", "place_route", "signoff", "fpga", "measured", "published"}
ACCEPTED_EVIDENCE: dict[str, set[str]] = {
    "energy": _POWER, "power": _POWER,
    "area": {"synthesis", "place_route", "signoff", "measured", "published"},
    "latency": _TIME, "throughput": _TIME,
    "accuracy": {"behavioral", "discrete_event", "specialized", "circuit_spice", "rtl_sim",
                 "fpga", "measured", "published"},
    "error_rate": {"circuit_spice", "measured", "published"},
    "yield": {"measured", "published"},
    "cost": {"published", "measured"},
    "data_movement": {"discrete_event", "specialized", "rtl_sim", "fpga", "measured",
                      "published"},
}

DEFAULT_CONTROL = {"dram": "CPU streaming the same data from DRAM over the channel",
                   "sram": "CPU load-load-op-store on the same SRAM array",
                   "analog": "digital MAC array of equal precision at the same node",
                   "memristor": "SRAM-based digital CIM of equal precision",
                   "ferrofluid": "conventional DRAM of equal capacity",
                   "generic": "the best conventional architecture for the same workload"}


def _family(metric: str) -> str:
    if "movement" in metric or "traffic" in metric:
        return "data_movement"
    for k in REQUIRED_FIDELITY:
        if k in metric:
            return k
    return "accuracy"


def plan(hypothesis: str, technology: str, metric: str, *, direction: str = "decrease",
         min_effect: float = 0.2, control: str | None = None, workload: str = "",
         brain: Any = None) -> dict[str, Any]:
    """Write a falsification plan. direction: "decrease" or "increase" of `metric` versus the
    control; min_effect: smallest relative change that would count (0.2 = 20 %)."""
    if direction not in ("decrease", "increase"):
        raise ValueError("direction must be decrease or increase")
    if not 0 < min_effect < 10:
        raise ValueError("min_effect must be in (0, 10)")
    tech = technology.lower()
    fam = _family(metric)
    need = REQUIRED_FIDELITY[fam]
    reasons = FAILURE_MODES.get(tech, []) + FAILURE_MODES["generic"]
    from_brain: list[str] = []
    if brain is not None:
        rec = brain.recall(f"{technology} {metric} {hypothesis}", per_kind=3)
        from_brain = [f"[limitation] {m['title']}" for m in rec["limitations"]] + \
                     [f"[rejected before] {m['title']}" for m in rec["rejected"]]
    sign = "<=" if direction == "decrease" else ">="
    thresh = 1 - min_effect if direction == "decrease" else 1 + min_effect
    body = {
        "hypothesis": hypothesis,
        "technology": tech,
        "workload": workload or "state the workload before running anything",
        "predictions": [
            f"{metric}(proposed) / {metric}(control) {sign} {thresh:.3g} on the stated workload",
            "the effect persists when the most uncertain assumed parameter is moved to the "
            "pessimistic end of its range",
            "the effect is not created by a weaker baseline: the control is tuned equally",
        ],
        "control": control or DEFAULT_CONTROL.get(tech, DEFAULT_CONTROL["generic"]),
        "failure_conditions": [
            f"ratio {'>' if direction == 'decrease' else '<'} {thresh:.3g} at the required "
            f"fidelity -> refuted",
            "ratio's 95 % interval includes 1.0 -> inconclusive, not supported",
            "the review gate rejects the producing simulation -> no evidence",
        ],
        "required_evidence": {
            "metric": metric, "min_fidelity": need,
            "accepted_fidelity": sorted(ACCEPTED_EVIDENCE[fam]),
            "also": ["independent benchmark or second solver agreeing",
                     "sensitivity sweep over assumed parameters",
                     "same workload and data for proposed and control"],
        },
        "reasons_it_might_fail": reasons + from_brain,
        "direction": direction, "min_effect": min_effect,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    body["plan_id"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    return body


def judge(plan_: dict[str, Any], proposed: float, control: float, *, fidelity: str,
          ratio_ci95: tuple[float, float] | None = None,
          review_verdict: str | None = None) -> dict[str, Any]:
    """Settle a plan with results. Never upgrades evidence that is below the required level."""
    if fidelity not in FIDELITY_LEVELS:
        return {"outcome": "inconclusive", "why": f"unknown fidelity {fidelity!r}"}
    if review_verdict == "rejected":
        return {"outcome": "inconclusive", "why": "the review gate rejected the simulation"}
    need = plan_["required_evidence"]["min_fidelity"]
    accepted = set(plan_["required_evidence"].get("accepted_fidelity") or
                   ACCEPTED_EVIDENCE[_family(plan_["required_evidence"]["metric"])])
    if control == 0:
        return {"outcome": "inconclusive", "why": "control value is zero"}
    ratio = proposed / control
    thresh = 1 - plan_["min_effect"] if plan_["direction"] == "decrease" else \
        1 + plan_["min_effect"]
    good = ratio <= thresh if plan_["direction"] == "decrease" else ratio >= thresh
    if fidelity not in accepted:
        return {"outcome": "inconclusive", "ratio": ratio,
                "why": f"{fidelity} evidence cannot settle this metric (needs {need} or one of "
                       f"{sorted(accepted)}); {'promising' if good else 'unpromising'} but not "
                       f"settled"}
    if ratio_ci95 and ratio_ci95[0] <= 1.0 <= ratio_ci95[1]:
        return {"outcome": "inconclusive", "ratio": ratio,
                "why": "95 % interval of the ratio includes 1.0"}
    return {"outcome": "supported" if good else "refuted", "ratio": ratio,
            "threshold": thresh, "fidelity": fidelity}
