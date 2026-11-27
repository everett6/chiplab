"""MCP tools for the in-memory-computing lab and the learning loop."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from chiplab import imc_lab
from chiplab.learning import LearningError, LearningLoop


@lru_cache(maxsize=4)
def _loop_at(path: str) -> LearningLoop:
    return LearningLoop(path) if path else LearningLoop()


def _loop() -> LearningLoop:
    return _loop_at(os.environ.get("CHIPLAB_LEARNING_DB", ""))


def _err(e: Exception) -> dict[str, Any]:
    return {"ok": False, "error": str(e)}


def imc_lab_catalog() -> dict[str, Any]:
    """List the buildable in-memory-computing designs (digital CIM macro, resistor crossbar):
    size limits, which real tools verify each, and which metrics a lab can measure."""
    return {"ok": True, "families": imc_lab.catalog()}


def imc_lab_build(family: str, spec: dict[str, Any], seed: int = 0,
                  record_predictions: bool = True, include_sources: bool = False
                  ) -> dict[str, Any]:
    """Generate one small IMC design, verify it with real tools, and return build steps.

    family: "digital_cim" (spec: rows, cols 1-16; wbits 2-8; xbits 1-8) or "analog_crossbar"
    (spec: rows, cols 1-8; vread, r_on_ohm, r_off_ohm, rf_ohm, wire_ohm, opamp_gain,
    opamp_gbw_hz, tolerance_pct, supply_v, series E24/E96/ideal).
    With record_predictions, ChipLab's predictions are logged before the build so a later
    measurement (learning_submit_measurement with the returned prediction_ids) can correct them.
    include_sources returns the Verilog / SPICE netlist text.
    """
    try:
        out = imc_lab.build(family, spec, loop=_loop() if record_predictions else None, seed=seed)
    except (ValueError, RuntimeError) as e:
        return _err(e)
    if not include_sources:
        for k in ("rtl", "netlist"):
            if k in out:
                out[k] = f"<{len(out[k].splitlines())} lines; include_sources=true to return>"
    return {"ok": True} | out


def imc_lab_recommend(family: str, specs: list[dict[str, Any]], objective: str,
                      minimize: bool = True, constraints: dict[str, list[Any]] | None = None
                      ) -> dict[str, Any]:
    """Rank 1-16 candidate specs of one family on learning-loop-corrected predictions.

    constraints: {metric: [min or null, max or null]}. Also returns the most informative next
    experiment (the top candidate whose model is least certain).
    """
    try:
        cands = imc_lab.candidates(family, specs)
        cons = {k: (v[0], v[1]) for k, v in (constraints or {}).items()}
        rec = _loop().recommend(cands, objective, minimize, cons)
    except (ValueError, RuntimeError, IndexError, TypeError) as e:
        return _err(e)
    return {"ok": True} | rec


def learning_submit_measurement(prediction_id: str, value: float, unit: str, source_kind: str,
                                method: str, measured_on: str, submitted_by: str,
                                attested_real: bool, uncertainty: float | None = None,
                                repeats: list[float] | None = None, reference: str = "",
                                confirm_outlier: bool = False) -> dict[str, Any]:
    """Submit a real measurement against a logged prediction. It is validated (source_kind
    bench/silicon/fpga/published/signoff, method, date, unit, uncertainty or >=3 repeats,
    outlier check); simulation output is refused. Only validated points are learned from."""
    try:
        return {"ok": True} | _loop().submit_measurement(
            prediction_id, value, unit, source_kind=source_kind, method=method,
            measured_on=measured_on, submitted_by=submitted_by, attested_real=attested_real,
            uncertainty=uncertainty, repeats=repeats, reference=reference,
            confirm_outlier=confirm_outlier)
    except (LearningError, ValueError) as e:
        return _err(e)


def learning_refit(model_id: str, metric: str) -> dict[str, Any]:
    """Refit the correction for one model metric from validated measurements. A new version is
    accepted only if it beats the uncorrected model out of sample on >= 3 points."""
    try:
        return {"ok": True} | _loop().refit(model_id, metric)
    except (LearningError, ValueError) as e:
        return _err(e)


def learning_rollback(model_id: str, metric: str, to_version: int) -> dict[str, Any]:
    """Re-issue an earlier correction version as the newest (history is kept)."""
    try:
        return {"ok": True} | _loop().rollback(model_id, metric, to_version)
    except (LearningError, ValueError) as e:
        return _err(e)


def learning_record_build(family: str, design_id: str, worked: bool, submitted_by: str,
                          notes: str = "") -> dict[str, Any]:
    """Record whether a physical build of a design worked; feeds build-success odds used in
    recommendations."""
    try:
        bid = _loop().record_build(family, design_id, worked, submitted_by, notes)
    except LearningError as e:
        return _err(e)
    return {"ok": True, "build_id": bid, "build_success": _loop().build_success(family)}


def learning_status() -> dict[str, Any]:
    """Counts of predictions and measurements by status, and the current correction per model."""
    return {"ok": True} | _loop().status()


READ_ONLY = (imc_lab_catalog, learning_status)
WRITES = (imc_lab_build, imc_lab_recommend, learning_submit_measurement, learning_refit,
          learning_rollback, learning_record_build)


def register(mcp: Any, read_only: Any, writes: Any) -> None:
    for fn in READ_ONLY:
        mcp.tool(annotations=read_only)(fn)
    for fn in WRITES:
        mcp.tool(annotations=writes)(fn)
