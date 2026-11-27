"""Experimental in-memory-computing lab: small designs a student lab can actually build.

Two families, each verified by real tools before anything is claimed:

- ``digital_cim``: bit-serial digital CIM macro (weights in flip-flops, adder tree beside them).
  Verilog generated per size, bit-exact against NumPy in Icarus Verilog, synthesised by Yosys to
  IHP sg13g2 cells. Buildable as a Tiny Tapeout IHP tile or on any FPGA.
- ``analog_crossbar``: differential resistor crossbar with op-amp TIAs from catalogue parts
  (E24/E96 resistors, quad op-amps). Simulated by ngspice and cross-checked by an independent
  nodal solver; comes with a bill of materials.

``build`` records ChipLab's predictions in the learning loop *before* anything is built, so a
later bench, FPGA or silicon measurement can correct the model (``chiplab.learning``).
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from chiplab.imc_lab import analog, digital

FAMILIES = {
    "digital_cim": {
        "title": "Bit-serial digital CIM macro (Tiny Tapeout / FPGA)",
        "limits": digital.LIMITS,
        "verified_by": ["Icarus Verilog bit-exact vs NumPy", "Yosys -> IHP sg13g2 cells"],
        "measurable": {"cell_area_um2": "um2", "cycles_per_mvm": "cycles"},
        "measure_how": {
            "cell_area_um2": "stdcell area from a real P&R run (e.g. LibreLane metrics "
                             "design__instance__area__stdcell); source_kind 'signoff'",
            "cycles_per_mvm": "logic analyser / testbench on the FPGA or TT chip; 'fpga' or "
                              "'silicon'"},
        "build_steps": [
            "Generate cim_macro.v (build returns it) and add it to a Tiny Tapeout IHP template "
            "or an FPGA project.",
            "Wrap the ports: TT has 8 in / 8 out / 8 bidirectional pins, so serialise w_data, "
            "x_in and y_out through a small SPI or byte shift register.",
            "Harden on TT GitHub Actions (or synthesise for the FPGA) and check timing there.",
            "Load weights, run vectors, compare y_out with the NumPy golden model.",
            "Submit the measured numbers against the prediction ids from build."],
    },
    "analog_crossbar": {
        "title": "Differential resistor crossbar + op-amp TIA (breadboard / PCB)",
        "limits": analog.LIMITS,
        "verified_by": ["ngspice .op", "independent NumPy nodal solver (must agree)"],
        "measurable": {"rms_error_pct_fs": "%FS", "static_power_w": "W",
                       "settle_time_s": "s", "energy_per_mvm_j": "J"},
        "measure_how": {
            "rms_error_pct_fs": "DMM/ADC on each output vs the ideal MVM over >= 10 random "
                                "vectors; 'bench'",
            "static_power_w": "source-meter current on the input DACs x Vread; 'bench'",
            "settle_time_s": "oscilloscope, input step to 0.1 % of final output; 'bench'",
            "energy_per_mvm_j": "static power x measured settle time; 'bench'"},
        "build_steps": [
            "Order the BOM from build (resistors snap to the chosen E-series).",
            "Lay out rows as input lines, columns as + / - output lines; keep wires short "
            "(wire_ohm models them).",
            "Each output line into an inverting TIA with Rf; difference + and - per output.",
            "Drive inputs from a DAC or dividers (0..Vread); read outputs with an ADC/DMM.",
            "Submit the measured numbers against the prediction ids from build."],
    },
}


class ImcLabError(ValueError):
    pass


def catalog() -> dict[str, Any]:
    return {k: {kk: v[kk] for kk in ("title", "limits", "verified_by", "measurable",
                                     "measure_how")}
            for k, v in FAMILIES.items()}


def _features(family: str, spec: dict[str, Any]) -> dict[str, float]:
    if family == "digital_cim":
        p = digital.validate(spec)
        return {k: float(p[k]) for k in ("rows", "cols", "wbits", "xbits")}
    p = analog.validate(spec)
    return {"rows": p["rows"], "cols": p["cols"], "wire_ohm": p["wire_ohm"],
            "tolerance_pct": p["tolerance_pct"], "vread": p["vread"],
            "series_steps": {"E24": 24.0, "E96": 96.0, "ideal": 0.0}[p["series"]]}


def design_id(family: str, spec: dict[str, Any]) -> str:
    return family + ":" + json.dumps(_features(family, spec), sort_keys=True)


@lru_cache(maxsize=128)
def _synth_cached(key: str) -> dict[str, Any]:
    return digital.synthesize(json.loads(key))


def predictions(family: str, spec: dict[str, Any], synthesize: bool = True) -> dict[str, Any]:
    """ChipLab's numbers for one design (the values the learning loop later checks)."""
    if family == "digital_cim":
        out = digital.predict(spec)
        if synthesize:
            syn = _synth_cached(json.dumps(digital.validate(spec), sort_keys=True))
            out["cell_area_um2"] = {"value": syn["cell_area_um2"], "unit": "um2",
                                    "provenance": f"yosys+abc on {syn['liberty']} (no P&R)"}
            out["cells"] = {"value": syn["cells"], "unit": "cell", "provenance": "yosys+abc"}
            # TT designs hardened at ~40-60 % density; 0.5 is an assumption, not a P&R result
            out["tt_tiles_est"] = {"value": round(syn["cell_area_um2"] / (0.5 * digital.TT_TILE_UM2),
                                                  2),
                                   "unit": "tile", "provenance": "assumed 50 % placement density"}
        return out
    if family == "analog_crossbar":
        return analog.predict(spec)
    raise ImcLabError(f"family must be one of {sorted(FAMILIES)}")


def build(family: str, spec: dict[str, Any], loop: Any = None, seed: int = 0) -> dict[str, Any]:
    """Generate, verify with real tools, and (optionally) log predictions for measurement."""
    if family not in FAMILIES:
        raise ImcLabError(f"family must be one of {sorted(FAMILIES)}")
    fam = FAMILIES[family]
    out: dict[str, Any] = {"family": family, "title": fam["title"],
                           "design_id": design_id(family, spec)}
    if family == "digital_cim":
        p = digital.validate(spec)
        sim = digital.simulate(p, n_vectors=16, seed=seed)
        out |= {"params": p, "verification": sim, "rtl": digital.rtl(p),
                "verified": sim["bit_exact"]}
    else:
        ev = analog.evaluate(spec, seed=seed)
        out |= {"params": ev["params"], "verification": {
            k: ev[k] for k in ("solvers_agree", "ngspice_vs_nodal_max_v", "rms_error_pct_fs",
                               "swing_ok", "y_ideal_v", "y_ngspice_v")},
            "netlist": ev["netlist"], "bom": ev["bom"],
            "verified": ev["solvers_agree"] and ev["swing_ok"]}
        if not ev["swing_ok"]:
            out["warnings"] = [f"output reaches {ev['max_abs_output_v']:.2f} V; lower rf_ohm or "
                               "vread to stay inside the op-amp rails"]
    preds = predictions(family, spec)
    out["predictions"] = preds
    out["build_steps"] = fam["build_steps"]
    if loop is not None and out["verified"]:
        feats = _features(family, spec)
        ids = {}
        for metric, unit in fam["measurable"].items():
            v = preds.get(metric, {}).get("value")
            if v is None:
                continue
            ids[metric] = loop.record_prediction(
                f"imc.{family}.{metric}", metric, float(v), unit, feats, out["design_id"],
                family, preds[metric]["provenance"])
            preds[metric]["corrected"] = loop.corrected(f"imc.{family}.{metric}", metric,
                                                        float(v), feats)
        out["prediction_ids"] = ids
        out["measure_how"] = fam["measure_how"]
    elif loop is not None:
        out["prediction_ids"] = {}
        out["note"] = "verification failed, so no predictions were logged for measurement"
    return out


def candidates(family: str, specs: list[dict[str, Any]], synthesize: bool = True
               ) -> list[dict[str, Any]]:
    """Turn specs into learning-loop recommendation candidates."""
    if not 1 <= len(specs) <= 16:
        raise ImcLabError("give 1-16 candidate specs")
    out = []
    for s in specs:
        pr = predictions(family, s, synthesize=synthesize)
        out.append({"id": design_id(family, s), "family": family,
                    "model_prefix": f"imc.{family}", "features": _features(family, s),
                    "spec": s,
                    "metrics": {k: v["value"] for k, v in pr.items()
                                if isinstance(v.get("value"), (int, float))}})
    return out
