"""Reproduction harness for the 2022/2023 two-port ferrofluid experiment.

Only statements read in the arXiv preprint are recorded here, with their location.  The
raw measurements are **not** bundled: the preprint says data are available on request.
Until the user supplies them every item is "reference_data_missing" and the experiment is
not reproduced, whatever a simulation looks like.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chiplab.models.materials import ferro_experiment as experiment
from chiplab.models.materials.ferro_statespace import (
    NOT_MODELLED,
    StateSpaceDevice,
    run_protocol,
)

CITATION = {
    "authors": "M. Crepaldi, C. Mohan, E. Garofalo, A. Adamatzky, K. Szacilowski, "
               "A. Chiolerio",
    "preprint": "Evidence of In-Memory Computing in a Ferrofluid, arXiv:2211.08152v1 "
                "(15 Nov 2022), https://arxiv.org/html/2211.08152v1",
    "published": "Adv. Mater. 2023, doi:10.1002/adma.202211406",
    "read": "main text and captions of the preprint only; its supplement, Materials and "
            "Methods and the published version were not read",
}

REPORTED = [
    {"what": "sample: water-based Fe3O4 ferrofluid", "where": "abstract"},
    {"what": "two-port vector network analyser with DC bias generator and two bias tees; "
             "the generator's negative terminal is grounded", "where": "Fig. 1A text"},
    {"what": "quasi-DC programming voltage VP of both polarities", "where": "Fig. 1B text"},
    {"what": "VP swept -3.8 V to 3.8 V in 0.1 V steps of 1 s each, repeated 50 times",
     "where": "Fig. 1C", "values": {"amplitude_v": 3.8, "step_v": 0.1, "dwell_s": 1.0,
                                    "iterations": 50}},
    {"what": "at VP = -0.85 V the starting summed impedances are Z11 14312 ohm, "
             "Z12 2320 ohm, Z21 2060 ohm, Z22 11795 ohm", "where": "Fig. 1C text",
     "values": {"bias_v": -0.85, "zc_ohm": {"11": 14312, "12": 2320, "21": 2060,
                                            "22": 11795}}},
    {"what": "read from 10 MHz to 6 GHz; reported quantity is the sum of impedance values "
             "over all scanned frequencies", "where": "Fig. 1 caption and Fig. 1B text",
     "values": {"f_min_hz": 1e7, "f_max_hz": 6e9}},
    {"what": "memory test: 3.3 V pulses of varying duration, after a closed-loop reset of "
             "summed Z11 to 14338 ohm", "where": "Fig. 2A/B",
     "values": {"pulse_v": 3.3, "reset_zc11_ohm": 14338}},
    {"what": "pinched hysteresis of Z11 shrinks at positive VP with iterations; Z22 "
             "shrinks over the whole range; Z12 and Z21 behave oppositely",
     "where": "Fig. 1C text"},
    {"what": "Z22 evolves towards impedance values that depend on pulse duration",
     "where": "Fig. 2B/C"},
    {"what": "over days of uninterrupted testing the impedance dynamics shrink; the "
             "behaviour is reversible", "where": "classification text (supplement S7)"},
    {"what": "software-trained readout accuracies 90 % (classification) and 90.6 % "
             "(reservoir, digits 0-3)", "where": "Fig. 3B, Fig. 4B"},
]

REQUIRED_ITEMS = {
    "hysteresis": "summed (or per-frequency) Z against time during the VP sweep",
    "pulse_duration": "Z against time during reset, 3.3 V write and hold for several pulse "
                      "durations",
    "long_term_drift": "Z during at least one day of repeated stimulation, with the drive "
                       "trace",
}

DATA_REQUEST = [
    "time-stamped two-port S-parameters (.s2p) or impedances for the Fig. 1C, Fig. 2 and "
    "supplement S7 experiments",
    "the voltage trace applied to each port, including reset-loop voltages and timing",
    "frequency grid, RF power, IF bandwidth, calibration plane, bias-tee models and sweep "
    "duration",
    "exact definition of the summed impedance (magnitudes or real parts; number of points)",
    "measurement uncertainty (repeat sweeps)",
    "sample composition, volume, particle size, vessel and electrode geometry",
    "temperature during each run",
]

NOT_REPRODUCIBLE_FROM_TEXT = [
    "impedance change per volt, relaxation and retention time constants (not given in the "
    "main text)",
    "pulse durations of the memory test, reset voltage and loop gain (not given)",
    "which port(s) the programming voltage drives and how the other is terminated",
    "up/down order of the voltage sweep",
    "digit classification and reservoir accuracy: needs the image data and trained "
    "software readout; not part of this harness",
    "Z12 differing from Z21 (2320 vs 2060 ohm): this reciprocal model cannot represent it; "
    "whether it is sample, bias or fixture behaviour is not stated",
]

DRIFT_MIN_DURATION_S = 86_400.0


def hysteresis_protocol(drive_port: str, other_port: str, other: str = "ground",
                        iterations: int = 50, temperature_k: float | None = None
                        ) -> dict[str, Any]:
    """The Fig. 1C sweep.  Triangle order (up then down) and the port wiring are assumed."""
    if other not in ("ground", "float"):
        raise ValueError("other must be 'ground' or 'float'")
    up = np.round(np.linspace(-3.8, 3.8, 77), 10)
    triangle = np.concatenate([up, up[-2:0:-1]])
    segments = []
    for v in triangle:
        segment: dict[str, Any] = {"duration_s": 1.0, "voltages_v": {
            drive_port: float(v), other_port: 0.0 if other == "ground" else None}}
        if temperature_k is not None:
            segment["temperature_k"] = float(temperature_k)
        segments.append(segment)
    return {"segments": segments, "repeat": int(iterations)}


def pulse_protocol(drive_port: str, other_port: str, duration_s: float, hold_s: float,
                   other: str = "ground", temperature_k: float | None = None
                   ) -> dict[str, Any]:
    """One 3.3 V write of the memory test followed by a 0 V hold (no reset loop)."""
    segments = []
    for volts, time in ((3.3, duration_s), (0.0, hold_s)):
        segment: dict[str, Any] = {"duration_s": float(time), "voltages_v": {
            drive_port: volts, other_port: 0.0 if other == "ground" else None}}
        if temperature_k is not None:
            segment["temperature_k"] = float(temperature_k)
        segments.append(segment)
    return {"segments": segments}


def _protocol_matches(item: str, protocol: dict) -> tuple[bool, str]:
    segments = protocol.get("segments", [])
    voltages = [v for s in segments for v in s["voltages_v"].values() if v is not None]
    total = sum(float(s["duration_s"]) for s in segments) * protocol.get("repeat", 1)
    if item == "hysteresis":
        dwell = [float(s["duration_s"]) for s in segments
                 if any(v for v in s["voltages_v"].values())]
        ok = (bool(voltages) and abs(max(voltages) - 3.8) <= 0.05
              and abs(min(voltages) + 3.8) <= 0.05
              and len({round(v, 2) for v in voltages}) >= 70
              and abs(float(np.median(dwell)) - 1.0) <= 0.2)
        return ok, "sweep must span -3.8 V to 3.8 V in about 0.1 V steps of about 1 s"
    if item == "pulse_duration":
        lengths = {round(float(s["duration_s"]), 9) for s in segments
                   if any(v is not None and abs(v - 3.3) <= 0.05
                          for v in s["voltages_v"].values())}
        return len(lengths) >= 2, "needs 3.3 V pulses of at least two durations"
    ok = total >= DRIFT_MIN_DURATION_S and any(abs(v) > 0 for v in voltages)
    return ok, "needs at least one day of stimulation"


def _coverage(item: str, datasets: list[dict]) -> tuple[bool, str]:
    """Is there enough data to say anything about the item?"""
    entries = {(str(d.get("to_port")), str(d.get("from_port"))) for d in datasets
               if d["kind"] != "current"}
    if sum(a == b for a, b in entries) < 2:
        return False, "needs impedance data for both reflection entries (Z11 and Z22)"
    times = np.concatenate([np.asarray(d["time_s"], dtype=float) for d in datasets])
    span = float(times.max() - times.min())
    needed = {"hysteresis": 150.0, "pulse_duration": 0.0,
              "long_term_drift": DRIFT_MIN_DURATION_S}[item]
    if span < needed or len(np.unique(times)) < 5:
        return False, f"data must hold at least 5 read times spanning {needed:g} s"
    return True, ""


def _model_behaviour(model: dict, options: dict) -> dict[str, Any]:
    """What the user's parameters predict for the cited protocols (trends only)."""
    device = StateSpaceDevice(model)
    if len(device.ports) != 2:
        raise ValueError("the reference experiment has two ports")
    drive = str(options.get("drive_port", device.ports[0]))
    other_port = device.ports[1 - device.index(drive)]
    other = options.get("other_port", "ground")
    temperature = options.get("temperature_k")
    iterations = int(options.get("iterations", 50))
    if not 2 <= iterations <= 50:
        raise ValueError("iterations must be 2..50")
    points = int(options.get("frequency_points", 201))
    if not 2 <= points <= 2001:
        raise ValueError("frequency_points must be 2..2001")
    definition = options.get("zc_definition", "sum_of_magnitudes")
    if definition not in experiment.ZC_DEFINITIONS:
        raise ValueError(f"zc_definition must be one of {experiment.ZC_DEFINITIONS}")
    grid = np.linspace(1e7, 6e9, points)

    def summed(snapshot: dict) -> np.ndarray:
        z = device.impedance_matrix(grid, snapshot["state"], snapshot["node_v"],
                                    snapshot["temperature_k"])
        return np.sum(np.abs(z) if definition == "sum_of_magnitudes" else z.real, axis=0)

    protocol = hysteresis_protocol(drive, other_port, other, iterations, temperature)
    per_sweep = len(protocol["segments"])
    times = []
    for sweep in (0, iterations - 1):
        times += [float(sweep * per_sweep + k + 1) for k in range(per_sweep)]
    snapshots = run_protocol(device, protocol, times)
    up = np.array([s["voltages_v"][drive] for s in protocol["segments"]])
    out: dict[str, Any] = {"assumed": {
        "drive_port": drive, "other_port": other, "sweep_order": "up then down",
        "frequency_points": points, "zc_definition": definition,
        "initial_state": "unwritten", "iterations": iterations}}
    loops = {}
    curves = {}
    for name, snaps in (("first", snapshots[:per_sweep]), ("last", snapshots[per_sweep:])):
        z = np.array([summed(s) for s in snaps])
        curves[name] = z
    for i, a in enumerate("12"):
        for j, b in enumerate("12"):
            entry = {}
            for name, z in curves.items():
                value = z[:, i, j]
                for side, mask in (("positive", up >= 0), ("negative", up <= 0)):
                    # |area| enclosed between the rising and falling branches on one side
                    rising = {round(float(v), 3): value[k] for k, v in enumerate(up[:77])
                              if mask[k]}
                    falling = {round(float(up[k]), 3): value[k]
                               for k in range(77, per_sweep) if mask[k]}
                    shared = sorted(set(rising) & set(falling))
                    gap = np.array([abs(rising[v] - falling[v]) for v in shared])
                    entry[f"{name}_{side}_loop_area_ohm_v"] = float(
                        np.sum(0.5 * (gap[1:] + gap[:-1]) * np.diff(shared)))
            for side in ("positive", "negative"):
                first = entry[f"first_{side}_loop_area_ohm_v"]
                last = entry[f"last_{side}_loop_area_ohm_v"]
                entry[f"{side}_loop_shrinks"] = bool(first > 0 and last < first)
            loops[f"Z{a}{b}"] = entry
    out["hysteresis"] = {
        "loops": loops,
        "curve": {"vp_v": up.tolist(), "z11_first_ohm": curves["first"][:, 0, 0].tolist(),
                  "z11_last_ohm": curves["last"][:, 0, 0].tolist(),
                  "z22_last_ohm": curves["last"][:, 1, 1].tolist()}}
    durations = options.get("pulse_durations_s")
    if durations:
        rows = []
        for duration in durations:
            p = pulse_protocol(drive, other_port, duration, max(duration, 1e-9), other,
                               temperature)
            written, held = run_protocol(device, p, [float(duration),
                                                     float(duration) + max(duration, 1e-9)])
            rows.append({"duration_s": float(duration),
                         "z22_after_write_ohm": float(summed(written)[1, 1]),
                         "z22_after_equal_hold_ohm": float(summed(held)[1, 1])})
        out["pulse_duration"] = {"rows": rows}
    else:
        out["pulse_duration"] = {"status": "not run: give pulse_durations_s (the cited "
                                           "durations are not in the main text)"}
    out["long_term_drift"] = {
        "status": "not run: needs the multi-day drive trace",
        "fatigue_enabled": bool(np.any(device.p["fatigue_depth"] > 0))}
    return out


def reproduce(model: dict | None = None, measurements: dict | list | None = None,
              options: dict | None = None) -> dict[str, Any]:
    """Status of the reproduction: per item, data against model, or what is missing."""
    options = options or {}
    allowed = {"drive_port", "other_port", "temperature_k", "iterations",
               "frequency_points", "zc_definition", "pulse_durations_s",
               "skip_model_behaviour"}
    if set(options) - allowed:
        raise ValueError(f"unknown option(s) {sorted(set(options) - allowed)}")
    comparison = None
    records = experiment.parse_records(measurements) if measurements else []
    for record in records:
        for data, _ in record.datasets:
            tag = data.get("reference_item")
            if tag is not None and tag not in REQUIRED_ITEMS:
                raise ValueError(f"reference_item must be one of {list(REQUIRED_ITEMS)}")
    if records and model is not None:
        comparison = experiment.compare(model, measurements)
    items = {}
    for item, need in REQUIRED_ITEMS.items():
        rows = [r for r in (comparison or {}).get("datasets", [])
                if r["reference_item"] == item]
        holders = [k for k, record in enumerate(records)
                   if any(d.get("reference_item") == item for d, _ in record.datasets)]
        entry: dict[str, Any] = {"needs": need}
        if not holders:
            entry["status"] = "reference_data_missing"
        elif model is None:
            entry["status"] = "model_parameters_missing"
        else:
            checks = [_protocol_matches(item, records[k].protocol) for k in holders]
            covered, why = _coverage(item, [
                d for k in holders for d, _ in records[k].datasets
                if d.get("reference_item") == item])
            if not all(ok for ok, _ in checks):
                entry.update(status="protocol_differs",
                             reason=next(r for ok, r in checks if not ok))
            elif not covered:
                entry.update(status="insufficient_coverage", reason=why)
            elif not all(r["independent_of_calibration"] for r in rows):
                entry["status"] = "not_independent"
            else:
                entry["status"] = "agrees" if all(r["agrees"] for r in rows) else "disagrees"
            entry["datasets"] = rows
        items[item] = entry
    reproduced = all(e["status"] == "agrees" for e in items.values())
    behaviour: dict[str, Any] | None = None
    if model is not None and not options.get("skip_model_behaviour"):
        behaviour = _model_behaviour(model, options)
    return {
        "status": "reproduced_on_supplied_data" if reproduced else "not_reproduced",
        "items": items,
        "model_behaviour": behaviour if behaviour is not None else {
            "status": "not run: no model parameters (this model has no defaults)"
            if model is None else "skipped"},
        "model_behaviour_note": "predictions of your parameters; resembling a reported "
                                "trend is not reproduction",
        "citation": CITATION,
        "reported_in_source": REPORTED,
        "missing_from_source": NOT_REPRODUCIBLE_FROM_TEXT,
        "data_to_request": DATA_REQUEST,
        "limitations": [
            "a pass means agreement with the files supplied; ChipLab cannot verify they are "
            "the published measurements",
            "parameters adjusted by hand on the same data are not detected as dependent",
            *NOT_MODELLED],
    }
