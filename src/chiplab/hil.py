"""Hardware-in-the-loop lab (FPGA first).

ChipLab cannot reach a board by itself. This module closes the loop once the user has run a
design on real hardware:

  * parse_fpga_report(text)  pulls Fmax, timing slack and resource use out of nextpnr,
                             Vivado or Quartus report text.
  * import_measurements(...) turns a CSV of bench measurements (one row per prediction) into
                             learning-loop measurements with source_kind="fpga". The loop's
                             own checks still apply: attestation, method, date, outliers.
  * fpga_plan(design)        lists what to measure and which ChipLab predictions to record
                             first, so every measurement has something to correct.

Numbers parsed from a synthesis/place-and-route report are tool output, not measurements:
they are returned as `fidelity="place_route"` evidence for the ledger, never fed to the
learning loop as measured data.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Any

_FMAX = [re.compile(r"Max frequency for clock\s+'?[\w$.\[\]]+'?:\s*([\d.]+)\s*MHz"),  # nextpnr
         re.compile(r"Fmax\s*[:=]\s*([\d.]+)\s*MHz", re.I),                                # quartus
         re.compile(r"([\d.]+)\s*MHz\s+[\d.]+\s*MHz\s+\w+", re.I)]
_WNS = [re.compile(r"WNS\(ns\)\s+TNS\(ns\).*?\n[-\s]+\n\s*(-?[\d.]+)", re.S),             # vivado
        re.compile(r"Worst Negative Slack\s*[:=]\s*(-?[\d.]+)", re.I)]
_RES = {"luts": re.compile(r"(?:Slice LUTs\*?|ICESTORM_LC|LUT4)\s*[|:]\s*(\d+)"),
        "ffs": re.compile(r"(?:Slice Registers|Register as Flip Flop|DFF)\s*[|:]\s*(\d+)"),
        "brams": re.compile(r"(?:Block RAM Tile|ICESTORM_RAM|RAMB\w*)\s*[|:]\s*(\d+)")}


def parse_fpga_report(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"fidelity": "place_route", "is_measurement": False}
    fmax = [float(m) for rx in _FMAX for m in rx.findall(text)]
    if fmax:
        out["fmax_mhz"] = min(fmax)
    for rx in _WNS:
        m = rx.search(text)
        if m:
            out["wns_ns"] = float(m.group(1))
            break
    for k, rx in _RES.items():
        m = rx.search(text)
        if m:
            out[k] = int(m.group(1))
    if len(out) == 2:
        out["warning"] = "no Fmax, slack or resource figures recognised"
    return out


REQUIRED_COLS = ("prediction_id", "value", "unit", "method", "measured_on")


def import_measurements(csv_text: str, loop: Any, *, submitted_by: str, board: str,
                        attested_real: bool) -> dict[str, Any]:
    """Submit each CSV row to the learning loop as an FPGA measurement.

    Columns: prediction_id, value, unit, method, measured_on, and optionally uncertainty,
    repeats (semicolon-separated), reference."""
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    if not rows:
        return {"accepted": 0, "results": [], "error": "no rows"}
    missing = [c for c in REQUIRED_COLS if c not in rows[0]]
    if missing:
        return {"accepted": 0, "results": [], "error": f"missing columns {missing}"}
    results = []
    for i, r in enumerate(rows, 1):
        try:
            reps = [float(x) for x in (r.get("repeats") or "").split(";") if x.strip()]
            unc = float(r["uncertainty"]) if (r.get("uncertainty") or "").strip() else None
            res = loop.submit_measurement(
                r["prediction_id"].strip(), float(r["value"]), r["unit"].strip(),
                source_kind="fpga", method=f"{r['method'].strip()} on {board}",
                measured_on=r["measured_on"].strip(), submitted_by=submitted_by,
                attested_real=attested_real, uncertainty=unc, repeats=reps or None,
                reference=(r.get("reference") or "").strip())
        except Exception as e:  # noqa: BLE001 - report the row, keep going
            res = {"status": "rejected", "reasons": [repr(e)]}
        results.append({"row": i} | res)
    ok = sum(1 for r in results if r.get("status") == "validated")
    return {"accepted": ok, "not_accepted": len(results) - ok, "results": results}


def fpga_plan(design: str = "sram_nmc") -> dict[str, Any]:
    return {
        "design": design,
        "steps": [
            "Record ChipLab predictions first (learning loop record_prediction) for each metric "
            "below, so measurements have something to correct.",
            "Build the RTL for the board (yosys + nextpnr for iCE40/ECP5, or vendor tools); "
            "keep the report text and import it with parse_fpga_report (place_route evidence).",
            "Run the self-checking testbench pattern on the board through UART/JTAG; a single "
            "mismatch fails the run.",
            "Measure and write one CSV row per prediction, then import_measurements.",
        ],
        "measure": [
            {"metric": "fmax_mhz", "unit": "MHz", "method": "frequency sweep until first "
             "functional failure, 3 boards or 3 repeats"},
            {"metric": "cycles_per_op", "unit": "cycle", "method": "on-chip cycle counter"},
            {"metric": "board_power_w", "unit": "W", "method": "shunt / USB power meter, "
             "idle subtracted"},
            {"metric": "functional_errors", "unit": "count", "method": "testbench mismatches "
             "over N random vectors"},
        ],
        "caveat": "FPGA timing and power do not transfer to an ASIC; use FPGA data to correct "
                  "functional/cycle models, not ASIC energy models.",
    }
