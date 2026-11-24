"""TT GitHub Actions hardening import: log parsing on synthetic logs and report wiring."""

from __future__ import annotations

import pytest

from chiplab.asic import flow, hardening, report

PFX = "gds\tBuild GDS\t2026-11-24T10:30:50.1234567Z "


def _log(*lines: str) -> str:
    return "\n".join(PFX + ln for ln in lines)


FAILED = _log(
    "[INFO IFP-0104] Effective utilization:                0.352",
    "[10:29:49] INFO     Check for Magic DRC errors clear.             checker.py:131",
    "[10:30:50] INFO     Check for LVS errors clear.                   checker.py:131",
    "[10:30:50] WARNING  Max Slew violations found in the following    checker.py:626",
    "Pin violations: 1", "Net violations: 1",
    "[10:30:50] ERROR    The following error was encountered while    __main__.py:189",
    "                    Setup violations found in the following                     ",
    "                    corners:                                                    ",
    "                    * nom_slow_1p08V_125C                                       ",
    "[10:30:50] ERROR    LibreLane will now quit.                     __main__.py:190",
    "2026-11-24 10:30:52,038 - project    - ERROR    - harden failed",
)

PASSED = _log(
    "Check for Magic DRC errors clear.", "Check for LVS errors clear.",
    "timing__setup__ws,3.25", "timing__hold__ws,0.12", "design__instance__utilization,0.41",
    "design__instance__area__stdcell,512345.6", "power__total,0.0123", "route__drc_errors,0", "timing__setup__tns,nan",
)


def test_parse_failed_run():
    p = hardening.parse_log(FAILED)
    assert p["routed"] and p["drc_clear"] and p["lvs_clear"] and not p["flow_completed"]
    assert p["setup_violation_corners"] == ["nom_slow_1p08V_125C"] and p["hold_violation_corners"] == []
    assert (p["antenna_pins"], p["antenna_nets"], p["effective_utilization"]) == (1, 1, 0.352)
    assert p["max_slew_violations"] and not p["max_cap_violations"] and p["metrics"] == {}


def test_parse_metrics_skips_nan():
    p = hardening.parse_log(PASSED)
    assert p["flow_completed"] and p["metrics"]["timing__setup__ws"] == 3.25
    assert "timing__setup__tns" not in p["metrics"] and p["metrics"]["route__drc_errors"] == 0


def test_parse_empty_log_reports_nothing():
    p = hardening.parse_log("")
    assert not p["routed"] and not p["drc_clear"] and p["antenna_pins"] is None and p["effective_utilization"] is None


def test_import_rejects_bad_run_id(tmp_path):
    with pytest.raises(ValueError):
        hardening.import_run(-3, repo_dir=tmp_path)
    assert hardening.import_run(None, repo_dir=tmp_path / "missing")["ok"] is False


def _rec(parsed, conclusion, jobs):
    return {"ok": True, "run_id": 7, "url": "u", "conclusion": conclusion, "jobs": jobs, "clock_mhz": 25.0,
            "rtl_matches_repo": True, **parsed, "utilization": parsed["metrics"].get(
                "design__instance__utilization", parsed["effective_utilization"]),
            "utilization_source": "x", "setup_wns_ns": parsed["metrics"].get("timing__setup__ws"),
            "instance_area_um2": parsed["metrics"].get("design__instance__area__stdcell"),
            "power_w": parsed["metrics"].get("power__total"), "route_drc_errors": parsed["metrics"].get("route__drc_errors"), "clock_period_ns": 40.0}


def test_report_uses_failed_run(monkeypatch):
    rec = _rec(hardening.parse_log(FAILED), "failure", {"gds": "failure", "precheck": "skipped"})
    monkeypatch.setattr(report, "_hardening", lambda: rec)
    assert report._r8_status().startswith("met")
    assert report._r9(True)["status"] == "failing: setup at nom_slow_1p08V_125C"
    m = report._hardening_metrics(rec)
    assert m["tile_utilization"]["value"] == 0.352 and m["timing_margin_ns"]["value"] is None
    steps = [{"step": k, "status": "ready", "result": None} for k, *_ in flow.FLOW_STEPS]
    flow._apply_hardening(steps, rec)
    st = {s["step"]: s["status"] for s in steps}
    assert st["routing"].startswith("passed") and st["sta"].startswith("failed") and st["gds"].startswith("failed")
    assert st["tt_precheck"].startswith("not run") and st["lint"] == "ready"


def test_report_uses_passed_run(monkeypatch):
    rec = _rec(hardening.parse_log(PASSED), "failure",
               {"gds": "success", "precheck": "success", "gl_test": "success", "viewer": "failure"})
    monkeypatch.setattr(report, "_hardening", lambda: rec)
    assert report._r9(True)["status"] == "met"          # a failed Pages viewer job does not fail the flow
    assert report._flow_items(lambda n, ok, e: ok, {})[0] is True
    m = report._hardening_metrics(rec)
    assert m["timing_margin_ns"]["value"] == 3.25 and m["power_mw"]["value"] == 12.3
    assert m["fmax_mhz"]["value"] == round(1000 / (40.0 - 3.25), 1)
    assert m["routing_congestion"]["value"] == 0 and m["mapped_cell_area_um2"]["value"] == 512345.6
