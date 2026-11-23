"""New MCP tools (twin, calibration, PDK, handoff, IP), activity log and the live app."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

import chiplab.mcp_server as srv
from chiplab.activity import ActivityLog, ActivityMiddleware
from chiplab.app.server import create_app, summarize
from chiplab.experiments.store import ExperimentStore

FIX = Path(__file__).parent / "fixtures" / "eda"


@pytest.fixture()
def db(monkeypatch, tmp_path):
    monkeypatch.setenv("CHIPLAB_PRICE_FILE", str(tmp_path / "prices.json"))
    monkeypatch.setenv("CHIPLAB_EXPORT_DIR", str(tmp_path / "exports"))
    monkeypatch.setenv("CHIPLAB_DATA_DIRS", str(FIX))
    monkeypatch.setenv("CHIPLAB_PRIVATE_PDK_DIR", str(tmp_path / "private_pdks"))
    fd, path = tempfile.mkstemp(suffix=".db", dir=tmp_path)
    os.close(fd)
    monkeypatch.setattr(srv, "store", ExperimentStore(db_path=path))
    return path


SMALL = {"population": 16, "generations": 4, "mc_trials": 48, "top_k": 3}


def test_run_digital_twin_search_is_compact_and_persisted(db):
    r = srv.run_digital_twin_search(**SMALL)
    assert len(r["top"]) == 3 and r["top"][0]["rank"] == 1
    assert set(r["top"][0]["key_metrics"]) <= set(srv._TWIN_KEY_METRICS)
    assert "metrics" not in r["top"][0]
    assert len(json.dumps(r)) < 8000  # token-efficient
    assert all("most_affects" in s for s in r["sensitivity"])
    saved = srv.get_digital_twin_run(r["run_id"], include_rows=5)
    assert saved["summary"]["run_id"] == r["run_id"] and len(saved["trace_rows"]) == 5
    assert saved["trace_total"] == r["evaluated"]


def test_twin_config_dict_and_blackboxes(db):
    cfg = {"objectives": [{"name": "perf_per_watt", "goal": "max"},
                          {"name": "area_mm2", "goal": "min"}],
           "constraints": {"max_temp_c": 85}, "workload": {"vector_fraction": 0.8}}
    r = srv.run_digital_twin_search(config=cfg, blackboxes=["example_ucie_phy_x16"], **SMALL)
    assert set(r["top"][0]["objectives"]) == {"perf_per_watt", "area_mm2"}
    assert all(p["key_metrics"]["area_mm2"] > 0.44 for p in r["top"])  # PHY area included


def test_ingest_byo_pdk_and_use_calibration(db):
    pdk = srv.ingest_byo_pdk("N7", liberty_path=str(FIX / "mini.lib"),
                             lef_path=str(FIX / "mini.lef"), private=False)
    names = {f["name"] for f in pdk["factors"]}
    assert names == {"gate_delay", "leakage"}
    assert pdk["blackbox_candidates"][0]["name"] == "UCIE_PHY_X16"
    assert pdk["liberty"]["fo4_ps"] == pytest.approx(7.0)
    r = srv.run_digital_twin_search(calibration_id=pdk["calibration_id"], **SMALL)
    assert r["calibrated_parameters"] == ["gate_delay", "leakage"]
    assert "leakage" not in r["unknown_parameters"]


def test_calibrate_from_eda_report_inline_and_path(db):
    rep = srv.calibrate_from_eda_report("N7", timing_report=(FIX / "opensta.rpt").read_text(),
                                        spef_path=str(FIX / "mini.spef"))
    assert {"gate_delay", "wire_rc"} <= {f["name"] for f in rep["factors"]}
    assert rep["timing_paths"][0]["status"] == "MET"
    assert srv.store.get_calibration(rep["calibration_id"])["node"] == "N7"
    with pytest.raises(PermissionError):
        srv.calibrate_from_eda_report("N7", timing_report_path="/etc/hosts.rpt")
    with pytest.raises(ValueError):
        srv.calibrate_from_eda_report("N7")


def test_export_pareto_handoff_bundle(db):
    r = srv.run_digital_twin_search(blackboxes=["example_lpddr5x_phy"], **SMALL)
    b = srv.export_pareto_handoff_bundle(r["run_id"], rank=2)
    assert b["structural_lint"] == ["PASS"] and "constraints.sdc" in b["files"]
    sv = (Path(b["directory"]) / "soc_top.sv").read_text()
    assert "example_lpddr5x_phy" in sv  # black boxes carried over from the run config
    with pytest.raises(ValueError):
        srv.export_pareto_handoff_bundle(r["run_id"], rank=99)


def test_list_ip_catalog(db):
    cat = srv.list_ip_catalog("core")
    assert {i["name"] for i in cat["ip"]} >= {"cva6", "rocket", "xiangshan", "ibex"}
    assert "example_ucie_phy_x16" in cat["blackbox_examples"]


def test_new_tools_registered():
    tools = asyncio.run(srv.create_mcp_server().list_tools())
    names = {t.name for t in tools}
    assert {"run_digital_twin_search", "calibrate_from_eda_report", "ingest_byo_pdk",
            "export_pareto_handoff_bundle", "list_ip_catalog", "get_digital_twin_run"} <= names


# --- activity log + middleware -------------------------------------------------------------

def test_activity_middleware_records_client_and_calls(tmp_path):
    log = ActivityLog(str(tmp_path / "a.db"))
    mw = ActivityMiddleware(log)

    async def ok(ctx):
        return {"content": [{"type": "text", "text": json.dumps({"design_id": "d1", "x": 1})}]}

    async def boom(ctx):
        raise RuntimeError("bad")

    async def run():
        await mw(SimpleNamespace(method="initialize", session_id=None,
                                 params={"clientInfo": {"name": "codex"}}), ok)
        await mw(SimpleNamespace(method="tools/call", session_id=None,
                                 params={"name": "chip_evaluate", "arguments": {"a": 1}}), ok)
        with pytest.raises(RuntimeError):
            await mw(SimpleNamespace(method="tools/call", session_id=None,
                                     params={"name": "chip_get_design", "arguments": {}}), boom)
        await mw(SimpleNamespace(method="tools/list", session_id=None, params={}), ok)

    asyncio.run(run())
    ev = log.recent()
    assert [e["tool"] for e in ev] == ["chip_evaluate", "chip_get_design"]
    assert ev[0]["client"] == "codex" and ev[0]["design_id"] == "d1" and ev[0]["ok"] == 1
    assert ev[1]["ok"] == 0 and "bad" in ev[1]["error"]
    assert log.since(ev[0]["id"])[0]["tool"] == "chip_get_design"


# --- live app ------------------------------------------------------------------------------

def test_app_endpoints(db):
    srv.chip_design_from_template("edge_npu", name="npu")
    r = srv.run_digital_twin_search(**SMALL)
    log = ActivityLog(db)
    log.record("chip_evaluate", {"design_id_or_name": "npu_0.1"},
               {"design_id": "npu_0.1", "cost": {"unit_cost_usd": 80},
                "performance": {"throughput_tokens_per_sec": 40, "peak_compute_tops": 40},
                "power_thermal": {"total_power_watts": 5}, "constraint_checks": []},
               client="claude-code")
    log.record("run_digital_twin_search", {}, r, client="codex")
    client = TestClient(create_app(db))
    assert "ChipLab Live" in client.get("/live").text
    assert "GPU Studio" in client.get("/").text
    ev = client.get("/api/events").json()
    assert [e["tool"] for e in ev] == ["chip_evaluate", "run_digital_twin_search"]
    assert ev[0]["summary"].startswith("$80/unit") and "designs" in ev[1]["summary"]
    assert client.get(f"/api/events?after={ev[0]['id']}").json()[0]["tool"] == \
        "run_digital_twin_search"
    designs = client.get("/api/designs").json()
    assert designs[0]["design_id"] == "npu_0.1" and designs[0]["evaluated"]
    d = client.get("/api/designs/npu_0.1").json()
    assert d["cost"]["unit_cost_usd"] > 0 and d["evaluation"]["cost"]["unit_cost_usd"] == 80
    assert client.get("/api/designs/missing").status_code == 404
    runs = client.get("/api/twin_runs").json()
    assert runs[0]["run_id"] == r["run_id"]
    tr = client.get(f"/api/twin_runs/{r['run_id']}").json()
    assert tr["points"] and any(p["front"] for p in tr["points"])
    assert client.get("/api/twin_runs/nope").status_code == 404
    st = client.get("/api/status").json()
    assert st["designs"] == 1 and st["twin_runs"] == 1 and "claude-code" in st["active_agents"]
    assert client.get("/api/pareto").json()[0]["design_id"] == "npu_0.1"


def test_feed_summaries():
    base = {"ok": 1, "args": {}}
    assert "fails power_w" in summarize({**base, "tool": "chip_evaluate", "result": {
        "cost": {"unit_cost_usd": 10}, "constraint_checks": [{"constraint": "power_w",
                                                              "pass": False}]}})
    assert summarize({"ok": 0, "tool": "x", "error": "boom"}) == "boom"
    assert "gate_delay=1.1" in summarize({**base, "tool": "ingest_byo_pdk", "result": {
        "node": "N7", "factors": [{"name": "gate_delay", "value": 1.1}]}})
