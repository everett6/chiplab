"""eCore (Echo Lab analog DRAM CIM) backend. Echo Lab-dependent tests skip when it's absent."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest

from chiplab.backends import EcoreBackend, EcoreUnavailableError, backend_status
from chiplab.backends import ecore as ecore_mod
from chiplab.schemas.design import DesignSpec
from chiplab.schemas.simulation import SimulationConfig

ROOT = Path(__file__).resolve().parents[1]
ECHO_ECORE_YAML = ROOT / "projects" / "echo" / "echo_v0_1_ecore.yaml"
ECHO_YAML = ROOT / "projects" / "echo" / "echo_v0_1.yaml"

needs_echo_lab = pytest.mark.skipif(not EcoreBackend().is_available(),
                                    reason="Echo Lab eCore not available")


# --- Always-run tests (no Echo Lab needed) ---------------------------------------------------

def test_module_has_no_shell_eval_exec_or_subprocess():
    tree = ast.parse(Path(ecore_mod.__file__).read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    imports = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
               for a in n.names}
    assert not {"eval", "exec", "compile", "__import__"} & names
    assert not {"system", "popen", "Popen", "run", "call", "check_output"} & attrs - {"run"}
    assert "subprocess" not in imports and "os.system" not in attrs


@pytest.mark.parametrize("kw", [
    {"rows": 4}, {"cols": 4096}, {"precision": "bf16"}, {"adc_bits": 7},
    {"organization": "org_c"}, {"rows": True}, {"seed": -1},
])
def test_invalid_arguments_rejected(kw):
    with pytest.raises(ValueError):
        EcoreBackend().simulate(**kw)


def test_unavailable_raises_and_never_fakes_success(tmp_path, monkeypatch):
    monkeypatch.setenv("CHIPLAB_ECHO_LAB_PATH", str(tmp_path / "missing"))
    ecore_mod._load_echo_lab.cache_clear()
    try:
        be = EcoreBackend()
        assert be.is_available() is False
        st = be.status()
        assert st["usable"] is False and "reason" in st
        with pytest.raises(EcoreUnavailableError):
            be.simulate()
        design = DesignSpec.from_yaml(ECHO_ECORE_YAML.read_text())
        with pytest.raises(EcoreUnavailableError):
            be.run(design, None, SimulationConfig())
        from chiplab.engine import run_analytical_simulation
        with pytest.raises(EcoreUnavailableError):
            run_analytical_simulation(design, None, SimulationConfig())
    finally:
        monkeypatch.delenv("CHIPLAB_ECHO_LAB_PATH")
        ecore_mod._load_echo_lab.cache_clear()


def test_missing_calibration_is_unavailable(tmp_path, monkeypatch):
    fake = tmp_path / "Echo Lab"
    (fake / "simulator" / "ecore").mkdir(parents=True)
    (fake / "simulator" / "ecore" / "ecore.py").write_text("")
    monkeypatch.setenv("CHIPLAB_ECHO_LAB_PATH", str(fake))
    ecore_mod._load_echo_lab.cache_clear()
    try:
        assert "calibration" in (EcoreBackend().unavailable_reason() or "")
    finally:
        monkeypatch.delenv("CHIPLAB_ECHO_LAB_PATH")
        ecore_mod._load_echo_lab.cache_clear()


def test_designs_without_ecore_are_untouched():
    design = DesignSpec.from_yaml(ECHO_YAML.read_text())
    out, recs, warns = ecore_mod.apply_ecore_to_design(design)
    assert out is design and recs == [] and warns == []


def test_backend_status_includes_ecore():
    assert "ecore" in {s["name"] for s in backend_status()}


def test_mcp_tool_registered_read_only():
    import asyncio

    from chiplab.mcp_server import create_mcp_server
    tools = {t.name: t for t in asyncio.run(create_mcp_server().list_tools())}
    ann = tools["simulate_ecore_array"].annotations
    assert getattr(ann, "read_only_hint", getattr(ann, "readOnlyHint", None)) is True


# --- Echo Lab-backed tests -------------------------------------------------------------------

@needs_echo_lab
def test_simulate_returns_full_tagged_result():
    r = EcoreBackend().simulate(rows=64, cols=64, precision="int8", adc_bits=8,
                                organization="org_b")
    assert r["success"] is True and r["fidelity"] == "specialized"
    acc = r["accuracy"]
    for k in ("snr_db_vs_reference", "rel_l2_error", "rmse", "max_abs_error"):
        assert k in acc
    assert acc["method"] == "ecore_bitlevel_signed_int_matvec"
    for block in ("latency_ns", "energy_fj", "area_um2"):
        tot = r[block]["total"]
        assert tot["value"] > 0 and tot["source"].startswith("echo_lab:")
        assert tot["provenance"] in ("estimated", "assumed")
        assert r[block]["breakdown"]
    cal = r["calibration"]
    assert cal["kind"] == "spice" and cal["measured_silicon"] is False
    assert cal["date"] and len(cal["sha256"]) == 64
    assert r["limitations"] and "Level-1" in r["limitations"][0]
    d = r["derived"]
    assert math.isclose(d["array_tops"]["value"],
                        2 * 64 * 64 / r["latency_ns"]["total"]["value"] / 1e3)
    json.dumps(r)  # MCP-serializable


@needs_echo_lab
def test_deterministic_for_same_seed():
    be = EcoreBackend()
    a = be.simulate(rows=32, cols=32, seed=7)
    b = be.simulate(rows=32, cols=32, seed=7)
    assert a == b
    a["accuracy"]["rmse"] = -1.0  # returned copies are independent of the cache
    assert be.simulate(rows=32, cols=32, seed=7)["accuracy"]["rmse"] != -1.0


@needs_echo_lab
def test_fp_path_is_labeled_statistical():
    r = EcoreBackend().simulate(rows=64, cols=64, precision="fp8_e4m3")
    assert r["accuracy"]["method"] == "calibrated_statistical_fp_model"
    assert any("statistical" in lim for lim in r["limitations"])


@needs_echo_lab
def test_poor_end_to_end_accuracy_is_reported_not_hidden():
    r = EcoreBackend().simulate(rows=64, cols=64, precision="int8")
    snr = r["accuracy"]["snr_db_vs_reference"]
    if snr is not None and snr < 10.0:
        assert any("does not reproduce W@x" in lim for lim in r["limitations"])
    assert "Not an end-to-end" in r["per_mac_snr_db"]["note"]


@needs_echo_lab
def test_pim_chiplet_takes_numbers_from_ecore():
    design = DesignSpec.from_yaml(ECHO_ECORE_YAML.read_text())
    out, recs, _ = ecore_mod.apply_ecore_to_design(design)
    pim = next(c for c in out.chiplets if c.id == "pim_0")
    prov = out.metadata["ecore_provenance"]["pim_0"]
    r = EcoreBackend().simulate(rows=256, cols=256, precision="int8")
    assert prov["arrays"] == int(250.0 * 0.6 // r["derived"]["array_area_mm2"]["value"])
    assert math.isclose(pim.compute.peak_tops, prov["arrays"] * r["derived"]["array_tops"]["value"])
    assert math.isclose(pim.tdp_watts, prov["arrays"] * r["derived"]["array_power_w"]["value"])
    assert pim.area_mm2 == 250.0  # area kept when array count is derived from it
    assert {rec.parameter for rec in recs} >= {"pim_0.peak_tops", "pim_0.energy_pj_per_op",
                                               "pim_0.tdp_watts"}
    assert all(rec.source.startswith("ecore:") for rec in recs
               if rec.parameter.endswith(("peak_tops", "tdp_watts", "energy_pj_per_op")))
    assert prov["calibration"]["kind"] == "spice"


@needs_echo_lab
def test_explicit_array_count_sets_area():
    data = DesignSpec.from_yaml(ECHO_ECORE_YAML.read_text()).model_dump()
    pim = next(c for c in data["chiplets"] if c["id"] == "pim_0")
    pim["compute"]["ecore"]["arrays"] = 100
    out, _, _ = ecore_mod.apply_ecore_to_design(DesignSpec.model_validate(data))
    r = EcoreBackend().simulate(rows=256, cols=256, precision="int8")
    got = next(c for c in out.chiplets if c.id == "pim_0").area_mm2
    assert math.isclose(got, 100 * r["derived"]["array_area_mm2"]["value"] / 0.6)


@needs_echo_lab
def test_echo_design_evaluates_with_ecore_numbers(tmp_path, monkeypatch):
    from chiplab.experiments.store import ExperimentStore
    from chiplab import mcp_server
    monkeypatch.setattr(mcp_server, "store", ExperimentStore(db_path=str(tmp_path / "x.db")))
    res = mcp_server.chip_evaluate(spec=ECHO_ECORE_YAML.read_text())
    assert res["valid"], res["validation_errors"]
    assert "eCore" in res["fidelity_disclaimer"]
    assert any("from eCore" in w for w in res["warnings"])
    sim = mcp_server.store.get_simulation(res["run_id"])
    assert any(a.source and a.source.startswith("ecore:") for a in sim.assumptions)
    assert "pim_0" in sim.raw_details["ecore"]
    # Same design without eCore uses the hand-entered 50 TOPS PIM; eCore changes the total.
    base = mcp_server.chip_evaluate(spec=ECHO_YAML.read_text())
    assert res["performance"]["peak_compute_tops"] != base["performance"]["peak_compute_tops"]


@needs_echo_lab
def test_backend_run_labels_package_level_analytical():
    design = DesignSpec.from_yaml(ECHO_ECORE_YAML.read_text())
    res = EcoreBackend().run(design, None, SimulationConfig())
    assert res.success and res.fidelity.value == "analytical"
    assert "pim_0" in res.raw_details["ecore"]


@needs_echo_lab
def test_mcp_tool_returns_result():
    from chiplab.mcp_server import simulate_ecore_array
    r = simulate_ecore_array(rows=32, cols=32, precision="int4", adc_bits=6,
                             organization="org_a")
    assert r["inputs"]["weight_bits"] == 4 and r["inputs"]["organization"] == "org_a"
