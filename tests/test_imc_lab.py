"""IMC lab: designs are checked by real tools and by independent references."""

from __future__ import annotations

import numpy as np
import pytest

from chiplab import imc_lab
from chiplab.asic import flow
from chiplab.imc_lab import analog, digital
from chiplab.learning import LearningLoop
from chiplab.tools import registry

HAVE_IVERILOG = bool(flow._which("iverilog") and flow._which("vvp"))
HAVE_YOSYS = bool(flow._which("yosys")) and digital.LIBERTY.is_file()
HAVE_NGSPICE = registry.get("ngspice").available


@pytest.mark.skipif(not HAVE_IVERILOG, reason="iverilog not installed")
@pytest.mark.parametrize("spec", [{"rows": 1, "cols": 1, "wbits": 2, "xbits": 1},
                                  {"rows": 3, "cols": 5, "wbits": 3, "xbits": 2},
                                  {"rows": 8, "cols": 8, "wbits": 8, "xbits": 8}])
def test_digital_rtl_bit_exact_against_numpy(spec):
    r = digital.simulate(spec, n_vectors=12, seed=3)
    assert r["bit_exact"] and r["outputs_checked"] == 12 * spec["rows"]
    assert r["cycles_per_mvm"] == digital.predict(spec)["cycles_per_mvm"]["value"]


def test_digital_accumulator_holds_worst_case():
    p = digital.validate({"rows": 2, "cols": 16, "wbits": 8, "xbits": 8})
    worst = 16 * 128 * 255
    assert -(1 << (digital.acc_bits(p) - 1)) <= -worst


@pytest.mark.skipif(not HAVE_YOSYS, reason="yosys or IHP liberty missing")
def test_digital_synthesis_flop_count_matches_structure():
    spec = {"rows": 2, "cols": 3, "wbits": 3, "xbits": 2}
    syn = digital.synthesize(spec)
    assert syn["flip_flops"] == digital.predict(spec)["flip_flops"]["value"]
    assert syn["cells"] > syn["flip_flops"] and syn["cell_area_um2"] > 0


def test_digital_rejects_out_of_range():
    with pytest.raises(digital.DigitalCimError):
        digital.validate({"rows": 17})


@pytest.mark.skipif(not HAVE_NGSPICE, reason="ngspice not installed")
def test_analog_ngspice_and_nodal_solver_agree():
    e = analog.evaluate({"rows": 3, "cols": 4, "wire_ohm": 2.0, "series": "E24",
                         "tolerance_pct": 5}, seed=2)
    assert e["solvers_agree"], e["ngspice_vs_nodal_max_v"]


def test_analog_ideal_parts_reproduce_ideal_mvm():
    e = analog.evaluate({"rows": 4, "cols": 4, "series": "ideal", "tolerance_pct": 0,
                         "wire_ohm": 0, "opamp_gain": 1e8}, use_ngspice=False)
    assert np.allclose(e["y_nodal_v"], e["y_ideal_v"], atol=1e-6 * e["full_scale_v"])


def test_analog_error_grows_with_explainable_causes():
    base = {"rows": 4, "cols": 4, "wire_ohm": 0.0, "series": "ideal", "tolerance_pct": 0.0}
    errs = [analog.monte_carlo(base | ch, trials=30)["rms_error_pct_fs_mean"]
            for ch in ({}, {"tolerance_pct": 1.0}, {"tolerance_pct": 5.0})]
    assert errs[0] < errs[1] < errs[2]
    wire = [analog.evaluate(base | {"wire_ohm": w}, use_ngspice=False)["rms_error_pct_fs"]
            for w in (0.0, 50.0)]
    assert wire[1] > wire[0]


def test_e_series_snapping():
    assert analog.snap(10_300, "E96") == pytest.approx(10_200)
    assert analog.snap(10_300, "E24") == pytest.approx(10_000)
    assert analog.snap(4_650, "E24") == pytest.approx(4_700)


def test_analog_swing_warning_and_bom():
    e = analog.evaluate({"rows": 2, "cols": 2, "rf_ohm": 1e7, "supply_v": 3.3},
                        use_ngspice=False)
    assert not e["swing_ok"]
    assert sum(i["qty"] for i in e["bom"] if i["part"].startswith("resistor")) == 8


@pytest.mark.skipif(not HAVE_NGSPICE, reason="ngspice not installed")
def test_build_logs_predictions_for_measurement(tmp_path):
    loop = LearningLoop(tmp_path / "l.db")
    b = imc_lab.build("analog_crossbar", {"rows": 2, "cols": 2}, loop=loop)
    assert b["verified"] and set(b["prediction_ids"]) == set(
        imc_lab.FAMILIES["analog_crossbar"]["measurable"])
    pred = loop.prediction(b["prediction_ids"]["static_power_w"])
    assert pred["value"] == pytest.approx(b["predictions"]["static_power_w"]["value"])
    assert pred["features"]["rows"] == 2


def test_build_does_not_log_unverified_designs(tmp_path):
    loop = LearningLoop(tmp_path / "l.db")
    b = imc_lab.build("analog_crossbar", {"rows": 2, "cols": 2, "rf_ohm": 1e7,
                                          "supply_v": 3.3}, loop=loop) if HAVE_NGSPICE else None
    if b is None:
        pytest.skip("ngspice not installed")
    assert not b["verified"] and b["prediction_ids"] == {} and b["warnings"]
