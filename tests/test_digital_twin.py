"""Zero-token digital twin: Pareto search, Monte Carlo robustness, sensitivity, fallback."""

from __future__ import annotations

import math
import random

import pytest

from chiplab.engine import digital_twin as dt
from chiplab.schemas.common import Provenance
from chiplab.schemas.digital_twin import (
    BlackBoxMacro,
    DigitalTwinConfig,
    Objective,
    PinTiming,
    SearchParameter,
    UncertaintyPrior,
)


@pytest.fixture(autouse=True)
def _prices(monkeypatch, tmp_path):
    monkeypatch.setenv("CHIPLAB_PRICE_FILE", str(tmp_path / "prices.json"))


def _hypervolume_2d(points: list[tuple[float, float]], ref: tuple[float, float]) -> float:
    """Hypervolume for 2 minimization objectives."""
    pts = sorted(p for p in points if p[0] < ref[0] and p[1] < ref[1])
    hv, best_y = 0.0, ref[1]
    for x, y in pts:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return hv


TWO_OBJ = [Objective(name="perf", goal="max"), Objective(name="power_w", goal="min")]


# --- building blocks -------------------------------------------------------------------------

def test_nondominated_sort_and_constrained_domination():
    items = [([1.0, 5.0], 0.0), ([2.0, 2.0], 0.0), ([3.0, 3.0], 0.0), ([0.0, 0.0], 1.0),
             ([5.0, 1.0], 0.0)]
    fronts = dt.nondominated_sort(items)
    assert sorted(fronts[0]) == [0, 1, 4]
    assert fronts[1] == [2]
    assert fronts[-1] == [3]  # infeasible ranks last even with the "best" objectives


def test_latin_hypercube_is_stratified():
    n, d = 50, 4
    pts = dt.latin_hypercube(n, d, random.Random(1))
    for k in range(d):
        bins = sorted(int(p[k] * n) for p in pts)
        assert bins == list(range(n))


def test_model_physics_monotonic():
    base = dict(cores=8, node="N7", freq_ghz=1.5, vdd=0.75)
    lo_v = dt.evaluate_point({**base, "vdd": 0.6})
    hi_v = dt.evaluate_point({**base, "vdd": 0.9})
    assert hi_v["fmax_ghz"] > lo_v["fmax_ghz"] and hi_v["power_w"] > lo_v["power_w"]
    n16 = dt.evaluate_point({**base, "node": "N16"})
    n3 = dt.evaluate_point({**base, "node": "N3"})
    assert n3["area_mm2"] < n16["area_mm2"] and n3["fmax_ghz"] > n16["fmax_ghz"]
    more = dt.evaluate_point({**base, "cores": 32})
    assert more["perf"] > dt.evaluate_point(base)["perf"]
    assert more["area_mm2"] > dt.evaluate_point(base)["area_mm2"]
    with pytest.raises(ValueError):
        dt.evaluate_point({**base, "node": "N9"})


def test_pure_python_backend_matches_numpy():
    cfg = DigitalTwinConfig(seed=3)
    p_np, p_py = dt._Problem(cfg), dt._Problem(cfg, force_pure_python=True)
    designs = [p_np.decode(x) for x in dt.latin_hypercube(20, len(p_np.space), random.Random(3))]
    for a, b in zip(p_np.metrics(designs), p_py.metrics(designs)):
        for k in a:
            assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-9)


# --- multi-objective search --------------------------------------------------------------

def test_nsga2_converges_hypervolume_improves():
    cfg = DigitalTwinConfig(objectives=TWO_OBJ, population=40, generations=25, mc_trials=0,
                            seed=11)
    _, trace = dt.run_digital_twin(cfg)
    rows = trace["rows"]
    feas = [r for r in rows if r["violation"] == 0]
    objs = lambda rs: [(-r["metrics"]["perf"], r["metrics"]["power_w"]) for r in rs]  # noqa: E731
    ref = (0.0, max(r["metrics"]["power_w"] for r in feas) * 1.1)
    hv0 = _hypervolume_2d(objs([r for r in feas if r["gen"] == 0]), ref)
    hv_all = _hypervolume_2d(objs(feas), ref)
    assert hv_all > hv0 * 1.05
    # The returned front is mutually non-dominated.
    front = [rows[i] for i in trace["front"] if rows[i]["violation"] == 0]
    pts = objs(front)
    for i, a in enumerate(pts):
        for j, b in enumerate(pts):
            if i != j:
                assert not (b[0] <= a[0] and b[1] <= a[1] and b != a)


def test_search_is_deterministic_per_seed():
    cfg = DigitalTwinConfig(population=16, generations=4, mc_trials=32, seed=5)
    a, _ = dt.run_digital_twin(cfg)
    b, _ = dt.run_digital_twin(cfg)
    assert [p.params for p in a.top] == [p.params for p in b.top]
    assert [p.robustness for p in a.top] == [p.robustness for p in b.top]
    c, _ = dt.run_digital_twin(cfg.model_copy(update={"seed": 6}))
    assert [p.params for p in a.top] != [p.params for p in c.top]


@pytest.mark.parametrize("strategy", ["lhs", "bayes"])
def test_other_strategies_find_feasible_designs(strategy):
    cfg = DigitalTwinConfig(strategy=strategy, lhs_samples=200, bayes_iterations=15,
                            mc_trials=32, top_k=3, seed=2)
    s, _ = dt.run_digital_twin(cfg)
    assert s.feasible > 0 and s.top and all(p.feasible for p in s.top)
    assert s.evaluated == (200 if strategy == "lhs" else max(16, 4 * 11) + 15)


def test_fixed_parameters_and_bad_space():
    s, _ = dt.run_digital_twin(DigitalTwinConfig(population=16, generations=3, mc_trials=0,
                                                 fixed={"node": "N5", "cores": 4}))
    assert all(p.params["node"] == "N5" and p.params["cores"] == 4 for p in s.top)
    with pytest.raises(ValueError, match="bogus"):
        dt.run_digital_twin(DigitalTwinConfig(
            search_space=[SearchParameter(name="bogus", low=0, high=1)], mc_trials=0))
    with pytest.raises(ValueError):
        SearchParameter(name="x", kind="float", low=2, high=1)


def test_dlen_never_exceeds_vlen():
    cfg = DigitalTwinConfig(lhs_samples=300, strategy="lhs", mc_trials=0)
    _, trace = dt.run_digital_twin(cfg)
    assert all(r["params"]["dlen"] <= r["params"]["vlen"] for r in trace["rows"])


def test_constraints_respected_in_top():
    cfg = DigitalTwinConfig(population=32, generations=10, mc_trials=0, seed=1,
                            constraints={"max_power_w": 3.0, "min_slack_ps": 50.0})
    s, _ = dt.run_digital_twin(cfg)
    for p in s.top:
        assert p.metrics["power_w"] <= 3.0 + 1e-9 and p.metrics["slack_ps"] >= 50.0


# --- Monte Carlo robustness ---------------------------------------------------------------

def _critical_design() -> tuple[dt._Problem, dict, dict]:
    prob = dt._Problem(DigitalTwinConfig(mc_trials=400, seed=9))
    d = {**dt.DEFAULTS, "node": "N3", "vdd": 0.75}
    m0 = prob.metrics([d])[0]
    d["freq_ghz"] = m0["fmax_ghz"] * 0.99  # ~1% timing margin
    return prob, d, prob.metrics([d])[0]


def test_monte_carlo_yield_under_assumed_wire_parasitics():
    prob, d, nominal = _critical_design()
    priors = [UncertaintyPrior(name="wire_rc", provenance=Provenance.ASSUMED, spread_pct=25),
              UncertaintyPrior(name="gate_delay", provenance=Provenance.ESTIMATED, spread_pct=15)]
    f = dt._sample_factors(priors, 400, 9, prob.numpy)
    rep, _ = dt.monte_carlo(prob, d, nominal, f, 400)
    assert 0.2 < rep.slack_yield < 0.95  # unknowns make a 1%-margin design risky
    assert rep.worst_slack_ps_p5 < 0 < nominal["slack_ps"]
    assert rep.robustness_score < 100 * rep.slack_yield + 1e-9


def test_published_or_measured_parameters_are_not_perturbed():
    priors = [UncertaintyPrior(name="wire_rc", provenance=Provenance.PUBLISHED),
              UncertaintyPrior(name="gate_delay", provenance=Provenance.MEASURED),
              UncertaintyPrior(name="leakage", provenance=Provenance.ASSUMED, spread_pct=20)]
    f = dt._sample_factors(priors, 50, 1, True)
    assert set(f) == {"leakage"}
    assert dt._sample_factors(priors, 50, 1, False).keys() == {"leakage"}


def test_wider_uncertainty_lowers_robustness():
    prob, d, nominal = _critical_design()
    reps = []
    for spread in (5, 25):
        pr = [UncertaintyPrior(name="wire_rc", spread_pct=spread),
              UncertaintyPrior(name="gate_delay", spread_pct=spread)]
        reps.append(dt.monte_carlo(prob, d, nominal, dt._sample_factors(pr, 400, 3, True), 400)[0])
    assert reps[1].slack_yield < reps[0].slack_yield
    assert reps[1].robustness_score < reps[0].robustness_score


def test_thermal_violation_probability():
    prob = dt._Problem(DigitalTwinConfig(constraints={"max_temp_c": 40.0}, theta_ja_c_per_w=1.0))
    d = {**dt.DEFAULTS, "cores": 32, "freq_ghz": 1.0, "vdd": 0.8}
    nominal = prob.metrics([d])[0]
    assert 25 < nominal["temp_c"] < 60
    pr = [UncertaintyPrior(name="theta_ja", spread_pct=25),
          UncertaintyPrior(name="cdyn", spread_pct=20)]
    rep, _ = dt.monte_carlo(prob, d, nominal, dt._sample_factors(pr, 300, 4, True), 300)
    assert 0.0 <= rep.thermal_violation_prob <= 1.0
    expected = nominal["temp_c"] > 40.0
    assert (rep.thermal_violation_prob > 0.5) == expected


def test_sensitivity_ranks_the_parameter_that_matters():
    cfg = DigitalTwinConfig(population=16, generations=3, mc_trials=200, seed=4, priors=[
        UncertaintyPrior(name="gate_delay", spread_pct=20),
        UncertaintyPrior(name="wafer_price", spread_pct=20),
        UncertaintyPrior(name="theta_ja", spread_pct=1)])
    s, _ = dt.run_digital_twin(cfg)
    by = {e.parameter: e for e in s.sensitivity}
    assert by["gate_delay"].impact["slack_ps"] > 0.9
    assert by["wafer_price"].impact["slack_ps"] < 0.3
    assert s.sensitivity[0].parameter == "gate_delay"


def test_calibration_marks_parameters_measured():
    cfg = DigitalTwinConfig(population=16, generations=2, mc_trials=64,
                            calibration={"gate_delay": 1.1, "wire_rc": 1.3})
    s, _ = dt.run_digital_twin(cfg)
    assert s.calibrated_parameters == ["gate_delay", "wire_rc"]
    assert "gate_delay" not in s.unknown_parameters and "wire_rc" not in s.unknown_parameters
    assert all(e.parameter not in ("gate_delay", "wire_rc") for e in s.sensitivity)
    slow = dt.evaluate_point({"node": "N5"}, calibration={"gate_delay": 1.5})
    assert slow["fmax_ghz"] < dt.evaluate_point({"node": "N5"})["fmax_ghz"]


def test_blackbox_macro_area_power_and_timing():
    bb = BlackBoxMacro(name="phy", width_um=1000, height_um=500, keepout_um=50,
                       max_freq_ghz=1.0, count=2,
                       pins=[PinTiming(name="d", direction="input", setup_ps=100)],
                       rails=[{"name": "VDDA", "voltage_v": 0.8, "typical_mw": 200}])
    assert bb.area_mm2 == pytest.approx(0.5)
    assert bb.area_with_keepout_mm2 == pytest.approx(1.1 * 0.6)
    base = dt.evaluate_point({"freq_ghz": 1.5})
    with_bb = dt.evaluate_point({"freq_ghz": 1.5}, blackboxes=[bb])
    assert with_bb["area_mm2"] == pytest.approx(base["area_mm2"] + 2 * 0.66)
    assert with_bb["power_w"] > base["power_w"] + 0.39
    assert with_bb["slack_ps"] < 0 < base["slack_ps"]  # PHY fmax 1 GHz < 1.5 GHz clock


def test_compact_summary_and_trace():
    s, trace = dt.run_digital_twin(DigitalTwinConfig(population=16, generations=3, mc_trials=32,
                                                     top_k=2))
    assert len(s.top) == 2 and s.top[0].rank == 1
    assert s.fidelity_disclaimer and s.evaluated == len(trace["rows"])
    assert s.mc_evaluations > 0
    assert trace["monte_carlo"] and trace["config"]["seed"] == 42


def test_benchmark_throughput():
    np_bench = dt.benchmark(5000)
    py_bench = dt.benchmark(500, force_pure_python=True)
    assert np_bench["backend"] == "numpy" and np_bench["designs_per_sec"] > 20_000
    assert py_bench["designs_per_sec"] > 1_000
    assert math.isfinite(np_bench["seconds"])


def test_pure_python_end_to_end():
    s, _ = dt.run_digital_twin(DigitalTwinConfig(population=12, generations=3, mc_trials=24),
                               force_pure_python=True)
    assert s.top and any("Pure-Python" in n for n in s.notes)
