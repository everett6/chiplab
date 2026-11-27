"""Learning loop: validation gate, correction fitting, acceptance rule and recommendations.

The "measured" numbers here are synthetic, generated from a known truth, so the tests can check
the loop recovers that truth. They exercise the algorithm only; no claim about any device.
"""

from __future__ import annotations

import numpy as np
import pytest

from chiplab.learning import LearningLoop

M, K = "imc.test.power", "power"
GOOD = dict(source_kind="bench", method="SMU 2450", measured_on="2026-11-26",
            submitted_by="tester", attested_real=True)


@pytest.fixture()
def loop(tmp_path):
    return LearningLoop(tmp_path / "learn.db")


def _pred(loop, value, **feats):
    return loop.record_prediction(M, K, value, "W", feats or {"rows": 1.0})


def _feed(loop, truth, n, rel_unc=0.02, seed=0, confirm=False):
    rng = np.random.default_rng(seed)
    for i in range(n):
        rows = float(1 + i % 6)
        pv = 1e-5 * rows
        pid = _pred(loop, pv, rows=rows)
        meas = truth(pv, rows) * (1 + rng.normal(0, rel_unc))
        st = loop.submit_measurement(pid, meas, "W", uncertainty=meas * rel_unc,
                                     confirm_outlier=confirm, **GOOD)
        assert st["status"] == "validated", st


@pytest.mark.parametrize("change,reason", [
    ({"source_kind": "simulation"}, "model output"),
    ({"attested_real": False}, "attest"),
    ({"method": ""}, "method"),
    ({"measured_on": ""}, "date"),
    ({"source_kind": "published"}, "reference"),
])
def test_validation_gate_rejects(loop, change, reason):
    pid = _pred(loop, 1e-5)
    st = loop.submit_measurement(pid, 1.1e-5, "W", uncertainty=1e-7, **(GOOD | change))
    assert st["status"] == "rejected" and any(reason in r for r in st["reasons"])


def test_unit_and_uncertainty_required(loop):
    pid = _pred(loop, 1e-5)
    assert loop.submit_measurement(pid, 1e-2, "mW", uncertainty=1e-4, **GOOD)["status"] == \
        "rejected"
    assert loop.submit_measurement(pid, 1e-5, "W", **GOOD)["status"] == "rejected"
    st = loop.submit_measurement(pid, 1.0e-5, "W", repeats=[0.99e-5, 1.0e-5, 1.01e-5], **GOOD)
    assert st["status"] == "validated"


def test_rejected_points_are_never_learned_from(loop):
    for _ in range(5):
        pid = _pred(loop, 1e-5)
        loop.submit_measurement(pid, 5e-5, "W", uncertainty=1e-7,
                                **(GOOD | {"source_kind": "simulation"}))
    assert loop.refit(M, K)["status"] == "no_data"


def test_recovers_multiplicative_bias_and_accepts(loop):
    _feed(loop, lambda p, r: 1.3 * p, 8)
    fit = loop.refit(M, K)
    assert fit["status"] == "accepted"
    assert fit["bias_factor"] == pytest.approx(1.3, rel=0.03)
    c = loop.corrected(M, K, 2e-5, {"rows": 2.0})
    assert c["corrected"] and c["interval_90"][0] < 2.6e-5 < c["interval_90"][1]


def test_too_few_points_stay_candidate(loop):
    _feed(loop, lambda p, r: 1.3 * p, 2)
    fit = loop.refit(M, K)
    assert fit["status"] == "candidate"
    assert not loop.corrected(M, K, 1e-5)["corrected"]


def test_unbiased_model_is_not_changed(loop):
    _feed(loop, lambda p, r: p, 8, seed=4)
    fit = loop.refit(M, K)
    if fit["status"] == "accepted":           # allowed only if the shift is negligible
        assert fit["bias_factor"] == pytest.approx(1.0, abs=0.03)


def test_learns_feature_dependent_error(loop):
    # model is right for 1 row and increasingly low for bigger arrays (e.g. unmodelled wiring)
    _feed(loop, lambda p, r: p * (1 + 0.1 * (r - 1)), 18)
    assert loop.refit(M, K)["status"] == "accepted"
    small = loop.corrected(M, K, 1e-5, {"rows": 1.0})["value"]
    big = loop.corrected(M, K, 6e-5, {"rows": 6.0})["value"]
    assert small == pytest.approx(1e-5, rel=0.08)
    assert big == pytest.approx(6e-5 * 1.5, rel=0.08)


def test_outlier_needs_review_then_confirm(loop):
    _feed(loop, lambda p, r: 1.3 * p, 8)
    loop.refit(M, K)
    pid = _pred(loop, 1e-5, rows=1.0)
    st = loop.submit_measurement(pid, 1e-4, "W", uncertainty=1e-7, **GOOD)
    assert st["status"] == "needs_review"
    st = loop.submit_measurement(pid, 1e-4, "W", uncertainty=1e-7, confirm_outlier=True, **GOOD)
    assert st["status"] == "validated"


def test_versions_are_kept_and_rollback_reissues(loop):
    _feed(loop, lambda p, r: 1.3 * p, 6)
    v1 = loop.refit(M, K)["version"]
    _feed(loop, lambda p, r: 2.0 * p, 30, seed=9, confirm=True)   # a shift the gate holds
    v2 = loop.refit(M, K)
    assert v2["version"] == v1 + 1
    rb = loop.rollback(M, K, v1)
    assert rb["version"] == v2["version"] + 1 and rb["restored"] == v1
    assert loop.corrected(M, K, 1e-5, {"rows": 1.0})["value"] == pytest.approx(1.3e-5, rel=0.05)


def test_recommend_uses_corrections_and_picks_next_experiment(loop):
    cands = [{"id": "a", "family": "f", "model_prefix": "imc.test", "features": {"rows": 1.0},
              "metrics": {"power": 1.0e-5}},
             {"id": "b", "family": "f", "model_prefix": "imc.test", "features": {"rows": 6.0},
              "metrics": {"power": 0.9e-5}}]
    before = loop.recommend(cands, "power")
    assert before["best"] == "b" and before["next_experiment"]["id"] in {"a", "b"}
    _feed(loop, lambda p, r: p * (1 + 0.1 * (r - 1)), 18)   # big arrays draw 50 % more
    loop.refit(M, K)
    after = loop.recommend(cands, "power")
    assert after["best"] == "a"                             # 1.0e-5 beats 0.9e-5 * 1.5
    capped = loop.recommend(cands, "power", constraints={"power": (None, 1.2e-5)})
    assert [r["feasible"] for r in capped["ranking"]] == [True, False]


def test_build_success_posterior(loop):
    for ok in (True, True, False):
        loop.record_build("f", "d", ok, "tester")
    assert loop.build_success("f")["p_success_mean"] == pytest.approx(3 / 5)
