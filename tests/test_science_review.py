"""Review gate: the verdict is computed by checks, never taken from the claimant."""

from __future__ import annotations

import math

import numpy as np

from chiplab.science import landauer_j, review, review_ledger_run

PROV = {"tool": "chiplab.analytic"}


def model(p):
    return {"energy_per_op_j": p["c_f"] * p["v"] ** 2, "latency_s": p["n"] * 1e-9}


INPUTS = {"c_f": 1e-15, "v": 1.0, "n": 4}


def test_good_claim_with_model_is_plausible_not_validated():
    r = review("toy", model(INPUTS), fidelity="analytical", provenance=PROV,
               evaluate=model, inputs=INPUTS, claimed_status="success")
    assert r.verdict == "plausible"
    assert any("ignored" in n for n in r.notes)


def test_independent_benchmark_validates():
    r = review("toy", model(INPUTS), fidelity="analytical", provenance=PROV, evaluate=model,
               inputs=INPUTS, benchmarks={"energy_per_op_j": {"value": 1.05e-15,
                                                              "rel_tol": 0.1,
                                                              "source": "ngspice run 12"}})
    assert r.verdict == "validated"


def test_self_benchmark_is_refused():
    r = review("toy", model(INPUTS), fidelity="analytical", provenance=PROV, evaluate=model,
               inputs=INPUTS, benchmarks={"energy_per_op_j": {"value": 1e-15,
                                                              "source": "chiplab.analytic"}})
    assert r.verdict == "rejected"


def test_hard_coded_numbers_fail_sensitivity():
    def fake(p):
        return {"energy_per_op_j": 1e-15, "latency_s": 4e-9}
    r = review("fake", fake(INPUTS), fidelity="analytical", provenance=PROV, evaluate=fake,
               inputs=INPUTS)
    assert r.verdict == "rejected"
    assert any(c.name == "input_sensitivity" and c.status == "fail" for c in r.checks)


def test_unseeded_randomness_fails_reproducibility():
    rng = np.random.default_rng()

    def noisy(p):
        return {"energy_per_op_j": p["c_f"] * (1 + rng.normal(0, 0.1)), "latency_s": 1e-9}
    r = review("noisy", noisy(INPUTS), fidelity="analytical", provenance=PROV, evaluate=noisy,
               inputs=INPUTS)
    assert r.verdict == "rejected"


def test_claim_that_does_not_match_rerun_fails():
    claim = model(INPUTS) | {"energy_per_op_j": 1e-17}
    r = review("toy", claim, fidelity="analytical", provenance=PROV, evaluate=model,
               inputs=INPUTS)
    assert r.verdict == "rejected"


def test_physics_bounds_and_landauer():
    assert math.isclose(landauer_j(300), 2.87e-21, rel_tol=1e-2)
    for bad in ({"energy_per_bit_j": 1e-22}, {"power_w": -1.0}, {"yield": 1.2},
                {"latency_s": float("nan")}):
        r = review("b", bad, fidelity="analytical", provenance=PROV)
        assert r.verdict == "rejected", bad


def test_unknown_fidelity_or_missing_provenance_rejected():
    assert review("x", {"power_w": 1.0}, fidelity="unknown", provenance=PROV).verdict == \
        "rejected"
    assert review("x", {"power_w": 1.0}, fidelity="analytical").verdict == "rejected"


def test_numbers_alone_are_only_unverified():
    assert review("x", {"power_w": 1.0}, fidelity="analytical",
                  provenance=PROV).verdict == "unverified"


def test_legacy_ledger_row_is_rejected():
    run = {"run_id": "r1", "metrics": {"energy_per_bit_j": 5e-15}, "fidelity": "unknown",
           "provenance": {}, "duplicate_of": "r0"}
    r = review_ledger_run(run)
    assert r.verdict == "rejected" and any("duplicates" in n for n in r.notes)
