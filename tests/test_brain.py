"""Project Brain: storage, idempotent ingest, grouped recall, decisions."""

from __future__ import annotations

import pytest

from chiplab.brain import Brain
from chiplab.brain.ingest import seed_science


@pytest.fixture()
def brain(tmp_path):
    return Brain(tmp_path / "b.db")


def test_recall_groups_by_kind_and_surfaces_limitations(brain):
    seed_science(brain)
    brain.remember("experiment", "DRAM TRA Monte Carlo", "charge sharing error rate 1e-4",
                   tags=("dram",))
    r = brain.recall("Can we combine DRAM with local computation?")
    assert {"science", "experiment", "project"} <= set(r["results"])
    assert any("Ambit" in m["title"] for m in r["results"]["science"])
    assert any("fabricate" in m["title"] for m in r["limitations"])


def test_ingest_is_idempotent(brain):
    seed_science(brain)
    n = brain.stats()["total"]
    seed_science(brain)
    assert brain.stats()["total"] == n


def test_rejected_ideas_are_kept_and_recalled(brain):
    brain.decide("Ferrofluid main memory for v1", "no validated model", rejected=True,
                 tags=("ferrofluid",))
    r = brain.recall("ferrofluid memory")
    assert r["rejected"] and r["rejected"][0]["status"] == "rejected"


def test_superseded_hidden_by_default(brain):
    mid = brain.remember("experiment", "old sram run", "sram add", tags=("sram",))
    brain.set_status(mid, "superseded")
    assert not brain.recall("sram")["results"]
    assert brain.recall("sram", include_superseded=True)["results"]


def test_validation(brain):
    with pytest.raises(ValueError):
        brain.remember("gossip", "x")
    with pytest.raises(ValueError):
        brain.remember("project", " ")
    assert brain.recall("the and of")["results"] == {}
