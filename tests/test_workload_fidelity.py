"""Workload compiler and multi-fidelity scheduler."""

from __future__ import annotations

import pytest

from chiplab.pim.workload import WorkloadError, compile_workload, lower
from chiplab.science import fidelity as fid


def test_matmul_lowering_counts():
    w = lower({"kind": "matmul", "m": 64, "n": 32, "k": 16, "dtype": "int8"})
    assert w.macs == 64 * 32 * 16 and w.unique_bytes == 64 * 16 + 16 * 32 + 64 * 32


def test_compiler_supports_and_refuses_honestly():
    r = compile_workload({"kind": "matmul", "m": 512})
    impl = r["implementations"]
    assert not impl["dram_in_array"]["supported"]
    assert impl["analog_crossbar"]["meets_precision"] is False      # 6 effective bits < 8
    assert "analog_crossbar" not in r["ranking"]
    r4 = compile_workload({"kind": "matmul", "m": 512, "dtype": "int4"})
    assert "analog_crossbar" in r4["ranking"]


def test_bitwise_prefers_in_array_on_energy():
    r = compile_workload({"kind": "bitwise", "n_bytes": 2**26})
    assert r["ranking"][0] == "dram_in_array"
    assert r["implementations"]["dram_in_array"]["data_movement_bytes"] == 0


def test_memory_bound_vs_compute_bound():
    big = compile_workload({"kind": "matmul", "m": 4096})["implementations"]["cpu"]
    red = compile_workload({"kind": "reduction", "n": 1e9})["implementations"]["cpu"]
    assert big["bound"] == "compute" and red["bound"] == "memory"


def test_attention_has_hybrid():
    r = compile_workload({"kind": "attention", "seq": 1024})
    assert r["implementations"]["hybrid"]["supported"]


def test_bad_workload():
    with pytest.raises(WorkloadError):
        lower({"kind": "matmul", "m": -1})
    with pytest.raises(WorkloadError):
        lower({"kind": "teleport"})


def _ladder(values_by_level):
    def mk(level):
        return fid.Level(level, lambda s: {"ok": s.get("ok", True),
                                           "value": values_by_level[level](s)}, 1.0)
    return [mk(lv) for lv in values_by_level]


def test_scheduler_prunes_hopeless_and_keeps_contenders():
    cands = [("a", {"v": 1.0}), ("b", {"v": 1.2}), ("c", {"v": 10.0})]
    r = fid.run(cands, _ladder({"analytical": lambda s: s["v"],
                                "synthesis": lambda s: s["v"]}))
    ids = {s["id"] for s in r["survivors"]}
    assert "c" not in ids and "a" in ids
    assert any(t["decision"] == "pruned" and t["candidate"] == "c" for t in r["trace"])


def test_scheduler_eliminates_functional_failures():
    cands = [("good", {"v": 2.0}), ("broken", {"v": 1.0, "ok": False})]
    r = fid.run(cands, _ladder({"analytical": lambda s: s["v"]}))
    assert [s["id"] for s in r["survivors"]] == ["good"]


def test_scheduler_flags_band_violation():
    r = fid.run([("a", {"v": 1.0})], _ladder({"analytical": lambda s: s["v"],
                                               "synthesis": lambda s: 3 * s["v"]}))
    assert r["band_violations"] and not r["trustworthy"]


def test_scheduler_budget_stops():
    r = fid.run([("a", {"v": 1.0})], [fid.Level("analytical", lambda s: {"ok": True,
                                                                         "value": 1.0}, 5000)],
                max_cost_s=10)
    assert r["trace"][0]["decision"] == "stopped"
