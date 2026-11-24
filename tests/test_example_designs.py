"""Every shipped design YAML parses, validates and evaluates end to end."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from chiplab import mcp_server as srv
from chiplab.experiments.store import ExperimentStore

ROOT = Path(__file__).resolve().parents[1]
YAMLS = sorted(p for p in [*ROOT.glob("examples/*.yaml"), *ROOT.glob("projects/**/*.yaml")]
               if p.read_text().lstrip().startswith("design:") or "\ndesign:" in p.read_text())
# (projects/ also holds exported Tiny Tapeout projects whose info.yaml is not a ChipLab design)


@pytest.mark.parametrize("path", YAMLS, ids=lambda p: p.name)
def test_design_evaluates(path, tmp_path, monkeypatch):
    if "ecore" in path.name:
        from chiplab.backends import EcoreBackend
        if not EcoreBackend().is_available():
            pytest.skip("Echo Lab eCore not available")
    monkeypatch.setattr(srv, "store", ExperimentStore(db_path=str(tmp_path / "e.db")))
    text = path.read_text()
    data = yaml.safe_load(text)["design"]
    wl = (data.get("metadata") or {}).get("workload_hint") or data.get("workload")
    r = srv.chip_evaluate(spec=text, workload=wl)
    assert r["valid"], r["validation_errors"]
    assert r["performance"]["peak_compute_tops"] > 0
    assert r["cost"]["unit_cost_usd"] > 0
    assert math.isfinite(r["power_thermal"]["total_power_watts"])


@pytest.mark.parametrize("name", ["riscv_soc", "hbm_ai_accelerator", "stacked_3d_accelerator"])
def test_new_examples_meet_their_constraints(name, tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "store", ExperimentStore(db_path=str(tmp_path / "e.db")))
    text = (ROOT / "examples" / f"{name}.yaml").read_text()
    wl = yaml.safe_load(text)["design"]["metadata"]["workload_hint"]
    r = srv.chip_evaluate(spec=text, workload=wl)
    assert all(c["pass"] for c in r["constraint_checks"]), r["constraint_checks"]
