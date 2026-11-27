"""Tests for the Custom Memory Type Explorer backend API."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from chiplab.app.custom_memory_routes import MEMORY_PRESETS, evaluate_custom_memory
from chiplab.app.server import create_app
from chiplab.experiments.store import ExperimentStore
from chiplab.schemas.design import (
    ChipletSpec,
    ComputeSpec,
    DesignSpec,
    MemorySpec,
    PhysicalConstraints,
)

EXPECTED_PRESET_NAMES = {
    "HBM3e-36GB",
    "HBM4-2048b",
    "3D-ReRAM-Crossbar",
    "FeFET-CIM",
    "STT-MRAM-LLC",
    "3D-V-SRAM",
    "Optical-DRAM",
}

REQUIRED_PRESET_FIELDS = {
    "name",
    "tech_class",
    "node_nm",
    "stacks",
    "dies_per_stack",
    "capacity_gb_per_stack",
    "channels_per_stack",
    "io_gbps_per_pin",
    "bus_width_bits",
    "energy_pj_per_bit",
    "read_latency_ns",
    "write_latency_ns",
    "density_gb_per_mm2",
    "retention_s",
    "endurance_cycles",
    "cim_capable",
    "cim_tops_per_w",
}


def _make_sample_design() -> DesignSpec:
    return DesignSpec(
        name="Test-AI-Accelerator",
        version="1.0",
        chiplets=[
            ChipletSpec(
                id="compute0",
                type="compute",
                count=2,
                area_mm2=250.0,
                tdp_watts=180.0,
                process_nm=5,
                compute=ComputeSpec(architecture="systolic", peak_tops=1200.0),
            ),
            ChipletSpec(
                id="mem0",
                type="memory",
                count=8,
                area_mm2=110.0,
                tdp_watts=15.0,
                memory=MemorySpec(
                    technology="HBM3e",
                    capacity_per_chiplet_gb=24.0,
                    channels=16,
                    stack_layers=8,
                ),
            ),
        ],
        constraints=PhysicalConstraints(
            ambient_temp_c=25.0,
            max_temp_c=85.0,
            thermal_resistance_c_per_w=0.07,
            max_power_watts=700.0,
        ),
    )


def test_presets_endpoint_returns_all_templates(tmp_path: Path) -> None:
    app = create_app(str(tmp_path / "exp.db"))
    client = TestClient(app)

    resp = client.get("/api/lab/custom_memory/presets")
    assert resp.status_code == 200
    body = resp.json()
    presets = body["presets"]
    assert len(presets) == len(MEMORY_PRESETS) == 7
    assert {p["name"] for p in presets} == EXPECTED_PRESET_NAMES
    for preset in presets:
        assert REQUIRED_PRESET_FIELDS.issubset(preset.keys())


def test_evaluate_reram_crossbar_formulas_and_cim(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "exp.db")
    res = evaluate_custom_memory(store, {"preset": "3D-ReRAM-Crossbar"})

    assert res["design_context"]["source"] == "baseline_gpu"
    mem = res["custom_memory"]
    m = res["metrics"]
    rf = res["roofline"]

    assert mem["name"] == "3D-ReRAM-Crossbar"
    assert mem["tech_class"] == "reram"
    assert mem["cim_capable"] is True

    expected_cap = 8 * 64.0
    expected_bw = 8 * (1024 * 6.4 / 8.0) / 1000.0
    expected_footprint = max(12.0, 64.0 / max(0.12 * 8, 0.05))
    expected_power = (expected_bw * 1e12 * 8.0) * (1.8 * 1e-12) * 0.65
    expected_cim_tops = expected_power * 45.0

    assert m["total_capacity_gb"] == pytest.approx(expected_cap, rel=1e-4)
    assert m["total_bandwidth_tb_s"] == pytest.approx(expected_bw, rel=1e-4)
    assert m["stack_footprint_mm2"] == pytest.approx(expected_footprint, rel=1e-4)
    assert m["memory_power_w"] == pytest.approx(expected_power, rel=1e-3)
    assert m["cim_bonus_tops"] == pytest.approx(expected_cim_tops, rel=1e-3)
    assert m["effective_compute_tops"] > 2000.0
    assert m["bottleneck"] in {"memory-bandwidth", "compute-bound", "thermal-limited", "balanced"}
    assert "device_scaling" in res
    assert rf["compute_ceiling_tops"] == pytest.approx(m["effective_compute_tops"], rel=1e-4)
    assert rf["memory_bw_tb_s"] == pytest.approx(m["total_bandwidth_tb_s"], rel=1e-4)
    assert rf["ridge_point"] == pytest.approx(m["ridge_point_flops_per_byte"], rel=1e-4)
    assert "3D-ReRAM-Crossbar" in res["comparison_summary"]


def test_evaluate_fefet_cim_with_node_scaling_and_store_design(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "exp.db")
    design_id = store.save_design(_make_sample_design())

    res_default = evaluate_custom_memory(
        store, {"design_id": design_id, "preset": "FeFET-CIM"}
    )
    res_scaled = evaluate_custom_memory(
        store, {"preset": "FeFET-CIM", "memory": {"node_nm": 3.0}}
    )

    assert res_default["design_context"]["source"] == "store"
    assert res_default["design_context"]["design_id"] == design_id
    # When design_id is omitted, should pick first design in store
    assert res_scaled["design_context"]["design_id"] == design_id
    # Scaling FeFET from 7nm -> 3nm should reduce energy/bit and increase density
    assert (
        res_scaled["custom_memory"]["energy_pj_per_bit"]
        < res_default["custom_memory"]["energy_pj_per_bit"]
    )
    assert (
        res_scaled["custom_memory"]["density_gb_per_mm2"]
        > res_default["custom_memory"]["density_gb_per_mm2"]
    )


def test_evaluate_custom_user_tech_and_bottlenecks(tmp_path: Path) -> None:
    app = create_app(str(tmp_path / "exp.db"))
    client = TestClient(app)

    custom_payload = {
        "memory": {
            "name": "Cryo-Hybrid-X",
            "tech_class": "custom",
            "node_nm": 4.0,
            "stacks": 12,
            "dies_per_stack": 16,
            "capacity_gb_per_stack": 48.0,
            "channels_per_stack": 32,
            "io_gbps_per_pin": 16.0,
            "bus_width_bits": 2048,
            "energy_pj_per_bit": 4.2,
            "read_latency_ns": 12.0,
            "write_latency_ns": 15.0,
            "density_gb_per_mm2": 0.08,
            "retention_s": 1e5,
            "endurance_cycles": 1e12,
            "cim_capable": False,
            "cim_tops_per_w": 0.0,
        },
        "operating_ai": 64.0,
    }
    resp = client.post("/api/lab/custom_memory", json=custom_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["custom_memory"]["name"] == "Cryo-Hybrid-X"
    assert body["custom_memory"]["tech_class"] == "custom"
    assert body["metrics"]["total_capacity_gb"] == pytest.approx(576.0)
    # Huge bandwidth at 4.2 pJ/bit drives high power -> thermal-limited
    assert body["metrics"]["bottleneck"] == "thermal-limited"
    assert body["metrics"]["peak_temp_c"] > 85.0

    # Low power, high operating AI -> compute-bound
    resp_cb = client.post(
        "/api/lab/custom_memory",
        json={"preset": "Optical-DRAM", "operating_ai": 500.0},
    )
    assert resp_cb.status_code == 200
    assert resp_cb.json()["metrics"]["bottleneck"] == "compute-bound"


def test_compare_endpoint_multiple_memories(tmp_path: Path) -> None:
    app = create_app(str(tmp_path / "exp.db"))
    client = TestClient(app)

    resp = client.post(
        "/api/lab/custom_memory/compare",
        json={
            "operating_ai": 48.0,
            "memories": [
                "HBM3e-36GB",
                "3D-ReRAM-Crossbar",
                {"preset": "FeFET-CIM", "stacks": 8},
                {
                    "name": "User-PCM-Stack",
                    "tech_class": "pcm",
                    "node_nm": 14.0,
                    "stacks": 6,
                    "dies_per_stack": 8,
                    "capacity_gb_per_stack": 64.0,
                    "io_gbps_per_pin": 6.4,
                    "bus_width_bits": 1024,
                    "cim_capable": True,
                    "cim_tops_per_w": 30.0,
                },
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    comparisons = data["comparisons"]
    assert len(comparisons) == 4
    names = [c["custom_memory"]["name"] for c in comparisons]
    assert names == ["HBM3e-36GB", "3D-ReRAM-Crossbar", "FeFET-CIM", "User-PCM-Stack"]
    for item in comparisons:
        assert "metrics" in item
        assert "roofline" in item
        assert "comparison_summary" in item
