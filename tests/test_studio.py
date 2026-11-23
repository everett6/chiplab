"""GPU Studio: scene builder, package thermal/airflow model and the studio edit API.
Synthetic designs only (plus the repo's example YAMLs)."""

from __future__ import annotations

import pytest

from chiplab.schemas.design import (
    ChipletSpec, ComputeSpec, DesignSpec, InterconnectSpec, MemorySpec, PackageSpec,
    PhysicalConstraints)
from chiplab.studio import thermal as th
from chiplab.studio.scene import apply_edit, build_scene

ROOT = "/home/everett/Chip Lab"


def _load(path: str) -> DesignSpec:
    with open(path) as f:
        return DesignSpec.from_yaml(f.read())


def _gpu(hbm: int = 4, layers: int = 8, r: float | None = None) -> DesignSpec:
    return DesignSpec(
        name="studio_gpu", package=PackageSpec(type="2.5D", interposer="silicon"),
        interconnect=InterconnectSpec(topology="mesh", link_bandwidth_gbps=500, latency_ns=10,
                                      energy_pj_per_bit=0.5),
        constraints=PhysicalConstraints(max_temp_c=95, thermal_resistance_c_per_w=r),
        chiplets=[
            ChipletSpec(id="gpu", type="compute", area_mm2=600, tdp_watts=400,
                        compute=ComputeSpec(architecture="tensor_engine", peak_tops=1000,
                                            mac_units=4096, sram_mb=50)),
            ChipletSpec(id="hbm", type="HBM3", count=hbm, area_mm2=110, tdp_watts=15,
                        memory=MemorySpec(technology="HBM3", capacity_per_chiplet_gb=24,
                                          channels=16, bandwidth_per_channel_gbps=64,
                                          stack_layers=layers, latency_ns=40)),
        ])


# --- thermal model ---------------------------------------------------------------------------

def test_thermal_energy_balance_and_symmetry():
    comps = [{"id": "a", "x": -12, "y": 0, "w": 10, "d": 10, "power_w": 60},
             {"id": "b", "x": 12, "y": 0, "w": 10, "d": 10, "power_w": 60}]
    r = th.solve((-25, -12, 50, 24), comps, 0.0, th.Cooling(), 25.0, n=48)
    assert r["heat_out_w"] == pytest.approx(120.0, rel=1e-4)        # heat out = power in
    a, b = r["components"]
    assert a["max_c"] == pytest.approx(b["max_c"], abs=0.05)         # mirror-symmetric
    assert a["max_c"] > 25.0
    # spreader mean rise ~ P * R_heatsink (most heat leaves through the sink)
    ihs = sum(map(sum, r["grid"]["ihs_c"])) / (r["grid"]["nx"] * r["grid"]["ny"]) - 25.0
    r_par = 1 / (1 / r["r_heatsink_c_per_w"] + 1 / th.Cooling().board_r_c_per_w)
    assert ihs == pytest.approx(120.0 * r_par, rel=0.05)


def test_thermal_transient_rises_monotonically_to_steady_state():
    comps = [{"id": "a", "x": 0, "y": 0, "w": 20, "d": 20, "power_w": 200}]
    r = th.solve((-20, -20, 40, 40), comps, 0.0, th.Cooling(), 25.0, n=40)
    ser = [s["max_c"] for s in r["transient"]["series"]]
    assert all(b >= a - 1e-6 for a, b in zip(ser, ser[1:]))
    assert ser[-1] == pytest.approx(r["components"][0]["max_c"], rel=0.02)
    assert len(r["transient"]["frames"]) == len(ser)


def test_more_airflow_lowers_heatsink_resistance_and_temperature():
    comps = [{"id": "a", "x": 0, "y": 0, "w": 20, "d": 20, "power_w": 300}]
    slow = th.solve((-20, -20, 40, 40), comps, 0.0, th.Cooling(fan_cfm=20), 25.0, n=32)
    fast = th.solve((-20, -20, 40, 40), comps, 0.0, th.Cooling(fan_cfm=120), 25.0, n=32)
    assert fast["airflow"]["velocity_ms"] > slow["airflow"]["velocity_ms"]
    assert fast["r_heatsink_c_per_w"] < slow["r_heatsink_c_per_w"]
    assert fast["hotspot"]["temp_c"] < slow["hotspot"]["temp_c"]
    assert fast["airflow"]["air_rise_c"] < slow["airflow"]["air_rise_c"]


def test_stack_profile_base_die_is_hottest():
    t = th.stack_profile(60.0, [3.0, 1.0, 1.0, 1.0, 1.0], 110.0)
    assert t[-1] == 60.0 and t[0] > t[1] > t[-1]


# --- scene -----------------------------------------------------------------------------------

def test_scene_geometry_power_and_activity():
    s = build_scene(_gpu(), "studio_gpu_0.1")
    comps = {c["id"]: c for c in s["components"]}
    assert set(comps) == {"gpu[0]", "hbm[0]", "hbm[1]", "hbm[2]", "hbm[3]"}
    hbm = comps["hbm[0]"]
    assert hbm["kind"] == "hbm_stack" and len(hbm["layers"]) == 9           # base + 8 dies
    assert hbm["h"] == pytest.approx(0.06 + 8 * 0.05 + 8 * 0.02, abs=1e-6)
    assert hbm["tsv"]["count_estimate"] > 0 and hbm["capacity_gb"] == 24
    temps = [L["temp_c"] for L in hbm["layers"]]
    assert temps[0] >= temps[-1]                                            # base die hottest
    assert not any("overlaps" in w for w in s["warnings"])
    ip = s["package"]["interposer"]
    for c in s["components"]:                                               # all on the interposer
        assert abs(c["x"] - ip["x"]) + c["w"] / 2 <= ip["w"] / 2 + 1e-6
    assert s["power"]["package_w"] == pytest.approx(sum(c["power_w"] for c in s["components"]))
    assert s["thermal"]["heat_out_w"] == pytest.approx(s["power"]["package_w"], rel=1e-3)
    assert s["simulation"] and s["links"]
    assert comps["gpu[0]"]["internal"]["kind"] == "compute"
    assert hbm["internal"]["channels"] == 16 and len(hbm["internal"]["bank_access"]) == 16
    assert s["pcb"]["vrm"]["phases"] >= s["power"]["package_w"] / 0.8 / 60


def test_link_power_follows_energy_per_bit_not_capacity():
    lo = build_scene(_gpu(), "x")
    d = _gpu()
    d.interconnect.energy_pj_per_bit = 5.0
    hi = build_scene(d, "x")
    assert hi["power"]["interconnect_w"] == pytest.approx(10 * lo["power"]["interconnect_w"], rel=1e-3)
    assert lo["power"]["interconnect_w"] < 50          # not the old flat 0.1 W per GB/s


def test_design_thermal_resistance_overrides_airflow_model():
    s = build_scene(_gpu(r=0.02), "x", simulate=False)
    assert s["thermal"]["r_heatsink_c_per_w"] == 0.02
    assert s["thermal"]["r_source"] == "design constraints"


@pytest.mark.parametrize("path", [f"{ROOT}/projects/echo/echo_v0_1.yaml",
                                  f"{ROOT}/projects/echo/echo_v0_2.yaml",
                                  f"{ROOT}/examples/transformer_accelerator.yaml"])
def test_example_designs_build(path):
    s = build_scene(_load(path), "x")
    assert s["components"] and s["thermal"]["hotspot"]["temp_c"] > 25
    kinds = {c["kind"] for c in s["components"]}
    assert kinds & {"hbm_stack", "dram_chiplet", "compute_die", "pim_chiplet"}


def test_echo_dram_chiplets_show_compute_in_memory_region():
    s = build_scene(_load(f"{ROOT}/projects/echo/echo_v0_1.yaml"), "x")
    dram = [c for c in s["components"] if c["kind"] == "dram_chiplet"]
    assert dram and all(c["internal"]["cim_fraction"] > 0 for c in dram)


def test_specless_design_still_renders():
    d = DesignSpec(name="bare", package=PackageSpec(type="chiplets"),
                   chiplets=[ChipletSpec(type="compute"), ChipletSpec(type="memory")])
    s = build_scene(d, "bare")
    assert len(s["components"]) == 2
    assert any("No compute or memory specs" in w for w in s["warnings"])


# --- edits -----------------------------------------------------------------------------------

def test_apply_edit_placement_stack_and_cooling():
    d = apply_edit(_gpu(), {
        "placement": {"hbm[0]": {"x": 80, "y": 40}},
        "chiplets": [{"chiplet_id": "hbm", "count": 6, "memory": {"stack_layers": 12}}],
        "studio": {"die_thickness_um": 30, "cooling": {"fans": 3}},
        "constraints": {"max_temp_c": 90}})
    s = build_scene(d, "x", simulate=False)
    comps = {c["id"]: c for c in s["components"]}
    assert len([c for c in comps if c.startswith("hbm")]) == 6
    assert comps["hbm[0]"]["x"] == 80 and comps["hbm[0]"]["moved"]
    assert len(comps["hbm[1]"]["layers"]) == 13
    assert len(s["package"]["fans"]) == 3 and s["max_temp_c"] == 90
    reset = build_scene(apply_edit(d, {"reset_placement": True}), "x", simulate=False)
    assert not any(c.get("moved") for c in reset["components"])


@pytest.mark.parametrize("patch", [{"studio": {"bogus": 1}}, {"constraints": {"name": 1}},
                                   {"chiplets": [{"chiplet_id": "nope"}]},
                                   {"chiplets": [{"chiplet_id": "hbm", "memory": {"technology": "x"}}]}])
def test_apply_edit_rejects_unknown_fields(patch):
    with pytest.raises(ValueError):
        apply_edit(_gpu(), patch)


def test_studio_api_scene_and_edit(tmp_path):
    from starlette.testclient import TestClient

    from chiplab.app.server import create_app
    from chiplab.experiments.store import ExperimentStore

    db = str(tmp_path / "s.db")
    ExperimentStore(db_path=db).save_design(_gpu())
    client = TestClient(create_app(db_path=db))
    assert "GPU Studio" in client.get("/").text
    assert client.get("/studio.js").status_code == 200
    r = client.get("/api/studio/scene", params={"design": "studio_gpu_0.1", "workload": "llm70_decode"})
    assert r.status_code == 200 and r.json()["workload"] == "llm70_decode"
    assert client.get("/api/studio/scene", params={"design": "studio_gpu_0.1",
                                                   "workload": "bogus"}).status_code == 400
    origin = {"Origin": "http://testserver"}
    body = {"design_id": "studio_gpu_0.1", "patch": {"chiplets": [{"chiplet_id": "hbm", "count": 6}]}}
    assert client.post("/api/studio/edit", json=body, headers={"Origin": "http://evil.example"}).status_code == 403
    r = client.post("/api/studio/edit", json=body, headers=origin)
    assert r.status_code == 200, r.text
    assert sum(c["chiplet_id"] == "hbm" for c in r.json()["components"]) == 6
    r = client.post("/api/studio/edit", json={**body, "new_version": True}, headers=origin)
    assert r.json()["design_id"] == "studio_gpu_0.2" and r.json()["version"] == "0.2"
    bad = {"design_id": "studio_gpu_0.1", "patch": {"chiplets": [{"chiplet_id": "hbm", "memory": {"channels": 0}}]}}
    r = client.post("/api/studio/edit", json=bad, headers=origin)
    assert r.status_code == 400 and "rejected" in r.json()["error"]
