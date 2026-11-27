"""The physics validation gate, ferrofluid routing and the further analyses.

Every "measurement" below is SYNTHETIC, generated from the model itself, so that the
gate's logic can be exercised.  A device accepted here is not validated against reality.
"""

import copy
import json

import numpy as np
import pytest

from chiplab import mcp_experimental_tools as experimental
from chiplab import mcp_industrial_tools as industrial
from chiplab import mcp_server
from chiplab.cost.model import estimate_unit_cost
from chiplab.credibility import CredibilityError, is_ferrofluid
from chiplab.engine.analytical import run_analytical_simulation
from chiplab.engine.discrete_event import run_discrete_event_simulation
from chiplab.engine.specialized import run_specialized_simulation
from chiplab.engine.validator import validate_design
from chiplab.experiments.store import ExperimentStore
from chiplab.models.materials import ferro_interface
from chiplab.models.materials.ferro_calibration import fit_parameters
from chiplab.models.materials.ferro_particles import ParticleEnsemble
from chiplab.models.materials.ferro_validation import evaluate_record
from chiplab.schemas.design import DesignSpec
from chiplab.schemas.simulation import SimulationConfig
from chiplab.schemas.workload import WorkloadSpec

MATERIAL = {
    "volume_fraction": 0.01, "core_diameter_nm": 10.0, "size_sigma_ln": 0.2,
    "coating_nm": 2.0, "domain_magnetization_a_m": 4.46e5, "anisotropy_j_m3": 2e4,
    "neel_attempt_time_s": 1e-9, "viscosity_pa_s": 2e-3, "temperature_k": 300.0,
}
PORT = {"conductor": "loop", "radius_um": 10, "turns": 2, "wire_radius_um": 1,
        "series_resistance_ohm": 50, "axis": [0, 0, 1]}
SOURCE = {"source": "SYNTHETIC test data generated from the model, not a measurement"}
WRITES = [{"port": "w", "voltage_v": 1.0, "duration_ns": 2000}]
READS = [{"port": "r", "frequency_hz": 1e6, "time_ns": 1900}]


def base_record():
    array = mcp_server.configure_material_array(
        "ferrofluid", 2, "linear", 30,
        [{"id": "w", "role": "write", **PORT}, {"id": "r", "role": "read", **PORT}])
    spec = {"technology": "ferrofluid", **MATERIAL,
            "fluid_region": {"center_um": [15, 0, 6], "size_um": [60, 30, 8],
                             "grid": [30, 15, 6]}}
    return {"device_spec": spec, "electrode_array": array,
            "write_sequence": copy.deepcopy(WRITES),
            "read_sequence": copy.deepcopy(READS), "validation_measurements": {}}


def synthetic_measurements(record, seed=1):
    rng = np.random.default_rng(seed)
    device = ferro_interface.build_device(
        ferro_interface.ports_from_array(record["electrode_array"]),
        record["device_spec"])
    ensemble = ParticleEnsemble(device.params)

    def noisy(values, sigma):
        return (values + rng.normal(0, 1, len(values)) * sigma).tolist()

    h = np.logspace(1, 5, 20)
    m = ensemble.magnetization(h)
    f = np.logspace(4, 7.5, 20)
    chi = ensemble.complex_susceptibility(f)[0]
    sr = 0.02 * np.abs(chi)
    z = (2j * np.pi * f * device.fluid_inductance(f, np.zeros(2))[:, 1, 1])
    sz = 0.02 * np.abs(z)
    return {"provenance": dict(SOURCE), "datasets": [
        {"kind": "magnetization_curve", "H_a_per_m": h.tolist(),
         "M_a_per_m": noisy(m, 0.01 * m), "sigma_M_a_per_m": (0.01 * m).tolist()},
        {"kind": "susceptibility_spectrum", "frequency_hz": f.tolist(),
         "chi_real": noisy(chi.real, sr), "chi_loss": noisy(-chi.imag, sr),
         "sigma_chi_real": sr.tolist(), "sigma_chi_loss": sr.tolist()},
        {"kind": "port_impedance_change", "port": "r", "frequency_hz": f.tolist(),
         "delta_z_real_ohm": noisy(z.real, sz), "delta_z_imag_ohm": noisy(z.imag, sz),
         "sigma_delta_z_real_ohm": sz.tolist(), "sigma_delta_z_imag_ohm": sz.tolist()},
    ]}


@pytest.fixture(scope="module")
def complete():
    record = base_record()
    record["validation_measurements"] = synthetic_measurements(record)
    return record


def statuses(record):
    report = evaluate_record(record)
    return report["accepted"], {c["id"]: c["status"] for c in report["checks"]}


EXPERIMENTS = ("experiment_equilibrium", "experiment_dynamics", "experiment_port_response")


class TestGateDecisions:
    def test_complete_evidence_passes_every_check(self, complete):
        report = evaluate_record(complete)
        assert report["accepted"], [c for c in report["checks"] if c["status"] != "pass"]
        assert report["summary"] == {"pass": 11, "fail": 0, "not_run": 0}
        assert all(c["evidence"] for c in report["checks"])
        json.dumps(report, allow_nan=False)

    def test_uncalibrated_device_is_rejected_by_default(self):
        accepted, status = statuses(base_record())
        assert not accepted
        assert [status[i] for i in EXPERIMENTS] == ["not_run"] * 3

    @pytest.mark.parametrize("kind,check", [
        ("magnetization_curve", "experiment_equilibrium"),
        ("susceptibility_spectrum", "experiment_dynamics"),
        ("port_impedance_change", "experiment_port_response")])
    def test_each_missing_measurement_blocks(self, complete, kind, check):
        record = copy.deepcopy(complete)
        data = record["validation_measurements"]["datasets"]
        data[:] = [d for d in data if d["kind"] != kind]
        accepted, status = statuses(record)
        assert not accepted and status[check] == "not_run"

    def test_measurements_without_a_source_do_not_count(self, complete):
        record = copy.deepcopy(complete)
        record["validation_measurements"]["provenance"] = {}
        accepted, status = statuses(record)
        assert not accepted and status["experiment_equilibrium"] == "not_run"

    @pytest.mark.parametrize("change", [
        lambda d: d.update(M_a_per_m=[1.4 * v for v in d["M_a_per_m"]]),  # disagrees
        lambda d: d.update(sigma_M_a_per_m=[1e6] * 20),  # uncertainty hides anything
        lambda d: d.pop("sigma_M_a_per_m"),  # no uncertainty
        lambda d: d.update(temperature_k=340.0),  # other temperature
        lambda d: d.update({k: d[k][:3] for k in  # too few points
                            ("H_a_per_m", "M_a_per_m", "sigma_M_a_per_m")}),
    ])
    def test_inadequate_measurements_fail(self, complete, change):
        record = copy.deepcopy(complete)
        change(record["validation_measurements"]["datasets"][0])
        accepted, status = statuses(record)
        assert not accepted and status["experiment_equilibrium"] == "fail"

    def test_data_must_cover_the_operating_range(self, complete):
        record = copy.deepcopy(complete)
        record["read_sequence"][0]["frequency_hz"] = 9e7
        accepted, status = statuses(record)
        assert not accepted
        assert status["experiment_dynamics"] == "fail"
        assert status["experiment_port_response"] == "fail"

    def test_calibration_data_cannot_validate(self, complete):
        record = copy.deepcopy(complete)
        curve = record["validation_measurements"]["datasets"][0]
        fit = fit_parameters([curve], ["volume_fraction"], SOURCE, dict(MATERIAL))
        record["device_spec"]["calibration"] = fit
        accepted, status = statuses(record)
        assert not accepted and status["experiment_equilibrium"] == "fail"

    def test_assumed_parameter_or_geometry_blocks(self, complete):
        record = copy.deepcopy(complete)
        del record["device_spec"]["viscosity_pa_s"]
        assert statuses(record)[1]["inputs_established"] == "fail"
        record = copy.deepcopy(complete)
        del record["electrode_array"]["port_config"][0]["turns"]
        assert statuses(record)[1]["inputs_established"] == "fail"

    @pytest.mark.parametrize("change", [
        lambda r: r["write_sequence"][0].update(voltage_v=200.0),  # melts the wire
        lambda r: r["device_spec"].update(core_diameter_nm=30.0),  # particles chain
        lambda r: r["device_spec"].update(volume_fraction=0.18, coating_nm=0.5),  # too dense
    ])
    def test_unrealistic_device_is_rejected(self, complete, change):
        record = copy.deepcopy(complete)
        change(record)
        accepted, status = statuses(record)
        assert not accepted and status["model_validity"] == "fail"

    def test_device_outside_the_model_is_rejected_with_a_report(self, complete):
        record = copy.deepcopy(complete)
        record["device_spec"]["volume_fraction"] = 0.2  # coated particles overfill
        out = to_chiplet(record)
        assert out["accepted"] is False and out["chiplet"] is None
        accepted, status = statuses(record)
        assert not accepted and status.pop("device_definition") == "fail"
        assert set(status.values()) == {"not_run"} and len(status) == 11

    def test_no_operating_conditions_means_not_run(self, complete):
        record = copy.deepcopy(complete)
        record["write_sequence"] = []
        accepted, status = statuses(record)
        assert not accepted and status["zero_input"] == "not_run"


def to_chiplet(record):
    return mcp_server.material_device_to_chiplet(
        record["device_spec"], record["electrode_array"], record["write_sequence"],
        record["read_sequence"], record["validation_measurements"])


def design_with(chiplet, name="gate_test"):
    return {"name": name, "chiplets": [
        {"id": "core", "type": "compute", "area_mm2": 100, "process_nm": 5, "tdp_watts": 50,
         "compute": {"architecture": "systolic", "peak_tops": 100, "sram_mb": 8}}, chiplet]}


HAND_WRITTEN = {"id": "ff", "type": "memory", "memory": {
    "technology": "ferrofluid", "capacity_per_chiplet_gb": 1000,
    "bandwidth_per_channel_gbps": 999, "latency_ns": 0.1}}


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    store = ExperimentStore(db_path=str(tmp_path / "gate.db"))
    monkeypatch.setattr(mcp_server, "store", store)
    return store


def engine_calls(design):
    config, workload = SimulationConfig(), WorkloadSpec()
    return [
        lambda: run_analytical_simulation(design, workload, config),
        lambda: run_discrete_event_simulation(design, workload, config),
        lambda: run_specialized_simulation(design, workload, config),
        lambda: estimate_unit_cost(design),
    ]


class TestNoWayAround:
    def test_conversion_returns_no_chiplet_without_measurements(self):
        out = to_chiplet(base_record())
        assert out["accepted"] is False and out["chiplet"] is None
        assert "experiment_equilibrium: not_run" in out["error_message"]

    def test_other_materials_have_no_checks_and_are_rejected(self):
        out = mcp_server.material_device_to_chiplet(
            {"technology": "memristor"}, {"n_ports": 2})
        assert out["accepted"] is False and out["chiplet"] is None

    @pytest.mark.parametrize("chiplet", [
        HAND_WRITTEN,
        {"id": "x", "type": "compute", "compute": {"architecture": "Ferro-Fluid PIM"}},
        {"id": "magnetic_fluid_bank", "type": "memory"},
        {"id": "x", "type": "material", "area_mm2": 1},
        {"id": "material_memristor", "type": "memory"},
        {"id": "x", "type": "material", "material_device": {"accepted": True}},
    ])
    def test_hand_written_chiplets_are_refused_everywhere(self, chiplet, isolated_store):
        spec = design_with(chiplet)
        design = DesignSpec.model_validate(spec)
        assert not validate_design(design).valid
        for call in engine_calls(design):
            with pytest.raises(CredibilityError):
                call()
        for call in (
                lambda: mcp_server.chip_create_design(spec),
                lambda: mcp_server.chip_evaluate(spec=spec),
                lambda: mcp_server.chip_estimate_cost(spec=spec),
                lambda: mcp_server.chip_uncertainty(spec=spec),
                lambda: isolated_store.save_design(design)):
            with pytest.raises(CredibilityError):
                call()

    def test_template_override_and_model_copy_do_not_bypass(self, isolated_store):
        name = next(iter(mcp_server.chip_list_templates()["templates"]))
        with pytest.raises(CredibilityError):
            mcp_server.chip_design_from_template(name, overrides={"chiplets": [HAND_WRITTEN]})
        clean = DesignSpec.model_validate({"name": "c", "chiplets": [
            design_with(HAND_WRITTEN)["chiplets"][0]]})
        dirty = clean.model_copy(update={"chiplets": DesignSpec.model_validate(
            design_with(HAND_WRITTEN)).chiplets})  # skips pydantic validation
        with pytest.raises(CredibilityError):
            run_analytical_simulation(dirty, WorkloadSpec(), SimulationConfig())

    def test_design_already_in_the_database_is_still_refused(self, isolated_store,
                                                            monkeypatch):
        design = DesignSpec.model_validate(design_with(HAND_WRITTEN))
        with monkeypatch.context() as patch:  # as if saved before the gate existed
            patch.setattr("chiplab.experiments.store.require_credible_design", lambda d: None)
            design_id = isolated_store.save_design(design)
        for call in (lambda: mcp_server.chip_generate_rtl(design_id),
                     lambda: mcp_server.chip_verify_design(design_id, ["connectivity"]),
                     lambda: mcp_server.chip_export_full_project(design_id),
                     lambda: mcp_server.chip_run_simulation(design_id)):
            with pytest.raises(CredibilityError):
                call()
        assert mcp_server.chip_validate_design(design_id)["valid"] is False

    def test_python_integration_route_is_gated(self):
        from chiplab.models.materials.chip_integration import (
            MaterialDeviceSpec,
            TechnologyType,
            derive_chip_parameters,
            integrate_into_design,
        )

        device = MaterialDeviceSpec(TechnologyType.FERROFLUID, (2.0, 2.0, 1.0), 64, 10.0,
                                    10.0, 15.0, {}, "sealed", 1.0)
        base = DesignSpec.model_validate({"name": "d", "chiplets": [
            design_with(HAND_WRITTEN)["chiplets"][0]]})
        with pytest.raises(CredibilityError):
            integrate_into_design(base, device)
        with pytest.raises(CredibilityError):
            derive_chip_parameters(
                {"cell_count": 4, "bits_per_cell": 2, "cell_area_mm2": 1,
                 "write_energy_pj": 1, "read_energy_pj": 1, "operating_frequency_hz": 1e6},
                {"electrode_count": 2, "active_area_mm2": 1})

    def test_accepted_device_enters_with_only_modelled_numbers(self, complete,
                                                              isolated_store):
        out = to_chiplet(complete)
        assert out["accepted"], out["error_message"]
        chiplet = out["chiplet"]
        assert set(chiplet) == {"id", "type", "area_mm2", "tdp_watts", "material_device"}
        assert chiplet["tdp_watts"] == pytest.approx(
            1.0 / ferro_interface.ports_from_array(
                complete["electrode_array"])[0].resistance_ohm())
        spec = design_with(chiplet)
        assert mcp_server.chip_create_design(spec)["validation"]["valid"]
        design = DesignSpec.model_validate(spec)
        result = run_analytical_simulation(design, WorkloadSpec(), SimulationConfig())
        assert any("not modelled" in w for w in result.warnings)
        cost = estimate_unit_cost(design)
        assert any("NOT MODELLED" in w for w in cost["warnings"])
        line = next(c for c in cost["chiplets"] if c["kind"] == "material")
        assert line["subtotal_usd"] == 0
        without = run_analytical_simulation(
            DesignSpec.model_validate({"name": "base", "chiplets": spec["chiplets"][:1]}),
            WorkloadSpec(), SimulationConfig()).metrics
        assert (result.metrics.total_power_watts - without.total_power_watts
                == pytest.approx(chiplet["tdp_watts"], rel=1e-6))
        assert result.metrics.peak_compute_tops == without.peak_compute_tops
        assert mcp_server.chip_evaluate(spec=spec)["design"] == "gate_test"
        for fidelity in ("discrete_event", "specialized"):
            assert mcp_server.chip_run_simulation("gate_test", fidelity=fidelity)["metrics"]
        bands = mcp_server.chip_uncertainty("gate_test", samples=10)
        assert "chiplets[1]" not in json.dumps(bands)

    @pytest.mark.parametrize("tamper", [
        lambda c: c.update(area_mm2=c["area_mm2"] * 2),
        lambda c: c.update(tdp_watts=1e-9),
        lambda c: c.update(memory=HAND_WRITTEN["memory"]),
        lambda c: c.update(type="memory"),
        lambda c: c.update(process_nm=3),
        lambda c: c["material_device"].update(validation_measurements={}),
        lambda c: c["material_device"]["device_spec"].update(volume_fraction=0.05),
    ])
    def test_tampering_with_an_accepted_chiplet_is_caught(self, complete, tamper):
        chiplet = copy.deepcopy(to_chiplet(complete)["chiplet"])
        tamper(chiplet)
        design = DesignSpec.model_validate(design_with(chiplet))
        with pytest.raises(CredibilityError):
            estimate_unit_cost(design)


class TestRouting:
    """The second plugin report: generic or preset numbers for a ferrofluid."""

    NAMES = ["ferrofluid", "Ferrofluid memory", "ferro-fluid", "magnetic_fluid",
             "magnetic nanoparticle suspension"]

    @pytest.mark.parametrize("name", NAMES)
    def test_every_spelling_reaches_the_particle_model(self, name):
        assert is_ferrofluid(name)
        out = experimental.chiplab_simulate_experimental_tech(name)
        assert out["metric_status"]["retention_s"] == "not_modelled"
        assert "model_status" not in mcp_server.simulate_material_device(name, {})

    @pytest.mark.parametrize("spec", [
        {"material": "ferrofluid"}, {"type": "ferrofluid"}, {"name": "ferrofluid device"},
        {"regions": [{"material": "ferrofluid"}]}])
    def test_every_field_reaches_the_particle_model(self, spec):
        array = mcp_server.configure_material_array("generic", 2, "linear", 100)
        out = mcp_server.run_multiphysics_simulation(spec, array, [], [])
        assert out["device_technology"] == "ferrofluid" and "model_status" not in out
        assert "model_status" not in mcp_server.simulate_material_circuit(
            spec, array, {}, {}, "dc")

    def test_conflicting_declarations_are_refused(self):
        with pytest.raises(CredibilityError, match="ambiguous"):
            mcp_server.run_multiphysics_simulation(
                {"technology": "memristor", "name": "ferrofluid"}, {"n_ports": 2}, [], [])

    @pytest.mark.parametrize("name,low,high", [
        ("volume_fraction", 0.01, 0.02), ("core_diameter_nm", 10, 13),
        ("viscosity_pa_s", 2e-3, 8e-3), ("coating_nm", 2, 6),
        ("anisotropy_j_m3", 2e4, 4e4), ("temperature_k", 300, 350)])
    def test_six_material_parameters_each_change_the_output(self, name, low, high):
        first, second = (mcp_server.simulate_material_device(
            "ferrofluid", {name: v})["metrics"] for v in (low, high))
        assert first["read_flux_linkage_wb"] != second["read_flux_linkage_wb"]

    def test_array_aliases(self):
        spec = mcp_server.configure_material_array(
            "ferrofluid", 4, "square", 100, {"0": {"type": "write"}})
        assert spec["port_config"][0]["role"] == "write"

    def test_generic_paths_refuse_ferrofluid(self):
        from chiplab.engines import get_solver
        from chiplab.engines.multiphysics_solver import ParticleScaleSolver
        from chiplab.manufacturing import ManufacturingPipeline
        from chiplab.multiphysics import FerrofluidCoupling

        calls = [
            lambda: experimental.chiplab_scale_technology_node("ferrofluid", [28]),
            lambda: industrial.chiplab_reliability_analyze("ferro-fluid", 400, 300, 10),
            lambda: industrial.chiplab_manufacturing_signoff(
                "d", material_spec={"material": "ferrofluid"}),
            lambda: industrial.chiplab_discover_architectures({"memory": ["ferrofluid"]}),
            lambda: industrial.chiplab_digital_twin_manage(
                "evaluate", design_data={"memory": "ferrofluid"}),
            lambda: mcp_server.component_create({"name": "ferrofluid_cell"}),
            lambda: ManufacturingPipeline().material_stage({"material": "ferrofluid"}, 1.0),
            lambda: get_solver("analytical", material="ferrofluid"),
            lambda: ParticleScaleSolver(material="ferrofluid"),
            lambda: get_solver("analytical").solve(
                np.array([1.0, 0, 0]), {"material": "ferrofluid"}, 1e-12),
            lambda: FerrofluidCoupling(None, None),
        ]
        for call in calls:
            with pytest.raises(CredibilityError):
                call()


def analyze(kind, **options):
    record = base_record()
    return mcp_server.chiplab_ferrofluid_analyze(
        kind, record["device_spec"], record["electrode_array"], record["write_sequence"],
        record["read_sequence"], options)


class TestAnalyses:
    def test_sensitivity_limits(self):
        # a write much longer than every relaxation time, so the state is equilibrated
        record = base_record()
        record["write_sequence"][0]["duration_ns"] = 1e9
        record["read_sequence"][0]["time_ns"] = 1e9
        out = mcp_server.chiplab_ferrofluid_analyze(
            "sensitivity", record["device_spec"], record["electrode_array"],
            record["write_sequence"], record["read_sequence"], {})
        table = out["parameters"]
        key = "read0_r.flux_wb"
        assert table["drive_voltage"]["sensitivity"][key] == pytest.approx(1.0, abs=0.01)
        # weak field, mean-field: flux grows slightly faster than the particle fraction
        assert 1.0 <= table["volume_fraction"]["sensitivity"][key] < 1.3
        decay = table["viscosity_pa_s"]["sensitivity"]["r.decay_1_over_e_s"]
        assert 0 < decay <= 1.01  # Brownian time is proportional to viscosity, Neel is not
        assert out["no_effect_on_these_outputs"] == []

    def test_read_disturbance_vanishes_with_amplitude_and_grows_as_square(self):
        runs = [analyze("read_disturbance", amplitude_v=v, frequency_hz=1e7, cycles=3)
                for v in (1e-4, 0.5, 1.0)]
        disturbance = [r["disturbance_fraction"] for r in runs]
        residue = [r["read_residue_fraction"] for r in runs]
        assert disturbance[0] < 1e-8
        assert disturbance[2] / disturbance[1] == pytest.approx(4.0, rel=0.1)
        assert residue[2] / residue[1] == pytest.approx(2.0, rel=0.05)

    def test_identical_states_are_never_distinguishable(self):
        out = analyze("retention", state_b_write_sequence=WRITES)
        assert out["distinguishable"] is False and out["distinguishable_until_s"] is None

    def test_distinguishable_time_matches_single_exponential(self):
        record = base_record()
        record["device_spec"].update(size_sigma_ln=0.0)
        record["write_sequence"][0]["voltage_v"] = 5.0
        out = mcp_server.chiplab_ferrofluid_analyze(
            "retention", record["device_spec"], record["electrode_array"],
            record["write_sequence"], None, {"bandwidth_hz": 1e9, "snr_threshold": 0.002})
        tau = ParticleEnsemble(ferro_interface.particle_params(
            record["device_spec"])).tau_zero_field[0]
        expected = out["earliest_sample_s"] + tau * np.log(
            out["snr_at_earliest_sample"] / 0.002)
        assert out["distinguishable"]
        assert out["distinguishable_until_s"] == pytest.approx(expected, rel=0.05)

    def test_scaling_is_checked_against_direct_simulation(self):
        out = analyze("scaling", small_shape=[3, 3], large_shape=[4, 4], voltage_v=0.5,
                      geometry=PORT, pitch_um=40)
        assert out["status"] == "validated_against_direct_model", out["reason"]
        assert out["worst_relative_error"] <= 0.1 and "4 x 4" in out["validity_range"]

    @pytest.mark.parametrize("options,reason", [
        ({"large_shape": [8, 8], "voltage_v": 0.5}, "cannot be checked"),
        ({"large_shape": [3, 3], "voltage_v": 200.0}, "nonlinear"),
    ])
    def test_scaling_reports_not_modelled(self, options, reason):
        out = analyze("scaling", geometry=PORT, pitch_um=40, **options)
        assert out["status"] == "not_modelled" and reason in out["reason"]
        assert "ports" not in out
