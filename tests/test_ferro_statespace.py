"""Electrical state-space model, measurement import/fit, reference harness, registry.

All "measurements" are SYNTHETIC, generated from the model with arbitrary parameters.
They test the code, not any ferrofluid.
"""

import copy
import json

import numpy as np
import pytest

from chiplab import mcp_server, model_registry
from chiplab.models.materials import ferro_experiment as ex
from chiplab.models.materials import ferro_reference as ref
from chiplab.models.materials import ferro_statespace as ss

GROUND = {
    "r_inf_ohm": 60.0, "delta_r_ohm": 400.0, "tau_dielectric_s": 2e-9, "alpha": 0.8,
    "capacitance_f": 1e-13, "fast_write_voltage_v": 0.8, "fast_write_time_s": 5.0,
    "fast_saturation_voltage_v": 2.0, "fast_retention_s": 20.0, "fast_gain_delta_r": 0.3,
}
SLOW = {"slow_write_voltage_v": 1.0, "slow_write_time_s": 50.0,
        "slow_saturation_voltage_v": 2.0, "slow_retention_s": 5000.0,
        "slow_gain_delta_r": 0.1}
FATIGUE = {"fatigue_depth": 0.5, "fatigue_voltage_v": 1.0, "fatigue_time_s": 100.0,
           "fatigue_recovery_s": 1000.0}
SYNTHETIC = {"source": "SYNTHETIC data generated from the model, not a measurement"}
F = [1e7, 1e8, 1e9]


def two_port(ground=None, coupling=None):
    g = {**GROUND, **(ground or {})}
    return {"ports": ["a", "b"], "ground": g,
            "coupling": {**g, "delta_r_ohm": 900.0, "target_symmetry": "even",
                         **(coupling or {})}}


def one_port(**ground):
    return {"ports": ["a"], "ground": {**GROUND, **ground}, "coupling": None}


def hold(volts, seconds, temperature=None):
    segment = {"duration_s": seconds, "voltages_v": volts}
    if temperature is not None:
        segment["temperature_k"] = temperature
    return segment


def final_state(model, *segments):
    device = ss.StateSpaceDevice(model)
    total = sum(s["duration_s"] for s in segments)
    return ss.run_protocol(device, {"segments": list(segments)}, [total])[0]["state"]


class TestStateEquation:
    def test_zero_drive_leaves_rest_state(self):
        state = final_state(two_port(SLOW), hold({"a": 0, "b": 0}, 100.0))
        assert not state.fast.any() and not state.slow.any() and not state.fatigue.any()

    def test_exact_step_matches_numerical_integration(self):
        v, y, dt = 1.7, 0.0, 1e-4
        drive = (np.cosh(v / 0.8) - 1) / 5.0
        for _ in range(20_000):  # 2 s of explicit midpoint steps
            half = y + 0.5 * dt * (drive * (np.tanh(v / 2.0) - y) - y / 20.0)
            y += dt * (drive * (np.tanh(v / 2.0) - half) - half / 20.0)
        assert final_state(one_port(), hold({"a": v}, 2.0)).fast[0] == pytest.approx(
            y, rel=1e-6)

    def test_bipolar_writes_are_opposite(self):
        up = final_state(one_port(), hold({"a": 2.0}, 3.0)).fast[0]
        down = final_state(one_port(), hold({"a": -2.0}, 3.0)).fast[0]
        assert up > 0 and down == pytest.approx(-up)

    def test_longer_and_stronger_pulses_write_more(self):
        by_time = [final_state(one_port(), hold({"a": 1.5}, t)).fast[0] for t in (0.1, 1, 10)]
        by_volts = [final_state(one_port(), hold({"a": v}, 1.0)).fast[0] for v in (0.5, 1, 2)]
        assert by_time == sorted(by_time) and by_volts == sorted(by_volts)
        assert by_time[0] < by_time[-1] and by_volts[0] < by_volts[-1]

    def test_retention_and_nonvolatile_option(self):
        write, rest = hold({"a": 2.5}, 100.0), hold({"a": 0.0}, 200.0)
        written = final_state(one_port(**SLOW), write)
        later = final_state(one_port(**SLOW), write, rest)
        assert later.fast[0] == pytest.approx(written.fast[0] * np.exp(-10), rel=1e-6)
        assert later.slow[0] == pytest.approx(written.slow[0] * np.exp(-0.04), rel=1e-6)
        forever = one_port(fast_retention_s="never")
        assert final_state(forever, write, rest).fast[0] == pytest.approx(
            final_state(forever, write).fast[0])

    def test_temperature_follows_arrhenius(self):
        model = one_port(fast_retention_activation_ev=0.4, reference_temperature_k=300.0)
        write = hold({"a": 2.0}, 5.0, 300.0)
        start = final_state(model, write).fast[0]
        for temperature in (300.0, 330.0):
            after = final_state(model, write, hold({"a": 0.0}, 10.0, temperature)).fast[0]
            tau = 20.0 * np.exp(0.4 / ss.KB_EV * (1 / temperature - 1 / 300.0))
            assert after == pytest.approx(start * np.exp(-10.0 / tau), rel=1e-6)
        with pytest.raises(ValueError, match="temperature"):
            final_state(model, hold({"a": 1.0}, 1.0))

    def test_prolonged_operation_shrinks_response_reversibly(self):
        device = ss.StateSpaceDevice(one_port(**FATIGUE))

        def response(state):  # change caused by one test pulse from a rested memory state
            probe = state.copy()
            probe.fast[:] = 0
            written = device.step(probe, {"a": 2.0}, 0.5)
            written.fatigue[:] = state.fatigue
            z = [device.impedance_matrix([1e7], s)[0, 0, 0] for s in (probe, written)]
            return abs(z[1] - z[0])

        fresh = device.initial_state()
        tired = device.step(fresh, {"a": 2.0}, 2000.0)
        rested = device.step(tired, {"a": 0.0}, 1e5)
        assert response(tired) < 0.7 * response(fresh)
        assert response(rested) == pytest.approx(response(fresh), rel=1e-6)

    def test_sweep_shows_hysteresis(self):
        sweep = [*np.linspace(-2, 2, 21), *np.linspace(1.8, -1.8, 19)]
        device = ss.StateSpaceDevice(one_port())
        snaps = ss.run_protocol(
            device, {"segments": [hold({"a": float(v)}, 1.0) for v in sweep], "repeat": 2},
            [40.0 + k for k in range(1, 41)])
        z = np.array([abs(device.impedance_matrix([1e7], s["state"])[0, 0, 0])
                      for s in snaps])
        assert abs(z[10] - z[30]) > 1.0  # 0 V rising and falling differ


class TestImpedance:
    def test_single_branch_limits(self):
        device = ss.StateSpaceDevice(one_port(capacitance_f=0.0))
        z = device.impedance_matrix([0.0, 1e15], device.initial_state())[:, 0, 0]
        assert z[0] == pytest.approx(460.0)
        assert abs(z[1]) == pytest.approx(60.0, rel=1e-3)

    @pytest.mark.parametrize("ports", [2, 4, 8, 16])
    def test_n_port_matrix_is_reciprocal_and_passive(self, ports):
        names = [f"p{i}" for i in range(ports)]
        model = {**two_port(SLOW), "ports": names}
        volts = {p: (2.0 if i % 2 else None) for i, p in enumerate(names)}
        out = ss.simulate(model, {"segments": [hold(volts, 3.0)]}, [0.0, 3.0], F)
        for read in out["reads"]:
            z = np.array(read["z"]["real_ohm"]) + 1j * np.array(read["z"]["imag_ohm"])
            assert z.shape == (3, ports, ports)
            np.testing.assert_allclose(z, z.transpose(0, 2, 1), rtol=1e-10)
            assert np.linalg.eigvalsh((z + z.conj().transpose(0, 2, 1)) / 2).min() > 0
        assert out["substep_convergence"] < 0.01
        assert (len(out["warnings"]) > 1) == (ports > 2)  # extrapolation flagged
        json.dumps(out, allow_nan=False)

    def test_floating_port_divides_voltage_and_shares_state(self):
        device = ss.StateSpaceDevice(two_port())
        state = device.initial_state()
        node = device.port_voltages(state, {"a": 2.0, "b": None}, None)
        r_ground, r_coupling = 460.0, 960.0
        assert node[1] == pytest.approx(2.0 * r_ground / (r_ground + r_coupling), rel=1e-6)
        after = device.step(state, {"a": 2.0, "b": None}, 5.0)
        assert np.all(after.fast > 0)  # writing a moved b's branch and the coupling branch
        grounded = device.step(state, {"a": 2.0, "b": 0.0}, 5.0)
        assert grounded.fast[1] == 0


class TestMultiport:
    def test_unconnected_ports_do_not_interact(self):
        model = {"ports": ["a", "b"], "ground": GROUND, "coupling": None}
        out = ss.multiport_report(model, {"a": 2.0, "b": -2.0}, 3.0, F, idle_ports="ground",
                                  noise_sigma_ohm=0.01)
        assert out["interaction_residual_fraction"] < 1e-9
        assert out["retrieval"]["status"] == "all_patterns_distinguishable"

    def test_shared_fluid_interacts_and_noise_limits_retrieval(self):
        args = (two_port(), {"a": 2.0, "b": 2.0}, 3.0, F)
        out = ss.multiport_report(*args, noise_sigma_ohm=1e4)
        assert out["interaction_residual_fraction"] > 0.05
        assert out["retrieval"]["status"] == "not_distinguishable"
        assert ss.multiport_report(*args)["retrieval"]["status"] == "not_computed"

    def test_sixteen_ports_report_only_a_partial_check(self):
        model = {**two_port(), "ports": [f"p{i}" for i in range(16)]}
        out = ss.multiport_report(model, {p: 2.0 for p in model["ports"]}, 3.0, [1e8],
                                  noise_sigma_ohm=1e-3)
        retrieval = out["retrieval"]
        assert retrieval["exhaustive"] is False and retrieval["possible_patterns"] == 65536
        assert retrieval["status"] in ("partial_check_passed", "not_distinguishable")


class TestInputs:
    def test_no_default_values(self):
        broken = dict(GROUND)
        del broken["delta_r_ohm"]
        with pytest.raises(ValueError, match="no default"):
            ss.StateSpaceDevice({"ports": ["a"], "ground": broken, "coupling": None})

    @pytest.mark.parametrize("change", [
        {"alpha": 1.5}, {"r_inf_ohm": -1}, {"typo_ohm": 1}, {"slow_write_time_s": 1.0},
        {"fatigue_depth": 0.5}, {"fast_write_activation_ev": 0.2},
        {"slow_gain_delta_r": 0.1}, {"target_symmetry": "both"}])
    def test_bad_parameters_rejected(self, change):
        with pytest.raises(ValueError):
            ss.StateSpaceDevice(one_port(**change))

    def test_every_port_voltage_is_explicit(self):
        with pytest.raises(ValueError, match="every port"):
            final_state(two_port(), hold({"a": 1.0}, 1.0))

    def test_sensitivity_explains_idle_parameters(self):
        model = one_port(reference_temperature_k=300.0, resistance_activation_ev=0.1,
                         **SLOW)
        out = ss.sensitivity(model, {"segments": [hold({"a": 0.0}, 5.0, 300.0)]}, F)
        assert "ground.resistance_activation_ev" in out["no_effect"]
        assert out["parameters"]["ground.delta_r_ohm"]["affects_this_protocol"]
        written = ss.sensitivity(model, {"segments": [hold({"a": 2.0}, 5.0, 320.0)]}, F)
        assert "ground.resistance_activation_ev" not in written["no_effect"]
        assert written["parameters"]["ground.fast_write_time_s"]["max_state_change"] > 0


class TestImport:
    def test_trace_to_protocol(self):
        protocol = ex.protocol_from_trace(
            [10.0, 11.0, 12.0, 14.0], {"a": [1.0, 1.0, 0.0, 0.0]}, 300.0)
        assert [s["duration_s"] for s in protocol["segments"]] == [2.0, 2.0]
        assert ex.normalise_protocol({"time_s": [0, 1], "voltages_v": {"a": [1, 1]}})

    def test_csv_columns(self, tmp_path):
        path = tmp_path / "trace.csv"
        path.write_text("# synthetic\ntime_s,a\n0,1\n1,2\n")
        assert ex.read_csv_columns(path) == {"time_s": [0.0, 1.0], "a": [1.0, 2.0]}
        path.write_text("time_s,a\n0,x\n")
        with pytest.raises(ValueError, match="row 2"):
            ex.read_csv_columns(path)

    @pytest.mark.parametrize("fmt", ["RI", "MA", "DB"])
    def test_touchstone_recovers_impedance(self, fmt):
        z = np.array([[100 + 20j, 30 + 0j], [10 + 0j, 70 - 5j]])  # deliberately Z12 != Z21
        s = (z - 50 * np.eye(2)) @ np.linalg.inv(z + 50 * np.eye(2))

        def pair(x):
            if fmt == "RI":
                return f"{x.real:.12g} {x.imag:.12g}"
            size = abs(x) if fmt == "MA" else 20 * np.log10(abs(x))
            return f"{size:.12g} {np.degrees(np.angle(x)):.12g}"

        row = " ".join(pair(s[i, j]) for i, j in ((0, 0), (1, 0), (0, 1), (1, 1)))
        text = f"! synthetic\n# MHz S {fmt} R 50\n10 {row}\n20 {row}\n"
        data = ex.touchstone_datasets(text, ["a", "b"], 1.0, sigma_relative=0.01)
        found = {(d["to_port"], d["from_port"]):
                 d["z_real_ohm"][0] + 1j * d["z_imag_ohm"][0] for d in data}
        assert found[("a", "b")] == pytest.approx(30)
        assert found[("b", "a")] == pytest.approx(10)
        assert found[("a", "a")] == pytest.approx(100 + 20j)
        assert data[0]["frequency_hz"] == [1e7, 2e7]
        with pytest.raises(ValueError, match="uncertainty"):
            ex.touchstone_datasets(text, ["a", "b"], 1.0)


def synthetic_record(model, protocol, times, entries=("a",), seed=0, relative=0.002,
                     item=None):
    """Noisy readings generated from `model` (synthetic)."""
    rng = np.random.default_rng(seed)
    device = ss.StateSpaceDevice(model)
    snapshots = ss.run_protocol(device, protocol, times)
    datasets = []
    for port in entries:
        i = device.index(port)
        rows = {k: [] for k in ("time_s", "frequency_hz", "z_real_ohm", "z_imag_ohm",
                                "sigma_z_real_ohm", "sigma_z_imag_ohm")}
        for s in snapshots:
            z = device.impedance_matrix(F, s["state"], s["node_v"], s["temperature_k"])
            for f, value in zip(F, z[:, i, i], strict=True):
                sigma = relative * abs(value)
                noisy = value + sigma * complex(rng.normal(), rng.normal())
                for key, x in zip(rows, (s["time_s"], f, noisy.real, noisy.imag, sigma,
                                         sigma), strict=True):
                    rows[key].append(float(x))
        data = {"kind": "impedance", "to_port": port, "from_port": port, **rows}
        if item:
            data["reference_item"] = item
        datasets.append(data)
    return {"provenance": dict(SYNTHETIC), "protocol": protocol, "datasets": datasets}


PROTOCOL = {"segments": [hold({"a": 2.0}, 10.0), hold({"a": 0.0}, 40.0),
                         hold({"a": -1.5}, 10.0), hold({"a": 0.0}, 20.0)]}
TIMES = [0.0, 1, 2, 4, 7, 10, 15, 25, 40, 50, 52, 56, 60, 70, 80]
FITTED = ["ground.delta_r_ohm", "ground.fast_gain_delta_r", "ground.fast_write_time_s",
          "ground.fast_retention_s"]


@pytest.fixture(scope="module")
def calibrated():
    truth = one_port()
    start = one_port(delta_r_ohm=300.0, fast_gain_delta_r=0.2, fast_write_time_s=9.0,
                     fast_retention_s=40.0)
    return truth, ex.fit(start, synthetic_record(truth, PROTOCOL, TIMES), FITTED)


class TestCalibration:
    def test_fit_recovers_synthetic_parameters(self, calibrated):
        truth, fit = calibrated
        assert fit["status"] == "fitted_to_user_data", fit["warnings"]
        assert fit["independently_validated"] is False
        for path, row in fit["parameters"].items():
            expected = truth["ground"][path.split(".")[1]]
            assert row["value"] == pytest.approx(expected, rel=0.05)
            assert row["ci95"][0] < row["value"] < row["ci95"][1]
        json.dumps(fit, allow_nan=False)

    def test_comparison_separates_calibration_and_held_out_data(self, calibrated):
        truth, fit = calibrated
        reused = ex.compare(fit["model"], synthetic_record(truth, PROTOCOL, TIMES))
        assert reused["independent_datasets"] == 0
        assert reused["independent_agreement"] is None
        other = {"segments": [hold({"a": 1.2}, 20.0), hold({"a": 0.0}, 30.0)]}
        held = ex.compare(fit["model"], synthetic_record(truth, other, [0, 5, 10, 20, 35, 50],
                                                         seed=7))
        assert held["independent_agreement"] is True

    def test_wrong_model_disagrees(self, calibrated):
        _, fit = calibrated
        wrong = synthetic_record(one_port(fast_gain_delta_r=-0.3), PROTOCOL, TIMES, seed=2)
        assert ex.compare(fit["model"], wrong)["all_agree"] is False

    def test_undetermined_parameter_is_reported(self):
        record = synthetic_record(one_port(**SLOW), {"segments": [hold({"a": 0.0}, 10.0)]},
                                  [0, 2, 4, 6, 8, 10])
        out = ex.fit(one_port(**SLOW), record,
                     ["ground.delta_r_ohm", "ground.fast_write_time_s"])
        assert out["parameters"]["ground.fast_write_time_s"]["identifiable"] is False
        assert out["status"] == "fitted_with_reservations"

    def test_current_and_summed_impedance_predictions(self):
        grid = F
        device = ss.StateSpaceDevice(one_port(capacitance_f=0.0))
        z = np.sum(np.abs(device.impedance_matrix(grid, device.initial_state())[:, 0, 0]))
        t = [1.0, 2.0, 3.0, 4.0, 5.0]
        record = {"provenance": SYNTHETIC,
                  "protocol": {"segments": [hold({"a": 0.0}, 5.0)]},
                  "datasets": [
                      {"kind": "summed_impedance", "to_port": "a", "from_port": "a",
                       "time_s": t, "zc_ohm": [float(z)] * 5, "sigma_zc_ohm": [1.0] * 5,
                       "frequency_grid_hz": grid, "definition": "sum_of_magnitudes"},
                      {"kind": "current", "port": "a", "time_s": t, "current_a": [0.0] * 5,
                       "sigma_current_a": [1e-6] * 5}]}
        rows = ex.compare(one_port(capacitance_f=0.0), record)["datasets"]
        assert rows[0]["max_abs_residual"] < 1e-9 and rows[1]["max_abs_residual"] == 0
        assert rows[1]["agrees"] is False  # an all-zero signal tests nothing

    @pytest.mark.parametrize("spoil", [
        lambda r: r.update(provenance={}),
        lambda r: r["datasets"][0].pop("sigma_z_real_ohm"),
        lambda r: r["datasets"][0].update(sigma_z_real_ohm=[0.0] * 45),
        lambda r: r.pop("protocol")])
    def test_measurements_need_source_uncertainty_and_protocol(self, spoil):
        record = synthetic_record(one_port(), PROTOCOL, TIMES)
        spoil(record)
        with pytest.raises(ValueError):
            ex.compare(one_port(), record)


def reference_records(model):
    """Synthetic data that follow the cited protocols (NOT the published measurements)."""
    sweep = ref.hysteresis_protocol("a", "b", iterations=1)
    pulses = {"segments": [s for d in (1.0, 10.0)
                           for s in ref.pulse_protocol("a", "b", d, 20.0)["segments"]]}
    days = {"segments": [hold({"a": 3.3, "b": 0.0}, 43_200.0),
                         hold({"a": -3.3, "b": 0.0}, 43_200.0)]}
    return [
        synthetic_record(model, sweep, [1, 30, 60, 90, 120, 152], ("a", "b"),
                         item="hysteresis"),
        synthetic_record(model, pulses, [0, 1, 10, 31, 41], ("a", "b"),
                         item="pulse_duration"),
        synthetic_record(model, days, [0, 100, 1000, 43_200, 86_400], ("a", "b"),
                         item="long_term_drift")]


class TestReferenceHarness:
    def test_without_data_nothing_is_reproduced(self):
        for model in (None, two_port(FATIGUE)):
            out = ref.reproduce(model, options={"iterations": 3, "frequency_points": 11,
                                                "pulse_durations_s": [0.5, 5.0]}
                                if model else None)
            assert out["status"] == "not_reproduced"
            assert {i["status"] for i in out["items"].values()} == {"reference_data_missing"}
            assert out["data_to_request"] and "2211.08152" in out["citation"]["preprint"]
            json.dumps(out, allow_nan=False)
        assert out["model_behaviour"]["hysteresis"]["loops"]["Z11"]["positive_loop_shrinks"]

    def test_full_agreement_needs_every_item(self):
        model = two_port(FATIGUE)
        records = reference_records(model)
        skip = {"skip_model_behaviour": True}
        full = ref.reproduce(model, records, skip)
        assert full["status"] == "reproduced_on_supplied_data", full["items"]
        part = ref.reproduce(model, records[:2], skip)
        assert part["status"] == "not_reproduced"
        assert ref.reproduce(None, records, skip)["items"]["hysteresis"]["status"] == (
            "model_parameters_missing")
        other = ref.reproduce(two_port(FATIGUE, {"delta_r_ohm": 300.0}), records, skip)
        assert other["items"]["hysteresis"]["status"] == "disagrees"

    def test_wrong_protocol_coverage_and_reused_data_do_not_count(self):
        model = two_port(FATIGUE)
        sweep, pulses, days = reference_records(model)
        skip = {"skip_model_behaviour": True}
        wrong = copy.deepcopy(pulses)
        for d in wrong["datasets"]:
            d["reference_item"] = "hysteresis"
        assert ref.reproduce(model, wrong, skip)["items"]["hysteresis"]["status"] == (
            "protocol_differs")
        thin = copy.deepcopy(sweep)
        thin["datasets"] = thin["datasets"][:1]
        assert ref.reproduce(model, thin, skip)["items"]["hysteresis"]["status"] == (
            "insufficient_coverage")
        fitted = ex.fit(model, sweep, ["ground.delta_r_ohm"])["model"]
        assert ref.reproduce(fitted, sweep, skip)["items"]["hysteresis"]["status"] == (
            "not_independent")


class TestToolsAndRegistry:
    def test_mcp_tool(self):
        tool = mcp_server.chiplab_ferrofluid_state_model
        assert any(p["required"] for p in tool("describe")["parameters"])
        out = tool("simulate", two_port(), {"time_s": [0, 1, 2],
                                            "voltages_v": {"a": [2, 0, 0], "b": [0, 0, 0]}},
                   options={"read_times_s": [2.0], "frequency_hz": F})
        assert len(out["reads"][0]["z"]["real_ohm"]) == 3
        assert tool("reproduce_reference")["status"] == "not_reproduced"
        for bad in (lambda: tool("simulate", two_port()), lambda: tool("unknown"),
                    lambda: tool("fit", two_port(), options={"typo": 1})):
            with pytest.raises(ValueError):
                bad()

    def test_state_model_cannot_become_a_chiplet(self):
        out = mcp_server.material_device_to_chiplet(
            {"technology": "ferrofluid", **two_port()}, {"n_ports": 2})
        assert out["accepted"] is False and out["chiplet"] is None

    def test_registry_entries_resolve(self):
        listed = model_registry.describe()
        assert {"ferrofluid", "memristor", "pcm", "spintronics", "photonics"} <= set(
            listed["technologies"])
        for entry in listed["models"]:
            assert entry["checks"]["entry_points_import"], entry["checks"]
            assert not entry["checks"]["missing_test_files"]
        assert model_registry.load("ferrofluid_electrical_statespace") is ss.simulate
        json.dumps(listed)

    def test_new_model_plugs_in_and_incomplete_one_is_refused(self, monkeypatch):
        monkeypatch.setattr(model_registry, "_REGISTRY", dict(model_registry._REGISTRY))
        base = model_registry.get("pcm_basic")
        fields = {**base.__dict__, "id": "new_material", "technology": "new"}
        model_registry.register(model_registry.ModelRecord(**fields))
        assert model_registry.describe("new")["models"][0]["id"] == "new_material"
        with pytest.raises(ValueError, match="already"):
            model_registry.register(model_registry.ModelRecord(**fields))
        with pytest.raises(ValueError, match="empty"):
            model_registry.register(model_registry.ModelRecord(
                **{**fields, "id": "x", "known_limitations": ()}))
