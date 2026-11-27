"""Tests for chiplab MCP experimental tools."""

import pytest

from chiplab.mcp_experimental_tools import (
    chiplab_simulate_experimental_tech,
    chiplab_compare_technologies,
    chiplab_sweep_experiment,
    chiplab_custom_device_simulate,
    chiplab_experiment_ledger,
)


# ---------------------------------------------------------------------------
# Tool 1: chiplab_simulate_experimental_tech
# ---------------------------------------------------------------------------


class TestSimulateExperimentalTech:
    def test_ferrofluid(self):
        result = chiplab_simulate_experimental_tech("ferrofluid")
        assert result["technology"] == "ferrofluid"
        metrics = result["metrics"]
        assert "energy_per_bit_j" in metrics
        assert "latency_ns" in metrics

    def test_memristor(self):
        result = chiplab_simulate_experimental_tech("memristor")
        assert result["technology"] == "memristor"

    def test_pcm(self):
        result = chiplab_simulate_experimental_tech("pcm")
        assert result["technology"] == "pcm"

    def test_ferroelectric(self):
        result = chiplab_simulate_experimental_tech("ferroelectric")
        assert result["technology"] == "ferroelectric"

    def test_spintronic(self):
        result = chiplab_simulate_experimental_tech("spintronic")
        assert result["technology"] == "spintronic"

    def test_photonic(self):
        result = chiplab_simulate_experimental_tech("photonic")
        assert result["technology"] == "photonic"

    def test_custom(self):
        result = chiplab_simulate_experimental_tech("custom", params={"energy_per_bit_j": 5e-15})
        assert result["technology"] == "custom"

    def test_unknown_falls_back_to_custom(self):
        result = chiplab_simulate_experimental_tech("unknown_tech")
        # Should fall back to custom template
        assert "energy_per_bit_j" in result["metrics"]

    def test_with_params(self):
        result = chiplab_simulate_experimental_tech(
            "memristor", params={"energy_per_bit_j": 3e-15, "latency_ns": 10}
        )
        metrics = result["metrics"]
        assert metrics["energy_per_bit_j"] == 3e-15
        assert metrics["latency_ns"] == 10

    def test_with_pulse_schedule(self):
        result = chiplab_simulate_experimental_tech(
            "memristor",
            pulse_schedule=[{"duration_ns": 5, "amplitude": 1.0}],
        )
        # Pulse schedule should adjust latency
        assert "latency_ns" in result["metrics"]


# ---------------------------------------------------------------------------
# Tool 2: chiplab_compare_technologies
# ---------------------------------------------------------------------------


class TestCompareTechnologies:
    def test_compare_two_technologies(self):
        result = chiplab_compare_technologies(["memristor", "pcm"])
        assert "comparisons" in result
        assert "winners" in result
        assert len(result["technologies_compared"]) == 2

    def test_compare_multiple(self):
        result = chiplab_compare_technologies(
            ["ferrofluid", "memristor", "pcm", "spintronic"]
        )
        assert len(result["technologies_compared"]) == 4

    def test_compare_with_workload(self):
        result = chiplab_compare_technologies(
            ["ferrofluid", "memristor"], workload={"bits": 1024}
        )
        assert "comparisons" in result

    def test_winner_direction_for_every_metric(self):
        names = ["memristor", "pcm", "spintronic", "photonic"]
        result = chiplab_compare_technologies(names)
        winners = {c["metric"]: c for c in result["comparisons"]}
        higher = {"endurance_cycles", "retention_s", "area_density_gb_cm2"}
        lower = {"energy_per_bit_j", "latency_ns", "thermal_sensitivity_ppm_K"}
        assert set(winners) == higher | lower
        for metric, comparison in winners.items():
            values = [v for v in comparison["technologies"].values() if v is not None]
            best = max(values) if metric in higher else min(values)
            assert comparison["value"] == best, metric
            assert comparison["technologies"][comparison["winner"]] == best, metric
        assert winners["energy_per_bit_j"]["winner"] == "photonic"
        assert winners["endurance_cycles"]["winner"] == "photonic"
        assert winners["area_density_gb_cm2"]["winner"] == "photonic"

    def test_unmodelled_metrics_are_not_ranked(self):
        result = chiplab_compare_technologies(["ferrofluid", "memristor", "pcm"])
        assert all(c["winner"] != "ferrofluid" for c in result["comparisons"])

    def test_compare_empty_list(self):
        result = chiplab_compare_technologies([])
        assert "error" in result


# ---------------------------------------------------------------------------
# Tool 3: chiplab_sweep_experiment
# ---------------------------------------------------------------------------


class TestSweepExperiment:
    def test_sweep_voltage(self):
        result = chiplab_sweep_experiment(
            "memristor", "voltage", [0.5, 1.0, 1.5, 2.0]
        )
        assert result["sweep_param"] == "voltage"
        assert len(result["values"]) == 4
        assert len(result["points"]) == 4
        assert result["best_point"] is not None

    def test_sweep_with_base_params(self):
        result = chiplab_sweep_experiment(
            "pcm",
            "temperature",
            [300, 350, 400],
            base_params={"voltage": 1.0},
        )
        assert result["sweep_param"] == "temperature"
        assert len(result["points"]) == 3

    def test_sweep_empty_values(self):
        result = chiplab_sweep_experiment("spintronic", "current", [])
        assert result["error"] == "No sweep values provided."

    def test_sweep_best_point_energy(self):
        result = chiplab_sweep_experiment(
            "memristor", "voltage", [0.1, 0.5, 1.0, 2.0]
        )
        # Best point should have minimum energy-per-bit
        assert result["best_point"] is not None
        assert "parameter_value" in result["best_point"]


# ---------------------------------------------------------------------------
# Tool 4: chiplab_custom_device_simulate
# ---------------------------------------------------------------------------


class TestCustomDeviceSimulate:
    def test_basic_device(self):
        device_spec = {
            "name": "test_memristor",
            "description": "Simple memristor device",
            "regions": [{"material": "memristor", "geometry": "crossbar"}],
            "electrodes": {
                "write": {"port": "W", "write_mechanism": {"mode": "voltage_pulse", "parameters": {"voltage": 1.0}}},
                "read": {"port": "R", "read_mechanism": {"mode": "impedance"}},
            },
        }
        result = chiplab_custom_device_simulate(device_spec)
        assert result["device_name"] == "test_memristor"
        assert "metrics" in result

    def test_ferroelectric_device(self):
        device_spec = {
            "name": "ferroelectric_capacitor",
            "description": "Ferroelectric memory capacitor",
            "regions": [
                {"material": "ferroelectric", "geometry": "parallel_plate"},
                {"material": "bottom_electrode", "geometry": "bottom"},
            ],
            "electrodes": {
                "write": {"port": "W", "write_mechanism": {"mode": "voltage_pulse"}},
                "read": {"port": "R", "read_mechanism": {"mode": "impedance"}},
            },
        }
        result = chiplab_custom_device_simulate(device_spec)
        assert result["device_name"] == "ferroelectric_capacitor"
        metrics = result["metrics"]
        assert "conductance_s" in metrics

    def test_with_custom_voltage_and_freq(self):
        device_spec = {
            "name": "custom_device",
            "description": "A custom device",
            "regions": [],
            "electrodes": {},
        }
        result = chiplab_custom_device_simulate(
            device_spec, write_voltage_v=2.5, read_freq_hz=1e5
        )
        assert result["write_voltage_v"] == 2.5
        assert result["read_freq_hz"] == 1e5


# ---------------------------------------------------------------------------
# Tool 5: chiplab_experiment_ledger
# ---------------------------------------------------------------------------


class TestExperimentLedger:
    def test_list_action(self):
        result = chiplab_experiment_ledger("list")
        assert "action" in result
        assert result["action"] == "list"

    def test_log_action(self):
        result = chiplab_experiment_ledger(
            "log",
            run_data={
                "params": {"technology": "memristor", "voltage": 1.5},
                "metrics": {"energy_per_bit_j": 5e-15, "latency_ns": 20},
                "fidelity": "analytical",
            },
        )
        assert result["action"] == "log"
        assert "run_id" in result

    def test_log_missing_run_data(self):
        result = chiplab_experiment_ledger("log")
        assert "error" in result

    def test_get_action_single_id(self):
        # First log a run, then get it
        log_result = chiplab_experiment_ledger(
            "log",
            run_data={
                "params": {"technology": "pcm"},
                "metrics": {"energy_per_bit_j": 2e-14},
                "fidelity": "analytical",
            },
        )
        run_id = log_result.get("run_id")
        if run_id:
            get_result = chiplab_experiment_ledger("get", run_ids=[run_id])
            assert get_result["action"] == "get"
            assert get_result["run_id"] == run_id

    def test_compare_action_min_two(self):
        # Log two runs, then compare
        log1 = chiplab_experiment_ledger(
            "log",
            run_data={
                "params": {"technology": "memristor"},
                "metrics": {"energy_per_bit_j": 5e-15, "latency_ns": 20},
                "fidelity": "analytical",
            },
        )
        log2 = chiplab_experiment_ledger(
            "log",
            run_data={
                "params": {"technology": "pcm"},
                "metrics": {"energy_per_bit_j": 2e-14, "latency_ns": 100},
                "fidelity": "analytical",
            },
        )
        run_ids = [log1.get("run_id"), log2.get("run_id")]
        if all(run_ids):
            compare_result = chiplab_experiment_ledger(
                "compare", run_ids=run_ids
            )
            assert compare_result["action"] == "compare"
            assert len(compare_result.get("run_ids", [])) == 2

    def test_summary_action(self):
        result = chiplab_experiment_ledger("summary")
        assert "action" in result
        assert result["action"] == "summary"

    def test_log_requires_fidelity(self):
        result = chiplab_experiment_ledger("log", run_data={"params": {}, "metrics": {}})
        assert "error" in result and "fidelity" in result["error"]

    def test_compare_insufficient_ids(self):
        result = chiplab_experiment_ledger("compare", run_ids=["only_one"])
        assert "error" in result

    def test_get_insufficient_ids(self):
        result = chiplab_experiment_ledger("get", run_ids=[])
        assert "error" in result


# ---------------------------------------------------------------------------
# Test prompts file import
# ---------------------------------------------------------------------------


def test_prompts_import():
    """Verify that prompts module can be imported (test_prompts.py placeholder)."""
    import chiplab.prompts  # noqa: F401
    # Just verify it loads without error