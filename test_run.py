#!/usr/bin/env python
import sys
sys.path.insert(0, '/home/everett/Chip Lab')
from chiplab.mcp_experimental_tools import (
    chiplab_simulate_experimental_tech,
    chiplab_compare_technologies,
    chiplab_sweep_experiment,
    chiplab_custom_device_simulate,
    chiplab_experiment_ledger,
)

# Run all test functions from the test file
errors = []

# Test 1-10: TestSimulateExperimentalTech
try:
    r = chiplab_simulate_experimental_tech("ferrofluid")
    assert r["technology"] == "ferrofluid", f"Expected ferrofluid, got {r['technology']}"
    assert "energy_per_bit_j" in r["metrics"]
    assert "latency_ns" in r["metrics"]
    print("PASS: test_ferrofluid")
except AssertionError as e:
    errors.append(f"test_ferrofluid: {e}")
    print(f"FAIL: test_ferrofluid: {e}")

try:
    r = chiplab_simulate_experimental_tech("memristor")
    assert r["technology"] == "memristor"
    print("PASS: test_memristor")
except AssertionError as e:
    errors.append(f"test_memristor: {e}")
    print(f"FAIL: test_memristor: {e}")

try:
    r = chiplab_simulate_experimental_tech("pcm")
    assert r["technology"] == "pcm"
    print("PASS: test_pcm")
except AssertionError as e:
    errors.append(f"test_pcm: {e}")
    print(f"FAIL: test_pcm: {e}")

try:
    r = chiplab_simulate_experimental_tech("ferroelectric")
    assert r["technology"] == "ferroelectric"
    print("PASS: test_ferroelectric")
except AssertionError as e:
    errors.append(f"test_ferroelectric: {e}")
    print(f"FAIL: test_ferroelectric: {e}")

try:
    r = chiplab_simulate_experimental_tech("spintronic")
    assert r["technology"] == "spintronic"
    print("PASS: test_spintronic")
except AssertionError as e:
    errors.append(f"test_spintronic: {e}")
    print(f"FAIL: test_spintronic: {e}")

try:
    r = chiplab_simulate_experimental_tech("photonic")
    assert r["technology"] == "photonic"
    print("PASS: test_photonic")
except AssertionError as e:
    errors.append(f"test_photonic: {e}")
    print(f"FAIL: test_photonic: {e}")

try:
    r = chiplab_simulate_experimental_tech("custom", params={"energy_per_bit_j": 5e-15})
    assert r["technology"] == "custom"
    print("PASS: test_custom")
except AssertionError as e:
    errors.append(f"test_custom: {e}")
    print(f"FAIL: test_custom: {e}")

try:
    r = chiplab_simulate_experimental_tech("unknown_tech")
    assert "energy_per_bit_j" in r["metrics"]
    print("PASS: test_unknown_falls_back_to_custom")
except AssertionError as e:
    errors.append(f"test_unknown_falls_back_to_custom: {e}")
    print(f"FAIL: test_unknown_falls_back_to_custom: {e}")

try:
    r = chiplab_simulate_experimental_tech("memristor", params={"energy_per_bit_j": 3e-15, "latency_ns": 10})
    assert r["metrics"]["energy_per_bit_j"] == 3e-15
    assert r["metrics"]["latency_ns"] == 10
    print("PASS: test_with_params")
except AssertionError as e:
    errors.append(f"test_with_params: {e}")
    print(f"FAIL: test_with_params: {e}")

try:
    r = chiplab_simulate_experimental_tech("memristor", pulse_schedule=[{"duration_ns": 5, "amplitude": 1.0}])
    assert "latency_ns" in r["metrics"]
    print("PASS: test_with_pulse_schedule")
except AssertionError as e:
    errors.append(f"test_with_pulse_schedule: {e}")
    print(f"FAIL: test_with_pulse_schedule: {e}")

# Test 11-14: TestCompareTechnologies
try:
    r = chiplab_compare_technologies(["memristor", "pcm"])
    assert "comparisons" in r
    assert "winners" in r
    assert len(r["technologies_compared"]) == 2
    print("PASS: test_compare_two_technologies")
except AssertionError as e:
    errors.append(f"test_compare_two_technologies: {e}")
    print(f"FAIL: test_compare_two_technologies: {e}")

try:
    r = chiplab_compare_technologies(["ferrofluid", "memristor", "pcm", "spintronic"])
    assert len(r["technologies_compared"]) == 4
    print("PASS: test_compare_multiple")
except AssertionError as e:
    errors.append(f"test_compare_multiple: {e}")
    print(f"FAIL: test_compare_multiple: {e}")

try:
    r = chiplab_compare_technologies(["ferrofluid", "memristor"], workload={"bits": 1024})
    assert "comparisons" in r
    print("PASS: test_compare_with_workload")
except AssertionError as e:
    errors.append(f"test_compare_with_workload: {e}")
    print(f"FAIL: test_compare_with_workload: {e}")

try:
    r = chiplab_compare_technologies([])
    assert "error" in r
    print("PASS: test_compare_empty_list")
except AssertionError as e:
    errors.append(f"test_compare_empty_list: {e}")
    print(f"FAIL: test_compare_empty_list: {e}")

# Test 15-18: TestSweepExperiment
try:
    r = chiplab_sweep_experiment("memristor", "voltage", [0.5, 1.0, 1.5, 2.0])
    assert r["sweep_param"] == "voltage"
    assert len(r["values"]) == 4
    assert len(r["points"]) == 4
    assert r["best_point"] is not None
    print("PASS: test_sweep_voltage")
except AssertionError as e:
    errors.append(f"test_sweep_voltage: {e}")
    print(f"FAIL: test_sweep_voltage: {e}")

try:
    r = chiplab_sweep_experiment("pcm", "temperature", [300, 350, 400], base_params={"voltage": 1.0})
    assert r["sweep_param"] == "temperature"
    assert len(r["points"]) == 3
    print("PASS: test_sweep_with_base_params")
except AssertionError as e:
    errors.append(f"test_sweep_with_base_params: {e}")
    print(f"FAIL: test_sweep_with_base_params: {e}")

try:
    r = chiplab_sweep_experiment("spintronic", "current", [])
    assert r["error"] == "No sweep values provided."
    print("PASS: test_sweep_empty_values")
except AssertionError as e:
    errors.append(f"test_sweep_empty_values: {e}")
    print(f"FAIL: test_sweep_empty_values: {e}")

try:
    r = chiplab_sweep_experiment("memristor", "voltage", [0.1, 0.5, 1.0, 2.0])
    assert r["best_point"] is not None
    assert "parameter_value" in r["best_point"]
    print("PASS: test_sweep_best_point_energy")
except AssertionError as e:
    errors.append(f"test_sweep_best_point_energy: {e}")
    print(f"FAIL: test_sweep_best_point_energy: {e}")

# Test 19-21: TestCustomDeviceSimulate
try:
    device_spec = {
        "name": "test_memristor",
        "description": "Simple memristor device",
        "regions": [{"material": "memristor", "geometry": "crossbar"}],
        "electrodes": {
            "write": {"port": "W", "write_mechanism": {"mode": "voltage_pulse", "parameters": {"voltage": 1.0}}},
            "read": {"port": "R", "read_mechanism": {"mode": "impedance"}},
        },
    }
    r = chiplab_custom_device_simulate(device_spec)
    assert r["device_name"] == "test_memristor"
    assert "metrics" in r
    print("PASS: test_basic_device")
except AssertionError as e:
    errors.append(f"test_basic_device: {e}")
    print(f"FAIL: test_basic_device: {e}")

try:
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
    r = chiplab_custom_device_simulate(device_spec)
    assert r["device_name"] == "ferroelectric_capacitor"
    assert "conductance_s" in r["metrics"]
    print("PASS: test_ferroelectric_device")
except AssertionError as e:
    errors.append(f"test_ferroelectric_device: {e}")
    print(f"FAIL: test_ferroelectric_device: {e}")

try:
    device_spec = {
        "name": "custom_device",
        "description": "A custom device",
        "regions": [],
        "electrodes": {},
    }
    r = chiplab_custom_device_simulate(device_spec, write_voltage_v=2.5, read_freq_hz=1e5)
    assert r["write_voltage_v"] == 2.5
    assert r["read_freq_hz"] == 1e5
    print("PASS: test_with_custom_voltage_and_freq")
except AssertionError as e:
    errors.append(f"test_with_custom_voltage_and_freq: {e}")
    print(f"FAIL: test_with_custom_voltage_and_freq: {e}")

# Test 22-29: TestExperimentLedger
try:
    r = chiplab_experiment_ledger("list")
    assert "action" in r
    assert r["action"] == "list"
    print("PASS: test_list_action")
except AssertionError as e:
    errors.append(f"test_list_action: {e}")
    print(f"FAIL: test_list_action: {e}")

try:
    r = chiplab_experiment_ledger("log", run_data={"params": {"technology": "memristor", "voltage": 1.5}, "metrics": {"energy_per_bit_j": 5e-15, "latency_ns": 20}})
    assert r["action"] == "log"
    assert "run_id" in r
    print("PASS: test_log_action")
except AssertionError as e:
    errors.append(f"test_log_action: {e}")
    print(f"FAIL: test_log_action: {e}")

try:
    r = chiplab_experiment_ledger("log")
    assert "error" in r
    print("PASS: test_log_missing_run_data")
except AssertionError as e:
    errors.append(f"test_log_missing_run_data: {e}")
    print(f"FAIL: test_log_missing_run_data: {e}")

# Test get_action_single_id - needs a run_id from logging first
try:
    log_result = chiplab_experiment_ledger("log", run_data={"params": {"technology": "pcm"}, "metrics": {"energy_per_bit_j": 2e-14}})
    run_id = log_result.get("run_id")
    if run_id:
        get_result = chiplab_experiment_ledger("get", run_ids=[run_id])
        assert get_result["action"] == "get"
        assert get_result["run_id"] == run_id
        print("PASS: test_get_action_single_id")
    else:
        print("SKIP: test_get_action_single_id (no run_id)")
except AssertionError as e:
    errors.append(f"test_get_action_single_id: {e}")
    print(f"FAIL: test_get_action_single_id: {e}")

try:
    log1 = chiplab_experiment_ledger("log", run_data={"params": {"technology": "memristor"}, "metrics": {"energy_per_bit_j": 5e-15, "latency_ns": 20}})
    log2 = chiplab_experiment_ledger("log", run_data={"params": {"technology": "pcm"}, "metrics": {"energy_per_bit_j": 2e-14, "latency_ns": 100}})
    run_ids = [log1.get("run_id"), log2.get("run_id")]
    if all(run_ids):
        compare_result = chiplab_experiment_ledger("compare", run_ids=run_ids)
        assert compare_result["action"] == "compare"
        assert len(compare_result.get("run_ids", [])) == 2
        print("PASS: test_compare_action_min_two")
    else:
        print("SKIP: test_compare_action_min_two (no run_ids)")
except AssertionError as e:
    errors.append(f"test_compare_action_min_two: {e}")
    print(f"FAIL: test_compare_action_min_two: {e}")

try:
    r = chiplab_experiment_ledger("summary")
    assert "action" in r
    assert r["action"] == "summary"
    print("PASS: test_summary_action")
except AssertionError as e:
    errors.append(f"test_summary_action: {e}")
    print(f"FAIL: test_summary_action: {e}")

try:
    r = chiplab_experiment_ledger("compare", run_ids=["only_one"])
    assert "error" in r
    print("PASS: test_compare_insufficient_ids")
except AssertionError as e:
    errors.append(f"test_compare_insufficient_ids: {e}")
    print(f"FAIL: test_compare_insufficient_ids: {e}")

try:
    r = chiplab_experiment_ledger("get", run_ids=[])
    assert "error" in r
    print("PASS: test_get_insufficient_ids")
except AssertionError as e:
    errors.append(f"test_get_insufficient_ids: {e}")
    print(f"FAIL: test_get_insufficient_ids: {e}")

# Test 30: test_prompts_import
try:
    import chiplab.prompts
    print("PASS: test_prompts_import")
except Exception as e:
    errors.append(f"test_prompts_import: {e}")
    print(f"FAIL: test_prompts_import: {e}")

print()
if errors:
    print(f"{len(errors)} test(s) FAILED:")
    for e in errors:
        print(f"  - {e}")
else:
    print("ALL 30 TESTS PASSED!")