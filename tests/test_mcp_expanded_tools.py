"""Tests for the 6 expanded MCP tools in chiplab.mcp_experimental_tools."""

from __future__ import annotations

import json
import sys
import types

import pytest

from chiplab.mcp_experimental_tools import (
    chiplab_2d_beyond_cmos_simulate,
    chiplab_agent_run,
    chiplab_list_experimental_technologies,
    chiplab_list_process_nodes,
    chiplab_quantum_device_simulate,
    chiplab_spice_netlist_run,
)


def _assert_json_serializable(obj: dict) -> None:
    dumped = json.dumps(obj)
    assert isinstance(dumped, str) and len(dumped) > 0


class TestListProcessNodes:
    def test_list_default_tt_300k(self) -> None:
        res = chiplab_list_process_nodes()
        _assert_json_serializable(res)
        assert res["corner"] == "TT"
        assert res["temperature_k"] == 300.0
        assert len(res["pdk_nodes"]) >= 7
        assert len(res["interpolated_nodes"]) == 8
        assert res["total_count"] == len(res["pdk_nodes"]) + 8
        interp_nms = [n["node_nm"] for n in res["interpolated_nodes"]]
        for expected in (90.0, 45.0, 22.0, 14.0, 10.0, 5.0, 1.4, 1.0):
            assert expected in interp_nms

    def test_list_ff_corner_hot(self) -> None:
        res = chiplab_list_process_nodes(corner="FF", temperature_k=358.15)
        _assert_json_serializable(res)
        assert res["corner"] == "FF"
        assert res["pdk_nodes"][0]["corner"] == "FF"


class TestListExperimentalTechnologies:
    def test_categories_and_counts(self) -> None:
        res = chiplab_list_experimental_technologies()
        _assert_json_serializable(res)
        cats = res["categories"]
        for expected_cat in (
            "Neuromorphic/Memory",
            "Quantum/Cryogenic",
            "2D/Beyond-CMOS",
            "Custom Circuits & Process Nodes",
        ):
            assert expected_cat in cats
            assert len(cats[expected_cat]) >= 2
        assert res["total_technologies"] >= 20


class Test2DBeyondCMOSSimulate:
    @pytest.mark.parametrize("tech", ["tmd_fet", "ncfet", "tfet", "graphene_pn"])
    def test_all_2d_beyond_cmos_technologies(self, tech: str) -> None:
        res = chiplab_2d_beyond_cmos_simulate(
            technology=tech,
            node_nm=5.0,
            vdd_v=0.7,
            vgs_v=0.5,
            device="cpu",
        )
        _assert_json_serializable(res)
        assert "error" not in res
        assert res["ion_a"] > 0.0
        assert res["ioff_a"] > 0.0
        assert res["on_off_ratio"] >= 1.0
        assert res["subthreshold_swing_mv_dec"] > 0.0
        assert len(res["curve_samples"]) == 25
        assert "ids_at_vgs_a" in res

    def test_tmd_material_selection(self) -> None:
        res = chiplab_2d_beyond_cmos_simulate(
            technology="tmd_fet",
            node_nm=3.0,
            material="WSe2",
            device="cpu",
        )
        _assert_json_serializable(res)
        assert res["material"] == "WSe2"
        assert res["quantum_capacitance_f_cm2"] > 0.0

    def test_unknown_technology_returns_error(self) -> None:
        res = chiplab_2d_beyond_cmos_simulate(technology="unknown_fet")
        assert "error" in res


class TestQuantumDeviceSimulate:
    def test_transmon(self) -> None:
        res = chiplab_quantum_device_simulate("transmon", {"ej_ghz": 15.0, "ec_ghz": 0.25})
        _assert_json_serializable(res)
        assert res["technology"] == "transmon"
        assert len(res["energies_ghz"]) == 10
        assert res["freq_01_ghz"] > 0.0
        assert res["anharmonicity_mhz"] < 0.0

    def test_fluxonium(self) -> None:
        res = chiplab_quantum_device_simulate("fluxonium", {"n_levels": 6, "flux": 0.5})
        _assert_json_serializable(res)
        assert res["technology"] == "fluxonium"
        assert len(res["energies_ghz"]) == 6
        assert res["freq_01_ghz"] > 0.0

    def test_majorana(self) -> None:
        res = chiplab_quantum_device_simulate("majorana", {"b_field_t": 1.2, "wire_length_um": 2.0})
        _assert_json_serializable(res)
        assert res["technology"] == "majorana"
        assert res["topological_gap_mev"] >= 0.0
        assert res["wavefunction_overlap_mev"] >= 0.0

    def test_sfq_logic(self) -> None:
        res = chiplab_quantum_device_simulate(
            "sfq_logic",
            {"family": "RSFQ", "n_fluxons": 12, "frequency_hz": 20e9},
        )
        _assert_json_serializable(res)
        assert res["technology"] == "sfq_logic"
        assert res["total_power_w"] > 0.0
        assert res["fluxons_per_cycle"] == 12

    def test_adiabatic_qc(self) -> None:
        res = chiplab_quantum_device_simulate(
            "adiabatic_qc",
            {"n_qubits": 4, "n_samples": 40, "t_total_ns": 50.0},
        )
        _assert_json_serializable(res)
        assert res["technology"] == "adiabatic_qc"
        assert len(res["schedule_samples"]) == 5
        assert len(res["best_spins"]) == 4

    def test_unknown_quantum_technology(self) -> None:
        res = chiplab_quantum_device_simulate("unknown_qubit")
        assert "error" in res


class TestSpiceNetlistRun:
    def test_dc_resistor_divider(self) -> None:
        netlist = """
        * Simple resistive divider
        V1 in 0 DC 1.0
        R1 in out 1000
        R2 out 0 1000
        """
        res = chiplab_spice_netlist_run(netlist=netlist, node_nm=28.0, analysis="dc", device="cpu")
        _assert_json_serializable(res)
        assert "error" not in res
        assert res["component_count"] == 3
        assert "normalized_spice_netlist" in res
        assert "dc" in res

    def test_transient_rc_circuit(self) -> None:
        netlist = """
        V1 in 0 DC 1.0
        R1 in out 1000
        C1 out 0 1e-12
        """
        res = chiplab_spice_netlist_run(
            netlist=netlist,
            node_nm=14.0,
            analysis="transient",
            t_stop_s=2e-9,
            dt_s=1e-10,
            device="cpu",
        )
        _assert_json_serializable(res)
        assert "error" not in res
        assert "transient" in res
        assert "normalized_spice_netlist" in res

    def test_empty_netlist_error(self) -> None:
        res = chiplab_spice_netlist_run(netlist="   ")
        assert "error" in res


class TestAgentRun:
    def test_agent_run_with_mocked_openrouter_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeOpenRouterAgent:
            def __init__(self, model: str | None = None) -> None:
                self.model = model or "fake/model"

            def run(self, prompt: str) -> dict:
                return {"echo": prompt, "model": self.model}

        fake_mod = types.ModuleType("chiplab.agent")
        fake_mod.OpenRouterAgent = _FakeOpenRouterAgent  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "chiplab.agent", fake_mod)

        res = chiplab_agent_run(prompt="Simulate a 5nm TMD FET", model="anthropic/claude-3.7-sonnet")
        _assert_json_serializable(res)
        assert res["status"] == "ok"
        assert res["model"] == "anthropic/claude-3.7-sonnet"
        assert res["result"]["echo"] == "Simulate a 5nm TMD FET"

    def test_agent_run_empty_prompt(self) -> None:
        res = chiplab_agent_run(prompt="")
        assert "error" in res


class TestMCPServerRegistration:
    def test_new_tools_registered_on_server(self) -> None:
        from chiplab.mcp_server import create_mcp_server

        server = create_mcp_server()
        tool_names = set(server._tool_manager._tools.keys())
        for expected in (
            "chiplab_list_process_nodes",
            "chiplab_list_experimental_technologies",
            "chiplab_2d_beyond_cmos_simulate",
            "chiplab_quantum_device_simulate",
            "chiplab_spice_netlist_run",
            "chiplab_agent_run",
        ):
            assert expected in tool_names
