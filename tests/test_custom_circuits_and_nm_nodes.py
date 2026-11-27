"""Tests for multi-nm process node scaling, custom circuits, 2D/beyond-CMOS devices, and MCP tools."""

from __future__ import annotations

import numpy as np
import pytest

from chiplab.mcp_experimental_tools import (
    chiplab_custom_circuit_simulate,
    chiplab_scale_technology_node,
    chiplab_simulate_experimental_tech,
)
from chiplab.models.materials.custom_circuit import (
    ComponentType,
    CustomCircuit,
    build_1t1r_cell,
    build_6t_sram_cell,
    build_cmos_inverter,
    build_hybrid_crossbar_column,
    build_ring_oscillator,
)
from chiplab.models.materials.graphene_pn import (
    GraphenePNParams,
    gnr_bandgap_ev,
    graphene_pn_iv,
    klein_tunneling_transmission,
)
from chiplab.models.materials.ncfet import (
    NCFETParams,
    ncfet_capacitance_matching,
    ncfet_iv,
)
from chiplab.models.materials.process_nodes import (
    ProcessCorner,
    TransistorArch,
    get_process_node,
    mosfet_iv,
    scale_experimental_device,
    sweep_technology_nodes,
)
from chiplab.models.materials.tfet import (
    TFETMaterial,
    TFETParams,
    tfet_btbt_iv,
    tfet_subthreshold_swing,
)
from chiplab.models.materials.tmd_fet import (
    TMDFETParams,
    TMDMaterial,
    tmd_fet_iv,
    tmd_quantum_capacitance,
    tmd_scale_sweep,
)


class TestProcessNodes:
    def test_node_ladder_and_interpolation(self):
        nodes = [180, 130, 65, 45, 28, 14, 7, 5, 3, 2, 1.4, 1.0]
        params = [get_process_node(nm) for nm in nodes]
        assert params[0].architecture == TransistorArch.PLANAR_BULK
        assert params[4].architecture == TransistorArch.PLANAR_HKMG
        assert params[6].architecture == TransistorArch.FINFET
        assert params[8].architecture == TransistorArch.GAA_NANOSHEET
        assert params[-1].architecture == TransistorArch.CFET
        # Vdd and SRAM area monotonically decrease with scaling
        assert params[0].vdd.value > params[4].vdd.value > params[8].vdd.value
        assert params[0].sram_bitcell_um2.value > params[6].sram_bitcell_um2.value

    def test_corners_and_temperature(self):
        tt = get_process_node(7.0, corner="TT", temperature_k=300.0)
        ff = get_process_node(7.0, corner="FF", temperature_k=300.0)
        ss = get_process_node(7.0, corner="SS", temperature_k=300.0)
        hot = get_process_node(7.0, corner="TT", temperature_k=375.0)
        assert ff.vth_n.value < tt.vth_n.value < ss.vth_n.value
        assert ff.fo4_delay_ps.value < tt.fo4_delay_ps.value < ss.fo4_delay_ps.value
        assert hot.leakage_nw_per_gate.value > tt.leakage_nw_per_gate.value

    def test_mosfet_iv_and_scaling(self):
        n28 = get_process_node(28.0)
        vgs = np.linspace(0.0, n28.vdd.value, 10)
        vds = np.full_like(vgs, n28.vdd.value)
        ids = mosfet_iv(n28, vgs, vds, device="auto")
        assert ids.shape == (10,)
        assert ids[-1] > ids[0] > 0.0

    def test_scale_experimental_and_sweep(self):
        sweep = sweep_technology_nodes([130, 65, 28, 7, 3, 2], technology="memristor")
        assert len(sweep["sweep"]) == 6
        # Density increases as nm shrinks
        assert sweep["sweep"][-1]["area_density_gb_cm2"] > sweep["sweep"][0]["area_density_gb_cm2"]
        scaled_3nm = scale_experimental_device("fefet", 3.0)
        assert scaled_3nm["node_nm"] == 3.0
        assert scaled_3nm["cell_area_um2"] > 0.0


class TestCustomCircuit:
    def test_cmos_inverter_dc_transient_and_node_compare(self):
        inv = build_cmos_inverter(node_nm=28.0)
        dc = inv.simulate_dc(device="auto")
        assert "out" in dc["node_voltages"]

        tran = inv.simulate_transient(t_stop=1.5e-9, dt=2e-11, device="auto")
        assert tran["avg_power_w"] > 0.0
        assert len(tran["waveforms"]["out"]) == len(tran["times_s"])
        # Inverter output should swing
        v_out = np.asarray(tran["waveforms"]["out"])
        assert np.max(v_out) - np.min(v_out) > 0.3

        cmp = inv.compare_across_nodes([65.0, 28.0, 7.0, 3.0], t_stop=1e-9, dt=5e-11)
        assert len(cmp["node_comparison"]) == 4

    def test_templates_rosc_sram_1t1r_crossbar(self):
        rosc = build_ring_oscillator(n_stages=5, node_nm=14.0)
        assert rosc.simulate_transient(t_stop=5e-10, dt=2e-11)["avg_power_w"] > 0.0

        sram = build_6t_sram_cell(node_nm=7.0)
        dc_sram = sram.simulate_dc()
        assert "q" in dc_sram["node_voltages"]

        cell_1t1r = build_1t1r_cell(node_nm=28.0, tech="memristor")
        res_1t1r = cell_1t1r.simulate_transient(t_stop=1e-9, dt=5e-11)
        assert "Dev" in res_1t1r["final_device_states"]

        xbar = build_hybrid_crossbar_column(n_rows=4, node_nm=5.0, tech="mtj")
        dc_xbar = xbar.simulate_dc()
        assert dc_xbar["static_power_w"] > 0.0

    def test_custom_mixed_circuit_and_spice_roundtrip(self):
        circ = CustomCircuit(name="hybrid_ncfet_mem", node_nm=5.0, corner="FF")
        vdd = circ.process_node.vdd.value
        circ.add_vsource("Vdd", "vdd", "0", dc=vdd)
        circ.add_vsource("Vin", "in", "0", waveform="pulse", v_low=0.0, v_high=vdd, period=1e-9)
        circ.add_emerging_fet("Mnc", ComponentType.NCFET, "mid", "in", "0")
        circ.add_memristor("Rmem", "vdd", "mid", r_on=2e3, r_off=5e4)
        circ.add_capacitor("Cout", "mid", "0", 2e-15)

        tran = circ.simulate_transient(t_stop=1e-9, dt=5e-11)
        assert tran["total_energy_j"] > 0.0

        mc = circ.simulate_monte_carlo(n_runs=5, t_stop=4e-10, dt=5e-11)
        assert mc["power_mean_w"] > 0.0

        spice = circ.to_spice_netlist()
        assert ".end" in spice
        parsed = CustomCircuit.from_spice_netlist(
            "V1 vdd 0 DC 0.9\nR1 vdd out 1000\nC1 out 0 1e-15\n.end",
            node_nm=28.0,
        )
        assert len(parsed.components) == 3
        assert parsed.simulate_dc()["node_voltages"]["out"] == pytest.approx(0.9, rel=1e-3)


class Test2DAndBeyondCMOS:
    def test_tmd_fet(self):
        p = TMDFETParams(material=TMDMaterial.MOS2)
        assert tmd_quantum_capacitance(p) > 0.0
        sweep = tmd_scale_sweep([10.0, 5.0, 3.0, 1.5], device="auto")
        assert len(sweep["sweep"]) == 4
        assert sweep["sweep"][0]["on_off_ratio"] > 10.0

    def test_ncfet(self):
        p = NCFETParams()
        match = ncfet_capacitance_matching(p)
        assert match["subthreshold_swing_mv_dec"] < 60.0
        ids = ncfet_iv(p, np.array([0.1, 0.5, 0.7]), np.array([0.7, 0.7, 0.7]), device="auto")
        assert ids[-1] > ids[0] > 0.0

    def test_tfet(self):
        p = TFETParams(material=TFETMaterial.INGAAS_GASB)
        ids = tfet_btbt_iv(p, np.linspace(0.0, 0.5, 6), np.full(6, 0.5), device="auto")
        assert ids[-1] > ids[0] > 0.0
        ss = tfet_subthreshold_swing(p)
        assert ss > 0.0

    def test_graphene_pn(self):
        p = GraphenePNParams()
        assert gnr_bandgap_ev(5.0) > gnr_bandgap_ev(20.0)
        trans = klein_tunneling_transmission(np.array([0.0, 0.3, 0.8]), 1e8, 10.0, device="auto")
        assert trans[0] == pytest.approx(1.0)
        assert trans[0] > trans[1] > trans[2]
        cur = graphene_pn_iv(p, np.array([0.1, 0.2]), v_split_gate=0.4, device="auto")
        assert cur[1] > cur[0] > 0.0


class TestMCPCustomCircuitAndNodes:
    def test_mcp_custom_circuit_simulate(self):
        res = chiplab_custom_circuit_simulate(
            template="cmos_inverter",
            node_nm=7.0,
            corner="TT",
            t_stop_s=6e-10,
            dt_s=3e-11,
            compare_nodes_nm=[65.0, 28.0, 7.0],
        )
        assert res["node_nm"] == 7.0
        assert res["transient"]["avg_power_w"] > 0.0
        assert len(res["node_comparison"]["node_comparison"]) == 3

    def test_mcp_scale_technology_node_and_sim(self):
        scaled = chiplab_scale_technology_node("cmos", nodes_nm=[130, 28, 7, 2])
        assert len(scaled["sweep"]) == 4
        mem_3nm = chiplab_simulate_experimental_tech("memristor", node_nm=3.0, corner="FF")
        assert mem_3nm["node_nm"] == 3.0
        assert mem_3nm["metrics"]["architecture"] == "gaa_nanosheet"
