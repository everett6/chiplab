"""
Tests for Quantum & Cryogenic Technologies
===========================================

Tests for:
- Transmon Qubit (transmon.py) - basic tests
- Fluxonium Qubit (fluxonium.py) - basic tests
- Majorana Zero Modes (majorana.py)
- SFQ Logic (sfq_logic.py)
- Adiabatic Quantum Computing (adiabatic_qc.py)
"""

import pytest
import numpy as np

from chiplab.models.materials.transmon import (
    TransmonParams,
    TransmonState,
    TransmonType,
    transmon_hamiltonian,
    transmon_compute_spectrum,
)

from chiplab.models.materials.fluxonium import (
    FluxoniumParams,
    FluxoniumState,
    fluxonium_hamiltonian,
    fluxonium_spectrum_vs_flux,
)

from chiplab.models.materials.majorana import (
    MZMMaterial,
    WireGeometry,
    MZMParams,
    MZMState,
    mzm_topological_gap,
    mzm_wavefunction_overlap,
    mzm_zero_bias_conductance,
    mzm_braiding_operation,
    mzm_parity_measurement,
    mzm_quasiparticle_poisoning,
    mzm_majorana_transmon_coupling,
    mzm_kitaev_chain_spectrum,
    mzm_topo_phase_diagram,
    mzm_braiding_protocol,
    mzm_simulate_experiment,
)

from chiplab.models.materials.sfq_logic import (
    SFQFamily,
    JJModel,
    JJParams,
    SFQGateParams,
    SFQState,
    sfq_josephson_current,
    sfq_junction_voltage,
    sfq_rsj_step,
    sfq_jtl_step,
    sfq_splitter_step,
    sfq_merger_step,
    sfq_and_gate_step,
    sfq_or_gate_step,
    sfq_not_gate_step,
    sfq_dff_step,
    sfq_ndro_step,
    sfq_clock_generator,
    sfq_power_estimation,
    sfq_timing_analysis,
    sfq_simulate_circuit,
    sfq_ersfq_ac_power_step,
    sfq_rql_differential_step,
)

from chiplab.models.materials.adiabatic_qc import (
    AnnealingSchedule,
    QubitType,
    AQCParams,
    AQCState,
    aqc_ising_hamiltonian,
    aqc_driver_hamiltonian,
    aqc_annealing_schedule,
    aqc_time_evolution_step,
    aqc_exact_evolution_step,
    aqc_sqa_step,
    aqc_qaoa_step,
    aqc_qaoa_classical_step,
    aqc_embed_ising,
    aqc_reverse_annealing,
    aqc_sample_solutions,
    aqc_quantum_monte_carlo,
    aqc_adiabatic_condition,
)


class TestTransmon:
    """Tests for Transmon Qubit - basic functionality."""

    def test_transmon_params_creation(self):
        params = TransmonParams()
        assert params.ec.value > 0
        assert params.ej.value > 0
        assert params.ec.provenance is not None

    def test_transmon_state_creation(self):
        state = TransmonState()
        assert state.params.ec.value > 0
        assert state.n_levels == 4

    def test_transmon_hamiltonian(self):
        params = TransmonParams()
        params.ec.value = 0.3 * 6.626e-34  # GHz * h
        params.ej.value = 20.0 * 6.626e-34  # GHz * h
        params.phi_ext.value = 0.0

        eigvals, eigvecs = transmon_hamiltonian(params, n_levels=4, flux=0.0)

        assert len(eigvals) == 4
        assert eigvecs.shape[1] == 4

    def test_transmon_compute_spectrum(self):
        params = TransmonParams()
        params.ec.value = 0.3 * 6.626e-34
        params.ej.value = 20.0 * 6.626e-34
        params.phi_ext.value = 0.0

        eigvals, freq_10, anharm = transmon_compute_spectrum(params)

        assert len(eigvals) >= 4
        assert freq_10 > 0
        assert anharm < 0  # Negative anharmonicity for transmon


class TestFluxonium:
    """Tests for Fluxonium Qubit - basic functionality."""

    def test_fluxonium_params_creation(self):
        params = FluxoniumParams()
        assert params.ec.value > 0
        assert params.el.value > 0
        assert params.ej.value > 0

    def test_fluxonium_state_creation(self):
        state = FluxoniumState()
        assert state.params.ec.value > 0
        assert state.n_levels == 10

    def test_fluxonium_hamiltonian(self):
        params = FluxoniumParams()
        params.ec.value = 0.5 * 6.626e-34
        params.el.value = 0.5 * 6.626e-34
        params.ej.value = 5.0 * 6.626e-34
        params.phi_ext.value = 0.5

        eigvals, eigvecs = fluxonium_hamiltonian(params, n_levels=10, flux=0.5)

        assert len(eigvals) == 10
        assert eigvecs.shape == (1024, 10)

    def test_fluxonium_spectrum_vs_flux(self):
        params = FluxoniumParams()
        params.ec.value = 0.5 * 6.626e-34
        params.el.value = 0.5 * 6.626e-34
        params.ej.value = 5.0 * 6.626e-34

        flux_vals, eigvals = fluxonium_spectrum_vs_flux(params, n_levels=10, n_flux=51)

        assert len(flux_vals) == 51
        assert eigvals.shape == (51, 10)


class TestMajorana:
    """Tests for Majorana Zero Modes."""

    def test_mzm_params_creation(self):
        params = MZMParams()
        assert params.spin_orbit_energy.value > 0
        assert params.induced_gap.value > 0
        assert params.critical_field.value > 0

    def test_mzm_state_creation(self):
        state = MZMState()
        assert state.params.material == MZMMaterial.IN_AS_AL
        assert state.geometry == WireGeometry.STRAIGHT
        assert state.fermion_parity.value in [1.0, -1.0]

    def test_mzm_topological_gap(self):
        params = MZMParams()
        params.induced_gap.value = 0.2  # meV
        params.critical_field.value = 0.5  # T

        # Below critical field - trivial
        gap = mzm_topological_gap(params, b_field=0.3)
        assert gap == 0.0

        # Above critical field - topological
        gap = mzm_topological_gap(params, b_field=1.0)
        assert gap > 0
        assert gap <= params.induced_gap.value

    def test_mzm_wavefunction_overlap(self):
        params = MZMParams()
        params.induced_gap.value = 0.2
        params.critical_field.value = 0.5
        params.wire_length.value = 1.0  # um

        overlap = mzm_wavefunction_overlap(params, wire_length=1.0)

        assert overlap >= 0
        assert overlap < params.induced_gap.value

    def test_mzm_zero_bias_conductance(self):
        state = MZMState()
        state.params.tunnel_coupling_left.value = 0.01
        state.params.tunnel_coupling_right.value = 0.01
        state.params.temperature.value = 20.0  # mK
        state.params.induced_gap.value = 0.2

        g = mzm_zero_bias_conductance(state, voltage=0.0)

        assert 0 <= g <= 1.0

    def test_mzm_braiding_operation(self):
        state = MZMState()
        state.current_braiding_phase.value = 0.0

        # Braid 1-2
        new_state = mzm_braiding_operation(state, "braid_12")
        assert abs(new_state.current_braiding_phase.value - np.pi/2) < 1e-10

        # Braid 2-3
        new_state = mzm_braiding_operation(new_state, "braid_23")
        assert abs(new_state.current_braiding_phase.value - np.pi) < 1e-10

    def test_mzm_braiding_protocol(self):
        state = MZMState()

        # Hadamard protocol
        new_state = mzm_braiding_protocol(state, "hadamard")
        assert len(new_state.braiding_history) > 0

        # Phase protocol
        new_state = mzm_braiding_protocol(state, "phase")
        assert len(new_state.braiding_history) > 0

    def test_mzm_kitaev_chain_spectrum(self):
        params = MZMParams()
        params.chemical_potential.value = 0.0
        params.induced_gap.value = 0.2
        params.spin_orbit_energy.value = 0.5
        params.g_factor.value = 20.0
        params.b_field.value = 1.0

        eigvals, _ = mzm_kitaev_chain_spectrum(params, n_sites=20)

        assert len(eigvals) == 40  # 2 * n_sites (particle-hole)
        # Should have near-zero modes in topological phase
        # Use a larger threshold due to finite size effects
        zero_modes = np.sum(np.abs(eigvals) < 0.1)
        assert zero_modes >= 2

    def test_mzm_topo_phase_diagram(self):
        params = MZMParams()
        params.induced_gap.value = 0.2
        params.critical_field.value = 0.5

        b_fields = np.linspace(0, 1.5, 10)
        mu_values = np.linspace(-2, 2, 10)

        gap_map = mzm_topo_phase_diagram(params, b_fields, mu_values)

        assert gap_map.shape == (10, 10)
        # Gap should be zero below critical field
        assert np.all(gap_map[0, :] == 0)  # b=0

    def test_mzm_simulate_experiment(self):
        state = MZMState()
        state.params.induced_gap.value = 0.2
        state.params.critical_field.value = 0.5
        state.params.tunnel_coupling_left.value = 0.01
        state.params.tunnel_coupling_right.value = 0.01

        # Conductance vs B field
        result = mzm_simulate_experiment(state, "conductance_vs_b",
            {"b_fields": np.linspace(0, 1.5, 20)})

        assert "b_fields" in result
        assert "conductances" in result
        assert len(result["conductances"]) == 20


class TestSFQLogic:
    """Tests for SFQ Logic."""

    def test_jj_params_creation(self):
        params = JJParams()
        assert params.ic.value > 0
        assert params.rn.value > 0
        assert params.c.value > 0

    def test_sfq_state_creation(self):
        state = SFQState()
        assert state.family == SFQFamily.RSFQ
        assert state.phi0 == 2.067833848e-15

    def test_sfq_josephson_current(self):
        params = JJParams()
        params.ic.value = 100e-6

        i = sfq_josephson_current(np.pi/2, params)
        assert abs(i - params.ic.value) < 1e-12

        i = sfq_josephson_current(0, params)
        assert abs(i) < 1e-12

    def test_sfq_junction_voltage(self):
        params = JJParams()
        v = sfq_junction_voltage(0, 1e12, params)
        phi0 = 2.067833848e-15
        expected = phi0 / (2 * np.pi) * 1e12
        assert abs(v - expected) < 1e-18

    def test_sfq_rsj_step(self):
        state = SFQState()
        state.jj_params.ic.value = 100e-6
        state.jj_params.rn.value = 2.0

        new_state = sfq_rsj_step(state, "node1", 150e-6, 1e-12)

        assert "node1" in new_state.node_phases
        assert "node1" in new_state.node_voltages

    def test_sfq_jtl_step(self):
        state = SFQState()
        state.gate_params["JTL"] = SFQGateParams()
        state.gate_params["JTL"].l_jtl.value = 2e-12

        # Inject fluxon at input
        state.node_phases["in"] = 2 * np.pi

        new_state = sfq_jtl_step(state, "in", "out", 1e-12)

        # Fluxon should propagate
        assert "out" in new_state.node_phases

    def test_sfq_splitter_step(self):
        state = SFQState()
        state.node_phases["in"] = 2 * np.pi

        new_state = sfq_splitter_step(state, "in", ["out1", "out2"], 1e-12)

        # Both outputs should get fluxon
        assert new_state.node_phases.get("out1", 0) > np.pi
        assert new_state.node_phases.get("out2", 0) > np.pi

    def test_sfq_merger_step(self):
        state = SFQState()
        state.node_phases["in1"] = 2 * np.pi
        state.node_phases["in2"] = 0

        new_state = sfq_merger_step(state, ["in1", "in2"], "out", 1e-12)

        # Output should get fluxon (OR logic)
        assert new_state.node_phases.get("out", 0) > np.pi

    def test_sfq_and_gate_step(self):
        state = SFQState()
        state.node_phases["a"] = 2 * np.pi
        state.node_phases["b"] = 2 * np.pi

        new_state = sfq_and_gate_step(state, "a", "b", "out", 1e-12)

        # Both inputs have fluxon -> output gets fluxon
        assert new_state.logic_state.get("out", 0) == 1

        # Test with only one input
        state2 = SFQState()
        state2.node_phases["a"] = 2 * np.pi
        state2.node_phases["b"] = 0

        new_state2 = sfq_and_gate_step(state2, "a", "b", "out", 1e-12)
        assert new_state2.logic_state.get("out", 0) == 0

    def test_sfq_or_gate_step(self):
        state = SFQState()
        state.node_phases["a"] = 2 * np.pi
        state.node_phases["b"] = 0

        new_state = sfq_or_gate_step(state, "a", "b", "out", 1e-12)

        # One input has fluxon -> output gets fluxon
        assert new_state.logic_state.get("out", 0) == 1

    def test_sfq_not_gate_step(self):
        state = SFQState()
        state.gate_params["NOT"] = SFQGateParams()
        state.gate_params["NOT"].bias_current.value = 0.8

        new_state = sfq_not_gate_step(state, "in", "out", 1e-12)

    def test_sfq_dff_step(self):
        state = SFQState()
        state.node_phases["data"] = 2 * np.pi
        state.node_phases["clk"] = 2 * np.pi

        new_state = sfq_dff_step(state, "data", "clk", "q", "qn", 1e-12)

        # Clock edge with data=1 -> Q=1, Qn=0
        assert new_state.logic_state.get("q", 0) == 1
        assert new_state.logic_state.get("qn", 0) == 0

    def test_sfq_ndro_step(self):
        state = SFQState()
        state.node_phases["set"] = 2 * np.pi

        new_state = sfq_ndro_step(state, "set", "reset", "q", "qn", 1e-12)

        assert new_state.logic_state.get("q", 0) == 1
        assert new_state.logic_state.get("qn", 0) == 0

    def test_sfq_clock_generator(self):
        state = SFQState()
        state.clock_frequency.value = 10e9
        state.family = SFQFamily.ERSFQ

        new_state = sfq_clock_generator(state, 1e-12)

        # Phase should advance
        expected_phase = 2 * np.pi * 10e9 * 1e-12
        assert abs(new_state.clock_phase - expected_phase) < 1e-10

    def test_sfq_power_estimation(self):
        state = SFQState()
        state.family = SFQFamily.RSFQ
        state.jj_params.ic.value = 100e-6
        state.jj_params.rn.value = 2.0

        power = sfq_power_estimation(state, n_fluxons=1000, frequency=10e9)

        assert "static_power_w" in power
        assert "dynamic_power_w" in power
        assert "total_power_w" in power
        assert power["total_power_w"] > 0

    def test_sfq_timing_analysis(self):
        state = SFQState()
        state.gate_params["JTL"] = SFQGateParams()
        state.gate_params["JTL"].propagation_delay.value = 3e-12
        state.gate_params["JTL"].setup_time.value = 2e-12
        state.gate_params["JTL"].hold_time.value = 1e-12
        state.clock_frequency.value = 10e9

        timing = sfq_timing_analysis(state, ["JTL", "JTL", "JTL"])

        assert "total_delay_s" in timing
        assert "max_frequency_hz" in timing
        assert abs(timing["total_delay_s"] - 9e-12) < 1e-20

    def test_sfq_simulate_circuit(self):
        state = SFQState()
        state.family = SFQFamily.RSFQ

        # Simple circuit: input -> JTL -> output
        netlist = {
            "gates": {
                "jtl1": {"type": "JTL", "inputs": ["in"], "outputs": ["out"]}
            },
            "outputs": ["out"]
        }

        input_signals = {"in": np.array([1, 0, 1, 0, 1])}

        result = sfq_simulate_circuit(state, netlist, input_signals, 1e-12, 5)

        assert "output_traces" in result
        assert "out" in result["output_traces"]
        assert len(result["output_traces"]["out"]) == 5

    def test_sfq_ersfq_ac_power_step(self):
        state = SFQState()
        state.family = SFQFamily.ERSFQ

        new_state = sfq_ersfq_ac_power_step(state, "node1", 150e-6, 10e9, 1e-12)

        assert "node1" in new_state.node_phases

    def test_sfq_rql_differential_step(self):
        state = SFQState()
        state.family = SFQFamily.RQL
        state.node_phases["p"] = np.pi
        state.node_phases["n"] = 0

        new_state = sfq_rql_differential_step(state, "p", "n", 1e-12)

        # Differential fluxon should propagate
        assert new_state.fluxon_count >= 0


class TestAdiabaticQC:
    """Tests for Adiabatic Quantum Computing."""

    def test_aqc_params_creation(self):
        params = AQCParams()
        assert params.n_qubits == 100
        assert params.max_h_field.value > 0
        assert params.max_j_coupling.value > 0

    def test_aqc_state_creation(self):
        state = AQCState()
        assert state.params.n_qubits == 100
        assert state.anneal_fraction == 0.0

    def test_aqc_ising_hamiltonian(self):
        h = np.array([1.0, -1.0, 0.5])
        J = np.array([[0, 0.5, 0], [0.5, 0, -0.3], [0, -0.3, 0]])

        H = aqc_ising_hamiltonian(h, J)

        assert H.shape == (8, 8)  # 2^3

    def test_aqc_driver_hamiltonian(self):
        H = aqc_driver_hamiltonian(3)

        assert H.shape == (8, 8)

    def test_aqc_annealing_schedule(self):
        # Linear schedule
        a, b = aqc_annealing_schedule(0, 10, AnnealingSchedule.LINEAR)
        assert a == 1.0 and b == 0.0

        a, b = aqc_annealing_schedule(5, 10, AnnealingSchedule.LINEAR)
        assert a == 0.5 and b == 0.5

        a, b = aqc_annealing_schedule(10, 10, AnnealingSchedule.LINEAR)
        assert a == 0.0 and b == 1.0

    def test_aqc_time_evolution_step(self):
        state = AQCState()
        state.params.n_qubits = 4
        state.params.anneal_time.value = 20e-6
        state.h_bias = np.array([1.0, -1.0, 0.5, -0.5])
        state.j_coupling = np.zeros((4, 4))
        state.j_coupling[0, 1] = 0.5
        state.j_coupling[1, 0] = 0.5

        new_state = aqc_time_evolution_step(state, 1e-6)

        assert new_state.anneal_fraction > 0
        assert new_state.time_elapsed == 1e-6

    def test_aqc_qaoa_step(self):
        state = AQCState()
        state.params.n_qubits = 4
        state.h_bias = np.array([1.0, -1.0, 0.5, -0.5])
        state.j_coupling = np.zeros((4, 4))

        new_state = aqc_qaoa_step(state, gamma=0.5, beta=0.3)

        assert new_state.instantaneous_energy is not None

    def test_aqc_qaoa_classical_step(self):
        state = AQCState()
        state.params.n_qubits = 10
        state.h_bias = np.random.randn(10)
        state.j_coupling = np.zeros((10, 10))
        state.classical_spins = np.random.choice([-1, 1], size=10)

        new_state = aqc_qaoa_classical_step(state, gamma=0.5, beta=0.3)

        assert len(new_state.classical_spins) == 10

    def test_aqc_quantum_monte_carlo(self):
        h = np.array([1.0, -1.0])
        J = np.array([[0, 0.5], [0.5, 0]])

        spins = aqc_quantum_monte_carlo(h, J, n_slices=10, beta=10.0, gamma=1.0, n_sweeps=100)

        assert spins.shape == (10, 2)
        assert np.all(np.abs(spins) == 1)

    def test_aqc_sample_solutions(self):
        state = AQCState()
        state.params.n_qubits = 4
        state.h_bias = np.array([1.0, -1.0, 0.5, -0.5])
        state.j_coupling = np.zeros((4, 4))
        state.classical_spins = np.array([1, -1, 1, -1])

        result = aqc_sample_solutions(state, n_samples=100)

        assert "samples" in result
        assert "energies" in result
        assert "best_spins" in result
        assert "best_energy" in result
        assert len(result["samples"]) == 100

    def test_aqc_adiabatic_condition(self):
        state = AQCState()
        state.params.n_qubits = 4
        state.params.anneal_time.value = 20e-6
        state.params.schedule_type = AnnealingSchedule.LINEAR
        state.time_elapsed = 10e-6
        state.h_bias = np.array([1.0, -1.0, 0.5, -0.5])
        state.j_coupling = np.zeros((4, 4))

        result = aqc_adiabatic_condition(state)

        assert "gap" in result
        assert "matrix_element" in result
        assert "adiabatic_parameter" in result
        assert "is_adiabatic" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])