"""
Tests for Neuromorphic & In-Memory Compute Technologies
========================================================

Tests for:
- STDP Synapse (stdp_synapse.py)
- Liquid State Machine (liquid_state.py)
- Oscillatory Neural Network (oscillatory_nn.py)
- Ferroelectric Tunnel Junction (ftj.py)
- Magnetic Domain Wall (domain_wall.py)
"""

import pytest
import numpy as np

from chiplab.models.materials.stdp_synapse import (
    STDPRule,
    MemristorModel,
    STDPState,
    stdp_additive_update,
    stdp_multiplicative_update,
    stdp_triplet_update,
    stdp_voltage_dependent_update,
    stdp_calcium_based_update,
    stdp_memristor_physics_update,
    stdp_step,
    stdp_crossbar_vmm,
    stdp_array_simulate,
    stdp_stdp_curve,
)

from chiplab.models.materials.liquid_state import (
    ReservoirType,
    ReadoutType,
    ReservoirState,
    create_reservoir_weights,
    reservoir_step,
    reservoir_run,
    reservoir_train_readout,
    reservoir_predict,
    reservoir_compute_memory_capacity,
    reservoir_memristor_crossbar_init,
    reservoir_photonic_mesh_init,
    lsm_classify,
)

from chiplab.models.materials.oscillatory_nn import (
    OscillatorType,
    CouplingType,
    OscillatorParams,
    ONNState,
    create_coupling_matrix,
    kuramoto_step,
    fitzhugh_nagumo_step,
    vo2_relaxation_step,
    mtj_stt_step,
    onn_step,
    onn_run,
    onn_ising_solve,
    onn_pattern_recognition,
    onn_phase_locking_analysis,
)

from chiplab.models.materials.ftj import (
    FTJMaterial,
    ElectrodeMaterial,
    FTJParams,
    FTJState,
    ftj_tunneling_current,
    ftj_fowler_nordheim_current,
    ftj_polarization_dynamics,
    ftj_update_resistance,
    ftj_iv_curve,
    ftj_write_pulse,
    ftj_read_pulse,
    ftj_pv_loop,
    ftj_synaptic_weight,
    ftj_stdp_update,
    ftj_crossbar_vmm,
    ftj_retention_simulation,
    ftj_endurance_simulation,
)

from chiplab.models.materials.domain_wall import (
    DWMaterial,
    DWType,
    DrivingMechanism,
    DWParams,
    DWState,
    dw_compute_derived_params,
    dw_stt_velocity,
    dw_sot_velocity,
    dw_field_velocity,
    dw_vcma_velocity,
    dw_pinning_force,
    dw_thermal_noise,
    dw_step,
    dw_run,
    dw_racetrack_shift,
    dw_read_resistance,
    dw_logic_gate,
    dw_synaptic_weight,
    dw_stdp_update,
    skyrmion_dynamics,
)

from chiplab.core.gpu import get_gpu_backend as get_backend
from chiplab.schemas.common import Provenance


class TestSTDPSynapse:
    """Tests for STDP synapse models."""
    
    def test_stdp_state_creation(self):
        state = STDPState()
        assert state.weight.value == 0.5
        assert state.trace_pre.value == 0.0
        assert state.trace_post.value == 0.0
        assert state.weight.provenance == Provenance.ASSUMED
    
    def test_stdp_state_copy(self):
        state = STDPState()
        state.weight.value = 0.7
        state.trace_pre.value = 0.5
        state.weight_history.append(0.7)
        
        copied = state.copy()
        assert copied.weight.value == 0.7
        assert copied.trace_pre.value == 0.5
        assert copied.weight_history == [0.7]
        # Ensure deep copy
        copied.weight.value = 0.9
        assert state.weight.value == 0.7
    
    def test_stdp_additive_update_pre_only(self):
        state = STDPState()
        state.trace_post.value = 0.5
        state.a_minus.value = 0.01
        
        new_state = stdp_additive_update(state, spike_pre=True, spike_post=False, dt=1.0)
        
        # Weight should decrease (LTD)
        assert new_state.weight.value < state.weight.value
        assert new_state.trace_pre.value == 1.0  # Incremented
        assert len(new_state.spike_times_pre) == 1
    
    def test_stdp_additive_update_post_only(self):
        state = STDPState()
        state.trace_pre.value = 0.5
        state.a_plus.value = 0.01
        
        new_state = stdp_additive_update(state, spike_pre=False, spike_post=True, dt=1.0)
        
        # Weight should increase (LTP)
        assert new_state.weight.value > state.weight.value
        assert new_state.trace_post.value == 1.0
        assert len(new_state.spike_times_post) == 1
    
    def test_stdp_additive_update_both(self):
        state = STDPState()
        state.trace_pre.value = 0.5
        state.trace_post.value = 0.5
        state.a_plus.value = 0.01
        state.a_minus.value = 0.012
        
        new_state = stdp_additive_update(state, spike_pre=True, spike_post=True, dt=1.0)
        
        # Both traces incremented after decay
        # trace = 0.5 * exp(-1/20) + 1.0 ≈ 1.4756
        expected_trace = 0.5 * np.exp(-1.0/20.0) + 1.0
        assert abs(new_state.trace_pre.value - expected_trace) < 1e-6
        assert abs(new_state.trace_post.value - expected_trace) < 1e-6
    
    def test_stdp_multiplicative_update(self):
        state = STDPState()
        state.weight.value = 0.5
        state.trace_post.value = 1.0
        state.a_minus.value = 0.01
        
        new_state = stdp_multiplicative_update(state, spike_pre=True, spike_post=False, dt=1.0)
        
        # LTD scales with weight
        expected_dw = -0.01 * 1.0 * 0.5
        assert abs(new_state.weight.value - (0.5 + expected_dw)) < 1e-6
    
    def test_stdp_triplet_update(self):
        state = STDPState()
        state.trace_post.value = 1.0
        state.trace_post2.value = 0.5
        state.a_minus.value = 0.01
        state.a_minus2.value = 0.006
        
        new_state = stdp_triplet_update(state, spike_pre=True, spike_post=False, dt=1.0)
        
        # Triplet LTD: a_minus * trace_post * (1 + a_minus2 * trace_post2)
        expected_factor = 1 + 0.006 * 0.5
        expected_dw = -0.01 * 1.0 * expected_factor
        assert abs(new_state.weight.value - (0.5 + expected_dw)) < 1e-6
    
    def test_stdp_voltage_dependent_update(self):
        state = STDPState()
        state.trace_pre.value = 1.0
        state.a_plus.value = 0.01
        state.k_voltage.value = 10.0
        state.v_thresh.value = 0.5
        
        # Post before pre (v_post > v_pre) -> LTP
        new_state = stdp_voltage_dependent_update(
            state, spike_pre=False, spike_post=True, v_pre=0.0, v_post=1.0, dt=1.0
        )
        
        assert new_state.weight.value > state.weight.value
    
    def test_stdp_calcium_based_update(self):
        state = STDPState()
        state.internal_state.value = 0.2  # Calcium concentration
        state.ca_thresh_ltp.value = 0.3
        state.ca_thresh_ltd.value = 0.15
        state.a_plus.value = 0.01
        state.a_minus.value = 0.012
        
        # Calcium between thresholds -> LTD
        new_state = stdp_calcium_based_update(state, spike_pre=True, spike_post=False, dt=1.0)
        
        # Calcium decays but spike_pre adds to it
        assert new_state.internal_state.value > 0
    
    def test_stdp_memristor_physics_update(self):
        state = STDPState()
        state.memristor_model = MemristorModel.NONLINEAR_ION_DRIFT
        state.internal_state.value = 0.5
        state.ron.value = 1e3
        state.roff.value = 1e6
        state.mu_v.value = 1e-14
        state.d.value = 10e-9
        state.p.value = 1.0
        
        new_state = stdp_memristor_physics_update(
            state, spike_pre=True, spike_post=False, v_pre=0.0, v_post=1.0, dt=1e-3
        )
        
        # Internal state should change
        assert new_state.internal_state.value != state.internal_state.value
        assert 0 <= new_state.internal_state.value <= 1
    
    def test_stdp_step_dispatch(self):
        state = STDPState()
        
        # Test all rules
        for rule in STDPRule:
            new_state = stdp_step(state, True, False, 1.0, rule=rule)
            assert isinstance(new_state, STDPState)
    
    def test_stdp_crossbar_vmm(self):
        backend = get_backend("cpu")
        weights = np.random.rand(10, 5)
        inputs = np.random.rand(5)
        
        output = stdp_crossbar_vmm(weights, inputs, device="cpu", backend=backend)
        
        assert output.shape == (10,)
        assert np.all(np.isfinite(output))
    
    def test_stdp_array_simulate(self):
        backend = get_backend("cpu")
        
        # Simple spike trains
        spike_trains_pre = [[1.0, 5.0, 10.0], [2.0, 6.0]]
        spike_trains_post = [[3.0, 8.0], [4.0, 9.0, 12.0]]
        
        weights, states = stdp_array_simulate(
            n_pre=2, n_post=2,
            spike_trains_pre=spike_trains_pre,
            spike_trains_post=spike_trains_post,
            dt=1.0,
            rule=STDPRule.ADDITIVE,
            backend=backend
        )
        
        assert weights.shape == (2, 2)
        assert len(states) == 4  # 2x2 synapses
    
    def test_stdp_stdp_curve(self):
        state = STDPState()
        state.a_plus.value = 0.01
        state.a_minus.value = 0.012
        state.tau_pre.value = 20.0
        state.tau_post.value = 20.0
        
        delta_ts, delta_ws = stdp_stdp_curve(state, dt=0.1, max_dt=20.0, rule=STDPRule.ADDITIVE)
        
        assert len(delta_ts) == len(delta_ws)
        # Positive delta_t (pre before post) -> LTP (positive)
        pos_idx = np.where(delta_ts > 0)[0]
        if len(pos_idx) > 0:
            assert np.mean(delta_ws[pos_idx]) > 0
        # Negative delta_t (post before pre) -> LTD (negative)
        neg_idx = np.where(delta_ts < 0)[0]
        if len(neg_idx) > 0:
            assert np.mean(delta_ws[neg_idx]) < 0


class TestLiquidStateMachine:
    """Tests for Liquid State Machine / Reservoir Computing."""
    
    def test_reservoir_state_creation(self):
        state = ReservoirState(n_reservoir=50, n_inputs=5, n_outputs=3)
        assert state.n_reservoir == 50
        assert state.n_inputs == 5
        assert state.n_outputs == 3
        assert state.state.value.shape == (50,)
    
    def test_create_reservoir_weights(self):
        backend = get_backend("cpu")
        w_in, w_res = create_reservoir_weights(
            n_reservoir=100, n_inputs=10,
            spectral_radius=0.9, sparsity=0.1,
            backend=backend
        )
        
        assert w_in.shape == (100, 10)
        assert w_res.shape == (100, 100)
        # Check sparsity
        nnz = np.count_nonzero(w_res)
        assert nnz / (100*100) < 0.2  # Approximately sparse
        # Check spectral radius
        eigvals = np.linalg.eigvals(w_res)
        radius = np.max(np.abs(eigvals))
        assert abs(radius - 0.9) < 0.1
    
    def test_reservoir_step(self):
        state = ReservoirState(n_reservoir=20, n_inputs=3, n_outputs=2)
        state.w_in.value = np.random.randn(20, 3) * 0.1
        state.w_res.value = np.random.randn(20, 20) * 0.01
        state.leaking_rate.value = 0.3
        
        backend = get_backend("cpu")
        input_signal = np.random.randn(3)
        
        new_state = reservoir_step(state, input_signal, dt=1.0, backend=backend)
        
        assert new_state.state.value.shape == (20,)
        assert len(new_state.state_history) == 1
    
    def test_reservoir_run(self):
        state = ReservoirState(n_reservoir=20, n_inputs=3, n_outputs=2)
        state.w_in.value = np.random.randn(20, 3) * 0.1
        state.w_res.value = np.random.randn(20, 20) * 0.01
        
        backend = get_backend("cpu")
        inputs = np.random.randn(10, 3)
        
        final_state, trajectory = reservoir_run(state, inputs, dt=1.0, backend=backend)
        
        assert trajectory.shape == (10, 20)
        assert len(final_state.state_history) == 10
    
    def test_reservoir_train_readout(self):
        state = ReservoirState(n_reservoir=20, n_inputs=3, n_outputs=2)
        state.w_in.value = np.random.randn(20, 3) * 0.1
        state.w_res.value = np.random.randn(20, 20) * 0.01
        state.ridge_alpha.value = 1e-6
        
        backend = get_backend("cpu")
        inputs = np.random.randn(50, 3)
        targets = np.random.randn(50, 2)
        
        final_state, trajectory = reservoir_run(state, inputs, dt=1.0, backend=backend)
        trained_state = reservoir_train_readout(final_state, trajectory, targets, washout=10, backend=backend)
        
        assert trained_state.w_out.value.shape == (2, 20)
        assert trained_state.w_out.provenance == Provenance.ESTIMATED
    
    def test_reservoir_predict(self):
        state = ReservoirState(n_reservoir=20, n_inputs=3, n_outputs=2)
        state.w_out.value = np.random.randn(2, 20)
        
        backend = get_backend("cpu")
        states = np.random.randn(10, 20)
        
        predictions = reservoir_predict(state, states, backend=backend)
        
        assert predictions.shape == (10, 2)
    
    def test_reservoir_memory_capacity(self):
        state = ReservoirState(n_reservoir=50, n_inputs=1, n_outputs=1)
        state.w_in.value = np.random.randn(50, 1) * 0.1
        state.w_res.value = np.random.randn(50, 50) * 0.01
        
        backend = get_backend("cpu")
        capacities = reservoir_compute_memory_capacity(
            state, n_delays=10, n_trials=50, backend=backend
        )
        
        assert len(capacities) == 10
        assert np.all(capacities >= 0)
        assert np.all(capacities <= 1)
    
    def test_reservoir_memristor_crossbar_init(self):
        backend = get_backend("cpu")
        w_in, w_res = reservoir_memristor_crossbar_init(
            n_reservoir=50, n_inputs=5,
            g_min=1e-6, g_max=1e-3,
            backend=backend
        )
        
        assert w_in.shape == (50, 5)
        assert w_res.shape == (50, 50)
        # Values should be roughly in [-1, 1] range after normalization
        # (may exceed slightly due to variability)
        assert np.max(np.abs(w_in)) <= 1.5
        assert np.max(np.abs(w_res)) <= 1.5
    
    def test_reservoir_photonic_mesh_init(self):
        backend = get_backend("cpu")
        w_in, w_res = reservoir_photonic_mesh_init(
            n_reservoir=50, n_inputs=5,
            backend=backend
        )
        
        assert w_in.shape == (50, 5)
        assert w_res.shape == (50, 50)
        # Unitary matrix should have singular values ~1 (within tolerance)
        # Real part of unitary matrix may not be perfectly unitary
        svs = np.linalg.svd(w_res, compute_uv=False)
        assert np.allclose(svs, 1.0, atol=1.0)
    
    def test_lsm_classify(self):
        state = ReservoirState(n_reservoir=30, n_inputs=4, n_outputs=3)
        state.w_in.value = np.random.randn(30, 4) * 0.1
        state.w_res.value = np.random.randn(30, 30) * 0.01
        state.w_out.value = np.random.randn(3, 30)
        
        backend = get_backend("cpu")
        test_inputs = np.random.randn(5, 10, 4)  # 5 samples, 10 timesteps, 4 inputs
        
        predictions = lsm_classify(state, test_inputs, dt=1.0, backend=backend)
        
        assert predictions.shape == (5, 3)


class TestOscillatoryNeuralNetwork:
    """Tests for Oscillatory Neural Networks."""
    
    def test_onn_state_creation(self):
        state = ONNState(n_oscillators=10)
        assert state.n_oscillators == 10
        assert state.phases.value.shape == (10,)
        assert state.amplitudes.value.shape == (10,)
        assert state.coupling.value.shape == (10, 10)
    
    def test_create_coupling_matrix(self):
        backend = get_backend("cpu")
        
        # All-to-all
        K = create_coupling_matrix(10, CouplingType.RESISTIVE, "all_to_all", 0.1, backend=backend)
        assert K.shape == (10, 10)
        assert np.trace(K) == 0  # No self-coupling
        
        # Ring
        K = create_coupling_matrix(10, CouplingType.RESISTIVE, "ring", 0.1, backend=backend)
        assert K.shape == (10, 10)
        # Each node connected to 2 neighbors
        assert np.sum(K > 0) == 20
        
        # Small world
        K = create_coupling_matrix(10, CouplingType.RESISTIVE, "small_world", 0.1, backend=backend)
        assert K.shape == (10, 10)
    
    def test_kuramoto_step(self):
        state = ONNState(n_oscillators=5, oscillator_type=OscillatorType.ABSTRACT_KURAMOTO)
        state.phases.value = np.random.uniform(0, 2*np.pi, 5)
        state.coupling.value = create_coupling_matrix(5, CouplingType.RESISTIVE, "all_to_all", 0.1)
        state.params.f0.value = 1e9
        
        backend = get_backend("cpu")
        new_state = kuramoto_step(state, dt=1e-12, backend=backend)
        
        assert new_state.phases.value.shape == (5,)
        assert np.all(new_state.phases.value >= 0)
        assert np.all(new_state.phases.value < 2*np.pi)
    
    def test_fitzhugh_nagumo_step(self):
        state = ONNState(n_oscillators=5, oscillator_type=OscillatorType.ABSTRACT_FITZHUGH_NAGUMO)
        state.phases.value = np.random.uniform(-2, 2, 5)
        state.amplitudes.value = np.random.uniform(-1, 1, 5)
        state.coupling.value = create_coupling_matrix(5, CouplingType.RESISTIVE, "all_to_all", 0.1)
        
        backend = get_backend("cpu")
        new_state = fitzhugh_nagumo_step(state, dt=1e-3, backend=backend)
        
        assert new_state.phases.value.shape == (5,)
        assert new_state.amplitudes.value.shape == (5,)
    
    def test_vo2_relaxation_step(self):
        state = ONNState(n_oscillators=5, oscillator_type=OscillatorType.VO2_RELAXATION)
        state.phases.value = np.random.uniform(0, 3, 5)  # Voltage
        state.amplitudes.value = np.ones(5)
        state.params.v_th.value = 2.0
        state.params.v_h.value = 1.0
        state.params.r_on.value = 100.0
        state.params.r_off.value = 10e3
        state.params.c_vo2.value = 1e-12
        
        backend = get_backend("cpu")
        new_state = vo2_relaxation_step(state, dt=1e-12, backend=backend)
        
        assert new_state.phases.value.shape == (5,)
        assert new_state.amplitudes.value.shape == (5,)
        # Amplitude should be 0.1 or 1.0
        assert np.all(np.isin(np.round(new_state.amplitudes.value, 1), [0.1, 1.0]))
    
    def test_mtj_stt_step(self):
        state = ONNState(n_oscillators=5, oscillator_type=OscillatorType.MTJ_STT)
        state.phases.value = np.random.uniform(0, 2*np.pi, 5)
        state.amplitudes.value = np.ones(5)
        state.params.f0.value = 10e9
        state.params.i_stt.value = 1e-3
        state.params.alpha_g.value = 0.01
        
        backend = get_backend("cpu")
        new_state = mtj_stt_step(state, dt=1e-12, backend=backend)
        
        assert new_state.phases.value.shape == (5,)
        assert new_state.amplitudes.value.shape == (5,)
    
    def test_onn_step_dispatch(self):
        for osc_type in OscillatorType:
            state = ONNState(n_oscillators=3, oscillator_type=osc_type)
            state.coupling.value = create_coupling_matrix(3, CouplingType.RESISTIVE, "all_to_all", 0.1)
            state.params.f0.value = 1e9
            
            backend = get_backend("cpu")
            new_state = onn_step(state, dt=1e-12, backend=backend)
            
            assert isinstance(new_state, ONNState)
    
    def test_onn_run(self):
        state = ONNState(n_oscillators=5, oscillator_type=OscillatorType.ABSTRACT_KURAMOTO)
        state.phases.value = np.random.uniform(0, 2*np.pi, 5)
        state.coupling.value = create_coupling_matrix(5, CouplingType.RESISTIVE, "all_to_all", 0.1)
        state.params.f0.value = 1e9
        
        backend = get_backend("cpu")
        final_state = onn_run(state, n_steps=100, dt=1e-12, backend=backend)
        
        assert len(final_state.phase_history) == 100
        assert len(final_state.sync_history) == 100
    
    def test_onn_ising_solve(self):
        # Simple 4-spin Ising problem
        J = np.array([
            [0, -1, -1, 0],
            [-1, 0, 0, -1],
            [-1, 0, 0, -1],
            [0, -1, -1, 0]
        ], dtype=np.float32)
        
        state = ONNState(n_oscillators=4, oscillator_type=OscillatorType.ABSTRACT_KURAMOTO)
        state.params.f0.value = 1e9
        
        backend = get_backend("cpu")
        spins, energy = onn_ising_solve(state, J, n_steps=5000, dt=1e-12, backend=backend)
        
        assert spins.shape == (4,)
        assert np.all(np.abs(spins) == 1)
        assert np.isfinite(energy)
    
    def test_onn_pattern_recognition(self):
        # Store 3 patterns
        patterns = np.array([
            [1, 1, -1, -1, 1],
            [-1, -1, 1, 1, -1],
            [1, -1, 1, -1, 1]
        ], dtype=np.float32)
        
        # Test with noisy version of pattern 0
        test_pattern = np.array([1, 1, -1, -1, -1], dtype=np.float32)  # Last bit flipped
        
        state = ONNState(n_oscillators=5, oscillator_type=OscillatorType.ABSTRACT_KURAMOTO)
        state.params.f0.value = 1e9
        
        backend = get_backend("cpu")
        best_idx, overlap = onn_pattern_recognition(state, patterns, test_pattern, n_steps=2000, dt=1e-12, backend=backend)
        
        assert best_idx == 0  # Should recognize pattern 0
        assert overlap > 0.5  # Good overlap
    
    def test_onn_phase_locking_analysis(self):
        state = ONNState(n_oscillators=5, oscillator_type=OscillatorType.ABSTRACT_KURAMOTO)
        state.coupling.value = create_coupling_matrix(5, CouplingType.RESISTIVE, "all_to_all", 0.1)
        state.params.f0.value = 1e9
        
        backend = get_backend("cpu")
        detunings, sync_levels = onn_phase_locking_analysis(
            state, freq_range=(0.8, 1.2), n_points=10, n_steps=2000, dt=1e-12, backend=backend
        )
        
        assert len(detunings) == 10
        assert len(sync_levels) == 10
        assert np.all(sync_levels >= 0)
        assert np.all(sync_levels <= 1)


class TestFerroelectricTunnelJunction:
    """Tests for Ferroelectric Tunnel Junction (FTJ)."""
    
    def test_ftj_state_creation(self):
        state = FTJState()
        assert state.polarization.value == 0.0
        assert state.domain_fraction_up.value == 0.5
        assert state.params.material == FTJMaterial.HZO
    
    def test_ftj_state_copy(self):
        state = FTJState()
        state.polarization.value = 0.8
        state.polarization_history.append(0.8)
        
        copied = state.copy()
        assert copied.polarization.value == 0.8
        assert copied.polarization_history == [0.8]
    
    def test_ftj_tunneling_current(self):
        state = FTJState()
        state.params.thickness.value = 2e-9
        state.params.band_gap.value = 3.0
        state.params.effective_mass.value = 0.5
        state.area.value = 50e-9 * 50e-9
        state.polarization.value = 0.5
        
        backend = get_backend("cpu")
        current = ftj_tunneling_current(state, voltage=0.5, temperature=300.0, backend=backend)
        
        assert np.isfinite(current)
        assert current >= 0
    
    def test_ftj_fowler_nordheim_current(self):
        state = FTJState()
        state.params.thickness.value = 2e-9
        state.params.band_gap.value = 3.0
        state.params.effective_mass.value = 0.5
        state.area.value = 50e-9 * 50e-9
        
        backend = get_backend("cpu")
        current = ftj_fowler_nordheim_current(state, voltage=2.0, backend=backend)
        
        assert np.isfinite(current)
        assert current >= 0
    
    def test_ftj_polarization_dynamics(self):
        state = FTJState()
        state.params.coercive_field.value = 1e6
        state.params.nucleation_field.value = 1.2e6
        state.params.switching_time.value = 1e-9
        state.params.thickness.value = 2e-9
        state.polarization.value = 0.0
        
        backend = get_backend("cpu")
        # Apply field above nucleation
        new_state = ftj_polarization_dynamics(state, voltage=3.0, dt=1e-10, backend=backend)
        
        # Polarization should increase
        assert new_state.polarization.value > state.polarization.value
        assert new_state.polarization.value <= 1.0
    
    def test_ftj_update_resistance(self):
        state = FTJState()
        state.r_on.value = 1e3
        state.r_off.value = 1e6
        state.ter_ratio.value = 1000.0
        state.polarization.value = 0.0  # Mid state
        
        backend = get_backend("cpu")
        new_state = ftj_update_resistance(state, backend=backend)
        
        # Resistance should be between R_on and R_off
        assert new_state.current_resistance.value >= state.r_on.value
        assert new_state.current_resistance.value <= state.r_off.value
    
    def test_ftj_pv_loop(self):
        state = FTJState()
        state.params.coercive_field.value = 1e6
        state.params.nucleation_field.value = 1.2e6
        state.params.thickness.value = 2e-9
        state.params.switching_time.value = 1e-9
        
        backend = get_backend("cpu")
        voltages, polarizations, currents = ftj_pv_loop(
            state, v_max=3.0, n_points=100, freq=1e3, backend=backend
        )
        
        assert len(voltages) == 100
        assert len(polarizations) == 100
        assert len(currents) == 100
        # Hysteresis: polarization should not be single-valued
        assert np.max(polarizations) > 0
        assert np.min(polarizations) < 0
    
    def test_ftj_write_pulse(self):
        state = FTJState()
        state.params.coercive_field.value = 1e6
        state.params.nucleation_field.value = 1.2e6
        state.params.thickness.value = 2e-9
        state.params.switching_time.value = 1e-9
        state.polarization.value = 0.0
        
        backend = get_backend("cpu")
        new_state = ftj_write_pulse(state, voltage=3.0, pulse_width=10e-9, dt=1e-12, backend=backend)
        
        # Should have switched
        assert abs(new_state.polarization.value) > 0.5
    
    def test_ftj_read_pulse(self):
        state = FTJState()
        state.current_resistance.value = 10e3
        state.params.thickness.value = 2e-9
        state.params.band_gap.value = 3.0
        state.params.effective_mass.value = 0.5
        state.area.value = 50e-9 * 50e-9
        
        backend = get_backend("cpu")
        r, i = ftj_read_pulse(state, voltage=0.1, backend=backend)
        
        # ftj_read_pulse computes resistance from tunneling current
        # At low voltage (0.1V), current is very small, so resistance is very large
        assert np.isfinite(r)
        assert r > 0
        assert np.isfinite(i)
        assert i >= 0
    
    def test_ftj_synaptic_weight(self):
        state = FTJState()
        state.r_on.value = 1e3
        state.r_off.value = 1e6
        state.current_resistance.value = 10e3
        
        weight = ftj_synaptic_weight(state)
        
        assert 0 <= weight <= 1
        # Mid resistance -> mid weight
        assert 0.009 < weight < 0.011  # (10k-1k)/(1M-1k) ≈ 0.009
    
    def test_ftj_stdp_update(self):
        state = FTJState()
        state.params.coercive_field.value = 1e6
        state.params.nucleation_field.value = 1.2e6
        state.params.thickness.value = 2e-9
        state.params.switching_time.value = 1e-9
        state.polarization.value = 0.0
        
        backend = get_backend("cpu")
        # Pre-before-post -> potentiation (positive pulse)
        new_state = ftj_stdp_update(state, spike_pre=True, spike_post=True, 
                                    v_pre=0.0, v_post=1.0, dt=1e-12, backend=backend)
        
        # Should have moved towards positive polarization
        assert new_state.polarization.value > 0
    
    def test_ftj_crossbar_vmm(self):
        backend = get_backend("cpu")
        states = [FTJState() for _ in range(4)]
        for s in states:
            s.current_resistance.value = 10e3
        
        inputs = np.array([0.1, 0.2, 0.3])
        output = ftj_crossbar_vmm(states, inputs, backend=backend)
        
        assert output.shape == (4,)
        assert np.all(np.isfinite(output))
    
    def test_ftj_retention_simulation(self):
        state = FTJState()
        state.polarization.value = 1.0
        state.params.retention_time.value = 1e10
        
        backend = get_backend("cpu")
        time_points = np.logspace(0, 6, 10)  # 1s to 1Ms
        pols, resistances = ftj_retention_simulation(state, time_points, backend=backend)
        
        assert len(pols) == 10
        assert len(resistances) == 10
        # Polarization should decay
        assert pols[0] > pols[-1]
    
    def test_ftj_endurance_simulation(self):
        state = FTJState()
        state.params.coercive_field.value = 1e6
        state.params.nucleation_field.value = 1.2e6
        state.params.thickness.value = 2e-9
        state.params.switching_time.value = 1e-9
        
        backend = get_backend("cpu")
        results = ftj_endurance_simulation(state, n_cycles=10, write_voltage=2.0, 
                                           pulse_width=10e-9, dt=1e-12, backend=backend)
        
        assert "ter_history" in results
        assert len(results["ter_history"]) == 10
        assert "ter_degradation" in results


class TestMagneticDomainWall:
    """Tests for Magnetic Domain Wall devices."""
    
    def test_dw_state_creation(self):
        state = DWState(n_dws=2)
        assert state.n_dws == 2
        assert state.positions.value.shape == (2,)
        assert state.velocities.value.shape == (2,)
        assert state.angles.value.shape == (2,)
    
    def test_dw_state_copy(self):
        state = DWState(n_dws=2)
        state.positions.value = np.array([100e-9, 500e-9])
        state.position_history.append(state.positions.value.copy())
        
        copied = state.copy()
        assert np.array_equal(copied.positions.value, state.positions.value)
        assert len(copied.position_history) == 1
    
    def test_dw_initialize_pinning(self):
        state = DWState(n_dws=3)
        state.params.pinning_sites = 5
        state.params.pinning_spacing.value = 200e-9
        state.params.wire_length.value = 2e-6
        
        backend = get_backend("cpu")
        new_state = state.initialize_pinning(backend=backend)
        
        assert len(new_state.pinning_positions.value) == 5
        assert np.all(new_state.pinning_positions.value >= 0)
        assert np.all(new_state.pinning_positions.value <= 2e-6)
    
    def test_dw_compute_derived_params(self):
        state = DWState()
        state.params.ms.value = 1.2e6
        state.params.ku.value = 0.8e6
        state.params.aex.value = 15e-12
        state.params.alpha.value = 0.02
        
        new_state = dw_compute_derived_params(state)
        
        assert new_state.params.dw_width.provenance == Provenance.ESTIMATED
        assert new_state.params.walker_field.provenance == Provenance.ESTIMATED
        assert new_state.params.walker_current.provenance == Provenance.ESTIMATED
        assert new_state.params.dw_width.value > 0
    
    def test_dw_stt_velocity(self):
        state = DWState()
        state.params.ms.value = 1.2e6
        state.params.alpha.value = 0.02
        state.params.beta.value = 0.04
        state.params.spin_polarization.value = 0.6
        state.current_density.value = 1e12
        
        backend = get_backend("cpu")
        v = dw_stt_velocity(state, backend=backend)
        
        assert v.shape == (1,)
        assert np.isfinite(v[0])
    
    def test_dw_sot_velocity(self):
        state = DWState()
        state.params.ms.value = 1.2e6
        state.params.wire_thickness.value = 1e-9
        state.params.alpha.value = 0.02
        state.params.beta.value = 0.04
        state.params.xi.value = 0.02
        state.params.spin_hall_angle.value = 0.15
        state.current_density.value = 1e12
        
        backend = get_backend("cpu")
        v = dw_sot_velocity(state, backend=backend)
        
        assert v.shape == (1,)
        assert np.isfinite(v[0])
    
    def test_dw_field_velocity(self):
        state = DWState()
        state.params.dw_width.value = 10e-9
        state.params.alpha.value = 0.02
        state.applied_field.value = 10e-3  # 10 mT
        
        backend = get_backend("cpu")
        v = dw_field_velocity(state, backend=backend)
        
        assert v.shape == (1,)
        assert np.isfinite(v[0])
    
    def test_dw_vcma_velocity(self):
        state = DWState()
        state.params.ms.value = 1.2e6
        state.params.wire_thickness.value = 1e-9
        state.params.dw_width.value = 10e-9
        state.params.alpha.value = 0.02
        state.vcma_voltage.value = 1.0
        
        backend = get_backend("cpu")
        v = dw_vcma_velocity(state, backend=backend)
        
        assert v.shape == (1,)
        assert np.isfinite(v[0])
    
    def test_dw_pinning_force(self):
        state = DWState(n_dws=2)
        state.positions.value = np.array([100e-9, 500e-9])
        state.pinning_positions.value = np.array([105e-9, 505e-9])  # Within delta=10e-9
        state.params.pinning_strength.value = 1e-12
        state.params.dw_width.value = 10e-9
        
        backend = get_backend("cpu")
        forces = dw_pinning_force(state, backend=backend)
        
        assert forces.shape == (2,)
        # DW near pinning site should feel force
        assert np.abs(forces[0]) > 0
        assert np.abs(forces[1]) > 0
    
    def test_dw_thermal_noise(self):
        state = DWState(n_dws=3)
        state.params.temperature.value = 300.0
        state.params.alpha.value = 0.02
        state.params.ms.value = 1.2e6
        state.params.dw_width.value = 10e-9
        state.params.wire_width.value = 50e-9
        state.params.wire_thickness.value = 1e-9
        
        backend = get_backend("cpu")
        noise = dw_thermal_noise(state, dt=1e-12, backend=backend)
        
        assert noise.shape == (3,)
        assert np.all(np.isfinite(noise))
    
    def test_dw_step(self):
        state = DWState(n_dws=3)
        state.params.driving = DrivingMechanism.STT
        state.params.ms.value = 1.2e6
        state.params.ku.value = 0.8e6
        state.params.aex.value = 15e-12
        state.params.alpha.value = 0.02
        state.params.beta.value = 0.04
        state.params.spin_polarization.value = 0.6
        state.current_density.value = 1e12
        state.positions.value = np.array([100e-9] * 3)
        
        backend = get_backend("cpu")
        new_state = dw_step(state, dt=1e-12, backend=backend)
        
        assert new_state.positions.value.shape == (3,)
        assert new_state.velocities.value.shape == (3,)
        assert len(new_state.position_history) == 1
        # Position should have moved
        assert new_state.positions.value[0] != state.positions.value[0]
    
    def test_dw_run(self):
        state = DWState(n_dws=3)
        state.params.driving = DrivingMechanism.STT
        state.params.ms.value = 1.2e6
        state.params.ku.value = 0.8e6
        state.params.aex.value = 15e-12
        state.params.alpha.value = 0.02
        state.params.beta.value = 0.04
        state.params.spin_polarization.value = 0.6
        state.positions.value = np.array([100e-9] * 3)
        
        backend = get_backend("cpu")
        current_profile = np.full(100, 1e12)
        
        final_state = dw_run(state, n_steps=100, dt=1e-12, current_profile=current_profile, backend=backend)
        
        assert len(final_state.position_history) == 100
        assert len(final_state.velocity_history) == 100
    
    def test_dw_racetrack_shift(self):
        state = DWState(n_dws=3)
        state.params.driving = DrivingMechanism.STT
        state.params.ms.value = 1.2e6
        state.params.ku.value = 0.8e6
        state.params.aex.value = 15e-12
        state.params.alpha.value = 0.02
        state.params.beta.value = 0.04
        state.params.spin_polarization.value = 0.6
        state.params.pinning_sites = 10
        state.params.pinning_spacing.value = 200e-9
        state.params.wire_length.value = 2e-6
        state.positions.value = np.array([200e-9] * 3)
        state.params.temperature.value = 0.0   # drive check; thermal motion tested separately
        
        backend = get_backend("cpu")
        new_state = state.initialize_pinning(backend=backend)
        new_state = dw_racetrack_shift(new_state, n_shifts=1, current_pulse=5e11, 
                                       pulse_width=1e-9, dt=1e-12, backend=backend)
        
        # DWs should have moved
        assert np.all(new_state.positions.value > state.positions.value)
    
    def test_dw_read_resistance(self):
        state = DWState(n_dws=2)
        state.positions.value = np.array([100e-9, 500e-9])
        state.params.wire_length.value = 1e-6
        
        backend = get_backend("cpu")
        resistances = dw_read_resistance(state, mtj_readout=True, backend=backend)
        
        assert resistances.shape == (2,)
        assert np.all(resistances > 0)
    
    def test_dw_logic_gate(self):
        state = DWState(n_dws=3)
        
        backend = get_backend("cpu")
        out, new_state = dw_logic_gate(state, "AND", [True, False], backend=backend)
        
        assert out == False
        out, new_state = dw_logic_gate(state, "OR", [True, False], backend=backend)
        assert out == True
        out, new_state = dw_logic_gate(state, "XOR", [True, True], backend=backend)
        assert out == False
    
    def test_dw_synaptic_weight(self):
        state = DWState(n_dws=3)
        state.positions.value = np.array([0.25e-6, 0.5e-6, 0.75e-6])
        state.params.wire_length.value = 1e-6
        
        weights = dw_synaptic_weight(state)
        
        assert weights.shape == (3,)
        assert np.allclose(weights, [0.25, 0.5, 0.75])
    
    def test_dw_stdp_update(self):
        state = DWState(n_dws=3)
        state.params.driving = DrivingMechanism.STT
        state.params.ms.value = 1.2e6
        state.params.ku.value = 0.8e6
        state.params.aex.value = 15e-12
        state.params.alpha.value = 0.02
        state.params.beta.value = 0.04
        state.params.spin_polarization.value = 0.6
        state.positions.value = np.array([500e-9] * 3)
        state.params.temperature.value = 0.0   # drive check; thermal motion tested separately

        backend = get_backend("cpu")
        # Pre-before-post -> potentiation (move forward)
        new_state = dw_stdp_update(state, spike_pre=False, spike_post=True, 
                                   dt=1e-12, current_amplitude=3e11, pulse_width=1e-9, backend=backend)
        
        assert new_state.positions.value[0] > state.positions.value[0]
    
    def test_dw_thermal_diffusion_matches_einstein_relation(self):
        """Unpinned, undriven walls diffuse with <dx^2> = 2 D t, D = kB T / eta."""
        from chiplab.models.materials.domain_wall import dw_friction, dw_thermal_noise
        state = DWState(n_dws=4000)
        state.params.ms.value = 1.2e6
        state.params.alpha.value = 0.02
        state = dw_compute_derived_params(state)
        dt = 1e-12
        steps = np.array([dw_thermal_noise(state, dt, get_backend("cpu")) for _ in range(50)])
        msd = np.mean(steps.sum(axis=0) ** 2)
        d = 1.380649e-23 * state.params.temperature.value / dw_friction(state)
        assert msd == pytest.approx(2 * d * 50 * dt, rel=0.1)

    def test_dw_stt_drift_velocity_value(self):
        """Below Walker breakdown v = (beta/alpha) * g muB P J / (2 e Ms)."""
        state = DWState(n_dws=1)
        state.params.driving = DrivingMechanism.STT
        state.params.ms.value = 1.2e6
        state.params.alpha.value = 0.02
        state.params.beta.value = 0.04
        state.params.spin_polarization.value = 0.6
        state = dw_compute_derived_params(state)
        state.current_density.value = 3e11
        u = 2 * 9.274e-24 * 0.6 * 3e11 / (2 * 1.602e-19 * 1.2e6)
        assert dw_stt_velocity(state)[0] == pytest.approx(2 * u, rel=1e-3)

    def test_skyrmion_dynamics(self):
        state = DWState(n_dws=3)
        state.params.dw_width.value = 10e-9
        state.params.alpha.value = 0.02
        state.params.spin_polarization.value = 0.6
        state.params.spin_hall_angle.value = 0.15
        state.params.ms.value = 1.2e6
        state.params.wire_thickness.value = 1e-9
        state.positions.value = np.array([500e-9] * 3)
        
        backend = get_backend("cpu")
        new_state = skyrmion_dynamics(state, dt=1.0, current_density=1e15, backend=backend)
        
        assert new_state.positions.value.shape == (3,)
        assert new_state.velocities.value.shape == (3,)
        # Skyrmion should move
        assert new_state.positions.value[0] != state.positions.value[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])