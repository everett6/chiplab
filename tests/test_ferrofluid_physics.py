"""Ferrofluid particle model against analytic limits and independent calculations."""

import asyncio

import numpy as np
import pytest

from chiplab import mcp_experimental_tools as experimental
from chiplab import mcp_server
from chiplab.models.materials.ferro_calibration import fit_parameters
from chiplab.models.materials.ferro_device import (
    CoilPort,
    FerrofluidDevice,
    FluidRegion,
    ReadEvent,
    WritePulse,
    mutual_inductance_h,
    port_field,
    simulate_sequence,
)
from chiplab.models.materials.ferro_particles import (
    KB,
    MU0,
    FerrofluidParams,
    ParticleEnsemble,
    brownian_field_factors,
    brownian_rotation_monte_carlo,
    langevin,
    langevin_derivative,
    log_sinhc,
)
from chiplab.models.materials.particle_dynamics import brownian_motion


def mono(**kw):
    values = {"size_sigma_ln": 0.0, "interaction_model": "none"}
    values.update(kw)
    return FerrofluidParams(**values)


def small_device(**kw):
    ports = [
        CoilPort("w0", "write", position_um=(0, 0, 0), radius_um=10, turns=5),
        CoilPort("w1", "write", position_um=(30, 0, 0), radius_um=10, turns=5),
        CoilPort("r0", "read", position_um=(15, 20, 0), radius_um=10),
    ]
    region = FluidRegion((15, 8, 6), (60, 44, 8), (20, 16, 5))
    return FerrofluidDevice(ports, kw.pop("params", None), region, **kw)


# -- particle physics ---------------------------------------------------------


class TestLangevin:
    def test_weak_and_strong_field_limits(self):
        assert langevin(1e-4) == pytest.approx(1e-4 / 3, rel=1e-6)
        assert langevin(200.0) == pytest.approx(1 - 1 / 200, rel=1e-9)
        assert langevin_derivative(0.0) == pytest.approx(1 / 3)
        assert langevin(-2.0) == pytest.approx(-langevin(2.0))

    def test_series_joins_closed_form(self):
        x = np.array([0.00999999, 0.01000001])
        for function in (langevin, langevin_derivative, log_sinhc):
            low, high = function(x)
            assert low == pytest.approx(high, rel=1e-5)

    def test_derivative_matches_finite_difference(self):
        x = np.array([0.3, 2.0, 15.0])
        numeric = (langevin(x + 1e-5) - langevin(x - 1e-5)) / 2e-5
        assert np.allclose(langevin_derivative(x), numeric, rtol=1e-7)

    def test_saturation_magnetization(self):
        p = mono(volume_fraction=0.02)
        ensemble = ParticleEnsemble(p)
        expected = 0.02 * p.domain_magnetization_a_m
        assert ensemble.magnetization(1e9) == pytest.approx(expected, rel=1e-3)
        assert ensemble.magnetization(0.0) == 0.0

    def test_initial_susceptibility_ideal_and_mean_field(self):
        p = mono(volume_fraction=0.05, core_diameter_nm=10.0)
        moment = p.domain_magnetization_a_m * np.pi * (10e-9) ** 3 / 6
        chi_l = MU0 * moment * p.domain_magnetization_a_m * 0.05 / (3 * KB * 300)
        assert ParticleEnsemble(p).magnetization(1.0) == pytest.approx(chi_l, rel=1e-5)
        interacting = mono(volume_fraction=0.05, interaction_model="mmf1")
        value = ParticleEnsemble(interacting).magnetization(1.0)
        assert value == pytest.approx(chi_l * (1 + chi_l / 3), rel=1e-5)

    def test_size_quadrature_matches_lognormal_moment(self):
        sigma = 0.3
        p = FerrofluidParams(size_sigma_ln=sigma, size_bins=20)
        ensemble = ParticleEnsemble(p)
        # volume-weighted mean volume = <V^2>/<V> of the number distribution
        v0 = np.pi * (10e-9) ** 3 / 6
        expected = v0 * np.exp((36 - 9) * sigma**2 / 2)
        assert np.sum(ensemble.volume_weight * ensemble.core_volume) == pytest.approx(
            expected, rel=1e-8)


class TestRelaxationTimes:
    def test_brownian_and_neel_values(self):
        p = mono(core_diameter_nm=10, coating_nm=2, viscosity_pa_s=1e-3,
                 anisotropy_j_m3=2e4, neel_attempt_time_s=1e-9)
        e = ParticleEnsemble(p)
        kt = KB * 300
        tau_b = 3 * 1e-3 * (np.pi * (14e-9) ** 3 / 6) / kt
        tau_n = 1e-9 * np.exp(2e4 * np.pi * (10e-9) ** 3 / 6 / kt)
        assert e.tau_brown[0] == pytest.approx(tau_b, rel=1e-12)
        assert e.tau_neel[0] == pytest.approx(tau_n, rel=1e-12)
        assert e.tau_zero_field[0] == pytest.approx(tau_b * tau_n / (tau_b + tau_n))
        assert 1e-7 < tau_b < 1e-5  # microsecond scale for a 14 nm particle in water

    def test_faster_channel_dominates(self):
        large = ParticleEnsemble(mono(core_diameter_nm=30))  # Neel blocked
        small = ParticleEnsemble(mono(core_diameter_nm=6))  # Neel fast
        assert large.tau_zero_field[0] == pytest.approx(large.tau_brown[0], rel=1e-3)
        assert small.tau_zero_field[0] == pytest.approx(small.tau_neel[0], rel=0.05)

    def test_immobilised_particles_only_relax_by_neel(self):
        e = ParticleEnsemble(mono(core_diameter_nm=25, carrier_mobile=False))
        assert np.isinf(e.tau_brown[0]) and e.diffusivity[0] == 0
        assert e.tau_zero_field[0] == e.tau_neel[0]

    def test_field_dependent_limits(self):
        par, perp = brownian_field_factors(np.array([0.0, 1e-3, 500.0]))
        assert par[0] == perp[0] == 1.0
        assert par[1] == pytest.approx(1.0, abs=1e-6)
        assert par[2] * 500 == pytest.approx(1.0, rel=0.01)
        assert perp[2] * 500 == pytest.approx(2.0, rel=0.01)

    def test_debye_loss_peak_and_limits(self):
        e = ParticleEnsemble(mono())
        f_peak = 1 / (2 * np.pi * e.tau_zero_field[0])
        f = np.array([1e-3, f_peak / 3, f_peak, 3 * f_peak, 1e6 * f_peak])
        chi, perp = e.complex_susceptibility(f)
        loss = -chi.imag
        assert np.all(loss >= 0) and loss.argmax() == 2
        assert chi[2].real == pytest.approx(chi[0].real / 2, rel=1e-6)
        assert abs(chi[-1]) < 1e-5 * abs(chi[0])
        assert np.allclose(chi, perp)

    def test_static_bias_susceptibility_is_the_slope_of_m(self):
        e = ParticleEnsemble(FerrofluidParams(volume_fraction=0.05))
        h = 2e4
        slope = (e.magnetization(h + 1) - e.magnetization(h - 1)) / 2
        par, perp = e.complex_susceptibility(1e-6, h)
        assert par[0].real == pytest.approx(slope, rel=1e-6)
        assert perp[0].real == pytest.approx(e.magnetization(h) / h, rel=1e-6)
        assert par[0].real < perp[0].real < e.complex_susceptibility(1e-6)[0][0].real

    def test_remanence_of_one_size_is_exponential(self):
        e = ParticleEnsemble(mono())
        tau = e.tau_zero_field[0]
        t = np.array([0.0, tau, 3 * tau])
        assert np.allclose(e.remanence(t, 1e4), np.exp(-t / tau))


class TestParticleResolved:
    """Individual stochastic particles, independent of the reduced-order equations."""

    @pytest.mark.parametrize("xi", [0.5, 2.0, 5.0])
    def test_equilibrium_reaches_langevin(self, xi):
        run = brownian_rotation_monte_carlo(xi, 6.0, n_particles=20000, n_steps=600, seed=1)
        assert run["mean_alignment"][-1] == pytest.approx(float(langevin(xi)), abs=0.015)

    def test_zero_field_decay_follows_brownian_time(self):
        run = brownian_rotation_monte_carlo(0.0, 2.0, n_particles=20000, n_steps=400,
                                            start="aligned", seed=2)
        expected = np.exp(-run["time_over_tau_b"])
        assert np.max(np.abs(run["mean_alignment"] - expected)) < 0.02

    def test_translational_mean_square_displacement(self):
        np.random.seed(3)
        particles = np.zeros((40000, 6))
        moved = brownian_motion(particles, 1e-3, 300.0)
        d = 1.38e-23 * 300 / (3 * np.pi * 1e-3 * 10e-9)
        assert np.mean(np.sum(moved[:, :3] ** 2, axis=1)) == pytest.approx(6 * d * 1e-3,
                                                                          rel=0.02)

    def test_magnetophoresis_conserves_particles_and_follows_boltzmann(self):
        e = ParticleEnsemble(FerrofluidParams())
        field = np.array([0.0, 1e3, 5e3])  # sinh(xi) stays finite for all sizes
        volume = np.array([1.0, 2.0, 3.0])
        u = e.concentration_equilibrium(field, volume)
        assert np.allclose(volume @ u, 0.0, atol=1e-12)
        xi = field[:, None] * e.xi_per_field
        ratio = (1 + u[2]) / (1 + u[0])
        assert np.allclose(ratio, np.sinh(xi[2]) / xi[2])
        assert np.allclose(e.concentration_equilibrium(np.zeros(3), volume), 0.0)


# -- geometry and ports -------------------------------------------------------


class TestFields:
    def test_loop_on_axis(self):
        port = CoilPort("a", radius_um=20, turns=3)
        z = np.array([0.0, 15e-6, 60e-6])
        points = np.stack([np.zeros(3), np.zeros(3), z], axis=1)
        r = 20e-6
        exact = 3 * r**2 / (2 * (r**2 + z**2) ** 1.5)
        h = port_field(port, points)
        assert np.allclose(h[:, 2], exact, rtol=2e-3)
        assert np.allclose(h[:, :2], 0, atol=1e-3)

    def test_long_line_matches_ampere(self):
        port = CoilPort("l", conductor="line", start_um=(-5000, 0, 0), end_um=(5000, 0, 0))
        h = port_field(port, np.array([[0.0, 10e-6, 0.0]]))
        assert h[0, 2] == pytest.approx(1 / (2 * np.pi * 10e-6), rel=1e-4)

    def test_distant_coaxial_loops_match_dipole_mutual_inductance(self):
        a = CoilPort("a", radius_um=10)
        b = CoilPort("b", radius_um=15, position_um=(0, 0, 500))
        d = 500e-6
        dipole = MU0 * np.pi * (10e-6) ** 2 * (15e-6) ** 2 / (2 * d**3)
        assert mutual_inductance_h(a, b) == pytest.approx(dipole, rel=5e-3)

    def test_coupling_falls_with_separation(self):
        a = CoilPort("a", radius_um=10)
        near = CoilPort("b", radius_um=10, position_um=(30, 0, 0))
        far = CoilPort("c", radius_um=10, position_um=(90, 0, 0))
        assert abs(mutual_inductance_h(a, near)) > 10 * abs(mutual_inductance_h(a, far))


class TestImpedance:
    def test_passive_and_reciprocal_with_and_without_bias(self):
        device = small_device()
        freqs = np.logspace(2, 8, 13)
        state = device.initial_state()
        bias = device.current({"w0": 3.0, "w1": -2.0})
        state = device.evolve(state, bias, 1.0)
        for current in (np.zeros(3), bias):
            z = device.impedance_matrix(freqs, current, state)
            assert np.allclose(z, np.swapaxes(z, 1, 2))
            for matrix in z:
                hermitian = (matrix + matrix.conj().T) / 2
                assert np.linalg.eigvalsh(hermitian).min() > 0
            fluid = device.fluid_inductance(freqs, current, state.concentration)
            for k in range(len(freqs)):  # the fluid alone only dissipates
                assert np.linalg.eigvalsh(-fluid[k].imag).min() >= -1e-24

    def test_no_particles_gives_the_air_coil(self):
        device = small_device(params=FerrofluidParams(volume_fraction=0.0))
        z = device.impedance_matrix([1e6])[0]
        expected = np.diag(device.resistance) + 2j * np.pi * 1e6 * device.inductance_air
        assert np.allclose(z, expected, rtol=0, atol=1e-15)

    def test_fluid_term_scales_with_concentration_without_interaction(self):
        low = small_device(params=mono(volume_fraction=0.001))
        high = small_device(params=mono(volume_fraction=0.004))
        a = low.fluid_inductance([1e6], np.zeros(3))
        b = high.fluid_inductance([1e6], np.zeros(3))
        assert np.allclose(b, 4 * a)

    def test_grid_converges(self):
        assert small_device().grid_convergence() < 0.05

    def test_electrode_pair_obeys_rc_relation(self):
        ports = [CoilPort("a", electrode_radius_um=5),
                 CoilPort("b", position_um=(100, 0, 0), electrode_radius_um=5)]
        device = FerrofluidDevice(ports, conductivity_s_per_m=0.01, relative_permittivity=40)
        z = device.electrical_pair_impedance("a", "b", [1.0, 1e9])
        resistance = z[0].real
        height = device.region.size_um[2] * 1e-6
        assert resistance == pytest.approx(np.arccosh(100 / 10) / (np.pi * 0.01 * height),
                                           rel=1e-6)
        tau = 8.8541878128e-12 * 40 / 0.01  # RC = eps / sigma for a uniform medium
        assert z[1] == pytest.approx(resistance / (1 + 2j * np.pi * 1e9 * tau), rel=1e-6)

    def test_electrical_path_not_invented(self):
        assert small_device().electrical_pair_impedance("w0", "w1", [1e6]) is None


class TestDynamics:
    def test_long_pulse_reaches_equilibrium_then_decays(self):
        device = small_device(params=mono())
        current = device.current({"w0": 2.0})
        tau = device.ensemble.tau_zero_field[0]
        written = device.evolve(device.initial_state(), current, 50 * tau)
        h = device.field(current)
        magnitude = np.linalg.norm(h, axis=1)
        m_eq = device.ensemble.equilibrium(magnitude, written.concentration)[0][:, 0]
        assert np.allclose(np.linalg.norm(written.magnetization[:, 0], axis=1), m_eq,
                           rtol=1e-6)
        later = device.evolve(written, np.zeros(3), tau)
        assert np.allclose(device.flux_linkage(later),
                           device.flux_linkage(written) * np.exp(-1))

    def test_time_steps_compose(self):
        device = small_device(params=FerrofluidParams(carrier_mobile=False))
        current = device.current({"w0": 2.0, "w1": 1.0})
        whole = device.evolve(device.initial_state(), current, 3e-8)
        parts = device.initial_state()
        for _ in range(3):
            parts = device.evolve(parts, current, 1e-8)
        assert np.allclose(whole.magnetization, parts.magnetization)

    def test_emf_is_minus_flux_derivative(self):
        device = small_device()
        current = device.current({"w0": 2.0})
        state = device.evolve(device.initial_state(), current, 2e-8)
        dt = 1e-13
        after = device.evolve(state, current, dt)
        numeric = -(device.flux_linkage(after) - device.flux_linkage(state)) / dt
        assert np.allclose(device.emf(state, current), numeric, rtol=1e-3)

    def test_decay_time_in_sequence_matches_particle_time(self):
        device = small_device(params=mono())
        result = simulate_sequence(device, [WritePulse("w0", 1.0, 1e-5)], [ReadEvent("r0")])
        decay = result["retention"]["per_port"]["r0"]["time_to_1_over_e_s"]
        assert decay == pytest.approx(device.ensemble.tau_zero_field[0], rel=0.05)

    def test_concentration_memory_builds_and_relaxes(self):
        device = small_device()
        current = device.current({"w0": 10.0})
        tau = device.tau_concentration
        written = device.evolve(device.initial_state(), current, 10 * tau)
        peak = np.abs(written.concentration).max()
        assert peak > 0
        rested = device.evolve(written, np.zeros(3), tau)
        assert np.abs(rested.concentration).max() == pytest.approx(peak * np.exp(-1), rel=1e-6)


class TestPortInteraction:
    @staticmethod
    def residual(voltage):
        result = simulate_sequence(
            small_device(),
            [WritePulse("w0", voltage, 1e-5), WritePulse("w1", voltage, 1e-5)],
            [ReadEvent("r0", 1e6, time_s=9e-6)])
        return result["port_interaction"]["reads"][0]["flux"]["relative_to_combined"]

    def test_superposition_holds_weakly_and_breaks_strongly(self):
        weak, strong = self.residual(0.05), self.residual(20.0)
        assert weak < 1e-5
        assert strong > 100 * weak

    def test_ports_are_independent_and_signed(self):
        device = small_device()
        read = [ReadEvent("r0", 1e6, time_s=9e-6)]

        def flux(writes):
            return simulate_sequence(device, writes, read)["reads"][0][
                "magnetization_flux_linkage_wb"]

        first = flux([WritePulse("w0", 1.0, 1e-5)])
        assert flux([WritePulse("w0", -1.0, 1e-5)]) == pytest.approx(-first)
        assert flux([WritePulse("w0", 0.0, 1e-5)]) == 0.0
        delayed = flux([WritePulse("w0", 1.0, 1e-6, start_s=2e-5)])
        assert delayed == 0.0  # not yet written when read

    def test_weak_field_signal_is_quadratic_in_voltage(self):
        def signal(v):
            r = simulate_sequence(small_device(), [WritePulse("w0", v, 1e-5)],
                                  [ReadEvent("r0", 1e6, time_s=9e-6)])
            s = r["reads"][0]["signal_vs_unwritten_ohm"]
            return complex(s["real"], s["imag"])

        assert abs(signal(0.2)) / abs(signal(0.1)) == pytest.approx(4.0, rel=0.01)

    def test_role_and_overlap_checks(self):
        device = small_device()
        with pytest.raises(ValueError, match="read-only"):
            simulate_sequence(device, [WritePulse("r0", 1.0, 1e-6)], [])
        with pytest.raises(ValueError, match="overlapping"):
            simulate_sequence(device, [WritePulse("w0", 1.0, 1e-6),
                                       WritePulse("w0", 1.0, 1e-6, 5e-7)], [])
        with pytest.raises(ValueError, match="unknown port"):
            simulate_sequence(device, [WritePulse("nope", 1.0, 1e-6)], [])


# -- calibration ---------------------------------------------------------------


def synthetic_curve(noise=0.005, seed=0, **true):
    """Synthetic data generated from the model itself: tests the fit, not the physics."""
    ensemble = ParticleEnsemble(FerrofluidParams(**true))
    h = np.logspace(2, 6, 30)
    m = ensemble.magnetization(h)
    sigma = noise * m
    m = m + np.random.default_rng(seed).normal(0, sigma)
    return {"kind": "magnetization_curve", "H_a_per_m": h.tolist(), "M_a_per_m": m.tolist(),
            "sigma_M_a_per_m": sigma.tolist()}


SOURCE = {"source": "synthetic test data (not a measurement)"}


class TestCalibration:
    def test_recovers_known_parameters_with_uncertainty(self):
        data = synthetic_curve(volume_fraction=0.025, core_diameter_nm=11.5)
        fit = fit_parameters([data], ["volume_fraction", "core_diameter_nm"], SOURCE)
        assert fit["status"] == "fitted_to_user_data"
        assert fit["independently_validated"] is False
        for name, truth in (("volume_fraction", 0.025), ("core_diameter_nm", 11.5)):
            row = fit["parameters"][name]
            assert abs(row["value"] - truth) < 4 * row["standard_error"]
            assert 0 < row["standard_error"] < 0.05 * truth
        assert 0.4 < fit["fit_quality"]["reduced_chi_square"] < 2.0
        assert len(fit["provenance"]["data_sha256"]) == 64

    def test_requires_source_and_data(self):
        data = synthetic_curve(volume_fraction=0.02)
        with pytest.raises(ValueError, match="source"):
            fit_parameters([data], ["volume_fraction"], {})
        with pytest.raises(ValueError, match="no datasets"):
            fit_parameters([], ["volume_fraction"], SOURCE)

    def test_flags_parameters_the_data_cannot_determine(self):
        data = synthetic_curve(volume_fraction=0.02)
        fit = fit_parameters([data], ["volume_fraction", "viscosity_pa_s"], SOURCE)
        assert fit["parameters"]["viscosity_pa_s"]["identifiable"] is False
        assert fit["parameters"]["viscosity_pa_s"]["standard_error"] is None
        assert "viscosity_pa_s" not in fit["parameters_for_simulation"]
        assert fit["status"] == "fitted_with_reservations"

    def test_model_mismatch_is_reported(self):
        data = synthetic_curve(volume_fraction=0.02, core_diameter_nm=14)
        fit = fit_parameters([data], ["volume_fraction"], SOURCE,
                             fixed_parameters={"core_diameter_nm": 7})
        assert fit["status"] == "fitted_with_reservations"
        assert fit["fit_quality"]["reduced_chi_square"] > 10

    def test_spectrum_fit_and_reuse_in_simulation(self):
        truth = FerrofluidParams(viscosity_pa_s=6e-3, size_sigma_ln=0.0, core_diameter_nm=16)
        f = np.logspace(2, 7, 40)
        chi, _ = ParticleEnsemble(truth).complex_susceptibility(f)
        fit = mcp_server.calibrate_material_model(
            "ferrofluid",
            {"provenance": SOURCE,
             "fixed_parameters": {"size_sigma_ln": 0.0, "core_diameter_nm": 16},
             "datasets": [{"kind": "susceptibility_spectrum", "frequency_hz": f.tolist(),
                           "chi_real": chi.real.tolist(), "chi_loss": (-chi.imag).tolist()}]},
            ["viscosity_pa_s"])
        model = fit["calibrated_model"]
        assert model["parameters"]["viscosity_pa_s"]["value"] == pytest.approx(6e-3, rel=1e-3)
        result = experimental.chiplab_simulate_experimental_tech(
            "ferrofluid", {"calibration": model})
        rows = {r["name"]: r for r in result["simulation"]["parameters"]}
        assert rows["viscosity_pa_s"]["provenance"].startswith("fitted to user data")

    def test_other_technologies_do_not_claim_a_fit(self):
        out = mcp_server.calibrate_material_model("perovskite", {"mobility": [1.0, 3.0]},
                                                  ["mobility"])
        model = out["calibrated_model"]
        assert out["success"] is False and model["status"] == "not_calibrated"
        assert model["fit_quality"]["r_squared"] is None
        assert model["parameter_uncertainty"]["mobility"] is None


# -- MCP tools -----------------------------------------------------------------


class TestArrayHelper:
    @pytest.mark.parametrize("arrangement", ["linear", "grid", "grid_2x2", "grid_4x4",
                                             "circular", "hexagonal"])
    def test_every_arrangement_builds_distinct_positions(self, arrangement):
        """'circular' raised UnboundLocalError: math was imported only in the grid branch."""
        spec = mcp_server.configure_material_array("ferrofluid", 4, arrangement, 100.0)
        positions = {str(p["position"]) for p in spec["port_config"]}
        assert len(positions) == 4

    def test_circular_spacing(self):
        spec = mcp_server.configure_material_array("x", 6, "circular", 100.0)
        first = spec["port_config"][0]["position"]
        assert first["radius_um"] == pytest.approx(600 / (2 * np.pi))
        assert spec["total_length_um"] == pytest.approx(2 * first["radius_um"])

    def test_port_config_forms(self):
        for config in ({"1": {"role": "read"}}, {"port_1": {"role": "read"}},
                       [{}, {"role": "read"}]):
            spec = mcp_server.configure_material_array("ferrofluid", 2, "linear", 50, config)
            assert spec["port_config"][1]["role"] == "read"
            assert spec["port_config"][1]["impedance_ohm"] is None

    @pytest.mark.parametrize("args", [
        (4, "spiral", 100.0, None),
        (0, "linear", 100.0, None),
        (4, "linear", -1.0, None),
        (5, "grid_2x2", 100.0, None),
        (4, "linear", 100.0, {"0": 1.0}),
        (4, "linear", 100.0, {"9": {}}),
        (4, "linear", 100.0, {"0": {"radus_um": 5}}),
        (4, "custom", 100.0, None),
    ])
    def test_bad_input_gives_a_clear_value_error(self, args):
        n, arrangement, pitch, config = args
        with pytest.raises(ValueError):
            mcp_server.configure_material_array("ferrofluid", n, arrangement, pitch, config)


def array4():
    return mcp_server.configure_material_array(
        "ferrofluid", 4, "grid", 40.0,
        [{"role": "write", "radius_um": 10, "turns": 5}] * 2
        + [{"role": "read", "radius_um": 10}] * 2)


class TestMcpTools:
    def test_experimental_tool_responds_to_voltage_geometry_frequency(self):
        def metrics(**params):
            return experimental.chiplab_simulate_experimental_tech("ferrofluid", params)[
                "metrics"]

        runs = [metrics(voltage_v=v) for v in (0.2, 0.5, 1.0)]
        for key in ("write_current_a", "write_peak_field_a_m", "read_flux_linkage_wb"):
            values = [abs(r[key]) for r in runs]
            assert values[0] < values[1] < values[2]
        assert runs[2]["write_current_a"] == pytest.approx(5 * runs[0]["write_current_a"])
        base = metrics(voltage_v=1.0)
        for change in ({"radius_um": 10}, {"separation_um": 80}):
            assert metrics(voltage_v=1.0, **change)["read_flux_linkage_wb"] != \
                base["read_flux_linkage_wb"]
        assert metrics(read_frequency_hz=1e7)["read_impedance_magnitude_ohm"] > \
            metrics(read_frequency_hz=1e5)["read_impedance_magnitude_ohm"]

    def test_unmodelled_metrics_are_null_and_labelled(self):
        out = experimental.chiplab_simulate_experimental_tech("ferrofluid")
        for key in ("energy_per_bit_j", "latency_ns", "endurance_cycles", "retention_s",
                    "area_density_gb_cm2", "thermal_sensitivity_ppm_K"):
            assert out["metrics"][key] is None
            assert out["metric_status"][key] == "not_modelled"
        assert out["simulation"]["assumed_inputs"]

    def test_unknown_parameter_is_rejected(self):
        with pytest.raises(ValueError, match="unknown"):
            experimental.chiplab_simulate_experimental_tech("ferrofluid", {"voltge": 1})

    def test_other_technologies_are_labelled_heuristic(self):
        out = experimental.chiplab_simulate_experimental_tech("memristor")
        assert "assumed heuristic" in out["metric_status"]["retention_s"]

    def test_sweep_and_compare_tolerate_unmodelled_values(self):
        sweep = experimental.chiplab_sweep_experiment("ferrofluid", "voltage_v", [0.5, 1.0])
        assert sweep["best_point"] is None and len(sweep["points"]) == 2
        comparison = experimental.chiplab_compare_technologies(["ferrofluid", "memristor"])
        assert comparison["winners"]["latency_ns"] == "Insufficient data"

    def test_four_electrode_custom_device(self):
        spec = {"name": "4e", "regions": [{"material": "ferrofluid"}],
                "electrodes": {n: {"radius_um": 10} for n in ("w0", "w1", "r0", "r1")}}
        low = experimental.chiplab_custom_device_simulate(spec, write_voltage_v=0.2)
        high = experimental.chiplab_custom_device_simulate(spec, write_voltage_v=1.0)
        assert low["metrics"]["conductance_s"] is None
        flux = [r["metrics"]["reads"][0]["magnetization_flux_linkage_wb"] for r in (low, high)]
        assert abs(flux[1]) > 4 * abs(flux[0])
        assert "port_interaction" in high["simulation"]
        assert {r["port"] for r in high["metrics"]["reads"]} == {"r0", "r1"}

    def test_custom_device_electrical_values_need_user_properties(self):
        spec = {"regions": [{"material": "ferrofluid"}],
                "conductivity_s_per_m": 1e-3, "relative_permittivity": 10,
                "electrodes": {"w0": {"electrode_radius_um": 5},
                               "r0": {"electrode_radius_um": 5}}}
        one = experimental.chiplab_custom_device_simulate(spec)["metrics"]
        spec["conductivity_s_per_m"] = 2e-3
        two = experimental.chiplab_custom_device_simulate(spec)["metrics"]
        assert two["conductance_s"] == pytest.approx(2 * one["conductance_s"])
        assert two["capacitance_f"] == pytest.approx(one["capacitance_f"])

    def test_multiphysics_independent_ports(self):
        def run(writes):
            return mcp_server.run_multiphysics_simulation(
                {"technology": "ferrofluid"}, array4(), writes,
                [{"port": "port_2", "time_ns": 900}, {"port": "port_3", "time_ns": 900}])

        one = run([{"port": "port_0", "voltage_v": 1, "duration_ns": 1000}])
        two = run([{"port": "port_0", "voltage_v": 1, "duration_ns": 1000},
                   {"port": "port_1", "voltage_v": -2, "duration_ns": 500,
                    "start_ns": 500}])
        assert one["success"] and "port_interaction" in two
        assert [r["port"] for r in two["reads"]] == ["port_2", "port_3"]
        assert one["reads"][0]["magnetization_flux_linkage_wb"] != \
            two["reads"][0]["magnetization_flux_linkage_wb"]
        assert two["total_energy_pj"] == pytest.approx(
            sum(w["drive_energy_j"] for w in two["writes"]) * 1e12)
        assert len(two["impedance_spectrum"]) > 3

    def test_circuit_analyses(self):
        spec, array = {"technology": "ferrofluid"}, array4()
        dc = mcp_server.simulate_material_circuit(spec, array, {"voltage": 0.5}, {}, "dc")
        assert dc["current_a"] == pytest.approx(0.5 / dc["resistance_ohm"])
        ac = mcp_server.simulate_material_circuit(spec, array, {}, {}, "ac")
        assert ac["impedance_response"][-1]["magnitude"] != \
            ac["impedance_response"][0]["magnitude"]
        noise = mcp_server.simulate_material_circuit(spec, array, {}, {}, "noise")
        floor = np.sqrt(4 * KB * 300 * dc["resistance_ohm"] * 1e6)
        assert noise["thermal_noise_v_rms"] == pytest.approx(floor, rel=0.02)
        tran = mcp_server.simulate_material_circuit(
            spec, array, {"duration_ns": 200}, {}, "transient")
        assert tran["waveforms"][0]["read_flux_linkage_wb"] == 0
        assert tran["waveforms"][10]["read_flux_linkage_wb"] != 0

    def test_material_sweep_assigns_each_parameter_its_own_value(self):
        out = mcp_server.sweep_material_parameters(
            {"technology": "ferrofluid"},
            {"voltage_v": [0.5, 1.0], "radius_um": [10, 20]}, "peak_field")
        seen = {(p["parameters"]["voltage_v"], p["parameters"]["radius_um"])
                for p in out["sweep_points"]}
        assert seen == {(0.5, 10), (0.5, 20), (1.0, 10), (1.0, 20)}
        assert out["pareto_front"][0]["parameters"] == {"voltage_v": 1.0, "radius_um": 10}
        unmodelled = mcp_server.sweep_material_parameters(
            {"technology": "ferrofluid"}, {"voltage_v": [1.0]})
        assert unmodelled["pareto_front"] == []
        generic = mcp_server.sweep_material_parameters(
            {"technology": "perovskite"}, {"a": [1, 2], "b": [3]})
        assert generic["sweep_points"][1]["parameters"] == {"a": 2, "b": 3}

    def test_nested_parameters_and_ignored_keys_are_not_silent(self):
        out = mcp_server.run_multiphysics_simulation(
            {"technology": "ferrofluid", "write_duration_ns": 5,
             "parameters": {"volume_fraction": 0.02, "conductivity_s_per_m": 1e-3,
                            "relative_permittivity": 5}},
            mcp_server.configure_material_array(
                "ferrofluid", 2, "linear", 60, [{"electrode_radius_um": 5}] * 2), [], [])
        rows = {r["name"]: r for r in out["parameters"]}
        assert rows["volume_fraction"]["value"] == 0.02
        assert isinstance(out["electrical_electrode_pairs"], list)
        assert any("ignored" in w for w in out["warnings"])

    def test_device_and_chiplet(self):
        device = mcp_server.simulate_material_device("ferrofluid", {"voltage_v": 0.3})
        assert device["success"] and device["energy_pj"] > 0
        chip = mcp_server.material_device_to_chiplet({"technology": "ferrofluid"}, array4())
        assert chip["accepted"] is False and chip["chiplet"] is None  # no measurements

    def test_through_the_mcp_protocol_layer(self):
        server = mcp_server.create_mcp_server()

        async def call():
            return await server.call_tool("configure_material_array", {
                "technology": "ferrofluid", "n_ports": 4, "arrangement": "circular",
                "pitch_um": 100})

        assert "circular arrangement" in str(asyncio.run(call()))
