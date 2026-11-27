"""Physics checks for the stochastic-LLG ParticleScaleSolver against closed-form results."""

from __future__ import annotations

import math

import numpy as np
import pytest

from chiplab.credibility import CredibilityError
from chiplab.engines.multiphysics_solver import KB, MU0, ParticleScaleSolver


def test_uniform_field_runs_and_keeps_unit_moments():
    ps = ParticleScaleSolver(num_particles=8, device="cpu", seed=1)
    out = ps.solve({}, np.array([0.0, 0.0, 1e4]), 5e-11)
    assert out["moments"].shape == (3, 8)
    assert np.allclose(np.linalg.norm(out["moments"], axis=0), 1.0)
    assert out["substeps"] == 50


def test_larmor_precession_rate_at_zero_temperature():
    """Single isotropic spin, small damping: azimuth advances at gamma0*H/(1+a^2)."""
    a, h = 0.01, 1e5
    ps = ParticleScaleSolver(num_particles=1, device="cpu", alpha=a, temperature_k=0,
                             anisotropy_j_per_m3=0, dt=1e-14)
    ps.solve({"moments": [1.0, 0.0, 0.0]}, np.array([0.0, 0.0, h]), 1e-14)
    t = 2e-11
    phi0 = ps.orientations[0]
    ps.solve(None, np.array([0.0, 0.0, h]), t)
    dphi = (ps.orientations[0] - phi0) % (2 * math.pi)   # -m x H: counter-clockwise about +z
    expected = (ps.gamma0 * h / (1 + a * a) * t) % (2 * math.pi)
    assert dphi == pytest.approx(expected, rel=1e-3)


def test_thermal_equilibrium_matches_langevin_function():
    """Non-interacting isotropic particles: <m_z> = L(xi), xi = mu0*mu*H/(kB*T)."""
    ps = ParticleScaleSolver(num_particles=400, grid_size=(20, 20, 1), device="cpu", alpha=1.0,
                             anisotropy_j_per_m3=0, dipolar=False, seed=7)
    xi = 2.0
    h = xi * KB * 300.0 / (MU0 * ps.moment_am2)
    field = np.array([0.0, 0.0, h])
    ps.solve({}, field, 3e-9)                       # ~10 relaxation times
    samples = [ps.solve(None, field, 1e-10)["mean_m"][2] for _ in range(20)]
    langevin = 1 / math.tanh(xi) - 1 / xi
    assert np.mean(samples) == pytest.approx(langevin, abs=0.04)


@pytest.mark.parametrize("ratio,switches", [(0.9, False), (1.1, True)])
def test_stoner_wohlfarth_switching_field(ratio, switches):
    """At T=0 an easy-axis reversing field switches the moment only above H_K = 2K/(mu0 Ms)."""
    ps = ParticleScaleSolver(num_particles=1, device="cpu", alpha=1.0, temperature_k=0,
                             dt=1e-12)
    hk = ps.anisotropy_field_a_per_m
    tilt = 1e-2   # break the exact symmetry; grows ~e^18 over 50 ns at 1.1 H_K
    ps.solve({"moments": [tilt, 0.0, 1.0], "easy_axes": [0.0, 0.0, 1.0]},
             np.array([0.0, 0.0, -ratio * hk]), 5e-8)
    assert bool(ps.moments[2, 0] < 0) == switches


def test_dipole_field_matches_closed_form():
    d = 10e-9
    ps = ParticleScaleSolver(num_particles=2, device="cpu", diameter_m=d)
    ps.solve({"positions": [[0, 0, 0], [0, 0, 3 * d]], "moments": [0, 0, 1]},
             np.zeros(3), 1e-15)
    m = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    r = 3 * d
    on_axis = ps._dipole_interaction(0, 1, m)
    assert on_axis == pytest.approx([0, 0, 2 * ps.moment_am2 / (4 * math.pi * r ** 3)])
    assert np.allclose(ps._dipole_fields(m)[:, 0], on_axis)


def test_dipolar_coupling_changes_dynamics():
    def run(dipolar):
        ps = ParticleScaleSolver(num_particles=8, grid_size=(2, 2, 2), device="cpu",
                                 temperature_k=0, dipolar=dipolar, spacing_m=11e-9,
                                 ms_a_per_m=1.4e6, seed=3)
        return ps.solve({}, np.zeros(3), 2e-10)["moments"]
    assert not np.allclose(run(True), run(False), atol=1e-3)


def test_seed_reproducible_and_inputs_validated():
    a = ParticleScaleSolver(num_particles=5, device="cpu", seed=11).solve({}, np.zeros(3), 1e-11)
    b = ParticleScaleSolver(num_particles=5, device="cpu", seed=11).solve({}, np.zeros(3), 1e-11)
    assert np.array_equal(a["moments"], b["moments"])
    ps = ParticleScaleSolver(num_particles=2, device="cpu")
    with pytest.raises(ValueError):
        ps.solve({"positions": [[0, 0, 0], [0, 0, 1e-9]]}, np.zeros(3), 1e-12)   # overlap
    with pytest.raises(ValueError):
        ParticleScaleSolver(num_particles=2, device="cpu").solve({}, np.zeros(4), 1e-12)
    with pytest.raises(CredibilityError):
        ps.solve({"material": "ferrofluid"}, np.zeros(3), 1e-12)
