"""Tests for GPU-accelerated multiphysics and engine solvers."""

from __future__ import annotations

import numpy as np
import pytest

from chiplab.engines.material_solver import ReducedOrderSolver
from chiplab.engines.multiphysics_solver import ParticleScaleSolver
from chiplab.multiphysics.fields import FieldSystem


def test_field_system_device_and_copy():
    for device in ("cpu", "auto"):
        fs = FieldSystem(device=device)
        fs_copy = fs.copy()
        assert fs_copy is not fs
        assert np.allclose(fs_copy.E, fs.E)
        assert np.allclose(fs_copy.H, fs.H)


def test_reduced_order_solver_cpu_vs_auto():
    solver_cpu = ReducedOrderSolver(device="cpu")
    solver_auto = ReducedOrderSolver(device="auto")

    state = np.array([0.0, 0.0, 1.0])
    inputs = {"H_ext": np.array([1000.0, 0.0, 0.0])}
    m_cpu = solver_cpu.solve(state=state, inputs=inputs, dt=1e-9)
    m_auto = solver_auto.solve(state=state, inputs=inputs, dt=1e-9)
    assert np.all(np.isfinite(m_cpu))
    assert np.all(np.isfinite(m_auto))
    assert np.allclose(m_cpu, m_auto, rtol=1e-4, atol=1e-5)


def test_particle_scale_solver_init_and_dipole():
    for device in ("cpu", "auto"):
        ps = ParticleScaleSolver(num_particles=4, device=device)
        moments = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
            ]
        )
        field = ps._dipole_interaction(0, 1, moments)
        assert field.shape == (3,)
        assert np.all(np.isfinite(field))
