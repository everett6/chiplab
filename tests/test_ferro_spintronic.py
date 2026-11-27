"""Tests for Ferroelectric (FeFET/FeRAM) and Spintronic (STT/SOT-MRAM) models."""

from __future__ import annotations

import numpy as np

from chiplab.models.materials.ferroelectric import (
    FerroelectricState,
    fefet_threshold_shift,
    ferroelectric_pv_loop,
    landau_khalatnikov_step,
)
from chiplab.models.materials.spintronic import (
    MRAMState,
    sot_switching_step,
    stt_switching_step,
    thermal_stability_factor,
    tmr_read_resistance,
)
from chiplab.schemas.common import TrackedValue


def test_ferroelectric_state_and_copy():
    s = FerroelectricState()
    s2 = s.copy()
    assert s2 is not s
    assert isinstance(s.polarization, TrackedValue)


def test_landau_khalatnikov_step_and_pv_loop():
    s = FerroelectricState()
    s_next = landau_khalatnikov_step(s, voltage=2.0, dt=1e-9)
    assert isinstance(s_next, FerroelectricState)
    assert np.isfinite(s_next.polarization.value)

    sweep = np.linspace(-3.0, 3.0, 21).tolist() + np.linspace(3.0, -3.0, 21).tolist()
    loop = ferroelectric_pv_loop(s, voltage_sweep=sweep)
    assert isinstance(loop, dict)
    assert len(loop) > 0


def test_fefet_threshold_shift():
    s = FerroelectricState()
    res = fefet_threshold_shift(s, program_voltage=3.0, program_duration=1e-6, n_cycles=10)
    assert isinstance(res, dict)
    assert len(res) > 0


def test_mram_state_stt_sot_tmr_and_stability():
    m = MRAMState()
    m_copy = m.copy()
    assert m_copy is not m
    assert isinstance(m.magnetization, TrackedValue)

    m_stt = stt_switching_step(m, current_density=1e11, pulse_duration=5e-9)
    assert isinstance(m_stt, MRAMState)

    m_sot = sot_switching_step(m, current_density=1e11, pulse_duration=2e-9)
    assert isinstance(m_sot, MRAMState)

    tmr = tmr_read_resistance(m)
    assert isinstance(tmr, dict) and len(tmr) > 0

    stab = thermal_stability_factor(m, temperature=300.0)
    assert isinstance(stab, dict) and len(stab) > 0
