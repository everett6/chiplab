"""
Graphene p-n Junction & GNR FET Model
=====================================

Implements Dirac-fermion ballistic transport, Klein tunneling transmission
across p-n potential steps, and Graphene Nanoribbon (GNR) bandgap opening
at scaled nm widths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chiplab.core.gpu import GPUBackend, get_gpu_backend as get_backend
from chiplab.schemas.common import Provenance, TrackedValue


@dataclass
class GraphenePNParams:
    """Graphene p-n junction / GNR FET parameters with provenance tracking."""
    fermi_velocity_m_s: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.0e6, provenance=Provenance.PUBLISHED, metadata={"unit": "m/s"}
    ))
    channel_length_nm: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=20.0, provenance=Provenance.MEASURED, metadata={"unit": "nm"}
    ))
    ribbon_width_nm: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=5.0, provenance=Provenance.MEASURED, metadata={"unit": "nm"}
    ))
    junction_step_nm: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=10.0, provenance=Provenance.ESTIMATED, metadata={"description": "Split-gate p-n transition length", "unit": "nm"}
    ))
    mobility_cm2_vs: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=15000.0, provenance=Provenance.MEASURED, metadata={"unit": "cm^2/(V*s)"}
    ))
    temperature_k: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=300.0, provenance=Provenance.MEASURED, metadata={"unit": "K"}
    ))


def gnr_bandgap_ev(ribbon_width_nm: float) -> float:
    """Empirical armchair GNR bandgap scaling E_g ~ 0.8 / W_nm (eV) for sub-10nm ribbons."""
    return float(0.8 / max(ribbon_width_nm, 0.4))


def klein_tunneling_transmission(
    angles_rad: float | np.ndarray,
    kf_m_inv: float,
    step_length_nm: float,
    device: str = "auto",
) -> np.ndarray:
    """Compute Cheianov-Fal'ko Klein tunneling transmission T(theta) = exp(-pi * k_F * d * sin^2(theta))."""
    backend: GPUBackend = get_backend(device=device)
    xp = backend.xp
    theta = backend.to_device(np.asarray(angles_rad, dtype=np.float64))
    d_m = step_length_nm * 1e-9
    trans = xp.exp(-np.pi * kf_m_inv * d_m * (xp.sin(theta) ** 2))
    return backend.to_numpy(trans)


def graphene_pn_iv(
    params: GraphenePNParams,
    v_bias: float | np.ndarray,
    v_split_gate: float = 0.4,
    device: str = "auto",
) -> np.ndarray:
    """GPU-accelerated Dirac-cone Landauer current through a graphene p-n junction."""
    backend: GPUBackend = get_backend(device=device)
    xp = backend.xp

    q = 1.602e-19
    h = 6.626e-34
    hbar = h / (2.0 * np.pi)
    vf = params.fermi_velocity_m_s.value

    # Fermi wavevector from split-gate potential
    ef_j = max(abs(v_split_gate), 0.05) * q
    kf = ef_j / (hbar * vf)

    # Integrate Klein transmission over incidence angles [-pi/2, pi/2]
    thetas = np.linspace(-0.49 * np.pi, 0.49 * np.pi, 65)
    t_avg = float(np.mean(klein_tunneling_transmission(thetas, kf, params.junction_step_nm.value, device="cpu") * np.cos(thetas)))

    w_m = params.ribbon_width_nm.value * 1e-9
    n_modes = max(1.0, (2.0 * kf * w_m) / np.pi)
    g_ballistic = (4.0 * q * q / h) * n_modes * t_avg

    # GNR bandgap suppression when W_nm is ultra-narrow
    eg_ev = gnr_bandgap_ev(params.ribbon_width_nm.value)
    gap_factor = 1.0 / (1.0 + np.exp((eg_ev - abs(v_split_gate)) / 0.05))

    vb = backend.to_device(np.asarray(v_bias, dtype=np.float64))
    return backend.to_numpy(g_ballistic * gap_factor * vb)


__all__ = [
    "GraphenePNParams",
    "gnr_bandgap_ev",
    "klein_tunneling_transmission",
    "graphene_pn_iv",
]
