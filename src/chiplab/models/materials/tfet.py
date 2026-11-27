"""
Tunnel FET (TFET) Model
=======================

Implements band-to-band tunneling (BTBT) Kane-Sze physics for steep-slope
(<60 mV/dec) Tunnel FETs (InGaAs/GaSb heterojunction, Si/Ge, 2D TFETs)
with arbitrary nm gate-length & EOT scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from chiplab.core.gpu import GPUBackend, get_gpu_backend as get_backend
from chiplab.schemas.common import Provenance, TrackedValue


class TFETMaterial(Enum):
    """TFET heterojunction / homojunction systems."""
    SI_HOMO = "Si"
    INGAAS_GASB = "InGaAs/GaSb"
    GE_SI = "Ge/Si"
    TMD_HETERO = "MoS2/WTe2"


@dataclass
class TFETParams:
    """TFET parameters with provenance tracking."""
    material: TFETMaterial = TFETMaterial.INGAAS_GASB
    node_nm: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=14.0, provenance=Provenance.ESTIMATED, metadata={"unit": "nm"}
    ))
    width_nm: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=50.0, provenance=Provenance.MEASURED, metadata={"unit": "nm"}
    ))
    eot_nm: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.9, provenance=Provenance.ESTIMATED, metadata={"unit": "nm"}
    ))
    tunnel_bandgap_ev: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.35, provenance=Provenance.PUBLISHED, metadata={"description": "Effective staggered tunnel barrier", "unit": "eV"}
    ))
    reduced_mass_m0: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.05, provenance=Provenance.PUBLISHED, metadata={"unit": "m0"}
    ))
    v_onset_v: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.12, provenance=Provenance.ESTIMATED, metadata={"unit": "V"}
    ))
    temperature_k: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=300.0, provenance=Provenance.MEASURED, metadata={"unit": "K"}
    ))


def tfet_btbt_iv(
    params: TFETParams,
    vgs: float | np.ndarray,
    vds: float | np.ndarray,
    device: str = "auto",
) -> np.ndarray:
    """GPU-accelerated Kane band-to-band tunneling I-V model for TFET."""
    backend: GPUBackend = get_backend(device=device)
    xp = backend.xp

    q = 1.602e-19
    hbar = 1.055e-34
    m0 = 9.109e-31
    m_r = params.reduced_mass_m0.value * m0
    eg_j = params.tunnel_bandgap_ev.value * q

    # Screening length lambda = sqrt((eps_s / eps_ox) * t_body * t_ox)
    t_body_m = max(3.0, 0.35 * params.node_nm.value) * 1e-9
    t_ox_m = params.eot_nm.value * 1e-9
    lam_m = np.sqrt((12.0 / 3.9) * t_body_m * t_ox_m)

    # Kane parameters
    b_kane = (np.pi * np.sqrt(m_r) * (eg_j ** 1.5)) / (2.0 * np.sqrt(2.0) * q * hbar)  # V/m
    a_kane = (q ** 2) * np.sqrt(2.0 * m_r) / (18.0 * np.pi * (hbar ** 2) * np.sqrt(eg_j))

    vgs_arr = backend.to_device(np.asarray(vgs, dtype=np.float64))
    vds_arr = backend.to_device(np.asarray(vds, dtype=np.float64))
    vds_abs = xp.abs(vds_arr)
    sign_vds = xp.where(vds_arr >= 0, 1.0, -1.0)

    # Smooth onset above V_onset
    v_eff = 0.02 * xp.log(1.0 + xp.exp(xp.clip((vgs_arr - params.v_onset_v.value) / 0.02, -40.0, 40.0)))
    e_field = xp.clip(v_eff / max(lam_m, 1e-9), 1e4, None)  # V/m

    # Tunneling generation rate scaled by tunnel window & area
    w_m = params.width_nm.value * 1e-9
    tunnel_area = w_m * t_body_m
    j_btbt = a_kane * (e_field ** 2) * xp.exp(-xp.clip(b_kane / e_field, 0.0, 60.0))
    # Drain bias saturation factor
    sat = xp.tanh(vds_abs / 0.08)
    # Leakage floor (thermionic/ambipolar)
    i_leak = 1e-13 * (params.width_nm.value * 1e-3) * float(np.exp(np.clip(-eg_j / (2.0 * 1.381e-23 * params.temperature_k.value), -40.0, 0.0)))

    i_total = sign_vds * (j_btbt * tunnel_area * lam_m * sat + i_leak)
    return backend.to_numpy(i_total)


def tfet_subthreshold_swing(params: TFETParams, vdd: float = 0.5) -> float:
    """Estimate average subthreshold swing (mV/dec) between Ioff and Ion."""
    vgs_grid = np.linspace(0.0, vdd, 25)
    ids = np.abs(tfet_btbt_iv(params, vgs_grid, np.full_like(vgs_grid, vdd), device="cpu"))
    i_min = max(float(np.min(ids)), 1e-18)
    i_max = max(float(np.max(ids)), i_min * 10.0)
    decades = np.log10(i_max / i_min)
    return float((vdd * 1e3) / max(decades, 0.5))


__all__ = [
    "TFETMaterial",
    "TFETParams",
    "tfet_btbt_iv",
    "tfet_subthreshold_swing",
]
