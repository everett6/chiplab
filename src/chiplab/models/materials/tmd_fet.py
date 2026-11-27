"""
2D Transition Metal Dichalcogenide (TMD) FET Model
==================================================

Implements monolayer & few-layer TMD FET physics (MoS2, WSe2, WS2, MoSe2)
for sub-5nm/sub-2nm beyond-CMOS scaling and valleytronics.
Includes ballistic/quasi-ballistic Landauer transport, quantum capacitance,
Schottky barrier contact resistance, and gate-length scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from chiplab.core.gpu import GPUBackend, get_gpu_backend as get_backend
from chiplab.schemas.common import Provenance, TrackedValue


class TMDMaterial(Enum):
    """2D TMD channel materials."""
    MOS2 = "MoS2"
    WSE2 = "WSe2"
    WS2 = "WS2"
    MOSE2 = "MoSe2"


@dataclass
class TMDFETParams:
    """2D TMD FET parameters with provenance tracking."""
    material: TMDMaterial = TMDMaterial.MOS2
    gate_length_nm: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=3.0, provenance=Provenance.MEASURED, metadata={"unit": "nm"}
    ))
    width_nm: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=20.0, provenance=Provenance.MEASURED, metadata={"unit": "nm"}
    ))
    eot_nm: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.8, provenance=Provenance.ESTIMATED, metadata={"unit": "nm"}
    ))
    bandgap_ev: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.8, provenance=Provenance.PUBLISHED, metadata={"unit": "eV"}
    ))
    effective_mass_m0: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.45, provenance=Provenance.PUBLISHED, metadata={"unit": "m0"}
    ))
    mobility_cm2_vs: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=120.0, provenance=Provenance.MEASURED, metadata={"unit": "cm^2/(V*s)"}
    ))
    schottky_barrier_ev: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.12, provenance=Provenance.ESTIMATED, metadata={"unit": "eV"}
    ))
    contact_res_ohm_um: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=150.0, provenance=Provenance.MEASURED, metadata={"unit": "ohm*um"}
    ))
    vth_v: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.25, provenance=Provenance.ESTIMATED, metadata={"unit": "V"}
    ))
    temperature_k: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=300.0, provenance=Provenance.MEASURED, metadata={"unit": "K"}
    ))


@dataclass
class TMDFETState:
    """2D TMD FET state."""
    params: TMDFETParams = field(default_factory=TMDFETParams)
    vgs: float = 0.0
    vds: float = 0.0
    ids: float = 0.0
    quantum_cap_f_cm2: float = 0.0
    ballisticity: float = 0.0

    def copy(self) -> TMDFETState:
        return TMDFETState(
            params=self.params,
            vgs=self.vgs,
            vds=self.vds,
            ids=self.ids,
            quantum_cap_f_cm2=self.quantum_cap_f_cm2,
            ballisticity=self.ballisticity,
        )


def tmd_quantum_capacitance(params: TMDFETParams) -> float:
    """Compute 2D density-of-states quantum capacitance C_q = q^2 * g_2D (F/cm^2)."""
    q = 1.602176634e-19
    hbar = 1.054571817e-34
    m0 = 9.1093837e-31
    m_eff = params.effective_mass_m0.value * m0
    # Valley degeneracy g_v = 2, spin degeneracy g_s = 2 -> g_2D = 2 * m_eff / (pi * hbar^2)
    g_2d_m2 = 2.0 * m_eff / (np.pi * hbar ** 2)
    g_2d_cm2 = g_2d_m2 * 1e-4
    return float((q ** 2) * g_2d_cm2)


def tmd_fet_iv(
    params: TMDFETParams,
    vgs: float | np.ndarray,
    vds: float | np.ndarray,
    device: str = "auto",
) -> np.ndarray:
    """GPU-accelerated quasi-ballistic 2D TMD FET I-V model."""
    backend: GPUBackend = get_backend(device=device)
    xp = backend.xp

    q = 1.602e-19
    kb = 1.381e-23
    temp = max(params.temperature_k.value, 10.0)
    vt = kb * temp / q

    lg_cm = params.gate_length_nm.value * 1e-7
    w_um = params.width_nm.value * 1e-3
    w_cm = params.width_nm.value * 1e-7

    # Gate oxide + quantum capacitance series combination
    cox = (3.9 * 8.854e-14) / (params.eot_nm.value * 1e-7)
    cq = tmd_quantum_capacitance(params)
    c_eff = (cox * cq) / (cox + cq)

    # Mean free path & ballisticity T = lambda / (lambda + L_g)
    v_th = np.sqrt(2.0 * kb * temp / (params.effective_mass_m0.value * 9.109e-31)) * 100.0  # cm/s
    mfp_cm = 2.0 * params.mobility_cm2_vs.value * vt / max(v_th, 1e4)
    ballisticity = mfp_cm / (mfp_cm + lg_cm)

    # Electrostatics (2D monolayer suppresses DIBL even at 1-3 nm)
    ss_ideality = 1.0 + 0.08 * (2.0 / max(params.gate_length_nm.value, 0.8))
    n_vt = ss_ideality * vt
    dibl = 0.02 * (3.0 / max(params.gate_length_nm.value, 0.8))

    vgs_arr = backend.to_device(np.asarray(vgs, dtype=np.float64))
    vds_arr = backend.to_device(np.asarray(vds, dtype=np.float64))
    vds_abs = xp.abs(vds_arr)
    sign_vds = xp.where(vds_arr >= 0, 1.0, -1.0)

    vth_eff = params.vth_v.value - dibl * vds_abs
    arg = xp.clip((vgs_arr - vth_eff) / max(n_vt, 1e-3), -40.0, 40.0)
    q_inv = c_eff * n_vt * xp.log(1.0 + xp.exp(arg))

    # Ballistic injection velocity + Landauer transmission
    v_inj = 0.5 * v_th * ballisticity + params.mobility_cm2_vs.value * (vds_abs / max(lg_cm, 1e-7)) * (1.0 - ballisticity)
    v_inj = xp.clip(v_inj, None, 1.5e7)

    # Contact resistance attenuation (Schottky + Rc)
    rc_total = (2.0 * params.contact_res_ohm_um.value / max(w_um, 1e-3)) * np.exp(
        max(0.0, params.schottky_barrier_ev.value - 0.1) / max(vt, 0.01)
    )
    i_ideal = w_cm * q_inv * v_inj * xp.tanh(vds_abs / max(2.0 * vt, 0.02))
    i_total = sign_vds * i_ideal / (1.0 + rc_total * xp.abs(i_ideal) / xp.clip(vds_abs, 0.05, None))
    return backend.to_numpy(i_total)


def tmd_scale_sweep(
    gate_lengths_nm: list[float],
    material: TMDMaterial = TMDMaterial.MOS2,
    vdd: float = 0.7,
    device: str = "auto",
) -> dict[str, Any]:
    """Sweep 2D TMD FET performance across gate lengths (e.g. 10nm down to 1nm)."""
    rows = []
    for lg in gate_lengths_nm:
        p = TMDFETParams(material=material)
        p.gate_length_nm.value = float(lg)
        ion = float(tmd_fet_iv(p, vdd, vdd, device=device))
        ioff = float(tmd_fet_iv(p, 0.0, vdd, device=device))
        rows.append({
            "gate_length_nm": float(lg),
            "ion_ua_per_um": ion / (p.width_nm.value * 1e-3) * 1e6,
            "ioff_na_per_um": ioff / (p.width_nm.value * 1e-3) * 1e9,
            "on_off_ratio": ion / max(ioff, 1e-18),
        })
    return {"material": material.value, "vdd_v": vdd, "sweep": rows}


__all__ = [
    "TMDMaterial",
    "TMDFETParams",
    "TMDFETState",
    "tmd_quantum_capacitance",
    "tmd_fet_iv",
    "tmd_scale_sweep",
]
