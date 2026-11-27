"""
Negative Capacitance FET (NCFET) Model
=======================================

Implements FE-HfZrO2 Negative Capacitance FET physics via Landau-Khalatnikov
ferroelectric capacitance matching in series with baseline CMOS/FinFET/GAA MOSCAP,
enabling sub-60 mV/dec subthreshold swing and ultra-low Vdd operation at any nm node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chiplab.core.gpu import GPUBackend, get_gpu_backend as get_backend
from chiplab.models.materials.process_nodes import ProcessNodeParams, get_process_node, mosfet_iv
from chiplab.schemas.common import Provenance, TrackedValue


@dataclass
class NCFETParams:
    """NCFET parameters with provenance tracking."""
    node_nm: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=7.0, provenance=Provenance.ESTIMATED, metadata={"unit": "nm"}
    ))
    fe_thickness_nm: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=4.0, provenance=Provenance.MEASURED, metadata={"unit": "nm"}
    ))
    alpha_fe: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=-3.5e8, provenance=Provenance.ESTIMATED, metadata={"description": "Landau alpha (m/F)", "unit": "m/F"}
    ))
    beta_fe: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.2e9, provenance=Provenance.ESTIMATED, metadata={"description": "Landau beta (m^5/(F*C^2))", "unit": "m^5/(F*C^2)"}
    ))
    de_thickness_nm: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.0, provenance=Provenance.ESTIMATED, metadata={"description": "Interfacial dielectric EOT", "unit": "nm"}
    ))
    temperature_k: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=300.0, provenance=Provenance.MEASURED, metadata={"unit": "K"}
    ))


@dataclass
class NCFETState:
    """NCFET operating state."""
    params: NCFETParams = field(default_factory=NCFETParams)
    voltage_gain: float = 1.0
    subthreshold_swing_mv_dec: float = 45.0
    hysteresis_free: bool = True

    def copy(self) -> NCFETState:
        return NCFETState(
            params=self.params,
            voltage_gain=self.voltage_gain,
            subthreshold_swing_mv_dec=self.subthreshold_swing_mv_dec,
            hysteresis_free=self.hysteresis_free,
        )


def ncfet_capacitance_matching(params: NCFETParams) -> dict[str, Any]:
    """Compute ferroelectric negative capacitance |C_FE|, C_MOS, voltage gain A_v, and SS."""
    t_fe = params.fe_thickness_nm.value * 1e-9
    t_de = params.de_thickness_nm.value * 1e-9
    c_fe_neg = 1.0 / (2.0 * abs(params.alpha_fe.value) * t_fe)  # F/m^2
    c_mos = (3.9 * 8.854e-12) / max(t_de, 1e-10)  # F/m^2
    # Stability condition for hysteresis-free operation: |C_FE| > C_MOS
    hysteresis_free = c_fe_neg > c_mos
    ratio = min(0.88, c_mos / max(c_fe_neg, 1e-9))
    a_v = 1.0 / (1.0 - ratio)
    ss_base = 60.0 * (params.temperature_k.value / 300.0) * 1.08
    ss_nc = ss_base / a_v
    return {
        "c_fe_abs_f_m2": float(c_fe_neg),
        "c_mos_f_m2": float(c_mos),
        "voltage_gain": float(a_v),
        "subthreshold_swing_mv_dec": float(ss_nc),
        "hysteresis_free": bool(hysteresis_free),
    }


def ncfet_iv(
    params: NCFETParams,
    vgs: float | np.ndarray,
    vds: float | np.ndarray,
    width_nm: float | None = None,
    device: str = "auto",
) -> np.ndarray:
    """Compute NCFET drain current using amplified internal gate potential V_int = A_v * V_gs."""
    match = ncfet_capacitance_matching(params)
    a_v = match["voltage_gain"]
    node: ProcessNodeParams = get_process_node(
        node_nm=params.node_nm.value,
        temperature_k=params.temperature_k.value,
    )
    v_int = np.asarray(vgs, dtype=np.float64) * a_v
    return mosfet_iv(node, v_int, vds, width_nm=width_nm, device=device)


__all__ = [
    "NCFETParams",
    "NCFETState",
    "ncfet_capacitance_matching",
    "ncfet_iv",
]
