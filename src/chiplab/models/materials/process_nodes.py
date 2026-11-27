"""
Multi-Node Process Technology & Scaling Engine (180 nm -> 1 nm)
===============================================================

Provides continuous & discrete process node modeling across bulk CMOS,
planar HKMG, FDSOI, FinFET, GAA nanosheet, CFET, and 2D-channel nodes.
Bridges ChipLab's PDK_LADDER (180, 130, 65, 28, 15, 7, 3, 2 nm) with
arbitrary nm interpolation, corner/temperature derating, GPU-accelerated
short-channel MOSFET I-V, and experimental device scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from chiplab.core.gpu import GPUBackend, get_gpu_backend as get_backend
from chiplab.library.pdk_library import PDK_LADDER, PdkProfile
from chiplab.schemas.common import Provenance, TrackedValue


class ProcessCorner(Enum):
    """Standard foundry process corners."""
    TT = "TT"
    FF = "FF"
    SS = "SS"
    FS = "FS"
    SF = "SF"


class TransistorArch(Enum):
    """Transistor architecture by technology generation."""
    PLANAR_BULK = "planar_bulk"
    PLANAR_HKMG = "planar_hkmg"
    FDSOI = "fdsoi"
    FINFET = "finfet"
    GAA_NANOSHEET = "gaa_nanosheet"
    CFET = "cfet"
    TMD_2D = "tmd_2d"


@dataclass
class ProcessNodeParams:
    """Process technology node parameters with provenance tracking."""
    node_nm: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=28.0,
        provenance=Provenance.PUBLISHED,
        metadata={"description": "Technology node feature size", "unit": "nm"},
    ))
    architecture: TransistorArch = TransistorArch.PLANAR_HKMG
    corner: ProcessCorner = ProcessCorner.TT
    pdk_key: str = "n28"

    vdd: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.9,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Nominal supply voltage", "unit": "V"},
    ))
    vth_n: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.32,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "NMOS threshold voltage", "unit": "V"},
    ))
    vth_p: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=-0.32,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "PMOS threshold voltage", "unit": "V"},
    ))
    gate_length_nm: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=28.0,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Physical gate length", "unit": "nm"},
    ))
    eot_nm: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.2,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Equivalent oxide thickness", "unit": "nm"},
    ))
    mobility_n: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=350.0,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Effective electron mobility", "unit": "cm^2/(V*s)"},
    ))
    mobility_p: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=140.0,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Effective hole mobility", "unit": "cm^2/(V*s)"},
    ))
    vsat_cm_s: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.0e7,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Carrier saturation velocity", "unit": "cm/s"},
    ))
    dibl_mv_v: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=55.0,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Drain-induced barrier lowering", "unit": "mV/V"},
    ))
    subthreshold_swing_mv_dec: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=72.0,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Subthreshold swing at 300K", "unit": "mV/dec"},
    ))
    fo4_delay_ps: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=11.0,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Fanout-of-4 inverter delay", "unit": "ps"},
    ))
    gate_density_kgates_mm2: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=2000.0,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Standard-cell logic density", "unit": "kgates/mm^2"},
    ))
    sram_bitcell_um2: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.127,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "6T SRAM bitcell area", "unit": "um^2"},
    ))
    dynamic_energy_fj_per_gate: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.73,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Switching energy per gate", "unit": "fJ"},
    ))
    leakage_nw_per_gate: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.5,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Static leakage per gate", "unit": "nW"},
    ))
    metal1_pitch_nm: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=90.0,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Metal 1 minimum pitch", "unit": "nm"},
    ))
    wire_r_ohm_per_um: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=4.5,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "M1 interconnect resistance per micron", "unit": "ohm/um"},
    ))
    wire_c_ff_per_um: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.20,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "M1 interconnect capacitance per micron", "unit": "fF/um"},
    ))
    contact_res_ohm: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=35.0,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Source/drain contact resistance", "unit": "ohm"},
    ))
    sigma_vt_mv: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=18.0,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Pelgrom Vt standard deviation", "unit": "mV"},
    ))
    temperature_k: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=300.0,
        provenance=Provenance.MEASURED,
        metadata={"description": "Operating temperature", "unit": "K"},
    ))

    def copy(self) -> ProcessNodeParams:
        """Create a deep copy using Pydantic v2 model_copy."""
        new_node = ProcessNodeParams(
            architecture=self.architecture,
            corner=self.corner,
            pdk_key=self.pdk_key,
        )
        for fname in self.__dataclass_fields__:
            val = getattr(self, fname)
            if isinstance(val, TrackedValue):
                setattr(new_node, fname, val.model_copy())
            else:
                setattr(new_node, fname, val)
        return new_node


def _infer_architecture(node_nm: float) -> TransistorArch:
    if node_nm >= 40.0:
        return TransistorArch.PLANAR_BULK
    if node_nm >= 20.0:
        return TransistorArch.PLANAR_HKMG
    if node_nm >= 4.0:
        return TransistorArch.FINFET
    if node_nm >= 1.6:
        return TransistorArch.GAA_NANOSHEET
    return TransistorArch.CFET


def _interpolate_pdk_value(node_nm: float, field_name: str, log_interp: bool = True) -> float:
    """Interpolate a scalar field from PDK_LADDER across arbitrary node_nm (1nm..180nm)."""
    # Sort unique node_nm entries from PDK_LADDER (prefer sky130 over ihp at 130nm)
    by_nm: dict[float, PdkProfile] = {}
    for prof in PDK_LADDER.values():
        if prof.node_nm not in by_nm or prof.key == "sky130":
            by_nm[prof.node_nm] = prof
    sorted_nms = sorted(by_nm.keys())

    # Clamp or extrapolate below 2 nm down to 1 nm
    if node_nm >= sorted_nms[-1]:
        return by_nm[sorted_nms[-1]].scalar(field_name)
    if node_nm <= sorted_nms[0]:
        # Extrapolate smoothly from 3nm -> 2nm down to target (<2nm)
        n1, n2 = sorted_nms[0], sorted_nms[1]
        v1, v2 = by_nm[n1].scalar(field_name), by_nm[n2].scalar(field_name)
        ratio = max(node_nm, 0.5) / n1
        if log_interp and v1 > 0 and v2 > 0:
            slope = np.log(v1 / v2) / np.log(n1 / n2)
            return float(v1 * (ratio ** slope))
        return float(v1 + (v1 - v2) * (n1 - node_nm) / (n1 - n2))

    # Bracket interpolation
    for i in range(len(sorted_nms) - 1):
        lo, hi = sorted_nms[i], sorted_nms[i + 1]
        if lo <= node_nm <= hi:
            v_lo = by_nm[lo].scalar(field_name)
            v_hi = by_nm[hi].scalar(field_name)
            if log_interp and v_lo > 0 and v_hi > 0:
                alpha = np.log(node_nm / lo) / np.log(hi / lo)
                return float(np.exp(np.log(v_lo) + alpha * (np.log(v_hi) - np.log(v_lo))))
            alpha = (node_nm - lo) / (hi - lo)
            return float(v_lo + alpha * (v_hi - v_lo))

    return by_nm[sorted_nms[0]].scalar(field_name)


def get_process_node(
    node_nm: float = 28.0,
    corner: str | ProcessCorner = ProcessCorner.TT,
    temperature_k: float = 300.0,
    pdk_key: str | None = None,
) -> ProcessNodeParams:
    """Build a ProcessNodeParams for any technology node (e.g. 180, 130, 65, 45, 28, 14, 7, 5, 3, 2, 1.4, 1.0 nm).

    Applies corner (TT, FF, SS, FS, SF) and temperature derating.
    """
    if node_nm <= 0:
        raise ValueError(f"node_nm must be positive, got {node_nm}")

    corner_enum = corner if isinstance(corner, ProcessCorner) else ProcessCorner(corner.upper())

    # Exact PDK lookup if specified or exact match
    matched_pdk: PdkProfile | None = None
    if pdk_key and pdk_key in PDK_LADDER:
        matched_pdk = PDK_LADDER[pdk_key]
        node_nm = matched_pdk.node_nm
    else:
        for prof in PDK_LADDER.values():
            if abs(prof.node_nm - node_nm) < 1e-6:
                matched_pdk = prof
                break

    if matched_pdk is not None:
        vdd = matched_pdk.scalar("nominal_vdd_v")
        vth = matched_pdk.scalar("vth_v")
        fo4 = matched_pdk.scalar("fo4_delay_ps")
        density = matched_pdk.scalar("gate_density_kgates_mm2")
        sram = matched_pdk.scalar("sram_bitcell_um2")
        dyn_e = matched_pdk.scalar("dynamic_energy_fj_per_gate")
        leak = matched_pdk.scalar("leakage_nw_per_gate")
        m1_pitch = matched_pdk.scalar("metal1_pitch_nm")
        wire_c = matched_pdk.scalar("wire_cap_ff_per_um")
        key_str = matched_pdk.key
        prov = Provenance.PUBLISHED
    else:
        vdd = _interpolate_pdk_value(node_nm, "nominal_vdd_v", log_interp=True)
        vth = _interpolate_pdk_value(node_nm, "vth_v", log_interp=True)
        fo4 = _interpolate_pdk_value(node_nm, "fo4_delay_ps", log_interp=True)
        density = _interpolate_pdk_value(node_nm, "gate_density_kgates_mm2", log_interp=True)
        sram = _interpolate_pdk_value(node_nm, "sram_bitcell_um2", log_interp=True)
        dyn_e = _interpolate_pdk_value(node_nm, "dynamic_energy_fj_per_gate", log_interp=True)
        leak = _interpolate_pdk_value(node_nm, "leakage_nw_per_gate", log_interp=True)
        m1_pitch = _interpolate_pdk_value(node_nm, "metal1_pitch_nm", log_interp=True)
        wire_c = _interpolate_pdk_value(node_nm, "wire_cap_ff_per_um", log_interp=False)
        key_str = f"n{node_nm:g}"
        prov = Provenance.ESTIMATED

    arch = _infer_architecture(node_nm)

    # Physical gate length & EOT scaling
    gate_len = max(5.0, float(node_nm) * (0.9 if node_nm >= 28 else 1.2 if node_nm >= 5 else 2.2))
    eot = max(0.55, 0.7 + 0.018 * node_nm)

    # Mobility & electrostatics by architecture
    if arch in (TransistorArch.GAA_NANOSHEET, TransistorArch.CFET):
        ss_300 = 64.0
        dibl = 35.0
        mob_n, mob_p = 280.0, 160.0
    elif arch == TransistorArch.FINFET:
        ss_300 = 67.0
        dibl = 42.0
        mob_n, mob_p = 310.0, 150.0
    else:
        ss_300 = 75.0 + 0.05 * node_nm
        dibl = 55.0 + 0.1 * node_nm
        mob_n, mob_p = 420.0, 140.0

    # Wire R scales inversely with cross-section area (~pitch^2) plus grain-boundary scattering
    wire_r = 0.08 * (180.0 / max(m1_pitch, 12.0)) ** 1.85
    contact_r = 12.0 * (28.0 / max(node_nm, 1.0)) ** 0.75
    sigma_vt = 12.0 + 18.0 * np.sqrt(28.0 / max(node_nm, 1.0))

    # Corner adjustments (Vt shift & mobility factor)
    vt_n_shift, vt_p_shift = 0.0, 0.0
    mob_n_mult, mob_p_mult = 1.0, 1.0
    if corner_enum == ProcessCorner.FF:
        vt_n_shift, vt_p_shift = -0.045, 0.045
        mob_n_mult, mob_p_mult = 1.12, 1.12
    elif corner_enum == ProcessCorner.SS:
        vt_n_shift, vt_p_shift = 0.050, -0.050
        mob_n_mult, mob_p_mult = 0.88, 0.88
    elif corner_enum == ProcessCorner.FS:
        vt_n_shift, vt_p_shift = -0.040, -0.040
        mob_n_mult, mob_p_mult = 1.10, 0.90
    elif corner_enum == ProcessCorner.SF:
        vt_n_shift, vt_p_shift = 0.040, 0.040
        mob_n_mult, mob_p_mult = 0.90, 1.10

    # Temperature derating relative to 300 K
    dt = temperature_k - 300.0
    temp_ratio = max(temperature_k, 50.0) / 300.0
    mob_temp_factor = temp_ratio ** (-1.4)
    vt_temp_shift = -0.0012 * dt  # ~ -1.2 mV/K
    ss_temp = ss_300 * temp_ratio
    leak_temp_factor = float(np.exp(np.clip(dt / 25.0, -5.0, 5.0)))

    vth_n = max(0.08, vth + vt_n_shift + vt_temp_shift)
    vth_p = -max(0.08, vth - vt_p_shift + vt_temp_shift)
    delay_factor = (1.0 / (0.5 * (mob_n_mult + mob_p_mult))) * (temp_ratio ** 0.35)

    return ProcessNodeParams(
        node_nm=TrackedValue(value=float(node_nm), provenance=prov, metadata={"unit": "nm"}),
        architecture=arch,
        corner=corner_enum,
        pdk_key=key_str,
        vdd=TrackedValue(value=float(vdd), provenance=prov, metadata={"unit": "V"}),
        vth_n=TrackedValue(value=float(vth_n), provenance=Provenance.ESTIMATED, metadata={"unit": "V"}),
        vth_p=TrackedValue(value=float(vth_p), provenance=Provenance.ESTIMATED, metadata={"unit": "V"}),
        gate_length_nm=TrackedValue(value=float(gate_len), provenance=Provenance.ESTIMATED, metadata={"unit": "nm"}),
        eot_nm=TrackedValue(value=float(eot), provenance=Provenance.ESTIMATED, metadata={"unit": "nm"}),
        mobility_n=TrackedValue(value=float(mob_n * mob_n_mult * mob_temp_factor), provenance=Provenance.ESTIMATED, metadata={"unit": "cm^2/(V*s)"}),
        mobility_p=TrackedValue(value=float(mob_p * mob_p_mult * mob_temp_factor), provenance=Provenance.ESTIMATED, metadata={"unit": "cm^2/(V*s)"}),
        vsat_cm_s=TrackedValue(value=1.0e7, provenance=Provenance.ESTIMATED, metadata={"unit": "cm/s"}),
        dibl_mv_v=TrackedValue(value=float(dibl), provenance=Provenance.ESTIMATED, metadata={"unit": "mV/V"}),
        subthreshold_swing_mv_dec=TrackedValue(value=float(ss_temp), provenance=Provenance.ESTIMATED, metadata={"unit": "mV/dec"}),
        fo4_delay_ps=TrackedValue(value=float(fo4 * delay_factor), provenance=prov, metadata={"unit": "ps"}),
        gate_density_kgates_mm2=TrackedValue(value=float(density), provenance=prov, metadata={"unit": "kgates/mm^2"}),
        sram_bitcell_um2=TrackedValue(value=float(sram), provenance=prov, metadata={"unit": "um^2"}),
        dynamic_energy_fj_per_gate=TrackedValue(value=float(dyn_e), provenance=prov, metadata={"unit": "fJ"}),
        leakage_nw_per_gate=TrackedValue(value=float(leak * leak_temp_factor), provenance=prov, metadata={"unit": "nW"}),
        metal1_pitch_nm=TrackedValue(value=float(m1_pitch), provenance=prov, metadata={"unit": "nm"}),
        wire_r_ohm_per_um=TrackedValue(value=float(wire_r), provenance=Provenance.ESTIMATED, metadata={"unit": "ohm/um"}),
        wire_c_ff_per_um=TrackedValue(value=float(wire_c), provenance=prov, metadata={"unit": "fF/um"}),
        contact_res_ohm=TrackedValue(value=float(contact_r), provenance=Provenance.ESTIMATED, metadata={"unit": "ohm"}),
        sigma_vt_mv=TrackedValue(value=float(sigma_vt), provenance=Provenance.ESTIMATED, metadata={"unit": "mV"}),
        temperature_k=TrackedValue(value=float(temperature_k), provenance=Provenance.MEASURED, metadata={"unit": "K"}),
    )


def mosfet_iv(
    node: ProcessNodeParams,
    vgs: float | np.ndarray,
    vds: float | np.ndarray,
    width_nm: float | None = None,
    length_nm: float | None = None,
    is_pmos: bool = False,
    device: str = "auto",
) -> np.ndarray:
    """GPU-accelerated short-channel MOSFET I-V model (alpha-power + subthreshold + DIBL).

    Args:
        node: ProcessNodeParams for the target nm node
        vgs: Gate-source voltage (V) — scalar or array
        vds: Drain-source voltage (V) — scalar or array
        width_nm: Transistor width in nm (defaults to 4 * node_nm)
        length_nm: Gate length in nm (defaults to node.gate_length_nm)
        is_pmos: True for PMOS (uses |Vgs|, |Vds|, |Vth_p| and returns positive magnitude)
        device: 'auto', 'cuda', or 'cpu'

    Returns:
        Drain current array (A)
    """
    backend: GPUBackend = get_backend(device=device)
    xp = backend.xp

    w_cm = (width_nm if width_nm is not None else 4.0 * node.node_nm.value) * 1e-7
    l_cm = (length_nm if length_nm is not None else node.gate_length_nm.value) * 1e-7
    eot_cm = node.eot_nm.value * 1e-7
    eps_ox = 3.9 * 8.854e-14  # F/cm
    cox = eps_ox / eot_cm

    mu = node.mobility_p.value if is_pmos else node.mobility_n.value
    vth0 = abs(node.vth_p.value) if is_pmos else node.vth_n.value
    vsat = node.vsat_cm_s.value
    dibl = node.dibl_mv_v.value * 1e-3  # V/V
    ss = node.subthreshold_swing_mv_dec.value * 1e-3  # V/dec
    n_vt = ss / np.log(10.0)

    vgs_arr = backend.to_device(np.abs(np.asarray(vgs, dtype=np.float64)))
    vds_arr = backend.to_device(np.asarray(vds, dtype=np.float64))
    sign_vds = xp.where(vds_arr >= 0, 1.0, -1.0)
    vds_abs = xp.abs(vds_arr)

    # Effective threshold with DIBL
    vth_eff = vth0 - dibl * vds_abs

    # Smooth overdrive voltage (log-sum-exp for C1 continuity across subthreshold/strong inversion)
    arg = xp.clip((vgs_arr - vth_eff) / max(n_vt, 1e-3), -40.0, 40.0)
    vov = n_vt * xp.log(1.0 + xp.exp(arg))

    # Velocity saturation field E_sat = 2*vsat/mu
    esat_l = (2.0 * vsat / max(mu, 1.0)) * l_cm
    vdsat = (vov * esat_l) / (vov + esat_l + 1e-12)
    vds_eff = xp.minimum(vds_abs, vdsat)

    # Drain current with channel-length modulation (lambda)
    k = mu * cox * (w_cm / max(l_cm, 1e-8))
    lam = 0.08 * (28.0 / max(node.node_nm.value, 1.0)) ** 0.35
    id_strong = k * (vov * vds_eff - 0.5 * vds_eff * vds_eff) / (1.0 + vds_eff / max(esat_l, 1e-9))
    id_strong = id_strong * (1.0 + lam * vds_abs)

    # Contact resistance degradation
    r_sd = 2.0 * node.contact_res_ohm.value * (100.0 / max(width_nm or (4.0 * node.node_nm.value), 5.0))
    id_total = sign_vds * id_strong / (1.0 + r_sd * xp.abs(id_strong) / xp.clip(vds_abs, 0.05, None))

    return backend.to_numpy(id_total)


def scale_experimental_device(
    technology: str,
    target_node_nm: float,
    base_metrics: dict[str, Any] | None = None,
    corner: str = "TT",
    temperature_k: float = 300.0,
) -> dict[str, Any]:
    """Scale an experimental memory/compute technology to any target nm node.

    Combines PDK interconnect & access-transistor scaling with device-specific
    area, energy, latency, and density scaling laws.
    """
    from chiplab.credibility import refuse_ferrofluid

    refuse_ferrofluid("CMOS process-node scaling", technology, base_metrics)
    node = get_process_node(target_node_nm, corner=corner, temperature_k=temperature_k)
    tech = technology.lower().strip()

    # Reference 28nm baseline metrics if not provided
    defaults = {
        "memristor": {"energy_per_bit_j": 5e-15, "latency_ns": 20.0, "endurance_cycles": 1e10, "retention_s": 1e6, "area_density_gb_cm2": 5.0, "f2_factor": 12.0},
        "reram": {"energy_per_bit_j": 5e-15, "latency_ns": 20.0, "endurance_cycles": 1e10, "retention_s": 1e6, "area_density_gb_cm2": 5.0, "f2_factor": 12.0},
        "pcm": {"energy_per_bit_j": 2e-14, "latency_ns": 80.0, "endurance_cycles": 1e8, "retention_s": 1e9, "area_density_gb_cm2": 20.0, "f2_factor": 8.0},
        "ferroelectric": {"energy_per_bit_j": 8e-15, "latency_ns": 15.0, "endurance_cycles": 1e11, "retention_s": 1e7, "area_density_gb_cm2": 8.0, "f2_factor": 10.0},
        "fefet": {"energy_per_bit_j": 4e-15, "latency_ns": 12.0, "endurance_cycles": 1e11, "retention_s": 1e8, "area_density_gb_cm2": 12.0, "f2_factor": 8.0},
        "spintronic": {"energy_per_bit_j": 3e-15, "latency_ns": 10.0, "endurance_cycles": 1e13, "retention_s": 1e9, "area_density_gb_cm2": 3.0, "f2_factor": 16.0},
        "mram": {"energy_per_bit_j": 3e-15, "latency_ns": 10.0, "endurance_cycles": 1e13, "retention_s": 1e9, "area_density_gb_cm2": 3.0, "f2_factor": 16.0},
        "ftj": {"energy_per_bit_j": 1.5e-15, "latency_ns": 8.0, "endurance_cycles": 1e12, "retention_s": 1e7, "area_density_gb_cm2": 15.0, "f2_factor": 6.0},
        "photonic": {"energy_per_bit_j": 5e-16, "latency_ns": 2.0, "endurance_cycles": 1e15, "retention_s": None, "area_density_gb_cm2": 0.5, "f2_factor": 1e4},
        "ncfet": {"energy_per_bit_j": 3e-16, "latency_ns": 0.05, "endurance_cycles": 1e15, "retention_s": None, "area_density_gb_cm2": 25.0, "f2_factor": 4.0},
        "tfet": {"energy_per_bit_j": 2e-16, "latency_ns": 0.2, "endurance_cycles": 1e15, "retention_s": None, "area_density_gb_cm2": 22.0, "f2_factor": 4.5},
        "tmd_fet": {"energy_per_bit_j": 2.5e-16, "latency_ns": 0.03, "endurance_cycles": 1e15, "retention_s": None, "area_density_gb_cm2": 40.0, "f2_factor": 3.5},
    }
    base = dict(defaults.get(tech, {"energy_per_bit_j": 1e-14, "latency_ns": 50.0, "endurance_cycles": 1e8, "retention_s": 1e5, "area_density_gb_cm2": 5.0, "f2_factor": 12.0}))
    if base_metrics:
        base.update(base_metrics)

    scale = 28.0 / max(float(target_node_nm), 0.5)
    f_um = float(target_node_nm) * 1e-3

    # Photonic wavelength doesn't shrink with CMOS nm; electronics around it do
    if tech == "photonic":
        area_scale = scale ** 0.3
        energy_scale = (node.vdd.value / 0.9) ** 1.2
        latency_scale = 1.0
    else:
        area_scale = scale ** 1.75
        energy_scale = (1.0 / scale) ** 0.85 * (node.vdd.value / 0.9) ** 1.5
        # Interconnect RC partially offsets intrinsic speedup below 10nm
        rc_penalty = 1.0 + 0.06 * max(0.0, 10.0 - target_node_nm)
        latency_scale = (node.fo4_delay_ps.value / 11.0) ** 0.5 * rc_penalty

    cell_area_um2 = base.get("f2_factor", 10.0) * (f_um ** 2)
    access_id_ua = float(np.atleast_1d(mosfet_iv(node, node.vdd.value, node.vdd.value, width_nm=4.0 * target_node_nm))[0]) * 1e6

    return {
        "technology": technology,
        "node_nm": float(target_node_nm),
        "architecture": node.architecture.value,
        "corner": node.corner.value,
        "temperature_k": float(temperature_k),
        "vdd_v": node.vdd.value,
        "cell_area_um2": float(cell_area_um2),
        "access_transistor_ion_ua": float(access_id_ua),
        "wire_r_ohm_per_um": node.wire_r_ohm_per_um.value,
        "wire_c_ff_per_um": node.wire_c_ff_per_um.value,
        "energy_per_bit_j": float(base["energy_per_bit_j"] * energy_scale),
        "latency_ns": float(base["latency_ns"] * latency_scale),
        "endurance_cycles": base.get("endurance_cycles"),
        "retention_s": base.get("retention_s"),
        "area_density_gb_cm2": float(base["area_density_gb_cm2"] * area_scale),
        "sram_bitcell_um2": node.sram_bitcell_um2.value,
        "fo4_delay_ps": node.fo4_delay_ps.value,
    }


def sweep_technology_nodes(
    nodes_nm: list[float],
    technology: str = "cmos",
    corner: str = "TT",
    temperature_k: float = 300.0,
) -> dict[str, Any]:
    """Sweep CMOS or experimental technology metrics across a list of nm nodes."""
    results: list[dict[str, Any]] = []
    for nm in nodes_nm:
        if technology.lower().strip() == "cmos":
            node = get_process_node(nm, corner=corner, temperature_k=temperature_k)
            ion_n = float(np.atleast_1d(mosfet_iv(node, node.vdd.value, node.vdd.value))[0]) * 1e6
            results.append({
                "node_nm": float(nm),
                "architecture": node.architecture.value,
                "vdd_v": node.vdd.value,
                "vth_n_v": node.vth_n.value,
                "fo4_delay_ps": node.fo4_delay_ps.value,
                "gate_density_kgates_mm2": node.gate_density_kgates_mm2.value,
                "sram_bitcell_um2": node.sram_bitcell_um2.value,
                "dynamic_energy_fj_per_gate": node.dynamic_energy_fj_per_gate.value,
                "leakage_nw_per_gate": node.leakage_nw_per_gate.value,
                "metal1_pitch_nm": node.metal1_pitch_nm.value,
                "wire_r_ohm_per_um": node.wire_r_ohm_per_um.value,
                "wire_c_ff_per_um": node.wire_c_ff_per_um.value,
                "nmos_ion_ua": ion_n,
            })
        else:
            results.append(scale_experimental_device(
                technology=technology,
                target_node_nm=nm,
                corner=corner,
                temperature_k=temperature_k,
            ))

    return {
        "technology": technology,
        "corner": corner,
        "temperature_k": temperature_k,
        "nodes_nm": [float(n) for n in nodes_nm],
        "sweep": results,
    }


__all__ = [
    "ProcessCorner",
    "TransistorArch",
    "ProcessNodeParams",
    "get_process_node",
    "mosfet_iv",
    "scale_experimental_device",
    "sweep_technology_nodes",
]
