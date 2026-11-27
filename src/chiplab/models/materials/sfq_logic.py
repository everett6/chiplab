"""
Single Flux Quantum (SFQ) Logic Model
======================================

Implements SFQ digital logic circuits based on Josephson junctions.
Supports RSFQ (Rapid SFQ), ERSFQ (Energy-efficient RSFQ),
and RQL (Reciprocal Quantum Logic) families.

References:
- Likharev & Semenov (1991) - RSFQ logic/memory family
- Mukhanov (2011) - Energy-efficient SFQ circuits
- Olaya et al. (2017) - Reciprocal Quantum Logic (RQL)
- VERILOG-A models for SFQ (Hypres, MIT LL, AIST)
- IARPA SuperTools design kit
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np

from chiplab.schemas.common import Provenance, TrackedValue
from chiplab.core.gpu import GPUBackend, get_gpu_backend as get_backend


class SFQFamily(Enum):
    """SFQ logic families."""
    RSFQ = "rsfq"           # Rapid SFQ (biased with DC current)
    ERSFQ = "ersfq"         # Energy-efficient RSFQ (AC power)
    RQL = "rql"             # Reciprocal Quantum Logic (AC power, differential)
    eSFQ = "esfq"           # Energy-efficient SFQ (no static power)
    DC_SFQ = "dc_sfq"       # DC-powered SFQ


class JJModel(Enum):
    """Josephson junction models."""
    RSJ = "rsj"             # Resistively Shunted Junction (classical)
    QUANTUM = "quantum"     # Quantum phase dynamics
    CAPACITIVE = "capacitive"  # With capacitance (Stewart-McCumber)


@dataclass
class JJParams:
    """Josephson junction parameters with provenance."""
    # Critical current
    ic: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=100e-6,  # A
        provenance=Provenance.MEASURED,
        metadata={"description": "Critical current", "unit": "A"}
    ))

    # Normal resistance
    rn: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=2.0,  # ohm
        provenance=Provenance.MEASURED,
        metadata={"description": "Normal state resistance", "unit": "ohm"}
    ))

    # Capacitance
    c: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.5e-12,  # F
        provenance=Provenance.MEASURED,
        metadata={"description": "Junction capacitance", "unit": "F"}
    ))

    # Gap voltage
    vg: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=2.8e-3,  # V (Nb)
        provenance=Provenance.MEASURED,
        metadata={"description": "Superconducting gap voltage", "unit": "V"}
    ))

    # McCumber parameter
    beta_c: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1.0,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Stewart-McCumber parameter beta_c = 2*pi*Ic*Rn^2*C/Phi0"}
    ))

    # Inductance (for geometric inductance)
    inductance: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=10e-12,  # H
        provenance=Provenance.MEASURED,
        metadata={"description": "Loop inductance", "unit": "H"}
    ))

    # Operating temperature
    temperature: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=4.2,  # K
        provenance=Provenance.MEASURED,
        metadata={"description": "Operating temperature", "unit": "K"}
    ))

    # Process variations
    ic_variation: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.05,  # 5% std
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Critical current variation (std)"}
    ))
    rn_variation: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.05,
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Normal resistance variation (std)"}
    ))


@dataclass
class SFQGateParams:
    """SFQ gate/cell parameters."""
    # Gate type
    gate_type: str = "JTL"  # JTL, SPLIT, MERGE, AND, OR, XOR, NOT, DFF, NDRO

    # Bias currents
    bias_current: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.8,  # Normalized to Ic
        provenance=Provenance.MEASURED,
        metadata={"description": "Bias current (fraction of Ic)"}
    ))

    # Input/output inductances
    l_in: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=5e-12,  # H
        provenance=Provenance.MEASURED,
        metadata={"description": "Input inductance", "unit": "H"}
    ))
    l_out: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=5e-12,  # H
        provenance=Provenance.MEASURED,
        metadata={"description": "Output inductance", "unit": "H"}
    ))

    # JTL section inductance
    l_jtl: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=2e-12,  # H
        provenance=Provenance.MEASURED,
        metadata={"description": "JTL section inductance", "unit": "H"}
    ))

    # Timing
    setup_time: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=2e-12,  # s
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Setup time", "unit": "s"}
    ))
    hold_time: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e-12,  # s
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Hold time", "unit": "s"}
    ))
    propagation_delay: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=3e-12,  # s
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Propagation delay", "unit": "s"}
    ))

    # Power
    static_power: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=0.0,  # W (for ERSFQ/RQL)
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Static power dissipation", "unit": "W"}
    ))
    dynamic_power_per_fluxon: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=1e-18,  # J per fluxon
        provenance=Provenance.ESTIMATED,
        metadata={"description": "Energy per fluxon", "unit": "J"}
    ))


@dataclass
class SFQState:
    """
    SFQ circuit state with full provenance tracking.

    Tracks fluxon positions, phases, and logic states.
    """
    # Circuit parameters
    family: SFQFamily = SFQFamily.RSFQ
    jj_params: JJParams = field(default_factory=JJParams)
    gate_params: dict[str, SFQGateParams] = field(default_factory=dict)

    # Flux quantum
    phi0: float = 2.067833848e-15  # Wb (h/2e)

    # Circuit state
    # For each node: phase (normalized to 2pi), voltage, current
    node_phases: dict[str, float] = field(default_factory=dict)
    node_voltages: dict[str, float] = field(default_factory=dict)
    node_currents: dict[str, float] = field(default_factory=dict)

    # Fluxon tracking
    fluxon_positions: list[dict] = field(default_factory=list)  # List of {node, time, direction}
    fluxon_count: int = 0

    # Logic state (for digital simulation)
    logic_state: dict[str, int] = field(default_factory=dict)  # 0 or 1

    # Clock
    clock_phase: float = 0.0  # For AC-powered families (ERSFQ, RQL)
    clock_frequency: TrackedValue = field(default_factory=lambda: TrackedValue(
        value=10e9,  # Hz
        provenance=Provenance.MEASURED,
        metadata={"description": "Clock frequency", "unit": "Hz"}
    ))

    # History
    voltage_history: dict[str, list[float]] = field(default_factory=dict)
    phase_history: dict[str, list[float]] = field(default_factory=dict)
    fluxon_history: list[dict] = field(default_factory=list)

    def copy(self) -> SFQState:
        """Create a deep copy."""
        new_state = SFQState()
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, TrackedValue):
                setattr(new_state, field_name, value.copy())
            elif isinstance(value, (JJParams, SFQGateParams, SFQFamily)):
                setattr(new_state, field_name, value)
            elif isinstance(value, (int, float)):
                setattr(new_state, field_name, value)
            elif isinstance(value, dict):
                setattr(new_state, field_name, {k: v.copy() if isinstance(v, list) else v for k, v in value.items()})
            elif isinstance(value, list):
                setattr(new_state, field_name, value.copy())
            else:
                setattr(new_state, field_name, value)
        return new_state


def sfq_josephson_current(
    phase: float,
    jj_params: JJParams,
    backend: GPUBackend | None = None
) -> float:
    """
    Josephson current-phase relation.

    I = Ic * sin(phi) for ideal junction
    With shunting: I = Ic * sin(phi) + V/Rn + C*dV/dt

    Args:
        phase: Phase difference across junction (rad)
        jj_params: JJParams
        backend: GPU backend

    Returns:
        Current in Amperes
    """
    ic = jj_params.ic.value
    return float(ic * np.sin(phase))


def sfq_junction_voltage(
    phase: float,
    dphase_dt: float,
    jj_params: JJParams,
    backend: GPUBackend | None = None
) -> float:
    """
    Josephson voltage-phase relation (second Josephson equation).

    V = (Phi0 / 2pi) * dphi/dt

    Args:
        phase: Phase difference (rad)
        dphase_dt: Time derivative of phase (rad/s)
        jj_params: JJParams
        backend: GPU backend

    Returns:
        Voltage in Volts
    """
    phi0 = 2.067833848e-15
    return float(phi0 / (2 * np.pi) * dphase_dt)


def sfq_rsj_step(
    state: SFQState,
    node: str,
    current_input: float,
    dt: float,
    backend: GPUBackend | None = None
) -> SFQState:
    """
    Single RSJ (Resistively Shunted Junction) time step.

    Solves: C*dV/dt + V/Rn + Ic*sin(phi) = I_in
    With: V = (Phi0/2pi) * dphi/dt

    Args:
        state: SFQState
        node: Node name
        current_input: Input current (A)
        dt: Time step (s)
        backend: GPU backend

    Returns:
        Updated state
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()
    jj = new_state.jj_params

    # Get current phase
    phi = new_state.node_phases.get(node, 0.0)

    # RSJ model: dphi/dt = (2pi/Phi0) * (I_in - Ic*sin(phi)) * Rn / (1 + beta_c * cos(phi))
    # Simplified: dphi/dt = (2pi/Phi0) * Rn * (I_in - Ic*sin(phi))
    phi0 = 2.067833848e-15
    ic = jj.ic.value
    rn = jj.rn.value

    # Current through junction
    i_jj = ic * np.sin(phi)

    # Voltage across junction
    v = rn * (current_input - i_jj)

    # Phase update
    dphi_dt = 2 * np.pi * v / phi0
    phi_new = phi + dphi_dt * dt

    # Normalize phase to [-pi, pi] for tracking, but keep unwrapped for voltage
    new_state.node_phases[node] = phi_new
    new_state.node_voltages[node] = v

    # Detect fluxon (phase slip of 2pi)
    if phi_new - phi > np.pi:  # Positive fluxon passed
        new_state.fluxon_count += 1
        new_state.fluxon_positions.append({
            "node": node,
            "time": len(new_state.fluxon_history) * dt,
            "direction": 1,
        })
        new_state.fluxon_history.append({
            "node": node,
            "time": len(new_state.fluxon_history) * dt,
            "direction": 1,
        })
    elif phi_new - phi < -np.pi:  # Negative fluxon
        new_state.fluxon_count -= 1
        new_state.fluxon_positions.append({
            "node": node,
            "time": len(new_state.fluxon_history) * dt,
            "direction": -1,
        })
        new_state.fluxon_history.append({
            "node": node,
            "time": len(new_state.fluxon_history) * dt,
            "direction": -1,
        })

    # Record history
    if node not in new_state.voltage_history:
        new_state.voltage_history[node] = []
        new_state.phase_history[node] = []
    new_state.voltage_history[node].append(v)
    new_state.phase_history[node].append(phi_new)

    return new_state


def sfq_jtl_step(
    state: SFQState,
    input_node: str,
    output_node: str,
    dt: float,
    backend: GPUBackend | None = None
) -> SFQState:
    """
    Josephson Transmission Line (JTL) step.

    JTL propagates fluxons between nodes via inductive coupling.

    Args:
        state: SFQState
        input_node: Input node name
        output_node: Output node name
        dt: Time step (s)
        backend: GPU backend

    Returns:
        Updated state
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()

    # Get JTL inductance
    l_jtl = new_state.gate_params.get("JTL", SFQGateParams()).l_jtl.value

    # Current through inductor
    phi_in = new_state.node_phases.get(input_node, 0.0)
    phi_out = new_state.node_phases.get(output_node, 0.0)
    phi0 = 2.067833848e-15

    # Inductive current: I = (Phi0/2pi) * (phi_in - phi_out) / L
    i_ind = phi0 / (2 * np.pi) * (phi_in - phi_out) / l_jtl

    # Update output node with inductive current
    new_state = sfq_rsj_step(new_state, output_node, i_ind, dt, backend)

    return new_state


def sfq_splitter_step(
    state: SFQState,
    input_node: str,
    output_nodes: list[str],
    dt: float,
    backend: GPUBackend | None = None
) -> SFQState:
    """
    SFQ splitter (fan-out) step.

    Distributes fluxon to multiple outputs.

    Args:
        state: SFQState
        input_node: Input node
        output_nodes: List of output node names
        dt: Time step (s)
        backend: GPU backend

    Returns:
        Updated state
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()

    # Fluxon arrives at input
    fluxon_arrived = abs(new_state.node_phases.get(input_node, 0)) > np.pi

    if fluxon_arrived:
        # Distribute to outputs
        for out_node in output_nodes:
            # Each output gets a fluxon (with some loss)
            new_state.node_phases[out_node] = new_state.node_phases.get(out_node, 0) + 2 * np.pi
            new_state.fluxon_count += 1
            new_state.fluxon_positions.append({
                "node": out_node,
                "time": len(new_state.fluxon_history) * dt,
                "direction": 1,
            })

    return new_state


def sfq_merger_step(
    state: SFQState,
    input_nodes: list[str],
    output_node: str,
    dt: float,
    backend: GPUBackend | None = None
) -> SFQState:
    """
    SFQ merger (fan-in) step.

    Combines fluxons from multiple inputs.

    Args:
        state: SFQState
        input_nodes: List of input node names
        output_node: Output node name
        dt: Time step (s)
        backend: GPU backend

    Returns:
        Updated state
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()

    # Count fluxons arriving at inputs
    fluxon_count = 0
    for in_node in input_nodes:
        if abs(new_state.node_phases.get(in_node, 0)) > np.pi:
            fluxon_count += 1

    # Merger outputs fluxon if any input has fluxon (OR logic)
    if fluxon_count > 0:
        new_state.node_phases[output_node] = new_state.node_phases.get(output_node, 0) + 2 * np.pi
        new_state.fluxon_count += 1
        new_state.fluxon_positions.append({
            "node": output_node,
            "time": len(new_state.fluxon_history) * dt,
            "direction": 1,
        })

    return new_state


def sfq_and_gate_step(
    state: SFQState,
    input_a: str,
    input_b: str,
    output: str,
    dt: float,
    backend: GPUBackend | None = None
) -> SFQState:
    """
    SFQ AND gate step.

    Outputs fluxon only if both inputs have fluxons simultaneously.

    Args:
        state: SFQState
        input_a: First input node
        input_b: Second input node
        output: Output node
        dt: Time step (s)
        backend: GPU backend

    Returns:
        Updated state
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()

    # Check for coincident fluxons
    phi_a = abs(new_state.node_phases.get(input_a, 0))
    phi_b = abs(new_state.node_phases.get(input_b, 0))

    if phi_a > np.pi and phi_b > np.pi:
        new_state.node_phases[output] = new_state.node_phases.get(output, 0) + 2 * np.pi
        new_state.fluxon_count += 1
        new_state.fluxon_positions.append({
            "node": output,
            "time": len(new_state.fluxon_history) * dt,
            "direction": 1,
        })
        new_state.logic_state[output] = 1
    else:
        new_state.logic_state[output] = 0

    return new_state


def sfq_or_gate_step(
    state: SFQState,
    input_a: str,
    input_b: str,
    output: str,
    dt: float,
    backend: GPUBackend | None = None
) -> SFQState:
    """
    SFQ OR gate step.

    Outputs fluxon if either input has fluxon.

    Args:
        state: SFQState
        input_a: First input node
        input_b: Second input node
        output: Output node
        dt: Time step (s)
        backend: GPU backend

    Returns:
        Updated state
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()

    phi_a = abs(new_state.node_phases.get(input_a, 0))
    phi_b = abs(new_state.node_phases.get(input_b, 0))

    if phi_a > np.pi or phi_b > np.pi:
        new_state.node_phases[output] = new_state.node_phases.get(output, 0) + 2 * np.pi
        new_state.fluxon_count += 1
        new_state.fluxon_positions.append({
            "node": output,
            "time": len(new_state.fluxon_history) * dt,
            "direction": 1,
        })
        new_state.logic_state[output] = 1
    else:
        new_state.logic_state[output] = 0

    return new_state


def sfq_not_gate_step(
    state: SFQState,
    input_node: str,
    output_node: str,
    dt: float,
    backend: GPUBackend | None = None
) -> SFQState:
    """
    SFQ NOT gate (inverter) step.

    Uses a loop with bias current to invert fluxon presence.

    Args:
        state: SFQState
        input_node: Input node
        output_node: Output node
        dt: Time step (s)
        backend: GPU backend

    Returns:
        Updated state
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()

    phi_in = abs(new_state.node_phases.get(input_node, 0))

    # Inverter: fluxon in -> no fluxon out (and vice versa with bias)
    # Simplified: output fluxon if no input fluxon (with clock bias)
    if phi_in < np.pi:
        # No input fluxon -> generate output fluxon from bias
        bias = new_state.gate_params.get("NOT", SFQGateParams()).bias_current.value
        ic = new_state.jj_params.ic.value
        if bias * ic > 0.5 * ic:  # Sufficient bias
            new_state.node_phases[output_node] = new_state.node_phases.get(output_node, 0) + 2 * np.pi
            new_state.fluxon_count += 1
            new_state.logic_state[output_node] = 1
        else:
            new_state.logic_state[output_node] = 0
    else:
        new_state.logic_state[output_node] = 0

    return new_state


def sfq_dff_step(
    state: SFQState,
    data_node: str,
    clock_node: str,
    output_q: str,
    output_qn: str,
    dt: float,
    backend: GPUBackend | None = None
) -> SFQState:
    """
    SFQ D Flip-Flop (DFF) step.

    Edge-triggered flip-flop using SFQ logic.

    Args:
        state: SFQState
        data_node: Data input
        clock_node: Clock input
        output_q: Q output
        output_qn: Q_bar output
        dt: Time step (s)
        backend: GPU backend

    Returns:
        Updated state
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()

    # Detect clock edge (fluxon arrival)
    phi_clk = abs(new_state.node_phases.get(clock_node, 0))
    phi_data = abs(new_state.node_phases.get(data_node, 0))

    clock_edge = phi_clk > np.pi

    if clock_edge:
        # Latch data
        if phi_data > np.pi:
            new_state.node_phases[output_q] = new_state.node_phases.get(output_q, 0) + 2 * np.pi
            new_state.logic_state[output_q] = 1
            new_state.logic_state[output_qn] = 0
        else:
            new_state.node_phases[output_qn] = new_state.node_phases.get(output_qn, 0) + 2 * np.pi
            new_state.logic_state[output_q] = 0
            new_state.logic_state[output_qn] = 1

    return new_state


def sfq_ndro_step(
    state: SFQState,
    set_node: str,
    reset_node: str,
    output_q: str,
    output_qn: str,
    dt: float,
    backend: GPUBackend | None = None
) -> SFQState:
    """
    SFQ Non-Destructive Readout (NDRO) cell step.

    Memory cell that retains state after readout.

    Args:
        state: SFQState
        set_node: Set input
        reset_node: Reset input
        output_q: Q output
        output_qn: Q_bar output
        dt: Time step (s)
        backend: GPU backend

    Returns:
        Updated state
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()

    phi_set = abs(new_state.node_phases.get(set_node, 0))
    phi_reset = abs(new_state.node_phases.get(reset_node, 0))

    if phi_set > np.pi:
        new_state.logic_state[output_q] = 1
        new_state.logic_state[output_qn] = 0
    elif phi_reset > np.pi:
        new_state.logic_state[output_q] = 0
        new_state.logic_state[output_qn] = 1

    # Generate output fluxons based on state
    if new_state.logic_state.get(output_q, 0) == 1:
        new_state.node_phases[output_q] = new_state.node_phases.get(output_q, 0) + 2 * np.pi
    else:
        new_state.node_phases[output_qn] = new_state.node_phases.get(output_qn, 0) + 2 * np.pi

    return new_state


def sfq_clock_generator(
    state: SFQState,
    dt: float,
    backend: GPUBackend | None = None
) -> SFQState:
    """
    Generate clock phases for AC-powered SFQ families (ERSFQ, RQL).

    Args:
        state: SFQState
        dt: Time step (s)
        backend: GPU backend

    Returns:
        Updated state with clock phase
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()

    f_clk = new_state.clock_frequency.value
    new_state.clock_phase = (new_state.clock_phase + 2 * np.pi * f_clk * dt) % (2 * np.pi)

    # For RQL: differential clock (clk and clk_bar)
    # For ERSFQ: single-phase AC power

    return new_state


def sfq_power_estimation(
    state: SFQState,
    n_fluxons: int,
    frequency: float,
    backend: GPUBackend | None = None
) -> dict:
    """
    Estimate power dissipation for SFQ circuit.

    Args:
        state: SFQState
        n_fluxons: Number of fluxons per cycle
        frequency: Operating frequency (Hz)
        backend: GPU backend

    Returns:
        Power breakdown dict
    """
    if backend is None:
        backend = get_backend()

    phi0 = 2.067833848e-15

    # Static power (for RSFQ with DC bias)
    static_power = 0.0
    if state.family == SFQFamily.RSFQ:
        # I_bias * V_bias per junction
        ibias = state.jj_params.ic.value * 0.8  # Typical bias
        vbias = ibias * state.jj_params.rn.value
        n_junctions = 100  # Estimate
        static_power = n_junctions * ibias * vbias

    # Dynamic power: E = n_fluxons * Phi0 * Ic per cycle
    dynamic_energy_per_cycle = n_fluxons * phi0 * state.jj_params.ic.value
    dynamic_power = dynamic_energy_per_cycle * frequency

    # AC power for ERSFQ/RQL
    ac_power = 0.0
    if state.family in [SFQFamily.ERSFQ, SFQFamily.RQL]:
        # AC power ~ C * V^2 * f for clock distribution
        c_load = 1e-12  # F
        v_clock = 2.5e-3  # V
        ac_power = c_load * v_clock**2 * frequency

    return {
        "static_power_w": static_power,
        "dynamic_power_w": dynamic_power,
        "ac_power_w": ac_power,
        "total_power_w": static_power + dynamic_power + ac_power,
        "energy_per_op_j": dynamic_energy_per_cycle,
        "fluxons_per_cycle": n_fluxons,
    }


def sfq_timing_analysis(
    state: SFQState,
    critical_path: list[str],
    backend: GPUBackend | None = None
) -> dict:
    """
    Perform timing analysis on SFQ circuit.

    Args:
        state: SFQState
        critical_path: List of node names in critical path
        backend: GPU backend

    Returns:
        Timing analysis results
    """
    if backend is None:
        backend = get_backend()

    total_delay = 0.0
    setup_violation = False
    hold_violation = False

    for i, node in enumerate(critical_path):
        gate_params = state.gate_params.get(node, SFQGateParams())
        total_delay += gate_params.propagation_delay.value

        if i > 0:
            prev_gate = state.gate_params.get(critical_path[i-1], SFQGateParams())
            # Check setup/hold
            if gate_params.setup_time.value > prev_gate.propagation_delay.value:
                setup_violation = True
            if gate_params.hold_time.value > prev_gate.propagation_delay.value:
                hold_violation = True

    max_freq = 1.0 / total_delay if total_delay > 0 else 0.0

    return {
        "total_delay_s": total_delay,
        "max_frequency_hz": max_freq,
        "setup_violation": setup_violation,
        "hold_violation": hold_violation,
        "slack_s": 1.0/state.clock_frequency.value - total_delay if state.clock_frequency.value > 0 else 0,
    }


def sfq_simulate_circuit(
    state: SFQState,
    circuit_netlist: dict,
    input_signals: dict[str, np.ndarray],
    dt: float,
    n_steps: int,
    backend: GPUBackend | None = None
) -> dict:
    """
    Simulate full SFQ circuit from netlist.

    Args:
        state: Initial SFQState
        circuit_netlist: Dict describing circuit connectivity
        input_signals: Dict of input node -> signal array (0/1 for fluxon presence)
        dt: Time step (s)
        n_steps: Number of simulation steps
        backend: GPU backend

    Returns:
        Simulation results
    """
    if backend is None:
        backend = get_backend()

    run_state = state.copy()
    output_traces = {node: [] for node in circuit_netlist.get("outputs", [])}

    for step in range(n_steps):
        # Apply input signals
        for node, signal in input_signals.items():
            if step < len(signal) and signal[step] == 1:
                run_state.node_phases[node] = run_state.node_phases.get(node, 0) + 2 * np.pi

        # Update clock for AC families
        if run_state.family in [SFQFamily.ERSFQ, SFQFamily.RQL]:
            run_state = sfq_clock_generator(run_state, dt, backend)

        # Process gates in topological order
        for gate_name, gate_info in circuit_netlist.get("gates", {}).items():
            gate_type = gate_info.get("type", "JTL")
            inputs = gate_info.get("inputs", [])
            outputs = gate_info.get("outputs", [])

            if gate_type == "JTL" and len(inputs) == 1 and len(outputs) == 1:
                run_state = sfq_jtl_step(run_state, inputs[0], outputs[0], dt, backend)
            elif gate_type == "SPLIT" and len(inputs) == 1:
                run_state = sfq_splitter_step(run_state, inputs[0], outputs, dt, backend)
            elif gate_type == "MERGE" and len(outputs) == 1:
                run_state = sfq_merger_step(run_state, inputs, outputs[0], dt, backend)
            elif gate_type == "AND" and len(inputs) == 2 and len(outputs) == 1:
                run_state = sfq_and_gate_step(run_state, inputs[0], inputs[1], outputs[0], dt, backend)
            elif gate_type == "OR" and len(inputs) == 2 and len(outputs) == 1:
                run_state = sfq_or_gate_step(run_state, inputs[0], inputs[1], outputs[0], dt, backend)
            elif gate_type == "NOT" and len(inputs) == 1 and len(outputs) == 1:
                run_state = sfq_not_gate_step(run_state, inputs[0], outputs[0], dt, backend)
            elif gate_type == "DFF" and len(inputs) == 2 and len(outputs) == 2:
                run_state = sfq_dff_step(run_state, inputs[0], inputs[1], outputs[0], outputs[1], dt, backend)
            elif gate_type == "NDRO" and len(inputs) == 2 and len(outputs) == 2:
                run_state = sfq_ndro_step(run_state, inputs[0], inputs[1], outputs[0], outputs[1], dt, backend)

        # Record outputs
        for out_node in circuit_netlist.get("outputs", []):
            fluxon_present = 1 if abs(run_state.node_phases.get(out_node, 0)) > np.pi else 0
            output_traces[out_node].append(fluxon_present)

    return {
        "output_traces": output_traces,
        "final_state": run_state,
        "fluxon_history": run_state.fluxon_history,
        "total_fluxons": run_state.fluxon_count,
    }


def sfq_ersfq_ac_power_step(
    state: SFQState,
    node: str,
    ac_amplitude: float,
    ac_frequency: float,
    dt: float,
    backend: GPUBackend | None = None
) -> SFQState:
    """
    ERSFQ AC power step - junction powered by AC current.

    I_bias(t) = I_ac * sin(2*pi*f_ac*t)

    Args:
        state: SFQState
        node: Node name
        ac_amplitude: AC current amplitude (A)
        ac_frequency: AC frequency (Hz)
        dt: Time step (s)
        backend: GPU backend

    Returns:
        Updated state
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()

    # AC bias current at this time step
    t = len(new_state.voltage_history.get(node, [0])) * dt
    i_ac = ac_amplitude * np.sin(2 * np.pi * ac_frequency * t)

    # RSJ step with AC bias
    new_state = sfq_rsj_step(new_state, node, i_ac, dt, backend)

    return new_state


def sfq_rql_differential_step(
    state: SFQState,
    node_p: str,
    node_n: str,
    dt: float,
    backend: GPUBackend | None = None
) -> SFQState:
    """
    RQL differential logic step.

    Uses differential pair of junctions with AC power.
    Logic encoded in phase difference: 0 = '0', pi = '1'

    Args:
        state: SFQState
        node_p: Positive node
        node_n: Negative node
        dt: Time step (s)
        backend: GPU backend

    Returns:
        Updated state
    """
    if backend is None:
        backend = get_backend()

    new_state = state.copy()

    # Differential phase
    phi_p = new_state.node_phases.get(node_p, 0)
    phi_n = new_state.node_phases.get(node_n, 0)
    phi_diff = phi_p - phi_n

    # RQL: AC power clocked, differential signaling
    # Simplified: propagate differential fluxon
    if abs(phi_diff) > np.pi:
        # Fluxon present in differential mode
        new_state.node_phases[node_p] = phi_p + 2 * np.pi
        new_state.node_phases[node_n] = phi_n - 2 * np.pi
        new_state.fluxon_count += 1

    return new_state


# Export key functions and classes
__all__ = [
    "SFQFamily",
    "JJModel",
    "JJParams",
    "SFQGateParams",
    "SFQState",
    "sfq_josephson_current",
    "sfq_junction_voltage",
    "sfq_rsj_step",
    "sfq_jtl_step",
    "sfq_splitter_step",
    "sfq_merger_step",
    "sfq_and_gate_step",
    "sfq_or_gate_step",
    "sfq_not_gate_step",
    "sfq_dff_step",
    "sfq_ndro_step",
    "sfq_clock_generator",
    "sfq_power_estimation",
    "sfq_timing_analysis",
    "sfq_simulate_circuit",
    "sfq_ersfq_ac_power_step",
    "sfq_rql_differential_step",
]