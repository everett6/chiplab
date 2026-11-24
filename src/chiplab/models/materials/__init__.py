"""Materials module for chiplab - electrode array and ferrofluid device models."""

from __future__ import annotations

from chiplab.models.materials.electrode_array import (
    Port,
    ElectrodeArray,
    create_array,
    simultaneous_write,
    independent_read,
    write_read_interference,
    cross_port_coupling,
    frequency_dependent_impedance,
    state_retention,
    signal_noise,
)

__all__ = [
    "Port",
    "ElectrodeArray",
    "create_array",
    "simultaneous_write",
    "independent_read",
    "write_read_interference",
    "cross_port_coupling",
    "frequency_dependent_impedance",
    "state_retention",
    "signal_noise",
]
