"""ChipLab Discovery Engine - Automated architecture discovery and design-space exploration.

Provides Latin Hypercube, Sobol, and evolutionary candidate generation across
compute, memory, interconnect, and novel materials domains, with Pareto-frontier
extraction (PPA: performance, power, area, thermal) and bottleneck identification.
"""

from __future__ import annotations

__all__ = [
    "Candidate",
    "DesignSpace",
    "ParetoFrontier",
    "BottleneckType",
    "explore_design_space",
    "identify_bottlenecks",
]