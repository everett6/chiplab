# Chiplab Multiscale Modeling Module
"""Multiscale modeling for chip-scale simulation with automatic model reduction.

This module provides automatic model reduction techniques for multiscale simulation,
enabling efficient simulation of large devices (256 GB+) by calibrating larger
mathematical models from small detailed simulations.

Key components:
- Scale definitions and hierarchy (nanoparticle -> system)
- Model reduction (POD, DMD, Balanced Truncation, PGD, Neural Operators)
- Upscaling (homogenization, volume averaging)
- Downscaling (submodeling, adaptive refinement)
- Automatic calibration workflows
- End-to-end multiscale pipelines
"""

from __future__ import annotations

from .scales import Scale, ScaleDefinition, ScaleHierarchy, RepresentativeVolumeElement
from .reduction import (
    ProperOrthogonalDecomposition,
    DynamicModeDecomposition,
    BalancedTruncation,
    ProperGeneralizedDecomposition,
    NeuralOperatorReduction,
    ReducedOrderModel,
)
from .upscaling import (
    Homogenization,
    VolumeAveraging,
    AsymptoticExpansion,
    NumericalHomogenization,
    EffectivePropertyCalculator,
)
from .downscaling import (
    Submodeling,
    AdaptiveRefinement,
    ConcurrentMultiscale,
)
from .calibration import (
    CalibrationWorkflow,
    ParameterSpaceSampling,
    ExtrapolationDetector,
)
from .pipeline import (
    MultiscalePipeline,
    ScaleSelector,
)

__all__ = [
    "Scale",
    "ScaleDefinition",
    "ScaleHierarchy",
    "RepresentativeVolumeElement",
    "ProperOrthogonalDecomposition",
    "DynamicModeDecomposition",
    "BalancedTruncation",
    "ProperGeneralizedDecomposition",
    "NeuralOperatorReduction",
    "ReducedOrderModel",
    "Homogenization",
    "VolumeAveraging",
    "AsymptoticExpansion",
    "NumericalHomogenization",
    "EffectivePropertyCalculator",
    "Submodeling",
    "AdaptiveRefinement",
    "ConcurrentMultiscale",
    "CalibrationWorkflow",
    "ParameterSpaceSampling",
    "ExtrapolationDetector",
    "MultiscalePipeline",
    "ScaleSelector",
]
