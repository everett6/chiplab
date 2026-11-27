"""ChiLab experimental calibration laboratory.

This package provides a complete framework for experimental calibration,
including data import, parameter fitting, sensitivity analysis,
uncertainty quantification, and automated calibration pipelines.
"""

from . import data_import
from . import parameter_fitting
from . import sensitivity
from . import uncertainty

__all__ = [
    "data_import",
    "parameter_fitting",
    "sensitivity",
    "uncertainty",
]

__version__ = "0.1.0"
