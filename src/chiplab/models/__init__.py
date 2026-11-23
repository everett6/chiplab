from __future__ import annotations

from typing import Any, Dict, List, Optional

from chiplab.models.memory import estimate_memory_chiplet
from chiplab.models.compute import estimate_compute_chiplet
from chiplab.models.interconnect import estimate_interconnect
from chiplab.models.packaging import estimate_packaging
from chiplab.models.power_thermal import estimate_power_thermal


__all__ = [
    "estimate_memory_chiplet",
    "estimate_compute_chiplet",
    "estimate_interconnect",
    "estimate_packaging",
    "estimate_power_thermal",
]


def test_imports() -> bool:
    """Verify that all model functions are importable."""
    try:
        from chiplab.models import (
            estimate_memory_chiplet,
            estimate_compute_chiplet,
            estimate_interconnect,
            estimate_packaging,
            estimate_power_thermal,
        )
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False


if __name__ == "__main__":
    success = test_imports()
    if success:
        print("All model imports successful!")
    else:
        print("Model import failed!")