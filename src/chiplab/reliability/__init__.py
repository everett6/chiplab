"""Reliability and aging models for ChipLab.

This module provides degradation and aging models for various memory and
compute technologies, all returning TrackedValue with Provenance tracking.
"""
from __future__ import annotations

from chiplab.schemas.common import Provenance, TrackedValue

__all__ = [
    "degradation",
    "aging",
]
