"""Ramulator 2.x backend: cycle-level DRAM command timing for one memory channel.

ChipLab sends a DRAM standard and an access stream (sequential or random 64 B lines with a
read/write mix); Ramulator simulates the controller (FR-FCFS, open row, MOP4CLXOR mapping,
all-bank refresh) against JEDEC timing presets. ChipLab takes the achieved fraction of the
channel's peak bandwidth and the loaded read latency, and applies them to the design's own
memory spec (channels are independent under cache-line interleaving).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ..schemas.common import FidelityLevel
from .base import RunnerBackend

# ChipLab memory technology -> Ramulator standard (runner's fixed table).
TECH_TO_STANDARD = {
    "HBM2": "HBM2", "HBM2E": "HBM2", "HBM3": "HBM3", "HBM3E": "HBM3", "HBM4": "HBM4",
    "LPDDR5": "LPDDR5", "LPDDR5X": "LPDDR5", "DDR4": "DDR4", "DDR5": "DDR5", "GDDR7": "GDDR7",
}
PATTERNS = ("sequential", "random")


def standard_for(technology: str | None) -> str | None:
    t = str(technology or "").upper().replace("-", "").replace(" ", "")
    return TECH_TO_STANDARD.get(t)


class RamulatorBackend(RunnerBackend):
    name: str = "ramulator"
    display_name: str = "Ramulator 2"
    fidelity: FidelityLevel = FidelityLevel.SPECIALIZED
    runner: str = "_ramulator_runner.py"
    env_var: str = "CHIPLAB_RAMULATOR_PYTHON"
    default_python: str = "~/.local/opt/ramulator-venv/bin/python"
    probe_module: str = "ramulator"

    def measure(self, technology: str, pattern: str = "sequential",
                write_fraction: float = 0.0, n_requests: int = 100_000,
                seed: int = 42) -> dict[str, Any]:
        """Simulate one channel of ``technology``; returns efficiency, latency, provenance."""
        std = standard_for(technology)
        if std is None:
            raise ValueError(f"Ramulator has no supported preset for memory technology "
                             f"{technology!r}; supported: {sorted(TECH_TO_STANDARD)}")
        if pattern not in PATTERNS:
            raise ValueError(f"pattern must be one of {PATTERNS}")
        if not 0.0 <= write_fraction <= 0.5:
            raise ValueError("write_fraction must be in [0, 0.5]")
        if not 1000 <= n_requests <= 2_000_000:
            raise ValueError("n_requests must be in [1000, 2000000]")
        return dict(_measure_cached(self.python(), std, pattern, round(write_fraction, 3),
                                    int(n_requests), int(seed)))


@lru_cache(maxsize=256)
def _measure_cached(python: str, std: str, pattern: str, write_fraction: float,
                    n_requests: int, seed: int) -> dict[str, Any]:
    raw = RamulatorBackend().call_runner({"standard": std, "pattern": pattern,
                                         "n_requests": n_requests,
                                         "write_fraction": write_fraction, "seed": seed})
    peak_mbps = raw["rate_mtps"] * raw["channel_width_bits"] / 8.0
    eff = raw["total_throughput_mbps"] / peak_mbps
    if not 0.0 < eff <= 1.0001:
        raise RuntimeError(f"Ramulator returned an implausible efficiency {eff:.3f}")
    return {
        "standard": std, "pattern": pattern, "write_fraction": write_fraction,
        "n_requests": n_requests, "seed": seed,
        "simulated_channel_peak_gbps": peak_mbps / 1000.0,
        "achieved_gbps": raw["total_throughput_mbps"] / 1000.0,
        "efficiency": min(eff, 1.0),
        "avg_read_latency_ns": raw["avg_read_latency_cycles"] * raw["tck_ps"] / 1000.0,
        "row_hit_rate": raw["row_hits"] / max(1, raw["row_hits"] + raw["row_misses"]
                                              + raw["row_conflicts"]),
        "config": {k: raw[k] for k in ("dram", "org_preset", "timing_preset", "controller",
                                       "addr_mapper", "scheduler", "row_policy", "refresh",
                                       "rate_mtps", "channel_width_bits")},
        "tool": f"Ramulator {raw.get('ramulator_version', '2.1')}",
    }
