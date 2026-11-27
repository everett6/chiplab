from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from chiplab.engine.digital_twin import _backend, TwinModel
from chiplab.schemas.common import Provenance
from chiplab.schemas.digital_twin import (
    DigitalTwinConfig,
    Objective,
    SearchParameter,
    TwinConstraints,
    TwinWorkload,
    UncertaintyPrior,
    CalibrationReport,
    CalibrationFactor,
)


class DigitalTwin:
    """Persistent DigitalTwin manager with geometry, materials, governing equations,
    calibration history, live state snapshots, drift/residual tracking, and JSON
    save/load persistence.

    Wraps the engine's TwinModel and adds persistence, calibration tracking, and
    drift monitoring capabilities.
    """

    DISCLAIMER = (
        "Digital-twin results come from a parametric analytical PPA model (alpha-power-law "
        "timing, CV^2f + leakage power, SRAM/logic density scaling, yield-based cost). They rank "
        "design points and expose sensitivities; they are not signoff. Calibrate with "
        "calibrate_from_eda_report / ingest_byo_pdk against real synthesis and P&R runs."
    )

    def __init__(
        self,
        config: DigitalTwinConfig | None = None,
        workload: TwinWorkload | None = None,
        calibration: dict[str, float] | None = None,
        blackboxes: list[Any] | None = None,
        ambient_c: float = 25.0,
        theta_ja: float | None = None,
        memory: Any | None = None,
        persist_path: str | Path | None = None,
    ) -> None:
        self.config = config or DigitalTwinConfig()
        self.twin = TwinModel(
            workload=workload,
            calibration=calibration,
            blackboxes=blackboxes,
            ambient_c=ambient_c,
            theta_ja=theta_ja,
            memory=memory,
        )
        self.persist_path = Path(persist_path) if persist_path else Path("digital_twin.json")
        self.calibration_history: list[CalibrationReport] = []
        self.live_state: dict[str, Any] = {}
        self.drift_tracking: list[dict[str, float]] = []
        self._param_hash: str | None = None
        self._last_evaluate_params: dict[str, Any] | None = None
        self._xp = _backend(force_pure_python=True)

    # --- JSON persistence ---

    def save(self, path: str | Path | None = None) -> None:
        """Save current state to JSON file."""
        path = Path(path) if path else self.persist_path
        data = {
            "config": self.config.model_dump(mode="json")
            if hasattr(self.config, "model_dump")
            else {},
            "live_state": self.live_state,
            "calibration_history": [
                c.model_dump(mode="json") for c in self.calibration_history
            ],
            "drift_tracking": self.drift_tracking,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        path.write_text(json.dumps(data, indent=2))
        self.persist_path = path

    def load(self, path: str | Path | None = None) -> None:
        """Load state from JSON file."""
        path = Path(path) if path else self.persist_path
        if not path.exists():
            raise FileNotFoundError(f"DigitalTwin persist file not found: {path}")
        data = json.loads(path.read_text())
        if "config" in data and data["config"]:
            self.config = DigitalTwinConfig(**data["config"])
        self.live_state = data.get("live_state", {})
        self.calibration_history = [
            CalibrationReport(**c) for c in data.get("calibration_history", [])
        ]
        self.drift_tracking = data.get("drift_tracking", [])
        self.persist_path = path

    # --- Calibration ---

    def calibrate(self, factors: dict[str, float], source: str = "manual", detail: str = "") -> CalibrationReport:
        """Apply calibration factors and record in history."""
        report = CalibrationReport(
            calibration_id=f"{datetime.utcnow().isoformat()}__{source}",
            factors=[
                CalibrationFactor(
                    name=k, value=v, provenance=Provenance.MEASURED, source=source, detail=detail
                )
                for k, v in factors.items()
            ],
            sources=[{"name": k, "value": v} for k, v in factors.items()],
        )
        self.calibration_history.append(report)
        self.twin.calib.update(factors)
        if self._last_evaluate_params is not None:
            drift = self._compute_drift(factors)
            self.drift_tracking.append(drift)
        return report

    def _compute_drift(self, new_factors: dict[str, float]) -> dict[str, float]:
        """Compute drift/residual between current and previous parameter states."""
        drift = {}
        for name, value in new_factors.items():
            prev_val = self.live_state.get(f"calib_{name}")
            if prev_val is not None:
                drift[name] = (value - prev_val) / max(prev_val, 1e-12)
            else:
                drift[name] = 0.0
        return drift

    # --- Evaluate ---

    def evaluate(self, params: dict[str, Any], calibration: dict[str, float] | None = None) -> dict[str, Any]:
        """Evaluate PPA model with given params and optional calibration.

        Uses the pure-Python backend by default for compatibility.

        Args:
            params: dict of geometry and operating parameters
            calibration: optional dict of calibration factors (overrides twin calib)

        Returns:
            dict of PPA model metrics
        """
        # Merge calibration: use provided calibration, fallback to twin's calib
        cal = dict(self.twin.calib)
        if calibration:
            cal.update(calibration)

        # Normalize defaults and string node names for partial param dicts
        node_map = {"N3": 0, "N5": 1, "N7": 2, "N12": 3, "N14": 3, "N28": 4, "N45": 5}
        norm_params: dict[str, Any] = {
            "node": 2,
            "cores": 16,
            "vlen": 256,
            "dlen": 256,
            "l1_kb": 32,
            "l2_kb": 256,
            "llc_mb": 16,
            "f_ghz": 2.0,
            "freq_ghz": 2.0,
            "vdd": 0.85,
            "utilization": 0.75,
            "ucie_lanes": 16,
            "mem_bw": 256,
            "noc_bw": 512,
        }
        norm_params.update(params)
        if isinstance(norm_params.get("node"), str):
            norm_params["node"] = node_map.get(norm_params["node"].upper(), 2)

        result = self.twin.evaluate(norm_params, cal, self._xp)
        if isinstance(result, dict):
            result = {
                **result,
                **({"performance": result["perf"]} if "perf" in result and "performance" not in result else {}),
                **({"power": result["power_w"]} if "power_w" in result and "power" not in result else {}),
                **({"area": result["area_mm2"]} if "area_mm2" in result and "area" not in result else {}),
                **({"cost": result["cost_usd"]} if "cost_usd" in result and "cost" not in result else {}),
                **({"temperature": result["temp_c"]} if "temp_c" in result and "temperature" not in result else {}),
            }
        # Update live state
        self.live_state["last_eval"] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "params": params,
            "metrics": result,
        }
        # Track parameter hash for drift detection
        param_str = json.dumps(params, sort_keys=True)
        self._param_hash = hashlib.sha256(param_str.encode()).hexdigest()[:16]
        self._last_evaluate_params = params
        return result

    # --- Shortcuts ---

    def eval_with_geometry(
        self,
        cores: int,
        vlen: int,
        dlen: int,
        l1_kb: int,
        l2_kb: int,
        llc_mb: float,
        vdd: float,
        freq_ghz: float,
        utilization: float,
        node: str,
        ucie_lanes: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate with explicit geometry parameters."""
        params = {
            "cores": cores,
            "vlen": vlen,
            "dlen": dlen,
            "l1_kb": l1_kb,
            "l2_kb": l2_kb,
            "llc_mb": llc_mb,
            "vdd": vdd,
            "freq_ghz": freq_ghz,
            "utilization": utilization,
            "node": node,
            "ucie_lanes": ucie_lanes,
        }
        params.update(kwargs)
        return self.evaluate(params)

    def snapshot(self) -> dict[str, Any]:
        """Return a live state snapshot."""
        return {
            "config_summary": {
                "strategy": self.config.strategy,
                "population": self.config.population,
                "generations": self.config.generations,
                "seed": self.config.seed,
                "objectives": [
                    {"name": o.name, "goal": o.goal} for o in self.config.objectives
                ],
            },
            "calibration_count": len(self.calibration_history),
            "drift_count": len(self.drift_tracking),
            "live_state_keys": list(self.live_state.keys()),
            "param_hash": self._param_hash,
            "last_eval_summary": {
                "timestamp": self.live_state.get("last_eval", {}).get("timestamp", ""),
                "params_keys": list(
                    self.live_state.get("last_eval", {}).get("params", {}).keys()
                ),
            }
            if self.live_state.get("last_eval")
            else {},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


__all__ = ["DigitalTwin"]