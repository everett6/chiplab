"""Persistent ExperimentLedger using SQLite + JSON metadata.

Stores:
- experiment runs with design/material parameters (Provenance-aware)
- simulation metrics
- Pareto candidates
- calibration records
- manufacturing signoff summaries

Key methods:
- log_run(): Store a new experiment run
- get_run(): Retrieve a run by ID
- query_runs(): Query runs by various filters
- compare_runs(): Compare multiple runs
- export_summary(): Export summary data
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple

from chiplab.schemas.common import Provenance, TrackedValue, unwrap_value, get_provenance
from chiplab.schemas.design import DesignSpec


# Ordered from cheapest to closest-to-reality. "unknown" is only for legacy rows logged before
# fidelity was required; those rows are kept but never used for learning or review.
FIDELITY_LEVELS: tuple[str, ...] = (
    "analytical", "behavioral", "discrete_event", "specialized", "circuit_spice", "rtl_sim",
    "synthesis", "place_route", "signoff", "fpga", "published", "measured",
)
LEGACY_FIDELITY = "unknown"
_METRIC_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_EXTRA_COLUMNS = {
    "content_hash": "TEXT",
    "duplicate_of": "TEXT",
    "repeats": "INTEGER DEFAULT 1",
    "last_seen_at": "TEXT",
    "logged_by": "TEXT",
    "excluded_from_learning": "INTEGER DEFAULT 0",
    "exclusion_reason": "TEXT",
}


def default_db_path() -> str:
    return os.environ.get("CHIPLAB_DB_PATH", "chiplab_experiments.db")


def validate_fidelity(fidelity: str | None) -> str:
    if fidelity not in FIDELITY_LEVELS:
        raise ValueError(
            f"fidelity is required and must be one of {', '.join(FIDELITY_LEVELS)} "
            f"(got {fidelity!r})")
    return fidelity


def content_hash(design_name: str, design_version: str, params: Any, metrics: Any,
                 fidelity: str | None) -> str:
    blob = json.dumps([design_name, design_version, params, metrics, fidelity],
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


_COLS = ("run_id, design_id, design_name, design_version, created_at, params_json, "
         "metrics_json, provenance_json, fidelity, fidelity_disclaimer, content_hash, "
         "duplicate_of, repeats, logged_by, excluded_from_learning, exclusion_reason")


def _decode(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "run_id": row[0], "design_id": row[1], "design_name": row[2], "design_version": row[3],
        "created_at": row[4], "params": json.loads(row[5] or "{}"),
        "metrics": json.loads(row[6] or "{}"), "provenance": json.loads(row[7] or "{}"),
        "fidelity": row[8], "fidelity_disclaimer": row[9], "content_hash": row[10],
        "duplicate_of": row[11], "repeats": row[12] or 1, "logged_by": row[13],
        "excluded_from_learning": bool(row[14]), "exclusion_reason": row[15],
    }


class ExperimentLedger:
    """SQLite-backed persistent ledger for experiment runs."""
    
    def __init__(self, db_path: str | Path = "experiment_ledger.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=10, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
    
    def _create_tables(self) -> None:
        """Create all tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiment_runs (
                run_id TEXT PRIMARY KEY,
                design_id TEXT,
                design_name TEXT NOT NULL,
                design_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                params_json TEXT,
                metrics_json TEXT,
                provenance_json TEXT,
                fidelity TEXT,
                fidelity_disclaimer TEXT
            );
            
            CREATE TABLE IF NOT EXISTS calibration_records (
                calibration_id TEXT PRIMARY KEY,
                run_id TEXT,
                node TEXT,
                report_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id)
            );
            
            CREATE TABLE IF NOT EXISTS manufacturing_signoff (
                signoff_id TEXT PRIMARY KEY,
                run_id TEXT,
                status TEXT NOT NULL,
                reviewer TEXT,
                comments TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id)
            );
            
            CREATE TABLE IF NOT EXISTS pareto_candidates (
                run_id TEXT PRIMARY KEY,
                objective TEXT NOT NULL,
                rank INTEGER,
                dominated_by INTEGER DEFAULT NULL,
                FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id)
            );
        """)
        have = {r[1] for r in self._conn.execute("PRAGMA table_info(experiment_runs)")}
        for col, decl in _EXTRA_COLUMNS.items():
            if col not in have:
                self._conn.execute(f"ALTER TABLE experiment_runs ADD COLUMN {col} {decl}")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_hash ON experiment_runs(content_hash)")
        self._conn.commit()

    def record(
        self,
        design: DesignSpec,
        params: dict[str, Any],
        metrics: dict[str, Any],
        fidelity: str | None,
        fidelity_disclaimer: str | None = None,
        run_id: str | None = None,
        logged_by: str | None = None,
    ) -> dict[str, Any]:
        """Log a run. Fidelity is required. An identical run (same design, params, metrics and
        fidelity) is not stored again: the original's repeat count goes up and its id is
        returned with duplicate=True."""
        validate_fidelity(fidelity)
        h = content_hash(design.name, design.version, json.loads(json.dumps(params, default=str)),
                         json.loads(json.dumps(metrics, default=str)), fidelity)
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        row = self._conn.execute(
            "SELECT run_id, repeats FROM experiment_runs WHERE content_hash = ? "
            "AND duplicate_of IS NULL ORDER BY created_at LIMIT 1", (h,)).fetchone()
        if row is not None and (run_id is None or run_id == row[0]):
            self._conn.execute(
                "UPDATE experiment_runs SET repeats = COALESCE(repeats, 1) + 1, last_seen_at = ? "
                "WHERE run_id = ?", (now, row[0]))
            self._conn.commit()
            return {"run_id": row[0], "duplicate": True, "repeats": (row[1] or 1) + 1}
        rid = self.log_run(design, params, metrics, fidelity, fidelity_disclaimer, run_id,
                           _hash=h, _logged_by=logged_by)
        return {"run_id": rid, "duplicate": False, "repeats": 1}

    def migrate_legacy(self, dry_run: bool = False) -> dict[str, Any]:
        """Non-destructive cleanup of rows logged before fidelity was required.

        Rows without a fidelity get fidelity "unknown" and are excluded from learning. Every
        row gets a content hash; later copies of an identical row are marked duplicate_of the
        first one. Nothing is deleted.
        """
        rows = self._conn.execute(
            "SELECT run_id, design_name, design_version, params_json, metrics_json, fidelity, "
            "content_hash, duplicate_of FROM experiment_runs ORDER BY created_at, rowid"
        ).fetchall()
        first: dict[str, str] = {}
        unknown = dup = hashed = 0
        updates: list[tuple[Any, ...]] = []
        for rid, name, ver, pj, mj, fid, old_h, old_dup in rows:
            try:
                params, metrics = json.loads(pj or "{}"), json.loads(mj or "{}")
            except json.JSONDecodeError:
                params, metrics = pj, mj
            new_fid = fid if fid else LEGACY_FIDELITY
            h = content_hash(name, ver, params, metrics, new_fid)
            dup_of = old_dup or (first.get(h) if first.get(h) != rid else None)
            first.setdefault(h, rid)
            reasons = []
            if not fid:
                unknown += 1
                reasons.append("fidelity not recorded at logging time")
            if dup_of and not old_dup:
                dup += 1
            if dup_of:
                reasons.append(f"duplicate of {dup_of}")
            if not old_h:
                hashed += 1
            updates.append((new_fid, h, dup_of, 1 if reasons else 0,
                            "; ".join(reasons) or None, rid))
        if not dry_run:
            self._conn.executemany(
                "UPDATE experiment_runs SET fidelity = ?, content_hash = ?, duplicate_of = ?, "
                "excluded_from_learning = MAX(COALESCE(excluded_from_learning, 0), ?), "
                "exclusion_reason = COALESCE(exclusion_reason, ?) WHERE run_id = ?", updates)
            self._conn.commit()
        return {"rows": len(rows), "marked_unknown_fidelity": unknown,
                "marked_duplicate": dup, "hashed": hashed, "deleted": 0, "dry_run": dry_run}

    def log_run(
        self,
        design: DesignSpec,
        params: dict[str, Any],
        metrics: dict[str, Any],
        fidelity: str | None = None,
        fidelity_disclaimer: str | None = None,
        run_id: str | None = None,
        *,
        _hash: str | None = None,
        _logged_by: str | None = None,
    ) -> str:
        """Log a new experiment run with design, parameters, and metrics.
        
        Args:
            design: The DesignSpec for this run
            params: Design/material parameters (will store with Provenance tracking)
            metrics: Simulation/experiment metrics
            fidelity: Fidelity level (e.g., 'analytical', 'discrete_event')
            fidelity_disclaimer: Disclaimer text
            run_id: Optional custom run ID; if None, auto-generate
        
        Returns:
            The run_id of the stored run
        """
        validate_fidelity(fidelity)
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if run_id is None:
            run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        
        # Store design info
        design_id = f"{design.name}_{design.version}"
        spec_json = design.model_dump_json(indent=2)
        
        # Serialize params with provenance info
        params_serialized = json.dumps(params, default=str)
        
        # Serialize metrics
        metrics_serialized = json.dumps(metrics, default=str)
        
        # Get provenance for parameters
        provenance_data = self._extract_provenance(params)
        provenance_serialized = json.dumps(provenance_data, default=str)
        
        if _hash is None:
            _hash = content_hash(design.name, design.version, json.loads(params_serialized),
                                 json.loads(metrics_serialized), fidelity)
        if self._conn.execute("SELECT 1 FROM experiment_runs WHERE run_id = ?",
                              (run_id,)).fetchone():
            raise ValueError(f"run_id {run_id!r} already exists; runs are never overwritten")
        self._conn.execute(
            """INSERT INTO experiment_runs
               (run_id, design_id, design_name, design_version, created_at,
                params_json, metrics_json, provenance_json, fidelity, fidelity_disclaimer,
                content_hash, logged_by, last_seen_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                design_id,
                design.name,
                design.version,
                now,
                params_serialized,
                metrics_serialized,
                provenance_serialized,
                fidelity,
                fidelity_disclaimer,
                _hash,
                _logged_by,
                now,
            ),
        )
        self._conn.commit()
        
        return run_id
    
    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a run by ID.
        
        Returns:
            dict with run data including params, metrics, provenance, or None if not found
        """
        row = self._conn.execute(f"SELECT {_COLS} FROM experiment_runs WHERE run_id = ?",
                                 (run_id,)).fetchone()
        return None if row is None else _decode(row)
    
    def query_runs(
        self,
        design_name: str | None = None,
        fidelity: str | None = None,
        min_metric: dict[str, float] | None = None,
        max_metric: dict[str, float] | None = None,
        limit: int = 50,
        include_duplicates: bool = False,
        learnable_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Query runs by various filters.
        
        Args:
            design_name: Filter by design name
            fidelity: Filter by fidelity level
            min_metric: Minimum metric values (dict of metric_name -> min_value)
            max_metric: Maximum metric values (dict of metric_name -> max_value)
            limit: Maximum number of results
        
        Returns:
            List of run dicts matching the criteria
        """
        query = f"SELECT {_COLS} FROM experiment_runs WHERE 1=1"
        params: list[Any] = []
        if not include_duplicates:
            query += " AND duplicate_of IS NULL"
        if learnable_only:
            query += " AND COALESCE(excluded_from_learning, 0) = 0 AND fidelity != ?"
            params.append(LEGACY_FIDELITY)
        for name in list(min_metric or {}) + list(max_metric or {}):
            if not _METRIC_NAME.match(str(name)):
                raise ValueError(f"invalid metric name {name!r}")
        
        if design_name:
            query += " AND design_name = ?"
            params.append(design_name)
        
        if fidelity:
            query += " AND fidelity = ?"
            params.append(fidelity)
        
        if min_metric:
            for metric_name, min_val in min_metric.items():
                query += f" AND json_extract(metrics_json, '$.{metric_name}') >= ?"
                params.append(min_val)
        
        if max_metric:
            for metric_name, max_val in max_metric.items():
                query += f" AND json_extract(metrics_json, '$.{metric_name}') <= ?"
                params.append(max_val)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [_decode(r) for r in self._conn.execute(query, params).fetchall()]
    
    def compare_runs(
        self, run_ids: list[str], metric: str | None = None
    ) -> dict[str, Any]:
        """Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            metric: Optional specific metric to compare by; if None, all metrics compared
        
        Returns:
            Comparison dict with metrics deltas, winners, etc.
        """
        runs = []
        for rid in run_ids:
            run = self.get_run(rid)
            if run is not None:
                runs.append(run)
        
        if len(runs) < 2:
            return {"error": "Need at least 2 runs to compare"}
        
        # Get all metric keys from all runs
        all_metric_keys: set[str] = set()
        for run in runs:
            all_metric_keys.update(run["metrics"].keys())
        
        if metric:
            all_metric_keys = {metric}
        
        comparison = {
            "run_ids": run_ids,
            "metric_deltas": {},
            "winner_by_metric": {},
            "summary": "",
        }
        
        for metric_key in all_metric_keys:
            values: dict[str, float | int] = {}
            for i, run in enumerate(runs):
                # Use run index + design info to avoid key collisions
                run_name = f"run_{i}: {run['design_name']} v{run['design_version']}"
                val = run["metrics"].get(metric_key)
                if val is not None:
                    values[run_name] = val
            
            if len(values) >= 2:
                sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)
                winner = sorted_values[0][0]
                comparison["metric_deltas"][metric_key] = values
                comparison["winner_by_metric"][metric_key] = winner
                comparison["summary"] += f"{metric_key}: winner is {winner} with value {values.get(winner, 0)}\n"
        
        return comparison
    
    def export_summary(self) -> dict[str, Any]:
        """Export summary of all runs in the ledger.
        
        Returns:
            Summary dict with counts, ranges, and aggregated metrics
        """
        cursor = self._conn.execute(
            """SELECT fidelity, fidelity_disclaimer, metrics_json
               FROM experiment_runs WHERE duplicate_of IS NULL"""
        )
        rows = cursor.fetchall()
        total, dups, excluded = self._conn.execute(
            "SELECT COUNT(*), SUM(duplicate_of IS NOT NULL), "
            "SUM(COALESCE(excluded_from_learning, 0)) FROM experiment_runs").fetchone()

        summary = {
            "total_rows": total,
            "duplicate_rows": dups or 0,
            "excluded_from_learning": excluded or 0,
            "total_runs": len(rows),
            "by_fidelity": {},
            "metric_ranges": {},
            "designs": set(),
        }
        
        all_metric_keys: set[str] = set()
        
        for row in rows:
            fidelity = row[0] or "unknown"
            summary["by_fidelity"][fidelity] = summary["by_fidelity"].get(fidelity, 0) + 1
            
            try:
                metrics = json.loads(row[2] or "{}")
                all_metric_keys.update(metrics.keys())
                summary["designs"].add(metrics.get("design_name", "unknown"))
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Compute metric ranges
        if all_metric_keys:
            for metric_key in all_metric_keys:
                values: list[float | int] = []
                # Re-query to get all values
                for row in rows:
                    try:
                        metrics = json.loads(row[2] or "{}")
                        if metric_key in metrics:
                            values.append(float(metrics[metric_key]))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
                
                if values:
                    summary["metric_ranges"][metric_key] = {
                        "min": min(values),
                        "max": max(values),
                        "count": len(values),
                    }
        
        summary["designs"] = list(summary["designs"])
        return summary
    
    def _extract_provenance(self, params: dict[str, Any]) -> dict[str, Any]:
        """Extract provenance information from parameters.
        
        Args:
            params: Parameter dict, possibly containing TrackedValue objects
        
        Returns:
            Dict with provenance tracking info
        """
        result: dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, TrackedValue):
                result[key] = {
                    "value": unwrap_value(value),
                    "unit": value.unit,
                    "provenance": value.provenance.value if hasattr(value.provenance, 'value') else str(value.provenance),
                    "source": value.source,
                    "uncertainty_pct": value.uncertainty_pct,
                    "notes": value.notes,
                }
            elif isinstance(value, dict):
                result[key] = self._extract_provenance(value)
            else:
                result[key] = value
        return result
    
    def __del__(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()