"""SQLite-backed persistence for ChipLab designs, simulations, sweeps, and comparisons."""

from __future__ import annotations

import sqlite3
import json
import time
import uuid
from pathlib import Path
from typing import Any

from chiplab.schemas.design import DesignSpec
from chiplab.schemas.simulation import ComparisonResult, SimulationResult, SweepResult


class ExperimentStore:
    """SQLite-backed persistence for ChipLab designs, simulations, sweeps, and comparisons."""

    def __init__(self, db_path: str | Path = "chiplab_experiments.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Tools run in worker threads and several agent processes share the file.
        self._conn = sqlite3.connect(str(self.db_path), timeout=10, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        """Create all tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS designs (
                design_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                parent_version TEXT,
                description TEXT,
                spec_yaml TEXT,
                spec_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS simulations (
                run_id TEXT PRIMARY KEY,
                design_id INTEGER NOT NULL,
                design_name TEXT NOT NULL,
                design_version TEXT NOT NULL,
                workload_name TEXT NOT NULL,
                fidelity TEXT NOT NULL,
                result_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (design_id) REFERENCES designs(design_id)
            );

            CREATE TABLE IF NOT EXISTS sweeps (
                sweep_id TEXT PRIMARY KEY,
                design_id INTEGER NOT NULL,
                fidelity TEXT,
                result_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (design_id) REFERENCES designs(design_id)
            );

            CREATE TABLE IF NOT EXISTS digital_twin_runs (
                run_id TEXT PRIMARY KEY,
                strategy TEXT,
                summary_json TEXT,
                trace_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS calibrations (
                calibration_id TEXT PRIMARY KEY,
                node TEXT,
                report_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS comparisons (
                comparison_id TEXT PRIMARY KEY,
                result_json TEXT,
                created_at TEXT NOT NULL
            );
        """)
        self._conn.commit()

    # --- Designs ---

    def save_design(self, design: DesignSpec, design_id: str | None = None) -> str:
        """Save a design spec to the database. Returns the design_id."""
        import yaml

        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        spec_yaml = yaml.dump(design.model_dump(), default_flow_style=False, sort_keys=False)
        spec_json = design.model_dump_json(indent=2) if hasattr(design, "model_dump_json") else json.dumps(design.model_dump())

        if design_id is None:
            design_id = f"{design.name}_{design.version}"

        self._conn.execute(
            """INSERT OR REPLACE INTO designs (design_id, name, version, parent_version, description, spec_yaml, spec_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (design_id, design.name, design.version, design.parent_version,
             design.description, spec_yaml, spec_json, now),
        )
        self._conn.commit()
        return design_id

    def get_design(self, design_id_or_name: str, version: str | None = None) -> tuple[str, DesignSpec]:
        """Get a design by ID or name. Returns (design_id, DesignSpec)."""
        cursor = self._conn.execute(
            """SELECT design_id, name, version, parent_version, description, spec_yaml, spec_json, created_at
               FROM designs WHERE design_id = ? OR name = ?""",
            (design_id_or_name, design_id_or_name),
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Design not found: {design_id_or_name}")
        design_id = row[0]
        design_spec = DesignSpec.model_validate(json.loads(row[6]))
        return (design_id, design_spec)

    def list_designs(self, name: str | None = None) -> list[dict]:
        """List designs, optionally filtered by name. Returns list of dicts."""
        if name:
            cursor = self._conn.execute(
                """SELECT design_id, name, version, parent_version, description, created_at
                   FROM designs WHERE name LIKE ? ORDER BY created_at DESC""",
                (f"%{name}%",),
            )
        else:
            cursor = self._conn.execute(
                """SELECT design_id, name, version, parent_version, description, created_at
                   FROM designs ORDER BY created_at DESC"""
            )
        rows = cursor.fetchall()
        return [
            {
                "design_id": row[0],
                "name": row[1],
                "version": row[2],
                "parent_version": row[3],
                "description": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]

    def get_design_history(self, name: str) -> list[dict]:
        """Get design history (all versions) by name. Returns list of dicts."""
        cursor = self._conn.execute(
            """SELECT design_id, name, version, parent_version, description, spec_yaml, spec_json, created_at
               FROM designs WHERE name = ? ORDER BY version""",
            (name,),
        )
        rows = cursor.fetchall()
        return [
            {
                "design_id": row[0],
                "name": row[1],
                "version": row[2],
                "parent_version": row[3],
                "description": row[4],
                "spec_yaml": row[5],
                "spec_json": row[6],
                "created_at": row[7],
            }
            for row in rows
        ]

    # --- Simulations ---

    def save_simulation(self, result: SimulationResult) -> str:
        """Save a simulation result to the database. Returns the run_id."""
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        result_json = result.model_dump_json()

        if not result.run_id:
            result.run_id = (f"{result.design_name}_{result.design_version}_{int(time.time())}_"
                             f"{uuid.uuid4().hex[:6]}")

        fidelity_str = result.fidelity.value if hasattr(result.fidelity, "value") else str(result.fidelity)
        self._conn.execute(
            """INSERT OR REPLACE INTO simulations (run_id, design_id, design_name, design_version, workload_name, fidelity, result_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (result.run_id, result.design_id, result.design_name, result.design_version,
             result.workload_name, fidelity_str, result_json, now),
        )
        self._conn.commit()
        return result.run_id

    def get_simulation(self, run_id: str) -> SimulationResult:
        """Get a simulation result by run_id."""
        cursor = self._conn.execute(
            """SELECT run_id, design_id, design_name, design_version, workload_name, fidelity, result_json, created_at
               FROM simulations WHERE run_id = ?""",
            (run_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Simulation not found: {run_id}")
        return SimulationResult.model_validate(json.loads(row[6]))

    def list_simulations(self, design_id: str | None = None, limit: int = 50) -> list[dict]:
        """List simulation results, optionally filtered by design_id."""
        if design_id:
            cursor = self._conn.execute(
                """SELECT run_id, design_id, design_name, design_version, workload_name, fidelity, created_at
                   FROM simulations WHERE design_id = ? ORDER BY created_at DESC LIMIT ?""",
                (design_id, limit),
            )
        else:
            cursor = self._conn.execute(
                """SELECT run_id, design_id, design_name, design_version, workload_name, fidelity, created_at
                   FROM simulations ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            )
        rows = cursor.fetchall()
        return [
            {
                "run_id": row[0],
                "design_id": row[1],
                "design_name": row[2],
                "design_version": row[3],
                "workload_name": row[4],
                "fidelity": row[5],
                "created_at": row[6],
            }
            for row in rows
        ]

    # --- Sweeps ---

    def save_sweep(self, sweep: SweepResult, design_id: str) -> str:
        """Save a sweep result to the database. Returns the sweep_id."""
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        result_json = sweep.model_dump_json()

        if not sweep.sweep_id:
            sweep.sweep_id = f"sweep_{design_id}_{int(time.time())}"

        fidelity_str = sweep.fidelity.value if hasattr(sweep.fidelity, "value") else str(sweep.fidelity)
        self._conn.execute(
            """INSERT OR REPLACE INTO sweeps (sweep_id, design_id, fidelity, result_json, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (sweep.sweep_id, design_id, fidelity_str, result_json, now),
        )
        self._conn.commit()
        return sweep.sweep_id

    def get_sweep(self, sweep_id: str) -> SweepResult:
        """Get a sweep result by sweep_id."""
        cursor = self._conn.execute(
            """SELECT sweep_id, design_id, fidelity, result_json, created_at
               FROM sweeps WHERE sweep_id = ?""",
            (sweep_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Sweep not found: {sweep_id}")
        return SweepResult.model_validate(json.loads(row[3]))

    # --- Comparisons ---

    def save_comparison(self, comp: ComparisonResult) -> str:
        """Save a comparison result to the database. Returns the comparison_id."""
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        comp_dict = comp.model_dump() if hasattr(comp, "model_dump") else comp.__dict__
        result_json = json.dumps(comp_dict)

        self._conn.execute(
            """INSERT OR REPLACE INTO comparisons (comparison_id, result_json, created_at)
               VALUES (?, ?, ?)""",
            (comp.comparison_id, result_json, now),
        )
        self._conn.commit()
        return comp.comparison_id

    # --- Digital twin runs and calibrations ---

    def save_twin_run(self, summary: dict[str, Any], trace: dict[str, Any]) -> str:
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self._conn.execute(
            "INSERT OR REPLACE INTO digital_twin_runs VALUES (?,?,?,?,?)",
            (summary["run_id"], summary.get("strategy"), json.dumps(summary, default=str),
             json.dumps(trace, default=str), now))
        self._conn.commit()
        return summary["run_id"]

    def get_twin_run(self, run_id: str, with_trace: bool = False) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT summary_json, trace_json, created_at FROM digital_twin_runs WHERE run_id = ?",
            (run_id,)).fetchone()
        if not row:
            raise KeyError(f"Digital twin run '{run_id}' not found")
        out = {"summary": json.loads(row[0]), "created_at": row[2]}
        if with_trace:
            out["trace"] = json.loads(row[1])
        return out

    def list_twin_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT run_id, strategy, created_at FROM digital_twin_runs "
            "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [{"run_id": r[0], "strategy": r[1], "created_at": r[2]} for r in rows]

    def save_calibration(self, report: dict[str, Any]) -> str:
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self._conn.execute("INSERT OR REPLACE INTO calibrations VALUES (?,?,?,?)",
                           (report["calibration_id"], report.get("node"),
                            json.dumps(report, default=str), now))
        self._conn.commit()
        return report["calibration_id"]

    def get_calibration(self, calibration_id: str) -> dict[str, Any]:
        row = self._conn.execute("SELECT report_json FROM calibrations WHERE calibration_id = ?",
                                 (calibration_id,)).fetchone()
        if not row:
            raise KeyError(f"Calibration '{calibration_id}' not found")
        return json.loads(row[0])

    def __del__(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()