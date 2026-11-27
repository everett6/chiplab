"""Tests for ExperimentLedger: run logging, provenance preservation, querying, comparison, and persistence across reopen."""

import pytest
import os
import tempfile
import json
from chiplab.experiment_db.ledger import ExperimentLedger
from chiplab.schemas.common import Provenance
from chiplab.schemas.design import DesignSpec, ChipletSpec


class TestExperimentLedgerBasic:
    """Basic tests for ExperimentLedger initialization and run logging."""

    def test_ledger_initialization(self):
        """Test that ledger initializes and creates tables."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        try:
            ledger = ExperimentLedger(db_path=db_path)
            assert ledger is not None
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_log_and_get_run(self):
        """Test logging a run and retrieving it."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        try:
            ledger = ExperimentLedger(db_path=db_path)

            design = DesignSpec(
                name="test_design",
                version="0.1",
                architecture="heterogeneous_accelerator",
                chiplets=[ChipletSpec(type="compute", architecture="SIMD", peak_tops=100.0)],
            )

            params = {
                "peak_tops": 100.0,
                "area_mm2": 50.0,
            }

            metrics = {
                "throughput_tokens_per_sec": 1000.0,
                "power_watts": 25.0,
                "temp_c": 65.0,
            }

            run_id = ledger.log_run(design, params, metrics, fidelity="analytical")
            assert run_id is not None
            assert isinstance(run_id, str)

            retrieved = ledger.get_run(run_id)
            assert retrieved is not None
            assert retrieved["run_id"] == run_id
            assert retrieved["design_name"] == "test_design"
            assert retrieved["design_version"] == "0.1"
            assert retrieved["metrics"]["throughput_tokens_per_sec"] == 1000.0
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestExperimentLedgerProvenance:
    """Tests for provenance preservation."""

    def test_provenance_preservation(self):
        """Test that provenance info is preserved in logged runs."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        try:
            ledger = ExperimentLedger(db_path=db_path)

            design = DesignSpec(
                name="provenance_test",
                version="0.1",
                architecture="test",
                chiplets=[],
            )

            # Create params with TrackedValue
            from chiplab.schemas.common import TrackedValue
            params = {
                "peak_tops": TrackedValue(value=100.0, provenance=Provenance.MEASURED, source="simulation"),
                "area_mm2": TrackedValue(value=50.0, provenance=Provenance.MEASURED, source="layout"),
            }

            metrics = {"throughput": 500.0}

            run_id = ledger.log_run(design, params, metrics, fidelity="analytical")
            assert run_id is not None

            retrieved = ledger.get_run(run_id)
            assert retrieved is not None
            provenance = retrieved["provenance"]
            
            # Check that provenance was stored
            assert "peak_tops" in provenance
            assert provenance["peak_tops"]["provenance"] == "measured"
            assert provenance["peak_tops"]["source"] == "simulation"
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestExperimentLedgerQuerying:
    """Tests for querying runs."""

    def test_query_runs_by_design(self):
        """Test querying runs by design name."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        try:
            ledger = ExperimentLedger(db_path=db_path)

            design = DesignSpec(
                name="query_test",
                version="0.1",
                architecture="test",
                chiplets=[],
            )

            # Log multiple runs with same design
            for i in range(3):
                params = {"param": float(i)}
                metrics = {"value": float(i * 10)}
                run_id = ledger.log_run(design, params, metrics, fidelity="analytical")

            # Query by design name
            results = ledger.query_runs(design_name="query_test")
            assert len(results) == 3
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_query_runs_by_fidelity(self):
        """Test querying runs by fidelity level."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        try:
            ledger = ExperimentLedger(db_path=db_path)

            design = DesignSpec(
                name="fidelity_test",
                version="0.1",
                architecture="test",
                chiplets=[],
            )

            # Log runs with different fidelities
            ledger.log_run(design, {"p": 1}, {"v": 10}, fidelity="analytical")
            ledger.log_run(design, {"p": 2}, {"v": 20}, fidelity="discrete_event")
            ledger.log_run(design, {"p": 3}, {"v": 30}, fidelity="analytical")

            # Query by fidelity
            analytical_runs = ledger.query_runs(fidelity="analytical")
            assert len(analytical_runs) == 2
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_query_runs_min_max_metric(self):
        """Test querying runs with min/max metric filters."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        try:
            ledger = ExperimentLedger(db_path=db_path)

            design = DesignSpec(
                name="metric_test",
                version="0.1",
                architecture="test",
                chiplets=[],
            )

            # Log runs with different values
            ledger.log_run(design, {"p": 1}, {"throughput": 100}, fidelity="analytical")
            ledger.log_run(design, {"p": 2}, {"throughput": 200}, fidelity="analytical")
            ledger.log_run(design, {"p": 3}, {"throughput": 300}, fidelity="analytical")

            # Query with min metric
            results = ledger.query_runs(min_metric={"throughput": 150})
            assert len(results) == 2  # throughput 200 and 300
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestExperimentLedgerComparison:
    """Tests for comparing runs."""

    def test_compare_runs_basic(self):
        """Test basic comparison of two runs."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        try:
            ledger = ExperimentLedger(db_path=db_path)

            design = DesignSpec(
                name="compare_test",
                version="0.1",
                architecture="test",
                chiplets=[],
            )

            # Log two runs with different values and capture run IDs
            run1 = ledger.log_run(design, {"p": 1}, {"throughput": 100, "power": 25}, fidelity="analytical")
            run2 = ledger.log_run(design, {"p": 2}, {"throughput": 200, "power": 30}, fidelity="analytical")

            comparison = ledger.compare_runs(
                [run1, run2], metric="throughput"
            )
            
            assert "metric_deltas" in comparison
            assert "winner_by_metric" in comparison
            assert "throughput" in comparison["metric_deltas"]
            assert comparison["winner_by_metric"]["throughput"] is not None
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_compare_runs_insufficient_runs(self):
        """Test comparison with fewer than 2 runs."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        try:
            ledger = ExperimentLedger(db_path=db_path)

            design = DesignSpec(
                name="compare_insufficient",
                version="0.1",
                architecture="test",
                chiplets=[],
            )

            # Log just one run and capture its ID
            run_id = ledger.log_run(design, {"p": 1}, {"throughput": 100}, fidelity="analytical")

            comparison = ledger.compare_runs([run_id])
            assert "error" in comparison
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestExperimentLedgerPersistence:
    """Tests for persistence across reopens."""

    def test_persistence_across_reopen(self):
        """Test that data persists across ledger reopens."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        try:
            # First session
            ledger1 = ExperimentLedger(db_path=db_path)
            design = DesignSpec(
                name="persistence_test",
                version="0.1",
                architecture="test",
                chiplets=[],
            )
            params = {"p": 42}
            metrics = {"v": 100}
            run_id = ledger1.log_run(design, params, metrics, fidelity="analytical")

            # Close and reopen
            del ledger1

            ledger2 = ExperimentLedger(db_path=db_path)
            retrieved = ledger2.get_run(run_id)
            assert retrieved is not None
            assert retrieved["run_id"] == run_id
            assert retrieved["design_name"] == "persistence_test"
            assert retrieved["metrics"]["v"] == 100
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_persistence_multiple_runs(self):
        """Test persistence of multiple runs."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        try:
            ledger = ExperimentLedger(db_path=db_path)

            design = DesignSpec(
                name="multi_persistence",
                version="0.1",
                architecture="test",
                chiplets=[],
            )

            # Log multiple runs
            run_ids = []
            for i in range(5):
                params = {"p": float(i)}
                metrics = {"v": float(i * 10)}
                run_id = ledger.log_run(design, params, metrics, fidelity="analytical")
                run_ids.append(run_id)

            # Verify all runs are retrievable
            for run_id in run_ids:
                retrieved = ledger.get_run(run_id)
                assert retrieved is not None
                assert retrieved["run_id"] == run_id

            # Query all runs
            all_runs = ledger.query_runs()
            assert len(all_runs) == 5
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestExperimentLedgerExportSummary:
    """Tests for export_summary."""

    def test_export_summary_basic(self):
        """Test basic export_summary functionality."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        try:
            ledger = ExperimentLedger(db_path=db_path)

            design = DesignSpec(
                name="summary_test",
                version="0.1",
                architecture="test",
                chiplets=[],
            )

            # Log a few runs
            for i in range(3):
                params = {"p": float(i)}
                metrics = {"throughput": float(i * 50), "power": float(20 + i)}
                ledger.log_run(design, params, metrics, fidelity="analytical")

            summary = ledger.export_summary()
            assert summary["total_runs"] == 3
            assert "by_fidelity" in summary
            assert "metric_ranges" in summary
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)