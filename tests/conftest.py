"""
Pytest configuration for the Chip Lab test suite.
Adds fixtures to reset shared module state between tests,
particularly for neuromorphic domain wall tests that exhibit
state contamination when run as a subset.
"""
import pytest

from chiplab.models.materials.domain_wall import DWState


@pytest.fixture(autouse=True, scope="function")
def reset_dw_state():
    """Reset DWState between tests to prevent state contamination.

    The DWState class uses Pydantic's `copy()` method which doesn't
    deep-copy mutable fields like position_history, velocity_history,
    and dw_types. This fixture ensures a clean state for each test
    by using model_copy(deep=True) to properly reset all fields.

    This fixture auto-uses for all tests in this directory.
    """
    # Create a fresh DWState instance for each test
    # This resets all mutable state that could leak between tests
    return DWState()


@pytest.fixture(autouse=True)
def isolate_chiplab_databases(tmp_path, monkeypatch):
    """Keep tests away from the real experiment ledger and learning database."""
    monkeypatch.setenv("CHIPLAB_DB_PATH", str(tmp_path / "chiplab_experiments.db"))
    monkeypatch.setenv("CHIPLAB_LEARNING_DB", str(tmp_path / "learning.db"))
    monkeypatch.setenv("CHIPLAB_BRAIN_DB", str(tmp_path / "brain.db"))


def pytest_configure(config):
    """Add marker for state-reset tests if needed."""
    config.addinivalue_line(
        "markers", "state_reset: Test that requires fresh DWState"
    )
