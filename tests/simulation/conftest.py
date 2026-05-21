"""Pytest configuration for simulation tests.

All tests in this directory are automatically marked as simulation tests.
Run only simulation tests with: pytest -m simulation
Run all tests except simulation tests with: pytest -m "not simulation"
"""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    """Automatically mark all tests in the simulation folder with the 'simulation' marker."""
    for item in items:
        if "simulation" in Path(str(item.fspath)).parts:
            item.add_marker(pytest.mark.simulation)
