"""Pytest configuration for unit tests.

All tests in this directory are automatically marked as unit tests.
Run only unit tests with: pytest -m unit
Run all tests except unit tests with: pytest -m "not unit"
"""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    """Automatically mark all tests in the unit folder with the 'unit' marker."""
    for item in items:
        if "unit" in Path(str(item.fspath)).parts:
            item.add_marker(pytest.mark.unit)
