"""Pytest configuration for integration tests.

All tests in this directory are automatically marked as integration tests.
Run only integration tests with: pytest -m integration
Run all tests except integration tests with: pytest -m "not integration"
"""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    """Automatically mark all tests in the integration folder with the 'integration' marker."""
    for item in items:
        if "integration" in Path(str(item.fspath)).parts:
            item.add_marker(pytest.mark.integration)
