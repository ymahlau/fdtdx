"""Pytest configuration for docs tests.

All tests in this directory are automatically marked as docs tests.
Run only docs tests with: pytest -m docs
Run all tests except docs tests with: pytest -m "not docs"
"""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    """Automatically mark all tests in the docs folder with the 'docs' marker."""
    for item in items:
        if "docs" in Path(str(item.fspath)).parts:
            item.add_marker(pytest.mark.docs)
