from __future__ import annotations

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Path to the RST file.
# This test lives at tests/docs/, so go up 3 levels to reach the repo root.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.parent  # tests/docs/ -> tests/ -> repo root
API_RST = REPO_ROOT / "docs" / "source" / "07_api.rst"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_rst_entries(rst_path: Path) -> set[str]:
    """Return the set of bare symbol names listed in the autosummary directive."""
    text = rst_path.read_text(encoding="utf-8")
    return {
        m.group(1)
        for m in re.finditer(r"^\s+fdtdx\.(\S+)", text, re.MULTILINE)
    }


def _parse_rst_entries_list(rst_path: Path) -> list[str]:
    """Return all entries in order, including duplicates."""
    text = rst_path.read_text(encoding="utf-8")
    return [
        m.group(1)
        for m in re.finditer(r"^\s+fdtdx\.(\S+)", text, re.MULTILINE)
    ]


def _documentable_exports() -> set[str]:
    """Return fdtdx.__all__ minus any submodule entries that can't go in autosummary."""
    import fdtdx  # noqa: PLC0415

    return {
        name for name in fdtdx.__all__
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestApiDocsCompleteness:

    def test_rst_file_exists(self):
        """The API RST file must be present."""
        assert API_RST.exists(), (
            f"API docs RST not found at {API_RST}. "
            "Update the API_RST path in this test if your layout differs."
        )

    def test_all_init_exports_in_rst(self):
        """Every documentable name in fdtdx.__all__ must appear in the RST.

        Submodules are excluded because autosummary cannot document them
        without triggering docutils errors from the built-in dict docstring.
        """
        exported = _documentable_exports()
        documented = _parse_rst_entries(API_RST)

        missing = sorted(exported - documented)
        assert not missing, (
            f"{len(missing)} symbol(s) are in fdtdx.__all__ but missing from {API_RST}:\n"
            + "\n".join(f"  fdtdx.{name}" for name in missing)
        )

    def test_no_extra_entries_in_rst(self):
        """Every fdtdx.<name> entry in the RST must correspond to a name in fdtdx.__all__.

        Catches stale entries left after a symbol is removed from the package.
        """
        exported = _documentable_exports()
        documented = _parse_rst_entries(API_RST)

        extra = sorted(documented - exported)
        assert not extra, (
            f"{len(extra)} RST entry/entries reference symbols not in fdtdx.__all__:\n"
            + "\n".join(f"  fdtdx.{name}" for name in extra)
        )

    def test_no_duplicate_entries_in_rst(self):
        """The RST must not list any fdtdx.<name> entry more than once."""
        entries = _parse_rst_entries_list(API_RST)
        seen: dict[str, int] = {}
        for name in entries:
            seen[name] = seen.get(name, 0) + 1

        duplicates = sorted(name for name, count in seen.items() if count > 1)
        assert not duplicates, (
            f"{len(duplicates)} duplicate entry/entries in {API_RST}:\n"
            + "\n".join(
                f"  fdtdx.{name} (appears {seen[name]}x)" for name in duplicates
            )
        )