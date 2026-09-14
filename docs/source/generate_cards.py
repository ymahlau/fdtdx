from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "notebooks").is_dir():
    # script lives alongside the source files themselves, e.g. docs/source/generate_cards.py
    SOURCE_DIR = SCRIPT_DIR
    DOCS_DIR = SCRIPT_DIR.parent
else:
    # script lives one level up, e.g. docs/generate_cards.py with docs/source/
    DOCS_DIR = SCRIPT_DIR
    SOURCE_DIR = SCRIPT_DIR / "source"

NOTEBOOKS_DIR = SOURCE_DIR / "notebooks"
OVERRIDES_FILE = SOURCE_DIR / "notebook_cards.yaml"

# category name -> the .rst page whose card grid should list it
CATEGORY_PAGES = {
    "quickstart": SOURCE_DIR / "01_quickstart.rst",
    "physics": SOURCE_DIR / "02_physics.rst",
    "components": SOURCE_DIR / "03_components.rst",
    "advanced": SOURCE_DIR / "04_advanced.rst",
}

START_MARKER = ".. GENERATED-CARDS-START"
END_MARKER = ".. GENERATED-CARDS-END"

DESCRIPTION_MAX_LEN = 220


def load_overrides() -> dict:
    if not OVERRIDES_FILE.exists():
        return {}
    data = yaml.safe_load(OVERRIDES_FILE.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        sys.exit(f"{OVERRIDES_FILE}: expected a mapping of docname -> card fields")
    return data


def clean_markdown(text: str) -> str:
    """Strip the most common inline markdown so it reads fine as plain prose."""
    text = re.sub(r"`([^`]*)`", r"\1", text)  # `code`
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)  # **bold**
    text = re.sub(r"\*([^*]*)\*", r"\1", text)  # *italic*
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [text](url)
    return re.sub(r"\s+", " ", text).strip()


def cell_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def first_paragraph(lines: list[str]) -> str:
    """First run of non-blank, non-heading lines in a list of markdown lines."""
    para: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if para:
                break
            continue
        if stripped.startswith("#"):
            break
        para.append(stripped)
    return clean_markdown(" ".join(para))


def extract_title_and_description(notebook_path: Path) -> tuple[str, str]:
    try:
        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  warning: couldn't read {notebook_path.name} ({exc}); using filename", file=sys.stderr)
        nb = {"cells": []}

    md_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "markdown"]

    title = None
    description = ""

    for idx, cell in enumerate(md_cells):
        lines = cell_text(cell).splitlines()
        heading_idx = next((i for i, l in enumerate(lines) if l.strip().startswith("#")), None)
        if heading_idx is None:
            continue

        title = clean_markdown(lines[heading_idx].strip().lstrip("#").strip())
        description = first_paragraph(lines[heading_idx + 1:])

        if not description and idx + 1 < len(md_cells):
            description = first_paragraph(cell_text(md_cells[idx + 1]).splitlines())
        break

    if title is None:
        stem = re.sub(r"^\d+[_\-]+", "", notebook_path.stem)
        title = stem.replace("_", " ").replace("-", " ").strip().title() or notebook_path.stem
        print(
            f"  warning: {notebook_path.relative_to(NOTEBOOKS_DIR)} has no top-level "
            f"markdown heading (e.g. '# {title}') -- using a filename-derived title for "
            f"its card, but nbsphinx will still warn/fail a -W build because the page "
            f"itself has no title for its toctree entry. Add a heading to the notebook's "
            f"first cell to fix this properly.",
            file=sys.stderr,
        )

    if not description:
        description = "See this notebook for details."
    if len(description) > DESCRIPTION_MAX_LEN:
        description = description[:DESCRIPTION_MAX_LEN].rsplit(" ", 1)[0].rstrip(",.;:") + "\u2026"

    return title, description


def build_card(number: int, docname: str, emoji: str, title: str, description: str) -> str:
    heading = f"{emoji} {number}. {title}".strip() if emoji else f"{number}. {title}"
    return (
        f"   .. grid-item-card:: {heading}\n"
        f"      :link: {docname}\n"
        f"      :link-type: doc\n"
        f"\n"
        f"      {description}\n"
    )


def generate_grid_block(category: str, notebooks_dir: Path, overrides: dict) -> str | None:
    notebooks = sorted(notebooks_dir.glob("*.ipynb"))
    if not notebooks:
        return None

    cards = []
    for number, nb_path in enumerate(notebooks, start=1):
        docname = f"notebooks/{category}/{nb_path.stem}"
        override = overrides.get(docname, {}) or {}

        auto_title, auto_description = extract_title_and_description(nb_path)

        title = override.get("title", auto_title)
        description = override.get("description", auto_description)
        # normalize YAML ">" folded scalars, which keep a trailing newline
        description = " ".join(str(description).split())
        emoji = override.get("emoji", "")

        cards.append(build_card(number, docname, emoji, title, description))

    lines = [
        START_MARKER,
        "   (Auto-generated by docs/generate_cards.py -- do not edit by hand.",
        "   To customize a card, add an entry to notebook_cards.yaml instead.)",
        "",
        ".. grid:: 1 2 2 2",
        "   :gutter: 3",
        "",
    ]
    lines.append("\n".join(cards))
    lines.append(END_MARKER)
    return "\n".join(lines).rstrip("\n") + "\n"


def update_rst_file(rst_path: Path, new_block: str) -> None:
    if not rst_path.exists():
        sys.exit(f"{rst_path}: file not found")

    text = rst_path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER), re.DOTALL)

    if not pattern.search(text):
        sys.exit(
            f"{rst_path}: no {START_MARKER} / {END_MARKER} markers found -- "
            "add them around the block that should be auto-generated."
        )

    new_text = pattern.sub(lambda _: new_block.rstrip("\n"), text, count=1)

    if new_text != text:
        rst_path.write_text(new_text, encoding="utf-8")
        print(f"updated {rst_path.relative_to(DOCS_DIR.parent)}")
    else:
        print(f"no changes for {rst_path.relative_to(DOCS_DIR.parent)}")


def main() -> None:
    overrides = load_overrides()

    for category, rst_path in CATEGORY_PAGES.items():
        notebooks_dir = NOTEBOOKS_DIR / category
        if not notebooks_dir.exists():
            print(f"skipping {category}: {notebooks_dir} does not exist")
            continue

        block = generate_grid_block(category, notebooks_dir, overrides)
        if block is None:
            print(f"skipping {category}: no notebooks found in {notebooks_dir}")
            continue

        update_rst_file(rst_path, block)


if __name__ == "__main__":
    main()