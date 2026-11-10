#!/usr/bin/env python3
"""Generate the MkDocs API module index from the ``pysharpe`` source tree."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "pysharpe"
OUTPUT = ROOT / "docs" / "api.md"


def main() -> None:
    """Write API directives for every Python module in the package."""
    modules = sorted(
        ".".join(path.relative_to(SOURCE.parent).with_suffix("").parts)
        for path in SOURCE.rglob("*.py")
        if path.name != "__init__.py"
    )

    lines = [
        "# API Reference",
        "",
        "This reference is generated from the Python modules under `src/pysharpe`.",
        "Each section is rendered from the module's source docstrings by mkdocstrings.",
        "Regenerate with `make build_docs`.",
        "",
    ]

    previous_group = None
    for module in modules:
        group = module.split(".")[1] if module.count(".") else "Core"
        if group != previous_group:
            lines.extend((f"## {group.replace('_', ' ').title()}", ""))
            previous_group = group
        lines.extend((f"::: {module}", ""))

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
