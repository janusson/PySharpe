"""Guard against stale test paths advertised by agent skills.

The skill documentation is executable guidance for coding agents. Keep every
referenced test path present in the repository so a documented command cannot
silently rot.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = REPO_ROOT / ".agents" / "skills"


def test_skill_referenced_test_paths_exist() -> None:
    """Every test path named in a skill must exist in the repository."""
    missing: list[str] = []
    pattern = re.compile(r"(?:pytest\s+)?(tests/[A-Za-z0-9_./-]+\.py)")
    for skill in SKILLS_ROOT.glob("*/SKILL.md"):
        text = skill.read_text(encoding="utf-8")
        for path in sorted(set(pattern.findall(text))):
            if not (REPO_ROOT / path).is_file():
                missing.append(f"{skill.relative_to(REPO_ROOT)} -> {path}")

    assert not missing, (
        "Skill documentation references missing test files:\n" + "\n".join(missing)
    )
