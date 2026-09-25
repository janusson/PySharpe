"""Guard tests for the agent skill packages.

The `.agents/skills/` documents are the repo's agent-facing contract, and they
tell readers (human and model) exactly which pytest files cover which module.
Those references rot silently when suites are merged or renamed, leaving
documented commands that cannot run — see `AUDIT-docs-drift.md` §A2 for the
seven references that had gone stale.

These tests fail instead, so a merged suite forces the skills to be updated in
the same change.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO_ROOT / ".agents" / "skills"
TEST_REFERENCE_PATTERN = re.compile(r"tests/[A-Za-z0-9_]+\.py")


def test_documented_test_files_exist() -> None:
    """Every `tests/*.py` path named by a skill must exist in the repository."""
    skill_documents = sorted(SKILLS_DIR.rglob("*.md"))
    assert skill_documents, f"no skill documents found under {SKILLS_DIR}"

    missing: dict[str, list[str]] = {}
    for document in skill_documents:
        references = set(TEST_REFERENCE_PATTERN.findall(document.read_text("utf-8")))
        for reference in sorted(references):
            if not (REPO_ROOT / reference).is_file():
                relative = document.relative_to(REPO_ROOT).as_posix()
                missing.setdefault(reference, []).append(relative)

    assert not missing, "skills reference test files that do not exist: " + "; ".join(
        f"{reference} (named in {', '.join(documents)})"
        for reference, documents in sorted(missing.items())
    )


def test_documented_test_commands_are_runnable() -> None:
    """Every `uv run pytest ...` command in a skill names only existing files."""
    offenders: list[str] = []
    for document in sorted(SKILLS_DIR.rglob("*.md")):
        for line in document.read_text("utf-8").splitlines():
            if "pytest" not in line:
                continue
            arguments = [
                token
                for token in line.replace("`", " ").split()
                if token.startswith("tests/") and token.endswith(".py")
            ]
            for argument in arguments:
                if not (REPO_ROOT / argument).is_file():
                    offenders.append(
                        f"{document.relative_to(REPO_ROOT).as_posix()}: {argument}"
                    )

    assert not offenders, "documented pytest commands name missing files: " + "; ".join(
        sorted(set(offenders))
    )
