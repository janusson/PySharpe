"""Repository ignore rules for local credential-bearing files."""

import os
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SENSITIVE_LOCAL_PATHS = (
    ".env",
    ".env.local",
    ".codex/config.toml",
    ".mcp.json",
    ".claude/settings.json",
    ".claude/settings.local.json",
    ".claude/settings.private.json",
    ".streamlit/secrets.toml",
    "repomix-output.xml",
)


@pytest.mark.parametrize("path", SENSITIVE_LOCAL_PATHS)
def test_credential_bearing_local_paths_are_ignored(path: str) -> None:
    """Keep common local credential files out of Git's staging candidates."""
    environment = os.environ.copy()
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = os.devnull

    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", path],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
    )

    assert result.returncode == 0, f"{path} is not ignored"


def test_credential_bearing_local_paths_are_not_tracked() -> None:
    """Ignore rules cannot protect credential files already in the index."""
    result = subprocess.run(
        ["git", "ls-files", "--"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    tracked_paths = set(result.stdout.splitlines())

    assert not tracked_paths.intersection(SENSITIVE_LOCAL_PATHS)


@pytest.mark.parametrize("path", (".env.example", ".env.template", ".mcp.json.example"))
def test_safe_configuration_examples_remain_available(path: str) -> None:
    """Allow placeholder examples to be committed for contributor guidance."""
    environment = os.environ.copy()
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = os.devnull

    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", path],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
    )

    assert result.returncode == 1, f"{path} is ignored"
