"""Shared helpers for optional visualisation dependencies."""

from __future__ import annotations


def require_matplotlib():
    """Return the ``matplotlib.pyplot`` module, raising a friendly error if missing."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - handled by callers
        raise RuntimeError("matplotlib must be installed to render charts.") from exc
    return plt


__all__ = ["require_matplotlib"]
