"""Minimal example module so the test/coverage scaffolding has real code to exercise.

This is placeholder logic for the planning phase; replace it as the system grows.
"""

from __future__ import annotations


def greeting(name: str = "researcher") -> str:
    """Return a friendly greeting for ``name``.

    Args:
        name: The name to greet. Defaults to ``"researcher"``.

    Returns:
        A greeting string.

    Raises:
        ValueError: If ``name`` is empty or only whitespace.
    """
    if not name or not name.strip():
        raise ValueError("name must not be empty")
    return f"Hello, {name.strip()}!"
