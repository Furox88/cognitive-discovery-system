"""Lazy matplotlib import for the optional ``cds[plot]`` extra."""

from __future__ import annotations

import os
from typing import Any

_MATPLOTLIB_INSTALL_HINT = (
    "matplotlib is an optional dependency. Install it with "
    "`pip install cognitive-discovery-system[plot]` "
    "(or `pip install matplotlib`)."
)


def require_matplotlib() -> Any:
    """Import and return ``matplotlib.pyplot``, with a clear install hint.

    Selects the non-interactive Agg backend when the user has not set
    ``MPLBACKEND``, so headless CI and servers can still build figures.
    """
    try:
        import matplotlib

        if os.environ.get("MPLBACKEND") is None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_MATPLOTLIB_INSTALL_HINT) from exc
    return plt
