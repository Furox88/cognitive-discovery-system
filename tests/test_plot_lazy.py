"""Tests for optional ``cds[plot]`` lazy-loading (always run, no matplotlib)."""

from __future__ import annotations

import builtins
import sys

import pytest

import cds.plot._backend as plot_backend


def test_missing_matplotlib_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("simulated missing matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "matplotlib", raising=False)
    monkeypatch.delitem(sys.modules, "matplotlib.pyplot", raising=False)

    with pytest.raises(ImportError, match="optional dependency"):
        plot_backend.require_matplotlib()


def test_plot_package_exports() -> None:
    import cds.plot as plot

    for name in (
        "plot_series",
        "plot_histogram",
        "plot_waveform",
        "plot_spectrum",
        "plot_acf",
        "plot_optimization_path",
    ):
        assert callable(getattr(plot, name))
