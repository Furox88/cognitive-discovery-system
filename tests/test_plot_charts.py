"""Tests for ``cds.plot`` chart helpers (skipped without matplotlib)."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("matplotlib")

from cds.plot import (  # noqa: E402
    plot_acf,
    plot_histogram,
    plot_optimization_path,
    plot_series,
    plot_spectrum,
    plot_waveform,
)


def test_plot_series_basic() -> None:
    fig = plot_series([1.0, 2.0, 3.0, 2.5], title="demo")
    assert fig is not None
    assert len(fig.axes) == 1


def test_plot_series_with_x() -> None:
    fig = plot_series([1.0, 4.0, 9.0], x=[1.0, 2.0, 3.0])
    assert fig is not None


def test_plot_series_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_series([])


def test_plot_series_x_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        plot_series([1.0, 2.0], x=[1.0])


def test_plot_histogram() -> None:
    fig = plot_histogram([1.0, 1.1, 2.0, 2.2, 3.0, 3.5, 3.6], bins=5)
    assert fig is not None


def test_plot_histogram_bad_bins() -> None:
    with pytest.raises(ValueError, match="bins"):
        plot_histogram([1.0, 2.0], bins=0)


def test_plot_waveform_index_and_time() -> None:
    sig = [math.sin(2 * math.pi * 5 * t / 64) for t in range(64)]
    fig1 = plot_waveform(sig)
    fig2 = plot_waveform(sig, sample_rate=64.0)
    assert fig1 is not None and fig2 is not None


def test_plot_waveform_bad_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        plot_waveform([0.0, 1.0], sample_rate=0.0)


def test_plot_spectrum() -> None:
    mags = [abs(math.sin(i / 3)) for i in range(32)]
    fig = plot_spectrum(mags, sample_rate=32.0)
    assert fig is not None


def test_plot_spectrum_bin_axis() -> None:
    """Cover the default bin-index x-axis (no sample_rate)."""
    fig = plot_spectrum([1.0, 0.5, 0.25, 0.1])
    assert fig is not None


def test_plot_spectrum_bad_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        plot_spectrum([1.0, 2.0], sample_rate=-1.0)


def test_plot_acf_and_pacf() -> None:
    # AR(1)-like series with mild persistence
    data = [0.0]
    for i in range(1, 80):
        data.append(0.7 * data[-1] + (0.1 if i % 2 == 0 else -0.05))
    fig_acf = plot_acf(data, max_lag=15)
    fig_pacf = plot_acf(data, max_lag=15, partial=True, title="PACF")
    assert fig_acf is not None and fig_pacf is not None


def test_plot_acf_too_short() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        plot_acf([1.0])


def test_plot_optimization_path() -> None:
    path = [(0.0, 0.0), (0.5, 0.2), (0.8, 0.5), (1.0, 1.0)]
    fig = plot_optimization_path(path)
    assert fig is not None


def test_plot_optimization_path_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_optimization_path([])


def test_plot_optimization_path_bad_point() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        plot_optimization_path([(1.0,)])


def test_require_matplotlib_respects_mplbackend(monkeypatch: pytest.MonkeyPatch) -> None:
    """When MPLBACKEND is set, require_matplotlib must not force Agg."""
    import cds.plot._backend as backend

    monkeypatch.setenv("MPLBACKEND", "Agg")
    plt = backend.require_matplotlib()
    assert plt is not None


def test_require_matplotlib_sets_agg_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    import cds.plot._backend as backend

    monkeypatch.delenv("MPLBACKEND", raising=False)
    plt = backend.require_matplotlib()
    assert plt is not None
