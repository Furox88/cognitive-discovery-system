"""High-level plotting helpers built on the optional matplotlib backend.

Every public function:

* accepts plain Python sequences (``list[float]``), matching CDS core APIs;
* returns a matplotlib ``Figure`` (typed as ``Any`` so mypy stays clean without
  matplotlib stubs in the default environment);
* never imports matplotlib at module import time.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from cds.plot._backend import require_matplotlib
from cds.stats.time_series import autocorrelation_function, partial_autocorrelation


def _as_floats(values: Sequence[float], *, name: str) -> list[float]:
    data = [float(v) for v in values]
    if not data:
        raise ValueError(f"{name} must be non-empty")
    return data


def plot_series(
    y: Sequence[float],
    *,
    x: Sequence[float] | None = None,
    title: str = "Series",
    xlabel: str = "index",
    ylabel: str = "value",
) -> Any:
    """Line plot of a 1-D series.

    Args:
        y: y-values.
        x: optional x-values (defaults to ``0..n-1``).
        title: figure title.
        xlabel: x-axis label.
        ylabel: y-axis label.

    Returns:
        A matplotlib ``Figure``.

    Raises:
        ImportError: if matplotlib is not installed.
        ValueError: if ``y`` is empty or ``x`` length mismatches ``y``.
    """
    ys = _as_floats(y, name="y")
    if x is None:
        xs: list[float] = [float(i) for i in range(len(ys))]
    else:
        xs = _as_floats(x, name="x")
        if len(xs) != len(ys):
            raise ValueError(f"x and y must have the same length (got {len(xs)} vs {len(ys)})")

    plt = require_matplotlib()
    fig, ax = plt.subplots()
    ax.plot(xs, ys, color="#0d9488", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_histogram(
    values: Sequence[float],
    *,
    bins: int = 20,
    title: str = "Histogram",
    xlabel: str = "value",
    ylabel: str = "count",
) -> Any:
    """Histogram of a sample.

    Args:
        values: sample values.
        bins: number of bins (must be >= 1).
        title: figure title.
        xlabel: x-axis label.
        ylabel: y-axis label.

    Returns:
        A matplotlib ``Figure``.
    """
    data = _as_floats(values, name="values")
    if bins < 1:
        raise ValueError("bins must be >= 1")

    plt = require_matplotlib()
    fig, ax = plt.subplots()
    ax.hist(data, bins=bins, color="#0d9488", edgecolor="white", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_waveform(
    signal: Sequence[float],
    *,
    sample_rate: float | None = None,
    title: str = "Waveform",
) -> Any:
    """Time-domain waveform plot for signal-processing demos.

    Args:
        signal: samples.
        sample_rate: if given, x-axis is seconds (``n / sample_rate``);
            otherwise sample index.
        title: figure title.

    Returns:
        A matplotlib ``Figure``.
    """
    ys = _as_floats(signal, name="signal")
    if sample_rate is not None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        xs = [i / sample_rate for i in range(len(ys))]
        xlabel = "time (s)"
    else:
        xs = list(range(len(ys)))
        xlabel = "sample"
    return plot_series(ys, x=xs, title=title, xlabel=xlabel, ylabel="amplitude")


def plot_spectrum(
    magnitudes: Sequence[float],
    *,
    sample_rate: float | None = None,
    title: str = "Spectrum",
) -> Any:
    """Magnitude spectrum (e.g. from ``abs(fft(...))``).

    Args:
        magnitudes: non-negative magnitude bins (full or half spectrum).
        sample_rate: if given, x-axis is Hz assuming uniform DFT bins over
            ``[0, sample_rate)``; otherwise bin index.
        title: figure title.

    Returns:
        A matplotlib ``Figure``.
    """
    mags = _as_floats(magnitudes, name="magnitudes")
    n = len(mags)
    if sample_rate is not None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        freqs = [k * sample_rate / n for k in range(n)]
        xlabel = "frequency (Hz)"
    else:
        freqs = list(range(n))
        xlabel = "bin"
    return plot_series(mags, x=freqs, title=title, xlabel=xlabel, ylabel="magnitude")


def plot_acf(
    data: Sequence[float],
    *,
    max_lag: int | None = None,
    title: str = "Autocorrelation (ACF)",
    partial: bool = False,
) -> Any:
    """Stem plot of the sample ACF (or PACF) via :mod:`cds.stats`.

    Args:
        data: ordered observations (``n >= 2``).
        max_lag: largest lag (passed through to the stats helpers).
        title: figure title.
        partial: if ``True``, plot partial autocorrelation instead of ACF.

    Returns:
        A matplotlib ``Figure``.
    """
    series = _as_floats(data, name="data")
    if len(series) < 2:
        raise ValueError("data must have at least 2 observations")

    if partial:
        # PACF returns [phi_00=1, phi_11, ..., phi_kk] (same lag indexing as ACF).
        coeffs = partial_autocorrelation(series, max_lag=max_lag)
    else:
        coeffs = autocorrelation_function(series, max_lag=max_lag)

    lags = list(range(len(coeffs)))
    # Approximate 95% confidence band under white-noise null: ±1.96/√n
    band = 1.96 / math.sqrt(len(series))

    plt = require_matplotlib()
    fig, ax = plt.subplots()
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(band, color="#94a3b8", linestyle="--", linewidth=1.0, label="≈95% band")
    ax.axhline(-band, color="#94a3b8", linestyle="--", linewidth=1.0)
    markerline, stemlines, baseline = ax.stem(lags, coeffs, basefmt=" ")
    plt.setp(markerline, color="#0d9488", markersize=4)
    plt.setp(stemlines, color="#0d9488", linewidth=1.2)
    plt.setp(baseline, visible=False)
    ax.set_title(title)
    ax.set_xlabel("lag")
    ax.set_ylabel("correlation")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_optimization_path(
    history: Sequence[Sequence[float]],
    *,
    title: str = "Optimization trajectory",
    xlabel: str = "x₀",
    ylabel: str = "x₁",
) -> Any:
    """2-D path of iterates ``[(x0, x1), ...]`` from an optimizer.

    Args:
        history: sequence of points; each point must have length >= 2
            (only the first two coordinates are plotted).
        title: figure title.
        xlabel: x-axis label.
        ylabel: y-axis label.

    Returns:
        A matplotlib ``Figure``.
    """
    if not history:
        raise ValueError("history must be non-empty")
    xs: list[float] = []
    ys: list[float] = []
    for i, point in enumerate(history):
        if len(point) < 2:
            raise ValueError(f"history[{i}] must have at least 2 coordinates")
        xs.append(float(point[0]))
        ys.append(float(point[1]))

    plt = require_matplotlib()
    fig, ax = plt.subplots()
    ax.plot(xs, ys, color="#0d9488", linewidth=1.5, marker="o", markersize=3)
    ax.scatter([xs[0]], [ys[0]], color="#22c55e", s=40, zorder=3, label="start")
    ax.scatter([xs[-1]], [ys[-1]], color="#ef4444", s=40, zorder=3, label="end")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
