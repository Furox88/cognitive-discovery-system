"""Optional matplotlib plotting helpers for CDS numerical modules.

This package is **optional**: matplotlib is imported lazily inside each
plotting call, so ``import cds`` and ``import cds.plot`` stay free of
heavy dependencies. Install with::

    pip install cognitive-discovery-system[plot]

Design mirrors :mod:`cds.data_analysis.pandas_io`: clear install-hint
``ImportError`` when matplotlib is missing, pure-Python inputs
(``list[float]`` / lag arrays from ``cds.stats``), and Agg-friendly
figures suitable for scripts, notebooks, and headless CI.
"""

from __future__ import annotations

from cds.plot.charts import (
    plot_acf,
    plot_histogram,
    plot_optimization_path,
    plot_series,
    plot_spectrum,
    plot_waveform,
)

__all__ = [
    "plot_series",
    "plot_histogram",
    "plot_waveform",
    "plot_spectrum",
    "plot_acf",
    "plot_optimization_path",
]
