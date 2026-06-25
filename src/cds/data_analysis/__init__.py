"""Data loading, analysis and visualization helpers."""

from cds.data_analysis.dataset import DataSet
from cds.data_analysis.loader import DataTable, load_csv
from cds.data_analysis.transform import moving_average, normalize, z_score
from cds.data_analysis.viz import plot_bar, plot_line

__all__ = [
    "load_csv",
    "DataTable",
    "DataSet",
    "normalize",
    "z_score",
    "moving_average",
    "plot_bar",
    "plot_line",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazily expose the optional pandas bridge.

    ``pandas_io`` imports pandas at call time, so it is only resolved on first
    access — keeping the core import zero-dependency.
    """
    if name in {"to_dataframe", "from_dataframe"}:
        from cds.data_analysis import pandas_io

        return getattr(pandas_io, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
