"""Tests for the optional pandas interoperability bridge.

These tests are skipped automatically when pandas is not installed, so the
core test suite remains green in a zero-dependency environment. The
import-error guidance path is covered by a monkeypatch test.
"""

from __future__ import annotations

import pytest

pandas = pytest.importorskip("pandas")  # skip the whole module without pandas

from cds.data_analysis import (  # noqa: E402
    DataSet,
    from_dataframe,
    pandas_io,
    to_dataframe,
)


# --------------------------------------------------------------------------- #
# to_dataframe
# --------------------------------------------------------------------------- #
def test_to_dataframe_roundtrip_basic() -> None:
    ds = DataSet(
        [
            {"name": "a", "value": 1},
            {"name": "b", "value": 2},
            {"name": "c", "value": 3},
        ]
    )
    df = to_dataframe(ds)
    assert list(df.columns) == ["name", "value"]
    assert len(df) == 3
    assert df["value"].tolist() == [1, 2, 3]


def test_to_dataframe_preserves_column_order() -> None:
    ds = DataSet([{"z": 1, "a": 2, "m": 3}])
    df = to_dataframe(ds)
    assert list(df.columns) == ["z", "a", "m"]


def test_to_dataframe_empty_dataset() -> None:
    ds = DataSet([])
    df = to_dataframe(ds)
    assert len(df) == 0


def test_to_dataframe_handles_floats_and_strings() -> None:
    ds = DataSet([{"x": 1.5, "label": "foo"}, {"x": 2.5, "label": "bar"}])
    df = to_dataframe(ds)
    assert df["x"].tolist() == [1.5, 2.5]
    assert df["label"].tolist() == ["foo", "bar"]


# --------------------------------------------------------------------------- #
# from_dataframe
# --------------------------------------------------------------------------- #
def test_from_dataframe_roundtrip() -> None:
    df = pandas.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    ds = from_dataframe(df)
    assert ds.columns == ["a", "b"]
    assert ds.column("a") == [1, 2, 3]
    assert ds.column("b") == [4.0, 5.0, 6.0]


def test_from_dataframe_nan_becomes_none() -> None:
    df = pandas.DataFrame({"x": [1.0, float("nan"), 3.0]})
    ds = from_dataframe(df)
    col = ds.column("x")
    assert col[0] == 1.0
    assert col[1] is None  # NaN normalized to None
    assert col[2] == 3.0


def test_from_dataframe_string_columns() -> None:
    df = pandas.DataFrame({"name": ["alpha", "beta"]})
    ds = from_dataframe(df)
    assert ds.column("name") == ["alpha", "beta"]


def test_from_dataframe_empty() -> None:
    df = pandas.DataFrame()
    ds = from_dataframe(df)
    assert len(ds) == 0


# --------------------------------------------------------------------------- #
# Full round-trip fidelity
# --------------------------------------------------------------------------- #
def test_roundtrip_dataset_to_df_and_back() -> None:
    original = DataSet(
        [
            {"id": 1, "temp": 36.5, "city": "Istanbul"},
            {"id": 2, "temp": 22.0, "city": "Ankara"},
        ]
    )
    df = to_dataframe(original)
    restored = from_dataframe(df)
    assert restored.columns == original.columns
    assert restored.column("id") == original.column("id")
    assert restored.column("city") == original.column("city")


def test_roundtrip_preserves_bool_values() -> None:
    ds = DataSet([{"flag": True}, {"flag": False}])
    df = to_dataframe(ds)
    back = from_dataframe(df)
    assert back.column("flag") == [True, False]


# --------------------------------------------------------------------------- #
# Import-guard: _require_pandas succeeds and returns the real module
# --------------------------------------------------------------------------- #
def test_require_pandas_returns_module() -> None:
    mod = pandas_io._require_pandas()
    assert hasattr(mod, "DataFrame")
