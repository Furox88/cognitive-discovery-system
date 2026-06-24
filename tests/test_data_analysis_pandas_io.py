"""Tests for the optional pandas interoperability bridge.

These tests are skipped automatically when pandas is not installed, so the
core test suite remains green in a zero-dependency environment. The
import-error guidance path is covered by a monkeypatch test.
"""

from __future__ import annotations

import builtins
import sys

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
# Import-guard: missing pandas raises a helpful ImportError
# --------------------------------------------------------------------------- #
def test_missing_pandas_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the lazy import inside _require_pandas to fail.
    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "pandas":
            raise ImportError("simulated missing pandas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Drop any cached pandas module so the import is re-attempted.
    monkeypatch.delitem(sys.modules, "pandas", raising=False)

    with pytest.raises(ImportError, match="optional dependency"):
        pandas_io._require_pandas()


def test_module_attribute_error_for_unknown_name() -> None:
    import cds.data_analysis as da

    with pytest.raises(AttributeError):
        _ = da.nonexistent_function  # noqa: B018


def test_lazy_attribute_exposes_bridge() -> None:
    import cds.data_analysis as da

    # Accessing through the package __getattr__ should resolve to the module fn.
    assert callable(da.to_dataframe)
    assert callable(da.from_dataframe)


def test_require_pandas_returns_module() -> None:
    mod = pandas_io._require_pandas()
    assert hasattr(mod, "DataFrame")
