"""Tests for data_analysis module."""

import tempfile

import pytest

from cds.data_analysis.loader import DataTable, load_csv
from cds.data_analysis.transform import moving_average, normalize, z_score

# --- CSV loading ---


def test_load_csv() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("name,value\n")
        f.write("a,1\n")
        f.write("b,2\n")
        f.write("c,3\n")
        path = f.name
    dt = load_csv(path)
    assert dt.n_rows == 3
    assert dt.n_cols == 2
    assert dt.headers == ["name", "value"]


def test_load_csv_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_csv("/nonexistent/path/file.csv")


def test_load_csv_single_column() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("x\n1\n2\n3\n")
        path = f.name
    dt = load_csv(path)
    assert dt.n_cols == 1
    assert dt.n_rows == 3


def test_load_csv_empty_rows() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("a,b\n")
        path = f.name
    dt = load_csv(path)
    assert dt.n_rows == 0
    assert dt.n_cols == 2


# --- DataTable ---


def test_column_as_float() -> None:
    dt = DataTable(headers=["x", "y"], rows=[["1", "2"], ["3", "4"]])
    assert dt.column_as_float("x") == [1.0, 3.0]


def test_column_by_name() -> None:
    dt = DataTable(headers=["x", "y"], rows=[["a", "b"], ["c", "d"]])
    assert dt.column("y") == ["b", "d"]


def test_head() -> None:
    dt = DataTable(headers=["x"], rows=[["1"], ["2"], ["3"], ["4"], ["5"], ["6"]])
    assert len(dt.head(3)) == 3
    assert len(dt.head()) == 5  # default


def test_describe() -> None:
    dt = DataTable(headers=["v"], rows=[["10"], ["20"], ["30"]])
    desc = dt.describe()
    assert "v" in desc
    assert desc["v"]["mean"] == 20.0


def test_describe_skips_non_numeric() -> None:
    dt = DataTable(headers=["name", "val"], rows=[["a", "1"], ["b", "2"]])
    desc = dt.describe()
    assert "name" not in desc
    assert "val" in desc


def test_datatable_empty() -> None:
    dt = DataTable()
    assert dt.n_rows == 0
    assert dt.n_cols == 0


# --- Normalize ---


def test_normalize() -> None:
    result = normalize([10, 20, 30])
    assert result == [0.0, 0.5, 1.0]


def test_normalize_same_values() -> None:
    result = normalize([5, 5, 5])
    assert result == [0.0, 0.0, 0.0]


def test_normalize_two_values() -> None:
    result = normalize([0, 100])
    assert result == [0.0, 1.0]


def test_normalize_negative() -> None:
    result = normalize([-10, 0, 10])
    assert abs(result[0]) < 1e-9
    assert abs(result[1] - 0.5) < 1e-9
    assert abs(result[2] - 1.0) < 1e-9


# --- Z-score ---


def test_z_score() -> None:
    result = z_score([10, 20, 30])
    assert abs(sum(result)) < 1e-9  # mean should be ~0


def test_z_score_same_values() -> None:
    result = z_score([5, 5, 5])
    assert result == [0.0, 0.0, 0.0]


# --- Moving average ---


def test_moving_average() -> None:
    result = moving_average([1, 2, 3, 4, 5], window=3)
    assert result[0] == 1.0
    assert result[1] == 1.5
    assert result[2] == 2.0
    assert result[3] == 3.0
    assert result[4] == 4.0


def test_moving_average_window_1() -> None:
    data: list[float] = [5.0, 10.0, 15.0]
    result = moving_average(data, window=1)
    assert result == [5.0, 10.0, 15.0]


def test_moving_average_invalid_window() -> None:
    with pytest.raises(ValueError):
        moving_average([1, 2, 3], window=0)
