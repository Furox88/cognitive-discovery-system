"""Tests for data_analysis module."""
import tempfile

from cds.data_analysis.loader import DataTable, load_csv
from cds.data_analysis.transform import moving_average, normalize, z_score


def test_load_csv():
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


def test_column_as_float():
    dt = DataTable(headers=["x", "y"], rows=[["1", "2"], ["3", "4"]])
    assert dt.column_as_float("x") == [1.0, 3.0]


def test_describe():
    dt = DataTable(headers=["v"], rows=[["10"], ["20"], ["30"]])
    desc = dt.describe()
    assert "v" in desc
    assert desc["v"]["mean"] == 20.0


def test_normalize():
    result = normalize([10, 20, 30])
    assert result == [0.0, 0.5, 1.0]


def test_z_score():
    result = z_score([10, 20, 30])
    assert abs(sum(result)) < 1e-9  # mean should be ~0


def test_moving_average():
    result = moving_average([1, 2, 3, 4, 5], window=3)
    assert result[0] == 1.0
    assert result[1] == 1.5
    assert result[2] == 2.0
    assert result[3] == 3.0
    assert result[4] == 4.0
