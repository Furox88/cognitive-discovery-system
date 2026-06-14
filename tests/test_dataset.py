"""Tests for the DataSet (Mini-Pandas) implementation."""
import pytest
from cds.data_analysis import DataSet

def test_dataset_basic():
    data = [
        {"name": "Alice", "age": 25, "score": 88},
        {"name": "Bob", "age": 30, "score": 92},
        {"name": "Charlie", "age": 25, "score": 70}
    ]
    ds = DataSet(data)
    
    assert ds.shape == (3, 3)
    assert len(ds) == 3
    assert ds.columns == ["name", "age", "score"]
    assert ds.column("age") == [25, 30, 25]

def test_dataset_filter():
    data = [
        {"name": "Alice", "age": 25},
        {"name": "Bob", "age": 30},
        {"name": "Charlie", "age": 22}
    ]
    ds = DataSet(data)
    filtered = ds.filter(lambda x: x["age"] >= 25)
    
    assert len(filtered) == 2
    assert "Bob" in filtered.column("name")
    assert "Charlie" not in filtered.column("name")

def test_dataset_group_by():
    data = [
        {"city": "NYC", "temp": 20},
        {"city": "NYC", "temp": 22},
        {"city": "LA", "temp": 30},
        {"city": "LA", "temp": 32}
    ]
    ds = DataSet(data)
    grouped = ds.group_by("city")
    
    means = grouped.mean("temp")
    assert means["NYC"] == 21.0
    assert means["LA"] == 31.0
    
    counts = grouped.count()
    assert counts["NYC"] == 2
    assert counts["LA"] == 2

def test_dataset_select():
    data = [{"a": 1, "b": 2, "c": 3}]
    ds = DataSet(data)
    selected = ds.select("a", "c")
    
    assert selected.columns == ["a", "c"]
    assert "b" not in selected[0]

def test_dataset_head_tail():
    ds = DataSet([{"a": i} for i in range(10)])
    assert len(ds.head(3)) == 3
    assert len(ds.tail(3)) == 3
    assert ds.head(3)[0]["a"] == 0
    assert ds.tail(3)[-1]["a"] == 9

def test_dataset_empty_repr():
    ds = DataSet([])
    assert "empty" in repr(ds)
    assert ds.columns == []
    assert ds.shape == (0, 0)

def test_dataset_to_list():
    data = [{"a": 1}, {"a": 2}]
    ds = DataSet(data)
    lst = ds.to_list()
    assert lst == data
    assert lst is not ds.data # Should be a copy

def test_dataset_group_by_empty():
    data = [{"cat": "A", "val": "not_a_number"}]
    ds = DataSet(data)
    grouped = ds.group_by("cat")
    # Mean of non-numeric should return 0.0 per our implementation
    assert grouped.mean("val")["A"] == 0.0
