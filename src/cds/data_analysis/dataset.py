"""Pure Python DataFrame-like structure for structured data analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class DataSet:
    """A lightweight, pure Python 'DataFrame' for structured data.

    Data is stored internally as a list of dictionaries where keys are column names.
    """

    def __init__(self, data: list[dict[str, Any]]):
        self.data = data
        self._columns = list(data[0].keys()) if data else []

    @property
    def columns(self) -> list[str]:
        """Return the list of column names."""
        return self._columns

    @property
    def shape(self) -> tuple[int, int]:
        """Return (rows, columns) tuple."""
        return len(self.data), len(self._columns)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.data[idx]

    def column(self, name: str) -> list[Any]:
        """Extract a single column as a list."""
        if name not in self._columns:
            raise ValueError(f"Column '{name}' not found. Available: {self._columns}")
        return [row[name] for row in self.data]

    def filter(self, predicate: Callable[[dict[str, Any]], bool]) -> DataSet:
        """Filter the dataset based on a predicate function."""
        filtered_data = [row for row in self.data if predicate(row)]
        return DataSet(filtered_data)

    def head(self, n: int = 5) -> DataSet:
        """Return the first n rows."""
        return DataSet(self.data[:n])

    def tail(self, n: int = 5) -> DataSet:
        """Return the last n rows."""
        return DataSet(self.data[-n:])

    def select(self, *names: str) -> DataSet:
        """Select a subset of columns."""
        for name in names:
            if name not in self._columns:
                raise ValueError(f"Column '{name}' not found.")

        new_data = [{name: row[name] for name in names} for row in self.data]
        return DataSet(new_data)

    def group_by(self, column_name: str) -> DataGroup:
        """Group data by a specific column for aggregation."""
        if column_name not in self._columns:
            raise ValueError(f"Column '{column_name}' not found.")

        groups: dict[Any, list[dict[str, Any]]] = {}
        for row in self.data:
            key = row[column_name]
            if key not in groups:
                groups[key] = []
            groups[key].append(row)

        return DataGroup(groups, column_name)

    def to_list(self) -> list[dict[str, Any]]:
        """Export data as a list of dictionaries."""
        return [row.copy() for row in self.data]

    def __repr__(self) -> str:
        if not self.data:
            return "DataSet(empty)"
        return f"DataSet(rows={len(self.data)}, cols={len(self._columns)})"


class DataGroup:
    """Helper class for grouped data aggregations."""

    def __init__(self, groups: dict[Any, list[dict[str, Any]]], group_col: str):
        self.groups = groups
        self.group_col = group_col

    def mean(self, numeric_col: str) -> dict[Any, float]:
        """Calculate the mean of a numeric column for each group."""
        result = {}
        for key, rows in self.groups.items():
            values = [
                row[numeric_col] for row in rows if isinstance(row.get(numeric_col), (int, float))
            ]
            if values:
                result[key] = sum(values) / len(values)
            else:
                result[key] = 0.0
        return result

    def count(self) -> dict[Any, int]:
        """Count the number of rows in each group."""
        return {key: len(rows) for key, rows in self.groups.items()}
