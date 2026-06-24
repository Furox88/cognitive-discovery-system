"""Pure Python DataFrame-like structure for structured data analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

# A cell value in a DataSet row. Tabular data here is expected to be scalar
# primitives — numbers, strings, booleans, or null — rather than nested
# containers, which keeps column extraction and aggregation type-safe.
Scalar: TypeAlias = int | float | str | bool | None

# A single row maps column name -> cell value.
Row: TypeAlias = dict[str, Scalar]


class DataSet:
    """A lightweight, pure Python 'DataFrame' for structured data.

    Data is stored internally as a list of dictionaries where keys are column names.
    """

    def __init__(self, data: list[Row]):
        """Store ``data`` and derive column names from the first row's keys."""
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
        """Return the number of rows."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Row:
        """Return the row at integer index ``idx``."""
        return self.data[idx]

    def column(self, name: str) -> list[Scalar]:
        """Extract a single column as a list."""
        if name not in self._columns:
            raise ValueError(f"Column '{name}' not found. Available: {self._columns}")
        return [row[name] for row in self.data]

    def filter(self, predicate: Callable[[Row], bool]) -> DataSet:
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

        groups: dict[Scalar, list[Row]] = {}
        for row in self.data:
            key = row[column_name]
            if key not in groups:
                groups[key] = []
            groups[key].append(row)

        return DataGroup(groups, column_name)

    def to_list(self) -> list[Row]:
        """Export data as a list of dictionaries."""
        return [row.copy() for row in self.data]

    def __repr__(self) -> str:
        """Return a compact ``DataSet(rows=..., cols=...)`` summary."""
        if not self.data:
            return "DataSet(empty)"
        return f"DataSet(rows={len(self.data)}, cols={len(self._columns)})"


class DataGroup:
    """Helper class for grouped data aggregations."""

    def __init__(self, groups: dict[Scalar, list[Row]], group_col: str):
        """Store the grouped rows and the name of the grouping column."""
        self.groups = groups
        self.group_col = group_col

    def mean(self, numeric_col: str) -> dict[Scalar, float]:
        """Calculate the mean of a numeric column for each group."""
        result: dict[Scalar, float] = {}
        for key, rows in self.groups.items():
            values: list[float] = [
                float(v) for row in rows if isinstance((v := row.get(numeric_col)), (int, float))
            ]
            if values:
                result[key] = sum(values) / len(values)
            else:
                result[key] = 0.0
        return result

    def count(self) -> dict[Scalar, int]:
        """Count the number of rows in each group."""
        return {key: len(rows) for key, rows in self.groups.items()}
