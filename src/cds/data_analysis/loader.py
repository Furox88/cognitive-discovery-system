"""Simple CSV loader — no pandas dependency."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataTable:
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return len(self.headers)

    def column(self, name: str) -> list[str]:
        idx = self.headers.index(name)
        return [row[idx] for row in self.rows]

    def column_as_float(self, name: str) -> list[float]:
        return [float(v) for v in self.column(name)]

    def head(self, n: int = 5) -> list[list[str]]:
        return self.rows[:n]

    def describe(self) -> dict[str, dict[str, float]]:
        """Quick summary stats for numeric columns."""
        from cds.stats.descriptive import mean, median, stdev

        result: dict[str, dict[str, float]] = {}
        for h in self.headers:
            try:
                vals = self.column_as_float(h)
                result[h] = {
                    "count": len(vals),
                    "mean": mean(vals),
                    "std": stdev(vals),
                    "min": min(vals),
                    "median": median(vals),
                    "max": max(vals),
                }
            except (ValueError, TypeError):
                pass
        return result


def load_csv(path: str | Path) -> DataTable:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no such file: {p}")
    with open(p, newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)
    return DataTable(headers=headers, rows=rows)
