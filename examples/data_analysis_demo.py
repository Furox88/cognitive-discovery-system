"""Data analysis demo — loading, transforming, and plotting tabular data."""

import os
import tempfile

from cds.data_analysis import (
    DataSet,
    load_csv,
    moving_average,
    normalize,
    plot_bar,
    plot_line,
    z_score,
)
from cds.data_analysis.dataset import Scalar


def main() -> None:
    # Build a tiny CSV in a temp file so the demo needs no fixtures.
    csv_text = "day,temperature,sales\n1,18.0,100\n2,19.5,110\n3,22.0,140\n4,21.5,135\n5,23.0,160\n"
    path = os.path.join(tempfile.gettempdir(), "_cds_demo.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write(csv_text)

    print("=== Loading CSV ===")
    table = load_csv(path)
    print(f"headers: {table.headers}")
    print(f"rows: {len(table.rows)}")

    # Extract numeric columns (DataTable stores everything as strings).
    temps = [float(row[1]) for row in table.rows]
    sales = [float(row[2]) for row in table.rows]

    print("\n=== Normalisation ===")
    norm = normalize(temps)
    z = z_score(temps)
    print(f"normalized temps: {[round(v, 3) for v in norm]}")
    print(f"z-score temps   : {[round(v, 3) for v in z]}")

    print("\n=== Smoothing ===")
    ma = moving_average(sales, window=2)
    print(f"2-day moving avg of sales: {[round(v, 3) for v in ma]}")

    print("\n=== ASCII Visualisation ===")
    plot_line(temps, title="temperature")
    plot_bar({"Mon": 100.0, "Tue": 110.0, "Wed": 140.0, "Thu": 135.0, "Fri": 160.0})

    print("\n=== DataSet helper ===")
    # DataSet expects a list of row-dicts; annotate so mypy widens the
    # value type to Scalar (int | float | str | bool | None) instead of
    # narrowing it to float, which would clash with the invariant list.
    data: list[dict[str, Scalar]] = [
        {"day": i, "temp": float(t)} for i, t in enumerate(temps)
    ]
    ds = DataSet(data=data)
    print(f"columns: {ds.columns}")
    print(f"shape: {ds.shape}")
    print(ds.head(3))


if __name__ == "__main__":
    main()
