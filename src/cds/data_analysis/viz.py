"""Pure Python ASCII visualization tools for structured data."""
from __future__ import annotations

import math


def plot_bar(data: dict[str, float], title: str = "Bar Chart", width: int = 50) -> str:
    """Generate an ASCII bar chart from a dictionary.
    
    Args:
        data: Mapping from label to numeric value.
        title: Chart title.
        width: Maximum bar width in characters.
    """
    if not data:
        return "No data to plot."
    
    max_val = max(data.values())
    min_val = min(0.0, min(data.values()))
    val_range = max_val - min_val
    
    lines = [f"\n[bold]{title}[/]", "─" * len(title)]
    
    for label, val in data.items():
        bar_len = int((val / max_val) * width) if max_val > 0 else 0
        bar = "█" * bar_len
        lines.append(f"{label:<15} | {bar} ({val:.2f})")
    
    return "\n".join(lines)


def plot_line(y_values: list[float], title: str = "Line Plot", height: int = 10, width: int = 60) -> str:
    """Generate a simple ASCII line plot.
    
    Args:
        y_values: List of numeric values.
        title: Plot title.
        height: Number of rows.
        width: Number of columns (will sample data to fit).
    """
    if not y_values:
        return "No data to plot."

    # Sample/Interpolate to fit width
    if len(y_values) > width:
        indices = [int(i * (len(y_values) - 1) / (width - 1)) for i in range(width)]
        sampled = [y_values[i] for i in indices]
    else:
        sampled = y_values
        width = len(y_values)

    max_y = max(sampled)
    min_y = min(sampled)
    y_range = max_y - min_y if max_y != min_y else 1.0

    # Create grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    for x, y in enumerate(sampled):
        # Calculate row (inverted because row 0 is top)
        norm_y = (y - min_y) / y_range
        row = height - 1 - int(norm_y * (height - 1))
        grid[row][x] = "•"

    lines = [f"\n[bold]{title}[/]", "─" * len(title)]
    for row in grid:
        lines.append("".join(row))
    
    lines.append(f"{'min: ' + f'{min_y:.2f}':<{width//2}}{'max: ' + f'{max_y:.2f}':>{width//2}}")
    return "\n".join(lines)
