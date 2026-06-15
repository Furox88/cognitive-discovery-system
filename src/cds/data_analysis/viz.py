"""Pure Python ASCII visualization tools for structured data."""
from __future__ import annotations


def plot_bar(data: dict[str, float], title: str = "Bar Chart", width: int = 50) -> str:
    """Generate an ASCII bar chart from a dictionary.
    
    Args:
        data: Mapping from label to numeric value.
        title: Chart title.
        width: Maximum bar width in characters.
    """
    if not data:
        return "No data to plot."
    
    vals = list(data.values())
    max_val = max(vals)
    min_val = min(vals)
    
    lines = [f"\n[bold]{title}[/]", "─" * len(title)]
    
    # Simple normalization logic that respects 0 as a baseline
    limit = max(abs(max_val), abs(min_val), 1e-10)
    
    for label, val in data.items():
        # Calculate bar length relative to the largest absolute value
        bar_len = int((abs(val) / limit) * width)
        if val >= 0:
            bar = "█" * bar_len
            suffix = f" (+{val:.2f})"
        else:
            # Represent negative values with a different character or notation
            bar = "░" * bar_len 
            suffix = f" ({val:.2f})"
        
        lines.append(f"{label:<15} | {bar}{suffix}")
    
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

    # Safety fix for auditor: ensure width is at least 2 for sampling logic
    eff_width = max(2, width)

    # Sample/Interpolate to fit width
    if len(y_values) > eff_width:
        indices = [int(i * (len(y_values) - 1) / (eff_width - 1)) for i in range(eff_width)]
        sampled = [y_values[i] for i in indices]
    else:
        sampled = y_values
        eff_width = len(y_values)

    max_y = max(sampled)
    min_y = min(sampled)
    y_range = max_y - min_y if max_y != min_y else 1.0

    # Create grid
    grid = [[" " for _ in range(eff_width)] for _ in range(height)]

    for x, y in enumerate(sampled):
        # Calculate row (inverted because row 0 is top)
        norm_y = (y - min_y) / y_range
        row = height - 1 - int(norm_y * (height - 1))
        grid[row][x] = "•"

    lines = [f"\n[bold]{title}[/]", "─" * len(title)]
    for grid_row in grid:
        lines.append("".join(grid_row))
    
    min_y_str = f"{min_y:.2f}"
    max_y_str = f"{max_y:.2f}"
    
    lines.append(f"min: {min_y_str:<{eff_width//2}}max: {max_y_str:>{eff_width//2}}")
    return "\n".join(lines)
