"""Pure-Python ASCII visualisation primitives for the educational NLP track.

Three renderers that let a learner *see* what the tokenizer, attention block,
and training loop are doing, without pulling in matplotlib. All three return
``str`` (so they compose, log, and test cleanly) and never touch the network
or optional dependencies.

* :func:`render_training_curve`        — ASCII loss-vs-step plot
* :func:`render_attention_heatmap`     — ASCII row x col attention grid
* :func:`render_embedding_projection`  — ASCII 2-D PCA scatter of embeddings

The projection uses :func:`cds.math_utils.linalg.power_iteration` on the
covariance matrix (top-2 eigenvectors via deflation), keeping the renderer
zero-dependency while still being *real* PCA — not a placeholder.
"""

from __future__ import annotations

from collections.abc import Sequence


def render_training_curve(
    losses: Sequence[float],
    width: int = 50,
    height: int = 10,
) -> str:
    """Render an ASCII loss curve.

    Args:
        losses: per-step training losses (monotonic-decreasing looks best,
            but any sequence is accepted; a single point renders as one cell).
        width:  plot width in characters (>= 1).
        height: plot height in characters (>= 1).

    Returns:
        A multi-line ``str`` with y-axis label, the curve, and an x-axis
        showing the step range. Always ends with a trailing newline so it
        composes cleanly under ``print()``.
    """
    if width < 1 or height < 1:
        raise ValueError("width and height must be >= 1")
    if not losses:
        raise ValueError("losses must contain at least one value")

    lo = min(losses)
    hi = max(losses)
    span = hi - lo if hi > lo else 1.0  # avoid divide-by-zero for flat curves
    n = len(losses)

    # Sample ``width`` columns from the loss series. Each column maps to the
    # loss at that fractional position, then to a plot row.
    grid: list[list[str]] = [[" "] * width for _ in range(height)]
    for col in range(width):
        idx = int(col * (n - 1) / max(1, width - 1)) if n > 1 else 0
        loss = losses[idx]
        # Invert: high loss -> row 0 (top), low loss -> bottom row.
        row = int((hi - loss) / span * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][col] = "*"

    lines: list[str] = []
    lines.append(f"{hi:.4g} |" + "".join(grid[0]))
    for r in range(1, height):
        lines.append(" " * (len(f"{hi:.4g}")) + " |" + "".join(grid[r]))
    lines.append(" " * (len(f"{hi:.4g}")) + " +" + "-" * width)
    # Right-align the last step index against ``width``. Use max(1, ...) so the
    # format width is never negative when width < ~10 chars (the label "step 0"
    # already accounts for the left side; padding only fills what remains).
    last_step = n - 1 if n > 1 else 0
    pad = max(0, width - len(f"step 0{last_step}"))
    lines.append(f"step 0{'':>{pad}}{last_step}")
    return "\n".join(lines) + "\n"


# 10 levels, low -> high. Space reads as "no attention", '#' as "max".
_ATTENTION_SHADES: tuple[str, ...] = (" ", ".", ":", "-", "=", "+", "o", "*", "#")


def _shade(value: float, lo: float, span: float) -> str:
    """Map ``value`` in ``[lo, lo+span]`` to one of ``_ATTENTION_SHADES``."""
    if span <= 0:
        return _ATTENTION_SHADES[-1]  # all-equal -> treat as max
    t = (value - lo) / span
    idx = int(t * (len(_ATTENTION_SHADES) - 1))
    idx = max(0, min(len(_ATTENTION_SHADES) - 1, idx))
    return _ATTENTION_SHADES[idx]


def render_attention_heatmap(
    attn_weights: Sequence[Sequence[float]],
    row_tokens: Sequence[str],
    col_tokens: Sequence[str],
) -> str:
    """Render an attention matrix as an ASCII heatmap.

    Args:
        attn_weights: ``[rows][cols]`` matrix of attention weights. Values are
            min-max normalised per render, so any real range works; rows are
            expected to sum to ~1 (softmaxed) but this is not enforced.
        row_tokens: one label per row (e.g. query tokens).
        col_tokens: one label per column (e.g. key tokens).

    Returns:
        A multi-line ``str``: a header row of column tokens, then one line per
        row token followed by its shaded cells.

    Raises:
        ValueError: if the matrix / label shapes do not line up.
    """
    if not attn_weights or not attn_weights[0]:
        raise ValueError("attn_weights must be a non-empty [rows][cols] matrix")
    rows = len(attn_weights)
    cols = len(attn_weights[0])
    if len(row_tokens) != rows:
        raise ValueError(f"row_tokens length {len(row_tokens)} != rows {rows}")
    if len(col_tokens) != cols:
        raise ValueError(f"col_tokens length {len(col_tokens)} != cols {cols}")

    flat = [w for r in attn_weights for w in r]
    lo, hi = min(flat), max(flat)
    span = hi - lo

    label_w = max(len(t) for t in row_tokens)
    header_w = max(len(t) for t in col_tokens)
    header = " " * label_w + " | " + " ".join(f"{t:>{header_w}}" for t in col_tokens)
    sep = "-" * label_w + "-+-" + "-" * (cols * (header_w + 1) - 1)

    lines = [header, sep]
    for label, weights in zip(row_tokens, attn_weights):
        cells = " ".join(_shade(w, lo, span).center(header_w) for w in weights)
        lines.append(f"{label:>{label_w}} | {cells}")
    return "\n".join(lines) + "\n"


def _pca_2d(
    embeddings: Sequence[Sequence[float]],
) -> list[tuple[float, float]]:
    """Project rows of ``embeddings`` to 2-D via top-2 PCA components.

    Uses :func:`cds.math_utils.linalg.power_iteration` on the feature
    covariance matrix. The second eigenvector is recovered by deflation
    (subtract ``lambda1 * v1 v1^T`` from the covariance, then iterate again).
    This is the textbook pure-Python PCA and keeps the renderer zero-dependency.
    """
    from cds.math_utils.linalg import power_iteration

    n = len(embeddings)
    if n == 0:
        return []
    d = len(embeddings[0])
    if d == 0:
        return []

    # Column means, then centre.
    mean = [sum(e[j] for e in embeddings) / n for j in range(d)]
    centred = [[e[j] - mean[j] for j in range(d)] for e in embeddings]

    # Covariance: (d x d), biased estimator (divide by n). Sign of the
    # eigenvectors is arbitrary, which is fine for a scatter.
    cov = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            cov[i][j] = sum(centred[k][i] * centred[k][j] for k in range(n)) / n

    lambda1, v1 = power_iteration(cov)
    # Deflate to find the second principal component.
    deflated = [[cov[i][j] - lambda1 * v1[i] * v1[j] for j in range(d)] for i in range(d)]
    _, v2 = power_iteration(deflated)

    return [
        (sum(centred[k][i] * v1[i] for i in range(d)), sum(centred[k][i] * v2[i] for i in range(d)))
        for k in range(n)
    ]


def render_embedding_projection(
    embeddings: Sequence[Sequence[float]],
    labels: Sequence[str] | None = None,
    top_n: int = 10,
    width: int = 50,
    height: int = 12,
) -> str:
    """Render a 2-D PCA scatter of embedding vectors as ASCII.

    Args:
        embeddings: ``[n_vectors][dim]`` matrix.
        labels: optional per-vector label. If ``None``, the row index is used.
        top_n: render at most this many points (highest-variance first along PC1)
            so large vocabularies stay readable. ``<= 0`` renders all.
        width: canvas width in characters.
        height: canvas height in characters.

    Returns:
        A multi-line ``str`` with x/y axis labels and one character per point.
    """
    if not embeddings or not embeddings[0]:
        raise ValueError("embeddings must be a non-empty [n][d] matrix")
    if top_n <= 0:
        top_n = len(embeddings)
    if labels is not None and len(labels) != len(embeddings):
        raise ValueError("labels length must match number of embeddings")

    pts = _pca_2d(embeddings)
    # Keep the ``top_n`` points with the largest |PC1| so the spread is visible.
    # Tag each projected point with its original row index so a legend can map
    # back to ``labels`` after the descending-|PC1| sort reorders them.
    indexed = sorted(
        ((p, k) for k, p in enumerate(pts)),
        key=lambda item: abs(item[0][0]),
        reverse=True,
    )[:top_n]
    if labels is not None:
        labels = list(labels)
    else:
        labels = [str(i) for i in range(len(embeddings))]

    xs = [p[0] for p, _ in indexed]
    ys = [p[1] for p, _ in indexed]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    xspan = xhi - xlo if xhi > xlo else 1.0
    yspan = yhi - ylo if yhi > ylo else 1.0

    grid: list[list[str]] = [[" "] * width for _ in range(height)]
    marks = "o*+x#@%&123456789abcdefghijklmnopqrstuvwxyz"
    legend: list[str] = []
    for k, ((x, y), orig_idx) in enumerate(indexed):
        col = int((x - xlo) / xspan * (width - 1))
        # Invert y so larger PC2 is at the top.
        row = int((yhi - y) / yspan * (height - 1))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        mark = marks[k % len(marks)]
        grid[row][col] = mark
        legend.append(f"{mark}={labels[orig_idx]}")

    lines: list[str] = []
    lines.append(f"PC2 {yhi:.3g} |" + "".join(grid[0]))
    for r in range(1, height):
        lines.append(" " * (3 + len(f"{yhi:.3g}")) + "|" + "".join(grid[r]))
    lines.append(" " * (3 + len(f"{yhi:.3g}")) + "+" + "-" * width)
    # Right-align the x-axis hi label. pad = width - (label widths already on
    # the line); clamp at 0 so a tiny width never produces a negative format
    # width ("Sign not allowed in string format specifier").
    left = f"{'PC1':>{3 + len(f'{yhi:.3g}')}} {xlo:.3g}"
    pad = max(0, width - len(left) - len(f"{xhi:.3g}"))
    lines.append(f"{left}{'':>{pad}}{xhi:.3g}")
    # Mark→label legend so caller-supplied ``labels`` are actually surfaced.
    if legend:
        lines.append("  legend: " + "  ".join(legend))
    return "\n".join(lines) + "\n"
