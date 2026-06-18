# NLP Visualisation Implementation Plan (Sprint 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the educational NLP track (Sprint 5) with three pure-Python ASCII visualisation primitives — attention heatmap, embedding projection, and training-loss curve — so a learner can *see* what attention, embeddings, and training dynamics look like without pulling matplotlib.

**Architecture:** A single new module `src/cds/nlp/viz.py` exposing three functions. ASCII rendering is the default code path (zero-dependency). The 2D embedding projection reduces high-dimensional vectors via PCA implemented on top of `cds.math_utils.linalg.power_iteration` (top-2 eigenvectors of the covariance matrix, found by deflation) — this is the project's signature "pure-Python, but real algorithm" style. The new symbols are re-exported from `cds.nlp.__init__`, which subsystem D owns exclusively (no other plan touches `__init__.py`).

**Tech Stack:** pure Python 3.10+, `cds.math_utils.linalg.power_iteration`. No new required imports anywhere in `src/cds/`.

**Spec reference:** `docs/superpowers/specs/2026-06-18-project-completion-design.md` §D

**Cross-cutting constraints honoured:**
- Zero-dependency: matplotlib is **not** imported on the default path.
- No coverage regression: the new module ships with a dedicated test file so the 98.59% floor is not breached (Task 6).
- Single owner of `src/cds/nlp/__init__.py` (this plan) and `docs/tutorials/nlp_viz.md` (this plan writes it; subsystem B wires it into `mkdocs.yml`).

---

## File Structure

**Created (4 files):**
- `src/cds/nlp/viz.py` — the three renderers + a private `_pca_2d` helper
- `tests/test_nlp_viz.py` — edge-case + output-shape tests
- `examples/nlp_viz_demo.py` — runnable demonstration of all three
- `docs/tutorials/nlp_viz.md` — narrative tutorial (B adds the nav entry)

**Modified (1 file):**
- `src/cds/nlp/__init__.py` — export the three functions

**Out of scope here (owned by other plans):** `mkdocs.yml` nav entry (B), API reference prose (B).

---

### Task 1: Module skeleton + `render_training_curve`

**Files:**
- Create: `src/cds/nlp/viz.py`

- [ ] **Step 1: Write the module docstring + the training-curve renderer**

Create `src/cds/nlp/viz.py`:

```python
"""Pure-Python ASCII visualisation primitives for the educational NLP track.

Sprint 5 of ``cds.nlp``: three renderers that let a learner *see* what the
tokenizer, attention block, and training loop are doing, without pulling in
matplotlib. All three return ``str`` (so they compose, log, and test cleanly)
and never touch the network or optional dependencies.

* :func:`render_training_curve`        — ASCII loss-vs-step plot
* :func:`render_attention_heatmap`     — ASCII row x col attention grid
* :func:`render_embedding_projection`  — ASCII 2-D PCA scatter of embeddings

The projection uses :func:`cds.math_utils.linalg.power_iteration` on the
covariance matrix (top-2 eigenvectors via deflation), keeping the renderer
zero-dependency while still being *real* PCA — not a placeholder.
"""

from __future__ import annotations

from typing import Sequence


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
    lines.append(f"step 0{'':>{width - 10}}{n - 1 if n > 1 else 0}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 2: Smoke-test it in isolation**

Run:
```bash
python -c "from cds.nlp.viz import render_training_curve; print(render_training_curve([10.0,8.0,6.5,5.0,4.2,3.7,3.3,3.0,2.8,2.7], 40, 8))"
```
Expected: a 40-column, 8-row ASCII plot with a descending `*` staircase, no traceback.

- [ ] **Step 3: Type-check the new file**

Run: `python -m mypy src/cds/nlp/viz.py`
Expected: 0 issues (the file is strict-clean by construction).

- [ ] **Step 4: Commit**

```bash
git add src/cds/nlp/viz.py
git commit -m "feat(nlp): viz module skeleton + render_training_curve (Sprint 5)"
```

---

### Task 2: `render_attention_heatmap`

**Files:**
- Modify: `src/cds/nlp/viz.py` (append)

- [ ] **Step 1: Append the heatmap renderer**

Append to `src/cds/nlp/viz.py`:

```python
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
        raise ValueError(
            f"row_tokens length {len(row_tokens)} != rows {rows}"
        )
    if len(col_tokens) != cols:
        raise ValueError(
            f"col_tokens length {len(col_tokens)} != cols {cols}"
        )

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
```

- [ ] **Step 2: Smoke-test**

Run:
```bash
python -c "from cds.nlp.viz import render_attention_heatmap; print(render_attention_heatmap([[0.7,0.2,0.1],[0.1,0.8,0.1],[0.05,0.15,0.8]], ['the','cat','sat'], ['the','cat','sat']))"
```
Expected: a header row, a separator, then three labelled rows; the diagonal cells are the darkest shade (`#`), the off-diagonal cells are lighter.

- [ ] **Step 3: Type-check**

Run: `python -m mypy src/cds/nlp/viz.py`
Expected: 0 issues.

- [ ] **Step 4: Commit**

```bash
git add src/cds/nlp/viz.py
git commit -m "feat(nlp): render_attention_heatmap ASCII renderer"
```

---

### Task 3: `render_embedding_projection` (PCA via power_iteration + deflation)

**Files:**
- Modify: `src/cds/nlp/viz.py` (append)

- [ ] **Step 1: Append the PCA helper + projection renderer**

Append to `src/cds/nlp/viz.py`:

```python
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
    d = len(embeddings[0])
    if n == 0 or d == 0:
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
    deflated = [
        [cov[i][j] - lambda1 * v1[i] * v1[j] for j in range(d)]
        for i in range(d)
    ]
    _, v2 = power_iteration(deflated)

    return [
        (sum(centred[k][i] * v1[i] for i in range(d)),
         sum(centred[k][i] * v2[i] for i in range(d)))
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
        width, height: canvas size in characters.

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
    pts.sort(key=lambda p: abs(p[0]), reverse=True)
    pts = pts[:top_n]
    if labels is not None:
        labels = list(labels)
    else:
        labels = [str(i) for i in range(len(embeddings))]

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    xspan = xhi - xlo if xhi > xlo else 1.0
    yspan = yhi - ylo if yhi > ylo else 1.0

    grid: list[list[str]] = [[" "] * width for _ in range(height)]
    marks = "o*+x#@%&123456789abcdefghijklmnopqrstuvwxyz"
    for k, (x, y) in enumerate(pts):
        col = int((x - xlo) / xspan * (width - 1))
        # Invert y so larger PC2 is at the top.
        row = int((yhi - y) / yspan * (height - 1))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        grid[row][col] = marks[k % len(marks)]

    lines: list[str] = []
    lines.append(f"PC2 {yhi:.3g} |" + "".join(grid[0]))
    for r in range(1, height):
        lines.append(" " * (3 + len(f"{yhi:.3g}")) + "|" + "".join(grid[r]))
    lines.append(" " * (3 + len(f"{yhi:.3g}")) + "+" + "-" * width)
    lines.append(f"{'PC1':>{3 + len(f'{yhi:.3g}')}} {xlo:.3g}{'':>{width - 12}}{xhi:.3g}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 2: Smoke-test with a clearly-separable toy embedding set**

Run:
```bash
python -c "from cds.nlp.viz import render_embedding_projection; emb=[[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]]; print(render_embedding_projection(emb, labels=['a','b','c','ab','bc','ac'], top_n=6))"
```
Expected: a scatter with six distinct marks spread across the canvas, no traceback. If all marks collapse to one cell, the covariance is rank-deficient for the toy input — try adding a 4th feature column to break symmetry, or confirm `_pca_2d` is being reached by inspecting that `v1`/`v2` differ.

- [ ] **Step 3: Type-check**

Run: `python -m mypy src/cds/nlp/viz.py`
Expected: 0 issues.

- [ ] **Step 4: Commit**

```bash
git add src/cds/nlp/viz.py
git commit -m "feat(nlp): render_embedding_projection via power_iteration PCA"
```

---

### Task 4: Re-export from `cds.nlp.__init__`

**Files:**
- Modify: `src/cds/nlp/__init__.py`

- [ ] **Step 1: Add the import block + `__all__` entries**

In `src/cds/nlp/__init__.py`, add a new import group after the `training` import (keep the existing ordering convention: group by submodule):

```python
from cds.nlp.viz import (
    render_attention_heatmap,
    render_embedding_projection,
    render_training_curve,
)
```

Add the three names to `__all__`, in a new `# viz` section at the end of the list (matching the existing per-section comment style):

```python
    # viz
    "render_attention_heatmap",
    "render_embedding_projection",
    "render_training_curve",
]
```

Also extend the module docstring's "Scope" bullet list with a short note (this is a public API surface change and the docstring is the contract):

```python
* :func:`~cds.nlp.viz.render_attention_heatmap`   — ASCII attention heatmap
* :func:`~cds.nlp.viz.render_embedding_projection` — ASCII 2-D PCA scatter
* :func:`~cds.nlp.viz.render_training_curve`      — ASCII loss curve
```

- [ ] **Step 2: Verify the public surface**

Run:
```bash
python -c "import cds.nlp as n; print(n.render_training_curve([1.0,0.5,0.25], 20, 5)); print('render_attention_heatmap' in n.__all__)"
```
Expected: the ASCII curve prints, then `True`.

- [ ] **Step 3: Type-check the package**

Run: `python -m mypy src/cds/nlp/__init__.py`
Expected: 0 issues.

- [ ] **Step 4: Commit**

```bash
git add src/cds/nlp/__init__.py
git commit -m "feat(nlp): re-export viz renderers from cds.nlp"
```

---

### Task 5: Runnable example `examples/nlp_viz_demo.py`

**Files:**
- Create: `examples/nlp_viz_demo.py`

- [ ] **Step 1: Write the example (mirrors `nlp_bpe_demo.py` style: docstring, `main()`, `if __name__`) **

Create `examples/nlp_viz_demo.py`:

```python
"""End-to-end demo of ``cds.nlp`` visualisation primitives (Sprint 5).

Trains a tiny BPE tokenizer, runs a short attention + embedding forward
pass, then renders all three ASCII visualisations. Runs in <2s with no
optional dependencies.

Run::

    python examples/nlp_viz_demo.py
"""

from __future__ import annotations

from cds.nlp import (
    BPEMerge,
    BPETokenizer,
    PositionalEncoding,
    TokenEmbedding,
    add_positional,
    multi_head_attention,
    render_attention_heatmap,
    render_embedding_projection,
    render_training_curve,
    scaled_dot_product_attention,
    train_bpe,
)


def main() -> None:
    # 1. Tokenise a short sentence.
    corpus = "the quick brown fox the lazy dog the quick brown fox "
    tokenizer = train_bpe(corpus, vocab_size=40, min_frequency=1)
    text = "the quick brown fox"
    ids = tokenizer.encode(text)
    tokens = [tokenizer.id_to_token[i] for i in ids]
    print(f"text:   {text!r}")
    print(f"tokens: {tokens}")

    # 2. Embed (d_model=8) + add positional encoding.
    d_model = 8
    embed = TokenEmbedding(vocab_size=tokenizer.vocab_size, dim=d_model, seed=0)
    pos = PositionalEncoding(dim=d_model, max_len=len(ids))
    x = add_positional(embed(ids), pos(len(ids)))

    # 3. Attention weights via scaled dot-product (single head).
    attn = scaled_dot_product_attention(x, x, x)
    print("\n--- Attention heatmap ---")
    print(render_attention_heatmap(attn, tokens, tokens))

    # 4. Embedding projection: show the whole vocab in 2-D.
    vocab_vectors = [embed([i])[0] for i in range(min(12, tokenizer.vocab_size))]
    vocab_labels = [
        tokenizer.id_to_token[i] or f"<{i}>"
        for i in range(min(12, tokenizer.vocab_size))
    ]
    print("--- Embedding projection (top-12 vocab) ---")
    print(render_embedding_projection(vocab_vectors, labels=vocab_labels, top_n=12))

    # 5. A pretend training curve.
    losses = [3.50, 3.10, 2.80, 2.55, 2.30, 2.10, 1.95, 1.82, 1.70, 1.60,
              1.52, 1.45, 1.39, 1.34, 1.30, 1.27, 1.24, 1.22, 1.20, 1.19]
    print("--- Training loss curve ---")
    print(render_training_curve(losses, width=50, height=10))


if __name__ == "__main__":
    main()
```

> **Adjustment hook:** if `TokenEmbedding` / `PositionalEncoding` / `scaled_dot_product_attention` constructors differ from the call shapes above, run the Step 2 introspection first and align the kwargs. Do not invent attributes — only use the public `cds.nlp.*` exports.

- [ ] **Step 2: Introspect the real signatures before running**

Run:
```bash
python -c "import inspect; from cds.nlp import TokenEmbedding, PositionalEncoding, scaled_dot_product_attention; print(inspect.signature(TokenEmbedding.__init__)); print(inspect.signature(PositionalEncoding.__init__)); print(inspect.signature(scaled_dot_product_attention))"
```
Align the example's call shapes to whatever is printed (especially `embed(ids)` vs `embed.forward(ids)`, and `pos(len(ids))` vs `pos.encode(len(ids))`).

- [ ] **Step 3: Run the example end-to-end**

Run: `python examples/nlp_viz_demo.py`
Expected: prints tokenised text, then three labelled ASCII panels (heatmap, projection, loss curve), exit 0. **No matplotlib import.** If any panel errors, fix the call shape from Step 2 — do not silence the error.

- [ ] **Step 4: Commit**

```bash
git add examples/nlp_viz_demo.py
git commit -m "docs(nlp): runnable nlp_viz_demo example for Sprint 5 renderers"
```

---

### Task 6: Tests `tests/test_nlp_viz.py`

**Files:**
- Create: `tests/test_nlp_viz.py`

> This task is **required**, not optional: the new `viz.py` adds ~100 statements of production code. Without tests the project-wide coverage gate (`fail_under = 99`) would regress below the 98.59% floor that subsystems A/B/D/E are obligated not to breach.

- [ ] **Step 1: Write the tests**

Create `tests/test_nlp_viz.py`:

```python
"""Tests for the Sprint 5 NLP visualisation primitives."""

from __future__ import annotations

import pytest

from cds.nlp import (
    render_attention_heatmap,
    render_embedding_projection,
    render_training_curve,
)


# --- render_training_curve -------------------------------------------------


class TestTrainingCurve:
    def test_renders_descending_curve(self) -> None:
        out = render_training_curve([10.0, 1.0], width=20, height=5)
        assert "*" in out
        assert "step" in out

    def test_single_point_does_not_crash(self) -> None:
        out = render_training_curve([5.0])
        assert out.endswith("\n")

    def test_flat_curve_uses_safe_span(self) -> None:
        # All-equal losses: division-by-zero must be guarded.
        out = render_training_curve([3.0, 3.0, 3.0])
        assert "*" in out

    def test_rejects_bad_dimensions(self) -> None:
        with pytest.raises(ValueError):
            render_training_curve([1.0], width=0)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            render_training_curve([])


# --- render_attention_heatmap ---------------------------------------------


class TestAttentionHeatmap:
    def test_identity_matrix_highlights_diagonal(self) -> None:
        m = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
        out = render_attention_heatmap(m, ["a", "b", "c"], ["a", "b", "c"])
        lines = out.strip().split("\n")
        # header + sep + 3 rows
        assert len(lines) == 5
        # The diagonal cells should be the darkest shade available.
        assert "#" in out

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="row_tokens"):
            render_attention_heatmap([[0.5, 0.5]], ["only_one"], ["a", "b"])

    def test_rejects_empty_matrix(self) -> None:
        with pytest.raises(ValueError):
            render_attention_heatmap([], [], [])


# --- render_embedding_projection ------------------------------------------


class TestEmbeddingProjection:
    def test_orthogonal_inputs_spread_out(self) -> None:
        # Three orthogonal axes -> PCA should spread them across the canvas.
        emb = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        out = render_embedding_projection(emb, labels=["x", "y", "z"], top_n=3)
        # At least three distinct mark characters drawn.
        marks = set(ch for ch in out if ch.isalnum())
        assert len(marks) >= 3

    def test_top_n_limits_points(self) -> None:
        emb = [[float(i), float(i) ** 2, 1.0] for i in range(8)]
        out = render_embedding_projection(emb, top_n=3, width=30, height=6)
        # top_n=3 means at most 3 marks placed; count non-space, non-axis chars.
        drawn = sum(1 for ch in out if ch in "o*+x#@%&" or ch.isdigit())
        assert drawn <= 4  # allow a little slack for axis digits

    def test_rejects_label_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="labels"):
            render_embedding_projection([[1.0, 2.0]], labels=["a", "b"])

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            render_embedding_projection([])


# --- public surface --------------------------------------------------------


class TestPublicExports:
    def test_all_three_in_cds_nlp_namespace(self) -> None:
        import cds.nlp as n

        assert "render_training_curve" in n.__all__
        assert "render_attention_heatmap" in n.__all__
        assert "render_embedding_projection" in n.__all__
```

- [ ] **Step 2: Run the new tests**

Run: `python -m pytest tests/test_nlp_viz.py -v`
Expected: all tests pass. If `test_identity_matrix_highlights_diagonal` fails because normalisation makes off-diagonal zeros the darkest (when all values are 0/1 and `min==max` after a degenerate input), check `_shade`'s `span <= 0` early return — for the identity matrix `span == 1`, so the diagonal (value 1.0) maps to `#` and zeros map to space. If still wrong, inspect the rendered output and adjust the shade thresholds.

- [ ] **Step 3: Confirm the new module is fully covered**

Run: `python -m pytest tests/test_nlp_viz.py --cov=cds.nlp.viz --cov-branch --cov-report=term-missing -q`
Expected: `viz.py` shows **no** `Missing` lines (every branch — bad dims, empty input, flat curve, label mismatch, deflation path — is exercised).

- [ ] **Step 4: Commit**

```bash
git add tests/test_nlp_viz.py
git commit -m "test(nlp): cover viz renderers + PCA deflation path"
```

---

### Task 7: Tutorial `docs/tutorials/nlp_viz.md`

**Files:**
- Create: `docs/tutorials/nlp_viz.md`

> Subsystem B owns `mkdocs.yml`; this task only writes the tutorial body and leaves the nav entry to B.

- [ ] **Step 1: Write the tutorial**

Create `docs/tutorials/nlp_viz.md`:

````markdown
# Visualising NLP Internals (Sprint 5)

The `cds.nlp.viz` module renders the three things a learner most wants to
*see* when reading about transformers — attention, embeddings, and the
training-loss curve — as ASCII, so you need no plotting backend. Every
renderer returns a `str`, which means they compose under `print()`, log
cleanly, and are trivially testable.

## 1. The training-loss curve

```python
from cds.nlp import render_training_curve

losses = [3.5, 3.1, 2.8, 2.55, 2.3, 2.1, 1.95, 1.82, 1.7, 1.6]
print(render_training_curve(losses, width=50, height=10))
```

The curve is min-max normalised to the canvas. A single point or an
all-equal series is handled safely (no divide-by-zero).

## 2. The attention heatmap

```python
from cds.nlp import render_attention_heatmap

attn = [[0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.05, 0.15, 0.8]]
tokens = ["the", "cat", "sat"]
print(render_attention_heatmap(attn, tokens, tokens))
```

Weights are normalised across the whole matrix per render, then mapped to
nine ASCII shades (`' '` → `'#'`). The diagonal of a causal or
identity-style attention pattern lights up as the darkest cells.

## 3. The embedding projection

```python
from cds.nlp import render_embedding_projection

# Six 3-D embedding vectors; imagine six vocabulary entries.
emb = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
       [1, 1, 0], [0, 1, 1], [1, 0, 1]]
labels = ["a", "b", "c", "ab", "bc", "ac"]
print(render_embedding_projection(emb, labels=labels, top_n=6))
```

The projection is **real PCA**, not a placeholder: the covariance matrix of
the embeddings is built in pure Python, then
[`cds.math_utils.linalg.power_iteration`](../api.md#cds.math_utils.linalg.power_iteration)
recovers the top-2 eigenvectors (the second via deflation). This is the
project's signature "slow but honest" trade-off — the math is exactly what
`sklearn.decomposition.PCA` does, just without the BLAS.

`top_n` keeps large vocabularies readable by plotting only the
highest-variance points along PC1.

## Why ASCII?

Three reasons, in priority order:

1. **Zero-dependency.** No matplotlib import on the default path keeps
   `cds` installable with nothing but the standard library.
2. **Composability.** A `str` return value logs, prints, and diffs in a
   test exactly like any other value.
3. **Teaching clarity.** The renderer source is short enough to read in
   one sitting, which is the point of the whole `cds.nlp` track.

For publication-quality plots, export the matrices
(`attn_weights`, the `_pca_2d` result, `losses`) and plot them with your
own toolchain — the data path is the same.
````

- [ ] **Step 2: Commit**

```bash
git add docs/tutorials/nlp_viz.md
git commit -m "docs(nlp): Sprint 5 visualisation tutorial"
```

---

### Task 8: Final verification

**Files:** none.

- [ ] **Step 1: Full strict-gate run for the NLP subsystem**

Run:
```bash
ruff check src/cds/nlp/viz.py src/cds/nlp/__init__.py examples/nlp_viz_demo.py tests/test_nlp_viz.py && \
ruff format --check src/cds/nlp/viz.py src/cds/nlp/__init__.py examples/nlp_viz_demo.py tests/test_nlp_viz.py && \
mypy src/cds/nlp/viz.py src/cds/nlp/__init__.py && \
pytest tests/test_nlp_viz.py -v
```
Expected: all green.

- [ ] **Step 2: Confirm no project-wide coverage regression**

Run: `python -m pytest tests/ --cov=cds --cov-branch -q`
Expected: no `FAIL Required test coverage`. The number should be **≥ the 98.59% baseline** (the new `viz.py` is fully covered by Task 6, so it can only add or hold, never subtract).

- [ ] **Step 3: Confirm the example still runs after all changes**

Run: `python examples/nlp_viz_demo.py`
Expected: exit 0, three ASCII panels printed.

- [ ] **Step 4: Confirm matplotlib was never imported**

Run:
```bash
python -c "import sys, cds.nlp; assert 'matplotlib' not in sys.modules, 'matplotlib leaked into default path'; print('clean')"
```
Expected: prints `clean`. If this fails, find the stray import in `viz.py` and remove it — the zero-dependency rule is non-negotiable.
