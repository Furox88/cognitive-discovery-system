"""Tests for the Sprint 5 NLP visualisation primitives."""

from __future__ import annotations

import pytest

from cds.nlp import (
    render_attention_heatmap,
    render_embedding_projection,
    render_training_curve,
)
from cds.nlp.viz import _pca_2d, _shade

# --- render_training_curve -------------------------------------------------


class TestTrainingCurve:
    def test_renders_descending_curve(self) -> None:
        out = render_training_curve([10.0, 1.0], width=20, height=5)
        assert "*" in out
        assert "step" in out

    def test_single_point_does_not_crash(self) -> None:
        # Exercises the ``n == 1`` branch (idx = 0, last line prints 0).
        out = render_training_curve([5.0])
        assert out.endswith("\n")
        assert "*" in out

    def test_flat_curve_uses_safe_span(self) -> None:
        # All-equal losses: the ``hi > lo`` guard makes span = 1.0.
        out = render_training_curve([3.0, 3.0, 3.0])
        assert "*" in out

    def test_rejects_bad_dimensions(self) -> None:
        with pytest.raises(ValueError):
            render_training_curve([1.0], width=0)
        with pytest.raises(ValueError):
            render_training_curve([1.0], height=0)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            render_training_curve([])

    def test_width_one_uses_safe_divisor(self) -> None:
        # ``max(1, width - 1)`` guards the column index when width == 1.
        out = render_training_curve([1.0, 2.0, 3.0], width=1, height=3)
        assert "*" in out


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
        # 2 rows in the matrix but only 1 row_token -> genuine row mismatch.
        with pytest.raises(ValueError, match="row_tokens"):
            render_attention_heatmap([[0.5, 0.5], [0.3, 0.7]], ["only_one"], ["a", "b"])

    def test_rejects_col_mismatch(self) -> None:
        with pytest.raises(ValueError, match="col_tokens"):
            render_attention_heatmap([[0.5, 0.5]], ["a"], ["only_one"])

    def test_rejects_empty_matrix(self) -> None:
        with pytest.raises(ValueError):
            render_attention_heatmap([], [], [])

    def test_rejects_empty_first_row(self) -> None:
        # ``not attn_weights[0]`` branch — matrix with a zero-width row.
        with pytest.raises(ValueError):
            render_attention_heatmap([[]], ["a"], [])

    def test_uniform_matrix_uses_max_shade(self) -> None:
        # All-equal weights -> span == 0 -> _shade returns the darkest shade.
        out = render_attention_heatmap([[0.5, 0.5], [0.5, 0.5]], ["a", "b"], ["x", "y"])
        assert "#" in out


# --- _shade helper ---------------------------------------------------------


class TestShade:
    def test_zero_span_returns_darkest(self) -> None:
        assert _shade(1.0, 1.0, 0.0) == "#"

    def test_normal_range(self) -> None:
        # value at the low end -> lightest shade (space), high end -> darkest.
        assert _shade(0.0, 0.0, 1.0) == " "
        assert _shade(1.0, 0.0, 1.0) == "#"


# --- _pca_2d helper --------------------------------------------------------


class TestPca2d:
    def test_empty_input_returns_empty(self) -> None:
        assert _pca_2d([]) == []
        assert _pca_2d([[]]) == []  # d == 0

    def test_returns_one_point_per_row(self) -> None:
        pts = _pca_2d([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        assert len(pts) == 3
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pts)


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
        # Count marks only inside the grid body: grid rows are the lines that
        # contain a '|' separator (the axis lines and the '+---' rule do not).
        # This avoids counting digits from the float axis labels (e.g. "56.0").
        marks = set("o*+x#@%&0123456789abcdefghijklmnopqrstuvwxyz")
        grid_chars = {
            ch
            for line in out.splitlines()
            if "|" in line
            for ch in line.split("|", 1)[1]
            if ch in marks
        }
        assert len(grid_chars) <= 3

    def test_top_n_zero_renders_all(self) -> None:
        # ``top_n <= 0`` branch: render every input point.
        emb = [[float(i), 1.0, float(i)] for i in range(5)]
        out = render_embedding_projection(emb, top_n=0, width=40, height=8)
        marks = set(ch for ch in out if ch.isalpha())
        assert len(marks) >= 4

    def test_default_labels_when_none(self) -> None:
        # ``labels is None`` branch -> row-index labels used internally.
        out = render_embedding_projection([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], top_n=2)
        assert "PC1" in out and "PC2" in out

    def test_flat_spans_do_not_divide_by_zero(self) -> None:
        # Construct inputs whose PC1 and PC2 both collapse (degenerate span)
        # so the ``xhi > xlo`` / ``yhi > ylo`` guards engage.
        emb = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        out = render_embedding_projection(emb, top_n=3)
        assert out.endswith("\n")

    def test_rejects_label_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="labels"):
            render_embedding_projection([[1.0, 2.0]], labels=["a", "b"])

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            render_embedding_projection([])

    def test_rejects_empty_first_row(self) -> None:
        with pytest.raises(ValueError):
            render_embedding_projection([[]])


# --- public surface --------------------------------------------------------


class TestPublicExports:
    def test_all_three_in_cds_nlp_namespace(self) -> None:
        import cds.nlp as n

        assert "render_training_curve" in n.__all__
        assert "render_attention_heatmap" in n.__all__
        assert "render_embedding_projection" in n.__all__
