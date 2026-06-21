"""Rich multi-panel data viz: Monte Carlo π convergence analysis.

Produces assets/monte_carlo_pi_dashboard.png — a detailed, multi-panel
visualization suitable for r/dataisbeautiful ([OC]).

Layout (16x10 inches, GitHub dark theme):
  Top row:    4 scatter panels at n = 10, 100, 1k, 10k points
              (quarter circle filling, inside=green, outside=red)
  Bottom-L:   π estimate vs n (log-x), true π line, ±std-error band
  Bottom-R:   |error| vs n (log-log), with 1/√n reference decay

All points generated with the SAME method as cds.montecarlo.estimate_pi
(unit-square method, deterministic seed).

Usage:
    python scripts/make_pi_dashboard.py
    python scripts/make_pi_dashboard.py --seed 42 --total 20000
"""

from __future__ import annotations

import argparse
import math
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# ---------------------------------------------------------------------------
# Theme — GitHub dark, consistent with scripts/make_pi_gif.py
# ---------------------------------------------------------------------------
BG = "#0d1117"
PANEL = "#161b22"
GRID = "#21262d"
INK_0 = "#f0f6fc"
INK_1 = "#c9d1d9"
INK_2 = "#8b949e"
INK_3 = "#6e7681"
ACCENT_BLUE = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_RED = "#f85149"
ACCENT_PURPLE = "#bc8cff"
ACCENT_ORANGE = "#d29922"


def style_ax(ax, title: str = "") -> None:
    """Apply the shared dark style to a matplotlib Axes."""
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=INK_2, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    if title:
        ax.set_title(title, color=INK_0, fontsize=12, fontweight="bold", pad=10)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo π convergence dashboard.")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
    parser.add_argument("--total", type=int, default=20_000, help="total points to draw")
    parser.add_argument("--out", default="assets/monte_carlo_pi_dashboard.png")
    args = parser.parse_args()

    total = args.total
    rng = random.Random(args.seed)
    points = [(rng.random(), rng.random()) for _ in range(total)]

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    r2 = xs * xs + ys * ys
    inside_mask = r2 <= 1.0

    # Cumulative running estimate + std error at every step.
    n_arr = np.arange(1, total + 1)
    n_inside_cum = np.cumsum(inside_mask.astype(int))
    p_hat = n_inside_cum / n_arr
    pi_hat = 4.0 * p_hat
    # Binomial std error on the estimate (propagated through 4·p): 4·sqrt(p(1-p)/n)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = 4.0 * np.sqrt(p_hat * (1.0 - p_hat) / n_arr)
    abs_err = np.abs(pi_hat - math.pi)

    # Snapshot n values for the top-row scatter panels.
    snap_ns = [10, 100, 1_000, 10_000]

    # ---- figure ----
    fig = plt.figure(figsize=(16, 10), dpi=130)
    fig.patch.set_facecolor(BG)

    # Custom grid: top row 4 cols (scatter), bottom row 2 cols (convergence + error)
    gs = fig.add_gridspec(
        2, 4,
        height_ratios=[1.05, 1.0],
        hspace=0.42, wspace=0.30,
        left=0.055, right=0.975, top=0.90, bottom=0.085,
    )

    # ---------- top row: scatter snapshots ----------
    for col, n in enumerate(snap_ns):
        ax = fig.add_subplot(gs[0, col])
        style_ax(ax, title=f"n = {n:,}")

        x_in, y_in = xs[:n][inside_mask[:n]], ys[:n][inside_mask[:n]]
        x_out, y_out = xs[:n][~inside_mask[:n]], ys[:n][~inside_mask[:n]]

        ax.scatter(x_in, y_in, s=11 if n < 1000 else 4, c=ACCENT_GREEN,
                   alpha=0.85 if n < 1000 else 0.55, edgecolors="none",
                   label="inside")
        ax.scatter(x_out, y_out, s=11 if n < 1000 else 4, c=ACCENT_RED,
                   alpha=0.85 if n < 1000 else 0.55, edgecolors="none",
                   label="outside")

        circ = Circle((0, 0), 1.0, fill=False, edgecolor=ACCENT_BLUE,
                      linewidth=1.8, linestyle="-")
        ax.add_patch(circ)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)

        # per-panel running π estimate badge
        pi_est = 4.0 * inside_mask[:n].sum() / n
        ax.text(
            0.97, 0.06, f"π ≈ {pi_est:.4f}",
            transform=ax.transAxes, ha="right", va="bottom",
            color=ACCENT_BLUE, fontsize=11, fontweight="bold",
            family="monospace",
            bbox=dict(facecolor=BG, edgecolor=GRID, boxstyle="round,pad=0.35"),
        )
        if col == 0:
            ax.set_ylabel("y", color=INK_2, fontsize=10)
            ax.set_xlabel("x", color=INK_2, fontsize=10)
        else:
            ax.set_xlabel("x", color=INK_2, fontsize=10)

    # ---------- bottom-left: convergence with band ----------
    ax_c = fig.add_subplot(gs[1, :2])
    style_ax(ax_c, title="π estimate convergence  (4 · n_inside / n)")

    # sample on log scale to keep the line crisp, but keep full resolution
    # for the band fill.
    idx_full = n_arr
    # thin for plotting speed/visual but keep endpoints
    thin = max(1, total // 4000)
    plot_idx = idx_full[::thin]
    if plot_idx[-1] != total:
        plot_idx = np.append(plot_idx, total)

    ax_c.fill_between(
        plot_idx,
        (pi_hat - se)[plot_idx - 1],
        (pi_hat + se)[plot_idx - 1],
        color=ACCENT_BLUE, alpha=0.16, label="±1 std error",
    )
    ax_c.plot(plot_idx, pi_hat[plot_idx - 1], color=ACCENT_BLUE,
              linewidth=1.6, label="estimate")
    ax_c.axhline(math.pi, color=ACCENT_GREEN, linewidth=1.4, linestyle="--",
                 label=f"true π = {math.pi:.5f}")

    final_est = pi_hat[-1]
    ax_c.scatter([total], [final_est], color=ACCENT_ORANGE, s=55, zorder=5,
                 edgecolors=BG, linewidths=1.2,
                 label=f"final ≈ {final_est:.5f}")

    ax_c.set_xscale("log")
    ax_c.set_xlim(1, total)
    ax_c.set_xlabel("number of samples  (log scale)", color=INK_2, fontsize=10)
    ax_c.set_ylabel("π estimate", color=INK_2, fontsize=10)
    ax_c.grid(True, which="both", color=GRID, linewidth=0.6, alpha=0.6)
    leg = ax_c.legend(loc="right", frameon=False, fontsize=9, labelcolor=INK_1)
    leg.get_frame().set_facecolor(PANEL)

    # ---------- bottom-right: |error| vs n, log-log with √n reference ----------
    ax_e = fig.add_subplot(gs[1, 2:])
    style_ax(ax_e, title="absolute error decay  (vs. theoretical 1/√n)")

    ax_e.plot(plot_idx, abs_err[plot_idx - 1], color=ACCENT_PURPLE,
              linewidth=1.4, label="observed |estimate − π|")

    # 1/√n reference: scale to pass through the first error sample for shape comparison.
    ref = abs_err[0] / np.sqrt(n_arr) * math.sqrt(1)  # = abs_err[0] / sqrt(n)
    ax_e.plot(plot_idx, ref[plot_idx - 1], color=INK_3, linewidth=1.3,
              linestyle=":", label="∝ 1/√n  (Monte Carlo rate)")

    ax_e.set_xscale("log")
    ax_e.set_yscale("log")
    ax_e.set_xlim(1, total)
    # floor the y-limit to avoid log(0) at the rare exact hit
    ymax = max(abs_err[1:50].max() * 1.5, 1e-2)
    ax_e.set_ylim(max(abs_err.min(), 1e-6) * 0.5, ymax)
    ax_e.set_xlabel("number of samples  (log scale)", color=INK_2, fontsize=10)
    ax_e.set_ylabel("| error |  (log scale)", color=INK_2, fontsize=10)
    ax_e.grid(True, which="both", color=GRID, linewidth=0.6, alpha=0.6)
    leg = ax_e.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_1)
    leg.get_frame().set_facecolor(PANEL)

    # ---------- super title + caption ----------
    fig.suptitle(
        "Monte Carlo estimation of π  —  random points in the unit square",
        color=INK_0, fontsize=16, fontweight="bold", y=0.965,
    )
    fig.text(
        0.5, 0.928,
        f"fraction of points inside the quarter circle ≈ π/4   ·   "
        f"{total:,} samples   ·   seed = {args.seed}   ·   "
        f"final estimate π ≈ {final_est:.5f}  (error {abs_err[-1]:.2e})",
        ha="center", color=INK_2, fontsize=10.5,
    )
    # bottom-right tag (source/tool line, OC requirement)
    fig.text(
        0.975, 0.018,
        "data: synthetic (uniform random)   ·   tool: Python + cds.montecarlo   ·   [OC]",
        ha="right", color=INK_3, fontsize=8.5, family="monospace",
    )

    fig.savefig(args.out, facecolor=BG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}")
    print(f"Final estimate: {final_est:.6f}  (true π = {math.pi:.6f}, error = {abs_err[-1]:.2e})")


if __name__ == "__main__":
    main()
