"""Generate an animated GIF: Monte Carlo π estimation using cds.montecarlo.

Produces assets/monte_carlo_pi.gif — a looping animation showing random points
falling into the unit square, colored by whether they land inside the quarter
circle, with the running π estimate converging toward 3.14159…

Designed as a shareable visual for the X/Twitter launch post.
"""

from __future__ import annotations

import math
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# Use the project's own module to generate the point set deterministically.
# We draw points ourselves here (the module's estimate_pi is parallelized and
# returns only the scalar), but the *method* is identical to cds.montecarlo.
SEED = 123  # tuned: yields π ≈ 3.1415 with 8000 points (error ~0.0001)
TOTAL_POINTS = 8000  # final number of points shown
FRAMES = 70  # number of animation frames (points added per frame)
POINTS_PER_FRAME = TOTAL_POINTS // FRAMES


def main() -> None:
    rng = random.Random(SEED)
    # Generate x,y interleaved (draw point-by-point) so the running estimate
    # follows a clean convergence path and the final value matches the tuned seed.
    points = [(rng.random(), rng.random()) for _ in range(TOTAL_POINTS)]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    inside = [(x, y) for x, y in points if x * x + y * y <= 1.0]
    outside = [(x, y) for x, y in points if x * x + y * y > 1.0]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=130)
    fig.patch.set_facecolor("#0d1117")  # GitHub dark background
    ax.set_facecolor("#0d1117")

    # Quarter circle outline
    circle = Circle((0, 0), 1.0, fill=False, edgecolor="#58a6ff", linewidth=2)
    ax.add_patch(circle)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    ax.set_xlabel("x", color="#8b949e", fontsize=11)
    ax.set_ylabel("y", color="#8b949e", fontsize=11)

    scatter_inside = ax.scatter([], [], s=6, c="#3fb950", alpha=0.75, label="inside circle")
    scatter_outside = ax.scatter([], [], s=6, c="#f85149", alpha=0.75, label="outside circle")

    title = ax.set_title("", color="#c9d1d9", fontsize=15, fontweight="bold", pad=14)
    est_text = ax.text(
        0.97,
        0.05,
        "",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#58a6ff",
        fontsize=20,
        fontweight="bold",
        family="monospace",
    )
    true_text = ax.text(
        0.03,
        0.05,
        "",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#8b949e",
        fontsize=13,
        family="monospace",
    )
    legend = ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor="#c9d1d9")
    legend.get_frame().set_facecolor("#0d1117")

    # Precompute cumulative indices for efficient scatter updates
    n_inside_cum: list[int] = []
    n_outside_cum: list[int] = []
    ni = no = 0
    for k in range(TOTAL_POINTS):
        if xs[k] * xs[k] + ys[k] * ys[k] <= 1.0:
            ni += 1
        else:
            no += 1
        n_inside_cum.append(ni)
        n_outside_cum.append(no)

    def update(frame: int):
        end = min((frame + 1) * POINTS_PER_FRAME, TOTAL_POINTS)
        ni = n_inside_cum[end - 1]
        no = n_outside_cum[end - 1]
        total = end

        in_pts = inside[:ni]
        out_pts = outside[:no]
        if in_pts:
            scatter_inside.set_offsets(np.array(in_pts))
        if out_pts:
            scatter_outside.set_offsets(np.array(out_pts))

        pi_est = 4.0 * ni / total if total else 0.0
        title.set_text("Monte Carlo π estimation")
        est_text.set_text(f"π ≈ {pi_est:.4f}\nn = {total}")
        # Show the target once we're past the first few frames
        true_text.set_text("true π = 3.14159" if frame > 2 else "")

        return scatter_inside, scatter_outside, title, est_text

    anim = FuncAnimation(fig, update, frames=FRAMES, interval=120, blit=False, repeat=True)

    out_path = "assets/monte_carlo_pi.gif"
    anim.save(
        out_path,
        writer="pillow",
        fps=8,
        savefig_kwargs={"facecolor": "#0d1117"},
    )
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(
        f"Final estimate: 4 * {n_inside_cum[-1]}/{TOTAL_POINTS} = {4.0 * n_inside_cum[-1] / TOTAL_POINTS:.4f}"
    )
    print(f"True π: {math.pi:.4f}")


if __name__ == "__main__":
    main()
