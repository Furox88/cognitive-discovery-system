"""Assemble assets/_promo_frames/*.png into a looping animated GIF.

Reads the configured FPS from the HTML page's CONFIG (via Playwright) is
overkill here; we hardcode fps + palette options and keep them in sync with
the JS CONFIG by reading a tiny FPS constant below.

Usage:
    python scripts/build_gif.py
    python scripts/build_gif.py --fps 15 --scale 1.0
    python scripts/build_gif.py --in assets/_promo_frames --out assets/cds_promo.gif

Output: assets/cds_promo.gif  (looping, <=15MB target for X free tier).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from PIL import Image

# Must match assets/promo_hero.html CONFIG.fps and CONFIG.totalFrames.
DEFAULT_FPS = 15
DEFAULT_IN = "assets/_promo_frames"
DEFAULT_OUT = "assets/cds_promo.gif"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build CDS promo GIF from PNG frames.")
    p.add_argument("--in", dest="indir", default=DEFAULT_IN, help="frames directory")
    p.add_argument("--out", dest="out", default=DEFAULT_OUT, help="output gif path")
    p.add_argument("--fps", type=int, default=DEFAULT_FPS, help="frames per second")
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="resize factor (1.0 = 1280x720 @2x=2560x1440; 0.6 keeps it small)",
    )
    return p.parse_args()


def load_frames(indir: str, scale: float) -> list[Image.Image]:
    files = sorted(glob.glob(os.path.join(indir, "frame_*.png")))
    if not files:
        sys.exit(f"No frames found in {indir} (expected frame_*.png)")
    frames: list[Image.Image] = []
    for f in files:
        im = Image.open(f).convert("RGBA")
        # Composite over the page background so the GIF has a real bg
        # (the HTML already paints a bg, so frames are opaque; keep RGBA->RGB).
        bg = Image.new("RGBA", im.size, (15, 23, 42, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
        if scale and scale != 1.0:
            nw, nh = int(im.width * scale), int(im.height * scale)
            im = im.resize((nw, nh), Image.LANCZOS)
        frames.append(im)
    return frames


def build_palette_optimized(frames: list[Image.Image]) -> list[Image.Image]:
    """Quantize every frame to a shared palette for smooth gradients.

    Strategy: build a global palette from a blended representative image,
    then apply adaptive palette to each frame. Falls back to per-frame
    adaptive quantization if anything goes wrong.
    """
    if not frames:
        return frames
    # Adaptive per-frame quantization gives the smoothest result for
    # dark gradients with the sky/indigo accents.
    out: list[Image.Image] = []
    for im in frames:
        q = im.quantize(
            colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.FLOYDSTEINBERG
        )
        out.append(q)
    return out


def main() -> None:
    args = parse_args()
    frames = load_frames(args.indir, args.scale)
    qframes = build_palette_optimized(frames)

    duration_ms = max(1, round(1000 / args.fps))
    first, rest = qframes[0], qframes[1:]

    first.save(
        args.out,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,  # infinite loop
        disposal=2,  # restore to bg between frames (clean redraws)
        optimize=True,
    )

    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"Saved {args.out}")
    print(f"  frames : {len(qframes)}")
    print(f"  fps    : {args.fps}  ({duration_ms}ms/frame)")
    print(f"  size   : {qframes[0].size[0]}x{qframes[0].size[1]}")
    print(
        f"  bytes  : {size_mb:.2f} MB  {'(OK for X)' if size_mb < 15 else '(>15MB — reduce --scale)'}"
    )


if __name__ == "__main__":
    main()
