"""Assemble assets/_promo_frames/*.png into a looping promo video (MP4 + WebM).

Uses the ffmpeg binary shipped by `imageio-ffmpeg` (no system ffmpeg install
needed). Reads the same frame set the GIF builder uses and produces:
  * MP4 (H.264, yuv420p) — best compatibility for X, GitHub README, LinkedIn
  * WebM (VP9)           — smaller, great for web embedding

Why video instead of GIF: an 8 MB GIF -> ~0.5-1.5 MB MP4, with smooth gradients
(GIF's 256-color palette banding is gone) and true frame timing.

Usage:
    python scripts/build_video.py
    python scripts/build_video.py --fps 15 --scale 1.0
    python scripts/build_video.py --in assets/_promo_frames --mp4 assets/cds_promo.mp4
    python scripts/build_video.py --format mp4            # mp4 only
    python scripts/build_video.py --crf 20                # quality (lower=finer)

Output: assets/cds_promo.mp4 and assets/cds_promo.webm (looping-friendly).
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

import imageio_ffmpeg
from PIL import Image

# Must match assets/promo_hero.html CONFIG.fps and CONFIG.totalFrames.
DEFAULT_FPS = 15
DEFAULT_IN = "assets/_promo_frames"
DEFAULT_MP4 = "assets/cds_promo.mp4"
DEFAULT_WEBM = "assets/cds_promo.webm"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build CDS promo MP4/WebM from PNG frames.")
    p.add_argument("--in", dest="indir", default=DEFAULT_IN, help="frames directory")
    p.add_argument("--fps", type=int, default=DEFAULT_FPS, help="frames per second")
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="resize factor (1.0 = 2560x1440 native; 0.5 = 1280x720)",
    )
    p.add_argument(
        "--crf", type=int, default=18, help="quality (lower=finer; 18 visually lossless, 23 ok)"
    )
    p.add_argument("--mp4", default=DEFAULT_MP4, help="output mp4 path")
    p.add_argument("--webm", default=DEFAULT_WEBM, help="output webm path")
    p.add_argument(
        "--format",
        choices=["mp4", "webm", "both"],
        default="both",
        help="which container(s) to produce",
    )
    return p.parse_args()


def ffmpeg() -> str:
    """Path to the ffmpeg binary bundled by imageio-ffmpeg."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def list_frames(indir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(indir, "frame_*.png")))
    if not files:
        sys.exit(f"No frames found in {indir} (expected frame_*.png)")
    return files


def src_dims(frames: list[str]) -> tuple[int, int]:
    with Image.open(frames[0]) as im:
        return im.width, im.height


def target_size(src_w: int, src_h: int, scale: float) -> tuple[int, int]:
    """Even dimensions (H.264/yuv420p needs even width & height)."""
    w = max(2, round(src_w * scale))
    h = max(2, round(src_h * scale))
    if w % 2:
        w += 1
    if h % 2:
        h += 1
    return w, h


def run_piped(cmd: list[str], frames: list[str]) -> None:
    """Feed raw RGBA frames into ffmpeg via stdin."""
    print("  cmd:", cmd[0], " ".join(cmd[1:]))
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    assert proc.stdin is not None
    try:
        for f in frames:
            with Image.open(f) as im:
                proc.stdin.write(im.convert("RGBA").tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    _, err = proc.communicate()
    if proc.returncode != 0:
        sys.stderr.write((err or b"")[-4000:].decode("utf-8", "replace") + "\n")
        sys.exit(f"ffmpeg failed (exit {proc.returncode})")


def build_mp4(frames: list[str], out: str, fps: int, crf: int, size: tuple[int, int]) -> None:
    """MP4 via libx264, yuv420p, +faststart for web streaming."""
    w, h = size
    src_w, src_h = src_dims(frames)
    cmd = [
        ffmpeg(),
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{src_w}x{src_h}",
        "-pix_fmt",
        "rgba",
        "-r",
        str(fps),
        "-i",
        "-",
        "-vf",
        f"scale={w}:{h},format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        out,
    ]
    run_piped(cmd, frames)


def build_webm(frames: list[str], out: str, fps: int, crf: int, size: tuple[int, int]) -> None:
    """WebM via libvpx-vp9; CRF mapped loosely from the x264 CRF scale."""
    w, h = size
    src_w, src_h = src_dims(frames)
    vp_crf = min(63, crf + 13)
    cmd = [
        ffmpeg(),
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{src_w}x{src_h}",
        "-pix_fmt",
        "rgba",
        "-r",
        str(fps),
        "-i",
        "-",
        "-vf",
        f"scale={w}:{h},format=yuva420p",
        "-c:v",
        "libvpx-vp9",
        "-b:v",
        "0",
        "-crf",
        str(vp_crf),
        "-row-mt",
        "1",
        "-an",
        out,
    ]
    run_piped(cmd, frames)


def report(path: str) -> None:
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  bytes : {size_mb:.2f} MB")
    print(f"  path  : {path}")


def main() -> None:
    args = parse_args()
    frames = list_frames(args.indir)
    src_w, src_h = src_dims(frames)
    size = target_size(src_w, src_h, args.scale)
    print(f"Found {len(frames)} frames ({src_w}x{src_h}) -> {size[0]}x{size[1]} @ {args.fps}fps")

    if args.format in ("mp4", "both"):
        print("Building MP4 (H.264) ...")
        build_mp4(frames, args.mp4, args.fps, args.crf, size)
        print(f"Saved {args.mp4}")
        report(args.mp4)

    if args.format in ("webm", "both"):
        print("Building WebM (VP9) ...")
        build_webm(frames, args.webm, args.fps, args.crf, size)
        print(f"Saved {args.webm}")
        report(args.webm)


if __name__ == "__main__":
    main()
