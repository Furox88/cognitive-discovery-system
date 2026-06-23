"""Crop the full-page README_PREVIEW.png into review-friendly vertical tiles.

The full PNG is very tall (28k px). We slice it into N tiles so a human can
review each part at readable scale.

Run:  python scripts/crop_readme_png.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "README_FULL.png"
OUT_DIR = ROOT / "README_TILES"

# target tile height in original pixels (the source is device-scale-factor=2,
# so ~2600 original px ≈ one ~1300px logical viewport — readable)
TILE_H = 2600


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    img = Image.open(SRC)
    w, h = img.size
    n = (h + TILE_H - 1) // TILE_H
    for i in range(n):
        top = i * TILE_H
        bottom = min((i + 1) * TILE_H, h)
        tile = img.crop((0, top, w, bottom))
        out = OUT_DIR / f"tile_{i + 1:02d}_of_{n:02d}.png"
        tile.save(out, optimize=True)
        print(f"[ok] {out.name}  ({bottom - top}px)")
    print(f"\nDone. {n} tiles in {OUT_DIR}")


if __name__ == "__main__":
    main()
