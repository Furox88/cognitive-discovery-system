"""Render README.md into section-by-section screenshots for review.

Produces (in the project root):
  - README_TOP.png    : hero / first fold  (≈ first viewport)
  - README_SECTIONS/  : one PNG per <h2> section
  - README_FULL.png   : full-page (kept for completeness)

Run:  python scripts/render_readme_sections.py
"""

from __future__ import annotations

import base64
import re
import textwrap
from pathlib import Path

import markdown as md
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
ASSETS = ROOT / "assets"
OUT_DIR = ROOT / "README_SECTIONS"
OUT_TOP = ROOT / "README_TOP.png"
OUT_FULL = ROOT / "README_FULL.png"
HTML_FILE = ROOT / "README_PREVIEW_SECTIONS.html"


def _data_uri(path: Path) -> str:
    mime = {
        ".gif": "image/gif",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def build_html() -> str:
    md_text = README.read_text(encoding="utf-8")
    md_text = md_text.replace('src="assets/logo.svg"', f'src="{_data_uri(ASSETS / "logo.svg")}"')
    md_text = md_text.replace(
        'src="assets/cds_promo.gif"', f'src="{_data_uri(ASSETS / "cds_promo.gif")}"'
    )

    body = md.markdown(
        md_text,
        extensions=["extra", "sane_lists", "tables", "fenced_code", "toc"],
    )

    css = textwrap.dedent(
        """
        :root {
          --bg:#ffffff; --fg:#1f2328; --fg-muted:#656d76; --border:#d0d7de;
          --code-bg:#f6f8fa; --link:#0969da; --table-head:#f6f8fa;
        }
        *{box-sizing:border-box}
        html,body{margin:0;padding:0}
        body{background:var(--bg);color:var(--fg);
          font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans",Helvetica,Arial,sans-serif;
          font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
        .wrap{max-width:980px;margin:0 auto;padding:32px 48px 64px;
          background:var(--bg);border:1px solid var(--border);border-radius:8px}
        .markdown-body h1,.markdown-body h2,.markdown-body h3,
        .markdown-body h4,.markdown-body h5,.markdown-body h6{
          margin:24px 0 16px;font-weight:600;line-height:1.25}
        .markdown-body h1{font-size:2em;padding-bottom:.3em;border-bottom:1px solid var(--border)}
        .markdown-body h2{font-size:1.5em;padding-bottom:.3em;border-bottom:1px solid var(--border)}
        .markdown-body h3{font-size:1.25em}
        .markdown-body p{margin:0 0 16px}
        .markdown-body a{color:var(--link);text-decoration:none}
        .markdown-body a:hover{text-decoration:underline}
        .markdown-body img{max-width:100%;vertical-align:middle;background:var(--bg)}
        .markdown-body hr{height:.25em;padding:0;margin:24px 0;background-color:var(--border);border:0}
        .markdown-body blockquote{padding:0 1em;color:var(--fg-muted);
          border-left:.25em solid var(--border);margin:0 0 16px}
        .markdown-body blockquote>:first-child{margin-top:0}
        .markdown-body blockquote>:last-child{margin-bottom:0}
        .markdown-body code{padding:.2em .4em;margin:0;font-size:85%;
          background-color:var(--code-bg);border-radius:6px;
          font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace}
        .markdown-body pre{padding:16px;overflow:auto;font-size:85%;line-height:1.45;
          background-color:var(--code-bg);border-radius:6px;margin:0 0 16px}
        .markdown-body pre code{padding:0;margin:0;font-size:100%;background:transparent}
        .markdown-body table{display:block;width:100%;width:max-content;max-width:100%;
          overflow:auto;border-spacing:0;border-collapse:collapse;margin:0 0 16px}
        .markdown-body table th,.markdown-body table td{padding:6px 13px;border:1px solid var(--border)}
        .markdown-body table tr{background-color:var(--bg);border-top:1px solid var(--border)}
        .markdown-body table tr:nth-child(2n){background-color:var(--table-head)}
        .markdown-body table th{font-weight:600}
        .markdown-body ul,.markdown-body ol{padding-left:2em;margin:0 0 16px}
        .markdown-body li+li{margin-top:.25em}
        .markdown-body li>p{margin-top:16px}
        p[align="center"],h1[align="center"],h2[align="center"],div[align="center"]{text-align:center}
        """
    ).strip()

    return textwrap.dedent(
        f"""
        <!doctype html><html lang="en"><head><meta charset="utf-8">
        <title>CDS README Sections</title><style>{css}</style></head>
        <body><div class="wrap"><article class="markdown-body" id="root">
        {body}
        </article></div></body></html>
        """
    ).strip()


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    html = build_html()
    HTML_FILE.write_text(html, encoding="utf-8")
    url = HTML_FILE.resolve().as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1080, "height": 1400}, device_scale_factor=2)
        page = ctx.new_page()
        page.goto(url, wait_until="load")
        page.wait_for_timeout(800)

        # --- 1. First-viewport (hero) screenshot, clipped to one screen ---
        page.screenshot(path=str(OUT_TOP))
        print(f"[ok] {OUT_TOP.name}")

        # --- 2. Full-page, for reference ---
        page.screenshot(path=str(OUT_FULL), full_page=True)
        print(f"[ok] {OUT_FULL.name}")

        # --- 3. Per-h2-section screenshots ---
        # Find every H2's y position and slice the document.
        headings = page.eval_on_selector_all(
            "h2",
            "els => els.map(e => ({text: e.innerText, y: e.getBoundingClientRect().top + window.scrollY, id: e.id}))",
        )
        page_height = page.evaluate("document.body.scrollHeight")
        slices = []
        for i, h in enumerate(headings):
            top = int(h["y"])
            # stop ~24px above the next heading so the border shows
            bottom = int(headings[i + 1]["y"]) - 24 if i + 1 < len(headings) else page_height
            # guard against degenerate / off-canvas clips
            bottom = min(bottom, page_height)
            if bottom - top < 80:
                continue
            slices.append((slug(h["text"]), top, bottom))

        for slug_name, top, bottom in slices:
            clip = {"x": 0, "y": top, "width": 1080, "height": bottom - top}
            out = OUT_DIR / f"{slug_name}.png"
            try:
                page.screenshot(path=str(out), clip=clip)
                print(f"[ok] sections/{slug_name}.png")
            except Exception as e:
                print(f"[skip] {slug_name}: {e}")

        browser.close()


def slug(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    s = re.sub(r"[\s_-]+", "-", s)
    return s or "section"


if __name__ == "__main__":
    main()
