"""Render README.md to a GitHub-style HTML page, then screenshot it via Playwright.

Produces:
  - README_PREVIEW.html  (self-contained, for browser review)
  - README_PREVIEW.png   (full-page screenshot, GitHub-light theme)

Run:  python scripts/render_readme.py
"""

from __future__ import annotations

import base64
import sys
import textwrap
from pathlib import Path

import markdown as md

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
OUT_HTML = ROOT / "README_PREVIEW.html"
OUT_PNG = ROOT / "README_PREVIEW.png"

# Embed the promo GIF as a data URI so the screenshot renders it even if the
# file path resolution differs. The logo.svg is small and also embedded.
ASSETS = ROOT / "assets"


def _data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {
        ".gif": "image/gif",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


def build_html() -> str:
    md_text = README.read_text(encoding="utf-8")

    # Rewrite local asset refs to data URIs so the rendered page is portable
    # and the screenshot actually shows the images.
    promo_uri = _data_uri(ASSETS / "cds_promo.gif")
    logo_uri = _data_uri(ASSETS / "logo.svg")
    md_text = md_text.replace('src="assets/logo.svg"', f'src="{logo_uri}"')
    md_text = md_text.replace('src="assets/cds_promo.gif"', f'src="{promo_uri}"')

    html_body = md.markdown(
        md_text,
        extensions=["extra", "sane_lists", "tables", "fenced_code", "toc"],
    )

    css = textwrap.dedent(
        """
        :root {
          --bg: #ffffff;
          --fg: #1f2328;
          --fg-muted: #656d76;
          --border: #d0d7de;
          --code-bg: #f6f8fa;
          --code-fg: #1f2328;
          --link: #0969da;
          --quote-bg: #f6f8fa;
          --quote-border: #d0d7de;
          --table-head: #f6f8fa;
        }
        * { box-sizing: border-box; }
        html, body { margin: 0; padding: 0; }
        body {
          background: var(--bg);
          color: var(--fg);
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans",
            Helvetica, Arial, sans-serif;
          font-size: 16px;
          line-height: 1.6;
          -webkit-font-smoothing: antialiased;
        }
        .wrap {
          max-width: 980px;
          margin: 0 auto;
          padding: 32px 48px 64px;
          background: var(--bg);
          border: 1px solid var(--border);
          border-radius: 8px;
        }
        .markdown-body { color: var(--fg); }
        .markdown-body h1, .markdown-body h2, .markdown-body h3,
        .markdown-body h4, .markdown-body h5, .markdown-body h6 {
          margin: 24px 0 16px;
          font-weight: 600;
          line-height: 1.25;
        }
        .markdown-body h1 { font-size: 2em; padding-bottom: .3em;
          border-bottom: 1px solid var(--border); }
        .markdown-body h2 { font-size: 1.5em; padding-bottom: .3em;
          border-bottom: 1px solid var(--border); }
        .markdown-body h3 { font-size: 1.25em; }
        .markdown-body h4 { font-size: 1em; }
        .markdown-body p { margin: 0 0 16px; }
        .markdown-body a { color: var(--link); text-decoration: none; }
        .markdown-body a:hover { text-decoration: underline; }
        .markdown-body img { max-width: 100%; vertical-align: middle;
          background-color: var(--bg); }
        .markdown-body hr { height: .25em; padding: 0; margin: 24px 0;
          background-color: var(--border); border: 0; }
        .markdown-body blockquote {
          padding: 0 1em; color: var(--fg-muted);
          border-left: .25em solid var(--quote-border);
          margin: 0 0 16px;
        }
        .markdown-body blockquote > :first-child { margin-top: 0; }
        .markdown-body blockquote > :last-child { margin-bottom: 0; }
        .markdown-body code {
          padding: .2em .4em; margin: 0; font-size: 85%;
          background-color: var(--code-bg); border-radius: 6px;
          font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas,
            "Liberation Mono", monospace;
        }
        .markdown-body pre {
          padding: 16px; overflow: auto; font-size: 85%; line-height: 1.45;
          background-color: var(--code-bg); border-radius: 6px; margin: 0 0 16px;
        }
        .markdown-body pre code {
          padding: 0; margin: 0; font-size: 100%; background: transparent;
        }
        .markdown-body table {
          display: block; width: 100%; width: max-content; max-width: 100%;
          overflow: auto; border-spacing: 0; border-collapse: collapse;
          margin: 0 0 16px;
        }
        .markdown-body table th, .markdown-body table td {
          padding: 6px 13px; border: 1px solid var(--border);
        }
        .markdown-body table tr { background-color: var(--bg); border-top: 1px solid var(--border); }
        .markdown-body table tr:nth-child(2n) { background-color: var(--table-head); }
        .markdown-body table th { font-weight: 600; }
        .markdown-body ul, .markdown-body ol { padding-left: 2em; margin: 0 0 16px; }
        .markdown-body li + li { margin-top: .25em; }
        .markdown-body li > p { margin-top: 16px; }
        /* GitHub alignment attributes from raw HTML (align="center") */
        p[align="center"], h1[align="center"], h2[align="center"], div[align="center"] {
          text-align: center;
        }
        .markdown-body hr { margin: 28px 0; }
        """
    ).strip()

    return textwrap.dedent(
        f"""
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <title>CDS README Preview</title>
          <style>{css}</style>
        </head>
        <body>
          <div class="wrap">
            <article class="markdown-body">
            {html_body}
            </article>
          </div>
        </body>
        </html>
        """
    ).strip()


def main() -> int:
    if not README.exists():
        print(f"ERROR: {README} not found", file=sys.stderr)
        return 1

    html = build_html()
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"[ok] wrote {OUT_HTML}")

    # Screenshot via Playwright (sync API)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "[warn] playwright not installed; skipping screenshot. "
            "Install with: pip install playwright && playwright install chromium",
            file=sys.stderr,
        )
        return 0

    url = OUT_HTML.resolve().as_uri()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": 1080, "height": 1400},
            device_scale_factor=2,
        )
        page = ctx.new_page()
        page.goto(url, wait_until="load")
        # let the embedded GIF poster frame paint
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT_PNG), full_page=True)
        browser.close()
    print(f"[ok] wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
