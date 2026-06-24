"""Tests for the CLI's ANSI colour helpers (``_wrap``/``_render``).

Coverage of the TTY-enabled code paths: under pytest ``capsys`` stdout is not a
real terminal, so ``_supports_color`` normally returns ``False`` and the ANSI
branches are skipped. These tests flip ``sys.stdout.isatty`` to ``True`` via
``monkeypatch`` to exercise the colour-emitting paths in both helpers, plus the
markup-parsing edge cases (unclosed tag, unknown combined tag) that the
no-colour path also contains.
"""

from __future__ import annotations

import sys

import pytest

from cds.cli import _format_table, _render, _supports_color, _wrap

_RESET = "\033[0m"


def _force_tty(monkeypatch: pytest.MonkeyPatch, value: bool) -> None:
    """Make ``_supports_color`` report the requested TTY state.

    The helper reads ``sys.stdout.isatty()`` at call time, so patching that
    attribute (rather than the helper itself) is what drives the real branches.
    """

    monkeypatch.setattr(sys.stdout, "isatty", lambda: value)


# --------------------------------------------------------------------------- #
# _wrap — colour vs plain output
# --------------------------------------------------------------------------- #
def test_wrap_plain_when_no_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_tty(monkeypatch, False)
    assert _wrap("red", "hello") == "hello"


def test_wrap_emits_ansi_on_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_tty(monkeypatch, True)
    out = _wrap("red", "hello")
    assert out.startswith("\033[31m")
    assert out.endswith(_RESET)
    assert "hello" in out


def test_wrap_unknown_style_falls_back_to_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    # An unmapped style name yields an empty code -> the helper returns the raw
    # text even on a TTY rather than wrapping it in a bare reset.
    _force_tty(monkeypatch, True)
    assert _wrap("no-such-style", "hello") == "hello"


def test_supports_color_reflects_isatty(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_tty(monkeypatch, True)
    assert _supports_color() is True
    _force_tty(monkeypatch, False)
    assert _supports_color() is False


# --------------------------------------------------------------------------- #
# _render — markup parsing (unclosed tag, unknown combined tag)
# --------------------------------------------------------------------------- #
def test_render_unclosed_tag_emitted_verbatim() -> None:
    # No closing ']' after '[' -> the remaining text is appended as-is.
    assert _render("text [bold rest") == "text [bold rest"


def test_render_no_tty_strips_known_tags() -> None:
    # Default (non-TTY) run: tags are removed, text kept, no ANSI codes.
    out = _render("[red]danger[/] plain")
    assert out == "danger plain"


def test_render_tty_emits_open_and_close_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_tty(monkeypatch, True)
    out = _render("[red]danger[/]")
    assert "\033[31m" in out
    assert _RESET in out
    assert "danger" in out


def test_render_unknown_combined_tag_falls_back_to_first_word(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # "bold purple" is not a key in _STYLES, but its first word "bold" is, so
    # the renderer falls back to the bold SGR code (exercising the `code is None
    # and " " in tag` branch at the first-word fallback).
    _force_tty(monkeypatch, True)
    out = _render("[bold purple]hi[/]")
    assert "\033[1m" in out  # bold escape
    assert "hi" in out


def test_render_plain_text_without_tags() -> None:
    assert _render("just plain text") == "just plain text"


# --------------------------------------------------------------------------- #
# _format_table — the ``if title:`` branch (titleless render skips the title rule)
# --------------------------------------------------------------------------- #
def test_format_table_with_title_has_bold_rule() -> None:
    out = _format_table("Title", ["A", "B"], [["1", "2"]])
    assert "Title" in out
    assert "A" in out and "1" in out


def test_format_table_without_title_skips_rule() -> None:
    # Empty title exercises the False side of ``if title:`` (135->137): the
    # title rule line is omitted and only the border/header/rows remain.
    out = _format_table("", ["A"], [["1"]])
    assert out.startswith("+--")  # border first, no title rule before it
    assert "A" in out and "1" in out
