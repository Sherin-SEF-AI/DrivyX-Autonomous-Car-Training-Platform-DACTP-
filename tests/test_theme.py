"""Theme tokens and stylesheet substitution (CLAUDE.md sections 12.1, 12.2).

No QApplication is constructed: token substitution is pure string work and must stay
testable on a headless CPU-only machine.
"""

from __future__ import annotations

import re

import pytest

from drivyx.gui.theme import tokens


def test_spec_token_values_are_verbatim() -> None:
    """Section 12.1 fixes these values; a drift here silently rethemes the app."""
    assert tokens.BG_WINDOW == "#1d1d1d"
    assert tokens.BG_AREA == "#303030"
    assert tokens.BG_PANEL == "#282828"
    assert tokens.BG_HEADER == "#232323"
    assert tokens.BG_INPUT == "#1a1a1a"
    assert tokens.TEXT == "#e6e6e6"
    assert tokens.TEXT_DIM == "#9d9d9d"
    assert tokens.BORDER == "#3d3d3d"
    assert tokens.WIDGET == "#585858"
    assert tokens.WIDGET_HI == "#676767"
    assert tokens.ACCENT == "#4772b3"
    assert tokens.ACCENT_HOVER == "#5a86c5"
    assert tokens.OK == "#6fa85c"
    assert tokens.WARN == "#d9a23c"
    assert tokens.ERR == "#c4453c"


def test_stylesheet_has_no_unresolved_tokens() -> None:
    rendered = tokens.load_stylesheet()
    assert tokens.unresolved_tokens(rendered) == []
    assert "%" not in re.sub(r"/\*.*?\*/", "", rendered, flags=re.DOTALL)


def test_accent_hover_is_not_clobbered_by_accent() -> None:
    """%ACCENT% is a prefix of %ACCENT_HOVER%; longest-first substitution must win.

    A naive replace would turn %ACCENT_HOVER% into '#4772b3_HOVER%'.
    """
    rendered = tokens._substitute("%ACCENT_HOVER% %ACCENT%", tokens.TOKENS)

    assert rendered == f"{tokens.ACCENT_HOVER} {tokens.ACCENT}"
    assert "_HOVER" not in rendered


def test_undefined_token_aborts_loudly() -> None:
    """A typo in the qss must fail, not ship a stylesheet Qt silently ignores."""
    rendered = tokens._substitute("color: %NOT_A_TOKEN%;", tokens.TOKENS)
    assert tokens.unresolved_tokens(rendered) == ["NOT_A_TOKEN"]


def test_stylesheet_contains_spec_baseline_rules() -> None:
    """Section 12.2's baseline must be present (the qss may extend, not contradict)."""
    qss = tokens.load_stylesheet()

    assert "QPlainTextEdit#logConsole" in qss
    assert f"background: {tokens.BG_WINDOW}" in qss
    assert 'QPushButton[primary="true"]' in qss
    assert "QProgressBar::chunk" in qss
    assert "QTabBar::tab:selected" in qss


def test_stylesheet_is_cached() -> None:
    assert tokens.load_stylesheet() is tokens.load_stylesheet()


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("ok", tokens.OK),
        ("done", tokens.OK),
        ("running", tokens.ACCENT),
        ("queued", tokens.TEXT_DIM),
        ("warn", tokens.WARN),
        ("interrupted", tokens.WARN),
        ("failed", tokens.ERR),
        ("error", tokens.ERR),
        ("unknown-state", tokens.TEXT_DIM),
    ],
)
def test_state_colors(state: str, expected: str) -> None:
    """Section 12.1: colour only where it carries state."""
    assert tokens.state_color(state) == expected


def test_state_color_is_case_insensitive() -> None:
    assert tokens.state_color("OK") == tokens.OK
    assert tokens.state_color("Failed") == tokens.ERR
