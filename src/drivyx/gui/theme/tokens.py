"""Blender-style theme tokens (CLAUDE.md section 12.1).

The token values are the spec's, verbatim. blender.qss references them as %NAME% and
load_stylesheet() substitutes them, so a colour is defined exactly once.

Design language (section 12.1): matte flat surfaces, 1 px borders, 4 px radii, monospace for
every numeric readout, motion limited to binary state changes, colour only where it carries
state.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

# Surfaces
BG_WINDOW = "#1d1d1d"
BG_AREA = "#303030"
BG_PANEL = "#282828"
BG_HEADER = "#232323"
BG_INPUT = "#1a1a1a"

# Text
TEXT = "#e6e6e6"
TEXT_DIM = "#9d9d9d"

# Chrome
BORDER = "#3d3d3d"
WIDGET = "#585858"
WIDGET_HI = "#676767"

# State colours. Section 12.1: colour only where it carries state.
ACCENT = "#4772b3"
ACCENT_HOVER = "#5a86c5"
OK = "#6fa85c"
WARN = "#d9a23c"
ERR = "#c4453c"

# Metrics
RADIUS = "4px"
BORDER_W = "1px"
FONT_UI = '"DejaVu Sans"'
FONT_UI_SIZE = "10pt"
FONT_MONO = '"DejaVu Sans Mono"'
FONT_MONO_SIZE = "9pt"

#: Substitution table for blender.qss. Longest keys must substitute first so that %ACCENT%
#: does not partially match inside %ACCENT_HOVER%; _substitute sorts to guarantee this.
TOKENS: dict[str, str] = {
    "BG_WINDOW": BG_WINDOW,
    "BG_AREA": BG_AREA,
    "BG_PANEL": BG_PANEL,
    "BG_HEADER": BG_HEADER,
    "BG_INPUT": BG_INPUT,
    "TEXT": TEXT,
    "TEXT_DIM": TEXT_DIM,
    "BORDER": BORDER,
    "WIDGET": WIDGET,
    "WIDGET_HI": WIDGET_HI,
    "ACCENT": ACCENT,
    "ACCENT_HOVER": ACCENT_HOVER,
    "OK": OK,
    "WARN": WARN,
    "ERR": ERR,
    "RADIUS": RADIUS,
    "BORDER_W": BORDER_W,
    "FONT_UI": FONT_UI,
    "FONT_UI_SIZE": FONT_UI_SIZE,
    "FONT_MONO": FONT_MONO,
    "FONT_MONO_SIZE": FONT_MONO_SIZE,
}

_QSS_PATH = Path(__file__).with_name("blender.qss")


def _substitute(template: str, tokens: dict[str, str]) -> str:
    """Replace %NAME% with its token value.

    Keys are applied longest-first so %ACCENT_HOVER% is consumed before %ACCENT% can match
    its prefix.
    """
    out = template
    for name in sorted(tokens, key=len, reverse=True):
        out = out.replace(f"%{name}%", tokens[name])
    return out


#: /* ... */ comments, which Qt ignores. Stripped before validation so that prose describing
#: the placeholder syntax is not mistaken for an undefined token.
_RE_QSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_RE_PLACEHOLDER = re.compile(r"%([A-Z_][A-Z0-9_]*)%")


def unresolved_tokens(rendered: str) -> list[str]:
    """Any %NAME% left in the style rules after substitution, meaning a typo in the qss.

    Comments are excluded: they are not style rules, and the file's own header explains the
    %NAME% convention using that literal syntax.
    """
    return sorted(set(_RE_PLACEHOLDER.findall(_RE_QSS_COMMENT.sub("", rendered))))


@lru_cache(maxsize=1)
def load_stylesheet() -> str:
    """Read blender.qss and substitute the tokens.

    Raises ValueError when the qss references a token that does not exist, rather than
    shipping a stylesheet with a literal '%ACCENT%' in it that Qt silently ignores.
    """
    template = _QSS_PATH.read_text(encoding="utf-8")
    rendered = _substitute(template, TOKENS)
    missing = unresolved_tokens(rendered)
    if missing:
        raise ValueError(
            f"{_QSS_PATH.name} references undefined theme tokens: {', '.join(missing)}. "
            f"Known tokens: {', '.join(sorted(TOKENS))}."
        )
    return rendered


def state_color(state: str) -> str:
    """Map a job/check state to its token colour (section 12.1)."""
    return {
        "ok": OK,
        "done": OK,
        "running": ACCENT,
        "queued": TEXT_DIM,
        "warn": WARN,
        "interrupted": WARN,
        "err": ERR,
        "error": ERR,
        "failed": ERR,
    }.get(state.lower(), TEXT_DIM)
