"""Label/value rows and status dots (CLAUDE.md sections 12.1, 12.3).

Section 12.1: "monospace for every numeric readout", and colour only where it carries state.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from drivyx.gui.theme import tokens


class StatusDot(QLabel):
    """A coloured dot carrying job or check state (section 12.3)."""

    GLYPH = "●"

    def __init__(self, state: str = "queued", parent: QWidget | None = None) -> None:
        super().__init__(self.GLYPH, parent)
        self.setFixedWidth(14)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_state(state)

    def set_state(self, state: str) -> None:
        self._state = state
        self.setStyleSheet(f"color: {tokens.state_color(state)}; background: transparent;")
        self.setToolTip(state)

    @property
    def state(self) -> str:
        return self._state


class StatRow(QWidget):
    """One `label  value` row with a monospace value and an optional state dot."""

    def __init__(
        self,
        label: str,
        value: str = "",
        *,
        state: str | None = None,
        label_width: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(6)

        self._dot: StatusDot | None = None
        if state is not None:
            self._dot = StatusDot(state)
            layout.addWidget(self._dot)

        self._label = QLabel(label)
        self._label.setProperty("dim", "true")
        if label_width:
            self._label.setFixedWidth(label_width)
        layout.addWidget(self._label)

        self._value = QLabel(value)
        self._value.setProperty("mono", "true")
        self._value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._value, 1)

    def set_value(self, value: str) -> None:
        self._value.setText(value)

    def set_state(self, state: str) -> None:
        if self._dot is not None:
            self._dot.set_state(state)

    def set_value_color(self, color: str | None) -> None:
        """Colour the value. Used only where the number itself carries state."""
        self._value.setStyleSheet(
            f"color: {color}; background: transparent;" if color else "background: transparent;"
        )
