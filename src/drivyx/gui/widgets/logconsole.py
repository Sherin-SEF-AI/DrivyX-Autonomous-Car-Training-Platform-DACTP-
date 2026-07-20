"""Monospace log console with a filter box (CLAUDE.md section 12.3).

"bottom dock LogConsole (monospace, follows the active job, filter box)".

Lines are retained in a bounded deque and the visible document is re-rendered on filter
change, so filtering never loses history. The cap exists because an 8-hour training run
emits far more than a QPlainTextEdit should hold.
"""

from __future__ import annotations

import re
from collections import deque

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

#: Retained lines. Beyond this the oldest are dropped; the run's own run.log is the complete
#: record (section 6.3), so the console is a live view, not an archive.
MAX_LINES = 5000


class LogConsole(QWidget):
    """Bounded, filterable log view."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._lines: deque[str] = deque(maxlen=MAX_LINES)
        self._filter = ""
        self._regex: re.Pattern[str] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        controls = QHBoxLayout()
        controls.setSpacing(6)

        controls.addWidget(QLabel("Filter"))
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("substring, or /regex/")
        self._filter_edit.textChanged.connect(self._on_filter_changed)
        controls.addWidget(self._filter_edit, 1)

        self._autoscroll = QCheckBox("Follow")
        self._autoscroll.setChecked(True)
        controls.addWidget(self._autoscroll)

        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear)
        controls.addWidget(clear)

        layout.addLayout(controls)

        self._view = QPlainTextEdit()
        self._view.setObjectName("logConsole")
        self._view.setReadOnly(True)
        self._view.setMaximumBlockCount(MAX_LINES)
        self._view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._view.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        layout.addWidget(self._view, 1)

    def append(self, line: str) -> None:
        """Add one line. Filtered-out lines are retained but not shown."""
        line = line.rstrip("\n")
        self._lines.append(line)
        if self._matches(line):
            self._view.appendPlainText(line)
            if self._autoscroll.isChecked():
                self._view.moveCursor(QTextCursor.MoveOperation.End)

    def append_block(self, text: str) -> None:
        for line in text.splitlines():
            self.append(line)

    def clear(self) -> None:
        self._lines.clear()
        self._view.clear()

    def _matches(self, line: str) -> bool:
        if not self._filter:
            return True
        if self._regex is not None:
            return self._regex.search(line) is not None
        return self._filter.lower() in line.lower()

    def _on_filter_changed(self, text: str) -> None:
        """Rebuild the view from retained history.

        A /slashed/ filter is treated as a regex; an invalid one falls back to substring
        matching rather than showing nothing, since the user is mid-typing.
        """
        self._filter = text.strip()
        self._regex = None
        if len(self._filter) >= 2 and self._filter.startswith("/") and self._filter.endswith("/"):
            try:
                self._regex = re.compile(self._filter[1:-1], re.IGNORECASE)
            except re.error:
                self._regex = None

        self._view.clear()
        matched = [line for line in self._lines if self._matches(line)]
        if matched:
            self._view.setPlainText("\n".join(matched))
            if self._autoscroll.isChecked():
                self._view.moveCursor(QTextCursor.MoveOperation.End)
