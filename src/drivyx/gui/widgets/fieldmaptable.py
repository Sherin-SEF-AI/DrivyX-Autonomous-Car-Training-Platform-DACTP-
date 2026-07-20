"""FieldMapTable (CLAUDE.md sections 8, 12.4).

"the manifest in a FieldMapTable where the user can override any column mapping; overrides
are saved back into the manifest", with "unconfirmed rows amber".

This is the confirmation surface for everything mm-inventory could only guess. It is the one
place a human is asked to look at the data and agree, so it shows the evidence behind each
proposal (the confidence, the alternative columns, the measured rate) rather than just a
value to rubber-stamp.
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from drivyx.gui.theme import tokens

logger = logging.getLogger(__name__)

COLUMNS = ("field", "proposed", "confidence", "evidence", "state")


class _NoScrollComboBox(QComboBox):
    """A combo box that ignores the mouse wheel unless it has focus.

    Qt's default is that a wheel event over an unfocused combo changes its value. In a
    scrollable table that means scrolling the list silently rewrites whichever mapping happens
    to pass under the cursor, and this table's edits are confirmations that get written
    straight into mm_manifest.json. That is how a stray scroll corrupted d1.gps.frame from
    'image_idx' to 'timestamp' (docs/DECISIONS.md D030).

    Passing the event to the parent keeps the table scrolling normally.
    """

    def wheelEvent(self, event: object) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class FieldMapTable(QWidget):
    """Editable view of a manifest's column mappings and measured fields."""

    #: (route, field, column_or_none, value_or_none, confirmed)
    changed = pyqtSignal(str, str, object, object, bool)
    confirm_all_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._manifest: dict[str, Any] | None = None
        self._rows: list[dict[str, Any]] = []
        self._loading = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QHBoxLayout()
        self._summary = QLabel("Run mm-inventory to discover the multimodal layout.")
        self._summary.setProperty("dim", "true")
        self._summary.setWordWrap(True)
        header.addWidget(self._summary, 1)

        self._confirm_all = QPushButton("Confirm all proposals")
        self._confirm_all.setProperty("primary", "true")
        self._confirm_all.clicked.connect(self.confirm_all_requested)
        self._confirm_all.setEnabled(False)
        header.addWidget(self._confirm_all)
        layout.addLayout(header)

        self._table = QTableWidget(0, len(COLUMNS))
        self._table.setHorizontalHeaderLabels(list(COLUMNS))
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        head = self._table.horizontalHeader()
        if head is not None:
            head.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            head.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            head.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            head.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
            head.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self._table, 1)

    def load(self, manifest: dict[str, Any]) -> None:
        """Render a manifest. Safe to call repeatedly."""
        self._loading = True
        try:
            self._manifest = manifest
            self._rows = list(_iter_rows(manifest))
            self._table.setRowCount(len(self._rows))
            for index, row in enumerate(self._rows):
                self._render_row(index, row)
            pending = sum(1 for r in self._rows if r["state"] != "confirmed")
            self._confirm_all.setEnabled(pending > 0)
            self._summary.setText(
                f"{len(self._rows)} mappings across {len(manifest.get('routes', {}))} routes, "
                f"{pending} unconfirmed. mm-label refuses to run until every required row is "
                "confirmed."
            )
            self._summary.setStyleSheet(
                f"color: {tokens.WARN if pending else tokens.OK}; background: transparent;"
            )
        finally:
            self._loading = False

    def _render_row(self, index: int, row: dict[str, Any]) -> None:
        confirmed = row["state"] == "confirmed"

        field = QTableWidgetItem(f"{row['route']}.{row['field']}")
        self._table.setItem(index, 0, field)

        # A column mapping is a combo of the candidates the discovery found, so a human can
        # disagree without editing JSON. A measured scalar (the clock offset) is not: its
        # value is evidence, and retyping it by hand is not a thing to encourage.
        if row["kind"] == "column":
            combo = _NoScrollComboBox()
            # StrongFocus, not WheelFocus: the combo takes focus by click or tab, never by
            # the wheel passing over it.
            combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            options = row["candidates"] or ([row["proposed"]] if row["proposed"] else [])
            for option in row["all_columns"]:
                if option not in options:
                    options.append(option)
            combo.addItems([o for o in options if o])
            if row["proposed"]:
                combo.setCurrentText(str(row["proposed"]))
            combo.currentTextChanged.connect(lambda text, r=row: self._on_column_changed(r, text))
            self._table.setCellWidget(index, 1, combo)
        else:
            value = QTableWidgetItem(str(row["proposed"]))
            value.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(index, 1, value)

        self._table.setItem(index, 2, QTableWidgetItem(str(row["confidence"])))
        self._table.setItem(index, 3, QTableWidgetItem(row["evidence"]))

        state = QTableWidgetItem("confirmed" if confirmed else "unconfirmed")
        # Section 12.4: "unconfirmed rows amber".
        state.setForeground(QColor(tokens.OK if confirmed else tokens.WARN))
        self._table.setItem(index, 4, state)

        button = QPushButton("Confirmed" if confirmed else "Confirm")
        button.setEnabled(not confirmed)
        button.clicked.connect(lambda _checked, r=row: self._on_confirm(r))
        self._table.setCellWidget(index, 4, button)

    def _on_column_changed(self, row: dict[str, Any], text: str) -> None:
        if self._loading or not text:
            return
        # A "change" back to the value already proposed is not an edit. Without this, any
        # spurious signal during population would rewrite the manifest with what it already
        # said, marking a row confirmed that the human never looked at.
        if text == row.get("proposed"):
            return
        # Choosing a column is itself a confirmation: the human has looked and decided.
        self.changed.emit(row["route"], row["field"], text, None, True)

    def _on_confirm(self, row: dict[str, Any]) -> None:
        value = row["proposed"] if row["kind"] == "scalar" else None
        column = row["proposed"] if row["kind"] == "column" else None
        self.changed.emit(row["route"], row["field"], column, value, True)


def _iter_rows(manifest: dict[str, Any]):
    """Flatten a manifest into FieldMapTable rows."""
    for route, block in sorted(manifest.get("routes", {}).items()):
        for kind in ("gps", "obd"):
            table_block = block.get(kind)
            if not table_block:
                continue
            all_columns: list[str] = []
            for table in table_block.get("tables", []):
                for column in table.get("columns", []):
                    if column not in all_columns:
                        all_columns.append(column)
            rate = table_block.get("rate") or {}
            for role, guess in sorted(table_block.get("roles", {}).items()):
                yield {
                    "route": route,
                    "field": f"{kind}.{role}",
                    "kind": "column",
                    "proposed": guess.get("column"),
                    "confidence": guess.get("confidence", "?"),
                    "candidates": list(guess.get("candidates", [])),
                    "all_columns": all_columns,
                    "state": guess.get("state", "unconfirmed"),
                    "evidence": (
                        f"{len(table_block.get('tables', []))} table(s), "
                        f"{rate.get('hz', '?')} Hz, columns: {', '.join(all_columns)}"
                    ),
                }

        offset = block.get("clock_offset")
        if offset:
            yield {
                "route": route,
                "field": "clock_offset",
                "kind": "scalar",
                "proposed": offset.get("proposed_s"),
                "confidence": offset.get("confidence", "measured"),
                "candidates": [],
                "all_columns": [],
                "state": offset.get("state", "unconfirmed"),
                "evidence": (
                    f"{offset.get('hypothesis')}; raw offset "
                    f"{offset.get('raw_offset_s')}s, residual {offset.get('residual_s')}s "
                    "(logger start skew)"
                ),
            }

        tolerance = block.get("obd_tolerance")
        if tolerance:
            yield {
                "route": route,
                "field": "obd_tolerance",
                "kind": "scalar",
                "proposed": tolerance.get("proposed_s"),
                "confidence": "measured",
                "candidates": [],
                "all_columns": [],
                "state": tolerance.get("state", "unconfirmed"),
                "evidence": (
                    f"OBD logs at {tolerance.get('measured_hz')} Hz, so one sampling "
                    f"interval is {tolerance.get('measured_median_dt_s')}s. CLAUDE.md "
                    f"section 8 specifies {tolerance.get('spec_s')}s, which would keep "
                    "about 10 percent of frames."
                ),
            }
