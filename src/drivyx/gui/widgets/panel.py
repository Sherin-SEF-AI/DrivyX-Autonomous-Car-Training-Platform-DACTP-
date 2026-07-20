"""Collapsible panel, the Blender N-panel pattern (CLAUDE.md section 12.3).

"header row with disclosure arrow, title, optional header widget". Clicking the header
toggles the body. Section 12.1 limits motion to binary state changes, so the body is shown
or hidden outright with no animation.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class Panel(QFrame):
    """A titled, collapsible container."""

    toggled = pyqtSignal(bool)

    #: Unicode disclosure triangles, matching Blender's collapsed/expanded arrows.
    ARROW_OPEN = "▾"
    ARROW_CLOSED = "▸"

    def __init__(
        self,
        title: str,
        *,
        collapsed: bool = False,
        header_widget: QWidget | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self._collapsed = collapsed

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._header = _PanelHeader()
        self._header.clicked.connect(self.toggle)
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(6, 4, 6, 4)
        header_layout.setSpacing(6)

        self._arrow = QLabel(self.ARROW_CLOSED if collapsed else self.ARROW_OPEN)
        self._arrow.setObjectName("panelArrow")
        self._arrow.setFixedWidth(12)
        header_layout.addWidget(self._arrow)

        self._title = QLabel(title)
        self._title.setObjectName("panelTitle")
        header_layout.addWidget(self._title)
        header_layout.addStretch(1)

        if header_widget is not None:
            # Clicks on a header widget must not toggle the panel underneath it.
            header_widget.setParent(self._header)
            header_layout.addWidget(header_widget)

        outer.addWidget(self._header)

        self._body = QWidget()
        self._body.setObjectName("panelBody")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(8, 8, 8, 8)
        self._body_layout.setSpacing(6)
        outer.addWidget(self._body)

        self._body.setVisible(not collapsed)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def body(self) -> QVBoxLayout:
        """Layout to add content into."""
        return self._body_layout

    def add_widget(self, widget: QWidget) -> QWidget:
        self._body_layout.addWidget(widget)
        return widget

    def set_title(self, title: str) -> None:
        self._title.setText(title)

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    def toggle(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self._body.setVisible(not collapsed)
        self._arrow.setText(self.ARROW_CLOSED if collapsed else self.ARROW_OPEN)
        self.toggled.emit(not collapsed)


class _PanelHeader(QWidget):
    """Clickable header strip."""

    clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panelHeader")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)
