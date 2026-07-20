"""Shared workspace scaffolding (CLAUDE.md section 12.3).

"Each workspace is a QSplitter: left column of collapsible panels (Blender N-panel pattern),
center main view, bottom dock LogConsole".

The LogConsole is owned by the main window and shared across workspaces (it "follows the
active job"), so a workspace supplies only the left column and the centre view.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from drivyx.gui.process import JobQueue
from drivyx.gui.widgets.panel import Panel

#: Width of the N-panel column. Blender's is ~300 px at this font size.
SIDEBAR_WIDTH = 320


class Workspace(QWidget):
    """Base class: N-panel sidebar on the left, main view on the right."""

    #: Tab label (section 12.3).
    title = "WORKSPACE"

    def __init__(self, queue: JobQueue, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.queue = queue

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(self._splitter)

        # The sidebar scrolls: collapsible panels can exceed the window height.
        self._sidebar_scroll = QScrollArea()
        self._sidebar_scroll.setWidgetResizable(True)
        self._sidebar_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        sidebar = QWidget()
        self._sidebar_layout = QVBoxLayout(sidebar)
        self._sidebar_layout.setContentsMargins(6, 6, 6, 6)
        self._sidebar_layout.setSpacing(6)
        self._sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._sidebar_scroll.setWidget(sidebar)
        self._sidebar_scroll.setMinimumWidth(240)
        self._splitter.addWidget(self._sidebar_scroll)

        self._main = QWidget()
        self._main_layout = QVBoxLayout(self._main)
        self._main_layout.setContentsMargins(6, 6, 6, 6)
        self._main_layout.setSpacing(6)
        self._splitter.addWidget(self._main)

        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([SIDEBAR_WIDTH, 900])

        self.build()

    def build(self) -> None:
        """Populate the workspace. Overridden by subclasses."""

    def add_panel(self, panel: Panel) -> Panel:
        self._sidebar_layout.addWidget(panel)
        return panel

    def main_layout(self) -> QVBoxLayout:
        return self._main_layout

    def add_main(self, widget: QWidget) -> QWidget:
        self._main_layout.addWidget(widget)
        return widget


class PlaceholderView(QLabel):
    """Centred note naming the milestone that fills a view in.

    M1 ships the shell with empty panels by design (section 14: "M1 Shell: GUI app with
    theme, workspaces (empty panels)"). This states which milestone delivers the content so
    an empty view is never mistaken for a broken one.
    """

    def __init__(self, milestone: str, description: str, parent: QWidget | None = None) -> None:
        super().__init__(f"{description}\n\nDelivered by milestone {milestone}.", parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setProperty("dim", "true")
        self.setWordWrap(True)
