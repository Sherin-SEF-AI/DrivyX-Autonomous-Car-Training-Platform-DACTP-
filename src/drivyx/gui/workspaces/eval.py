"""EVAL workspace (CLAUDE.md section 12.4).

M1 ships the shell; M6 delivers the run picker, metrics tables, confusion matrix, and the
overlay browser, and M8 adds the video preview.
"""

from __future__ import annotations

from drivyx.gui.widgets.panel import Panel
from drivyx.gui.workspaces.base import PlaceholderView, Workspace


class EvalWorkspace(Workspace):
    title = "EVAL"

    def build(self) -> None:
        runs = Panel("Runs")
        runs.add_widget(PlaceholderView("M6", "Run picker"))
        self.add_panel(runs)

        metrics = Panel("Metrics")
        metrics.add_widget(PlaceholderView("M6", "Per-class IoU, mIoU, ADE/FDE, lateral error"))
        self.add_panel(metrics)

        self.add_main(PlaceholderView("M6", "Confusion matrix and overlay browser with prev/next"))
