"""EXPORT workspace (CLAUDE.md section 12.4).

M1 ships the shell; M7 delivers the precision picker, the export/parity/bench queued
sequence, and the 33 ms budget indicator.
"""

from __future__ import annotations

from drivyx.gui.widgets.panel import Panel
from drivyx.gui.workspaces.base import PlaceholderView, Workspace


class ExportWorkspace(Workspace):
    title = "EXPORT"

    def build(self) -> None:
        precision = Panel("Precision")
        precision.add_widget(PlaceholderView("M7", "fp16 / int8 picker"))
        self.add_panel(precision)

        pipeline = Panel("Pipeline")
        pipeline.add_widget(PlaceholderView("M7", "export, parity, bench as one queued sequence"))
        self.add_panel(pipeline)

        self.add_main(
            PlaceholderView(
                "M7", "Latency p50/p95/p99 against the 33 ms frame budget, and parity results"
            )
        )
