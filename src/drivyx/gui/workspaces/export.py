"""EXPORT workspace (CLAUDE.md section 12.4).

"precision picker, export + parity + bench pipeline as one queued sequence, results with the
33 ms budget indicator."

The three stages are submitted as separate jobs into the FIFO queue rather than one long
command, so each is individually cancellable and shows its own progress. Section 6.2 runs one
job at a time, which gives the sequencing for free.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from drivyx.gui.process import Job
from drivyx.gui.theme import tokens
from drivyx.gui.widgets.jobcard import JobCard
from drivyx.gui.widgets.panel import Panel
from drivyx.gui.widgets.statrow import StatRow
from drivyx.gui.workspaces.base import Workspace

logger = logging.getLogger(__name__)

#: Section 11: seg and ctrl combined must fit one 30 fps frame.
FRAME_BUDGET_MS = 33.0


class ExportWorkspace(Workspace):
    """Export a run to TensorRT, verify parity, and measure latency."""

    title = "EXPORT"

    OWNED_COMMANDS = ("export", "bench")

    def build(self) -> None:
        self._cards: dict[int, JobCard] = {}

        # --- sidebar ---

        source = Panel("Source")
        self._run_picker = QComboBox()
        source.body().addWidget(QLabel("run"))
        source.add_widget(self._run_picker)

        self._model_picker = QComboBox()
        self._model_picker.addItems(["seg", "ctrl"])
        source.body().addWidget(QLabel("model"))
        source.add_widget(self._model_picker)

        self._precision_picker = QComboBox()
        self._precision_picker.addItems(["fp16", "int8"])
        source.body().addWidget(QLabel("precision"))
        source.add_widget(self._precision_picker)

        refresh = QPushButton("Refresh runs")
        refresh.clicked.connect(self.refresh_artifacts)
        source.add_widget(refresh)
        self.add_panel(source)

        pipeline = Panel("Pipeline")
        self._export_btn = QPushButton("Export, parity, bench")
        self._export_btn.setProperty("primary", "true")
        self._export_btn.clicked.connect(self._on_export)
        pipeline.add_widget(self._export_btn)

        self._bench_btn = QPushButton("Benchmark existing engines")
        self._bench_btn.clicked.connect(self._on_bench)
        pipeline.add_widget(self._bench_btn)

        self._status_rows = {
            "engines": StatRow("engines", "-", label_width=84),
            "parity": StatRow("parity", "-", state="queued", label_width=84),
            "combined p50": StatRow("combined p50", "-", state="queued", label_width=84),
            "budget": StatRow("budget", f"{FRAME_BUDGET_MS:.0f} ms", label_width=84),
        }
        for row in self._status_rows.values():
            pipeline.add_widget(row)
        self.add_panel(pipeline)

        jobs = Panel("Jobs")
        self._jobs_container = QWidget()
        self._jobs_layout = QVBoxLayout(self._jobs_container)
        self._jobs_layout.setContentsMargins(0, 0, 0, 0)
        self._jobs_layout.setSpacing(4)
        self._jobs_empty = QLabel("No jobs yet.")
        self._jobs_empty.setProperty("dim", "true")
        self._jobs_layout.addWidget(self._jobs_empty)
        jobs.add_widget(self._jobs_container)
        self.add_panel(jobs)

        # --- main view ---

        self._budget_banner = QLabel("No benchmark yet.")
        self._budget_banner.setProperty("dim", "true")
        self._budget_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.add_main(self._budget_banner)

        self._engines = QTableWidget(0, 6)
        self._engines.setHorizontalHeaderLabels(
            ["engine", "p50 ms", "p95 ms", "p99 ms", "qps", "size MB"]
        )
        self._engines.verticalHeader().setVisible(False)
        self._engines.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        head = self._engines.horizontalHeader()
        if head is not None:
            head.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            for column in range(1, 6):
                head.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.add_main(self._engines)

        self._parity = QTableWidget(0, 6)
        self._parity.setHorizontalHeaderLabels(
            ["model", "precision", "metric", "torch", "engine", "delta"]
        )
        self._parity.verticalHeader().setVisible(False)
        self._parity.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        head = self._parity.horizontalHeader()
        if head is not None:
            head.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._parity.setMaximumHeight(160)
        self.add_main(self._parity)

        self.refresh_artifacts()

    # --- artifacts ---

    def refresh_artifacts(self) -> None:
        from drivyx.paths import get_paths

        try:
            paths = get_paths()
        except (OSError, ValueError):
            return

        if paths.runs.is_dir():
            current = self._run_picker.currentText()
            runs = sorted((p.name for p in paths.runs.iterdir() if p.is_dir()), reverse=True)
            self._run_picker.blockSignals(True)
            self._run_picker.clear()
            self._run_picker.addItems(runs)
            if current in runs:
                self._run_picker.setCurrentText(current)
            self._run_picker.blockSignals(False)

        engines = sorted(paths.export.glob("*.engine")) if paths.export.is_dir() else []
        self._status_rows["engines"].set_value(f"{len(engines)} built" if engines else "none built")

        bench = paths.export / "bench.json"
        if bench.is_file():
            try:
                self._render_bench(json.loads(bench.read_text()), engines)
            except (OSError, ValueError, KeyError) as exc:
                logger.warning("cannot read %s: %s", bench, exc)

        parity = paths.export / "parity.json"
        if parity.is_file():
            try:
                self._render_parity(json.loads(parity.read_text()))
            except (OSError, ValueError, KeyError) as exc:
                logger.warning("cannot read %s: %s", parity, exc)

    def _render_bench(self, payload: dict[str, Any], engines: list[Path]) -> None:
        """Latency table plus the section 11 budget indicator."""
        sizes = {p.name: p.stat().st_size / 1e6 for p in engines}
        rows = payload.get("engines", [])
        self._engines.setRowCount(len(rows))

        for index, entry in enumerate(rows):
            name = Path(entry["engine"]).name
            values = [
                name,
                f"{entry.get('p50_ms', 0):.2f}",
                f"{entry.get('p95_ms', 0):.2f}",
                f"{entry.get('p99_ms', 0):.2f}",
                f"{entry.get('throughput_qps', 0):.0f}",
                f"{sizes.get(name, 0):.1f}",
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self._engines.setItem(index, column, item)

        combined = payload.get("combined_p50_ms", 0.0)
        budget = payload.get("frame_budget_ms", FRAME_BUDGET_MS)
        status = payload.get("budget_status", "err")
        colour = {"ok": tokens.OK, "warn": tokens.WARN}.get(status, tokens.ERR)
        headroom = budget - combined

        self._status_rows["combined p50"].set_state(status)
        self._status_rows["combined p50"].set_value(f"{combined:.2f} ms")
        self._budget_banner.setText(
            f"combined p50 {combined:.2f} ms against a {budget:.0f} ms frame budget "
            f"({headroom:+.2f} ms headroom, {100 * combined / budget:.0f}% used)"
        )
        self._budget_banner.setStyleSheet(f"color: {colour}; font-weight: bold;")

    def _render_parity(self, payload: dict[str, Any]) -> None:
        checks = payload.get("checks", [])
        self._parity.setRowCount(len(checks))
        for index, check in enumerate(checks):
            values = [
                check.get("model", ""),
                check.get("precision", ""),
                check.get("metric", ""),
                f"{check.get('torch', 0):.4f}",
                f"{check.get('engine', 0):.4f}",
                f"{check.get('delta', 0):.4f} / {check.get('tolerance', 0):.2f}",
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if not check.get("passed", False):
                    from PyQt6.QtGui import QColor

                    item.setForeground(QColor(tokens.ERR))
                self._parity.setItem(index, column, item)

        passed = payload.get("passed", False)
        self._status_rows["parity"].set_state("ok" if passed else "err")
        self._status_rows["parity"].set_value("pass" if passed else "FAIL")

    # --- jobs ---

    def _on_export(self) -> None:
        """Section 12.4: export, parity, and bench as one queued sequence.

        Submitted as two jobs because `export` already performs parity and a single-engine
        benchmark; the trailing `bench` measures every engine together, which is what the
        combined 33 ms budget is about.
        """
        run = self._run_picker.currentText()
        if not run:
            return
        model = self._model_picker.currentText()
        precision = self._precision_picker.currentText()

        self._export_btn.setEnabled(False)
        self._bench_btn.setEnabled(False)
        self.queue.submit(
            ["export", "--model", model, "--run", run, "--precision", precision],
            f"export {model} {precision}",
        )
        self.queue.submit(["bench"], "bench all engines")

    def _on_bench(self) -> None:
        self._bench_btn.setEnabled(False)
        self.queue.submit(["bench"], "bench all engines")

    def _owns(self, job: Job) -> bool:
        return bool(job.args) and job.args[0] in self.OWNED_COMMANDS

    def on_job_added(self, job: Job) -> None:
        if not self._owns(job):
            return
        self._jobs_empty.setVisible(False)
        card = JobCard(job)
        card.cancel_requested.connect(self.queue.cancel)
        self._cards[job.job_id] = card
        self._jobs_layout.addWidget(card)

    def on_job_state_changed(self, job: Job) -> None:
        card = self._cards.get(job.job_id)
        if card is not None:
            card.refresh()

    def on_job_finished(self, job: Job) -> None:
        card = self._cards.get(job.job_id)
        if card is not None:
            card.refresh()
        if not self._owns(job):
            return
        self._export_btn.setEnabled(True)
        self._bench_btn.setEnabled(True)
        self.refresh_artifacts()
