"""EVAL workspace (CLAUDE.md section 12.4).

"run picker, metrics tables, confusion matrix image, overlay browser with prev/next."
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from drivyx.gui.process import Job
from drivyx.gui.widgets.jobcard import JobCard
from drivyx.gui.widgets.panel import Panel
from drivyx.gui.widgets.statrow import StatRow
from drivyx.gui.workspaces.base import Workspace

logger = logging.getLogger(__name__)


class EvalWorkspace(Workspace):
    """Pick a run, evaluate it, and browse the results."""

    title = "EVAL"

    OWNED_COMMANDS = ("eval-seg", "eval-ctrl")

    def build(self) -> None:
        self._cards: dict[int, JobCard] = {}
        self._overlays: list[Path] = []
        self._overlay_index = 0

        # --- sidebar ---

        picker = Panel("Run")
        self._run_picker = QComboBox()
        self._run_picker.currentTextChanged.connect(self._on_run_selected)
        picker.add_widget(self._run_picker)

        refresh = QPushButton("Refresh runs")
        refresh.clicked.connect(self.refresh_artifacts)
        picker.add_widget(refresh)

        self._ckpt_picker = QComboBox()
        self._ckpt_picker.addItems(["best", "last"])
        picker.body().addWidget(QLabel("checkpoint"))
        picker.add_widget(self._ckpt_picker)
        self.add_panel(picker)

        actions = Panel("Evaluate")
        self._eval_seg_btn = QPushButton("Run eval-seg")
        self._eval_seg_btn.setProperty("primary", "true")
        self._eval_seg_btn.clicked.connect(self._on_eval_seg)
        actions.add_widget(self._eval_seg_btn)

        self._eval_ctrl_btn = QPushButton("Run eval-ctrl")
        self._eval_ctrl_btn.clicked.connect(self._on_eval_ctrl)
        actions.add_widget(self._eval_ctrl_btn)

        self._summary_rows = {
            "kind": StatRow("kind", "-", label_width=84),
            "mIoU": StatRow("mIoU", "-", label_width=84),
            "pixel acc": StatRow("pixel acc", "-", label_width=84),
            "ADE": StatRow("ADE", "-", label_width=84),
            "FDE": StatRow("FDE", "-", label_width=84),
            "lateral@1s": StatRow("lateral@1s", "-", label_width=84),
        }
        for row in self._summary_rows.values():
            actions.add_widget(row)
        self.add_panel(actions)

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

        self._views = QTabWidget()
        self._views.setDocumentMode(True)

        self._metrics = QTableWidget(0, 2)
        self._metrics.setHorizontalHeaderLabels(["metric", "value"])
        self._metrics.verticalHeader().setVisible(False)
        self._metrics.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        head = self._metrics.horizontalHeader()
        if head is not None:
            head.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            head.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._views.addTab(self._metrics, "Metrics")

        self._confusion_scroll = QScrollArea()
        self._confusion_scroll.setWidgetResizable(True)
        self._confusion_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._confusion_label = QLabel("Run eval-seg to produce a confusion matrix.")
        self._confusion_label.setProperty("dim", "true")
        self._confusion_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._confusion_scroll.setWidget(self._confusion_label)
        self._views.addTab(self._confusion_scroll, "Confusion")

        browser = QWidget()
        browser_layout = QVBoxLayout(browser)
        browser_layout.setContentsMargins(0, 6, 0, 0)

        controls = QHBoxLayout()
        prev_btn = QPushButton("Previous")
        prev_btn.clicked.connect(lambda: self._step_overlay(-1))
        controls.addWidget(prev_btn)
        next_btn = QPushButton("Next")
        next_btn.clicked.connect(lambda: self._step_overlay(1))
        controls.addWidget(next_btn)
        self._overlay_caption = QLabel("No overlays yet.")
        self._overlay_caption.setProperty("dim", "true")
        controls.addWidget(self._overlay_caption, 1)
        holder = QWidget()
        holder.setLayout(controls)
        browser_layout.addWidget(holder)

        self._overlay_view = QLabel()
        self._overlay_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        overlay_scroll = QScrollArea()
        overlay_scroll.setWidgetResizable(True)
        overlay_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        overlay_scroll.setWidget(self._overlay_view)
        browser_layout.addWidget(overlay_scroll, 1)
        self._views.addTab(browser, "Overlays")

        self.add_main(self._views)
        self.refresh_artifacts()

    # --- runs ---

    def _runs_root(self) -> Path | None:
        from drivyx.paths import get_paths

        try:
            return get_paths().runs
        except (OSError, ValueError):
            return None

    def refresh_artifacts(self) -> None:
        """Repopulate the run picker and reload the selected run's metrics."""
        root = self._runs_root()
        if root is None or not root.is_dir():
            return

        current = self._run_picker.currentText()
        runs = sorted((p.name for p in root.iterdir() if p.is_dir()), reverse=True)

        self._run_picker.blockSignals(True)
        self._run_picker.clear()
        self._run_picker.addItems(runs)
        if current in runs:
            self._run_picker.setCurrentText(current)
        self._run_picker.blockSignals(False)

        if self._run_picker.currentText():
            self._on_run_selected(self._run_picker.currentText())

    def _on_run_selected(self, name: str) -> None:
        root = self._runs_root()
        if not name or root is None:
            return
        run_dir = root / name
        # A ctrl run has no mIoU and a seg run has no ADE, so only the relevant buttons and
        # rows are meaningful. The kind is in the directory name by the section 6.3 contract.
        kind = "ctrl" if "_ctrl_" in name else "seg"
        self._summary_rows["kind"].set_value(kind)
        self._eval_seg_btn.setEnabled(kind == "seg")
        self._eval_ctrl_btn.setEnabled(kind == "ctrl")

        for row in ("mIoU", "pixel acc", "ADE", "FDE", "lateral@1s"):
            self._summary_rows[row].set_value("-")

        seg_metrics = run_dir / "eval" / "seg_metrics.json"
        ctrl_metrics = run_dir / "eval" / "ctrl_metrics.json"
        if seg_metrics.is_file():
            self._render_seg(json.loads(seg_metrics.read_text()), run_dir)
        elif ctrl_metrics.is_file():
            self._render_ctrl(json.loads(ctrl_metrics.read_text()), run_dir)
        else:
            self._metrics.setRowCount(0)
            self._confusion_label.setPixmap(QPixmap())
            self._confusion_label.setText("No evaluation yet for this run.")
            self._load_overlays([])

    # --- rendering ---

    def _set_metrics(self, rows: list[tuple[str, str]]) -> None:
        self._metrics.setRowCount(len(rows))
        for index, (name, value) in enumerate(rows):
            self._metrics.setItem(index, 0, QTableWidgetItem(name))
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._metrics.setItem(index, 1, item)

    def _render_seg(self, payload: dict[str, Any], run_dir: Path) -> None:
        self._summary_rows["mIoU"].set_value(f"{payload['mIoU']:.4f}")
        self._summary_rows["pixel acc"].set_value(f"{payload['pixel_accuracy']:.4f}")

        rows: list[tuple[str, str]] = [
            ("checkpoint", str(payload.get("checkpoint", "-"))),
            ("epoch", str(payload.get("epoch", "-"))),
            ("samples", f"{payload.get('samples', 0):,}"),
            ("mIoU", f"{payload['mIoU']:.4f}"),
            ("pixel accuracy", f"{payload['pixel_accuracy']:.4f}"),
            ("mean accuracy", f"{payload['mean_accuracy']:.4f}"),
        ]
        for name, value in payload.get("per_class_iou", {}).items():
            rows.append((f"IoU {name}", "n/a" if value is None else f"{value:.4f}"))
        self._set_metrics(rows)

        confusion = run_dir / payload.get("confusion_png", "")
        if confusion.is_file():
            pixmap = QPixmap(str(confusion))
            self._confusion_label.setText("")
            self._confusion_label.setPixmap(
                pixmap.scaledToWidth(900, Qt.TransformationMode.SmoothTransformation)
                if pixmap.width() > 900
                else pixmap
            )

        self._load_overlays([run_dir / p for p in payload.get("overlays", [])])

    def _render_ctrl(self, payload: dict[str, Any], run_dir: Path) -> None:
        self._summary_rows["ADE"].set_value(f"{payload['ade']:.4f} m")
        self._summary_rows["FDE"].set_value(f"{payload['fde']:.4f} m")
        self._summary_rows["lateral@1s"].set_value(f"{payload['lateral_1s']:.4f} m")

        rows: list[tuple[str, str]] = [
            ("checkpoint", str(payload.get("checkpoint", "-"))),
            ("seg run", str(payload.get("seg_run", "-"))),
            ("val frames", f"{payload.get('val_frames', 0):,}"),
            ("ADE", f"{payload['ade']:.4f} m"),
            ("FDE (2.5 s)", f"{payload['fde']:.4f} m"),
            ("lateral error at 1.0 s", f"{payload['lateral_1s']:.4f} m"),
        ]
        for horizon, value in zip(
            payload.get("horizons_s", []), payload.get("ade_per_horizon", [])
        ):
            rows.append((f"displacement error at {horizon:.1f} s", f"{value:.4f} m"))

        # The straight-line baseline is shown next to the model because on this dataset most
        # frames are nearly straight, so ADE alone does not say whether anything was learned.
        rows.append(("baseline (always straight) ADE", f"{payload['baseline_straight_ade']:.4f} m"))
        rows.append(
            ("beats straight baseline", "yes" if payload.get("beats_straight_baseline") else "no")
        )
        self._set_metrics(rows)

        self._confusion_label.setPixmap(QPixmap())
        self._confusion_label.setText("Confusion matrices apply to segmentation runs only.")
        self._load_overlays([run_dir / p for p in payload.get("overlays", [])])

    # --- overlay browser ---

    def _load_overlays(self, paths: list[Path]) -> None:
        self._overlays = [p for p in paths if p.is_file()]
        self._overlay_index = 0
        self._show_overlay()

    def _step_overlay(self, delta: int) -> None:
        if not self._overlays:
            return
        self._overlay_index = (self._overlay_index + delta) % len(self._overlays)
        self._show_overlay()

    def _show_overlay(self) -> None:
        if not self._overlays:
            self._overlay_view.setPixmap(QPixmap())
            self._overlay_view.setText("No overlays for this run.")
            self._overlay_caption.setText("No overlays yet.")
            return

        path = self._overlays[self._overlay_index]
        pixmap = QPixmap(str(path))
        self._overlay_view.setText("")
        self._overlay_view.setPixmap(
            pixmap.scaledToWidth(1100, Qt.TransformationMode.SmoothTransformation)
            if pixmap.width() > 1100
            else pixmap
        )
        self._overlay_caption.setText(
            f"{self._overlay_index + 1} of {len(self._overlays)}: {path.name}"
        )

    # --- job wiring ---

    def _submit(self, command: str) -> None:
        run = self._run_picker.currentText()
        if not run:
            return
        self._eval_seg_btn.setEnabled(False)
        self._eval_ctrl_btn.setEnabled(False)
        self.queue.submit(
            [command, "--run", run, "--ckpt", self._ckpt_picker.currentText()],
            f"{command} {run}",
        )

    def _on_eval_seg(self) -> None:
        self._submit("eval-seg")

    def _on_eval_ctrl(self) -> None:
        self._submit("eval-ctrl")

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
        self._on_run_selected(self._run_picker.currentText())
        self.refresh_artifacts()
