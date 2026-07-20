"""TRAIN workspace (CLAUDE.md section 12.4).

"config editor (form generated from the pydantic schema, YAML round-trip), probe results as a
schedule table (size, secs/epoch, projected total), Start/Resume, live pyqtgraph loss and mIoU
curves from events.jsonl, epoch table, latest val overlay thumbnails."

The curves are fed by the JobQueue's event stream (section 6.4), which the QProcess wrapper
already tails incrementally. This workspace never reads a checkpoint or imports torch: it
plots numbers the engine emitted.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
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

#: Scalars plotted on the loss chart, and the colour each gets.
LOSS_SERIES = {
    "train/loss": tokens.ACCENT,
    "train/main": tokens.OK,
    "train/aux": tokens.WARN,
    "train/boundary": tokens.ERR,
}
#: Scalars plotted on the metric chart.
METRIC_SERIES = {"val/mIoU": tokens.OK}

DEFAULT_CONFIG = "configs/seg_pidnet_s.yaml"


class TrainWorkspace(Workspace):
    """Configure, launch, and watch a training run."""

    title = "TRAIN"

    OWNED_COMMANDS = ("train-seg", "train-ctrl")

    def build(self) -> None:
        self._cards: dict[int, JobCard] = {}
        self._series: dict[str, tuple[list[float], list[float]]] = {}
        self._curves: dict[str, Any] = {}
        self._epochs: list[dict[str, Any]] = []
        self._config_widgets: dict[str, QWidget] = {}

        # --- sidebar ---

        run_panel = Panel("Run")
        self._config_edit = QLineEdit(DEFAULT_CONFIG)
        run_panel.body().addWidget(QLabel("config"))
        run_panel.add_widget(self._config_edit)

        self._start_btn = QPushButton("Start training")
        self._start_btn.setProperty("primary", "true")
        self._start_btn.clicked.connect(self._on_start)
        run_panel.add_widget(self._start_btn)

        buttons = QHBoxLayout()
        self._probe_btn = QPushButton("Probe")
        self._probe_btn.clicked.connect(self._on_probe)
        buttons.addWidget(self._probe_btn)
        self._resume_btn = QPushButton("Resume latest")
        self._resume_btn.clicked.connect(self._on_resume)
        buttons.addWidget(self._resume_btn)
        holder = QWidget()
        holder.setLayout(buttons)
        run_panel.add_widget(holder)

        self._run_rows = {
            "run": StatRow("run", "-", label_width=76),
            "epoch": StatRow("epoch", "-", label_width=76),
            "step": StatRow("step", "-", label_width=76),
            "loss": StatRow("loss", "-", label_width=76),
            "val mIoU": StatRow("val mIoU", "-", label_width=76),
            "eta": StatRow("eta", "-", label_width=76),
        }
        for row in self._run_rows.values():
            run_panel.add_widget(row)
        self.add_panel(run_panel)

        # Section 12.4: "config editor (form generated from the pydantic schema)".
        config_panel = Panel("Config", collapsed=True)
        self._config_form = QFormLayout()
        self._config_form.setContentsMargins(0, 0, 0, 0)
        form_host = QWidget()
        form_host.setLayout(self._config_form)
        config_panel.add_widget(form_host)
        save_btn = QPushButton("Save to YAML")
        save_btn.clicked.connect(self._on_save_config)
        config_panel.add_widget(save_btn)
        self.add_panel(config_panel)

        jobs_panel = Panel("Jobs")
        self._jobs_container = QWidget()
        self._jobs_layout = QVBoxLayout(self._jobs_container)
        self._jobs_layout.setContentsMargins(0, 0, 0, 0)
        self._jobs_layout.setSpacing(4)
        self._jobs_empty = QLabel("No jobs yet.")
        self._jobs_empty.setProperty("dim", "true")
        self._jobs_layout.addWidget(self._jobs_empty)
        jobs_panel.add_widget(self._jobs_container)
        self.add_panel(jobs_panel)

        # --- main view ---

        self._views = QTabWidget()
        self._views.setDocumentMode(True)

        self._curves_host = QWidget()
        self._curves_layout = QVBoxLayout(self._curves_host)
        self._curves_layout.setContentsMargins(0, 6, 0, 0)
        self._curves_status = QLabel("Start a run to see live curves.")
        self._curves_status.setProperty("dim", "true")
        self._curves_layout.addWidget(self._curves_status)
        self._views.addTab(self._curves_host, "Curves")
        self._build_curves()

        self._schedule = QTableWidget(0, 6)
        self._schedule.setHorizontalHeaderLabels(
            ["size", "s/batch", "img/s", "s/epoch", "projected", "peak GB"]
        )
        self._schedule.verticalHeader().setVisible(False)
        self._schedule.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        head = self._schedule.horizontalHeader()
        if head is not None:
            head.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._views.addTab(self._schedule, "Schedule")

        self._epoch_table = QTableWidget(0, 5)
        self._epoch_table.setHorizontalHeaderLabels(
            ["epoch", "loss", "val mIoU", "secs", "eta (min)"]
        )
        self._epoch_table.verticalHeader().setVisible(False)
        self._epoch_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        head = self._epoch_table.horizontalHeader()
        if head is not None:
            head.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._views.addTab(self._epoch_table, "Epochs")

        self.add_main(self._views)
        self._load_config_form()
        self.refresh_artifacts()

    # --- curves ---

    def _build_curves(self) -> None:
        """Create the two pyqtgraph plots, imported lazily to protect the launch budget."""
        try:
            import pyqtgraph as pg
        except ImportError as exc:
            self._curves_status.setText(f"pyqtgraph unavailable: {exc}")
            return

        pg.setConfigOption("background", tokens.BG_INPUT)
        pg.setConfigOption("foreground", tokens.TEXT_DIM)

        self._loss_plot = pg.PlotWidget()
        self._loss_plot.setLabel("left", "loss")
        self._loss_plot.setLabel("bottom", "step")
        self._loss_plot.showGrid(x=True, y=True, alpha=0.2)
        self._loss_plot.addLegend(offset=(-10, 10))
        self._curves_layout.addWidget(self._loss_plot, 2)

        self._metric_plot = pg.PlotWidget()
        self._metric_plot.setLabel("left", "val mIoU")
        self._metric_plot.setLabel("bottom", "step")
        self._metric_plot.showGrid(x=True, y=True, alpha=0.2)
        self._metric_plot.setYRange(0, 1)
        self._curves_layout.addWidget(self._metric_plot, 1)

        for name, color in LOSS_SERIES.items():
            self._curves[name] = self._loss_plot.plot(
                pen=pg.mkPen(color, width=1), name=name.split("/")[-1]
            )
        for name, color in METRIC_SERIES.items():
            self._curves[name] = self._metric_plot.plot(
                pen=pg.mkPen(color, width=2), symbol="o", symbolSize=5, symbolBrush=color
            )

    def on_event(self, event: Any) -> None:
        """Consume one events.jsonl event (section 6.4)."""
        kind = getattr(event, "type", None)
        if kind == "scalar":
            self._on_scalar(event)
        elif kind == "epoch":
            self._on_epoch(event)
        elif kind == "status":
            self._on_status(event)

    def _on_scalar(self, event: Any) -> None:
        name = event.name
        if name is None:
            return
        xs, ys = self._series.setdefault(name, ([], []))
        xs.append(float(event.step if event.step is not None else len(xs)))
        ys.append(float(event.value))

        curve = self._curves.get(name)
        if curve is not None:
            curve.setData(xs, ys)

        if name == "train/loss":
            self._run_rows["loss"].set_value(f"{event.value:.4f}")
            self._run_rows["step"].set_value(f"{event.step:,}")
            self._curves_status.setVisible(False)
        elif name == "val/mIoU":
            self._run_rows["val mIoU"].set_value(f"{event.value:.4f}")

    def _on_epoch(self, event: Any) -> None:
        raw = event.raw
        self._epochs.append(raw)
        self._run_rows["epoch"].set_value(str(raw.get("epoch", "-")))
        eta = raw.get("eta_min")
        if eta is not None:
            hours, minutes = divmod(int(eta), 60)
            self._run_rows["eta"].set_value(f"{hours}h {minutes:02d}m" if hours else f"{minutes}m")
        self._render_epochs()

    def _on_status(self, event: Any) -> None:
        value = str(event.value)
        self._run_rows["run"].set_state(
            {"done": "ok", "failed": "err", "interrupted": "warn"}.get(value, "running")
        )

    def _render_epochs(self) -> None:
        self._epoch_table.setRowCount(len(self._epochs))
        losses = {x: y for x, y in zip(*self._series.get("train/epoch_loss", ([], [])))}
        mious = dict(zip(*self._series.get("val/mIoU", ([], []))))
        for row, entry in enumerate(self._epochs):
            epoch = entry.get("epoch")
            values = [
                str(epoch),
                _nearest(losses, epoch),
                _nearest(mious, epoch),
                f"{entry.get('secs', 0):.1f}",
                f"{entry.get('eta_min', 0):.1f}",
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if col:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self._epoch_table.setItem(row, col, item)
        self._epoch_table.scrollToBottom()

    # --- job wiring ---

    def _submit(self, args: list[str], title: str) -> None:
        from drivyx.paths import get_paths

        self._set_buttons_enabled(False)
        # Passing the run dir lets the QProcess wrapper tail its events.jsonl (section 6.4).
        run_dir = None
        try:
            runs = sorted(p for p in get_paths().runs.iterdir() if p.is_dir())
            run_dir = runs[-1] if runs and "--resume" in args else None
        except (OSError, ValueError):
            run_dir = None
        self.queue.submit(args, title, run_dir=run_dir)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        for button in (self._start_btn, self._probe_btn, self._resume_btn):
            button.setEnabled(enabled)

    def _on_start(self) -> None:
        self._reset_series()
        self._submit(["train-seg", "--config", self._config_edit.text()], "train-seg")

    def _on_probe(self) -> None:
        self._submit(["train-seg", "--config", self._config_edit.text(), "--probe"], "probe")

    def _on_resume(self) -> None:
        self._submit(
            ["train-seg", "--config", self._config_edit.text(), "--resume", "latest"],
            "train-seg resume",
        )

    def _reset_series(self) -> None:
        self._series.clear()
        self._epochs.clear()
        for curve in self._curves.values():
            curve.setData([], [])
        self._epoch_table.setRowCount(0)

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
        self._set_buttons_enabled(True)
        self.refresh_artifacts()

    # --- artifacts ---

    def refresh_artifacts(self) -> None:
        """Re-read the newest run's probe.json and events.jsonl from disk."""
        from drivyx.paths import get_paths

        try:
            runs = sorted(p for p in get_paths().runs.iterdir() if p.is_dir())
        except (OSError, ValueError):
            return
        if not runs:
            return

        latest = runs[-1]
        self._run_rows["run"].set_value(latest.name)

        probe = latest / "probe.json"
        if not probe.is_file():
            # The newest run may be a training run; fall back to the newest probe anywhere.
            probes = [p / "probe.json" for p in reversed(runs) if (p / "probe.json").is_file()]
            probe = probes[0] if probes else probe
        if probe.is_file():
            try:
                self._render_schedule(json.loads(probe.read_text()))
            except (OSError, ValueError, KeyError) as exc:
                logger.warning("cannot read %s: %s", probe, exc)

        self._replay_events(latest / "events.jsonl")

    def _replay_events(self, path: Path) -> None:
        """Load a finished run's curves, so selecting a run shows its history.

        Uses the same parser as the live tailer, so a replayed curve and a live one cannot
        disagree.
        """
        if not path.is_file():
            return
        from drivyx.jobs.events import read_events

        self._reset_series()
        for event in read_events(path):
            self.on_event(event)

    def _render_schedule(self, payload: dict[str, Any]) -> None:
        """Section 12.4: probe results as a schedule table."""
        sizes = payload.get("sizes", [])
        self._schedule.setRowCount(len(sizes))
        for row, entry in enumerate(sizes):
            hours = entry.get("projected_total_hours")
            projected = "OOM" if entry.get("oom") else f"{hours:.1f} h ({hours / 24:.1f} d)"
            values = [
                f"{entry['width']}x{entry['height']}",
                f"{entry.get('secs_per_batch', 0):.3f}",
                f"{entry.get('images_per_sec', 0):.1f}",
                f"{entry.get('secs_per_epoch', 0):.0f}",
                projected,
                f"{entry.get('peak_memory_gb', 0):.2f}",
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if col:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                if entry.get("oom"):
                    from PyQt6.QtGui import QColor

                    item.setForeground(QColor(tokens.ERR))
                self._schedule.setItem(row, col, item)

    # --- config form (section 12.4: generated from the pydantic schema) ---

    def _load_config_form(self) -> None:
        """Build the form from SegConfig's schema, not a hardcoded field list.

        Generating it means a field added to the schema appears here automatically, and the
        types and bounds shown are the ones pydantic will actually enforce.
        """
        from drivyx.train.config import SegConfig

        while self._config_form.rowCount():
            self._config_form.removeRow(0)
        self._config_widgets.clear()

        schema = SegConfig.model_json_schema()
        defaults = SegConfig()
        for name, spec in schema.get("properties", {}).items():
            value = getattr(defaults, name, None)
            widget = _widget_for(spec, value)
            if widget is None:
                continue
            self._config_widgets[name] = widget
            label = QLabel(name)
            if spec.get("description"):
                label.setToolTip(spec["description"])
                widget.setToolTip(spec["description"])
            self._config_form.addRow(label, widget)

    def _on_save_config(self) -> None:
        """Write the edited fields back to YAML (section 12.4: YAML round-trip)."""
        import yaml

        from drivyx.train.config import SegConfig

        path = Path(self._config_edit.text())
        try:
            raw = yaml.safe_load(path.read_text()) if path.is_file() else {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("cannot read %s: %s", path, exc)
            return

        for name, widget in self._config_widgets.items():
            raw[name] = _value_of(widget)

        try:
            SegConfig(**raw)
        except Exception as exc:
            logger.warning("refusing to save an invalid config: %s", exc)
            return

        path.write_text(yaml.safe_dump(raw, sort_keys=False))
        logger.info("wrote %s", path)


def _widget_for(spec: dict[str, Any], value: Any) -> QWidget | None:
    """Map one schema property to an input widget. Nested models are skipped."""
    if isinstance(value, bool):
        widget = QCheckBox()
        widget.setChecked(value)
        return widget
    if isinstance(value, int):
        widget = QSpinBox()
        widget.setRange(int(spec.get("minimum", 0)), int(spec.get("maximum", 1_000_000)))
        widget.setValue(value)
        return widget
    if isinstance(value, float):
        widget = QDoubleSpinBox()
        widget.setDecimals(6)
        widget.setRange(float(spec.get("minimum", 0.0)), float(spec.get("maximum", 1e6)))
        widget.setValue(value)
        return widget
    if isinstance(value, str):
        if "enum" in spec or "const" in spec:
            widget = QComboBox()
            widget.addItems([str(v) for v in spec.get("enum", [spec.get("const")])])
            widget.setCurrentText(value)
            return widget
        widget = QLineEdit(value)
        return widget
    return None


def _value_of(widget: QWidget) -> Any:
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QSpinBox):
        return widget.value()
    if isinstance(widget, QDoubleSpinBox):
        return widget.value()
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, QLineEdit):
        return widget.text()
    return None


def _nearest(series: dict[float, float], epoch: int | None) -> str:
    """The series value recorded closest to an epoch boundary.

    Scalars carry a step, epochs carry an index, and the two only align at the boundary, so
    the table looks up by proximity rather than assuming a shared x axis.
    """
    if epoch is None or not series:
        return "-"
    return f"{list(series.values())[min(epoch, len(series) - 1)]:.4f}"
