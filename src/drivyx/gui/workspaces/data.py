"""DATA workspace (CLAUDE.md section 12.4).

Renders the verify-data report (counts table + red/green checks), the LUT as a
colour-swatched table, and the shard class-pixel histogram as a bar chart, plus triggers for
gen-masks, build-lut, and pack-shards and the job card list.
"""

from __future__ import annotations

import json
import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
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


class DataWorkspace(Workspace):
    """Data verification and preparation."""

    title = "DATA"

    def build(self) -> None:
        self._cards: dict[int, JobCard] = {}
        self._report: dict | None = None

        # --- sidebar ---

        verify_panel = Panel("Verify")
        self._verify_btn = QPushButton("Run verify-data")
        self._verify_btn.setProperty("primary", "true")
        self._verify_btn.clicked.connect(self.run_verify)
        verify_panel.add_widget(self._verify_btn)

        self._summary_rows = {
            "status": StatRow("status", "not run", state="queued", label_width=78),
            "images": StatRow("images", "-", label_width=78),
            "sequences": StatRow("sequences", "-", label_width=78),
            "multimodal": StatRow("multimodal", "-", label_width=78),
            "backbone": StatRow("backbone", "-", state="queued", label_width=78),
            "disk free": StatRow("disk free", "-", label_width=78),
        }
        for row in self._summary_rows.values():
            verify_panel.add_widget(row)
        self.add_panel(verify_panel)

        # Section 12.4: "buttons for gen-masks, build-lut (renders lut.json as a
        # color-swatched table), pack-shards". Ordered as the pipeline runs, since each stage
        # consumes the previous one's output.
        prep_panel = Panel("Prepare")
        self._prep_buttons: dict[str, QPushButton] = {}
        for label, args, title in (
            ("gen-masks", ["gen-masks"], "gen-masks"),
            ("build-lut", ["build-lut"], "build-lut"),
            ("pack-shards (train)", ["pack-shards", "--split", "train"], "pack-shards train"),
            ("pack-shards (val)", ["pack-shards", "--split", "val"], "pack-shards val"),
        ):
            button = QPushButton(label)
            # Section 12.1: never more than one accent-coloured primary button per panel, so
            # these stay default-styled; verify-data above is this workspace's primary action.
            button.clicked.connect(lambda _checked, a=args, t=title: self._submit(a, t))
            prep_panel.add_widget(button)
            self._prep_buttons[label] = button

        self._prep_status = StatRow("shards", "not packed", state="queued", label_width=78)
        prep_panel.add_widget(self._prep_status)
        self.add_panel(prep_panel)

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

        header = QHBoxLayout()
        self._headline = QLabel("Run verify-data to inventory the data on disk.")
        self._headline.setProperty("dim", "true")
        header.addWidget(self._headline)
        header.addStretch(1)
        header_widget = QWidget()
        header_widget.setLayout(header)
        self.add_main(header_widget)

        self._counts = QTableWidget(0, 6)
        self._counts.setHorizontalHeaderLabels(
            ["split", "images", "polygons", "sequences", "unpaired", "formats"]
        )
        self._counts.verticalHeader().setVisible(False)
        self._counts.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._counts.setMaximumHeight(150)
        counts_header = self._counts.horizontalHeader()
        if counts_header is not None:
            counts_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.add_main(self._counts)

        self._checks = QTableWidget(0, 3)
        self._checks.setHorizontalHeaderLabels(["", "check", "detail"])
        self._checks.verticalHeader().setVisible(False)
        self._checks.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        checks_header = self._checks.horizontalHeader()
        if checks_header is not None:
            checks_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
            checks_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            checks_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._checks.setColumnWidth(0, 26)

        # The verify report, the LUT, and the shard histogram are three separate readings of
        # the data, so they get their own tabs rather than competing for one column.
        self._views = QTabWidget()
        self._views.setDocumentMode(True)

        inventory = QWidget()
        inventory_layout = QVBoxLayout(inventory)
        inventory_layout.setContentsMargins(0, 6, 0, 0)
        inventory_layout.addWidget(self._counts)
        inventory_layout.addWidget(self._checks, 1)
        self._views.addTab(inventory, "Inventory")

        self._lut_table = QTableWidget(0, 5)
        self._lut_table.setHorizontalHeaderLabels(
            ["", "train id", "class", "level3Ids", "source labels"]
        )
        self._lut_table.verticalHeader().setVisible(False)
        self._lut_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lut_header = self._lut_table.horizontalHeader()
        if lut_header is not None:
            lut_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
            lut_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            lut_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            lut_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            lut_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self._lut_table.setColumnWidth(0, 34)
        self._views.addTab(self._lut_table, "LUT")

        self._histogram_host = QWidget()
        self._histogram_layout = QVBoxLayout(self._histogram_host)
        self._histogram_layout.setContentsMargins(0, 6, 0, 0)
        self._histogram_status = QLabel("Run pack-shards to build the class histogram.")
        self._histogram_status.setProperty("dim", "true")
        self._histogram_layout.addWidget(self._histogram_status)
        self._histogram_plot: object | None = None
        self._views.addTab(self._histogram_host, "Histogram")

        self.add_main(self._views)
        self.refresh_artifacts()

    # --- job wiring ---

    #: Commands this workspace owns. Job callbacks ignore anything else, so another
    #: workspace's jobs never render a card here.
    OWNED_COMMANDS = ("verify-data", "gen-masks", "build-lut", "pack-shards")

    def run_verify(self) -> None:
        """Submit verify-data. Section 2: the GUI only ever shells out to the CLI."""
        self._headline.setText("Running verify-data ...")
        self._summary_rows["status"].set_state("running")
        self._summary_rows["status"].set_value("running")
        self._submit(["verify-data"], "verify-data")

    def _submit(self, args: list[str], title: str) -> None:
        """Queue a CLI command and disable the buttons while it runs.

        Section 6.2 keeps one job at a time, so leaving the buttons live would only queue
        work the user cannot see the result of yet.
        """
        self._set_buttons_enabled(False)
        self.queue.submit(args, title)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self._verify_btn.setEnabled(enabled)
        for button in self._prep_buttons.values():
            button.setEnabled(enabled)

    def _owns(self, job: Job) -> bool:
        return bool(job.args) and job.args[0] in self.OWNED_COMMANDS

    def on_job_added(self, job: Job) -> None:
        if not self._owns(job):
            return
        if self._jobs_empty.isVisible():
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

        if job.args[0] != "verify-data":
            # gen-masks, build-lut, and pack-shards write artifacts to disk; re-read them
            # rather than parsing stdout, so the panel shows what is actually on disk.
            self.refresh_artifacts()
            return

        try:
            report = json.loads(job.stdout_text)
        except json.JSONDecodeError as exc:
            # verify-data emits JSON on stdout even when it fails; unparseable output means
            # the process died before writing, and the LogConsole holds the reason.
            self._headline.setText(
                f"verify-data produced no readable report (exit {job.exit_code}). "
                "See the log below."
            )
            self._summary_rows["status"].set_state("err")
            self._summary_rows["status"].set_value(f"exit {job.exit_code}")
            logger.warning("could not parse verify-data stdout: %s", exc)
            return

        self.render_report(report)

    # --- artifacts on disk ---

    def refresh_artifacts(self) -> None:
        """Re-read lut.json and shards/index.json and re-render their views.

        Reading the artifacts is cheap (two small JSON files), so this runs on the main
        thread. Anything heavier would need a worker per rule 26.
        """
        from drivyx.paths import get_paths

        try:
            paths = get_paths()
        except (OSError, ValueError) as exc:
            logger.warning("cannot resolve paths: %s", exc)
            return

        try:
            from drivyx.data.lut import read_lut

            self.render_lut(read_lut(paths.lut_json))
        except (FileNotFoundError, ValueError, KeyError):
            self._lut_table.setRowCount(0)

        try:
            from drivyx.data.shards import read_index

            self.render_histogram(read_index(paths))
        except (FileNotFoundError, ValueError, KeyError):
            self._prep_status.set_state("queued")
            self._prep_status.set_value("not packed")

    def render_lut(self, document: dict) -> None:
        """Section 12.4: render lut.json as a colour-swatched table."""
        from PyQt6.QtGui import QColor

        rows = list(document["groups"]) + [
            {
                "train_id": document["ignore"]["train_id"],
                "name": "ignore",
                "color": [40, 40, 40],
                "level3_ids": document["ignore"]["level3_ids"],
                "members": document["ignore"]["members"],
            }
        ]
        self._lut_table.setRowCount(len(rows))
        for row, group in enumerate(rows):
            swatch = QTableWidgetItem("")
            swatch.setBackground(QColor(*group["color"]))
            self._lut_table.setItem(row, 0, swatch)

            for col, text in enumerate(
                (
                    str(group["train_id"]),
                    group["name"],
                    ", ".join(str(i) for i in group["level3_ids"]),
                    ", ".join(group["members"]),
                ),
                start=1,
            ):
                self._lut_table.setItem(row, col, QTableWidgetItem(text))

    def render_histogram(self, index: dict) -> None:
        """Section 12.4: the class pixel histogram as a bar chart.

        pyqtgraph is imported lazily, as everywhere in the GUI, to protect the launch budget.
        """
        splits = index.get("splits", {})
        train = splits.get("train")
        summary = "  ".join(
            f"{name}: {data['samples']:,} in {data['shards']} shards"
            for name, data in sorted(splits.items())
        )
        self._prep_status.set_state("ok" if train else "warn")
        self._prep_status.set_value(summary or "not packed")

        source = train or next(iter(splits.values()), None)
        if source is None:
            return

        try:
            import pyqtgraph as pg
        except ImportError as exc:
            self._histogram_status.setText(f"pyqtgraph unavailable: {exc}")
            return

        counts = source["class_pixels"]
        names = index["classes"]
        total = sum(counts)
        if total == 0:
            self._histogram_status.setText("The histogram is empty: no labelled pixels.")
            return

        if self._histogram_plot is None:
            pg.setConfigOption("background", tokens.BG_INPUT)
            pg.setConfigOption("foreground", tokens.TEXT_DIM)
            plot = pg.PlotWidget()
            plot.setLabel("left", "share of labelled pixels (%)")
            plot.showGrid(x=False, y=True, alpha=0.2)
            plot.setMouseEnabled(x=False, y=False)
            plot.hideButtons()
            self._histogram_layout.addWidget(plot, 1)
            self._histogram_plot = plot
        plot = self._histogram_plot

        plot.clear()
        percentages = [100.0 * c / total for c in counts]
        # One bar per class, coloured with that class's LUT swatch so the chart and the LUT
        # table read as the same object.
        from drivyx.data.lut import GROUP_COLORS

        for i, (pct, color) in enumerate(zip(percentages, GROUP_COLORS)):
            bar = pg.BarGraphItem(x=[i], height=[pct], width=0.7, brush=pg.mkBrush(color))
            plot.addItem(bar)

        axis = plot.getAxis("bottom")
        axis.setTicks([[(i, name) for i, name in enumerate(names)]])

        ignored = source["ignore_pixels"]
        # The ignore share is ~0.1% on IDD, whose polygons tile the frame, so a fixed G scale
        # renders a real 5.9M count as "0.0 G" and reads as a bug. Scale to the magnitude and
        # give the share, which is the number that actually matters to the loss.
        self._histogram_status.setText(
            f"{source['samples']:,} samples, {_si(total)} labelled pixels, "
            f"{_si(ignored)} ignored ({100.0 * ignored / (total + ignored):.2f}%)"
        )

    # --- rendering ---

    def render_report(self, report: dict) -> None:
        """Render a verify-data report (section 12.4: counts table + red/green checks)."""
        self._report = report
        ok = bool(report.get("ok"))
        blocking = report.get("blocking_failures", [])
        warnings = report.get("warnings", [])

        self._summary_rows["status"].set_state("ok" if ok else "err")
        self._summary_rows["status"].set_value("ok" if ok else f"{len(blocking)} blocking")

        seg = report.get("seg", {})
        self._summary_rows["images"].set_value(
            f"{seg.get('total_images', 0):,} / ~{seg.get('expected_total_images', 0):,}"
        )
        self._summary_rows["sequences"].set_value(str(seg.get("total_sequences", 0)))

        mm = report.get("multimodal", {})
        if mm.get("present"):
            self._summary_rows["multimodal"].set_value(
                f"{mm.get('file_count', 0):,} files  {mm.get('total_bytes', 0) / 1e9:.1f} GB"
            )
        else:
            self._summary_rows["multimodal"].set_value("absent")

        pre = report.get("pretrained", {})
        self._summary_rows["backbone"].set_state("ok" if pre.get("present") else "warn")
        self._summary_rows["backbone"].set_value(
            f"{pre.get('bytes', 0) / 1e6:.0f} MB" if pre.get("present") else "absent"
        )

        disk = report.get("disk", {})
        if "free_bytes" in disk:
            self._summary_rows["disk free"].set_value(f"{disk['free_bytes'] / 1e9:.0f} GB")

        headline = (
            f"{seg.get('total_images', 0):,} images across "
            f"{seg.get('total_sequences', 0)} sequences"
        )
        if warnings:
            headline += f"  .  {len(warnings)} warning(s)"
        if blocking:
            headline += f"  .  {len(blocking)} blocking failure(s)"
        self._headline.setText(headline)
        self._headline.setStyleSheet(
            f"color: {tokens.OK if ok else tokens.ERR}; background: transparent;"
        )

        self._render_counts(seg.get("splits", {}))
        self._render_checks(report.get("checks", []))

    def _render_counts(self, splits: dict) -> None:
        self._counts.setRowCount(len(splits))
        for row, (name, data) in enumerate(splits.items()):
            unpaired = data.get("n_images_without_polygons", 0) + data.get(
                "n_polygons_without_images", 0
            )
            formats = ", ".join(
                f"{k.lstrip('.')}:{v:,}" for k, v in data.get("image_suffixes", {}).items()
            )
            values = [
                name,
                f"{data.get('images', 0):,}",
                f"{data.get('polygons', 0):,}",
                str(data.get("sequences", 0)),
                str(unpaired),
                formats,
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if col:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                if col == 4 and unpaired:
                    item.setForeground(_color(tokens.ERR))
                self._counts.setItem(row, col, item)

    def _render_checks(self, checks: list[dict]) -> None:
        self._checks.setRowCount(len(checks))
        for row, check in enumerate(checks):
            passed = check.get("ok")
            severity = check.get("severity", "error")
            if passed:
                glyph, color = "●", tokens.OK
            elif severity == "warn":
                glyph, color = "●", tokens.WARN
            else:
                glyph, color = "●", tokens.ERR

            dot = QTableWidgetItem(glyph)
            dot.setForeground(_color(color))
            dot.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._checks.setItem(row, 0, dot)
            self._checks.setItem(row, 1, QTableWidgetItem(check.get("name", "")))
            self._checks.setItem(row, 2, QTableWidgetItem(check.get("detail", "")))


def _color(hex_code: str):
    from PyQt6.QtGui import QColor

    return QColor(hex_code)


def _si(count: int) -> str:
    """Render a pixel count at its own magnitude.

    A fixed scale misreports the values this panel actually shows: labelled pixels are
    billions while ignored pixels are millions, and forcing both to G renders a real count as
    "0.0 G".
    """
    for limit, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "K")):
        if count >= limit:
            return f"{count / limit:.2f} {suffix}"
    return str(count)
