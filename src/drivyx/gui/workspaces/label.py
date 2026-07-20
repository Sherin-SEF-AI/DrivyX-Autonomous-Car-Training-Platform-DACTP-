"""LABEL workspace (CLAUDE.md section 12.4).

"mm-inventory trigger, manifest FieldMapTable (confirm/override column mappings, unconfirmed
rows amber), mm-label trigger, QC gallery (track plots, waypoint overlays), dataset stats".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from drivyx.gui.process import Job
from drivyx.gui.theme import tokens
from drivyx.gui.widgets.fieldmaptable import FieldMapTable
from drivyx.gui.widgets.jobcard import JobCard
from drivyx.gui.widgets.panel import Panel
from drivyx.gui.widgets.statrow import StatRow
from drivyx.gui.workspaces.base import Workspace

logger = logging.getLogger(__name__)


class LabelWorkspace(Workspace):
    """Multimodal discovery, confirmation, and waypoint QC."""

    title = "LABEL"

    OWNED_COMMANDS = ("mm-inventory", "mm-label")

    def build(self) -> None:
        self._cards: dict[int, JobCard] = {}
        self._manifest: dict | None = None

        # --- sidebar ---

        discover = Panel("Discover")
        self._inventory_btn = QPushButton("Run mm-inventory")
        self._inventory_btn.setProperty("primary", "true")
        self._inventory_btn.clicked.connect(lambda: self._submit(["mm-inventory"], "mm-inventory"))
        discover.add_widget(self._inventory_btn)
        self._discover_rows = {
            "routes": StatRow("routes", "-", label_width=88),
            "unconfirmed": StatRow("unconfirmed", "-", state="queued", label_width=88),
        }
        for row in self._discover_rows.values():
            discover.add_widget(row)
        self.add_panel(discover)

        build = Panel("Build")
        self._label_btn = QPushButton("Run mm-label")
        self._label_btn.clicked.connect(lambda: self._submit(["mm-label"], "mm-label"))
        build.add_widget(self._label_btn)
        self._build_rows = {
            "frames": StatRow("frames", "-", label_width=88),
            "train/val": StatRow("train/val", "-", label_width=88),
            "retained": StatRow("retained", "-", label_width=88),
            "OBD match": StatRow("OBD match", "-", label_width=88),
        }
        for row in self._build_rows.values():
            build.add_widget(row)
        self.add_panel(build)

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

        self._field_map = FieldMapTable()
        self._field_map.changed.connect(self._on_field_changed)
        self._field_map.confirm_all_requested.connect(self._on_confirm_all)
        self._views.addTab(self._field_map, "Field map")

        self._qc_host = QScrollArea()
        self._qc_host.setWidgetResizable(True)
        self._qc_host.setFrameShape(QScrollArea.Shape.NoFrame)
        self._qc_inner = QWidget()
        self._qc_layout = QVBoxLayout(self._qc_inner)
        self._qc_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._qc_empty = QLabel("Run mm-label to produce the QC gallery.")
        self._qc_empty.setProperty("dim", "true")
        self._qc_layout.addWidget(self._qc_empty)
        self._qc_host.setWidget(self._qc_inner)
        self._views.addTab(self._qc_host, "QC gallery")

        self.add_main(self._views)
        self.refresh_artifacts()

    # --- job wiring ---

    def _submit(self, args: list[str], title: str) -> None:
        self._set_buttons_enabled(False)
        self.queue.submit(args, title)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self._inventory_btn.setEnabled(enabled)
        self._label_btn.setEnabled(enabled)

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

    # --- field map edits ---

    def _on_field_changed(
        self, route: str, field: str, column: object, value: object, confirmed: bool
    ) -> None:
        """Persist one FieldMapTable edit (section 8: overrides are saved back)."""
        if self._manifest is None:
            return
        from drivyx.data.mm_inventory import set_confirmation

        try:
            set_confirmation(
                self._manifest,
                route,
                field,
                column=column if isinstance(column, str) else None,
                value=float(value) if isinstance(value, (int, float)) else None,
                confirmed=confirmed,
            )
            _write_manifest(self._manifest)
        except (KeyError, ValueError) as exc:
            logger.warning("cannot confirm %s.%s: %s", route, field, exc)
            return
        self.refresh_artifacts()

    def _on_confirm_all(self) -> None:
        """Confirm every proposal at once.

        This is the same act as clicking each row, kept as one button because 24 rows over 3
        routes is the real shape of this dataset and clicking each is friction without
        safety: the evidence for every row is on screen either way.
        """
        if self._manifest is None:
            return
        from drivyx.data.mm_inventory import confirm_all
        from drivyx.paths import get_paths

        changed = confirm_all(get_paths(), self._manifest)
        logger.info("confirmed %d manifest fields from the FieldMapTable", len(changed))
        self.refresh_artifacts()

    # --- artifacts ---

    def refresh_artifacts(self) -> None:
        """Re-read the manifest and the mm-label summary from disk."""
        from drivyx.paths import get_paths

        try:
            paths = get_paths()
        except (OSError, ValueError):
            return

        try:
            from drivyx.data.mm_inventory import read_manifest, unconfirmed_fields

            manifest = read_manifest(paths)
            self._manifest = manifest
            self._field_map.load(manifest)

            pending = unconfirmed_fields(manifest)
            self._discover_rows["routes"].set_value(
                ", ".join(sorted(manifest.get("routes", {}))) or "none"
            )
            self._discover_rows["unconfirmed"].set_state("warn" if pending else "ok")
            self._discover_rows["unconfirmed"].set_value(
                f"{len(pending)} field(s)" if pending else "none: mm-label can run"
            )
            self._label_btn.setEnabled(not pending)
            self._label_btn.setToolTip(
                "Confirm every field map row first (section 8)." if pending else ""
            )
        except (FileNotFoundError, ValueError, KeyError):
            self._discover_rows["routes"].set_value("run mm-inventory")
            self._label_btn.setEnabled(False)

        summary_path = paths.waypoints / "summary.json"
        if summary_path.is_file():
            try:
                self._render_summary(json.loads(summary_path.read_text()))
            except (OSError, ValueError, KeyError) as exc:
                logger.warning("cannot read waypoint summary: %s", exc)

    def _render_summary(self, summary: dict) -> None:
        routes = summary.get("routes", {})
        if not routes:
            return
        total = summary.get("total_frames", 0)
        train = sum(r["train"] for r in routes.values())
        val = sum(r["val"] for r in routes.values())
        retained = [r["retained_pct"] for r in routes.values()]
        matched = [r["obd_matched_pct"] for r in routes.values()]

        self._build_rows["frames"].set_value(f"{total:,} across {len(routes)} routes")
        self._build_rows["train/val"].set_value(f"{train:,} / {val:,}")
        self._build_rows["retained"].set_value(f"{sum(retained) / len(retained):.1f}% of GPS fixes")
        self._build_rows["OBD match"].set_value(f"{sum(matched) / len(matched):.1f}% of frames")

        self._render_gallery(routes)

    def _render_gallery(self, routes: dict) -> None:
        """Section 12.4: the QC gallery (track plots, waypoint overlays)."""
        while self._qc_layout.count():
            item = self._qc_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for route, data in sorted(routes.items()):
            heading = QLabel(
                f"{route}: {data['frames']:,} frames, {data['segments']} segments, "
                f"{data['retained_pct']}% retained, OBD {data['obd_speed_units']}"
            )
            heading.setStyleSheet(f"color: {tokens.TEXT}; font-weight: bold;")
            self._qc_layout.addWidget(heading)

            reasons = QLabel(
                "dropped: " + ", ".join(f"{k}={v:,}" for k, v in data["drop_reasons"].items())
            )
            reasons.setProperty("dim", "true")
            self._qc_layout.addWidget(reasons)

            for name, path in sorted(data.get("qc", {}).items()):
                self._qc_layout.addWidget(_image_card(name, Path(path)))


def _image_card(title: str, path: Path) -> QWidget:
    """One QC image with its caption, scaled to a readable width."""
    card = QWidget()
    layout = QVBoxLayout(card)
    layout.setContentsMargins(0, 4, 0, 12)

    caption = QLabel(f"{title}  ({path.name})")
    caption.setProperty("dim", "true")
    layout.addWidget(caption)

    label = QLabel()
    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        label.setText(f"cannot load {path}")
        label.setProperty("dim", "true")
    else:
        label.setPixmap(
            pixmap.scaledToWidth(900, Qt.TransformationMode.SmoothTransformation)
            if pixmap.width() > 900
            else pixmap
        )
    layout.addWidget(label)
    return card


def _write_manifest(manifest: dict) -> None:
    """Persist the manifest after an edit (section 8: "overrides are saved back")."""
    import json as _json

    from drivyx.paths import get_paths

    paths = get_paths()
    paths.mm_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.mm_manifest.write_text(_json.dumps(manifest, indent=2) + "\n")
