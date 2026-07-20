"""DRIVYX desktop application (CLAUDE.md section 12).

Entry point for `drivyx-gui`. Section 2 is absolute: this package contains zero training or
data logic. Every action runs a `drivyx` CLI subcommand through QProcess (gui/process.py)
and reads back stdout or events.jsonl.

Nothing here imports torch. Section 14's M1 gate requires launch in under 3 s, and importing
torch alone costs more than that.
"""

from __future__ import annotations

import logging
import sys
import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from drivyx import __version__
from drivyx.branding import APP_TITLE, ORG_NAME
from drivyx.gui.monitor import MonitorThread, TegraSample, format_status
from drivyx.gui.process import Job, JobQueue, JobState
from drivyx.gui.theme import tokens
from drivyx.gui.widgets.logconsole import LogConsole
from drivyx.gui.widgets.statrow import StatusDot
from drivyx.gui.workspaces.data import DataWorkspace
from drivyx.gui.workspaces.eval import EvalWorkspace
from drivyx.gui.workspaces.export import ExportWorkspace
from drivyx.gui.workspaces.label import LabelWorkspace
from drivyx.gui.workspaces.system import SystemWorkspace
from drivyx.gui.workspaces.train import TrainWorkspace
from drivyx.jobs.events import HEARTBEAT_INTERVAL_S, STALE_AFTER_S

logger = logging.getLogger(__name__)

WINDOW_MIN_SIZE = (1280, 800)
#: Section 6.4: the GUI marks a run stale after 60 s of event silence.
STALE_CHECK_MS = 5000


class MainWindow(QMainWindow):
    """Shell: workspace tabs, device badge, status bar, shared log dock (section 12.3)."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_TITLE} {__version__}")
        self.setMinimumSize(*WINDOW_MIN_SIZE)

        self.queue = JobQueue(self)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Amber banner shown whenever the device is not at MAXN (section 12.3). Created
        # hidden; the environment report decides. Section 12.1 limits motion to binary state
        # changes, so it appears and disappears outright.
        self._maxn_banner = QLabel()
        self._maxn_banner.setObjectName("maxnBanner")
        self._maxn_banner.setVisible(False)
        self._maxn_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._maxn_banner)

        root.addWidget(self._build_topbar())

        # Vertical splitter: workspace above, LogConsole docked below (section 12.3).
        self._vsplit = QSplitter(Qt.Orientation.Vertical)
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._vsplit.addWidget(self._tabs)

        self.log = LogConsole()
        self._vsplit.addWidget(self.log)
        self._vsplit.setStretchFactor(0, 1)
        self._vsplit.setStretchFactor(1, 0)
        self._vsplit.setSizes([620, 180])
        root.addWidget(self._vsplit, 1)

        self.setCentralWidget(central)

        self.data_ws = DataWorkspace(self.queue)
        self.label_ws = LabelWorkspace(self.queue)
        self.train_ws = TrainWorkspace(self.queue)
        self.eval_ws = EvalWorkspace(self.queue)
        self.export_ws = ExportWorkspace(self.queue)
        self.system_ws = SystemWorkspace(self.queue)
        for ws in (
            self.data_ws,
            self.label_ws,
            self.train_ws,
            self.eval_ws,
            self.export_ws,
            self.system_ws,
        ):
            self._tabs.addTab(ws, ws.title)

        # Land on DATA. Section 12.3 lists it first and it is the pipeline's entry point:
        # everything downstream depends on verify-data having passed. Set explicitly rather
        # than relying on Qt's default, which a workspace can perturb during construction.
        self._tabs.setCurrentIndex(0)

        self._build_statusbar()
        self._wire_queue()
        self._start_monitor()

        self.system_ws.environment_ready.connect(self.apply_environment)

        self._last_event_at = time.monotonic()
        self._stale = False
        self._stale_timer = QTimer(self)
        self._stale_timer.timeout.connect(self._check_stale)
        self._stale_timer.start(STALE_CHECK_MS)

    # --- chrome ---

    def _build_topbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("topBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(6, 4, 6, 0)
        layout.setSpacing(8)

        title = QLabel(APP_TITLE)
        font = QFont()
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        layout.addStretch(1)

        # Right-aligned device badge (section 12.3). Filled in by the environment report.
        self._device_badge = QLabel("AGX Orin 64GB . JetPack ... . ...")
        self._device_badge.setObjectName("deviceBadge")
        # Never let the stretch above squeeze the badge: its text grows when the device is
        # off MAXN ("15W (not MAXN)"), and a QLabel's default policy allows shrinking below
        # its sizeHint, which clipped the power state exactly when it mattered most.
        self._device_badge.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        layout.addWidget(self._device_badge)
        return bar

    def _build_statusbar(self) -> None:
        """Always-visible status bar (section 12.3): job + progress, then live readouts."""
        bar = self.statusBar()
        if bar is None:
            return

        self._job_dot = StatusDot("queued")
        bar.addWidget(self._job_dot)

        self._job_label = QLabel("idle")
        bar.addWidget(self._job_label)

        self._telemetry = QLabel("GPU   --  MEM    --  SOC  --  PWR  --")
        bar.addPermanentWidget(self._telemetry)

    # --- wiring ---

    def _wire_queue(self) -> None:
        self.queue.stderr_line.connect(self.log.append)
        self.queue.stdout_line.connect(self._on_stdout_line)
        self.queue.job_event.connect(self._on_event)
        self.queue.job_added.connect(self._on_job_added)
        self.queue.job_state_changed.connect(self._on_job_state)
        self.queue.job_finished.connect(self._on_job_finished)

        # Every workspace that owns CLI commands gets the job stream; each filters to the
        # commands it owns, so a job never renders a card in an unrelated workspace.
        for ws in (self.data_ws, self.label_ws, self.train_ws, self.eval_ws, self.export_ws):
            self.queue.job_added.connect(ws.on_job_added)
            self.queue.job_state_changed.connect(ws.on_job_state_changed)
            self.queue.job_finished.connect(ws.on_job_finished)

        # TRAIN plots the events.jsonl stream live (section 12.4).
        self.queue.job_event.connect(self.train_ws.on_event)

    def _on_stdout_line(self, line: str) -> None:
        # stdout carries machine-readable reports (verify-data JSON). Echoing it into the
        # console would bury the log, so only stderr is shown; the workspace parses stdout.
        logger.debug("stdout: %s", line)

    def _on_job_added(self, job: Job) -> None:
        self.log.append(f"$ {job.command_line}")

    def _on_job_state(self, job: Job) -> None:
        if job.state == JobState.RUNNING:
            # Reset the staleness clock: a fresh run has not gone silent, it has not spoken.
            self._last_event_at = time.monotonic()
            self._stale = False
            self._job_dot.set_state("running")
            self._job_label.setText(f"{job.title} ...")

    def _on_job_finished(self, job: Job) -> None:
        self._job_dot.set_state(job.state.value)
        self._job_label.setText(f"{job.title}: {job.state.value} (exit {job.exit_code})")
        self.log.append(f"[{job.state.value}] {job.title} exit={job.exit_code}")

    def _on_event(self, event: object) -> None:
        """Any event, including a heartbeat, proves the run is alive (section 6.4)."""
        self._last_event_at = time.monotonic()
        if self._stale:
            self._stale = False
            self._job_dot.set_state("running")

    def _check_stale(self) -> None:
        """Section 6.4: mark a run stale after 60 s of event silence.

        Only applies to jobs that own a run directory and therefore emit heartbeats every
        15 s. Data commands have no event stream, so silence from them means nothing and
        they are exempt.
        """
        job = self.queue.active
        if job is None or job.run_dir is None or job.state != JobState.RUNNING:
            self._stale = False
            return

        silence = time.monotonic() - self._last_event_at
        if silence > STALE_AFTER_S and not self._stale:
            self._stale = True
            self._job_dot.set_state("warn")
            self._job_label.setText(f"{job.title}: stale ({silence:.0f}s without events)")
            self.log.append(
                f"[stale] {job.title} has emitted no events for {silence:.0f}s "
                f"(heartbeat interval is {HEARTBEAT_INTERVAL_S:.0f}s)"
            )

    # --- monitor ---

    def _start_monitor(self) -> None:
        self._monitor = MonitorThread()
        self._monitor.sample.connect(self._on_sample)
        self._monitor.failed.connect(self._on_monitor_failed)
        self._monitor.start()

    def _on_sample(self, sample: TegraSample) -> None:
        self._telemetry.setText(format_status(sample))
        self.system_ws.on_sample(sample)

    def _on_monitor_failed(self, message: str) -> None:
        logger.warning("monitor: %s", message)
        self._telemetry.setText("telemetry unavailable")
        self._telemetry.setStyleSheet(f"color: {tokens.ERR};")
        self.system_ws.on_monitor_failed(message)

    def apply_environment(self, report: dict) -> None:
        """Fill the device badge and decide the MAXN banner (section 12.3)."""
        l4t = report.get("l4t", {})
        jetpack = report.get("jetpack") or (
            l4t.get("release", "unknown") if l4t.get("present") else "unknown"
        )
        power = report.get("power", {})
        mode = power.get("mode_name", "unknown")
        is_maxn = power.get("is_maxn")

        state = mode if is_maxn else f"{mode} (not MAXN)"
        self._device_badge.setText(f"AGX Orin 64GB . JetPack {jetpack} . {state}")

        if is_maxn:
            self._maxn_banner.setVisible(False)
        else:
            self._maxn_banner.setText(
                f"Device is in power mode {mode}, not MAXN. Training will be slower. "
                "Run: sudo nvpmodel -m 0 && sudo jetson_clocks"
            )
            self._maxn_banner.setVisible(True)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Tear down every thread and child process before the widget tree goes away.

        Order matters: threads that hold references into the widget tree are joined first,
        so nothing emits into a destroyed receiver.
        """
        self._stale_timer.stop()
        self._monitor.stop()
        self.system_ws.shutdown()
        self.queue.shutdown()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    """Entry point for `drivyx-gui`."""
    from drivyx.logging_setup import configure_logging

    configure_logging()

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setOrganizationName(ORG_NAME)
    app.setStyleSheet(tokens.load_stylesheet())

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
