"""Job card (CLAUDE.md section 12.3).

"each card shows command line, elapsed, progress, Cancel (SIGINT semantics from 6.2), Open
run dir".
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from drivyx.gui.process import Job, JobState
from drivyx.gui.widgets.statrow import StatusDot


def _format_elapsed(seconds: float) -> str:
    total = int(seconds)
    if total < 3600:
        return f"{total // 60:02d}:{total % 60:02d}"
    return f"{total // 3600:d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


class JobCard(QFrame):
    """One job in the queued/running/finished list."""

    cancel_requested = pyqtSignal(object)

    def __init__(self, job: Job, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("jobCard")
        self.job = job

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        top = QHBoxLayout()
        top.setSpacing(6)
        self._dot = StatusDot(job.state.value)
        top.addWidget(self._dot)

        self._title = QLabel(job.title)
        top.addWidget(self._title)
        top.addStretch(1)

        self._elapsed = QLabel("00:00")
        self._elapsed.setProperty("mono", "true")
        top.addWidget(self._elapsed)
        layout.addLayout(top)

        self._command = QLabel(job.command_line)
        self._command.setObjectName("jobCommand")
        self._command.setWordWrap(True)
        layout.addWidget(self._command)

        # Indeterminate until a job reports progress; most data commands never do, and a
        # fake percentage would be a lie about work we cannot measure.
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(6)
        layout.addWidget(self._progress)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        actions.addStretch(1)

        self._open_dir = QPushButton("Open run dir")
        self._open_dir.clicked.connect(self._on_open_dir)
        self._open_dir.setVisible(job.run_dir is not None)
        actions.addWidget(self._open_dir)

        self._cancel = QPushButton("Cancel")
        self._cancel.clicked.connect(lambda: self.cancel_requested.emit(self.job))
        actions.addWidget(self._cancel)
        layout.addLayout(actions)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(500)

        self.refresh()

    def _tick(self) -> None:
        self._elapsed.setText(_format_elapsed(self.job.elapsed_s))

    def refresh(self) -> None:
        """Re-render from the job's current state."""
        job = self.job
        self._dot.set_state(job.state.value)
        self._elapsed.setText(_format_elapsed(job.elapsed_s))
        self._open_dir.setVisible(job.run_dir is not None)

        if job.is_terminal:
            self._timer.stop()
            self._progress.setRange(0, 1)
            self._progress.setValue(1)
            self._cancel.setEnabled(False)
            suffix = {
                JobState.DONE: "done",
                JobState.FAILED: f"failed (exit {job.exit_code})",
                JobState.INTERRUPTED: "interrupted",
            }.get(job.state, job.state.value)
            self._title.setText(f"{job.title}  .  {suffix}")
        elif job.state == JobState.RUNNING:
            self._progress.setRange(0, 0)
            self._cancel.setEnabled(True)
        else:
            self._progress.setRange(0, 1)
            self._progress.setValue(0)
            self._cancel.setEnabled(True)

    def set_progress(self, fraction: float) -> None:
        """Switch to determinate progress once a job reports it (epoch events)."""
        self._progress.setRange(0, 1000)
        self._progress.setValue(int(max(0.0, min(1.0, fraction)) * 1000))

    def _on_open_dir(self) -> None:
        if self.job.run_dir is None:
            return
        self._open_path(self.job.run_dir)

    @staticmethod
    def _open_path(path: Path) -> None:
        """Open a directory in the desktop file manager.

        xdg-open is spawned detached: QDesktopServices can block the Qt main thread while
        the handler starts, and rule 26 forbids that.
        """
        subprocess.Popen(
            ["xdg-open", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
