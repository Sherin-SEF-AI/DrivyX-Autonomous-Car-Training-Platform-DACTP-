"""QProcess wrapper around the CLI, plus the events.jsonl tailer (CLAUDE.md sections 5, 6.2).

Section 2 is absolute: the GUI contains zero training or data logic and every action shells
out to `drivyx`. This module is the only place the GUI starts a process, so the SIGINT
semantics of section 6.2 are implemented exactly once.

Cancellation (section 6.2): SIGINT -> the trainer checkpoints and exits 130 -> SIGKILL only
after a 30 s grace period.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, QTimer, pyqtSignal

from drivyx.jobs.events import EventTailer

logger = logging.getLogger(__name__)

#: Section 6.2: "SIGKILL only after a 30 s grace period".
SIGINT_GRACE_MS = 30_000
#: Section 6.2 exit code for a cleanly interrupted job.
EXIT_INTERRUPTED = 130
#: events.jsonl poll cadence. QFileSystemWatcher alone is unreliable for rapid appends to a
#: single file, so a modest timer backs it up (section 6.4 wants incremental reads).
TAIL_POLL_MS = 250


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


def resolve_drivyx() -> list[str]:
    """Build the argv prefix that invokes the CLI.

    Prefers the `drivyx` console script beside the running interpreter, so the GUI and the
    engine always share one venv. Falls back to `python -m drivyx.cli`, which works from a
    source checkout with no console script installed.
    """
    candidate = Path(sys.executable).parent / "drivyx"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return [str(candidate)]
    found = shutil.which("drivyx")
    if found:
        return [found]
    return [sys.executable, "-m", "drivyx.cli"]


@dataclass
class Job:
    """One queued or running CLI invocation."""

    args: list[str]
    title: str
    run_dir: Path | None = None
    state: JobState = JobState.QUEUED
    exit_code: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    stdout: list[str] = field(default_factory=list)
    job_id: int = 0

    @property
    def command_line(self) -> str:
        """What the job card shows (section 12.3)."""
        return "drivyx " + " ".join(self.args)

    @property
    def elapsed_s(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return end - self.started_at

    @property
    def stdout_text(self) -> str:
        return "".join(self.stdout)

    @property
    def is_terminal(self) -> bool:
        return self.state in (JobState.DONE, JobState.FAILED, JobState.INTERRUPTED)


class JobRunner(QObject):
    """Runs one Job as a QProcess and streams its output.

    stdout is buffered on the Job (commands like verify-data emit a JSON report there);
    stderr carries logging and is forwarded line-by-line to the LogConsole.
    """

    stderr_line = pyqtSignal(str)
    stdout_line = pyqtSignal(str)
    #: Named job_event, not event: `event` would shadow QObject.event(), the virtual Qt
    #: calls to dispatch every event to this object, breaking it at runtime.
    job_event = pyqtSignal(object)
    state_changed = pyqtSignal(object)
    finished = pyqtSignal(object)

    def __init__(self, job: Job, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.job = job
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self._proc.readyReadStandardOutput.connect(self._on_stdout)
        self._proc.readyReadStandardError.connect(self._on_stderr)
        self._proc.finished.connect(self._on_finished)
        self._proc.errorOccurred.connect(self._on_error)

        self._stderr_buffer = ""
        self._tailer: EventTailer | None = None
        self._tail_timer: QTimer | None = None
        self._kill_timer: QTimer | None = None
        self._interrupt_sent = False

    def start(self) -> None:
        argv = resolve_drivyx() + self.job.args
        self.job.started_at = time.monotonic()
        self._set_state(JobState.RUNNING)
        logger.info("starting job: %s", " ".join(argv))
        self._proc.start(argv[0], argv[1:])

        if self.job.run_dir is not None:
            from drivyx.jobs.events import EVENTS_FILENAME

            self._tailer = EventTailer(self.job.run_dir / EVENTS_FILENAME)
            self._tail_timer = QTimer(self)
            self._tail_timer.timeout.connect(self._poll_events)
            self._tail_timer.start(TAIL_POLL_MS)

    def cancel(self) -> None:
        """Section 6.2: SIGINT, then SIGKILL after a 30 s grace period.

        QProcess.terminate() sends SIGTERM, which the trainer does not trap; SIGINT is what
        triggers its graceful checkpoint, so the signal is sent directly to the pid.
        """
        if self._proc.state() != QProcess.ProcessState.Running:
            return
        pid = int(self._proc.processId())
        if pid <= 0:
            return

        logger.info("cancelling job %d: SIGINT to pid %d", self.job.job_id, pid)
        self._interrupt_sent = True
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            return

        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._force_kill)
        self._kill_timer.start(SIGINT_GRACE_MS)

    def _force_kill(self) -> None:
        if self._proc.state() == QProcess.ProcessState.Running:
            logger.warning(
                "job %d ignored SIGINT for %d s; sending SIGKILL",
                self.job.job_id,
                SIGINT_GRACE_MS // 1000,
            )
            self._proc.kill()

    def _on_stdout(self) -> None:
        chunk = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not chunk:
            return
        # Buffered whole: verify-data and mm-inventory emit one JSON document here.
        self.job.stdout.append(chunk)
        for line in chunk.splitlines():
            self.stdout_line.emit(line)

    def _on_stderr(self) -> None:
        chunk = bytes(self._proc.readAllStandardError()).decode("utf-8", errors="replace")
        self._stderr_buffer += chunk
        lines = self._stderr_buffer.split("\n")
        self._stderr_buffer = lines.pop()
        for line in lines:
            self.stderr_line.emit(line)

    def _poll_events(self) -> None:
        if self._tailer is None:
            return
        for evt in self._tailer.poll():
            self.job_event.emit(evt)

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.stderr_line.emit(
                f"Failed to start: {' '.join(resolve_drivyx() + self.job.args)}. "
                "Is the drivyx package installed in this environment?"
            )

    def _on_finished(self, exit_code: int, status: QProcess.ExitStatus) -> None:
        if self._tail_timer is not None:
            self._tail_timer.stop()
        self._poll_events()
        if self._kill_timer is not None:
            self._kill_timer.stop()
        if self._stderr_buffer:
            self.stderr_line.emit(self._stderr_buffer)
            self._stderr_buffer = ""

        self.job.finished_at = time.monotonic()
        self.job.exit_code = exit_code

        if status == QProcess.ExitStatus.CrashExit and self._interrupt_sent:
            # SIGKILLed after ignoring the grace period, or died on the signal itself.
            self._set_state(JobState.INTERRUPTED)
        elif exit_code == EXIT_INTERRUPTED or self._interrupt_sent:
            self._set_state(JobState.INTERRUPTED)
        elif exit_code == 0:
            self._set_state(JobState.DONE)
        else:
            self._set_state(JobState.FAILED)

        logger.info(
            "job %d finished: %s (exit %s)", self.job.job_id, self.job.state.value, exit_code
        )
        self.finished.emit(self.job)

    def _set_state(self, state: JobState) -> None:
        self.job.state = state
        self.state_changed.emit(self.job)


class JobQueue(QObject):
    """FIFO queue running one job at a time (section 6.2: one GPU, one heavy job).

    Serialising every job, not just GPU ones, keeps the model simple and matches the spec;
    no data command is long enough for the queue to be a bottleneck.
    """

    job_added = pyqtSignal(object)
    job_state_changed = pyqtSignal(object)
    job_finished = pyqtSignal(object)
    stderr_line = pyqtSignal(str)
    stdout_line = pyqtSignal(str)
    #: See JobRunner.job_event: `event` is a QObject virtual and must not be shadowed.
    job_event = pyqtSignal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pending: list[Job] = []
        self._runner: JobRunner | None = None
        self._next_id = 1
        self.jobs: list[Job] = []

    @property
    def active(self) -> Job | None:
        return self._runner.job if self._runner is not None else None

    def submit(self, args: list[str], title: str, run_dir: Path | None = None) -> Job:
        job = Job(args=args, title=title, run_dir=run_dir, job_id=self._next_id)
        self._next_id += 1
        self.jobs.append(job)
        self._pending.append(job)
        self.job_added.emit(job)
        self._pump()
        return job

    def cancel_active(self) -> None:
        if self._runner is not None:
            self._runner.cancel()

    def cancel(self, job: Job) -> None:
        """Cancel a running job, or drop it from the queue if it has not started."""
        if self._runner is not None and self._runner.job is job:
            self._runner.cancel()
            return
        if job in self._pending:
            self._pending.remove(job)
            job.state = JobState.INTERRUPTED
            self.job_state_changed.emit(job)
            self.job_finished.emit(job)

    def _pump(self) -> None:
        if self._runner is not None or not self._pending:
            return
        job = self._pending.pop(0)
        runner = JobRunner(job, self)
        runner.stderr_line.connect(self.stderr_line)
        runner.stdout_line.connect(self.stdout_line)
        runner.job_event.connect(self.job_event)
        runner.state_changed.connect(self.job_state_changed)
        runner.finished.connect(self._on_job_finished)
        self._runner = runner
        runner.start()

    def _on_job_finished(self, job: Job) -> None:
        self._runner = None
        self.job_finished.emit(job)
        self._pump()

    def shutdown(self) -> None:
        """Cancel the active job on app close so no orphan process survives the GUI."""
        self._pending.clear()
        if self._runner is not None:
            self._runner.cancel()
