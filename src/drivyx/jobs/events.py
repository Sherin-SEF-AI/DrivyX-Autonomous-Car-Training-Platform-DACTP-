"""Append-only JSONL event stream (CLAUDE.md section 6.4).

This is the only data channel between engine and GUI besides exit codes, so the writer and
reader live together and are tested as a round trip (section 13).

Schema changes are additive only (section 6.4): readers ignore unknown keys and unknown
event types rather than failing, so a newer engine never breaks an older GUI.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EVENTS_FILENAME = "events.jsonl"

# Event type names from section 6.4.
TYPE_SCALAR = "scalar"
TYPE_EPOCH = "epoch"
TYPE_STATUS = "status"
TYPE_IMAGE = "image"
TYPE_HEARTBEAT = "heartbeat"

# Status values from section 6.4.
STATUS_RUNNING = "running"
STATUS_INTERRUPTED = "interrupted"
STATUS_FAILED = "failed"
STATUS_DONE = "done"

#: Section 6.4: heartbeat every 15 s; the GUI marks a run stale after 60 s of silence.
HEARTBEAT_INTERVAL_S = 15.0
STALE_AFTER_S = 60.0

TERMINAL_STATUSES = frozenset({STATUS_INTERRUPTED, STATUS_FAILED, STATUS_DONE})


class EventWriter:
    """Append-only writer, flushed per write (section 6.4).

    Timestamps are seconds since the writer was constructed, matching section 6.4's example
    ("ts": 1721.3), so a run's timeline is readable without knowing when it started.

    Flushing every line is deliberate: the GUI tails this file live, and a crashed run must
    leave a complete record up to the moment it died. os.fsync is NOT called, because that
    would cost a disk round trip per scalar and the OS buffer survives process death, which
    is the failure this guards against.
    """

    def __init__(self, run_dir: Path, *, start_time: float | None = None) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / EVENTS_FILENAME
        self._start = start_time if start_time is not None else time.monotonic()
        self._fh = self.path.open("a", encoding="utf-8")
        self._last_heartbeat = 0.0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def _emit(self, payload: dict[str, Any]) -> None:
        payload["ts"] = round(self.elapsed, 3)
        self._fh.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._fh.flush()

    def scalar(self, name: str, value: float, *, step: int, epoch: int | None = None) -> None:
        """A plotted metric, e.g. train/loss. The GUI plots scalars by name (section 6.4)."""
        event: dict[str, Any] = {
            "type": TYPE_SCALAR,
            "name": name,
            "value": float(value),
            "step": step,
        }
        if epoch is not None:
            event["epoch"] = epoch
        self._emit(event)

    def epoch(self, epoch: int, secs: float, eta_min: float | None = None) -> None:
        event: dict[str, Any] = {"type": TYPE_EPOCH, "epoch": epoch, "secs": round(secs, 2)}
        if eta_min is not None:
            event["eta_min"] = round(eta_min, 2)
        self._emit(event)

    def status(self, value: str, detail: str = "") -> None:
        self._emit({"type": TYPE_STATUS, "value": value, "detail": detail})

    def image(self, name: str, path: str, *, epoch: int | None = None) -> None:
        """Path is relative to the run directory (section 6.3: nothing lives outside it)."""
        event: dict[str, Any] = {"type": TYPE_IMAGE, "name": name, "path": path}
        if epoch is not None:
            event["epoch"] = epoch
        self._emit(event)

    def heartbeat(self, *, force: bool = False) -> bool:
        """Emit a heartbeat if HEARTBEAT_INTERVAL_S has passed. Returns True if emitted.

        Rate-limited internally so a training loop can call this every step without
        thinking about it.
        """
        now = self.elapsed
        if not force and (now - self._last_heartbeat) < HEARTBEAT_INTERVAL_S:
            return False
        self._last_heartbeat = now
        self._emit({"type": TYPE_HEARTBEAT})
        return True

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()

    def __enter__(self) -> EventWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class Event:
    """One parsed event. `raw` retains every field, including additive future ones."""

    type: str
    ts: float
    raw: dict[str, Any]

    @property
    def name(self) -> str | None:
        return self.raw.get("name")

    @property
    def value(self) -> Any:
        return self.raw.get("value")

    @property
    def step(self) -> int | None:
        return self.raw.get("step")

    @property
    def epoch(self) -> int | None:
        return self.raw.get("epoch")


def parse_line(line: str) -> Event | None:
    """Parse one JSONL line into an Event, or None if it is not a usable event.

    Returns None rather than raising for a malformed line: the GUI tails a file being
    written concurrently and can legitimately observe a torn final line, which is not a
    data integrity problem and must not kill the reader.
    """
    line = line.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    event_type = payload.get("type")
    if not isinstance(event_type, str):
        return None
    ts = payload.get("ts")
    return Event(
        type=event_type,
        ts=float(ts) if isinstance(ts, (int, float)) else 0.0,
        raw=payload,
    )


def read_events(path: Path) -> list[Event]:
    """Read a complete events file. Malformed lines are skipped."""
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [e for e in (parse_line(line) for line in fh) if e is not None]


class EventTailer:
    """Incremental reader for a file being appended to (section 6.4).

    Holds a byte offset and returns only events appended since the last call, so the GUI
    re-reads nothing. A partial trailing line (the writer flushed mid-line) is left in the
    buffer and completed on the next poll rather than being parsed and dropped.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._offset = 0
        self._pending = ""

    def reset(self) -> None:
        self._offset = 0
        self._pending = ""

    def poll(self) -> list[Event]:
        """Return events appended since the last poll."""
        if not self.path.is_file():
            return []

        try:
            size = self.path.stat().st_size
        except OSError:
            return []

        # A shrunk file means it was truncated or replaced (a fresh run reusing the dir);
        # start over rather than reading from a stale offset into the middle of a line.
        if size < self._offset:
            logger.debug(
                "%s shrank from %d to %d bytes; restarting tail",
                self.path,
                self._offset,
                size,
            )
            self.reset()
        if size == self._offset:
            return []

        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self._offset)
                chunk = fh.read()
                self._offset = fh.tell()
        except OSError as exc:
            logger.debug("tail of %s failed: %s", self.path, exc)
            return []

        buffer = self._pending + chunk
        lines = buffer.split("\n")
        # The last element is whatever followed the final newline: either empty, or a
        # partially written line to be completed by a later append.
        self._pending = lines.pop()
        return [e for e in (parse_line(line) for line in lines) if e is not None]
