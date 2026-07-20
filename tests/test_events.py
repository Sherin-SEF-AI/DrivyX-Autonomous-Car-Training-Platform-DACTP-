"""Events writer/reader round trip (CLAUDE.md sections 6.4, 13).

Section 13 requires this test by name. The events file is the only data channel between
engine and GUI, so a break here is invisible until a training run's curves stay empty.
"""

from __future__ import annotations

import json
from pathlib import Path

from drivyx.jobs.events import (
    EVENTS_FILENAME,
    STATUS_DONE,
    STATUS_RUNNING,
    TYPE_HEARTBEAT,
    TYPE_SCALAR,
    EventTailer,
    EventWriter,
    parse_line,
    read_events,
)


def test_round_trip_every_event_type(tmp_path: Path) -> None:
    """Write one of each section 6.4 event type and read them all back."""
    with EventWriter(tmp_path) as w:
        w.scalar("train/loss", 0.412, step=1200, epoch=3)
        w.epoch(3, secs=214.8, eta_min=771.2)
        w.status(STATUS_RUNNING, "starting")
        w.image("val/overlay", "eval/ep3_0.jpg", epoch=3)
        w.heartbeat(force=True)

    events = read_events(tmp_path / EVENTS_FILENAME)
    assert [e.type for e in events] == ["scalar", "epoch", "status", "image", "heartbeat"]

    scalar = events[0]
    assert scalar.name == "train/loss"
    assert scalar.value == 0.412
    assert scalar.step == 1200
    assert scalar.epoch == 3

    assert events[1].raw["secs"] == 214.8
    assert events[1].raw["eta_min"] == 771.2
    assert events[2].value == STATUS_RUNNING
    assert events[3].raw["path"] == "eval/ep3_0.jpg"


def test_every_event_carries_a_timestamp(tmp_path: Path) -> None:
    with EventWriter(tmp_path) as w:
        w.status(STATUS_RUNNING)
        w.status(STATUS_DONE)

    events = read_events(tmp_path / EVENTS_FILENAME)
    assert all(isinstance(e.ts, float) for e in events)
    assert events[1].ts >= events[0].ts


def test_lines_are_flushed_per_write(tmp_path: Path) -> None:
    """Section 6.4: flushed per write, so a crashed run leaves a complete record."""
    w = EventWriter(tmp_path)
    w.scalar("a", 1.0, step=1)
    # Deliberately not closed: the line must already be on disk.
    assert read_events(tmp_path / EVENTS_FILENAME)[0].name == "a"
    w.close()


def test_writer_appends_across_sessions(tmp_path: Path) -> None:
    """A resumed run must not truncate the events of the run it continues."""
    with EventWriter(tmp_path) as w:
        w.scalar("a", 1.0, step=1)
    with EventWriter(tmp_path) as w:
        w.scalar("b", 2.0, step=2)

    assert [e.name for e in read_events(tmp_path / EVENTS_FILENAME)] == ["a", "b"]


def test_one_json_object_per_line(tmp_path: Path) -> None:
    with EventWriter(tmp_path) as w:
        w.scalar("train/loss", 1.5, step=1)
        w.heartbeat(force=True)

    lines = (tmp_path / EVENTS_FILENAME).read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        assert isinstance(json.loads(line), dict)


def test_heartbeat_is_rate_limited(tmp_path: Path) -> None:
    """Section 6.4: every 15 s. A training loop calls this per step without thinking."""
    with EventWriter(tmp_path) as w:
        assert w.heartbeat(force=True) is True
        assert w.heartbeat() is False
        assert w.heartbeat() is False

    beats = [e for e in read_events(tmp_path / EVENTS_FILENAME) if e.type == TYPE_HEARTBEAT]
    assert len(beats) == 1


# --- reader robustness ------------------------------------------------------------------


def test_parse_line_rejects_malformed() -> None:
    assert parse_line("") is None
    assert parse_line("   ") is None
    assert parse_line("not json") is None
    assert parse_line('{"no": "type"}') is None
    assert parse_line("[1,2,3]") is None
    assert parse_line('{"type": 42}') is None


def test_parse_line_keeps_unknown_fields() -> None:
    """Section 6.4: schema changes are additive only, so unknown keys must survive."""
    event = parse_line('{"type":"scalar","ts":1.0,"name":"x","value":2,"future_field":"keep"}')
    assert event is not None
    assert event.raw["future_field"] == "keep"


def test_unknown_event_type_is_preserved() -> None:
    """An older GUI must not choke on a newer engine's event type."""
    event = parse_line('{"type":"something_new","ts":1.0}')
    assert event is not None
    assert event.type == "something_new"


def test_read_events_of_missing_file(tmp_path: Path) -> None:
    assert read_events(tmp_path / "absent.jsonl") == []


def test_read_events_skips_corrupt_lines(tmp_path: Path) -> None:
    path = tmp_path / EVENTS_FILENAME
    path.write_text(
        '{"type":"scalar","ts":1.0,"name":"a","value":1}\n'
        "garbage not json\n"
        '{"type":"scalar","ts":2.0,"name":"b","value":2}\n'
    )
    assert [e.name for e in read_events(path)] == ["a", "b"]


# --- tailer -----------------------------------------------------------------------------


def test_tailer_returns_only_new_events(tmp_path: Path) -> None:
    w = EventWriter(tmp_path)
    tailer = EventTailer(tmp_path / EVENTS_FILENAME)

    w.scalar("a", 1.0, step=1)
    first = tailer.poll()
    assert [e.name for e in first] == ["a"]

    # Nothing appended: a second poll must return nothing, not replay history.
    assert tailer.poll() == []

    w.scalar("b", 2.0, step=2)
    assert [e.name for e in tailer.poll()] == ["b"]
    w.close()


def test_tailer_on_missing_file(tmp_path: Path) -> None:
    assert EventTailer(tmp_path / "absent.jsonl").poll() == []


def test_tailer_handles_partial_trailing_line(tmp_path: Path) -> None:
    """The GUI can observe a half-written line; it must complete, not be dropped."""
    path = tmp_path / EVENTS_FILENAME
    path.write_text('{"type":"scalar","ts":1.0,"name":"a","value":1}\n{"type":"scal')
    tailer = EventTailer(path)

    assert [e.name for e in tailer.poll()] == ["a"]

    with path.open("a") as fh:
        fh.write('ar","ts":2.0,"name":"b","value":2}\n')

    assert [e.name for e in tailer.poll()] == ["b"]


def test_tailer_restarts_when_file_shrinks(tmp_path: Path) -> None:
    """A fresh run reusing the directory truncates the file; the offset must reset."""
    path = tmp_path / EVENTS_FILENAME
    path.write_text('{"type":"scalar","ts":9.0,"name":"old","value":1}\n' * 5)
    tailer = EventTailer(path)
    assert len(tailer.poll()) == 5

    path.write_text('{"type":"scalar","ts":1.0,"name":"new","value":1}\n')
    assert [e.name for e in tailer.poll()] == ["new"]


def test_tailer_reads_scalars_by_name(tmp_path: Path) -> None:
    """Section 6.4: the GUI plots scalars by name."""
    with EventWriter(tmp_path) as w:
        w.scalar("train/loss", 1.0, step=1)
        w.scalar("val/mIoU", 0.5, step=1)
        w.scalar("train/loss", 0.9, step=2)

    events = read_events(tmp_path / EVENTS_FILENAME)
    losses = [e.value for e in events if e.type == TYPE_SCALAR and e.name == "train/loss"]
    assert losses == [1.0, 0.9]
