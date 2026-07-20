"""GUI structural contracts (CLAUDE.md sections 2, 6.2, 12).

These assert properties that are cheap to check and expensive to discover at runtime. They
construct no QApplication, so they stay CPU-only per section 13.
"""

from __future__ import annotations

import inspect

import pytest
from PyQt6.QtCore import QObject, QThread

from drivyx.gui import app as app_module
from drivyx.gui import monitor, process

#: Names that QObject and its subclasses define as methods. A pyqtSignal with one of these
#: names shadows the real method, and Qt then calls a signal where it expects a function.
#: `event` is the dangerous one: Qt calls QObject.event() to dispatch every event to the
#: object, so shadowing it breaks the object at runtime with
#: "TypeError: native Qt signal is not callable".
QOBJECT_RESERVED = frozenset(
    {
        "event",
        "eventFilter",
        "children",
        "parent",
        "deleteLater",
        "blockSignals",
        "connect",
        "disconnect",
        "emit",
        "timerEvent",
        "childEvent",
        "customEvent",
        "installEventFilter",
        "setParent",
        "startTimer",
        "killTimer",
        "thread",
        "moveToThread",
        "property",
        "setProperty",
        "objectName",
        "setObjectName",
        "isWidgetType",
        "sender",
        "findChild",
        "findChildren",
        "dumpObjectInfo",
        "inherits",
        "metaObject",
        "signalsBlocked",
        "tr",
    }
)

#: Every QObject subclass the GUI defines.
GUI_QOBJECTS = [
    process.JobRunner,
    process.JobQueue,
    monitor.MonitorThread,
]


def _signal_names(cls: type) -> set[str]:
    """Names bound to a pyqtSignal on the class itself."""
    from PyQt6.QtCore import pyqtSignal

    names = set()
    for name, value in vars(cls).items():
        # An unbound pyqtSignal is not an instance of pyqtSignal at class level in all
        # PyQt versions, so match on the type name too.
        if isinstance(value, pyqtSignal) or type(value).__name__ == "pyqtSignal":
            names.add(name)
    return names


@pytest.mark.parametrize("cls", GUI_QOBJECTS, ids=lambda c: c.__name__)
def test_no_signal_shadows_a_qobject_method(cls: type) -> None:
    """A signal named `event` breaks Qt's event dispatch for that object.

    This regression test exists because JobRunner and JobQueue both declared
    `event = pyqtSignal(object)`, which shadowed QObject.event() and produced
    "TypeError: native Qt signal is not callable" on every event delivered to them.
    """
    clashes = _signal_names(cls) & QOBJECT_RESERVED
    assert not clashes, (
        f"{cls.__name__} declares signal(s) {sorted(clashes)} that shadow QObject methods. "
        "Rename them (e.g. `event` -> `job_event`)."
    )


@pytest.mark.parametrize("cls", GUI_QOBJECTS, ids=lambda c: c.__name__)
def test_qobject_event_stays_callable(cls: type) -> None:
    """The concrete check: Qt must still be able to dispatch events to the class."""
    assert callable(getattr(cls, "event", None)), (
        f"{cls.__name__}.event is not callable; Qt cannot dispatch events to it."
    )


def test_job_queue_exposes_job_event() -> None:
    assert "job_event" in _signal_names(process.JobQueue)
    assert "job_event" in _signal_names(process.JobRunner)


# --- section 2: the GUI holds no engine logic -------------------------------------------


def test_gui_does_not_import_torch() -> None:
    """Section 2 and the M1 3 s launch gate: importing torch here would break both."""
    import sys

    for mod in list(sys.modules):
        if mod.startswith("drivyx.gui"):
            del sys.modules[mod]
    had_torch = "torch" in sys.modules

    import drivyx.gui.app  # noqa: F401

    if not had_torch:
        assert "torch" not in sys.modules, (
            "importing the GUI pulled in torch; section 2 forbids engine logic in the GUI "
            "and the M1 gate requires a sub-3s launch"
        )


def test_gui_package_has_no_torch_references() -> None:
    """No module under gui/ may import torch, even lazily.

    env_report imports torch, but the SYSTEM workspace calls it on a worker thread, which is
    a subprocess-free exception the spec allows for a diagnostic read.
    """
    import pathlib

    gui_dir = pathlib.Path(process.__file__).parent
    offenders = []
    for path in gui_dir.rglob("*.py"):
        text = path.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import torch", "from torch")):
                offenders.append(f"{path.name}: {stripped}")
    assert not offenders, f"GUI modules must not import torch: {offenders}"


# --- section 6.2: cancellation semantics -------------------------------------------------


def test_sigint_grace_period_matches_spec() -> None:
    """Section 6.2: "SIGKILL only after a 30 s grace period"."""
    assert process.SIGINT_GRACE_MS == 30_000


def test_interrupted_exit_code_matches_spec() -> None:
    """Section 6.2: the trainer exits 130 on SIGINT."""
    assert process.EXIT_INTERRUPTED == 130


def _code_without_docstring(func: object) -> str:
    """Source of a function with its docstring removed.

    The docstrings here deliberately name the APIs they avoid, so a naive substring search
    over raw source matches the prose explaining the rule rather than a violation of it.
    """
    source = inspect.getsource(func)  # type: ignore[arg-type]
    doc = inspect.getdoc(func)
    if doc:
        for line in doc.splitlines():
            source = source.replace(line, "")
    return source


def test_cancel_sends_sigint_not_sigterm() -> None:
    """QProcess.terminate() sends SIGTERM, which the trainer does not trap.

    Only SIGINT triggers its graceful checkpoint (section 6.2), so cancel() must signal the
    pid directly rather than calling terminate().
    """
    code = _code_without_docstring(process.JobRunner.cancel)

    assert "signal.SIGINT" in code
    assert "terminate()" not in code, "cancel() must send SIGINT, not SIGTERM"


def test_monitor_is_a_qthread() -> None:
    """Section 12.5 and rule 26: tegrastats must not be read on the main thread."""
    assert issubclass(monitor.MonitorThread, QThread)


def test_job_queue_is_a_qobject() -> None:
    assert issubclass(process.JobQueue, QObject)


# --- section 6.2: one heavy job at a time ------------------------------------------------


def test_queue_runs_one_job_at_a_time() -> None:
    """Section 6.2: "One GPU, therefore one heavy job at a time"."""
    source = inspect.getsource(process.JobQueue._pump)
    assert "if self._runner is not None" in source, (
        "the queue must refuse to start a second job while one is running"
    )


def test_resolve_drivyx_prefers_local_venv() -> None:
    """The GUI and engine must share one venv, or the GUI would run a different install."""
    argv = process.resolve_drivyx()
    assert argv
    assert argv[0].endswith("drivyx") or argv[:2] == [__import__("sys").executable, "-m"]


def test_main_window_closes_threads_in_order() -> None:
    """closeEvent must join every thread before the widget tree is destroyed.

    Omitting system_ws.shutdown() core-dumped the process when the app was closed while the
    environment worker was still importing torch.
    """
    source = inspect.getsource(app_module.MainWindow.closeEvent)
    for call in ("_monitor.stop()", "system_ws.shutdown()", "queue.shutdown()"):
        assert call in source, f"closeEvent must call {call}"
