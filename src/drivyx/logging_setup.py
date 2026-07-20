"""Logging configuration (CLAUDE.md section 7).

Per-module loggers throughout; the root handler is installed once by the CLI. Training jobs
additionally attach a file handler into their run directory so a run's log is self-contained
(section 6.3: "Nothing about a run lives anywhere else").
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(filename)s:%(lineno)d: %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_configured = False


def configure_logging(*, verbose: bool = False, quiet: bool = False) -> None:
    """Install the console handler on the root logger. Idempotent.

    Logs go to stderr so that commands emitting JSON to stdout (verify-data, mm-inventory)
    stay machine-parseable when piped.
    """
    global _configured
    if _configured:
        return

    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    _configured = True


def attach_run_log(run_dir: Path) -> logging.Handler:
    """Tee logging into <run_dir>/run.log for the lifetime of a job.

    Returns the handler so the caller can detach it; the file handler always records DEBUG
    regardless of console verbosity, because a long training run's log is the only forensic
    record when something fails at hour six.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(handler)
    return handler


def detach_run_log(handler: logging.Handler) -> None:
    """Remove and close a handler installed by attach_run_log."""
    logging.getLogger().removeHandler(handler)
    handler.close()
