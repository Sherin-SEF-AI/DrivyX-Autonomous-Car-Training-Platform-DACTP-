"""Run directory contract and signal handling (CLAUDE.md sections 6.2, 6.3).

Section 6.3: `runs/<YYYYmmdd-HHMMSS>_<seg|ctrl>_<tag>/` contains config.yaml (frozen
snapshot), env.txt (git SHA, pip freeze, JetPack, nvpmodel state), events.jsonl, ckpt/last.pt,
ckpt/best.pt, eval/ artifacts. "Nothing about a run lives anywhere else."

That last sentence is the whole point: a run must be interpretable months later from its own
directory, without the repository it came from being at the same commit.
"""

from __future__ import annotations

import logging
import signal
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from drivyx.jobs.events import EventWriter
from drivyx.logging_setup import attach_run_log, detach_run_log

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "config.yaml"
ENV_FILENAME = "env.txt"
CKPT_DIR = "ckpt"
LAST_CKPT = "last.pt"
BEST_CKPT = "best.pt"
EVAL_DIR = "eval"


@dataclass(frozen=True)
class RunDir:
    """One training run's directory."""

    path: Path
    kind: str
    tag: str

    @property
    def config(self) -> Path:
        return self.path / CONFIG_FILENAME

    @property
    def env(self) -> Path:
        return self.path / ENV_FILENAME

    @property
    def ckpt_dir(self) -> Path:
        return self.path / CKPT_DIR

    @property
    def last_ckpt(self) -> Path:
        return self.ckpt_dir / LAST_CKPT

    @property
    def best_ckpt(self) -> Path:
        return self.ckpt_dir / BEST_CKPT

    @property
    def eval_dir(self) -> Path:
        return self.path / EVAL_DIR

    @property
    def name(self) -> str:
        return self.path.name


def create_run(runs_root: Path, kind: str, tag: str = "default") -> RunDir:
    """Make a new run directory named per section 6.3.

    The timestamp is local, second-resolution, and sorts lexicographically, which is what makes
    `ls runs/` a chronological list without any index file.
    """
    if kind not in ("seg", "ctrl"):
        raise ValueError(f"run kind must be 'seg' or 'ctrl', got {kind!r}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_tag = "".join(c if c.isalnum() or c in "-_" else "-" for c in tag)
    path = runs_root / f"{stamp}_{kind}_{safe_tag}"
    path.mkdir(parents=True, exist_ok=False)
    (path / CKPT_DIR).mkdir()
    (path / EVAL_DIR).mkdir()
    logger.info("run directory: %s", path)
    return RunDir(path=path, kind=kind, tag=safe_tag)


def open_run(path: Path) -> RunDir:
    """Reopen an existing run directory, for --resume."""
    if not path.is_dir():
        raise FileNotFoundError(f"run directory not found: {path}")
    parts = path.name.split("_", 2)
    kind = parts[1] if len(parts) > 1 else "seg"
    tag = parts[2] if len(parts) > 2 else "default"
    (path / CKPT_DIR).mkdir(exist_ok=True)
    (path / EVAL_DIR).mkdir(exist_ok=True)
    return RunDir(path=path, kind=kind, tag=tag)


def resolve_run(runs_root: Path, reference: str) -> RunDir:
    """Resolve a run by name, by path, or by the alias 'latest'.

    Accepting a bare name matters for usability: the CLI's --run takes what `ls runs/` prints.
    """
    if reference == "latest":
        candidates = sorted(p for p in runs_root.iterdir() if p.is_dir())
        if not candidates:
            raise FileNotFoundError(f"no runs under {runs_root}")
        return open_run(candidates[-1])

    direct = Path(reference)
    if direct.is_dir():
        return open_run(direct)

    nested = runs_root / reference
    if nested.is_dir():
        return open_run(nested)

    available = (
        sorted(p.name for p in runs_root.iterdir() if p.is_dir()) if runs_root.is_dir() else []
    )
    raise FileNotFoundError(
        f"run {reference!r} not found under {runs_root}. Available: {available or 'none'}"
    )


def freeze_config(run: RunDir, config: dict[str, Any]) -> None:
    """Write the config snapshot (section 6.3).

    Snapshotted rather than referenced: the file under configs/ will be edited, and a run has
    to remain readable against the values it actually used.
    """
    run.config.write_text(yaml.safe_dump(config, sort_keys=False, default_flow_style=False))


def write_env(run: RunDir, extra: dict[str, Any] | None = None) -> None:
    """Write env.txt (section 6.3: git SHA, pip freeze, JetPack, nvpmodel state).

    Includes the wheel provenance and the sm_87 situation from D015, because "which torch was
    this trained with" is exactly the question a surprising result raises six months later.
    """
    from drivyx.env_report import full_report, pip_freeze

    report = full_report()
    lines = [
        "# DRIVYX run environment (CLAUDE.md section 6.3)",
        f"run            : {run.name}",
        f"created        : {datetime.now().isoformat(timespec='seconds')}",
        f"drivyx         : {report['drivyx_version']}",
        f"git SHA        : {report['git_sha']}",
        f"python         : {report['python']['version']} ({report['python']['executable']})",
        f"venv           : {report['python']['in_venv']}",
        f"platform       : {report['platform']['system']} {report['platform']['release']} "
        f"{report['platform']['machine']}",
        f"L4T            : {report['l4t'].get('release', 'n/a')}",
        f"JetPack        : {report.get('jetpack') or 'not installed'}",
        f"nvpmodel       : {report['power'].get('mode_name', 'unknown')} "
        f"(MAXN={report['power'].get('is_maxn')})",
    ]

    torch_report = report.get("torch", {})
    if torch_report.get("installed"):
        lines += [
            f"torch          : {torch_report.get('version')}",
            f"torch CUDA     : {torch_report.get('cuda_build')}",
            f"wheel variant  : {torch_report.get('wheel_variant')}",
            f"wheel source   : {torch_report.get('source')}",
        ]
    trt = report.get("tensorrt", {})
    lines.append(f"tensorrt       : {trt.get('version', 'not installed')}")

    for key, value in (extra or {}).items():
        lines.append(f"{key:15s}: {value}")

    lines += ["", "# pip freeze", pip_freeze()]
    run.env.write_text("\n".join(lines) + "\n")
    logger.debug("wrote %s", run.env)


class GracefulInterrupt:
    """SIGINT handling per section 6.2.

    "Cancel sends SIGINT; trainer traps it, checkpoints last.pt, writes status=interrupted,
    exits 130."

    The handler only sets a flag. Checkpointing from inside a signal handler is unsafe: the
    handler can fire between any two bytecodes, including in the middle of the optimiser step
    or a tensor write, and torch.save of a half-updated model would produce a checkpoint that
    loads without error and is silently wrong. The training loop polls `requested` at a point
    where the state is known-consistent.

    A second SIGINT restores the default handler and re-raises, so an operator who has decided
    not to wait for the checkpoint is not held hostage by their own cancel.
    """

    def __init__(self) -> None:
        self.requested = False
        self._count = 0
        self._previous: Any = None

    def _handle(self, signum: int, frame: types.FrameType | None) -> None:
        self._count += 1
        if self._count == 1:
            self.requested = True
            logger.warning(
                "SIGINT received: finishing the current step, then checkpointing and exiting "
                "130. Press Ctrl-C again to abort immediately without saving."
            )
            return
        logger.warning("second SIGINT: aborting now, the checkpoint may be incomplete")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise KeyboardInterrupt

    def __enter__(self) -> GracefulInterrupt:
        self._previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)


class RunContext:
    """Everything a job needs to honour sections 6.2, 6.3, and 6.4 at once.

    Owns the events writer, the run log, and the interrupt flag, and guarantees a terminal
    status event is written no matter how the run ends. Without that guarantee the GUI's
    heartbeat monitor cannot tell a crashed run from a slow one.
    """

    def __init__(self, run: RunDir) -> None:
        self.run = run
        self.events = EventWriter(run.path)
        self._log_handler = attach_run_log(run.path)
        self._interrupt = GracefulInterrupt()
        self._status_written = False

    @property
    def interrupted(self) -> bool:
        return self._interrupt.requested

    def __enter__(self) -> RunContext:
        self._interrupt.__enter__()
        self.events.status("running")
        return self

    def status(self, value: str, detail: str = "") -> None:
        self.events.status(value, detail)
        self._status_written = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> bool:
        try:
            if not self._status_written:
                if exc_type is KeyboardInterrupt:
                    self.events.status("interrupted", "aborted by a second SIGINT")
                elif exc is not None:
                    self.events.status("failed", f"{exc_type.__name__}: {exc}")
                else:
                    self.events.status("done")
        finally:
            self.events.close()
            detach_run_log(self._log_handler)
            self._interrupt.__exit__()
        # Never suppress: the CLI maps the exception to an exit code.
        return False
