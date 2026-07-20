"""CLI smoke tests (CLAUDE.md section 13: every subcommand --help exits 0)."""

from __future__ import annotations

import pytest

from drivyx.cli import build_parser, main, registered_commands


def test_at_least_one_command_registered() -> None:
    assert registered_commands(), "the CLI must register at least one subcommand"


def test_top_level_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "drivyx" in capsys.readouterr().out


@pytest.mark.parametrize("command", registered_commands())
def test_subcommand_help_exits_zero(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Section 13 requires this for every subcommand the CLI offers.

    The parametrisation reads the live registry, so the test covers exactly what is
    implemented at the current milestone (docs/DECISIONS.md D012).
    """
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip()


def test_no_command_prints_help_and_fails(capsys: pytest.CaptureFixture[str]) -> None:
    """Bare `drivyx` is a usage error, not a silent success."""
    assert main([]) != 0
    assert "usage" in capsys.readouterr().out.lower()


def test_parser_builds_without_side_effects() -> None:
    """build_parser must not import torch or touch the filesystem.

    The GUI builds argument lists from this parser and must stay fast (section 12: under a
    3 s launch), so parser construction has to stay cheap.
    """
    import sys

    torch_was_loaded = "torch" in sys.modules
    build_parser()
    if not torch_was_loaded:
        assert "torch" not in sys.modules, "building the parser must not import torch"
