"""Environment and provenance reporting (CLAUDE.md sections 3, 6.3, 15).

These run on CPU-only machines (section 13), so nothing here asserts device specifics. The
Jetson-shaped inputs are supplied as fixtures and the parsers are tested against them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drivyx import env_report

# --- L4T release parsing ----------------------------------------------------------------

R39_LINE = (
    "# R39 (release), REVISION: 2.0, GCID: 45755727, BOARD: generic, EABI: aarch64, "
    "DATE: Mon Jun  1 09:28:48 PM UTC 2026\n"
    "# KERNEL_VARIANT: oot\n"
)
R36_LINE = "# R36 (release), REVISION: 4.3, GCID: 12345, BOARD: generic, EABI: aarch64\n"


def test_l4t_absent_reports_not_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(env_report, "_L4T_RELEASE_FILE", tmp_path / "nope")
    assert env_report.l4t_release() == {"present": False}


@pytest.mark.parametrize(
    ("text", "major", "revision"),
    [(R39_LINE, 39, "2.0"), (R36_LINE, 36, "4.3")],
)
def test_l4t_parsed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str, major: int, revision: str
) -> None:
    release = tmp_path / "nv_tegra_release"
    release.write_text(text)
    monkeypatch.setattr(env_report, "_L4T_RELEASE_FILE", release)

    report = env_report.l4t_release()

    assert report["present"] is True
    assert report["major"] == major
    assert report["revision"] == revision
    assert report["release"] == f"R{major}.{revision}"


def test_l4t_unparseable_still_reports_raw(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An unexpected format must surface the raw line, not silently report nothing."""
    release = tmp_path / "nv_tegra_release"
    release.write_text("something entirely different\n")
    monkeypatch.setattr(env_report, "_L4T_RELEASE_FILE", release)

    report = env_report.l4t_release()

    assert report["present"] is True
    assert report["raw"] == "something entirely different"
    assert "major" not in report


# --- trtexec resolution (DECISIONS.md D004) ---------------------------------------------


def test_trtexec_prefers_spec_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Section 3's documented path wins when it exists and is executable."""
    spec_path = tmp_path / "trtexec"
    spec_path.write_text("#!/bin/sh\n")
    spec_path.chmod(0o755)
    monkeypatch.setattr(env_report, "TRTEXEC_CANDIDATES", (str(spec_path),))

    assert env_report.find_trtexec() == spec_path.resolve()


def test_trtexec_ignores_non_executable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A present but non-executable file must not be reported as usable."""
    spec_path = tmp_path / "trtexec"
    spec_path.write_text("not executable")
    spec_path.chmod(0o644)
    monkeypatch.setattr(env_report, "TRTEXEC_CANDIDATES", (str(spec_path),))
    monkeypatch.setattr(env_report.shutil, "which", lambda _: None)

    assert env_report.find_trtexec() is None


def test_trtexec_falls_back_to_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When the spec path is absent, PATH is consulted (D004: it is a symlink on JP7.2)."""
    on_path = tmp_path / "trtexec"
    on_path.write_text("#!/bin/sh\n")
    on_path.chmod(0o755)
    monkeypatch.setattr(env_report, "TRTEXEC_CANDIDATES", (str(tmp_path / "absent"),))
    monkeypatch.setattr(
        env_report.shutil, "which", lambda name: str(on_path) if name == "trtexec" else None
    )

    assert env_report.find_trtexec() == on_path.resolve()


def test_require_trtexec_names_where_it_looked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_report, "TRTEXEC_CANDIDATES", ("/nowhere/trtexec",))
    monkeypatch.setattr(env_report.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="/nowhere/trtexec"):
        env_report.require_trtexec()


# --- git provenance (section 6.3) -------------------------------------------------------


def test_git_sha_without_repo(tmp_path: Path) -> None:
    """No repo degrades to a marker; a training run must not die over provenance metadata."""
    assert env_report.git_sha(tmp_path) == "not-a-git-repo"


def test_git_sha_in_repo_is_a_marker_or_sha(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    sha = env_report.git_sha(tmp_path)
    assert sha and isinstance(sha, str)


# --- subprocess helper ------------------------------------------------------------------


def test_run_returns_none_for_missing_binary() -> None:
    assert env_report._run(["definitely-not-a-real-binary-xyz"]) is None


def test_run_returns_none_on_nonzero_exit() -> None:
    assert env_report._run(["false"]) is None


def test_run_captures_stdout() -> None:
    assert env_report._run(["echo", "hello"]) == "hello"


# --- provenance -------------------------------------------------------------------------


def test_provenance_of_absent_distribution() -> None:
    info = env_report._distribution_provenance("no-such-distribution-xyz")
    assert info["installed"] is False


def test_provenance_of_present_distribution() -> None:
    """pydantic is a hard dependency, so it is always installed alongside the tests."""
    info = env_report._distribution_provenance("pydantic")
    assert info["installed"] is True
    assert info["version"]
    assert info["source"] in {"pip wheel", "system (apt/dist-packages)"}


def test_power_state_never_raises() -> None:
    """nvpmodel is absent on a dev machine; the GUI must still get a dict."""
    state = env_report.power_state()
    assert "mode_name" in state


def test_full_report_shape() -> None:
    report = env_report.full_report()
    for key in ("drivyx_version", "git_sha", "python", "platform", "l4t", "torch", "tensorrt"):
        assert key in report, f"full_report missing {key}"
    assert report["python"]["in_venv"] in {True, False}


def test_version_summary_renders() -> None:
    rendered = env_report.version_summary().render()
    assert "DRIVYX" in rendered
    assert "wheel provenance" in rendered


@pytest.mark.device
def test_torch_report_on_device() -> None:
    """On the Orin, torch must be a CUDA build (D002: never silently CPU torch)."""
    report = env_report.torch_report()
    assert report["installed"] is True
    assert report["is_cuda_build"] is True, "torch is not a CUDA build; setup_orin.sh must abort"
    assert report["wheel_variant"].startswith("cu"), report["wheel_variant"]


@pytest.mark.device
def test_opencv_is_headless_on_device() -> None:
    """Section 3: opencv-python's bundled Qt5 plugins break PyQt6's xcb platform plugin."""
    report = env_report.opencv_report()
    assert report["headless_ok"] is True, (
        "opencv-python (non-headless) is installed; it will break the GUI's xcb plugin"
    )


@pytest.mark.device
def test_trtexec_present_on_device() -> None:
    assert env_report.find_trtexec() is not None


@pytest.mark.device
def test_l4t_present_on_device() -> None:
    report = env_report.l4t_release()
    assert report["present"] is True
    assert report["major"] >= 36, f"unexpected L4T major: {report}"
