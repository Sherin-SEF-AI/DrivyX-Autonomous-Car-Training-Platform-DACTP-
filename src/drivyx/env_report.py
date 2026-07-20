"""Environment and provenance reporting.

Serves three consumers: `drivyx --version` (section 15), the `env.txt` snapshot in every run
directory (section 6.3), and the SYSTEM workspace environment report (section 12.4), which
must show "wheel provenance" and which PyQt6 install path was taken (section 3).

Heavy imports (torch, tensorrt) are deferred into functions so that `drivyx --help` and the
GUI stay fast and remain importable without CUDA.
"""

from __future__ import annotations

import importlib.metadata as md
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drivyx import __version__

logger = logging.getLogger(__name__)

#: Section 3 says trtexec is at /usr/src/tensorrt/bin/trtexec. On JetPack 7.2 that path is a
#: symlink into /usr/bin. Probe the spec path first, then PATH. See DECISIONS.md D004.
TRTEXEC_CANDIDATES = ("/usr/src/tensorrt/bin/trtexec",)

_L4T_RELEASE_FILE = Path("/etc/nv_tegra_release")
_SUBPROCESS_TIMEOUT_S = 10


def _run(cmd: list[str], *, timeout: int = _SUBPROCESS_TIMEOUT_S) -> str | None:
    """Run a command and return stripped stdout, or None if it is unavailable or fails."""
    if shutil.which(cmd[0]) is None and not Path(cmd[0]).is_file():
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("command %s failed: %s", cmd, exc)
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def find_trtexec() -> Path | None:
    """Resolve trtexec, honouring the spec path first (DECISIONS.md D004)."""
    for candidate in TRTEXEC_CANDIDATES:
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    found = shutil.which("trtexec")
    return Path(found).resolve() if found else None


def require_trtexec() -> Path:
    """find_trtexec() for commands that cannot proceed without it."""
    path = find_trtexec()
    if path is None:
        raise RuntimeError(
            "trtexec not found. Looked at "
            f"{', '.join(TRTEXEC_CANDIDATES)} and on PATH. Install the TensorRT binaries "
            "(apt package libnvinfer-bin) or add trtexec to PATH."
        )
    return path


def git_sha(repo_root: Path | None = None) -> str:
    """Short git SHA of the working tree, or a precise reason it is unavailable.

    Section 6.3 requires a SHA per run. A tree with no commits yet still trains, so this
    degrades to a marker rather than raising (DECISIONS.md D013).
    """
    root = repo_root or Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return "not-a-git-repo"
    sha = _run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"])
    if sha is None:
        return "uncommitted"
    dirty = _run(["git", "-C", str(root), "status", "--porcelain"])
    return f"{sha}-dirty" if dirty else sha


def _distribution_provenance(dist_name: str) -> dict[str, Any]:
    """Where an installed distribution came from.

    "Wheel provenance" (section 15) is recovered from the installer metadata pip writes:
    direct_url.json records the index or URL a wheel was fetched from, and INSTALLER
    distinguishes pip from apt-managed dist-packages.
    """
    info: dict[str, Any] = {"name": dist_name, "installed": False}
    try:
        dist = md.distribution(dist_name)
    except md.PackageNotFoundError:
        return info

    info["installed"] = True
    info["version"] = dist.version
    location = dist.locate_file("")
    info["location"] = str(location)
    # apt-installed python packages land in dist-packages; pip wheels in site-packages.
    info["source"] = (
        "system (apt/dist-packages)" if "dist-packages" in str(location) else "pip wheel"
    )

    # INSTALLER and direct_url.json are optional metadata: pip writes them, apt-managed
    # dist-packages do not. Their absence is itself provenance information (it indicates a
    # system package), so a missing file enriches the report rather than failing it.
    for key, filename in (("installer", "INSTALLER"), ("direct_url", "direct_url.json")):
        try:
            text = dist.read_text(filename)
        except (OSError, md.PackageNotFoundError):
            continue
        if text:
            info[key] = text.strip()

    return info


def torch_report() -> dict[str, Any]:
    """Torch version, CUDA linkage, and wheel provenance. Never initialises CUDA."""
    report: dict[str, Any] = _distribution_provenance("torch")
    try:
        import torch
    except ImportError as exc:
        report["import_error"] = str(exc)
        return report

    report["version"] = torch.__version__
    report["cuda_build"] = torch.version.cuda
    report["cudnn_build"] = torch.backends.cudnn.version()
    # The local version segment ("+cu130") is the wheel's own statement of which CUDA index
    # it came from, and is the most reliable provenance marker after the fact.
    local = torch.__version__.partition("+")[2]
    report["wheel_variant"] = local or "none (upstream PyPI or source build)"
    report["is_cuda_build"] = torch.version.cuda is not None
    return report


def tensorrt_report() -> dict[str, Any]:
    """TensorRT python binding version and provenance."""
    report: dict[str, Any] = _distribution_provenance("tensorrt")
    try:
        import tensorrt as trt
    except ImportError as exc:
        report["import_error"] = str(exc)
        return report
    report["version"] = trt.__version__
    report["trtexec"] = str(find_trtexec() or "not found")
    return report


def pyqt_report() -> dict[str, Any]:
    """Which PyQt6 install path was taken (section 3 requires this be reported)."""
    report: dict[str, Any] = _distribution_provenance("PyQt6")
    try:
        from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
    except ImportError as exc:
        report["import_error"] = str(exc)
        return report
    report["pyqt_version"] = PYQT_VERSION_STR
    report["qt_version"] = QT_VERSION_STR
    return report


def opencv_report() -> dict[str, Any]:
    """OpenCV build, and whether it is the headless wheel section 3 demands.

    A non-headless opencv bundles Qt5 plugins that hijack the xcb platform plugin and break
    PyQt6, so the distribution name actually installed is the thing worth reporting.
    """
    report: dict[str, Any] = {}
    for name in ("opencv-python-headless", "opencv-python", "opencv-contrib-python"):
        info = _distribution_provenance(name)
        if info["installed"]:
            report[name] = info
    report["headless_ok"] = "opencv-python-headless" in report and "opencv-python" not in report
    try:
        import cv2
    except ImportError as exc:
        report["import_error"] = str(exc)
        return report
    report["version"] = cv2.__version__
    report["file"] = getattr(cv2, "__file__", None)
    return report


def l4t_release() -> dict[str, Any]:
    """Parse /etc/nv_tegra_release (section 3)."""
    if not _L4T_RELEASE_FILE.is_file():
        return {"present": False}
    text = _L4T_RELEASE_FILE.read_text(errors="replace")
    first = text.splitlines()[0] if text else ""
    match = re.search(r"R(\d+).*?REVISION:\s*([\d.]+)", first)
    report: dict[str, Any] = {"present": True, "raw": first.strip()}
    if match:
        report["major"] = int(match.group(1))
        report["revision"] = match.group(2)
        report["release"] = f"R{match.group(1)}.{match.group(2)}"
    return report


def jetpack_version() -> str | None:
    """Installed nvidia-jetpack meta-package version, if any."""
    out = _run(["dpkg-query", "--showformat=${Version}", "--show", "nvidia-jetpack"])
    return out or None


def power_state() -> dict[str, Any]:
    """nvpmodel mode and jetson_clocks state (section 3 and the section 12.3 MAXN badge).

    nvpmodel -q works without sudo for querying; when it does not, the fields report
    unknown rather than failing, so the GUI degrades to an unknown badge.
    """
    report: dict[str, Any] = {}
    out = _run(["nvpmodel", "-q"])
    if out:
        report["nvpmodel_raw"] = out
        mode_name = re.search(r"NV Power Mode:\s*(\S+)", out)
        if mode_name:
            report["mode_name"] = mode_name.group(1)
            report["is_maxn"] = mode_name.group(1).upper().startswith("MAXN")
        mode_id = re.findall(r"^\s*(\d+)\s*$", out, flags=re.MULTILINE)
        if mode_id:
            report["mode_id"] = int(mode_id[-1])
    else:
        report["mode_name"] = "unknown"
        report["is_maxn"] = None
    return report


@dataclass(frozen=True)
class VersionSummary:
    """One-line-per-field summary for `drivyx --version` (section 15)."""

    version: str
    git_sha: str
    torch: str
    tensorrt: str
    provenance: str

    def render(self) -> str:
        return "\n".join(
            [
                f"DRIVYX {self.version}",
                f"  git SHA        : {self.git_sha}",
                f"  torch          : {self.torch}",
                f"  tensorrt       : {self.tensorrt}",
                f"  wheel provenance: {self.provenance}",
            ]
        )


def version_summary() -> VersionSummary:
    """Assemble `drivyx --version` output."""
    t = torch_report()
    trt = tensorrt_report()

    if t.get("installed"):
        torch_str = f"{t.get('version')} (CUDA build {t.get('cuda_build') or 'none'})"
        provenance = f"{t.get('wheel_variant')} via {t.get('source', 'unknown')}"
    else:
        torch_str = f"not installed ({t.get('import_error', 'absent')})"
        provenance = "n/a"

    trt_str = trt.get("version") if trt.get("installed") or "version" in trt else "not installed"

    return VersionSummary(
        version=__version__,
        git_sha=git_sha(),
        torch=torch_str,
        tensorrt=str(trt_str),
        provenance=provenance,
    )


def full_report() -> dict[str, Any]:
    """Complete environment report for env.txt and the SYSTEM workspace."""
    return {
        "drivyx_version": __version__,
        "git_sha": git_sha(),
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "prefix": sys.prefix,
            "in_venv": sys.prefix != sys.base_prefix,
        },
        "platform": {
            "machine": platform.machine(),
            "release": platform.release(),
            "system": platform.system(),
        },
        "l4t": l4t_release(),
        "jetpack": jetpack_version(),
        "power": power_state(),
        "torch": torch_report(),
        "tensorrt": tensorrt_report(),
        "pyqt": pyqt_report(),
        "opencv": opencv_report(),
    }


def pip_freeze() -> str:
    """pip freeze text for the run directory snapshot (section 6.3)."""
    out = _run([sys.executable, "-m", "pip", "freeze"], timeout=60)
    return out or "pip freeze unavailable"
