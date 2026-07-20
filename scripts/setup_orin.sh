#!/usr/bin/env bash
#
# DRIVYX environment bootstrap for NVIDIA Jetson AGX Orin.
#
# Encodes CLAUDE.md section 3, re-derived for the platform actually on this device
# (JetPack 7.2 / L4T R39.2 / CUDA 13.2 / Python 3.12). See docs/DECISIONS.md D001-D004 for
# why each spec'd value moved.
#
# Idempotent: safe to re-run. Every step checks before it acts.
#
# Usage:
#   bash scripts/setup_orin.sh [--yes] [--venv PATH] [--skip-jetpack]
#
#   --yes           Do not prompt; accept the apt install and power mode changes.
#   --venv PATH     Virtualenv location (default: .venv).
#   --skip-jetpack  Do not install the nvidia-jetpack meta-package even if CUDA is absent.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_PATH=".venv"
ASSUME_YES=0
SKIP_JETPACK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y) ASSUME_YES=1; shift ;;
    --venv) VENV_PATH="$2"; shift 2 ;;
    --skip-jetpack) SKIP_JETPACK=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

# --- output helpers ---------------------------------------------------------------------

if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_INFO=$'\033[34m'; C_OK=$'\033[32m'
  C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_BOLD=$'\033[1m'
else
  C_RESET=""; C_INFO=""; C_OK=""; C_WARN=""; C_ERR=""; C_BOLD=""
fi

info() { echo "${C_INFO}[ .. ]${C_RESET} $*"; }
ok()   { echo "${C_OK}[ ok ]${C_RESET} $*"; }
warn() { echo "${C_WARN}[warn]${C_RESET} $*" >&2; }
die()  { echo "${C_ERR}[FAIL]${C_RESET} $*" >&2; exit 1; }
step() { echo; echo "${C_BOLD}== $* ==${C_RESET}"; }

confirm() {
  # Prompt unless --yes. Returns 0 for yes.
  local prompt="$1"
  if [[ "$ASSUME_YES" == "1" ]]; then return 0; fi
  if [[ ! -t 0 ]]; then
    warn "Not an interactive shell and --yes not given; skipping: $prompt"
    return 1
  fi
  read -r -p "$prompt [y/N] " reply
  [[ "$reply" =~ ^[Yy] ]]
}

# --- wheel sources ----------------------------------------------------------------------
#
# CLAUDE.md section 3 mandates https://pypi.jetson-ai-lab.dev/jp6/cu126. That host no longer
# resolves in DNS, and its surviving mirror serves cp310/CUDA-12.6 wheels that cannot install
# on this cp312/CUDA-13.2 device. The official cu130 index publishes real CUDA aarch64
# (SBSA) wheels that match. DECISIONS.md D002; confirmed by the user 2026-07-17.

# torch and torchvision ship as a matched pair: torchvision pins an exact torch version
# (0.28.0 requires torch==2.13.0), so these two constants must always be bumped together.
TORCH_INDEX="https://download.pytorch.org/whl/cu130"
TORCH_SPEC="torch==2.13.0"
TORCHVISION_SPEC="torchvision==0.28.0"
# Retained for provenance reporting and for the on-device evidence trail.
SPEC_JETSON_INDEX="https://pypi.jetson-ai-lab.dev/jp6/cu126"

# --- 1. platform ------------------------------------------------------------------------

step "Platform"

if [[ -f /etc/nv_tegra_release ]]; then
  L4T_LINE="$(head -1 /etc/nv_tegra_release)"
  info "L4T: $L4T_LINE"
  # Section 3: "must show R36.x. Warn and continue if not."
  if grep -qE '^# R36' /etc/nv_tegra_release; then
    ok "L4T R36.x as CLAUDE.md section 3 expects"
  else
    L4T_REL="$(grep -oE '^# R[0-9]+' /etc/nv_tegra_release | tr -d '# ' || echo unknown)"
    warn "L4T is ${L4T_REL}, not R36.x. CLAUDE.md section 3 says warn and continue."
    warn "Wheel selection follows the detected platform. See docs/DECISIONS.md D001."
  fi
else
  warn "/etc/nv_tegra_release absent. This does not look like a Jetson; continuing."
fi

info "Python: $(python3 --version 2>&1)  ($(command -v python3))"
info "Arch:   $(uname -m)"

if [[ "$(uname -m)" != "aarch64" ]]; then
  warn "Not aarch64. The CUDA wheels selected below are aarch64-only and will not install."
fi

# --- 2. power ---------------------------------------------------------------------------

step "Power mode"

if command -v nvpmodel >/dev/null 2>&1; then
  NVP_OUT="$(nvpmodel -q 2>/dev/null || true)"
  NVP_MODE="$(echo "$NVP_OUT" | grep -oP 'NV Power Mode:\s*\K\S+' || echo unknown)"
  if [[ "$NVP_MODE" == MAXN* ]]; then
    ok "nvpmodel: $NVP_MODE"
  else
    warn "nvpmodel: $NVP_MODE (not MAXN). Training will be slower and the GUI shows an amber banner."
    if confirm "Switch to MAXN (mode 0) and lock clocks now? (needs sudo)"; then
      sudo nvpmodel -m 0 && sudo jetson_clocks && ok "MAXN set and clocks locked"
    else
      info "Left as ${NVP_MODE}. Run: sudo nvpmodel -m 0 && sudo jetson_clocks"
    fi
  fi
else
  warn "nvpmodel not found; cannot verify power mode."
fi

# jetson_clocks --show refuses to run as a non-root user, so only query it when we already
# have a cached sudo ticket. Its absence is cosmetic: nvpmodel above is the load-bearing check.
if command -v jetson_clocks >/dev/null 2>&1; then
  if sudo -n true 2>/dev/null; then
    info "jetson_clocks --show (first 3 lines):"
    sudo jetson_clocks --show 2>/dev/null | head -3 | sed 's/^/       /' || true
  else
    info "jetson_clocks --show needs root; skipping (run 'sudo jetson_clocks --show' to inspect)."
  fi
fi

# --- 3. CUDA toolkit --------------------------------------------------------------------

step "CUDA toolkit and cuDNN"

# DECISIONS.md D003: this device shipped without nvidia-jetpack, so there is no CUDA
# toolkit and no cuDNN. The torch cu130 wheels bundle their own CUDA runtime libraries, but
# TensorRT engine building and the INT8 calibrator link against the system CUDA stack.
CUDA_PRESENT=0
if ls -d /usr/local/cuda* >/dev/null 2>&1 || dpkg -l 2>/dev/null | grep -q '^ii  cuda-cudart'; then
  CUDA_PRESENT=1
  ok "CUDA toolkit present: $(ls -d /usr/local/cuda* 2>/dev/null | head -1)"
else
  warn "No CUDA toolkit found (no /usr/local/cuda*, no cuda-cudart package)."
fi

if [[ "$CUDA_PRESENT" == "0" && "$SKIP_JETPACK" == "0" ]]; then
  JETPACK_CAND="$(apt-cache policy nvidia-jetpack 2>/dev/null | awk '/Candidate:/{print $2}')"
  if [[ -n "${JETPACK_CAND:-}" && "$JETPACK_CAND" != "(none)" ]]; then
    info "nvidia-jetpack candidate: $JETPACK_CAND (several GB download)"
    if confirm "Install nvidia-jetpack now? (needs sudo)"; then
      sudo apt-get update
      sudo apt-get install -y nvidia-jetpack
      ok "nvidia-jetpack installed"
    else
      warn "Skipped. TensorRT export (M7) needs the CUDA toolkit; re-run with --yes when ready."
    fi
  else
    warn "nvidia-jetpack is not available from apt. Check your apt sources for the L4T repo."
  fi
fi

if command -v trtexec >/dev/null 2>&1 || [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
  ok "trtexec: $(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)"
else
  warn "trtexec not found. Export (M7) will abort until libnvinfer-bin is installed."
fi

# --- 4. virtualenv ----------------------------------------------------------------------

step "Virtualenv"

# Ubuntu splits venv support out of the base python3 package, and this image shipped without
# it, so `python3 -m venv` fails with "ensurepip is not available". Installing the versioned
# python3.X-venv package is the reproducible fix (docs/DECISIONS.md D014).
if ! python3 -c 'import ensurepip' >/dev/null 2>&1; then
  PY_MM="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  warn "ensurepip is unavailable, so 'python3 -m venv' cannot create a virtualenv."
  info "This needs the python${PY_MM}-venv apt package."
  if confirm "Install python${PY_MM}-venv via apt? (needs sudo)"; then
    sudo apt-get update
    # Fall back to the unversioned package on distros that do not ship a versioned one.
    sudo apt-get install -y "python${PY_MM}-venv" || sudo apt-get install -y python3-venv
    python3 -c 'import ensurepip' >/dev/null 2>&1 \
      || die "ensurepip is still unavailable after installing python${PY_MM}-venv."
    ok "python${PY_MM}-venv installed"
  else
    die "Cannot continue without venv support. Run: sudo apt install python${PY_MM}-venv"
  fi
fi

# Section 3: MUST use --system-site-packages so the JetPack tensorrt bindings and
# DeepStream libs stay importable. They are apt-installed into dist-packages and are not
# available on any pip index.
if [[ -d "$VENV_PATH" ]]; then
  if [[ ! -f "$VENV_PATH/pyvenv.cfg" ]]; then
    die "$VENV_PATH exists but is not a virtualenv. Remove it and re-run."
  fi
  if ! grep -q 'include-system-site-packages = true' "$VENV_PATH/pyvenv.cfg"; then
    die "$VENV_PATH was created without --system-site-packages, so 'import tensorrt' will fail.
       Remove it and re-run: rm -rf $VENV_PATH && bash scripts/setup_orin.sh"
  fi
  # A venv built while ensurepip was missing looks structurally valid but has no pip. Treat
  # that as broken and rebuild rather than failing later on an obscure pip error.
  if ! "$VENV_PATH/bin/python" -m pip --version >/dev/null 2>&1; then
    warn "$VENV_PATH exists but has no working pip (likely a half-built venv). Recreating."
    rm -rf "$VENV_PATH"
  else
    ok "Reusing $VENV_PATH (system-site-packages enabled)"
  fi
fi

if [[ ! -d "$VENV_PATH" ]]; then
  info "Creating $VENV_PATH with --system-site-packages"
  python3 -m venv --system-site-packages "$VENV_PATH" \
    || die "venv creation failed. If this says 'ensurepip is not available', install the
       python3-venv package and re-run."
  ok "Created $VENV_PATH"
fi

# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"
PY="$VENV_PATH/bin/python"

info "venv python: $($PY --version 2>&1)"
"$PY" -m pip install --upgrade pip setuptools wheel >/dev/null
ok "pip $($PY -m pip --version | awk '{print $2}')"

# --- 5. torch ---------------------------------------------------------------------------

step "PyTorch"

# Evidence trail: prove on-device that the spec'd index is gone before using the fallback.
if getent hosts "$(echo "$SPEC_JETSON_INDEX" | awk -F/ '{print $3}')" >/dev/null 2>&1; then
  info "Spec'd Jetson index host resolves; checking whether it serves this platform."
else
  warn "Spec'd index $SPEC_JETSON_INDEX does not resolve (DNS). Using $TORCH_INDEX (D002)."
fi

if "$PY" -c 'import torch, sys; sys.exit(0 if torch.version.cuda else 1)' 2>/dev/null; then
  ok "torch already installed: $("$PY" -c 'import torch; print(torch.__version__)')"
else
  info "Installing $TORCH_SPEC $TORCHVISION_SPEC from $TORCH_INDEX"
  # numpy is pinned first so the resolver cannot drag in numpy 2.x as a torch dependency.
  "$PY" -m pip install "numpy==1.26.4"
  "$PY" -m pip install --index-url "$TORCH_INDEX" "$TORCH_SPEC" "$TORCHVISION_SPEC"
  ok "torch installed"
fi

# --- 6. drivyx and dependencies ---------------------------------------------------------

step "DRIVYX package"

info "Installing drivyx (editable) with gui and dev extras"
"$PY" -m pip install -e ".[gui,dev]"
ok "drivyx installed"

# Section 3: opencv-python bundles Qt5 plugins that hijack the xcb platform plugin and break
# PyQt6. Only the headless wheel may be present.
if "$PY" -m pip show opencv-python >/dev/null 2>&1; then
  warn "opencv-python (non-headless) is installed and will break PyQt6's xcb plugin. Removing."
  "$PY" -m pip uninstall -y opencv-python
fi

# --- 7. PyQt6 provenance ----------------------------------------------------------------

step "PyQt6"

# Qt 6.5+ needs libxcb-cursor0 to load the xcb platform plugin. Without it Qt does NOT fail:
# it silently falls back to the "offscreen" platform, so the app "runs" and renders nowhere.
# Installing it is the reproducible fix (docs/DECISIONS.md D017).
if ! ldconfig -p 2>/dev/null | grep -q 'libxcb-cursor\.so'; then
  warn "libxcb-cursor0 is missing. Qt would silently fall back to the offscreen platform"
  warn "plugin, and drivyx-gui would render nothing on the desktop."
  if confirm "Install libxcb-cursor0 via apt? (needs sudo)"; then
    sudo apt-get install -y libxcb-cursor0
    ok "libxcb-cursor0 installed"
  else
    warn "Skipped. The CLI engine is unaffected; drivyx-gui will not display."
  fi
fi

# Section 3: try the pip wheel, fall back to apt (the venv sees it via system-site-packages),
# and report which path was taken. The SYSTEM workspace surfaces this at runtime.
PYQT_PATH="unknown"
if "$PY" -c 'import PyQt6.QtWidgets' 2>/dev/null; then
  PYQT_LOC="$("$PY" -c 'import PyQt6, pathlib; print(pathlib.Path(PyQt6.__file__).parent)')"
  if [[ "$PYQT_LOC" == *dist-packages* ]]; then
    PYQT_PATH="apt (python3-pyqt6, via system-site-packages)"
  else
    PYQT_PATH="pip wheel"
  fi
  ok "PyQt6 available via $PYQT_PATH: $("$PY" -c 'from PyQt6.QtCore import PYQT_VERSION_STR; print(PYQT_VERSION_STR)')"
else
  warn "PyQt6 pip wheel did not resolve. Falling back to apt python3-pyqt6."
  if confirm "Install python3-pyqt6 via apt? (needs sudo)"; then
    sudo apt-get install -y python3-pyqt6 python3-pyqt6.qtsvg
    if "$PY" -c 'import PyQt6.QtWidgets' 2>/dev/null; then
      PYQT_PATH="apt (python3-pyqt6, via system-site-packages)"
      ok "PyQt6 available via $PYQT_PATH"
    else
      die "PyQt6 still not importable after apt install. The venv may lack --system-site-packages."
    fi
  else
    warn "PyQt6 unavailable. The CLI engine works headless; drivyx-gui will not start."
  fi
fi

# --- 8. onnxsim (optional) --------------------------------------------------------------

step "onnxsim (optional)"

# Section 3: optional on aarch64. The export path must work without it, so a failure here is
# informational only and onnx_export.py guards the import.
if "$PY" -c 'import onnxsim' 2>/dev/null; then
  ok "onnxsim present"
elif "$PY" -m pip install onnxsim >/dev/null 2>&1; then
  ok "onnxsim installed"
else
  warn "onnxsim did not build on aarch64. Export proceeds without graph simplification."
fi

# --- 9. verification --------------------------------------------------------------------

step "Verification"

# Section 3 and the M0 gate: abort with a clear message if CUDA is not actually available.
"$PY" - <<'PYCODE' || die "Environment verification failed. See the errors above."
import sys

failures = []

try:
    import torch
    print(f"  torch        : {torch.__version__}  (CUDA build: {torch.version.cuda})")
    if not torch.cuda.is_available():
        failures.append(
            "torch.cuda.is_available() is False.\n"
            "       The installed wheel may be CPU-only, or the driver/toolkit is mismatched.\n"
            f"       torch.version.cuda={torch.version.cuda!r}\n"
            "       Check: nvidia-smi, and that nvidia-jetpack is installed."
        )
    else:
        cc = torch.cuda.get_device_capability(0)
        arch = f"sm_{cc[0]}{cc[1]}"
        print(f"  cuda device  : {torch.cuda.get_device_name(0)}")
        print(f"  capability   : {arch}")
        print(f"  bf16 support : {torch.cuda.is_bf16_supported()}")
        print(f"  arch list    : {', '.join(torch.cuda.get_arch_list())}")

        # torch.cuda.is_available() is necessary but NOT sufficient: it returns True even
        # when the wheel ships no kernels for this GPU's architecture. On this device the
        # cu130 wheels exclude sm_87 and reach Orin only by JIT-compiling compute_80 PTX.
        # So actually execute a kernel, in the bf16 + channels_last configuration section 3
        # mandates for training, and let a real failure surface here rather than at hour six
        # of a training run. See docs/DECISIONS.md D015.
        if arch not in torch.cuda.get_arch_list():
            print(f"  NOTE         : {arch} is absent from the wheel's arch list; kernels reach"
                  f" this GPU by PTX JIT. Verifying by execution.")
        conv = torch.nn.Conv2d(8, 8, 3, padding=1).cuda().to(memory_format=torch.channels_last)
        x = torch.randn(2, 8, 64, 64, device="cuda").to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = conv(x).float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(loss):
            failures.append(f"CUDA smoke test produced a non-finite loss: {loss}")
        elif conv.weight.grad is None or not torch.isfinite(conv.weight.grad).all():
            failures.append("CUDA smoke test produced non-finite gradients.")
        else:
            print(f"  cuda kernel  : conv2d+bf16+backward OK (loss={loss.item():.4f})")
except Exception as exc:
    failures.append(
        f"torch CUDA verification failed: {type(exc).__name__}: {exc}\n"
        "       If this says 'no kernel image is available for execution on the device',\n"
        "       the installed wheel has no kernels for this GPU and cannot train."
    )

try:
    import tensorrt
    print(f"  tensorrt     : {tensorrt.__version__}")
except Exception as exc:
    failures.append(
        f"import tensorrt failed: {exc}\n"
        "       The venv must be created with --system-site-packages."
    )

try:
    import cv2
    print(f"  cv2          : {cv2.__version__}")
except Exception as exc:
    failures.append(f"import cv2 failed: {exc}")

try:
    import os

    from PyQt6.QtCore import PYQT_VERSION_STR
    print(f"  PyQt6        : {PYQT_VERSION_STR}")

    # Importing PyQt6 proves nothing about whether it can display. Qt silently falls back to
    # the "offscreen" platform when libxcb-cursor0 is absent, so the GUI would launch and
    # render into the void. Only instantiating a QApplication reveals which platform plugin
    # actually loaded. Skipped without a display, where offscreen is the correct answer.
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        from PyQt6.QtWidgets import QApplication
        _app = QApplication([])
        platform = _app.platformName()
        if platform == "offscreen":
            failures.append(
                "Qt loaded the 'offscreen' platform plugin despite a display being present.\n"
                "       drivyx-gui would run but render nothing. Install libxcb-cursor0:\n"
                "         sudo apt install libxcb-cursor0"
            )
        else:
            print(f"  Qt platform  : {platform}")
        del _app
    else:
        print("  Qt platform  : not checked (no DISPLAY; engine works headless)")
except Exception as exc:
    print(f"  PyQt6        : NOT AVAILABLE ({exc})")

try:
    import numpy
    print(f"  numpy        : {numpy.__version__}")
except Exception as exc:
    failures.append(f"import numpy failed: {exc}")

if failures:
    print("\nFAILURES:", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    sys.exit(1)
PYCODE

ok "Environment verified"

step "Done"
echo "  Activate:    source $VENV_PATH/bin/activate"
echo "  PyQt6 path:  $PYQT_PATH"
echo "  Next:        bash scripts/stage_data.sh    (extract IDD archives)"
echo "               drivyx verify-data | python -m json.tool"
