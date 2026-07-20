"""Torch runtime configuration (CLAUDE.md section 3).

Import-time CUDA calls are forbidden by section 3 ("importable and unit-testable on a
CPU-only machine"), so every flag is applied inside configure(), which callers invoke once
at process start. Importing this module touches nothing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

#: bf16 on Orin's Ampere GPU has the same exponent range as fp32, so training needs no loss
#: scaling and section 3 accordingly specifies autocast bf16 with no GradScaler.
AUTOCAST_DTYPE = torch.bfloat16


@dataclass(frozen=True)
class DeviceInfo:
    """What configure() actually found, for env.txt and the SYSTEM workspace."""

    cuda_available: bool
    device_name: str
    torch_version: str
    cuda_version: str | None
    cudnn_version: int | None
    capability: tuple[int, int] | None
    bf16_supported: bool

    def summary(self) -> str:
        if not self.cuda_available:
            return f"torch {self.torch_version} (CPU only)"
        cap = f"sm_{self.capability[0]}{self.capability[1]}" if self.capability else "unknown"
        return (
            f"torch {self.torch_version} / CUDA {self.cuda_version} / "
            f"cuDNN {self.cudnn_version} / {self.device_name} ({cap})"
        )


def configure(*, deterministic: bool = False) -> DeviceInfo:
    """Apply the section 3 runtime flags and report what the device supports.

    Safe to call on a CPU-only machine: the CUDA-specific flags are skipped and
    DeviceInfo.cuda_available reports False rather than raising.

    Args:
        deterministic: When True, disable cudnn.benchmark and request deterministic
            algorithms. Benchmark autotuning picks different kernels per shape, which
            defeats reproducibility, so the two are mutually exclusive.
    """
    torch.set_default_dtype(torch.float32)

    cuda_available = torch.cuda.is_available()

    if deterministic:
        # Autotuning and determinism cannot both hold; section 6.3 wants reproducibility, so
        # the caller chooses which one applies to a given run.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # TF32 costs a few mantissa bits on matmul/conv accumulate and buys a large speedup on
    # Ampere. Section 3 allows it for both matmul and cudnn.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not cuda_available:
        info = DeviceInfo(
            cuda_available=False,
            device_name="cpu",
            torch_version=torch.__version__,
            cuda_version=None,
            cudnn_version=None,
            capability=None,
            bf16_supported=False,
        )
        logger.info("Torch configured: %s", info.summary())
        return info

    capability = torch.cuda.get_device_capability(0)
    info = DeviceInfo(
        cuda_available=True,
        device_name=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        capability=capability,
        bf16_supported=torch.cuda.is_bf16_supported(),
    )
    logger.info("Torch configured: %s", info.summary())
    return info


def require_cuda() -> DeviceInfo:
    """configure() for jobs that cannot run without the GPU.

    Training, export, and benchmarking abort here rather than silently falling back to CPU
    and producing numbers that mean nothing (rule: fail loudly).
    """
    info = configure()
    if not info.cuda_available:
        raise RuntimeError(
            "CUDA is not available to torch, and this command requires the GPU. "
            "Run scripts/setup_orin.sh and check that the installed torch wheel is a CUDA "
            "build (torch.version.cuda must not be None). "
            f"Current: torch {info.torch_version}, torch.version.cuda={torch.version.cuda}."
        )
    if not info.bf16_supported:
        raise RuntimeError(
            f"bf16 autocast is required by CLAUDE.md section 3 but {info.device_name} does "
            "not support it."
        )
    return info
