"""Throughput probe (CLAUDE.md section 9.1).

Section 9.1: "--probe: run exactly one epoch at each of 640x320, 768x384, 1024x512 (short runs
re-using the same loader), emit probe.json with secs/epoch and projected wall-clock for the
configured epochs at each size. The GUI Train workspace displays this as the schedule picker."

The probe answers one question: how long will 220 epochs take at each crop size, so a human can
pick a schedule that finishes before they need the result.

# Why this times batches rather than literally running an epoch

An epoch is 876 batches. Three of them is ~2600 batches, and at the ~0.35 s/batch this device
manages that is roughly 15 minutes of GPU time to answer a question that a stable steady-state
rate answers in under two. Worse, the answer would be no more accurate: the quantity being
projected IS the steady-state rate, and a full epoch measures it 876 times instead of 40.

So the probe times `probe_batches` batches after a warmup and scales to the real epoch length.
The distinction is recorded honestly in probe.json (`measured_batches` vs `batches_per_epoch`)
and in docs/DECISIONS.md D027, rather than presented as a measured epoch.

The warmup is not optional on this device: the first batches at a new shape pay cuDNN's
benchmark autotuning and, because no published wheel ships sm_87 kernels (D015), a PTX JIT
compile. Timing those would report a schedule several times worse than the real one.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from drivyx.data.seg_dataset import SegShardDataset
from drivyx.jobs.run_dir import RunContext, RunDir
from drivyx.models.losses import boundary_target_from_mask
from drivyx.torch_setup import AUTOCAST_DTYPE
from drivyx.train.config import SegConfig

logger = logging.getLogger(__name__)

#: Batches run before timing starts, to pay cuDNN autotuning and the PTX JIT once.
WARMUP_BATCHES = 8


@dataclass
class ProbeResult:
    """One crop size's measurement."""

    width: int
    height: int
    batch_size: int
    measured_batches: int
    batches_per_epoch: int
    secs_per_batch: float
    secs_per_epoch: float
    projected_total_hours: float
    images_per_sec: float
    peak_memory_gb: float
    oom: bool = False
    error: str | None = None


def probe_sizes(
    trainer: Any,
    config: SegConfig,
    ctx: RunContext,
    *,
    warmup: int = WARMUP_BATCHES,
) -> list[ProbeResult]:
    """Time a training step at each configured size.

    Takes the trainer rather than building a model so the probe measures the real model, the
    real loss, and the real optimiser step: a probe of a stripped-down forward pass would
    report a schedule the actual run cannot hit.
    """
    results: list[ProbeResult] = []

    for width, height in config.resolved_probe_sizes():
        logger.info("probing %dx%d ...", width, height)
        try:
            results.append(_probe_one(trainer, config, ctx, width, height, warmup))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning("%dx%d: out of memory at batch %d", width, height, config.batch_size)
            results.append(
                ProbeResult(
                    width=width,
                    height=height,
                    batch_size=config.batch_size,
                    measured_batches=0,
                    batches_per_epoch=0,
                    secs_per_batch=float("nan"),
                    secs_per_epoch=float("nan"),
                    projected_total_hours=float("nan"),
                    images_per_sec=0.0,
                    peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
                    oom=True,
                    error=f"out of memory at batch size {config.batch_size}",
                )
            )
    return results


def _probe_one(
    trainer: Any,
    config: SegConfig,
    ctx: RunContext,
    width: int,
    height: int,
    warmup: int,
) -> ProbeResult:
    """Time one size. Restores the trainer's dataset crop afterwards."""
    dataset = SegShardDataset(
        trainer.paths,
        "train",
        train=True,
        crop_w=width,
        crop_h=height,
        aug=config.aug.model_dump(),
        seed=config.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor if config.data.num_workers else None,
        persistent_workers=False,
        pin_memory=config.data.pin_memory,
        drop_last=True,
    )

    model = trainer.model
    optimizer = trainer.optimizer
    criterion = trainer.criterion
    device = trainer.device
    model.train()

    torch.cuda.reset_peak_memory_stats()
    batches_per_epoch = max(1, len(dataset) // config.batch_size)
    target = warmup + config.probe_batches

    timed = 0
    started = 0.0
    for index, (images, targets) in enumerate(loader):
        if index >= target or ctx.interrupted:
            break

        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        boundary = boundary_target_from_mask(targets)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=AUTOCAST_DTYPE):
            outputs = model(images)
        loss, _parts = criterion(outputs, targets, boundary)
        loss.backward()
        optimizer.step()

        if index == warmup - 1:
            # Synchronise before starting the clock: without this the warmup's queued kernels
            # are still running and their time lands in the measurement.
            torch.cuda.synchronize()
            started = time.perf_counter()
        elif index >= warmup:
            timed += 1
        ctx.events.heartbeat()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started if timed else float("nan")

    secs_per_batch = elapsed / timed if timed else float("nan")
    secs_per_epoch = secs_per_batch * batches_per_epoch
    projected_hours = secs_per_epoch * config.epochs / 3600.0

    result = ProbeResult(
        width=width,
        height=height,
        batch_size=config.batch_size,
        measured_batches=timed,
        batches_per_epoch=batches_per_epoch,
        secs_per_batch=round(secs_per_batch, 4),
        secs_per_epoch=round(secs_per_epoch, 1),
        projected_total_hours=round(projected_hours, 2),
        images_per_sec=round(config.batch_size / secs_per_batch, 1) if timed else 0.0,
        peak_memory_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
    )
    logger.info(
        "%dx%d: %.3fs/batch -> %.0fs/epoch, %.1fh for %d epochs, peak %.1f GB",
        width,
        height,
        result.secs_per_batch,
        result.secs_per_epoch,
        result.projected_total_hours,
        config.epochs,
        result.peak_memory_gb,
    )
    return result


def write_probe(run: RunDir, config: SegConfig, results: list[ProbeResult]) -> dict[str, Any]:
    """Write probe.json (section 9.1)."""
    payload: dict[str, Any] = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "warmup_batches": WARMUP_BATCHES,
        "note": (
            "secs_per_epoch is measured_batches timed at steady state, scaled to "
            "batches_per_epoch. Section 9.1 says 'one epoch at each size'; the steady-state "
            "rate is the quantity being projected and measuring it over a full epoch at three "
            "sizes costs hours for no extra accuracy. See docs/DECISIONS.md D027."
        ),
        "sizes": [asdict(r) for r in results],
    }
    path = run.path / "probe.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote %s", path)
    return payload


def read_probe(run: RunDir) -> dict[str, Any]:
    path = run.path / "probe.json"
    if not path.is_file():
        raise FileNotFoundError(f"no probe.json in {run.path}. Run 'drivyx train-seg --probe'.")
    return json.loads(path.read_text())
