"""PIDNet-S trainer (CLAUDE.md sections 6.2, 6.3, 6.4, 9.1).

Implements section 9.1's schedule, section 6.2's SIGINT semantics, section 6.3's run directory
contract, and section 6.4's event stream.

Three rules from the spec shape this file more than anything else:

  - "NaN loss aborts the run with status=failed and the offending batch indices logged"
    (9.1). A NaN is not a bad step to skip: it means the run is already ruined, and every
    later checkpoint would be poison.
  - "Resume restores model, optimizer, scheduler, epoch, and RNG state" (9.1). Anything less
    is not a resume, it is a warm restart that quietly changes the experiment.
  - "bf16 autocast, channels_last, no grad scaler" (9.1, section 3). bf16 has fp32's exponent
    range, so there is nothing for a scaler to rescue; adding one would only add a failure
    mode.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from drivyx.data.lut import IGNORE_ID
from drivyx.data.seg_dataset import SegShardDataset
from drivyx.data.shards import read_index
from drivyx.jobs.run_dir import RunContext, RunDir
from drivyx.models.losses import PIDNetLoss, boundary_target_from_mask, compute_class_weights
from drivyx.models.pidnet import build_pidnet
from drivyx.paths import Paths
from drivyx.torch_setup import AUTOCAST_DTYPE, require_cuda
from drivyx.train.config import SegConfig

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1


def _as_byte_cpu(state: torch.Tensor) -> torch.Tensor:
    """Coerce a saved RNG state back to the CPU uint8 tensor torch requires.

    Needed because checkpoints are loaded with map_location=cuda, which relocates every
    tensor in the file, and the RNG states are tensors like any other.
    """
    return state.detach().to(device="cpu", dtype=torch.uint8)


def seed_everything(seed: int) -> None:
    """Seed every RNG the run touches (section 6.3: reproducibility)."""
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def poly_lr(base_lr: float, step: int, total_steps: int, power: float, warmup: int) -> float:
    """Section 9.1: lr 0.01 poly decay power 0.9.

        lr = base * (1 - step/total) ** power

    Poly decay spends most of the run at a high rate and collapses near the end, which suits a
    fixed 220-epoch budget: there is no early stopping to protect, so there is no reason to
    decay early.
    """
    if warmup and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, max(0.0, step / max(1, total_steps)))
    return base_lr * (1.0 - progress) ** power


@dataclass
class ConfusionMatrix:
    """Streaming confusion matrix for mIoU (section 10).

    Accumulated as counts on the GPU rather than by storing predictions: a full val pass at
    1024x512 over 2036 images is 1e9 pixels, which is not something to hold in memory to
    compute a ratio.
    """

    num_classes: int
    matrix: torch.Tensor

    @classmethod
    def zeros(cls, num_classes: int, device: torch.device) -> ConfusionMatrix:
        return cls(
            num_classes=num_classes,
            matrix=torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device),
        )

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        valid = target != IGNORE_ID
        if not valid.any():
            return
        # bincount over (true * C + pred) is the standard trick: one pass, no python loop.
        indices = target[valid] * self.num_classes + pred[valid]
        counts = torch.bincount(indices, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def iou(self) -> torch.Tensor:
        intersection = self.matrix.diag().float()
        union = (self.matrix.sum(0) + self.matrix.sum(1) - self.matrix.diag()).float()
        # A class absent from both prediction and ground truth has union 0. Its IoU is
        # undefined, not zero: counting it as zero would drag mIoU down for a class the val
        # set never contained. NaN here, and nanmean below, keeps it out of the average.
        return torch.where(union > 0, intersection / union, torch.full_like(union, float("nan")))

    def miou(self) -> float:
        return float(torch.nanmean(self.iou()))


class SegTrainer:
    """Trains PIDNet-S per section 9.1."""

    def __init__(self, paths: Paths, config: SegConfig, run: RunDir) -> None:
        self.paths = paths
        self.config = config
        self.run = run
        self.device = torch.device("cuda")

        require_cuda()
        seed_everything(config.seed)

        index = read_index(paths)
        weights = compute_class_weights(
            index["splits"]["train"]["class_pixels"], cap=config.loss.class_weight_cap
        )
        logger.info(
            "class weights from the shard histogram: %s",
            [round(float(w), 3) for w in weights],
        )

        backbone = self._find_backbone()
        self.model, load_report = build_pidnet(config.num_classes, pretrained=backbone)
        logger.info("backbone: %s", load_report.summary())
        self.model = self.model.to(self.device, memory_format=torch.channels_last)

        self.criterion = PIDNetLoss(
            class_weights=weights,
            thresh=config.loss.ohem_thresh,
            min_kept=config.loss.ohem_min_kept,
            aux_weight=config.loss.aux_weight,
            boundary_weight=config.loss.boundary_weight,
            boundary_aware_weight=config.loss.boundary_aware_weight,
        ).to(self.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.optim.lr,
            momentum=config.optim.momentum,
            weight_decay=config.optim.weight_decay,
        )

        self.train_set = SegShardDataset(
            paths,
            "train",
            train=True,
            crop_w=config.data.crop_width,
            crop_h=config.data.crop_height,
            aug=config.aug.model_dump(),
            seed=config.seed,
        )
        self.val_set = SegShardDataset(
            paths,
            "val",
            train=False,
            val_w=config.data.val_width,
            val_h=config.data.val_height,
        )

        self.epoch = 0
        self.global_step = 0
        self.best_miou = float("-inf")
        self.steps_per_epoch = max(1, len(self.train_set) // config.batch_size)
        self.total_steps = self.steps_per_epoch * config.epochs

    def _find_backbone(self) -> Path:
        """Locate the ImageNet checkpoint, aborting with the hint if absent (section 9.1)."""
        candidates = sorted(self.paths.pretrained.glob("*.pth")) + sorted(
            self.paths.pretrained.glob("*.pth.tar")
        )
        if not candidates:
            raise FileNotFoundError(
                f"No PIDNet-S backbone in {self.paths.pretrained}. train-seg cannot start "
                "without it (CLAUDE.md section 9.1). Run 'drivyx verify-data' for the "
                "download hint."
            )
        return candidates[0]

    def _loader(self, dataset: SegShardDataset, *, train: bool) -> DataLoader:
        cfg = self.config.data
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size if train else 1,
            shuffle=train,
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers else None,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            pin_memory=cfg.pin_memory,
            drop_last=train,
        )

    # --- checkpointing (section 9.1: resume restores model, optim, scheduler, epoch, RNG) ---

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": CHECKPOINT_VERSION,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_miou": self.best_miou,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.model_dump(),
            # The scheduler is a closed-form function of global_step (poly_lr), so there is no
            # separate scheduler state to save: restoring global_step restores the schedule
            # exactly. Saving a torch scheduler object here would be redundant state that could
            # disagree with the step counter.
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
        }

    def save_checkpoint(self, path: Path) -> None:
        """Write atomically: a SIGINT during torch.save must not truncate last.pt.

        Saving directly to last.pt and being killed mid-write leaves a file that exists, has a
        plausible size, and fails to load, which is the worst of the three outcomes.
        """
        tmp = path.with_suffix(".tmp")
        torch.save(self.state_dict(), tmp)
        tmp.replace(path)
        logger.debug("checkpoint -> %s", path)

    def load_checkpoint(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"{path} has checkpoint version {state.get('version')}, expected "
                f"{CHECKPOINT_VERSION}."
            )
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.best_miou = state["best_miou"]

        rng = state.get("rng", {})
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            # RNG states must be CPU ByteTensors. torch.load(map_location=cuda) moves every
            # tensor in the file to the GPU, these included, and set_rng_state then rejects
            # them with "RNG state must be a torch.ByteTensor". Converting back is not
            # cosmetic: without it --resume fails outright.
            torch.set_rng_state(_as_byte_cpu(rng["torch"]))
            try:
                torch.cuda.set_rng_state_all([_as_byte_cpu(s) for s in rng["cuda"]])
            except (RuntimeError, ValueError, TypeError) as exc:
                # A resume on a different GPU count cannot restore per-device CUDA RNG. Say so
                # rather than pretending the resume is bit-exact.
                logger.warning("could not restore CUDA RNG state: %s", exc)
        logger.info(
            "resumed from %s at epoch %d, step %d, best mIoU %.4f",
            path,
            self.epoch,
            self.global_step,
            self.best_miou,
        )

    # --- the loop ---

    def train_epoch(self, loader: DataLoader, ctx: RunContext) -> tuple[float, bool]:
        """One epoch. Returns (mean loss, interrupted).

        Checks the interrupt flag between steps, where the model, optimiser, and step counter
        are mutually consistent, so the checkpoint written afterwards is loadable (section
        6.2).
        """
        self.model.train()
        total_loss = 0.0
        counted = 0

        for batch_index, (images, targets) in enumerate(loader):
            if ctx.interrupted:
                logger.info("interrupt acknowledged at epoch %d step %d", self.epoch, batch_index)
                return (total_loss / max(1, counted)), True

            images = images.to(self.device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            targets = targets.to(self.device, non_blocking=True)
            boundary = boundary_target_from_mask(targets)

            lr = poly_lr(
                self.config.optim.lr,
                self.global_step,
                self.total_steps,
                self.config.optim.poly_power,
                self.config.optim.warmup_iters,
            )
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=AUTOCAST_DTYPE):
                outputs = self.model(images)
            loss, parts = self.criterion(outputs, targets, boundary)

            # Section 9.1: "NaN loss aborts the run with status=failed and the offending batch
            # indices logged". Checked before backward: a NaN gradient would already have
            # contaminated the optimiser's momentum buffers by the time we noticed after.
            if not torch.isfinite(loss):
                indices = list(
                    range(
                        batch_index * self.config.batch_size,
                        (batch_index + 1) * self.config.batch_size,
                    )
                )
                message = (
                    f"non-finite loss ({float(loss)}) at epoch {self.epoch}, batch "
                    f"{batch_index}, global step {self.global_step}. Offending sample indices "
                    f"{indices[0]}..{indices[-1]}. Parts: {parts}"
                )
                ctx.status("failed", message)
                raise RuntimeError(message)

            loss.backward()

            # Gradient clipping is the safety net against divergence. The first run at lr 0.01
            # with no warmup and SGD momentum 0.9 trained for four epochs then walked out of
            # its basin: the loss climbed from 29 to over 400 and val mIoU fell from 0.43 to
            # 0.05 (docs/DECISIONS.md D032). PIDNet's D branch, SPP, and heads start random on
            # top of a pretrained backbone, so an early bad step can produce a large gradient
            # that momentum then amplifies. Clipping the global norm bounds that step without
            # changing the loss the model optimises. A clip of 0 disables it.
            if self.config.optim.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optim.grad_clip
                )
                if self.global_step % 10 == 0:
                    ctx.events.scalar(
                        "train/grad_norm", float(grad_norm), step=self.global_step, epoch=self.epoch
                    )

            self.optimizer.step()

            value = float(loss.detach())
            total_loss += value
            counted += 1
            self.global_step += 1

            if self.global_step % 10 == 0:
                ctx.events.scalar("train/loss", value, step=self.global_step, epoch=self.epoch)
                ctx.events.scalar("train/lr", lr, step=self.global_step, epoch=self.epoch)
                for name, part in parts.items():
                    ctx.events.scalar(
                        f"train/{name.split('/')[1]}", part, step=self.global_step, epoch=self.epoch
                    )
            ctx.events.heartbeat()

        return (total_loss / max(1, counted)), False

    @torch.no_grad()
    def validate(
        self, ctx: RunContext, *, max_batches: int | None = None
    ) -> tuple[float, list[float]]:
        """Val mIoU at the configured size (sections 9.1, 10)."""
        self.model.eval()
        loader = self._loader(self.val_set, train=False)
        confusion = ConfusionMatrix.zeros(self.config.num_classes, self.device)

        for i, (images, targets) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            if ctx.interrupted:
                break
            images = images.to(self.device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            targets = targets.to(self.device, non_blocking=True)

            with torch.autocast("cuda", dtype=AUTOCAST_DTYPE):
                logits = self.model(images)
            # The head is 1/8 resolution; mIoU is defined against the full-size mask, so the
            # prediction is lifted rather than the target being downsampled (which would
            # measure a coarser problem than the one being solved).
            logits = torch.nn.functional.interpolate(
                logits.float(), size=targets.shape[-2:], mode="bilinear", align_corners=False
            )
            confusion.update(logits.argmax(1), targets)
            ctx.events.heartbeat()

        ious = [float(v) for v in confusion.iou()]
        return confusion.miou(), ious

    def fit(self, ctx: RunContext) -> dict[str, Any]:
        """Run the schedule. Returns a summary for the CLI."""
        loader = self._loader(self.train_set, train=True)
        logger.info(
            "training %d epochs, %d steps/epoch, %d total steps",
            self.config.epochs,
            self.steps_per_epoch,
            self.total_steps,
        )

        start_epoch = self.epoch
        interrupted = False
        epoch_times: list[float] = []

        for epoch in range(start_epoch, self.config.epochs):
            self.epoch = epoch
            self.train_set.set_epoch(epoch)
            started = time.monotonic()

            mean_loss, interrupted = self.train_epoch(loader, ctx)
            elapsed = time.monotonic() - started
            epoch_times.append(elapsed)

            remaining = self.config.epochs - epoch - 1
            eta_min = (sum(epoch_times) / len(epoch_times)) * remaining / 60.0
            ctx.events.epoch(epoch, elapsed, eta_min)
            ctx.events.scalar("train/epoch_loss", mean_loss, step=self.global_step, epoch=epoch)
            logger.info(
                "epoch %d: loss %.4f in %.1fs (eta %.1f min)", epoch, mean_loss, elapsed, eta_min
            )

            if interrupted:
                break

            # Section 9.1: checkpoint every epoch, track best val mIoU every 5 epochs.
            self.save_checkpoint(self.run.last_ckpt)

            is_last = epoch == self.config.epochs - 1
            if (epoch + 1) % self.config.val_every == 0 or is_last:
                miou, ious = self.validate(ctx)
                ctx.events.scalar("val/mIoU", miou, step=self.global_step, epoch=epoch)
                for i, iou in enumerate(ious):
                    if math.isfinite(iou):
                        ctx.events.scalar(f"val/IoU_{i}", iou, step=self.global_step, epoch=epoch)
                logger.info("epoch %d: val mIoU %.4f", epoch, miou)

                if miou > self.best_miou:
                    self.best_miou = miou
                    self.save_checkpoint(self.run.best_ckpt)
                    logger.info("new best mIoU %.4f -> %s", miou, self.run.best_ckpt.name)

        # Always leave last.pt current, including after an interrupt: it is what --resume reads.
        self.save_checkpoint(self.run.last_ckpt)

        if interrupted:
            ctx.status("interrupted", f"stopped at epoch {self.epoch}, step {self.global_step}")
        else:
            ctx.status("done", f"completed {self.config.epochs} epochs")

        return {
            "run": str(self.run.path),
            "status": "interrupted" if interrupted else "done",
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_miou": self.best_miou if math.isfinite(self.best_miou) else None,
            "last_ckpt": str(self.run.last_ckpt),
            "best_ckpt": str(self.run.best_ckpt) if self.run.best_ckpt.exists() else None,
        }
