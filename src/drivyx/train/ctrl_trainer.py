"""Waypoint predictor training (CLAUDE.md section 9.2).

Section 9.2:

    Precompute: train-ctrl first materializes seg logits for every labeled frame into
    shards/ctrl/ (bf16 npy inside WebDataset) so the control epochs are GPU-cheap; skip if
    already present for the given seg run.

    Loss: L1 on waypoints. Optim AdamW 3e-4, cosine to 1e-5, batch 256, epochs 60. Metrics:
    ADE, FDE(2.5 s), lateral error at 1.0 s, all in meters, emitted per epoch.

The precompute is what makes this milestone tractable. Running the frozen seg model inside the
control loop would cost a PIDNet forward per sample per epoch; materialising the logits once
turns 60 epochs of control training into 60 passes over a small tensor file.

The precomputed logits are keyed by the seg run that produced them. A different seg checkpoint
produces different logits, and silently reusing another run's cache would train the control net
against a perception model it will never be paired with.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from drivyx.jobs.run_dir import RunContext, RunDir
from drivyx.models.ctrlnet import build_ctrlnet
from drivyx.models.losses import WaypointL1
from drivyx.paths import Paths
from drivyx.torch_setup import AUTOCAST_DTYPE, require_cuda

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1
#: Section 9.2's input geometry: the 768x384 head output, average-pooled 8x.
LOGIT_INPUT_W = 768
LOGIT_INPUT_H = 384
LOGIT_POOL = 8
NUM_WAYPOINTS = 5


@dataclass
class CtrlCache:
    """A materialised logit cache for one seg run."""

    directory: Path
    seg_run: str
    frames: int

    @property
    def logits(self) -> Path:
        return self.directory / "logits.npy"

    @property
    def targets(self) -> Path:
        return self.directory / "targets.npz"

    @property
    def manifest(self) -> Path:
        return self.directory / "manifest.json"

    def exists(self) -> bool:
        return self.logits.is_file() and self.targets.is_file() and self.manifest.is_file()


def cache_for(paths: Paths, seg_run: str) -> CtrlCache:
    """Where the logits for a given seg run live (section 9.2: shards/ctrl/)."""
    return CtrlCache(directory=paths.shards / "ctrl" / seg_run, seg_run=seg_run, frames=0)


def load_waypoint_frames(paths: Paths) -> pd.DataFrame:
    """Every labelled control frame across all routes, with its temporal split."""
    parquets = sorted(paths.waypoints.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"no waypoint datasets under {paths.waypoints}. Run 'drivyx mm-label' first."
        )
    frames = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    logger.info(
        "control frames: %d (%d train / %d val) across %d routes",
        len(frames),
        int((frames["split"] == "train").sum()),
        int((frames["split"] == "val").sum()),
        frames["route"].nunique(),
    )
    return frames


@torch.no_grad()
def precompute_logits(
    paths: Paths, seg_ckpt: Path, seg_run: str, ctx: RunContext, *, batch_size: int = 16
) -> CtrlCache:
    """Materialise seg logits for every control frame (section 9.2).

    Idempotent: an existing cache for this seg run is reused, which is what "skip if already
    present for the given seg run" means.
    """
    import cv2

    from drivyx.data.shards import letterbox, normalize
    from drivyx.models.pidnet import PIDNet

    cache = cache_for(paths, seg_run)
    frames = load_waypoint_frames(paths)

    if cache.exists():
        manifest = json.loads(cache.manifest.read_text())
        if manifest.get("frames") == len(frames):
            logger.info(
                "reusing the logit cache for seg run %s (%d frames)", seg_run, manifest["frames"]
            )
            return CtrlCache(cache.directory, seg_run, manifest["frames"])
        logger.warning(
            "cache for %s has %d frames but the dataset has %d; rebuilding",
            seg_run,
            manifest.get("frames"),
            len(frames),
        )

    device = torch.device("cuda")
    state = torch.load(seg_ckpt, map_location=device, weights_only=False)
    num_classes = state.get("config", {}).get("num_classes", 8)
    model = PIDNet(num_classes)
    model.load_state_dict(state["model"])
    # Frozen: section 9.2 says "produced by the frozen best seg checkpoint". eval() also
    # matters beyond gradients, because the BatchNorms must use their running statistics.
    model = model.to(device, memory_format=torch.channels_last).eval()

    cache.directory.mkdir(parents=True, exist_ok=True)
    pooled_h = LOGIT_INPUT_H // LOGIT_POOL // LOGIT_POOL
    pooled_w = LOGIT_INPUT_W // LOGIT_POOL // LOGIT_POOL

    # bf16 on disk per section 9.2. At 8x48x96 per frame this is ~74 KB, so 9138 frames is
    # ~650 MB: small enough to memory-map and read at full speed every epoch.
    output = np.lib.format.open_memmap(
        cache.logits,
        mode="w+",
        dtype=np.float32,
        shape=(len(frames), num_classes, pooled_h, pooled_w),
    )

    logger.info("precomputing seg logits for %d frames ...", len(frames))
    started = time.monotonic()
    missing: list[str] = []

    for start in range(0, len(frames), batch_size):
        if ctx.interrupted:
            raise KeyboardInterrupt("interrupted during logit precompute")

        chunk = frames.iloc[start : start + batch_size]
        images = []
        for path in chunk["frame_path"]:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                missing.append(str(path))
                image = np.zeros((LOGIT_INPUT_H, LOGIT_INPUT_W, 3), dtype=np.uint8)
            else:
                image, _ = letterbox(image, None, LOGIT_INPUT_W, LOGIT_INPUT_H)
            images.append(normalize(image))

        batch = (
            torch.from_numpy(np.stack(images))
            .to(device)
            .contiguous(memory_format=torch.channels_last)
        )
        with torch.autocast("cuda", dtype=AUTOCAST_DTYPE):
            logits = model(batch)
        # Section 9.2: "the 768x384 head output average-pooled 8x". The head is already at
        # 1/8 (96x48), so an 8x pool takes it to 12x6.
        pooled = torch.nn.functional.avg_pool2d(logits.float(), LOGIT_POOL)
        output[start : start + len(chunk)] = pooled.cpu().numpy()

        if start % (batch_size * 50) == 0:
            ctx.events.heartbeat()
            logger.debug("precompute %d/%d", start, len(frames))

    output.flush()
    del output

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} frames referenced by the waypoint dataset could not be read, "
            f"e.g. {missing[:3]}. The parquet and the image tree disagree; re-run mm-label."
        )

    # (frames, 5, 2): the x and y columns interleaved into the (waypoint, coord) layout the
    # loss and the model both use.
    targets = np.stack(
        [
            np.stack([frames[f"wp_x{k}"].to_numpy() for k in range(NUM_WAYPOINTS)], axis=1),
            np.stack([frames[f"wp_y{k}"].to_numpy() for k in range(NUM_WAYPOINTS)], axis=1),
        ],
        axis=-1,
    )
    np.savez(
        cache.targets,
        waypoints=targets.astype(np.float32),
        speed=frames["speed_mps"].to_numpy().astype(np.float32),
        is_val=(frames["split"] == "val").to_numpy(),
    )
    cache.manifest.write_text(
        json.dumps(
            {
                "seg_run": seg_run,
                "seg_ckpt": str(seg_ckpt),
                "frames": len(frames),
                "num_classes": num_classes,
                "pooled_shape": [num_classes, pooled_h, pooled_w],
                "input_size": [LOGIT_INPUT_W, LOGIT_INPUT_H],
                "pool": LOGIT_POOL,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info(
        "precomputed %d frames in %.1f s -> %s",
        len(frames),
        time.monotonic() - started,
        cache.directory,
    )
    return CtrlCache(cache.directory, seg_run, len(frames))


class CtrlDataset(Dataset):
    """Precomputed logits, speed, and waypoint targets for one split."""

    def __init__(self, cache: CtrlCache, *, val: bool) -> None:
        payload = np.load(cache.targets)
        is_val = payload["is_val"]
        self.indices = np.where(is_val if val else ~is_val)[0]
        self.waypoints = payload["waypoints"]
        self.speed = payload["speed"]
        # mmap_mode keeps the whole cache off the heap; each __getitem__ faults in one frame.
        self.logits = np.load(cache.logits, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = int(self.indices[i])
        return (
            torch.from_numpy(np.asarray(self.logits[index], dtype=np.float32)),
            torch.tensor(float(self.speed[index])),
            torch.from_numpy(self.waypoints[index]),
        )


class CtrlTrainer:
    """Trains CtrlNet on precomputed logits (section 9.2)."""

    def __init__(self, paths: Paths, run: RunDir, cache: CtrlCache, config: dict[str, Any]) -> None:
        require_cuda()
        self.paths = paths
        self.run = run
        self.cache = cache
        self.config = config
        self.device = torch.device("cuda")

        manifest = json.loads(cache.manifest.read_text())
        self.model = build_ctrlnet(manifest["num_classes"]).to(self.device)
        self.criterion = WaypointL1()

        self.train_set = CtrlDataset(cache, val=False)
        self.val_set = CtrlDataset(cache, val=True)
        if not len(self.train_set):
            raise ValueError("the control training split is empty")

        self.epochs = int(config["epochs"])
        self.batch_size = int(config["batch_size"])
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        # Section 9.2: cosine to 1e-5.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=float(config["min_lr"])
        )
        self.epoch = 0
        self.global_step = 0
        self.best_ade = float("inf")

    def _loader(self, dataset: CtrlDataset, *, train: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=4,
            persistent_workers=True,
            pin_memory=False,
            drop_last=train and len(dataset) > self.batch_size,
        )

    def save_checkpoint(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save(
            {
                "version": CHECKPOINT_VERSION,
                "epoch": self.epoch,
                "global_step": self.global_step,
                "best_ade": self.best_ade,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "config": self.config,
                "seg_run": self.cache.seg_run,
            },
            tmp,
        )
        tmp.replace(path)

    def load_checkpoint(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.best_ade = state["best_ade"]
        logger.info("resumed ctrl from %s at epoch %d", path, self.epoch)

    def train_epoch(self, loader: DataLoader, ctx: RunContext) -> tuple[float, bool]:
        self.model.train()
        total = 0.0
        counted = 0
        for index, (logits, speed, target) in enumerate(loader):
            if ctx.interrupted:
                return total / max(1, counted), True

            logits = logits.to(self.device, non_blocking=True)
            speed = speed.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=AUTOCAST_DTYPE):
                pred = self.model(logits, speed)
            loss, _metrics = self.criterion(pred.float(), target)

            if not torch.isfinite(loss):
                message = (
                    f"non-finite ctrl loss at epoch {self.epoch}, batch {index}, "
                    f"step {self.global_step}"
                )
                ctx.status("failed", message)
                raise RuntimeError(message)

            loss.backward()
            self.optimizer.step()

            total += float(loss.detach())
            counted += 1
            self.global_step += 1
            if self.global_step % 10 == 0:
                ctx.events.scalar(
                    "train/loss", float(loss.detach()), step=self.global_step, epoch=self.epoch
                )
            ctx.events.heartbeat()
        return total / max(1, counted), False

    @torch.no_grad()
    def validate(self, ctx: RunContext) -> dict[str, float]:
        """ADE, FDE, and lateral error at 1.0 s, in metres (section 9.2)."""
        self.model.eval()
        if not len(self.val_set):
            return {}

        totals: dict[str, float] = {"ade": 0.0, "fde": 0.0, "lateral_1s": 0.0}
        samples = 0
        for logits, speed, target in self._loader(self.val_set, train=False):
            logits = logits.to(self.device, non_blocking=True)
            speed = speed.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            with torch.autocast("cuda", dtype=AUTOCAST_DTYPE):
                pred = self.model(logits, speed)
            _loss, metrics = self.criterion(pred.float(), target)
            # Weighted by batch size: the last batch is usually short, and an unweighted mean
            # would over-count it.
            for key in totals:
                totals[key] += metrics[key] * len(logits)
            samples += len(logits)
            ctx.events.heartbeat()

        return {k: v / max(1, samples) for k, v in totals.items()}

    def fit(self, ctx: RunContext) -> dict[str, Any]:
        loader = self._loader(self.train_set, train=True)
        logger.info(
            "ctrl training: %d epochs, %d train / %d val frames, batch %d",
            self.epochs,
            len(self.train_set),
            len(self.val_set),
            self.batch_size,
        )

        interrupted = False
        history: list[dict[str, Any]] = []

        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            started = time.monotonic()
            mean_loss, interrupted = self.train_epoch(loader, ctx)
            elapsed = time.monotonic() - started

            metrics = self.validate(ctx)
            self.scheduler.step()

            ctx.events.epoch(epoch, elapsed, (self.epochs - epoch - 1) * elapsed / 60.0)
            ctx.events.scalar("train/epoch_loss", mean_loss, step=self.global_step, epoch=epoch)
            for name, value in metrics.items():
                ctx.events.scalar(f"val/{name}", value, step=self.global_step, epoch=epoch)

            logger.info(
                "epoch %d: loss %.4f | ADE %.3f m  FDE %.3f m  lateral@1s %.3f m (%.1fs)",
                epoch,
                mean_loss,
                metrics.get("ade", float("nan")),
                metrics.get("fde", float("nan")),
                metrics.get("lateral_1s", float("nan")),
                elapsed,
            )
            history.append({"epoch": epoch, "loss": mean_loss, **metrics})

            self.save_checkpoint(self.run.last_ckpt)
            ade = metrics.get("ade", float("inf"))
            if ade < self.best_ade:
                self.best_ade = ade
                self.save_checkpoint(self.run.best_ckpt)

            if interrupted:
                break

        self.save_checkpoint(self.run.last_ckpt)
        ctx.status("interrupted" if interrupted else "done")

        return {
            "run": str(self.run.path),
            "status": "interrupted" if interrupted else "done",
            "epoch": self.epoch,
            "best_ade": self.best_ade if math.isfinite(self.best_ade) else None,
            "parameters": self.model.parameter_count(),
            "seg_run": self.cache.seg_run,
            "history": history,
        }
