"""Segmentation evaluation (CLAUDE.md section 10).

"eval-seg: per-class IoU + mIoU on IDD val at 1024x512, confusion matrix PNG, 24 qualitative
overlays. Emits eval/seg_metrics.json."

Section 10 also says "No accuracy gates are hardcoded; the numbers are the deliverable." So
nothing here passes or fails a run: it measures and reports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drivyx.data.lut import IGNORE_ID, read_lut
from drivyx.data.seg_dataset import SegShardDataset
from drivyx.jobs.run_dir import RunContext, RunDir
from drivyx.paths import Paths
from drivyx.torch_setup import AUTOCAST_DTYPE, require_cuda

logger = logging.getLogger(__name__)

#: Section 10: "24 qualitative overlays".
NUM_OVERLAYS = 24


@dataclass
class SegMetrics:
    """Per-class and aggregate segmentation metrics."""

    class_names: list[str]
    iou: list[float]
    miou: float
    pixel_accuracy: float
    mean_accuracy: float
    confusion: np.ndarray
    samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mIoU": self.miou,
            "pixel_accuracy": self.pixel_accuracy,
            "mean_accuracy": self.mean_accuracy,
            "samples": self.samples,
            "per_class_iou": {
                name: (None if np.isnan(v) else float(v))
                for name, v in zip(self.class_names, self.iou)
            },
            "confusion": self.confusion.tolist(),
        }


def metrics_from_confusion(
    confusion: torch.Tensor, class_names: list[str], samples: int
) -> SegMetrics:
    """Derive IoU, mIoU, and accuracies from a confusion matrix.

    Rows are ground truth, columns are prediction.

    A class absent from both the prediction and the ground truth has union zero and an
    undefined IoU. It is reported as NaN and excluded from the mean rather than counted as
    zero, which would penalise the model for a class the val set never contained.
    """
    matrix = confusion.double()
    intersection = matrix.diag()
    union = matrix.sum(0) + matrix.sum(1) - intersection

    iou = torch.where(union > 0, intersection / union, torch.full_like(union, float("nan")))
    total = matrix.sum()
    pixel_accuracy = float(intersection.sum() / total) if total > 0 else 0.0

    per_class_total = matrix.sum(1)
    per_class_acc = torch.where(
        per_class_total > 0,
        intersection / per_class_total,
        torch.full_like(per_class_total, float("nan")),
    )

    return SegMetrics(
        class_names=class_names,
        iou=[float(v) for v in iou],
        miou=float(torch.nanmean(iou)),
        pixel_accuracy=pixel_accuracy,
        mean_accuracy=float(torch.nanmean(per_class_acc)),
        confusion=confusion.cpu().numpy(),
        samples=samples,
    )


@torch.no_grad()
def evaluate_seg(
    paths: Paths,
    run: RunDir,
    ctx: RunContext,
    *,
    checkpoint: str = "best",
    max_samples: int | None = None,
    overlays: int = NUM_OVERLAYS,
) -> dict[str, Any]:
    """Evaluate a seg run on IDD val (section 10)."""

    from drivyx.eval.viz import overlay_mask
    from drivyx.models.pidnet import PIDNet

    require_cuda()
    device = torch.device("cuda")

    ckpt_path = run.best_ckpt if checkpoint == "best" else run.last_ckpt
    if not ckpt_path.is_file():
        # best.pt only appears after the first validation, so a short run may have only last.
        fallback = run.last_ckpt if checkpoint == "best" else run.best_ckpt
        if not fallback.is_file():
            raise FileNotFoundError(f"no checkpoint in {run.ckpt_dir}")
        logger.warning("%s not found, using %s", ckpt_path.name, fallback.name)
        ckpt_path = fallback

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = state.get("config", {})
    num_classes = config.get("num_classes", 8)

    model = PIDNet(num_classes)
    model.load_state_dict(state["model"])
    model = model.to(device, memory_format=torch.channels_last).eval()
    logger.info("evaluating %s (epoch %s)", ckpt_path.name, state.get("epoch"))

    data_cfg = config.get("data", {})
    dataset = SegShardDataset(
        paths,
        "val",
        train=False,
        val_w=data_cfg.get("val_width", 1024),
        val_h=data_cfg.get("val_height", 512),
    )

    lut = read_lut(paths.lut_json)
    class_names = [g["name"] for g in lut["groups"]]

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    overlay_dir = run.eval_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Overlays are sampled evenly across the val set rather than taken from the front, so the
    # 24 images span the dataset's variety instead of one contiguous sequence.
    total = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    overlay_at = set(np.linspace(0, total - 1, num=min(overlays, total), dtype=int).tolist())
    written: list[str] = []

    for index in range(total):
        if ctx.interrupted:
            logger.info("evaluation interrupted at sample %d", index)
            break

        image, target = dataset[index]
        batch = image.unsqueeze(0).to(device).contiguous(memory_format=torch.channels_last)
        target = target.unsqueeze(0).to(device)

        with torch.autocast("cuda", dtype=AUTOCAST_DTYPE):
            logits = model(batch)
        logits = torch.nn.functional.interpolate(
            logits.float(), size=target.shape[-2:], mode="bilinear", align_corners=False
        )
        prediction = logits.argmax(1)

        valid = target != IGNORE_ID
        if valid.any():
            pairs = target[valid] * num_classes + prediction[valid]
            confusion += torch.bincount(pairs, minlength=num_classes**2).reshape(
                num_classes, num_classes
            )

        if index in overlay_at:
            path = overlay_dir / f"val_{index:05d}.jpg"
            _write_overlay(path, image, prediction[0], target[0], overlay_mask)
            written.append(str(path.relative_to(run.path)))
            ctx.events.image("val/overlay", str(path.relative_to(run.path)))

        if index % 50 == 0:
            ctx.events.heartbeat()

    metrics = metrics_from_confusion(confusion, class_names, total)
    matrix_path = _write_confusion(run, metrics)

    payload = metrics.to_dict()
    payload.update(
        {
            "run": run.name,
            "checkpoint": ckpt_path.name,
            "epoch": state.get("epoch"),
            "confusion_png": str(matrix_path.relative_to(run.path)),
            "overlays": written,
        }
    )
    out = run.eval_dir / "seg_metrics.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")

    logger.info(
        "mIoU %.4f | pixel acc %.4f over %d samples -> %s",
        metrics.miou,
        metrics.pixel_accuracy,
        total,
        out,
    )
    for name, value in zip(class_names, metrics.iou):
        logger.info("  IoU %-14s %s", name, "n/a" if np.isnan(value) else f"{value:.4f}")

    return payload


def _write_overlay(
    path: Path, image: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor, overlay_fn
) -> None:
    """Write a prediction-over-image overlay beside the ground truth."""
    import cv2

    from drivyx.data.shards import IMAGENET_MEAN, IMAGENET_STD

    # Undo the normalisation so the overlay shows the real photograph.
    chw = image.numpy()
    rgb = (chw.transpose(1, 2, 0) * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN)) * 255.0
    bgr = np.clip(rgb, 0, 255).astype(np.uint8)[:, :, ::-1].copy()

    predicted = overlay_fn(bgr, prediction.cpu().numpy().astype(np.uint8))
    truth = overlay_fn(bgr, target.cpu().numpy().astype(np.uint8))

    cv2.putText(
        predicted,
        "prediction",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        truth,
        "ground truth",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), np.vstack([predicted, truth]), [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def _write_confusion(run: RunDir, metrics: SegMetrics) -> Path:
    """Render the confusion matrix as a PNG (section 10).

    Rows are normalised to sum to one, so the picture shows where each true class went rather
    than which classes are simply common. An unnormalised matrix on this dataset would be a
    single bright cell for background and nothing else legible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = metrics.confusion.astype(np.float64)
    totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(normalised, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics.class_names)))
    ax.set_yticks(range(len(metrics.class_names)))
    ax.set_xticklabels(metrics.class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(metrics.class_names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title(f"{run.name}\nrow-normalised confusion, mIoU {metrics.miou:.4f}", fontsize=10)

    for i in range(len(metrics.class_names)):
        for j in range(len(metrics.class_names)):
            value = normalised[i, j]
            if value > 0.005:
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value < 0.6 else "black",
                )

    fig.colorbar(image, ax=ax, fraction=0.046)
    path = run.eval_dir / "confusion.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
