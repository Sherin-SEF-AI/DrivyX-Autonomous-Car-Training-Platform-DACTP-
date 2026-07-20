"""Control evaluation (CLAUDE.md section 10).

"eval-ctrl: ADE/FDE/lateral on the temporal val split; overlay renderer draws predicted
(accent color) vs ground-truth (white) waypoint chains projected into the image with a fixed
pinhole assumption documented in viz.py."

The metrics are reported per horizon as well as aggregated. Aggregate ADE hides the shape of
the error: a model that is accurate at 0.5 s and wrong at 2.5 s scores the same as one that is
uniformly mediocre, and those are very different models to deploy.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import torch

from drivyx.data.waypoints import HORIZONS_S
from drivyx.jobs.run_dir import RunContext, RunDir
from drivyx.paths import Paths
from drivyx.torch_setup import AUTOCAST_DTYPE, require_cuda

logger = logging.getLogger(__name__)

#: Section 10 does not fix an overlay count for ctrl; 20 matches the QC gallery in section 8.
NUM_OVERLAYS = 20


@torch.no_grad()
def evaluate_ctrl(
    paths: Paths,
    run: RunDir,
    ctx: RunContext,
    *,
    checkpoint: str = "best",
    overlays: int = NUM_OVERLAYS,
) -> dict[str, Any]:
    """Evaluate a ctrl run on the temporal val split (section 10)."""

    from drivyx.models.ctrlnet import CtrlNet
    from drivyx.train.ctrl_trainer import CtrlDataset, cache_for

    require_cuda()
    device = torch.device("cuda")

    ckpt_path = run.best_ckpt if checkpoint == "best" else run.last_ckpt
    if not ckpt_path.is_file():
        fallback = run.last_ckpt if checkpoint == "best" else run.best_ckpt
        if not fallback.is_file():
            raise FileNotFoundError(f"no checkpoint in {run.ckpt_dir}")
        ckpt_path = fallback

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    seg_run = state["seg_run"]
    cache = cache_for(paths, seg_run)
    if not cache.exists():
        raise FileNotFoundError(
            f"the logit cache for seg run {seg_run} is missing at {cache.directory}. "
            "Re-run 'drivyx train-ctrl' to rebuild it."
        )

    manifest = json.loads(cache.manifest.read_text())
    model = CtrlNet(manifest["num_classes"])
    model.load_state_dict(state["model"])
    model = model.to(device).eval()

    dataset = CtrlDataset(cache, val=True)
    if not len(dataset):
        raise ValueError("the control val split is empty")
    logger.info("evaluating %s on %d val frames", ckpt_path.name, len(dataset))

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    speeds: list[float] = []

    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
    for logits, speed, target in loader:
        if ctx.interrupted:
            break
        logits = logits.to(device, non_blocking=True)
        speed_dev = speed.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=AUTOCAST_DTYPE):
            pred = model(logits, speed_dev)
        predictions.append(pred.float().cpu().numpy())
        targets.append(target.numpy())
        speeds.extend(speed.tolist())
        ctx.events.heartbeat()

    pred = np.concatenate(predictions)
    truth = np.concatenate(targets)

    metrics = compute_ctrl_metrics(pred, truth)
    overlay_paths = _write_overlays(paths, run, cache, pred, truth, overlays, ctx)

    payload = {
        "run": run.name,
        "checkpoint": ckpt_path.name,
        "epoch": state.get("epoch"),
        "seg_run": seg_run,
        "val_frames": len(pred),
        **metrics,
        "overlays": overlay_paths,
    }
    out = run.eval_dir / "ctrl_metrics.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")

    logger.info(
        "ADE %.3f m | FDE %.3f m | lateral@1s %.3f m over %d frames -> %s",
        metrics["ade"],
        metrics["fde"],
        metrics["lateral_1s"],
        len(pred),
        out,
    )
    for horizon, value in zip(HORIZONS_S, metrics["ade_per_horizon"]):
        logger.info("  displacement error at %.1fs: %.3f m", horizon, value)

    return payload


def compute_ctrl_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """ADE, FDE, lateral error, and the per-horizon breakdown, all in metres.

    pred and truth are (N, 5, 2) in the ego frame.

    A baseline is included deliberately. On this dataset 78% of frames are within 0.5 m of
    straight (docs/DECISIONS.md D022), so a model that always predicts "continue straight at
    the current heading" already scores well. Reporting the constant-zero-curvature baseline
    alongside the model is the only way to see whether the model learned anything beyond that.
    """
    error = np.linalg.norm(pred - truth, axis=-1)

    lateral = np.abs(pred[:, :, 1] - truth[:, :, 1])
    longitudinal = np.abs(pred[:, :, 0] - truth[:, :, 0])

    # Baseline: predict the true forward distance with zero lateral offset, i.e. perfect speed
    # and perfectly straight. Isolates how much of the score comes from curvature.
    straight = truth.copy()
    straight[:, :, 1] = 0.0
    baseline_error = np.linalg.norm(straight - truth, axis=-1)

    return {
        "ade": float(error.mean()),
        "fde": float(error[:, -1].mean()),
        "lateral_1s": float(lateral[:, 1].mean()),
        "ade_per_horizon": [float(v) for v in error.mean(axis=0)],
        "lateral_per_horizon": [float(v) for v in lateral.mean(axis=0)],
        "longitudinal_per_horizon": [float(v) for v in longitudinal.mean(axis=0)],
        "horizons_s": list(HORIZONS_S),
        "baseline_straight_ade": float(baseline_error.mean()),
        "baseline_straight_fde": float(baseline_error[:, -1].mean()),
        "beats_straight_baseline": bool(error.mean() < baseline_error.mean()),
    }


def _write_overlays(
    paths: Paths,
    run: RunDir,
    cache: Any,
    pred: np.ndarray,
    truth: np.ndarray,
    count: int,
    ctx: RunContext,
) -> list[str]:
    """Predicted (accent) over ground truth (white), per section 10."""
    import cv2

    from drivyx.eval.viz import draw_waypoint_comparison
    from drivyx.train.ctrl_trainer import load_waypoint_frames

    frames = load_waypoint_frames(paths)
    val_frames = frames[frames["split"] == "val"].reset_index(drop=True)
    if len(val_frames) != len(pred):
        logger.warning(
            "val frame table has %d rows but %d predictions; overlays may be misaligned",
            len(val_frames),
            len(pred),
        )

    overlay_dir = run.eval_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    limit = min(count, len(pred), len(val_frames))
    chosen = np.linspace(0, limit - 1, num=limit, dtype=int)
    written: list[str] = []

    for i in chosen:
        row = val_frames.iloc[int(i)]
        image = cv2.imread(str(row["frame_path"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        canvas = draw_waypoint_comparison(
            image, truth[i, :, 0], truth[i, :, 1], pred[i, :, 0], pred[i, :, 1]
        )
        error = float(np.linalg.norm(pred[i] - truth[i], axis=-1).mean())
        cv2.putText(
            canvas,
            f"{row['speed_mps']:.1f} m/s   ADE {error:.2f} m",
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        path = overlay_dir / f"ctrl_{int(i):05d}.jpg"
        cv2.imwrite(str(path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        written.append(str(path.relative_to(run.path)))
        ctx.events.image("val/waypoints", str(path.relative_to(run.path)))

    return written
