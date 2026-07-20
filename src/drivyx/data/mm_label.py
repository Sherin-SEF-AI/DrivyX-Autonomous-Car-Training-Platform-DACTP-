"""Waypoint dataset construction and QC (CLAUDE.md section 8).

Produces "one parquet per route with columns frame_path, t, speed_mps, wp_x[5], wp_y[5],
route, segment", plus the QC artifacts the GUI renders: a smoothed-vs-raw track plot per
segment, waypoint arrows over 20 random frames, and a histogram of lateral offsets.

Val split is the last 15 percent of each route by time, never random (section 8). Sampling
randomly would leak: consecutive frames at 15 Hz are near-duplicates, and the 2.5 s horizon
means a training frame's targets literally are a val frame's positions.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drivyx.data.mm_sync import RouteData, sync_route
from drivyx.data.waypoints import (
    HORIZONS_S,
    NUM_WAYPOINTS,
    latlon_to_enu,
    split_segments,
    waypoints_for_segment,
)
from drivyx.paths import Paths

logger = logging.getLogger(__name__)

#: Section 8: "Val split = last 15 percent of each route by time".
VAL_FRACTION = 0.15

#: Section 8: "waypoint arrows drawn over 20 random frames".
QC_OVERLAY_COUNT = 20

#: OBD speed is reported in km/h in the IDD tables. This is asserted rather than assumed:
#: see _detect_speed_units.
KMH_TO_MPS = 1.0 / 3.6


def _detect_speed_units(obd_speed: np.ndarray, sg_speed: np.ndarray) -> tuple[float, str]:
    """Decide whether OBD speed is m/s or km/h, by comparing it against GPS-derived speed.

    The manifest records a column name, not a unit, and the unit is not stated anywhere in
    the dataset. Guessing wrong scales every fused speed by 3.6, which would silently corrupt
    the ctrl-net's speed input while still looking like a plausible number.

    GPS-derived speed is independent and unambiguously m/s (metres from ENU, seconds from
    timestamps), so the ratio between the two over frames where both are valid identifies the
    unit. The ratio clusters near 1.0 for m/s and near 3.6 for km/h; anything else means the
    columns are mismapped and this aborts rather than picking the closer of two wrong answers.
    """
    valid = np.isfinite(obd_speed) & (sg_speed > 2.0) & (obd_speed > 0)
    if valid.sum() < 20:
        raise ValueError(
            "cannot determine OBD speed units: fewer than 20 frames have both a moving GPS "
            "speed and an OBD reading. Check the manifest's OBD mapping and clock offset."
        )

    ratio = float(np.median(obd_speed[valid] / sg_speed[valid]))
    if 0.7 <= ratio <= 1.4:
        return 1.0, "m/s"
    if 2.9 <= ratio <= 4.3:
        return KMH_TO_MPS, "km/h"
    raise ValueError(
        f"OBD speed does not match GPS-derived speed: the median ratio is {ratio:.2f}, which "
        "is neither ~1 (m/s) nor ~3.6 (km/h). The OBD column mapping or the clock offset is "
        "probably wrong. Refusing to guess a unit."
    )


@dataclass
class RouteResult:
    """One route's built dataset and its QC counters."""

    route: str
    frames: pd.DataFrame
    segments: int
    total_fixes: int
    drop_reasons: dict[str, int]
    obd_matched: int
    speed_units: str
    val_cutoff_t: float


def _frame_path(images_dir: Path, frame_index: int, example: str) -> str:
    """Reconstruct an image filename from its index.

    The width and extension are taken from an example file in the directory rather than
    assumed, because nothing in the dataset documents them: the observed convention is
    zero-padded to 7 digits with a .jpg extension, but that is an observation, not a contract.
    """
    stem = Path(example).stem
    suffix = Path(example).suffix
    return str(images_dir / f"{frame_index:0{len(stem)}d}{suffix}")


def _left_image_dir(route_block: dict, route: str) -> tuple[Path, str]:
    """The left camera directory for a route, and an example filename from it.

    Section 9.2 feeds the ctrl net segmentation logits from the forward camera. The left half
    of the stereo pair is chosen because the seg model is trained on IDD Segmentation's
    leftImg8bit; feeding it right-camera frames at inference would be a domain shift for no
    reason.
    """
    candidates = route_block.get("image_dirs", [])
    if not candidates:
        raise ValueError(f"route {route}: the manifest lists no image directories.")

    left = [d for d in candidates if d.get("side") == "left"]
    chosen = left[0] if left else None
    if chosen is None:
        if len(candidates) == 1:
            chosen = candidates[0]
            logger.warning(
                "route %s: no directory is identifiable as the left camera; using the only "
                "one present (%s)",
                route,
                chosen["name"],
            )
        else:
            raise ValueError(
                f"route {route}: cannot tell which of {[d['name'] for d in candidates]} is "
                "the left camera. mm-inventory could not resolve a side, so the manifest "
                "needs a human decision."
            )
    examples = chosen.get("examples") or []
    if not examples:
        raise ValueError(f"route {route}: image directory {chosen['path']} appears empty.")
    return Path(chosen["path"]), examples[0]


def build_route(manifest: dict, route: str, paths: Paths) -> RouteResult:
    """Run section 8's pipeline over one route and return its frame table."""
    route_block = manifest["routes"][route]
    data: RouteData = sync_route(manifest, route)
    images_dir, example = _left_image_dir(route_block, route)

    segments = split_segments(route, data.t, data.lat, data.lon, data.frame)
    if not segments:
        raise ValueError(f"route {route}: GPS fixes produced no usable segments.")

    # The unit check needs GPS-derived speed, which is only available per segment. The first
    # sufficiently long segment is enough to identify a unit that is constant for the route.
    scale, units = _resolve_speed_units(segments, data)

    rows: list[dict[str, Any]] = []
    totals = {"slow": 0, "target_past_segment_end": 0, "lateral_over_limit": 0}

    for segment in segments:
        mask = np.isin(data.frame, segment.frame)
        obd_slice = data.obd_speed[mask] * scale

        result = waypoints_for_segment(segment, obd_speed=obd_slice)
        for key in totals:
            totals[key] += result.drop_reasons[key]

        # Raw ENU, for the QC track plot's smoothed-vs-raw comparison (section 8). The
        # smoothed track comes back from waypoints_for_segment; the raw fixes are recomputed
        # here against the same origin so the two are directly comparable.
        raw_x, raw_y = latlon_to_enu(
            segment.lat, segment.lon, float(segment.lat[0]), float(segment.lon[0])
        )

        for i in result.keep:
            row: dict[str, Any] = {
                "frame_path": _frame_path(images_dir, int(segment.frame[i]), example),
                "t": float(segment.t[i]),
                "speed_mps": float(result.speed[i]),
                "route": route,
                "segment": segment.index,
                "enu_x": float(result.x_s[i]),
                "enu_y": float(result.y_s[i]),
                "enu_x_raw": float(raw_x[i]),
                "enu_y_raw": float(raw_y[i]),
                "heading_rad": float(result.heading[i]),
            }
            for k in range(NUM_WAYPOINTS):
                row[f"wp_x{k}"] = float(result.wp_x[i, k])
                row[f"wp_y{k}"] = float(result.wp_y[i, k])
            rows.append(row)

    if not rows:
        raise ValueError(
            f"route {route}: every frame was filtered out. Drop reasons: {totals}. "
            "Check the confirmed clock offset and column mappings."
        )

    frames = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)

    # Section 8: the last 15% by time, per route, never random.
    cutoff = float(frames["t"].quantile(1.0 - VAL_FRACTION))
    frames["split"] = np.where(frames["t"] >= cutoff, "val", "train")

    logger.info(
        "route %s: %d frames from %d segments (%d train / %d val), OBD units %s",
        route,
        len(frames),
        len(segments),
        int((frames["split"] == "train").sum()),
        int((frames["split"] == "val").sum()),
        units,
    )
    return RouteResult(
        route=route,
        frames=frames,
        segments=len(segments),
        total_fixes=len(data),
        drop_reasons=totals,
        obd_matched=data.obd_matched,
        speed_units=units,
        val_cutoff_t=cutoff,
    )


def _resolve_speed_units(segments: list, data: RouteData) -> tuple[float, str]:
    """Identify the OBD speed unit once per route."""
    if not np.isfinite(data.obd_speed).any():
        return 1.0, "none (no OBD)"

    from drivyx.data.waypoints import latlon_to_enu, smooth_enu

    for segment in sorted(segments, key=len, reverse=True):
        if len(segment) < 60:
            continue
        x_e, y_n = latlon_to_enu(
            segment.lat, segment.lon, float(segment.lat[0]), float(segment.lon[0])
        )
        _xs, _ys, vx, vy = smooth_enu(x_e, y_n, segment.t)
        sg_speed = np.hypot(vx, vy)
        mask = np.isin(data.frame, segment.frame)
        try:
            return _detect_speed_units(data.obd_speed[mask], sg_speed)
        except ValueError:
            continue
    raise ValueError(
        "could not identify the OBD speed unit on any segment of this route. Check the "
        "confirmed clock offset: a wrong offset pairs OBD samples with unrelated frames."
    )


def write_qc(result: RouteResult, paths: Paths, *, seed: int = 0xD8) -> dict[str, str]:
    """Write the QC artifacts section 8 requires.

    Matplotlib is imported here and forced to the Agg backend: this runs inside a CLI
    subprocess that must work headless over SSH (section 3), and any interactive backend
    would try to reach a display.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qc_dir = paths.waypoints / "qc" / result.route
    qc_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    frames = result.frames

    # --- track plot: the driven path, per segment (section 8's "smoothed-vs-raw track") ---
    fig, ax = plt.subplots(figsize=(8, 8))
    for segment_id, group in frames.groupby("segment"):
        ax.plot(
            group["enu_x"],
            group["enu_y"],
            linewidth=1.2,
            label=f"segment {segment_id} ({len(group)} frames)",
        )
    ax.scatter(
        frames["enu_x_raw"],
        frames["enu_y_raw"],
        s=1.5,
        color="#c4453c",
        alpha=0.35,
        label="raw fixes",
        zorder=0,
    )
    ax.set_title(f"route {result.route}: smoothed track vs raw GPS")
    ax.set_xlabel("east (m from segment start)")
    ax.set_ylabel("north (m from segment start)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    path = qc_dir / "track.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    written["track"] = str(path)

    # --- lateral offset by time, per horizon ---
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, horizon in enumerate(HORIZONS_S):
        ax.plot(frames["t"], frames[f"wp_y{k}"], linewidth=0.6, label=f"{horizon}s")
    ax.set_title(f"route {result.route}: waypoint lateral offsets by horizon")
    ax.set_xlabel("time (s since midnight)")
    ax.set_ylabel("lateral offset (m, left positive)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = qc_dir / "lateral_by_time.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    written["lateral_by_time"] = str(path)

    # --- histogram of lateral offsets ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(frames["wp_y4"], bins=80, color="#4772b3")
    ax.set_title(f"route {result.route}: lateral offset at 2.5 s (n={len(frames)})")
    ax.set_xlabel("lateral offset (m, left positive)")
    ax.set_ylabel("frames")
    ax.grid(alpha=0.3)
    path = qc_dir / "lateral_histogram.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    written["lateral_histogram"] = str(path)

    # --- speed profile ---
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(frames["t"], frames["speed_mps"], linewidth=0.7, color="#6fa85c")
    ax.axvline(result.val_cutoff_t, color="#d9a23c", linestyle="--", label="val split")
    ax.set_title(f"route {result.route}: fused speed ({result.speed_units} OBD)")
    ax.set_xlabel("time (s since midnight)")
    ax.set_ylabel("speed (m/s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = qc_dir / "speed_profile.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    written["speed_profile"] = str(path)

    # --- waypoint arrows over 20 random frames ---
    rng = random.Random(seed)
    sample = rng.sample(range(len(frames)), min(QC_OVERLAY_COUNT, len(frames)))
    overlay = _draw_waypoint_overlays(frames.iloc[sorted(sample)], qc_dir)
    if overlay:
        written["waypoint_overlays"] = overlay

    return written


def _draw_waypoint_overlays(sample: pd.DataFrame, qc_dir: Path) -> str | None:
    """Draw the waypoint chain over each sampled frame, as a contact sheet.

    The projection is the fixed pinhole assumption documented in eval/viz.py, kept identical
    here so the QC gallery and the eval overlays cannot disagree about where a waypoint lands.
    """
    import cv2

    from drivyx.eval.viz import draw_waypoints

    tiles: list[np.ndarray] = []
    for _, row in sample.iterrows():
        image = cv2.imread(row["frame_path"], cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("QC overlay: cannot read %s", row["frame_path"])
            continue
        wp_x = np.array([row[f"wp_x{k}"] for k in range(NUM_WAYPOINTS)])
        wp_y = np.array([row[f"wp_y{k}"] for k in range(NUM_WAYPOINTS)])
        drawn = draw_waypoints(image, wp_x, wp_y, color=(80, 255, 80))
        cv2.putText(
            drawn,
            f"{row['speed_mps']:.1f} m/s",
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(cv2.resize(drawn, (480, 270)))

    if not tiles:
        return None

    columns = 4
    rows_needed = (len(tiles) + columns - 1) // columns
    blank = np.zeros_like(tiles[0])
    tiles += [blank] * (rows_needed * columns - len(tiles))
    sheet = np.vstack(
        [np.hstack(tiles[r * columns : (r + 1) * columns]) for r in range(rows_needed)]
    )
    path = qc_dir / "waypoint_overlays.jpg"
    cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return str(path)


def mm_label(paths: Paths, *, route: str | None = None) -> dict[str, Any]:
    """Build the waypoint dataset for every route (or one), with QC (section 8)."""
    from drivyx.data.mm_inventory import read_manifest, unconfirmed_fields

    manifest = read_manifest(paths)

    pending = unconfirmed_fields(manifest)
    if pending:
        raise ValueError(
            "mm-label refuses to run while required mappings are unconfirmed (CLAUDE.md "
            "section 8). Confirm these in the LABEL workspace's FieldMapTable:\n  - "
            + "\n  - ".join(pending)
        )

    routes = [route] if route else sorted(manifest.get("routes", {}))
    if not routes:
        raise ValueError("the manifest lists no routes. Run 'drivyx mm-inventory'.")

    paths.waypoints.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"routes": {}, "waypoints_dir": str(paths.waypoints)}
    total = 0

    for name in routes:
        result = build_route(manifest, name, paths)
        parquet = paths.waypoints / f"{name}.parquet"
        result.frames.to_parquet(parquet, index=False)
        qc = write_qc(result, paths)
        total += len(result.frames)

        report["routes"][name] = {
            "parquet": str(parquet),
            "frames": len(result.frames),
            "train": int((result.frames["split"] == "train").sum()),
            "val": int((result.frames["split"] == "val").sum()),
            "segments": result.segments,
            "gps_fixes": result.total_fixes,
            "retained_pct": round(100.0 * len(result.frames) / result.total_fixes, 2),
            "obd_matched": result.obd_matched,
            "obd_matched_pct": round(100.0 * result.obd_matched / result.total_fixes, 2),
            "obd_speed_units": result.speed_units,
            "drop_reasons": result.drop_reasons,
            "val_cutoff_t": result.val_cutoff_t,
            "qc": qc,
        }
        logger.info("route %s -> %s (%d frames)", name, parquet, len(result.frames))

    report["total_frames"] = total
    (paths.waypoints / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def read_route(paths: Paths, route: str) -> pd.DataFrame:
    """Read one route's parquet, pointing at mm-label when absent."""
    path = paths.waypoints / f"{route}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"No waypoint dataset at {path}. Run 'drivyx mm-label' first.")
    return pd.read_parquet(path)
