"""Overlay rendering shared by eval and the GUI preview (CLAUDE.md sections 5, 10).

Section 10: "overlay renderer draws predicted (accent color) vs ground-truth (white) waypoint
chains projected into the image with a fixed pinhole assumption documented in viz.py".

This module is that documentation. It is deliberately the single implementation: mm-label's
QC gallery, eval-ctrl's overlays, and infer-preview all call it, so a projection change can
never make the QC images and the eval images disagree about where a waypoint lands.

# The pinhole assumption

IDD Multimodal ships no camera intrinsics or extrinsics. Section 10 anticipates this and asks
for a *fixed assumption* rather than a calibration, because the overlay's job is qualitative:
a human looks at it and answers "do the arrows follow the road". So the numbers below are
chosen to be reasonable for a forward dashcam and are stated openly as assumptions.

Camera frame convention (right-handed, camera looking down +Z):
    X_cam right, Y_cam down, Z_cam forward

Ego frame convention (section 8.4):
    x forward, y left, z up

So a waypoint (x, y) on the ground at height 0 maps to the camera as:
    X_cam = -y            (ego left is camera right-negative)
    Y_cam = +CAMERA_HEIGHT_M   (the ground is below the camera, and Y_cam points down)
    Z_cam = +x            (ego forward is camera forward)

The pinhole projection is then:
    u = fx * X_cam / Z_cam + cx
    v = fy * Y_cam / Z_cam + cy

Focal length is derived from an assumed horizontal field of view rather than hardcoded in
pixels, so the projection stays correct across the 720p and 1080p frames IDD mixes:
    fx = (W / 2) / tan(HFOV / 2)

A point at Z_cam <= MIN_DEPTH_M is behind or level with the camera and is not drawn: the
projection would place it at infinity or, worse, mirror it into the image as a plausible
looking point that is actually behind the vehicle.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Assumed horizontal field of view of the IDD forward camera. A typical automotive/dashcam
#: lens is 60 to 90 degrees; 70 sits in the middle and matches the apparent geometry of the
#: IDD frames. Stated as an assumption: no intrinsics ship with the dataset.
HFOV_DEG = 70.0

#: Assumed camera height above the road, metres. A car roof/windscreen mount is 1.2 to 1.6 m.
CAMERA_HEIGHT_M = 1.4

#: Points closer than this in Z are behind or beside the camera and are not projected.
MIN_DEPTH_M = 0.5

#: Section 10: predicted chains are drawn in the accent colour, ground truth in white. BGR,
#: because OpenCV.
COLOR_PREDICTED = (179, 114, 71)  # ACCENT #4772b3 in BGR
COLOR_GROUND_TRUTH = (255, 255, 255)


def focal_length_px(width: int, hfov_deg: float = HFOV_DEG) -> float:
    """fx from the assumed field of view.

    Derivation: a pinhole camera of focal length f images a point at angle theta from the
    optical axis at u = f*tan(theta). The image half-width W/2 corresponds to the half-FOV,
    so W/2 = f*tan(HFOV/2), hence f = (W/2)/tan(HFOV/2).
    """
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def project_ego_to_image(
    wp_x: np.ndarray,
    wp_y: np.ndarray,
    width: int,
    height: int,
    *,
    hfov_deg: float = HFOV_DEG,
    camera_height_m: float = CAMERA_HEIGHT_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project ego-frame ground points into pixel coordinates.

    Args:
        wp_x: forward distances in metres (ego x).
        wp_y: left offsets in metres (ego y).

    Returns:
        (u, v, visible) where u and v are pixel coordinates and `visible` marks the points
        that are in front of the camera. Points behind it are returned with their computed
        coordinates but marked invisible, so callers can drop them without re-deriving why.

    Square pixels are assumed (fy = fx), and the principal point is assumed to be the image
    centre. Both are conventional and unverifiable without intrinsics.
    """
    fx = focal_length_px(width, hfov_deg)
    fy = fx
    cx, cy = width / 2.0, height / 2.0

    z_cam = np.asarray(wp_x, dtype=float)
    x_cam = -np.asarray(wp_y, dtype=float)
    y_cam = np.full_like(z_cam, camera_height_m)

    visible = z_cam > MIN_DEPTH_M
    safe_z = np.where(visible, z_cam, np.nan)

    u = fx * x_cam / safe_z + cx
    v = fy * y_cam / safe_z + cy
    return u, v, visible


def draw_waypoints(
    image: np.ndarray,
    wp_x: np.ndarray,
    wp_y: np.ndarray,
    *,
    color: tuple[int, int, int] = COLOR_GROUND_TRUTH,
    thickness: int = 2,
    label: str | None = None,
) -> np.ndarray:
    """Draw one waypoint chain over a copy of `image`.

    The chain is drawn from the camera outward: a line from the vehicle's own position
    through each waypoint in order, with a circle at each and an arrowhead at the last. The
    circles shrink with distance, which gives the eye the same depth cue the road does and
    makes a chain that drifts off the road obvious.
    """
    canvas = image.copy()
    height, width = canvas.shape[:2]
    u, v, visible = project_ego_to_image(wp_x, wp_y, width, height)

    points = [
        (int(round(float(uu))), int(round(float(vv))))
        for uu, vv, ok in zip(u, v, visible)
        if ok and np.isfinite(uu) and np.isfinite(vv)
    ]
    if not points:
        return canvas

    # Start the chain at the bottom centre, which is where the vehicle itself is.
    chain = [(width // 2, height - 1), *points]
    for a, b in zip(chain, chain[1:]):
        cv2.line(canvas, a, b, color, thickness, cv2.LINE_AA)

    for i, point in enumerate(points):
        radius = max(2, 7 - i)
        cv2.circle(canvas, point, radius, color, -1, cv2.LINE_AA)

    if len(chain) >= 2:
        cv2.arrowedLine(canvas, chain[-2], chain[-1], color, thickness, cv2.LINE_AA, tipLength=0.4)

    if label:
        cv2.putText(
            canvas, label, (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
        )
    return canvas


def draw_waypoint_comparison(
    image: np.ndarray,
    gt_x: np.ndarray,
    gt_y: np.ndarray,
    pred_x: np.ndarray,
    pred_y: np.ndarray,
) -> np.ndarray:
    """Section 10: predicted in the accent colour over ground truth in white."""
    canvas = draw_waypoints(image, gt_x, gt_y, color=COLOR_GROUND_TRUTH, label="ground truth")
    return draw_waypoints(canvas, pred_x, pred_y, color=COLOR_PREDICTED, label=None)


def colorize_mask(mask: np.ndarray, colors: list[tuple[int, int, int]] | None = None) -> np.ndarray:
    """Map a collapsed class mask to BGR using the LUT's own colours.

    Sharing the LUT colours means a class looks the same in the DATA workspace's swatch
    table, the histogram bars, and every overlay.
    """
    from drivyx.data.lut import GROUP_COLORS, IGNORE_ID

    palette = colors if colors is not None else list(GROUP_COLORS)
    table = np.zeros((256, 3), dtype=np.uint8)
    for index, rgb in enumerate(palette):
        table[index] = (rgb[2], rgb[1], rgb[0])  # RGB -> BGR
    table[IGNORE_ID] = (0, 0, 0)
    return table[mask]


def overlay_mask(image: np.ndarray, mask: np.ndarray, *, alpha: float = 0.45) -> np.ndarray:
    """Blend a colourised mask over an image.

    Ignore pixels are left unblended so the eye is not drawn to regions the loss never saw.
    """
    from drivyx.data.lut import IGNORE_ID

    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    colored = colorize_mask(mask)
    blended = cv2.addWeighted(image, 1.0 - alpha, colored, alpha, 0.0)
    keep = (mask == IGNORE_ID)[..., None]
    return np.where(keep, image, blended)
