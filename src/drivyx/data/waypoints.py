"""Waypoint supervision from GPS and OBD (CLAUDE.md section 8).

Section 8 specifies the math exactly and asks for docstrings deriving each step. The five
steps below map one-to-one onto section 8's numbered list.

Everything here is closed-form or classical (Savitzky-Golay, plane geometry), per line 29's
deterministic-first rule. There is no fitting, no iteration, and no learned component: given
the same fixes, this produces the same waypoints forever.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

# --- section 8.1: ENU constants ---------------------------------------------------------
#
# Metres per degree, the local flat-earth approximation section 8 mandates ("City-scale
# segments make this approximation sufficient; no pyproj dependency").
#
# 111320 is the equatorial length of one degree of longitude: 2*pi*a/360 with a = 6378137 m
# (WGS84 semi-major axis) gives 111319.49. It is scaled by cos(lat0) because meridians
# converge toward the poles.
#
# 111132 is one degree of latitude near the equator. The true value varies from 110574 m at
# the equator to 111694 m at the poles because the earth is oblate; 111132 is the standard
# mid-latitude constant. At Hyderabad (17.5 N) the exact value is about 110752 m, so this
# constant carries roughly +0.34% scale error. Over a 300 m segment that is ~1 m of absolute
# position, but it is a *constant* scale factor, so it cancels almost entirely in the
# ego-frame differences that become the waypoints: a 25 m waypoint inherits ~8 cm of error.
METERS_PER_DEG_LON_EQUATOR = 111320.0
METERS_PER_DEG_LAT = 111132.0

# --- section 8.2: smoothing -------------------------------------------------------------
#: Savitzky-Golay window in samples. At 15 Hz, 15 samples is a 1.0 s window.
SG_WINDOW = 15
#: Polynomial order. Cubic tracks the curvature of a turn without ringing.
SG_POLYORDER = 3
#: Weight given to OBD speed when fusing it with the SG-derived speed.
OBD_FUSION_WEIGHT = 0.5

# --- section 8.3: heading ---------------------------------------------------------------
#: Below this speed the heading from velocity is dominated by GPS noise, so the last valid
#: heading is held instead.
HEADING_MIN_SPEED_MPS = 1.5

# --- section 8.4: targets ---------------------------------------------------------------
#: Prediction horizons in seconds. Section 1: "5 future ego-frame waypoints over 2.5 s".
HORIZONS_S = (0.5, 1.0, 1.5, 2.0, 2.5)
NUM_WAYPOINTS = len(HORIZONS_S)

# --- section 8.5: filters ---------------------------------------------------------------
#: Frames slower than this are dropped: a stationary vehicle has no meaningful trajectory and
#: its heading is undefined.
MIN_SPEED_MPS = 1.0
#: A 2.5 s lateral offset beyond this is a turnaround artifact, not a manoeuvre.
MAX_LATERAL_M = 25.0
#: Segments break on GPS gaps larger than this (section 8: "> 0.34 s"). At 15 Hz the nominal
#: interval is 66.7 ms, so 0.34 s is roughly five missed fixes.
MAX_GPS_GAP_S = 0.34


@dataclass(frozen=True)
class Segment:
    """A contiguous run of GPS fixes with no gap larger than MAX_GPS_GAP_S.

    Waypoints may never be interpolated across a gap: the vehicle's path between two fixes
    0.5 s apart is unknown, and a straight-line guess through a turn would teach the model a
    trajectory that never happened.
    """

    route: str
    index: int
    t: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    frame: np.ndarray

    def __len__(self) -> int:
        return len(self.t)


def latlon_to_enu(
    lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Section 8.1: local ENU metres around a segment's first fix.

        x_e = (lon - lon0) * 111320 * cos(lat0_rad)
        y_n = (lat - lat0) * 111132

    This is a tangent-plane approximation: it treats the earth as locally flat and linearises
    the mapping from degrees to metres at the origin. The error grows with the square of the
    distance from the origin, which is why the origin is each segment's own first fix rather
    than a route-wide datum: segments here span ~300 m, where the approximation is good to
    well under a centimetre.

    cos(lat0) is evaluated once at the origin rather than per sample. Over a 300 m segment
    latitude changes by ~0.003 degrees, and cos varies by ~1e-5 relative, which is far below
    GPS noise.

    Returns (east, north) in metres.
    """
    x_e = (lon - lon0) * METERS_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat0))
    y_n = (lat - lat0) * METERS_PER_DEG_LAT
    return x_e, y_n


def smooth_enu(
    x_e: np.ndarray,
    y_n: np.ndarray,
    t: np.ndarray,
    *,
    window: int = SG_WINDOW,
    polyorder: int = SG_POLYORDER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Section 8.2: Savitzky-Golay smoothing, and velocity by its first derivative.

    Savitzky-Golay fits a low-order polynomial over a sliding window by least squares and
    evaluates it (or its derivative) at the centre. It is the right tool here because it
    suppresses GPS jitter while preserving the peaks and curvature of a real trajectory,
    which a moving average would flatten. It is also closed-form, satisfying line 29.

    Taking velocity as the analytic derivative of the fitted polynomial, rather than
    differencing the smoothed positions, matters: differencing re-amplifies exactly the
    high-frequency noise the smoothing removed.

    The window must be odd and longer than polyorder. Short segments shrink the window to the
    largest valid odd length rather than failing, because dropping a whole segment for being
    a second long loses real supervision.

    Returns (x_smooth, y_smooth, vx, vy) with velocities in m/s.
    """
    n = len(t)
    if n < polyorder + 2:
        raise ValueError(
            f"segment has {n} samples, too few for a polyorder-{polyorder} fit "
            f"(need at least {polyorder + 2})"
        )

    effective = min(window, n if n % 2 == 1 else n - 1)
    if effective <= polyorder:
        effective = polyorder + 1 + (polyorder % 2 == 0)
    if effective % 2 == 0:
        effective -= 1
    effective = max(effective, polyorder + 2 - (polyorder % 2))
    if effective % 2 == 0:
        effective += 1

    # The sampling interval for the derivative. GPS here is near-uniform at 15 Hz; the median
    # resists the occasional jitter in the reported timestamps.
    deltas = np.diff(t)
    dt = float(np.median(deltas)) if len(deltas) else 1.0
    if dt <= 0:
        raise ValueError("GPS timestamps do not advance within the segment")

    x_s = savgol_filter(x_e, effective, polyorder)
    y_s = savgol_filter(y_n, effective, polyorder)
    vx = savgol_filter(x_e, effective, polyorder, deriv=1, delta=dt)
    vy = savgol_filter(y_n, effective, polyorder, deriv=1, delta=dt)
    return x_s, y_s, vx, vy


def fuse_speed(
    sg_speed: np.ndarray,
    obd_speed: np.ndarray | None,
    *,
    weight: float = OBD_FUSION_WEIGHT,
) -> np.ndarray:
    """Section 8.2: rescale the SG velocity magnitude toward OBD speed, weight 0.5.

        fused = (1 - w) * sg + w * obd,  where both are valid

    The two measure different things: SG speed is the derivative of GPS position (accurate
    over time, noisy instant to instant), while OBD speed is wheel-derived (smooth and
    directly measured, but biased by tyre wear and quantised to 1 km/h here). Averaging them
    at w=0.5 is section 8's prescription and needs no tuning.

    Frames where OBD is unavailable keep the SG speed unchanged, which section 8.2 allows
    explicitly ("when both are valid"). With OBD at 0.65 Hz this is the majority-of-frames
    path, not an edge case (docs/DECISIONS.md D010).
    """
    if obd_speed is None:
        return sg_speed
    valid = np.isfinite(obd_speed)
    fused = sg_speed.copy()
    fused[valid] = (1.0 - weight) * sg_speed[valid] + weight * obd_speed[valid]
    return fused


def compute_heading(
    vx: np.ndarray, vy: np.ndarray, speed: np.ndarray, *, min_speed: float = HEADING_MIN_SPEED_MPS
) -> np.ndarray:
    """Section 8.3: heading = atan2(v_n, v_e) where speed >= 1.5 m/s, else hold the last.

    Note the argument order: atan2(north, east) measures the angle anticlockwise from east,
    the standard mathematical convention on an ENU plane, so heading 0 points east and pi/2
    points north.

    Below the threshold the velocity vector is mostly GPS noise and its direction is
    meaningless: a stationary vehicle's apparent heading spins randomly. Holding the last
    valid heading keeps the ego frame stable through a stop. Frames before any valid heading
    exists (a segment that starts stationary) get the first valid heading in the segment,
    which is the only defensible choice: there is no earlier information, and leaving them NaN
    would drop frames the speed filter would drop anyway.
    """
    heading = np.arctan2(vy, vx)
    valid = speed >= min_speed

    if not valid.any():
        # Nothing in this segment ever moved. The speed filter will drop every frame, so the
        # value here only has to be finite.
        return np.zeros_like(heading)

    # Forward-fill the last valid heading across invalid runs.
    indices = np.where(valid, np.arange(len(valid)), -1)
    np.maximum.accumulate(indices, out=indices)
    # Frames before the first valid heading backfill from it.
    first_valid = int(np.argmax(valid))
    indices[indices < 0] = first_valid
    return heading[indices]


def interpolate_positions(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Positions at arbitrary times by linear interpolation between fixes (section 8.4).

    Linear rather than higher-order: at 15 Hz consecutive fixes are 67 ms apart, over which a
    vehicle at 15 m/s travels 1 m along a path whose curvature is negligible. A spline would
    add overshoot at the segment ends for no accuracy gain.

    Callers must ensure `targets` lie within a gap-free span; np.interp clamps outside the
    input range rather than extrapolating, which would silently invent positions.
    """
    return np.interp(targets, t, x), np.interp(targets, t, y)


def to_ego_frame(dx: np.ndarray, dy: np.ndarray, heading: float) -> tuple[np.ndarray, np.ndarray]:
    """Section 8.4: rotate an ENU offset into the ego frame. x forward, y left, metres.

    The ego frame has its x axis along the heading and its y axis 90 degrees to the left.
    Expressing a world offset (dx east, dy north) in that frame is a rotation by -heading:

        x_forward =  dx*cos(h) + dy*sin(h)
        y_left    = -dx*sin(h) + dy*cos(h)

    Sanity check: with h = 0 (heading east), a point due east (dx>0, dy=0) gives x_forward=dx
    and y_left=0, i.e. straight ahead. A point due north (dx=0, dy>0) gives x_forward=0 and
    y_left=dy: north is to the left of an east-facing vehicle, which is correct.
    """
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    x_forward = dx * cos_h + dy * sin_h
    y_left = -dx * sin_h + dy * cos_h
    return x_forward, y_left


def split_segments(
    route: str,
    t: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    frame: np.ndarray,
    *,
    max_gap_s: float = MAX_GPS_GAP_S,
) -> list[Segment]:
    """Section 8: "Segments break on GPS gaps > 0.34 s".

    A gap means the vehicle's path is unknown across it. Everything downstream (smoothing,
    velocity, waypoint interpolation) assumes continuity, so the break is enforced here once
    rather than guarded for repeatedly.

    Input must be sorted by time.
    """
    if len(t) == 0:
        return []

    gaps = np.diff(t)
    breaks = np.where(gaps > max_gap_s)[0] + 1
    bounds = [0, *breaks.tolist(), len(t)]

    segments: list[Segment] = []
    for index, (start, stop) in enumerate(zip(bounds, bounds[1:])):
        if stop - start < 2:
            continue
        segments.append(
            Segment(
                route=route,
                index=index,
                t=t[start:stop],
                lat=lat[start:stop],
                lon=lon[start:stop],
                frame=frame[start:stop],
            )
        )
    logger.debug(
        "route %s: %d fixes -> %d segments (%d gaps > %.2fs)",
        route,
        len(t),
        len(segments),
        len(breaks),
        max_gap_s,
    )
    return segments


@dataclass
class SegmentWaypoints:
    """Per-frame waypoint targets for one segment, before filtering."""

    segment: Segment
    #: Kept frame indices, relative to the segment.
    keep: np.ndarray
    speed: np.ndarray
    wp_x: np.ndarray  # (n, 5) forward metres
    wp_y: np.ndarray  # (n, 5) left metres
    x_s: np.ndarray
    y_s: np.ndarray
    heading: np.ndarray
    #: Why each dropped frame was dropped, for the QC report.
    drop_reasons: dict[str, int]


def waypoints_for_segment(
    segment: Segment,
    obd_speed: np.ndarray | None = None,
    *,
    horizons: tuple[float, ...] = HORIZONS_S,
    min_speed: float = MIN_SPEED_MPS,
    max_lateral: float = MAX_LATERAL_M,
) -> SegmentWaypoints:
    """Run section 8's five steps over one segment.

    Filters (section 8.5), applied in this order so the reasons are attributable:
      1. speed < 1.0 m/s
      2. any target beyond the segment's end (a target past a GPS gap)
      3. |y| of the 2.5 s point > 25 m (turnaround artifact)
    """
    lat0, lon0 = float(segment.lat[0]), float(segment.lon[0])
    x_e, y_n = latlon_to_enu(segment.lat, segment.lon, lat0, lon0)
    x_s, y_s, vx, vy = smooth_enu(x_e, y_n, segment.t)

    sg_speed = np.hypot(vx, vy)
    speed = fuse_speed(sg_speed, obd_speed)
    heading = compute_heading(vx, vy, speed, min_speed=HEADING_MIN_SPEED_MPS)

    t = segment.t
    n = len(t)
    horizon_array = np.asarray(horizons)

    # Targets for every frame at once: (n, 5) absolute times.
    target_times = t[:, None] + horizon_array[None, :]

    # A target beyond the segment's last fix would be clamped by np.interp to the final
    # position, silently inventing a stop. Those frames are dropped instead: the horizon
    # extends past a gap or past the end of the recording, and neither is observed data.
    within = target_times[:, -1] <= t[-1]

    flat_x, flat_y = interpolate_positions(t, x_s, y_s, target_times.reshape(-1))
    target_x = flat_x.reshape(n, len(horizons))
    target_y = flat_y.reshape(n, len(horizons))

    dx = target_x - x_s[:, None]
    dy = target_y - y_s[:, None]

    wp_x = np.empty_like(dx)
    wp_y = np.empty_like(dy)
    for i in range(n):
        wp_x[i], wp_y[i] = to_ego_frame(dx[i], dy[i], float(heading[i]))

    fast_enough = speed >= min_speed
    lateral_ok = np.abs(wp_y[:, -1]) <= max_lateral

    keep_mask = fast_enough & within & lateral_ok
    reasons = {
        "slow": int((~fast_enough).sum()),
        "target_past_segment_end": int((fast_enough & ~within).sum()),
        "lateral_over_limit": int((fast_enough & within & ~lateral_ok).sum()),
    }

    keep = np.where(keep_mask)[0]
    return SegmentWaypoints(
        segment=segment,
        keep=keep,
        speed=speed,
        wp_x=wp_x,
        wp_y=wp_y,
        x_s=x_s,
        y_s=y_s,
        heading=heading,
        drop_reasons=reasons,
    )
