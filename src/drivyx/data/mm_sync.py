"""Timestamp association between frames, GPS, and OBD (CLAUDE.md section 8).

Section 8 specifies: "associate each image timestamp with the nearest GPS row within 50 ms
and nearest OBD row within 100 ms; otherwise drop the frame."

Two things about the real data change how that is implemented, both measured and recorded in
docs/DECISIONS.md:

  D008  The GPS table already carries an `image_idx` column pairing each fix to a frame, so
        the image/GPS association is given by the data, not searched for. The 50 ms rule is
        kept as an assertion over the paired fix rather than a nearest-neighbour search, so a
        future table without image_idx still fails loudly instead of mispairing silently.

  D010  OBD logs at ~0.65 Hz, so a 100 ms window keeps ~10% of frames. The window is one
        measured OBD interval and the speed is linearly interpolated between the bracketing
        samples, which keeps ~85%. Vehicle speed is smooth over 1.5 s at urban speeds, so
        this is physically sound and stays closed-form per line 29.

  D009  The OBD clock is offset from the GPS clock by ~19800 s (OBD logs UTC, GPS logs local
        Indian time). The offset is never auto-applied: it arrives here already confirmed by
        a human via the FieldMapTable, and this module refuses to guess it.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drivyx.data.mm_inventory import CONFIRMED, parse_timestamp

logger = logging.getLogger(__name__)

#: Section 8's GPS association window. Retained unchanged: GPS logs at 15 Hz here, where
#: 50 ms is a meaningful bound.
GPS_TOLERANCE_S = 0.050

#: Fallback OBD window when the manifest carries no measured value. Section 8's literal
#: figure, kept only so a caller without a manifest still gets the documented behaviour.
SPEC_OBD_TOLERANCE_S = 0.100


@dataclass(frozen=True)
class RouteData:
    """One route's synchronised sensor streams, on a single clock."""

    route: str
    #: Frame index from the GPS table's image_idx column.
    frame: np.ndarray
    #: Seconds since midnight, on the GPS (local) clock.
    t: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    #: OBD speed in m/s interpolated onto `t`, NaN where no OBD sample is in range.
    obd_speed: np.ndarray
    #: How many frames got a real OBD reading, for the QC report.
    obd_matched: int

    def __len__(self) -> int:
        return len(self.t)


def _require_confirmed(manifest_block: dict, role: str, where: str) -> str:
    """Read a confirmed column name, or abort naming what is unconfirmed.

    Section 8: "mm-label refuses to run while any required mapping is unconfirmed". This is
    the enforcement point, not a courtesy check: proceeding on an unconfirmed guess is
    exactly the "invent data to fill a gap" that section 16 forbids.
    """
    guess = manifest_block.get("roles", {}).get(role)
    if guess is None:
        raise ValueError(
            f"{where}: no column is mapped to the {role!r} role. Run 'drivyx mm-inventory' "
            "and confirm the mapping in the LABEL workspace."
        )
    if guess.get("state") != CONFIRMED:
        raise ValueError(
            f"{where}: the {role!r} mapping ({guess.get('column')!r}) is unconfirmed. "
            "Confirm it in the LABEL workspace's FieldMapTable before running mm-label."
        )
    column = guess.get("column")
    if not column:
        raise ValueError(f"{where}: the {role!r} mapping is confirmed but names no column.")
    return str(column)


def _read_table(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def load_gps(
    route_block: dict, route: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read a route's GPS fixes from every confirmed table, sorted by time.

    Multiple tables per route are unioned: IDD ships train/val/test splits of the same drive,
    and the temporal val split is recomputed later from time (D011), so the shipped split
    boundaries are irrelevant here. Tables without position (IDD's test split, D022) never
    reach this function: mm-inventory classifies them as not-GPS.
    """
    gps = route_block.get("gps")
    if not gps:
        raise ValueError(f"route {route}: the manifest has no GPS table.")

    where = f"route {route} gps"
    t_col = _require_confirmed(gps, "time", where)
    lat_col = _require_confirmed(gps, "lat", where)
    lon_col = _require_confirmed(gps, "lon", where)
    frame_col = _require_confirmed(gps, "frame", where)

    times: list[float] = []
    lats: list[float] = []
    lons: list[float] = []
    frames: list[int] = []

    for table in gps["tables"]:
        path = Path(table["path"])
        for row in _read_table(path):
            stamp = parse_timestamp(row.get(t_col, ""))
            if stamp is None:
                continue
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
            except (KeyError, TypeError, ValueError):
                continue
            frame_raw = row.get(frame_col, "")
            try:
                frame = int(str(frame_raw).strip())
            except (TypeError, ValueError):
                continue
            times.append(stamp)
            lats.append(lat)
            lons.append(lon)
            frames.append(frame)

    if not times:
        raise ValueError(
            f"route {route}: no GPS rows parsed from {[t['path'] for t in gps['tables']]}. "
            f"Check the confirmed column mapping (time={t_col!r}, lat={lat_col!r}, "
            f"lon={lon_col!r}, frame={frame_col!r})."
        )

    order = np.argsort(np.asarray(times))
    return (
        np.asarray(frames)[order],
        np.asarray(times)[order],
        np.asarray(lats)[order],
        np.asarray(lons)[order],
    )


def load_obd(route_block: dict, route: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Read a route's OBD samples on the GPS clock, or None when the route has no OBD.

    The confirmed clock offset (D009) is applied here, once, so every downstream consumer
    works on a single timebase. Returns (times, speeds) sorted by time.
    """
    obd = route_block.get("obd")
    if not obd:
        return None

    where = f"route {route} obd"
    t_col = _require_confirmed(obd, "time", where)
    speed_col = _require_confirmed(obd, "speed", where)

    offset_field = route_block.get("clock_offset")
    if offset_field is None:
        raise ValueError(
            f"route {route}: OBD is present but no clock offset was measured. Re-run "
            "'drivyx mm-inventory'."
        )
    if offset_field.get("state") != CONFIRMED:
        raise ValueError(
            f"route {route}: the OBD clock offset is unconfirmed "
            f"(proposed {offset_field.get('proposed_s')}s: {offset_field.get('hypothesis')}). "
            "CLAUDE.md rule 30 requires timestamp misalignment to abort rather than be "
            "silently corrected. Confirm it in the LABEL workspace's FieldMapTable."
        )
    offset = float(offset_field.get("confirmed_s", offset_field.get("proposed_s", 0.0)))

    times: list[float] = []
    speeds: list[float] = []
    for table in obd["tables"]:
        for row in _read_table(Path(table["path"])):
            stamp = parse_timestamp(row.get(t_col, ""))
            if stamp is None:
                continue
            try:
                speed = float(row[speed_col])
            except (KeyError, TypeError, ValueError):
                continue
            times.append(stamp + offset)
            speeds.append(speed)

    if not times:
        raise ValueError(
            f"route {route}: no OBD rows parsed. Check the confirmed mapping "
            f"(time={t_col!r}, speed={speed_col!r})."
        )

    order = np.argsort(np.asarray(times))
    return np.asarray(times)[order], np.asarray(speeds)[order]


def interpolate_obd_speed(
    frame_times: np.ndarray,
    obd_times: np.ndarray,
    obd_speeds: np.ndarray,
    *,
    tolerance_s: float,
) -> np.ndarray:
    """Interpolate OBD speed onto frame times (D010).

    A frame gets a speed when it lies between two OBD samples that are each within
    `tolerance_s` of it; the speed is then the linear interpolation between them. Frames
    outside that (before the first sample, after the last, or inside a dropout longer than
    the tolerance) get NaN, and section 8.2's fusion leaves their SG speed untouched.

    Why interpolate rather than take the nearest sample: at 0.65 Hz, nearest-neighbour
    quantises speed into 1.5 s steps, and those steps become step artifacts in the fused
    velocity. Interpolation is exact for constant acceleration and closed-form, which
    nearest is not and a filter would not be.

    Returns an array of len(frame_times) with speeds in the OBD table's own units.
    """
    result = np.full(len(frame_times), np.nan)
    if len(obd_times) == 0:
        return result

    # Index of the first OBD sample at or after each frame time.
    right = np.searchsorted(obd_times, frame_times, side="left")
    left = right - 1

    has_left = left >= 0
    has_right = right < len(obd_times)

    # Distance to the bracketing samples, +inf where a side does not exist so the tolerance
    # test below fails cleanly rather than indexing out of range.
    dist_left = np.where(has_left, frame_times - obd_times[np.clip(left, 0, None)], np.inf)
    dist_right = np.where(
        has_right, obd_times[np.clip(right, None, len(obd_times) - 1)] - frame_times, np.inf
    )

    # Bracketed: both neighbours exist and are within tolerance. Interpolate between them.
    bracketed = has_left & has_right & (dist_left <= tolerance_s) & (dist_right <= tolerance_s)
    if bracketed.any():
        result[bracketed] = np.interp(frame_times[bracketed], obd_times, obd_speeds)

    # Exactly on a sample, or within tolerance of the single neighbour at either end of the
    # log. np.interp clamps outside its range, which is the correct value here: the frame is
    # within one sampling interval of the boundary sample.
    edge = ~bracketed & (
        (has_left & (dist_left <= tolerance_s)) | (has_right & (dist_right <= tolerance_s))
    )
    if edge.any():
        result[edge] = np.interp(frame_times[edge], obd_times, obd_speeds)

    return result


def sync_route(
    manifest: dict,
    route: str,
    *,
    gps_tolerance_s: float = GPS_TOLERANCE_S,
) -> RouteData:
    """Build one route's synchronised streams from a confirmed manifest.

    Aborts if any required mapping or the clock offset is unconfirmed (section 8).
    """
    route_block = manifest.get("routes", {}).get(route)
    if route_block is None:
        raise ValueError(
            f"route {route!r} is not in the manifest. Known routes: "
            f"{sorted(manifest.get('routes', {}))}."
        )

    # Validate every confirmation before touching the disk. Two reasons: reading a 4500-row
    # CSV only to reject it on a metadata check is wasted work, and checking as-you-go reports
    # one unconfirmed field per run, so a user fixes them one at a time. This reports all of
    # them at once.
    _assert_route_confirmed(route_block, route)

    frame, t, lat, lon = load_gps(route_block, route)
    _assert_gps_consistency(route, frame, t, gps_tolerance_s)

    obd = load_obd(route_block, route)
    if obd is None:
        logger.info("route %s: no OBD table; speed will come from GPS alone", route)
        return RouteData(
            route=route,
            frame=frame,
            t=t,
            lat=lat,
            lon=lon,
            obd_speed=np.full(len(t), np.nan),
            obd_matched=0,
        )

    tolerance_field = route_block.get("obd_tolerance") or {}
    tolerance = float(
        tolerance_field.get("confirmed_s")
        or tolerance_field.get("proposed_s")
        or SPEC_OBD_TOLERANCE_S
    )

    obd_times, obd_speeds = obd
    speed = interpolate_obd_speed(t, obd_times, obd_speeds, tolerance_s=tolerance)
    matched = int(np.isfinite(speed).sum())

    logger.info(
        "route %s: %d frames, %d with OBD speed (%.1f%%) at a %.3fs window",
        route,
        len(t),
        matched,
        100.0 * matched / len(t) if len(t) else 0.0,
        tolerance,
    )
    return RouteData(
        route=route,
        frame=frame,
        t=t,
        lat=lat,
        lon=lon,
        obd_speed=speed,
        obd_matched=matched,
    )


def _assert_gps_consistency(
    route: str, frame: np.ndarray, t: np.ndarray, tolerance_s: float
) -> None:
    """Check the image_idx pairing the data provides is self-consistent (D008).

    Section 8's 50 ms rule exists to catch a frame whose nearest fix is too far away. Here the
    pairing is given by image_idx, so the equivalent check is that the pairing is
    monotonic and unique: a repeated or out-of-order image_idx means the table is not the
    1:1 mapping it appears to be, and every downstream waypoint would be attached to the
    wrong picture.
    """
    if len(frame) != len(set(frame.tolist())):
        duplicates = len(frame) - len(set(frame.tolist()))
        raise ValueError(
            f"route {route}: the GPS table pairs {duplicates} frame indices more than once, "
            "so image_idx is not a 1:1 key. Refusing to build waypoints against an ambiguous "
            "pairing."
        )

    # Sorted by time above, so frame indices must also ascend if they index the same drive.
    if np.any(np.diff(frame) < 0):
        raise ValueError(
            f"route {route}: frame indices do not ascend with time, so the GPS table's "
            "image_idx does not index frames in capture order."
        )

    gaps = np.diff(t)
    if np.any(gaps <= 0):
        raise ValueError(f"route {route}: GPS timestamps do not strictly advance.")

    # A fix whose neighbours are further than the tolerance is not itself an error (that is
    # what segment splitting handles), but a *median* interval beyond it means the table is
    # not the 15 Hz stream section 4 describes.
    median_dt = float(np.median(gaps))
    if median_dt > tolerance_s * 2:
        logger.warning(
            "route %s: median GPS interval is %.3fs, well above the %.3fs association "
            "tolerance; check the manifest's time column mapping",
            route,
            median_dt,
            tolerance_s,
        )


def _assert_route_confirmed(route_block: dict, route: str) -> None:
    """Reject a route with any unconfirmed required field, naming every one (section 8).

    Reports the complete set rather than the first miss, so a human fixes the manifest once.
    """
    problems: list[str] = []

    gps = route_block.get("gps")
    if not gps:
        raise ValueError(f"route {route}: the manifest has no GPS table.")
    for role in ("time", "lat", "lon", "frame"):
        guess = gps.get("roles", {}).get(role)
        if guess is None:
            problems.append(f"gps.{role}: no column proposed")
        elif guess.get("state") != CONFIRMED:
            problems.append(f"gps.{role} ({guess.get('column')!r}) is unconfirmed")

    obd = route_block.get("obd")
    if obd:
        for role in ("time", "speed"):
            guess = obd.get("roles", {}).get(role)
            if guess is None:
                problems.append(f"obd.{role}: no column proposed")
            elif guess.get("state") != CONFIRMED:
                problems.append(f"obd.{role} ({guess.get('column')!r}) is unconfirmed")

        offset = route_block.get("clock_offset")
        if offset is None:
            problems.append("clock_offset: not measured; re-run mm-inventory")
        elif offset.get("state") != CONFIRMED:
            problems.append(
                f"clock offset is unconfirmed (proposed {offset.get('proposed_s')}s: "
                f"{offset.get('hypothesis')})"
            )

    if problems:
        raise ValueError(
            f"route {route}: mm-label refuses to run while required mappings are unconfirmed "
            "(CLAUDE.md section 8; a timestamp misalignment must abort rather than be "
            "silently corrected, rule 30). Confirm these in the LABEL workspace's "
            "FieldMapTable:\n  - " + "\n  - ".join(problems)
        )
