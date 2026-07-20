"""Waypoint math (CLAUDE.md sections 8, 13).

Section 13 requires "waypoint synthetic-circle accuracy (cpu)" and "ENU round-trip sanity
(cpu)". Section 8 sets the bar: "synthetic circular drive at constant speed; assert recovered
waypoints match the analytic circle within 5 cm".

A circle is the right fixture because every quantity has a closed form (position, speed,
heading, and the ego-frame offset to any future point), so the test compares against
mathematics rather than against a previous run of the same code.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from drivyx.data.waypoints import (
    HORIZONS_S,
    MAX_GPS_GAP_S,
    METERS_PER_DEG_LAT,
    METERS_PER_DEG_LON_EQUATOR,
    NUM_WAYPOINTS,
    Segment,
    compute_heading,
    fuse_speed,
    interpolate_positions,
    latlon_to_enu,
    smooth_enu,
    split_segments,
    to_ego_frame,
    waypoints_for_segment,
)

# Hyderabad, where the IDD multimodal routes were recorded.
LAT0 = 17.4962518306
LON0 = 78.4156015873

#: Section 8's accuracy bar.
TOLERANCE_M = 0.05


# --- ENU (section 8.1) ------------------------------------------------------------------


def test_enu_origin_is_zero() -> None:
    x, y = latlon_to_enu(np.array([LAT0]), np.array([LON0]), LAT0, LON0)
    assert abs(float(x[0])) < 1e-9
    assert abs(float(y[0])) < 1e-9


def test_enu_axes_point_east_and_north() -> None:
    """x_e must grow with longitude, y_n with latitude."""
    x, y = latlon_to_enu(
        np.array([LON0 + 0.001, LAT0]) * 0 + np.array([LAT0, LAT0 + 0.001]),
        np.array([LON0 + 0.001, LON0]),
        LAT0,
        LON0,
    )
    assert x[0] > 0, "east offset must be positive for increased longitude"
    assert y[1] > 0, "north offset must be positive for increased latitude"


def test_enu_scale_matches_the_documented_constants() -> None:
    """One degree of latitude is METERS_PER_DEG_LAT; longitude shrinks by cos(lat0)."""
    x, y = latlon_to_enu(np.array([LAT0 + 1.0]), np.array([LON0 + 1.0]), LAT0, LON0)

    assert float(y[0]) == pytest.approx(METERS_PER_DEG_LAT, rel=1e-9)
    expected_x = METERS_PER_DEG_LON_EQUATOR * math.cos(math.radians(LAT0))
    assert float(x[0]) == pytest.approx(expected_x, rel=1e-9)


def test_enu_round_trip_sanity() -> None:
    """Section 13: "ENU round-trip sanity".

    Invert the closed form and confirm a known metric offset comes back to the same lat/lon.
    """
    east, north = 120.0, -75.0
    lat = LAT0 + north / METERS_PER_DEG_LAT
    lon = LON0 + east / (METERS_PER_DEG_LON_EQUATOR * math.cos(math.radians(LAT0)))

    x, y = latlon_to_enu(np.array([lat]), np.array([lon]), LAT0, LON0)

    assert float(x[0]) == pytest.approx(east, abs=1e-6)
    assert float(y[0]) == pytest.approx(north, abs=1e-6)


def test_enu_distance_matches_haversine_locally() -> None:
    """The flat-earth approximation must agree with the real geodesic at city scale.

    This is the assumption section 8.1 makes ("City-scale segments make this approximation
    sufficient"); at 300 m it should hold to well under a centimetre.
    """
    east, north = 200.0, 200.0
    lat = LAT0 + north / METERS_PER_DEG_LAT
    lon = LON0 + east / (METERS_PER_DEG_LON_EQUATOR * math.cos(math.radians(LAT0)))

    x, y = latlon_to_enu(np.array([lat]), np.array([lon]), LAT0, LON0)
    flat = math.hypot(float(x[0]), float(y[0]))

    # Haversine on the WGS84 mean radius.
    radius = 6371008.8
    p1, p2 = math.radians(LAT0), math.radians(lat)
    dphi = p2 - p1
    dlam = math.radians(lon - LON0)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    great_circle = 2 * radius * math.asin(math.sqrt(a))

    # Within 0.5%: the residual is the latitude-constant approximation documented in the
    # module, not an error in the transform.
    assert flat == pytest.approx(great_circle, rel=0.005)


# --- ego frame (section 8.4) ------------------------------------------------------------


def test_ego_frame_heading_east() -> None:
    """Facing east: due east is straight ahead, due north is to the left."""
    x, y = to_ego_frame(np.array([10.0, 0.0]), np.array([0.0, 10.0]), 0.0)

    assert x[0] == pytest.approx(10.0) and y[0] == pytest.approx(0.0)
    assert x[1] == pytest.approx(0.0) and y[1] == pytest.approx(10.0)


def test_ego_frame_heading_north() -> None:
    """Facing north: due north is ahead, due east is to the right (negative y)."""
    x, y = to_ego_frame(np.array([0.0, 10.0]), np.array([10.0, 0.0]), math.pi / 2)

    assert x[0] == pytest.approx(10.0) and y[0] == pytest.approx(0.0, abs=1e-9)
    assert x[1] == pytest.approx(0.0, abs=1e-9) and y[1] == pytest.approx(-10.0)


def test_ego_frame_preserves_distance() -> None:
    """A rotation cannot change a length."""
    dx, dy = np.array([3.0, -7.0, 12.0]), np.array([4.0, 24.0, -5.0])
    for heading in (0.0, 0.7, math.pi, -2.1):
        x, y = to_ego_frame(dx, dy, heading)
        assert np.allclose(np.hypot(x, y), np.hypot(dx, dy))


# --- heading (section 8.3) --------------------------------------------------------------


def test_heading_from_velocity() -> None:
    vx = np.array([1.0, 0.0, -1.0])
    vy = np.array([0.0, 1.0, 0.0])
    speed = np.array([5.0, 5.0, 5.0])

    heading = compute_heading(vx, vy, speed)

    assert heading[0] == pytest.approx(0.0)
    assert heading[1] == pytest.approx(math.pi / 2)
    assert abs(heading[2]) == pytest.approx(math.pi)


def test_heading_holds_last_valid_below_threshold() -> None:
    """Section 8.3: "below that, hold the last valid heading"."""
    vx = np.array([1.0, 0.01, 0.01, 1.0])
    vy = np.array([0.0, -0.02, 0.03, 0.0])
    speed = np.array([5.0, 0.1, 0.1, 5.0])

    heading = compute_heading(vx, vy, speed)

    assert heading[1] == pytest.approx(heading[0]), "slow frame must hold the previous heading"
    assert heading[2] == pytest.approx(heading[0])


def test_heading_backfills_before_first_valid() -> None:
    """A segment that starts stationary has no earlier heading to hold."""
    vx = np.array([0.01, 0.01, 1.0])
    vy = np.array([0.0, 0.0, 0.0])
    speed = np.array([0.1, 0.1, 5.0])

    heading = compute_heading(vx, vy, speed)

    assert np.all(np.isfinite(heading))
    assert heading[0] == pytest.approx(heading[2])


def test_heading_all_slow_is_finite() -> None:
    heading = compute_heading(np.zeros(4), np.zeros(4), np.zeros(4))
    assert np.all(np.isfinite(heading))


# --- speed fusion (section 8.2) ---------------------------------------------------------


def test_fuse_speed_averages_at_weight_half() -> None:
    sg = np.array([10.0, 20.0])
    obd = np.array([12.0, 18.0])

    fused = fuse_speed(sg, obd, weight=0.5)

    assert fused[0] == pytest.approx(11.0)
    assert fused[1] == pytest.approx(19.0)


def test_fuse_speed_without_obd_returns_sg() -> None:
    """Section 8.2 fuses "when both are valid"; with OBD at 0.65 Hz this is the common path."""
    sg = np.array([10.0, 20.0])
    assert np.array_equal(fuse_speed(sg, None), sg)


def test_fuse_speed_skips_invalid_obd_frames() -> None:
    sg = np.array([10.0, 20.0, 30.0])
    obd = np.array([12.0, np.nan, 34.0])

    fused = fuse_speed(sg, obd, weight=0.5)

    assert fused[0] == pytest.approx(11.0)
    assert fused[1] == pytest.approx(20.0), "a NaN OBD sample must leave the SG speed alone"
    assert fused[2] == pytest.approx(32.0)


# --- segments (section 8) ---------------------------------------------------------------


def test_split_segments_breaks_on_gap() -> None:
    """Section 8: "Segments break on GPS gaps > 0.34 s"."""
    t = np.array([0.0, 0.067, 0.134, 1.5, 1.567, 1.634])

    segments = split_segments("d0", t, np.zeros(6), np.zeros(6), np.arange(6))

    assert len(segments) == 2
    assert len(segments[0]) == 3
    assert len(segments[1]) == 3


def test_split_segments_tolerates_gap_at_threshold() -> None:
    t = np.array([0.0, MAX_GPS_GAP_S - 0.001, 2 * MAX_GPS_GAP_S - 0.002])
    assert len(split_segments("d0", t, np.zeros(3), np.zeros(3), np.arange(3))) == 1


def test_split_segments_drops_singletons() -> None:
    """A one-fix run carries no trajectory."""
    t = np.array([0.0, 5.0, 5.067, 5.134])

    segments = split_segments("d0", t, np.zeros(4), np.zeros(4), np.arange(4))

    assert len(segments) == 1
    assert len(segments[0]) == 3


def test_split_segments_empty() -> None:
    assert split_segments("d0", np.array([]), np.array([]), np.array([]), np.array([])) == []


# --- interpolation ----------------------------------------------------------------------


def test_interpolate_positions_linear() -> None:
    t = np.array([0.0, 1.0, 2.0])
    x = np.array([0.0, 10.0, 20.0])
    y = np.array([0.0, -5.0, -10.0])

    ix, iy = interpolate_positions(t, x, y, np.array([0.5, 1.5]))

    assert ix == pytest.approx([5.0, 15.0])
    assert iy == pytest.approx([-2.5, -7.5])


# --- the synthetic circle (section 8, section 13) ---------------------------------------


def _circular_drive(
    radius_m: float = 50.0,
    speed_mps: float = 10.0,
    rate_hz: float = 15.0,
    duration_s: float = 40.0,
) -> tuple[Segment, float, float]:
    """A constant-speed anticlockwise circle, expressed as GPS fixes.

    The vehicle starts at the circle's easternmost point heading north. Angular rate is
    omega = v / r, so the closed form of every derived quantity is known:
        position(t) = (r cos(omega t), r sin(omega t)) in ENU metres from the centre
        speed       = v, constant
        heading(t)  = omega t + pi/2, i.e. always tangent to the circle
    """
    n = int(duration_s * rate_hz)
    t = np.arange(n) / rate_hz
    omega = speed_mps / radius_m

    east = radius_m * np.cos(omega * t)
    north = radius_m * np.sin(omega * t)

    lat = LAT0 + north / METERS_PER_DEG_LAT
    lon = LON0 + east / (METERS_PER_DEG_LON_EQUATOR * math.cos(math.radians(LAT0)))

    return (
        Segment(route="synthetic", index=0, t=t, lat=lat, lon=lon, frame=np.arange(n)),
        radius_m,
        speed_mps,
    )


def test_synthetic_circle_recovers_speed() -> None:
    segment, _radius, speed = _circular_drive()

    result = waypoints_for_segment(segment)

    # Savitzky-Golay's polynomial fit is biased at the window edges, so the interior is what
    # the accuracy claim is about; the edge frames are dropped by the horizon filter anyway.
    interior = result.speed[20:-20]
    assert np.allclose(interior, speed, atol=0.02), (
        f"recovered speed {interior.min():.4f}..{interior.max():.4f} != {speed}"
    )


def test_synthetic_circle_waypoints_match_analytic_within_5cm() -> None:
    """Section 8: "assert recovered waypoints match the analytic circle within 5 cm".

    For a circle traversed at constant speed, the ego-frame offset to the position `h`
    seconds ahead has a closed form. With angular step a = omega*h, the chord from the
    current point to the future point has length 2*r*sin(a/2), and it sits at angle a/2 from
    the current tangent. So in the ego frame (x forward, y left):

        x = 2*r*sin(a/2) * cos(a/2) = r*sin(a)
        y = 2*r*sin(a/2) * sin(a/2) = r*(1 - cos(a))

    y is positive because the drive is anticlockwise, curving to the left.
    """
    segment, radius, speed = _circular_drive()
    omega = speed / radius

    result = waypoints_for_segment(segment)

    assert len(result.keep) > 0, "every frame was filtered out"

    expected_x = np.array([radius * math.sin(omega * h) for h in HORIZONS_S])
    expected_y = np.array([radius * (1.0 - math.cos(omega * h)) for h in HORIZONS_S])

    interior = [i for i in result.keep if 20 <= i < len(segment) - 20]
    assert interior, "no interior frames survived filtering"

    errors = []
    for i in interior:
        errors.append(np.hypot(result.wp_x[i] - expected_x, result.wp_y[i] - expected_y))
    worst = float(np.max(errors))

    assert worst <= TOLERANCE_M, (
        f"worst waypoint error {worst * 100:.2f} cm exceeds the 5 cm bar. "
        f"expected x={expected_x.round(3)} y={expected_y.round(3)}"
    )


def test_synthetic_circle_curves_left() -> None:
    """An anticlockwise drive must produce positive (left) lateral offsets that grow."""
    segment, _radius, _speed = _circular_drive()

    result = waypoints_for_segment(segment)
    i = int(result.keep[len(result.keep) // 2])

    assert np.all(result.wp_y[i] > 0), "anticlockwise drive must curve left"
    assert np.all(np.diff(result.wp_y[i]) > 0), "lateral offset must grow with the horizon"
    assert np.all(np.diff(result.wp_x[i]) > 0), "forward distance must grow with the horizon"


def test_straight_drive_has_no_lateral_offset() -> None:
    """A straight line is the degenerate case the circle test cannot check."""
    n, rate, speed = 400, 15.0, 12.0
    t = np.arange(n) / rate
    east = speed * t
    north = np.zeros(n)
    lat = LAT0 + north / METERS_PER_DEG_LAT
    lon = LON0 + east / (METERS_PER_DEG_LON_EQUATOR * math.cos(math.radians(LAT0)))
    segment = Segment(route="s", index=0, t=t, lat=lat, lon=lon, frame=np.arange(n))

    result = waypoints_for_segment(segment)
    i = int(result.keep[len(result.keep) // 2])

    assert np.allclose(result.wp_y[i], 0.0, atol=0.01), "a straight drive must not curve"
    expected_x = np.array([speed * h for h in HORIZONS_S])
    assert np.allclose(result.wp_x[i], expected_x, atol=TOLERANCE_M)


def test_waypoint_shape_is_five_by_two() -> None:
    segment, _r, _s = _circular_drive()
    result = waypoints_for_segment(segment)

    assert result.wp_x.shape == (len(segment), NUM_WAYPOINTS)
    assert result.wp_y.shape == (len(segment), NUM_WAYPOINTS)
    assert NUM_WAYPOINTS == 5


# --- filters (section 8.5) --------------------------------------------------------------


def test_slow_frames_are_dropped() -> None:
    """Section 8.5: "drop frames with speed < 1.0 m/s"."""
    segment, _r, _s = _circular_drive(speed_mps=0.4)

    result = waypoints_for_segment(segment)

    assert len(result.keep) == 0
    assert result.drop_reasons["slow"] > 0


def test_frames_whose_horizon_passes_the_end_are_dropped() -> None:
    """np.interp clamps rather than extrapolating, which would invent a stop."""
    segment, _r, _s = _circular_drive(duration_s=10.0)

    result = waypoints_for_segment(segment)

    assert result.drop_reasons["target_past_segment_end"] > 0
    # The last 2.5 s of frames cannot have a 2.5 s target.
    assert int(result.keep.max()) < len(segment) - 1


def test_turnaround_lateral_filter() -> None:
    """Section 8.5: drop when |y| of the 2.5 s point exceeds 25 m.

    Sizing this fixture takes care. On a circle the ego-frame lateral offset is
    r*(1 - cos(omega*h)), which is bounded by the diameter 2r, so any circle with r < 12.5 m
    can never trip a 25 m filter no matter how fast it is driven. The offset peaks when the
    vehicle sweeps half a turn within the horizon, i.e. omega*2.5 = pi.

    So: r = 20 m and omega = pi/2.5 give a 2.5 s lateral of ~40 m, comfortably over the
    limit. That implies v = omega*r = 25 m/s around a 20 m radius, about 3 g of lateral
    acceleration, which no car does. That is the point: such a reading is a GPS artifact, and
    rejecting it is exactly what section 8.5 asks for.
    """
    radius, omega = 20.0, math.pi / 2.5
    segment, _r, _s = _circular_drive(radius_m=radius, speed_mps=omega * radius, duration_s=20.0)

    result = waypoints_for_segment(segment)

    expected_peak = radius * (1.0 - math.cos(omega * HORIZONS_S[-1]))
    assert expected_peak > 25.0, f"fixture is mis-sized: peak lateral is only {expected_peak:.1f} m"
    assert result.drop_reasons["lateral_over_limit"] > 0, (
        "a physically impossible turn must be rejected as a turnaround artifact"
    )


def test_drop_reasons_are_attributable() -> None:
    segment, _r, _s = _circular_drive()
    result = waypoints_for_segment(segment)

    assert set(result.drop_reasons) == {"slow", "target_past_segment_end", "lateral_over_limit"}
    assert all(v >= 0 for v in result.drop_reasons.values())


def test_short_segment_does_not_crash_smoothing() -> None:
    """A segment shorter than the SG window must shrink the window, not fail."""
    t = np.arange(7) / 15.0
    x = np.linspace(0, 5, 7)
    y = np.zeros(7)

    xs, ys, vx, vy = smooth_enu(x, y, t)

    assert (
        len(xs) == 7
        and np.all(np.isfinite(vx))
        and np.all(np.isfinite(ys))
        and np.all(np.isfinite(vy))
    )


def test_smoothing_rejects_a_segment_too_short_to_fit() -> None:
    with pytest.raises(ValueError, match="too few"):
        smooth_enu(np.zeros(3), np.zeros(3), np.arange(3) / 15.0)
