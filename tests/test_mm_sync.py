"""Timestamp association and manifest gating (CLAUDE.md sections 8, 13).

Section 13 requires "mm-sync tolerance windows with crafted timestamps (cpu)". Crafted rather
than real: the point is to pin the window's edges exactly, which real 15 Hz data cannot do.
"""

from __future__ import annotations

import numpy as np
import pytest

from drivyx.data.mm_inventory import CONFIRMED, UNCONFIRMED, parse_timestamp
from drivyx.data.mm_sync import interpolate_obd_speed, sync_route

# --- timestamp parsing ------------------------------------------------------------------


def test_parses_idd_multimodal_format() -> None:
    """The observed IDD format: HH-MM-SS-microseconds."""
    assert parse_timestamp("09-00-31-289685") == pytest.approx(9 * 3600 + 0 * 60 + 31 + 0.289685)


def test_parses_obd_format_with_trailing_zeros() -> None:
    """OBD stamps like 03-31-13-000000 must not be read as 0 microseconds of a huge number."""
    assert parse_timestamp("03-31-13-000000") == pytest.approx(3 * 3600 + 31 * 60 + 13)


def test_fractional_field_scales_by_its_own_width() -> None:
    """The fraction is scaled by its digit count, not assumed to be 6 digits."""
    assert parse_timestamp("00-00-01-5") == pytest.approx(1.5)
    assert parse_timestamp("00-00-01-50") == pytest.approx(1.5)
    assert parse_timestamp("00-00-01-500000") == pytest.approx(1.5)


def test_parses_clock_and_numeric_forms() -> None:
    assert parse_timestamp("01:02:03") == pytest.approx(3723.0)
    assert parse_timestamp("01:02:03.5") == pytest.approx(3723.5)
    assert parse_timestamp("1234.5") == pytest.approx(1234.5)


def test_rejects_unparseable() -> None:
    assert parse_timestamp("") is None
    assert parse_timestamp("not a time") is None
    assert parse_timestamp("   ") is None


# --- OBD interpolation windows (section 13's named test) --------------------------------


def test_frame_exactly_on_a_sample() -> None:
    obd_t = np.array([10.0, 20.0])
    obd_v = np.array([5.0, 15.0])

    out = interpolate_obd_speed(np.array([10.0]), obd_t, obd_v, tolerance_s=1.0)

    assert out[0] == pytest.approx(5.0)


def test_frame_between_samples_is_interpolated() -> None:
    """The D010 behaviour: linear between bracketing samples, not nearest."""
    obd_t = np.array([10.0, 20.0])
    obd_v = np.array([0.0, 10.0])

    out = interpolate_obd_speed(np.array([15.0]), obd_t, obd_v, tolerance_s=10.0)

    assert out[0] == pytest.approx(5.0), "must interpolate, not snap to a neighbour"


def test_interpolation_is_not_nearest_neighbour() -> None:
    """A frame 60% of the way between samples must get 60% of the way between speeds.

    Nearest-neighbour would return the closer sample's value and quantise speed into 1.5 s
    steps, which is exactly what D010 rejects.
    """
    obd_t = np.array([0.0, 10.0])
    obd_v = np.array([0.0, 100.0])

    out = interpolate_obd_speed(np.array([6.0]), obd_t, obd_v, tolerance_s=10.0)

    assert out[0] == pytest.approx(60.0)
    assert out[0] != pytest.approx(100.0), "this is nearest-neighbour behaviour"


def test_frame_outside_the_window_is_nan() -> None:
    """A dropout longer than the tolerance must yield NaN, not a stale value."""
    obd_t = np.array([0.0, 100.0])
    obd_v = np.array([5.0, 15.0])

    out = interpolate_obd_speed(np.array([50.0]), obd_t, obd_v, tolerance_s=1.0)

    assert np.isnan(out[0]), "a frame mid-dropout must have no OBD speed"


def test_window_edges_are_inclusive() -> None:
    """Exactly at the tolerance is inside; just beyond is outside."""
    obd_t = np.array([0.0])
    obd_v = np.array([7.0])

    inside = interpolate_obd_speed(np.array([1.0]), obd_t, obd_v, tolerance_s=1.0)
    outside = interpolate_obd_speed(np.array([1.001]), obd_t, obd_v, tolerance_s=1.0)

    assert inside[0] == pytest.approx(7.0)
    assert np.isnan(outside[0])


def test_before_the_first_sample_within_tolerance() -> None:
    """A frame just before the log starts is within one interval of its first sample."""
    obd_t = np.array([10.0, 20.0])
    obd_v = np.array([5.0, 15.0])

    out = interpolate_obd_speed(np.array([9.5]), obd_t, obd_v, tolerance_s=1.0)

    assert out[0] == pytest.approx(5.0), "np.interp clamps, which is right at the boundary"


def test_before_the_first_sample_beyond_tolerance() -> None:
    obd_t = np.array([10.0])
    obd_v = np.array([5.0])

    out = interpolate_obd_speed(np.array([5.0]), obd_t, obd_v, tolerance_s=1.0)

    assert np.isnan(out[0])


def test_after_the_last_sample() -> None:
    obd_t = np.array([10.0, 20.0])
    obd_v = np.array([5.0, 15.0])

    inside = interpolate_obd_speed(np.array([20.5]), obd_t, obd_v, tolerance_s=1.0)
    outside = interpolate_obd_speed(np.array([25.0]), obd_t, obd_v, tolerance_s=1.0)

    assert inside[0] == pytest.approx(15.0)
    assert np.isnan(outside[0])


def test_empty_obd_log() -> None:
    out = interpolate_obd_speed(np.array([1.0, 2.0]), np.array([]), np.array([]), tolerance_s=1.0)
    assert np.isnan(out).all()


def test_realistic_rate_retention() -> None:
    """The D010 measurement, reproduced in miniature.

    OBD at 0.65 Hz against frames at 15 Hz: a 100 ms window (the spec's literal figure) keeps
    a small fraction, while one sampling interval keeps nearly all of them. This is the
    arithmetic the deviation rests on, pinned so it cannot drift.
    """
    obd_dt = 1.536
    obd_t = np.arange(0.0, 60.0, obd_dt)
    obd_v = np.full(len(obd_t), 10.0)
    frames = np.arange(0.0, 58.0, 1.0 / 15.0)

    tight = interpolate_obd_speed(frames, obd_t, obd_v, tolerance_s=0.100)
    wide = interpolate_obd_speed(frames, obd_t, obd_v, tolerance_s=obd_dt)

    tight_pct = 100.0 * np.isfinite(tight).sum() / len(frames)
    wide_pct = 100.0 * np.isfinite(wide).sum() / len(frames)

    assert tight_pct < 20.0, f"the spec's 100ms window kept {tight_pct:.0f}%, expected ~13%"
    assert wide_pct > 95.0, f"one sampling interval kept {wide_pct:.0f}%, expected nearly all"


def test_interpolation_tracks_acceleration_linearly() -> None:
    """Speed rising linearly must be recovered exactly: interpolation is exact for a ramp."""
    obd_t = np.array([0.0, 1.5, 3.0])
    obd_v = np.array([0.0, 15.0, 30.0])
    frames = np.array([0.75, 2.25])

    out = interpolate_obd_speed(frames, obd_t, obd_v, tolerance_s=1.5)

    assert out[0] == pytest.approx(7.5)
    assert out[1] == pytest.approx(22.5)


# --- manifest gating (section 8) --------------------------------------------------------


def _manifest(*, gps_state: str = CONFIRMED, offset_state: str = CONFIRMED) -> dict:
    """A minimal manifest with crafted confirmation states."""
    return {
        "routes": {
            "d0": {
                "image_dirs": [
                    {
                        "name": "leftCamImgs",
                        "side": "left",
                        "path": "/x",
                        "examples": ["0000000.jpg"],
                    }
                ],
                "gps": {
                    "tables": [
                        {
                            "path": "/x/gps.csv",
                            "rows": 2,
                            "columns": ["timestamp", "image_idx", "latitude", "longitude"],
                        }
                    ],
                    "roles": {
                        "time": {
                            "role": "time",
                            "column": "timestamp",
                            "confidence": "exact",
                            "state": gps_state,
                        },
                        "lat": {
                            "role": "lat",
                            "column": "latitude",
                            "confidence": "exact",
                            "state": gps_state,
                        },
                        "lon": {
                            "role": "lon",
                            "column": "longitude",
                            "confidence": "exact",
                            "state": gps_state,
                        },
                        "frame": {
                            "role": "frame",
                            "column": "image_idx",
                            "confidence": "exact",
                            "state": gps_state,
                        },
                    },
                },
                "obd": {
                    "tables": [
                        {"path": "/x/obd.csv", "rows": 2, "columns": ["timestamp", "speed"]}
                    ],
                    "roles": {
                        "time": {
                            "role": "time",
                            "column": "timestamp",
                            "confidence": "exact",
                            "state": CONFIRMED,
                        },
                        "speed": {
                            "role": "speed",
                            "column": "speed",
                            "confidence": "exact",
                            "state": CONFIRMED,
                        },
                    },
                },
                "clock_offset": {"proposed_s": 19800, "state": offset_state, "hypothesis": "IST"},
                "obd_tolerance": {"proposed_s": 1.536, "state": CONFIRMED},
            }
        }
    }


def test_sync_refuses_unconfirmed_gps_mapping() -> None:
    """Section 8: mm-label refuses while any required mapping is unconfirmed."""
    with pytest.raises(ValueError, match="unconfirmed"):
        sync_route(_manifest(gps_state=UNCONFIRMED), "d0")


def test_sync_refuses_unconfirmed_clock_offset() -> None:
    """Rule 30: a timestamp misalignment must abort, never be silently corrected."""
    with pytest.raises(ValueError, match="clock offset is unconfirmed"):
        sync_route(_manifest(offset_state=UNCONFIRMED), "d0")


def test_sync_names_the_route_it_cannot_find() -> None:
    with pytest.raises(ValueError, match="not in the manifest"):
        sync_route(_manifest(), "nope")


def test_unconfirmed_fields_lists_every_pending_row() -> None:
    from drivyx.data.mm_inventory import unconfirmed_fields

    pending = unconfirmed_fields(_manifest(gps_state=UNCONFIRMED, offset_state=UNCONFIRMED))

    assert any("gps.time" in p for p in pending)
    assert any("clock_offset" in p for p in pending)


def test_unconfirmed_fields_empty_when_all_confirmed() -> None:
    from drivyx.data.mm_inventory import unconfirmed_fields

    assert unconfirmed_fields(_manifest()) == []


# --- device: the real manifest -----------------------------------------------------------


@pytest.mark.device
def test_real_manifest_is_fully_confirmed_and_syncs() -> None:
    from drivyx.data.mm_inventory import read_manifest, unconfirmed_fields
    from drivyx.paths import get_paths

    paths = get_paths()
    try:
        manifest = read_manifest(paths)
    except FileNotFoundError:
        pytest.skip("no manifest; run 'drivyx mm-inventory' first")

    if unconfirmed_fields(manifest):
        pytest.skip("manifest not confirmed; run 'drivyx mm-confirm --yes'")

    data = sync_route(manifest, "d0")

    assert len(data) > 1000
    assert np.all(np.diff(data.t) > 0), "GPS timestamps must strictly advance"
    assert data.obd_matched > 0
    # D010's measured claim on the real route.
    matched_pct = 100.0 * data.obd_matched / len(data)
    assert 60.0 < matched_pct < 95.0, f"OBD match is {matched_pct:.1f}%, expected ~77%"
