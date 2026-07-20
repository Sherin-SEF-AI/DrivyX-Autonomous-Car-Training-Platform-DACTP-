"""Multimodal layout discovery (CLAUDE.md section 8).

Section 4 declares the multimodal layout "UNKNOWN at spec time: discovered by mm-inventory",
and the master prompt forbids hardcoding any multimodal path or column name outside the
manifest. So nothing here asserts a route name, a directory name, or a column name: every
one is classified from the bytes on disk by extension and header sniffing, and written to
`multimodal/mm_manifest.json` for a human to confirm in the FieldMapTable.

What the manifest carries, per section 8: routes, per-route image dirs (left/right if
distinguishable), the GPS table (path plus column mapping guesses for time/lat/lon), the OBD
table (path plus guesses for time/speed), and sample rates measured from the data.

Two fields exist that section 8 does not name, both forced by what the data turned out to be:

  clock_offset_s  The OBD and GPS clocks are offset by ~5.5 hours on every route (OBD logs
                  UTC, GPS logs local Indian time). Rule 30 says timestamp misalignment must
                  abort, never warn-and-continue, so the offset is measured and proposed but
                  never auto-applied: mm-label refuses while it is unconfirmed.
                  See docs/DECISIONS.md D009.
  obd_tolerance_s OBD logs at ~0.65 Hz, so section 8's flat 100 ms window would drop ~90% of
                  frames. The tolerance is derived from the measured median interval rather
                  than hardcoded. See docs/DECISIONS.md D010.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from drivyx.paths import Paths

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1

#: Confirmation states for a FieldMapTable row (section 8: "mm-label refuses to run while any
#: required mapping is unconfirmed").
UNCONFIRMED = "unconfirmed"
CONFIRMED = "confirmed"

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
TABLE_SUFFIXES = frozenset({".csv", ".txt", ".tsv"})
LIDAR_SUFFIXES = frozenset({".npy", ".bin", ".pcd", ".ply"})

#: Header tokens that suggest a column's role. Ordered by specificity: the first pattern that
#: matches a column name wins, so 'timestamp' beats a bare 't'. These are ranked *guesses*
#: offered to the FieldMapTable, never applied without confirmation.
TIME_PATTERNS = (r"^timestamp$", r"^time$", r"^ts$", r"^t$", r".*time.*", r".*stamp.*")
LAT_PATTERNS = (r"^latitude$", r"^lat$", r".*latitude.*", r".*\blat\b.*")
LON_PATTERNS = (r"^longitude$", r"^lon$", r"^lng$", r".*longitude.*", r".*\blon\b.*")
SPEED_PATTERNS = (r"^speed$", r"^velocity$", r"^vel$", r".*speed.*", r".*velocity.*")
FRAME_PATTERNS = (r"^image_idx$", r"^frame$", r"^frame_idx$", r".*image.*idx.*", r".*frame.*")

#: Timestamp shapes seen in the wild. IDD multimodal uses HH-MM-SS-microseconds, which is not
#: an ISO or epoch format, so it is parsed explicitly rather than guessed at.
_RE_HMS_US = re.compile(r"^(\d{1,2})-(\d{2})-(\d{2})-(\d+)$")
_RE_HMS = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)$")

#: Rows read when sniffing a table's shape and rate. Enough to measure a stable median
#: interval without reading a 4500-row file twice.
SNIFF_ROWS = 4000


@dataclass
class ColumnGuess:
    """A proposed column mapping, pending human confirmation (section 8)."""

    role: str
    column: str | None
    confidence: str
    state: str = UNCONFIRMED
    candidates: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.column is not None and self.state == CONFIRMED


def parse_timestamp(value: str) -> float | None:
    """Seconds since midnight, or None if the value is not a recognised timestamp.

    Handles the IDD multimodal format (HH-MM-SS-microseconds), clock strings, and bare
    numerics (already-elapsed seconds or an epoch). Returning None rather than raising lets
    the caller decide: a table where nothing parses is a discovery failure worth reporting,
    but one unparseable row is not.
    """
    value = value.strip()
    if not value:
        return None

    match = _RE_HMS_US.match(value)
    if match:
        hours, minutes, seconds, fraction = match.groups()
        # The fractional field is microseconds, zero padded to a variable width in practice,
        # so it is scaled by its own digit count rather than assumed to be 6 digits.
        micro = int(fraction) / (10 ** len(fraction))
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + micro

    match = _RE_HMS.match(value)
    if match:
        hours, minutes, seconds = match.groups()
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    try:
        return float(value)
    except ValueError:
        return None


def _match_role(columns: list[str], patterns: tuple[str, ...]) -> tuple[str | None, str, list[str]]:
    """Rank columns against a role's patterns.

    Returns (best column, confidence, all candidates). Confidence is "exact" when the column
    name equals the role's canonical name, "fuzzy" when it merely contains it. The
    distinction is surfaced so the FieldMapTable can show a fuzzy guess as needing a closer
    look.
    """
    candidates: list[str] = []
    best: str | None = None
    confidence = "none"

    for index, pattern in enumerate(patterns):
        regex = re.compile(pattern, re.IGNORECASE)
        for column in columns:
            if regex.match(column.strip().lower()):
                if column not in candidates:
                    candidates.append(column)
                if best is None:
                    best = column
                    # The first two patterns of each role are anchored exact names.
                    confidence = "exact" if index <= 1 else "fuzzy"
    return best, confidence, candidates


def _sniff_table(path: Path) -> dict[str, Any]:
    """Read a delimited table's header, row count, and column roles."""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            delimiter = dialect.delimiter
        except csv.Error:
            # A single-column file gives the sniffer nothing to work with; comma is the IDD
            # convention and the header check below will catch a real mismatch.
            delimiter = ","
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            return {"path": str(path), "error": "file is empty", "columns": []}
        rows = sum(1 for _ in reader)

    columns = [c.strip() for c in header]
    info: dict[str, Any] = {
        "path": str(path),
        "delimiter": delimiter,
        "columns": columns,
        "rows": rows,
    }
    return info


def _read_column_values(
    path: Path, column: str, delimiter: str, limit: int = SNIFF_ROWS
) -> list[str]:
    """Read up to `limit` raw values of one column."""
    values: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            value = row.get(column)
            if value is not None:
                values.append(value)
            if len(values) >= limit:
                break
    return values


def _measure_rate(path: Path, time_column: str, delimiter: str) -> dict[str, Any]:
    """Measure a table's sample rate from its timestamps (section 8: "measured from data").

    The median interval is used rather than the mean: a single gap or a clock jump would
    drag a mean far off, while the median reports the cadence the sensor actually holds.
    """
    raw = _read_column_values(path, time_column, delimiter)
    times = [t for t in (parse_timestamp(v) for v in raw) if t is not None]
    if len(times) < 2:
        return {"parsed": len(times), "error": "fewer than two parseable timestamps"}

    times.sort()
    deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not deltas:
        return {"parsed": len(times), "error": "timestamps do not advance"}

    deltas.sort()
    median = deltas[len(deltas) // 2]
    return {
        "parsed": len(times),
        "unparsed": len(raw) - len(times),
        "first": times[0],
        "last": times[-1],
        "span_s": round(times[-1] - times[0], 3),
        "median_dt_s": round(median, 6),
        "hz": round(1.0 / median, 3) if median > 0 else None,
    }


def _classify_dir(path: Path) -> dict[str, Any]:
    """Count a directory's files by kind, without descending into subdirectories."""
    counts: Counter[str] = Counter()
    total_bytes = 0
    names: list[str] = []
    for entry in path.iterdir():
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            kind = "image"
        elif suffix in TABLE_SUFFIXES:
            kind = "table"
        elif suffix in LIDAR_SUFFIXES:
            kind = "lidar"
        else:
            kind = "other"
        counts[kind] += 1
        try:
            total_bytes += entry.stat().st_size
        except OSError:
            continue
        if len(names) < 3:
            names.append(entry.name)
    return {"counts": dict(counts), "bytes": total_bytes, "examples": sorted(names)}


def _side_of(name: str) -> str | None:
    """Guess a stereo side from a directory name (section 8: "left/right if distinguishable").

    Substring matching is deliberate: the observed names are `leftCamImgs` and `rightCamImgs`,
    but nothing guarantees that spelling, and a directory that says neither is reported as
    unknown rather than assigned a side.
    """
    lowered = name.lower()
    has_left = "left" in lowered
    has_right = "right" in lowered
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    return None


def _discover_tables(root: Path) -> list[dict[str, Any]]:
    """Every delimited table under root, with its columns and role guesses."""
    tables: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TABLE_SUFFIXES:
            continue
        info = _sniff_table(path)
        if info.get("error"):
            tables.append(info)
            continue

        columns = info["columns"]
        roles: dict[str, Any] = {}
        for role, patterns in (
            ("time", TIME_PATTERNS),
            ("lat", LAT_PATTERNS),
            ("lon", LON_PATTERNS),
            ("speed", SPEED_PATTERNS),
            ("frame", FRAME_PATTERNS),
        ):
            column, confidence, candidates = _match_role(columns, patterns)
            if column is not None:
                roles[role] = asdict(
                    ColumnGuess(
                        role=role,
                        column=column,
                        confidence=confidence,
                        candidates=candidates,
                    )
                )
        info["roles"] = roles

        # A table's kind follows from the roles present, not from its filename: a table with
        # lat and lon is GPS whatever it is called, and one with speed but no position is OBD.
        if "lat" in roles and "lon" in roles:
            info["kind"] = "gps"
        elif "speed" in roles:
            info["kind"] = "obd"
        else:
            info["kind"] = "unknown"

        if "time" in roles:
            info["rate"] = _measure_rate(path, roles["time"]["column"], info["delimiter"])
        tables.append(info)
    return tables


def _discover_image_dirs(root: Path) -> list[dict[str, Any]]:
    """Every directory holding images, with a stereo side guess."""
    found: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        summary = _classify_dir(path)
        if summary["counts"].get("image", 0) == 0:
            continue
        found.append(
            {
                "path": str(path),
                "name": path.name,
                "images": summary["counts"]["image"],
                "bytes": summary["bytes"],
                "side": _side_of(path.name),
                "examples": summary["examples"],
            }
        )
    return found


def _discover_lidar_dirs(root: Path) -> list[dict[str, Any]]:
    """LiDAR directories. Section 4: inventory it, then ignore it for v1."""
    found: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        summary = _classify_dir(path)
        if summary["counts"].get("lidar", 0) == 0:
            continue
        found.append(
            {
                "path": str(path),
                "frames": summary["counts"]["lidar"],
                "bytes": summary["bytes"],
                "note": "inventoried and ignored: LiDAR is out of scope for v1 (section 4)",
            }
        )
    return found


def _route_of(path: Path, root: Path) -> str | None:
    """The route a path belongs to.

    A route is identified structurally, as the path component that recurs across the archive
    groups rather than by any hardcoded name: the observed tree is
    <root>/<group>/<route>/... for primary and secondary, and
    <root>/supplement/<modality>/<route>/... for the supplement. Both put the route directly
    above the leaf data, so the candidate is read from the path rather than assumed.
    """
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return None
    # Walk from the leaf upward and take the first component that looks like a route id: a
    # short token, not a known modality or side name.
    for part in reversed(parts[:-1] if path.is_file() else parts):
        lowered = part.lower()
        if lowered in {"primary", "secondary", "supplement", "lidar", "obd"}:
            continue
        if _side_of(part) is not None or "cam" in lowered or "img" in lowered:
            continue
        return part
    return None


def _group_by_route(
    root: Path, tables: list[dict], image_dirs: list[dict], lidar_dirs: list[dict]
) -> dict[str, Any]:
    """Assemble per-route entries from the discovered pieces."""
    routes: dict[str, dict[str, Any]] = {}

    def entry(name: str) -> dict[str, Any]:
        return routes.setdefault(
            name,
            {
                "route": name,
                "image_dirs": [],
                "gps": None,
                "obd": None,
                "lidar": None,
                "other_tables": [],
            },
        )

    for image_dir in image_dirs:
        route = _route_of(Path(image_dir["path"]), root)
        if route:
            entry(route)["image_dirs"].append(image_dir)

    for lidar_dir in lidar_dirs:
        route = _route_of(Path(lidar_dir["path"]), root)
        if route:
            entry(route)["lidar"] = lidar_dir

    for table in tables:
        route = _route_of(Path(table["path"]), root)
        if not route:
            continue
        target = entry(route)
        kind = table.get("kind")
        if kind == "gps":
            # A route can ship several GPS tables (IDD splits them train/val/test). They are
            # kept as a list so mm-label can read the union; the split provided by the dataset
            # is not used for the val split (docs/DECISIONS.md D011).
            if target["gps"] is None:
                target["gps"] = {
                    "tables": [],
                    "roles": table.get("roles", {}),
                    "rate": table.get("rate"),
                }
            target["gps"]["tables"].append(
                {"path": table["path"], "rows": table["rows"], "columns": table["columns"]}
            )
        elif kind == "obd":
            if target["obd"] is None:
                target["obd"] = {
                    "tables": [],
                    "roles": table.get("roles", {}),
                    "rate": table.get("rate"),
                }
            target["obd"]["tables"].append(
                {"path": table["path"], "rows": table["rows"], "columns": table["columns"]}
            )
        else:
            target["other_tables"].append(table)

    return routes


def _measure_clock_offset(
    gps: dict[str, Any] | None, obd: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Propose the OBD-to-GPS clock offset for a route (docs/DECISIONS.md D009).

    Never applied automatically. Rule 30 makes timestamp misalignment an abort condition, so
    this measures the offset, states the hypothesis that explains it, and leaves the field
    unconfirmed for a human. mm-label refuses to run until it is confirmed.

    The offset is estimated by aligning the two windows' starts. That is deliberately crude:
    a cross-correlation would fit the shape of the overlap, but the two clocks measure
    different quantities (position and wheel speed) and the estimate only needs to be good
    enough for a human to recognise a timezone when they see one.
    """
    if not gps or not obd:
        return None
    gps_rate, obd_rate = gps.get("rate"), obd.get("rate")
    if not gps_rate or not obd_rate:
        return None
    if "first" not in gps_rate or "first" not in obd_rate:
        return None

    raw_offset = gps_rate["first"] - obd_rate["first"]
    overlap = min(gps_rate["last"], obd_rate["last"]) - max(gps_rate["first"], obd_rate["first"])

    # Common timezone offsets in seconds, nearest first. The data is from Hyderabad, so IST
    # is expected, but the table is general and the match is reported, not assumed.
    known = {
        19800: "UTC+5:30 (IST): OBD logged in UTC while GPS logged local Indian time",
        0: "no offset",
        18000: "UTC+5:00",
        16200: "UTC+4:30",
        21600: "UTC+6:00",
    }
    best_seconds, best_label = min(known.items(), key=lambda kv: abs(raw_offset - kv[0]))
    residual = raw_offset - best_seconds

    return asdict(
        ColumnGuess(
            role="clock_offset_s",
            column=None,
            confidence="measured",
            state=UNCONFIRMED,
        )
    ) | {
        "proposed_s": best_seconds,
        "raw_offset_s": round(raw_offset, 3),
        "residual_s": round(residual, 3),
        "hypothesis": best_label,
        "raw_overlap_s": round(overlap, 3),
        "note": (
            "Add proposed_s to every OBD timestamp to bring it onto the GPS clock. The "
            "residual is the logger start skew: OBD logging began after the camera, so a "
            "few tens of seconds is expected and is not corrected. This is unconfirmed: "
            "CLAUDE.md rule 30 requires timestamp misalignment to abort rather than be "
            "silently corrected, so mm-label refuses to run until a human confirms it."
        ),
    }


def _propose_obd_tolerance(obd: dict[str, Any] | None) -> dict[str, Any] | None:
    """Derive the OBD association window from the measured rate (docs/DECISIONS.md D010).

    Section 8 specifies a flat 100 ms. OBD here logs at ~0.65 Hz, so a 100 ms window keeps
    about 10% of frames. The window is therefore one measured sampling interval, which keeps
    ~85%, and is derived rather than hardcoded so a faster OBD log tightens it automatically.
    """
    if not obd or not obd.get("rate") or "median_dt_s" not in obd["rate"]:
        return None
    median = obd["rate"]["median_dt_s"]
    return {
        "proposed_s": round(median, 3),
        "spec_s": 0.100,
        "measured_median_dt_s": median,
        "measured_hz": obd["rate"].get("hz"),
        "state": UNCONFIRMED,
        "note": (
            "CLAUDE.md section 8 specifies a 100 ms OBD window, which assumes a rate this "
            "log does not have. One measured sampling interval is used instead, with speed "
            "linearly interpolated between bracketing samples. See docs/DECISIONS.md D010."
        ),
    }


def mm_inventory(paths: Paths) -> dict[str, Any]:
    """Walk multimodal/ and build the manifest (section 8)."""
    root = paths.multimodal
    if not root.is_dir():
        raise FileNotFoundError(
            f"No multimodal tree at {root}. Run scripts/stage_data.sh, then 'drivyx verify-data'."
        )

    logger.info("mm-inventory: walking %s", root)
    tables = _discover_tables(root)
    image_dirs = _discover_image_dirs(root)
    lidar_dirs = _discover_lidar_dirs(root)
    routes = _group_by_route(root, tables, image_dirs, lidar_dirs)

    for route in routes.values():
        route["clock_offset"] = _measure_clock_offset(route.get("gps"), route.get("obd"))
        route["obd_tolerance"] = _propose_obd_tolerance(route.get("obd"))
        route["gps_tolerance"] = {
            "proposed_s": 0.050,
            "spec_s": 0.050,
            "state": CONFIRMED,
            "note": "Section 8's 50 ms GPS window, unchanged: GPS logs at 15 Hz here.",
        }

    manifest: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "root": str(root),
        "routes": {name: routes[name] for name in sorted(routes)},
        "unassigned_tables": [t for t in tables if _route_of(Path(t["path"]), root) is None],
        "note": (
            "Generated by mm-inventory from the bytes on disk. Every column mapping and the "
            "clock offset start unconfirmed; confirm them in the LABEL workspace's "
            "FieldMapTable. mm-label refuses to run while any required mapping is "
            "unconfirmed (section 8)."
        ),
    }
    logger.info(
        "mm-inventory: %d routes, %d tables, %d image dirs, %d lidar dirs",
        len(routes),
        len(tables),
        len(image_dirs),
        len(lidar_dirs),
    )
    return manifest


def write_manifest(paths: Paths, manifest: dict[str, Any]) -> Path:
    """Write mm_manifest.json, preserving confirmations from a previous run.

    Re-running mm-inventory must not silently discard a human's confirmations: that would
    make the FieldMapTable's whole purpose evaporate on any re-scan. Confirmed states are
    carried forward wherever the discovered column still matches what was confirmed.
    """
    if paths.mm_manifest.is_file():
        try:
            previous = json.loads(paths.mm_manifest.read_text())
            manifest = _merge_confirmations(previous, manifest)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not read previous manifest (%s); writing a fresh one", exc)

    paths.mm_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.mm_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("wrote %s", paths.mm_manifest)
    return paths.mm_manifest


def _merge_confirmations(previous: dict[str, Any], fresh: dict[str, Any]) -> dict[str, Any]:
    """Carry confirmed states forward onto a freshly discovered manifest.

    A confirmation only survives if it still refers to the same column. If the data changed
    under it, the row reverts to unconfirmed rather than silently keeping a stale approval.
    """
    for name, route in fresh.get("routes", {}).items():
        old_route = previous.get("routes", {}).get(name)
        if not old_route:
            continue

        for kind in ("gps", "obd"):
            new_block, old_block = route.get(kind), old_route.get(kind)
            if not new_block or not old_block:
                continue
            for role, guess in new_block.get("roles", {}).items():
                old_guess = old_block.get("roles", {}).get(role)
                if (
                    old_guess
                    and old_guess.get("state") == CONFIRMED
                    and old_guess.get("column") == guess.get("column")
                ):
                    guess["state"] = CONFIRMED

        for key in ("clock_offset", "obd_tolerance"):
            new_field, old_field = route.get(key), old_route.get(key)
            if not new_field or not old_field:
                continue
            if old_field.get("state") == CONFIRMED and old_field.get("proposed_s") == new_field.get(
                "proposed_s"
            ):
                new_field["state"] = CONFIRMED
                if "confirmed_s" in old_field:
                    new_field["confirmed_s"] = old_field["confirmed_s"]
    return fresh


def read_manifest(paths: Paths) -> dict[str, Any]:
    """Read mm_manifest.json, pointing at mm-inventory when absent."""
    if not paths.mm_manifest.is_file():
        raise FileNotFoundError(
            f"Manifest not found at {paths.mm_manifest}. Run 'drivyx mm-inventory' first."
        )
    return json.loads(paths.mm_manifest.read_text())


def unconfirmed_fields(manifest: dict[str, Any]) -> list[str]:
    """Every required mapping still awaiting confirmation (section 8).

    mm-label calls this and refuses to run while it is non-empty. Only the fields mm-label
    actually consumes are required: a route with no OBD is usable (GPS-derived speed alone,
    per section 8.2), so its OBD rows are not demanded.
    """
    pending: list[str] = []
    for name, route in manifest.get("routes", {}).items():
        gps = route.get("gps")
        if not gps:
            continue
        for role in ("time", "lat", "lon", "frame"):
            guess = gps.get("roles", {}).get(role)
            if guess is None:
                pending.append(f"{name}.gps.{role}: no column proposed")
            elif guess.get("state") != CONFIRMED:
                pending.append(f"{name}.gps.{role} ({guess.get('column')!r})")

        obd = route.get("obd")
        if obd:
            for role in ("time", "speed"):
                guess = obd.get("roles", {}).get(role)
                if guess is None:
                    pending.append(f"{name}.obd.{role}: no column proposed")
                elif guess.get("state") != CONFIRMED:
                    pending.append(f"{name}.obd.{role} ({guess.get('column')!r})")

            offset = route.get("clock_offset")
            if offset and offset.get("state") != CONFIRMED:
                pending.append(
                    f"{name}.clock_offset_s (proposed {offset.get('proposed_s')}s: "
                    f"{offset.get('hypothesis')})"
                )
            tolerance = route.get("obd_tolerance")
            if tolerance and tolerance.get("state") != CONFIRMED:
                pending.append(f"{name}.obd_tolerance_s (proposed {tolerance.get('proposed_s')}s)")
    return pending


def set_confirmation(
    manifest: dict[str, Any],
    route: str,
    field: str,
    *,
    column: str | None = None,
    value: float | None = None,
    confirmed: bool = True,
) -> None:
    """Confirm or override one FieldMapTable row, in place.

    `field` is either "<block>.<role>" (e.g. "gps.time") or a scalar field name
    ("clock_offset", "obd_tolerance"). Overriding a column also confirms it: a human who
    typed a column name has, by doing so, confirmed it.
    """
    route_block = manifest.get("routes", {}).get(route)
    if route_block is None:
        raise KeyError(f"route {route!r} is not in the manifest")

    if "." in field:
        block_name, role = field.split(".", 1)
        block = route_block.get(block_name)
        if block is None:
            raise KeyError(f"route {route!r} has no {block_name!r} block")
        guess = block.setdefault("roles", {}).setdefault(
            role, asdict(ColumnGuess(role=role, column=None, confidence="manual"))
        )
        if column is not None:
            if column not in _columns_of(block):
                raise ValueError(
                    f"{route}.{field}: {column!r} is not a column of that table. "
                    f"Available: {_columns_of(block)}"
                )
            guess["column"] = column
            guess["confidence"] = "manual"
        guess["state"] = CONFIRMED if confirmed else UNCONFIRMED
        return

    scalar = route_block.get(field)
    if scalar is None:
        raise KeyError(f"route {route!r} has no {field!r} field")
    if value is not None:
        scalar["confirmed_s"] = float(value)
    scalar["state"] = CONFIRMED if confirmed else UNCONFIRMED


def _columns_of(block: dict[str, Any]) -> list[str]:
    """Every column name across a block's tables."""
    columns: list[str] = []
    for table in block.get("tables", []):
        for column in table.get("columns", []):
            if column not in columns:
                columns.append(column)
    return columns


def confirm_all(paths: Paths, manifest: dict[str, Any], *, route: str | None = None) -> list[str]:
    """Confirm every proposed mapping and write the manifest back.

    This accepts mm-inventory's own guesses wholesale, which is only safe because each guess
    is reported with its confidence and the caller has seen them (the CLI prints them and
    requires --yes). It exists so the engine stays usable headless per section 3; the
    FieldMapTable remains the path where a human can disagree with an individual row.
    """
    routes = [route] if route else sorted(manifest.get("routes", {}))
    changed: list[str] = []

    for name in routes:
        route_block = manifest["routes"][name]
        for block_name in ("gps", "obd"):
            block = route_block.get(block_name)
            if not block:
                continue
            for role, guess in block.get("roles", {}).items():
                if guess.get("column") and guess.get("state") != CONFIRMED:
                    guess["state"] = CONFIRMED
                    changed.append(f"{name}.{block_name}.{role} = {guess['column']!r}")

        for field in ("clock_offset", "obd_tolerance"):
            scalar = route_block.get(field)
            if scalar and scalar.get("state") != CONFIRMED:
                scalar["state"] = CONFIRMED
                scalar["confirmed_s"] = scalar.get("proposed_s")
                changed.append(f"{name}.{field} = {scalar.get('proposed_s')}")

    # write_manifest merges confirmations from the file on disk onto a fresh scan, so it is
    # bypassed here: this manifest IS the newer one.
    paths.mm_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.mm_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("confirmed %d manifest fields", len(changed))
    return changed
