"""Data inventory and integrity report (CLAUDE.md sections 6.1 and 7).

`verify-data` is the gate for everything downstream, so it reports what is actually on disk
rather than what the spec expects to be there, and names the exact mismatch when the two
disagree.

The multimodal tree is deliberately inspected only shallowly here. Section 4 declares its
internal layout unknown at spec time and section 8 makes mm-inventory the sole authority on
it, so this module counts files and lists top-level entries without interpreting them.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from drivyx import __version__
from drivyx.branding import APP_NAME
from drivyx.paths import Paths

logger = logging.getLogger(__name__)

#: Section 4: "20,000 images (14K train / 2K val / 4K test) across 350 sequences".
#: Held as expectations to compare against, never as assertions: the published IDD 20k
#: counts are not exactly round, so the report shows both and lets the reader judge.
EXPECTED_SPLIT_IMAGES: dict[str, int] = {"train": 14000, "val": 2000, "test": 4000}
EXPECTED_TOTAL_IMAGES = 20000
EXPECTED_SEQUENCES = 350

#: Fraction by which an actual split count may differ from EXPECTED_SPLIT_IMAGES before the
#: check is flagged. The published splits sit within a couple of percent of the round
#: numbers in section 4; anything beyond 10 percent means the wrong data is staged.
SPLIT_COUNT_TOLERANCE = 0.10

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
IMAGE_MARKER = "_leftImg8bit"
POLYGON_MARKER = "_gtFine_polygons"

#: Splits carrying ground-truth polygons. `test` is withheld by IDD, so it has images only.
ANNOTATED_SPLITS = ("train", "val")
ALL_SPLITS = ("train", "val", "test")

#: Checkpoint filenames accepted for the PIDNet-S ImageNet backbone (section 9.1). The
#: upstream release name is listed first.
PRETRAINED_CANDIDATES = ("PIDNet_S_ImageNet.pth.tar", "PIDNet_S_ImageNet.pth")
PRETRAINED_SUFFIXES = (".pth", ".pth.tar", ".tar")

PRETRAINED_HINT = (
    "PIDNet-S ImageNet backbone not found. Download 'PIDNet_S_ImageNet.pth.tar' from the "
    "PIDNet repository (https://github.com/XuJiacong/PIDNet, see its 'Pretrained Models' "
    "section, hosted on Google Drive) and place it at {target}. "
    "Segmentation training (train-seg) cannot start without it."
)

SEVERITY_ERROR = "error"
SEVERITY_WARN = "warn"


@dataclass
class Check:
    """One pass/fail assertion in the report.

    Severity `error` means a downstream stage is blocked; `warn` means degraded but usable.
    """

    name: str
    ok: bool
    detail: str
    severity: str = SEVERITY_ERROR

    @property
    def blocking(self) -> bool:
        return not self.ok and self.severity == SEVERITY_ERROR


@dataclass
class SplitReport:
    """Per-split image and polygon accounting for the segmentation tree."""

    images: int = 0
    polygons: int = 0
    sequences: int = 0
    #: Truncated samples for readability; the n_* fields carry the true totals.
    images_without_polygons: list[str] = field(default_factory=list)
    polygons_without_images: list[str] = field(default_factory=list)
    n_images_without_polygons: int = 0
    n_polygons_without_images: int = 0
    image_suffixes: dict[str, int] = field(default_factory=dict)
    per_sequence_images: dict[str, int] = field(default_factory=dict)


def _stem_without_marker(filename: str, marker: str) -> str:
    """Strip the IDD role marker and extension to get the frame key.

    '820516_leftImg8bit.png' -> '820516'
    'frame0149_gtFine_polygons.json' -> 'frame0149'

    Files lacking the marker return their bare stem, which keeps an unexpected naming
    convention visible as an unpaired frame rather than silently dropping it.
    """
    stem = filename
    for suffix in (".json", *IMAGE_SUFFIXES):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if stem.endswith(marker):
        stem = stem[: -len(marker)]
    return stem


def _scan_split_images(split_dir: Path) -> tuple[dict[str, set[str]], Counter[str]]:
    """Map sequence id -> frame keys, plus a histogram of file extensions."""
    per_sequence: dict[str, set[str]] = {}
    suffixes: Counter[str] = Counter()

    if not split_dir.is_dir():
        return per_sequence, suffixes

    for seq_entry in os.scandir(split_dir):
        if not seq_entry.is_dir():
            continue
        keys: set[str] = set()
        for file_entry in os.scandir(seq_entry.path):
            if not file_entry.is_file():
                continue
            suffix = Path(file_entry.name).suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                continue
            suffixes[suffix] += 1
            keys.add(_stem_without_marker(file_entry.name, IMAGE_MARKER))
        per_sequence[seq_entry.name] = keys

    return per_sequence, suffixes


def _scan_split_polygons(split_dir: Path) -> dict[str, set[str]]:
    """Map sequence id -> frame keys for *_polygons.json."""
    per_sequence: dict[str, set[str]] = {}

    if not split_dir.is_dir():
        return per_sequence

    for seq_entry in os.scandir(split_dir):
        if not seq_entry.is_dir():
            continue
        keys = {
            _stem_without_marker(f.name, POLYGON_MARKER)
            for f in os.scandir(seq_entry.path)
            if f.is_file() and f.name.endswith(".json")
        }
        per_sequence[seq_entry.name] = keys

    return per_sequence


def _report_split(seg_root: Path, split: str, *, max_listed: int = 20) -> SplitReport:
    """Count and pair one segmentation split.

    Unpaired frames are listed but capped: a systematic pairing failure would otherwise
    produce a report thousands of lines long, and the count already conveys the scale.
    """
    images_by_seq, suffixes = _scan_split_images(seg_root / "leftImg8bit" / split)
    report = SplitReport(
        images=sum(len(v) for v in images_by_seq.values()),
        sequences=len(images_by_seq),
        image_suffixes=dict(suffixes),
        per_sequence_images={k: len(v) for k, v in sorted(images_by_seq.items())},
    )

    if split not in ANNOTATED_SPLITS:
        return report

    polygons_by_seq = _scan_split_polygons(seg_root / "gtFine" / split)
    report.polygons = sum(len(v) for v in polygons_by_seq.values())

    missing_polygons: list[str] = []
    missing_images: list[str] = []
    for seq in sorted(set(images_by_seq) | set(polygons_by_seq)):
        img_keys = images_by_seq.get(seq, set())
        poly_keys = polygons_by_seq.get(seq, set())
        missing_polygons.extend(f"{split}/{seq}/{k}" for k in sorted(img_keys - poly_keys))
        missing_images.extend(f"{split}/{seq}/{k}" for k in sorted(poly_keys - img_keys))

    report.n_images_without_polygons = len(missing_polygons)
    report.n_polygons_without_images = len(missing_images)
    report.images_without_polygons = missing_polygons[:max_listed]
    report.polygons_without_images = missing_images[:max_listed]
    return report


def _seg_section(paths: Paths, checks: list[Check]) -> dict[str, Any]:
    """Inventory the segmentation tree and append its checks."""
    seg_root = paths.seg
    section: dict[str, Any] = {"root": str(seg_root), "present": seg_root.is_dir()}

    if not section["present"]:
        checks.append(
            Check(
                name="seg.extracted",
                ok=False,
                detail=(
                    f"Segmentation tree missing at {seg_root}. Run scripts/stage_data.sh to "
                    "extract IDD Segmentation Parts I and II from the downloaded archives."
                ),
            )
        )
        return section

    splits: dict[str, Any] = {}
    total_images = 0
    total_sequences = 0

    for split in ALL_SPLITS:
        rep = _report_split(seg_root, split)
        splits[split] = asdict(rep)
        splits[split]["expected_images"] = EXPECTED_SPLIT_IMAGES.get(split)
        total_images += rep.images
        total_sequences += rep.sequences

        if split in ANNOTATED_SPLITS:
            paired = rep.n_images_without_polygons == 0 and rep.n_polygons_without_images == 0
            checks.append(
                Check(
                    name=f"seg.pairing.{split}",
                    ok=paired,
                    detail=(
                        f"{split}: {rep.images} images / {rep.polygons} polygons, fully paired"
                        if paired
                        else (
                            f"{split}: {rep.images} images vs {rep.polygons} polygons. "
                            f"{rep.n_images_without_polygons} images lack polygons "
                            f"(e.g. {rep.images_without_polygons[:5] or 'none'}); "
                            f"{rep.n_polygons_without_images} polygons lack images "
                            f"(e.g. {rep.polygons_without_images[:5] or 'none'})"
                        )
                    ),
                )
            )

        expected = EXPECTED_SPLIT_IMAGES.get(split)
        if expected:
            drift = abs(rep.images - expected) / expected
            checks.append(
                Check(
                    name=f"seg.count.{split}",
                    ok=drift <= SPLIT_COUNT_TOLERANCE,
                    detail=(
                        f"{split}: {rep.images} images (spec section 4 expects ~{expected}, "
                        f"drift {drift * 100:.1f}%)"
                    ),
                    severity=SEVERITY_WARN,
                )
            )

    section["splits"] = splits
    section["total_images"] = total_images
    section["total_sequences"] = total_sequences
    section["expected_total_images"] = EXPECTED_TOTAL_IMAGES
    section["expected_sequences"] = EXPECTED_SEQUENCES

    checks.append(
        Check(
            name="seg.total",
            ok=abs(total_images - EXPECTED_TOTAL_IMAGES) / EXPECTED_TOTAL_IMAGES
            <= SPLIT_COUNT_TOLERANCE,
            detail=(
                f"{total_images} images across {total_sequences} sequences "
                f"(spec section 4 expects ~{EXPECTED_TOTAL_IMAGES} across "
                f"~{EXPECTED_SEQUENCES})"
            ),
            severity=SEVERITY_WARN,
        )
    )
    return section


def _multimodal_section(paths: Paths, checks: list[Check]) -> dict[str, Any]:
    """Shallow multimodal inventory.

    Counts files by extension and lists the top two directory levels. Interpreting the
    layout is mm-inventory's job (section 8); doing it here would hardcode exactly what the
    spec forbids hardcoding.
    """
    mm_root = paths.multimodal
    section: dict[str, Any] = {"root": str(mm_root), "present": mm_root.is_dir()}

    if not section["present"]:
        checks.append(
            Check(
                name="multimodal.extracted",
                ok=False,
                detail=(
                    f"Multimodal tree missing at {mm_root}. Run scripts/stage_data.sh to "
                    "extract the IDD Multimodal archives."
                ),
            )
        )
        return section

    suffixes: Counter[str] = Counter()
    file_count = 0
    total_bytes = 0
    for dirpath, _dirnames, filenames in os.walk(mm_root):
        for name in filenames:
            suffixes[Path(name).suffix.lower() or "<none>"] += 1
            file_count += 1
            try:
                total_bytes += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                # A file vanishing mid-walk is not an integrity claim this report makes;
                # the count above still records it.
                continue

    top_level = sorted(p.name for p in mm_root.iterdir() if p.is_dir())
    second_level: dict[str, list[str]] = {}
    for entry in top_level:
        second_level[entry] = sorted(p.name for p in (mm_root / entry).iterdir() if p.is_dir())[:10]

    section["file_count"] = file_count
    section["total_bytes"] = total_bytes
    section["extensions"] = dict(suffixes.most_common())
    section["top_level"] = top_level
    section["second_level"] = second_level
    section["manifest_present"] = paths.mm_manifest.is_file()
    section["note"] = (
        "Layout is not interpreted here. Run 'drivyx mm-inventory' to discover routes, "
        "GPS/OBD tables, and column mappings into mm_manifest.json (section 8)."
    )

    checks.append(
        Check(
            name="multimodal.extracted",
            ok=file_count > 0,
            detail=(
                f"{file_count} files, {total_bytes / 1e9:.1f} GB, top level: {top_level or 'empty'}"
            ),
        )
    )
    return section


def _find_pretrained(pretrained_dir: Path) -> Path | None:
    """Locate the backbone checkpoint, preferring the upstream filename."""
    if not pretrained_dir.is_dir():
        return None
    for candidate in PRETRAINED_CANDIDATES:
        path = pretrained_dir / candidate
        if path.is_file():
            return path
    for entry in sorted(pretrained_dir.iterdir()):
        if entry.is_file() and entry.name.endswith(PRETRAINED_SUFFIXES):
            return entry
    return None


def _pretrained_section(paths: Paths, checks: list[Check]) -> dict[str, Any]:
    """Check for the PIDNet-S ImageNet backbone.

    Absence is a warning, not an error: section 4 says verify-data "checks presence and
    prints the download hint if absent", and M0 through M3 do not need it. train-seg aborts
    on its own if it is still missing.
    """
    found = _find_pretrained(paths.pretrained)
    target = paths.pretrained / PRETRAINED_CANDIDATES[0]
    section: dict[str, Any] = {
        "dir": str(paths.pretrained),
        "present": found is not None,
        "path": str(found) if found else None,
        "bytes": found.stat().st_size if found else None,
    }
    if found is None:
        section["hint"] = PRETRAINED_HINT.format(target=target)

    checks.append(
        Check(
            name="pretrained.backbone",
            ok=found is not None,
            detail=(
                f"found {found} ({found.stat().st_size / 1e6:.1f} MB)"
                if found
                else PRETRAINED_HINT.format(target=target)
            ),
            severity=SEVERITY_WARN,
        )
    )
    return section


def _raw_section(paths: Paths, checks: list[Check]) -> dict[str, Any]:
    """Inventory the preserved source archives (section 4: never deleted by code)."""
    raw_root = paths.raw
    section: dict[str, Any] = {"root": str(raw_root), "present": raw_root.is_dir()}
    if not section["present"]:
        section["archives"] = []
        return section

    archives = []
    for entry in sorted(raw_root.iterdir()):
        if entry.is_file():
            archives.append({"name": entry.name, "bytes": entry.stat().st_size})
    section["archives"] = archives
    return section


def _generated_section(paths: Paths) -> dict[str, Any]:
    """Report which generated stages already exist, so the GUI can grey out redone work."""
    return {
        "masks": paths.masks.is_dir() and any(paths.masks.iterdir()),
        "lut": paths.lut_json.is_file(),
        "shards": paths.shards.is_dir() and any(paths.shards.glob("*.tar")),
        "shard_index": paths.shard_index.is_file(),
        "waypoints": paths.waypoints.is_dir() and any(paths.waypoints.glob("*.parquet")),
        "runs": sorted(p.name for p in paths.runs.iterdir() if p.is_dir())
        if paths.runs.is_dir()
        else [],
    }


def _disk_section(paths: Paths) -> dict[str, Any]:
    """Free space on the data root's filesystem."""
    probe = paths.data_root if paths.data_root.exists() else paths.data_root.anchor
    try:
        usage = os.statvfs(probe)
    except OSError as exc:
        return {"error": f"cannot stat {probe}: {exc}"}
    return {
        "probe": str(probe),
        "total_bytes": usage.f_blocks * usage.f_frsize,
        "free_bytes": usage.f_bavail * usage.f_frsize,
    }


def verify_data(paths: Paths) -> dict[str, Any]:
    """Build the full inventory and integrity report.

    Returns the report as a plain dict for JSON serialisation. Never raises on missing
    data: absence is a reported check, since the GUI DATA workspace renders this report to
    tell the user what to do next.
    """
    logger.info("Verifying data under %s", paths.data_root)
    checks: list[Check] = []

    checks.append(
        Check(
            name="data_root",
            ok=paths.data_root.is_dir(),
            detail=(
                f"{paths.data_root} exists"
                if paths.data_root.is_dir()
                else f"{paths.data_root} does not exist. Run scripts/stage_data.sh."
            ),
        )
    )

    report: dict[str, Any] = {
        "app": APP_NAME,
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_root": str(paths.data_root),
        "disk": _disk_section(paths),
        "raw": _raw_section(paths, checks),
    }

    report["seg"] = _seg_section(paths, checks)
    report["multimodal"] = _multimodal_section(paths, checks)
    report["pretrained"] = _pretrained_section(paths, checks)
    report["generated"] = _generated_section(paths)

    report["checks"] = [asdict(c) for c in checks]
    report["ok"] = not any(c.blocking for c in checks)
    report["blocking_failures"] = [c.name for c in checks if c.blocking]
    report["warnings"] = [c.name for c in checks if not c.ok and c.severity == SEVERITY_WARN]

    logger.info(
        "verify-data: ok=%s, %d checks, %d blocking, %d warnings",
        report["ok"],
        len(checks),
        len(report["blocking_failures"]),
        len(report["warnings"]),
    )
    return report
