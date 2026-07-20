"""verify-data inventory and integrity checks (CLAUDE.md sections 6.1 and 7).

Fixtures build real directory trees on disk rather than substituting the filesystem, so the
scanning code under test runs exactly as it does on the device.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from drivyx.data.verify import (
    _stem_without_marker,
    verify_data,
)
from drivyx.paths import Paths


def _check(report: dict, name: str) -> dict:
    for check in report["checks"]:
        if check["name"] == name:
            return check
    raise AssertionError(
        f"check {name!r} not in report; have {[c['name'] for c in report['checks']]}"
    )


def _make_seg_tree(
    root: Path,
    *,
    counts: dict[str, int],
    sequences: int = 2,
    drop_polygons: int = 0,
    orphan_polygons: int = 0,
    suffix: str = ".png",
) -> None:
    """Build a segmentation tree with the IDD naming convention."""
    for split, total in counts.items():
        per_seq = max(1, total // sequences)
        made = 0
        for seq in range(sequences):
            seq_id = f"{seq:03d}"
            img_dir = root / "leftImg8bit" / split / seq_id
            img_dir.mkdir(parents=True, exist_ok=True)
            gt_dir = root / "gtFine" / split / seq_id
            if split in ("train", "val"):
                gt_dir.mkdir(parents=True, exist_ok=True)

            for _ in range(per_seq):
                if made >= total:
                    break
                key = f"frame{made:05d}"
                (img_dir / f"{key}_leftImg8bit{suffix}").write_bytes(b"\x89PNG")
                if split in ("train", "val"):
                    (gt_dir / f"{key}_gtFine_polygons.json").write_text("{}")
                made += 1

    # Remove polygons from otherwise-paired frames to simulate a broken export.
    if drop_polygons:
        polys = sorted((root / "gtFine" / "train").rglob("*_polygons.json"))
        for p in polys[:drop_polygons]:
            p.unlink()

    # Add polygons with no matching image.
    if orphan_polygons:
        gt_dir = root / "gtFine" / "train" / "000"
        gt_dir.mkdir(parents=True, exist_ok=True)
        for i in range(orphan_polygons):
            (gt_dir / f"orphan{i:03d}_gtFine_polygons.json").write_text("{}")


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "idd"
    root.mkdir()
    return root


@pytest.fixture
def paths(data_root: Path, tmp_path: Path) -> Paths:
    return Paths(data_root=data_root, archive_source=tmp_path / "downloads")


# --- stem parsing -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "marker", "expected"),
    [
        ("820516_leftImg8bit.png", "_leftImg8bit", "820516"),
        ("820516_leftImg8bit.jpg", "_leftImg8bit", "820516"),
        ("frame0149_gtFine_polygons.json", "_gtFine_polygons", "frame0149"),
        ("frame0149.png", "_leftImg8bit", "frame0149"),
    ],
)
def test_stem_without_marker(filename: str, marker: str, expected: str) -> None:
    assert _stem_without_marker(filename, marker) == expected


# --- missing data -----------------------------------------------------------------------


def test_absent_seg_tree_blocks(paths: Paths) -> None:
    report = verify_data(paths)

    assert report["ok"] is False
    assert "seg.extracted" in report["blocking_failures"]
    assert "stage_data.sh" in _check(report, "seg.extracted")["detail"]


def test_report_is_json_serialisable(paths: Paths) -> None:
    """The CLI writes this straight to stdout for `| python -m json.tool`."""
    report = verify_data(paths)
    assert json.loads(json.dumps(report))["app"] == "DRIVYX"


# --- healthy tree -----------------------------------------------------------------------


def test_paired_tree_passes(paths: Paths, data_root: Path) -> None:
    _make_seg_tree(data_root / "seg", counts={"train": 20, "val": 6, "test": 8})
    (data_root / "multimodal" / "primary").mkdir(parents=True)
    (data_root / "multimodal" / "primary" / "a.jpg").write_bytes(b"x")

    report = verify_data(paths)

    assert _check(report, "seg.pairing.train")["ok"] is True
    assert _check(report, "seg.pairing.val")["ok"] is True
    assert report["seg"]["splits"]["train"]["images"] == 20
    assert report["seg"]["splits"]["train"]["polygons"] == 20
    assert report["seg"]["total_images"] == 34
    # test carries images only; IDD withholds its ground truth.
    assert report["seg"]["splits"]["test"]["polygons"] == 0


def test_jpg_and_png_both_counted(paths: Paths, data_root: Path) -> None:
    """Section 4: images are '*.jpg|png' and IDD mixes the two across parts."""
    _make_seg_tree(data_root / "seg", counts={"train": 4}, sequences=1, suffix=".jpg")

    report = verify_data(paths)

    assert report["seg"]["splits"]["train"]["images"] == 4
    assert report["seg"]["splits"]["train"]["image_suffixes"] == {".jpg": 4}


# --- integrity failures -----------------------------------------------------------------


def test_missing_polygons_fail_loudly(paths: Paths, data_root: Path) -> None:
    _make_seg_tree(data_root / "seg", counts={"train": 10, "val": 4}, drop_polygons=3)

    report = verify_data(paths)

    check = _check(report, "seg.pairing.train")
    assert check["ok"] is False
    assert report["ok"] is False
    assert report["seg"]["splits"]["train"]["n_images_without_polygons"] == 3


def test_orphan_polygons_fail_loudly(paths: Paths, data_root: Path) -> None:
    _make_seg_tree(data_root / "seg", counts={"train": 10, "val": 4}, orphan_polygons=2)

    report = verify_data(paths)

    assert _check(report, "seg.pairing.train")["ok"] is False
    assert report["seg"]["splits"]["train"]["n_polygons_without_images"] == 2


def test_unpaired_listing_is_truncated_but_counted(paths: Paths, data_root: Path) -> None:
    """A systematic failure must not produce a thousand-line report."""
    _make_seg_tree(data_root / "seg", counts={"train": 60}, sequences=1, drop_polygons=40)

    report = verify_data(paths)
    split = report["seg"]["splits"]["train"]

    assert split["n_images_without_polygons"] == 40
    assert len(split["images_without_polygons"]) == 20


def test_wrong_scale_data_warns_on_counts(paths: Paths, data_root: Path) -> None:
    """A tiny tree is a count warning, not a pairing error: the data is consistent but wrong."""
    _make_seg_tree(data_root / "seg", counts={"train": 10, "val": 4, "test": 4})

    report = verify_data(paths)

    assert _check(report, "seg.count.train")["ok"] is False
    assert _check(report, "seg.count.train")["severity"] == "warn"
    assert _check(report, "seg.pairing.train")["ok"] is True
    # Count drift alone must not block: it is a warning, and blocking is for integrity.
    assert "seg.count.train" in report["warnings"]


# --- pretrained backbone ----------------------------------------------------------------


def test_absent_backbone_warns_with_hint(paths: Paths, data_root: Path) -> None:
    """Section 4: verify-data 'checks presence and prints the download hint if absent'."""
    _make_seg_tree(data_root / "seg", counts={"train": 4})

    report = verify_data(paths)
    check = _check(report, "pretrained.backbone")

    assert check["ok"] is False
    assert check["severity"] == "warn"
    assert "PIDNet" in check["detail"]
    assert "PIDNet_S_ImageNet.pth.tar" in report["pretrained"]["hint"]
    # Absence must not block M0 through M3.
    assert "pretrained.backbone" not in report["blocking_failures"]


def test_backbone_found(paths: Paths, data_root: Path) -> None:
    pretrained = data_root / "pretrained"
    pretrained.mkdir(parents=True)
    (pretrained / "PIDNet_S_ImageNet.pth.tar").write_bytes(b"\x00" * 2048)

    report = verify_data(paths)

    assert report["pretrained"]["present"] is True
    assert report["pretrained"]["bytes"] == 2048
    assert _check(report, "pretrained.backbone")["ok"] is True


# --- multimodal -------------------------------------------------------------------------


def test_multimodal_is_inventoried_not_interpreted(paths: Paths, data_root: Path) -> None:
    """Section 8 makes mm-inventory the sole authority on multimodal layout."""
    for route in ("d0", "d1"):
        d = data_root / "multimodal" / "primary" / route
        d.mkdir(parents=True)
        (d / "0000001.jpg").write_bytes(b"x" * 10)
        (d / "train.csv").write_text("timestamp,image_idx\n")

    report = verify_data(paths)

    assert report["multimodal"]["present"] is True
    assert report["multimodal"]["file_count"] == 4
    assert report["multimodal"]["extensions"] == {".jpg": 2, ".csv": 2}
    assert report["multimodal"]["top_level"] == ["primary"]
    assert sorted(report["multimodal"]["second_level"]["primary"]) == ["d0", "d1"]
    # The report must not claim to know what these files mean.
    assert "mm-inventory" in report["multimodal"]["note"]


def test_absent_multimodal_blocks(paths: Paths, data_root: Path) -> None:
    _make_seg_tree(data_root / "seg", counts={"train": 4})

    report = verify_data(paths)

    assert _check(report, "multimodal.extracted")["ok"] is False
    assert "multimodal.extracted" in report["blocking_failures"]
