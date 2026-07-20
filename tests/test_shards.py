"""Shard writing, reading, and class weights (CLAUDE.md sections 7, 9.1, 13).

Section 13 requires "shard write/read augmentation shapes (cpu)". Fixtures build real tars on
disk from real numpy arrays, so the encode/decode path under test is the one that runs on the
device.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from drivyx.data.lut import IGNORE_ID, NUM_CLASSES
from drivyx.data.shards import (
    SHORT_SIDE,
    ShardStats,
    class_weights,
    iter_samples,
    read_index,
    resize_short_side,
    shard_paths,
    write_index,
)
from drivyx.paths import Paths

# --- resize (section 7) -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("h", "w", "expected_h", "expected_w"),
    [
        (1080, 1920, 512, 910),  # 1080p landscape: short side is the height
        (720, 1280, 512, 910),  # 720p
        (512, 1024, 512, 1024),  # already at the short side: unchanged
        (1024, 512, 1024, 512),  # portrait: short side is the width
    ],
)
def test_resize_short_side_keeps_aspect(h: int, w: int, expected_h: int, expected_w: int) -> None:
    """Section 7: "resized so short side = 512 keeping aspect"."""
    image = np.zeros((h, w, 3), dtype=np.uint8)

    out = resize_short_side(image, SHORT_SIDE, nearest=False)

    assert out.shape[:2] == (expected_h, expected_w)
    assert min(out.shape[:2]) == SHORT_SIDE


def test_resize_preserves_aspect_ratio_closely() -> None:
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = resize_short_side(image, SHORT_SIDE, nearest=False)

    assert abs((out.shape[1] / out.shape[0]) - (1920 / 1080)) < 0.01


def test_mask_resize_invents_no_class_ids() -> None:
    """The load-bearing asymmetry: masks must use nearest neighbour.

    Interpolating between road=0 and nondrivable=2 would produce 1, which is alt_drivable, a
    class that was never annotated at that pixel.
    """
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:, 50:] = 2

    out = resize_short_side(mask, SHORT_SIDE, nearest=True)

    assert set(np.unique(out).tolist()) == {0, 2}, "nearest neighbour invented a class id"


def test_image_resize_uses_interpolation() -> None:
    """Images are downscaled with area interpolation, which must smooth rather than pick.

    The ratio is deliberately non-integral (100 -> 37). At an exact 2x with a grid-aligned
    edge, INTER_AREA averages two identical pixels and yields no intermediate value, so such
    a case cannot distinguish area interpolation from nearest neighbour.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, 50:] = 255

    area = resize_short_side(image, 37, nearest=False)
    nearest = resize_short_side(image, 37, nearest=True)

    assert len(np.unique(area)) > 2, "INTER_AREA produced no blended pixels"
    assert set(np.unique(nearest).tolist()) == {0, 255}, "nearest must never blend"


def test_resize_is_identity_at_target() -> None:
    image = np.zeros((512, 700, 3), dtype=np.uint8)
    assert resize_short_side(image, 512, nearest=False) is image


# --- class weights (section 9.1) --------------------------------------------------------


def test_class_weights_formula() -> None:
    """Section 9.1: w = 1/log(1.02 + freq), capped at 10x the minimum."""
    pixels = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]

    weights = class_weights(pixels)

    assert len(weights) == NUM_CLASSES
    expected = 1.0 / np.log(1.02 + 0.125)
    assert all(abs(w - expected) < 1e-9 for w in weights)


def test_rare_classes_weigh_more_than_common_ones() -> None:
    pixels = [10_000_000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]

    weights = class_weights(pixels)

    assert weights[0] < weights[1], "the dominant class must get the smallest weight"


def test_class_weights_are_capped() -> None:
    """Without the cap, a vanishingly rare class would dominate the loss."""
    pixels = [10_000_000, 1, 1, 1, 1, 1, 1, 1]

    weights = class_weights(pixels, cap=10.0)

    assert max(weights) <= min(weights) * 10.0 + 1e-9


def test_class_weights_reject_empty_histogram() -> None:
    with pytest.raises(ValueError, match="no labelled pixels"):
        class_weights([0] * NUM_CLASSES)


def test_class_weights_bounded_for_absent_class() -> None:
    """freq = 0 gives log(1.02), which is finite. The 1.02 offset exists for exactly this."""
    weights = class_weights([1000] * 7 + [0])
    assert all(np.isfinite(weights))


# --- index ------------------------------------------------------------------------------


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    return Paths(data_root=tmp_path, archive_source=tmp_path / "dl")


def _write_lut(paths: Paths) -> None:
    from drivyx.data.lut import build_lut, write_lut

    write_lut(paths.lut_json, build_lut())


def test_write_index_records_counts_and_histogram(paths: Paths) -> None:
    _write_lut(paths)
    stats = ShardStats(
        samples=100,
        shards=2,
        bytes_written=1234,
        class_pixels=[500, 100, 100, 50, 50, 100, 100, 200],
        ignore_pixels=42,
        sequences={"a", "b"},
    )

    index = write_index(paths, {"train": stats})

    assert index["num_classes"] == NUM_CLASSES
    assert index["splits"]["train"]["samples"] == 100
    assert index["splits"]["train"]["class_pixels"] == stats.class_pixels
    assert index["splits"]["train"]["ignore_pixels"] == 42
    assert len(index["class_weights"]) == NUM_CLASSES
    assert index["classes"][0] == "drivable"


def test_index_frequencies_sum_to_one(paths: Paths) -> None:
    _write_lut(paths)
    stats = ShardStats(class_pixels=[100, 200, 300, 400, 500, 600, 700, 800], samples=1)

    index = write_index(paths, {"train": stats})

    assert abs(sum(index["splits"]["train"]["class_frequency"]) - 1.0) < 1e-9


def test_index_weights_come_from_train_only(paths: Paths) -> None:
    """Section 9.1 weights the loss from the training histogram; val must not influence it."""
    _write_lut(paths)
    train = ShardStats(class_pixels=[800] + [100] * 7, samples=1)
    val = ShardStats(class_pixels=[1] + [1000] * 7, samples=1)

    index = write_index(paths, {"train": train, "val": val})

    assert index["class_weights"] == class_weights(train.class_pixels)


def test_index_round_trip(paths: Paths) -> None:
    _write_lut(paths)
    write_index(paths, {"train": ShardStats(samples=5, class_pixels=[1] * 8)})

    assert read_index(paths)["splits"]["train"]["samples"] == 5


def test_read_index_missing_points_at_pack_shards(paths: Paths) -> None:
    with pytest.raises(FileNotFoundError, match="pack-shards"):
        read_index(paths)


def test_index_is_json_serialisable(paths: Paths) -> None:
    _write_lut(paths)
    write_index(paths, {"train": ShardStats(samples=1, class_pixels=[1] * 8, sequences={"s"})})

    assert json.loads(paths.shard_index.read_text())["splits"]["train"]["sequences"] == 1


# --- shard read/write round trip (section 13) -------------------------------------------


def _make_shard(path: Path, count: int) -> None:
    """Build a real tar of encoded samples, as pack_split writes them."""
    import io

    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tf:
        for i in range(count):
            image = np.full((512, 910, 3), i, dtype=np.uint8)
            mask = np.full((512, 910), i % NUM_CLASSES, dtype=np.uint8)
            for name, payload in (
                (f"seq_{i:04d}.jpg", cv2.imencode(".jpg", image)[1].tobytes()),
                (f"seq_{i:04d}.png", cv2.imencode(".png", mask)[1].tobytes()),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))


def test_shard_round_trip_shapes(tmp_path: Path) -> None:
    """Section 13: "shard write/read augmentation shapes"."""
    shard = tmp_path / "train" / "train-00000.tar"
    _make_shard(shard, 4)

    samples = list(iter_samples(shard))

    assert len(samples) == 4
    for key, image, mask in samples:
        assert key.startswith("seq_")
        assert image.shape == (512, 910, 3)
        assert mask.shape == (512, 910)
        assert image.dtype == np.uint8
        assert mask.dtype == np.uint8


def test_shard_masks_survive_png_round_trip(tmp_path: Path) -> None:
    """PNG is lossless, so a class id must come back bit-exact.

    If this ever fails, the mask is being written as JPEG and every label is corrupted.
    """
    shard = tmp_path / "train-00000.tar"
    _make_shard(shard, NUM_CLASSES)

    for i, (_key, _image, mask) in enumerate(iter_samples(shard)):
        assert set(np.unique(mask).tolist()) == {i % NUM_CLASSES}


def test_iter_samples_rejects_unpaired(tmp_path: Path) -> None:
    """A sample missing its mask must abort, not be silently skipped."""
    import io

    import cv2

    shard = tmp_path / "broken.tar"
    with tarfile.open(shard, "w") as tf:
        payload = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
        info = tarfile.TarInfo("lonely.jpg")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="unpaired"):
        list(iter_samples(shard))


def test_shard_paths_are_ordered(tmp_path: Path) -> None:
    paths = Paths(data_root=tmp_path, archive_source=tmp_path)
    for i in (2, 0, 1):
        _make_shard(tmp_path / "shards" / "train" / f"train-{i:05d}.tar", 1)

    found = shard_paths(paths, "train")

    assert [p.name for p in found] == ["train-00000.tar", "train-00001.tar", "train-00002.tar"]


# --- device: real shards ----------------------------------------------------------------


@pytest.mark.device
def test_real_shards_decode_and_carry_valid_classes() -> None:
    from drivyx.paths import get_paths

    paths = get_paths()
    shards = shard_paths(paths, "val")
    if not shards:
        pytest.skip("no val shards; run 'drivyx pack-shards --split val' first")

    valid = set(range(NUM_CLASSES)) | {IGNORE_ID}
    checked = 0
    for key, image, mask in iter_samples(shards[0]):
        assert image is not None and mask is not None, f"failed to decode {key}"
        assert min(image.shape[:2]) == SHORT_SIDE, f"{key} short side is {min(image.shape[:2])}"
        assert image.shape[:2] == mask.shape[:2], f"{key} image/mask size mismatch"
        assert set(np.unique(mask).tolist()) <= valid, f"{key} has classes outside 0..7 + 255"
        checked += 1
        if checked >= 20:
            break

    assert checked > 0
