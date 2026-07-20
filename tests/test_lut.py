"""LUT resolution and coverage (CLAUDE.md sections 7, 13).

Section 13 requires two tests by name:
  - "LUT name resolution against vendored labels (cpu)"
  - "LUT full-coverage on real generated masks (device)"

The device test is the one that matters most: it loads real generated masks and asserts every
pixel maps through the LUT, which is the only way to catch a silent drift between the mask
generator's id-type and the collapse.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from drivyx.data.lut import (
    GROUP_COLORS,
    GROUPS,
    IGNORE_ID,
    IGNORE_NAMES,
    NUM_CLASSES,
    build_lut,
    known_level3_ids,
    load_labels,
    lut_array,
    normalize_name,
    read_lut,
    write_lut,
)

# --- name normalisation (section 7) -----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("road", "road"),
        ("drivable fallback", "drivablefallback"),
        ("rail track", "railtrack"),
        ("non-drivable fallback", "nondrivablefallback"),
        ("obs-str-bar-fallback", "obsstrbarfallback"),
        ("guard rail", "guardrail"),
        ("traffic sign", "trafficsign"),
        ("fallback background", "fallbackbackground"),
        ("ego vehicle", "egovehicle"),
        ("Rail_Track", "railtrack"),
        ("OBS-STR-BAR-FALLBACK", "obsstrbarfallback"),
    ],
)
def test_normalize_name(raw: str, expected: str) -> None:
    """Section 7: "lowercase, strip spaces/hyphens/underscores"."""
    assert normalize_name(raw) == expected


# --- structure --------------------------------------------------------------------------


def test_eight_classes_named_as_the_spec_lists_them() -> None:
    assert NUM_CLASSES == 8
    assert [name for name, _ in GROUPS] == [
        "drivable",
        "alt_drivable",
        "nondrivable",
        "vru",
        "twowheeler",
        "vehicle",
        "structure",
        "background",
    ]


def test_every_group_has_a_colour() -> None:
    assert len(GROUP_COLORS) == NUM_CLASSES
    for color in GROUP_COLORS:
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)


def test_group_colours_are_distinct() -> None:
    """The GUI swatch table and overlays are unreadable if two classes share a colour."""
    assert len(set(GROUP_COLORS)) == NUM_CLASSES


# --- name resolution (section 13, cpu) --------------------------------------------------


def test_vendored_labels_load() -> None:
    labels = load_labels()
    assert labels, "anue_labels.py yielded no labels"
    names = {e.normalized for e in labels}
    assert "road" in names
    assert "obsstrbarfallback" in names


def test_every_spec_name_resolves() -> None:
    """Section 7: "every listed name must resolve to a level3Id or the build aborts"."""
    labels = load_labels()
    known = {e.normalized for e in labels}

    listed = [m for _, members in GROUPS for m in members] + list(IGNORE_NAMES)
    unresolved = [m for m in listed if normalize_name(m) not in known]

    assert not unresolved, f"these section 7 names do not exist in anue_labels.py: {unresolved}"


def test_build_lut_covers_every_level3_id() -> None:
    """Section 7: "every level3Id present in anue_labels must land in exactly one group"."""
    doc = build_lut()
    labels = load_labels()

    all_ids = {e.level3_id for e in labels}
    covered = known_level3_ids(doc)

    assert all_ids == covered, f"uncovered level3Ids: {sorted(all_ids - covered)}"


def test_no_level3_id_lands_in_two_groups() -> None:
    doc = build_lut()
    seen: dict[int, int] = {}
    for group in doc["groups"]:
        for level3_id in group["level3_ids"]:
            assert level3_id not in seen, (
                f"level3Id {level3_id} is in both group {seen[level3_id]} and {group['train_id']}"
            )
            seen[level3_id] = group["train_id"]
    for level3_id in doc["ignore"]["level3_ids"]:
        assert level3_id not in seen, f"level3Id {level3_id} is both grouped and ignored"


def test_specific_mappings_from_the_real_table() -> None:
    """Pin the mappings whose correctness is not obvious by inspection.

    curb has level3Id 13, numerically adjacent to the vehicle ids (8..12) but grouped as
    nondrivable because section 7 groups by name. A range-based collapse would get this
    wrong, so it is pinned here (docs/DECISIONS.md D021).
    """
    table = lut_array(build_lut())

    assert table[0] == 0, "road -> drivable"
    assert table[1] == 1, "parking / drivable fallback -> alt_drivable"
    assert table[13] == 2, "curb -> nondrivable, despite sitting among the vehicle ids"
    assert table[4] == 3, "person / animal share level3Id 4 -> vru"
    assert table[5] == 3, "rider -> vru"
    assert table[12] == 5, "caravan / trailer / train / vehicle fallback share 12 -> vehicle"
    assert table[25] == 7, "sky / fallback background share 25 -> background"
    assert table[IGNORE_ID] == IGNORE_ID, "255 stays ignore"


def test_shared_level3_ids_are_not_conflicts() -> None:
    """person/animal share id 4 and both are vru, so the shared id must not abort the build."""
    labels = load_labels()
    by_id: dict[int, set[str]] = {}
    for entry in labels:
        by_id.setdefault(entry.level3_id, set()).add(entry.normalized)

    assert {"person", "animal"} <= by_id[4]
    assert build_lut()["id_to_train_id"]["4"] == 3


# --- lut_array --------------------------------------------------------------------------


def test_lut_array_shape_and_dtype() -> None:
    table = lut_array(build_lut())
    assert table.shape == (256,)
    assert table.dtype == np.uint8


def test_lut_array_maps_unknown_ids_to_ignore() -> None:
    """A value anue_labels never defines must degrade to ignore, not to class 0.

    pack-shards separately aborts on such a pixel; this is the second line of defence.
    """
    doc = build_lut()
    table = lut_array(doc)
    known = known_level3_ids(doc)

    for value in range(256):
        if value not in known:
            assert table[value] == IGNORE_ID, f"undefined level3Id {value} must map to ignore"


def test_lut_array_applies_as_a_gather() -> None:
    table = lut_array(build_lut())
    mask = np.array([[0, 1, 13], [4, 12, 255]], dtype=np.uint8)

    collapsed = table[mask]

    assert collapsed.tolist() == [[0, 1, 2], [3, 5, 255]]
    assert collapsed.dtype == np.uint8


# --- round trip -------------------------------------------------------------------------


def test_write_and_read_lut(tmp_path: Path) -> None:
    path = tmp_path / "lut.json"
    written = write_lut(path, build_lut())
    read = read_lut(path)

    assert read == written
    assert read["num_classes"] == NUM_CLASSES
    assert len(read["name_table"]) >= len(load_labels())


def test_read_lut_missing_points_at_build_lut(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build-lut"):
        read_lut(tmp_path / "absent.json")


def test_lut_document_is_gui_renderable() -> None:
    """Section 12.4 renders lut.json as a colour-swatched table."""
    doc = build_lut()

    for group in doc["groups"]:
        assert isinstance(group["train_id"], int)
        assert group["name"]
        assert len(group["color"]) == 3
        assert group["members"]
        assert group["level3_ids"]


# --- device: real masks (section 13) ----------------------------------------------------


@pytest.mark.device
def test_lut_covers_every_pixel_of_real_masks() -> None:
    """Section 13: "LUT full-coverage on real generated masks (device)".

    Loads 25 random real masks and asserts every pixel value maps through the LUT. This is
    the check that catches a mask generated with the wrong --id-type: such a mask still
    looks like a valid PNG and would train to garbage.
    """
    from drivyx.data.masks import iter_masks
    from drivyx.paths import get_paths

    paths = get_paths()
    doc = build_lut()
    known = known_level3_ids(doc)
    table = lut_array(doc)

    masks = iter_masks(paths, "train") + iter_masks(paths, "val")
    if not masks:
        pytest.skip("no generated masks; run 'drivyx gen-masks' first")

    rng = random.Random(0xD819)
    sample = rng.sample(masks, min(25, len(masks)))

    import cv2

    seen: set[int] = set()
    for mask_path in sample:
        level3 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert level3 is not None, f"could not decode {mask_path}"
        values = set(np.unique(level3).tolist())
        unknown = values - known
        assert not unknown, f"{mask_path} has level3Ids outside the LUT: {sorted(unknown)}"
        seen |= values

        collapsed = table[level3]
        valid = set(range(NUM_CLASSES)) | {IGNORE_ID}
        assert set(np.unique(collapsed).tolist()) <= valid

    assert 0 in seen, "no road pixels in 25 random driving masks, which cannot be right"


@pytest.mark.device
def test_real_masks_exercise_most_classes() -> None:
    """Across 25 real masks, most collapsed classes should appear.

    A LUT that mapped everything to one class would still pass the coverage test above; this
    asserts the collapse actually discriminates.
    """
    import cv2

    from drivyx.data.masks import iter_masks
    from drivyx.paths import get_paths

    paths = get_paths()
    masks = iter_masks(paths, "train") + iter_masks(paths, "val")
    if not masks:
        pytest.skip("no generated masks; run 'drivyx gen-masks' first")

    table = lut_array(build_lut())
    rng = random.Random(0xD819)
    classes: set[int] = set()
    for mask_path in rng.sample(masks, min(25, len(masks))):
        level3 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        classes |= set(np.unique(table[level3]).tolist())

    classes.discard(IGNORE_ID)
    assert len(classes) >= 6, f"only classes {sorted(classes)} appear across 25 real masks"
