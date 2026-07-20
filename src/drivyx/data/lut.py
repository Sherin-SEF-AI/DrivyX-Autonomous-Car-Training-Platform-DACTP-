"""8-class collapse LUT, resolved by name (CLAUDE.md section 7).

The grouping in section 7 is written in terms of label NAMES, while the generated masks carry
level3Ids. This module resolves one to the other against the vendored `anue_labels.py`, which
is the only authority on that mapping, and refuses to produce a LUT it cannot fully justify:

  - every name listed in the spec must resolve to a level3Id, or the build aborts naming the
    miss;
  - every level3Id present in anue_labels must land in exactly one group or in ignore, or the
    build aborts naming the orphan.

Both directions matter. The first catches a typo or an upstream rename; the second catches a
label DRIVYX forgot about, which would otherwise train as whatever group it accidentally fell
into.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Pixel value for pixels excluded from the loss (section 7).
IGNORE_ID = 255

#: The 8 collapsed training classes, verbatim from CLAUDE.md section 7. Index is the train id.
#: Names are matched after normalisation (see normalize_name), so "drivable fallback",
#: "drivablefallback", and "Drivable-Fallback" are the same label.
GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("drivable", ("road",)),
    ("alt_drivable", ("parking", "drivablefallback")),
    ("nondrivable", ("sidewalk", "railtrack", "nondrivablefallback", "curb")),
    ("vru", ("person", "rider", "animal")),
    ("twowheeler", ("motorcycle", "bicycle")),
    (
        "vehicle",
        (
            "car",
            "truck",
            "bus",
            "autorickshaw",
            "caravan",
            "trailer",
            "train",
            "vehiclefallback",
        ),
    ),
    (
        "structure",
        (
            "wall",
            "fence",
            "guardrail",
            "billboard",
            "trafficsign",
            "trafficlight",
            "pole",
            "polegroup",
            "obsstrbarfallback",
            "building",
            "bridge",
            "tunnel",
        ),
    ),
    ("background", ("vegetation", "sky", "fallbackbackground")),
)

#: Names mapped to IGNORE_ID (section 7).
IGNORE_NAMES: tuple[str, ...] = (
    "unlabeled",
    "egovehicle",
    "rectificationborder",
    "outofroi",
    "licenseplate",
)

NUM_CLASSES = len(GROUPS)

#: Display colours for the collapsed classes, used by the GUI swatch table (section 12.4) and
#: the overlay renderer. Chosen to stay recognisable against Indian road scenes and to keep
#: drivable/alt_drivable distinguishable, which a per-class IoU reader needs most.
GROUP_COLORS: tuple[tuple[int, int, int], ...] = (
    (128, 64, 128),  # drivable      road purple, the Cityscapes convention
    (81, 0, 81),  # alt_drivable  darker variant of drivable
    (244, 35, 232),  # nondrivable   sidewalk magenta
    (220, 20, 60),  # vru           crimson
    (0, 0, 230),  # twowheeler    blue
    (0, 0, 142),  # vehicle       dark blue
    (153, 153, 153),  # structure     grey
    (107, 142, 35),  # background    olive
)


def normalize_name(name: str) -> str:
    """Section 7: "lowercase, strip spaces/hyphens/underscores".

    'obs-str-bar-fallback' -> 'obsstrbarfallback'
    'Rail Track'           -> 'railtrack'
    """
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


@dataclass(frozen=True)
class LabelEntry:
    """One upstream label, reduced to what the LUT cares about."""

    name: str
    normalized: str
    level3_id: int
    color: tuple[int, int, int]


def _autonue_helpers_dir() -> Path:
    """Locate the vendored helpers/ directory."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "third_party" / "autonue" / "helpers"
        if (candidate / "anue_labels.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Vendored anue_labels.py not found. Expected it at "
        "third_party/autonue/helpers/anue_labels.py (see third_party/PROVENANCE.txt)."
    )


def load_labels() -> list[LabelEntry]:
    """Import the vendored anue_labels and extract its label table.

    The module is loaded by file path rather than as a package import: it lives outside the
    drivyx package, does a flat `from collections import namedtuple`, and must stay a
    verbatim copy of upstream (third_party/PATCHES.md), so it is not made importable as
    drivyx.third_party.
    """
    import importlib.util

    helpers = _autonue_helpers_dir()
    spec = importlib.util.spec_from_file_location("anue_labels", helpers / "anue_labels.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {helpers / 'anue_labels.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    entries = [
        LabelEntry(
            name=label.name,
            normalized=normalize_name(label.name),
            level3_id=int(label.level3Id),
            color=tuple(label.color[:3]),
        )
        for label in module.labels
    ]
    logger.debug("loaded %d labels from %s", len(entries), helpers / "anue_labels.py")
    return entries


def build_lut() -> dict[str, Any]:
    """Resolve the 8-class collapse and return the LUT document.

    Raises ValueError with the exact miss when resolution is incomplete in either direction.
    """
    labels = load_labels()
    by_normalized: dict[str, list[LabelEntry]] = {}
    for entry in labels:
        by_normalized.setdefault(entry.normalized, []).append(entry)

    # --- direction 1: every spec name must resolve ---

    resolved: dict[str, list[LabelEntry]] = {}
    missing: list[str] = []
    for group_name, member_names in GROUPS:
        for member in member_names:
            found = by_normalized.get(normalize_name(member))
            if not found:
                missing.append(f"{member!r} (group {group_name!r})")
            else:
                resolved[member] = found
    for member in IGNORE_NAMES:
        found = by_normalized.get(normalize_name(member))
        if not found:
            missing.append(f"{member!r} (ignore)")
        else:
            resolved[member] = found

    if missing:
        raise ValueError(
            "build-lut: these CLAUDE.md section 7 label names do not resolve against the "
            f"vendored anue_labels.py: {', '.join(sorted(missing))}. "
            f"Known names: {', '.join(sorted(by_normalized))}."
        )

    # --- build the id -> group table ---

    id_to_group: dict[int, int] = {}
    conflicts: list[str] = []
    name_table: list[dict[str, Any]] = []

    for train_id, (group_name, member_names) in enumerate(GROUPS):
        for member in member_names:
            for entry in resolved[member]:
                previous = id_to_group.get(entry.level3_id)
                if previous is not None and previous != train_id:
                    conflicts.append(
                        f"level3Id {entry.level3_id} (via {entry.name!r}) is claimed by both "
                        f"group {previous} ({GROUPS[previous][0]}) and group {train_id} "
                        f"({group_name})"
                    )
                id_to_group[entry.level3_id] = train_id
                name_table.append(
                    {
                        "name": entry.name,
                        "normalized": entry.normalized,
                        "level3Id": entry.level3_id,
                        "train_id": train_id,
                        "group": group_name,
                    }
                )

    for member in IGNORE_NAMES:
        for entry in resolved[member]:
            previous = id_to_group.get(entry.level3_id)
            if previous is not None and previous != IGNORE_ID:
                conflicts.append(
                    f"level3Id {entry.level3_id} (via {entry.name!r}) is claimed by group "
                    f"{previous} ({GROUPS[previous][0]}) and also by ignore"
                )
            id_to_group[entry.level3_id] = IGNORE_ID
            name_table.append(
                {
                    "name": entry.name,
                    "normalized": entry.normalized,
                    "level3Id": entry.level3_id,
                    "train_id": IGNORE_ID,
                    "group": "ignore",
                }
            )

    if conflicts:
        raise ValueError(
            "build-lut: the section 7 grouping is inconsistent with anue_labels.py because "
            "these level3Ids fall in more than one group: " + "; ".join(sorted(set(conflicts)))
        )

    # --- direction 2: every level3Id in anue_labels must be covered ---

    all_ids = {e.level3_id for e in labels}
    orphans = sorted(all_ids - set(id_to_group))
    if orphans:
        orphan_names = {i: sorted({e.name for e in labels if e.level3_id == i}) for i in orphans}
        raise ValueError(
            "build-lut: these level3Ids exist in anue_labels.py but are in no group and not "
            f"ignored: {orphan_names}. Section 7 requires every level3Id to land in exactly "
            "one group or ignore."
        )

    document: dict[str, Any] = {
        "num_classes": NUM_CLASSES,
        "ignore_id": IGNORE_ID,
        "groups": [
            {
                "train_id": i,
                "name": name,
                "color": list(GROUP_COLORS[i]),
                "members": list(members),
                "level3_ids": sorted(lid for lid, g in id_to_group.items() if g == i),
            }
            for i, (name, members) in enumerate(GROUPS)
        ],
        "ignore": {
            "train_id": IGNORE_ID,
            "members": list(IGNORE_NAMES),
            "level3_ids": sorted(lid for lid, g in id_to_group.items() if g == IGNORE_ID),
        },
        # id -> train id, as a plain dict with string keys for JSON. Consumers wanting an
        # array should call lut_array().
        "id_to_train_id": {str(k): v for k, v in sorted(id_to_group.items())},
        "name_table": sorted(name_table, key=lambda r: (r["train_id"], r["level3Id"], r["name"])),
        "source": "third_party/autonue/helpers/anue_labels.py",
    }
    logger.info(
        "build-lut: %d labels -> %d classes + ignore; %d distinct level3Ids covered",
        len(labels),
        NUM_CLASSES,
        len(id_to_group),
    )
    return document


def lut_array(document: dict[str, Any] | None = None) -> np.ndarray:
    """A 256-entry uint8 lookup table: mask_level3Id -> train id.

    Applying the collapse is then `lut[mask]`, a single vectorised gather. Every level3Id not
    named by anue_labels maps to IGNORE_ID, so a corrupt pixel value degrades to ignore
    rather than silently becoming class 0. pack-shards separately asserts that no such pixel
    exists (section 7's coverage requirement).
    """
    doc = document if document is not None else build_lut()
    table = np.full(256, IGNORE_ID, dtype=np.uint8)
    for level3_id, train_id in doc["id_to_train_id"].items():
        table[int(level3_id)] = train_id
    return table


def known_level3_ids(document: dict[str, Any]) -> set[int]:
    """Every level3Id the LUT explicitly maps, ignore included."""
    return {int(k) for k in document["id_to_train_id"]}


def write_lut(path: Path, document: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write masks/lut.json (section 7)."""
    doc = document if document is not None else build_lut()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    logger.info("wrote %s", path)
    return doc


def read_lut(path: Path) -> dict[str, Any]:
    """Read masks/lut.json, failing with a pointer to build-lut when absent."""
    if not path.is_file():
        raise FileNotFoundError(f"LUT not found at {path}. Run 'drivyx build-lut' first.")
    return json.loads(path.read_text())
