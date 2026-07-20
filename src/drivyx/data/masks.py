"""AutoNUE level3Id mask generation (CLAUDE.md section 7).

Wraps the vendored `preperation/createLabels.py` and runs it as a subprocess, per section 7
("run its preperation/createLabels.py with --id-type level3Id --num-workers 12"). It is not
imported as a library because it drives a multiprocessing Pool off module-level globals set
in `__main__`, which is only safe as its own process.

Output location. Upstream writes each mask beside its source polygon file:

    dst = fn.replace("_polygons.json", "_label{id_type}s.png")

That would write into `seg/`, which section 4 declares a read-only input, while section 4
also puts masks under `masks/`. Rather than patch upstream's path handling (which would be a
larger and more fragile patch than the seven already in third_party/PATCHES.md), this module
mirrors the polygon files into `masks/gtFine/<split>/<seq>/` as symlinks and points
`--datadir` at `masks/`. Upstream then writes its PNGs exactly where section 4 wants them and
never touches `seg/`. The symlinks are removed once the run succeeds, so `masks/` ends up
holding only the generated PNGs.

Idempotency (section 7: "skip sequences whose outputs already exist") falls out of the same
mechanism: only polygons whose mask is missing get a symlink, so upstream's glob sees exactly
the outstanding work.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from drivyx.paths import Paths

logger = logging.getLogger(__name__)

POLYGON_SUFFIX = "_polygons.json"
#: Upstream's naming: "_label{}s.png".format("level3Id"). Because it derives the mask name by
#: substituting into the polygon filename, the annotation marker survives:
#:   035471_gtFine_polygons.json -> 035471_gtFine_labellevel3Ids.png
MASK_SUFFIX = "_labellevel3Ids.png"
#: The annotation marker sitting between the frame key and the suffix above. It must come off
#: as well to recover the frame key, because the image is named 035471_leftImg8bit.png, with
#: no gtFine marker at all.
ANNOTATION_MARKER = "_gtFine"
ID_TYPE = "level3Id"

#: Section 7's documented invocation.
DEFAULT_WORKERS = 12

#: Splits carrying polygons. IDD withholds test ground truth.
ANNOTATED_SPLITS = ("train", "val")


@dataclass
class MaskPlan:
    """What a gen-masks run intends to do."""

    pending: list[tuple[Path, Path]]
    existing: int
    total: int

    @property
    def complete(self) -> bool:
        return not self.pending


def autonue_dir() -> Path:
    """Locate the vendored AutoNUE tree."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "third_party" / "autonue"
        if (candidate / "preperation" / "createLabels.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Vendored AutoNUE tooling not found. Expected "
        "third_party/autonue/preperation/createLabels.py (see third_party/PROVENANCE.txt)."
    )


def mask_path_for(polygon: Path, seg_root: Path, masks_root: Path) -> Path:
    """Where the mask for a polygon file belongs under masks/.

    Mirrors the gtFine/<split>/<seq>/ layout so upstream's in-place naming lands correctly.
    """
    relative = polygon.relative_to(seg_root)
    return (masks_root / relative).with_name(polygon.name.replace(POLYGON_SUFFIX, MASK_SUFFIX))


def plan_masks(paths: Paths) -> MaskPlan:
    """Enumerate outstanding conversions."""
    seg_root = paths.seg
    gt_root = seg_root / "gtFine"
    if not gt_root.is_dir():
        raise FileNotFoundError(
            f"No gtFine tree at {gt_root}. Run scripts/stage_data.sh, then 'drivyx verify-data'."
        )

    pending: list[tuple[Path, Path]] = []
    existing = 0
    total = 0
    for split in ANNOTATED_SPLITS:
        split_dir = gt_root / split
        if not split_dir.is_dir():
            continue
        for polygon in sorted(split_dir.glob(f"*/*{POLYGON_SUFFIX}")):
            total += 1
            mask = mask_path_for(polygon, seg_root, paths.masks)
            if mask.is_file() and mask.stat().st_size > 0:
                existing += 1
            else:
                pending.append((polygon, mask))

    logger.info(
        "gen-masks plan: %d polygons, %d masks present, %d outstanding",
        total,
        existing,
        len(pending),
    )
    return MaskPlan(pending=pending, existing=existing, total=total)


def _build_symlink_farm(
    pending: list[tuple[Path, Path]], seg_root: Path, masks_root: Path
) -> list[Path]:
    """Mirror outstanding polygon files into masks/ as symlinks.

    Returns the symlinks created, so they can be cleaned up afterwards.
    """
    links: list[Path] = []
    for polygon, _mask in pending:
        link = masks_root / polygon.relative_to(seg_root)
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(polygon)
        links.append(link)
    return links


def _clear_stale_links(masks_root: Path) -> int:
    """Remove any polygon symlinks left by an interrupted run.

    Upstream globs every *_polygons.json under the datadir, so a leftover symlink would make
    it redo finished work. Cleaning first keeps the run idempotent.
    """
    removed = 0
    gt_root = masks_root / "gtFine"
    if not gt_root.is_dir():
        return 0
    for link in gt_root.glob(f"*/*/*{POLYGON_SUFFIX}"):
        if link.is_symlink():
            link.unlink()
            removed += 1
    if removed:
        logger.debug("removed %d stale polygon symlinks", removed)
    return removed


def gen_masks(paths: Paths, *, workers: int = DEFAULT_WORKERS) -> dict[str, object]:
    """Generate level3Id masks for train and val.

    Idempotent: re-running after a complete pass does no work. Interrupting mid-run leaves
    the finished masks in place, and the next run resumes with the remainder.
    """
    autonue = autonue_dir()
    seg_root = paths.seg
    masks_root = paths.masks
    masks_root.mkdir(parents=True, exist_ok=True)

    _clear_stale_links(masks_root)
    plan = plan_masks(paths)

    if plan.complete:
        logger.info("gen-masks: all %d masks already present, nothing to do", plan.total)
        return {
            "total": plan.total,
            "generated": 0,
            "existing": plan.existing,
            "masks_root": str(masks_root),
        }

    links = _build_symlink_farm(plan.pending, seg_root, masks_root)
    logger.info("gen-masks: converting %d polygon files with %d workers", len(links), workers)

    script = autonue / "preperation" / "createLabels.py"
    cmd = [
        sys.executable,
        str(script),
        "--datadir",
        str(masks_root),
        "--id-type",
        ID_TYPE,
        "--num-workers",
        str(workers),
    ]

    # Upstream's rasteriser is CPU bound and its Pool already saturates the cores; letting
    # each worker spin up its own BLAS thread pool oversubscribes the 12 Orin cores badly.
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")

    try:
        result = subprocess.run(cmd, env=env, check=False)
    finally:
        # Always clean the farm, including on SIGINT, so masks/ holds only PNGs and the next
        # run's plan is computed from real outputs.
        for link in links:
            if link.is_symlink():
                link.unlink()
        _prune_empty_dirs(masks_root / "gtFine")

    if result.returncode != 0:
        raise RuntimeError(
            f"createLabels.py failed with exit code {result.returncode}. Command: {' '.join(cmd)}"
        )

    after = plan_masks(paths)
    generated = after.existing - plan.existing
    if not after.complete:
        raise RuntimeError(
            f"gen-masks: createLabels.py exited 0 but {len(after.pending)} masks are still "
            f"missing, e.g. {[str(p) for p, _ in after.pending[:3]]}. Refusing to report "
            "success on an incomplete conversion."
        )

    logger.info("gen-masks: generated %d masks (%d total)", generated, after.existing)
    return {
        "total": after.total,
        "generated": generated,
        "existing": plan.existing,
        "masks_root": str(masks_root),
    }


def _prune_empty_dirs(root: Path) -> None:
    """Remove directories left empty by the symlink cleanup."""
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def iter_masks(paths: Paths, split: str) -> list[Path]:
    """Every generated mask for a split, sorted."""
    split_dir = paths.masks / "gtFine" / split
    if not split_dir.is_dir():
        return []
    return sorted(split_dir.glob(f"*/*{MASK_SUFFIX}"))


def frame_key(mask: Path) -> str:
    """Recover the frame key from a mask filename.

    '035471_gtFine_labellevel3Ids.png' -> '035471'

    Both the mask suffix and the gtFine annotation marker come off: the image for this frame
    is '035471_leftImg8bit.png', which carries no gtFine marker.
    """
    name = mask.name
    if name.endswith(MASK_SUFFIX):
        name = name[: -len(MASK_SUFFIX)]
    if name.endswith(ANNOTATION_MARKER):
        name = name[: -len(ANNOTATION_MARKER)]
    return name


def image_path_for_mask(mask: Path, paths: Paths) -> Path:
    """The leftImg8bit image a mask corresponds to.

    Section 4 allows both suffixes and the staged tree mixes them: Part I ships PNG and Part
    II ships JPG (docs/DECISIONS.md D016), so both are probed and a miss aborts rather than
    silently dropping the pair.
    """
    split_seq = mask.relative_to(paths.masks / "gtFine").parent
    frame = frame_key(mask)
    base = paths.seg / "leftImg8bit" / split_seq
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = base / f"{frame}_leftImg8bit{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No leftImg8bit image for mask {mask}. Looked for {frame}_leftImg8bit"
        f"{{.png,.jpg,.jpeg}} in {base}."
    )
