"""WebDataset shard packing and reading (CLAUDE.md section 7).

Writer: tars of (jpg image resized so short side = 512 keeping aspect, png collapsed mask
nearest-neighbour), roughly 500 samples per shard, plus `shards/index.json` carrying counts
and a per-class pixel histogram used for the loss class weights (section 9.1).

Reader: implements the section 9.1 training augmentation on the fly.

Resampling choice is load-bearing and not symmetric: the image is resized with area/bilinear
interpolation, the mask with nearest neighbour. Interpolating a label map would invent class
ids that were never annotated (a pixel between road=0 and sidewalk=2 would become 1, which is
alt_drivable), so the mask must never be interpolated.
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from drivyx.data.lut import IGNORE_ID, NUM_CLASSES, known_level3_ids, lut_array
from drivyx.data.masks import frame_key, iter_masks
from drivyx.paths import Paths

logger = logging.getLogger(__name__)

#: Section 7: "resized so short side = 512 keeping aspect".
SHORT_SIDE = 512
#: Section 7: "~500 samples per shard".
SAMPLES_PER_SHARD = 500
#: JPEG quality for the packed image. 95 keeps recompression artifacts well below the
#: annotation's own boundary precision while roughly halving shard size against 100.
JPEG_QUALITY = 95

SPLITS = ("train", "val")


@dataclass
class ShardStats:
    """Accounting for one pack-shards run."""

    samples: int = 0
    shards: int = 0
    bytes_written: int = 0
    #: Pixel count per collapsed class, index 0..NUM_CLASSES-1, plus ignore tracked apart.
    class_pixels: list[int] = field(default_factory=lambda: [0] * NUM_CLASSES)
    ignore_pixels: int = 0
    sequences: set[str] = field(default_factory=set)

    @property
    def total_labelled_pixels(self) -> int:
        return sum(self.class_pixels)


def resize_short_side(image: np.ndarray, short_side: int, *, nearest: bool) -> np.ndarray:
    """Resize so the short side equals `short_side`, preserving aspect ratio.

    Nearest neighbour for masks (a label id must survive resampling unchanged); area
    interpolation for images, which is the correct choice for downscaling and avoids the
    aliasing bilinear leaves behind.
    """
    h, w = image.shape[:2]
    if min(h, w) == short_side:
        return image
    scale = short_side / min(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def _encode_jpeg(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed for jpg")
    return buf.tobytes()


def _encode_png(mask: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError("cv2.imencode failed for png")
    return buf.tobytes()


def _collapse(level3: np.ndarray, table: np.ndarray, known: set[int], source: Path) -> np.ndarray:
    """Apply the LUT, aborting on any pixel the LUT does not know.

    Section 7 requires every level3Id present to map; a pixel outside the LUT means the mask
    was produced by different tooling or a different id-type, and silently folding it into
    ignore would corrupt training while looking fine.
    """
    present = set(np.unique(level3).tolist())
    unknown = present - known
    if unknown:
        raise ValueError(
            f"{source} contains level3Ids {sorted(unknown)} that are not in masks/lut.json. "
            "Either the mask was generated with a different --id-type, or anue_labels.py "
            "changed and build-lut must be re-run."
        )
    return table[level3]


def pack_split(
    paths: Paths,
    split: str,
    *,
    samples_per_shard: int = SAMPLES_PER_SHARD,
    short_side: int = SHORT_SIDE,
    limit: int | None = None,
) -> ShardStats:
    """Pack one split into WebDataset tars and return its statistics."""
    masks = iter_masks(paths, split)
    if not masks:
        raise FileNotFoundError(
            f"No masks for split {split!r} under {paths.masks / 'gtFine' / split}. "
            "Run 'drivyx gen-masks' first."
        )
    if limit is not None:
        masks = masks[:limit]

    from drivyx.data.lut import read_lut
    from drivyx.data.masks import image_path_for_mask

    document = read_lut(paths.lut_json)
    table = lut_array(document)
    known = known_level3_ids(document)

    out_dir = paths.shards / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.tar"):
        stale.unlink()

    stats = ShardStats()
    shard_index = 0
    tar: tarfile.TarFile | None = None
    in_shard = 0

    def open_shard(index: int) -> tarfile.TarFile:
        path = out_dir / f"{split}-{index:05d}.tar"
        logger.debug("opening %s", path)
        return tarfile.open(path, "w")

    def add(tf: tarfile.TarFile, name: str, payload: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    try:
        for mask_path in masks:
            if tar is None:
                tar = open_shard(shard_index)
                in_shard = 0

            image_path = image_path_for_mask(mask_path, paths)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cv2 could not decode {image_path}")
            level3 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if level3 is None:
                raise RuntimeError(f"cv2 could not decode {mask_path}")
            if image.shape[:2] != level3.shape[:2]:
                raise ValueError(
                    f"image/mask size mismatch: {image_path.name} is {image.shape[:2]} but "
                    f"{mask_path.name} is {level3.shape[:2]}"
                )

            collapsed = _collapse(level3, table, known, mask_path)
            image_r = resize_short_side(image, short_side, nearest=False)
            mask_r = resize_short_side(collapsed, short_side, nearest=True)

            counts = np.bincount(mask_r.reshape(-1), minlength=256)
            for class_id in range(NUM_CLASSES):
                stats.class_pixels[class_id] += int(counts[class_id])
            stats.ignore_pixels += int(counts[IGNORE_ID])

            sequence = mask_path.parent.name
            key = f"{sequence}_{frame_key(mask_path)}"
            add(tar, f"{key}.jpg", _encode_jpeg(image_r))
            add(tar, f"{key}.png", _encode_png(mask_r))

            stats.samples += 1
            stats.sequences.add(sequence)
            in_shard += 1
            if in_shard >= samples_per_shard:
                tar.close()
                tar = None
                shard_index += 1
    finally:
        if tar is not None:
            tar.close()

    stats.shards = len(list(out_dir.glob("*.tar")))
    stats.bytes_written = sum(p.stat().st_size for p in out_dir.glob("*.tar"))
    logger.info(
        "pack-shards %s: %d samples in %d shards (%.2f GB)",
        split,
        stats.samples,
        stats.shards,
        stats.bytes_written / 1e9,
    )
    return stats


def class_weights(class_pixels: list[int], *, cap: float = 10.0) -> list[float]:
    """Section 9.1: w = 1/log(1.02 + freq), capped at 10x the minimum weight.

    freq is each class's share of labelled pixels (ignore excluded, since those pixels never
    reach the loss). The 1.02 offset bounds the weight of a vanishingly rare class: without
    it, freq -> 0 gives log(1) = 0 and the weight diverges.
    """
    total = sum(class_pixels)
    if total == 0:
        raise ValueError("cannot compute class weights: no labelled pixels were counted")
    weights = [1.0 / np.log(1.02 + (count / total)) for count in class_pixels]
    lowest = min(weights)
    return [float(min(w, lowest * cap)) for w in weights]


def write_index(paths: Paths, per_split: dict[str, ShardStats]) -> dict[str, Any]:
    """Write shards/index.json (section 7).

    Class weights are derived from the train histogram only: val must not influence the loss,
    and section 9.1 asks for weights "from the shard histogram" of what is trained on.
    """
    from drivyx.data.lut import read_lut

    document = read_lut(paths.lut_json)
    group_names = [g["name"] for g in document["groups"]]

    index: dict[str, Any] = {
        "num_classes": NUM_CLASSES,
        "ignore_id": IGNORE_ID,
        "short_side": SHORT_SIDE,
        "samples_per_shard": SAMPLES_PER_SHARD,
        "classes": group_names,
        "splits": {},
    }
    for split, stats in per_split.items():
        labelled = stats.total_labelled_pixels
        index["splits"][split] = {
            "samples": stats.samples,
            "shards": stats.shards,
            "sequences": len(stats.sequences),
            "bytes": stats.bytes_written,
            "class_pixels": stats.class_pixels,
            "ignore_pixels": stats.ignore_pixels,
            "class_frequency": [(c / labelled) if labelled else 0.0 for c in stats.class_pixels],
            "pattern": f"{split}/{split}-{{00000..{max(stats.shards - 1, 0):05d}}}.tar",
        }

    if "train" in per_split:
        index["class_weights"] = class_weights(per_split["train"].class_pixels)

    paths.shard_index.parent.mkdir(parents=True, exist_ok=True)
    paths.shard_index.write_text(json.dumps(index, indent=2) + "\n")
    logger.info("wrote %s", paths.shard_index)
    return index


def read_index(paths: Paths) -> dict[str, Any]:
    """Read shards/index.json, pointing at pack-shards when absent."""
    if not paths.shard_index.is_file():
        raise FileNotFoundError(
            f"Shard index not found at {paths.shard_index}. Run 'drivyx pack-shards' first."
        )
    return json.loads(paths.shard_index.read_text())


def shard_paths(paths: Paths, split: str) -> list[Path]:
    """Every shard tar for a split, in order."""
    return sorted((paths.shards / split).glob("*.tar"))


def iter_samples(shard: Path) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Decode (key, image_bgr, mask) triples from one shard.

    Used by the reader and by tests. WebDataset groups members by basename, so the jpg and
    png of a sample are adjacent; this pairs them by key rather than by position so a
    reordered tar still reads correctly.
    """
    pending: dict[str, dict[str, bytes]] = {}
    with tarfile.open(shard, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            key, _, ext = member.name.rpartition(".")
            handle = tf.extractfile(member)
            if handle is None:
                continue
            pending.setdefault(key, {})[ext] = handle.read()

            entry = pending[key]
            if "jpg" in entry and "png" in entry:
                image = cv2.imdecode(np.frombuffer(entry["jpg"], np.uint8), cv2.IMREAD_COLOR)
                mask = cv2.imdecode(np.frombuffer(entry["png"], np.uint8), cv2.IMREAD_GRAYSCALE)
                del pending[key]
                yield key, image, mask

    if pending:
        raise ValueError(
            f"{shard} has {len(pending)} unpaired samples, e.g. "
            f"{sorted(pending)[:3]}. Every sample needs both a .jpg and a .png."
        )


# --- training reader (CLAUDE.md section 7: "Reader implements the training augmentation") ---

#: ImageNet statistics, in RGB. The PIDNet-S backbone was pretrained under these, so the input
#: normalisation is not a free choice: feeding it differently-scaled inputs wastes the
#: pretraining the whole D024 exercise existed to obtain.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize(image_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC -> normalised RGB float32 CHW."""
    rgb = image_bgr[:, :, ::-1].astype(np.float32) / 255.0
    rgb = (rgb - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
        IMAGENET_STD, dtype=np.float32
    )
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def augment(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    crop_w: int,
    crop_h: int,
    hflip_prob: float,
    scale_min: float,
    scale_max: float,
    brightness: float,
    contrast: float,
    saturation: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Section 9.1's augmentation: hflip, random scale, random crop, colour jitter.

    Order matters. Scale then crop, never crop then scale: cropping first would fix the field
    of view and make the scale a zoom on an already-chosen window, which is a different (and
    weaker) augmentation than sampling a window from a rescaled image.

    Colour jitter is applied to the image only, and after the geometry, so it operates on the
    pixels that actually reach the network.

    A crop larger than the scaled image is padded: the image with the ImageNet mean (which is
    zero after normalisation, so the network sees "no information") and the mask with
    IGNORE_ID, so the padding never contributes a gradient.
    """
    if rng.random() < hflip_prob:
        image = image[:, ::-1]
        mask = mask[:, ::-1]

    scale = rng.uniform(scale_min, scale_max)
    if abs(scale - 1.0) > 1e-3:
        h, w = image.shape[:2]
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        # INTER_AREA downscales without aliasing; INTER_LINEAR is the right upscale. The mask
        # is nearest either way: interpolating labels invents classes (see test_shards.py).
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        image = cv2.resize(image, (new_w, new_h), interpolation=interp)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    image, mask = _pad_to(image, mask, crop_w, crop_h)
    image, mask = _random_crop(image, mask, crop_w, crop_h, rng)
    image = _color_jitter(image, brightness, contrast, saturation, rng)
    return image, mask


def _pad_to(
    image: np.ndarray, mask: np.ndarray, crop_w: int, crop_h: int
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    pad_h, pad_w = max(0, crop_h - h), max(0, crop_w - w)
    if not (pad_h or pad_w):
        return image, mask
    # The image pads with the ImageNet mean in uint8 terms, which normalises to ~0.
    fill = tuple(int(round(255 * c)) for c in reversed(IMAGENET_MEAN))
    image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=fill)
    mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=int(IGNORE_ID))
    return image, mask


def _random_crop(
    image: np.ndarray, mask: np.ndarray, crop_w: int, crop_h: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    top = int(rng.integers(0, h - crop_h + 1))
    left = int(rng.integers(0, w - crop_w + 1))
    return (
        image[top : top + crop_h, left : left + crop_w],
        mask[top : top + crop_h, left : left + crop_w],
    )


def _color_jitter(
    image: np.ndarray,
    brightness: float,
    contrast: float,
    saturation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Brightness/contrast/saturation jitter, each sampled in [1-x, 1+x].

    Done in float32 and clipped once at the end rather than per operation, so three successive
    adjustments do not each quantise back to uint8 and compound rounding error.
    """
    if not (brightness or contrast or saturation):
        return image

    out = image.astype(np.float32)
    if brightness:
        out *= rng.uniform(max(0.0, 1 - brightness), 1 + brightness)
    if contrast:
        mean = out.mean()
        out = (out - mean) * rng.uniform(max(0.0, 1 - contrast), 1 + contrast) + mean
    if saturation:
        # Luminance in BGR order, matching cv2's channel layout.
        grey = (out * np.array([0.114, 0.587, 0.299], dtype=np.float32)).sum(axis=2, keepdims=True)
        out = (out - grey) * rng.uniform(max(0.0, 1 - saturation), 1 + saturation) + grey
    return np.clip(out, 0, 255).astype(np.uint8)


def letterbox(
    image: np.ndarray, mask: np.ndarray | None, width: int, height: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Resize to fit (width, height) preserving aspect, padding the remainder.

    Section 9.1: "val at full resized 1024x512 (letterbox as needed)". Letterboxing rather
    than stretching keeps val geometry identical to train geometry; a stretched val image
    would measure the model on an aspect ratio it never trained on.
    """
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    image = cv2.resize(image, (new_w, new_h), interpolation=interp)
    if mask is not None:
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    pad_w, pad_h = width - new_w, height - new_h
    if pad_w or pad_h:
        fill = tuple(int(round(255 * c)) for c in reversed(IMAGENET_MEAN))
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=fill)
        if mask is not None:
            mask = cv2.copyMakeBorder(
                mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=int(IGNORE_ID)
            )
    return image, mask
