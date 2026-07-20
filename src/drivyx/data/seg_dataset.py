"""Torch Dataset over the WebDataset shards (CLAUDE.md sections 7, 9.1).

The shards are tars of (jpg, png) pairs. This reads them as a map-style Dataset rather than
webdataset's iterable pipeline, for one reason: section 9.1 needs deterministic, resumable
epochs and a stable val set, and an iterable dataset with `num_workers=8` shards the stream by
worker, which makes "epoch 3, step 40" mean different data on a resume. A map-style dataset
with a seeded sampler reproduces exactly.

The tars stay memory-mapped rather than decoded up front: 2.5 GB of JPEG decodes to ~30 GB,
which would not fit alongside training.
"""

from __future__ import annotations

import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from drivyx.data.shards import augment, letterbox, normalize, shard_paths
from drivyx.paths import Paths

logger = logging.getLogger(__name__)

# OpenCV spawns its own thread pool per worker, which oversubscribes the Orin's 12 cores when
# multiplied by 8 DataLoader workers and makes decoding slower than single-threaded.
cv2.setNumThreads(0)


@dataclass(frozen=True)
class SampleRef:
    """Where one sample lives: which shard, and the byte offsets of its two members."""

    shard: int
    key: str
    image_offset: int
    image_size: int
    mask_offset: int
    mask_size: int


def _index_shard(path: Path, shard_id: int) -> list[SampleRef]:
    """Read a tar's member table once, so __getitem__ can seek directly.

    tarfile walks the archive linearly to find a member, which would make random access O(n)
    per sample. Recording the offsets up front turns it into a seek.
    """
    pending: dict[str, dict[str, tuple[int, int]]] = {}
    with tarfile.open(path, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            key, _, ext = member.name.rpartition(".")
            if ext not in ("jpg", "png"):
                continue
            pending.setdefault(key, {})[ext] = (member.offset_data, member.size)

    refs: list[SampleRef] = []
    for key in sorted(pending):
        entry = pending[key]
        if "jpg" not in entry or "png" not in entry:
            raise ValueError(
                f"{path}: sample {key!r} is missing its "
                f"{'mask' if 'jpg' in entry else 'image'}. Re-run 'drivyx pack-shards'."
            )
        refs.append(
            SampleRef(
                shard=shard_id,
                key=key,
                image_offset=entry["jpg"][0],
                image_size=entry["jpg"][1],
                mask_offset=entry["png"][0],
                mask_size=entry["png"][1],
            )
        )
    return refs


class SegShardDataset(Dataset):
    """Images and collapsed masks from the packed shards.

    In train mode applies section 9.1's augmentation; in val mode letterboxes to the
    configured size and does nothing else, because a val number computed under augmentation
    measures the augmentation.
    """

    def __init__(
        self,
        paths: Paths,
        split: str,
        *,
        train: bool,
        crop_w: int = 768,
        crop_h: int = 384,
        val_w: int = 1024,
        val_h: int = 512,
        aug: dict[str, float] | None = None,
        seed: int = 0,
    ) -> None:
        self.shards = shard_paths(paths, split)
        if not self.shards:
            raise FileNotFoundError(
                f"no shards for split {split!r} under {paths.shards / split}. "
                f"Run 'drivyx pack-shards --split {split}'."
            )
        self.train = train
        self.crop_w, self.crop_h = crop_w, crop_h
        self.val_w, self.val_h = val_w, val_h
        self.aug = aug or {}
        self.seed = seed
        self._handles: dict[int, object] = {}

        self.refs: list[SampleRef] = []
        for shard_id, shard in enumerate(self.shards):
            self.refs.extend(_index_shard(shard, shard_id))
        logger.info("%s: %d samples across %d shards", split, len(self.refs), len(self.shards))

    def __len__(self) -> int:
        return len(self.refs)

    def _read(self, ref: SampleRef) -> tuple[np.ndarray, np.ndarray]:
        # One file handle per shard per worker, opened lazily: opening in __init__ would not
        # survive the fork into DataLoader workers.
        handle = self._handles.get(ref.shard)
        if handle is None:
            handle = open(self.shards[ref.shard], "rb")
            self._handles[ref.shard] = handle

        handle.seek(ref.image_offset)
        image_bytes = handle.read(ref.image_size)
        handle.seek(ref.mask_offset)
        mask_bytes = handle.read(ref.mask_size)

        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"could not decode sample {ref.key} from {self.shards[ref.shard]}")
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ref = self.refs[index]
        image, mask = self._read(ref)

        if self.train:
            # Seeded per (epoch-independent) index so a worker's augmentation is reproducible
            # given the same seed, which is what makes a resumed run comparable.
            rng = np.random.default_rng((self.seed * 1_000_003 + index) & 0xFFFFFFFF)
            image, mask = augment(
                image,
                mask,
                crop_w=self.crop_w,
                crop_h=self.crop_h,
                hflip_prob=self.aug.get("hflip_prob", 0.5),
                scale_min=self.aug.get("scale_min", 0.5),
                scale_max=self.aug.get("scale_max", 2.0),
                brightness=self.aug.get("brightness", 0.4),
                contrast=self.aug.get("contrast", 0.4),
                saturation=self.aug.get("saturation", 0.4),
                rng=rng,
            )
        else:
            image, mask = letterbox(image, mask, self.val_w, self.val_h)

        return (
            torch.from_numpy(normalize(image)),
            torch.from_numpy(np.ascontiguousarray(mask)).long(),
        )

    def set_epoch(self, epoch: int) -> None:
        """Vary the augmentation across epochs while staying reproducible.

        Without this every epoch would apply the identical crop and flip to each sample, which
        turns 220 epochs of augmentation into one.
        """
        self.seed = self.seed ^ (epoch * 0x9E3779B1)

    def __del__(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
