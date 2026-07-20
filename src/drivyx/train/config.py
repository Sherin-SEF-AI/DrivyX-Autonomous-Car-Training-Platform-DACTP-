"""Pydantic-validated training configs (CLAUDE.md sections 7, 9.1, 9.2).

Section 7 requires "pydantic-validated YAML configs". The schema is also what the TRAIN
workspace's form is generated from (section 12.4), so field descriptions here are UI text, not
just documentation.

Every default is CLAUDE.md's stated value. Where a field has no spec'd value, the default is
chosen and the reason is in the field description.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SegDataConfig(BaseModel):
    """Section 9.1's data shapes."""

    model_config = ConfigDict(extra="forbid")

    crop_width: int = Field(768, ge=64, description="Train crop width (section 9.1: 768x384).")
    crop_height: int = Field(384, ge=64, description="Train crop height.")
    val_width: int = Field(1024, ge=64, description="Val width (section 9.1: 1024x512).")
    val_height: int = Field(512, ge=64, description="Val height.")
    #: Section 3's Orin DataLoader defaults, which are performance-critical and not arbitrary:
    #: pin_memory is False because the Orin has unified memory, so pinning copies for nothing.
    num_workers: int = Field(8, ge=0, description="DataLoader workers (section 3: 8).")
    prefetch_factor: int = Field(4, ge=1, description="Batches prefetched per worker.")
    persistent_workers: bool = Field(True, description="Keep workers alive between epochs.")
    pin_memory: bool = Field(
        False, description="Section 3: False on Orin, whose memory is already unified."
    )


class SegAugConfig(BaseModel):
    """Section 9.1: hflip 0.5, random scale 0.5-2.0, random crop, color jitter 0.4/0.4/0.4."""

    model_config = ConfigDict(extra="forbid")

    hflip_prob: float = Field(0.5, ge=0.0, le=1.0)
    scale_min: float = Field(0.5, gt=0.0)
    scale_max: float = Field(2.0, gt=0.0)
    brightness: float = Field(0.4, ge=0.0)
    contrast: float = Field(0.4, ge=0.0)
    saturation: float = Field(0.4, ge=0.0)

    @model_validator(mode="after")
    def _check_scale(self) -> SegAugConfig:
        if self.scale_min > self.scale_max:
            raise ValueError(f"scale_min {self.scale_min} exceeds scale_max {self.scale_max}")
        return self


class SegLossConfig(BaseModel):
    """Section 9.1's loss parameters."""

    model_config = ConfigDict(extra="forbid")

    ohem_thresh: float = Field(0.9, gt=0.0, lt=1.0, description="Keep pixels below this prob.")
    ohem_min_kept: int = Field(26000, ge=1, description="Floor on pixels kept per batch.")
    class_weight_cap: float = Field(10.0, gt=1.0, description="Cap at N x the min weight.")
    aux_weight: float = Field(0.4, ge=0.0, description="P-branch head weight.")
    boundary_weight: float = Field(20.0, ge=0.0, description="D-branch BCE weight.")
    boundary_aware_weight: float = Field(1.0, ge=0.0, description="Boundary-gated CE weight.")


class SegOptimConfig(BaseModel):
    """Section 9.1: SGD momentum 0.9, lr 0.01 poly decay power 0.9, weight decay 5e-4."""

    model_config = ConfigDict(extra="forbid")

    lr: float = Field(0.01, gt=0.0)
    momentum: float = Field(0.9, ge=0.0, lt=1.0)
    weight_decay: float = Field(5e-4, ge=0.0)
    poly_power: float = Field(0.9, gt=0.0, description="Poly LR decay exponent.")
    warmup_iters: int = Field(
        0,
        ge=0,
        description=(
            "Linear warmup steps. Zero by default: section 9.1 does not specify warmup, and "
            "adding one unasked would change the schedule the spec defines."
        ),
    )


class SegConfig(BaseModel):
    """The full seg training config (section 9.1)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["seg"] = "seg"
    tag: str = Field("default", description="Suffix for the run directory name.")
    seed: int = Field(0xD819, description="Section 6.3 requires the seed be recorded.")

    num_classes: int = Field(8, ge=2, description="Section 7's collapsed classes.")
    batch_size: int = Field(16, ge=1, description="Section 9.1: 16.")
    epochs: int = Field(220, ge=1, description="Section 9.1: 220.")

    val_every: int = Field(5, ge=1, description="Section 9.1: track best val mIoU every 5 epochs.")
    overlays_per_val: int = Field(
        4, ge=0, description="Val overlays emitted as image events per validation."
    )

    data: SegDataConfig = Field(default_factory=SegDataConfig)
    aug: SegAugConfig = Field(default_factory=SegAugConfig)
    loss: SegLossConfig = Field(default_factory=SegLossConfig)
    optim: SegOptimConfig = Field(default_factory=SegOptimConfig)

    #: Probe sizes from section 9.1: "one epoch at each of 640x320, 768x384, 1024x512".
    probe_sizes: list[tuple[int, int]] = Field(
        default_factory=lambda: [(640, 320), (768, 384), (1024, 512)]
    )
    probe_batches: int = Field(
        40,
        ge=1,
        description=(
            "Batches timed per probe size. Section 9.1 says 'one epoch at each', but a full "
            "877-batch epoch at three sizes would take hours to answer a question a stable "
            "steady-state rate answers in minutes; the projection is scaled to the real epoch "
            "length. See docs/DECISIONS.md D027."
        ),
    )

    def resolved_probe_sizes(self) -> list[tuple[int, int]]:
        return [tuple(s) for s in self.probe_sizes]


def load_seg_config(path: Path, overrides: dict[str, Any] | None = None) -> SegConfig:
    """Read and validate a seg config, applying CLI overrides.

    Overrides are dotted paths ("optim.lr=0.02"), applied after the file is parsed so a
    smoke run can override epochs without editing the file the run then snapshots.
    """
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a YAML mapping, got {type(raw).__name__}")

    for dotted, value in (overrides or {}).items():
        _apply_override(raw, dotted, value)

    return SegConfig(**raw)


def _apply_override(payload: dict[str, Any], dotted: str, value: Any) -> None:
    """Set payload["a"]["b"] from "a.b"."""
    keys = dotted.split(".")
    cursor = payload
    for key in keys[:-1]:
        nested = cursor.get(key)
        if not isinstance(nested, dict):
            nested = {}
            cursor[key] = nested
        cursor = nested
    cursor[keys[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    """Parse "optim.lr=0.02" into ("optim.lr", 0.02).

    Values go through the YAML scalar parser, so ints, floats, and booleans arrive as the
    types the schema expects rather than as strings pydantic would have to coerce.
    """
    if "=" not in text:
        raise ValueError(f"override {text!r} must be KEY=VALUE, e.g. epochs=3")
    key, _, raw = text.partition("=")
    return key.strip(), yaml.safe_load(raw)
