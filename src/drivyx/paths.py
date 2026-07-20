"""Filesystem layout, loaded from configs/paths.yaml and validated with pydantic.

CLAUDE.md section 4 fixes the directory names under the data root. Only `data_root` and
`archive_source` are configurable; the subdirectory names are part of the spec and are
derived, not settable, so a typo in YAML cannot silently point a stage at the wrong tree.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

#: Environment override for the config file location, used by tests and by the GUI when it
#: launches CLI subprocesses with a non-default layout.
PATHS_ENV_VAR = "DRIVYX_PATHS_CONFIG"

_DEFAULT_CONFIG_RELPATH = Path("configs") / "paths.yaml"


class Paths(BaseModel):
    """Resolved data layout.

    Subdirectory names come from CLAUDE.md section 4 and are intentionally properties
    rather than fields.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    data_root: Path
    archive_source: Path

    @field_validator("data_root", "archive_source", mode="before")
    @classmethod
    def _expand(cls, value: object) -> object:
        """Expand ~ and $VARS so a config written by hand behaves like a shell path."""
        if isinstance(value, str):
            return Path(os.path.expandvars(value)).expanduser()
        return value

    # Inputs (read-only; populated by scripts/stage_data.sh).

    @property
    def raw(self) -> Path:
        """Original archives. Never deleted by code (section 4)."""
        return self.data_root / "raw"

    @property
    def seg(self) -> Path:
        """Extracted IDD Segmentation 20k Parts I+II."""
        return self.data_root / "seg"

    @property
    def multimodal(self) -> Path:
        """Extracted IDD Multimodal. Internal layout is discovered by mm-inventory."""
        return self.data_root / "multimodal"

    @property
    def pretrained(self) -> Path:
        """User-supplied PIDNet-S ImageNet backbone lives here."""
        return self.data_root / "pretrained"

    # Generated artifacts.

    @property
    def shards(self) -> Path:
        return self.data_root / "shards"

    @property
    def masks(self) -> Path:
        return self.data_root / "masks"

    @property
    def waypoints(self) -> Path:
        return self.data_root / "waypoints"

    @property
    def runs(self) -> Path:
        return self.data_root / "runs"

    @property
    def export(self) -> Path:
        return self.data_root / "export"

    # Named files with cross-module readers.

    @property
    def lut_json(self) -> Path:
        return self.masks / "lut.json"

    @property
    def mm_manifest(self) -> Path:
        return self.multimodal / "mm_manifest.json"

    @property
    def shard_index(self) -> Path:
        return self.shards / "index.json"

    def generated_dirs(self) -> tuple[Path, ...]:
        """Directories DRIVYX creates and owns. Inputs are excluded on purpose."""
        return (self.shards, self.masks, self.waypoints, self.runs, self.export)


def _default_config_path() -> Path:
    """Locate configs/paths.yaml by walking up from this file to the repo root.

    Works from a source checkout and from an installed package alike; falls back to the
    current working directory when the package is installed outside a checkout.
    """
    override = os.environ.get(PATHS_ENV_VAR)
    if override:
        return Path(override).expanduser()

    for parent in Path(__file__).resolve().parents:
        candidate = parent / _DEFAULT_CONFIG_RELPATH
        if candidate.is_file():
            return candidate
    return Path.cwd() / _DEFAULT_CONFIG_RELPATH


def load_paths(config_path: Path | None = None) -> Paths:
    """Read and validate the path config.

    Raises FileNotFoundError when the config is absent and ValueError when it is malformed,
    both with the resolved path named, per the fail-loudly rule.
    """
    path = config_path or _default_config_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Path config not found at {path}. Copy configs/paths.yaml into place or set "
            f"{PATHS_ENV_VAR} to its location."
        )

    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Path config {path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Path config {path} must be a YAML mapping, got {type(raw).__name__}.")

    return Paths(**raw)


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    """Process-wide cached paths for CLI entry points."""
    return load_paths()
