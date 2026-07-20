"""Path config loading and validation (CLAUDE.md sections 4 and 7)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from drivyx.paths import Paths, load_paths


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    config = tmp_path / "paths.yaml"
    config.write_text(yaml.safe_dump(payload))
    return config


def test_loads_and_derives_subdirectories(tmp_path: Path) -> None:
    config = _write_config(tmp_path, {"data_root": "/data/idd", "archive_source": "/downloads"})
    paths = load_paths(config)

    assert paths.data_root == Path("/data/idd")
    # Section 4 fixes these names; they are derived, not configurable.
    assert paths.seg == Path("/data/idd/seg")
    assert paths.multimodal == Path("/data/idd/multimodal")
    assert paths.raw == Path("/data/idd/raw")
    assert paths.pretrained == Path("/data/idd/pretrained")
    assert paths.shards == Path("/data/idd/shards")
    assert paths.masks == Path("/data/idd/masks")
    assert paths.waypoints == Path("/data/idd/waypoints")
    assert paths.runs == Path("/data/idd/runs")
    assert paths.export == Path("/data/idd/export")


def test_named_files_resolve_under_their_owners(tmp_path: Path) -> None:
    config = _write_config(tmp_path, {"data_root": "/d", "archive_source": "/s"})
    paths = load_paths(config)

    assert paths.lut_json == Path("/d/masks/lut.json")
    assert paths.mm_manifest == Path("/d/multimodal/mm_manifest.json")
    assert paths.shard_index == Path("/d/shards/index.json")


def test_expands_user_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRIVYX_TEST_ROOT", "/expanded")
    config = _write_config(
        tmp_path, {"data_root": "$DRIVYX_TEST_ROOT/idd", "archive_source": "~/dl"}
    )
    paths = load_paths(config)

    assert paths.data_root == Path("/expanded/idd")
    assert paths.archive_source == Path.home() / "dl"


def test_missing_config_names_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "absent.yaml"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        load_paths(missing)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    """extra='forbid': a typo must fail loudly rather than be silently ignored."""
    config = _write_config(
        tmp_path, {"data_root": "/d", "archive_source": "/s", "dataroot": "/typo"}
    )
    with pytest.raises(ValueError, match="dataroot"):
        load_paths(config)


def test_missing_required_key_is_rejected(tmp_path: Path) -> None:
    config = _write_config(tmp_path, {"data_root": "/d"})
    with pytest.raises(ValueError, match="archive_source"):
        load_paths(config)


def test_malformed_yaml_names_the_file(tmp_path: Path) -> None:
    config = tmp_path / "paths.yaml"
    config.write_text("data_root: [unclosed\n")
    with pytest.raises(ValueError, match="not valid YAML"):
        load_paths(config)


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "paths.yaml"
    config.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_paths(config)


def test_paths_are_frozen() -> None:
    """Paths is frozen so a caller cannot repoint a stage's data root mid-run."""
    paths = Paths(data_root=Path("/a"), archive_source=Path("/b"))
    with pytest.raises(ValidationError, match="frozen"):
        paths.data_root = Path("/c")  # type: ignore[misc]


def test_repo_default_config_is_valid() -> None:
    """The checked-in configs/paths.yaml must always load."""
    paths = load_paths()
    assert paths.data_root.is_absolute()
    assert paths.archive_source.is_absolute()
