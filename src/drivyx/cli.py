"""DRIVYX command line interface (CLAUDE.md section 6.1).

Every capability is a subcommand here, and the GUI does nothing except run these via
QProcess (section 2). Each subcommand is a thin entry over a library function so the same
code path serves the CLI, the GUI, and the tests.

Subcommands are registered as their milestones land (docs/DECISIONS.md D012). Registering a
subcommand that errors "not implemented" would be a stub, which rule 23 forbids, so the
parser only ever offers what actually works.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from drivyx.branding import APP_NAME
from drivyx.logging_setup import configure_logging
from drivyx.paths import Paths, load_paths

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILURE = 1
#: Section 6.2: SIGINT -> graceful checkpoint -> exit 130 (128 + SIGINT).
EXIT_INTERRUPTED = 130


def _resolve_paths(args: argparse.Namespace) -> Paths:
    """Load the path config named by the global --paths-config flag."""
    return load_paths(Path(args.paths_config) if args.paths_config else None)


def _emit_json(payload: dict[str, Any]) -> None:
    """Write a report to stdout as JSON.

    Logging goes to stderr (logging_setup), so stdout stays a clean JSON stream for
    `| python -m json.tool` and for the GUI's parser.
    """
    json.dump(payload, sys.stdout, indent=2, sort_keys=False)
    sys.stdout.write("\n")
    sys.stdout.flush()


# --- verify-data (M0) -------------------------------------------------------------------


def _cmd_verify_data(args: argparse.Namespace) -> int:
    """Inventory + integrity report (sections 6.1, 7)."""
    from drivyx.data.verify import verify_data

    paths = _resolve_paths(args)
    report = verify_data(paths)
    _emit_json(report)

    if not report["ok"]:
        logger.error(
            "verify-data found blocking failures: %s", ", ".join(report["blocking_failures"])
        )
        return EXIT_FAILURE
    if report["warnings"]:
        logger.warning("verify-data warnings: %s", ", ".join(report["warnings"]))
    return EXIT_OK


def _register_verify_data(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "verify-data",
        help="Inventory and integrity report for the IDD data on disk (JSON to stdout).",
        description=(
            "Counts images per split and sequence, checks image/polygon pairing, reports the "
            "multimodal tree shallowly, and checks for the PIDNet-S backbone. Exits non-zero "
            "when a blocking failure would stop a downstream stage."
        ),
    )
    parser.set_defaults(func=_cmd_verify_data)


# --- gen-masks (M2) ---------------------------------------------------------------------


def _cmd_gen_masks(args: argparse.Namespace) -> int:
    """AutoNUE level3Id PNG generation (sections 6.1, 7)."""
    from drivyx.data.masks import gen_masks

    result = gen_masks(_resolve_paths(args), workers=args.workers)
    _emit_json(result)
    return EXIT_OK


def _register_gen_masks(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "gen-masks",
        help="Generate level3Id label PNGs from the IDD polygons via the vendored AutoNUE tooling.",
        description=(
            "Runs third_party/autonue/preperation/createLabels.py with --id-type level3Id. "
            "Idempotent: masks that already exist are skipped, so an interrupted run resumes."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Conversion processes (default: 12, the value CLAUDE.md section 7 documents).",
    )
    parser.set_defaults(func=_cmd_gen_masks)


# --- build-lut (M2) ---------------------------------------------------------------------


def _cmd_build_lut(args: argparse.Namespace) -> int:
    """Resolve the 8-class LUT and write masks/lut.json (sections 6.1, 7)."""
    from drivyx.data.lut import build_lut, write_lut

    paths = _resolve_paths(args)
    document = write_lut(paths.lut_json, build_lut())
    _emit_json(
        {
            "lut": str(paths.lut_json),
            "num_classes": document["num_classes"],
            "groups": [
                {
                    "train_id": g["train_id"],
                    "name": g["name"],
                    "level3_ids": g["level3_ids"],
                    "members": g["members"],
                }
                for g in document["groups"]
            ],
            "ignore": document["ignore"],
        }
    )
    return EXIT_OK


def _register_build_lut(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "build-lut",
        help="Resolve the 8-class collapse by name and write masks/lut.json.",
        description=(
            "Resolves CLAUDE.md section 7's grouping against the vendored anue_labels.py. "
            "Aborts if any listed name fails to resolve, or if any level3Id in anue_labels "
            "would fall in more than one group or in none."
        ),
    )
    parser.set_defaults(func=_cmd_build_lut)


# --- pack-shards (M2) -------------------------------------------------------------------


def _cmd_pack_shards(args: argparse.Namespace) -> int:
    """WebDataset shard packing (sections 6.1, 7)."""
    from drivyx.data.shards import SPLITS, pack_split, read_index, write_index

    paths = _resolve_paths(args)
    splits = [args.split] if args.split else list(SPLITS)

    per_split = {split: pack_split(paths, split, limit=args.limit) for split in splits}

    # Packing one split must not discard the other's statistics from a previous run: the
    # class weights and the GUI histogram are read from the merged index.
    merged = dict(per_split)
    if paths.shard_index.is_file():
        try:
            previous = read_index(paths)
        except (OSError, ValueError):
            previous = {"splits": {}}
        for split, data in previous.get("splits", {}).items():
            if split not in merged:
                from drivyx.data.shards import ShardStats

                merged[split] = ShardStats(
                    samples=data["samples"],
                    shards=data["shards"],
                    bytes_written=data["bytes"],
                    class_pixels=data["class_pixels"],
                    ignore_pixels=data["ignore_pixels"],
                    sequences=set(range(data["sequences"])),
                )

    index = write_index(paths, merged)
    _emit_json(index)
    return EXIT_OK


def _register_pack_shards(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "pack-shards",
        help="Pack images and collapsed masks into WebDataset shards.",
        description=(
            "Writes ~500-sample tars of (jpg image short-side 512, png collapsed mask) and a "
            "shards/index.json with counts and the per-class pixel histogram used for the "
            "loss class weights."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default=None,
        help="Split to pack (default: both).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Pack at most N samples. For smoke runs only; the index records the true count.",
    )
    parser.set_defaults(func=_cmd_pack_shards)


# --- mm-inventory (M3) ------------------------------------------------------------------


def _cmd_mm_inventory(args: argparse.Namespace) -> int:
    """Discover the multimodal layout into mm_manifest.json (sections 6.1, 8)."""
    from drivyx.data.mm_inventory import (
        mm_inventory,
        unconfirmed_fields,
        write_manifest,
    )

    paths = _resolve_paths(args)
    manifest = mm_inventory(paths)
    path = write_manifest(paths, manifest)

    from drivyx.data.mm_inventory import read_manifest

    saved = read_manifest(paths)
    pending = unconfirmed_fields(saved)
    _emit_json(
        {
            "manifest": str(path),
            "routes": {
                name: {
                    "image_dirs": [
                        {"name": d["name"], "side": d["side"], "images": d["images"]}
                        for d in route["image_dirs"]
                    ],
                    "gps_rows": sum(t["rows"] for t in route["gps"]["tables"])
                    if route.get("gps")
                    else 0,
                    "gps_hz": (route.get("gps") or {}).get("rate", {}).get("hz"),
                    "obd_rows": sum(t["rows"] for t in route["obd"]["tables"])
                    if route.get("obd")
                    else 0,
                    "obd_hz": (route.get("obd") or {}).get("rate", {}).get("hz"),
                    "clock_offset": route.get("clock_offset"),
                    "obd_tolerance": route.get("obd_tolerance"),
                }
                for name, route in saved["routes"].items()
            },
            "unconfirmed": pending,
            "next": (
                "Confirm the mappings in the LABEL workspace, then run 'drivyx mm-label'."
                if pending
                else "All mappings confirmed. Run 'drivyx mm-label'."
            ),
        }
    )
    return EXIT_OK


def _register_mm_inventory(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "mm-inventory",
        help="Discover the multimodal layout from disk into multimodal/mm_manifest.json.",
        description=(
            "Classifies every file under multimodal/ by extension and header sniffing, "
            "measures sample rates from the data, and proposes column mappings and the "
            "OBD clock offset. Nothing is applied until confirmed in the LABEL workspace. "
            "Re-running preserves existing confirmations."
        ),
    )
    parser.set_defaults(func=_cmd_mm_inventory)


# --- mm-confirm (M3) --------------------------------------------------------------------


def _cmd_mm_confirm(args: argparse.Namespace) -> int:
    """Confirm manifest mappings from the CLI (section 3: the engine must work headless)."""
    from drivyx.data.mm_inventory import confirm_all, read_manifest, unconfirmed_fields

    paths = _resolve_paths(args)
    manifest = read_manifest(paths)
    pending_before = unconfirmed_fields(manifest)

    if not args.yes:
        _emit_json(
            {
                "manifest": str(paths.mm_manifest),
                "would_confirm": pending_before,
                "note": (
                    "Dry run. Re-run with --yes to accept these proposals, or confirm them "
                    "individually in the LABEL workspace's FieldMapTable."
                ),
            }
        )
        return EXIT_OK

    changed = confirm_all(paths, manifest, route=args.route)
    _emit_json(
        {
            "manifest": str(paths.mm_manifest),
            "confirmed": changed,
            "remaining_unconfirmed": unconfirmed_fields(read_manifest(paths)),
        }
    )
    return EXIT_OK


def _register_mm_confirm(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "mm-confirm",
        help="Accept mm-inventory's proposed mappings without the GUI.",
        description=(
            "The LABEL workspace's FieldMapTable is the intended confirmation path, but "
            "CLAUDE.md section 3 requires the engine to be fully usable headless over SSH. "
            "Without --yes this lists what would be confirmed and changes nothing."
        ),
    )
    parser.add_argument("--yes", action="store_true", help="Actually confirm.")
    parser.add_argument("--route", default=None, help="Confirm one route only.")
    parser.set_defaults(func=_cmd_mm_confirm)


# --- mm-label (M3) ----------------------------------------------------------------------


def _cmd_mm_label(args: argparse.Namespace) -> int:
    """Build the waypoint dataset and QC artifacts (sections 6.1, 8)."""
    from drivyx.data.mm_label import mm_label

    _emit_json(mm_label(_resolve_paths(args), route=args.route))
    return EXIT_OK


def _register_mm_label(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "mm-label",
        help="Build the waypoint dataset (parquet per route) and QC artifacts.",
        description=(
            "Refuses to run while any required column mapping or the OBD clock offset is "
            "unconfirmed (section 8). Val split is the last 15 percent of each route by "
            "time, never random."
        ),
    )
    parser.add_argument("--route", default=None, help="Build one route (default: all).")
    parser.set_defaults(func=_cmd_mm_label)


# --- train-seg (M4) ---------------------------------------------------------------------


def _cmd_train_seg(args: argparse.Namespace) -> int:
    """PIDNet-S training, probe, and resume (sections 6.1, 9.1)."""
    from drivyx.jobs.run_dir import RunContext, create_run, freeze_config, resolve_run, write_env
    from drivyx.torch_setup import configure
    from drivyx.train.config import load_seg_config, parse_override
    from drivyx.train.probe import probe_sizes, write_probe
    from drivyx.train.seg_trainer import SegTrainer

    paths = _resolve_paths(args)
    overrides = dict(parse_override(item) for item in (args.set or []))
    config = load_seg_config(Path(args.config), overrides)
    configure()

    if args.resume:
        run = resolve_run(paths.runs, args.resume)
        logger.info("resuming %s", run.path)
    else:
        run = create_run(paths.runs, "seg", config.tag)
        freeze_config(run, config.model_dump())
        write_env(run, {"command": "train-seg", "probe": str(bool(args.probe))})

    trainer = SegTrainer(paths, config, run)

    if args.resume:
        if not run.last_ckpt.is_file():
            raise FileNotFoundError(
                f"cannot resume {run.path}: no {run.last_ckpt.name}. The run may have been "
                "interrupted before its first epoch completed."
            )
        trainer.load_checkpoint(run.last_ckpt)

    with RunContext(run) as ctx:
        if args.probe:
            results = probe_sizes(trainer, config, ctx)
            payload = write_probe(run, config, results)
            ctx.status("done", "probe complete")
            _emit_json({"run": str(run.path), "probe": str(run.path / "probe.json"), **payload})
            return EXIT_OK

        summary = trainer.fit(ctx)
        _emit_json(summary)
        # Section 6.2: a cleanly interrupted job exits 130.
        return EXIT_INTERRUPTED if summary["status"] == "interrupted" else EXIT_OK


def _register_train_seg(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "train-seg",
        help="Train PIDNet-S on the packed shards.",
        description=(
            "Writes a self-contained run directory (config snapshot, env, events.jsonl, "
            "checkpoints). SIGINT checkpoints last.pt, writes status=interrupted, and exits "
            "130; --resume continues from it with model, optimiser, epoch, and RNG restored."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to a seg config YAML.")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Measure secs/epoch at each configured size, write probe.json, and exit.",
    )
    parser.add_argument(
        "--resume", metavar="RUN", default=None, help="Run name, path, or 'latest'."
    )
    parser.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="Override a config field, e.g. --set epochs=3. Repeatable.",
    )
    parser.set_defaults(func=_cmd_train_seg)


# --- train-ctrl (M5) --------------------------------------------------------------------


def _cmd_train_ctrl(args: argparse.Namespace) -> int:
    """Waypoint predictor training with logit precompute (sections 6.1, 9.2)."""
    import yaml

    from drivyx.jobs.run_dir import RunContext, create_run, freeze_config, resolve_run, write_env
    from drivyx.torch_setup import configure
    from drivyx.train.config import parse_override
    from drivyx.train.ctrl_trainer import CtrlTrainer, precompute_logits

    paths = _resolve_paths(args)
    config = yaml.safe_load(Path(args.config).read_text())
    for item in args.set or []:
        key, value = parse_override(item)
        config[key] = value
    configure()

    # Section 9.2: the logits come from "the frozen best seg checkpoint".
    seg = resolve_run(paths.runs, args.seg_run)
    seg_ckpt = seg.best_ckpt if seg.best_ckpt.is_file() else seg.last_ckpt
    if not seg_ckpt.is_file():
        raise FileNotFoundError(
            f"seg run {seg.name} has no checkpoint. Train it first with 'drivyx train-seg'."
        )
    logger.info("using seg checkpoint %s", seg_ckpt)

    if args.resume:
        run = resolve_run(paths.runs, args.resume)
    else:
        run = create_run(paths.runs, "ctrl", str(config.get("tag", "default")))
        freeze_config(run, config)
        write_env(run, {"command": "train-ctrl", "seg_run": seg.name, "seg_ckpt": str(seg_ckpt)})

    with RunContext(run) as ctx:
        cache = precompute_logits(paths, seg_ckpt, seg.name, ctx)
        trainer = CtrlTrainer(paths, run, cache, config)
        if args.resume and run.last_ckpt.is_file():
            trainer.load_checkpoint(run.last_ckpt)
        summary = trainer.fit(ctx)

    _emit_json(summary)
    return EXIT_INTERRUPTED if summary["status"] == "interrupted" else EXIT_OK


def _register_train_ctrl(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "train-ctrl",
        help="Train the waypoint predictor on precomputed segmentation logits.",
        description=(
            "Materialises seg logits for every labelled control frame once per seg run, then "
            "trains CtrlNet (under 2 M parameters) with L1 on waypoints. Emits ADE, FDE, and "
            "lateral error at 1.0 s per epoch, all in metres."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to a ctrl config YAML.")
    parser.add_argument(
        "--seg-run",
        default="latest",
        help="Seg run whose best checkpoint produces the logits (name, path, or 'latest').",
    )
    parser.add_argument("--resume", metavar="RUN", default=None, help="Ctrl run to continue.")
    parser.add_argument(
        "--set", action="append", metavar="KEY=VALUE", help="Override a config field."
    )
    parser.set_defaults(func=_cmd_train_ctrl)


#: Subcommand name -> registration function. Each milestone adds its entries here, and this
#: mapping is the single source of truth for what the CLI offers (see registered_commands).
_REGISTRARS: dict[str, Any] = {
    "verify-data": _register_verify_data,
    "gen-masks": _register_gen_masks,
    "build-lut": _register_build_lut,
    "pack-shards": _register_pack_shards,
    "mm-inventory": _register_mm_inventory,
    "mm-confirm": _register_mm_confirm,
    "mm-label": _register_mm_label,
    "train-seg": _register_train_seg,
    "train-ctrl": _register_train_ctrl,
}


def build_parser() -> argparse.ArgumentParser:
    """Construct the full parser. Used by main() and by the CLI smoke test."""
    parser = argparse.ArgumentParser(
        prog="drivyx",
        description=(
            f"{APP_NAME}: perception and waypoint-control training for Indian driving data."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version, git SHA, torch/TRT versions, and wheel provenance, then exit.",
    )
    parser.add_argument(
        "--paths-config",
        metavar="PATH",
        default=None,
        help="Override the configs/paths.yaml location.",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="Warnings and errors only.")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    for register in _REGISTRARS.values():
        register(sub)
    return parser


def registered_commands() -> list[str]:
    """Names of the subcommands currently registered.

    The smoke test (section 13) enumerates this rather than a hardcoded list so it stays
    honest at every milestone (docs/DECISIONS.md D012).
    """
    return sorted(_REGISTRARS)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code; never raises for expected failures."""
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose, quiet=args.quiet)

    if args.version:
        from drivyx.env_report import version_summary

        print(version_summary().render())
        return EXIT_OK

    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_FAILURE

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        # Section 6.2: SIGINT is a first-class outcome, not a crash. Long jobs install their
        # own handler to checkpoint first; this is the backstop for the short ones.
        logger.warning("Interrupted.")
        return EXIT_INTERRUPTED
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_FAILURE
    except (ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
