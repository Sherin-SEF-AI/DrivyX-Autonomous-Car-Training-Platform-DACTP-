# DRIVYX

Perception and waypoint control training platform for the India Driving Dataset (IDD), targeting the NVIDIA Jetson AGX Orin 64GB Developer Kit.

DRIVYX is a CLI first training engine with a PyQt6 desktop application on top. Everything runs on the Orin itself. The engine is fully usable headless over SSH, and the GUI launches CLI subcommands as subprocesses rather than containing any training logic of its own.

## What it produces

Two models trained from IDD data:

1. **seg**: PIDNet-S semantic segmentation over 8 collapsed classes, trained on IDD Segmentation 20k (Parts I and II).
2. **ctrl**: a waypoint predictor (5 ego frame waypoints over 2.5 seconds) trained on GPS and OBD supervision extracted from IDD Multimodal, consuming segmentation logits plus vehicle speed.

## Current status

| Stage | State |
|---|---|
| M0 Bootstrap: environment, `verify-data` | Complete |
| M1 Shell: GUI, theme, monitor, job queue | Complete |
| M2 Segmentation data: masks, LUT, shards | Complete |
| M3 Multimodal: inventory, sync, waypoints, QC | Complete |
| M4 Segmentation training | Probe, SIGINT, and resume verified; 3 epoch smoke run in progress |
| M5 Control training | Code complete, not yet run end to end |
| M6 Evaluation | Not started |
| M7 Export and benchmark | Not started |
| M8 Preview and polish | Not started |

Tests: 258 CPU tests and 10 device tests pass. Lint (ruff) is clean.

## Data prepared so far

Measured on device, not estimated:

```
Segmentation
  20,101 images across 552 sequences
  train 14,027 / val 2,036 / test 4,038, zero unpaired image or polygon files
  16,063 level3Id masks generated
  34 WebDataset shards, 2.54 GB

Class distribution (train)
  drivable 26.4%   alt_drivable 5.9%   nondrivable 3.5%   vru 1.4%
  twowheeler 1.1%  vehicle 7.2%        structure 17.6%    background 36.9%

Multimodal
  3 routes, stereo pairs at 15.0 Hz GPS, 0.65 Hz OBD
  9,138 waypoint frames (7,766 train / 1,372 val), 97.9% of GPS fixes retained
```

## Requirements

- NVIDIA Jetson AGX Orin (developed and tested on JetPack 7.2, L4T R39.2, CUDA 13.2, Ubuntu 24.04, Python 3.12)
- The IDD Segmentation and IDD Multimodal archives
- A PIDNet-S ImageNet backbone checkpoint

The setup script warns and continues on other L4T releases. The engine itself imports and unit tests on a CPU only machine, with device specific tests marked and skipped.

## Setup

```bash
bash scripts/setup_orin.sh      # venv, CUDA torch, PyQt6, dependency checks
bash scripts/stage_data.sh      # extract the IDD archives into the data root
drivyx verify-data              # inventory and integrity report
```

`setup_orin.sh` creates the virtualenv with `--system-site-packages` so the JetPack TensorRT bindings stay importable, installs a CUDA build of torch, and verifies the result by executing a real conv2d plus bf16 backward pass rather than trusting `torch.cuda.is_available()` alone.

`stage_data.sh` reads `configs/paths.yaml` for the archive source and data root. Archives are preserved under `raw/` and never deleted.

## Command line

```
drivyx verify-data                       inventory and integrity report (JSON to stdout)
drivyx gen-masks [--workers N]           level3Id mask generation
drivyx build-lut                         resolve the 8 class collapse, write masks/lut.json
drivyx pack-shards [--split train|val]   WebDataset shards plus a class pixel histogram
drivyx mm-inventory                      discover the multimodal layout into mm_manifest.json
drivyx mm-confirm [--yes]                accept the discovered column mappings without the GUI
drivyx mm-label [--route R]              waypoint parquet datasets plus QC artifacts
drivyx train-seg --config C [--probe] [--resume RUN] [--set KEY=VALUE]
drivyx train-ctrl --config C [--seg-run RUN] [--resume RUN]
```

Every subcommand writes JSON to stdout and logs to stderr, so output can be piped without being corrupted by log lines.

## Desktop application

```bash
make gui        # or: drivyx-gui
```

Six workspaces: DATA, LABEL, TRAIN, EVAL, EXPORT, SYSTEM. The GUI runs each action as a `drivyx` subprocess, tails the run's `events.jsonl` for live plots, and reads tegrastats for the status bar. It does not import torch, which is what keeps launch under one second.

Cancelling a job sends SIGINT. The trainer traps it, finishes the current step, writes a checkpoint, records `status=interrupted`, and exits 130. SIGKILL follows only after a 30 second grace period.

## Run directory

Each training run writes a self contained directory under `runs/`:

```
runs/<YYYYmmdd-HHMMSS>_<seg|ctrl>_<tag>/
  config.yaml      frozen config snapshot
  env.txt          git SHA, pip freeze, L4T, JetPack, nvpmodel state, torch provenance
  events.jsonl     append only event stream (scalars, epochs, status, images, heartbeats)
  run.log          full debug log for the run
  probe.json       throughput measurements, when --probe was used
  ckpt/last.pt     every epoch
  ckpt/best.pt     best validation mIoU
  eval/            evaluation artifacts
```

Nothing about a run lives outside its directory. A run stays interpretable later without the repository being at the same commit.

## Measured performance

Training throughput on this device at batch 16, from `drivyx train-seg --probe`:

| Crop | s/batch | img/s | s/epoch | 220 epochs | Peak GPU memory |
|---|---|---|---|---|---|
| 640x320 | 2.35 | 6.8 | 2,058 | 125.8 h | 1.9 GB |
| 768x384 | 3.20 | 5.0 | 2,804 | 171.4 h | 2.8 GB |
| 1024x512 | 4.68 | 3.4 | 4,100 | 250.6 h | 4.8 GB |

## Known limitations

These are properties of the data and the platform, recorded so results are read in context.

**No published PyTorch wheel ships sm_87 kernels for Python 3.12 on JetPack 7.** Kernels reach the Orin GPU by JIT compiling compute_80 PTX. Verified working by execution across the full training configuration, and measured at 31.5 TFLOPS bf16 matmul (roughly 75% of the part's dense peak), but convolution reaches a lower fraction of roofline, which is why the epoch times above are what they are.

**The multimodal routes are mostly straight driving.** Across all 9,138 waypoint frames the lateral offset at 2.5 seconds spans only -3.73 m to +5.64 m (standard deviation 0.79 m), and 78% of frames are within 0.5 m of straight. Each route is about five minutes and contains roughly two real corners. A waypoint model trained on this will mostly learn to go straight and will score well on ADE without being tested much on turning.

**OBD logs at 0.65 Hz, not the rate the specification assumed.** A literal 100 ms association window keeps about 10% of frames. The implemented window is one measured sampling interval with linear interpolation of speed, which keeps about 78%.

**The OBD clock is offset from the GPS clock by 19800 seconds** (OBD in UTC, GPS in local Indian time). This is measured per route and proposed in the manifest, but never applied automatically. `mm-label` refuses to run until a human confirms it.

**IDD withholds test split GPS.** The multimodal test CSVs carry timestamps and frame indices but no position, so control supervision comes from the train and val tables, re-split temporally.

## Repository layout

```
configs/            paths, segmentation, control, export configuration
scripts/            setup_orin.sh, stage_data.sh
src/drivyx/
  cli.py            argparse entry point, one subcommand per capability
  paths.py          data layout, pydantic validated
  torch_setup.py    runtime flags, applied once, never at import
  env_report.py     versions, wheel provenance, power state
  data/             verification, masks, LUT, shards, multimodal, waypoints
  models/           PIDNet-S, the waypoint net, losses
  train/            trainers, configs, throughput probe
  eval/             overlay renderer shared by QC, evaluation, and preview
  jobs/             run directory contract, event stream, signal handling
  gui/              PyQt6 application, one module per workspace
tests/              pytest, CPU by default, device marked tests for the Orin
third_party/        vendored AutoNUE tooling with every patch documented
docs/               DECISIONS.md, PROGRESS.md, screenshots
```

## Documentation

- `docs/DECISIONS.md` records every point where the implementation departs from the specification or resolves something the specification left open, with the measurement that drove it.
- `docs/PROGRESS.md` records each completed milestone with its gate evidence.
- `third_party/PATCHES.md` documents every line changed in the vendored AutoNUE tooling, with the failure each patch fixes and a full diff.

## Third party code

`third_party/autonue/` contains five files vendored from the AutoNUE public-code repository (commit 5d9a93b), used to rasterise IDD polygon annotations into label images. Seven patches were applied, each documented. The upstream repository carries no licence file, which is recorded in `third_party/autonue/PROVENANCE.txt`.

## Author

sherin joseph roy
