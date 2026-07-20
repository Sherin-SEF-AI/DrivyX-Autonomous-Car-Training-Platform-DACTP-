# CLAUDE.md - DRIVYX

Perception + waypoint-control training platform for Indian driving data.
Target machine: NVIDIA Jetson AGX Orin 64GB Developer Kit (JetPack 6.x, L4T r36, CUDA 12.6, Ubuntu 22.04 aarch64).
Deliverable: a PyQt6 desktop application with a Blender-style dark UI that wraps a CLI-first training engine.
Codename DRIVYX is a single constant in `src/drivyx/branding.py`; renaming must touch nothing else.

This file is the source of truth. If the master prompt and this file conflict, this file wins.

---

## 1. Mission

Build one repo that takes the already-downloaded IDD data and produces two deployable TensorRT engines:

1. `seg`: PIDNet-S semantic segmentation, 8 collapsed classes, trained on IDD Segmentation 20k (Parts I + II).
2. `ctrl`: waypoint predictor (5 future ego-frame waypoints over 2.5 s) trained on IDD Multimodal GPS/OBD supervision, consuming `seg` logits + speed.

Everything (data prep, labeling, training, eval, export, benchmark) runs on the Orin itself and is drivable from both the CLI and the GUI.

## 2. Non-negotiable rules

- Production code only. No placeholders, no mock data, no `TODO`, no `pass` stubs, no simulated results. If something cannot be known until runtime (e.g. multimodal folder layout), build a discovery step that produces a manifest, never a hardcoded guess.
- The GUI contains zero training/data logic. Every GUI action shells out to the `drivyx` CLI via `QProcess`. The engine must be fully usable headless over SSH.
- Never install `torch` from upstream PyPI. Jetson wheels only (section 3). Never install `opencv-python` (its bundled Qt5 plugins break PyQt6); use `opencv-python-headless`.
- No work on the Qt main thread except painting. All file IO, decoding, and parsing in worker threads or subprocesses.
- Long-running jobs must be cancellable (SIGINT -> graceful checkpoint -> exit 130) and resumable.
- Every training run writes a self-contained run directory (section 6.3). Reproducibility: config snapshot, git SHA, pip freeze, seed.
- Deterministic-first: prefer closed-form / classical math (Savitzky-Golay, geometry) over learned or iterative components wherever equivalent.
- Fail loudly. Unresolved label names, unmapped level3 ids, timestamp misalignment, NaN losses: abort with a precise message, never warn-and-continue.
- No em-dash characters anywhere in code, comments, docs, or UI strings.

## 3. Environment truths (encode these in `scripts/setup_orin.sh`)

- Verify platform: `cat /etc/nv_tegra_release` must show R36.x. Warn and continue if not.
- Power: check `nvpmodel -q` for MAXN and `jetson_clocks --show`; the app shows a persistent amber banner when not at MAXN/max clocks. Setup script offers `sudo nvpmodel -m 0 && sudo jetson_clocks`.
- Python venv MUST be created with `--system-site-packages` so the JetPack-provided `tensorrt` python bindings and DeepStream libs remain importable.
- PyTorch: `pip install torch torchvision --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126`. If that index is unreachable, fall back to the direct NVIDIA wheel URL for the installed L4T version; abort with instructions rather than silently installing CPU torch. After install, assert `torch.cuda.is_available()`.
- Pin `numpy==1.26.4` (Jetson torch wheels are built against numpy 1.x; numpy 2 breaks them).
- PyQt6: try `pip install PyQt6`; if no aarch64 wheel resolves, fall back to `sudo apt install python3-pyqt6` (venv already sees it via system-site-packages). Detect and report which path was taken.
- Other pips: `pyqtgraph webdataset pydantic pyyaml opencv-python-headless pillow tqdm pandas scipy onnx jetson-stats`.
- `onnxsim` is optional on aarch64: attempt install; if it fails, export path must work without simplification (guarded import).
- `trtexec` lives at `/usr/src/tensorrt/bin/trtexec`.
- Torch runtime flags set once in `drivyx.torch_setup`: `cudnn.benchmark=True`, TF32 allowed for matmul and cudnn, default dtype fp32, autocast bf16 for training, `channels_last` for all 4D model inputs.
- DataLoader defaults for Orin: `num_workers=8`, `persistent_workers=True`, `prefetch_factor=4`, `pin_memory=False` (unified memory), `non_blocking=True` on `.to(device)`.
- The engine must be importable and unit-testable on a CPU-only machine: no CUDA calls at import time.

## 4. Data inventory (already on disk, read-only inputs)

Root: `/mnt/nvme/idd` (configurable via `configs/paths.yaml`, but this is the default).

```
/mnt/nvme/idd/
  raw/               original archives (never deleted by code)
  seg/               extracted IDD Segmentation 20k Parts I+II
                     (leftImg8bit/{train,val,test}/<seq>/*.jpg|png,
                      gtFine/{train,val}/<seq>/*_polygons.json)
  multimodal/        extracted IDD Multimodal Primary + Secondary + Supplement
                     (internal layout UNKNOWN at spec time: discovered by mm-inventory)
  pretrained/        PIDNet-S ImageNet backbone checkpoint (user supplies file;
                     verify-data checks presence and prints the download hint if absent)
  shards/            WebDataset output (generated)
  masks/             level3Id label PNGs (generated by AutoNUE tooling)
  waypoints/         control supervision parquet + QC artifacts (generated)
  runs/              training runs (generated)
  export/            onnx + trt engines + benchmark reports (generated)
```

Dataset facts to rely on: 20,000 images (14K train / 2K val / 4K test) across 350 sequences, mixed 720p/1080p, front-facing camera. Multimodal collectively holds stereo front camera at 15 fps, GPS at 15 Hz, 16-channel LiDAR, and OBD logs across three routes; which archive holds what is discovered, not assumed. LiDAR is out of scope for v1: inventory it, then ignore it.

## 5. Repo layout

```
drivyx/
  CLAUDE.md
  pyproject.toml            (single package, `drivyx` console entry point)
  scripts/setup_orin.sh
  configs/
    paths.yaml
    seg_pidnet_s.yaml
    ctrl_waypoint.yaml
    export.yaml
  src/drivyx/
    branding.py
    torch_setup.py
    cli.py                  (argparse, subcommand per section 6.1)
    jobs/                   (run dir contract, events writer, signal handling)
    data/
      verify.py
      masks.py              (AutoNUE wrapper)
      lut.py                (8-class collapse, name-resolved)
      shards.py             (WebDataset packer + reader)
      mm_inventory.py
      mm_sync.py            (timestamp association)
      waypoints.py          (smoothing, ENU, ego projection, dataset writer)
    models/
      pidnet.py             (PIDNet-S, loads ImageNet backbone)
      ctrlnet.py
      losses.py             (OHEM CE, boundary loss, L1 waypoint)
    train/
      seg_trainer.py
      ctrl_trainer.py
      probe.py
    eval/
      seg_eval.py           (mIoU, per-class IoU, confusion)
      ctrl_eval.py          (ADE/FDE, lateral error)
      viz.py                (overlay renderer shared by eval and GUI preview)
    export/
      onnx_export.py
      trt_build.py          (trtexec wrapper + INT8 entropy calibrator)
      bench.py
      parity.py
    gui/
      app.py                (entry: `drivyx-gui`)
      theme/tokens.py
      theme/blender.qss
      widgets/              (Panel, JobCard, LogConsole, StatRow, FieldMapTable)
      workspaces/           (data.py, label.py, train.py, eval.py, export.py, system.py)
      monitor.py            (tegrastats thread)
      process.py            (QProcess wrapper around CLI, JSONL tailer)
  tests/
```

## 6. Architecture

### 6.1 CLI surface (each is a thin entry over a library function)

```
drivyx verify-data                      inventory + integrity report (JSON to stdout)
drivyx gen-masks [--workers 12]         AutoNUE level3Id PNG generation
drivyx build-lut                        resolve 8-class LUT, write masks/lut.json
drivyx pack-shards --split train|val    WebDataset shards (images + collapsed masks)
drivyx mm-inventory                     discover multimodal layout -> multimodal/mm_manifest.json
drivyx mm-label [--route R]             waypoint dataset -> waypoints/*.parquet + QC plots
drivyx train-seg --config C [--probe] [--resume RUN]
drivyx train-ctrl --config C [--resume RUN]
drivyx eval-seg --run RUN [--ckpt best]
drivyx eval-ctrl --run RUN
drivyx export --model seg|ctrl --run RUN --precision fp16|int8
drivyx bench --engine PATH
drivyx infer-preview --source DIR|MP4 --seg-run RUN [--ctrl-run RUN] --out DIR
```

### 6.2 Job model

One GPU, therefore one heavy job at a time. The GUI keeps a FIFO queue; each job is a `QProcess` running a CLI command. Cancel sends SIGINT; trainer traps it, checkpoints `last.pt`, writes `status=interrupted`, exits 130. SIGKILL only after a 30 s grace period.

### 6.3 Run directory contract

`runs/<YYYYmmdd-HHMMSS>_<seg|ctrl>_<tag>/` contains: `config.yaml` (frozen snapshot), `env.txt` (git SHA, pip freeze, JetPack, nvpmodel state), `events.jsonl`, `ckpt/last.pt`, `ckpt/best.pt`, `eval/` artifacts. Nothing about a run lives anywhere else.

### 6.4 Events schema (the only GUI<->engine data channel besides exit codes)

Append-only JSONL, one object per line, flushed per write:

```
{"ts": 1721.3, "type": "scalar", "name": "train/loss", "value": 0.412, "step": 1200, "epoch": 3}
{"ts": ..., "type": "epoch",  "epoch": 3, "secs": 214.8, "eta_min": 771.2}
{"ts": ..., "type": "status", "value": "running|interrupted|failed|done", "detail": "..."}
{"ts": ..., "type": "image",  "name": "val/overlay", "path": "eval/ep3_0.jpg", "epoch": 3}
{"ts": ..., "type": "heartbeat"}   (every 15 s; GUI marks a run stale after 60 s silence)
```

The GUI tails this file (QFileSystemWatcher + incremental reader) and plots scalars by name in pyqtgraph. Schema changes are additive only.

## 7. Segmentation data pipeline

- `verify-data`: counts images per split/sequence, checks image/polygon pairing, checks pretrained checkpoint presence, prints a JSON report. Gate for everything downstream.
- `gen-masks`: clone-or-locate `AutoNUE/public-code`, run its `preperation/createLabels.py` with `--id-type level3Id --num-workers 12` against the seg root, into `masks/`. Apply any minimal Python 3 compatibility patches inside a vendored copy under `third_party/autonue/` with the patch recorded in `third_party/PATCHES.md`. Idempotent: skip sequences whose outputs already exist.
- `build-lut` (8 collapsed train classes + ignore 255). Resolution is BY NAME against the vendored `anue_labels.py`, after normalization (lowercase, strip spaces/hyphens/underscores). Grouping:

```
0 drivable        road
1 alt_drivable    parking, drivablefallback
2 nondrivable     sidewalk, railtrack, nondrivablefallback, curb
3 vru             person, rider, animal
4 twowheeler      motorcycle, bicycle
5 vehicle         car, truck, bus, autorickshaw, caravan, trailer, train, vehiclefallback
6 structure       wall, fence, guardrail, billboard, trafficsign, trafficlight, pole,
                  polegroup, obsstrbarfallback, building, bridge, tunnel
7 background      vegetation, sky, fallbackbackground
ignore(255)       unlabeled, egovehicle, rectificationborder, outofroi, licenseplate
```

  Rules: every listed name must resolve to a level3Id or the build aborts naming the miss; every level3Id present in `anue_labels` must land in exactly one group or ignore, else abort. Write `masks/lut.json` (id -> group, plus the resolved name table) for GUI display. A unit test loads 25 random generated masks and asserts every pixel value maps through the LUT.
- `pack-shards`: WebDataset tars of (jpg image resized so short side = 512 keeping aspect, png collapsed mask nearest-neighbor). ~500 samples per shard. Writer records a `shards/index.json` with counts and a per-class pixel histogram (used for loss class weights). Reader implements the training augmentation (section 9.1) on the fly.

## 8. Multimodal pipeline (control supervision)

- `mm-inventory`: walk `multimodal/`, classify every file (image dir, csv, lidar, other) by extension + header sniffing, count rows, parse candidate timestamp columns, and emit `mm_manifest.json`: routes, per-route image dirs (left/right if distinguishable), GPS table (path + column mapping guesses for time/lat/lon), OBD table (path + column guesses for time/speed), sample rates measured from data. Reference for expected formats: the AutoNUE 2019 localization challenge repo. The GUI Label workspace shows the manifest in a FieldMapTable where the user can override any column mapping; overrides are saved back into the manifest. `mm-label` refuses to run while any required mapping is `unconfirmed`.
- `mm-sync`: associate each image timestamp with the nearest GPS row within 50 ms and nearest OBD row within 100 ms; otherwise drop the frame. Segments break on GPS gaps > 0.34 s.
- `waypoints.py` math (implement exactly, with docstrings deriving each step):
  1. Per segment, convert lat/lon to local ENU meters around the segment's first fix: `x_e = (lon-lon0) * 111320 * cos(lat0_rad)`, `y_n = (lat-lat0) * 111132`. City-scale segments make this approximation sufficient; no pyproj dependency.
  2. Smooth ENU with Savitzky-Golay (window 15 samples, polyorder 3). Velocity by SG first derivative; fuse OBD speed by rescaling the SG velocity magnitude toward OBD speed with weight 0.5 when both are valid.
  3. Heading = atan2(v_n, v_e) where smoothed speed >= 1.5 m/s; below that, hold the last valid heading.
  4. For each frame at time t with position p_t and heading h_t: targets are positions at t + {0.5, 1.0, 1.5, 2.0, 2.5} s (linear interpolation between fixes), rotated into the ego frame: x forward, y left, meters.
  5. Filters: drop frames with speed < 1.0 m/s, with any target beyond a GPS gap, or with |y| of the 2.5 s point > 25 m (turnaround artifacts).
- Output: one parquet per route with columns `frame_path, t, speed_mps, wp_x[5], wp_y[5], route, segment`. Plus QC artifacts the GUI renders: smoothed-vs-raw track plot per segment, waypoint arrows drawn over 20 random frames, histogram of lateral offsets. Val split = last 15 percent of each route by time, never random (temporal leakage).
- Unit test: synthetic circular drive at constant speed; assert recovered waypoints match the analytic circle within 5 cm.

## 9. Models and training

### 9.1 seg (PIDNet-S)

- Implementation in-repo (`models/pidnet.py`), loading the user-supplied ImageNet backbone from `pretrained/`. Abort with the download hint if missing.
- Config defaults (`configs/seg_pidnet_s.yaml`), all overridable:
  - train crop 768x384 from images resized short-side 512; val at full resized 1024x512 (letterbox as needed)
  - batch 16, epochs 220, SGD momentum 0.9, lr 0.01 poly decay power 0.9, weight decay 5e-4
  - losses: OHEM cross entropy (thresh 0.9, min_kept 26000) with class weights from the shard histogram (w = 1/log(1.02 + freq), capped at 10x min), plus PIDNet boundary loss with its standard weighting
  - augmentation: hflip 0.5, random scale 0.5-2.0, random crop, color jitter 0.4/0.4/0.4
  - bf16 autocast, channels_last, no grad scaler
- `--probe`: run exactly one epoch at each of 640x320, 768x384, 1024x512 (short runs re-using the same loader), emit `probe.json` with secs/epoch and projected wall-clock for the configured epochs at each size. The GUI Train workspace displays this as the schedule picker.
- Checkpoint every epoch (`last.pt`), track best val mIoU every 5 epochs (`best.pt`). Resume restores model, optimizer, scheduler, epoch, and RNG state.
- NaN loss aborts the run with `status=failed` and the offending batch indices logged.

### 9.2 ctrl (waypoint net)

- Input: seg logits (8 x 96 x 48, i.e. the 768x384 head output average-pooled 8x, detached, produced by the frozen best seg checkpoint) + speed scalar.
- Architecture: 4 conv blocks (32, 64, 96, 128 channels, stride 2, GroupNorm(8), SiLU) -> global average pool -> concat with speed MLP (1 -> 32) -> MLP 160 -> 128 -> 10 (5 waypoints x,y in meters). Parameter budget must print at startup and stay under 2 M.
- Precompute: `train-ctrl` first materializes seg logits for every labeled frame into `shards/ctrl/` (bf16 npy inside WebDataset) so the control epochs are GPU-cheap; skip if already present for the given seg run.
- Loss: L1 on waypoints. Optim AdamW 3e-4, cosine to 1e-5, batch 256, epochs 60. Metrics: ADE, FDE(2.5 s), lateral error at 1.0 s, all in meters, emitted per epoch.

## 10. Evaluation

- `eval-seg`: per-class IoU + mIoU on IDD val at 1024x512, confusion matrix PNG, 24 qualitative overlays. Emits `eval/seg_metrics.json`.
- `eval-ctrl`: ADE/FDE/lateral on the temporal val split; overlay renderer draws predicted (accent color) vs ground-truth (white) waypoint chains projected into the image with a fixed pinhole assumption documented in `viz.py`.
- No accuracy gates are hardcoded; the numbers are the deliverable. The parity gate in section 11 is the only pass/fail.

## 11. Export and benchmark

- `export`: torch -> ONNX opset 17, static batch 1, input 1x3x384x768 (seg) and 1x8x48x96 + 1x1 (ctrl). Run onnxsim if importable. Build with `trtexec`: fp16 flags, or int8 with an entropy calibrator over 512 val images (calibration cache stored under `export/`).
- `parity`: run 200 val images through torch and the TRT engine; abort (exit 1) if seg mIoU delta > 1.0 absolute or ctrl ADE delta > 0.05 m. Writes `export/parity.json`.
- `bench`: parse trtexec latency percentiles into `export/bench.json`; GUI shows p50/p95/p99 and a green/amber budget indicator against a 33 ms frame budget (seg + ctrl combined).

## 12. GUI specification (PyQt6, Blender-style)

### 12.1 Theme tokens (`theme/tokens.py`, consumed by `blender.qss` via string substitution)

```
BG_WINDOW  #1d1d1d      TEXT       #e6e6e6      ACCENT        #4772b3
BG_AREA    #303030      TEXT_DIM   #9d9d9d      ACCENT_HOVER  #5a86c5
BG_PANEL   #282828      BORDER     #3d3d3d      OK            #6fa85c
BG_HEADER  #232323      WIDGET     #585858      WARN          #d9a23c
BG_INPUT   #1a1a1a      WIDGET_HI  #676767      ERR           #c4453c
RADIUS 4px   BORDER_W 1px   FONT_UI "DejaVu Sans" 10pt   FONT_MONO "DejaVu Sans Mono" 9pt
```

Design language: matte flat surfaces, 1 px borders, 4 px radii, monospace for every numeric readout, motion limited to binary state changes (no easing animations), color only where it carries state (accent for selection/primary action, OK/WARN/ERR for status). Never more than one accent-colored primary button visible per panel.

### 12.2 Baseline QSS (extend, do not contradict)

```
QWidget { background: %BG_AREA%; color: %TEXT%; font-family: %FONT_UI%; }
QMainWindow, QDialog { background: %BG_WINDOW%; }
QPushButton { background: %WIDGET%; border: 1px solid %BORDER%; border-radius: 4px;
              padding: 4px 12px; }
QPushButton:hover { background: %WIDGET_HI%; }
QPushButton:pressed, QPushButton[primary="true"] { background: %ACCENT%; color: white; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: %BG_INPUT%;
              border: 1px solid %BORDER%; border-radius: 4px; padding: 3px 6px; }
QTabBar::tab { background: %BG_HEADER%; padding: 6px 14px; border: 1px solid %BORDER%;
               border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
QTabBar::tab:selected { background: %BG_AREA%; color: %TEXT%; }
QProgressBar { background: %BG_INPUT%; border: 1px solid %BORDER%; border-radius: 4px;
               text-align: center; }
QProgressBar::chunk { background: %ACCENT%; border-radius: 3px; }
QGroupBox { border: 1px solid %BORDER%; border-radius: 4px; margin-top: 10px; }
QPlainTextEdit#logConsole { background: %BG_WINDOW%; font-family: %FONT_MONO%; }
```

### 12.3 Shell layout

- Top bar: workspace tabs DATA | LABEL | TRAIN | EVAL | EXPORT | SYSTEM (Blender workspace-tab pattern), right-aligned device badge: `AGX Orin 64GB . JetPack <ver> . <MAXN|amber warning>`.
- Each workspace is a QSplitter: left column of collapsible panels (Blender N-panel pattern: header row with disclosure arrow, title, optional header widget), center main view, bottom dock LogConsole (monospace, follows the active job, filter box).
- Status bar (always visible): active job name + progress, then live monospace readouts from the monitor thread: `GPU 87%  MEM 21.4/64G  SOC 71C  PWR 48W`, colored dot for job state.
- Job cards: queued/running/finished list in TRAIN and DATA workspaces; each card shows command line, elapsed, progress, Cancel (SIGINT semantics from 6.2), Open run dir.

### 12.4 Workspace contents

- DATA: verify-data report (counts table + red/green checks), buttons for gen-masks, build-lut (renders `lut.json` as a color-swatched table), pack-shards; shard index summary with the class pixel histogram as a bar chart.
- LABEL: mm-inventory trigger, manifest FieldMapTable (confirm/override column mappings, unconfirmed rows amber), mm-label trigger, QC gallery (track plots, waypoint overlays), dataset stats.
- TRAIN: config editor (form generated from the pydantic schema, YAML round-trip), probe results as a schedule table (size, secs/epoch, projected total), Start/Resume, live pyqtgraph loss and mIoU curves from events.jsonl, epoch table, latest val overlay thumbnails.
- EVAL: run picker, metrics tables, confusion matrix image, overlay browser with prev/next.
- EXPORT: precision picker, export + parity + bench pipeline as one queued sequence, results with the 33 ms budget indicator.
- SYSTEM: environment report (versions, wheel provenance, nvpmodel, disk free on NVMe), tegrastats live charts (GPU util, RAM, temps, power) over the last 10 minutes, buttons to copy diagnostics to clipboard.

### 12.5 Monitor thread

Spawn `tegrastats --interval 1000`, parse lines with a compiled regex into a dataclass, publish via Qt signal. If tegrastats is unavailable, degrade to jetson-stats' python API; if both fail, the SYSTEM workspace shows ERR state, everything else keeps working.

## 13. Testing policy

Pytest, runnable on CPU-only dev machines (`-m "not device"` default; `device` marker for Orin-only tests).

Required tests: LUT full-coverage on real generated masks (device), LUT name resolution against vendored labels (cpu), waypoint synthetic-circle accuracy (cpu), mm-sync tolerance windows with crafted timestamps (cpu), ENU round-trip sanity (cpu), events writer/reader round-trip (cpu), shard write/read augmentation shapes (cpu), seg forward/backward one step bf16 (device), ctrl param count < 2 M (cpu), export parity harness on a 10-image stub (device), CLI smoke: every subcommand `--help` exits 0 (cpu).

## 14. Milestones (build strictly in order; each gate must pass on the Orin before the next begins)

- M0 Bootstrap: `scripts/setup_orin.sh`, `torch_setup`, `verify-data`. Gate: script completes on device, `verify-data` JSON matches the disk inventory, `python -c "import torch, tensorrt, cv2, PyQt6"` succeeds in the venv.
- M1 Shell: GUI app with theme, workspaces (empty panels), monitor thread, LogConsole, QProcess wrapper running `verify-data` as the first wired job. Gate: launches < 3 s, tegrastats live in status bar, verify-data job renders its report in DATA.
- M2 Seg data: gen-masks, build-lut, pack-shards + DATA workspace wiring. Gate: LUT coverage test green on device, shards for train+val complete, histogram rendered.
- M3 Multimodal: mm-inventory, FieldMapTable, mm-sync, mm-label, QC gallery. Gate: parquet datasets exist for all routes, synthetic-circle test green, QC overlays visually sane.
- M4 Seg training: trainer, probe, TRAIN workspace with live curves. Gate: probe.json produced; a 3-epoch smoke run reaches decreasing loss, checkpoints, resumes correctly after SIGINT.
- M5 Ctrl training: logit precompute, ctrl trainer, metrics. Gate: 3-epoch smoke run with ADE reported and decreasing.
- M6 Eval: seg + ctrl eval, EVAL workspace. Gate: metrics JSONs + overlays render.
- M7 Export: onnx, trt fp16 + int8, parity, bench, EXPORT workspace. Gate: parity passes on the smoke checkpoints, bench.json rendered with budget indicator.
- M8 Preview + polish: infer-preview, EVAL-integrated video preview, README, `make` targets, full pytest pass, final lint (ruff) clean.

## 15. Commands

`make setup | test | test-device | gui | lint`. `drivyx --version` prints package version, git SHA, torch/TRT versions, and wheel provenance.

## 16. Handling unknowns

Anything not determinable from this spec (multimodal columns, AutoNUE patch details, exact PIDNet checkpoint key names) is resolved empirically on device, recorded in `docs/DECISIONS.md` with date and evidence, and surfaced in the GUI where a human should confirm (FieldMapTable pattern). Never invent data to fill a gap.