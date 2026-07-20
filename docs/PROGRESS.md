# DRIVYX Progress

One entry per milestone, written only after its gate passes on the device (CLAUDE.md
section 14). Each records what was built, the gate evidence, and deviations.

---

## M3 Multimodal - PASSED 2026-07-17

**Built:** `src/drivyx/data/{mm_inventory,mm_sync,mm_label,waypoints}.py`, `src/drivyx/eval/viz.py`
(the shared overlay renderer with the documented pinhole assumption), the `mm-inventory`,
`mm-confirm`, and `mm-label` subcommands, `gui/widgets/fieldmaptable.py`, and the LABEL
workspace with its QC gallery.

**Gate evidence** (on the device):

```
drivyx mm-inventory   -> 3 routes discovered from the bytes, 0 unassigned tables
  d0/d1/d2: leftCamImgs (side=left) + rightCamImgs (side=right)
  GPS 14.99 Hz, roles time/lat/lon/frame all matched "exact"
  OBD 0.651 Hz, roles time/speed matched "exact"
  clock_offset proposed 19800s on all three routes (raw 19758/19784/19765s)
  obd_tolerance proposed 1.536s (measured), vs section 8's 0.100s
  lidar inventoried and ignored (section 4, v1 scope)

drivyx mm-label (unconfirmed)  -> exit 1, naming all 24 pending fields  [section 8 gate]
drivyx mm-confirm --yes        -> 24 fields confirmed
drivyx mm-label                -> exit 0
  d0: 3,025 frames (2,571 train / 454 val)  1 segment  96.31% retained  OBD 77.4%
  d1: 2,998 frames (2,548 train / 450 val)  1 segment  98.75% retained  OBD 78.7%
  d2: 3,115 frames (2,647 train / 468 val)  1 segment  98.70% retained  OBD 79.1%
  total 9,138 frames; OBD speed units detected as km/h from the data

pytest tests/test_waypoints.py -> 31 passed
  synthetic circle: max error 0.111 cm, mean 0.067 cm  (section 8 bar: 5 cm)
pytest tests/test_mm_sync.py   -> 21 passed (cpu) + 1 (device)
make lint        -> All checks passed; 52 files already formatted
pytest           -> 214 passed (cpu)
pytest -m device -> 8 passed
screenshots      -> docs/screens/{m3_fieldmap,m3_qc_gallery,m3_turns}.png
```

**QC verdict, stated plainly as the gate requires: the arrows follow the road.** The clearest
evidence is `docs/screens/m3_turns.jpg`: at a junction where the road bends right, the waypoint
chain visibly curves right onto that branch; the left-turn frames curve toward the left branch.
On straight sections the chains run straight down the lane. Chain lengths match the physics
(median 15.5 m forward at 2.5 s, at a 5.8 m/s median speed). The track plot shows the smoothed
path sitting directly on the raw fixes across a ~1 km zigzag with no over-smoothing, and each
route is a single segment, meaning zero GPS gaps above 0.34 s. In the lateral-by-time plot the
five horizons nest correctly at every turn (0.5 s smallest through 2.5 s largest), which is a
consequence of the heading and rotation being right.

**Deviations:** the OBD window is one measured sampling interval with linear interpolation
rather than section 8's flat 100 ms (D010, confirmed by the user); the clock offset is measured
and human-confirmed rather than auto-applied (D009); GPS association uses the data's own
image_idx rather than a nearest-neighbour search (D008); the temporal val split overrides IDD's
shipped splits (D011); `mm-confirm` was added to the CLI surface because section 3 requires the
engine to be usable headless while section 8 puts confirmation in the GUI (D025).

**Findings this milestone produced:**

1. **OBD speed is km/h, not m/s**, detected by the code itself rather than assumed. Nothing in
   the dataset documents the unit, and a wrong guess scales every fused speed by 3.6 while
   still looking plausible. `_detect_speed_units` compares OBD against GPS-derived speed (which
   is unambiguously m/s) and aborts rather than guessing if the ratio is near neither 1 nor 3.6.
2. **IDD withholds test-split GPS** (D022), so "2 GPS tables, 3141 rows" against 3 CSVs on disk
   is correct, not a discovery bug.
3. **LiDAR shares the camera/GPS clock** (D023), which upgrades D009 from "a 5.5 h offset
   exists" to "the OBD logger specifically is on UTC".
4. **The routes are almost entirely straight.** Across all 9,138 frames the 2.5 s lateral offset
   spans only -3.73 m to +5.64 m (std 0.79 m), and 78% of frames are within 0.5 m of straight;
   only 0.12% exceed 5 m. Each ~5 minute route contains roughly two real corners. A waypoint
   predictor trained on this will overwhelmingly learn "go straight" and will score well on ADE
   while barely being tested on turning. This is a property of IDD Multimodal, not of the
   pipeline, and it is the context in which M5's ADE numbers must be read.

**Bugs found and fixed during the gate:**

1. `write_qc` plotted a 3025-length array against an empty list (a leftover line), crashing
   mm-label after it had already produced correct frames. Replaced with the smoothed-vs-raw
   track plot section 8 actually asks for, which required recording the ENU columns.
2. `sync_route` read the GPS CSV before checking the clock offset, so an unconfirmed manifest
   failed on file IO instead of naming what was unconfirmed. Validation now runs before any IO
   and reports every pending field at once rather than one per run.
3. The LABEL sidebar clipped "unconfirmed" at a 76 px label column.

---

## M2 Seg data - PASSED 2026-07-17

**Built:** `third_party/autonue/` (5 files vendored from 59 MB upstream, 7 patches),
`third_party/PATCHES.md` (line-by-line justification plus the real diff),
`src/drivyx/data/{masks,lut,shards}.py`, the `gen-masks`, `build-lut`, and `pack-shards`
subcommands, and the DATA workspace's Prepare panel, LUT swatch table, and class histogram.

**Gate evidence** (on the device):

```
drivyx gen-masks     -> 16,063 masks (14,027 train + 2,036 val), exit 0
                        re-run reports generated: 0 (idempotent)
drivyx build-lut     -> 8 classes + ignore, wrote masks/lut.json
  0 drivable     [0]              4 twowheeler  [6, 7]
  1 alt_drivable [1]              5 vehicle     [8, 9, 10, 11, 12]
  2 nondrivable  [2, 3, 13]       6 structure   [14..23]
  3 vru          [4, 5]           7 background  [24, 25]
  255 ignore     [255]
  all 27 level3Ids in anue_labels covered, 0 orphans, 0 conflicts

drivyx pack-shards --split val   -> 2,036 samples, 5 shards, 0.32 GB
drivyx pack-shards --split train -> 14,027 samples, 29 shards, 2.22 GB
  class frequency (train): drivable 26.4%  alt_drivable 5.9%  nondrivable 3.5%
                           vru 1.4%  twowheeler 1.1%  vehicle 7.2%
                           structure 17.6%  background 36.9%
  class weights: [4.004, 13.166, 18.604, 30.087, 30.454, 11.370, 5.581, 3.045]

pytest -m device tests/test_lut.py -> 2 passed   (section 13's LUT coverage gate)
make lint        -> All checks passed; 46 files already formatted
pytest           -> 190 passed (cpu)
pytest -m device -> 7 passed
screenshots      -> docs/screens/{m2_lut,m2_histogram}.png
```

Train and val class frequencies agree within 0.6% on every class, which is the distribution
check that matters: a mis-collapsed or mis-paired split would not land there by accident.

**Deviations:** the AutoNUE breakages are API rot, not the Python 2 syntax the master prompt
predicted (D020); the repository ships no licence (D018); `gen-masks` writes through a
symlink farm to keep `seg/` read-only (D019); the section 7 grouping was verified consistent
with the real label table rather than assumed (D021).

**Bugs found and fixed during the gate:**

1. `image_path_for_mask` stripped only `_labellevel3Ids.png` from a mask name, leaving
   `035471_gtFine`, and then looked for `035471_gtFine_leftImg8bit.png`. The image is
   `035471_leftImg8bit.png`: the gtFine marker survives upstream's filename substitution and
   has to come off too. Caught by running the pairing on real data rather than trusting it.
2. The histogram reported "0.0 G ignored" for a real 5.85 M count, because a fixed G scale
   cannot render 0.09%. The count was right and the display was lying; it now scales to the
   magnitude and states the share.
3. `test_image_resize_uses_interpolation` asserted INTER_AREA blends at an exact 2x
   downscale with a grid-aligned edge, where it provably cannot. The test's premise was
   wrong, not the resize; it now uses a non-integral ratio and checks nearest as a control.
4. `test_turnaround_lateral_filter` (written for M3) sized its fixture as an 8 m circle, whose
   lateral offset is bounded by the diameter at 16 m and can never trip a 25 m filter. The
   code was correct to keep those frames.

---

## M1 Shell - PASSED 2026-07-17

**Built:** `gui/theme/{tokens.py,blender.qss}` (section 12.2's baseline QSS verbatim, extended
not contradicted, with a validator that rejects an undefined token rather than shipping a
stylesheet Qt silently ignores), `gui/monitor.py` (tegrastats QThread whose regexes are
written against real JetPack 7.2 output), `gui/process.py` (QProcess wrapper implementing
section 6.2's SIGINT then 30 s grace then SIGKILL, plus a FIFO one-job-at-a-time queue and
the events.jsonl tailer), `jobs/events.py` (section 6.4 writer and reader),
`gui/widgets/{panel,logconsole,jobcard,statrow}.py`, all six workspaces, and `gui/app.py`
with the workspace tabs, device badge, MAXN banner, status bar, and shared log dock.

**Gate evidence** (on the device, DISPLAY=:1):

```
LAUNCH        : 0.752 s   (gate < 3 s)  PASS
platform      : xcb                     (a real display, not the offscreen fallback)
status bar    : 'GPU  19%  MEM  8.4/61G  SOC  48C  PWR  10W'   (live tegrastats)
device badge  : 'AGX Orin 64GB . JetPack R39.2.0 . MAXN'
MAXN banner   : hidden at MAXN; shown with the remediation command when not
DATA workspace: verify-data job ran through the queue and rendered
                ok=True, 20,101 images, 3 count rows, 9 check rows, 1 job card

make lint        -> All checks passed; 38 files already formatted
pytest           -> 107 passed (cpu)
pytest -m device -> 4 passed
grep -rnE "TODO|FIXME|^\s*pass\s*$" src/  -> none
screenshots      -> docs/screens/{m1_data,m1_maxn_banner,m1_system}.png
```

**Deviations:** `libxcb-cursor0` was absent, so Qt silently loaded the `offscreen` platform
plugin and the GUI rendered nowhere while reporting no error. setup_orin.sh now installs it
and asserts `platformName() != "offscreen"` when a display is present (D017). The GUI imports
neither torch nor pyqtgraph at module scope, which is what keeps launch at 0.75 s against the
3 s budget; both are tested. LABEL, TRAIN, EVAL, and EXPORT show panels naming the milestone
that fills them (M3/M4/M6/M7) rather than dead buttons, per section 14's "M1 Shell: workspaces
(empty panels)" and rule 23's ban on stubs.

**Bugs found and fixed during the gate:**

1. `JobRunner` and `JobQueue` both declared `event = pyqtSignal(object)`, which shadows
   `QObject.event()`, the virtual Qt calls to dispatch every event to an object. Qt therefore
   called a signal where it expected a method, raising `TypeError: native Qt signal is not
   callable` on every event delivered to the queue. Renamed to `job_event`, with
   `tests/test_gui_contracts.py` asserting no GUI signal shadows a QObject method.
2. Closing the window while the SYSTEM workspace's environment thread was still importing
   torch destroyed the QThread mid-run and aborted the process ("QThread: Destroyed while
   thread is still running"). `closeEvent` now joins threads in dependency order; the case
   that core-dumped now exits cleanly and is covered by a test.
3. The DATA sidebar clipped its longest labels ("sequences", "multimodal") at a 64 px label
   column, and the device badge could be squeezed below its sizeHint by the top bar's stretch
   when off MAXN made its text longer. Both fixed and re-verified by measurement
   (sizeHint 370 == actual 370).

**Caveat on the MAXN gate:** section 14 suggests toggling `sudo nvpmodel` to observe both
banner states. sudo is interactive here, so the device was not physically downclocked.
Instead both states were exercised at `apply_environment`, which is the single decision point
for the badge and banner, and `power_state()` was verified against this device's real
`nvpmodel -q` output (correctly reporting MAXN). The untested link is only nvpmodel emitting a
non-MAXN string, which the tested regex already covers. A real toggle would close that gap.

---

## M0 Bootstrap - PASSED 2026-07-17

**Built:** `scripts/setup_orin.sh` (platform check, power/MAXN check, CUDA toolkit probe,
`--system-site-packages` venv, CUDA torch install, PyQt6 path detection, optional onnxsim,
execution-verified environment check), `scripts/stage_data.sh` (archive preservation by
hardlink, seg Parts I+II merge, multimodal extraction, backbone hint), `configs/paths.yaml`
with a pydantic-validated loader, `src/drivyx/{branding,paths,torch_setup,logging_setup,
env_report,cli}.py`, and `src/drivyx/data/verify.py` implementing `drivyx verify-data`.

**Gate evidence** (all run on the device):

```
bash scripts/setup_orin.sh            -> exit 0
  torch        : 2.13.0+cu130  (CUDA build: 13.0)
  cuda device  : Orin, capability sm_87, bf16 support True
  cuda kernel  : conv2d+bf16+backward OK (loss=0.3560)
  tensorrt     : 10.16.2.10
  cv2          : 4.11.0
  PyQt6        : 6.11.0 (pip wheel)
  numpy        : 1.26.4

python -c "import torch,cv2,PyQt6;print(torch.__version__,torch.cuda.is_available())"
  -> 2.13.0+cu130 True
python -c "import tensorrt as trt; print(trt.__version__)"
  -> 10.16.2.10

drivyx verify-data | python -m json.tool   -> exit 0, ok: true, 0.98 s
  seg   : 20101 images / 552 sequences; train 14027/14027 paired, val 2036/2036 paired,
          test 4038 images; 0 unpaired, 0 orphan polygons
  mm    : 36049 files, 23.2 GB; primary+secondary+supplement, routes d0/d1/d2, 15 csv
  warn  : pretrained.backbone absent (spec'd behaviour: hint printed, non-blocking)

make lint    -> All checks passed, 13 files already formatted
pytest       -> 49 passed (cpu)
pytest -m device -> 4 passed
```

**Deviations** (all in docs/DECISIONS.md with measured evidence):

The device is JetPack 7.2 / L4T R39.2 / CUDA 13.2 / Ubuntu 24.04 / Python 3.12, not the
JetPack 6.x / CUDA 12.6 / Ubuntu 22.04 stack CLAUDE.md section 3 assumes (D001). Section 3's
mandated wheel index no longer resolves in DNS and its cp310/CUDA-12.6 wheels cannot install
here, so torch comes from the official cu130 aarch64 index instead, confirmed by the user
(D002). No published wheel carries native sm_87 kernels for cp312 on JetPack 7; Orin is
reached by PTX JIT from compute_80, verified by executing the full bf16 training
configuration and measured at 31.5 TFLOPS bf16 matmul, about 75% of the part's peak (D015).
`python3-venv` was absent and setup_orin.sh now installs it reproducibly (D014). The data was
not "already on disk" as section 4 states but sat as unextracted archives, so
`scripts/stage_data.sh` was added (D006). `/mnt/nvme` did not exist because the root
filesystem is itself the NVMe; the spec's default path is retained and created there (D005).
Staged seg has 552 sequences rather than section 4's ~350, and mixes PNG (Part I) with JPG
(Part II), so every image reader accepts both suffixes (D016). CLI subcommands register as
milestones land rather than existing as stubs, and the smoke test enumerates the live
registry (D012).

**Bug found and fixed during the gate:** the first version of setup_orin.sh printed
"Environment verified" while torch simultaneously reported "No published PyTorch CUDA builds
for release 2.13.0+cu130 support this GPU". `torch.cuda.is_available()` is necessary but not
sufficient: it returns True even when the wheel ships no kernels for the device's
architecture. The check now executes a conv2d + bf16 + backward pass and asserts the loss and
gradients are finite, so an unusable wheel fails at setup rather than at hour six of a
training run.

**Carried forward:** the PIDNet-S ImageNet backbone is still absent. This is a warning, not a
blocker, per section 4, and M1 through M3 do not need it. `train-seg` (M4) aborts without it.
