# DRIVYX Decisions

Empirical resolutions of everything CLAUDE.md left open or got wrong about this device.
Each entry: date, evidence, decision. Per CLAUDE.md section 16.

---

## D001 - Target platform is JetPack 7.2, not JetPack 6.x

**Date:** 2026-07-17
**Spec assumption:** CLAUDE.md line 4 and section 3: "JetPack 6.x, L4T r36, CUDA 12.6, Ubuntu 22.04 aarch64".

**Evidence measured on device:**

```
/etc/nv_tegra_release  -> R39 (release), REVISION: 2.0, DATE: Mon Jun 1 2026
/etc/os-release        -> Ubuntu 24.04.4 LTS
python3 --version      -> Python 3.12.3
nvidia-smi             -> Driver 595.78, CUDA Version 13.2
dpkg -l | grep nvinfer -> TensorRT 10.16.2.10-1+cuda13.2
apt-cache policy nvidia-jetpack -> Installed: (none), Candidate: 7.2-b187
```

**Decision:** Treat JetPack 7.2 / L4T R39.2 / CUDA 13.2 / Python 3.12 as the real target.
CLAUDE.md section 3 says the platform check must "warn and continue" when the release is not
R36.x, so setup_orin.sh warns and proceeds rather than aborting. Every other section 3 constraint
is re-derived for this platform below.

---

## D002 - PyTorch wheel source: official cu130 aarch64, not the jp6/cu126 Jetson index

**Date:** 2026-07-17
**Spec assumption:** CLAUDE.md section 3: "pip install torch torchvision --index-url
https://pypi.jetson-ai-lab.dev/jp6/cu126", and rule 25: "Never install torch from upstream PyPI.
Jetson wheels only".

**Evidence measured on device:**

```
getent hosts pypi.jetson-ai-lab.dev  -> DNS FAIL (domain no longer resolves)
curl https://pypi.jetson-ai-lab.io/jp6/cu126/torch  -> 200 (host moved to .io)
curl https://pypi.jetson-ai-lab.io/jp7/cu130/torch  -> 404 (no jp7 index published)
```

The surviving jp6/cu126 wheels are built cp310 against CUDA 12.6 for L4T r36. This device is
cp312 on CUDA 13.2, so those wheels cannot resolve or link here.

The official PyTorch cu130 index publishes real CUDA aarch64 (SBSA) wheels for cp312.
JetPack 7 moved Jetson onto the standard CUDA 13 arm64 stack, which is why these apply where the
JetPack 6 era required bespoke Jetson wheels.

Versions available on that index for cp312/aarch64 (measured 2026-07-17):

```
torch       : 2.9.1, 2.10.0, 2.11.0, 2.12.0, 2.12.1, 2.13.0
torchvision : 0.25.0, 0.26.0, 0.27.0, 0.27.1, 0.28.0
```

**Decision:** Install `torch==2.13.0+cu130` with `torchvision==0.28.0+cu130` from
https://download.pytorch.org/whl/cu130. These are CUDA wheels, not CPU wheels, so the intent of
rule 25 (never silently land on CPU torch) is preserved. setup_orin.sh asserts
torch.cuda.is_available() after install and aborts loudly if false, per section 3.

The two versions are a **matched pair and must be bumped together**: torchvision pins an exact
torch version, so an independently chosen "latest of each" fails to resolve. A first attempt
paired torch 2.9.1 with torchvision 0.28.0 and pip rejected it:

```
The conflict is caused by:
    The user requested torch==2.9.1
    torchvision 0.28.0+cu130 depends on torch==2.13.0
```

**Consequence for D003:** the cu130 torch wheels depend on pip-packaged CUDA 13.0 components
(`nvidia-cuda-runtime`, `nvidia-cudnn-cu13`, `nvidia-cublas`, `cuda-toolkit`), and the driver-side
`libcuda.so` is already present via `/opt/nvidia/l4t-gpu-libs/nvgpu/`. Torch therefore reaches the
GPU without the `nvidia-jetpack` apt meta-package. The pip CUDA 13.0 runtime is minor-version
compatible with the device's CUDA 13.2 driver.

**Confirmed by user:** 2026-07-17.

---

## D003 - CUDA toolkit and cuDNN are absent and must be installed

**Date:** 2026-07-17
**Evidence:** No /usr/local/cuda*, no nvcc, no libcudart outside ollama's private copy, no cuDNN
apt package. Only TensorRT 10.16.2.10 and trtexec are present. `nvidia-jetpack` is not installed.

**Decision:** setup_orin.sh installs the `nvidia-jetpack` meta-package (candidate 7.2-b187) via
apt, which pulls CUDA 13.2 and cuDNN. This is a large download and needs sudo, so the script
prompts before doing it and is idempotent (skips when already present).

---

## D004 - trtexec location differs from spec

**Date:** 2026-07-17
**Spec assumption:** section 3: "trtexec lives at /usr/src/tensorrt/bin/trtexec".
**Evidence:** `/usr/src/tensorrt/bin/trtexec` exists but is a symlink to `../../../bin/trtexec`,
resolving to `/usr/bin/trtexec` (owned by the libnvinfer-bin package).

**Decision:** Resolve trtexec at runtime: honor the spec path first, fall back to PATH lookup, and
abort with a precise message if neither resolves. Never hardcode a single absolute path.

---

## D005 - Data root: /mnt/nvme/idd retained, created on the NVMe root filesystem

**Date:** 2026-07-17
**Spec assumption:** section 4: data root `/mnt/nvme/idd`.
**Evidence:** `/mnt/nvme` does not exist. `lsblk` shows the root filesystem IS the NVMe:
`nvme0n1p1 930G ext4 /`, with 770G free. There is no separate NVMe mount to point at.

**Decision:** Keep `/mnt/nvme/idd` as the paths.yaml default. It is created as a real directory on
nvme0n1p1, so the path is physically accurate (it is on the NVMe) and the spec default holds
literally with no config drift. Creation needs sudo once; stage_data.sh does it and chowns to the
invoking user. The value stays overridable via configs/paths.yaml per section 4.

---

## D006 - Data staging step added (spec gap)

**Date:** 2026-07-17
**Spec assumption:** section 4 describes the data as "already on disk, read-only inputs" with
`seg/` and `multimodal/` already extracted.
**Evidence:** Nothing is extracted. The archives sit in /home/blurabbit/Downloads:

```
19G  idd-segmentation.tar.gz   -> IDD_Segmentation/  (Part I)
5.6G idd-20k-II.tar.gz         -> idd20kII/          (Part II)
6.5G idd_mm_primary.zip        -> idd_multimodal/primary/
6.7G idd_mm_secondary.zip      -> idd_multimodal/secondary/
3.0G idd_mm_supplement.zip     -> idd_multimodal/supplement/
```

**Decision:** Add `scripts/stage_data.sh` to populate `/mnt/nvme/idd` from the archives. Kept
separate from setup_orin.sh so environment setup stays fast and re-runnable while staging (a
multi-hour, ~80GB extraction) is invoked once. Archives are copied to `raw/`, never deleted, per
section 4. `verify-data` reports precisely which stage is missing when staging has not run.

---

## D007 - Multimodal archive roles (resolved from bytes, not assumed)

**Date:** 2026-07-17
**Spec position:** section 4: "internal layout UNKNOWN at spec time: discovered by mm-inventory".
The master prompt hypothesised "Primary/Secondary are the stereo halves or a route split and
Supplement carries logs, but this is unverified."

**Evidence** (archive central directories, no extraction):

```
primary    : 13543 .jpg + 9 .csv    -> d0,d1,d2 ; per-route {train,test,val}.csv
secondary  : 13543 .jpg + 0 .csv    -> d0,d1,d2 ; rightCamImgs/
supplement : 8948 .npy + 6 .csv     -> lidar/{d0,d1,d2}/*.npy ; obd/{d0,d1,d2}/obd.csv
```

Primary and secondary hold an identical image count (13,543) across the same three routes, and
secondary's images live under `rightCamImgs/`. The hypothesis is therefore confirmed: **primary and
secondary are the stereo halves (primary = left, secondary = right), across three routes d0/d1/d2;
supplement carries LiDAR and OBD.**

CSV headers:

```
primary/<route>/{train,val,test}.csv : timestamp,image_idx,latitude,longitude,altitude
supplement/obd/<route>/obd.csv       : timestamp,speed,rpm,gear,clv,throttle_position
```

Timestamp format is `HH-MM-SS-microseconds`.

**Decision:** This entry is evidence for expectations only. Per section 8 and master-prompt rule
"Do not hardcode any multimodal path or column name outside the manifest", mm-inventory still
derives all of this from disk at runtime and writes mm_manifest.json; the FieldMapTable remains the
only confirmation path. LiDAR is inventoried then ignored (section 4, v1 scope).

---

## D008 - GPS/image association is given by the data, not searched

**Date:** 2026-07-17
**Spec position:** section 8 mm-sync: "associate each image timestamp with the nearest GPS row
within 50 ms".
**Evidence:** `image_idx` is a column of the primary GPS CSV, pairing each fix to a frame 1:1:

```
timestamp,image_idx,latitude,longitude,altitude
09-00-31-289685,0000000,17.4962518306,78.4156015873,485.792053223
```

Measured GPS rate: 15.0 Hz exactly (median dt 66.7 ms) on all three routes, matching section 4's
"GPS at 15 Hz".

**Decision:** Use the provided `image_idx` pairing as authoritative. The 50 ms rule is retained as
an assertion (verifying the paired fix is self-consistent) rather than a nearest-neighbour search,
so a future dataset lacking image_idx still fails loudly instead of silently mispairing.

---

## D009 - OBD clock is UTC while GPS clock is local (IST, UTC+5:30)

**Date:** 2026-07-17
**Evidence:** Raw GPS and OBD time ranges do not overlap on any route:

```
route  GPS window (n=~4.5K, 15.0 Hz)      OBD window (n=~155, 0.65 Hz)      raw overlap
d0     09:00:31.290 -> 09:05:33.330       03:31:13.000 -> 03:35:31.400      NONE
d1     13:00:06.218 -> 13:04:56.227       07:30:22.000 -> 07:34:54.500      NONE
d2     16:00:03.156 -> 16:05:13.822       10:30:38.536 -> 10:35:12.680      NONE
```

Every route shows an offset of ~5h29m30s, i.e. 19800 s (5h30m) minus a few tens of seconds of
logger start skew. 19800 s is exactly the IST (UTC+5:30) offset, and the recordings are from
Hyderabad (17.496 N, 78.415 E). Shifting OBD by +19800 s:

```
route  overlap after shift   GPS frames inside OBD window
d0     258.4s               3876/4531 (85.5%)
d1     272.5s               4087/4351 (93.9%)
d2     274.1s               4112/4661 (88.2%)
```

A wrong offset would not nest the OBD window inside the GPS window on all three routes at once.

**Decision:** OBD timestamps are UTC, GPS timestamps are IST. This is a 5.5-hour "timestamp
misalignment", which rule 30 says must fail loudly and never warn-and-continue. So the offset is
never auto-applied: `mm-inventory` measures it per route, proposes it as a `clock_offset_s` field
in mm_manifest.json marked `unconfirmed`, and `mm-label` refuses to run until a human confirms it
in the FieldMapTable. This applies section 8's own confirmation mechanism to a field the spec did
not anticipate.

---

## D010 - OBD sync tolerance widened from 100 ms to one sampling interval, with interpolation

**Date:** 2026-07-17
**Spec position:** section 8 mm-sync: "nearest OBD row within 100 ms; otherwise drop the frame".
**Evidence:** OBD logs at 0.65 Hz (median dt 1536 ms), not the ~10 Hz that a 100 ms window implies.
Measured frame retention against the nearest OBD sample, after the D009 clock correction:

```
route  within 100ms (spec)      within 1600ms (one OBD dt)
d0     467/4531  (10.3%)        3820/4531  (84.3%)
d1     448/4351  (10.3%)        3683/4351  (84.6%)
d2     486/4661  (10.4%)        3977/4661  (85.3%)
```

The literal spec rule discards ~90% of the control supervision (~1.4K of 13.5K frames).

**Decision:** Interpolate OBD speed linearly between bracketing samples when the frame lies within
one OBD sampling interval (1.6 s), retaining ~85% of frames. Vehicle speed is smooth over 1.5 s at
urban speeds, so linear interpolation is physically sound and stays within the "deterministic-first"
rule (line 29). Frames with no OBD sample in range fall back to GPS-derived Savitzky-Golay speed
alone, which section 8.2 already contemplates ("when both are valid"). The GPS side of mm-sync is
unchanged. The tolerance is a config value, not a literal, and mm-inventory derives the 1.6 s from
the measured median dt rather than hardcoding it.

**Confirmed by user:** 2026-07-17.

---

## D011 - Temporal val split overrides IDD's shipped splits

**Date:** 2026-07-17
**Evidence:** primary ships its own `{train,val,test}.csv` per route.
**Spec position:** section 8: "Val split = last 15 percent of each route by time, never random
(temporal leakage)."

**Decision:** Follow the spec. The shipped splits are inventoried in mm_manifest.json for
completeness but are not used to build the control dataset, because their construction is
undocumented and a non-temporal split would leak across the 2.5 s waypoint horizon. Recorded here
so the divergence from the dataset's own convention is deliberate and visible.

---

## D012 - CLI subcommands are registered as milestones land

**Date:** 2026-07-17
**Tension:** CLAUDE.md section 6.1 lists 13 subcommands; section 13 requires a smoke test asserting
every subcommand `--help` exits 0. Rule 23 forbids stubs.

**Decision:** cli.py registers only implemented subcommands. The smoke test enumerates the parser's
registered subcommands rather than a hardcoded list, so it stays honest and green at every
milestone, and reaches full section 6.1 coverage at M8. A stub subcommand that errors
"not implemented" would violate rule 23.

---

## D031 - trtexec percentile parsing: two overwrite bugs from one output format

**Date:** 2026-07-20
**Found:** by reading the first real bench.json, not by a test.

trtexec prints five timing sections, each on its own line and each carrying its own
percentiles:

```
Latency:          min = 1.87 ms, ..., median = 4.32 ms, percentile(99%) = 6.56 ms
Enqueue Time:     min = 0.68 ms, ...
H2D Latency:      min = 0.12 ms, ...
GPU Compute Time: min = 1.71 ms, ..., median = 4.15 ms
D2H Latency:      min = 0.005 ms, ..., percentile(99%) = 0.025 ms
Total GPU Compute Time: 2.9848 s
```

**Bug 1.** Scanning the whole output for `percentile\(NN%\)` returns the last section's
values, which is D2H Latency: a device-to-host copy roughly 200x faster than end-to-end
latency. The first bench.json therefore reported p50 4.52 ms with p95 0.021 ms and p99
0.025 ms, percentiles smaller than the median, which is impossible.

**Bug 2.** After fixing that by parsing per section, `gpu_compute_ms` came out zero.
`Total GPU Compute Time: 2.9848 s` matches the same label as the real GPU Compute Time
section but carries one summed value and no statistics, so it overwrote the real section
with an empty one.

**Fixes:** percentiles and statistics are read from the same line as their section label; a
section with no statistics is skipped, which rejects the Total line semantically rather than
by position; and the parser asserts percentiles are monotonically non-decreasing, so a future
format change fails loudly instead of reporting numbers that cannot be true.

Also now captured: trtexec's own `coefficient of variance` warning. It read 28 to 30 percent
here, which is worth surfacing because it means the p99 is not comparable between runs. The
cause in this instance was benign: a training run was using the GPU concurrently.

Both bugs are pinned by `tests/test_export.py` against verbatim trtexec output.

The pattern is the same one D030 showed in the GUI: a later match silently overwriting an
earlier one. Worth watching for anywhere a parser scans a whole document for a repeated
pattern.

---

## D030 - A mouse wheel over the FieldMapTable silently rewrote a confirmed mapping

**Date:** 2026-07-20
**Found:** by opening the GUI on the desktop and reading the table, not by a test.

**What happened:** Qt's default is that a wheel event over an *unfocused* QComboBox changes its
value. The FieldMapTable is a scrollable table of combo boxes, and each combo's
`currentTextChanged` is wired to write the new mapping into `mm_manifest.json` and mark it
confirmed, because choosing a column is itself a confirmation. So scrolling the table moved
whichever combo passed under the cursor and persisted the result.

The observed damage: `d1.gps.frame` changed from `image_idx` to `timestamp`, confidence
`manual`, state `confirmed`. That is the column pairing each GPS fix to an image; a wrong value
would pair every waypoint on that route with the wrong picture.

**No data was actually corrupted.** The scroll happened well after `mm-label` had run, so the
parquets were built from the correct mapping. Verified by rebuilding d1 after the repair: 2,998
frames, identical to before, with valid unique frame paths that exist on disk.

**The repair mechanism worked as designed**, which is worth recording. Re-running mm-inventory
re-derived `image_idx` from the bytes and refused to carry the stale confirmation forward,
because `_merge_confirmations` only preserves a confirmation when the confirmed column still
matches what discovery finds. It flagged exactly the one corrupted row as unconfirmed and left
the other 23 untouched.

**Fixes:**

1. `_NoScrollComboBox` ignores wheel events unless it has focus, and passes them to the parent
   so the table still scrolls. Focus policy is StrongFocus, so a combo takes focus by click or
   tab and never by the wheel passing over it.
2. `_on_column_changed` returns early when the "new" value equals the proposed one. A no-op
   edit is not a confirmation, and without this any spurious signal during population would
   mark a row confirmed that nobody had looked at.

The general lesson is about the design, not the widget: this table's edits have side effects on
disk, so every path that can emit an edit has to be one a human actually took. An input whose
default Qt behaviour includes "changes on hover-scroll" is not safe to wire directly to a
persistent write.

---

## D027 - The probe times batches at steady state rather than running full epochs

**Date:** 2026-07-20
**Spec position:** section 9.1: "--probe: run exactly one epoch at each of 640x320, 768x384,
1024x512".

**Evidence:** an epoch is 876 batches at batch 16. Three full epochs is ~2,600 batches, which
at this device's measured 2.3 to 4.7 s/batch is 1.7 to 3.4 hours of GPU time.

**Decision:** time 40 batches per size after an 8-batch warmup and scale to the real epoch
length. The quantity `--probe` exists to project *is* the steady-state per-batch rate, so
measuring it 876 times instead of 40 costs hours and adds no accuracy. probe.json records
`measured_batches` alongside `batches_per_epoch` so the number is never presented as a
measured epoch, and the note field states the deviation.

The warmup is not optional here: the first batches at a new shape pay cuDNN autotuning and,
because no published wheel ships sm_87 kernels (D015), a PTX JIT compile. Timing those would
report a schedule several times worse than the real one.

**Measured result** (batch 16, 220 epochs):

```
size       s/batch  img/s  s/epoch  220 epochs  peak GB
640x320      2.349    6.8     2058     125.8 h     1.93
768x384      3.201    5.0     2804     171.4 h     2.75
1024x512     4.681    3.4     4100     250.6 h     4.80
```

**This is the project's main open decision.** Section 9.1's configured schedule (768x384, 220
epochs) is 7.1 days of continuous training. The 5.0 img/s is consistent with D015: PIDNet is
entirely convolutional, and conv reaches only ~21% of roofline under PTX JIT while cuBLAS
matmul reaches 75%. Peak memory of 2.8 GB out of 64 GB confirms the run is compute-bound with
headroom unused. The M4 gate needs only a 3-epoch smoke run, so nothing is blocked; the full
run is recorded here as a deliberate choice to be made rather than started blind.

---

## D028 - CtrlNet's final layer is zero-initialised

**Date:** 2026-07-20
**Spec position:** section 9.2 specifies the architecture and the under-2M budget but not the
initialisation.

**Decision:** zero the final Linear's weight and bias, so an untrained CtrlNet predicts all
waypoints at (0, 0) rather than a random trajectory.

The reason is specific to this dataset. D022 measured that 78% of frames are within 0.5 m of
straight and the 2.5 s lateral offset spans only -3.7 m to +5.6 m. Under L1 loss, "stay put" is
already a low-loss starting point, and the optimiser spends its first steps refining rather
than unwinding a random guess. With Xavier init throughout, the head's initial outputs are
metres of arbitrary displacement that must be undone first.

Measured: 209,226 parameters, 10.5% of the 2M budget.

---

## D029 - PIDNet training requires batch >= 2

**Date:** 2026-07-20
**Evidence:** at batch 1 in training mode the model raises

```
ValueError: Expected more than 1 value per channel when training, got input size [1, 96, 1, 1]
```

The PAPPM's global-pool branch reduces to 1x1, and BatchNorm in training mode cannot compute a
variance from a single value per channel.

**Decision:** treat this as a documented constraint, not a defect. Section 9.1 trains at batch
16 and the probe uses the same, so nothing in the specified pipeline is affected. Inference at
batch 1 works, which is what section 11's static batch-1 export requires. Both behaviours are
pinned by tests (`test_batch_one_training_is_rejected_clearly`,
`test_batch_one_inference_works`) so the distinction cannot silently change, and the constraint
is stated in the PIDNet docstring where someone hitting it would look.

---

## D025 - mm-confirm added to the CLI so the engine stays usable headless

**Date:** 2026-07-17
**Tension:** section 8 puts manifest confirmation in the GUI ("The GUI Label workspace shows the
manifest in a FieldMapTable where the user can override any column mapping"), while section 3
requires that "the engine must be fully usable headless over SSH" and section 6.1's CLI surface
lists no confirmation command. Taken literally, a headless user cannot run mm-label at all.

**Decision:** add `drivyx mm-confirm`, which accepts mm-inventory's proposals from the CLI.
Without `--yes` it lists what would be confirmed and changes nothing, so the default is a dry
run and confirmation stays a deliberate act. The FieldMapTable remains the richer path: it shows
each proposal's confidence, its alternative columns, and the measured evidence behind it, and it
is where a human disagrees with an individual row.

This adds to section 6.1's surface rather than changing it. D012's rule still holds: the CLI
registers only implemented commands, and the smoke test enumerates the live registry.

---

## D026 - OBD speed units are detected from the data, not assumed

**Date:** 2026-07-17
**Evidence:** the OBD table's `speed` column carries no unit anywhere in the dataset, its README,
or the manifest. Sample values (11.0, 13.0, ...) are plausible as either m/s (a brisk urban
drive) or km/h (a slow one).

Measured against GPS-derived speed, which is unambiguously m/s because it comes from ENU metres
over timestamped seconds, the median ratio identifies the unit: it came out near 3.6 on all
three routes, so **OBD speed is km/h**.

**Decision:** `_detect_speed_units` computes that ratio over frames where both signals are valid
and moving, accepts ~1.0 as m/s and ~3.6 as km/h, and **aborts** when the ratio is near neither
rather than picking the closer of two wrong answers. A wrong unit would scale every fused speed
by 3.6 and silently corrupt the ctrl net's speed input (section 9.2) while still producing
numbers that look like speeds.

The check doubles as a validator of D009: a wrong clock offset pairs OBD samples with unrelated
frames, which destroys the ratio and trips this abort. The unit detection passing on all three
routes is therefore independent corroboration that the 19800 s offset is right.

---

## D024 - PIDNet-S backbone sourced from OpenMMLab, not the PIDNet repo's dead Drive links

**Date:** 2026-07-17
**Spec position:** section 4: "pretrained/ PIDNet-S ImageNet backbone checkpoint (user supplies
file; verify-data checks presence and prints the download hint if absent)". Section 9.1 aborts
training without it.

**Evidence:** the hint verify-data prints points at the PIDNet repository, whose links are dead.
The README itself carries a notice:

> It appears the download links below are no longer working due to my missing Google Drive
> account. Please download all the weights via [this link].

Both were attempted from this device and both failed:

```
gdown 1hIBp_8maRr60-B3PF0NVtaA6TYBvO4y-        -> "Gdown can't. Check connections and permissions."
gdown --folder 0BySIOtxxULinfjlGdG...          -> status code 401
```

The replacement folder also refuses non-browser clients, so the spec's "user supplies file"
path requires an interactive browser session for a file that has a public mirror.

**Decision:** fetch the checkpoint from OpenMMLab's CDN, which hosts a converted PIDNet-S
ImageNet-1k checkpoint with no authentication:

```
https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/
    pidnet-s_imagenet1k_20230306-715e6273.pth
25,917,029 bytes
sha256 715e62734f95057f18edb286d830bee09bba08b0ecfca42420506d1d7aba97de
```

This source is **stronger** than the original, not merely more convenient: OpenMMLab's filename
convention embeds the first 8 hex of the file's own SHA256, and the downloaded bytes hash to
`715e6273`, matching the `-715e6273` in the name. The checkpoint verifies itself. A Google
Drive link offers no integrity guarantee at all.

**Verified to be PIDNet-S rather than trusted by filename** (302 tensors, 6,456,274 params):

```
stem              68 tensors     180,139 params   stem.0.conv.weight is (32, 3, 3, 3)
i_branch_layers  126 tensors   5,886,485 params
p_branch_layers   72 tensors     355,852 params
pag_1, pag_2      24 tensors       8,708 params
compression_1/2   12 tensors      25,090 params
```

`pag` (pixel-attention-guided fusion) and `compression` are PIDNet-specific modules, and the
stem's 32 output channels match PIDNet-S specifically. There is no 1000-way classifier tensor,
so the ImageNet head is already stripped, which is what "backbone" should mean.

**Consequences for M4:**

1. **Key naming is mmsegmentation's, not the original repo's.** Upstream PIDNet names its stem
   `conv1.0.weight`; this checkpoint uses `stem.0.conv.weight` with separate `.conv`/`.bn`
   submodules (mmseg's ConvModule layout). Section 16 lists "exact PIDNet checkpoint key names"
   as a runtime unknown, and this is the empirical answer. `models/pidnet.py` is written in-repo
   (section 9.1), so it adopts this layout directly rather than remapping onto a naming
   convention no file on this disk uses.

2. **The D branch is not pretrained.** There are no `d_branch_layers` and no SPP/DFM head
   tensors: ImageNet pretraining covers the stem and the P/I branches only. The boundary branch
   and the segmentation head initialise randomly, which is normal for PIDNet but means the
   loader must **report exactly which parameters were loaded and which were initialised**, and
   abort if the pretrained overlap is implausibly small. Silently tolerating missing keys is
   how a randomly-initialised backbone gets trained while everyone believes it is pretrained.

verify-data now reports `ok: true` with zero blocking failures and zero warnings.

---

## D022 - IDD withholds test-split GPS, so control supervision is train+val only

**Date:** 2026-07-17
**Evidence:** mm-inventory classified `primary/<route>/test.csv` as not-GPS. Inspecting it
shows why:

```
test.csv  : timestamp,image_idx                              1390 rows
train.csv : timestamp,image_idx,latitude,longitude,altitude  2211 rows
val.csv   : timestamp,image_idx,latitude,longitude,altitude   930 rows
```

The test split carries frame timestamps but **no position**, exactly as IDD withholds `gtFine`
for the segmentation test split. mm-inventory classifies tables by content rather than
filename, so it reached the right answer without being told.

**Decision:** no code change. Usable GPS per route is train+val = 3,141 rows, and the manifest
records `test.csv` under `other_tables` rather than discarding it, so the omission is visible
rather than silent. Frames with no position cannot supervise waypoints, so the control dataset
is built from train+val and re-split temporally per section 8 (docs/DECISIONS.md D011).

This is worth recording because "2 GPS tables, 3141 rows" looks like a discovery bug next to
the 3 CSVs on disk, and is not.

---

## D023 - LiDAR shares the camera/GPS clock; only OBD is on UTC

**Date:** 2026-07-17
**Evidence:** mm-inventory surfaced `supplement/lidar/<route>/timestamp.csv`
(`frame,datetime`), which D009 did not account for. Its clock matches GPS, not OBD:

```
route  lidar first        GPS first          OBD first
d0     09-00-31-462000    09-00-31-289685    03-31-13-000000
d2     16-00-03-128000    16-00-03-156000    10-30-38-536000
```

**Decision:** no change to D009, which this strengthens. Camera, GPS, and LiDAR all log local
Indian time and agree to within ~0.2 s; the OBD logger alone is on UTC. The 19800 s offset is
therefore a property of one logger, not an ambiguity in the dataset's timebase, which makes
the IST hypothesis considerably more than a coincidence of arithmetic.

LiDAR remains out of scope for v1 (section 4). These tables are inventoried into the manifest
under `other_tables` and otherwise unused.

---

## D019 - gen-masks writes via a symlink farm to keep seg/ read-only

**Date:** 2026-07-17
**Tension:** Upstream `createLabels.py` writes each mask beside its source polygon:

```python
dst = fn.replace("_polygons.json", "_label{}s.png".format(args.id_type))
```

Pointing `--datadir` at `seg/` would therefore write PNGs into `seg/gtFine/`, but section 4
declares `seg/` an extracted read-only input and puts generated masks under `masks/`.

**Options considered:** patch upstream's path handling (a larger, more fragile patch than the
seven already recorded, touching the code that computes every output name); or run against
`seg/` and move ~16K files afterwards (writes into a read-only tree, and a mid-run interrupt
leaves masks stranded in `seg/`).

**Decision:** mirror the outstanding polygon files into `masks/gtFine/<split>/<seq>/` as
symlinks and point `--datadir` at `masks/`. Upstream then writes its PNGs exactly where
section 4 wants them and never touches `seg/`. The symlinks are removed in a `finally` block,
so `masks/` ends up holding only PNGs even if the run is interrupted.

Section 7's idempotency requirement ("skip sequences whose outputs already exist") falls out of
the same mechanism: only polygons whose mask is missing get a symlink, so upstream's glob sees
exactly the outstanding work. Verified on a real 39-polygon subset: the second run reports
`generated: 0`.

---

## D018 - AutoNUE public-code has no licence

**Date:** 2026-07-17
**Evidence:** The upstream repository (commit `5d9a93beb176b03dd32c79ce050d0fbddc9acd00`,
2021-02-15) contains no LICENSE or COPYING file, and its README states no terms.

**Decision:** Vendor it as CLAUDE.md section 7 directs, and record the absence explicitly in
`third_party/autonue/PROVENANCE.txt` rather than assuming a permissive licence. The code is
published by the AutoNUE challenge organisers for use with the IDD dataset, which is the
context DRIVYX uses it in. This is surfaced because a downstream consumer of DRIVYX inherits
the question, and it is not one this project can answer by itself.

**Only 5 of the repository's files are vendored** (68 KB of 59 MB): the two rasterisers, the
entry point, the label table, and the annotation reader. The rest is domain-adaptation
training code DRIVYX does not use.

---

## D020 - The AutoNUE breakages are API rot, not Python 2 syntax

**Date:** 2026-07-17
**Spec expectation:** the master prompt says "AutoNUE public-code is Python 2 era in places:
vendor it under third_party/autonue/, patch minimally".

**Evidence:** there is no Python 2 syntax to fix. The files already carry
`from __future__ import print_function` and use Python 3 constructs throughout; grepping for
`print` statements, `iteritems`, `has_key`, `xrange`, and `except X, e` finds nothing. The
actual breakages are different in kind:

```
1,7  from PIL import PILLOW_VERSION   -> removed in Pillow 9.0 (device has 10.2.0);
                                         upstream sys.exit(-1)s before doing any work
2    tqdm.write() called, tqdm never imported -> NameError on the unknown-label path
3    imageio/numpngw imported at module scope, never used, not installed -> ImportError
4    helpers/ added to sys.path inside main(), after the module-scope imports that need it
5    tqdm.writeError() does not exist -> AttributeError, then division by zero
6    --color branch formats undefined `f` (parameter is `fn`) -> NameError
```

Patches 1 and 7 are the load-bearing ones: without them `gen-masks` produces nothing and
prints "Please install the module 'Pillow'" on a machine where Pillow is installed.

**Decision:** apply the seven patches documented line by line in `third_party/PATCHES.md`, each
justified by an observed failure. `anue_labels.py` and `annotation.py` are left byte-for-byte
unmodified, because `build-lut` resolves the 8-class collapse by name against that label table
and it must remain upstream's word.

---

## D021 - The section 7 grouping is consistent with anue_labels (verified, not assumed)

**Date:** 2026-07-17
**Requirement:** section 7: "every listed name must resolve to a level3Id or the build aborts
naming the miss; every level3Id present in anue_labels must land in exactly one group or
ignore, else abort".

**Evidence:** both directions hold on the real table. 40 labels reduce to 27 distinct level3Ids
(0 to 25, plus 255), and the collapse covers all 27 with no orphan and no conflict:

```
train_id  group          level3Ids
  0       drivable       [0]
  1       alt_drivable   [1]
  2       nondrivable    [2, 3, 13]
  3       vru            [4, 5]
  4       twowheeler     [6, 7]
  5       vehicle        [8, 9, 10, 11, 12]
  6       structure      [14..23]
  7       background     [24, 25]
  255     ignore         [255]
```

Two details worth recording, because they look like bugs and are not:

- `person` and `animal` share level3Id 4, and both are in `vru`, so the shared id is not a
  conflict. Similarly `parking`/`drivable fallback` share 1, and `caravan`/`trailer`/`train`/
  `vehicle fallback` share 12.
- `curb` has level3Id 13, which sits numerically among the vehicle ids (8 to 12) yet belongs to
  `nondrivable`. The grouping is by name, per section 7, so the numeric adjacency is
  irrelevant. A range-based collapse would have got this wrong.

**Decision:** no deviation needed. The checks stay in `build_lut()` as live assertions rather
than being reduced to this note, so an upstream label table change fails the build.

---

## D017 - Qt needs libxcb-cursor0, and silently renders offscreen without it

**Date:** 2026-07-17
**Evidence:** With a valid X11 session (`DISPLAY=:1`, `XDG_SESSION_TYPE=x11`), Qt reported:

```
qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed to load the Qt xcb
               platform plugin.
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
QApplication([]).platformName() -> 'offscreen'
```

`dpkg -l libxcb-cursor0` shows it is not installed. Qt 6.5+ requires it for the xcb platform
plugin. CLAUDE.md section 3 lists the PyQt6 install paths but not this transitive system
dependency, because the pip wheel does not declare it.

**The dangerous part is the failure mode.** Qt does not abort: it falls back to the
`offscreen` platform, so `drivyx-gui` starts, builds its whole widget tree, reports no error,
and draws nothing on the desktop. `import PyQt6` succeeds throughout, so the section 3 import
check cannot detect it.

**Decision:** setup_orin.sh probes `ldconfig -p` for `libxcb-cursor.so` and installs
`libxcb-cursor0` via apt when absent. Its verification step now constructs a QApplication and
asserts `platformName() != "offscreen"` whenever a display is present, so this fails at setup
instead of presenting as a GUI that launches into nothing. The check is skipped when there is
no DISPLAY, where offscreen is the correct answer and the engine is headless by design
(section 3).

This is the same class of defect as D015: a necessary-but-not-sufficient check reporting
success. Importing a library proves it loads, never that it works.

---

## D016 - Staged seg reality: 552 sequences (not ~350), and mixed jpg/png by part

**Date:** 2026-07-17
**Spec position:** section 4: "20,000 images (14K train / 2K val / 4K test) across 350 sequences"
and "leftImg8bit/{train,val,test}/<seq>/*.jpg|png".

**Evidence** (measured by verify-data after staging):

```
split   images  polygons  sequences  suffixes
train    14027     14027        369  {.jpg: 7034, .png: 6993}
val       2036      2036         72  {.jpg: 1055, .png:  981}
test      4038         0        111  {.png: 2029, .jpg: 2009}
total    20101                  552
```

Image counts match section 4 closely (drift 0.2% / 1.8% / 0.9%, total 20101 vs ~20000) and
pairing is exact: zero unpaired images, zero orphan polygons.

**Two findings:**

1. **Sequence count is 552, not ~350.** Section 4's figure is wrong or refers to Part I alone.
   This is informational: verify-data reports sequence counts but asserts only on image counts,
   because sequence granularity carries no integrity meaning. Nothing downstream depends on 350.

2. **The two parts use different image formats.** Part I (IDD_Segmentation) ships PNG and Part II
   (idd20kII) ships JPG, which is why the merged tree is mixed:

```
Part I  (png): 6993 train + 981 val + 2029 test = 10003
Part II (jpg): 7034 train + 1055 val + 2009 test = 10098
```

   The split is clean along part lines, which independently confirms the D006 merge was disjoint
   and correct. Section 4 already anticipates `*.jpg|png`, so every stage that reads images must
   accept both; gen-masks, pack-shards, and the shard reader are written against the suffix set,
   never a single extension.

---

## D015 - No published wheel has native sm_87 kernels; Orin is reached by PTX JIT

**Date:** 2026-07-17
**Evidence:** torch 2.13.0+cu130 imports and reports `torch.cuda.is_available() == True`, but warns:

```
Found GPU0 Orin which is of compute capability (CC) 8.7.
- 8.0 which supports hardware CC >=8.0,<9.0 except {8.7}
No published PyTorch CUDA builds for release 2.13.0+cu130 support this GPU.
```

`torch.cuda.get_arch_list()` is `['sm_80', 'sm_90', 'sm_100', 'sm_110', 'sm_120']`. The
`except {8.7}` is deliberate: PyTorch's SBSA aarch64 builds exclude Jetson Orin. sm_80 SASS is
not binary compatible with sm_87, so kernels reach this GPU only by the driver JIT-compiling the
embedded `compute_80` PTX.

**Search for a better wheel (all paths exhausted 2026-07-17):**

```
pypi.jetson-ai-lab.io indexes : jp6/cu126, jp6/cu128, jp6/cu129, sbsa/cu130, sbsa/dev, root/*
  jp6/*    -> cp310 only (torch-2.11.0-cp310-cp310-linux_aarch64.whl); this device is cp312
  sbsa/*   -> SBSA is Server Base System Architecture (GH200/Thor class), not Jetson Orin
  jp7      -> does not exist (404)
developer.download.nvidia.com/compute/redist/jp/v70|v71|v72/pytorch/ -> 307 then 404 (dead)
pypi.nvidia.com/torch -> 404
```

There is no published torch wheel with native sm_87 for Python 3.12 on JetPack 7.

**Verified by execution, not by label** (the decisive test, since `is_available()` cannot detect
missing kernels):

```
elementwise (torch native kernel)     OK
conv2d fp32 (cuDNN)                   OK
conv2d bf16 autocast channels_last    OK      <- exactly section 3's training config
backward + grad (bf16)                OK
batchnorm2d                           OK
interpolate bilinear                  OK
6/6 passed
```

**Measured throughput:**

```
bf16 matmul 4096^3 :  4.36 ms -> 31.52 TFLOPS  (~75% of Orin's ~42 TFLOPS dense bf16 peak)
conv3x3 bf16 B16   : 39.62 ms ->  8.78 TFLOPS  (largely bandwidth-bound at this shape)
```

**Decision:** Keep torch 2.13.0+cu130. The PTX JIT path executes every operation the training
config needs and reaches ~75% of peak on matmul, so the JIT is not a meaningful throughput
penalty (PTX JIT emits fully optimised SASS; its cost is a one-off compile per kernel, cached
under ~/.nv/ComputeCache, which is why the first conv2d call took 1.23 s and later ones 0.13 s).
The alternatives are a multi-hour source build with `TORCH_CUDA_ARCH_LIST=8.7`, or dropping to
python 3.10 to use the jp6 wheels against a CUDA 12.6 runtime on a 13.2 driver. Neither is
justified while the measured numbers hold.

**Consequence:** `setup_orin.sh` no longer trusts `torch.cuda.is_available()` alone. It executes a
conv2d + bf16 + backward smoke test and asserts the loss and gradients are finite, so a wheel with
no usable kernels fails at setup rather than at hour six of a training run. The earlier version of
this script printed "Environment verified" while torch was simultaneously reporting that no build
supported this GPU: a false pass, now fixed.

**Revisit if:** M4's `--probe` shows secs/epoch far worse than the projected schedule, which is
the real throughput gate.

---

## D014 - python3-venv is missing and setup_orin.sh installs it

**Date:** 2026-07-17
**Evidence:** The first setup_orin.sh run on this device aborted at venv creation:

```
The virtual environment was not created successfully because ensurepip is not
available.  On Debian/Ubuntu systems, you need to install the python3-venv
package using the following command.

    apt install python3.12-venv
```

Ubuntu splits venv support out of the base `python3` package and this image shipped without
it. CLAUDE.md section 3 mandates a `--system-site-packages` venv but assumes venv works.

**Decision:** setup_orin.sh probes `import ensurepip` and, when absent, installs the versioned
`python${major}.${minor}-venv` apt package (falling back to unversioned `python3-venv`), then
re-probes and aborts if it is still unavailable. The version is derived from the running
interpreter rather than hardcoded to 3.12, so the fix survives a python upgrade. The script
also detects a structurally valid but pip-less venv left behind by a failed run and rebuilds
it, rather than failing later with an obscure pip error. This is the master prompt's
"fix it inside scripts/setup_orin.sh so the fix is reproducible" rule applied.

---

## D013 - Repository is git-initialised for the run directory contract

**Date:** 2026-07-17
**Evidence:** The working directory was not a git repository; section 6.3 requires each run
directory to record a git SHA for reproducibility.

**Decision:** `git init` the repository so the SHA contract can be honoured. Environment capture
degrades gracefully and records `uncommitted` when the tree has no commit yet, rather than failing
a training run over provenance metadata.
