#!/usr/bin/env bash
#
# DRIVYX data staging: populate <data_root> from the downloaded IDD archives.
#
# CLAUDE.md section 4 describes the data as "already on disk" and read-only. On this device
# it is not: the archives sit unextracted in the download directory. This script bridges
# that gap and is the deviation recorded in docs/DECISIONS.md D006.
#
# It is kept separate from setup_orin.sh because environment setup is fast and re-runnable
# while this is a one-time, multi-hour, roughly 50 GB extraction.
#
# Idempotent: each archive records a marker on success and is skipped on re-run.
#
# Usage:
#   bash scripts/stage_data.sh [--source DIR] [--data-root DIR] [--force ARCHIVE]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SOURCE_DIR=""
DATA_ROOT=""
FORCE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE_DIR="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --force) FORCE="$2"; shift 2 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_INFO=$'\033[34m'; C_OK=$'\033[32m'
  C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_BOLD=$'\033[1m'
else
  C_RESET=""; C_INFO=""; C_OK=""; C_WARN=""; C_ERR=""; C_BOLD=""
fi
info() { echo "${C_INFO}[ .. ]${C_RESET} $*"; }
ok()   { echo "${C_OK}[ ok ]${C_RESET} $*"; }
warn() { echo "${C_WARN}[warn]${C_RESET} $*" >&2; }
die()  { echo "${C_ERR}[FAIL]${C_RESET} $*" >&2; exit 1; }
step() { echo; echo "${C_BOLD}== $* ==${C_RESET}"; }

# --- resolve paths from the single source of truth ---------------------------------------

# configs/paths.yaml is authoritative (section 4). Parsing it here rather than duplicating
# the default keeps the shell and the python engine from drifting apart.
read_config() {
  python3 - "$1" <<'PYCODE'
import sys, pathlib, yaml
key = sys.argv[1]
cfg = yaml.safe_load(pathlib.Path("configs/paths.yaml").read_text()) or {}
value = cfg.get(key)
if value is None:
    sys.exit(f"configs/paths.yaml has no key '{key}'")
print(pathlib.Path(value).expanduser())
PYCODE
}

[[ -n "$DATA_ROOT" ]]   || DATA_ROOT="$(read_config data_root)"
[[ -n "$SOURCE_DIR" ]]  || SOURCE_DIR="$(read_config archive_source)"

RAW="$DATA_ROOT/raw"
SEG="$DATA_ROOT/seg"
MM="$DATA_ROOT/multimodal"
PRETRAINED="$DATA_ROOT/pretrained"
MARKERS="$DATA_ROOT/.staged"
STAGE_TMP="$DATA_ROOT/.stage_tmp"

step "Plan"
echo "  source    : $SOURCE_DIR"
echo "  data_root : $DATA_ROOT"

[[ -d "$SOURCE_DIR" ]] || die "Archive source $SOURCE_DIR does not exist."

# --- archive manifest -------------------------------------------------------------------
#
# name|archive|kind|strip_prefix
#   seg archives are merged into a single tree: Parts I and II each carry their own top
#   level directory (IDD_Segmentation/ and idd20kII/) wrapping the same
#   leftImg8bit/ + gtFine/ structure, so both are stripped one level into seg/.
#   The multimodal zips all share an idd_multimodal/ prefix and are staged then moved.

SEG_ARCHIVES=(
  "seg_part1|idd-segmentation.tar.gz|IDD_Segmentation"
  "seg_part2|idd-20k-II.tar.gz|idd20kII"
)
MM_ARCHIVES=(
  "mm_primary|idd_mm_primary.zip"
  "mm_secondary|idd_mm_secondary.zip"
  "mm_supplement|idd_mm_supplement.zip"
)

# --- 1. create the data root -------------------------------------------------------------

step "Data root"

# DECISIONS.md D005: /mnt/nvme does not exist and /mnt is root-owned, so creating the data
# root needs sudo exactly once. The root filesystem is itself the NVMe, so this directory is
# physically on the NVMe as section 4 intends.
if [[ ! -d "$DATA_ROOT" ]]; then
  parent="$(dirname "$DATA_ROOT")"
  if [[ -w "$parent" ]] 2>/dev/null; then
    mkdir -p "$DATA_ROOT"
  else
    info "Creating $DATA_ROOT (needs sudo; $parent is not writable)"
    sudo mkdir -p "$DATA_ROOT"
    sudo chown "$(id -u):$(id -g)" "$DATA_ROOT"
  fi
  ok "Created $DATA_ROOT"
else
  ok "$DATA_ROOT exists"
fi

[[ -w "$DATA_ROOT" ]] || die "$DATA_ROOT is not writable by $(id -un)."

mkdir -p "$RAW" "$SEG" "$MM" "$PRETRAINED" "$MARKERS" "$STAGE_TMP"

# --- 2. disk space check -----------------------------------------------------------------

step "Disk space"

archive_bytes=0
for entry in "${SEG_ARCHIVES[@]}" "${MM_ARCHIVES[@]}"; do
  archive="$(echo "$entry" | cut -d'|' -f2)"
  path="$SOURCE_DIR/$archive"
  [[ -f "$path" ]] || die "Missing archive: $path"
  size=$(stat -c %s "$path")
  archive_bytes=$((archive_bytes + size))
done

# Compressed image data expands by roughly 1.2x here: the payload is already JPEG/PNG, so
# gzip/deflate mostly repacks rather than compresses. 1.5x is the safety margin.
need_bytes=$((archive_bytes * 3 / 2))
avail_bytes=$(( $(stat -f -c %a "$DATA_ROOT") * $(stat -f -c %S "$DATA_ROOT") ))

printf "  archives  : %.1f GB\n" "$(echo "$archive_bytes" | awk '{print $1/1e9}')"
printf "  need (est): %.1f GB\n" "$(echo "$need_bytes" | awk '{print $1/1e9}')"
printf "  available : %.1f GB\n" "$(echo "$avail_bytes" | awk '{print $1/1e9}')"

if (( avail_bytes < need_bytes )); then
  die "Not enough free space on $DATA_ROOT. Need ~$((need_bytes/1000000000)) GB, have $((avail_bytes/1000000000)) GB."
fi
ok "Sufficient space"

# --- 3. preserve archives in raw/ --------------------------------------------------------

step "Preserving archives in raw/"

# Section 4: raw/ holds the "original archives (never deleted by code)". Hardlink when the
# source is on the same filesystem: same inode, zero extra bytes, and the data survives
# deletion of either name. Fall back to a copy across filesystems.
for entry in "${SEG_ARCHIVES[@]}" "${MM_ARCHIVES[@]}"; do
  archive="$(echo "$entry" | cut -d'|' -f2)"
  src="$SOURCE_DIR/$archive"
  dst="$RAW/$archive"
  if [[ -e "$dst" ]]; then
    continue
  fi
  if ln "$src" "$dst" 2>/dev/null; then
    info "hardlinked $archive"
  else
    info "copying $archive (cross-filesystem)"
    cp --reflink=auto "$src" "$dst"
  fi
done
ok "raw/ holds $(find "$RAW" -maxdepth 1 -type f | wc -l) archives"

# --- 4. extract segmentation -------------------------------------------------------------

step "Segmentation (Parts I + II -> seg/)"

for entry in "${SEG_ARCHIVES[@]}"; do
  name="$(echo "$entry" | cut -d'|' -f1)"
  archive="$(echo "$entry" | cut -d'|' -f2)"
  prefix="$(echo "$entry" | cut -d'|' -f3)"
  marker="$MARKERS/$name"

  if [[ -f "$marker" && "$FORCE" != "$name" && "$FORCE" != "all" ]]; then
    ok "$name already staged ($(cat "$marker"))"
    continue
  fi

  info "Extracting $archive -> seg/ (stripping $prefix/)"
  # --keep-old-files turns a sequence-id collision between Parts I and II into a hard error
  # instead of a silent overwrite, per the fail-loudly rule. The two parts are expected to
  # carry disjoint sequence ids.
  if ! tar -xzf "$RAW/$archive" -C "$SEG" --strip-components=1 --keep-old-files; then
    die "Extraction of $archive failed. If the error is 'File exists', Parts I and II share a
       sequence id, which means the archives are not the disjoint pair section 4 assumes.
       Inspect $SEG before re-running with --force $name."
  fi
  date -Iseconds > "$marker"
  ok "$name extracted"
done

for split in train val test; do
  count=$(find "$SEG/leftImg8bit/$split" -type f \( -name '*.png' -o -name '*.jpg' \) 2>/dev/null | wc -l)
  seqs=$(find "$SEG/leftImg8bit/$split" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  echo "  leftImg8bit/$split : $count images across $seqs sequences"
done
for split in train val; do
  count=$(find "$SEG/gtFine/$split" -type f -name '*_polygons.json' 2>/dev/null | wc -l)
  echo "  gtFine/$split      : $count polygon files"
done

# --- 5. extract multimodal ---------------------------------------------------------------

step "Multimodal (primary + secondary + supplement -> multimodal/)"

for entry in "${MM_ARCHIVES[@]}"; do
  name="$(echo "$entry" | cut -d'|' -f1)"
  archive="$(echo "$entry" | cut -d'|' -f2)"
  marker="$MARKERS/$name"

  if [[ -f "$marker" && "$FORCE" != "$name" && "$FORCE" != "all" ]]; then
    ok "$name already staged ($(cat "$marker"))"
    continue
  fi

  info "Extracting $archive"
  # All three zips share an idd_multimodal/ top level, so they are unzipped into a staging
  # directory and their contents moved up into multimodal/.
  rm -rf "${STAGE_TMP:?}/idd_multimodal"
  unzip -q -o "$RAW/$archive" -d "$STAGE_TMP" || die "unzip of $archive failed."

  [[ -d "$STAGE_TMP/idd_multimodal" ]] || die "$archive did not contain the expected idd_multimodal/ prefix.
       Contents: $(ls "$STAGE_TMP" | head -5)"

  for sub in "$STAGE_TMP/idd_multimodal"/*; do
    [[ -e "$sub" ]] || continue
    target="$MM/$(basename "$sub")"
    if [[ -e "$target" ]]; then
      info "merging into existing $(basename "$sub")/"
      cp -rn "$sub"/. "$target"/
      rm -rf "$sub"
    else
      mv "$sub" "$target"
    fi
  done
  rm -rf "${STAGE_TMP:?}/idd_multimodal"

  date -Iseconds > "$marker"
  ok "$name extracted"
done

rmdir "$STAGE_TMP" 2>/dev/null || true

echo "  multimodal top level: $(find "$MM" -mindepth 1 -maxdepth 1 -type d -printf '%f ' 2>/dev/null)"
echo "  multimodal files    : $(find "$MM" -type f | wc -l)"

# --- 6. pretrained -----------------------------------------------------------------------

step "Pretrained backbone"

if find "$PRETRAINED" -maxdepth 1 -type f \( -name '*.pth' -o -name '*.pth.tar' \) | grep -q .; then
  ok "backbone present: $(find "$PRETRAINED" -maxdepth 1 -type f | head -1)"
else
  # Not fatal: section 4 says verify-data prints the hint, and M1 through M3 do not need it.
  warn "No PIDNet-S backbone in $PRETRAINED."
  echo "       Download PIDNet_S_ImageNet.pth.tar from https://github.com/XuJiacong/PIDNet"
  echo "       (see its Pretrained Models section) and place it at:"
  echo "         $PRETRAINED/PIDNet_S_ImageNet.pth.tar"
  echo "       train-seg (M4) aborts without it; earlier milestones are unaffected."
fi

step "Done"
echo "  Next: drivyx verify-data | python -m json.tool"
