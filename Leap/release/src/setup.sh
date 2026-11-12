#!/bin/bash
# setup.sh: the whole install in one command, on an Apple-silicon Mac (INSTALL.md has the manual steps it automates).
#
#   bash setup.sh                  # everything, into ~/exactstream-data (or --data DIR)
#   bash setup.sh --dry-run        # check the Mac and report what is done and what would run; changes nothing
#
# 1. Checks the Mac: Apple silicon, macOS, memory (64 GB; a 36 GB Mac runs only --mem 24, untested on such a Mac),
#    Python 3.12, CMake 3.27 or newer, the Xcode command-line tools, and the free disk that what is left needs.
# 2. Creates .venv and builds the extension (install.sh without the weights), so a missing tool stops it before any
#    download.
# 3. Downloads the three public pieces WEIGHTS.md lists; no account or login is needed:
#    - the model: MTPLX's 4-bit pack of Qwen3.8-Flash-Next on Hugging Face, every file but the vision tower (105.4 GB),
#      into DIR/model;
#    - the block drafter's base: DFlash2 (1.2 GB), into the Hugging Face cache, where run.py looks for it;
#    - our two drafter files (1.4 GB, release v1.0.0 of this repository), into DIR/drafters.
#    Revisions, sizes and checksums are pinned in setup/files.tsv. A stopped download resumes where it stopped.
# 4. Converts the expert store into DIR/model/records3 (convert/convert_store.sh: 68.5 GB, about 10 minutes on an M5 Pro,
#    CPU and SSD only; a stopped run resumes at the next unfinished layer).
# 5. Runs install.sh --store DIR/model --weights DIR/drafters.
# Run it again at any time: each step checks what is done (files present at their sizes, layers converted) and skips it.
#
# Options: --data DIR (default ~/exactstream-data, or EXACTSTREAM_DATA); --model DIR and --drafters DIR (pieces you already
# have elsewhere); --verify (also check every file's checksum, which reads about 107 GB); --dry-run; --no-download (stop
# instead of downloading); --python PY, --jobs N, --offline, --find-links DIR (passed to install.sh).
set -uo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd -P)
DATA=${EXACTSTREAM_DATA:-$HOME/exactstream-data}; MODEL=""; DRAFT=""; DRY=0; VERIFY=0; NODL=0; PY=""; IARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATA=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --drafters) DRAFT=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --verify) VERIFY=1; shift ;;
    --no-download) NODL=1; shift ;;
    --python) PY=$2; IARGS+=("$1" "$2"); shift 2 ;;
    --jobs|--find-links) IARGS+=("$1" "$2"); shift 2 ;;
    --offline) IARGS+=("$1"); shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "setup.sh: unknown option $1 (see --help)"; exit 2 ;;
  esac
done
MODEL=${MODEL:-$DATA/model}; DRAFT=${DRAFT:-$DATA/drafters}
F=$ROOT/setup/files.tsv
REPO=Youssofal/Qwen3.8-Flash-Next-MTPLX-Bare-Speed; REV=4285a4a04041511bf6581f543fdbfd154c93a608
DREPO=incoai/Qwen3.8-Flash-Next-DFlash2; DREV=a22a1744e85c17364dcc54a6a420af7135d4c7f6
HUB=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
DSNAP=$HUB/models--incoai--Qwen3.8-Flash-Next-DFlash2/snapshots/$DREV
STORE_LAYER=1427000000; STORE_WORK=4400000000; FLOOR=25000000000; VENV=1000000000
say() { echo "[setup] $*"; }
die() { echo "[setup] STOP: $*"; exit 2; }
gb() { awk -v b="$1" 'BEGIN { printf "%.1f GB", b / 1e9 }'; }
size() { stat -L -f %z "$1" 2>/dev/null || echo -1; }
[ -f "$F" ] || die "no $F (run setup.sh from the repository)"

# ---- 1. the Mac ----
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || die "this needs an Apple-silicon Mac (arm64 macOS)"
OSV=$(sw_vers -productVersion); MEMG=$(( $(sysctl -n hw.memsize) / 1073741824 ))
say "this Mac: $(sysctl -n machdep.cpu.brand_string), $MEMG GB of memory, macOS $OSV"
[ "${OSV%%.*}" -ge 26 ] 2>/dev/null || say "WARNING: tested on macOS 26.5 only; macOS $OSV is untested"
if [ $MEMG -ge 64 ]; then say "memory: every setting fits (--mem 45, 32 and 24)"
elif [ $MEMG -ge 36 ]; then say "WARNING: with $MEMG GB only --mem 24 fits, and it is untested on a Mac this size (INSTALL.md)"
else die "$MEMG GB of memory: the smallest setting (--mem 24) needs a Mac with 36 GB or more"; fi
if [ -z "$PY" ]; then
  for c in python3.12 /opt/homebrew/opt/python@3.12/bin/python3.12 /usr/local/bin/python3.12; do
    command -v "$c" >/dev/null 2>&1 && { PY=$(command -v "$c"); break; }
  done
fi
[ -n "$PY" ] && "$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12)' 2>/dev/null \
  || die "Python 3.12 not found (for example: brew install python@3.12; or --python /path/to/python3.12)"
CMV=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}')
[ -n "$CMV" ] || die "CMake not found (for example: brew install cmake)"
awk -v v="$CMV" 'BEGIN { split(v, a, "."); exit !(a[1] > 3 || (a[1] == 3 && a[2] >= 27)) }' || die "CMake $CMV is older than 3.27"
xcode-select -p >/dev/null 2>&1 || die "the Xcode command-line tools are missing (xcode-select --install)"
command -v git >/dev/null 2>&1 && command -v curl >/dev/null 2>&1 || die "git and curl are needed"
say "tools: Python $("$PY" -c 'import platform; print(platform.python_version())'), CMake $CMV, Xcode command-line tools"

# ---- what is done ----
missing() {    # <piece> <dir>: "<files missing or of the wrong size> <their bytes>"
  local n=0 b=0 pc p s h src
  while IFS=$'\t' read -r pc p s h src; do
    [ "$pc" = "$1" ] || continue
    [ "$(size "$2/$p")" = "$s" ] || { n=$((n + 1)); b=$((b + s)); }
  done < <(grep -v '^#' "$F")
  echo "$n $b"
}
nfiles() { grep -c "^$1"$'\t' "$F"; }
store_left() { # MoE layers not yet converted (records3/index.json layout native+p11+pad and the layer file present)
  "$PY" -I - "$MODEL" <<'PYX'
import json, os, sys
m = sys.argv[1]
try:
    wm = json.load(open(os.path.join(m, "model.safetensors.index.json")))["weight_map"]
except Exception:
    print(48, 48); sys.exit()
L = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in wm if ".mlp.switch_mlp.gate_proj.weight" in k})
try:
    lay = json.load(open(os.path.join(m, "records3", "index.json")))["layers"]
except Exception:
    lay = {}
done = [l for l in L if lay.get(str(l), {}).get("layout") == "native+p11+pad" and os.path.isfile(os.path.join(m, "records3", f"L{l:02d}.bin"))]
print(len(L) - len(done), len(L))
PYX
}
dflash_ok() { [ "$(size "$DSNAP/config.json")" = 1236 ] && [ "$(size "$DSNAP/model.safetensors")" = 1206140008 ]; }
venv_ok() { [ -x "$ROOT/.venv/bin/python" ] && ls "$ROOT"/engine/ext/build/bigmoe_ext*.so >/dev/null 2>&1; }
status() {
  read -r MN MB < <(missing model "$MODEL"); read -r WN WB < <(missing drafters "$DRAFT"); read -r SL ST < <(store_left)
  DB=0; dflash_ok || DB=1206140008
  VB=0; venv_ok || VB=$VENV
  NEED=$((MB + WB + DB + SL * STORE_LAYER + VB + FLOOR)); [ "$SL" -gt 0 ] && NEED=$((NEED + STORE_WORK))
  d=$DATA; [ -d "$MODEL" ] && d=$MODEL; while [ ! -d "$d" ]; do d=$(dirname "$d"); done
  FREE=$(df -k "$d" | awk 'NR == 2 {printf "%d", $4 * 1024}')
}
status
say "data folder: $DATA (model: $MODEL, drafter files: $DRAFT)"
say "1/5 venv + extension: $(venv_ok && echo done || echo 'to do (install.sh)')"
say "2/5 model ($REPO @ ${REV:0:7}, $(nfiles model) files, 105.4 GB): $([ "$MN" = 0 ] && echo done || echo "$MN files to download ($(gb $MB))")"
say "3/5 drafter base ($DREPO @ ${DREV:0:7}, 1.2 GB, Hugging Face cache): $(dflash_ok && echo done || echo 'to download')"
say "    drafter files (release v1.0.0, 1.4 GB): $([ "$WN" = 0 ] && echo done || echo "$WN files to download ($(gb $WB))")"
say "4/5 expert store ($MODEL/records3, 68.5 GB): $([ "$SL" = 0 ] && echo "done ($ST layers)" || echo "$SL of $ST layers to convert")"
say "5/5 install.sh --store --weights: runs last, every time (it links the store and the drafter files)"
say "disk: $(gb $FREE) free where the data goes; what is left needs $(gb $NEED) (including $(gb $FLOOR) kept free for the conversion and swap)"
say "licences: the model is Qwen3.8-Flash-Next by Qwen (Copyright (c) 2026 Qwen), under the Qwen Community License 1.0. Its"
say "  weights and everything derived from them, the converted expert store too, stay under that licence, not this"
say "  repository's GPL-3, and its conditions bind whoever uses them (third_party/LICENSE-qwen). The drafter base, DFlash2 by"
say "  Inco AI, is Apache-2.0; our drafter files: l120_ft.safetensors under Apache-2.0, d22_b5.safetensors under the Qwen"
say "  Community License 1.0 (third_party/NOTICE-l120_ft, third_party/NOTICE-d22_b5)."

verify_piece() {   # <piece> <dir>: checksums of every file of the piece that has one (sha256, or the git blob id)
  "$PY" -I - "$F" "$1" "$2" <<'PYX'
import hashlib, os, sys
f, piece, d = sys.argv[1:4]
bad = n = 0
for ln in open(f):
    if ln.startswith("#") or not ln.strip():
        continue
    pc, p, s, h, _ = ln.rstrip("\n").split("\t")
    if pc != piece or h == "-":
        continue
    kind, want = h.split(":")
    g = hashlib.sha256() if kind == "sha256" else hashlib.sha1(b"blob %d\0" % int(s))
    with open(os.path.join(d, p), "rb") as fh:
        while True:
            b = fh.read(1 << 24)
            if not b:
                break
            g.update(b)
    n += 1
    if g.hexdigest() != want:
        bad += 1
        print(f"[setup] CHECKSUM MISMATCH: {os.path.join(d, p)}")
print(f"[setup] checksums: {piece}: {n - bad} of {n} match")
sys.exit(1 if bad else 0)
PYX
}

if [ $DRY = 1 ]; then
  [ "$FREE" -ge "$NEED" ] || say "the disk is short by $(gb $((NEED - FREE))): a real run would stop here"
  if [ $VERIFY = 1 ]; then
    [ "$MN" = 0 ] && { verify_piece model "$MODEL" || true; }
    [ "$WN" = 0 ] && { verify_piece drafters "$DRAFT" || true; }
  fi
  say "DRY RUN: nothing changed, nothing downloaded"; exit 0
fi
[ "$FREE" -ge "$NEED" ] || die "$(gb $FREE) free, but what is left needs $(gb $NEED). Free some space (or use --data on a larger disk), then run setup.sh again"

# ---- 2. venv + extension (before any download) ----
if venv_ok; then say "venv + extension: done"
else bash "$ROOT/install.sh" ${IARGS[@]+"${IARGS[@]}"} || die "install.sh failed (see above); run setup.sh again after fixing it"; fi
hfdl() { HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1 "$ROOT/.venv/bin/hf" download "$@"; }
nodl() { [ $NODL = 0 ] || die "--no-download is set, but $1 still need downloading"; }

# ---- 3. downloads ----
if [ "$MN" != 0 ]; then
  nodl "the model files"; mkdir -p "$MODEL"
  say "downloading the model: $MN files, $(gb $MB) (it resumes if stopped)"
  hfdl "$REPO" --revision "$REV" --exclude model-vision.safetensors --local-dir "$MODEL" >/dev/null \
    || die "the model download stopped; run setup.sh again to resume"
  read -r MN MB < <(missing model "$MODEL"); [ "$MN" = 0 ] || die "after the download, $MN model files still differ from setup/files.tsv"
fi
if ! dflash_ok; then
  nodl "the drafter base files"; say "downloading the drafter base (1.2 GB) into the Hugging Face cache ($HUB)"
  hfdl "$DREPO" config.json model.safetensors --revision "$DREV" >/dev/null || die "the drafter base download stopped; run setup.sh again"
  dflash_ok || die "the drafter base is not complete in $DSNAP after the download"
fi
if [ "$WN" != 0 ]; then
  nodl "the drafter files"; mkdir -p "$DRAFT"
  while IFS=$'\t' read -r pc p s h url; do
    [ "$pc" = drafters ] || continue
    [ "$(size "$DRAFT/$p")" = "$s" ] && continue
    say "downloading $p ($(gb $s))"
    [ "$(size "$DRAFT/$p.part")" = "$s" ] || curl -fL --retry 5 --retry-delay 5 -C - -o "$DRAFT/$p.part" "$url" \
      || die "the download of $p stopped; run setup.sh again to resume"
    [ "$(shasum -a 256 "$DRAFT/$p.part" | cut -d' ' -f1)" = "${h#sha256:}" ] || { rm -f "$DRAFT/$p.part"; die "$p: checksum mismatch (removed; run setup.sh again)"; }
    mv "$DRAFT/$p.part" "$DRAFT/$p"
  done < <(grep -v '^#' "$F")
fi
if [ $VERIFY = 1 ]; then
  say "checking every file's checksum (reads about 107 GB)"
  verify_piece model "$MODEL" && verify_piece drafters "$DRAFT" || die "a checksum differs (see above): delete that file and run setup.sh again"
fi

# ---- 4. the expert store ----
read -r SL ST < <(store_left)
if [ "$SL" != 0 ]; then
  say "converting the expert store: $SL of $ST layers (about 10 minutes on an M5 Pro; it resumes if stopped)"
  PY="$ROOT/.venv/bin/python" bash "$ROOT/convert/convert_store.sh" "$MODEL" "$MODEL/records3" \
    || die "the conversion stopped (its report: $MODEL/records3.work/convert_store_report.txt); run setup.sh again to resume"
  read -r SL ST < <(store_left); [ "$SL" = 0 ] || die "$SL layers are still not converted"
fi

# ---- 5. install.sh: links the store and the drafter files, checks the drafter base ----
bash "$ROOT/install.sh" ${IARGS[@]+"${IARGS[@]}"} --store "$MODEL" --weights "$DRAFT" || die "install.sh failed (see above)"
say "DONE. Try: .venv/bin/python run.py --config chat --mem 24 --prompt 'Explain how a refrigerator works.'"
say "  or Claude Code on this model: tools/claude_code.sh (INSTALL.md)"
