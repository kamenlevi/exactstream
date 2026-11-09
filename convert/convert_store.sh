#!/bin/bash
# convert/convert_store.sh <model dir> <output records3 dir> [layers ...]   (default: every MoE layer of the model)
# Builds the converted expert store one layer at a time (CPU + SSD only, no GPU). For each layer:
#   1. convert/repack_records.py   safetensors -> <out>.work/records/LNN.bin (the 9 expert pieces per record, 16 KB-aligned)
#   2. convert/make_records3.py    -> <out>/LNN.bin row-major (3-bit + 1-bit parts), with its verifier on every expert
#                                  (--no-mlx: the byte checks run, the MLX cross-checks are skipped), and the layer's entry
#                                  in <out>/index.json, merged with the layers already there
#   3. convert/convert_records_p11.py --pad   that layer -> native+p11+pad, in place, with its inverse check
# A layer's 4-bit pieces are deleted after step 2, so the extra disk at a time is one layer's pieces plus one layer's
# rewrite (about 3-4.4 GB) on top of the growing store (48 layers x 1.43 GB = 68.5 GB). FLOOR_GB (default 25) is the free
# disk every step keeps. A layer already native+p11+pad in index.json is skipped, so a stopped run resumes; the first
# failing step stops the run. convert/test_layers.sh runs the same steps on a few layers and compares the bytes with an
# existing store. Point the engine at the result by putting it at <model dir>/records3. The report and the 4-bit pieces live
# in <out>.work beside it: convert_records_p11.py refuses to rewrite a layer while any file in <out> is open.
set -uo pipefail
MD=${1:?model dir (safetensors + config.json)}; OUT=${2:?output records3 dir}; shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); PY=${PY:-$ROOT/.venv/bin/python}; FLOOR=${FLOOR_GB:-25}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd -P); WK="$OUT.work"; mkdir -p "$WK"; R="$WK/convert_store_report.txt"; TMP="$WK/records"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$R"; }
if [ $# -gt 0 ]; then LAYERS="$*"
else LAYERS=$("$PY" -I -c 'import json, sys; wm = json.load(open(sys.argv[1] + "/model.safetensors.index.json"))["weight_map"]; print(" ".join(str(l) for l in sorted({int(k.split(".layers.")[1].split(".")[0]) for k in wm if ".mlp.switch_mlp.gate_proj.weight" in k})))' "$MD"); fi
[ -n "$LAYERS" ] || { say "no MoE layers found in $MD/model.safetensors.index.json"; exit 2; }
layout() { [ -f "$OUT/index.json" ] && "$PY" -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["layers"].get(sys.argv[2], {}).get("layout", "none"))' "$OUT/index.json" "$1" || echo none; }
say "convert_store: model $MD -> $OUT, layers $(echo $LAYERS | wc -w | tr -d ' '), floor $FLOOR GB"
T0=$(date +%s); n=0
for L in $LAYERS; do
  if [ "$(layout "$L")" = "native+p11+pad" ]; then say "L$L: already native+p11+pad, skipped"; continue; fi
  rm -rf "$TMP"; mkdir -p "$TMP"
  t0=$(date +%s)
  "$PY" "$ROOT/convert/repack_records.py" "$MD" --layers "$L" --min-free-gb "$FLOOR" --out "$TMP" >> "$R" 2>&1 \
    || { say "L$L: repack_records FAILED (see $R)"; exit 1; }
  t1=$(date +%s)
  "$PY" "$ROOT/convert/make_records3.py" --layers "$L" --src "$TMP" --out "$OUT" --config "$MD/config.json" \
    --min-free-gb "$FLOOR" --device cpu --no-mlx >> "$R" 2>&1 || { say "L$L: make_records3 FAILED (see $R)"; exit 1; }
  t2=$(date +%s)
  rm -rf "$TMP"                                            # the 4-bit pieces are not needed after the split
  "$PY" "$ROOT/convert/convert_records_p11.py" "$OUT" --pad --layers "$L" --yes --floor-gb "$FLOOR" >> "$R" 2>&1 \
    || { say "L$L: convert_records_p11 FAILED (see $R)"; exit 1; }
  t3=$(date +%s); n=$((n + 1))
  say "L$L: native+p11+pad; repack $((t1 - t0)) s, split + verify $((t2 - t1)) s, p11 $((t3 - t2)) s"
done
say "STORE DONE: $n layers converted in $(( $(date +%s) - T0 )) s; $(ls "$OUT"/L*.bin 2>/dev/null | wc -l | tr -d ' ') layer files in $OUT"
