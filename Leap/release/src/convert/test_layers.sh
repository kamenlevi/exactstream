#!/bin/bash
# convert/test_layers.sh <model dir> <reference records3 dir> <work dir> [layers ...]   (default layers: 0 47)
# The conversion test: runs the three store-building steps on a few layers and compares the result byte for byte with an
# existing converted store (CPU + SSD only, no GPU):
#   1. convert/repack_records.py   safetensors -> records/L.bin (the 9 expert pieces per record, 16 KB-aligned)
#   2. convert/make_records3.py    records/ -> records3/ row-major (3-bit + 1-bit parts) + its verifier on every expert
#                                  (--no-mlx: the byte checks run, the MLX cross-checks are skipped)
#   3. convert/convert_records_p11.py --pad   row-major -> native+p11+pad, in place, with its inverse check
# then cmp of each layer file with the reference and a comparison of the layer's index entry. One layer at a time
# (about 4.4 GB of extra disk), cleaned after each. FLOOR_GB (default 25) = the free disk the steps keep.
set -uo pipefail
MD=${1:?model dir (safetensors + config.json)}; REF=${2:?reference records3 dir}; W=${3:?work dir}; shift 3
LAYERS=${*:-0 47}
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); PY=${PY:-$ROOT/.venv/bin/python}; FLOOR=${FLOOR_GB:-25}
mkdir -p "$W"; W=$(cd "$W" && pwd -P); R="$W/convert_report.txt"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$R"; }
say "conversion test: layers $LAYERS, model $MD, reference $REF, floor $FLOOR GB"
fails=0
for L in $LAYERS; do
  rm -rf "$W/records" "$W/records3"; mkdir -p "$W/records" "$W/records3"
  t0=$(date +%s)
  "$PY" "$ROOT/convert/repack_records.py" "$MD" --layers "$L" --min-free-gb "$FLOOR" --out "$W/records" >> "$R" 2>&1 \
    || { say "L$L: repack_records FAILED"; fails=$((fails + 1)); continue; }
  t1=$(date +%s)
  "$PY" "$ROOT/convert/make_records3.py" --layers "$L" --src "$W/records" --out "$W/records3" --config "$MD/config.json" \
    --min-free-gb "$FLOOR" --device cpu --no-mlx >> "$R" 2>&1 || { say "L$L: make_records3 FAILED"; fails=$((fails + 1)); continue; }
  t2=$(date +%s)
  rm -rf "$W/records"                                      # records/ is not needed after the split
  "$PY" "$ROOT/convert/convert_records_p11.py" "$W/records3" --pad --layers "$L" --yes --floor-gb "$FLOOR" >> "$R" 2>&1 \
    || { say "L$L: convert_records_p11 FAILED"; fails=$((fails + 1)); continue; }
  t3=$(date +%s)
  f=$(printf 'L%02d.bin' "$L")
  if cmp -s "$W/records3/$f" "$REF/$f"; then
    say "L$L: $f == the reference byte for byte ($(stat -f%z "$REF/$f") B); repack $((t1 - t0)) s, split + verify $((t2 - t1)) s, p11 $((t3 - t2)) s"
  else
    say "L$L: $f DIFFERS from the reference: $(cmp "$W/records3/$f" "$REF/$f" 2>&1 | head -1)"; fails=$((fails + 1))
  fi
  "$PY" - "$W/records3/index.json" "$REF/index.json" "$L" >> "$R" 2>&1 <<'EOF'
import json, sys
a, b = (json.load(open(p))["layers"][sys.argv[3]] for p in sys.argv[1:3])
keys = sorted(set(a) | set(b))
diff = [k for k in keys if a.get(k) != b.get(k)]
print(f"  index entry L{sys.argv[3]}: " + ("== the reference" if not diff else f"DIFFERS in {diff}"))
sys.exit(1 if diff else 0)
EOF
  [ $? = 0 ] && say "L$L: index entry == the reference" || { say "L$L: index entry DIFFERS (see $R)"; fails=$((fails + 1)); }
  rm -rf "$W/records3"
done
[ $fails = 0 ] && say "CONVERSION TEST PASS (layers $LAYERS)" || say "CONVERSION TEST FAIL ($fails)"
exit $fails
