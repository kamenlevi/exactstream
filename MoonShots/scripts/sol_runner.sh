#!/bin/sh
# MoonShots sol_runner: speed-of-light GDN layer, v2 HC (RB 2 and 4) and v1 HC; per-dispatch parts. Launch holding the lock.
OUT=$HOME/MoonShots/data/day14; PY=$HOME/BigMoE/.venv/bin/python; S=$HOME/MoonShots/patches/sol_layer/sol_layer.py
echo "$(date +%H:%M) sol_runner started" >> "$OUT/summary.txt"
for v in "2 2 4" "2 4 8" "1 2 16"; do
  set -- $v
  f="$OUT/sol_layer_hc$1_rb$2_ra$3.txt"
  SOL_HC=$1 SOL_HCD_RB=$2 SOL_G4_RA=$3 $PY $S > "$f" 2>&1
  echo "$(date +%H:%M) done sol hc$1 rb$2 ra$3: $(grep -E '^(M=|Traceback|[A-Za-z]*Error)' "$f" | cut -c1-110 | tr '\n' ' ')" >> "$OUT/summary.txt"
done
$PY $HOME/MoonShots/patches/sol_layer/persist_pair.py > "$OUT/persist_pair.txt" 2>&1
echo "$(date +%H:%M) done persist_pair: $(grep -E '^(M=|tiny|    persistent|Traceback|[A-Za-z]*Error)' "$OUT/persist_pair.txt" | cut -c1-120 | tr '\n' ' ')" >> "$OUT/summary.txt"
$PY $HOME/MoonShots/patches/sol_layer/ref_gemv.py > "$OUT/ref_gemv.txt" 2>&1
echo "$(date +%H:%M) done ref_gemv: $(grep -cE 'GB/s' "$OUT/ref_gemv.txt") shapes" >> "$OUT/summary.txt"
$PY $HOME/MoonShots/patches/planes_engine/test_tierglu4.py > "$OUT/test_tierglu4_bc.txt" 2>&1
echo "$(date +%H:%M) done test_tierglu4_bc (records4 fused combine): $(grep -E '^(PASS|FAIL)|Error' "$OUT/test_tierglu4_bc.txt" | head -3 | tr '\n' ' ')" >> "$OUT/summary.txt"
peer status "MoonShots: 35B refit test (activation-aware 2/3-bit tiers, ~15 min) under the same lock" > /dev/null 2>&1
sh $HOME/MoonShots/scripts/refit_runner.sh
SOL_G4V=mlx $PY $S > "$OUT/sol_layer_v3_mlx.txt" 2>&1
echo "$(date +%H:%M) done sol v3 (MLX GEMVs + epilogues): $(grep -E '^(M=|Traceback|[A-Za-z]*Error)' "$OUT/sol_layer_v3_mlx.txt" | cut -c1-110 | tr '\n' ' ')" >> "$OUT/summary.txt"
peer unlock > /dev/null 2>&1
echo "$(date +%H:%M) sol_runner finished" >> "$OUT/summary.txt"
