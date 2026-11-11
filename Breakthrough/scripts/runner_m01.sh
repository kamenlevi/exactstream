#!/bin/sh
# Breakthrough R-48 M01-12 (NewIdeas Thu 17:2x; lines in SCOREBOARD 4b90156, written before the run): BigMoE's M-01
# records (records3m, the per-group fields re-packed at a fixed width, exact) on the 12 GB line. Engine v3n = v3m +
# BigMoE's unpack (BIGMOE_M01=1); A = v3m on records3. One 14-min quiet lock: the page-cache check, then A1 S1 S2 A2
# (scripts/m01_arms.sh -> m01_ab.py per arm: p2k, p8k, write16 at fold-exact defaults, 256 tokens), then m01_sum.py.
#   PASS: every hash equal in all 4 runs AND decode SSD bytes/token (rusage) -2.5 % or better AND lifetime peak not
#         higher (max S <= max A + 0.02 GB). KILL: any hash differs, or a saving under 1.5 %. BETWEEN: tok/s decides.
#   INVALID (re-run): A1 vs A2 decode bytes/token differ by more than 1 %, or C_m / C_l differ.
# Log: data/m01/summary.txt
cd "$HOME/Breakthrough" || exit 1
OUT=$HOME/Breakthrough/data/m01; mkdir -p "$OUT"; LOG=$OUT/summary.txt
. scripts/lockrun.sh
log "runner_m01 start (pid $$)"
caffeinate -i -w $$ &
LOCK_MIN=14 item m01_ab 830 "R-48 M01-12 A/B: A1 S1 S2 A2 (~9 min)" sh scripts/m01_arms.sh
log "m01_ab: $(grep -E 'VERDICT|noise' "$OUT/m01_ab.txt" | tr '\n' ' ')"
log "runner_m01 done"
