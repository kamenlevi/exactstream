#!/bin/sh
# Breakthrough FFX (Thu 29 Oct ~15:5x; lines written before the run, SCOREBOARD): fold-fast (b12llff's env) on the R-45
# fix candidate (BT_DIR = $1) with C_m capped at 60 up to 8.6K-token prompts (B12_FOLD_CM points
# 2700:60,8600:60,16500:56,32800:47; the default gives short prompts C_m 64). R-41C's first 8 math rows of the b12llff
# process, in one process, through Fit12's harness (scripts/creep_fix.py, F12_ARM=ff). For NewIdeas' R-41C part B call
# (keep the pin, or move B to the fix with a C_m cap). Fit12 R-45F (gc between rows = the fix's effect): 12.212 GB at C_m 64,
# so 4 steps x 0.103 GB -> ~11.80 expected.
#   PASS: tokens = the suite's b12llff rows on all 8 (C_m is bits-neutral under ALLWAIT; the fix is exact) and lifetime
#         peak <= 11.85 GB (0.10 GB left for the suite's 16-row process). BETWEEN: 11.85-11.95 (cap 59 then).
#   KILL: a token differs, or the lifetime peak > 11.95 GB
# Log: data/ffx/summary.txt
cd "$HOME/Breakthrough" || exit 1
OUT=$HOME/Breakthrough/data/ffx; mkdir -p "$OUT"; LOG=$OUT/summary.txt
. scripts/lockrun.sh
SP=$1
[ -f "$SP/bt_b12.py" ] || { echo "usage: runner_ffx.sh <dir with the candidate bt_b12.py>"; exit 2; }
log "runner_ffx start (pid $$), candidate $SP/bt_b12.py"
caffeinate -i -w $$ &
item ffx 600 "FFX fold-fast + R-45 fix, C_m cap 60, 8 math rows (~8 min)" \
  env BT_DIR=$SP F12_ARM=ff F12_CM=2700:60,8600:60,16500:56,32800:47 F12_NROWS=8 F12_OUT=$OUT $PY scripts/creep_fix.py ffx
log "runner_ffx done"
