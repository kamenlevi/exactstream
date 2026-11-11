#!/bin/sh
# Breakthrough R-41 part 2: the grid hypothesis (write16's 4096 grid ends in an 84-row MoE chunk; PFR's balanced grid
# makes one chunk). One quiet lock per arm (scripts/lockrun.sh), w16_bisect.py, 64 tokens:
#   pfr_keep    MTP, fold PFR only, B12_PFR_GRID=keep (release the pools, keep the 4096 grid)      expect = OFF
#   exact_keep  MTP, fold-exact (MOV + PFR-keep + rows target, C_m 51, no ALLWAIT)                  expect = OFF
#   ar_pfr1 / ar_pfr0 / ar_keep   AR path, write16's prompt: shipped PFR (balanced) / PFR off / PFR-keep; shas compared
# Log: data/w16/summary.txt
cd "$HOME/Breakthrough" || exit 1
OUT=$HOME/Breakthrough/data/w16; mkdir -p "$OUT"; LOG=$OUT/summary.txt
. scripts/lockrun.sh
SP=/private/tmp/claude-501/-Users-kamen-OjoX/2c7bb7fe-3ba1-4c03-91ba-f6be33886b82/scratchpad/btbis
ON="B12_FOLD=1 B12_FOLD_CM=2700:64,8600:60,16500:56,32800:47 B12_FOLD_ROWS=2048"
log "runner_w16b start (pid $$) (AR arms only: the 16:2x AR arms crashed at load on the MTP env; harness fixed)"
caffeinate -i -w $$ &
w() { item "w16_$1" 300 "R-41 write16 grid check, arm $1 (~2 min)" env $2 $PY scripts/w16_bisect.py "w16_$1"; }
w ar_pfr1    "B12_ENGINE=v3l BI_AR=1 B12_PFR=1"
w ar_pfr0    "B12_ENGINE=v3l BI_AR=1 B12_PFR=0"
w ar_keep    "B12_ENGINE=v3l BI_AR=1 B12_PFR=1 B12_PFR_GRID=keep BT_DIR=$SP"
log "runner_w16b done"
