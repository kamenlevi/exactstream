#!/bin/sh
# Breakthrough R-41 (NewIdeas Wed 10-28): the write16 bisect, one quiet lock per arm (scripts/lockrun.sh), each arm a
# fresh process decoding the Eval suite's agent/write16 like decode._gen_mtp (scripts/w16_bisect.py, 64 tokens; the
# fold-on / fold-off rows split at token 32). Controls on the LIVE bt_b12 (= the suite's arms), pieces on the scratch
# copy with per-piece switches (BT_DIR; defaults = the live fold). Log: data/w16/summary.txt
cd "$HOME/Breakthrough" || exit 1
OUT=$HOME/Breakthrough/data/w16; mkdir -p "$OUT"; LOG=$OUT/summary.txt
. scripts/lockrun.sh
SP=/private/tmp/claude-501/-Users-kamen-OjoX/2c7bb7fe-3ba1-4c03-91ba-f6be33886b82/scratchpad/btbis
ON="B12_FOLD=1 B12_FOLD_CM=2700:64,8600:60,16500:56,32800:47 B12_FOLD_ROWS=2048"
log "runner_w16 start (pid $$)"
caffeinate -i -w $$ &
w() { item "w16_$1" 300 "R-41 write16 bisect, arm $1 (~2 min)" env $2 $PY scripts/w16_bisect.py "w16_$1"; }
w off      "B12_ENGINE=v3l B12_FOLD=0"
w on       "B12_ENGINE=v3m $ON"
w on_bis   "B12_ENGINE=v3m $ON BT_DIR=$SP"
w exact    "B12_ENGINE=v3m $ON BT_DIR=$SP B12_FOLD_AW=0 B12_FOLD_CM=0"
w pfr      "B12_ENGINE=v3m $ON BT_DIR=$SP B12_FOLD_AW=0 B12_FOLD_CM=0 B12_FOLD_MOV=0 B12_FOLD_ROWS=0"
w rows     "B12_ENGINE=v3m $ON BT_DIR=$SP B12_FOLD_AW=0 B12_FOLD_CM=0 B12_FOLD_MOV=0 B12_FOLD_PFR=0"
w mov      "B12_ENGINE=v3m $ON BT_DIR=$SP B12_FOLD_AW=0 B12_FOLD_CM=0 B12_FOLD_PFR=0 B12_FOLD_ROWS=0"
w pfrcm    "B12_ENGINE=v3m $ON BT_DIR=$SP B12_FOLD_AW=0 B12_FOLD_MOV=0 B12_FOLD_ROWS=0"
w aw       "B12_ENGINE=v3l B12_FOLD=0 B12_ALLWAIT=1"
log "runner_w16 done"
