#!/bin/sh
# Breakthrough: pre-flight of the Eval suite's fold-fast variant (b12llff = B12_FOLD_AW=1 B12_FOLD_CM=1
# B12_PFR_GRID=balanced) on the LIVE bt_b12 (31b63af defaults underneath), before the suite spends hours on it. That env
# combination has not run on the rebuilt code (R-41 part 3 checked fold-exact only). Lines written before the run:
#   ff_mtp   fold_check p2k, p8k, p16k, p32k in one process, 256 tokens. PASS: rc 0; p2k / p8k every hash = the
#            regression's (fold on = off there; FR-Spec fix draft logits): p2k 98c6bc1c0d 82995cdb4e a0774ec635 37028a6545
#            8563204e11, p8k 61b2520e58 0b0d91cd43 ed67050746 5a760388d6 f59755cf27; C_m per prompt = the fold points
#            (64 / 60 / 56 / 47); ALLWAIT calls > 0 on every prompt. Peaks recorded (the 11.95 line is the suite's verdict)
#   ff_w16   write16, same env: = the suite's ON row (old fold-on = ALLWAIT + C_m + balanced grid)
#   ff_w16k  write16, grid keep (B12_FOLD_AW=1 B12_FOLD_CM=1): = the suite's OFF row (the bisect: ALLWAIT alone = OFF)
#   KILL (tell the suite to hold b12llff, fix first): a crash, a p2k / p8k hash mismatch, C_m not resized, ALLWAIT 0
# Log: data/ff/summary.txt
cd "$HOME/Breakthrough" || exit 1
OUT=$HOME/Breakthrough/data/ff; mkdir -p "$OUT"; LOG=$OUT/summary.txt
. scripts/lockrun.sh
FF="B12_FOLD_AW=1 B12_FOLD_CM=1"
log "runner_ff start (pid $$), live bt_b12, $FF"
caffeinate -i -w $$ &
item ff_mtp 600 "fold-fast pre-flight, MTP p2k-p32k (~5 min)" \
  env -u BT_DIR $FF B12_PFR_GRID=balanced FC_PROMPTS=p2k,p8k,p16k,p32k FC_NDEC=256 $PY scripts/fold_check.py ff_mtp
item ff_w16 300 "fold-fast pre-flight, write16 (~1 min)" \
  env -u BT_DIR $FF B12_PFR_GRID=balanced $PY scripts/w16_bisect.py ff_w16
item ff_w16k 300 "fold-fast pre-flight, write16 grid keep (~1 min)" \
  env -u BT_DIR $FF $PY scripts/w16_bisect.py ff_w16k
log "runner_ff done"
