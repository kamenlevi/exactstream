#!/bin/sh
# Breakthrough: the R-45 fix check (Fit12 R-45 PASS (b), Thu 00:42: the per-request creep is the FR-Spec draft head, kept
# alive by the cycle of mtplx's local class _FullVocabDraftHead until a gen-2 GC). Candidate = live bt_b12 (c3f957cba6)
# + B12_FRS_NOCYCLE (default 1: the class's closure drops the pruned head / ids once built), in BT_DIR ($1); the live
# file stays frozen for the Eval suite's R-41C. Lines written before the run (SCOREBOARD):
#   gc_fx   Fit12's harness (scripts/creep_fix.py = fit_creep.py + BT_DIR), fold-exact, R-41C's first 6 math rows in one
#           process. PASS: tokens = the suite's b12llfx rows on all 6; gc finds no mx garbage on rows 2-6 (<= 0.01 GB each;
#           unfixed 0.095, Fit12 c1); MLX active at a row end <= 9.34 GB (unfixed 9.427); row maxima <= Fit12 c1's + 0.02.
#           KILL: a token differs, or >= 0.05 GB of mx garbage on any row 2-6
#   gc_mtp  fold_check p2k, p8k, 128 tokens: every hash = def_mtp's (the draft head's own output included: logits)
# NIGHT_OK=1 only while Kamen's lift of the silent night holds (orchestrator Thu 00:28); PAUSE still stops it.
# Log: data/gcfix/summary.txt
cd "$HOME/Breakthrough" || exit 1
OUT=$HOME/Breakthrough/data/gcfix; mkdir -p "$OUT"; LOG=$OUT/summary.txt
. scripts/lockrun.sh
SP=$1
[ -f "$SP/bt_b12.py" ] || { echo "usage: runner_gc.sh <dir with the candidate bt_b12.py>"; exit 2; }
log "runner_gc start (pid $$), candidate $SP/bt_b12.py"
caffeinate -i -w $$ &
item gc_fx 600 "R-45 fix check, 6 math rows (~7 min)" \
  env BT_DIR=$SP F12_NROWS=6 F12_OUT=$OUT $PY scripts/creep_fix.py gc_fx
item gc_mtp 420 "R-45 fix check, MTP p2k + p8k (~3 min)" \
  env -u B12_ENGINE -u B12_FOLD -u B12_PFR_GRID BT_DIR=$SP FC_PROMPTS=p2k,p8k FC_NDEC=128 $PY scripts/fold_check.py gc_mtp
log "runner_gc done"
