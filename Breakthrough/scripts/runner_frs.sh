#!/bin/sh
# Breakthrough: the FR-Spec rebuild fix's check (NewIdeas GO Tue ~21:4x; written before the run). bt_b12's
# b12_prefill_state now rebuilds the FR-Spec draft head with the prompt's ids first (scratch _bt_b12_frs.py: parsed,
# then moved live inside the first item, under the lock; Fit12's R-01 imports its own copy, F12_BTC=1). One 10-min quiet
# lock per item (scripts/lockrun.sh), engine v3m, MTP path, 256 tokens:
#   frs_off   B12_FOLD=0, prompts p2k, p8k, p8k
#   frs_on    B12_FOLD=1, prompts p8k (process-fresh), p2k, p8k
# PASS: every p8k run has the same prefill logits hash (lpre) as the process-fresh p8k (frs_on pos 1), i.e. a repeated
# prompt = a fresh one; and sha256 / kv / head are unchanged vs the regression (p2k a0774ec635 / 37028a6545 / 8563204e11,
# p8k ed67050746 / 5a760388d6 / f59755cf27) in both runs. The logits hash itself is EXPECTED to move where a prompt has
# ids outside the base 64K FR-Spec set (the first draft now sees them): draft-only. Log: data/fold/summary.txt
cd "$HOME/Breakthrough" || exit 1
OUT=$HOME/Breakthrough/data/fold; mkdir -p "$OUT"; LOG=$OUT/summary.txt
. scripts/lockrun.sh
log "runner_frs start (pid $$)"
caffeinate -i -w $$ &
item frs_off 420 "FR-Spec fix check, fold off: MTP p2k, p8k, p8k on v3m (~4 min)" \
  sh -c "$PY -I -c 'import ast, sys; ast.parse(open(sys.argv[1]).read())' scripts/_bt_b12_frs.py && mv scripts/_bt_b12_frs.py \
scripts/bt_b12.py && env B12_ENGINE=v3m B12_FOLD=0 FC_PROMPTS=p2k,p8k,p8k $PY scripts/fold_check.py frs_off" || {
  log "frs_off failed: see frs_off.txt (bt_b12 swapped only if the parse passed)"; exit 1; }
item frs_on 420 "FR-Spec fix check, fold on: MTP p8k, p2k, p8k on v3m (~4 min)" \
  env B12_ENGINE=v3m B12_FOLD=1 FC_PROMPTS=p8k,p2k,p8k $PY scripts/fold_check.py frs_on
log "runner_frs done"
