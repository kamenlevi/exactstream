#!/bin/sh
# Breakthrough: the fold-ON port check (NewIdeas GO Tue 21:0x, before Fit12's R-01), one 10-min quiet lock
# (scripts/lockrun.sh). MTP path, p2k then p8k, 256 tokens, engine v3m, B12_FOLD=1 (PFR on the MTP prefill, C_m per
# prompt: 65 / 63, QSA rows target, MOV, ALLWAIT inside spec125.generate). Lines (written before the run, SCOREBOARD):
# clean within 11.95 GB, and token / kv / head hashes = Fit12's PFM / AW arms for these settings (p2k sha256 a0774ec635,
# kv 37028a6545, head 8563204e11; p8k logits 7d045d7b7d, sha256 ed67050746, kv 5a760388d6, head f59755cf27). Any
# difference = the port: stop and diff before R-01. Log: data/fold/summary.txt
cd "$HOME/Breakthrough" || exit 1
OUT=$HOME/Breakthrough/data/fold; mkdir -p "$OUT"; LOG=$OUT/summary.txt
. scripts/lockrun.sh
log "runner_foldon start (pid $$)"
caffeinate -i -w $$ &
item fold_on 420 "fold-ON port check: MTP p2k + p8k on v3m, B12_FOLD=1 (~3 min)" \
  env B12_ENGINE=v3m B12_FOLD=1 $PY scripts/fold_check.py fold_on
log "runner_foldon done"
