#!/bin/sh
# CA run 3 (Leap, Mon 2026-10-26 ~19:10; QUEUE row CA): the per-part split of an exact resident check by knockout
# differences in one process, no trace (xctrace's temp files stopped runs 1 and 2). diffsmoke (1K, 16 rows, 1 round)
# must finish with every commit landed, then diff (1K + 8K x 8 / 16 / 32 rows, 6 paired rounds). ~3 min.
# Run under a held quiet lock via run_locked.sh, cwd ~/Leap.   usage: ca3_batch.sh
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
D=$HOME/Leap/data
say() { echo "[ca3] $* $(date +%H:%M:%S); swap $(sysctl -n vm.swapusage | cut -d'(' -f1| sed 's/ *$//'), disk $(df -g / | tail -1 | awk '{print $4}') GB free"; }
trap 'exit 143' TERM INT
say "start"
$PY scripts/ca_anat.py diffsmoke > $D/ca_diffsmoke.log 2>&1 || { say "diffsmoke FAILED"; tail -20 $D/ca_diffsmoke.log; exit 1; }
tail -4 $D/ca_diffsmoke.log
say "diff"
$PY scripts/ca_anat.py diff > $D/ca_diff.log 2>&1 || { say "diff FAILED"; tail -20 $D/ca_diff.log; exit 1; }
grep -v '^\[' $D/ca_diff.log | tail -20
say "done"
