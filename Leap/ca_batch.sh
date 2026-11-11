#!/bin/sh
# CA (Leap, Mon 2026-10-26; NewIdeas QUEUE row CA, pre-registered in ~/Leap/TABLE_350.md 'CA'): anatomy of an exact
# resident check at 8 / 16 / 32 rows on a 12-layer slice of the 125B. smoke (P's tokens = A's, commits land) -> time
# (A / B walls, drafting) -> prof under xctrace (B2 / P, one command buffer per part) -> export the GPU intervals ->
# ca_report.py. The trace goes to a 2 GB RAM disk (the SSD is at its 100 GB floor), detached on exit.
# Run under a held quiet lock via run_locked.sh (~15 min), cwd ~/Leap.   usage: ca_batch.sh
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
D=$HOME/Leap/data; RD=/Volumes/LeapRAM; TR=$RD/ca.trace
say() { echo "[ca] $* $(date +%H:%M:%S); swap $(sysctl -n vm.swapusage | cut -d'(' -f1| sed 's/ *$//'), disk $(df -g / | tail -1 | awk '{print $4}') GB free"; }
unram() { [ -d $RD ] && hdiutil detach $RD > /dev/null 2>&1; }
trap unram EXIT
trap 'exit 143' TERM INT
say "start"
if [ ! -d $RD ]; then
  DEV=$(hdiutil attach -nomount ram://4194304 | awk '{print $1}')
  diskutil erasevolume HFS+ LeapRAM $DEV > /dev/null || { say "RAM disk FAILED"; hdiutil detach $DEV; exit 1; }
fi
if [ ! -s $D/ca_time.json ]; then                         # run 1 (14:05) finished smoke + time before its stop
  say "smoke"
  $PY scripts/ca_anat.py smoke > $D/ca_smoke.log 2>&1 || { say "smoke FAILED"; tail -20 $D/ca_smoke.log; exit 1; }
  tail -4 $D/ca_smoke.log
  say "time"
  $PY scripts/ca_anat.py time > $D/ca_time.log 2>&1 || { say "time FAILED"; tail -20 $D/ca_time.log; exit 1; }
fi
say "draft pieces"
$PY scripts/ca_draft.py > $D/ca_draft.log 2>&1 || { say "draft FAILED"; tail -20 $D/ca_draft.log; }
# The GPU instrument alone: 'Metal System Trace' also streams GPU counters (4.7 GB in 40 s on run 1, written to the
# per-user temp dir whatever the output path; health ACT at disk < 100 GB). The GPU instrument's kernel trace is
# ~11 KB per command buffer (~0.3 GB here); it is left in the temp dir and deleted by hand after the run.
say "prof (GPU instrument trace)"
rm -rf "$TR"
xcrun xctrace record --instrument GPU --output "$TR" --launch -- $PY scripts/ca_anat.py prof \
  > $D/ca_prof.log 2>&1 || { say "prof FAILED"; tail -20 $D/ca_prof.log; exit 1; }
du -sh "$TR"
xcrun xctrace export --input "$TR" --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \
  > $RD/ca_intervals.xml 2> $D/ca_export.log || { say "export FAILED"; tail -5 $D/ca_export.log; exit 1; }
rm -rf "$TR"
ls -la $RD/ca_intervals.xml
gzip -c $RD/ca_intervals.xml > $D/ca_intervals.xml.gz
$PY scripts/ca_report.py $RD/ca_intervals.xml > $D/ca_report.log 2>&1 || { say "report FAILED"; tail -20 $D/ca_report.log; exit 1; }
say "done"
