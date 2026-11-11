#!/bin/bash
# Install NEVER-WAIT v3 (b)'s nw3 arms into the decision map (Breakthrough, Fri 2026-11-06 21:xx; NewIdeas QUEUE dc9aff8 S1
# + 6fd0224). The map's patch recipe: the anchored patch (nw3_patch.py) on scratch copies, the CPU tests, then an atomic
# move with the old files kept beside as *.pre_nw3.
#   CPU tests, each first on the live files (--base) and then on the patched ones:
#   - Fable FAR's never-wait suite (~/FableGod/far/build_1105/test_nw.py, read only; its scratch goes to ours): every old
#     arm bit-equal, never-wait's own checks unchanged;
#   - test_nw3.py: nw3's own checks (sections A-H).
# NW3_TEST_ONLY=1: patch and test, install nothing. Exit 0 = installed (or already in, or tests passed in test-only mode),
# 1 = failed (the map's files unchanged), 2 = a map process is running (nothing done). Log: <scratch>/install.log.
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; H=~/Breakthrough/scripts/dm_s7; FAR=~/FableGod/far/build_1105
S=~/Breakthrough/data/abl_scratch/nw3/$(date +%m%d_%H%M%S); mkdir -p $S/base $S/patched; LOG=$S/install.log
say() { echo "[install_nw3] $* $(date '+%a %H:%M:%S')" | tee -a $LOG; }
pgrep -f "run_ma[p].py" > /dev/null && { say "a map process is running: nothing done"; exit 2; }
if grep -q "NEVER-WAIT v3 (b)" $DM/dm_common.py && grep -q "nw3_mask" $DM/dm_ablations.py; then say "already in"; exit 0; fi
cp -p $DM/dm_common.py $DM/dm_ablations.py $S/base/ || exit 1
m0c=$(md5 -q $DM/dm_common.py); m0a=$(md5 -q $DM/dm_ablations.py); say "live: dm_common.py $m0c, dm_ablations.py $m0a; scratch $S"
$PY -I -B $H/nw3_patch.py $S/base $S/patched >> $LOG 2>&1 || { say "the patch FAILED (an anchor): nothing installed"; tail -3 $LOG; exit 1; }
run() {   # $1 = label, $2 = log, rest = the command; exit 1 unless rc 0 and (a --base run, or its last line is ALL OK)
  "${@:3}" > $S/$2 2>&1; local rc=$?
  if [ $rc = 0 ] && { [[ " ${*:3} " == *" --base "* ]] || tail -1 $S/$2 | grep -q "^ALL OK"; }; then say "$1: rc 0, $(tail -1 $S/$2 | cut -c1-90)"
  else say "$1: FAILED (rc $rc; $S/$2)"; tail -12 $S/$2; exit 1; fi
}
mkdir -p $S/prenw && cp -p $DM/dm_common.py.pre_nw $S/prenw/dm_common.py && cp -p $DM/dm_ablations.py.pre_nw $S/prenw/dm_ablations.py || exit 1
run "FAR's suite, its base = the pre-never-wait files" far_base.log env NW_TEST_SCRATCH=$S/far $PY -I -B $FAR/test_nw.py --base $S/prenw
run "FAR's suite on the patched files" far_patched.log env NW_TEST_SCRATCH=$S/far $PY -I -B $FAR/test_nw.py $S/patched
run "test_nw3 on the live files" t3_base.log env NW3_TEST_SCRATCH=$S $PY -I -B $H/test_nw3.py --base $S/base
run "test_nw3 on the patched files" t3_patched.log env NW3_TEST_SCRATCH=$S $PY -I -B $H/test_nw3.py $S/patched
[ -n "$NW3_TEST_ONLY" ] && { say "test-only: nothing installed (patched files in $S/patched)"; exit 0; }
[ "$(md5 -q $DM/dm_common.py)" = "$m0c" ] && [ "$(md5 -q $DM/dm_ablations.py)" = "$m0a" ] || { say "the live files changed during the tests: nothing installed"; exit 1; }
pgrep -f "run_ma[p].py" > /dev/null && { say "a map process started: nothing installed"; exit 2; }
cp -p $DM/dm_common.py $DM/dm_common.py.pre_nw3 && cp -p $DM/dm_ablations.py $DM/dm_ablations.py.pre_nw3 || exit 1
cp $S/patched/dm_common.py $DM/dm_common.py.new && cp $S/patched/dm_ablations.py $DM/dm_ablations.py.new || exit 1
mv $DM/dm_common.py.new $DM/dm_common.py && mv $DM/dm_ablations.py.new $DM/dm_ablations.py || { say "the move FAILED"; exit 1; }
say "installed: dm_common.py $(md5 -q $DM/dm_common.py), dm_ablations.py $(md5 -q $DM/dm_ablations.py); old files as *.pre_nw3"
