#!/bin/bash
# Friday afternoon's never-wait map run (Breakthrough, Fri 2026-11-06 13:0x; NewIdeas' GPU ORDER item 6, b6e90b5, 12:5x):
# FAR's NEVER-WAIT tier 1 + 1b on code, moved up from Saturday night, plus the 36-window reading NewIdeas asked for (all three
# domains). Every run takes a share lock "MEM 18 GB" and goes alone: the runners wait while another session holds or waits for
# a quiet window or declares a shared run with MEM > 5 GB, the lid is closed, PAUSE is set, or NO-SWAP is low; a quiet window
# (3 strikes), swap +2 GB or the hard stop stops a run. Run ATTACHED (Bash run_in_background), cwd ~/Breakthrough.
#   1. the GPU self-check (gslot.sh nw_selfcheck, ~2 min), last start 13:30 (NewIdeas: "if it cannot start by 13:30, it stays
#      first in tonight's map"). Exit 0 (PASS or REPORT) runs the arms; anything else runs none, and tonight's sat_night.sh
#      (unchanged, armed for Sat 00:30) runs them as planned;
#   2. the s67 runner on data/fri_nw_extra.txt, copied into data/s67_extra.txt for the run (the old file is put back after, so
#      tonight's sat_night.sh saves Friday night's list as designed): tier 1, tier 1b, then the 36-window arms. Hard stop 14:10
#      (EvalSuite's long-code smoke follows; NewIdeas: "please end by ~14:15");
#   3. nw_read.py (CPU) into the log. The self-check's outputs are copied to data/dm_s7/results_nw/ (tonight's sat_night.sh
#      runs the self-check again and rewrites data/nw_selfcheck/).
# An arm done here is skipped tonight (the runner's todo() counts its windows); an arm not done stays in tonight's list.
# Log: data/fri_nw.log; the runner's own lines go to data/s67_slot.log, the self-check's to data/gslot/nw_selfcheck.log.
export BIGMOE_EXT_BUILD=build_nat
H=~/Breakthrough; PY=~/BigMoE/.venv/bin/python; LOG=$H/data/fri_nw.log; X=$H/data/s67_extra.txt; L=$H/data/fri_nw_extra.txt
FAR=~/FableGod/far/build_1105; DM=~/FableGod/decision_map; R=$H/data/dm_s7/results_nw; KEEP=$H/data/s67_extra_before_fri_nw.txt
say() { echo "[fri_nw] $* $(date '+%a %H:%M:%S')" >> $LOG; }
kids() { local p=$1 c; for c in $(pgrep -P $p); do kids $c; echo $c; done; }
SAVED=
stopall() {   # once: stop my runs (TERM, then KILL if frozen; never SIGCONT), put s67_extra.txt back, unlock
  [ -n "$DONE_STOP" ] && return; DONE_STOP=1; trap - EXIT INT TERM HUP
  for c in $(kids $$); do kill -TERM $c 2>/dev/null; done; sleep 5
  for c in $(kids $$); do ps -o stat= -p $c 2>/dev/null | grep -q T && kill -KILL $c 2>/dev/null; done
  [ -n "$SAVED" ] && cp $KEEP $X && say "s67_extra.txt put back"
  peer unlock >> $LOG 2>&1; say "ended ($1)"
}
trap 'stopall "exit"' EXIT; trap 'stopall "stopped"; exit 1' INT TERM HUP
grep -q "NEVER-WAIT" $DM/dm_common.py || { say "never-wait harness patch not installed: nothing run"; exit 1; }
say "armed: self-check (last start 13:30), then $(grep -v '^#' $L | awk '{print $1}' | tr '\n' ' '); hard stop 14:10"

# 0. (Fri 13:0x) The first launch's self-check started at 13:01 beside Fit12's kp2, a 12 GB timing run under a share lock
#    "100% GPU" that started at 13:00. Its "MEM <n> GB" is cut off in peer ls, so the runners' guard (MEM > 5 GB) could not
#    see it; that launch was stopped at 13:02. So first wait while another session's shared run declares >= 50 % GPU.
gpu_others() { peer ls 2>/dev/null | grep -v "^You:" | sed -n 's/.*shared run[^:]*: about [0-9]*% CPU, \([0-9]*\)% GPU.*/\1/p' | awk '$1 >= 50 {f = 1} END {exit !f}'; }
LG=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-06 13:30:00" +%s); GW=
while gpu_others; do
  [ $(date +%s) -ge $LG ] && { say "another session's shared GPU run is still up at 13:30: nothing run; tonight's sat_night.sh keeps the arms"; exit 1; }
  [ -z "$GW" ] && { GW=1; say "waiting: another session's shared run declares >= 50 % GPU"; }; sleep 20
done
[ -n "$GW" ] && say "no other shared GPU run now"

# 1. the GPU self-check
GS_LASTGO="2026-11-06 13:30:00" GS_DEADLINE="2026-11-06 14:10:00" bash $H/scripts/dm_s7/gslot.sh nw_selfcheck 4 18 \
  env BIGMOE_EXT_BUILD=build_nat $PY $FAR/nw_selfcheck.py & wait $!
rc=$?
say "never-wait self-check: rc $rc, verdict '$(tail -1 $H/data/nw_selfcheck/VERDICT 2>/dev/null)'"
mkdir -p $R; [ -d $H/data/nw_selfcheck ] && cp -R $H/data/nw_selfcheck $R/nw_selfcheck_fri; cp $H/data/gslot/nw_selfcheck.log $R/nw_selfcheck_fri.log 2>/dev/null
[ $rc = 0 ] || { say "no arms run (the self-check gave neither PASS nor REPORT, or could not start by 13:30); tonight's sat_night.sh keeps them"; exit 1; }

# 2. the runner
cp $X $KEEP && SAVED=1
{ grep -v '^#' $L; echo "ALONE 0 00:00"; } > $X
now="$(date '+%Y-%m-%d %H:%M:%S')"
S67_START="$now" S67_LASTGO="2026-11-06 14:02:00" S67_DEADLINE="2026-11-06 14:10:00" S67_XALONE="$now" bash $H/scripts/dm_s7/s67_slot.sh & wait $!
say "runner ended (rc $?)"
cp $KEEP $X && SAVED= && say "s67_extra.txt put back"

# 3. the read-out (CPU, the map's JSONs only)
$PY -I -B $FAR/nw_read.py >> $LOG 2>&1; say "nw_read: $(grep '^VERDICT' $LOG | tail -8 | tr '\n' ' ')"
exit 0
