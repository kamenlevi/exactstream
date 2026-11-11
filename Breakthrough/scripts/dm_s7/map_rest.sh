#!/bin/bash
# The rest of the map, Saturday (Breakthrough, Fri 2026-11-06 23:5x; NewIdeas 7e74d6c: "the rest of the map follows the
# decodes"). It starts once nw3_first.sh is done (data/nw3_first.DONE, its process gone) AND EvalSuite's CERT reference decodes
# (runner 53) have ended: its pid read at arming, and q53.pid's. NewIdeas' S0-S3 quiet A/Bs go ahead: the runners wait while
# another session holds or waits for a quiet window. Then:
#   1. the s67 runner (S67_KEEPLOCK=1) on data/map_rest_extra.txt: nw3's two 36-window arms (chat on record), then step 0's
#      three arms (PLAN_12GB_90.md);
#   2. Fable A's T1 (sat_t1.sh: its own guards, outputs only under ~/FableGod/lossy350a/run_1105), with Saturday times.
# The NW self-check (FAR's nw_selfcheck.py) is left out: nw3's control arm checks the patched path on the GPU, and FAR's whole
# CPU suite passed on the patched files. Not named *sat_night* (EvalSuite's runner 49 gate). Run ATTACHED (Bash
# run_in_background), cwd ~/Breakthrough. Log: data/map_rest.log; the runner's lines go to data/s67_slot.log, T1's to its run.log.
# MR_LASTGO / MR_DEADLINE ("YYYY-mm-dd HH:MM:SS"): no start after / hard stop. Defaults Sat 15:00 / 16:00: S4 runs by 16:00, and
# S4, S5, the FREEZE (~20:00) and CERT outrank the map. What is left then moves to Sunday night with the 24 GB arms.
export BIGMOE_EXT_BUILD=build_nat
H=~/Breakthrough; LOG=$H/data/map_rest.log; X=$H/data/s67_extra.txt; L=$H/data/map_rest_extra.txt
KEEP=$H/data/s67_extra_before_map_rest.txt; ES=$HOME/EvalSuite/results
LG="${MR_LASTGO:-2026-11-07 15:00:00}"; DL="${MR_DEADLINE:-2026-11-07 16:00:00}"
LASTGO=$(date -j -f "%Y-%m-%d %H:%M:%S" "$LG" +%s)
P53=$(cat $ES/q53.pid 2>/dev/null)
say() { echo "[map_rest] $* $(date '+%a %H:%M:%S')" >> $LOG; }
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
alive53() { { [ -n "$P53" ] && kill -0 $P53 2>/dev/null; } || { [ -f $ES/q53.pid ] && kill -0 "$(cat $ES/q53.pid 2>/dev/null)" 2>/dev/null; }; }
first_up() { pgrep -f "bash scripts/dm_s7/nw3_firs[t].sh" > /dev/null; }
say "armed: after data/nw3_first.DONE and runner 53's end (pid ${P53:-none} at arming; q53.pid); then $(grep -v '^#' $L | awk '{print $1}' | tr '\n' ' ')+ T1; last start $(date -r $LASTGO '+%a %H:%M'), hard stop $DL"

# 1. nw3's code arms done, then the CERT reference decodes
w0=
while first_up || [ ! -e $H/data/nw3_first.DONE ] || alive53; do
  [ $(date +%s) -ge $LASTGO ] && { say "still waiting at the last start ($w0): nothing run; the arms move to Sunday night"; exit 1; }
  if first_up || [ ! -e $H/data/nw3_first.DONE ]; then w="nw3's code arms (nw3_first.sh)"; else w="EvalSuite's CERT reference decodes (runner 53)"; fi
  [ "$w" != "$w0" ] && say "waiting: $w"; w0=$w; sleep 15
done
say "nw3's code arms and the CERT reference decodes are done"

# 2. the runner (its own gates: quiet windows, others' shared runs > 5 GB, lid, PAUSE, NO-SWAP)
cp $X $KEEP && SAVED=1
{ grep -v '^#' $L; echo "ALONE 0 00:00"; } > $X
now="$(date '+%Y-%m-%d %H:%M:%S')"
S67_KEEPLOCK=1 S67_START="$now" S67_LASTGO="$LG" S67_DEADLINE="$DL" S67_XALONE="$now" bash $H/scripts/dm_s7/s67_slot.sh & wait $!
say "runner ended (rc $?)"
cp $KEEP $X && SAVED= && say "s67_extra.txt put back"

# 3. Fable A's T1 (code arms, then xs48 on chat if time allows)
if [ $(date +%s) -lt $LASTGO ]; then
  T1_CODE_LASTGO="$LG" T1_CODE_DEADLINE="$DL" T1_CHAT_LASTGO="$LG" T1_CHAT_DEADLINE="$DL" bash $H/scripts/dm_s7/sat_t1.sh & wait $!
  say "T1 ended (rc $?): $(grep 'batch over' ~/FableGod/lossy350a/run_1105/run.log 2>/dev/null | tail -1)"
else say "T1 not started: past the last start"; fi
touch $H/data/map_rest.DONE
exit 0
