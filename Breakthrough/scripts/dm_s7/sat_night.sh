#!/bin/bash
# Saturday night's map batch (Breakthrough, Thu 2026-11-05 19:4x; the orchestrator 19:4x): the night starting Sat 7 Nov
# 00:30, after BigMoE's FAR-G1 and EvalSuite's runner 46 (~2.4 h, MEM 36 GB). No GO handoff that night. Every map process runs
# with BIGMOE_EXT_BUILD=build_nat from cwd ~/Breakthrough. The runners themselves wait while another session holds or waits for
# a quiet window or declares a shared run with MEM > 5 GB, the lid is closed, PAUSE is set, or NO-SWAP is low.
#   1. wait for Sat 00:30 (SAT_AT);
#   2. Fable A's T1 (input sparsity xs64/48/40/32 on code, then xs48 on chat if code leaves room): sat_t1.sh, a copy of its
#      run_t1.sh with Saturday's times; its outputs stay in ~/FableGod/lossy350a/run_1105;
#   3. FAR's never-wait: only if its harness patch is installed (install_nw.sh, run by hand on Friday after the Friday batch):
#      the GPU self-check (gslot.sh nw_selfcheck, ~2 min); exit 0 (PASS or REPORT) keeps tier 1 + 1b in the arm list,
#      anything else drops those lines;
#   4. the s67 runner on data/sat_extra.txt (never-wait first, then step 0 of PLAN_12GB_90.md); then nw_read.py.
# Hard stop 09:45 (the weekend day starts ~10:00). Log: data/sat_night.log.
export BIGMOE_EXT_BUILD=build_nat
H=~/Breakthrough; PY=~/BigMoE/.venv/bin/python; LOG=$H/data/sat_night.log; X=$H/data/s67_extra.txt; DM=~/FableGod/decision_map
FAR=~/FableGod/far/build_1105
say() { echo "[sat] $* $(date '+%a %H:%M:%S')" >> $LOG; }
kids() { local p=$1 c; for c in $(pgrep -P $p); do kids $c; echo $c; done; }
stopall() {   # once: stop my runs (TERM, then KILL if frozen; never SIGCONT), unlock
  [ -n "$DONE_STOP" ] && return; DONE_STOP=1; trap - EXIT INT TERM HUP
  for c in $(kids $$); do kill -TERM $c 2>/dev/null; done; sleep 5
  for c in $(kids $$); do ps -o stat= -p $c 2>/dev/null | grep -q T && kill -KILL $c 2>/dev/null; done
  peer unlock >> $LOG 2>&1; say "ended ($1)"
}
trap 'stopall "exit"' EXIT; trap 'stopall "stopped"; exit 1' INT TERM HUP
at=$(date -j -f "%Y-%m-%d %H:%M" "${SAT_AT:-2026-11-07 00:30}" +%s)
say "armed: from $(date -r $at '+%a %H:%M') behind the quiet queue (FAR-G1, EvalSuite 46 first): T1, never-wait tier 1 + 1b, step 0; hard stop 09:45"
while [ $(date +%s) -lt $at ]; do sleep 30; done

# 2. Fable A's T1 (its own guards; code last start 09:00, chat by 09:45)
T1_CODE_LASTGO="2026-11-07 09:00:00" T1_CODE_DEADLINE="2026-11-07 09:30:00" T1_CHAT_LASTGO="2026-11-07 09:30:00" \
  T1_CHAT_DEADLINE="2026-11-07 09:45:00" bash $H/scripts/dm_s7/sat_t1.sh & wait $!
say "T1 ended (rc $?): $(grep 'batch over' ~/FableGod/lossy350a/run_1105/run.log 2>/dev/null | tail -1)"

# 3. never-wait: the harness patch, then the GPU self-check
NWOK=
if grep -q "NEVER-WAIT" $DM/dm_common.py 2>/dev/null; then
  GS_LASTGO="2026-11-07 09:00:00" GS_DEADLINE="2026-11-07 09:45:00" bash $H/scripts/dm_s7/gslot.sh nw_selfcheck 4 18 \
    env BIGMOE_EXT_BUILD=build_nat $PY $FAR/nw_selfcheck.py & wait $!
  rc=$?; [ $rc = 0 ] && NWOK=1
  say "never-wait self-check: rc $rc, verdict '$(tail -1 $H/data/nw_selfcheck/VERDICT 2>/dev/null)' -> $([ -n "$NWOK" ] && echo 'arms run' || echo 'arms dropped')"
else say "never-wait harness patch not installed: its arms are dropped"; fi

# 4. the runner: never-wait (if NWOK), then step 0
cp $X $H/data/s67_extra_fri.txt 2>/dev/null
{ if [ -n "$NWOK" ]; then cat $H/data/sat_extra.txt; else grep -vE "_nw|_c208|_c266" $H/data/sat_extra.txt; fi; echo "ALONE 0 00:00"; } > $X
say "arm list: $(grep -v '^#' $X | grep -vE '^(WHEN|ALONE)' | awk '{print $1}' | tr '\n' ' ')"
now="$(date '+%Y-%m-%d %H:%M:%S')"
S67_START="$now" S67_LASTGO="2026-11-07 09:15:00" S67_DEADLINE="2026-11-07 09:45:00" S67_XALONE="$now" bash $H/scripts/dm_s7/s67_slot.sh & wait $!
say "runner ended (rc $?)"
[ -n "$NWOK" ] && { $PY -I -B $FAR/nw_read.py >> $LOG 2>&1; say "nw_read: $(grep '^VERDICT' $LOG | tail -3 | tr '\n' ' ')"; }
exit 0
