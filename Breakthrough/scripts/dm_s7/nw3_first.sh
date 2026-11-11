#!/bin/bash
# nw3 FIRST (Breakthrough, Fri 2026-11-06 23:4x; NewIdeas 7e74d6c, after Kamen opened the window at 23:30): NEVER-WAIT v3 (b)'s
# three code arms (data/nw3_code_extra.txt; pre-registered in SCOREBOARD 'Fri 21:3x: PRE-REGISTRATION, NW3'), the first map
# work after BigMoE's ticket, Leap, Fit12 and EvalSuite's runner 49 start; EvalSuite's CERT reference decodes (runner 53) and
# the rest of the map (map_rest.sh) follow.
# RE-KEYED Sat 2026-11-07 01:3x (NewIdeas: "take the NEXT runner-49 chunk boundary instead of runner 49's end"). Runner 49 runs
# short stack processes (1-2 min each) under its own share lock. Before each one its gate waits while another session's shared
# run declares MEM > 5 GB (checked again every 60 s), then for NO-SWAP memory, then takes its share lock. The gate is passed
# BEFORE the lock, and a lock refused for someone's quiet window is retried every 60 s with no new check: at 01:29 its process 7
# started 4 s after this driver's last arm (the end of BigMoE's quiet ticket; 46 + 18 GB do not fit). The first version (wait
# for runner 49's end) saw the end of its first instance at 00:54 and ran the two 208-slot arms by 01:01; the published-pool arm
# was stopped by quiet windows at 01:03 and 01:30. So now the claim goes in BEFORE a boundary and an arm starts only once runner
# 49 has logged that it waits behind the claim:
#  1. wait until no other session holds or queues a quiet window and none but EvalSuite runs a shared job > 5 GB or >= 50 % GPU
#     (the claim may go in beside runner 49's process), no PAUSE / HOLD, lid open;
#  2. claim: a share lock whose reason starts "MEM 18 GB" (20 min, renewed every 10 min while it waits); released if a quiet
#     window turns up first;
#  3. go once runner 49's log, after the claim, ends on "another session's MEM job holds a shared lock: waiting" and EvalSuite
#     holds no lock (runner 49 then checks every 60 s; the lock stays held to the last arm, S67_KEEPLOCK); or once it already
#     waited behind a MEM lock when the claim went in and its log has not moved for 75 s of the claim (it logs that wait once
#     per 10 min); or once runner 49 is over (its pid read at arming, and q49.pid's). Always with NO-SWAP >= 25 GB;
#  4. the s67 runner on the arms left (its own clear checks; a quiet window, a closed lid or swap +2 GB stops an arm). With
#     S67_STOP_EXIT it ends after a stopped arm or instead of waiting; this driver then unlocks and claims the next boundary
#     again (at most 6 rounds);
#  5. the read-out (scripts/dm_s7/nw3_read.sh: data/nw3_code_read.txt; code_read.py lines also in data/s67_slot.log), then
#     data/nw3_first.DONE (map_rest.sh waits for it).
# Not named *sat_night* (EvalSuite's runner 49 waits while `pgrep -f 'sat_night\.sh'` matches). Run ATTACHED (Bash
# run_in_background), cwd ~/Breakthrough. Log: data/nw3_first.log. NW3_LASTGO / NW3_DEADLINE ("YYYY-mm-dd HH:MM:SS"): no
# start after / hard stop (default Sat 14:00 / 15:00). NW3_DRY=1: print the gate's view and exit (no lock, nothing run).
export BIGMOE_EXT_BUILD=build_nat
H=~/Breakthrough; PY=~/BigMoE/.venv/bin/python; LOG=$H/data/nw3_first.log; X=$H/data/s67_extra.txt; L=$H/data/nw3_code_extra.txt
DM=~/FableGod/decision_map; KEEP=$H/data/s67_extra_before_nw3.txt; Q=$HOME/.local/share/quiet; ES=$HOME/EvalSuite/results
R49=$ES/day49_runner.log
LG="${NW3_LASTGO:-2026-11-07 14:00:00}"; DL="${NW3_DEADLINE:-2026-11-07 15:00:00}"
LASTGO=$(date -j -f "%Y-%m-%d %H:%M:%S" "$LG" +%s)
P49=$(cat $ES/q49.pid 2>/dev/null)
say() { echo "[nw3_first] $* $(date '+%a %H:%M:%S')" >> $LOG; }
kids() { local p=$1 c; for c in $(pgrep -P $p); do kids $c; echo $c; done; }
SAVED=
stopall() {   # once: stop my runs (TERM, then KILL if frozen; never SIGCONT), put s67_extra.txt back, unlock
  [ -n "$DONE_STOP" ] && return; DONE_STOP=1; trap - EXIT INT TERM HUP
  for c in $(kids $$); do kill -TERM $c 2>/dev/null; done; sleep 5
  for c in $(kids $$); do ps -o stat= -p $c 2>/dev/null | grep -q T && kill -KILL $c 2>/dev/null; done
  [ -n "$SAVED" ] && cp $KEEP $X && say "s67_extra.txt put back"
  peer unlock >> $LOG 2>&1; say "ended ($1)"
}
grep -q "NEVER-WAIT v3 (b)" $DM/dm_common.py && grep -q "nw3_mask" $DM/dm_ablations.py || { say "nw3 is not installed in the map: nothing run"; exit 1; }
alive49() { { [ -n "$P49" ] && kill -0 $P49 2>/dev/null; } || { [ -f $ES/q49.pid ] && kill -0 "$(cat $ES/q49.pid 2>/dev/null)" 2>/dev/null; }; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
others() { peer ls 2>/dev/null | grep -v "^You:"; }
why_not() {   # empty = clear. EvalSuite's own shared run (runner 49's process) does not count: the claim may go in beside it
  local pl; pl=$(others)
  if echo "$pl" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; then echo "a quiet window (others)"
  elif echo "$pl" | grep -v "^- Eval suite" | grep "shared run" | grep -oE "MEM ~?[0-9]+" | tr -d "~" | awk '$2 > 5 {f = 1} END {exit !f}'; then echo "another session's shared run > 5 GB"
  elif echo "$pl" | grep -v "^- Eval suite" | sed -n 's/.*shared run[^:]*: about [0-9]*% CPU, \([0-9]*\)% GPU.*/\1/p' | awk '$1 >= 50 {f = 1} END {exit !f}'; then echo "another session's shared run >= 50 % GPU"
  elif [ -e $Q/PAUSE ] || [ -e $Q/HOLD ]; then echo "PAUSE / HOLD"
  elif lid_closed; then echo "lid closed"; fi
}
ev_lock() { others | grep "^- Eval suite" | grep -q "shared run"; }   # EvalSuite holds a share lock (a runner-49 process runs)
r49_since() {   # runner 49's last lock event after line $1 of its log: mem (it waits behind a MEM lock), refused, grant, or empty
  tail -n +$(( $1 + 1 )) $R49 2>/dev/null | grep -E "^[0-9:]+ shared lock: |holds a shared lock: waiting|share lock refused" | tail -1 \
    | awk '/holds a shared lock: waiting/ {print "mem"; next} /share lock refused/ {print "refused"; next} {print "grant"}'; }
r49_last() { r49_since 0; }   # over the whole log
r49_lines() { local n; n=$(wc -l < $R49 2>/dev/null | tr -d ' '); echo ${n:-0}; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
nwin() { [ -f $DM/out/$1.json ] && $PY -c "import json; d = json.load(open('$DM/out/$1.json')); print(sum(1 for v in d.values() if isinstance(v, dict) and 'tk' in v))" 2>/dev/null || echo 0; }
left() { local a; for a in $(grep -v '^#' $L | awk 'NF >= 3 {print $1}'); do [ "$(nwin $a)" = 12 ] || echo -n "$a "; done; }
if [ -n "$NW3_DRY" ]; then
  n=$(r49_lines); echo "why_not: '$(why_not)'; EvalSuite lock: $(ev_lock && echo yes || echo no); runner 49 ($P49 / q49.pid): $(alive49 && echo alive || echo over)"
  echo "runner 49's last lock event in its last 40 lines: '$(r49_since $((n > 40 ? n - 40 : 0)))' (log lines $n); arms left: $(left)"; exit 0
fi
trap 'stopall "exit"' EXIT; trap 'stopall "stopped"; exit 1' INT TERM HUP

handoff() {   # steps 1-3: returns at the go (the claim held), exits at the last start
  local now w w0= claimed= CT=0 CL=0 C0=0 n e
  while :; do
    now=$(date +%s)
    [ $now -ge $LASTGO ] && { say "no go by the last start ($LG; ${w0:-claimed, runner 49 never waited behind it}): nothing run"; exit 1; }
    w=$(why_not)
    if [ -n "$w" ]; then
      [ -n "$claimed" ] && { peer unlock >> $LOG 2>&1; claimed=; say "claim released: $w"; }
      [ "$w" != "$w0" ] && say "waiting: $w"; w0=$w; sleep 5; continue
    fi
    w0=
    if [ -z "$claimed" ] || [ $((now - CT)) -ge 600 ]; then
      if peer lock share 20m --cpu 200 --gpu 100 "MEM 18 GB: Breakthrough nw3 code arm(s) at runner 49's next boundary (NewIdeas Sat 01:3x); flips/KL, no timing" >> $LOG 2>&1; then
        [ -z "$claimed" ] && { CL=$(r49_lines); C0=$now; say "claimed (share lock, MEM 18 GB; runner 49's log at line $CL): waiting for runner 49 to wait behind it"; }
        claimed=1; CT=$now
      else say "claim not granted"; claimed=; sleep 30; continue; fi
    fi
    alive49 || { say "runner 49 is over (pid $P49 gone; q49.pid $([ -f $ES/q49.pid ] && echo "holds $(cat $ES/q49.pid)" || echo gone)): go"; return 0; }
    n=$(r49_lines); [ "$n" -lt "$CL" ] && CL=0   # its log was started again
    e=$(r49_since $CL)
    if ! ev_lock && [ "$(noswap)" -ge 25 ]; then
      # (a) after the claim, runner 49's gate logged that it waits behind a MEM lock: it checks again every 60 s
      [ "$e" = mem ] && { say "runner 49 waits behind the claim ($(tail -1 $R49 | cut -c1-150)): go"; return 0; }
      # (b) it already waited behind a MEM lock when the claim went in (it logs that once per 10 min) and has written nothing
      #     for 75 s of the claim: a gate passed before the claim would have logged its grant (or a memory wait) by now
      [ "$n" = "$CL" ] && [ $((now - C0)) -ge 75 ] && [ "$(r49_last)" = mem ] && { say "runner 49 still waits behind a MEM lock, nothing new in its log for $((now - C0)) s of the claim: go"; return 0; }
    fi
    sleep 5
  done
}

say "armed (RE-KEYED Sat 01:3x, NewIdeas: runner 49's next boundary): runner 49 = pid ${P49:-none} (q49.pid); arms left: $(left); last start $(date -r $LASTGO '+%a %H:%M'), hard stop $DL"
round=0
while [ -n "$(left)" ]; do
  round=$((round + 1)); [ $round -gt 6 ] && { say "6 rounds and still left: $(left): stop (re-arm by hand)"; exit 1; }
  handoff
  cp $X $KEEP && SAVED=1
  { grep -v '^#' $L; echo "ALONE 0 00:00"; } > $X
  now="$(date '+%Y-%m-%d %H:%M:%S')"
  S67_STOP_EXIT=1 S67_MAXRUNS=3 S67_KEEPLOCK=1 S67_START="$now" S67_LASTGO="$LG" S67_DEADLINE="$DL" S67_XALONE="$now" bash $H/scripts/dm_s7/s67_slot.sh & wait $!
  say "round $round: runner ended (rc $?); left: $(left)"
  cp $KEEP $X && SAVED= && say "s67_extra.txt put back"
  peer unlock >> $LOG 2>&1   # whatever is left of the claim: the next round claims afresh
  [ $(date +%s) -ge $LASTGO ] && [ -n "$(left)" ] && { say "past the last start; left: $(left)"; exit 1; }
done
peer unlock >> $LOG 2>&1
bash $H/scripts/dm_s7/nw3_read.sh
say "read-out: $(awk '/^== nw3 code read-out/ {s = ""} {s = s "\n" $0} END {print s}' $H/data/nw3_code_read.txt | grep -E "^(-- |VERDICT)" | cut -c1-170 | tr '\n' ' ')"
touch $H/data/nw3_first.DONE
exit 0
