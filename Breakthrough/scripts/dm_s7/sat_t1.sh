#!/bin/bash
# sat_t1.sh (Breakthrough, Thu 2026-11-05 19:4x; the orchestrator 19:4x): a COPY of Fable A's ~/FableGod/lossy350a/run_1105/run_t1.sh
# for the night starting Sat 7 Nov 00:30. Changed: the code / chat last-go and hard-stop times come from T1_CODE_LASTGO,
# T1_CODE_DEADLINE, T1_CHAT_LASTGO, T1_CHAT_DEADLINE (defaults Sat 09:00 / 09:30 / 09:30 / 09:45). Everything else is Fable
# A's: it still writes only under its run folder (run.log, out/, out_chat/, DONE). Fable C's Thursday yield is in the past.
# run_t1.sh (Fable A, Thu 2026-11-05 evening; the orchestrator 18:5x: "RUN T1 yourself tonight, code first, from a copy in
# your folder"): the map's input-sparsity arms xs64 / xs48 / xs40 / xs32 on the code windows, then xs48 on chat if code
# leaves room and time allows. Pattern and guards = Fable C's fablec/build_1105/run/run_1105.sh of the same evening
# (itself Breakthrough's thu_splits.sh / gslot.sh pattern): the converted store with BIGMOE_EXT_BUILD=build_nat, ONE map
# process at a time, a share lock "MEM 18 GB" per run, guards every 15 s.
# The orchestrator's numbers: start only after Fable C's code arms (its runner gone, or 21:44), no other session's
# shared run over 5 GB, NO-SWAP >= 24 GB; swap growing by more than 1 GB stops the run and ends the batch; last start of the
# code arms 21:45; the chat arm ends by 22:15.
# Also clear = no map process, no other session's quiet window, lid open, no PAUSE, no ~/FableGod/STOP: 3 clear checks 10 s
# apart before every run. During a run: others' quiet window or a closed lid (3 strikes), a non-fanwatch PAUSE over 5 min,
# or a STOP file stop it. Never SIGCONT.
# What is written: only under this folder ($R: run.log, out/, out_chat/, DONE). The harness files here are copies with the
# T1 patch (build_1105); everything else resolves to the map's own files through PYTHONPATH, read only. The shared
# decision_map folder is not touched. The map process's working folder is ~/Breakthrough for one reason: fanwatch freezes
# research jobs by working folder during a heat pause (as in run_1105.sh); nothing is read or written there.
export BIGMOE_EXT_BUILD=build_nat PYTHONDONTWRITEBYTECODE=1 DM_XS=1
R=$HOME/FableGod/lossy350a/run_1105; B=$HOME/FableGod/lossy350a/build_1105; DM=$HOME/FableGod/decision_map
PY=$HOME/BigMoE/.venv/bin/python; LOG=$R/run.log; Q=$HOME/.local/share/quiet
NOSWAP_MIN=24; SWAP_MAX_MB=1024
ts() { date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s; }
FC_YIELD=$(ts "${T1_FC_YIELD:-2026-11-05 21:44:00}")
say() { echo "[t1] $* $(date '+%a %H:%M:%S')" >> $LOG; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
quiet_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
big_others() { peer ls 2>/dev/null | grep -v "^You:" | grep "shared run" | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5 {f = 1} END {exit !f}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
fc_busy() { [ $(date +%s) -lt $FC_YIELD ] && pgrep -f "run_1105.s[h]" > /dev/null; }
kill_tree() { local p=$1 c; for c in $(pgrep -P $p); do kill_tree $c; done; kill -TERM $p 2>/dev/null; }
stop_run() { kill_tree $1; sleep 5; ps -o stat= -p $1 2>/dev/null | grep -q T && kill -KILL $1 2>/dev/null; }   # a frozen run holds SIGTERM until thawed; never SIGCONT
why_not() {
  if [ -e "$HOME/FableGod/STOP" ]; then echo "STOP file"
  elif fc_busy; then echo "Fable C's batch is still armed (I yield until $(date -r $FC_YIELD +%H:%M))"
  elif pgrep -f "run_ma[p].py" > /dev/null; then echo "a map process is running"
  elif quiet_others; then echo "a quiet window (others)"; elif big_others; then echo "another session's shared run > 5 GB"
  elif lid_closed; then echo "lid closed"
  elif [ -e "$Q/PAUSE" ]; then echo "PAUSE"; elif [ "$(noswap)" -lt $NOSWAP_MIN ]; then echo "NO-SWAP < $NOSWAP_MIN GB"; fi
}
if [ "$1" = "dry" ]; then
  echo "dry: why_not='$(why_not)'; NO-SWAP $(noswap) GB; swap $(swap_mb) MB; fc_busy $(fc_busy && echo yes || echo no); quiet_others $(quiet_others && echo yes || echo no); big_others $(big_others && echo yes || echo no); lid_closed $(lid_closed && echo yes || echo no); yield until $(date -r $FC_YIELD '+%a %H:%M')"
  exit 0
fi
ENDBATCH=0; P=
guarded() {   # guarded <name> <est min> <last go, epoch> <hard stop, epoch> <command ...>: 0 = ran to its end with rc 0; 1 = not
  local NAME=$1 EST=$2 LASTGO=$3 DEADLINE=$4; shift 4; local need=3 last= w bad pz stopped rc S0
  while :; do
    [ $(date +%s) -ge $LASTGO ] && { say "$NAME: the last start ($(date -r $LASTGO +%H:%M)) passed; not run"; return 1; }
    [ $(( $(date +%s) + EST * 60 )) -le $DEADLINE ] || { say "$NAME: no longer fits before the hard stop ($(date -r $DEADLINE +%H:%M)); not run"; return 1; }
    w=$(why_not)
    if [ -n "$w" ]; then need=3; [ "$w" != "$last" ] && say "$NAME waiting: $w (NO-SWAP $(noswap) GB, swap $(swap_mb) MB)"; last=$w; sleep 10; continue; fi
    need=$((need - 1)); [ $need -gt 0 ] && { sleep 10; continue; }
    break
  done
  local lk=0
  until peer lock share $(( 3 * EST + 10 ))m --cpu 200 --gpu 100 "Fable A T1 map arms (flips, no timing; the orchestrator 18:5x): MEM 18 GB: $NAME" >> $LOG 2>&1; do
    lk=$((lk + 1)); [ $lk -ge 6 ] || [ $(date +%s) -ge $LASTGO ] && { say "$NAME: share lock not granted; not run"; return 1; }
    say "$NAME: share lock not granted yet (try $lk)"; sleep 30
  done
  S0=$(swap_mb); say "go $NAME (est $EST min; NO-SWAP $(noswap) GB, swap $S0 MB)"
  (cd $HOME/Breakthrough && exec "$@") >> $LOG 2>&1 & P=$!
  bad=0; pz=0; stopped=
  while kill -0 $P 2>/dev/null; do
    sleep 15; w=
    [ $(date +%s) -ge $DEADLINE ] && { stopped="the hard stop ($(date -r $DEADLINE +%H:%M))"; ENDBATCH=1; stop_run $P; break; }
    [ $(( $(swap_mb) - S0 )) -gt $SWAP_MAX_MB ] && { stopped="swap grew > 1 GB ($S0 -> $(swap_mb) MB)"; ENDBATCH=1; stop_run $P; break; }
    [ -e "$HOME/FableGod/STOP" ] && { stopped="STOP file"; ENDBATCH=1; stop_run $P; break; }
    if [ -e "$Q/PAUSE" ] && [ ! -e "$Q/PAUSE.fanwatch" ]; then pz=$((pz + 1)); else pz=0; fi   # fanwatch's heat pause freezes and thaws the run: not a reason to stop it
    if quiet_others; then w="a quiet window (others)"; elif lid_closed; then w="lid closed"; elif [ $pz -ge 20 ]; then w="PAUSE over 5 min"; fi
    if [ -n "$w" ]; then bad=$((bad + 1)); else bad=0; fi
    [ $bad -ge 3 ] && { stopped="$w"; ENDBATCH=1; stop_run $P; break; }
  done
  wait $P 2>/dev/null; rc=$?; P=; peer unlock >> $LOG 2>&1
  if [ -n "$stopped" ]; then say "$NAME STOPPED for: $stopped; unlocked (swap now $(swap_mb) MB)"; return 1; fi
  say "$NAME ended (rc $rc; swap $S0 -> $(swap_mb) MB); unlocked"; [ $rc = 0 ]
}
finish() { say "batch over: $1"; date '+%a %H:%M:%S' > $R/DONE; exit 0; }
trap 'peer unlock >> $LOG 2>&1' EXIT
trap 'say "interrupted"; [ -n "$P" ] && stop_run $P; date > $R/DONE; exit 1' INT TERM HUP
rm -f $R/DONE
say "armed: xs64+code xs48+code xs40+code xs32+code, then xs48 on chat if code leaves room; NO-SWAP >= $NOSWAP_MIN GB to start, swap +$SWAP_MAX_MB MB ends the batch; code last start Thu 21:45, hard stop 22:00; chat ends by 22:15; BIGMOE_EXT_BUILD=$BIGMOE_EXT_BUILD"

# 0. the wiring, on the CPU: the map process will import this folder's patched copies and the map's own files for the rest
W=$(cd $R && DM_SETS=code DM_OUT=$R/out PYTHONPATH=$DM $PY $R/wiring_check.py 2>&1); echo "$W" >> $LOG
case "$W" in *"WIRING OK"*) ;; *) finish "the wiring check failed: nothing runs";; esac

# 1. the four code arms, one process (one model load); xs64 first: the identity arm is the control
guarded code 11 $(ts "${T1_CODE_LASTGO:-2026-11-07 09:00:00}") $(ts "${T1_CODE_DEADLINE:-2026-11-07 09:30:00}") env DM_SETS=code DM_OUT=$R/out PYTHONPATH=$DM $PY $R/run_map.py xs64+code xs48+code xs40+code xs32+code --max 12 --skip-done
RC=$?
say "code arms: guarded rc $RC; files: $(ls $R/out | grep -c 'xs.*code.json') of 4"
D=$($PY -I $R/decide.py $R/out $DM/out/baseline.json 2>&1); echo "$D" >> $LOG
$PY -I $B/read_xs.py --out $R/out --base $DM/out/baseline.json xs64+code xs48+code xs40+code xs32+code >> $LOG 2>&1
[ $ENDBATCH = 1 ] && finish "ended after the code arms (a guard stopped the batch)"
case "$D" in *"CHAT GO"*) ;; *) finish "no chat arm: $(echo "$D" | tail -1)";; esac

# 2. xs48 on the chat windows (IFEval + multi): the arm that says which quality reading holds; must end by 22:15
guarded chat 7 $(ts "${T1_CHAT_LASTGO:-2026-11-07 09:30:00}") $(ts "${T1_CHAT_DEADLINE:-2026-11-07 09:45:00}") env DM_SETS=ifeval,multi DM_OUT=$R/out_chat PYTHONPATH=$DM $PY $R/run_map.py xs48 --max 12 --skip-done
say "chat arm: guarded rc $?; file: $(ls $R/out_chat | grep -c '^xs48.json')"
$PY -I $B/read_xs.py --out $R/out_chat --base $DM/out/baseline.json xs48 >> $LOG 2>&1
finish "code and chat done"
