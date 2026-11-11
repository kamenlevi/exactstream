#!/bin/bash
# run_1105.sh (Fable C, Thu 2026-11-05 evening; the orchestrator 18:3x: "RUN your code-window arms yourself tonight"): the map's
# code-window arms K8, K6, A, A', B, then K4, from COPIES in this folder. Pattern = Breakthrough's thu_splits.sh / gslot.sh /
# s67_slot.sh (read, not run): the converted store with BIGMOE_EXT_BUILD=build_nat, the scores-capture smoke first, ONE map
# process at a time, a share lock "MEM 18 GB" per run, guards every 15 s.
# The evening's guards (the orchestrator's numbers, tighter than gslot's): clear needs NO-SWAP >= 24 GB; swap growing by more
# than 1 GB stops the run AND ends the batch; no run starts after 21:15; hard stop 21:30; then unlock.
# Also clear = no map process, no other session's quiet window held or waited for, no other session's shared run over 5 GB,
# lid open, no PAUSE: 3 clear checks 10 s apart before every run. During a run: others' quiet window or a closed lid (3
# strikes) or a non-fanwatch PAUSE over 5 min stop it (it is retried once from its start if it still fits). Never SIGCONT.
# What is written: only under this folder ($R: run.log, out/, control/, scap_smoke/). The map's files are read only; its masks
# for these arms are the snapshot in decision_map/out/fc (installed before this script), registered in THIS folder's copy of
# dm_ablations.py; dm_common and the rest resolve to the map's own files through PYTHONPATH. The map process's working folder
# is ~/Breakthrough for one reason: fanwatch freezes research jobs by working folder during a heat pause and ~/FableGod is not
# one (s67_slot.sh, Wed 18:1x); nothing is read or written there.
# After the smoke, a CONTROL: one code window of the twin qom_0_live_ball through this folder's copies must give the top-8
# tokens of the twin on record (decision_map/out/qom_0_live_ball+code.json) row for row; if not, no arm runs.
export BIGMOE_EXT_BUILD=build_nat PYTHONDONTWRITEBYTECODE=1
B=$HOME/FableGod/fablec/build_1105; R=$B/run; DM=$HOME/FableGod/decision_map; PY=$HOME/BigMoE/.venv/bin/python; LOG=$R/run.log; Q=$HOME/.local/share/quiet
ARMS="qom_0_split0.99_k8_live_ball qom_0_split0.99_k6_live_ball qom_0_split0.99_r0.7_live_ball qom_0_split0.99_m0.9_live_ball qom_0_split0.99_r0.7_z0.999_live_ball qom_0_split0.99_k4_live_ball"
TWINS="qom_0_live_ball+code qom_0_split0.99_live_ball+code qom_0_split0.9_live_ball+code"
NOSWAP_MIN=24; SWAP_MAX_MB=1024
ts() { date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s; }
LASTGO=$(ts "${RUN_LASTGO:-2026-11-05 21:15:00}"); DEADLINE=$(ts "${RUN_DEADLINE:-2026-11-05 21:30:00}")
say() { echo "[fc] $* $(date '+%a %H:%M:%S')" >> $LOG; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
quiet_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
big_others() { peer ls 2>/dev/null | grep -v "^You:" | grep "shared run" | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5 {f = 1} END {exit !f}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
kill_tree() { local p=$1 c; for c in $(pgrep -P $p); do kill_tree $c; done; kill -TERM $p 2>/dev/null; }
stop_run() { kill_tree $1; sleep 5; ps -o stat= -p $1 2>/dev/null | grep -q T && kill -KILL $1 2>/dev/null; }   # a frozen run holds SIGTERM until thawed; never SIGCONT
why_not() {
  if pgrep -f "run_ma[p].py" > /dev/null; then echo "a map process is running"
  elif quiet_others; then echo "a quiet window (others)"; elif big_others; then echo "another session's shared run > 5 GB"
  elif lid_closed; then echo "lid closed"
  elif [ -e "$Q/PAUSE" ]; then echo "PAUSE"; elif [ "$(noswap)" -lt $NOSWAP_MIN ]; then echo "NO-SWAP < $NOSWAP_MIN GB"; fi
}
nwin() { [ -f "$1" ] && $PY -c "import json, sys; d = json.load(open(sys.argv[1])); print(sum(1 for v in d.values() if isinstance(v, dict) and 'tk' in v))" "$1" 2>/dev/null || echo 0; }
ENDBATCH=0; P=
guarded() {   # guarded <name> <est min> <command ...>: 0 = the command ran to its end with rc 0; 1 = not (could not start, stopped, or rc != 0)
  local NAME=$1 EST=$2; shift 2; local need=3 last= w bad pz stopped rc S0
  while :; do
    [ $(date +%s) -ge $LASTGO ] && { say "$NAME: the last start ($(date -r $LASTGO +%H:%M)) passed; not run"; return 1; }
    [ $(( $(date +%s) + EST * 60 )) -le $DEADLINE ] || { say "$NAME: no longer fits before the hard stop ($(date -r $DEADLINE +%H:%M)); not run"; return 1; }
    w=$(why_not)
    if [ -n "$w" ]; then need=3; [ "$w" != "$last" ] && say "$NAME waiting: $w (NO-SWAP $(noswap) GB, swap $(swap_mb) MB)"; last=$w; sleep 10; continue; fi
    need=$((need - 1)); [ $need -gt 0 ] && { sleep 10; continue; }
    break
  done
  local lk=0
  until peer lock share $(( 3 * EST + 10 ))m --cpu 200 --gpu 100 "Fable C map arms on the code windows (flips, no timing; the orchestrator 18:3x): MEM 18 GB: $NAME" >> $LOG 2>&1; do
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
    if [ -e "$Q/PAUSE" ] && [ ! -e "$Q/PAUSE.fanwatch" ]; then pz=$((pz + 1)); else pz=0; fi   # fanwatch's heat pause freezes and thaws the run: not a reason to stop it
    if quiet_others; then w="a quiet window (others)"; elif lid_closed; then w="lid closed"; elif [ $pz -ge 20 ]; then w="PAUSE over 5 min"; fi
    if [ -n "$w" ]; then bad=$((bad + 1)); else bad=0; fi
    [ $bad -ge 3 ] && { stopped="$w"; stop_run $P; break; }
  done
  wait $P 2>/dev/null; rc=$?; P=; peer unlock >> $LOG 2>&1
  if [ -n "$stopped" ]; then say "$NAME STOPPED for: $stopped; unlocked (swap now $(swap_mb) MB)"; return 1; fi
  say "$NAME ended (rc $rc; swap $S0 -> $(swap_mb) MB); unlocked"; [ $rc = 0 ]
}
# THE GO GATE (Thu 18:4x): the Fable sessions' standing rule is no GPU, and it is Kamen's to lift. This script does nothing
# unless $R/KAMEN_YES exists; that file is written only after Kamen's own yes has arrived (directly, or relayed by the
> (a quote from the project owner, kept private)
> (a quote from the project owner, kept private)
trap 'peer unlock >> $LOG 2>&1' EXIT
trap 'say "interrupted"; [ -n "$P" ] && stop_run $P; exit 1' INT TERM HUP
say "armed: smoke, control, then $ARMS (+code); NO-SWAP >= $NOSWAP_MIN GB to start, swap +$SWAP_MAX_MB MB ends the batch; last start $(date -r $LASTGO '+%a %H:%M'), hard stop $(date -r $DEADLINE '+%a %H:%M'); BIGMOE_EXT_BUILD=$BIGMOE_EXT_BUILD"

# 0. the scores-capture smoke (Breakthrough's scap_pass.py, a copy; DM_OUT = this folder): the store and the wiring
guarded smoke 3 env DM_OUT=$R/scap_smoke $PY $R/scap_pass.py smoke || { say "the smoke did not pass: no arm runs"; exit 1; }
say "smoke: $(grep '^\[scap\] checks' $LOG | tail -1)"
[ $ENDBATCH = 1 ] && { say "batch ended after the smoke"; exit 1; }

# 1. the control: one code window of the twin through this folder's copies == the twin on record
rm -f "$R/control/qom_0_live_ball+code.json"
guarded control 2 env DM_SETS=code DM_OUT=$R/control PYTHONPATH=$DM $PY $R/run_map.py qom_0_live_ball+code --max 1
C=$($PY - "$R/control/qom_0_live_ball+code.json" "$DM/out/qom_0_live_ball+code.json" <<'EOF' 2>&1
import json, sys
try: a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
except Exception as ex: print(f"CONTROL FAIL: {ex}"); sys.exit(0)
w = [k for k, v in a.items() if isinstance(v, dict) and "tk" in v]
if len(w) != 1 or w[0] not in b: print(f"CONTROL FAIL: windows {w}"); sys.exit(0)
ta, tb = a[w[0]]["tk"], b[w[0]]["tk"]; la, lb = a[w[0]]["lp"], b[w[0]]["lp"]
same8 = sum(x == y for x, y in zip(ta, tb)); same1 = sum(x[0] == y[0] for x, y in zip(ta, tb))
dl = max(abs(p - q) for x, y in zip(la, lb) for p, q in zip(x, y)) if len(ta) == len(tb) else float("nan")
ok = len(ta) == len(tb) and same1 == len(ta)
print(f"CONTROL {'PASS' if ok else 'FAIL'}: {w[0]}: {len(ta)} rows vs {len(tb)} on record; top-1 equal on {same1}, all top-8 equal on {same8}; max |log-prob diff| {dl:.2e}")
EOF
)
say "$C"
case "$C" in "CONTROL PASS"*) ;; *) say "the control did not reproduce the twin on record: no arm runs"; exit 1;; esac
[ $ENDBATCH = 1 ] && { say "batch ended after the control"; exit 1; }

# 2. the arms, in the pre-registered order; a soft stop is retried once
for a in $ARMS; do
  f="$R/out/$a+code.json"
  for try in 1 2; do
    [ "$(nwin "$f")" = 12 ] && break
    [ $ENDBATCH = 1 ] && break
    [ $try = 2 ] && say "$a: retry"
    guarded "$a" 4 env DM_SETS=code DM_OUT=$R/out PYTHONPATH=$DM $PY $R/run_map.py "$a+code" --max 12 --skip-done
    [ $(date +%s) -ge $LASTGO ] && break
  done
  if [ "$(nwin "$f")" = 12 ]; then
    ln -sf "$f" "$R/read/$a+code.json"
    o=$(READ_OUT=$R/read $PY $R/code_read_run.py "$a+code" $TWINS 2>&1); echo "$o" >> $LOG
    say "read-out: $(echo "$o" | grep '^VERDICT2' || echo "$a none")"
  else say "$a: NOT done ($(nwin "$f") of 12 windows)"; fi
  [ $ENDBATCH = 1 ] && { say "the batch ends here"; break; }
done
say "batch over"
