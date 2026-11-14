#!/bin/bash
# cc3_runner.sh (Leap, Sat 14 Nov 2026 19:0x): WEEK98 route 3 screen 3 (the orchestrator 18:2x: "yes to screen 3, the real
# 32-row check at 45 GB split into dense / experts / misses ... queue a quiet lock around 20:30 (peer lock quiet --queue
# 30m) and let it wait its turn"). Runs on the v121 clean clone (engine unchanged; the probe rides the loop's own
# R57_COPYPROBE hook, scripts/week98/cc3_probe.py), 4 HumanEval keys of the shipped REF2 rows, cap 256:
#   b24  lossy-24 (C 100, at most 13 rows; QOM)  gate 28 = 23.8 + 4   the probe's own trigger every 4th check (CC3_EVERY=4)
#   a45  exact-45 (C 256)                        gate 47 = 42.5 + 4   the loop's warm split every 4th check (R57_WARMSPLIT=4)
# A failed run (rc != 0, not a skip) retries with fewer probe passes (no s32; then no d32).
# v5 (21:1x, after v4's ticket at 20:58 got nothing: b24 died at start-up on w24p2:159, R57_ROWCAP vs R57_WARMSPLIT; a45 was
# skipped at 34 GB):
#   - per-run flags: b24 has no warm split (the 24 GB loop forbids it with R57_ROWCAP); the probe triggers itself.
#   - at arming, every run's exact run.py arguments go through flagcheck.py: the REAL loop's own checks, up to its model
#     load (the orchestrator 21:0x: "head-test runners against the real loop's flag checks, not fakes"). Any FAIL: no ticket.
#   - no idle tickets: ticket 1 runs b24, and a45 only if one read already shows its gate; otherwise a45 waits, holding
#     nothing, until NO-SWAP >= 47 on the lower of two reads 30 s apart, and only then queues its own ticket (at most two).
#   - every run ends by the cutoff (the orchestrator 21:0x: a45 by 01:00, when NewIdeas and MTP Head start; else re-arm
#     for tomorrow night).
# Gates: NO-SWAP = free + purgeable + file-backed (the 11-14 05:4x formula), the lower of two reads 30 s apart, parts logged
# (18:2x); the power mode logged at each grant and run (19:1x). Swap +2 GB stops; HOLD / PAUSE / lid / HEALTH ACT checked
# before each run; no other engine may run. Never lowers a gate.
# DRY=1: the head test of this script: fake lock, 1 s waits, gates 1 GB, and flagcheck.py in place of run.py (so each run's
# exact arguments still meet the real loop's checks). Usage: cc3_runner.sh "<FROM YYYY-MM-DD HH:MM:SS>" "<CUTOFF ...>"
V=/Users/kamen/Leap/release/cleanclone/v121; CL=$V/clone; H2=$V/home2; HFH=$HOME/.cache/huggingface
D=${CC3_DIR:-/Users/kamen/Leap/data/week98/cc3}; W=/Users/kamen/Leap/scripts/week98; PROBE=$W/cc3_probe.py
Q=$HOME/.local/share/quiet; LOG=$D/cc3.log; PY=/opt/homebrew/opt/python@3.12/bin/python3.12
KEYS=humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6
FROM=$(date -j -f "%Y-%m-%d %H:%M:%S" "${1:?FROM}" +%s); CUT=$(date -j -f "%Y-%m-%d %H:%M:%S" "${2:?CUTOFF}" +%s)
DRY=${DRY:-0}; G24=28; G45=47; ZZ=30; ST=20; PW=60
[ $DRY = 1 ] && { G24=1; G45=1; ZZ=1; ST=1; PW=1; }
F24="CC3_EVERY=4 CC3_REFLOOSE=1"; F45="R57_WARMSPLIT=4 CC3_REFLOOSE=1"
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|run_map[.]py|fit_onecall[.]py|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
pmode() { pmset -g | awk '/^ *powermode/ {print $2; exit}'; }
parts() { vm_stat | awk '/page size of/ {ps=$8} {v=$NF; sub("[.]","",v)} /^Pages free:/ {f=v} /^Pages purgeable:/ {p=v}
  /^File-backed pages:/ {fb=v} /^Pages speculative:/ {sp=v} /^Anonymous pages:/ {an=v} /^Pages occupied by compressor:/ {c=v}
  END {g=ps/1e9; printf "free %.1f + purgeable %.1f + file-backed %.1f (speculative %.1f inside); anon %.1f, compressor %.1f", f*g, p*g, fb*g, sp*g, an*g, c*g}'; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
gate_ok() {   # the LOWER of two reads $ZZ s apart must reach the gate; both logged
  local a b pa pb
  a=$(noswap); pa=$(parts); sleep $ZZ; b=$(noswap); pb=$(parts)
  say "  gate $1: read 1 $a GB ($pa); read 2 (+$ZZ s) $b GB ($pb)"
  [ $a -ge $1 ] && [ $b -ge $1 ]
}
lock_q() { if [ $DRY = 1 ]; then echo "(dry) fake ticket" >> $LOG; sleep 1; return 0; fi
           PEER_QUEUE_TIMEOUT=$1 peer lock quiet --queue 30m "$2" >> $LOG 2>&1; }
unlock() { [ $DRY = 1 ] || peer unlock >/dev/null 2>&1; }
killtree() { local c; for c in $(pgrep -P $1 2>/dev/null); do killtree $c; done; kill -TERM $1 2>/dev/null; }
LOCKED=0; CP=""; GUARD=""; RC=""; GR=0; A45=todo
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null
            [ -n "$CP" ] && kill -0 $CP 2>/dev/null && killtree $CP
            [ $LOCKED = 1 ] && { unlock; say "ticket released"; }
            say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT; trap 'say "stopped by a signal"; exit 1' INT TERM
for f in $PROBE $W/flagcheck.py $CL/run.py $CL/harness/results/ref2/decode.jsonl; do [ -e $f ] || { echo "no $f"; exit 2; }; done
mkdir -p $D $H2
args() {   # args <tag> <config> <mem> <passes> <flags>: run.py's arguments for one run
  local e a_="--config $2 --mem $3 --keys $KEYS --max-tokens 256 --env R57_COPYPROBE=1 --env R57_COPYPROBE_PY=$PROBE"
  a_="$a_ --env CC3_OUT=$D/${1}_probe.jsonl --env CC3_PASSES=$4"
  for e in $5; do a_="$a_ --env $e"; done
  echo "$a_ --out $D/$1"
}
say "armed (pid $$, v5$([ $DRY = 1 ] && echo ", DRY")): cc3 screen 3, b24 then a45 with the probe; from $1, every run ends by $2"
# the head test at arming: each run's exact arguments against the real loop's checks (flagcheck.py), before any ticket
for spec_ in "b24 lossy 24 d1,d16,d32,s32 $F24" "a45 exact 45 d1,d16,d32,s32 $F45"; do
  set -- $spec_; t_=$1; c_=$2; m_=$3; p_=$4; shift 4
  fc_=$( cd /Users/kamen/Leap && env HOME=$H2 HF_HOME=$HFH $CL/.venv/bin/python $W/flagcheck.py $CL $(args fc_$t_ $c_ $m_ $p_ "$*") 2>&1 )
  rm -rf $D/fc_$t_ $D/fc_${t_}_probe.jsonl
  echo "$fc_" | grep -q "^PASS" || { say "STOP: the flag check of $t_ failed: $(echo "$fc_" | grep -E "Error|FAIL" | head -2 | tr '\n' ' ')"; exit 2; }
  say "  flag check $t_ ($c_-$m_, $*): PASS (the real loop reached its model load)"
done
guard_on() { local s0=$(swap); ( while :; do [ $(( $(swap) - s0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) & GUARD=$!; }
guard_off() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null; GUARD=""; }
ticket() {   # ticket <est s> <reason>: queue, wait for the grant, settle; 1 if the ticket left the queue without a grant
  local to=$(( CUT - $(date +%s) - $1 - 120 ))
  [ $to -lt 120 ] && { say "SKIP the ticket: under $(( ($1 + 240) / 60 )) min to the cutoff"; return 1; }
  say "queue the quiet ticket (PEER_QUEUE_TIMEOUT $to s; NO-SWAP $(noswap) GB)"
  lock_q $to "$2" || { say "the ticket left the queue without a grant"; return 1; }
  LOCKED=1; GR=$(date +%s); say "granted (NO-SWAP $(noswap) GB, power mode $(pmode)); settle $ST s"; sleep $ST; guard_on
}
unticket() { guard_off; unlock; LOCKED=0; say "ticket released"; }
run() {   # run <tag> <gate GB> <est s> <config> <mem> <passes> <flags>
  local t=$1 gate=$2 est=$3 cfg=$4 mem=$5 pas=$6 fl=$7
  [ -e $Q/HOLD ] && { say "STOP: HOLD"; exit 0; }
  local w=0; while [ -e $Q/PAUSE ] && [ $w -lt 120 ]; do sleep 5; w=$((w + 5)); done
  [ -e $Q/PAUSE ] && { say "SKIP $t: PAUSE"; RC=skip; return 1; }
  lid_closed && { say "STOP: lid closed"; exit 0; }
  grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null && { say "STOP: HEALTH ACT"; exit 0; }
  w=0; while [ -n "$(procs)" ] && [ $w -lt 60 ]; do sleep 5; w=$((w + 5)); done
  [ -n "$(procs)" ] && { say "SKIP $t: an engine runs ($(procs))"; RC=skip; return 1; }
  local lim=$(( $(date +%s) + 60 )); [ $(( GR + 240 )) -gt $lim ] && lim=$(( GR + 240 ))
  local ok=0
  while :; do
    if [ "$(noswap)" -ge $gate ] && gate_ok $gate; then ok=1; break; fi
    [ $(date +%s) -ge $lim ] && break
    sleep 5
  done
  [ $ok = 1 ] || { say "SKIP $t: NO-SWAP below $gate (its peak + 4) on the lower of two reads $ZZ s apart; now $(noswap) GB ($(parts))"; RC=skip; return 1; }
  [ $(( $(date +%s) + est )) -gt $CUT ] && { say "SKIP $t: it could end after the cutoff (est $est s)"; RC=skip; return 1; }
  say "run $t: $cfg-$mem, probe passes $pas, $fl (NO-SWAP $(noswap) GB, swap $(swap) MB, power mode $(pmode))"
  echo "power mode $(pmode) at $(date '+%m-%d %H:%M:%S')" > $D/${t}_pmode.txt
  rm -f $D/${t}_probe.jsonl
  local ex=$CL/run.py; [ $DRY = 1 ] && ex="$W/flagcheck.py $CL"
  ( cd /Users/kamen/Leap && exec env HOME=$H2 HF_HOME=$HFH $CL/.venv/bin/python $ex $(args $t $cfg $mem $pas "$fl") ) > $D/$t.txt 2>&1 &
  CP=$!; wait $CP; RC=$?; CP=""
  say "  $t rc $RC; $(grep -o 'lifetime peak [0-9.]* GB' $D/$t.txt | tail -1); probe lines $(cat $D/${t}_probe.jsonl 2>/dev/null | wc -l | tr -d ' '); $(grep -o 'tokens == REF2 row [0-9]* / [0-9]*' $D/$t.txt | tr '\n' ';')$(grep -m1 -o '^PASS' $D/$t.txt)"
  [ "$RC" = 0 ]
}
run_a45() { if ! run a45 $G45 420 exact 45 d1,d16,d32,s32 "$F45" && [ "$RC" != skip ]; then
              if ! run a45b $G45 360 exact 45 d1,d16,d32 "$F45" && [ "$RC" != skip ]; then run a45c $G45 360 exact 45 d1,d16 "$F45"; fi
            fi
            [ "$RC" = skip ] && A45=skip || A45=done; }
while [ $(date +%s) -lt $FROM ]; do sleep 20; done
# ticket 1: b24; a45 too only if one read already shows its gate (no idle minutes in the lock)
if ticket 900 "Leap: WEEK98 cheap-check screen, lossy-24 (check cost at 1 / 13 / 32 rows, 4 short prompts), MEM 28 GB, ~7 min"; then
  if ! run b24 $G24 420 lossy 24 d1,d16,d32,s32 "$F24" && [ "$RC" != skip ]; then run b24b $G24 360 lossy 24 d1,d16 "$F24"; fi
  if [ "${A45_SAME:-1}" = 1 ] && [ "$(noswap)" -ge $G45 ]; then run_a45; else say "a45 not in this ticket: one read $(noswap) GB vs $G45"; fi
  unticket
fi
# a45: wait holding nothing until its gate shows on two reads, then its own ticket (at most two tickets)
n_=0
while [ $A45 != done ] && [ $n_ -lt 2 ] && [ $(( $(date +%s) + 420 + 240 )) -le $CUT ]; do
  if [ ! -e $Q/HOLD ] && [ ! -e $Q/PAUSE ] && ! lid_closed && [ "$(noswap)" -ge $G45 ] && gate_ok $G45; then
    n_=$((n_ + 1))
    if ticket 420 "Leap: WEEK98 cheap-check screen, exact-45 (32-row check split: dense / experts / misses, 4 short prompts), MEM 47 GB, ~7 min"; then
      run_a45; unticket
    fi
  else
    sleep $PW
  fi
done
[ $A45 = done ] || say "a45 did not run by the cutoff (NO-SWAP $(noswap) GB; $(parts)); re-arm for another night"
$PY $W/cc3_report.py $D | tee -a $LOG
say "DONE"
exit 0
