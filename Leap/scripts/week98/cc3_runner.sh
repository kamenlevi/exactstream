#!/bin/bash
# cc3_runner.sh (Leap, Sat 14 Nov 2026 19:0x): WEEK98 route 3 screen 3 (the orchestrator 18:2x: "yes to screen 3, the real
# 32-row check at 45 GB split into dense / experts / misses ... queue a quiet lock around 20:30 (peer lock quiet --queue
# 30m) and let it wait its turn"). One quiet ticket, two runs back to back on the v121 clean clone (engine unchanged; the
# probe rides the loop's own R57_COPYPROBE hook, scripts/week98/cc3_probe.py), 4 HumanEval keys of the shipped REF2 rows,
# cap 256, R57_WARMSPLIT=4 (the loop's cold / warm 1-4-8-16 split every 4th check):
#   A  exact-45 (C 256; tokens must equal REF2, so the probe's rollbacks are checked by the run itself)   gate 47 = 42.5 + 4
#   B  lossy-24 (C 100, at most 13 rows; QOM)                                                            gate 28 = 23.8 + 4
# A retries with fewer probe passes if it fails (no s32; then no d32). Gates per run: NO-SWAP = free + purgeable +
# file-backed (the 11-14 05:4x formula); the first wait stops at grant + 4 min (the lock must not idle past minute 8).
# Swap +2 GB stops; HOLD / PAUSE / lid / HEALTH ACT checked before each run; no other engine may run. Nothing may end
# after CUTOFF. Usage: cc3_runner.sh "<FROM YYYY-MM-DD HH:MM:SS>" "<CUTOFF YYYY-MM-DD HH:MM:SS>"
V=/Users/kamen/Leap/release/cleanclone/v121; CL=$V/clone; H2=$V/home2; HFH=$HOME/.cache/huggingface
D=/Users/kamen/Leap/data/week98/cc3; PROBE=/Users/kamen/Leap/scripts/week98/cc3_probe.py; Q=$HOME/.local/share/quiet
LOG=$D/cc3.log; PY=/opt/homebrew/opt/python@3.12/bin/python3.12
KEYS=humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6
FROM=$(date -j -f "%Y-%m-%d %H:%M:%S" "${1:?FROM}" +%s); CUT=$(date -j -f "%Y-%m-%d %H:%M:%S" "${2:?CUTOFF}" +%s)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|run_map[.]py|fit_onecall[.]py|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
parts() { vm_stat | awk '/page size of/ {ps=$8} {v=$NF; sub("[.]","",v)} /^Pages free:/ {f=v} /^Pages purgeable:/ {p=v}
  /^File-backed pages:/ {fb=v} /^Pages speculative:/ {sp=v} /^Anonymous pages:/ {an=v} /^Pages occupied by compressor:/ {c=v}
  END {g=ps/1e9; printf "free %.1f + purgeable %.1f + file-backed %.1f (speculative %.1f inside); anon %.1f, compressor %.1f", f*g, p*g, fb*g, sp*g, an*g, c*g}'; }
gate_ok() {   # v3 (the orchestrator 18:2x, the 05:44 trap): the LOWER of two reads 30 s apart must reach the gate; both logged
  local a b pa pb
  a=$(noswap); pa=$(parts); sleep 30; b=$(noswap); pb=$(parts)
  say "  gate $1: read 1 $a GB ($pa); read 2 (+30 s) $b GB ($pb)"
  [ $a -ge $1 ] && [ $b -ge $1 ]
}
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
killtree() { local c; for c in $(pgrep -P $1 2>/dev/null); do killtree $c; done; kill -TERM $1 2>/dev/null; }
LOCKED=0; CP=""; GUARD=""; RC=""
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null
            [ -n "$CP" ] && kill -0 $CP 2>/dev/null && killtree $CP
            [ $LOCKED = 1 ] && { peer unlock >/dev/null 2>&1; say "ticket released"; }
            say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT; trap 'say "stopped by a signal"; exit 1' INT TERM
for f in $PROBE $CL/run.py $CL/harness/results/ref2/decode.jsonl; do [ -e $f ] || { echo "no $f"; exit 2; }; done
mkdir -p $D $H2
say "armed (pid $$): cc3 screen 3, runs ${RUNS:-b24 a45} with the probe, one quiet ticket from $1, every run ends by $2"
while [ $(date +%s) -lt $FROM ]; do sleep 20; done
TO=$(( CUT - $(date +%s) - 1200 )); [ $TO -lt 120 ] && { say "SKIP: under 22 min to the cutoff"; exit 0; }
say "queue the quiet ticket (PEER_QUEUE_TIMEOUT $TO s; NO-SWAP $(noswap) GB)"
if PEER_QUEUE_TIMEOUT=$TO peer lock quiet --queue 30m "Leap: WEEK98 cheap-check screen (check cost at 1 / 16 / 32 rows, exact-45 + lossy-24, 4 short prompts each), MEM 47 GB, ~15 min" >> $LOG 2>&1; then
  LOCKED=1
else
  say "the ticket left the queue without a grant"; exit 0
fi
GR=$(date +%s); say "granted (NO-SWAP $(noswap) GB); settle 20 s"; sleep 20
S0=$(swap); ( while :; do [ $(( $(swap) - S0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) &
GUARD=$!
run() {   # run <tag> <gate GB> <est s> <config> <mem> <passes>
  local t=$1 gate=$2 est=$3 cfg=$4 mem=$5 pas=$6
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
  [ $ok = 1 ] || { say "SKIP $t: NO-SWAP below $gate (its peak + 4) on the lower of two reads 30 s apart; now $(noswap) GB ($(parts))"; RC=skip; return 1; }
  [ $(( $(date +%s) + est )) -gt $CUT ] && { say "SKIP $t: it could end after the cutoff (est $est s)"; RC=skip; return 1; }
  say "run $t: $cfg-$mem, probe passes $pas (NO-SWAP $(noswap) GB, swap $(swap) MB)"
  rm -f $D/${t}_probe.jsonl
  ( cd /Users/kamen/Leap && exec env HOME=$H2 HF_HOME=$HFH $CL/.venv/bin/python $CL/run.py --config $cfg --mem $mem --keys $KEYS \
      --max-tokens 256 --env R57_WARMSPLIT=4 --env R57_COPYPROBE=1 --env R57_COPYPROBE_PY=$PROBE \
      --env CC3_OUT=$D/${t}_probe.jsonl --env CC3_PASSES=$pas --out $D/$t ) > $D/$t.txt 2>&1 &
  CP=$!; wait $CP; RC=$?; CP=""
  say "  $t rc $RC; $(grep -o 'lifetime peak [0-9.]* GB' $D/$t.txt | tail -1); probe lines $(wc -l < $D/${t}_probe.jsonl 2>/dev/null | tr -d ' '); $(grep -o 'tokens == REF2 row [0-9]* / [0-9]*' $D/$t.txt | tr '\n' ';')"
  [ "$RC" = 0 ]
}
# v2 (18:3x, the orchestrator 18:2x: exact-45's gate 47 is unlikely at 20:30, ~38-42 GB with Chrome open): lossy-24 FIRST
# (its pressure may move idle pages into the compressor and raise NO-SWAP, as stage 2 did on 11-14 00:40), then exact-45.
# RUNS picks the runs ("b24 a45" default; "a45" to re-arm exact-45 alone later). A failed run (rc != 0, not a skip)
# retries with fewer probe passes: no s32, then no d32.
for r_ in ${RUNS:-b24 a45}; do
  case $r_ in
    a45) if ! run a45 47 420 exact 45 d1,d16,d32,s32 && [ "$RC" != skip ]; then
           if ! run a45b 47 360 exact 45 d1,d16,d32 && [ "$RC" != skip ]; then run a45c 47 360 exact 45 d1,d16; fi
         fi ;;
    b24) if ! run b24 28 420 lossy 24 d1,d16,d32,s32 && [ "$RC" != skip ]; then run b24b 28 360 lossy 24 d1,d16; fi ;;
  esac
done
say "runs done at grant + $(( $(date +%s) - GR )) s"
peer unlock >/dev/null 2>&1; LOCKED=0; say "ticket released (before the report)"
$PY /Users/kamen/Leap/scripts/week98/cc3_report.py $D | tee -a $LOG
say "DONE"
exit 0
