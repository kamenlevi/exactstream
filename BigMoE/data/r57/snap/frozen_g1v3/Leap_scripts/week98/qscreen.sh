#!/bin/bash
# qscreen.sh (Leap, Sun 15 Nov 2026 01:1x): one WEEK98 screen under its own short quiet ticket (the orchestrator 01:0x:
# "Each runs under its own short quiet ticket (<= 15 min per item, released between items) ... Use the two-read NO-SWAP
# gate. Record High Power Mode.").
#   - The gate comes first, holding nothing: NO-SWAP (free + purgeable + file-backed) >= gate on the LOWER of two reads
#     30 s apart, parts logged. It waits up to WAIT s (default 1800) for that, then gives up without queueing.
#   - Then a quiet ticket (peer lock quiet --queue); after the grant, settle 20 s, re-check the gate (two reads), log the
#     power mode, start a swap guard (+2 GB stops the screen), check HOLD / PAUSE / lid / HEALTH ACT.
#   - The command runs from ~/Leap with its output in data/week98/screens/<tag>.txt; a time cap kills it at <est> + 120 s.
#   - The ticket is released at once (trap on every exit).
# DRY=1: fake lock, 1 s waits, gate 1 GB (the head test). Never lowers a gate.
# Usage: qscreen.sh <tag> <gate GB> <est s> "<reason, with MEM n GB>" <command ...>
TAG=${1:?tag}; GATE=${2:?gate}; EST=${3:?est}; WHY=${4:?reason}; shift 4
[ $# -gt 0 ] || { echo "no command"; exit 2; }
D=/Users/kamen/Leap/data/week98/screens; LOG=$D/screens.log; Q=$HOME/.local/share/quiet
DRY=${DRY:-0}; ZZ=30; ST=20; WAIT=${WAIT:-1800}
[ $DRY = 1 ] && { GATE=1; ZZ=1; ST=1; }
mkdir -p $D
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
pmode() { pmset -g | awk '/^ *powermode/ {print $2; exit}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
parts() { vm_stat | awk '/page size of/ {ps=$8} {v=$NF; sub("[.]","",v)} /^Pages free:/ {f=v} /^Pages purgeable:/ {p=v}
  /^File-backed pages:/ {fb=v} /^Pages speculative:/ {sp=v} /^Anonymous pages:/ {an=v} /^Pages occupied by compressor:/ {c=v}
  END {g=ps/1e9; printf "free %.1f + purgeable %.1f + file-backed %.1f (speculative %.1f inside); anon %.1f, compressor %.1f", f*g, p*g, fb*g, sp*g, an*g, c*g}'; }
say() { echo "$(date '+%m-%d %H:%M:%S') [$TAG] $*" | tee -a $LOG; }
gate_ok() { local a b pa pb
  a=$(noswap); pa=$(parts); sleep $ZZ; b=$(noswap); pb=$(parts)
  say "  gate $GATE: read 1 $a GB ($pa); read 2 (+$ZZ s) $b GB ($pb)"
  [ $a -ge $GATE ] && [ $b -ge $GATE ]; }
killtree() { local c; for c in $(pgrep -P $1 2>/dev/null); do killtree $c; done; kill -TERM $1 2>/dev/null; }
LOCKED=0; CP=""; GUARD=""
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null
            [ -n "$CP" ] && kill -0 $CP 2>/dev/null && killtree $CP
            if [ $LOCKED = 1 ]; then [ $DRY = 1 ] || peer unlock >/dev/null 2>&1; say "ticket released"; fi
            say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT; trap 'say "stopped by a signal"; exit 1' INT TERM
stop_checks() { [ -e $Q/HOLD ] && { say "STOP: HOLD"; exit 0; }
                [ -e $Q/PAUSE ] && { say "STOP: PAUSE"; exit 0; }
                lid_closed && { say "STOP: lid closed"; exit 0; }
                grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null && { say "STOP: HEALTH ACT"; exit 0; }; }
say "start: gate $GATE GB, est $EST s, power mode $(pmode); $*"
stop_checks
t0=$(date +%s); ok=0
while [ $(( $(date +%s) - t0 )) -lt $WAIT ]; do
  if [ "$(noswap)" -ge $GATE ] && gate_ok; then ok=1; break; fi
  sleep $(( DRY == 1 ? 1 : 30 ))
done
[ $ok = 1 ] || { say "SKIP: NO-SWAP below $GATE for $WAIT s; now $(noswap) GB ($(parts))"; exit 3; }
say "queue the quiet ticket (NO-SWAP $(noswap) GB)"
if [ $DRY = 1 ]; then echo "(dry) fake ticket" >> $LOG; else
  PEER_QUEUE_TIMEOUT=3600 peer lock quiet --queue 30m "$WHY" >> $LOG 2>&1 || { say "the ticket left the queue without a grant"; exit 4; }
fi
LOCKED=1; say "granted (NO-SWAP $(noswap) GB, power mode $(pmode)); settle $ST s"; sleep $ST
stop_checks
gate_ok || { say "SKIP after the grant: NO-SWAP below $GATE ($(parts))"; exit 3; }
s0=$(swap); ( while :; do [ $(( $(swap) - s0 )) -gt 2048 ] && { echo "$(date +%T) [$TAG] SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) & GUARD=$!
echo "power mode $(pmode) at $(date '+%m-%d %H:%M:%S')" > $D/${TAG}_pmode.txt
say "run (power mode $(pmode), swap $(swap) MB)"
( cd /Users/kamen/Leap && exec "$@" ) > $D/$TAG.txt 2>&1 &
CP=$!
( sleep $(( EST + 120 )); kill -0 $CP 2>/dev/null && { echo "$(date +%T) [$TAG] TIME CAP: stopped at $(( EST + 120 )) s" >> $LOG; kill -TERM $CP; } ) &
CAP=$!
wait $CP; RC=$?; CP=""; kill $CAP 2>/dev/null
say "rc $RC after $(( $(date +%s) - t0 )) s since start; output data/week98/screens/$TAG.txt ($(wc -l < $D/$TAG.txt | tr -d ' ') lines)"
exit $RC
