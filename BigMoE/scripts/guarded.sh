#!/bin/bash
# guarded.sh SECS LOG CMD...: one day test with c45_runner's guards: CMD in its own process group under an alarm (SECS), the
# swap guard (swap grows > 2 GB over the start: stop), and a NO-SWAP gate (NEED GB, env; default none). For shared-lock
# exactness / function screens (DAY9 11-02: NO-SWAP >= peak + 4 GB, swap guard +2 GB); timings take a quiet lock instead.
# WAITUNTIL=HHMM waits for NEED GB until then (else exit 9). LOCKWHY="..." first waits until no other session holds or awaits
# a quiet lock (peer wait), then takes a shared lock for the run and drops it after.
# usage: NEED=34 WAITUNTIL=1800 LOCKWHY="BigMoE: ..." bash scripts/guarded.sh 900 data/r57/x.out env R57_...=... \
#          .venv/bin/python scripts/r57_screen.py 160 64
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
secs=$1 log=$2; shift 2
cd "$HOME/BigMoE" || exit 1
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}'; }
memwait() {
  a=$(noswap)
  if [ -n "$NEED" ] && [ -n "$WAITUNTIL" ] && [ "$a" -lt "$NEED" ]; then
    echo "$(date +%H:%M:%S) waiting for NO-SWAP >= $NEED GB (now $a) until $WAITUNTIL" >> "$log"
    while [ "$a" -lt "$NEED" ] && [ "$(date +%H%M)" -lt "$WAITUNTIL" ]; do sleep 30; a=$(noswap); done
  fi
  [ -z "$NEED" ] || [ "$a" -ge "$NEED" ]
}
memwait || { echo "$(date +%H:%M:%S) skipped: NO-SWAP $a GB < $NEED" >> "$log"; exit 9; }
if [ -n "$LOCKWHY" ]; then
  peer wait --timeout 60m >> "$log" 2>&1
  memwait || { echo "$(date +%H:%M:%S) skipped after peer wait: NO-SWAP $a GB < $NEED" >> "$log"; exit 9; }
  peer lock share "${LOCKMIN:-30}m" --cpu 200 --gpu 80 "$LOCKWHY" >> "$log" 2>&1
fi
S0=$(swap_gb)
echo "$(date +%H:%M:%S) start (NO-SWAP $a GB, swap $S0 GB): $*" >> "$log"
set -m
perl -e 'alarm shift; exec @ARGV' "$secs" "$@" >> "$log" 2>&1 &
jp=$!
while kill -0 $jp 2>/dev/null; do
  sleep 5
  if awk -v s="$(swap_gb)" -v s0="$S0" 'BEGIN { exit !(s - s0 > 2) }'; then
    echo "$(date +%H:%M:%S) SWAP GUARD: stopped" >> "$log"; kill -TERM -- -$jp; sleep 3; kill -9 -- -$jp 2>/dev/null; break
  fi
done
wait $jp; rc=$?
echo "$(date +%H:%M:%S) end rc=$rc (swap $(swap_gb) GB)" >> "$log"
[ -z "$LOCKWHY" ] || peer unlock >> "$log" 2>&1
exit $rc
