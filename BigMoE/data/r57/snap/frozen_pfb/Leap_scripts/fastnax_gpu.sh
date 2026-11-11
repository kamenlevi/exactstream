#!/bin/bash
# FAST-NAX GPU checks (Leap, Thu 2026-11-05 19:1x; TABLE_350 'FAST-NAX'; the orchestrator: GPU only after T0, Fri 00:15).
# From Fri 00:16, at the first moment with no quiet lock held or queued, no PAUSE / HOLD and NO-SWAP >= 12 GB:
#   1. a share lock (MEM ~4 GB)
#   2. ref2.self_test() (the live module got the default-off FAST-NAX switch Thu 19:1x)
#   3. scripts/ooo1cq_check.py with OOQ_DIAG=1 OOQ_TIME=1 (900 s alarm, swap guard +2 GB)
#   4. unlock
# It gives up at Fri 11:00. Usage (an attached harness background task): bash ~/Leap/scripts/fastnax_gpu.sh
# -> data/fastnax_gpu.log, data/ooo1cq_check.txt
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/Leap" || exit 1
LOG="$HOME/Leap/data/fastnax_gpu.log"
PY=$HOME/BigMoE/.venv/bin/python
Q="$HOME/.local/share/quiet"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
avail() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap_mb() { sysctl -n vm.swapusage | awk '{for (i = 1; i <= NF; i++) if ($i == "used") {v = $(i + 2); sub("M", "", v); print int(v)}}'; }
NB=${FNX_NOTBEFORE:-202611060016}; NA=${FNX_NOTAFTER:-202611061100}
say "FAST-NAX GPU checks armed (pid $$, from $NB, give up at $NA)"
until [ "$(date +%Y%m%d%H%M)" -ge "$NB" ]; do sleep 30; done
while :; do
  [ "$(date +%Y%m%d%H%M)" -ge "$NA" ] && { say "no gap before $NA: nothing ran"; exit 3; }
  out=$(peer ls 2>/dev/null)
  if [ -n "$out" ] && ! printf '%s' "$out" | grep -qE 'benchmarking: quiet lock until|waiting for a quiet window' \
     && [ ! -e "$Q/PAUSE" ] && [ ! -e "$Q/HOLD" ] && [ "$(avail)" -ge 12 ]; then break; fi
  sleep 30
done
peer lock share 20m --cpu 200 --gpu 60 "Leap: FAST-NAX GPU checks (ooo1cq_check + ref2 self_test), MEM ~4 GB" >> "$LOG" 2>&1 \
  || { say "share lock refused: nothing ran"; exit 4; }
SW0=$(swap_mb)
say "share lock taken (NO-SWAP $(avail) GB, swap $SW0 MB)"
guard() {   # guard PID: stop it if swap grows > 2 GB
  while kill -0 "$1" 2>/dev/null; do
    sleep 5
    if [ $(( $(swap_mb) - SW0 )) -gt 2048 ]; then say "SWAP GUARD: stopping pid $1"; kill "$1" 2>/dev/null; sleep 10; kill -9 "$1" 2>/dev/null; return; fi
  done
}
VECLIB_MAXIMUM_THREADS=4 perl -e 'alarm shift; exec @ARGV' 300 "$PY" -B -c "
import os, sys
for p in ('~/BigMoE', '~/Leap/third_party/ms_gdn', '~/Leap/scripts'):
    sys.path.insert(0, os.path.expanduser(p))
import ref2
ref2.apply()          # self_test needs _ORIG from apply() (Thu 23:22: KeyError qmm without it, a false restore)
r = ref2.self_test()
ref2.remove()
print('ref2.self_test ->', r, flush=True)
sys.exit(0 if r and all(r.values()) else 1)" >> "$LOG" 2>&1 &
guard $!; wait $!; st=$?; say "ref2.self_test rc=$st"
OOQ_DIAG=1 OOQ_TIME=1 OOQ_COND="share lock" VECLIB_MAXIMUM_THREADS=4 perl -e 'alarm shift; exec @ARGV' 900 \
  "$PY" -B scripts/ooo1cq_check.py >> "$LOG" 2>&1 &
guard $!; wait $!; ck=$?; say "ooo1cq_check rc=$ck"
peer unlock >> "$LOG" 2>&1
say "unlocked; results: data/ooo1cq_check.txt"
# BigMoE 19:2x: their Fri ~01:40 run imports ref2 live and pre-flighted the pre-19:1x ref2, so ANY failure here (the
# self-test, the check's (a) / (b), or either process) restores that version, atomically, and logs it
blk=$(awk '/^FAST-NAX check /{b=""} {b = b $0 "\n"} END {printf "%s", b}' "$HOME/Leap/data/ooo1cq_check.txt" 2>/dev/null)
fail=
[ "$st" != 0 ] && fail="ref2.self_test rc $st"
[ "$ck" != 0 ] && fail="$fail; ooo1cq_check rc $ck"
printf '%s' "$blk" | grep -qE '^\((a|b)\) .*FAIL' && fail="$fail; (a) / (b) FAIL"
printf '%s' "$blk" | grep -qE '^\(a\) .*PASS' || fail="$fail; no (a) PASS line"
if [ -n "$fail" ]; then
  BK="$HOME/Leap/data/ref2_before_nax1cq.py"; R2="$HOME/Leap/scripts/ref2.py"
  if [ "$(md5 -q "$BK")" = 04f18c02b4503618365f44b84dc52e6b ] && cp "$BK" "$R2.restore_tmp" && mv "$R2.restore_tmp" "$R2" \
     && [ "$(md5 -q "$R2")" = 04f18c02b4503618365f44b84dc52e6b ]; then
    say "RESTORED ref2.py to the pre-19:1x version (md5 04f18c02) because: ${fail#; }"
  else
    say "RESTORE FAILED (because: ${fail#; }): restore ~/Leap/data/ref2_before_nax1cq.py by hand"
  fi
else
  say "all checks passed: ref2 stays (REF2_NAX_1CQ default off)"
fi
