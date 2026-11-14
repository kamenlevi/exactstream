#!/bin/bash
# FLAT-2 at the largest C that fits NO-SWAP (Leap, Thu 2026-11-05 23:4x).
# Why: the orchestrator expects NO-SWAP ~33-38 GB tonight, and FLAT does not depend on C (zero's INSTALL).
# zero's run_flat2.sh stops at C 160 (36.0 GB). This wrapper adds BigMoE's fri_night rungs (NO-SWAP needed = lifetime
# peak + ~4 GB, 0.134 GB per C slot), picks C at run time and passes it as FLAT2_C. zero's files stay untouched.
# The chosen C goes to the log line below and to TABLE_350's record.
# usage: turn45's CMD (HAVE_LOCK=1 KEEP_LOCK=1 pass through). FLAT2_AVAIL=<GB> and FLAT2_DRY=1 work as in run_flat2.sh.
LOG="$HOME/Leap/data/flat2_ladder.log"
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${FLAT2_AVAIL:-$(avail_gb)}
CC=${FLAT2_C:-}
if [ -z "$CC" ]; then
  for c in 224:44.3 208:42.2 192:40.0 176:37.9 160:35.7 144:33.6 128:31.4; do
    awk -v a="$A" -v n="${c#*:}" 'BEGIN {exit !(n <= a)}' && { CC=${c%%:*}; break; }
  done
fi
[ -z "$CC" ] && { echo "$(date '+%m-%d %H:%M:%S') FLAT-2: NO-SWAP $A GB < 31.4, nothing fits, nothing runs" | tee -a "$LOG"; exit 0; }
echo "$(date '+%m-%d %H:%M:%S') FLAT-2 at C $CC (NO-SWAP $A GB)${FLAT2_DRY:+ (dry)}" | tee -a "$LOG"
FLAT2_C=$CC FLAT2_AVAIL=$A exec /bin/bash "$HOME/FableGod/zero/build_1105/run_flat2.sh" "$@"
