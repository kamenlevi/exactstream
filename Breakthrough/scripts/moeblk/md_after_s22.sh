#!/bin/bash
# md_after_s22.sh (Breakthrough, Tue 17 Nov 2026 17:4x; MOE BLOCK DIAG): the orchestrator 17:4x moved the pre-registered
# quiet split (docs/MOEBLOCK.md DIAG + amendments; runner data/moeblk/md_run2.frozen.sh on frozen_q4, both UNCHANGED) to the
# first quiet gap AFTER S22's training ends (~20:35-20:39) and before BigMoE's M4 (23:35).
# This waiter holds no lock and no ticket. It starts md_run2 (MD_START = now) only when, on two reads ZZ s apart:
#   1. TrainLab's s22.sh (the pid read at arming, S22PID) is gone: a queued quiet ticket makes S22 yield (its guard counts
#      'waiting for a quiet window'; 17:38 it did, once), so no ticket may exist while S22 lives;
#   2. no other session holds or queues a lock (peer ls '- ' lines: shared run / benchmarking / waiting for a quiet window /
#      quiet lock lapsed).
# md_run2 keeps its own gates (holds, away >= 900 s, NO-SWAP >= 13 at the grant) and quits on a grant after MD_LATEST + 1 h,
# so with MD_LATEST 22:15 nothing starts after 23:15 and the run (<= 2 x 4 min caps) ends before M4.
# Gives up at MD_LATEST (MD_END "SKIP ..."). Env: DRY=1 (needs XQ_DRYDIR; PEERLS_CMD, RJ_AWAY_CMD, RJ_JOB pass through).
cd "$HOME/Breakthrough" || exit 2
DRY=${DRY:-0}; ZZ=60; [ "$DRY" = 1 ] && ZZ=1
S22PID=${S22PID:?the s22.sh pid read at arming}
MDL=${MD_LATEST:-2026-11-17 22:15:00}
LATEST=$(date -j -f "%Y-%m-%d %H:%M:%S" "$MDL" +%s)
D=data/moeblk; [ "$DRY" = 1 ] && D=${XQ_DRYDIR:?DRY needs XQ_DRYDIR}; L=$D/md_runner.log; END=$D/MD_END
say() { echo "$(date '+%a %H:%M:%S') $*" >> "$L"; }
s22_alive() { ps -p "$S22PID" -o command= 2>/dev/null | grep -q 's22[.]sh'; }
others_locks() { ${PEERLS_CMD:-peer ls} 2>/dev/null | grep '^- ' |
                 grep -cE '· (shared run|benchmarking|waiting for a quiet window|quiet lock lapsed)'; }
busy() {
  s22_alive && { echo "S22 (pid $S22PID) still running"; return; }
  local n; n=$(others_locks); [ "${n:-0}" -gt 0 ] && { echo "$n other lock(s) held or queued"; return; }
  echo ""
}
say "after-S22 waiter armed (pid $$): S22 pid $S22PID, then a quiet gap (no other lock on two reads ${ZZ} s apart); latest start $(date -r "$LATEST" '+%a %H:%M')"
last=""
while :; do
  b=$(busy)
  if [ -z "$b" ]; then sleep $ZZ; b=$(busy); [ -z "$b" ] && break; fi
  if [ "$(date +%s)" -ge "$LATEST" ]; then
    echo "MD END $(date '+%m-%d %H:%M:%S'): SKIP: no quiet gap after S22 by the latest start ($b)" > "$END"
    say "END: SKIP: no quiet gap after S22 by the latest start ($b)"; exit 7
  fi
  [ "$b" != "$last" ] && { say "waiting: $b"; last=$b; }
  sleep $ZZ
done
say "quiet gap after S22: starting md_run2 now"
export MD_START="$(date '+%Y-%m-%d %H:%M:%S')" MD_LATEST="$MDL"
exec bash data/moeblk/md_run2.frozen.sh
