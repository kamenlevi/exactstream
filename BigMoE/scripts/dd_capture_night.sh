#!/bin/bash
# DEEPDRAFT night capture (Thu 12 Nov; the orchestrator 00:0x; docs/PREREG_DEEPDRAFT_CAPTURE_2026-11-12.md). Runs
# scripts/dd_capture.py (DLOG_SRC=magicoder, C 224) until the 13:30 hard stop (an epoch deadline). It resumes after a crash,
# since the capture skips prompts it already has (at most 4 tries). DLOG-ONPATH-FAIL (rc 3) is not retried. A swap guard
# stops it if swap grows more than 2 GB, and each try needs NO-SWAP >= 42 GB first. The capture's own time budget is set
# from the moment each try starts.
# usage (under turn45, last start 12:30): DD_NIGHT=n1112 bash scripts/dd_capture_night.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
export BIGMOE_EXT_BUILD=${BIGMOE_EXT_BUILD:-build_qom}   # 00:5x amendment 2: the current stack's ext (dlog's default build_next
                                                     # lacks set_direct_split for the store's native+p11+pad records3)
NIGHT=${DD_NIGHT:-n1112}
D=$HOME/FableGod/deepdraft/capture; mkdir -p "$D"
LOG=$D/night_$NIGHT.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=$HOME/BigMoE/.venv/bin/python
END=$(date -j -f "%Y-%m-%d %H:%M:%S" "$(date +%Y-%m-%d) ${DD_END:-13:30}:00" +%s)
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap_mb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'; }
# 00:2x (the orchestrator: Kamen's CLI sessions keep ~15 GB, NO-SWAP ~32-37 GB): the cache C from the measured room, peak [P]
# ~ 23.7 GB at C 100 rising ~0.164 GB per C to ~44 at C 224, + 8 GB margin. Tokens and features do not depend on C.
pick_c() { local a=$1; if [ "$a" -ge 52 ]; then echo 224; elif [ "$a" -ge 47 ]; then echo 192; elif [ "$a" -ge 42 ]; then echo 160;
  elif [ "$a" -ge 37 ]; then echo 128; elif [ "$a" -ge 32 ]; then echo 100; else echo 0; fi; }
S0=$(swap_mb)
say "night $NIGHT start (pid $$; hard stop $(date -r $END '+%a %H:%M'); swap0 ${S0} MB)"
tries=0
while :; do
  left=$((END - $(date +%s) - 120))
  [ $left -lt 600 ] && { say "under 10 min to the hard stop: done"; break; }
  A=$(avail_gb)
  C=$(pick_c "$A")
  if [ "$C" -eq 0 ]; then say "NO-SWAP ${A} GB < 32: waiting 120 s"; sleep 120; continue; fi
  say "try $tries: NO-SWAP ${A} GB -> C $C; budget $((left - 300)) s, hard ${left} s"
  DLOG_SRC=magicoder DLOG_NIGHT=$NIGHT DLOG_TARGET=50000000 DLOG_CAP_GB=${DD_CAP_GB:-80} DLOG_MIN_FREE_GB=115 \
    DLOG_BUDGET=$((left - 300)) DLOG_HARD=$left "$PY" scripts/dd_capture.py $C >> "$D/capture_$NIGHT.log" 2>&1 &
  P=$!
  stop=""
  while kill -0 $P 2>/dev/null; do
    S=$(swap_mb)
    if [ "$S" -gt $((S0 + 2048)) ]; then say "swap ${S} MB > ${S0}+2048: stopping"; stop=swap; kill -TERM $P; break; fi
    if [ "$(date +%s)" -ge "$END" ]; then say "hard stop: stopping"; stop=hard; kill -TERM $P; break; fi
    sleep 20
  done
  wait $P; rc=$?
  say "capture rc=$rc${stop:+ (stopped: $stop)}"
  [ -n "$stop" ] && break
  [ $rc -eq 0 ] && break
  [ $rc -eq 3 ] && { say "DLOG-ONPATH-FAIL: not retried"; break; }
  tries=$((tries + 1)); [ $tries -ge 4 ] && { say "4 failed tries: stop"; break; }
  sleep 60
done
say "night $NIGHT done: $(ls "$D/$NIGHT" 2>/dev/null | grep -c '\.npz$') files, $(du -sh "$D/$NIGHT" 2>/dev/null | cut -f1)"
