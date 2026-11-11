#!/bin/sh
# Leap's forty-sixth detached screen runner (Fri 2026-10-30 23:25; the orchestrator's 23:2x FAST INTEGRATION addendum for
# 45 GB-route runs): runs S45-STACK (REF2b's exact live loop at C 224, BigMoE's R-55 gate x L1-14 truncation, 5 runs per
# prompt; TABLE_350 'S45-STACK') QUEUE-FIRST: before queueing it waits only for PAUSE / a closed lid / ACT / soft_end,
# then takes `peer lock quiet --queue` (a queued quiet ticket stops new shared runs, and the grant waits for running ones
# to end, so memory drains for it). After the grant it settles 30 s, then checks vm_stat available >= the lifetime peak
# + 8 GB (49 GB); if short, it releases, waits 2 min and queues again. The swap guard (+2 GB) in run_locked.sh stays.
# One 45 GB ticket at a time. No retry after a run. Log: data/runner_r.log
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
Q=$HOME/.local/share/quiet
export PEER_QUEUE_TIMEOUT=21600
AVAIL_MIN=49                                      # GB available (vm_stat): C 224's ~40.5 GB lifetime peak + 8 GB
log() { echo "[runner46] $* $(date '+%H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//'), disk $(df -g / | tail -1 | awk '{print $4}') GB free"; }
avail_gb() {  # GB a new process can get: free + inactive + speculative + purgeable pages (vm_stat)
  vm_stat | awk '/page size of/ {ps = $8} /^Pages (free|inactive|speculative|purgeable):/ {v = $NF; sub("[.]", "", v); s += v} END {printf "%d", s * ps / 1e9}'
}
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
ok_pre() {   # what blocks queueing: PAUSE, a closed lid, the health monitor at ACT, soft_end's ramp (never memory)
  [ ! -e $Q/PAUSE ] && ! lid_closed && ! grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null \
    && ! pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null
}
held() {     # this session holds an unexpired quiet lock (taken by a background `peer lock quiet --queue` in the session)
  $PY -I -S -c 'import json, sys, time
d = json.load(open(sys.argv[1]))
sys.exit(0 if d.get("mode") == "quiet" and d.get("until", 0) > time.time() else 1)' "$HOME/.claude-peers/locks/$CLAUDE_PID.json" 2>/dev/null
}
screen() {   # screen <id> <why> <command...>
  id=$1; why=$2; shift 2
  while :; do
    w=0; while ! ok_pre; do [ $w -eq 0 ] && log "$id: waiting (PAUSE / ACT / lid / soft_end)"; w=1; sleep 60; done
    if held; then log "$id: lock already held ($(avail_gb) GB available)"; else log "$id: queued ($(avail_gb) GB available)"; fi
    if held || peer lock quiet --queue 15m "Leap $id: $why"; then
      sleep 30                                       # let the previous holder's memory drain
      if ! ok_pre || [ "$(avail_gb)" -lt $AVAIL_MIN ]; then
        peer unlock; log "$id: after the grant $(avail_gb) GB available (< $AVAIL_MIN) or PAUSE / lid / ACT: released"
        sleep 120; continue
      fi
      log "$id: lock ($(avail_gb) GB available)"
      ./run_locked.sh 10 "$@"; rc=$?
      log "$id: rc $rc"
      return $rc
    fi
    log "$id: no lock, queueing again"; sleep 60
  done
}
{
  log "then: S45-STACK (queue-first; slot / gate x block 16 / t 0.3, interleaved per prompt, C 224, CAP 192)"
  screen S45STACK "S45-STACK: REF2b live loop at C 224, the R-55 gate x L1-14 truncation (5 runs per prompt, every token equal), full streamed 125B (~9 min)" \
    $PY scripts/r40s3k_screen.py 224 192 speed 16,g16,0.3,g0.3,16 interleave
  cat data/s45c2_c224.txt
  log "done"
} >> data/runner_r.log 2>&1
