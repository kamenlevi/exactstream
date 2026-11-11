#!/bin/sh
# Leap's forty-ninth detached screen runner (Fri 2026-10-30 23:58; the orchestrator's 23:5x go for one more C 224 try with
# the NO-SWAP gate): S45-STACK at C 224, the 45 GB point (REF2b's exact live loop, the R-55 gate x L1-14 truncation, 5
# runs per prompt, every token equal). Runner 48's 23:51 try tripped the swap guard at 45.0 GB vm_stat 'available'
# because 'inactive' held anonymous app memory (Chrome had been reopened). QUEUE-FIRST: before queueing it waits only for
# PAUSE / a closed lid / ACT / soft_end; after the grant a 30 s settle, then NO-SWAP memory (free + speculative + purgeable
# + file-backed) >= the lifetime peak + 4 GB (44.5 GB); if short it releases and waits 15 min. The swap guard (+2 GB)
# stays. If this trips too, C 224 stops for tonight (the orchestrator). One run, no retry. Log: data/runner_r.log
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
Q=$HOME/.local/share/quiet
export PEER_QUEUE_TIMEOUT=21600
AVAIL_MIN=445                                     # tenths of a GB available (vm_stat): C 224's ~40.5 GB lifetime peak + 4 GB (quiet-lock margin, orchestrator 23:3x)
log() { echo "[runner49] $* $(date '+%H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//'), disk $(df -g / | tail -1 | awk '{print $4}') GB free"; }
avail_gb() {  # NO-SWAP memory (orchestrator 23:5x rule): free + speculative + purgeable + file-backed pages (vm_stat); 'inactive' holds anonymous app memory that only swap frees
  vm_stat | awk '/page size of/ {ps = $8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v = $NF; sub("[.]", "", v); s += v} END {printf "%d", s * ps / 1e8}'   # tenths of a GB, NO-SWAP
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
    if held; then log "$id: lock already held ($(avail_gb) / 10 GB available)"; else log "$id: queued ($(avail_gb) / 10 GB available)"; fi
    if held || peer lock quiet --queue 15m "Leap $id: $why"; then
      sleep 30                                       # let the previous holder's memory drain
      if ! ok_pre || [ "$(avail_gb)" -lt $AVAIL_MIN ]; then
        peer unlock; log "$id: after the grant $(avail_gb) / 10 GB available (< $AVAIL_MIN / 10) or PAUSE / lid / ACT: released"
        sleep 900; continue
      fi
      log "$id: lock ($(avail_gb) / 10 GB available)"
      ./run_locked.sh 10 "$@"; rc=$?
      log "$id: rc $rc"
      return $rc
    fi
    log "$id: no lock, queueing again"; sleep 60
  done
}
{
  log "then: S45-STACK at C 224 (the 45 GB point; queue-first; slot / gate x block 16 / t 0.3, CAP 192)"
  screen S45STACK "S45-STACK at C 224 (the 45 GB point): REF2b live loop, the R-55 gate x L1-14 truncation (5 runs per prompt, every token equal), full streamed 125B (~9 min)" \
    $PY scripts/r40s3k_screen.py 224 192 speed 16,g16,0.3,g0.3,16 interleave
  cat data/s45c2_c224.txt
  log "done"
} >> data/runner_r.log 2>&1
