#!/bin/sh
# Leap's fifty-seventh detached screen runner (Sun 2026-11-01 06:58; night mode): K-62L: the 45 GB default's live loop
# (REF2-T 4433333333 / 3c on the R-55 gate + L1-14 truncation t 0.3, C 224) with the tier GLU on 3c vs on NAX4 (ref2 CFG
# tier_nax4), ABBA per prompt, 4 prompts. QUEUE-FIRST; after the grant a 30 s settle, then NO-SWAP memory (free +
# speculative + purgeable + file-backed) picks C for REF2-T (peak 33.4 GB at C 224, + 4): >= 37.5 GB -> 224, >= 34.5 -> 192,
# >= 32.0 -> 160, else release and wait 15 min.
# The swap guard (+2 GB) stays (run_locked.sh). One run, no retry. Log: data/runner_r.log
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
Q=$HOME/.local/share/quiet
export PEER_QUEUE_TIMEOUT=21600
AVAIL_MIN=450                                     # tenths of a GB available (vm_stat NO-SWAP)
log() { echo "[runner57] $* $(date '+%H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//'), disk $(df -g / | tail -1 | awk '{print $4}') GB free"; }
avail_gb() {
  vm_stat | awk '/page size of/ {ps = $8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v = $NF; sub("[.]", "", v); s += v} END {printf "%d", s * ps / 1e8}'   # tenths of a GB, NO-SWAP
}
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
ok_pre() {
  [ ! -e $Q/PAUSE ] && ! lid_closed && ! grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null \
    && ! pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null
}
held() {
  $PY -I -S -c 'import json, sys, time
d = json.load(open(sys.argv[1]))
sys.exit(0 if d.get("mode") == "quiet" and d.get("until", 0) > time.time() else 1)' "$HOME/.claude-peers/locks/$CLAUDE_PID.json" 2>/dev/null
}
screen() {   # screen <id> <why> <command...>
  id=$1; why=$2; shift 2
  while :; do
    w=0; while ! ok_pre; do [ $w -eq 0 ] && log "$id: waiting (PAUSE / ACT / lid / soft_end)"; w=1; sleep 60; done
    if held; then log "$id: lock already held ($(avail_gb) / 10 GB available)"; else log "$id: queued ($(avail_gb) / 10 GB available)"; fi
    if held || peer lock quiet --queue 20m "Leap $id: $why"; then
      sleep 30
      A=$(avail_gb)                                  # adaptive C (REF2-T's lifetime peak + 4 GB; NewIdeas RANKED_45GB #4:
      # REF2-T peaks 33.0-33.4 GB at C 224): 224 >= 37.5 GB / 192 >= 34.5 / 160 >= 32.0
      if [ "$A" -ge 375 ]; then CC=224; elif [ "$A" -ge 345 ]; then CC=192; elif [ "$A" -ge 320 ]; then CC=160; else CC=0; fi
      if ! ok_pre || [ $CC -eq 0 ]; then
        peer unlock; log "$id: after the grant $A / 10 GB available (< 32.0 GB for C 160) or PAUSE / lid / ACT: released"
        sleep 900; continue
      fi
      log "$id: lock ($A / 10 GB available): C $CC"
      ./run_locked.sh 16 "$@" $CC 256; rc=$?
      log "$id: rc $rc"
      return $rc
    fi
    log "$id: no lock, queueing again"; sleep 60
  done
}
{
  log "then: K-62L (C 224 / 192 / 160 by memory, the 45 GB default with 3c vs NAX4 tier GLU, ABBA, 4 prompts, CAP 256)"
  screen K62L "K-62L at C 224: the 45 GB default's live loop (REF2-T on the gate + t 0.3), tier GLU 3c vs NAX4 (matrix units), ABBA x 4 prompts, full streamed 125B (~12 min)" \
    env BIGMOE_MAXK=160 BIGMOE_EXT_BUILD=build_next R57_TIER=1 R57_TRUNC=0.3 R57_ARMS=gate+t,gate+t+nax R57_PROMPTS=4 R57_WARM=1 R57_TAG=k62l \
    $PY scripts/k62_live.py
  cat data/k62_live/k62l_screen.txt
  log "done"
} >> data/runner_r.log 2>&1
