#!/bin/sh
# Leap's thirty-eighth detached screen runner (Fri 2026-10-30 00:40; RULES.md 'Rapid screening'): waits for the pid
# given as $1 (0: none), then runs S45b: S45's exact live loop (REF2 defaults, DFlash2 drafting, 6 prompts) at C 224,
# 256 and 288, each under its own quiet lock with the 30 s memory settle. Each C waits until memory_pressure's free
# share covers most of its peak (lifetime ~36.3 GB at C 192 + ~135 MB per slot: 66 / 69 / 75 %, ~42 / 44 / 48 GB; the
# compressor takes the rest), and skips that C if the gates stay shut ~46 min. The orchestrator (00:3x, Kamen 10-29 22:06):
# 45 GB is an estimate, so measure past it if that cuts misses; the swap guard (+2 GB) stays. If this session already
# holds a quiet lock when a screen starts, the screen uses it instead of queueing. run_locked.sh releases the lock after
# each screen, so the later screens queue as usual. No time-of-day gate; Kamen's stops come as PAUSE, which every gate
# here honours. No retry. Lines are pre-registered in TABLE_350.md. Log: data/runner_r.log
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
Q=$HOME/.local/share/quiet
export PEER_QUEUE_TIMEOUT=21600
FREE_MIN=58
log() { echo "[runner38] $* $(date '+%H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//'), disk $(df -g / | tail -1 | awk '{print $4}') GB free"; }
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
ok_now() {   # also waits while soft_end.sh ramps held jobs back (it SIGSTOPs their new children too: Fit12, 10-27)
  [ ! -e $Q/PAUSE ] && ! lid_closed && ! grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null \
    && ! pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null \
    && [ "$(memory_pressure 2>/dev/null | awk '/free percentage/ {sub("%", "", $NF); print int($NF)}')" -ge $FREE_MIN ]   # set per C below
}
held() {     # this session holds an unexpired quiet lock
  $PY -I -S -c 'import json, sys, time
d = json.load(open(sys.argv[1]))
sys.exit(0 if d.get("mode") == "quiet" and d.get("until", 0) > time.time() else 1)' "$HOME/.claude-peers/locks/$CLAUDE_PID.json" 2>/dev/null
}
screen() {   # screen <id> <why> <command...>
  id=$1; why=$2; shift 2
  while :; do
    w=0; while ! ok_now; do [ $w -eq 0 ] && log "$id: waiting (PAUSE / ACT / lid / memory >= $FREE_MIN % free)"; w=$((w + 1))
      if [ $w -gt 23 ]; then log "$id: skipped, the gates stayed shut for ~46 min"; return 99; fi; sleep 120; done
    if held; then log "$id: lock already held"; else log "$id: queued"; fi
    if held || peer lock quiet --queue 10m "Leap $id: $why"; then
      if ! ok_now; then peer unlock; log "$id: lock came under PAUSE / ACT / closed lid / low memory: released"; continue; fi
      sleep 30                                       # let the previous holder's memory drain
      if ! ok_now; then peer unlock; log "$id: memory not back 30 s after the grant: released"; sleep 60; continue; fi
      log "$id: lock"
      ./run_locked.sh 10 "$@"; rc=$?
      log "$id: rc $rc"
      return $rc
    fi
    log "$id: no lock, queueing again"; sleep 60
  done
}
{
  log "then: S45b (S45 at C 224 / 256 / 288)"
  while [ "$1" != 0 ] && kill -0 "$1" 2>/dev/null; do sleep 30; done
  FREE_MIN=66
  screen S45B224 "S45b: REF2 exact path's live speed at C 224 (~41 GB peak), 6 prompts, full streamed 125B (~3 min)" \
    $PY scripts/r40s3k_screen.py 224 256 speed
  cat data/s45_speed_c224.txt
  FREE_MIN=69
  screen S45B256 "S45b: REF2 exact path's live speed at C 256 (~45 GB peak), 6 prompts, full streamed 125B (~3 min)" \
    $PY scripts/r40s3k_screen.py 256 256 speed
  cat data/s45_speed_c256.txt
  FREE_MIN=75
  screen S45B288 "S45b: REF2 exact path's live speed at C 288 (~49 GB peak), 6 prompts, full streamed 125B (~3 min)" \
    $PY scripts/r40s3k_screen.py 288 256 speed
  cat data/s45_speed_c288.txt
  log "done"
} >> data/runner_r.log 2>&1
