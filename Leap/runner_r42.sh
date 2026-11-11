#!/bin/sh
# Leap's forty-second detached screen runner (Fri 2026-10-30 04:00; RULES.md 'Rapid screening'): waits for the pid given
# as $1 (0: none), then runs S45c (NewIdeas' TRUNC125: L1-14's running-confidence truncation in REF2b's exact live loop
# at C 224, arms block 16 / t 0.3 / t 0.2 on S45's 6 prompts, plus REF2b's key-bucket cost) under its own quiet lock,
# with the 30 s memory settle and the usual gates (66 % free: C 224 peaks at ~40.5 GB, S45b). If this session already
# holds a quiet lock when the screen starts, the screen uses it. No time-of-day gate; Kamen's stops come as PAUSE, which
# every gate here honours. No retry. Lines are pre-registered in TABLE_350.md. Log: data/runner_r.log
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
Q=$HOME/.local/share/quiet
export PEER_QUEUE_TIMEOUT=21600
FREE_MIN=66
log() { echo "[runner42] $* $(date '+%H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//'), disk $(df -g / | tail -1 | awk '{print $4}') GB free"; }
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
ok_now() {   # also waits while soft_end.sh ramps held jobs back (it SIGSTOPs their new children too: Fit12, 10-27)
  [ ! -e $Q/PAUSE ] && ! lid_closed && ! grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null \
    && ! pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null \
    && [ "$(memory_pressure 2>/dev/null | awk '/free percentage/ {sub("%", "", $NF); print int($NF)}')" -ge $FREE_MIN ]
}
held() {     # this session holds an unexpired quiet lock
  $PY -I -S -c 'import json, sys, time
d = json.load(open(sys.argv[1]))
sys.exit(0 if d.get("mode") == "quiet" and d.get("until", 0) > time.time() else 1)' "$HOME/.claude-peers/locks/$CLAUDE_PID.json" 2>/dev/null
}
screen() {   # screen <id> <why> <command...>
  id=$1; why=$2; shift 2
  while :; do
    w=0; while ! ok_now; do [ $w -eq 0 ] && log "$id: waiting (PAUSE / ACT / lid / memory >= $FREE_MIN % free)"; w=1; sleep 120; done
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
  log "then: the REF2b merge check (bitwise, ~10 s), then S45c (truncation arms in REF2b's exact loop, C 224)"
  while [ "$1" != 0 ] && kill -0 "$1" 2>/dev/null; do sleep 30; done
  FREE_MIN=58
  screen KBTEST "REF2b merge check: ref2.py's key buckets == the 4c module, bitwise (synthetic SDPA, ~10 s)" \
    $PY scripts/kb_merge_test.py
  tail -3 logs/job.out
  FREE_MIN=66
  screen S45C "S45c: L1-14 truncation (block 16 / t 0.3 / t 0.2) in REF2b's exact live loop at C 224, 6 prompts, full streamed 125B (~7 min)" \
    $PY scripts/r40s3k_screen.py 224 256 speed 16,0.3,0.2
  cat data/s45_speed_c224_s45c.txt
  log "done"
} >> data/runner_r.log 2>&1
