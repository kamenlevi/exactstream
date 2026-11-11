#!/bin/sh
# Leap's fiftieth detached screen runner (Sat 2026-10-31 01:50; the orchestrator's 01:4x NIGHT MODE: "a longer, steadier
# 45 GB number (the same stack on more prompts)"): waits for the pid given as $1 (runner 49, the 5-arm C 224 stack), then
# runs S45-LONG: REF2b's exact live loop at C 224, slot block 16 / gate + t 0.3 / slot block 16 on 12 more prompts (6
# HumanEval, 3 MATH-500, 3 GSM8K, all with EvalSuite REF2 rows), CAP 256, under one 25-min quiet ticket (run_locked 20
# min). QUEUE-FIRST; after the grant a 30 s settle, then NO-SWAP memory (free + speculative + purgeable + file-backed)
# >= 44.5 GB, else release and wait 15 min. The swap guard (+2 GB) stays. One run, no retry. Log: data/runner_r.log
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
Q=$HOME/.local/share/quiet
export PEER_QUEUE_TIMEOUT=21600
AVAIL_MIN=445                                     # tenths of a GB available (vm_stat): C 224's ~40.5 GB lifetime peak + 4 GB (quiet-lock margin, orchestrator 23:3x)
log() { echo "[runner50] $* $(date '+%H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//'), disk $(df -g / | tail -1 | awk '{print $4}') GB free"; }
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
    if held || peer lock quiet --queue 25m "Leap $id: $why"; then
      sleep 30                                       # let the previous holder's memory drain
      if ! ok_pre || [ "$(avail_gb)" -lt $AVAIL_MIN ]; then
        peer unlock; log "$id: after the grant $(avail_gb) / 10 GB available (< $AVAIL_MIN / 10) or PAUSE / lid / ACT: released"
        sleep 900; continue
      fi
      log "$id: lock ($(avail_gb) / 10 GB available)"
      ./run_locked.sh 20 "$@"; rc=$?
      log "$id: rc $rc"
      return $rc
    fi
    log "$id: no lock, queueing again"; sleep 60
  done
}
{
  log "then: S45-LONG (C 224, 12 prompts, slot b16 / gate + t 0.3 / slot b16, CAP 256)"
  while [ "$1" != 0 ] && kill -0 "$1" 2>/dev/null; do sleep 30; done
  screen S45LONG "S45-LONG: REF2b live loop at C 224, slot vs gate + t 0.3 on 12 prompts (every token equal), full streamed 125B (~13 min)" \
    env S45_KEYS=humaneval/HumanEval/101,humaneval/HumanEval/27,humaneval/HumanEval/117,humaneval/HumanEval/4,humaneval/HumanEval/86,humaneval/HumanEval/143,math500/test/intermediate_algebra/1422.json,math500/test/algebra/2232.json,math500/test/prealgebra/1655.json,gsm8k/705,gsm8k/715,gsm8k/724 \
    S45_BUDGET=1080 S45_OUT=s45long_c224.txt $PY scripts/r40s3k_screen.py 224 256 speed 16,g0.3,16 interleave
  cat data/s45long_c224.txt
  log "done"
} >> data/runner_r.log 2>&1
