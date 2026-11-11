#!/bin/sh
# Leap's fifty-fifth detached screen runner (Sun 2026-11-01 05:2x; night mode): R-61a GDNX: REF2b's exact live loop at C 224
# with the stack (R-55 gate + truncation t 0.3), each prompt as stack / stack + the exact fused GDN pieces (ref2 CFG
# gdn_exact: MoonShots' gdn_exact prework + gated norm, bitwise at S 1-16), A-B-A-B. QUEUE-FIRST; after the grant a 30 s
# settle, then NO-SWAP memory (free + speculative + purgeable + file-backed) >= 44.5 GB, else release and wait 15 min.
# The swap guard (+2 GB) stays. One run, no retry. Log: data/runner_r.log
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
Q=$HOME/.local/share/quiet
export PEER_QUEUE_TIMEOUT=21600
AVAIL_MIN=445                                     # tenths of a GB available (vm_stat): C 224's ~40.5 GB lifetime peak + 4 GB (quiet-lock margin, orchestrator 23:3x)
log() { echo "[runner55] $* $(date '+%H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//'), disk $(df -g / | tail -1 | awk '{print $4}') GB free"; }
avail_gb() {  # GB a new process can get: free + inactive + speculative + purgeable pages (vm_stat)
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
  log "then: R-61a GDNX (C 224, the stack with vs without the exact fused GDN pieces, A-B-A-B, CAP 192)"
  screen R61GDNX "R-61a GDNX at C 224: REF2b live loop, the exact fused GDN pieces on vs off under the R-55 gate x L1-14 truncation (5 runs per prompt, every token equal), full streamed 125B (~9 min)" \
    $PY scripts/r40s3k_screen.py 224 192 speed g0.3,xg0.3,g0.3,xg0.3 interleave
  cp data/s45c2_c224.txt data/r61a_gdnx_c224.txt; cat data/r61a_gdnx_c224.txt
  log "done"
} >> data/runner_r.log 2>&1
