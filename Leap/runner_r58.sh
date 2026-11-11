#!/bin/sh
# Leap's fifty-eighth detached screen runner (Sun 2026-11-01 07:1x; FAST INTEGRATION: a lead on a SHARED GPU slot, rough
# numbers; a quiet-lock confirmation at C 224 follows if it shows a gain): K-62L at C 160 (REF2-T's lifetime peak ~29.6 GB
# there; 38.0 GB at C 224 per BigMoE's S45-T B run), 3c vs NAX4, ABBA x 4 prompts. Starts only if NO-SWAP memory (free +
# speculative + purgeable + file-backed) >= 37.6 GB (peak + 8); a swap guard stops the job's tree if swap grows > 2 GB.
# Waits for PAUSE / a closed lid / ACT; a share lock is refused during others' quiet locks (retry every 2 min). One run.
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
Q=$HOME/.local/share/quiet
log() { echo "[runner58] $* $(date '+%H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//')"; }
noswap10() { vm_stat | awk '/page size of/ {ps = $8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v = $NF; sub("[.]", "", v); s += v} END {printf "%d", s * ps / 1e8}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
ok_pre() { [ ! -e $Q/PAUSE ] && ! lid_closed && ! grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null && [ "$(noswap10)" -ge 376 ]; }
swap_mb() { sysctl -n vm.swapusage | awk '{for (i = 1; i <= NF; i++) if ($i == "used") {v = $(i + 2); sub("M", "", v); print int(v)}}'; }
tree() { echo "$1"; for c in $(pgrep -P "$1"); do tree "$c"; done; }
{
  log "then: K-62L shared (C 160, the 45 GB default with 3c vs NAX4 tier GLU, ABBA x 4 prompts, CAP 256, rough)"
  while :; do
    w=0; while ! ok_pre; do [ $w -eq 0 ] && log "waiting (PAUSE / lid / ACT / NO-SWAP $(noswap10) / 10 GB < 37.6)"; w=1; sleep 120; done
    if peer lock share 25m --cpu 200 --gpu 90 "Leap K-62L shared: 45 GB default live loop at C 160 (~30 GB), 3c vs NAX4 tier GLU, ABBA x 4 prompts (~15 min, rough)" >/dev/null 2>&1; then
      log "K62L: share lock (NO-SWAP $(noswap10) / 10 GB)"
      SW0=$(swap_mb)
      env BIGMOE_MAXK=160 BIGMOE_EXT_BUILD=build_next R57_TIER=1 R57_TRUNC=0.3 R57_ARMS=gate+t,gate+t+nax R57_PROMPTS=4 \
        R57_WARM=1 R57_TAG=k62l_c160 perl -e 'alarm shift; exec @ARGV' 1500 $PY scripts/k62_live.py 160 256 >> logs/job.out 2>&1 &
      P=$!
      while kill -0 $P 2>/dev/null; do
        sleep 5
        SW=$(swap_mb)
        if [ $((SW - SW0)) -gt 2048 ]; then log "SWAP GUARD: swap $SW MB (start $SW0): stopping the job"; T=$(tree $P); kill $T 2>/dev/null; sleep 20; kill -9 $T 2>/dev/null; break; fi
        if [ -e $Q/PAUSE ] || lid_closed; then log "PAUSE / lid: the job is left to the quiet tools (they SIGSTOP it)"; fi
      done
      wait $P; rc=$?
      peer unlock >/dev/null 2>&1
      log "K62L: rc $rc"
      cat data/k62_live/k62l_c160_screen.txt 2>/dev/null
      break
    fi
    sleep 120
  done
  log "done"
} >> data/runner_r.log 2>&1
