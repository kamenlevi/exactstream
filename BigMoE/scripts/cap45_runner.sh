#!/bin/bash
# CAP45 (NewIdeas 18:5x: data for Fit12's R-57b-R CPU replay; Kamen 18:45 FAST INTEGRATION: data runs share the GPU): the
# exact live DFlash2 loop at C 192 with the 16-row checks on the gate, S45's six prompts, one run each, BIGMOE_S10B=1 with
# every layer's MoE input -> data/cap45/<prompt>_gate_0.npz (seq, L, rows, ids, x, tokens) + miss.bin (loader flags) +
# cap45_tokens.json. Timing is irrelevant here (shared GPU, rough). Under `peer lock share` (retried while a quiet lock
# refuses it); a 45 GB-route load: waits for >= 50 GB available (vm_stat) and no other process over 20 GB resident.
# usage (the orchestrator 23:2x: 45 GB runs take a quiet ticket first): bash scripts/turn45.sh 37 20 "why" bash scripts/cap45_runner.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
set -m
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/cap45"; mkdir -p "$OUT"
LOG="$OUT/cap45.log"
PY="$HOME/BigMoE/.venv/bin/python"
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }   # NO-SWAP (the orchestrator 23:5x, RULES): free + purgeable + file-backed (speculative sits inside file-backed: the orchestrator 11-14 05:5x)
say "cap45 start (CLAUDE_PID=$CLAUDE_PID)"
mem_ok() { free=$(avail_gb); big=$(ps -axo rss= | awk '$1 > 20000000' | wc -l)
           [ ! -e "$HOME/.local/share/quiet/PAUSE" ] && [ "$free" -ge 50 ] && [ "$big" -eq 0 ]; }
# memory FIRST (a ~36 GB load needs >= 50 GB available and no other big model), so no lock is held while waiting
# (22:41 review: an idle shared reservation shrinks what others see as free); then the shared lock, then memory again
got=0; [ -n "$HAVE_LOCK" ] && { got=1; free=$(avail_gb); }   # under turn45.sh's quiet ticket (the orchestrator 23:2x)
for i in $(seq 360); do                                  # up to 3 h
  [ $got = 1 ] && break
  if mem_ok; then
    if peer lock share 20m --cpu 200 --gpu 60 "BigMoE CAP45: 45 GB live-loop routing + MoE-input capture for Fit12's R-57b replay (data, not timing)" >> "$LOG" 2>&1; then
      mem_ok && { got=1; break; }
      peer unlock >> "$LOG" 2>&1                         # memory went while the lock was taken: give it back, wait again
    fi
  fi
  sleep 30
done
[ $got = 1 ] || { say "no memory + shared lock in 3 h ($free GB available, big $big)"; exit 1; }
say "run capture at C ${CAP45_C:-192} ($free GB available)"
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}'; }
S0=$(swap_gb)
L48=$(echo {0..47} | tr ' ' ',')                         # not `seq -s,`: BSD seq ends with a separator (crash 00:21)
perl -e 'alarm shift; exec @ARGV' 1200 env BIGMOE_MAXK=160 BIGMOE_EXT_BUILD=build_next R57_ARMS=gate R57_PROMPTS=6 R57_REPS=1 \
  R57_TAG=cap45 BIGMOE_S10B=1 BIGMOE_S10B_LAYERS=$L48 BIGMOE_S10B_MISS=$OUT/miss.bin R57_S10B_DIR=$OUT \
  $PY scripts/r57_screen.py ${CAP45_C:-192} 256 >> "$OUT/cap45_run.txt" 2>&1 &   # CAP45_C: the orchestrator 23:3x, C 128 fits ~37 GB
JP=$!
while kill -0 $JP 2>/dev/null; do
  sleep 5
  S=$(swap_gb)
  if awk -v s="$S" -v s0="$S0" 'BEGIN { exit !(s - s0 > 2) }'; then
    say "SWAP GUARD: swap $S GB (start $S0): stopping the capture"; kill -TERM -- -$JP; sleep 5; kill -9 -- -$JP 2>/dev/null; break
  fi
  [ $((SECONDS % 240)) -lt 5 ] && "$PY" -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$HOME/.claude-peers/locks/${CLAUDE_PID}.json"   # lock keepalive (RULES)
done
wait $JP; rc=$?; kill -9 -- -$JP 2>/dev/null
say "capture rc=$rc"
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "cap45 end: $(ls $OUT/*.npz 2>/dev/null | wc -l) npz, $(du -sh $OUT | cut -f1)"
