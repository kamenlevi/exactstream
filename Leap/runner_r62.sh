#!/bin/sh
# Leap's sixty-second detached runner (Sun 2026-11-01; NewIdeas 12:4x: K-80 GO tonight, FIFTH in the quiet line, after
# BigMoE C45 -> D-23c -> MoonShots M-16 -> EvalSuite runner 36; plus the second stacked every-token check). From 23:10 it
# waits until no other session holds or waits for a quiet lock for 20 minutes in a row (the line has drained, so it goes
# last), then ONE quiet ticket (60 min) for three processes at C 160 on the 45 GB default (R57_PRESET=45):
#   1. k80_diag: the first-check excess diagnostic (Breakthrough FIXED-PART): warm-up, agent/bash18, agent/text23,
#      agent/bash18 again (CAP 64), the per-check split of the first 8 checks (scripts/k80_diag.py);
#   2. k83_A: the default on S45's six prompts (CAP 256), every token vs REF2-T's rows (the timing control);
#   3. k83_K: + K73_EXACT=1 K83_EXACT=1 = both exact stacks (R-70/72/73/74 + R-76/77/78/82/83), every token vs REF2-T.
# No start after 09:00 (big runs belong to the night); PAUSE / HOLD / lid / ACT respected; NO-SWAP >= 41 GB and no other
# process over 8 GB before each process; swap guard +2 GB.
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
Q=$HOME/.local/share/quiet
log() { echo "[runner62] $* $(date '+%H:%M:%S'); swap $(sysctl -n vm.swapusage | cut -d'(' -f1 | sed 's/ *$//')"; }
noswap10() { vm_stat | awk '/page size of/ {ps = $8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v = $NF; sub("[.]", "", v); s += v} END {printf "%d", s * ps / 1e8}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
bigproc() { ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me {n++} END {print n + 0}'; }
others_quiet() { peer ls 2>/dev/null | grep -v '^You:' | grep -c -E 'waiting for a quiet window|quiet lock until'; }
hm() { date +%H%M | sed 's/^0*//'; }
too_late() { t=$(hm); [ -n "$t" ] && [ "$t" -ge 900 ] && [ "$t" -lt 2300 ]; }
ok_pre() { [ ! -e $Q/PAUSE ] && [ ! -e $Q/HOLD ] && ! lid_closed && ! grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null \
           && [ "$(noswap10)" -ge 410 ] && [ "$(bigproc)" -eq 0 ]; }
swap_mb() { sysctl -n vm.swapusage | awk '{for (i = 1; i <= NF; i++) if ($i == "used") {v = $(i + 2); sub("M", "", v); print int(v)}}'; }
tree() { echo "$1"; for c in $(pgrep -P "$1"); do tree "$c"; done; }
G="R57_PRESET=45 BIGMOE_EXT_BUILD=build_next R57_REPS=1 R57_WARM=1"
REF="R57_REFTOK=$HOME/BigMoE/data/r57/s45t_B_tokens.json:gate+t"
one() {  # one <tag> <script> <cap> <alarm s> <extra env...>
  tag=$1; script=$2; cap=$3; al=$4; shift 4
  w=0; while ! ok_pre; do [ $w -eq 0 ] && log "$tag: waiting (PAUSE / HOLD / lid / ACT / NO-SWAP $(noswap10) / 10 GB < 41 / big procs $(bigproc))"; w=1; sleep 60; done
  log "$tag: start (NO-SWAP $(noswap10) / 10 GB)"
  SW0=$(swap_mb)
  env $G R57_TAG=$tag "$@" perl -e 'alarm shift; exec @ARGV' $al $PY scripts/$script 160 $cap >> logs/job.out 2>&1 &
  P=$!
  while kill -0 $P 2>/dev/null; do
    sleep 5
    SW=$(swap_mb)
    if [ $((SW - SW0)) -gt 2048 ]; then log "SWAP GUARD: swap $SW MB (start $SW0): stopping $tag"; T=$(tree $P); kill $T 2>/dev/null; sleep 20; kill -9 $T 2>/dev/null; break; fi
  done
  wait $P; log "$tag: rc $?"
}
{
  log "queued: K-80 diagnostic + K-83 stacked every-token check (C 160, 3 processes, one quiet ticket after the night line drains)"
  while [ "$(hm)" -lt 2310 ] && [ "$(hm)" -ge 900 ]; do sleep 120; done
  quiet=0
  while [ $quiet -lt 20 ]; do                           # 20 minutes in a row without another session's quiet lock / ticket
    if too_late; then log "not started by 09:00: giving up (night only)"; exit 0; fi
    if [ "$(others_quiet)" -eq 0 ]; then quiet=$((quiet + 1)); else quiet=0; fi
    sleep 60
  done
  log "the quiet line has drained: queueing the ticket"
  until PEER_QUEUE_TIMEOUT=28800 peer lock quiet --queue 60m "Leap K-80 first-check diagnostic (agent prompts) + K-83 stacked every-token check vs REF2-T, C 160 (~33 GB), 3 processes ~45 min" >/dev/null 2>&1; do
    log "quiet ticket lapsed or refused: re-queueing"; sleep 60
  done
  log "quiet lock"
  one k80_diag k80_diag.py 64 1500 K80_PICK=agent:agent/bash18,agent:agent/text23,agent:agent/bash18 K80_HARD=1300
  one k83_A k73_live.py 256 900 R57_PROMPTS=6 $REF
  one k83_K k73_live.py 256 900 R57_PROMPTS=6 $REF K73_EXACT=1 K83_EXACT=1
  peer unlock >/dev/null 2>&1
  grep -E "K-80|warm-up" data/k80/k80_diag_screen.txt
  for t in k83_A k83_K; do echo "== $t"; grep -E "K-73|K-83|\[gate\+t\]|gate\+t:|tier flags|lifetime" data/r73l/${t}_screen.txt; done
  log "done"
} >> data/runner_r.log 2>&1
