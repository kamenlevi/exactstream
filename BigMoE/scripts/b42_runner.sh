#!/bin/bash
# b42 = R-16 overlap screen (BigMoE Wed 16:05; data/r16_port_price.txt: the W24-ec decode only pays if it overlaps the
# SSD waits): can the GPU run MoonShots' rANS decode beside the 24 GB engine without slowing it? 8 runs A1 B1 B2 A2 A3 B3 B4
# A4 of scripts/s10_anatomy.py 0.25 256 (code / prose / edit, shipped stack incl. MTP_PF_GROUP 48) in one 10-min quiet lock;
# B = A + scripts/r16_bg.py in its own process: real MSB-3 codes decoded on the GPU in a loop, paced to 6 GB/s (~ the
# engine's MSB read rate). PASS: engine tok/s >= -2 % AND each B run's decode >= 5.5 GB/s AND 0 mismatching bytes. KILL:
# < -5 %, decode < 3 GB/s, or any mismatch. Else BETWEEN. (Screens GPU sharing only, not the decode-into-slot integration.)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
set -m   # each run is its own process group: time-up or a lost lock stops the whole tree (RULES)
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/b42"; mkdir -p "$OUT"
LOG="$OUT/b42.log"
PY="$HOME/BigMoE/.venv/bin/python"
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
LOCKF="$HOME/.claude-peers/locks/${CLAUDE_PID}.json"
keepalive() { "$PY" -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
source "$HOME/BigMoE/scripts/ssd_gate.sh"
run() {  # run NAME TIMEOUT OUTFILE CMD...  (memory gate at most 30 s: the lock is only 10 min)
  local name=$1 to=$2 of=$3; shift 3
  keepalive || { say "lost the lock before $name: stopping the batch"; exit 1; }
  while [ -e "$HOME/.local/share/quiet/PAUSE" ]; do sleep 5; keepalive || { say "lock ended during a PAUSE"; exit 1; }; done
  ssd_wait 66
  local i=0
  while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 60 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $i -lt 6 ]; do sleep 5; i=$((i+1)); done
  [ $i -ge 6 ] && say "memory busy at the start of $name (ran anyway after 30 s)"
  say "run $name"
  perl -e 'alarm shift; exec @ARGV' "$to" "$@" >> "$OUT/$of" 2>&1 &
  local jp=$!
  while kill -0 $jp 2>/dev/null; do
    for k in $(seq 36); do kill -0 $jp 2>/dev/null || break; sleep 5; done
    kill -0 $jp 2>/dev/null || break
    keepalive || { say "lost the lock during $name: stopping"; kill -- -$jp; sleep 5; kill -9 -- -$jp 2>/dev/null; exit 1; }
  done
  wait $jp; local rc=$?; kill -9 -- -$jp 2>/dev/null; say "done $name rc=$rc"   # the job's whole group (RULES)
  ssd_check || say "SSD hot after $name"
  return $rc
}
say "b42 start (CLAUDE_PID=$CLAUDE_PID)"
if [ -z "$HAVE_LOCK" ]; then
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 10m "BigMoE b42 (R-16 overlap screen): engine decode alone vs beside a paced GPU rANS decode process (24 GB v3, 8 runs x 3 prompts x 256 tok, ~8 min)" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
fi
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
G="BIGMOE_GDN_EXACT=1"
PG="BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1"
FULL="$G $V3 $PG BIGMOE_EVICT_LFU=0.99 BIGMOE_VERIFY_ROWS_SDPA=1"
J="BIGMOE_JOINT=1"
if lsof 2>/dev/null | grep -q "BigMoE/ext/build/bigmoe_ext"; then export BIGMOE_EXT_BUILD=build_next; B=ext/build_next; else B=ext/build; fi
(cd $B && make 2>&1 | tail -3) >> "$LOG" 2>&1
if [ ! "$B/bigmoe_ext.cpython-312-darwin.so" -nt ext/bigmoe_ext.cpp ]; then   # never run a stale ext
  say "BUILD FAILED in $B: stopping"; [ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1; exit 1; fi
say "ext built in $B"
for R in A1 B1 B2 A2 A3 B3 B4 A4; do
  if [ "${R:0:1}" = "B" ]; then                         # the background decode load, its own process group
    rm -f "$OUT/bg_$R.log.stop"; $PY scripts/r16_bg.py 6 200 "$OUT/bg_$R.log" > "$OUT/bg_$R.out" 2>&1 & BG=$!
    for k in $(seq 60); do grep -q "ready" "$OUT/bg_$R.log" 2>/dev/null && break; sleep 1; done
    grep -q "ready" "$OUT/bg_$R.log" 2>/dev/null || { say "$R: background decode not ready"; kill -- -$BG 2>/dev/null; break; }
  fi
  run "$R" 120 "$R.txt" env $X $FULL $J BIGMOE_MTP_PF_GROUP=48 $PY scripts/s10_anatomy.py 0.25 256 "$OUT/$R.json" || { say "$R failed: stopping the batch"; break; }
  if [ "${R:0:1}" = "B" ]; then touch "$OUT/bg_$R.log.stop"; wait $BG; kill -9 -- -$BG 2>/dev/null; say "$R background: $(tail -1 "$OUT/bg_$R.log")"; fi
done
if ! lsof 2>/dev/null | grep -q "BigMoE/ext/build/bigmoe_ext"; then (cd ext/build && make 2>&1 | tail -1) >> "$LOG" 2>&1; say "ext/build updated"; fi
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "runs done"
AB_LINES=r16 taskpolicy -b $PY scripts/r37_ab.py "$OUT" > "$OUT/ab.txt" 2>&1; say "ab rc=$?"
say "b42 end"
