#!/bin/sh
# MTP diagnosis run under the session's quiet lock: CPU mapping check of the official bf16 head, then the GPU A/B
# (mtp_ab125.py). Keeps the lock alive by stamping active_at every 3 min (night_runner.sh's keepalive), unlocks at the end.
cd "$HOME/MoonShots" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
OFF=data/mtp_bf16/mtp_official_bf16.safetensors
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
keepalive || { echo "no quiet lock held: not running"; exit 1; }
echo "$(date +%T) check"; $PY scripts/mtp_bf16_check.py $OFF > data/mtp_bf16/check.txt 2>&1; tail -3 data/mtp_bf16/check.txt
keepalive
echo "$(date +%T) A/B"
$PY scripts/mtp_ab125.py $OFF 0.25 "$@" > data/mtp_ab125.txt 2>&1 & G=$!
i=0
while kill -0 $G 2>/dev/null; do
  sleep 5; i=$((i + 5))
  [ $((i % 180)) -eq 0 ] && { keepalive || echo "$(date +%T) lost the quiet lock (run continues)"; }
  [ $i -ge 2400 ] && { echo "$(date +%T) time limit: stopping"; kill $G; sleep 5; kill -9 $G 2>/dev/null; break; }
done
wait $G 2>/dev/null
echo "$(date +%T) done"; tail -25 data/mtp_ab125.txt
peer unlock > /dev/null 2>&1
