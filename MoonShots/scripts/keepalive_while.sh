#!/bin/sh
# keepalive_while.sh PID: stamp active_at on this session's quiet lock every 60 s while process PID lives (the peer lock lapses
# after ~8 min without tool-call stamps; runners started from a Bash call need this). Stops if the lock is gone or not quiet.
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
while kill -0 "$1" 2>/dev/null; do
  /usr/bin/python3 -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF" || exit 0
  sleep 60
done
