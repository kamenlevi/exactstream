#!/bin/bash
# Tonight's chain under ONE quiet ticket (Breakthrough 11-03 19:2x: separate tickets leave gaps in which a FLOOR step can
# start and hold NO-SWAP under the next turn's 33 GB pre-queue gate). Queued right after Breakthrough's 00:30 map prefix
# (turn45 AFTER=after_prefix.sh, no WAITMEM); inside the lock it waits for NO-SWAP >= 33 GB (C 160's peak ~29.5 + MG 3 + 0.5),
# keeping the lock alive, until 01:45 at the latest (then it gives the night to FLOOR); then, in order, with start cutoffs:
#   the wall-1 sprint (sprint_s160.sh, by 02:30) -> U3 at C 160 (by 03:15) -> the combination at C 160 without its
#   measurement-2 block (by 03:30; no new arm after 05:00) -> mission 20 (by 04:15)
# usage: inside turn45 (HAVE_LOCK=1 KEEP_LOCK=1); CLAUDE_PID set
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
LOG="$HOME/BigMoE/data/s160/night.log"; mkdir -p data/s160
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY="$HOME/BigMoE/.venv/bin/python"
LOCKF="$HOME/.claude-peers/locks/${CLAUDE_PID}.json"
keepalive() { "$PY" -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
before() { local t; t=$(date +%H%M); [ "$t" -ge 1200 ] || [ "$t" -lt "$1" ]; }   # an evening start (22:xx) is always before an after-midnight cutoff
say "night chain CONTINUATION start (pid $$, HAVE_LOCK=$HAVE_LOCK, NO-SWAP $(noswap) GB)"
while [ "$(noswap)" -lt 33 ]; do
  before 0145 || { say "NO-SWAP $(noswap) GB < 33 at 01:45 (continuation): giving the night to FLOOR"; exit 3; }
  keepalive || { say "lost the lock while waiting for memory"; exit 1; }
  sleep 30
done
say "memory OK (NO-SWAP $(noswap) GB)"
before 0200 && { SC_C=160 SC_MG=3 /bin/bash data/r57/snap/stale_check_sc.sh; say "stale-slot check rc=$? (verdict in data/sc/sc.log)"; }   # FIRST (the orchestrator 22:5x)
keepalive || { say "lock gone after the stale check"; exit 1; }
before 0200 && { S_C=160 S_MG=3 S_DEADLINE=2400 /bin/bash data/r57/snap/sprint_s160.sh; say "sprint rc=$?"; }
keepalive || { say "lock gone after the sprint"; exit 1; }
before 0245 && { U3_C=160 U3_MG=3 U3_DEADLINE=2400 /bin/bash data/r57/snap/u3_smoke_u3.sh; say "U3 rc=$?"; }
keepalive || { say "lock gone after U3"; exit 1; }
before 0315 && { CMB_C=160 CMB_MG=3 CMB_M2=0 CMB_DEADLINE=2400 /bin/bash data/r57/snap/combo_tue_cmb.sh; say "combination rc=$?"; }
keepalive || { say "lock gone after the combination"; exit 1; }
before 0400 && { PW_CS="160 224" PW_MG=3 PW_DEADLINE=2400 /bin/bash data/r57/snap/mission20_pw2_pw.sh; say "mission 20 rc=$?"; }
say "night chain end"
