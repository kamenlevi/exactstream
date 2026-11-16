#!/bin/bash
# lband_go_1116.sh (Breakthrough, Mon 16 Nov 2026 19:5x): writes the GO marker for TONIGHT's band screen (NewIdeas 19:3x:
# after Sandbox98's screen 2, start by ~21:00, end by 22:50; wait for BigMoE's key test or Leap's gpub if they started).
# GO once ALL hold on two reads DT s apart:
#   - Sandbox98's share lock is gone (no live lock whose reason matches GO_SB_PAT);
#   - no other session holds a live quiet lock (Leap's gpub takes one);
#   - BigMoE's key test (GO_KT_PAT) is not running.
# Order guard: if NO-SWAP >= 40 while BigMoE's follower (GO_FOLLOW_PAT) still waits, its gate is open, so it gets GRACE s to
# start its key test first; if it starts, the loop waits for it to end. Gives up (rc 3, no GO) at GO_BY (default 21:00).
# Reads lock files and process lists only; never stops anything. Usage: bash scripts/m1/lband_go_1116.sh GO_PATH
# Sim overrides: GO_LOCKS (lock dir), GO_SB_PAT, GO_KT_PAT, GO_FOLLOW_PAT, GO_NS_CMD, GO_BY (epoch), GO_DT, GO_GRACE, GO_LOG.
GO=${1:?usage: lband_go_1116.sh GO_PATH}
LK=${GO_LOCKS:-$HOME/.claude-peers/locks}
SBPAT=${GO_SB_PAT:-^Sandbox98}
KTPAT=${GO_KT_PAT:-scripts/g1_keytes[t][.]py}
FWPAT=${GO_FOLLOW_PAT:-scripts/g1_follo[w][.]sh}
BY=${GO_BY:-$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-16 21:00:00" +%s)}
DT=${GO_DT:-30}
GRACE=${GO_GRACE:-480}
LOG=${GO_LOG:-$(dirname "$GO")/go.log}
say() { echo "$(date '+%a %H:%M:%S') $*" >> "$LOG"; }
ns() {
  if [ -n "${GO_NS_CMD:-}" ]; then eval "$GO_NS_CMD"; return; fi
  vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'
}
locks() {   # one "mode|reason" line per live lock file (until > now); a lapsed-but-unexpired lock counts as held
  /usr/bin/python3 -I -B -c '
import glob, json, os, sys, time
for f in glob.glob(os.path.join(sys.argv[1], "*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if d.get("until", 0) > time.time():
        print("%s|%s" % (d.get("mode"), d.get("why", "")))' "$LK"
}
sb_live() { locks | cut -d'|' -f2- | grep -q -E "$SBPAT"; }
quiet_live() { locks | grep -q '^quiet|'; }
kt_up() { pgrep -f -- "$KTPAT" > /dev/null; }
fw_up() { pgrep -f -- "$FWPAT" > /dev/null; }
clear_now() { ! sb_live && ! quiet_live && ! kt_up; }
yn() { "$@" && echo y || echo n; }
say "armed (pid $$): GO -> $GO once Sandbox98's lock is gone, no live quiet lock and no key test (two reads $DT s apart); if NO-SWAP >= 40 while BigMoE's follower waits, it gets ${GRACE} s to start first; gives up $(date -r "$BY" '+%a %H:%M')"
G0=""; SAWKT=0
while :; do
  if [ "$(date +%s)" -ge "$BY" ]; then
    say "GIVE UP: no GO by $(date -r "$BY" '+%H:%M') (Sandbox98 lock $(yn sb_live), quiet lock $(yn quiet_live), key test $(yn kt_up), NO-SWAP $(ns) GB)"
    exit 3
  fi
  if kt_up && [ $SAWKT = 0 ]; then SAWKT=1; say "BigMoE's key test is running: wait for its end"; fi
  if clear_now; then
    sleep "$DT"
    if clear_now; then
      a=$(ns)
      if [ $SAWKT = 0 ] && fw_up && [ "$a" -ge 40 ]; then   # once its key test has run, it has had its turn
        [ -z "$G0" ] && { G0=$(date +%s); say "NO-SWAP $a GB >= 40 while BigMoE's follower waits: it gets ${GRACE} s to start its key test"; }
        if [ $(( $(date +%s) - G0 )) -lt "$GRACE" ]; then sleep "$DT"; continue; fi
        say "BigMoE's key test did not start within ${GRACE} s"
      fi
      echo "LBAND GO $(date '+%m-%d %H:%M:%S'): Sandbox98 lock gone, no quiet lock, no key test (NO-SWAP $a GB; follower waiting $(yn fw_up); key test seen $SAWKT)" > "$GO"
      say "GO written (NO-SWAP $a GB)"
      exit 0
    fi
  fi
  sleep "$DT"
done
