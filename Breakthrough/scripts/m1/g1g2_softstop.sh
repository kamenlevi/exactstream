#!/bin/bash
# G1/G2 map screen: soft stop (Breakthrough, Mon 16 Nov 2026 17:3x; EvalSuite's term in amendment (g)). Run only on
# EvalSuite's word that the orchestrator granted B2 an at-peak start: let the current arm finish, then TERM the map
# process; at the deadline (default 18:20 today) TERM regardless. The frozen runner then unlocks, reads out the finished
# arms and writes G1G2_MAP_END.
# An arm's out_code24/<arm>.json is written once, at the arm's end (dm_common.measure's json.dump), so a TERM during
# that write would truncate it: the stop waits until the new file's size is stable over 3 s and it parses as JSON.
# Usage: bash scripts/m1/g1g2_softstop.sh ["YYYY-mm-dd HH:MM:SS"]    log: data/m1/g1g2_map/softstop.log
O=${SS_O:-$HOME/FableGod/decision_map/out_code24}      # SS_O / SS_PAT / SS_LOG / SS_DT: tests only
PAT=${SS_PAT:-scripts/m1/g1g2_run_ma[p][.]py}
L=${SS_LOG:-$HOME/Breakthrough/data/m1/g1g2_map/softstop.log}
DT=${SS_DT:-5}
DL=$(date -j -f "%Y-%m-%d %H:%M:%S" "${1:-$(date +%Y-%m-%d) 18:20:00}" +%s) || exit 2
ARMS="bt_g1h8 bt_dense3 bt_l20g2 bt_g1h8_res bt_g1 bt_restore bt_l20comb bt_c6comb"
say() { echo "$(date '+%a %H:%M:%S') $*" >> "$L"; }
done_arms() { local a; for a in $ARMS; do [ -f "$O/$a.json" ] && echo "$a"; done; }
complete() {   # $1 = a json file: size stable over 3 s, > 0, and parses
  local s1 s2; s1=$(stat -f %z "$1" 2>/dev/null); sleep 3; s2=$(stat -f %z "$1" 2>/dev/null)
  [ -n "$s1" ] && [ "$s1" = "$s2" ] && [ "$s1" -gt 0 ] && /usr/bin/python3 -I -B -c 'import json, sys; json.load(open(sys.argv[1]))' "$1" 2>/dev/null
}
p=$(pgrep -f "$PAT" | head -1)
[ -z "$p" ] && { say "no map process: nothing to stop"; exit 0; }
before=$(done_arms | tr '\n' ' ')
say "armed (pid $$): map process $p; done before: ${before:-none}; TERM once the next arm's file is complete, or at $(date -r "$DL" '+%H:%M:%S')"
while kill -0 "$p" 2>/dev/null; do
  for a in $(done_arms); do
    case " $before " in *" $a "*) continue ;; esac
    if complete "$O/$a.json"; then say "arm $a finished (file complete): TERM map process $p"; kill -TERM "$p"; exit 0; fi
  done
  if [ "$(date +%s)" -ge "$DL" ]; then say "deadline: TERM map process $p (done: $(done_arms | tr '\n' ' '))"; kill -TERM "$p"; exit 0; fi
  sleep "$DT"
done
say "map process ended on its own"
