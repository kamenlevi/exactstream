#!/bin/bash
# Keep a running runner 61 from starting GPU work during the quiet PAUSE (a heat pause or a day hold). Its loop checks PAUSE
# before each chunk, but a decode client that already waits for its share lock does not, and the lock is granted when
# another session's quiet lock ends, which a heat pause does (11-08 18:32: a chunk started 30 s into a PAUSE).
# Every 5 s while the runner lives and PAUSE exists:
#   - a decode client with no stage child yet (still waiting for the lock): stop it;
#   - a stage younger than 120 s (still loading; no row yet): stop it and its client, then peer unlock.
# A stage that already decodes rows is left alone: fanwatch freezes it, and a quiet ticket stops it at its next row.
# The runner sees the stopped chunk as rc 143 with no new row, then waits in its own PAUSE check.
#   card_pause_guard.sh <runner pid>
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
R=$1; P=$HOME/.local/share/quiet/PAUSE; L=results/day61_ref_runner.log
log() { echo "$(date '+%H:%M:%S') $*" | tee -a "$L"; }
kids() { ps -A -o pid=,ppid= | awk -v p="$1" '$2 == p {print $1}'; }
age_s() { ps -o etime= -p "$1" 2>/dev/null | awk -F'[-:]' '{s = 0; for (i = 1; i <= NF; i++) s = s * 60 + $i; if (NF == 4) s = $1 * 86400 + $2 * 3600 + $3 * 60 + $4; print s + 0}'; }
log "pause guard (pid $$) on runner $R: no decode starts while PAUSE exists"
while kill -0 "$R" 2>/dev/null; do
  if [ -e "$P" ]; then
    for sh in $(kids "$R"); do
      for cli in $(kids "$sh"); do
        ps -o command= -p "$cli" 2>/dev/null | grep -q "evalsuite.cli import main.*gpu decode" || continue
        st=$(kids "$cli" | head -1)
        if [ -z "$st" ]; then
          kill "$cli" 2>/dev/null && log "pause guard: PAUSE: stopped decode client $cli (it was waiting for its lock)"
        elif [ "$(age_s "$st")" -lt 120 ]; then
          kill "$st" "$cli" 2>/dev/null && log "pause guard: PAUSE: stopped a loading stage $st ($(age_s "$st") s) and client $cli"
          sleep 2; peer unlock > /dev/null 2>&1
        fi
      done
    done
  fi
  sleep 5
done
log "pause guard: runner $R ended; guard ends"
