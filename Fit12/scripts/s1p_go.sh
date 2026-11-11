#!/bin/sh
# Fit12 s1p_go.sh (Mon 9 Nov 23:0x). NewIdeas 23:05: S1p goes last in tonight's chain, only once BigMoE's W24c has ended or
# given up, i.e. a NEW line in /Users/kamen/BigMoE/data/turn45.log (after line N0, counted at arming) with 'W24c' and one of
# 'done rc=', 'past NOTAFTER', 'giving up'. Then this swaps /Users/kamen/Fit12/data/scr/s1p/WAIT -> GO. s1p.sh (S1P_IDLE=1,
# start by arg 1, default Tue 01:30; NewIdeas 23:2x) keeps its own gate. Ends when s1p.sh writes DONE, or at arg 1.
O=/Users/kamen/Fit12/data/scr/s1p; TL=/Users/kamen/BigMoE/data/turn45.log
N0=${2:-$(wc -l < $TL | tr -d ' ')}; TE=$(date -j -f "%Y-%m-%d %H:%M" "${1:-2026-11-10 01:30}" +%s) || exit 9   # (23:1x) arg 1 = the end time (S1p's start-by), arg 2 = the log line to read after
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
log "s1p_go armed: WAIT -> GO on a W24c end line after line $N0 of $TL"
while [ ! -e $O/DONE ] && [ "$(date +%s)" -lt "$TE" ]; do
  if [ -e $O/WAIT ]; then
    l=$(tail -n +$((N0 + 1)) $TL | grep 'W24c' | grep -E 'done rc=|past NOTAFTER|giving up' | head -1)
    if [ -n "$l" ]; then
      rm -f $O/WAIT; touch $O/GO
      log "W24c ended: '$(echo "$l" | cut -c1-160)': WAIT removed, GO set"
      echo "$(date +%H:%M:%S) SCR s1p GO set by s1p_go.sh (W24c ended; NewIdeas 23:05)" >> /Users/kamen/Fit12/logs/summary.txt
    fi
  fi
  sleep 20
done
log "s1p_go ends (DONE: $(cat $O/DONE 2>/dev/null || echo none))"
