#!/bin/bash
# 1.1.0 candidate (pkg_110c) clean-clone check with chat.py and serve.py on the real engine (orchestrator 20:2x): when the Mac
# is free (Kamen idle >= 600 s, no chat.py / serve.py of his, no other lock / engine, NO-SWAP >= 34, no PAUSE / HOLD), else by
# 08:30 tomorrow (before the go). tools/clean_clone_test.sh run with CONFIGS="lossy-24 chat-24", the text prompt on chat-24,
# then chat.py and serve.py --mem 24 (CHAT_SERVE=1). Share lock 'MEM 27 GB'; swap guard +2 GB; 20-min cap.
W=/Users/kamen/Leap/release/cleanclone/v110; LOG=$W/runner.log
LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-11 08:30:00" +%s)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
busy() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*" | grep -cE "^· (benchmarking|waiting for a quiet window)|MEM ([6-9]|[1-9][0-9]) GB"; }
procs() { ps -axo pid=,comm=,args= | awk '$2 ~ /[Pp]ython/ && /r57_screen_|run_map[.]py|fit_onecall[.]py|[ \/](chat|serve)[.]py( |$)/ {printf "%s ", $1}'; }
held() { [ -e ~/.local/share/quiet/PAUSE ] || [ -e ~/.local/share/quiet/HOLD ]; }
desc() { local c; for c in $(pgrep -P "$1"); do desc "$c"; echo "$c"; done; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
LOCKED=0; BP=""
trap '[ -n "$BP" ] && kill -0 $BP 2>/dev/null && { for c in $(desc $BP); do kill -TERM $c 2>/dev/null; done; kill -TERM $BP; }; [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1; say "exit"' EXIT
say "armed (pid $$): waits for a free Mac, last start 11-11 08:30"
CLR=0; NOTE=""
while :; do
  [ "$(date +%s)" -ge $LAST ] && { say "SKIP: no free window by 08:30 (last wait: $NOTE)"; exit 0; }
  why=""
> (a quote from the project owner, kept private)
  [ -z "$why" ] && [ -n "$(procs)" ] && why="an engine / chat / server process runs ($(procs))"
  [ -z "$why" ] && [ "$(busy)" -gt 0 ] && why="another lock is held or queued"
  [ -z "$why" ] && held && why="PAUSE / HOLD"
  [ -z "$why" ] && [ "$(noswap)" -lt 34 ] && why="NO-SWAP $(noswap) GB < 34"
  if [ -z "$why" ]; then CLR=$((CLR + 1)); else CLR=0; fi
  [ -z "$why" ] && [ $CLR -lt 2 ] && why="clear once: one more poll"
  if [ -z "$why" ]; then
    peer lock share 30m --cpu 300 --gpu 100 "MEM 27 GB: Leap 1.1.0 clean-clone check (24 GB configs + chat.py + serve.py), ~6 min" > /dev/null 2>&1 && LOCKED=1
    sleep 20
    [ "$(busy)" = 0 ] && [ -z "$(procs)" ] && break
    say "after the lock: something else started: unlock and wait"; peer unlock >/dev/null 2>&1; LOCKED=0; CLR=0; why="after the lock"
  fi
  [ "$why" != "$NOTE" ] && say "wait: $why"; NOTE=$why
  sleep 60
done
S0=$(swap); say "GO: NO-SWAP $(noswap) GB, swap $S0 MB, idle $(idle) s"
WHEELS=/Users/kamen/Leap/release/cleanclone/wheels CONFIGS="lossy-24 chat-24" TEXT_CONFIG=chat-24 CHAT_SERVE=1 CS_MEM=24 \
  bash $W/clone/tools/clean_clone_test.sh $W/clone $W run >> $LOG 2>&1 &
BP=$!; T0=$(date +%s)
while kill -0 $BP 2>/dev/null; do
  if [ $(( $(swap) - S0 )) -gt 2048 ] || [ $(( $(date +%s) - T0 )) -gt 1200 ]; then
    say "STOP: swap $(swap) MB (start $S0) or 20 min"; for c in $(desc $BP); do kill -TERM $c 2>/dev/null; done; kill -TERM $BP; break
  fi
  sleep 5
done
wait $BP 2>/dev/null; BP=""
say "check ended ($(( $(date +%s) - T0 )) s): $(tail -1 $W/report.txt)"
