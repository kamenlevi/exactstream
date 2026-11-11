#!/bin/bash
# Idle CPU of the 1.1.0 candidate's serve.py (the orchestrator saw the agent-test server, idle after its last request, at
# ~100 % CPU): serve.py --mem 24, then --mem 32, from the v110 clean clone; one short request, 10 s rest, then CPU time over
# 60 s idle; a 5-s `sample` of the process when idle CPU > 20 %. Each server is stopped (SIGTERM), waited for, and its port
# checked. Gates as the other runners (Kamen idle >= 600 s, no other lock / engine, no PAUSE / HOLD, NO-SWAP >= 40), from
# 00:15, last start 07:00. Share lock 'MEM 36 GB'; swap guard +2 GB.
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent; V=/Users/kamen/Leap/release/cleanclone/v110/clone
LOG=$A/idle_cpu.log; PORT=8098
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s; }
FROM=$(ep "2026-11-11 01:00:00"); LAST=$(ep "2026-11-11 12:00:00")
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
busy() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*" | grep -cE "^· (benchmarking|waiting for a quiet window)|MEM ([6-9]|[1-9][0-9]) GB"; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|run_map[.]py|fit_onecall[.]py|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
held() { [ -e ~/.local/share/quiet/PAUSE ] || [ -e ~/.local/share/quiet/HOLD ]; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
cpus() { ps -o time= -p $1 2>/dev/null | awk -F: '{if (NF == 3) print $1 * 3600 + $2 * 60 + $3; else print $1 * 60 + $2}'; }
LOCKED=0; SP=""; GUARD=""
stopsrv() { [ -z "$SP" ] && return; kill -TERM $SP 2>/dev/null
            for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done
            kill -0 $SP 2>/dev/null && { say "  serve.py did not stop in 60 s: SIGKILL"; kill -KILL $SP; }
            lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && say "  WARNING: port $PORT still in use"; SP=""; }
trap '[ -n "$GUARD" ] && kill $GUARD 2>/dev/null; stopsrv; [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1; touch $A/idle_cpu.done; say "exit (NO-SWAP $(noswap) GB)"' EXIT
trap 'say "stopped by a signal"; exit 1' INT TERM
say "armed (pid $$): from 00:15, last start 07:00"
CLR=0; NOTE=""
while :; do
  now=$(date +%s); [ $now -ge $LAST ] && { say "SKIP: no free window by 07:00 (last wait: $NOTE)"; exit 0; }
  why=""
  [ $now -lt $FROM ] && why="before 00:15"
> (a quote from the project owner, kept private)
  [ -z "$why" ] && [ -n "$(procs)" ] && why="an engine / chat / server / run.py process runs"
  [ -z "$why" ] && [ "$(busy)" -gt 0 ] && why="another lock"
  [ -z "$why" ] && held && why="PAUSE / HOLD"
  [ -z "$why" ] && [ "$(noswap)" -lt 40 ] && why="NO-SWAP $(noswap) < 40"
  if [ -z "$why" ]; then CLR=$((CLR + 1)); else CLR=0; fi
  [ -z "$why" ] && [ $CLR -lt 2 ] && why="clear once"
  if [ -z "$why" ]; then
    peer lock share 15m --cpu 200 --gpu 100 "MEM 36 GB: Leap idle-CPU check of serve.py (24 + 32 GB), ~4 min" > /dev/null 2>&1 && LOCKED=1
    sleep 20; [ "$(busy)" = 0 ] && [ -z "$(procs)" ] && break
    peer unlock >/dev/null 2>&1; LOCKED=0; CLR=0; why="after the lock"
  fi
  [ "$why" != "$NOTE" ] && say "wait: $why"; NOTE=$why
  sleep 60
done
S0=$(swap); say "GO: NO-SWAP $(noswap) GB, swap $S0 MB"
( while :; do [ $(( $(swap) - S0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) &
GUARD=$!
for M in 24 32; do
  ( cd $V && exec .venv/bin/python serve.py --mem $M --port $PORT ) > $A/idle_serve_$M.log 2>&1 &
  SP=$!; t0=$(date +%s)
  until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
    kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 300 ] || { say "$M GB: serve.py did not come up"; stopsrv; continue 2; }; sleep 2
  done
  r=$(curl -s -m 300 http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' \
      -d '{"model": "exactstream", "max_tokens": 16, "messages": [{"role": "user", "content": "Say hello in three words."}]}' | cut -c1-160)
  sleep 10; c0=$(cpus $SP); sleep 60; c1=$(cpus $SP)
  pct=$(awk -v a="$c0" -v b="$c1" 'BEGIN {printf "%.0f", (b - a) / 60 * 100}')
  say "$M GB: ready in $(( $(date +%s) - t0 - 70 )) s; request: $r"
  say "$M GB: idle CPU over 60 s after the request: $pct % of one core (cpu time $c0 -> $c1 s); threads: $(ps -M -p $SP 2>/dev/null | tail -n +2 | wc -l | tr -d ' ')"
  if [ "$pct" -gt 20 ]; then
    sample $SP 5 -file $A/idle_sample_$M.txt >/dev/null 2>&1
    say "$M GB: sample written ($A/idle_sample_$M.txt); busiest thread lines: $(grep -E '^ +[0-9]+ Thread_' $A/idle_sample_$M.txt 2>/dev/null | sort -k1 -n -r | head -3 | tr -s ' ' | tr '\n' ';' | cut -c1-300)"
  fi
  stopsrv; say "$M GB: stopped (NO-SWAP $(noswap) GB)"
done
