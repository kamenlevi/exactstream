#!/bin/bash
# T1 (BigMoE CONV B1) on the real engine: the cache clone (release/cleanclone/cache = release/next's server + BigMoE 7eb3401
# engine/loop/convcache.py + the hook) as serve.py --mem 32 (chat config, PF-A on, DFlash2) with CONV_CACHE_CHECK=1 (every
# hit is checked against a cold prefill: PASS / FAIL in the cache log) and CONV_CACHE_LOG, replaying the 45 recorded requests
# of sessions_real_1110 in order (replay.py). From FROM, when the Mac is free (Kamen idle, no other lock / engine, no
# PAUSE / HOLD, NO-SWAP >= 42, 2 clear polls); last start LAST; share lock; swap +2 GB stops it; the server is stopped and
# checked gone. Usage: t1_cache.sh <run dir> "<FROM>" "<LAST>" [pid ...]
RUN=$1; FROMS=$2; LASTS=$3; shift 3; AFTER="$*"
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
C=/Users/kamen/Leap/release/cleanclone/cache/clone; PORT=8121; NEED=42; mkdir -p $RUN; LOG=$RUN/runner.log
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s; }
FROM=$(ep "$FROMS"); LAST=$(ep "$LASTS")
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
busy() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*" | grep -cE "^· (benchmarking|waiting for a quiet window)|MEM ([6-9]|[1-9][0-9]) GB"; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|run_map[.]py|fit_onecall[.]py|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
held() { [ -e ~/.local/share/quiet/PAUSE ] || [ -e ~/.local/share/quiet/HOLD ]; }
desc() { local c; for c in $(pgrep -P "$1"); do desc "$c"; echo "$c"; done; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
LOCKED=0; SP=""; GUARD=""; TP=""
stopsrv() { [ -z "$SP" ] && return; kill -TERM $SP 2>/dev/null
            for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done
            kill -0 $SP 2>/dev/null && { say "serve.py did not stop in 60 s: SIGKILL"; kill -KILL $SP; }
            lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && say "WARNING: port $PORT still in use"; SP=""; }
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null
            [ -n "$TP" ] && kill -0 $TP 2>/dev/null && kill -TERM $TP
            stopsrv; [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1; touch $RUN/runner.done; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT
trap 'say "stopped by a signal"; exit 1' INT TERM
say "armed (pid $$): T1 conv cache check, 32 GB -> $RUN; after [$AFTER], from $FROMS, last start $LASTS"
CLR=0; NOTE=""
while :; do
  now=$(date +%s); [ $now -ge $LAST ] && { say "SKIP: no free window by $LASTS (last wait: $NOTE)"; exit 0; }
  why=""
  [ $now -lt $FROM ] && why="before $FROMS"
  [ -z "$why" ] && for w in $AFTER; do kill -0 $w 2>/dev/null && why="waits for pid $w"; done
> (a quote from the project owner, kept private)
  [ -z "$why" ] && [ -n "$(procs)" ] && why="an engine / chat / server runs"
  [ -z "$why" ] && [ "$(busy)" -gt 0 ] && why="another lock"
  [ -z "$why" ] && held && why="PAUSE / HOLD"
  [ -z "$why" ] && [ "$(noswap)" -lt $NEED ] && why="NO-SWAP $(noswap) < $NEED"
  if [ -z "$why" ]; then CLR=$((CLR + 1)); else CLR=0; fi
  [ -z "$why" ] && [ $CLR -lt 2 ] && why="clear once"
  if [ -z "$why" ]; then
    peer lock share 60m --cpu 400 --gpu 100 "MEM $NEED GB: Leap + BigMoE T1 conversation-cache check (serve.py --mem 32, CHECK on)" > /dev/null 2>&1 && LOCKED=1
    sleep 20; [ "$(busy)" = 0 ] && [ -z "$(procs)" ] && break
    peer unlock >/dev/null 2>&1; LOCKED=0; CLR=0; why="after the lock"
  fi
  [ "$why" != "$NOTE" ] && say "wait: $why"; NOTE=$why
  sleep 60
done
S0=$(swap); say "GO: NO-SWAP $(noswap) GB, swap $S0 MB"
( while :; do [ $(( $(swap) - S0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) &
GUARD=$!
( cd $C && CONV_CACHE=1 CONV_CACHE_CHECK=1 CONV_CACHE_LOG=$RUN/convcache.log EXACTSTREAM_LOG_REQUESTS=$RUN/requests.jsonl exec .venv/bin/python serve.py --mem 32 --port $PORT ) > $RUN/serve.log 2>&1 &
SP=$!; t0=$(date +%s)
until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
  kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 600 ] || { say "serve.py did not come up (tail: $(tail -2 $RUN/serve.log | tr '\n' ' '))"; exit 2; }; sleep 2
done
say "serve.py ready in $(( $(date +%s) - t0 )) s; $(grep -a 'conversation cache\|installed' $RUN/serve.log $RUN/convcache.log 2>/dev/null | head -2 | tr '\n' ' ' | cut -c1-200)"
/opt/homebrew/opt/python@3.12/bin/python3.12 -I $A/replay.py $PORT $A/sessions_real_1110.jsonl $RUN/replay.jsonl > $RUN/replay.out 2>&1 &
TP=$!; R0=$(date +%s); while kill -0 $TP 2>/dev/null; do [ $(( $(date +%s) - R0 )) -gt 3600 ] && { say "replay: 60-min cap"; kill -TERM $TP; }; sleep 5; done; TP=""
say "replay: $(tail -1 $RUN/replay.out); cache lines: hits $(grep -c ' hit ' $RUN/convcache.log 2>/dev/null), checks PASS $(grep -c 'PASS' $RUN/convcache.log 2>/dev/null), FAIL $(grep -c 'FAIL' $RUN/convcache.log 2>/dev/null)"
say "server: $(grep -c 'done:' $RUN/serve.log) requests; errors $(grep -c -E 'Traceback|Error' $RUN/serve.log)"
stopsrv; say "serve.py stopped (NO-SWAP $(noswap) GB)"
