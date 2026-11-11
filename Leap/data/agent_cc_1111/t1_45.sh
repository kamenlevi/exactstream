#!/bin/bash
# T1-45: the conversation-cache exactness replay at 45 GB, before any cached 45 GB agent run (Leap 05:2x). Module: BigMoE
# 3c068d7 (sha256 55631cb8, bitwise clones; T1e PASS 12/12 at 32 GB), branches ON; serve.py = cachepin (d6b44ed0); both =
# release/next 80c8041 and already in the cache clone (re-installed and sha-checked at GO). serve.py --mem 45 (chat-45:
# PF-D, no PF-A, so the module restores chunk-major), CONV_CACHE_GB=2.5 (memory: NO-SWAP ~51-52 at night),
# CONV_CACHE_CHECK=1, the 45 recorded requests (replay.py). Stand-in gate: MOCK_PFLM=0 (chunk-major); MOCK_PFLM=1 info.
# Verdict: BigMoE's t1_verdict.py (3c068d7) -> <run>/verdict.json. Starts after the cached 32 GB run's runner (pid
# WAITPID) ends: one clear poll, 10-s polls; NO-SWAP >= 50; last start 11:30 (the cached 45 GB run must end by 13:45).
# Share lock; swap +2 GB stops all; serve.py stopped and the port checked.
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
C=/Users/kamen/Leap/release/cleanclone/cache/clone
RUN=$A/t1_runs/45_1111f; MK=$A/t1f_mock; mkdir -p $RUN $MK; LOG=$RUN/runner.log
PORT=8125; NEED=50; WAITPID=${1:-60586}; COMMIT=3c068d7; WANT=55631cb8e1760878d58567a576bc63c9555831d2b15d2b6a456e89471c368f1a
PIN=$A/cachepin/serve.py; SERVE_WANT=d6b44ed034dbcf7b
LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-11 11:30:00" +%s)
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
stoptp() { [ -n "$TP" ] && kill -0 $TP 2>/dev/null && { for c in $(desc $TP); do kill -TERM $c 2>/dev/null; done; kill -TERM $TP; }; TP=""; }
waitcap() { local cap=$1 t0=$(date +%s) p=$TP rc; [ -z "$p" ] && return 1
            while kill -0 $p 2>/dev/null; do [ $(( $(date +%s) - t0 )) -gt $cap ] && { say "  cap $cap s: stopping"; stoptp; break; }; sleep 3; done
            wait $p 2>/dev/null; rc=$?; TP=""; return $rc; }
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null; stoptp; stopsrv
            [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1; touch $RUN/runner.done; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT
trap 'say "stopped by a signal"; exit 1' INT TERM
say "armed (pid $$): T1-45 (BigMoE 3c068d7, branches on, --mem 45, CONV_CACHE_GB 2.5) -> $RUN; after pid $WAITPID (the cached 32 GB run); last start 11:30"
NOTE=""
while :; do
  now=$(date +%s); [ $now -ge $LAST ] && { say "SKIP: no free window by 11:30 (last wait: $NOTE)"; exit 0; }
  why=""
  kill -0 $WAITPID 2>/dev/null && why="waits for pid $WAITPID (the cached 32 GB run)"
> (a quote from the project owner, kept private)
  [ -z "$why" ] && [ -n "$(procs)" ] && why="an engine / chat / server runs"
  [ -z "$why" ] && [ "$(busy)" -gt 0 ] && why="another lock"
  [ -z "$why" ] && held && why="PAUSE / HOLD"
  [ -z "$why" ] && [ "$(noswap)" -lt $NEED ] && why="NO-SWAP $(noswap) < $NEED"
  if [ -z "$why" ]; then
    peer lock share 40m --cpu 400 --gpu 100 "MEM $NEED GB: Leap + BigMoE T1-45 (conversation cache 3c068d7 at 45 GB, serve.py --mem 45), ~15 min" > /dev/null 2>&1 && LOCKED=1
    sleep 2
    [ -z "$(procs)" ] && break
    peer unlock >/dev/null 2>&1; LOCKED=0; why="after the lock"
  fi
  [ "$why" != "$NOTE" ] && say "wait: $why"; NOTE=$why
  sleep 10
done
for f in convcache.py test_mock.py t1_verdict.py; do git -C ~/BigMoE show $COMMIT:conv/$f > $RUN/${f%.py}_$COMMIT.py; done
[ "$(shasum -a 256 $RUN/convcache_$COMMIT.py | cut -d' ' -f1)" = "$WANT" ] || { say "STOP: BigMoE $COMMIT's convcache.py is not $WANT"; exit 3; }
[ "$(shasum -a 256 $PIN | cut -c1-16)" = "$SERVE_WANT" ] || { say "STOP: cachepin/serve.py is not $SERVE_WANT"; exit 3; }
cp -p $C/engine/loop/convcache.py $C/engine/loop/convcache.py.before_t1_45; cp $RUN/convcache_$COMMIT.py $C/engine/loop/convcache.py
[ -e $C/serve.py.b5a46d4b ] || cp -p $C/serve.py $C/serve.py.b5a46d4b; cp -p $PIN $C/serve.py
HAVE=$(shasum -a 256 $C/engine/loop/convcache.py | cut -d' ' -f1)
[ "$HAVE" = "$WANT" ] || { say "STOP: the clone's convcache.py is $HAVE, not BigMoE's $COMMIT ($WANT)"; exit 3; }
[ "$(shasum -a 256 $C/serve.py | cut -c1-16)" = "$SERVE_WANT" ] || { say "STOP: the clone's serve.py is not cachepin's"; exit 3; }
S0=$(swap); say "GO: NO-SWAP $(noswap) GB, swap $S0 MB; module sha256 $HAVE; serve.py $SERVE_WANT (cachepin)"
( while :; do [ $(( $(swap) - S0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) &
GUARD=$!

# 0. the stand-in on the module (BigMoE's test_mock.py, run as .../run.py so the other runners' engine gate sees it)
cp $RUN/test_mock_$COMMIT.py $MK/run.py; cp -p $C/engine/loop/convcache.py $MK/convcache.py
( cd $MK && PYTHONDONTWRITEBYTECODE=1 CONV_CACHE_CHECK=1 MOCK_PFLM=0 exec ~/BigMoE/.venv/bin/python $MK/run.py $RUN/mock_pflm0.json ) > $RUN/mock_pflm0.out 2>&1 &
TP=$!; waitcap 900; MRC=$?
say "stand-in (MOCK_PFLM=0 = chunk-major, the gate, 3c068d7): rc $MRC; $(grep -c 'check PASS' $RUN/mock_pflm0.out) PASS / $(grep -c 'check FAIL' $RUN/mock_pflm0.out) FAIL lines; tail: $(tail -1 $RUN/mock_pflm0.out | cut -c1-120)"
MOK=1; { [ $MRC != 0 ] || grep -q Traceback $RUN/mock_pflm0.out || ! tail -1 $RUN/mock_pflm0.out | grep -q '^PASS'; } && MOK=0
( cd $MK && PYTHONDONTWRITEBYTECODE=1 CONV_CACHE_CHECK=1 MOCK_PFLM=1 exec ~/BigMoE/.venv/bin/python $MK/run.py $RUN/mock_pflm1.json ) > $RUN/mock_pflm1.out 2>&1 &
TP=$!; waitcap 900; say "stand-in (MOCK_PFLM=1, info): rc $?; $(grep -c 'check PASS' $RUN/mock_pflm1.out) PASS / $(grep -c 'check FAIL' $RUN/mock_pflm1.out) FAIL lines; tail: $(tail -1 $RUN/mock_pflm1.out | cut -c1-100)"

# 1. T1c
if [ $MOK = 1 ]; then
  ( cd $C && CONV_CACHE=1 CONV_CACHE_GB=2.5 CONV_CACHE_CHECK=1 CONV_CACHE_LOG=$RUN/convcache.log EXACTSTREAM_LOG_REQUESTS=$RUN/requests.jsonl exec .venv/bin/python serve.py --mem 45 --port $PORT ) > $RUN/serve.log 2>&1 &
  SP=$!; t0=$(date +%s); UP=1
  until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
    kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 600 ] || { say "T1-45: serve.py did not come up (tail: $(tail -2 $RUN/serve.log | tr '\n' ' '))"; UP=0; break; }; sleep 2
  done
  if [ $UP = 1 ]; then
    say "T1-45: serve.py ready in $(( $(date +%s) - t0 )) s"
    /opt/homebrew/opt/python@3.12/bin/python3.12 -I $A/replay.py $PORT $A/sessions_real_1110.jsonl $RUN/replay.jsonl > $RUN/replay.out 2>&1 &
    TP=$!; waitcap 3600
    say "T1-45: $(tail -1 $RUN/replay.out); hits $(grep -c ' hit ' $RUN/convcache.log 2>/dev/null), PASS $(grep -c 'check PASS' $RUN/convcache.log 2>/dev/null), FAIL $(grep -c 'check FAIL' $RUN/convcache.log 2>/dev/null); drift lines $(grep -c 'drift (' $RUN/convcache.log 2>/dev/null), restore lines $(grep -c 'restore (' $RUN/convcache.log 2>/dev/null); server errors $(grep -c -E 'Traceback|Error' $RUN/serve.log)"
    grep -E 'check FAIL|drift \(|restore \(|census \(' $RUN/convcache.log | cut -c1-500 | while read l; do say "  $l"; done
    /usr/bin/python3 -I $RUN/t1_verdict_$COMMIT.py $RUN $RUN/verdict.json > $RUN/verdict.out 2>&1; say "T1-45 verdict (rc $?): $(tail -3 $RUN/verdict.out | tr '\n' ' ' | cut -c1-300)"
  fi
  stopsrv; say "T1-45: serve.py stopped (NO-SWAP $(noswap) GB)"
else
  say "T1-45 NOT RUN: the stand-in failed on 3c068d7 (see $RUN/mock_pflm1.out)"
fi
say "done: NO-SWAP $(noswap) GB, swap $(swap) MB (start $S0)"
