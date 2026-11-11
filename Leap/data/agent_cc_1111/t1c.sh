#!/bin/bash
# T1c (BigMoE 134e811, conv/convcache.py sha256 e3434b13: my tested 00488b23 + a CHECK-only per-row drift diagnostic for
# signature B, the layer-35 QSA keys after a copy). Same setup as T1b: the cache clone's serve.py --mem 32 (chat-32, PF-A,
# DFlash2), CONV_CACHE_CHECK=1, CONV_CACHE_LOG, the 45 recorded requests of sessions_real_1110 in order (replay.py); first
# BigMoE's stand-in (conv/test_mock.py, MOCK_PFLM=1) on the module. Raw convcache.log and serve.log kept in the run folder.
# Placed (Leap 02:1x) right after the pkg_110g checks (pid 28781), before the 64K / 128K runs: one clear poll, 10-s polls
# (long_ctx needs two 60-s polls); between long_ctx's arms otherwise. Last start 10:00. Share lock; swap +2 GB stops all.
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
C=/Users/kamen/Leap/release/cleanclone/cache/clone
RUN=$A/t1_runs/32_1111c; MK=$A/t1c_mock; mkdir -p $RUN $MK; LOG=$RUN/runner.log
PORT=8121; NEED=44; WAITPID=28781; WANT=e3434b13ffc6dcd3fdb700404b44235c4359c16a9a07eb579bcf2d3f16f7704d
LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-11 10:00:00" +%s)
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
say "armed (pid $$): T1c (BigMoE 134e811) -> $RUN; after pid $WAITPID; last start 10:00"
NOTE=""
while :; do
  now=$(date +%s); [ $now -ge $LAST ] && { say "SKIP: no free window by 10:00 (last wait: $NOTE)"; exit 0; }
  why=""
  kill -0 $WAITPID 2>/dev/null && why="waits for pid $WAITPID (the pkg_110g checks)"
> (a quote from the project owner, kept private)
  [ -z "$why" ] && [ -n "$(procs)" ] && why="an engine / chat / server runs"
  [ -z "$why" ] && [ "$(busy)" -gt 0 ] && why="another lock"
  [ -z "$why" ] && held && why="PAUSE / HOLD"
  [ -z "$why" ] && [ "$(noswap)" -lt $NEED ] && why="NO-SWAP $(noswap) < $NEED"
  if [ -z "$why" ]; then
    peer lock share 40m --cpu 400 --gpu 100 "MEM $NEED GB: Leap + BigMoE T1c (conversation-cache drift diagnostic, serve.py --mem 32), ~15 min" > /dev/null 2>&1 && LOCKED=1
    sleep 2
    [ -z "$(procs)" ] && break
    peer unlock >/dev/null 2>&1; LOCKED=0; why="after the lock"
  fi
  [ "$why" != "$NOTE" ] && say "wait: $why"; NOTE=$why
  sleep 10
done
HAVE=$(shasum -a 256 $C/engine/loop/convcache.py | cut -d' ' -f1)
[ "$HAVE" = "$WANT" ] || { say "STOP: the clone's convcache.py is $HAVE, not BigMoE's 134e811 ($WANT)"; exit 3; }
S0=$(swap); say "GO: NO-SWAP $(noswap) GB, swap $S0 MB; module sha256 $HAVE"
( while :; do [ $(( $(swap) - S0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) &
GUARD=$!

# 0. the stand-in on the module (BigMoE's test_mock.py, run as .../run.py so the other runners' engine gate sees it)
cp -p ~/BigMoE/conv/test_mock.py $MK/run.py; cp -p $C/engine/loop/convcache.py $MK/convcache.py
( cd $MK && PYTHONDONTWRITEBYTECODE=1 CONV_CACHE_CHECK=1 MOCK_PFLM=1 exec ~/BigMoE/.venv/bin/python $MK/run.py $RUN/mock_pflm1.json ) > $RUN/mock_pflm1.out 2>&1 &
TP=$!; waitcap 900; MRC=$?
say "stand-in (MOCK_PFLM=1, 134e811): rc $MRC; $(grep -c 'check PASS' $RUN/mock_pflm1.out) PASS / $(grep -c 'check FAIL' $RUN/mock_pflm1.out) FAIL lines; drift lines $(grep -c 'drift (' $RUN/mock_pflm1.out); tail: $(tail -1 $RUN/mock_pflm1.out | cut -c1-120)"
MOK=1; { [ $MRC != 0 ] || grep -q Traceback $RUN/mock_pflm1.out || ! tail -1 $RUN/mock_pflm1.out | grep -q '^PASS'; } && MOK=0

# 1. T1c
if [ $MOK = 1 ]; then
  ( cd $C && CONV_CACHE=1 CONV_CACHE_CHECK=1 CONV_CACHE_LOG=$RUN/convcache.log EXACTSTREAM_LOG_REQUESTS=$RUN/requests.jsonl exec .venv/bin/python serve.py --mem 32 --port $PORT ) > $RUN/serve.log 2>&1 &
  SP=$!; t0=$(date +%s); UP=1
  until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
    kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 600 ] || { say "T1c: serve.py did not come up (tail: $(tail -2 $RUN/serve.log | tr '\n' ' '))"; UP=0; break; }; sleep 2
  done
  if [ $UP = 1 ]; then
    say "T1c: serve.py ready in $(( $(date +%s) - t0 )) s"
    /opt/homebrew/opt/python@3.12/bin/python3.12 -I $A/replay.py $PORT $A/sessions_real_1110.jsonl $RUN/replay.jsonl > $RUN/replay.out 2>&1 &
    TP=$!; waitcap 3600
    say "T1c: $(tail -1 $RUN/replay.out); hits $(grep -c ' hit ' $RUN/convcache.log 2>/dev/null), PASS $(grep -c 'check PASS' $RUN/convcache.log 2>/dev/null), FAIL $(grep -c 'check FAIL' $RUN/convcache.log 2>/dev/null); drift lines $(grep -c 'drift (' $RUN/convcache.log 2>/dev/null), restore lines $(grep -c 'restore (' $RUN/convcache.log 2>/dev/null); server errors $(grep -c -E 'Traceback|Error' $RUN/serve.log)"
    grep -E 'check FAIL|drift \(|restore \(' $RUN/convcache.log | cut -c1-500 | while read l; do say "  $l"; done
  fi
  stopsrv; say "T1c: serve.py stopped (NO-SWAP $(noswap) GB)"
else
  say "T1c NOT RUN: the stand-in failed on 134e811 (see $RUN/mock_pflm1.out)"
fi
say "done: NO-SWAP $(noswap) GB, swap $(swap) MB (start $S0)"
