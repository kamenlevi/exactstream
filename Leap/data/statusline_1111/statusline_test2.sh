#!/bin/bash
# Status line test 2 (Leap 11-11): release/next's serve.py after the lane fix (the idle line keeps the last main/sub answer;
# Claude Code's small "-side" calls show only before one), in the scratch tree on the stand-in engine (mock_tools_sl.py:
# MOCK_THINK=3 s before the first token, MOCK_CC=1 a stand-in cache record), port 8134; Kamen's 8080 and the GPU untouched.
#   1 before any request; 2 a side call alone; 3 a SLOW main answer polled every 0.5 s, a side call queued at t+5 s;
#   4 after both; 5 statusline.sh timed (warm) and against a closed port; 6 /stats.
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
CL=/Users/kamen/Leap/release/cleanclone/cache/clone; N=/Users/kamen/Leap/release/next; S=$A/sltest/clone; PORT=8134; SP=""
cp $N/serve.py $S/serve.py; cmp -s $N/tools/statusline.sh $S/tools/statusline.sh || cp $N/tools/statusline.sh $S/tools/
stopsrv() { [ -z "$SP" ] && return; kill -TERM $SP 2>/dev/null
            for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done
            kill -0 $SP 2>/dev/null && { echo "SIGKILL"; kill -KILL $SP; }
            lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && echo "WARNING: port $PORT still in use"; SP=""; }
trap stopsrv EXIT
( cd $S && MOCK_CC=1 MOCK_THINK=3 MOCK_STORE=$CL/engine/models/bare MOCK_WORKDIR=$A/sltest/work exec .venv/bin/python serve.py --mem 32 --port $PORT --_screen $A/mock_tools_sl.py ) > $A/sltest/serve3.log 2>&1 &
SP=$!; t0=$(date +%s)
until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
  kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 120 ] || { echo "no server: $(tail -3 $A/sltest/serve3.log)"; exit 2; }; sleep 1; done
sl() { curl -s -m 1 http://127.0.0.1:$PORT/statusline; }
req() { curl -s -m 120 http://127.0.0.1:$PORT/v1/messages -H 'content-type: application/json' -H 'x-api-key: local' \
  -d "{\"model\":\"$1\",\"max_tokens\":4000,\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}]}"; }
echo "1 before any request: $(sl)"
req exactstream-side "a short title please" > $A/sltest/side1.json
echo "2 after a side call alone: $(sl)"
req exactstream "SLOW please write a long answer" > $A/sltest/main.json &
CP=$!
for i in $(seq 1 16); do
  [ $i = 10 ] && { req exactstream-side "another title" > $A/sltest/side2.json & SC=$!; }
  echo "3 t+$(echo "$i * 0.5" | bc)s: $(sl)"; sleep 0.5; done
wait $CP; echo "3 main done: $(sl)"
wait $SC; echo "4 after the queued side call too: $(sl)"
ms() { perl -MTime::HiRes=time -e 'printf "%.1f", time*1000'; }
for k in 1 2 3; do a=$(ms); L=$($S/tools/statusline.sh $PORT); b=$(ms); echo "5 statusline.sh $PORT: $(echo "$b - $a" | bc) ms '$L'"; done
a=$(ms); L=$($S/tools/statusline.sh 8199); rc=$?; b=$(ms); echo "5 statusline.sh 8199 (closed): $(echo "$b - $a" | bc) ms rc $rc '$L'"
echo "6 /stats: $(curl -s -m 1 http://127.0.0.1:$PORT/stats)"
grep "done:" $A/sltest/serve3.log | cut -c1-150
stopsrv; echo "mock serve.py stopped"
