#!/bin/bash
# Status-line timing (Leap 11-11): why did statusline.sh take 975 ms once? The mock serve.py on 8134 again (scratch tree from
# statusline_test.sh), idle: 8 calls of raw curl, 8 of statusline.sh; then the same while a SLOW request writes. Stops it.
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
CL=/Users/kamen/Leap/release/cleanclone/cache/clone; S=$A/sltest/clone; PORT=8134; SP=""
stopsrv() { [ -z "$SP" ] && return; kill -TERM $SP 2>/dev/null
            for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done
            kill -0 $SP 2>/dev/null && { echo "SIGKILL"; kill -KILL $SP; }
            lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && echo "WARNING: port $PORT still in use"; SP=""; }
trap stopsrv EXIT
( cd $S && MOCK_CC=1 MOCK_THINK=3 MOCK_STORE=$CL/engine/models/bare MOCK_WORKDIR=$A/sltest/work exec .venv/bin/python serve.py --mem 32 --port $PORT --_screen $A/mock_tools_sl.py ) > $A/sltest/serve2.log 2>&1 &
SP=$!; t0=$(date +%s)
until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
  kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 120 ] || { echo "no server"; exit 2; }; sleep 1; done
ms() { perl -MTime::HiRes=time -e 'printf "%.1f", time*1000'; }
echo "curl: $(which curl) $(curl --version | head -1 | cut -d' ' -f1-2); proxies: $(env | grep -ic proxy)"
for k in 1 2 3 4 5 6 7 8; do a=$(ms); curl -s -m 0.3 -o /dev/null -w '%{time_connect} %{time_starttransfer} %{time_total}' http://127.0.0.1:$PORT/statusline > $A/sltest/w.txt; b=$(ms)
  echo "idle raw curl $k: $(echo "$b - $a" | bc) ms (connect/first byte/total s: $(cat $A/sltest/w.txt))"; done
for k in 1 2 3 4 5 6 7 8; do a=$(ms); L=$($S/tools/statusline.sh $PORT); b=$(ms); echo "idle statusline.sh $k: $(echo "$b - $a" | bc) ms '$L'"; done
curl -s -m 120 http://127.0.0.1:$PORT/v1/messages -H 'content-type: application/json' -H 'x-api-key: local' \
  -d '{"model":"exactstream","max_tokens":4000,"messages":[{"role":"user","content":"SLOW please write a long answer"}]}' > /dev/null &
CP=$!
for k in 1 2 3 4 5 6 7 8 9 10; do a=$(ms); L=$($S/tools/statusline.sh $PORT); b=$(ms); echo "busy statusline.sh $k: $(echo "$b - $a" | bc) ms '$L'"; sleep 0.4; done
wait $CP
for k in 1 2 3; do a=$(ms); L=$($S/tools/statusline.sh $PORT); b=$(ms); echo "after statusline.sh $k: $(echo "$b - $a" | bc) ms '$L'"; done
stopsrv; echo "stopped"
