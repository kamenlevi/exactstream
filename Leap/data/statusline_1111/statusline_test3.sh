#!/bin/bash
# Status line test 3 (Leap 11-11): the final statusline.sh against (a) the cache clone's current serve.py (no /statusline:
> (a quote from the project owner, kept private)
# (c) a closed port. Stand-in engine (mock_tools_sl.py), port 8134, scratch tree; Kamen's 8080 and the GPU untouched.
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
CL=/Users/kamen/Leap/release/cleanclone/cache/clone; N=/Users/kamen/Leap/release/next; S=$A/sltest/clone; PORT=8134; SP=""
cp $N/tools/statusline.sh $S/tools/statusline.sh.new && mv -f $S/tools/statusline.sh.new $S/tools/statusline.sh
stopsrv() { [ -z "$SP" ] && return; kill -TERM $SP 2>/dev/null
            for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done
            kill -0 $SP 2>/dev/null && { echo "SIGKILL"; kill -KILL $SP; }
            lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && echo "WARNING: port $PORT still in use"; SP=""; }
trap stopsrv EXIT
start() { ( cd $S && MOCK_CC=1 MOCK_THINK=1 MOCK_STORE=$CL/engine/models/bare MOCK_WORKDIR=$A/sltest/work exec .venv/bin/python serve.py --mem 32 --port $PORT --_screen $A/mock_tools_sl.py ) > $A/sltest/serve_$1.log 2>&1 &
          SP=$!; local t0=$(date +%s)
          until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
            kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 120 ] || { echo "no server"; exit 2; }; sleep 1; done; }
req() { curl -s -m 120 http://127.0.0.1:$PORT/v1/messages -H 'content-type: application/json' -H 'x-api-key: local' \
  -d "{\"model\":\"$1\",\"max_tokens\":$3,\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}]}" > /dev/null; }
ms() { perl -MTime::HiRes=time -e 'printf "%.1f", time*1000'; }
run() { local a=$(ms); local L=$($S/tools/statusline.sh $1); local rc=$?; local b=$(ms); echo "$2: '$L' rc $rc, $(echo "$b - $a" | bc) ms"; }
$S/tools/statusline.sh 8199 > /dev/null                   # the new file's one-time first-run check, outside the timings
cp $CL/serve.py $S/serve.py; start old
echo "(a) /statusline status on the clone's serve.py: HTTP $(curl -s -o /dev/null -w '%{http_code}' -m 1 http://127.0.0.1:$PORT/statusline)"
run $PORT "(a) clone's current serve.py"; run $PORT "(a) again"
stopsrv
cp $N/serve.py $S/serve.py; start new
run $PORT "(b) new serve.py, before any request"
req exactstream-side "a short title please" 4000; run $PORT "(b) after a side call"
req exactstream "SLOW long answer" 200; run $PORT "(b) after a main answer"
req exactstream-side "another title" 4000; run $PORT "(b) after another side call"
stopsrv
run 8199 "(c) closed port"
echo "servers stopped; port $PORT: $(lsof -nP -iTCP:$PORT -sTCP:LISTEN 2>/dev/null | wc -l | tr -d ' ') listeners"
