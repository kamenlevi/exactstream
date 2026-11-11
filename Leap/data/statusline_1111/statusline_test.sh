#!/bin/bash
# Test of the status line (Leap 11-11, Kamen's ask): release/next's new serve.py (/stats, /statusline), tools/statusline.sh
# and claude_code.sh, in a scratch tree that symlinks the cache clone's files except serve.py and tools/ (the clone itself
# and Kamen's server on 8080 untouched). The stand-in engine (mock_tools.py; no GPU) on port 8134:
#   1 /statusline before any request; 2 a slow request ("SLOW": 1,500 words at ~60 tokens/s) while /statusline is polled
#   every 0.5 s; 3 after it; 4 statusline.sh against the live server and a closed port, timed; 5 /stats JSON.
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
CL=/Users/kamen/Leap/release/cleanclone/cache/clone; N=/Users/kamen/Leap/release/next; S=$A/sltest/clone; PORT=8134
rm -rf $A/sltest; mkdir -p $S/tools $A/sltest/work
for f in $CL/* $CL/.venv; do b=$(basename $f); [ "$b" = serve.py ] || [ "$b" = tools ] || ln -s $f $S/$b; done
for f in $CL/tools/*; do ln -s $f $S/tools/$(basename $f); done
rm -f $S/tools/statusline.sh $S/tools/claude_code.sh
cp $N/serve.py $S/serve.py; cp $N/tools/statusline.sh $N/tools/claude_code.sh $S/tools/
SP=""
stopsrv() { [ -z "$SP" ] && return; kill -TERM $SP 2>/dev/null
            for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done
            kill -0 $SP 2>/dev/null && { echo "serve.py did not stop in 60 s: SIGKILL"; kill -KILL $SP; }
            lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && echo "WARNING: port $PORT still in use"; SP=""; }
trap stopsrv EXIT
( cd $S && MOCK_CC=1 MOCK_THINK=3 MOCK_STORE=$CL/engine/models/bare MOCK_WORKDIR=$A/sltest/work exec .venv/bin/python serve.py --mem 32 --port $PORT --_screen $A/mock_tools_sl.py ) > $A/sltest/serve.log 2>&1 &
SP=$!; t0=$(date +%s)
until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
  kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 120 ] || { echo "mock serve.py did not come up: $(tail -3 $A/sltest/serve.log)"; exit 2; }; sleep 1
done
echo "1 before any request: $(curl -s -m 1 http://127.0.0.1:$PORT/statusline)"
curl -s -m 120 http://127.0.0.1:$PORT/v1/messages -H 'content-type: application/json' -H 'x-api-key: local' \
  -d '{"model":"exactstream","max_tokens":4000,"messages":[{"role":"user","content":"SLOW please write a long answer"}]}' > $A/sltest/answer.json &
CP=$!
for i in $(seq 1 14); do echo "2 t+$(echo "$i * 0.5" | bc)s: $(curl -s -m 1 http://127.0.0.1:$PORT/statusline)"; sleep 0.5; done
wait $CP
echo "3 after: $(curl -s -m 1 http://127.0.0.1:$PORT/statusline)"
s0=$(perl -MTime::HiRes=time -e 'printf "%.4f", time'); L=$($S/tools/statusline.sh $PORT); s1=$(perl -MTime::HiRes=time -e 'printf "%.4f", time')
echo "4 statusline.sh $PORT -> '$L' in $(echo "($s1 - $s0) * 1000" | bc | cut -d. -f1) ms"
s0=$(perl -MTime::HiRes=time -e 'printf "%.4f", time'); L=$($S/tools/statusline.sh 8199); rc=$?; s1=$(perl -MTime::HiRes=time -e 'printf "%.4f", time')
echo "4 statusline.sh 8199 (nothing there) -> '$L' rc $rc in $(echo "($s1 - $s0) * 1000" | bc | cut -d. -f1) ms"
echo "5 /stats: $(curl -s -m 1 http://127.0.0.1:$PORT/stats)"
echo "answer: $(head -c 160 $A/sltest/answer.json)"
grep "done:" $A/sltest/serve.log | tail -1 | cut -c1-200
stopsrv; echo "mock serve.py stopped"
