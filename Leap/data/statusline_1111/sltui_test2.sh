#!/bin/bash
# Does Claude Code itself show the status line? (Leap 11-11) The interactive Claude Code (2.1.322, CLAUDE_BIN) started by
# release/next's tools/claude_code.sh, from a launcher-only tree (tools/ copies; no .venv, no serve.py: it cannot start an
# engine), against the stand-in serve.py on 8134 (scratch tree, mock engine, no GPU), in a pty (script) for ~14 s, with a
# throwaway config folder (onboarding marked done, the throwaway folder trusted); no prompt is typed, so no tool runs.
# v2: a SLOW request from outside at t+4 s (no Claude Code event): the line must tick by refreshInterval alone.
# The screen is saved, stripped of escapes, and searched for the line. Everything is stopped by pid at the end.
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
CL=/Users/kamen/Leap/release/cleanclone/cache/clone; N=/Users/kamen/Leap/release/next; S=$A/sltest/clone; PORT=8134
T=$A/sltui; rm -rf $T; mkdir -p $T/clone/tools $T/cfg $T/work
cp $N/tools/claude_code.sh $N/tools/claude_local.sh $N/tools/statusline.sh $T/clone/tools/
cp $N/serve.py $S/serve.py; cmp -s $N/tools/statusline.sh $S/tools/statusline.sh || cp $N/tools/statusline.sh $S/tools/
python3 -c "import json,sys; json.dump({'hasCompletedOnboarding': True, 'theme': 'dark', 'projects': {sys.argv[1]: {'hasTrustDialogAccepted': True, 'hasCompletedProjectOnboarding': True}}}, open(sys.argv[2], 'w'))" "$T/work" "$T/cfg/.claude.json"
SP=""; CP=""
stopall() { [ -n "$CP" ] && { for p in $(pgrep -P $CP) $CP; do for q in $(pgrep -P $p); do kill -TERM $q 2>/dev/null; done; kill -TERM $p 2>/dev/null; done; }
            [ -n "$SP" ] && { kill -TERM $SP 2>/dev/null; for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -0 $SP 2>/dev/null && kill -KILL $SP; }
            lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && echo "WARNING: port $PORT still in use"; SP=""; CP=""; }
trap stopall EXIT
( cd $S && MOCK_CC=1 MOCK_THINK=1 MOCK_STORE=$CL/engine/models/bare MOCK_WORKDIR=$T/work exec .venv/bin/python serve.py --mem 32 --port $PORT --_screen $A/mock_tools_sl.py ) > $T/serve.log 2>&1 &
SP=$!; t0=$(date +%s)
until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
  kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 120 ] || { echo "no mock server"; exit 2; }; sleep 1; done
cd $T/work
( sleep 20 ) | env TERM=xterm-256color EXACTSTREAM_PORT=$PORT CLAUDE_BIN=/Users/kamen/.local/bin/claude CLAUDE_CONFIG_DIR=$T/cfg \
  script -q $T/tty.raw /bin/bash -c "stty cols 150 rows 40; exec $T/clone/tools/claude_code.sh" > /dev/null 2>&1 &
CP=$!
sleep 4; curl -s -m 60 http://127.0.0.1:$PORT/v1/messages -H "content-type: application/json" -H "x-api-key: local" -d "{\"model\":\"exactstream\",\"max_tokens\":300,\"messages\":[{\"role\":\"user\",\"content\":\"SLOW long answer\"}]}" > $T/outside.json & OC=$!; sleep 12; kill $OC 2>/dev/null
ps -o pid,ppid,command -g $(ps -o pgid= -p $CP | tr -d ' ') 2>/dev/null | cut -c1-140 | head -8
stopall
perl -pe 's/\e\[[0-9;?]*[ -\/]*[@-~]//g; s/\e\][^\a]*\a//g; s/\e[()][A-Z0-9]//g; s/\r/\n/g' $T/tty.raw > $T/tty.txt
echo "screen bytes: $(wc -c < $T/tty.raw); lines with the status line:"
grep -o "✓ ready\|⏳ thinking [0-9]*s .\|⚡ [0-9.]* tok/s · [0-9,]* tok\|⚡ writing\|✓ last: [^·]*· [0-9.]* tok/s" $T/tty.txt | uniq | head -30
echo "--- the last screen lines:"; grep -v '^[[:space:]]*$' $T/tty.txt | tail -12 | cut -c1-150
echo "mock server log:"; grep -c "POST\|done:" $T/serve.log; grep "done:" $T/serve.log | cut -c1-120 | head -3
