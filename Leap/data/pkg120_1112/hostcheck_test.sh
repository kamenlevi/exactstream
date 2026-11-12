#!/bin/bash
# Host / Origin check test (Leap 11-12 19:0x; the orchestrator: refuse non-loopback Host and Origin, allow no Origin and the
# loopback forms with the port, test with Claude Code 2.1.322 and the Codex scripts). The fixed serve.py of the candidate
# pkg_120b on the stand-in engine (scratchpad mock_tools_sl.py: no GPU, CPU only), port 8136, in a scratch clone of the
# candidate (its venv from the installed clean clone v120b). Then real Claude Code 2.1.322 through tools/claude_code.sh (it
# finds the server and runs claude_local.sh; sandbox-exec: writes only in this test's folder; throwaway config; only Read
# allowed) and Codex through tools/codex_code.sh (its own workspace-write sandbox, no network, no /tmp; its own CODEX_HOME).
# The server is stopped by pid at the end and the port checked. Usage: hostcheck_test.sh > hostcheck_test.out
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent; T=$A/hosttest
PKG=/Users/kamen/Leap/release/out/pkg_120b; V=/Users/kamen/Leap/release/cleanclone/v120b/clone; PORT=8136; SP=""
CCBIN=/Users/kamen/Leap/logs/claude_versions/2.1.322; CXBIN=/Users/kamen/.local/bin/codex
stopsrv() { [ -z "$SP" ] && return; kill -TERM $SP 2>/dev/null
            for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done
            kill -0 $SP 2>/dev/null && { echo "SIGKILL"; kill -KILL $SP; }; SP=""
            echo "server stopped; port $PORT listeners: $(lsof -nP -iTCP:$PORT -sTCP:LISTEN 2>/dev/null | grep -c LISTEN)"; }
trap stopsrv EXIT
rm -rf $T; mkdir -p $T/work_cc $T/work_cx $T/cfg $T/codex_home
git clone -q --single-branch $PKG $T/clone; ln -s $V/.venv $T/clone/.venv
mkdir -p $T/clone/engine/models; ln -s /Users/kamen/BigMoE/models/bare $T/clone/engine/models/bare
echo "candidate $(git -C $T/clone rev-parse --short HEAD) (pkg_120b $(git -C $PKG rev-parse --short HEAD)); serve.py sha256 $(shasum -a 256 $T/clone/serve.py | cut -c1-16)"
python3 -c "import json,sys; json.dump({'hasCompletedOnboarding': True, 'theme': 'dark', 'projects': {sys.argv[1]: {'hasTrustDialogAccepted': True, 'hasCompletedProjectOnboarding': True}}}, open(sys.argv[2], 'w'))" "$T/work_cc" "$T/cfg/.claude.json"
cat > $T/sandbox.sb <<SB
(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (remote ip "localhost:*"))
(allow network-outbound (remote unix-socket))
(deny file-write*)
(allow file-write* (subpath "$T") (subpath "/private/var/folders") (regex #"^/dev/"))
SB
( cd $T/clone && MOCK_STORE=/Users/kamen/BigMoE/models/bare MOCK_WORKDIR=$T/work_cc exec .venv/bin/python serve.py --mem 32 --port $PORT --_screen $A/mock_tools_sl.py ) > $T/serve.log 2>&1 &
SP=$!; t0=$(date +%s)
until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
  kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 120 ] || { echo "no stand-in server"; exit 2; }; sleep 1; done
B=http://127.0.0.1:$PORT; MSG='{"model":"exactstream","max_tokens":32,"messages":[{"role":"user","content":"Say hi."}]}'
code() { curl -s -o /dev/null -w '%{http_code}' -m 60 "$@"; }
post() { code $B/v1/messages -H 'content-type: application/json' -H 'x-api-key: local' "$@" -d "$MSG"; }
ok=0; bad=0; chk() { if [ "$2" = "$3" ]; then ok=$((ok + 1)); r=ok; else bad=$((bad + 1)); r=WRONG; fi; echo "  $r: $1 -> $2 (want $3)"; }
echo "== allowed: loopback Host forms, no Origin, loopback Origins"
chk "GET /health, curl's own Host (127.0.0.1:$PORT)" "$(code $B/health)" 200
for h in "127.0.0.1:$PORT" "localhost:$PORT" "[::1]:$PORT" "localhost" "LocalHost:$PORT"; do chk "GET /stats, Host $h" "$(code -H "Host: $h" $B/stats)" 200; done
chk "POST /v1/messages, no Origin" "$(post)" 200
for o in "http://localhost:3000" "http://127.0.0.1:$PORT" "http://[::1]:5173"; do chk "POST /v1/messages, Origin $o" "$(post -H "Origin: $o")" 200; done
chk "GET /health, HTTP/1.0 with no Host (a non-browser client)" "$(printf 'GET /health HTTP/1.0\r\n\r\n' | nc -w 5 127.0.0.1 $PORT | head -1 | awk '{print $2}')" 200
chk "POST /v1/chat/completions from Python urllib" "$(/opt/homebrew/opt/python@3.12/bin/python3.12 -I -c '
import json, sys, urllib.request
r = urllib.request.urlopen(urllib.request.Request(sys.argv[1] + "/v1/chat/completions", json.dumps({"model": "exactstream", "max_tokens": 16,
    "messages": [{"role": "user", "content": "Say hi."}]}).encode(), {"content-type": "application/json"}), timeout=60)
print(r.status)' $B)" 200
echo "== refused: foreign Host (DNS rebinding) and foreign Origin (cross-site)"
for h in "evil.example:$PORT" "evil.example" "127.0.0.1.evil.example:$PORT" "192.168.1.5:$PORT" "0.0.0.0:$PORT"; do chk "GET /stats, Host $h" "$(code -H "Host: $h" $B/stats)" 403; done
chk "POST /v1/messages, Host evil.example:$PORT, no Origin" "$(post -H "Host: evil.example:$PORT")" 403
for o in "https://evil.example" "null" "http://localhost.evil.example" "http://127.0.0.1.nip.io:$PORT" "file://"; do chk "POST /v1/messages, Origin $o" "$(post -H "Origin: $o")" 403; done
chk "POST /v1/responses, Origin https://evil.example" "$(code $B/v1/responses -H 'content-type: application/json' -H 'Origin: https://evil.example' -d '{"model":"exactstream","input":"hi"}')" 403
echo "refused lines in the server log: $(grep -c 'refused a ' $T/serve.log) (12 expected)"
n0=$(grep -c 'refused a ' $T/serve.log); d0=$(grep -c ' done: ' $T/serve.log)
echo "== Claude Code $($CCBIN --version 2>/dev/null | head -1) through tools/claude_code.sh (sandboxed, only Read allowed)"
cd $T/work_cc
sandbox-exec -f $T/sandbox.sb env CLAUDE_CONFIG_DIR=$T/cfg EXACTSTREAM_PORT=$PORT CLAUDE_BIN=$CCBIN \
  $T/clone/tools/claude_code.sh -p "Create a file named hello.txt here." --allowedTools Read --max-turns 3 --output-format json < /dev/null > $T/cc.out 2> $T/cc.err
echo "  rc $?; result: $(/opt/homebrew/opt/python@3.12/bin/python3.12 -I -c 'import json, sys; d = json.load(open(sys.argv[1])); print(d.get("subtype"), "|", (d.get("result") or "")[:80], "| turns", d.get("num_turns"))' $T/cc.out 2>&1)"
d1=$(grep -c ' done: ' $T/serve.log); n1=$(grep -c 'refused a ' $T/serve.log)
echo "  server: $((d1 - d0)) Claude Code requests answered, $((n1 - n0)) refused; hello.txt written: $([ -e $T/work_cc/hello.txt ] && echo yes || echo no) (Write was not allowed)"
echo "== Codex $($CXBIN --version 2>/dev/null | head -1) through tools/codex_code.sh (its workspace-write sandbox, no network)"
cd $T/work_cx
env EXACTSTREAM_CODEX_HOME=$T/codex_home EXACTSTREAM_PORT=$PORT CODEX_BIN=$CXBIN $T/clone/tools/codex_code.sh \
  exec --sandbox workspace-write --skip-git-repo-check -c sandbox_workspace_write.exclude_tmpdir_env_var=true \
  -c sandbox_workspace_write.exclude_slash_tmp=true -c sandbox_workspace_write.network_access=false -C $T/work_cx \
  "Create a file named hello.txt here." < /dev/null > $T/cx.out 2>&1
echo "  rc $?; last line: $(tail -1 $T/cx.out | cut -c1-100)"
d2=$(grep -c ' done: ' $T/serve.log); n2=$(grep -c 'refused a ' $T/serve.log)
echo "  server: $((d2 - d1)) Codex requests answered, $((n2 - n1)) refused; hello.txt = '$(cat $T/work_cx/hello.txt 2>/dev/null)'"
echo "== totals: $ok as expected, $bad wrong; server errors: $(grep -c -E 'Traceback|Error' $T/serve.log)"
grep 'refused a ' $T/serve.log | head -3 | cut -c1-200
stopsrv; trap - EXIT
