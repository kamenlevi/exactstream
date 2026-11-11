#!/bin/bash
# Stand-in check of the per---mem window defaults in release/next (orchestrator option (b): 64K at --mem 32 / 45, 128K at
# 24; the window variable only from 100K), run from the cache clone (its tools = release/next's after this change).
# The stand-in engine (mock_tools.py; no GPU, < 1 GB) is started here once on its own port, so the one-command launchers
# reuse it (they start serve.py only when nothing answers on the port). Per setting:
#   claude_code.sh --mem M -p ... (real Claude Code 2.1.321, isolated config folder, neutral HOME, a scratch cwd, no tools
#   allowed) -> the result line's contextWindow (expect 65536 at 45 / 32, 131072 at 24), and the env Claude Code got
#   (AUTO_COMPACT_WINDOW unset below 100K);
#   codex_code.sh --mem M with CODEX_BIN=/usr/bin/true and a scratch EXACTSTREAM_CODEX_HOME -> config.toml's
#   model_context_window and model_auto_compact_token_limit;
#   EXACTSTREAM_CONTEXT=100000 claude_code.sh --mem 32 -> the override wins (100000).
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
CC=/Users/kamen/Leap/release/cleanclone/cache/clone; PORT=8133; B=$A/window_defaults; rm -rf $B; mkdir -p $B/work $B/home
cp $A/agent_home_tpl/.zshrc $A/agent_home_tpl/.gitconfig $B/home/
SP=""
stopsrv() { [ -z "$SP" ] && return; kill -TERM $SP 2>/dev/null
            for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done
            kill -0 $SP 2>/dev/null && { echo "serve.py did not stop in 60 s: SIGKILL"; kill -KILL $SP; }
            lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && echo "WARNING: port $PORT still in use"; SP=""; }
trap stopsrv EXIT
( cd $CC && MOCK_STORE=$CC/engine/models/bare MOCK_WORKDIR=$B/work exec .venv/bin/python serve.py --mem 32 --port $PORT --_screen $A/mock_tools.py ) > $B/serve.log 2>&1 &
SP=$!; t0=$(date +%s)
until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
  kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 120 ] || { echo "mock serve.py did not come up: $(tail -3 $B/serve.log)"; exit 2; }; sleep 1
done
echo "mock serve.py ready in $(( $(date +%s) - t0 )) s (pid $SP)"
win() { /opt/homebrew/opt/python@3.12/bin/python3.12 -I -c '
import json, sys
w = None; res = None
for l in open(sys.argv[1]):
    try: d = json.loads(l)
    except Exception: continue
    if d.get("type") == "result":
        res = d.get("subtype")
        for k, v in (d.get("modelUsage") or {}).items():
            if v.get("contextWindow"): w = (k, v.get("contextWindow"))
print(f"result {res}; contextWindow {w}")' "$1"; }
run_claude() {  # $1 = tag, then env assignments for the launcher
  local tag=$1; shift
  rm -rf $B/cc_$tag; mkdir -p $B/cc_$tag
  ( cd $B/work && env HOME=$B/home CLAUDE_CONFIG_DIR=$B/cc_$tag EXACTSTREAM_PORT=$PORT CLAUDE_BIN=/Users/kamen/.local/bin/claude "$@" \
      $CC/tools/claude_code.sh $MEMARG -p "Say hello." --output-format stream-json --verbose --max-turns 1 --allowedTools "" \
      < /dev/null > $B/$tag.jsonl 2> $B/$tag.err )
  echo "claude $tag: rc $?; $(win $B/$tag.jsonl)"
}
for M in 45 32 24; do MEMARG="--mem $M"; run_claude m$M; done
MEMARG="--mem 32"; run_claude override100k EXACTSTREAM_CONTEXT=100000
# what the launcher exports (the clamped variable must be absent below 100K): print it through a stand-in for claude
for M in 32 24; do
  echo "env at --mem $M: $(cd $B/work && env HOME=$B/home EXACTSTREAM_PORT=$PORT CLAUDE_BIN=/usr/bin/env $CC/tools/claude_code.sh --mem $M 2>/dev/null | grep -E '^CLAUDE_CODE_(MAX_CONTEXT_TOKENS|AUTO_COMPACT_WINDOW)=' | sort | tr '\n' ' ')"
done
for M in 45 32 24; do
  rm -rf $B/codex_$M; ( cd $B/work && env HOME=$B/home EXACTSTREAM_PORT=$PORT EXACTSTREAM_CODEX_HOME=$B/codex_$M CODEX_BIN=/usr/bin/true \
      $CC/tools/codex_code.sh --mem $M < /dev/null > $B/codex_$M.out 2>&1 ); rc=$?
  echo "codex --mem $M: rc $rc; $(grep -E '^model_(context_window|auto_compact_token_limit)' $B/codex_$M/config.toml | tr '\n' ' ')"
done
echo "server requests: $(grep -c 'done:' $B/serve.log); still ours: $(kill -0 $SP 2>/dev/null && echo yes)"
stopsrv; echo "mock serve.py stopped"
