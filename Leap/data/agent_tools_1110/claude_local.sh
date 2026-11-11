#!/bin/bash
# Claude Code on the local server (start it first: .venv/bin/python serve.py --mem 24). The settings below apply to this
# process only; ~/.claude is not changed. EXACTSTREAM_PORT / EXACTSTREAM_CONTEXT override the port and the context window.
# All three model names run the same engine; the suffixes tell serve.py which calls are Claude Code's background ones
# (-side: titles, command checks) and which are a subagent's (-sub), so a conversation cache can keep the main one.
PORT=${EXACTSTREAM_PORT:-8080}; CTX=${EXACTSTREAM_CONTEXT:-131072}
export ANTHROPIC_BASE_URL=http://127.0.0.1:$PORT ANTHROPIC_AUTH_TOKEN=local ANTHROPIC_MODEL=exactstream \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=exactstream-side ANTHROPIC_SMALL_FAST_MODEL=exactstream-side \
  CLAUDE_CODE_SUBAGENT_MODEL=exactstream-sub \
  CLAUDE_CODE_MAX_CONTEXT_TOKENS=$CTX CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
exec "${CLAUDE_BIN:-claude}" "$@"
