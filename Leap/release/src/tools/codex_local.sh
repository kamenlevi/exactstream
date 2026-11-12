#!/bin/bash
# Codex on the local server (start it first: .venv/bin/python serve.py --mem 24), with a CODEX_HOME of its own (default
# ~/.exactstream/codex; ~/.codex is not changed) holding a provider entry for the Responses API.
PORT=${EXACTSTREAM_PORT:-8080}; CTX=${EXACTSTREAM_CONTEXT:-131072}
export CODEX_HOME=${EXACTSTREAM_CODEX_HOME:-$HOME/.exactstream/codex}; mkdir -p "$CODEX_HOME"
cat > "$CODEX_HOME/config.toml" <<CFG
model = "exactstream"
model_provider = "exactstream"
model_context_window = $CTX
model_auto_compact_token_limit = $((CTX * 9 / 10))

[model_providers.exactstream]
name = "ExactStream (local)"
base_url = "http://127.0.0.1:$PORT/v1"
wire_api = "responses"
env_key = "EXACTSTREAM_API_KEY"
CFG
export EXACTSTREAM_API_KEY=${EXACTSTREAM_API_KEY:-local}
exec "${CODEX_BIN:-codex}" "$@"
