#!/bin/bash
# Claude Code's status line for this Mac's model. tools/claude_code.sh sets it for its own session through --settings, so
# ~/.claude is not changed. It prints one line from serve.py's GET /statusline, for example:
#   ⏳ thinking 12s ◐ · prompt 21,345 tok (18,900 cached)   the prompt pass, before the first token; the mark turns each second
#   ⚡ 38.2 tok/s · 210 tok · thought 6.8s                 writing: live speed, tokens so far, time to the first token
#   ✓ last: 412 tok · 37.9 tok/s · thought 6.8s            idle: the last answer (Claude Code's small background calls and
#                                                           subagents are marked "· background" / "· subagent")
# It is fast by design: one curl with a 0.3 s limit. If the server does not answer, or answers without a status line (one
# started before this version), it prints a plain line and exits 0.
# Usage: statusline.sh [port]   (default: EXACTSTREAM_PORT, else 8080)
PORT=${1:-${EXACTSTREAM_PORT:-8080}}
L=$(curl -sf -m 0.3 "http://127.0.0.1:$PORT/statusline" 2>/dev/null)
if [ -n "$L" ]; then
  printf '%s\n' "$L"
elif curl -sf -m 0.3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "ExactStream on port $PORT: restart serve.py for the live status line"
else
  echo "ExactStream: no answer on port $PORT"
fi
exit 0
