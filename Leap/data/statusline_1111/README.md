# Status line for Claude Code (Kamen's ask, Wed 11 Nov 16:5x via the orchestrator)

> (a quote from the project owner, kept private)

## What was built (release/next, installed in the cache clone 17:1x)

- **serve.py**: two read-only endpoints from the job's own timestamps. Nothing here touches the engine or the tokens.
  - `GET /stats`: JSON with `state` (loading / idle / thinking / writing), `queued`, `lane`, `prompt`, `cached`,
    `think_s`, `out`, `tps`, and `last`.
    - `last` is the last finished answer of the conversation or a subagent. Claude Code's small background calls (the
      `side` lane: titles, command checks) show only until a real answer exists.
  - `GET /statusline`: the same as one short line:
    - `⏳ thinking 12s ◐ · prompt 21,345 tok (18,900 cached)`: the prompt pass before the first token. The mark turns
      each second, and the cached count comes from the conversation cache's record for this prompt.
    - `⚡ 38.2 tok/s · 210 tok · thought 6.8s`: writing. Live tok/s = tokens after the first commit / time since the
      first token. It converges to the engine's whole-decode figure: 56.3 live vs 56.4 final on the stand-in.
    - `✓ last: 412 tok · 37.9 tok/s · thought 6.8s`: idle.
    - Tags: `· background` (side lane), `· subagent` (sub lane), `· N queued`.
- **tools/statusline.sh**:
  - one `curl -sf -m 0.3` to /statusline; always one line, exit 0;
  - a server that answers /health but not /statusline (one started before this change) gets
    `ExactStream on port N: restart serve.py for the live status line`;
  - no server gets `ExactStream: no answer on port N`.
- **tools/claude_code.sh** passes
  `--settings '{"statusLine":{"type":"command","command":"'<root>/tools/statusline.sh' <port>","padding":0,"refreshInterval":1}}'`.
  - This is for its own session only; ~/.claude is not changed.
  - refreshInterval 1 re-runs the line every second, so it ticks while the prompt is read. Claude Code 2.1.322's
    binary documents it: "Re-run the status line command every N seconds in addition to event-driven updates".
- SHA256SUMS: updated (serve.py, tools/claude_code.sh) and tools/statusline.sh added; `shasum -c` OK.

## Tests

All tests ran on the stand-in engine: `mock_tools_sl.py` = scratch mock_tools.py plus two test knobs (see
mock_tools_sl.diff):
- MOCK_THINK: seconds before the first token;
- MOCK_CC: a stand-in conversation-cache record, for the cached count.

Setup common to every test:
- port 8134, in a scratch tree that symlinks the cache clone;
- no GPU; Kamen's server on 8080 untouched;
- every server is stopped by its pid, with a port check at the end.

| test | what | result |
|---|---|---|
| statusline_test.sh | one SLOW main answer (1,500 tok at ~56 tok/s), /statusline polled every 0.5 s | ready -> thinking 0..3s with the mark + "prompt 19 tok (14 cached)" -> "⚡ writing" -> "⚡ 54.6..56.1 tok/s" -> "✓ last: 1,502 tok · 56.3 tok/s · thought 3.0s"; closed port: plain line, rc 0 |
| sl_timing.sh | why statusline.sh took 975 ms once | warm runs 13-14 ms idle, 15-25 ms while writing; raw curl 11 ms (server 0.5 ms). A fresh copy's FIRST run costs 165-190 ms (macOS checks a new executable once), second 14-16 ms: the 975 ms was that first run of a freshly copied file on a busy Mac (swap 8.4 GB) |
| statusline_test2.sh | lanes: a side call alone, then the main answer with a side call queued at t+5 s | "✓ last: 7 tok ... (side)" alone; "· 1 queued" while it waits; after both: the main answer's "1,502 tok · 56.4 tok/s" (the side call finished later but is skipped); warm 14-18 ms |
| sltui_test.sh | the real Claude Code 2.1.322 TUI started by tools/claude_code.sh from a launcher-only tree (no .venv: it cannot start an engine), throwaway config folder, pty, no prompt typed | the status bar shows "✓ ready" |
| sltui_test2.sh | the same, plus a request from outside at t+4 s: no Claude Code event, so only refreshInterval can update the line | the status bar ticks: "✓ ready" -> "⏳ thinking 1s ◓ · prompt 16 tok (11 cached)" -> "⚡ 55.2 tok/s · 43 tok · thought 1.0s" -> "✓ last: 300 tok · 56.6 tok/s · thought 1.0s" |
| statusline_test3.sh | the final script against (a) the clone's old serve.py, (b) the final serve.py, (c) a closed port | (a) HTTP 404 -> "restart serve.py for the live status line"; (b) "✓ ready", "· background" after a side call alone, then the main answer stays after another side call; (c) "no answer"; all rc 0, 15-25 ms |

Not tested: the real engine. Its prompt pass and commits feed the same `j.t0` / `j.t1` / `j.out` that the stand-in feeds,
and serve.py's own `done:` line already reads them.

## Install in the cache clone (Kamen runs its tools/claude_code.sh)

- serve.py, tools/claude_code.sh and tools/statusline.sh were copied then RENAMED into place (new inodes).
  - Backups: serve.py.pre_sl_1111, tools/claude_code.sh.pre_sl_1111.
  - Kamen's running launcher (pid 33904, since 16:54) keeps reading its old copy: fd 255 still on inode 49151946.
  - His server (pid 33915, port 8080) is untouched.
- **He picks it up by exiting Claude Code and running tools/claude_code.sh again.**
  - Its exit trap stops the server it started, and the new run loads the new serve.py (about a minute).
  - If he restarts only Claude Code against the old server, the line says "restart serve.py for the live status line".
