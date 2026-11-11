# Tool calls for Claude Code and Codex: serve.py working copy (Tue 10 Nov, evening)

> (a quote from the project owner, kept private)
model in Claude Code and Codex. The work is in a separate working copy (release/cleanclone/agent/clone, gitignored):
tomorrow's 1.1.0 candidate (pkg_110c) does not change; this becomes a later release. The files below are copies of it.

## What is built

- **agentio.py** (new, data/agent_tools_1110/agentio.py) turns the three wire formats into the chat template's messages
  and parses the model's streamed text back into text pieces and tool calls:
  - the template lists tools in the system turn and has the model answer a call as
    `<tool_call>` / `<function=NAME>` / `<parameter=ARG>` VALUE `</parameter>` ... `</function>` / `</tool_call>`;
  - parameter values are typed from the tool's schema (strings stay raw; other types are JSON-decoded);
  - the parser holds back text only while it could be the start of `<tool_call>` and emits a call when `</tool_call>`
    arrives.
- **serve.py** (data/agent_tools_1110/serve_tools.diff against release/src/serve.py):
  - Anthropic Messages with tools, streamed as message_start, text blocks, content_block_start tool_use,
    input_json_delta, content_block_stop, message_delta (stop_reason tool_use / end_turn / max_tokens), message_stop.
    Nothing is buffered beyond the parser's hold-back.
  - OpenAI Responses (POST /v1/responses, Codex): function tools; response.created / in_progress, message items with
    output_text deltas, function_call items with arguments delta / done, response.completed with usage.
  - OpenAI chat completions keep working, now with tools (tool_calls deltas, finish_reason tool_calls).
  - Accepted and ignored: thinking (adaptive), cache_control, context_management, output_config, metadata, the
    anthropic-beta / anthropic-version headers, include / store / prompt_cache_key / reasoning (Responses).
  - Mid-conversation system messages become user turns in place (the template allows a system turn only first).
  - Errors in each API's own shape, e.g. Anthropic `invalid_request_error` "prompt is too long: N tokens > M maximum",
    OpenAI `context_length_exceeded`; `--max-context` (default 131,072).
  - The model id is "exactstream" (no "claude-" prefix).
  - PF-A prefill at 32 / 24 GB, as chat.py (fixed during this work: the first draft also set it at 45 GB, where the
    engine refuses it).
- **Setup commands** (data/agent_tools_1110/):
  - start the server: `.venv/bin/python serve.py --mem 24` (or 32 / 45);
  - Claude Code: `tools/claude_local.sh` sets ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN, ANTHROPIC_MODEL,
    ANTHROPIC_DEFAULT_HAIKU_MODEL, ANTHROPIC_SMALL_FAST_MODEL, CLAUDE_CODE_SUBAGENT_MODEL, CLAUDE_CODE_MAX_CONTEXT_TOKENS
    and CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 for that process only, then runs `claude`;
  - Codex: `tools/codex_local.sh` writes its own CODEX_HOME (default ~/.exactstream/codex) with a model_providers entry
    (base_url, wire_api "responses", model_context_window, model_auto_compact_token_limit), then runs `codex`.
  - Neither touches ~/.claude or ~/.codex.

## What the clients send (from a local stub server, Tue 20:1x; raw captures stay in the scratchpad)

- **Claude Code 2.1.321** (`-p`): POST /v1/messages?beta=true with context_management, max_tokens 32000, messages,
  metadata (user_id: a JSON string with device_id, account_uuid, session_id), model, output_config, stream, system
  (three blocks: a billing-header line, an Agent SDK line, the ~5.7K-char main prompt), thinking {"type": "adaptive"},
  and 12 tools (~30K chars of JSON). The environment block (cwd, platform, date) arrives as a role "system" message after
  the user's prompt. Headers: anthropic-beta, anthropic-version, X-Claude-Code-Session-Id, x-app, User-Agent.
- **Codex** (`exec`): POST /v1/responses with client_metadata, include [], input (a developer message first), instructions
  (~20.7K chars), model, parallel_tool_calls false, prompt_cache_key, reasoning null, store false, stream, tool_choice
  auto, and tools: functions exec_command, write_stdin, update_plan, request_user_input, view_image, get_goal,
  create_goal, update_goal, plus a namespace (multi_agent_v1) and web_search (not offered to the model). Headers:
  originator, session-id, thread-id, x-codex-turn-metadata. Codex reads stdin when it is not a TTY: run it with </dev/null.

## Containment for the tests (a lossy local model drives the tools)

- Claude Code: the binary is called directly (/Users/kamen/.local/bin/claude; the shell alias adds
  --dangerously-skip-permissions), with an isolated CLAUDE_CONFIG_DIR, `--allowedTools Read Write Edit Glob Grep`, under a
  sandbox-exec profile (data/agent_tools_1110/sandbox.sb): writes only in the test folders and the isolated configs,
  network only to localhost.
- Codex: its own seatbelt sandbox, workspace-write on the test folder only, /tmp and $TMPDIR excluded, network off, its
  own CODEX_HOME. (An outer sandbox-exec breaks Codex's own sandbox: nested seatbelt profiles fail silently.)

## Protocol tests on a mock engine (no GPU; mock_tools.py stands in for the decode loop)

- Claude Code: the mock answers with a Write call; Claude Code ran it inside the test folder (hello.txt written), sent the
  tool_result back, and printed the mock's final answer. rc 0.
- Codex: the mock answers with an exec_command call; Codex ran it in its sandbox ("sandbox: workspace-write [workdir]"),
  sent function_call_output "Process exited with code 0" back, and printed the final answer. rc 0.

## Hooks for a conversation cache (orchestrator 20:4x, for BigMoE's docs/CONV_CACHE_DESIGN.md)

Each prompt item serve.py hands the loop now carries `sys_end`, `side` and `agent` (agentio.meta); the current loops
ignore them.
1. **sys_end:** the token index where the system turn (system prompt + tool list) ends: the first `<|im_start|>` after
   the opening one. Claude Code: 10,017 of ~12.3K prompt tokens; Codex: 7,848 of ~8.4K.
2. **side / lane:** main, side (must not evict the main conversation's state) or sub (a subagent's own conversation).
   - side: Claude Code's small-model calls (titles, command checks; claude_local.sh names that model
     exactstream-side), quota-style probes (max_tokens <= 16), and a tool-less call while an agent session (a main
     request with tools) was live in the last 30 minutes;
   - sub: model exactstream-sub (CLAUDE_CODE_SUBAGENT_MODEL), or the side model with tools;
   - `conv`: a one-way 12-hex key of the client's session id; `prefix_key`: a hash of the system turn's tokens.
3. **Recorded sessions:** EXACTSTREAM_LOG_REQUESTS=<file> logs, per request, the body (plus user-agent / x-app /
   anthropic-beta / anthropic-version / originator headers), the rendered ids with the meta, and the result (events,
   the answer's token ids, stats). They stay in the scratchpad (sessions_real_1110.jsonl), not in git.
4. **The same history renders to the same tokens:** one assistant message per model turn however a client groups it
   (consecutive assistant messages merge; a Responses client's function_call items rejoin the turn that made them, even
   with outputs sent between them, from a call-id map serve.py keeps), and Claude Code's billing-header line is left out
   of the system prompt (it is not for the model and could vary).

**Template check:** past assistant turns render as `<think>\n\n</think>\n\n` + content (preserve_thinking is on by
default), the same bytes the non-thinking generation prompt ends with, and a call renders as
`\n\n<tool_call>\n<function=...` after trimmed text, as the model writes it. So a turn the model wrote in the template's
own form re-renders to the same tokens.

**Mock check (prefix_check.py on the mock run's log):** for both clients, the follow-up request starts with the first
request's prompt plus the answer, token for token (Claude Code: 12,391 of 12,391; Codex: 8,404 of 8,404), and the system
turn's tokens are the same in every request of a session. PREFIX: 2 of 2.

## Real engine (serve.py --mem 24, PF-A; Tue 21:03-21:16): tool calls work in Claude Code and Codex

> (a quote from the project owner, kept private)
swap flat at 6,854-6,862 MB). Evidence: data/agent_tools_1110/real/. 45 requests, 0 server errors.

- **Tool-call probe (probe.py: 10 requests x the Anthropic, Responses and chat-completions endpoints, 6 tools): 27 / 30**
  well-formed calls, right tool, right arguments. All 27 calls made were correct in every argument (strings, numbers,
  arrays, a "from" key). The 3 misses are one request on all three endpoints ("Search for files named report*, max 3
  results."): the model answered in prose with made-up file names instead of calling search_files. The rendered prompt
  is the same on the three endpoints, so the answer is too (greedy). A model miss, not a protocol one; the lossy 24 GB
  line was not compared with an exact config here.
- **Claude Code** (claude binary, isolated config, sandbox profile, tools Read / Write / Edit / Glob / Grep):
  - T1 "Create a file named notes.txt in this folder that contains exactly the word hello.": Write ran; notes.txt =
    "hello". PASS.
  - T2 "Read notes.txt ... tell me the word it contains.": Read ran; "The file contains the word: **hello**". PASS.
  - T3 "In notes.txt ..., replace the word hello with goodbye.": notes.txt = "goodbye". PASS.
- **Codex** (its own sandbox, workspace-write on the test folder, no network, no /tmp):
  - T1 create: `printf 'hello' > notes.txt`; notes.txt = "hello". PASS.
  - T2 "How many files are in this folder? Answer with just the number.": ran a listing; answered "1". PASS.
  - T3 "Append a second line containing the word world to notes.txt": it looked at the file and ran
    `printf 'world\n' >> notes.txt` in one command, but the file had no final newline (T1 wrote exactly "hello"), so the
    result is one line, "helloworld", and Codex reported that ("now contains `helloworld`"). PARTIAL: the protocol worked;
    the model missed the missing newline.
- **Side lane:** a request with model exactstream-side and no tools came back lane "side", side True; every Claude Code and
  Codex request was lane "main".
- **Prefix check (prefix_check.py on the recorded sessions): PREFIX 8 of 8.** In all 3 Claude Code sessions (2, 2 and 3
  requests) and all 3 Codex sessions (2, 3 and 2), each follow-up request starts with the earlier prompt plus the model's
  answer, token for token. The system turn's tokens are the same in every request of a session (sys_end 10,019 for
  Claude Code, 7,850 for Codex). So with the real model's answers, a conversation cache would prefill only the new turn.
- **Cost per turn without a cache (24 GB, PF-A):** Claude Code's 12.3-12.6K-token prompts take 30.5-32.2 s to the first
  token (about 400 tok/s), Codex's 8.4-8.6K-token prompts 21.7-23.2 s; a 700-token probe prompt 6.1-6.2 s (the floor at
  24 GB: a prefill reads nearly every expert from the SSD). Answers are short (2-123 tokens). With the cache, a
  follow-up would prefill only its new 90-230 tokens.
- **Memory:** serve.py's lifetime peak was 24.43-24.47 GB on the probe's ~700-token prompts and 27.32 GB on the 12.6K-token
  Claude Code prompts (run.py's chat-24 with PF-A: 26.54 GB at 10K). Agent prompts take the 24 GB line over 24 GB; the
  docs for a tool-call release must say so.
- **Recorded sessions for BigMoE's turn-replay test:** scratchpad/agent/sessions_real_1110.jsonl (1.7 MB: 45 requests'
  bodies, rendered ids + meta, results with the answers' token ids). Checked: no leak or quote pattern, no email, no home
  paths. Not in git.

## Tonight: long-context runner (armed; from 00:30)

long_ctx.sh: the chat configs at 24 / 32 / 45 GB on a 64,000- and a 128,000-token prompt (the five papers, cut; 128K
repeats them from the start of MAIN), 64 new tokens, PF-A and the default prefill at 24 / 32 GB, 45 GB as published.
Per run: first token, whole decode, lifetime peak, NO-SWAP before, swap growth. Arms run in order of predicted lifetime
peak (linear from the measured 10K / 30K peaks): 24 default 64K 32.2 GB, 24 PF-A 64K 37.5, 24 default 128K 39.7,
32 default 128K 42.7, 32 PF-A 64K 44.7, 24 PF-A 128K 50.5, 45 64K 51.7, 32 PF-A 128K 56.4, 45 128K 56.5. A measured 64K
peak replaces the guess for the same arm's 128K. An arm starts only when NO-SWAP >= its prediction + 2 GB, with the usual
night gates; swap +2 GB, Kamen back or a 50-min cap stop a run; last start 07:30.

## Found along the way

- My runners' "another engine runs" check used `ps -o comm=`, which macOS cuts to 16 characters: a venv Python shows as
  `/opt/homebrew/Ce` and never matched "python". Fixed in the new runners (full command line). The lock and Kamen-idle
  checks had still held.
- macOS /bin/bash is 3.2 (no associative arrays): the first long_ctx.sh exited at once; rewritten with a state file.

## The test server outlived its test (orchestrator 21:2x)

The orchestrator found the agent-test server still running at 21:2x: serve.py --mem 24 on port 8097 (pid 88198, parent 1),
27.4 GB, ~100 % CPU. agent_tests.sh started it as `cd $CL && ... serve.py ... &`, so `$!` was the wrapper subshell, and
the script's SIGTERM stopped the subshell but not serve.py. Stopped at 21:18 with SIGTERM: it exited in ~1 s with its
"stopped" line; NO-SWAP back to 42 GB, swap flat, port free, no engine left. Fixed in agent_tests.sh: serve.py runs as the
job itself (`( cd ... && exec ... ) &`), and the stop step waits up to 60 s for it to exit (then SIGKILL) and checks the
port. long_ctx.sh already stops the whole process tree on its stop paths, and its normal path ends with run.py.

**Open: ~100 % CPU while idle.** The server had had no request for ~2.5 min and its engine log shows nothing after the
last request, so a thread that does not log was spinning. The 1.1.0 chat.py and serve.py share the engine glue.
idle_cpu.sh (armed: from 00:15, the usual gates) measures the 1.1.0 candidate's serve.py at --mem 24 and 32: CPU time
over 60 s idle after one request, and a 5-s `sample` of the process when it is over 20 %.
