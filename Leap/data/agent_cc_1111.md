> (a quote from the project owner, kept private)

Target (orchestrator): Thu 12 Nov 17:00, Claude Code working well on this Mac. One command starts the server and Claude
Code; tool calls reliable on EvalSuite's agent test; turns start in seconds with BigMoE's conversation cache; sessions to
64-128K tokens with compaction. Leap's parts: harden /v1/messages, compaction, wire the cache, real sessions + the
default memory setting, setup. The 1.1.0 candidate stays frozen; this is the next release (Leap release/next).

## Tue 23:0x-23:3x: /v1/messages hardened (release/next 2b03787, 243874c)

- **Several tool calls in one turn, several results back:** real Claude Code on the stand-in engine (a turn with two Read
  calls): it ran both and sent two tool_results; the next prompt held both. PASS.
- **Images / documents:** a block becomes "[an image was attached here; this model reads text only and cannot see it]"
  in user turns and in tool results (Claude Code's Read of a PNG): no error, the model is told. PASS.
- **tool_choice** auto / any / tool / none accepted (none = no tools). **stop_sequences** honoured, streamed and not
  (stop_reason stop_sequence + the sequence); OpenAI `stop` too. **max_tokens** caps the answer.
- **count_tokens** = the rendered prompt's length (275 = 275 in the check).
- **"prompt is too long: N tokens > M maximum"** (400 invalid_request_error) past --max-context: Claude Code 2.1.321 matches
  exactly `^prompt is too long: \d+ tokens > \d+ maximum$` (its own string table).
- **One engine queue, main > sub > side:** a main request queued after a side one runs first. PASS.
- **Keep-alive pings** every 10 s while a request waits or prefills (Anthropic `ping` events, SSE comments for OpenAI /
  Responses); a client that leaves mid-answer stops the answer at the next check; one that leaves while queued is dropped
  before it runs. PASS.
- Quota probes = max_tokens <= 4 (was 16: an 8-token main request was wrongly side).
- HTTP checks on the stand-in engine: **9 / 9** (harden_test.py).

## Claude Code 2.1.321's window settings (from its string table)

For a model it does not know, Claude Code assumes a window and says "set CLAUDE_CODE_MAX_CONTEXT_TOKENS to its real
window"; "To enforce it, set CLAUDE_CODE_AUTO_COMPACT_WINDOW=". claude_local.sh now sets both to EXACTSTREAM_CONTEXT
(default 131,072). A run's own usage line confirms contextWindow 131072 and maxOutputTokens 32000. Other knobs seen:
CLAUDE_AUTOCOMPACT_PCT_OVERRIDE, CLAUDE_CODE_MAX_OUTPUT_TOKENS, CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT.

## EvalSuite's agent test (PREREG_agentcc_2026-11-11, 63218d0): Leap's runner

agentcc_night.sh <GB>: gates (Kamen idle, no lock / engine, NO-SWAP), a share lock, the pinned server (the clone's files
must match release/next/SHA256SUMS), probe_cc.py, then agentcc_tasks.py: per task a fresh folder (git init + commit), a
fresh Claude Code config folder, claude -p (stream-json, 25 turns, the manifest's timeout, process group killed on
timeout) in a sandbox (writes only in the run folder and temp; network only to localhost; ~/.ssh, ~/.claude*, ~/.codex,
Keychains, Mail, Chrome, Documents, Desktop, Downloads, ~/Papers and the chat sessions unreadable), then the check,
sandboxed. Smoke test on the stand-in engine (t01, t11): git, transcript, check, results.jsonl all as specified.
Armed: 32 GB from Wed 00:30 (Kamen's go for tonight: apps closed 00:29-07:30), then the 64K / 128K long-context runs.

## Tue 23:2x-23:4x: one command, and compaction end to end (release/next, this commit)

- **tools/claude_code.sh** starts serve.py (unless one already answers on the port), waits until it is loaded, runs
  Claude Code with claude_local.sh's settings, and stops the server it started when Claude Code exits; the server gets
  its own process group, so Ctrl-C in Claude Code does not reach it. Stand-in test: Claude Code answered, the server
  stopped afterwards, the port was free.
- **Compaction, real Claude Code on the stand-in engine:**
  - Auto-compact with a 24K window and two large reads: three compact_boundary events; Claude Code cleared old tool
    results locally (36.7K -> 12.7K tokens) and measured big results with our count_tokens. No error.
  - **/compact** (`claude -p --continue "/compact"`): the summary request uses **the session's own system turn and tools**
    (sys_end 10,017, 14 tools) plus a final instruction ("CRITICAL: Respond with TEXT ONLY. Do NOT call any tools. ...
    an <analysis> block followed by a <summary> block"); a reply with tool calls makes Claude Code retry once with a
    stronger paragraph; a proper summary is accepted and the session continues from it (13,865 -> 13,008 tokens).
  - So with the conversation cache a compaction re-prefills only its instruction. serve.py flags these requests
    (meta compact: true) for the cache.
- The billing-header line changes between requests of one session (cc_version=2.1.321.49c, then .655): dropping it from
  the rendered prompt (done on Tue) keeps the prefix stable.
- **tools/codex_code.sh** (Codex second): the same one-command pattern with codex_local.sh. Stand-in test: Codex ran its
  exec_command in its own sandbox (hello.txt written), finished, and the server stopped with the port free. Kept in
  data/agent_cc_1111/ tonight so that release/next stays at the commit EvalSuite pinned (39947b1); it moves there after
  the run.

## Tue 23:3x: BigMoE's conversation cache wired (a separate clone), T1 armed

BigMoE's convcache (BigMoE 7eb3401, conv/convcache.py; PF-A supported in one sweep, the taps lever in) is in a separate
clone, release/cleanclone/cache (release/next's server + engine/loop/convcache.py + the hook, serve_cache_hook.diff), so
tonight's pinned EvalSuite clone stays unchanged. The hook: CC.install(G()) at the first next_item (an import / install
failure logs "conversation cache off" and serve.py runs cold: checked on the stand-in engine); CC.hint(sys_end, lane, side,
conv) before each item; CC.last() into the done line and the request log. CONV_CACHE_GB defaults 0.6 / 2.5 / 6 by --mem;
DFlash2 throughout while the cache is on (BigMoE: an MTP-head turn can restore only from an MTP-head entry).
**T1** (t1_cache.sh + replay.py): serve.py --mem 32 with CONV_CACHE_CHECK=1 (every hit compared with a cold prefill),
replaying the 45 recorded requests of sessions_real_1110 in order. Night order: two quick checks -> T1 (00:29) ->
EvalSuite 32 GB cold (pin 39947b1) -> 64K / 128K runs. If T1 passes: the cached agent test on Wednesday (32, then 45 GB).

## Wed 01:01-01:13: T1 FAIL (5 of 12 hits bitwise); cause of 4 found (BigMoE's path rule); T1b armed

T1 ran at 01:01 (the 00:29 start moved to 01:00 by Kamen): serve.py --mem 32 (chat-32, PF-A, DFlash2) in the cache clone
with BigMoE's convcache 7eb3401 unchanged, CONV_CACHE_CHECK=1, the 45 recorded requests replayed in order
(data/agent_cc_1111/t1_32_1111/). 45 / 45 HTTP 200, 0 server errors, ready in 18 s. 33 misses (30 short ~700-token
requests and the 18-token quota probe: one chunk, never stored; the first prompt of each conversation), **12 hits: check
PASS 5, FAIL 7.** k = the replay index (30-36 Claude Code /v1/messages, 37-43 Codex /v1/responses):

| k | N | c (source) | new tokens | warm prefill | check |
|---|---|---|---|---|---|
| 30 | 12,302 | miss | 12,302 | 29.0 s | - |
| 31 | 12,496 | 12,288 (live) | 208 | 5.5 s | PASS |
| 32 | 12,300 | 9,216 (copy) | 3,084 | 12.4 s | FAIL B |
| 33 | 12,404 | 12,288 (live) | 116 | 4.1 s | FAIL A |
| 34 | 12,300 | 9,216 (copy) | 3,084 | 12.2 s | FAIL B |
| 35 | 12,404 | 12,288 (live) | 116 | 4.0 s | FAIL A |
| 36 | 12,627 | 12,288 (live) | 339 | 6.3 s | FAIL B |
| 37 | 8,390 | miss | 8,390 | 21.4 s | - |
| 38 | 8,491 | 8,192 (live) | 299 | 7.0 s | PASS |
| 39 | 8,388 | 8,192 (live) | 196 | 6.9 s | FAIL A |
| 40 | 8,479 | 8,192 (live) | 287 | 6.7 s | PASS |
| 41 | 8,573 | 8,192 (live) | 381 | 7.8 s | PASS |
| 42 | 8,389 | 8,192 (live) | 197 | 6.0 s | FAIL A |
| 43 | 8,617 | 8,192 (live) | 425 | 7.8 s | PASS |

(Warm prefill = the cached path alone; the check's own cold prefill, test-only, is excluded, so the logged first tokens
of 28-41 s include it.)

- **Signature A** (k 33, 35, 39, 42): "last logits, states (119 of 122), taps tail" at 116 / 116 / 196 / 197 new tokens;
  every hit with 208-425 new tokens on the same path passed. **Cause (BigMoE agrees, its bug):** convcache chose the
  prefill path by the new part (`N - c > chunk` -> PF-A), the loop by the whole prompt (r57_screen_rel1.py:987,
  `len(ids) > chunk`). A tail of one chunk or less therefore ran chunk-major (tier.prefill_call's compact scratch of the
  call's unique experts) while the cold prefill runs it inside the PF-A sweep (mode 1's full-E scratch on the original
  ids). pflm.py:15-17 already warns that E = 512 vs len(unique) changes MLX's gather kernel choice on short calls; the
  boundary (197 FAIL, 208 PASS) fits a kernel switch on the call's size; 119 of 122 leaves = from the first MoE call on.
  BigMoE's confound: all four also had N below the entry's previous prompt + reply (stale QSA rows past N); T1b replays
  the same order, so a PASS there rules it out.
- **Signature B** (k 32, 34, 36): "states (1 of 122)" only, last logits and taps tail equal: a copy restore at 9,216
  (3,084 new tokens through PF-A from c) and a live trim at 12,288 (339 new tokens chunk-major). The check did not name
  the leaf. BigMoE's suspects: the PLE layer's slots (chunk-major vs PF-A's own PLE staging), or a QSA raw / pooled leaf
  after a copy (the state setter resets the reserved capacity to c).
- **What it implies for PF-A itself:** at a hit the restored state at c is the cold PF-A's own (the 5 PASS rows) and only
  the tail's path differs, so PF-A (mode 1) and the default prefill should differ in bits whenever a prompt's last
  1,024-token chunk is short (under ~200 tokens). The earlier PF-A bitwise records never had a short last chunk (30K:
  292; 6,980: 836). Direct check below (T1b's runner, step 2).
- Ops: 01:14-01:25 the queue waited on a stale research PAUSE (fanwatch's 61-s heat pause at T1's end; its clear kept the
  file because `pgrep -f quiet_window.sh` matched a mere command line). The orchestrator fixed fanwatch and removed it.

**T1b** (t1b_pfa.sh, armed 01:28, after the two quick checks; BigMoE agreed): my copy of convcache.py = 7eb3401 + (a)
the path chosen by the whole prompt, `if PFLM and N > chunk` (PF-A from c, also for a one-chunk tail) + (b) the check
names each differing leaf (cache index:class[slot], warm vs cold shape) (convcache_t1b.diff). Step 0: BigMoE's stand-in
(conv/test_mock.py, MOCK_PFLM=1) on the patched module. Step 1: the same replay. Step 2: PF-A vs the default prefill at
exact-32 in the 1.1.0 clean clone, P = 5 x 1,024 + 116 and + 300, R57_LOGHASH=1, 64 tokens. Then the cold 32 GB agent
test (pin 39947b1, unchanged), then 64K / 128K. The cached agent runs wait for a passing T1.

## Wed 01:27-01:30: idle CPU of the 1.1.0 serve.py: one core, the extension's loader spin (lever: BIGMOE_IDLE_MS)

idle_cpu.sh (the orchestrator saw the agent server at ~100 % CPU while idle): the v110 clean clone's serve.py at --mem 24
and 32, one 16-token request, 10 s rest, then CPU time over 60 s idle: **100 % of one core at both settings** (24 GB:
47.4 -> 107.5 s; 32 GB: 68.0 -> 128.1 s; 52 threads). A 5-s `sample` (idle_cpu_1111/busy_thread_32.txt) shows one busy
thread, entirely inside bigmoe_ext's loader (no system call on its stack): the loader thread's idle pass ends in
`__builtin_arm_yield()` (engine/ext/bigmoe_ext.cpp:1804), i.e. it spins by design ("CPU (native thread): spin on the
sequence number", line 8). The ext already has the lever: **BIGMOE_IDLE_MS=n** (Breakthrough 11-02, bigmoe_ext.cpp:1521-1526,
in all three builds here): after n ms without a served call the loader naps 200 us per pass instead of yielding; calls
within a request come far less than 50 ms apart, so decoding never sees it; the first call after a long idle waits at most
~0.2 ms; timing only, so exact. Neither release/src (1.1.0) nor release/next sets it. Proposed: BIGMOE_IDLE_MS=50 as an
env default in chat.py / serve.py, after an idle-CPU + decode-speed A/B (a ~5-min job, next free slot); 1.1.0 is frozen,
so adopting it there is the orchestrator's and Kamen's call.

## Wed 01:32-01:45: T1b: signature A fixed (7 of 12 PASS); B = one leaf, QSACache keys at cache index 35, after a copy

T1b = T1 again (same replay order) on my patched copy of convcache.py (sha256 00488b23...; convcache_t1b.diff: (a) the
path chosen by the whole prompt, (b) the check names differing leaves). Data: data/agent_cc_1111/t1b_32_1111/.
- **BigMoE's stand-in** (conv/test_mock.py, one QSA + one GDN layer with the real cache classes, MOCK_PFLM=1) on the
  patched module: rc 0, 4 PASS / 0 FAIL, hits as expected (including its copy at 9,216).
- **T1b: 12 hits, PASS 7, FAIL 5; 45 / 45 HTTP 200, 0 server errors.** Signature A is gone: the Codex hits with 196 and
  197 new tokens now pass (all 6 Codex hits pass), and the 116-token Claude Code hits no longer differ in logits, taps or
  the other 119 states. Warm prefill with the fix: 2.8 s for 116 new tokens, 3.8-5.6 s for 196-425, 12.1 s for 3,084
  (cold 28.0 s for 12,302 and 21.5 s for 8,390).
- **Signature B** (all 5 FAILs): `states (1 of 122: 35:QSACache[0] (1, 2, N, 256) vs (1, 2, N, 256))`: the attention
  KEYS of the QSA cache at cache index 35 (QSACache.state = (*kv.state, raw, pooled)); same shape, different values;
  values, raw, pooled and the other 121 leaves, the last logits and the taps tail are equal. Pattern: in the Claude Code
  conversation the first live hit (before any copy) passes; the first COPY (k 32, from 9,216) fails, and so does every
  later hit, including live hits on the MAIN entry (k 33 / 35 / 36) and the second copy (k 34). The Codex conversation
  never copied and passed all 6, live trims below a decoded reply included. So the copy changes or aliases the main
  entry's layer-35 keys (or the copy's prefill writes into a buffer shared with it); the stand-in's copy passes, so it
  takes something the real engine has. Sent to BigMoE (owner) for the fix; T1c on it next.

## Wed 02:1x: T1c armed (BigMoE 134e811: my tested module + a per-row drift diagnostic for signature B)

BigMoE committed my tested file as 91c9a5f (sha256 00488b23, byte for byte) and sent 134e811 (sha256 e3434b13): the same
plus a CHECK-only diagnostic: per-row integer prints of every QSA leaf (keys, values, raw, pooled) at each prefill's end,
re-checked at the next request's start, after a restore (rows below c vs the source's prefill) and after the prefill;
the FAIL line names the warm-vs-cold rows. Read before use: logs only, nothing else changes. Installed in the cache clone
(my 00488b23 kept as convcache.py.00488b23). t1c.sh: BigMoE's stand-in, then the T1b replay; run folder
t1_runs/32_1111c (raw convcache.log + serve.log kept). Placed after the cold agent test and the pkg_110g checks, before
64K / 128K (~15 min). How to read it (BigMoE): drift at a request's start = the decode changed stored rows; a restore line
= the copy's read; FAIL rows only in [c, N) = the warm prefill itself.

Cold 32 GB agent test so far: probe 27 / 30 pass, 0 invented tools (02:11, 762 s); the 12 tasks started 02:11.

## Wed 02:3x: T1d armed after T1c (BigMoE b855883: CONV_CACHE_BRANCH=0, the orchestrator's 09:00 fallback, checked early)

The orchestrator's rule (02:1x): if there is no clean fix for signature B by 09:00, a branch (copy) becomes a miss; T1 runs
again on that at once, and if it passes, the cached 32 GB agent test goes next. BigMoE's b855883 (sha256 266f76e5) =
134e811 + CONV_CACHE_BRANCH (default 1, unchanged; 0 = a would-be copy is a cold miss and a new entry; the decision now
runs before the check's cold prefill). Read before use: the switch only. t1d.sh: after T1c, installs b855883 in the cache
clone (sha checked from BigMoE's repo and in the clone), BigMoE's stand-in with BRANCH=0, then the T1b replay with
CONV_CACHE_BRANCH=0; run folder t1_runs/32_1111d. Expected (BigMoE): 10 hits (the 2 copies at 9,216 become misses), all
PASS, unless the main entry's own decode changes its layer-35 keys (T1c's drift lines decide that).

**Correction (02:3x, BigMoE's predict_hits.py 3a2dc46, which matches T1 / T1b hit for hit):** the T1b section above says
the later failing live hits (k 33 / 35 / 36) were on the MAIN entry. They were not: by the module's rules k 33 is a live
hit on entry 1 (the copy k 32 made; it shares more with k 32 than with k 31) and k 35 / 36 on entry 2 (the copy k 34
made). All 5 FAILs are on copy-made entries; the main entry (entry 0) passed its only live hit (k 31) and was never
restored live again; Codex (entry 3, live take-overs only) passed all 6. So signature B is either (a) the copy itself, or
(b) entry 0's layer-35 keys below 9,216 changing during k 31's decode, seen only by the copies. T1c separates them: (b)
-> a `drift (request start): entry 0 ... 35[0]` line at k 32; (a) -> `restore (copy at 9216): ... 35[0]` with no start
drift; neither -> FAIL rows only in [9216, N). T1d's prediction (BigMoE data/conv/t1_predicted_branch0.json): 10 hits,
all live, 41.3 % of prompt tokens prefilled.

## Wed 01:58-03:45: EvalSuite's cold 32 GB agent test (agentcc_runs/32_1111): ran, but Bash never worked (harness bug)

Pin: the agent clone = release/next 39947b1 (provenance.txt), serve.py --mem 32 (chat-32), claude 2.1.321, the sandbox
and launcher as in agentcc_night.sh. Records: agent_cc_1111/agentcc_32_1111/ (results, probe, serve.log, runner.log,
sandbox.sb, fanwatch_freezes.txt; tool_calls.jsonl, the per-call summary; see below). The raw request log (175 requests, 22 MB) is local
only: logs/agentcc_32_1111_server_requests.jsonl.gz (gitignored).
- **Probe (EvalSuite's 30 direct API calls): 27 / 30, 0 calls with an invented tool.** Fails: case 6 and case 10 (Bash, "case
  check failed"), case 25 (Bash instead of Write). EvalSuite: case 10's check was its own bug (it rejected `ls -la`
  looking for the substring "-a"); its pre-registered 32 GB line is 28, so the probe verdict as registered is FAIL.
- **Tasks: 11 / 12 checks passed, 0 timeouts, server 175 requests, 0 errors.** t09_refactor failed: the model extracted
  format_money() but kept the "$" in each caller, so format_money(1234.5) gives "1,234.50" (behavior-preserving, not
  what the check asks). Task walls 97-969 s, 5,619 s in all.
- **Harness bug: all 79 Bash calls failed** with `EPERM: operation not permitted, mkdir
  '/private/tmp/claude-501/<the task folder, dashed>'`: Claude Code makes a per-folder temp dir under
  /private/tmp/claude-<uid>/ before it runs any command, and the sandbox allows writes only in the run folder,
  /private/var/folders and /dev. So the 12 tasks ran with no working shell (the probe checks the call, not its run, so it
  was unaffected). EvalSuite found it at the same time (03:5x). Tool calls across the tasks (EvalSuite's count): Bash 79
  (all EPERM), Read 37, Glob 16, Edit 15, Grep 5, Write 2, Agent 1 (the subagent lane worked: exactstream-sub), ToolSearch 1.
- **What a lossy model does when the shell fails** (observed): retries with `dangerouslyDisableSandbox: true` (Claude
  Code's own sandbox is off here, so the flag changes nothing; the outer seatbelt still applies), `cd /tmp`,
  run_in_background, `ls -la /Users/kamen` (t12), and reading Claude Code's own config folder: t01 Read cc_home's
  .claude.json and its **shell snapshot (snapshot-zsh-*.sh, built from the user's ~/.zshrc: his aliases, including the
  private account names)**. All stayed on this Mac, inside the sandbox's allowed reads. The transcripts were first
  committed redacted (those two tool results withheld; the leak pattern found 0 hits). EvalSuite told (03:5x) not to
  commit that transcript or the request log as is. **Correction (Wed 07:3x):** under the 04:0x rule raw transcripts stay
  out of git. The transcripts left HEAD at 07:3x; agentcc_32_1111/tool_calls.jsonl is the per-call summary. The
  transcripts are local only, in logs/agentcc_transcripts/.
- **Cold-run timing** (no conversation cache: every turn prefills the whole prompt): probe prompts ~8.7K tokens, first
  token median 21.3 s; task prompts 9.1-20.2K (median 13.8K), first token median 32.6 s (~419 prompt tok/s); answers
  median 110 tokens, whole decode median 58 tok/s (short answers); lifetime peak 37.46 GB. The 7 first tokens over 60 s
  (103-124 s) are all fanwatch heat freezes: the server was frozen 7 times for ~80 s (02:08, 02:19, 02:32, 02:46, 03:02,
  03:16, 03:31; fanwatch_freezes.txt), ~9.5 min of the 94-min task phase.

## Wed 03:5x: the Bash fix (CLAUDE_CODE_TMPDIR) and the 12-task rerun at 32 GB armed (agentcc_runs/32_1111b)

- **Fix:** `CLAUDE_CODE_TMPDIR=<run>/cctmp` in Claude Code's environment (a documented variable in Claude Code's string
  table: it moves the per-uid temp root). Claude Code then makes its per-folder temp dir inside the run folder, which the
  profile already allows; the sandbox profile is unchanged (its generator is byte-identical; nothing is written outside
  the scratchpad). With a long override Claude Code keeps child processes' TMPDIR at the system default
  (/var/folders/.../T/, allowed), since a socket path there would be too long.
- **Mock check 03:54** (bashfix_check.sh + mock_bash.py = mock_tools.py with one Bash call; no GPU, < 1 GB; output
  bashfix_check.out): V0 (as 32_1111) gives the same EPERM; V1 (the fix) runs the command: TMPDIR=/var/folders/.../T/,
  a file written in the task folder, a Python tempfile written, a write to $HOME denied (no file appeared outside).
- **Rerun** (agentcc_rerun.sh, armed 03:55 after T1d's runner, one clear poll, 10-s polls; last start 08:00): the same
  server files (the clone checked against release/next's SHA256SUMS at 39947b1, not HEAD), serve.py --mem 32 on port 8124,
  the same manifest, base config and launcher, no probe; a guard stops the run at once if t01's transcript still holds
  the EPERM. EvalSuite's PREREG_agentcc amendment 2 (e0fea3a, 03:57) covers the fix, the new folder, its corrected case
  10 check and a scorer fix (the init event names the subagent tool Task, the API array Agent).

## Wed 04:0x: no account names in tracked files; the rerun re-armed with a regex read-deny (agentcc_rerun2.sh)

The orchestrator (04:0x): the sandbox profiles in the record named the two account config folders literally, and the
public sync's leak check refuses those strings. Fixed in every tracked file (git grep for them: 0 hits):
- the ~/.claude read-deny rules (three subpaths) are now one rule, `(regex #"^/Users/[^/]+/\.claude[^/]*(/|$)")`,
  covering every ~/.claude* entry, ~/.claude.json included (the old list did not cover ~/.claude.json, which holds the
  account's config). Applied by deny_regex.py to agentcc_night.sh, agentcc_cache.sh, agentcc_rerun.sh, bashfix_check.sh
  and, as a marked record copy, agentcc_32_1111/sandbox.sb;
- scripts/copyspan_lookup.py + copyspan_width.py read the second projects folder from COPYSPAN_DIRS; HANDOVER reworded.
- **Mock check 04:05** (bashfix_check2.out, the regex profile): claude rc 0, Bash runs, file + tempfile written, write
  outside denied, `~/.claude*`: 7 entries, 0 readable; ~/.ssh denied.
- **The rerun:** agentcc_rerun.sh (pid 38588, still waiting for T1d) was stopped by SIGKILL (no trap, so no runner.done
  for EvalSuite's watcher; its log kept as 32_1111b/runner_rerun1_armed_0355.log) and agentcc_rerun2.sh armed 04:05:46
  (task b2h18m0yw): the same folder, gate and settings, only the read-deny rule as above.
- Note for the next runs: the shell snapshot that t01 read lives in the run's own Claude Code config folder (cc_home),
  which Claude Code itself must read, so no read-deny can hide it; keeping the user's aliases out of it would mean a
  neutral zsh startup (ZDOTDIR), which changes PATH (python3) for the tasks: not done without EvalSuite's call.

## Wed 03:57-04:09: T1c (BigMoE 134e811): signature B is the copy itself, layer 35 QSA keys, row 0 only

Same setup as T1b (cache clone, serve.py --mem 32, CONV_CACHE_CHECK=1, the 45 recorded requests; t1c_32_1111/). BigMoE's
stand-in on the module first: 4 PASS / 0 FAIL. Replay: 45 requests, **12 hits, 7 PASS, 5 FAIL** (as T1b), server errors 0.
- **drift lines 0**: entry 0's layer-35 keys did not change during k 31's decode (BigMoE's (b) is out).
- **restore lines 2**, one per copy (k 32, k 34): `restore (copy at 9216): rows below c differ from the source's prefill:
  35[0] rows 0..0 (1 of 9216)`: BigMoE's (a), the copy itself.
- Every FAIL (copies k 32 / 34 at c 9,216, N 12,300; live k 33 / 35 at c 12,288, N 12,404; live k 36, N 12,627) is
  `35:QSACache[0] ... QSA rows warm vs cold: 35[0] rows 0..0 (1 of N)`: one row, row 0 of layer 35's QSA keys, made by
  the copy and carried into the copy-made entries' later live hits. k 31 (live on entry 0) PASS.
- Warm prefill: copies restore 81-114 ms + 3,084 tokens in 12.2-12.3 s; live hits restore 59-102 ms + 116-339 tokens in
  2.9-4.7 s (cold: 12,302 tokens 28.2 s, 8,390 tokens 23.1 s).
Sent to BigMoE 04:1x. T1d (b855883, CONV_CACHE_BRANCH=0: no copies) started 04:09:56; its stand-in 3 PASS / 0 FAIL.

## Wed 04:1x: a neutral shell startup for the next agent runs (orchestrator 04:0x): ZDOTDIR does not work; HOME does

The orchestrator's call: tonight's 32_1111b stays as armed; from the next runs on (cached 32 GB, 45 GB) the claude
processes get a neutral zsh startup that keeps today's exact PATH, without the user's aliases or functions; checked on
the mock first (python3, git and the task tools must resolve to the same binaries).
- **ZDOTDIR=<run>/zdot (bashfix_zdot.sh / .out): no effect.** PATH and all 32 compared items equal, but aliases 5 -> 5
  and functions 8 -> 8: Claude Code builds its shell snapshot by sourcing `$HOME/.zshrc` itself (its string table: the
  snapshot script's `source "<config file>" < /dev/null`; ZDOTDIR is only read by its `alias claude=` check), so
  ZDOTDIR never reaches it.
- **HOME=<run>/home with .zshrc = `export PATH='<the exact PATH of today's runs>'` (bashfix_home.sh / .out): works.**
  All 33 compared items equal (PATH, the binary of python3 / python / pip3 / git / sh / bash / zsh / ls / cat / grep /
  sed / awk / find / wc / sort / head / tail / diff / patch / make / env / xargs / tr / cut / node / npm / pytest,
  python3 3.14 (Homebrew), git's version, the locale and Python's encodings, the env var names); aliases 5 -> 2 (zsh's
  own run-help / which-command), functions 8 -> 0. Side effects: `~` is the neutral folder (so `ls ~` no longer lists
  the user's home; absolute paths still behave as before), and git has no global identity (proposed: a .gitconfig in
  the neutral home with the template's identity, agentcc@localhost). Not adopted yet: sent to the orchestrator and
  EvalSuite (an amendment) before the cached runs.
- **Approved (orchestrator 04:2x) for the cached 32 GB and 45 GB runs; EvalSuite told for an amendment.** The home
  template is agent_cc_1111/agent_home_tpl/ (.zshrc: only the exact PATH; .gitconfig: agentcc / agentcc@localhost);
  agentcc_cache.sh copies it to <run>/home, records it in provenance.txt and runs agentcc_tasks.py (so every claude) with
  HOME=<run>/home. **Mock check of exactly that setup (bashfix_final.sh / .out), in a git repo task folder:** 32 of 33
  items equal to today's (the one difference: `git status` ok, the baseline's mock folder was no repo); aliases 2 (zsh's
  own), functions 0; `git commit` works, author agentcc@localhost. Tonight's 32_1111b keeps the old home, as armed.

## Wed 04:09-04:23: T1d (BigMoE b855883, CONV_CACHE_BRANCH=0): PASS, 10 of 10 hits bitwise

The orchestrator's 09:00 fallback, checked early: b855883 (sha256 266f76e5) installed in the cache clone, BigMoE's
stand-in with BRANCH=0 first (3 PASS / 0 FAIL), then the T1b replay with CONV_CACHE_BRANCH=0 (t1d_32_1111/).
**45 requests; 10 hits, 10 PASS, 0 FAIL; drift 0, restore 0; server errors 0**, as BigMoE predicted: the 2 would-be
copies are cold misses (`miss N 12300 c 0 (none, branch (CONV_CACHE_BRANCH=0))`, 27.5 / 28.0 s), every hit is live (c
12,288: k 31 / 33 / 35 / 36; c 8,192: the 6 Codex turns), restore 58-93 ms, 116-425 new tokens prefilled in 2.9-5.7 s.
The first long miss (N 12,302) took 135 s instead of 28 s: fanwatch froze the server 04:15:35-04:17:16 (heat pause at
die 84.3 C; fanwatch_0415.txt). Short one-chunk misses (~700 tokens, the side calls) take ~8 s each at 32 GB.
So the cache is exact with copies off; with copies on, the copy changes row 0 of layer 35's QSA keys (T1c). Under the
orchestrator's rule the cached 32 GB agent test runs on b855883 + BRANCH=0 at 09:00 unless BigMoE has a clean copy fix.

## Wed 04:2x: git in 32_1111b's setup: `git commit` fails (the user's gitconfig signs with an SSH key under ~/.ssh)

EvalSuite asked (04:2x) whether git works for the model in 32_1111b (real HOME). Mock check of exactly that setup in a
git repo (bashfix_git.sh / .out; R = 32_1111b's, F = the cached runs' neutral HOME): R: `git status`, `git log`, `git
diff` ok (so Claude Code's own startup git status works), **`git commit` FAILS**: `error: Couldn't load public key
<home>/.ssh/id_ed25519.pub` (the global gitconfig turns on commit signing with an SSH key: commit.gpgsign, gpg.format,
user.signingkey; the profile denies ~/.ssh, so a model can never sign with the user's key). F: all four ok. (The earlier
"B: GIT status FAIL" in bashfix_final.out was the mock's task folder, not a repo.) None of the 12 prompts asks for a
commit. 32_1111b had already started (04:23:49) when the answer came, so it runs as armed (EvalSuite's option b: commit
failures reported as a setup limit of that rerun); agentcc_rerun3.sh (neutral HOME) was ready but not used. The cached
runs use the neutral HOME, where commits work.

## Wed 04:23-05:07: the 32 GB rerun with Bash working (agentcc_runs/32_1111b): tasks 11 / 12, 43 min (cold 94)

agentcc_rerun2.sh (records: agent_cc_1111/agentcc_32_1111b/; raw request log local only, logs/, gitignored). Same server
files as 32_1111 (the clone = release/next at 39947b1, checked against that commit's SHA256SUMS), serve.py --mem 32,
claude 2.1.321, the same manifest / base config / launcher / timeouts; changed: CLAUDE_CODE_TMPDIR=<run>/cctmp (the Bash
fix) and the ~/.claude* read-deny as one regex; the real HOME (EvalSuite's option b). No probe (EvalSuite: unaffected).
- **Tasks: 11 / 12 checks passed, 0 timeouts; server 68 requests, 0 errors.** Bash works: 30 calls, 0 EPERM (the guard
  saw t01 clean at 04:27:43). Tool calls: Bash 30, Read 21, Edit 15, Glob 2, Grep 2, Write 2; tool errors 2, both
  ordinary (t09 wrote a scratch file to /tmp, which the sandbox refuses as it should; t11 used GNU `cat -A`, which macOS's
  cat lacks). 3-16 turns per task. No `git commit` attempt (so 32_1111b's signing limit never mattered); no read outside
  the task folders (summarize_calls.py: outside_task 0); no private-pattern hit in the transcripts. **Correction (Wed
  07:3x):** the transcripts were committed as they were (05:09), against the 04:0x rule that raw transcripts stay out of
  git. They left HEAD at 07:3x; agentcc_32_1111b/tool_calls.jsonl is the per-call summary. The transcripts are local only
  (logs/agentcc_transcripts/). They were in the public snapshots of 05:25 and 06:24; the orchestrator has been told.
- **t09_refactor fails again, as in 32_1111:** format_money(amount) returns "1,234.50" and the callers keep the "$", so the
  output is unchanged and the format string appears once, but the check also asserts format_money(1234.5) == "$1,234.50",
  which the prompt ("Extract it into one helper function named format_money(amount) and use it in all three places.
  The output must not change.") does not state. A spec ambiguity both runs resolved the same way; told to EvalSuite.
- **Time:** task walls 127-364 s, 2,595 s in all (32_1111, Bash dead: 97-969 s, 5,619 s). Every turn is cold (no cache):
  prompts 12.5-17.0K tokens (median 13.2K), first token median 31.9 s (~416 prompt tok/s), answers median 110 tokens,
  whole decode median 61 tok/s, lifetime peak 36.69 GB. The 2 first tokens over 60 s (109 / 111 s) are fanwatch heat
  freezes (3 freezes of ~80 s: 04:32, 04:49, 05:05; fanwatch_freezes.txt). With the conversation cache a turn would
  prefill only its new tokens (T1d: 116-425 tokens in 2.9-5.7 s), the target of the cached runs.

## Wed 05:07-05:20: T1e (BigMoE 3c068d7, bitwise clones, branches ON): PASS, 12 of 12 hits bitwise

BigMoE (04:3x): T1c's signature B was the copy's `x + 0`: an add of zero is not a bit copy for floats (-0.0 + 0.0 =
+0.0; NaN bits rewritten; checked in MLX: 0x8000 -> 0x0000), and row 0 of layer 35's keys holds such a value, so the copy
changed its bits but not its value (logits, values and taps matched). 3c068d7 (conv/convcache.py sha256 55631cb8):
`_clone` copies bits (an integer add of zero on the same-size unsigned view), and convcache takes and restores the GDN
checkpoints itself with that clone (they used the same rule twice). t1e.sh (records t1e_32_1111/): the cache clone,
serve.py = cachepin (T1c's + BIGMOE_IDLE_MS=50, timing only), --mem 32, branches on, CONV_CACHE_CHECK=1, the 45 recorded
requests. Stand-in: MOCK_PFLM=1 4 PASS / 0 FAIL (the gate), MOCK_PFLM=0 4 / 0. **Replay: 12 hits, 12 PASS, 0 FAIL; drift
0, restore 0, server errors 0. BigMoE's verdict (t1_verdict.py at 3c068d7 -> t1e_32_1111/verdict.json): "PASS: 45 cache
lines for 45 requests; hits 12 (predicted 12); checks 12/12 PASS".** The census lines (CHECK-only) find, on every hit
(2 copies at 9,216, 4 live at 12,288, 6 live at 8,192), `35[0] rows 0..0 (1 rows, 1 values, first 0x8000)`: the -0.0
the old clone rewrote. So the cache is exact at 32 GB with branches on. Pin for the cached runs: release/next 80c8041
(serve.py = cachepin, engine/loop/convcache.py = 3c068d7; the cache clone's files equal it).

## Wed 07:1x: a 64K window for Claude Code (orchestrator 07:1x, option (b)): stand-in check

**What was asked.** The orchestrator chose (b) from data/long_ctx_1111.md: tell Claude Code 64K at `--mem 32` and `--mem 45`
for now. Before relying on it, check that Claude Code really compacts before 64K, and report it if Claude Code clamps the
window to 100K or more.

**Claude Code 2.1.321's code.** `CLAUDE_CODE_AUTO_COMPACT_WINDOW` accepts 100,000 to 1,000,000 tokens (constants
`pI=1e5`, `dW=1e6`). A smaller value is classed "below_minimum" and read as 100K. The settings schema says the same
("100000 to 1000000"). In `-p` mode this happened silently. Interactive mode goes through a `[autocompact] ...` warning
path ("Set it to a size from 100k to 1M"). `CLAUDE_CODE_MAX_CONTEXT_TOKENS` has no such floor; the run's own usage line
reports contextWindow 65536.

**Runs.** Real Claude Code on the stand-in engine (mock_tools.py), through the clone's `tools/claude_local.sh` with
`EXACTSTREAM_CONTEXT`. Setup:
- the night runs' sandbox, an isolated config folder and the neutral HOME;
- serve.py's limit left at 131,072, so a clamped window would show as a conversation growing past 64K;
- each turn a `claude -p --continue` with new text; the mock answers with one Write call, then "Done".

| window told (launcher) | step per turn | last turn without compaction (largest prompt) | auto compaction (pre -> post) |
|---|---|---|---|
| 65,536 (sets both variables) | ~6K tokens | turn 2 (25,487) | turn 3: 34,362 -> 11,563 |
| 65,536 (sets both variables) | ~1.6K tokens | turn 12 (30,365) | turn 13: 32,134 -> 4,463 |
| 65,536 (MAX_CONTEXT_TOKENS only; see below) | ~1.6K tokens | turn 12 (30,392) | turn 13: 32,161 -> 4,464 |
| 131,072 (sets both variables) | ~7K tokens | turn 11 (90,189) | turn 12: 99,115 -> 11,618 |

In every run the session went on after the compaction (one more turn each, rc 0).

**Answer.** With a 64K window, Claude Code compacts at about 32K tokens, well before 64K. The window variable is clamped
to 100K, but compaction follows the context window from `CLAUDE_CODE_MAX_CONTEXT_TOKENS`. Compaction comes about 33K below
the window Claude Code is told: about 32K at 64K, and between 90K and 99K at 128K. So a 64K session compacts at about
half its window. A turn that adds a lot, such as a large file read, can still take a prompt past 32K before the next
check.

**Launcher variant** (`agent_cc_1111/compact64/launcher_variant/tools/claude_local.sh`).
- It sets `CLAUDE_CODE_AUTO_COMPACT_WINDOW` only from 100K up, so 64K sessions avoid the clamp and its possible
  interactive warning.
- The same compaction point (32,161 against 32,134).
- Env check: at 65,536 only MAX_CONTEXT_TOKENS is set; at 131,072 both are.
- For release/next once the cached 45 GB runner has passed its provenance step (release/next stays frozen until then),
  together with a per-`--mem` default in tools/claude_code.sh: 65,536 at 32 and 45 GB, 131,072 at 24 GB, and
  `EXACTSTREAM_CONTEXT` still overrides.

Scripts and logs: agent_cc_1111/compact64/. The stream transcripts of these stand-in runs stay local.

## Wed 07:01-07:21: the cached 32 GB agent run (agentcc_runs/32c_1111): probe 29 / 30, tasks 11 / 12, 15 min (cold rerun 43)

**Records and setup.** Runner agentcc_cache.sh; records in agent_cc_1111/agentcc_32c_1111/. The raw request log is local
only (logs/agentcc_32c_1111_server_requests.jsonl.gz, gitignored). The setup is as registered (EvalSuite amendment 3
part 2):
- the cache clone = release/next at 80c8041: serve.py = cachepin d6b44ed0 + hook, engine/loop/convcache.py = BigMoE
  3c068d7 (sha 55631cb8), all 7 SHA256SUMS OK;
- serve.py --mem 32 with CONV_CACHE=1, CONV_CACHE_BRANCH=1 and CONV_CACHE_GB 2.5 (the 32 GB default);
- claude 2.1.321;
- the neutral HOME: .zshrc = the exact PATH only; git identity agentcc <agentcc@localhost>;
- CLAUDE_CODE_TMPDIR and the regex read-deny sandbox.

It started at 07:01:41, once long_ctx had no arm left that fitted (long_ctx was stopped by design), and ended at
07:21:08. There were no heat freezes in that window.

- **Probe: 29 / 30, 0 calls with an invented tool.** The one fail is case 6 (Bash, "case check failed"), which also
  failed in the cold run 32_1111 (27 / 30 there; cases 10 and 25 pass now).
- **Tasks: 11 / 12 checks passed, 0 timeouts; server 118 requests, 0 errors.**
  - t09_refactor fails as in both earlier runs: format_money returns "1,234.50" and the callers keep the "$", while the
    check wants "$1,234.50".
  - 3-15 turns per task. Tool calls: Bash 50, Read 21, Edit 16, Grep 3, Glob 2, Write 1.
  - 7 git commands, all read-only (diff, log, show, status, grep, check-attr). There was no commit, so the git identity
    was never used.
  - **Three calls went outside the task folder, all in t12** (#9-#11, Bash). EvalSuite found them; my first count looked
    only at file paths, not at Bash commands. Corrected at 07:3x.
    - #9 listed the run folder (`ls` of tasks/, `find <run> -maxdepth 2`).
    - #10 read its own check and stderr files in the run folder: the check file was still empty, and stderr held only
      the model warning.
    - #11 read its own live transcript.
    The model was looking for the grader and found nothing useful. EvalSuite reports it as information: scoring t12 a
    FAIL would give 10 / 12, still at the line. No call in any of the three runs read ~/EvalSuite, the manifest or a
    .agentcc_check.py (EvalSuite's regex scan).
  - The transcripts have 0 private-pattern hits and 0 account names. Under the 04:0x rule they stay out of git: local
    only in logs/agentcc_transcripts/. agentcc_32c_1111/tool_calls.jsonl is the per-call summary (summarize_calls.py:
    tool, error class, sandbox flag, outside_task; no inputs or outputs).
- **Tool errors: 4.**
  - t02: an `ls` of a missing name (exit 1).
  - t05: `python` (macOS has python3 only; exit 127).
  - t12: GNU `cat -A`, which BSD cat lacks.
  - t09: **a heredoc in Bash failed with "can't create temp file for here document: operation not permitted".** zsh
    writes heredoc temp files under TMPPREFIX (default /tmp/zsh), and the sandbox allows writes only in the run folder,
    /private/var/folders and /dev. This is a setup limit, like 32_1111's EPERM but much smaller: one call, and the model
    carried on.
  - A fix for later runs: `export TMPPREFIX="$TMPDIR/zsh"` in the neutral .zshrc, since the Bash tool's TMPDIR is under
    /private/var/folders. It needs a stand-in check, and EvalSuite and the orchestrator must agree first. It is not
    applied to the armed 45 GB run, whose setup is registered.
- **Time: task walls 29-171 s, 916 s in all, against 2,595 s in the cold rerun 32_1111b (2.8x).**

| task | cached (s) | cold rerun (s) | check |
|---|---|---|---|
| t01_fix_test | 57.0 | 214.1 | pass |
| t02_add_function | 92.3 | 364.4 | pass |
| t03_config | 42.5 | 153.7 | pass |
| t04_traceback | 53.3 | 127.2 | pass |
| t05_rename | 78.9 | 269.3 | pass |
| t06_cli_flag | 47.3 | 167.7 | pass |
| t07_import | 83.2 | 205.4 | pass |
| t08_readme | 29.3 | 171.8 | pass |
| t09_refactor | 98.1 | 297.0 | fail (both) |
| t10_type_hints | 57.7 | 199.3 | pass |
| t11_shell | 104.5 | 248.1 | pass |
| t12_requirements | 171.4 | 177.2 | pass |

- **Cache: 117 of 118 requests hit (106 live, 11 copy).**
  - The one miss was the first probe request: 8,738 tokens cold, first token 20.4 s.
  - Prompts: median 13.0K tokens, max 30.2K.
  - New tokens prefilled per request: median 664, p90 3,294, max 11,248.
  - **First token: median 6.3 s, p90 12.7 s, max 36.4 s** (the max is the 11,248-token prefill). The cold rerun's median
    was 31.9 s.
  - Answers: median 100 tokens; whole decode median 72.9 tok/s.
  - Lifetime peak 37.07 GB (cold rerun 36.69). At the end the cache held one entry: 2.58 GB, 29.7K tokens.
- **Provenance** (provenance.txt) has everything EvalSuite's amendment 3 part 2 asks for:
  - the pin's sums: release/next's 7 files OK; the clone = release/next for serve.py, agentio.py, the two local
    launchers and engine/loop/convcache.py; commit 80c8041;
  - the neutral HOME with its PATH line;
  - agentcc <agentcc@localhost>;
  - CONV_CACHE_BRANCH=1.

## Wed 07:3x: the zsh heredoc limit, and TMPPREFIX for the runs after the cached 45 GB one

**The limit.** In 32c_1111, t09 hit it: zsh writes heredoc and here-string temp files under TMPPREFIX (default /tmp/zsh),
and the sandbox does not allow writes there.

**Approvals.** The orchestrator (07:3x) approved TMPPREFIX pointing at the run's own temp folder, for runs after the
armed 45 GB one, with an EvalSuite amendment first. EvalSuite agreed for later runs, with a dated amendment. The 45 GB run
stays as registered, and agent_home_tpl is unchanged (the armed runner copies it).

**Stand-in checks** (mock_bash.py, the night runs' sandbox, CLAUDE_CODE_TMPDIR, HOME=<run>/home from the template; the
command runs a heredoc and a here-string, then prints TMPPREFIX, PATH, the git identity and the alias count):

| variant | heredoc / here-string | TMPPREFIX in the Bash tool | .zshrc |
|---|---|---|---|
| H: as registered | both fail ("can't create temp file for here document") | /tmp/zsh | df313e2ac75d (PATH only) |
| T: `export TMPPREFIX='<run>/cctmp/zsh'` added to .zshrc | **both still fail** | **/tmp/zsh** | 677818f72d7c |
| E: `TMPPREFIX=<run>/cctmp/zsh` in the claude process environment | **both work** | <run>/cctmp/zsh | df313e2ac75d (unchanged) |

- In all three variants the PATH is identical, the git identity is agentcc <agentcc@localhost>, and there are 2
  aliases.
- After E, zsh left 0 temp files in cctmp.
- So a `.zshrc` export does not reach Claude Code's Bash tool: its shell snapshot did not carry TMPPREFIX here. The
  environment does, because zsh takes TMPPREFIX from its environment (also checked directly: `env TMPPREFIX=... zsh -f`
  runs a heredoc in that folder).
- **The form for the amendment:** `TMPPREFIX=<run>/cctmp/zsh` in the claude process environment (the runner's `env`
  line, beside CLAUDE_CODE_TMPDIR), with the .zshrc unchanged.

Scripts and outputs: agent_cc_1111/heredoc/ (the PATH lines are replaced by a note).

## Wed 07:21-07:45: T1-45 (BigMoE 3c068d7, branches ON, --mem 45, CONV_CACHE_GB 2.5): PASS, 12 of 12 hits bitwise

**Setup.** Runner t1_45.sh; records in t1_45_1111f/. It is T1e's setup at 45 GB:
- the cache clone with serve.py = cachepin (d6b44ed0) and engine/loop/convcache.py = 3c068d7 (sha 55631cb8);
- `serve.py --mem 45`, CONV_CACHE_BRANCH=1, CONV_CACHE_GB=2.5, CONV_CACHE_CHECK=1;
- the 45 recorded requests.

It started at 07:21:15, after the cached 32 GB run (NO-SWAP 52 GB).

**Stand-in.** MOCK_PFLM=0 (the gate at 45 GB, chunk-major): 4 PASS / 0 FAIL. MOCK_PFLM=1 (information): 4 / 0.

**Replay: 45 requests; 12 hits, 12 PASS, 0 FAIL; drift 0, restore 0, server errors 0.**
- BigMoE's t1_verdict.py at 3c068d7 (written to t1_45_1111f/verdict.json): "PASS: 45 cache lines for 45 requests; hits 12
  (predicted 12); checks 12/12 PASS, fail []; not as predicted []; prefilled 50607 of 167343 tokens; max cache 2.067 GB".
- On every hit (2 copies at 9,216, 4 live at 12,288, 6 live at 8,192) the census finds `35[0] rows 0..0 (1 rows, 1 values,
  first 0x8000)`, the same -0.0 as at 32 GB.
- **So the cache is exact at 45 GB too, with branches on.**

**Timing at 45 GB** (with CHECK on, so it is information only):
- hits prefilled 116-3,084 new tokens (median 293) in 3.0-23.2 s (median 4.9 s); the two 3,084-token copy hits took about
  23 s;
- the 2 misses (8.4K and 12.3K tokens, cold) took 61-89 s, about 138 tok/s with 45 GB's own prefill (PF-D);
- serve.py's highest lifetime peak was 50.02 GB. The cached 45 GB run's gate is therefore NO-SWAP >= 54 GB (peak + 3,
  rounded up).

## Wed 07:47-08:11: the cached 45 GB agent run (agentcc_runs/45c_1111): probe 29 / 30, task checks 12 / 12, but t10 read other tasks' checks

**Records and setup.** Runner agentcc_cache45.sh; records in agent_cc_1111/agentcc_45c_1111/. The per-call summary is
tool_calls.jsonl; the transcripts and the raw request log are local only, in logs/. The setup is as registered (EvalSuite
amendment 3 part 3):
- the cache clone = release/next at 80c8041, with all sums OK;
- `serve.py --mem 45` with CONV_CACHE_GB 2.5 and CONV_CACHE_BRANCH=1;
- 3,000 s timeouts;
- the neutral HOME (agentcc <agentcc@localhost>) and the same sandbox.

It started at 07:47:12 on T1-45's PASS, with the gate at NO-SWAP 54 GB (= T1-45's peak 50.02 + 3) and NO-SWAP exactly
54. There was no swap growth and no heat freeze. Amendment 3 part 4 (EvalSuite, 07:27), in before the start, says a call
naming a check fails its task.

- **Probe: 29 / 30, 0 invented tools.** Case 6 fails again (Bash, "case check failed").
- **Task checks: 12 / 12 passed, 0 timeouts; server 118 requests, 0 errors.** t09_refactor passed for the first time in
  four runs.
- **t10_type_hints left its task folder in 7 Bash calls (#6-#12)**, all succeeding:
  - #6 listed the run folder and tasks/.
  - #7 listed two finished tasks' folders.
  - **#8 printed two finished tasks' `.agentcc_check.py` (t09's and t01's).**
  - #9-#11 read provenance.txt, tasks.out and results.jsonl.
  - #12 ran a `find` of the run folder for "*check*" (described as "Look for check script for t10").

  t10's own check is copied in only after claude ends, so it was never readable. But under amendment 3 part 4 these calls
  name checks, so EvalSuite will score t10 a FAIL, which gives tasks 11 / 12. Scoring is EvalSuite's.
  - This is the gap EvalSuite raised at 07:2x: the profile allows reads of the run folder, and earlier tasks keep their
    `.agentcc_check.py`. For the runs after this one, agentcc_tasks2.py denies reads of ~/EvalSuite/agentcc and of the
    run folder outside the task's own folder, its config folder, cctmp and home, with one profile per task. It is to be
    checked on the stand-in (profile_check.sh) and registered as an amendment before use.
- **Tool calls.** Bash 51, Read 21, Edit 15, Grep 2, Write 2, Glob 1; 3-16 turns per task.
- **Tool errors: 4.**
  - t07: command not found.
  - t09: the zsh heredoc temp file (the TMPPREFIX limit) and one "operation not permitted".
  - t11: one "operation not permitted".
- **Time.** Task walls 42-205 s, 1,097 s in all (32c: 916 s). t10's 205 s includes its search.
- **Cache: 117 of 118 hits (107 live, 10 copy).**
  - The one miss is again the first probe request: 8,738 tokens cold, **72.5 s** at 45 GB (32c: 20.4 s).
  - Prompts: median 13.0K tokens, max 22.7K. New tokens per request: median 746, p90 3,292. 8.8 % of prompt tokens were
    prefilled.
  - **First token: median 6.8 s, p90 26.3 s, max 72.5 s.** 45 GB's own prefill (PF-D) is slower than 32 GB's PF-A on
    larger new chunks: a 4,321-token hit took 34.5 s.
  - Answers: median 102 tokens; whole decode median 77.7 tok/s (32c: 72.9).
  - Lifetime peak 49.43 GB. The cache was at most 2.57 GB.

## Wed 08:12-08:13: the per-task read-deny profile for the runs after 45c (EvalSuite 07:2x): stand-in check PASS

**What changed.** agentcc_tasks2.py (in profile/, with agentcc_tasks2.diff against agentcc_tasks.py) builds one profile
per task:
- the run's sandbox.sb, plus `(deny file-read-data ...)` of ~/EvalSuite/agentcc, of the manifest's folder and of the run
  folder;
- then `(allow file-read-data ...)` of the task's own folder, its config folder, cctmp and home. File contents and
  listings are denied; metadata stays readable, so path lookups through the run folder still work.

It also sets `TMPPREFIX=<run>/cctmp/zsh` in claude's environment, and the check runs under the same per-task profile.

**Stand-in test.** profile_check.sh on mock_bash.py, with two scratch tasks; each task's Bash call probes and prints labels
only:

| probe | base profile (agentcc_tasks.py, control) | per-task profile (agentcc_tasks2.py) |
|---|---|---|
| write + read its own folder, git status | ok | ok |
| <run>/tasks (siblings) | READ_OK | **DENIED** |
| the other task's folder | READ_OK (t00b; t00a ran before t00b existed) | **DENIED** |
| <run>/runner.log | READ_OK | **DENIED** |
| its own transcript <run>/<id>.jsonl | READ_OK | **DENIED** |
| ls ~/EvalSuite/agentcc, its manifest.json | READ_OK | **DENIED** |
| heredoc | fails (TMPPREFIX /tmp/zsh) | **ok** |
| Claude Code run, then the task's check | success, CHECK_OK | success, CHECK_OK |

**Use.** It goes into the next runs only after EvalSuite registers it as an amendment, together with the TMPPREFIX env
form. With it, 32c's t12 and 45c's t10 reads would have been refused.

## Wed 08:2x (main account): the window defaults in release/next (orchestrator option (b)), stand-in check PASS

**What changed.** release/next/tools now has next_patch/'s three files (diff: next_patch/release_next.diff), with
SHA256SUMS updated (all 7 OK):
- claude_code.sh and codex_code.sh set `EXACTSTREAM_CONTEXT` by `--mem`: 65,536 at 45 and 32 GB, 131,072 at 24 GB.
  A value the user sets still wins.
- claude_local.sh sets `CLAUDE_CODE_AUTO_COMPACT_WINDOW` only from 100K up, because Claude Code reads a smaller value as
  100K.

The cache clone's tools were updated to match; the old copies are kept as tools/*.sh.80c8041. The 80c8041 pin of the
night's runs is unchanged as a commit.

**Stand-in check** (window_defaults/: one mock server on its own port, reused by the launchers; real Claude Code, with
an isolated config folder and the neutral HOME). **Correction (08:2x):** the installed Claude Code updated itself to
2.1.322 at 08:15, so this check ran 2.1.322, not 2.1.321 (the init events say so). The 07:1x compaction checks and the
08:12 profile test ran 2.1.321.

| launcher, setting | what Claude Code / Codex got |
|---|---|
| claude_code.sh --mem 45 | contextWindow 65536 |
| claude_code.sh --mem 32 | contextWindow 65536; env: MAX_CONTEXT_TOKENS=65536, no AUTO_COMPACT_WINDOW |
| claude_code.sh --mem 24 | contextWindow 131072; env: both 131072 |
| EXACTSTREAM_CONTEXT=100000 claude_code.sh --mem 32 | contextWindow 100000 (the override wins) |
| codex_code.sh --mem 45 / 32 | model_context_window 65536, model_auto_compact_token_limit 58982 |
| codex_code.sh --mem 24 | model_context_window 131072, model_auto_compact_token_limit 117964 |

- Each Claude Code run ended with error_max_turns, as expected: the mock answers with a tool call, and the run allowed
  1 turn.
- The launchers reused the running server and left it alone. The test then stopped it, and the port was free.
- Codex gets the same defaults as Claude Code for the same memory reason. That extension is Leap's, flagged to the
  orchestrator.
- With 64K, Claude Code compacts at about 32K (the 07:1x check above); Codex compacts at 90 % of its window.

**Re-check on Claude Code 2.1.322** (Kamen's version since 08:15), with release/next's new claude_local.sh at a 64K
window (compact_check_v3.sh, 1.6K steps): auto compaction at turn 13, 32,243 -> 4,465 tokens, after which the session
goes on, the same as 2.1.321 (32,134 / 32,161). In 2.1.322's binary the clamp is unchanged: `pI=1e5,dW=1e6` and the
below_minimum code (compact64/check_v3_cc2322_65536.log). A verified copy of 2.1.321 is kept in
logs/claude_versions/2.1.321 (gitignored; sha256 abc9feeeca8091d5...) for the 32c2 rerun, which EvalSuite registered
on 2.1.321.

## Wed 08:2x: 32c2, the cached 32 GB rerun with the per-task profile (EvalSuite amendment 4 + 4a): runner ready, not armed

**Registration.** EvalSuite registered the per-task profile as PREREG_agentcc amendment 4 (ad2e532, arm "32c2": the
cached 32 GB confirmation rerun the orchestrator asked for, most likely tonight).
- Leap found that the registered agentcc_tasks2.py hardcodes `CLAUDE_BIN=/Users/kamen/.local/bin/claude` in claude's
  environment. Since 08:15 that path is Claude Code 2.1.322, so the run could not pin 2.1.321.
- EvalSuite chose the fix in amendment 4a (66a608e): **agentcc_tasks3.py**, which is agentcc_tasks2.py with only that
  line reading CLAUDE_BIN from the environment (profile/agentcc_tasks3.diff; sha256 b6e24c30d57ea13b...).

**Pins and checks.**
- Claude Code is pinned to Leap's copy at logs/claude_versions/2.1.321 (sha256 abc9feeeca8091d5..., 323,686,048 bytes,
  equal to versions/2.1.321).
- Stand-in pass with agentcc_tasks3.py and that binary (profile/profile_check_tasks3.out): both tasks' init events read
  claude_code_version 2.1.321, and every profile probe behaves as before.

**Runner agentcc_cache2.sh.** The file is agentcc_cache2.sh here, and the copy that runs is in the scratchpad (same
sha256). It is as 32c, plus:
- the 80c8041 server files, checked against 80c8041's SHA256SUMS, and the 80c8041 launcher (tools/claude_local.sh.80c8041,
  e5b900fa);
- a frozen, sha-checked copy of agentcc_tasks3.py, and the pinned CLAUDE_BIN with its sha256, size and `--version` in
  provenance;
- TMPPREFIX and an explicit CONV_CACHE_GB=2.5 in provenance;
- a stop if t01's init event is not 2.1.321, and a count of 2.1.321 transcripts at the end;
- NO-SWAP >= 45 and the swap guard.

It starts only when <run>/GO exists (made after EvalSuite's dated line), inside the orchestrator's window.

**Dry run.** DRYRUN=1 wrote provenance with every pin check passing (profile/provenance_32c2_dryrun.txt).

**Status.** Not armed yet: it waits for the orchestrator's window.

## Wed 08:24-08:30: a sandbox hole in the 32c2 setup (security review of 68ec944) and the fix (agentcc_tasks4.py / agentcc_cache3.sh)

**The hole.** The base profile let the agent write anywhere in <run>. The runner (agentcc_tasks*.py) runs unsandboxed, so
an agent-planted link could make it write outside the sandbox in three ways:
- **Runner outputs:** it opened its outputs in <run> with a plain open(..., "w"): the transcript, stderr, check.txt,
  results.jsonl and sandbox_<id>.sb.
- **The check copy:** after claude ended it shutil.copy'd the check to <task>/.agentcc_check.py, a folder the agent
  writes to.
- **HOME (<run>/home):** HOME was writable. The runner runs the next task's `git init/add/commit` unsandboxed with that
  HOME, so a planted .gitconfig could have run code.

So a planted link could turn a runner write into an overwrite outside the sandbox, against Kamen's rule that a bad tool
call never reaches outside the test folder. The 32c2 runner (agentcc_cache2.sh) was armed at 08:23 but waiting for its GO
file. It was stopped at 08:24 before it started. EvalSuite voided amendment 4b and holds 32c2 for the fix.

**The night's runs.** 32_1111, 32_1111b, 32c and 45c have no symlinks. EvalSuite also found default .git/config files
everywhere and no non-sample hooks.

**The fix** (agentcc_tasks4.py, from tasks3; profile/agentcc_tasks4.diff):
- Per task, the agent writes only in its own task folder, its config folder and its own temp folder <run>/cctmp/<id>,
  plus /private/var/folders and /dev from the base. CLAUDE_CODE_TMPDIR and TMPPREFIX point to <run>/cctmp/<id>, and HOME
  is read-only.
- The base profile may not name <run>; the script refuses one that does.
- Every runner output is opened with O_NOFOLLOW (O_EXCL when new).
- The check copy refuses a non-plain task folder or name, removes any entry at the check's name, and writes with
  O_CREAT|O_EXCL|O_NOFOLLOW.
- The task's process group is also stopped after a normal end.
- The check runs under the same per-task profile.
- After claude starts, nothing unsandboxed runs git, or reads or executes anything, in a task folder except that guarded
  copy. The next task's git runs on a fresh template copy, with a HOME the agent cannot write.

agentcc_cache3.sh (from cache2; profile/agentcc_cache3.diff) does the following:
- it writes the base profile without <run>;
- it runs a frozen tasks4 copy, checked against its sha;
- it lists the full per-task write set and the read denies in provenance;
- it refuses an existing run folder;
- at the end it logs `find <run> -type l`.

**Stand-in** (profile_check3.sh; two tasks; CLAUDE_BIN = the 2.1.321 copy; a canary file outside the run folder):

| probe or outcome | tasks4 + its base (profile_check3_new.out) | tasks3 + 32c2's base, the control (profile_check3_old.out) |
|---|---|---|
| write own folder, git commit, own temp folder | OK | OK |
| write a file in <run> | DENIED | OK |
| link at the next task's transcript name (<run>/t00b.jsonl) | DENIED | OK: t00b's transcript went into the canary |
| write in HOME | DENIED | OK |
| write in the other task's temp folder | DENIED | (shared folder) |
| link at <task>/.agentcc_check.py to the canary | made, then removed by the runner | made, then followed |
| canary at the end | **UNCHANGED** | **OVERWRITTEN** |
| checks | pass (the check ran from a real file) | fail (the check name stayed a link) |
| init events | 2.1.321, 2.1.321 | 2.1.321, (t00b's transcript lost via the link) |
| reads, as profile_check.sh; heredoc | denied as before; OK | - |

**Dry run.** DRYRUN=1 provenance passes every pin check (profile/provenance_32c2b_dryrun.txt).

**Fresh folder:** agentcc_runs/32c2b.

## Wed 08:3x: a second review of tasks4, and agentcc_tasks5.py / agentcc_cache4.sh

The second review raised three points.

**(1) The check copy was path-based.** It ran lexists, then unlink, then an O_EXCL|O_NOFOLLOW open. It already failed
safe, but tasks5 does the whole copy through a directory fd: open with O_DIRECTORY|O_NOFOLLOW, fstat, then stat, unlink
and open relative to that fd.

**(2) Processes outside the task's process group.** Processes that setsid or daemonize could outlive claude and tamper
with the task's check.
- Reading another process's environment is not possible here: `ps -E` shows none, and KERN_PROCARGS2 returns 29 bytes.
  So tasks5 finds the task's processes by `sandbox_check(pid, NULL, 0) == 1` (sandboxed) together with lsof: a cwd or an
  open file in the task's folders. The unsandboxed serve.py is never a match.
- It kills them in rounds until none are left (logged to <run>/lingering.jsonl).
- **The check now runs on the runner's copy of the task folder**, <run>/checks/<id>. The copy goes through O_NOFOLLOW
  directory fds at every level, keeps links as links and mode bits / times, and skips FIFOs and other file types. It runs
  under <run>/sandbox_check_<id>.sb, whose only writable folder is that copy (+ /private/var/folders, /dev).
- So a process that escapes the sweep can still write in the agent's own folders, but not in what the check grades.
- All 12 checks run `/usr/bin/python3 -I -B .agentcc_check.py`, and t11's runs `/bin/sh count.sh`. None uses git,
  absolute paths or `__file__`, so the copy works for them.

**(3) Check integrity: the agent's modules are imported by the check.** This is EvalSuite's to decide. It keeps the checks
byte-identical for 32c2 and pre-registers a scoring-time screen (tools/agentcc_screen.py): new top-level importable files
and exit / import-machinery lines. There were 0 flags in the four finished runs.

**Stand-in** (profile_check4.sh, tasks5, CLAUDE_BIN = the 2.1.321 copy) adds two racers from the task's Bash call:
- **R1:** setsid, stays in the task folder, after 8 s writes a file and plants a link.
- **R2:** setsid, chdir to /, after 15 s writes into the task folder by path.

| check | result |
|---|---|
| R1 killed by the sweep (lingering.jsonl: one pid per task) | yes; raced1.txt and its link never appeared |
| R2 escaped the sweep and wrote raced2.txt into the task folder | yes |
| raced2.txt in the check copy | no |
| check file in the copy is a plain file | yes |
| checks | CHECK_OK |
| canary | UNCHANGED |
| init events | 2.1.321 x 2 |
| other probes | as in profile_check3 |

The links the agent itself planted in its task folder stay there; they are inert, because the runner never follows them.

**Files.** agentcc_tasks5.py (profile/agentcc_tasks5.diff against tasks4) and agentcc_cache4.sh
(profile/agentcc_cache4.diff against cache3; tasks5's sha pinned). DRYRUN=1 provenance passes every pin check
(profile/provenance_32c2b_dryrun4.txt). The folder 32c2b is still unused.

## Wed 08:4x: a third review of tasks5 (hardening for later runs); 32c2 armed as registered (amendment 4c)

**Registration.** EvalSuite registered amendment 4c (bc78043, 08:40:19): cache4 (ff16664f) + tasks5 (28caec4c) on
agentcc_runs/32c2b. Its validity rules:
- both write sets in provenance;
- the per-task sandbox_<id>.sb and sandbox_check_<id>.sb files;
- any link outside the task's own tasks/, checks/, cc_home/ and cctmp/ folders voids the run;
- the checks are unchanged, with EvalSuite's integrity screen at scoring, which diffs checks/<id>.

**The third review raised three more points.** Leap judged none of them an escape for this run, so 32c2 was armed as
registered, and the points go to later runs.
1. **The sweep finds lingering processes by cwd and open files in the task's folders.** A process that setsid's, chdirs
   away and holds no files survives. It stays sandboxed (no writes outside its own folders, no reads of the run) and
   cannot touch checks/<id>, so this is integrity, not escape.
   - Killing every sandboxed process of the user is out: it would hit system XPC services and apps.
   - A per-profile fingerprint through sandbox_check's path queries does not work here. ctypes' variadic call returned 1
     for everything, and so did a compiled C helper (profile/sbcheck.c), even for unsandboxed processes. Only
     `sandbox_check(pid, NULL, 0)` (sandboxed or not) is reliable.
   - Later options: a dedicated uid per task, or recording every pid the task's processes create.
2. **The snapshot keeps links as links.** A link in checks/<id> to a shared writable place (/private/var/folders) that an
   escaped process controls could change what the check reads. This is integrity only, since the check runs the agent's
   code anyway. Fix: skip links in the snapshot.
3. **/private/var/folders (the user's shared temp) is in the agent's write set**, as in every run so far (the original
   night setup). An agent could plant links there for other unsandboxed processes. Fix: a per-task TMPDIR with
   /var/folders denied, which needs a Claude Code test first; Claude Code sets the Bash tool's TMPDIR under /var/folders.

**Armed.** agentcc_cache4.sh on 32c2b at 08:41:27 (pid 91419), with GO at 08:41:39 and the watchdog_1345.sh watchdog
(TERM at 13:40). After the run, Leap audits the Mac for leftover sandboxed processes started during it.

## Wed 08:43-09:03: 32c2, the cached 32 GB confirmation rerun with the per-task profile (amendment 4c): probe 29 / 30, tasks 11 / 12

**Records and setup.** Runner agentcc_cache4.sh (ff16664f) on agentcc_runs/32c2b, with agentcc_tasks5.py (28caec4c) and
Claude Code pinned to 2.1.321. Records are in agent_cc_1111/agentcc_32c2b/: results, probe, provenance, runner and serve
logs, convcache.log, checks_all.txt, stderr_all.txt, tool_calls.jsonl, and every sandbox_<id>.sb / sandbox_check_<id>.sb
in profiles/. The transcripts and the raw request log are local only, in logs/.
- GO at 08:43:47 (NO-SWAP 49 GB); serve.py ready 08:44:06; done 09:02:39 (NO-SWAP 52 GB, no swap growth).
- The watchdog was not needed.
- Provenance: every pin check passed (the 80c8041 server files and launcher, the frozen tasks5, CLAUDE_BIN's sha256 /
  size / `--version` 2.1.321), and it lists both write sets.

- **Probe: 29 / 30, 0 invented tools** (case 6 again).
- **Tasks: 11 / 12 checks passed; server 123 requests, 0 errors.**
  - t09 passed. In 32c it failed on the format_money "$".
  - **t11_shell failed.** The model looped: the same command six times (#20-#25), then the 25-turn cap (error_max_turns,
    rc 1). It left count.sh printing 14 where the check wants 7.
  - Not the harness: its one /tmp write was refused in 32c too, and it worked in its own folder instead. 32c's t11
    passed. With paths and timestamps in the prompts differing between runs, the trajectories differ.
  - Task walls 29-192 s, 875 s in all (32c: 916 s).
- **Validity items:**
  - transcripts with claude_code_version 2.1.321: **12 of 12**;
  - symlinks in the run folder: **0**;
  - lingering processes killed: **0**;
  - per-task profiles: 12 + 12 check profiles, kept.
- **The profile at work.** t09's Grep tool tried to search the whole run folder and was refused ("could not be opened
  (EPERM)"), its only call outside its folder; it then passed. t09's and t11's /tmp writes were refused as before.
- **Tool calls.** Bash 50, Read 21, Edit 15, Write 4, Grep 4, Glob 2. Tool errors 3: t07 command not found; t09 the
  refused Grep and a "no such file".
- **Cache: 122 of 123 hits (111 live, 11 copy).**
  - The miss was the first probe request.
  - New tokens per request: median 636, p90 3,284; 8.2 % of prompt tokens prefilled.
  - **First token: median 6.3 s, p90 12.4 s, max 21.0 s** (32c: 6.3 / 12.7 / 36.4).
  - Answers median 91 tokens; decode median 72.7 tok/s; lifetime peak 36.17 GB; the cache was at most 2.50 GB.
- **After the run, the Mac.** 224 of the user's 621 processes are sandboxed (system services). The only sandboxed ones
  started since 08:43 are two macOS services (CoreParsec, Spotlight's Metadata worker) and the audit's own `ps`. Nothing
  from the run is left, and port 8122 is free.

**EvalSuite's verdict (b1a2154): AGENTCC 32c2 PASS, confirmed 2 of 2** (32c and 32c2), with every validity item clean and 0
flags from the integrity screen. EvalSuite notes that the 30 probe replies equal 32c's exactly (a deterministic replay), so
the confirmation rests on the tasks.
