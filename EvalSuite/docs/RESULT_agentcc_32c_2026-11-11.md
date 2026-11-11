# RESULT: Claude Code with our local model at 32 GB with the conversation cache (arm 32c, Wed 2026-11-11 07:01-07:21) — PASS: probe 29 / 30 (line 28), tasks 11 / 12 (line 10)

Pre-registered in docs/PREREG_agentcc_2026-11-11.md before the run, as amendment 3 parts 1 and 2 (a514d09, 8587639)
under amendments 1-2a. Leap's runner ran it in Leap's sandbox. EvalSuite scored it with tools/agentcc_score.py.

**Verdict [K]** (results/agentcc_32c/agentcc_score_32c.txt):
- **Probe:** 29 / 30 pass, 0 calls with an invented tool. PASS: the line is at least 28 / 30 and 0 invented.
- **Tasks:** 11 / 12 pass (t09_refactor failed, wrong edit). PASS: the line is at least 10 / 12.
- **AGENTCC 32c: PASS.** Claude Code works with this model at 32 GB with the conversation cache, by the registered
  lines.
- The 32 GB verdict without the cache stays FAIL (32_1111, docs/RESULT_agentcc_32_2026-11-11.md).

**Disclosure (part 2) [K]:** the probe half is independent evidence, not a replay of 32_1111.
- 22 of the 30 tool calls equal 32_1111's (name and input). 19 of 30 whole replies (text and calls) are equal, and 28
  of 30 use the same tool names.
- The verdict does not depend on amendment 1's case 10 fix. Case 10's call was `ls -la /work/demo` again, so under the
  original check (the substring "-a") it would fail, and the probe would be 28 / 30, still at the line.

## Validity and provenance [K]
- **Pin:** provenance.txt shows release/next's 7 sums OK and commit 80c8041. The clone's serve.py is d6b44ed034db...,
  agentio.py 60e4520efc8d..., tools/claude_local.sh e5b900fae5fa... and engine/loop/convcache.py 55631cb8e176..., all
  as registered.
- **Environment:**
  - The cache settings are CONV_CACHE=1 and CONV_CACHE_BRANCH=1, with CLAUDE_CODE_TMPDIR=<run>/cctmp.
  - HOME=<run>/home has the exact PATH line (.zshrc sha256 df313e2ac75d), and git's identity under it is agentcc
    <agentcc@localhost>.
  - Also claude 2.1.321, manifest a44aaeeeefea, setting 32 GB, port 8122.
- **Scripts:** the runner agentcc_cache.sh (46f58cd9a9fdb6de...), agentcc_tasks.py (3e84be3864b4017b...) and
  probe_cc.py (1fffa3b687d332be...) match the registered sums.
- **Bash worked:** no Bash result in any transcript holds the EPERM.
- **Timing:**
  - GO 07:01:41 (NO-SWAP 54 GB, swap 7,437 MB). The server was ready at 07:01:58 (config chat-32, C 160).
  - The probe ran from 07:02:00 to 07:05:46 (226 s), and the tasks from 07:05:46 to 07:21:04.
  - The server served 118 requests with 0 errors and was stopped at 07:21:08. Swap ended at 7,421 MB.

## The probe [K]
- **Case 6, "Run the tests with pytest.",** is the only failure. The model looked around first: its call was Bash `ls
  -la /work/demo && ls /work/demo/tests /work/demo/src`, with the text "I'll look at the project structure and run the
  tests." 32_1111 failed this case the same way.
- **Case 25,** which failed in 32_1111 (Bash `ls`), now passes: the call is Write data/sample.csv.
- The other differences from 32_1111 are in the arguments, and both versions pass: descriptions, a Grep without a
  path, `def parse_` against `def parse_\w+`, `*.py` against `**/*.py`, and one Glob in place of two.
- **Time:** 226 s against 762 s in 32_1111. The median first token was 6.0 s, against 21.3 s.

## Per task [K]
Every session ended with subtype success. There were no timeouts and no max-turn ends.

| Task | Check | Turns | Wall (s) | s / turn | Tool calls | Bash calls | Tool errors |
|---|---|---|---|---|---|---|---|
| t01_fix_test | PASS | 7 | 57 | 8.1 | 6 | 3 | 0 |
| t02_add_function | PASS | 10 | 92 | 9.2 | 9 | 5 | 1 |
| t03_config | PASS | 5 | 42 | 8.5 | 4 | 0 | 0 |
| t04_traceback | PASS | 6 | 53 | 8.9 | 5 | 2 | 0 |
| t05_rename | PASS | 15 | 79 | 5.3 | 14 | 4 | 1 |
| t06_cli_flag | PASS | 5 | 47 | 9.5 | 4 | 2 | 0 |
| t07_import | PASS | 11 | 83 | 7.6 | 10 | 4 | 0 |
| t08_readme | PASS | 3 | 29 | 9.8 | 2 | 0 | 0 |
| t09_refactor | FAIL (wrong edit) | 10 | 98 | 9.8 | 9 | 7 | 1 |
| t10_type_hints | PASS | 6 | 58 | 9.6 | 5 | 2 | 0 |
| t11_shell | PASS | 13 | 104 | 8.0 | 12 | 10 | 0 |
| t12_requirements | PASS | 14 | 171 | 12.2 | 13 | 11 | 1 |

- **Totals:** 105 turns and 915.5 s, or 8.7 s per turn; the median task took 8.5 turns and 68 s.
  - Tool calls: Bash 50, Read 21, Edit 16, Grep 3, Glob 2, Write 1.
- **The 4 tool errors** are command failures, not malformed tool calls (bad input 0):
  - t02: an `ls` of a glob with no match (`*.cfg`);
  - t05: `python`, which is not on the PATH (the model switched to python3);
  - t09: a heredoc. zsh's temp file for it goes under /tmp/zsh (TMPPREFIX), and the sandbox denies that write.
  - t12: `cat -A`, which macOS's cat does not support.
- **git:** there were 7 read-only git calls (t09 1, t12 6) and no commits.
- **Sandbox flags:** no call set dangerouslyDisableSandbox.
- **t09_refactor:** the model made the same edit to report.py as in 32_1111 and 32_1111b (the same blob, 903d5d0). Its
  format_money returns the number without "$", and the check requires "$1,234.50".
  - This is the prompt ambiguity noted in the 32 GB RESULT. The task stays a failure, and it is not re-scored.

## Actions outside the task folder [K]
- **t12 searched the run folder for the grader,** in three Bash calls (#9-#11).
  - #9, described as "Look for sibling task dirs and instructions": `ls -la <run>/tasks/; find <run> -maxdepth 2`. It
    listed the run folder: the other tasks' folders and transcripts, provenance.txt, runner.log and
    server_requests.jsonl.
  - #10, described as "Read task check and stderr files": `cat t12_requirements.check.txt` (empty, because the check
    had not run yet) and t12_requirements.stderr (only Claude Code's unknown-model warning).
  - #11: a Python read of its own live transcript, t12_requirements.jsonl (23,065 characters, starting with its init
    event).
- **Nothing it read bears on its check:** each check is copied in only after claude ends. Its edit, requests==2.32.3
  between numpy and urllib3, follows from the file's case-insensitive order, which it stated.
- **Even scored as a failure,** t12 would leave the tasks at 10 / 12, still at the line, so the verdict holds either way.
  - No call in 32_1111, 32_1111b or 32c named ~/EvalSuite, the manifest or a .agentcc_check.py, by a regex scan of every
    tool input.
- **Denied writes:** two tried to write outside the task folder, and the sandbox denied both.
  - t11 wrote /tmp/t11_probe.txt. The command ended with `; rm -f`, so it did not count as a tool error.
  - t09's heredoc temp file is the error above.
- **Home folders:** no call named a home folder, cc_home, cctmp, ~/.ssh or ~/.claude*.
- **What changed:** Leap's profile allows reads outside its deny list, so the checks under ~/EvalSuite/agentcc/checks,
  the manifest and finished tasks' .agentcc_check.py are readable from inside a task.
  - PREREG amendment 3 part 4 (d2db78b, 07:27, after this run and before 45c starts): from 45c on, a task fails when
    any of its calls names a check. Under that rule this run scores the same.
  - Leap will deny those reads in the profile for the runs after 45c. That needs its own dated line.

## The conversation cache, server timing and memory [K] (results/agentcc_32c/serve.log, convcache.log)
- **Cache:** 117 of 118 requests were hits: 106 live and 11 copies. The one miss was the first probe request.
  - The copies are the first requests of t02-t12, which branch from the stored 9,216-token prefix.
  - Prefill covered 141,318 of 1,519,622 prompt tokens (9.3 %); the median request prefilled 664 new tokens.
  - The store held one or two entries. The largest was 2.58 GB, above the 2.50 GB budget, at the end of t12.
  - The cache's bitwise check was off in this run. Exactness rests on Leap's T1e (part 2).
- **Probe requests (30):** prompts of 8,736-8,845 tokens.
  - The miss's first token came at 20.4 s. The 29 hits had a median of 6.0 s (5.9-6.6 s).
- **Task requests (88):** prompts of 12,502-30,221 tokens, median 13,362. The median first token was 6.8 s, against
  31.9 s in 32_1111b.
  - Live hits: median 6.4 s (4.3-36.4 s). Copies: median 13.0 s (12.5-13.7 s).
  - The slowest was 36.4 s, right after t12 read its own transcript. That request's prompt had grown from 18,954 to
    29,680 tokens, with 11,248 new tokens prefilled.
  - The median whole decode was 74.2 tok/s, and the median reply 120 tokens. The served path differs from 32_1111b's
    (the cache, DFlash2 on every request, BIGMOE_IDLE_MS=50), so this is not a like-for-like decode comparison.
- **Memory:** the lifetime peak was 34.08 GB during the probe and 34.66-37.07 GB during the tasks. The highest was at
  t12's 29,957-token request.
  - The peak was above 32.0 GB on every request again. The PREREG sets no memory line.
- **Heat:** fanwatch froze nothing during the run (results/agentcc_32c/fanwatch_0701_0721.log).
  - Its state was 0 throughout, with a die maximum of 83.1 C and one WARM notice at 07:04:19.
  - So, unlike 32_1111 and 32_1111b, these times hold no heat pauses.

## Against the runs without the cache (information)
| | 32_1111 (Bash blocked) | 32_1111b (Bash working) | 32c (cache) |
|---|---|---|---|
| Probe | 27 / 30 (762 s) | not run | 29 / 30 (226 s) |
| Tasks passed | 11 / 12 | 11 / 12 | 11 / 12 |
| Turns | 160 | 84 | 105 |
| Task wall (s) | 5,619 | 2,595 | 916 |
| s / turn | 35.1 | 30.9 | 8.7 |
| Median first token, tasks (s) | 32.6 | 31.9 | 6.8 |
| Heat freezes in tasks | 6 (487 s) | 3 (243 s) | 0 |
| Lifetime peak (GB) | 34.79-37.46 | 35.55-36.69 | 34.08-37.07 |

- **Speed:** against 32_1111b, the tasks took 65 % less wall time (2.8x) with 25 % more turns. Per turn that is 3.5x
  faster, or 3.2x against 32_1111b's 28.0 s per turn without its freezes.
- **Caveat:** the HOME differs too (amendment 3), and the model's choices vary from run to run (22 / 30 probe calls
  equal). These are figures for this build, not a controlled A/B.

## What it means
- **The verdict:** with the conversation cache, Claude Code works with this model at 32 GB by the registered lines. The
  probe is one above its line, and the tasks one above theirs. Both halves are independent of amendment 1's fix.
- **In use:** a turn's wait fell from about 31 s to about 9 s, because each turn now prefills only its new tokens. The
  12 small tasks took 15 min of wall time, against 43 min without the cache.
- **Limits:** these are small single-step tasks. The probe scores one call per prompt, the run is a single one, and the
  memory peak sits above the 32 GB label. The model also went looking for the grader once; from 45c on, that is checked
  by rule (part 4).
- **Next:** the cached 45 GB arm (45c, part 3) runs if T1-45 passes. Its last start is 12:15.

## Data
results/agentcc_32c/ holds:
- provenance.txt, probe_cc.jsonl and .out, results.jsonl, runner.log, tasks.out, serve.log, convcache.log and
  sandbox.sb (no account names; its ~/.claude* deny is the regex);
- per task: <id>.check.txt, <id>.stderr and <id>.diff (the final repo against its first commit, hidden check
  excluded);
- tool_calls.jsonl: per task, its turns, how it ended, and each call's tool, error class, sandbox flag and git-commit
  flag. It holds no inputs or outputs.
- agentcc_score_32c.txt, and fanwatch_0701_0721.log.

All of it was scanned with the public build's leak regex: 0 hits.

Not committed, kept locally in data/agentcc/runs/32c_1111/:
- claude's raw stream-json transcripts;
- the server's request log, which holds Claude Code's system prompt and tool text.
