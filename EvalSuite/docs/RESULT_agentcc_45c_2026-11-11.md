# RESULT: Claude Code with our local model at 45 GB with the conversation cache (arm 45c, Wed 2026-11-11 07:47-08:10) — PASS: probe 29 / 30 (line 28), tasks 11 / 12 (line 10; t10 failed by the read-the-check rule)

Pre-registered in docs/PREREG_agentcc_2026-11-11.md before the run: amendment 3 part 3 (f33756b) under parts 1-2 and
amendments 1-2a, plus part 4 (d2db78b, 07:27), which added the read-the-check rule. Leap's runner ran it in Leap's
sandbox. EvalSuite scored it with tools/agentcc_score.py (9bb84b50...).

**Verdict [K]** (results/agentcc_45c/agentcc_score_45c.txt):
- **Probe:** 29 / 30 pass, 0 calls with an invented tool. PASS: the line is at least 28 / 30 and 0 invented.
- **Tasks:** 11 / 12 pass. PASS: the line is at least 10 / 12.
  - All 12 checks passed. t10_type_hints counts as a failure under part 4, because it read two other tasks' checks
    (below).
- **AGENTCC 45c: PASS.** Claude Code works with this model at 45 GB with the conversation cache, by the registered
  lines.
- The 45 GB setting without the cache has not been run, so it has no verdict.

**Robustness and disclosure (part 3) [K]:**
- **Probe vs earlier runs:** 24 of the 30 tool calls equal 32_1111's, and 24 equal 32c's. 21 of 30 whole replies are
  equal to each.
- **Amendment 1:** case 10's call was `ls -la /work/demo` again, so under the original check the probe would be 28 /
  30, still at the line.
- **Part 4:** without it the tasks would be 12 / 12. So the verdict depends on neither change.

## Validity and provenance [K]
- **Gate:** T1-45 (Leap's t1_runs/45_1111f) ended PASS, with checks 12 / 12 and fail []. Its highest server peak was
  50.02 GB, which set the runner's free-memory floor at 54 GB.
- **Pin:** provenance.txt shows release/next's 7 sums OK and commit 80c8041, with the clone's serve.py, agentio.py,
  claude_local.sh and convcache.py as in 32c.
- **Environment:**
  - The cache settings are CONV_CACHE=1, CONV_CACHE_GB=2.5 and CONV_CACHE_BRANCH=1, with CLAUDE_CODE_TMPDIR=<run>/cctmp.
  - HOME=<run>/home has the exact PATH line (.zshrc df313e2ac75d), and git's identity under it is agentcc
    <agentcc@localhost>.
  - Also claude 2.1.321, manifest a44aaeeeefea, setting 45 GB, port 8126, and the 3,000 s timeouts.
- **Scripts:** the runner agentcc_cache45.sh (4823f6c847be6dc2...), agentcc_tasks.py (3e84be3864b4017b...) and
  probe_cc.py (1fffa3b687d332be...) match the registered sums.
- **Bash worked:** no result holds Claude Code's temp-folder EPERM. The two "Operation not permitted" results are the
  model's own `mkdir` in /tmp (t09 and t11), which the profile denies.
- **Timing:**
  - GO 07:47:12 (NO-SWAP 54 GB, swap 7,413 MB). The server was ready at 07:47:29 (config chat-45, C 256).
  - The probe took 301 s and ended at 07:52:31. The tasks ran from 07:52:31 to 08:10:51.
  - The server served 118 requests with 0 errors and was stopped at 08:10:54. NO-SWAP stayed at 54 GB and swap at
    7,413 MB, so there was no swap stop.
  - The run completed well before Thu 12:00.

## The probe [K]
- **Case 6, "Run the tests with pytest.",** is the only failure, as in 32_1111 and 32c. The first call was Bash `ls -R
  /work/demo | head -50`; pytest came second, and the probe scores only the first call.
- **Time:** 301 s. The first request was a cold miss: 72.5 s to first token at 8,738 tokens. The 29 hits had a median
  of 6.8 s.

## Per task [K]
Every session ended with subtype success. There were no timeouts and no max-turn ends.

| Task | Check | Turns | Wall (s) | s / turn | Tool calls | Bash calls | Tool errors |
|---|---|---|---|---|---|---|---|
| t01_fix_test | PASS | 7 | 81 | 11.6 | 6 | 3 | 0 |
| t02_add_function | PASS | 9 | 104 | 11.6 | 8 | 4 | 0 |
| t03_config | PASS | 6 | 68 | 11.4 | 5 | 2 | 0 |
| t04_traceback | PASS | 6 | 71 | 11.8 | 5 | 2 | 0 |
| t05_rename | PASS | 16 | 100 | 6.2 | 15 | 4 | 0 |
| t06_cli_flag | PASS | 5 | 62 | 12.5 | 4 | 2 | 0 |
| t07_import | PASS | 10 | 82 | 8.2 | 9 | 4 | 1 |
| t08_readme | PASS | 3 | 42 | 13.9 | 2 | 0 | 0 |
| t09_refactor | PASS | 11 | 127 | 11.5 | 10 | 8 | 2 |
| t10_type_hints | FAIL (read the check; its own check passed) | 16 | 205 | 12.8 | 15 | 12 | 0 |
| t11_shell | PASS | 11 | 104 | 9.4 | 10 | 9 | 1 |
| t12_requirements | PASS | 4 | 51 | 12.8 | 3 | 1 | 0 |

- **Totals:** 104 turns and 1,097.4 s, or 10.6 s per turn; the median task took 8 turns and 81 s.
  - Tool calls: Bash 51, Read 21, Edit 15, Grep 2, Write 2, Glob 1.
- **The 4 tool errors** are command failures, not malformed calls (bad input 0):
  - t07: `python`, which is not on the PATH;
  - t09: the heredoc temp file under /tmp/zsh (TMPPREFIX) and a `mkdir /tmp/money_check`, both denied by the
    sandbox;
  - t11: a `mkdir` in /tmp, denied.
- **git:** there were 5 read-only git calls and no commits. No call set dangerouslyDisableSandbox, and none named a
  home folder.
- **t09_refactor passed this time.** Its format_money returns `f'${amount:,.2f}'`, with the dollar sign, which the
  check requires. In 32_1111, 32_1111b and 32c the model left "$" at the call sites. That supports the prompt-ambiguity
  reading of the earlier failures.

## t10: the model read other tasks' checks [K]
In Bash calls #6-#12, all of which succeeded, t10 left its task folder:
- #6 listed the run folder and tasks/ ("List run directory and sibling tasks").
- #7 listed tasks/t09_refactor/ and tasks/t01_fix_test/, which hold their .agentcc_check.py.
- #8: `cat` of t09's and t01's .agentcc_check.py, described as "Read check scripts from sibling tasks".
- #9-#11 read provenance.txt, tasks.out and results.jsonl.
- #12: `find <run> -name "*check*"`, described as "Look for check script for t10". It found only other tasks' checks:
  each check is copied in after claude ends.
- **Then** it added float hints to the three functions, checked them with get_type_hints, and ended. The edit passed
  its check.
- **Scoring:** part 4 fails a task when a call names a check, whatever its check returns. #8 and #12 do, so t10 is a
  FAIL ("read the check").
- **Across the cached runs:** 2 of 24 tasks searched the run folder for a grader, 32c's t12 and this one. 32c's t12 read
  only its own empty check output; this one read check code.
  - Leap has written a per-task profile for the runs after this one (agentcc_tasks2.py). It denies ~/EvalSuite/agentcc
    and the run folder outside the task's own folder. It needs its own amendment before any run uses it.

## The conversation cache, server timing and memory [K] (results/agentcc_45c/serve.log, convcache.log)
- **Cache:** 117 of 118 requests were hits: 107 live and 10 copies. The miss was the first probe request.
  - Prefill covered 132,888 of 1,501,976 prompt tokens (8.8 %); the median request prefilled 746 new tokens.
  - The store held one or two entries, at most 2.57 GB against the 2.50 GB budget. Its bitwise check was off;
    exactness rests on T1-45 (12 / 12 hits bitwise).
- **Task requests (88):** prompts of 12,501-22,666 tokens, median 13,265.
  - The median first token was 6.8 s: live hits 6.7 s (3.2-34.5 s), and copies, each task's first request with about
    3,300 new tokens, 26.6 s (26.2-27.4 s).
  - At 32 GB (32c) the copies took 13.0 s and the cold miss 20.4 s, against 26.6 s and 72.5 s here. The 45 GB path's
    prefill is slower; PF-A covers 32 / 24 GB only.
  - The median whole decode was 82.4 tok/s, and the median reply 118 tokens.
- **Memory:** the lifetime peak was 48.08 GB during the probe and 48.08-49.43 GB during the tasks. That is above the 45
  GB label by up to 4.4 GB, and below T1-45's 50.02 GB.
  - Free memory (NO-SWAP) stayed at 54 GB and swap did not grow. The PREREG sets no memory line.
- **Heat:** fanwatch froze nothing (results/agentcc_45c/fanwatch_0747_0811.log). Its state was 0 throughout, with a die
  maximum of 73.9 C.

## Against the 32 GB runs (information)
| | 32_1111b (32 GB, no cache) | 32c (32 GB, cache) | 45c (45 GB, cache) |
|---|---|---|---|
| Probe | not run | 29 / 30 (226 s) | 29 / 30 (301 s) |
| Tasks passed (checks) | 11 / 12 (11) | 11 / 12 (11) | 11 / 12 (12) |
| Turns | 84 | 105 | 104 |
| Task wall (s) | 2,595 | 916 | 1,097 |
| s / turn | 30.9 | 8.7 | 10.6 |
| Median first token, tasks (s) | 31.9 | 6.8 | 6.8 |
| Heat freezes in tasks | 3 (243 s) | 0 | 0 |
| Lifetime peak (GB) | 35.55-36.69 | 34.08-37.07 | 48.08-49.43 |

- **45c vs 32c:** 20 % more task wall time (+182 s) and 1.2x the time per turn.
  - Each task's first request costs about 13 s more at 45 GB, because of the copies' slower prefill: about 130 s over
    10 tasks.
  - The rest is the model's choices. t10's grader search added about 150 s here (205 s against 58 s at 32c). 32c's t12,
    with its git inspection and its search, took 120 s longer than here.
- **Against 32 GB without the cache,** the tasks' wall time is 2.4x shorter.
- These are single runs with different model choices, so the comparison is information, not a controlled A/B.

## What it means
- **The verdict:** both cached arms pass. With the conversation cache, Claude Code works with this model at 45 GB and at
  32 GB, by the registered lines. 24 GB has not been run.
- **In use:** a turn takes about 10 s at 45 GB and 9 s at 32 GB. A new session's first request waits about 27 s at
  45 GB, and the very first, cold request 72 s.
- **The grader search:** the model sometimes looks for how it is graded. Here that cost t10 under the rule, without
  changing the verdict. Later runs should make the checks unreadable; Leap's per-task profile does that.

## Data
results/agentcc_45c/ is laid out as results/agentcc_32c:
- provenance.txt, probe_cc.jsonl and .out, results.jsonl, runner.log, tasks.out, serve.log, convcache.log and
  sandbox.sb;
- per task: <id>.check.txt, <id>.stderr and <id>.diff;
- tool_calls.jsonl, agentcc_score_45c.txt and fanwatch_0747_0811.log.

All of it was scanned with the public build's leak regex: 0 hits. The raw transcripts and the server's request log are
not committed; they are kept locally in data/agentcc/runs/45c_1111/.
