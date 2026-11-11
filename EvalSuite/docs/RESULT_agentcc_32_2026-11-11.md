# RESULT: Claude Code with our local model at the 32 GB setting (Wed 2026-11-11 03:45) — FAIL on the probe line (27 / 30, line 28); tasks 11 / 12 PASS; the sandbox blocked every Bash call; the rerun with Bash: tasks 11 / 12 (information)

Pre-registered in docs/PREREG_agentcc_2026-11-11.md before any run: 63218d0, with the pin 8046bac, the re-pin to Leap
39947b1 (89651df), and the dated schedule line (ea53db1). Leap's runner ran it in Leap's sandbox. EvalSuite scored it
with tools/agentcc_score.py.

**Verdict [K]** (results/agentcc_32/agentcc_score_32.txt):
- **Probe:** 27 / 30 pass, 0 calls with an invented tool. FAIL: the line is at least 28 / 30 and 0 invented.
- **Tasks:** 11 / 12 pass (t09_refactor failed, wrong edit). PASS: the line is at least 10 / 12.
- **AGENTCC 32: FAIL.**

Two problems in the setup are reported below: the sandbox blocked Bash (Leap's profile), and one probe check was wrong
(mine). Neither changes the registered verdict.

## Provenance [K]
- **Code:** provenance.txt shows the clone's serve.py, agentio.py and tools/claude_local.sh equal to release/next at
  commit 39947b1 (serve.py c9d0e5b5051f..., agentio.py 60e4520efc8d..., claude_local.sh e5b900fae5fa...).
- **Run:** claude 2.1.321, manifest sha256 a44aaeeeefea, setting 32 GB, port 8120.
- **Server:** serve.py --mem 32 was ready at 01:59:06 (config chat-32, C 160). It served 175 requests with 0 errors.
- **Timing:** the run started at 01:58:50. The probe took 762 s and ended at 02:11:50; the tasks ran from 02:11:50 to
  03:45:31.

## Problem 1: the sandbox blocked Claude Code's Bash tool in every task [K]
- **What happened:** all 79 Bash calls returned "EPERM: operation not permitted, mkdir
  '/private/tmp/claude-501/<the task folder's path, as a slug>'".
  - Claude Code creates that temp folder for each working folder.
  - Leap's profile (results/agentcc_32/sandbox.sb) allows writes only under the run folder, /private/var/folders and
    /dev.
- **Scale:** no Bash call succeeded, and these 79 are the run's only tool errors. The other calls were Read 37, Glob 16,
  Edit 15, Grep 5, Write 2, Agent 1 and ToolSearch 1.
- **What it means:** the tasks did not run as registered. The PREREG says Bash, python3 and git run inside the sandbox.
  - The model could not run tests or scripts at all. 11 of 12 tasks passed with Read, Edit, Write, Glob and Grep alone.
  - Bash inside real tasks is untested.
  - Turns and times below include the failed Bash calls.
- **The probe is unaffected:** it calls the API directly and runs no tool.

## Problem 2: one probe check was wrong (EvalSuite's) [K]
- **The case:** case 10, "List the files in this directory, including hidden ones."
- **The call:** the model ran `ls -la /work/demo`, which lists hidden files. The check wanted the substring "-a", which
  "ls -la" does not contain, so it rejected a correct call.
- **Information only, after the run:** with a check that accepts any ls flag group containing a or A, the probe would be
  28 / 30, at the line. This was found after the run, so the verdict stays FAIL.

## The other two probe failures (the model's behaviour) [K]
In both, the model looks around before acting. The single-turn probe scores only the first call (PREREG, "Known limits").
- **Case 6, "Run the tests with pytest.":** the first call was Bash `ls -la /work/demo && ls /work/demo/tests`, with the
  text "I'll locate the test setup and run pytest."
- **Case 25, "Create data/sample.csv with the header line name,age":** the first call was Bash `ls /work/demo`, not
  Write.

## Per task [K]
Every session ended with subtype success. There were no timeouts and no max-turn ends; t11 used 24 of its 25 turns.

| Task | Check | Turns | Wall (s) | s / turn | Tool calls | Failed Bash calls |
|---|---|---|---|---|---|---|
| t01_fix_test | PASS | 23 | 969 | 42.1 | 22 | 12 |
| t02_add_function | PASS | 15 | 806 | 53.7 | 22 | 13 |
| t03_config | PASS | 6 | 194 | 32.4 | 5 | 1 |
| t04_traceback | PASS | 12 | 452 | 37.7 | 11 | 8 |
| t05_rename | PASS | 20 | 354 | 17.7 | 19 | 5 |
| t06_cli_flag | PASS | 11 | 456 | 41.4 | 10 | 6 |
| t07_import | PASS | 15 | 355 | 23.6 | 14 | 5 |
| t08_readme | PASS | 3 | 97 | 32.3 | 2 | 0 |
| t09_refactor | FAIL (wrong edit) | 8 | 347 | 43.4 | 7 | 3 |
| t10_type_hints | PASS | 10 | 333 | 33.3 | 9 | 4 |
| t11_shell | PASS | 24 | 815 | 34.0 | 23 | 15 |
| t12_requirements | PASS | 13 | 441 | 33.9 | 12 | 7 |

- **Totals:** 160 turns and 5,619 s. That is 35.1 s per turn overall; the median task took 12.5 turns and 398 s.
- **t09_refactor:** the model's format_money(amount) returns the number without the dollar sign, and "$" stays at the
  three call sites (results/agentcc_32/t09_refactor.diff).
  - The outputs are unchanged, and the format appears once.
  - The registered check asserts format_money(1234.5) == '$1,234.50'. The prompt asked to extract "the same
    money-formatting code", which was `${amount:,.2f}`, so this stays a failure.
- **t02 and the "invented" flag:** the scorer's t02 row lists "invented ['Agent']". That is a false positive in the
  scorer.
  - The init event lists the subagent tool as "Task", while the tools array sent to the model calls it "Agent".
  - The subagent ran: stderr shows `{"model":"exactstream-sub","query_source":"agent:builtin:general-purpose"}`.
  - No score changes, because t02 passed.
- **Unknown-model warnings:** every task's stderr has Claude Code's unrecognized_model warning for "exactstream".

## Actions outside the task folder [K]
- **Private reads:** in t01, with Bash blocked, the model used Read and Glob on Claude Code's own config file and shell
  snapshot, in the isolated config folder. These are reads the sandbox allows.
  - The snapshot is built from the user's shell setup, so it is private.
  - Leap flagged them. They were in the first push (2717dd8, ~03:50) and were withheld from HEAD at 04:00 (edd5e32).
    The last public refresh was at 03:22, so they were never published.
  - Since 04:0x the raw transcripts are not committed at all (the orchestrator and Leap asked). They are kept locally
    in data/agentcc/runs/32_1111/, with those 3 results withheld.
- **Sandbox flag:** 10 Bash calls set dangerouslyDisableSandbox: true (t01 1, t02 3, t04 1, t05 1, t07 1, t11 2, t12 1).
  - Claude Code's own sandbox is not on, so the flag changes nothing, and the outer profile still applied.
  - These calls failed on the EPERM, like every Bash call.
- **Other attempts:** t12 tried `ls -la` on the home folder, and t02 and t09 used /tmp in Bash commands. All failed on the
  EPERM.
- **With Bash working,** reads like these would succeed, because the profile allows reads outside its deny list. Every
  later run's transcripts are scanned and redacted before any commit.

## Server timing and memory [K] (results/agentcc_32/serve.log)
- **Probe requests (30):** prompts of 8,736-8,845 tokens; Claude Code's tools array alone is large. Median time to first
  token 21.3 s, at most 103.6 s (case 26, during a heat freeze, below).
- **Task requests (145):** prompts of 9,106-20,187 tokens, median 13,824.
  - Median time to first token 32.6 s, at most 124.0 s.
  - Every first token above 100 s fell in a heat freeze.
  - Median whole decode 58.1 tok/s on replies of 20 tokens or more; median reply 110 tokens.
- **Heat pauses [K] (~/Library/Logs/fanwatch.log):** fanwatch froze the server (pid 26802) 7 times, for 81-82 s each
  (568 s in all), at 02:08:49, 02:19:55, 02:32:42, 02:47:29, 03:03:17, 03:17:25 and 03:32:12.
  - One fell in the probe. The other six fell in t01, t02, t04, t06, t09 and t11.
  - Those six total 487 s, or 8.7 % of the tasks' 5,619 s. Without them the tasks average 32.1 s per turn, against 35.1
    s with them.
  - No task came near its timeout because of them.
- **Memory:** the server's "peak" is the engine's mem[6], the lifetime peak footprint after decode. It was above 32.0 GB
  on every request: 34.79 GB after the first request (8,738 prompt tokens), and 37.46 GB at most.
  - The PREREG sets no memory line, so this is information.
  - It does show that the "32 GB" setting does not stay within 32 GB at Claude Code's prompt sizes.

## What it means
- **The verdict:** by the registered lines, Claude Code does not pass at 32 GB. The probe is one call short, and one of
  its three failures is EvalSuite's check.
- **The tasks:** Claude Code finished small real coding tasks with this model: 11 of 12, with every session ending in
  success, and no timeouts, loops or invented tools. It did so with Bash blocked, so Bash inside real tasks is untested.
- **Next (proposed to Leap and the orchestrator, Wed 03:5x; the scorer fixes are PREREG amendment 1, 9cbd4da):**
  1. Leap fixes the sandbox narrowly.
  2. A rerun of the 12 tasks at 32 GB, with an amendment written before it starts.
  3. Before the 45 GB run, fixes to probe case 10's check and the scorer's Agent / Task names, disclosed as written after
     this run. The 32 GB verdict stays FAIL.

## Data
results/agentcc_32/ holds:
- provenance.txt, probe_cc.jsonl and .out, results.jsonl, runner.log, tasks.out, serve.log and sandbox.sb;
- per task: <id>.check.txt, <id>.stderr, and <id>.diff (the final repo against its first commit, hidden check
  excluded);
- tool_calls.jsonl: per task, its turns, how it ended, and each tool call's name, error class and sandbox flag. It
  holds no inputs or outputs.
  - sandbox.sb has the two account folder names replaced, as the public build does;
- agentcc_score_32.txt, and fanwatch_0158_0346.log (fanwatch's lines for the run).

Not committed:
- claude's raw stream-json transcripts, which can hold reads outside the task folder. They are kept locally in
  data/agentcc/runs/32_1111/; agentcc_score.py reads them from there or from Leap's run folder.
- the server's request log, which holds Claude Code's system prompt and tool text (kept locally as
  data/agentcc/runs/32_1111_server_requests.jsonl);
- the isolated Claude Code config folder.

## Addendum: the 12 tasks again with Bash working (32_1111b, Wed 04:23-05:07) — information; tasks 11 / 12
Pre-registered as amendments 2 and 2a (e0fea3a, 211650c) before it started. The known git commit limit (9b5ce10) was
written after the start and before any result.

- **Run [K]:** the provenance shows pin 39947b1 with matching sums, claude 2.1.321, manifest a44aaeeeefea, the 32 GB
  setting and port 8124.
  - GO 04:23:49; the server was ready at 04:24:08 (config chat-32, C 160).
  - Tasks ran from 04:24:08 to 05:07:25: 68 requests, 0 server errors.
- **Valid:** no Bash result in any of the 12 transcripts holds the EPERM, and 28 of 30 Bash calls ran without error.
- **Tasks [K]** (results/agentcc_32b/agentcc_score_32b.txt): 11 / 12 pass, and t09_refactor fails (wrong edit). That
  meets the registered task line (at least 10 / 12) with the task half run as designed.
  - The 32 GB verdict stays FAIL; it was set by the probe half of 32_1111.
  - The scorer's AGENTCC line reads INCOMPLETE because this folder has no probe, as amendment 2 says.

| Task | Check | Turns | Wall (s) | s / turn | Tool calls | Bash calls | Tool errors |
|---|---|---|---|---|---|---|---|
| t01_fix_test | PASS | 7 | 214 | 30.6 | 6 | 3 | 0 |
| t02_add_function | PASS | 9 | 364 | 40.5 | 8 | 4 | 0 |
| t03_config | PASS | 5 | 154 | 30.7 | 4 | 0 | 0 |
| t04_traceback | PASS | 5 | 127 | 25.4 | 4 | 1 | 0 |
| t05_rename | PASS | 16 | 269 | 16.8 | 15 | 4 | 0 |
| t06_cli_flag | PASS | 5 | 168 | 33.5 | 4 | 2 | 0 |
| t07_import | PASS | 9 | 205 | 22.8 | 8 | 3 | 0 |
| t08_readme | PASS | 3 | 172 | 57.3 | 2 | 0 | 0 |
| t09_refactor | FAIL (wrong edit) | 9 | 297 | 33.0 | 8 | 6 | 1 |
| t10_type_hints | PASS | 6 | 199 | 33.2 | 5 | 2 | 0 |
| t11_shell | PASS | 7 | 248 | 35.4 | 6 | 5 | 1 |
| t12_requirements | PASS | 3 | 177 | 59.1 | 2 | 0 | 0 |

- **Totals:** 84 turns and 2,595 s, or 30.9 s per turn; the median task took 6.5 turns and 202 s.
  - Against 32_1111 (Bash blocked, 160 turns and 5,619 s) that is 48 % fewer turns and 54 % less wall time.
  - Every session ended in success, with no timeouts. Tool calls: Bash 30, Read 21, Edit 15, Glob 2, Grep 2, Write 2.
- **The 2 tool errors:** both are command failures, not malformed tool calls.
  - In t09, a Bash command wrote a scratch file to /tmp, and the sandbox denied it.
  - In t11, the model ran `cat -A`, which macOS's cat does not support.
- **git commit:** no task tried one, so the signing limit never applied.
- **Outside the task folder:** there were no reads outside the task folders and no dangerouslyDisableSandbox flags. The
  only out-of-folder action was t09's denied /tmp write.
- **t09_refactor:** the model made the same final edit as in 32_1111 (identical diff).
  - The prompt does not say whether the helper includes the dollar sign, and the check requires it. This is a prompt
    ambiguity.
  - Under the registered check the task stays a failure, and it is not re-scored.
- **Server [K]:**
  - Prompts were 12,504-17,003 tokens, median 13,195.
  - Median time to first token 31.9 s, at most 111.5 s (in a heat freeze).
  - Median whole decode 61.3 tok/s; median reply 110 tokens.
  - The lifetime peak was 35.55-36.69 GB, above 32.0 GB on every request again.
- **Heat freezes [K]** (results/agentcc_32b/fanwatch_0423_0508.log): 3 of 81 s each, at 04:32:23, 04:49:12 and 05:05:20.
  - They fell in t02, t08 and t12: 243 s in all, or 9.4 % of the tasks' wall time.
  - Without them the tasks average 28.0 s per turn.
- **Data:** results/agentcc_32b/, laid out as results/agentcc_32, with no raw transcripts or request logs (kept locally
  in data/agentcc/runs/32_1111b/). It was scanned with the public build's leak regex: 0 hits.

**What the rerun shows (information):** with Bash working, Claude Code and this model finished 11 of 12 small coding
tasks at 32 GB. They needed about half the turns and time of the blocked run, with two ordinary command errors and no
reads outside the task folders. The registered 32 GB verdict is still FAIL, on the probe (27 / 30, one call short; one
of the three failures is EvalSuite's check). The next verdict run is 45 GB, under amendments 1-3.
