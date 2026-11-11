# RESULT: the confirmation rerun at 32 GB with the conversation cache, in a hardened sandbox (arm 32c2, Wed 2026-11-11 08:43-09:02) — PASS: probe 29 / 30, tasks 11 / 12; the cached 32 GB PASS is confirmed, 2 of 2

Pre-registered in docs/PREREG_agentcc_2026-11-11.md before the run: amendment 4 (ad2e532), 4a (66a608e) and 4c
(bc78043, 08:40:19; 4b was void). Leap's runner v5 ran it in Leap's sandbox. EvalSuite scored it with
tools/agentcc_score.py (9bb84b50...) and screened it with tools/agentcc_screen.py.

**Verdict [K]** (results/agentcc_32c2/agentcc_score_32c2.txt):
- **Probe:** 29 / 30 pass, 0 invented tools. PASS (line 28).
- **Tasks:** 11 / 12 pass; t11_shell failed (loop: it ended at the 25-turn limit). PASS (line 10).
- **AGENTCC 32c2: PASS.** With 32c's PASS, the cached 32 GB result is **confirmed, 2 of 2**, as amendment 4 defined it.
  - There is no pooling. Pooled, the tasks are 22 / 24 (information).
- **The integrity screen** found 0 flags.
- **Disclosure, the probe half is a replay:** its 30 replies equal 32c's exactly, every text and tool call.
  - The two runs share the same server, pin and settings, and the probe's prompts hold no paths or times. So the server
    reproduced its own output, and the probe half adds no independent evidence.
  - The confirmation rests on the task half. The task prompts carry each run's paths and times, so the trajectories
    differ: t09, which failed in 32c, passed here, and t11, which passed in 32c, failed.

## Validity [K] (all items of amendments 4, 4a and 4c)
- **Provenance:**
  - release/next 80c8041 with every sum OK, and the launcher claude_local.sh.80c8041 (e5b900fa);
  - the frozen tasks5 copy (28caec4c...), and CLAUDE_BIN = Leap's 2.1.321 copy (abc9feee..., 323,686,048 bytes), whose
    `--version` reads 2.1.321;
  - CONV_CACHE=1, CONV_CACHE_GB=2.5 and CONV_CACHE_BRANCH=1, with CLAUDE_CODE_TMPDIR and TMPPREFIX per task;
  - claude's write set and the check's write set, as registered;
  - the neutral HOME with its PATH line, and agentcc <agentcc@localhost>.
- **Claude Code version:** the init events of all 12 transcripts read claude_code_version 2.1.321.
- **Profiles:** 12 sandbox_<id>.sb and 12 sandbox_check_<id>.sb, each naming only its own task's folders.
- **Clean run:** there are no symlinks anywhere in the run folder, and no Claude Code temp-folder EPERM.
- **Lingering processes:** lingering.jsonl is empty, so nothing was killed. Leap's check after the run found nothing of
  the run still running.
- **Diffs:** made from checks/<id>, what each check graded. tasks/<id> gives identical diffs for all 12, so there were no
  late writes.
- **Timing:**
  - GO 08:43:47 (NO-SWAP 49 GB, swap 6,781 MB). The server was ready at 08:44:06.
  - The probe took 231 s, and the tasks ran from 08:47:57 to 09:02:37. The server was stopped at 09:02:39.
  - Leap's watchdog (TERM 13:40, KILL 13:44, slightly stricter than 4c's 13:45) was not needed, and swap did not grow.

## Per task [K]
| Task | Check | Turns | Wall (s) | s / turn | Tool calls | Tool errors |
|---|---|---|---|---|---|---|
| t01_fix_test | PASS | 7 | 67 | 9.6 | 6 | 0 |
| t02_add_function | PASS | 8 | 76 | 9.5 | 7 | 0 |
| t03_config | PASS | 5 | 44 | 8.7 | 4 | 0 |
| t04_traceback | PASS | 5 | 41 | 8.1 | 4 | 0 |
| t05_rename | PASS | 18 | 96 | 5.3 | 17 | 0 |
| t06_cli_flag | PASS | 5 | 50 | 10.0 | 4 | 0 |
| t07_import | PASS | 9 | 62 | 6.9 | 8 | 1 |
| t08_readme | PASS | 5 | 45 | 9.1 | 4 | 0 |
| t09_refactor | PASS | 12 | 126 | 10.5 | 11 | 2 |
| t10_type_hints | PASS | 5 | 47 | 9.4 | 4 | 0 |
| t11_shell | FAIL (loop, 25-turn limit) | 26 | 192 | 7.4 | 25 | 0 |
| t12_requirements | PASS | 3 | 29 | 9.7 | 2 | 0 |

- **Totals:** 108 turns and 874.7 s, or 8.1 s per turn; the median task took 6 turns and 56 s.
  - Tool calls: Bash 50, Read 21, Edit 15, Grep 4, Write 4, Glob 2.
  - 11 sessions ended in success and one at the turn limit. There were no git commits and no sandbox flags.
- **t11_shell:** count.sh prints 14 where the check wants 7.
  - Its final count.sh (#15) pipes `find ... -exec wc -l {} +` into awk and sums every line, including wc's total
    line.
  - It saw the double count but misread the cause. It then ran the same `printf | cat | wc -l` test six times in a row
    (#20-#25), until it hit the turn limit.
  - One of its /tmp writes was refused, as in every run, and it worked with scratch files in its own folder instead.
  - The sandbox did not cause the failure.
- **t09_refactor passed.** Its format_money includes the "$", as in 45c.
- **The 3 tool errors:**
  - t07: `python`, which is not on the PATH;
  - t09: a Grep of the whole run folder, which the new profile refused ("could not be opened (EPERM)");
  - t09: a `mkdir` in /tmp, refused.

## The new sandbox in use [K]
- **Outside calls:** only two calls reached outside a task's own folders, and the profile refused both.
  - t09's Grep searched the run folder for `,\.\d*f|format_money|def .*\(`. It names no check, so the part 4 rule does
    not apply.
  - t11's /tmp write was the other.
  - No task read the run folder, another task's folder or the checks.
- **Inside the folders:** no process outlived its task, nothing changed a task folder after grading, and the graded
  copies hold no links.
- **Against 32c and 45c:** with this sandbox, the grader searches seen there (32c's t12 and 45c's t10) would have been
  refused. Here, one search was attempted (t09) and refused.

## Server, cache and memory [K] (serve.log, convcache.log)
- **Requests:** 123, with 0 errors. 122 were cache hits (111 live, 11 copies). The miss was the first probe request.
  - Prefill covered 127,625 of 1,565,321 prompt tokens (8.2 %).
- **Tasks (93 requests):**
  - Prompts were 12,494-16,905 tokens, median 13,356.
  - The median first token was 6.7 s: live hits 6.4 s (3.8-14.5 s), and copies 12.6 s.
  - The median whole decode was 81.4 tok/s.
- **Memory:** the lifetime peak was 34.12 GB during the probe and 34.66-36.17 GB during the tasks, above 32 GB again.
- **Heat:** no freezes (results/agentcc_32c2/fanwatch_0843_0903.log). The state was 0 throughout, with a die maximum of
  78.3 C and one WARM notice at 08:47:29.

## 32c and 32c2 (information)
| | 32c | 32c2 |
|---|---|---|
| Probe | 29 / 30 (226 s) | 29 / 30 (231 s), the same 30 replies |
| Tasks | 11 / 12 (t09 wrong edit) | 11 / 12 (t11 loop) |
| Turns | 105 | 108 |
| Task wall (s) | 916 | 875 |
| s / turn | 8.7 | 8.1 |
| Median first token, tasks (s) | 6.8 | 6.7 |
| Cache hits | 117 / 118 | 122 / 123 |
| Lifetime peak (GB) | 34.08-37.07 | 34.12-36.17 |
| Out-of-folder reads | 3 (t12, allowed then) | 0 (1 attempt, refused) |

## What it means
- **The confirmation:** the cached 32 GB PASS held on a second run, in a sandbox where the checks and the run folder are
  unreadable and nothing the agent leaves behind can reach the graded copy.
- **The task half** is independent evidence: 11 / 12 twice, with a different task failing each time. Two of 24 tasks
  failed, one by a wrong edit and one by a loop.
- **The probe half** repeated itself exactly. It shows the server is deterministic at this setting, not a second sample
  of the probe.
- **Speed is unchanged:** about 8 s per turn, and a first token at about 6.5 s.

## Data
results/agentcc_32c2/ holds:
- provenance.txt, SHA256SUMS.80c8041, probe_cc.jsonl and .out, results.jsonl, runner.log, tasks.out, serve.log,
  convcache.log and the base sandbox.sb;
- the 12 sandbox_<id>.sb and 12 sandbox_check_<id>.sb;
- per task: <id>.check.txt, <id>.stderr and <id>.diff (from checks/<id>, against the template);
- tool_calls.jsonl, agentcc_score_32c2.txt, screen_32c2.txt and fanwatch_0843_0903.log.

All of it was scanned with the public build's leak regex: 0 hits. The raw transcripts and the server's request log are
not committed; they are kept locally in data/agentcc/runs/32c2b/.
