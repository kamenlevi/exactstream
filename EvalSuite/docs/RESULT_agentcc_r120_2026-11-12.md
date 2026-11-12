# RESULT: the 1.2 release gate, the agent test on the package candidate pkg_120b with Claude Code 2.1.322 at 32 GB with the conversation cache (arm r120, Thu 2026-11-12 19:14-19:36) — PASS: probe 29 / 30, tasks 11 / 12

Pre-registered in docs/PREREG_agentcc_2026-11-11.md before the run: amendment 5 (048804e), 5a (0df951c), 5b (bccbde8),
5c (72c180e) and 5d (69dfd11, 19:08:37). Leap's runner agentcc_r120.sh (9e0fbc21...) ran it in the hardened sandbox.
EvalSuite scored it with tools/agentcc_score.py and screened it with tools/agentcc_screen.py.

**Verdict [K]** (results/agentcc_r120/agentcc_score_r120.txt):
- **Probe:** 29 / 30 pass, 0 invented tools. PASS (line 28).
- **Tasks:** 11 / 12 pass; t09_refactor failed (wrong edit, the known prompt ambiguity). PASS (line 10).
- **AGENTCC r120: PASS.** By the registered lines, Claude Code 2.1.322 works with the 1.2 candidate as shipped at 32
  GB.
- **The integrity screen:** 0 flags. No task's calls named a check.
- **Disclosure: the probe half repeats 32c2.** All 30 replies equal 32c2's exactly. The probe sends EvalSuite's fixed
  tools file, so it shows only that the candidate server answers it exactly as 80c8041 did; it is not new evidence. The
  task half runs Claude Code 2.1.322 end to end on the candidate.

## Validity [K]
- **The candidate, as shipped:** pkg_120b 21fb59b6, its clean clone at that commit, and all 9 agent-path files equal to
  its MANIFEST.
  - serve.py is 4c15a1d4, with the Host / Origin check.
  - The server ran on its own defaults: the runner's environment held no cache, engine or window variable, and
    EXACTSTREAM_CONTEXT=65536 was set as tools/claude_code.sh sets it.
- **Claude Code:** CLAUDE_BIN was Leap's 2.1.322 copy (a9d23794..., 323,702,464 bytes). The init events of all 12
  transcripts read claude_code_version 2.1.322.
- **The sandbox:**
  - 12 sandbox_<id>.sb and 12 sandbox_check_<id>.sb profiles;
  - no symlinks, and no Claude Code temp-folder EPERM;
  - no lingering processes;
  - no late writes: the graded copies equal the task folders.
- **The cache was on:** convcache.log shows it installed (2.50 GB), and 123 of 124 requests were hits. serve.log has no
  "cache off" line.
- **The new check refused 0 requests.** Every Claude Code request passed it.

## Per task [K]
| Task | Check | Turns | Wall (s) | s / turn | Tool calls | Tool errors |
|---|---|---|---|---|---|---|
| t01_fix_test | PASS | 8 | 108 | 13.5 | 7 | 0 |
| t02_add_function | PASS | 9 | 80 | 8.9 | 8 | 0 |
| t03_config | PASS | 5 | 48 | 9.5 | 4 | 0 |
| t04_traceback | PASS | 5 | 40 | 8.0 | 4 | 0 |
| t05_rename | PASS | 14 | 80 | 5.7 | 13 | 0 |
| t06_cli_flag | PASS | 5 | 49 | 9.8 | 4 | 0 |
| t07_import | PASS | 9 | 71 | 7.9 | 8 | 1 |
| t08_readme | PASS | 3 | 30 | 10.0 | 2 | 0 |
| t09_refactor | FAIL (wrong edit) | 8 | 77 | 9.6 | 7 | 0 |
| t10_type_hints | PASS | 5 | 131 | 26.1 | 4 | 0 |
| t11_shell | PASS | 15 | 150 | 10.0 | 14 | 0 |
| t12_requirements | PASS | 20 | 174 | 8.7 | 19 | 5 |

- **Totals:** 106 turns and 1,038 s, or 9.8 s per turn. Every session ended in success, with no timeouts or turn limits.
- **The grader search again:** t12 tried to list the run folder ("Look for sibling task dirs and spec files"). The
  profile refused both calls, which are 2 of its 5 tool errors. The others are `cat -A`, a `sort -c` exit and a missing
  file.
- **Claude Code's memory:** t12 also wrote memory notes into its own config folder, cc_home/<id>/memory, through
  Claude Code's memory feature. That folder is in its write set and is thrown away after the task.
- **Refused /tmp writes:** t11's /tmp scratch writes were refused, as in every run, and the task passed.
- **No compaction:** the longest prompt was 19,474 tokens, under the shipped 64K window's compaction point of about 32K.
- **t10's 131 s** comes from a heat freeze, below.

## Server, cache, memory and heat [K]
- **Requests:** 124, with 0 errors. 123 were cache hits (live and copy); the one miss was the first probe request.
- **First token, tasks:** median 6.5 s (live hits 6.2 s, copies 13.2 s), against 6.7 s in 32c2.
- **Decode:** median 67.8 tok/s (32c2: 81.4).
- **Memory:** the lifetime peak was 34.11 GB during the probe and 34.68-36.50 GB during the tasks, within 32c2's 40 GB
  floor.
- **Heat:** fanwatch froze the server twice, at 19:18:50-19:19:31 (81 s, around t01's start) and 19:28:56-19:30:17
  (121 s, t10's first request, with a 94.5 s first token). The die maximum was 84.4 C, with the fans at medium.

## Against 32c2 (information)
| | 32c2 (80c8041, 2.1.321) | r120 (pkg_120b, 2.1.322) |
|---|---|---|
| Probe | 29 / 30 | 29 / 30, the same 30 replies |
| Tasks | 11 / 12 (t11 loop) | 11 / 12 (t09 wrong edit) |
| Turns, task wall | 108, 875 s | 106, 1,038 s (2 heat freezes) |
| s / turn | 8.1 | 9.8 |
| First token median (tasks) | 6.7 s | 6.5 s |
| Peak (GB) | 34.12-36.17 | 34.11-36.50 |

## What it means
- **The release gate passes:** Claude Code 2.1.322 works with package 1.2 as shipped, at 32 GB with the conversation
  cache, through the new Host / Origin check, by the registered lines.
- **Across three cached 32 GB runs** (32c, 32c2, r120), the task half held at 11 / 12 each time, with a different task
  failing in each.
- **Not covered here:** 24 GB, which was dropped for tonight, and 45 GB, whose cached arm 45c passed on 80c8041.

## Data
results/agentcc_r120/ holds:
- the run's files (provenance.txt, MANIFEST copy, probe, results, runner log, tasks.out, serve.log, convcache.log,
  sandbox.sb);
- the 24 profiles;
- per task: check.txt, stderr and diff (from checks/);
- tool_calls.jsonl, the score, the screen and the fanwatch lines.

All of it was scanned with the public build's leak regex. The raw transcripts and the request log stay local in
data/agentcc/runs/r120b/.
