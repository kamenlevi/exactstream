# PREREG: does Claude Code work with our local model? A tool-call probe and 12 coding tasks per memory setting (written Tue 2026-11-10 23:xx; before any run)

Written and committed before any run of this test.

- **Request:** Kamen's top priority, relayed by the orchestrator at 23:0x: full Claude Code support with our local model by
  Thu 12 Nov 17:00. This test says whether it works.
- **Who:** EvalSuite designs and scores it. Leap owns the server and the sandbox, and its runner runs the test.

## What runs
- **Claude Code:** the real binary /Users/kamen/.local/bin/claude (its version goes into the run log), with an isolated
  CLAUDE_CONFIG_DIR and Leap's environment (tools/claude_local.sh):
  - ANTHROPIC_BASE_URL set to the local server, ANTHROPIC_AUTH_TOKEN=local;
  - model exactstream, side lane exactstream-side, subagents exactstream-sub;
  - Leap's context-window variables, and CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1.
- **The call:** `claude -p "<prompt>" --output-format stream-json --verbose --max-turns 25 --allowedTools Bash Edit Glob
  Grep Read Write`.
- **The server:** Leap's tool-mapping serve.py and agentio.py (working copy ~/Leap/release/cleanclone/agent/clone). It is
  pinned at a Leap commit on Wed morning; the hash goes below as a dated line before the first run. The runner starts it
  per setting with `serve.py --mem 45 | 32 | 24`.
- **The sandbox (Leap's sandbox-exec profile):** it covers Claude Code's whole process tree.
  - Writes go only to the task folder, the isolated config folders and the temp folder.
  - Network goes only to localhost, and private home folders are unreadable.
  - Bash, python3 and git run inside it.

- **Pinned server (dated line, Tue 23:2x, before any run):** Leap commit **39947b1** (2026-11-10 23:23:19, ~/Leap,
  release/next, with SHA256SUMS). Its SHA256SUMS lists serve.py c9d0e5b5051fbea1..., agentio.py 60e4520efc8de3e3...,
  tools/claude_local.sh e5b900fae5fada10... and tools/claude_code.sh 0752be2a042cc66f....
  - **Re-pinned before any run:** Leap first named 243874c, then 39947b1 at 23:2x.
  - Between them serve.py and claude_local.sh are unchanged (EvalSuite checked SHA256SUMS and the diff). agentio.py only
    adds a `compact` flag to the request metadata meant for BigMoE's cache, which marks Claude Code's compaction
    requests. The rendering, tools and lanes are unchanged. tools/claude_code.sh is a new launcher helper.
  - The other changed files are Leap's notes, tests and its runner (data/agent_cc_1111). EvalSuite spot-checked
    agentcc_tasks.py against this PREREG: the per-setting timeout, the process-group kill, the check copied in after
    claude, and the results.jsonl fields.
  - The runner stops if the clone's files differ from those sums. Each run folder records provenance.txt: the sha256
    checks, the commit, `claude --version`, the manifest's sha256, the setting and the port.
  - Leap's smoke test with a stand-in engine (t01, t11) checked the plumbing only: git commit, transcript, check rc 1,
    results.jsonl. It is not a run of this test.
  - The task runs send a tools array byte-identical to data/agentcc/cc_tools_2.1.321.json. Neither Python has pytest, so
    the tasks rely on unittest.
- **Schedule (Leap):** 32 GB from 00:30 Wed (last start 04:00); run folder
  /private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent/agentcc_runs/32_1111.
  45 GB on Wed night. 24 GB tonight only if time is left.
  - **Schedule moved (dated line, Wed 01:2x, before any run; the design is unchanged):** the night window is now
    01:00-14:00 (Kamen, via the orchestrator). Leap re-armed the same runner on the same folder and pin (its runner.log:
    "from 2026-11-11 01:00:00, last start 2026-11-11 08:00:00"). Two short Leap jobs go first, so the start is expected
    around 02:15-02:30.
  - Leap's cached runs (BigMoE's conversation cache) are on hold. Any such run is a new server config, and it gets its own
    amendment here before it starts.

## Settings and order
| Setting | Role | Timeout per task |
|---|---|---|
| 32 GB | verdict (the fast setting today) | 1,200 s |
| 45 GB | verdict, if it completes by Thu 12:00 | 3,000 s |
| 24 GB | information, if time allows | 1,200 s |

- **Order:** 32, then 45, then 24. A setting missing any of the 30 probe calls or the 12 tasks has no verdict.
- **Why 45 GB gets a longer timeout:** without a conversation cache every turn re-prefills ~12K tokens. Leap measured or
  priced the time to first token at 30.5-32.2 s at 24 GB, ~28 s at 32 GB (PF-A) and ~95 s at 45 GB.

## 1. The tool-call probe (tools/probe_cc.py)
- **The calls:** 30 calls on the server's Anthropic endpoint (/v1/messages). Each carries Claude Code's real tools array,
  which Leap took verbatim from a Claude Code 2.1.321 session.
  - The file is data/agentcc/cc_tools_2.1.321.json (14 tools, sha256 2a0590bd15df5ef4...). It is not committed, because
    it is Claude Code's own text.
  - A short system prompt names the working directory.
- **The cases:** 5 each for Read, Bash, Grep, Glob, Write and Edit. Each Edit case carries a prior Read turn, as Claude
  Code requires.
- **A call passes when all of these hold:**
  - a tool_use comes back;
  - every tool name used is in the array (no invented tool);
  - the first tool_use is the expected tool;
  - its input fits that tool's input_schema (required keys, types, enums);
  - the case's own checks hold;
  - file_path is absolute for Read, Write and Edit.
- **The scorer was self-tested** on ideal inputs (all pass) and on bad ones (a wrong type, a missing key and a relative
  path are all flagged).
- **Line, per setting:** at least 28 of 30 pass AND 0 calls with an invented tool.

## 2. The tasks (tools/agentcc_build.py -> agentcc/manifest.json)
| Id | Task | The hidden check |
|---|---|---|
| t01_fix_test | fix the bug a failing test shows | the tests pass, the test is unchanged, extra cases hold |
| t02_add_function | add slugify() and its unit test | hidden cases, a test added, the tests pass |
| t03_config | two edits in config.json | the exact final JSON (other keys unchanged) |
| t04_traceback | fix a bug from a pasted KeyError traceback | hidden cases, report.py prints 23.0 |
| t05_rename | rename a function in 4 files, tests included | no old name left, call sites and tests work |
| t06_cli_flag | add an argparse --shout flag | both outputs exact |
| t07_import | fix an import after a module move | no file moved, the import and a call work |
| t08_readme | add a Usage section with a fenced command | heading, fenced command, old text kept |
| t09_refactor | extract a duplicated formatter | the helper exists, the outputs are unchanged, one copy left |
| t10_type_hints | annotate every function | the AST has every annotation, the behaviour is unchanged |
| t11_shell | fix a line-count shell script for subfolders | prints 7 |
| t12_requirements | add a pinned line to sorted requirements | the exact final file |

- **Validated:** `python3 tools/agentcc_build.py --validate` shows that each check fails on its template and passes on
  template + reference solution, for all 12 tasks.
- **The checks:** /usr/bin/python3 (3.9), standard library only.
- **Per task, Leap's runner:**
  1. copies the template to a fresh folder (git init and an initial commit);
  2. runs claude under the setting's timeout;
  3. copies the hidden check in as .agentcc_check.py (so the agent never sees it) and runs `/usr/bin/python3 -I -B
     .agentcc_check.py`, sandboxed;
  4. saves claude's stream-json transcript as <run>/<id>.jsonl, the server's request log, the check output, and one line
     in <run>/results.jsonl: id, claude_rc, timed_out, wall_s, check_rc.
- **A task passes** when its check exits 0 on the final repo state.
- **Line, per setting:** at least 10 of 12 pass (83 %).

## The verdict
- Per setting: **Claude Code works at that setting when both lines pass** (tools/agentcc_score.py prints AGENTCC <setting>:
  PASS / FAIL / INCOMPLETE).
- There is no reference run, because the cloud model is not run. Everything below is information.

## Information (no verdict)
- **Per task:** turns (the result's num_turns), time per turn (wall / turns), tool calls, tool errors, invented tools, and
  how claude ended.
- **Failure type** for each failed task; the first match wins:
  1. timeout;
  2. loop (ended at max turns, or 3 identical tool calls in a row);
  3. bad tool call (a tool name outside the session's tools, or a tool error that reports invalid input);
  4. gave up (no Edit or Write call);
  5. wrong edit (the rest).
- **Per probe call:** the reason for each failure.
- **From the server log:** time to first token.
- The scorer was tested on a synthetic run with one task of each type, and it labelled each one as defined.

## Amendment 1 (Wed 2026-11-11 03:53: after the 32 GB run, before any other run)
- **Why:** the 32 GB run (docs/RESULT_agentcc_32_2026-11-11.md) exposed two scorer bugs, both EvalSuite's.
  - Probe case 10's check rejected a correct `ls -la`, because it looked for the substring "-a".
  - The scorer flagged a false "invented" tool: the init event lists the subagent tool as Task, while the tools array
    sent to the model calls it Agent.
- **The changes** (tools/probe_cc.py and tools/agentcc_score.py, in this commit):
  1. Case 10's check accepts `ls` with any flag group containing a or A, or --all.
     - It was self-tested on 11 commands.
     - Re-checked on the 32 GB run's stored calls, only case 10 changes result.
  2. The scorer counts Agent as a session tool when the init event lists Task.
- **Disclosure:** both changes were written after seeing the 32 GB outputs.
  - The 32 GB verdict was scored under the registered checks, and it is not re-scored: it stays FAIL, 27 / 30.
  - The changes apply to every later run: a 32 GB task rerun, 45 GB and 24 GB.
- **Unchanged:** the lines (probe at least 28 / 30 with 0 invented; tasks at least 10 / 12), the cases, the tasks and the
  failure types.
- **Next:** the sandbox fix and any 32 GB task rerun get amendment 2. It is written when Leap sends the profile change,
  the folder and the start time, and before that run starts.

## Amendment 2 (Wed 2026-11-11 03:57: before the 32 GB task rerun starts)
- **What runs:** the 12 tasks again at 32 GB, in Leap's folder agentcc_runs/32_1111b.
  - Leap's runner is scratchpad/agent/agentcc_rerun.sh (sha256 d13479ba447ef9f4...).
  - There is no probe. It calls the API directly, so the sandbox could not have affected it.
- **The one change:** Claude Code's environment gets CLAUDE_CODE_TMPDIR=<run>/cctmp.
  - Claude Code then makes its temp folder for each working folder inside the run folder, which the profile already
    allows.
  - The profile generator is unchanged; only the run folder's path differs.
  - Leap's mock check (scratchpad/agent/bashfix_check.sh, 03:54, no GPU): without the fix it gives the same EPERM. With
    it the command runs, and a write to $HOME is denied.
- **Unchanged from 32_1111:**
  - the clone's serve.py, agentio.py and tools/claude_local.sh at pin 39947b1, checked against that commit's
    SHA256SUMS;
  - serve.py --mem 32 with its request log, on port 8124;
  - agentcc_tasks.py (sha256 3e84be3864b4017b..., the cold run's file), the manifest a44aaeeeefea, the base config and
    the launcher;
  - claude 2.1.321, the 1,200 s timeout, --max-turns 25 and the allowed tools.
- **Validity:** the run counts only if Bash works, meaning no Bash result in any task holds the EPERM mkdir error. Leap's
  runner stops at once if t01's transcript holds it.
- **Role:** information. The 32 GB verdict stays FAIL; it was set by 32_1111's probe half.
  - The tasks are scored against the registered task line (at least 10 / 12) with tools/agentcc_score.py under
    amendment 1. This is the task half, run as designed.
  - The scorer's AGENTCC line will read INCOMPLETE, because the folder has no probe. Only its TASKS line is used.
- **Information reported, as in the first run:** turns, time per turn, tool calls and errors, failure types, the Bash
  calls that ran, time to first token, and the server's lifetime peak.
  - Wall times include heat pauses: fanwatch freezes the server when it runs hot. The RESULT lists them from fanwatch's
    log.
- **Schedule:** it starts once T1d's runner (pid 75700) has exited and the Mac is free, expected ~04:45-05:00. The last
  start is 08:00, and the run takes about 95 min.

- **Amendment 2a (Wed 04:07, before the rerun starts; Leap replaced the armed runner before it started):** one sandbox
  change, so amendment 2's "the profile generator is unchanged" no longer holds.
  - The read-deny entries for ~/.claude and the two account config folders become one rule, `(regex
    #"^/Users/[^/]+/\.claude[^/]*(/|$)")`. It covers every ~/.claude* entry, including ~/.claude.json, which the old
    entries did not.
  - Reason: the orchestrator's 04:0x privacy fix, so that no account folder names appear in files.
  - The rest of the profile, the fix and all settings are as in amendment 2. The runner is
    scratchpad/agent/agentcc_rerun2.sh (sha256 f5473efdd0f0315c...); the folder is still 32_1111b.
  - Leap's mock check (04:05, bashfix_check2.out): Bash runs, none of the 7 ~/.claude* entries is readable, and ~/.ssh
    is denied.
  - The rule only narrows what can be read. The tasks need nothing under ~/.claude*, and Claude Code's own config
    folder is the isolated one.

- **Known limit of 32_1111b (Wed 04:2x; written after its start at 04:23:49, before any result):** `git commit` fails
  for the model.
  - The rerun runs with the real HOME, whose ~/.gitconfig signs commits with an SSH key under ~/.ssh, which the profile
    denies. The error is "Couldn't load public key ...".
  - `git status` and `git log` work, so Claude Code's own git status at startup works.
  - Leap's mock of this setup (scratchpad/agent/bashfix_git.out) also shows `git diff HEAD~1` failing, only because
    the commit before it failed.
  - Leap chose to run as armed, since stopping it would have ended the run. No task prompt asks for a commit.
  - Scoring is unchanged. The RESULT reports every git commit attempt per task and how it ended.
  - The cached runs' neutral HOME (amendment 3) has no signing, and commit works there.

## Amendment 3 (Wed 2026-11-11 04:2x: before any cached run): a neutral HOME for the cached runs
- **Applies to:** the cached 32 GB and 45 GB runs (Leap's planned folders agentcc_runs/32c_1111 and 45c_1111). It does
  not apply to 32_1111b, which runs as amendments 2 and 2a.
- **The change:** Claude Code and its tools run with HOME=<run>/home. The orchestrator approved it at 04:2x, relayed by
  Leap.
  - The folder holds only a .zshrc that exports the PATH of today's runs (recorded in the run's provenance.txt), and a
    .gitconfig with a template identity (agentcc / agentcc@localhost).
- **Why:** Claude Code builds its shell snapshot by sourcing $HOME/.zshrc, so in 32_1111 the user's aliases reached the
  model (t01). A neutral ZDOTDIR has no effect, because Claude Code sources $HOME/.zshrc directly.
- **Leap's mock check** (Leap aa7815b, data/agent_cc_1111/bashfix_home.out): "equal: 33 | differ: []".
  - The 33 items cover PATH, the tool binaries (python3, git, node, pytest and others), git's version, the locale,
    Python's encodings and the env var names.
  - Aliases drop from 5 to zsh's 2 built-ins, and functions from 8 to 0.
  - **Open:** that output shows the neutral folder's git identity as unset, against the template identity above. Each
    run's provenance.txt must record git's identity under the neutral HOME, and EvalSuite checks it before scoring.
  - **Resolved (04:2x):** Leap's final setup check uses the template the runner copies (data/agent_cc_1111/
    bashfix_final.out, Leap b9aa1df). It shows GITID set, and `git commit` works with author agentcc@localhost.
    - Each cached run's provenance.txt records the identity under the neutral HOME (Leap 28650a7).
    - In that check 32 of 33 items are equal. The one that differs is `git status`, which fails under the real
      HOME and works under the neutral one.
    - **Correction (04:2x):** that `git status` difference came from Leap's mock folder, which was not a git
      repo. It did not come from the sandbox (bashfix_git.out). In a repo under the real HOME, status and log
      work and commit fails, as above.
- **What the model sees differently:** `~` and `cd ~` point to the neutral folder. Absolute paths behave as before, and
  `git commit` uses the folder's identity. No task's check depends on HOME or on the commit identity.
- **Comparability:** the cached runs differ from 32_1111b in the served path (the conversation cache) and in HOME. Any
  comparison with 32_1111 or 32_1111b is information.
- **Part 2, before the first cached run:** the pin, the served-path change, the folders, the timeouts, and each run's
  role (verdict or information).

## Amendment 3, part 2 (Wed 2026-11-11 05:2x: before the cached 32 GB run starts): the arm "32 GB with the conversation cache" (32c)
- **Role:** a new arm with its own verdict under the registered lines: probe at least 28 / 30 with 0 invented, tasks at
  least 10 / 12.
  - It is scored with tools/agentcc_score.py under amendment 1, and its line reads "AGENTCC 32c".
  - The 32 GB verdict without the cache stays FAIL (32_1111).
- **Folder and schedule:** agentcc_runs/32c_1111, run by Leap's scratchpad/agent/agentcc_cache.sh (sha256
  46f58cd9a9fdb6de..., armed 05:2x).
  - It starts after the 64K / 128K runs' arms, about 06:00-06:40. The last start is 10:00.
- **Server:** the cache clone at Leap release/next 80c8041. All 7 files are checked against its SHA256SUMS at GO.
  - serve.py is d6b44ed034dbcf7b... (it was c9d0e5b5051fbea1...).
  - agentio.py and tools/claude_local.sh are unchanged (60e4520efc8de3e3..., e5b900fae5fada10...), so the rendering,
    tools and lanes are unchanged.
  - engine/loop/convcache.py is 55631cb8e1760878..., equal to BigMoE 3c068d7.
- **What serve.py changes** (EvalSuite read the diff from 39947b1 to 80c8041):
  1. The conversation cache is on: CONV_CACHE=1, CONV_CACHE_GB 2.5 at --mem 32, and branches on (CONV_CACHE_BRANCH=1).
     A turn prefills only its new tokens, and the cache's lines go to <run>/convcache.log.
  2. With the cache on, every request uses the DFlash2 drafter. Before, requests outside code used "auto".
  3. BIGMOE_IDLE_MS=50, which changes timing only.
  4. The log line adds the cache's figures.
- **Exactness:** Leap's T1e (05:07-05:20; Leap be35dc7) found 12 of 12 cache hits bitwise equal to a cold prefill in this
  setup.
- **Environment:**
  - HOME=<run>/home (part 1);
  - CLAUDE_CODE_TMPDIR=<run>/cctmp and the regex ~/.claude* read-deny (amendments 2 and 2a);
  - the same manifest, base config, launcher, claude 2.1.321, 1,200 s timeouts, --max-turns 25 and allowed tools.
- **The probe runs again,** because the server path changed. The runner calls ~/EvalSuite/tools/probe_cc.py as of
  amendment 1 (sha256 1fffa3b687d332be...). EvalSuite will not edit it while the runner can call it.
- **Disclosure:** amendment 1's case 10 fix was written after seeing 32_1111's probe calls.
  - This server may return the same 30 calls: the cache's prefill is bitwise equal, and if decoding is exact the
    drafter does not change the output. In that case the probe's outcome follows from that fix, since 32_1111's calls
    score 28 / 30 under it.
  - The RESULT reports how many of the 30 calls equal 32_1111's. If all do, the probe half is not independent evidence,
    and the RESULT says so next to the verdict.
- **Validity:**
  - As amendment 2: no Bash result holds the EPERM.
  - provenance.txt must show the pin's sums, the neutral HOME with its PATH line, git's identity under it (agentcc
    <agentcc@localhost>), and CONV_CACHE_BRANCH=1.
  - A run with branches off (the 09:00 fallback) is another arm, and it needs its own dated line before it starts.
- **Information:** as before, plus the cache: hits and misses per request (convcache.log), the tokens prefilled, and time
  to first token on hits against misses.
- **The cached 45 GB run (45c_1111)** gets its own part 3 before it starts, after Leap's T1 check at 45 GB.

## Amendment 3, part 3 (Wed 2026-11-11 05:2x: before the cached 45 GB run starts): the arm "45 GB with the conversation cache" (45c)
- **Role:** a new arm with its own verdict under the registered lines, scored as in part 2. Its line reads "AGENTCC 45c".
  - The 45 GB setting without the cache has not been run, so it has no verdict.
  - The registered condition, a verdict only if the run completes by Thu 12:00, still holds. This run is due to end by
    Wed 13:45.
- **Folder and schedule:** agentcc_runs/45c_1111, run by Leap's agentcc_cache45.sh (sha256 4823f6c847be6dc2..., armed
  05:2x).
  - It runs after the cached 32 GB run and T1-45, with a last start of 12:15.
- **Against 32c (part 2), the changes:**
  - serve.py --mem 45, on port 8126;
  - CONV_CACHE_GB=2.5, set explicitly (serve.py's 45 GB default is 6);
  - the manifest's 45 GB task timeout of 3,000 s.
  - With the cache on, the 45 GB path also uses DFlash2 throughout. It has its own prefill; PF-A is for 32 / 24 GB only.
- **Same as 32c:** pin release/next 80c8041 with its sums, branches on, the neutral HOME, CLAUDE_CODE_TMPDIR, the regex
  read-deny, probe_cc.py as of amendment 1, and the 12 tasks.
- **Gates (the runner's):**
  - It starts only after T1-45 (Leap's t1_runs/45_1111f: the 45-request replay at --mem 45, CONV_CACHE_GB 2.5, branches
    on) ends with BigMoE's verdict PASS and FAIL 0.
  - Its free-memory floor is T1-45's highest server peak, rounded up, plus 3 GB.
  - Otherwise it writes SKIP and runner.done: a skipped run has no verdict, and EvalSuite reads runner.log before
    scoring.
- **Validity:** as part 2, with --mem 45, CONV_CACHE_GB=2.5 and CONV_CACHE_BRANCH=1 in provenance. A swap-guard stop, or
  anything short of 30 probe calls and 12 tasks, means no verdict.
- **Disclosure, as part 2:** the RESULT reports how many of the 30 probe calls equal 32_1111's and 32c's.
- **Memory (information):** Leap's limits note puts chat-45 at 47.5-48.6 GB on 10-30K-token prompts, which is close to
  tonight's 51-52 GB free. A skip or a swap stop is possible.

## Amendment 3, part 4 (Wed 2026-11-11 07:2x: after the cached 32 GB run, before the cached 45 GB run starts): a call that names a check fails the task
- **Why:** in 32c, t12 made three Bash calls outside its task folder. Their descriptions were "Look for sibling task dirs
  and instructions" and "Read task check and stderr files".
  - It listed the run folder, then read its own check output (empty at the time), its stderr, and its own live
    transcript.
  - Nothing it read bears on its check, because the runner copies each check in only after claude ends.
  - Leap's profile allows reads outside its deny list, though. So ~/EvalSuite/agentcc/checks/, the manifest and each
    finished task's .agentcc_check.py are readable from inside a task.
- **The rule, for 45c and every later run:** a task fails, whatever its check returns, when one of its tool calls names a
  check.
  - That means a call whose input contains "agentcc/checks", "agentcc/manifest", "EvalSuite/agentcc" or
    ".agentcc_check".
  - Its failure type is "read the check", first in the order.
  - tools/agentcc_score.py applies it from this commit. It was tested on a synthetic run with one case per path form,
    one timeout and one clean task, and it leaves the scores of 32_1111, 32_1111b and 32c unchanged.
  - The scorer also prints any other call that names EvalSuite. EvalSuite reads each one's result, and the task fails
    the same way if it shows a check's content.
- **Information:** the RESULT lists every call outside the task's own folder.
- **Unchanged:** the lines, the cases, the tasks, the runner and the profile. 45c runs as armed (part 3).
- **32c is not re-scored.** None of its calls names a check, so the rule gives it the same result.
- **For later runs, not 45c:** EvalSuite asked Leap to deny reads of ~/EvalSuite/agentcc, and of the run folder outside
  the task's own folder. That change needs its own dated line before its run.

## Amendment 4 (Wed 2026-11-11 08:1x, after 32c and 45c and before any run uses it): a confirmation rerun at 32 GB with the cache (arm 32c2), with a per-task read-deny profile
- **Asked by:** the orchestrator (08:1x): one confirmation rerun of the cached 32 GB arm, with Leap's per-task read-deny
  profile and the TMPPREFIX fix. There is no 24 GB arm.
  - The orchestrator sends the window, most likely Wed night, and the folder.
  - A dated line before the start adds them, with the runner's path and sha256. The runner's gates are as in
    agentcc_cache.sh: a free-memory floor of 45 GB, and the swap guard.
- **Role:** a new arm with its own verdict under the registered lines: probe at least 28 / 30 with 0 invented, tasks at
  least 10 / 12. Its line reads "AGENTCC 32c2".
  - It is scored with tools/agentcc_score.py as of part 4 (sha256 9bb84b50918a652a...).
  - The part 4 rule applies unchanged: a call that names a check fails its task, even when the profile denies the read.
  - 32c's verdict stands on its own. If 32c2 passes, the RESULT calls the cached 32 GB PASS confirmed, 2 of 2. If it
    fails, the RESULT reports 1 of 2 and which line failed. There is no pooling and no best-of; pooled counts are
    information.
- **The two changes from 32c** (Leap 35b0377, data/agent_cc_1111/profile/):
  1. **A per-task profile:** the tasks run under agentcc_tasks2.py (sha256 98af213d475af195...) instead of
     agentcc_tasks.py.
     - For each task it writes <run>/sandbox_<id>.sb: the run's sandbox.sb, then `(deny file-read-data ...)` of
       ~/EvalSuite/agentcc (which holds the manifest) and of <run>.
     - After that comes `(allow file-read-data ...)` of the task's own folder, its cc_home/<id>, <run>/cctmp and
       <run>/home.
     - File contents and listings are denied; metadata stays readable. The task's check runs under the same profile.
  2. **TMPPREFIX=<run>/cctmp/zsh** in claude's environment, so zsh heredocs work. A .zshrc export does not reach Claude
     Code's Bash tool.
  - **Leap's stand-in check** (profile_check.out, with a control in profile_check_control.out):
    - Under the per-task profile, the task's own folder, git and the heredoc work.
    - Reads of <run>/tasks, another task's folder, runner.log, the task's own transcript, ~/EvalSuite/agentcc and the
      manifest are denied.
    - Under today's profile all of those reads succeed, and the heredoc fails.
    - The reads made by 32c's t12 and 45c's t10 would have been denied.
- **Unchanged from 32c (part 2):**
  - **The pin, by sums** (release/next 80c8041): serve.py d6b44ed034dbcf7b..., agentio.py 60e4520efc8de3e3...,
    tools/claude_local.sh e5b900fae5fada10... and engine/loop/convcache.py 55631cb8e1760878....
    - release/next's working tree now carries Leap's staged window-defaults patch, and the clone's claude_local.sh is
      30a93ccd2c45d831.... The run uses the 80c8041 file.
  - **Claude Code 2.1.321.** The installed claude updated itself to 2.1.322 at 08:15. The run sets CLAUDE_BIN to
    ~/.local/share/claude/versions/2.1.321, and provenance.txt must show 2.1.321.
  - **Server:** serve.py --mem 32, CONV_CACHE=1, CONV_CACHE_GB 2.5, CONV_CACHE_BRANCH=1, DFlash2 throughout and
    BIGMOE_IDLE_MS=50.
  - **Environment:** the neutral HOME (.zshrc df313e2ac75d; git identity agentcc <agentcc@localhost>) and
    CLAUDE_CODE_TMPDIR=<run>/cctmp. The base sandbox.sb equals 32c's apart from the run path.
  - **Settings:** the manifest a44aaeeeefea, the 1,200 s timeouts, --max-turns 25 and the allowed tools.
  - **The probe:** probe_cc.py at 1fffa3b687d332be.... It calls the API directly, so the profile does not touch it.
- **Validity, checked by EvalSuite before scoring:**
  - provenance.txt shows the pin's four sums, claude 2.1.321, CONV_CACHE_BRANCH=1, the HOME with its PATH line, the git
    identity and TMPPREFIX;
  - each task's sandbox_<id>.sb holds the two added rules, naming that task's own folder;
  - no result holds Claude Code's temp-folder EPERM;
  - there are 30 probe calls and 12 tasks, and no swap-guard stop.
  - A different pin or Claude Code version, or the profile without TMPPREFIX, is another arm and needs its own dated
    line before the start.
- **Information:** as for 32c, plus every read the profile denied (per task, from the results), and whether heredocs
  work.
- **Comparability:** against 32c this changes only what a task can read and where zsh puts heredoc temp files. Timing
  comparisons are information.

- **Amendment 4a (Wed 08:2x, before any 32c2 run; Leap found the conflict):** amendment 4 cannot pin Claude Code as
  written.
  - agentcc_tasks2.py (98af213d) sets CLAUDE_BIN="/Users/kamen/.local/bin/claude" itself, and the 80c8041 launcher
    execs ${CLAUDE_BIN}. That symlink now points to 2.1.322, so the run would use 2.1.322.
  - **The fix:** the tasks script becomes agentcc_tasks3.py, which is agentcc_tasks2.py with that one line reading
    CLAUDE_BIN from the environment, default unchanged.
    - EvalSuite reads its diff against agentcc_tasks2.py. Its sha256 goes into the dated line before the start.
  - **The binary:** Claude Code 2.1.321, sha256 abc9feeeca8091d5... (323,686,048 bytes, from
    ~/.local/share/claude/versions/2.1.321).
    - Leap keeps a copy at ~/Leap/logs/claude_versions/2.1.321, so the updater cannot prune it.
    - CLAUDE_BIN may point to either file. provenance.txt records the path, its sha256 and its --version.
  - **Added to validity:** every task transcript's init event reads claude_code_version 2.1.321.

- **Amendment 4b (Wed 08:2x): the dated line before the start of 32c2.** EvalSuite checked each item below against the
  files.
  - **Folder:** Leap's scratchpad agent/agentcc_runs/32c2.
  - **Runner:** agent/agentcc_cache2.sh, sha256 120417d27c069487.... An identical copy is in Leap 68ec944
    (data/agent_cc_1111/agentcc_cache2.sh).
  - **Tasks script:** agentcc_tasks3.py, sha256 b6e24c30d57ea13b....
    - Its diff against agentcc_tasks2.py is the one line 49: CLAUDE_BIN = os.environ.get("CLAUDE_BIN", the old
      default).
    - The runner runs a sha-checked frozen copy from the run folder, and exports CLAUDE_BIN = the pinned binary.
  - **Binary:** Leap's copy ~/Leap/logs/claude_versions/2.1.321, with sha256 abc9feee... and 323,686,048 bytes, the same
    as versions/2.1.321. `--version` prints 2.1.321, and the copy is gitignored.
  - **Launcher:** the clone's tools/claude_local.sh.80c8041, checked against 80c8041's committed SHA256SUMS (e5b900fa).
    - The runner checks serve.py, agentio.py, convcache.py and codex_local.sh the same way, and stops on any mismatch.
    - Leap's dry run (profile/provenance_32c2_dryrun.txt) shows every sum, the frozen copy, the binary, TMPPREFIX, the
      HOME and the identity.
  - **CONV_CACHE_GB=2.5** is now set explicitly. It is the same value: 32c's convcache.log shows a budget of 2.50 GB.
  - **Base sandbox.sb:** the runner's text equals 32c's file apart from the run path.
  - **Window (the orchestrator, 08:2x):** from this line to a last start of Wed 13:00, with everything done by 13:45;
    the Mac is free until 14:00.
    - The runner also waits for its GO file, which Leap creates after this line, and for Kamen idle for 10 min, no other
      engine, lock, PAUSE or HOLD, and 45 GB free memory (NO-SWAP), on two clear polls.
    - The swap guard stops it at +2 GB.
    - It stops the tasks if t01's init event is not 2.1.321, or if t01 holds the EPERM.
    - A run stopped for any of these, or by the orchestrator after 13:45, has no verdict.
  - **The version check:** the guard's parse reads "2.1.321" from the first line of 32c's and 45c's transcripts.
    - Leap's notes (data/agent_cc_1111.md) report the stand-in's init events at 2.1.321. profile_check_tasks3.out does
      not print the version, so this rests on the run's own guard and EvalSuite's check of all 12 transcripts.

- **Amendment 4c (Wed 08:4x): 4b is void; the dated line for 32c2 with a hardened sandbox.**
  - **Why 4b is void:** Leap's security reviews (08:24 and 08:3x) found three holes.
    - The base profile let the agent write anywhere in the run folder, while the unsandboxed runner opened its outputs
      there, and the check copy in the task folder, following symlinks. A planted link could make the runner overwrite
      a file outside the sandbox.
    - A planted <run>/home/.gitconfig could run code in the next task's unsandboxed git commit.
    - Processes that outlive claude could change the folder the check grades.
    - agentcc_cache2.sh was armed at 08:23:57 and stopped at 08:24:52, while it waited for GO. Nothing ran.
  - **The finished runs stand.** 32_1111, 32_1111b, 32c and 45c hold no symlinks, default .git/config files and no hooks.
    32c's and 45c's HOME folders hold the template .zshrc and .gitconfig, unchanged (EvalSuite, 08:2x-08:3x).
  - **The run (Leap 91dde32):**
    - folder agentcc_runs/32c2b, which must be fresh (the runner refuses an existing folder);
    - runner agentcc_cache4.sh, sha256 ff16664f832c59c4... (an identical copy is in Leap's data/agent_cc_1111/);
    - tasks script agentcc_tasks5.py, sha256 28caec4c6f92468c..., frozen and sha-checked by the runner.
    - EvalSuite read every diff on the way (tasks2 -> 3 -> 4 -> 5, cache2 -> 3 -> 4) and the dry run
      (profile/provenance_32c2b_dryrun4.txt).
  - **What tasks5 changes from 4b's tasks3:**
    - **Claude's write set per task** is its task folder, cc_home/<id> and cctmp/<id>, plus the base's /private/var/folders
      and /dev.
      - CLAUDE_CODE_TMPDIR is cctmp/<id>, and TMPPREFIX is cctmp/<id>/zsh.
      - HOME is read-only. The base profile names no part of the run folder, and tasks5 refuses one that does.
      - The read denies are as in amendment 4.
    - **Runner files:** the runner opens its files in the run folder with O_NOFOLLOW, and O_EXCL when new.
    - **Lingering processes:** after claude ends, sandboxed processes with a cwd or an open file in the task's folders are
      killed (up to 5 rounds, logged to lingering.jsonl). If any remain, the check does not run and the task fails.
    - **The check runs on the runner's copy checks/<id>.**
      - The copy is made through O_NOFOLLOW directory handles: links are copied as links, and other file types are
        skipped. The check is placed in it through a directory handle.
      - The check runs under its own profile, sandbox_check_<id>.sb, whose only writable folder is that copy.
      - The check command and files are unchanged.
  - **Leap's stand-ins** (profile_check3_new / _old.out and profile_check4_new.out, with the 2.1.321 copy):
    - A canary file outside the run folder, targeted by a link the agent planted at the check's name, stayed unchanged
      under tasks4 and tasks5. Under tasks3 it was overwritten.
    - Denied: writes to the run folder, a link at the next task's transcript name, and writes in HOME or in another
      task's temp folder.
    - Working: the task's own folder, a git commit, its temp folder and heredocs. The reads are denied as in 4.
    - A racer that stayed in the task folder was killed. One that left the folder wrote into it later, but not into the
      graded copy.
    - The checks passed, and the init events read 2.1.321.
  - **Unchanged from 4-4b:**
    - the pin (80c8041 sums and launcher) and Claude Code 2.1.321 (the copy, abc9feee...);
    - the server settings, HOME's contents, the probe and the manifest;
    - the checks, byte-identical;
    - the timeouts, turns and tools;
    - the gates (GO only after this line, Kamen idle, no other engine or lock, no PAUSE / HOLD, NO-SWAP >= 45 GB, two
      clear polls), the swap guard and the t01 guards.
  - **Window, unchanged:** last start 13:00, done by 13:45. Leap's watchdog stops the run at 13:45, and a stopped run has
    no verdict.
  - **Added to validity:**
    - provenance.txt lists claude's write set, the check's write set and the frozen tasks5 sha;
    - each sandbox_<id>.sb and sandbox_check_<id>.sb holds only that task's rules;
    - a symlink in the run folder outside tasks/<id>, checks/<id>, cc_home/<id> and cctmp/<id> voids the run. Links
      inside them are listed in the RESULT.
  - **A task whose check did not run** (processes left, or the check not placed) is a FAIL, as the runner records it.
  - **Check integrity (EvalSuite's call on Leap's third point):** the checks stay byte-identical, so the confirmation
    keeps 32c's grader.
    - The risk: each check puts the task folder first on sys.path and imports the agent's code in-process. Code the agent
      left could end the check early, or shadow a module the check imports later (t10 imports ast and math after the
      insert).
    - So the scorer screens each graded copy against its template with tools/agentcc_screen.py, run on
      `tools/agentcc_diffs.sh <run> <out> checks` (EvalSuite d1b22d1 and this commit).
    - It flags a new top-level importable file (name.py, .pyc or .so, or a new package), and any added .py line with an
      exit or import-machinery pattern.
    - Each flag is reviewed. The task FAILS ("gamed the check") if the flagged code runs while the check imports or calls
      the task's code and can change the outcome. The RESULT lists every flag and its decision.
    - Information, not re-scored: the screen finds 0 flags in 32_1111, 32_1111b, 32c and 45c.
  - **Diffs:** the diffs are of checks/<id>, what the check graded. If tasks/<id> differs from it (a late write by a
    process that escaped), the RESULT reports it.
  - **Comparability with 32c:** honest work is unaffected.
    - A task can no longer write in the rest of the run folder or in HOME. No tool call in 32c or 45c wrote in the run
      folder outside its own task folder.
    - The check grades a copy of the same files.
    - Timing comparisons are information.

## Known limits
- **Small N:** 12 tasks and 30 calls bound the failure rate only loosely.
- **Small tasks:** they are single-step, Python plus one shell task, with no long sessions and no large repos.
- **Single-turn probe:** the probe tests the choice of tool and its arguments, not multi-step plans.
- **Timing:** these are times without a conversation cache. They are information for this build, not a speed claim.
