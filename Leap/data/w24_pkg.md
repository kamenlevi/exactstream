# 24 GB package: `--mem 24` (w13b as gated) in the local clean clone (Tue 10 Nov 08:4x-)

Asked by the orchestrator (Tue 08:4x): the w24 card gate passed on w13b at 08:41, and Kamen wants the 24 GB version out by
Wednesday. Built in ~/Leap/release/cleanclone/final/clone only; nothing is pushed, and publishing waits for Kamen's word.

## What was built (CPU only, 08:46-08:55)

- **Configs:** `configs/lossy-24.env` and `configs/chat-24.env`, identical (copies in data/w24_pkg/):
  - BigMoE's data/stack/card_w24d_env.txt env line verbatim (git blob e06f495e, commit 87dd1a3).
  - `C=100`: runner 61 took C from the env comment's "Sat 100".
  - `SNAPSHOT=w24`.
  - `R57_PFCHUNK=1024` on its own line. It equals the loop's default, so the gate ran this value.
    - W24e (BigMoE, Tue 09:00) keeps it. No smaller chunk qualified under the pre-registered rule (tokens AND hashes
      equal to 1024 on all six prompts).
    - 512 matched 0/6 and peaked at 24.345 GB. 256 got under 24 GB (23.954 GB) but matched tokens on 1/6 and hashes on
      0/6, with the first token 2.0-2.9x slower.
    - The repeat at 1024 matched 6/6 (25.025 vs 24.961 GB). Source: BM:data/w24/w24e_report.txt.
    - So the label stays: under 24 GB on prompts up to about 560 tokens, up to 25.1 GB on longer ones.
  - Parsed by the package's read_config: the gate's 42 keys and values are all equal, plus R57_PFCHUNK only.
- **Engine:**
  - `engine/loop/r57_screen_w24.py`:
    - Built by applying BigMoE's `diff -u data/r57/snap/rel1/r57_screen_rel1.py data/r57/snap/w24/r57_screen_w24.py`
      (the w24 file last changed at 016cb7f; 6 hunks) to the package's rel1 copy with `patch -F0`. It applied cleanly, so
      the added code is byte-identical to BigMoE's.
    - Then make_release.py's step-4 scrub: 1 quote in a new comment redacted, AST without docstrings unchanged. One
      comment reworded, AST unchanged.
    - The helper modules (inert_span, pflm, relax_delta, relax_span) are identical between BigMoE's rel1 and w24
      snapshots.
  - `engine/bigmoe/stream.py`: the only bigmoe/ change between 0b4df8c (rel1) and 016cb7f, patched with -F0. Its AST
    equals BigMoE 016cb7f.
    - It adds an optional `gate_req` bound that only R57_GATEREQ=1 sets. w13b does not set it, so it is inert for every
      config, 24 GB included.
  - ext/ is unchanged between the two commits.
- **run.py** (diff in data/w24_pkg/run_py.diff):
  - `--mem 24`; DEFAULT_C["24"] = 100.
  - read_config merges several env lines in order. Every existing config has one line, so they parse as before.
  - `snapshot(cpath)`: a config's `SNAPSHOT=` line picks its decode loop, else engine/loop/SNAPSHOT (rel1).
- **chat.py, serve.py** (local only, not committed): `--mem 24`, and the decode loop from the config via `R.snapshot`.
- **README** (clone): a 24 GB row in the configs table, with the memory label.
- **Dry run:** `run.py --config lossy --mem 24 --keys humaneval/0 --max-tokens 2048 --dry-run` printed lossy-24, C 100,
  cap 2048, snapshot w24, and the gate's env.
- **Differences from the gate's launch** (EvalSuite tools/day_runner61.sh, arm stage):
  - BIGMOE_EXT_BUILD is "build" in the package, against build_qom. Every package config uses one extension build; the
    45 / 32 lossy configs that ran build_qom in research matched their references on it (FINAL clean-clone test).
  - run.py sets R57_PROMPTS, HF_HUB_OFFLINE and PYTHONDONTWRITEBYTECODE.
  - run.py loads the engine its own way; the gate went through tools/r57_wrap.py.
  - The GPU replay below checks whether any of this matters.

## GPU checks (armed: data/w24_pkg/runner_w24.sh, queued after W24e and c47)

1. Replay 3 IFBench + 3 GPQA gate rows at the gate's CAP 2048 (no QOM on those sets). The tokens must equal
   ~/EvalSuite/results/card_w24_fwd/decode.jsonl, and the prompt tokens n_prompt. The GPQA / IFBench items go into the
   clone only for this run and are removed after (never published).
2. BigMoE W24d's 12 HumanEval keys (w13eh_100) at CAP 256: whole decode against 130.33 pooled, lifetime peak against
   23.78-23.82 GB, first token median against 3.54 s.

### Results (Tue 09:59:40-10:02:50; share lock 'MEM 26 GB', after c47, nothing else running; NO-SWAP 48 GB; swap flat at 6,683 MB)

| check | result | reference |
|---|---|---|
| replay, 3 IFBench rows (CAP 2048) | **tokens equal on 3/3** (60, 61, 63 tokens); prompt tokens equal (71, 39, 32) | card_w24_fwd/decode.jsonl |
| replay, 3 GPQA diamond rows (CAP 2048) | **tokens equal on 3/3** (627, 714, 798 tokens); prompt tokens equal (161, 222, 183) | card_w24_fwd/decode.jsonl |
| replay process | lifetime peak 23.79 GB; first token 3.39 s median | gate max 25.112 GB (on longer prompts) |
| 12 HumanEval (BigMoE W24d keys, CAP 256) | **tokens identical on 12/12** vs both of BigMoE's runs (w13eh_100, w13fh_100); 2,610 tokens | BigMoE W24d |
| HumanEval whole decode | **135.9 tok/s** (one pass) | 130.33 pooled (130.2 / 130.5) |
| HumanEval lifetime peak / first token | 23.78 GB / 3.63 s median (3.29-4.82) | 23.78-23.82 GB / 3.54 s |

- The packaged lossy-24 reproduces w13b exactly. Its tokens equal the gate's on the no-QOM sets, and equal BigMoE's on
  HumanEval, where QOM is on (the same 12 prompts in the same process order).
- It reaches the gated speed and memory: +4.3 % on HumanEval in one pass, a share lock with nothing else running.
- Files (data/w24_pkg/): check_summary.txt; he12.out with the run's log, runs.json, tokens.json and modules.json
  (cli_1110_100130); replay_runs.json (statistics only).
- The replay's decoded GPQA answers are NOT kept: GPQA text stays out of public files, as in EvalSuite's own records.
- The GPQA / IFBench items were in the clone only for the run and were removed after it.
- ~/Leap/data/pkg24_check.done was touched at 10:03 for BigMoE's W24f.

## Not verified: a 36 GB M5 Max (not available here)

- **Memory:**
  - The engine sets `mx.set_wired_limit(max_recommended_working_set_size)` (R57_WIRED=1). It never raises the system's
    limit, and sets no MLX memory limit; the MLX cache is capped at 0.25 GB (R57_CACHE_GB).
  - Here the recommended working set is 55.7 GB of 64 GB (81 %). On a 36 GB Mac it is reported at about 27-29 GB
    (not measured), against the gate's 25.112 GB lifetime footprint, which also counts CPU-side memory.
  - With a ~25 GB engine, a 36 GB Mac has ~11 GB for macOS and apps. Long prompts reach 25.1 GB. A browser beside it can
    push other apps into swap. The engine's buffers are wired, so they are not swapped.
- **GPU type:** MLX reports this Mac's GPU as `applegpu_g17s`. A Max chip reports a different suffix, and MLX picks some
  matmul tile shapes by that suffix. The logits' last bits, and so the tokens, can differ from these references there.
  The quality gate's scores should carry over, but nothing was run on that Mac. Speed scales with its GPU and SSD.
- **SSD:** ~24 MB read per kept token (W24d) is ~3.1 GB/s at 130 tok/s. A smaller or slower SSD lowers tok/s.
- **Disk:** the 68.5 GB converted store, plus ~20 GB free for swap; the conversion needs the model's own files beside it.
- **Cores:** nothing pins core counts. The C extension uses GCD's global queues, with one spinner thread at
  QOS_USER_INTERACTIVE; R57_PFD_THREADS=12 is PF-D only, which w13b does not use.
- **This Mac's tooling:** the engine checks ~/.local/share/quiet/PAUSE (this Mac's heat guard) every 50 ms. On another
  Mac the file does not exist, so nothing pauses.
- **Thermals:** a MacBook Pro's sustained clocks under a long run were not measured.
- **Time budget:** R57_HARD=1200 (in the gated env line) stops new checks after 1,200 s in a run.py batch. chat.py and
  serve.py lift it.

## Public candidate with the 24 GB line (Tue 10:16-10:24; orchestrator 10:1x: fold in, build locally, no push)

- **Source:** release/src (commit 429f844).
  - run.py: --mem 24, multi-line configs, snapshot().
  - configs/lossy-24.env and chat-24.env.
  - README: two prose sentences and the Install line; it stays 40 lines with no tables.
  - INSTALL.md: the 36 GB notes in plain words.
  - CONFIGS.md: a 24 GB row.
  - tools/clean_clone_test.sh: each config runs the first two prompts of its own reference.
- **make_release.py `--extra-snap w24`:**
  - Archives data/r57/snap/w24 at the same BigMoE commit and places only the decode loop.
  - Refuses the build if the snapshot's other modules differ from rel1's.
  - Gives it the same path rewrites, plus one non-private comment rewording.
  - refs: release/refs_rel1_w24.json, where lossy-24 and chat-24 point at BigMoE's w13eh_100 run.
- **Build:** `make_release.py rel1 --commit 016cb7f --extra-snap w24 --refs release/refs_rel1_w24.json --es-commit 98c6072d`.
  - Result: release/out/pkg_w24 (0f076a5), 239 files, 6.5 MB.
  - Leaks 0, unsafe quote lines 0, live research paths 0, syntax errors 0; 6 quotes redacted (the w24 loop's one among
    them).
  - Against the previous candidate it differs only in the intended files.
  - Its w24 loop and stream.py are byte-identical to the copies the morning check passed (after @ROOT@).
- **Docs-only rebuild:** release/out/pkg_w24b (b02d0ba) with INSTALL.md's test sentence updated. It differs from pkg_w24
  only in INSTALL.md and MANIFEST.json.
- **Clean-clone test** (`prep`: install.sh rc 0 in 8 s, 8 dry runs OK; `run`: 10:18:37-10:23:06, 271 s, share lock,
  NO-SWAP 51 GB, swap flat): **CLEAN-CLONE PASS**.
  - All 8 configs matched their references 2/2: exact-45 and exact-32 prefix-equal; lossy-45 / 32, chat-45 / 32 and
    lossy-24 / chat-24 prefix-equal with the reference's history.
  - Closure 0 / 0 for every config: no module or opened file outside the allowed roots.
  - The chat-45 text prompt answered sensibly.
- **serve.py first real-engine test** (`--mem 24`; serve.py and chat.py copied into the test clone only; 10:23:09-10:24:05):
  **SERVE TEST PASS**.
  - Ready in 15 s.
  - OpenAI stream: 20 chunks, [DONE], finish_reason stop, usage 30 + 89 tokens.
  - Anthropic stream: 21 events in the right order, stop_reason end_turn, 62 output tokens.
  - The server's lines: 89 tokens at 31.9 tok/s (first token 2.16 s) and 62 at 32.1 tok/s (1.81 s); lifetime peak
    23.68 GB; 0 errors.
  - Stopping: it ignored SIGINT for 30 s, then exited on SIGTERM, with memory freed (NO-SWAP 53 GB after). This comes from
    the test: a background job of a non-interactive shell starts with SIGINT ignored, and Python then keeps it ignored
    (no KeyboardInterrupt). Ctrl-C in a terminal is a normal SIGINT (not tested here). serve.py handles KeyboardInterrupt
    only. On SIGTERM (`kill`, launchd) it stops without its shutdown path. A two-line handler would fix both; not done
    (the orchestrator: report, don't rework).
- **Evidence:** data/w24_pkg/candidate/: report.txt, cand.log, serve.log, openai.sse, anthropic.sse, models.json and the
  two scripts.

## Docs with the W24f numbers, and serve.py's stop path (Tue 10:4x; orchestrator 10:3x)

- **README:** "and generates about 130 tokens per second on HumanEval, 91 on MBPP and 31 on chat"; still 40 lines.
- **RESULTS.md:** a 24 GB row, pooled from BigMoE's run records (two passes each), in the table's order: whole decode / whole
  request / first token median / lifetime peak.
  - HumanEval (W24d w13eh + w13fh): 130.3 / 40.8 / 3.54 / 23.82.
  - MBPP (W24f bf1m + bf2m): 91.2 / 26.1 / 3.34 / 23.74.
  - IFEval (W24f bf1i + bf2i): 30.6 / 18.1 / 3.09 / 23.66.
  - Math: not measured.
  - The pooling reproduces BigMoE's per-run figures exactly. MBPP is 91.2 when pooled per prompt; 91.3 comes from pooling
    the two runs' rounded figures.
- **CONFIGS.md:** the same three speeds in the 24 GB row, with math not measured.
- **Docs-only rebuild:** release/out/pkg_w24c (8a3adde). Leaks 0, unsafe 0, live paths 0, syntax 0. It differs from the
  tested build pkg_w24 (0f076a5) only in CONFIGS.md, INSTALL.md, README.md, RESULTS.md and MANIFEST.json, so the
  clean-clone pass still covers its code.
- **serve.py (local only, uncommitted, a separate step for Kamen):**
  - The fix: `signal.signal` for SIGINT and SIGTERM, raising KeyboardInterrupt into the existing shutdown path
    (srv.shutdown(), "stopped").
  - The test, on the mock engine (no GPU): started from a script as a background job, the case that failed at 10:24, then
    one streamed request. **STOP TEST PASS** for both SIGINT and SIGTERM: exit within 1 s, rc 130, "stopped" logged, port
    free.
  - Control: the pre-fix copy under the same test is still running 10 s after SIGINT (FAIL), so the test catches the old
    behaviour.

## LIMITS.md 24 GB lines; candidate pkg_w24d (Tue 10:5x; orchestrator after NewIdeas' number check item 3)

- **LIMITS.md** gains:
  - 24 GB memory: under 24 GB on prompts up to about 560 tokens, up to 25.1 GB on longer ones.
  - 24 GB against Strata, same Mac and prompts: Strata's first token is 1.9-2.5x faster on every set; on chat its whole
    request is ahead (20.3-21.0 against 18.1 tok/s); our decode is ahead on code (3.3-5.0x) and 1.1x on chat (SMALL_RAM
    section 6.5, checked).
  - Math was not measured at 24 GB.
  - The clean-clone line now reads "IFEval; HumanEval for the 24 GB configs".
- **Docs-only rebuild:** release/out/pkg_w24d (886ad0d). Leaks 0, unsafe 0, live paths 0, syntax 0. It differs from the
  tested build pkg_w24 (0f076a5) only in CONFIGS / INSTALL / LIMITS / README / RESULTS .md and MANIFEST.json. This is
  the current candidate.
- **Staged, not applied:** release/staged/limits_pfd_erratum.patch replaces LIMITS lines 12-13 with last night's PF-D 10K
  / 30K results. Apply only if Kamen approves the erratum; release/staged/README.md has how.

## 1.1.0 candidate (Tue 14:2x): release/out/pkg_110 (dac4850)

- **Built from release/src**, which carries:
  - the v1.0.1 corrections: LIMITS' PF-D lines, run.py's --env help, the 32 GB long-prompt peaks;
  - the 24 GB line;
  - the math speed: about 41 tok/s (W24g: 40.8 / 27.4 / 2.86 / 24.01);
  - the measured label: under 24 GB to about 500 tokens, then about 25.0 / 25.7 / 25.8 / 28.2 GB at 3K / 7K / 10K / 30K;
    INSTALL says a 30K-token prompt on a 36 GB Mac leaves about 8 GB;
  - CITATION 1.1.0.
- **Build checks:** leaks 0, unsafe 0, live paths 0, syntax 0.
- **Against the clean-clone-tested build** pkg_w24 (0f076a5): docs (CITATION, CONFIGS, INSTALL, LIMITS, README, RESULTS),
  MANIFEST.json, and run.py's help string only (the v1.0.1 correction). So the clean-clone PASS covers its code.
- **Push tomorrow** on the go: `PKG_DIR=release/out/pkg_110 push_pkg.sh pkg` on top of main 0130ead (v1.0.1) keeps docs/
  as main has it. The 1.1.0 site (SMALL_RAM etc.) comes from the Editor and Breakthrough, folded in with AMEND.

## 1.1.0 candidate pkg_110c (6ef9142): clean-clone check on the real engine, PASS (Tue 20:57-20:59)

> (a quote from the project owner, kept private)
later."). Its clone was prepped at 20:25 (install.sh rc 0); the armed runner (candidate_110c/runner_v110.sh) waited for a
free Mac (Kamen idle, no other lock or engine, NO-SWAP >= 34) and ran at 20:57 with NO-SWAP 40 GB, swap flat at 6,862 MB.

| step | result | lifetime peak |
|---|---|---|
| lossy-24, HumanEval/68 + /16, cap 256 | 2/2 keys prefix-equal with lossy-24_w13eh.json; closure 0 / 0 outside; 131.6 tok/s whole decode | 23.71 GB |
| chat-24, the same keys | 2/2 prefix-equal with chat-24_w13eh.json; closure 0 / 0; 133.1 tok/s | 23.68 GB |
| chat-24, text prompt ("Explain how a refrigerator works.") | a correct 77-token answer; 27.2 tok/s, first token 2.12 s | 23.49 GB |
| chat.py --mem 24, one message | rc 0; 72 tokens, 25.3 tok/s, first token 2.14 s, MTP head 2.35 tokens per check | 23.63 GB |
| serve.py --mem 24, one streamed OpenAI + one streamed Anthropic request, then SIGTERM | [DONE] + finish_reason; message_stop + 4 text deltas; one "stopped" line | - |

**CLEAN-CLONE PASS** at 20:59:16 (116 s). Every peak stays under 24 GB. The candidate is ready for tomorrow's go: push
pkg_110c on top of 0130ead (v1.0.1) and fold in the new site; README line 30 then points to the published 12 / 24 GB paper.
Evidence: data/w24_pkg/candidate_110c/ (report.txt, runner.log, runner_v110.sh).

A gap in the runner's "another engine runs" check, found after the run and fixed in the next runners: `ps -o comm=` cuts
the command to 16 characters, so a venv Python shows as `/opt/homebrew/Ce` and never matched "python". The check now
matches the full command line. Here it changed nothing: the only other process, a no-GPU mock server of mine (0.5 GB),
ran beside the check, and the lock and Kamen-idle checks still held.

## 1.1.0 candidate pkg_110d (328a19f): answers up to 65,536 tokens (Tue 21:3x)

> (a quote from the project owner, kept private)
- **release/src (9f36cf9):** chat.py's `--max-tokens` default 16384 -> 65536 (docstring "up to 65,536 tokens per answer";
  help "default 65536; a long answer uses memory as a long prompt does"); serve.py's `--max-tokens` (the most any request
  may generate, also the cap for a request without max_tokens) 16384 -> 65536 (help: "a request's own max_tokens can ask
  for less"). No README / INSTALL / LIMITS / CONFIGS line named 16,384; run.py's own default (256) is unchanged.
- **Memory:** CAP is only the loop's stop bound in both loops (rel1 and w24: read from argv, `while len(out) < CAP`, the
  output slice, a log line, 64 during the warm-up); nothing is sized by it, so the default alone costs no memory. A long
  answer grows the KV as a long prompt does.
- **Rebuild:** `make_release.py rel1 --commit 016cb7f --extra-snap w24 --refs release/refs_rel1_w24.json --es-commit
  98c6072d --out release/out/pkg_110d` -> 328a19f; leaks 0, unsafe 0, live paths 0, syntax 0. Against pkg_110c: chat.py,
  serve.py (the lines above) and MANIFEST.json (leap_commit + hashes) only, so pkg_110c's real-engine clean-clone PASS
  covers its code.
- **Check (CPU only; Kamen was chatting):** clean-clone prep of pkg_110d (release/cleanclone/v110d: install.sh rc 0, dry
  runs lossy-24 / chat-24 / chat-32 OK); `chat.py --help` and `serve.py --help` show "default 65536"; chat.py on the
  stand-in loop (`--_screen`) starts with "ExactStream chat: config chat-32 (C 160), up to 65536 tokens per answer",
  answers, and exits rc 0.
- **Clones:** the tool-call working copy (agent/clone) has the same change. The local clone Kamen chats from
  (release/cleanclone/final/clone) gets it from a watcher (scratchpad/agent/maxtok_final.sh: exact lines, atomic renames)
  once nothing runs from that clone; he runs `--max-tokens 65536` meanwhile.
- **The go now pushes pkg_110d** (it supersedes pkg_110c).

## 1.1.0 candidate pkg_110g (7f61f48): PF-A qualifier + BIGMOE_IDLE_MS=50 (Wed 01:5x-02:0x; for the orchestrator's review)

The orchestrator (01:5x) approved both changes for 1.1.0, to be reviewed in the morning before Kamen's 14:00 go:
- **PF-A wording** (data/pfa_tail_1111.md: PF-A differs from the default prefill when a prompt's last 1,024-token chunk is
  under ~200 tokens): run.py's --env help, INSTALL.md 44-45, chat.py's docstring and comment.
- **BIGMOE_IDLE_MS=50** as an env default in chat.py and serve.py (data/idle_ms_1111.md: idle CPU 100 % -> 1.5 %, decode
  within noise, tokens identical; BIGMOE_IDLE_MS=0 in the shell keeps the old spin).
- **release/src d37fb68; rebuild:** `make_release.py rel1 --commit 016cb7f --extra-snap w24 --refs release/refs_rel1_w24.json
  --es-commit 98c6072d --out release/out/pkg_110g` -> 7f61f48: 241 files, 6.6 MB; leaks 0, unsafe 0, live research paths
  0, syntax errors 0 (candidate_110g/build.txt).
- **Diff vs pkg_110f:** INSTALL.md, chat.py, run.py, serve.py (the lines above only) and MANIFEST.json (leap_commit, those
  four files' hashes, and one source label: convert/convert_records_p11.py "Leap 83d99bb" -> "Leap 97d84e2", same sha256).
  pkg_110f's engine/loop/__pycache__ (untracked, ignored) is the only other difference on disk.
- **Clean-clone prep** (CPU only, 02:00): release/cleanclone/v110g, install.sh rc 0 in 10 s, dry runs lossy-24 / chat-24 /
  chat-32 OK (candidate_110g/prep_report.txt).
- **Real-engine checks armed** (cc110g.sh, right after the cold 32 GB agent test): 8 more 32 GB IDLE_MS runs (B A A B B A
  A B), the clean-clone check as pkg_110c's (lossy-24 + chat-24, text prompt, chat.py + serve.py --mem 24), and the
  packaged chat.py / serve.py at idle with no variable set. Results below when run. The go would push pkg_110g instead
  of pkg_110f once the orchestrator has reviewed it.

### pkg_110g real-engine checks (Wed 03:45-03:57, cc110g.sh): all PASS; held for the orchestrator's review

Run right after the cold 32 GB agent test, in the prepped clean clone release/cleanclone/v110g (outputs:
w24_pkg/candidate_110g/checks_0345/).
1. **BIGMOE_IDLE_MS confirmation at 32 GB** (run.py chat --kind code, 3 HumanEval x 256 tokens, 8 processes, B A A B B A A
   B; A = 0, B = 50): tokens EQUAL in all 8. Run 3 (arm A, 5.6 tok/s, process wall 137 s) is invalid: fanwatch's heat
   pause froze the research jobs 03:48:21-03:49:21 (fanwatch_0347.txt; run 3 ran 03:47:05-03:49:24 and is the only run
   inside that window; its median cycle 59.8 ms is normal, the freeze is in the wall time). Without it: A 132.7 tok/s (n 3,
   sd 3.00), B 132.4 (n 4, sd 2.85), B/A 0.998, -0.30 tok/s = -0.13 SE. Pooled with 01:5x's 4 runs: A 132.0 (n 5), B
   131.3 (n 6), B/A 0.995, -0.68 tok/s = -0.43 SE. **No measurable change** (data/idle_ms_1111.md). The runner's own summary
   line (A 100.9, "B/A 1.31") includes the frozen run and is not the result.
2. **Clean-clone check (as pkg_110c's): CLEAN-CLONE PASS.** lossy-24 and chat-24: 2/2 keys prefix-equal with the shipped
   references (w13eh), 133.4 / 132.5 tok/s, lifetime peak 23.68 GB; closure 848 modules, 0 outside the clone + stdlib,
   3,302 files opened, 0 outside the allowed roots; chat.py rc 0 (71 tokens, MTP head); serve.py --mem 24: one streamed
   OpenAI + one streamed Anthropic request, stopped on SIGTERM.
3. **Packaged defaults at idle (no variable set):** serve.py --mem 24 1.6 % of one core over 60 s (was ~100 %); chat.py
   --mem 24 1.5 % after its answer.
So pkg_110g (7f61f48) is ready for the orchestrator's review; the go would push it instead of pkg_110f.

## 1.1.0 candidate pkg_110h (02b6f98): the orchestrator's review of pkg_110g, docs only (Wed 04:4x)

The orchestrator reviewed pkg_110g (04:3x): code and most docs check out; three doc changes before the go.
- **README line 32:** "A 12 GB and 24 GB paper is coming soon; ..." -> "The 24 GB and 12 GB lines are written up in [A 125B
  mixture-of-experts model in 32, 24 and 12 GB of memory on one Mac](docs/pdf/small-ram.pdf)." (the title is SMALL_RAM.md's
  heading; small-ram.pdf is the papers build's slug for SMALL_RAM, so Breakthrough's v1.1.0 site build names it so).
- **LIMITS:** a PF-A line: at 32 and 24 GB, chat.py's layer-major prefill gives the default prefill's tokens except when a
  prompt's last 1,024-token chunk is under about 200 tokens; R57_PFLM=0 turns it off.
- **RESULTS footnote (2):** "These runs are not in the RELEASE paper" -> "These runs are in the 12 GB and 24 GB paper and in
  the RELEASE paper, section 3."
- release/src 16b495d; rebuilt as pkg_110g (same command, --out release/out/pkg_110h) -> 02b6f98: 241 files, 6.6 MB;
  leaks 0, unsafe 0, live research paths 0, syntax errors 0 (candidate_110h/build.txt). **Diff vs pkg_110g:** README.md,
  LIMITS.md, RESULTS.md, MANIFEST.json only (leap_commit, those three hashes, one source label; candidate_110h/
  diff_vs_110g.txt). Independent scan of pkg_110h: private patterns 0 files, secret patterns 0, /Users/ in the docs 0,
  "laptop" 0. Docs only, so no new clean-clone run (pkg_110g's checks stand).
- **push_pkg.sh docs (40c6668):** the old guard refused any small-ram file in the site (SMALL_RAM withheld until Wed),
  which would block the 1.1.0 docs push; now small-ram files are refused only while ~/Papers/public/src/publish_set.txt
  withholds SMALL_RAM, and every docs/pdf/*.pdf the package README links must be in the site (a missing README stops).
  Tested on temporary folders (5 cases). Today it STOPs: the site has no small-ram.pdf until Breakthrough's v1.1.0 build.

## 1.1.0 candidate pkg_110i (db2b7f0): the 24 GB memory label in lossy-24 / chat-24.env (Wed 04:5x, comment only)

The orchestrator's full number check (04:5x): configs/lossy-24.env and chat-24.env line 6 still said "under 24 GB on
prompts up to about 560 tokens, up to 25.1 GB on longer ones". Now (release/src 754ce56): "under 24 GB on prompts up to
about 500 tokens, more on longer ones (28.2 GB at 30K; the gate: ..." with the gate data unchanged on line 7. Comment lines
only: the env lines are identical (parsed: equal to pkg_110h's, 4 keys each; run.py's loader skips '#' lines), and no
code reads the comment; no other "560 tokens" / "25.1 GB" left in release/src. Rebuilt (same command, --out
release/out/pkg_110i) -> db2b7f0: 241 files, 6.6 MB; leaks 0, unsafe 0, live research paths 0, syntax errors 0.
**Diff vs pkg_110h:** configs/lossy-24.env, configs/chat-24.env, MANIFEST.json only (leap_commit, the two hashes, one
source label). Independent scan: private patterns 0, secrets 0, /Users/ in docs + configs 0, "laptop" 0.

## 1.1.0 candidate pkg_110j (717df58): the stale W24e clause in lossy-24 / chat-24.env (Wed 04:5x, comment only; final)

The orchestrator (04:5x): line 7's "Unless BigMoE's W24e changes R57_PFCHUNK below." is stale (W24e is done and kept 1024).
Now (release/src 33f181f): "R57_PFCHUNK below stays 1024: smaller prompt chunks change the tokens (W24e)." (wrapped onto a
new comment line 8; R57_PFCHUNK=1024 is set in both files). Comment lines only: the parsed env equals pkg_110g's in both
files. Rebuilt (--out release/out/pkg_110j) -> 717df58: 241 files, 6.6 MB; leaks 0, unsafe 0, live research paths 0,
syntax errors 0. **Diff vs pkg_110i:** the two configs + MANIFEST only; **vs pkg_110g** (the GPU-checked candidate):
README.md, LIMITS.md, RESULTS.md, configs/lossy-24.env, configs/chat-24.env, MANIFEST.json (candidate_110j/). Scan:
private patterns 0, secrets 0, /Users/ 0, "laptop" 0. The orchestrator: the final candidate unless the papers' fixes need
something. Go: PKG_DIR=release/out/pkg_110j for the pkg and the docs push (docs waits for the v1.1.0 site with
small-ram.pdf).

## 1.1.0 candidate pkg_110k (932591e): memory at 64K and 128K tokens (Wed 07:1x, docs only; final)

The long-context runs (data/long_ctx_1111.md) measured memory past the docs' 30K figures. The orchestrator (07:1x) asked for
a docs-only rebuild with two additions (release/src 4ea3ff6):
- LIMITS.md, a new line after the 24 GB line: "Longer prompts take more. At 24 GB, 32.0 GB at 64K tokens and 39.2 GB at
  128K with the default prefill, and 37.9 and 51.6 GB with `chat.py`'s layer-major prefill, which is 2.4x faster at 64K
  and about 2x at 128K. At 32 GB, `chat.py` reaches 46.3 GB at 64K; at 45 GB, 52.6 GB."
- INSTALL.md, the 36 GB note in Requirements, after the 30K sentence: "Prompts near 64K tokens need about 32 GB (38 GB
  with `chat.py`'s layer-major prefill), more than a 36 GB Mac can spare."

The wording is as approved: the orchestrator approved the LIMITS line as Leap drafted it and gave the INSTALL
sentence. The only change is backticks around chat.py, as everywhere else in these files.
Rebuilt with the same command (--out release/out/pkg_110k), giving 932591e: 241 files, 6.6 MB; leaks 0, unsafe 0, live
research paths 0, syntax errors 0.
- **Diff vs pkg_110j:** INSTALL.md, LIMITS.md and MANIFEST.json only. MANIFEST changes the two doc hashes, leap_commit
  and one source label.
- **Diff vs pkg_110g** (the GPU-checked candidate): README.md, LIMITS.md, RESULTS.md, INSTALL.md, configs/lossy-24.env,
  configs/chat-24.env and MANIFEST.json, all docs or comments (candidate_110k/).
- **Independent scan of all 243 files:** private patterns 0, account folder names 0, secrets 0, /Users/ in docs and
  configs 0, "laptop" 0.

For the go: PKG_DIR=release/out/pkg_110k for the pkg and the docs push. The docs push waits for the v1.1.0 site with
small-ram.pdf.

## 1.1.0 candidate pkg_110l (fd1e02e): the two qualifiers on LIMITS' 24 GB line (Wed 12:1x, text only; final for the go)

**The change.** The orchestrator's number check found that pkg_110k's LIMITS.md line 22 lacked two qualifiers the papers
carry (~/Papers/RELEASE.md lines 93-98):
- (a) the 500-token figure is a measurement with little room, not a bound: 448 tokens peaked at 23.954 GB, 520 at
  24.01 GB;
- (b) the 25.0 GB at 3K tokens was measured on the simulated 36 GB Mac, and the 7K / 10K / 30K figures were not.

The new wording is in release/src a31794a. It also keeps the papers' "default prefill" for the 7K-30K figures, matching
the 64K / 128K line under it.

**The rebuild** used the same command (--out release/out/pkg_110l) and gave fd1e02e: 241 files, 6.6 MB; leaks 0, unsafe
0, live research paths 0, syntax errors 0.
- **Diff vs pkg_110k:** LIMITS.md and MANIFEST.json only. In MANIFEST, the LIMITS hash, leap_commit and one source label
  changed (candidate_110l/diff_vs_110k.txt).
- **Scan of all 243 files:** private patterns 0, account names 0, secrets 0, /Users/ in docs and configs 0, "laptop" 0.

**For the go:** PKG_DIR=release/out/pkg_110l, for both the pkg and the docs push.

**Push dry run (Wed 14:3x, DRY=1, PKG_DIR=release/out/pkg_110l).**
- **pkg:** fetched the public repo's main (0130ead, v1.0.1). It would push 67ec60c, the same tree on top of it (17 files
  changed, among them chat.py, serve.py, configs/chat-24.env and lossy-24.env, r57_screen_w24.py and run.py), and nothing
  was pushed. As designed, the package repo's local HEAD now sits on top of the remote main.
- **docs:** stops with "local HEAD 67ec60c is not the remote main 0130ead: push pkg first", the intended order. It can only
  be checked fully after the pkg push.
- **At Kamen's go:** `PKG_DIR=release/out/pkg_110l PUBLIC_OK=1 release/push_pkg.sh pkg`, then the same with `docs` once
  Breakthrough's v1.1.0 site with small-ram.pdf is in ~/Papers/public/site. The repo is public, so PUBLIC_OK=1 is only for
  this owner-requested change.

## 1.1.0 PUSHED (Wed 11 Nov 16:51): eb72552 on exactstream main, one commit with the package and docs; the release drafted

**Go and site.**
> (a quote from the project owner, kept private)
  His decision on the transcript snapshots: "leave the two snapshots" (no history purge).
- Breakthrough's v1.1.0 site was built 16:49, PASS. All 15 files match Breakthrough's sha256 list, both in
  ~/Papers/public/site/ and in the docs/ copied into the commit. publish_set lists SMALL_RAM, and all 7 PDFs the README
  links are present.

**Steps.**
1. `DRY=1 push_pkg.sh pkg`: the package commit 67ec60c on top of 0130ead, as at 14:3x.
2. `DRY=1 AMEND=1 push_pkg.sh docs`: docs/ folded into it, giving eb72552. The docs checks for paths and private names in
   the HTML and PDFs passed.
3. `PUBLIC_OK=1 push_pkg.sh pkg`: "PUSHED eb72552 to main (fast-forward)", and remote main == local HEAD. Pushing with
   pkg mode leaves an already-folded HEAD unchanged; a second AMEND=1 docs run would have put "; docs: ..." in the commit
   subject twice.

**Diff vs 0130ead (v1.0.1): 29 files, +3,359 / -186.**
- Package, 17 files: CITATION.cff, CONFIGS.md, INSTALL.md, LIMITS.md, MANIFEST.json, README.md, RESULTS.md, chat.py,
  configs/chat-24.env, configs/lossy-24.env, engine/bigmoe/stream.py, engine/loop/r57_screen_w24.py, run.py, serve.py,
  tools/clean_clone_test.sh, and verify/chat-24_w13eh.json and lossy-24_w13eh.json.
- docs/, 12 files: complete.html, index.html, kernels.html, release.html, the new small-ram.html, 6 updated PDFs and the
  new pdf/small-ram.pdf. main.html, drafter.html and evaluation.html are unchanged from v1.0.1.

**Release.** A draft (gh release create --draft), "ExactStream 1.1.0", tag v1.1.0 (created when published), target
eb72552, 0 assets (the drafter files stay on v1.0.0). Its body equals release/staged/v110_notes.md. URL (draft):
https://github.com/kamenlevi/exactstream/releases/tag/untagged-b046589ed7cb86bf3e68. The orchestrator publishes it.
