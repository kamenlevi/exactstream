> (a quote from the project owner, kept private)

Candidate: `release/out/pkg_120` at **58aad586** (parent e52af19 = the public main, 1.1.0 + the docs-only commit). Not
pushed. It ships only after the orchestrator's check, EvalSuite's agent test (arm r120) and Kamen's go.

## What changed against 1.1.0 (pkg120_vs_110.stat / .diff: 31 files, +4,148 / -227)

- **Engine.** Every 1.1.0 engine file is byte-identical (MANIFEST sha256 compared file by file). Added:
  - `engine/loop/r57_screen_w24p2.py`: BigMoE 5f53556's snapshot w24p2, the P2 gate's loop byte for byte (sha256 0e2cc385 =
    data/r57/snap/w24ft/r57_screen_w24ft.py). In the package, its diff from the packaged w24 loop equals BigMoE's w24 -> w24p2
    diff exactly (the path rewrites and the one comment fix are the same as w24's).
  - `engine/loop/convcache.py`: BigMoE 3c068d7 conv/convcache.py (sha256 55631cb8, the T1e / T1-45 copy), unchanged.
- **Configs.** chat-24 and lossy-24: SNAPSHOT=w24p2 and R57_P2=256,R57_P2_ROWS=13,R57_P2_VERIFY=0 before R57_QOM_SETS; the env
  line equals BigMoE data/stack/card_w24p2_env.txt line 2 (blob 6ff0296f) exactly (asserted). BigMoE (18:4x) agreed to P2 in
  both. exact-32 / exact-45: the defensive MTPLX_QSA_PREFILL=0 pin (a no-op here; data/prefill15_1112.md section 0).
- **Server and tools.** serve.py = release/next's (tool calls, the conversation cache, the status line) with three changes:
  the docstring (it still said "text only"); the cache OFF by default at --mem 24 (unchecked there, and BigMoE's caveat: P2's
  short prompts bypass it, leaving CC.last() stale; CONV_CACHE=1 turns it on); CONV_CACHE_GB at 45 GB 6 -> 2.5, the setting
  T1-45 and the 45c run used (no run used 6). agentio.py, tools/claude_local.sh, claude_code.sh, statusline.sh,
  codex_local.sh and codex_code.sh as in release/next. The Codex scripts are shipped, labelled smoke-tested only (Tue
  10 Nov: 3 tasks at 24 GB, 2 pass + 1 partial; their requests replayed through the cache in T1 / T1-45).
- **setup.sh** (new): checks the Mac (arm64, macOS, memory 64 / 36 GB, Python 3.12, CMake >= 3.27, Xcode CLT, free disk for
  what is left incl. a 25 GB floor), install.sh first (venv + extension before any download), then the three public
  pieces, the conversion (convert/convert_store.sh) and install.sh --store --weights. Resumable; --dry-run, --verify,
  --no-download, --model / --drafters for pieces elsewhere; prints the Qwen licence lines. Pins in setup/files.tsv:
  - the MTPLX pack at Hugging Face revision 4285a4a0 (main, last modified 2026-09-22), 35 files = the repo minus
    model-vision.safetensors (the vision tower), 105,438,915,020 bytes. From one anonymous metadata call to the public HF
    API (hf_tree_*.json, hf_revision_*.json here; no model bytes). All 35 sizes equal our local copy
    (BigMoE/models/bare); 18 checksums spot-checked against it: the 15 small files' git blob ids, mtp.safetensors'
    sha256 and both drafter files' sha256 (GitHub's asset digests): 18 / 18 match. The 32 GB ngram table stays in: MTPLX's
    model code reads it (models/qwen4_exp.py, ple_row_gather.py);
  - DFlash2 at a22a1744 (config.json 1,236 + model.safetensors 1,206,140,008 bytes, in the HF cache);
  - the two drafter files from release v1.0.0 with GitHub's sha256 digests.
  - A fresh setup needs 205.8 GB free (105.4 + 68.5 store + 1.2 + 1.4 + 1 venv + 4.4 conversion work + 25 floor).
- **Docs.** README (minimal, no tables; setup.sh first; a "Coding agents" section), INSTALL (one-command install first, the
  manual steps kept, a Claude Code / Codex section with the window defaults, the status line and the 36 GB Mac settings),
  LIMITS (P2's label and scope, the Strata comparison with P2, agents tested at 32 / 45 GB only, the cache's peaks),
  RESULTS (a P2 row, the agent-test table, the cache's exactness), CONFIGS (the 24 GB row, the pin), WEIGHTS (the revision
  pin), CITATION 1.2.0, verify/README. Release notes: release/staged/v120_notes.md.
  - P2's RESULTS row is pooled here from W24T's two P2 runs files (w4p1_100 + w4p2_100): 31.4 / 24.4 / 1.12 / 23.00
    (the peak as W24T's report gives it; the runs files' max is 22.97); control 30.8 / 17.9 / 3.11 / 23.70.
- **Build.** make_release.py: the extra-snapshot path rewrite now runs longest name first (w24 is a prefix of w24p2), the
  w24 row-cap comment fix also applies to w24p2, and convcache.py comes from BigMoE git 3c068d7. refs_120.json: the 24 GB
  references = BigMoE data/r57/w4p1_100_tokens.json (P2's gated arm, W24T; w4p2 = w4p1 on 12 / 12). Build: 251 files, leaks 0,
  unsafe quotes 0, live paths 0, syntax errors 0. push_pkg.sh with DRY=1 put it on the remote main (no push); commit message
  amended to the 1.2.0 one.

## Checks done (CPU)

- **Clean-clone prep: PASS** (cleanclone_prep_58aad58.out): clone of 58aad58, install.sh --offline rc 0, `@ROOT@` left in
  nothing, run.py --dry-run OK for all 8 configs; the 24 GB dry runs show snapshot w24p2 and the three P2 keys, the exact ones
  MTPLX_QSA_PREFILL=0, chat-45 neither.
- **setup.sh** (setup_test_v120s.out, setup_dryrun_58aad58.out), on fresh clones against the local copies:
  1. --dry-run: every piece done but the venv; disk 217 GB free, 26 GB needed; rc 0;
  2. a real run with --no-download --offline: install.sh (venv + extension), no download, no conversion, install.sh with the
     links; rc 0; engine/models/bare and weights/ linked; run.py --dry-run chat-24 then picks w24p2;
  3. a rerun: every step "done"; rc 0;
  4. one drafter file missing: the dry run lists it, --no-download stops with the reason; an unknown option exits 2;
  5. on the final clean clone (58aad58): --dry-run rc 0, all done.
  Not run: an actual download (Kamen's rule: downloads go through the orchestrator; the local copies equal the pinned files).
- **Agent path vs the 32c2 pin 80c8041** (agentpath_vs_80c8041.diff): agentio.py, engine/loop/convcache.py and
  tools/codex_local.sh identical; serve.py, claude_local.sh, claude_code.sh, codex_code.sh changed; statusline.sh new.

## Armed (GPU; they start by themselves, one at a time)

1. **rel120_check.sh** (pid 32385; gates as prof_pipe2 v2, NO-SWAP >= 28, last start 23:00):
   - stage 1 = clean_clone_test.sh run on chat-24 + lossy-24 (2 keys each vs the w24p2 references, the closure), a text
     prompt on chat-24, chat.py and serve.py at 24 GB; then the **P2 smoke**: chat-24 on all 12 reference prompts at cap 256
     vs the gated arm; then, information only, serve.py --mem 24 with CONV_CACHE=1 over four turns (BigMoE's suggested
     two-turn check);
   - stages 2 (NO-SWAP >= 36: the 32 GB configs, exact-32 with the pin) and 3 (>= 52: the 45 GB configs) only if there is
     room.
2. **r120** (pid 32724): EvalSuite's release gate (PREREG_agentcc amendment 5), runner agentcc_r120.sh = agentcc_cache4.sh +
   the candidate as shipped + Claude Code 2.1.322 (agentcc_r120_vs_cache4.diff). After pid 32385, GO, NO-SWAP >= 40; last
   start 23:30; watchdog_r120.sh TERM at 00:15. Dry-run provenance: provenance_r120_dryrun.txt (every pin check passes).
3. **r120s** (pid 32945): the 24 GB arm, information only, as shipped (no cache at 24 GB), after r120, NO-SWAP >= 33 (32c2's
   longest prompt 16,905 tokens; 24 GB serve.py peaked at 27.3 GB on 12.6K-token prompts; + margin + 3), watchdog 00:15.
   Dry-run provenance: provenance_r120s_dryrun.txt.

The x32k runner (prefill15) was stopped at 18:4x so it cannot start beside these; it is re-armed after them.

## 19:0x: the Host / Origin fix; the candidate is now pkg_120b at 21fb59b6 (58aad586 superseded)

**Why.** A background security review of e005071 flagged serve.py: it binds 127.0.0.1 but answers every request that
reaches it, so a web page in the browser could drive it. Cross-site POSTs (simple requests need no preflight) could queue
work and tie up the engine. DNS rebinding (a site whose name resolves to 127.0.0.1) could also read the answers and
/stats, and time conversation-cache hits to test guesses about a conversation's opening text. 1.1.0's serve.py had the same
gap. The orchestrator (19:0x): the fix ships in 1.2.0. EvalSuite pulled r120's GO until its amendment 5d.

**The fix** (serve.py only; no generation change; serve_py_21fb59b_vs_58aad586.diff): do_GET and do_POST first call
`_local()`. It refuses (403, a "refused a ..." line in the server log) a Host whose name is not 127.0.0.1, localhost or ::1,
and an Origin, if one is sent, whose host is not one of those ("null" and file:// included). A request with no Host is
allowed: browsers always send one. Docs: serve.py's docstring, INSTALL's serve.py paragraph, and the v1.2.0 notes line
the orchestrator asked for (1.1.0 users should update).

**Candidate pkg_120b at 21fb59b6** (parent e52af19; release/out/pkg_120 at 58aad586 is kept as the earlier record).
Against 58aad586 only serve.py (+33 lines), INSTALL.md (+2) and MANIFEST.json change (pkg120b_vs_58aad586.diff). serve.py
sha256 4c15a1d4bbc32f89067a6153b8c7dca2def588bf1492dd408e99630401cebe2a; every other agent-path file is unchanged.
Clean-clone prep: PASS (cleanclone_prep_21fb59b.out: 8 of 8 dry runs; the clone differs only by the @ROOT@ rewrite).

**Stand-in test** (hostcheck_test.sh -> hostcheck_test.out; stand-in engine, CPU only, port 8136, a scratch clone of 21fb59b):
**24 of 24 as expected.**
- Allowed (200): curl's own Host; Host 127.0.0.1:port, localhost:port, [::1]:port, bare localhost, LocalHost:port; POST with
  no Origin; Origin http://localhost:3000, http://127.0.0.1:port and http://[::1]:5173; HTTP/1.0 with no Host; Python urllib.
- Refused (403): Host evil.example:port, evil.example, 127.0.0.1.evil.example:port, 192.168.1.5:port and 0.0.0.0:port;
  a POST with a foreign Host; Origin https://evil.example, null, http://localhost.evil.example, http://127.0.0.1.nip.io:port
  and file://; /v1/responses with a foreign Origin. The server log has the 12 matching "refused" lines.
- **Claude Code 2.1.322** through tools/claude_code.sh (sandbox-exec, throwaway config, only Read allowed): rc 0, "success",
  2 requests answered, 0 refused; it did not write the file (Write not allowed).
- **Codex 0.144.1** through tools/codex_code.sh (its workspace-write sandbox, no network): rc 0, 2 requests answered,
  0 refused; its command wrote hello.txt in its own folder.
- The one "Error" line in the stand-in's log is "conversation cache off: ModuleNotFoundError: No module named 'convcache'".
  The stand-in loop runs outside engine/loop, so the stand-in has no cache; this is not the real engine's path.

**Re-armed on 21fb59b** (the 58aad586 runners were stopped by their tasks; their folders were kept aside unused:
agentcc_runs/r120_armed1855, r120_armed1901_58aad58 and r120s_armed1855_dropped):
- rel120_check.sh now takes W / PKG from the environment (default v120b / pkg_120b): pid 53806, last start 01:30 (the
  orchestrator). At 19:06 the gates were already clear (NO-SWAP 37 GB).
- r120: agentcc_r120.sh, whose only change from the registered f0d86608 is the CL / PKG / PIN lines
  (agentcc_r120_5d_vs_5a.diff). New sha256 9e0fbc21c29abbbad16e23cdac4c361006cdacf200fb693900579fe32e0ec0d5. pid 55449,
  fresh folder agentcc_runs/r120b, after pid 53806, last start 01:30, watchdog 02:15. **No GO until EvalSuite's 5d.**
  Dry run: provenance_r120b_dryrun.txt (every pin check passes against 21fb59b6). r120s is dropped for tonight (5c).

## 19:07-19:10: GPU checks on 21fb59b, stage 1 (the 24 GB configs): PASS (rel120_check_21fb59b.log, cleanclone_report_21fb59b.txt)

The gates cleared at 19:07:27 (NO-SWAP 38 GB, BigMoE's lock gone); share lock "MEM 45 GB"; heat irrelevant (tokens only).
- **Clean-clone run, chat-24 + lossy-24: CLEAN-CLONE PASS.** Each matches the w24p2 reference on 2 / 2 keys; first token
  1.07 / 1.06 s, peak 22.93 / 22.90 GB; closure 848 modules and 3,302 files, 0 outside. The chat-24 text prompt: first token
  0.64 s. chat.py --mem 24: rc 0, first token 1.64 s, peak 23.05 GB. serve.py --mem 24 (the new Host / Origin check, real
  engine): the streamed OpenAI and Anthropic requests both complete; it stops on SIGTERM.
- **P2 SMOKE: chat-24 on all 12 reference prompts, cap 256: 12 / 12 keys equal to the gated arm (w4p1_100)** with the
  reference's history (p2_smoke_check.txt). Whole decode 31.8, whole request 24.6 tok/s, first token 1.14 s median
  (0.83-1.93), lifetime peak 22.89 GB (W24T: 31.4 / 24.4 / 1.12 / 23.00).
- **Information only: serve.py --mem 24 with CONV_CACHE=1 (unshipped), four turns:** HTTP 200 x 4; 0 server errors.
  - The two short turns (17 and 107 tokens, P2) carry no cache line, since P2 bypasses the cache (BigMoE's caveat).
  - The 670-token turn is a miss (the first time), and so is its 714-token follow-up. Expected: the cache stores its first
    checkpoint at 8,192 tokens (CONV_CACHE_GRID), so a short chat has nothing to restore.
  - No crash, sane lines; this does not check the cache at 24 GB.
- **Stage 2, the 32 GB configs (19:10:45-19:12:50): CLEAN-CLONE PASS.** exact-32, with the MTPLX_QSA_PREFILL=0 pin, equals
  the exact reference on 2 / 2 keys (first token 2.85 s, peak 30.39 GB). lossy-32 and chat-32 equal theirs on 2 / 2 (peaks
  31.08 / 31.06 GB). Closure: 0 modules or files outside the clone. The chat-32 text prompt runs.
- **Stage 3 (the 45 GB configs) skipped for memory: NO-SWAP 45 < 52.** One more try is chained right after r120 (STAGES=3,
  first gate NO-SWAP >= 46, the stage itself >= 52, else skipped). The 45 GB configs' engine files are byte-identical to
  1.1.0's, which passed its clean-clone run on 10 Nov; exact-45 adds only the no-op pin.

## Tonight's queue (the orchestrator 19:0x: rel120_check, then r120, then the night queue), each chained on the previous pid

rel120_check (pid 53806, done 19:12:50) -> r120 (pid 55449, GO 19:09:05 after EvalSuite's 5d 69dfd11) -> rel120_check
STAGES=3 (after.sh pid 7671; the 45 GB configs) -> x32k_exact.sh (after.sh pid 8769; NO-SWAP >= 46; last start Fri 08:00) ->
prof_pipe2.sh (after.sh on 8769) -> prof_qsa2.sh (waits for pipe2.done) ->
prof_pipe3.sh (waits for qsa2.done) -> onebuild_45.sh (after.sh on prof_pipe3's pid 68110; NEW,
scripts/prefill15/onebuild/onebuild_45.sh + hook/sitecustomize.py, as pre-registered: R0 PF-D vs R1-R3 at 10K / 30K,
NO-SWAP >= 56; head-tested; work dir scratchpad/agent/onebuild45). prof_fast4 is dropped. The prof chain and onebuild time
things only as information (fans at the minimum).

## 19:14-19:36: r120, EvalSuite's release gate (PREREG_agentcc amendments 5a-5d) on 21fb59b, run by Leap's runner

Records: agentcc_r120b/ (runner.log, provenance.txt, results.jsonl, probe_cc.*, convcache.log, serve.log, sandbox.sb +
profiles/, and tool_calls.jsonl from summarize_calls.py). The 12 transcripts and the raw request log stay local only
(scratchpad agent/agentcc_runs/r120b). EvalSuite scores it; its verdict is the gate. The runner's own counts:
- GO 19:14:38 (NO-SWAP 44 GB); serve.py --mem 32 ready in 19 s; done 19:36:21; no swap growth; no watchdog needed.
- **Probe: 29 / 30, 0 invented tools** (the line: 28 and 0).
- **Tasks: 11 / 12 checks pass** (the line: 10). t09_refactor failed its check (claude rc 0, 76.7 s). In 32c2 t09 passed and
  t11 failed; in 32c t09 failed. Task walls 30-174 s, 1,038 s in all (32c2: 875 s).
- Validity: 12 / 12 transcripts claude_code_version 2.1.322; 12 + 12 profiles kept; 0 symlinks in the run folder; 0
  lingering processes.
- Tool calls: Bash 50, Read 22, Edit 15, Write 7; 6 errors. 6 calls named a path outside the task folder (t11 x1, t12 x5),
  of which 2 Bash calls were refused by the sandbox (EPERM). For EvalSuite's screen.
- **Server (the package as shipped): 124 requests, 0 errors, 0 refused by the Host / Origin check, the cache installed
  (no "cache off" line): 123 of 124 hits** (the miss: the first request, 8,738 tokens, 20.9 s).
  - 8.1 % of prompt tokens prefilled: new tokens median 621, p90 3,281.
  - First token median 6.3 s, p90 13.0 s, max 94.5 s. The max was a copy hit at 9,216 with 3,287 new tokens (32c2's
    comparable hits: 12-21 s). Cause (EvalSuite): fanwatch's heat freeze stopped the server (pid 17850) from 19:28:56 to
    19:30:17, while t10's first request ran; an earlier freeze, 19:18:50-19:19:31, fell around t01's start. The fans were at
    the minimum tonight.
  - Decode median 67.2 tok/s; answers median 95 tokens; lifetime peak 36.50 GB (32c2: 36.17).
  - Prompts up to 19,474 tokens (median 13,062), so the 64K window (compaction near 32K) was never reached.

**EvalSuite's verdict: r120 PASS** (EvalSuite b397046, RESULT_agentcc_r120_2026-11-12.md).
- Probe 29 / 30, with the same 30 replies as 32c2. Tasks 11 / 12 (t09: a wrong edit). The counts match the runner's.
- Screen and validity: 0 flags; 12 / 12 transcripts at 2.1.322; 24 profiles; 0 links, 0 lingering processes, 0 late writes;
  0 refused requests; the cache on, 123 / 124 hits.
- The 6 outside_task calls are no task-rule issue (none names a check):
  - t12 #8 / #9 listed the run folder and were refused by the profile;
  - t12 #17-#19 were Claude Code's memory notes in its own cc_home/<id>/memory (inside its write set);
  - t11 #12 was a /tmp scratch folder, refused as in every run.
The push waits for the orchestrator's OK (the orchestrator 19:1x: then push 21fb59b and publish v1.2.0; the 45 GB clean-clone
stage runs after the release when memory allows).

> (a quote from the project owner, kept private)

- `PUBLIC_OK=1 PKG_DIR=release/out/pkg_120b bash release/push_pkg.sh pkg`: **PUSHED 21fb59b to main (fast-forward), remote
  main == local HEAD** (e52af19 -> 21fb59b6, the gated commit, unchanged; the package repo was clean).
- release/staged/v120_notes.md got the orchestrator's two body-only edits: the agent test "passed at 32 GB three times, the
  last on this exact release with Claude Code 2.1.322 (29 of 30 tool calls, 11 of 12 tasks), and once at 45 GB"; and the
  last test's 32 GB peak, 36.50 GB, a little above LIMITS.md's 36.17 GB, which gets the new number in the next package.
  Leak check on the notes: 0.
- **Draft release "ExactStream 1.2.0"**, tag v1.2.0 at 21fb59b6, notes = v120_notes.md:
  https://github.com/kamenlevi/exactstream/releases/tag/untagged-3ab11c2b359ab1fb3058. The orchestrator publishes it and
  checks the release and Pages.
- For the record (the orchestrator, no release action): the probe's decode was 66.6 vs 32c2's 70.4 tok/s on identical
  replies (-5 %). The die reached 84 C with two heat freezes, and memory headroom was 44 vs 49 GB; the engine is unchanged.

## Fri 13 Nov 02:27-02:29: stage 3, the 45 GB clean-clone check of the published v1.2.0: PASS (cleanclone_report_45gb_21fb59b.txt)

Run by scripts/night/night45_1113.sh in the orchestrator's order (01:4x): right after BigMoE's 45 GB A/B, ahead of TTT30B.
- **Lock.** The runner saw the A/B's ticket at 02:15:38 and queued one quiet ticket behind it (#1). The grant came at
  02:26:46, after 11 min.
- **The stage.** After a 30 s settle, NO-SWAP was 52 GB. clean_clone_test.sh ran in the installed clone v120b at 21fb59b
  (= main = v1.2.0):

| config | tokens vs its reference (ifeval/1220, ifeval/3280, cap 256) | first token | whole decode | lifetime peak |
|---|---|---|---|---|
| exact-45 (with the MTPLX_QSA_PREFILL=0 pin) | 2/2 prefix-equal to exact-45_ar.json (exact) | 2.63 s | 32.0 tok/s | 41.48 GB |
| lossy-45 | 2/2 prefix-equal to lossy-45_rli256.json | 2.74 s | 35.1 tok/s | 42.18 GB |
| chat-45 | 2/2 prefix-equal to chat-45_pdz4a.json | 2.39 s | 45.3 tok/s | 43.70 GB |
| chat-45, a text prompt | (runs; 87 tokens) | 1.87 s | 42.9 tok/s | 43.69 GB |

- **Closure.** Every config loads modules only from the clone and the standard library, and opens files only inside the
  allowed roots.
- **Verdict: CLEAN-CLONE PASS at 02:29:15.** With 11-12's stage 1 (24 GB) and stage 2 (32 GB), the published 1.2.0 now
  passes from a clean clone at all three sizes. No 1.2.1 is needed for this.
- **Onebuild did not start.** It needs NO-SWAP 56; there was 50 GB 20 s after the stage. The ticket was released at
  02:29:35, about 3 min after the grant. Onebuild goes to the next night, as the orchestrator set. Logs: night45_1113.log,
  stage3_45gb_1113.out.
