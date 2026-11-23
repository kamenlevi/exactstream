> (a quote from the project owner, kept private)

Candidate: `release/out/pkg_121` at **b830caf** (parent 21fb59b = the public main, v1.2.0). Staged with
`DRY=1 push_pkg.sh pkg` (nothing pushed; docs/ kept from the remote), message amended to the 1.2.1 one. Nothing goes
public until the orchestrator brings Kamen's word on the finished, checked package.

## Scope (the orchestrator 15:2x)
P2 in the 45 GB `chat` config only. Not `exact` (P2 is lossy), not `lossy`-45 (L + COH, not gated). chat-45 runs the gated
loop. The gate: EvalSuite f25a578, docs/RESULT_p2gate45_2026-11-13.md, PREREG_p2gate45 + amendments 1, 1a, 1b: IFBench
+2.00 pt (p 0.68), GPQA Diamond's 67 prompts of <= 256 tokens -1.49 pt (p 1.00), 5 longer-prompt controls token-identical.
Arms: BigMoE data/stack/chat45_env.txt line 2 (blob 40eee33e) vs chat45_p2_env.txt line 2 (blob e4012e80), differing by
exactly R57_P2=256, R57_P2_ROWS=16, R57_P2_VERIFY=0; both on BigMoE's P3 snapshot c477b60 (screen sha256 e410133cfe69).

## How it is built (the orchestrator 15:3x: BigMoE's c45p2, pflm.py stays rel1's)
`make_release.py rel1 --commit 3bf3dbd --extra-snap w24 --extra-snap w24p2 --extra-snap c45p2 --refs release/refs_121.json
--es-commit 98c6072d --out release/out/pkg_121` (BigMoE 3bf3dbd = "Snapshot c45p2 for package 1.2.1"). Checked here:
- c45p2's loop = the gate's loop (c477b600 data/r57/snap/p3/r57_screen_p3.py, sha256 e410133cfe69) with its 19
  `frozen_p3` paths renamed `frozen_rel1`, nothing else (diff after the rename: empty).
- c45p2's siblings (inert_span, pflm, relax_delta, relax_span) = rel1's at 5f53556, blob for blob. So pflm.py is 1.2.0's;
  the gate ran P3's copy (adds R57_PFD_FOLD, unset in both gate arms, and refactors the read-ahead around it).
- bigmoe/ (tree b21c3563 at 5f53556, c477b60 and 3bf3dbd), ext/, rel1, frozen_rel1, w24, w24p2 and the by-path files:
  unchanged 5f53556 -> 3bf3dbd.
- P3's loop with P2 off reproduces 1.2.0's chat-45 reference: BigMoE p3c1 tokens = pdz4a on 12 / 12 prompts.

Against pkg_120b (MANIFEST sha256, file by file): changed CITATION.cff (version 1.2.1), CONFIGS.md, INSTALL.md, LIMITS.md,
RESULTS.md, configs/chat-45.env, serve.py, verify/README.md; added engine/loop/r57_screen_c45p2.py and
verify/chat-45_p2.json; removed verify/chat-45_pdz4a.json. Every other file is byte-identical (67 carry the new BigMoE
commit in their source label only). Build: 252 files, leaks 0, unsafe quotes 0, live paths 0, syntax errors 0.

- **configs/chat-45.env:** SNAPSHOT=c45p2, C=256, and BigMoE chat45_p2_env.txt line 2 verbatim (asserted equal to blob
  e4012e80). Against 1.2.0's line: + R57_P2=256,R57_P2_ROWS=16,R57_P2_VERIFY=0; R57_QOM_SETS gains lcb6, ifbench, gpqad
  (the gate's own QOM sets; a typed prompt's set, multi / humaneval / math, is handled as before); key order differs.
- **verify/chat-45_p2.json** = BigMoE data/r57/p2p1_256_tokens.json (sha256 52c3738d18c0): the P2 arm on the 12 N45
  IFEval prompts at CAP 256 (A/B bb61289). p2p2 gave the same tokens on 12 / 12. The A/B's p2p1 env = the gated P2 line
  except R57_P2_ROWS (16 = the default) and QOM sets humaneval,mbpp,ifeval,multi (the c45 runner's): ifeval is a QOM set in
  both, so the 12 prompts run the same.
- **serve.py:** with the cache on, a P2 prompt (<= R57_P2 tokens) reported the PREVIOUS request's cache lookup
  (convcache.last() is set only by the cached prefill, which P2 bypasses): the done line, its stats and the status line's
  "cached". Now such a prompt reports no cache use. Statistics only. Tokens: P2 builds its own fresh cache and, with the
  cache on, every request drafts on DFlash2 (no MTPHead), so no stored entry is touched (convcache.py read 13 Nov).
- **Docs:** LIMITS (P2 at 45 GB: label, gate, speed, Strata on the same prompts; the pflm.py note and tonight's two token
  checks; the gate's 45.42 GB at 258-269 tokens; 36.17-36.50 GB at 32 GB; the 45 GB clean-clone peaks 41.48 / 42.18 /
  43.70 GB), RESULTS (a `chat` 45-with-P2 row + note 4), CONFIGS (the chat row, SNAPSHOT=c45p2), INSTALL (P2 at 45 GB,
  the reference), verify/README, CITATION 1.2.1. Release notes: release/staged/v121_notes.md (one sentence,
  RELEASE_CHECKS_SENTENCE, is filled after tonight's checks).
- **Numbers quoted** (same pooling as RESULTS' existing rows; it reproduces 1.2.0's chat-45 row 50.7 / 28.8 / 2.26 / 43.69
  from pdz4a exactly): P2, BigMoE p2p1 + p2p2 (24 rows): whole decode 50.7, whole request 43.2 tok/s, first token median
  0.47 s, peak 42.36 GB; the A/B's control p3c1 + p3c2: 50.2 / 27.6 / 2.44 / 43.70. Strata on the same 12 prompts
  (~/StrataEval/h2h/results_arena33, its 33 GB arena): first token 0.83 s (median), whole decode 40.1-41.4, end to end
  29.2-29.7 tok/s (its REPORT.md, arms B / D).

## Checks done (CPU)
- **Clean-clone prep: PASS** (release/cleanclone/v121_prep.out): clone of b830caf, install.sh --offline rc 0 (19 s),
  run.py --dry-run OK for all 8 configs. Dry runs: chat-45 = snapshot c45p2 + R57_P2 256 / 16 / 0 + the widened QOM sets,
  no R57_PFD_FOLD; chat-32, lossy-45, exact-45 = rel1, no P2; chat-24 / lossy-24 = w24p2 + P2 256 / 13 / 0 (as 1.2.0).

## Not used: the own-folder variant (test build, Fri 15:2x-15:3x)
Before BigMoE's handoff, a `--sub-snap NAME@COMMIT` option for make_release.py put P3's loop WITH its own siblings in
engine/loop/p3/ (P3's pflm.py; no other config's file changes), asserting that the 40 frozen modules the package places
equal frozen_p3's after the path rewrite and that bigmoe/ + ext/ match (diffs: make_release_subsnap_variant.diff,
launchers_subsnap_variant.diff: run.py screen_path(), chat.py, serve.py's convcache import by path, needed because a loop in
a subfolder does not put engine/loop on sys.path; without it the 45 GB cache would have switched itself off). Its packaged
loop file equals c45p2's byte for byte (sha256 90da118e66ce, two independent routes). The orchestrator chose c45p2 with
rel1's pflm.py plus two token checks instead; the variant is not in the package.

## Tonight (after onebuild, under its own quiet ticket; the orchestrator 15:2x / 15:3x)
1. Clean-clone stages at 24, 32 and 45 GB on b830caf.
2. chat-45's first token on the 12 N45 IFEval prompts (the A/B: 0.47 s median), tokens vs verify/chat-45_p2.json.
3. Token check 1 (closes the pflm.py gap): one prompt well over 256 tokens (P2 off), packaged 1.2.1 chat-45 vs 1.2.0
   chat-45 (release/cleanclone/v120b/clone): tokens bit-identical (+ logits hashes, R57_LOGHASH=1).
4. Token check 2: the first 5 keys of the gate's P2 arm (card_p2_45_fwd, process 1), in the gate's order, with the
   packaged chat-45: tokens equal the gate's rows.
5. Long-prompt peaks of chat-45 at 45 GB (LIMITS.md gets the answer).
6. (added here) serve.py at 45 GB with the cache on: a long turn, a short P2 request, the long turn's follow-up, with
   CONV_CACHE_CHECK=1 (each hit compared with a cold prefill) and the P2 request's statistics.
If check 3 or 4 fails there is no release; the orchestrator is told.

## Tonight (Sat 14 Nov), as it runs
- **00:30-00:37, the order changed:** onebuild released its grant without a run (NO-SWAP 47 < 52; data/prefill15_1112.md
  section 13). The first checks runner (rel121_check.sh, v1) skipped A and B at 47 < 48. It was stopped (TERM 00:36:43)
  and re-armed as rel121_check_v2.sh: the same parts, gates and cutoff, with G (32 GB) and H (24 GB) first. v2 has held
  its own quiet ticket since 00:37:03.
  - After G, NO-SWAP rose 47 -> 50: the first big runs pushed idle pages into the compressor.
  - rel121_check_v3.sh (A, B and C gates at 47, from 1.2.1's own P2 peaks + 4) is prepared but not to be used (the
    orchestrator 00:4x: keep the gates tonight; a part that skips goes to the morning). v2's A started at
    NO-SWAP 49.
- **G, stage 2 (32 GB): CLEAN-CLONE PASS (00:37:33-00:40:23).**
  - exact-32: 2/2 keys prefix-equal.
  - lossy-32 and chat-32: 2/2 with the reference's history.
  - Closures: 0 modules or files outside the clone. A text prompt on chat-32 ran, chat.py rc 0, and serve.py's OpenAI and
    Anthropic streams completed and it stopped on SIGTERM.
  - Lifetime peaks: 30.39 / 31.07 / 31.08 GB; text 31.07; chat.py 31.27.
- **H, stage 1 (24 GB): CLEAN-CLONE PASS (00:40:24-00:41:50).**
  - lossy-24 and chat-24: 2/2 with history; closures clean. Text, chat.py and serve.py OK.
  - Peaks 22.90 GB (chat.py 23.05).
  - First token 1.05 s on the 2 IFEval keys; 0.63 s on the text prompt (P2).

## Results, Sat 14 Nov 00:37-00:54 (rel121_check_v2.sh, one quiet ticket; rel121_results_1114.txt, rel121_check_1114.log)
**TOKEN CHECK 1 FAILED, so there is no release (the orchestrator's rule, 15:3x). The orchestrator was told at 00:5x.** Nothing
was pushed.

| part | result |
|---|---|
| G stage 2, 32 GB | CLEAN-CLONE PASS (above) |
| H stage 1, 24 GB | CLEAN-CLONE PASS (above) |
| A stage 3, 45 GB | CLEAN-CLONE PASS. exact-45: 2/2 prefix-equal (peak 41.50 GB). lossy-45: 2/2 with history (42.20). chat-45: 2/2 vs verify/chat-45_p2.json with history, first token 0.52 s, whole decode 52.0 tok/s (42.35). Text on chat-45: 0.31 s (42.33). chat.py rc 0 (42.50). serve.py: both streams completed, stopped on SIGTERM. Closures clean. |
| B P2 smoke, 45 GB | PASS: 12/12 keys prefix-equal with history; first token median 0.48 s (0.22-0.65), P2 on 12/12; lifetime peak 42.34 GB (the A/B: 0.47 s, 42.36) |
| C token check 2 | PASS: the gate's process 1, first 5 keys (gpqad, cap 2048, warm-up on, its dec_gpqad.jsonl md5 5af744d9), tokens equal the gate's rows on 5/5 |
| D token check 1 | **FAIL**: 1,972-token prompt, cap 256, R57_LOGHASH=1, 1.2.1 chat-45 vs 1.2.0 chat-45 (v120b clone): tokens DIFFER from token 1 (token 0 = 1919 in both; then 9665 vs 5392). The first logits hash already differs (102 vs 109 hashes). Peaks 46.60 / 46.78 GB. |
| E serve.py 45 + cache | PASS: 5 requests, HTTP 200 x5; prompt tokens 1972 / 17 / 2062 / 106 / 2116; both cache hits' cold-prefill checks PASS; the 2 short (P2) requests show no cache statistics; 0 errors. Peaks 46.83 -> 47.21 GB. |
| F long-prompt peaks, chat-45 | 289 tokens 44.89 GB (first token 4.75 s); 965: 46.07 (6.97 s); 3,999: 46.77 (26.85 s); 1,972 from D: 46.60 |
| F 8,953 tokens, I 28,966 | SKIPPED: NO-SWAP 43 / 44 < 52 / 53. Another session's training resumed at 00:51 while this ticket was still held (between parts), taking ~9 GB. Per the orchestrator 00:4x, skips go to the morning. |

**Check 1, what is known (check1_1114/: both runs' runs / tokens / spec / modules / log):**
- Both are fresh processes. Each warms up on gsm8k/708 (short, 3 drafters, discarded) before the prompt.
  - In 1.2.1 that warm-up takes P2; in 1.2.0, the PF-D prefill. MLX prefill peak 40.7 vs 42.2 GB; lifetime 42.4 vs 43.7.
  - So the long prompt starts from different expert residency: PF-D's A/B/C counts are 6562 / 12099 / 16078 vs 4883 /
    9950 / 19906, and its SSD bytes 53.5 vs 62.6 GB.
  - checked / bad_A / bad_B / fallback are 0 in both.
- Two explanations remain, and the data so far do not separate them:
  1. the history (P2's warm-up) changes the prefill's or the decode's residency-dependent paths;
  2. c45p2's loop with rel1's pflm.py really differs from 1.2.0 on the long-prompt path. Closing that gap was the
     check's purpose.
- Proposed diagnosis (asked 00:5x), same prompt, cap 256, LOGHASH:
  1. 1.2.0 again (is it reproducible run to run?);
  2. 1.2.1 with R57_P2=0 (the same warm-up path as 1.2.0);
  3. 1.2.1 again.

**LIMITS answer (long prompts at 45 GB): chat-45 goes over its 45 GB line on every prompt over 256 tokens measured
tonight.** It peaked at 44.89 GB at 289 tokens and 46.1-46.8 GB from 965 to 3,999 tokens; serve.py with the cache reached
47.21 GB at 2.1K. P2 prompts (<= 256 tokens) stay at 42.3-42.5 GB. 10K / 30K: pending.

**Check 1, static analysis (Sat 01:0x, CPU only; check1_1114/loop_rel1_vs_c45p2.diff, the packaged files): 1.2.1's
long-prompt path is unchanged from 1.2.0's by construction.**
- c45p2's loop is rel1's loop plus only the P2 additions (52 diff lines, 45 changed): the P2MAX / P2ROWS settings and an
  assert, P2ST / P2VER and prefill_p2(), the branch `if P2MAX and len(ids) <= P2MAX: prefill_p2(...) else:
  prefill(ids, mtp, chunk=PFCHUNK)`, and the `p2` statistics field.
- A prompt over 256 tokens therefore calls the identical prefill() and decode. pflm.py is the same file in both packages.
- The env's extra QOM sets (lcb6, ifbench, gpqad) only add FASTARMS["QOM"] entries for those sets. They touch neither a
  typed prompt (set "multi", in both lists) nor the warm-up (gsm8k/708, in neither).
- What remains in check 1's two processes is the warm-up: it takes P2 in 1.2.1 and the PF-D prefill in 1.2.0, so the long
  prompt starts from a different expert state (PF-D A/B/C above).
- The proposed diagnosis (scripts/night/rel121_diag1.sh, not armed) tests this directly:
  1. 1.2.1 vs 1.2.0 both --no-warmup: expected bit-identical;
  2. 1.2.1 R57_P2=0 vs 1.2.0;
  3. 1.2.0 cold vs warm: does 1.2.0's own warm-up already change the long prompt's tokens?

**Check 1 diagnosis, Sat 01:02-01:13 (the orchestrator 01:0x / 01:1x: six runs in two quiet tickets of three, each ending
within 8 min of its grant; check1_1114/diag1.log, diag2.log): ONE of six ran.**
- **1.2.0 again** (ticket 1, granted 01:02:10; run 01:02:41-01:03:22, peak 46.78 GB): tokens EQUAL and logits hashes
  EQUAL to check 1's 1.2.0 run, and identical PF-D A/B/C counts (4883 / 9950 / 19906).
  - 1.2.0 is reproducible run to run, so check 1's reference is solid and its FAIL is real.
  - The PF-D counts are deterministic for a given history.
- **Skipped for memory** (gate 51 = peak 46.78 + 4; the orchestrator: no cut margins):
  - Ticket 1's 1.2.1 R57_P2=0 and 1.2.1 again: NO-SWAP stayed at 50 after the first run; skipped at minute 6:49.
  - Ticket 2's three cold runs (1.2.1 / 1.2.0 / 1.2.1 --no-warmup; granted 01:09:01): NO-SWAP 49-50; it stopped at grant
    + 4 min (01:13:01).
  - Nothing big was running. The compressor grew 2.9 -> 5.3 GB (idle apps' pages), and the floor stayed at ~49-50 GB.
- **Morning:** the 5 runs (NO-SWAP 51, one at a time), with the same runners. Of the orchestrator's two outcomes, (b)
  "1.2.0 not reproducible" is ruled out; (a) needs the P2=0 run and 1.2.1's reproducibility.

**pflm.py, rel1's (shipped) vs P3's (the gate's), CPU code review (Sat 01:1x; check1_1114/pflm_rel1_vs_p3.diff,
from BigMoE 3bf3dbd data/r57/snap/rel1/pflm.py and c477b60 data/r57/snap/p3/pflm.py, read-only): with R57_PFD_FOLD
unset, as in both gate arms and in 1.2.1, the two files run the same operations in the same order.** The diff (90 lines):
- `FOLD = AHEAD and R57_PFD_FOLD == "1"` (off).
- stage_reads() keeps its zeros + eval and hands the identical view / pread submission to an extracted
  stage_submit().
- stage_many() is used only when FOLD is set.
- predict() computes the same router forecast through the extracted forecast() / forecast_tier() when no `top` is passed
  (always, with FOLD off).
- The call's folded sync, and skipping the separate assemble eval, are both behind FOLD.
- Two statistics keys ("fold", "tJ") are added, and a docstring path changes.
P3's own comment calls the fold exact ("Same reads, same values"). So the packaged chat-45 computes what the gated P2
arm computed, and token check 2 (5/5 = the gate's rows) agrees. **The pflm.py gap is closed by code review. Check 1's
difference is not from pflm.py.** It comes from the warm-up history, which the gated configuration has too. The morning
runs (1.2.1 R57_P2=0, the cold pair, 1.2.1's reproducibility) are to confirm that.

**Check 1 diagnosis completed, Sat 01:16-01:20 (the orchestrator 01:2x: the 5 remaining runs at gate 50 = peak 46.78 +
3.2, tokens / hashes only; one quiet ticket, granted 01:16:55, runs 01:17:25-01:20:13, peaks 46.60-46.77 GB, swap flat;
check1_1114/diag3.log + each run's runs / tokens): CHECK 1'S DIFFERENCE IS P2'S WARM-UP HISTORY. The orchestrator's
outcome (a) holds.**

| run (1,972 tokens, cap 256) | PF-D A / B / C | first tokens | same tokens and logits hashes as |
|---|---|---|---|
| 1.2.0 warm: check 1, and again | 4883 / 9950 / 19906 | "This paper details ..." | each other; **1.2.1 R57_P2=0** |
| 1.2.1 R57_P2=0, warm | 4883 / 9950 / 19906 | "This paper details ..." | **1.2.0 warm** |
| 1.2.1 warm: check 1, and again | 6562 / 12099 / 16078 | "This draft details ..." | each other |
| cold (--no-warmup): 1.2.1, 1.2.0, 1.2.1 again | 0 / 0 / 34739 | "This draft paper details ..." | **all three each other** |

- **Both packages are reproducible.** 1.2.0 warm x2 and 1.2.1 warm x2 match bit for bit; the cold runs x3 match each other.
- **Matched history means identical output.** With 1.2.0's warm-up path (R57_P2=0), 1.2.1 is bit-identical to 1.2.0:
  tokens, every check's logits hash, and the PF-D counts. With no history (cold), 1.2.1 = 1.2.0 bit for bit.
  - So the long-prompt path is unchanged, and the matched-history check 1 passes both ways.
  - This agrees with the static analysis: c45p2's loop is rel1's + P2 only, and pflm.py is closed by code review.
- **History dependence is chat-45's, not new in 1.2.1.** 1.2.0's own output on this prompt changes with its warm-up:
  cold vs warm differ at token 1.
  - Mechanism, from the code: chat-45 decodes with FAST=QOM (r57_screen_c45p2.py ~604-606). A sure row's experts outside
    ranks 0-1 and the check's union become optional uses, which the loader serves only when resident.
  - So output depends on what earlier prompts left in memory. The PF-D counts follow that state.
  - In 1.2.1, the warm-up (and any short prompt earlier in a process) goes through P2 and leaves a different state.
- **The splits are near-tie word choices.** Token 1 is " paper" (5392) or " draft" (9665); cold takes both, " draft
  paper". The 256-token summaries then drift into differently worded summaries of the same draft: 75 / 126 / 67 of 256
  tokens fall in common runs (1.2.0 warm vs 1.2.1 warm / vs cold; 1.2.1 warm vs cold).
  - The outputs keep only per-check logits hashes, not top-2 logits, so the margin at token 1 is not measured. The
    draft / paper / draft paper pattern shows a near-tie.
- **What is left for Kamen:** check 1 as specified (1.2.1 vs 1.2.0, each with its own warm-up) can never pass for a prompt
  whose output depends on history. The matched-history versions pass. The decision on 1.2.1 is his.

**NO-SWAP formula correction, Sat 05:4x: the 10K / 30K peak runs NOT run (stood down); 38 runner scripts patched;
tonight's start decisions re-checked, none changes.**

- **05:45:** the orchestrator asked for F-prompt_10k (gate 52) and I (gate 53) under one quiet ticket after BigMoE's
  t30x4 smoke. Its basis was a NO-SWAP of 62 GB that BigMoE's sequencer read at 05:44:18, 41 s after its t30x2 eval
  ended.
- **05:46-05:47:** Leap read NO-SWAP 44 GB, steady, with nothing big running (free 38.7 + speculative 0.6 + file-backed
  5.3; anonymous 16.8, wired 4.1, compressor 2.8). That is below both gates, so nothing was queued.
- **05:48:** the orchestrator corrected itself: stand down, the 62 was an overcount. Leap confirmed at 05:49 (nothing
  queued or armed). BigMoE's t30x4 train started at 05:48:59.
- **The bug: speculative pages are counted twice.** The shared NO-SWAP sum (free + speculative + purgeable +
  file-backed) counts them twice: speculative pages are already inside file-backed. Leap's vm_stat read at 05:46:21
  checks this: active + inactive + speculative + throttled = 666,143 + 642,255 + 37,249 + 0 = 1,345,647 pages = file-backed
  320,643 + anonymous 1,025,004. After big file reads (an eval, a model load), speculative can be many GB, which is how
  the 05:44:18 read reached 62.
- **Fix: NO-SWAP = free + purgeable + file-backed.** It is applied to every Leap script that used the old sum (38 files: 37
  tracked plus the gitignored scripts/stepb/run_flat2.sh). None was running; `bash -n` passes on all 38. At 05:51:04, with
  BigMoE's train running, the old sum read 11.03 GB and the new one 10 GB.
  - Older "available" measures that add inactive pages (run_g2.sh, run_wc2_night.sh, wc2_load.py) are a different,
    retired measure with no double count. They are left as they ran.
- **Re-check of tonight's starts (the orchestrator: only those within ~2 GB of their gates).** NO-SWAP here is the old
  sum, as logged. The speculative share at each start was not logged, so each margin below may have been smaller by
  that amount.

| start | gate | NO-SWAP read (old sum) | margin |
|---|---|---|---|
| A stage 3, 00:41:51 | 48 | 49 | +1 |
| D token check 1, 00:46:20 | 50 | 52 | +2 |
| E serve + cache, 00:47:30 | 50 | 52 | +2 |
| F len_300 / len_1k, 00:48:36 / 00:48:58 | 50 | 52 / 52 | +2 |
| F len_4k, 00:49:19 | 51 | 52 | +1 |
| diag3 diag_121_r3, 01:18:08 | 50 | 50 | +0 |
| diag3 diag_121_cold / diag_120_cold / diag_121_cold2, 01:18:38-01:19:41 | 50 | 51 | +1 |

  - Outside the 2 GB band: G +7, H +22, B +3, C +3; diag1's diag_120_r1 +7 and diag3's diag_121_nop2 +7.
- **No decision changes.** Swap stayed at 5,813 MB at every part start and at every runner's exit, and the +2 GB swap guard
  never fired. So no run went short of memory, even where the true margin was below +4.
  - Tokens and logits hashes do not depend on free memory. The engine's expert pools are fixed, and QOM's "resident"
    means resident in those pools. The diagnosis reproduced bit for bit across runs started at 50-58.
  - The peaks are each engine's own lifetime footprint, so the formula does not touch them.
  - The skips stay skipped: F-prompt_10k at 43 < 52 and I at 44 < 53 would read lower still. They remain with Kamen's
    morning PyCharm call.
- **The shipped package has the same sum: not changed, reported.** harness/evalsuite/cli.py free_gb() in the 1.2.1
  candidate (and in the published 1.2.0) is EvalSuite's code, vendored. It uses the same sum for its memory gate before
  each eval stage (need + 4), so after big file reads it can start a stage with less room than it reports. The engine
  (run.py, chat.py, serve.py) has no free-memory gate; setup.sh reads only the total RAM (hw.memsize).
  - A change there is a non-doc package change (rebuild and re-check), so it is Kamen's call. It went to the orchestrator.

**Prefill chunk dependence (BigMoE LAYERS a293386, Sat 16:58; the orchestrator 17:0x asked whether it touches an exact
mode): it does not touch the exact label. Checked in the 1.2.1 candidate's code, Sat 17:1x.**
- **Which modes:** every config. run.py sets R57_PRESET=45 for all of them (REF2-T tiers: BIGMOE_TIER=4433333333). Each
  config's loop prefills in 1,024-token chunks with set_arm("slot").
  - Exceptions are the P2 prompts of 256 tokens or fewer (chat-45 in 1.2.1, the 24 GB configs): they take 16- / 13-row
    forwards on the check path instead. P2 is lossy and gated.
- **The 40 is not a cap.** In the slot arm, GATE_N = 40 counts ids, i.e. rows x top-10 (bigmoe/stream.py:77
  `indices.size <= GATE_N[0]`).
  - A layer call of 4 rows or fewer takes the decode's gate path: REF2-T tiers, ranks 2-9 at 3 bits.
  - A larger call takes tier.prefill_call: exact 4-bit, every expert the call uses, sorted, in one scratch.
  - No routed expert is dropped.
- **The prefill is still chunk-dependent, in every mode, exact included.**
  - The call's shape picks the path above and MLX's gather kernel. The code notes that a different kernel choice changes
    bits on short prompts.
  - A bit difference that flips a routing choice moves a row a lot. This is likely what moves BigMoE's rows 8-18 %; it
    was not measured here.
  - Prefill rows and decode rows also differ by design: 4-bit ranks 2-9 against the verify path's 3-bit tiers.
- **Why the exact label holds:** exact means tokens bitwise equal to the exact reference (NAX6-N). The reference runs this
  same prefill on the same 1,024-token grid, and the exact configs reproduce that grid.
  - Chunk invariance was never claimed. 1.2.1 changes no exact config, and stage 2 / 3's exact checks passed 2 / 2.
- **Where it could matter, it is already handled:**
  - The conversation cache checkpoints only at multiples of the chunk and re-prefills [c, N) on the cold grid, so a hit is
    a cold prefill bit for bit (convcache.py). 1.2.1's check E passed its 2 cache-hit checks.
  - The 24 GB configs pin R57_PFCHUNK=1024, because W24e found that smaller chunks change the tokens.
  - PF-A (layer-major) is bitwise chunk-major over the same chunks.
- **What it means for research:** features must come from the path the runtime uses. That is the prefill on the 1,024 grid
  for the prompt, and the verify path for decoded rows; BigMoE's "exact re-prefill" does this.
  - A one-line LIMITS.md note on the chunking is possible in the docs-only rebuild, if Kamen wants it.
