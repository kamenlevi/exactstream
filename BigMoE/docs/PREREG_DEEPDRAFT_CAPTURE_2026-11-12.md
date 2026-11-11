# PREREG DEEPDRAFT capture, night 1 (Thu 12 Nov 00:1x, before the run): the 125B's own code answers + the drafter's inputs

**Origin.** Kamen's new half-main goal DEEPDRAFT (~/FableGod/deepdraft/PLAN.md, step 2: the data). The orchestrator
00:0x: capture tonight on fresh code prompts, none from the HumanEval+ / MBPP / LCB eval sets, after BigMoE's 45 GB A/B,
> (a quote from the project owner, kept private)
~/FableGod/deepdraft/PREREG.md.

## Prompt source

- **The dataset.** Magicoder-OSS-Instruct-75K, https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K,
  revision 5f839b1f368a76b161028bb9edff055db34022b2. Licence MIT (dataset card).
  - File data-oss_instruct-decontaminated.jsonl: 203,201,169 bytes, sha256
    f988f14d3516a9f958eeea310f63903727c6d3463bc9cb435288b6ca4bb09cf9, 75,197 rows.
> (a quote from the project owner, kept private)
    leave 20gb for swap"), with the orchestrator's go. It is kept in ~/datasets/deepdraft/magicoder/, outside every git
    repo and the public sync.
- **Selection** (BigMoE scripts/dd_prompts.py; the counts are in data/deepdraft/prompts_py_v1_summary.json):
  1. Python only: 38,284. Problem length 200-3,000 chars: 38,222. Unique: 38,222.
  2. Eval hygiene against EvalSuite's local items, the sets we measure on:
     - the items: dec_humaneval 60, dec_humanevalt 41, dec_humanevalx 104, dec_mbpp 100, dec_lcb6 175, plus
       raw/lcb6/test6.jsonl 175;
     - any word 10-gram shared with an eval text: 313 dropped;
     - any eval function name (>= 7 chars) as a whole word: 2,458 dropped.
  3. **Kept: 35,451.** They are shuffled with seed 1112 into prompts_py_v1.jsonl (sha256 c088afa6...8340) and used in that
     order.
- **Render.** The problem text is the user turn as written, with thinking off.

## Capture

- **The script.** BigMoE scripts/dd_capture.py, a copy of TrainLab sticky/dlog_capture.py. It changes only two things: the
  output directory, and the `magicoder` source.
- **The harness, unchanged.** D-23's 45 GB preset: REF2-T tiers 4433333333 / 3c, C_l 112, MAXK 160, build_next, the
  DFlash2 L1-20 live loop (dfl arm); C 224; answers capped at 384 tokens (DLOG_MAXTOK); greedy.
- **Per prompt**, ~/FableGod/deepdraft/capture/n1112/<key>.npz:
  - h, DFlash2's context input, the drafter's input feature: 2560, bf16 bits;
  - wd, the trunk's pre-mixer widened stream: 10240, fp16;
  - tk / lp, the 125B's top-32 per row;
  - ids, asst, gen, key, src.
- **Self-check.** On every prompt, the logged top-1 must equal the next token on every answer row. Below 99.5 % the run
  stops (DLOG-ONPATH-FAIL).
- **Run control.** scripts/dd_capture_night.sh under turn45 (a quiet ticket; NO-SWAP >= 42 GB per try):
  - the last start is 12:30 and the hard stop 13:30;
  - DLOG_CAP_GB 80 written, DLOG_MIN_FREE_GB 115, so the disk stays above 110 GB;
  - a swap guard stops it at +2 GB;
  - it resumes after a crash (prompts already done are skipped), at most 4 tries; DLOG-ONPATH-FAIL is not retried.

## Expected [P]

About 200-290K answer tokens per hour, from the live loop's ~55-80 tok/s on code. Over the time left after the chain
(~9-10 h), that is about 2M answer tokens if the 80 GB cap allows. At about 25 KB per logged row the cap allows about
3.2M rows, prompt rows included. The first try also serves as the smoke test: it was not possible earlier, because
NO-SWAP was 22 GB at 00:08.

## Reported

Files, answer tokens, the per-prompt self-check, GB written, the stop reason, and tokens per hour (night_n1112.log). The
outputs stay out of every git repo and the public sync (*.npz).

## Amendment 1 (Thu 00:2x, before the run): the cache C follows the measured memory

The orchestrator, 00:2x. Kamen's apps are quit at 00:25, but about 15 GB stays in his Claude CLI sessions, so NO-SWAP is
expected at about 32-37 GB, not the 45 GB preset's 42+. The capture therefore runs at the largest C that fits the room
measured at each try, with an 8 GB margin. The peaks are estimates [P], interpolated between the 24 GB card (C 100, 23.7
GB) and the 45 GB default (C 224, ~44 GB):

| NO-SWAP | C |
|---|---|
| >= 52 GB | 224 |
| >= 47 GB | 192 |
| >= 42 GB | 160 |
| >= 37 GB | 128 |
| >= 32 GB | 100 |
| below 32 GB | wait |

The tokens and features do not depend on C, which sets only the speed: greedy REF2-T, batch-invariant, enforced by the
per-prompt self-check. The run log records the C used on each try. The 45 GB P3 + P2 A/B moves to the next night with
room. W24T (24 GB) goes first tonight, then the capture. turn45's gate for the capture is now PEAK 24 (NO-SWAP >= 32).

## Amendment 2 (Thu 00:5x, before the night run): the ext build, and the smoke test

- **The fix.** The first queued run (00:46) failed at model load on all 4 tries: "native+p11+pad needs an ext from the
  lever-1 build". dlog's default ext build (build_next) lacks set_direct_split, which the store's records3 layout needs.
  The capture now runs with BIGMOE_EXT_BUILD=build_qom, the build that both the current 45 GB stack (s5p_env) and the
  24 GB card pin.
- **The smoke test** (00:51, C 128, 2 prompts; ~/FableGod/deepdraft/capture/smoke1112/):
  - the warm-up and both prompts were on-path 1.0000;
  - 233 + 131 answer tokens at 29.9 / 39.0 tok/s, 32.6 tok/s pooled;
  - files of 13.0 / 8.1 MB;
  - 20.9 GB active, footprint peak 24.9 GB.
- **The revised estimate [P].** At C 128 about 115K answer tokens per hour, so about 1.4M by the 13:30 stop. At about
  55 KB per answer token with the prompt rows, the 80 GB cap allows about 1.5M.
