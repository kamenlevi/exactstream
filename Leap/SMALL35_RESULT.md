# Smaller-model check for 350: Qwen3.6-35B-A3B fully in RAM. Leap, Sat 2026-10-24 07:56
Pre-registration: SMALL35_PREREG.md (06:56). Kill line (NewIdeas): best code tok/s < 200 with drafting -> no.
Run 07:53-07:55 under a quiet lock: scripts/small35_bench.py, data/small35.txt.
- Setup:
  - Youssofal 4-bit body (MTPLX build) via mlx_lm 0.31.3, fully in RAM, 22.6 GB peak, swap unchanged;
  - z-lab's own MLX DFlash loop with z-lab/Qwen3.6-35B-A3B-DFlash (0.4B), greedy.
- Sets:
  - 12 HumanEval+ prompts: EvalSuite's tf_code items, re-rendered with the 35B's template; identical to the 125B's
    rendering, 12 of 12;
  - 2 of Kamen's agentic samples: bash17, edit00.

## Verdict: NOT killed (code 219 tok/s >= 200), but 350 is not reachable this way: 0.63x of it on code, ~0.3x on agentic

| mode | code | agentic |
|---|---|---|
| plain decode | 106.2 tok/s | 97.0 |
| DFlash block 16 | **218.9** (10.92 + 1 per block; 156-314 by prompt) | 65.7 (3.84 + 1) |
| DFlash block 8 | 207.6 (6.69 + 1) | 79.2 (3.42 + 1) |

- "Exact" here means greedy with a batched verify. DFlash output equals plain decode on 5 of 12 code prompts; on the
  others it leaves plain at a near-tie after 149-294 tokens, as with every batched verifier (SpecDecode FINDINGS 3).
- A bit-exact mode (batch-invariant passes) costs ~10 % (SpecDecode 15).
- Predictions:
  - plain ~85-95: measured 106, low;
  - DFlash 16 code ~190 (160-240): 219, in range;
  - accepted 12.0 + 1: 10.9 + 1. Lower, but a block costs ~55 ms, not the ~65-70 I estimated;
  - P(>= 200) ~45 %: yes;
  - DFlash 8 ~150: 208, low;
  - agentic DFlash ~70: 66.

## Against the 125B and against 350
- Code:
  - the 35B with DFlash gives ~219 exact-ish;
  - the 125B gives ~65 (agentic-mix today) to ~126 (code with a 6.8 drafter in RAM, Chat1000 REPORT), and it cannot
    sit fully in RAM under the 45 GB rule;
  - so ~1.7-3.4x on code.
- Agentic: DFlash loses to plain on this model. MTP / lookup are the drafters there, measured locally at ~95-130 tok/s
  (Q8 lookup-else-MTP 1.7-2.5x; 284 on copy-heavy edits on the full 4-bit model; MTPLX card D1 138). 350 is ~2.5-3.5x
  away.
- Intelligence: Artificial Analysis index 18.2 vs the 125B's 39.8, less than half. No Eval suite row exists for the
  35B; one would take a full suite run (hours of GPU).
- Headroom left on code:
  - A 16-row block costs ~55 ms in this loop. Its bytes are ~4 GB, ~16 ms at 90 % of 280 GB/s. So kernels (MTPLX-style
    fused paths) could in principle give ~2-3x.
  - Confidence row truncation (DFL2E_RESULT.md side finding) adds ~10 % on top.
  - 350 on code is not physically ruled out for the 35B, unlike the exact 125B. It needs real kernel work, and it buys
    less than half the intelligence. Kamen's call whether that trade is interesting at all.
