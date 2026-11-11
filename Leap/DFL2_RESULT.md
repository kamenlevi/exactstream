# L2 result: the native DFlash2 drafter on our 125B (Leap, Fri 2026-10-23 15:55)
Pre-registration: DFL2_PREREG.md (15:35).
- Run: one quiet-lock batch, 15:27-15:49. Features = 151,629 tokens prefilled; replay 2 min.
- Raw output: data/dfl2_result.txt, logs/job.out.

**Verdict.** ADOPT-CANDIDATE for code / math. NOT a lead for agentic or prose.

| class | DFlash2 tau8 (predicted) | P(first j right), j = 1..7 | true token in top-16, per depth | MTP K=7, same tokens |
|---|---|---|---|---|
| code (HumanEval+, 24) | **6.15** (5.0, 4.0-6.0) | .97 .92 .88 .85 .80 .77 .74 | 1.00 .99 .98 .97 .95 .94 .93 | - |
| GSM8K (60) | **5.26** (4.8) | .92 .84 .75 .67 .59 .52 .45 | 1.00 .99 .97 .95 .92 .90 .87 | - |
| IFEval (30) | 2.48 (3.2) | .65 .37 .21 .12 .08 .06 .04 | .94 .78 .67 .58 .52 .49 .48 | - |
| multilingual (24) | 2.09 (3.3) | .53 .26 .14 .08 .05 .03 .02 | .80 .63 .53 .47 .44 .40 .39 | - |
| agentic (12 real turns) | 2.86 (3.6) | .64 .41 .31 .22 .17 .14 .11 | .87 .77 .66 .59 .58 .55 .51 | **3.54 (DFlash2 = x0.81)** |

**Against the pre-registered lines.**
- Kill line (agentic < 1.05x MTP AND code < 5.0): not met, because code is 6.15.
- Adopt line (agentic >= 1.10x MTP OR code >= 5.5): met on code.

**Agentic, turn by turn.** DFlash2 loses to MTP on 11 of 12 turns; it wins only on edit06 (6.76 vs 6.17).
- Its first-token acceptance is 0.64, vs MTP's 0.85 on the same tokens.
- Likely cause: its attention is a 2,048-token sliding window over target features, and these prompts are 8-10K
  tokens. The MTP layer attends to the whole context.
- Prose / multilingual: probably the training mix (the card's evaluation is thinking-mode reasoning, code and math).

**Method and sanity checks.**
- Teacher-forced replay of the official loop (z-lab model_mlx.py, unmodified), greedy.
- Features: the 125B prefill on BigMoE's streamed runtime. Taps 5/15/25/35/45, mean of the 4 HC streams (as the capture
  patch).
- The prefill's own top-1 matches the stored continuations 0.99-1.00 on tf_gen and 0.90-1.00 on the agentic turns.
- MTP alignment checked: depth-1 agreement 0.846, as Chat1000's.

**What it means for 350 (code route).**
- 6.15 tokens per 8-row pass is 0.77 tokens per verify row. Compare:
  - our DFlash retarget: 0.43 per row (6.83 / 16);
  - the best published one-pass drafter in Chat1000's table: 0.54 (8.7 / 16);
  - the 350 plan's bar: tau16 ~10.9 per 16-row pass at 31 ms, i.e. 0.68 per row.
- **The drafter piece of the code route is met by a published drafter at block 8.**
- What remains is pass time. 350 on code needs an 8-row pass plus drafting in about 17.6 ms (6.15 / 350 s).
  - Floor estimate: dense 8.6 ms (with the diet) + a budgeted expert union ~8-11 ms + a 4-bit drafter ~2.5 ms
    = ~20 ms, i.e. ~300 tok/s.
  - The drafter in bf16 reads 1.2 GB per block (~4.3 ms), so quantize it.
- A small tree over its top-16 candidates (coverage 0.93-1.00 per depth on code) is the obvious way to lift 6.15
  toward 7-7.5.
- Today's kernels: an 8-row pass in RAM is ~62 ms, so ~100 tok/s on code vs ~90 for MTP k=3.

**Next (proposed).**
1. Engine integration for code / math, opt-in and exact: a greedy verify keeps outputs bit-identical.
   - Needs the 5-tap capture in the verify pass, the drafter in MLX (vendored code) and a per-block drafter choice:
     DFlash2 when its first draft agrees with MTP's first, else MTP (SpecDecode's cross-check pattern).
   - Owner: BigMoE or SpecDecode.
2. CPU / GPU bound for the hybrid: DFlash2 and MTP run lengths at every anchor, best-of per block. About one minute of
   GPU, no 125B load (features are on disk).
3. The tree over DFlash2's top-16 candidates for code.
4. For agentic: a full-context variant of the drafter (not published) or MTP alone. DFlash2 does not help there.
