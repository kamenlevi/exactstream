# Cloud Fable 5.1, task 16 (CLOUD_TASK_NEWIDEAS.md): its summary as pasted by Kamen, Mon 2 Nov 2026 ~22:1x
(The full ANSWER.md stays in the cloud session: 1,230 words, 4 of 6 Python runs, no web.)

Six ideas, two per piece, none of them in the "tried" table:

1. **SSD, exact: pre-route on the partial sum.**
   - OOO already computes resident experts first and waits on misses. Run layer L+1's router at that moment, on layer L's output minus the in-flight experts, and issue the next reads before L finishes.
   - R-57 reads this layer's input (recall 0.69); this reads the near-complete output, one sublayer closer.
   - Chain 22 -> ~4-5 ms at recall 0.9: -8 ms halved.
2. **SSD, FAST: sketch-on-miss.**
   - A rank-32 shadow of every cold expert in RAM (2.8 GB) serves a miss instantly, while the exact expert loads in the background.
   - Shadows touch ~1.6 % of expert uses, so the deviation is ~0.05 % of tokens.
   - Exposed SSD -> ~2 ms, day and night: -10 ms halved.
3. **Dense, exact: seeded exact argmax for the LM head.** The drafted token seeds a chunked Cauchy-Schwarz bound that prunes the 248K vocabulary per row, exactly. Head bytes -80-90 %, -1 ms halved per pass.
4. **Dense, new reference: predict-then-correct dynamic mixers.** Predict the HC coefficients a layer early, launch the big GEMM, and correct with an exact low-rank term. It breaks the 5-kernel chain: -2.5 ms halved. Void if the mixers are static (they are not: HC_READ.md, every mixing coefficient is input-dependent).
5. **Kept, exact: MTP-seeded block drafting.** The target's own MTP head gives depth 1 at 0.94 instead of 0.898, and DFlash2 drafts depths 2-16 from it. +0.25 kept halved, every domain.
6. **Kept, FAST: lookahead-consistency acceptance.** R1's test on row r plus a veto from row r+1: accept a near-tie only when the target's next row, conditioned on the draft, is sure. +8 % chat kept halved, at a better exchange rate than R1.

- **Cloud's pick:** #1. It's exact, needs zero RAM and no model, and takes a 10-minute screen on recorded data. "Pass or fail, it tells you in ten minutes whether any cheaper forecast can ever work."
- **The all-models idea:** #3, the seeded exact argmax. Every speculative verifier holds a strong candidate yet throws the full logits away, and the win grows as models shrink.
- **Its correction to task 15:** hedge candidates for a miss at row i need the target's correction token. Coverage is ~0.25, not 0.5, so the hidden-draft row saves ~1.6 ms, not 3.2, and the ladder ends near 220 rather than 224. That matches Fable A's DRAFT_HIDDEN (usable share 0.22-0.59).

## Orchestrator's check (22:1x)
- **#1** is close to R-57 / OR1 (look-ahead from earlier states, TRIED: recall 0.69 at a 1-layer lead), but it uses a later state: the partial MoE output. Its premise, recall 0.9 from the partial sum, is unmeasured, and a replay needs per-layer partial MoE outputs that may not be recorded.
- **#2** resembles published low-rank expert substitutes. Its deviation estimate (~0.05 %) needs checking against the 12 GB frugal kills, which failed on quality.
- **#3** relates to published vocabulary pruning (FR-Spec), but an exact bound-based argmax is plausible new.
- **#5** is close to D-22 / the hybrid walks in MTP Head's queue.
- **#6** is new as a rule.
- **Routed:** Fable A mission 30 gets #1, #2, #5 and #6; Fable B mission B3 gets #3, #4 and the 32-row crossover.
