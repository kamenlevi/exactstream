# MISSION B3 (Fable B): CLOUD'S TWO DENSE IDEAS, AND THE 32-ROW CROSSOVER

(Draft by the orchestrator for Kamen, Mon 2 Nov 2026 ~22:1x. This replaces the earlier B3 draft: your BOUNDARIES.md section 4F already priced the FAST dense floor, ~16-17 ms, or ~14-15 with HC8.)

**Why.** Cloud task 16 (~/FableGod/CLOUD_ANSWER_NEWIDEAS.md) proposed two genuinely new dense-pass ideas:
1. **#3 the seeded exact argmax for the LM head** (exact; cloud's "speeds up all models" idea): the drafted token seeds a chunked Cauchy-Schwarz bound that prunes the 248K-word vocabulary per row, so most head chunks are skipped while the argmax stays exact. Cloud: head bytes -80-90 %, -1 ms halved per pass.
2. **#4 predict-then-correct mixers** (a new reference): predict the HC coefficients a layer early, launch the big GEMM, then correct with an exact low-rank term. It breaks the HC read's dependent chain. Cloud: -2.5 ms halved.

The 32-row question is still open:
- Cloud predicts 32 rows lose on this week's engine (66 vs 78 tok/s).
- The dense pass is flat in rows, the expert union grows x1.5-1.6 per doubling, and the copy override pays x1.211 at 32 rows vs x1.041 at 16.

**Deliver**
1. **#3:**
   - Is the bound exact in bf16 / fp32 arithmetic (the argmax must equal the full head's, ties included)?
   - What fraction of the vocabulary does it prune, on real final hidden states (from any capture with the trunk's final stream) and the real head weights? Do a CPU numeric check.
   - Bytes and ms saved at 1 / 16 rows.
   - Does it hold for the MTP head's and the drafter's heads too?
   - Prior art (FR-Spec, vocabulary pruning, sparse softmax / MIPS bounds): is it new?
2. **#4:** price it from HC_READ.md's math.
   - How well do layer L-1's states predict layer L's mixer coefficients (CPU, where states exist)?
   - The correction's cost, and whether the result is row-invariant (FAST-K) or needs a new reference.
3. **The 32-row crossover, on both tracks, at the floor engine:** at which dense ms, expert-kernel efficiency and kept do 32-row checks beat 16? Use B3's measured per-row costs and the copy override's 32-row gain.
4. **Verdicts and pre-registered lines** for Leap's screens, and the pick.

**Rules**
- CPU only; writes only in ~/FableGod/fableb/; no downloads. Halve gains. Mark every number [K] / [P] / [G].
- Two tracks: nothing is dropped for not being bitwise.
