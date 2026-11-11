# CLOUD TASK 16: invent. Genuinely new ideas for the three pieces that decide 350.

A closed thinking task, and a creative one. Don't use the web; think. You know this engine from tasks 1-15.

> (a quote from the project owner, kept private)
1. **The SSD stack:** -23 of the -48 ms. Exposed SSD is ~22 ms if R-57's look-ahead fails at 45 GB.
2. **The dense pass:** exact stops at ~24-26 ms against ~14 ms of floors. The remainder is ~1,000 kernel boundaries and dependent chains per check.
3. **Kept per check:** ~10.5 on code (12-13 at the drafter ceiling) and ~2.4 on chat, where the 125B is unsure on 55-58 % of rows and the token after a hard one is usually hard too.

**Already tried, so don't propose these** (measured unless marked):

| piece | tried |
|---|---|
| SSD | LRU / LFU / Belady eviction; rank-aware admission and rank-priority eviction (both KILL); pinning; prompt protection; next-layer look-ahead prefetch (recall 0.69, in build as R-57); 2-3-layer forecasts (recall 15-40 %); cross-pass forecast from the MTP head (KILL); trained pre-routers (top-2 precision 0.68); entropy coding of experts (<= 14-15 % fewer bytes, decode too slow); plane-aware prefetch; one-record packs; JOINT base + sidecar reads; per-layer pool sizes; sidecar x1.5 slots; the gate; the 12 GB two-pass; frugal routing (quality KILL); smaller top-k at runtime (quality KILL); expert pruning (excluded by the goal: all 512 experts) |
| dense pass | fused HC reads (S4, X5, kernel B: -0.74 ms); a one-dispatch GDN body (~0); mx.compile; fused gates; 8-bit HC weights (quality PASS with a lean); the row-interleaved GDN recurrence (G3, queued); persistent / mega-kernels (Fable B analysing now); reordered or chunked GDN verify (KILL for exact, open for fast) |
| kept per check | DFlash2 and fine-tunes; the native MTP chain; two-drafter pick (DIVERSE2); copy override + gate; relaxed acceptance R1 / R2; the comment loophole; trees (exact branching is cheap, ~1 ms per fork, but a rescued hard token continues only ~1.1 tokens on chat: no pay); early-tap drafting (EARLYTAP, Tuesday); the CPU as drafter (KILL, 62 ms); the ANE as drafter (12.4 ms per 3-candidate call) |

**Deliver**
1. **Six genuinely new ideas, two per piece** (<= 150 words each). "New" means you know of no paper or engine that does it. For each give:
   - the mechanism, in one paragraph
   - the closest prior art you know, and why this differs
   - exact or quality-defined (both tracks are first-class now: EXACT bitwise, and FAST judged by same-scores tests)
   - the expected gain on this Mac, halved (our estimates run ~2x high)
   - the cheapest test that kills or keeps it: a CPU replay on recorded routing or walks, or a 10-minute GPU screen, with a pre-registered line
2. **Your pick** (<= 150 words): the one idea to test first, and why.
3. **One "speeds up all models" idea** (<= 150 words): something from this project that would help any MoE or hybrid model on any hardware, not just this 125B on this Mac.

**Rules**
- The full 125B (all 512 experts), this MacBook only, no external hardware.
- Mark every number as given / arithmetic / judgement.

**Budget, so the answer always arrives**
- No web. At most 6 short Python runs (each under a minute).
- Within your first ~20 minutes, write a complete first draft to ANSWER.md. Update it once or twice after that.
- If you're running long, cut depth, not ideas: every idea gets at least its mechanism, novelty and test.
- When done, print ANSWER.md as your final message, in <= 1,400 words.
- Don't ask questions or wait for approval: make reasonable calls and note each in one line. Don't change or commit anything in the repository, and don't install or download anything.
