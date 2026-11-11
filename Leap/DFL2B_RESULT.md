# L2b result: does a longer drafter context fix agentic DFlash2? (Leap, Fri 2026-10-23 17:37)
- Pre-registration: DFL2_PREREG.md, L2b final design (17:30, commit 9ecf3f0). Bet assigned by NewIdeas at 16:26.
- Run: one quiet lock, 17:28-17:36. Raw output: logs/job.out and data/dfl2_result_{w*,hyb*}.txt.

**Verdict: KILL (NewIdeas' line).**
- At window 16,384 every turn is seen in full. tau8 = 2.93 vs MTP K=7 3.54 on the same tokens (x0.83).
- The best training-free variant, NTK-aware RoPE x8, reaches 3.11 (x0.88). Still below the line.

| drafter window (12 agentic turns) | tau8 | first-token acceptance | vs MTP 3.54 |
|---|---|---|---|
| 256 | 2.58 | .59 | x0.73 |
| 512 | 2.64 | .59 | x0.75 |
| 1,024 | 2.76 | .62 | x0.78 |
| **2,048 (as trained; reproduces L2 per turn)** | 2.86 | .64 | x0.81 |
| 4,096 | 2.83 | .63 | x0.80 |
| 8,192 | 2.78 | .62 | x0.79 |
| **16,384 (all context)** | **2.93** | .65 | x0.83 |
| 16,384 + NTK RoPE x8 (base 82,685) | 3.11 | .68 | x0.88 |

**Against the predictions.**
- Full window: predicted 2.9-3.3 (15:58) and 2.6-3.0 (17:30). Got 2.93. First token: predicted .66-.72, got .65.
- NTK: predicted 2.7-3.0. Got 3.11, better than I expected.
- Sweep 512 -> 2,048: I predicted flat (< 5 %). **Wrong:** +8 % (2.64 -> 2.86), about +3.5 % per doubling. Context
  does help, a little.
- Reproducibility: OK.
  - Window 2,048 on the new full-context features gives L2's per-turn tau8 exactly.
  - The prefill's top-1 agreement is identical on every turn.

**Where context helps.** Full context + NTK lifts three bash turns and text27 by ~23-30 %, to within 0-5 % of MTP:
- bash17 2.42 -> 2.98, bash19 2.28 -> 2.91, bash21 2.68 -> 3.49, text27 2.33 -> 2.87.
- edit06 drops 6.76 -> 6.08.
- write14, already fully visible at 2,048, is unchanged (1.72 -> 1.70).

**Hybrid bound (NewIdeas' point 2).** Per-anchor run lengths of both drafters on the same tokens:

| window | MTP alone | cross-check (DFlash2 if its first draft = MTP's, else MTP) | oracle best-of per block |
|---|---|---|---|
| 2,048 | 3.54 | 3.37 (x0.95) | 3.86 (+9 %) |
| 16,384 | 3.54 | 3.40 (x0.96) | 3.86 (+9 %) |

- Predicted: cross-check 3.6-3.8, oracle 4.0-4.3. Both came in lower.
- On agentic text the cross-check LOSES to MTP alone. When the two agree on the first token, DFlash2's later drafts are
  worse than MTP's.
- For the engine integration (BigMoE / SpecDecode):
  - Agentic turns: MTP alone; no cross-check.
  - Code / math: measure the per-block policy against real MTP chains before choosing it. We have no MTP chains on code
    yet.

**What it means.**
- The window explains only a small part of the agentic gap. Training-free full context recovers about a third of it at
  best: 2.86 -> 3.11, against MTP's 3.54.
- A full-context fine-tune: extrapolating the in-distribution trend (+3.5-5 % per doubling) gives ~3.2-3.3, still below
  MTP. This is a projection, not a measurement.
- The main gap is the drafter's training mix (code / math / reasoning):
  - write14 is fully visible and still loses by the same x0.81.
  - IFEval and multilingual have short prompts, and they are weak too.
- A drafter that beats MTP on agentic turns needs agentic training data (the 125B's own outputs) as well as full
  context. That is a bigger job than a context fine-tune.
- For 350: agentic needs about 5 accepted tokens per 8-row pass (at a ~20 ms pass). That is near-copy-level fidelity
  (A1), out of reach for a 0.5B drafter.
- **Recommendation:** agentic stays on MTP, and DFlash2 serves code / math only. No fine-tune for context alone.
