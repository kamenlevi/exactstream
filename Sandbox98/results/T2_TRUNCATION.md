# T2: context-truncation oracle (orchestrator 22:4x for Kamen: solve it in theory, then test it as fast as possible)

Pre-registered Sat 14 Nov 22:5x, before any code ran.

**Theory under test.** The fresh state helps the head through (a) RETRIEVAL from the far prefix (attention in every
layer) and (b) deep COMPOSITION of the newest tokens. Cutting the far context off isolates (a).

**Test (no training; s30 unchanged).** For each sampled MBPP anchor s, the 2B runs on only the K tokens ending at s
(positions s-K+1..s; K = 0 means none) plus the drafted rows s+1..s+30 (= the true tokens, as on a correct chain). Its
post-norm final hidden at s+j-1 is the stand-in at depth j (REPLACE, as the oracle). K = 0 / 16 / 64 / 256; "full" = the
oracle (the captured true state). The chain and the oracle are scored on the same anchors.
- Anchors: MBPP's first 50 answers, every 3rd anchor in each answer (scoring counts only those rows; each anchor's chain
  is independent of the others). Plain truncation: no attention-sink tokens kept. If small K comes out low, a side run
  keeps the first 4 tokens as sinks, to tell lost information from an attention-sink artifact.
- Measure: bands 0 / 1-2 / ... / 23-30 per K. The line band is 23-30.

**Lines (MBPP, 23-30, sampled anchors):**
- **LOCAL:** K <= 64 within 3 pt of the oracle -> the missing information is local. A small deep model of the recent
  window can supply it: the drafter needs a cheap local trunk over the last ~K tokens, not retrieval.
- **RETRIEVAL:** K = 256 still > 5 pt below the oracle, with K <= 64 far below -> far retrieval dominates. The drafter
  needs lookups (a copy pointer, or attention over the target's KV cache).
- **MIXED:** otherwise. The knee K says how much window the drafter's state model must see.
- Side line: K = 0 (only the drafted tokens) vs the chain tells whether the head's carried state holds more than "the
  new tokens alone".

## Result (Sat 14 Nov 23:03; 50 MBPP answers, every 3rd anchor: 3,857 anchors; results/t2_trunc.json)
Sanity: the batched forward over the full prefix reproduces the captured state exactly (direction error 0.0000).
| stand-in at depth j (s30 unchanged) | 0 | 1-2 | 3-6 | 7-10 | 11-14 | 15-22 | **23-30** |
|---|---|---|---|---|---|---|---|
| s30's own chain | 87.63 | 78.08 | 68.14 | 63.81 | 61.51 | 58.90 | **55.42** |
| K = 0: the 2B over ONLY the drafted tokens | 87.63 | 17.50 | 43.46 | 52.92 | 58.53 | 64.44 | **69.62** |
| K = 16 | 87.63 | 66.30 | 66.96 | 69.78 | 71.87 | 73.86 | **75.47** |
| K = 64 | 87.63 | 79.44 | 79.12 | 79.54 | 79.63 | 80.05 | **80.61** |
| K = 256 (52 % of anchors actually truncated; the rest had the full prefix) | 87.63 | 86.82 | 86.15 | 86.23 | 86.45 | 86.08 | **86.25** |
| full prefix (oracle) | 87.63 | 88.24 | 87.72 | 87.90 | 87.90 | 87.73 | **87.71** |
**Verdict by the lines: MIXED.** LOCAL fails (K = 64 is 7.1 pt below the oracle); RETRIEVAL fails (K = 256 is only 1.5 pt
below). The knee is between 64 and 256 tokens of prior context.
**What it says (theory):**
1. **Deep composition of the new tokens is the biggest single piece.** The full 24-layer 2B over ONLY the drafted tokens
   (no prefix at all) beats the head's carried state by +14.2 pt at 23-30 and from depth ~15 on. At 1-2 it is useless
   (17.5): one or two tokens alone mean nothing. So the head's one layer composes the drafted tokens far worse than a
   deep stack does, even though the head sees the whole true prefix through its attention.
2. **Context then adds steadily, not in one jump:** +5.9 (16 tokens), +5.1 (64), +5.6 (256), +1.5 (full). Most of what
   the head needs lies within the last ~256 tokens, which the head can already attend to: its K / V cover every true pair.
3. So the deep gap is mainly **how deeply the drafter composes the drafted tokens with the recent context**, not access
   to far information. The 125B's capacity tests added WIDTH (a dense branch: no effect / hurts); DEPTH (more layers
   over the drafted tokens) is untested.
**Next test this points to:** a deeper drafter, i.e. 1 -> 2-4 layers over the same inputs, the extra layers zero-init
residual (step 0 = s30), TTT-trained on the fast format, against a control that holds. A cheap version first: the
head's own layer applied twice (weight-tied), which costs no new parameters.
