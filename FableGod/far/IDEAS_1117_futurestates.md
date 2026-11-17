# IDEAS_1117_futurestates: a drafter that reads the 125B's FUTURE STATES from placeholder rows (Fable FAR, Tue 17 Nov 2026 07:3x-07:4x, desk + one CPU screen)

BREAK98, FAR's lens: the drafter that consumes partial or approximate 125B information; an architecture + training
recipe TrainLab can run this week, with a 2B analog in Sandbox98 first. Route 1 (true-state-class information), as the
orchestrator refined at 07:3x. Marks: [K] measured (source), [P] arithmetic on [K], [G] judgement.

## 0. The physics that decides it, from the records
- A 125B row is cheap only when it RIDES A BATCH. The check reads every weight once for its 32 rows; rows 33-64 cost
  ~0.28 ms each (0.15 int8 dense compute + 0.08 expert-union growth x1.19 at 64 rows + 0.04 cores; band 0.19-0.45)
  against the 32-row average of 0.68 ms [K IDEAS_1116b, check25_bound_real] . A row's price is therefore NOT the
  "believed limit" (a sequential partial pass, ~0.5+ GB per token); the limit is that a batched row needs its TOKEN
  before the check, and the drafter's tokens are what we lack. Trees spend the cheap rows on sibling tokens, and 66-86 %
  of those rows die [K S11]; Lunatic priced that at 2.1-2.8 ms per extra kept, above the 1.8 ms bar [K lunatic 1116c].
- The information the drafter lacks is the 125B's own computation on the window (its "plan"): drift is 92 % persistent
  per window across blocks [K fablec 1116c], and the state that fixes it is the mid-depth re-plan, which no head-side
  lever reaches (SIB, ASSIM, carryac, taps: all KILL / NO EFFECT) [K records].
- So the question becomes: **can a batched 125B row carry the 125B's plan for a position WITHOUT knowing its token?**
  If yes, the check of block n can compute, at ~0.28 ms per row, the 125B's view of positions the next block will draft,
  and the drafter reads them. That is a 125B state at every drafted position (up to k deep) for ~0.3-0.4 of a row's
  marginal price and ~1/50 of a 32-row check's average per-row price.

## 1. FUTURE-STATE ROWS (best): placeholder rows in the check batch give the next block's drafter the 125B's state at its first k drafted positions  [new here; published roots: PaSS lookahead embeddings (2023), gated-LoRA mask tokens ("Your LLM knows the future", 2025); NOT SIB: no rejected-row state is used]

**Mechanism.** In block n's check batch, under spine row r hang a CHAIN of k placeholder rows: row j sits at position
r + j, holds a placeholder embedding (no token), and attends to the prefix + spine rows 1..r + placeholder rows 1..j-1
(an ordinary tree-attention mask; the engine's tree machinery already does arbitrary ancestor sets). The 125B
processes them in the same pass as the 32 check rows. Its state at placeholder j = its 48-layer computation of
"what comes j positions after row r", conditioned on the true prefix and nothing wrong: not a sibling state (SIB fed
states conditioned on a WRONG token; KILL), not a mask the model was trained on (the 125B is frozen; no LoRA on a
streamed 125B this week), but the frozen model's response to an unknown-content position.
When the check ends at accepted row m-1 with bonus token t_m, block n+1 drafts from the true state at m-1 as today, and
the head at depth j (predicting position m+j) reads, beside its own carried state, the placeholder state at offset j
under row m-1, IF a chain hangs there. Where no chain hangs (uncovered ends) and past depth k the head runs on its own
state exactly as today, so nothing is lost. The head learns to read the new input through a zero-initialised projection
(D-16's recipe), trained on captured placeholder states along the true text (the training rows are the accepted rows at
runtime: the same distribution, no mismatch).

**Where to hang the chains (the CPU screen, done; `screens_1117/endcover.py` -> `endcover.out`, topk cells,
teacher-forced 31-deep rows, so an approximation of the real chain's end).** The block ends at the first miss, and the
head's own p flags it: on HE+ walk starts the end is under the END row or the single lowest-p row in 70.8 % of blocks
(2 chains), under the lowest 4 + end in 79.1 % (5 chains), under every row with p < 0.5 + end in 85.5 % (7.1 chains);
MBPP 58.8 / 66.8 / 79.2 %. All anchors (every position): 87.4 / 91.3 / 92.0 % HE+, 66.7 / 74.7 / 80.9 % MBPP.

**Price per block on the 125B [P].** chains x k rows at the margin: 2 chains x 8 = 16 rows ~4.5 ms today's engine
(0.28 ms/row; 3.0-7.2 across the band), ~+8 % of a 56 ms check, +18 % at the 25 ms bound; 5 x 8 = 40 rows ~11 ms.
Unknown and measured by the capture: the placeholder rows' routing. If their top-10 experts fall >= 80 % inside the
check's union, 0.28 ms/row stands; if they route to odd experts the union grows and the price rises toward 0.45+.
Draft ms: +0 levels; the head's extra input projection is read once per level (~20 MB at 4-bit if 10,240 -> 2,048).

**The number it can buy, bounded [K/P].** Ceiling = the true-state chain: 13.77 -> 20.77 HE+ / 6.92 -> 12.72 MBPP, and
with a tree 24.5-26.8 [K lunatic]. A placeholder state at offset j has the prefix's plan but not the j-1 tokens in
between; the head supplies those. So the bound for depths 1..k is the LAB gap at bands 1-8 (walk-strict own -> true:
1-2 +1.6, 3-6 +4.0, 7-10 +3.5 pt on HE+ [K fablec audit]) x coverage; what fraction a frozen-model placeholder carries is
the unknown [G: a third to two thirds]. The pay line at the bound with 2 chains: kept >= +18 % (16.26 -> 19.2) on HE+;
with the future states also feeding the tree's shallow bands (where trees sit BELOW the zero-drift chain: 93.7 / 94.3 /
96.3 at 3-14 [K]) that is +3-4 pt at bands 3-14, which is what the 2B screen reads out.

**Screen 1: Sandbox98, today, <= 5 GB, ~30 min, before any 125B capture.**
1. Future states: for each anchor of the screen pool + HE 60 / MBPP 100 windows, run the frozen 2B over
   [prefix .. anchor] + 8 placeholder rows (KV cache of the prefix reused; a chain, so plain causal attention), store
   W at the 8 placeholder rows. Three placeholder variants, each its own arm: (A) token `<|fim_pad|>` (the base saw it
   as padding inside FIM training; id 248063 on the 125B, the 2B's own id from its tokenizer); (B) the mean of the
   input-embedding table (an embedding override); (C) `<|vision_pad|>` (never seen: a control for "any foreign
   vector"). ~48 K rows through the 2B: 1-2 min.
2. Head arms (`sb/train_multi.py`, 300-400 steps, pool2 setup): ctl; fut_A / fut_B / fut_C = the head with a third
   input stream, fc over [emb, W_own, P(W_fut_j)] with P zero-initialised, where at depth j <= 8 W_fut_j is the
   placeholder state at offset j under the anchor and at j > 8 it is zero. Training-time test as today (own states
   on the chain; placeholder states from step 1 at the matching offsets).
3. Read-out: `sb/evalx.py` walk-strict 1-2 / 3-6 / 7-8 / 1-8 and G = (fut - own) / (true - own) per band (the SIB
   definition; true = the LAB stand-in), then `sb98/tree8.py` tree-32 kept with the future states at depths <= 8.
   **PASS: G(1-8) >= 0.30 on HE with the CI's lower end > 0 AND tree-32 kept >= +5 %; BETWEEN 0.10-0.30; KILL: G < 0.10
   on every variant** (then the frozen model's placeholder rows carry no plan, and only a LoRA'd 125B could; out this
   week). Also print: the 2B's OWN top-1 at each placeholder row vs the true token at that position (free zero-shot
   lookahead accuracy by offset; if offset 1 is far below the head's depth-1 accuracy, the plan is weak).

**Screen 2 (125B, tonight's capture; name by 18:00, below).** TrainLab trains t30b + the zero-init future-state
projection on the captured placeholder states along the true text (HE+ / MBPP banks + the n1112 answers' anchors as
far as captured), 1-2 h on the existing recipe; BigMoE's cells / tree harness reads out walk-strict bands and tree-32
kept with future states hung under the end row + lowest-p row. PASS line as Screen 1, on the 125B's numbers.

**Capture request for BigMoE tonight (by 18:00):** in the mid_{heplus24,mbppbank} pass, for every 4th true-text row a
(>= 1,000 anchors per set), 8 placeholder rows hung as a chain under a at positions a+1..a+8 with token id 248063
(`<|fim_pad|>`; a second variant with the mean input embedding if the engine takes an embedding override), storing per
placeholder row: the head's input stream (the pre-mixer widened stream W, 10,240 bf16), the layer-16 residual (already
captured for true rows), the 125B's top-8 ids + log-probs (zero-shot lookahead accuracy by offset), and the top-10 expert
ids per layer (union growth vs the window's true rows). Size ~1,000 x 8 x 10,240 x 2 B = 164 MB per set per variant
plus the small arrays; the rows ride the capture's own batches (a tree mask), ~8 x the anchors' rows at the margin.

**Quality risk.** None at the check: exact verification, the drafter only reads more. A wrong placeholder state costs
kept, never quality.

**What kills it.** (1) Screen 1 KILL on all three variants: a frozen trunk gives no usable plan at a placeholder.
(2) Placeholder rows route to experts outside the check's union at > ~30 %, so the price per row is a full tree row's
and the lever pays only with early death. (3) Coverage: on MBPP the end is under the lowest-p row + end in only 59 %;
there the lever needs 5-7 chains (40-56 rows), which only pays at the bound if kept rises >= +40 %.

**Why it is not on the kill list.** SIB = the 125B's states conditioned on the head's WRONG token (a different
sequence; direction error 0.95-1.10 vs own 0.85-0.99 [K]). Placeholder rows are conditioned on the TRUE prefix and an
explicit unknown; the state is the 125B's expectation over continuations, which is the quantity drift loses (the plan).
Span tree / copies, keyed widening, taps (per-token 125B passes), ASSIM / ERRMEM (head-side vectors) are different
mechanisms. Nobody on record has put placeholder rows in a speculative check batch to feed the next draft.

## 2. PREFIX-DEPTH INPUTS: the head reads the 125B's mid-layer computation on the ACCEPTED rows and its attention caches  [published roots: EAGLE-3 feature fusion (2025), GLIDE target-KV reuse (2024); new here as a drift lever; free]

**Mechanism.** Today the head reads only the final widened stream of the prefix (its own K/V built from it). The 125B's
mid-layer residuals of every accepted row (layers 8 / 16 / 24: tonight's capture IS the training data) and its
attention-layer K/V over the prefix (12 full-attention layers, ~1 KB per token per layer: ~18 MB for 1.5 K tokens,
stored anyway) are free. Two input changes to t30b: (a) the anchor and accepted rows feed [emb, W, R8, R16, R24] through
a wider fc (EAGLE-3's fusion; the extra columns are read once per block, not per level: ~0.4 ms at 126 MB bf16, ~0.1 at
4-bit); (b) one or two extra attention sub-layers in the head whose queries come from the head's state and whose K/V
are the 125B's layer-16 / 24 caches of the prefix (GLIDE), so the head sees where the 125B looks mid-depth.
**Why it may attack drift:** the persistent per-window component is context information the head does not carry in a
final-layer summary; the mid-layer residuals and caches are the 125B's own representation of that context.
**Number [G].** EAGLE-3's fusion ablation is worth ~10 % of acceptance length on top of EAGLE-2-style heads; GLIDE's
KV reuse was worth more for weak drafters. Expected here: +1-2 pt walk-strict at bands 3-14, kept +3-8 %. Not x1.5.
**Screen (Sandbox98, <= 30 min):** the 2B's taps {7,12,17} after layers (data/*/KEY.taps.npz exist) as inputs (a) and
the 2B's attention K/V of layers 12 / 17 for (b); arms ctl / fuse / fuse+kv; read-out as idea 1. PASS: bands 3-14 >= +1.5
pt, tree-32 kept >= +4 %. **Cost per block on the 125B:** 0 bytes new (the check keeps 3 x 32 x 10,240 bf16 = 2 MB of
residuals); draft +0.1-0.4 ms. **Kills it:** the 2B gains < 0.5 pt (the final stream already summarises the context
for the head's purposes), or the 125B's attention layers are too few / too late to carry the plan.
**Training by Thursday:** yes: the t30b recipe with three extra input streams and tonight's capture; (b) needs the
125B's K/V in the capture (name: the layer-16 and layer-24 attention K/V caches per true-text row; ~1 KB per row per
layer: small).

## 3. Combined form (if both pass): future-state rows for depths 1-8, prefix-depth inputs for the rest
Idea 1 fixes the shallow depths where ties and drift both act and where trees sit below the zero-drift chain; idea 2
is free and acts at every depth. Both are drafter-side, exact, and the same head can carry both input streams.

## Not proposed
- A batched partial pass over the drafted rows for a second draft wave (Fable B 1116c #4: the taps are true only up to
  the first changed token; serial cost 10-14 ms).
- Soft-token rows (the head's expected embedding at unsure positions instead of a hard token): a milder SIB; worth one
  arm in Screen 1 only if variants A-C all land BETWEEN (a 4th variant: the head's soft token at offset 1, placeholders
  after).
- A LoRA'd 125B for mask tokens (the published form): no backward through a streamed 125B this week.

## Summary line
Best: FUTURE-STATE ROWS: k = 8 placeholder rows (`<|fim_pad|>` / mean embedding) hung in the check batch under the end
row and the lowest-p rows (2 chains cover 71 % of HE+ block ends; 16 rows ~4.5 ms at 0.28 ms/row) give the next block's
head the 125B's state at its first 8 drafted positions; ceiling = the true-state chain (20.77 vs 13.77); Sandbox98 screen
today (PASS G(1-8) >= 0.30 HE, tree-32 kept >= +5 %); capture request by 18:00 (placeholder rows' W + top-8 + experts).
Then PREFIX-DEPTH INPUTS (EAGLE-3 fusion + GLIDE K/V of the prefix; free; tonight's capture trains it).
