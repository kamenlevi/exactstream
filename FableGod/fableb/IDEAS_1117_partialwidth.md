# IDEAS_1117: partial-width 125B rows at the drafted positions (Fable B, Tue 17 Nov 07:3x-08:3x)

Lens (BREAK98): dense + attention only, shared or top-1 experts, union-constrained routing. What fraction of the true
state does each recover, at what bytes? Labels [K] measured / [P] derived / [G] estimate (halved). Sources: the 125B's
config (BigMoE cfg), check25 README (dense 3.44 GB per check, 2.784 MB per 4-bit expert, 280 GB/s), Lunatic's
drift_* outputs, THINK_1116c / BREAK98 facts, the 2B sandbox records (S5 INJECT, S12 tapmlp, fork re-anchor).

## 0. The arithmetic that decides the lens: a per-token channel buys ~ONE layer of the 125B

The information the brief wants is the 125B's computation at a drafted position, which exists only after that token
is drafted: it is sequential per token, so every weight it touches is read once PER TOKEN. The channel budget is "a
few ms per block": take 3 ms per 30 tokens = 100 us per token = **28 MB per token at 280 GB/s** (less in practice:
each step is a dependent chain of dispatches, ~20-40 us of latency each).

What the 125B's own pieces weigh per row, from the config [K] (hidden 2560; GDN qkv|z 16384 x 2560, out 6144 x 2560;
attention q 12288 x 2560 at every 4th layer; experts 3 x 2560 x 640 = 2.784 MB at 4-bit, 10 routed + 1 shared per row;
HC low-rank 320):

| width of the row (per token, 4-bit) | bytes per token | ms per 30-token block [P] | x over the 28 MB budget |
|---|---:|---:|---:|
| full row (dense 3.44 GB incl. embeddings/head + 10 experts x 48) | 4.8 GB | 514 | ~170x |
| dense + attention + shared expert, 48 layers (no routed experts) | ~1.6 GB (layer dense ~1.45 + shared 0.13) | 171 | ~57x |
| dense + shared + TOP-1 routed, 48 layers | ~1.75 GB | 187 | ~62x |
| union-constrained routing (the previous check's resident union) | the same bytes: resident in DRAM, not in cache | | |
| dense + shared, layers 0-7 only | ~0.27 GB | 29 | ~10x |
| ONE layer, exact (dense ~30 MB + shared 2.8 + 10 routed 27.8) | ~60 MB | 6.4 | ~2x |
| the lm_head alone (248,320 x 2560) | 318 MB | 34 | ~11x |

- **Readings.** No partial-WIDTH form of a sequential 125B row fits the channel. Width cuts bytes linearly; the budget
  is ~1 % of the row, i.e. about one layer. And a layer's input is the previous layer's output at that position, so
  the only exact one-layer form is layer 0 (its input is the embedding): 6.4 ms per block per layer, worth nothing
  for drift (the re-plan lives deep; LAYERS: "the state lives near L45"; the 2B's useful tap was layer 12 of 24).
- Union-constrained routing changes WHICH experts, not the bytes: the union is resident in DRAM (~4.6 GB), not in
  any cache; a per-token read of 10 resident experts is still 27.8 MB per layer.
- **What a partial row would recover, if it were free (information side):** the bar is low. The carried state's
  direction error to the true state is 0.85-0.97 at depths 1-8 (SIB, HE) and the window mean of the true W already
  sits at 0.87; a partial row with direction error < ~0.8 would beat the drafter's own state. Top-1-of-10 routing
  keeps the largest routing weight (with 10 experts renormalised the top-1 share is typically 0.25-0.4 of the mass
  [G]); dropping the other nine at every layer compounds over 48 layers. I have no record of the compounded error; the
  capture ask in section 3 measures it for the record. It does not change the verdict: the bytes kill the channel by
  10-60x before quality enters.

## 1. The batching trap (why "batched partial rows before the check" cannot be route 1)

The only way to read weights once for many positions is to batch positions, and the only positions available before
the check are a SPECULATED sequence (the drafter's own 30 tokens). A batched partial pass over them gives true-ish
states only along the correct prefix. Used to re-draft, it repairs the first miss (the prefix before it is correct, so
the state there is true) and then drifts again from that point: that is exactly FORK RE-ANCHOR, measured by Lunatic
on the real dumps at +0.33 kept HE+ (+2.4 %; 13.77 -> 14.10), with the full 125B state. Iterating it (re-draft, pass,
re-draft) converges to the true-state chain (20.77) over ~5-7 rounds, each >= one partial pass (dense-only 8 layers
~2 ms + draft 3 ms): 25-35 ms per block, more than the whole cycle budget. **KILL.** The check already is the batched
pass over the speculated sequence; a second one in front of it duplicates it at partial width.

## 2. Where partial-width rows could pay: NOT the drafter's channel, but the check's extra rows (route 2, recorded)

- Rows 33-64 of a tree check with their routing constrained to the union the first 32 rows load: 0 extra expert
  bytes, 0 extra dense bytes; cost = compute and launches only. The union law (WEEK98: ~1.2 new experts per layer per
  marginal row at K96-128) says a constrained row keeps ~9 of its 10 experts per layer, so its logits are a lossy
  version of the 125B's (EvalSuite-gated, like the 1c tiers that already drop 61.9 % of the union at 16 rows).
- Price [P]: the dense matrix floor at 32 rows is 1.14x the byte floor (7.9 vs 6.9 ms); at 64 rows it is 2.28x: +7.9
  ms, plus the GDN step (107 us per layer per 32 rows: +3.9 ms), attention and HC reads (+1-2), expert compute (ooo1cq
  is compute-bound per MoonShots: +2-4). **+15-18 ms for rows 33-64**, against Lunatic's line (rows 33-64 <= 3.7 ms in
  all for K64 to pay). KILL at the desk. Only early death (Wed) revives it: if ~70 % of the rows die by layer 8, the
  compute cost drops below the line, but then plain rows die too and the constraint buys only the expert bytes early
  death did not already save. This is the orchestrator's route 2 (speed only), not route 1.

## 3. The capture ask (by 18:00; cheap for BigMoE, batched along the true text; settles the information side)

For the record and for FAR / Lunatic's lanes (a drafter that consumes approximate 125B information needs to know
how much a top-1 or shared-only row carries), add to tonight's capture, on the same rows as mid_{heplus24,mbppbank}:
- **Ablated forwards**, teacher-forced along the true text, dumping the final widened state (10,240) and the top-8
  logits per row, for routing = {shared only; shared + top-1; shared + top-2; shared + top-4}; and one more with
  routing constrained to the experts of a 32-row window (the row's own window) as the engine's union would be.
  Each is one batched pass per set (the cost of a prefill, minutes), ~1.35 GB fp16 per arm as the mid capture.
- Read-outs (CPU, 10 min, drift_rankceil's harness): direction error of the ablated final state vs the true one;
  the head's walk-strict accuracy at depths 1-30 when fed the ablated state instead of the true one (the LAB harness
  with a different state); top-1 agreement of the ablated logits with the 125B's.
- Lines: a form is "useful information" if the head holds >= 95 % walk-strict at depth 30 from it (vs 97.8 true,
  89.65 own); it is "worthless" below 92 %. Whatever the result, it does not revive a per-token channel (section 0);
  it prices what a distilled or partial proxy must reproduce.
If BigMoE can only afford one arm: shared + top-1.

## 4. Killed at the desk, so nobody re-proposes them

- Sequential partial-width rows per drafted token (any width >= one layer): bytes, 2-170x over the budget (section 0).
- Low-rank slices of the dense weights per token (rank-128 qkv|z = 1.2 MB per layer, 44 MB per token): fits the
  byte budget only by becoming a trained approximation whose INPUT (the layer's residual at that position) still
  needs the full stack below it. Circular; it is FAR's distilled proxy, not a 125B row.
- Batched partial pass over the speculated block before the check: fork re-anchor's +2.4 % per round (section 1).
- Routers as a per-token signal (48 x 512 x 2560 at 8-bit = 63 MB per token): the routing at layer L needs the
  layer-L residual at that position. Circular.
- Pipelining a partial row under the drafter's own step: bytes per token unchanged.

## 5. Bottom line

1. A per-token channel at "a few ms per block" buys ~28 MB per token, about ONE layer of the 125B. Every partial-width
   form of a sequential 125B row (dense-only, shared-only, top-1, union-constrained) is 10-170x over it. [P from the
   config and 280 GB/s]
2. Batching across positions needs speculation, and speculation only informs the correct prefix: the measured value
   of that is fork re-anchor's +2.4 % kept per round. KILL.
3. So partial-width rows cannot be route 1. The information they would carry is cheap to measure for the record
   (section 3: ablated forwards on the true text, shared + top-1 first) and prices what FAR's proxy must reproduce.
4. Union-constrained rows 33-64 inside the check are compute-bound at +15-18 ms [P] vs the 3.7 ms line: route-2
   material only if early death kills most rows by layer 8.
5. Route 1 at this budget needs a source of the deep re-plan that is NOT a 125B row per token: the drafter's own
   stack must reproduce it (FAR / Lunatic), or the check must get cheaper per row so that more rows pay (route 2).
