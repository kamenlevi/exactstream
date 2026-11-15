# TREE16: a 16-node tree check vs a 16-row chain check on the engine (Leap, desk pricing, Sun 15 Nov 10:3x EET)

**Ask.** The orchestrator, 10:3x. MTP Head's S9 TREE-DYN (TrainLab d0aafb2, data/sb98/tree_price.txt) leaves one exact
lead: an EAGLE-2-style 16-node tree at chain-16's row count. Carried from s30 to the 125B, it is worth ~+17 % kept
(MBPP) / ~+5 % (HE+). It holds only if a tree check costs about chain-16's 61-64 ms. Estimate first, no build.

**Short answer.**
- A tree check costs about a chain check in everything except exact attention.
  - FAST / d1: about chain-16 + 0.3-1 ms (~+1 %).
  - Exact (bitwise vs REF2-T): about chain-16 + 0.5-1.5 ms, but only with an attention build or a new reference.
  - Exact with no new kernel: +3-6 ms at 1-2K context, +10-20 ms at 8K.
- Close enough that BigMoE's top-3 dump is worth running.

All numbers are estimates [est] unless marked measured [M], with the source given.

## What mixes rows in this model
Everything else is row-local:
- dense projections, the experts, router, shared expert;
- HC reads and writes, norms, head.

Row-local parts see 16 rows in a tree just as in a chain. The experts stay one call over the 16 nodes' union.
Siblings share a prefix, so the union should be no larger than a chain's.

Four things mix rows:
- **36 GDN layers:** the causal conv (kernel 4) and the recurrent state.
- **12 QSA layers:**
  - attention over the cache plus earlier rows;
  - above the indexer's 2,048-token budget, the block selection. It runs on pooled 4-token blocks, including blocks
    completed inside the window.
- **The PLE layer (layer 2):** hashed 3-gram ids and a depthwise conv (kernel 4, dilation 3, so 9 rows back).
- **RoPE:** a node's position is its depth.

## Per part
Chain-16 is 61-64 ms on the engine [M, G32T within-run].

| part | chain-16 today | tree-16, exact | how |
|---|---|---|---|
| row-local (~85 %) | ~50-55 ms | same | 16 rows; one expert call over the union |
| GDN body, 36 layers | ~2.4-2.9 ms (68-80 us / layer [M], TABLE_350 GDNB) | +0.1-0.5 ms | DFS walk in the one-dispatch body (below) |
| (alternative: GDN paths batched) | | +0.5-1.5 ms | P x 48 threadgroups; shared prefixes recomputed |
| PLE conv + 3-grams, RoPE | ~0 | ~0 | ancestor gathers; positions = depth |
| attention, 12 layers | ~7 ms (~0.6 ms / layer [M], R-72) | FAST: +0; exact: see below | |
| commit | ~1-2 ms | +0.1-0.3 ms, only when acceptance leaves the main chain | gathers of the accepted path's rows |

**GDN (exact by construction).**
- Today's body (scripts/gdnbody.py) runs one threadgroup per value head (48 x 256 threads). The fp32 state sits in
  registers, and mlx_lm's loop body steps through the rows in order.
- The tree version walks the nodes depth-first:
  - at a node with more than one child, save the state (64 KB per head; 3.1 MB per layer);
  - restore it before the next sibling.
- Phase A reads each node's conv window from an ancestor table.
- Each node then runs the same arithmetic on the same values as row d of its path's chain check.
- Cost: about 1-3 us per fork x 3-5 forks per layer [est].
- The batch-of-paths form reuses the kernel with a path grid. It is simpler to build, but it recomputes shared
  prefixes on P x 48 threadgroups.

## Exact attention: why a plain tree mask is not bitwise
- MLX's SDPA at head dim 256 runs unfused: matmul, then softmax, then matmul, in MTPLX's 12 head chunks (TABLE_350
  GDNB-2: 144 calls per check, 21.9 us each [M]).
- The softmax's sum and the P·V product group the key columns in a fixed way.
- Lay out the main chain first and every main-chain node sees its keys in the same columns as its AR call. Those
  nodes are exact with a plain mask.
- A side-branch node's ancestors and the node itself sit in other columns, with masked keys between. Its sums group
  differently. That gives last-bit differences and rare token flips: not exact.

**Exact routes:**
- **(a) Paths batched, no new kernel.**
  - Each side path is laid out as a chain: K / V = the cache + that path's tail, padded to the chain's 16-row shape.
  - All paths run in one batched call per layer. Each batch element is literally a chain check, which gives exactness
    if MLX's batched matmul and softmax keep each element's arithmetic (a 1-minute bit test).
  - Cost: one K / V copy per extra path per layer (2 KB per token), plus P times the SDPA work. That is +3-6 ms at
    1-2K context (+5-9 %), +10-20 ms at 8K, for P ~ 4-6 paths [est].
- **(b) Keep MLX's scores and softmax; port only the second matmul.**
  - Scores are exact per element.
  - Permute each node's tail columns into its chain order. The softmax then sees exactly its chain row (same N class).
  - Do P·V with our own bitwise port of MLX's GEMM, reading a per-row V tail.
  - Cost: +0.2-0.5 ms per check. Build: ~1 day plus bit tests on captured inputs.
- **(c) A new exact reference.**
  - An attention whose row sums run over the row's visible keys in logical order (per-row key lists). Then tree ==
    chain == AR by construction, at ~+0, and possibly faster than today's 12 chunks.
  - MTPLX's compiled QSA rows-gather lane may already be such a kernel (unchecked).
  - It changes REF2-T's bits; SD == AR still holds bitwise. Adopting it is Kamen's call.
- **Above 2K context:** each node's block selection also needs its path's pooled window blocks (<= 4 per path) and a
  per-node score row. Small.

## Totals and economics
- FAST / d1: tree-16 is about chain-16 + 0.3-1 ms.
- Exact with (b) or (c): about chain-16 + 0.5-1.5 ms (~62-65 ms, +1-2 %).
- Exact with (a): +3-6 ms at 1-2K context, more at long contexts.
- Net, against S9's EAGLE-2-16 estimate (+17 % MBPP / +5 % HE+ kept):
  - (b) or (c): ~+15 % / ~+3-4 %;
  - (a) at 1-2K: ~+7-11 % / ~-4 to 0 %.

## Drafting side (pillar 3)
- EAGLE-2 expands each depth's frontier as a small batch.
- The MTP step is launch-bound (P2: ~0.54 ms fixed), so a 2-4-row step costs about one row's.
- With ~11-12 depths instead of 15, tree drafting costs about chain-16's or a bit less [est].
- The rerank is trivial CPU work.

## If built (all exact by construction)
1. A DFS variant of gdnbody: node order, a parent table and fork slots; phase A from an ancestor table.
   Bit test: every node == the chain check of its path.
2. Attention route (b), or (c) on Kamen's word.
3. PLE, RoPE and indexer per path, plus the commit's gathers.
4. A C3-style test on the 4-layer slice:
   - T1: every node bitwise == its path's chain check;
   - T2: commit any accepted path, then a probe check.

Effort: ~1.5-2 days with (b). (a) is ~0.5 day (no kernel) but costs more per check.
