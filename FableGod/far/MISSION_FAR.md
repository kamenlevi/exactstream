# MISSION FAR: make the far-ahead draft pay (Wed 4 Nov ~21:4x -> ~23:30)

> (a quote from the project owner, kept private)

**His idea** (Wed ~20:4x) comes from an index pre-pass in an image viewer. One cheap pass runs far ahead and prepares what the slow pass will need. Here: a drafter runs ~50-200 tokens ahead and tells the expert cache which experts the next checks will use.

## Where it stands [K]
- **M-38 FORESIGHT PASS** (MoonShots, commit 99df774, `~/MoonShots/data/foresight45.txt`; the replay code and trace paths are in that commit). The question is "is this cached expert used again within the next 8 checks?":
  - At the 45 GB cache, a forecast at recall ~0.92 / precision ~0.72 cuts misses 26.5 % vs LRU (31.5 vs 42.9 per kept token).
  - At the r 0.5 noise level: -19 %.
  - Recency alone: recall 0.83 / precision 0.52. The learned eviction key (M-14): -7.8 %.
- **Perfect-foresight curves:**
  - 24 GB (TRIED, W24-belady): 1 check ahead -10 %, 2 checks -17 %, 5 checks -29 %, MIN -41 % (vs LFU).
  - 12.7 GB (TRIED, short-horizon line): 1-2 passes -7-13 % of bytes, OPT -42 %; "the rest needs 8-16 passes of foresight".
- **The real drafter, one window ahead** (TRIED, Rnd17#2): -2 % misses, a fifth of the oracle's -10 % at 1 check.
- **Already in the engine or tried** (TRIED):
  - V1 look-ahead: layer L+1's router on layer L's input, recall@12 0.72-0.79.
  - The W1 prediction line: "verify misses are unpredicted, not late".
  - CORR.
  - M1a learned prefetch (S10c-1): -7.9 % bytes, tok/s flat, ~2 ms per token of f32 cost.
  - OOO and per-layer slot budgets.
- **Costs:**
  - The drafter takes ~5.4 ms per 16-token block today (Leap 17:0x), so running 100+ tokens ahead every check is ~6 blocks. Find where that time hides (e.g. EXACT's exposed SSD wait) or how one far draft serves several checks.
  - EXACT check at 45 GB: ~126 ms, of which the SSD wait is ~55-60 ms ([P], ~/FableGod/goal/GOAL.md).
- **Prior art:**
  - MoE-SpeQ (arXiv 2511.14102): a quantized draft, <= 32 tokens ahead, lookahead-aware eviction.
  - SP-MoE (2510.10302): one round ahead from draft states plus the target's gates, 89 % top-1.
  - SPICE (2608.21240): confidence-adaptive lookahead.
  - Read-ME (a pre-gating router; Belady with lookahead), MoE-Infinity (request-level traces), ProMoE (stride prefetch).
- **Where it pays:**
  - EXACT at 45 GB, and bytes per kept token at 24 / 12 GB (BYTES10).
  - FAST at 45 GB may go SSD-free through the trellis store (Leap's TRL-NAX kernel 0.87x of NAX6-N; quality screen 7 tonight), so don't optimise FAST at 45 GB.

## The target
One forecaster that:
- reaches recall >= 0.92 AND precision >= 0.72 on "cached expert reused within 8 checks" at the 45 GB cache (M-38's PASS point), then the same at 24 GB;
- costs <= ~1 ms per check of extra compute and no extra SSD reads;
- is EXACT-safe: it only steers what the cache keeps and the read order.

## Signals to consider (examples, not rules)
- The drafter run 50-200 tokens ahead: its tokens or hidden states, mapped to the 125B's 48 routers (a learned map like SP-MoE's, or a token -> expert table).
- The prompt's own expert footprint from prefill.
- Recency, frequency and reuse distance per (layer, expert).
- Cross-layer links: an expert used at layer L predicting layer L+k.
- The answer's structure: code repeats body patterns.

## Deliver (one file: ~/FableGod/far/FAR.md; written as you go; done by ~23:30)
1. **The ceiling first, on recorded data (CPU).** Treat the TRUE future tokens as a perfect far draft. Map them through the cheapest token -> expert map you can build from the traces: what recall / precision do you get at H 8? If even that misses 0.92 / 0.72, say which signal must carry the rest.
2. **Your best forecaster:** mechanism, features, cost per check, recall / precision on held-out traces (CPU), and the miss cut it buys in MoonShots' replay (reuse its code read-only).
3. **The GPU test (<= 10 min)** that measures it with the real far-ahead drafter, with PASS / KILL written now. The orchestrator routes it.
4. **Arithmetic:** EXACT tok/s at 45 GB, and MB per kept token at 24 GB, if it lands.
5. **Prior-art status:** published / adapted / new, with links.

## Rules
- **CPU-only tests are allowed:**
  - one process, <= 1 core, <= 4 GB, <= 10 min per run;
  - each run under `peer lock share 15m --cpu 100 --gpu 0 "FAR: <what>"`, released right after;
  - no run while `~/.local/share/quiet/PAUSE` exists.
- No GPU, no downloads, no edits outside ~/FableGod/far/. Read anything.
- Marks [K] / [P] / [G]; every number with its source.
- When done, or at 23:30, send the orchestrator one message: the best forecaster in one line, its recall / precision, and the path. Include any TRIED lines you want recorded.
