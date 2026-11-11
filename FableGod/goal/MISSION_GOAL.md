# MISSION GOAL: where is the missing 2-3x?

You get the problem, the goals and today's MEASURED facts, but not the list of methods tried before. Don't read any file
outside ~/FableGod/goal/ (that keeps your view independent). Use the web for published methods.

## The problem
- **Model:** a 125B MoE.
  - 48 layers: 36 gated-delta-net linear attention + 12 full attention (24 heads, 2 KV). Hidden 2,560.
  - 512 experts, top-10 (width 640, SwiGLU) + 1 shared expert. Vocabulary 248,320. One MTP layer.
  - Experts quantized to ~3-4 bits (~2-3 MB each).
- **Machine:** MacBook Pro M5 Pro, 64 GB. GPU memory streaming ~286 GB/s (measured). SSD ~12-13 GB/s.
- **Lines:**
  - 45 GB RAM (main): EXACT = bit-identical to token-by-token decoding; FAST = same benchmark scores.
  - 24 GB and 12 GB (small-RAM).
- **Goals:**
  - 300 tok/s on code in FAST and 200 in EXACT at 45 GB;
  - <= 10 MB of SSD reads per kept token at 24 GB;
  - eventually ~120-150 tok/s in <= 12 GB.

## Measured today [K, from the team's runs]
- **Generation:** speculative decoding. A small drafter (~0.5B) proposes 16 tokens; the 125B verifies all 16 in one
  pass (a "check"), and the agreeing prefix is kept.
  - Code: ~10.5 tokens kept per check (~92-94 % per-token agreement). Chat: ~3.
  - A perfect pick among the drafter's own top candidates would reach only ~12.4.
- **Speed today, code:**
  - FAST: ~110 tok/s at 45 GB.
  - EXACT: ~83 tok/s at 45 GB (51 on a 14-prompt mixed set).
  - EXACT 24 GB: ~33 tok/s (mixed).
  - EXACT 12 GB: ~41 tok/s on code.
- **One FAST check on code at 45 GB:** ~90 ms. Drafter ~10 ms, now cut to ~5.4. Waiting on SSD expert reads ~16-60 ms
  (typically ~24). The 125B's compute ~50 ms. Commit ~5 ms.
- **Expert kernel:** at 16 rows it runs at 77 % of the streaming rate exact, ~82 % FAST. Twelve redesigns failed. The
  limit is compute per byte: per expert only ~2 of the 16 rows are real, but the matrix unit computes all 16.
- **Dense part:** already 4-bit, ~74 % of its byte floor.
- **SSD bytes per kept token, code:**
  - 45 GB FAST: ~15 MB.
  - 24 GB FAST: ~19.5 MB [P]. That stack skips sure rows' missing experts, gates the last row with a 17th draft, uses a
    3-bit tail and 11-bit scales.
  - A 2-bit trellis-coded copy of the experts might reach ~4-7 MB [P].
- **Chat at 24 GB:** no FAST setup keeps quality except a milder skip rule at ~60-90 MB per kept token.
- **Relaxed acceptance** (keep a draft the 125B rates nearly as likely as its top choice): kept 10.5 -> 11.7 at 1.2 %
  changed tokens, 12.9 at 2.4 %.
- **Drafter training:** about +0.13 kept per doubling of data; a 2-4B drafter is estimated at ~12-12.5 kept.

## Deliver (by ~06:30; one file: ~/FableGod/goal/GOAL.md)
1. Where the missing 2-3x most likely hides, ranked by size, with the arithmetic.
2. Your top 3 levers, each with:
   - the mechanism;
   - the expected gain;
   - EXACT or FAST;
   - published or new (link the closest prior art);
   - the cheapest test (<= 10 min) with a pre-written PASS / KILL line.
3. Your one bet.

## Rules
- Thinking and writing only: no GPU, no downloads, no files outside your folder.
- Marks [K] / [P] / [G].
- Stop when complete or at 06:30. When done, send the orchestrator one message: your best lever in one line + the path.
