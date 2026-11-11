# Far-ahead expert foresight for an SSD-streamed MoE: what a far draft can and cannot predict (CPU replay measurements)
- Date: 2026-11-04 22:10 EET. Sessions: MoonShots (M-38 replay frame), Fable FAR (measurements), orchestrator (this note).
- Question: a drafter runs far ahead (8 checks, ~100 tokens). Can its continuation forecast which cached experts get
  used again soon, so the expert cache keeps the right ones (approximately Belady-optimal)?
- Frame [K, M-38, ~/MoonShots commit 99df774]: 45 GB cache (K 266), 16-row speculative checks, horizon 8 checks.
  - LRU 42.9 misses per kept token; OPT 19.2.
  - A forecast with recall 0.92 / precision 0.72 on "resident reused within 8 checks" would cut misses 26.5 %.
  - Recency alone: 0.83 / 0.52. The best learned eviction key (M-14): -7.8 %.
- Numbers [K, CPU replays, ~/FableGod/far/]:
  - Token path, KILL. The TRUE future tokens through their EXACT routes reach only recall 0.68 / precision 1.00, -20.3 %
    misses (K 147, the 24 GB cache: 0.76, -26.3 %).
    - The drafter's rejected rows hold 32 % of the cache reuse, and no true-path forecast names them.
    - Honest token -> expert maps (this prompt's memo + a leave-one-prompt-out table): 0.53 / 0.66, -6.0 %.
    - Deep-layer routes are mostly context: per-row recall@10 from the token alone is 0.40 below layer 8.
  - State path, BETWEEN. A leave-one-prompt-out ridge map from a shallow hidden state to routes (every future row) reaches
    0.70 / 0.64: -14.6 % misses at K 266, -11.9 % at K 147 (235 MB per kept token). Nearest-neighbour states: -7.1 %.
  - Misses are not stalls. Perfect 8-check foresight cuts misses 34 % but layers-with-a-miss only 5 % (42.9 -> 40.6 of 48).
    Where the SSD wait is per-layer round trips, foresight buys bytes, not tokens per second.
  - Combination: foresight + the learned key on the unforecast residents is additive with a precise forecast (perfect
    kept-row routes -20.3 -> -26.9 %), and adds nothing with the ridge stand-in.
- Files: ~/FableGod/far/FAR.md (sections 1-6), far_ceiling.txt, far_state.txt, far_ridge.txt, far_combo.txt and their scripts.
- Prior art:
  - The far draft as an expert forecaster is published: MoE-SpeQ, arXiv 2511.14102 (<= 32 tokens ahead).
  - Draft-state pseudo-routers are adapted from SP-MoE (arXiv 2510.10302).
  - The three measurements above appear new: the rejected-row share of reuse, the token-vs-state ceilings, and misses vs stalls.
- Status: CPU stand-ins only; EXACT-safe (it changes only what the cache keeps).
- Next: GPU capture FAR-G1, ~4 min. Run the MTP chain 64 steps ahead per check and the 125B's routers on those tokens.
  - PASS: >= 20 % fewer misses than LRU at K 266 with set recall >= 0.80. Then build.
  - BETWEEN: 10-20 %. Keep it as a 24 / 12 GB bytes lever only.
  - KILL: < 10 %, or no better than the learned key.
- Update 22:12 (FAR section 7, far_wait.txt):
  - The SSD wait per check follows misses: wait ~ 10-16 ms floor + 0.15-0.17 ms per miss (R2 0.86-1.00) on BigMoE's 106
    per-check logs, EXACT and FAST alike. The per-miss term is 2.16 MB / 0.155 ms ~ 14 GB/s, the SSD's streaming rate.
    The floor is ~48 layers x ~0.25 ms of first-read latency. Exceptions: FAST + QOM tags (optional misses break the line).
  - So above the floor the wait is bandwidth-bound, not round-trip-bound. Every removed miss saves its bytes' time:
    - at EXACT C 160 (483 misses per check), -26.5 % misses = -20 ms of a ~138 ms check (+14.5 %);
    - at today's live 100-200 misses per check, -4 to -8 ms (+6-9 %).
  - EXACT's demand reads sit on the critical path in full (service time = io + 0.3-2 ms).
  - FAR-G1 origins: every 4th check with >= 96 far tokens reproduces the every-check ceiling.
- Idea added 22:16: the best look-ahead distance may not be ~100 tokens.
  - FAR-G1 captures up to 192 far tokens per origin. The read-out scores horizons 16-192 against eviction look-ahead
    H 4 / 8 / 16, per layer band and domain, at K 266 and K 147.
  - It also tests a confidence-adaptive horizon (SPICE-style, published) and a per-layer horizon (possibly new).
  - With perfect knowledge the gain saturates by ~96 far tokens at K 266 (F 64: -18.9 %; F 96 = F 128: -20.3 %).
    A real draft decays with distance, so the real optimum is likely shorter. Untested.
- Idea added 22:18: two forecasters together, near (next 1-2 checks, precise) + far (8+ checks, broad), like a hybrid branch
  predictor. Three combinations: tiered eviction, a per-layer chooser, a distance-weighted score.
  - Prior art: hybrid / tournament branch predictors (McFarling 1993) and multi-predictor cache replacement in CPUs.
    MoE-SpeQ's hierarchical cache is the nearest MoE work.
  - CPU test queued in FAR's frame (FAR.md section 8).
- Idea added 22:21: a fully dynamic horizon, scored from the same capture.
  - Hard per-token stop: stop at the first far step with p_top1 < tau (SPICE-style, published).
  - Soft per-token weighting: each far token's experts weighted by the chain's cumulative confidence, with no cutoff
    (possibly new).
  - An acceptance-rate horizon (F ~ 8 checks' worth of kept tokens: code ~90-130, math ~50-70) and per-layer horizons.
- Result 22:24 (FAR section 8, far_hybrid.txt, CPU 39 s): the near + far hybrid reaches the PASS line that neither part
  reaches alone.
  - Tiered eviction: near (2 checks) > far (8 checks) > LRU.
    - K 266 (45 GB): -20.2 % misses vs LRU. Far alone -14.6 %, near alone -11.1 %.
    - K 147 (24 GB): -22.3 %. Far alone -11.0 %, near alone -17.8 %.
    - With M-14's learned key among the rest: -22.0 % / -22.7 %.
  - A per-layer chooser adds nothing (it swaps tiers on noise). A distance-weighted score reproduces the tier.
  - The near horizon must be 2 checks (1 check: -17.6 %).
  - Caveat: the near forecast is a noise model at V1's recall (0.75, independent errors, optimistic). FAR-G1's GPU capture
    scores it directly: tiered hybrid >= -20 % at K 266 PASS, 10-20 % BETWEEN, < 10 % KILL.
  - Cost: none extra (same chain, same map, a two-level eviction key). EXACT-safe.
- Prior art, refreshed 22:26 (standard web search; abstracts and summaries only):
  - Published:
    - draft-based expert forecasting one draft ahead: MoE-SpeQ 2511.14102, SP-MoE 2510.10302, MoE-SpAc (a short-horizon
      demand sensor for prefetch and eviction);
    - a confidence-adaptive horizon with a hard stop: SPICE 2608.21240, which argues fixed horizons pollute the cache;
    - multi-signal / multi-level schemes: FineMoE / fMoE 2502.05370 (prompt semantic hints + layer-trajectory matching),
      HOBBIT 2411.01433 (token / layer / sequence levels);
    - Least-Stale eviction: SpecMD 2602.03921.
  - Not found: a near-draft tier + a far-draft tier (~100+ tokens) in one tiered eviction; soft per-token confidence
    weighting without a cutoff; FAR's measurements. Tag: adapted. The new parts are the far horizon, the near/far tiering,
    the soft weighting and the measurements.
- Correction 22:30 (MoonShots' novelty check, research/FAR_NOVELTY_CHECK.md, 27ff044):
  - Soft per-token confidence weighting is published in its near form: PSR section 7 (2610.34721), AcceptMoE (2608.02989)
    and MoE-SpeQ's confidence buffer. Only the far-chain extension is ours.
  - The near / far tiering and the ~100-token draft horizon are "adapted" (not found in this form).
  - The new parts are the four measurements. Clean claim note: 2026-11-04_2240_near-far-expert-forecasting.md.
