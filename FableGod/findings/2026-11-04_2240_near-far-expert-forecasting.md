# Near + far draft forecasting for MoE expert caching on an SSD-streamed 125B: dated claim note
- Date: 2026-11-04, 22:40 EET.
- Idea: Kamen (project owner). Measurements: Fable FAR (CPU replays) on MoonShots' replay frame. Novelty check:
  MoonShots ([../../MoonShots/research/FAR_NOVELTY_CHECK.md](../../MoonShots/research/FAR_NOVELTY_CHECK.md), commit 27ff044).
- Lab log with every step and number: [2026-11-04_2210_far-ahead-foresight.md](2026-11-04_2210_far-ahead-foresight.md).
  Full method, scripts and raw outputs: [../far/FAR.md](../far/FAR.md).

## Setting
- Model: Qwen3.8-Flash-Next, a 125B MoE (512 experts per layer, top-10, 48 layers).
- Machine: one MacBook Pro M5 Pro, 64 GB. Experts that are not in RAM stream from the internal SSD.
- Decoding: speculative, with 16-row checks (a drafter proposes 16 tokens, the 125B checks them in one pass).
- Cache: K expert slots per layer. K 266 is the 45 GB line, K 147 the 24 GB line.

## New measurements (not found in published work)
1. **Rejected rows matter.** The drafter's rejected rows hold 32 % of the expert-cache reuse. A forecast of the true path
   alone tops out at recall 0.68 (-20.3 % misses), even with perfect tokens and exact routes.
2. **States, not tokens.** Token -> expert maps fail for deep layers: per-row recall@10 from the token alone is 0.40 below
   layer 8. State-based maps reach recall 0.70 / precision 0.64 (-14.6 % misses).
3. **The SSD wait law.** The per-check SSD wait is about a 10-16 ms floor + 0.155 ms per missed expert (~14 GB/s, the
   SSD's streaming rate; R2 >= 0.95 over 106 logged runs). Above the floor, every avoided miss saves its bytes' time.
4. **Misses are not stalls.** Perfect 8-check foresight cuts misses 34 % but layers-with-a-miss only 5 %.

## Adapted designs (this exact form not found; close relatives are published)
1. **A two-tier draft forecaster for expert-cache eviction.**
   - Near tier: the experts the next 2 checks need. Far tier: experts predicted from a draft chain running ~100+ tokens
     (8 checks) ahead, through a state -> route map. Eviction priority: near > far > LRU.
   - CPU replay on recorded traces:

     | cache | hybrid | far alone | near alone |
     |---|---|---|---|
     | K 266 (45 GB) | **-20.2 %** misses vs LRU | -14.6 % | -11.1 % |
     | K 147 (24 GB) | **-22.3 %** | -11.0 % | -17.8 % |

   - Caveat: the near tier is a noise model at the engine's look-ahead recall. GPU validation is pending.
   - Nearest relatives: fMoE's two-signal eviction score (not split by horizon); MoE-SpAc / MoE-SpeQ (one draft window
     each); SeqMoE. The combiner is a CPU-style hybrid branch predictor (McFarling, 1993).
2. **A ~100-token draft horizon.** Not found for draft-fed forecasts: MoE-SpeQ looks <= 32 tokens ahead, and ExpertFlow /
   DoMoE count their horizon in layers. Long-horizon forecasts that are not draft-fed are published (MoE-Infinity, HOBBIT,
   fMoE).
3. **Soft per-token confidence weighting.** Published in its near form: PSR section 7 (arXiv 2610.34721, a running product
   of drafter confidences), AcceptMoE (2608.02989) and MoE-SpeQ's per-token confidence buffer. We only extend it to a
   ~100-token far chain. Designed, not yet scored.

## Prior art (links)
- Draft-based expert prediction:
  - MoE-SpeQ, https://arxiv.org/abs/2511.14102;
  - SP-MoE, https://arxiv.org/abs/2510.10302.
- A confidence-adaptive look-ahead: SPICE, https://arxiv.org/abs/2608.21240.
- Multi-signal and multi-level expert prefetch / caching:
  - FineMoE / fMoE, https://arxiv.org/abs/2502.05370;
  - HOBBIT, https://arxiv.org/abs/2411.01433;
  - MoE-Infinity.
- Least-Stale eviction: SpecMD, https://arxiv.org/abs/2602.03921.
- Confidence weighting: PSR, https://arxiv.org/abs/2610.34721; AcceptMoE, https://arxiv.org/abs/2608.02989.

## Status and next test
- Everything above is CPU replays on recorded traces of the real engine.
- Next: GPU capture FAR-G1 (scheduled Thu 5 Nov, night slot). The drafter's MTP chain runs 192 tokens ahead every 8th
  check, and the 125B's routers run on the far tokens.
  - Lines written before the run: PASS >= 20 % fewer misses than LRU at K 266 (also for the tiered hybrid);
    BETWEEN 10-20 %; KILL < 10 %.
  - The capture also scores every look-ahead distance from 16 to 192 tokens, plus the adaptive and soft per-token rules.
