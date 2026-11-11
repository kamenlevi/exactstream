# MISSION 20: THE CACHE THAT KNOWS THE PROMPT. Measure it tonight, on the CPU.

(Draft by the orchestrator for Kamen, Mon 2 Nov 2026 ~18:0x.)

**Kamen's brief.** Research the cache, with tests that run right now and come back fast. Everything here is a CPU replay on routing we already recorded: no GPU, no waiting for other runs.

**What we know**
- **Your safari** (SAFARI.md H4): on code, the prompt's experts cover 91.1 % of the answer's rank-0/1 slots and 84.0 % of ranks 2-9. The prompt plus the first 100 answer rows cover 95.3 / 91.5 % of the rest. On chat the prompt covers less (70-72 %) and the first 100 rows more (93 %).
- **Cloud Fable's 12 GB design:**
  - a two-pass check: pass 1 with the drafter's tiers (top-2 / top-5), certify rows whose own p1 >= 0.9, then exact top-10 only for uncertified rows up to the first contradiction
  - plus a pinned hot set
  - its claim: 70-80 tok/s on code at 12 GB if the oracle's low cache churn ("few exact rows, so the cache stops churning") holds for this realizable design; near 60 if pass 2 lands above ~1.8 GB per check
- **Your FLOOR replay** (floor_bytes.py section 2): LRU per layer at C slots for the 12 / 24 / 45 / 51 GB lines. Its absolute misses run 1.25-1.4x above the engine's measured ones, but its ratios hold.
- **A6:** the cold-expert burst is 58-73 % of the first cycle's excess after a long prompt.

**Deliver** (write CACHE.md as results come in, so a partial answer is already useful)
1. **Policies against LRU** at C 51 / 160 / 256 / 320 (12 / 24 / 45 / ~51 GB), as misses per kept token on code, math and chat:
   - (a) LRU, the baseline
   - (b) LRU + prompt protection: decaying priority over the first N = 10 / 30 / 100 checks
   - (c) LRU + a pinned hot set: the N most-used experts per layer, from the prompt or from a global prior
   - (d) (b) + (c)
   - (e) Belady's MIN, the ceiling
2. **The start-up:** misses in the first 1-3 checks after the prompt, with and without pre-loading the prompt's experts at the end of the prefill.
3. **The two-pass check** at 12 and 24 GB, with pass 1 and pass 2 as above (per-row certificates from floor_rows.json):
   - SSD bytes per check and per kept token, under LRU and under the best policy from 1
   - does the oracle's lower churn hold for this realizable design? That is cloud Fable's Test 1: above 1.8 GB per check, 12 GB tops out near 60.
4. **Tok/s at each line,** priced from the measured miss ratios (conservatively), and one engine test for BigMoE: a flag, exact, with token equality against the reference. Name it with PASS / KILL lines.

**Speed**
- Reuse floor_bytes.py's replay and decision_map/out/routing (36 windows, 13 MB). Keep each run to a few minutes, and aim to finish within ~45 minutes.

**Rules**
- Cache policies never change a token, so they are exact. The two-pass gate is quality-defined: mark it.
- Check ~/TRIED/TRIED.md: LRU vs LFU (KILL), pinning the hottest experts permanently (KILL), Belady-trained / popularity-prior eviction (BETWEEN), context matching (KILL), SEED45's popularity prior (BETWEEN), A6. Say why your policy differs from each.
- CPU only, no downloads, writes only in ~/FableGod. Mark every number as measured / arithmetic / judgement.

**Inputs**
- ~/FableGod/floor_bytes.py / .out, floor_rows.py / floor_rows.json, decision_map/out/routing, safari_routing.py / .out, SAFARI.md, FLOOR_125B.md, DAY9_2026-11-02.md
- ~/BigMoE/scripts/seed45_replay.py, replay_bench.py; ~/MoonShots/scripts/cachesize45.py; ~/BigMoE/data/a6
- ~/TRIED/TRIED.md
