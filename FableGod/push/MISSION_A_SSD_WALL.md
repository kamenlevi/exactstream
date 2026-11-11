# MISSION PUSH-A (Fable A): THE SSD WALL

Make the SSD time per check vanish at 45 GB, then at 24 GB. This is part of the BREAKTHROUGH PUSH (~/FableGod/push/BREAKTHROUGH_PUSH.md): read that first.

## Why

At C 256 a 16-row check costs 96.8 ms [K]:
- draft: 12.0 ms, with the SSD idle;
- check: 84.2 ms, of which the SSD is busy 42.0 ms and the gates spin 42.6 ms;
- 171 expert misses per check.

The SSD is THE wall at 45 GB:
- ~0.5 GB of misses per check at ~12-14 GB/s [P];
- reads of >= 128 KB keep >= 90 % of peak [K].

350 tok/s at ~10 kept per check needs a cycle of ~29 ms [P].

## Read first

- Your SSD_WALL.md (mission 32 phase 1). This is phase 2, with today's data.
- MTP Head's REPORT 'ROUTEPRED' (~/TrainLab):
  - next-check expert prediction is dead (precision 0.07 at C 256);
  - inside a check, layer L+1's router on L's input names 50.5 % of the misses at 3.5 reads per miss;
  - draft-window ceiling ~x1.28 on short checks, ~x1.07 on long code checks;
  - full-hiding ceiling ~x1.5-1.8.
- Leap's PURGE-1 result: macOS reclaims volatile slabs before compression or swap.
- BigMoE's volatile expert tier (2a99aa7; smoke tonight).
- NewIdeas' RANKED row 33 (C 320 roughly halves code MB per kept token vs C 256).
- MoonShots' data/band45.txt.
- NewIdeas' union-stack prices (FAST_TRACK 2 / 2d).
- TRIED: the prefetch, cache, layout and compression lines.

## Deliver

Write ~/FableGod/push/SSD_WALL2.md. A complete first draft within ~45 minutes, then refine it and stop.

1. **The SSD floor per check at 45 GB and at 24 GB.** How many bytes per kept token must come from the SSD in the best exact design, and how much of that can hide behind compute.
2. **Your three best designs to remove the exposed SSD time.** Exact first; mark FAST variants. For each:
   - a physics budget: bytes, ms, hidden vs exposed;
   - a prior-art tag: published / adapted / new;
   - the main risk.
3. **At least one design from outside caching and prefetch.** Options, not rules, your methods: lossless compression of expert bytes, expert-major scheduling across checks, layout by co-activation.
4. **One pre-registered screen of 10 minutes or less per design**, a CPU replay where possible.

## Rules

- CPU only, no downloads. Write only in ~/FableGod/push/.
- Both tracks are first-class.
- Mark every number [K] / [P] / [G], and halve modeled gains.
