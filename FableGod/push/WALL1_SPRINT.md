# WALL 1 SPRINT: make the SSD feel like RAM

> (a quote from the project owner, kept private)

## The wall

The 125B is bigger than RAM, so its experts stream from the SSD [K] (BigMoE c45l_256, Fable A SSD_WALL2):

| | Value |
|---|---|
| SSD bandwidth | ~13 GB/s, vs ~280 GB/s for RAM |
| Check cycle at 45 GB, C 256 | 96.8 ms |
| SSD busy per check | 42 ms, of which exposed ~24 ms |
| Fixed cost per loader call | ~20 ms per check (0.28 ms x ~70-77 calls) |
| Bytes | ~21 ms per check |
| SSD MB per kept token, exact | ~62 |

## The rule of this sprint

**It doesn't come back as "impossible".** Every report gives two things:
1. the best MEASURED number so far;
2. the next concrete step.

When a lever dies, the next one goes in. Physics floors (Fable A's section 2) are targets, not excuses. Exact first (bitwise to its reference); FAST levers ride on top.

## Targets

| Line | Target | Today |
|---|---|---|
| 45 GB, per-call overhead | <= 5 ms per check | ~20 |
| 45 GB, exposed SSD | <= 6 ms per check | ~24 |
| 45 GB, steady exact tok/s | +25 % or more | 63.6 |
| Night (~52 GB, volatile tier), SSD MB per kept token | <= 33 (the 350 floor's bytes) | ~62 |
| 24 GB, exact tok/s | 1.5x or more | BigMoE measures it first, same prompts |

## The levers, in order (all exact)

**1. Kill the per-call overhead.**
- One burst per layer (JOINT: built, +7-8 % at 24 GB, never measured at 45 GB).
- The SSD keepalive (built).
- Demand records split into parallel >= 256 KB pieces.
- Fewer loader calls: JOINT 70.8 -> 41.9 per check [P].
- Tonight measures JOINT and the keepalive (c45tj, ka1).

**2. Never let the queue empty.** Layer L+1's forecast reads join layer L's open burst, landing in the volatile tier. L+1's router on L's input already names 50.5 % of the misses [K ROUTEPRED]. Tonight's pre-route measures it (pr_224); if it misses its recall line, MTP Head fits a learned correction.

**3. A bigger cache for free.**
- The volatile expert tier: built, smoke test tonight.
- The approved conversion (native + 11-bit): smaller records, so more experts per GB.

**4. Fewer bytes per miss.**
- 11-bit scales: code SSD bytes -17.6 %; the conversion's test runs Fri 00:30.
- The M-01 pack: -4.8 % bytes, already on disk.

**5. The same stack at 24 GB**, then the 12 GB two-pass. Its pass-1 is 57 % SSD wait [K T3P].

## Team (focused: one lead)

- **BigMoE (LEAD):** the engine, the loader and the nightly measurement. It owns the scoreboard numbers.
- **Fable A (designer on call):** turns each night's trace into the next build step. Designs 1-3 in SSD_WALL2.
- **MoonShots:** SSD micro-experiments (queue depth, piece size, first-byte latency, wake-up gaps), on a quiet SSD by day.
- **MTP Head:** the learned L+1 predictor, only if the pre-route misses its line.
- **NewIdeas:** the sprint scoreboard. One row per night:

  | Night | SSD busy | Exposed | Calls / check | MB per kept token | Tok/s | Exact? |
  |---|---|---|---|---|---|---|

- **Leap:** kernels or I/O helpers when the lead asks.

## Cadence

- Each night: one measured step.
- Each morning: a scoreboard row and the next step, for Kamen.
- Sessions on account-1 (BigMoE, MoonShots) move to main on Wed morning, after their night runs. A move kills attached runs, so never before they end.
