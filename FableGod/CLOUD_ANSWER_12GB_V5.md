# Cloud Fable 5.1, task 13 (CLOUD_TASK_12GB_V5.md): its summary as pasted by Kamen, Mon 2 Nov 2026 ~21:1x
(The full ANSWER.md stays in the cloud session: 1,091 words, 2 of 8 Python runs, no web.)

- **Where 120 and 150 sit:**
  - 120: at dense 22 and kept 12-13 it needs pass-1 bytes <= 0.66-0.78 GB per check. The measured band is 0.51-0.71, so 120 is inside the band already: it needs the lower-middle of the band + dense 22 + the drafter ceiling, nothing new.
  - 150 needs <= 0.37-0.58 GB, which is 0.04-0.34 below the band.
  - Belady's 0.29 clears every 150 line, so the caching ceiling would be enough, but LRU is not. The gap is cold-miss structure (new experts per block), not policy. Stop spending nights on cache policy.
- **Levers, halved:**
  - The 3-bit ranks 0-1 arm is the single biggest: -5.3 ms, quality-defined, Tuesday.
  - The exact set together takes pass-1 bytes 0.62 -> ~0.46 GB:
    - the 4-bit drafter buys ~13 slots per layer (bytes scale as 1/C on the C 51 -> 160 pair)
    - block trimming at the drafter's first low-confidence row
    - early-layer prefetch under pass 2
  - Hiding inside pass 1 is not a lever: layer L's reads can overlap only layer L's resident experts, because the next dense step waits for the last miss. The 0.78 exposed share is structural.
- **The plan:**
  - 120 = theta 0.8 + dense <= 22 + kept >= 12 + bytes <= 0.66 (cycle 91-97 ms).
  - 150 = dense <= 20 + kept 13 + bytes <= 0.45: either the 3-bit arm alone or the whole exact set.
  - The honest ceiling with this week's pieces is ~140 at kept 13.
- **Chat:** the two-pass buys chat nothing but a second dense pass (pass 2 runs in 91-97 % of blocks): 18-23 tok/s. Drop pass 1 on chat and run one exact + gate pass: 25-31.
- **Pushback:** the quality budget moved silently from "<= 1 % code tokens" to "EvalSuite same scores" (theta 0.8 changes 1.63 % of code rows on a 1.16 % floor). Say explicitly that EvalSuite is the contract before theta 0.7 or top-1-on-peaked gets tested.

## Orchestrator's check (21:1x)
- **1/C scaling holds** on Breakthrough's pair: pass-1 0.708 GB at C 51 vs 0.224 at C 160, a ratio of 3.16 for C ratio 3.14. +13 slots (51 -> 64) gives x0.80.
- **The quality contract is already Kamen's rule** (memory small-ram-bar-task-results, 10-21 / 10-23): the <= 12 GB line and the optional fast modes are judged by task results (same scores on math / code / instruction tests); the exact default stays exact. Cloud is told in task 14.
- **Routed to Breakthrough** (12 GB plan; Fit12 = engine):
  - stop cache-policy work at 12 GB
  - replay the exact byte set (C 64 via the 4-bit drafter; block trimming at the first low-confidence row; early-layer prefetch under pass 2)
  - the 3-bit ranks 0-1 arm Tuesday
  - chat as a single exact + gate pass
- **Next cloud task:** CLOUD_TASK_24GB.md (task 14), the 24 GB line.
