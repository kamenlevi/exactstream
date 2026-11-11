# Cloud Fable 5.1, task 12 (CLOUD_TASK_SSD45.md): its summary as pasted by Kamen, Mon 2 Nov 2026 ~21:0x
(The full ANSWER.md stays in the cloud session: 1,148 words, 2 of 8 Python runs, no web.)

- **The budget.** A 36 ms cycle moves at most 0.41 GB.

  | case | demand per check | share of the cycle |
  |---|---|---|
  | day, C 256, no gate | 0.54 GB | more than the whole cycle |
  | day, C 256, gate 0.99 | 0.32 GB | 78 % |
  | night, C 320 + gate | 0.16 GB | 40 % |

  So the bytes fit only at night, and the open question there is the chain.
- **The chain.** Today's 65 ms exposed = ~47 ms of bytes + ~22 ms of per-layer latency (48 x ~0.45 ms of issue, first byte and ramp) - ~6 ms that OOO recovers.
  - At the 350 engine each layer computes for only ~0.75 ms. A miss hides inside its layer only if issued at the layer's start, and two unprefetched misses in one layer stall it.
  - The limit is ~60 demand misses per check.
- **Levers, halved:**

  | lever | saving |
  |---|---|
  | look-ahead (the SSD has slack for its waste at 45 GB, unlike at 12 GB) | -7.6 ms of chain, -1.5 of bytes |
  | rank-aware residency as a priority (protect ranks 0-1, evict 2-9 first) | -1.8 ms |
  | per-layer slot counts | -1.1 ms |
  | precision on miss | ~-0.9 ms |
  | IO hygiene | ~-0.9 ms |
  | first-layer prefetch under the draft | ~-0.9 ms |
- **The plan.** At night: C 320 + gate 0.99 + rank priority with per-layer slots + look-ahead + OOO + IO hygiene, giving ~2 ms exposed (upper bound; 3-4 ms halved).
  - Required: <= 60 demand misses per check, recall >= 0.69 with waste <= 0.5x saved, <= 0.3 ms latency per miss.
  - By day at 42 GB the SSD does not disappear: ~6-10 ms exposed, giving 275-300 tok/s. <= 2 ms by day needs C >= 300, i.e. the 45-46 GB waypoint itself.
- **Pushback.** The 22 ms chain term is judgement fitted to one measured number. Measurement 2 (the per-layer timeline at C 320 with OOO on) must run before the look-ahead is finished: it decides whether the chain is latency (the look-ahead fixes it) or bytes (only residency does).

## Orchestrator's check (21:0x)
- **Arithmetic holds** on our inputs:
  - 11.3 GB/s x 36 ms = 0.41 GB.
  - C 256: 29.6 / 1.3 misses per kept token x 12.5 kept x ~1.9 MB ≈ 0.54 GB; the gate's -41 % gives 0.32.
  - C 320: 15.3 / 1.3 x 12.5 x 1.9 ≈ 0.28; the gate gives ~0.16.
- **Its look-ahead requirement sits exactly on R-57b-R's replay:** recall 0.693, waste 98 / saved 187 = 0.52x.
- **Uncertain:** "rank priority -1.8 ms". Breakthrough's replay (RANKCACHE_read.txt) found rank-aware admission flat in total at C 160, because pass 2 rose as pass 1 fell. At 45 GB the gated rows need only ranks 0-1, so a priority may still pay, but it needs BigMoE's replay at C 256 / 320 with the gate.
- **Routed:** BigMoE gets measurement 2 (per-layer stall timeline, C 320, OOO on; ride on tonight's ticket if cheap) and the rank-priority replay. R-57's look-ahead build continues.
