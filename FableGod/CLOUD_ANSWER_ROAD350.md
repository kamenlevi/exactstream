# Cloud Fable 5.1, task 15 (CLOUD_TASK_ROAD350.md, the /btw chat's draft updated by the orchestrator): its summary as pasted by Kamen, Mon 2 Nov 2026 ~21:4x
(The full ANSWER.md stays in the cloud session: 981 words, 4 of 8 Python runs, no web.)

- **The ladder:**
  - Eleven rows from ~110 at night, every gain halved: ~225 tok/s (56 ms cycle at kept 12.6).
  - The same pieces at their unhalved targets (dense 17, experts at 85 % of the gated floor, SSD 2 ms, draft hidden): 38.5 ms, giving 327 at kept 12.6 and 353 at 13.6.
  - "350 is not a twelfth idea; it is this ladder with nothing halved and the drafter at the top of its ceiling."
  - Rows 1-4 land this week with no decisions; row 5 is the whole game.
- **The critical piece: the SSD stack** (row 5, -23 of the -48 ms). It has no substitute: in exact mode the SSD can only be hidden, and the gate already took the bytes it can take.
  - If R-57's look-ahead is a loss at 45 GB (as the ranks 0-1 prefetch was at 12 GB), exposed SSD stays ~22 ms and the ceiling is ~170 at night.
  - Second: the dense reference change. Without it, everything else at target gives ~275.
  - NAX7 refused costs ~30.
- **The draft design:**
  - The early-tap drafter reads taps from layer <= 24.
  - The ANE drafts three candidate blocks from three anchors (all accepted, and the drafter's two lowest-confidence rows) while the check finishes its last 24 layers. The matching candidate goes straight in.
  - Exposed draft falls 6.3 -> ~3 ms at 50 % anchor coverage.
  - Tuesday must show:
    - kept >= 0.97x with early taps
    - the ANE's 12.4 ms holding under a busy GPU
    - a sync-free hand-off
    - >= 50 % anchor coverage (the decision map can answer this today)
- **Three decisions for Kamen, in evidence order:**
  1. The gate 0.99 fast-mode default, after Wednesday's arms A and B. R1 is Thursday's sub-decision, and "the one to refuse".
  2. The attention kernel as the first dense-path reference change, after its re-test and a bitwise AR check.
  3. NAX7 as the exact expert reference, after 3c's bitwise copy says whether it can wait a week.
- **Pushback:** trees stay off the ladder. A continuing branch costs 2.4 GB of node state for +0.5-1 kept, and pays only if node state drops under ~0.3 GB.

## Orchestrator's check (21:4x)
- **Consistent with tonight's local results** and with the orchestrator's estimate: fast built ~200-300, exact ~140-180.
- **Decision 3 can't wait:** the 3c bitwise copy is already dropped (Leap D3C: 3c is ALU-bound, no-DRAM only 8.9 % faster). So the exact expert gain needs the NAX reference: Kamen's call after NAX7's numbers and certification. NAX7 runs on the FAST track meanwhile.
- **Routed:**
  - MTP Head: anchor coverage (CPU replay on stored walks). How often does the true accepted length equal one of 3 anchors (full accept; the drafter's 2 lowest-confidence rows)?
  - BigMoE: the SSD stack is the critical piece; measurement 2 tonight; the R-57 build.
- **Cloud paused until Tuesday's results:** the SSD trace, the combined build, EARLYTAP, NAX7.
