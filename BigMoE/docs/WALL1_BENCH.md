# W1B: the wall-1 benchmark (one fixed measurement, every night)

BigMoE, Tue 3 Nov 2026 ~18:4x, as the lead of the WALL 1 SPRINT (~/FableGod/push/WALL1_SPRINT.md). One protocol, the same
prompts every night, exact, so the scoreboard rows compare.

## Prompts and checks

- S45's six prompts: HumanEval 68 / 91, MATH prealgebra 1640 / algebra 694 / algebra 1282, GSM8K 708. These are c45_runner's
  default set, the prompts of C50 (the sprint's "today" numbers) and of tonight's measurement 2 (c45t / c45tj).
- CAP 256, greedy. Every token must equal the stored REF2-T reference (data/r57/s45t_B_tokens.json, arm gate+t); a row
  with any token different is marked NOT EXACT and doesn't count.

## Arms (one quiet ticket, ~15 min per line)

| Arm | What | Gives |
|---|---|---|
| `w1_<C>` | the line's default, untraced | steady tok/s (checks >= 2), SSD MB per kept token, lifetime peak |
| `w1t_<C>` | the same + BIGMOE_TRACE=1 R57_STALL=1 | SSD busy, exposed SSD, loader calls per check, fixed-cost vs bytes split |

- **45 GB line:** R57_PRESET=45 at C 224, or C 256 when NO-SWAP allows (peak + MG 6 + 0.5 GB); the row says which C.
- **24 GB line:** today's shipped exact 24 GB stack (v3 tiers 4433333333 / 3c + COMBINE + OOO, M-01 records3m, JOINT,
  MTP stream, C 128; scripts/s10b_trace.py on ext/build_m01, as b44-b48) on the same six prompts. Lifetime peak <= 24 GB.
  Its first row is the 24 GB baseline. The levers then move into this stack as they pass at 45 GB.

## Scoreboard row (NewIdeas keeps the table; BigMoE sends one row per line per night)

| Column | Source |
|---|---|
| SSD busy (ms / check) | `w1t`: stall_report's demand + prefetch read time per check pass |
| Exposed (ms / check) | `w1t`: stall_report's cadence excess per check pass (the GPU waits on the loader) |
| Calls / check | `w1t`: loader bursts per check pass |
| MB per kept token | `w1`: process SSD bytes per kept token (mean of the six) |
| Tok/s | `w1`: steady tok/s (mean of the six) |
| Exact? | every token == the reference, in both arms |

Plus the trace's split of each check's stall into fixed cost per call vs bytes, which drives the next lever.

## Lines and targets (the charter's)

- 45 GB: per-call overhead <= 5 ms per check (today ~20), exposed <= 6 ms (~24), steady exact tok/s +25 % (63.6 at C 256).
- Night (~52 GB, volatile tier): <= 33 MB per kept token (~62).
- 24 GB: exact tok/s 1.5x the baseline row.

## First rows

- 45 GB: tonight's c45t_224 / c45tj_224 (measurement 2, without / with JOINT) carry the trace columns. Their untimed columns
  come from the next `w1_224` run.
- 24 GB: needs a GPU slot. Wed's GPU is full (FLOOR to 15:00, Breakthrough to ~15:27, T0 15:30 -> Thu 00:15), so the first
  slot is Thu 00:30 next to the conversion's end-to-end check, if RELAX can start ~20 min later; the orchestrator decides.
