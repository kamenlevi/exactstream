# BUILD DAY, Wed 4 Nov 13:45-23:30 (orchestrator, written 12:1x)

> (a quote from the project owner, kept private)
lossy numbers." This page is the starting list: what is measured, what each build is worth, who owns it.

Marks: [K] measured, [P] arithmetic on measured numbers, [G] judgement.

## Where code stands at 45 GB (BigMoE 5b, C 224, 12 HumanEval prompts, 39.4 GB peak)

- EXACT (new default, == REF2-T): 83.1 tok/s [K].
- FAST U3C: 110.1 tok/s at 15.0 MB per kept token [K]. EvalSuite same-scores still to run (RELAX tonight).
- FAST is GPU-bound at 45 GB: U3 -> U3C cut SSD bytes 27 % and gained only 3.7 % [K]. More byte cuts help the
  24 / 12 GB lines (BYTES10), not code at 45 GB.

## What 200 / 300 mean

- They are the code-only projections of Mission 14's full stack (FINDINGS.md section "MISSION 14", item 5): code ~300
  FAST / ~200 EXACT, mixed ~150-170 [P/G].
- Two of that stack's pieces failed since: the 32-row check (Tue night: tokens differ, code -62 %, KILL) and the k-head
  (KILL).
- The by-day GPU floor on code is ~230 EXACT / ~300 FAST per 16-row check at kept 12.6 (FINDINGS: dense 19-21 ms,
  experts ~30, draft 3-6, 2.5 rest) [P]. So 300 FAST = every GPU part at its floor; 200 EXACT = within ~15 % of it.

## Today's builds, biggest first (code, 45 GB)

| # | build | owner | track | worth [P unless marked] | ready? |
|---|---|---|---|---|---|
| 1 | Dense pass (Leap's honest size, 14:1x) | Leap | both | EXACT ~-2.5..-3 ms per check: DENSE-N -1.46 ms [K], then D5 ~-0.5 + D1 kernel A ~-0.3..-0.8. FAST ~-8..-9 ms: + the HC read with 8-bit HC weights, one pass (~9.9 -> 4-5 ms GPU), + a small-shape bundle ~-1. The earlier -15.5 ms is NOT reachable with exact levers | BUILT 15:0x, all wired, default off: EXACT ~-3.0 ms per check = DENSE-N-RAM ~-2.0 (no RAM cost, 2190e10) + hc_b kernel B ~-1.0 (bitwise, 4 quiet runs); FAST ~-7 ms = + HC8-1P v4 ~-4.8 (34206f9, bitwise == the certified v2, -0.6 GB). In BigMoE's next 45 GB slot (7 arms, ~38 min) and EvalSuite's F1 FAST batch. Friday prep bug fixed (52cf2c9: NAX6-NH had no p11 path) |
| 2 | Lever 2: layer-ahead forecast reads into the tier, SAMELAYER-T router | BigMoE + MTP Head (router) | EXACT | exposed SSD ~24 -> 6-8 ms at C 256: ~+9 % | router trained (recall 0.823 code / 0.851 chat, BETWEEN) [K]; reads not built |
| 3 | Native record layout in the engine (NAX6-N / NAX6-NH) | Leap (BigMoE reviews) | FAST unless Kamen certifies a NAX reference | expert kernel +18.7 % (N7-13, quiet [K]) / +23.3 % (N7-14 [K]); ~+5-8 % end to end [P]. LEAP 14:5x: it is EXACT only vs a NAX reference: NAX6's bits differ from 3c's (REF2-T's kernel) in ~70 % of expert outputs, and 3c-N (bitwise == REF2-T) is 4-7 % SLOWER. The NAX-reference call (decision 4's same-scores test, ~1 night, ~+12-14 % EXACT) is Kamen's, unanswered since the Mon FOR KAMEN note | Pick A (14:5x): no throwaway repack-at-load patch. Friday's in-place conversion (~2 min each way) runs NAX6-N in the engine with no new code; Leap pre-registers the Friday arms today (native-NAX vs row-major-NAX vs 3c, equal RAM) |
| 4 | ~~Drafter ensemble~~ OFF THE LIST (MTP Head 13:5x) | MTP Head | - | already KILLED in two live walks on 11-03: DUALDRAFT x1.026 kept (line x1.08), DIVERSE2 x1.003-1.010 (~/TrainLab/REPORT.md :2411 / :2432). The 94.0 vs 90.9 % was an either-right oracle; the drafters miss at the same places (P 0.87-0.90), and a second member costs a second drafter pass | the drafter lever left is a re-ranker (right token in top-3 at 0.888 of mid rows): a test, not a build |
| 5 | One-call OOO | BigMoE | EXACT | -3.7 ms per check: ~+3-6 % | not pre-registered |
| 6 | QOM in the engine + gated last row + 3-bit (code) | BigMoE, NewIdeas masks, Breakthrough map | FAST | the 24 GB code line: 35.7 MB per kept token on replays calibrated to the engine; ~32 after the 11-bit store, ~29 with POOLSPLIT [P] (NewIdeas 55e170f, 13:40: the earlier ~22 double-counted 11-bit). PLUS DRAFT17 (a 17th draft gates every last row): 35.7 -> ~25.5 [P, NewIdeas x0.706, MoonShots' independent x0.723, 5679af7]; ~0 ms after Leap's _rows16 fix (landed 05fafe7, bitwise) | quality PASS at the 24 GB pool [K]; DRAFT17's gated rows = map screen 6 (Thu ~07:4x); not in the engine |
| 8 | Trellis FAST copy of the experts (QTIP-style TRL-2 / 2.40 b / TRL-3) for drafter-sure rows | Fit12 (encoder), NewIdeas (screen), Breakthrough (map hook), Leap (decode kernel) | FAST | with DRAFT17: code 4.0 (TRL-2) / 7.2 (2.40 b) MB per kept token at 24 GB [P, UNCHECKED] | started 14:2x on Kamen's word: CPU encodes of a subset today, map screen 7 tonight / Thu; the full store (31-36 GB disk, Metal encoder) is Kamen's call after it |
| 7 | Agent fix | BigMoE | EXACT | agent -9.9 % back | tonight's leave-one-out names the lever |

## If every build lands today [P/G]

- Code EXACT ~90-105 tok/s; code FAST ~120-135 (14:1x, with Leap's honest #1; was 95-115 / 125-150 at 13:5x and
  105-135 / 135-170 at 12:1x). The low ends halve each predicted gain (our usual rule). Per check: EXACT ~150 ms - dense 3
  - native 8 - forecast reads 16 - OOO 3.7; FAST ~115 ms - dense 8.5 - native 5 - forecast 5 - OOO 3.7. Lever 2's 16 ms
  assumes recall >= 0.85; SAMELAYER-T has 0.823 on code, so read #2 as the high end.
- So today's EXACT gain is mostly lever 2 (forecast reads) and the native expert layout; FAST gains more (the HC read).
- Reaching 200 / 300 also needs the expert kernel near its streaming floor (NAX6 43.2 ms vs 26-29 at S 16 [K C2]) and
  the SSD fully hidden. Those are the next builds after today's.

## Rules for the day

- Pre-register each build's line before its run; EXACT builds are checked token for token against REF2-T.
- GPU by day: one > 5 GB job at a time (MEM in the share lock); quiet locks only for headline numbers.
- Everything on the GPU ends by 23:15. RELAX starts 23:30 (EvalSuite), then BigMoE's Thu slot, then T0.
