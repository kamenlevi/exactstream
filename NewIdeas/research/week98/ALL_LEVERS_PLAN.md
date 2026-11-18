# ALL LEVERS IN PARALLEL to Saturday (NewIdeas, Wed 18 Nov 07:3x; for the orchestrator)

Kamen 05:0x: 550 on lossy and exact, as high as possible, and no optimization skipped because of time. Orchestrator
05:0x: re-plan the week with every lever in parallel; each lands as a flag when it passes its test; whatever has landed
by Saturday is in. Nothing below is armed by this file.

## 0. Rules for the week
- A lever lands as an opt-in flag (off == today's bits) when its own test passes. A lever that misses Saturday keeps
  going and lands after it. Nothing is stopped for time.
- EXACT levers: a bit test in BigMoE's loop, then a timing. A bitwise drafter change (PK) gives the same tokens, so it
  needs no quality gate in either mode.
- LOSSY levers that change outputs: BigMoE's key test, then a tier-1 pass (EvalSuite), then the card gate on the final
  stack. The pin rule extends: the card gate runs exactly the blob that passed its tier 1.
- New work goes first to the sessions off the main account. Leap and BigMoE stay lean until main's Thu 17:00 reset,
  then take the late engine builds.

## 1. Where Saturday lands at measured costs (adaptk_sat.py, CPU replay, 7 s; t30b cells, CC(1))
Costs: 16-row lossy check 42.7 ms (slot #3's no-tier F16nx 45.73 minus REF2 ~3; this IS blob 1's check, since blob 1
carries no R2 / QOM and c25's no-tier arms have no QOM masks either), 32-row check x1.458 (the E14
ratio), overhead 3 ms, Q8H32 sync-free draft 0.94 ms per level (MoonShots ra_sf.py). Self-check: kept 11.263 / 16.263 /
16.910 and v4pass +9.0 % reproduce.

| config, lossy CC(1) | HE+ tok/s | MBPP tok/s |
|---|---|---|
| fixed tree-16 (Thursday's stack) | 204 | 144 |
| fixed tree-32 | 199 | 120 |
| ADAPTIVE K, frozen tau 0.9 | 220 (+7.7 % [+6.2, +9.3]) | 144 (+0.1 %) |
| ADAPTIVE K + PK (0.80 realistic / 0.76 best) | 228 / 230 | 148 / 149 |
| ADAPTIVE K, c2 draft (0.59; c2's own kept loss not modelled) | 240 | 154 |

- PK CORRECTED 07:4x (MoonShots' census of the live Q8H32 pop: 178.8 MB per pop, SF 0.93 ms; cold GEMVs cap at ~260 GB/s,
  so no form of these bytes goes under ~0.69 ms per pop; M-46's tuned ceiling ~0.76, realistic 0.80-0.86). The first
  version used 0.66, below this format's floor. PK alone: fixed tree-16 204 -> 211 (0.80) / 213 (0.76), HE+ +3..+4 %.
- (A 38 ms variant, 222 HE+, assumed QOM-lowered bytes; blob 1 has no QOM, so it is dropped.) Charging per pop instead of per level: tree-16 197 vs
  tree-32 179 (HE+). Exact acceptance gives the same picture (ADAPTIVE K +6.9..+8.2 % HE+).
- **Item 11 is answered: both headlines stay TREE-16** (the exact one stays chain-16 unless the contract is YES).
  32 rows pay only inside ADAPTIVE K. Re-check only if the MoE block cuts the 32-row check much more than the 16-row.
- Stack projection, HE+ lossy [P] (corrected 07:5x: blob 1 has no R2 / QOM; PK 0.80 / 0.76 ms; no MoE-block step): blob 1
  ~199-204 (Q8H32 vs the full head's kept) -> + PK + overhead ~213-218 -> + ADAPTIVE K ~230-235, the realistic top of the
  tree line. MBPP ~144 -> ~150-155. The fallback (section 1b) is measured beside it Thursday night.
- EXACT [P]: chain-16 ~130-150 without the contract; under it, tree-16 ~150-165, ~165-185 with PK + c2 + overhead.
- 550 needs a ~28-30 ms 32-row check (the compiled check; the MoE block is its first module) plus a draft of <= 3-4 ms
  per block (<= ~0.19 ms per level: a smaller-bytes head + PK). Neither lands by Saturday; both start now.

## 1b. The lossy headline candidates (orchestrator + BigMoE, 07:5x)
TREE16 in r57 is exact-only by construction (it asserts no FAST / RLX_D / SPAN / QOM_MTP / INERT). Lossy-45's QOM masks
need DFlash's cf, and R2 (RELAX_SPAN) is chain-only. So:
- (a) BLOB 1 = the base path's exact-safe verify levers + tree-16 + CC(1) on the tree walk + tgl16eu, with no R2 / QOM.
  Tier 1 #1 Thu.
- (c) THE FALLBACK = ccx45cc1 (DFlash chain-16 + QOM + R2 + CC(1); tier 1 PASSED both orders Mon night) + bitwise flags
  only: BOF / HC_B, GDN16, tgl16e. tgl16eu is FAST (argmax-equal, not bitwise), so adding it would leave Monday's PASS.
  - AUTOMATIC UPGRADE: if #2a doesn't run (slot #4's w4a8u arm under ~1 ms, decided ~10:00 Thu), its Thursday-night
    slot runs key test + tier 1 on blob 2u = the fallback with tgl16eu (~-2.7 ms per check). A pass lets the fallback
    carry tgl16eu.
- (b) R2 + QOM_MTP on trees (BigMoE's tree ports, ~1 day after the reset) enter blob 2b only if bit-tested by Fri 07:00.
- Thursday night's lossy timing (i) has two arms, blob 1 and the fallback, in halves (~90 min).
- RULE: Thursday's stack = blob 1 if it passed tier 1 #1 AND beats the fallback by >= 1 % pooled in (i); else the
  fallback (fewer new parts). Later stacks (2a / 2u / 2b) need >= 1 % over Thursday's stack. Saturday reports whichever
  passes its gates and wins its timing; the card gate takes the winner.

## 2. The levers
Each: owner -> what lands -> its test -> ETA -> gain [P] -> odds of landing by Saturday.

1. **PK, the persistent draft level** - MoonShots. Resume the tuned bitwise form (M-46: x1.08-1.10 on the draft; KILLed
   Tue only by the skeleton line). Q8H32 first (lossy + exact arm A), c2 second (arm B). Test: PK == SF on the 120 real
   trees + a quiet timing. ETA: skeleton v2 Wed (share lock) -> bit test + quiet timing Thu morning -> flag
   R57_TREE_PK=1 to BigMoE Thu 18:00 -> exact timing #2 Fri + lossy timing #2 Fri night (bitwise: no tier 1).
   Gain HE+ ~+3..+4 % (replay 204 -> 211 / 213 at 0.80 / 0.76 ms per pop; corrected 07:4x, see section 1). First gate:
   today's skeleton v3 must beat the live SF's 0.93 ms per pop at all (~14:00, share lock); if it can't get under ~0.88,
   MoonShots says so before the bitwise phases. Bitwise freebies it takes: skip the unused extra-screen rows (-inf) and
   an exact top-8 selection instead of argpartition's full sort (~11 dispatches per pop). Odds ~40 %.
2. **c2 QAT head, EXACT** - MTP Head + MoonShots + BigMoE. As set: Thursday-night A/B, A = Q8H32, B = c2 dd_t30qa
   (sha256 353965c5...), halves, 264 / 264. Without the contract it runs on chain-16, with it on contract tree-16.
   - New: Thu ~04:30-08:00 (16 GB) is free, because S22's qa ran on Tuesday. MTP Head may use it for an S22
     continuation from dd_t30qa. The new weights enter arm B only if MoonShots bit-tests them by Thu 15:00 and BigMoE
     re-pins the sha by 18:00; else arm B keeps dd_t30qa.
   - c2 in the LOSSY stack: only after the A/B verdict, through tier 1 #2b (a drafter change moves CC(1)'s picks).
   Gain pooled ~+3-8 % [P]. Odds: the A/B runs for sure.
3. **The MoE block** - Breakthrough. CORRECTED 07:5x (Breakthrough's record, MOEBLOCK.md): the glue is 110 us per block, but
   ~80 us of it is the shared expert's weight traffic, which no fusion removes. The removable ~30 us per block is ~-0.7 ms
   per 16-row check / ~-1.4 per 32-row [P]. It needs a megakernel, whose pre-registered line read BETWEEN: a multi-day
   build, AFTER Saturday. Module 1a's replay saves CPU time the GPU-paced check already hides (Leap's gpubB: GPU busy
   96 % of 56.1 ms, idle 2.0), so it is folded into item 12. No MoE-block arm in slot #4. The -5 ms in the first version
   came from the glue total, shared expert included.
4. **S1 tree-chunk GDN + 64-row clamped trees** - Leap (S1, 2-4 h, after Thu 17:00), BigMoE (the 64-row path), EvalSuite
   (the clamp's gate, pre-registered after the verdict). Small at Saturday's draft cost (+3 % at 1.1 ms per level, GRANDMA C1); +20 %
   only at a <= 0.19 ms-per-level draft. It needs ADAPTIVE K's 32-row path first and its own quality gate. ETA after
   Saturday. Odds ~5 %.
5. **The reference contract** - Kamen. CONTRACT_YES_PLAN.md (899926d) stands. YES: Thursday night (ii) = contract
   chain-16 + the tree-16 A/B, then Part A ~4 h; it also makes ADAPTIVE K and tree drafting exact. NO: (ii) = chain-16
   on NAX6-N.
6. **ZERO's DROP form** (lossy). At 16 rows it is -0.25 GB only (~-0.9 ms), under ZERO's own 0.5 GB bar; at 32 rows
   -0.91 GB. So it rides with ADAPTIVE K's 32-row checks. NewIdeas re-screens it on BigMoE's real tree rows when
   Wednesday's plumbing writes them (CPU, <= 10 min). If it holds >= 0.5 GB per check at ADAPTIVE K's row mix,
   Breakthrough writes an installer on lossy-45's keep-mask hook, BigMoE wires it, tier 1 #2b. Odds ~15 %.
7. **W4A8** (lossy). Breakthrough's w4a8u knob exists (declined Tue for its 3 % expert-output error). It REPLACES
   tgl16eu (EvalSuite read the code: one mode at a time, the same upgrade grouping), so blob 2a = blob 1 with
   R57_XRW=w4a8u and nothing else. Its knob_v2 release check PASSES (Breakthrough 644f189: off == today bitwise,
   3 repeats identical). It is bitwise to nothing: rel L2 vs tgl16eu mean 0.028, max 0.075, ~6x tgl16eu's own error vs
   fp32.
   - CORRECTED 07:4x (Breakthrough): the gain over tgl16eu is only ~-0.3 ms per 16-row check / ~-0.6 per 32-row [P]; the
     -1..-2 ms held only against a stack without tgl16eu. No engine arm has run it.
   - So tier 1 #2a is CONDITIONAL. A w4a8u arm joins Leap's slot #4 (Thu 08:00; Breakthrough hands it over with the
     MoE-block arm by Wed 22:00). #2a and blob 2a happen only if that arm shows >= ~1 ms per 16-row check over tgl16eu,
     deterministic, with no extra expert GB: the bar tgl16eu cleared in slot #3. The decision is known by ~10:00 Thu.
     Odds of running: low.
8. **TRL-3** (lossy weights, 3.07-bit down, -1.4 ms). The records: the trellis family stopped 11-0x (screen 7 KILL;
   TRL-RESCUE KILL; a down-only arm was underpowered). Leap's record says the encoder is done (CPU, bitwise; a full
   store ~1.1-1.5 h on 3-4 processes); the blocker was disk (102 GB free now). So it needs Kamen's disk OK, BigMoE's TRL
   tier and a tier-1 pass. No free builder this week. ETA after Saturday. Odds ~5 %.
9. **Sparse-dense head (GRANDMA C4)** - MTP Head. The 2B analog in its LAB today (an MoE MLP at 1/4 active vs the
   control, tree-32 kept, PASS >= x0.97, ~2 h, share lock <= 4 GB). PASS -> the 125B head (16 GB, 3-6 h) in the first
   16 GB window, then MoonShots' engine side after PK. Drafter only, so exact-safe. ETA after Saturday. Odds ~5 %.
10. **Per-check dynamic vocabulary (GRANDMA C5)** - MTP Head's CPU coverage screen: are the true token and the head's
   top-8 inside a 4-8K per-check screen as often as inside the static screen? It needs a dump with prompt ids (the bank
   or CERTIFY98's dev set). PASS -> MoonShots' per-check gathered screen in the sync-free drafter. Exact: a timing.
   Lossy: tier 1 #2b. Gain +3-5 %. Odds ~20 %.
11. **Tree size per headline** - answered in section 1: tree-16 for both.
12. **Overhead / commit + the replayed check (module 1a milestone 2), ONE lever** (item 3's replay folded in; their gains
   overlap in the same idle time). NewIdeas splits the ~3 ms (commit / host sync / Python) from BigMoE's
   data/r57/<tag>_checks.json timers once BigMoE names the fields (CPU), or from one profiled cycle in Thursday's key test.
   Breakthrough's bitwise engine test of the replayed check, rec1a_c25, is ARMED for Thu 07:30-07:50 (~20 min, share
   lock MEM 35 GB, NO-SWAP >= 43). If bitwise, the timed arm goes in Leap's first slot after; Breakthrough writes the
   installer; BigMoE wires it; bit test + both timings #2, no gate. Ceiling: the exposed part of the per-cycle time,
   ~-2 ms (+3-4 %). Odds ~35 %.
13. **ADAPTIVE K (added; stopped Tue for time only)** - 16 or 32 rows per check by path_p, frozen tau 0.9 (ADAPTK.md,
   adaptk_fresh). At Saturday's costs HE+ +6.9..+8.7 % (exact and CC(1)), MBPP +0.1 % (section 1); Tuesday's switch screen keeps +6.9 %
   with a switch cost of up to ~8 ms. It needs the 32-row tree path in r57 (the C32one hook) + the switch. BigMoE builds
   it after Thu 17:00. Lossy: tier 1 #2b (it changes CC(1)'s outputs). Exact: only under the contract. Odds ~25 %.
15. **R2 + QOM_MTP tree ports (added, lossy)** - BigMoE, ~1 day after the Thu 17:00 reset. Blob 2b only if bit-tested
   by Fri 07:00 (section 1b). Odds ~25 %.
14. **Small exact items (added):** tgl16ef (exact, ~-0.19 ms; Leap's slot #4) and REF2_GDN_LAZY on trees (<= -0.5 ms;
   BigMoE's Wednesday bit test). Each goes ON in exact timing #2 when bitwise.

Closed on the record, not reopened: LANDMARK-ANS (KILL Tue 09:4x), the pool drafter, prefix eyes, code || commentary,
the SLC CPU fill, 1c tiers (V2 FAIL, so G1 stays out), close-call d2 (tier 1 FAIL).

## 3. GPU windows
45 GB = the whole Mac. Every big job waits for Kamen away >= 900 s; <= ~6 GB while he is at the Mac.

| window | what runs |
|---|---|
| W1 Wed now-15:00 | BigMoE's plumbing bit tests (REF2 on the lossy loop, GDN_LAZY on trees); MoonShots' PK skeleton (share lock); MTP Head's 2B analog (share lock) |
| W2 Wed 15:00-23:30 | <= ~6 GB: CPU screens (overhead split, DROP re-screen, vocabulary coverage) and builds |
| W3 Wed 23:30-~04:30 | CERTIFY98 slim capture (BigMoE), as set |
| W4 Thu ~04:30-07:25 | 16 GB: MTP Head's S22 continuation, if it wants it (it must end by 07:25) -> Breakthrough's rec1a_c25 07:30-07:50 (share lock MEM 35 GB) |
| W5 Thu 08:00-15:00 | Leap's slot #4 08:00 (tgl16ef + the w4a8u pair F16nw / F16gw) -> PK quiet timing -> BigMoE's key test (blob 1) ~11:30 -> tier 1 #1 12:00-14:30, as set |
| W6 Thu 15:00-23:30 | main resets 17:00: ADAPTIVE K, the replayed-check flag if bitwise, blob 2a (w4a8u) by 22:00 only if slot #4's arm cleared ~1 ms |
| W7 Thu 23:35-08:00 | (i) lossy timing #1, blob 1 + the fallback in halves (~90 min) -> (ii) exact + the c2 A/B -> (iii) count check (~03:05) -> NO: key test + tier 1 on blob 2a if it exists, else on blob 2u (the fallback with tgl16eu), or the tgl16eu-off bisect if #1 failed (~03:05-06:15); YES: Part A ~4 h |
| W8 Fri 08:00-15:00 | (YES: stage 2 if needed, <= 3.3 h) -> blob 2b key test + tier 1 #2b -> (NO: one bisect pass if it failed) -> exact timing #2 ~13:45 |
| W9 Fri 15:00-23:30 | <= ~6 GB; the final lossy stack frozen 22:00 |
| W10 Fri 23:35-Sat ~07:00 | lossy timing #2 (45 min, only if the stack changed) -> card gate (5.8-6.5 h) -> verdict ~07:00 |

## 4. The second tier-1 pass, priced
- One pass = BigMoE's key test (~30 min, quiet) + EvalSuite's tier 1 in both orders (~2.5 h at 45 GB) + EvalSuite's
  amendment at the desk: ~3 h of the whole Mac. The lossy re-timing (runner 69, ~45 min) runs once, Fri 23:35, before
  the card gate, which still ends ~07:00.
- Without the contract: #2a (w4a8u alone, only if slot #4's arm clears ~1 ms) Thursday night after (iii); #2b (the bundle: c2 in lossy, ADAPTIVE K, the
  DROP, dynamic vocabulary, w4a8u if #2a passed) Fri 08:00-11:00. On a FAIL, one bisect pass
  (11:15-13:45) drops the newest lever first.
- With the contract: Thursday night is Part A, so there is no #2a, and #2b follows stage 2 (key test + tier 1
  ~11:20-14:20 if stage 2 runs), with no room for a bisect by day; exact timing #2 then likely moves to Fri 23:35.
- The card gate runs the last stack that passed tier 1 and won its timing; else Thursday's stack. Thursday's stack itself is
  blob 1 or the fallback by section 1b's rule. Any passing later stack not yet timed (2u, 2b) gets its timing in a free
  quiet slot Fri by day, else as an extra arm of timing #2 Fri night (the verdict then slips ~45 min).
- EvalSuite's price check (07:4x): ~3 h 10 per pass at night (Monday night's tier 1 took 2 h 26 for both orders, plus the
  30-min key test and its 15-min smoke). By day it stretches with at-Mac yields, so #2b's 08:00-11:00 holds only if
  Kamen is away from 08:00.
- Rules agreed with EvalSuite (07:4x; in its amendment 6 draft, b881eb7, filed Thu midday before any stack data):
  1. If tier 1 #1 fails, the tgl16eu-off bisect takes #2a's night slot (w4a8u shares tgl16eu's grouping, so it is
     out); timing #1 times the bisect blob; 2b builds on whichever passed.
  2. "Won its timing": pooled whole-decode tok/s (HE+ 164 + MBPP 100) above timing #1's by >= 1 %; else Thursday's
     stack goes to the card gate (fewer new parts, lower card-gate risk). Both are quoted.
  3. If #1 failed, the card gate runs the latest passing stack (rule 2 decides between two); if nothing passed, there
     is no card gate and the reading is "measured, not gated".
  4. Blob 2a = blob 1 with R57_XRW=w4a8u and nothing else; EvalSuite's chain checks the diff.
  5. 2b's cut list names, per flag, its passing test record, its runs.json engagement field and its landing time; a
     flag with no engagement field stays out.
  6. BigMoE wakes EvalSuite at the 2b cut; EvalSuite files the 2b addendum before #2b's data.

## 5. Cut-offs
- Blob 1: Thu ~11:30, as set.
- Blob 2a (w4a8u): Thu 22:00, only if slot #4's arm cleared ~1 ms (decided ~10:00 Thu).
- Blob 2b (the bundle): Fri 07:30, holding only the flags whose bit or unit tests passed by Fri 07:00.
- Exact flags for timing #2: bit test by Fri 12:00. If Kamen is back early, exact timing #2 moves to Fri 23:35 and the
  verdict slips to ~07:35.
- The final lossy stack: frozen Fri 22:00.

## 6. Kamen's calls
- The reference contract, YES / NO (asked again 05:0x).
- TRL-3's disk, only if a builder frees up.

## 7. Usage rule (orchestrator, received ~07:40; through Saturday's 07:00 verdict)
- On the second research account, every session except EvalSuite stops at 85 % of its week (63 % on Wed morning; this
  week's average burn would empty it ~Fri 04:00, before tier 1 #2b and the card gate). EvalSuite keeps 85-100 % for the
  gates.
- Breakthrough and MoonShots move their builds to their main-account sessions after main's Thu 17:00 reset, or earlier
  if the account reaches 85 % first (a HANDOVER hand-off; one session owns each repo and its runners).
- MTP Head, on its own account, takes the head work: items 2, 9 and 10.
- NewIdeas stays coordination-only and lean; at 85 % routine coordination returns to the orchestrator.
- Defaults until Kamen answers (orchestrator 07:4x): contract NO, so the exact headline is chain-16, the c2 A/B runs on
  chain-16, and Thursday night takes key test 2a + tier 1 #2a.
