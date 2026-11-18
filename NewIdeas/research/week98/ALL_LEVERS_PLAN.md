# ALL LEVERS IN PARALLEL to Saturday (NewIdeas, Wed 18 Nov 07:3x; for the orchestrator)

Kamen 05:0x: 550 on lossy and exact, as high as possible, and no optimization skipped because of time. Orchestrator
05:0x: re-plan the week with every lever in parallel; each lands as a flag when it passes its test; whatever has landed
by Saturday is in. Nothing below is armed by this file.

**Wed 11:1x: CONTRACT YES** (Kamen, relayed by the orchestrator). CONTRACT_YES_PLAN.md runs, merged with today's changes and
approved by the orchestrator. The contract goes into the EXACT build only; the lossy stack, its gates and CERTIFY98 stay on
NAX6-N. CERTIFY98 goes FULL tonight (the disk floor is 50 GB now). Tonight and Thursday night: section 3a.

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
  11:4x: trees are NO-GO for exact this week (M4C 15 / 16), so the exact headline is contract chain-16: ~130-150 plus the
  contract's ~-2.6 ms per check (~+5 % [P]) and c2 if it wins (ii)'s A/B.
- 550 needs a ~28-30 ms 32-row check (the compiled check; the MoE block is its first module) plus a draft of <= 3-4 ms
  per block (<= ~0.19 ms per level: a smaller-bytes head + PK). Neither lands by Saturday; both start now.

## 1b. The lossy headline candidates (orchestrator + BigMoE, 07:5x)
TREE16 in r57 is exact-only by construction (it asserts no FAST / RLX_D / SPAN / QOM_MTP / INERT). Lossy-45's QOM masks
need DFlash's cf, and R2 (RELAX_SPAN) is chain-only. So:
- (a) BLOB 1 = the base path's exact-safe verify levers + tree-16 + CC(1) on the tree walk + tgl16eu, with no R2 / QOM.
  Tier 1 #1 Thu.
  - BigMoE 10:59 (fda65e7e): PK and the sync-free drafter pass its loop bit test, so R57_TREE_PK=1 is in. REF2_BOF /
    REF2_HC_B never engage on lossy-45's base path (BOF 0 fires, HC_B 0: its dense layers run native and its HC read goes
    through HC8 / HCB), so they are out of blob 1. The fallback runs the same base path, so expect the same there ((i)'s
    counters show it); EvalSuite drops them from the fallback by its rule 5 (11:4x). Under the contract the exact headline
    runs HC_B OFF and BOF inert (Leap 11:24). CC(1) and tgl16eu engage (2 + 2 close
    calls; 0 calls to today's kernel). GDN16 is not in the snapshot yet; it is bitwise, so it can join for timing without
    touching tier 1.
  - BigMoE 11:0x (d942e0e6): the fallback identity pre-check PASSED 6 / 6 (lx45cc1's line on s98b vs its stored
    humanevalx ids, flags off and on). The 104-key check runs in (i).
- (c) THE FALLBACK = lx45cc1 + bitwise flags only (BOF / HC_B, GDN16, tgl16e) (orchestrator 08:0x, on EvalSuite's finding,
  53e0c83). Scored post hoc vs the EXACT refs (both orders pooled): ccx45cc1, i.e. lx45cc1 + CC(1), FAILS the runaway guard
  (14 / 2, p 0.0042; accuracy passes, -1.70 pt), while lx45cc1 PASSES (16 / 9, p 0.23; -0.57 pt). CC(1) tips runaways
  against exact, and the gate standard is lossy vs EXACT. tgl16eu is FAST (argmax-equal, not bitwise), so it stays out
  of the fallback unless tier 1 covers it.
  - AUTOMATIC UPGRADE: if #2a doesn't run (slot #4's w4a8u arm under ~1 ms, decided ~10:00 Thu), its Thursday-night
    slot runs key test + tier 1 on blob 2u = the fallback with tgl16eu (~-2.7 ms per check). A pass lets the fallback
    carry tgl16eu.
  - A 104-key humanevalx identity check runs on (i)'s fallback arm.
- BLOB 1's FIRST BISECT, by the failing line: a runaway fail -> CC(1) off; an accuracy fail -> tgl16eu off. Blob 1 carries
  CC(1) on the tree walk, so the runaway tip is its likeliest failure. If CC(1) fails on trees too (exact acceptance,
  adaptk_sat): blob 1 HE+ 204 -> 196, MBPP 144 -> 135, and the tree line's top ~230-235 -> ~220-225 HE+.
- (b) R2 + QOM_MTP on trees (BigMoE's tree ports, ~1 day after the reset) enter blob 2b only if bit-tested by Fri 07:00.
- Thursday night's lossy timing (i) has two arms, blob 1 and the fallback, in halves (~90 min).
- RULE: Thursday's stack = blob 1 if it passed tier 1 #1 AND beats the fallback by >= 1 % pooled in (i); else the
  fallback (fewer new parts). Later stacks (2a / 2u / 2b) need >= 1 % over Thursday's stack. Saturday reports whichever
  passes its gates and wins its timing; the card gate takes the winner.

## 2. The levers
Each: owner -> what lands -> its test -> ETA -> gain [P] -> odds of landing by Saturday.

1. **PK, the persistent draft level** - MoonShots. Resume the tuned bitwise form (M-46: x1.08-1.10 on the draft; KILLed
   Tue only by the skeleton line). Q8H32 first (lossy + exact arm A), c2 second (arm B). Test: PK == SF on the 120 real
   trees + a quiet timing.
   - 08:1x SKELETON v3 = GO (MoonShots, QUEUE M-46, 720cd87): 0.758 ms per pop at 20 TGs x 1024 threads, f 0.150 (M-46's GO
     line), 73 us above the 260 GB/s cold floor; live SF 0.93-0.95. Fewer, larger TGs halve the redundant prologue work.
     Bitwise projection 0.78-0.81 ms per pop: -2.0..-2.4 ms per tree-16 block, HE+ ~+3-4 %.
   - ETA, honest: the bitwise phases (BET 7's fused2 steps ported into the persistent form, plus the format deltas) take
     ~2 days. So the 120-tree bit test lands Thu evening / Fri morning, the quiet timing needs a Fri-morning slot
     (MoonShots asks when the bit test passes), and the flag makes lossy timing #2 (Fri 23:35) if bitwise by Fri 12:00.
     If the account reaches 85 % first, MoonShots' HANDOVER carries PK to its main-account session after Thu 17:00.
   - 08:4x BUILT AND BITWISE, a day early (MoonShots d62c6b6; logs checked by NewIdeas): 120 / 120 real trees == SF (HE+ t16
     48 / 48, HE+ t32 24 / 24, MBPP t16 48 / 48), 0 fallbacks; 0.77 vs SF 0.93 ms per pop at tree-16, 0.786 vs 1.015 at
     tree-32 (share lock, indicative): -2.4 ms per tree-16 block, HE+ ~+4 %. Flag R57_TREE_PK=1 in r57_draft (default off;
     falls back to SF past 1024 keys or on a barrier timeout, same bits).
   - LANDING (NewIdeas as number checker): M-46's f thresholds (0.16 = PARTIAL) decided the build; PK lands if a quiet
     pk_timing (PK vs SF interleaved, 12 trees x 8 rounds; GO today in a W1 gap) shows it faster by more than the
     round-to-round spread AND BigMoE's loop bit test passes (tonight). Then R57_TREE_PK=1 goes into BLOB 1 (Thu ~11:30)
     and exact arm A: bitwise, so tier 1 #1's outputs are unchanged and Thursday night's timings include it.
   - c2 PK (arm B, ~1 day of format work): in arm B only if bit-tested by Thu 18:00; else the A/B compares the shippable
     forms (Q8H32 + PK vs c2 + SF) and exact timing #2 re-checks if c2 PK lands Friday.
   - 08:5x QUIET TIMING CONFIRMS (MoonShots b5c7c80): SF 0.938 -> PK 0.772 ms per pop interleaved (12 trees, 6 rounds kept;
     round spread ~0.01), -2.49 ms per tree-16 block, bitwise 12 / 12 after a harness fix (the timing harness drafted earlier
     windows on the last window's screen). It lands on BigMoE's loop bit test of the HAND-OFF commit (b5c7c80 or later;
     the 120-tree test is re-run on that commit, since 43cd82e and b5c7c80 came after d62c6b6). c2 PK waits for main's reset.
   - 08:5x RE-RUN AT THE HAND-OFF COMMIT b5c7c80 (no uncommitted .py; evidence da66380, checked by NewIdeas): 120 / 120 trees
     == SF, 0 fallbacks; R57_TREE_PK=1 through r57_draft 24 / 24 == Route A. BigMoE pins b5c7c80 into blob 1 after its loop
     bit test tonight.
   - 10:59 BIGMOE'S LOOP BIT TEST PASS (BigMoE fda65e7e; snapshot s98b, r57_draft b5c7c80, q8h32, 4 keys x 128 tokens):
     exact tokens identical host-synced / SF / SF / PK; verify logits PK == SF 60 / 60, SF == SF 60 / 60, SF == HS 60 / 60;
     check_install 18 / 18. R57_TREE_PK=1 is IN blob 1. PK has landed. (11:4x: the exact headline is contract chain-16,
     where PK / SF don't apply; PK returns to exact with exact trees.)
   Gain HE+ ~+3..+4 % (replay 204 -> 211 / 213 at 0.80 / 0.76 ms per pop; corrected 07:4x, see section 1). First gate:
   today's skeleton v3 must beat the live SF's 0.93 ms per pop at all (~14:00, share lock); if it can't get under ~0.88,
   MoonShots says so before the bitwise phases. Bitwise freebies it takes: skip the unused extra-screen rows (-inf) and
   an exact top-8 selection instead of argpartition's full sort (~11 dispatches per pop). Odds ~40 %.
2. **c2 QAT head, EXACT** - MTP Head + MoonShots + BigMoE. As set: Thursday-night A/B, A = Q8H32, B = c2 dd_t30qa
   (sha256 353965c5...), halves, 264 / 264. Without the contract it runs on chain-16, with it on contract tree-16.
   - ARM B IS FINAL: dd_t30qa (353965c5). MTP Head skipped the S22 continuation (Wed 10:3x: a second QAT pass might add
     ~1 pt on MBPP, but it could not be bit-tested and re-pinned in time), so the Thu 16 GB slot is not used.
   - CONTRACT YES (11:1x) + M4C 15 / 16 (11:4x): the A/B runs Thursday night in (ii) on CONTRACT CHAIN-16 (exact trees are
     NO-GO this week), in order-balanced halves, token identity 264 / 264 vs the contract chain's own AR. BigMoE's c2 loop
     test PASSED 11:4x (SF == HS tokens, == q8h32's exact tokens).
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
5. **The reference contract** - Kamen: **YES (11:1x, relayed by the orchestrator)**. CONTRACT_YES_PLAN.md (899926d) runs,
   merged with today's changes (section 3a): the EXACT build only; Leap's refc, BigMoE's wiring and M4C; Thursday night (ii)
   = contract chain-16 + the tree-16 Q8H32 vs c2 A/B, then Part A ~4 h (or Part A tonight, the bonus). It also makes
   ADAPTIVE K and tree drafting exact. If M4C fails: NO-GO this week, (ii) = chain-16 on NAX6-N, and (ii-b) (Thursday night, the
   same runner) is the exact headline.
   - 11:4x M4C 15 / 16 (BigMoE): under the contract chain-16 == AR on 16 / 16 and mbpp/16 is fixed, but the tree still flips
     mbpp/30's token-95 near-tie, so EXACT TREE-16 IS NO-GO this week by the rule. DECIDED (the orchestrator 11:4x): exact (ii)
     = contract chain-16 + GDN16 + tgl16e + the Q8H32 vs c2 A/B (PK / SF don't apply on chain; HC_B OFF; BOF inert). Part A
     stays and certifies the contract for chain-16. Trees only if BigMoE's 30-min diagnostic (right after the capture) puts
     the first tree-vs-chain divergence in a GDN layer: then Leap builds a tree-invariant GDN by Fri 12:00, and trees get M4C
     again + Part A's tree exactness + exact timing #2. If it is attention, exact trees (and exact ADAPTIVE K) are after
     Saturday.
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
   - RESULT Wed 15:05 (slot #4 ran by day, on the marker chain): NO-GO on all three clauses. F16nw - F16nx = -0.55 ms
     [IQR -0.78, -0.33] (< ~1 ms); expert GB +0.02; NOT deterministic (pass K re-run vs pass T: 9 / 16 rows in all 3
     windows; tgl16eu / e / ef re-run bitwise). #2a and blob 2a are OUT. Breakthrough records knob_v2's w4a8u mode as
     UNSAFE (Leap: the int8 path); the fix waits for main's reset.
8. **TRL-3** (lossy weights, 3.07-bit down, -1.4 ms). The records: the trellis family stopped 11-0x (screen 7 KILL;
   TRL-RESCUE KILL; a down-only arm was underpowered). Leap's record says the encoder is done (CPU, bitwise; a full
   store ~1.1-1.5 h on 3-4 processes). The blocker was disk; Kamen lowered the floor to 50 GB free (Wed 11:1x), so it now
   needs only a builder (BigMoE's TRL tier) and a tier-1 pass. No free builder this week. ETA after Saturday. Odds ~5 %.
9. **Sparse-dense head (GRANDMA C4)** - MTP Head. The 2B analog in its LAB today (an MoE MLP at 1/4 active vs the
   control, tree-32 kept, PASS >= x0.97, ~2 h, share lock <= 4 GB). PASS -> the 125B head (16 GB, 3-6 h) in the first
   16 GB window, then MoonShots' engine side after PK. Drafter only, so exact-safe. ETA after Saturday. Odds ~5 %.
   - FAIL Wed 10:3x (MTP Head 4e521d3, S23 SMOE): the 1/4-active MoE head loses 22-27 % kept on the 2B, like pruning.
     C4 closes.
10. **Per-check dynamic vocabulary (GRANDMA C5)** - MTP Head's CPU coverage screen: are the true token and the head's
   top-8 inside a 4-8K per-check screen as often as inside the static screen? It needs a dump with prompt ids (the bank
   or CERTIFY98's dev set). PASS -> MoonShots' per-check gathered screen in the sync-free drafter. Exact: a timing.
   Lossy: tier 1 #2b. Gain +3-5 %. Odds ~20 %.
   - FAIL Wed 09:5x (MTP Head 0b27cc4, S24 VOCABCOV): a dynamic 8K screen beats the static 8K but not the static 32K.
     C5 closes, and it leaves #2b's list.
11. **Tree size per headline** - answered in section 1: tree-16 for both.
12. **Overhead / commit + the replayed check (module 1a milestone 2), ONE lever** (item 3's replay folded in; their gains
   overlap in the same idle time). NewIdeas splits the ~3 ms (commit / host sync / Python) from BigMoE's
   data/r57/<tag>_checks.json timers once BigMoE names the fields (CPU), or from one profiled cycle in Thursday's key test.
   Breakthrough's bitwise engine test of the replayed check, rec1a_c25, is ARMED for Thu 07:30-07:50 (~20 min, share
   lock MEM 35 GB, NO-SWAP >= 43). If bitwise, the timed arm goes in Leap's first slot after; Breakthrough writes the
   installer; BigMoE wires it; bit test + both timings #2, no gate. Ceiling: the exposed part of the per-cycle time,
   ~-2 ms (+3-4 %). Odds ~35 %.
   - 11:2x OVERHEAD SPLIT (research/week98/ovh/: ovh_split.py on BigMoE's s98b loop records, 4 keys x 128 tokens, 6 arms):
     between checks, outside the check and draft timers, only ~0.05-0.2 ms per cycle, so commit + bookkeeping are not the
     ~3 ms. Inside the check timer: build 90-107 + wait 15-16 + a remainder of 3.4-6.7 ms (medians; E4pk: cycle 125.6 =
     check 112.3 + draft 13.5 + rest 0.07). These short runs are SSD-bound (535-595 misses, ~650 MB, ~65 ms of reads per
     check), so build is mostly I/O. The steady-state split needs a warm run with R57_TCPU=1 (BigMoE checks it is a pure
     counter before Thursday's key test). rec1a_c25 moves to the marker chain (section 3a).
13. **ADAPTIVE K (added; stopped Tue for time only)** - 16 or 32 rows per check by path_p, frozen tau 0.9 (ADAPTK.md,
   adaptk_fresh). At Saturday's costs HE+ +6.9..+8.7 % (exact and CC(1)), MBPP +0.1 % (section 1); Tuesday's switch screen keeps +6.9 %
   with a switch cost of up to ~8 ms. It needs the 32-row tree path in r57 (the C32one hook) + the switch. BigMoE builds
   it after Thu 17:00. Lossy: tier 1 #2b (it changes CC(1)'s outputs). Exact: under the contract (YES 11:1x), as a flag in exact
   timing #2 if it is built and token-identical vs the contract chain by Fri 12:00. Odds ~25 %.
15. **R2 + QOM_MTP tree ports (added, lossy)** - BigMoE, ~1 day after the Thu 17:00 reset. Blob 2b only if bit-tested
   by Fri 07:00 (section 1b). Odds ~25 %.
14. **Small exact items (added):** tgl16ef (exact, ~-0.19 ms; Leap's slot #4) and REF2_GDN_LAZY on trees (<= -0.5 ms;
   BigMoE's Wednesday bit test). Each goes ON in exact timing #2 when bitwise.
   - tgl16ef RESULT Wed 15:05 (slot #4): bitwise == tgl16e at 16 and 32 rows, deterministic; -0.11 ms [IQR -0.40, +0.05] at
     16 rows (not separable from 0), -0.41 [-0.50, -0.30] at 32. NOT THIS WEEK (NewIdeas 15:1x): exact (ii) is chain-16, and
     adding it would time code Part A didn't certify (s98n is pinned) for ~0.1 %. It goes into the first post-Saturday build
     and any 32-row config frozen after this.

Closed on the record, not reopened: LANDMARK-ANS (KILL Tue 09:4x), the pool drafter, prefix eyes, code || commentary,
the SLC CPU fill, 1c tiers (V2 FAIL, so G1 stays out), close-call d2 (tier 1 FAIL).

## 3. GPU windows
45 GB = the whole Mac. Every big job waits for Kamen away >= 900 s; <= ~6 GB while he is at the Mac.

| window | what runs |
|---|---|
| W1 Wed now-15:00 | BigMoE's plumbing bit tests (REF2 on the lossy loop, GDN_LAZY on trees); MoonShots' PK skeleton (share lock); MTP Head's 2B analog (share lock) |
| W2 Wed 11:37-23:30 | BY DAY: BigMoE's CERTIFY98 FULL capture (started 11:37, ETA ~13:45-14:30) -> its 30-min tree-vs-chain diagnostic -> (iii) on chain-16 if EvalSuite amends -> NIGHT1118_BIGMOE_END; the marker chain then runs in any away window. Else <= ~6 GB while Kamen is at the Mac |
| W3 Wed 23:35-Thu 07:25 | The marker chain's night part (section 3a): rec1a_c25 -> slot #4 -> blob 1's key test -> tier 1 #1 (~2 h 40 with the smoke) -> Part A (chain-16) if ready by ~22:00 and room remains, else TOPK / T8 |
| W4 | merged into W3. rec1a_c25 starts by Thu 07:30 at the latest (hard stop <= 09:30). MTP Head doesn't use its 16 GB slot |
| W5 Thu 08:00-15:00 | (Part A's tail, if it ran tonight) -> if tier 1 #1 ran tonight: 2u's tier 1 if #1 passed, or B1 (blob 1's bisect by the failing line) if it failed; else BigMoE's key test (blob 1; R57_TCPU=1 if neutral) ~11:30 -> tier 1 #1 12:00-14:30, as set. CERTIFY98's TOPK / T8 go around the tier-1 pass, never ahead of it; stage 2 here if Part A ran and needs it. Slot #4 moved to the night chain; PK's quiet timing was done Wed 08:50 (b5c7c80) |
| W6 Thu 15:00-23:30 | main resets 17:00: ADAPTIVE K, the replayed-check flag if bitwise, blob 2a OUT (w4a8u NO-GO Wed 15:05) |
| W7 Thu 23:35-08:00 | (i) lossy timing #1, blob 1 + the fallback in halves (~90 min) -> (ii) contract chain-16, Q8H32 vs c2 in halves (+ GDN16 + tgl16e; ~90 min) -> (ii-b) the NAX6-N chain-16 insurance arm (~45 min, the same runner as (ii)) -> Part A (~4 h, chain-16): ~7 h 45; a late start pushes Part A into Friday's away window. If Part A ran tonight, its place goes to #2a's key test + tier 1 if slot #4 qualifies w4a8u (amendment 6, section 8); else no tier-1 pass, and NewIdeas re-plans the time by Thu 18:00 (a 2u arm in (i) if 2u passed Thursday day, the tree-GDN path if ready, CERTIFY98 leftovers). M4C 15 / 16 failed trees only; the contract stays for chain-16 |
| W8 Fri 08:00-15:00 | stage 2 if Part A needs it (<= 3.3 h) -> blob 2b key test + tier 1 #2b -> a bisect pass only if Part A ran tonight -> exact timing #2 ~13:45, else Fri 23:35 |
| W9 Fri 15:00-23:30 | <= ~6 GB; the final lossy stack frozen 22:00 |
| W10 Fri 23:35-Sat ~07:00 | lossy timing #2 (45 min, only if the stack changed) -> card gate (5.8-6.5 h) -> verdict ~07:00 |

### 3a. Tonight and Thursday night, re-placed (Wed 11:1x; approved by the orchestrator, clock-checked)
- Why: the orchestrator postponed CERTIFY98 slim at 11:0x. Kamen then lowered the disk floor to 50 GB free, so FULL fits,
  and BigMoE runs it tonight 23:35 -> ~01:45 on NAX6-N. Kamen said YES to the contract. As first set, Thursday night was ~9 h
  ((i) two arms ~90 min; (ii) contract chain-16 ~45 + tree-16 Q8H32 vs c2 in halves ~90; (ii-b) ~45; (iii) ~30; Part A
  ~4 h) vs 23:35-08:00, so Part A would have ended ~08:35 and pushed stage 2 and #2b.
- THE MARKER CHAIN (11:2x; the orchestrator: no idle GPU while Kamen sleeps). Each step waits for Kamen's away gate and the
  previous step's end marker, so it runs whenever both hold:
  1. BigMoE: the CERTIFY98 FULL capture RAN BY DAY (started 11:37, ETA ~13:45-14:30; last night's NO-SWAP sat at 29-30 GB
     and the capture peaks at 37.7 GB) -> its 30-min tree-vs-chain diagnostic (item 5) -> (iii), which can't run as chain-32
     (tree_meta holds 16 nodes; r57 has no plain MTP chain-32), so BigMoE proposes chain-16 (kept / 16, cap 15; EvalSuite's
     call) -> touches NIGHT1118_BIGMOE_END on any end, right after the diagnostic if (iii) moves to Thursday.
  2. Breakthrough (re-armed 11:4x): rec1a_c25 (~20 min, share lock MEM 35 GB, NO-SWAP >= 43; flag check OK vs Leap b31dd1f)
     on NIGHT1118_BIGMOE_END, starting by Thu 07:30 at the latest, hard stop start + 90 min (<= 09:30) ->
     ~/Breakthrough/data/m1/rec1a_engine/REC1A_END on any end (sent to Leap). Its timed arm goes in a later slot.
  3. Leap (confirmed 11:29): slot #4 = tgl16ef (F16nf / F32c6xf bit checks) + the w4a8u pair F16nw / F16gw, ~3 min of GPU,
     quiet ticket, NO-SWAP 28 -> 26, swap guard -> ~/Leap/data/week98/NIGHT1118_SLOT4_END on any end. Thu 08:00 is freed.
     The rec1a timed arm rides only if Breakthrough hands Leap a c25 knob before it re-arms.
  4. BigMoE's key test on blob 1 (PASS / FAIL marker) -> EvalSuite's ONE chain: its smoke -> tier 1 #1 in both orders (~2 h
     40) -> Part A (chain-16) if its contract PREREG is filed by ~22:00. If the key test hasn't passed by ~00:30, Part A
     starts on the slot-#4 marker and tier 1 #1 stays Thursday. Else BigMoE's TOPK / T8 if its tool is ready.
  MTP Head doesn't use its 16 GB slot; its S22 chain on NIGHT1118_BIGMOE_END is cancelled.
- **15:1x STATUS: THE CHAIN RAN BY DAY.**
  - NIGHT1118_BIGMOE_END 14:58 (BigMoE ee8847b7): the capture PASSED 647 / 647 (frozen by EvalSuite 69b304d; contamination
    clean; scored HE+ 66 / MBPP 297 / everyday 150). Snapshot s98n = s98 + Leap ec06e43 (refc on all 7 shapes). M4C chain ==
    AR on s98n: 16 / 16 keys and 2,604 / 2,604 kept-row logit hashes. (iii) chain-16 on the first 30 gated HE+: 27 / 30
    answers equal the capture's, kept 14.18 per check (EvalSuite scores it vs kept_walk). The dev rehearsal ran (kept 11.35).
  - DIAG95 (item 5's diagnostic): from one plain prefill of mbpp/30, the tree verify, the 'chain' commit and two side-branch
    'tree' commits are bitwise to the chain route at every sublayer, and both routes pick 10598; in the loop, chain and AR
    pick 364. So it is neither GDN nor attention: the flip comes from the loop's history (check boundaries / the loop's
    prefill path). By the rule, NO EXACT TREE THIS WEEK. The cheapest localizer (after main's reset Thu 17:00; Leap and
    BigMoE's call, sent 15:1x): the mbpp/30 tree arm with R57_LOGHASH_ROWS vs M4CA's AR hashes (data/r57/post1118). The
    first differing answer index names the check; a DIAG95 sublayer diff there names the op. If it is one GDN / commit op
    fixable by Fri 12:00, Thursday night's freed slot checks the fix.
    Leap 15:13: YES (it takes a one-op fix by Fri 12:00: bitwise + the tree-mask self-test + an M4C re-run), ahead of S1;
    the localizer was sent to BigMoE 15:1x for after its reset.
  - rec1a_c25 14:59-15:02: NOT RUN (a harness bug: it read the gate's pass seq from the module, not the streamer's Gate
    instance; the stub engine hid it). Fixed in adacf34 (fake-engine test PASS, same steps and lines). RE-RUN (NewIdeas
    15:1x): today, in the first free share-lock window after 16:03. MEM 46 (43.06 measured + the 3 GB guard); start gate
    free + purgeable >= 47 (measured peak + 4, Leap's convention for this harness); away >= 900; +2 GB swap stop; 20-min
    cap. Latest start 18:30, else Thu 07:30 (hard stop 09:30).
  - Slot #4 15:03-15:05 DONE (Leap; data/week98/c25/s4_run.txt): w4a8u NO-GO (item 7), so #2a is OUT; tgl16ef is bitwise
    but ~0.1 ms at 16 rows, so not this week (item 14); tgl16eu repeats -2.70 ms (slot #3: -2.67).
  - NEXT: MoonShots' PK bundle (~16:00) -> blob 1 -> BigMoE's one-key head test -> EvalSuite's b1head (lcb6 + ifbench) ->
    the key test (~/BigMoE/data/r57/BLOB1_KEYTEST_END, '^BLOB1 KEYTEST PASS', newer than the env) -> EvalSuite's head test
    + amendment 6 -> stk1_chain.sh (tier 1 #1; latest first start Thu 00:30). Part A: EvalSuite's word by ~21:00.
  - PART A TONIGHT: YES (EvalSuite 15:1x; PREREG_refc filed f48c491; tools/refc_chain.sh ARMED, pid 92160). It starts on
    results/STK1_CHAIN_END, or from Thu 00:45 if no tier 1 #1 chain is alive; latest first start 06:00, cutoff 09:00; if it
    misses 06:00 it re-arms for Thursday night after (ii-b), and a cut resumes there. Pins: s98n at BigMoE ee8847b7 (screen
    e9a6218c, .so 5333899c), EvalSuite's own env copies. Head test 18 / 18 PASS; the dry run on the real pins reaches every
    step. End marker ~/EvalSuite/results/REFC_CHAIN_END. EvalSuite also rules tgl16ef out of Part A and (ii).
    CERTIFY98: the dev rehearsal's tokens per check are 0.985x the replay's (inside +-3 %); (iii) 22 of the first 24
    identical, the comparison waits for the TOPK dump. TOPK / T8 offered to BigMoE on REFC_CHAIN_END (latest start 05:00),
    else Thursday. So THURSDAY NIGHT = (i) -> (ii) -> (ii-b) -> the freed slot, re-planned by Thu 18:00.
- DECIDED 11:3x (the orchestrator, on NewIdeas' proposal; best effort, no corners cut; it needs the capture + (iii) by day):
  (1) TIER 1 #1 ON BLOB 1 TONIGHT if BigMoE freezes blob 1 (s98b + PK; GDN16 may join later, it's bitwise) and sends it to
  EvalSuite by ~20:00, EvalSuite files amendment 6 first, and BigMoE's key test PASSES (a PASS / FAIL marker). Thursday's
  day then takes 2u's tier 1, or blob 1's bisect by the failing line: one extra tier-1 pass before Saturday. (2) THE BONUS
  Part A (now the chain-16 certification), if ready by ~22:00 with room left after tier 1 #1 (the orchestrator 11:4x;
  EvalSuite files amendment 6 first, then the contract PREREG). EvalSuite runs (1) then (2) as ONE chain, never in
  parallel. Night order when all is ready: rec1a_c25 + slot #4 (unless they ran by day) -> the key test (~30 min) -> tier 1
  #1 (~2 h 40 with the smoke) -> Part A (~4 h, chain-16), ~8 h from 23:35, so Part A's tail may run into Thursday morning's away
  window. If the key test hasn't passed by ~00:30, Part A starts on the slot-#4 marker and tier 1 #1 stays Thursday. If
  anything is late, tonight holds (2) + (3) BigMoE's TOPK / T8 and Thursday keeps the original order.
- (ii-b) STAYS THURSDAY NIGHT (11:2x, approved): BigMoE's exact timing runner isn't built, and (ii) + (ii-b) on one runner
  keep the two exact headlines comparable. (ii-b) is the exact headline if M4C or Part A fails.
- 12:0x EvalSuite: PART A MOST LIKELY THURSDAY NIGHT; tonight carries tier 1 #1 only. BigMoE's main is at 90 %, so
  EvalSuite runs Part A's rows itself on a small new runner (runner 69's r57_wrap recipe: the 210 AR via R57_MTP_K=0 on the
  s98m contract snapshot, then chain-16 vs AR with per-kept-row logit hashes on R-56d's 36). It must be head-tested on the
  real loop, and the snapshot + flags come ~19:00. EvalSuite says by ~21:00 whether it can still run tonight. Its PREREG
  draft: results/refc/PREREG_refc_DRAFT_2026-11-18.md (68b3c2e): chain-16 only, d4x.py vs ref2tnax with stage 2, exactness
  by kept-row hashes, runner 64's tier 0 dropped. Lossy amendment 5g (fac490e) files the lineage rule. Tonight's remainder
  after tier 1 #1: EvalSuite DECLINED pulling Thursday day's pass forward (12:0x: B1 means up to 3 new blobs with pins, an
  amendment and key tests, plus an unattended failing-line classifier, which is new code). So the remainder goes to BigMoE's
  CERTIFY98 TOPK / T8 (the single look; pins re-filed as amendment 10 before the pass if the dump tools change), and Thursday
  day keeps section 8.
- BONUS (updated 11:4x): M4C ran by day: 15 / 16, so exact trees are NO-GO this week and Part A certifies the contract for
  chain-16. It runs tonight after tier 1 #1 if ready by ~22:00 and room remains (into Thursday morning's away window if
  needed); else Thursday night as set. Then Thursday night's freed slot is non-gate work (#2a is out: w4a8u NO-GO 15:05), and the
  exact chain-16 headline counts as soon as (ii) measures it.
- THURSDAY NIGHT: (i) -> (ii) contract chain-16 Q8H32 vs c2 -> (ii-b) -> Part A, ~7 h 45, a late start pushing Part A into
  Friday's away window. If Part A ran tonight: (i) -> (ii) -> (ii-b) -> the freed slot. #2a is OUT (w4a8u NO-GO 15:05),
  so non-gate work, re-planned by Thu 18:00 (candidates: a fix check if the DIAG95 localizer finds one op, a warm
  R57_TCPU=1 overhead-split run, TOPK / T8's remainder).
- THURSDAY DAY (EvalSuite's amendment 6, section 8, confirmed 11:4x): if tier 1 #1 ran tonight, 2u's tier 1 if #1 passed,
  else B1 (blob 1's bisect by the failing line). FRIDAY: #2b at 08:00 (cut 07:30), one bisect 11:15-13:45 on a FAIL, timing
  #2 + the card gate as set. Blob 1 = BigMoE 10:59's composition (PK in; BOF / HC_B out; GDN16 only if in the ~20:00
  freeze, a later GDN16 rides 2b). The fallback also drops BOF / HC_B (no engagement; EvalSuite's rule 5).
- Unchanged: blob 1's freeze + key test Thu ~11:30 (+ M4C unless it ran tonight); tier 1 #1 Thu 12:00-14:30. Kamen's away
  gates and the swap guard apply.
- Owners (11:1x lines): Leap, refc for r57's exact paths + the tree-mask self-test + the drafter check, by ~17:00; BigMoE,
  the env + snapshot + M4C + the follower behind the capture + c2's loop bit test; EvalSuite, the PREREG + flagcheck + dry
  run, and the count check tonight.
- Disk (re-checked 11:1x): in this plan only TRL-3 (item 8) and CERTIFY98's delete were disk-gated. TRL-3 now waits only on
  a builder. 101 GB free at 11:1x, so ~46 GB above the 55 GB alarm line for FULL's dumps and any night swap.
- 11:24 LEAP DELIVERED refc a-c (Leap 3ec85e8): REF2_REFC=1 (unset = today's build, bitwise). The exact set under the
  contract: contract ON, BOF inert, HC_B OFF (ON only if M4C runs with it), GDN16 + tgl16e ride, GDN_LAZY off (recipe
  ~/Leap/data/week98/check25/REFC_CONTRACT_RECIPE.md, sent to BigMoE with the snapshot files + md5s). The VOID self-test
  refc_rowinv.py: 18 / 18 at kernel level (tree masks end to end = M4C). The drafter is untouched by construction. Price:
  ~-2.6 ms per 16-row exact check vs today's BOF + HC_B build [P].
- LINEAGE (EvalSuite 11:1x for CERTIFY98; NewIdeas 11:2x for the timing; APPROVED by the orchestrator): b5's l120 train
  split holds 72 HE+ problems (+ 27 GSM; 8 HE + 2 GSM in val; no MBPP), and t30b = b5 + one pass over Magicoder capture
  night 1. So 72 of the timing's HE+ 164 are in t30b's lineage; MBPP 100 and the 24 d16w windows are clean (section 1's
  replay used those). RULE, filed before any stack data: every HE+ number (tok/s, tier-1 and card-gate deltas) is quoted
  on the 92 outside the lineage BESIDE the 164, with MBPP as the clean anchor. CERTIFY98's 72-vs-68 split sizes the effect
  on Thursday.

## 4. The second tier-1 pass, priced
- One pass = BigMoE's key test (~30 min, quiet) + EvalSuite's tier 1 in both orders (~2.5 h at 45 GB) + EvalSuite's
  amendment at the desk: ~3 h of the whole Mac. The lossy re-timing (runner 69, ~45 min) runs once, Fri 23:35, before
  the card gate, which still ends ~07:00.
- Without the contract: #2a (w4a8u alone, only if slot #4's arm clears ~1 ms) Thursday night after (iii); #2b (the bundle: c2 in lossy, ADAPTIVE K, the
  DROP, dynamic vocabulary, w4a8u if #2a passed) Fri 08:00-11:00. On a FAIL, one bisect pass
  (11:15-13:45) drops the newest lever first.
- With the contract: Thursday night is Part A, so there is no #2a, and #2b follows stage 2 (key test + tier 1
  ~11:20-14:20 if stage 2 runs), with no room for a bisect by day; exact timing #2 then likely moves to Fri 23:35.
  CONTRACT YES (11:1x), so this line holds, unless the bonus Part A runs tonight (section 3a): then Thursday night has
  #2a's slot again, and stage 2 (if needed) runs Thursday by day.
- The card gate runs the last stack that passed tier 1 and won its timing; else Thursday's stack. Thursday's stack itself is
  blob 1 or the fallback by section 1b's rule. Any passing later stack not yet timed (2u, 2b) gets its timing in a free
  quiet slot Fri by day, else as an extra arm of timing #2 Fri night (the verdict then slips ~45 min).
- EvalSuite's price check (07:4x): ~3 h 10 per pass at night (Monday night's tier 1 took 2 h 26 for both orders, plus the
  30-min key test and its 15-min smoke). By day it stretches with at-Mac yields, so #2b's 08:00-11:00 holds only if
  Kamen is away from 08:00.
- Rules agreed with EvalSuite (07:4x; in its amendment 6 draft, b881eb7, filed Thu midday before any stack data):
  1. If tier 1 #1 fails, blob 1's bisect by the failing line takes #2a's night slot (runaway -> CC(1) off; accuracy ->
     tgl16eu off, and then w4a8u, which shares tgl16eu's grouping, is out too); timing #1 times the bisect blob; 2b builds
     on whichever passed. (Refined 08:0x by the orchestrator, from 'tgl16eu off' first.)
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
- The contract snapshot + flags + EvalSuite's contract PREREG: by Wed ~22:00 for Part A tonight (after tier 1 #1); else Part
  A runs Thursday night. M4C ran Wed by day (15 / 16: chain-16 only).
- The tree-invariant GDN (only if BigMoE's diagnostic blames GDN): Leap, by Fri 12:00.
- Blob 2a (w4a8u): OUT (slot #4, Wed 15:05: NO-GO).
- Blob 2b (the bundle): Fri 07:30, holding only the flags whose bit or unit tests passed by Fri 07:00.
- Exact flags for timing #2: bit test by Fri 12:00. If Kamen is back early, exact timing #2 moves to Fri 23:35 and the
  verdict slips to ~07:35.
- The final lossy stack: frozen Fri 22:00.

## 6. Kamen's calls
- The reference contract: YES (11:1x).
- TRL-3's disk: ANSWERED (Kamen lowered the floor to 50 GB free, 11:1x); it waits only on a builder.

## 7. Usage rule (orchestrator, received ~07:40; through Saturday's 07:00 verdict)
- On the second research account, every session except EvalSuite stops at 85 % of its week (63 % on Wed morning; this
  week's average burn would empty it ~Fri 04:00, before tier 1 #2b and the card gate). EvalSuite keeps 85-100 % for the
  gates.
- Breakthrough and MoonShots move their builds to their main-account sessions after main's Thu 17:00 reset, or earlier
  if the account reaches 85 % first (a HANDOVER hand-off; one session owns each repo and its runners).
- MTP Head, on its own account, takes the head work: items 2, 9 and 10.
- NewIdeas stays coordination-only and lean; at 85 % routine coordination returns to the orchestrator.
- PRIORITY ORDER on the second account until main's Thu 17:00 reset (orchestrator, 08:4x): 1. EvalSuite's gates; 2. MoonShots'
  r57 drafter-side integration (blob 1 is Saturday-critical); 3. PK's quiet timing + its hand-off to BigMoE; 4. Breakthrough's
  overhead installer (item 12); 5. everything else waits for main's reset (c2 PK, the MoE megakernel, the DROP installer,
  anything for after Saturday) and resumes on the main-account sessions at Thu 17:00. If the account nears 85 % before
  then, items 3-4 pause before 2.
- Defaults until Kamen answered (orchestrator 07:4x): contract NO. SUPERSEDED 11:1x by Kamen's YES (section 3a): the exact
  headline is tree-16 under the contract (chain-16 on NAX6-N, measured tonight, is the insurance), and the c2 A/B runs on
  contract tree-16.
