# Breakthrough: Wednesday 4 Nov, Kamen's research day 13:45-23:30 (prepared Tue 22:3x EET)

Marks: [K] measured, [P] arithmetic or replay model, [G] judgement. SCOREBOARD hashes in brackets.

## 1. Done or running before 13:45 (no action needed)
| when | what | task | output |
|---|---|---|---|
| 22:32 -> ~23:00 | prefix: NewIdeas' three QOM arms + the records3 layout check (resumes after yields; heat pauses add minutes) | bcthuogm9 | out/qom_*.json (NewIdeas' qom_read.py); out_layout/LAYOUT_check.txt (for BigMoE and the orchestrator) |
| "prefix done" -> ~04:30 | BigMoE's night: one quiet ticket | BigMoE | its own |
| ~04:30 -> 13:00 | FLOOR night 1 (see below) | blqmrr20w | data/floor1/night.log, FLOOR_read.txt |
| 13:05-13:40 | Fit12's T3R2 | Fit12 | its own |

FLOOR night 1's steps:
- S0: the 12 GB pass-1 stack p1_oea_k1_ku4_t3, plus top_k8.
- A1 / A2: capture and gates on 78 windows.
- D: extras on the 36.
- B: masks.
- C, on the 78: q1u_cf09, q1u_cf09_t3, oea_k8, then the COLD arms (hyb_cold4_t3, the u8+t3 control, pmtc, len).
- R: read-outs.
- Leftovers resume Wed night (--skip-done).

## 2. Morning read-outs (CPU, ~10 min, once FLOOR's step C is done)
1. FLOOR verdicts: data/floor1/night.log, then FLOOR_read.txt against a277380 / ceeda3f's lines. S0 goes to NewIdeas.
2. COLD on the 42 held-out windows (verdict lines 12b3d63):
   `cd ~/FableGod/decision_map && DM_OUT=~/Breakthrough/data/floor1 COLD_HELDOUT=1 ~/BigMoE/.venv/bin/python dm_cold_lru.py && DM_OUT=~/Breakthrough/data/floor1 COLD_HELDOUT=1 ~/BigMoE/.venv/bin/python dm_cold_read.py`
   Read the PROMPT lines for pmtc (confirms or withdraws be67ff0's GENERAL) and len (decides the length-aware table) in
   data/floor1/COLD_read_heldout.txt. All 78 windows as information: the same without COLD_HELDOUT.

## 3. Research-day proposals (13:45-23:30), in priority order

**Morning update (Wed 07:1x, SCOREBOARD 45fb226).** On the 42 held-out windows the per-request table is CODE+, not
GENERAL. pmtc costs on multi (6.70 vs 5.66 %, z +2.78; IFEval z +0.92, code z -1.88), so P1 narrows to code. The
length-aware table (len: the prompt footprint alone when the prompt has >= 128 tokens) holds on code: 0.81 vs 1.11 %
(z -1.88), at cached bytes x0.196 vs u8+t3's x0.272 (-28 %) [P, 45 GB plain-LRU replay]. P1 therefore = len on code
requests only; chat keeps U3 unchanged. P2 is folded into P1. FLOOR night 1: bands BETWEEN; Q1 at 0.999 clean on code
(0/3163); oea_k8 PASS x3 at 78 windows.

**P1 RESULT (Wed 08:5x).** BigMoE's 5b (u3c_wed.sh, 12 HumanEval, A-B-A-B at C 224, R57_PRESET=45): U3C in len's form vs
U3 = x0.732 SSD MB per kept token (-27 %), steady tok/s +3.7 % (110.1 vs 106.4 at 15.0 vs 20.5 MB per kept token) = PASS
by the line (<= 0.85x AND tok/s >= U3's) [K, BigMoE data/u3c/u3c_report.txt]. Mask check of dump_224 (dm_u3c_replay.py):
PASS (1,248 records bitwise equal; warm follows the 128-token rule on 103 / 367 / 144-token prompts). Next: EvalSuite's
same-scores on code (FAST). Deployment needs a code-request switch (WED_TESTS caveat).

### P1. The per-request COLD table in the 45 GB engine
Owner BigMoE (engine). Mine: the spec, reference masks, the mask check and the read-out.
- **What it is.** On drafter-unsure rows, drop a rank 4-9 expert unless it is "warm" for this request:
  - the request's own prefill routed to it at least once, OR the corpus table (cap45 share >= 0.05) marks it popular;
  - experts already in the check's union are always kept.
  Sure rows are exactly U3: top-2 at 3 bits plus union members. It is token-repeatable and needs no domain switch.
- **Map evidence [K, 36 windows, live checks].** No measurable cost against u8 + t3 on code / IFEval / multi
  (z -1.79 / +1.63 / +0.62). It does NOT match U3's quality on chat: U3 keeps all ten experts on unsure rows. So in the
  engine it is a FAST step, and task results decide it (EvalSuite same-scores).
- **Bytes [P, 45 GB plain-LRU replay, against EXACT].** U3 x0.390 / 0.810 / 0.867; pmtc x0.285 / 0.520 / 0.539, i.e.
  -27 / -36 / -38 % vs U3. The length-aware table (len, pending the held-out verdict): code x0.199 (-49 % vs U3).
  Halved for decisions: -14 to -25 %.
- **Engine spec.**
  1. During prefill, per layer, set warm[l] = the experts any prompt token routed to (top-10). OR in the corpus table,
     prior45 >= 0.05: data/seed45/prior45.npz should be cap45's share-of-checks, i.e. Fable C's pop_cap45.npy; verify
     before use. For len: use the footprint alone when the prompt has >= 128 tokens.
  2. In the verify pass on top of U3: on unsure rows keep ranks 0-3, plus ranks 4-9 that are in warm[l] or in the check's
     union (the base sets of all rows, including the rejected ones). Kept experts use the model's own weights (no renorm).
- **Mask check (mine), agreed with BigMoE 22:4x.** BigMoE dumps one .npz per prompt (HumanEval 68, ifeval/1220,
  multi/de3) with per (check, layer) ids / sure / keep / rows, plus warm and prompt_len. I replay it:
  `~/BigMoE/.venv/bin/python ~/FableGod/decision_map/dm_u3c_replay.py <dump>.npz ...` -> "U3C MASK CHECK: PASS" when
  every record's keep is bitwise equal and warm holds the cap45 table.
- **Test.** At 45 GB, quiet lock, BigMoE's R57 prompts: SSD MB per kept token and tok/s, U3 vs U3 + table, per domain.
  Line to pre-register with BigMoE before the run: PASS = chat MB per kept <= 0.85 x U3's AND tok/s >= U3's. Then
  EvalSuite same-scores.

### P2. If len holds on the held-out windows: the code path's table is the prompt footprint alone
On code that means -33 % bytes vs u8 + t3, -49 % vs U3, at no measurable cost [36-window K + P]. It goes into P1's spec.

### P3. The 12 GB line
COLD is NOT a 12 GB byte lever. At C 51 every COLD table reads 11-18 % MORE than u8 + t3 (51c67f5): the warm set does not
fit. The 12 GB day items:
- S0's map verdict on the pass-1 stack (P1-K1 KU 4 at 3 bits; Fit12's T3R measured x1.227 kept on shared text);
- Fit12's T3R2 timing (13:05-13:40);
- the two-pass engine (Fit12; PLAN_12GB_TWOPASS.md T1-T3).
Mine: S0's read-out, plus a C 51 bytes replay of the stacked pass 1 (CPU, ~5 min).

### P4. Write-up
research/novelty_cold_cut.md gets the held-out results. NewIdeas merges them into ~/TRIED/NOVELTY.md.

## 4. Constraints that still hold
- bt_b12.py and v3p stay frozen until EvalSuite's T0_DONE (T0 moves out of Kamen's day).
- By day, one job over 5 GB at a time, with "MEM <n> GB" in the share lock. Map arms are MEM 17 GB, ~5 min each at 36
  windows and ~11 min at 78.
- Fans are capped at 5,000 rpm by day, so heat pauses happen (Tue 22:28, die 74-76 °C). Runners cap on the wall clock and
  resume.
