# FINDINGS — Fable C (auditor / inventor), running log (newest last)

Files: `AUDIT.md` (task 1), `INVENT.md` (task 2), `replay_c1.py` → `replay_c1.out` (the CPU replays, 18 s, numpy at nice 15). Rules kept: CPU only, no downloads / installs / commits / messages to other sessions; full 125B; Kamen's MacBook. Marks: [K] measured · [P] arithmetic · [G] judgement.

## Mon 2 Nov 2026, 22:3x-23:5x EET — mission C1

### Audit, one line each (details and file:line in AUDIT.md)
- **(a) Fable B's floors (EXACT ~19-21, FAST ~16-17 ms): CONFIRMED.** Every per-piece number re-derives from `c0_c160_probes.json` within 0.02 ms (GDN 2.06 build / 10.33 eval; HC 4.60 + 4.68; attention 5.82; experts 50.91) [K]; the 14 ms bytes floor re-derives from the config [P]. Soft spots: the FAST ledger sums to 17.3, printed "~16-17" [P]; ~3 ms of "sequential structure" is unmeasured in situ [G].
- **(b) Leap's NAX6 / hybrid verdicts: CONFIRMED.** The order probe's 45 % best fit with sign-symmetric errors kills a bitwise ALU replica [K]; N7-10 (three dispatches, the last named exact lever) reported KILL at 22:41 (+1.2 %) [K]. One untested lever named: register-trimming toward the next residency step (N7-8 only added registers) [G].
- **(c) The cloud's "SSD critical, only hideable, 2 ms at target": CRITICAL yes, the rest REFUTED.** The default's demand reads run at 11.7 GB/s and take 124 ms of a 171 ms code check [K c45l_256]; at C 320 the replayed 37.9 MB per kept token is 37-41 ms of transfer per check — the ladder's whole 38.5 ms cycle [K cachesize45, P]. Exact at C 320 is SSD-capped at ~310-340 tok/s on code with the drive at 100 %, and at ~70-80 on chat [K floor_bytes]. The ladder has a bytes floor booked as a hiding row.

### The three ideas (INVENT.md)
1. **C1-1 refined look-ahead (EXACT):** a second L+1 prediction from OOO's phase-1 partial output. Calibration tonight: the gate-time predictor recalls 0.665 of L+1's union on cap45's real checks [K] — room exists. Halved: −3-4 ms per cycle at C 320 night. Test: a log-only GPU probe in r57_screen (PASS recall@96 ≥ 0.85, precision ≥ 0.80).
2. **C1-2 Q1-on-miss (FAST):** on drafter-sure rows drop only the missing rank 2-9 experts. Q1's bytes by construction, far fewer flips; cashed as a lower theta: halved +3 % over Q1. Test: one map arm (PASS flips ≤ 0.05 % at 0.99).
3. **C1-3 two-stage check (EXACT):** verify the prefix, run the tail only on full acceptance. On cap45's 16-row checks the model gives +8-20 ms per check [K, overstated vs t 0.3]; halved ~+3 % today, more as the dense pass falls. Test: the cf-transplant replay (PASS ≥ +5 % modelled cycle).

### Killed tonight by the replay (pre-registered in `replay_c1.py`'s docstring) [K]
- **Coupled plane eviction (drop an expert's LSB when its base is evicted; evict dead LSBs first): KILL.** Dead-LSB share of the sidecar 0.5 % at C 256, 0.1 % at C 320; sidecar misses −0.1 %. The sidecar's 21.5 misses per kept token (C 256) are 92.5 % "upgrade" misses — the base is resident, the LSB was evicted — and only 22 % were at ranks 2-4 in the previous check (no rank-climb signal). The sidecar is a pure capacity problem; M-17's split is the optimum, and the LSB lever is FAST's (MoonShots' "top-2 at the resident tier").
- **Token-id → expert table for a draft-window prefetch (R-70D's missing predictor): KILL.** Leave-one-window-out recall@24 on the DAY9 capture: code L0-3 0.56 (vs 0.17 for the previous position's own top-10), deeper layers 0.37-0.42, chat 0.13-0.36. Covers ~half of layers 0-3's misses at ~25 % precision — the wrong reads would evict (R-57b).
- Replay sanity: my two-pool replay reproduces MoonShots' numbers (36.9 vs 37.9 MB per kept token at C 320 + M-17; 67.2 at C 256 on 16-row checks vs the engine's 62.4 on t 0.3 checks).

### Follow-ups (not done here)
- The GPU probes above belong to BigMoE (C1-1 log, C1-3 live A/B) and Breakthrough (C1-2 map arm).
- The cf-transplant replay for C1-3 needs MoonShots' `band45.py` transplant code (D-23 cf vectors); ~30 min of CPU.
- Leap: the register-trim residency test on NAX6 kernel A (AUDIT b).

## Mon 2 Nov 2026, 23:3x-00:0x EET — mission C2: audit the 350 plan (AUDIT_350.md, replay_c2.py → replay_c2.out)
- **Verdicts on the plan's terms (code, per 16-row check):** kept 12.5-13 SOFT (measured 9.6-10.5; 11.0 this week, 12.2 only with DFlash3); draft "hidden" REFUTED (DRAFT_HIDDEN: 2.3-3.2 ms exposed at best; 6.3 today); dense 17 SOFT (FAST's floor with HC8, expected 17.2-18.7; EXACT ~23); experts 17-18 REFUTED (the kernel's measured ceiling is 61 % of streaming: FAST 18-23 at θ 0.9 with H-1i + tier merge, 21-27 at θ 0.99; EXACT 32-42 by rows); SSD 2-3 ms REFUTED on both tracks and **Q1's ×0.60 REFUTED at the target sizes: ×0.88 (θ 0.99) / ×0.79 (θ 0.9) at C 320** because the LSB sidecar's 12.5 misses per kept token do not shrink with C or with the gate [K replay]; commit 0.5 SOFT (D-COMMIT Tue); CPU spans 2 and rest 1.5 CONFIRMED.
- **Corrected budget [P/G]:** FAST ~190 at 45 GB (expected), ~245 at every target; ~220 / ~270-280 at C 320. EXACT ~125 at 45 GB, ~145-155 at C 320. **FAST 350 is not reachable at any RAM with the measured kernel ceiling and gate cut; EXACT closes near 150.** The binding terms: the expert kernel (latency-bound, 10.15 µs per group), then the sidecar bytes on the SSD, then kept.
- **Three measurements:** the gated expert term live (fastBEh tonight; PASS ≤ 26 ms at ≤ 12 rows), SSD MB per kept token at C 320 with the gate (B2 + trace; PASS ≤ 28), kept with COPYGATE + D-23x (PASS ≥ 11.0).
- **New idea: Q1-UNION** (FAST, deterministic): on sure rows drop only experts no other row needs; 52 % / 45 % of Q1's dropped pairs are union members [K], so bytes and groups stay Q1's while sure rows keep ~6 of 10 experts. It replaces C1-2 (same bytes, no cache-state dependence). Test: one Breakthrough map arm (PASS flips ≤ 0.03 % at 0.99).
- Replay sanity: my SHORT-check EXACT at C 256 is 54.5 MB per kept token vs band45's 63.1 (other seeds); the ratios match band45 (×0.81 / ×0.68 vs ×0.77 / ×0.65) and the union GB per check matches to 3 %.

## Tue 3 Nov 2026, 00:0x-00:3x EET — mission C3: the 50 GB route to 350 and the way back (ROUTE_50.md; replay_c3.py → .out, budget_c3.py → .out; 3 Python runs)
- **At ~50 GB the SSD stops being the longest pole, not an exposed cost [K replay_c3, P]:** C 320 + M-17 EXACT 30.8 MB per kept token, FAST Q1 θ 0.9 24.3, sidecar dropped 14.7 (joint +2.07 GB at 52.6 GB: 27.8 / 22.0 / 13.8); per code check 15-31 ms of transfer, of which today's engine hides 38 % (HumanEval/68: 45 of 124 ms) → 9-19 ms exposed. The joint sizes at equal RAM are within noise of M-17 here (MoonShots: −1.4 / −2.8 %); the +2.07 GB step is −10..−12 %.
- **FAST at 50 GB:** ~188 tok/s expected at kept 12.6 (170 at 11.0) with NAX6-NH + Q1 θ 0.9 + tier merge; ~225 at every target; the sidecar drop → 207 / 241. The chain floor is 46.3 ms (OOO two calls) / 41.8 (one call) → 272 / 302 if the SSD were fully hidden. **FAST 350 is 12-20 ms per check (or ~2 kept) short at every target.** EXACT at 50 GB: 124-140 expected with NAX6-N on the NAX reference (110-121 on 3c), 145-163 at targets: closed.
- **Back to 45 GB (44.1 GB = joint C 256 +2.07):** FAST 175-194 expected / 207-230 at targets with Q1 θ 0.9 + the sidecar drop; EXACT 112-123 / 134-148: ~10-12 % under the 52.6 GB point. Lever order: Q1-UNION (θ 0.99 ×0.82, θ 0.9 ×0.68 of EXACT bytes), the sidecar drop (×0.66 again; frees 3.3-4.6 GB), joint +2.07 GB (−12 %), the 11-bit pack (×0.891, disk-blocked), elastic slabs (PURGE-1), kept.
- **Model calibration:** the cycle model reproduces HumanEval/68's measured 187 ms at 183; on the six-prompt mix (short math checks) the SSD's burst efficiency is lower than 11.7 GB/s, so the mix's cycles run ~20 % above the model: the numbers are code numbers, as 350 is.
- **Kamen's calls listed with costs:** the NAX reference + which store to convert (reversible GPU repack first), the sidecar drop's quality bar (sure rows only; all rows was KILLed at 24 GB), FAST adoption (same-scores, not bitwise), the night RAM (52.6 GB needs ~55 GB free).

## Mission C4 (Tue 3 Nov, 16:59-17:19 EET): THE UNION STACK -> the 350 stack (UNION_STACK.md; replay_c4.py/.out, budget_c4.py/.out, pop_cap45.py -> masks/pop_cap45.npy; 5 Python runs, no agents, no GPU)
- Union arms replayed at LIVE lengths on cap45 (8.4 rows / 5.9 kept per code check): HYB 0.9 u8 + t3 reads x0.46 of EXACT's bytes at 45 GB / x0.65 at C 320 and halves the kernel's groups (x0.50); Q1-UNION 0.9 x0.68 / x0.83, groups x0.67; OEA k8 x0.80 / x0.88, groups x0.85; U3 x0.56 / x0.70.
- Budget (code, kept 12.6, targets): the best measured arms 257 tok/s at 45 GB / 265 at 50.5 GB (C 320 via the volatile tier); one-call OOO 291 / 297; left 6-7 ms = the exposed SSD (bytes) -> and, per Fable A's split, ~17 ms of fixed per-call SSD cost that the union stack does not cut (calls, not bytes). At kept 10 the chain alone (35-36 ms) exceeds the 28.6 ms budget: 350 on code needs kept >= 12.2 (row wall) AND the SSD hidden. Chat (kept ~3): 56-63 tok/s; 350 needs kept >= 17: the row wall only.
- New cuts: COLD-CUT (deterministic popularity-keyed drop on unsure rows): code x0.35 / x0.51 at u8's pair count; in-domain prior (math) -19 % bytes vs u8 + t3. MISS-CUT (cache-state twin) x0.33-0.35, not repeatable. MASS-CUT KILLED: the renormalised top-10 router mass is flat (0.185 ... 0.071), so top-p = top-k (recomputed from cap45 x + routers.npz; top-10 overlap with the recorded ids 0.70, layer 0 missing).
- EXACT: 3c-N 134 / 155, NAX6-N 157 / 186 at 50.5 GB; closed on the expert kernel + dense. No union cut is exact; the exact use is staged verification (Fable B).
- Screen pre-registered: HYB-COLD 4/0.05 + t3 masks on Breakthrough's map (PASS <= 1.34 % code flips, BETWEEN <= 2.0, KILL above; 50/35/15 predicted).
- Clock note: the draft was complete at 17:16 by the file clock, not "17:5x" as first labelled.

## Mission C5 (Wed 4 Nov 2026, 00:00-00:35 EET): BYTES10, from ~62 to <= 10 MB of SSD reads per kept token (BYTES10.md; bytes10_replay.py -> .out/.json, bytes10_bonus_masks.py -> masks/; 6 CPU Python runs at nice 15, no agents, no GPU, no messages)
- Floor replay on the map's 36 windows with the ledger's conventions (plain LRU, prompt warm-up, one session per domain, live cuts) plus OPT, pinning, rank-class pools, compulsory floors, a mixed session, synthetic kept with the drafter's own cf, and bonus-row variants. The ledger itself had not run (starts ~04:30; a 06:47 self check-in compares its rows with section 2).
- **45 GB is not a cache problem:** EXACT 18.7 / 15.4 / 12.6 MB per kept token in a warm session (code / IFEval / multi), compulsory per request 14.7 (decode) / 43.8 (with the prefill's reads); W1B's 62.4 [K] = process bytes at the live kept: 43.8 x 7.91 / 5.9 = 59 [P]. The domain mix adds 5 %. At 24 / 12 GB the bytes are capacity misses: EXACT 96 / 93 / 83 and 186 / 208 / 174; OPT halves them; pinning and rank-class pools take <= 13 % of that gap (KILL).
- **Ledger audit:** only the record format multiplies cleanly; kept per check is FLAT for EXACT with a perfect drafter (24 GB: 72.7 at k 4, 74.5 at k 16) and acts only through wasted rows (x1.1-1.8) and, for FAST, the bonus row; QOM's cut is x0.18 / x0.34 / x0.30 at 45 / 24 / 12 GB (never carry a ratio across pool lines); t3 is x0.25 at 45 GB but x0.62-0.69 at 24 / 12; COLD / u8 / the two-pass check overlap with QOM and the bonus gate (same rows).
- **FAST floor with today's levers:** QOM cf 0 + t3 = 32.6 / 46.2 / 45.8 at 24 GB, 55.9 / 97.2 / 93.0 at 12 GB [P].
- **New mechanism, the GATED BONUS ROW:** the one ungated row of a certainty-gated block holds 34 / 65 / 71 % of the block's expert pairs at cf 0; drafting one more token so the gate covers it gives x0.66 / x0.48 / x0.47 at 24 GB (21.5 / 22.3 / 21.3) and x0.68 / x0.49 / x0.44 at 12 GB. Web + NOVELTY: nothing budgets the bonus position (nearest: PSR s7 / AcceptMoE weight positions by reach); tag new. Masks for the map screen built (masks/qom_0_live_bonus, qom_0.5_live_bonus5, q1u_0.9_live_bonus9).
- **Ranked path at 24 GB:** bonus gate -> sure rows at top-1 + union (11.6 / 12.5 / 11.7) -> 11-bit scales = 9.6 / 10.3 unhalved, 19 / 24 halved; then a learned replacement policy (OPT bound x0.5) and 2-bit sure-row records -> ~13-17 halved. 12 GB: 16.6 / 20.9 / 17.7 unhalved, ~24 halved; <= 10 there needs the resident set to hold the sure rows' top-2 working set (115 / 85 / 80 per layer vs 51 slots). EXACT: 24 GB ~45-60, 12 GB ~90-120 [P]: physics at top-10 routing.
- Three screens for 13:45 pre-registered in section 5: the gated bonus row on the map (two arms, masks ready), sure rows at top-1 at live cuts, the W1B 24 GB baseline's calibration against the replay (CPU).

## Mission C5 correction (Wed 4 Nov 2026, 06:47-07:3x EET; the self-scheduled ledger check-in)

- **My 00:35 BYTES10 levels were 2-4x too low.** `bytes10_replay.py` line 135 built the keep mask on the layer axis
  (`keep[sure, :ks]` on a [rows, 48, 10] array): base records were read on layers 0-9 only. Found by NewIdeas at 03:4x, who
  ran a fixed copy (`~/NewIdeas/data/ledger/bytes10_replay_fix.out`) and corrected their TEN_X_LEDGER.md. Fixed here at
  06:5x; my rerun (306 s) is identical to their output to the last digit. The 00:35 outputs are `bytes10_replay_bug.*`.
- **Corrected levels (code / IFEval / multi, MB per kept token, my count; the ledger's before-block count in brackets):**
  EXACT 45 GB 60.4 / 38.1 / 33.2 (55.4 / 37.7 / 33.0) = W1B's 62.4 [K] within 10 %, no kept scaling; 24 GB 245 / 210 / 183
  (213 / 200 / 176); 12 GB 637 / 622 / 487 (461 / 546 / 445). FAST QOM cf 0 + 3-bit: 24 GB 61.7 / 86.1 / 83.1; + the gated
  bonus row 30.8 / 20.0 / 14.2 (x0.50 / x0.23 / x0.17); + top-1 on sure rows 16.8 / 11.8 / 7.9; + rank-class pools 8.8 / 8.7
  / 6.9; + 11-bit 7.3 / 7.2 / 5.7 unhalved, ~25 / 33 / 32 halved. 12 GB: 24.3 / 26.1 / 18.6 unhalved, ~65 / 104 / 89 halved.
- **Ledger comparison:** after the fix the two independent replays agree within 10-15 % at 45 / 24 GB on every shared row;
  the only > 20 % difference is 12 GB EXACT (x1.38): in-loop vs before-block miss counting. The ledger's convention is the
  reachable one (resident members first); BYTES10.md's tables carry both and read the ledger's as the floor.
- **Verdicts that changed:** "45 GB is not a cache problem" -> 2/3 per-request footprint, 1/3 capacity (OPT 23.9); QOM's cut
  is domain-, not pool-dependent (x0.22 / x0.25 / x0.21 on code at 45 / 24 / 12 GB); pinning and rank-class pools: KILL on
  EXACT stands, withdrawn on the FAST stack (RCP x0.52 / x0.74 / x0.87 at 24 GB); EXACT floors 24 GB ~90-120, 12 GB
  ~250-400; the 45 GB metric call (process vs decode bytes) matters less: both ~60.
- **Verdicts that survive:** the mechanism ranking, the gated bonus row's novelty, kept per check flat for EXACT, wasted rows
  x1.25-1.8, 24 / 12 GB as capacity misses with OPT ~x0.5, the working sets, union growth and bonus-row shares, the masks.
- **The gated bonus row ran on the map overnight** (NewIdeas / Breakthrough): QOM form PASS code (z -0.28), BETWEEN chat
  (+1.56 / +1.49 points, under Q1 at 0.99's level); union-only form PASS / BETWEEN / KILL. With the gate file's own cf for
  the row (NewIdeas' note: my next-segment proxy was one position too far) it is a cf-0 lever: 1,368 / 165 / 7 bonus rows
  gated at cf 0 / 0.5 / 0.9. Rebuilt the two threshold arms as `masks/q1u_0.9_live_bonus9g` and `masks/qom_0.5_live_bonus5g`
  (null checks); `bytes10_bonus_masks.py` takes a third argument `g` (default) or `next`.
- **Lesson for my replays:** assert the mask's rank axis against one block's true union before the first table (a one-line
  check would have caught it at 00:27). Cost of the pass: ~0.3 % of the week (over the 0.1 % planned: the correction
  rewrote the deliverable).

## Screen 1(b) / (c) results (Wed 4 Nov 2026, 19:38-20:3x EET; Breakthrough's map run, my CPU read)
- **Results [K map, ba53ede]:** (b) `q1u_0.9_live_bonus9g` PASS within +0.1 on all three domains (7 rows: the null result).
  (c) `qom_0.5_live_bonus5g` COST +0.12 / +0.20 / +0.28 points (McNemar 5 / 1, 48 / 41, 33 / 23; pooled z +1.71).
- **What the rows say (`bytes10_bonus_cfcheck.py` -> `.out`):** the 1,368 realized bonus rows are (A) cut inside an accepted
  run 37 / 94 / 76, (B) cut on the check's first miss (the gate value is the drafter's confidence in a rejected token) 68 /
  386 / 623, (C) end of a fully kept check, no draft 83 / 1 / 0. Joining the arm's argmax with the baseline's (reproduces
  the read-out's 16 / 98 / 68): gated bonus rows flip at 9.7 % vs 4.0 % for the same kinds left exact (+5.7 points per
  gated row, z 3.3); QOM-gated drafted rows 0.2 / 0.8 / 0.5 %; first-miss rows with cf >= 0.5 under QOM's own resident keep
  1.7 / 4.6 / 2.5 % = the exact first-miss rows' 3.9 / 4.6 / 2.8 %. The cost is my mask's union-only treatment, not the
  bonus row's cf.
- **Verdicts:** the lever (gated bonus row) is unchanged: its quality numbers were already the resident-keep form's (PASS
  code / BETWEEN chat at cf 0). Closed: the union-only form at any threshold; a cf 0.3 resident-keep form on chat
  (~x0.74 / x0.61 / x0.76 [G], no better than COLD). Open as before: EvalSuite's same-scores test of `qom_0_live_bq` on chat.
- **Two lessons:** (1) my pre-registered alternative ("the gate file's cf and the walk's disagree") could never be true:
  `dm_gate.py` builds one from the other; check that two quantities are independent before pre-registering them as a
  contrast. (2) A per-row join of the arm's argmax with the baseline's, by row kind, is a 10-second CPU read that turns a
  COST verdict into a mechanism; do it before passing any screen result on. Cost of the pass ~0.15 % of the week; CPU only.
- **For the orchestrator:** a findings note in the README's shape is drafted at `fablec/FINDING_gated_bonus_row.md` (with
  the TRIED line at its end); my brief limits writes to fablec, so it is not in `~/FableGod/findings/` or `~/TRIED/`.
