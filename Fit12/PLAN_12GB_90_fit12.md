# 12 GB LOSSY to 90-130 tok/s: Fit12's section (Thu 5 Nov 19:4x; for Breakthrough's ~/Breakthrough/PLAN_12GB_90.md)

Goal (Kamen 19:4x via the orchestrator): 12 GB LOSSY at 90-130 tok/s (today ~57 on code), then publish. Constraint: code
quality (T0, the top-2 two-pass at 12 GB: HumanEval+ 80.0 vs 91.7 = KILL, IFEval -6.0; math PASS; memory 10.84 GB PASS).
Fit12 cannot write in ~/Breakthrough (standing rule), so this file is for Breakthrough to merge or link.

## Where the 57 tok/s check goes [K: T3R2's run, Wed 4 Nov 13:05, /Users/kamen/Fit12/data/scr/t3r2/p47_layers.txt]
Arm t80ck1u (P1-K1 pass 1, KU 4; certificate theta 0.8 + cf gate 0.9; the exact pass at REF2-T's tiers 4433333333, which is
not REF2-T's tokens: the 12 GB loop is verified greedy, corrected Sat 7 Nov), HumanEval prompts, quiet lock.
Per check ~261 ms for ~15 kept tokens (57.3 tok/s):
| part | ms per check | share | inside |
|---|---|---|---|
| draft | 13 | 5 % | 11.1 ms per draft phase |
| pass 1 (P1-K1, 16 rows) | 130 | 50 % | loader wait 53, compute gaps 71, edge 6; 341 misses, 461 MB |
| exact passes (46 % of checks, 1.56 passes each) | 105 | 40 % | wait 75, gaps 26, edge 4 per check; 568 misses, 1,072 MB per pass |
| commit | 12 | 5 % | |
Exposed SSD wait ~128 ms (49 %), compute ~97 ms (37 %). The SSD time is all exposed (1.23 GB per check at ~9.6 GB/s).

## Levers (factor = check-time arithmetic on the table above, tokens per check held)
1. EXACT-PASS DIET (FAST; Fit12 builds it, on its own T3 two-pass code): exact passes are 105 of 261 ms. Halving their
   cost or count = x1.25; removing them = x1.67 (T0-like, KILL on code). Routes: (a) a certificate calibrated on code so
   fewer rows go exact (today 46 % of checks); (b) a mid pass for the moderately unsure rows (4-bit top-4 of those rows
   only) before any full REF2-T pass. First test: CPU, from T3R2's logged checks: why rows go exact (margin, cf, which
   rows) and what each threshold would save. Then a T3R2-style A/B (16 HumanEval prompts, ~20 min quiet) and a map screen
   on code before EvalSuite. Expected x1.15-1.3.
2. PASS-1 COST (exact kernels + cache): pass 1 is 130 ms, 71 of them compute for 16 rows x ~2.4 experts. Kernel levers
   (Leap's NAX / FAST-NAX at 12 GB) and keeping likely rank-0 experts resident (pass 1 reads only the rows' rank-0
   experts: 341 misses per pass). Expected x1.1-1.2. First test: Leap's kernel arms on the T3R2 harness; a residency replay
   of rank-0 misses (CPU).
3. TOKENS PER CHECK (drafter): every component is per check, so +20 % kept per check = ~x1.15-1.2 after the extra rows'
   misses. Owner: MTP Head (P-OFF / D98), DFlash2. First test: kept per check on the T3R2 prompts with the new drafter.
Stack: x1.15-1.3 x x1.1-1.2 x x1.15-1.2 = x1.45-1.87, so 57 -> 83-107 tok/s on code. 130 needs more: the exact passes
nearly gone with code quality intact, or the SSD wait hidden (12 GB look-ahead prefetch was KILLED in its frugal form:
TRIED 'PREFETCH (frugal)'; recall@10 0.61-0.73).
Not levers here: the trellis / data-aware stores (screen 7 KILL, family stopped 4 Nov); HALF-K (a compute lever: whole
experts still stream from SSD).

## Lever 1, first numbers (Thu 5 Nov 19:5x; /Users/kamen/Fit12/scripts/t3_exact_split.py on T3R2's own logs, CPU)
t80ck1u, 16 HumanEval prompts, full outputs (44.9 tok/s; the 57.3 is the shared-text prefix), 357 checks:
| check mode | checks | kept per check | ms per check | MB per check | time | tokens |
|---|---|---|---|---|---|---|
| pass 1 certified | 55 % | 12.7 | 158 | 614 | 41 % | 72 % (~80 tok/s inside) |
| pass 1 + exact pass 2 | 28 % | 7.3 | 308 (pass 2 163) | 1,655 | 41 % | 22 % |
| cf gate -> exact check | 17 % | 3.4 | 236 | 1,778 | 19 % | 6 % |
So 59 % of the time buys 28 % of the tokens; the exact parts read 1.7-1.8 GB per check. Today's certificate = max softmax
>= 0.8 per row + the draft's cf >= 0.9; a sharper certificate that certifies more rows at the same divergence count vs
REF2-T is the lever (halving the slow modes' extra cost = x1.25-1.35).
First test (built, waits for a GPU slot): arm g1k1u + F12_CAL=1 in /Users/kamen/Fit12/scripts/fit_t3.py: t80ck1u's
pass 1 on every check, then the exact check (outputs = ex), logging per row p1 / top-2 margin / argmax / the draft's cf
beside the exact tokens (~8 min, 12 GB, share lock). Then an offline replay (CPU) prices any (theta, margin, cfmin)
rule: tokens per check, time per check from the table above, and divergences vs REF2-T at the first draft miss.

## Lever 1 RESULT (Thu 5 Nov 19:52; calibration run 19:40:55-19:51:37, 640 s, rc 0, swap flat; replay t3cal_replay.py)
- Run: arms g1k1u (P1-K1 pass 1) + g1k2 (top-2 pass 1), each = pass 1 on every check, then the exact check; 16 HumanEval
  prompts, 343 checks per arm; all 32 runs = ex tokens (as designed). Data /Users/kamen/Fit12/data/scr/t3cal.
- The time model is calibrated: today's rules replay to 45.8 (t80ck1u) / 40.2 (t80ck2) tok/s vs 44.2-44.9 / 40.8 measured
  on full outputs (T3R2).
- The certificate is already token-exact: committed tokens differ from the exact model's in 0-1 of ~3,400 at every theta
  and cf gate. So a sharper certificate has nothing to trade (lever 1 as framed: KILL). The map's damage is the context
  (pass 1's KV in certified rows; Breakthrough 19:4x).
- Speed by theta, cf gate 0.9 on (full outputs; modelled):
  | pass 1 | theta 0.8 | 0.9 | 0.95 | 0.98 |
  | P1-K1 (t80ck1u's) | 45.8 | 41.0 | 37.9 | 35.3 |
  | top-2 (t80ck2's) | 40.2 | 37.6 | 35.9 | 33.4 |
  Without the cf gate every point is ~8 % slower.
- With the map's code line (1.34 %): top-2 at theta 0.95 (1.22 %, PASS) is the code-valid point the map has: ~36 tok/s
  full outputs, ~20 % under t80ck1u's 45.8 (whose map screen is pending; Breakthrough expects it over the line).
- Next: price a wider pass 1 (map: top-5 at theta 0.9 = 0.92 %, PASS, 89.8 % of rows certified). It needs a new pass-1 form
  in t3_twopass.py (ranks 0-4) and its cost at 12 GB. The replay prices it once those are measured.

## Lever 2 (the wider pass 1, Breakthrough 19:5x) RESULT (Thu 5 Nov 20:0x; top-k form built in t3_twopass.py _TOPK)
- Form: each row's ranks 0..k-1 at REF2-T's tiers (0-1 at 4 bits, 2..k-1 at 3 bits), top-10 mass weights, no renorm; the
  map's topk sources use the same weights at 4 bits (Breakthrough 20:0x: no switch needed). Needs BIGMOE_MAXK=80 (the
  pinned engine's default 40 < k x 16 ids; supported to 92; Breakthrough's B12 variants run 80).
- top-4 / top-5 cannot run as one 16-row pass at 12 GB: the per-layer union reaches 54-80 experts > the pool's C_m 47
  slots ("bigmoe: 54 experts > C=47 at layer 0", abort 20:00:28, run c52, after its first top-3 prompt = ex).
- top-3 calibration (c53, 20:01-20:06, 305 s, 16 / 16 = ex): pass 1 ~207 ms (G1 mode) vs ~170 for top-2.
  Speed (full outputs, cf 0.9) at theta 0.8 / 0.9 / 0.95 / 0.98:
  | pass 1 | 0.8 | 0.9 | 0.95 | 0.98 |
  | P1-K1 | 45.8 | 41.0 | 37.9 | 35.3 |
  | top-2 | 40.2 | 37.6 | 35.9 | 33.4 |
  | top-3 | 36.3 | 34.5 | 33.1 | 31.3 |
- Verdict: a wider pass 1 LOSES speed at 12 GB (its extra reads cost more than the extra certified rows save). The
  code-valid two-pass ceiling on these prompts is ~34-38 tok/s full outputs, whichever narrow form passes the map.
  90-130 needs fewer bytes per kept token or a cheaper exact pass, not a pass-1 width. Options for top-4/5, if a map
  result ever needs them: 2 x 8-row pass-1 calls (slower) or an adaptive per-row width (needs its own map arm).

## Lever 3: Leap's Friday kernel switches on the 12 GB path (Fri 6 Nov 09:4x; the orchestrator 09:3x)
- Code read: the 12 GB T3 path (pinned bt_b12 8db71cc537 + v3p tier engine + MTPLX's own dense and attention calls + the
  d7b_live drafter on MLX's quantized_matmul) reads NONE of Leap's switches as they are: it loads no ref2, d7b_nax or r57.
  - REF2_DN_KS rides ref2's native dense path (qmm16n, +~1.5 GB native copy: too big for 12 GB).
  - D7BNAX_32 is a d7b_nax switch; my drafter block is 16 rows, so only its <= 16-row rule (qmm16c + per-shape ks) applies.
  - att_single / fs4 are ref2 CFG switches. MTPLX head-chunks the verify SDPA for 3-32 query rows (16 rows: 12 calls).
  - The exact check runs <= 4-row calls on MLX's kernels (B12's invariance), so attention / dense ports can only act on
    pass 1, the FAST part. FAST-NAX (R57_NAX / REF2_NAX_1CQ) has no counterpart: v3p's experts are gather_qmm on records3m.
- Ported in my copy (/Users/kamen/Fit12/scripts/f12_kports.py; pinned kernels /Users/kamen/Fit12/pinned/leap_1106):
  dnax = the drafter's 4-bit projections on qmm16c + per-shape ks (exact for the target: drafts are verified); att1 = pass
  1's 16-row SDPA as one call; dn1 = pass 1's 4-bit g64 dense (N >= 1024) on qmm16c + per-shape ks (no extra memory).
  GPU self-test: qmm16c+ks vs MLX 0.8-7e-3 max relative; one SDPA == the 12 chunks bitwise at 300 keys, not at 700.
- PRE-REGISTERED (before the run; share lock, ROUGH timings): arms ex, exd, t80ck2, t80ck2a, t80ck2n, t80ck2f on T3R2's 16
  HumanEval prompts (256 tokens, C_m 47 / C_l 32, F12_ALT=1).
  - exd: tokens = ex on 16 / 16, else a bug. Its price = the draft phase's ms per check.
  - A FAST port PAYS if its arm's tok/s is >= +3 % against t80ck2 (geo mean per prompt) with the saving visible in pass 1's
    ms per run; NOTHING if < +1 %. Quality stays the map's question (the ports change pass 1 only).
- kp1 (09:42) was stopped by my swap guard (+1.08 GB in 50 s beside a 39 GB job); kp2 waits for a clear memory slot.

## Lever 3 RESULT (Fri 6 Nov 13:2x; kp2 ran 12:59:03-13:17:31 under a share lock, rc 0, swap flat 11.99 -> 11.93 GB, 96 / 96 arm outputs; ROUGH)
- Data /Users/kamen/Fit12/data/scr/kp2 (kp2.txt, kp2_read.json = all 16, kp2_read_skip.json = 14 clean); reader
  /Users/kamen/Fit12/scripts/kp1_read.py (KP_SKIP=...). Breakthrough's never-wait self-check ran beside the run 13:00:50-
  13:02:15 (full GPU; it told me): prompts HumanEval/42 and /68 are left out of the numbers below (their arms fell in it).
- 14 clean prompts, full outputs (256 cap), C_m 47 / C_l 32, lifetime 11.39 GB in every arm (the ports add no memory):

  | arm | tok/s | vs base (geo / prompt) | kept | ms per check: draft / pass 1 / exact / commit / total | pass 1 per run |
  |---|---|---|---|---|---|
  | ex (exact loop) | 18.58 | - | 9.71 | 11.2 / - / 510.7 / - / 522.5 | - |
  | exd (+ drafter port) | 18.93 | x1.020 (warm-order bias) | 9.67 | 8.4 / - / 501.6 / - / 510.6 | - |
  | t80ck2 (FAST base) | 38.72 | - | 9.32 | 11.1 / 124.4 / 91.1 / 11.0 / 240.6 | 156.9 ms, 935 MB |
  | t80ck2a (+ att1) | 39.16 | x1.013 | 9.32 | 11.1 / 121.9 / 90.9 / 10.9 / 238.0 | 153.8 |
  | t80ck2n (+ dn1) | 40.28 | x1.054 | 9.22 | 11.2 / 114.5 / 88.9 / 11.3 / 229.0 | 139.7, 908 MB |
  | t80ck2f (all three) | 40.10 | x1.047 | 9.17 | 8.1 / 112.5 / 94.1 / 10.9 / 228.7 | 139.2 |

- Per phase, paired by prompt (the cleaner read: the FAST arms write different texts, so their tok/s ratios spread
  x0.59-1.47): dn1 pass 1 -17.5 ms per run (median, lower on 14 / 14); att1 -1.0 (10 / 14, noisy); all three -15.8 (14 /
  14); the drafter port's draft -3.2 ms per check (13 / 14), -3.0 inside t80ck2f (14 / 14). SSD bytes unchanged.
- Engaged per run: dn1 ~181 qmm16c calls per pass 1 (N >= 1024; ~496 other quantized calls stayed on MLX: 8-bit, N < 1024,
  not broken down); att1 ~12 per pass 1 (one per attention layer); dnax every drafter projection of <= 16 rows (ks > 1).
- VERDICTS (lines above): dn1 PAYS (+5.4 %); att1 BETWEEN (+1.3 %, ~nothing: 16-row attention at 300-600 keys is small
  here); all three PAYS (+4.7 %, no better than dn1 alone within noise); the drafter port is ~-3 ms per check (+0.5 % on
  the exact loop's 522 ms check, ~+1.3 % on t80ck2's 240 ms). The 12 GB check is bytes-bound (t80ck2 ~1.47 GB per check,
  ex ~4.2 GB), so dense kernels move it ~5 %.
- exd = ex on 14 / 16 (12 / 14 clean), against my pre-registered 16 / 16. Not a port bug: ex itself is reproducible (kp2's
  ex = Thursday's c51 on 16 / 16), and both divergences follow the known keep-1 replay dependence (WV3, Oct: MTPLX's commit
  replays the PLE layer at width 1 when a <= 4-row call keeps 1 row, so B12's exact tokens depend on the draft pattern at
  near-ties). HumanEval/17: exd has extra keep-1 commits at tokens 133 / 146, divergence at 178; HumanEval/77: ex has an
  extra one at 28, divergence at 134. PR1 (2-row replay, PASS in Oct: draft-policy-invariant) is not in the pinned engine,
  so any drafter change on the 12 GB exact line moves tokens at near-ties until PR1 is installed. My 16 / 16 line forgot it.
- CORRECTION for the shared plan: t80ck2 is not the exact mode. It commits pass-1-certified rows (theta 0.8, cf 0.9) without
  the exact check (= ex on 5 / 16 here), and the map scored top-2 at theta 0.8 a code KILL (1.63 %). The 12 GB exact loop
  (ex) runs 18.4-18.9 tok/s on HumanEval (T3t 18.9 and T3R 18.42, both quiet; kp2 18.58 share). The 40.8 quoted as the
  12 GB exact line is T3R2's t80ck2 full-output number.

## What the 12 GB exact loop (arm ex) is exact to (Fri 6 Nov 21:3x; fact-check for FULL's paper, from code + records, no run)
- ex = DFlash2 L1-20 (4-bit) drafts 15 tokens per check; the exact check verifies them in <= 4-row calls (4 rows nearly
  always), stopping at the first miss. Pinned bt_b12 8db71cc537 with the suite's env (B12_LATE3=2,10): multi-row calls take
  the verify flags FL_MR, every one of the top-10 ranks read and waited for (NF = 10; bt_b12.py l.370-373, 459-463). Tier
  plan 4433333333: ranks 0-1 at 4 bits, ranks 2-9 at 3 bits (MSB-3). No rank is late in ex.
- The late rule (cold MSB-3 ranks 2-9 read after phase A and added half a layer late) is the 1-row path with B12_ALLWAIT=0
  (the default): AR decode, and in ex only a possible 1-row tail call at the token cap, where drafts are truncated.
- So ex is not bit-identical to the line's one-token decoding. The 1-row path keeps the late rule (pool-dependent bits)
  and uses decode-width arithmetic (WV1: widths 2-4 are bitwise equal; WV3: a width-1 forward differs). What holds: ex is
  deterministic and independent of pool state (Fri kp2 ex = Thu c51's g1 arms on 16 / 16), and every committed token is
  the greedy argmax of the 12 GB computation at 2-4 rows per call. Near-tie tokens depend on the draft pattern through
  keep-1 replays until PR1 (Lever 3 RESULT).
- Amendment (Fri 21:3x, found while testing the new reader on kp2's log): the harness also recorded a 40-s PAUSE (171-211 s
  into kp2), inside HumanEval/68's arms (its t80ck2a took 48 s). /68 was already left out, so the numbers above stand.

## 12GB-CLEAN (pre-registered Fri 6 Nov 21:3x; the orchestrator 21:3x, NewIdeas QUEUE f2243e8; runs Saturday morning)
- One clean number for the release: the 12 GB VERIFIED-GREEDY loop (arm ex; not exact) ALONE in its own process, so its
  lifetime peak footprint is its own. Replaces the mix of four runs and subsets (18.38-18.9; 11.39 GB from a six-arm process).
- Run: /Users/kamen/Fit12/scripts/ex16.sh (task b8qfeow0m, armed 21:37) -> fit_t3.py arm ex only, no ports, F12_RERUN=1,
  all 16 of T3R2's HumanEval prompts, greedy, cap 256, C_m 47 / C_l 32, quiet lock (~8 min). Start: Saturday from 06:00 at
  the first 10 clear minutes (no other quiet lock / ticket, no MEM > 5 GB lock, no PAUSE / HOLD; 1 minute once
  /Users/kamen/Fit12/data/scr/ex16/GO exists); deadline 18:00. Guards: swap +1 GB, 30 min. DONE file at the end.
- Reported by /Users/kamen/Fit12/scripts/ex16_read.py (tested on kp2's log and on two FAKE runs, one with a PAUSE):
  decode tok/s per prompt and their GEOMETRIC MEAN (the headline), pooled tok/s, end-to-end tok/s, tokens and ms per
  check, SSD GB per check (own reads), the process's lifetime peak phys_footprint (RUSAGE_INFO_V4, GB and GiB), the
  engine's settings lines verbatim (tier plan, C_m / C_l, pools, HC, late ranks), and any voided prompt.
- No pass / fail line: a measurement. Validity: every counted prompt PAUSE-free (a paused prompt is re-run once, then left
  out and named); the lifetime peak must be under 12 GB (the line); rc 0 and a 'done' line.
- AMENDMENT Fri 23:34 (the slot only; arm, prompts, settings, report and validity unchanged): the orchestrator 23:3x: the
  window opened at 23:30, so it may run before 06:00 in a GPU gap. Re-armed as task b6utph004 with NOT_BEFORE = Fri 23:34
  (same gate: 10 clear minutes); /Users/kamen/Fit12/data/scr/ex16/WAIT (NewIdeas may touch it to keep the slot for later)
  blocks the gate like HOLD.

## 12GB-CLEAN RESULT (Sat 7 Nov 00:3x; ex16 ran 00:28:27-00:32:42 under a quiet lock, rc 0, swap flat 6742 MB, 16 / 16 counted)
- The 12 GB VERIFIED-GREEDY loop (arm ex; not exact) alone in its own process, all 16 HumanEval prompts, greedy, cap 256:
  **18.42 tok/s decode, geometric mean** (pooled 18.47: 3,456 tokens in 187.11 s; min 17.16 on /16, max 20.30 on /156).
  End to end with prefill (3.4-5.2 s per prompt of 103-367 tokens): 13.66 (geometric mean).
- Lifetime peak phys_footprint of the process (RUSAGE_INFO_V4): **11.144 GB = 10.379 GiB**, under the 12 GB line (valid);
  10.583 GB after load.
- Settings (the engine's lines verbatim in /Users/kamen/Fit12/data/scr/ex16/ex16_report.txt): tier plan 4433333333 (ranks
  0-1 at 4 bits, 2-9 at 3 bits MSB-3), C_m 47 / C_l 32 expert slots per layer (9 %), pools 6.03 GB, HC 8, embedding 4-bit
  g64, engine v3p (bt_b12 8db71cc537); late MSB-3 ranks (2, 10) apply to one-row calls only, verify calls of 2-4 rows wait
  for all ten ranks; drafter DFlash2 L1-20 at 4 bits (0.928 GB), 15 drafts per check, SW 2048.
- Per check (343 checks): 10.08 tokens kept, 11.2 ms draft + 533.7 ms verify, 4.42 GB of the process's own SSD reads.
- Decode tok/s per prompt: /156 20.30, /42 19.19, /68 17.49, /91 18.26, /16 17.16, /6 18.86, /79 17.95, /19 19.97,
  /161 18.42, /17 18.51, /56 17.43, /141 18.17, /101 19.19, /63 18.65, /114 17.75, /77 17.80 (the harness's own print
  differs by 0.01 on four prompts: rounding of the decode seconds; same geometric mean).
- Validity: rc 0, a 'done' line, no PAUSE interval (nothing re-run or excluded), no Error / Traceback, swap flat, GPU 0 %
  at the start under the quiet lock (NewIdeas' slot: after BigMoE's 200 STACK and Leap's slot, before runner 49). Outputs
  token-identical (count + sha) to kp2's ex arm on 16 / 16, from a different process, arm mix and cache history.
- Against the earlier mix: kp2 18.38 on all 16 (share lock, six-arm process, two prompts disturbed) / 18.58 on its 14
  clean prompts; T3R 18.42 (8 prompts); T3t 18.9 (4 prompts). The release number is 18.42 (geometric mean) / 18.47
  (pooled) at 11.144 GB; the six-arm process's 11.39 GB held the other arms' state too.
- Data: /Users/kamen/Fit12/data/scr/ex16 (ex16.txt, ex16_report.txt / .json, MANIFEST.txt with shasums + ENV, per-prompt
  _ex / _ex_toks / _ex_ph json). Sent to the orchestrator 00:3x.

## t80ck1u MEASURED (Sat 7 Nov 08:2x; the orchestrator 08:1x; c1u16 ran 08:19:27-08:21:48 under a quiet lock, rc 0, swap flat)
- The 12 GB FAST candidate (Breakthrough's map step 0, be2c175: P1-K1 theta 0.8 KU 4 passes the code line, 45.8 tok/s [P])
  ALONE in its own process, on 12GB-CLEAN's footing: all 16 HumanEval prompts, greedy, cap 256, full outputs, C_m 47 / C_l 32,
  F12_RERUN=1, F12_PHLOG=1. Runner /Users/kamen/Fit12/scripts/c1u16.sh; reader /Users/kamen/Fit12/scripts/c1u16_read.py (the
  orchestrator's metrics since 05:0x); data /Users/kamen/Fit12/data/scr/c1u16 (c1u16_report.txt / .json, MANIFEST.txt).
- WHOLE DECODE (n / decode wall time, prefill excluded): **46.33 tok/s geometric mean**, pooled 44.34 (3,415 tokens in 77.01 s),
  per prompt 29.81 (/91) to 86.83 (/68). FULL-PERIOD STEADY (wall time after each prompt's first check): 49.14 geometric mean,
  45.66 pooled. END TO END with prefill: 24.49 geometric mean, 24.92 pooled.
- Lifetime peak phys_footprint (RUSAGE_INFO_V4): **11.245 GB = 10.473 GiB**, under the 12 GB line.
- Per check (357 checks): 9.57 tokens kept; draft 11.1, pass 1 106.4, pass 2 / exact 83.8, commit 11.3 ms; 101 checks with a
  pass 2 and 60 exact checks after the cf gate. The first check + setup costs ~473 ms per prompt (mean; ~175-230 ms when
  pass 1 certifies check 0, ~630-910 ms when it goes exact).
- Validity: rc 0, a 'done' line, no PAUSE (16 / 16 counted), no Error / Traceback, swap flat at 6797 MB, GPU 0 % at the start.
- Against the replay: the measured pooled 44.34 sits within the earlier measured 44.2-44.9 and 3 % under the replay's 45.8
  [P]; the geometric mean (the headline form) is 46.33. Against 12GB-CLEAN (18.42 / 18.47 at 11.144 GB): x2.52 geometric
  mean, x2.40 pooled.
- Tokens (information): = the verified-greedy loop (ex16) on 4 / 16 prompts (first divergences at 104-252); = REF2-T through
  256 tokens on 4 / 16. ex16 = REF2-T on 13 / 16 (divergences /101 @164, /6 @117, /77 @134: verified greedy, near-ties).
- Quality is EvalSuite's code same-scores (docs/PREREG_t80ck1u_2026-11-07.md, HumanEval+ 60 x 2 orders). Entry point
  (Sat 08:3x): fit_t3.py with F12_IDS_OUT (one arm: t80ck1u / t90ck1u / ex), F12_CAP=item, F12_KEYS in run order; it writes
  {key: ids} + _meta.json after every prompt. Tested in FAKE mode. Backup fit_t3.py.bak_es; new sha256 7ebdce481ea5....
  EvalSuite told: the VOID rule "ex reproduces REF2-T on 4 rows" can fail without an engine bug (see above).
- Findings note: /Users/kamen/FableGod/findings/2026-11-07_0823_12gb-fast-t80ck1u-46tps.md.
- Sat 10:5x: the entry point now writes r57rows.py's input ({key: {arm: [ids]}}; fit_t3.py sha256 ec0964f190b7), and
  F12_WIRED=1 adds Leap's wired-limit call. WIRED A/B pre-registered (no pass / fail; an engine A/B): /Users/kamen/Fit12/scripts/
  wab.sh runs t80ck1u W (F12_WIRED=1) then B in separate processes on this section's footing. [P] from c1u16's phase logs:
  check 0 runs ~500 ms over the later same-mode checks when it goes to pass 2 / exact, 3.6 s in all = 4.7 % of decode, so
  W <= +4.9 % whole decode; tokens W = B expected. On the verified-greedy loop (ex16) the same excess is 2.0 % of decode.
  Starts once /Users/kamen/Fit12/data/scr/wab/GO exists (after the usage hold) and 10 minutes are clear.

## WIRED A/B RESULT (Sat 7 Nov 15:5x; wab.sh ran 15:49:51-15:55:07 under a quiet lock, rc 0, swap flat, no PAUSE; GO 14:01)
- t80ck1u W (F12_WIRED=1: mx.set_wired_limit(55.66 GB), was 0) then B, each alone in its own process, 16 HumanEval, cap 256:
  whole decode W 46.34 vs B 46.16 geometric mean (W / B x1.0039 per prompt), pooled 44.34 vs 44.21; full-period 49.18 vs 49.12
  (x1.0013); end to end 24.33 vs 24.58 (x0.990: W's prefill 60.87 vs 59.25 s, +2.7 %, W ran first); lifetime 11.242 vs
  11.244 GB; tokens W = B on 16 / 16 (= c1u16 too: three runs, one output).
- Check 0 unchanged: the excess over later same-mode checks is 3.64 s (W) vs 3.82 s (B) = 4.7 % vs 4.9 % of decode; per
  prompt 850 / 861, 888 / 914, 899 / 925, 896 / 912, 865 / 887, 815 / 833, 627 / 659 ms when check 0 goes exact.
- VERDICT: KILL at 12 GB (no gain). The 12 GB first-check excess is not IOGPU un-wiring (Leap's 45 GB cause). It looks like
  the first exact pass after prefill filling ranks 2-9 into cold pools (an expert-cache warm-up). Open lever, not routed:
  prefetch the first check's ranks 2-9 during the prefill's tail (<= ~4.9 % whole decode [P]).
- Findings note: /Users/kamen/FableGod/findings/2026-11-07_1555_12gb-wired-limit-null.md. Data /Users/kamen/Fit12/data/scr/wab.

## TABLE 12 GB CELLS 12E + 12F (pre-registered Sat 7 Nov 17:2x; the orchestrator 17:1x, NewIdeas 17:2x QUEUE c8dfaac)
- Kamen's 45 / 24 / 12 GB x exact / lossy table (Sun ~08:00). The 12 GB cells run on BigMoE's S5 prompt set so the rows
  compare: 12 HumanEval (68, 16, 91, 6, 79, 19, 161, 17, 56, 141, 101, 63) + MBPP12 (16, 100, 146, 161, 183, 208, 224, 246,
  268, 296, 305, 319), CAP 256 as S5, greedy, full outputs to EOS / cap.
  - 12E (slot 3): arm ex, the verified-greedy loop (NOT bit-exact; 12GB-CLEAN's 18.42 was on 16 HumanEval).
  - 12F (slot 4): arm t95ck1u = P1-K1 KU 4, cf 0.9, theta 0.95 (t80ck1u -11.7 and t90ck1u -8.3 pt failed EvalSuite's code
    gate; the map passes theta 0.95 at 1.10 % code flips, 37.9 tok/s [P] on 16 HumanEval). FAST.
- Runner /Users/kamen/Fit12/scripts/tab12.sh: each arm alone in its own process, back to back under one quiet lock, 12GB-CLEAN's
  footing (C_m 47 / C_l 32, F12_RERUN=1, F12_PHLOG=1). Each run writes r57rows-form ids for EvalSuite (tab12{e,f}_ids.json).
- Report (c1u16_read.py): whole decode (headline), full period, end to end, each as geometric mean and pooled, overall and per
  set (HumanEval / MBPP); lifetime peak; 12E's tokens vs ex16 (12 shared HumanEval keys: same arm and cap, so they should
  match), 12F's tokens vs 12E. No pass / fail: measurements. Validity: rc 0, a 'done' line, PAUSE-free counted prompts,
  lifetime < 12 GB. 12F's quality verdict is EvalSuite's gate (slot 7, its own pre-registration).
- fit_t3.py sha256 11cadf9ec149: + the t95ck1u arm; keys missing from b12llff4's rows (MBPP) come from EvalSuite's
  data/items/dec_<set>.jsonl. FAKE-tested 17:23 (both arms, the reader's per-set split, r57rows-form ids).
- RESULT (Sat 17:4x; tab12.sh 17:36:52-17:46:44, one quiet lock, rc 0 both, no PAUSE, swap flat 6733 MB; GO 17:24):
  - 12E verified greedy (ex): whole decode 12 HE geo 18.58 / pooled 18.64, MBPP12 18.18 / 17.79 (all 24: 18.38 / 18.34);
    full period 18.57 / 18.43; end to end 13.87 / 11.15; lifetime 11.140 GB = 10.375 GiB; 450 checks, 8.86 kept, 471 ms
    each. Tokens = ex16 on the 12 shared HumanEval keys (12 / 12).
  - 12F theta 0.95 (t95ck1u): whole decode 12 HE geo 40.59 / pooled 39.87, MBPP12 18.74 / 18.66 (all 24: 27.58 / 27.90);
    full period 45.33 / 18.97; end to end 22.73 / 11.56; lifetime 11.280 GB = 10.505 GiB; 529 checks, 7.74 kept; per check
    pass 1 98.3, pass 2 / exact 158.5 ms; checks with a pass 2 261, exact 129. Tokens = 12E on 6 / 24.
  - Reading: FAST x2.18 over verified greedy on HumanEval, x1.03 on MBPP. MBPP keeps 5-8 tokens per check, so pass 1
    rarely certifies, and some MBPP prompts run slower than verified greedy (/268 13.9 vs 17.5, /224 17.0 vs 19.8).
  - Findings note: /Users/kamen/FableGod/findings/2026-11-07_1747_12gb-table-cells.md. Ids for EvalSuite:
    /Users/kamen/Fit12/data/scr/tab12/tab12{e,f}_ids.json (r57rows form, CAP 256).
- EvalSuite 20:16 (its RESULT_t80ck1u, t95 section): t95ck1u FAIL at the bar, HE+ 90.0 vs REF2-T 95.0 (-5.00, p 0.25) in both
  orders; -1.67 vs NAX6 (information). With t80ck1u (-11.7) and t90ck1u (-8.3), the 12 GB lossy cell fails code quality on
  every certificate tried; the verified-greedy loop (12E) is the 12 GB row.

## 12 GB CHAT + PREFILL (pre-registered Sat 7 Nov 23:5x; NewIdeas 00:2x, QUEUE 9c65385 PROGRAM CHAT 150 / PREFILL 500)
- Why: Kamen 23:5x wants 150 tok/s on chat at 32 / 24 / 12 GB. At ~12 GB/s, 150 on chat needs <= ~80 MB of SSD per kept
  token; the 12 GB loop reads ~440 on code today. The MB per kept token on chat is the input for Breakthrough's chat BYTES10.
- 1) chat12: the verified-greedy loop (arm ex, tab12's footing) on BigMoE N45's 12 IFEval prompts, CAP 256 as N45. Report:
  whole decode / full period / end to end (geometric mean, pooled), tokens kept per check, SSD MB per kept token (the
  process's own reads inside the decode phases, from the phase marks), lifetime peak; r57rows-form ids for EvalSuite.
- 2) pf12a / pf12b: 12 GB PREFILL on BigMoE's PFX prompts (agent/edit10 1,565, agent/write13 4,730, agent/write11 7,932 tokens,
  ascending; one process per arm; CAP 32). A = today's b12_prefill (s_moe 4096: each layer's experts read once per 4,096 prompt
  rows; 1.6K is already one MoE call), B = s_moe 8192 (one MoE call per prompt, BigMoE's PFX form; the docstring says the
  split leaves the arithmetic unchanged). Report: prefill s and tok/s (ids[:-1]), the prefill's own SSD reads, TTFT = prefill +
  check 0 + the drafter's setup (the first tokens appear at check 0's commit), lifetime peak per length (does a 4.7K / 7.9K
  prompt fit 12 GB?), decoded tokens A = B. [P] NewIdeas: an SSD floor of ~4.6 s per prompt at 12 GB for one expert pass.
- Runner /Users/kamen/Fit12/scripts/chat12.sh (armed 23:55:40): after EvalSuite's results/CERT_DONE, 10 clear minutes (GO = 1
  min, WAIT = hold), one 30-min quiet lock, start by Sun 09:00, swap +1 GB / 40 min guards, DONE at the end. Code: t3_twopass.py
  F12_SMOE + the prefill's SSD bytes (sha256 800323c3ca97; backup .bak_pf); fit_t3.py PREFILL line (2338f9367b00; backup
  .bak_pf); c1u16_read.py SSD MB per kept token; pf12_read.py. FAKE-tested 23:5x.

## BC2 (Sun 8 Nov 00:1x; NewIdeas 01:1x for Leap's CHAT-150 strike, QUEUE c16f510): is there a 2-bit expert read path?
- No read path exists, at 12 GB or 45 GB. records3 stores each expert's MSB-3 part as ONE MLX bits=3 bitstream (value i at
  bits 3i..3i+2; Breakthrough's v3p tier.py planes(): m = q >> 1, s3 = 2s, b3 = b + s * the group mean of the dropped bit)
  plus the LSB as a 1-bit stream. The top two bits of each weight sit inside the 3-bit stream, so a "top 2 planes" read of
  records3 reads every byte of the MSB-3 part: 0 % saved. Every MSB-2 number we have (O2, Q2, B2, s7_cal / s7_cal_all) came
  from in-memory emulation.
- Store sizes (du, Sun 00:1x): /Users/kamen/BigMoE/models/bare/records3 and records3m are separate copies of 64 GB each
  (1.43 / 1.42 GB per layer). records3 = 4.75 b/w (w3 3 + s3/b3 0.5 + l1 1 + b 0.25). Disk: 91 GB free.
- Option A, a separate MSB-2 store: q2 = q >> 2 as an MLX bits=2 stream, s2 = 4s, b2 = b + s * mean(q & 3) per 64-group
  (records3's bias rule) = 2.5 b/w, -29 % per read vs MSB-3's 3.5. Compute: mx.gather_qmm bits=2 natively (no new kernel). Work:
  a third part type in r57's loader / pools and a tier plan with 2-bit ranks. Every expert can sit at rank 6-9, so the copy is
  the whole model: ~0.75 GB per layer, ~36 GB. It does not fit (91 GB free, floor 100) without freeing space (Kamen's call).
- Option B, re-lay records3 as prefix records [s3 b3 | MSB-2 plane | bit plane | LSB, b] with b2 = b3 + 0.5 s3 (midpoint, no
  extra field): no new disk; one request per read at every precision (MSB-2 = the first segment, MSB-3 = the first two), so
  it avoids the request overhead that killed plane-split reads; MSB-3 rebuilt on the GPU (the pattern of tier.py's q = 2m + l
  kernel). Cost: it rewrites the store every exact mode reads (NAX6-N), so every loader changes and every exact mode needs a
  re-proof (bitwise by construction, but checked).
- IFEval expectation: MSB-2 = 5x MSB-3's weight MSE (15 vs 3 E4; O2 table above) and 7.8x its output error (RTN-2, Q2); the
  data-aware 2-bit codes stayed 1.7-2.3x MSB-3 (B3b, B10-r: route closed). NewIdeas' s7_cal_all: IFEval KL 0.109, chat 8-19 %
  above the linear-in-MSE model; today's FAST stack sits at 8.22 % IFEval flips vs the 8.13 % line [G]. Ranks 6-9 carry a
  small share of the routing weight squared (my guess ~20-30 % of ranks 2-9's error, [G], not measured), but they are about
  half the MSB-3 misses. Sure rows only does not isolate the error: lever 1 showed the certificate is token-exact and the
  damage enters through the committed rows' KV. Expect BETWEEN at best for ranks 6-9 and KILL for 2-9 against NAX6-N.
- Recommendation: measure before building. One map arm with ranks 6-9 at MSB-2 emulated in memory (the s7_cal machinery; no
  store), read with map_kl.py on IFEval (KL + flips); build a store only if it holds, and only after disk is freed.
- Reusable from the 12 GB loop: no 2-bit store and no 2-bit kernel exist. What carries over: the emulation formulas and quality
  anchors (this file and /Users/kamen/Fit12/PLAN.md: O2, Q2, B3b, B10-r), the bias rule, and v3p tier.py's planes() / rebuild
  kernel as the template for option B (Breakthrough's code). The 12 GB pools and loader are v3p's, not r57's: nothing ports
  directly. Posted in the lounge Sun 00:1x (five notes).
- RESULT (Sun 05:07:45-05:12:41, chat12.sh after CERT PASS 04:54, one quiet lock, rc 0 x3, no PAUSE, swap flat 4703 MB):
  - chat12 (verified greedy, 12 IFEval, CAP 256): whole decode 12.88 tok/s geo / 12.76 pooled (1,577 tokens in 123.59 s; 9.26-
    17.06), full period 13.16 / 12.85, end to end 9.02 / 10.09; 675 checks, 2.34 tokens kept per check, 172.1 ms verify + 10.4
    ms draft; SSD 1.30 GB per check = 556 MB per kept token pooled (479-749 per prompt); lifetime 10.962 GB = 10.21 GiB.
  - Prefill A (s_moe 4096) / B (8192), tokens 1,565 / 4,730 / 7,932: prefill 8.21 / 12.52 / 17.19 s = 190.6 / 377.7 / 461.3
    tok/s (B 191.6 / 376.0 / 458.1, x1.005 / 0.995 / 0.993); TTFT 8.53 / 13.03 / 17.50 s; prefill SSD 63.4 / 66.0 / 67.1 GB in
    both arms; lifetime 10.619 / 10.773 / 10.948 GB (all fit 12 GB); decoded tokens A = B on 3 / 3.
  - Reading: b12_prefill is one call per prompt; its stream reads each layer's experts once per call, so the 12 GB prefill is
    already expert-once (bytes ~ one pass of records3m's ~68 GB at every length). s_moe only batches the MoE compute: no
    gain. It is 2-3.7x the 45 GB exact prefill (N45 98 / 118 / 124 tok/s), whose screen re-streams per 1,024-row call; BigMoE's
    PFX is this form. Effective SSD 7.7 / 5.3 / 3.9 GB/s: longer prompts turn compute-bound (Leap's PF-B lever).
  - Findings note: /Users/kamen/FableGod/findings/2026-11-08_0515_12gb-prefill-expert-once-chat.md. Lounge posts 05:1x.

## T2 CHAT WIDTH (the 12 GB chat lever; pre-registered Sun 08 Nov 07:21, before any GPU run)
> (a quote from the project owner, kept private)
small tests under share locks. T1 = chat12 (12.88 tok/s, 556 MB per kept token) + the prefill A/B.
- CPU audit [K] (/Users/kamen/Fit12/scripts/chatw_audit.py on chat12's own logs; 663 checks, check 0 dropped):
  - One exact call costs 59.4 ms fixed + 81.7 ms per GB read (12.2 GB/s marginal; rms residual 2.5 ms).
  - A chat check = 1.10 calls of 4 rows: 65.6 ms fixed + 103.8 ms of SSD (1.27 GB, 690 misses, 1.84 MB per miss) + 10.2 ms
    draft. Kept per check 2.33: 39.5 % of checks keep 1 token, 27.0 % keep 2. 47 % of the rows read are wasted.
  - The first draft's selector cf predicts short checks: cf0 < 0.5 (27 % of checks) keep 1 token 79 % of the time;
    cf0 >= 0.97 (22 %) keep 3.80 on average.
- Lever: the exact check's FIRST call reads w1 = 1-3 rows when cf0 is low (later calls 4 rows), so rejected rows' experts
  are not read. Exact-preserving: the verify is chunking-invariant for calls of >= 2 rows (WV1 / WV2); a chosen 1-row call
  runs with ALLWAIT (the verify rows' flags; without it the late rule makes 1-row bits depend on the pools).
  - Price depends on the per-width fixed cost, not yet measured. If the 59 ms is per call at any width, narrow calls pay
    only on cf0 < ~0.5: +3-8 % [P]. If it shrinks with width: up to ~+15 % [P]. Related: F3 adaptive MTP width (KILL,
    10-23) changed how many drafts the MTP head made; this keeps DFlash2's 15 drafts and changes only the verify chunking.
- Code: t3_twopass.py F12_W1 = 4 | rand | cf:C:W[:C2:W2] (+ F12_W1SEED, F12_W1AW) and a per-call log (<tag>_<key>_ex_calls.json:
  check, call index, width, rows kept, marks at start / end); fit_t3.py dumps it. Backups *.bak_w1. FAKE dry test passed.
- Test (/Users/kamen/Fit12/scripts/chw12.sh; one share lock, MEM 12 GB, ~10 min; reader chw12_read.py): chat12's 12 IFEval
  prompts, CAP 128, arm ex, two processes: A = W1 4 (control), R = W1 rand (1-4 rows per check, seeded).
  - A = chat12's tokens on the first 128, 12 / 12. Otherwise the harness change is not inert: stop.
  - R = A in tokens AND kept-row logits hash (lsha), 12 / 12: exact-preserving. Otherwise name the widths that differ and
    restrict the policy to the matching ones.
  - Pricing: the best cf policy, priced on A's checks with R's per-width costs: >= +5 % whole decode [P] -> stage 2;
    < +3 % -> KILL (the fixed per-call cost eats the bytes saved).
  - Timings count only if no other GPU job ran (locks.txt, one line per minute). Bytes and tokens count either way.
- Stage 2 (only after stage 1 passes): the chosen policy vs W1 4 in separate processes, idle-GPU share slot or quiet; PASS
  whole decode geo >= +5 % with the same tokens; KILL < +2 %.

## PREFILL BUG CHECK (pre-registered Sun 08 Nov 08:36, before any GPU run; orchestrator 08:2x + Leap TABLE_350 a521e91)
- The claim: MLX 0.32.2's sorted NAX gather (affine_gather_qmm_rhs_nax) leaves rows UNWRITTEN when one call has > 32,767
  sorted rows and rows % 64 != 0 (Leap: 79,320 rows -> 32,776 missing; aligned -> 0). The orchestrator marked my 4.7K / 7.9K
  prefill figures (378 / 461 tok/s) void and asked for a re-measure with Leap's pad-to-64 fix.
- My correction first: the pf12 A/B (s_moe 4096 vs 8192) was NULL. bt_b12 defaults to B12_MODE=fast, which sets PFR's grid
  to "balanced" with PFR_MIN 512 (bt_b12.py:124-129, :763), so every PFX prompt ran ONE MoE chunk per prompt in both arms;
  F12_SMOE did nothing. x0.99-1.005 = run-to-run noise. The pf12a / pf12b logits hashes are equal 3 / 3, as expected for one
  configuration. Still true: the shipped 12 GB prefill reads each expert once per prompt (one chunk up to 16K tokens).
- Code reading [K, not yet measured]: inside that one chunk, the tier prefill MoE runs per group of ~64 experts
  (BIGMOE_TIER_PF_GROUP=64 in fit_t3.py's ENV; v3p tier.py _prefill_grouped, _groups: <= 2G experts and >= 4 rows per
  expert per group, else one single call). Every group call pads its rows to a multiple of 64 (tier.py:599, "BUG1") and holds
  ~1/8 of the rows (~10K at 7.9K). So neither of the bug's conditions holds, unless _groups falls back to the single call
  (79,310 rows at 7.9K: exposed).
- Prior evidence on the same MLX (0.32.2, installed Oct 17): PFR-a (Nov 2) gave byte-identical KV and logits at 64K between
  16,384- and 4,096-token MoE chunks. Both had unaligned tails > 32,767 rows (158,480 and 35,590), which a single-call path
  with the bug would have broken differently.
- Test (/Users/kamen/Fit12/scripts/pf13.sh; quiet lock, after chw12, ~8 min; reader pf13_read.py): arm ex, CAP 32, PFX's
  1.6K / 4.7K / 7.9K, one process per arm.
  - F (pf13f): the shipped grid + a gather_qmm audit in my process (t3_twopass.gq_install: every sorted call's rows, exposed =
    rows > 32,767 and rows % 64 != 0) + Leap's pad on exposed calls (copies of the last row and its expert, output sliced).
  - K (pf13k): B12_PFR_GRID=keep, s_moe 3072. Every MoE call <= 30,720 rows, so no call can reach the bug.
  - NOT HIT (378 / 461 stand): exposed = 0 in both arms, F = K in the kept-row logits hash (lsha) AND tokens 3 / 3, and
    F's tokens = pf12a's = pf12b's 3 / 3. Re-measured timings are reported either way.
  - HIT: exposed > 0. Then F's figures replace pf12's (the pad engaged), and F = K bitwise tells whether the fix is exact on
    this engine.
  - Other: exposed = 0 but F != K means a chunking dependence that is not this bug. Report it and look into it.
- Code: t3_twopass.py GQ / gq_install + st gq / pfr per prefill; fit_t3.py F12_GQAUDIT / F12_GQPAD / F12_PFRGRID + the PREFILL
  line's pfr / gq fields (backups *.bak_gq). CPU check: the wrapper pads an exposed 32,790-row call bitwise = unpadded (CPU
  kernel). FAKE dry test passed.

## T2 CHAT WIDTH stage 1 RESULT (ran Sun 09:19:34-09:24:11, share lock, GPU 0 % at start; locks.txt: only BigMoE FU queued, no other GPU job; swap flat, rc 0 x2, no PAUSE; data /Users/kamen/Fit12/data/scr/chw12)
- A (W1 4, the control) = chat12's tokens on the first 128: 12 / 12. The harness change is inert and the loop repeats.
- R (W1 rand 1-4) != A: tokens 5 / 12, kept-row logits hash 0 / 12. NOT exact-preserving as built.
  - Cause (WV1-WV3, 24 Oct): one verify pass and full-accept chunking give the same bits at widths >= 2, but a keep-1 commit
    replays the PLE layer as a 1-row forward (decode-width arithmetic), and so does a 1-row call. Moving the call
    boundaries moves the keep-1 events: A keeps 1 row when kept = 1, 5, 9 ...; a 2-row first call adds kept = 3, 7 ... .
  - PR1 (scripts/fit_pr1.py: 2-row replay of keep-1 commits; PASS 24 Oct on the MTP path, draft-policy-invariant for verify
    width >= 2) is not in the pinned engine. So without PR1 no width matches, and the pre-registered fallback ("restrict to
    the matching widths") has nothing to restrict to.
- Cost per exact call [K] (R: 651 calls, check 0 dropped, rms 3.4 ms): fixed 42.3 / 43.5 / 54.3 / 58.9 ms for 1 / 2 / 3 / 4
  rows (later 4-row calls 58.0); 84.1 ms per GB (11.9 GB/s marginal); 300 / 531 / 868 / 1,158 MB per first call (~290 MB
  per row: rows share few experts beyond what the pools hold).
- Pricing on A's 498 checks (kept 2.337; 10.9 ms per check outside the exact calls; all-4 = 174.7 exact ms per check):
  - best "cf0 < 0.9 -> 2 rows, else 3": 145.1 ms, x1.190 decode [P]; "cf0 < 0.9 -> 2, else 4": x1.174 [P];
  - with 1-row calls, "cf0 < 0.5 -> 1, < 0.9 -> 2": x1.178 [P]. No better: a 1-row call still costs 42 ms fixed.
- Measured side note [K; random widths, not a policy]: R 12.97 vs A 12.54 tok/s whole decode geo (+3.4 %); 509 vs 559 MB
  per kept token; lifetime peaks 10.99 / 10.96 GB.
- Stage 1 verdict: PRICE PASS (x1.17-1.19 [P], line +5 %); EXACTNESS FAIL as built. Stage 1b below.

## T2 CHAT WIDTH stage 1b (pre-registered Sun 08 Nov 09:27, before any run): PR1 + widths 2-4 only
- Build: fit_t3.py F12_PR1=1 -> fit_pr1.install(MTPLX qwen4exp) (asserts the patched source still matches). t3_twopass: the
  width policy never picks 1 row, and a tail rule never leaves a 1-row last chunk (rem - w == 1 -> w - 1 if w >= 3, else
  w + 1). The only 1-row call left is the cap's last token (no drafts), at the same position in every arm.
- Arms (share lock, chat12's 12 IFEval, CAP 128, one process each): A' = W1 4 + PR1 + tail rule (the new reference);
  P' = "cf:0.9:2:1.01:3" + PR1 + tail; Q' = W1 rand over {2, 3, 4} + PR1 + tail (a harder invariance probe).
- Lines:
  - P' = A' and Q' = A' in tokens AND kept-row logits hash, 12 / 12 -> exact-preserving w.r.t. A'. Any mismatch -> KILL the
    exact form (PR1 does not cover the T3 path); the lossy form would need EvalSuite's IFEval gate.
  - A' vs chat12: information (expected to differ at near-ties where ex committed a keep-1 pass, as U2 vs U1 in October).
  - Speed P' vs A' (whole decode geo; timings count only with an idle GPU): >= +8 % -> stage 2 (quiet, 12 IFEval at CAP 256,
    + 16 HumanEval so code is not hurt); < +5 % -> KILL.
- Adoption: PR1 gives the 12 GB loop new tokens at near-ties (same model math), so adopting it = EvalSuite same-scores
  (FAST-track rule, 11-04) or Kamen's call, as PR1's U2 in October.
- Stage 1b build note (Sun 09:43, before any run): with the tail rule on, the cap's last token (no drafts) is the only 1-row call.
  It now takes ALLWAIT (verify flags), because its late-rule bits would depend on the pools, which differ between arms. PR1's
  source match checked on the installed MTPLX (qwen4_exp.py: 1 match). FAKE dry test passed with the stage 1b flags.

## PREFILL BUG CHECK RESULT (pf13.sh, quiet lock 09:52:17-09:54:12, GPU 0 % at start, swap flat, rc 0 x2; data /Users/kamen/Fit12/data/scr/pf13, pf13_report.txt)
- NOT HIT: all three pre-registered lines pass.
  - Exposed sorted gather_qmm calls (> 32,767 rows and rows % 64 != 0): 0 in both arms. Largest call: 4,672 / 10,496 /
    15,872 rows (one-chunk grid) and 4,672 / 6,080 / 6,400 (3,072-token chunks), all multiples of 64. Leap's pad never acted.
  - F (shipped grid + pad) = K (3,072-token chunks) bitwise: kept-row logits hash 3 / 3, tokens 3 / 3.
  - F's tokens = pf12a's = pf12b's 3 / 3.
- PFR's log confirms the grids: F one MoE chunk per prompt (s_moe 2,048 / 5,120 / 8,192, the balanced grid); K 3,072.
  Sorted calls per prompt: F 1,140 / 474 / 396 (the one-chunk path forms 3-5 larger groups per layer at 4.7K / 7.9K, 8 at
  1.6K); K 1,140 / 1,870 / 2,538.
- Re-measured [K] (F): 191.6 / 373.4 / 459.4 tok/s, TTFT 8.48 / 13.17 / 17.57 s, SSD 63.8 / 66.8 / 68.9 GB, lifetime
  10.587 / 10.773 / 10.949 GB. x1.005 / 0.989 / 0.996 vs pf12a: the 05:1x figures stand.
- The real chunk A/B [K] (replacing the null pf12 A/B): K's 3,072-token chunks re-read the store once per chunk: 187.3 /
  259.4 / 272.7 tok/s, TTFT 8.68 / 18.74 / 29.41 s, SSD 63.3 / 122.3 / 180.5 GB. The shipped one-chunk grid is x1.44
  (4.7K) and x1.68 (7.9K) with the same bits; each extra store pass costs ~5.6-5.9 s (~56 GB).
- Findings note 2026-11-08_0515 updated (header VALID, UPDATE 09:5x block). Leap's 08:42 note agreed: tier.py's BUG1 pad covers it.

## T2 CHAT WIDTH stage 1b RESULT (chw13.sh, share lock 10:00:27-10:08:10, GPU 0 % at start, no other session's lock during it, swap flat, rc 0 x3, PR1 installed in all three; data /Users/kamen/Fit12/data/scr/chw13)
| arm (ex + PR1 + tail rule, 12 IFEval, CAP 128) | whole decode geo (pooled) | MB per kept token | kept / check | first-call widths | tokens / lsha = A' |
|---|---|---|---|---|---|
| A' W1 4 (reference) | 12.68 (12.56) | 564 | 2.348 | 4: 487, 3: 6, 2: 2, 1: 3 (cap-end tails) | - |
| P' cf0 < 0.9 -> 2 rows, else 3 | 14.05 (14.07): x1.108 (x1.120) | 475 (x0.843) | 2.326 | 2: 339, 3: 171, 1: 3 | 5 / 12, 1 / 12 |
| Q' random 2-4 | 13.13 (13.15): x1.035 (x1.047) | 520 (x0.922) | 2.410 | 2: 185, 3: 170, 4: 130, 1: 2 | 3 / 12, 0 / 12 |
- EXACT FORM: KILL (pre-registered line: any mismatch). Even with PR1 (264 / 217 keep-1 replays on 2 rows), widths >= 2
  and the tail rule, the committed state depends on the call widths on this T3 path. Example: ifeval/1220 diverges at
  token 3, after two checks with the same kept counts (2, 1) done as 2-row vs 4-row calls. WV1-WV3 (24 Oct) covered the MTP
  path (B12LL) at widths 2-4 from one state; the T3 path's commits evidently are not covered. Not chased further (low pace).
- SPEED [K, rough: share lock, idle GPU, texts differ on 7 / 12 prompts]: x1.108 geo / x1.120 pooled for P' (priced
  x1.19). PR1 itself costs ~0 (A' 12.68 vs stage 1's A 12.54 without it).
- What the lever is now: a rounding-level change of the verified-greedy loop. Every committed token is still the full
  model's own verify argmax, as in today's loop, whose tokens already depend on the drafts' acceptance pattern (exd, WV3).
  Adopting it = EvalSuite same-scores (FAST-track rule) on IFEval + code, after a code check (a 3-row first call on
  high-cf code checks may add a call on full accepts).

## T2 CHAT WIDTH stage 2 (pre-registered Sun 08 Nov 10:11, before any run): the 2-or-4 rule on chat + a code check (rough, share lock)
- Why: the exact form is killed, so the lever is a rounding-level change that EvalSuite must judge. Before asking for that
  gate: (1) does the code-safe rule P2' = "cf0 < 0.9 -> 2 rows, else 4" (x1.174 [P]) keep most of P''s chat gain, and (2) is
  code speed unhurt? P''s 3-row first call on high-cf code checks can add a call on full accepts; P2' keeps 4 rows there.
- Runner /Users/kamen/Fit12/scripts/chw14.sh (share lock, idle GPU, ~11 min; after chw13's DONE). All arms ex + PR1 + tail
  rule, one process each: c2 = P2' on chat12's 12 IFEval, CAP 128; ha = W1 4 on 16 HumanEval (c1u16's keys), CAP 256;
  h2 = P2' on the same 16 HumanEval.
- Lines:
  - Chat: c2 vs stage 1b's A' (12.68 tok/s geo, same day) >= +8 % -> P2' is the candidate. Otherwise report it; P' stays
    the faster rule pending its own code check.
  - Code: h2 vs ha whole decode geo >= -2 % -> code not hurt.
  - Both lines pass -> hand P2' to EvalSuite: IFEval + HumanEval same-scores at 12 GB, both orders, its protocol.
  - Tokens: information.

## T2 CHAT WIDTH stage 2 RESULT (chw14.sh, share lock 10:18:01-10:30:16, GPU 0 % at start, no other session's lock during it, swap flat, rc 0 x3; data /Users/kamen/Fit12/data/scr/chw14, chw14_chat.txt / chw14_code.txt)
| arm (ex + PR1 + tail rule) | prompts | whole decode geo (pooled) | MB per kept token | kept / check | tokens / lsha = ref |
|---|---|---|---|---|---|
| A' W1 4 (stage 1b, 10:00) | 12 IFEval, CAP 128 | 12.68 (12.56) | 564 | 2.348 | ref |
| P2' cf0 < 0.9 -> 2 rows, else 4 | 12 IFEval, CAP 128 | 13.89 (13.97): x1.096 (x1.112) | 491 (x0.870) | 2.326 | 5 / 12, 1 / 12 |
| ha W1 4 | 16 HumanEval, CAP 256 | 18.68 (18.69) | 441 | 10.095 | ref |
| h2 P2' | 16 HumanEval, CAP 256 | 18.50 (18.47): x0.990 (x0.988) | 439 (x0.995) | 10.376 | 12 / 16, 6 / 16 |
- Chat line (>= +8 %): PASS, +9.6 % geo / +11.2 % pooled. P' (2 or 3 rows) was +10.8 / +12.0 %; P2' keeps 4 rows on high cf.
- Code line (>= -2 %): PASS, -1.0 % geo (51 of 333 code checks took the 2-row first call; texts differ on 4 / 16).
- ha = 18.68 tok/s matches 12E's 18.58 (HumanEval, no PR1): PR1 + the tail rule cost nothing on code either.
- Verdict: both lines pass. P2' goes to EvalSuite as a FAST-track candidate (IFEval + HumanEval same-scores at 12 GB, its
  protocol); NewIdeas places it. It stays a rounding-level change of the verified-greedy loop (not bit-exact, like today's ex).

## T2 FIXED COST PROFILE (pre-registered Sun 08 Nov 11:37, before any run): where an exact call's 42-59 ms fixed cost goes
NewIdeas 11:3x (QUEUE 489db84 "GPU QUEUE, RE-ORDERED 11:3x"): Fit12's next piece runs right after Breakthrough's repro + map
gates. After stage 2 the 12 GB chat check is ~1.1 exact calls at 42-59 ms fixed + 84 ms per GB + ~10 ms draft: the fixed
part is ~1 / 3 of a chat check. Rule (memory "profile the live path per stage"): measure the stages before routing any
fixed-cost bet. This is a measurement only: no setting changes.
- Code: t3_twopass.py F12_GSTAT=1 (TS log, `<tag>_<key>_ex_ts.json`; sha cc191f320b67; backup .bak_gs). Per exact call it logs:
  - uptime stamps: t0 start, t1 after forward() (graph built), t2 after mx.eval, t3 after tolist + compare, t4 after
    commit, t5 after the lsha hash;
  - the loader thread's own clocks (ext.stats: proc_s, io_s, cpu_wait_s, misses, stalled, uses) at t0, t1, t2.
  - fit_t3.py dumps the log (sha 42266985905b; backup .bak_gs).
  - Row layouts of the existing logs are unchanged.
- Run chw15p: P2' (F12_W1 cf:0.9:2 + PR1 + the tail rule, the FAST-track candidate's env) on chat12's 12 IFEval, CAP 128,
  F12_PHLOG=1 + F12_GSTAT=1. One process, share lock, idle GPU, ~5 min. Gate as chw14 (no other > 5 GB lock or quiet ticket
  for 6 min, >= 20 GB free, no PAUSE / HOLD / ACT / ramp / closed lid). Stops on swap +1 GB or 20 min. Deadline Sun 18:00.
- Per call width and call index, the reader (chw15_read.py) gives mean ms for:
  - build (t1 - t0), eval (t2 - t1), host (t3 - t2), commit (t4 - t3), hash (t5 - t4);
  - inside eval: the loader busy (proc), of which reads (io) and overhead (proc - io), and the loader idle (eval - proc,
    the GPU side's critical path);
  - fits of each part against the call's GB read (its intercept = that part's share of the fixed cost);
  - per check: draft ms and the time outside draft + exact.
- Sanity checks: tokens + lsha = chw14c2 (same env) on 12 / 12 expected, and whole decode within +-5 % of 13.89. A token
  mismatch means run-to-run variation or a perturbing stamp: reported either way.
- Routing lines (decided before the data):
  - Any host stage (build, host, commit, hash) >= 8 ms per call on average (~15 % of the fixed cost): a lever in my harness /
    copies; I price it [P], then pre-register a build.
  - Loader overhead (proc - io) >= 8 ms per call: a loader question for Breakthrough / Leap, with the table.
  - The loader-idle intercept (eval - proc at 0 GB) >= 25 ms per call: a GPU-side dispatch / kernel question for Leap, with
    the table.
  - None of the three: the fixed cost is spread thin; I close the fixed-cost lever on my side (KILL).
  - The hash (t5 - t4) is harness-only: if >= 2 ms per call, my 12 GB whole-decode figures understate the shipped loop by
    that much (information).

## T2 FIXED COST PROFILE RESULT (chw15.sh, share lock 12:02:13-12:04:19, GPU 0 % at start, no other session's lock during it (locks.txt), swap flat, rc 0; data /Users/kamen/Fit12/data/scr/chw15, chw15_prof.txt (reader v2) / chw15_eq.txt)
- Sanity: tokens 12 / 12 and lsha 12 / 12 = chw14c2 (same env, run 2 h earlier). The stamps change nothing, and the loop
  repeats bit for bit. Whole decode 13.80 geo (13.90 pooled) vs 13.89: x0.993.
- 629 exact calls (first calls: 340 at 2 rows, 164 at 4; later calls: 113 at 4). Per call, ms = a[w] + b * GB:

  | stage | a[1] | a[2] | a[3] | a[4] | ms per GB |
  |---|---|---|---|---|---|
  | whole call (t0..t5) | 43.3 | 43.6 | 53.2 | 58.8 | 82.7 |
  | loader reading (io over t0..t2) | 9.4 | 9.4 | 14.3 | 15.0 | 80.3 |
  | loader overhead (proc - io) | 0.1 | 0.1 | 0.0 | 0.0 | 0.0 |
  | loader idle over t0..t2 ("nl": Python + GPU side + handoffs) | 32.0 | 31.0 | 36.1 | 40.4 | 2.4 |
  | tolist + compare | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
  | commit | 0.0 | 1.2 | 1.3 | 1.8 | -0.1 |
  | lsha hash (harness only) | 0.5 | 0.7 | 0.9 | 1.1 | 0.0 |

- Shares of draft + exact time: reads 63.7 %, loader idle 27.5 %, commit 1.0 %, hash 0.6 %, drafts 6.3 % (10.4 ms per check).
- DEVIATION from the pre-registered stamp meaning:
  - forward() is not graph-only: the harness env sets BIGMOE_ASYNC_EVERY=1, which submits every layer as it is built.
  - So the loader is busy 30-124 ms inside forward(), and the mx.eval window holds only the last 3.5-5 ms + 6.6 ms per GB.
  - So 'build' (t1 - t0) is not graph building.
  - And 'eval - proc' (3.0-4.1 ms) is only the tail of the loader-idle time. The quantity the idle line meant is nl.
- Routing (the pre-registered lines):
  - Host stages: tolist + compare 0.0, commit 1.2-1.8, hash 0.5-1.1 ms per call, all < 8 ms: no host lever.
    'build' is 115 ms mean by the letter, but it holds the reads and the GPU work.
  - Loader overhead 0.1 ms per call (< 8): no loader lever. KILL.
  - Loader idle by the letter (eval - proc): 3.0-4.1 ms < 25, so not triggered. The meant quantity nl = 31-40 ms >= 25.
    - But nl mixes Python graph building with the GPU side (+ the GPU <-> loader handoffs), and these stamps cannot split
      them.
    - Reads add at the full read rate (82.7 vs 80.3 ms per GB), so nothing hides under Python. A Python-bound pipeline is
      therefore unlikely: [K] slope, [P] reading.
    - Before anything goes to Leap: chw16 (below) splits Python from the GPU side.
  - Hash 0.5-1.1 ms per call (< 2): my 12 GB whole-decode figures understate the shipped loop by ~0.6 % (information).
- Also [K]: the reads carry 9-15 ms of fixed time per call (per stalled layer: latency, not bytes; 19-47 misses per call).

## T2 FIXED COST SPLIT (chw16, pre-registered Sun 08 Nov 12:0x: `date` 12:08 at arming; the chw16.sh header says 12:1x in error; before any run): Python vs the GPU side inside nl
- Same run as chw15p (P2' + PR1 + tail, 12 IFEval, CAP 128, F12_GSTAT=1).
- Each TS row adds at its end the main thread's CPU time (time.thread_time) at t0, t1 and t2.
- cpu_f = the main thread's CPU over t0..t1; the reader fits it per width like the other stages.
- Sanity: tokens + lsha = chw14c2 / chw15p 12 / 12.
- Routing:
  - cpu_f intercept >= 0.8 x nl's intercept (Python busy for the idle time): Python is on the critical path. That is my
    lever (my copies: fewer ops per layer, or building ahead); I price it [P].
  - cpu_f intercept <= 0.5 x nl's: nl is the GPU side + handoffs. It goes to Leap / Breakthrough with both tables; my side
    of the fixed-cost lever closes (KILL).
  - Between: both, reported.

## T2 CHAT WIDTH GATE ENV (Sun 08 Nov 12:23; NewIdeas 12:2x placed the gate as a 12 GB IFBench arm, Monday by day, scored vs NAX6-N's card rows)
- Env file /Users/kamen/Fit12/data/gate/card12_chw_env.txt (EvalSuite's format; it parses with runner 61's parse_env). It is
  not an r57 screen, so it is driven like runner 57.
- Harness frozen at /Users/kamen/Fit12/snap/chw1108 (read-only): stage 2's code. Its fit_t3.py (sha 7a9ffd8375ed) differs
  only in the sys.path line, which loads its own folder.
- IFBench maps through EvalSuite's dec_ifbench.jsonl (300 items, max_tokens 1536).
- Smoke test chw17 (/Users/kamen/Fit12/scripts/chw17.sh) armed after chw16: ifeval/1220 + ifeval/3280 must = chw14c2
  (tokens + lsha); ifbench/10 must decode.

## T2 SMOKE + HISTORY TEST (chw18, pre-registered Sun 08 Nov 15:29, before any run; EvalSuite's question: is P2' history-independent, or does the IFBench arm need 2 orders?)
- What is shown so far: the base ex path is pool-independent (Fri kp2 ex = Thu c51 g1 on 16 / 16 HumanEval, token count +
  sha, other process, arm mix and cache history). P2' adds only >= 2-row calls (2-row first calls, PR1's 2-row replays) and
  one 1-row call at the cap with ALLWAIT. All of these are pool-independent by design, but P2' was never run in two orders.
- chw18 (/Users/kamen/Fit12/scripts/chw18.sh) replaces chw17 (smoke) and chw16 (profile). Both were stopped while they
  waited in their gates; neither had run. One share lock, MEM 12 GB, ~10 min, three processes:
  - chw18s: the frozen harness (snap/chw1108) on ifeval/1220, ifeval/3280, ifbench/10, CAP 128.
    - SMOKE PASS = both IFEval keys = chw14c2 (tokens + lsha; same positions) and ifbench/10 decodes (rc 0).
  - chw18r: the same harness and env on ifbench/10 first, then the 12 IFEval in reversed order, CAP 128.
    - HISTORY PASS = 12 / 12 IFEval = chw14c2 (tokens + lsha) and ifbench/10 = chw18s's (3rd there, 1st here).
    - PASS -> one order is enough for the IFBench arm (evidence over 128-token prefixes; lsha checks the kept rows' logits
      bit for bit).
    - Any mismatch -> history-dependent: EvalSuite runs 2 orders, and I find the pool-dependent call.
  - chw16p: chw16's pre-registered profile ('T2 FIXED COST SPLIT'), same tag, folder, env and live shas (fit_t3 42266985905b,
    t3_twopass 75b89b71b40b).

## T2 SMOKE + HISTORY TEST RESULT (chw18; recorded Sun 08 Nov 16:44)
- Run: share lock 16:36:27-16:42:09, GPU 0 % at start, 49 GB free, swap flat (5558 MB), no other lock in locks.txt, rc 0 x3.
- Data: /Users/kamen/Fit12/data/scr/chw18 (chw18_eq.txt, chw18_hist.txt). Checked again per prompt by a separate script.
- SMOKE: PASS.
  - The frozen harness (snap/chw1108, fit_t3 7a9ffd8375ed) gives stage 2's tokens + lsha on ifeval/1220 and ifeval/3280,
    which ran in the same positions.
  - ifbench/10 loads and decodes: 76 tokens (EOS before the cap), prefill 63 tokens, lifetime 10.906 GB.
- HISTORY: PASS, 13 / 13 prompts bit-identical (token list, sha, kept-row logits hash) in another order.
  - The 12 IFEval prompts ran reversed vs chw14c2's forward order. ifbench/10 ran 1st in chw18r vs 3rd in chw18s.
  - The cache contents really differed: each prompt's decode SSD bytes moved by -1.1 % to +12.5 % between the orders
    (ifeval/2239 +12.5 %, ifbench/10 +2.0 %). The kept rows' logits matched anyway.
  - For EvalSuite's IFBench arm: one order is enough.
  - Scope: 13 prompts and 128-token prefixes. Beyond 128 tokens, the mechanism covers it: P2' makes no pool-dependent call
    (ex reads and waits for all top-10 ranks; the 1-row cap call runs with ALLWAIT).
- Speed (rough, share lock): chw18r 13.63 tok/s geo vs chw14c2 13.89 (x0.981).

## T2 FIXED COST SPLIT RESULT (chw16p, run inside chw18 16:39:13-16:42:09; recorded Sun 08 Nov 16:44)
- Run: rc 0; tokens + lsha = chw14c2 12 / 12. Data: /Users/kamen/Fit12/data/scr/chw16 (chw16_prof.txt, chw16_eq.txt).
- 629 exact calls in 513 checks. Fits: ms = a[w] + b * GB.

| per exact call | 1 row | 2 rows | 3 rows | 4 rows | per GB |
|---|---|---|---|---|---|
| nl (loader idle in t0..t2) | 31.6 | 31.0 | 36.1 | 40.4 | 2.4 |
| cpu_f (main thread CPU in forward()) | 25.3 | 21.8 | 20.2 | 20.9 | 1.9 |
| nl - cpu_f (>= GPU side + handoffs) | 6.3 | 9.2 | 15.9 | 19.5 | 0.5 |
| cpu_f / nl | 0.80 | 0.70 | 0.56 | 0.52 | |

- Calls per width: 3 (1 row), 342 (2 rows), 7 (3 rows), 277 (4 rows). cpu_e (main thread CPU in mx.eval) is 0.3 ms.
- Whole call: 42.4 / 43.9 / 52.5 / 59.4 + 83.7 ms per GB. This repeats chw15's 43.3 / 43.6 / 53.2 / 58.8 + 82.7.
- Verdict: BETWEEN (pre-registered: >= 0.8 means Python, <= 0.5 means GPU side). Both levers are real; both are reported.
- Reading:
  - The main thread spends 21-25 ms of CPU per call in forward() (48 layers, ~0.45 ms per layer), and that doesn't grow with
    width.
  - The loader-idle time grows with width. At least 6-20 ms of it is GPU side + handoffs (more rows: more GPU work per layer).
- Price [P], upper bounds on chat:
  - Python: 23.2 ms per call x 1.226 calls per check = 28.4 ms of a 168.6 ms check.
    - Only the part not hidden behind reads / GPU counts. Removing all of it: <= x1.20; halving it: <= x1.09.
  - GPU side: >= 13.8 ms per call (call-weighted) = 16.9 ms per check. Removing all of it: <= x1.11.
- Next:
  - A causal CPU-pad screen [P -> K]: a known busy-wait at the start of each layer's graph build. The call grows by the pad
    if the main thread is on the critical path, and by ~0 if it is hidden.
  - Send the GPU-side table to Leap / Breakthrough.

## T2 C2c: STRATA'S CHAT WIDTH (pre-registered Sun 08 Nov 19:15, before any run; the orchestrator 19:1x: take what works from Strata, MIT, keep its notice)
- **The two loops compared.** Strata read-only: ~/StrataEval/src/metal/src/engine/stream.h, docs/MAC.md; its numbers are
  from FableGod findings/2026-11-08_1900_strata-low-memory.md.
  - Strata keeps an LRU arena of expert slots. Each expert is one F_NOCACHE pread from a contiguous ".experts" copy, with
    several reads in flight on I/O threads. It reads ahead the experts that layer L+1's router predicts from layer L's FFN
    input.
  - On chat it drafts ONE token per pass (MTP, 2 rows): 1.67 tokens per pass at 115.7 ms per pass. Its smallest cache
    (3.37 GB) peaks at 14.19 GB.
  - Ours: a 2-row verify call + its draft takes 93.5 + 10.4 ms [K]. That is already faster than Strata's pass, at
    10.96-11.0 GB.
  - But our check verifies up to 15 drafts in 4-row calls (157-181 ms each), while chat keeps 2.36 tokens per check and
    40 % of checks keep only 1 [K: chw13a].
  - So on chat the biggest difference is the speculation width, not the read path. Read-ahead is the next difference; it is
    an engine change (the loader + a next-router kernel), not tried here.
- **Price [P]** (/Users/kamen/Fit12/scripts/rows_price.py, a CPU replay of measured calls).
  - Calibration: replays of A' 12.49 vs 12.56 measured, P2' 14.18 vs 13.97 (chat); code A' 18.51 vs 18.69, P2' 18.54 vs 18.47.
  - C2c (one draft per check, then 2-row calls until the first miss): chat 15.63-16.36 tok/s whole decode, x1.10-1.15 over
    P2'; code 18.63-18.88, x1.006-1.02 over A'. The range is the later-2-row-call cost: measured, or the first-call cost.
  - C2 (re-draft after every 2 rows): chat 15.07; code 17.00 (x0.92).
- **Build:** /Users/kamen/Fit12/scripts/t3_twopass.py gets F12_WL, the later-call width (default 4: unchanged).
  - sha 7d424eb4a4fe; backup .bak_wl = 75b89b71b40b. fit_t3 42266985905b.
  - The FAKE dry test stubs generate(), so the first GPU run is the code's first test. Check the later-call widths in the
    per-call logs.
- **Screen chw19** (/Users/kamen/Fit12/scripts/chw19.sh): one share lock, MEM 12 GB, ~10 min, ex + PR1 + tail.
  - chw19c: C2c (F12_W1=2, F12_WL=2), 12 IFEval, CAP 128.
  - chw19p: P2' (cf:0.9:2, WL 4) on the same prompts, as the same-lock reference. Its tokens + lsha must = chw14c2's 12 / 12.
  - chw19h: C2c on 16 HumanEval, CAP 256, vs chw14ha (A', 18.69) and chw14h2 (P2', 18.47).
- **Lines:** whole decode pooled. Whole request + first token are reported beside it.
  - Chat, chw19c vs chw19p:
    - >= +6 %: PASS. C2c becomes the 12 GB chat candidate. It is lossy at rounding level like P2', so it needs its own gate.
      It is history-independent by design (all calls >= 2 rows + the ALLWAIT tail), to be confirmed.
    - +2..+6 %: MARGINAL (reported).
    - < +2 %: KILL.
  - Code: chw19h vs chw14ha >= -2 %. Else C2c is chat-only, e.g. P2'c (cf0 < 0.9 -> C2c, else 4-row calls; 15.26 on chat [P]).

## T2 C2c RESULT (chw19, Mon 09 Nov 16:20:44-16:30:48; share lock, MEM 12 GB, GPU idle before start, rc 0): chat MARGINAL, code FAIL
- **Ran as registered.** /Users/kamen/Fit12/scripts/chw19.sh, three processes in one share lock, ex + PR1 + tail. 12 / 12 and
  16 / 16 prompts counted (PAIR ok). No other session's lock in /Users/kamen/Fit12/data/scr/chw19/locks.txt.
  - Gate: GO at 16:19:56 after Leap's full store conversion wrote its done file (NewIdeas 12:51).
  - Widths (chw19_widths.txt): C2c's later calls really ran at 2 rows (chat 255 later-w2, 1 w3; code 1454 later-w2, 4 w3).
  - Reference repeat: chw19p (P2') = chw14c2 on tokens 12 / 12 and kept-row logits hash 12 / 12 (chw19_rep.txt). Speed
    x0.983 run-to-run (13.73 vs 13.97 pooled).
- **Chat, C2c (chw19c) vs P2' (chw19p), the same lock** [K] (chw19_rows.txt, chw19_chat.txt):

  | run | whole decode tok/s pooled (geo) | whole request pooled (geo) | first token s median (min-max) | lifetime peak GB | MB per kept token |
  |---|---|---|---|---|---|
  | chw19c C2c | 14.46 (14.23) | 10.39 (9.39) | 2.87 (2.28-3.64) | 10.972 | 449 |
  | chw19p P2' | 13.73 (13.64) | 9.97 (9.04) | 3.07 (2.25-3.72) | 10.980 | 491 |

  - x1.053 pooled whole decode (x1.043 geo): inside the MARGINAL band (+2..+6 %), under the +6 % PASS line.
  - C2c changes tokens: = P2' on 4 / 12, logits hash 0 / 12. It is lossy at rounding level, like P2'.
- **Code, C2c (chw19h) vs A' (chw14ha, Sun 8 Nov, another lock)** [K] (chw19_code.txt; chw14ha from
  /Users/kamen/Fit12/data/paper/small_ram_12gb_rows.txt):
  - Whole decode 16.48 vs 18.69 pooled: x0.881. That is far under the -2 % line, so C2c is not a code mode. (x0.892 vs P2',
    chw14h2's 18.47.)
  - Whole request 12.70 vs 14.11 pooled. First token 4.64 s (4.16-6.00) vs 4.48 (4.22-5.64). Peak 11.169 vs 11.145 GB.
  - Tokens = A' on 12 / 16, logits hash 0 / 16.
- **Why the price [P] (+10-15 % chat, x1.006-1.02 code) missed.** A 2-row call costs more than the replay assumed.
  - Measured per call [K] (calls logs, t1 - t0):
    - chat: first w2 98.6 ms (n 486), later w2 107.3 ms (n 255);
    - code: first w2 108.6 ms (n 339), later w2 115.3 ms (n 1454).
  - The replay used about 93.5 ms, chw14c2's first w2. Later w2 calls were too rare in Sunday's logs to measure.
  - Per row, a 2-row call costs 54-58 ms against 46-48 ms for a 4-row call (chat later w4 182.3, code 192.2 ms).
  - Chat saves rows: 1493 vs 1816 verified rows, 449 vs 491 MB per kept token, and calls take 75.9 vs 81.2 s in all. But it
    pays the per-call fixed cost on 746 calls instead of 629.
  - Code keeps 10.1 tokens per check, so C2c needs 1800 calls (3605 rows) instead of 972 (3871 rows), and pays that cost
    on every one: calls take 205.4 vs 180.5 s.
  - Drafts (10.5 ms) and the per-check rest (< 2 ms) did not change.
- **Verdict (pre-registered lines):** chat MARGINAL (+5.3 %), code FAIL (-11.9 %). Nothing is adopted. A chat mode would be
  lossy and need its own gate, and adopting it is Kamen's call.
- **Left open [G]:** P2'c (C2c only when cf0 < 0.9, else 4-row calls) re-priced with today's measured call costs. That is a
  CPU replay; no GPU run without a new line.

> (a quote from the project owner, kept private)
- **Measured on code [K]**, A' (chw14ha, 16 HumanEval, CAP 256; /Users/kamen/Fit12/data/scr/chw14):
  - 342 checks, 972 exact calls (2.84 per check, almost all 4 rows), 3452 tokens (10.09 kept per check);
  - 441 MB per kept token, 540 ms per check: calls 528 (first w4 175.9 ms, later w4 192.2), draft 11.0, rest 1.2.
- **Fit [K]:** one call costs a[w] + 83.7 ms per GB, with a[1-4] = 42.4 / 43.9 / 52.5 / 59.4 ms (chat calls,
  /Users/kamen/Fit12/data/scr/chw16/chw16_prof.txt l21). It reproduces code's w4 calls: about 1.57 GB per call gives
  190 ms vs 176-192 measured.
- **Price [P]:** at zero SSD bytes, the exact check would still take 2.84 x 59.4 + 11.0 + 1.2 = 181 ms, which is 10.09 /
  0.181 = 56 tok/s.
  - So the exact <= 4-row call structure cannot reach 90 on code at any byte count.
  - 90 at ~10 kept per check means a 111 ms check. That needs both of these, and both are lossy (not batch-invariant or
    fewer experts), so both need the gate:
    - (a) ~40-60 ms per check outside reads;
    - (b) <= ~60-85 MB per kept token.
  - Breakthrough's first pricing agrees (/Users/kamen/Breakthrough/docs/PRICE_12GB_90_CODE.md, b69de76).
- **BigMoE's fast path at 12 GB [K, from its W24 header]:** a base slot costs 0.0991 GB and a sidecar slot 0.0346 GB. The rest
  of the lifetime peak is ~10.1-10.6 GB (F24k: 13.19 GB pools, 23.7 GB peak).
  - This build's rest is ~5.1 GB (11.145 GB peak, 6.03 GB pools).
  - So that engine fits 12 GB only if ~5 GB of its non-pool memory goes. Check on CPU first: what the 10.3 GB holds.
- **Split** (Breakthrough 19:0x): Breakthrough prices the levers tonight on CPU (a code-window byte replay at 12 GB). Fit12
  runs the engine screens and the gate.
  - Screens S1 (one wide call per check with a sure-row rank cut that fits C_m 47), S2 (BigMoE's fast path at 12 GB, if
    it fits) and S3 (byte-cut arms) wait for the ranking.
  - NewIdeas gets their GPU minutes, peak and lock type by Tue 08:00. No GPU tonight: W24 has the machine.

## 12 GB -> 90: WHY <= 4 ROWS, AND THE FIRST SCREEN (Mon 09 Nov 21:2x; Breakthrough 19:3x-19:5x, its docs/PRICE_12GB_90_CODE.md 2f79c62)
- **Breakthrough's code byte replay at these pools (51 + 32)** (data/m1/code12_replay.out, 8441677) [P]:
  - exact 5.04 GB per check (this loop's A' measures 4.45);
  - QOM0 + ball 0.81 GB; Q1-t3 + bonus 0.55 GB.
  - The map's code line at 51 slots: QOM0 with the bonus row exact passes, 1.16 % [0.54], at 1.29 GB per check. QOM0 + ball
    fails at 1.40 %. Map lines are information only; the gate decides.
- **Why this loop's calls stop at 4 rows:**
  - Pool (hard): a call's top-10 union must fit C_m 47 base slots (4 x 10 = 40). A 16-row top-4/5 pass already aborted (c52).
  - Gate row (hard): the sync-free gate posts at most 92 ids per layer and call
    (/Users/kamen/Breakthrough/v3p/bigmoe_v3f/gated.py l16-20; default 40). 16 rows x top-10 = 160.
  - Numerics (half known): WV1 (24 Oct) found widths 2-4 bitwise equal and width 1 different. Widths 5-16 are untested.
    Any 16-row route here is lossy through its expert cut anyway, so this only changes its label.
    A width test (S0 WV16) is optional and not on the speed path.
- **Residency.** The loader knows which experts are resident only when ids are posted. post() returns a slot or -1 per id,
  and posting a miss reads it.
  - So true QOM0 ("skip non-resident ranks 2-9") needs a lookup-only probe in the native loader: a copy of Breakthrough's
    C++ extension. It is not buildable in this loop's Python today.
- **Buildable now: S1u, union-only one-call check.**
  - One 16-row call per check. U = every drafted row's ranks 0-1, plus the last (bonus) row's ranks 0-9 (<= 40 ids, which
    fits MAXK 40 and C_m 47).
  - Each drafted row computes its ranks 0-9 that are in U, at the model's weights. The bonus row is exact.
  - Commit pass 1's rows up to the first miss; no exact pass.
  - It generalizes this loop's P1-K1 wide pass (k1u_moe: U = rank 0 there). Bytes are about QOM0's (U's misses only),
    with more dropped compute, so its map line is needed before any gate. Breakthrough was asked 21:2x.
- **S1u pre-registration (before any run):** 16 HumanEval, CAP 256, <= 12 GB, share lock, ~15 min. Measure whole decode
  (pooled, geo), ms per check (call / draft / rest), MB per kept token, kept per check, tokens vs ex.
  - Lines: the one-call route stands if the call's time outside reads is <= 90 ms and MB per kept token <= 150. Else the
    route is re-priced before S2 (overlap).
  - No quality claim: it is lossy and needs its map line, then the gate.
- **S1u BUILT (Mon 21:1x; CPU checks only, no GPU before 23:00: Kamen has the Mac).** Files, shas at arming:
  - /Users/kamen/Fit12/scripts/t3_onecall.py c61419e920ad: a copy of t3_twopass 7d424eb4a4fe + u01b_moe + the wide-branch
    hook + "checks of <= 2 rows take the exact check";
  - fit_onecall.py 1f9e7c587564: a copy of fit_t3 42266985905b, + arm u01b (theta 0, _u01b);
  - s1u_read.py e5df6a7c0370; s1u.sh 45bdb608be31.
  - The live t3_twopass / fit_t3 and the gate snapshot are untouched.
  - Checked: the patch's anchors apply once each to the pinned bt_b12 _install, and the patched source compiles with the
    branch in it. py_compile is clean. The reader on c1u16 gives pooled 44.34 / geo 46.33 = paper_rows' figures.
  - By-product [K, c1u16 records]: Saturday's two-pass wide call (P1-K1 pass 1) costs 127.9 ms per call at 0.486 GB, i.e.
    87.2 ms outside reads, and 117 MB per kept token over the run.
- **Map [Breakthrough 21:1x]:** S1u = the map's Q1-UNION at theta 0. It is not scored yet; its neighbours q1u_0.9_live
  (1.25 %) and qom_0_live_c51 (1.16 %) pass the 1.34 % line. One map arm decides (q1u_0_live), asked of NewIdeas after
  23:00.
  - Fit [its CPU replay 148ad96]: U <= 40 by construction. Sidecar ranks 0-1 peak at 32 = C_l. Exact 16-row calls would
    overflow C_m 47 in 68 % of live checks.
- **Armed** Mon 21:13 (task be69wq8hc), held on WAIT for NewIdeas' GO. Gate per its board: >= 23:00, Kamen idle, NO-SWAP
  >= 20 GB, no quiet ticket / big lock, idle GPU. Start-by Tue 00:58, hard stop 01:15.

> (a quote from the project owner, kept private)
- **How it ran.** /Users/kamen/Fit12/scripts/s1u_now.sh: the pre-registered screen, with s1u.sh's 23:00 / idle / queue /
  GPU-idle gates dropped on Kamen's word.
  - Beside his apps (Chrome, Spotify; nothing closed). NO-SWAP 23 GB, swap 8.6 GB flat, one share lock, 15-min hard stop.
  - S1u (s1uh) ran 22:37:30-22:40:01, rc 0, at GPU 87 % (locks.txt), with no PAUSE inside (16 / 16 "PAUSE 0.0 s").
  - The A' reference (s1ue) was cut by the hard stop: 8 prompts read, 7 counted, one VOID on a 40.4 s PAUSE. A quiet
    PAUSE file has existed since 22:44.
  - Its tokens = chw14ha on 7 / 8 (lsha 8 / 8).
  - CORRECTED 22:5x (NewIdeas' number check): the first read-out's "x3.235 pooled" mixed unpaired sets (16 vs 8) and kept
    the VOID prompt. On the 7 prompts both arms counted [K]: S1u 42.64 vs A' 17.87 tok/s pooled = x2.386 (x2.422 geo).
    The reference ran x0.956 of chw14ha there. S1u's tokens = A''s on 4 / 7 (HumanEval/156, /91 and /79 differ).
    The reader now pairs prompts; the first read-out is kept as s1u_read_v1.txt.
  - The "= ex tokens True" field of a single-arm process compared the arm with itself (vacuous). This affects every
    single-arm line from fit_t3.py too. fit_onecall.py now prints n/a there.
- **S1u [K]** (/Users/kamen/Fit12/data/scr/s1u: s1u_read.txt, s1u_rows.txt; raw s1uh.txt + s1uh_*_u01b_ph.json):

  | | value |
  |---|---|
  | whole decode tok/s pooled (geo) | 41.22 (41.79) |
  | whole request pooled (geo) | 23.21 (22.71) |
  | first token s median (min-max) | 4.18 (3.87-5.49) |
  | lifetime peak | 11.335 GB |
  | prompts / tokens / checks | 16 / 3353 / 325 (2 short checks took the exact path) |
  | kept per check | 10.32 |
  | MB per kept token | 138 |
  | ms per check | 250.3 = draft 11.6 + wide call 217.8 + commit 15.6 + exact 0.7 + rest 4.6 |
  | the check's read / non-read [P: GB x the chw16 fit's 83.7 ms per GB] | 119.5 (1.427 GB) / 130.8 |
  | the wide call | 323 calls, 219.1 ms [K], 1.326 GB [K]; 108.2 ms outside reads [P, same fit] |
  | experts computed per row-layer | 5.161 of 10 (4.674-5.535 by prompt; 245,136 row-layers) |

- **Lines (pre-registered):** outside reads <= 90 ms FAIL (108.2 [P]). MB per kept token <= 150 PASS (138). So the
  one-call route is re-priced before S2.
  - The outside-reads number is the pre-registered fit split (s1u_read.py's docstring, written before the run), not a
    measured stall.
  - It is fit-sensitive. At T3R2's measured exposed rate (~9.6 GB/s, this plan's first table), the call's reads would take
    ~138 ms and its non-read ~81 ms, under the line.
  - So this FAIL is marginal. The wide call's real split needs the loader's io / idle clocks on pass-1 calls; today only
    exact calls have them (F12_GSTAT).
- **Against the price.** Bytes held: 1.326 GB per wide call vs Breakthrough's LRU price of 1.29 for QOM0's misses at
  51 + 32. The non-read part did not: 108 ms [P, fit split] vs the ~50-90 assumed (~81 at a 9.6 GB/s read rate).
  - Context [K]: x2.21 the verified loop's 18.69 (chw14ha, Sun, share lock); x0.93 the two-pass mode's 44.34 (c1u16, Sat,
    quiet), which failed its gate. Saturday's narrower wide call (P1-K1): 127.9 ms at 0.486 GB, 87.2 outside reads.
- **Re-price [P, from these numbers]:**
  - The call still computes all 10 tier_glu columns, with weight 0 on the ~4.8 dropped ranks. Compaction and less Python
    are the first non-read cut (target ~60 ms).
  - Additive, at ~60 ms: 11.6 + 60 + 111 + 15.6 + 5 = 203 ms, ~51 tok/s.
  - Reads hidden behind compute (S2): ~143 ms, ~72.
  - Plus an eviction policy near Breakthrough's OPT bound (0.74 GB per check): ~94 ms, ~110. Additive at that byte count: ~67.
  - So 90 needs the non-read cut, the overlap and better eviction together, or kept 12+.
- **Quality: untested.** Breakthrough's q1u_0_live map arm runs after 23:00, then the gate if it passes. Nothing is adopted.

## S1p: S1u's split, measured (pre-registered Mon 09 Nov 23:1x, before any run; NewIdeas 22:55: S1u's 108.2 ms is a fit split [P])
- **Build (CPU only; checked by py_compile, the pinned-source patch check and a synthetic row file):**
  - t3_onecall.py 652eedf7d804 adds TS1 under F12_GSTAT=1. Per wide pass-1 call it logs stamps t0 / t1 / t2 / t3, the
    loader's proc / io clocks at t0 / t1 / t2, the main thread's CPU and the process's disk bytes.
  - fit_onecall.py 4079a8a1259f writes the rows as <tag>_<key>_<arm>_p1ts.json.
  - Reader /Users/kamen/Fit12/scripts/s1u_prof.py bec67fb27b03. Runner /Users/kamen/Fit12/scripts/s1p.sh bdd45f4ca3b1:
    one process (arm u01b, 16 HumanEval, CAP 256), ~4 min, <= 12 GB, share lock, held for NewIdeas' GO.
- **What it decides:** the S1u line "outside reads <= 90 ms", read on the measured loader-idle time of the wide call
  (wall t0..t2 minus the loader's busy time; chw16's 'nl' convention, a lower bound of the non-read work).
  - It also gives the call's real read rate (GB over the loader's io time) and a busy-vs-GB fit, to replace the 83.7 ms
    per GB in the re-price.
  - Loader-idle <= 90 ms: the S1u line passes, and the re-price uses the measured split.
  - Loader-idle > 90 ms: the non-read cut (compaction, Python) comes before S2's overlap.

## S1p RESULT (Tue 10 Nov 00:01:48-00:04:14; share lock, after W24c, Kamen idle; no other lock, no PAUSE inside (16 / 16), rc 0): the measured non-read time is 102.4 ms, so S1u's 90 ms line FAILS on the measured split too
- **Ran as registered.** /Users/kamen/Fit12/scripts/s1p.sh (S1P_IDLE=1), one process, arm u01b with F12_GSTAT=1, 16 HumanEval
  at CAP 256.
  - S1u reproduced: whole decode 41.79 tok/s pooled (42.46 geo) vs S1u's 41.22. Whole request 23.85. First token 3.97 s
    median (3.65-5.18). Kept 10.32. 138 MB per kept token. Peak 11.314 GB.
  - Read-outs in /Users/kamen/Fit12/data/scr/s1p: s1p_prof.txt, s1p_read.txt, s1p_rows.txt.
- **The wide call, measured per call [K]** (323 calls, 15.8 rows; s1u_prof.py on the p1ts rows), mean (median):

  | wall | build (to graph built) | eval | commit | loader busy | loader io | loader idle | Python CPU in build | GB | misses |
  |---|---|---|---|---|---|---|---|---|---|
  | 217.4 (220.1) | 200.1 (202.2) | 17.3 | 14.9 | 115.0 (116.6) | 113.4 (115.1) | 102.4 (103.2) | 29.1 (28.7) | 1.323 | 896 |

  - Read rate 11.67 GB/s over the loader's io time. Loader busy = 13.2 + 76.9 ms per GB, against the chw16 exact-call fit's
    83.7.
  - Most of the call (200 of 217 ms) passes before the graph is built: the per-layer gate waits sit inside forward().
- **Line (pre-registered):** outside reads = the loader-idle time, 102.4 ms [K] <= 90: FAIL. This run's fit split is
  106.9 ms [P] (S1u's was 108.2 [P]), so the fit was close.
  - NewIdeas' number check (02:1x): the figures hold, and S1p's tokens equal S1u's on 16 / 16 prompts by sha, so the
    one-call check is deterministic run to run.
  - The 102 ms is Python 29 ms of CPU plus ~73 ms of GPU critical path and syncs with the loader idle. That is the
    non-read work a one-call route must cut (compaction of the ~4.8 weight-0 expert columns per row, fewer syncs).
- **Map verdict for S1u's cut [Breakthrough 00:0x, information only]:** q1u_0_live FAILS code: all-row 1.96 % (line 1.34),
  non-tie 1.36 % (line 0.87). It is worse than qom_0_live_c51 (McNemar 39 / 12) and q1u_0.9_live
  (~/Breakthrough/data/m1/SCREEN_read_q1u_0_live.txt). So S1u (union-only) is CLOSED.
  - The route left is QOM0 with a lookup-only residency probe: qom_0_live_c51 PASSES at 1.16 %; Breakthrough has the c47
    masks built and asked for a 16 GB map slot.
  - Breakthrough's eviction replay: the best simple policy cuts bytes 6 % (decayed LFU); OPT 39 %. A small lever.

## IFB12 RUN: the 12 GB chat mode (P2') on the IFBench gate keys (Tue 10 Nov 07:14:32-09:58:01; EvalSuite's amendment 13, 37a03ca; EvalSuite scores it with tools/set_d4.py ifbench vs ref2tnaxn.r6)
- **What ran.** /Users/kamen/Fit12/scripts/ifb12.sh 2143f25c7ff7. The frozen harness /Users/kamen/Fit12/snap/chw1108/fit_t3.py
  7a9ffd8375ed with the env line of card12_chw_env.txt (git blob da1eb63570ee).
  - The 150 gate keys of EvalSuite's results/card/card_keys.txt, CAP = max_tokens 1536, one order, 8 chunks.
  - c01 ran beside the w24 gate (07:14-07:41). EvalSuite then held the arm (SSD heat pauses every ~2 min with both runs),
    so c02-c08 ran alone after the gate (08:41-09:53).
  - Then ifb12rev: the first 10 keys in reverse order (09:53-09:58).
- **Result [K]** (/Users/kamen/Fit12/data/scr/ifb12/ifb12_merge.txt):
  - 150 / 150 keys; finish stop 134, length 16; 57,672 tokens; clashes none.
  - REVERSE CHECK 10 / 10, 0 differ: OK.
  - Largest harness lifetime peak 11.218 GB.
  - Files for EvalSuite: ifb12_tokens.json, ifb12_runs.json, ifb12_meta.json (speeds rough).
- **Guard caveat.** The live footprint probe read 10.737 GB (exactly 10 GiB) in every chunk: footprint prints whole GB at
  that size, so the probe's resolution was 1 GiB. The 12.0 GB rule held on the harness's own lifetime figures.
- **Verdict: EvalSuite's.** Decision 4: paired delta > -5 pt and exact McNemar p >= 0.05, IFBench alone. It gates only the
  12 GB chat row (now "not gated" in SMALL_RAM).

## 12 GB -> 90: QOM0 needs C_m 51, and it fits (Tue 10 Nov 10:0x; Breakthrough 09:16 / 09:18)
- **Map [Breakthrough, information only]:**
  - qom_0_live_c47 (QOM0 at this build's C_m 47) fails code by ONE flip: all-row 1.37 % (46 / 3366; the line is 45),
    non-tie 0.69 % passes.
  - c51 passes (1.16 %). c47 vs c51: 14 / 7 only-flips, z +1.53, not significant.
  - By its pre-registered rule, the residency probe needs the larger pool. C_m 51 cuts QOM0's reads 3.6 % (1334 -> 1286 MB
    per check [P], data/m1/code12_pool47.out).
- **Slot price in this engine [K]:** the pool prints give C_m 47 / C_l 32 = 6.03 GB and C_m 51 / C_l 32 = 6.44 GB, so a base
  slot costs 0.1025 GB over 48 layers.
  - A measured run at 51 + 32: c51, Tue 3 Nov, /Users/kamen/Fit12/data/scr/t3p/c51.txt (4 HumanEval prompts x 5 arms at CAP 256), lifetime peak 11.734 GB.
  - So QOM0 at 51 + 32 fits under 12.0 GB with ~0.25 GB margin. S1u's 11.335 + 0.41 = ~11.75 [P] agrees.
- **Next build:** QOM0 at 51 + 32 with a lookup-only residency probe. That means a copy of the loader's C++ extension,
  Breakthrough's v3p ext, used read-only and built in a Fit12 copy.
  - Plus the non-read cut (S1p: 102.4 ms per call [K]), then the overlap.
- **IFB12 VERDICT (EvalSuite 10:0x; ~/EvalSuite/docs/RESULT_card_ifb12_2026-11-10.md, EvalSuite 74b1dc2) [K]:**
  - PASS on IFBench loose, the gate's metric: 40.0 % vs 44.0 % (NAX6-N), -4.00 pt, McNemar p 0.2101 (arm only 5 /
    reference only 11). Reverse check re-verified 10 / 10; no clashes; hashes unchanged after the run.
  - Information only: IFBench strict 32.7 % vs 40.0 %, -7.33 pt, p 0.0127 (3 / 14): significantly below. The 24 / 45 GB
    configs scored 36.7 % strict on these keys. So no "equal instruction following" claim rests on this arm.
  - 16 rows hit the 1,536 cap (the reference and the 24 GB config: 10 each).
  - Fit12's look at those 16 [K, CPU, ifb12_tokens.json]: share of repeated 8-grams in the last 512 tokens.
    - Near-pure loops (>= 0.87): ifbench/181, /214, /182, /192, /229.
    - Partly repetitive (0.33-0.56): /210, /239, /277, /159.
    - Long but varied (<= 0.21): the other 7.
    - For comparison, stop rows of >= 512 tokens average 0.04.
  - So part of the extra cap hits are repetition loops. Whether the reference loops on the same keys is EvalSuite's data.
  - Label for SMALL_RAM's 12 GB chat row: "lossy, same scores on IFBench (loose, the card gate's metric); strict 7.3 pt
    below (information only)". The editor's call.

## S1q: QOM0 at C_m 51 with a residency probe (pre-registered Tue 10 Nov 10:3x, before any run; built in the Fit12 copies)
- **Why a probe, and why it must post.** The loader publishes, per (pass, layer), the residency table that post_ids reads on
  the GPU (bigmoe_ext.cpp l494-533; tab_offset l649).
  - But make_room evicts any LRU expert that is not in the CURRENT post (l351-360).
  - So an expert that is resident but not posted can lose its slot while this call reads it. QOM0's extra experts must be
    posted.
  - Posting them is free (they are hits), and it protects them (they join make_room's 'need').
- **Build.** t3_onecall.qom0_moe (arm q0r in fit_onecall.py).
  - A Metal probe kernel reads that table (gate.tables at tab_offset(L, gate.seq)) for the drafted rows' ranks 2-9. It
    appends the unique residents not in U to U's post: one post of 92 ids, so BIGMOE_MAXK=92.
  - Each row computes its posted ranks. The bonus row is exact. Commit from that call.
  - At MAXK 92 calls of <= 4 rows are narrow, so those checks take the exact check.
  - Self-check: every resident the probe adds must show a slot in post_ids' own dep too ('probe misreads').
- **Pool:** C_m 51 / C_l 32 (the map's passing pool). It measured 11.734 GB lifetime in c51 (3 Nov).
- **Screen:** /Users/kamen/Fit12/scripts/s1q.sh, one process, 16 HumanEval at CAP 256, F12_GSTAT=1, share lock, ~4 min,
  <= 12 GB.
  - Lines: VALID if probe misreads = 0 and lifetime peak <= 12.0 GB. Then it can go to EvalSuite's gate (HumanEval+ /
    MBPP). Speed is reported, not lined.
- **S1q RESULT (Tue 10:26:57-10:29:22; share lock after an idle-GPU gate, no PAUSE inside, rc 0; the orchestrator's GO 10:2x): VALID.**
  - Attempt 1 (10:25) aborted at load ("bigmoe: 55 experts > C=51 at layer 0": a post's distinct experts must fit C_m).
    The probe now adds at most C_m - |distinct U| residents and pads the rest; re-unit-tested.
  - Lines [K]: probe misreads 0 in 16,128 layer-calls; lifetime peak 11.765 GB <= 12.0. VALID.
  - Residents added: 14.14 per layer-call. Dropped for room: 1,392 (0.086 per layer-call).
  - Speed [K]: whole decode 41.77 tok/s pooled (43.20 geo), whole request 23.97, first token 4.02 s median, kept 10.01,
    132 MB per kept token.
  - Wide call 215.0 ms: loader busy 111.2 (1.260 GB at 11.53 GB/s; -4.8 % bytes vs S1p at C_m 47) + loader idle 103.7. That
    is +1.3 ms over S1p's 102.4 for the extra experts.
  - So QOM0 at 51 + 32 runs as fast as S1u, with the cut the map passes (1.16 %, information only). Next is the gate's call
    (EvalSuite, HumanEval+ / MBPP). Toward 90, the 104 ms non-read time is still the wall.
- **S1q CODE GATE (Tue 10:41-11:02; EvalSuite PREREG_code12_s1q_2026-11-10, 89d6938; route (i)): STOPPED by the 12.0 GB guard, NO verdict.**
  - fwd ran 118 / 160 keys (60 HumanEval+, 58 MBPP); rev never started.
  - The live guard ('footprint -f bytes', exact) read the peak at 12.027 GB in chunk qgf06, on mbpp/228 (126-token prompt,
    cap 1,024). Ten seconds earlier it read 11.673 GB.
  - The harness's lifetime figures on completed prompts stayed <= 11.856 GB.
  - So S1q at C_m 51 / C_l 32 does not stay under the 12 GB line on MBPP's longer outputs. A retry needs less memory, and the
    map line again if the pool shrinks (QOM0 at C_m 47 failed by one flip).
  - Also: s1qg_merge.py looked for arm "ex" while the chunk files say "q0r" (KeyError). Any resume needs that fixed, as an
    amendment, since the merge's hash is registered.
- **S1q memory pre-checks (Tue 11:10-11:16; share lock, idle GPU; /Users/kamen/Fit12/scripts/s1m.sh, s1m2.sh; data
  /Users/kamen/Fit12/data/scr/s1m, s1m2) [K]:**
  - mbpp/228 alone, in a fresh process: footprint peak 11.383 GB (exact); 348 tokens.
  - The gate's qgf06 chunk again (fwd keys 101-120, one process): peak 11.725 GB; tokens = the gate's qgf06 run on 18 / 18
    keys, so S1q is deterministic given the history. Order does matter: mbpp/228 makes 222 tokens as the 19th key.
  - Per prompt, MLX's own memory is flat: active 10.09-10.28 GB, peak 10.40-10.46 GB, cache <= 0.083 GB. Nothing builds up
    in MLX. The rest of the footprint (~1.0-1.3 GB) moves run to run.
  - The same computation peaked at 12.027 GB in the gate (10:58) and 11.725 GB here (11:16): ~0.3 GB of footprint noise.
  - So "<= 11.8 GB on mbpp/228" is met (11.38 alone, 11.73 in the chunk) but gives no margin. To stay under 12.0 GB in
    every chunk, the arm needs ~0.4 GB of real cuts, i.e. ~11.3 GB in a chunk re-run.
  - Candidates [G], none built: the drafter's bf16 parts at 4 bits (the drafter is 0.928 GB of own params); MLX cache cap
    64 -> 16 MB (<= 0.07); smaller chunks (the per-process peak ratchets over ~3 prompts).
