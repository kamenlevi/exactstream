# WALL 1 SPRINT SCOREBOARD: make the SSD feel like RAM (NewIdeas keeps it; BigMoE owns the numbers)

> (a quote from the project owner, kept private)

Rule of the sprint:
- Every row gives the best MEASURED number and the next concrete step. No "impossible" verdicts: when a lever dies,
  the next one goes in.
- Exact first (bitwise to its reference); FAST rides on top.
- Marks: [K] measured (source named), [P] arithmetic, [G] judgement.

## Targets

| line | target | today [K] | best measured so far | next step |
|---|---|---|---|---|
| 45 GB per-call overhead | <= 5 ms per check | ~20 ms (0.28 ms x ~70-77 calls, Fable A's fit) | - | JOINT + keepalive tonight (c45tj, ka1); Wed 13:45+ lever 1's arms (prediction: fixed ~9.5-12.5 ms, BETWEEN); then zero-copy Z3b on the converted store (1g) |
| 45 GB exposed SSD | <= 6 ms per check | ~24 ms (0.56 x 42 busy, FIXED-PART) | - | the pre-route recall tonight (pr_224) -> forecast reads into the tier |
| 45 GB steady exact tok/s | +25 % (>= 79.5) | 63.6 | 63.6 | the stack below, one step per night |
| Night (~52 GB, volatile tier) SSD MB per kept token | <= 33 | ~62 (C 256) | - | tier smoke tonight (vt1_224) |
| 24 GB exact tok/s | 1.5x | BigMoE measures first | - | after the 45 GB steps |

> (a quote from the project owner, kept private)

**RAM stops deciding the speed.** Two lines, judged per track (EXACT / FAST) and per RAM line (45 / 24 / 12 GB):
1. **The GPU never waits for the SSD:** exposed SSD <= 1 ms per check at 45 GB (stall_report's cadence excess, W1B's six
   prompts, steady). Today ~24 ms [P: 0.56 x busy 42.0].
2. **SSD MB per kept token <= 10** (untraced W1B means), first at 24 GB, then at 12 GB. Today 62.4 at 45 GB (C 256) [K];
   the 24 GB number comes from BigMoE's W1B baseline (Wed 13:45).
   - Why 10: at the SSD's ~12.7 GB/s effective, 10 MB per kept token leaves >= 1,270 tok/s of disk headroom. The disk
     then stops being the limit; the drafter (kept per check) and the dense pass decide the speed.
- Not BROKEN: any +20-30 % step that leaves MB per kept token above ~10 or exposure above a few ms. Those go in the rows
  below as progress.
- The ledger behind the line: research/TEN_X_LEDGER.md (due Wed 13:30): each multiplicative factor (kept per check,
  per-request locality, skips, precision, format, tier) priced on its own, per domain, EXACT vs FAST, and their product
  per RAM line.
- First signal [P] (capture, live-cut checks, C 256): FAST QOM at cf 0 + 3-bit sure rows reads 6.5 / 4.6 / 4.6 MB per
  kept token on code / IFEval / multi (EXACT 47.0 / 31.1 / 30.7). That is under line 2 at 45 GB, IF its quality holds:
  cf 0 runs ~05:00, the 3-bit bound is pre-registered. EXACT has no factor near 10 yet.

## One row per night

Columns per the charter and BigMoE's W1B protocol (~/BigMoE/docs/WALL1_BENCH.md: S45's six prompts, CAP 256, every token
== the stored REF2-T reference).
- Traced arm w1t_<C>: SSD busy, exposed (cadence excess), loader bursts per check, and the fixed-vs-bytes split.
- Untraced arm w1_<C>: MB per kept token and steady tok/s (means of the six).
- Rows per line (45 GB, then 24 GB), plus the next step.

| night | config | SSD busy ms per check | exposed ms | loader calls per check | MB per kept token | tok/s steady | exact? | next step |
|---|---|---|---|---|---|---|---|---|
| baseline (Mon 2 Nov) | default: C 256 + OOO + K-73 + M-17 sizes (BigMoE c45l_256) | 42.0 [K] | ~24 [P: 0.56 x busy] (gate spin 42.6 [K]) | ~70-77 [P, Fable A's fit; replay 70.8] | 62.4 [K] | 63.6 [K] | yes | tonight's C 224 steps |
| Wed 4 Nov, C 160 (BigMoE's one quiet ticket 00:30-03:07; ~/BigMoE/docs/NIGHT_1104.md) | traced c45t_160, untraced ka0_160; uniform C_m 164 + C_l 112 | 110.4 [K] (bytes 92.3 + fixed 18.1: 76.5 bursts x 0.237 ms) | 112.0 [K] | 76.5 [K] | 169.4 [K]; volatile tier 64/32: 118 (-30 %) [K] | 42.0 [K] | yes (every token == REF2-T) | lever 1 at 13:45: its ceiling at C 160 is the 18.1 ms fixed part (11 % of the 160 ms check); the wall at C 160 is bytes (BYTES10) |

**Wed 4 Nov read-out (NewIdeas 04:1x) against REVISION 2 (registered Tue 20:34, before any run).**

| item | my prediction [P] / [G] | measured [K] | verdict |
|---|---|---|---|
| loader calls per check (EXACT, C 160) | 77.2 (calls_count.py) | 76.5 | hit |
| SSD busy per check | 108.4 ms (replay mean) | 110.4 ms | hit |
| bytes share of SSD time | 80 % (86.5 of 108.4) | 84 % (92.3 of 110.4) | hit |
| MB per kept token (C 160 uniform) | 171.5 (victim_tier VT_BASE) | 169.4 | hit |
| JOINT (c45tj vs c45t) | busy -3.2..-4.1 ms; bursts 77 -> 45 | stall +2.0 ms, loader service +0.7 ms; per-burst fit 0.237 -> 0.286 ms. JOINT did run (45 % of sidecar reads issued early); overlap-aware bursts 76.5 -> 68.1 (BigMoE cd525aa; stall_report had counted base and sidecar apart). Layer span 2.486 -> 2.489 ms | MISS: KILL at C 160 by BigMoE's own line. My premise failed for another reason: without JOINT the sidecar read already starts 17 us after the base burst (the engine overlaps the pair), so there was no per-pair wake for JOINT to save (M-28's -120..-155 us was measured on separate calls with idle gaps). Corrected 04:4x (my 04:1x text said 'no merge': wrong) |
| keepalive 50 us | about -2.0 ms (with JOINT) | about -3 ms per check on the 4 clean prompts (-2..-4; two one-off spikes in the A arms); tok/s geo +3.9 % (noise ~10 %) | hit (untraced; Wed's traced w1tk decides the -0.04 ms intercept line) |
| volatile tier 64/32 (vt1_160) | 171.5 -> 106.3 MB (-38.0 %) | 169 -> 118 (-30 %), exact; tok/s only +3.1 % (the copy out of the volatile slots costs nearly what the read did) | direction right, 8 points high; the time gain needs zero-copy vtier (BigMoE's next) |
| pre-route recall (pr_160) | p3 ~0.75-0.85 [G] | p3 0.565, p6 0.566 (R-57's own predictor 0.634) | WITHDRAWN (BigMoE 05:5x, 958ea4b): the scoring read L+1's router input on the attention-side HC read; the router sees the MLP side (MTP Head's catch). Fixed and CPU-tested; re-run = WED_TESTS 3b (~4 min, no timing). Verdict pending |
| U3 (FAST) | Q1 0.9 + 3-bit -48 % bytes [P, C 224] | crashed at start (fixed a2584ff; WED_TESTS 5a) | no data |

Also measured: the DAY9 chat switch +21.6 % geo on 20 chat prompts, exact (3,902 / 3,902 tokens); S0's zero-hit class
is empty (0 of 10,224 base calls), as my replay said (0.00 per check); PREFILL_WARM KILL on time at C 160; 32-row checks
KILL (tokens differ, code -62 %).

**REVISION 3 of the lever-1 predictions (Wed 04:1x, before the 13:45 run), after the night's JOINT result.** The w1tj
row above assumed each merged pair saves a wake. The night showed the engine already overlaps the base + sidecar
pair (the sidecar starts 17 us after the base burst), so JOINT has nothing to save at C 160. New calls:
- w1tj JOINT: busy +-1 ms (no gain); KILL by BigMoE's line at C 160 (BigMoE rechecked the build 04:1x: it ran).
- Lever 1 overall (w1uf vs w1u), without zero-copy in the stack:
  - fixed part 18.1 - keepalive ~3 - COPY_PAR 0.3-0.9 - pickup ~0.6 = ~13.6-14.2 ms: above the 10 ms PASS line;
  - steady +2.5..+5 % (per check -4..-8 ms of ~160).
  - Call: BETWEEN 55 % / KILL 35 % / PASS 10 %.
- With zero-copy (1g, Z3b, -2.4..-5.0 ms at C 160) also in, the fixed part is ~8.6-11.8 ms. Call: BETWEEN 55 % /
  PASS 35 % / KILL 10 %.
- At C 160 the fixed part is 11 % of the check: lever 1 is capped near +12 % even if it removes all of it. The wall at
  C 160 is bytes (92 of 110 ms); the BYTES10 ledger's levers are the ones that move it.

Morning sources (BigMoE runs them; I read the outputs, I do not run BigMoE's scripts):
- stall_report.py data/r57 c45t_224,c45tj_224: per-burst fit, demand SSD busy split into bytes and fixed cost,
  bursts after SSD idle gaps;
- pr_report.py pr_224: pre-route recall;
- la45_tue.sh's ka0 / ka1 / ka0r / ka1r and vt0 / vt1 lines;
- data/r57/u3*_224_runs.json: the FAST U3 bytes.

## Predictions for Wednesday's lever 1 run (BigMoE's WED_TESTS item 2; registered Tue 22:5x, before any run)

Arms, stacked, on ext/build_w1 (scripts/w1_lever1.sh after 37b9f8b): w1t default -> w1tj + JOINT -> w1tk + keepalive 50
-> w1td + COPY_PAR -> w1tc + coherent one-row spin -> w1tp + system-scope post; then untraced w1u vs w1uf (all on).
C 160 if NO-SWAP < 44 GB (likely by day), else C 224. Inputs:
- calls_count.py: JOINT bursts 45.1 / 43.4 per check at C 160 / 224.
- burst_mix.py (data/sprint/burst_mix_c160.txt, _c224.txt; replay, EXACT):
  - base loader calls per check 44.4 / 39.8; with 1-3 jobs (the only calls COPY_PAR touches) 11.4 (26 %) / 21.4 (54 %);
    jobs per base call mean 10.6 / 4.7;
  - sidecar reads 108.3 / 133.0 per check (under JOINT each is its own 1-job call);
  - zero-hit layers: base 0.00 per check, sidecar 0.35 / 0.39 (all in 2-3 % of checks).
- M-29 [K]: copy tail 122 us per lone base record, 81 split three ways; keepalive 50 saves 53-75 us per 2-record call.
- Leap's W1-PICKUP [K]: pickup ~+29 / +54 / +104 us at 0.6 / 1 / 2 ms waits -> ~3 us.

| arm (stacked) | prediction [P] | the arm's line | my call |
|---|---|---|---|
| w1tj JOINT | SSD busy -3.2..-4.1 ms per check (REVISION 2 below); bursts 45 / 43 | BigMoE: busy <= -8 ms AND bursts <= 50 | busy line MISSED (Fable A's rev 2 line: PASS / BETWEEN boundary); bursts PASS |
| w1tk + keepalive 50 | burst fit intercept -0.044..-0.075 ms; busy -2.0..-3.4 ms | d -0.04 ms | PASS |
| w1td + COPY_PAR | -41 us only where a 1-3-job base call ends the burst: busy -0.3..-0.9 ms | (none set) | within noise; the real copy lever is zero-copy (1g) |
| w1tc + coherent spin | cadence a - burst a falls 0.03-0.10 ms (longer waits at C 160 gain more); -1.4..-4.5 ms per check | falls >= 0.03 ms | PASS (boundary at C 224) |
| w1tp + system-scope post | ~-13 us of 'see' per burst = ~-0.6 ms per check; no ms-long stalls, because the replay has ~0 zero-hit base layers | (none set; S0's zero-hit split decides) | small. If S0 shows >= 0.1 zero-hit base layers per steady check, the replay's cache model is wrong and this arm is worth more |
| w1uf vs w1u (lever 1 overall) | per check -5.8..-11 ms -> steady +4.5..+9 %; fixed part ~9.5-12.5 ms | exact AND fixed <= 10 ms AND steady >= +5 % = PASS; < +1 % = KILL | BETWEEN 60 % / PASS 30 % / KILL 10 %. Fable A's own budget leaves 12.0 ms after S1 + S2 with S3 dead; zero-copy (1g, Z3b, revised -2.4..-5.0 ms at C 160) then reaches ~5-10 ms |

Fixed part arithmetic: (77.2 calls x 0.283 - 32.1 merged pairs x 0.120..0.155) x 0.82 (replay -> measured, C 256) =
13.8..14.8 ms after JOINT, minus keepalive 2.0..3.4 and COPY_PAR 0.3..0.9 = ~9.5..12.5 ms at C 160 (C 224 the same within 0.5).

## Predictions for tonight (registered Tue 18:4x, before any run)

**REVISION 2 20:34 (before any run): tonight runs at C 160, not C 224.** BigMoE re-armed it as one ticket after
Breakthrough's 00:30 map prefix (~/BigMoE/HANDOVER.md 'TONIGHT ... ONE TICKET AT C 160').
- C 160 has no fitted per-layer sizes, so the engine uses uniform pools: C_m 164 + C_l 112 per layer (the C-unit
  budget of laysz45_256; BIGMOE_TIER_CL 112).
- My replays at those pools:
  - calls_count.py: EXACT 577.8 misses per check, 77.2 loader calls -> 45.1 with JOINT; SSD busy 108.4 ms on the
    replay's mean, of which the fixed cost is only 20 % (bytes 86.5).
  - **JOINT: 32.1 merged pairs x 120-155 us [K M-28] = -3.9..-5.0 ms on the replay's mean, ~-3.2..-4.1 scaled [P].**
    That is again at Fable A's rev 2 line (PASS <= -3.5).
  - **Keepalive with JOINT: 45.1 x 44 us = -2.0 ms [P].**
  - **vt1_160 (uniform C 160 + 64 / 32): SSD MB per kept token 171.5 -> 106.3 (-38.0 %) [P]**
    (scripts/victim_tier.py VT_BASE / VT_PLUS); with Q1 0.9, 90.6 -> 60.0.
  - pr_160: same guess as pr_224 (p3 ~0.75-0.85 [G]).
- At C 160 the bytes dominate the SSD time (80 %), so the per-call levers move a small share. The row reads the
  absolute ms against Fable A's lines.

**REVISION 19:15 (before any run), after MoonShots' M-28 SSDMICRO and Fable A's WALL1_D1_SPEC rev 2.** M-28 measured
JOINT at -120..-155 us per merged pair, not the 283 us per call my first number assumed.
- JOINT at C 224: 76.2 - 43.4 = 32.8 merged pairs per check -> -3.9..-5.1 ms on the replay's mean -> **-3.2..-4.2 ms
  scaled to the measured medians [P]**.
- Fable A's rev 2 line: PASS SSD busy <= -3.5 ms AND bursts <= 50 AND tokens identical; BETWEEN -1.5..-3.5; KILL > -1.5.
  My call: the PASS / BETWEEN boundary.
- Keepalive with JOINT: +44 us wake removed per burst after >= 0.25 ms idle [K M-28] x 43.4 bursts = **-1.9 ms [P]**
  (Fable A: -2.0). Fable A's line: PASS <= -1.5 ms.
- The 18:4x numbers below stay on record as the first registration.

**JOINT, c45tj_224 vs c45t_224.**
- Replay (scripts/calls_count.py; cap45 live checks, laysz45_224 pools, LRU + 30 x popularity): loader calls with
  misses 76.2 -> 43.4 per check (base rows 39.8 + LSB rows 36.4 -> one call per layer with any miss).
- At Fable A's 0.283 ms per call that is -9.3 ms of SSD busy per check on the replay's mean.
- Scaled to the measured medians (replay mean 51.4 vs C50's 42.0 at C 256: x0.82): **-7.6 ms [P]**.
- Fable A's lines: PASS <= -8, BETWEEN -4..-8, KILL > -4. My call: PASS / BETWEEN boundary.
- Tokens must equal REF2-T.

**Keepalive, ka1 vs ka0.** Fable A: -0.06 ms per call when every call pays the idle-gap wake. At 43-76 calls that is
-2.6..-4.6 ms of busy [P]. BigMoE's line: >= +2 % steady.

**Volatile tier, vt1_224** (C 224 + 64 base / 32 sidecar slots, ~7.8 GB). scripts/victim_tier.py on cap45, tier never
purged: SSD MB per kept token **79.2 -> 52.1 (-34 %)** vs C 224 alone; with Q1 0.9, 50.4 -> 34.9.
- BigMoE's lines: PASS = MB per kept token <= -10 % with tok/s not down; KILL = MB down < 3 % or tok/s down > 2 %.
- Tokens == REF2-T.

**Pre-route, pr_224.** BigMoE's line: recall p3 >= 0.85 (recall@10 per row), p6 >= 0.90, precision@96 >= 0.80.
- Prior: MTP Head's L-input forecast recall@10 is 0.611 per row; a check's rows name 50.5 % of L+1's misses at 3.49
  reads per miss.
- My guess [G]: the partial-sum pre-route sits between that and the line, p3 ~0.75-0.85. If it misses, the charter's
  next step is MTP Head's learned correction.

**FAST beside, u3 vs u3c** (Fable C's replay x0.54 of live code bytes). At C 224: Q1 0.9 + 3-bit sure rows reads 265 vs
505 MB per check on the replay (-48 %); calls 67.0 vs 76.2. U3 cuts bytes much more than calls, so JOINT matters more
under U3: its busy fixed share is 48 % vs EXACT's 35 % [P].

## Lever inventory (exact first), with the price and next step

| # | lever | price [K / P] | status | next step |
|---|---|---|---|---|
| 1a | JOINT: one burst per layer | calls 76.2 -> 43.4 at C 224; -7.6..-9.3 ms busy [P] | built, first 45 GB run tonight | c45tj_224 tonight |
| 1b | SSD keepalive (4 KB / 50 us) | -0.06 ms per call [K M-20] -> -2.6..-4.6 ms [P]; M-29 [K]: real spacing up to ~340 us (usleep 150) still holds the penalty <= 10 us; keepalive 50 saves 53-75 us per 2-record call | built | ka1 tonight; Wed w1tk |
| 1b' | wake-ahead ping: one 4 KB read ~200 us before a burst instead of a trickle (MoonShots M-29) | PASS micro [K]: removes 97.5 % of the +48 us wake (T 100: 22 %, T 400: 78 %); ~46 reads per check instead of ~300 | not built; same gain as 1b, less SSD traffic | an engine flag (ping at the router launch) if ka1 passes; tag adapted |
| 1c | demand records in >= 256 KB parallel pieces | KILL [K M-28 L1]: pieces are slower, +39 us per record in 8 pieces | dead | none (the record-level pipeline already exists: each record's copy overlaps the next read) |
| 1d+ | W1-PICKUP: a coherent(system) spin instead of the 128-pair release + polled spin (Leap, TABLE_350 'W1-PICKUP', 20:34) | PASS (microbench, 4,200 trials correct): release + pickup ~23-80 us -> ~3-4 us per burst, flat in the wait [K]; at ~45 bursts per check -0.9..-3.4 ms [P] | patch handed to BigMoE (patches/w1_pickup) | into the engine; measured in a W1B arm |
| 1e | the ~96 us per burst outside the I/O path (last record's copy ~53, release ~8, pickup 0-40, plan ~10-30) [K M-28 L2 split: drive-side 187 of 283 us] | S0 instruments it; S3 zero-copy reads into page-aligned slots -55-60 us per burst bound | S0 Wed (BigMoE), S3 after the conversion | Fable A's D1 floor ~7.4 ms fixed per check (46 bursts x ~160 us) |
| 1f | COPY_PAR: the record copy split over 3 threads in calls with < 4 jobs (BigMoE 37b9f8b) | 122 -> 81 us per lone base record [K M-29]; replay: only 26 % (C 160) / 54 % (C 224) of base calls qualify -> -0.3..-0.9 ms busy [P] | built | Wed w1td |
| 1g | zero-copy: the read lands in the slot, no bounce copy (M-29, M-29b) | per call -113 us (1 record) / -173 (2) [K M-29]; at 4 / 8 / 15 records the gain swings run to run: Z3b -21..-45 / -8..-61 / -60..-158 us [K M-29b + its 22:47 repeat]. **REVISED Tue 22:5x (before any engine run):** on burst_mix's call sizes -> **Z3b -2.4..-5.0 ms per check at C 160, -2.8..-4.2 at C 224 [P]** (the first registration's -4.1..-7.8 assumed 1-2-record calls) | NOT built. Today's unaligned direct preadv is +843 us (the direct arm is out, 37b9f8b). **Z3b first**: 4 KB-padded pieces read with per-piece iovecs into 4 KB-aligned slots; BigMoE's conversion now writes native + p11 + pad, so the disk change rides Thursday's rewrite (MoonShots 22:47). Z1 (one read into a contiguous record-image slot, no disk change) pays only on 1-2-record calls (-104 / -154..-165 us) and loses at 4-15 (-27 / -6 / +23 us): 45 GB line's small calls only. Both need the expert kernels to take a slot stride | Z3b on the converted store (after Thu's conversion). After lever 1's Wed arms (fixed ~9.5-12.5 ms) it brings the fixed part to ~5-10 ms |
| 1h | head ping: a 4 KB read of a forecast record's head to warm the drive's map / NAND page before its full read (M-31) | KILL [K]: the later full read is never shorter (+0.6..+3.2 us with the keepalive on); the first byte has no warmable map or page cache | dead | none: only overlapping calls (lever 2) hide the first byte |
| 1d | release pickup (tighter spin on short stalls) | ~-2 ms [G] | knobs exist (SC5) | after 1a-1b |
| 2 | forecast reads join the open burst, landing in the tier | exposed 24 -> 6-8 ms bound [P]; M-30 QUEUEMIX [K]: forecasts MIXED into a demand burst cost it ~165 us per record, CHAINED after it nothing; a busy drive saves the demand only the ~50 us wake, never its first byte; IOPOL_THROTTLE shields the demand but delays forecasts 1.6-2.5 ms | needs the pre-route recall | pr_160 tonight; the design: forecasts chained right after each demand burst at normal priority (never mixed in, never throttled); then Fable A's CPU exposure replay (<= 8 ms) |
| 3a | volatile tier | -34 % MB per kept token at +7.8 GB [P] | built | vt1_224 smoke tonight; Wed by day the full run |
| 3b | native layout + 11-bit packed scales | -17.6 % code bytes at C 256 [P] | Kamen-approved | conversion Thu >= 15:00; test Fri 00:30 |
| 4 | M-01 pack (12-bit scale fields, already on disk) | -4.8 % bytes [K b44] | the 45 GB preset reads unpacked records3 | one run c45m_256 with BIGMOE_M01=1 |
