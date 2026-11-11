# GAPS: the gap tree, from today's numbers to each target and to the physical limits (NewIdeas, 2026-10-21 12:45)

Method (NEWIDEAS_ROLE.md): per target, what must be true, today's measured number, every factor between them with its
multiplier, the biggest factor, the "why" chain down to an unquestioned assumption, and the mechanism aimed at it.
Machine: ~300 GB/s RAM, SSD 14.6 GB/s at QD32 (loader bursts reach 6-7.6), ~0.34 ms per cold read, 3.5-7.6 us per
dependent kernel. Model: 480 routed expert uses per token, 2.77 MB each at 4 bits (2.15 MSB-3), dense ~3.4 GB per pass.
Sources: MoonShots FINDINGS 1-11, BigMoE HANDOVER, Breakthrough SCOREBOARD/speed_path, MoEPrefetch FINDINGS.

## G. Repetition guard: the shipped loop guard overrides the model in ~1 in 5 long math answers (NewIdeas replay, 10-22 15:55)
Trigger: Breakthrough 15:45 (math500 prealgebra/1640: the guard armed on a normal formula 12-gram and steered '\times 32' into
'\times 16'; the answer then ran to the cap). Replay (scripts/rg_replay.py, CPU only: MTPLX's real LoopGuard class driven step
by step over stored GUARD-OFF decodes; data/rg/replay_*.json). A greedy output can change only where the guard subtracts a
penalty from the token the model chose (penalties on other tokens cannot move the argmax), so rows without such a "hit" are
provably unchanged. Validated on Breakthrough's pair: the replay arms at 736 with a 16-logit penalty; the real guard-on output
diverges at 740.
| guard-off run | rows | shipped rule (R0): rows hit (math / other), of them >= 8 logits | R1: LaTeX math masked | R2: R1 + arm only on >= 50 % repetition | real loops caught R0 / R1 / R2 |
|---|---|---|---|---|---|
| ref (4-bit) | 367 | 13 (9 / 4), 11 | 4 (0 math) | 0 | - |
| plan4433333333 (main track's 3-bit plan) | 234 | 13 (10 / 3), 12 | 3 | 1 | ifeval/1219 at 320 / 320 / 480 |
| floor | 247 | 11 (8 / 3), 11 | 3 | 1 | - |
| w2f_r10 | 210 | 10 (9 / 1), 7 | 1 | 0 | - |
| b12_noguard | 148 | 12 (10 / 2), 10 | 2 | 1 | - |
| Breakthrough's synthetic loop | 1 | | | | 256 / 256 / 256 |
Totals: R0 59 of 1,206 rows, 46 of 250 math rows (18 %), 51 rows at >= 8 logits (mostly the 16-logit cap: the chosen token is
effectively forbidden). R1 13 rows, 0 math. R2 3 rows, all IFEval prompts that ask for repetitive structure (rhyming dialogue,
HTML limericks); both loops still caught (ifeval/1219 160 tokens later).
Why: the arming test (any 12-gram 4x in 2,048 tokens) is met by normal math (a formula prefix restated at each step), and DRY
steering then penalises verbatim restatements of >= 12 tokens (16-logit cap by ~19 tokens). Math, JSON, HTML and non-Latin
scripts (several byte tokens per word: mgsm/te rows) repeat legitimately; MTPLX already masks tool-call spans for this reason.
CORRECTION 16:00 (loose-loop test, rg_replay.py --loose: a 16-token connective sentence every 16 + K tokens of fresh prose,
MTPLX's chess-marathon pattern): R2 NEVER arms once K >= 20, so R2 is NOT the fix. R1 arms and hits exactly where the shipped
rule does on every loop tested (tight, loose up to a 336-token cycle, ifeval/1219, synthetic). R3 (R1 + arm on R2's coverage OR
one unmasked 12-gram >= 8x): 4 rows of ~1,200 touched, loose loops caught 2-4x later (cycle 176: 1,420 vs 716), cycle 336 missed.
RECOMMENDATION: R1 now (LaTeX math spans masked like tool-call spans; published parts: DRY steering, p-e-w 2024, and MTPLX's
own span masking; new here: the math span). It removes every math misfire (0 of 250 math rows) with no recall loss; 13 rows of
~1,200 keep a misfire (HTML / JSON layouts, a code comment, Telugu). R3 only if those matter more than late arming on loose loops.
Known limit: a loop made only of math (repeated equations) would no longer arm; none of the ~1,200 stored rows has one (every
capped math row is long reasoning, repetition share < 0.6). One function in the guard, no decode-speed effect. BigMoE owns it (read-only for me); check on the board (TASKS.md): decode only
the 13 flagged rows of the plan run + ifeval/1219 with R0 and R1 on the main engine (~10 min GPU); expect R0 to change most of
the 13, R1 only the 3 IFEval layout rows, and 1219 still stopped. Changing the guard is BigMoE's / Kamen's call.
B12 verdict: in the no-guard run's own outputs the guard could touch only 12 of 148 rows (10 math, 2 IFEval), yet the guard and
no-guard runs differ on 139 rows: B12's frugal pool carries history across rows. So most of IFEval -3.1 (no guard, PASS) vs -5.0
(guard, FAIL) is B12's own run-to-run noise. Like-for-like: B12 guard-off vs ref (62 rows missing: 2 IFEval + 60 HumanEval+),
or both sides with R1. Eval suite's choice.
16:02 B12 run-to-run noise with the suite's scorers (PLAN_B.md, scripts/b12_runnoise.py): math losses vs ref all in guard-touchable
rows (40 untouchable rows: same pass/fail as ref on every row; no-guard 88 vs 86); IFEval: two B12 runs disagree on 11 of 98 rows
(as often as with ref), deficit ~2-3 points on untouchable rows; truncations 14 vs 12 without the guard (+1 over the line).
B12 = borderline, not a clear FAIL; history-dependent variants need two runs or a per-row pool reset per verdict.
ADOPTED (BigMoE RG1, ~18:40, ea6e139): R1 is the main-track default. On the 14 rows the replay flagged, R0 armed on 7 of 10 math
rows and changed answers from token 556-903; R1 armed on 0 of 10; IFEval tokens identical R0 vs R1. BigMoE also found the decode
loops never passed the guard a tokenizer, so tool-call spans had not been masked at all. CAVEAT for the replay's "rows without a
hit are unchanged": it assumes a reproducible engine; BigMoE found long v3 outputs differ run to run past ~1K tokens even with
the guard never armed (b17 / b19 bisect it). Treat exactness as verified to 256 tokens until that lands.

## A. Main track: 80 tok/s EXACT at 24 GB: gap now 1.75-2.5x (v3, 22:13)
OPEN, READY TO RUN (15:34, TrainLab HANDOVER 'READY TO RUN'): the per-pass expert budget (non-exact opt-in) on v3 at 24 GB, MTP 3:3:0, JIT
on (turns/turn14.sh, ~25 min lock; TL_BUDGET=b18 / b16). A trace LRU replay predicts misses per 4-row pass at C=128 fall to 0.44-0.56x
of natural at b18 (0.17-0.24x at 16-row passes with b32), at KLD near the strict targets (b18: 0.0019 / 0.0050 / 0.0245 / 0.0126).
Unlike deferral this cuts READS, so it should add to JIT prefetch. Reading: kill < ~5 % tok/s; adopt-candidate >= ~10 % on both
prompts (Kamen's call). The most promising main-track lever left for the main account's week.
MEASURED (16:48, TrainLab turn14, lock 16:34-16:45, REPORT.md last section, data/budget24.txt; v3 24 GB, JIT on, MTP 3:3:0, 256 tokens,
one run per arm, 2 prompts): b16 +31.9 % code (34.38 -> 45.35 tok/s), +22.4 % long 6K ctx (24.98 -> 30.58) = ADOPT-CANDIDATE by the
pre-set line; b18 +23.5 % code, -12.7 % long (one anomalous run: diverged at token 5, more loader io despite 31 % fewer misses).
Decomposition (b16): ms/pass 100.6 -> 79.5 (1.27x) code, 101.5 -> 87.2 (1.16x) long; tok/pass +4-5.5 % (different text, treat as
noise). Robust part = per-pass time, +16-27 %. Misses/pass 0.56x / 0.68x (b16) vs the replay's 0.36-0.46x: the teacher-forced,
JIT-free replay was ~1.5x optimistic on the miss cut (wrong drafts widen the union; JIT already covered some misses). JIT still
issues ~240 reads/pass sized for natural routing; only 50-64 used under the budget (104 natural): a budget-aware prefetch
(predict the pass set) is the next lever on top. QUALITY: the KLD screen (exact 4-bit engine, not v3) puts b16 at 1.56x (gsm) and
1.78x (ifeval) over the strict targets, b18 at 1.11x / 1.20x; code and multi inside. Stacked on v3's own divergence (the
"combined check" Kamen set for fast modes) it is likely further over (not measured). So NOT a passing fast mode yet. Path (main account's week): (1) KLD
screen of b20 / b22 combined with v3 tiers, pick the largest budget that passes; (2) tok/s at that budget on >= 6 prompts (natural +
budget once each: prompts matter more than repeats, natural reproduces to 0.04 %); (3) budget-aware JIT to win back speed; (4)
decode sets + Chat1000's agentic row; (5) R=16 KLD for the 350 table. Tag: published (XShare ICML'26, OEA 2511.02237), first
measured here on SSD-streamed 24 GB: +22-32 % at a quality cost over the strict gate.
PRE-REGISTERED (17:04, before TrainLab's turn15 runs; additive approximation: v3 control + the budget's increment over the 4-row
floor, budgets 20-28 extrapolated geometrically from the 16 -> 18 step): v3's own tiers already use 52 / 66 / 86 / 48 % of the
code / gsm / ifeval / multi targets, leaving ifeval headroom 0.0028 and gsm 0.0015. Predicted combined / target, worst part:
b16 2.45 (ifeval), b18 1.87, b20 1.48, b22 1.21, b24 1.03, b26 0.95 (distinct-expert cut ~5 %). So under the STRICT combined
check no budget with a useful cut passes: the first pass is ~b26, worth a few %. Kill line for this prediction: if turn15
shows b20 or b22 passing every part, the additive model is wrong and the budget lives under the strict bar. Strategic reading:
v3 consumed most of the strict gate's ifeval headroom, so EVERY further non-exact main-track lever (budget, deferral, ...) hits
the same wall. Its future is Kamen's choice of bar for opt-in fast modes: strict per-part KLD (then dead) or task results, as
for the 12 GB line (then b16-b18 likely pass: their KLD is 3-7x below frugal-10 %, itself borderline-pass on tasks).
PRIOR ART for the budget, re-checked 19:44: MoE-Spec (2602.16052) = training-free per-layer expert cap in verification, +10-30 %
over EAGLE-3 (our b16 / b18 is this lever). AcceptMoE (2608.02989) = self-sizing verifier sets: utility = commitment-weighted
router mass (position-depth prior), set size from the demand's effective rank, ROW 0's natural top-k always kept (anchor),
residency-aware pruning under offloading; -0.27 pt mean task accuracy, 1.29x in RAM, 2.06x under expert offloading (RTX 5090).
EcoSpec (2607.12696) = lossless draft-tree choice; EVICT (2605.00342) adaptive verification. => next budget version = AcceptMoE's
selector (published, untested here): the row-0 anchor alone makes every pass's first committed token exactly routed.
NEW BET (19:44, named): UNCERTAINTY-GATED BUDGET (UGB) = run the budget only on passes where the model / drafter is confident,
natural routing elsewhere (code ~4 % uncertain positions, IFEval ~35 %). Prior art checked: MoE-Spec, AcceptMoE (no
uncertainty fallback), EcoSpec, EVICT, CARE (2607.26052) / DA-MoE (2409.06669) (k by ROUTER confidence, not a verify-budget
fallback): not found. Aim: pass the STRICT combined check while keeping most of the code gain. Kill test (CPU, pre-registered in
scripts/ugb_gate.py): on turn15's per-position arrays (v3 tiers), an oracle gate on the exact arm's own confidence must bring all
four parts under the strict targets while the budget covers >= 80 % of code positions; else KILL. Then the realistic gate (MTP
drafter confidence, known before the pass) on the GPU.
RESULT (21:01; TrainLab turn15 = the strict combined screen on v3: NO plain budget passes (b22 -21 % distinct experts fails on
code mean x1.10 [..1.52] / P99.9 x1.80, ifeval top-1; b20 x1.16-1.28; b18 x1.41-1.68): my pre-registered prediction HELD).
UGB with the suite's OWN rules (scripts/ugb_rule.py, data/ugb_rule*.txt; per 4-row pass, gate = min top-1 log-prob of the pass):
ORACLE gate (exact arm's confidence) and POST gate (budget arm's own) PASS every part for b20 at top-1 >= ~0.82 (budget on 90 %
of code positions, 70 % gsm, 13-14 % ifeval, 19-20 % multi) and b22 at >= ~0.95 (code 85 %); b18 passes only on the oracle.
The cheapest PRE-pass gate, the previous pass's confidence (LAG), FAILS everywhere: code's large-KLD positions are surprises inside
confident stretches. Caveat: P99.9 on ~1.4K code positions is the worst 1-2 positions (b22 failing where b20 passes is that noise).
Realizable forms: (a) POST + truncation (commit rows before the first unsure row, run the next pass exactly): no rerun, est. ~+8-9 %
on code; (b) a pre-pass gate on the MTP drafter's own confidence for the current drafts: ~coverage x budget gain = ~+12-14 % on
code if it tracks the oracle. Prose stays ~exact either way. DECISIVE NEXT TEST: a tf screen that also records the MTP head's top-1
probability per position (TrainLab, after turn16). Framing for Kamen: strict bar -> ~+8-14 % on code only (UGB); task-result bar
-> b16 +22-32 % on everything.
VERDICT (06:45, MoonShots stall/idle.py, per-layer gate traces, v3 at 24 GB): the SSD-idle time is real but fragmented into 48
windows of 1-4 reads; an IDEAL look-ahead gains only +13-19 % (code 86.5 -> 74 ms, prose 53 -> 45.5, edit 161 -> 136), saturating
at top-3/4 (precision 0.90-0.95); realistic ~5-10 % = a retune of the existing JIT prefetch, not a build. With compute near SoL,
2-bit tails failing the gate and residency weak, EXACT decode at 24 GB is near its floor-minus-overlap: 80 tok/s exact is NOT
reachable with this hardware and format. The remaining big lever is overlap inside a token = non-exact deferral (lower ranks
half a layer late) or a model change; its per-part quality screen at C=128 is next (MoonShots, tf_defer125 dh_w4 / dh_w2 /
d1_w4); adoption is Kamen's call.
RESULT (07:16, MoonShots, tf_defer125 at C=128 with exact 4-bit loads, every 8, 3,216 positions): dh_w4 per-part KLD gsm 0.0015 /
code 0.0006 / ifeval 0.0087 / multi 0.0041 (exact streaming control 0.0009 / 0.0006 / 0.0089 / 0.0019; targets 0.0045 / 0.0025 /
0.0204 / 0.0206): 2.3-5x under every target, top-1 0.96-1.00 (dh_w2 and d1_w4 similar). Speed (trace replay, 72 % of reads late,
before engine overheads): code 86.5 -> 69.7 ms (1.24x), prose 53.0 -> 43.5 (1.22x), edit 161 -> 126 (1.28x); the pass becomes
SSD-bytes-bound. Gating check before Kamen decides: v3's 3-bit tiers + dh_w4 combined (code is tight: 0.00172 + ~0.0006 vs
0.00253). Build piece: a multi-row phase B for MTP verify (Breakthrough's late-rank-1 on v3 lost its gain to overheads).
COMBINED CHECK PASSED (07:57, Breakthrough tf_b12.py on the real v3f decoder, plan 4433333333, C=128, 3,216 positions): v3 control gsm
0.00298 / code 0.00129 / ifeval 0.01763 / multi 0.00989; v3 + dh_w4 0.00334 / 0.00101 / 0.01807 / 0.01108 vs targets 0.00449 /
0.00253 / 0.0204 / 0.0206: all four pass on the point estimate; IFEval is tight (89 % of target; bootstrap 95 % upper bound
0.02071 just crosses; the control itself sits at 86 %). Per Kamen (07:40) the opt-in fast mode may now be built, below the main
items; a denser IFEval re-check (every 4) goes with the build.
IFEval tightness = SUBSET ARTIFACT (10:21, MoonShots gate_cluster.py on Breakthrough's npz): the gating run's IFEval was only 4
sequences; paired dh_w4 - control by sequence +0.00045 [-0.00018, +0.00071]; on all 30 sequences v3 sits at 0.01367 (67 % of
target), so v3 + dh_w4 ~0.014 (~70 %). MoonShots' turn 1 (re-queued 10:20, ~12:15) scores all 30 IFEval sequences, both arms.
ALL 30 IFEval sequences (11:21, MoonShots, 6,983 positions, real v3f decoder, step scorer): dh_w4 0.01400 (69 % of 0.0204) vs v3
control 0.01303 (64 %); paired +0.00097 [+0.00065, +0.00127] by position, [+0.00060, +0.00142] by sequence; top-1 0.950 -> 0.947;
upper bound 0.0143-0.0163. IFEval is not the tight part. Bit checks, overhead and decode A/B report ~12:10.
CLOSED (11:38, MoonShots): no speed on deployed v3 at 24 GB: with JIT prefetch on, v3 and dh both run 34.9 tok/s on code
(without JIT: 28.1 -> 34.6): deferral substitutes for the JIT prefetcher in the same inter-layer windows. Not shipped. The
morning's 'strictly serial / idle windows' anatomy excluded prefetch reads, so the exact track has more overlap than stated.
REFINED (08:21, MoonShots defer/sim2.py): adding v3f's loader reading late experts as a second SSD batch, ~0.12 ms of GPU tail per
layer and phase-B overhead, the fast mode is +12-17 % (code 86.5 -> 77.6 ms, 1.12x; edit 1.16-1.18x; prose 1.10-1.14x), or
1.14-1.22x if the loader overlaps the late batch's latency with the waited one. Build written (~/MoonShots/defer: pinned v3f
copy, BIGMOE_DEFER_FULL ext change for S=8, multi-row dh path behind BIGMOE_DH=W, bit-identity checks DH=10 == v3); turn 1 =
denser IFEval re-check + bit checks + all-hits overhead + streamed A/B.
STALL ANATOMY first cut (05:46, MoonShots ~/MoonShots/stall/REPORT.md; b13 env + GDN_EXACT, streamed spec125, 160 tokens):
reads and compute are STRICTLY SERIAL in every mode (exposed stall = demand I/O; v3's OoO ~0 overlap: it hides only the hit
experts' ~0.3 ms/layer behind ~0.8-1.2 ms of reads per stalled layer); the SSD runs at 13.6-14.9 GB/s while reading.
| mode | prompt | tok/s | pass ms | compute | stall | MB/pass | floor |
|---|---|---|---|---|---|---|---|
| v3 24 GB | code | 34.5 | 85.1 | 35.4 | 49.6 | 637 | 51.4 |
| v3 24 GB | prose | 29.7 | 54.0 | 28.6 | 25.4 | 280 | 30.9 |
| v3 24 GB | edit | 42.2 | 160.7 | 56.5 | 104.2 | 1534 | 113.6 |
| B12 10 % | code | 31.1 | 84.0 | 36.9 | 47.1 | 459 | 40.3 |
| B12 10 % | prose | 22.6 | 89.0 | 37.0 | 52.0 | 521 | 42.3 |
Levers: (1) bytes per miss: records4 2-bit tails (~0.55x bytes) work with ZERO overlap: v3 code 85 -> ~60 ms (1.4x), prose
54 -> 41, edit 161 -> 113 — FAILED THE GATE (06:14): 4422 / 4322 / 4222 are 2.4-3.9x over every per-part target (DEAD.md);
only last-rank 2-bit might pass (~5 % of bytes);
(2) overlap needs reads before routing: prediction at high precision (waste > ~20 % eats the SSD idle time, which is
what sank I8 on B12: +580 MB of predicted reads per pass at ~40 % waste); with overlap + 0.55x bytes code -> ~38 ms (2.25x);
(3) residency (B12: ~50 % misses on the waited ranks, 188 waited reads per code pass).
Residency sweep (06:00, TrainLab): C_m 51 -> 56 (the most that fits 12 GB at 2.6K) +7 % on the long prompt, waited reads -6 %;
66 cuts waited reads only -15 %: residency is a small lever at this size.
AGENTIC LATENCY (05:46, Chat1000 replay of 21,791 real calls): per call median 11.2 s / p90 28.3 / mean 15.6 at 24 GB exact,
prefill 37 % (exact short-prompt prefill ~7.1 s at 327 tokens; LEDGER PF1's 'never landed' 7-10 s -> ~2.8 s); B12 is quicker
(9.8 / 28.0 / 14.6 s). Porting D1i's bitwise parts (grouped sub-batches, read-ahead ring, copy-free rebuild, streaming >= 1K)
to BigMoE's exact prefill = TrainLab's next job.
RESULT (06:07, TrainLab, copy of BigMoE 86e548c, v3 tier, C=128; shas identical to unpatched incl. 2.9K / 5.7K): grouped tier
prefill (grouping only where the reference takes the sorted gather, rows padded to 64) + logit-free chunks cut time to first
token: 54 tok 3.32 -> 3.02 s, 327 tok 6.23 -> 4.73 (-24 %), 346 tok 6.42 -> 4.87, 2.9K 19.3 -> 12.7 (-34 %), 5.7K 27.6 -> 21.2;
footprint at 5.7K 27.0 -> 25.5 GB. Union reads are SSD-bound (code: 3.3 of 4.7 s reading ~39 GB). Next exact cut: LSB-only
reads for MSB-resident experts (~29 %): code ~3.8 s. Minimal patch to BigMoE today.
ADOPTED (08:30, BigMoE LEDGER PFG1): the tier-prefill patch is in the v3 main track; BigMoE's checks: v3b shas identical at 96 /
256 tokens on code / edit / prose, spec == ar2, GSM8K 0-39 40/40 identical; code prefill 5.95 -> 4.73 s in one window.
Skip-residents v3 exact but +1.0 s (fresh per-group buffers serialized reads with compute); v4 (preallocated scratch, lazy
scatter) queued with the same kill line.
PARKED (09:13): skip-residents v4 exact but -0.35 s at 327 tokens (kill line 0.5) and +1.2 GB on the 24 GB track (code 22.5 ->
23.7 GB); edit -0.45 s, 5.7K -1.3 s; bytes -26 %. Fits only a setting with memory headroom or an in-place ext scatter.
STALL ANATOMY part 1 (05:01, MoonShots ~/MoonShots/data/stall/; PRE-v3 exact path, C=128, SpecDecode's adaptive MTP loop):
true all-hit compute 29.9 / 41.2 / 57.2 / 66.0 ms per pass at S = 2 / 4 / 6 / 8. Prose 22.0 tok/s: 79.2 ms per pass = 33.5
compute + 45.7 exposed stall, demand I/O 45.0 ms -> ZERO overlap; 207 misses = 609 MB per pass at 13.5 GB/s; floor
max(33.5, 43.5) = 50.6 ms (bytes-bound); perfect overlap alone ~52 ms (1.5x). Edit 30.2 tok/s (6.96 tok/pass, S 7.1): 1,280
misses = 3.75 GB per pass at 13.6 GB/s: SSD-bandwidth-bound outright (bytes per miss, residency). The shipping v3 tier
mode (OoO only +1-6 %) is in part 2.
UPDATE 22:13 (BigMoE b13, M24g, AW1, exact v3 = 3-bit tiers + 3c + combine + OoO, whole process 23.7-25.3 GB): code 26.8
-> 36.1 tok/s, edit 32.7 -> 45.7, prose 23.9 -> 32.3 (+35-40 %); OoO +1-6 % bitwise. Gap to 80: 2.2x code, 1.75x edit, 2.5x
prose. Next exact levers (BigMoE): GPU-side tier rebuild, loader/compute overlap with MoEPrefetch; then the 2-bit-heavy
tiers (quality-gated) for bytes. The table below is from the morning (28.7-29.4 code).
Must be true: ~3.1 tokens per pass -> a pass of <= 39 ms (today ~110 ms = ~70 loader stalls + ~40 compute).
| factor | multiplier | why it is so | who / mechanism |
|---|---|---|---|
| A1 loader and compute run in SERIES (110 = 70 + 40, not max) | 1.57x | layer L's experts are known only after layer L's router; prediction covers 67-73 % (recall@10), the rest stall | MoEPrefetch / BigMoE: OOO (+3-8 %), look-ahead prefetch, loader levers; exact overlap caps ~1.3x |
| A2 SSD efficiency: bursts 6-7.6 GB/s (11 incl. prefetch) vs 14.6 at QD32 | 1.33x | per-layer bursts of 1-8 reads are latency-bound (0.34 ms + 0.083 ms/MB) | deeper queues need reads issued earlier (prediction) |
| A3 SSD bytes: 765 MB/pass; 3-bit tiers -> 543, 2-bit-heavy -> ~420 | 1.3-1.8x | Q4 experts at 25 % residency; Belady would cut 37-49 % but needs 8-16 passes of foresight | 3-bit tiers adopted; 2-bit-heavy plans gated on GSM8K (MoonShots records4) |
| A4 compute 40 ms vs ~23 ms byte floor | 1.74x | ~40 dependent small kernels per layer on the critical chain | MoonShots / BigMoE chain fusion |
| A5 tokens per pass 3.1 (MTP caps ~3.5-3.9) | 1.1-1.25x | drafter acceptance | SpecDecode |
**Biggest: A1 (1.57x).** Exactness forbids the clean fix (I1 deferral is not bit-exact): the exact ceiling is prediction-
limited. Non-exact option for Kamen: I1 turns A1 and A2 into max(SSD, compute) with no prediction.

## A2. (DROPPED by Kamen 21:50: the main track is the 24 GB setting; the 40+ GB runs were tests) 80 tok/s exact at ~46-48 GB
Today ~50 tok/s exact on code with AW1 (BigMoE 0.50: 40.5-42.1 plain spec; edit 42.7-46.7). Must be true: ~3.2 tokens per
pass in <= ~40 ms. Today's pass at 0.50 is ~62-67 ms = compute ~46.6 (MoonShots' 34 GB breakdown; compute is ~the same at
any residency) + loader stalls ~15-20 (fewer misses than at 24 GB: ~1 cold expert per layer per pass) + ~1-3 between
passes (pipelined under ASYNC_EVERY=1).
| factor | multiplier | what closes it (exact) | owner |
|---|---|---|---|
| A2-1 compute ~46.6 ms vs ~30 ms byte floor | 1.5x | 3-bit tier engine + 3c fused GLU (-22..-28 % per MoE layer part, in progress b5-b10), exact GDN prework (-1.2 ms/pass), fused HC block (-5.8 ms/pass est.), the exact pieces of MoonShots' speed-of-light kernels (bandwidth-shaped GEMVs) | BigMoE + MoonShots |
| A2-2 loader stalls ~15-20 ms in series | 1.3-1.4x | OOO experts (+3-8 %), look-ahead prefetch with MoEPrefetch's estimator (precise: waste costs bandwidth), per-expert release; at 50 % residency only ~1 miss per layer per pass, so near-full overlap is plausible | MoEPrefetch + BigMoE |
| A2-3 tokens per pass | 1.1-1.2x | AW1 adaptive width + lookup on edits/code (SpecDecode) | SpecDecode |
Estimate: compute 46.6 -> ~34 ms (3c + fusions) and stalls -> ~5 ms gives ~39-40 ms per pass = ~80 tok/s at 3.2 tokens/pass:
**A2 closes with work already in flight**; the 3-bit tier engine's validation (v3, GSM8K 450/472) is the gating item, then
OOO + estimator prefetch, then the fused exact pieces. Biggest single factor: compute (1.5x).

## B. Small RAM: 100-150 tok/s at 6-12 GB, Q4-like (today at Q4-like KLD: ~13-16 tok/s eager at 10 %): gap 7-10x
Must be true at 100 tok/s: <= 10 ms per token; SSD <= 146 MB/token at 14.6 GB/s (150 tok/s: <= 97 MB) with reads fully
hidden; compute per token <= 10 ms (3.2 tokens per <= 32 ms pass).
Today (Breakthrough 10-21, 10 %, eager): exact 12.9 tok/s at 422 MB/token (77.5 ms = ~30 compute + ~47 stalls, while
422 MB at 14.6 GB/s is 29 ms); wait-to-85 % mass (KLD 0.0094 = ar4) 106.6 reads = 295 MB/token; frugal (KLD 0.057,
preliminary decode rows show no task loss) 25.3 tok/s at 82 MB/token, 39.8 with MTP.
| factor | multiplier | why it is so | mechanism |
|---|---|---|---|
| B1 SSD reads stall the token (latency-bound, in series with compute) | 2.5-3x | a miss must be resident AT its layer; reads issued per layer, 1-8 at a time | **I1 deferred experts (NewIdeas, testing)**: a miss joins the token d layers later, reads overlap compute with no prediction |
| B2 SSD bytes at Q4-like coverage: 200-300 MB/token vs <= 97-146 | 2-3x | flat routing (90 % of mass needs 8.8 ranks), orthogonal experts (no substitutes), cold tail (Belady recovers 10 %), MTP rows don't share cold misses, experts incompressible beyond bits (I2/I3 dead) | bits (3-bit tiers, 2-bit tails on the RD curve); otherwise only a quality-bar decision (frugal) or a trained model (fewer experts) |
| B3 engine compute per pass ~80 ms vs ~28 ms roofline | 2.9x | kernels at ~45 % of the read ceiling, 48 gate waits, Python graph build | MoonShots / BigMoE kernels; I4 (MoE off the chain) |
| B4 tokens per pass 3.2 | (does not cut SSD bytes: cold misses don't amortize) | | |
**Biggest: B1 (stalls, 2.5-3x), then B2 (bytes, 2-3x) and B3 (engine, 2.9x) almost tied.** If the Eval suite passes
frugal, B2 disappears (82 MB/token) and B3 becomes the biggest. With I1 at Q4-like coverage: max(SSD 20 ms, compute)
per token -> ~33-40 tok/s on today's engine, ~50-75 with 3-bit tiers and engine work: B3 then decides.
Why chain for B2: why 295 MB? ~106 cold reads x 2.77 MB. Why 106? Q4-like KLD needs ~85 % of the top-10 mass and 1/3 of
those uses miss a 10 % cache. Why can't fewer experts do? Experts are orthogonal and incompressible (I2, I3), so every
uncomputed expert's energy is lost (KLD ~ E^1.5). Unquestioned assumption: the quality bar = per-token KLD near ar4.
Frugal's decode rows (math 90 vs 86, IFEval 88 vs 90) say the bar may be far looser than KLD suggests: the verdict is the
single most valuable measurement for this target.

### B update 15:20: the official gate and the stall-free simulation decide B
- **Eval suite verdict (15:10):** frugal 10 % FAILS 5/8 rules, 5 % FAILS 4/8. The gate is per corpus part: 90 % CI upper
  bound of KLD / ar4 <= 1.10 (ar4's KLD: code 0.0023, gsm 0.0041, ifeval 0.0186, multi 0.0187), P99.9 <= 1.10 x ar4,
  top-1 within -0.5 pt. Easy text binds (code needs <= ~0.0025). On Breakthrough's subsets only wait-to-85 % (m85_nr)
  passes every part (~107-112 reads/token at 10 %).
- **Stall-free coverage on real 125B traces** (scripts/sfc_sim.py, data/sfc/sfc_sim.txt; MoEPrefetch's look-ahead
  predictions, 1,600 decode tokens, C=51): exact coverage 153 reads -> 12.8 tok/s eager; + look-ahead prefetch top-10 +
  half-late misses: waited reads 153 -> 6, but 237 reads/token (76 wasted) -> bandwidth-bound 22 tok/s (26 with MSB-3
  bytes). Stalls can be engineered away; bytes then bind. (Half-late itself costs KLD that the code/gsm budget may not
  allow: 125B rows on Breakthrough's turn on both subsets.)
- **Near-tie re-routing scales with residency (15:35, 125B expA traces, top-10 coverage, ranks 0-1 pinned):** reads/token
  delta 0 / 0.25 / 0.5: C=51 155.7 / 110.8 / 89.2; C=77 118.0 / 77.9 / 59.1; C=97 (~19 %) 97.4 / 60.7 / 44.6; C=128 73.5 /
  42.9 / 30.4. With a 3-bit cache + the dense diet (~19 % at 12 GB) and delta 0.25: ~130 MB/token = ~112 tok/s SSD ceiling
  under the formal gate IF b25 passes it per part (Breakthrough runs d1_w10_b25 first). At 24 GB (non-exact mode) delta 0.25
  is -42 % reads, the same cut the 80 tok/s plan expected from 2-bit tiers.
  CORRECTION 16:05 (Breakthrough caught it): the '0.7 % / 1.9 % / 4.2 % swapped mass' figures were the NET change in kept router weight; a swap costs the displaced AND the added expert (orthogonal outputs). Displaced true top-10 mass per layer is 6.7 % (delta 0.25) / 10.8 % (0.5) at C=51, 6.0 % at C=97; swap gate energy 0.0081 / 0.0126 / 0.0073 -> the energy law predicts KLD ~0.02 at delta 0.25 (Breakthrough's per-text fit: hard 0.019, easy 0.0055: gsm FAILS, code borderline), ~2.2x the energy per saved read of the mass rule. Re-routing is probably dominated by wait-by-mass under the gate; the 125B probe d1_w10_b25 (first in Breakthrough's queue) and the 35B frontier decide.
- **Breakthrough's corrections (15:45), accepted:** (1) RAM: measured floors (MLX active 4.2 GB at C=16; process RSS ~7.5
  GB) put a real 12 GB at C ~44-75 with an MSB-3 cache (8.6-14.6 %), not 19 %; (2) the MSB-3 cache already spends 62-68 % of
  the easy-text gate (plan4433333333: code 0.0017 / gsm 0.0028 vs 0.0025 / 0.0045), so b25 must fit in ~0.0008 code /
  0.0017 gsm; (3) compute binds first: today's engine has a ~28.4 ms/token zero-read floor (eager <= ~35 tok/s; bytes
  roofline 17 ms = ~59 eager; 4-row MTP roofline ~107 tok/s code / ~78 prose; measured frugal+MTP 39.2). Realistic at the
  formal gate with b25 at C ~60-75 on today's engine: ~19-22 tok/s eager, ~28-30 with MTP on code. **Under any bar, once
  reads are cut and hidden, the ENGINE (compute per pass, ~2-2.7x above its byte roofline) is the binding factor** -
  MoonShots' speed-of-light layer decides how much of it a rebuild can take.
- Intra-pass sharing bias (rows of a verify pass prefer experts other rows chose): only -1..-4 % reads beyond the resident
  bias (data/sfc/share_sim.txt): near-ties of consecutive tokens rarely coincide. Dead.
- **So at the formal gate, B is a bandwidth problem:** Q4-like coverage = ~110 cold reads/token at 10 % residency =
  305 MB (4-bit) / 245 MB (MSB-3 tiers) -> <= 48 / 60 tok/s at 14.6 GB/s with PERFECT latency hiding and no prefetch
  waste, at ~14 GB. Inference-only levers left: near-tie re-route (-29 % reads, gate pending), 2-bit tails (4433332222
  suite pending), the dense diet (residency 10 -> 15 % at 12 GB), prefetch precision. Hardware: SSD bandwidth (RAID-0 with
  Thunderbolt-5 NVMe; once stall-free, tok/s scales with GB/s). Model: a trained variant (fewer active experts, locality,
> (a quote from the project owner, kept private)
  drastically different answers") vs the formal per-part KLD gate. Frugal's decode rows (math 90 vs 86, IFEval -2.0,
  HumanEval -1.7 pass, answer agreement 91-92 % vs the floor's 97-98 %) would pass the words and fail the gate. Under the
  words, ~100 tok/s at ~14 GB is an engineering path (frugal + prefetch + 3-bit/3c + kernels); under the gate, it needs a
  trained model or 2x more SSD bandwidth on top of everything else.

### B update 01:30 (task-result bar; B12 = frugal 10 % on v3, HC8; true footprints)
Today: B12 eager 24.2 tok/s at 11.6 GB (fits 12 GB); B12 + MTP 38.6 code / 26.4 prose at 13.1 GB (trim to < 12 GB running).
Factor tree for 100 tok/s, fast-engine world (model, 4-row MTP pass ~52 ms = ~33 ms compute + ~19 ms waits):
| factor | today -> needed | mechanism | est. | owner |
|---|---|---|---|---|
| engine compute per pass | all-hit pass ~38 ms already (BigMoE, near SoL) | engine BIG BUILD STOPPED 04:36: compute prize only ~10-20 % | small | - |
| waits for cold ranks 0-1 | ~30 % of a fast pass in the MODEL (unverified) | I8 FALSIFIED on the eager engine (03:53: per-pass time flat despite 125 on-time prefetches/pass); profile wait vs compute in the new engine before any prefetch work | ? | MoonShots M2 |
| tokens per pass (prose 2.2, code 3.1) | -> 3-4 | 125B MTP head retrain KILLED 03:22 in its first form (tau -4.6 %: depth-training context != inference); a TTT-inside-QSA recipe on the 125B's own text needs multiple GPU nights (proposal); span-aware prose acceptance KILLED (+10.5-11 %) | open | Kamen's call |
| RAM with MTP | 13.1 -> < 12 GB | MLX cache cap, prefill chunks, sidecar, C 51 -> 46 | fit | Breakthrough |
Product of the estimates: code ~38.6 x ~1.5 x 1.17 x 1.2 ~ 80 tok/s; 100+ needs the engine near its byte floor (~2x).
SUPERSEDED 04:37: the engine factor (1.5x), I8 (+17 %) and the head retrain (+20 %) all failed tonight. B12 + MTP code
~37 tok/s = ~84 ms per pass, of which ~38 ms compute (near SoL) and ~46 ms stalls. Levers now: residency (D1i freed
~1.6 GB under 12 GB -> C 51 -> ~66), bytes per waited read (MSB-3-only ranks 0-1, quality?), overlap of hits with
misses (OOO), waiting for rank 0 only (quality). Ceiling if stalls vanished: ~3.12 / 0.038 ~ 80 tok/s on code. The
stall anatomy (MoonShots) decides the order.
**B5 MEASURED (03:53, Breakthrough, B12 eager, 2,617-token prompt, cache-only chunks):** chunk 2048 175 tok/s at 13.2 GB (with
the unused logits 173 / 13.9 GB), 512 81 tok/s / 11.8 GB, 256 51 tok/s / 11.4 GB: ~3x below the SSD-bound estimate (reads
~3.5 s of ~7.5 s per 2048-chunk; the CPU rebuild of 4-bit from MSB+LSB and serial read/rebuild/compute are the likely
rest). Prefill is then ~35-40 % of a mean agentic call on the M5 Pro. 32K context: B12 eager 18.7 GB, + MTP 19.1 GB peak
(prefill attention over ~37K keys per 2048-query chunk + expert scratch + ~0.9 GB KV), prefill 189 tok/s: B12 is
'12 GB for prompts of a few K tokens' today. D1i (TrainLab) now targets >= 400 tok/s at <= ~12 GB: double-buffered
reads / GPU rebuild / compute, attention query chunk decoupled from the expert chunk. MTP's +1.4 GB is the 4-row VERIFY
(head without drafting = eager footprint; hot cache irrelevant). RESOLVED 04:04 (Breakthrough's width scan, chunk 256):
decode sits at ~10.0 GB active (~10.4 footprint) at verify width 4 / 2 / 1 (width adds ~0.1 GB); the +1.4-1.5 GB is
spec125's PREFILL with the head on (per-chunk logits + the hidden stream captured for the head + the MTP layer's own
prefill). Fix: a lean head prefill (cache-only chunks except the head's history, head prefill in small chunks, logits for
the last position only; greedy sha unchanged) in Breakthrough's copy. With it and TrainLab's attention chunking, B12 + MTP
approaches its decode level (~10.4 GB) plus the prefill's own need at a given prompt length.
**B5c (05:10): SMALL-CALL prefill is now the big agentic cost.** Breakthrough's crossover rows (same shas both arms): the
fixed-order sweep costs a flat ~5.6 s and wins above ~0.6K tokens (1,237 tokens 5.67 vs 7.01 s; 1,891 5.87 vs 8.11; D1I_STREAM_
MIN = 1024), but the union read of SHORT prompts is slow too: 53 tokens 3.2 s, 326 ~5 s (cold). With Kamen's median call
at 150 new tokens and decode of the median call ~4.4 s, prefill is ~40-50 % of an agentic turn. Questions: are warm pools
reused (skip resident experts)? where do the non-read ~1.9 s go for 53 tokens? Next lever if it stays ~3 s: decode-style
frugal prefill for short tool outputs (cold ranks 0-1 or 0-3 only; ~2-4x fewer reads) with an agentic quality check.
Breakthrough (05:12): the union read never consults the decode pools (F_NOCACHE re-reads residents; only both-part residents,
<= C_l 20/layer, could skip a read: ~15-25 % fewer bytes); per layer it is serial (router sync -> preads -> CPU rebuild ->
MoE). Exact lever first: pre-read layer L+1's PREDICTED union (union_forecast.py): at 150 tokens the corrected d=1 top-10
covers 86 % of the true union for +18 % bytes (same-layer top-10 88 % / +14 %; top-16 94-96 % / +36-39 %); floor = union
bytes ~29 GB ~2.3 s per 150-token call at 12.5 GB/s -> ~3-4 s -> ~2.5 s expected; then skip residents; quality trades last.
FALSIFIED 05:51 (Breakthrough D1k): the union read already runs at the SSD ceiling (177 tokens: 3.83 s = 2.51 s reads at 13 GB/s
+ ~1.3 s rebuild / compute in series; router waits 0.03-0.05 s); look-ahead +40-48 % at 177-326 tokens. Remaining exact cuts:
union reads through the raw-pread + GPU-rebuild (RB2) pipeline (overlap the ~1.3 s: ~2.6 s at 177 tokens), then skip
both-part residents (15-25 %); non-exact: frugal prefill of short outputs (agentic quality check).
Also: B12 + MTP streams too (ring 4 + RB2: 11.8 GB at 341 tok/s); 32K: B12 + MTP 13.6 GB (head-history fix D1I_HEAD_EACH and
the n-gram hot-row LRU suspected for the 1.0 -> 1.8 GB non-MLX growth; day_ctx2 queued).
**D1i streaming (04:35, TrainLab):** fixed-order streaming of all experts (v3: raw records3 preads in a GIL-free thread pool +
exact GPU rebuild) on the long prompt: ring 6 322-352 tok/s at 11.8 GB, ring 8 350-389 at 12.2 GB (no streaming 278-298 at
10.9); same greedy sha, decode unchanged. What is left is not I/O: the per-layer router readback (CPU sync) and sequential
attention launches. 32K: streaming 326 tok/s at 13.7 GB; no streaming 270 at 12.5 GB (the 32K setting). Next: B8-class
footprints with the decoupled prefill (the 6-8 GB question), then GPU-side routing in prefill (no per-layer sync).
**D1i RESULT (04:18, TrainLab, ~/TrainLab/d1i/PATCH.md, patch to Breakthrough):** decoupled prefill (attention in 256-512
query chunks, MoE once per layer over 4,096 tokens, grouped exact sub-batches, logit-free chunks): long prompt (2,617)
10.9 GB at 272-308 tok/s (vs 13.9 GB at 162-172 for chunk 2048) -> below the eager decode footprint; 32K (37,294 tokens)
12.5 GB at 270 tok/s (vs 17.6 GB at 215). MTP: BIGMOE_PF_GROUP=64 removes the MTP layer's prefill scratch: 11.8 GB at
chunk 256 (C_m 51 / C_l 20); decoupled + spec125 240 tok/s at 12.0 GB. Exactness: grouped sub-batches alone = same sha;
the decoupled prefill is not bitwise vs any plain chunking (gather kernel follows rows per group) -> task check before
adoption. Next: fixed-order streaming of all 512 experts per layer through a ~4-buffer ring (est. ~430 tok/s, +0.35 GB).
**D1i interim (04:03, TrainLab, ~/TrainLab/d1i/out/summary.txt):** expert sub-batches (v3's exact _sync_grouped on records3,
G=64, CPU rebuild double-buffered) are bitwise-exact and +25 % prefill (172 -> 206 tok/s) but barely move the peak (13.2
-> 13.1 GB): the peak is ATTENTION-side (~0.7 MB alive per chunk token; attention part 12.0 GB, MoE part 12.5). Prefill is
SSD-bound (~96 GB read at ~12 GB/s for 2,617 tokens: the 569-token tail chunk reads nearly as much as the 2048 chunk); a
GPU rebuild is slower (180). One 4096 chunk: 300-312 tok/s, half the reads, 13.0 GB. Warming FAILED (waited 116 vs 106 over
tokens 0-16: prefill's experts are not decode's) -> parked. Next: attention in 512-query chunks + the MoE once per layer
over the whole prompt (chunk-512 memory at chunk-4096 reads); then a routing-free full sweep (all 512 experts per layer,
~68 GB, ~4.9 s: ~530 tok/s at 2.6K) for long prompts and the forecast union for short ones.
**B5b (03:21): cold cache after prefill.** v3's prefill reads into a scratch, so decode starts cold or stale: on the traces
frugal C=51 from an empty cache waits 47 / 32 / 26 reads per token over tokens 0-16 / 16-32 / 32-64 vs ~18.5 steady
(~890 extra waited reads, ~0.3 s per call, ~4 % of a mean agentic call; Breakthrough's 'long' rows 75-83 waited/token
for 16 tokens incl. sidecar misses). Fix inside D1i: copy each layer's most-used experts of the last prefill chunk into
the pools (~7 GB of copies, ~50 ms).
**B6 (new 02:04): context memory.** B12's footprints were measured at short context. KV per token (config: 12 full-
attention layers x 2 KV heads x head_dim 256 x K,V bf16 + the QSA indexer) = ~25 KB -> +0.4 GB at 16K, +0.8 GB at 32K,
+1.6 GB at 64K, +3.3 GB at 128K; GDN states are fixed (~0.11 GB). Claude Code sessions run tens of K tokens, so B12 eager
(11.6 GB) leaves 12 GB at ~16K, and B12 + MTP needs its trim plus KV room. Levers: 8-bit KV (-50 %, standard; MTPLX's
QSA cache support unknown), shrinking the expert cache as the context grows (costs residency: at 64K C 51 -> ~28, i.e.
B8-class quality), a context cap. Measure: footprint at 8K / 32K / 64K. Warning: BigMoE's LC1 (0.50 setting) measured peak
43.6 / 45.7 / 47.4 GB at 4K / 16K / 32K, i.e. +3.8 GB from 4K to 32K, ~5x the KV alone (prefill scratch, QSA banks): the
real growth must be measured, not assumed. Breakthrough (02:06): 32K rows (B12 eager and + MTP after a ~32K prefill,
peak footprint + prefill tok/s) ride in its A/B slot after trim2. 8-bit KV is NOT available: qwen4_exp builds a bf16
QSACache for the 12 full-attention layers; MTPLX's q8/q4 KV (kv_quant.py) only serves the paged vLLM-Metal cache, which
our decode path never builds, so it would be new code (quantise on write, dequantise in the attention read): build it
only if the 32K rows show the context, not prefill scratch, is what breaks 12 GB.
**B5 (new 01:35): prefill in small RAM = the agentic turn's hidden cost.** v3's exact prefill reads every expert a chunk
uses from the SSD (cache unused). Distinct experts per layer per chunk (125B prefill capture, 30 docs, scripts/
prefill_union.py): 64 tokens 125, 256 221, 1024 331, 1535 356 of 512 -> SSD-bound prefill ~127 tok/s at 256-token chunks,
~585 at 2048 (~3.5 s per 2K-token tool output; Mac mini ~4.5x slower). The chunk size is the lever, and the RAM trim
pushes it the wrong way: keep big chunks, stream each layer's chunk scratch in expert sub-batches (non-bitwise, fine for
B12), skip resident experts. Floor: one expert sweep (~50-68 GB, 3.5-4.7 s) per multi-K-token prompt.
Sized on Kamen's sessions (Chat1000 REPORT.md section 6: 20,561 calls; new input per call median 150 / p75 437 / p90 986 /
p99 5,572 tokens, mean 526; output median 170, mean 337): cost is SUBLINEAR (150 tokens already touch ~184 experts/layer),
so per call (v3 today / cache used): median 1.67 / 1.26 s, mean 2.5 / 2.1 s, p90 3.0 / 2.5 s, p99 10.3 / 8.8 s (~4 s as one
sweep), vs decode ~4.4 s (median) / 8.7 s (mean) at 38.6 tok/s: prefill is ~20-30 % of an agentic call on the M5 Pro and
dominates on the Mac mini (median 5.7 s, p90 11.5 s at 3.2 GB/s). Levers: cache use (-25 %), large chunks + sub-batched
scratch, one layer-major sweep for big calls; floor ~1 s per small call (the experts a call touches).
Over the actual per-call distribution (Chat1000 01:42, 2K chunks, no residency): mean 2.03 s per call; prefill = 19 % of
agentic call time on the M5 Pro (decode at 38.6 tok/s), 51 % on the Mac mini (median 7.6 s, p90 13.5 s).
Cache policy for frugal's waited ranks (belady_frugal.py, 2,926 tokens): LRU 19.8 waited/token at C=51; best online
(LFU, decay 0.99, rank 0-1 uses only) 18.5 (-7 %); Belady OPT 12.3 (-38 %): the headroom is future knowledge (forecasts,
I8), not eviction rules. LFU-0.99 is a cheap +~7 % for the SSD-bound Mac mini.


> (a quote from the project owner, kept private)
## 1,000 is investigated after 350) - plan 07:47
**Measuring conditions.** Visible tokens per second during decode of one chat (batch 1, greedy, prefill excluded) on Kamen's
M5 Pro 64 GB, model in RAM: MSB-3 (v3) pools for all experts (~53 GB) + dense ~3.4 GB, ~60 GB process, the machine otherwise
idle. Exact = verified bit-identical to the v3 reference (preferred); non-exact routes reported separately with the Eval
suite's rules. Two workloads, always reported separately: (a) general text (the Eval suite's code / prose / math prompts),
(b) Kamen's agentic turns (Chat1000's 12 held-out real Claude Code turns). Reference points: the 24 GB and 0.50 settings.
**Arithmetic.** visible tok/s = [1/(1-F)] x S x A / T_pass. Today (stock engine in RAM, native MTP chain, measured acceptance
0.85 / 0.71 / 0.61 / 0.51 cumulative on agentic turns; Chat1000 path350.py): ~65 agentic / ~53 prose. 350 = 5.4x / 6.6x.
MoonShots' clean measurement fixes T_pass: a true all-hit 4-row verify pass is 37 ms (v3), near the speed-of-light layer, so
kernels alone cannot carry it. With today's drafter EVERY other lever stacked gives <= ~130 (agentic ~170 with copy pointers),
not 250-300: the earlier composite assumed passes of 18-24 ms from levers that since failed (HC 4-bit) or need training
(sticky routing). The gap to 350 is ~2-2.7x and only a much better drafter closes it.
| lever | prize (path350) | tag | chance |
|---|---|---|---|
| one-pass block drafter holding ~0.90-0.92/token over ~15 drafts | 2.4x | published ingredients (EAGLE-3 2503.01840, FastMTP 2509.18362, mask-token parallel drafting) but untested on a hybrid GDN/QSA MoE's native MTP layer; ~0.92 one-pass over 15 tokens on novel agentic text is BEYOND published results (~0.86-0.89); our chain-head fine-tune lost 4.6 % | low |
| trained sticky routing (neighbouring tokens reuse experts) | KILLED 11:03 (35B: layer-local training worse than zero-shot caps) | published (ReMoE, Temporal MoE, expert coactivation) | - |
| per-pass union budget (batch-aware routing inside a verify pass, training-free) | 35B real KLD: 16 rows U16 32 (-39 % experts) 0.069, 4 rows U4 16 (-25 %) 0.046, 1.7-2.5x below caps; THE union lever now | published (XShare ICML'26, OEA 2511.02237); untested on the 125B (run approved 11:03); non-exact | moderate |
| grouped expert kernel (each expert read once per wide pass) | ~1.4x | published (MegaBlocks 2211.15841, vLLM fused MoE); untested in MLX/Metal at our shapes | moderate-high |
| cost-aware MoE draft trees (prefer drafts reusing loaded experts) | overlaps sticky | published (EcoSpec 2607.12696, EVICT 2605.00342, MoE-Spec 2602.16052); untested here | moderate |
| dense diet | ~1.1x | published quantization; HC 8-bit done; HC 4-bit failed quality here | moderate |
| copy-by-pointer output (F) | up to 1.33x, agentic only | UNTESTED / NEW for agent tool calls (Chat1000's check, ~35 queries, REPORT.md section 13): closest is Copy-as-Decode (2604.18170, single-document editing, closed-form bounds only, v2 withdrawn; fragile to line-number noise); lossless copy speculation (prompt lookup, CopySpec 2502.08923, SuffixDecoding 2411.04975) still verifies every copied token; needs a fine-tune of the 125B's output format; anchor by content, not line numbers | low-moderate (training) |
All four of the first, plus the diet: agentic 313 (x1.33 with pointers ~416) / prose 402 tok/s; without the drafter <= ~130 / ~102.
**The new bet in flight (Kamen's rule):** the drafter ceiling. First a cheap decisive measurement (Chat1000, ~15 min GPU):
per-token top-1 agreement of a near-copy of the 125B (frugal 10 % routing, i.e. the full dense model with 10 % of its experts)
with the exact 125B, teacher-forced on the 125B's own continuations: Kamen's 12 agentic turns + the tf_gen sets. If even a
near-copy stays below ~0.92 on agentic text, no small drafter will hold 0.92 and 350 is out of reach on agentic turns with
this model (code / math may still reach it); if it is >= ~0.95, the drafter program starts: a block drafter built from the
125B's own MTP layer (mask-token parallel drafting, training-time test inside QSA, the 125B's own continuations as data).
**Drafter ceiling, interim (07:57, Chat1000 ~/Chat1000/data/drafter_ceiling.txt; the Eval suite's tfstep runs of frugal 10 %
= the full dense model + top-2 waits, scored against the 125B's own tf_gen tokens, 34,457 positions):** per-token top-1 /
P(first 15 right) / block tau at K=15: code 0.983 / 0.839 / 13.4; gsm 0.970 / 0.680 / 12.2; ifeval 0.872 / 0.205 / 6.6; multi
0.901 / 0.211 / 8.0; pooled 0.939 / 0.521 / 9.7 (frugal 5 %: code 0.979, ifeval 0.829). Through path350's best pass model as a
one-pass drafter: code 515, gsm 470, ifeval 255, multi 308, pooled 374 tok/s (CORRECTED 08:22 from 558 / 505 / 291 / 312 / 434:
blocks start after rejections, so they are simulated along each sequence; a CEILING: the near-copy costs a full decode per step).
Why prose fails: 87 % of the misses sit where the 125B's own top-1 probability is < 0.7 (35 % of IFEval tokens, 4 % of code):
forks decided by small logit gaps. Reading: 350 is reachable in principle on code / math (the bar is met with room), not on
prose-like text with any approximate drafter; Kamen's agentic turns (queued, ~40 min) decide where his workload falls.
**Drafter ceiling RESULT (08:21, Chat1000 REPORT.md section 12; agentic row still queued):** 350 needs tau16 >= 9.1 tokens per
verify pass (16-token blocks) on path350's best engine. tau16 per class, near-copy 10 % / AutoRound int4 / float-order floor /
best published one-pass drafter: code 13.4 / 14.8 / 15.2 / 8.70 (DFlash 0.8B on Qwen3.5-122B-A10B, HumanEval; MBPP 7.24); math
12.2 / 14.2 / 15.0 / 7.18 (GSM8K; MATH500 7.84); IFEval 6.6 / 10.6 / 13.2 / 5.44 (MT-Bench); multilingual 8.0 / 11.1 / 13.7 / -.
Even AutoRound int4 ('the same model') holds only 0.947 / 0.952 per token on IFEval / multilingual, and a summation-order
change flips 2.3 % of IFEval tokens. Reading: (a) 350 on prose / chat is CLOSED for any drafter (ceiling ~250-290); (b) on
code / math the ceiling clears the bar with room and the best published one-pass drafter (DFlash, on a hybrid MoE like ours)
is ~5 % short on HumanEval, so a drafter program aimed at tau16 >= 9.1 is the road; (c) agentic decides which side Kamen's
workload falls on. No one-pass drafter has published agentic numbers (EAGLE-3 degrades on SWE-bench traces).
**The drafter program, concretely (08:33, Chat1000 ~/Chat1000/DRAFTER.md, last section):** retarget the published DFlash drafter
instead of training from zero: the 35B DFlash drafter (0.4B, 6 layers, width 2048, Apache-2.0, z-lab; already local with an MLX
loop) keeps the 35B's embedding and LM head (the 125B's tokenizer is IDENTICAL to Qwen3.6-35B's: 248,044 tokens, same merges),
and only a new input projection W_c (5 taps x 2560 -> 2048, 26M parameters) is learned. Data: teacher-forced 125B features on
Kamen's sessions (600-1,800 tok/s, 2-5M tokens per night), labels counted only up to the first token where the text leaves the
125B's argmax (that prefix IS the 125B's own continuation). Verify: spec125 already verifies external 16-token drafts with GDN
rollback. Kill test 0 TODAY (~30 min GPU, no training): ridge-fit W_c, offline tau16 per class vs native MTP. Then the 2-week
program (~4 GPU nights + 2-3 day slots) goes to Kamen: kill at week 1 if code tau16 < 4.5 or agentic < 3.5 and flat; gate code
>= 7 keeps 350 open. Expected: agentic ~4-5, code ~6-8 tokens per block (~1.5-2x over native MTP on today's engine).
**Correction (08:37, Chat1000):** the native MTP chain gives tau16 4.00 on the 12 agentic turns with blocks simulated (not 5.57,
which used the stationary chain); K=3 2.89. path350's native-MTP column drops 11-18 %: stock MLX 65 -> 56 tok/s agentic; every
engine lever with native MTP 128 -> 108 (prose 102 -> 91). So 350 is ~2.9x over native MTP. The near-copy / AutoRound / floor
tau16 values were already simulated and stand. Kill test 0's bar: agentic tau16 > 4.00.
**NEW BET, tagged (08:44):** regeneration-free on-policy drafter data: from Kamen's existing session text, count drafter labels only
up to the first token where the text leaves the 125B's greedy path, so every labelled prefix IS the 125B's own continuation
without generating anything. Prior-art check (Chat1000, DRAFTER.md, searched: target-argmax labels + first-mismatch masking;
regeneration-free drafter data; DistillSpec / HASS / EAGLE-3 training-time test): nearest are Spec-AUF (2607.01893, cuts at the
DRAFTER's first predicted failure), GRIFFIN (masks after a target token outside the drafter's top-K), CTC-drafter (2412.00061,
relabels with the base model's greedy decode), DistillSpec / Draft-OPD (2605.29343) on-policy, NVIDIA DSpark (regenerates every
response); the cut at the first divergence of EXISTING text from the target's greedy path was not found -> NEW. It feeds the
DFlash retarget; kill test 0 (stitch only, no training) is first in Chat1000's queued lock.
**Correction (08:45, Chat1000 DRAFTER.md 386a117): drafter and engine levers MULTIPLY.** On today's engine a 16-row verify pulls
U(16) ~66 experts per layer, so best K stays 5-10 and a better drafter alone gives little: tau16 4 / 5 / 6 / 7 / 8 -> 60 / 68 / 76 /
85 / 94 tok/s (1.06-1.68x over native MTP); with all engine levers (grouped kernel + trained m=1 + diet) 150 / 185 / 222 / 259 /
296. 350 on code needs tau16 >= 9.1 AND all levers. Sequencing: drafter kill test 0 (today) -> grouped expert kernel (MoonShots,
un-parked for the 350 road, after the fast mode) -> trained sticky routing feasibility on the 125B (TrainLab). Agentic
expectation with everything: ~150-185 tok/s.
**Sticky routing, feasibility (09:16, TrainLab):** path350's 1.7x assumes cap m=1 (U(16) ~22), which no post-hoc method supports
(ReMoE trains through the whole model and moves overlap 27 -> 35 %; StickyMoE needs pretraining). Zero-shot caps on 125B traces:
m=1 U(16) ~22, KLD 0.07-0.30; m=2 ~32, 0.035-0.16; m=3 ~40, 0.02-0.09; m=5 49-57, 0.006-0.027 (natural 56-72). Realistic prize
~1.3-1.5x (m~2-3 at KLD ~0.05), NON-EXACT. Kill test approved (35B, layer-local router training: KL + reuse + smoothness, no
cross-layer backprop; kill if < ~2x KLD cut vs zero-shot caps at equal U(16), or compounding far above the per-layer prediction).
With 1.3-1.5x instead of 1.7x the all-levers column drops ~15-25 %, so 350 on code needs a drafter above tau16 ~9.1.
**Better than sticky routing, training-free (09:44, TrainLab data/budget125.txt): a per-pass UNION BUDGET.** In a 16-row verify all
router outputs are known before the experts run: take the U experts with the largest summed w^2 across rows, and let each row
use its top-10 within that set. Published as batch-aware routing (XShare ICML'26; OEA 2511.02237, -39 % MoE decode latency on
Qwen3-30B at batch 16); untested here. On the 125B traces at U(16)=32 (natural 57-73, ~1.8-2.2x fewer experts per pass) the
law estimate is KLD 0.016-0.027 code / math and 0.043-0.083 bulgarian / chat, i.e. inside frugal-10 %'s band with NO training,
1.4-1.9x less error than hard caps at equal union; a clairvoyant bound leaves little room for a trained router; soft stickiness is
worse. Non-exact (Kamen's call). The 35B kill test now measures trained routers, caps and the budget with real end-to-end KLD.
**Kill test 0 RESULT (10:06, Chat1000 DRAFTER.md):** ridge W_c (82,679 agentic prompt positions, no training), z-lab's greedy loop
against the 125B's own continuations, tau16: code 5.29 (perfect-stitch ceiling with the 35B's own features 9.45), gsm 3.09
(7.55), agentic 2.11 (2.31) vs native MTP 4.00. Code passes with zero training; agentic FAILS, and not because of the stitch: a
drafter trained on the 35B's outputs does not predict what the 125B writes on agentic turns (the 125B copies 10-29 % from
context vs the 35B's 46-95 %). Decision: CODE / MATH-FIRST drafter program. Step 1 (one GPU night): train W_c alone against the
drafter's block loss on teacher-forced 125B code / math text with first-divergence labels; kill test 1 on cached features: code
tau16 >= 7 and gsm >= 5. Agentic parked (realistic best ~5, ~1.25x native, needs training on the 125B's own agentic outputs).
**Agentic ceiling RESULT (10:17, Chat1000 REPORT 12):** 10 % near-copy vs EXACT on the same runtime (6 turns, 960 positions): 0.886
per token, P(first 15) 0.41, tau16 6.8 (one-pass ceiling with all levers ~260 tok/s); vs the tokens the 125B emitted (12 turns):
0.836, tau16 5.2 (~200). Two exact 125B runtimes disagree on 4.1 % of agentic tokens (near-ties flip with kernels). Agentic misses
are not only forks (only 58 % at p1 < 0.7; the near-copy agrees 0.85 where the 125B is 90-99 % sure): long contexts reward the
dropped experts. => 350 is CLOSED on agentic and prose for any one-pass drafter (agentic best ~200-260 tok/s); reachable only on
code / math (near-copy tau16 12-13) with a strong drafter plus every engine lever. Step 1 of the code drafter (W_c training)
fits one ~60-min day slot (queued).
25 % near-copy (10:21): agentic 0.905 per token vs exact on the same runtime, tau16 7.6 (vs emitted 0.869 / 6.1): 2.5x the resident
experts buys only ~0.8 tau16; one-pass ceiling with all levers ~290 tok/s. The agentic verdict stands.
Ceiling work COMPLETE (10:29, Chat1000 REPORT 12): 25 % near-copy on every 4th tf_gen sequence (7,954 positions), per token / tau16:
code 0.995 / 15.1, gsm 0.978 / 13.1, IFEval 0.918 / 8.5, multilingual 0.939 / 10.4, agentic (vs exact) 0.905 / 7.6. IFEval and
agentic stay under 9.1 even at 2.5x the expert RAM; multilingual crosses for near-copies, not for small drafters.
**EXACT 350 is closed on every class (11:17, Chat1000 REPORT 11 after the sticky kill):** with the grouped kernel, dense diet, a 3 ms
one-pass drafter and 16-row passes: natural (exact) routing 45.1 ms per pass needs tau16 15.8 (of max 16) for 350; code with a
near-copy-quality drafter (13.4) gives ~297 tok/s, with the best published drafter (8.7) ~193. A non-exact per-pass budget shortens
the pass: U16=40 34.4 ms (tau16 for 350: 12.0; 390 / 253 tok/s at 13.4 / 8.7), U16=32 31.1 ms (10.9; 431 / 280), U16=24 27.8 ms
(9.7; 482 / 313). So 350 on code needs BOTH the non-exact budget (U16 <= ~30, Kamen's call, task-result bar) AND a drafter at
tau16 ~10-11, above the best published (8.7). The drafter program still pays in exact mode: code tau16 7-8.7 -> ~155-190 tok/s vs
native MTP's ~90 on the same engine. Step 1 runs ~11:51 regardless.
**Grouped expert kernel, CPU bound (11:44, MoonShots wide/union_sizes.py on the 125B traces):** distinct experts per verify pass at
S = 4 / 8 / 16 / 32 rows: code 26.9 / 43.1 / 67.3 / 101 (0.67 / 0.54 / 0.42 / 0.32 of S x 10), chat 25.8 / 41.3 / 64.7 / 95, math
27.8 / 46.3 / 72.9 / 111, bulgarian 22.8 / 35.3 / 56.1 / 86.5. Upper bound on the expert-byte cut of reading each expert once: 1.5x
at S=4, 2.4x at S=16, 3.1x at S=32; 5x at S=16 with a U=32 budget. The GPU microbench (3c vs sorted gather_qmm vs the bytes
floor, S = 4-32, budgets 16-64) runs ~15:00.
**Kill test 1 FAILED narrowly (12:25):** W_c alone: code tau16 6.83 (bar 7; saturating), gsm 5.20 (passes), agentic 2.06 (below
native MTP 4.00). Stopped per rule (DEAD.md). Code-chat exact speed ~1.5x possible later via LoRA + on-policy data (night run,
main-account handoff, Kamen's call); not a 350 road.
**Per-pass expert budget on the 125B (13:03, TrainLab budget125_tf.py, 4-row decode_verify passes, 4 tf parts, ~8K positions / arm):**
KLD code / gsm / ifeval / multi and distinct experts per pass-layer (natural 27.1): exact 4-row floor 0.0002 / 0.0007 / 0.0040 /
0.0023; budget 18 (17.9, -34 %) 0.0019 / 0.0050 / 0.0245 / 0.0126; budget 16 (-41 %) 0.0021 / 0.0070 / 0.0364 / 0.0177; budget
14 (-48 %) 0.0031 / 0.0096 / 0.0557 / 0.0268; frugal-10 % 0.0147 / 0.0307 / 0.1153 / 0.0924; targets 0.0025 / 0.0045 / 0.0204 /
0.0205. b18 is 4.7-7.7x below frugal-10 % and meets code and multi, misses gsm / ifeval narrowly; the w^2 rule beats OEA at
equal union. Non-exact (a row's routing depends on its pass-mates, i.e. the drafts). NEXT (13:03): the hook in a copy of
Breakthrough's B12 + MTP first (natural fit: already non-exact, stall-bound), real tok/s + waited reads + tf_b12 KLD by ~16:30;
24 GB second.
CORRECTED (13:05, TrainLab waited125.py): the budget cannot cut B12's waits: B12 waits only for each row's ranks 0-1, which always
have the largest w^2 and stay in the pass set (waited set 5.2-6.1 distinct experts per 4-row pass, identical at budgets 14-18);
frugal already skips the low-weight experts the budget removes. A pass-level WAITED budget (Uw 5 / 4: waited reads -13..-35 %)
is just another frugal quality-for-speed knob (+20-50 % hard-text KLD at Uw 4). The budget pays where every selected expert is
read: the 24 GB exact track (misses and bytes ~ distinct experts per pass) and in RAM -> TrainLab tests 24 GB first (non-exact
opt-in, behind the Eval suite's GPU time).
**Grouped expert kernel, MEASURED (13:51, MoonShots wide/MICROBENCH.md, one real layer's v3 pools, S = 4-32):** today's 3c costs
~6.0 us per (row, rank) pair at S >= 8 regardless of distinct experts (16 or 150): it is bound by per-pair dequant + dot, not DRAM
(repeated experts already come from the cache), so reading each expert once buys NOTHING by itself, and expert-major kernels
that keep the per-pair work are 10-20 % slower (MLX sorted gather 1.139 ms vs 3c 0.971 at S=16). The prize is a NEW multi-row
'dequant-once' kernel (unpack each weight once, FMA into every row of the expert): its floor sits at ~270 GB/s; per MoE layer 3c /
floor = 1.4-3.1x at S=16 natural, 3.5-3.7x at U=32. Pass level (all hits, S=16): today ~97 ms -> ~75-80 ms natural (1.2-1.3x),
~63-69 ms with U=32 (1.4-1.5x). Dense half: plain MLX matmuls grow only 2.1-2.2x from 4 to 16-32 rows (today's 5-row chunking
would be 3.5-6.4x). MoonShots builds the kernel (aim: bit-identical to 3c per row, so S=4 exact can use it too); GPU validation
at the first free slot or handover.
**Drafter-training recommendation, re-checked:** last night's "not now" was about the chain head (+10-20 % headroom). For 350
the one-pass block drafter is the deciding lever, so the recommendation becomes: start the drafter program right after the
ceiling test, if the test leaves room; it needs several local GPU nights.

## C. 1,000 tok/s for ONE chat, model in RAM (today ~40-46 tok/s at the 0.50 setting): gap ~22x
Must be true: 1 ms per token. A pass of K rows reads dense ~3.4 GB + the union of experts U(K) x 48 x 2.77 MB, with
U(K) ~ 10 K^0.67 per layer (4: 26.9, 16: 68.6, 32: 102.3). Even at 100 % acceptance (a = K tokens per pass):
K = 16 -> 12.5 GB / 300 GB/s = 42 ms -> 380 tok/s; K = 32 -> 17 GB -> 560; K = 64 -> ~25 GB -> ~770; K = 128 -> ~1,060.
| factor | multiplier | why it is so |
|---|---|---|
| C1 accepted tokens per pass: ~3.2 today vs >= ~128 at 4-bit experts with perfect acceptance | ~40x | drafts diverge (D3: 7 tokens between must-fix exact, 28 lenient); and long passes are byte-heavy because the expert union grows k^0.67 |
| C2 engine ~3x its roofline (80 ms vs 28 ms per 4-row pass) | 2.9x | dependent kernel chain, gate waits |
| C3 dispatch latency: 48 layers x ~40 dependent kernels x 3.5-7.6 us = 7-15 ms per pass | needs >= 7-15 tokens/pass on its own | one dependent kernel per op |
**Biggest: C1.** Physics: at Q4-class bytes this model cannot reach 1,000 tok/s for one sequence on 300 GB/s even with a
perfect drafter unless passes keep >= ~100 tokens; any real route needs fewer bytes per accepted token (fewer or
smaller active experts per token = a trained variant) or many parallel sequences. 1,000 tok/s AGGREGATE is within
physics: a layer-major batch of ~256 streams reads every expert once per step (68 GB / 300 GB/s = 0.23 s per 256 tokens).
Open question for invention: a single chat that legitimately decodes as many parallel sequences (Kamen's bar: answers
must not change drastically).
**Chat1000 (new researcher, 16:50, ~/Chat1000/GAPS_C.md): the first physically open route.** On the 125B expA traces the
expert union per layer is 127-158 at K=64 and 171-212 at K=128 (exponent 0.47-0.54 over 16-128 rows, below the 32-row
k^0.67 fit). Route S (one strict stream) needs per-token acceptance >= 0.993-0.998 (exact match caps ~0.97 by near-ties):
lossless ~160-210 tok/s, trained judge ~440 (code/math), copy ~900. Route P: N far-apart SECTIONS of the same answer cost
only 1.06-1.23x one chain's union (128x1: 225 vs 193 experts/layer); experts follow the topic, not the position; N
different chats cost ~1.5x. So writing one answer as many parallel sections has a ~1,200-1,440 tok/s byte ceiling at
MSB-3 in RAM; its open questions are compute (~19 TFLOP/s needed at 128 rows per ~88 ms vs 16-27 measured) and quality
(answers written as sections differ in form: SoT family; Kamen's bar). Lossless variant (parallel scouts as lookup drafts
for the sequential stream) caps ~720.
Chat1000 mid-point (17:50): ~4.4x from the best lossless single stream; ~34 % of Claude Code output is naturally parallel
(Amdahl caps Route P per chat); the leading lever is U1, sticky routing (Temporal MoE / ReMoE: <= 1 expert switched per
token, router fine-tune): U(64) 142 -> ~73, one sequential stream ~670-990 tok/s on bytes at acceptance 0.95-0.98. It also
raises small-RAM cache hits (B, D): assigned to TrainLab as item 2 (local, 35B first).
**Chat1000 batch 2 (19:40): C is bounded.** Relaxed or floor-tolerant acceptance fails Kamen's bar (35B + Qwen3.5-2B,
GSM8K / chat / 9 real Claude Code turns: answers change 4-10x more often than the floor; agentic tool calls change for real,
e.g. a Read call became `cat`, port 8080 became 1234), so acceptance must stay EXACT (best shown alpha ~0.88-0.9). Engine:
at 128 rows routed experts are 78 % of the pass (184 ms vs 66 ms byte floor); a grouped small-M expert kernel + fused HC
mixers takes a 128-row pass 237 -> ~110 ms. Calibrated model (within 4 % of measured passes), one stream at exact alpha
~0.9: stock kernels ~100 tok/s, grouped kernel ~177, + sticky routing (m=1) + diet ~347, + a 0.94-exact drafter (never
shown) ~490. Zero-shot switch caps cost ~10x the gate's error budget (training needed). **Verdict: ~350 tok/s is the
realistic single-stream ceiling for this model on this Mac; 1,000 only for naturally parallel output or long copies.**
Chat1000 batch 1 (18:05): stock MLX wide passes run 2.5-4x below the roofline (16 rows 86 ms / 64 rows 242 / 128 rows 237 per
48-layer pass: wide passes need their own kernels); Route P passes Kamen's bar only on lists and how-tos (math, prose, naive
code fail), so it adds only 1.1-1.4x on typical chats; lossless scouts dead; a floor-level drafter would reach acceptance
0.990, Qwen3.5-2B gets 0.779: acceptance 0.98 needs a TRAINED target-conditioned drafter (~0.96 exact). Stack for ~1,000
tok/s one stream: wide-pass engine at roofline x trained drafter (alpha ~0.98) x sticky routing (U1) x dense diet.
Literature (research/single_stream_1000.md, 15:40): nobody reports >= 500 tok/s single-stream for a >= 30B model on <= 1 TB/s.
A second wall is COMPUTE: ~24 GFLOP per verified row and the M5 Pro's measured 16-27 TFLOP/s (BaseRT 2607.19438) give
~700-1,100 rows/s even at 100 % acceptance, so INT8-class matmuls (W4A8 on the neural accelerators) are required. The
only stack that reaches ~1,000 tok/s on paper: 32-row passes, verifier expert set capped at ~25/layer by commitment
weighting (AcceptMoE 2608.02989: 23-34 of 128 for a 64-token tree, -0.27 pt), a target-conditioned block drafter
(DFlash/DSpark lineage, ~2 ms per block), a learned judge lifting tau to ~17-25 on code/math (AutoJudge, Judge Decoding),
copy/suffix drafts merged in, INT8 compute: ~49 passes/s x tau 20 = ~980 tok/s. Every part needs training except the cap.
Copy-heavy single chat (15:20, 29 Claude Code turns, 35B traces): 57 % of generated tokens copy a prompt 3-gram, but a
copied token shares only 22 % of its source's experts and 64 generated tokens touch 46 % of a layer's experts: even a
perfect 64-token copy pass reads ~30 GB on the 125B (~640 tok/s ceiling in RAM).
Aggregate throughput (Kamen's second target, 15:15): batched layer-major decoding of N chats reads the expert union
512 x (1 - (1 - 10/512)^N) per layer per step. In RAM it needs the experts resident: all-MSB-3 = 52.8 GB + dense 3.9 GB
on a 64 GB Mac, and each stream carries ~113 MB of fp32 GDN state + 24 KB/token of KV. N = 32: union 239/layer = 24.7 GB +
3.9 GB per step = ~95 ms -> ~340 tok/s aggregate, state 3.6 GB + KV ~1.5 GB (2K tokens each): fits only just. N = 64 does
not fit (GDN state 7.2 GB). So ~300-350 tok/s aggregate is the in-RAM ceiling on this Mac; 1,000 aggregate needs ~256
streams (29 GB of GDN state) or a bigger-memory machine. GDN state in bf16 or shared prefixes would raise N.

## D. Base Mac mini (16 GB, SSD ~3.2 GB/s): 50-75 tok/s
Must be true: <= 43-64 MB/token from the SSD with reads hidden. Q4-like coverage costs 200-300 MB: 4-6x (B2 x a 4.6x
slower SSD). Frugal's 56-82 MB/token fits only with reads fully hidden (I1) and frugal's quality accepted.
**Biggest: SSD bytes (4-6x) at Q4-like; latency (B1) at frugal bytes.**
Update 10-22 01:48 (stage_time.py, SSD 3.2 GB/s with an assumed 0.10 ms floor, GPU at 2.5x the M5 Pro fast engine's
layer time): B12 + MTP ~21.5 tok/s (139 ms per 4-row pass; 61 waited reads x ~0.9 ms each dominate), + demand-first
prefetch k=1 24.4 (k=2 adds reads and loses); eager ~14-15. Prefill = 51 % of an agentic call (median 7.6 s). So 50-75 needs
~3x fewer waited bytes per token: wait rank 0 only (quality?), 3-bit waited reads (-22 %), MSB-2 pools for ~1.5x residency
(quality?), rank-aware LFU (I9, -7 %). The mini is a quality-for-bytes trade, not an engine problem.

## E. <= 4 GB version (8 GB Mac): 30-50 tok/s
Must be true: dense + KV + scratch + experts <= 4 GB. Measured floor without any experts 7.5 GB (dense 3.4 GB incl. bf16 HC
mixers 1.27, MTP, n-gram cache 1 GB, scratch). **Biggest: the RAM floor itself (~1.9x over budget before any expert).** Measured non-routed weights (safetensors,
14:48): 3.89 GB = GDN projections 1.17 (in_proj_qkv 0.53, in_proj_z 0.32, out_proj 0.32) + hyper-connection mixers 1.26
in bf16 (4 x 0.315; 32 % of the trunk) + lm_head 0.36 + embed_tokens 0.36 (one row per token: can live on the SSD) + QSA
attention 0.36 + router gates 0.07 + shared experts 0.13 + PLE projections 0.07 + MTP 0.10. An aggressive diet (HC 4-bit,
embeddings streamed, lm_head 3-bit, no n-gram cache) reaches ~2.5 GB of weights + ~1 GB runtime: a 4 GB build would hold
~0.3 % of the experts, i.e. top-2-only routing (35B: KLD 0.34). E needs a different (trained/smaller) model; the diet
still matters for B: at 12 GB it lifts 3-bit residency from ~10 % to ~15 %.
Streaming the dense part costs 3.4 GB per pass (0.23 s at 14.6 GB/s): not viable; it must shrink to ~1.5-2 GB (HC mixers
8/4-bit, GDN/attention at ~3 bits with rotation) and experts must stream with reads hidden (I1).

## TOP FACTOR NOW (04:36): cold-read stalls / SSD bytes per pass, for BOTH the 24 GB main track and the 12 GB line
True all-hit 4-row verify passes (MoonShots m1b_fixed, 0 misses): BigMoE 38.2 ms, engine 38.5, gate-free prototype 39.1 ->
compute is near the speed-of-light layer (~0.80 vs 0.65-0.72 ms/layer at M=4): compute headroom ~10-20 % only; the per-layer
gate costs ~nothing. Streamed passes: ~100 ms at 24 GB (~62 % stalls), B12 + MTP ~84 ms at 10 % (~55 %). At 24 GB a pass
reads ~0.68 GB (~48 ms at 14 GB/s), so once overlap is good the main track is near SSD-bytes-bound: bytes per miss (3-bit
tiers done; 2-bit-heavy records4 plans gated on GSM8K), residency and overlap (OOO) are the levers, not kernels. The
BIG BUILD is stopped (ENGINE_DESIGN.md); MoonShots measures the stall anatomy (24 GB exact and B12) next.

## RETRACTED 04:34 (MoonShots): the 'CPU-bound gate path' below is WRONG as stated. The 57 ms of 'graph build' was the CPU
## blocked inside the per-layer mx.async_eval waiting on GPU and gate work (ASYNC_EVERY=0: build 5-6 ms, passes slower,
## 69.8 ms); Python wrappers are ~0.1 ms/layer. The 'all-hits' runs drifted (60 % misses vs the prototype's residency
## snapshot), so 60 vs 38 ms mixes gate cost and stalls. A true all-hits test (same positions, restored state) decides.
## (original text kept for the record:) NEW TOP FACTOR (04:04, MoonShots M1b, data/day15/m1b_*.txt): the per-layer CPU gate path
With every expert resident (24 GB tier env, 4-row verify, fresh process), a pass is CPU-bound in the MoE/gate path: BigMoE
60.7 ms per all-hits pass, 57.1 ms of it graph build until the last eval; the engine with BigMoE's MoE block 59.5 ms; the
engine's dense path with the shared expert only 22.8 ms (GPU-bound, 2.5 ms of build). BigMoE's MoE block costs ~1.1 ms of
CPU per layer (post_ids/spin, tier-GLU wrapper, look-ahead routers, per-layer async_eval): ~53 ms per pass. Gate-free hit
path target (GPU-side residency lookup, CPU only on misses; routed experts ~0.3 ms/layer of GPU at M=4): ~37 ms per
all-hits pass, ~1.6x. Streamed passes (101-105 ms) are stall-bound today. This also explains I8's falsification: the GPU
was waiting on the CPU's per-layer build and gating, not on reads. It applies to BOTH lines: B (frugal waits only for
ranks 0-1, so most layers need no CPU at all) and A (the exact main track's hit layers; exactness can be kept if the
GPU-side lookup computes the same residency decisions). Owner: MoonShots (engine) first; port to BigMoE's exact engine.

## RE-RANKED 10-22 01:43 (supersedes 22:00; tonight's evidence)
1. **12 GB line (B).** B12 (frugal 10 % on v3, HC8): eager 11.6 GB footprint / 24.2 tok/s; + MTP 13.1 GB / 38.6 code, 26.4
   prose (trim to < 12 GB queued; Eval suite verdicts in the morning). Factors in the fast-engine world: (a) engine compute
   ~1.5-2.4x (MoonShots, M1a correctness); (b) waits ~30 % of a fast MTP pass: I8 demand-first staging prefetch +16-19 %
   modelled (Breakthrough A/B after the trim), same-layer forecast ~+6 % more, rank-aware LFU (I9) +2-7 %; (c) tokens per
   pass: 125B head retrain (TrainLab, tens of %); span-aware relaxed acceptance on prose KILLED (+10.5-11 %, changes nearly
   every answer); (d) B5 prefill: 19 % of an agentic call (M5 Pro), 51 % (Mac mini): keep chunks large.
   Product ~80 tok/s code; 100+ needs the engine near its byte floor (~17 ms per 4-row pass).
2. **~1,000 tok/s (C): OPEN.** Filled table (Chat1000): F 0.27, S 1.005, stacked ~450-600; 1,000 needs ~15-20 accepted
   tokens per pass on novel text (per-step acceptance ~0.95-0.97) or F ~0.6-0.7. The 35B's high tau was copy-heavy stored
   text (Chat1000 mtpgap); the 125B head already accepts 0.75-0.90 per step on its own novel text. Live levers: head retrain
   (A), wide-pass engine (T_pass), copy pointers (F); fuzzy copy +9 % only.
3. **A (24 GB exact):** main-account sessions; tonight's addition: SSD contention A/B (PF_QD 1/2/4/8, demand-first rule).
4. **D (Mac mini):** SSD-bound decode (~40-57 tok/s ceiling for B12 + MTP) and prefill-bound agentic calls (median 7.6 s).
   Levers: I9 LFU, larger chunks, lower-bit pools for more residency (quality unknown).

## RE-RANKED 22:00 (Kamen 21:50: research continues toward ~1,000 tok/s and the 12 GB line; A2 dropped) -- SUPERSEDED
1. **12 GB line (B):** candidate config B12 (frugal 10 % on the v3 tier engine + 4-bit HC mixers, ~11.3 GB) in its checks
   now; decode verdict (math, IFEval, HumanEval, multilingual, agentic) decides quality. Speed: today ~40 tok/s (frugal +
   MTP); the new roofline engine (1.45-1.5x, MoonShots) and cold rank 1 half-late (+7-9 %) -> ~60-70; then tokens per pass
   (MTP ~3.2: chained-MTP fine-tune inconclusive so far) for 100+. Routing-prediction levers are information-limited on the
   125B (pre-routing top-2 precision 62 %, estimator fits pending); zero-shot stickiness transfers almost nothing (cap4: LRU
   hit 0.676 -> 0.712, cap_sim.txt).
2. **~1,000 tok/s (C): OPEN (Kamen 23:45, standing goal).** What tonight capped is ONE ROUTE - exact draft-and-verify on the
   125B with today's drafters and today's pass cost: General text ~350 with every lever; agentic turns on the
   125B (Chat1000 REPORT.md section 4): MTP-only tokens/pass bash 3.5, text 4.0, edit 6.2, write 2.8; best realistic policy
   4.4 tokens/pass; ~75-100 tok/s in RAM with a grouped expert kernel; even a perfect copy drafter caps ~620 tok/s on
   copy-heavy edits (~53 ms per 32-row pass). NEW LEVER from it: the 125B's MTP head is the weak link (the 35B head gets bash
   10.3 / text 8.8 on the same turns), so fine-tuning the 125B's MTP head on agentic text (TrainLab, EAGLE-3-style, local)
   is an exact-safe tokens-per-pass lever for EVERY track, Kamen's workload first.
   (Earlier text kept below.) Bounded at ~350 for general text with exact acceptance (Chat1000 REPORT.md). Kamen's own workload
   is the exception worth chasing: the 35B's stock MTP accepts tau(16) = 10.3 on bash, 8.8 text, 7.7 edits vs 2.8 prose
   (TrainLab turn 3), and copies reach ~900 tok/s. Research focus: agentic/edit turns (lookup + MTP hybrid drafting, wide
   passes with a grouped expert kernel, sticky routing only if training makes it transfer to the 125B).
3. **A (24 GB exact):** in BigMoE/MoEPrefetch/MoonShots hands (3-bit tiers, OOO, 2-bit gate), unchanged.

## What would have to be true for ~1,000 tok/s in one chat (for Kamen's morning summary, 00:15)
visible tok/s = [1 / (1 - F)] x S x A / T_pass, where F = share of visible tokens that need no 125B pass, S = parallel
streams inside the chat, A = accepted tokens per verify pass, T_pass = verify-pass time.
| assumption behind tonight's cap | today (125B, agentic) | for ~1,000 | route that breaks it | cheapest decisive test |
|---|---|---|---|---|
| every visible token is decided by a 125B pass (F = 0) | F = 0 | F ~0.6-0.8 | copy-by-reference output (the model points at a span, the runtime expands it; ~41 % of agentic tokens are copies, ambiguity removed by an explicit pointer); grammar-forced tokens (tool-call / JSON syntax) | CPU on Kamen's sessions: share of output in unambiguous copy spans >= 16/32/64 tokens + share of grammar-forced tokens |
| drafts must match exactly, token by token | A ~4.4 | A ~20-50 | workload-trained draft head (TrainLab, tonight); span-aware acceptance (exact on tool-call args and code, relaxed on prose) | 35B decode: span-aware rule vs the numeric floor on the same agentic turns |
| verify passes cost ~1.85 ms per row (stock kernels) | 237 ms / 128 rows | <= ~0.9 ms per row | grouped small-M expert kernel + fused HC (Chat1000 spec: 128 rows 237 -> ~110 ms); W4A8 / INT8 matmuls | MoonShots microbench on the new engine (scope extension) |
| bytes per pass grow with the expert union | U(128) ~171-212 / 512 | ~2x smaller | sticky routing or fewer experts per token (TRAINED; zero-shot does not transfer to the 125B) | TrainLab 35B curve first, 125B check on traces |
| one sequential stream per chat (S = 1) | S = 1 | S ~2-4 on parallel parts | independent tool calls / files / list items as parallel streams (34 % of Claude Code output) | Chat1000's Route P numbers on agentic turns (Amdahl-bounded) |
**FILLED by Chat1000 (01:12, REPORT.md section 5; 102 of Kamen's sessions, 7.0M output tokens):** output mix bash 46 % (mostly
heredoc scripts), Write 16 %, prose 14 %, tool args 10 %, grammar-forced 10 %, Edit 3.4 %. F: pointer-emittable verbatim copies
>= 16 / 32 / 64 tokens 24.7 / 15.8 / 7.7 % gross, 16.5 / 12.7 / 6.9 % net of a 12-token pointer -> F ~0.27 (1.37x), an upper
bound. S_eff = 1.005 (one long call dominates each message). A: span-aware acceptance (exact in tool calls / code, ratio 0.3
on prose) keeps tool calls intact (8/9 turns identical after the MLX-bug re-check; the one change is a prose reply), tau16
+17 % on the 35B, but prose is 14 % of Kamen's output (~+3 %); the A
lever that matters is the 125B MTP head retrain (stock 125B chain 0.85 / 0.71 / 0.61). T_pass: best composite ~22-24 ms at
9 rows; floor = the 2.1 GB dense read (~9 ms). **Composite: today ~55-60 visible tok/s in RAM; every lever stacked (each
unproven on the 125B) ~450, ~600 with 3-bit dense.** ~1,000 needs ~15-20 accepted tokens per pass on novel agentic text or
F ~0.6-0.7; Kamen's data shows neither today. The goal stays open; the drafter (A) is the biggest lever.
**REVISED 03:54 (Chat1000 REPORT.md section 7 addendum, after the head fine-tune kill):** with today's drafter A ~4 on Kamen's
workload, every other lever stacked (trained sticky routing + grouped kernels + HC 4-bit, pass ~18-24 ms, F 0.25 by
pointers) gives ~250-300 visible tok/s, not ~450-600 (which assumed A ~8). The deciding lever for ~1,000 is a drafter
that accepts far more than ~4 tokens per pass on NEW agentic text; nothing measured tonight provides one. A proper
drafter run (EAGLE-3 / FastMTP-style training-time test inside QSA, the 125B's own text, several GPU nights) is Kamen's call. NewIdeas recommends NOT now: FastMTP's gain was mostly teaching recursion (vanilla 70/10/0 % per position -> 80/56/36 %);
the 125B head already chains (85/71/61/51 % cumulative), so the headroom is domain adaptation (+10-20 % at best), the
125B's own text costs ~8-10 GPU hours per 1M tokens here (FastMTP ~1e8), and tonight's attempt lost 4.6 %. Revisit when
the new engine makes wide draft trees cheap.
(Original illustration:) Example that reaches ~1,000: F = 0.7 (copies + grammar), A = 8 (better head, span-aware), T_pass = 30 ms per 16-row pass
with the grouped kernel: 3.3 x 8 / 0.030 = ~880 tok/s visible; with S = 1.2 on parallel parts ~1,050. None of these needs
new hardware; F is a protocol/format change, A a drafter/acceptance change, T_pass a kernel change.

## Prize x chance (Kamen's rule 7, 13:00): gain if fully solved, what it would take, best so far
| factor | prize if solved | what solving it requires | best so far | chance |
|---|---|---|---|---|
| B2 bytes at Q4-like (small RAM, mini) | 2-3x (mini 4-6x) | <= 100-146 MB/token at the bar | gate (15:10): frugal FAILS; only ~110 reads/token (m85) passes = 245-305 MB; near-tie re-route -29 % (gate pending) | needs a bar decision by Kamen (words vs formal gate), a trained model, or 2x SSD bandwidth |
| B3 / A4 / C2 engine compute vs roofline | 1.7x exact, 2.9x small RAM and in-RAM | per-layer kernels at ~80 % of the read ceiling, <= ~10 dependent dispatches per layer | 3c GLU -23..-28 % per layer, exact fusions ~+30 % est. | medium-high: the best prize x chance, and it multiplies every target |
| B1 stalls in small RAM | 2.5-3x | experts known >= 1 layer ahead at recall ~0.95, or late compute without quality loss | look-ahead recall@10 0.67-0.73; deferral d=1 leaves 29 % of drop's damage (I1); Edge0 trains a pre-router (non-exact) | low training-free (information limit below); medium with training |
| A1 serial loader, exact | 1.57x | recall@10 ~0.98 one layer ahead | look-ahead +14 %, ridge correction +2.9 %, OOO +3-8 % | low: layer L+1's routing depends on attn_{L+1} and on layer L's own MoE output (35B: router recall 0.72-0.82 @8 without L+1's attention, 0.96 with it), which do not exist one layer ahead |
| E dense floor (4 GB) | 1.9x on RAM | dense <= 2 GB (HC mixers 4-bit, GDN/attention ~3-bit rotated) | HC 8-bit -0.63 GB | medium, quality risk on GDN |
| C1 accepted tokens per pass | ~20-40x | >= ~100 tokens per pass at 4-bit, or far fewer bytes per token | ~3.2 (MTP) | ~0 with this model: a trained parallel decoder or many streams |

## Ranking of the factors across targets (what to attack first)
1. **SSD latency in series with compute (A1/A2, B1, D): 1.3-3x everywhere.** I1 deferred experts tested 12:50: weak (a
   late expert returns only 55-70 % of its value, 29 % of drop's damage left at d=1): kept as a d=1 trade for low ranks
   (Breakthrough measures it on the 125B). I5 lazy exactness (repair the context, not the token) tests whether the stall
   can be traded for background reads instead.
2. **The quality-bar question (B2, D)**: frugal's verdict moves B2 from 2-3x to ~1x. Eval suite (in progress).
3. **Engine compute vs roofline (A4, B3, C2): ~1.7-2.9x** -> MoonShots / BigMoE fusion; I4 (routed experts one layer late,
   taking the MoE off the dependent chain).
4. **Accepted tokens per pass for C (~40x)**: needs a new decoding structure, not a better drafter; physics says a trained
   variant or parallel sequences.

Final C report: ~/Chat1000/REPORT.md (19:50). Chat1000 stood down after it.
