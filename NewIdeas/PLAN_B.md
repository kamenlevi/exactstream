# Small-RAM line: the plan after Kamen's decisions (NewIdeas, chief researcher; 2026-10-21 17:20)
(The earlier three-option note is DECISION_B_options_1535.md.)

Kamen, 17:10: (1) the small-RAM line is judged by TASK RESULTS (decode tests within a few points: math, IFEval,
HumanEval, a multilingual test to add; similar answers), not the per-part KLD gate. Frugal routing at 10 % residency
qualifies (math 90 vs 86, IFEval -2.0, HumanEval -1.7); at 5 % it fails (IFEval -7). (2) No external GPUs, ever: all
training is local.

## 14:06: B12's Eval suite verdict: FAIL, 3/5 rules (Breakthrough report_b12.txt, 14:03; 210 core rows, mass weights, guard, MTP)
IFEval 85 % vs ref 90 % (-5.0 [-10.4, +0.4], McNemar 8/3): FAIL; truncations / loops 18 / 2 vs 13 / 2 (5 extra capped rows): FAIL;
math 82 vs 86 (-4.0 [-10.6, +2.6]); HumanEval+ 90 vs 91.7 PASS; length x1.005 PASS; hesitation PASS; answer agreement 91.0 % vs the
floor's 97.1 %. B12-mass hard KLD on the real v3f path 0.1117 (vs frugal 10 % 0.1009 on fork125bt: v3's 3-bit lower ranks add
~10 %). Next: B8's core sets first in the suite's queue (B8 ~0.085 expected, -25 %); the next 12 GB candidate B12 + cold ranks 2-3
late (LATE3=2,4 / 2,10) for quality, with MoonShots' multi-row phase B as its speed fix (handover). Open question: what made B12
worse than yesterday's w2f_r10 (IFEval -2.0)? v3 tiers, MTP, guard, or noise (CI ~10 points).
ANSWER (14:07, Breakthrough): mostly v3's 3-bit lower ranks. Every decode signal moved together from w2f_r10 (fork125, 4-bit, AR) to
B12 (v3f tiers, mass, MTP, guard): math 90 -> 82, IFEval 88 -> 85, truncations 12 -> 18, hesitation words 0.20 -> 0.44 per 1K
(ref 0.27), agreement 92.4 -> 91.0 %. Same 2,353 hard-text positions: exact-4-bit frugal 0.1009, v3f tiers renormalised 0.1230
(+22 %), v3f tiers + mass 0.1117 (+11 %); HC8 adds ~0. So 4-bit pools should buy most of it back (mass on top may beat
w2f_r10), but RAM: all-4-bit pools at C_m 51 = 7.2 GB vs 6.0 (B12 + MTP ~12.4 GB at 2.6K, over 12); C_m ~46 at 4 bits ~11.8 GB.
Next session (Breakthrough's HANDOVER): tf_b12 KLD for B12-mass 4444444444 at C_m 51 / 46, B12 + LATE3=2,4 / 2,10, B8-mass, next to
0.1117; then ONE decode verdict on the best that fits 12 GB; LATE3 speed via MoonShots' multi-row phase B.
15:55 CONFOUND (Breakthrough 15:45 + NewIdeas replay, GAPS G): the verdict run had the repetition guard on, the reference did not,
and the guard misfires on normal math (prealgebra/1640: '32' -> '16', run to the cap). Replaying the guard over the no-guard B12
run: it could touch 12 of 148 rows (10 math, 2 IFEval), but the two runs differ on 139 rows because the frugal pool carries
history across rows. So B12's own run-to-run noise is large: no guard IFEval -3.1 (PASS), truncations 14 vs 12; guard -5.0
(FAIL at the boundary), 18 vs 13. Like-for-like decides: B12 guard-off vs ref (62 rows missing: 2 IFEval + 60 HumanEval+) or both
with a fixed guard (R2). For paired A/B runs of B12 variants (next session's single verdict), reset the pool at each row (or run
twice) so the comparison is not dominated by history noise. Verdict stays FAIL-pending until then; the next-session plan above
is unchanged (4-bit pools / LATE3 lower the KLD regardless).
16:02 B12 RUN-TO-RUN NOISE (scripts/b12_runnoise.py: the suite's own scorers on results/ref, b12 (guard), b12_noguard; "could not
touch" = no guard penalty on a chosen token in the no-guard output, rg_replay.py):
| set, rows | ref | B12 guard | B12 no-guard | B12 vs B12 discordant | B12 guard vs ref (win, loss) | B12 no-guard vs ref |
|---|---|---|---|---|---|---|
| math, 50 shared | 86.0 | 82.0 | 88.0 | 0 / 3 | 1 / 3 | 2 / 1 |
| math, 40 the guard could not touch | 92.5 | 92.5 | 95.0 | 0 / 1 | 0 / 0 (same pass/fail on every row) | 1 / 0 |
| IFEval, 98 shared | 89.8 | 85.7 | 86.7 | 5 / 6 | 3 / 7 | 3 / 6 |
| IFEval, 96 the guard could not touch | 89.6 | 87.5 | 86.5 | 5 / 4 | 3 / 5 | 3 / 6 |
Capped answers on the 148 shared rows: ref 12 (math 7, IFEval 5), B12 guard 16 (9, 7), B12 no-guard 14 (6, 8).
Reading: (1) math: every math loss of the guard run vs the reference sits in the 10 rows the guard could act on; without the
guard B12 matches or beats the reference (88 vs 86). (2) IFEval: two B12 runs disagree on 11 of 98 rows, as many as either run
disagrees with the reference (10, 9). B12's IFEval deficit is ~2-3 points on the rows the guard could not touch, inside the
5-point margin, and one run's noise (~2.4 points SD) is half the margin. (3) Truncations stay marginal without the guard (14 vs
12, the rule allows 13), driven by IFEval (8 vs 5). So B12 is BORDERLINE, not a clear FAIL: math passes, IFEval ~-3 +- 2.4,
truncations +-1 of the line. For history-dependent variants (B12, B8) one run cannot settle a rule whose margin is ~2 SD of
the run noise: the Eval suite should judge them on two runs or with a per-row pool reset (its choice).
The truncation excess without the guard is real: 3 IFEval answers capped in B12 no-guard but not in ref are runaway generation
without verbatim repetition (ifeval/1248 a rap cycling "he's got a X, he's got a Y" 1,024 tokens vs ref 417; 2304 a speech
1,024 vs 600; 3538 an endless hashtag list 1,024 vs 968). No guard variant arms on them (rg_replay: R0 / R1 none), so no guard
fixes them; math truncations are net -1 for B12. A mild B12 quality signal for the next step (4-bit pools / LATE3) to remove.
20:08 B8 VERDICT (Eval suite report_b8.txt 19:42; like-for-like: guard OFF both sides, B8 EAGER = 5 % residency + LATE3=2,4, mass):
FAIL 3/5. IFEval 84 vs 90 (-6.0 [-11.1, -0.9], McNemar 8/2) FAIL; truncations / loops 17 / 3 vs 13 / 2 FAIL (B8's loops:
ifeval/1219, 1675, 3439: guard-off, so the known 3-bit loop runs to the cap on both lines); math 86 = 86; HumanEval+ 88.3 vs 91.7
PASS; answer agreement 93.3 % (floor 97.1 %). Same shape as B12: IFEval at the -5 boundary (one run's noise ~2.4 points SD; the
.r2 second run decides), truncations a few over. Reading: both small-RAM settings sit AT the bar, not clearly inside: the next
quality step (4-bit pools / LATE3 at 10 %, Breakthrough's day_d1n.sh KLD screen) is likely needed for a clean pass. MoonShots'
multi-row phase B (GPU from 20:08) measures B8 and B12 + LATE3 speed under MTP.
21:56 THE QUALITY LEVER IS LATENESS, NOT 4-BIT POOLS (Breakthrough tf_b12, real v3f path, ifeval + multi every 8, 2,353 hard
positions, mass weights): B12 0.1117; 4-bit pools C_m 51 0.0993 (-11 %, +1.1 GB), C_m 46 0.1082 (-3 %); B12 + LATE3=2,4 (cold
ranks 2-3 half a layer late) 0.0508 (-55 %); B12 + LATE3=2,10 (all cold ranks 2-9 late) 0.0178 (-84 %): ifeval 0.0186 / multi
0.0170 = INSIDE the strict per-part targets (0.0204 / 0.0206), near the v3 control at C=128. CORRECTION of my 14:07 relay ("cause
mostly v3's 3-bit lower ranks"): on the same runtime the tiers cost ~11 %; DROPPING cold ranks is frugal's real cost.
Price = SSD bytes: every cold rank is read (late). Pre-registered estimate (NewIdeas, before Breakthrough's speed rows): with
MTP 4-row passes at C_m 51, LATE3=2,10 reads roughly 1.0-1.5 GB per pass late (~500-650 MSB-3 reads), i.e. 75-110 ms of SSD at
13-14 GB/s: SSD-bandwidth-bound even with phase B hiding the latency -> ~25-32 tok/s on code (B12 today 35-39). LATE3=2,4 adds
~0.5-0.7 GB per pass (~40-50 ms), which phase B can hide under ~38 ms compute only partly -> ~B12 speed minus 0-15 %. Ask: report
> (a quote from the project owner, kept private)
B8 + MTP 192-token check passes on the late-write fix: 27.2 / 17.1 tok/s code / prose.

## 11:39: next speed lever for the 8-9 GB line (Breakthrough's HANDOVER, OPEN (5))
MoonShots' multi-row phase B (cold experts half a layer late inside MTP verify passes; bit-exact build in ~/MoonShots/defer) is
the right lever for B8 + MTP, whose block waits for the late ranks 2-3 inside verify today. On the deployed 24 GB engine it
substitutes for JIT prefetch (no gain), but small RAM has no effective prefetch. Needs the BIGMOE_DEFER_FULL ext rebuild in v3f
(not while the Eval suite has v3f loaded) and a multi-row late path in B12's frugal block (~1-2 h); measure B8 + MTP tok/s and B8
KLD on tf_b12.

## 07:05: B12 weight normalisation: mass, not renormalised (Breakthrough's proposal, agreed)
B12's decode renormalised its weights over the experts it computes (w2f). Wednesday's tf_frugal_var at 10 %, ifeval + multi,
every 4th, 3,308 positions: base (substitutes, renormalised) 0.1066; subnr (substitutes at their own router probability /
the true top-10 mass, no renormalisation) 0.0961 (-10 %); noren 0.1039; nosub (no substitutes, renormalised) 0.1408. Built as
an opt-in B12_NORM=mass (47377d7), no speed / RAM effect. Decision: the verdict runs on B12_NORM=mass, gated only by a GSM8K-20
sanity row; the confirming tf run on today's HC8 config follows for the record; if the Eval suite already started on w2f,
judge mass in a paired follow-up instead.

## 06:48: skip-residents small for B12; C_m 56 per-pass gain only +1.5-2.4 % (Breakthrough)
Prefill union already resident: MSB part 14-17 %, both parts 6-7 % -> skip-residents saves <= ~12 % of a small call's bytes
(~0.4 s of ~3.5 s): not built for B12 (the main track at 24 GB has ~30 % MSB-resident; TrainLab's careful version there).
GSM8K-10 at the decoupled default 10/10, all equal to exact eager. C_m 56 on Breakthrough's tree fits (11.8-11.9 GB) but
gains only +1.5-2.4 % per verify pass (long 25.7 vs 24.6 tok/s; code 34.2 vs 35.3 with different text): B12-11 stays a named
short-context option for its -5 % KLD.

## 06:40: C_m 56 is the better B12 setting under 12 GB (TrainLab)
Hard-text KLD (tf_frugal_var base_hc8, ifeval + multi, every 8, 2,353 positions): C 51 0.1030 (ifeval 0.1185 top-1 0.870, multi
0.0843 / 0.907, 20.9 reads/step) -> C 56 0.0980 (0.1127 / 0.872, 0.0803 / 0.906, 19.4 reads/step): -5 % KLD, -7 % reads, and
+7-14 % decode at 11.8-12.0 GB (server path). Recommendation: B12 := C_m 56 if Breakthrough's MTP footprint at 2.6K holds
<= 12.0 GB; the morning verdict at C 51 is then a conservative lower bound on quality.
DECIDED 06:41 (Breakthrough, agreed): B12 stays load_b12(0.10) = C_m 51 as the default and the verdict candidate, because
pools are sized at load and Kamen's agentic sessions run tens of K tokens (51 keeps 12 GB up to ~16-20K with the drafter;
56 is 12.0 GB already at 2.6K, ~13.5 at 32K). C_m 56 = load_b12(0.11) becomes the named short-context speed setting
'B12-11' (+7-14 % decode, -5 % KLD, chats under ~3K tokens), pending Breakthrough's own footprint row.

## 06:30: RAM line by context (Breakthrough; peak physical footprint, cache cap 64 MB)
| | 2.6K | 32K | 64K |
|---|---|---|---|
| B12 eager | 10.9 GB | 12.5 GB (12.1 with attention 256 / MoE 2048, -26 % prefill) | 14.1 GB |
| B12 + MTP | 11.3 GB | 12.9 GB | 14.6 GB |
The long-context peak is MLX memory in the full-attention layers' prefill (fp32 scores ~ query chunk x keys x 16 heads x 4 B:
~1.2 GB at 37K, 2.1 GB at 66K) plus the MoE transient (~1.0 GB at 4096 rows); at 64K the resident state alone is ~11.7 GB
(KV ~1.6). So B12 fits 12 GB up to roughly 16-20K tokens of context with the drafter. Default stays 512/4096; 256/2048 is an
opt-in low-memory mode; 64K under 12 GB needs 8-bit KV (new code, after the verdict) or a smaller expert cache at long
context (quality check). Correction: 'peak active' figures quoted after 04:25 were per segment (reset per MoE call);
footprint figures stand. RB2 on the union path is slower than C++ read_rebuilt (177 tokens 4.39 vs 3.83 s): C++ stays.

## 06:00: residency sweep (TrainLab; B12 + MTP 3:3:0, server path D1I_FRESH, PF_GROUP 64, C_l = 0.4 C_m, 128 tokens)
| C_m | long: tok/s, tok/pass, waited/pass, peak | code: tok/s, tok/pass, waited/pass, peak |
|---|---|---|
| 51 | 25.2, 2.56, 256, 11.4 GB | 36.2, 3.28, 201, 11.2 GB |
| 56 | 26.9, 2.56, 241, 12.0 GB | 41.4, 3.56, 193, 11.8 GB |
| 61 | 26.8, 2.56, 242, 12.6 GB | 41.9, 3.56, 178, 12.4 GB |
| 66 | 26.5, 2.46, 217, 13.2 GB | 39.7, 3.46, 180, 13.0 GB |
C_m 56 is the largest within 12 GB at 2.6K: +7 % on the long prompt (equal tokens/pass); code's +14 % is partly different
text (frugal outputs change with residency). Stalls fall slowly with residency (waited -6 % at 56, -15 % at 66): residency
is a small lever at this size. KLD 51 vs 56 next (expected to improve with more residency).

## 05:57: 32K with the drafter 13.6 -> 12.9 GB (Breakthrough)
MTP head fed per MoE chunk (D1I_HEAD_EACH): exact (same sha at 2.6K and 32K), now the default. n-gram hot cache off: no gain
(eager 12.6, MTP 13.1). The rest is a transient during long prefill only (footprint exceeds MLX peak active by ~1.8 GB; both
read 11.5 after decode): a 50 ms footprint sampler tagged by (MoE chunk, layer, part) is queued. RB2 (copy-free GPU rebuild)
wired into the union path (BIGMOE_TIER_PF_GPU=1 D1I_RB2=1): 177 / 326-token rows with sha checks queued.

## 05:40: the decoupled prefill PASSES the paired long-context check (Chat1000 REPORT.md section 8, data/longctx/)
B12, Breakthrough's env, one process per mode, 14 tasks / 114K prompt tokens (needles 4K / 8K, key-value ~10K, variable
tracking 8K, 3-hop over a ~10K tool log, RULER needle 16K, one real 9K agentic turn): chunk512 11/13, prefill 1,036 s, 11.4 GB;
decoupled 11/13, 13/14 answers identical to chunk512 (one word choice at token 6), prefill 340 s (3.0x), 11.7 GB; chunk2048
(floor) 11/13, 12/14 identical, 438 s, 14.1 GB. Decoupled is inside the floor between two plain chunkings: adopt.

## 04:59: B8-fast + MTP (TrainLab): <= 8.5 GB, 23-28 tok/s
frac 0.05, wait ranks 0-2, MTP_C 32, PF_GROUP 64, spec 3:3:0: long prompt 8.2-8.4 GB at 23.7-23.8 tok/s (2.67-2.78 tok/
pass); code prompt 8.2-8.5 GB at 22.7-27.8 tok/s (the decoupled row's text differs, so its acceptance does). Streaming ring 4
+ rebuild kernel: B12 long 388-423 tok/s at 11.5 GB (~400 at the decode footprint), B8-fast long 383 tok/s at 8.5 GB.

## 04:55: D1i PASSES its kill line (TrainLab): B12 long prompt (2,617 tokens) prefill 426-474 tok/s at 11.8 GB
v3 streaming ring 6 + a copy-free GPU rebuild kernel (reads the 3-bit / 1-bit words in place from the raw records ring);
greedy sha identical to the non-streamed decoupled prefill; decode unaffected (19.5 / 22.6 tok/s). Reference: 162-172 tok/s
at 13.9 GB (2.7x faster, 2.1 GB less). Ring 8 no longer helps (read-bound: all 512 experts ~72 GB per 2.6K at 12-13 GB/s).
Batch-to-batch variance ~10-15 %. Use streaming only for long prompts: a full sweep costs ~6 s whatever the length, while
the non-streamed union read is cheaper below ~1.5-2K tokens (150 tokens touch ~184 experts/layer). Adoption: Breakthrough.

## 04:54: the 6-8 GB end holds with real prompts (TrainLab, decoupled prefill; long prompt 2,617 tokens, all fluent)
| config | prefill tok/s | peak footprint | decode tok/s |
|---|---|---|---|
| B8-fast (5 %, wait ranks 0-2), chunk 256 | 58 | 7.9 GB | 17.5 |
| B8-fast, decoupled 256/4096 | 283-306 | 7.9 GB | 16.8-17.5 |
| B8-fast, + streaming ring 4 | 361-389 | 8.5 GB | 16.8-17.4 |
| B8 (late3 2,4), decoupled | 282-305 | 8.0 GB | 15.1-15.4 |
At 32K (37,106 tokens): B8-fast and B8 ~280 tok/s prefill at 9.6 GB (KV + 32K attention state +1.7 GB). Decoupled = plain
bit for bit on one-query-chunk prompts. Router readback is not the prefill bottleneck (0.06 s); the v3 rebuild's strided
slicing was (~3 s of the MoE phase): a copy-free rebuild kernel is next (ceiling ~436 tok/s for a full sweep at 12 GB/s).

## 04:44: smoke test PASSED; the Eval suite has the hook (Breakthrough 04:34-04:45)
Peak physical footprint, cache cap 64 MB, decoupled 512/4096 + grouped tier prefill 64:
- B12 eager, long (2.6K): prefill 278 tok/s, 10.9 GB, fluent. The Eval suite's own path (generate_ar, chunk 512): fluent,
  19.4 / 18.6 / 19.8 tok/s (code / prose / IFEval-style), 10.9 GB; tokens identical with and without env.py's PF1 flags.
- **B12 + MTP (PF_GROUP 64 + d1i prefilled state): code 35.3 tok/s at 11.2 GB, prose 24.4 at 11.2 GB; GSM8K-20 via MTP
  3:3:0 20/20, all answers equal to exact eager (eager B12 19/20), 34.7 tok/s.**
- 32K (37,123 tokens) eager: 271 tok/s prefill, 12.5 GB (0.5 GB over; MoE transient over 4,096 rows + ~0.9 GB KV).
Verdict config: generate_ar + MTPLX_PREFILL_CHUNK_SIZE=512 + BIGMOE_TIER_PF_GROUP=64 (AR-only caveat stated; MTP covered by
the GSM row). Queued: B12 + MTP at 32K / 64K with the prefill inside generate, 32K at MoE chunk 2048, eager 64K. TrainLab's
streaming (ring 6, +0.9 GB) stays an eager-only opt-in for now (MTP would cross 12 GB).

## 03:53: Breakthrough's slot (03:28-03:49) - prefetch falsified, prefill measured, long context breaks 12 GB
- Prefetch arm 1 (pre-registered +11-14 %): per-pass time code +0 % (83.9 -> 84.2 ms), prose -5 %; end-to-end code -11 %
  (residency changed frugal's outputs, tokens/pass 3.12 -> 2.78). 125 prefetches used per pass, 99 % on time. SHELVED
  (DEAD.md); for the new engine: profile per-layer wait vs compute first.
- Prefill, B12 eager, 2,617 tokens, cache-only chunks: 2048 175 tok/s / 13.2 GB; 512 81 / 11.8; 256 51 / 11.4 (~3x below
  the SSD-bound estimate). The logits fix saved 0.7 GB at 2048. D1i (TrainLab) targets >= 400 tok/s at <= ~12 GB.
- 32K context: B12 eager 18.7 GB, + MTP 19.1 GB peak (prefill attention over ~37K keys per 2048-query chunk + expert
  scratch + ~0.9 GB KV); decode 25.7 / 25.5 tok/s. B12 = '12 GB for prompts of a few K tokens' today.
- MTP's +1.4 GB is the 4-row verify (head alone = eager footprint; hot cache irrelevant); width scan queued.

## The biggest factors now (target: 100-150 tok/s at <= 12 GB on the M5 Pro)
| factor | today | needed | mechanism | owner |
|---|---|---|---|---|
| RAM: frugal 10 % peaks at 14.3 GB | 14.3 GB | <= 12 GB | dense diet (HC mixers 8/4-bit, embeddings on the SSD, n-gram cache off, lm_head 3-bit for drafts; -2-3 GB) + MSB-3 cache (1.29x experts per GB) | Breakthrough (diet; D1e port to BigMoE's tier engine) |
| engine compute per pass | ~82 ms per 4-row pass (39.2 tok/s code) | ~35-40 ms | roofline engine: 1.45-1.5x measured on the speed-of-light layer; GPU-side gating, no per-layer CPU sync | MoonShots (BIG BUILD, approved 16:30) |
| waits for cold ranks 0-1 | ~20 ms per pass | ~5 ms | cold rank 1 half a layer late (+7-9 %, measured); pre-routed top-2 (TrainLab zero-shot pending); look-ahead prefetch LOSES at equal RAM | Breakthrough, TrainLab |
| SSD bytes | ~82 MB/token at 10 % | fine | frugal reads ~29.5/token: ~200 tok/s of SSD headroom at 14.6 GB/s | - |
| quality margin below 10 % | 5 % fails IFEval -7 | pass | (a) cold ranks 2-3 half a layer late (35B: half-late cut 5 % frugal's KLD 0.193 -> 0.048); (b) local LoRA healing of frugal 5 % (TrainLab turn 1); (c) the multilingual decode row | Breakthrough, TrainLab, Eval suite |
Expected path: frugal 10 % on the tier engine with the diet at ~11-12 GB; ~40 -> ~60 tok/s with prefetch + 3c; ~80-100 with
the new engine and MTP (speed-of-light 4-row pass + drafts ~34-38 ms in RAM).

## Pre-routed frugal (19:00, TrainLab, 35B) - FAILED on the 125B at 19:45 (+11-26 % KLD; the look-ahead names the true
## top-2 only ~62 % of the time); revival = a trained top-2 estimator (TrainLab). B12 keeps late rank 1 (pending decode).
Layers 8-39: frugal's waited top-2 of layer L are chosen at layer L-1 (layer L's router + norm on L-1's post-attention
state); ranks 2-9 = the true router's best outside them, computed only if resident; weights = L's TRUE router
probabilities. vs plain frugal: C26 KLD 0.141 vs 0.168, cold reads 16.4 vs 18.6/token; C13 0.174 vs 0.202 - better
quality, fewer bytes, and every waited read gets a whole layer of slack (no stalls). All layers 1-39: 0.237 (early layers
fragile); the pre-router's own weights instead of the true ones: 2x worse. Next: 125B teacher-forced 'pre2' on both
subsets at 10 % / 5 % (Breakthrough), then GSM8K-50 decode (predicted routing for ALL ranks collapsed GSM8K in Speculating
Experts 2603.19289). If it holds, B12 swaps late rank 1 (+5.5 % hard-text KLD) for pre-routing.

## UPDATE 22:30 (Breakthrough's B12 runs, 125B, 10 %, eager)
- B12 PLAIN fits 12 GB as is: frugal 10 % on v3 + HC mixers bf16 (compiled HC) + no late: peak 11.3 / 10.6 GB, 23.8 / 22.7
  tok/s (code / prose); GSM8K-20 of the full variant 19/20, equal to exact eager. This is the B12 that goes to the Eval suite.
- Late rank 1 gains nothing on v3 (two phase-B spins + a one-expert 3c call per layer eat the overlap): dropped.
- HC mixers 4-bit are NOT free: +0.0243 hard-text KLD (top-1 0.980 -> 0.939), +0.0066 easy: dropped; HC 8-bit probe next.
- **New candidate "B8" (6-8 GB end of the target):** frugal at 5 % + cold ranks 2-3 half a layer late (l2t4): hard-text KLD
  0.0755 vs frugal 5 % 0.1740 and frugal 10 % 0.1009, at 67.4 reads/step (37.3 late). Needs: easy-text KLD, eager speed with
  the late reads on v3 (do they hide?), peak RAM, GSM8K-20, then the Eval suite verdict with B12.

## The 12 GB candidate configuration "B12" (18:35; one decode verdict instead of one per lever)
Frugal at 10 % residency on the v3 tier engine (MSB-3 pools + LSB sidecar for ranks 0-1) + hyper-connection mixers at
4-bit + cold rank 1 half a layer late. Measured pieces (Breakthrough 18:30): RAM floor 3.89 GB -> 2.94 with HC 4-bit;
peak at 10 % = pools + floor + ~2.3 GB temporaries: 4-bit pools 13.0 GB, + HC 4-bit ~12.0, v3 MSB-3 pools + HC 4-bit
~11.3 GB -> fits 12 GB. Late rank 1: +7-9 % eager speed, teacher-forced KLD +5.5 % on ifeval+multi (significant), +0 on
gsm+code. HC 4-bit KLD: Breakthrough's turn 2. Verdict wanted from the Eval suite on B12 as one config: math, IFEval,
HumanEval, multilingual and agentic decode rows vs the reference and vs frugal 10 % (w2f_r10, which passes today with
IFEval -2.0). If B12 fails IFEval, drop late rank 1 first (it carries the measured hard-text cost).

## TrainLab zero-shot (18:00, 35B): two findings that change the plan
- **Pre-routing is the cheap stall-free dataflow:** layers 8-39 routed one layer ahead (layer L's router on L-1's
  post-attention state, weights from L's true router over the chosen set): KLD 0.050 training-free, every read with a
  whole layer of slack, no cache dependence, no extra bytes (half-late at C26: 0.044 but needs the cache). Next: the same
  for frugal's top-2 only (pre-routed ranks 0-1 + frugal ranks >= 2) - if close to frugal, its waits vanish by routing.
- **Risk to the task-result bar:** copy-heavy tool logs (Claude-Code-like) lose their copy confidence under ANY MoE
  perturbation (KLD 0.8-1.4 for every flow vs 0.04-0.2 on prose/gsm/code); multilingual de is the weakest segment
  everywhere (frugal C13 0.54). The Eval suite gets an agentic decode row next to the multilingual one.

## 02:24: trim2 (Breakthrough 02:20; physical footprint, HC8, MTP_C 32)
B12 + MTP code, cumulative: cache cap 64 MB (did not take: MTPLX resets it at load; fixed) 36.0 tok/s / 13.2 GB; + prefill
chunk 256 39.4 / 13.0; + sidecar C_l 12 38.3 / 12.7; + base C_m 46 37.0 / 12.2 GB. Eager (cap + chunk 256) 11.5 GB; MTP prose
11.9. **A ~2K-token prompt sets the peak: B12 eager chunk 2048 -> 14.4 GB, 512 -> 12.0, 256 -> 11.5**, so the exact-prefill
scratch decides RAM for any real prompt, and the sub-batched prefill (D1i) is the fix for RAM and prefill speed alike (moves
up: right after the verdict). MTP costs ~1.5 GB over eager at equal settings (unexplained; hot-cache-off and draft-off rows
queued). Verdict policy (NewIdeas): judge quality at C_m 51 with chunk 256; C_m 46 only if the engineering fixes fail.
03:30 TrainLab (D1i, ~/TrainLab/d1i): bt_b12's prefill loop computes full-vocab logits for every chunk position (~1 GB
bf16 per 2048-token chunk, unused); MTPLX's emit_logits=False skips the vocab head with identical cache state: possibly
a third of the 2.9 GB chunk-2048 peak gap, before any expert sub-batching. Breakthrough told before its A/B slot.
Scope: SpecDecode's spec125 prefill already passes emit_logits=False, so the waste sits only in the plain loops (bt_b12
eager, BigMoE decode.greedy's chunk-major path): it explains the EAGER 2K-prompt peak, not MTP's +1.5 GB.

## 01:30: true footprints, trim, and the waits lever
Breakthrough 01:25 (lifetime-max physical footprint = Activity Monitor's Memory; code prompt incl. prefill; HC8; process
max RSS understates by 2.4-3.7 GB, MLX peak active by 0.5-1.0 GB): **B12 eager 11.6 GB (24.2 tok/s, fits 12 with 0.4 GB
margin); B12 + MTP 13.1 GB (38.6 tok/s code, 3.12 tok/pass; prose 26.4 at 2.21): does NOT fit yet; B8 eager 8.7 GB (18.9);
B8-fast + MTP 10.2 GB (32.6 code / 18.1 prose).** Settings: MTP_C 32 (= 64 in speed), FRSPEC on (+4 %), spec 3:3:0.
Trim queued for B12 + MTP (~-1.5 GB): MLX cache cap 64 MB, smaller prefill chunks, sidecar C_l 20 -> 12, then C_m 51 -> 46.
Waits (IDEAS I8, model on the 125B traces): in the fast engine waits are ~30 % of an MTP pass (oracle bound 1.41-1.49x).
Prefetch issued WITH the demand reads loses (-6 % MTP, matches Breakthrough's 18:10 loss: NVMe has no priority);
demand-first + a global staging ring + the corrected d=1 forecast: +16-19 % (B12 57.8 -> 67.8, B8-class 50.7 -> 60.3 tok/s
modelled). Hardware A/B queued at Breakthrough after the trim; design note to MoonShots' loader ring.

## CAVEAT (01:15): every RAM figure above and below is MLX's peak ACTIVE memory, which excludes MLX's buffer cache (real
process memory). From the trim run on, configurations are judged on the process max RSS (Breakthrough prints active, cache
and RSS); a cache cap (B12_CACHE_MB) is one of the trim levers. Expect RSS above the active figures until then.

## 01:05: HC8 default + MTP on v3 (Breakthrough 00:55-00:58)
HC mixers 8-bit are free in quality and not slower: B12 24.2 / 22.2 tok/s at 10.7 / 10.4 GB; B8 18.9 / 16.0 at 7.7 / 7.5;
B8-fast 20.1 / 18.3 at 7.7 / 7.5 (B12_HC=8 default). MTP 3:3:0 on the v3 copy: **B12 code 36.2 tok/s (3.37 tok/pass, +52 %),
prose 29.5 (+30 %)**; B8-fast code 27.5 (+39 %). MTP adds ~2 GB of peak (B12 + MTP 13.3 GB code / 12.0 prose; B8-fast + MTP
10.4): trim next (HC8, MTP_C 32, the FRSPEC head, temporaries). B12 + MTP equals the exact 24 GB engine's 36.1 tok/s on code
at about half the RAM. Eval suite to run the morning verdicts with MTP on.

## Candidates measured (Breakthrough 23:14-23:28, v3 copy, eager, M5 Pro)
| config | residency | peak GB | code / prose tok/s | GSM8K-20 | hard-text KLD |
|---|---|---|---|---|---|
| B12 (frugal 10 % on v3, bf16 mixers) | 10 % | 11.3 / 10.6 | 23.8 / 22.7 | 19/20 = exact | 0.1009 |
| B8 (frugal 5 % + cold ranks 2-3 half-late) | 5 % | 8.4 / 7.7 | 16.6 / 16.1 | 19/20 = exact | 0.0755 |
| frugal 5 % on v3 (no late) | 5 % | 8.4 / 7.7 | 22.8 / 20.7 | - | 0.1740 |
| B8-fast (frugal 5 % + wait ranks 0-2) | 5 % | 8.4 / 7.7 | 19.8 / 18.3 | 19/20 = exact | 0.1073 (~ frugal 10 %) |
B8 easy text 0.0198 vs frugal 5 % 0.0507 and frugal 10 % 0.0349: B8 beats frugal 10 % on BOTH kinds of text at half the
cache. HC mixers 8-bit are FREE (KLD +0.0000 hard, +0.0001 easy) and save 0.62 GB (B12 ~10.7 GB, B8 ~7.1-7.8 GB); speed
rows next. Eval suite order (00:20): B12 -> B8 -> B8-fast (B8 first: if it fails, B8-fast would too).
B8's phase B does not hide its 44 late reads/token and churns the 26-slot cache (waited misses 70 -> 88): -22..27 % speed
for its quality. Next 6-8 GB variant: "B8-fast" = wait ranks 0-2 at 5 % (hard KLD 0.1073 ~ frugal 10 %), no phase B.
Both B12 and B8 go to the Eval suite's morning batch (B12 first). Speed roadmap from ~23 tok/s eager: MTP on v3 (~1.5x),
the new engine (1.45-1.5x), a better 125B MTP head on agentic text (TrainLab) -> ~70-90 tok/s at ~11 GB if all land.

## Local healing (TrainLab, 23:00): slow, frugal-only
LoRA r16 + router KD on frugal 5 % (35B): KLD 0.2017 -> 0.1691 at 246K tokens (17 % of the gap); fit (KLD - floor) ~
tokens^-0.135 -> ~0.142 at 1M, ~0.123 at 3M (41 %): no-go at Mac budgets by the pre-registered rule (one recipe check with a
larger effective batch pending). Transfers across residency (C26 0.168 -> 0.144); hurts the exact flow (0.063: a frugal-only
adapter); halves the copy-heavy log's KLD (1.46 -> 0.76), the one place it might still pay. B8 (training-free) already
reaches frugal-10 % quality at 5 %.

## Agentic turns are sensitive even to numerics (Chat1000, 35B; CORRECTED 01:34 after the MLX gather_qmm bug)
The first runs prefilled 9K-token turns in one forward, which hit MLX 0.32.2's sorted gather_qmm bug (rows unwritten above
32,767 routed rows); re-measured with 2K-token chunked prefill (Chat1000 batch 6, REPORT.md section 5). Turns identical to
exact out of 9: numeric floor (every MoE output x 1.0078) 5/9 (the floor alone changes code on 4/9: narrower grep scope,
other struct fields); relaxed acceptance ratio 0.3 3/9 (a different command, an Edit old_string drifting to another
excerpt, different file contents); span-aware acceptance 8/9 (its one change is a prose reply; tool calls intact). The
"port 8080 -> 1234" and "Read -> Bash cat" examples are WITHDRAWN (corrupted contexts). So the agentic decode row must be
judged against the same-weights floor on the same turns, and any non-exact engine will move about half of agentic turns
mildly. All 125B results (MTPLX's chunked, guarded prefill) and short-prompt rows are unaffected.
## Latency budget, measured (MoonShots 17:16, data/day15/latency.txt, engine/PLAN.md)
Command-buffer end -> CPU handler 50 us (p90 57); cold MSB-3 expert reads (2.16 MB, F_NOCACHE): QD1 0.30 ms, QD4 0.80, QD6
1.13, QD10 1.81 (~12 GB/s). The new engine's half-layer window (attention half) is only ~0.3 ms at M=4 / ~0.2 at M=1, so
half-layer deferral hides ~70 % of one cold read and ~25 % of a 6-read batch. Consequence: the faster the engine, the
shorter any deferral window in TIME while SSD latency is fixed, so latency must be hidden by reads issued a layer or more
EARLIER. **Measured 18:10 (Breakthrough, M5 Pro, 10 %): precise look-ahead prefetch LOSES for frugal at equal RAM**
(eager code 22.7 vs 26.2 tok/s; MTP 37.6 vs 39.2): spare slots come out of the cache and the corrected d=1 predictor
rarely names this layer's ranks 0-1 in time. What wins: **cold rank 1 half a layer late** (+6.7 % code / +8.8 % prose
eager; waited misses 27 -> 11; ~45-50 % of each moved read hidden today, ~70 % expected in the new engine; quality
screen running; on the 35B it cost +3 % KLD). Next latency candidate: pre-routed frugal (TrainLab zero-shot). The new
engine stays the main speed lever.

## Mac mini (16 GB, SSD ~3.2 GB/s)
Same stack: diet + MSB-3 cache give ~15 % residency in ~12 GB usable; frugal ~22 reads/token = ~47 MB -> ~68 tok/s SSD
ceiling with perfect hiding; the mini's GPU then binds: expect ~35-50 tok/s on the new engine.

## 4 GB version
Bounded by the RAM floor (non-routed weights 3.89 GB): needs a smaller or distilled model; with local-only training,
TrainLab's 35B curves decide whether any healing is affordable (1e7 tokens of the 125B ~ 15-30 h of the whole Mac).

## 1,000 tok/s one chat (Chat1000)
Route P (one answer written as many parallel sections) is the only physically open route: sections of one answer cost
1.06-1.23x one chain's expert union (~1,200-1,440 tok/s byte ceiling). Under the task-result bar it must give similar
answers; compute (~19 TFLOP/s at 128 rows) is the next wall.

## Deprioritised for the small-RAM line
Strict-gate levers (near-tie re-routing, dh_w4 at K=4) drop: frugal is already the operating point and re-routing does
nothing for frugal. They stay in IDEAS.md for a possible high-fidelity mode.

22:20 LINE 2 STATE (Fit12 F1-F3 / DF1-2 / P1 / J1; Breakthrough; Eval suite): B12LL (B12 + LATE3=2,10) = 5/5 PASS on BOTH runs (same
numbers): the first <12 GB setup to pass every rule. Under MTP it is v3's EXACT routing at 10 % residency: the v3 model in 11.3 GB
(needs BIGMOE_PF_GROUP=64; 12.85 GB without). Speed exact: 19-21 / 12-13 tok/s code / prose; per pass 150-185 ms, 71-77 % loader
stalls, BYTES bind (~80 %; 1.1-1.5 GB per pass at 8.6-11.1 GB/s). Exact ceiling at 12 GB with k=3: ~26-29 tok/s; ~39 with full
overlap at the SSD ceiling. Levers measured: async +4 % (kept); budget b16 +32-54 % (non-exact; verdict running, half done);
width policies, JIT, LFU, KC 23: no gain; DFlash2 at 12 GB +14-23 % over b16 (PARKED); more slots C_m 59 -8 % misses +7 % but
12.17 GB, so the exact lever is RAM (~1.0 GB non-MLX footprint under study, Fit12 M1). Prefill: 257-275 tok/s within 12 GB;
long-context memory, not prefill code, binds agentic prompts (8K faster modes break 12 GB).