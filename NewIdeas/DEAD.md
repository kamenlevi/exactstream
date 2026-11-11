# DEAD (NewIdeas: killed ideas, the test that killed them, and the numbers)

## Trainable dataflow at Mac-scale budgets: healing frugal by local LoRA (killed 01:00, TrainLab REPORT.md)
35B frugal 5 %, LoRA r16 on GDN/attention/shared + router, KD to the online teacher. lr 1e-4 / 2K tokens per step: 0.2017 ->
0.1691 at 246K; lr 3e-5 / 8K per update: 0.1751 (131K) -> 0.1651 (262K) -> 0.1649 (438K), a PLATEAU at ~19 % of the gap
(the low-rank correction saturates: the deletion-type pattern). Transfers across residency (10 %: 0.168 -> 0.141), hurts
the exact flow (0.040-0.063: a frugal-only adapter), halves the copy-heavy log's KLD (1.46 -> 0.65). The training-free B8
(half-late cold ranks 2-3) already beats it. Training stays alive only for the 125B MTP draft head (TrainLab turn 6).

## Trained top-2 pre-router estimator for the 125B (killed 22:40, TrainLab feasibility capture + CPU fits)
44.6K tokens, 30 docs, BigMoE's engine at C25; layers 8/16/24/32/39, held out by document. Top-2 precision: tap 0.527,
full-rank ridge on x_{L-1} 0.683 (L32 0.760, L39 0.516), + routing history 0.689, MLP + history 0.609 (overfits); true rank
0 in the predicted top-2 0.646 -> 0.805. Decode traces give less (0.61-0.62 with MoEPrefetch's correction). Signal saturates
near ~0.7: the routing-prediction line (pre-routing, prefetch for frugal) is information-limited on the 125B and closed.

## Zero-shot switch caps (sticky routing without training) on the 125B (killed 22:00, expA traces, CPU)
At most M new experts per token per layer (extra new ones swapped for the previous token's best): C=51 LRU hit 0.676 ->
0.696 / 0.712 / 0.723 at M = 6 / 4 / 2, full-coverage reads 155.7 -> 145.9 / 138.0 / 132.9, frugal reads unchanged (19.8),
swap energy 0.0025 / 0.0058 / 0.0114 per layer (b25's 0.0081 predicted ~0.02 KLD). The 35B showed cap4 hit 0.60 -> 0.72
(C26) and 0.43 -> 0.60 (C13): again the 35B overstates routing-locality levers. Sticky routing survives only as a TRAINED
lever, and its 125B effect is unproven.

## I7 pre-routed frugal with the current look-ahead predictor, on the 125B (killed 19:45, Breakthrough's paired screen)
125B, every-8 subsets, paired vs frugal 'base': 10 % ifeval+multi 0.1009 -> 0.1273 (+26 %), gsm+code 0.0349 -> 0.0389
(+11 %); 5 % ifeval+multi 0.1740 -> 0.2126 (+22 %). Pre-pick precision: true top-2 61-62 %, top-10 94-95 %; true rank 0
picked 71-79 %, rank 1 46-52 % by band. The 35B gain (0.141 vs 0.168) did not transfer: the 125B's flatter routing makes
its top-2 much harder to predict one layer ahead. LESSON: the 35B proxy overstates routing predictability; check routing
ideas on the 125B traces first. Revival path: a trained top-2 estimator (precision >= ~85 %) - TrainLab task.

## Look-ahead prefetch of ranks 0-1 for frugal (killed 18:10 by Breakthrough's measurement, 125B, 10 %, quiet lock)
Corrected d=1 predictor, 2 per layer into 4 spare slots: eager code 22.7 vs 26.2 tok/s, prose 21.7 vs 25.7; MTP code 37.6
vs 39.2, prose 23.9 vs 28.6; 4 per layer / uncorrected: worse. Waited misses fall at most ~3/token and RISE on MTP prose
(42.7 -> 51.6): the spare slots come out of C and the predictor rarely names this layer's ranks 0-1 in time. (My 15:25
sims assumed free spare capacity; at equal RAM it is not free.)

## I6. Near-tie re-routing toward resident experts as a bytes lever (killed 17:45, 35B batch 4 + energy law)
Every chosen expert read: +0.25 on resident logits = 92.7 reads at KLD 0.030; wait-by-mass m85 = 97.9 reads at 0.028. A
swap displaces 6.7 % of the true top-10 mass per layer and pays the displaced AND the added expert's energy (orthogonal
outputs), ~2.2x the mass rule's energy per saved read (Breakthrough's law); with frugal it does nothing. Published as
Cache-Prior (2412.00099); on this model it buys nothing the mass rule does not.

## Top-16 weight normalisation and static top-k (killed 17:45, 35B batch 4)
Weights p / sum(top-16 p) instead of the model's top-8 normalisation (from 2609.04575's MMLU result): exact 0.118 (vs
0.011), static top-5 0.179 (vs 0.084 native), frugal 0.257 (vs 0.163). Static top-5 / top-4 with native weights: 0.084 /
0.229. MMLU-level claims do not survive a per-token fidelity test on this family.

## Intra-pass sharing bias for verify passes (killed 15:45, 125B expA traces, CPU)
Rows of a 4-row pass prefer experts already chosen by earlier rows (on top of the resident bias): reads/token C=51 104.8 ->
101.7 at +0.25, 85.4 -> 81.6 at +0.5; C=97 58.8 -> 57.7; C=128 42.0 -> 41.3 (-1..-4 %). Near-ties of consecutive tokens
rarely coincide; batch-aware re-routing (XShare, OEA) needs many independent requests, not one chat's drafts.

## Linear on-time stand-in for a late MoE output (killed 14:46, 35B capture, CPU 1 min)
Ridge map from the MoE input x_L to the routed MoE output (cap35, 24.6K rows per layer, fit even / test odd segments):
held-out NMSE 0.97 / 0.86-1.06 / 0.86 at layers 10 / 24 / 38 (mean-only 0.97-0.99; rank 32-512 no better). The routed
output is not linearly predictable from its own input, so a late expert cannot be pre-approximated on time cheaply.

## I5. Lazy exactness: frugal decoding + exact background repair of past tokens' K/V and GDN state (killed 14:41, 35B)
ctx35.py, decode mode, C26 W2, 1620 positions, four streams on the same tokens: frugal token on an EXACT context 0.0642
(top-1 0.919); context repaired every 16 tokens 0.1070 (0.890); frugal everywhere 0.1729 (0.854); floor ~0.010.
Physics kept: ~63 % of frugal's damage lives in the CONTEXT (the K/V and GDN state the tokens leave behind), 37 % in the
current token; the context damage is recency-weighted (a 16-token repair lag recovers only 38 %). Why dead: repair has to
be near-immediate, i.e. reads at exact-streaming cost with a lag, while half-layer deferral (I1-half) reaches 0.038 at
the same bytes with no repair machinery and no lag.

## I4. Routed experts one layer late everywhere, to take the MoE off the dependent chain (killed 12:50, 35B)
defer35.py arm defer1_C0_W0 (every routed expert of layer L computed on x_L and added at the end of layer L+1; shared
expert in place): KLD 0.400, top-1 0.757 (floor 0.0103, frugal drop 0.163). With rank 0 in place (defer1_C0_W1) 0.206.
A layer's routed output feeds the next layer's routing and attention; training-free it cannot be pipelined. ScMoE /
Pipeline-Native get this only by training from scratch.

## I3. Neuron-granular expert reads chosen by a RAM-resident index (killed 2026-10-21 12:05, ~6 min of CPU)
Test: scripts/ana35.py sparsity on MoEPrefetch's probe35 capture (35B, 4-bit, 5 domains, held-out second halves),
layers 5 / 20 / 35, 120 tokens x 8 routed experts each. Neurons ranked by |h_i| ||v_i|| (h = silu(g.x) u.x).
- Per expert, ORACLE: 90 / 96 / 99 % of an expert's output energy needs 31 / 50 / 62 % of its neurons (all 3 layers).
- Whole routed sum, oracle, neurons chosen across all 8 experts by w_e |h_i| ||v_i||: 96 % needs 37.5 % (P90 50 %).
- A real selector (h from a 2-bit copy of gate/up, which alone costs ~44 % of those bytes): 96 % needs 75 % of neurons.
- No static hot set: the neurons that are hottest on average leave 35-41 % of the energy out at a 50 % share.
Why dead: the quality bar (Breakthrough's KLD law, ar4 = ~96 % of routed energy) caps the saving at 2.7x even with a
free perfect oracle, and the cheapest selector halves that while adding RAM and a second dependent read round per
layer; activation is dynamic, so neuron caching (PowerInfer-style) has nothing to hold. Consistent with the published
cutoffs being loose: 2605.08575 reads gate fully and keeps "95 % of baseline accuracy" (Qwen3.5-35B-A3B: -2.7 points
already at 57 % sparsity). Data: data/ana35/results.txt.

## I2. Experts see only a sliver of input space, so store them projected (killed 2026-10-21 12:00, ~1 min of CPU)
Test: scripts/ana35.py subspace, layer 20 of the 35B (probe35, 155 experts with >= 64 fit rows, median 193).
- Per-expert basis from the expert's own routed fit inputs: held-out routed inputs keep 60-79 % of their energy
  OUTSIDE it (r = 16..193); the expert's output NMSE on the projected input is 0.81-0.98.
- One layer-wide basis: input energy outside 0.59 / 0.30 / 0.14 at r = 256 / 1024 / 1536 of 2048, expert output NMSE
  0.87 / 0.47 / 0.22; experts amplify exactly the low-variance directions a projection throws away.
Why dead: the routed inputs are high-dimensional (RMS-normalised residuals), so no input-side low-rank store can
approximate an expert; together with orthogonal expert outputs (Breakthrough) and dynamic neurons (I3), an expert's
on-distribution function is about as incompressible as its weights. Bytes per expert can only fall through bits
(RD-optimal codes, ~1.4x), not through structure.

## B12-R: span-aware relaxed acceptance on prose for the 12 GB line (killed 10-22 02:48, Chat1000 REPORT.md section 7)
Exact inside code / tool calls, ratio 0.3 on prose, stock 125B MTP k=3 at the 0.50 setting, 256 tokens, the Eval suite's
IFEval decode items 1-30 + the 24 tf_multi prompts: prose tokens/pass IFEval 2.56 -> 2.83 (+10.5 %), multilingual 2.46 ->
2.73 (+11.0 %), below the +15 % line; answers identical to exact 2/30 and 0/24, first divergence at token ~13 (median).
Why dead: with 3-token drafts a pass is capped at 4 tokens, so relaxing acceptance has little room, while every relaxed
accept rewrites the rest of the answer (the opposite of "similar answers"). Prose speed for B has to come from the drafter
(TrainLab's head retrain) or the engine, not from loosening verification.

## 125B MTP head fine-tune, DeepSeek-style depth training on Claude Code text (killed 10-22 03:22, TrainLab turn 9)
MTPLX's own head, fp32 projections, depth 4, lr 1e-5, 15 Claude Code turns (131K positions); training-window KL per depth
fell ~45 %, but on Chat1000's 12 held-out turns (125B's own greedy text, exactly paired): tau16 4.04 -> 3.85 (-4.6 %, 90 %
CI -8.4..-0.6), no class better (edit -8.0 %, write -6.2 %). Step acceptance d1 0.846 -> 0.858 (bash +5.7 pt) but d2-d4
worse by 1-2 pt; depths 11-15 untrained and worse (edit copy runs). Why dead in this form: depth j trained on a fresh cache
of depth-(j-1) outputs, while the recursive head at inference attends to the true-pair history plus its own chain, so the
optimised context is the wrong one; and Claude-written file contents are off-policy (write worse at every depth). A
recipe that could work: EAGLE-3 training-time test INSIDE MTPLX's QSA attention (true-pair + own-chain keys, all 15
depths) on the 125B's OWN continuations (FastMTP 2509.18362 used ~1e8 self-distilled tokens; this Mac trains ~0.1M per
45-min turn): a multi-night local proposal for Kamen, not a tonight job.
General text confirms it (TrainLab 03:30): the Eval suite's tf_gen (the 125B's own continuations, 22 items) tau16 -7.5 %
(90 % CI -9.9..-5.2): GSM8K -15.3 %, multilingual -8.2 %, code -7.4 %, IFEval -1.7 %; stored at 8 bits -8.8 %. Same
signature: depth 1 up (ALL 0.785 -> 0.797), deeper steps down.

## I8 demand-first staging prefetch on the EAGER B12 engine (falsified 10-22 03:49, Breakthrough's A/B slot)
Pre-registered: arm 1 (C=47 + 4 spares, MoEPrefetch-corrected d=1 k=2 per row, issued after the layer's demand reads, whole
records) +11-14 % on B12 + MTP code. Measured: code 37.2 -> 33.0 tok/s end to end (tokens/pass 3.12 -> 2.78: prefetching
changes residency, hence frugal's outputs and the drafts' acceptance), per-pass time 83.9 -> 84.2 ms (+0 %); prose 87.4 ->
82.9 ms per pass (-5 %). Counters: 210 issued / 125 used per pass, 99 % of the used ones on time. Why the model failed: it
treats a waited read as pure SSD time on the critical path, with issuing, bookkeeping and promotion free; on the eager
engine the pass did not get shorter even though the prefetches landed, so the time is elsewhere (CPU-side issue / promote
costs, spare slots, the residency side effect). Kept only as a question for MoonShots' M2: profile per-layer wait vs
compute in the new engine before building any loader ring. The capture and model files stay (data/stage/, samelayer/).

## D1k: predicted-union read-ahead for small-call prefill (falsified 10-22 05:50, Breakthrough)
Corrected d=1 top-10 forecast, thread-pool F_RDADVISE advisories for layer L+1's predicted union (exact math). Mechanism works
(advised experts read in 5-9 ms vs 23 ms cold), but the union read already runs at the SSD ceiling (177 tokens: 2.51 s of
reads at 13 GB/s ~ 232 experts/layer), so the extra bytes cost directly: 53 tokens -14 %, 177 tokens +40 %, 326 tokens +48 %
(~10-15 % over-prediction plus double reads of records still in flight). My union_forecast.py estimate (~3-4 s -> ~2.5 s)
assumed the SSD idles between layers; it does not. Exact floor of a small call = its union bytes / ~13 GB/s; below it only
fewer bytes help (skip both-part residents; non-exact frugal prefill). Look-ahead stays opt-in (B12_PF_LA=1), off by default.

## 2-bit-heavy expert tails for the exact main track (failed the formal gate 10-22 06:13, MoonShots data/kl125/screen.txt)
Per-part prefill KLD vs the 4-bit reference on the Eval suite's own-output tf sets (138 sequences, 34,457 positions); targets
1.10 x ar4 = code 0.00253 / gsm 0.00449 / ifeval 0.0204 / multi 0.0206: 4433332222 0.00267 / 0.00433 / 0.0215 / 0.0186 (over on
code and ifeval); 4422222222 0.0067 / 0.0111 / 0.0500 / 0.0495; 4322222222 0.0074 / 0.0123 / 0.0552 / 0.0553; 4222222222 0.0098 /
0.0172 / 0.0750 / 0.0768 (2.4-3.9x over every part). GSM8K was blind to it (4422222222 scored 49/50). Why dead: the stall-free
"bytes per miss" lever (1.4x on code with zero overlap) needs 2-bit on ranks the quality bar cannot give up; only last-rank
2-bit might pass (~5 % of expert bytes, not worth a screen); an activation-aware refit closes at most ~1.2x of a 2.5x gap.
Lesson: GSM8K accuracy is not a quality screen for weight formats; use the per-part KLD first.

## Frugal prefill of short tool outputs on B12 (killed 10-22 ~07:40, Chat1000 REPORT.md section 10)
Exact prefill up to the last <tool_response>, the rest (83-731 tokens; 3,251 on bash17) as 1-row frugal decode steps; 9 real
agentic turns, 64 decoded tokens: identical to exact 2/9 vs the numeric floor 4/9, and frugal adds severe changes: an Edit's
old_string merged two Swift lines (the Edit would no longer match the file), a different command, a Bash call instead of a text
answer from token 0. Why dead: copy-heavy tool outputs are read back verbatim later, so any lost precision in their context
breaks tool calls. The short-prompt prefill lever needs an exact method (bytes), not a cheaper approximation.

## Layer-local trained sticky routing (killed 10-22 ~11:00, TrainLab sticky35.py on the 35B)
ReMoE's router terms without the LM loss (KL to the original router + reuse + smoothness), trained layer-locally (1,212 steps x
1K tokens; betas 0.05-0.44 + a reuse-only control), then evaluated end to end: all 15 trained points have MORE KLD than the
zero-shot caps at equal U16 (0.26-0.93x of the caps' quality; beta 0.44: 0.40 at U16 31 vs the caps' 0.135). No compounding
(trained points 1.01-1.35x over the zero-shot error law). Why dead: a small router KL (0.0045 nats) still re-weights every
token's top-k, so the per-layer MoE output error at equal union is ~3x the caps'; KL-to-router + reuse is the wrong local
objective. The union lever comes training-free instead: the per-pass union budget (batch-aware routing, XShare / OEA) beats the
caps 1.7-2.5x on the 35B (16 rows: U16 32 = -39 % experts at KLD 0.069; 4 rows: U4 16 = -25 % at 0.046).

## dh_w4 'fast mode' on the deployed v3 24 GB engine (closed 10-22 ~11:35, MoonShots)
Quality passed (all 30 IFEval sequences 0.0140 vs target 0.0204) and the multi-row build is bit-exact (dh10 == v3, forced-late ==
synchronous reference at S = 4 / 8), but decode tok/s on b13 v3 with JIT prefetch: code 34.8 -> 34.9, prose 30.4 -> 30.3, edit
44.6 -> 45.6 (text identical; verify time x0.93-1.04). Why: with JIT prefetch OFF, v3 runs 28.1 and dh 34.6 (+23 %, the model's
number); with JIT ON both run 34.9. Deferral and the JIT prefetcher overlap reads with compute in the same inter-layer windows,
so deferral SUBSTITUTES for prefetch instead of adding to it; the trace replay counted those windows as idle because prefetch
reads are not demand reads. Lesson: count prefetch traffic as busy SSD time; this also qualifies the morning's 'strictly
serial' anatomy. Could still pay where prefetch is ineffective (small RAM, B12 / B8 MTP verify): multi-row phase B is ready.

## DFlash retarget, step 1 (W_c alone) for the 350 road (stopped 10-22 12:23 by kill test 1, Chat1000)
The 35B DFlash drafter (0.4B, frozen) + a trained 26M input projection W_c from the 125B's taps, 361K teacher-forced 125B tokens with
first-divergence labels, 3 epochs: held-out tau16 code 5.28 (ridge) -> 6.07 / 6.64 / 6.83 (bar 7), gsm 3.09 -> 5.20 (bar 5), agentic
2.11 -> 2.06 (not trained on; native MTP 4.00). Why stopped: W_c alone saturates near 7 on code (gains +0.79 / +0.57 / +0.19), and
350 on code now needs ~10-11 (non-exact budget) or 15.8 (exact). Partial value: at ~6.8 a code drafter gives ~1.5x over native
MTP on code in exact mode (~85 tok/s on today's engine in RAM, ~150 with the grouped kernel), but it LOSES on agentic turns.
Revival path (main-account night run, Kamen's call): LoRA on the drafter + on-policy 125B code continuations (perfect-stitch bound
9.45; the drafter reaches 13.4 on its own target); Chat1000 leaves a self-contained runner in its handover.

## SuffixDecoding-style drafting on Kamen's agentic turns (closed 10-22 ~13:25, Chat1000 REPORT 14)
Published method (SuffixDecoding 2411.04975; 4.5x on SWE-Bench agent traces), untested on Kamen's workload until now. A global
index over 6.6M tokens of Kamen's own assistant outputs (rendered in the 125B's format, 32/16/8/4-gram, frequency-greedy; 17 whole
source sessions held out) + per-request prompt lookup, blocks <= 15 simulated against the 125B's own continuations of the 12 agentic
turns: tau16 native MTP 4.00; prompt lookup only 1.39; global index only 1.30 (drafts in 19 % of blocks); lookup-else-MTP 4.08;
suffix-else-MTP 4.03; an ORACLE best of the three per block 4.22 (bar 5.2). Why dead: the 125B rarely continues with existing
text, and its own MTP already predicts what recurs (tool-call syntax, idioms). Caveat: the index held Claude's outputs, not the
125B's; a deployed index would fill with the 125B's own. With this every drafter road for agentic text is measured and closed for
350; agentic exact speed stays at native MTP (+ lookup on edits).

## Bit-identical "dequant-once" multi-row expert kernel (built and measured 10-22 ~14:40, MoonShots wide/mr3c.py)
Groups (expert, tier) pairs in chunks of <= RM rows, loads the code bytes and computes the dequant terms once per weight row,
then MLX's exact qdot per pair: bit-identical to 3c at S = 4 / 8 / 16 / 32 on three layers, but 1.6-2x SLOWER (S=4: 0.43-0.66 vs 0.266
ms; S=16: 1.55-2.10 vs 0.969). Why dead: on this GPU 3c is bound by per-pair dequant + dot work, and the restructure (2 output rows
per threadgroup, unfused combine, pair indirection) costs more than re-unpacking saves; more rows per chunk cost registers and
occupancy. What is left: an MMA-tile kernel (dequantize a tile into threadgroup memory, simdgroup-matrix FMA over the expert's rows),
which changes the summation order (not bit-identical; MLX's sorted qmm works that way and was also slower at these row counts).
The read-once bound itself is small at today's 4-row passes: <= 1.06-1.08x per in-RAM pass, ~2-3 % at 24 GB streamed.

## Loop-guard arming by repetition share (R2), 10-22 16:00 (NewIdeas)
Idea: arm the repetition guard only when >= 50 % of the last 512 unmasked tokens are 12-grams recurring >= 4x (Welleck et
al. 2020's seq-rep-n as the arming test), on top of masking LaTeX math. On 1,206 stored rows it touched only 3 (vs 13 for
math masking alone). Killed by the loose-loop test (rg_replay.py --loose): a connective sentence recurring with >= 20 fresh
tokens between cycles, the pattern MTPLX's guard exists for, never arms. Math masking alone (R1) keeps the shipped recall
and is the recommendation (GAPS.md G).

## 4-bit tier by router weight instead of by rank (equal bytes), 10-22 19:41 (NewIdeas)
Idea (published relative: HOBBIT 2411.01433, precision by gating score; untested here): give the LSB-1 upgrade to the experts
with the largest router weight (global threshold, same mean of 2 per token-layer as plan 4433333333) instead of to ranks 0-1.
Kill test (scripts/wtier_alloc.py, 125B traces, error proxy = sum of w^2 left at 3 bits): weight/rank = 0.964-0.970 on code,
math, chat and Bulgarian (only -3..-4 %; kill line 0.85). Why dead: top-10 weights are flat and similarly shaped across tokens
(mean w by rank 0.21 / 0.15 / 0.12), so rank already picks nearly the same experts as weight. data/wtier/alloc.txt.

## Multi-row late-expert phase B for B8 / B12 + LATE3 under MTP (MoonShots b8mr, 10-22 22:17; recorded 23:48)
Pre-registered: verify -15 % per pass. Measured: +16 % on code at 5 % and 10 % (wait 109.8 -> 128.0 ms), tok/s -14..-17 %,
KLD vs wait-all +9..+35 %. Waited reads fell only 17-27 % while total reads rose 4-17 %: one attention block of slack at S=4 cannot
hide ~100-140 late reads, and late reads evict what the next pass needs from a 26-51-slot cache that a 4-row frugal pass already
oversubscribes (80 candidate ids). Keep wait-all verify (FL_MR). ~/MoonShots/data/b8mr/.

> (a quote from the project owner, kept private)
DFlash2 fine-tuned 2 epochs on 35K anchors from 33 of Kamen's sessions, labels = the 125B's own top-32 at each position of
Claude's text (no generation). In-distribution it learned a lot (on-path top-1 depth 7 .515 -> .817), but on the 12 held-out
agentic turns of the 125B's OWN output: tau8 2.86 -> 2.91 vs MTP 3.54. Why dead: the contexts and targets come from Claude's
text, and the 125B's own agentic output differs in style (write turns -11 %); only shared commands / paths transfer (bash +4-21 %).
The remaining published route = train on the 125B's own agentic outputs (projected ~4.0, still short of agentic 350's ~5.1).

## Uncertainty-gated budget (UGB), my 10-22 genuinely new bet, realizable gate, 10-25 06:23 (TrainLab, CPU)
The oracle gate passed the strict rules (b20 at 90 % code coverage, 10-22 21:01), but the realizable gate does not. MTP drafter
depth-1 confidence, per 4-row pass, joined bit-exactly from turn15 x turn18: the best full PASS keeps the budget on 65 % of code
passes (line 70-80 %); the code P99.9 tail fails first, then the ifeval / multi top-1 CIs. At 8-16 rows the passing thresholds
cover only 33-46 % / 16-25 % of code passes. Why dead: the drafter is less sure than the target exactly where the budget hurts,
so a gate it can compute misses the rare large-KLD positions. The same holds for gated 2-bit cold reads (B2, 51-63 %).
~/TrainLab/data/ugb_r4_draft_turn15x18.txt.

## B10 trellis-coded residency (2-bit data-aware expert codes) and B8 forecast eviction (killed Sun 25 Oct 10:34, Fit12)
B10a: TRL-2 (2.06 b, BlockLDLQ) fit on 31K tokens of local technical documents, judged on held-out eval text: 2.49x MSB-3's
output error (all tokens), 2.59x on routed tokens, line 1.5x; in-sample it had been 0.94-1.15x (B3), so the gap is
generalization. 2.40 b mixed 1.87x. Rank 0-1 residual vs W4: k_r=2 0.66 E4 (line 0.2). B10-r (Fit12, 17:12) was the final test: broad,
judge-disjoint calibration (32,768 fit tokens over 7 domains) + H damping chosen on val. TRL-2 was still 1.85x on B, 1.96x routed,
2.29x on an unseen domain (lines 1.5 / 1.5 / 1.8), and 1.74x on held-out text from the calibration mix itself. Domain shift
explains part of the gap, not enough: data-aware 2-bit expert codes are closed. TRL-3 (3.07 b) at 0.50x MSB-3 lives on as T3R. B8: Belady with perfect 2-pass foresight beats LRU by only 4.7-8.2 % at C 128-192 (line 10 %), so no
real forecast can reach the line; unbounded Belady stays -33..-44 %, but that needs foresight no drafter or router gives.

## T3R: TRL-3 trellis base for every expert + 1-bit trellis residual for ranks 0-1 (killed Sun 25 Oct 21:47, Fit12 K1)
Bytes: at 12 GB a 16-row check reads 214 vs v3's 256 MB / token (-16.4 %, line 20 %); 24 GB -29 %. At 12 GB one check's
expert union is larger than the cache, so the ~16 % extra slots buy nothing and only the record sizes (0.87x / 0.85x) count.
Quality (K2 passed): ranks 0-1 within 0.475 E4 of W4, ranks 2-9 at 0.5x MSB-3, so vs the bf16 weights it trades top-2
accuracy (+48 %) for ranks 2-9 accuracy (-36 %). Knowledge kept: a 1-bit trellis residual removes ~69 % of a trellis base's
error. Reopen only for a non-exact 24 GB mode with disk for a second format. Data: ~/Fit12/data/t3r/.

## ROWS-FREE: wider chat drafts once the check is CPU-bound (killed Sun 8 Nov 15:32, NewIdeas, CPU replay of logs)
Idea: Leap's CHS5 note says the 1-2-row chat check is CPU-bound (bld ~31.5 of ~33 ms). If so, extra verify rows ride on
GPU slack and cost ~0, so longer chains or depth-1 trees (TREE_WALL priced them at ~3-4 ms per row) would pay now.
Kill test (pre-registered before the fit): KILL if a verify row costs >= 1.0 ms in the current check, which with the
head's 2.6 ms per drafted row puts the 4th row below the cycle's average yield. Data: BigMoE CHS5 per-check logs
(data/r57/{zsgcb,zsgc,zs}_256_checks.json, 12 IFEval, C 256, ~710 checks each, first check per key dropped);
scripts/chat/rows_free.py, data/chat/rows_free.txt.
Result [K]: ck = 25.5 + 1.96 x rows + 0.161 x misses (zsgcb, resid sd 1.7 ms); zsgc 25.7 + 1.98 / 0.161; zs 27.7 +
1.95 / 0.168. Medians by rows: 1 31.5 (26 misses; drafter-unsure positions), 2 30.6, 3 33.7, 4 36.9, 5 39.7, 6+ 44.5
ms. bld grows with rows at the same ~2 ms (the per-layer syncs hold GPU time), so the build is not pure CPU.
Why dead: a row costs ~2.0 (verify) + 2.6 (draft) + misses ~= 4.6-5 ms. The chain's survival to depth 4 is ~0.15
tokens (head b5 depth hits, sr2.txt), i.e. ~0.03 tokens per ms against the cycle's 2.35 / 43.2 = 0.054. Depth 2 pays
(~0.10), depth 3 is break-even (~0.06): today's cut 0.3 (~3 rows) already sits at the optimum. A depth-1 top-2 branch
adds ~0.09 tokens [G] for ~2 ms = ~0.045, also below average. What remains is the fixed ~25.5 ms per check, which is
Leap's ENC2 / encode lever, and acceptance (F2a).

## PAR-CHAT: parallel answer streams (Skeleton-of-Thought style) as a chat fast mode (killed Sun 8 Nov 15:40, NewIdeas, CPU)
Idea (IDEAS.md, pre-registered 15:38, 1cac993): CHS5's chat check is 25.5 ms fixed + 1.96 ms per row, so B independent
streams of one answer share the fixed part. Priced per stream at 53.8 / 44.1 / 37.4 / 32.4 tok/s with 1-4 streams
(aggregate 2.41x at 4); 150 was out of reach even before the test (it needs ~8 streams and >= ~90 % parallel text).
Kill test: the 125B's own reference answers (NAX6-N AR; ref2tnax IFEval 100 + ref2tnaxn.r6 IFBench 150), parallel
groups = >= 3 sibling list items / sections (STRICT), plus >= 3 plain paragraphs (LOOSE); dynamic assignment of
branches to <= 4 streams; skeleton 8 tokens per branch + 40 ms per start event. scripts/chat/par_chat.py,
data/chat/par_chat.txt.
Result [K, CPU]: STRICT finds a group in 21 / 100 IFEval and 33 / 150 IFBench answers (14-16 % of characters):
x1.091 / x1.078, pooled x1.083. LOOSE (stanzas and essay paragraphs count as parallel, which a real expansion would
hurt) x1.295 / x1.212, pooled x1.242. Pre-registered KILL line: LOOSE < 1.25. KILL, and the generous LOOSE number
is an upper bound: it ignores quality (independent stanzas, paragraphs that build on each other) and the multi-stream
engine change (per-stream GDN state + KV in one check) is days. Spot check: only 2 / 100 answers with >= 3 item / head
lines got no STRICT group (inline bold "First," paragraphs, an all-caps resume), and LOOSE takes both.
Why: benchmark chat answers are short (IFEval median 218 tokens) and mostly prose; lists / sections hold ~15 % of the
text, so Amdahl caps the gain near x1.1 however cheap the streams are.
