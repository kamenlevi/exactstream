# GOAL A screens from MoonShots' list (orchestrator 01:0x), pre-registered Sun 15 Nov 00:5x, before any of them runs

Source: ~/FableGod/deepdraft/LEVERS_A_MOONSHOTS.md, levers 2 and 3. Their lines are kept; the sharpenings below fix
details the text leaves open. All screens are eval-only (s30 unchanged), so no training budget is involved.

## Lever 2, KEYFRAMES (sb/trunc.py --keyframes)
- Composer = the 2B over the K = 64 tokens ending at the anchor plus the drafted (= true) tokens: T2's K = 64 state.
- At depth j with j % k == 0 (j = k, 2k, ...; MoonShots' convention), the stand-in becomes the composer's state at row
  s + j - 1; at every other depth the head carries its own output. k = 1 is T2's every-row run, so the head carries at
  most k - 1 steps.
- Arms: k = 1, 4, 8, 16, and the chain, on T2's anchors (50 MBPP answers, every 3rd anchor). A second set refreshes from
  the TRUE state (the full prefix) at the same depths: the k-curve with a perfect composer.
- LINE (MoonShots'): PASS if k = 8 keeps >= 70 % of its own every-row gain at MBPP 23-30 (>= chain + 0.7 x (k1 - chain);
  73.1 with T2's numbers). If only k = 4 passes, cost the composer at 1 / 4. KILL if k = 4 < 65.
- Cost: a composer block pass over 64 + k tokens (2B, 24 layers, no lm_head) per k drafted tokens, measured on the GPU,
  shown as ms per drafted token beside the head's own 2.54.

## Lever 3, kNN STATE MEMORY (sb/knn.py)
- Index: every row of the Magicoder captures (data/mc 540 + data/mc2 500 answers, prompt and answer rows), keyed by a
  64-bit rolling hash of the last n tokens ending at the row, n = 16 / 12 / 8 / 6 / 4; value = that row's true state W
  (post-norm, fp16, memory-mapped). No MBPP or HumanEval rows.
- At depth j for anchor s, the n-gram ends at row s + j - 1 (drafted tokens = the true tokens on the scored chain). The
  longest n with a hit, n >= n_min, gives the stand-in; no hit keeps the carried one.
- Arms: (a) one neighbour, the most recent, n >= 6; (b) the mean of up to 8 most recent, n >= 6; (c) one neighbour,
  n >= 12. Plus the chain. MBPP-50, all anchors.
- Logged: hit rate per band; accuracy on hit cells against the chain on the same cells.
- LINE (MoonShots'): PASS if MBPP-50 23-30 >= +2 pt over the chain (sign p < 0.05), or hit-cell accuracy >= chain + 10
  pt at a >= 25 % hit rate. If (a) beats (b) by >= 3 pt, "sample, don't average" holds.
- Cost: the CPU lookup (hashes + 1-8 stored rows) per drafted token, measured.

## For Leap: deep read-out to depth 63 (no pass line)
- s30 chain and TRUE-state read-outs walked to depth 63 on MBPP-50 and HE-30 (s30 was trained 30 deep, so depths 31-63
  are extrapolation for the chain). Bands 31-38 / 39-46 / 47-54 / 55-63, and kept per check walked with a cap of 32, 48
  and 64 rows.

## Lever 2 result (Sun 15 Nov 01:02; 50 MBPP answers, every 3rd anchor, 3,857 anchors; results/t2_trunc_keyframes.json)
| refresh at depths j % k == 0 | 1-2 | 3-6 | 7-10 | 11-14 | 15-22 | 23-30 | share of the every-row gain | composer ms / drafted token |
|---|---|---|---|---|---|---|---|---|
| chain (no refresh) | 78.08 | 68.14 | 63.81 | 61.51 | 58.90 | 55.42 | 0 % | 0 |
| K = 64 composer, k = 1 | 79.44 | 79.12 | 79.54 | 79.63 | 80.05 | 80.61 | 100 % | 39.1 |
| K = 64 composer, k = 4 | 78.08 | 73.74 | 73.30 | 72.09 | 72.01 | 71.56 | 64 % | 9.75 |
| K = 64 composer, k = 8 | 78.08 | 68.14 | 73.00 | 64.22 | 66.30 | 65.27 | 39 % | 4.80 |
| K = 64 composer, k = 16 | 78.08 | 68.14 | 63.81 | 61.51 | 65.89 | 56.64 | 5 % | 2.36 |
| true state, k = 4 | 78.08 | 78.08 | 77.80 | 77.31 | 76.40 | 75.84 | 63 % | (oracle) |
| true state, k = 8 | 78.08 | 68.14 | 77.41 | 64.80 | 69.13 | 68.35 | 40 % | (oracle) |
- **Verdict: neither PASS nor KILL.** k = 8 reaches 65.27 (PASS needed >= 73.06); k = 4 keeps 64 % (< 70 %); k = 4 is above
  the KILL line of 65 (71.56).
- **The head carries a refreshed state for only ~3 steps.** With a PERFECT composer every 8 (true state), the band after
  each refresh is high and decays within ~4 depths (7-10 77.4 -> 11-14 64.8). The shares of the gain are the same for
  the composer and the true state (64 / 39 / 5 % vs 63 / 40 / 5 %), so the limit is the head's carry, not the composer.
- **Cost on the 2B** (shared GPU, measured): a composer pass over 64 + k tokens is ~38-39 ms whatever k is (weight-bound;
  MLX's 24-layer 2B, no lm_head), i.e. 9.75 ms per drafted token at k = 4 and 4.80 at k = 8, against the head's 2.54. In
  the sandbox the composer IS the target, so its cost here is a full target pass; lever 1's smaller composer is the
  125B-side question.

## Lever 3 result (Sun 15 Nov 01:10; MBPP-50, all anchors; results/knn.json; one yield to a quiet ticket, rerun clean)
Index: 1,040 Magicoder answers, 737,010 rows (3.02 GB fp16, memory-mapped). Lookup 1.5 us per row on the CPU.
| arm | 23-30 vs chain (55.47) | hit rate at 23-30 | hit cells: arm vs chain on the same cells |
|---|---|---|---|
| (a) most recent neighbour, n >= 6 | 55.40 (-0.07, CI -1.02..+1.06, p 0.89) | 17.3 % | 52.75 vs 65.24 (-12.5) |
| (b) mean of up to 8, n >= 6 | 56.44 (+0.97, CI +0.02..+2.06, p 0.11) | 17.3 % | 56.22 vs 65.24 (-9.0) |
| (c) most recent, n >= 12 | 55.67 (+0.21, CI -0.05..+0.53, p 0.10) | 1.2 % | 66.25 vs 66.77 (-0.5) |
- **Verdict: KILL.** No arm reaches +2 pt at p < 0.05, and on hit cells every arm is BELOW the head's own carried state.
- "Sample, don't average" does NOT hold here: (a) is worse than (b), on hit cells by 3.5 pt.
- Reading: a real state from a different context that shares the last 6+ tokens is a worse stand-in than the head's
  carry. The state encodes the context (T2: 16 tokens of this context's own window already give 75.5), and attention over
  the true prefix can't correct a state borrowed from another prefix. Hits are also rare (17 % at n >= 6, 1 % at n >= 12).
- Cost would have been ~free (1.5 us per row plus reading 1-8 rows of 4 KB).

## For Leap: depth-63 read-out (Sun 15 Nov 01:21; results/deep63.json; s30 trained 30 deep, so 31-62 is extrapolation)
| s30 read-out | 23-30 | 31-38 | 39-46 | 47-54 | 55-62 | kept/32 | kept/48 | kept/64 |
|---|---|---|---|---|---|---|---|---|
| MBPP-50 chain | 55.47 | 53.40 | 51.46 | 50.50 | 50.07 | 4.32 | 4.32 | 4.32 |
| MBPP-50 true state | 87.75 | 87.76 | 87.83 | 87.84 | 87.71 | 7.75 | 7.97 | 8.02 |
| HE-30 chain | 68.72 | 66.54 | 64.79 | 63.08 | 61.56 | 6.24 | 6.26 | 6.26 |
| HE-30 true state | 93.26 | 93.08 | 92.88 | 92.70 | 92.50 | 12.23 | 13.30 | 13.81 |
- With today's drafter a 48- or 64-row check keeps no more tokens than a 32-row one (MBPP 4.32 / 4.32 / 4.32; HE 6.24 /
  6.26 / 6.26): the chain almost never strings 31 right guesses.
- Even with a perfect fresh state: +3 % (MBPP, 7.75 -> 8.02) and +13 % (HE, 12.23 -> 13.81) at 64 rows. A 48-64-row check
  pays only if its cost grows by less than that over a 32-row check.
- Drafting cost grows linearly: 63 drafted tokens ~ 160 ms at 2.54 ms per token for the 1-layer head.

## Lever 1c, distil the 2B toward the 125B (pre-registered Sun 15 Nov 01:3x, before any run; sb/lever1c.py)
Owner moved to Sandbox98 (WEEK98.md 01:2x, NewIdeas 01:17). Port of MoonShots' ~/MoonShots/week98/lever1c_distill.py
(read-only); outputs stay in ~/Sandbox98.
- Form (MoonShots'): the frozen 2B runs layers 0-22 once per file and caches the input to layer 23 (full attention, so no
  Gated-DeltaNet backward). LoRA r 16, scale 10 on layer 23's 7 linears; Adam lr 1e-4; loss = cross-entropy to the 125B's
  renormalized top-32 (the bank's tk / lp) on the answer rows of the bank's 360 train files (n1112 files t30b never saw;
  95,070 answer rows). Final norm and tied lm_head frozen. 8 min of training.
- Read-out: teacher-forced top-1 agreement with the 125B's runtime top-1 (tk[:, 0]) on HE+ 24, MBPP 100 and val 40
  (the next 40 n1112 files; no overlap with train, checked). Control = LoRA zero; it must reproduce lever7_agree.py's
  94.42 / 85.67 / 87.29 +- 0.1. Read at 1, 2, 4 and 8 min of training: the slope says whether a miss needs more budget.
- Also logged: sure rows (125B margin >= 4 nats) and unsure rows (< 4, where 87 % of the 2B's MBPP misses sit), rows
  gained / lost against the control, and a 95 % bootstrap CI over files.
- LINE (NewIdeas 01:17): PASS if MBPP >= 89.0 and val rises too. Information line for the 98 %-class route: HE+ >= 97.0
  and MBPP >= 91.0. KILL if MBPP < control + 1 pt.
- Cost if it passes: none at draft time beyond the 2B auditor itself (LoRA merges into layer 23's weights; 1.06 GB per
  2B pass, MoonShots' pricing).
- Training head: the loss uses a bf16 dequantized copy of the tied lm_head (the 4-bit path's backward to its input is ~5x
  slower, core.py); every read-out uses the target's own 4-bit head, so the control is exact.

## S5 INJECT, the no-download analog (pre-registered Sun 15 Nov 01:3x, before any run; sb/s5inject.py)
> (a quote from the project owner, kept private)
final norm + tied lm_head (pretrained composition, early exit) and the "big model" is the full 2B.
- Corrector: frozen layers 0-10 (cached per file) -> layer 11 (full attention) with LoRA r 16 -> final norm -> lm_head.
- Arm I adds a zero-init cross-attention after layer 11 (8 heads x 128, RoPE on positions, output projection zero) that
  reads the full 2B's post-norm final states W[u] at prefix positions u <= s only (block-causal: s = the block's anchor).
  It sits after the last corrector layer, so no row reads another row's injection. Arm N: the same without it. A separate
  softmax (not extra keys in layer 11's own attention) keeps arm I = arm N exactly at init.
- Blocks: anchors a0 + off + 30 k (a0 = gen - 2); a row p predicts token p + 1 at lag p - s in 1..30. Lag 1 = the chain's
  depth band 0 (the MTP head reads W[s] and token s + 1 to predict s + 2). Training draws off in 0..29; read-outs use
  off = 0, 10, 20 and band by lag (0-0, 1-2, 3-6, 7-10, 11-14, 15-22, 23-29 in depth terms).
- Both arms: the same LoRA init, the same files and offsets, the same steps (alternating per step); Adam lr 1e-4;
  cross-entropy to the full 2B's renormalized top-32 on answer rows of my Magicoder base 380 (the 2B's own greedy
  answers). 8 min of training, read-outs at 2, 4 and 8 min on MBPP-50 / HE-30 (agreement with the full 2B's top-1).
- LINE (NewIdeas 01:22): PASS if injection adds >= +3 pt (sign test p < 0.05 over files) to the all-lag agreement on
  MBPP-50 -> family (c) is worth a build on the 125B. Information: the 23-29 band (does the prefix help deep, or only
  near?), and the reader ceiling (arm I at lag 1 vs s30's 88.54 at depth 0).
- Cost: the corrector's pass over a 32-row block (12 layers + lm_head, measured) next to the head's 2.54 ms per token.

## CHAIN SIZE (orchestrator 01:3x after Leap's KNEE probe 4be8cf9; pre-registered Sun 15 Nov 01:5x, before any run; sb/chainsize.py)
How small can the chain head be and keep its deep accuracy? Leap: <= ~8-12 MB per step stays in the GPU cache (~1 ms per
30 steps); 16-48 MB reads at ~500 GB/s (~3-5 ms per 30 steps). Today's 2B head per step at 4-bit: 34 MB body (60.8M
params) + 286 MB lm_head (248,320 tokens).
- Arms trained in one process, the same 300 steps x depth 30 (96 rows, pool2, s30o's Adam continued, cosine re-anneal,
  lr 1e-5): ctl (full head), ffn3072 (FFN 1/2), ffn1536 (FFN 1/4). FFN cut: neurons ranked by output energy
  E[a_i^2] ||down_i||^2 on the head's own chain rows (4 pool2 files); down_proj refit by ridge least squares to the full
  FFN's output; Adam moments sliced to the kept neurons.
- Eval-only on each trained arm: 4-bit weights (group 64, post-training), and lm_head shortlists of the 32K / 16K / 8K /
  4K most frequent answer tokens of the training captures (base + pool + mc2; argmax over the list). Also the pruned +
  refit heads before training (1/2, 1/4, 1/8; MBPP-50 only).
- Read-out: chain MBPP-50 / HE-30, bands 0-0 and 23-30, kept/32; paired vs ctl fp32 full-vocab. MB per step at 4-bit per
  arm and list. No checkpoints written (in-process eval).
- LINE (orchestrator's): a <= 48 MB head (better <= 12 MB) that holds MBPP-50 23-30 within ~1 pt of the full head at
  matched steps -> the cache-resident chain is a live GOAL A form; a large drop -> block drafting stays the route.
- Arithmetic first (4-bit, 0.5625 B per weight): full body + 8K list 43.6 MB; FFN 1/2 + 8K 33 MB; FFN 1/4 + 8K 27.7 MB;
  FFN 1/4 + 4K 23 MB. <= 12 MB needs the body itself near ~7 MB (~12M params), below what this native-initialized head
  can be cut to without a narrower architecture (fc 2d->d alone is 4.7 MB, q_proj 4.7 MB).

## Lever 1c result (Sun 15 Nov 01:51; results/lever1c_a.json; one yield at 01:38 before training, rerun clean)
Control (LoRA zero) reproduced lever7_agree.py exactly: HE+ 94.42, MBPP 85.67, val 87.29.
| training (steps, epochs) | HE+ 24 | MBPP 100 | val 40 | MBPP unsure rows |
|---|---|---|---|---|
| control | 94.42 | 85.67 | 87.29 | 57.23 |
| 1 min (450, 1.2) | 94.68 (+0.27) | 86.18 (+0.52, CI +0.14..+0.94) | 88.72 (+1.43, CI +0.88..+2.03) | 58.07 |
| 2 min (840, 2.2) | 94.85 (+0.44) | 85.97 (+0.30) | 88.69 (+1.40) | 57.59 |
| 4 min (1,550, 4.3) | 94.38 (-0.03) | 85.15 (-0.52) | 88.25 (+0.96) | 56.31 |
| 8 min (3,024, 8.4) | 94.32 (-0.09) | 84.86 (-0.81, CI -1.38..-0.31) | 87.71 (+0.42) | 55.74 |
- **Verdict: KILL.** Even the best read (1 min) is MBPP +0.52, under the +1 line and far from PASS (89.0 = +3.3). After 8
  min MBPP is below the control (537 rows gained, 663 lost). Training loss 0.412 -> 0.337 while MBPP falls: it overfits.
- In-distribution it moves (val +1.4 at ~1 epoch), but that barely carries to MBPP (+0.5) and not to HE+ (+0.3-0.4).
- What could make it work (Kamen's rule, a negative is about this setup): the limits here are data (the bank's 360
  train files, 95,070 answer rows; the peak comes at ~1 epoch) and capacity (rank-16 LoRA on the last layer only). A
  real 1c needs ~10x the 125B's answers and deeper adaptation (LoRA through the trunk, with a Gated-DeltaNet backward):
  a build, not a screen. A lower lr alone can't pass the ~1-epoch peak this data allows.
- Drafter form and price: the 2B as a one-pass block auditor, 1.06 GB per pass (~4-9 ms, MoonShots' pricing); the LoRA
  merges into layer 23 at no cost. Since agreement didn't rise, the audit stays flat at the 2B's own ~85.7 / 94.4.
- Writes: 2.6 MB (the LoRA) + JSON.
- Amended 02:0x, before the full run (from the smoke): (1) 200 steps, not 300 (the 3 arms cost ~3.3-4.6 s per step);
  (2) 8-bit weights added (the smoke's 4-bit post-training cost ~7 pt at 23-30 on 2 files); (3) the shortlists come from
  the 2B's tokenizer over Magicoder OSS-Instruct's Python part (decontaminated, 19.6M tokens) + the captures' answers.
  The first lists, from the 417K capture tokens alone, covered only ~89 % of MBPP tokens at 8K.

## CARRY400 confirmation for MTP Head (orchestrator 02:2x; pre-registered Sun 15 Nov 02:3x, before the run; sb/carryconf.py)
MTP Head's 400-step screen (~/TrainLab/REPORT.md "WEEK98 S2 CARRY400"): allrows (no accept-until-fail truncation in the
training KL) vs n1 from the native head, MBPP-50 23-30 +8.58 (48 / 48 windows), HE-30 +5.75; depth 0 -0.38 / -0.09; but
kept/32 4.076 -> 4.066 (MBPP) and 5.79 -> 5.69 (HE), and the strict conditional at depths 1-4 slightly LOWER (MBPP
0.881/0.811/0.775/0.752 -> 0.877/0.804/0.767/0.743). The band gain sits on teacher-forced cells after an earlier miss.
- Confirmation at s30's full recipe, two arms in one process (same slices, order, schedule): n1 (s30's recipe: native,
  fresh bias-corrected Adam, base 1,889 slices of 96 in seed-1114 order, depth 30, lr 1e-5, 20-step warm-up, cosine to
  1e-6, clip 1.0; control check: n1 should land on s30's MBPP-50 23-30 55.47 within ~1 pt) and ar (the same + MTP Head's
  allrows lever, unmodified, by absolute path).
- Lines: MTP Head's: CONFIRMED if MBPP-50 23-30 >= +2.0 pt (p < 0.05) at 1,889 steps. Stop line at step 945 (MBPP-50):
  stop unless ar >= +2.0 at 23-30. Run-time value (information, read first): kept/32 (paired, CI) and the conditional
  at depths 1-8, where accept-until-fail decoding earns its tokens.
- Cost: ~35-60 min of GPU, <= 5 GB; no checkpoints (in-process read-outs); resumes after yields.

## CHAIN SIZE result (Sun 15 Nov 02:38; results/chainsize.json; 200 steps x depth 30, three arms in one process, 39 min)
Paired against ctl fp32 with the full vocabulary (the control at matched steps: MBPP-50 23-30 55.05, s30 55.47). MB per
chain step = body at that precision + the lm_head list at 4-bit. Shortlist coverage of eval answer tokens: MBPP 32K
99.6 / 16K 97.7 / 8K 95.5 / 4K 91.9 %; HE 98.8 / 96.4 / 93.9 / 90.9 %.
| arm | MB/step | MBPP 0-0 | MBPP 23-30 | MBPP kept/32 | HE 0-0 | HE 23-30 | HE kept/32 |
|---|---|---|---|---|---|---|---|
| ctl fp32, full vocab | - | 88.07 | 55.05 | 4.219 | 93.14 | 66.68 | 5.921 |
| ctl 8-bit, full | 350.7 | 88.09 | 55.07 | 4.213 | 93.10 | 66.73 | 5.886 |
| ctl 8-bit, 32K list | 102.4 | 87.88 | 55.02 | 4.184 | 92.06 | 66.04 | 5.639 |
| ctl 8-bit, 16K | 83.5 | 86.44 | 54.39 | 4.046 | 90.14 | 64.98 | 5.246 |
| ctl 4-bit, full | 320.3 | 87.29 | 48.90 | 3.947 | 92.64 | 61.93 | 5.292 |
| ctl 4-bit, 8K | 43.7 | 84.17 | 47.68 | 3.688 | 87.56 | 59.36 | 4.547 |
| FFN 1/2, 8-bit, full | 330.7 | 86.13 | 52.05 | 3.919 | 92.19 | 64.65 | 5.513 |
| FFN 1/2, 4-bit, 8K | 33.1 | 82.42 | 45.00 | 3.451 | 86.74 | 58.03 | 4.347 |
| FFN 1/4, 8-bit, full | 320.6 | 83.92 | 48.40 | 3.555 | 91.28 | 61.74 | 5.036 |
| FFN 1/4, 8-bit, 8K | 44.0 | 81.00 | 47.20 | 3.331 | 86.25 | 58.95 | 4.328 |
| FFN 1/4, 4-bit, 8K | 27.7 | 80.13 | 42.23 | 3.209 | 85.79 | 53.96 | 4.047 |
- **Verdict by the line: not met.** No head at <= 48 MB per step holds MBPP-50 23-30 within ~1 pt. The best one (full
  body 4-bit + 8K list, 43.7 MB) is -7.4 pt at 23-30, kept/32 -12.6 % (MBPP) / -23 % (HE). So block drafting stays the
  route; <= 12 MB is out of reach for this head (the body alone is 34 MB at 4-bit).
- Cost of each cut, kept/32 MBPP / HE: 8-bit body ~free (-0.1 / -0.6 %); 4-bit body (post-training, group 64) -6.4 /
  -10.6 %; static list 32K -0.8 / -4.2 %, 16K -4.0 / -10.7 %, 8K -8.1 / -17.4 %; FFN 1/2 -7.4 / -6.4 %, FFN 1/4 -15.9 /
  -15.0 %. Depth 0 pays too (16K list -1.65 / -2.96), and EvalSuite's 02:3x finding says the first ~15 guesses after a
  check decide kept/32.
- The FFN refit kept the pruned FFN's output close on its own stats rows (relative error 0.2 % at 1/2, 1.2 % at 1/4),
  yet the chain lost 3-6 pt: small per-step errors compound over 30 steps. 200 steps of recovery barely moved it (1/2:
  23-30 51.93 before training, 52.06 after, vs the control's 55.05).
- What could change it (each its own screen): 4-bit with quantization-aware training (8-bit is lossless, so the 4-bit
  loss is noise compounding along the chain); a list of the static top tokens plus the prompt's own tokens (the misses
  look like prompt identifiers; eval-only on s30, minutes); pruning recovered by distillation from the full head over
  the full budget.
- For Leap's P2 option "4-bit + head 8K" (36.7 -> 27.6 ms per 32-step chain on the 125B): on the 2B that form, without
  QAT and with a static 8K list, keeps 12.6 % (MBPP) / 23 % (HE) fewer tokens per check.
- Writes: results/chainsize.json + data/shortlist_counts.npy (2 MB); no checkpoints.

## S5 INJECT analog result (Sun 15 Nov 02:48; results/s5inject.json; 1,976 steps per arm, 8 min)
| MBPP-50 / HE-30, agreement with the full 2B's top-1 | control (LoRA zero) | N (no injection) | I (injection) | I - N |
|---|---|---|---|---|
| all lags | 0.60 / 0.43 | 51.98 / 51.70 | 33.63 / 30.85 | -18.36 (CI -19.99..-16.81) / -20.85 |
| lag 1 (depth band 0) | 0.47 / 0.92 | 55.49 / 58.25 | 34.07 / 38.64 | -21.4 / -19.6 |
| lags 24-30 (23-29) | 0.66 / 0.33 | 51.26 / 50.07 | 33.12 / 29.48 | -18.1 / -20.6 |
| kept run per 30-row block | 0.005 / 0.009 | 1.656 / 1.565 | 0.714 / 0.843 | -0.94 / -0.72 (50 / 50 and 29 / 30 files worse) |
- **Verdict: VOID as a test of family (c)**, not a KILL. By its line it is no PASS. The injection arm is 18-21 pt WORSE,
  even at lag 1, where it reads the full 2B's state one row back. Information can't make lag 1 worse, so this is a
  broken optimization, not an answer about prefix features.
- Two setup faults: (1) the 12-layer early exit is not a competent small LM (0.6 % agreement at init, 52 % after 8 min;
  a real 0.8B would start near 80-85 %), so the screen ran in the wrong regime. (2) The zero-init output projection
  trained by Adam at 1e-4 grows by ~lr per weight per step: after a few hundred steps the injected vector dwarfs the
  residual stream. Flamingo-style tanh gating (a scalar gate at 0) or a far lower adapter lr is the standard fix.
- Cost measured: the 12-layer corrector pass over a 32-row block, 15.7 ms on the 2B (shared GPU).
- The faithful no-download version exists. The BANK holds the 125B's own final state for every row ('wd', 10240 =
  the trunk's widened stream after the last layer, the MTP head's input; BigMoE dd_drift.py). So family (c) can be
  screened on the REAL pair: the competent 2B (85.67 % MBPP agreement with the 125B) plus the 125B's prefix states at
  u <= s, with a gated injection, against the 125B's top-1 per lag.

## S5b INJECT on the REAL pair (pre-registered Sun 15 Nov 02:5x, before any run; sb/s5b.py)
NewIdeas' family (c) without a download: the competent 2B as the small LM, the 125B's own final prefix states as the
injected features (the BANK's 'wd', 10240-d), agreement with the 125B as the read-out.
- Both arms: frozen 2B layers 0-22 (cached), LoRA r 16 (scale 10, Adam lr 1e-4) on layer 23, final norm, tied lm_head;
  cross-entropy to the 125B's renormalized top-32 on the bank's 360 train files (answer rows) = lever 1c's form.
- Arm I adds a cross-attention after layer 23 (8 heads x 128, RoPE on positions, RMSNorm on the 10240-d states) reading
  the 125B's wd at prefix rows u <= s only (block-causal), behind a Flamingo tanh gate (alpha = 0 at the start, so arm I
  starts as arm N exactly); the adapter has its own Adam at 3e-4. Same LoRA init, files, block offsets, steps (the arms
  alternate per step). 8 min of training; read-outs at 1, 2, 4 and 8 min (1c peaked at ~1 epoch).
- Read-out on the bank's HE+ 24, MBPP 100, val 40 (all answer rows), block offsets 0 / 10 / 20, lags 1-30: kept run per
  30-row block and the strict conditional per lag FIRST (orchestrator 02:2x), then agreement by lag band; paired I vs N.
- LINE (NewIdeas 01:22): PASS if injection adds >= +3 pt (sign p < 0.05) to MBPP agreement -> family (c) is worth a build
  on the 125B; kept run beside it. HURTS if <= -1 pt (p < 0.05). Reader ceiling: the 125B head's depth 0 with the true
  state, MBPP 94.0 / HE+ 97.9.
- Cost if it passes: a 2B pass per block plus the cross-attention over the 125B's prefix states (no extra 125B pass: the
  states exist after each check).

## S5b result (Sun 15 Nov 03:14; results/s5b.json): VOID, the gate never opened
- The Flamingo gate ended at tanh(alpha) = +0.0003 after 2,726 steps, so arm I was arm N. The final differences are
  noise: MBPP +0.30 (p 0.11), HE+ -0.16, val -0.05; kept run MBPP 7.965 -> 8.050. With the gate at 0 the adapter gets no
  gradient and stays random; a random adapter gives the gate no consistent push. Chicken and egg.
- Useful reference (the plain 2B, LoRA zero) vs the 125B's top-1, teacher-forced, per 30-row block: agreement HE+ 94.42
  / MBPP 85.67 / val 87.29 (all answer rows); kept run per block HE+ 19.86 / MBPP 8.29 / val 9.95.
- After 8 min (2,726 steps, 7.6 epochs) arm N (= lever 1c) sits at MBPP 84.66 (-1.0 vs the plain 2B): 1c's overfit again.

## S5c = S5b with injections that engage (pre-registered Sun 15 Nov 03:2x, before any run; sb/s5c.py)
- Arms, the same LoRA init / files / offsets / steps: N (LoRA only = 1c); I (cross-attention over the 125B's prefix
  states u <= s, output projection ZERO-init, no gate); A (the 125B's state at the row's own anchor s: RMSNorm -> a ZERO-
  init 10240 -> 2048 projection added before the final norm, i.e. the freshest verified state, as the MTP head reads it).
  Adapters on their own Adam at 2e-5 (S5's analog blew up at 1e-4) with grad clip 1.0. Every read logs |inj| / |h|.
- 9 min of training; read-outs at 3, 6, 9 min. Read-out, lines and sets as S5b (kept run first; PASS if an injection arm
  adds >= +3 pt at p < 0.05 to MBPP agreement vs N).

## S5c result (Sun 15 Nov 03:28; results/s5c.json; 1,792 steps per arm, 5.0 epochs, 9 min)
Both injections engaged (|inj| / |h| at the end: I 0.077, A 0.358). Paired against N (LoRA only = 1c):
| vs the 125B's top-1 | N | I (cross-attention, prefix u <= s) | A (state at the anchor) |
|---|---|---|---|
| MBPP kept run per 30-row block | 7.950 | 8.131 (+0.18, CI -0.10..+0.48) | 7.726 (-0.22, CI -0.54..+0.10) |
| MBPP agreement, all lags | 85.10 | 85.47 (+0.37, CI +0.03..+0.70, p 0.011) | 84.40 (-0.69, CI -1.09..-0.32, p 0.003) |
| MBPP lag 1 / lags 24-30 | 87.11 / 84.47 | 86.50 / 85.12 | 86.34 / 83.94 |
| HE+ kept run / all lags | 19.765 / 94.54 | 20.104 (+0.34) / 94.55 (+0.01) | 19.767 (+0.00) / 94.22 (-0.32) |
| val kept run / all lags | 10.746 / 88.20 | 10.828 (+0.08) / 88.07 (-0.13) | 10.733 (-0.01) / 87.60 (-0.60) |
- **Verdict: NO EFFECT for both** (the line needs +3 pt at p < 0.05). I trends slightly up on MBPP (+0.37 agreement,
  +0.18 kept run); A is slightly down. Reads at 3 and 6 min say the same.
- Not even lag 1 gains, where the row reads the 125B's state one row back (wd[p-1]): I -0.6, A -0.8 vs N. Yet the
  125B's own MTP head turns exactly that state (plus the token at p) into 94.0 % on MBPP. The information is there; an
  adapter added after the 2B's last layer can't compose it with the current token. That takes a transformer layer
  reading both (the 125B's head is one), or an injection INTO the 2B's own layers.
- What could still make family (c) pay (Kamen's rule): a deep injection (the 125B's states as extra keys / values in
  several of the 2B's attention layers, so its own layers compose them; needs a Gated-DeltaNet backward), trained on far
  more of the 125B's answers. 1c, S5b and S5c all hit the same wall: 360 files of the 125B's answers overfit within ~1-2
  epochs. A build, not a screen.
- Writes: results JSON only (the resume state file is removed at the end).

## CHAIN SIZE follow-up: prompt-aware lm_head lists (Sun 15 Nov 03:43; results/listctx.json; s30, eval-only, 139 s)
The drafter's argmax is taken over a static list of the K most frequent answer tokens, plus (+prompt) the window's
prompt tokens, plus (+context) the answer tokens already verified at the anchor (ids[gen .. s + 1]). Extra rows per
window: prompt ~57 (MBPP) / 82 (HE), answer so far up to ~83 / 96.
| s30 chain | MBPP-50 0-0 | 23-30 | kept/32 (vs full) | HE-30 0-0 | 23-30 | kept/32 (vs full) |
|---|---|---|---|---|---|---|
| full vocab (248,320) | 88.04 | 55.47 | 4.319 | 93.58 | 68.72 | 6.239 |
| static 16K | 86.44 | 54.88 | 4.125 (-4.5 %) | 90.60 | 66.84 | 5.513 (-11.6 %) |
| static 8K | 84.94 | 54.18 | 3.950 (-8.5 %) | 88.37 | 65.51 | 5.110 (-18.1 %) |
| 8K + prompt | 86.51 | 55.10 | 4.128 (-4.4 %) | 92.76 | 68.36 | 6.095 (-2.3 %) |
| 8K + context | 87.15 | 55.36 | 4.216 (-2.4 %) | 93.20 | 68.54 | 6.171 (-1.1 %) |
| 16K + context | 87.50 | 55.42 | 4.256 (-1.5 %) | 93.33 | 68.58 | 6.166 (-1.2 %) |
| 4K + context | 86.55 | 55.30 | 4.167 (-3.5 %) | 92.67 | 68.49 | 6.034 (-3.3 %) |
- **The static list's loss is mostly the prompt's own tokens** (identifiers, names): adding them recovers HE from -18 %
  to -2.3 % at 8K, and the verified answer tokens take it to -1.1 % (MBPP -2.4 %).
- Bytes: 8K + context = ~8.4K rows = ~9.7 MB per step at 4-bit, vs 286 MB for the full head (~30x less). The rows change
  only between checks (the prompt plus the answer so far), so the list can be gathered once per check.
- New CHAIN SIZE arithmetic: the head part is solved (~10 MB, <= ~2.5 % kept). The body is the remaining blocker: 34 MB
  at 4-bit (post-training 4-bit costs 6-11 % kept; quantization-aware training is the open lever), 64.6 MB at 8-bit
  (free). 4-bit body (QAT) + 8K context list = ~44 MB per step, inside the <= 48 MB band.
- For the 125B (Leap's P2 "screened head 8K"; Breakthrough's 16K code shortlist in the lossy check): the same trick
  applies. A static list plus the prompt and the answer so far.

## QAT 4-bit chain head (pre-registered Sun 15 Nov 03:5x, before any run; sb/qat.py)
The last blocker for a ~44 MB cache-resident chain (4-bit body 34 MB + 8K context list ~10 MB): post-training 4-bit
costs -6.4 % (MBPP) / -10.6 % (HE) kept/32.
- Two arms from s30o, the same 200 steps x depth 30 (pool2, s30o's Adam continued, cosine re-anneal, lr 1e-5): ctl
  (plain) and qat (every Linear trains through fake 4-bit weights, group 64, straight-through gradient).
- Read-out (in-process): ctl fp32, ctl 4-bit post-training, qat fp32 weights, qat at REAL 4-bit; each with the full
  vocab and with the 8K + context list; chain MBPP-50 / HE-30; kept/32 first, paired vs ctl fp32 full vocab.
- LINE: PASS if qat at 4-bit keeps kept/32 within 1 % of ctl fp32 on MBPP and HE -> a 4-bit body + 8K context list
  (~44 MB per step) is a live cache-resident chain form. KILL if QAT recovers less than half of post-training 4-bit's
  kept loss. Between: more QAT budget is the question.
- Amended 03:4x before the smoke started (orchestrator 03:4x): the bar is the 8-bit body at matched steps, with the SAME
  list. The read-out adds ctl at 8-bit; PASS = qat at 4-bit within 1 % of ctl 8-bit in kept/32 on MBPP and HE (full vocab
  and 8K + context).

## QAT 4-bit chain head result (Sun 15 Nov 04:04; results/qat.json; 200 steps x depth 30 each, 1.47 s/step, 10.4 min)
kept/32 against the bar, the 8-bit body at matched steps with the SAME list (paired, % of the bar):
| head | MBPP full | MBPP 8K + context | HE full | HE 8K + context |
|---|---|---|---|---|
| ctl 8-bit (the bar) | 4.213 | 4.116 | 5.886 | 5.791 |
| ctl 4-bit post-training | 3.947 (-6.29 %) | 3.895 (-5.37 %) | 5.292 (-10.11 %) | 5.250 (-9.36 %) |
| qat at real 4-bit | 4.103 (-2.61 %, CI -4.5..-0.9) | 4.031 (-2.07 %) | 5.544 (-5.81 %) | 5.482 (-5.34 %) |
| qat weights unquantized | 4.143 (-1.66 %) | 4.055 (-1.48 %) | 5.687 (-3.38 %) | 5.635 (-2.71 %) |
- **Verdict: BETWEEN.** QAT recovers 59-61 % of the 4-bit loss on MBPP and 43 % on HE, so no KILL and no PASS (the bar
  needs within 1 % on both). More QAT budget is the open question.
- Part of the remaining gap is the short QAT run itself: the QAT weights read UNquantized already sit -1.7 / -3.4 % below
  ctl, so 200 steps through quantization noise under-train the head. A longer QAT (or QAT over s30's whole budget from
  the native head) is the natural next screen, with group size 32 (0.625 B per weight, body ~38 MB) as a second lever.
- The ~44 MB cache-resident form today (4-bit QAT body + 8K context list): kept/32 4.031 (MBPP) / 5.482 (HE) vs the 8-bit
  full-vocab head's 4.213 / 5.886, i.e. -4.3 / -6.9 %.

## QAT 4-bit chain head at FULL BUDGET, group 32 (pre-registered Sun 15 Nov 06:4x, before any run; sb/qat.py --full)
Asked by the orchestrator at 06:3x (the Mac idle). The night's 200-step QAT was BETWEEN, and its unquantized weights sat
-1.7 / -3.4 % below ctl, so the short run under-trained. This run gives QAT s30's whole budget.
- Two arms from the NATIVE head over s30's own recipe, in one process (the same slices in the same order): the 380 base
  files, one pass of 96-anchor slices (seed 1114) = 1,889 steps x depth 30, lr 1e-5 (20-step warm-up, cosine to 10 %),
  fresh Adam, clip 1.0.
  - ctl = plain training. It re-makes s30, and its read-out is checked against s30's (MBPP-50 4.319 / HE-30 6.239 kept/32,
    full vocab, results/listctx.json).
  - qat = every Linear of the head trains through fake 4-bit weights at GROUP 32 (MLX affine quantizer, straight-through
    gradient).
- Body sizes: 4-bit group 32 = 0.625 B per weight = ~38 MB (group 64 ~34 MB, 8-bit 64.6 MB). With the 8K + context list
  (~9.7 MB): ~48 MB per step.
- Read-out in-process: ctl fp32, ctl 8-bit (THE BAR), ctl 4-bit post-training at group 32 and 64, qat fp32 weights, qat
  at REAL 4-bit group 32. Each with the full vocab and the 8K + context list; kept/32 first, paired against the bar with
  the same list.
  - Sets: MBPP-50 / HE-30 carry the line, as pre-registered. The held-out second halves (MBPP 51-100, HE 31-60) and the
    pooled sets are the confirmation.
- LINE (unchanged): PASS if qat at real 4-bit keeps kept/32 within 1 % of ctl 8-bit (same list, matched steps) on
  MBPP-50 and HE-30, full vocab and 8K + context. Then the ~48 MB form (4-bit group-32 body + 8K context list) is a live
  cache-resident chain.
  - "PASS, not confirmed" if a second half sits below -2 %.
  - KILL if QAT recovers less than half of post-training 4-bit's kept loss (group 32, full vocab) on both sets (the
    night's reading). Otherwise BETWEEN.
- Stop line at matched steps: at step 944 (the half), ctl 8-bit vs qat at real 4-bit on MBPP-50, full vocab. STOP if qat
  sits below -6.3 % (no better than the night's post-training 4-bit), i.e. QAT is not working; otherwise run on.
- Guards (orchestrator 06:3x):
  - A full-length smoke first: the 3 longest training windows, the 2 longest files of each set, the stop line, a state
    save + reload and every read-out form. It measures the lifetime peak footprint (/usr/bin/time -l).
  - The share lock carries that MEM. The start gate needs NO-SWAP >= peak + 8 GB.
  - Kill and stop on swap +1 GB since the start, or NO-SWAP < 8 GB on two reads in a row; SIGTERM saves the state first.
  - MLX cache cleared per window. Yields to quiet tickets and resumes from the saved state.
- Writes: the resume state (~1.5 GB, one file overwritten at steps 500 / 1000 / 1500 and after training, removed at the
  end; ~7 GB written in all with the smoke's), the qat head at real 4-bit (~38 MB, the deliverable form), and
  results/qat_full_g32.json.
- Time: ~46 min of training (1.47 s per two-arm step in the night) + ~13 min of read-outs.

## QAT 4-bit chain head at FULL BUDGET, group 32: result (Sun 15 Nov 07:26; results/qat_full_g32.json; 1,889 steps x depth 30 per arm, 1.03 s per two-arm step, 43 min in all)
kept/32 against the bar, the 8-bit body at matched steps with the SAME list (paired, % of the bar; MBPP-50 / HE-30 carry
the line):
| head | MBPP full | MBPP 8K + context | HE full | HE 8K + context |
|---|---|---|---|---|
| ctl 8-bit (the bar) | 4.350 | 4.242 | 6.269 | 6.185 |
| ctl fp32 (= s30) | 4.319 (-0.72 %) | 4.216 (-0.63 %) | 6.239 (-0.47 %) | 6.171 (-0.23 %) |
| ctl 4-bit g64 post-training | 3.990 (-8.28 %) | 3.924 (-7.49 %) | 5.587 (-10.88 %) | 5.505 (-11.00 %) |
| ctl 4-bit g32 post-training | 4.170 (-4.15 %) | 4.082 (-3.77 %) | 5.856 (-6.59 %) | 5.783 (-6.51 %) |
| qat weights unquantized | 4.200 (-3.44 %) | 4.095 (-3.46 %) | 5.913 (-5.68 %) | 5.830 (-5.75 %) |
| qat at real 4-bit g32 | 4.228 (-2.80 %, CI -4.4..-1.2) | 4.143 (-2.35 %, CI -4.1..-0.7) | 6.128 (-2.25 %, CI -5.6..+0.6) | 6.076 (-1.77 %, CI -5.5..+1.3) |
- **Verdict: BETWEEN.** No PASS: the line needs within 1 % in all four cells, and the qat head misses by 0.8-1.8 points.
  No KILL: QAT recovers 32 % (MBPP) and 66 % (HE) of the group-32 post-training loss (KILL needed under half on both).
- The confirmation agrees.
  - Second halves (MBPP 51-100 / HE 31-60): -2.45 / -2.36 % full vocab, -2.42 / -1.76 % with the 8K + context list.
  - Pooled over 100 / 60 files: -2.63 % (CI -3.8..-1.5) / -2.30 % (CI -4.6..-0.2) full vocab; -2.38 / -1.76 % with the
    list.
- 23-30 band: qat at 4-bit 54.09 / 67.02 vs the bar 55.51 / 68.72 (-1.4 / -1.7 pt); with the list -1.4 / -1.6 pt.
- Against the night (200 steps from s30o, group 64): HE -5.81 -> -2.25 %, MBPP -2.61 -> -2.80 % (flat). Group 32 alone
  halves post-training 4-bit's loss (g64 -8.3 / -10.9 % -> g32 -4.2 / -6.6 %); QAT over the whole budget then takes it
  to -2.8 / -2.3 %.
- Matched steps: at the halfway stop line (step 944) the qat head at 4-bit read level with ctl 8-bit on MBPP-50 (3.935
  vs 3.936, -0.03 %, CI -1.4..+1.5); by the end it sat -2.8 %. The training-loss gap held at ~0.02 in both halves (no
  jump after the stop line). So either QAT stops keeping up while the lr anneals (the late fp32 gains are smaller than
  the 4-bit grid step) or the halfway read was lucky.
- The control reproduces s30 exactly: the same loss at every logged step (0.5278 at step 1880) and the same read-out
  (4.319 / 6.239, bands 88.04 / 55.47 and 93.58 / 68.72). The harness is deterministic, so s30 is the matched control
  for any from-native run on s30's recipe; later screens can drop the ctl arm and halve their cost.
- The ~48 MB form (4-bit g32 QAT body ~38 MB + 8K context list ~9.7 MB) keeps 4.143 (MBPP) / 6.076 (HE): -4.8 / -3.1 %
  vs the 8-bit full-vocab head (4.350 / 6.269). The 8-bit body with the same list (~74 MB per step) keeps 4.242 / 6.185.
- Writes: ~7.3 GB in all (five 1.46 GB state saves incl. the smoke's; the state file is removed). Kept:
  ckpt/qat_full_g32_qat_q4.safetensors (45.7 MB because the scales and biases are stored fp32; ~38 MB with fp16) and
  results/qat_full_g32.json. Lifetime peak footprint 5.68 GB; swap 11.5 -> 10.9 GB over the run.
- Open for the last ~1.5 points, each a <= 20 min single-arm screen against s30:
  - mixed precision: keep the most sensitive Linears at 8-bit (published; an eval-only post-training sensitivity pass
    first);
  - learned group scales instead of min-max (LSQ-style, published);
  - a higher lr floor for the QAT anneal (untested here; aimed at the halfway-vs-end gap).

## QAT follow-ups: three single-arm screens against s30 (pre-registered Sun 15 Nov 07:4x, before any run; sb/qat2.py)
Asked by the orchestrator at 07:3x: one at a time, in this order. Why it pays even as BETWEEN: ~48 MB per step runs at
~500 GB/s while ~74 MB falls toward DRAM speed (Leap KNEE), so the 4-bit form saves ~5 ms per 30-step block on the lossy
road; closing the last ~1.5 points makes that free.
- Each trains ONE arm from the native head over s30's own recipe (1,889 steps x depth 30, lr 1e-5, cosine to 10 %, fresh
  Adam, clip 1.0). The harness is deterministic, so s30 is the matched control. The single-arm trainer was checked
  first: in plain-QAT mode it reproduces this morning's qat arm exactly (step-1 loss 1.0741, 20-step mean 0.9108).
- Read-out in-process:
  - s30 at 8-bit (THE BAR);
  - this morning's plain QAT g32 head (the reference: 4.228 / 6.128 full vocab, 4.143 / 6.076 with the list);
  - the lever at real quantization, and its latent fp32 weights (info).
  - Full vocab and the 8K + context list. MBPP-50 / HE-30 carry the lines; the second halves (MBPP 51-100 / HE 31-60)
    read the bar and the lever.
- PASS (all three): the lever at real quantization keeps kept/32 within 1 % of the bar in all four cells (MBPP-50 /
  HE-30 x full / list). "PASS, not confirmed" if a second half sits below -2 %.
- Stop line at step 944: the lever at real quantization, MBPP-50 full vocab. STOP below 3.688 kept/32 (-6.3 % of the
  matched ctl's 3.936).
1. MIX, mixed 4/8-bit by sensitivity (body <= 44 MB; <= ~54 MB per step with the list).
   - Phase A, eval only: on s30, each of the 8 Linears alone at 4-bit g32 (the rest at 8-bit). Its cost = kept/32 vs
     all 8-bit, the average of MBPP-50 and HE-30 (full vocab).
   - The subset carrying the most cost within +6 MB stays at 8-bit (trained plain); the rest trains through fake 4-bit
     g32. Training runs only if that subset carries >= 40 % of the summed per-module cost; otherwise MIX = KILL from
     phase A.
   - KILL if it recovers less than half of the plain QAT head's gap to the bar on both sets (full vocab); else BETWEEN.
2. LSQ, learned group scales (38 MB).
   - Every Linear trains through a 4-bit g32 grid with a learned per-group scale and offset (scale = s0 exp(qa),
     offset = b0 + s0 qc; s0 / b0 the min-max start; qa / qc at lr 1e-3 on their own Adam and clip). Real 4-bit = the
     learned grid in MLX's affine format (export checked: matches the training forward to 7e-4 on outputs of ~0.9).
   - KILL if it does not beat the plain QAT head on either set (paired kept/32, full vocab, diff <= 0 on both); else
     BETWEEN.
3. FLOOR, a higher QAT lr floor (38 MB): plain QAT g32 with the cosine ending at 30 % of the peak lr (3e-6) instead of
   10 %, aimed at the gap that opened after the halfway stop line. KILL / BETWEEN as LSQ.
- Guards as this morning: share lock MEM from a full-length smoke per lever; kill and stop on swap +1 GB or NO-SWAP < 8 GB
  twice; MLX cache cleared per window; SIGTERM state saves and resume. New: the runner also yields (state saved, resumes
  later) on PAUSE (the Mac is needed) or when another session holds a lock with MEM >= 30 GB (a 45 GB slot); a queued
  quiet ticket already yields.
- Writes per screen: the resume state (~0.73 GB, overwritten at 500 / 1000 / 1500 and after training, removed at the
  end; ~3 GB written) and results/qat2_<lever>.json. The lever head at real quantization (~38-44 MB) is kept only on a
  PASS.
- Time per screen: ~12 min of training (0.37 s per single-arm step in the check) + ~6 min of read-outs; MIX adds ~9 min
  for phase A.

## QAT follow-up 1, MIX (mixed 4/8-bit by sensitivity) = BETWEEN (Sun 15 Nov 08:22; results/qat2_mix.json; 35 min in all)
- Phase A (eval only, ~7 min): s30 with one Linear at a time at 4-bit g32 and the rest at 8-bit; kept/32 vs all 8-bit,
  averaged over MBPP-50 and HE-30.
  - o_proj -2.77 % (+1.84 MB to keep at 8-bit), down_proj -1.50 (+5.51), fc -0.97 (+3.67), k_proj -0.91 (+0.46), q_proj
    -0.80 (+3.67), up_proj -0.69 (+5.51), gate_proj -0.69 (+5.51), v_proj -0.17 (+0.46).
  - The attention output (o_proj, 7 % of the weights) carries the largest single cost. The costs sum to 8.5 %, more than
    the all-4-bit cost (-5.4 % averaged), so they do not add up (noise and overlap).
  - Kept at 8-bit: fc + k_proj + o_proj (+5.97 MB, 55 % of the summed cost; the rule was >= 40 %). Body 43.97 MB, ~53.7
    MB per step with the 8K context list.
- Training: one arm, 1,889 steps (0.60 s per step, beside MTP Head's share-locked run). Stop line at step 944: 3.934 (the
  plain QAT run read 3.935 there).

kept/32 against the bar (s30 8-bit, the same list):
| head | MBPP full | MBPP 8K + context | HE full | HE 8K + context |
|---|---|---|---|---|
| s30 8-bit (the bar) | 4.350 | 4.242 | 6.269 | 6.185 |
| plain QAT g32 (the reference, 38 MB) | 4.228 (-2.80 %) | 4.143 (-2.35 %) | 6.128 (-2.25 %) | 6.076 (-1.77 %) |
| MIX latent fp32 | 4.321 (-0.68 %) | 4.214 (-0.66 %) | 6.128 (-2.25 %) | 6.076 (-1.77 %) |
| MIX at real quantization (44 MB) | 4.412 (+1.42 %, CI -0.1..+3.0) | 4.301 (+1.39 %, CI -0.1..+2.9) | 6.156 (-1.79 %, CI -4.4..+0.6) | 6.067 (-1.92 %, CI -4.8..+0.2) |
- **Verdict: BETWEEN.** MBPP-50 passes (above the bar); HE-30 misses the 1 % line by 0.8-0.9 points. No KILL: MIX closes
  all of the plain QAT head's MBPP gap (151 %) and 20 % of its HE gap.
- Against the plain QAT head (paired kept/32): MBPP +0.184 (CI +0.111..+0.248), HE +0.029 (CI -0.154..+0.197; with the
  list -0.009).
- The second halves flip: MBPP 51-100 -0.90 / -1.07 %, HE 31-60 +1.74 / +1.72 %. Averaging the two halves, MIX sits
  within ~0.3 % of the bar on both sets (~+0.3 % MBPP, ~0.0 % HE). So the HE-30 miss lies inside a 30-file set's noise
  (its CI spans -4.4..+0.6), but the registered line is the verdict.
- 23-30 band, full vocab: MIX 55.73 / 67.59 vs the bar 55.51 / 68.72 (+0.2 / -1.1 pt); plain QAT 54.09 / 67.02.
- The latent fp32 weights read below the quantized form (4.321 vs 4.412 MBPP): the weights adapted to the quantizer.
  (HE's equal kept for the latent and the reference is a coincidence; their bands differ, 66.44 vs 67.02 at 23-30.)
- Size and speed: Leap's KNEE measured 16-48 MB at ~500 GB/s and 96 MB+ at ~276 GB/s. ~54 MB per step sits just past
  the measured cache-speed range (not measured).
- Lifetime peak footprint 5.23 GB; ~3 GB of state saves written and removed; no head kept (not a PASS).

## QAT follow-up 2, LSQ (learned group scales) = BETWEEN (Sun 15 Nov 10:14; results/qat2_lsq.json; 110 min in all)
- Training: one arm, 1,889 steps x depth 30 from native on s30's recipe. Every Linear at 4-bit g32, with each group's
  scale and offset learned (straight-through rounding; log-scale and offset at lr 1e-3, clipped on their own). Body
  38.01 MB, the plain QAT head's size: the learned values fill the same fp16 scales and biases.
- 3.15 s per step: the GPU was shared five ways (three MoonShots decode arms and MTP Head's margin screen beside it).
  Stop line at step 944: 3.950 (line 3.688; plain QAT read 3.935 there).

kept/32 against the bar (s30 8-bit, the same list):
| head | MBPP full | MBPP 8K + context | HE full | HE 8K + context |
|---|---|---|---|---|
| s30 8-bit (the bar) | 4.350 | 4.242 | 6.269 | 6.185 |
| plain QAT g32 (the reference, 38 MB) | 4.228 (-2.80 %) | 4.143 (-2.35 %) | 6.128 (-2.25 %) | 6.076 (-1.77 %) |
| LSQ latent fp32 | 4.185 (-3.80 %) | 4.097 (-3.43 %) | 5.851 (-6.66 %) | 5.791 (-6.37 %) |
| LSQ at real quantization (38 MB) | 4.311 (-0.90 %, CI -2.4..+0.5) | 4.208 (-0.81 %, CI -2.4..+0.5) | 6.128 (-2.25 %, CI -4.9..-0.1) | 6.071 (-1.84 %, CI -5.1..+0.9) |
- **Verdict: BETWEEN.** MBPP-50 passes the 1 % line at no extra bytes; HE-30 misses it by 1.25 / 0.84 points. No KILL:
  against the plain QAT head, MBPP +0.083 kept (CI +0.017..+0.151; list +0.065, CI +0.005..+0.129), 68 % of its MBPP
  gap; HE +0.000 (CI -0.217..+0.195; list -0.005).
- Second halves: MBPP 51-100 -1.20 / -1.56 %, HE 31-60 +0.20 / +0.60 %. Averaged over both halves, LSQ sits ~1 % under
  the bar on both sets (MIX: within ~0.3 %).
- 23-30 band, full vocab: LSQ 54.48 / 67.05 vs the bar 55.51 / 68.72; plain QAT 54.09 / 67.02.
- The latent fp32 weights read far below the quantized form (HE -6.66 %): with the grid learned alongside, the weights
  only work on that grid. Only the quantized head is a deliverable.
- HE full ties the plain QAT head exactly (6.128), the third such tie today. kept/32 is the window length over the
  number of checks (sb/core.py kept_walk), so heads with the same number of checks tie. The bands differ (23-30: 67.05
  vs 67.02), so each form read its own tables.
- Against MIX: MIX buys more on MBPP (+0.184 vs +0.083 kept over plain QAT) for +6 MB; neither moves HE-30 (+0.029 /
  +0.000).
- Lifetime peak footprint 6.84 GB: 1.3 GB above the smoke's 5.55 GB, so 0.8 GB over the lock's MEM 6 GB. ~3 GB of state
  saves written and removed; no head kept (not a PASS).
- The 60 min share lock (sized from the smoke) lapsed at 09:24 mid-run. It was re-taken to the run's end, and guard_lib
  now sizes each lock from the run's own s/step (ee7a3f1).

## QAT follow-up 3, FLOOR (QAT lr floor 30 %) = KILL (Sun 15 Nov 10:44; results/qat2_floor.json; 27.5 min in all)
- Training: one arm, 1,889 steps x depth 30 from native on s30's recipe, every Linear at plain QAT 4-bit g32. The one
  change: the cosine ends at 30 % of lr 1e-5 instead of 10 % (final lr 3e-6 instead of 1e-6). Body 38.01 MB.
- 0.70 s per step (the GPU lighter once MoonShots' decode arms ended). Stop line at step 944: 3.816 (line 3.688; plain
  QAT read 3.935 there).

kept/32 against the bar (s30 8-bit, the same list):
| head | MBPP full | MBPP 8K + context | HE full | HE 8K + context |
|---|---|---|---|---|
| s30 8-bit (the bar) | 4.350 | 4.242 | 6.269 | 6.185 |
| plain QAT g32 (the reference, 38 MB) | 4.228 (-2.80 %) | 4.143 (-2.35 %) | 6.128 (-2.25 %) | 6.076 (-1.77 %) |
| FLOOR latent fp32 | 4.117 (-5.35 %) | 4.029 (-5.02 %) | 5.716 (-8.82 %) | 5.655 (-8.58 %) |
| FLOOR at real quantization (38 MB) | 4.217 (-3.05 %, CI -5.5..-0.9) | 4.123 (-2.80 %, CI -5.1..-0.7) | 5.758 (-8.15 %, CI -12.1..-5.2) | 5.687 (-8.05 %, CI -11.9..-5.0) |
- **Verdict: KILL.** Against the plain QAT head: MBPP -0.011 kept (CI -0.095..+0.073; list -0.019), HE -0.370 (CI
  -0.605..-0.195; list -0.389). At or below it on both sets, and clearly worse on HE.
- The second halves agree: MBPP 51-100 -5.04 / -5.35 %, HE 31-60 -7.02 / -6.50 % vs the bar.
- 23-30 band, full vocab: FLOOR 52.46 / 64.72 vs plain QAT 54.09 / 67.02 (-1.6 / -2.3 pt).
- So the low end of the schedule matters: ending at 3e-6 instead of 1e-6 costs ~6 % kept on HE. The opposite direction
  (a lower floor, or a low-lr cool-down after the cosine) is untested.
- Lifetime peak footprint 5.46 GB (smoke 5.11 GB; lock MEM 6 GB). ~2.9 GB of state saves written and removed; no head
  kept.
- The lock took 120 min at the start, was re-sized at step 40 to 62 min from the run's own 0.97 s/step, and was
  released at the run's end (10:44, 35 min early as the GPU emptied).

### The three follow-ups together (the line: within 1 % kept/32 of the 8-bit body on MBPP-50 and HE-30)
| lever | body | MBPP-50 | HE-30 | verdict |
|---|---|---|---|---|
| plain QAT g32 (the morning's run) | 38 MB | -2.80 % | -2.25 % | BETWEEN |
| MIX: fc + k_proj + o_proj at 8-bit | 44 MB | +1.42 % | -1.79 % | BETWEEN |
| LSQ: learned group scales | 38 MB | -0.90 % | -2.25 % | BETWEEN |
| FLOOR: lr floor 30 % | 38 MB | -3.05 % | -8.15 % | KILL |
- No lever passes both sets. MIX and LSQ each bring MBPP-50 inside the line (MIX more, for +6 MB). Neither moves HE-30
  (+0.029 / +0.000 kept over plain QAT).
- On the confirmation halves, MIX averages within ~0.3 % of the bar on both sets; LSQ sits ~1 % under.
- Untested: MIX and LSQ together; a lower lr floor or a cool-down (FLOOR points that way).

## L1CF, lever 1c as a FULL-DATA distilled auditor (PLAN4D item 2; orchestrator 21:4x; pre-registered Sun 15 Nov 21:5x, before any run, the smoke included; sb/lever1cf.py)
Question: does ~8.4x the 125B's answers lift the 2B's agreement with the 125B on HE+ by >= +1 pt? On HE+ the audited
deep bands sit at the auditor's own agreement (MoonShots' audit law: the 2B agrees 94.42 and its audited 23-30 is 94.32;
the 4B 95.48 and 95.29), so a lift here lifts t30b + the 2B audit (PLAN4D's C2 -> C2b) at no extra draft cost.
- Change vs 1c: the data only.
  - One epoch over every captured n1112 file except the bank's val 40: 3,032 files, 805,005 answer rows. 1c had 360
    files and 95,070 rows over 8 epochs.
  - The files come from ~/FableGod/deepdraft/capture/n1112 (read-only), shuffled once (seed 1115); no file repeats.
  - The bank's 400 files match their capture copies in ids / tk / lp / gen (checked on 2), so 1c's 360 are in the set.
  - HE+ 24 and MBPP 100 are separate captures, not in n1112 (checked: no name overlap).
- Recipe unchanged:
  - rank-16 LoRA, scale 10, on layer 23's 7 linears; Adam lr 1e-4 constant;
  - cross-entropy to the 125B's renormalized top-32 on the answer rows;
  - final norm and tied lm_head frozen (bf16 copy for the loss, the 4-bit head for every read-out).
  - Rank stays 16 so the read isolates the data. PLAN4D's "higher rank" is the next arm if this one ends BETWEEN or still
    climbs at the end.
- Form: 3,032 files' layer-23 inputs (~10 GB) don't fit, so each step runs the frozen trunk (layers 0-22) on its own
  file first. One epoch makes the same trunk passes 1c's caching made.
- Read-outs: 1c's, unchanged.
  - Sets: HE+ 24 (6,375 rows), MBPP 100 (15,641), val 40 (9,984 rows of the 40 held-out capture files).
  - Diff vs the control (LoRA zero) with a 95 % bootstrap CI over files; also the unsure rows (125B margin < 4 nats) and
    rows gained / lost.
  - Read after steps 1,000 / 2,000 / 3,032 (the end of the epoch). At 1c's pace (0.18 s trunk + 0.14 s step per file),
    that is ~5 / 10 / 16 min of training: the orchestrator's 5 / 10 / 15 min, fixed as steps so a shared GPU can't move
    them.
  - Each read saves the LoRA, with its sha256 in the JSON.
  - 1c's reads, for the slope: steps 450 / 840 / 1,550 / 3,024 gave HE+ +0.27 / +0.44 / -0.03 / -0.09, repeating 360
    files.
- VOID: the control is more than 0.1 off lever7_agree.py's 94.42 / 85.67 / 87.29.
- KILL (orchestrator 21:4x, PLAN4D): HE+ < +1.0 at every read, or val falls. "Val falls" means val is below the control
  at any read, or the last read's val is >= 1.0 pt under its best read.
- PASS: not KILL, and HE+ >= +1.0 at the end of the epoch. The final LoRA (sha) is then PLAN4D's C2b candidate for
  Tuesday's audit join.
- BETWEEN: not KILL, but HE+ >= +1.0 only at an earlier read. That read's LoRA (sha) is the candidate, and NewIdeas picks.
- Information: MBPP (PLAN4D expects +1-3 over the audited 85.1), the unsure rows, the loss curve and s/step.
- Machine (orchestrator 21:4x):
  - Monday by day, not tonight: the 45 GB chain runs from 23:35.
  - <= 7 GB lifetime footprint, one process, share lock with "MEM 7 GB". It is the day's first GPU job above 4 GB.
  - Kamen's floor: >= 5 GB free while he uses the Mac.
  - Start gate, from Mon 07:00: no HOLD, PAUSE or shut lid (with no external display); no quiet ticket; no other lock
    above MEM 4 GB; NO-SWAP >= 14 GB (7 + 5 + 2).
  - While it runs, it yields on a quiet ticket, PAUSE, a shut lid or a big slot, saving its state and resuming later. It
    is killed and stopped on swap +1 GB or NO-SWAP < 5 GB twice in a row. A HOLD stops it and writes a marker.
  - No first start after Mon 20:30. At 22:30 it is stopped with its state saved, for Monday night's runs.
- Smoke (a harness and memory check, not a result): the 20 longest files (up to 1,956 tokens), the full eval sets (so the
  control check runs), and reads at steps 10 and 20. Its lifetime peak is checked against the 7 GB cap.
- Cost if it passes: nothing at draft time beyond the 2B audit pass; the LoRA merges into layer 23's weights.
- Smoke done (Sun 15 Nov 21:53, scripts/l1cf_smoke.sh, under a share lock "MEM 7 GB"; harness only). The control
  reproduced 94.42 / 85.67 / 87.29 exactly. The lifetime peak footprint was 6.21 GB on the 20 longest files, under the
  7 GB cap. On those files a step took 0.47 s (trunk 0.31 + fit 0.15); the eval caching took 25 s and each read 5 s.
- All numbers here are lab numbers: teacher-forced top-1 agreement on the 125B's captured path, not real decoding.

## L1CF result = KILL (Mon 16 Nov 07:09; results/lever1cf_a.json; one epoch over 3,032 files, 8.5 min in all)
All numbers are LAB (true-state) numbers: teacher-forced top-1 agreement with the 125B's top-1 on its captured path, not
real decoding.
- The control reproduced 94.42 / 85.67 / 87.29 exactly, so the run is not VOID.
- Reads (diff vs the control, 95 % bootstrap CI over files):

| read (step) | HE+ 24 | MBPP 100 | val 40 | HE+ unsure rows | HE+ rows gained / lost |
|---|---|---|---|---|---|
| 1,000 | 94.87 (+0.45, CI +0.12..+0.81) | 86.41 (+0.75, CI +0.38..+1.13) | 88.93 (+1.64, CI +1.13..+2.19) | +2.87 | 77 / 48 |
| 2,000 | 95.22 (+0.80, CI +0.37..+1.25) | 86.43 (+0.77, CI +0.42..+1.13) | 89.22 (+1.93, CI +1.43..+2.50) | +4.94 | 89 / 38 |
| 3,032 (end) | 95.23 (+0.82, CI +0.33..+1.37) | 86.50 (+0.83, CI +0.54..+1.17) | 89.28 (+1.99, CI +1.45..+2.57) | +4.94 | 85 / 33 |

- Verdict, by the pre-registered lines: **KILL**. HE+ stayed under +1.0 at every read (+0.45 / +0.80 / +0.82). Val did
  not fall: it rose at every read. So the KILL rests on the HE+ line alone. The end CI (+0.33..+1.37) does not exclude
  +1.0, but every point estimate is below it.
- Against 1c, at matched steps (3,032 here, 3,024 there): HE+ +0.82 vs -0.09, MBPP +0.83 vs -0.81, val +1.99 vs +0.42.
  The number of updates is the same; only the data differs (each file seen once, 8.4x the rows). The extra data removed
  1c's overfit. Each end gain is about twice 1c's best read (HE+ +0.44, MBPP +0.52, val +1.43).
- HE+ went flat in the last third: +0.80 -> +0.82 over the last 1,032 files (MBPP +0.77 -> +0.83, val +1.93 -> +1.99).
  The training loss kept falling (by thirds 0.420 / 0.396 / 0.384). So more files of the same kind, at rank 16 on layer
  23, look unlikely to reach +1.0 (not tested).
- The gain is in the hard rows. On HE+, the rows where the 125B is unsure (margin < 4 nats, 627 rows) gained +4.94 pt;
  the sure rows gained +0.37. Val is the same (unsure +4.39, sure +1.22).
- For scale (MoonShots' numbers): the 4B auditor agrees 95.48. This 2B reaches 95.23, about three quarters of the
  2B -> 4B gap, at the 2B's cost. If the audit law holds (the audited HE+ 23-30 band sits at the auditor's agreement),
  this auditor's band would be ~95.2, against 94.32 now and 97.8 targeted. The audit join itself is untested.
- MBPP (information): agreement +0.83. PLAN4D hoped for +1-3 on MBPP's audited 23-30 band; only the audit join would
  show how agreement carries there.
- Timing: the run was faster than the pre-registered estimate (0.156 s/step: trunk 0.092 + fit 0.064, against ~0.32 at
  1c's pace). The reads came at the pre-registered steps, but at ~2.5 / 5.3 / 8.1 min of training (07:03:32, 07:06:20,
  07:09:10), not ~5 / 10 / 16. The eval caching took 10 s and each read 3 s.
- Machine: started 07:00:41 at the first gate check (NO-SWAP 47 GB) under a share lock "Sandbox98 MEM 7 GB cap (smoke
  peak 6.21 GB)". The lifetime peak footprint was 6.15 GB (MLX peak 5.07 GB, max RSS 1.66 GB). Swap stayed at 6,852 MB.
  There were no yields. The lock was released at 07:09:28.
- LoRA checkpoints (local in ckpt/, not in git; 2.56 MB each):
  - s1000 c4d9e7cd437e7cf1574dea5b9c2817f5a39e73a75b222310c388902e9d67011d
  - s2000 c7a129a57169c452724d48a6dae34bef2f26a49cbe924bbdfe52547f416e989a
  - s3032 c62257f4b918510c1fa810b1a6846c0040c3a7e077be88826097892066aa6ffe (the best read on all three sets)
- Next arm: none follows automatically. The pre-registration named a higher rank only for BETWEEN or a still-climbing
  end, and HE+ ended KILL and flat. Open for NewIdeas: a bigger adapter (rank or layers), the 4B auditor, or dropping C2b.

## SIB, sibling-state drafting: the sandbox analog (PLAN900 bet 9b; FAR's idea 1, ~/FableGod/far/IDEAS_1115_info.md section 1; NewIdeas 12:2x; pre-registered Mon 16 Nov 12:3x, before any run, the smoke included; sb/sibstate.py, levers/sibmix.py)
Question: after a miss, can the head draft the next block from the target's exact states at the just-rejected rows (the
sibling branch, which a real check has already computed) instead of its own carried state? FAR's line: at least half of
the own -> true gap closed at depths 1-8. No 125B capture holds rejected rows' states; the sandbox can make them.
- Analog: target = the 2B, drafter = s30 (frozen), 32-row checks.
- Cells: the check starts of s30's own decode walk (core.kept_walk on the teacher-forced chain table: each check keeps
  1 + the run of rights, capped at 31) that follow a miss. Sets: HE 60 (FAR's HumanEval line, primary) and MBPP 100, all
  windows (both are dev sets). Each window's first start and any start after a full 32-token check have no rejected rows
  and are left out.
- The sibling branch for the start s' = s + k + 1 after a miss at depth k of the check at anchor s:
  - The block that check verified is s30's FREE-RUNNING draft from s (its own guesses fed back, as a real chain drafts).
  - Its tokens from the missed position on (the wrong guess, then the drafter's own continuation) are the rejected
    continuation: L = min(8, 31 - k, rows left in the window) tokens, since the block ends at depth 30.
  - The sibling states: the 2B over the true tokens up to s' followed by those L tokens, through a KV / Gated-DeltaNet
    cache that branches at s'. The post-norm final hidden at rows s'+1 .. s'+L is the W the head reads.
- Arms at each cell, teacher-forced with the real tokens (dd_ttt's eval: depth d reads the token at s'+d+1):
  - own: the carried stand-in (the chain);
  - true: the stored true W at row s'+d (evalx's oracle);
  - true_r: the true W at row s'+d recomputed along the same cached path as the sibling rows (a harness control);
  - sib: the sibling state at row s'+d for d <= L, and the carried stand-in past L.
- Read-outs at these cells, per band 1-2 / 3-6 / 7-8 and over depths 1-8 as one band (depth 0 reads the same true anchor
  state in every arm):
  - walk-strict (CERTIFY98 amendment 4), the primary: P(right at d | right at 0..d-1), each arm conditioned on its own
    rights, pooled as right cells over cells (CERTIFY98's ratio estimate);
  - marginal: the per-depth accuracy at the same cells;
  - G = (sib - own) / (true - own), the share of the own -> true gap closed, with a bootstrap 95 % CI over windows
    (2,000 draws, rng 1116).
- Fine-tune variant (FAR's warning: the head never saw a state whose token history differs from its tokens):
  - sb/train_multi.py from s30o with its Adam: two arms on the same slices in the same order, ctl (plain) and sibmix.
  - sibmix: at the training windows' post-miss walk starts (s30's walk), depths 2-9 read the sibling rows instead of the
    carried stand-in. Every other anchor trains as usual.
  - Data: the screen pool (120 Magicoder answers s30 never saw), one pass (579 slices of 96 anchors), depth 12, lr 1e-5
    with a 20-step warm-up and a cosine to 10 % (the FAST format). The pool's sibling rows come from the same pipeline
    with s30.
  - Read on the same cells and sibling rows as s30 (s30's walk and continuations). The tuned heads' own continuations
    are not rebuilt: a stated approximation.
  - G_FT = (the sibmix head's sib - the ctl head's own) / (the ctl head's true - the ctl head's own): the deployable
    gain against a matched control.
- VOID (harness checks; the smoke runs them first, the full run again):
  1. s30's walk on its own chain table reproduces its gate kept/32 (MBPP 100 4.42, HE 60 6.43) to 0.01.
  2. The free-running drafts match the teacher-forced right / wrong pattern at depths 0..k on >= 99 % of starts.
  3. true_r reads within 0.3 pt of true (walk-strict, depths 1-8) on each set.
- Lines (walk-strict, depths 1-8):
  - PASS: G >= 0.50 on HE and >= 0.25 on MBPP, for the frozen s30 (G) or for the fine-tune (G_FT).
  - KILL: G < 0.25 on HE for both the frozen head and the fine-tune; or the gain sits only at depths 1-2, i.e. G over
    depths 3-8 < 0.25 on HE for both (FAR: "better only at depths 1-3, which the tree already covers").
  - BETWEEN: otherwise.
- Information: the target's margin at the misses (the share under 2 nats; FAR's 125B figure is 92 %), the direction
  error of the sibling and of the carried state against the true W per depth, the share of cells with L < 8, kept
  within depths 0-8, and the marginal read-outs.
- Machine: <= 5 GB, one process at a time, a share lock with "MEM 5 GB" in the reason. Start only with NO-SWAP >= 13 GB
  (peak + 8) and no quiet ticket. PAUSE, a quiet ticket or a shut lid make it yield; a HOLD stops it.
- All numbers are LAB (true-state, teacher-forced) numbers on the 2B: not real decoding, and not the 125B.
- Order (orchestrator 12:4x): phase 1 = the smoke and the main read-out (frozen s30), which ends with logs/sib_END so
  EvalSuite can resume. Phase 2 = the fine-tune variant, run later. Without the fine-tune, only a frozen-head PASS is
  final.
- Amendment (Mon 16 Nov 13:1x, after the smoke's harness check and before any full run). The smoke (3 + 3 windows)
  failed its harness check on MBPP: true_r read 89.18 against true 90.38 (HE 92.59 against 92.78). The cached path's
  states sat 2 % (direction error) off the stored ones, which flips cells at the thin margins of post-miss starts.
  - The sibling and true_r rows now come from one full causal forward per post-miss start with no cache (the capture's
    own path). It runs two rows of one length: the true tokens up to s' plus the continuation, and the true tokens up
    to s' + L. The lines are unchanged. The smoke check's limit on the recomputed rows' error drops from 2 % to 0.5 %.
  - For the record: the failed smoke's sibling arm read far below own (HE walk-strict 1-8: sib 59.4, own 83.9, true
    92.8). Its sibling states sat farther from the true state (direction error ~1.0-1.1) than the carried state
    (0.8-0.97). Six windows, a harness run only.
- Amendment 2 (Mon 16 Nov 14:3x, after the second smoke's harness check and before any full run). The second smoke
  (14:24) failed again: the recomputed true rows sat 1.36 % (HE) / 1.14 % (MBPP) off the stored ones, and MBPP's true_r
  read 89.23 against true 90.38 (HE 92.80 against 92.78).
  - Cause (scratch check 14:27, one MBPP window): the prefix-length forwards choose other matmul kernels on short
    prefixes. On those starts they sat ~1.8 % off; on the longer ones they were bit for bit equal. A forward shaped
    exactly like the capture's (core.target_states: batch 1, the whole length S, no cache) gave the stored states bit
    for bit, and stayed deterministic on a repeat. With a window of the answer replaced, every row before it stayed
    bit for bit unchanged.
  - Fix: each sibling row now comes from one capture-shaped forward. It runs over the true tokens, with the rows
    s'+1 .. s'+L replaced by the rejected continuation. true_r comes from one capture-shaped forward of the true
    tokens per window, so it should equal the stored rows bit for bit.
  - The smoke check adds: at least 99.9 % of the starts must keep their rows before the window bit for bit. The lines,
    the arms, the cells and the 0.5 % and 1 pt limits are unchanged.

## SIB main read-out = NOT PASS on the frozen head, KILL side (Mon 16 Nov 15:03; results/sib_s30.json, results/sib_verdict_main.json; build 25 min + score 3.4 min)

- What ran: the frozen s30 head on HE 60 + MBPP 100, every window. At each post-miss start, the carried stand-in at
  depths 1-8 was replaced by the 2B's sibling state, with the real tokens. The sibling state is the 2B's state over the
  rejected continuation, from one capture-shaped forward per start. These are LAB numbers (teacher-forced on the 2B's
  own greedy answers), not real decoding and not the 125B.
- The harness was exact:
  - The recomputed true rows equal the stored ones bit for bit (relative error 0.0 on every row).
  - Every start's rows before its window stayed bitwise.
  - VOID-1: walk kept/32 = the gate on both sets (HE 6.4309, MBPP 4.4211).
  - VOID-2: free-running agreement 1.00.
  - VOID-3: true_r − true (walk-strict 1-8) was −0.02 pt on HE and 0.00 on MBPP. The small HE gap comes from the
    starts within 8 rows of the answer's end (4.7 % HE / 2.6 % MBPP), which keep the carried state past L.
- Lifetime footprints: build 3.50 GB, score 2.68 GB.

Walk-strict (%): own / true / sib, and G = (sib − own) / (true − own) [95 % CI, bootstrap over windows]:

| band | HE own | HE true | HE sib | HE G | MBPP own | MBPP true | MBPP sib | MBPP G |
|---|---|---|---|---|---|---|---|---|
| 1-2 | 83.40 | 91.80 | 57.09 | −3.13 [−4.02, −2.44] | 75.79 | 87.36 | 51.23 | −2.13 [−2.38, −1.90] |
| 3-6 | 81.59 | 93.77 | 69.97 | −0.96 [−1.33, −0.65] | 75.53 | 89.90 | 67.22 | −0.58 [−0.76, −0.43] |
| 7-8 | 85.38 | 94.79 | 80.13 | −0.56 [−1.16, −0.11] | 75.71 | 90.02 | 75.22 | −0.03 [−0.42, +0.31] |
| 1-8 | 82.80 | 93.35 | 62.60 | −1.91 [−2.36, −1.48] | 75.67 | 89.03 | 56.82 | −1.41 [−1.57, −1.28] |
| 3-8 | 82.38 | 94.06 | 71.66 | −0.92 [−1.25, −0.67] | 75.56 | 89.93 | 68.39 | −0.50 [−0.67, −0.36] |

- Marginal 1-8: HE 72.81 / 90.36 / 39.86 (G −1.88); MBPP 67.19 / 86.71 / 38.06 (G −1.49).
- Kept within depths 0-8 from the post-miss start: HE own 4.13, true 5.93, sib 2.26; MBPP 3.13 / 4.83 / 1.90.
- Direction error to the true state, depth slots 1 → 8:
  - HE: own 0.85 → 0.99, sib 0.95 → 1.10.
  - MBPP: own 0.81 → 0.99, sib 0.94 → 1.10.
- The 2B's margin was under 2 nats at the misses on HE 44.7 % (median 2.50) and MBPP 63.1 % (median 1.37).
- Verdict (the pre-registered frozen-only branch): NOT PASS on the frozen head, KILL side.
  - G(1-8): HE −1.91 (3-8: −0.92), MBPP −1.41 (3-8: −0.50). All are under the 0.25 KILL line; the PASS line was
    ≥ 0.50 (HE) and ≥ 0.25 (MBPP).
  - The final verdict waits for the fine-tune (phase 2), as pre-registered.
- Reading:
  - The 2B's state after the wrong token belongs to a different sequence and carries that token's own next-token
    prediction.
  - Even with 45-63 % of the misses at a margin under 2 nats, this state sits farther from the true state than the
    head's own carried state at every depth.
  - It misleads the head most right after the miss (depths 1-2: −26 pt HE, −25 pt MBPP).
  - The deficit fades with depth, where walk-strict keeps only the cells whose chain survived.

### Mix oracles (NewIdeas 15:0x; sb/sib_oracle.py, sb/sib_mixacc.py; results/sib_oracle.json, results/sib_mixacc.json)

Asked after the main read-out, before phase 2: can a mix of the carried state and the sibling state beat the carried
state?

- Direction error (41 s, 3.11 GB):
  - The mix is unit(w·sib + (1 − w)·own). w is fitted per band on one half of the windows (split by parity) and scored
    on the other half, both ways.
  - Held-out gain over own: HE 1-2 6.7-8.1 %, 3-6 5.8-7.3 %, 7-8 4.1-4.9 %; MBPP 1-2 6.6-7.0 %, 3-6 6.3 %, 7-8
    4.7-4.8 %. The best w was 0.35-0.40.
  - The two errors are positively correlated: cos(own − true, sib − true) is 0.36-0.42. Uncorrelated errors of these
    sizes would allow ~30 %.
- Accuracy (180 s, 2.99 GB):
  - The frozen head reads the mixed stand-in online (the chain carries on from it), at the sibling row's RMS.
  - own and sib reproduce the main read-out exactly.

| walk-strict 1-8 | own | w 0.2 | w 0.4 | sib |
|---|---|---|---|---|
| HE | 82.80 | 83.35 (G +0.05) | 80.35 (G −0.23) | 62.60 (G −1.91) |
| MBPP | 75.67 | 76.65 (G +0.07) | 74.27 (G −0.10) | 56.82 (G −1.41) |

  - By band at w 0.2 (1-2 / 3-6 / 7-8): HE G +0.06 / +0.05 / +0.06; MBPP +0.05 / +0.09 / +0.12.
- Reading:
  - A small dose of the sibling state carries a little information: at w 0.2, walk-strict 1-8 gains +0.5 pt on HE and
    +1.0 pt on MBPP. A larger dose already hurts.
  - The best fixed mix closes 5-7 % of the own → true gap, against the 25 % KILL line and the 50 % PASS line.
  - NewIdeas' oracle rule: within ~5 % → a phase-2 KILL is near-certain; ≥ 15 % → phase 2 is worth its run. Here the
    direction error gains 6-8 %, and accuracy closes 5-7 % of the gap.
- Phase 2:
  - Armed at 15:05 beside EvalSuite and stopped at 15:07. It had started while EvalSuite waited for memory: EvalSuite
    needs 43 GB NO-SWAP, and the 5 GB job widened its gap.
  - Re-armed to start 5 min after EvalSuite's chain end (results/T1C_RESF2_END), with the 30 GB big-lock gate.
  - The 3 pool windows it built are kept.

## SIB final verdict = KILL, oracle-called (NewIdeas, Mon 16 Nov 15:1x; phase 2 disarmed 15:19, before any step)

- NewIdeas, who set the task and its lines, called the KILL from the accuracy oracle instead of the pre-registered
  phase 2:
  - The direct test's best fixed mix closes only 5-7 % of the gap: G +0.05 (HE) / +0.07 (MBPP), at most +0.12 in any
    band. That is 4-7x below the 0.25 KILL line.
  - The errors are correlated (cos 0.36-0.42), so a mix has little to average out.
  - The direction-error gain (6-8 %) sits just above NewIdeas' ~5 % call line; the call rests on the accuracy oracle
    and on the read that one learned pass over 120 answers won't reach 0.25.
- Label: phase 2 (the learned mix) was not run, so its upside is untested. Reopen only with a new mechanism (e.g. a
  margin- or depth-gated mix) and a reason it beats the fixed-mix oracle by 3x+.
- Why now (NewIdeas): the post-17:30 window is crowded, and MTP Head's S13 → S14 → S15 chain (QAT head + the remaining
  drift levers) is the better use of its sandbox capacity.
- This departs from the pre-registration, which tied a final KILL to G_FT. The frozen read-out and both oracles above
  are the record. The pool's sibling rows (3 windows) stay in data/sib/pool.
- PARKED micro-lever (NewIdeas): the w 0.2 fixed mix of the head's own state and the sibling state.
  - LAB gain on the 2B, walk-strict 1-8: +0.55 pt HE, +0.98 pt MBPP.
  - It is one read on the main read-out's cells, with no CI.
  - The check already computes the rejected row's state, so NewIdeas notes it costs nothing on the 125B.
  - It goes on the drafter's final-config list, to verify on the 125B's states before use.

## ASSIM: carried-state correction from the checks' accepted rows (drift screen 1; pre-registered Mon 16 Nov 18:5x, before any data)

NewIdeas' pick (18:5x) from THINK_1116c (~/FableGod/deepdraft/THINK_1116c.md): Fable C's ASSIMILATED DRAFTING
(~/FableGod/fablec/IDEAS_1116c_drift.md section 3) and Fable B's MEANPULL / ERRMEM (~/FableGod/fableb/IDEAS_1116c_drift.md
section 1), in one run. LAB = teacher-forced on the 2B with the exact true-state harness; not real decoding.

- Setup: the frozen s30 head on the SIB main read-out's cells (post-miss starts of HE 60 / MBPP 100, depths 0-8; data/sib).
  Code: sb/assim.py; runner scripts/drift_run.sh plans/assim.sh (share lock "MEM 5 GB cap"; NO-SWAP >= 13 GB; no
  lock >= 30 GB; no quiet ticket; HOLD / PAUSE / shut lid respected).
- The answer's decode trajectory = the SIB build's 32-row checks (data/sib 'kept'). Block b (anchor s_b) accepts k_b
  drafted tokens; its check gives the true W at s_b + d for d = 1 .. k_b + 1. The head's stand-in at depth d of block
  b stood in for W[s_b + d] on the same tokens, so r_{b,d} = unit(W[s_b + d]) - unit(stand-in) is exact and free.
  Everything is in unit-RMS space (the head's input norm makes scale irrelevant). A corrected stand-in keeps the
  carried row's RMS, and depth 0 is never touched.
- Arms (each corrects every block of the answer from the EARLIER blocks only):
  - assim (Fable C as written): e = the running mean of the uncorrected chain's residuals r_{b,d} over the earlier
    post-miss starts, d = 1 .. min(k_b, 8); the head reads unit(own + e) at depths 1-8.
  - assim_ema4: the same with an EMA over starts, half-life 4 starts.
  - assim_all (information): the same mean over EVERY earlier check, d = 1 .. k_b + 1 (the runtime's full set).
  - mp20 / mp35 / mp50 (Fable B (a)): u <- unit((1 - K) u + K m) at depths >= 3, K 0.2 / 0.35 / 0.5. m = unit(the EMA
    of the true W rows from the answer's first anchor up to the block's anchor), half-life 64 rows: the window's slow
    component over accepted rows. It is applied on the head's own input path, before its input norm.
  - errmem (Fable B (b)): e_d = the per-depth EMA (0.5 per check) of this arm's OWN pre-correction residuals at depth
    d (closed loop: what the drafter itself would compare after each check); u <- unit(u + e_d).
    ACCEPTED ROWS ONLY: Fable B's every-row form also uses the rejected rows' check states against the head's
    free-running stand-ins there, which these cells do not hold.
    One approximation: a residual's validity uses the uncorrected trajectory's k_b.
  - mp35_errmem: mp35, then errmem on top (its residuals measured after the pull).
- Read-out per arm: own / true / arm walk-strict at 1-2 / 3-6 / 7-8 / 1-8 / 3-8, G = (arm - own) / (true - own) with
  the bootstrap CI over windows (2000, rng 1116), and the direction error of the stand-in the head read, by depth.
  HE decides; MBPP is reported.
- LINES (the authors', NewIdeas 18:5x), per arm, on HE:
  - PASS: G(3-8) >= 0.25 with the CI's lower end > 0, AND the direction error at depths 3-8 down >= 15 %.
  - KILL: G(3-8) < 0.10, or the direction error not down.
  - BETWEEN otherwise.
  - Eight arms are read, so a lone PASS is confirmed before anything is built. Fable C's screen 2 (the fine-tune with
    the assim hook) needs NewIdeas' OK first.
  - For scale: SIB's best fixed mix gave G +0.05-0.07.
- VOID if:
  - own's or true's per-window counts differ from results/sib_s30.json's (the main read-out's cells);
  - the smoke's no-op arm differs from own in any cell.

## WALKMIX: FAR's WALK-MIX TTT (drift screen 2; pre-registered Mon 16 Nov 19:0x, before any data)

NewIdeas' pick (18:5x) from THINK_1116c: FAR's idea 1 (~/FableGod/far/IDEAS_1116c_drift.md). LAB = teacher-forced on
the 2B, not real decoding.

- Mechanism: t30b's loss drops a row from the chain at its first wrong argmax (accept-until-fail), so the rows after a
  miss never train. The walk-mix keeps survivors at KL weight 1 (the failing depth included) and keeps the missed rows
  in the chain at weight w from the next depth on. The depth weight uses the weighted row count, so w = 0 is today's
  loss and w = 1 the 'allrows' loss (MTP Head's ALLROWS: deep walk bands +5.8 / +8.6 pt, chain kept flat).
  Code: sb/core.py ttt_step (gated on the arm's hook marker _walkmix_w; every other run is unchanged);
  levers/walkmix3.py and walkmix5.py (w 0.3 / 0.5); levers/walkmix0.py (the smoke's check arm).
- Runs (scripts/drift_run.sh plans/walkmix.sh; share lock "MEM 5 GB cap"):
  - sb/train_multi.py, 400 steps at depth 30, from s30o with its Adam (pool2, warm-up to 1e-5, cosine to 1e-6).
  - Process A trains ctl and mix3 (tag wmix). Process B trains mix5 (tag wmixb): three arms in one process would peak
    ~5.2 GB, over the cap. The harness is deterministic, so B's slices, order and start equal A's.
  - Smoke first: 3 steps of ctl / w0 / w0.5 in one process; w0's per-step loss must equal ctl's within 1e-3
    (relative), and w0.5's must differ.
- Read-out:
  - Strict (walk-strict) per depth: sb/evalx.py on MBPP-50 / HE-30, runs s30, ctl, mix3, mix5.
  - Tree-32 kept: MTP Head's ~/TrainLab/sb98/tree8.py, run READ-ONLY from ~/Sandbox98 (sha256 b7e95b92c1cd...). It is
    the EAGLE-2 proxy, best-first over path probabilities, off-path children at gamma 1 (0.5 / 2 as the band), with
    window-bootstrap CIs on each arm / ctl ratio.
- LINES (NewIdeas 18:5x), per arm, on each set:
  - PASS: tree-32 kept (gamma 1) >= x1.03 of ctl on MBPP-50 AND HE-30, with walk-strict at depths 1-4 (pooled) within
    -0.2 pt of ctl on both.
  - KILL: tree-32 kept < x1.01 on both sets.
  - BETWEEN otherwise.
  - Information: FAR's shallow-loss flag (walk-strict 1-4 down > 0.5 pt); the control against s30 (it must hold: MBPP
    7-14 marginal not down > 1.5 pt).
- VOID if:
  - the smoke's w0 arm does not match ctl;
  - tree8's own self-check (its first window equals core.chain_table) fails.

## ASSIM result (Mon 16 Nov 19:05) = KILL, all 8 arms (results/assim_sib.json, logs/assim_main.log)

LAB (teacher-forced, the 2B's exact true-state harness); not real decoding. HE 60 windows (2,265 blocks, 2,181 scored
post-miss starts), MBPP 100 (5,205 blocks, 5,085 scored).

- VOIDs clean:
  - own's and true's per-window counts equal the SIB main read-out's on both sets;
  - the smoke's no-op arm equals own in every cell (3 + 3 windows);
  - the hook's row write-back was checked on a toy (rows replaced as intended, input untouched, bf16 too).
- HE (decides). Columns: G(3-8) [95 % CI]; walk-strict 3-8; direction error at depths 3-8. Own reads 82.38 walk-strict
  and 0.981 direction error; true reads 94.06.

  | arm | G(3-8) | walk-strict 3-8 | direction error 3-8 | verdict |
  |---|---|---|---|---|
  | assim (Fable C as written) | -3.57 [-4.09, -3.11] | 40.64 | 1.256 | KILL |
  | assim_ema4 | -3.35 [-3.87, -2.94] | 43.19 | 1.223 | KILL |
  | assim_all (every check, d <= k + 1) | -4.86 [-5.62, -4.24] | 25.62 | 1.259 | KILL |
  | errmem (Fable B, accepted rows) | -3.61 [-4.19, -3.14] | 40.17 | 1.146 | KILL |
  | mp20 | -0.16 [-0.24, -0.08] | 80.54 | 0.931 | KILL |
  | mp35 | -0.75 [-0.91, -0.62] | 73.65 | 0.929 | KILL |
  | mp50 | -1.58 [-1.85, -1.35] | 63.94 | 0.950 | KILL |
  | mp35_errmem | -3.22 [-3.76, -2.80] | 44.77 | 1.093 | KILL |

- MBPP points the same way. Own reads 75.56 at 3-8 and true 89.93. G(3-8): assim -2.47 [-2.75, -2.22], assim_ema4
  -2.20, assim_all -2.53, errmem -2.03, mp20 -0.09 [-0.14, -0.04], mp35 -0.57, mp50 -1.20, mp35_errmem -1.81.
- Reading:
  - The past residuals' mean is a large vector that is barely aligned with the current block's error.
  - At depth 1 the assim arm leaves the direction error flat (HE 0.846 -> 0.839), yet walk-strict 1-2 falls
    83.40 -> 64.22. The head reacts strongly to moves that the direction error does not register, and the error
    compounds from there (1.33 at depth 8).
  - The pull toward the window's mean true state cuts the direction error ~5 % and still costs accuracy, as the
    sibling states did in SIB. Target-space content in the carried state hurts the head's reading.
  - Fable C's premise (a persistent per-sequence drift GAP) may still hold. The error's DIRECTION is not persistent
    enough for an additive correction.
- Not run: Fable B's every-row errmem (it needs the head's free-running stand-ins at the rejected rows, not in these
  cells) and Fable C's fine-tune with the assim hook (moot after this read).
- Run: 18:54-19:05, wall 606 s; MLX peak 2.04 GB, footprint 2.61 GB; swap unchanged (7,610 MB); share lock "MEM 5 GB
  cap" released.

## WALKMIX result (Mon 16 Nov 20:25) = BETWEEN, both arms (results/walkmix_eval.json, walkmix_tree.json, walkmix_verdict.json)

LAB (teacher-forced on the 2B; tree-32 is the EAGLE-2 proxy); not real decoding. 400 steps at depth 30 from s30o with its
Adam; ctl + mix3 in process A (tag wmix), mix5 in process B (tag wmixb). Read on MBPP-50 / HE-30.

- VOIDs clean:
  - smoke: w0's per-step loss equals ctl's within 1e-3 relative (0.0182 / 0.27542 / 0.39798 vs 0.0182 / 0.27542 /
    0.39782); w0.5's differs (0.41912 / 0.98592 at steps 2-3);
  - tree8's self-check PASS on all 6 run x set.
- The control holds against s30: MBPP 7-14 marginal 62.88 -> 62.25 (-0.64 pt); HE 74.24 -> 73.32 (-0.91).
- Tree-32 kept (gamma 1), arm / ctl [95 % CI over windows]. Columns: MBPP-50; HE-30; the gamma 0.5 / 2 band (MBPP; HE);
  chain-32 and tree-16 ratios (MBPP / HE).

  | arm | MBPP-50 | HE-30 | gamma 0.5 / 2 | chain-32 | tree-16 | verdict |
  |---|---|---|---|---|---|---|
  | ctl | 5.911 | 7.825 | | 4.264 / 6.002 kept | 5.172 / 6.767 kept | |
  | mix3 (w 0.3) | 5.994 = x1.0141 [1.0026, 1.0274] | 7.802 = x0.9970 [0.9813, 1.0123] | x1.027 / x1.019; x0.998 / x0.998 | x1.0166 / x1.0030 | x1.017 / x1.010 | BETWEEN |
  | mix5 (w 0.5) | 5.985 = x1.0125 [1.0006, 1.0245] | 7.764 = x0.9921 [0.9749, 1.0111] | x1.019 / x1.015; x0.995 / x0.991 | x1.0079 / x0.9985 | x1.012 / x1.005 | BETWEEN |

- Strict per depth (walk-strict; d0 = the first guess), arm - ctl in pt, d0 .. d8:
  - MBPP mix3 -0.17 -0.29 +0.16 +1.03 +1.02 +1.34 +2.11 +2.64 +3.00; mix5 -0.26 -0.34 -0.07 +0.99 +1.14 +1.46 +2.63
    +3.37 +3.23.
  - HE mix3 +0.09 -0.17 -0.36 +0.44 +0.59 +0.05 +0.27 +0.44 +0.10; mix5 +0.14 -0.30 -0.15 +0.23 +0.57 +0.07 +0.41 +0.51
    +0.03.
  - ctl: MBPP 88.24 81.42 78.79 76.08 76.09 75.85 75.12 74.91 74.69; HE 93.29 89.70 87.80 86.34 85.64 85.70 85.69 85.57
    85.74.
  - The shallow line holds on either reading of "d1-d4". Pooled band depths 1-4: +0.34 / +0.27 (MBPP), +0.08 / +0.04
    (HE). Pooled 0-3: +0.09 / -0.01, -0.01 / -0.02. FAR's shallow-loss flag was not raised.
  - The deep walk-strict bands are thin on MBPP (14-52 rows at 23-30), so they are not read.
- Reading:
  - The tree's gain is the chain's gain. Tree-32 minus chain-32 (the off-path continuation FAR aimed at) moves
    1.646 -> 1.659 / 1.687 tokens per block on MBPP and 1.823 -> 1.781 / 1.770 on HE.
  - The training lifted rows that are already dead: marginal accuracy at depths 23-30 goes 54.97 -> 60.01 / 60.29
    (MBPP) and 67.40 -> 70.31 / 70.40 (HE). That is MTP Head's ALLROWS pattern (deep bands up, kept flat) at a
    smaller w.
  - MBPP's mid-depth walk-strict gain (d3-8 +1.0 to +3.4 pt) may come from the depth weights rather than from the
    post-miss rows' content. Each depth's weight is its weighted row count, so the missed rows also move loss weight
    to the deep depths.
  - The CIs cover windows only (one training run per arm). Earlier lever arms moved MBPP by about 1.5 % with HE going
    the other way, so this MBPP-only gain is within that spread. In practice it leans KILL.
- Possible follow-ups (NewIdeas' call):
  - a tree-aware mask: a missed row stays alive only when its true token was in the top-k at the miss depth, where a
    tree still walks on;
  - a depth-weight-only control.
- Run: 19:06-20:25.
  - trainA (ctl + mix3): 1,759 s, 4.38 s/step; MLX peak 4.3 GB, footprint 4.75 GiB.
  - trainB (mix5): 1,087 s, 2.71 s/step; MLX peak 3.4 GB, footprint 4.17 GiB.
  - eval 7 min, tree 21 min. Swap unchanged (7,594 MB); share lock "MEM 5 GB cap" released.

## TREEMASK: the tree-aware mask for the post-miss rows (WALK-MIX's follow-up; pre-registered Mon 16 Nov 20:4x, before any data)

NewIdeas' go (20:3x) for Sandbox98's follow-up (a) to WALK-MIX: keep a missed row alive only when its true token was in
the head's top-k at the miss depth. LAB = teacher-forced on the 2B, not real decoding.

- Why: WALK-MIX lifted rows that were already dead (marginal 23-30 +3 to +5 pt), not the tree's off-path part
  (tree-32 minus chain-32: MBPP +0.01 / +0.04, HE -0.04 / -0.05 tokens per block). A tree walks on after a miss only
  when the true token is among the few siblings it holds at that depth. The mask trains exactly those rows.
- Mechanism:
  - Survivors keep t30b's accept-until-fail (KL weight 1, the failing depth included).
  - A row that misses stays in the chain when its true token's rank under the head (the tokens with a strictly
    higher logit) is < k. From the next depth on it trains at KL weight w = 1, and the same rule applies at every
    later depth. A miss at rank >= k drops the row, as today.
  - k = 1 is today's loss exactly: the hit decides, never the rank.
  - Code: sb/core.py ttt_step + _kl_hit_rank, gated on the hook markers _treemask_k / _treemask_w (every other run
    is unchanged); levers/treemask1.py / treemask2.py / treemask4.py.
  - Toy test (CPU, scripts/test_treemask.py): the KL, hit and gradient equal _kl_hit's bitwise; the rank equals a
    numpy reference; k = 1 keeps exactly the hits.
- Arms: tm2 (k 2, w 1) and tm4 (k 4, w 1), each against ctl.
- Runs (scripts/drift_run.sh plans/treemask.sh; share lock "MEM 5 GB cap"):
  - sb/train_multi.py, 400 steps at depth 30, from s30o with its Adam (WALK-MIX's recipe).
  - Process A trains ctl and tm2 (tag tmask); process B trains tm4 (tag tmaskb). The harness is deterministic, so
    B's slices, order and start equal A's.
  - Smoke first: 3 steps of ctl / k1 / k4 in one process.
  - Start window: no start before Tue 17 Nov 15:00 (NewIdeas: Tuesday by day, after the night order) unless
    logs/treemask_GO appears; no step starts after Tue 22:00 (finished steps are kept; a re-arm resumes).
- Read-out:
  - sb/evalx.py strict per depth on MBPP-50 / HE-30 (runs s30, ctl, tm2, tm4).
  - Tree-32 kept: MTP Head's ~/TrainLab/sb98/tree8.py, read-only (sha256 b7e95b92c1cd...); gamma 1, with 0.5 / 2 as
    the band; window-bootstrap CIs on each arm / ctl ratio.
  - sb/treemask_verdict.py applies the lines.
- LINES (NewIdeas 20:3x), per arm:
  - PASS: tree-32 kept (gamma 1) >= x1.03 of ctl on MBPP-50 AND HE-30, with the shallow line held on both sets.
    The shallow line: pooled walk-strict at band depths 0-3 (the first four guesses; d0 = the first guess) AND at
    band depths 1-4, each within -0.2 pt of ctl. Both readings of "d1-d4" must hold, so the line does not hang on
    the naming.
  - KILL: tree-32 kept < x1.01 on HE.
  - BETWEEN otherwise.
- Information:
  - the off-path part (tree-32 minus chain-32) per arm minus ctl, with a paired window-bootstrap CI: the mechanism's
    own read, which should grow;
  - FAR's shallow-loss flag (> 0.5 pt down on either reading);
  - the control against s30 (MBPP 7-14 marginal not down > 1.5 pt);
  - whether ctl's weights equal WALK-MIX's ctl (sha256; the harness is deterministic, so they should). If they do,
    each arm's off-path part is also read against WALK-MIX's mix5 (k = all, w 0.5).
  - Under this KILL line, WALK-MIX would read KILL for both arms (HE x0.997 / x0.992).
- VOID if:
  - the smoke's k1 arm does not equal ctl (per-step loss within 1e-4 relative), or k4 does not differ (> 1e-3 at some
    step);
  - tree8's self-check fails.
- Not run: (b), the depth-weight-only control (NewIdeas: a diagnostic, only if (a) passes).

## B98S1: BREAK98 Screen 1, FAR's FUTURE-STATE ROWS on the 2B (orchestrator 07:3x; pre-registered Tue 17 Nov 08:0x, before any data)

Fable FAR's idea (~/FableGod/far/IDEAS_1117_futurestates.md, "Screen 1"); the lines are FAR's, as the orchestrator relayed
them. LAB = teacher-forced on the 2B, not real decoding.

- Question: does a FROZEN model's state at a placeholder row (no token) carry its plan for that position? On the 125B,
  K placeholder rows hung under a check row would give the next block's head the 125B's state at its first K drafted
  positions for ~0.28 ms per row (FAR's price). The 2B analog answers this before any 125B capture.
- Mechanism (sb/futstate.py; core.fut_in / depth1 / deeper; levers/futA.py / futB.py / futC.py):
  - The frozen 2B runs [ids 0..s] + K = 8 placeholder rows: a chain at positions s+1..s+8, plain causal attention over
    the true prefix and the chain. Its post-norm state at offset j is W_fut[s, j].
  - The head at depth j <= 8 adds P(norm_f(W_fut[s, j])) to its fused input, with P zero-initialised (step 0 = the head
    exactly). Depth j predicts ids[s+j+1], the token offset j's own lm_head targets (FAR's runtime map).
  - Past depth 8, and at anchors without states, there is no input (today's head). At depth 1 only the anchors' query
    rows read it; the true pairs' keys / values stay future-free.
  - Toy test (CPU, scripts/test_fut.py, 12 checks PASS):
    - P = 0 gives the loss bitwise and every gradient to rounding (worst 1.7e-7 relative).
    - With P != 0 the two-pass gradient equals the one-graph autograd (worst 4.4e-7).
    - Depth j reads offset j; there is no input past depth 8.
    - The read-out chain with P = 0 is bitwise the plain chain.
- Data (sb/futgen.py): lockstep batches of files on the target's own caches, with a snapshot / restore per anchor.
  - Harness check (variant T = the true tokens instead of placeholders; 4 HE files, 1,117 anchors): the states match the
    stored true W at offsets 1-8 (cos mean 0.9998, min 0.996; top-1 agreement 99.4-99.8 %).
  - States are made for the 33,844 training anchors (the 400 slices), 11,517 MBPP-50 and 7,941 HE-30 anchors: ~1.75 GB
    per variant, deleted after that variant's read-out.
- Variants, one arm each: A <|fim_pad|> (id 248063, the 125B's id too); B the mean of the input-embedding table; C
  <|vision_pad|> (248055; the "any foreign vector" control).
- Runs: scripts/drift_run.sh plans/b98s1.sh, share lock "MEM 5 GB cap".
  - The runner now has the at-Mac yield: with HIDIdleTime < 900 s no step starts, and a running step yields.
  - Each arm runs alone (tag b98s1V, arm fV): sb/train_multi.py, s30o + its Adam, 400 steps at depth 30 on pool2, cosine
    1e-5 -> 1e-6 (WALK-MIX's setup). P and norm_f train on their own Adam at 1e-5, constant after the 20-step warm-up
    (the tapmlp / cap2x convention).
  - Control: WALK-MIX's wmix_ctl (the same args; the harness is deterministic). It is read out again here, in chain and
    true-state modes. If its chain read-out does not equal WALK-MIX's exactly, tree8 re-reads its tree.
  - Smoke first (3 steps, ctl + fA in one process): fA's step-1 loss must equal ctl's (1e-6 relative), and P must move.
- Read-out:
  - sb/evalx.py walk-strict chain, MBPP-50 / HE-30.
  - G = (arm - own) / (true - own) per band, where own = ctl's chain and true = ctl with the true state at every drafted
    depth (the SIB / driftg definition). Paired window bootstrap (2000, seed 1116).
  - Bands are evalx's labels: band d = chain depth d + 1, so the future states feed depths 1-8 = bands 0-7.
  - FAR's bands: 1-2, 3-6, 7-8 and 1-8. Band 8 (depth 9) gets no direct input. Band 0 has no own -> true gap, so it is
    reported as a delta only.
  - Tree-32 kept: MTP Head's ~/TrainLab/sb98/tree8.py, read-only (sha256 7cbe1643ab13). Arm / ctl ratio with a window
    bootstrap (5000, seed 1115 per comparison); gamma 1, with 0.5 / 2 as the band.
  - sb/b98s1_verdict.py applies the lines.
- LINES (FAR's), per variant:
  - PASS: HE G(1-8) >= 0.30, with its 95 % CI's lower end > 0, AND HE tree-32 kept >= x1.05 of ctl. FAR's line names HE
    for G; I read the tree part on HE too. MBPP is printed.
  - KILL: HE G(1-8) < 0.10.
  - BETWEEN otherwise.
  - Screen: PASS if any variant passes. KILL only if all three are KILL; FAR: then a frozen model's placeholder rows
    carry no plan, and only a LoRA'd 125B could (out this week).
- Information:
  - MBPP; band 0's delta; G at 9-14 and 15-30; chain-32, tree-16 and the gamma band.
  - FAR's zero-shot print: the 2B's own top-1 at placeholder offset p vs ids[s+p+1], by offset, beside ctl's depth-1
    accuracy. Offset 1 far below the head's depth 1 = a weak plan.
  - |P|_F / |fc's W columns|_F per arm. Below 0.01 flags "may be under-trained at 1e-5"; NewIdeas decides on a re-run at
    3e-5.
- VOID if: the smoke fails; tree8's self-check fails; a step fails.
- Not run:
  - Coverage: every anchor carries a chain here. FAR's coverage (71-92 % of HE+ block ends) scales the 125B number.
  - FAR's 4th variant (the head's soft token at offset 1): only if A-C all land BETWEEN.
- Correction (records): WALK-MIX's tree read-out (20:25) ran on tree8.py as MTP Head changed it at 19:55 (sha256
  7cbe1643ab13, TrainLab 04e1cc6), not on the b7e95b92c1cd its pre-registration named (13:17, 96baf2c). The change only
  adds a '!s' option (a vocab-screened read); plain specs run the same code path (git diff checked). TREEMASK's section
  names the old sha too; it will run on 7cbe1643ab13.
- TREEMASK is parked: its runner was stopped at 07:48 before its window (NewIdeas: it stays parked behind BREAK98 work).

## LANDMARK: Breakthrough's LANDMARK-ANS screen 1 on the 2B (NewIdeas' queue after B98S1; Breakthrough owns the pre-registration; my reading written Tue 17 Nov 08:1x, before the run)

Sources: ~/Breakthrough/docs/IDEAS_1117_landmark.md ("Screens" item 1; the ADDENDUM, i.e. the orchestrator's audit; the
"Screen 1 alignment clarification", 08:05) and NewIdeas 07:4x-07:50 (G by match length; the idea lives only if G at
k <= 3 clears KILL-COPY). LAB = teacher-forced on the 2B, not real decoding.

- Mechanism (eval-only, frozen s30 head; sb/landmark.py):
  - The lookup is Breakthrough's ~/Breakthrough/scripts/break98/landmark_source.py, imported read-only (sha256
    1992503181c0).
  - Band d's cell (anchor s) reads an input state for p = s + d plus the token ids[p+1], and predicts ids[p+2].
  - The cell is covered when the landmark q <= s exists: the most recent ANSWER position whose last k tokens equal p's,
    with k = 8, 6, 4, 3, 2, longest first.
- Modes, each on its own chain:
  - own: the carried state.
  - true: the true W at every drafted depth.
  - lm1: the input state := W[q] at covered cells.
  - lm05: 0.5 W[q] + 0.5 carried at covered cells.
  - stale1: the input state := W[p-1], the true state one position back.
  - copy (i): ids[q+2] predicts ids[p+2].
  - copy (ii): re-matched lookup decoding at p+1 (q' <= s) gives ids[q'+1]; no re-match = the head's own call.
  - copy1 (info): ids[q+1] replaces band d-1's prediction.
- Read-out (HE 60 + MBPP 100):
  - Walk-strict over covered cells only, bands 1-2 / 3-6 / 7-8 / 1-8 (9-14 and 15-30 as info).
  - G = (mode - own) / (true - own), with a paired window bootstrap (2000, seed 1116).
  - Split by the longest matched k: k >= 4 vs k <= 3, and each k as info.
  - Info: all-cell walk-strict and whole-walk kept per 32-row check (the coverage priced as it falls); coverage per band.
  - Info: the angle between W[q] and W[p] (the share within 0.40 rad, beside MoonShots' 3-6 %), and between W[p-1] and
    W[p].
- LINES (Breakthrough's + the ADDENDUM's), per arm (lm1, lm05), on HE G(1-8) over all covered cells:
  - KILL: G < 0.10.
  - KILL-COPY: G <= G_copy + 0.10, where G_copy = the stronger of copy (i) / (ii).
  - PASS: G >= 0.50 on HE AND >= 0.25 on MBPP (my reading of "G >= 0.50 on HE (>= 0.25 MBPP)").
  - BETWEEN otherwise.
  - WARN (report only): G < 0.5 x G_stale1.
  - Screen: PASS needs an arm PASS whose k <= 3 cells also clear KILL-COPY (NewIdeas' rule). An arm PASS that clears
    KILL-COPY only at k >= 4 reads "BETWEEN (k >= 4 only)". KILL / KILL-COPY if both arms are.
- Toy test (CPU, scripts/test_landmark.py, 7 checks PASS):
  - The tables and copy (ii)'s re-match equal a brute-force search.
  - The stand-ins replace, mix or read W[p-1] exactly, and leave uncovered rows bitwise.
  - cov_counts is right on a hand example.
- Run: scripts/drift_run.sh plans/landmark.sh, chained to start once B98S1's END marker exists. One eval process,
  <= 5 GB (MLX cap 4.5 GB).

## B98S1 variant A result (Tue 17 Nov 08:34) = KILL, and negative (results/b98s1_verdict_A.json, b98s1_eval_A.json, b98s1_tree_A.json, b98s1_zeroshot_A.json; the control's read-out: b98s1_ctl_eval.json)

LAB = teacher-forced on the 2B, not real decoding. Variant A = <|fim_pad|>, the placeholder the 125B capture would use.
The control is wmix_ctl.

- VOID checks are clean:
  - The smoke: fA's step-1 loss 0.01819732 equals ctl's, and |P| was 3.5e-3 after 3 steps.
  - The control's chain read-out equals WALK-MIX's exactly (DETCHECK SAME), so WALK-MIX's ctl tree is the tree control.
  - tree8's self-check PASSes on both sets.
- Result (arm minus control; G = the share of the own -> true gap closed, 95 % CI over windows):

  | set | G(1-8) | 1-2 | 3-6 | 7-8 | depth 1 (band 0) | tree-32 kept | chain-32 | tree-16 |
  |---|---|---|---|---|---|---|---|---|
  | HE-30 | -0.105 [-0.145, -0.060] | -1.11 pt | -0.93 | -0.32 | -0.15 | 7.825 -> 7.513 = x0.9601 [0.9428, 0.9781] | x0.9585 | x0.9669 |
  | MBPP-50 | -0.072 [-0.10, -0.05] | -0.83 | -1.29 | -0.03 | -0.31 | 5.911 -> 5.734 = x0.9700 [0.9584, 0.9807] | x0.9771 | x0.9784 |

  The control's own -> true gap at 1-8: HE 86.92 -> 95.31, MBPP 77.80 -> 90.45.
- Zero-shot (FAR's print): the frozen 2B's own top-1 at placeholder offset 1..8, against the token that row predicts:
  - HE: 26.3 / 9.5 / 7.8 / 7.1 / 6.4 / 6.8 / 6.6 / 6.8 %.
  - MBPP: 22.2 / 9.5 / 7.7 / 6.9 / 6.4 / 6.0 / 6.3 / 6.4 %.
  - The head's depth-1 accuracy is 93.3 (HE) / 88.2 % (MBPP). By FAR's own reading, offset 1 far below the head's
    depth 1 means a weak plan.
- |P|_F / |fc's W columns|_F = 0.019. That is about the size a 400-step Adam random walk at 1e-5 reaches, so P found no
  consistent signal, and the input entered as noise: the arm is below the control at every band from depth 1 to 8.
- Reading: a frozen model's state at a <|fim_pad|> row does not carry its plan for that position in a form the head can
  use; the row mostly knows it sits on an unknown token. B and C are pending; the screen is KILL only if all three are.

## B98S1 result (Tue 17 Nov 09:21) = KILL, all three variants, every one negative (results/b98s1_verdict.json; per variant b98s1_eval_V / tree_V / zeroshot_V / verdict_V)

LAB = teacher-forced on the 2B, not real decoding. The control is wmix_ctl; its tree is WALK-MIX's read-out of the same
weights (its chain read-out reproduced exactly). VOID checks are clean: the smoke, DETCHECK, and tree8's self-check on
every run. No yields; each variant ran gen ~13 min (12-16 ms per anchor), train 3-5 min, read-outs ~5 min.

| variant | HE G(1-8) | HE pt vs ctl, d0 / 1-2 / 3-6 / 7-8 | HE tree-32 kept | MBPP G(1-8) | MBPP tree-32 kept | P ratio |
|---|---|---|---|---|---|---|
| A <\|fim_pad\|> | -0.105 [-0.145, -0.060] | -0.15 / -1.11 / -0.93 / -0.32 | x0.9601 [0.9428, 0.9781] | -0.072 [-0.096, -0.047] | x0.9700 [0.9584, 0.9807] | 0.0190 |
| B mean embedding | -0.130 [-0.168, -0.090] | -0.14 / -0.96 / -1.19 / -1.33 | x0.9583 [0.9383, 0.9793] | -0.048 [-0.071, -0.027] | x0.9710 [0.9582, 0.9838] | 0.0184 |
| C <\|vision_pad\|> | -0.174 [-0.221, -0.123] | -0.23 / -1.25 / -1.74 / -1.42 | x0.9493 [0.9343, 0.9645] | -0.094 [-0.123, -0.064] | x0.9576 [0.9457, 0.9694] | 0.0190 |

(HE tree-32: control 7.825; MBPP 5.911. The P ratio is |P|_F / |fc's W columns|_F.)

- The line: KILL = HE G(1-8) < 0.10 on every variant. All three are below zero, with CIs that exclude zero.
- Zero-shot (FAR's print): the frozen 2B's own top-1 at a placeholder row, against the token that row predicts.
  - HE, offsets 1..8: A 26.3 / 9.5 / 7.8 / 7.1 / 6.4 / 6.8 / 6.6 / 6.8 %; B 26.3 / 10.9 / 7.1 / ...; C 26.0 / 9.2 / 7.7 / ...
  - MBPP: offset 1 is 21-22 %; offsets 2-8 are 6-11 %.
  - The head's depth-1 accuracy is 93.3 / 88.2 %.
  - All three placeholders read alike: the 2B treats each as an unknown token. From an unknown token it predicts the
    next one about a quarter of the time, and further out 6-11 %.
- Reading:
  - A frozen model's placeholder row carries little of its plan for that position in a form the head can use. The
    ordering (A least harmful, C, the never-seen token, most) follows how familiar the token is, not any plan.
  - FAR's own kill condition (1) holds: "Screen 1 KILL on all three variants: a frozen trunk gives no usable plan at a
    placeholder". Only a LoRA'd 125B could, which is out this week. The placeholder-row capture for the 125B is not
    needed.
  - Why negative rather than zero (my reading, not tested): P grew to the same ~1.9 % in every variant, about a
    400-step Adam random walk at 1e-5 (Adam steps ~lr whatever the gradient). So a zero-init input that carries no
    signal enters as noise.
    - Untested control: a pure-noise input arm.
    - Implication: on this setup a no-information zero-init input reads about -0.1 G, not 0.

## LANDMARK result (Tue 17 Nov 09:30) = KILL, both arms; every state swap hurts the frozen head (results/landmark_verdict.json; raw per-window counts results/landmark_s1.json.gz, sha256 of the json 511c51a6b31e9ee3)

LAB = teacher-forced on the 2B, not real decoding. Frozen s30 head, HE 60 + MBPP 100, 475 s. Covered cells: 37 % (HE)
/ 35 % (MBPP) of the cells at bands 1-8. Cells are walk-strict, each mode on its own chain.

| HE, covered cells, walk-strict 1-8 | own | true | lm1 | lm05 | stale1 | copy (i) | copy (ii) |
|---|---|---|---|---|---|---|---|
| all covered (27,789 cells) | 86.16 | 95.29 | 60.22, G -2.84 [-3.68, -2.24] | 62.85, G -2.56 | 73.79, G -1.36 | 53.33, G -3.60 | 70.20, G -1.75 |
| k >= 4 (12,264) | 86.46 | 96.28 | 71.40, G -1.53 | 71.76, G -1.50 | 73.71, G -1.30 | 72.77, G -1.39 | 74.26, G -1.24 |
| k <= 3 (15,525) | 85.93 | 94.60 | 50.43, G -4.09 | 55.16, G -3.55 | 73.84, G -1.39 | 34.34, G -5.95 | 66.64, G -2.22 |
| k = 8 (4,255) | 88.70 | 97.58 | 82.51, G -0.70 | 82.70, G -0.67 | 80.44, G -0.93 | 85.53, G -0.36 | 86.35, G -0.26 |

- MBPP, all covered (35,241 cells): own 80.86, true 91.58; lm1 58.36 (G -2.10), lm05 60.10 (-1.94), stale1 66.02
  (-1.38), copy (i) 54.10 (-2.50), copy (ii) 65.59 (-1.42).
- Whole-walk kept per 32-row check (HE / MBPP): own 6.431 / 4.421; true 12.744 / 8.049; lm1 4.768 / 3.705 (-26 % /
  -16 %); lm05 4.899 / 3.747; stale1 5.515 / 3.981; copy (i) 4.511 / 3.576; copy (ii) 5.415 / 3.895.
- Angles (the 2B's head-input state):
  - Landmarks within 0.40 rad of W[p]: 12.6 % HE / 9.7 % MBPP (cos median 0.716 / 0.731). By k on HE: k8 39 %, k6
    20 %, k4 13 %, k3 8.5 %, k2 5 %.
  - The true state one position back: 0.0 % within 0.40 rad (cos 0.518 / 0.539).
  - Copy (ii) re-matched at 63-65 % of covered cells.
- Lines:
  - KILL on both arms (G < 0.10) on both sets.
  - The k <= 3 cells also fail KILL-COPY (lm1 -4.09 vs copy -2.22).
  - The WARN fires (G < 0.5 x G_stale1).
  - Screen = KILL.
- Reading:
  - The ADDENDUM's testable claim fails: this head does not handle one step of staleness. Fed the TRUE state one
    position back at a covered cell, the frozen head falls from 86 % to 74 % on HE (G -1.36). Its own carried state for
    the right position beats a stale true one.
  - A landmark is closer in angle to W[p] than the 1-back state (cos 0.716 vs 0.518; on the 125B bank 0.725 vs 0.619),
    yet it reads worse (60 % vs 74 %). Angle closeness is not what the head uses.
  - Even 8-token matches hurt (k8: 82.5 vs 88.7). The landmark state carries less than lookup decoding (copy (ii)
    -1.75), and far less than the head's own call.
  - Untested: a head TRAINED to read a landmark beside its carried state (zero-init); this frozen screen bounds only the
    plug-in use. B98S1 showed a zero-init input without signal reads about -0.1 G, so such a screen needs a real signal
    to show.
- Memory correction (WALK-MIX last night, B98S1 and LANDMARK today):
  - The read-out steps peaked above the "MEM 5 GB cap" their share locks declared, in lifetime phys_footprint
    (/usr/bin/time -l): evalx 6.4-7.3 GB, tree8 7.1-7.5 GB, this landmark run 7.3 GB, and WALK-MIX's 2-arm training
    5.1 GB.
  - Generation and 1-arm training stayed at 3.7-4.4 GB.
  - Swap stayed flat (7,493 MB) and NO-SWAP was 33-41 GB throughout.
  - From now on, read-out steps lock 8 GB (and 2-arm training 6 GB), with the NO-SWAP start gate at the lock + 8.

## B98C: BREAK98c, FAR's PREFIX EYES on the 2B (orchestrator 11:2x; pre-registered Tue 17 Nov 11:3x, before any data)

Fable FAR's idea (~/FableGod/far/IDEAS_1117c_prefix_eyes.md, section 1, "The one decisive test"); the lines are FAR's, as
the orchestrator relayed them. LAB = teacher-forced on the 2B, not real decoding.

- Question: does the frozen target's OWN view of the verified prefix (its attention keys / values at a mid-late layer)
  give the head information that cuts its drift at depth? On the 125B the head would read the 125B's K/V cache of the
  prefix (final, already in memory; FAR prices ~12 MB per pop at a 2K prefix). The 2B analog answers it first.
- Mechanism (sb/kvx.py; core.kvx_in / block / depth1 / deeper; levers/kvx3.py, levers/kvx10.py):
  - The head's one decoder layer gains a cross-attention block between its self-attention and its MLP (GLIDE's decoder
    form). A row with anchor s, at any depth j, reads the frozen trunk's cached K / V of the verified prefix (positions
    0..s) at the full-attention layers 15 and 19 (16 / 24 and 20 / 24 of the depth, the two attention layers around 3/4;
    FAR: "one layer; two if cheap", and the cache costs ~1 GB of disk, so two).
  - Per trunk layer: query = q_x(norm_x(h)) in the trunk's gated geometry (8 query heads over its 2 KV heads, head dim 256,
    the per-head output gate), q-norm, RoPE at the head's own position s + j - 1 over the trunk's cached (normed,
    RoPE'd) keys and values at positions <= s; output o_x(attn x sigmoid(gate)), added to the residual.
  - Init: norm_x, q_x (with its gate half) and the q-norm from the trunk layer's own input_layernorm, q_proj (dequantized
    from 4-bit) and q_norm, so the read starts from the trunk's own attention pattern; o_x = 0, so step 0 is the head
    exactly.
  - FAR's caveat: ctl keeps the native head's own attention over its true pairs (W[u], ids[u+1]), u <= s, i.e. the
    trunk's FINAL features of the prefix (the EAGLE form). The arm adds only the mid-layer K / V.
  - At depth 1 only the anchors' rows read it. The true pairs' self-attention keys / values are computed before the
    block, so they are unchanged.
- Data (sb/kvxgen.py): one capture-shaped forward per file (batch 1, whole length). Only layers 15 and 19 get KVCache
  objects; every other layer runs as the capture did. Output: data/kvx/<dir>/<key>.safetensors, K and V [2 layers, 2 KV
  heads, S, 256] bf16.
  - Files: the 297 training files of the 400 slices (217,491 rows) plus MBPP-50 / HE-30 (31,203 rows), ~1.0 GB.
  - This is the run's own scratch, deleted by the same run after its read-outs.
  - Check on the first files: the forward's final post-norm state equals the stored W.
- Toy test before the run (scripts/test_kvx.py):
  - o_x = 0: loss bitwise; gradients equal to rounding.
  - o_x != 0: the two-pass gradient equals the one-graph autograd.
  - A row with anchor s does not see trunk K / V rows > s.
  - The block with the trunk's own layer input, q_proj and o_proj reproduces the trunk's layer-15 / 19 attention output
    (RoPE, cache and mask conventions).
  - The chain with o_x = 0 is bitwise the plain chain, and the new core is bitwise the committed one with no hook.
- Arms: both in one process, on the same slices in the same order.
  - Setup (sb/train_multi.py, tag b98c): s30o + its Adam, 400 steps at depth 30 on pool2, cosine 1e-5 -> 1e-6. This is
    WALK-MIX's setup, the same args as wmix_ctl.
  - The new block trains on its own Adam (prefix kvx_), at a constant rate after the 20-step warm-up:
    - kvx3: module lr 3e-5, the harness's rate for a new attention block (deep2 / deep3).
    - kvx10: module lr 1e-4. 400 steps is short for a block that starts at zero, so this arm guards against reading
      "too slow" as "no information".
  - Control: WALK-MIX's wmix_ctl (the same args; the harness is deterministic).
    - Its chain + true read-out on the same windows is results/b98s1_ctl_eval.json, which equals WALK-MIX's exactly.
    - Its tree is results/walkmix_tree.json.
  - Smoke first: 3 steps, ctl + kvx3 in one process.
    - kvx3's step-1 loss must equal ctl's (1e-6 relative).
    - ctl's 3 losses must equal wmix_ctl's logged ones (the core change leaves the training path as it was).
    - o_x must move.
- Read-out:
  - Chain: sb/evalx.py walk-strict chain on MBPP-50 / HE-30, both arms.
  - Trees: MTP Head's ~/TrainLab/sb98/tree8.py, read-only (sha256 7cbe1643ab13): tree-32 / tree-16 / chain-32.
  - G = (arm - own) / (true - own). own = ctl's chain; true = ctl with the true state at every drafted depth.
    Walk-strict is pooled over the band's depths; paired window bootstrap (2000, seed 1116).
  - Bands are evalx's labels (band d = chain depth d + 1). FAR's "bands 9-30" = labels 9..30 pooled. "Every band" = 1-2,
    3-6, 7-10, 11-14, 15-22 and 23-30 (band 0 has no own -> true gap).
  - Tree ratio arm / ctl: tree8's estimator (ratio of sums, 5000 window resamples, seed 1115).
  - sb/b98c_verdict.py applies the lines.
- LINES (FAR's), per arm. HE is the primary set, as in B98S1; MBPP is printed beside it:
  - PASS: HE G(9-30) >= 0.30 AND HE tree-32 kept >= x1.08 of ctl.
  - KILL: HE G < 0.10 at every band.
  - BETWEEN otherwise.
  - Screen: PASS if either arm passes; KILL only if both are KILL.
- Information:
  - G's CIs per band; MBPP's G and tree ratio; tree-16, chain-32 and the gamma band.
  - |o_x|_F / |o_proj|_F per layer and arm (how far the block grew); the training loss against wmix_ctl's.
- VOID if: the toy test or the smoke fails; tree8's self-check fails; a step fails.
- Not run:
  - FAR's second input (a copied prompt span's prefill states replace the carried state). FAR gives it its own 10-min
    join once a capture holds prefill states.
  - Which of the two layers carries a gain (only on a PASS).
- Amendment (Tue 17 Nov 11:5x, before any data): SPLIT CLIP. sb/train_multi.py clips every gradient together (global norm
  1.0), so a new module's gradient can shrink the head's own steps. ctl has no such module, so this would bias an arm low.
  - From now on, a lever with SPLIT_CLIP = True (sb/kvx.py) clips the head's gradients apart from its module's, each at
    1.0. With o_x = 0, the head's first step is then bitwise ctl's.
  - Earlier levers keep the shared clip. B98S1's arms shared it, which is one possible part of their small negative G
    (untested).
  - The training log keeps each step's two gradient norms (gnorm), as information.
  - The smoke line becomes: kvx3's step-1 loss bitwise equals ctl's, ctl's 3 losses equal wmix_ctl's logged ones, and
    o_x moves.
- Harness checks (before the run; scripts/test_kvx.py, logs/b98c_toy.log): 15 / 15 PASS.
  - o_x = 0: the loss is bitwise, and every shared gradient is bitwise too.
  - Two-pass gradient vs one-graph autograd: worst 3.2e-7.
  - Causality holds bitwise.
  - The new core with no hook is bitwise the committed one.
  - Real 2B (GPU, 64 tokens of an HE file):
    - kvxgen's capture path gives a final state bitwise equal to core.target_states'.
    - With the trunk's own layer input, q_proj and o_proj, the block reproduces the trunk's attention output: layer 15
      rel err 5.3e-3, layer 19 1.1e-2, min row cos 0.9998. RoPE, cache, gate and mask conventions match.
  - On the CPU the 4-bit trunk takes > 10 min for 64 tokens (a slow reference matmul), so (g) runs on the GPU (~1.7 GB).

## CHAN: Lunatic's CODE || COMMENTARY quality pre-screen on the 2B (orchestrator 11:5x; pre-registered Tue 17 Nov 12:0x, before any data)

Lunatic's idea (~/FableGod/lunatic/IDEAS_1117c_channels.md, sections 3.5-3.6): the code decodes as one causal channel
that never reads the answer's comments or docstrings, and each comment / docstring is a branch forked at its slot. The
orchestrator asks for the 2B pre-screen named in 3.5 ("the same run on the 2B in Sandbox98"). This is a QUALITY
pre-screen only: adopting the output form is Kamen's call. The lines below are my reading of Lunatic's spec, written
before any data and sent to Lunatic. Real decoding of the 2B (not LAB).

- Panel, prompts and caps: as MoonShots' J(CC) (LEVERS_A_MOONSHOTS.md section J(CC)).
  - EvalSuite's code panel: 264 keys = HumanEval+ 164 + MBPP 100.
  - The items' user text in the 2B's chat template, thinking off (sb/gen.chat_ids); the items' max_tokens; an answer
    stops at an end token.
  - One deterministic greedy draw per arm.
- Coupled arm = J(CC)'s exact arm (MoonShots' week98/jcc/arm0.jsonl, its checks and its EvalSuite scores), read-only.
  - It is the 2B's own batch-1 greedy, with s30 drafting and 32-row checks.
  - Condition: my loop, with the split off, must reproduce arm0's answer ids exactly on the smoke keys. If it doesn't,
    I run my own coupled arm.
- CODE || COMMENTARY arm (sb/chan.py; the same batch-1 loop, s30 drafts, 32-row checks, exact):
  - Trunk = the code channel: the prompt plus the answer's code only.
    - A slot is where the committed token would open a '#' comment or a docstring, by MoonShots' span tracker
      (week98/span_tracker.py, read-only): the token's characters take the tracker into a comment or docstring.
    - At a slot, the trunk commits its best token that opens neither (the target's logits in order). A draft is kept
      only if it equals that token.
    - After a slot, openers stay masked without a new slot until the trunk commits a non-blank character, so one slot
      cannot fork twice.
  - Branch, one per slot. Its context is the prompt, the trunk's tokens before the slot, and the opener; it does not
    read earlier comments.
    - It decodes greedily (the same draft / check loop) until the tracker leaves the comment or docstring.
    - Caps: 128 tokens for a '#' comment, 512 for a docstring. A capped docstring is closed.
  - Output: each branch is spliced into the code at its slot.
    - Full-line comments and docstrings go at the slot's line, with its indentation.
    - Inline comments go at the end of the slot's line.
- Scoring: EvalSuite's scorers, read-only, as J(CC). HE+: evalplus base + plus tests. MBPP: the 3 asserts.
- LINES:
  - Quality, on the code channel (comments do not execute; the spliced answer is scored too and must agree, any
    difference listed).
    - These are J(CC)'s lines, the pre-screen Lunatic's 3.5 names. Pass@1 vs the coupled arm, paired by key over the
      264 keys:
    - HOLDS: the pooled drop is <= 1.0 pt AND the exact McNemar p > 0.2 AND neither set loses with p < 0.05.
    - FAILS: the arm is lower with p < 0.05 pooled, OR the pooled drop is > 3 pt.
    - BETWEEN otherwise.
  - Speed claim (Lunatic's 3.5): the code channel's check-ending misses per code token against the coupled decode's
    (misses at code + string tokens per code + string token). KILL the speed claim if > 1.5x.
- Reported, with no line:
  - Sweeps: coupled checks vs the channel critical path. The critical path is the max of the trunk's checks and, per
    branch, the trunk's check count at its slot plus the branch's checks. Also the fewer-sweeps ratio and the live
    channels per sweep (mean, p90).
  - Token agreement with the coupled answer's code: comments and docstrings stripped, blank lines dropped. The share of
    answers with identical code, the line-level match ratio, and the code position of the first difference.
  - Comment consistency (Lunatic's guard): in each branch, the 2B's own next line after the comment vs the trunk's next
    code line (the share of slots that agree).
  - Slots per answer, branch caps, cap hits. The numerics noise floor from J(CC)'s control B (+0.76 pooled, 34
    discordant keys).
- Caveats:
  - The 2B is a harsh proxy for the 125B (J(CC) says the same).
  - A branch here is a fresh prefill, not a forked cache, so its numerics differ at near-ties.
  - A FAIL here is evidence against the form on the 2B, not a 125B verdict. EvalSuite's gate decides.
- VOID if the loop's identity check fails and no coupled arm of my own is run, or a scorer fails.
- Amendment (Tue 17 Nov 12:1x, before any data; Lunatic's 12:0x reply). Lunatic confirms the reading: no change to the
  lines, arms or scoring. Two additions:
  - When the code channel's and the spliced answer's scores differ, the verdict reads from the SPLICED answer (what a user
    gets). The code channel's line is printed beside it.
  - Three diagnostics, no line:
    - the coupled arm's check-ending misses by class;
    - INDENT CARRY: full-line slots where the trunk's next code line indents differently from the branch's own next
      line, i.e. the trunk committed the comment's indentation before the masked '#';
    - rows per sweep (32 x the live channels; mean, p90).
- Harness (before the run):
  - sb/chan.py: the decode. MoonShots' tracker logic is copied, and the class replay of arm0 equals arm0's own class
    counts on all 264 keys (sb/chan_cls.py).
  - sb/chan_score.py: the scorer + verdict, tested on a no-change fake arm (16 keys): HOLDS, x1.0, and EvalSuite's fresh
    scores of the same texts equal MoonShots' cached ones.
  - scripts/chan_par.sh: two processes, HE+ / MBPP.
  - plans/chan.sh: it starts at B98C's end marker. A tracker fix before any run: a docstring's span lasts to its third
    closing quote (the tracker's in_span() turns False at the first).
  - The splice is unit-tested on synthetic code: full-line, docstring, inline and two-slot cases.
- Information already in hand (the coupled arm, no CHAN data): the 2B's exact arm ends its checks at comments 29.5 %,
  docstrings 37.2 % and code + string 33.2 % of the time (12,883 check-ending misses). Comments + docstrings = 66.7 %,
  against Lunatic's 66 % (HE+) / 74 % (MBPP) on the 125B's cells, so the 2B has the same structure.
- Smoke check corrected (Tue 17 Nov 12:1x, before the training; no B98C read-out exists). The 12:13 smoke went VOID on
  my own check: "ctl's 3 losses equal wmix_ctl's logged ones" cannot hold. A 3-step run's cosine schedule spans 3 steps,
  so its step-2 update uses lr x0.55 where the 400-step run used x1.00, and step 3's loss differs (0.39782 vs 0.39254).
  - What the smoke showed:
    - ctl's steps 1-2 equal wmix_ctl's logged losses bitwise (0.018197324576478137, 0.27542117043012615), so the core
      change leaves the training path as it was;
    - kvx3's step-1 loss is bitwise ctl's;
    - o_x moved (|o_x|_F 1.10e-2 per layer after 3 steps).
  - The check now compares steps 1-2 only. The runner re-armed and reused the smoke's own logs; the step was not re-run.
  - Gradient norms in the smoke (head, block): 6.5 / 1.9, 34.9 / 12.2, 68.4 / 25.0. The head's clip is always active,
    so a shared clip would have shrunk the head's steps by ~4-13 %; SPLIT CLIP prevents that.
  - Correction (12:16): the re-armed check itself ran vacuously. My edit put an apostrophe inside the shell-quoted Python,
    which cut it short, so it printed nothing and exited 0. The runner went on to training (started 12:15).
    - Checked by hand at 12:16 (logs/b98c_smokecheck.log): ctl's steps 1-2 bitwise equal wmix_ctl's logged losses, and
      kvx3's step-1 loss is bitwise ctl's (0.018197324576478137). o_x moved in the 12:13 check.
    - The plan's comment no longer has an apostrophe. The smoke's ckpt files were deleted by the step, as planned.
    - Whether the trained checkpoints hold the block's weights is checked at the read-out: the verdict reads kvx_o*.

## B98C result (Tue 17 Nov 12:33) = KILL, both arms (results/b98c_verdict.json, b98c_eval.json, b98c_tree.json; ctl: b98s1_ctl_eval.json + walkmix_tree.json)

LAB (the teacher-forced 2B), MBPP-50 / HE-30, against wmix_ctl (the same 400-step setup). HE own -> true walk-strict at
bands 9-30 (labels; the drift band): 85.27 -> 97.26.

| arm (block lr) | HE G(9-30) [95 % CI] | HE G by band 1-2 / 3-6 / 7-10 / 11-14 / 15-22 / 23-30 | HE tree-32 (ctl 7.825) | MBPP G(9-30) | MBPP tree-32 (ctl 5.911) | verdict |
|---|---|---|---|---|---|---|
| kvx3 (3e-5) | +0.023 [-0.080, +0.135] | -0.08 / -0.02 / +0.01 / +0.02 / +0.03 / -0.01 | 7.711, x0.9854 [0.966, 1.008] | -0.038 | x0.9778 | KILL |
| kvx10 (1e-4) | -0.077 [-0.200, +0.046] | -0.11 / -0.03 / -0.01 / -0.03 / -0.18 / -0.22 | 7.621, x0.9740 [0.959, 0.991] | +0.029 | x0.9793 | KILL |

- Every band's G is below 0.10 for both arms, so FAR's KILL line holds. tree-16 x0.987 / x0.997; chain-32 x0.976 / x0.973.
- The block did grow and got gradients:
  - |o_x|_F / |o_proj|_F: 0.039 / 0.038 (kvx3, layers 15 / 19) and 0.111 / 0.103 (kvx10).
  - Median gradient norms, head / block: 37.7 / 3.2 (kvx3) and 38.9 / 1.0 (kvx10).
  - Its output was not useful: the training loss rose (last 100 steps 0.5384 / 0.5399 vs wmix_ctl's 0.5288).
- Reading: on the 2B, the trunk's own mid-late K/V of the verified prefix (layers 15 + 19, read through the trunk's own
  query weights) gives the head nothing its attention over the trunk's final states (the native head's true pairs)
  doesn't already have. The head's drift (HE 85.3 -> 97.3 own -> true at 9-30) is not a missing view of the prefix.
- Checks: tree8's self-check PASS on all four reads; the cache W check was bitwise on 3 files; data/kvx (1.02 GB, the
  run's own scratch) was deleted by the run.
  - Lifetime footprints vs locks: capture 2.53 (5), smoke 5.11 (7), 2-arm training 5.55 (7), evalx 6.48 (8), tree8
    7.17 GB (8). Training ran at 1.17 s/step for the two arms.
- Not tested:
  - more steps, or a q learned from scratch (here q starts from the trunk's own);
  - the other attention layers (3-11, 23);
  - FAR's second input (copy-source prefill states).

## CHAN result (Tue 17 Nov 12:48) = quality FAILS (by the 3-pt rule; p 0.36), speed claim stands (results/chan/chan_verdict.json, scores.json; the rows: chan_rows.json.gz, the jsonl files stay local)

Real decoding of the 2B, 264 keys, against J(CC)'s exact arm0. The identity check passed: my loop with the split off
reproduces arm0's ids and checks on the 3 smoke keys. The code channel's and the spliced answer's scores agree on all
264 keys.

| pass@1 | coupled | CODE ‖ COMMENTARY | delta | arm-only / coupled-only | McNemar p |
|---|---|---|---|---|---|
| pooled (264) | 36.74 | 33.71 | **-3.03** | 25 / 33 | 0.358 |
| HE+ (164) | 41.46 | 36.59 | -4.88 | 16 / 24 | 0.268 |
| MBPP (100) | 29.00 | 29.00 | +0.00 | 9 / 9 | 1.0 |

- LINE: FAILS. The pooled drop is 3.03 pt, past the 3-pt line by 0.03, while the McNemar p is 0.36 (not significant).
  J(CC)'s numerics-only control moved +0.76 pt with 34 discordant keys; here 58 keys are discordant.
- Mechanism, on HE+:
  - 7 of the 33 lost passes are answers whose code no longer parses.
  - HE+ code parses: coupled 158 / 164, spliced 148 / 164. MBPP: 93 -> 97 / 100.
  - The 2B re-emits the prompt's signature and docstring. With the docstring opener masked right after the signature,
    it writes the docstring's text as bare code, e.g. HumanEval/11: ">>> string_xor('010', '110')" lines, then the
    docstring again as a plain string.
  - Without those 7, HE+ would read about 16 / 17 discordant. MBPP, which has no prompt docstring to copy, is flat.
- SPEED CLAIM: stands.
  - Check-ending misses per code (+ string) token: 0.1651 coupled vs 0.1722 code channel = x1.043 (HE+ x1.029, MBPP
    x1.059; the line is 1.5).
  - Sweeps: 13,288 coupled vs 7,711 channel critical path = x1.72 fewer (HE+ x1.71, MBPP x1.74).
  - Live channels per sweep 1.52 (p90 2), i.e. rows per sweep 48.6 (p90 64).
- Information:
  - The code is identical to the coupled answer's in 32 / 264 answers (line match 0.49).
  - Cap hits: 16 trunk vs 26 coupled.
  - Tokens: 34,658 trunk + 33,996 branch vs 76,834 coupled.
  - Slots: 1.97 per answer (255 '#', 266 docstrings); 5 branch caps.
  - Comment consistency (the branch's own next line equals the trunk's): 25.6 % ('#' 14.1 %, docstrings 36.8 %).
  - Coupled miss split on the 2B: comments 29.5 %, docstrings 37.2 %, code + string 33.2 % (as Lunatic's 125B 66-74 %).
- INDENT CARRY (Lunatic's diagnostic) = 0 / 509 full-line slots.
  - The scorer's printed 0.988 was a bug: it measured the trunk's line from the slot position, after the indentation
    the trunk had already committed.
  - Recomputed from the stored rows (results/chan/indent_carry_fixed.json); sb/chan.py is fixed for later runs.
- Reading: on the 2B the code channel is as easy to draft (x1.04 misses per code token) and takes x1.7 fewer sweeps.
  The quality loss sits on HE+, where masking a docstring opener that the model wants to copy from the prompt pushes the
  2B to write that docstring as code.
  - An engine form would need a guard there: e.g. the trunk keeps a copied prompt docstring, or a masked docstring
    opener ends the trunk's current statement.
  - Untested. The 2B is a harsh proxy; EvalSuite's gate decides on the 125B.
