# Validity gate: does the sandbox reproduce the 125B's day-1 pattern?

Pre-registered Sat 14 Nov 2026 ~18:40, before any drafter was trained or evaluated (only the untrained native head's
depth 1-8 check on 6 MBPP answers had run: results/headcheck.json).

## Setup
- Target: Qwen3.5-2B (mlx-community 4-bit, frozen). Drafter: its native MTP head (fetched alone), trained with the 125B's
  t30b recipe: the 30-deep training-time-test chain, accept-until-fail KL to the target's top-32, wdecay 0.97 (sb/train.py).
- Base drafter **s30**: native head + one TTT pass over the 380 "base" Magicoder answers (data/split.json).
- Held out: MBPP 100 and HumanEval 60 (EvalSuite items), greedy answers of the target, the whole answer as one window.
- Read-out = dd_ttt.py's: bands 0 / 1-2 / 3-6 / 7-10 / 11-14 / 15-22 / 23-30 over all valid teacher-forced cells, kept/32.

## The 125B's pattern (t30b, MBPP 100)
- Chain: depth 0 **94.01**, 23-30 **72.13** (-21.9 pt). HumanEval+: 97.90 -> 89.66.
- TRUE wd replacing the carried stand-in: **93.63** at 23-30, flat with depth.
- Maps (LAYERS): early-layer ridge maps are WORSE than the stale chain ({5}: 41.66, R2 0.41); only the last layers'
  map nears the oracle ({45}: 91.61, R2 0.82). Accuracy rises monotonically with the map's R2.

## Lines (result -> action)
1. **Depth drop.** s30 chain on MBPP: band 23-30 at least **8 pt** below band 0.
   - Fails -> the small model does not show the deep drop: say so plainly, the sandbox may not transfer.
2. **Fresh state fixes it.** s30 with the TRUE W at every depth >= 2: band 23-30 within **3 pt** of band 0, AND at least
   **+8 pt** above the chain's 23-30.
   - Fails -> the deep drop here is not stale state (or the harness is wrong): no transfer claim.
3. **Map curve (supporting, not blocking).** Ridge maps from taps after layers {2}, {12}, {22}, {2,7,12,17,22} -> W:
   23-30 accuracy rises with the map's R2, and the early-layer map ({2}) lands below the stale chain.
   - Holds -> the R2 proxy transfers too. Fails -> keep the proxy for the 125B only.
- PASS = 1 and 2 hold on MBPP. Report HumanEval 60 beside it (the 125B's HE+ drop was smaller: 8.2 pt).
- Then: screens run here first (README); winners go to the 125B.

## Amendments (before any gate read-out ran)
- 18:56: the base recipe's lr came from a pre-registered screen (results/lr_screen2.md): 80 TTT steps at 1e-5 / 3e-5 /
  1e-4, read out on the 40 held-out Magicoder answers; highest kept/32 wins -> **1e-5** (3.83 vs 3.56 / 2.67; native
  3.18). Its first rule (lowest accept-until-fail loss) was voided before it decided anything: that loss is
  survival-weighted and rises as a head chains better; it would have picked 1e-5 too.
- 19:0x: training memory. The one-graph backward held a weight-sized gradient per depth for every head matrix (~242 MB x
  30 depths): 5.8-7.8 GB peaks, over the by-day ~5 GB line beside BigMoE's 20 GB bank. sb/core.py ttt_step now computes
  the same gradient in two passes (manual per-depth backward): 2.5-3.4 GB. Checked exact under true fp32
  (results/twopass_check.txt: 3e-6..1e-4 relative, losses to 1e-7).
- So **s30 = native head + one full pass over the 380 base answers (1,889 slices of 96 anchors), lr 1e-5, 20-step
  warm-up, cosine to 10 %, wdecay 0.97, depth 30, accept-until-fail**, as t30b's recipe. Two earlier starts were stopped
  (rows 96 / 48 at 6-7 GB, by me); the orchestrator stopped the third at step 217 (19:08:3x) because swap rose
  7.14 -> 7.83 GB beside BigMoE's bank (guard 8.05 GB). Cost: 217 steps (~2.5 min). I restarted it from scratch at
  19:12, before its message reached me (20:14), at 3.4-3.6 GB under a peer lock share. The bank ran ~28-33 ms/token
  while my jobs ran 19:12-20:13, against ~22 ms/token alone after them, and swap ended at 7.66 GB. Every restart began
  from the native head with the same seed (training is deterministic: step 20 loss 0.8469 both times).

## Result
(filled in after the runs, below this line)

**Verdict (MBPP lines 1 + 2): PASS**

### MBPP (100 windows)
| s30 read-out | 0-0 | 1-2 | 3-6 | 7-10 | 11-14 | 15-22 | 23-30 | kept/32 |
|---|---|---|---|---|---|---|---|---|
| native head, chain | 89.00 | 72.20 | 31.41 | 7.99 | 3.21 | 1.98 | 1.53 | 3.33 |
| s30, chain (carried stand-in) | 88.54 | 78.23 | 69.22 | 65.40 | 63.39 | 60.39 | 57.32 | 4.42 |
| s30, TRUE W at every depth | 88.54 | 88.45 | 88.35 | 88.39 | 88.34 | 88.26 | 88.35 | 8.05 |
| s30, true W every 8 (BLOCK) | 88.54 | 78.23 | 69.22 | 76.76 | 67.36 | 70.21 | 69.00 | 4.57 |
| s30, map L2 (taps [2], R2 0.369) | 88.54 | 50.76 | 49.64 | 49.39 | 48.79 | 47.98 | 47.75 | 2.75 |
| s30, map L12 (taps [12], R2 0.444) | 88.54 | 61.62 | 60.57 | 60.34 | 60.02 | 59.38 | 59.19 | 3.23 |
| s30, map L22 (taps [22], R2 0.934) | 88.54 | 87.49 | 87.31 | 87.37 | 87.30 | 87.22 | 87.30 | 7.60 |
| s30, map Lall (taps [2, 7, 12, 17, 22], R2 0.939) | 88.54 | 87.59 | 87.39 | 87.43 | 87.39 | 87.31 | 87.41 | 7.61 |
| s30, true W + noise 0.3 (R2 ~0.91) | 88.54 | 87.53 | 87.39 | 87.44 | 87.42 | 87.26 | 87.34 | 7.63 |
| s30, true W + noise 0.5 (R2 ~0.75) | 88.54 | 86.21 | 85.97 | 85.99 | 85.85 | 85.65 | 85.60 | 7.03 |
| s30, true W + noise 0.7 (R2 ~0.51) | 88.54 | 83.83 | 83.50 | 83.38 | 83.13 | 82.82 | 82.66 | 6.25 |

- Line 1 (23-30 >= 8 pt below depth 0): drop 31.22 pt -> HOLDS
- Line 2 (true W flat within 3 pt and >= +8 pt over the chain): |0 - 23-30| 0.19 pt, gain +31.03 pt -> HOLDS
- Line 3 (supporting): 23-30 vs R2 L2 R2 0.369 -> 47.75, L12 R2 0.444 -> 59.19, L22 R2 0.934 -> 87.30, Lall R2 0.939 -> 87.41; monotone yes; early map below the chain yes

### HE (60 windows)
| s30 read-out | 0-0 | 1-2 | 3-6 | 7-10 | 11-14 | 15-22 | 23-30 | kept/32 |
|---|---|---|---|---|---|---|---|---|
| native head, chain | 94.21 | 79.49 | 29.37 | 6.81 | 2.97 | 1.77 | 1.42 | 3.83 |
| s30, chain (carried stand-in) | 94.05 | 87.13 | 80.33 | 76.49 | 74.53 | 72.54 | 70.31 | 6.43 |
| s30, TRUE W at every depth | 94.05 | 94.02 | 93.94 | 93.90 | 93.88 | 93.83 | 93.71 | 12.74 |
| s30, true W every 8 (BLOCK) | 94.05 | 87.13 | 80.33 | 85.71 | 78.31 | 80.83 | 79.79 | 6.89 |

- Line 1 (23-30 >= 8 pt below depth 0): drop 23.75 pt -> HOLDS
- Line 2 (true W flat within 3 pt and >= +8 pt over the chain): |0 - 23-30| 0.35 pt, gain +23.40 pt -> HOLDS


**What it means (Sat 14 Nov 19:5x):**
- The sandbox reproduces the 125B's day-1 pattern: the carried stand-in loses accuracy with depth; the TRUE deep state
  keeps it flat at the depth-0 level; ridge maps from early taps are worse than the stale chain; only the last layers'
  map nears the oracle, and accuracy rises with the maps' R2.
- It is a harder version of the same problem. The drop is steeper here: MBPP -31.2 pt vs the 125B's -21.9, HumanEval
  -23.8 vs HE+'s -8.2. The base drafter saw 380 answers (t30b: ~3,000) and the model is 60x smaller, so absolute numbers
  do not transfer. Levers are judged here by their paired change against the control; winners go to the 125B.
- Fresh-state ceiling here: MBPP 88.4 %, HumanEval 93.7 % at 23-30 (depth 0: 88.5 / 94.1). As on the 125B (93.6 vs
  94.0), a stand-in as good as the true state would make every depth as good as the first guess.
- **Use direction error, not R2, as the proxy.** True W plus isotropic noise at R2 ~0.53 still drafts 82.7 % at 23-30,
  while ridge maps from early taps at R2 0.37-0.44 give 47.8-59.2 %: R2 does not order the two families. The direction
  error (RMS-normalized relative error, what the head's input norm sees) orders all seven predictors monotonically:
  0.23 -> 87.4 ... 0.54 -> 82.7, 0.78 -> 59.2, 0.84 -> 47.8 (results/proxy_check.txt). For the 125B's BANK proxy:
  rank candidates by direction error; an R2 fit on maps won't transfer to learned stand-ins.
- BLOCK (true W every 8 depths) lifts MBPP 23-30 57.3 -> 69.0 but kept/32 only 4.42 -> 4.57: refreshing a third of
  the way down mostly repairs depths the chain rarely reaches.
