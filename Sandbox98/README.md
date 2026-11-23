# Sandbox98: minute-scale screens for WEEK98's deep drafter

A small-model copy of the 125B deep-drafter setup (`~/FableGod/deepdraft/WEEK98.md`), so a lever can be screened in
minutes on the Mac instead of 15-90 min on the 125B. Winners then get confirmed on the 125B. It also gives the
"any model" check on a second (smaller) model.

**Gate: PASS** (Sat 14 Nov 19:51). The small model shows the same deep drop (MBPP 23-30: 88.5 -> 57.3 %) and the same flat
fresh-state oracle (88.4 %) as the 125B. Details in [Gate result](#gate-result) and GATE.md.

## Results so far (Sat 14 Nov 2026, for the 01:00 sessions; details in the linked files)

| test | verdict | the number | file |
|---|---|---|---|
| validity gate | PASS | MBPP 23-30: chain 57.32 vs true state 88.35 (depth 0 88.54) | GATE.md |
| transfer to the 125B | 4 / 4 match | depth drop, flat oracle, early vs late maps, t30f NO EFFECT; capacity + anchor-taps checks not run yet | TRANSFER.md |
| ROLLOUT (Markov top-state model as the stand-in) | KILL | 23-30: 20.9 / 22.3 vs chain 57.3; one step 66.8 vs 78.2 | results/ROLLOUT.md |
| T2 context truncation (theory) | MIXED | 23-30: chain 55.4; 2B over only the drafted tokens 69.6; +16 tokens 75.5; +64 80.6; +256 86.3; full 87.7 | results/T2_TRUNCATION.md |
| feat (t30f) | NO EFFECT | -0.26 pt | results/SCREENS.log |
| tapmlp (layer-12 tap -> stand-in) | NO EFFECT by line | +0.61 / +0.82 pt (p 0.002 / 0.016) | results/screens/tapmlp_LINE.md |
| distill (fresh-state teacher) | NO EFFECT vs a holding control | -0.48 pt (its +3.1 / +8.1 vs a drifting control was drift protection) | results/screens/DEPTH_LINES.md |
| headfeat / noisehead | small LOSS / NO EFFECT | -1.50 / -0.33 pt (300 steps x 2e-6) | results/screens/standin_LINES.md |
| deep2 / tie2 (depth added to s30) | HURTS | -2.48 / -2.38 pt | results/screens/DEPTH_LINES.md |
| 2 vs 1 layer, both from the native head | NO EFFECT | +0.05 pt; budget 400 -> 1,889 steps is worth +7.1 pt | results/screens/DEPTHN_LINES.md |

Method findings: judge arms by read-outs, not the accept-until-fail loss (survival-weighted); rank stand-ins by direction
error, not R2 (results/proxy_check.txt); never continue an annealed head at a constant high lr (the control drifts 7-15
pt): use the re-annealed pool2 setup below; at 400 steps the plain control doesn't move up, so short screens only see
fast-acting levers. Draft cost: 2.54 / 3.96 / 5.32 ms per drafted token at 1 / 2 / 3 layers (1.23 ms of it lm_head +
argmax; results/timedraft.json).

**Screen setup to use now:** `sb/train_multi.py --arms ctl:-,NAME:LEVER --depth 30 --steps 400` (defaults: pool2, one
pass, s30o's Adam, warm-up to 1e-5 then a cosine to 1e-6), read out with `sb/evalx.py --runs s30,TAG_ctl,TAG_NAME@LEVER
--depth 31 --nm 50 --nh 30`, and check the control against s30 first (scripts/depth.sh is a complete example). Pending:
the depth-12 fast-format calibration (results/screens/CALIBRATION.md v2) and transfer checks 5-6 (TRANSFER.md).

## What it is

| | 125B (BigMoE / TrainLab) | Sandbox98 |
|---|---|---|
| Target | Qwen3.8-Flash-Next 125B MoE, frozen | **Qwen3.5-2B** (mlx-community 4-bit, 1.6 GB, already on disk), frozen. Same family: hybrid Gated-DeltaNet + gated full attention |
| Drafter | the native MTP head (b5 -> t30b) | the **native MTP head** of Qwen3.5-2B (121.7 MB, fetched alone by byte ranges, DOWNLOADS.md), fp32 trainable copy |
| Head input | wd (pre-mixer widened stream) + next-token embedding | W = the target's **post-norm** final hidden + next-token embedding, fc over [emb, hidden] (headcheck: 88.5 % vs pre-norm 82.4 % at depth 1) |
| Chain | D-22 training-time test, 30 deep | the same, ported (sb/core.py `chain`, `ttt_loss`) |
| Loss | per-depth KL to the target's top-32, accept-until-fail, wdecay 0.97 | the same |
| Data | ~3,000 Magicoder answers (n1112) | 540 greedy answers to the **same** Magicoder prompts (prompts_py_v1 order): 380 base / 120 screen pool / 40 val |
| Held out | MBPP 100, HumanEval+ 24 windows | MBPP 100, HumanEval 60 (EvalSuite items; the target's own greedy answers) |
| Read-out | dd_ttt.py: bands 0, 1-2, 3-6, 7-10, 11-14, 15-22, 23-30 + kept/32 | the same, + strict conditional per depth |
| Taps | {5,15,25,35,45} of 48 + wd | after layers {2,7,12,17,22} of 24 + W (data/*/KEY.taps.npz) |

Costs on the shared GPU (BigMoE's bank at 80 % GPU beside it): a TTT step of 96 anchors x 30 depths ~0.5-0.7 s at an
MLX peak of ~3.4 GB (under the by-day ~5 GB line; the exact two-pass gradient, below); a read-out at 31 depths ~0.6 s
per MBPP window, ~1.3 s per Magicoder window; a screen (300 steps + 80 windows) ~5 min, plus the control once.

## One-command screens

```bash
cd ~/Sandbox98
./screen.sh NAME LEVER "LINE" [STEPS] [NMBPP] [NHE]
```

From an agent session, start long jobs detached (`scripts/detach.sh logs/screen_NAME.log ./screen.sh NAME LEVER "LINE"`)
so they outlive the tool call. Before any run beside a memory-heavy job: check swap (`sysctl vm.swapusage`) and take a
`peer lock share` sized to the run's measured LIFETIME footprint (/usr/bin/time -l's "peak memory footprint", not
the MLX peak: ~4.5 GB for 1-arm training, ~5.1 GB for 2 arms, 6.4-7.5 GB for evalx / tree8 read-outs; Tue 11-17). On Sat 19:08 the
orchestrator stopped a base run because swap rose 7.14 -> 7.83 GB beside BigMoE's bank (its guard: 8.05 GB).

- Trains `levers/LEVER.py` on top of the base drafter **s30** for STEPS (default 300, 96 anchors each) on the screen
  pool (answers s30 never saw), constant lr 2e-6 after a 20-step warm-up, then reads it out on 50 MBPP + 30 HumanEval
  windows. ~5 min per screen (the control adds ~5 min once).
- The control is s30 + the same steps on the same slices in the same order without the lever. It is trained and read out
  once per STEPS and lr and cached (`ckpt/ctl_s30_300_lr2e-6`, `results/screens/_eval_ctl_s30_300_lr2e-6_50_30.json`).
- **A training lever needs a budget at which the control itself measurably moves; otherwise NO EFFECT means untested**
  (orchestrator, Sat 21:3x). At 300 steps x 2e-6 the plain control moved only +0.49 pt. Long screens: `BASE=s30o OPT=s30o
  LR=1e-5 EPOCHS=2 ./screen.sh NAME LEVER "LINE" 1000` continues the base's Adam state (results/screens/LONG_LINES.md).
- **Screen lr: see results/screens/SETUP.md.** At constant 1e-5 with a fresh Adam the control itself drifted far below s30
  (MBPP-50 23-30 55.47 -> 47.31; s30's cosine ended at 1e-6), so screens moved to 2e-6 after a pre-registered check.
  Always look at the control against s30 on the same windows before reading a screen.
- Prints, per set, band 23-30 and depth 0 (control -> lever, pt, bootstrap 95 % CI over windows, windows better / worse,
  sign-test p) and kept/32, next to your LINE. Appends everything to `results/SCREENS.log`.
- STEPS = 0: an eval-only lever (e.g. a stand-in transform), compared against s30 itself.
- Write the LINE first ("PASS if MBPP 23-30 >= +2.0 pt with p < 0.05 -> confirm on full sets; else KILL").
  The rules: <= 15 min per screen; the full sets (100 + 60 windows) only confirm; winners go to the 125B.

Example (the first screen, t30f's feature loss ported):
```bash
./screen.sh feat feat "t30f was NO EFFECT on the 125B. PASS if MBPP 23-30 >= +2 pt (p < 0.05) -> transfer differs, look why; else agrees with the 125B"
```
Its result (Sat 20:11, lr 2e-6, results/SCREENS.log): MBPP 23-30 -0.26 pt (CI -0.71..+0.16), HumanEval +0.06, kept/32
flat: **NO EFFECT, the 125B's t30f verdict** (the first transfer check agrees). The paired screen resolves ~+-0.5 pt
at 23-30 on 50 windows. Caveat: at 2e-6 for 300 steps the head moves little, so a lever must act fast to show; raise
STEPS (or LR, checking the control) for slower levers.

Levers so far: `levers/feat.py` (t30f's feature loss), `levers/tapmlp.py` (WEEK98 item 1: a trained map from the
target's true layer-L tap at the drafted row, added to the stand-in). Screened 20:25 (tap 12, 300 steps): MBPP 23-30
+0.61 pt (p 0.002), HE +0.82 (p 0.016): NO EFFECT by its +2 pt line but a small real gain; next per its line: tap 17.
Learned-stand-in levers (orchestrator's items 1-3, 21:03-21:32): `levers/distill.py` (KL to the head's own fresh-state
output) NO EFFECT; `levers/headfeat.py` (state target in head space) a small LOSS (MBPP 23-30 -1.50 pt, p 0.013);
`levers/noisehead.py` (noise-robust head) NO EFFECT at depth, +0.3 pt at depth 0. All at 300 steps x 2e-6, where the
plain control moves only +0.5 pt: loss-reshaping levers need a longer screen (results/screens/standin_LINES.md).

> (a quote from the project owner, kept private)

```bash
./screen_fast.sh BATCH "LEVER1,LEVER2" "LINE" [STEPS] [DEPTH=12]
```

- A control and 1-2 levers train in ONE process (`sb/train_multi.py`) on the same slices in the same order, each arm with
  its own drafter copy continuing the base's Adam state (s30o, lr 1e-5), so the control cannot drift from a fresh Adam.
- Rollout depth 12 instead of 30; the read-out goes to depth 15: bands 0-14, a combined **7-14 band (the line band)**,
  kept/16. The stand-in's staleness is already large by 7-14 (s30's chain: 88.5 at depth 0, ~63 at 11-14), so a lever
  that keeps it fresh should show there. **Winners are confirmed at depth 30** with the long format
  (`BASE=s30o OPT=s30o LR=1e-5 EPOCHS=2 ./screen.sh NAME LEVER "LINE" 1000`).
- **STEPS: the calibrated budget**, the shortest at which the plain control measurably moves from the base (MBPP-50 7-14
  CI excludes 0 and |delta| >= 1 pt). Below it, NO EFFECT means untested. Pending: results/screens/CALIBRATION.md
  (queued after the long screens); until it reports, pass STEPS explicitly.
- Memory: ~1.5 GB + ~0.75 GB per arm + ~1.5 GB for the step (2 arms measured 4.1 GB). By day keep a batch at <= 3 arms
  (control included), ~5 GB; bigger batches at night.
- Output: results/SCREENS.log (per lever, MBPP and HE: band 7-14, depth 0, kept/16, paired CIs, sign tests),
  results/screens/fast_BATCH*.json.

## Writing a lever (`levers/NAME.py`)

All hooks are optional; sb/train.py and sb/evalx.py call them:

```python
def setup(D, args): ...                 # once, after the start weights load: add modules to D, set config
def extra(j, h, anc, m, x): ...         # an added loss term at depth j (weighted like the KL): h [1,n,d] the head's
                                        # depth-j output (= depth j+1's stand-in for W[anc + j]), anc [n] anchors,
                                        # m [n] the KL's row mask, x the window (x["W"] = true W rows, x["ids"], ...)
def standin(j, h, anc): ...             # transform the carried stand-in before depth j's input (train AND eval)
def freeze(D): ...                      # freeze parts of D (default: all of D trains)
LR_MULT = 1.0                           # lr multiplier for this lever's run
```

`levers/feat.py` is a complete example (12 lines). Parameters a lever adds in `setup` are saved in the run's checkpoint
and found again at eval (`--lever`).

## Other commands

```bash
PY=~/BigMoE/.venv/bin/python
$PY sb/train.py --run NAME --start s30 --files pool --steps 300 --const --lr 1e-5 [--lever L]   # train (96 anchors/step)
$PY sb/evalx.py --runs s30,NAME --sets mbpp,he --mode chain,true --out results/x.json [--nm 50 --nh 30]
$PY sb/compare.py results/a.json "s30|mbpp|chain" results/b.json "NAME|mbpp|chain" --band 23-30
$PY sb/fitmap.py --name L22 --taps 22          # ridge map taps -> W (R2 on held-out rows)
```

Eval modes: `chain` (the real chain), `true` (TRUE W at every depth >= 2: the oracle), `block:K` (true W every K depths),
`map:NAME` (a fitted map's prediction replaces the stand-in), `noise:F` (true W + noise, R2 ~ 1 - F^2).

## Judge arms by read-outs, never by the training loss

The accept-until-fail loss is a survival-weighted average of per-depth KLs: depth j's weight is the share of rows still
alive there. An arm that chains better carries more rows into the deep, high-KL depths, so its loss can RISE while it
gets better. In the lr screen, lr 3e-5 read 1.02 / 0.90 at steps 40 / 60 against 1e-5's 0.77 / 0.75, yet its 23-30
band was higher (32.1 vs 29.3 % on 40 held-out answers). Compare bands and kept/32 (compare.py), or per-depth KL on fixed
rows. (results/lr_screen2.md. The 125B's CAPQ closed capacity on a per-step ratio of this loss, 1.016.)

## Data layout

`data/{mbpp,he,mc}/KEY.npz`: ids, gen, asst, res (pre-norm final residual, fp16; the head reads final_norm(res)),
tk / lp (top-32 of each row, predicting the next token), am (the capture forward's argmax), stop.
`data/*/KEY.taps.npz`: taps [5, S, 2048] after layers 2, 7, 12, 17, 22. `data/split.json`: base / pool / val.
Capture: 700 answers (MBPP 100, HE 60, Magicoder 540), greedy, max 512 new tokens; the capture forward's argmax matches
the greedy token on 98.3-99.4 % of answer rows (batched-decode numerics).

## Differences from the 125B harness (read before trusting a transfer)

- Dense 2B target without hyper-connections; the head is one dense gated-attention layer (the 125B's has an MoE + HC).
- 380 base answers (the 125B's t30b saw ~3,000). The base drafter is weaker; levers that need data may look smaller here.
- Each training slice's loss covers exactly its R anchors (d22's covered R + depth anchors at shallow depths).
- Adam with bias correction (the 125B runs use MLX's default without it), bf16 dequantized lm_head for gradients
  (the 4-bit path's backward is ~5x slower); read-outs use the target's own 4-bit lm_head.
- Failed rows leave the training chain (accept-until-fail makes this exact: checked equal to the uncompacted loss,
  1.319591 vs 1.319593).
- The gradient comes from `ttt_step`: pass 1 runs the chain without autograd and takes each depth's KL gradient with
  respect to its normed rows; pass 2 is a manual backward through the chain, one depth at a time. Same gradient as
  autograd (3e-6..1e-4 relative under true fp32, results/twopass_check.txt), at 2.5-3.4 GB instead of 5.8-7.8 GB
  (autograd kept a weight-sized gradient per depth for every head matrix). MLX runs fp32 matmuls as TF32 by default;
  training does too, as on the 125B.
- PAUSE-aware (~/.local/share/quiet/PAUSE) between steps and windows, but ~/Sandbox98 is not in fanwatch's research
  folders, so a heat freeze does not SIGSTOP these jobs.

## Gate result

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

Pre-registered lines, amendments and the run history: GATE.md. Raw: results/gate*.json.
