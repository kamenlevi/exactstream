# PREREG: DEEPDRAFT TAPSQ, a ~25-min quick test of multi-layer taps for the MTP head (Sat 14 Nov 2026)

Written before the run. Script: scripts/dd_taps.py. Data: data/deepdraft/tapsq/.

## Why this test, and why it is short
- TTT30C asks whether the MTP head's size is the wall. t30f was NO EFFECT, and so was t30x2 (twice the capacity). t30x4 is
  scoring now.
  - The pre-registered reading if nothing pays: the wall is what the head can see. The next candidate is EAGLE-3's
    multi-layer taps (PREREG_DEEPDRAFT_TTT30C_2026-11-13.md, "What each outcome means").
> (a quote from the project owner, kept private)
  figure out in a 15minute test. not 13 hour one."
  - So this question starts with a quick test, and its result decides the next step.
  - A full taps arm comes only after a pass, and it gets a stop line at 10-25 % of its steps.

## What already exists (checked Sat 14 Nov 14:0x)
- Every n1112 capture file holds `h`: DFlash2's own fused context input, hidden_norm(fc(taps)).
  - The taps are the mean of the 4 hyper-connection streams after layers 5, 15, 25, 35 and 45.
  - It is 2560 wide, on every generated row and the last 2048 prompt rows.
  - The MTP head reads only `wd` today, the top pre-mixer stream (10240 = 4 x 2560).
  - So the first test needs no new prefill.
- 795 n1112 files are on disk that t30b never saw (3,072 on disk, 2,277 in t30b's split). They are pinned in
  data/deepdraft/tapsq/split.json at the first run.
- The MBPP eval set (TrainLab dlog/d1102mbpp, the first 100 answers) also holds `h`, from the same fused input:
  - the overall RMS is 1.031 vs 1.031 on 20 files each;
  - the per-dimension RMS profile correlates 0.92, and the per-dimension mean 0.96;
  - the median per-dimension RMS ratio is 0.998.
- The 24 HumanEval+ windows (d16w) have no `h`. They would need a re-prefill, and only if this test passes.

## Setup
- **Start:** t30b. The 125B is untouched.
- **Data:** one fixed order (seed 1114) of 192-anchor slices of the 795 unseen files, 80 steps.
  - Every arm reads the same slices in the same order.
  - Each step's loss is on a slice new to that arm, so it is a held-out (prequential) number, paired by step across arms.
- **Loss:** b5's own training loss: the KL to the 125B's top-32, chain to depth 30, wdecay 0.97, accept-until-fail.
- **Head:** Adam at lr 1e-5, constant after a 10-step warm-up, gradient clipped to norm 1.
- **Arms:**
  - **C:** t30b continues. The control.
  - **T:** C plus `xb_tap`, a zero-init Linear(2560 -> 2560) of `h`. It is added to the fused input of every target row,
    the way the embedding term enters (each of the 4 streams gets the same term).
    - It is not added on drafted rows: those carry the head's own output, which is EAGLE-3's place for the taps.
    - About 6.6M new parameters, on their own Adam at lr 1e-4 (10x, as TTT30C's branch), their gradient clipped to
      norm 1 on its own.
    - At 1e-5 a zero-init path barely moves in 80 steps.
  - **R (only if T passes):** T fed `h` from a random other nonzero row of the same window.
    - The same new path at the same lr, without the row's information. This is the orchestrator's control.
- **Smoke checks before the arms:**
  - With the zero-init tap the loss equals t30b's (EQUAL).
  - The tap gets a nonzero gradient.
  - One tap-only step changes the loss.
  - R's shuffled `h` gives a different loss.
- **Eval:** MBPP 100, depth bands as TTT30C's eval, for t30b, C and T, with per-window counts kept for a paired check.

## Lines (result -> action)
- **Measure 1:** the T/C ratio of the mean per-step loss over steps 41-80, with T lower on at least 70 % of those steps.
  - The depth-1 KL and the mean KL of depths 24-30 are logged per step.
  - Their T/C ratios over steps 41-80 are reported next to the all-depth one: an all-depth mean can hide a deep gain or a
    shallow loss.
- **Measure 2:** MBPP 23-30, T@80 vs C@80, plus a paired sign test over the windows.

| Result | Lines | Action |
|---|---|---|
| PASS | ratio <= 0.90, T lower on >= 70 % of steps 41-80, AND MBPP 23-30 T >= C + 2 pt | First run R (~8 min with its eval). If R does as well as T, the gain is the new path at 10x lr, not the taps, and the line closes as a confound. If T beats R by the same lines, re-prefill the 24 HumanEval+ windows with taps and pre-register a full taps arm from b5 on t30b's pinned split, with a stop line at 10-25 % of its steps against t30b's saved per-step losses. Then quick-test the raw 5 x 2560 taps against DFlash2's 2560 projection. |
| KILL | ratio within 0.97-1.03 (or above), AND MBPP 23-30 within 1 pt (or below) | One more ~20-min test with the raw taps: re-prefill ~50 answers, the 5 taps at full width into the same kind of adapter. DFlash2's fc was trained for DFlash2 and may drop what the MTP head needs. If that is flat too, the multi-layer line closes for the MTP head. |
| BETWEEN | anything else | 80 more steps per arm (~15 min) on the next slices, then the same lines. |

- Guards:
  - Peak 22 GB (MLX).
  - Swap +2 GB over the start stops the test.
  - Another session's quiet lock (held, lapsed or queued) stops the current arm; it reruns from the start after.
  - HEALTH ACT stops it. So does a closed lid.
> (a quote from the project owner, kept private)
  - Memory: an eval peaks ~10 GB, this ~16 GB, on the ~45 GB floor.
  - Neither run claims a timing, and the numbers here don't depend on speed.

## Amendment 1 (Sat 14 Nov 15:08, after the smoke and before arm T starts; C is running and unaffected)
- The smoke (15:05) passed all four checks, but its one tap-only step at the pre-registered 1e-4 lifted the loss
  0.35 -> 3.78. The tap's |w| max was 3.16e-4: MLX's Adam has no bias correction, so step 1 is 3.16x lr.
- Why so large: a zero-init 2560-wide input under Adam moves every weight by about lr x sign(g). The gradient is an outer
  product (output error x h), so after one step every row's output moves coherently by about lr x sum|h| (~0.65 per
  dimension at 1e-4). That swamps the fused input.
  - The pre-registered reason for 10x ("at 1e-5 a zero-init path barely moves in 80 steps") was wrong: it moves plenty.
  - The same mechanism likely shaped the capacity arms' early losses (t30x4's 20-step mean 4.07; t30x2 +95 % KL over steps
    21-100). Both had zero-init branches at 10x lr.
- So T (and R) use the tap lr 1e-5, the head's own, with the same 10-step warm-up and the separate clip. Nothing else
  changes: C, the slices, the lines and the actions stay as written.
- A one-step smoke at 1e-5 is logged before T starts (data/deepdraft/tapsq/run.log).

## The first run (15:05-15:31): VOID. Its control collapsed, so the harness failed, not the taps
- The lines read KILL: the all-depth KL ratio T/C was 1.364 over steps 41-80 (T lower on 8 %), and MBPP 23-30 was T
  57.34 vs C 58.26 (-0.92 pt). But the control is not a control:
  - C (t30b + 80 steps, no tap) fell from t30b's 72.13 to 58.26 at MBPP 23-30 (-13.87 pt), band 0 94.01 -> 93.06,
    kept per 32 6.92 -> 5.51. The damage grows with depth.
  - Weights: tq_C is 1.82 % from t30b (relative L2 over the head). t30b's whole 4,206-step run moved b5 only 2.88 %.
  - Cause: a fresh Adam (MLX's, no bias correction) at 1e-5, 10x t30b's final lr (its cosine ends at 1e-6). Fresh
    moments make every weight step about 1-3 x lr in a random walk, so 80 steps scramble a head annealed to 1e-6. T had
    the same shock plus the tap's.
- Two things checked and ruled out:
  - The loss is dd_ttt.py's own (T.ttt_loss with accept-until-fail, depth 30, wdecay 0.97). The smoke's "24-30:
    0.0000" means no anchor of slice 0 survived to depth 24: under accept-until-fail the deep depths train only on
    chains still right. So the logged per-depth KL at 24-30 is survivor-biased and isn't a deep-accuracy measure (T's
    0.43x there came with fewer, easier survivors).
  - The save and load carry every weight (strict loads), and the slices use dd_ttt.py's pass_slices on the unseen
    files.
- Data kept as the record: tq_C.json, tq_T.json, eval.json, run.log (15:05-15:31).

## Amendment 2 (15:4x, before the rerun): continue t30b properly, and a validity line
- **Head:** continues on t30b's own Adam state (its moments and step 4,206) at t30b's final lr, 1e-6, constant. So C
  should stay where t30b is.
- **Tap:** a fresh Adam with bias correction (step 1 = lr, not 3.16 x lr), lr 1e-5 after a 10-step warm-up, its own clip.
- **Validity line, first:** C must stay within 1 pt of t30b at MBPP 23-30 (72.13). If not, the rerun is VOID-HARNESS and T
  doesn't run.
- Runs tq2_C / tq2_T (/ tq2_R), evals eval2_*.json; scripts/dd_taps2.sh runs C -> its eval -> the validity line -> T -> its
  eval -> the judge.
- The lines and the actions are unchanged. The capacity retest (CAPQ) runs right after, not beside it.

## Result, the rerun (Sat 14 Nov 16:05): KILL. DFlash2's fused taps don't help the MTP head in a short warm-started test
- **Validity:**
  - C (t30b's Adam state at 1e-6, 80 steps) stays at t30b: MBPP 23-30 72.06 vs 72.13 (-0.07 pt), band 0 -0.04, kept
    per 32 6.94 vs 6.92.
  - Its weights moved 0.04 % (the void run's C: 1.82 %).
- **The lines:**
  - The all-depth KL ratio T/C over steps 41-80 is 1.083, T lower on 15 % of the steps (1.041 over 21-40). The depth-1
    KL ratio is 1.097.
  - MBPP 23-30 is T 71.43 vs C 72.06 (-0.63 pt). Per window, T is better on 35 and worse on 37 (sign test p 0.91).
  - **Verdict: KILL.** The ratio is >= 0.97 and the MBPP change is <= +1 pt.
- **MBPP 100 bands (t30b / C / T):**

  | band | t30b | C | T | T - C |
  |---|---|---|---|---|
  | 0 | 94.01 | 93.97 | 93.74 | -0.23 |
  | 3-6 | 83.45 | 83.52 | 82.89 | -0.63 |
  | 7-10 | 79.61 | 79.75 | 78.90 | -0.86 |
  | 15-22 | 74.38 | 74.37 | 73.81 | -0.57 |
  | 23-30 | 72.13 | 72.06 | 71.43 | -0.63 |
  | kept per 32 | 6.924 | 6.939 | 6.762 | -0.18 |

- The tap trained: |w| max 4.0e-4, rms 7.5e-5. T is slightly worse than C everywhere, so the tap adds noise rather than
  information.
- The per-depth KL at 24-30 (0.76 / 1.42) is survivor-biased under accept-until-fail, as noted, so it isn't read.
- Training: C 787 s, T 530 s, MLX peaks 15.5 / 15.6 GB. No timing is claimed.
- **Action, as pre-registered:** one more ~20-min test with the raw taps (5 x 2560 at full width, from a re-prefill of
  ~50 training answers + the MBPP 100 eval answers), because DFlash2's fc was trained for DFlash2.
  - If that is flat too, the multi-layer line closes for the MTP head.
  - It needs the 125B loaded alone (~45 GB), so it runs after CAPQ, with its own result -> action written first.
- **Reading with the CEILING test (LEARNABLE):**
  - The depth misses are on tokens the 125B was sure of, 23+ tokens past the last verified context.
  - Richer features of the verified context (taps) may not reach them. Fresher features inside the chain (a partial trunk
    pass on drafted rows) are the other candidate the CEILING verdict names.
