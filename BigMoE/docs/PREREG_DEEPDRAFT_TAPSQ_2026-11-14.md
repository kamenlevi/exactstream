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
