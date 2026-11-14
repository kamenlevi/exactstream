# PREREG: DEEPDRAFT FRESH-H, can the MTP head use fresh trunk features inside its chain? (Sat 14 Nov 2026, ~15 min)

Written before the run. The orchestrator's step 1 of "freshness first" (16:1x), after:
- CEILING = LEARNABLE: the depth misses are on tokens the 125B was sure of;
- TAPSQ2 = KILL: a richer view of the verified context doesn't help.
- Law 2 (RESULT.md, 11-11): with the exact trunk state two back, the reader holds 97.7 % (HumanEval+) / 93.5 % (MBPP) at
  every depth.
- So the question is freshness. Can our drafter use fresh trunk features on its drafted rows, through the same kind of
  adapter?

## Setup
- **Harness:** TAPSQ2's valid one (amendment 2). Warm start t30b, the head on t30b's own Adam state at 1e-6, the same 80
  slices (seed 1114) of the 795 unseen n1112 files.
- **Control:** tq2_C, already run. It stays at t30b: MBPP 23-30 72.06 vs 72.13.
- **Arm F:**
  - C plus a zero-init Linear(2560 -> 2560) on `h`, on a bias-corrected Adam at 1e-5 with a 10-step warm-up.
  - It is fed at every drafted row instead of the target rows. At depth j >= 2, row i (anchor s) gets the TRUE `h` of
    row s + j - 1, the position its carried stand-in approximates. It is added to the fused input the way the
    embedding term enters.
  - Under accept-until-fail, the rows that count are on a correct chain. There the drafted tokens are the true tokens,
    so the captured `h` is exactly what a fresh trunk pass over the drafts would give.
  - `h` is DFlash2's fused input of layers 5/15/25/35/45, so this is close to the full trunk's view: an upper bound for
    a cheap partial pass.
- **Eval:** MBPP 100 (d1102mbpp has `h`), the eval's chain to depth 31, with the true `h` on the drafted rows the same
  way. The eval feeds the true tokens at every depth, so this is per-guess accuracy given a correct prefix, as the
  headline bands are.
- Time: ~10 min of training + ~3 min of eval, alone. The capacity retest (CAPQ) runs first.

## Lines (result -> action; MBPP 23-30)
- Half the gap is C + 0.5 x (93.5 - C) = ~82.8 for C = 72.06.

| Result | Line | Action |
|---|---|---|
| WORKS | F >= ~82.8 | Freshness works inside our drafter. Step 2: re-prefill raw taps at full width on every row (~50 train answers + MBPP 100), then per-layer freshness (layer 5 alone, 5+15, ...): what a cheap partial trunk pass buys, which sets the engine's price question. |
| PARTIAL | C + 2 pt <= F < ~82.8 | Fresh features carry signal, but this adapter or 80 steps limit them. Step 2 as above, plus a replacement design (the fresh feature replaces the carried stand-in instead of adding to it). |
| FLAT | within 2 pt of C (or below) | The additive adapter can't use fresh features. Rethink (replace the stand-in) before spending the re-prefill. |

- Also reported:
  - the per-step all-depth KL ratio F/C over steps 21-40 and 41-80;
  - every MBPP band, kept per 32, and a per-window sign test.
- The per-depth KL at 24-30 is survivor-biased under accept-until-fail, and isn't read.

## Result (Sat 14 Nov 16:44): FLAT, in fact worse. Adding the true fresh features hurts, more with depth
- Arm F trained 80 steps alone (16:32-16:42; step 1's loss 0.3499 equals C's, so the zero-init tap starts as a no-op),
  then MBPP 100 (16:42-16:44).
- **MBPP 100 bands (t30b / C / F):**

  | band | t30b | C | F | F - C |
  |---|---|---|---|---|
  | 0 | 94.01 | 93.97 | 93.99 | +0.02 |
  | 3-6 | 83.45 | 83.52 | 82.75 | -0.77 |
  | 7-10 | 79.61 | 79.75 | 78.51 | -1.24 |
  | 15-22 | 74.38 | 74.37 | 72.71 | -1.66 |
  | 23-30 | 72.13 | 72.06 | 70.09 | -1.97 |
  | kept per 32 | 6.924 | 6.939 | 6.800 | -0.14 |

  - Per window at 23-30: F better on 13, worse on 64 (sign test p < 0.001).
  - Training KL F/C: 1.058 over steps 21-40, 1.069 over 41-80. At depth 1 (no fresh input there) it is 1.000 / 0.999.
- **Verdict: FLAT.** F is 1.97 pt below C (PARTIAL needed >= 74.06, WORKS >= 82.78).
- **Reading:**
  - Band 0 is untouched, and the loss grows with depth. The added fresh term perturbs the carried stand-in, and the
    head barely adapts (lr 1e-6, 80 steps), so the error compounds down the chain.
  - An additive input can't hand the head the fresh state it needs. Law 2 says the exact trunk state two back holds
    93.5 % at every depth, but this adapter turns the same information into noise.
- **Action, as pre-registered:** rethink before spending the re-prefill. The fresh feature should replace the carried
  stand-in instead of adding to it.
  - The next quick test is eval-only: fit a linear map from h to the trunk's widened stream on captured rows. Then
    score t30b's chain with the stand-in replaced by the map of the true h at each depth.
  - That shows how much of Law 2's 93.5 a reconstruction from h recovers, before any training.
