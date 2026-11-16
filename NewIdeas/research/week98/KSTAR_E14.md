# K* with E14's engine costs (NewIdeas, Tue 17 Nov 01:0x)

**K* = tree-16 (16-row checks) for the exact engine. On HE+ it is a tie; MBPP prefers tree-16 by 19-20 %.**
**Correction: the engine's R32/R16 is ~1.44, not 1.480. The 1.480 was a ratio of pooled medians, a cross-window
artifact.**

## The cost ratio

Source: Leap's E14 (~/Leap/data/week98/e14/e14.json, run 00:54-00:55 Tue; quiet lock; 3 windows x 16 rounds x 8 arms).
The check time is sum(w) per record; this reproduces the report's medians (A16 61.27, C32one 90.61 ms).

| estimator, C32one / A16 | E14 (Tue) | Mon CHECK55 profile (a082f8c) |
|---|---|---|
| median of the 48 paired ratios (E2's statistic) | 1.433 | 1.438 |
| ratio of means | 1.437 | 1.444 |
| ratio of pooled medians | 1.479 | **1.480 (the number quoted Mon-Tue)** |

Per window (ratio of means): E14 1.435 / 1.405 / 1.473, Monday 1.436 / 1.414 / 1.484. The pooled medians fall in
different windows for the two arms, so their ratio overstates the cost ratio by ~3 %. The tok/s model needs the
average cost, so the ratio of means is the right number. It agrees with the paired median.

Like for like, with the same lever on both shapes, from E14:
- BOF: C32oneB / A16b = 1.463 (means) or 1.457 (paired). BOF saves ~2.2-2.4 ms on both shapes.
- Reference contract: C32c / A16c = 1.459 or 1.457.

## K* (treek_final.py, costs_e14.json; 32-pop dumps; drafting hidden, overhead 4 ms)

| set | model | tree-16 tok/s | tree-32 tok/s | K* |
|---|---|---|---|---|
| HE+ | e14 (plain) | 165.5 | 167.0 | 32 by 0.9 % |
| HE+ | e14 + BOF | 172.2 | 170.9 | 16 by 0.8 % |
| HE+ | e14 + contract | 181.1 | 180.3 | 16 by 0.4 % |
| HE+ | old 1.478 | 165.5 | 162.5 | 16 by 1.8 % |
| MBPP | e14 / BOF / contract | 110.0 / 114.4 / 120.4 | 92.7 / 94.9 / 100.1 | 16 by 18.7-20.5 % |
| HE+ | lossy (byte-floor R) | 487.9 | 542.1 | 32 by 11.1 % |
| MBPP | lossy | 324.3 | 301.0 | 16 by 7.7 % |

- **Exact:** K* = tree-16. Inputs still to come can only push further toward tree-16:
  - drafting is hidden here, and a 32-node tree costs more to draft;
  - the lossy acceptance frontier is flat at quality (tree_frontier_t16.txt).
- **Lossy:** each set prefers a different K. Pooled over both sets it is a tie: tree-32 leads by 1.5 % on the
  geometric mean, tree-16 by 0.6 % on the harmonic mean. Draft cost tips it to tree-16.
- **Kept per check (dev):** HE+ 11.263 (16) vs 16.263 (32); MBPP 7.487 vs 9.031.

## Consequences

- CERTIFY98 amendment 8's T32/16 power was computed at R 1.480, which was overstated. At E2's 1.433 the bar is
  1.433 x 1.04 x 1.03 = 1.535, against dev 1.531: about a coin flip, not 0.00. Amendment 8a pins R from E2.
- Leap's treek_costs_v2 "onepass" R(32) of 1.478 was built on the same 1.480. The K > 32 values extrapolated from it
  [est] carry the same ~3 % overstatement.
- Monday's "K* = tree-16 on HE+ by ~4 %" (WEEK98) was really ~1-2 %. MBPP decided K* either way.
