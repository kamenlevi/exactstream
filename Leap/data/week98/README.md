# WEEK98: the price of a partial trunk pass on drafted rows (Leap, Sat 14 Nov 16:53-17:0x)

The orchestrator 16:5x asked to price layers 1..L of the 125B for ONE drafted token at a time, sequentially over draft
depth 1..30, on the verified prefix's cache. L = 5 / 15 / 25 / 45, at 45 and 24 GB, with the break-even kept per check
against today (t30b: 13.83 HumanEval+, 6.92 MBPP per 32-row check). The lead is BigMoE cc53cb5 (REPLACE-H: fresh
M(h) at every drafted row, MBPP 23-30 72.13 -> 91.20).

**Result: one fresh trunk pass per drafted token does not pay at 30 deep.**
- At 45 GB, L >= 15 would need more than 31 kept per check on HumanEval+, which a 30-deep draft cannot reach.
- L = 5 needs 27.8 on HumanEval+ (21.8 on a compiled 1-row trunk). That is about the 98 % goal itself (~23).
- On MBPP, L = 5 needs 13.9 / 10.9, against the 10.55 that all five taps reached.
- 24 GB is the same picture: L = 5 needs 23.5 / 19.4 on HumanEval+.
- The cause is that a pass costs ~0.62 ms per layer, flat in rows. Today's 1-row check takes 31.5 ms for 48 layers and is
  bound by the CPU graph build. So 30 sequential passes at L = 45 cost about as much as 30 tokens of plain AR decoding.

**Action: buy freshness in blocks.**
- A k-row pass costs about a 1-row pass: the dense part is flat to 16 rows, and each added row adds ~2.0 ms per 48
  layers. So refreshing every k drafted tokens divides the cost by ~k.
- With block refresh the break-even falls to reachable values (45 GB):

  | L | k | added per check | break-even HumanEval+ | break-even MBPP |
  |---|---|---|---|---|
  | 5 | 8 | +11 ms | 15.3 | 7.7 |
  | 15 | 8 | +27 ms | 17.6 | 8.8 |
  | 25 | 8 | +43 ms | 19.8 | 9.9 |
  | 45 | 16 | +23 ms | 17.0 | 8.5 |

- Next, in BigMoE's REPLACE-H harness (no training, ~15 min): kept versus refresh lag k in {2, 4, 8, 16} with M(h),
  crossed with the per-layer curve now running. Law 2 says lag 2 holds 97.7 % on HumanEval.
- The method's cost-feasible point is the cheapest (L, k) whose measured kept clears its break-even in
  trunk_price.txt.

**Also: today's t30b at a 32-row check is not ahead of the shipped loop.** Estimated at ~100 ms per check at 45 GB, it
gives ~138 / 69 tok/s. The shipped 16-row lossy-45 loop measured 164 / 113 (HumanEval / MBPP). So the deep drafter also
needs the charter's cheap check.

## How it is priced (trunk_price.py; existing numbers only, no run)
- **Why no run:** BigMoE's GPU job held the Mac, and a 45 GB microbench needs a quiet lock.
- **Confirming run, when the Mac is free:** the engine loop's 1-row and 8-row partial passes at L = 5 / 15 / 45, about 5 min
  under a quiet lock.
- **Per layer at 1 row, engine path: 0.62 ms.** The 1-row check is 31.4-31.9 ms (BigMoE CHS5, Sun 11-08, with SB1). It is
  CPU-bound: the build takes ~31.5 of ~33 ms. 48 layers plus ~1.5 ms of head and glue.
- **Fast path (NOT built), an optimistic bound: 0.39 ms per layer.** It is the 1-row bytes floor at 80 %:
  - dense ~66 MB per layer (D32's 32-row dense floor, 12.3 ms at 280 GB/s);
  - plus 10 experts x 2.135 MB.
  - Today's 1-row GPU time is ~0.6-0.7 ms per layer, so this needs a compiled trunk and faster 1-row kernels.
- **Fixed per pass:** 0.6 ms (engine) / 0.3 ms (fast) for embed, ridge, hand-off and one sync (0.22 ms measured).
- **24 GB:** +0.035 ms per layer for a drafted row's misses (W24d: 246 MB of SSD per 13-row check). The bytes are moved
  earlier from the verify, not added.
- **Today's per check (t30b frame), estimated:**
  - 45 GB: 100 ms. That is the measured 16-row HumanEval check at C 256 (61.4 ms, BigMoE rlh_256) plus rows 17-32
    (dense +4.2 from D32, experts +22.6 from NAX6's +1.41 ms per row, misses ~+4, GDN ~+5), plus a ~5 ms chain.
  - 24 GB: 145 ms. That is W24d's 13-row cycle (78 ms) plus 19 rows x ~3.3 ms plus ~4. The 24 GB line runs at most 13 rows
    per check today, so a 32-row verify there is itself unmeasured.
- **Reuse:** the verify can skip layers 1..L on rows a partial pass already did. Rows are batch-invariant in this engine,
  so this stays exact, but it needs GDN-state rollback. It saves (L / 48) x the verify. Full reuse needs the last token or
  block passed too (+-1 pass).
- **Break-even:** kept_new >= kept_today x (T0 + added) / T0. A 30-deep draft keeps at most 31.

## Addendum, Sat 17:49: with BigMoE's per-layer curve, trunk refresh loses to plain 16-row checks at every k
- **The curve:** BigMoE LAYERS a856e7e (MBPP 23-30 / kept per 32).
  - {5}: 41.66 / 2.72; {5,15}: 47.47 / 2.95; {5,15,25}: 56.14 / 3.32.
  - {45}: 91.61 / 10.98; all five taps: 91.52 / 10.82.
  - Early layers are worse than no refresh. Only L 45 is live.
- **At L 45 a refresh pass is a verify in all but name:** 45 of 48 layers over every row of the block.
  - It gets no bonus token, though, and it cannot stop early after a miss.
  - So the fair baseline is the plain k-row check loop with the same drafter, not t30b's 32-row check.

| MBPP, 45 GB | kept | ms per cycle | tok/s |
|---|---|---|---|
| t30b, 32-row check, no refresh | 6.92 | ~100 | ~69 |
| L 45, refresh every 16, upper bound (the per-row 10.98) | <= 10.98 | ~123 | <= ~89 |
| L 45, refresh every 8, upper bound | <= 10.98 | ~176 | <= ~62 |
| t30b, plain 16-row checks (kept from a constant-p fit to 6.92) | ~6.38 | ~66 | ~96 |
| shipped dfl, 16-row loop, measured (BigMoE rlm_256) | 6.51 | 56.1 | 112.7 |

- **L 45 refresh every 16 would need kept >= ~11.8 to tie t30b's plain 16-row loop.** Per-row fresh M(h45), the best any
  refresh schedule can keep, gives 10.98.
  - This holds unless t30b keeps <= ~5.9 within 16 rows. BigMoE's exact per-depth curve settles that.
- **Result -> action:** trunk refresh is closed as a speed lever. Early layers don't carry the state, and layer 45 costs
  a verify.
  - The 98 % method must make the drafter predict the deep state itself.
  - Cheapest screen (BigMoE's harness, ridge, no training): map [the carried stand-in, raw taps {5}] -> wd, instead of
    the taps alone. Refreshing L 5 every 8 costs +11 ms.
  - It pays only if that combined map clears its break-even (MBPP 7.7 kept at k 8). The taps alone gave 2.72.

## Update, Sat 18:0x (BigMoE 7d6e935): both trunk-freshness routes are closed
- **COMBO5 = CLOSED.** The ridge [carried stand-in, raw tap 5] -> wd reaches R2 0.448.
  - With t30b on MBPP 100, refreshing every 8 depths gives 23-30 65.85 and kept / 32 6.50. That is under the 7.7
    break-even and under t30b alone (6.92). Refreshing every depth gives 3.26.
- **t30b's exact kept per 16-row check on MBPP is 6.59** (BigMoE's per-depth eval, kept_walk capped at 15 drafts). My
  constant-p fit gave ~6.38.
  - So t30b's plain 16-row loop is ~99 tok/s (6.59 / ~66 ms), and L 45 refresh every 16 would need kept >= ~12.2 to tie.
  - The per-row upper bound is 10.98, so the L 45 verdict stands, more strongly.
- **BLOCK** was stopped after {5} / {5,15} at k 2 (23-30 46.23 / 52.44).
- **Result -> action:** freshness from the trunk is closed for WEEK98.
  - Layer 45 costs a verify, and layers 5-25 (with or without the stand-in) don't predict the deep state.
  - The confirming microbench offered at 17:01 is withdrawn: no decision depends on its numbers now.

**CORRECTION (Sat 18:1x, cheapcheck.md):** "CPU-bound graph build" above is the wrong label for the 0.62 ms per layer.
It is the 1-row check's wall per layer with SB overlapping the build, i.e. GPU time. The CPU build is ~0.21 ms per layer
(10.1 ms per 1-row check, rli_256). The numbers stand as wall-clock, and the trunk verdicts are structural. The "fast
path" bound (0.39) is a GPU kernel bound, not a compile-the-graph bound.
