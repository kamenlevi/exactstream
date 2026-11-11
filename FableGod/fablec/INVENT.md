# INVENT — three new SSD-stack ideas (Fable C, mission C1 task 2, Mon 2 Nov 2026, 23:0x EET)

Checked against `~/TRIED/TRIED.md` and `NOVELTY.md`; two first drafts died there (Rnd22, W24-online) and two in tonight's replay (`replay_c1.out`; FINDINGS). Context: at 45 GB the SSD is a **bytes** floor (AUDIT c). Gains halved.

## C1-1 Refined look-ahead: predict layer L+1 twice, the second time from OOO's partial output — EXACT (hiding)

- **Mechanism.** R-57 predicts L+1 once, at L's gate, with L+1's router on L's MoE *input*. In the OOO engine ~90 % of L's output exists after phase 1 (resident pairs + shared expert, `bigmoe/tier.py:152-158`), 0.5-0.8 ms later. Post a second prediction line from the GPU after phase 1; the loader keeps the coarse reads the refined list confirms, cancels unissued ones it drops, adds the new ones into victim slots. Wrong-guess evictions, R-57's killer at C 192, fall with the precision.
- **Prior art.** Pre-gated MoE, ProMoE, AdapMoE, BIGMOE_PF_CORR: all input-side. Feeding the half-finished layer back exists only with OOO; no TRIED / NOVELTY line does it. New [G].
- **Tonight [K].** On cap45's 199 real 16-row checks the gate-time predictor (top-96 by row votes) recalls **0.665** of L+1's union at precision 0.68 — well under the 0.88 that would make a refinement pointless.
- **Gain, halved [G].** R-57 hid 20 % of demand at 0.665 recall. At ~0.9 recall / ~0.85 precision: −6 to −8 ms per cycle at C 320 night → **−3 to −4 ms (+5-7 % at a 56 ms cycle); ~+2 % today at C 256**, where bursts are bandwidth-bound.
- **Test.** No recorded file holds hidden states: a log-only GPU probe in BigMoE's `r57_screen.py` (~10 min) logging per layer the true L+1 union, the gate-time list and the phase-1 list. **PASS: recall@96 ≥ 0.85 and precision ≥ 0.80. KILL: gain < +0.08 recall.**

## C1-2 Q1-on-miss: on drafter-sure rows drop only the *missing* rank 2-9 experts — FAST (Q1's bytes, less damage)

- **Mechanism.** Q1 runs top-2 on rows with cf ≥ 0.99 (code SSD ×0.60, 0.09 % gated flips). Its saving comes only from rank 2-9 experts that would *miss*; resident ones cost nothing. So keep every rank 2-9 expert the published residency table shows resident and drop only the misses. Bytes are Q1's by construction; sure rows keep ~9.8 of 10 experts (base misses ≈ 2 % of uses at C 256 [P, replay]), so flips should fall far below Q1's.
- **Prior art.** Cache-aware / sticky routing on all rows (I6, SWITCHCAP: KILL); Q1 (NOVELTY: new). Adapted: the certainty gate makes the cache-aware cut safe. FAST, same-scores; the output depends on cache state, so runs are not token-repeatable (flag for Kamen).
- **Gain, halved [G].** The quality headroom is cashed as a lower theta: band45 gives code ×0.595 at cf ≥ 0.9 vs ×0.68 at 0.99. **Halved: −6 % bytes → +3 % tok/s over Q1**, with fewer flips than Q1 at 0.99.
- **Test.** Bytes: none needed. Quality: one arm in Breakthrough's map harness (GPU, ~1 h), residency from the cap45 replay. **PASS: gated code flips ≤ 0.05 % at 0.99, ≤ 0.5 % at 0.9. KILL: ≥ Q1's at the same theta.**

## C1-3 Two-stage check: verify the prefix, compute the tail only on full acceptance — EXACT (bytes + rows)

- **Mechanism.** Rows 0..s−1 run as a normal check + commit; only if all s are accepted does a second check run rows s..n−1 from the committed state. s is where the drafter's running confidence drops under P* = tail / (tail + dense) (~0.4 today, ~0.6 at a 17 ms dense), inside today's t 0.3 chain. Exact: REF2-T's batch-invariant kernels make a row's result independent of its companions and the GDN commit replays rows sequentially in fp32, so stage 2 after stage 1's commit equals the one-batch check (TRIED REF2b).
- **Prior art.** Staged speculative decoding (Spector & Ré) stages *drafts*; TriSpec and the 12 GB two-pass check use a proxy (non-exact); t 0.3 cuts from the drafter alone. Here the 125B's own verdict on the prefix decides the tail. Adapted.
- **Tonight [K].** cap45's checks are 16 rows (pre-t 0.3, kept 6.1): at s = 8, P(prefix accepted) 0.27; the tail's exclusive misses 187 MB per rejecting check (16 ms at 11.7 GB/s) + 8 rows × 3 ms. Model +19.7 ms per check if SSD and rows add, **~+8 ms if they overlap**, at dense 34. Overstated: t 0.3 already trims the chain to ~9.1 rows.
- **Gain, halved [G].** **~+3 % today**; more as the dense pass falls and at the 350 cycle, where every tail row's bytes count against the 37 MB budget.
- **Test (CPU).** band45's cf transplant (D-23 cf vectors on cap45, LAYSZ C 256 / 320); stage 2 = rows with cumprod(cf) in [0.3, 0.5]. **PASS: modelled cycle ≥ +5 % (overlap form). KILL: < +2 %.** Then a 10-min live A/B in `r57_screen`.
