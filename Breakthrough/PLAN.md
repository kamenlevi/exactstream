# Breakthrough plan (account-1 horizon 18:00 Thu 2026-10-22; updated 17:06; goals: STATE.md "Breakthrough")
Bar (Kamen 2026-10-21 17:10): decode TASK results. Line: frugal on a pinned copy of BigMoE v3 (v3f). HANDOVER "STATE" = truth.
B12 (load_b12(0.10, mtp=True) + B12_NORM=mass): FAIL as judged with the guard; BORDERLINE once the guard's math damage is
removed (NewIdeas re-score). B8 verdict (eager, guard off both sides) lands ~19:00 in ~/EvalSuite/results/report_b8.txt.
  RAM: B12 + MTP 11.3 / 12.9 / 14.6 GB at 2.6K / 32K / 64K; B8 8.0 / 9.6. Speed: B12 + MTP 35-39 / 24 tok/s code / prose.
1. Read B8's / B8-fast's verdicts; record them; finish RESULT_2026-10-22.md.
2. Like-for-like B12 verdict: guard R1 or none on both sides; two runs per side or a per-row pool reset (new ext function).
3. If quality headroom is needed: IDEAS D1n (4-bit pools, cold ranks late), tf_b12 KLD first, kill line <= ~0.09.
4. Speed: D1m (multi-row phase B for late ranks under MTP; v3f ext rebuild when free); TrainLab's per-pass expert budget.
5. 8-bit KV for long context in 12 GB.
NOT doing: prefetch (D1h), prefill look-ahead (D1k), RB2 union path, skip-residents, adaptive chunks, frugal prefill of tool
outputs (D1l), streaming as default, C_m 46, pre-routing, late rank 1 / HC4; edits outside ~/Breakthrough; external GPUs.
