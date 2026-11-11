# The wired-limit fix (FIRSTCHK) does nothing on the 12 GB loop: x1.004 whole decode; its slow first checks are cold expert pools after prefill, not un-wiring
- Date: 2026-11-07 15:5x EET. Session: Fit12.
- Claim: mx.set_wired_limit(max_recommended_working_set_size) once after load (Leap's FIRSTCHK fix, +27.4 % whole decode at
  45 GB, BigMoE S6W) leaves the 12 GB FAST candidate t80ck1u unchanged. The 12 GB loop's first-check excess remains with the
  fix, so it is not the IOGPU wired collector un-wiring the pools during prefill.
- Number [K], FAST arm t80ck1u, 12 GB line, 16 HumanEval prompts, cap 256, each arm alone in its own process, one quiet lock,
  W first then B, no PAUSE, swap flat:
  - whole decode: W 46.34 vs B 46.16 tok/s geometric mean (W / B x1.004 per prompt); pooled 44.34 vs 44.21;
  - full-period steady x1.001; end to end x0.990 (W's prefill +2.7 %, W ran first);
  - check 0: 850 / 861, 888 / 914, 899 / 925 ms ... (W / B) when it goes to the exact pass; excess over the later same-mode
    checks 3.64 s (W) vs 3.82 s (B) over 16 prompts = 4.7 % vs 4.9 % of decode;
  - lifetime peak 11.242 vs 11.244 GB; tokens W = B on 16 / 16 (and = the 08:19 run).
- Reading: at 12 GB the pools are 6 GB, and the excess appears only when check 0 goes to the exact pass (~500 ms over the
  later exact checks; pass-1-certified first checks run ~10-40 ms over). It looks like the first exact pass after prefill
  reading ranks 2-9 into pools that the prefill left cold. That is an expert-cache warm-up, which wiring cannot remove.
- How: /Users/kamen/Fit12/scripts/wab.sh (fit_t3.py F12_WIRED=1, sha256 ec0964f190b7), /Users/kamen/Fit12/scripts/wab_cmp.py;
  data /Users/kamen/Fit12/data/scr/wab (wab_cmp.txt, wabw / wabb reports).
- Prior art: adapted (mlx-lm's wired_limit; Leap's FIRSTCHK diagnosis at 45 GB).
- Status: measured once (W then B). KILL at 12 GB.
- Next (not routed): a prefetch of the first check's ranks 2-9 during the prefill's tail would target this excess
  (<= ~4.9 % whole decode [P]).
