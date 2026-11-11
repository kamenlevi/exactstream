# The fixed cost of one check is 54 ms, and by itself it caps code near 220 tok/s at 12 kept per check
- Date: 2026-11-06 01:33-01:37 EET. Leap ran it under a quiet lock, at C 224, starting with NO-SWAP 48.2 GB; swap stayed flat
  at 11.1 GB. Fable ZERO built the test and its lines. Recorded by the orchestrator.
- Claim: with every owned FAST lever on, the fixed cost per check does not come down. It stays where Fable ZERO's model put
  today's engine. The pre-registered lines were PASS <= 30 ms and KILL > 36 ms; 350 tok/s on code needs <= ~25 ms at 12 kept.
  The result is a KILL.
  - FLAT = the steady 2-row cycle minus 2 x the warm per-row slope.
- Number [K, GPU]:
  - FLAT 53.7 ms for the Friday all-in, 55.2 ms with DKV + GDNB added.
  - Cycle 59.8 ms (IQR 59.3-60.2) = draft 9.1 + check and commit submit 50.6 (graph build 44.7, eval wait 5.6).
  - Warm 2-row check, timed piece by piece with one sync per piece (111.1 ms in total):
    - GDN 15.7, experts 17.4, HC reads 11.0, attention 8.6, router + shared 5.7, MoE tail 3.5, embed + norm + head 2.3.
    - Pure CPU graph build: 13.4 ms.
  - Only the experts grow with rows: +3.0-3.3 ms per added row (885 expert groups per 2-row check). Every other piece is flat
    from 2 to 16 rows.
  - Without the draft, the dense part is ~44.6 ms. ZERO's model of today's engine said ~47 ms; the measurement agrees.
- So what [P]:
  - At 12 kept per check, the fixed cost alone caps code at 12 / 54 ms = ~220 tok/s, even with zero expert cost. Today's
    owned levers cannot reach the 250-330 tok/s target on code at 45 GB.
  - Both parts have to fall, and neither reaches the target alone:
    - Today's expert cost on real QOM blocks is ~2.2 ms per kept token, as ZERO's model said. The 3.0 ms per added row
      below is the all-resident slope (corrected Fri 03:4x, Leap). At ~2.2 ms, 250 tok/s at 12 kept needs a fixed cost
      <= ~21 ms.
    - Suppose the expert part reaches ZERO's split-form target (0.77-0.9 ms per kept token on NAX6-NH). Then 250 tok/s needs
      a fixed cost <= ~37 ms, and 330 needs <= ~26 ms. At 54 ms, that same expert target gives ~185 tok/s.
  - For scale: FLAT-2's real-block passes ran ~104.5 ms per check at 14.4 kept, ~138 tok/s, with a few misses. That was one
    HumanEval prompt (#68), a workload that prompt copying flatters.
  - Fri 03:4x, with the split live: computed expert groups fell to 0.63x / 0.49x of plain FAST's, and tok/s did not rise at
    C 224 (BigMoE, 12 HumanEval prompts; NOT PASS, heat-disturbed; clean first passes 96.3 / 95.8 / 99.2 tok/s).
    - Check + commit fell only 6 / 5 % (98.8 -> 92.9 / 93.7 ms), and kept tokens fell 4.2 / 1.7 %, so the net is
      ~0..+1.5 % (NewIdeas, Fri 04:1x). Prices from expert tiles had predicted +11-19 %.
    - So the expert GPU time is mostly per-row work, not per-group reads: 11.1 ms at 2 rows, 57.0 ms at 16 (Leap,
      all-resident). NewIdeas took the split out of the FAST all-in; split 0.99 is re-tested once the fixed-cost cuts
      bring the cycle to <= ~80 ms.
  - The check is GPU-bound (Leap, Fri 03:5x, from the same synced records). Each piece's GPU time is its synced time minus
    the 0.133 ms round trip. They sum to 47.8 ms, about the whole unsynced check, so the 13.4 ms CPU graph build is hidden
    behind GPU work.
    - GPU ms per 2-row check: GDN 13.2, experts 11.1 (~4.6 of it fixed), HC reads 8.6, attention 6.4, router + shared 5.7,
      MoE tail 1.1, HC write 0.8, PLE 0.65.
  - Launch gaps are only ~1.5-3 ms in all (~1.3-1.6 us per dependent dispatch on M5). Most of the time is spent inside
    kernels running at low efficiency: ~2.7 GB of dense bytes per check move at ~60 GB/s, against ~250 GB/s possible,
    across ~1,000 small kernels.
  - Leap's pre-registered flat-cut plan re-prices to -6..-12 ms (lazy commit, HC reads, GDN, the experts' fixed part, a
    FAST-tax switch). That gives FLAT ~42-48, or ~40-46 with the draft, and likely misses the 37 ms line.
  - The probable route is structural: fewer, larger kernels per layer. A one-layer bench decides between fusion and better
    small-row kernels.
- Caveat: the split (0.9) was inert in both arms, because the one-call path bypassed it (fixed Fri 02:58, QOM-SPLIT-1C). So
  this is qall's fixed cost. The split acts on the per-row expert cost, not the fixed part, so FLAT is unaffected.
- How:
  - Runner and lines: ~/FableGod/zero/build_1105, LOSSY350_ZERO.md section 4.
  - Data: ~/Leap/data/flat2/. Record: TABLE_350 'FLAT-2 RESULT' (Leap c91fdb7) and the TRIED line in ~/TRIED/leap.md.
- Prior art: speculative-decoding cost models split a step into a fixed part and a per-token part (Leviathan et al. 2023,
  the draft-to-target cost ratio c). Our numbers are a per-piece measurement of a hybrid GDN + MoE check on an Apple GPU,
  with a 13 ms CPU graph build per check. Not searched outside our notes yet.
