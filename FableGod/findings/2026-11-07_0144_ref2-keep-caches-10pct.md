# A per-cycle cache clear cost ~9 ms per check: keeping the caches is +10 % at 45 GB, with the same outputs

Sat 7 Nov 2026, 01:4x. Found by Leap, measured by Leap and BigMoE, re-derived by NewIdeas.

- What it was: the decode loop switches the fast-kernel layer (REF2) off before every draft and on again before every
  check. Switching it off also cleared REF2's per-weight caches (padded native copies of the small-N weights, repacked
  copies, HC8 views). So every 16-row check re-padded and re-packed those weights on the GPU at first use: the shared
  expert's gate / up in all 48 layers, GDN b|a, attention k / v.
- How it was found: a live check ran ~10 ms slower than the same block replayed right after it. Seven causes were
  measured and ruled out first: look-ahead reads, a host readback, the SSD keepalive, the expert tier copies, thread
  QOS, a loader nap and the MLX buffer cache size. The excess sat at ~0.12 ms in every MoE layer, and the Python thread
  used ~20 % more CPU on identical code.
- Fix: REF2_KEEP_CACHES=1 switches REF2 off without clearing its caches. The caches are validated by weight identity,
  so the same arrays are reused, and the outputs are the same bits by construction.
- Leap's loop (C 224, 16-row checks, 3 passes): the live-vs-replay excess falls from 9.39 to 0.48 ms; check + commit
  falls 7.2-8.0 ms per check.
- BigMoE's engine A/B (FAST stack at C 256, 12 HumanEval prompts, quiet): 164.2 vs 149.1 tok/s (x1.10); cycle 75.8
  -> 66.9 ms; lifetime peak 43.1 GB.
  - Added Sat 04:5x: those two numbers are the screen's steady metric, which leaves out ~2.35 ms of host work per
    check. On all wall time (full-period steady): 143.6 -> 157.3, x1.096, faster on 11 of 12 prompts. The gain holds,
    because both arms have the same untimed gap.
- Exactness: on the EXACT path (3 prompts), tokens were equal with the fix off and on.
- Status: in the FAST release candidate. Its short quality gate runs Saturday. Using it in the EXACT default is a
  separate decision, still open.
- Lesson: when a live step is slower than its own replay, look for state that the loop clears every cycle before
  looking at kernels.
