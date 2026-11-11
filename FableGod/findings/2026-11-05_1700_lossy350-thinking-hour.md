# What eight parallel thinkers found on the road to 350 tok/s lossy: today's code number is two speeds, chat's wall is acceptance, and the fixed cost per check decides 350
- Date: 2026-11-05 17:00 EET. Sessions: eight Fable thinkers (A, B, C, FAR, FULL, GOAL, ZERO, Lunatic), one wall each, in
  one hour. Recorded by the orchestrator.
- Claim: no single lossy lever gives 3x. Three CPU-measured facts change how we read our own numbers, and one inequality
  says where 350 on code is decided. Every speed number below is a CPU replay or arithmetic ([P]); nothing here ran on the
  GPU yet.
- Number:
  - Two speeds [K, FULL, replay of the engine's 12 HumanEval prompts at CAP 256]: 67.8 % of generated tokens are one
    verbatim copy of the prompt (the re-emitted signature + docstring). At ~6.5 ms per copied token and 13.8 ms per other
    token, that reproduces the engine's measured 110-117 tok/s. So today's FAST code line is ~155 tok/s on the copy and
    ~72 on everything else. MBPP, math and IFEval answers copy 1-3 %.
  - Expert independence [K, Lunatic, real 4-bit weights, layers 3 / 23 / 43; Fable C, expert outputs on 176 real
    checks]: no shared base, no common subspace, no near-duplicate neurons, no low-rank experts. Routed expert outputs on
    one input are near-orthogonal (mean cos +0.02), so substituting a computed expert for a missing one equals dropping it.
  - The bar's tie floor [K, Fable C, ~75 decision-map arms joined with the 125B's row class]: rows with p1 - p2 <= 0.1
    (1.5 / 8.5 / 7.0 % of code / IFEval / multi rows) flip 20-45 % under any perturbation. 4-bit attention weights alone
    give 0.92 / 4.53 / 3.43 % all-row flips, against lines of 1.34 / 5.42 / 3.69. Sure rows almost never flip.
  - Chat is an acceptance wall [K, Lunatic + Fable C]: when every row of a check is kept, chat routes fewer experts per
    token than code. Chat's cost comes from short checks (~2 kept) and rejected rows. A 2x cut of expert groups on chat
    fails the map's bar (floor ~x0.8).
  - The 350 condition [P, ZERO + Fable B]: flat + e x K <= 2.857 ms x K. With the split form, the fixed cost per check
    (dense + draft + commit + SSD wait) must fall to ~20-25 ms at ~12 kept. Measured today: ~39 ms plus SSD waits, ~76-78 ms
    by day. The dense pass is flat in rows and made of dependent stages, so lossy row approximations don't shrink it.
- Leads, untested, each with a <= 10-minute pre-registered test:
  - Split + a uniform rank cap: code expert groups x0.43-0.52.
  - Wide strict copy passes for re-emitted prompt text: x1.4-1.6 on HumanEval's window, ~0 elsewhere.
  - Wholesale acceptance inside comments and docstrings: x1.6-1.8 on MBPP.
  - A comment-free FAST code mode: about x1.2-1.3 kept per check.
  - Never-wait for the SSD: deferral, re-opened at 45 GB; 95 -> 112-120 tok/s.
  - Half experts: cost per group x0.64-0.81.
  On paper all three code levers together give ~290-375 tok/s on HumanEval-style code and ~250-330 on MBPP-style code.
- So what: a 350 claim has to name its workload, because HumanEval's re-emitted prompt makes it the easiest code test.
  The decision map should report non-tie flips beside all-row flips. The next decisive measurement is the fixed cost per
  check with every owned lever on: PASS <= 30 ms means 350 on code is live; above 36 ms, code tops out near ~300.
- How: ~/FableGod/{LOSSY350_A.md, fableb/LOSSY350_B.md, far/LOSSY350_FAR.md, far/NEVERWAIT_TEST.md, goal/LOSSY350_GOAL.md,
  zero/LOSSY350_ZERO.md, full/LOSSY350_FULL.md, fablec/WALL_1105.md, lunatic/WALL_L_1105.md}, with scripts and outputs
  beside each. TRIED lines were added the same day.
- Prior art: copying from the prompt is published (prompt-lookup decoding). Deferring missed experts is published
  (KTransformers' expert deferral; our own B12LL / DH_W4). In-expert sparsity is published (TEAL, CATS). Batching streams
  is published. Not searched outside our notes yet: the two-speed reading of HumanEval tok/s, the tie-floor reading of the
  map's bar, and the comment-span view of drafter misses.
