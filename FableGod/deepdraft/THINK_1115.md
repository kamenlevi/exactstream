# THINK_1115: 30-minute idea sprint, Sun 15 Nov 22:1x -> 22:45

> (a quote from the project owner, kept private)
solve these issues in any way".

The answers go to the orchestrator. Proposals can feed Monday's 15-minute screens.

## Shared context (WEEK98, deadline Sat 21 Nov, "just the tests": certified on fresh data)
**The setup.**
- Exact speculative decoding of the 125B MoE: 48 layers, 512 experts with top-10 routing, hybrid GDN / attention,
  4-bit, a 64 GB M5 Pro.
- "Exact" means the output is bitwise equal to the 125B's own greedy decode (reference REF2-T). "Lossy" means
  EvalSuite's benchmark gate must pass.
- The drafter is t30b, an MTP-style head trained with training-time test.
- Kamen's target: 97.8 % right per guess, to 30 deep.

**Per-band accuracy.** Walk-strict means accuracy at the decode walk's own check starts, i.e. right after a miss.
HE+ dev, kept/32 = words kept per 32-row check.

| | band 0 | 1-2 | 3-6 | 7-10 | 11-14 | 15-22 | 23-30 | kept/32 |
|---|---|---|---|---|---|---|---|---|
| t30b chain | 92.0 | 91.5 | 92.4 | 94.3 | 96.6 | 96.7 | 97.7 | 13.77 |
| TRUE STATE (head given the 125B's exact state at every depth) | 93.2 | 95.3 | 96.4 | 97.8 | 98.8 | 99.2 | 99.3 | 20.77 |
| tree-16 (EAGLE-2, 16-row check) | 98.6 | 96.4 | 93.7 | 95.0 | 96.2 | n/a | n/a | n/a |
| t30b + frozen 2B audit, P = 2 | | | | | | 98.3 | 98.3 | 15.00 |

- **Near-ties.** Check starts sit on the 125B's own near-ties. 92 % of check-start misses have a 125B margin under
  2 nats. The head is a good gate: p1 >= 0.95 is right 99.5-100 % of the time; p1 < 0.5 is right 40-44 %.
- **K98** (TrainLab data/sb98/tree_k98.txt): no exact best-first tree up to 128 nodes holds 98 % per band. Once the
  true path takes a non-#1 step, it ranks below the main paths and only gets #1 continuations.
- **Tree-32 vs chain-16** kept: x1.371 MBPP / x1.531 HE+ (sim). Real decoding on a 2B sandbox MATCHES the sim.
- **Check cost** (Leap check55_plan.md, pipe_price.txt):
  - the exact floor is 38.8 ms at 16 rows and 52.0 ms at 32 rows (14.56 GB; the expert union grows 72.5 -> 108.5 per
    layer);
  - the engine today is 61-64 / 99-102 ms;
  - 4 sibling candidates at one position use ~21 experts vs ~10 for one;
  - the check is GPU- and bandwidth-bound.
- **Kept per check vs tree size** (MTP Head S11, 22:0x; real to K 32, calibrated proxy beyond), MBPP / HE+:
  K16 7.49 / 11.26; K32 9.03 / 16.26; K64 10.16 / 17.91; K128 11.24 / 19.14.
  - Returns diminish: K32 -> K128 is +24 % / +18 %.
  - Best-first shape: a wide band at depths 2-8 (K64 peaks ~5.6 wide at depth 4-5) and a thin tail. 66-86 % of the
    nodes sit off the #1 spine.
- **Tok/s** on HE+, exact at 52 ms / lossy at 25 ms: tree-32 ~313 / ~650; true state ~400 / ~830; a flat 97.8 %
  would be ~445 / ~925.

Read before proposing:
- this charter's top entries (WEEK98.md);
- ~/NewIdeas/research/week98/PLAN4D_REAIM978.md;
- ~/TRIED/TRIED.md (closed ideas). Grep it with >= 5 varied queries before calling an idea "new";
- reports/"Deep drafting and fast verification.md" (prior art).

## The five problems (one per thinker)
1. **WHERE TO BRANCH (fable-zero).** With a K-node budget (32-128), where should branches go to maximise words kept,
   especially at the 125B's near-ties in bands 1-14?
   - EAGLE-2 best-first by path probability is today's best. Calibration and shape tweaks were KILLed.
   - Wanted: a predictor of WHERE the 125B will be near-tied, from signals available before the check, plus a budget
     rule.
2. **CHEAP BIG TREES (fable-b).** A K-row check costs more as the expert union grows. Make 32-128-node trees cost close
   to a 16-row check, EXACTLY.
   - "Overlap-aware draft shapes" was KILLed on paper: it needs a routing predictor.
   - Think: shared prefixes, staged verification, reusing the last check's expert loads, the union's structure.
3. **CHEAP INFORMATION FOR THE DRAFTER (fable-far).** The head is 97.8 % flat with the true state; real drafting drifts.
   How can the 125B's future-state information reach the drafter almost instantly (<= 2-4 ms per 32-token block)?
   - The 2B audit is one route. Others: reuse the check's hidden states, state prediction, cheap partial passes,
     copy / retrieval for code. Check TRIED: self-speculation and early exit were KILLed on bytes.
4. **NEAR-TIES RIGHT AFTER A CHECK (fable-c).** Even the true state gives only 95.3 / 96.4 % at bands 1-2 / 3-6, because
   the 125B is ~50/50 there. The check already computes the 125B's full distribution at every verified position.
   - How can that, or anything else, cover the post-check near-ties exactly and cheaply?
5. **QUESTION THE LIMITS (lunatic).** Which of these walls are real and which are assumptions? Find what changes the
   game for 97.8 % / 400 exact / 1,000 lossy on this Mac by Saturday's certification, or show a wall is real.

## Rules for the 30 minutes
- Desk only: read files and run tiny CPU scripts (<= 1 GB, a few minutes). No GPU, no locks, no runs, no edits outside
  your own folder.
  - Kamen is using the Mac; keep >= 5 GB of memory free.
  - The 45 GB night chain starts at 23:35.
- Output: 3-5 ideas in your own folder, as IDEAS_1115_<problem>.md. Each idea has:
  - the mechanism;
  - the numbers from the records that support it;
  - its tag: published / tried (with the TRIED line) / new;
  - exact or not;
  - what would kill it;
  - a <= 15-minute test design for Monday.
- This folder goes public: no personal or account details, and no product names for hardware we don't own.
- By 22:45, send the orchestrator a <= 15-line summary with your best idea first.

## Added 22:05: the cost side of bigger trees (Leap treek_price.md 05eea11 + MTP Head S11, first cut)
- Measured expert unions per layer: 73 / 110 / 136 / 157 / 187 / 211 at 16 / 32 / 48 / 64 / 96 / 128 rows.
- R(K) = check cost / a 16-row check, one-pass [est]: 1.45 / 1.82 / 2.17 / 2.79 / 3.37 at 32 / 48 / 64 / 96 / 128.
  Byte floors: 0.85 / 1.00 / 1.12 / 1.30 / 1.44 of today's 16-row check. Tree overhead c(K): 1.03-1.16.
- Tok/s, combined (HE+, exact):
  - one-pass est: tree-16 176, tree-32 173, 64 124, 128 82;
  - at the byte floor: 283 / 294 / 239 / 192.
  - MBPP is similar or worse.
- **So today's cost structure puts the best tree at 16-32. Each extra row costs ~0.35-0.45 % of a 16-row check
  (union bytes + cores), while kept rises only ~18-24 % from K32 to K128.**
- Bigger trees pay only if the marginal row gets much cheaper. That is problem #2, and it decides #1's value.

> (a quote from the project owner, kept private)
## breakthrough".
- 450 exact means kept/cycle = 450 / s:
  - at tree-32's kept 16.26, a 36 ms cycle;
  - at the true state's 20.77, 46 ms;
  - at a flat 97.8's 23.15, 51 ms.
  The exact 32-row byte floor is 52 ms. So 450 exact needs ~97.8 %-flat drafting AND a check at or below today's
  bandwidth floor, i.e. fewer bytes per exact check (union, lossless packing, ...), with drafting hidden.
- 900 lossy needs an 18 ms cycle at kept 16.26, 23 ms at 20.77, or 26 ms at 23.15, with the drafter hidden.
- Ideas that move either line are what we want.
