# THINK_1116: 45-minute sprint on the three walls, Mon 16 Nov 12:0x -> 12:50

> (a quote from the project owner, kept private)
percent clear. I mean yeah - no compromise. And all until Saturday".

He also lifted the day limits on the Mac's compute until he says otherwise. Answers go to the orchestrator and feed
NewIdeas' 14:00 plan.

## The 900 budget (code; HE+ unless noted; one stream; lossy allowed, EvalSuite-gated)
tok/s = kept per check / (check ms + draft ms).

| | today (measured or priced) | needed for 900 |
|---|---|---|
| kept per 32-row check | 16.26 exact tree-32; 16.91 lossy K* (tree-32 + close-call); 20.77 true state | >= 23.15 (flat 97.8 % to 30) |
| check | lossy C5: ~61 ms as two 16-row calls; 49.8-53.3 one-pass (not built); ~41-45 on this month's ladder; byte floor 21.9 ms (6.12 GB on code) | <= 25 ms |
| draft | the t30b head reads 177.6 MB per level at 0.66-0.92 ms per level: tree-16 8.3-11.5 ms (12.6 levels), tree-32 13.8-19.1 (20.8), K128 26-36 (39.6) | ~<= 3-5 ms |

- At kept 23.15, 900 needs a 25.7 ms cycle. At kept 28 it needs 31 ms.
- The check is bandwidth-bound, so draft bytes add in full: they can't hide behind it.

## Facts from today (records: WEEK98.md top entries)
- **The lossy check's parts** (Fit12 LOSSY32, C5 at 32 rows):
  - FAST dense side 23.8-26.6 ms (3x its 7.6 ms byte floor; ~5 ms of cores);
  - experts 20.7-21.4 ms (1.43x their floor);
  - 45 GB misses 5.3 ms;
  - CPU build + encode + commit 27.9 ms, as a bound: 4,022 dispatches per one-pass 32-row check, 52 % of them MLX glue.
  - Best case, all at once: ~29.8 ms GPU.
- **Exact:**
  - the exact 32-row byte floor on code is 47.3 ms (REF2-T plane reads);
  - engine R32 = 1.480 (one-pass 32 rows 90.46 ms vs 16 rows 61.11 ms).
- **Real 127-node trees** (under the walk):
  - HE+ kept 11.26 / 16.26 / 20.77 / 23.70 at 16 / 32 / 64 / 128 rows; MBPP 9.03 / 10.19 / 11.29 at 32 / 64 / 128.
  - The extra is long runs (HE+ docstring copies). Walk-strict bands 3-14 stay 93-96 % even at 127 nodes.
- **Early death** (Fable B): a K128 check at 48-54 ms by bytes. With drafting NOT hidden that is 251-303 tok/s HE+ at
  the byte floor; the probe is untested.
- **Drafting:**
  - Post-hoc byte cuts of the head lose kept (M3): 8K screen x0.963 HE+ at tree-32; 4-bit trained parts x0.922;
    C1 x0.875.
  - A hand-fused level kernel cuts drafting ~30 %.
  - Speculative next-block drafting is a paper KILL (HE+ hit 27 %).
- **Drafter training:**
  - S10 PASS2: more fresh data does not lift HE, so t30c is off.
  - L1CF: the distilled 2B auditor gained +0.82 (KILL).
- **Lossy acceptance levers on record** (~/TRIED/TRIED.md):
  - close-call delta 1 / 2 (tonight's EvalSuite gate);
  - INERT acceptance inside comments + docstrings: replay x1.17-1.30 HE+ fresh, x1.63-1.79 MBPP; quality untested.
    On the 2B, a relaxed comment changes the next code in 48 % of cases;
  - SPANRELAX-C (top-3 inside comments): HE+ x1.074, MBPP x1.125;
  - COMMENT-FREE mode: kept x1.23-1.29 [P].
  - In-comment rows hold 71 % of HE+'s near-ties (the slot prior).
- **Dispatch diet** (Leap):
  - glue fusion: -11.7 ms CPU (1.5-2.5 weeks);
  - Python specialization: -1.5-2 us per dispatch (3-5 days);
  - together the CPU gets to ~12 ms. <= 5 ms needs replayed command buffers or megakernels (3-6 weeks).
- **1c-tiers** (this week's lossy check bytes) was built today; its key test passed, and EvalSuite's gate is running
  (V0 GO).

## The five problems (one per thinker)
1. **fable-c, WALL A (acceptance).** Get >= 97.8 % per guess to 30 deep on code, lossy allowed. The exact ceiling at
   bands 1-6 is the 125B's own near-ties.
   - Which lossy acceptance rules reach 97.8 % per band while code tests stay unchanged (EvalSuite's line)?
   - How do we prove the quality by Saturday?
2. **fable-zero, WALL A x the budget.** Find the joint shape (trees, chains, lossy acceptance, early death, comment
   rules) that maximizes kept / (check + draft) under the REAL draft cost, which grows with tree depth.
   - What gets kept >= 23-28 per check inside a ~26-31 ms cycle?
3. **fable-b, WALL B (the check).** A measured <= 25 ms lossy 32-row check by Friday.
   - Beyond the ladder (one-pass form, dense kernels, experts, misses, CPU): what cuts bytes or time further? Fewer
     experts per check row, lower-bit tiers, work shared across rows, work skipped.
   - What can be built and measured in 4 days?
4. **fable-far, WALL C (drafting).** <= 3-5 ms per block at t30b's accuracy.
   - A smaller drafter trained to keep accuracy: which shrink? Distillation? A multi-token-per-step head?
   - Or a draft source that costs no bytes: copy / lookup for long runs, or the check's own states.
   - What is trainable by Thursday?
5. **lunatic, QUESTION THE LIMITS.**
   - Which of the three walls is an assumption?
   - What does 900 tok/s need in bytes per token, and is it physically possible on this Mac (~273-300 GB/s)?
   - What would an honest 900 demonstration on code look like by Saturday? What changes the game?

## Rules for the 45 minutes
- Desk only: read files and run tiny CPU scripts (<= 1 GB, a few minutes). No GPU, no locks, no runs, no edits outside
  your own folder.
- Output: 3-5 ideas in your own folder, as IDEAS_1116_<problem>.md. Each idea has:
  - the mechanism;
  - the numbers from the records that support it;
  - its tag: published / tried (with the TRIED line) / new;
  - exact or lossy;
  - what would kill it;
  - a <= 15-minute test for today or tomorrow;
  - its expected gain in the 900 budget's terms (kept, check ms or draft ms).
- This folder goes public: no personal or account details, and no product names for hardware we don't own.
- By 12:50, send the orchestrator a <= 15-line summary with your best idea first.
