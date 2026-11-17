# BREAK98c: think, don't test. Break an ASSUMPTION of the frame (Tue 17 Nov 11:1x; desk sprint to ~12:30; orchestrator)

> (a quote from the project owner, kept private)
more smart new ideas with much less tests than anything"
Target unchanged: >= 550 tok/s lossy at quality on code (775 his stretch), on the full 125B, one stream, this Mac.

## How this round differs
- This morning screened ideas INSIDE the frame tok/s = kept / (check + draft + overhead), and they died on its walls.
  This round attacks the frame's ASSUMPTIONS.
- Desk thinking first. A test is allowed only if it takes <= 10 min AND decides the idea. Otherwise the idea comes with
  its first-principles number and the one test that would decide it.
- Every idea must say:
  1. which assumption below it breaks;
  2. why today's measured walls don't apply to it;
  3. its number, from measured parts;
  4. the single decisive test;
  5. its quality risk.
- Kamen's examples are options, not rules. His fixed constraints: the full 125B as the verifier (never a smaller one),
  one stream, and lossy only at quality (EvalSuite's gates).

## The walls as measured today (quote with headings; WEEK98 Tue 08:1x-11:0x)
- **Kept:** ~17-19 per 32-row check.
  - Own-information cap at 32 rows: 16.26 in expectation (ORACLE-31 18.48 is hindsight).
  - The head locates its misses (AUC 0.954) but lacks the 125B's decision there.
  - No cheap readout finds that decision before the last layers (VERDICT-DEPTH; Fable C: a cost result, not proof that
    the decision forms late).
- **Rows:** a marginal row is >= 0.15 ms of dense compute at the matrix peak (bf16 14 T MAC/s; int8 1.4-1.7x of that).
  The expert tiles absorb rows only up to ~64 (S2).
- **Check:** F32c6i 56.12 ms, F16 38.92 ms, GPU-paced (96-97 %).
  - Per class at 32 rows: experts 26.3, gdn_proj 8.3, hc_read 5.5, gdn_core 4.3, att_proj 2.6, moe_shared 2.2, router
    1.7, att_core 1.1, ple 0.8, hc_write 0.7.
  - Byte floor ~25 ms (C6).
  - Realistic this week: 32 rows ~48, 16 rows ~36.
- **Draft:** 163-178 MB per level; sync-free Q8H32 ~1.1 ms per pop (16.6 ms per tree-16 block); ~0.55 ms fixed + 0.56 ms
  of bytes per pop.
- **Closed today:** partial-width / partial-depth channels, in-check repair, early death, future-state rows, LANDMARK,
  session value prediction, 2B siblings, level pairing, a smaller student head, and two decisions per pass. Do not
  re-propose these without a NEW mechanism.

## The assumptions to attack (pick yours; cross over freely)
1. A check runs every row through all 48 layers, one block at a time.
2. A check re-reads each needed expert from DRAM. Nothing carries over between consecutive checks.
3. The drafter is a separate head that sees the 125B only at the last verified position.
4. Drafting and checking alternate: the next draft waits for this check.
5. Kept is per-position acceptance of a token chain or tree.
6. The answer is produced left to right, one token per position.
7. The weights are what they are (4-bit / MSB-3), read at ~290 GB/s, and the GPU does the work.
8. The GPU computes the experts the way today's kernels do (tiles 2/3 padding, compute-bound).
9. Anything else the frame takes for granted.

## Lenses
- **Lunatic:** 5 + 6. What decode shape, other than per-position acceptance, uses the 125B's full depth better?
- **ZERO:** 7 + 8. What on this chip is unused or misused today? Bytes, compute units, caches, the SLC, format.
- **FAR:** 3 + 4. A drafter or draft schedule that escapes the dependency and the information limit.
- **Fable B:** 1 + 2. Reuse across checks and rows: what does the check recompute or re-read that it already had?
- **Fable C:** audit the walls themselves. Which "wall" is really an assumption? Then invent.

## Output
- Write IDEAS_1117c_<topic>.md in your own folder by ~12:30: one or two ideas, each with the five points above.
- Then ONE line to the orchestrator (NewIdeas is paused until ~17:40): the best idea, its number, its decisive test.
