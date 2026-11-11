# PREREG: the agent FAIL's cause, one FAST lever off per arm (Fri 2026-11-06 11:28; before any row)

Written before any row of this bisect.
- **Request:** NewIdeas (Fri 11:2x, QUEUE 70a4951) after the agent check's FAIL (RESULT_agentfast).
  - Plain FAST put 85 blunders into 2 x 17 long agent turns, most in the long file writes.
  - NewIdeas' prior is QOM-KEEP: the only history-dependent lever. The two orders diverged by token 1-6.
- **Turns:** agentlong/write12, write13 and write14, the worst three (16-20 % of tokens off the reference's argmax). One
  process per arm and order holds all three turns.

## Arms
All arms run plain FAST as the agent check did (frozen outside code snap/r57frz-af1, NAX6-N, C 224, CAP 1024), with one
lever off. BigMoE gives each arm's exact env change; it is added under "Recipe" before any row.

| arm | lever off | orders |
|---|---|---|
| base | none: plain FAST rerun, so the arms compare with the same chunking | forward + reverse |
| qom | QOM-KEEP | forward + reverse |
| d17 | DRAFT17's bonus row | forward |
| hc8 | HC8 | forward |
| nax | FAST-NAX's base (back to 3c) | forward |
| lctx | the long-context pieces (DKV / att_single past 512 keys) | forward |

- **Scoring:** stage `tfwin` against the arm's own exact base. That is NAX6-N (`ref2tnaxn`, self-check on results/ref2tnax)
  for every arm but `nax`, which takes REF2-T (`ref2tf0` on the converted store, self-check on results/ref2t).
- **Order:** `qom` first, then `base`, then the others, so the prior's test lands first if time runs short.

## The lines (NewIdeas)
- **Per arm:** its forward blunders on the three turns, against `base`'s forward blunders.
  - <= 20 % of base: this lever carries the failure.
  - >= 50 % of base: it is not the main cause.
  - In between: partial.
  - If base has fewer than 10 forward blunders, the ratios are reported but called weak.
- **The sharp test (`qom`):** with QOM-KEEP off, the two orders must be token-identical on all three turns. That would
  confirm QOM-KEEP as the history-dependent lever. `base`'s two orders are expected to differ.

## Information
- Deviation rates per arm and order.
- Each arm's tokens vs base's: the first divergence.
- Blunders by tool-call part (the command / code vs the description).

## Runs
- Today by day if it can start by 14:15 (NewIdeas), else Saturday by day as the only big job.
- 8 FAST processes (~4-6 min each) + 8 tfwin passes (~3-4 min each), about 1-1.3 h.
- Share lock "MEM 39 GB", runner 48's gates and freeze checks.

## Recipe (BigMoE Fri 11:3x, checked against the frozen screen and its frozen ref2; added 11:36, before any row)
Each arm is runner 50's FASTENV with one change (results/bisect/arms.txt):
- **qom:** R57_QOM_SETS=humaneval,ifeval,multi (agentlong dropped).
  - HC8, DNR, HCB, SMALL-N, NAX-1CQ, D17 and D7BNAX stay on; they are load-time installs on every prompt.
  - QOM's rule is off for agent turns: exact acceptance, so no resident keep and no 3-bit tails.
  - Checked per process: no QOM optional uses on any agent run.
- **d17:** R57_D17 dropped. QOM then keeps the bonus row exact.
- **hc8:** both R57_HC8 and REF2_HC8_FAST dropped. ref2 turns on its 8-bit path from REF2_HC8_FAST alone; R57_HCB
  stays (exact).
- **nax:** both R57_NAX and REF2_NAX_1CQ dropped (never 1CQ alone). The base is REF2-T's tiers on the converted store,
  scored against REF2-T's own reference (`ref2tf0`, self-check on results/ref2t).
- **lctx: not applicable.** The FAST screen has no DKV. att_single is off unless R57_FASTTAX, which FASTENV does not set.
  The 512-key buckets are the base's own, as in the reference.
- **small (replaces lctx):** REF2_SMALL_NATIVE dropped. ref2 calls SMALL-N "FAST, off by default".
- **History:** BigMoE says d17 / hc8 / nax / small are all history-free. If qom's two orders are identical, the other
  arms can move only the blunder count.

## Status Fri 12:29: the nax arm voided; a full freeze for the rest (before the remaining rows)
- **Done on freeze af1:** qom (fwd + rev), base (fwd + rev), d17, hc8 are run and scored.
- **The void:** the nax arm loaded Leap's p11.py from the live tree. On the 3c base it loads tierglu_3cn.py, a frozen
  module the reference never loads, and its "~/Leap/scripts" path was not rewritten, because af1 rewrote only the
  reference's modules.
  - The live p11.py has the reference's content (cba73e517b61), but the rule voids it. Its tokens are not used.
- **The fix:** freeze af2 (tools/freeze_r57.py now rewrites every .py file of the frozen trees). All 33 reference
  modules are identical between af1 and af2 once the freeze directory's name is swapped, so af1's arms and af2's arms run
  the same code.
- **The rest:** nax and small run on af2.

