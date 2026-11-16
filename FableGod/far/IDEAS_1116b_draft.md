# IDEAS_1116b_draft: a drafting shape for lossy kept x1.3 (Fable FAR, Mon 16 Nov 2026 17:4x-19:0x, desk + CPU screens)

PUSH800 section 3, FAR's lens. Marks: [K] measured (source), [P] arithmetic on [K], [G] judgement. Screens ran on the
real 127-node dumps (BigMoE, `tree_t30b_heplus24_n127k8.npz`; MBPP part 0 of 5), CPU, one process, < 1 GB, scripts in
`~/FableGod/far/screens_1116b/` (`miss_types.py`, `adaptive_rows.py`, `adaptive_rows.out`).
Counting: kept(rows) = the deepest on-true node among the first `rows` best-first pops + 1 (bonus); "walk starts" =
the dump's real check starts (461 on HE+), "all anchors" = every row (6,375). My walk-start tree-32 kept is 14.25,
the record's 16.26; ratios below are within one counting rule, so quote the ratios, not my absolutes.

## 0. What the first misses of tree-32 paths are (HE+, all anchors, `miss_types.py`)  [K, new measurement]
Of 5,910 resolvable anchors: **3,137 (53 %) end the 32-pop path at the CAP, not at a miss** (the true child is the
head's top-1 but the 32 pops are spent; 76 % of those at depth >= 31). 2,773 end at a true miss. Of the misses:
- the true token is the head's #2 in 57 %, within its top-3 in 72 %; the head's p1 there is median 0.68 (40 % under
  0.6): the head KNOWS it is unsure, the tree cannot afford a second child at every such node within 32 rows;
- tokenization splits (same characters, different split): **3 of 2,773. KILL** for any tokenization-equivalent
  acceptance;
- one-token substitutions whose suffix re-syncs for >= 4 tokens: 322 (11.6 %), half of them identifier swaps
  (`is_palindrome` vs `make_palindrome`, `string[::-1]` vs `prefix[::-1]`): accepting them is wrong code. KILL as a
  lossy acceptance; the re-sync is only a draft-reuse lever (saves draft ms, not kept);
- insertions / deletions 3 %.
So on HE+ the kept wall inside a 32-row check is ROWS, not the head: half the blocks want depth past 31, the other half
want width at nodes the head already marks unsure. A fixed 32-node best-first serves both halves with one budget.

## 1. ADAPTIVE BLOCK LENGTH: extend the chain to 64-96 rows only when the head's 32-pop spine stayed sure  [new; exact]
**Mechanism.** Pop the best-first tree to 32 nodes as today. If the 32 pops reached depth >= 31 (the head's path_p
stayed high: the pops formed a chain, a signal the head has at draft time, no truth needed), keep drafting the spine
to 64 (or 96) rows and CHECK ALL ROWS IN ONE PASS. Otherwise check 32. The extra rows are the cheap ones: at 32 rows
the check pays the whole dense pass and the expert union; a row past 32 costs ~0.15 ms dense (int8, compute-bound)
+ ~0.08 ms expert-union growth (x1.19 bytes for 64 rows [K WEEK98:604]) + ~0.04 cores = **~0.28 ms per row (band
0.19-0.45)** against the 32-row average of 0.68 [P on PUSH800 section 1].
Fixed 64-row checks do not pay (Sandbox98 depth-63 [K WEEK98:2138]); my screen reproduces that (fixed-64 x1.00 tok/s
on walk starts). The adaptive form pays because the extension goes only to the blocks that would keep ~29 of 32.

**Numbers (HE+, `adaptive_rows.py`, walk starts n 461) [K].** Kept 32 / 48 / 64 / 96 / 127 pops = 14.25 / 17.59 /
20.07 / 22.95 / 24.16. Spine >= 31 in 32 pops: 27 % of blocks; their kept32 is 28.8 vs 8.9 for the others.
| policy | kept per check | tok/s vs fixed-32 at 0.28 ms/row | at 0.19 | at 0.45 |
|---|---|---|---|---|
| extend to 64 when spine >= 31 | 19.14 (x1.34) | **x1.21** | x1.26 | x1.14 |
| extend to 96 when spine >= 31 | 21.33 (x1.50) | **x1.23** | x1.32 | x1.10 |
| fixed 64 for every block | 20.07 (x1.41) | x1.00 | x1.08 | x0.89 |
All anchors (n 6,375): extend-to-64 x1.20, extend-to-96 x1.17. **MBPP (part 0 of 5, `adaptive_rows.out`): the lever
is nil, x1.00**: sure spines (depth >= 31 in 32 pops) are 0.7-0.9 % of blocks, kept32 is 7.2 (walk starts) / 8.4
(all), every block is miss-bound. The lever is HE+'s long sure runs (docstring copies), as WEEK98:140 warned about the
K32 -> K128 lift; on MBPP kept x1.3 has to come from the head's accuracy or a lossy rule, not from rows.
Own-state drift past depth 31 is INSIDE these numbers (the dump is the head's own rollout).

**In the 775 frame.** HE+ only: kept x1.34-1.50 per check at check cost x1.11-1.22: x1.21-1.23 tok/s, exact, before
any lossy rule; MBPP unchanged. Stacked with close-call delta 1 (+21 % kept on the 2B [K TRIED 1095]) it reaches the x1.43 lower edge of
NewIdeas' corrected requirement only if the two are additive, which TREES-x-CLOSECALL (additivity 0.54-0.62 [K TRIED
1072]) warns against: the joint screen is the first thing to run.

**Draft cost.** The extension is 32-64 more head levels per sure block: at today's head (0.65 ms per level) that is
+21-42 ms and the lever is dead; at the QAT + fused head (~0.1 ms) it is +3-6 ms on 27 % of blocks, +1-2 ms average.
This is a fast-head lever, like everything else in PUSH800 section 2.

**Quality risk.** None: every accepted token is the 125B's argmax; the rule changes how many rows a check holds.

**What kills it.** (a) a row past 32 costing > 0.45 ms (the dense pass not compute-bound at 32 rows, or the GDN /
cores scaling worse than linearly: Leap's call, a 64-row vs 32-row check timing on the exact engine answers it in one
run); (b) MBPP's sure-spine share under ~15 %; (c) the engine's one-pass check not accepting a variable row count (a
64-row-capable check with 32-row calls, padded or not).

**Screen (done, <= 10 min CPU):** `adaptive_rows.py <dump>`; the joint one with close-call delta 1 needs the lossy
kept rule on the dump's rows (NewIdeas' tree_shape_cc harness) and is the next 10 minutes.

## 2. WIDTH TO THE UNSURE NODE, from the same screen  [desk bound; close to keyed widening, so only a bound]
In the miss-bound half of blocks (spine < 24: kept32 ~7-10), 72 % of misses have the truth within the head's top-3 at
p1 ~0.68. Best-first already spends the spare pops by path_p; the bound says what a perfect second-child placement
would give: at most +1 kept per miss repaired (the next miss comes ~10 rows later on these blocks [K cells]), so a
ceiling of ~x1.1 on those blocks, ~x1.05 overall. Not the lever; recorded so nobody re-screens it.

## 3. TYPICAL (entropy) ACCEPTANCE in place of margin close-call delta 2  [published (Medusa typical acceptance); lossy]
Close-call delta 2 died on quality on the 2B (-6 pt, answers at the cap 26 -> 50 [K TRIED 1095]) while delta 1 holds
(+21 % kept). Delta 2 accepts a draft 2 nats below a PEAKED top-1 (p 0.8 vs 0.1): that is where quality goes. Typical
acceptance accepts only where the 125B's distribution is flat: p(draft) >= max(eps, delta * exp(-H)), H the 125B's
entropy at that row. The near-tie rows after a miss (median margin 3.9-4.4 nats, 11-12 % under 0.5 [K postmiss_d0])
are the flat ones. **Number:** between delta 1's +21 % and delta 2's +69 % kept per check; the quality proxy is the 2B
pass@1 harness that killed delta 2. **Screen (<= 10 min):** on `kdump_heplus24.jsonl` (326 misses with the 125B's
top-8 log-probs at the miss row and two branch rows): count the misses whose draft passes the typical rule at
(eps 0.1, delta 0.3) vs margin delta 1 / delta 2, and the share of those at a 125B entropy under 1 nat (the dangerous
ones). **Kills it:** the typical set on the 2B harness loses > 1 pt pass@1, or it accepts under half of what delta 2
accepts (then it is delta 1 again). **Screen status:** `typical_cc.py` ran but my reading of kdump's fields is wrong
(the draft token is in the recorded top-8 in 16 of 326 records: the top-8 rows are the branch rows, not the miss row),
so the count is NOT done; it needs BigMoE's field spec (kdump.log) first.

## 4. Not proposed, with the number
- Tokenization-equivalent acceptance: 3 of 2,773 misses [K section 0]. KILL.
- Substitution / edit-tolerant acceptance: 11.6 % of misses re-sync but half are identifier swaps [K]. KILL on quality.
- Draft reuse after a miss (seed the next chain with the old suffix where it re-synced): saves ~30 % of draft levels
  on 12 % of blocks (~1 ms average at today's head, ~0.1 ms at the fast head): too small for this sprint.
- Fixed 48 / 64-row checks: x1.00 on walk starts [K], the record's KILL stands.
- Anything feeding the 125B's branch states to the head (SIB, KILL 15:2x).

## Summary line
Best: ADAPTIVE BLOCK LENGTH (exact): extend the chain to 64-96 rows only when the head's 32-pop spine reached depth
31 (27 % of HE+ blocks, which keep ~29 of 32); kept x1.34-1.50 per check at ~0.28 ms per extra row = x1.21-1.23
tok/s on the real HE+ dump (fixed-64 gives x1.00), no quality risk, needs the fast head for the extra levels; the
joint screen with close-call delta 1 is next; on MBPP the lever is nil (no sure spines), so MBPP's x1.3 must come
from the head or a lossy rule. Then typical (entropy) acceptance as the lossy lever that delta 2 was
not. Tokenization splits and edit-tolerant acceptance are KILLed by the same screen.
