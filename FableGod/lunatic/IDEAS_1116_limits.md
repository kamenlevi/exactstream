# IDEAS_1116_limits: is 900 tok/s physical on this Mac, and which wall is the assumption? (Lunatic, problem 5)

Mon 16 Nov 12:0x-12:4x. Brief: ~/FableGod/deepdraft/THINK_1116.md, problem 5. Desk only: one tiny CPU script,
`lim900.py` -> `lim900.out`.
Marks: [K] measured (source named), [P] arithmetic on [K], [G] judgement.
Inputs [K THINK_1116 / WEEK98]:
- lossy 32-row code check 6.12 GB, byte floor 21.9 ms;
- t30b reads 177.6 MB per draft level, plus ~0.31 ms fixed per level;
- tree-16 / tree-32 / K128 = 12.6 / 20.8 / 39.6 levels;
- DFlash2 drafts a block for 0.52 GB.

## The answer in four lines
1. **900 on this Mac is a byte budget:** at most 303 / 311 / 333 MB of DRAM traffic per kept token at 273 / 280 / 300
   GB/s [P]. The lossy check alone takes 264 MB per kept token at kept 23.15 (flat 97.8 %): 85 % of the budget.
2. **The drafting wall is the physical one.** An autoregressive (AR) drafter pays at least one weight read per kept
   token beyond the first. So 900 needs the head at **<= 47 MB per level** at kept 23.15, or <= 92 MB at kept 28.
   - t30b reads 177.6 MB.
   - With it, 900 is out of reach at ANY tree shape and acceptance: even perfect acceptance (kept 33) gives 942 at the
     byte floor, and flat 97.8 % gives 660 [P].
3. **The acceptance wall is the assumption.** Per-guess accuracy is not the target; kept per byte is. The spec's three
   numbers don't compose to 900:
   - 97.8 % + a 25 ms check + a 3-5 ms draft = 772-827 tok/s [P];
   - 900 at a 25 ms check needs flat 98.4-98.9 %;
   - or the check at its 21.9 ms floor with <= 3.8 ms of drafting at 97.8 %.
4. **The check wall is real at 21.9 ms (bytes)**, but by Saturday it is a dispatch wall: the CPU side alone is 27.9 ms
   (4,022 dispatches) [K].

## Idea 1: the AR-drafter law (the wall that is physics)
**Mechanism:**
- A kept path of depth D needs D head levels. In a tree, the levels are the tree's depth, and each level reads the
  head's weights once for the whole frontier.
- So draft bytes per kept token >= S x (kept - 1) / kept, which is ~S, the head's bytes per level.
- The check is bandwidth-bound, so draft bytes add in full [K WEEK98 10:3x: "~2.2 GB per block adds in full"].
- Budget: 6.12 GB / kept + S <= 311 MB.

**Numbers** [P, lim900.out, byte floor, 280 GB/s]:

| scenario | kept | draft GB per block | tok/s | MB per kept token |
|---|---|---|---|---|
| today: lossy K* tree-32, t30b | 16.91 | 3.69 | 482 | 580 |
| flat 97.8 %, tree-32, t30b | 23.15 | 3.69 | 660 | 424 |
| flat 97.8 %, chain-30, t30b | 23.15 | 5.33 | 566 | 495 |
| perfect acceptance, tree-32, t30b | 33 | 3.69 | 942 | 297 |
| flat 97.8 %, only 6 head levels | 23.15 | 1.07 | 902 | 310 |
| flat 97.8 %, free drafting | 23.15 | 0 | 1,059 | 264 |
| kept 28, 6 levels + one 0.52 GB block read | 28 | 1.59 | 1,017 | 275 |

**What it means:** the head must get ~3.8x smaller per level at equal accuracy. Alternatively the deep run must stop
costing head levels (Idea 3), or kept must reach ~28 (lossy rules, Idea 2) so the line moves to 92 MB per level.
- M3's post-hoc cuts can't do it. Every byte cut fails the 1 % kept line: Q4 x0.922, C1 x0.875, C3 x0.767, C8 x0.186
  on HE+ tree-32 [K WEEK98 M3]. The smallest cut quoted, 2-bit + top-2 experts at 39.8 MB, is below the line in bytes
  but not in kept.
- This matches NewIdeas' "Pillar 3's real route is a smaller drafter", and puts a number on it: **<= 47 MB per level
  at 97.8 %**.

**Tag:** new as a bound [G: the pieces, memory-bound decoding and bytes per token, are textbook].

**Lossy / exact:** both.

**Kill:** a measured cycle where 1-2 GB of draft bytes hide under the check, contradicting the bandwidth-bound check.

**Test:** none needed. Certify the inputs on the record: the per-level bytes as stored, and the check's bytes.

**Gain in the budget's terms:** it sets the drafter line, <= 47 MB per level or <= ~1.1 GB per block.

## Idea 2: the acceptance wall is an assumption, so set the target as kept per byte
**Mechanism:**
- tok/s = kept / (check + draft). Every lossy rule that raises kept at the same bytes relaxes BOTH other walls.
- At kept 28, the 900 cycle is 31.1 ms. That leaves the floor check (21.9 ms) 9.2 ms for drafting: ~2.6 GB, a head of
  <= 92 MB per level.

**Numbers:**
- Spec triplet = 772-827 tok/s [P]. Needed flat accuracy at a 25 ms check: 98.40 % (3 ms draft) to 98.88 % (5 ms) [P].
- Lossy kept levers on record [K TRIED via THINK_1116]: INERT comments + docstrings x1.17-1.30 HE+ (replay, quality
  untested), SPANRELAX-C x1.074, COMMENT-FREE x1.23-1.29 [P].
- In-comment rows hold 71 % of HE+'s near-ties [K].
- Risk: on the 2B, a relaxed comment changes the next code in 48 % of cases [K].

**Tag:** tried / in pipeline (INERT, SPANRELAX-C, close calls). New here: the target re-statement and the composition
arithmetic.

**Lossy.**

**Kill:** EvalSuite's gate fails the comment rules at the kept they buy (>= x1.15).

**Test (15 min, CPU):** on the walk's dumps, kept/32 under close-call delta 2 + INERT comments, HE+ and MBPP. Report the
kept AND the head levels it costs, against the drafter line.

**Gain:** kept 23 -> 27-30 if quality holds. That moves the drafter line from 47 to 80-100 MB per level.

## Idea 3: take the deep run off the AR head (the game-changer candidate)
**Mechanism:**
- Spend AR head levels only where the tree branches: bands 0-6, the after-miss near-tie region where trees pay off
  (IDEAS_1115_limits Idea 1). That is ~6 levels, 1.07 GB.
- Draft the deep run (bands 7-30) with sources that don't read the head per token:
  - copy / lookup of a prior >= 8-token span: zero GPU bytes, CPU only;
  - or one block-drafter read (DFlash2-class 0.52 GB, all positions at once);
  - and stop the block where neither source is confident.
- Known tokens extend the head's context in one parallel level, not one level per token.

**Numbers:**
- 6 levels at flat 97.8 % = 902 tok/s at the floor [P]. With a 0.52 GB block read, 841 (needs kept 25.2 for 900).
- HE+'s long runs at 64-128 nodes are docstring copies [K THINK_1116].
- 58 % of HE+ drafter positions sit inside >= 8-token seen spans, where the lookup is right 99.2 % (drafter 96.1 %)
  [K FINDINGS line 89: drafter_memory_probe].
- But copy fixes almost none of the deep MISSES: 1.4 % (HE+) / 4.2 % (MBPP) are bigram-predictable [K LEVERS_A F]. It is
  a byte saver, not an accuracy fixer, and much weaker on MBPP.

**Tag:** adapted (prompt-lookup / REST / SAM drafting; hybrid drafters). New here: allocating AR levels by band under a
per-block byte budget.

**Lossy / exact:** both (drafter-side).

**Kill:** AR bands 0-6 + copy-or-block deep keeps < 0.85x of tree-32's kept on HE+, or < 0.75x on MBPP.

**Test (15 min, CPU, today):**
- On BigMoE's tree_t30b dumps plus the window texts, draft bands 0-6 with t30b's tree. Beyond that, take the longest
  prior match >= 8 tokens, else stop.
- Report kept/32 and head levels per block, HE+ and MBPP.
- PASS: >= 0.90x tree-32's kept at <= 7 levels per block on HE+.

**Gain:** draft 3.69 -> ~1.1 GB per block, which is -9 to -10 ms of cycle at the floor.

## Idea 4: what an honest 900 demonstration on code can be by Saturday
**Facts:**
- At the ladder's ~41-45 ms check plus 8.3-13.8 ms of drafting, 900 needs 44-53 kept per check: more than 32 rows
  hold [P].
- So a MEASURED single-stream 900 by Saturday is not supported by the records. The check needs megakernels or
  replayed command buffers (3-6 weeks, Leap) to reach <= 25 ms. The drafter needs <= 47 MB per level (a training
  project).

**What can be honest on Saturday, three labelled numbers:**
1. **MEASURED:** the best single-stream whole-decode tok/s (n / decode wall) on HE+ AND MBPP, lossy with EvalSuite
   same-scores. Expect roughly 300-420 on HE+ [G], and MBPP lower (its kept is ~60 % of HE+'s at 32-128 rows [K]).
2. **PROJECTED 900:** kept from the certified walk (LAB / C-lines), the check at its certified byte floor (21.9 ms),
   and drafting at a stated head size. Show the bytes-per-kept ledger (check GB + draft GB + misses) against the
   311 MB line. Every component is certified; the whole is labelled projected.
3. **AGGREGATE (optional, Kamen's call):** one user's B code streams in one pass, the head's levels batched across
   streams.
   - [P at the floor, today's head and acceptance]: B2 715, **B4 1,104 tok/s aggregate (276 per stream)**.
   - Code streams share experts like one stream [K WALL_L_1105 T2: groups per kept x0.55 at B4].
   - It needs the engine to batch streams (not built).

**Tag:** n/a (a reporting design). **Kill:** none; it is the honesty rule. Disclose the copy-span share on HE+.

## Idea 5: by Saturday the 25 ms check is a dispatch wall, not a byte wall
**Facts:**
- Byte floor 21.9 ms (6.12 GB) [K].
- CPU build + encode + commit 27.9 ms as a bound: 4,022 dispatches per one-pass 32-row check, 52 % of them MLX glue
  [K Fit12 LOSSY32].
- Dispatch diet: ~12 ms CPU in 1.5-2.5 weeks. <= 5 ms needs replayed command buffers or megakernels (3-6 weeks) [K Leap].

**The assumption to test:** "the CPU side adds to the GPU side". If the C5 check's ~61 ms is the sum of ~28-30 ms CPU
and ~30 ms GPU, then streaming the encode under the GPU's execution (commit each layer's buffer as it is built) gives
max(CPU, GPU) ~30 ms without removing a single dispatch.

**Test (15 min, today or tomorrow):** a Metal-trace-free timing split of one C5 check (libmtlprof GPU start / end
against the CPU encode window).
- PASS: GPU idle > 10 ms per check (the overlap is real money).
- KILL: GPU busy >= 85 % of the check.

**Lossy / exact:** both. **Tag:** engineering. Owners: Fable B / Leap.

**Gain:** up to ~-25 ms of check if the two halves are serial today [G until the split is measured].
