# IDEAS_1117c: what a check recomputes or re-reads that it already had (Fable B, Tue 17 Nov 11:1x-12:2x)

Lens (BREAK98c): assumptions 1 (every row through all 48 layers, one block at a time) and 2 (each check re-reads its
experts from DRAM; nothing carries over between checks). Desk only; one CPU count on the 127-node dumps (think1117c).
Labels [K] measured / [P] derived / [G] estimate.

## 0. The inventory: what exists after check n, and whether check n+1 can use it

| after check n, the engine holds | reusable by check n+1? | why / why not |
|---|---|---|
| KV + GDN states of the accepted prefix (chain to the end node A) | YES, reused today | the prefix is final |
| the 125B's logits at A: its top-1 t* and the top-8 with margins | YES, reused today (the bonus token t* is certain; depth 0 of the next block reads A's true state) | |
| the 125B's logits and states at the OTHER chain nodes | no new use | they were consumed by the walk; the chain is final |
| the states and logits of every UNREACHED row (depth > A, or off the chain) | **NO: all invalid** | each has a wrong token in its prefix: an off-chain node differs from the 125B's choice at its depth; a node below A hangs under a child of A that is not t* (else the walk would have continued). Count below. |
| the routing (expert ids) of every row | no | ids are per (row, layer); the next rows are new positions |
| the expert and dense WEIGHT bytes read | **NO: nothing on the chip holds them** | unified memory; the only on-chip stores are the SLC (tens of MB) and threadgroup memory (per dispatch). DRAM is the only place a 5 GB working set can live; "re-read from DRAM" is the floor of any pass, not a choice |
| the 45 GB resident pool (SSD -> DRAM) | YES, reused today | the loader's residency IS the carry-over of assumption 2, and the only one possible |
| the drafter's own levels | no | block n+1 starts at A+1 with t*; the head's states for block n's tokens are states of a rejected sequence past A |

**Count on the real 127-node trees [K, think1117c]:** rows the walk never reaches: HE+ 70.7 % / MBPP
91.0 % of a tree's rows; the walk ends at a leaf of the tree in 31.8 % / 32.5 % of blocks (the tree ran out; nothing
below it could continue either); the 125B's next token at the end node sits among the end
node's TREE children in 0.0 % / 0.0 % (if it did, the walk would have gone on: these are near-zero by
construction) and among the head's top-8 at the end node in 80.8 % / 72.9 %.

## 1. Assumptions 1 and 2 are not assumptions; they are theorems of (greedy walk) x (unified memory)

- **Assumption 2.** Carry-over of BYTES between checks needs a store faster than DRAM that holds > ~5 GB. There is
  none on this chip. Carry-over of RESULTS needs rows whose prefix is still valid: only the chain's (used). Carry-over
  of WORK on identical inputs (an expert output cache) is session value prediction: closed. So the two things that can
  carry over, carry over today; nothing else exists to carry.
- **Assumption 1.** "Every row through all 48 layers": a row's logits are the 125B's only through all 48 layers
  (partial depth closed), and a row's state is needed by its descendants at all 48 layers (the KV and GDN caches), so
  even a row whose logits are never read (a leaf) runs all layers if it has... no descendants: **leaves do not need
  their lm_head** (IDEAS_1117b: 0.6-1.3 ms per 128-row check), and that is the whole slack in assumption 1 at the row
  level. "One block at a time": the walk is a function of all rows' final logits, so the next block's first position
  is unknown until the pass ends; a draft started earlier is a draft from a guessed anchor (future-state rows: KILL;
  the P1 OVERLAP record: one GPU runs two streams serially, kappa 0.93).
- **What the check really "recomputes":** nothing twice. What it WASTES is the bets: 70.7 % of its rows on
  HE+ never meet the walk. That waste is the drafter's accuracy, not the engine's structure, and it is already priced
  by the K-curve (kept 16.26 / 18.86 / 20.77 / 23.70 at K32 / 48 / 64 / 128).

## 2. The one place the lens still finds slack: the walk's END NODE is where block n+1 is decided, and check n has
almost everything about it except the row for t*

- At the end node A, check n knows t* (the 125B's top-1) and A's true state. Block n+1's first drafted token sits at
  A+2 and is the HARD first guess (own accuracy after an unsure step ~75 % vs 96 %; WEEK98: "only a 125B pass on the
  bonus token carries it"). The row for t* (position A+1) is the one 125B row check n did not compute, because t* was
  not among A's expanded children. **If it had been, block n+1 would start one position later from a TRUE state with
  the 125B's own distribution at the hard spot: the hard first guess becomes a certain token.**
- How often t* is within reach: among the head's top-8 at A in 80.8 % of blocks (HE+) [K], so a tree that expands
  ALL top-8 children of every chain node holds the t* row in that share of blocks. That tree is 31 x 8 = 248 rows
  (the LAB "true state + tie siblings" line: 26.79 kept vs chain 20.77), i.e. route 2 at ~7x the rows, closed at the
  per-row cost (IDEAS_1117b). **The cheaper form: expand the 8 children only where the head's margin flags the
  chain (AUC 0.954 for first misses; the brief: the head knows WHERE).** Flag the ~4 lowest-margin chain nodes and
  give each its full top-8: +28 rows (~60 in all), and the t* row is in the tree whenever the first miss is at a
  flagged node and t* is in the top-8: recall 97.6 % at 20 % false flags x 80.8 % [K] = ~0.8 of blocks [P].
- **Number [P].** Where the t* row is present, block n+1 starts from A+1 with the 125B's distribution: it gains the
  hard first guess (own ~75 % -> certain) and one depth of true state. Per block that is +1 kept (the certain token)
  plus the post-miss recovery (the drift records: own accuracy after a p1 < 0.5 step 75-77 % vs 96 % none, at every
  depth band), i.e. the second-position accuracy rises ~20 pt in ~a third of blocks: ~+0.3 kept. **~+1.3 kept per
  block [P] for ~+28 rows (~+4-5 ms at today's 0.15-0.28 ms per row).** At 1.77 ms per kept (tree-32's average) that
  is 2.3 ms worth of kept for 4-5 ms of rows: BELOW the bar as a row lever (Lunatic's 1.8 ms per kept), and it is
  what EAGLE-2's best-first already approximates (K127 pops those children when path_p says so). So: not a break,
  a re-derivation of why trees at the flagged nodes are the right shape. Recorded so nobody re-proposes "the t* row".
- **Decisive test (CPU, 10 min, no GPU):** on the n127 dumps, re-simulate the walk with the 8 children of the 4
  lowest-p chain nodes forced into the tree at K60 and count the blocks whose next block starts from a t* row; compare
  kept per row against EAGLE-2's best-first at K60. If best-first at K60 already contains the t* row in >= 80 % of
  the blocks, the idea is empty. Quality risk: none (exact rows).

## 3. Bottom line

1. Assumptions 1 and 2 are consequences, not choices: no on-chip store holds a pass's bytes, and after a greedy walk
   every unreached row is invalid (70.7 % of a 127-row tree on HE+ [K]). The carry-overs that exist (prefix
   caches, the bonus token, the 45 GB residency) are all in use.
2. The only slack at the row level is the lm_head at leaves (IDEAS_1117b); at the block level the only missing row
   is the t* row at the walk's end, which a tree at the head's flagged nodes already approximates: ~+1.3 kept for
   ~+28 rows [P], below the row bar.
3. For the frame to break, it has to break elsewhere: positions per weight read are set by the drafter's accuracy
   (assumptions 3-6), or the per-row price by the compute unit (7-8). My lens closes 1 and 2 with numbers.
