# IDEAS_1117c_walls: the four walls audited; three are physics for this shape, one is made of assumptions (Fable C, Tue 17 Nov 2026, 11:1x-12:2x EET)

Desk, on the real dumps (cells_t30b, topk_t30b, tree_t30b_*_n127k8, misses_t30b json), TRIED.md, BREAK98c. Two CPU
replays, 1.5 s each (`pool_cursor_1117c.py` -> `.out/.json`; the depth count is in section 3). Marks: [K] measured,
[P] arithmetic on measured parts, [G] judgement. No GPU, nothing committed.

## 0. In five lines
1. **Kept 17-19 at 32 rows is information, not an assumption** [K]: 72 % of the head's shallow misses on HE+ sit at 125B
   margins < 1 nat (94 of 131 at band 0), where the 125B's own tip is a coin; the tree pays a sibling per tie; the
   hindsight cap is 18.48; sampling-based and low-temperature acceptance were tried and killed (TRIED A1, 23 Oct: code
   within +-0.001 of greedy); the A_line's 2.25 % disagreement is already spent by CC(1) (ties the head takes ~1.75 %
   of rows). The assumption INSIDE this wall is the number 32, and 32 is set by the draft, not by the check.
2. **A row >= 0.15 ms is physics** (a token's dense MACs at the matrix peak; rows share no compute; partial rows closed),
   and it is not the wall: 32 rows are 4.8 ms of compute against a 25 ms byte floor and a 48-56 ms kernel reality.
3. **The check's byte floor is physics at 290 GB/s** (dense ~2.1 GB once per check + a ~3.5 GB expert union + KV); the
   assumption inside it is "once per CHECK": K64 amortises the dense (R 1.12 vs 0.85) and that lever is ADAPTIVE K.
4. **The draft is the wall made of assumptions** [K]: 1.1 ms per pop = 0.55 fixed (a kernel chain) + 0.56 bytes (the
   163-178 MB Q8 head), 31 pops serial = 34 ms per tree-32, 16.6 per tree-16; its physics is bytes x sequential depth.
   New fact: on HE+ the tree-32 is a 25-deep chain (24.9 distinct depths in 31 pops, median 31, 1.24 pops per level),
   so level batching (EAGLE-2's trick) buys ~nothing on HE+ and x1.9 on MBPP (15.8 levels). The way through is not
   batching: it is the head's bytes (QAT 26 MB: 0.56 -> 0.09 ms per level) and CONCURRENCY (the draft off the cycle).
5. **Idea: the POOL DRAFTER** (section 4): the head keeps a drafted pool ahead of the cursor; each check takes the 31
   best pool nodes under the cursor; the pool is refilled frontier-wide and, once the head is <= ~40 MB, on the idle
   CPU during the check, so the draft leaves the cycle except at information ends. Measured lower bound today [K]: an
   UNREFILLED, un-rerooted pool already serves 19-25 % of HE+ checks with no draft at x0.978-0.949 kept. **Decisive
   test run 12:4x (section 7): FAIL / KILL.** The fresh-draft share passes (0.136 HE+, 0.167 MBPP), but a block drafted
   without a re-root keeps ~x0.88, because the drifted head is right 86.5 % at the first position after a correction
   against 94.2 % re-rooted (MBPP 76.2 vs 91.1); overall kept x0.90 < the x0.95 bar. Conclusion for 550: section 5.

## 1. Wall by wall: what the measurement held fixed, and whether that is physics
**Kept.** The 16.26 is the expected on-true count of the best 31 nodes by path_p for a calibrated head; ORACLE-31 18.48
is the hindsight cap over ANY 31-node selection from the head's own candidates. Held fixed: greedy acceptance, the
quality gates, 31 rows, one tree per check. Greedy is not an assumption we forgot: TRIED A1 (23 Oct) measured lossless
speculative sampling and temperatures 0.05-1.0 and found code acceptance within +-0.001 of greedy agreement (the loss
at agreeing rows equals the gain at ties). The gates are not an assumption either: the A_line (0.9775) is a token
agreement line, so every tie the head takes is a flip, and CC(1) already takes ~1.75 % of rows (3.5 % of rows have
m < 1 nat, the head is a coin there). A flip budget spent on the 1-2 nat misses instead (18 % of shallow misses, 0.38 %
of rows) would add ~+1 kept per block [P]. So within greedy + gates, kept at 32 rows is a wall. Rows are the lever
(K64 20.77, K127 23.70), and rows cost DRAFT far more than check: at the floor a marginal row is 0.11-0.18 ms in the
check and 1.1 ms in the draft (a pop). The number 32 is a draft number.
**The row.** Fable B's 0.15 ms is the dense MACs of one token at the int8 matrix peak. Siblings differ from the first
embedding on, so no compute is shared; partial width and depth were closed on the 125B's states. Physics. It is also
small: 32 rows x 0.15 = 4.8 ms. The 48-56 ms check is kernels (hc_read 5.5 + router 1.7 + ple 0.8 + hc_write 0.7 = 8.7
ms of fixed per-layer dispatch, and dense classes at 21.8 ms against a 7.3 ms byte floor), which is Leap's C6 line.
**The byte floor.** 25 ms = ~7.3 GB at 290 GB/s: the dense ~2.1 GB, the expert union ~3.5 GB (~40 experts per layer for
32 rows, 8x reuse of the 320 routed slots), KV and activations. Nothing between DRAM and the GPU holds GBs, so "read
once per check" is physics; "per CHECK" is the assumption, and amortising the dense over more rows is the K curve.
**The draft.** 1.1 ms per pop, two halves. The 0.56 ms of bytes is the Q8 head's 163-178 MB per forward: a one-row
matrix-vector pass touches every weight. The 0.55 ms fixed is a kernel chain, not compute (~0.2 GFLOP per pop, ~0.01
ms at peak). "31 pops serial" is the third factor. Each is an assumption: bytes fall 6x with the QAT head (S22, after
Saturday), the fixed cost fuses, and pops batch by level. But the batching factor is small where it matters (section 3),
so the draft's floor per block is sequential depth x bytes per forward: HE+ 25 x 0.56 = 14 ms with the Q8 head, 25 x
0.09 = 2.3 ms with the QAT head. Today's 34 ms is 2.4x / 15x above those. The check is 2x above its floor. **The
draft is the wall with the headroom, and the 550 arithmetic (kept 16.5 at check 25 + draft 3 + overhead 2) is
exactly "the draft at its physics"**: no new information is needed for 550, only the two engineering floors.

## 2. The pool replay: what a block loses without a re-root [K]
`pool_cursor_1117c.py`, walk along each file (kept per CHECK; the per-anchor mean is length-biased: 21.0 vs 16.3).
(A) today, a fresh tree-32 every check: HE+ 16.30 (published 16.26), MBPP 9.05 (9.03).
(B) the pool engine on the dumps: after a block ends at its correction c', if the 127-pop pool of the last fresh anchor
has c' expanded with >= MIN nodes under it, the next check is the first 31 pool nodes under c' (pop order), with no
re-root at the true state and no refill (the dumps cannot refill: a LOWER bound on a real pool); else a fresh pool.

| HE+ | kept/check | rows/check | fresh drafts per check |
|---|---|---|---|
| today | 16.30 | 32.0 | 1.000 |
| pool, continue if >= 31 nodes under c' | 15.94 (x0.978) | 32.0 | 0.807 |
| pool, >= 16 | 15.47 (x0.949) | 31.4 | 0.750 |
| pool, >= 8 | 15.07 (x0.924) | 29.9 | 0.690 |
| MBPP today / pool >= 16 / >= 8 | 9.05 / 8.92 / 8.55 | 32 / 31.7 / 29.9 | 1.0 / 0.963 / 0.876 |

- A pool block (31 rows, drafted at drift depth 16-31, no re-root) keeps ~14.4 vs 16.3 fresh on the same stretches [P
  from the >= 31 row], i.e. the re-root is worth ~12 % of a block's kept. The own chain's hazard by depth is flat
  (0.021-0.034 at depths 0-30, survivorship-biased), so most of that 12 % is the head's second-choice quality at the
  miss, not accumulated drift.
- End types per check on HE+ [K, per anchor]: c' expanded in the pool 35 %, c' a candidate of its parent but unexpanded
  16 %, c' absent from the parent's 8 candidates 10 %, block capped at 31 rows 39 %. A refilled pool that expands the
  candidates at the head's low-margin nodes (its own margin locates misses at AUC 0.954) turns the 16 % into ready
  blocks; the 10 % absent are the information ends and always need a fresh root.

## 3. Level batching: measured on the dumps [K]
Distinct depths among the first P pops (= batched head forwards per block under level-wise drafting), HE+ / MBPP:
P 15: 13.6 / 10.9 (1.11 / 1.37 pops per level); P 31: 24.9 / 15.8 (1.24 / 1.96); P 63: 41.7 / 20.3; P 127: 59.1 / 25.0.
On HE+ the head is confident and right deep, so its best-first tree IS a chain: batching saves 20 % of the forwards at
tree-32 and nothing at tree-16. On MBPP it saves half. The draft's sequential depth per block on HE+ is ~25 and is
inherent in autoregressive drafting; only bytes per forward (QAT) and concurrency reduce the term.

## 4. The idea: the POOL DRAFTER (breaks assumption 4, and 3 in part)
- **Mechanism.** The head never drafts "a tree for the next check". It keeps a pool: a best-first expansion ahead of
  the cursor (127-512 nodes), refilled frontier-wide after every check (all frontier nodes at a depth in one forward,
  the head's weights read once per level), and re-rooted only at information ends (c' absent). Each check takes the
  31 best pool nodes under the cursor (ADAPTIVE K picks K per check from the same pool at no extra draft). With the
  QAT head (26 MB, 0.09 ms of bytes per level) the refill runs on the CPU (AMX/SME; 18 cores idle during a GPU-paced
  check; ~0.4 GB of DRAM per block = 1.4 ms of the check's bandwidth, 5 %) while the GPU checks; with the Q8 head it
  cannot (25 levels x 163 MB = 4 GB per block would take 14 ms of the check's bandwidth), so this is an after-S22 design.
- **Why today's walls don't apply.** The draft term leaves the cycle for every block whose c' is in the pool (ready
  today 35 % unrefilled, ~50 % with the candidates expanded, bounded by the 10 % absent); the sequential depth is paid
  on the CPU in the check's shadow; the GPU does only checks. "A second GPU stream" (killed) is not involved.
- **Number [P].** At the floor (check 25, overhead 2, QAT draft 3 ms serial when fresh): today's shape 16.3 / 30 =
  543; pool engine kept 0.85 x 16.3 + 0.15 x 16.3 x 0.88 (pool blocks) ~ 16.0, cycle 25 + 0.15 x 3 + 2 = 27.5 -> 580
  (+7 %). With this week's realistic check (48 ms) and the QAT head (draft ~8 ms fused-less): 325 -> 350 (+8 %). If
  the fixed 0.55 ms per level does not fuse, the hidden draft is worth more (today's 16.6-34 ms term): +12-25 % [G].
  Honest: the pool is a +5-12 % lever at the floor, because at the floor the QAT draft is already small; its value is
  that it removes the draft from the critical path for ANY head size once the head fits the CPU's bandwidth budget.
- **Decisive test (<= 10 min CPU, extends `pool_cursor_1117c.py`).** Refill emulation: use the n127 pool plus the fresh
  pools at later anchors as the refill (nodes under c' from the pool at anchor A plus the fresh pool at A' for the
  depths the stale pool lacks) and report kept per check and the fresh-draft share at MIN 16; PASS if the fresh share
  <= 0.25 at >= x0.95 kept (then the draft term is hidden for >= 75 % of blocks); KILL if the share stays >= 0.6.
  The engine test later: the CPU refill's DRAM share during a check (must stay <= 10 % of the check's bytes).
- **Quality risk.** None to the output (every row is still verified by the 125B at full depth; the pool only changes
  WHICH rows are drafted). Kept risk: pool blocks lose ~12 % vs a re-rooted draft (section 2); MBPP gains less (fresh
  share 0.96 at MIN 16).

> (a quote from the project owner, kept private)
550 on HE+ is not blocked by information: it is kept 16.5 (today's head, today's tree) at a 30 ms cycle, i.e. the
check at its byte floor (25 ms; Leap's C6 + int8, 2x from today's 48-56) and the draft at its physics (3 ms = 25 levels
x the QAT head's 0.09 ms + a fused fixed cost; 10x from today's 34). Both are engineering walls with known steps, not
walls of ideas; neither is Saturday material (the QAT head is after Saturday; C6 is in progress). The drafter-
information route (kept 24 at 32 rows) is what this morning closed, and it stays closed under greedy + the gates.
The one thinking lever left on the drafter is to take the draft OFF the cycle (section 4), worth +5-12 % at the floor
and more while the head is big. Everything above 550 is then kept x check: 775 (Kamen's stretch) needs kept ~21 at the
same cycle, which is K64 at 1.32x the check (dead: 20.77 / 1.32 = 15.7 per 32-row check) or ADAPTIVE K + the pool's
free rows (+8-15 %) on top of 550 -> ~600-630 at the floor [P]. 775 needs a new information source or a cheaper row;
this file finds neither, and says so.

## 6. Files
`pool_cursor_1117c.py` (+ `.out`, `.json`): the walk-based pool replay (modes A / B / C) and the hazard table; the depth
count (section 3) and the drifted-vs-re-rooted comparison (section 7) are one-offs in the session log, numbers here.
Records read: BREAK98c.md, TRIED.md A1 (lines 936-938), the dumps' json headers (topk, misses, margins), BREAK98.md,
IDEAS_1117b_closure.md, Lunatic / Fable B status lines.

## 7. Decisive test, run 12:4x on the orchestrator's ask: FAIL / KILL [K]
`pool_cursor_1117c.py` mode C, the refill emulation along the walk: a block is READY when its correction c' is expanded
or a candidate in the current pool (a CPU refill during the check expands candidates); a ready block's draft is
emulated by the re-rooted pool at the new anchor with its kept discounted by the no-re-root factor; only an absent c'
(outside its parent's 8 candidates) forces a serial draft. Bar: fresh-draft share <= 0.25 at >= x0.95 kept.
- Share: PASS. Per check along today's walk the correction is expanded 66.5 %, a candidate 20.2 %, absent 13.3 % on
  HE+ (MBPP 58.7 / 24.8 / 16.4), so the fresh share with a refill is 0.136 / 0.167.
- Kept: FAIL. Overall kept = x1.000 / x0.949 / x0.899 / x0.831 at pool-block factors 1.00 / 0.94 / 0.88 / 0.80; the
  bar needs the pool blocks at >= x0.943, and the measured factor is x0.88 (section 2, on the stale pool's full blocks,
  an easy-selected sample). Where the loss comes from, measured at the 259 HE+ / 1,012 MBPP blocks whose c' is expanded:
  at the first position after the correction the DRIFTED head (the pool's node c') is right 86.5 % and the RE-ROOTED
  head (the fresh pool's root at the same position) 94.2 % (MBPP 76.2 / 91.1); their top-1 children agree 89.2 % /
  77.1 %, their top-8 sets overlap 58 % / 61 %; mean p(true) 0.81 vs 0.91 (MBPP 0.67 vs 0.84). The head's second choice
  after its own miss is its weak row, and that row is where a pool block begins.
- The only form that could pass is a hybrid: one serial head forward from the true state at c' (1.1 ms today, ~0.2
  after QAT), then the pool's subtree under the re-rooted top child when it matches the pool's (89 %), else a serial
  draft; on paper fresh share ~0.23 and kept ~x0.96 [G: the depth-1+ quality of a re-rooted-then-drifted draft is not
  in the dumps]. Its gain at the floor is 1-2 ms of a 30 ms cycle against ~-4 % kept: a wash. The pool pays only while
  the draft term is 25-40 % of the cycle (this week's Q8 head), where the CPU cannot host it (4 GB of DRAM per block).
- Verdict: KILL for the floor and for Saturday; the record's lesson is the number 86.5 vs 94.2: a draft must be
  re-rooted at the 125B's true state at the correction, which is the serial dependency the frame already pays.
