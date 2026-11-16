# IDEAS_1116_draft: WALL C, drafting at <= 3-5 ms per block (Fable FAR, Mon 16 Nov 2026 12:0x-12:4x, desk only)

Problem 4 of THINK_1116.md. Marks: [K] measured (source), [P] arithmetic on [K], [G] judgement. All ideas are drafter
changes: the check stays the judge, so they are exact unless said otherwise.

## 0. The arithmetic that decides the wall (from the records)

Per draft level (one tree depth, all its nodes in one pass) the time is the larger of two floors [K WEEK98 Mon 08:4x]:
- **bytes:** the head reads 177.6 MB per level at ~293 GB/s = 0.61 ms; of that the trained Linears are ~63 MB bf16 (the
  8-bit / 4-bit table: 146.9 / 130.5 MB) and the rest, ~115 MB, is the vocabulary side, which the 8K screen takes to
  10.5 MB [K line 1771, 1876];
- **dispatches:** ~144 dependent steps at ~2.1 us = 0.31 ms per level on stock MLX, whatever the bytes; a hand-fused
  level (<= ~35 steps) would be ~0.08 ms [K/P Leap, NewIdeas].
Levels per block: tree-16 12.6, tree-32 20.8, K128 39.6 [K]. Measured async: full head HE+ tree-32 13.4 ms, the
2-bit + top-2 cut (39.8 MB) 9.3 ms = the dispatch floor [K M0-M2].

| head per level | bytes | ms per level, stock | ms per level, fused | tree-32 block | K128 block |
|---|---|---|---|---|---|
| full t30b | 177.6 MB | 0.645 [K] | ~0.61 (bytes) | 13.4 [K] / 12.7 | 24 |
| 8K screen (H8) | ~73 MB | 0.31 (floor) | ~0.25 | 6.4 | 9.9 |
| H8 + 4-bit Linears, TRAINED | ~26 MB | 0.31 (floor) | ~0.09-0.1 | 2.1 | 4.0 |
| the same with the tail's levels paired (idea 2) | ~26 MB | | ~0.1 | ~1.6 | ~2.6-3.0 |

So the wall falls only with BOTH: bytes per level under ~30 MB and a fused level (or fewer levels). Either alone stops at
~6-10 ms per block. In the 900 budget (check 25 ms): kept 23.7 at K128 gives 23.7 / (25 + 24) = 484 tok/s with today's
head, 23.7 / (25 + 4) = 817 with idea 1 + the fused level, ~850 with idea 2 on top. That is the whole of wall C.

**The M3 line was the wrong line for this budget.** M3 killed every byte cut on "kept within 1 %" [K TRIED 1488]: Q4
x0.922, C1 x0.875, H8 x0.963 / x0.985. In tok/s terms a cut trades kept against draft ms: at a 25 ms check, tree-32,
H8 alone is 16.26 x 0.963 / (25 + 6.4) = 499 against 16.26 / (25 + 13.4) = 423 (+18 %) even on stock MLX. And M3 cut
the weights POST HOC; "weight cuts drift with depth (Q4 top-3 set 55 -> 23 %)" is the signature of untrained
quantisation, which training-time quantisation removes in every published head of this size [G]. Kamen's rule
(WEEK98, 15 Nov): a negative on their setup is a hypothesis, not a verdict.

## 1. QAT-DISTILLED t30b: the same head at 4-bit Linears + the 8K screen, trained to agree with itself  [new here; published method (QAT / self-distillation)]

**Mechanism.** Keep t30b's architecture and its features. Fine-tune it with fake 4-bit quantisation on every Linear
(straight-through) and the 8K vocabulary screen in the loss, with the TEACHER = the stored bf16 t30b: the loss is the
KL to t30b's own per-level distribution on the true-state cells plus the chain's own-state rollouts (the drift
regime), a few thousand steps from t30b's weights (continued Adam, the optimiser state restored: the MLX Adam
restart trap). Target: top-1 agreement with bf16 t30b >= 99 % at every depth, so the trees are the same trees.
Bytes per level 177.6 -> ~26 MB.

**Numbers.** H8's screen already keeps top-1 >= 99.3 % at every depth post hoc [K M3]; only the Linears need
training. Post-hoc 4-bit lost 7.8 % kept at tree-32 [K]; QAT on a 2-3-layer head recovers to within ~1 % in the
published record (LLM.int4 / QAT on draft heads) [G, to verify on our data]. Gain: tree-32 13.4 -> 6.4 ms today
(dispatch floor), -> ~2.1 ms with the fused level; K128 24 -> 4 ms.

**Exact:** yes. **Trainable by Thursday:** yes: one recipe change on the existing t30b pipeline, data already captured;
a 15-minute screen first.

**What would kill it.** The 15-minute screen: fake-quant t30b's Linears and run 300 steps of self-distillation on the
true-state cells; if top-1 agreement with bf16 t30b at depths 15-30 stays under 97 % (post hoc it is 23 % on the
top-3 set), 4-bit is too tight: fall back to 8-bit Linears (146.9 MB post hoc; ~42 MB with the screen) or 6-bit, which
still halves the bytes. Also killed if the engine cannot run 4-bit Linears at the width of a level without its own
dispatch cost (Leap's call).

**Test today (<= 15 min, MTP Head's harness):** `qat_screen`: load t30b, wrap Linears with fake-quant 4-bit, 300
steps at a small lr on 2,000 true-state cells, then the M3 harness at tree-32 on 24 HE+ windows: PASS = kept
>= 0.99 x bf16 t30b (M3's own line), bytes 26 MB per level.

## 2. LEVEL PAIRING by self-distillation: one level drafts two tokens where the head is sure  [new; Medusa-like, trained against the head itself]

**Mechanism.** Add a second output to t30b's last features: the token two steps ahead. Train it to reproduce what
t30b's NEXT level would output given its own top-1 (self-distillation: targets come from running t30b twice, no new
data, no 125B). At decode, on a node whose top-1 has p >= 0.95 (right 99.5-100 % of the time [K THINK_1115]), the
second output extends the path one more token without a level: the tree's thin tail (depths 10-30, 1-3 nodes per
level, mostly sure rows in long runs [K S11]) runs in half the levels. Where the second output disagrees with the
head's own next level (measured on the dumps), nothing changes: the engine falls back to a level.

**Numbers.** HE+ keeps ~3.3 nodes per depth down to depth 30; K128's median max depth is 29 [K S11]; the deep bands
are the sure ones (true-state 99.2-99.3 % at 15-30 [K]). If 60 % of levels past depth 8 pair, K128's 39.6 levels
become ~27: with idea 1 and the fused level, K128 drafting 4.0 -> 2.7 ms; tree-32 2.1 -> 1.6.

**Exact:** yes (a drafter-internal shortcut; a wrong pairing is a normal miss). **Trainable by Thursday:** yes, a
new output projection + small MLP on frozen t30b features, hours of training.

**What would kill it.** The 2-ahead output agrees with t30b's own next level on under 97 % of sure nodes (it lacks the
intermediate state; DFlash2's block form was far worse than the chain, tau 3.5-4.2 [K report]). The screen answers
this before any engine work.

**Test tomorrow (<= 15 min):** on the stored 127-node dumps, train the 2-ahead output on half the windows (frozen
features, 1,000 steps), score agreement with the head's next-level top-1 on the other half, by p1 band and depth.
PASS = >= 97 % agreement at p1 >= 0.95 and depth >= 8; report the share of pairable levels.

## 3. The fused level kernel is not optional (Leap; the enabling line)  [tried: priced, NO GO this week by NewIdeas]

Stock MLX's 0.31 ms per level [K] alone is 6.4 ms per tree-32 block and 12.3 per K128, above the 3-5 ms line with zero
bytes. Idea 1 without it stops at the floor (tree-32 6.4 ms); idea 1 with it reaches 2.1 ms. NewIdeas priced ~2-3 ms per
tree-16 block with <= 35 dependent steps and set it for next week; under the 900 target it moves up: it is the half
of wall C that training cannot replace. No test of mine; a Leap build (days).

## 4. TAIL LEVELS BY LOOKUP: skip the head where a long copy run is running  [tried as a coverage lever: LOOKUP-x-TREE32 KILL, L1-10 PASS; new as a bytes lever]

**Mechanism.** The K128 tail is long runs, HE+'s docstring copies [K THINK_1116]. Inside a prompt longest match of
>= 12 tokens, draft the next token from the lookup at 0 bytes and run the head only when the match breaks or the head's
last p1 < 0.95. LOOKUP-x-TREE32 killed the lookup as extra COVERAGE (it adds nothing the tree lacks); here it replaces
levels the head would spend bytes on. **Exact:** yes. **Gain:** depends on the share of tail levels inside long matches;
if a third of K128's levels, ~-30 % draft ms on HE+ (less on MBPP, which copies less).

**Kill:** on the dumps, the lookup's next token disagrees with the head's top-1 on > 3 % of those positions, or long
matches cover < 15 % of levels. **Test (<= 15 min, CPU):** on the 127-node dumps + prompts, for each level past depth
8 mark whether the path's last 12 tokens occur in the prompt and whether the lookup's continuation equals the head's
top-1; report coverage and agreement by set.

## 5. A SMALLER HEAD by distillation (the structural route)  [new; a 2-3 day project, Thursday is tight]

Split the head into a once-per-block part (reads the anchor state, 177 MB once) and a tiny recurrent cell per level
(<= 20 MB), distilled from t30b's per-level outputs on the true-state cells and own-state rollouts. It is the only
route to ~0.07 ms per level without a fused kernel. Numbers: bytes per level 177.6 -> ~20 MB; the published record
(EAGLE-3's one-layer heads, DFlash's block form) says a one-layer cell reaches ~90-93 % per guess, not 97.8 [K
report], so the cell must be a few layers and the drift regime must be in its training data. Kill: the cell's
own-state accuracy at 23-30 falls under 95 % after a day of training. Test: not in 15 minutes; a one-evening
distillation at 2B-sandbox scale first (Sandbox98), then the 125B.

## Not proposed
- Hiding the drafter behind the check: bytes add in full on a bandwidth-bound check [K]; speculative next-block drafting
  is a paper KILL (27 % hit).
- Post-hoc cuts as they are (M3): the line was kept alone; see section 0 for why the budget line differs.

## Summary line
Best: QAT-distill t30b to 4-bit Linears + the 8K screen (26 MB per level, trees unchanged if agreement >= 99 %),
with the fused level as the other half: tree-32 13.4 -> 2.1 ms, K128 24 -> 4 ms; then level pairing where the head
is sure (-35 % levels in the tail), lookup inside long copies, and the structural small head as the longer route.
