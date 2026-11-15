# IDEAS_1115_info: cheap information for the drafter (Fable FAR, Sun 15 Nov 2026 22:1x-22:4x, desk only)

Problem 3 of THINK_1115.md: the head is 97.8 % flat on the 125B's true state and drifts on its own carried state
(89.7 % at 23-30, the 8-point gap is drift). How does 125B future-state information reach it within 2-4 ms per
32-token block, with the check exact? Marks: [K] measured (source), [P] arithmetic on [K], [G] judgement.

The bytes rule first [K WEEK98 spec]: any extra read adds in full to a bandwidth-bound check, and the dense trunk alone
is 3.44 GB per pass (12 ms), so NO pass of the 125B over the draft can fit (self-speculation and early exit were KILLed
on exactly this, TRIED 920 / 970 / 1014). What fits is information the check has ALREADY computed and still holds:
exact 125B states and distributions at every verified row, kept or rejected, and the prefill's states. Ideas 1-3 use
only those; they cost ~0 bytes and a gather.

## 1. SIBLING-STATE DRAFTING (best): feed the head the 125B's exact states from the just-rejected rows  [new]

**Mechanism.** A 32-row check of block n computes the 125B's exact state at every row, including the rows after the
first miss m, which are exact states of the 125B conditioned on the drafted continuation t' (the sibling branch). The
next block is drafted from the bonus token t at m with the head's own carried state at depths 1, 2, ...; today that
state is the whole source of drift. Instead, for depth d <= 32 - m the head reads the 125B's exact state at rejected
row m + d (a gather from the check's activations, 0 bytes from disk), optionally concatenated with its own carried
> (a quote from the project owner, kept private)
such a row, so the pool of exact states deepens with K.

**Why it should carry information.** The rejected rows are not random continuations: 92 % of check-start misses have
a 125B margin under 2 nats [K THINK], so t' is a near-tie sibling of t, and the states one or more tokens downstream
of a near-tie are states of an almost identical prefix. For the 125B itself the margins right after a miss are thin
(median 3.87 nats vs 11.87 after a hit [K WEEK98 03:2x]), which is the band 1-14 problem; a true state of a sibling
prefix carries the 125B's view of THAT region, which the head's carried state does not. Rows available per check:
chain kept 13.77 of 32 [K], so ~18 rejected rows with exact states per check; a 32-node tree leaves 23 of 32 nodes
off the accepted path.

**Exact:** yes (a drafter input; the check stays the judge). **Cost:** a gather of up to 32 rows x 4 streams from the
check's last-layer activations, which exist anyway; the head's input width is unchanged if the state replaces the
carried one, or +1x if concatenated (a retrain of the input projection).

**What would kill it.** On captured checks, the head's per-depth accuracy with the sibling state at depth d is no
better than with its own carried state (the sibling branch diverges too fast after the near-tie), or it is better only
at depths 1-3, which the tree already covers (band 0 98.6 with tree-16 [K]). Also: if the head cannot read a state
whose token history differs from the tokens it is given (a mismatch it never saw in training), a short fine-tune with
sibling states in the mix is needed before the verdict.

**Monday test (<= 15 min, CPU/lab, no engine).** On the stored true-state cells (cells_t30b_heplus24, b316365: the head
with the TRUE trunk state at every anchor) plus the 127-node dump spec's rows, or on any capture that holds the 125B's
state for rejected rows: for each check start after a miss at m, take the sibling branch's states at m+1 .. m+8 and
score the head's depth-d guess from (sibling state at m+d, the real tokens t .. t+d-1) against the 125B's token; compare
with own-state (chain) and true-state rows of the same cells. PASS: >= half of the own-state -> true-state gap closed at
depths 1-8 on HE+ (own 91.5 / 92.4 -> true 95.3 / 96.4 at bands 1-2 / 3-6 [K THINK]). If no capture has the rejected
rows' states, Wednesday's capture adds them (one array per row; the check computes them already).

## 2. COPY-SOURCE STATES for code: the prefill's exact 125B states stand in for the future state inside a copied span  [new]

**Mechanism.** In code, long drafted spans copy the prompt (the longest-match lookup already finds them: L1-10, TRIED
843, PASS). While the draft is inside a copy of prompt span s .. s+L, the 125B's exact prefill state at s+d (free: the
prompt's states are in the cache) is an exact 125B state on the same local tokens. Feed the head that state at depth d
instead of its carried state (or beside it). Exact: yes. Cost: a gather, 0 bytes.

**Numbers.** Copy spans are where the chain goes deepest and where drift has the most depth to grow; the gap at 23-30
is 8 points [K report]. HE+ and MBPP copy rates: not in my records (the copy-gate session has them; the test measures
it). **Kills it:** the prefill state's context differs from the copy's (different preceding code), and the head reads
that difference as noise: no gain over own-state on copied spans in the lab join. Also harmless if copy spans are rare.

**Monday test (<= 15 min).** Lab join on the true-state cells: mark answer rows that extend a longest match >= 8 tokens
into the prompt; for those rows score the head on (prefill state at the source position + d, tokens) vs own-state vs
true-state. PASS: within 1 point of true-state on copy rows.

## 3. TREE-STATE REUSE at check starts: start the next block from the deepest EXACT state, not the bonus token  [new, with problem 1]

**Mechanism.** After a tree check the accepted path ends at a node whose 125B distribution names the next token t. If
the tree holds a node for t under that parent (siblings at the near-tie are exactly what best-first adds at thin
margins), that node's state is the exact true state one token deeper, and its own distribution names the next token;
walk this as long as the tree has the named child. Standard tree acceptance already does this walk for the DRAFTED
tokens; the extra is to keep walking through nodes the drafter ranked below #1, and to hand the head the exact state
at the walk's end as the next anchor. Exact: yes (every token on the walk is the 125B's own argmax).

**Numbers.** 66-86 % of K64-128 nodes sit off the #1 spine [K S11]; tree-16 lifts band 0 from 92.0 to 98.6 [K]. The
gain here is one or two more exact anchors per check at the near-tie, where the head is weakest. **Kills it:** if the
tree's off-spine nodes almost never hold the 125B's argmax child beyond the accepted path (K98 says the true path, once
off #1, gets only #1 continuations [K]): then the walk ends at once. The S11 dump answers this without a run.

**Monday test (<= 15 min).** From the 32- / 127-node dumps: at each check end, count how often the 125B's argmax child
exists in the tree, and the walk's length; the number of extra kept tokens per check is the result (PASS >= +0.5).

## 4. HALF-BLOCK RE-ANCHOR: a staged check gives one exact mid-block state  [tried in part: TRUNK REFRESH KILL, TRIED 1432]

**Mechanism.** Check rows 1-16, then draft rows 17-32 from the exact state at row 16 and check them: the drafter never
runs deeper than 16 from a true state. **Numbers [K THINK]:** two 16-row checks cost 2 x 38.8 = 77.6 ms against 52.0
for one 32-row check (x1.49), and the chain's first miss sits at ~14, so the second half is rarely reached by a chain;
true-state bands 7-14 are 97.8-98.8 vs the chain's 94.3-96.6. **Verdict on paper:** KILL on cost unless problem 2's
staged verification makes the second half nearly free (shared prefix, the first half's expert loads kept); then it is
worth its arm. Exact: yes. Not new in spirit (a refresh pass; TRUNK REFRESH was killed as a speed lever at 45 ms per
pass); the staged form is the only one that can be cheap.

## What I did not propose

- Any pass of the 125B over the draft (self-speculation, early exit, resident-expert or MSB passes): bytes, TRIED 920 /
  970 / 1014 / 1219. A dense-only block pass is 3.44 GB = 12 ms, 3-6x the budget.
- State composition from a small model (COMPOSER, TRIED 1020, R2 0.588: KILL) and learned choosers (TRIED 1021 / 1032).
- Overlap tricks: the head's 2-4 ms is already below the check; the limit is bytes, not time.

## Summary line

Best: SIBLING-STATE DRAFTING, the 125B's exact states at the rejected rows (near-tie siblings) as the head's input for
the next block; 0 bytes, exact, testable Monday on stored cells; then copy-source states for code, tree-state reuse at
check starts, and a staged half-block re-anchor only if problem 2 makes it free.
