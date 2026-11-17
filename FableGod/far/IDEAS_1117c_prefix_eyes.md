# IDEAS_1117c_prefix_eyes: the drafter sees the 125B's whole prefix, and the draft never waits (Fable FAR, Tue 17 Nov 2026 11:1x-11:4x, desk)

BREAK98c, lens 3 + 4: "the drafter is a separate head that sees the 125B only at the last verified position" and
"drafting and checking alternate". Marks: [K] measured (source), [P] arithmetic on [K], [G] judgement.

> **RESULT (Sandbox98 12:3x, LEVERS_A.md section B98C, e7d558a): PREFIX EYES = KILL on the 2B, both arms.** HE G(9-30)
> +0.023 [-0.080, +0.135] at block lr 3e-5 / -0.077 [-0.200, +0.046] at 1e-4; every band's G < 0.10; tree-32 x0.985 /
> x0.974 vs the control (MBPP x0.978 / x0.979). The arm was the GLIDE-form block reading the frozen trunk's cached K/V
> of the verified prefix at full-attention layers 15 + 19, zero-init output, with the control keeping the native head's
> attention over the trunk's final states (my caveat). The block grew (4-11 % of o_proj) and the training loss ROSE
> (0.538-0.540 vs 0.529): on the 2B the mid-layer prefix K/V add nothing to what the head already reads from the final
> states. Not tested: more steps, q learned from scratch, other layers, the copy-source prefill input (the one half of
> idea 1 still open; it needs a capture with prefill states).

## 0. What the two assumptions are worth, from the measured parts

- **Assumption 4 (alternation) is bounded at x1.16-1.35 and is really worth ~x1.1.** The draft is 8.8-16.6 ms of a
  57-65 ms cycle [K PLAN900 v4, BREAK98c walls]; hiding ALL of it is x1.16-1.35 at today's kept, and section 2 shows
  only ~30 % of HE+ blocks can have a draft ready before their check ends. 550 does not live here.
- **Assumption 3 is the only one of the two whose ceiling reaches 550.** The head on the 125B's TRUE state keeps
  20.77 per 32-row chain against 13.77 on its own carried state, and 26.79 with tie siblings at 32 rows [K WEEK98 LAB,
  BREAK98]; 26.79 / 48 ms = 558 tok/s with the draft hidden. The whole kept gap (+7 per check) is DRIFT: the head's
  carried state loses the 125B's context as the chain deepens (97.8 % flat on true state; 89.7 % at 23-30 on its own
  [K report]). VERDICT-DEPTH closed the route of reading the 125B's decision at the DRAFT row before its last layers.
  It did not close the other half of "true-state-class information": the 125B's exact view of the PREFIX, which is
  final, already computed, and in memory.

## 1. PREFIX EYES: the drafter reads the 125B's exact K/V of the whole verified prefix at every level  [new here; GLIDE (2024) form, plus copy-source states]

**Assumption broken:** 3. The head today gets one 125B state (the anchor) and then tokens; everything deeper is its own
state. Here every level of the draft also cross-attends into the 125B's own attention keys/values of the entire
prefix (prompt + generated) at 2-3 late attention layers: the 125B's memory of the context, through the 125B's eyes,
not the head's. On code the context IS the information (names, signatures, the docstring being copied). Second
input, same mechanism, zero cost: inside a copy of a prompt span (longest match >= 8, the L1-10 lookup [K TRIED 843])
the head's carried state at depth d is REPLACED by the 125B's exact prefill state at the source position + d: a true
state of the same local tokens, free (the prefill's states are cached).

**Why today's walls do not apply.**
- Not a 125B pass (TRIED 920 / 970 / 1014): no expert and no dense layer runs. Bytes per pop: the prefix's K/V at 3
  attention layers; the 125B is a hybrid with one attention layer in four and ~2 KB of K+V per token per layer, so a
  2K-token prefix is ~12 MB per pop [P], +7 % on the head's 163 MB [K], read from a cache that is in memory anyway.
- Not partial depth (VERDICT-DEPTH): those K/V are the final products of all 48 layers at the PREFIX positions; what
  VERDICT-DEPTH measured is the draft row's own decision at early layers. Different object.
- Not the head's own information: the 16.26 cap [K] counts what one anchor state + tokens + the head's small context
  hold. The 125B's attention over 2K tokens is new input. Not SIB (verified positions only, exact), not a student head
  (the head keeps its size), not future-state rows (nothing unknown is read).
- GLIDE (Du et al., 2024) got its acceptance gain from exactly this: a small drafter whose attention reads the
  target's KV cache; EAGLE-3 fuses prefix features for the same reason [G, prior art].

**Number [P on K].** Chain: own 13.77 -> true 20.77 (+7.0). Tree-32: 16.26 -> ~23 at true-state class; with tie
siblings 26.79 [K]. At G = share of the gap closed: G 0.5 -> tree-32 ~19.7, K64 ~24.3; G 0.8 -> tree-32 ~21.8, K64
~26.4. tok/s at the realistic check (32 rows 48 ms, 64 rows ~57 ms, draft 8.8, overhead 1.5 [K/P]): G 0.5 K64 364
(hidden draft 426); G 0.8 K64 396 (hidden 463); G 0.8 at 32 rows + ties ~25 -> 25 / 58.3 = 429, hidden 505. **550 needs G
>= ~0.85 with ties, or G ~0.8 plus the draft hidden plus Leap's 40 ms check (25 / 41.5 = 602).** It is the one lever
in my lens whose ceiling is above the line.

**The one decisive test (Sandbox98, ~40 min of its time; not a 10-min test).** Arm `kvx` vs `ctl`: the head's level
gains one cross-attention block (zero-init output gate) over the frozen 2B trunk's K/V of the whole prefix at layer
~3/4 depth (one layer; two if cheap); `sb/train_multi.py --arms ctl:-,kvx:KVX --depth 30 --steps 400`; read-out
`sb/evalx.py` walk-strict by band and `sb98/tree8.py`. PASS: G >= 0.3 at bands 9-30 (where drift lives) and tree-32
kept >= +8 %; KILL: G < 0.1 at every band. Caveat the arm must respect: if the 2B's native MTP head already attends
over the trunk's FINAL features of the prefix (EAGLE form), ctl must keep that, so the arm measures the mid-layer K/V
only. The copy-source half has its own 10-min lab join once a capture holds prefill states (IDEAS_1115 idea 2).

**Quality risk:** none. A drafter input; the check is unchanged and exact; lossy rules untouched.

## 2. THE DRAFT NEVER WAITS: pre-draft off the GPU in the check's bandwidth slack  [assumption 4; the reuse half TESTED today: KILL]

**Assumption broken:** 4. The check is GPU-paced at 96-97 % [K tue1] and reads ~4.6 GB in ~48 ms = ~96 GB/s, so
~190 GB/s of memory bandwidth idles during every check [P]. The head is bandwidth-bound (0.56 ms of bytes per pop
[K]), so a copy of it on the CPU cluster or the Neural Engine can draft DURING the check, from candidate ends. Three
sources of a ready draft:
- (a) the chain's own continuation (levels 32-62) when the chain runs to the end: ready with certainty;
- (b) the head's #2 token at the lowest-p rows (the miss locator, AUC 0.954 [K]): a candidate end each;
- (c) the OLD chain's suffix after the miss, re-used as the next draft's spine with no new levels.

**Tested today (10 min, screens_1117/reuse.py on the 127-node own-token tree dumps, walk-weighted): (c) is a KILL.**
After a walk first miss, the old chain's rows beyond the miss match the truth 14.1 / 12.4 / 13.1 % per row at
offsets 1-4 / 5-8 / 9-16 (HE+), against 62.6 / 49.1 / 43.5 % for a fresh chain from the new anchor; old and fresh agree
on 22-25 % of rows; kept of the next block 0.74 re-used vs 4.83 fresh on the same rows (MBPP 0.57 vs 3.56). A
wrong token poisons everything the head drafts after it; nothing downstream survives.

**Number [P on K].** Ready drafts per HE+ block: (a) P(full chain) = 0.249 at walk starts [K endcover]; (b) lowest-1
row + end covers 0.371, so the lowest-p row adds 0.12 of blocks, of which the #2 token is right ~0.5 walk-weighted
[K b98_2b_reconcile class]: +0.06; a second and third candidate (ANE + CPU can hold ~3 trees per check [P: 163 MB x 16
pops = 2.6 GB per tree, ~26 ms on the ANE's ~100 GB/s]) +0.04-0.05. Total ~0.35 of HE+ blocks, ~0.02 of MBPP. Saved
draft time 0.35 x 8.8-16.6 ms = 3-6 ms of 57-65: **x1.05-1.09 HE+, x1.01 MBPP.** Exact. Worth doing as engineering once
kept is up, not as a bet.

**Decisive test:** done for (c); (a) + (b) are arithmetic on measured coverage. The only open number is the ANE's
real per-pop time for the Q8 head (a one-afternoon Leap/MoonShots bench, not mine).
**Quality risk:** none (exact; the check judges every row).

## 3. Not proposed
- A wavefront check (rows enter the forward staggered so the drafter can revise later rows from earlier rows' states):
  each layer step then serves ~1 row, so the expert bytes multiply by 8-16x against a 25 ms byte floor [P]. Dead.
- The 125B as its own drafter at reduced width / depth: closed (partial channels).
- Future-state rows, SIB states, 2B siblings, level pairing, a student head: on the kill list.

## Summary line
(PREFIX EYES = KILL on the 2B, see the banner; the copy-source half is untested.) Best was: PREFIX EYES, the drafter cross-attends into the 125B's exact prefix K/V at 2-3 late layers (+ copy-source
prefill states inside copied spans): the only lens-3 lever whose ceiling (LAB + ties 26.79 at 32 rows = 558 at 48 ms)
is above 550; G 0.8 + hidden draft + Leap's 40 ms check = ~600; decisive test = Sandbox98 arm kvx, PASS G >= 0.3 at bands
9-30. Assumption 4 is worth ~x1.05-1.09 on HE+ (old-suffix reuse KILLED today: 13 % right per row vs 58-63 % fresh).
