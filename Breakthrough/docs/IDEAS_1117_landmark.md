# IDEAS_1117: LANDMARK, true 125B states by context match (Breakthrough, BREAK98 desk sprint, Tue 17 Nov 2026)

Lens (BREAK98.md): an idea from an unusual place that delivers the 125B's state at drafted positions far cheaper than
a 125B row. Brief: ~/FableGod/deepdraft/BREAK98.md (orchestrator 07:3x, via NewIdeas).

## Pre-registration of screen LANDMARK-1 (Tue 07:3x, before computing)
**Question:** at a drafted position p, a 125B state may already exist for an earlier position q with the same recent
context, from the prompt prefill or an earlier check. How often is there one, and how close is it to the true state at
p?

**Data:** TrainLab's D-17 capture ~/TrainLab/data/d17f.
- 8 HumanEval windows (code, decisive). 8 gsm, 8 ifeval and 8 multi windows are info only.
- Each window is the 125B's own greedy text (prompt + answer) with, per position:
  - f: the mean of the 4 hyper-connection streams after zero-based layers 5 / 15 / 25 / 35 / 45 (fp16, 2560);
  - top1: the 125B's next-token argmax.

**Mechanism simulated:**
- A drafted position p at depth d has true 125B states only up to p - d: the prompt plus everything checked so far.
- The landmark for p is the most recent q <= p - d whose last k tokens equal p's last k tokens. The longest k in
  {8, 6, 4, 3, 2} wins.
- Tokens are the true text, the walk-strict assumption: the draft is right up to p.
- Answer positions only, p >= gen + 1.

**Measures** (tap 45 decisive, the deepest captured, nearest the state the head reads; taps 25 / 35 info):
- **coverage:** the share of answer positions with a landmark (k >= 2).
- **cos(f[q], f[p]) at the landmarks.** Two baselines at the same positions:
  - the anchor f[p - d]: the last true state, which is what the drafter starts its block from;
  - the neighbour f[p - 1]: optimistic, since it is not available at d >= 2.
  - Also the unit-vector distance |u - v| = sqrt(2 - 2 cos), the 2B sandbox's "direction error" scale. There the head's
    carried state reads 0.85-0.99 and sibling states 0.95-1.10 (THINK_1116c, PLAN900 9b). Those are a different model;
    they are context, not a line.
- **next-token agreement:** top1[q] == top1[p] at the landmarks.

**Lines** (code set, d = 8, the middle of the drift bands 3-14):
- PROMISING (worth a head screen): coverage >= 25 %, AND median cos(landmark) >= median cos(anchor) + 0.15, AND top1
  agreement >= 80 % at landmarks.
- KILL: coverage < 10 %, OR median cos(landmark) <= median cos(anchor).
- Otherwise BETWEEN.
- d = 1 / 4 / 16 and the other sets are info.

**Cost:** CPU, numpy, < 1 GB, < 2 min.
- Script: scripts/break98/landmark_screen.py.
- Output: data/break98/landmark_screen.out.

## Result of LANDMARK-1 (Tue 07:3x; data/break98/landmark_screen.out; 0.4 s CPU)
| set (d = 8) | coverage | cos t45: landmark / anchor f[p-d] / neighbour f[p-1] | paired landmark - anchor | top1 agree |
|---|---|---|---|---|
| code (8 HE windows, 2,392 answer positions) | 81.0 % | 0.637 / 0.624 / 0.766 | +0.035 | 66.2 % |
| gsm | 60.2 % | 0.817 / 0.627 / 0.842 | +0.093 | 67.6 % |
| ifeval | 47.9 % | 0.865 / 0.648 / 0.812 | +0.147 | 86.9 % |
| multi | 27.8 % | 0.752 / 0.552 / 0.798 | +0.181 | 65.5 % |

**VERDICT (pre-registered, code, d = 8): BETWEEN**, on the KILL side for its intended use.
- Coverage is high (81 %, mostly exact 8-gram matches).
- But at tap 45 a landmark's state is no closer to the true state than the stale anchor (+0.035; the line was +0.15).
- Next-token agreement is 66 % (the line was 80 %).
- d = 1 / 4 / 16 read the same.
- As a stand-in for the late state the head reads, LANDMARK fails.

## Post-hoc (NOT pre-registered; information only, to be confirmed): the depth asymmetry
Where the code landmarks come from (d = 8):
- **60 % of answer positions match the PROMPT.** The 125B restates the signature and docstring before coding.
  - The next text token there agrees 96 %, but the tap-45 state does not (cos 0.606 vs anchor 0.612).
  - The model's state while copying differs from its state while first reading.
- **21 % match earlier ANSWER positions:** tap 45 is 0.892 vs anchor 0.782 (+0.09).

The same landmarks at MID-DEPTH taps are much closer (median cos, code, d = 8):

| tap (layer after) | landmark | anchor f[p-d] | neighbour f[p-1] |
|---|---|---|---|
| 15 | 0.785 | 0.405 | 0.554 |
| 25 | 0.687 | 0.485 | 0.610 |
| 35 | 0.493 | 0.375 | 0.463 |
| 45 | 0.637 | 0.624 | 0.766 |

- Answer-internal landmarks reach 0.914 / 0.841 / 0.710 at taps 15 / 25 / 35. gsm reads the same (tap 25: 0.788 vs 0.476).
- **Reading:**
  - The 125B's mid-depth state at a position is mostly a function of the LOCAL context, so an earlier position with the
    same recent tokens carries it.
  - Its late state is HISTORY-specific: no token-keyed lookup recovers it better than the stale anchor does.

## What this means for BREAK98 (the information question)
- **The believed limit holds for a whole class.** Retrieval, memoization, copying and any token-keyed state cache can
  deliver the 125B's mid-depth (local) information at drafted positions almost for free. They cannot deliver its late,
  history-specific state.
- The LAB ceiling (+51 % kept, 20.77 vs 13.77) rides on the late state the head reads. The only mid-depth lever measured
  so far, MTP Head's tapmlp, is far smaller:
  - S12: the true layer-12 residual of the 2B -> MLP -> added to the carried state;
  - HE walk-strict 23-30 89.3 -> 92.4, kept +5.1 % HE+ / +2.7 % MBPP.
  - "Mid-depth taps the 125B can't give" closed it: their only cost was a partial pass per drafted token.
- So a breakthrough channel must carry information computed WITH the verified history: attention over the real prefix,
  i.e. the late layers. It cannot be keyed on the drafted tokens alone. (For Lunatic's floor and ZERO's channel design.)

## The idea that follows: LANDMARK-TAP (free mid-depth taps for tapmlp). A small lever: an add-on, not the winner
- **Mechanism** (pedigrees: CPU value prediction's finite-context method; hippocampal landmark resets of path-integration
  drift; HEVC screen-content intra block copy):
  - The check already computes every layer of its rows, and the prefill every prompt position. Both write the
    layer-24 residual of each verified row into a GPU tap store: 2560 fp16 = 5 KB per row, 160 KB per 32-row check.
  - A GPU-resident rolling-hash table maps the last k = 8 / 6 / 4 / 3 / 2 verified tokens to the latest position.
  - Each drafted node hashes its own last k tokens in-kernel (no CPU round trip). On a hit, its tapmlp input is
    tapstore[q]; on a miss, no tap.
- **Delivers:** the 125B's true mid-depth state at a context-matched position, at ~80 % of drafted code positions.
  Measured cos 0.69 to the true tap-25 state, vs 0.49 for anything the drafter has today (post-hoc, 8 HE windows).
- **Price:** 0 bytes of 125B weights.
  - Per drafted level: 5 hash probes + one 5 KB read, plus tapmlp's own MLP weights (its width sets this; 2560 -> 512 ->
    2560 fp16 = 5 MB per level).
  - Per check: a 160 KB write. Exact: the check is unchanged.
- **Buys at most:** tapmlp's measured gain x coverage x G, where G is the share of the true-tap gain an approximate tap
  keeps. Ceiling ~ +5 % kept on HE+ (2B analog). Kamen's bar puts that among the small levers.
  - It is free and exact, so it is worth one screen as an add-on to whatever wins.
  - It is not the BREAK98 answer.
- **Screens:**
  1. Sandbox98 (2B, S12's frozen tapmlp head): at drafted positions, feed the landmark's true tap-12 (context-matched
     within the verified region) in place of the true tap-12.
     - Walk-strict bands vs no-tap and true-tap; G = (landmark - own) / (true - own).
     - PASS G >= 0.5 on HE; KILL G < 0.2. ~30 min, <= 5 GB.
  2. 125B confirmation of the depth asymmetry on tonight's capture (pre-registered before the data exists):
     - LANDMARK-1's code on mid_{heplus24,mbppbank}, layers 8 / 16 / 24, against the anchor.
     - CONFIRMED if layer-16 / 24 landmark cos >= anchor + 0.15 on HE+ answer positions.
- **Capture ask for tonight (by 18:00, to BigMoE):** for the mid_* rows, also save the token id of every captured row,
  and the PROMPT rows' layer-16 / 24 residuals (60 % of code landmarks sit in the prompt). Without them, screen 2 can only
  use answer-internal landmarks.
- **Quality risk:** none (verification unchanged). **What kills it:** screen 1's G < 0.2, or screen 2 not confirming the
  asymmetry on the 125B.

## Pointer for ZERO / FAR (desk only, untested): LANDMARK-QUERY
The asymmetry says the missing information is the late layers' read of the real history. That read can be reached
without a per-token pass.
- **Mechanism:**
  - After a draft, every drafted position has a landmark mid-depth state (free, above).
  - Push those states through the 125B's own late full-attention layers' query projections, and attend over the REAL
    verified KV cache of those layers. That is history-specific information computed with the true keys and values.
  - All drafted positions go in one batched pass. The inputs come from landmarks, not from the drafted rows' own
    computation, so there is no sequential dependency.
- **Desk price** (4 late attention layers): ~0.17 GB of q / o weights read once per draft + ~3 MB of KV per layer at a
  ~3K prefix, so ~1 ms. But the drafter must consume the readouts: a second, cheaper refinement step, or a head trained
  with them (FAR / TrainLab).
- **Risks:**
  - The queries are approximate (landmark cos 0.5-0.7 at taps 25-35).
  - Prior art exists for drafters reading the target's KV cache (GLIDE-style cross-attention); what is new here is the
    query source and the one-shot batching.
- **Screen it would need:** the d17f-style capture plus the late layers' KV and query projections, offline. Not doable
  by 09:00.

## Refinement, Tue 07:5x (post-hoc; supersedes LANDMARK-TAP as my best idea): LANDMARK-ANS
The prompt-copy finding suggests one change: look landmarks up in the ANSWER only (q >= gen). The model's state while
copying the prompt is not its state while reading it. Same data, same lag (q <= p - d), d = 8, tap 45 (median cos):

| set | coverage of answer positions (by answer third) | landmark | anchor f[p-d] | neighbour f[p-1] | landmark > neighbour | top1 agree |
|---|---|---|---|---|---|---|
| code (8 HE) | 38.0 % (20 / 46 / 49 %) | 0.795 | 0.644 | 0.798 | 64 % of positions | 66.3 % |
| gsm (8) | 50.8 % (24 / 57 / 72 %) | 0.858 | 0.659 | 0.856 | 50 % | 75.6 % |

- **At the late tap the head reads,** an answer-internal landmark is as close to the true state as the true state one
  position back (0.795 vs 0.798). A drafter never has that state at depth >= 2. The landmark is +0.15 over the anchor.
- Coverage grows along the answer, to about half of the positions past the first third.
- Measured on 125B states but post-hoc, and on 8 HE windows only.
- **A history correction does not help.** f[q] + (f[a] - f[q - d]), the codec's motion-compensated prediction, scores
  0.637 vs plain 0.637 on code at d = 8, and it HURTS answer-internal matches (0.894 -> 0.800). Not proposed.

**Mechanism (LANDMARK-ANS):**
- The check already produces the head-input state of every row it verifies; today the head keeps only the anchor's.
- Keep all accepted rows' head-input states in a GPU state store: 2560 fp16 = 5 KB per row, ~1.5 MB per 300-token
  answer.
- A GPU rolling-hash table maps the last k = 8 / 6 / 4 / 3 / 2 ANSWER tokens to the latest verified position.
- At each drafted node, hash the node's last k tokens in-kernel. On a hit q <= the anchor, the head reads
  store[q], replaced or mixed at weight w; on a miss it keeps its carried state.
- **Price:** zero 125B bytes; per drafted level, 5 probes + one 5 KB read (microseconds). Exact: verification is unchanged.

**Bound:**
- Covered positions get a late state as close as a 1-step-stale true state. Walk-strict, the true-state chain reads
  95.3 % at band 1-2 and 96.4-97.8 % at bands 3-10, vs own-state drift gaps of 4.0 / 3.5 / 2.2 pt at bands 3-6 / 7-10 /
  11-14.
- So, at ~40 % coverage, the ceiling is roughly 40 % of the drift gap closed on code, if the head treats a landmark
  like a true state.
- That falls between tapmlp's +5 % kept and LAB's +51 %. Whether the head does treat it so is the open question. SIB's
  sibling states looked like a source and were WORSE than the carried state, but they came from a different sequence
  (direction error 0.95-1.10). A landmark is a true state from the same answer at the same recent context.

**Screens (pre-registered here, before any of them runs):**
1. **Sandbox98, SIB's harness with a new state source** (2B, frozen s30 head, LAB, HE 60 + MBPP 100):
   - Landmark source: answer-internal, k = 8..2 longest, q <= p - d.
   - Arms w = 1.0 and w = 0.5 at covered positions; own carried state elsewhere.
   - G = (arm - own) / (true - own), walk-strict bands 1-8, covered positions; whole-walk kept as info.
   - PASS: G >= 0.50 on HE (>= 0.25 MBPP) on either arm. KILL: G < 0.10 on both.
   - <= 5 GB, ~30-50 min, the same scale as SIB's phase 1.
2. **125B confirmation** of the table above on tonight's capture. CONFIRMED if, on HE+ answer positions, answer-internal
   coverage >= 25 % and late-state landmark cos >= anchor + 0.10.

**Capture ask for tonight (replaces the earlier one):** for every mid_* row, prompt and answer:
- the token id;
- the head-input state the t30b head reads (the 125B's final state, as LAB feeds it);
- layers 16 / 24 as planned.
That is ~5 KB per row for the final state, ~0.2 GB over HE+ 24 + MBPP 100.
- **For the screen's harness:** scripts/break98/landmark_source.py is a drop-in LandmarkIndex.
  - Build it from the window's ids and gen; call lookup(path_ids, p, anchor) to get (q, k).
  - Its self-test checks the lookup against brute force on 200 random token streams, both modes, d = 1 / 4 / 8: PASS.

## LANDMARK-ANS pre-registration ADDENDUM (Tue 08:0x, BEFORE either screen runs; the orchestrator's audit via NewIdeas)
Context: MoonShots' SESSION VALUE PREDICTION (~/MoonShots/week98/break98/IDEAS_1117_valuepred.md) = KILL. It is the
same lookup, measured on the bank's wd (the head's true input state, 10,240 wide):
- answer-side n-gram hits 11-12 % on HE+, cos 0.757 there, R2 <= 0.22;
- the hindsight-best earlier state 0.67-0.76 rad off; only 3-6 % within 0.40 rad;
- the retrieved state's call equals n-gram copying (61-63 %).
- My cos 0.795 (~0.65 rad) is the same geometry.
The claimed difference is not the distance. It is (i) answer-only matching with k up to 8, and (ii) "stale by one": a
landmark is as close as the TRUE state one position back, and the head handles one step of drift well. That holds only
if a landmark carries what a 1-back true state carries; that is the claim under test.

**Screen 1 (Sandbox98) reports, at the SAME covered positions:**
- (a) G_landmark at w = 1.0 and w = 0.5;
- (b) G_stale1: the head fed the TRUE state one position back. This is the upper reference the cosine equality implies.
- (c) G_copy: no state change; the drafted token is replaced by the token that followed the matched earlier occurrence.
  This is the lookup-decoding baseline (on record: LOOKUP-SCREEN).
- The distance histogram: the share of landmarks within 0.40 rad, beside MoonShots' 3-6 %.
- G on covered positions AND the whole-walk kept, so the clustered ~38 % coverage is priced as it falls.

**Added lines** (alongside PASS G >= 0.5 / KILL G < 0.1):
- KILL-COPY if G_landmark <= G_copy + 0.10 (lookup in disguise: MoonShots' kill stands).
- WARN if G_landmark < 0.5 x G_stale1 (the cosine overstates the information; report only).

**Screen 2 amendment (data source only; lines unchanged):** run it now on the existing bank instead of tonight's
capture. ~/datasets/deepdraft/bank has, for prompt AND answer rows of HE+ 24 and MBPP 100, the ids and wd (the head's
input state), as MoonShots' svp_oracle.py reads them. So my capture ask is withdrawn.
- Same procedure as the d17f table: answer-internal landmarks, k = 8..2, q <= p - d, d = 1 / 4 / 8 / 16.
- cos(wd[q], wd[p]) vs the anchor wd[p - d] and vs the 1-back true state wd[p - 1].
- Also reported: the share of landmarks within 0.40 rad, and the 125B's top-1 agreement (tk[q, 0] == tk[p, 0]).
- CONFIRMED (as registered at 07:5x): on HE+ answer positions at d = 8, answer-internal coverage >= 25 % AND median
  landmark cos >= median anchor cos + 0.10. MBPP is info.
- Script: scripts/break98/landmark_bank.py; output: data/break98/landmark_bank.out.

## Screen 2 result (Tue 08:0x; data/break98/landmark_bank.out; CPU 0.7 s): CONFIRMED (pre-registered line)
On the bank, wd = the head's true input state (10,240 wide). Alignment check: tk[t, 0] == ids[t + 1] on answer rows,
median 1.000. Answer-internal landmarks, k = 8..2:

| set, d | coverage | cos landmark | cos anchor wd[p-d] | cos 1-back wd[p-1] | landmark - anchor | within 0.40 rad: landmark / 1-back | top-1 agree |
|---|---|---|---|---|---|---|---|
| HE+ 24, d 1 | 34.2 % | 0.724 | 0.619 | 0.619 | +0.106 | 5.5 % / 0.0 % | 46.8 % |
| HE+ 24, d 8 | 32.9 % | 0.725 | 0.460 | 0.619 | +0.246 | 5.6 % / 0.0 % | 48.9 % |
| HE+ 24, d 16 | 29.8 % | 0.714 | 0.461 | 0.619 | +0.224 | 5.5 % / 0.0 % | 48.0 % |
| MBPP 100, d 8 | 33.4 % | 0.753 | 0.478 | 0.673 | +0.256 | 9.4 % / 0.2 % | 49.8 % |

- **VERDICT:** HE+ d = 8, coverage 32.9 % (line 25) and landmark cos 0.725 vs anchor 0.460 (line +0.10): **CONFIRMED.**
- On the head's own input, a landmark is CLOSER than the true state one position back (0.725 vs 0.619). On d17f's tap
  45 they were equal.
- **The audit's distance histogram:** landmarks within 0.40 rad 5.5 % (HE+) / 9.4 % (MBPP), like MoonShots' 3-6 %.
  But the 1-back true state is within 0.40 rad 0.0 % / 0.2 % of the time.
  - So if the head handles one step of staleness, the 0.40-rad line cannot be what it needs.
  - If it does not, G_stale1 will show it. The head screen decides; the geometry cannot.
- Top-1 agreement equals copy-right (49 %) by construction on generated rows, as MoonShots found. That is why G_copy is
  the KILL-COPY baseline.
- **By match length** (HE+, d = 8, info):

  | longest k | share of positions | cos landmark | top-1 agree |
  |---|---|---|---|
  | 8 | 1.8 % | 0.779 | 61 % |
  | 6 | 2.8 % | 0.797 | 63 % |
  | 4 | 6.5 % | 0.755 | 60 % |
  | 3 | 7.2 % | 0.719 | 51 % |
  | 2 | 14.6 % | 0.691 | 39 % |

  - Long matches (k >= 4) are ~11 %, MoonShots' hit share.
  - Report-only addition for screen 1, before it runs: G and G_copy by longest-k band (k >= 4 vs k <= 3).

The capture ask is withdrawn: the bank already holds ids + wd for prompt and answer rows.

## Screen 1 alignment clarification (Tue 08:1x, BEFORE it runs; Sandbox98's 08:04 question)
In Sandbox98's 2B chain, band d's cell (anchor s) reads an input state for position p = s + d plus the token ids[p + 1],
and predicts ids[p + 2].
- **Landmark / stale1:** they replace that input state with W[q] / W[p - 1] when p is covered. The lookup is at p:
  answer-only, q <= s, longest k of 8..2.
- **G_copy, two variants on the same covered cells; KILL-COPY uses the STRONGER of the two:**
  - (i) same match: predict ids[p + 2] as ids[q + 2]. This is Sandbox98's default: the same cell, the same match.
  - (ii) re-matched lookup decoding: look up the context ending at p + 1, which contains the true ids[p + 1] on a
    walk-strict path (q' <= s, answer-only, longest k), and predict ids[p + 2] as ids[q' + 1]. If there is no
    re-match, the cell keeps the head's own call. This is what real lookup decoding would do at that cell. It can beat
    (i) when ids[q + 1] != ids[p + 1] breaks the old match.
  - Info only: ids[q + 1] replacing band d-1's prediction, one band earlier than the landmark acts.
- **Covered-position G:** walk-strict, counted only over covered cells, with own / true / each arm on its own chain.
  Plus the k split (8 / 6 / 4 vs 3 / 2) and whole-walk kept as info. Sets HE 60 + MBPP 100, frozen s30.
- **Registered by Sandbox98 (LEVERS_A section LANDMARK, 9c6d421, 08:10; accepted 08:1x).** Same lines, read per arm
  (lm1 = w 1.0, lm05 = w 0.5), HE covered-cell G over bands 1-8:
  - KILL < 0.10;
  - KILL-COPY if G <= max(copy (i), copy (ii)) + 0.10;
  - PASS if G >= 0.50 on HE AND >= 0.25 on MBPP;
  - else BETWEEN; WARN if G < 0.5 x G_stale1.
  - **Stricter, and accepted:** a screen PASS needs an arm PASS whose k <= 3 cells also clear KILL-COPY. A PASS that
    clears it only at k >= 4 reads "BETWEEN (k >= 4 only)", because k >= 4 alone covers ~11 % of positions, not ~33 %.
  - It starts after FAR's B98S1 (ETA ~10:15): HE 60 + MBPP 100, frozen s30, ~15 min.
