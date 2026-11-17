# IDEAS_1117_floor: the information floor of the drift wall (Lunatic, Tue 17 Nov 2026, 07:3x-08:3x EET)

BREAK98, route 1, my lens: which part of the 125B state (layer, subspace, bits) does the head need, and what are the
fewest bytes per drafted row that deliver it? Desk work plus three CPU screens on the existing dumps and the bank (no
GPU; one process, <= 2 GB). Marks: [K] measured (here or in the records, source named), [P] arithmetic on [K],
[G] judgement. Unless stated: chain-32 (31 guesses), walk-strict, exact acceptance, HE+ = heplus24 / MBPP = mbppbank
t30b cells, priced at Leap's C6 + int8 bound like IDEAS_1116c (own chain-32 462 / 232 tok/s, tree-32 566 / 321).

## 0. The answer in seven lines
1. **Bits.** The head lacks ~10 bits per block on HE+ and ~28 on MBPP: the per-position cross-entropy gap between the
   carried and the true state, summed over depths 1-30 [K drift_bits.out]. That is 1-4 bytes per block.
2. **As states, they must come at nearly every row.** A true state injected every 2nd / 4th / 8th / 16th depth recovers
   78 / 49 / 25 / 11 % of the true-state (LAB) gain on HE+ and 68 / 36 / 13 / 2 % on MBPP [K floor_fix.out; NewIdeas'
   b98sim.py reproduces 19.20 / 17.18].
3. **As decisions, 1-3 per block are enough, and they beat the state at all 31 rows.** Locate the chain's first miss
   and give it the 125B's own token, with no state at all: 19.98 / 23.65 / 26.34 kept on HE+ for 1 / 2 / 3 decisions,
   against 20.77 for the true state at every row; MBPP 11.50 / 15.39 / 18.59 against 12.83 [K floor_fix.out]. The
   true-state head itself fixes only 68.5 % (HE+) / 65.4 % (MBPP) of the own chain's first misses [K floor_lens.out].
4. **Subspace.** A decision needs the 125B's logits on the head's own top-8 at that row: 8 numbers. y is in the head's
   top-8 at 92.6 / 92.7 % of the first misses [K floor_lens.out].
5. **Layer: the 125B decides late.** On the bank's own mid-layer taps, a linear tuned lens finds the chain's first
   misses only weakly at layers 6 / 16 / 26 / 36 (AUC 0.64-0.70), far worse than the head's own margin
   (0.954 / 0.896). Then the verdict appears between layers 36 and 46 (0.917) and completes by 48
   (0.991 through the same lens; exact 0.999). Control: the stream-mean format is not the bottleneck [K floor_lens.out].
6. **So the fewest bytes per drafted row are a full-depth row.** The information is tiny, but it is computed only in
   the last ~12 layers of the right-prefix row. No shallow slice carries it, so the believed limit holds at the
   INFORMATION level, not only at the byte level. The cheapest delivery is a full-depth row inside the check's batch
   (marginal ~0.3-0.4 ms). That means route 1 via partial depth is closed on today's evidence, and the floor's 1-3
   decisions per block must come from more decisions per weight read (route 2) or a head that needs fewer.
7. **Best action (a screen, not a lever):** tonight's capture brackets where the verdict forms on the FULL widened state
   (please add layers 32 / 40 / 44). The existing floor_lens.py runs on it in < 1 min. If layer 16 / 24 reaches the
   decider bar of section 5 (phi >= 0.7 at psi <= 0.5 % per right row, for a round of <= 4 ms), LENS REPAIR
   (section 6) is live. If not, route 1 by partial depth is closed and the GPU goes to route 2.

## 1. Bits [K drift_bits.out, IDEAS_1116c]
Bits lost per position: 0.078 (depths 1-2) -> 0.460 (23-30) on HE+, and 0.227 -> 1.248 on MBPP. The sum over 30 depths is
10.1 / 28.3 bits. In drift errors y is the head's 2nd choice 55-75 % of the time (HE+) and in its top-8 89-97 %. So the
missing information is a few bits at a few rows. Section 3 locates them: they are the 125B's decisions at the chain's
misses.

## 2. Rows, as states: the channel must be dense [K floor_fix.out]
True state at depths 0, P, 2P, ... with the carried state in between (exact walk on the cells, re-anchor convention of
drift_forksplit.py):

| injections per block | HE+ kept | share of LAB gain | MBPP kept | share |
|---|---|---|---|---|
| none (own) | 13.77 | 0 | 6.92 | 0 |
| every 16th (1) | 14.52 | 11 % | 7.04 | 2 % |
| every 8th (3) | 15.51 | 25 % | 7.69 | 13 % |
| every 4th (7) | 17.18 | 49 % | 9.03 | 36 % |
| every 2nd (15) | 19.20 | 78 % | 10.93 | 68 % |
| every row (LAB, 31) | 20.77 | 100 % | 12.83 | 100 % |

A state channel pays ~15 deliveries per block for ~3/4 of the gain. That matches "silent drift" (IDEAS_1116c):
the carried state degrades at every step, not at a few forks.

## 3. Rows, as decisions: 2 located decisions beat 31 states [K floor_fix.out, floor_lens.out]
FIX-r: a locator finds the chain's FIRST miss and gives the 125B's own token there. The head re-drafts the tail
from its own state with that token (no re-anchor; exactly the teacher-forced cells) or from the true state at that row
(re-anchor). This repeats for up to r rounds, then the full check verifies everything.

| decisions per block (rounds) | HE+ no re-anchor | HE+ re-anchor | MBPP no re-anchor | MBPP re-anchor |
|---|---|---|---|---|
| 0 (own chain) | 13.77 | | 6.92 | |
| 1 | 19.98 | 21.73 | 11.50 | 13.29 |
| 2 | 23.65 | 25.68 | 15.39 | 18.85 |
| 3 | 26.34 | 27.58 | 18.59 | 22.86 |
| the true state at all 31 rows (LAB) | | 20.77 | | 12.83 |

- Decisions used per check at r = 2: 1.10 (HE+), 1.88 (MBPP). The kept 24-26 that 550 needs (BREAK98) is ~2 located
  decisions per block. The state at every row does not get there (20.77).
- Why states lose: at the own chain's first misses the true-state head is right only 68.5 % / 65.4 %. The remaining
  ~1/3 are rows where the head errs even with the true state (ties, knowledge). Only the 125B's own token fixes them.
- **The floor in decision form:** ~1-2 decisions per block, each ~5 bits of position (which row) + ~1-2 bits of choice
  (y = the head's #2 55-75 %). That is ~7-14 bits per block, consistent with section 1's 10 bits on HE+.

## 4. Layers: where the 125B's verdict forms [K floor_lens.out]
The bank stores, along the true text, the mean of the 4 hyper-connection streams after zero-based layers
5 / 15 / 25 / 35 / 45 (`raw`, TrainLab/sticky/taps_extract.py format) and the final widened stream `wd`.
- **Tuned lens:** a ridge from each RMS-normalised tap to the 125B's final mixed vector (NewIdeas' ed_lens.Lens.mixed),
  then the 125B's own lm_head. It is fitted on 60 independent code windows (datasets/deepdraft/rawtaps) and never on
  the banks.
- **Controls:** the FINAL layer's 4-stream mean through the same ridge ("48 mean"), and the exact final mixer ("48").
- **Rows scored:** the plain chain-32 walk's first-miss rows (337 HE+ / 2,144 MBPP) against the right rows before them
  in the same walks. The 125B's stored argmax = cell y on 99.42 / 100.00 % of cells.

| after layer | locator AUC, lens alone (HE+ / MBPP) | recall at 2 % false (HE+ / MBPP) | decider phi at first misses / psi on right rows (HE+, lens + head log p, tau 0) |
|---|---|---|---|
| head's own margin (no 125B) | 0.954 / 0.896 | 45.7 / 20.1 % | n/a |
| 6 | 0.682 / 0.648 | 7.1 / 6.4 % | 23.1 % / 8.2 % |
| 16 | 0.698 / 0.652 | 8.0 / 6.2 % | 24.3 % / 7.5 % |
| 26 | 0.685 / 0.678 | 8.0 / 8.0 % | 25.8 % / 6.2 % |
| 36 | 0.637 / 0.667 | 4.5 / 7.4 % | 24.6 % / 6.5 % |
| 46 | 0.917 / 0.915 | 35.9 / 37.4 % | 50.7 % / 1.3 % |
| 48, 4-stream mean (control) | 0.991 / 0.975 | 86.1 / 68.1 % | 57.3 % / 0.5 % |
| 48, exact mixer (control) | 0.999 / 0.997 | 99.1 / 97.7 % | (= the 125B) |

- **Readings:**
  - The verdict on the head's miss rows is not in the residual through layer 36 in any linearly readable form. It
    appears between layers 36 and 46 and completes in the last two.
  - The final-layer control (0.991 vs 0.999 exact) shows the stream-mean format plus a linear map is not what hides it.
  - (The decider column with lens + head weighting under-reads the strong lenses: it is a fixed beta = 1 mix. The
    locator AUC is the clean information measure.)
- **As a repair decider, mid layers HURT:** FIX-r with the layer 6-36 lenses lowers kept (13.77 -> 10-13 HE+,
  6.92 -> 4.7-6.5 MBPP), because they override right rows 6-17 % of the time. The layer-46 lens lifts kept to
  16.3-17.0 HE+ but costs 46/48 of a pass per round: 372-447 tok/s against 462. That is the TRUNK REFRESH KILL again
  (TRIED leap.md:523).
- **Consistent with the records:**
  - LAYERS (BigMoE a856e7e): the head fed ridge-mapped taps {5} / {5,15} / {5,15,25} reads 41.7-56.1, {45} reads 91.6.
  - R-32 STOPPRED: a layer-12 probe adds nothing over the drafter's own features.
  - L1-09: an untrained exit after 40 agrees 5.7 %.
- **Caveat (the one open door):** the taps are the 4-stream MEAN. At mid layers the streams may differ more than at the
  end, so the widened state (tonight's capture) could hold more than this lens reads. The final-layer control argues
  against a large effect but does not prove it for layer 16.
- **Information for the early-death probe (not a verdict):** at layer 16 the stream-mean lens flags 8 % of chain first
  misses at 2 % false, against 45.7 % for the head's own margin. The prereg requires target information to beat
  P-draft by >= 10 points, so unless the widened state reads very differently, expect that line to fail.

## 5. The decider bar: how to judge ANY route-1 channel [K floor_bar.out]
Every route-1 channel (FAR's future-state rows, ZERO's check-as-tap-pass, a 2B auditor, a widened-state probe) does one
thing for the chain: it supplies decisions. Judge it by three numbers on the chain's own rows:
- **phi:** it overrides the draft at the chain's first miss with the 125B's token;
- **psi:** it falsely overrides a right row, which ends the block;
- **c:** its ms per round.

Synthetic decider on the HE+ cells, chain-32 + r rounds, at the bound:

| r = 2 rounds | phi 0.5 | phi 0.7 | phi 0.9 |
|---|---|---|---|
| c 2 ms, psi 0.1 % | 18.17 (540) | 20.16 (595) | 22.16 (650) |
| c 4 ms, psi 0.1 % | 18.17 (511) | 20.16 (561) | 22.16 (611) |
| c 4 ms, psi 0.5 % | 17.18 (482) | 19.31 (535) | 21.04 (578) |
| c 8 ms, psi 0.1 % | 18.17 (461) | 20.16 (504) | 22.16 (545) |
| c 4 ms, psi 2 % | 15.13 (419) | 16.55 (455) | 17.83 (486) |

- **To beat tree-32 (566) on a chain:** phi >= 0.7 with psi <= 0.1-0.5 % at c <= 2-4 ms, or phi >= 0.9 at c <= 4 ms.
  At c = 8 ms nothing beats it. MBPP (tree-32 321): phi >= 0.9 at c <= 2-4 ms.
- **Measured deciders against the bar:**
  - The 125B's own layer 6-36 lenses: phi 0.2-0.26 at psi 6-17 % (far below).
  - Layer 46: phi 0.51 at psi 1.3 %, but c ~ a pass.
  - The frozen 2B audit: +8.9 % kept at P = 2 [K PLAN900:54], i.e. phi ~0.1-0.2 equivalent [P].
- floor_bar.py prices any new (phi, psi, c) in ~1 s.

## 6. Ideas
### Idea 1 (best; a screen that decides route 1's partial-depth branch): VERDICT-DEPTH on the widened capture
- **Mechanism / question.** Where does the 125B's verdict on the head's miss rows form, on the FULL widened state?
  Section 4 says between layers 36 and 46 on the stream mean. If the widened state at 16 / 24 already carries it, every
  "check's first L layers" channel lives (ZERO's idea 1, LENS REPAIR below, early death). If not, they are all dead
  together.
- **Information it delivers.** The verdict-formation curve: locator AUC and the decider's (phi, psi) by layer.
- **Price.** Zero GPU beyond tonight's capture. CPU < 1 min per set.
- **Screen.**
  - Run floor_lens.py with the taps replaced by the capture's widened states and the ridge replaced by ed_stage2's
    residual ridge (cross-fit by file parity; the stand-in self-check already passes).
  - Report for layers 8 / 16 / 24 (+ 32 / 40 / 44 if added) the locator AUC (lens alone and a cross-fitted logistic
    with the head's margin), recall at 2 % false, and phi / psi at the chain's first misses. Then run floor_bar.py on
    them.
- **Lines (written now):**
  - **LIVE:** at layer <= 24, cross-fitted lens + head locator AUC >= 0.975 (the head alone is 0.954) AND decider
    phi >= 0.7 at psi <= 0.5 % on HE+.
  - **CLOSED:** at layer 24, lens-alone AUC < 0.80 (today's stream mean reads 0.69), meaning partial-depth route 1 is
    closed.
  - **BETWEEN:** otherwise.
- **Capture ask (by 18:00, BigMoE):** add the widened residual after layers 32, 40 and 44 (same convention and rows as
  8 / 16 / 24, both sets). That brackets the 36 -> 46 jump on the full state: it says how deep a decider pass must go,
  and so its price. It costs ~3 x 20 KB per row on disk [P].
- **Kills it.** Nothing; it is a measurement. Its CLOSED outcome kills the family.

### Idea 2 (conditional on Idea 1 LIVE): LENS REPAIR, decisions from the check's own first layers
- **Mechanism.**
  1. Draft chain-32 with the head (own state).
  2. The check runs layers 1..L over all 32 rows as usual. At layer L a decider re-ranks the head's top-8 at every row
     (lens logits + head log p).
  3. The first row where it overrides gets the decider's token. The head re-drafts the tail from its own state with
     that token (no re-anchor needed: section 3, 23.65 vs 25.68 at r = 2).
  4. The re-drafted tail rows run layers 1..L; then layers L+1..48 run for all 32 rows; exact.
  5. Up to r rounds. Layers 1..L of the confirmed prefix are reused, so the extra bytes per round are the dense re-read
     of layers 1..L plus the tail's experts. ZERO's union routing for the tail's early layers would cut the experts.
- **Price per round [P].**
  - L 16: (16/48) x (7.6 + 16.4 U(n)/U(32)) + 0.8 + the head's re-draft = ~6.5-8.7 ms for tails of 8-16 rows,
    ~4-5 ms with tail experts inside the loaded union.
  - L 8: about half that.
- **Number it moves.** Perfect at L 16: 590 / 639 / 660 tok/s at r = 1 / 2 / 3 (HE+, at the bound) [K floor_fix]. A
  real decider must meet section 5's bar. On today's stream-mean lens it LOSES kept (section 4).
- **Quality risk.** None: exact acceptance; a wrong override only costs kept.
- **Kills it.** Idea 1 CLOSED, or phi < 0.7 / psi > 0.5 % at L <= 24.
- **Novelty.** TRIED holds mid-layer lenses only to DROP rows (STOPPRED, ED-CHAIN, Rnd17#7) or to verify early
  (L1-09); tail re-drafting without target information (SD-semiAR); and taps fed to the head (S12, LAYERS). A batched
  shallow pass whose DECISION repairs the draft, with its layers reused by the check, is not in the records. ZERO's
  IDEAS_1117_channel idea 1 is the same pass used for taps.

### Idea 3 (speculative, the only partial shape the data leaves open): LATE ENTRY
- **Mechanism.** Since the verdict forms in layers ~37-48, run ONLY those layers on the drafted rows, batched, from an
  approximate layer-36 residual. The approximation would come from the head trained to emit it, or a ridge from the
  previous row's true taps plus the new token. The true late layers attend to the prefix's TRUE late-layer KV / GDN
  caches, which the check computed. Then the verdict is read at 48 (the exact mixer: phi 0.93 ceiling on the top-8).
- **Price [P].** 12/48 of a pass per round, ~5-7 ms (dense 1.9 + tail experts + re-draft; its inputs are approximate,
  so the check cannot reuse it). That is past the bar's edge, so it needs phi >= 0.9 at psi <= 0.1 %.
- **What kills it.** The layer-36 input must itself be carried through the draft's depth, so it drifts the way the
  head's state does (direction error 0.85-0.99 in the 2B sandbox). If the late layers' verdict is sensitive to that
  error, it is the head's problem again.
- **Screen.** It needs the GPU: run layers 37-48 on (a) the true layer-36 residual and (b) a ridge-predicted one, at the
  chain's first-miss rows of 2 windows, and compare the verdicts. Not a CPU screen. [G] low odds; listed because the
  floor points here and nowhere shallower.

## 7. What this changes
- Route 1 should be judged on DECISIONS (phi, psi, c) at the chain's rows, not on state similarity. A channel that
  improves the carried state but leaves the first-miss verdict wrong buys little: even perfect states (LAB) fix only
  2/3 of first misses.
- **Partial depth: closed on today's taps.** The verdict needs >= ~46 of 48 layers. Tonight's widened capture is the
  last check (Idea 1, a few minutes).
- **Partial width:** closed by Fable B's arithmetic (10-170x over the per-token budget).
- **Free sources:** closed by the records (ASSIM KILL, analogs <= 2.4 % coverage, fork re-anchor 13.77 -> 14.10).
- **What remains for 24-26 kept:** ~2 full-depth decisions per block. Either they ride the check as extra rows that
  carry the RIGHT alternatives (route 2: the head's margin flags 46 % (HE+) / 20 % (MBPP) of first misses at 2 %
  false; ORACLE-31 caps its own placement at 18.48 / 11.21), or the head needs fewer decisions (training).
- To the brief's "beyond the physical limit": the bytes per decision are not the wall (a batched row is ~0.3-0.4 ms
  marginal). The wall is that the 125B forms the decision in its last ~12 layers and only on the right prefix. So a
  decision is only cheap inside a full-depth batch whose rows already hold the right alternatives.

## 8. Reproducibility (all in ~/FableGod/lunatic/, read-only on the dumps and the bank; CPU)
- **floor_fix.py / .out:** OWN / LAB / PER-P / FIX-r on the topk cells; 1 s.
- **floor_lens.py / .out:** tuned lens per tap, locator AUC, decider, FIX-r with the real decider; ~45 s, ~2 GB.
  - floor_lens_taps_only.out is the earlier run without the final-layer controls (same numbers for taps 5-45).
  - Row convention, checked: a topk cell's `row` is its own verifying bank row (tk[row, 0] == y at every depth).
- **floor_bar.py / .out:** synthetic decider (phi, psi, c, r) on the HE+ / MBPP cells; 1 s.

## 9. TRIED lines (ready-made, for NewIdeas / MoonShots to record)
- FLOOR-STATE-DENSITY (Lunatic, Tue 17 Nov 07:4x): the true state injected every 2nd / 4th / 8th / 16th depth of a
  chain-32 recovers 78 / 49 / 25 / 11 % of the LAB gain (HE+ 19.20 / 17.18 / 15.51 / 14.52 vs own 13.77, LAB 20.77)
  and 68 / 36 / 13 / 2 % on MBPP | wall: drafter | line: 45 | exact: n/a | verdict: INFO (a state channel must be
  dense) | ~/FableGod/lunatic/floor_fix.out | [lunatic]
- FLOOR-DECISIONS: a perfect locator + the 125B's token at the chain's first miss, r = 1 / 2 / 3 rounds, no
  re-anchor: HE+ 19.98 / 23.65 / 26.34 (re-anchor 21.73 / 25.68 / 27.58) vs LAB 20.77; MBPP 11.50 / 15.39 / 18.59
  (13.29 / 18.85 / 22.86) vs 12.83. The LAB head fixes 68.5 / 65.4 % of first misses | verdict: INFO (the floor is
  1-3 located decisions per block, not states) | floor_fix.out, floor_lens.out | [lunatic]
- FLOOR-LATE-VERDICT: tuned lens (ridge to the final mixer, fitted on 60 rawtaps code windows) on the bank's
  4-stream-mean taps; chain first misses vs right rows. Lens-alone locator AUC after layers 6 / 16 / 26 / 36 / 46 =
  0.68 / 0.70 / 0.69 / 0.64 / 0.92 HE+, 0.65 / 0.65 / 0.68 / 0.67 / 0.92 MBPP; final-layer mean control 0.991 / 0.975;
  exact 0.999 / 0.997; head margin alone 0.954 / 0.896. As a repair decider, layers <= 36 lower kept (13.77 -> 10-13
  HE+); 46 gives 16.3-17.0 at 46/48 of a pass per round (372-447 tok/s at the bound vs 462) | wall: drafter / route 1
  | verdict: KILL for shallow stream-mean deciders; the widened state is untested (VERDICT-DEPTH screen) |
  floor_lens.out | [lunatic]
- DECIDER-BAR: chain-32 + a synthetic decider: beating tree-32 at the C6 + int8 bound (566 HE+) needs phi >= 0.7 at
  psi <= 0.1-0.5 % and <= 2-4 ms per round (r = 2), or phi >= 0.9 at <= 4 ms; nothing at 8 ms | verdict: FRAMING |
  floor_bar.out | [lunatic]
