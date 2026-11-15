# IDEAS_1115_nearties: problem #4, near-ties right after a check (Fable C, Sun 15 Nov 2026, 22:0x-22:4x, desk only)

Question (THINK_1115 #4): even the true-state head gets only 95.3 / 96.4 % at bands 1-2 / 3-6 because the 125B is ~50/50
there; the check already computes the 125B's full distribution at every verified row. Can that, or anything else, cover the
post-check near-ties exactly and cheaply?

Marks: [K] measured, [P] arithmetic on measured numbers, [G] judgement. Modelled gains are halved before they are quoted.
Data used tonight (two CPU scripts, `nearties_1115.py` and `nearties_1115b.py`, 20 s each, < 1 GB): the decision map's
36 teacher-forced windows (12 HumanEval code, 12 IFEval, 12 multi) with the 125B's top-8 log-probs per answer row
(`decision_map/out/baseline.json`) and the map's recorded DFlash2 HPD-0 walks (checks, drafted tokens). Caveat: that is the
OLD drafter, not t30b, and the map's chat windows are much harder for it than HE+ is for t30b; the brief's t30b numbers
are quoted where they exist. Near-tie = the 125B's top-1 minus top-2 log-prob < 2 nats (the brief's own cut).

## 0. What the check's distribution can and cannot do (audit of the question's premise)

1. It cannot resolve the post-check near-tie. The check's distribution at the miss row gives the correction (its argmax)
   and nothing more for exact greedy decoding: the second-best token at a verified row is, by construction, the token the
   125B did NOT choose. The near-ties that cost us sit at the rows AFTER the correction, whose prefixes the 125B has not
   seen yet. An exact cover of a near-tie at band b of the next check needs the 125B run on both candidates at that row,
   i.e. a fork (a tree row). So for the EXACT line, the only cheap covers are better placement of forks, or crossing the
   hard region faster; for the LOSSY line, forgiving the miss.
2. What the check's distribution DOES give, free: the exact margin at every verified row, including the miss row. That is
   a near-tie DETECTOR for the rows just walked, and (new tonight) near-ties cluster, so it predicts the rows ahead:
   - [K] code: P(near-tie at t+1 | near-tie at t) = 35 % vs 4.8 % after a clear row; at lags 2..6 still 31 31 29 24 26 %
     (base 6.9 % of all rows). Chat clusters weakly: IFEval 48 vs 39 %, multi 37 vs 33 %.
   - [K] keyed by the miss row that started the check (code, n 124 / 173): after a NEAR-TIE miss the next check's band-0
     row is a near-tie 35.5 % and a band-0 MISS 16.9 %; after a CLEAR miss 6.9 % and 2.9 %. A 5.8x split in the band-0
     miss rate from a number the check already has. IFEval 37.0 vs 28.9 % (n 925 / 433), multi 60.3 vs 48.8 % (n 842 / 987).
   - [K] the hard region persists: on code the near-tie rate at bands 0..6 after a check start is 15 14 17 15 13 16 16 %,
     flat at 2.2x the base. The deep bands (15-30) look easy (96.7-99.3 %) partly by survivorship: reaching depth 15 selects
     easy regions. So "97.8 % flat with the true state" is a lab number for the easy regions; bands 1-6 after a hard start are
     coin flips the head cannot learn, with or without the true state. Any lever aimed at the head alone (problem #3) has a
     ceiling near the true-state line at bands 0-6 (93.2 / 95.3 / 96.4); forks are the only exact route past it.
3. The miss row's margin also says WHICH KIND of miss it was: a coin flip the drafter lost (the 125B was ~50/50) or a drafter
   error (the 125B was sure). [K] code misses with margin < 2 nats: 41.9 % (old drafter); the brief: 92 % of t30b's
   check-start misses on HE+ have margin < 2 nats. Drafter errors are what problem #3's information levers fix; coin flips
   are not.

## Idea 1 (best): margin-keyed restart, the check's exact margins decide the next check's shape

- Mechanism. After each check, read the 125B margin at the miss row (and the last 2-3 kept rows), already computed.
  Hard start (miss-row margin < 2 nats, or any of the last 3 verified rows a near-tie): draft the next check wide and
  shallow: the chain plus 2-wide forks at bands 0-2 at every position where the head's p1 < 0.9, each fork's alternative
  with a 2-3 row continuation (a depth floor, so the K98 starvation does not thin it to one node). Easy start (clear miss,
  no near-tie in the last rows): a plain deep chain (or the tree's spine extended), no forks at bands 0-2.
  The budget rule: the fork rows come from the spine's tail, whose rows are worth the least after a hard start
  (P(reach depth 20 | hard start) is low), and go back to the spine after an easy start, where the tail is worth the most.
- Numbers. [K] the 5.8x band-0 split above (16.9 vs 2.9 % band-0 miss on code); bands 1-2 after a hard start are near-ties
  31-35 % of the time, after an easy start ~7 % [K, lag profile]. [K] the brief: tree-16 lifts band 0 from 92.0 to 98.6 %
  and bands 1-2 from 91.5 to 96.4 %, so forks DO cover near-ties; what EAGLE-2 lacks is a signal for where NOT to fork
  (66-86 % of K64's nodes sit off the #1 spine, placed by the head's own p, which is blind to the 125B's margins: the head's
  cf < 0.5 rows are 125B ties only 13.8 % of the time on code [K floor_rows], and the brief's p1 < 0.5 is right 40-44 %).
  [K] a sibling row costs 0.84-0.91 of a chain row in new experts (M-10), so the shape change is near cost-neutral.
  [G] kept/32 gain over plain tree-32: +2-6 % (easy starts get 8-12 spine rows back at ~0.3 kept per row; hard starts trade
  low-value tail rows for 3 forks x ~0.3 catch x (1 + a 2-row continuation)); quoted at half: +1-3 %. Small but free, and it
  compounds with problem #1's budget rule; the Monday sim decides.
- Tag: NEW as a predictor keyed on the check's observed margins. Nearest TRIED: CALIB-TREE (WEEK98) KILL recalibrated the
  head's edge probabilities by depth x walk-start (x0.996 / x0.988) and concluded "ORACLE-15's headroom is information the
  head lacks" (this adds information the head lacks); sb98's margin_RESULT (mhx_mg1k, a margin-trained head) NO EFFECT is a
  training-side idea, not a shape rule; WEEK98-TREE / TREE32 PASS are the shapes being keyed; K98 is the budget finding.
- Exact: yes (shape only).
- Kill: on t30b's HE+ dev walk the per-band table bucketed by the previous miss row's margin shows no split (bands 0-2 miss
  rates within 1.5x of each other), or the keyed shape in the K98 sim gains < +2 % kept/32 over tree-32 on both sets.
- Monday test (<= 15 min, CPU, MTP Head's tree_k98 sim on the heplus24 / mbppbank cells, which hold the 125B's top-2
  log-probs since the "92 %" fact came from them): (a) add one key, the previous miss row's margin < 2 nats, and print the
  per-band table per bucket plus the bucket shares; (b) run tree-32 vs the keyed shape (chain-32 after an easy start,
  tree-32 with the bands 0-2 forks and 2-row continuations after a hard start), kept/32 on both sets. PASS: (a) a >= 2x
  split in the band-0..2 miss rate AND (b) >= +2 % kept/32 on both sets. If (a) passes and (b) fails, hand the key to #1.

## Idea 2: a near-tie predictor trained on the check's free labels (the learned form of idea 1)

- Mechanism. Every check yields 16-32 labelled rows for free: (the head's p1 at the row, the margins at the last verified
  rows, the band, a same-window bigram memo flag) -> was the row a 125B near-tie. A tiny logistic model fits online
  (~3,000 labels per second at 100 checks/s) and scores the candidate positions at bands 0-2 of the next check; forks go
  to the top-scored positions, EAGLE-2 keeps the rest. The memo: key = the last two tokens; value = whether the row after
  them was a near-tie the last time this pair occurred in the request.
- Numbers. [K] the three signals on code, each alone: the previous row's margin (35.5 vs 6.9 % at band 0); the head's cf
  (cf < 0.5: 13.8 % ties vs 0.0 % at cf >= 0.99, n 152 / 2211, floor_rows); the bigram memo (precision 49 %, recall 15 %,
  flags 2.1 % of rows at 7x the base rate; on chat precision 52-61 % but recall 5-7 %, so chat gets little from it).
  [G] combined at bands 0-2 on code: ~50 % precision at ~50 % recall; the gain is bounded by idea 1's and only beats the
  2-nat rule if the signals add (the Monday AUC says).
- Tag: NEW in the features (observed 125B margins, memo); CALIB-TREE KILL is the warning that calibrating the head's own
  signal did nothing, so this lives or dies by the added signals. M-05 SUFX KILL used a suffix datastore to DRAFT tokens;
  this uses a memo to predict difficulty, a different use.
- Exact: yes.
- Kill: AUC for near-ties at bands 0-2 on the held-out half of HE+ dev < 0.75, or precision at 30 % recall < 40 %.
- Monday test (<= 15 min, CPU): fit on half of the HE+ dev walk cells, test on the other half; print AUC and precision at
  10 / 30 / 50 % recall for each signal alone and combined; a held-out split, never in-sample (see the fitted-bases rule).

## Idea 3 (the LOSSY 900 line): near-tie forgiveness that keeps the rest of the check alive

- Mechanism. At a miss row whose 125B margin < eps nats AND whose wrong draft is the 125B's #2, accept the draft. The check
  rows after it were computed with that draft as input, so their distributions remain valid for the accepted path and the
  check simply continues (a cascade), no restart, no new rows. The exact margin from the check is the gate (eps in nats),
  not the drafter's p (R-61N's rho).
- Numbers. [K] forgivable misses (margin < 1 nat AND draft = 125B #2), old drafter: code 14 % of misses (29.5 % x 48 %),
  IFEval 12.8 % (42.6 x 30), multi 5.3 % (26.7 x 20); at eps 0.5: 6.7 / 5.4 / 2.6 %; at eps 0.25: 3.0 / 1.9 / 1.3 %.
  t30b's #2 should hold more of its misses (p1 < 0.5 is right 40-44 %, so the #2 carries most of the rest), so the code
  share is likely higher than 14 %. [P] each forgiven miss turns a restart into a continuation worth the remaining chain
  (~8-10 kept at a mid-check miss): +1.1-1.4 kept per check at 14 % of misses, ~+8-10 % tokens per check on code; quoted at
  half: +4-5 % at eps 1, +2 % at eps 0.5. On the 900 lossy budget that is the same percentage off the needed cycle time.
  [K] evidence the swap is benign at small eps: WEEK98-TREE's batch shape alone flipped 142 / 264 answers, all at near-ties
  (top-2 gap <= 0.375 logits), with identical pass@1.
- Tag: TRIED, BETWEEN: M-15 EXACT-NT (exact near-tie acceptance via a speculative cascade: chat +14.0 %) and R-61N CHAT-NT1
  (non-exact, p_d >= rho x p_top1: chat +8..+23 %); B12-R KILL (+10.5 % prose at 12 GB); WEEK98-H comment relaxation
  BETWEEN. New here: the gate is the check's exact margin, the already-computed rows stay alive, and the 900-lossy target
  did not exist when M-15 was priced. Published: Sharma 2606.30265 (relaxed acceptance at near-ties, theory only).
- Exact: NO (the output is a sample within the 125B's top-2 at near-ties); EvalSuite's gate is the line.
- Kill: HumanEval+ drift at eps 0.5-1.0 (the RELAX runs lost 6-7 points on worse levers), or the forgivable share on t30b
  code < 10 % of misses (then +3 % is not worth a lossy mode).
- Monday test (<= 15 min, CPU): replay the HE+ dev walk cells (or the map's 36 windows) with eps in {0.25, 0.5, 1.0}:
  forgiven misses, kept/32 gain, and the count of answer rows that change vs the exact walk. EvalSuite's HE+ gate later,
  on a real run, not Monday.

## Idea 4: check length keyed by the same margin (16 rows after a hard start on chat, 32 after an easy one)

- Mechanism. After a hard start the chain rarely gets deep, so the tail of a 32-row check is dead weight; after an easy
  start it is where the kept comes from. Pick K from the miss row's margin (and domain): chat hard start -> 16 rows, else
  32. Nothing else changes.
- Numbers. [K] the exact floor is 38.8 ms at 16 rows vs 52.0 at 32 (x0.75); one-pass R(32) = 1.45. [K] old drafter, chat:
  band-0 miss after a hard start 37-60 %, so the expected kept of that check is a few rows either way; code hard starts
  are rarer (42 % of misses) and keep more, so code probably stays at 32. [G] chat tok/s +5-10 % at equal kept, halved
  +3-5 %; code ~0.
- Tag: published-adjacent (dynamic draft length by confidence, e.g. SpecDec++; the 24 GB era's "adaptive width beat md7",
  TRIED L878 BETWEEN); the margin key is the twist.
- Exact: yes.
- Kill: in the K98 sim with R(16) = 1 / R(32) = 1.45, tok/s of the keyed K is not above the best fixed K on chat-like
  data (IFEval cells) by >= 3 %.
- Monday test (<= 15 min, CPU): the sim with K chosen by the previous miss row's margin, tok/s under the one-pass cost
  model, on both sets; chat needs an IFEval cell set, which may not exist in sb98 (then use the map's IFEval windows).

## Killed on the desk tonight (for the record, so nobody re-opens them)

- Recycling the old drafts after the correction (M-06 RECYCLE's spirit): [K] the drafter's already-drafted d(k+2) equals the
  true token after the correction only 18.5 % of the time on code (15.2 IFEval, 11.6 multi), vs 91.2 / 65.9 / 46.0 % for the
  fresh check's own first draft; recycled-OR-fresh adds only +0.6 / +1.5 / +1.5 points. Dead as a substitute, near-useless as a
  fork alternative.
- A 1-row "resolve" pass after each check (run the 125B on the correction alone to verify band 0 exactly): [P] ~1/7 of a
  16-row check's bytes (~1.2 GB at 10 experts x 48 layers) is >= 6 ms at the bandwidth floor plus the engine's fixed
  overhead, for at most +1 kept per check; a 33rd check row does the same for ~0.4 % of the check. Dead.
- "The check's distribution resolves the tie": no, see section 0. It locates ties; forks resolve them.

## Audit notes for the other thinkers

- #1 (where to branch): the "signal available before the check" it asks for exists and is free: the previous check's exact
  margins (section 0, item 2). It splits the band-0 miss rate 5.8x on code. Please key the budget rule on it.
- #3 (information for the drafter): the true-state ceiling at bands 0-6 (93.2 / 95.3 / 96.4) is set by clustered 125B coin
  flips after a hard start, not by state drift; information levers cannot pass it, forks or forgiveness can. The 97.8-flat
  number holds for the easy regions the deep bands select.
- #2 (cheap big trees): after an EASY start the extra rows should be spine rows, which share the chain's experts best; after
  a HARD start they are siblings at bands 0-2 (0.84-0.91 of a chain row in new experts, M-10). The margin key tells which.

Scripts and outputs: `nearties_1115.py`, `nearties_1115b.py` (this folder; outputs quoted above, re-run in 20 s each).
