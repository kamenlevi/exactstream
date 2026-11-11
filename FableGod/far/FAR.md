# FAR: making the far-ahead draft pay (Fable FAR, Wed 2026-11-04 21:4x -> 23:30)

Mission: ~/FableGod/far/MISSION_FAR.md (Kamen approved ~21:3x). Marks: [K] known from records, [P] priced / derived, [G] guess.
Every number names its source. Scripts and raw outputs sit next to this file (`far_ceiling.py/.txt`, `far_state.py/.txt`,
`far_ridge.py/.txt`, `far_frame.py`); all CPU, one core, < 3 GB, 12 s / 252 s / 111 s, under share locks, no GPU, no edits
outside ~/FableGod/far/.

## Verdict in six lines
1. **The token path cannot reach M-38's point.** With the TRUE future tokens and their EXACT routes, the "reused within 8
   checks" forecast reaches recall 0.68 (K 266) / 0.76 (K 147), not 0.92: a third of the cache reuse is in the drafter's
   REJECTED rows, which no true-path forecast names [K 1b]. A cheap token -> expert map loses another half (deep routes are
   ~60 % context): honest maps recall 0.5 / precision 0.66, -6 % misses [K 1b].
2. **States do better than tokens, but not enough on this data:** a learned state -> route map on an oracle shallow state of
   every future row (rejected rows included) gives recall 0.70 / precision 0.64 and **-14.6 % misses at 45 GB** (-11.9 % at
   24 GB), vs the learned eviction key -7.8 % and M-38's PASS point -26.5 % [K 1e]. BETWEEN.
3. **Equal recall / precision buys far fewer misses than M-38's noise model:** real forecasters repeat their errors, the model's
   are independent per use. The PASS point should be read as recall on the uses LRU would miss (M-38's r_m), not uniform.
4. **Misses are not stalls:** perfect H 8 foresight cuts misses 34 % but layers-with-a-miss only 5 % (42.9 -> 40.6 of 48). If
   the EXACT SSD wait is per-layer round trips, foresight buys bytes (24 / 12 GB lines), not tok/s [K 1b, P 4].
5. **Best forecaster (section 2): FAR-SET** = the MTP head chain run ~12 steps per check in the EXACT check's SSD-wait shadow
   (rolling ~100 tokens ahead) + a distilled per-layer pseudo-router on its hidden states (top-16 per far token, unioned per
   future check) + the memo of exact routes for repeated tokens + M-38's FH eviction. Extra compute ~0.3 ms per check for the
   map; the far draft (~26 ms GPU per check) must hide in the ~55-60 ms SSD wait, or it kills the gain [P 4].
6. **The GPU test (section 3, ~3-4 min) is a capture, not an A/B:** the far chain's tokens + states + the 125B's routes of those
   tokens on cap45's six prompts; the CPU replay then gives the far-draft ceiling (perfect map) and the r_m line. PASS >= 20 %
   fewer misses than LRU at K 266 with recall >= 0.80 on the far draft's own text; KILL < 10 % or not above the learned key.
7. **M-14's learned key among the unforecast residents stacks with a precise forecast, not with a noisy one** [K 1g]: perfect
   kept-row routes + the key = -26.9 % (= M-38's PASS point from the true path alone), perfect all rows + key -37.2 %; the
   ridge stand-in gains nothing (-14.6 -> -15.0 %) because its false positives sit on the old residents (precision 0.57 on
   positives last used > 2 checks back, 0.79 on recent ones). The map's precision on OLD residents is the number to train for.

## 0. Frame (what "recall / precision" means here) [K]
- Trace: BigMoE's cap45 (`~/BigMoE/data/cap45/*_gate_0.npz`, the exact REF2 loop at C 160, real DFlash2 16-row checks, 6
  prompts, 176 live checks). Per check and layer: the 16 rows' top-10 expert ids, the rows' MoE inputs (bf16), the kept count;
  `tokens` = the committed tokens of the prompt (sum of kept). `routers.npz` = the 47 routers (fp16).
- Replay: MoonShots' M-38 (`~/MoonShots/scripts/foresight45.py`, commit 99df774), chained, misses counted on prompts 2-6,
  K = 266 (45 GB EXACT) / 147 (24 GB) resident MSB-3 slots per layer. A check NEEDS the union of its 16 rows' experts per layer.
- The forecast question (M-38's PASS point): at each eviction decision, for each resident not needed now, "is it used again
  within the next 8 checks (within this prompt)?" M-38's r 0.7 forecast: recall 0.92 / precision 0.72 on FH's trajectory at
  K 266 -> 31.5 misses per kept token = -26.5 % vs LRU 42.9 (learned key -7.8 %, OPT -55 %). Base rate 0.46.
  Recency (used in the last 8 checks): recall 0.83 / precision 0.52.
- My scripts reproduce M-38's LRU 42.9 / OPT 19.2 / perfect H 8 28.4 (K 266) and 123.1 / 69.5 / 73.6 (K 147) exactly.

## 1. The ceiling with the TRUE future tokens (CPU): `far_ceiling.py` -> `far_ceiling.txt` (21:44, 12 s, one core)
Row -> token alignment: row j of a check carries `tokens[off + j]` for j < kept (shift 0 gives the highest layer-0
same-token Jaccard on all six prompts, 0.54-0.66, vs <= 0.38 for shifts -1 / +1 / +2; the commit entry at layer 1 holds the
kept rows). Rows j >= kept are REJECTED drafts whose tokens the capture does not hold (a perfect far draft would not have
produced them either; the check still loads their experts).

**1a. How much of a route is the token?** [K far_ceiling.txt] Per-row recall@10 of a kept row's true top-10 from the same
token's previous occurrence in the prompt (644 pairs) / from a leave-one-prompt-out token table's top-10 (663 rows):
layer 0: 0.75 / 0.68; layer 1: 0.50 / 0.44; layers 2-7: 0.48-0.50 / 0.36-0.39; layers 8-47: 0.40-0.41 / 0.24-0.28.
-> Below layer 8 the token identity carries only ~40 % of a route; the rest is context.

**1b. Set-level forecasts at H 8, K 266 (45 GB EXACT), M-38's FH policy and measurement** [K far_ceiling.txt]
(recall / precision of "resident used again within 8 checks" at the eviction decisions; misses per kept token, prompts 2-6;
LRU 42.9, OPT 19.2; stalled layers per check = layers with >= 1 miss, LRU 42.9 of 48):

| forecast of the next 8 checks | recall | precision | misses | vs LRU | stalled layers |
|---|---|---|---|---|---|
| PERFECT-ALL (every row's true experts; M-38's perfect H 8) | 1.00 | 1.00 | 28.4 | -33.7 % | 40.6 (-5.4 %) |
| **PERFECT-KEPT: the kept rows' true experts (= a perfect far draft through a PERFECT token -> route map)** | **0.68** | 1.00 | 34.2 | -20.3 % | 42.1 (-1.7 %) |
| ALL-20: token table from all 6 prompts, top-20 per token (has seen the held-out prompt) | 0.73 | 0.69 | 35.3 | -17.8 % | 42.1 |
| ALL-10 | 0.53 | 0.80 | 37.2 | -13.3 % | 42.4 |
| MEMO-4 + ALL-10 (online memo of this prompt's routes, unseen tokens -> ALL-10) | 0.63 | 0.71 | 37.7 | -12.1 % | 42.5 |
| MEMO-4 + LOPO-10 + popularity (honest: other prompts only) | 0.53 | 0.66 | 40.3 | -6.0 % | 42.8 |
| MEMO-2 only (this prompt's own history) | 0.38 | 0.71 | 41.8 | -2.6 % | 42.8 |
| LOPO-20 + pop (honest static table) | 0.51 | 0.59 | 40.7 | -5.2 % | 42.8 |
| popularity top-20 only | 0.06 | 0.71 | 42.8 | -0.3 % | 42.8 |
| (M-38's r 0.7 noise model, for scale) | 0.92 | 0.72 | 31.5 | -26.5 % | n/a |

K 147 (24 GB; LRU 123.1 = 266 MB per kept token, OPT 69.5): PERFECT-ALL recall 1.00 -> 73.6 (-40.2 %); PERFECT-KEPT 0.76 /
1.00 -> 90.7 (-26.3 %, 196 MB); ALL-20 0.83 / 0.73 -> 103.3 (-16.1 %); MEMO-4 + ALL-10 0.75 / 0.74 -> 107.5 (-12.7 %, 232 MB);
MEMO-4 + LOPO-10 0.66 / 0.69 -> 116.0 (-5.8 %).

**1c. Reading.**
1. **The ceiling of the token path is recall 0.68 at K 266 (0.76 at K 147), not 0.92.** Even a far draft that produced the
   TRUE tokens and a map that returned their EXACT routes misses 32 % of the reuse: those residents are used only by the
   REJECTED draft rows (j >= kept) of the next 8 checks. Math checks keep 5-8 of 16 rows, so half the rows a check loads are
   the drafter's wrong guesses, and they are a third of the cache reuse. A forecaster that follows only the true path cannot
   pass M-38's point; it must also cover "plausible wrong continuations".
2. **A token -> expert map loses another half.** Deep-layer routes are ~60 % context: the honest maps (this prompt's memo plus
   a table from other prompts) reach recall 0.5 / precision 0.66 and cut misses 6 % (the learned eviction key: 7.8 %). Even
   a table that has seen the prompt (ALL-20) stays at 0.73 / 0.69, -17.8 %. The cheapest map is not enough; the forecast
   must come from STATES (the far drafter's hidden states, or the 125B's own shallow layers), not token ids.
3. **Misses are not stalls.** Perfect foresight at H 8 cuts misses 34 % but stalled layers only 5 % (42.9 -> 40.6 of 48;
   Belady 34.6). If the EXACT check's SSD wait is per-layer round trips (GOAL.md's FAST reading: ~1.4 misses per layer),
   foresight buys bytes, not time; if it is bandwidth-bound (EXACT: ~450 misses x 2.16 MB ~ 1 GB per check at C 160 in this
   replay), the miss cut maps ~1:1 onto the ~55-60 ms wait. The GPU test logs both (section 3).
4. What must carry the rest: (a) the drafter's own continuation as the forecast source (its wrong rows are "plausible
   text", and so are the next checks' rejected rows), (b) state -> route maps per layer (1d / 1e test the state signal
   with the 125B's own shallow state as an oracle stand-in), (c) the memo for repeated tokens (exact routes already computed).

**1d. The state signal, nearest-neighbour form (oracle stand-in, CPU): `far_state.py` -> `far_state.txt` (21:48, 252 s)**
Forecast = each future row's nearest earlier row of the prompt (cosine on the 125B's MoE input at layer s; the future
rows' x_s is an oracle here), that row's true routes at every layer, m neighbours' union; all 16 rows (so the rejected
rows are forecast too) or the kept rows only. [K far_state.txt]
- Per-row recall@10 of deep routes from the nearest earlier row: s 1: L1 0.61, L2-7 0.59-0.60, L8-47 0.51-0.54 (the
  token memo: 0.40-0.50). More context than the token, but half the route is still new at every position.
- Set level, K 266: KNN-1 m2 all 16 rows recall 0.68 / precision 0.69 -> -7.1 % misses; m4 0.73 / 0.65 -> -7.4 %; kept
  rows only 0.55 / 0.73 -> -4.8 %. s 0 / 3 / 8 the same within 1 point. K 147: m2 all rows 0.77 / 0.72 -> -12.2 % (234 MB).
-> Within-prompt similarity (memo or neighbour) tops out near recall 0.7 / precision 0.7 and -7 % at 45 GB. Note the gap to
M-38's noise model at the same recall / precision (-26.5 %): M-38's errors are independent per use, so a resident used
three times is forecast with 0.97; a real forecaster's errors repeat (a route it gets wrong, it gets wrong every time), so
equal set-level recall buys far fewer misses. The PASS point must be read as "recall 0.92 on the uses LRU would miss".

**1e. The state signal, learned form (oracle stand-in, CPU): `far_ridge.py` -> `far_ridge.txt` (21:54, 111 s)**
Ridge from a row's MoE input at layer s (2560 + bias) to its top-10 membership at all 48 layers (the SP-MoE / DraftExpert /
LEAD1CORR shape, adapted), fit leave-one-prompt-out on the other five prompts' rows (2,816 rows; alpha = lambda / mean
squared row norm); forecast per check = the union of its rows' predicted top-k (all 16 rows = the rejected rows forecast
from their own oracle state; or the kept rows only). [K far_ridge.txt]
- Held-out per-row recall@10: s 1, alpha 0.3: L0 0.67, L1 0.61, L2-7 0.47-0.48, L8-47 0.34-0.37 (five prompts of data
  cannot fit 2560 -> 512 maps for deep layers; LEAD1CORR's lead-1 map reached 0.73 with the same data because the
  residual form helps at lead 1 only).
- Set level, K 266: **RIDGE-1 top-16, all 16 rows: recall 0.70 / precision 0.64 -> 36.6 misses = -14.6 % vs LRU** (stalls
  42.3, -1.4 %); top-10 0.57 / 0.71 -> -12.7 %; kept rows only -10.8 %. s 3 the same within 0.5 points.
  K 147: top-10 all rows 0.68 / 0.73 -> 108.7 = -11.7 % (235 MB); top-16 0.81 / 0.68 -> -11.0 %.
- Why the ridge beats the KNN at equal recall / precision (-14.6 vs -7.1 %): its false positives are popularity-shaped
  (regression to the mean names the busy experts, which LRU would have kept anyway) and its misses fall on the rare
  experts, while the KNN copies one earlier row's exact set, right or wrong.

**1f. Answer to the brief's question 1.** With the true future tokens through the cheapest token -> expert map: recall 0.53 /
precision 0.66 at H 8 (honest), 0.73 / 0.69 with a table that has seen the prompt; the ceiling of ANY map of the true tokens is
0.68 / 1.00. The rest must come from (i) the drafter's own wrong continuations, which carry a third of the reuse, and (ii)
hidden states mapped to routes per layer (a trained pseudo-router; on this data's oracle stand-in 0.70 / 0.64 -> -14.6 %).

**1g. Foresight + M-14's learned key among the unforecast residents: `far_combo.py` -> `far_combo.txt` (22:00, 30 s)**
FH keeps the forecast residents; the rest leave by M-14's learned key (leave-one-prompt-out, Belady labels on LRU states,
re-fit here: LRU 42.9 -> 39.5, -7.8 %, = M-38) instead of LRU. [K far_combo.txt]
| forecast | FH + LRU | FH + learned key | recall / precision on positives last used > 2 checks back (60 % of positives at K 266) |
|---|---|---|---|
| PERFECT-ALL | -33.7 % | **-37.2 %** | 1.00 / 1.00 |
| PERFECT-KEPT (true path, exact routes) | -20.3 % | **-26.9 %** | 0.66 / 1.00 |
| RIDGE-1 top-16, all rows (the state stand-in) | -14.6 % | -15.0 % | 0.68 / 0.57 (recent residents: 0.74 / 0.79) |
K 147: PERFECT-KEPT -26.3 -> -29.5 %; RIDGE -11.0 -> -10.8 % (old residents 0.88 / 0.57).
-> A precise forecast and the learned key are additive (+3.5 to +6.6 points); a forecast whose false positives fall on the
old residents (the ridge's: precision 0.57 there) gets nothing from the key, because the key's job is exactly to rank the
old residents and the forecast has already protected the wrong ones. The pseudo-router must be trained and judged on its
precision over residents last used > 2 checks back, not on uniform recall.

## 2. Best forecaster: FAR-SET (mechanism, features, cost, expected numbers)
**Mechanism (per check, rolling; EXACT-safe: it changes only residency and read order).**
1. **The far draft.** The MTP head chain (TrainLab's autoregressive head, the one that runs K 15 today [K REPORT D-23c'])
   self-feeds ~12 steps per check from the newest anchor, keeping a rolling window of ~100 far tokens (>= 8 checks at 10.5
   kept on code, ~16 at 6 kept on math); after each check the window is re-aligned to the committed tokens (the stale far
   tokens past the first rejection are kept as "plausible text", the next 12 are re-drafted from the corrected anchor).
   DFlash2 cannot be the far drafter: its inputs are the 125B's taps of every context token [K ANE_DRAFTER sec 1].
   Cost: 2.2 ms per step [K HEADBENCH], 1.47 ms of it the 248k lm_head; 12 steps = ~26 ms of GPU per check, to run INSIDE
   the EXACT check's SSD wait (~55-60 ms per check with the GPU idle [P GOAL.md 36-37]); a 2-bit draft copy of the head
   cuts it to ~18 ms [P HEADBENCH]. Where the time hides = that wait; if the wait is not idle GPU time, the far draft is
   on the critical path and costs 20 % of the check (section 4).
2. **The route map.** Per layer L a pseudo-router R_L: h_far (the head's hidden state of a far token, 2560) -> 512 scores,
   rank-256 (2560 x 256 + 256 x 512 per layer = 0.79 M params; 48 layers = 38 M, 76 MB bf16), trained offline by router
   agreement (DraftExpert's recipe: the 125B's true top-10 of the far chain's own tokens as targets; data from the section 3
   capture, then nightly captures). Forecast per far token = its top-16 per layer (the set-level sweet spot in 1e: top-16
   beats top-10 at K 266, top-10 at K 147); per future check = the union over the ~10 far tokens assigned to it by position.
   Cost: 12 tokens x 48 layers x 0.79 M MAC = 0.9 GFLOP ~ 0.3 ms GPU per check [P]; the unions are 48 x 512-bit masks.
3. **The memo.** Every row the engine has checked has an exact route per layer; a far token that repeats a token seen in this
   prompt takes the union of its last 2 occurrences' routes (recall 0.38-0.49 / precision 0.68-0.71 on its own [K 1b]) in
   addition to the pseudo-router's set. Cost: a hash lookup per (token, layer).
4. **Eviction and read order.** M-38's FH key: residents forecast within 8 checks stay, the farthest forecast leaves first;
   among the rest M-14's learned key (it adds +3.5-6.6 points to a precise forecast, 1g) once the map's precision on old
   residents is > ~0.7, LRU until then. Read order: the next check's forecast misses are issued first; no extra SSD reads
   (prefetch of unforecast experts is R-57b's KILL territory).
**Features, in order of measured value:** the far rows' states (1e: 0.70 / 0.64, -14.6 %), the kept rows' exact routes
(1b: the ceiling 0.68 / 1.00, -20.3 %, only with a perfect map), the memo (1b: +3-4 points of recall over a static table),
recency (M-38: 0.83 / 0.52 alone). Popularity adds nothing (1b: 0.06 recall).
**Expected on the GPU [P/G]:** the CPU stand-ins put a trained map on oracle states at -14.6 % misses (45 GB) / -11.9 %
(24 GB); a map trained on 100x the rows should pass 0.8 / 0.7 per set [G, DraftExpert 86-88 % per use]; the far draft's
drift is the unknown that the section 3 capture measures. My prediction for the section 3 line: BETWEEN (12-18 % fewer
misses at K 266 with a perfect map of the far draft's own text), PASS 25 %, KILL 20 %.
**Verdict on the CPU evidence: the token form is KILL; the state form is BETWEEN** (above the learned key's 7.8 %, below
M-38's 26.5 % PASS point) and worth the GPU capture because it is a 4-minute run that also produces the training data.

## 3. The GPU test: FAR-G1 (<= 10 min; the orchestrator routes it; BigMoE's harness)
**What runs.** BigMoE's cap45 capture (`scripts/r57_screen.py` gate mode: the exact REF2 loop at C 160, 6 prompts, 256 tokens
each, ~98 s, footprint 28-33 GB [K cap45_run.txt]) with two additions per check: (a) the MTP head chain run F = 64 steps from
the check's anchor (teacher-forced start = the trunk's hidden at the anchor), recording the 64 far tokens and the head's
hidden state per step (64 x 2560 bf16 = 330 KB per check); (b) the 125B's routers on the far draft's text: one prefill-style
pass of the 64 far tokens appended to the context (every layer's top-10 per far row, as cap45 records rows), discarded after
the capture (no state change: run on a KV copy or recorded before the check commits). Cost: (a) 64 x 2.2 ms = 0.14 s per check,
(b) ~0.3 s per check [P]: 176 checks -> ~80 s on top of 98 s; ~3-4 min with loading, MEM as cap45 (33 GB), one share lock.
Also log per check: SSD wait ms, misses, stalled layers (layers with >= 1 miss), the far chain's exposed ms.
**Read-out (CPU, my frame, < 1 min):** (1) far-draft token agreement with the true tokens by distance 1-64; (2) the far
draft's own text through the PERFECT map (its 125B routes) assigned to the next 8 checks by position, in M-38's FH replay at
K 266 / 147: set-level recall / precision, r_m (recall on the uses LRU misses), misses and stalled layers vs LRU; (3) the same
with a ridge pseudo-router fit leave-one-prompt-out on the captured (h_far, routes) pairs; (4) the regression of SSD wait on
misses and on stalled layers (latency- vs bandwidth-bound), which decides whether the miss cut is tok/s or bytes.
**Lines (written now).** On (2), K 266: PASS = >= 20 % fewer misses than LRU AND set recall >= 0.80 (the far draft's text covers
the rejected rows' experts) -> build FAR-SET in BigMoE's engine (the rolling chain in the SSD-wait shadow, the pseudo-router
trained on this capture). BETWEEN = 10-20 % -> a bytes lever only (24 / 12 GB, BYTES10), no 45 GB work. KILL = < 10 % or not
above M-14's learned key (7.8 %) -> the far-ahead draft dies as a cache forecaster; record and stop.
Secondary line on (4): if SSD wait tracks stalled layers and not misses (R^2 difference > 0.3), the 45 GB tok/s claim is
dropped even on PASS, and FAR-SET is filed under bytes.
**What it does not test:** the engine-side hiding of the far chain (a later A/B), the pseudo-router at scale, and the memo's
live cost.

> (a quote from the project owner, kept private)
The main verdict stays the PASS / KILL above (H 8, F >= 96). The read-out is extended, on the same capture, to:
1. **Grid.** Far horizon F in {16, 32, 48, 64, 96, 128, 160, 192} tokens x eviction look-ahead H in {4, 8, 16} checks, at
   K 266 and 147. At each decision the forecast = the far draft's first F tokens' routes (the perfect map first, the ridge
   pseudo-router second), assigned to the coming checks by position (cumulative kept), cut at H checks; scored as recall /
   precision on "resident used again within H checks" split by layer band (0 / 1-7 / 8-47, the bands of 1a) and by domain
   (code = HumanEval 68 / 91, math = the three MATH500 + gsm8k; cap45 has no chat prompt: chat needs its own capture), plus
   the miss cut and stalled layers vs LRU from the full replay. 24 replays per K, ~5 s each = ~4 min CPU, one share lock.
   Decisions whose origin is too stale for a given F (origin horizon left < F) are scored at the shorter view and counted;
   the origin decisions alone give the clean F curve. Ceiling expectation from 7a [K]: on the true path the gain saturates
   at F ~ H x kept (code 8 x 10.5 = 84-96, math 8 x 6 = 48-64); F beyond that pays only with H 16. The drift of the real
   chain sets the other end: the sweep finds where precision on old residents (1g) falls below ~0.6 per band and domain.
2. **Adaptive rules, scored on the same recorded chains (zero extra GPU):**
   (a) SPICE-style per-origin horizon (published, adapted): the chain runs while its running confidence (the product, or an
   EMA, of its per-step top-1 probability) stays above tau; emulated by cutting the recorded 192-step chain at the first
   step below tau, tau in {0.5, 0.3, 0.1}; the capture must record p_top1 per far step.
   (b) Acceptance-rate horizon (free proxy): F_o = c x the recent kept per check (c in {6, 8, 12}): code gets ~90-130,
   math ~50-70 tokens automatically.
   (c) Per-layer horizon (possibly new): F_L per band = the largest F at which that band's precision on old residents
   stays >= 0.6 (expected [G]: the full F in layers 0-7, where routes are token-driven, shorter in 8-47, where the memo
   and recency carry more); applied as a per-layer cut of the same forecast, no extra compute.
> (a quote from the project owner, kept private)
   chain's cumulative confidence c_s = prod_{u <= s} p_top1(u) (variants: tempered c_s^(1/T), T in {2, 4}; a floor
   max(c_s, 0.05) so the tail never vanishes; an EMA of p_top1 instead of the product); a resident's score at a decision =
   sum over the far tokens whose predicted top-k names it of c_s x [token assigned within H checks]; lowest score leaves
   first, LRU / learned key as tiebreak (the SCORE key of section 8 with c_s in place of gamma^(d - 1)); the near tier of
   section 8 sits on top (the first two blocks outrank). Re-decided at every origin for free, since the chain is re-drafted
   there, so the effective horizon follows the text (a confident code block keeps 150+ tokens alive, a wavering chat
   answer fades out after 20-30). Scored on the same capture at zero GPU (p_top1 is recorded per step), against the best
   fixed F and the tau rules in (a).
   Line: an adaptive rule replaces the fixed F only if it beats the best fixed F at equal mean horizon by >= 1 point of miss
   cut at K 266 (and never loses on the other K); (d) is judged by the same rule against both the best fixed F and the
   best tau.
3. **Capture spec, settled with BigMoE (22:3x-22:4x; BigMoE builds it Thursday, records it in its HANDOVER):** the 6 cap45
   prompts generate 448 tokens (256 + 192) in the exact loop at C 160; origins every 8th check (~38, ~22 with the full 192
   horizon), F = 192 far steps of the MTP head chain per origin (far tokens + head hidden recorded, ~1 MB per origin) and
   one prefill-style 125B pass over the 192 far rows (top-10 per layer per row; ~3-5 s and ~35-40 GB SSD per origin, the
   pools flushed after it: the live misses right after an origin are not LRU's, the CPU replay is unaffected); ~6-7 min,
   MEM ~33-35 GB. Per far step also: p_top1 and the top-2 logit margin (SPICE rule), the head's top-4 ids (a later
   alternatives-union test of the rejected-rows gap). Optional: the far rows' MoE input at layers 1 and 3 (~2 MB per origin)
   for a pseudo-router on the 125B's shallow state as a comparison. **The PASS / KILL verdict is read at F 96** (the
   FAR-SET design point; 7a: the ceiling saturates at F 96 for H 8) with 64 and 128 as sensitivity; 38 origins are enough
   for the K 266 line (it is a full replay over every decision, each using its last origin; 7a: S 8 needs F >= 128, and
   F 192 covers it). 12 prompts are not needed for the line; the code domain rests on 2 prompts and chat is absent, so the
   per-domain split is code vs math tonight; chat needs its own capture later if the sweep matters.

## 4. Arithmetic [P on K]
EXACT at 45 GB today: 126 ms per check at 10.5 kept = 83 tok/s; SSD wait ~55-60 ms of it [P GOAL.md]. Miss cuts from this
replay (K 266, C 160; the live default is C 256, so these are relative):
| forecast | misses vs LRU | stalled layers vs LRU | if the wait is bandwidth-bound (1:1 on ~58 ms) | if latency-bound (on stalls) |
|---|---|---|---|---|
| FAR-SET on the CPU stand-in (ridge top-16, all rows) | -14.6 % | -1.4 % | -8.5 ms -> 117.5 ms = 89 tok/s (+7 %) | -0.8 ms (+0.7 %) |
| M-38's PASS point (r 0.7) | -26.5 % | n/a | -15.4 ms -> 110.6 ms = 95 tok/s (+14 %) | ~-3 ms (+2.5 %) [G] |
| perfect H 8 | -33.7 % | -5.4 % | -19.5 ms -> 106.5 ms = 99 tok/s (+18 %) | -3.1 ms (+2.5 %) |
| Belady (all future) | -55.3 % | -19.3 % | -32 ms -> 94 ms = 112 tok/s (+34 %) | -11 ms (+9.5 %) |
[22:2x, section 7b: the per-check logs say bandwidth-bound at the margin (0.15-0.17 ms per miss = 12.7-14.4 GB/s, R2 >=
0.95) above a ~10-16 ms per-check latency floor, so the left column is the expectation and the right one the floor.]
Minus the far draft if it is NOT hidden: 12 steps x 2.2 ms = 26 ms per check on the critical path = -17 % tok/s, which
erases every row above. So the 45 GB case stands or falls on two facts the GPU test logs: the wait is bandwidth-bound, and
the chain fits in idle GPU time. Blended guess for FAR-SET at 45 GB, if both hold and the map trains to 0.8 / 0.7 [G]: +8-12 %
tok/s (83 -> 90-93). Not a 350 lever; a BYTES lever.
24 GB (replay K 147, MB per kept token = misses x 2.163): LRU 266 MB; FAR-SET stand-in 235 MB (-11.9 %); M-38's r 0.7 192 MB
(-28 %); perfect kept rows 196 MB; perfect all rows 159 MB; Belady 150 MB. On BYTES10's ledger (SSD bytes per kept token ~60 ->
~10 MB) a -12 to -28 % cut is a step, not the wall: it stacks with the store levers (bytes per expert), it does not replace them.

## 5. Prior-art status (checked from memory of the papers named in the brief and TRIED; no web search tonight)
| piece | status | source |
|---|---|---|
| a draft model running ahead feeds expert prefetch / lookahead-aware eviction | published | MoE-SpeQ (arXiv 2511.14102: a quantized draft, <= 32 tokens ahead); Kamen's idea is this with a 100-token horizon and set-level eviction |
| draft states through the target's gates one round ahead (89 % top-1) | published; FAR-SET's pseudo-router is adapted from it | SP-MoE (arXiv 2510.10302) |
| router-agreement distillation of the verifier's expert sets (86-88 % prefetch hits) | published; the training recipe for the pseudo-router, adapted | DraftExpert (arXiv 2607.24434) |
| confidence-adaptive lookahead depth | published; a later arm (gate the horizon on the chain's confidence), untested | SPICE (arXiv 2608.21240) |
| Belady-with-lookahead eviction, pre-gating routers, learned prefetch predictors | published | Read-ME, ProMoE, HOBBIT (2411.01433), MoE-Infinity, Pre-gated MoE |
| Belady-informed eviction from the draft look-ahead (one window) | TRIED: Rnd17#2 BETWEEN -2 % (MoonShots 10-19); R-57b KILL (BigMoE 10-31) | ~/TRIED/TRIED.md:104, :226 |
| the memo of exact routes for repeated tokens as a forecast feature | engineering, untested live (recall 0.38-0.49 alone here) | this file, 1b |
| NEW measurements: (i) the rejected draft rows hold a third of the cache reuse (the true-path ceiling is recall 0.68); (ii) misses vs stalled layers under foresight (34 % vs 5 %); (iii) equal recall / precision buys 2-4x fewer misses with correlated errors than with M-38's independent noise | new (for NOVELTY if the orchestrator wants them) | this file, 1b / 1d |

## 7. Follow-ups for FAR-G1 (orchestrator 22:1x, from BigMoE; CPU, 22:10-22:2x)

**7a. Origin spacing for the capture: `far_origin.py` -> `far_origin.txt`** [K]
PERFECT-KEPT forecast (the kept-row ceiling) with far-draft origins only every S-th check and F far tokens per origin (a
decision between origins uses the last origin; staleness modelled optimistically = horizon coverage only), M-38's frame:
| K 266 | F 64 | F 96 | F 128 | K 147 | F 64 | F 96 | F 128 |
|---|---|---|---|---|---|---|---|
| S 1 (176 origins) | -19.9 % (recall 0.67) | | -20.3 % (0.68) = all | S 1 | -26.1 % (0.76) | | -26.3 % |
| S 2 (89) | -19.6 % (0.66) | -20.3 % (0.68) | -20.3 % | S 2 | -25.9 % | -26.3 % | -26.3 % |
| **S 4 (46)** | **-18.9 % (0.64)** | **-20.3 % (0.68)** | -20.3 % | S 4 | -25.3 % (0.73) | -26.3 % (0.76) | -26.3 % |
| S 8 (24) | -17.1 % (0.56) | -20.0 % (0.67) | -20.3 % (0.68) | S 8 | -23.1 % (0.67) | -26.2 % | -26.3 % |
-> **44-46 origins (every 4th check) are enough for the K 266 line if each origin carries F >= 96 far tokens** (then every
decision up to 3 checks after its origin still sees 8 checks ahead; the line is identical to origins at every check). At F 64
the stale decisions lose their 8-check view on code (10-16 kept per check): -18.9 % and recall 0.64, 1.4 points under the
line. With F 128 even 24 origins (every 8th check) reproduce the every-check line. Recommendation: every 4th check, F 96
(or every 8th, F 128: half the far prefills). The lines in section 3 stay as written; the read-out replays the LRU on the
CPU, so the capture's pool pollution does not touch them.

**7b. Is the SSD wait latency- or bandwidth-bound? `far_wait.py` -> `far_wait.txt`** [K BigMoE r57 per-check logs, read-only]
Data: 106 `<tag>_checks.json` files (per check: miss, io = demand SSD busy ms, svc = gate service ms of calls with misses =
the GPU's spin bound, ck, rows, ws), checks >= 3, not warm-split, not paused; the stalled-layer count ('slow') is a per-run
median in r57's stdout only, not logged per check or per run in the json, so the latency part is read from the intercept.
Fit per tag, io ~ a + b * miss (and svc ~ the same):
- EXACT C 160 (cap45-like: c45t/ka/vt, 483 misses/check, io 91 ms): b 0.15-0.18 ms per miss, a 5-19 ms, R2 0.86-1.00;
  cmb*/mix/pr/dfa (150-200 misses, io 36-44 ms): b 0.15-0.17, a 10-14 ms, R2 0.93-0.99; chA/chB (chat, ~100-116 misses, io
  28-31 ms): b 0.17-0.18, a 9-10 ms, R2 0.96-0.98; u3c (854 misses, io 150 ms): b 0.152, a 19 ms, R2 0.96.
- EXACT C 224 (w1t*, 223 misses, io 46-53 ms): b 0.13-0.15, a 12-20 ms, R2 0.86-0.99; d256a (104 misses, io 29 ms): b 0.15,
  a 10 ms, R2 0.98.
- FAST C 160 (fastBE*, 140-170 misses): b 0.14-0.18, a 7-11 ms, R2 0.94-0.97; u3 (319 misses): b 0.172, a 13, R2 0.99.
  FAST with QOM (u3cq / qomq / qalq, 15:3x): b 0.22-0.29, a -2..12, R2 0.31-0.57 (optional misses break the line); C 128
  FAST (allin*/u3cb): svc 27-34 ms << io 64-79 ms (the GPU does not wait for all of the SSD time there), R2 0.5-0.6.
- svc = io + 0.3-2 ms on every EXACT tag: the GPU's spin on misses equals the SSD's busy time, so the demand reads are on the
  critical path in full (OOO overlaps ~nothing in these logs).
- io / miss falls from ~0.30 ms (low-miss tercile) to ~0.18 ms (high tercile) on every EXACT tag: a fixed part plus a
  per-miss part, as the fit says.
**Verdict: bandwidth-bound at the margin, with a ~10-16 ms latency floor per check.** 2.163 MB per miss / 0.15-0.17 ms =
**12.7-14.4 GB/s = the SSD's streaming rate**, so every miss removed saves its bytes' time: at cap45's 483 misses (io 91 ms,
check 138 ms) a -26.5 % cut = -128 misses = **-20 ms = -14.5 % of the check**; at the live C 256-like 100-200 misses (io 30-45
ms, check 70-85 ms) = -4 to -8 ms = -6 to -9 %. The fixed 10-16 ms (= ~48 layers x 0.2-0.3 ms, the first-read latency of
each layer's burst) is the latency-bound part GOAL.md saw at 69 misses (12 + 0.16 x 69 = 23 ms, its 24): it does not shrink
with fewer misses, only with fewer stalled layers (the replay: perfect foresight -5 %, Belady -19 %). Section 4's "latency-
bound" column is therefore the floor, not the expectation: the miss cut maps ~1:1 onto io above the ~12 ms floor. FAST and
EXACT agree where both exist at C 160 (b 0.15-0.18 both); QOM's optional misses are the exception (no linear law).

## 8. NEAR + FAR hybrid (Kamen's tournament-predictor idea, orchestrator 22:2x): `far_hybrid.py` -> `far_hybrid.txt` (22:22, 39 s)
Frame: M-38's chained cap45 replay, FH look-ahead H 8, K 266 / 147. **NEAR** = the next 1-2 checks' experts: NEAR-V1 = the
true next-check needs with per-use recall 0.75 (V1's layer look-ahead level, 0.72-0.79 [K TRIED]) and popularity false
positives (M-38's noise model: the same per-use-independent optimism as M-38's r 0.7, see 1d), NEAR-P = perfect (the
bound). **FAR** = the ridge state map top-16 on checks i + 1 .. i + 8 (1e: -14.6 %). Combinations: (a) TIER near-protected >
far-protected (farthest first) > LRU or the learned key; (b) CHOOSER: per layer an online EMA of each forecaster's precision
(near resolved one check later, far at the horizon) decides which takes the protected tier; (c) SCORE: s = beta * [near] +
gamma^(d - 1) * [far], lowest leaves first. [K far_hybrid.txt]

| K 266 (LRU 42.9) | misses | vs LRU | K 147 (LRU 123.1) | misses | vs LRU |
|---|---|---|---|---|---|
| learned key alone | 39.5 | -7.8 % | | 116.0 | -5.7 % |
| FAR ridge alone (+ LRU) | 36.6 | -14.6 % | | 109.5 | -11.0 % |
| FAR ridge + learned key | 36.4 | -15.0 % | | 109.8 | -10.8 % |
| NEAR-V1 next 1 check alone | 40.4 | -5.9 % | | 110.3 | -10.4 % |
| NEAR-V1 next 2 checks alone | 38.1 | -11.1 % | | 101.2 | -17.8 % |
| NEAR-P1 / P2 (perfect 1 / 2 checks) | 39.4 / 36.7 | -8.3 / -14.4 % | | 105.0 / 94.2 | -14.7 / -23.5 % |
| **(a) TIER V1(2) > FAR > LRU** | **34.2** | **-20.2 %** | | **95.6** | **-22.3 %** |
| (a) TIER V1(1) > FAR > LRU | 35.4 | -17.6 % | | 101.2 | -17.8 % |
| (a) TIER V1(2) > FAR > learned key | 33.5 | -22.0 % | | 95.1 | -22.7 % |
| (a) TIER P2 > FAR > learned key | 32.5 | -24.3 % | | 89.3 | -27.5 % |
| (b) CHOOSER (V1(2), FAR) + LRU | 34.4 | -19.7 % | | 101.4 | -17.6 % |
| (c) SCORE b 1.5 g 0.85 / b 3 g 0.7 + LRU | 34.2 | -20.2 % | | 95.6 | -22.3 % |
| (c) SCORE b 1.5 g 0.85 + learned key | 33.5 | -22.0 % | | 95.1 | -22.7 % |

Reading:
1. **Near and far are complementary, nearly additive:** FAR alone -14.6 %, NEAR-V1(2) alone -11.1 %, tiered -20.2 % at
   K 266 (+5.6 points over far alone); at K 147 -11.0 / -17.8 / -22.3 % (+11.3 over far alone: at 24 GB the near horizon
   matters more because the pool turns over faster). The tiered hybrid reaches M-38's PASS line (>= 20 %) that neither
   part reaches alone; with the learned key among the rest -22.0 / -22.7 %.
2. **Tiering is the right combiner; the chooser and the score add nothing.** (c) with any beta / gamma reproduces the tier
   exactly (a near forecast at distance 1-2 outranks every far one in both), and (b) is worse (-19.7 / -17.6 %): the
   online precision EMAs swap tiers on noise, and the near forecast is the better protected tier in every layer here
   (NEAR-V1's precision is uniform by construction, so the per-layer chooser has nothing real to learn on this stand-in;
   the capture will show whether deep layers trust near and shallow layers far).
3. **The near horizon must be 2 checks, not 1:** V1(1) -17.6 % vs V1(2) -20.2 % (K 266); a 1-check near forecast at r 0.75
   is worth less than the far map alone. Two checks = ~20-30 tokens = the far chain's first two blocks.
4. **Additivity with the learned key, recovered:** FAR-COMBO (1g) had +6.6 points from the key under a perfect kept-row
   forecast and ~0 under the ridge; the hybrid gets +1.8 points (20.2 -> 22.0 %) at K 266 and +0.4 at K 147, and with a
   perfect 2-check near +4.1 (20.2 -> 24.3 %). A realistic near forecast recovers roughly a third of the precise-forecast
   additivity; the rest needs the far map's precision on old residents (1g).
5. **What is realistic here:** NEAR-V1 is a noise model at V1's recall level, with independent errors per use (optimistic,
   like M-38's r 0.7). The engine source for it is the far chain's first 16-32 tokens through the state map (the drafter's
   own next blocks: token agreement ~65-85 % at the first block [K kept per check]), not V1's within-check look-ahead
   (which sees one layer ahead, not one check). The FAR-G1 capture measures it directly: the first two blocks of every
   origin at distance 1-2 checks, scored as the near tier. Lines for the capture: NEAR(2) real >= -10 % alone and the
   tiered hybrid >= -20 % at K 266 -> PASS (replaces the single-forecast line); hybrid 10-20 % BETWEEN; < 10 % KILL.
6. Cost: the near tier is the same far chain and the same map (no extra compute); the tiering is a two-level key in the
   eviction (free). EXACT-safe as before (residency only).

## 6. TRIED lines (for the orchestrator to record)
- FAR-CEIL: the true future tokens as a perfect far draft through token -> expert maps (memo / leave-one-prompt-out table /
  all-prompt table), H 8, M-38's cap45 replay | wall: SSD | line: 45 / 24 GB | exact: yes (residency only) | verdict: KILL for
  token maps | the ceiling of any map of the true tokens is recall 0.68 / precision 1.00 (-20.3 % misses) at K 266, 0.76 at
  K 147: the rejected draft rows hold 32 % of the reuse; honest maps 0.53 / 0.66, -6.0 % (learned key 7.8 %) | 11-04 |
  ~/FableGod/far/far_ceiling.txt
- FAR-STATE: shallow-state stand-ins for a far drafter's states (nearest earlier row; a leave-one-prompt-out ridge
  state -> route map), all 16 future rows | wall: SSD | line: 45 / 24 GB | exact: yes | verdict: BETWEEN | ridge top-16: recall
  0.70 / precision 0.64, -14.6 % misses at K 266 (stalled layers -1.4 %), -11.9 % at K 147 (235 MB per kept token); KNN -7.1 %;
  M-38's PASS point -26.5 % | 11-04 | ~/FableGod/far/far_state.txt, far_ridge.txt
- FAR-COMBO: FH foresight + M-14's learned key among the unforecast residents | wall: SSD | line: 45 / 24 GB | exact: yes |
  verdict: PASS with a precise forecast (perfect kept-row routes -20.3 -> -26.9 % at K 266, perfect all rows -33.7 -> -37.2 %),
  nothing with the ridge stand-in (-14.6 -> -15.0 %: its false positives sit on the old residents, precision 0.57 there) |
  11-04 | ~/FableGod/far/far_combo.txt
- FAR-G1 (proposed GPU capture, section 3): PASS >= 20 % fewer misses than LRU at K 266 with set recall >= 0.80 on the far
  chain's own text through a perfect map; BETWEEN 10-20 % (bytes lever only); KILL < 10 % or <= the learned key.
