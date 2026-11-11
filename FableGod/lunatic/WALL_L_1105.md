# WALL_L_1105: is "distinct experts per kept token" a law of this model? (Lunatic, Thu 5 Nov 2026)

Mission 34 (resumed, one target): break the wall Cloud Fable 3 worked on: distinct expert groups per kept token. Bar: >= 2x
on code AND chat at map-PASS quality (all-row flips <= 1.34 / 5.42 / 3.69 %). Side: first principles and the model's
structure (Fable C takes the data side).
First version 16:5x EET; refined after 17:00 (section 9).
Marks: [K] measured (file named), [P] computed here from named data, [G] judgement.
Runs: CPU only, numpy, `python -I`, one process at a time, peak < 2 GB. No GPU, no downloads. Writes only in
~/FableGod/lunatic/.
Tests (scripts and outputs beside this file):
- T1 `wl_redund.py` -> `wl_redund.out`: the real expert weights (models/bare, MLX 4-bit), layers 3 / 23 / 43.
- T2 `wl_width.py` -> `wl_width.out`: merged checks of B streams.
- T2b `wl_width2.py`: width times a perfect stop.
- T2c `wl_chatlen.py` -> `wl_chatlen.out`: cost against check length.
- T3 `wl_slot.py` -> `wl_slot.out`: the lookahead rows' coverage by the check's union.
Routing data: the 36 decision-map windows, MoonShots export (`cloud3/macpub/data/decision_map_moonshots`: teacher-forced
top-10 ids per row and layer along the 125B's own greedy text, with the HPD-0 walk's live-cut checks).

## 0. The answer in eight lines
1. **One part of it is a law of the model, and it is measured here.** Take R consecutive rows of the 125B's own text and
   keep all of them. The union costs f(R) groups per kept token [P on K, T2c]:
   - code: 480 / 276 / 218 / 163 at R = 1 / 8 / 16 / 32;
   - IFEval: 480 / 258 / 202 / 151;
   - multi: 480 / 226 / 177 / 133.
   - f falls ~25 % per doubling of R, because the experts are 512 independent functions (T1 [K]): no shared base, no
     shared subspace, no near-duplicate neurons, no low-rank single experts.
2. **Everything else is how we run it.** Groups per kept token = f(kept per check) x waste x (1 / streams sharing)
   x (k-cut factor):
   - kept per check is set by the drafter;
   - waste is the share of computed rows that get rejected, set by the stop rule;
   - streams sharing is the number of streams verified in one pass;
   - the k-cut factor comes from cutting experts per row, which is priced in flips (F3's floor).
3. **Chat's wall is not experts.** At equal check length, chat routes FEWER distinct experts per token than code: R16
   gives 202 / 177 vs 218. Today's chat cost (492 / 490 vs code 304.5) is entirely short checks (2.4 / 1.9 kept) and
   waste (x1.35-1.39 vs x1.16) [P on K].
4. **The 2x lines, per stream, EXACT, zero waste:**
   - code needs ~38 kept per check (f(38) ~ 152 [P, extrapolated]);
   - IFEval needs ~9-10 kept per check;
   - multi needs ~6-7 kept per check.
   - At today's waste ratios: ~52 / ~21 / ~15.
   - Today: 8.5-10.5 on code, 2.4 / 1.9 on chat.
5. **The expert side has nothing left to give.**
   - Shared base and deltas: KILL [K T1; TRIED 385 / 389 / 396 / 93].
   - Redundancy between experts: KILL [K T1].
   - Re-routing under a learned correction: KILL [TRIED 236; cloud3 P2 reading 4; T1].
   - Computing an expert once for many future tokens exists only as width or as deeper drafts (DEPTH32 x0.66 [K]).
6. **Width is the only measured break of the law on both domains, and it is EXACT** [P on K, T2 / T2b]. B streams'
   live-cut checks verified in one pass (groups per kept token vs one stream):
   - code: x0.55 at B4, x0.35 at B8;
   - IFEval: x0.81 at B4, x0.65 at B8, x0.54 at B12;
   - multi: x0.85, x0.71, x0.61;
   - with a perfect stop: chat x0.51 / x0.55 at B8 and x0.43 / x0.49 at B12.
   - Code streams share experts like one stream; chat streams on different topics and languages share little.
   - The mechanism is published (batched MoE decoding). Only the measured per-domain law and the one-user census are ours.
7. **Single best idea for this bar:** break "one stream per pass" (section 6, with its 10-minute test). Code passes at B
   >= 4. Chat reaches 2x only at B 8-12 with a good stop, or with a chat drafter that keeps 7-10 per check.
8. **No genuinely NEW expert-side mechanism survived its kill test:**
   - SLOT-CASCADE (section 5.3): KILL-likely. The union holds only 34-53 % of a chat lookahead row's routed mass.
   - REFRESH (section 5.4): a weak [G].
   - The hunt continues in section 9.

## 1. The decomposition (why the wall has five parts and only one belongs to the model)
For one check: groups = sum over layers of |union of the computed rows' top-k|, and kept = the committed tokens.
- With a perfect stop (only kept rows computed), groups / kept = f(kept): the union law of the 125B's own text.
- Every other factor is a choice:

| factor | what sets it | today code / IFEval / multi | the 2x lever |
|---|---|---|---|
| f(kept) | the model (independent experts, T1) + check length | f(8.5) ~ 272 / f(2.4) ~ 373 / f(1.9) ~ 379 [P] | longer accepted checks |
| waste = cost / perfect-stop cost | stop rule (live cut at cf product 0.3) | x1.16 / x1.39 / x1.35 [P: 304.5 / 263.0; 492.3 / 354.6; 490.0 / 364.4] | perfect stop (bound) |
| 1 / streams sharing | how many streams one pass verifies | 1 | width (W) |
| k-cut factor | per-row k / precision (FAST) | 1 (EXACT) | Q1 / split / QOM; F3's floor prices them |
| kept per check | the drafter | 8.5-10.5 / 2.4 / 1.9 | drafting (MTP Head) |

**Today's numbers reproduce cloud3 P2 exactly.** One stream, EXACT, live cut: 304.5 / 492.3 / 490.0 [P, T2 = cloud3 P2
section 6, row 1]. The mission's ~290 is FAST code.

**f(R) per domain** (groups per kept token with all R rows kept; the 125B's own text; 36 windows) [P on K, T2c]:

| R | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|
| code | 480 | 406 | 365 | 337 | 300 | 276 | 241 | 218 | 185 | 163 |
| IFEval | 480 | 391 | 347 | 319 | 282 | 258 | 225 | 202 | 172 | 151 |
| multi | 480 | 368 | 317 | 286 | 249 | 226 | 195 | 177 | 150 | 133 |

Readings:
- (a) **The law is gentle.** Each doubling of R saves 25 %. No per-stream regime exists where sharing suddenly gets cheap.
- (b) **Chat is the CHEAPER domain at equal length.** Consecutive chat rows overlap more (cloud3 P2: 38 / 47 % vs 27 %).
  The chat wall is check length and waste, an acceptance wall in disguise.
- (c) **Code at 2x per stream needs ~38 kept per check with zero waste.** An ideal drafter on code (alpha ~0.982 [G,
  LUNACY2 section 1b]) keeps ~38.5 of 64 rows: the edge of physics. Per stream, code's 2x at EXACT is out of reach for
  any real drafter. Code's 2x needs FAST cuts (the split: x0.51-0.63 at 1.31 % flips [K, brief]) or width.

## 2. What the model's structure says (T1, real weights) [K]
Method: models/bare (MLX 4-bit affine g64, the upstream checkpoint), switch_mlp gate / up / down, layers 3 / 23 / 43.
128 sampled experts per stacked statistic.

| statistic | measured (layers 3 / 23 / 43) | if experts were independent |
|---|---|---|
| index-aligned mean expert's share of energy (gate / up / down) | 0.89-1.02 % / 0.78 % / 0.78 % | 1 / 128 = 0.78 % |
| stacked input-side spectrum, top 64 / 256 / 1024 of 2560 dims (gate) | 4.6-5.9 / 15.3-18.7 / 50.1-55.2 % | 2.5 / 10 / 40 % (isotropic) |
| stacked output-side spectrum (down) | 4.2-5.6 / 15.0-18.5 / 51.4-54.8 % | same |
| one expert's gate, top 16 / 64 / 160 of 640 singular directions | 6.9-7.6 / 22.3-23.3 / 45.9-47.4 % | ~4.5 / 16 / 37 % (random) |
| best \|cos\| of a gate neuron with 19,840 neurons of 31 other experts | median 0.086-0.090, p99 0.14-0.21, > 0.5: 0.00-0.01 % | ~0.09 |

Verdict:
- **The 512 experts of a layer are independent functions.** There is a mild anisotropy, about 1.5-1.9x isotropic in the
  top directions; the hidden states' outlier dimensions would explain it.
- No base, subspace or duplicate can carry a missing expert. This confirms and extends TRIED 385 (Fit12 C1 O3), 389
  (MoonShots base+delta, Git Re-Basin aligned 0.038-0.045 vs null 0.035), 396 (Leap L1-16a) and 93 (M-27: deltas
  +0.23 b).
- New here: the common-subspace and near-duplicate-neuron checks, which those lines did not run.
- Consequence for the brief's options 1 and 4: dead on the weights.
- Consequence for option 3 (a learned correction): every union member's output is an independent direction, so no
  correction built from union members can carry a missing expert [G from T1].
  - This agrees with cloud3 P2 reading 4 ("KILL by function") and TRIED 236 (REROUTE b25).

## 3. Width: the one assumption whose break is measured on both domains (T2 / T2b) [P on K]
Setup:
- Merged pass j = check j of each of B windows (B concurrent streams), live-cut rows, top-10 on every row (EXACT).
- 6 random groupings per B. Rows per group = the expert tile's real rows (16 slots).

| groups per kept token (x vs one stream) | B1 | B2 | B4 | B8 | B12 |
|---|---|---|---|---|---|
| code | 304.5 | 236.9 (x0.78) | 167.7 (x0.55) | 105.5 (x0.35) | 79.9 (x0.26) |
| IFEval | 492.3 | 457.1 (x0.93) | 400.2 (x0.81) | 319.2 (x0.65) | 266.8 (x0.54) |
| multi | 490.0 | 459.4 (x0.94) | 415.5 (x0.85) | 346.2 (x0.71) | 301.3 (x0.61) |
| mixed domains | 429.9 | 380.2 (x0.88) | 314.2 (x0.73) | 227.2 (x0.53) | 183.9 (x0.43) |
| code + perfect stop | 263.0 | 212.7 | 155.8 (x0.51) | 100.5 (x0.33) | 76.8 (x0.25) |
| IFEval + perfect stop | 354.6 | 337.8 | 302.7 (x0.61) | 249.0 (x0.51) | 212.4 (x0.43) |
| multi + perfect stop | 364.4 | 341.8 | 314.9 (x0.64) | 270.7 (x0.55) | 240.0 (x0.49) |

Tile occupancy grows from 1.9 to 5.1 real rows per group at B8 on code (of 16 slots). Per EXPERT_KERNEL section 6, the
group's op set and bytes are fixed per record, so the extra rows ride nearly free [K EXPERT_KERNEL line 67-70].

Readings:
- Code streams share experts across prompts almost like consecutive rows of one stream. LUNACY2's "cross-stream union
  ~= consecutive" holds on code.
- Chat streams do not: IFEval prompts on different instructions and multi in four languages route apart.
- So width is >= 2x on code from B4. On chat it needs B ~8-12 plus a stop close to perfect.
- Q1 0.5 (sure rows top-2) on top of width [P]: code 105.1 -> 47.6 at B8; IFEval 258.5 -> 174.7; multi 305.5 -> 222.8.
  These are FAST, under F3's floor.
- Novelty, honestly: batching streams to share expert reads is published (throughput-oriented MoE offloading,
  MoE-Lightning 2024; FlexGen 2023; every fused-MoE server). Two pieces are ours:
  - this per-domain law on the 125B;
  - the single-user census: 63 % of the agent generation in the census week ran with >= 8 sessions at once [K
    LUNACY2 W].
  MoonShots should tag it "published / adapted".

## 4. Chat is an acceptance wall (the reframe) [P on K]
- Chat costs 492 / 490 because a check commits 2.4 / 1.9 tokens and computes 3.7 / 3.0 rows.
- Give chat code's check length and it is cheaper than code: f(8) = 258 / 226 vs 276.
- So no expert-side lever can bring chat to 2x per stream.
  - The perfect stop bounds every stop rule at x0.72 / x0.74 [K cloud3 P2 section 6].
  - Every k-cut on chat pays F3's 4 / 2.5-point context floor out of a 5.42 / 3.69 budget.
- What 2x on chat needs, per stream, EXACT: ~7-10 kept per chat check with a near-perfect stop.
  - That is a chat drafter about 3-4x better than today's (2.4 kept).
  - It is the drafter track's problem (MTP Head), not the expert track's.
- Width gives it at B 8-12 when the user really runs that many chat streams. Kamen's agents do; a single chat does not.

## 5. Ideas ranked
| # | idea (the assumption it breaks) | number it moves [P on K unless marked] | verdict | novelty | nearest prior art |
|---|---|---|---|---|---|
| 1 | **W, one user is a batch** ("one stream per pass") | code x0.55 (B4) / x0.35 (B8); chat x0.65-0.71 (B8), x0.54-0.61 (B12); + perfect stop chat x0.51-0.55 (B8) | PASS code; BETWEEN chat (needs B >= 8 + stops) | published mechanism, adapted use | MoE-Lightning; FlexGen; fused-MoE batched serving |
| 2 | **Chat is an acceptance wall** ("chat routes more experts") | chat 2x = 7-10 kept per check at zero waste; f(8) chat 258 / 226 < code 276 | a measurement; routes chat to the drafter track | new as a measurement | cloud3 P2 (growth per check), DRAFTER_PLAN_1104 |
| 3 | SLOT-CASCADE: Lookahead decoding inside the verify pass's empty tile slots, restricted to the check's union (zero bytes); the next check verifies its corrections | union holds 64 / 53 / 64 % (s1) to 51 / 34 / 44 % (s8) of a lookahead row's routed mass; rank-0 inside 70 / 60 / 73 % -> 57 / 40 / 51 % | KILL-likely: a filter that drops ~half the mass at every layer is far worse than F3's arms (18 % dropped on sure rows already flips 3.9 % on IFEval) | adapted (the union restriction is new) | Lookahead decoding (Jacobi), TriForce, cascade drafting |
| 4 | REFRESH: FAST cuts plus an exact re-prefill of the last K rows every K rows (bounded context drift) | cost 48 x U(K) / K = 124 (K 64) / 53 (K 256) groups per token [P from lu_union]; quality unknown | weak [G]: only cheap at K >= 256, and F3's floor is probably local | new-ish | CacheBlend (partial KV recompute), CALM state propagation |
| 5 | shared base + small deltas | mean expert 0.78-1.02 % of energy (independent 0.78 %) | KILL [K T1] | published (MoBE, D2-MoE, ResMoE, Sub-MoE) | same; TRIED 385 / 389 / 396 / 93 |
| 6 | weights' redundancy between experts (common subspace, duplicates) | top-256 of 2560 dims hold 15-19 % (isotropic 10 %); no neuron pair > 0.5 | KILL [K T1] | published | Sub-MoE, MC-SMoE, HC-SMoE |
| 7 | re-routing under a learned correction | substitutes are independent functions | KILL [TRIED 236; cloud3 P2; T1] | published | cache-conditional experts, Lynx |
| 8 | compute an expert once for many future tokens | only as width (#1) or deeper drafts (DEPTH32 x0.66 [K]) | no third form in one stream (each check's tokens depend on the last) | n/a | n/a |

### 5.3 SLOT-CASCADE, in five lines (kept because its kill is a coverage number, not a run)
1. A 16-row expert tile carries ~2 real rows at S16 (12 % [K EXPERT_KERNEL]). The other 14 slots are paid for anyway.
2. Fill them with S lookahead rows: the drafter's continuation past the check's last row.
3. Each lookahead row is computed only with the union's experts. That adds zero groups; its trunk costs ~0.2-0.3 ms
   per row.
4. The lookahead argmax confirms or corrects the drafter. The next check verifies those candidates exactly, so the
   output stays EXACT.
5. The gain would come from longer accepted chat checks: a filter that agrees with the 125B at q 0.9 means ~6-10 kept
   [G].
- The coverage kills it: the union holds only 34-53 % of a chat lookahead row's routed mass.
- Adding each lookahead row's missing rank-0 experts costs 13-29 groups per row (~10-15 % of a full row) and lifts
  coverage only ~15 points [P].

## 6. The single best idea and its 10-minute test
**W: verify several streams of one user in one pass.** It breaks "one stream per pass".

Mechanism (5 lines):
1. B concurrent streams (Kamen's agents) each draft their own check.
2. One verify pass carries all B checks' rows. A stream's rows attend only to their own context; the expert phase
   groups rows by expert across streams.
3. A group's bytes and op set are paid once and serve every stream's rows. The tiles go from 12 % to 30-40 % real
   rows at B8.
4. Output per stream is bit-identical to its solo run, because tile rows are independent.
5. Groups per kept token: x0.35 code / x0.65-0.71 chat at B8 [P on K, T2].

**Why it might not be published:** the batching is published. What isn't is that one person's agents on one laptop are
the batch, and that code streams share experts like one stream while chat streams don't. MoonShots: tag it.

**The cheapest test** (GPU, <= 10 min, after Fri 00:15; Leap / BigMoE):
- An expert-phase microbench with the NAX6 kernel on recorded routings.
- Two arms per domain:
  - B1: each window's live-cut check alone, summed over 8 windows;
  - B8: the same 8 checks merged into one routing (ids from T2's groupings; real activations for the rows from the
    d02a taps, or random).
- Measure the expert-phase ms per kept token. Check bit equality of each stream's rows between the two arms (fp32 / bf16
  outputs).
- **PASS:**
  - code B8 <= 0.45x of B1 (replay x0.35 + 30 % margin for groups past 16 rows and the per-row MACs);
  - IFEval B8 <= 0.80x;
  - every stream's rows bit-identical.
- **KILL:**
  - code B8 > 0.60x;
  - or any bit difference in a stream's rows (W would then be FAST, not EXACT).
- **BETWEEN:** in between. Then the per-group cost grows with real rows faster than EXPERT_KERNEL section 6 says;
  measure the slope.
- **Halved for decisions** (Kamen's rule): code x0.35 -> ~x0.5-0.6; chat x0.65 -> ~x0.8 at B8.

## 7. What would make me abandon each live line
- W:
  - the microbench's KILL;
  - or the engine cannot hold B >= 4 streams' state (36 GDN states x 3.1 MB x B plus KV) inside the 45 GB line;
  - or Kamen's concurrent streams do not share one engine (separate processes would each read every group).
- The chat reframe:
  - an engine run on chat with the reference text as drafts (perfect acceptance, 16 rows) costs > 260 groups per kept
    token. That would mean the map's f(R) overstates engine sharing.
- SLOT-CASCADE is revived only if a map arm shows the union-restricted lookahead argmax matching the 125B's on >= 90 %
  of chat rows.
- REFRESH is revived only if a map arm shows F3's floor at least halved by an exact context reset every 256 rows.

## 8. Reproducibility and the TRIED lines (for the recorder; I do not write outside this folder)
Commands, all `nice -n 15 ~/BigMoE/.venv/bin/python3 -I <script>` in ~/FableGod/lunatic, each seconds:
- `wl_redund.py 3`, then `wl_redund.py 23 43` -> wl_redund.out (10 + 23 s);
- `wl_width.py` -> wl_width.out (13 s);
- `wl_width2.py` (reuses wl_width.py; 16 s);
- `wl_chatlen.py` -> wl_chatlen.out (3 s);
- `wl_slot.py` -> wl_slot.out (1 s).

TRIED lines:
- **WL-T1 expert independence:** 512 experts per layer tested for a shared base, a common subspace, near-duplicate neurons
  and low-rank single experts, on the real 4-bit weights, layers 3 / 23 / 43.
  - wall: bytes; line: all; exact: n/a.
  - verdict: KILL for options 1 and 4 (base + deltas, redundancy).
  - numbers: mean 0.78-1.02 % (independent 0.78 %); top-256 of 2560 dims 15-19 % (isotropic 10 %); best neuron
    \|cos\| median 0.086-0.090 (chance 0.09).
  - 2026-11-05, lunatic/wl_redund.out.
- **WL-T2 width (W):** merged live-cut checks of B streams, EXACT.
  - wall: GPU / bytes; line: 45 / 24.
  - verdict: PASS code (CPU replay), BETWEEN chat.
  - numbers: code x0.55 / x0.35 at B4 / B8; IFEval x0.65 / x0.54 at B8 / B12; multi x0.71 / x0.61; + perfect stop chat
    x0.51-0.55 at B8.
  - 2026-11-05, lunatic/wl_width.out.
- **WL-T2c f(R):** groups per kept token for R fully kept rows of the 125B's own text.
  - verdict: FOUND (chat is cheaper than code at equal length).
  - numbers: code 276 / 218 / 163, IFEval 258 / 202 / 151, multi 226 / 177 / 133 at R 8 / 16 / 32.
  - 2026-11-05, lunatic/wl_chatlen.out.
- **WL-T3 SLOT-CASCADE coverage:** the union holds 51-64 % (code) and 34-64 % (chat) of a lookahead row's routed mass.
  - verdict: KILL-likely (CPU).
  - 2026-11-05, lunatic/wl_slot.out.

## 9. After 17:00 (refinement log)
### 9.1 On chat at 45 GB, experts are a third of the cycle, so a 2x in groups is ~1.2x in chat speed [P on K]
- A chat cycle at the live cut has three parts:
  - draft 12.0 ms [K charter];
  - trunk at 3.7 rows ~15-17.5 ms [P: 12.7 ms of trunk bytes + 3.6 fixed + 0.32 per row; PHYSICS 17-23 at 16 rows];
  - experts 4.5 + 1,181 groups x 7-10 us = 12.8-16.5 ms [P: 492 groups x 2.4 kept; NAX6 7 us, NAX4 fit 10.15 us].
- That is 40-46 ms per 2.4 kept = 52-60 tok/s, matching the measured 56-63 [K Fable C UNION_STACK].
- Experts are ~32-36 % of the cycle, the trunk ~38 %, the draft ~27-30 %.
- Halving chat's groups per kept token therefore buys ~1.2x chat tok/s at 45 GB. At 24 / 12 GB the expert bytes are the
  exposed term, and the 2x counts nearly in full.
- Chat's 45 GB lever is kept per check: it spreads the draft, the trunk and the groups at once. This agrees with Fable
  C's UNION_STACK chat line ("no union cut moves chat toward 350; it is the row wall and the drafter") and the drafter
  plan's ceiling (FINDINGS line 106: chat kept 3-3.5 with affordable drafters, 5-8 only with the trunk as drafter).
- The measurement added here: in f(R), chat is CHEAPER than code at equal length. So once chat checks get longer, their
  expert term falls faster than code's did.

### 9.2 Width on chat is worth more in time than in groups [P on K]
- One pass for B8 chat streams, EXACT:
  - one batched draft call ~15-20 ms [K ANE-2: 12.4 ms for 3 candidates, 16-17 for 7];
  - trunk 12.7 + 3.6 + 29.6 rows x 0.32 = 25.8 ms;
  - experts 4.5 + 6,125 groups x 7-10 us = 47-66 ms;
  - total 88-112 ms per 19.2 kept tokens.
- That is 170-220 tok/s aggregate = x3.3-3.7 of one chat stream, although its groups fall only to x0.65. The draft and
  the trunk are shared across streams.
- Per stream it is 21-27 tok/s, slower than solo (52-60). Width is an aggregate lever: the right one for a user running
  many agents, the wrong one for one person chatting.
- Code at B8, same arithmetic: 105.5 groups x 68 kept, 113-139 ms per pass, 490-600 tok/s aggregate (x3.1-4.4 of the
  model's one-stream code line).
- The 10-minute microbench in section 6 tests only the expert term. The trunk and draft sharing is arithmetic on
  measured terms and needs the engine's multi-stream pass to confirm.

### 9.3 REFRESH: KILL by arithmetic [P + G]
- A refresh close enough to matter (K <= 32 rows) costs 48 x U(32) / 32 = 172 groups per token. That is more than any
  cut it would protect saves: Q1 0.5 saves ~200 on code [P, T2: 304.5 -> 105.1].
- At an affordable K (>= 256: 53 groups per token) it removes only the perturbation of rows >= 256 back.
- 36 of 48 layers are gated linear attention with decaying state, so F3's floor is mostly recent rows [G]. The refresh
  cannot reach it.

### 9.4 Where the hunt ends (17:1x)
- No NEW expert-side mechanism survived. The per-stream wall is f(R), the 125B's own union law (independent experts, T1),
  times acceptance (the drafter).
- Both halves were known in parts:
  - the chat side in Fable C's UNION_STACK and the drafter plan;
  - the expert side in TRIED's base / delta / reroute kills.
- What this file adds:
  - f(R) per domain, with chat cheaper than code at equal length;
  - the expert-independence checks TRIED lacked (common subspace, near-duplicate neurons, single-expert rank);
  - width priced per domain on the map's routing (code streams share like one stream; chat streams do not) and its
    time effect on chat (draft + trunk shared: aggregate x3.3-3.7 at B8 where groups say x0.65);
  - the 2x lines in kept per check: code ~38, IFEval ~9-10, multi ~6-7 at zero waste.
- The live bet is W. Its owners would be BigMoE (multi-stream pass) and Leap (the microbench). Kamen decides whether one
  engine should serve his concurrent agents.
- A findings-note draft in the README's shape is beside this file (`FINDING_DRAFT_1105_union_law.md`) for the
  orchestrator to place; I do not write in ~/FableGod/findings/.
