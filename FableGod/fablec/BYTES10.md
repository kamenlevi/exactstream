# BYTES10 (Fable C, mission C5): from ~62 to <= 10 MB of SSD reads per kept token, at 24 GB then 12 GB

Wed 4 Nov 2026, 00:00-00:35 EET; **corrected 06:47-07:3x EET** (section 7). NewIdeas found at 03:4x that `bytes10_replay.py`
line 135 built its keep mask on the layer axis instead of the rank axis (`keep[sure, :ks]` on a `[rows, 48, 10]` array), so
the 00:35 version read 3-bit base records on layers 0-9 only and every level in it was 2-4x too low. Every number below is
from the fixed replay: NewIdeas' run of the fixed copy (`~/NewIdeas/data/ledger/bytes10_replay_fix.out`, 03:53, same code,
one line changed) and my own rerun (`bytes10_replay.out`, 06:54, 306 s: identical to the last digit). The 00:35 outputs are kept as
`bytes10_replay_bug.*`. CPU only (one core at nice 15), no GPU, no downloads; writes only in ~/FableGod/fablec/. Evidence:
`bytes10_replay.py` -> `bytes10_replay.out` / `.json` (the map's 36 windows), `bytes10_bonus_masks.py` -> `masks/`, `bytes10_bonus_cfcheck.py` -> `bytes10_bonus_cfcheck.out` (screen 1(c)'s rows, 20:2x). Inputs: the
C5 brief, NewIdeas' BYTES10.md, `scripts/ten_x_ledger.py` and its outputs (`data/ledger/ledger_{A,B,C}.txt`,
`ledger_lastrow.txt`, `research/TEN_X_LEDGER.md`, 03:2x-05:5x), FLOOR night 1's directory, TRIED/NOVELTY.md,
Breakthrough's `COLD_read.txt` / `COLD_lru.txt` / `dm_t3all_read.log`, NewIdeas' `QOM_read.txt` and
`band45_capture_qom.txt`, WALL1_SCOREBOARD, SSD_WALL2, MoonShots' FINDINGS and WED_1104_READY, my UNION_STACK.md. Marks: [K]
measured (source), [P] arithmetic or replay on recorded routing, [G] judgement. Gains are halved when deciding.

Two replays, two miss counts. My replay touches a block's union in expert-ID order and counts misses in the loop; NewIdeas'
`ten_x_ledger.py` counts a block's misses against the pool as it stood before the block, then touches the union. When a
block's union is larger than the free pool, in-loop counting evicts residents of the same union before they are used, so my
numbers are x1.09 / x1.15 / x1.38 the ledger's at 45 / 24 / 12 GB [P, both replays agree once the count is the same]. The
ledger's count is what an engine reaches (use the resident members first, then stream the misses), so where both exist the
tables give "mine (ledger's)" and the ledger's is the floor. Code / IFEval / multi throughout.

## 0. The answer in eight lines

1. **The 62.4 [K] at 45 GB is reproduced by the replay: a warm session reads 60.4 (55.4) MB per kept token EXACT with plain
   LRU** [P], no kept scaling needed (bytes per kept token are flat in kept, section 1). Of the 60 on code, the per-request
   compulsory set with the prefill's reuse is 40.3 (the 90 answer experts per layer the prompt never routed): two thirds
   footprint, one third capacity (OPT 23.9). The 00:35 reading "not a cache problem" was the bug.
2. **At 24 and 12 GB the bytes are capacity misses** (a code request's working set is 292 + 253 distinct experts per layer,
   the pools hold 147 / 51): EXACT 245 / 210 / 183 (213 / 200 / 176) at 24 GB and 637 / 622 / 487 (461 / 546 / 445) at
   12 GB; OPT 117 / 94 / 84 and 351 / 387 / 304. **EXACT has no path to 10 at 24 or 12 GB** (~90-120 and ~250-400 with
   every exact lever); at 45 GB EXACT lands at ~35-45.
3. **The ledger audit's verdicts survive:** the record format multiplies; kept per check does not multiply for EXACT (flat
   from k 4 to k 16; the drafter acts through wasted rows, x1.25-1.8); QOM / t3 / COLD / u8 / the bonus gate act on the same
   rows and must be replayed jointly. **One verdict changes:** QOM's cut is not pool-dependent (EXACT -> QOM cf 0 + 3-bit:
   x0.22 / x0.25 / x0.21 on code at 45 / 24 / 12 GB; x0.4-0.6 on chat); it is domain-dependent.
4. **FAST floor today's levers reach** (QOM cf 0 + 3-bit sure rows, the bonus row exact as the map tested it): 61.7 / 86.1
   / 83.1 (54.5 / 84.4 / 81.7) at 24 GB, 135 / 251 / 226 (124 / 239 / 218) at 12 GB, 13.5 / 19.3 / 20.2 at 45 GB [P].
   Under 10 at 24 GB needs x0.16 more on code and x0.12 on chat.
5. **The new mechanism, THE GATED BONUS ROW,** is worth more than the 00:35 price: x0.50 / x0.23 / x0.17 at 24 GB (30.8 /
   20.0 / 14.2), x0.52 / x0.30 / x0.25 at 12 GB, and 5.7 / 4.5 / 4.4 at 45 GB [P]. **It ran on the map overnight**
   (NewIdeas / Breakthrough, 04:41 and 05:06): in QOM's own form PASS on code (z -0.28) and BETWEEN on chat (+1.56 / +1.49
   points, under Q1 at 0.99's level) [K]. With the drafter's own cf for the row it is a cf-0 lever: 1,368 / 165 / 7 of the
   1,368 bonus rows are gated at cf 0 / 0.5 / 0.9. Tag: new (web + NOVELTY; the ledger's section 8 concurs).
6. **The ranked path at 24 GB** (unhalved -> halved): gated bonus row -> 30.8 / 20.0 / 14.2 -> sure rows at top-1 + union ->
   16.8 / 11.8 / 7.9 -> rank-class pools -> 8.8 / 8.7 / 6.9 -> 11-bit scales -> **7.3 / 7.2 / 5.7** [P]; **halved ~25 / 33 /
   32**. Under 10 at 24 GB exists on paper in FAST only, with four levers: two need quality results (screens 1-2; screen 1's
   QOM form is PASS code / BETWEEN chat already), one needs an engine build (the pool policy), one a conversion (11-bit).
   2-bit sure-row records (MoonShots M-34, x0.71-0.83 after the gate [P]) and fewer wasted rows come after.
7. **12 GB:** the same stack lands at 24.3 / 26.1 / 18.6 unhalved, ~65 / 104 / 89 halved. **<= 10 at 12 GB needs the
   resident set to hold the sure rows' working set** (115 / 85 / 80 top-2 experts per layer vs 51 slots): 2-bit resident
   records, or a check whose first pass fits the pool.
8. **Screens** (section 5): screen 1 (a) is done (above); (b) / (c) ran at 19:38 (Breakthrough, ba53ede): (b) PASS as the
   null result (7 rows); (c) COST +0.12 / +0.20 / +0.28 points on the 165 rows gated at cf 0.5 (pooled z +1.71). The rows
   say the cost is my mask's treatment (ranks 0-1 + union only), not the bonus row's cf: the same kind of row under QOM's
   resident keep costs nothing (section 4, 19:38). The lever's quality numbers were already the resident-keep form's
   (PASS code / BETWEEN chat): unchanged; the threshold forms are closed. Screen 2 (sure rows at top-1 at live cuts, QOM
   form) and screen 3 (the W1B 24 GB calibration: 213-245 on code) stand.

## 1. The ledger audit: what multiplies, what overlaps

The 10x ledger (NewIdeas' BYTES10.md, `ten_x_ledger.py`) prices factors on their own and multiplies them per domain. My
replay uses the ledger's conventions (plain LRU per layer, prompt rows warm the pools, one session per domain carried across
the map's windows in walk order, live-cut blocks at t 0.3 with the p + 97 stand-in, the same three pool lines) plus OPT,
pinning, rank-class pools, the compulsory floors, a mixed-domain session, synthetic kept with the drafter's own cf, and the
bonus-row variants. All numbers below are [P] on the map's recorded routing unless marked; code unless said.

| factor (ledger row) | multiplies? | evidence (MB per kept token) | rule for the ledger |
|---|---|---|---|
| record format: 11-bit scales (-17.6 % [K MoonShots]), M-01 (-4.8 %), entropy coding (-13 %) | yes, bytes per miss | independent of routing and pools | one of 11-bit / M-01 (same fields); entropy after 11-bit is smaller than -13 % [G -5 %] |
| kept per check (the drafter) | **no for EXACT**; only through wasted rows and, for FAST, the bonus row | perfect drafter, no waste: 24 GB EXACT k 4 183.1, k 8 184.6, k 16 187.7 (ledger's count: 171.6 -> 160.1, x0.93); 45 GB 51.1-51.3; 12 GB 462-542 (rises: a 16-row block's union of 75 exceeds 51 slots). Two wasted rows per check: k 4 325, k 8 267, k 16 235 (x1.78 / x1.45 / x1.25). Today's live waste (9.65 rows for 7.91 kept) costs +27 % on code, +61 % on chat against a perfect drafter at the same kept [P ledger part A]. FAST Q1 0 t3 with the walk's cf: 24 GB k 4 85.3 -> k 16 51.1 (x0.60), all of it the bonus row's share | price the drafter as (wasted rows) x (bonus share), not as a union curve; D98's drafter buys x0.6 on chat, x0.79 on code |
| cache size / policy / locality | yes, as the largest term at 24 / 12 GB, not a multiplier on the others | 45 GB LRU 60.4 vs compulsory per request 40.3 (prefill reuse): 2/3 footprint. 24 GB LRU 245 vs 40.3: 84 % capacity. OPT 117 (24 GB), 351 (12 GB). On EXACT, PIN (top C/2 by popularity) 231 / 540 and RCP (rank-class pools) 256 / 517: x0.94-1.04 at 24 GB. **On the FAST stack (top-1 sure rows + gated bonus) RCP is x0.52 / x0.74 / x0.87 and PIN x0.70 / x0.83 / x0.94 at 24 GB** | on EXACT the policy factor is OPT's x0.5 at best and only learned replacement (FlashMoE, +51 % hit rate claimed [K paper]) reaches into it [G x0.8]; on the FAST stack a rank-class pool is a lever (section 3, #3) |
| per-request locality: domain switches | no | MIX session (code, IFEval, multi interleaved, one pool): 45 GB EXACT 62.9 / 43.6 / 34.3 vs 60.4 / 38.1 / 33.2 single-domain (+4 / +14 / +3 %); identical at 24 / 12 GB | the domain switch is not the cold-start cost; the per-request set is |
| per-request locality: the prefill's reads | the metric decides less than written at 00:35 | compulsory per request: 40.3 / 47.3 / 60.7 with prefill reuse (decode bytes), 129.5 / 99.7 / 93.5 without (process bytes, W1B's definition: `WALL1_BENCH.md` line 32). W1B's 62.4 [K] is the warm-session LRU replay (60.4 / 55.4) within 10 %: in a warm session the prefill's reads are mostly hits | the 00:35 "43.8 x 7.91 / 5.9 = 59" is withdrawn; W1B's process bytes and the replay's decode bytes agree at 45 GB |
| QOM (= Q1's reads) | jointly with the pool line and the domain | EXACT -> Q1 0 t3: code x0.22 / x0.25 / x0.21 at 45 / 24 / 12 GB; IFEval x0.51 / x0.41 / x0.40; multi x0.61 / x0.46 / x0.46. The 00:35 "cut shrinks as C falls" was the bug | a QOM ratio belongs to its pool line and domain; never multiply across |
| 3-bit sure rows (t3) | jointly with QOM; set by the sidecar pool | Q1 0.5 -> Q1 0.5 t3: 45 GB 26.3 -> 15.1 (x0.57), 24 GB 87.9 -> 66.8 (x0.76), 12 GB 173.6 -> 146.8 (x0.85); IFEval x0.80 / x0.87 / x0.93 | a 45 GB lever first; smaller where base misses dominate |
| COLD (unsure rows drop cold ranks >= 4) | overlaps with cf 0 and the bonus gate (same rows) | at cf 0.9 (vs u8 + t3) 24 GB x1.07 / x0.75 / x0.64; at cf 0 the only unsure row is the bonus row: 56.2 / 53.4 / 43.2 vs 61.7 / 86.1 / 83.1 = x0.91 / x0.62 / x0.52, a partial gated bonus row (the full gate x0.50 / x0.23 / x0.17) | superseded by the bonus gate where it passes (code); the chat fallback while the gate is BETWEEN there (map PASS on code 1.25 % [K]) |
| u8 on unsure rows | overlaps with the bonus gate | HYB 0 u8 t3 53.1 / 65.9 / 60.2 vs 61.7 / 86.1 / 83.1 (x0.86 / x0.77 / x0.72): the bonus row at 8 ranks | superseded by gating the bonus row |
| volatile tier (+7.8 GB purgeable) | it is the pool-size factor at 45 GB | not on the 24 / 12 GB lines | keep it on the 45 GB row only |
| the 12 GB two-pass check (pass 1 top-1 + union, x0.465 reads [K Fit12]) | overlaps with QOM (same sure rows); pass 2 adds reads back | pass 1's reads without pass 2 ("Q1 0 k1 t3"): 103 / 222 / 209 at 12 GB vs 135 / 251 / 226 (x0.76 / x0.89 / x0.92) | price pass 1 + pass 2 jointly; QOM is its FAST twin |
| JOINT, keepalive, zero-copy, pickup, forecasts | 0 bytes | exposure, not this line | - |

Two calibration notes. (a) At 45 GB on code the replay gives 60.4 (mine) / 55.4 (ledger) for EXACT against W1B's 62.4 [K]
(C 256, live kept 5.9, process bytes): calibrated within 10 %. At C 160 BigMoE's W1B night measured 169.4 [K]
(`NIGHT_1104.md`) against the 24 GB line's 213-245 (147 base slots): 0.69-0.80x, consistent with the engine's LRU +
popularity and its ~12 % larger base pool (164 slots); screen 3 closes this. (b) The map's live cut keeps 7.91 / 2.32 / 1.86
rows per check against the engine's 5.9 on code [K cap45]; since bytes per kept token are flat in kept (the kept-per-check
rows above), no kept scaling applies: the 00:35 "x1.34 on code" is withdrawn.

## 2. The floor: minimum SSD MB per kept token at 45 / 24 / 12 GB

Live-cut checks, LRU session; "mine (ledger's)" where both exist; the ledger's count is the floor. "Bonus" = the block's last
row, which today is always exact (10 ranks, both parts); "+ gated bonus row" = that row gated like a drafted row (section 4):
"all" gates every realized bonus row, "engine" gates it by the drafter's own cf for the token it emits (the ledger's
reconciliation, `ledger_lastrow.txt`; at cf 0 the two differ only by the segments that used every draft).

| line | track | rule | LRU | OPT (bound) | compulsory per request (decode / with prefill) | what binds |
|---|---|---|---|---|---|---|
| 45 GB | EXACT | all rows exact | 60.4 / 38.1 / 33.2 (55.4 / 37.7 / 33.0) | 23.9 / 19.6 / 19.7 | 40.3 / 47.3 / 60.7 ; 129.5 / 99.7 / 93.5 | code: 2/3 the per-request footprint (prompt 253 + answer 292 distinct experts per layer, 90 of them new), 1/3 capacity; chat: the session's steady state (below its per-request set: a warm session reuses across requests); the session floor is 6.6 / 10.5 / 11.8; W1B's 62.4 [K] reproduced |
| 45 GB | FAST | QOM cf 0 + t3 (bonus exact, as the map tested: PASS) | 13.5 / 19.3 / 20.2 (12.0 / 19.2 / 20.2) | 5.3 / 11.2 / 13.1 | 13.2 / 32.4 / 45.8 ; 71.5 / 76.9 / 75.5 | the bonus row (34 / 65 / 71 % of the block's pairs) and the LSB sidecar |
| 45 GB | FAST | + gated bonus row (map: PASS code / BETWEEN chat) | all 5.7 / 4.5 / 4.4 (engine 7.7 / 4.5 / 4.4) | - | 6.8 / 12.0 / 15.5 ; 45.5 / 36.1 / 30.7 | the sure rows' top-2 re-reads; **<= 10 at 45 GB is met on paper in FAST** |
| 24 GB | EXACT | all rows exact | 245.2 / 210.2 / 182.6 (212.8 / 200.4 / 175.8) | 116.9 / 93.9 / 83.5 | 40.3 / 47.3 / 60.7 | capacity: 147 slots vs a 292 + 253 working set per layer |
| 24 GB | FAST | QOM cf 0 + t3 | 61.7 / 86.1 / 83.1 (54.5 / 84.4 / 81.7) | 22.0 / 37.0 / 38.5 | 13.2 / 32.4 / 45.8 | the bonus row and the top-2 working set (115 / 85 / 80 per layer) sharing 147 slots with the bonus rows' tails |
| 24 GB | FAST | + gated bonus row | all 30.8 / 20.0 / 14.2 (engine 35.7 / 19.4 / 13.9; my next-draft proxy 31.4 / 21.6 / 15.0) | 8.8 / 7.4 / 7.3 | 6.8 / 12.0 / 15.5 | the top-2 working set's re-reads |
| 24 GB | FAST | + sure rows at top-1 + union (every bonus row gated) | 16.8 / 11.8 / 7.9 | PIN 11.7 / 9.8 / 7.4; RCP 8.8 / 8.7 / 6.9 | 3.6 / 7.8 / 9.7 | the top-1 working set and the pool policy: here a rank-class pool pays |
| 12 GB | EXACT | all rows exact | 636.8 / 621.6 / 486.6 (461.3 / 546.4 / 444.6) | 351.2 / 386.8 / 304.4 | 40.3 / 47.3 / 60.7 | 51 slots vs 292 per layer; a 16-row block's union (75) exceeds the pool, so the count depends on the check design: read as a range |
| 12 GB | EXACT | ranks 0-1 at 3 bits everywhere (T3ALL; map: code +0.59 points = BETWEEN, chat +4-5 [K]) | 562.9 / 500.1 / 399.0 | 308.1 / 293.9 / 236.6 | 33.0 / 39.3 / 52.5 | as above, 62 slots |
| 12 GB | FAST | QOM cf 0 + t3 | 135.0 / 250.5 / 225.8 (123.6 / 238.7 / 217.5) | 75.6 / 134.9 / 125.1 | 13.2 / 32.4 / 45.8 | the top-2 working set (115 / 85 / 80) vs 51 slots |
| 12 GB | FAST | + gated bonus row | all 69.6 / 75.7 / 55.5 (engine 83.1 / 72.6 / 52.9) | 34.5 / 34.2 / 25.3 | 6.8 / 12.0 / 15.5 | same |
| 12 GB | FAST | + sure rows at top-1 + union | 32.1 / 33.5 / 24.4 | RCP 29.5 / 31.7 / 22.6 | 3.6 / 7.8 / 9.7 | the top-1 working set vs 51 slots; the policy gains little here |

Covering the brief's four items:
- **Per-request working sets** [K map, means over windows; unchanged by the fix]: prompt 253 / 154 / 96 distinct experts per
  layer; answer 292 / 207 / 219 at ranks 0-9, 115 / 85 / 80 at ranks 0-1; answer experts the prompt never routed 90 / 107 /
  150 (18 / 33 / 44 at ranks 0-1); answer length 280 / 283 / 296 tokens. At 4 bits a code request's answer set is 292 x 48 x
  2.93 MB = 41 GB: it fits the 45 GB pool and nothing smaller. The ledger's part B adds the mass view: 146 / 95 / 92 experts
  per layer cover 90 % of the answer's router mass, 188 / 123 / 124 cover 95 %; 86 / 66 / 63 % of the answer's mass is on
  experts the prefill already routed.
- **Union growth with rows** [K map, EXACT, no cache; unchanged]: distinct experts per layer for blocks of 1 / 2 / 4 / 8 / 16 /
  32 / 64 rows: code 10 / 17 / 29 / 47 / 75 / 111 / 158 (singletons 100 / 72 / 52 / 38 / 26 / 16 / 9 % of the block's
  pairs); IFEval 10 / 16 / 27 / 44 / 69 / 102 / 140; multi 10 / 15 / 24 / 38 / 59 / 89 / 125. A 16-row block's union (75) is
  half the 24 GB pool and 1.5x the 12 GB pool.
- **Drafter acceptance**: the live cut keeps 7.91 / 2.32 / 1.86 of 9.65 / 3.68 / 2.96 rows per check; the wasted rows cost
  x1.8 at k 4 and x1.25 at k 16 on EXACT at 24 GB (section 1); the cf 0 case treats every drafted row as sure, and QOM cf 0
  is a map PASS on all three domains (03:26 [K]), so these FAST rows are real.
- **Record sizes**: MSB-3 part 2,162,688 B, LSB-1 sidecar 770,048 B (records3); 3-bit sure rows read the MSB only; 11-bit scales
  x0.824 on every byte [K MoonShots]; a 2-bit read is a 1.54 MB prefix of the 3-bit part (MoonShots M-34 SURE2B, 05:5x), the
  4222222222 plan -41..-52 % of bytes [K MoonShots, quality open].

## 3. The ranked path to <= 10 MB per kept token

Start: QOM cf 0 + 3-bit sure rows (map PASS at cf 0 on code and IFEval; multi's 3-bit rows BETWEEN at cf 0, PASS at cf 0.5).
Gains are the replay's ratios on that stack, joint, not products of separate factors; "halved" halves each lever's gain.
Odds [G] are for the lever passing its quality line in its domain. Costs name the owner.

### 24 GB (start 61.7 / 86.1 / 83.1)

| # | lever | gain (replay) | halved | odds | cost | exact? | novelty |
|---|---|---|---|---|---|---|---|
| 1 | **gated bonus row** (section 4): the bonus row keeps ranks 0-1 + the block's union (+ QOM's resident keep) when the drafter has a draft for its token | x0.50 / x0.23 / x0.17 -> 30.8 / 20.0 / 14.2 | x0.75 / x0.62 / x0.59 -> 46 / 53 / 49 | code: map PASS already (QOM form, 05:06 [K]); chat: BETWEEN (+1.5 points), EvalSuite's same-scores test decides: 80 % code, 40 % chat; threshold forms closed 19:38 [K]: union-only at cf 0.5 COST (+0.12 / +0.20 / +0.28), the treatment not the row; a cf 0.3 resident-keep form would be ~x0.74 / x0.61 / x0.76 [G], no better than COLD (7) on chat | masks built (two forms ran); engine: the draft after the cut exists already on 68 / 99 / 99 % of segments (ledger section 4); MTP Head for the uncut code segments (a 16th draft) | FAST | **new**; position weighting by reach is published (PSR s7, AcceptMoE) |
| 2 | **sure rows at top-1 + union** (the oea_k1 / P1-OEA pass-1 family) | alone x0.82 / x0.94 / x0.95 (50.5 / 80.9 / 79.3); with 1: 16.8 / 11.8 / 7.9 (x0.55 / x0.59 / x0.56) | x0.77 / x0.80 / x0.78 -> 36 / 43 / 38 | 45 % code, 25 % chat: the union-only form at cf 0.9 ran overnight (BETWEEN code, KILL IFEval, BETWEEN multi [K ledger 0.7]); the QOM form is screen 2 | Breakthrough: masks + map arm (5 min GPU) | FAST | adapted (OEA k1, P1-OEA) |
| 3 | **rank-class pools** (two LRU lists: experts last needed at ranks 0-1 in 70 % of the slots, exact rows' tails in 30 %, promotion on a rank 0-1 need) | x0.52 / x0.74 / x0.87 -> 8.8 / 8.7 / 6.9 (PIN x0.70 / x0.83 / x0.94; OPT room x0.29 / x0.37 / x0.51 [P ledger]) | x0.76 / x0.87 / x0.94 -> 27 / 37 / 36 | exact for any rule (it changes only what stays resident); 60 % that an engine policy keeps half the replay's gain [G] | BigMoE: a loader policy; CPU replay on the traced 24 GB arm first (screen 3) | EXACT | adapted (SLRU / 2Q; SMoE score eviction 2508.18983) |
| 4 | **11-bit scales + native layout (M-27b)** | x0.824 -> 7.3 / 7.2 / 5.7 | x0.91 -> 25 / 33 / 32 | 85 % (bitwise) | MoonShots / Leap / BigMoE: conversion Thu (a 63 GB copy: the disk call), A/B Fri | EXACT | published practice |
| 5 | **2-bit records on sure rows** (M-34 SURE2B: a 2-bit read is a 1.54 MB prefix of the 3-bit part) | after 1: x0.83 / x0.72 / x0.71 fixed, x0.78 / x0.64 / x0.63 byte-budget [P MoonShots 05:5x] -> ~6.1 / 5.1 / 4.1 after 4 | x0.92 / x0.86 -> 23 / 29 / 28 | 30 % (sure rows at 3 bits cost 0.00 % [K map]; 2 bits unmeasured; `qom_0_live_bq_t2` pre-registered, NewIdeas QUEUE 05:5x) | MoonShots: `to2bit.py` (self-tested) handed to Breakthrough; BigMoE: a 2-bit prefix read | FAST | new signal (the drafter's certainty) on a published mechanism (mixed precision by rank, 2512.17073; SliceMoE) |
| 6 | **fewer wasted rows** (the drafter: D98, Test-Time Speculation) | EXACT x0.79 on code, x0.6 on chat (live vs a perfect drafter at the same kept [P ledger A]); less on the FAST stack (the gate already removes most of a wasted row's reads) | x0.9 | MTP Head's program | - | EXACT | published (drafter quality) |
| 7 | COLD on the bonus row | at cf 0: x0.91 / x0.62 / x0.52 of the ungated stack; superseded by 1 where 1 passes; the chat fallback while 1 is BETWEEN | - | map PASS code [K] | built (Breakthrough) | FAST | new signal on a published mechanism |
| 8 | entropy coding (LOSSLESS45) | x0.87 of the store [K]; decode cost KILL at 45 GB; after 11-bit ~x0.95 [G] | x0.97 | 20 % | MoonShots | EXACT | published |

Products at 24 GB, unhalved: code 61.7 -> 30.8 (1) -> 16.8 (2) -> 8.8 (3) -> 7.3 (4) -> ~6 (5); IFEval 86.1 -> 20.0 -> 11.8 ->
8.7 -> 7.2 -> ~5; multi 83.1 -> 14.2 -> 7.9 -> 6.9 -> 5.7 -> ~4. **Halved: code 61.7 -> 46 -> 36 -> 27 -> 25 -> 23; IFEval 86
-> 53 -> 43 -> 37 -> 33 -> 29; multi 83 -> 49 -> 38 -> 36 -> 32 -> 28.** The decision-grade reading: levers 1-2 decide it (16.8
/ 11.8 / 7.9 after both, unhalved); 10 unhalved needs 3 + 4 on code, 4 alone on multi; **10 halved is out of this stack's
reach (>= 23)**: the halved line needs the 12 GB levers (2-bit resident records for the sure rows' set) or a check design whose
first pass fits the pool. Decide on the halved column; the truth sits between ~6 and ~30 until screens 1-2 and the engine
calibration (screen 3) run.

EXACT at 24 GB: 245 (213) -> 11-bit 202 (175) -> no waste x0.77 155 (135) -> policy x0.8 [G] 124 (108) -> ranks 0-1 at 3 bits
as a new reference x0.88 109 (95) -> entropy x0.95 ~104 (90) [P, unhalved]; OPT on plain EXACT 117 / 94 / 84. **~90-120 is
the EXACT 24 GB floor with every exact lever; 10 needs the request's working set (41 GB at 4 bits) to fit, which it cannot
at 24 GB.** The disk headroom at 90-120 MB per kept token is ~105-140 tok/s. Next step: screen 3 (is the engine's 24 GB
number LRU-like or better?), then 11-bit.

### 12 GB (start 135.0 / 250.5 / 225.8; T3ALL pools: 62 base slots, no sidecar)

| # | lever | gain | halved | note |
|---|---|---|---|---|
| 1 | gated bonus row | x0.52 / x0.30 / x0.25 -> 69.6 / 75.7 / 55.5 (engine 83.1 / 72.6 / 52.9) | x0.76 / x0.65 / x0.62 | as at 24 GB; map PASS code / BETWEEN chat |
| 2 | sure rows at top-1 + union (with 1) | 32.1 / 33.5 / 24.4 (x0.46 / x0.44 / x0.44) | x0.73 / x0.72 / x0.72 | the top-1 working set is the first that approaches 51 slots |
| 3 | rank-class pools | 29.5 / 31.7 / 22.6 (x0.92 / x0.95 / x0.93) | x0.96 | little room: the sure rows' set does not fit either way |
| 4 | 11-bit | 24.3 / 26.1 / 18.6 | x0.91 | |
| 5 | no sidecar: ranks 0-1 at 3 bits on every row (T3ALL) -> 62 base slots | on the QOM stack x0.86 / x0.82 / x0.83 (116.6 / 206.3 / 188.1 vs 135.0 / 250.5 / 225.8); on EXACT x0.88 / x0.80 / x0.82 | x0.93 | BETWEEN on code (+0.59 points), KILL-grade on chat (+4-5) [K map]: code only, or 4 bits on unsure rows only |
| 6 | **2-bit resident records for sure rows** (M-34) | x0.81 / x0.74 on code, x0.71 / x0.54 on chat (fixed / byte-budget [P MoonShots]); and x1.5 slots at 2 bits (1.44 vs 2.16 MB) for the sure rows' set | x0.9 / x0.85 | the one lever that changes what binds at 12 GB |
| 7 | learned replacement | OPT x0.50 / x0.45 / x0.46 on the gated stack (34.5 / 34.2 / 25.3 vs 69.6 / 75.7 / 55.5); realistic x0.8 [G] | x0.9 | |

Products on code: unhalved 135 -> 70 (1) -> 32 (2) -> 29.5 (3) -> 24.3 (4) -> ~20 (6) -> ~16 (7); **halved 135 -> 103 -> 75 ->
72 -> 65 -> 59 -> 53.** Chat unhalved ~19-26 after 4, ~13-17 with 6; halved ~90-100. **At 12 GB, <= 10 is reachable only
when the resident set holds the sure rows' working set:** today 115 / 85 / 80 top-2 experts per layer against 51 slots (77 at
2 bits); the ways are 2-bit resident copies (lever 6), top-1 routing on sure rows (lever 2) and a check whose first pass fits
(Fit12's two-pass family), all FAST. EXACT at 12 GB: 637 (461) -> T3ALL 560 (406) -> 11-bit 462 (334) -> no waste 356 (257)
-> policy 285 (206) [P]: **~250-400**, a floor set by top-10 routing over 512 experts with 10 % of them resident; the disk
headroom there is ~30-50 tok/s EXACT. The roadmap's 120-150 tok/s at <= 12 GB is a FAST number.

### 45 GB

EXACT in a warm session is 60.4 / 38.1 / 33.2 (55.4 / 37.7 / 33.0), W1B's 62.4 [K] reproduced; 11-bit and fewer wasted rows
take it to ~40 / 19 / 16 [P] against a per-request set of 40.3 / 47.3 / 60.7 (code is at its per-request footprint; chat
below it, in the session's steady state) and a session floor of 6.6 / 10.5 / 11.8: **no EXACT path to 10 at 45 GB either**
(OPT 24 / 20 / 20). FAST: QOM cf 0 + t3 13.5 / 19.3 / 20.2 -> the gated bonus row 7.7 / 4.5 / 4.4 (engine convention; map
PASS code / BETWEEN chat) -> top-1 on sure rows 2.8 / 2.8 / 2.9 -> 11-bit 2.3 / 2.3 / 2.4 [P]; halved 10.6 / 11.9 / 12.3 ->
7.2 / 9.6 / 10.2 -> 6.6 / 8.8 / 9.3. **<= 10 at 45 GB is met in FAST on the halved column with levers 1, 2 and 11-bit, and
on paper with lever 1 alone.** The metric call (W1B's process bytes vs decode bytes) matters less than written at 00:35: the
two agree at 45 GB within 10 %.

## 4. The new mechanism: the gated bonus row (and two policies re-read)

**Observation [K map].** In every certainty-gated verify block one row is never gated: the bonus row, whose output token has
no draft to compare with, so every union arm (Q1-UNION, OEA hybrids, QOM, REACH-CUT, COLD) keeps it at all 10 ranks and both
precision parts. With the live cut at t 0.3 every segment ends in one. At cf 0 (QOM's every-drafted-row-sure case) the bonus
row is 1 of 9.65 rows on code but 34 % of the block's expert pairs; on IFEval 65 %, on multi 71 % (3.68 / 2.96 rows per
check). After QOM + t3 the bonus row is most of the SSD bytes on chat and half of them on code (x0.50 / x0.23 / x0.17 when
gated, section 2).

**Mechanism.** The drafter's draft for the bonus row's token is the draft after the cut: DFlash2 drafts the whole block in
one pass and the live cut drops only inputs, so on 68 / 99 / 99 % of the map's segments (code / IFEval / multi) the engine
already has it (the ledger's section 4); only segments that used every draft (mostly code) need a 16th draft from MTP Head.
The verify block keeps m + 1 rows as today; the last row is gated by that draft's certainty exactly like a drafted row: ranks
0-1 + the block's union members when sure (+ 3-bit when the stack says so, + QOM's resident keep), all 10 when not. Accepted:
the token is the draft (right with probability ~cf, as for any sure row). Rejected: the row's gated argmax is the
correction, as at any first miss. The row is still computed, so the drafter's next state is unchanged; nothing runs one
token behind. In EXACT mode nothing changes. Token count per block is unchanged.

**It is a cf-0 lever.** With the gate file's own cf for the token the bonus row emits (the value the engine has), 1,368 /
165 / 7 of the map's 1,368 realized bonus rows clear cf 0 / 0.5 / 0.9 (`masks/bonus_masks.txt`, 07:0x): after a cut the
drafter is rarely sure of the next token, so the gain lives at QOM's cf 0, where the gate is "a draft exists". My 00:34
builds gated by the next segment's first draft (one position too far; 922 / 453 rows): they are kept but superseded by the
`g` builds.

**Threshold screens (Breakthrough 19:38, commit ba53ede; my CPU join `bytes10_bonus_cfcheck.py` -> `.out`, 20:2x) [K map].**
(b) `q1u_0.9_live_bonus9g` (7 rows) PASS within +0.1 on all three domains (McNemar 0 / 0, 5 / 6, 6 / 3): the null result.
(c) `qom_0.5_live_bonus5g` (165 rows) COST: +0.12 / +0.20 / +0.28 points (5 / 1, 48 / 41, 33 / 23; pooled 86 / 65, z +1.71).
The pre-registered reading of a cost ("the gate file's cf and the walk's disagree") was void: `dm_gate.py` builds the gate
file from the walk's own cf, one quantity. What the rows say instead. The 1,368 realized bonus rows are of three kinds: (A)
the live cut fell inside an accepted run, the gate value is the cf of a draft the verifier accepted (37 / 94 / 76 on code /
IFEval / multi); (B) the cut fell on the check's first miss, the gate value is the drafter's confidence in a token the
verifier rejects (68 / 386 / 623); (C) the end of a fully kept 15-draft check, no draft, never gated by a threshold (83 / 1 /
0). At cf 0.5 the gated 34 / 73 / 58 rows are A 21 / 27 / 19 + B 13 / 46 / 39; at 0.9, 7 in all. Joining the arm's argmax
with the baseline's (the read-out's own flip definition; it reproduces 16 / 98 / 68): the gated bonus rows flip at 9.7 % (16
/ 165) against 4.0 % (45 / 1,117) for the same kinds left exact in the same arm, +5.7 points per gated row (z 3.3); the
QOM-gated drafted rows in that arm flip at 0.2 / 0.8 / 0.5 %; and the first-miss rows with cf >= 0.5, which QOM's own
treatment gates in both arms (the ledger's "disc" rows), flip at 1.7 / 4.6 / 2.5 %, the same as the exact first-miss rows
below 0.5 (3.9 / 4.6 / 2.8 %). **So the cost is the treatment, not the row:** my mask gives the bonus row ranks 0-1 + the
block's union only, QOM's resident keep does not, and a drafter-wrong row at cf >= 0.5 costs nothing under the resident
keep. Consequences: (1) the mechanism's quality numbers are the resident-keep form's (`qom_0_live_bq`: PASS code, +1.56 /
+1.49 chat), as read above; the union-only form is closed at every threshold. (2) On chat the +1.5 at cf 0 comes from the
88-94 % of bonus rows below cf 0.5 (the live cut lands where the drafter is unsure, by construction), so a threshold on the
resident-keep form trades bytes for points along the kinds table: cf 0.3 would gate 42 / 51 / 29 % of the bonus rows (75 /
51 / 29 % of the engine's cf-0 set), worth ~x0.74 / x0.61 / x0.76 of the stack's bytes on a linear share [G] against x0.65 /
x0.23 / x0.17 at cf 0: no better than COLD on the bonus row (lever 7, x0.62 / x0.52 on chat). (3) Next, if chat needs a
PASS: not another threshold; EvalSuite's same-scores test of `qom_0_live_bq` as written, with COLD as the fallback. To
confirm the treatment reading on the map: `qom_0.5_live_bq5` (the same 165 rows with QOM's resident keep) vs
`qom_0.5_live`, prediction PASS within +0.1, ~5 min GPU; it changes no price.

**Price [P, section 2]:** on the QOM cf 0 + t3 stack, 24 GB 61.7 / 86.1 / 83.1 -> 30.8 / 20.0 / 14.2 with every bonus row
gated (the engine's convention 35.7 / 19.4 / 13.9 [P ledger]); 12 GB 135 / 251 / 226 -> 69.6 / 75.7 / 55.5; 45 GB 13.5 /
19.3 / 20.2 -> 5.7 / 4.5 / 4.4 (engine 7.7 / 4.5 / 4.4). The ledger's reconciliation shows the same gate hid inside
NewIdeas' decision-2 capture price (-64.3 / -58.5 / -62.4 % at 45 GB): that replay gated the last row by its own draft; the
map's QOM arms kept it exact (-59 / -32 / -27 %). Decision 2's price is QOM plus the gated bonus row.

**Quality [K map, overnight].** Union-only form (`qom_0_live_bonus`, Breakthrough / NewIdeas 04:41): PASS on code (+0.24
points, z +1.79), BETWEEN on IFEval (+3.59, z +8.76), KILL on multi (+4.73, z +11.38). QOM's own form, with the resident
keep on that row (`qom_0_live_bq`, 05:06): PASS on code (z -0.28), BETWEEN on IFEval and multi (+1.56 / +1.49 points, under
Q1 at 0.99's level on all three); the resident keep removed 57-68 % of the union-only form's chat cost. The 3-bit form
(`qom_0_live_bq_t3`) was running at 05:5x. So: code takes the engine convention's bytes now; chat's adoption is EvalSuite's
same-scores call, with COLD (x0.62 / x0.52 on chat at cf 0) as the fallback.

**Prior art (NOVELTY.md + web, Wed 00:2x; the ledger's section 8 concurs).** No paper found budgets the bonus position's
experts or gates it by a draft: the MoE + speculative decoding systems (DraftExpert 2607.24434, MoE-Spec 2602.16052, SP-MoE
2510.10302, MoE-SpeQ 2511.14102, SpecMoEOff 2508.21706) budget or prefetch per block and leave the last position as is. The
nearest art weights each position's expert demand by the probability the block reaches it (PSR 2610.34721 s7 with the
drafter's running confidence; AcceptMoE 2608.02989 with offline commitment rates), which also touches the last position, but
by reach, with a reroute, and never by a draft of the token itself; PEARL (2408.11850) drafts ahead for latency, not for
gating. Tag: **new** (the signal and the cut); if a reviewer reads PSR s7 as covering it, the fallback tag is "new signal on
a published mechanism".

**In a paper:** (1) the bonus-row share of a gated block's expert pairs on the three domains (section 2); (2) SSD MB per kept
token at 45 / 24 / 12 GB with and without the gate on the same replay, both miss counts (section 2); (3) teacher-forced flips
of the gated bonus row at cf 0 on the decision map, paired McNemar against the same stack without the gate (done: PASS code,
BETWEEN chat); (4) EvalSuite same-scores under two prompt orders in BigMoE's engine; (5) W1B's MB per kept token and tok/s at
24 GB, A/B. The claim: "a certainty-gated verify block has exactly one ungated row; gating it by the draft the drafter
already holds halves the SSD bytes of code checks and cuts chat's by 4-6x."

**Two policies re-read [P].** Popularity pinning (the top C/2 experts of each layer by the domain's leave-one-window-out
popularity pinned, LRU for the rest) and rank-class pools (experts last needed at ranks 0-1 in 70 % of the slots, the exact
rows' tails in 30 %, promotion on a rank 0-1 need) do nothing on EXACT: x0.94 / x1.04 at 24 GB, x0.85 / x0.81 at 12 GB, worse
at 45 GB; the OPT gap there (x0.5) is the working set itself not fitting. On the FAST stack (top-1 sure rows + the gated
bonus row) they pay: rank-class pools x0.52 / x0.74 / x0.87 and pinning x0.70 / x0.83 / x0.94 at 24 GB, because the stack's
working set (the sure rows' top-1 / top-2 experts) is small enough to protect from the exact rows' tails. The 00:35 KILL
stands for EXACT and is withdrawn for FAST (section 3, #3). Tag: adapted (SLRU / 2Q; SMoE's score-based eviction,
2508.18983, is the published form).

## 5. Three pre-registered screens for Wed 13:45 (each <= 10 min of GPU; lines written before any run)

**Screen 1: the gated bonus row on the map (FAST).** (a) `qom_0_live_bonus` (every bonus row gated at cf 0) ran overnight in
two forms with the result above: PASS code / BETWEEN chat in QOM's form. What is left for 13:45 is the pair of threshold
arms, rebuilt at 07:0x with the gate file's own cf (`bytes10_bonus_masks.py <arm> <thr> g`): (b) `masks/q1u_0.9_live_bonus9g`
(7 bonus rows gated, their pairs x0.65) and (c) `masks/qom_0.5_live_bonus5g` (165 gated, x0.59 on those). Both are null checks
of the claim that the gate above cf 0 touches almost nothing: lines, per domain: all-row flips within +0.1 points of the
ungated arm = PASS (expected), paired McNemar; a larger cost there would say the gate file's cf and the walk's disagree.
**Result (19:38, Breakthrough, ba53ede; `~/Breakthrough/data/dm_s7/results_wed/SCREEN_read_*.txt`) [K]:** (b) PASS on all
three domains within +0.1 (1.25 = 1.25, 6.42 vs 6.45, 3.86 vs 3.77 %; McNemar 0 / 0, 5 / 6, 6 / 3): the null result. (c) COST
on all three: 0.48 vs 0.36, 2.88 vs 2.68, 1.91 vs 1.63 % (+0.12 / +0.20 / +0.28; 5 / 1, 48 / 41, 33 / 23; pooled 86 / 65, z
+1.71, the same sign everywhere). The pre-registered reading of a cost was wrong (the gate file is built from the walk's cf,
`dm_gate.py`); the verdict comes from the rows (section 4): the union-only treatment costs ~+6 points per gated bonus row at
any cf, QOM's resident keep does not. Closed; nothing in sections 2-3 changes. The
00:34 builds (`_bonus9`, `_bonus5`: 453 / 922 rows, the next segment's first draft as the cf) gate one position too far;
do not run them. The mechanism's real open question is not a map screen: EvalSuite's same-scores test of `qom_0_live_bq` on
chat.

**Screen 2: sure rows at top-1 + union at live cuts (FAST; `hyb_0.9_k1_live_t3`, or its QOM form).** Sure rows keep rank 0 +
the block's union members, 3-bit on that rank; unsure rows and the bonus row as the stack says. Price [P]: 24 GB x0.82 /
x0.94 / x0.95 alone (50.5 / 80.9 / 79.3 vs 61.7 / 86.1 / 83.1), x0.55 / x0.59 / x0.56 on top of the gated bonus row; 12 GB
x0.76 alone, x0.46 with the gate. The union-only form at cf 0.9 ran overnight: BETWEEN code, KILL IFEval, BETWEEN multi [K
ledger 0.7]; the QOM form (resident keep on sure rows) is the 13:45 arm. Lines: all-row flips <= Q1 at 0.99's level (1.34 /
5.42 / 3.69 %) = PASS; <= 1.5x = BETWEEN; paired McNemar vs the top-2 twin. Prediction [G]: BETWEEN on code (+0.3-0.6 points
over top-2), KILL on IFEval, BETWEEN on multi. ~5 min GPU (NewIdeas' dm_oea_masks parse() takes `hyb_<thr>_u<k>`; the k1 form
needs ks 1).

**Screen 3: the W1B 24 GB baseline against this replay (CPU read-out of BigMoE's 13:45 run, no extra GPU).** From the traced
arm, split the process SSD bytes into prefill and decode, and compute decode MB per kept token on the six prompts. Reference:
code 24 GB EXACT LRU 245 (in-loop count) / 213 (the ledger's count), no kept scaling. Lines: within 0.8-1.25x of 213-245 =
the replay is calibrated at 24 GB and section 3's prices stand; below 0.8x = the engine's LRU + popularity beats plain LRU and
lever 3 is partly cashed already (re-price it from the trace); above 1.25x = the engine reads more than its unions (prefetch
waste or the K-73 phases) and that is the first byte lever at 24 GB. Prediction [G]: 0.75-0.9x (BigMoE's C 160 night read
169.4 [K] with a 12 % larger base pool).

Not screens (they need builds first): 11-bit (Thu), the rank-class pool (a loader policy), the learned policy.

## 6. Caveats

- One capture: the map's 36 windows (12 per domain, ~280 answer tokens each, HumanEval / IFEval / multilingual prompts). Code
  is one task family; the per-request footprints of agent turns with long prompts are larger (prompt 253 of 512 already); the
  per-request share falls with longer answers (the ledger's part B: ~55-59 MB per token at 1K tokens, ~15 at 4K, extrapolated).
- Two miss counts (the head of this file): mine is x1.09 / x1.15 / x1.38 the ledger's at 45 / 24 / 12 GB; the ledger's is
  the reachable one. At 12 GB a 16-row block's union exceeds the pool, so the number depends on the check design: a range.
- Plain LRU, prefill warm-up, one session per domain: the engine's LRU + popularity reads fewer base misses (BigMoE's C 160
  night: 0.69-0.80x of the 24 GB line) and the mixed protocol costs +4-14 % at 45 GB, nothing at 24 / 12 GB.
- The 00:35 version's bug (section 7) touched every byte level; the working sets, union growth, bonus-row shares, mask builds
  and the mechanism ranking did not use the mask and stand.
- Halving is per lever; the halved products in section 3 are products of halved factors, harsher than halving the product.
  Decide on the halved column.
- Nothing here is GPU-timed; bytes only. Exposure (the other BROKEN line) is Fable A's and BigMoE's.

## 7. Correction note (Wed 4 Nov 2026, 06:47-07:3x EET)

- **What was wrong.** `bytes10_replay.py` line 135 (now 137): `keep[sure, :ks] = True; keep[~sure, :ku] = True` on a
  `[rows, 48 layers, 10 ranks]` array selected layers 0..ks-1 at all ranks instead of ranks 0..ks-1 on all layers. EXACT read
  base records on layers 0-9 only; sure rows read all 10 ranks on layers 0-1 only; the LSB sidecar mask (`lsb[:, :, :2]`) was
  right. Found by NewIdeas at 03:4x (`scripts/bytes10_bug_check.py`: one code block's EXACT union 120 / 79 / 0 / 0 experts
  on layers 0 / 9 / 10 / 47 against the true 80 / 89 on layers 10 / 47); fixed in their copy
  (`scripts/bytes10_replay_fix.py`, rerun 1,128 s, `data/ledger/bytes10_replay_fix.out`) and here at 06:5x (my rerun
  reproduces it; the 00:35 outputs are `bytes10_replay_bug.*`). No other script has the bug (NewIdeas searched FableGod,
  Breakthrough, MoonShots, NewIdeas); the screen masks index rows and layers explicitly and are right.
- **What changed (00:35 -> now, code / IFEval / multi).** EXACT 45 GB 18.7 / 15.4 / 12.6 -> 60.4 / 38.1 / 33.2; 24 GB 96.5
  / 92.8 / 82.5 -> 245 / 210 / 183; 12 GB 186 / 208 / 174 -> 637 / 622 / 487. FAST start (QOM cf 0 + t3) 24 GB 32.6 / 46.2 /
  45.8 -> 61.7 / 86.1 / 83.1. The gated bonus row x0.66 / x0.48 / x0.47 -> x0.50 / x0.23 / x0.17 at 24 GB. The 24 GB path
  9.6 / 10.3 unhalved (19 / 24 halved) -> 7.3 / 7.2 / 5.7 unhalved (25 / 33 / 32 halved) with one more lever (the pool
  policy). "45 GB is not a cache problem" -> 2/3 footprint, 1/3 capacity; the "43.8 x 7.91 / 5.9 = 59" calibration and the
  x1.34 kept scaling are withdrawn (W1B's 62.4 is the warm-session replay itself). QOM's cut is domain-, not pool-dependent.
  The policy KILL stands on EXACT and is withdrawn on the FAST stack. The 12 GB path 16.6 / 20.9 / 17.7 -> 24.3 / 26.1 /
  18.6 unhalved; EXACT 12 GB ~90-120 -> ~250-400; EXACT 24 GB ~45-60 -> ~90-120.
- **What survives.** The mechanism ranking (gated bonus row, then top-1 on sure rows, then 11-bit); the novelty tag; kept
  per check flat for EXACT; wasted rows x1.25-1.8; 24 / 12 GB as capacity misses with OPT at ~x0.5; the working sets, union
  growth and bonus-row shares; the screens' masks.
- **The ledger comparison the 06:47 check-in was for.** After the fix, NewIdeas' independent replay and mine agree within
  10-15 % at 45 / 24 GB on every shared row (EXACT, QOM cf 0.5 / 0, with and without 3-bit) and differ by more than 20 % only
  at 12 GB EXACT on code (637 vs 461, x1.38): the miss-count convention (in-loop vs before the block), not the data. The
  ledger's convention wins (the engine uses resident members first), so the floor tables carry both and read the ledger's
  as the floor. Rows where only my replay exists (OPT, PIN, RCP, the bonus variants, the compulsory floors, MIX, the kept
  curves) are in-loop counts: pessimistic by the same x1.09 / x1.15 / x1.38.
- **Masks.** `masks/q1u_0.9_live_bonus9g` and `masks/qom_0.5_live_bonus5g` built at 07:0x with the gate file's cf for the
  bonus row (NewIdeas' note on the proxy): 7 / 165 of 1,368 bonus rows gated. The 00:34 `_bonus9` / `_bonus5` builds stay for
  the record; do not run them.

Sources (web pass, Wed 00:2x): DraftExpert https://arxiv.org/abs/2607.24434 ; MoE-Spec https://www.arxiv.org/pdf/2602.16052 ;
SP-MoE https://arxiv.org/abs/2510.10302 ; MoE-SpeQ https://arxiv.org/abs/2511.14102 ; SpecMoEOff https://arxiv.org/html/2508.21706v2 ;
HOBBIT https://arxiv.org/abs/2411.01433 ; FlashMoE https://arxiv.org/html/2601.17063v1 ; DALI https://arxiv.org/html/2602.03495v1 ;
SMoE https://arxiv.org/pdf/2508.18983 ; local routing consistency https://arxiv.org/html/2505.16056v3 ; PSR, AcceptMoE, PEARL as
in TRIED/NOVELTY.md.

## 8. Screen 1(b) / (c) results (Wed 4 Nov 2026, 19:38-20:3x EET)

Breakthrough ran the two threshold arms on the map at 19:28-19:38 (36 windows each, 0 row-count warnings; commit ba53ede;
`~/Breakthrough/data/dm_s7/results_wed/`). (b) `q1u_0.9_live_bonus9g` vs `q1u_0.9_live`: PASS on all three domains within
+0.1 points, the null result on 7 rows. (c) `qom_0.5_live_bonus5g` vs `qom_0.5_live`: COST on all three, +0.12 / +0.20 /
+0.28 points (pooled McNemar 86 / 65, z +1.71). My CPU join of the arm's argmax with the baseline's
(`bytes10_bonus_cfcheck.py` -> `.out`; it reproduces the read-out's 16 / 98 / 68 flips) sorts the 1,368 realized bonus rows
into three kinds and shows the cost is the union-only treatment, not the row: the 165 gated rows flip at 9.7 % against 4.0 %
for the same kinds left exact (+5.7 points per gated row, z 3.3), while the first-miss rows with cf >= 0.5, which QOM's own
resident keep gates in both arms, show no excess over the exact ones (1.7 / 4.6 / 2.5 vs 3.9 / 4.6 / 2.8 %). The
pre-registered alternative (gate file vs walk cf) was void: `dm_gate.py` builds one from the other. Nothing in sections 2-3
changes: the lever's quality was already read from the resident-keep form (`qom_0_live_bq`: PASS code, +1.56 / +1.49 chat).
Closed: the union-only form at every threshold, and threshold forms of the resident keep as a route to a chat PASS (cf 0.3
would be ~x0.74 / x0.61 / x0.76 [G], no better than COLD on chat). Open, as before: EvalSuite's same-scores test of
`qom_0_live_bq` on chat. Cost of this pass ~0.15 % of the week; CPU only, no messages, nothing committed.
