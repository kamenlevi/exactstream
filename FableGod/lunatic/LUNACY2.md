# LUNACY2: 1000 tok/s on this MacBook. What would have to be true?
Lunatic, round 2 of the BREAKTHROUGH PUSH (Tue 3 Nov 2026, 17:0x-17:4x EET). Think first, read after: the budget and the
route list came from PHYSICS.md and first principles; files were opened only for kill tests. Tags: [K] measured,
[P] arithmetic, [G] judgment. Modeled gains are unhalved; halve them when deciding. CPU only, 6 Python runs
(lu_1000.py three times, one of them after a unit fix; lu_resync.py twice; lu_copyshare.py once). Outputs:
lu_1000.out, lu_resync*.out, lu_copyshare.out.

## 0. The answer in eight lines
1. **The budget:** at 1000 tok/s a kept token gets 1.0 ms, at most 270 MB of RAM reads, at most 12 MB of SSD reads and
   at most 14 G MAC at the bf16 peak [P]. It touches 4.47 GB of weights [P], so every byte read must serve at least
   16.6 kept tokens [P]. Today a byte serves about 4 [P].
2. **What binds first is the 125B itself, not the Mac.** On code, 13.5 % of rows are ones the 125B is unsure of [K].
   That caps any drafter's agreement near 0.98 [G], so one pass can keep at most ~40-50 tokens.
   - Even with the whole model in RAM and the draft hidden, one stream tops out near **315 tok/s EXACT** and
     **350 tok/s FAST** [P].
   - So single-stream 1000 on general code is **x3.2 (EXACT) and x2.8 (FAST) beyond physics**.
3. **1000 exists in only two places:** width (many streams sharing one expert read) and copying (agreement >= 0.999).
4. **Route W, one user is a batch:** FAST at 8-12 streams reaches **~880-1,270 tok/s** aggregate [P].
   - Its user-side test passed: **63 % of Kamen's agent generation in the last week happened while >= 8 sessions were
     generating at once** [K, transcripts, 10-min windows].
5. **Route C, the check verifies a document:** EXACT single-stream reaches **~940-1,130 tok/s** on near-exact copies
   [P], in agent edits.
   - It **failed its own 10-minute test on Kamen's traffic.** Copies are 37.3 % of the lines his agents write into
     files, but only **6.2 % of all their output** [K, a week of transcripts].
   - That is +4 % on average and x1.35 on file writes [P].
   - An Edit-tool agent already skips most copied text, so C raises tok/s more than it speeds up a task.
6. **Route X, EXACT owns the Mac:** EXACT reaches ~1000 aggregate **only if the EXACT store fits in RAM.**
   - Today the store is 71.8 GB against 64 GB [P], so the SSD caps EXACT near **285 tok/s at any width** [P].
   - EXACT's wall at 1000 is capacity, not bandwidth.
   - With the store resident (all-3-bit, Kamen's call) and the horizon fix, EXACT reaches ~540-670 at 8-12 streams
     and ~940 at 24 [P].
7. **At the 1000 point the M5 Pro has no idle resource left** [P]:
   - Expert reads, the trunk's compute (46-68 % of bf16) and the FAST SSD cap (1,335) bind within ~30 % of each other.
   - Moving work between engines can't make a breakthrough there, because every engine shares one 270 GB/s bus.
8. **Killed this round with numbers:**
   - Parareal decoding (one answer as many streams): after a one-token divergence, the 125B's other continuation
     drafts only 48.9 % of the text [K]; it needs >= 98 %.
   - 2-bit RAM residency: dead in TRIED on quality and on decode speed.
   - Trees and top-3 at the 1000 line: one row costs about a whole token's budget [P].
   - Route C for this user: see line 5.
   - **What's left:** W (FAST, alive, the strongest) and X (EXACT, a decision plus a footprint diet).

## 1. The 1000 tok/s budget (one user, the full 125B, all 512 experts)

### 1a. Per kept token
| | 1000 tok/s needs | today (16-row checks, kept 10.5 on code) | gap |
|---|---|---|---|
| time | 1.0 ms [P] | 15.7 ms [K] (63.6 tok/s) | x15.7 |
| RAM reads | <= 270 MB [P] (270 GB/s [K]) | ~1.08 GB [P] (3.44 GB trunk + 76.6 x 48 records per check) | x4.0 |
| SSD reads | <= 12 MB [P] (12 GB/s [K]) | 62.4 MB at C 256 [K], 37.9 at C 320 [K] | x5.2 / x3.2 |
| expert groups (10.15 us each, AUDIT fit [K]) | <= ~70 [P] (trunk ~30 % of the time) | ~350 [P] | x5 |
| compute | <= 14 G MAC at peak, ~8.4 at 60 % [P] | 7.7 G MAC [P] (16 rows x 5.05 G / 10.5) | fine today; binds at the line |
| draft | hidden behind the check | 12.0 ms serial per cycle [K] = 1.1 ms per kept | must overlap |
| reuse of each byte read | >= 16.6 kept tokens [P] (4.47 GB touched per token: experts 1.03 + trunk 3.44) | ~4.1 [P] | x4 |

- **The pass model** used below: trunk 12.7 ms of bytes + 3.6 ms fixed + 0.32 ms per row of bf16 compute at 60 %
  [P/G], plus experts at 4.5 ms + 10.15 us per (expert, layer) [K fit].
  - At 16 rows that gives 21.5 + 41.8 ms [P], the top of PHYSICS' measured ranges (17-23 / 20-40 [K]).
  - So the tok/s below are conservative by up to ~1.3x [G].
  - The model assumes the SSD and GPU overlap perfectly. Today they don't: 42.6 of the check's 84.2 ms are gate spin
    [K]. That is why it prints 140 tok/s for today's shape against 63.6 measured.
- **Trunk MACs per row:** 2.7 G is a [G] (range 2.0-3.4). Routed experts are 2.35 G [P] (10 x 48 x 4.9 M).

### 1b. What binds first, and by how much
1. **Single stream: agreement, which is the 125B's own uncertainty.**
   - Code rows are 86.5 % sure / 12.0 % mid / 1.5 % tie [K, lu_alpha]. DFlash2 gets mid rows right 65.9 % and ties
     30 % [K]; the MTP head gets 86 % of close calls at depth 1 [K, KWALL].
   - Past 1/(1-alpha) tokens kept per pass stop growing, while the union keeps growing as N^0.47 [K]. So wider checks
     stop paying.
   - Best case per agreement (whole model in RAM, draft hidden; lu_1000 B) [P]:

| agreement | what it means | best tok/s | at rows |
|---|---|---|---|
| 0.909 [K] | today's DFlash2, strict | 139 | 16 |
| 0.9485 [P] | sure rows perfect (horizon fixed) | 184 | 32 |
| 0.982 [G] | ideal drafter: sure 100 %, mid 90 %, tie 60 % | **315** | 64 |
| 0.974 [P] (FAST) | relaxed acceptance R2 + sure rows perfect | 258 | 64 |
| 0.974 + Q1-UNION, groups x0.6 [G] (FAST) | the passing FAST levers, stacked | **351** | 64 |
| 0.99 + Q1 (FAST, beyond any data) | | 565 | 128 |

   Single-stream 1000 is x3.2 beyond EXACT's ideal-drafter bound and x2.8 beyond the FAST stack. Even an agreement of
   0.99 that nothing has shown leaves it x1.8 short.
2. **With width: expert reads and trunk compute.**
   - At 3 bits an expert group is 2.15 MB in 10.15 us = 212 GB/s, 78 % of the bus [P]. Groups are bytes.
   - Once 8-12 streams merge, the trunk's per-row compute adds 0.32 ms per row (0.2 with int8) [P]. At 256-384 rows the
     trunk is 40-60 % of the pass [P], and the pass runs at 46-66 % of the bf16 peak [P].
3. **For EXACT: capacity.**
   - The EXACT store (3-bit records + the LSB sidecar) is 24,576 x 2.92 MB = 71.8 GB [P], on a 64 GB machine.
   - At C 320 the SSD supplies 37.9 MB per kept token [K], x1.11 at 8 streams [K, lu_batch_ssd]. That caps EXACT at
     **285 tok/s whatever the width** [P].
   - FAST without the sidecar (M-26, 8.1 MB per kept [K]) lifts the cap to 1,335 [P].
4. **For an agent: prefill compute.**
   - A prompt token costs 5.05 G MAC [P/G], so prefill tops out at 1,663 tok/s (60 % of bf16) or 2,661 (int8) [P].
   - A 3K-token tool output takes >= 1.1-1.8 s, against 0.3 s to write 300 tokens at 1000 tok/s [P].
   - At 1000 tok/s decode, an agent's turn is ~80 % prefill [P], and that part is compute-bound, not byte-bound.

## 2. Three routes most people would call impossible, which the physics allows

### Route W: one user is a batch (FAST ~1000 aggregate; EXACT per stream, capped by Route X)
- **Assumption broken:** "one user means batch 1". Width is the only cheap dimension on this Mac:
  - Eight code streams side by side touch 236.9 experts per layer per 16-row step, about the same as 128 consecutive
    rows of one answer (228.0) [K, lu_union].
  - On a single-user laptop that batch dimension is 100 % empty today.
- **Mechanism:** one pass verifies 32-row blocks from 8-12 of the user's streams. Each expert is read once per layer
  for every row that routes to it.
  - FAST byte cuts ride along: Q1-UNION across streams (a sure row keeps its top 2 plus whatever the merged union
    already loads, which at 8 streams is most of its top 10), the sidecar drop (M-26), and an int8 trunk.
  - The drafter runs batched on the ANE, overlapped with the check.
- **Numbers** (lu_1000 D) [P]:

| streams x 32 rows | code, Q1 + sidecar drop | mixed domains | mixed + int8 trunk | bf16 load |
|---|---|---|---|---|
| 8 | 961 | 883 | 1,042 | 46-54 % |
| 12 | 1,126 | 1,043 | 1,273 | 54-66 % |

  - The SSD cap is 1,335 at C 320 without the sidecar [P].
  - EXACT on the same pass shape: 285 [P], the SSD capacity cap (Route X).
- **Why it sounds crazy:** "a laptop cannot serve 1000 tok/s of a 125B." It can't for one stream. For one person's
  work it can.
- **What must be true:**
  1. The user keeps >= 8 streams busy. Measured below: 63 % of Kamen's generation already runs at width >= 8 [K].
  2. Q1-UNION and the sidecar drop pass same-scores (EvalSuite). Without them W falls to ~350-420 [P, the
     "store in RAM" EXACT rows].
  3. The kernel stays ~10 us per group at 5-9 rows per group [G].
  4. Batched drafting gives >= ~2,000 draft tokens/s without touching the GPU [P: ~1.4-2 rows per kept x 1000].
  5. 8-12 streams' GDN states (112 MB each [P]) and KV fit beside the pools.
- **Falsification test (<= 10 min):**
  - CPU: replay 8 streams x 32 rows on the DAY9 capture, mixed domains, with NewIdeas' Q1-UNION masks
    (dm_oea_masks.py), and count expert groups per kept token. **PASS <= 60, KILL > 80** [P: 80 groups ~ 800 tok/s
    with the int8 trunk].
  - The user side: count concurrently busy sessions in a week of agent transcripts. **KILL for this user if the
    activity-weighted median width is < 4.**
- **User-side result, run 17:28 (shell census of the same week of transcripts).** A session counts as generating in a
  window if it wrote an assistant message there.
  - Concurrency is bimodal: per active 10-min window the median is 2, p75 9, p90 10, max 30. Per hour the median is 4,
    p90 12 [K].
  - Weighted by activity, **75 % of generation happens at width >= 4, 63 % at >= 8 and 16 % at >= 12** [K].
  - **PASS for this user** (activity-weighted median width >= 8): most of his tokens are produced exactly when 8-12
    streams are busy.
  - The quiet 1-2-stream stretches (nights, single loops) run at single-stream speed, ~300-430 FAST [P].
  - Proxy caveat: these are Claude sessions. Moved to the local 125B, the pattern holds only if the same agents run
    there.
  - The CPU replay with Q1-UNION masks has not run yet.
- **Abandon if:**
  - the CPU replay kills;
  - or the first GPU merged-check screen (4 streams x 32 rows against 1 x 32, C 320) gives < 1.6x aggregate;
  - or both FAST byte cuts fail same-scores;
  - or Kamen's agents can't run on the local model at all (the orchestrator's call).
- **Owners:** MoonShots (replay), BigMoE (merged checks), NewIdeas (cross-stream masks).
- **Novelty:** adapted. Continuous batching and OEA's batch union are published; the drafter-certainty gate is new
  (NOVELTY.md), and applying it across one user's streams is the new setting.

### Route C: the check verifies a document (EXACT single-stream ~1000 on copy spans)
- **Assumption broken:** "a check verifies guesses". In agent work much of the output re-emits the user's own text:
  file rewrites, moved code, the unchanged lines around an edit, repeated paths. There the draft is a document, not a
  guess.
- **Mechanism:** diff-anchored drafting.
  - The draft is a pointer into the context (the old file, an earlier output), checked in 256-512-row passes.
  - At the first mismatch the 125B's own token is kept, and normal DFlash2 drafting writes the edit.
  - A suffix match re-anchors the pointer after the edit, and long checks resume.
  - EXACT by construction: every kept token is the 125B's greedy token.
- **Numbers** (lu_1000 C, whole store in RAM) [P]:

| agreement on copy spans [G] | 128 rows | 256 rows | 512 rows |
|---|---|---|---|
| 0.995 | 550 | 601 | 518 |
| 0.999 | 700 | **941** | **1,126** |

  - With the EXACT store streamed at C 320, the SSD caps it near ~500 [G] (a ~30 % union miss rate at 512 rows), so
    EXACT C past 500 also needs Route X. FAST bytes don't.
- **The blend that decides its worth:** if copies run at ~1,000 and fresh text at ~300 (FAST), a 50 % copy share gives
  ~460 tok/s and 90 % gives ~810 [P, harmonic]. C moves the average only where agents mostly copy.
- **Why it sounds crazy:** "EXACT 1000 tok/s, one stream, the full 125B, on a laptop." The physics allows exactly this,
  and only here.
- **What must be true:**
  1. Agreement >= 0.999 on spans the old text covers: the 125B copies rather than paraphrases [G, untested].
  2. Copy share >= 50 % of an agent's output tokens.
  3. 512-row checks cost what the group fit says (attention and GDN rollback included).
- **Falsification test (<= 10 min, CPU):** replay Kamen's own agent traffic. Take a week of Claude Code transcripts:
  the Write/Edit tool inputs against the text already seen in each session.
  - Measure the share of output in >= 64-character exact copies of earlier context, and the agreement of
    diff-anchored drafting on those spans (lu_resync-style).
  - **PASS: copy share >= 50 % and agreement >= 0.995. KILL: share < 30 %** (blended gain < 1.3x [P]).
  - This is the workload's shape, not the 125B's own text. The 29 "agent" prompts in EvalSuite are the second look.
- **Result, run 17:26 (lu_copyshare.py):**
  - Data: 147 transcripts from the last 7 days, 2,687 Write calls, 3,605 edits, 81.2 M characters of assistant output.
  - File writing is 16.5 % of all output [K].
  - Lines already seen earlier in the session make up 37.3 % of it (Write 35.3 %, Edit 49.0 %) [K], which is
    **6.2 % of all output** [K].
  - **KILL for Kamen's current workload:** +4 % on average [P].
  - x1.35 on file writes alone [P]: a published speculative-edits lever for NewIdeas, not a breakthrough.
  - The line-level proxy counts any line seen earlier, so it slightly overstates copying.
  - **The deeper reason:** copy tokens carry no new information. An Edit-tool agent already avoids emitting them, so
    C raises tok/s without making tasks faster.
- **Abandon if:**
  - the share is < 30 %;
  - or the 125B paraphrases (agreement < 0.995 on covered spans);
  - or 512-row checks cost > 1.3x the fit.
- **Owners:** NewIdeas (copy share), BigMoE (long checks), Leap (512-row kernels), EvalSuite (an edit workload in the
  measured mix; HumanEval can't see this: COPYGATE was x1.04 there [K]).
- **Novelty:** published (speculative edits, prompt-lookup and SAM decoding). The 512-row EXACT check on a local 125B
  is engineering.
- **W x C is one coding task forked into parallel, copy-heavy file edits.** That is where one task could pass
  1000-2000 tok/s [P]. It needs both routes to pass, so it is moot for Kamen while copies are 6 % of output.

### Route X: EXACT owns the Mac (EXACT ~1000 aggregate only if its store fits in RAM)
- **Assumption broken:** "EXACT's wall is bandwidth". At width, EXACT hits the SSD at 285 tok/s because its 71.8 GB
  store can't be resident [P]. FAST passes 1000 at the same width only because it drops 19 GB of sidecars.
- **What must be true (all of it):**
  1. Kamen re-bases EXACT on a store that fits: the all-3-bit plan, 52.8 GB [P]. It needs same-scores:
     - t3_all on the decision map is code BETWEEN, chat +4-5 points [K];
     - EvalSuite decides.
  2. The night process gets ~58 GB: 52.8 + trunk 3.4 + states ~2 [G]. That needs the non-pool lifetime footprint cut
     from 8.5 GB [K, BigMoE HANDOVER:607] to <= ~3 GB, and macOS wired memory <= ~3 GB [G].
  3. The horizon fix (agreement 0.909 -> 0.9485 [P]).
  4. 16-24 merged streams.
  5. An int8 trunk.
- **Numbers** (lu_1000 F, mixed domains, resident, int8 trunk) [P]:

| streams x 32 rows | 8 | 12 | 16 | 24 |
|---|---|---|---|---|
| today's drafter | 354 | 442 | 511 | 618 |
| horizon fixed | 539 | 673 | 779 | **942** |

- **Why it sounds crazy:** "EXACT 1000 needs a bigger machine." It needs the whole store resident and ~24 busy
  streams, and nothing else.
- **Falsification test (<= 10 min):** footprint arithmetic from measured numbers.
  - Inputs: BigMoE's lifetime non-pool peak (RUSAGE_INFO_V4) plus macOS wired memory (vm_stat).
  - **KILL if the two can't fall below ~6 GB together.**
  - Also KILL if all-3-bit fails same-scores on chat: then EXACT can't re-base, its store stays 71.8 GB, and EXACT
    at width stays near 285.
- **Abandon if** either kill fires.
  - Width is X's weak point. The census puts only 16 % of Kamen's generation at >= 12 streams [K], so X's realistic
    EXACT is the 8-12-stream row, **~540-670 tok/s** with the horizon fix [P].
  - The 24-stream ~940 is rare for this user.
- **Owners:** Kamen (re-base decision), BigMoE (footprint), EvalSuite (all-3-bit same-scores).
- **Novelty:** engineering plus a definition decision. It is listed because it is the only EXACT path to ~1000 the
  physics leaves.

## 3. Abandon criteria in one table
| route | 10-min test | PASS | abandon | status |
|---|---|---|---|---|
| W: one user is a batch | 8 x 32 merged replay with Q1-UNION masks; busy-session census | <= 60 groups per kept; activity-weighted median width >= 8 | > 80 groups; activity-weighted median < 4; GPU merge < 1.6x at 4 streams; both FAST cuts fail scores | alive: the user census PASSED (63 % of generation at >= 8 streams [K]); the union replay is still to run (MoonShots / NewIdeas) |
| C: the check verifies a document | copy share + diff-anchored agreement on a week of agent transcripts | share >= 50 %, agreement >= 0.995 | share < 30 %; agreement < 0.995; 512-row checks > 1.3x the fit | **KILLED for this workload** (6.2 % of output) [K] |
| X: EXACT owns the Mac | footprint arithmetic (non-pool + wired) | <= 6 GB | > 6 GB with no path down; all-3-bit fails chat scores | alive, needs Kamen's re-base decision |

## 4. Dead this round, with why
1. **Parareal decoding** (one EXACT answer as many streams: the 125B writes future sections from drafted prefixes,
   and an exact sweep stitches them).
   - The idea was first-principles: width is cheap, so parallelize one answer.
   - **Kill test** (lu_resync.py): two engine variants decode the same prompts. After their first divergence, use one
     text as the draft source for the other (suffix-match drafting).
     - b12 vs b12l, 343 prompts, 6 domains: **48.9 % drafted** [K] (humaneval 57.0, math 64.2, mgsm 56.9,
       ifeval 26.2, multi 33.1, agent 31.1), 1.9 tokens per check.
     - REF2-T vs REF2-CT, 222 IFEval prompts: 27.9 % [K].
     - The route needs >= 98 %. A single near-tie flip rewrites the rest of the answer.
   - Even at 99.5 % the pricing gives only 361 tok/s [P, lu_1000 E], barely above the single-stream bound (315),
     because the lookahead rows are real union.
   - Neighbours in TRIED: Jacobi / lookahead KILL (:620), SoT PARKED (:872), Hogwild KILL (:880).
2. **Whole model RAM-resident at ~2-2.3 bits (QTIP / AQLM / VQ)**, my first-principles "what is stored" route.
   TRIED had already killed it:
   - quality: held-out output error 2.2-2.5x MSB-3 (B10 / B3 / TRL-2, :279-284);
   - speed: 3INST decode-bound, -0.5 % tok/s at 0.45x the bytes (R-17, :312);
   - 2-bit is only x1.22 at 16 rows, with ~10 us per expert that doesn't shrink with bits (L1-30, :298).
3. **Single-stream 1000 on general code, either track:** beyond physics (x3.2 EXACT, x2.8 FAST; section 1b).
4. **Trees, top-3 use and deep drafts at the 1000 line.**
   - Even with everything in RAM, a marginal row costs ~1.2-1.5 ms [P]: 38.5 more experts per layer from 16 to 32
     rows, x48 x10.15 us / 16 rows, plus 0.32 ms of trunk. That is a token's whole budget.
   - A tree row is kept with p ~0.2 [K: top-2/3 add 23 points on mid rows].
   - Today's 7.8 ms per row is mostly SSD; the RAM-resident regime is x5-6 cheaper and still not cheap enough.
5. **"What runs where" as a breakthrough at 1000:** <= 30 %.
   - The ANE, CPU and GPU share one 270 GB/s bus.
   - Spare engines help only once compute binds, i.e. the trunk at 256-512 rows: int8 / ANE trunk +18-22 % [P].
6. **Reusing states of text the 125B already wrote** (to kill agent prefill): EXACT needs identical prefixes, and
   position-independent reuse is FAST and published (CacheBlend). Handed off.
7. **Route C as a 1000 route for Kamen:** killed by its own test (section 2, 6.2 % of output is copies). It survives as
   a published x1.35 file-write lever.

## 5. Assumptions broken this round
1. "One user is batch 1." (W)
2. "A check verifies guesses." (C)
3. "EXACT's wall is bandwidth." At width it is capacity. (X)
4. "Faster single-stream decoding is the road to 1000." The 125B's own uncertain rows forbid it.
5. "1000 is a decode number." For an agent it is a prefill-compute number.
6. "Attack the SSD wall." At 1000 the SSD must be out of the loop (<= 12 MB per kept), which only FAST bytes or a
   resident store achieve.

## 6. Reproducibility and method note
- `lu_1000.py` holds every [P] in sections 1-2: the union curve from lu_union.out, agreement from lu_alpha.out, the
  group fit from AUDIT_350, SSD from SSD_WALL and M-26.
- `lu_resync.py` holds the Parareal kill. Its header line names REF2-T / REF2-CT even when run on b12 / b12l.
- `lu_copyshare.py` holds the Route C test (Kamen's transcripts, aggregates only, nothing leaves the Mac).
- The Route W census is a shell pipeline (grep / awk, no Python): assistant-message timestamps per transcript, unique
  per minute, counted per 10-min window. The minute list is in the session scratchpad (minutes.txt).
- Think-first record:
  - Generated before reading: the budget, the agreement cap, width, copy spans, Parareal and low-bit residency.
  - Changed by reading: TRIED killed low-bit residency; my replay killed Parareal; the group fit (AUDIT) and the SSD
    numbers (SSD_WALL, M-26) set the prices; the capacity wall (Route X) came out of the pricing.
