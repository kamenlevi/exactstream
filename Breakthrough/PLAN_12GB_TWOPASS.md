# 12 GB plan: the top-2 two-pass check (Breakthrough, the 12 GB lead while Fit12 sleeps; Mon 2026-11-02 20:5x)

[K] = measured tonight on the decision map (36 windows, D-02a, 4-bit experts) or replayed on DAY9's routing capture.
[P] = estimate. Sources: ~/FableGod/CACHE.md (mission 20), CLOUD_ANSWER_12GB_V2/V3/V4.md, SCOREBOARD 593bfa4..b43930c,
evidence in ~/FableGod/decision_map/out/ (CERT_read.txt, RANKCACHE_read.txt, DEFER_read.txt, REPLAY_bias.txt).

## 1. Working point
- Top-2 two-pass check, theta 0.8 on code (orchestrator 20:2x). Map [K]:
  - certified code rows flip 8 / 2976 = 0.27 % [0.14-0.53]: PASS (line <= 1 %)
  - all code rows 1.63 %: a floor of 1.16 % at ANY theta (0.99 has no certified flip) + 0.47
  - pass 2 runs in 55 % of code blocks (walk: 8.7 kept per check)
- Chat is where theta 0.8 fails [K]:
  - certified rows flip 2.55 % (IFEval) and 4.63 % (multi): BETWEEN
  - all rows 8.51 % / 7.04 %, against 4.50 % / 2.70 % at theta 0.99 (the int4 twin's profile is ~5 % on chat)
  - pass 2 still runs in 91 % / 79 % of chat blocks, so the low threshold buys chat little speed
- Proposed fix (pre-registered b43930c; on the GPU tonight): a DOMAIN-ADAPTIVE certificate, p1 >= 0.5 where the
  drafter is sure (cf >= 0.99) AND its draft = pass 1's argmax, else p1 >= 0.95. CPU half [K]:
  - code: pass 2 in 59 % of blocks with 0 pass-1 certified flips (theta 0.8: 55 %, 10 flips)
  - chat: 0.8-1.3 % certified flips at 91-97 % pass 2 (theta 0.8: 2.8-4.6 % flips)
  - It works because the drafter is sure mostly on code.
- THE QUALITY CONTRACT for 12 GB is Kamen's rule: task results (EvalSuite same scores on math / code / instruction tests),
  not a token-deviation budget (orchestrator 21:1x; memory small-ram-bar-task-results). The exact default stays exact.
  The floor is the cut's context effect (every cut row feeds the exact rows), as with Q1's gated run (1.34 %) and the
  twin; no certificate removes it.
- CHAT MODE (cloud v5, orchestrator 21:1x): chat gets nothing from the two-pass but a second dense pass (pass 2 in 91-97 %
  of blocks; 18-23 tok/s). Chat runs ONE exact pass per check plus the gate instead (no pass 1): ~25-31 tok/s [P], and
  its tokens are exact, so the chat deviation question disappears. A per-check mode gate picks two-pass or exact from
  the drafter's confidence (the drafter is sure mostly on code); the gate's rule and its misroute cost are a CPU replay
  item (T6).

## 2. Bytes and time at 12 GB (C 51 slots)
- Replay, theta 0.8, LRU, rows after the first miss holding wrong drafts [K]. FLOOR's 1.3x replay pessimism in brackets:
  - pass 1: 0.71 GB per check (0.54)
  - pass 2: 0.33 GB (0.26)
  - total: 1.04 GB (0.80)
- Pass-1 reads are the binding resource: they sit on pass 1's per-layer chain (routing -> read -> compute), which no
  forecast or residency removes. Pass-2 reads can be issued at once from pass 1's routing and hide under compute.
- Dead levers [K]:

  | lever | why it is dead |
  |---|---|
  | top-5 pass 1 | 2.05 GB per check |
  | resident-only pass 1 | rows with both top-2 resident at all 48 layers: 0.2 % |
  | forecasting pass 1's reads | TRIED ceiling: precision 0.68, lead-2 recall <= 40 % |
  | rank-aware cache | pass-1 bytes 0.71 -> 0.67 GB, total +1.6 % |
  | any residency policy | Belady's MIN on pass 1's stream still reads 0.38 GB per check |
  | deferred anchor (no pass 2) | loses on code |

- Small levers [K]:
  - the sure-drafter arm at theta 0.8: pass 2 55 -> 52 % of blocks
  - the pipelined bonus: -4 to -6 % dense passes on code, -9 to -16 % on chat
- Cycle [P]: cloud v3's 99 ms per check assumed 0.45 GB of pass-1 reads. The measured 0.54-0.71 GB adds 7-20 ms, giving
  106-119 ms per check:

  | kept per check | tok/s |
  |---|---|
  | 10.5 (today) | 88-99 |
  | 12 | 101-113 |
  | 13 | 109-123 |

- Targets (cloud v5, orchestrator 21:1x): 120 = theta 0.8 + dense <= 22 ms + kept >= 12 + pass-1 bytes <= 0.66 GB
  (cycle 91-97 ms), inside the measured band; 150 = dense <= 20 + kept 13 + pass-1 bytes <= 0.45 GB, i.e. the 3-bit
  ranks 0-1 arm or the whole exact byte set below. Cloud's honest ceiling with this week's pieces: ~140 at kept 13.
- Cache-policy work at 12 GB is STOPPED (rank-aware KILL, pinning dead): the gap to Belady is cold-miss structure (new
  experts per block), not policy. The exact byte set (C 64 via the 4-bit drafter's freed slots, block trimming at the
  drafter's first low-confidence draft, early-layer prefetch under pass 2) replayed on CPU [K] (dm_byteset.py, 9098370):
  BETWEEN, 0.708 -> 0.493 GB per check but kept -6 %, judged 0.524 raw / 0.40 calibrated. The live engine's own
  running-confidence cut (0.3) already does most of the trimming; the new levers are C 64 (x0.85, cold misses do not
  scale as 1/C) and prefetch (oracle, -5 %): 60.2 MB per kept token = 0.52 GB per check at kept 8.6.
- Two quality-cleared levers on pass-1 bytes (Tue 11:1x, map):
  - 3-BIT PASS 1 (M-18's MSB-3 records, bytes x ~0.75): PASS. The 3-bit certificate (t3_top2_all) and its two-pass mixed
    run at theta 0.8: certified code flips 0.13 % (4-bit: 0.27 %), pass 2 in 57 vs 55 % of code blocks, all rows inside
    the 4-bit run's CI.
  - P1-K1 (NewIdeas: pass 1 = each row's top-1 + the experts its check's union already reads; OEA, arXiv 2511.02237):
    PASS end to end (certified code flips 7/2997 = plain top-2's 7/3019), pass-1 reads x0.465 (0.708 -> 0.329 GB per
    check at C 51).
  - Stacked [P]: ~0.25 GB of pass-1 reads per check at C 51 (the stack is not map-tested yet). Fit12 times both (and
    the 3-bit variants) in its 14:3x trace lock.

## 3. What a check does (engine semantics)
1. The drafter proposes 15 tokens (with each draft's cf).
2. Pass 1 runs the 16 block rows with experts at ranks 0-1 only, reading records on demand per layer.
3. Certify each row from pass 1's logits: p1 >= theta, or the adaptive rule. Walk the block on the certified rows'
   argmax, and stop at the first uncertified row or the first miss.
4. Pass 2 (only if the walk stopped on an uncertified row): run those rows, up to the first miss, with all 10 experts,
   attending to the committed context. Its reads are issued at once from pass 1's routing (ranks 2-9); routing can shift
   a little in pass 2.
5. Commit the accepted tokens plus the bonus. The KV keeps pass 1's entries for certified rows and pass 2's entries for
   the rows it ran.

Caveat [K]: the map runs this as ONE pass with per-row k, so a certified row after an uncertified one sees that row's
EXACT entries. In the engine it saw pass 1's top-2 entries. The engine's own exactness test (T2) checks the gap.

## 4. Build and test order (each with its line)
- T0 quality, Tue (Breakthrough + EvalSuite): a free-running two-pass greedy decode on the streamed model.
  - How: top-2 per token; keep the token if p1 >= theta, else re-run the row exact and replace its KV.
  - Configs: theta 0.8 for code, theta 0.95 for chat (the adaptive arm's chat behaviour).
  - EvalSuite scores it against REF (same-scores, decision 4). The adaptive arm needs the drafter, so it waits for T2.
- T1 engine: pass 1 reads only ranks 0-1 records, with per-row k in the streamed engine. Fit12's engine facts (Mon
  23:15): B12's routing row holds at most 92 ids (bigmoe_v3f/gated.py), so pass 1 posts ranks 0-1 only, 2 ids per row
  and 32 per gate row for 16 rows; pass 2's rows (<= 4 in 93-100 % of blocks) fit B12's 4-row calls; after pass 1's
  keep-n commit the certified tail's GDN state is gone, so option (a) ends the check at pass 2's last row + its bonus
  (kept -7.9 % code at theta 0.8, -6.2 % with the adaptive certificate, -2-5 % chat [K map]) and (b) one per-row-k call
  recovers it later. Chat mode gate: Fit12's history rule (2, 8) measured IFEval 10.4 -> 13.4-13.9 tok/s on MTP with
  code unchanged; the drafter-cf gate (first draft cf >= 0.9) goes in next.
  - theta > 1 must equal the exact engine bitwise; theta 0 must equal top-2 on every row bitwise.
- T2 engine: certify, walk, pass 2 on a row subset, KV patching.
  - Token agreement with the map's mixed runs on the 36 windows, reported (no line: the semantics differ in item 3).
- T3 timing at 12 GB (C 51, quiet lock), per-check components:
  - PASS: pass 1 <= its compute + bytes / 13 GB/s + 10 %, and the cycle at today's kept gives >= 90 tok/s
  - KILL: < 70 tok/s
- T4 levers on the cycle:
  - pass 2's reads issued at once (cloud v3)
  - dense <= 22-27 ms per pass (Leap D1)
  - the ANE draft in pass 2's shadow (PASS <= 2 ms exposed)
  - kept >= 12 at 12 GB (MTP Head / NewIdeas)
- T5 bytes (the biggest lever): the 3-bit pass-1 records (M-18 arm on the map, Tuesday's BIG window; then a 3-bit
  all-rows run for its certificate). The exact byte set's replay (C 64 / trimming / prefetch) beside it.
- T6 chat mode: the per-check mode gate (two-pass vs one exact pass) replayed on CPU: misroutes, bytes, passes per token.

## 5. Open
- The floor (1.16 % code, 4.5 % IFEval at any theta) is the deciding question, and it is EvalSuite's.
- Chat runs at 2.5 kept per check, so it stays SSD- and compute-bound near ~25-40 tok/s [P] whatever the threshold.
- Owners (orchestrator 21:0x): Fit12 has the engine side, T0-T2 code (handed Mon 21:0x by peer send); Breakthrough keeps
  the map quality runs, T5 and the analysis.
