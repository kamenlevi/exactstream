# GRANDMA_1117: a fresh-mind hunt for a >= 3x on the 125B (Tue 17 Nov 2026, 13:30-14:30 EET)

Brief: find a mechanism NOT on the closed list that gives >= 3x toward 775 tok/s lossy at quality on code, with the full
125B as the verifier, on this M5 Pro. Desk work plus one CPU replay of the existing t30b dumps (0.4 s, < 0.4 GB). No GPU
job; no other file touched; nothing committed.
Marks: [K] measured (source named), [P] arithmetic on measured parts, [G] judgement.

## 0. Verdict
1. **No mechanism found that gives >= 3x, and none that reaches 775 on code.** I say it plainly.
2. **New measurement (section 3):** the physical ceiling of TODAY's head on this Mac. I replayed the real 127-node t30b
   trees along the decode walk, for every tree size from 8 to 127 rows. Each size gets its kept count AND its sequential
   draft depth. I priced it at 100 % of every measured peak, with perfect byte / MMA overlap:
   - HE+ peaks at **553 tok/s** with no new lossy lever and a 0.19 ms-per-level draft. It reaches **664** with
     union-clamped 64-row trees, and **808** only if drafting is free.
   - MBPP stays at **336 / 351 / 397** in the same three cases.
   - For 775, MBPP needs **x1.8-3.2 the head's kept** in every case (x1.83 even at 100 % with a free draft and int8
     everywhere), so MBPP cannot reach 775 with this head.
3. **The x3 over Saturday's ~205 (HE+) is physically allowed only as a stack of four known engineering items:**
   - the check at ~90 % of its byte floor (x1.44);
   - the draft at <= 0.37 ms per level (x1.38);
   - 64-row trees with a clamped union (x1.27);
   - a draft at <= 0.19 ms per level (x1.15).
   Together they give **612** [P]. None of them is a simple mechanism, none is a Saturday item, and one of them is a
   lossy routing change with no quality gate yet.
4. **Best two real levers:**
   - **(1) The expert phase of the check.** It is 26.3 ms against a 16.4 ms byte floor, because bytes, MMA, decode and
     loads run serialized.
   - **(2) The draft cost per level.** At today's 1.1 ms per level, bigger trees LOSE even with a perfect check. At
     <= 0.19 ms, a 64-row clamped tree is worth +20 %.
   Their numbers and decisive tests are in section 7.

## 1. The budget at 775, from zero
- 775 tok/s = **1.290 ms per kept token**. At 280 GB/s that is **0.361 GB of DRAM reads per kept token** at 100 %,
  covering everything: check, draft and KV.
- Today's lossy check alone is 6.74 GB (C6 + int8 dense; Leap check25_bound_real.txt) for 16.91 kept (tree-32 + CC(1)):
  **0.399 GB per kept**. 775 x 0.399 = **309 GB/s**, above the measured 280-300, even with ZERO draft and ZERO overhead.
  So 775 at today's kept is not an engineering target on this DRAM (PUSH800 section 1 says the same).
- The draft reads 31 pops x 163-178 MB = 5.1-5.5 GB per tree-32 block today [P], about 80 % of the check's bytes. That
  is why it is the second wall.
- So any >= 3x needs one of three things:
  - (a) kept per pass up 1.2-2.6x at the same bytes;
  - (b) bytes per pass down that much at quality;
  - (c) one weight stream serving more than one pass's rows.

## 2. A conservation law, and a three-question test for any "simple mechanism"
**Law.**
- A pass streams each weight once. It can only serve rows whose input tokens exist before the stream reaches layer 1:
  the token at t+1 needs the 125B's layer-48 output at t.
- Every pipeline, stagger, rotated lap, half pass, deferred row or interleave I could write down does one of two things:
  - (i) it makes the later rows speculative on the earlier rows' outcome, which is a bigger tree on the same K(R)
    curve;
  - (ii) it streams the weights again, which costs more bytes.
- Hence, for every shape that verifies with the full 125B:
  **tok/s <= BW_eff x kept(R) / bytes(R) per pass, minus the draft's own serial time.**
  - kept(R) is the expected walk of rows fixed before the pass.
  - The draft's serial time is set by depth x cost per level.
- Fable B's "theorems of greedy walk x unified memory" (IDEAS_1117c_carryover.md) is the same statement, reached from
  the check side.

**So a candidate breaks the frame only if it answers YES to one of these:**
- (a) Does it give the drafter the 125B's DECISION at drafted positions for far less than a row?
  - Closed on record (most of it today): partial width, partial depth / lens (VERDICT-DEPTH), in-check repair,
    landmarks, session value prediction, placeholder rows, prefix eyes, the 2B, sibling states.
- (b) Does it cut bytes per pass at quality?
  - 2-bit: KILL. Top-6: KILL (2.98x MSB-3's error). 1c-tiers: FAIL. DENSE-3: FAIL. W4A8 experts: NO (3 % error).
  - The DROP form (-20 % of expert bytes) is the one open item, gated after Saturday.
- (c) Does one stream serve two passes' rows?
  - Impossible by the law.

Every idea below was run through this test.

## 3. NEW [K]: the frontier of today's head, kept AND draft depth by tree size
**Method.**
- Data: BigMoE's tree_t30b_{heplus24,mbppbank}_n127k8.npz.
- Each check is a fresh best-first tree of the first K pops (K + 1 rows with the root), with exact acceptance. The walk
  advances by kept, exactly as Fable C's pool_cursor_1117c.py mode A.
- It reproduces the record: HE+ 16.30 / MBPP 9.05 at 32 rows, and HE+ 20.77 at 64.
- "levels" = distinct depths among the K nodes, i.e. sequential head forwards under level batching. This is optimistic
  for today's per-pop drafting (31 forwards for tree-32).
- "ran out" = the tree held no child of the last right node: the head never explored where the walk went. "Wrong" = it
  explored there and guessed wrong.

| rows | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 96 | 127 |
|---|---|---|---|---|---|---|---|---|---|
| HE+ kept / check | 6.90 | 9.39 | 11.30 | 14.42 | **16.30** | 18.91 | **20.77** | 22.77 | 23.87 |
| HE+ draft levels / check | 6.6 | 9.8 | 12.5 | 17.2 | 20.8 | 26.3 | 30.3 | 36.1 | 39.7 |
| HE+ ran out / wrong | .82/.18 | .73/.27 | .68/.32 | .57/.43 | .50/.50 | .44/.56 | .40/.60 | .34/.66 | .35/.65 |
| MBPP kept / check | 5.56 | 6.77 | 7.50 | 8.47 | **9.05** | 9.74 | 10.21 | 10.91 | 11.31 |
| MBPP draft levels / check | 6.1 | 8.4 | 10.3 | 12.8 | 14.6 | 16.9 | 18.6 | 21.1 | 22.7 |

**Cost model at a stated efficiency of every measured peak.** It is an upper bound by construction.
- Bytes = dense 2.13 GB + C6 experts 4.60 GB x union(rows) / union(32). The union comes from Leap's treek_union.txt
  (73.2 / 109.7 / 156.5 / 211.1 experts per layer at 16 / 32 / 64 / 128 rows).
- CLAMP: rows beyond 32 route inside the 32-row union, so expert bytes stay at 4.60 GB. This is Leap S2's lossy form,
  measured at 1.15-1.19x of the 32-row call at 64 rows.
- Compute: 4.85 G MAC per row (dense 3.6 per Fable B + ~1.25 of diet experts [G]), at 14 T MAC/s bf16 or 22 T int8.
- Check = max(bytes / 280 GB/s, compute) + cores 1.2 ms.
- Draft = levels x cost per level. Overhead 1.5 ms.
- Kept x 1.04 for CC(1) (16.91 / 16.26).
- Calibration: the measured 38.9 ms 16-row check, 1.1 ms per level and 3 ms overhead give **211 tok/s HE+**, against
  the orchestrator's Saturday ~205 [K-calibrated].

Best tok/s over tree size (rows at the best size in brackets):

| efficiency | union | draft per level | HE+ | MBPP | 775 needs kept x (HE+ / MBPP) |
|---|---|---|---|---|---|
| 100 % | grow | today 1.10 ms | 347 (24) | 244 (12) | 2.23 / 3.18 |
| 100 % | grow | path A 0.37 | 492 (32) | 311 (16) | 1.58 / 2.49 |
| 100 % | grow | QAT 0.19 | 553 (48) | 336 (16) | 1.40 / 2.31 |
| 100 % | grow | free | 643 (48) | 367 (16) | 1.21 / 2.11 |
| 100 % | CLAMP | today 1.10 | 359 (64) | 244 (12) | 2.16 / 3.18 |
| 100 % | CLAMP | path A 0.37 | 569 (64) | 316 (64) | 1.36 / 2.45 |
| 100 % | CLAMP | QAT 0.19 | **664 (64)** | 351 (64) | 1.17 / 2.21 |
| 100 % | CLAMP | 0.10 | 725 (64) | 371 (64) | 1.07 / 2.09 |
| 100 % | CLAMP | free | 808 (64) | 397 (64) | 0.96 / 1.95 |
| 100 % | CLAMP + int8 everywhere | free | 886 (96) | 424 (96) | 0.88 / 1.83 |
| 90 % | CLAMP | path A 0.37 | **530 (64)** | 292 (64) | 1.46 / 2.65 |
| 90 % | CLAMP | QAT 0.19 | 612 (64) | 321 (64) | 1.27 / 2.41 |
| 90 % | CLAMP + int8 everywhere | free | 802 (96) | 384 (96) | 0.97 / 2.02 |

Reading it:
- **(i) At today's 1.1 ms per level, big trees lose even with a perfect check.** HE+ peaks at 24-32 rows. S1's
  deferral was right, and big-tree work should wait for the draft.
- **(ii) With the draft at <= 0.19 ms per level and the clamp, 64 rows is best, at +20 %** over the best grown tree.
- **(iii) 775 on HE+ appears only in rows that need every one of these together:** 100 % of every peak, the lossy
  clamp, int8 activations everywhere (W4A8 failed its 3 % screen) and a near-free draft.
- **(iv) MBPP is flat at ~300-420 under every engine.** It is the head's knowledge: the true-state chain is only 12.83
  at 32 rows (Lunatic). 775 on MBPP needs a different drafter, not a faster Mac.

## 4. NEW [K]: the walk is bimodal, an Amdahl cap on one-sided levers
- HE+ tree-32: **37 % of checks keep <= 8 tokens, but they hold only 12 % of the tokens** (p10 4 / p50 13 / p90 32).
- MBPP: 54 % of checks hold 31 % of the tokens.
- Every check costs about the same, so:
  - a lever that only speeds the long, easy checks (copy spans, bigger chains, docstrings) is capped at 1 / 0.37 =
    **2.7x** on HE+, even if those checks became free;
  - a lever that only fixes the short, decision-bound checks is capped at 1 / 0.63 = **1.6x**;
  - on MBPP the caps are 1.85x and 2.2x.
- **A real 3x must lower the cost of EVERY check (the engine) or raise kept everywhere (the drafter's information).**
  Nothing narrower can.

## 5. Candidates (not on the closed list), each audited
**C1. Big union-clamped trees, made to pay by a cheap draft (64-96 rows).**
- Mechanism: route the off-spine rows 33-64 inside the 32-row union (Leap S2). Those rows then cost only MMA, which
  runs under the byte stream at the floor. ADAPTIVE K picks 64 rows when path_p calls for it.
- Plausibility [K]:
  - S2: 1.15-1.19x of the 32-row expert call at 64 rows.
  - K63 kept 20.77.
- Number [P]: +20 % at 100 % (553 -> 664) with a 0.19 ms-per-level draft, but only +3 % at today's 1.1 ms (347 -> 359).
- Decisive test: EvalSuite's tier-0 replay (LOSSY_GATE 0a, A and F_hi) of the clamp on rows 33-64, after Saturday.
  The engine side is already measured (S2).
- Quality risk: rows the walk reaches with re-routed experts. These are rare by construction, but the rate is unknown.
- Audit: a real lever, but conditional on C3 / lever 2. x1.2, not 3x.

**C2. A LEARNED harm judge in place of close-call's margin rule (AutoJudge, arXiv 2504.20039).**
- Mechanism: a small classifier on the 125B's row state accepts the head's mismatched token when forcing it does not
  change the program's test result. The labels are mined from forced-token decodes. CC(1) stays as the floor.
- Plausibility:
  - AutoJudge's own claim: 25+ accepted tokens per cycle on LiveCodeBench at -2 % pass@1.
  - Here, 66 % (HE+) / 74 % (MBPP) of check-ending misses are comment or docstring wording [K, Lunatic
    IDEAS_1117c_channels.md].
- Number:
  - Bounded above by the measured WHOLESALE inert replay: trees x1.17-1.30 HE+, x1.63-1.79 MBPP [K, WEEK98 Mon 18:1x;
    that form is the recorded quality KILL CD16-DROP].
  - A judge accepts only a subset: [G] x1.1-1.2 HE+, x1.2-1.4 MBPP, IF quality holds.
- Decisive test: none in 10 min on CPU. The labels need GPU decodes with forced tokens, and the CPU ceiling above is
  already measured.
- Quality risk: HIGH on the record.
  - d2 FAILED (MBPP -6.0 pt).
  - Wholesale inert acceptance looped and drifted.
  - Comments steer ~3/4 of the next lines.
  - Tier-0's A line (0.9775 token agreement) kills any judge by design, so it needs Kamen's call on the gate itself.
- Audit: not 3x. It is the only lever on record with a large MBPP ceiling. Park it for after Saturday, and only if
  Kamen accepts a gate on task scores.

**C3. The draft off the critical path, or nearly free.**
- Forms: an SLC-resident persistent head kernel; a CPU head on SME from L2; SSD-style pre-drafting (arXiv 2603.03251).
- Number [P, frontier]: a free draft is worth +22 % over 0.19 ms per level at 64 clamped rows (664 -> 808). A draft at
  0.10 ms per level is worth +9 % (725).
- Kill / limits:
  - SSD is the POOL drafter: KILL, x0.88 without a re-root (Fable C, 12:4x).
  - Heads small enough for the SLC (<= 24-28 MB, MoonShots M-45) lose kept: S20 SMALLHEAD, a 2x smaller head keeps
    x0.69-0.80.
  - The persistent kernel is MoonShots' line already.
- Audit: a +10-20 % lever at the floor, not 3x. It is also lever 2's ceiling.

**C4. A sparse-dense head: keep the capacity, cut the bytes per level.**
- Mechanism: replace the head's dense MLP with a fine-grained MoE of the SAME total parameters and ~1/4-1/5 active.
  Bytes per level drop from ~94 MB (Q8H32) to ~20-30 MB without shrinking capacity. That is different from SMALLHEAD,
  which shrank the capacity.
- Number [P, frontier]: draft path A 0.37 -> ~0.17 ms per level, so HE+ at 90 % with the clamp goes 530 -> ~620
  (+17 %) at equal kept.
- Decisive test: a Sandbox98 2B analog (MoE MLP at 1/4 active vs the control, tree-32 kept). PASS >= x0.97. About 2 h,
  so it is not a 10-min screen.
- Quality risk: none to the output (drafter only). The risk is to kept.
- Audit: not 3x. A candidate for the draft wall after Saturday.

**C5. A dynamic vocabulary screen per check.**
- Mechanism: 4-8K tokens per check, from the prompt, the answer so far, the 125B's recent top-k and a global core,
  instead of the static 32K screen.
- Number [P]:
  - 32K costs +31 MB per level over 8K [K, MTP Head 22:3x], and the static 8K screen costs 1.5-2.4 pt of kept.
  - A dynamic screen that keeps 32K's coverage at 8K's bytes saves ~0.11 ms per level, for +3-5 %.
- Test: CPU coverage on the dumps (are the true next token and the head's top-8 inside the screen?). It needs the
  prompt ids, which the tree dumps lack.
- Audit: small.

**C6. Two-tempo exact checking.**
- Mechanism: clamp now; re-run the accepted prefix exactly inside the next check's batch; roll back on a flip.
- Kill: it adds ~17-21 re-run rows per check, plus their union. That makes 64-row checks compute-bound, and the lossy
  line does not need it.

**C7. Forms from 2025-2026 that collapse into closed items.** Each was checked against the law and the closed list:
- Mirror-SD (Apple, arXiv 2510.13161): early-exit-triggered rollouts on the NPU = mid-layer signals (VERDICT-DEPTH:
  lens AUC 0.69-0.73 through L32, decisions 20-29 %) + the ANE (too slow here).
- Speculative speculative decoding = the pool (KILL).
- Concurrent cursors in one answer (Hogwild!-style) = the code||commentary family (KILL: the form fails, ~x1.3).
- MoE-aware verification (EVICT 2605.00342, EcoSpec 2607.12696, MoE-Spec 2602.16052) = ADAPTIVE K + DROP + S2. The
  frontier's grow -> clamp gap (553 -> 664) is their whole territory. EVICT reports 1.21x over EAGLE-3, an
  MoE-unaware baseline.
- Commit-on-confidence: every token still needs a 125B row, so passes stay set by the misses. Dead by the law.

## 6. My own numbers, audited (harshest first)
**Optimistic:**
- perfect byte / MMA overlap;
- every peak at 100 % (or 90 %), including the cores at 1.2 ms and the overhead at 1.5;
- draft depth = distinct depths. Today's per-pop drafting does 31 forwards for tree-32, against 20.8 levels;
- CC(1)'s x1.04 assumed at every tree size;
- the clamp's quality assumed free;
- the QAT 0.19 ms-per-level head does not exist yet (S13 FAILED; S16 is running).

**Pessimistic:**
- 280 GB/s (300 measured streaming is worth +7 % at most);
- a fixed tree size per run (per-check ADAPTIVE K adds +5-10 % at the floor [K, ADAPTK: +4.8 % at the 25 ms floor
  with 16 / 32]);
- union growth taken from full top-10 routes (the lossy diet may grow more slowly).

**Net:**
- Grant all the pessimistic corrections: HE+ at 90 % with the clamp and a path-A draft goes ~530 -> ~600.
- 775 then still needs ~x1.3 the head's kept, and MBPP stays ~330.
- **The conclusions do not flip.**

## 7. Plain answer and the best real levers
**Plain answer.**
- Nothing found here reaches 3x as a single mechanism.
- 775 on code is beyond this head on this Mac: MBPP is short by 1.8-3.2x of kept under every engine, and HE+ touches
  775 only at the physical limit of every part at once.
- The only multiplier with that kind of size is NEW information for the drafter: the 125B's decisions at drafted
  positions for well under a row each. The record closes it nine ways (section 2), and I found no tenth.

**Lever 1: the expert phase of the check (largest single gap).**
- Size: 26.3 ms against a 16.4 ms byte floor at 32 rows; bytes 16.4 + MMA 6.1 + decode 3.1 + loads 2.9 run serialized
  (ZERO).
- Number [P]: moving it to ~90 % of its floor saves ~8 ms per 32-row check. The whole check at ~90 % of its floor
  (16 rows: 38.9 -> ~22 ms) takes Saturday's HE+ from 211 to **303 (x1.44)** at today's draft.
- Decisive test: Breakthrough's pre-registered one-layer MoE-block bench against its 85 % line (in flight; first read
  0.52). A >= 0.85 x 280 GB/s quiet read of one layer's expert phase at the real 32-row union (46 experts per layer,
  the C5 / C6 mix) is the go / no-go for the whole engine half.

**Lever 2: the draft cost per level.** It decides whether anything bigger than 32 rows ever pays.
- Number [P]:
  - At the measured 38.9 ms 16-row check, a fused level at f <= 0.1 ms on the path-A head (0.37 ms per level) takes HE+
    211 -> **261 (x1.24)**.
  - With lever 1, it reaches **418 (x1.98)**.
  - It is also the switch for C1 (+20 % at 64 clamped rows) and C4 (+17 %).
- Decisive test (~10 min, GPU, MoonShots): one persistent fused level of the Q8H32 head at 1-2 rows. PASS: <= 0.10 ms
  fixed per level (today 0.55).

**The x3 over Saturday, written out.**

| step | lever | HE+ tok/s |
|---|---|---|
| 1 | lever 1 (x1.44) | 303 |
| 2 | + lever 2 (x1.38) | 418 |
| 3 | + C1 clamp at 64 rows (x1.27) | 530 |
| 4 | + a 0.19 ms-per-level head (x1.15) | **612** |

- Every factor is known and owned (Leap / Breakthrough, MoonShots, MTP Head, EvalSuite for the clamp's gate).
- None is a Saturday item.
- MBPP on the same stack is ~321.

**Honest confidence:**
- ~90 % that no single mechanism inside the stated constraints gives >= 3x (the law plus the frontier). The residual is
  an information source nobody has found.
- ~95 % that 775 on code (MBPP included) is out for this head on this Mac.

## Sources
- Records: WEEK98.md (Tue 17 entries), BREAK98 / 98b / 98c, PUSH800, LOSSY_GATE.
- Leap: check25_bound_real.txt, treek_union.txt.
- Fable C: IDEAS_1117c_walls.md + pool_cursor_1117c.py.
- TRIED.md: top-k, Jacobi, judge entries.
- Literature (authors' claims, not replicated):
  - [AutoJudge](https://arxiv.org/html/2504.20039v1)
  - [Mirror-SD](https://arxiv.org/html/2510.13161v2)
  - [Speculative Speculative Decoding](https://arxiv.org/html/2603.03251v1)
  - [EVICT](https://arxiv.org/pdf/2605.00342)
  - [EcoSpec](https://www.alphaxiv.org/abs/2607.12696)
  - [MoE-Spec](https://arxiv.org/html/2602.16052v1)

## Appendix: the replay, so the numbers can be re-run
- The full scripts (frontier_1117g.py, frontier_1117g_b.py) and their outputs (frontier_1117g.out, frontier_1117g.json,
  frontier_1117g_b.out) are in the session scratchpad,
  /private/tmp/claude-501/-Users-kamen-OjoX/3c9dd883-f531-48f1-879c-0636665ab84b/scratchpad/grandma/. They are not
  copied into deepdraft, by the brief.
- The scratchpad can be cleared, so the walk core and the cost model are reproduced below.
- Run with `~/BigMoE/.venv/bin/python -I -B`.

```python
import numpy as np, os
CE = os.path.expanduser("~/BigMoE/data/deepdraft/ceiling")
def load(tag):
    tz = np.load(f"{CE}/tree_t30b_{tag}_n127k8.npz")
    fi, an, node, parent, dep, ot = (tz[k] for k in ("fi", "an", "node", "parent", "depth", "on_true"))
    key = fi.astype(np.int64) * 10**7 + an.astype(np.int64); o = np.lexsort((node, key))
    key, node, parent, dep, ot, fi, an = (x[o] for x in (key, node, parent, dep, ot, fi, an))
    st = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]); en = np.r_[st[1:], len(key)]
    return {(int(fi[s]), int(an[s])): (dep[s:e], ot[s:e], parent[s:e]) for s, e in zip(st, en)}
def check(pool, K):                      # a fresh best-first tree of the first K pops, exact acceptance
    dp, ot, pa = pool; idx = np.arange(min(K, len(dp)))
    dok = set(dp[idx][ot[idx]].tolist()); a = 0
    while a in dok: a += 1
    return a, len(set(dp[idx].tolist()))  # (accepted depths, draft levels)
def walk(pools, K):                      # Fable C's mode-A walk: start at anchor a0 + 1, advance by kept = a + 1
    kept = lev = n = 0
    for f in sorted({f for f, _ in pools}):
        an = sorted(a for ff, a in pools if ff == f); p = 1
        while p < len(an):
            a, lv = check(pools[(f, an[0] + p)], K); kept += a + 1; lev += lv; n += 1; p += a + 1
    return kept / n, lev / n
# cost: check = max((2.13 + 4.60 * union(rows) / union(32)) / 280, rows * 4.85e9 / MAC) + 1.2 ms (CLAMP: union ratio 1
# above 32 rows); tok/s = kept * 16.91 / 16.26 / (check + levels * c_level + 1.5); union from Leap's treek_union.txt.
```
