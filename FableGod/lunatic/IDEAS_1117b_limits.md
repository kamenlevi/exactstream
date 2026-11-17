# IDEAS_1117b_limits: the limits of free rows, located decisions and head knowledge (Lunatic, Tue 17 Nov 2026, 08:2x-09:0x EET)

BREAK98b, my lens: the limits of (a)-(c). What is the least an extra row can cost on this Mac, and how much can a head
of <= ~60 MB per level hold? Desk work plus three CPU screens on BigMoE's real 127-node t30b trees, the topk cells and
MoonShots' 2B / 4B per-row picks (no GPU; < 1 GB, one process, seconds each). Marks: [K] measured (source named),
[P] arithmetic on [K], [G] judgement. Prices are at Leap's C6 + int8 bound as in IDEAS_1116c / IDEAS_1117_floor
(32-row check 25.2 ms + overhead 1.5 + draft 3 ms per 31 levels): own chain-32 462 / 232 tok/s, tree-32 566 / 314-321
(HE+ / MBPP).

## 0. The answer in six lines
1. **(a) An exact extra row costs >= ~0.25-0.37 ms on this Mac, not 0.05.**
   - Its dense part alone is 3.73 G multiply-adds: 0.27 ms at the MEASURED matrix-unit peak of 14.0 T MAC/s bf16, or
     ~0.15 ms with int8 activations (est.).
   - The new experts it pulls into the union add 0.06-0.10 ms; the cores 0.04 ms.
   - So ~96 rows cost >= 24-36 ms, 5-7x the <= 5 ms the brief asks for. The ANE and the CPU add no usable compute.
2. **At that floor, more own-information rows are worth <= +3 %.** HE+ peaks at 48 rows: 566 -> 581 tok/s at the
   bound (m = 0.2). Bigger trees pay only at m <= 0.10 ms (64 rows: 633) or 0.05 ms (96 rows: 683), below the dense
   compute floor of an exact row [K limit_rows.out].
3. **(b) Two located decisions cannot fit 1.3 passes.**
   - A decision needs the 125B's last layers on the right prefix (VERDICT-DEPTH). A dependent second decision
     therefore needs a second weight stream: >= 0.30 pass of dense bytes + 0.27-0.44 of the tail's experts. So two
     decisions cost >= ~1.6-1.7 passes.
   - Inside ONE pass they are rows, i.e. (a). Under the head's calibration, best-first IS the optimal placement of
     those rows.
4. **The head's candidates, not the selection, cap HE+ near 23.7.** Even hindsight-perfect selection from the head's
   127 candidates reaches 18.48 / 21.39 / 22.45 / 23.35 / 23.70 at 31 / 47 / 63 / 95 / 127 rows. So kept 24 needs
   ~200+ own-candidate rows [G, extrapolated]. MBPP's candidates are exhausted at 11.28 by ~64 rows [K limit_rows.out].
5. **(c) Knowledge is not what the head lacks, and bigger independent models barely add it.**
   - The same head WITH the true state is right at 68.5 % of the chain's first misses.
   - The independent 2B / 4B, reading the true text with full depth, are right at only 52 / 58 % (HE+) and 50 / 57 %
     (MBPP), no better than the head's own #2 (58 / 56 %). Together the two reach 74-78 %.
   - As deciders the auditors lift a chain to <= 15.6 kept (<= 483 tok/s at the bound). As tree siblings they lose to
     best-first at equal rows [K limit_auditor.out, limit_auditree.out].
   - A <= 60 MB/level head (60-120 M multiply-adds per token, 1-2 % of the 125B's) can hold today's head's knowledge
     (QAT 26-95 MB: kept x0.97-0.99), not the 125B's decision.
6. **So kept 24 for the price of a 32-row check is beyond (a)-(c) on this Mac this week.** The kept levers left are
   worth <= ~+5 % (48-row trees + close-call d1). 550 lossy then needs the cycle, not kept: tree-32 + close-call d1
   (~17.0 kept) reaches 550 at ~31 ms per cycle, i.e. a check at ~95 % of the C6 + int8 bound with a <= 3 ms draft.
   **Best idea (d): re-aim the week at the engine and the draft; kept is exhausted near 17-19 at this Mac's row
   floor.** The screen is section 5's table.

## 1. (a) The least an extra row can cost [K Leap check25_plan.md:95, PF-KBENCH TABLE_350.md:9359; FAR's / Fable C's parts]
| part of one exact extra row (R > 32) | work | at 100 % of peak | source |
|---|---|---|---|
| dense (GDN / attention projections, router, HC, shared, lm_head) | 3.73 G MAC | 0.27 ms bf16 (14.0 T MAC/s, measured matrix-unit peak) / ~0.15 ms int8 activations (est., a FAST lever) | Leap check25_plan.md:95, :181 |
| routed experts | 10 x 48 x 4.9 M MAC; byte-bound at tree sizes (1-3 rows per expert) | the union growth: +19 % at 64 rows on real trees = 0.06-0.10 ms per row | FAR (treek_union), NewIdeas |
| cores (attention / GDN state per row) | | ~0.04 ms | FAR |
| **total** | | **~0.25 (int8) - 0.37 (bf16) ms** | [P] |

- **Throughput at tree sizes is far below peak with today's kernels:** MLX's 4-bit dense GEMMs reach 27-29 TF only from
  ~1K rows; at 64 rows they read 2-13 TF (PF-KBENCH). So the real per-row cost of today's engine is higher than this
  floor.
- **What cannot help an EXACT row:**
  - Sibling rows share the prefix's K/V but not their projections or experts: different tokens give different residuals
    from layer 0.
  - A certified argmax over the children only would cut the lm_head share (0.64 of 3.73 G MAC, -17 %).
  - The ANE needs its own weight stream (+2.13 GB per pass).
  - The CPU's ~1.0-1.2 TFLOPS is 4 % of the GPU (MoonShots cpudraft.txt).
  - The one unverified hint of more compute is BaseRT's claim of 3.9x MLX's prefill on an M5 Pro via the neural
    accelerators (MoonShots sweep_r4c.md). Even 2x on dense leaves ~0.15-0.2 ms per row.
- **Lossy rows (union-constrained routing, int8 activations):** ~0.12-0.19 ms per row, still 2.5-4x the target, and
  their acceptances are not the 125B's own.

## 2. Rows -> kept -> tok/s: the own-information K-curve and its hindsight ceiling [K limit_rows.out]
Best-first pops of the real 127-node trees, exact walk.
- **hindsight-K:** the true path inside the 127 candidates, capped at K nodes (the best ANY K-row selection from the
  head's candidates can do).
- **E-sum:** 1 + the sum of path_p (the calibrated expectation).

| rows (K+1) | HE+ kept | hindsight-K | E-sum | tok/s at m = 0.30 / 0.20 / 0.10 / 0.05 ms | MBPP kept | hindsight |
|---|---|---|---|---|---|---|
| 32 | 16.26 | 18.48 | 15.85 | 566 / 566 / 566 / 566 | 9.03 | 11.21 |
| 48 | 18.86 | 21.39 | 18.50 | 554 / 581 / 611 / 628 | 9.73 | 11.28 |
| 64 | 20.77 | 22.45 | 20.08 | 529 / 576 / 633 / 665 | 10.19 | 11.28 |
| 96 | 22.77 | 23.35 | 22.12 | 461 / 530 / 623 / 683 | 10.88 | 11.28 |
| 128 | 23.70 | 23.70 | 23.25 | 400 / 477 / 591 / 671 | 11.28 | 11.28 |

- **Best-first is optimal for the head's own information.**
  - Accepted length = the number of true-path nodes in the tree. Under calibration, E[accepted] = sum of path_p over the
    tree, and path_p falls along every path.
  - So the top-K pops are the best ancestor-closed K-set. The realized kept tracks E-sum within ~0.4-0.7 (the head is
    slightly under-confident).
  - Consequence: no re-ordering, keying or widening rule beats this curve without NEW information (consistent with the
    kill list: keyed widening, span trees, wide-shallow).
- **Selection vs candidates.** At 32 rows, perfect selection would add +2.2 (18.48 vs 16.26). At 128 rows it adds
  nothing: there the candidates are the limit.
  - HE+ kept 24 is beyond the head's 127 candidates even in hindsight.
  - MBPP's true path leaves the head's candidates at ~11 deep (knowledge-limited, as the LAB 12.83 also says).

## 3. (b) Located decisions: the byte floor of a dependent decision [P on check25_bound_real.txt + IDEAS_1117_floor]
- **Why it needs a second stream.** The 125B's decision forms in its last ~4 layers (VERDICT-DEPTH). The row after a
  decision needs the decided token at layer 1. So a dependent decision cannot share the first decision's weight stream:
  it is a second pass over the tail.
- **Price of that pass.** The dense bytes are the same at any row count: 7.6 / 25.2 = 0.30 pass. The tail's expert
  union adds 0.27 (8 rows) - 0.44 (16 rows) pass. So the second decision costs >= 0.57-0.74 pass and two cost
  >= 1.6-1.7 passes, before overheads.
- That is just "two smaller checks", which the K-curve already prices (16-row blocks lose to 32: IDEAS_1116c).
- **The only one-pass form is speculation:** the alternative and its continuation ride as rows (section 2). Best-first
  already places them optimally for the head's information.
- **Perfect located decisions** (IDEAS_1117_floor: 2 decisions -> 23.65) need the 125B's information about WHERE and
  WHICH. The head has WHERE (margin AUC 0.954) but not WHICH (its #2 is y at 58 % of first misses).

## 4. (c) How much knowledge a head can hold, and what knowledge buys [K limit_auditor.out, limit_auditree.out, records]
- **Budget.** <= ~60 MB per level = ~120 M active parameters at 4-bit (60 M at 8-bit) per drafted token: 1-2 % of the
  125B's ~6.5 G multiply-adds per token (Leap: ~13 GFLOP per prefill token).
  - Resident capacity can be far larger (an MoE head keeps all experts in memory). Knowledge storage (~2 bits per
    parameter in published scaling) is not the constraint; per-token computation is.
- **The head's knowledge is already enough where it has the state.**
  - The same head fed the true state reads 97.8 % per guess (marginal, HE+) and fixes 68.5 % of the own chain's first
    misses.
  - The measured state-side levers, capacity included, add <= +0.8 pt (records, FAR's IDEAS_1116c).
  - QAT heads at 26-95 MB per level keep x0.97-0.99 of kept (MTP Head / NewIdeas).
- **Bigger, independent models at the rows that matter** (MoonShots' per-row picks along the true text = what a batched
  audit sees up to the first miss):

| at the chain's first misses | HE+ (337) | MBPP (2,144) |
|---|---|---|
| y = the head's own #2 | 58.2 % | 56.4 % |
| y = the 2B's top-1 | 52.2 % | 49.7 % |
| y = the 4B's top-1 | 58.2 % | 56.7 % |
| 4B right where the head's #2 is not | 19.6 % | 21.0 % |
| head #2 or 4B | 77.7 % | 77.4 % |
| the head with the TRUE state right | 68.5 % | 65.4 % |
| y in the head's top-8 | 92.6 % | 92.7 % |

- **As deciders (override when the margin >= tau; FIX-r, priced at the bound with c = 3.8 / 7.6 ms per pass):**
  - The best 2B arm is tau 2: phi 29.7 %, psi 0.41 %. It gives kept 14.76 / 15.25 (r1 / r2), 473-483 tok/s.
  - The best 4B arm: 15.11 / 15.59, 466-477 tok/s.
  - Plain chain-32 is 13.77 (464); tree-32 is 566.
- **As siblings (the auditor's pick hangs beside the spine with the head's continuation):** HE+ 2B 14.72-15.94 kept at
  40-72 rows (355-407 tok/s), 4B 14.86-16.47 at 43-74 rows. Best-first gives 18.86 at 48 rows and 20.77 at 64.
  Independent knowledge is mostly redundant with the head's own #2 and arrives too late (a full pass) to be cheap.
- **Reading.** Doubling the auditor (2B -> 4B) buys +6 pt at the first misses. The decisions there are the 125B's
  idiosyncratic choices (near-ties, style), and only the 125B makes them at a high rate. A <= 60 MB/level head cannot
  hold them. What such a head can still win is drift: the remaining 29 % of first misses that the true state would fix
  (route 1, closed by depth).

## 5. (d) Best idea: re-aim the week at the cycle, because kept is exhausted near 17-19 at this Mac's row floor [P]
tok/s = kept / (check + draft + overhead 1.5). Lossy kept with close-call d1 (EvalSuite night1117 PASS, +3-6 % kept per
check): tree-32 ~17.0, 48-row tree ~19.6 (at m 0.25 ms: +4 ms of rows).

| check (32 rows; % of the 25.2 ms C6 + int8 bound) | draft 3 ms | draft 5 ms | draft 7.4 ms (QAT head [P]) |
|---|---|---|---|
| 25.2 (100 %) | 17.0 kept -> 572 | 536 | 499 |
| 26.5 (95 %) | 548 | 515 | 480 |
| 28.0 (90 %) | 523 | 493 | 461 |
| 30.0 (84 %) | 493 | 466 | 437 |
| 56.4 (today, 45 %) | 279 | 270 | 260 |

- **Reading.** 550 lossy on HE+ at quality needs a check at ~95 % of the C6 + int8 bound and a ~3 ms draft. With a
  48-row tree (~19.6 kept, +4 ms of rows) the same 550 needs the 32-row part at ~27 ms (~93 %) plus a 3 ms draft. The kept levers inside this Mac's limits move the bar by ~5 %, the engine by x2.
- **The screen (what decides it this week, not a new run).** Leap's measured 32-row lossy check after Wednesday's
  modules, and the QAT head's measured ms per level.
  - **Line:** check <= 28 ms with draft <= 3-4 ms keeps 550 alive with tree-32 / 48 + close-call d1.
  - **Kill:** a check > 30 ms or a draft > 5 ms. Then no kept lever within these limits reaches 550 (the table: <= 515).
- **What would change this verdict (watch for it):**
  - A measured dense GEMM rate at 32-128 rows well above 14 T MAC/s (the neural-accelerator claim): it lowers the row
    floor towards 0.1 ms and the 64-row tree pays (633 at m 0.10).
  - A head whose candidates contain the true path deeper (hindsight-31 above 18.48), i.e. less drift (training).

## 6. Kills and limits (measured today)
- **Free rows (a):** an exact row is >= 0.25-0.37 ms on this Mac. 96 rows for <= 5 ms is 5-7x past the measured
  matrix-unit peak. KILL by arithmetic.
- **Two located decisions in <= 1.3 passes (b):** a dependent decision needs a second weight stream (>= 0.57 pass).
  KILL by arithmetic. The one-pass form is a tree, already optimal for the head's information.
- **Auditor deciders / siblings (c, 2B / 4B):** <= 483 tok/s at the bound as deciders, below best-first as siblings.
  KILL for 550.
- **Own-information kept 24 on HE+:** beyond the head's 127 candidates even with hindsight selection (23.70).

## 7. Reproducibility (~/FableGod/lunatic/, read-only; CPU; seconds each)
- limit_rows.py / .out: K-curve, hindsight-K, E-sum, tok/s by marginal row cost.
- limit_auditor.py / .out: 2B / 4B as FIX-r deciders on the chain (phi, psi, kept, tok/s).
- limit_auditree.py / .out: complementarity at the first misses; auditor siblings vs best-first at equal rows.
- Joins checked: MoonShots' tk0 == the cell's y on 99.42 % (HE+) / 100 % (MBPP) of joined cells (100 % joined).

## 8. TRIED lines (ready-made, for NewIdeas / MoonShots to record)
- ROW-FLOOR (Lunatic, Tue 17 Nov 08:4x): least cost of an exact extra check row on the M5 Pro = dense 3.73 G MAC at the
  measured 14.0 T MAC/s bf16 matrix peak (0.27 ms; ~0.15 int8 est.) + union growth 0.06-0.10 + cores 0.04 = ~0.25-0.37
  ms; 96 rows >= 24-36 ms vs the <= 5 ms asked | wall: fixed-cost (rows) | line: 45 | exact: n/a (pricing) | verdict:
  KILL ("nearly free rows" by arithmetic) | IDEAS_1117b_limits.md | [lunatic]
- KCURVE-HINDSIGHT: best-first own-information trees (real 127-node dumps), HE+ kept 16.26 / 18.86 / 20.77 / 22.77 /
  23.70 at 32 / 48 / 64 / 96 / 128 rows, hindsight (true path in the candidates) 18.48 / 21.39 / 22.45 / 23.35 /
  23.70; MBPP 9.03 -> 11.28, hindsight 11.21 -> 11.28; tok/s at the bound peaks at 48 rows (581 at m 0.2) | verdict:
  INFO (best-first is optimal under calibration; HE+ kept 24 is outside the head's 127 candidates) | limit_rows.out |
  [lunatic]
- AUDITOR-AT-MISSES: at the chain-32 first misses the 2B / 4B top-1 = y 52.2 / 58.2 % (HE+), 49.7 / 56.7 % (MBPP); head
  #2 58.2 / 56.4 %; head with the true state 68.5 / 65.4 %; as FIX-r deciders <= 15.6 kept / <= 483 tok/s at the
  bound; as siblings below best-first at equal rows | wall: drafter | verdict: KILL for 550 | limit_auditor.out,
  limit_auditree.out | [lunatic]
