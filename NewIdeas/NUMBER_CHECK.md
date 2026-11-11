# NUMBER CHECK ledger (NewIdeas; brief ~/Papers/BRIEF.md: every headline in every paper re-derived from raw rows by Tue 12:00)

Method: for each headline I open the raw rows (runs.json / requests.jsonl / decode.jsonl), recompute the number with a script
in scripts/numcheck/ (or a named earlier script), and mark it here: OK (same to the shown digits), DIFF (what the rows say), or
SOURCE? (no raw rows found). Marks follow the brief: [K] measured, [P] priced, [G] judgement. A paper's number is OK only if
its raw source is named and its mode label (exact to what / lossy gated on what / lossy ungated) matches the run's env.

## 0. Inputs the papers will quote (checked before the drafts exist)

### 0a. findings/2026-11-08_1630_strata-head-to-head.md (the brief's rule 4 cites it for the abstracts)
Ours (12 IFEval = chat, N45 keys, C 256, one pass per arm, quiet), from BigMoE's CHS reports:
| note says | source row | check |
|---|---|---|
| chat lossy certified 44.2 | CHS4 y0 43.8 + y0b 44.5 (L, QOM on code sets only = the CERT config on chat); mean 44.15 | OK as the mean of two passes; the paper must say "mean of 2 passes" |
| chat exact 38.3 | CHS4 yes1 38.3 (E + SB1; SB1 bit-exact: 676 / 676 logits hashes, CHS5) | OK; label "exact to NAX6-N AR (E + SB1)" |
| M1-B + COH 49.9 | CHS5 z0b 49.9 (second pass; z0 49.7) | OK; IFEval gate PASS both orders (RESULT_chat_m1b 70b5cf8), card gates pending |
| stack 56.0 | CHS5 zsgcr 56.0 (56.05 by my runs.json read) | OK; lossy UNGATED |
| 32 GB lossy 51.3 at 31.2 GB, exact 33.8 at 30.6 GB | BigMoE c32 runs.json: 51.33 / 33.81, footprint 31.22 / 30.62 GB (NewIdeas 16:3x) | OK |
| 32 GB first token ~2.9 s, whole request 22.5 / 19.8 | c32 runs.json: TTFT mean 3.09 (zs) / 2.93 (E); e2e 22.52 / 19.82 | OK (the note's "~2.9 s" is E's mean; zs is 3.09) |
| whole request chat 24.4 (M1-B + COH) | [[to run: CHS5 z0 / z0b runs.json e2e]] | pending |
| our chat first token median 2.92 s (2.34-3.81), "1.7 s + 22 ms per prompt token" | [[to run: CHS5 z0 runs.json pfs + P]] | pending |
| break-even ~530 answer tokens | derived; [[to recompute from the two lines]] | pending |
| Strata side (all arms, both caches, all sets) | scripts/numcheck/strata_h2h.py (written 16:5x; runs after the orchestrator's quiet window) | pending |

Already re-derived today by NewIdeas (Strata, IFEval pooled decode): A1 28.0, A2 25.6, B1 37.4, B2 37.1, C1 25.5, C2 25.3,
D1 36.1, D2 35.6 (default cache); B 41.8, D 40.6 (full cache, peaks 40.76 / 41.15 GB). Matches REPORT.md's pooled column.

### 0a-2. RESULT (scripts/numcheck/ours_runs.py + strata_h2h.py, run after 17:13; outputs transcribed below)
FINDING 1 (affects every paper): the older c45-runner reports (S5', E5, N45: data/s5p/s5p_report.txt, data/e5/e5_report.txt,
data/n45/n45_report.txt) print WHOLE DECODE as the GEOMETRIC MEAN over prompts per pass (reproduced exactly: spb1 168.6,
spb2 179.6, spx1 181.7, spx2 180.9, spmb1 118.3, spmb2 121.9, nie1 39.9, nie2 39.8, nil1 45.2, nil2 44.8), not pooled.
STYLE.md section 3 requires pooled headlines ("geo" only when labelled). Their END TO END column is a third definition
(below both the pooled whole request and the mean of per-prompt rates). The CHS / C32 reports print pooled first: those
match. Pooled triples, passes pooled (whole decode / whole request / first token median, lifetime peak):
| row | report headline (geo) | POOLED [K] | == AR |
|---|---|---|---|
| S5' B HumanEval (cert config, lossy) | 174.4 | 171.0 / 36.0 / 4.56 s, 42.91 GB | - |
| S5' X HumanEval (lossy FAST) | 181.4 | 181.3 / 36.6 / 4.51 s, 42.91 GB | - |
| S5' B MBPP12 (cert, lossy) | 120.2 | 114.5 / 25.8 / 3.79 s, 42.55 GB | - |
| S5' X MBPP12 (lossy FAST) | 123.6 | 119.8 / 25.3 / 3.76 s, 42.57 GB | - |
| E5 E HumanEval (exact) | 99.4 | 97.8 / 31.7 / 4.59 s, 42.23 GB | 24 / 24 |
| E5 EB HumanEval (exact + bundle) | 102.7 | 101.4 / 32.6 / 4.41 s, 42.24 GB | 24 / 24 |
| E5 E MBPP12 (exact) | 71.3 | 68.4 / 21.8 / 3.73 s, 41.86 GB | 24 / 24 |
| E5 EB MBPP12 (exact + bundle) | 73.3 | 70.4 / 21.7 / 3.80 s, 41.92 GB | 24 / 24 |
| N45 E(B) IFEval (exact) | 39.9 | 37.1 / 20.8 / 2.89 s, 41.70 GB | 24 / 24 |
| N45 L IFEval (lossy cert) | 45.0 | 43.4 / 21.1 / 3.00 s, 42.37 GB | 6 / 24 |
| N45 E(B) math (exact) | 78.8 | 77.9 / 38.9 / 2.73 s, 42.50 GB | 24 / 24 |
| N45 L math (lossy cert) | 87.9 | 86.9 / 39.6 / 2.96 s, 43.10 GB | 7 / 24 |
| CHS4 y0 + y0b IFEval (L, cert on chat) | 43.8 / 44.5 | 44.2 / 20.8 / 3.05 s, 42.39 GB | - |
| CHS4 yes1 IFEval (E + SB1, exact) | 38.3 | 38.3 / 21.3 / 2.82 s, 41.69 GB | (E: 12 / 12 by CHS) |
| CHS5 z0 + z0b IFEval (M1-B + COH) | 49.7 / 49.9 | 49.8 / 24.4 / 2.92 s, 42.38 GB | - |
| CHS5 zs IFEval (M1-B + COH + SB1) | 51.9 | 51.9 / 24.8 / 2.98 s, 42.35 GB | - |
| CHS5 zsgcr IFEval (best stack, ungated) | 56.0 | 56.0 / 24.6 / 2.94 s, 42.37 GB | - |
| C32 e32 IFEval, C 160 (exact) | 33.8 | 33.8 / 19.8 / 2.83 s, 30.62 GB | 12 / 12 (BigMoE) |
| C32 zs32 IFEval, C 160 (lossy, ungated) | 51.3 | 51.3 / 22.5 / 2.99 s, 31.22 GB | - |
Note: S5' B's two HumanEval passes differ by 9 % pooled (163.9 vs 178.8); cause not checked.

FINDING 2, the Strata note (0a) against the rows:
- Strata side: OK to the digit (every arm, both caches, every set; decode, whole request, first token median), e.g. full
  cache B 47.0 / 45.6 / 41.8 / 47.1 decode and 40.9 / 36.9 / 33.7 / 40.5 whole request (HE / MBPP / chat / math); default
  B chat 37.3, D chat 35.8, first token 0.98 s, whole request 29.5 / 28.2; MTP +53 % (25.3 -> 38.7, all sets); cache
  25-28 -> 33 GB +12-21 % (B); B1 vs B2 38.4 / 38.2.
- DIFF: "the Q8_0 lossy read adds only 3-4 % over exact (38.7 vs 37.7)": those numbers are +2.7 %; per set +1.3-4.9 %.
- DIFF: our code / math decode numbers in the note (HE 181.4 / 174.4 / 102.7, MBPP 123.6 / 120.2 / 73.3, math 87.9 /
  78.8) are the geo headlines beside Strata's pooled ones. Pooled: HE 181.3 / 171.0 / 101.4, MBPP 119.8 / 114.5 / 70.4,
  math 86.9 / 77.9. Ratios vs Strata full cache move a little (HE x3.9 -> x3.86 FAST; MBPP x2.7 -> x2.63).
- DIFF: our code / math whole request in the note (HE 34.8-35.4, MBPP 23.4-23.6, math 40.3) are the old END TO END
  column. Pooled whole request: HE 36.0-36.6, MBPP 25.3-25.8, math 39.6 (L) / 38.9 (E). Strata still wins HE (40.9) and
  MBPP (36.9); math is level (40.5).
- OK: chat 44.2 (cert, mean of y0 / y0b), exact 38.3 (yes1), 49.9 (z0b; pooled z0 + z0b 49.8), 56.0, whole request 24.4,
  first token median 2.92 s; 32 GB 51.3 / 33.8 at 31.2 / 30.6 GB, whole request 22.5 / 19.8. 32 GB first token: the note's
  "~2.9 s" is E's mean (2.93); medians E 2.83 / zs 2.99 s.
- CORRECTION OF MY OWN 16:39 MESSAGE: I gave "exact, ours 39.9 vs Strata 40.6 = 0.98x" at 45 GB. 39.9 is N45's geo. Pooled:
  38.3 (CHS4 yes1) or 37.1 (N45, 2 passes) -> 0.91-0.94x. The note's "exact chat decode is behind" stands.
- Pending: "1.7 s + 22 ms per prompt token" and the ~530-token break-even (need the per-prompt fit; CPU, after 18:16).

### 0b. CT1 (chat prefill bytes), judged 16:40 with scripts/chat/ct1_judge.py: OK vs BigMoE (acf8ac3)
45 GB chat prefill 244.94 GB / 12 prompts = 100.9 % of 12 GB's 242.65 GB; mean base residency at start 45.7 %; routed-resident
65.5 % base. PF-D's TTFT 3.1 -> ~1.6 s is [P] (BigMoE's fit 0.32 s + GB / 7.33).

## 1. MAIN (v3-draft, file of Mon 00:29), first pass Mon 07:22-07:31 (`date`), before the 12:00 drafts
Scripts: scripts/numcheck/main_check.py (-> data/numcheck_main_0725.txt), main_check2.py (-> data/numcheck_main2_0728.txt),
plus the cited reports (CERT_VERDICT.txt, chs4 / chs5 / pfd_report_0009 / pfx reports, RESULT_chat_m1b) and two inline checks
(Strata's file size; kept per check). Exact McNemar p recomputed from the discordant counts.

### DIFF (sent to the editor)
1. Abstract l67 "728 pairs pooled: -0.96 points, p 0.27": CERT_VERDICT pools 3 sets x 2 orders (HumanEval+, math, MBPP =
   364 x 2); IFEval is not in the pool. p = 0.2649 (11 vs 18 discordant; recomputed) -> "p 0.26". Same label fix for Table
   5.2's "pooled, 728 pairs" row and 6.1b's first note.
2. Abstract l68 "111-115 on MBPP": S5 111.13, S5' 114.45 -> "111-114".
3. Section 7 l524 "its Q8_0 read adds 3-4 %": B vs D pooled +2.7 % default cache (38.73 / 37.71), +2.3 % full cache (45.75 /
   44.71); per set +1.3 to +4.7 % -> "2-3 % pooled (1-5 % per set)".
4. Section 7 l519-521 break-even: "past 2,000 on chat with the certified config": N45 L 2,473, CHS4 y0 + y0b 1,758 ->
   "past about 1,750". "about 530 with M1-B": the paragraph's own method (medians, pooled decode) gives 548 -> "about 550".
   (OK: 252 HumanEval, 246 MBPP, 201 math, 343 chat as shipped.)
5. Section 7 l525 "a larger cache adds 12-21 %": B +12.3 to +21.4 %, D +13.4 to +22.7 % -> "12-23 %" or name arm B.
6. Section 7 l534 "default-cache passes repeat within 0.5-5.5 %": pass 1 vs 2 decode per set: B 0.1-1.4 %, D 1.2-5.0 %
   (C 0.7-4.0 %, A 3.2-9.4 %) -> "within 5 % (B and D)".
7. Section 7 l533 "at most 4 of 24 rows are token-identical to our exact reference": full sequences 2 of 24 on HumanEval and
   IFEval (both default passes), 0 on MBPP and math; 4 of 24 HumanEval rows agree up to the shorter answer.
8. Table 6.1c l441 (M1-B + COH): peak "41.7-42.4 (CHS4 arms)": its own records give 42.36 / 42.38 -> "42.4". Status "a known
   multilingual benchmark decides" and 5.3 l371-373: stale (Kamen 11-08: multilingual is information only; the gate is card
   arm m1b, EV:PREREG_card_gates).
9. 6.1b l432 (and l946 / l948 per EvalSuite): "gate open until 15:00" -> 16:00 (MONDAY.md l10 / l15).
10. 5.3 l368 "+13 % over the certified config in the same run": CHS5 has no certified arm; 49.80 (CHS5 z0 + z0b) / 44.18
    (CHS4 y0 + y0b) = +12.7 % across two runs -> drop "in the same run". Misses 92 -> 25 OK (93.6 / 90.4 -> 24.6).
11. 6.3 l475-477 "44-79 %": over Tables 6.1a-c the first-token share runs 43 % (pdz4a) to 80 % (X HumanEval); 44-79 % holds
    for 6.1a-b. "6.03 s per request" -> 6.04 (217.1 / 35.96); "exact chat 3.80 of 6.78 s" -> 3.79 of 6.77 (140.6 / 37.14, / 20.78).
12. 4.3 l280 kept per check "5.9 on MBPP, about 6.6 on math": pooled tokens / checks: MBPP 5.96 (EB) / 6.13 (L), math
    6.48 / 6.49 -> "6.0" and "6.5" (HumanEval 10.68, chat 2.45-2.53: OK). If the source is a mean of per-prompt values, say so.

### OK (re-derived, same to the shown digits)
- Abstract: 96 / 96 code, 24 / 24 chat, 24 / 24 math (N45, prefix to n); 36 / 36 with yes1 (12 / 12); 101.4 / 70.4 / 77.9 /
  37.1 at 41.70-42.50 GB; lossy 171-175 HumanEval (170.96 / 175.35), 86.9 math, 43.4 chat at 42.38-43.10 GB; first token
  2.73-4.59 s; whole request 20.78-39.56; PF-D 2.908 -> 2.263 s; Strata 40.63-47.11 tok/s, 0.540-1.013 s; ratios x2.4-3.7,
  x1.8, x1.6-2.2, x1.7; 43.4 vs 41.8; 37.1-38.3 vs 40.6; Strata wins or ties every whole request (math 39.56 vs 40.55,
  38.86 vs 39.87); break-even 200-250 code and math, ~340 chat as shipped.
- Table 6.1a every cell (decode, whole request, first-token median and range, peak, == AR); 37.8 / 37.9 (ycoh / ycoh2,
  12 / 12) and 3 / 12 without coherent reads (ycoc / ycoc2 in chs4_report).
- Table 6.1b every cell incl. S5 175.35 / 111.13 (no pfs in S5 records, as stated); 6 / 24 and 7 / 24; S5' passes 8 / 12.
- Table 6.1c every cell except item 8: X rows; z0 + z0b 49.80 / 24.38 / 2.92 (2.34-3.81); zs 51.89 / 24.84 / 2.98
  (2.51-3.65), == z0 12 / 12; pdz4a 50.67 / 28.84 / 2.26 (1.55-2.59) / 43.69; pdz0 50.29 / 24.59 / 2.91 (2.34-3.74) / 42.40;
  12 / 12 tokens and 721 / 721 logits hashes; zsgcr 56.04 / 24.62 / 2.94 (2.55-3.74). PF-D v3 medians 2.25-2.41 (pdz1p,
  pdz1, pdz1q); 9.63 vs 7.50 GB; unused B 19.3 % / C 31.4 % (report totals); pdzfa 1.83 s, 30.45, 2 / 12 equal.
- Memory exceptions: npl 45.11 GB, npe 44.46 GB, plm1 45.10 GB; every Table 6.1 row <= 43.69 (others <= 43.10).
- 6.2 every cell (npe means 15.9 / 40.1 / 64.1; pe1024 15.2 / 37.3 / 61.9; plm1 8.0 / 12.9 / 18.0; SSD 109.3 -> 63.9,
  264.5 -> 67.2, 430.1 -> 68.3 GB; hashes 8 / 8, 28 / 28, 22 / 22); lossy 16.3 / 41.2 / 66.6 (npl means).
- 6.3: CT1 re-fit 0.318 s + GB / 7.33 GB/s, 20.41 GB per prompt; 1.63 s at 9.63 GB; 24.6 -> 28.8; 50.3 / 50.7.
- 7: file 75.84 GB; Table 7.1's 24 Strata cells; x1.21 / x0.86 / 2.71x; first token 2.70-7.03x; MTP +53.0 % (25.31 ->
  38.73), 86.8 % accepted; 32 GB rows (33.8 / 19.8 / 2.83 at 30.62; 51.3 / 22.5 / 2.99 at 31.22; Strata 35.83 / 37.26,
  28.20 / 29.54, 1.06 / 0.98 s).
- 4.1 arithmetic (4.9152 M per expert, 120.80 B, 2.36 B active, 6.0 / 6.65 B, 125.7 B); 67.9 GB and 1.33 GB per token
  agree (67.9 / 24,576 x 480); 45 GB = 41.91 GiB. 4.4: EB +3.7 / +2.9 %; 1-row check +1 % (38.28 vs 37.80 / 37.93) and
  +4.2 % lossy (51.89 / 49.80); NAX6-N 44.74 / 42.12 = +6.2 % (arithmetic only). 4.5: 1.9-3.4x vs pe1024; pde4k 87,912
  rows, 12 / 12. 5.2: every cell = CERT_VERDICT.txt. 3.4: 2 vs 7 -> p 0.1797; -8.33 with p 0.0625 consistent (0 vs 5
  of 60). 5.3: 334 per 10K (235 / 7,033); 86 vs 87, 83 vs 87.

### Not yet re-derived (cited sources, next pass)
3.2 audit counts (16,305; 15,690), 3.3 (11.65 vs 15.50), 3.4 table cells, 4.2 (-17.2 %, +8.3 %, 31.8 GB, ~250 ms, 142.2 ->
181.2), 4.4's T-numbered levers, 5.1 (2,888 copied tokens), 5.2 "1 of 12 IFEval rows (CERT)", the chs3 bisect counts, 2's
67.8 % HumanEval copying, 4.3's chat check cost at 1-2 rows, sections 8-12 and the appendices.

### MAIN DIFF 13 (found while checking RELEASE)
13. 4.1 l251 "the store is 72.1 GB (BM:DESIGN_3bit.md)": records3 on disk is 48 files, 68.451 GB (1,426,063,360 B per
    layer = 2.785 MB per expert); 72.1 GB was the design's RECSZ 2,932,736 B. RELEASE's 68.5 GB is right.

## 2. RELEASE (draft v1, file of Mon 07:22), first pass Mon 07:31-07:33 (`date`)
Same scripts and records as MAIN; plus the pack's safetensors headers (bytes by tensor name), du of records3 and of the drafter
file, pfd.log Part B, and pdz4a's pfd counters.
### DIFF
1. Section 2, row "45 GB E (+ bundle)": the IFEval cell is yes1 = E + coherent reads + the 1-row check, one pass (not EB, two
   passes like the other cells). Name it in the cell.
2. Row L, IFEval: 44.2 / 20.8 / 3.05, 42.39 is CHS4 y0 + y0b (OK as numbers). MAIN quotes N45 L (43.4 / 21.1 / 3.00, 42.37) for
   the same config in its abstract and Table 6.1b. One choice across the set, or "43.4-44.2".
3. Peaks: chat as shipped "43.70" -> 43.69 (pdz4a record); PF-A / PF-B "45.10 / 46.20" -> 45.09 / 46.23 (pdpa1 / pdb38s). The
   run reports round peaks to 0.1 GB and print two decimals; two-decimal peaks should come from the records.
4. Section 6 l139 "supplied 72-75 % of the rows read from the SSD" -> 71-75 % (B 74.97 %, C 71.48 %, pooled 73.7 %).
5. Not a number: l51 cites the Sun 18:2x 49.6 -> 40.0 drop "with PF-D"; the editor's note 10 says never cite it as a PF-D cost
   (BigMoE's WED_TESTS l1204: not reproduced, cause unknown). Editor's call.
### OK
- Section 2: every filled cell re-derived (E rows 101.36 / 32.55 / 4.41 / 42.239, 70.38 / 21.74 / 3.80 / 41.917, 38.28 / 21.27
  / 2.82 / 41.687, 77.89 / 38.86 / 2.73 / 42.499; L rows 170.96 / 35.96 / 4.56 / 42.907, 114.45 / 25.77 / 3.79 / 42.555, 44.18 /
  20.76 / 3.05 / 42.386, 86.91 / 39.56 / 2.96 / 43.099; M1-B + COH 49.80 / 24.38 / 2.92 / 42.377; chat as shipped 50.67 /
  28.84 / 2.26, tokens == pdz0 == zs == z0 on 12 / 12, hashes 721 / 721; e32 33.81 / 19.82 / 2.83 / 30.618; C30 HumanEval
  150.72 / 36.03 / 4.43 / 31.779, MBPP 111.03 / 26.78 / 3.69 / 31.436; zs32 51.33 / 22.52 / 2.99 / 31.223).
- "Where we lose" (0.5-1.0 s, 2.7-4.6 s, 2.26 s, 28.8, 0.83 s, 33.0-33.7, peaks 40.8-41.2, 2.7x, 50.7 vs 40.6-41.8); long
  prompts 433 / 500 tok/s (pdpa1 18.34 s, pdb38s 15.87 s at 7,932 tokens); PF-D 87,912 rows, 2.91 -> 2.26, 24.6 -> 28.8, 50.3
  / 50.7, E 2.89 -> 2.26 (pde0 -> pde1, CAP 64), extra peak 1.0-1.3 GB (E: 0.83), 1.83 s / 30.5 / 10 of 12.
- Section 3: pack 105.42 GB = routed 67.948 + n-gram 32.000 + dense 3.795 + MTP 1.673 (safetensors headers and file sizes);
  store 68.451 GB; ~174 GB; drafter file 1.141 GB. Section 5: 45.1-46.4 GB on 7.9K prompts (npl 45.11, pde4k 46.39); 19-31 %
  unused (B 19.3, C 31.4); 7.5 -> 9.6 GB; ~7.3 GB/s (CT1 fit 7.33).
- Open: TBD-16 (L + COH rows: rel1 at the lcoh -> m1b seam today), TBD-11 (32 GB exact code / math rows).

### MAIN DIFF 14 (found while checking DRAFTER)
14. 4.4 table, "drafter by task: chat +24.5 % on 23 fresh prompts": the +24.5 % is on the 19 chat prompts of D-23d's 23 (the
    other 4 are agent prompts, which stay on the block drafter) -> "on 19 fresh chat prompts".

## 3. DRAFTER (v1.1 full draft, file of Sun 23:51), first pass Mon 07:33-07:38 (`date`)
### Re-derived from raw data: OK
- Abstract / Appendix C row 1-3: tokens per 16-row pass, my own walk over TL:data/d22r_b3 / d22r_b4 against TL:data/d17
  (stock), same definition as d22_real.py (draft k at position p predicts out[p + k], answers cut at <|im_end|> /
  <|endoftext|>): code 7.78 -> 10.98 = x1.411; gsm 6.07 -> 6.86 = x1.131; IFEval 2.43 -> 2.49 = x1.024; multi 2.92 -> 3.00 =
  x1.028; b4: code 10.65 (x1.368), IFEval 2.51 (x1.031), multi 3.03 (x1.038).
- Abstract / row 10 (D-23d), from TL:logs/d23d.out per-prompt lines: chat (19 prompts) whole decode 23.12 -> 28.79 (+24.5 %),
  whole request 18.56 -> 21.95 (+18.3 %), prefill median 1.90 s both arms (1-decimal log values), lifetime peak 37.9 GB.
- Row 22, from BM records: x2e 36.61 -> x2ep4 36.94 (exact neutral); y0 43.80 -> y0p4 40.22 (-8.2 %).
- Row 27 / abstract: F2a held-out x1.014 (2.301 -> 2.334, 2.507 -> 2.540), on-policy x1.002 (2.444 -> 2.450) = f2a_eval.txt.
- Row 28: Strata +53.0 % (25.31 -> 38.73), 86.8 % accepted (MAIN check).
- Abstract "83 items, 41 killed, 27 between / parked / information": Appendix A1-A6 has 83 rows; 42 verdicts contain KILL,
  41 without D-12 ("KILL at the desk, overturned by D-17"); 11 BETWEEN + 19 other (27 without DROPPED, OPEN, withdrawn).
  Wording: several kills are desk / paper kills, not "by their own pre-registered lines".
### DIFF / notes (sent to MTP Head)
1. Row 23 / abstract "~114 tok/s" from "47.0 ms + 3.25 per row" (+ draft 0.43 + 2.59 per row): a plain least-squares fit of
   ck on rows over the same 1,280 checks (nie1-2, first 3 checks of each prompt skipped; median wall by rows matches
   chat150_read within 0.1-1 ms) gives ck = 44.9 + 4.00 x rows, and with it the 16-row ceiling is ~106 tok/s. The fixed share
   (74-85 % at 2-4 rows) holds. Name the estimator, or write "~105-115 tok/s".
2. Row 22: peaks "42.30 vs 42.40 GB" -> records 42.28 vs 42.37. Draft "9.1 -> 16.2" / "8.7 -> 14.9 ms" are the reports'
   per-prompt means; pooled per check 8.6 -> 14.4 / 7.9 -> 13.4 (the same +68-70 %).
### Not re-derived (result files cited): rows 4-9, 11-21, 24-26 (the walk data exist; next pass if time allows).

## 4. SMALL_RAM (file of Sun 22:31), first pass Mon 07:38-07:41 (`date`)
Scripts: scripts/numcheck/f12rows.py (Fit12's per-prompt lines + phase logs, my own
parser: whole decode = sum tok / sum decode s, whole request = sum tok / sum (prefill + decode), first token = prefill + end
of check 0's verify phase - window start), strata_low.py (Strata's request rows), E24 / E24m per-prompt log lines.
### DIFF (sent to Fit12)
1. 24 GB rows (abstract l42-43, Table 6.1, 6.5): "41.3 (per pass 41.8, 40.8)" and "38.4 (38.2, 38.6)" are geometric means
   over prompts (the runner's aggregate; my geo 41.28 / 41.79 / 40.77 and 38.36 / 38.19 / 38.53). Pooled (STYLE 3): HumanEval
   41.07 (passes 41.65 / 40.50), MBPP 37.43 (37.37 / 37.49) -> "41.1" and "37.4". The "1.4-1.6x" over Strata still holds
   (37.43 / 26.58 = 1.41; 41.07 / 26.42 = 1.55).
2. 6.4 l599 / abstract "Strata's first token 1.4-1.7 s at 24 GB and 2.0-2.8 s at its smallest settings": true on code and
   chat; math is 2.13-2.31 s at 24 GB and 3.23 s at the 4 GB arena -> add "on code and chat".
3. TBD-33 resolved: Strata's report "END TO END" = geometric mean over requests of generated / request wall time (StrataEval
   h2h/report.py l8): smallest setting chat 9.70 vs the pooled 11.22. The paper's whole-request numbers are pooled (OK).
### OK
- Table 6.1, all nine 12 GB rows to the digit: tab12e 18.34 (18.38) / 13.12 / 4.29 (3.47-5.85) / 11.140; chat12 12.76 (12.88)
  / 10.09 / 2.92 (2.33-3.71) / 10.962; tab12f 27.90 (27.58) / 17.60 / 4.15 (3.29-5.71) / 11.280; c1u16 44.34 (46.33) / 24.92
  / 4.30 (3.61-5.18) / 11.245; ex16 18.47 (18.42) / 13.90 / 4.51 (4.21-5.93) / 11.144; chw13a 12.56 / 9.32 / 2.90 / 10.962;
  chw14c2 13.97 / 10.07 / 3.04 / 11.012; chw14ha 18.69 / 14.11 / 4.48 / 11.145; chw14h2 18.47 / 13.98 / 4.48 / 11.159.
- 32 GB rows (e32, zs32); Table 6.5's 13 Strata cells and peaks 23.75 / 24.14 / 14.82 / 14.19 (summary.json peak_footprint_b).
- 6.2: tokens per check, pooled: code 8.86-10.09, chat 2.34-2.35; FAST x1.52 (27.90 / 18.34). 6.3: peaks 10.96-11.28;
  +0.38-0.70 GB over 10.583; Strata 14.19 - 3.37 = 10.82 GB. 6.4: 21,986 MB in 2.906 s, 21.3 tok/s (62 timed tokens, as
  the paper states). 6.5 bullets: +13 % (14.42 / 12.76), +11 % (11.22 / 10.09), 2.78 vs 2.92 s; 12.89 vs 13.12, 1.99 vs
  4.29 s, 18.34 vs 15.47.
### Not yet checked: sections 3.0, 4 (decision map) and 5 (BYTES10, incl. the abstract's 81.7 vs 210.2 MB [P]), 6.2's MB per
kept token (491-564 / 436-441), section 7 (verify-call time), section 11.

## 5. KERNELS (v1.0 full draft, file of Mon 00:18), first pass Mon 07:41-07:44 (`date`)
### OK (raw outputs re-read or re-derived)
- R2 (D:inv1_screen.txt): 37 records per forward both ways; logits rows equal 16 / 16; rows with a differing record 0 / 16.
- R5 (D:kern23.txt): dense part per 16-row check 56.8 -> 27.4 ms, floor 11.9.
- R23 (D:natfri/r57 runs.json, my pooling over the 13 prompts kept in all four arms): 3c-N 42.12, NAX6-N 44.74 = x1.0622;
  peaks 38.10 / 38.11.
- R24 (D:k90_arm012.txt "QUIET" section, bw_probe_k90.txt): R 295-305 GB/s at S 16-32 (287 at S 8 per TABLE_350); NAX6-N
  218-219 GB/s = 74 % at S 16, 70-72 % at S 32; NAX6 55-59 %; 4 GB probe 285-286 GB/s.
- R20 (pfd.log Part B; pdpa1 / pdb38s records): 215 / 432 / 500 vs 189 / 358 / 433 tok/s, x1.14-1.21; abstract's 46.2 GB.
- R28 (D:dnks_bench.txt): -7.74 ms per 2-row check; engine -4.19 ms (mean of 55.17 / 55.30 -> 51.05).
- x0.75 (11.65 / 15.50 = 0.752); 220 tok/s (12 / 54 ms); dispatch 2-6 ms (1,000-4,000 x 1.6 us); ten MLX findings (5.1-5.10).
### DIFF (sent to Leap)
1. Abstract l57 "ours (about 3-4.6 s)" -> "2.7-4.6 s" (MAIN 6.1 medians 2.73-4.59 s; 2.26 s on chat as shipped).
2. R20 / 4.6 l231 peaks "46.20 vs 45.10 GB" -> records 46.23 / 45.09 (the run reports print 0.1-GB-rounded peaks).
3. 6.4 DN-KS: the 7.74 ms is over 5 narrow kinds incl. shared down (16.7 -> 9.1 us x 48 = 0.36 ms); the text lists four,
   whose savings sum to 7.38 ms. Add shared down.
4. 6.4 table, attention query projection "243-247 GB/s": the row's own 17.7 MB / 74-75 us = 236-239 GB/s (DN-KS bench: 74.9 us,
   236 GB/s). k / v "18 GB/s": 0.8 MB / 48-49 us = 16-17. Check R27's raw.
5. 6.2 "At S 16 ... 287-305 GB/s ... 70-74 %": 287-305 and 70-74 % span S 8-32; at S 16 it is 295-296 GB/s and 74 %.
### Not yet re-derived: R1, R3-R4, R6-R19, R21-R22, R25-R31's raw files; section 5's MLX findings (reproductions).

## 6. EVALUATION (file of Mon 06:45), first pass Mon 07:44-07:45 (`date`)
### OK
- CERT row and section 5.2 of MAIN: every delta and p recomputed from the discordant counts in results/cert/CERT_*.txt (exact
  two-sided McNemar): 0 vs 4 -> 0.1250; 2 vs 2 -> 1.0; 1 vs 3 -> 0.6250; 1 vs 4 -> 0.3750; 5 vs 3 -> 0.7266; 3 vs 4 -> 1.0;
  4 vs 6 -> 0.7539; pooled 11 vs 18 -> 0.2649 (-7 / 728 = -0.96); accuracies match the counts (148 / 164 = 90.2 %).
- M1-B + COH chat gate: 4 vs 5 -> p 1.0 (-1.00); 4 vs 8 -> p 0.388 (-4.00).
- Tables 1a and 2: every p is the exact McNemar p of the smallest discordant split that gives the stated delta at the stated
  n (e.g. 0 vs 6 -> 0.031 for ship8; 2 vs 5 -> 0.4531 for STACK MBPP; 0 vs 7 -> 0.0156 for theta 0.8; 6 vs 3 -> 0.508 for the
  IFEval re-test); consistent. Abstract "8 points on 50 items" = REF2 math +8.0 (n 50).
### DIFF (sent to EvalSuite)
1. Abstract l18-19 "The 45 GB release configuration passed all 10 pre-registered lines" and Table 2's "CERT, the 45 GB release
   FAST configuration": the certificate covers L (the certified config); the release ships L + COH, which it does not cover
   (MAIN 6.1b, RELEASE 2; its own gate is card arm lcoh). The 10 lines are 8 set-order lines + pooled + the long-code line
   (the abstract names 9).
2. "Pooled over 728 pairs": name the 3 pooled sets (HumanEval+, math, MBPP; IFEval is not in the pool).
### Not re-derived: Table 1b's audit counts (EV:RESULT_r56* / nax6 files), the card-gate reference counts (381 / 381; rows at
the cap 10 / 54 / 88), section 4.6's power estimate (5-15 % [P]).

### SMALL_RAM second pass, Mon 07:45-07:47 (`date`), on the file Fit12 updated after the first pass
- OK, re-derived: 3.1 settings (ex16 log: C_m 47 / C_l 32 (9 %), pools 6.03 GB, MLX active 8.96 GB, drafter 0.928 GB, 10.583
  GB after load); 3.4 (e0 36.43 -> e32 33.81 = -7 %, zs 51.89 -> zs32 51.33 = -1 %; misses per check pooled from checks.json
  89.8 -> 168.4 and 24.6 -> 42.7); 5.4 table (x2c0 44.80 / 97.8 / 41.8, x2c0b 43.71 / 91.1 / 40.9, x2qm1 45.26 / 42.8 / 22.8,
  x2qm2 49.05 / 25.8 / 15.5, x2qm2s 52.01 / 28.2 / 15.1, x2e 36.61 / 89.8 / 40.0; +10.7 %, -63 %, -73 %); the gated config
  (z0 49.67, z0b 49.94, 24.39, first token median 2.9245 (2.40-3.72), 36.4 of 71.4 s); 5.5 and 6.2 SSD MB per kept token
  from the phase logs' byte counters (chat12 556, chw13a 564, chw14c2 491; code 436-441); replay outputs (24 GB EXACT 210.2
  vs QOM0.5+next 81.7 = -61 %; 45 GB EXACT 38.1; 5.5x and 16x); section 7's table is internally consistent (nl - CPU
  column; 81.4 ms per GB = 12.3 GB/s).
- DIFF (minor): 5.4's peak column and the gated config's "42.40 GB" are the report's 0.1-GB-rounded values; records:
  x2c0 42.37, x2c0b 42.41, x2qm1 42.38, x2qm2 42.38, x2qm2s 42.33, x2e 41.70, z0 42.36.
- Not re-derived: section 4's map outputs (DM:out/gate.txt, BD:data/m1/m1_quality.out), section 7's fit (F12P raw), 11.

### MAIN DIFF 15-16 (second pass on section 4, Mon 07:47-07:48 `date`)
15. 4.2 l267 "+27.4 % whole decode on the lossy config (142.2 -> 181.2 tok/s, 12 HumanEval)": geometric means over prompts
    (F:2026-11-07_0945 says "geo mean"). Pooled (s6b1-2 vs s6w1-2): 144.05 -> 180.13 = +25.1 %.
16. 4.4 table, cache keep "x1.10 on the lossy config": the geo of the screen's steady rate (tpsw; 149.1 -> 164.2, passes
    148.1 / 150.0 -> 161.6 / 166.5 reproduced), which STYLE 3 calls information only. Whole decode pooled (scb1-2 vs sck1-2):
    127.25 -> 134.92 = x1.06.
    General note: the older levers in 4.4 (T489, T309, T14, M-17, T782) carry their own runs' metrics (often geo or steady);
    label them "as measured then (metric)" or re-derive them pooled.

### 0a pending items, closed Mon 07:48 (`date`)
- "1.7 s + 22 ms per prompt token" (the Strata note): least squares on z0 + z0b's 24 rows gives 1.76 s + 22.1 ms x P (P 37-101);
  z0 alone 1.79 + 21.4; zs 1.93 + 19.1. The note's "1.7 s" rounds 1.76 down: "1.8 s". Information; no paper uses the fit.
- "~530 answer tokens" (49.9 config vs Strata's full-cache Q8_0 arm): by first-token medians and pooled decode it is 540
  (z0b, 49.94) to 548 (z0 + z0b, 49.80): "about 550" (MAIN DIFF 4).
### DRAFTER row 6 (Mon 07:48): the MTP K15 side re-derives from TL:data/d17 (code 7.78, gsm 6.07, IFEval 2.43, multi 2.92);
the block drafter's side (10.44 / 6.60 / 2.24 / 2.22 / agent 3.28) is from d17_route.txt (its per-position arrays are not in
d17/); the ratios 0.746 / 0.919 / 1.084 / 1.315 / 1.295 follow from those sums. OK.

### MAIN Appendix E and F (Mon 07:49 `date`)
- Appendix E (main_table.py output): every pooled value, whole request and first-token cell agrees with my re-derivations
  (sections 0a-2 and 1); the per-pass geos match the run reports (spb 168.6 / 179.6, nie 39.9 / 39.8, z0 53.4 / 54.1, ...). It
  has no PF-D rows (pdz4a / pdz0); add them if Table 6.1c keeps them.
- DIFF 17: F.1 "a check's 3,000-3,900 launches cost 2-6 ms": at 1.24-1.56 us per launch that is 3.7-6.1 ms ("4-6 ms"), and
  KERNELS 6.3 says "about 1,000-4,000 kernels" per check (-> 2-6 ms). Align the two papers on one launch count.

### DRAFTER rows 7-8 and EVALUATION Table 1b (Mon 07:50-07:51 `date`)
- DRAFTER section 3.2 table, D-20b and D-23, re-derived from TL:data/d20_runs.jsonl (C 224, head b3, 17 prompts with both
  arms) and d23_runs.jsonl: 15.95 / 20.30 (+27.3 %), chat 11.38 / 15.07, whole request 7.11 / 6.42 and chat 9.18 / 11.61
  (+26.5 %), first token 3.94 / 3.94 s; D-23 29.43 / 28.65 (-2.7 %), 7.09 / 7.04, 3.80 / 3.82 s. OK.
- DRAFTER DIFF 3 (Appendix C row 7 only): "chat +26.0 % (6 of 6), math +18.3 %, agent +27.0 %" are means of per-prompt
  rates (TrainLab REPORT l1636); pooled per type: chat +32.4 %, math +19.8 %, agent +42.0 % (code -14.7 %).
- EVALUATION Table 1b = the result files: R-56 28 / 36 (math 4 / 12), 2,153 / 2,199; R-56b 16,341 tokens, 16,305 / 16,305;
  R-56c and R-56d 15,525 tokens, 15,489 / 15,489; NAX6-N 15,726, 15,690 / 15,690; card line check 12 / 12. OK.
- KERNELS R-rows spot-checked in Leap's raw files (Mon 07:51 `date`): R1 (logits 0 / 16, 16 / 16 rows differ), R4 (14.0 vs
  35.2, floor 7.1), R7 (0.776 vs 1.242 ms, 75 % of the 0.580 floor), R9 (E 24.51 -> best 9.82 = -14.69 ms), R10 (qmm27),
  R12 (45.0 / 38.8 / 13.9 / 2.4 %), R14 (= ycoh / ycoh2: 37.80 / 37.93, 21.10 / 21.28, 2.83 s), R15 (304 rows at 1,565),
  R18 (bf16 210+ rows) all present as stated. OK.

### MAIN DIFF 18-21 (section 11, Mon 07:52 `date`)
18. 6.2 l470-471 "PF-B gains +15-22 % and is not bit-exact" and 11.2 "changes the logits (0 of 7-28 hashes equal) and peaks at
    46.0-46.2 GB": stale. Since the seg-per-chunk fix, pdb38s / pdb32s (pfd.log Part B) equal PF-A in tokens and logits
    hashes (8 / 8, 28 / 28, 22 / 22) at x1.12-1.21 (500 tok/s at 7.9K; peaks 46.23 / 45.81 GB). KERNELS R20 and RELEASE
    already say exact; align MAIN (the "not bit-exact" numbers are the pre-fix pfb_report arms).
19. 11.4 "the chat stack (56.0 tok/s, +27 % in its run)": +27 % is over the certified config's CHS4 runs (44.2); in its own
    run (CHS5) zsgcr is +12.8 % over z0. "wait for a known multilingual benchmark": stale (card arm). "the X build (+2.9
    to +4.0 % on code)": pooled X over B is +6.0 % HumanEval (181.29 / 170.96) and +4.7 % MBPP (119.83 / 114.45); geo +4.2 /
    +2.9 %.
20. 11.5 "+5.8 % whole decode" (never waiting for cold experts): pooled from s1bb1-2 / s1bn1-2 it is 141.42 -> 150.18 =
    +6.2 % (geo +5.9 %); BETWEEN unchanged (the pre-registered line used the full period, +6.9 %).
21. 11.6 "Every measured row sits at 41.7-43.1 GB": the PF-D row is 43.69 -> "41.7-43.7 GB (43.1 without PF-D)".
- OK in 11: 20.4 GB, 65.5 % / 25.1 % resident (CT1 pfst sums), 0.32 s + GB / 7.33, 3.10 -> ~1.6 s at 9.2 GB [P], 2.91 ->
  2.26, 1.83 s, 19-31 %, 0.54-1.01 s, 1.9-3.4x, 0.1 GB trim, 3.8-5.2 GB/s (67.2 / 12.9 and 68.3 / 18.0), 87 % / 53 %.
22. 10.0 row 1 "142.1 tok/s whole decode" (the correction of 180.3): the geo of per-pass geos (F:2026-11-07_0500 says so);
    pooled from s1n1-2 it is 142.77 -> "142.8". 10.0b repeats DIFF 18 (PF-B "not exact") and DIFF 20 (+5.8 %).
- SMALL_RAM 4.1-4.3 and 11 (Mon 07:54 `date`): 10,316 / 10,316 rows (3,366 / 3,397 / 3,553) = DM:out/gate.txt; lines 1.34
  [0.87] / 5.42 [2.22] / 3.69 [1.63] % = 45 / 3,366, 184 / 3,397, 131 / 3,553 (lines_check_gate_top2.out); section 11:
  21.986 GB / 2.906 s = 7.57 GB/s, ~1.8 s at 12 GB/s, OPT 386.8 vs 621.6 = -37.8 %, ceiling 2.43 / (48.5 - 5.7 ms) = ~57
  tok/s. OK. Not re-derived: 3.0 and 5.1-5.3 (SB / BD sources, [P] replays).
- EVALUATION 4.6 / abstract "5-15 %" [P] (Mon 07:55 `date`): with exactly N x d discordant rows split 50 / 50, P(delta <= -5)
  = 5.5 % (N 100, 10 %), 15.1 % (N 100, 15 %), ~5-7 % (N 150, 15 %): the paper's figures. With binomial discordance: 7.6 /
  12.1 / 5.7 %; 6 independent orders 29-54 % (paper "~25-50 %"). The abstract's 5-15 % holds either way. OK.
- EVALUATION abstract counts = Table 2: four extra passes (HC8, S4, C30 HumanEval+ only, M1-B + COH IFEval only); six
  quality modes failed (RELAX A_E / C_E / B_F / B_G / T3ALL, ship8); STACK on the -5 boundary; every 12 GB candidate (T0,
  theta 0.8 / 0.9 / 0.95). OK.
- DRAFTER rows 6, 11, 15-18, 20, 21, 25 = their TrainLab result files (Mon 07:55 `date`): d17_ci (0.746, CI 0.703-0.791, ...),
  e2_read (9.37 / 9.46 / 9.57; 9.79 / 9.85), rerank0 (11.57 x1.209, 12.37 x1.292), mtppick_sim (9.58 -> 10.28, x1.073),
  d24_read (7.82 ms per row), d23h_read (12.60 with +G), chat150_read (x0.44-0.59, hybrids x0.89-0.98, oracle x1.012),
  head_bench (1.475 ms), mtp_3bit_vs_4bit (4.02 -> 4.09). OK (result-file level).
- KERNELS (Mon 07:57 `date`): R16 / R17 (pfb_seg: 41 rows at 1,100, ...), R21 (r40s3s: 0 / 64 rows, 0 of 655,360 elements), R25
  (r40s4c: 220 / 220 logits rows) present. DIFF 6 (minor): R13 quotes D:s7b_probe.txt (B1 20 / 20, B2 505,808-524,288,
  barrier 16 / 20, coherent 0 / 20); the also-cited s7b_probe2.txt repeat gives B1 19 / 20, B2 min 96, barrier 20 / 20. Give
  the range ("barrier 16-20 / 20") or cite probe 1 only. Its cost "+0.7 % / +26.3 %" is coh_check.txt's third run under a
  share lock ("rough speed"; earlier runs +1.5 % / +inf); label it rough. (The engine A/B later found coherent reads cost
  nothing on chat: ycoc / ycoc2 37.7 / 38.0 vs ycoh / ycoh2 37.8 / 37.9.)
- MAIN 9.1 (Mon 07:56): every number matches its ledger entry by ID (M-20 +50-65 us, ~3 ms; M-28 283 / 96 us, wake 44,
  dispatch 8; spacing 270 / 44 us; M-29 400 us, 97.5 %; follow-up 87-94 %). OK.
- RELEASE section 3 / 4 environment (Mon 07:56): macOS 26.5 (25F71), Python 3.12.13, MLX 0.32.2, mlx-lm 0.31.3, MTPLX 2.12.2,
  vendored nanobind 2.15.0 (NB_VERSION in the package), d22_b5.safetensors 276.6 MB ("277 MB"), l120_ft 1.141 GB. OK.

## 7. New rows since the first pass (Mon 10:31-10:34 `date`)
### BigMoE REL ROWS (L + COH, the shipped lossy config; RELEASE TBD-16 / rows 27 and 32), re-derived from data/r57/rl*_runs.json
- rlh 164.05 / 36.16 / 4.60 s (mean 4.70) / 42.94 GB, == AR 5 / 12, 7 ended before the cap; rlm 112.71 / 25.28 / 3.72 / 42.60;
  rli 45.40 / 22.26 / 3.12 / 42.42; rlt 78.91 / 37.87 / 3.20 / 43.17; rli_160 38.95 / 21.12 / 2.90 / 31.30; rlr 44.92 / 22.52 /
  2.91 / 42.42; IFEval pooled (rli + rlr, 24 rows) 45.16 / 22.39 / 3.01 s / 42.42 GB. REPEAT rlr == rli 12 / 12; C 160 == C 256
  12 / 12. All = WED_TESTS l1231-1262. vs L = S5' X / N45 (CERT + the bundle): -9.5 % / -5.9 % / -9.2 %; IFEval +4.0 %
  (45.16 / 43.44; the note's "+4.1 %" uses rounded inputs).
- Env check (data/stack): s5p_env = cert_env + 6 keys (PAUSE_EVERY_MS 50, DQ1, S0CARRY, COMMIT_SUB 9, PLE_LATE, SB_FIRST =
  the Sunday bundle); lcoh_env = s5p_env + REF2_NAX_COH=1; scripts/n45.sh and chs*.sh build L from s5p_env. So N45 L and CHS4
  y0 / y0b ran CERT + bundle (as EvalSuite / BigMoE noted 10:1x), and only S5 / S5' B are the certified flags themselves.
### EvalSuite card gate lcoh (RESULT_card_lcoh_2026-11-09): PASS confirmed
- GPQA Diamond, my own recount from the raw rows (scripts/numcheck/gpqa_recount.py; simple-evals answer pattern; gold from
  data/items/dec_gpqad.jsonl): 47 vs 44 % in both orders, arm only 6 / reference only 3, +3.00, p 0.5078. Identical.
- LCB v6, my own code extraction + sha256 key into the grade cache (lcb_recount.py; 0 cache misses): forward 40 vs 42 (2 / 4,
  -1.53, p 0.6875), reverse 38 vs 42 (2 / 6, -3.05, p 0.2891). Identical.
- IFBench (n 150): 5 / 10 -> -3.33, p 0.3018 recomputed from the counts (not re-graded: its checkers are EvalSuite's).
### Re-check of the 10:3x drafts (Mon 10:34-10:36 `date`; scripts/numcheck/diffnums.py against the 07:49 snapshot)
- MAIN: DIFFs 1-22 applied as sent; every changed number re-checked OK (68.5 GB store, +25.1 %, 6.0-6.1 / 6.5 kept, x1.06,
  PF-B exact x1.12-1.21 at 45.8-46.2 GB, 43-80 %, 6.04 s, 3.79 of 6.77, 1,750-2,500, ~550, 2-3 %, 12-23 %, 2 of 24, 5 %,
  +6.2 %, 41.7-43.7 GB, Appendix E's PF-D rows, F.1's 1-6 ms). OPEN (label, sent to the editor): the "certified config"
  wording on N45 L / CHS4 y0 rows (CERT + bundle): abstract l69, Table 6.1b chat / math rows, 5.3, 7 (Table 7.1's "lossy
  certified" column; "two runs of the certified config"), 11.4; and 6.1b's "What the package ships" line (L + COH = CERT +
  bundle + COH; card gate PASS 10:07).
- RELEASE rows 26-33 and the new bullets: every cell = the records (REL ROWS above; L = S5' X + N45; CERT = S5' B).
- SMALL_RAM 5.4 new cells (whole request, first token, totals) = records: x2c0 20.41 / 3.05 (2.60-3.85), x2c0b 20.78 / 3.02,
  x2qm1 21.26 / 2.94, x2qm2 23.00 / 2.96, x2qm2s 22.11 / 3.10, x2e 20.66 / 2.97 (2.25-3.70); 1,480 vs 1,580 tokens; 38.5 vs 36.5 s;
  z0 36.4 of 71.4 s, z0b 36.6 of 71.5 s. OK.
- KERNELS v1.2: OK. Its 6.4 MB column now counts weights + input + output (e.g. 17.69 + 0.47 = 18.2 MB), so the GB/s
  (243-247) were right; my DIFF 4's direction was wrong, the inconsistency was real.
- EVALUATION v1.1: 381 = 150 + 100 + 131; 867,659 tokens in 346 min = 41.8 tok/s; 1,139 per row (762 rows) vs ~590. OK.
- DRAFTER: l312 still "142.1" (-> 142.8 pooled); sent to MTP Head.
### Precomputed for MAIN's re-base on L + COH (editor 10:5x), Mon 10:37 `date` (scripts/numcheck/lcoh_derived.py)
| set | L + COH (REL ROWS) decode / request / first token (range), peak | vs Strata full-cache B: decode, request, first token, break-even |
|---|---|---|
| HumanEval | 164.05 / 36.16 / 4.60 (3.78-6.07) s, 42.94 GB | x3.49, x0.88, 6.82x slower, ~259 tokens |
| MBPP | 112.71 / 25.28 / 3.72 (3.35-4.88) s, 42.60 GB | x2.47, x0.68, 6.87x, ~243 |
| IFEval (2 passes) | 45.16 / 22.39 / 3.01 (2.41-3.81) s, 42.42 GB | x1.08, x0.66, 3.61x, ~1,244 |
| math | 78.91 / 37.87 / 3.20 (2.71-6.80) s, 43.17 GB | x1.68, x0.93, 3.16x, ~256 |
So: lossy decode x2.5-3.5 on code, x1.7 math, x1.08 chat; Strata still wins every whole request; peaks 42.4-43.2 GB.

## 8. The 12:00 full drafts (re-check Mon 12:01-12:02 `date`; diffnums against the 10:36 snapshot + MAIN's re-based sections)
- MAIN (frozen 11:5x), re-based on CERT / L / L + COH: abstract, Table 6.1b (L + COH rows + CERT's code rows), 6.1c (L's rows),
  Table 7.1 (lossy column = L + COH) and section 7's text all equal the records and lcoh_derived.py: 164.0 / 36.2 / 4.60
  (3.78-6.07), 112.7 / 25.3 / 3.72 (3.35-4.88), 45.2 / 22.4 / 3.01 (2.41-3.81), 78.9 / 37.9 / 3.20 (2.71-6.80); peaks
  42.4-43.2; x2.5-3.5 code, x1.7 math, x1.08 chat; first token 2.7-7.0x; break-even 260 / 245 / 255 / ~1,200 (mine 259 / 243 /
  256 / 1,244); whole request 20.8-38.9; lossy-32 39.0 / 21.1 / 2.90 s at 31.3 GB. OK.
- RELEASE l117-121 "the lossy configs run what the card gate ran" (LP:data/release/lossy_card_equiv.txt): reproduced the key
  part read-only: git diff e057abf (card) 0b4df8c (rel1) -- bigmoe ext is empty (ext last changed 5bb2743, 11-04); the
  package's configs/lossy-45.env carries the same KEY=VALUE set as data/stack/card_lcoh_env.txt (QOM set names aside, as
  stated). OK.
- DRAFTER l315-316: 142.8 pooled. OK. The other changed lines in all six papers since 10:36 are the related-work pass
  (arXiv ids, citations: the editor's check, not this one) and status lines.
- Second-tier checks, Mon 12:02-12:04 (`date`): MAIN 5.2's bisect (chs3: x3oc 3 / 12, x3oc2 4 / 12, x3sn 3 / 12, the other
  eight 12 / 12), "1 of 12 IFEval rows (CERT)" (CERT_chat.txt), 5.1's 2,888 copied tokens (sat_report: stackb accepted 2888),
  2's 67.8 % (F:2026-11-05_1700) all = their sources. DRAFTER row 19: the engine figure is kept per check (stb / stbb vs sts /
  stsb: 11.16 vs 10.01 pooled = x1.115, geo x1.082); whole decode x1.077 pooled (geo x1.056 = the report's +5.6 %): wording
  note sent (optional).
  - DRAFTER row 19 applied (MTP Head 12:1x) as "x1.08 geo / x1.11 pooled (11.16 vs 10.01; 1.114 unrounded)": right; my
    script's unrounded ratio was x1.114 too (my message's "x1.115" divided the rounded values, as KS did).

## 9. Leap's clean-clone FINAL (12:10-12:14), re-checked Mon 12:46-12:47 (`date`; scripts/numcheck/final_check.py on out/)
- Tokens 12 / 12: exact-45 / exact-32 prefix-equal to BM:data/n45/ar_ref_tokens.json (ifeval/1220 153 = 153; 3280 256 of
  265); lossy-45 / lossy-32 = rli_256 / rli_160 to full length (154, 139); chat-45 = pdz4a (192, 235); chat-32 = zs32 (184, 256).
- Lossy / chat vs the engine's own runs of the two keys: same checks and kept per prompt (4 / 4); whole decode x0.996-1.020,
  whole request x0.996-1.012 (= RELEASE l173). DIFF (tiny, sent): "first token within 0.1 s" -> medians +0.044 / +0.085 /
  +0.050 / -0.107 s, per prompt up to 0.119 s. (Exact configs vs N45 nie1 / e32: decode x1.005 / x1.002, request x1.03 /
  x1.02, first token -0.24 / -0.17 s: not claimed in the text.)

## 10. m1b card gate (running) and the PDF build check (Mon 13:10 `date`)
- m1b forward order (results/card_m1b_fwd/decode.jsonl, 381 rows, 13:05): GPQA Diamond re-graded by me: 42 vs 44 %, arm only
  4 / reference only 6, -2.00 pt, p 0.7539. LCB: 39 forward answers not yet in the grade cache (EvalSuite grades at the
  verdict); lcb_recount.py now reports ungraded keys instead of counting them as fails. Reverse + LCB at ~16:20.
- The orchestrator (13:0x): the number check covers the built PDFs too (digits unchanged vs .md). scripts/numcheck/
  pdf_digits.py (pdftotext or HTML; number-token counts after normalising minus, x and spaces) is ready and tested; paths
  and ETA asked of Breakthrough.

## 11. Public edition, build 13:41 (Breakthrough): digits PASS (Mon 14:31-14:35 `date`)
Method: scripts/numcheck/pdf_digits.py on each out/<slug>/<slug>.public.md (the build's own source) vs its PDF (pdftotext),
then a substring recount of every number whose count differs. Explained, not digit changes:
- page numbers on pages where pdftotext does not put the number last (MAIN p. 5, 7, 11, 21; DRAFTER 4, 9, 12-14; EVALUATION
  5-8; RELEASE 1, 2; KERNELS 10, 11; SMALL_RAM 7): each such number +1, exactly;
- keys broken after "_" (allowed by the build) and table cells interleaved by pdftotext (e.g. EVALUATION "chat_m1b_" /
  "0.388" / "2026-11-08"): the substring recount restores them;
- "10^9" renders as "10ˆ9" (U+02C6), read as a letter by the tokenizer;
- EVALUATION: two keys broken after the "-" inside a date (RESULT_card_lcoh_2026- / 11-09, RESULT_smallram_2026- / 10-22),
  which pdftotext re-joins as "202611-09".
After that every multi-digit number and every decimal matches in all six PDFs. Site pages (site/<slug>.html) vs each
abstract: every abstract number is on its page; the extras are the title / subtitle numbers (shown twice) and the date.
Still to do on the final build (~16:40): the same, plus public.md == source minus the manifest's dropped ranges.

## 12. MAIN 10.10 (FAR-CEIL / FAR-STATE) and RELEASE's new abstract (Mon 14:36-14:37 `date`)
- MAIN 10.10 = FableGod far/far_ceiling.txt and far_ridge.txt: 176 checks over 6 prompts (5, 48, 25, 32, 34, 32); K 266 / 147;
  PERFECT-KEPT 20.3 % (recall 0.68) / 26.3 % (0.76); PERFECT-ALL 33.7 % misses, stalls 5.4 %; OPT 55.3 % / 19.3 %; token
  LOPO recall 0.238-0.283 (layers 8-47), previous occurrence ~0.40; best token map 6.0 % (MEMO-4 + LOPO-10 + pop); ridge
  (leave-one-prompt-out) RIDGE-1 a0.3 top16 all16: recall 0.70 / precision 0.64, 14.6 %, stalls 42.3 vs 42.9 = -1.4 %; K 147
  best 11.9 % (RIDGE-3 a0.3 top10). OK. Note (sent): the 14.6 % is also the best of 8 settings tried at K 266 (9.6-14.6 %);
  say "best of the settings tried" there too. A re-run of far_ceiling.py (a copy writing to my scratchpad) was refused by
  its own guard: fanwatch PAUSE set 14:35 (heat); deferred until after m1b.
- RELEASE abstract (l11-21): exact 37.1-101.4 tok/s, shipped lossy 45.2-164.0, peaks 41.7-43.2 GB, Strata 0.5-1.0 s vs ours
  2.7-4.6 s (2.26 s chat as shipped), six configs: all = section 2 and the records. OK.

## 13. m1b card gate (M1-B + COH), verdict lines 16:06, re-graded Mon 16:06-16:07 (`date`)
| line | EvalSuite d4 | my re-grade |
|---|---|---|
| GPQA Diamond fwd | 42.0 vs 44.0 %, -2.00, p 0.7539 (4 / 6) | identical (gpqa_recount.py) |
| GPQA Diamond rev | 42.0 vs 44.0 %, -2.00, p 0.7905 (6 / 8) | identical |
| LCB v6 fwd | 29.0 vs 32.1 %, -3.05, p 0.3877 (4 / 8) | identical (lcb_recount.py, 0 cache misses) |
| LCB v6 rev | 32.1 vs 32.1 %, +0.00, p 1.0 (3 / 3) | identical |
| IFBench fwd | 44.0 vs 44.0 %, +0.00, p 1.0 (9 / 9) | p recomputed from the counts: 1.0 |
| IFBench rev | 41.3 vs 44.0 %, -2.67, p 0.5034 (8 / 12) | p recomputed: 0.5034 |
All six inside the rule (delta > -5, p >= 0.05): the card gate PASSES (EvalSuite's RESULT doc not yet written at 16:07).

## 14. MAIN methods-first (editor 15:1x-15:3x), checked Mon 16:07-16:09 (`date`) before the 17:15 freeze
- Rule check (scripts/numcheck/newtext_nums.py): every number in the new subtitle, abstract, 1.2-1.6, section 7 and section 9
  occurs in the 10:36 MAIN, except 7.45 GB (the volatile tier), which = the run logs' "R57_VTIER: 64 / 32 slots ... = 7.45 GB"
  (3,072 x 2,162,688 B + 1,536 x 524,288 B = 7.449 GB).
- Moved sections: old 7 -> App G, old 8 -> App H, old 9 -> 8, old 12 -> App I: numbers equal except renumbered headings,
  re-dated words and one citation (MTPLX "v2.11.2 per its CITATION.cff" -> "v2.12.2 at commit 9882703f, the version we ran":
  = the package's requirements.txt and the venv's direct_url.json).
- New text read against the verified values: abstract (96 / 96, 24 / 24, 24 / 24; 101.4 / 70.4 / 77.9 / 37.1 at 41.7-42.5;
  164.0 / 112.7 / 78.9 / 45.2 at 42.4-43.2; 2.7-4.6 s; 0.54-1.01 s; 2.5-3.5x), 1.2-1.6 (120.8B; 2.7-7.0x), section 7 (x0.75;
  2.5-10.7 kept; +19 %, +24.5 % on 19 chat prompts; -17.2 % / +8.3 %; ~-20 %; 232.6 -> 117.3; +13-18.7 %; 171-175 / 111-114;
  -7.2 to -8.0 ms; +25.1 %; +6.2 %; +3.7 / +2.9 %; 3 / 12 -> 12 / 12; +1 / +4.2 %; 2.91 -> 2.26 s; 1.9-3.4x; 10-16 ms + 0.15-0.17
  ms; 10.7 / 6.0-6.1 / 6.5 / 2.5; ~51 ms; 87 / 89 / 84 / 90 = EVALUATION Table 2's HC8 / RELAX / ship8 vs REF2-T), section 9
  (40.6-47.1; 40.8-41.2 GB; x2.5-3.5 / x1.7 / x1.08; x1.6-2.2 / x1.7; 2.7-7.0x; 245-260; ~340). OK.
- DIFF (sent): 1.5 "Our decode leads only on code and math" contradicts section 9 / App G (lossy chat x1.08, 45.2 vs 41.8).

## 15. Final-candidate public build 16:11 (Mon 16:33-16:35 `date`)
- Since the 16:09 copy: MAIN / EVALUATION carry m1b's PASS (EVALUATION's m1b row = my re-grade: IFBench 44.0 / 41.3 (p 1.0 /
  0.50), GPQA 42.0 / 42.0 (p 0.75 / 0.79), LCB 29.0 / 32.1 (p 0.39 / 1.0)); MAIN 11.4 "L by itself +6.0 % HumanEval, +4.7 %
  MBPP over CERT" = 181.29 / 170.96 and 119.83 / 114.45; RELEASE's conversion paragraph = LP:data/release/convert_full.txt
  (16:07:42-16:19:50; 48 layers in 571 s = 9.5 min; 11-14 s per layer, median 12; 48 layer byte compares + index.json).
- Digits, .md -> public.md -> PDF (pdf_digits.py; page-number filter now drops only a line equal to the page's own number,
  after it had dropped a real table cell "463" at a page end): MAIN, KERNELS, EVALUATION PASS end to end (public.md = source
  minus the manifest's dropped sections + the date line; the PDF's remaining +1s are page numbers above the last lines).
- DRAFTER and RELEASE: the PDFs equal their public.md, but DRAFTER.md (16:17:47) and RELEASE.md (16:20:27) changed after the
  16:11:4x build: rebuild asked of Breakthrough.

## 16. Build 16:53 (five papers + complete.pdf): digits PASS (Mon 17:04-17:05 `date`)
- All five: public.md = source minus the manifest's dropped sections + the date line ("9 November 2026"); each PDF's
  remaining count differences are exactly its page numbers that sit above the page's last lines (MAIN 4, 9, 20, 24; KERNELS
  10, 11; DRAFTER 9, 12-14; EVALUATION 6-8; RELEASE 1, 2). DRAFTER (16:17:47) and RELEASE (16:20:27) now match their sources.
- complete.pdf (82 pp): each part's numbers = its single PDF's, exactly (MAIN pp 3-32 2,906; KERNELS 33-45 1,188; EVALUATION
  46-57 993; DRAFTER 58-74 1,207; RELEASE 75-82 579); the pages that differ in text differ only in the order pdftotext reads
  table cells (same words). Cover + contents add only 3 / 33 / 46 / 58 / 75 (= the part ranges), "64 GB" (title) and the date.
- MAIN after the editor's 17:05:49 write (title / labels, Kamen's word): numbers equal the 16:53 build's public.md (only the
  date line differs). Re-check after the editor's 17:30 changes.

## 17. MAIN section 11 draft "The path to more speed" (~/Papers/PATH_DRAFT.md, 17:08), checked Mon 17:08-17:11 (`date`)
- Multiplication: no gain is multiplied across runs; the draft says so (11.3: never-wait "never run with L, so we do not
  multiply"; 12 GB bets "not multiplied"; chat 150 tok/s = a joint requirement 3.75 kept / 25 ms [P], "neither wall broken").
- Re-derived from raw tonight: chw19c 14.46 / 10.39, 449 MB per kept token; chw19p 13.73 / 9.97, 491; chw19h 16.48 (Fit12's
  per-prompt lines + phase logs) -> x1.053, 449-491 MB, x0.881 vs chw14ha 18.69. Map: qom_0.5_live_c147 PASS x3 (147 slots).
- Every other number = my verified values (sections 1-16): today's rows; L +6.0 / +4.7 %; 56.0 / +12.8 % / 44.1 ms; zs32
  51.33 vs 33.81; PF-D 2.91 -> 2.26 / 1.83 / ~1.6 [P] / 19-31 %; 1.9-3.4x / 0.1 GB; PF-B 12-21 % / 45.8-46.2 (0.8-1.2 GB
  over); x1.09 [P]; 81.7 vs 210.2 = -61 %; 2.906 s / 21,986 MB / 7.6 -> ~1.8 s; ~51 / ~40 ms, 15-20 ms, 75-85 %; 232.6 ->
  117.3; 48.5 -> 44.1; 83 / 41; +24.5 %; ~105-115 [P]; 0.10M = 8 % of ~1.3M, x1.014 vs the x1.03 line; 87 %; 18.5 ms; ~4 ms;
  -17.2 / +8.3 / ~-20 %; OPT 38-55 % (386.8 vs 621.6); 31-40 ms; 41.7-43.7, 1.3 GB, +0.1 %; 65.5 %; 3.8-5.2 GB/s; 0.54-1.01
  s; 28.8 vs 50.7; 87 / 89 / 84 / 90; 44.34; 41.07 / 37.43; 18.34; 12.76; 13.97; 171 (Splash, self-reported).
- DIFF (sent; small): (1) 11.2 first-token row: "chat as shipped 2.26 s: 37-101-token prompts read 20.4 GB" mixes PF-D on
  and off: as shipped (PF-D) a chat prompt reads 9.6 GB; the 20.4 GB (65.5 % resident) is without PF-D. (2) 11.2 kept row
  "math 6.6" -> 6.5 (pooled; MAIN 4.3). (3) 11.3 32 GB chat: zs32 is "not gated" (no card gate ran), not "gate pending";
  and the shipped lossy-32 (L + COH at C 160) chat row is 39.0. (4) 11.1 never-wait key: s1b_report prints +5.8 %; the pooled
  +6.2 % is BM:data/r57/s1b{b,n}{1,2}_256_runs.json (as MAIN 10.0b). Optional: (5) the ledger counts: TRIED.md's own section
  totals (one wall per line) are fixed-cost 315, SSD 303, bytes 168 (471 together), drafter 218, memory 101, prefill 41,
  quality 74, exactness 74, no "kept" section; state which rule the text uses. (6) "x0.881 on code" is against A' of an
  earlier run (chw14ha).
- 17:12 (`date`): PATH_DRAFT 014dc69d has fixes (1)-(6) (9.6 GB with PF-D vs 20.4 without; math 6.5; lossy-32 39.0 not gated,
  zs32 not gated; s1b runs.json key; ledger totals 315 / 303 / 168 / 471; chw14ha earlier run). One label left (sent): 11.3
  l50 / l53 "Measured, gate pending" for L by itself and the chat stack -> "not gated" (no gate runs; 11.1 says so).
- 17:13 (`date`): PATH_DRAFT bb35b712: no "gate pending" / "gate open" left; L, the chat stack and zs32 read "Measured, not
  gated". The section 11 draft is NUMBER-CLEAN. Next: the merged MAIN and the final build.

## 18. FINAL build 17:14 (editor's FINAL GO) and the merged MAIN section 11 (Mon 17:14-17:15 `date`)
- Merged section 11 (MAIN l906-971) = the checked draft bb35b712 + the editor's stated edits (24 GB rows out; zs32 51.3 / 33.8
  from c32_report; kept per check 2.5 / 6.5 / 10.7 from 4.3; F12L key). Outside section 11, MAIN's numbers are those of 17:05
  apart from MLX issue #3830 and its date in the new section 8 citation.
- DIFF (sent 17:14, needs a MAIN rebuild): the added "(at 45 GB, OPT misses 55.3 % / 19.3 % fewer than LRU at K 266 / 147;
  section 10.10)" mis-pairs far_ceiling.txt: K 266 (45 GB) OPT = 55.3 % fewer misses and 19.3 % fewer stalled layers; K 147
  (24 GB) = 43.6 % / 4.7 %.
- Build 17:14: all five public.md = source minus dropped + the date line; each PDF's remaining count differences = its
  unplaced page numbers (MAIN 4, 9, 12, 13, 19, 27; KERNELS 10, 11; DRAFTER 9, 12-14; EVALUATION 6-8; RELEASE 1, 2);
  complete.pdf (84 pp, parts 3-34 / 35-47 / 48-59 / 60-76 / 77-84) = the five PDFs number for number (3,092 / 1,187 / 992 /
  1,206 / 578). PASS, except MAIN's one content DIFF above (in the .md itself).

> (a quote from the project owner, kept private)
- MAIN section 11 OPT line now = far_ceiling.txt ("55.3 % fewer misses and 19.3 % fewer stalled layers than LRU at K 266,
  45 GB"). MAIN's numbers vs 17:05: only section 11 (= the checked draft + the stated edits), MLX issue #3830 and its date,
  and two "64 GB Mac" where "laptop" stood (the machine's 64 GB). "laptop": 0 occurrences.
- Five papers: public.md = source minus the manifest's dropped sections + the date line; PDF digits = public.md (remaining
  counts = unplaced page numbers: MAIN 4, 9, 12, 13, 19, 27; KERNELS 10, 11; DRAFTER 9, 12-14; EVALUATION 7, 8; RELEASE 1, 2);
  site pages carry every abstract number (extras: title numbers and the date).
- complete.pdf, 84 pp: parts 3-34 / 35-47 / 48-59 / 60-76 / 77-84 = the five PDFs number for number (3,093 / 1,189 / 995 /
  1,208 / 580); cover + contents add only the start pages (3 / 35 / 48 / 60 / 77), "64 GB" and the date.
- Re-GO builds 17:33 / 17:34 (MAIN only; checked 17:35 `date`): MAIN's changes are citation details (MLX PR #1773 with "18 Jan
  2025" replaces issue #3830) and App. F.5's provenance note (69 us / 230 us unchanged, "from runs that were not saved").
  public.md = source minus dropped + the date line; the PDF's remaining counts = its unplaced page numbers (4, 9, 12, 13, 19,
  27); complete.pdf (84 pp) parts = the five PDFs; contents 3 / 35 / 48 / 60 / 77. PASS. Hand-off to Leap 17:46: clear.

## 20. Pre-check for MAIN section 12 "What is new in this work" (orchestrator 17:5x; due 18:25), Mon 17:51 (`date`)
The sweep's seven "new" rows (EvalSuite, ~/Papers/NEW_FINDINGS_SWEEP.md), against their raw sources:
- Row 1 SSD wake (MoonShots data/ssdwake.txt): burst 896 / 900 / 910 us at 0.25 / 0.5 / 1 ms vs ~847 base = +49-63 us
  ("+50-65"); 965 at 20 ms = +118 ("+100-120"); 270 us / +44 us / 400 us half / -3 ms per check = M-28 / M-29 entries (sec. 1).
- Row 4 constant address space: R32 = BM pdz4a_256.txt (3175ef3) error lines; 12 / 12 tokens, 721 / 721 hashes = pdz4a vs pdz0.
- Row 14 QUEUEMIX (data/ssdmicro3.txt): +165.2 us per mixed forecast record; throttle +3.1 us; ~51 us wake saved; throttled
  reads end ~2.2-2.5 ms later.
- Row 17 P-frame reuse (decision_map NIGHT1 / NIGHT2): 0.14 / 0.10 vs 22.7 / 18.9 %; 12-35 at once 1.03 %; chained 8.11 %;
  attention 29.0 %, GDN 88.5 %.
- Row 30 FAR: section 12 above. Row 34 matrix-unit order (Leap nax_order_probe.txt): 1,024 tests x 512 outputs; 45.0 / 38.8 /
  13.9 / 2.4 %; relaxed precision identical; integer mode 524,288 / 524,288 bitwise in every order.
- Row 40 union law (findings 2026-11-05_1715): 276 / 218 / 163, 258 / 202 / 151, 226 / 177 / 133; B4 x0.55, B8 x0.35;
  |cos| median 0.086-0.090 vs chance 0.09. Note: per doubling the drop is 21-25 % (log2 slopes -0.34 / -0.42), so "~25 %"
  is the upper end and "union ~ k^0.67" the lower slope; a [P] on [K] routing (36 map windows), not an engine measurement.
- MAIN section 12 (l976-1182, in MAIN 17:54:13), checked 17:54-17:55 (`date`): all 160 numbers occur in the verified MAIN
  or the pre-checked sweep rows (12.x section numbers aside); read in context: 12.1 (ssdmicro.txt: 0.10 ms -6.1 / +7.8 us =
  no penalty, 0.25 ms +48.9 / +55.3), 12.2-12.4, 12.5 (+ MoonShots FINDINGS item 3: OPT 37-49 %, 1-2-pass 7-13 %, online
  <= ~6 %; "a fifth of the stalled layers (OPT, 19.3 %)" = K 266), 12.6, 12.7 (+ item 4: k^0.67, 32 tokens 3.1x). Every
  explanation is marked [G]. PASS, no number or source errors. Optional wording: 12.7 "about 25 % per doubling" is the top of
  21-25 % (k^0.67 gives ~20 %).
- Section 12 revised 17:55:36 (QUEUEMIX dropped as not new; 12.1-12.6; "about 21-25 %"; two refs; intro "six findings ...
  (12.3)"): no number outside the verified sources. FINAL build 17:55:46-17:56:02 (checked 17:56 `date`): five public.md =
  source minus dropped + the date line; PDF digits = public.md (unplaced page numbers only: MAIN 12, 13, 19, 30; KERNELS 10,
  11; DRAFTER 9, 12-14; EVALUATION 7, 8; RELEASE 1, 2); complete.pdf 87 pp, parts 3-37 / 38-50 / 51-62 / 63-79 / 80-87 = the
  five PDFs; contents 3 / 38 / 51 / 63 / 80. PASS. Reported to the orchestrator, Breakthrough and the editor.

> (a quote from the project owner, kept private)
could be improved"), checked Mon 18:08-18:09 (`date`) against the 18:04 snapshot
- Credits (scripts/numcheck/credit_check.py, sections 4 and 8): 0 credit tokens lost (names, arXiv ids, PR / issue numbers,
  URLs of the old rows all present in the new tables or the References).
- Numbers: the added ones are all verified values (12.1 44 of 283 us [M-28], 97.5 % / 87-94 % [M-29], +12.1 %; 12.2 12 / 12,
  721 / 721, 2.91 -> 2.26 s, BM:scripts/pfd_unit.py l147 = the placeholder test; 12.3 x0.91, 8.11 %; 12.4 +6.2 %, 44.74 vs
  42.12); the removed ones belong to the findings that left section 12 (FAR, the union law) and two references. EVALUATION:
  one date removed. Intro "one of the four ... (12.3)" consistent.
- FIX (sent): 12.3 "The per-block form flips only 0.10-0.14 % in layers 12-35" -> per block (12-23 0.14 %, 24-35 0.10 %);
  both together 1.03 % (NIGHT2). Optional: 12.2 "rules out polling" rests on an unsaved run + [G].
- FINAL build 18:12:04-18:12:20 (checked 18:12 `date`): the 12.3 fix in word for word, 12.2 "argues against"; five public.md =
  source minus dropped + the date line; PDF digits = public.md (unplaced page numbers only: MAIN 9, 12, 13, 19, 29; KERNELS
  10, 11; DRAFTER 9, 12-14; EVALUATION 7, 8; RELEASE 1, 2); complete.pdf 86 pp = its parts; contents 3 / 37 / 50 / 62 / 79.
  PASS. Reported to the orchestrator, Breakthrough and the editor.
- Open for MAIN v2 (BigMoE 18:11, agreed not tonight): the sync-free gate's "+35 %" (4.4, section 7) is the steady metric;
  whole decode +41.5 % (mean of 6, BM:data/r57/r55l_summary.txt) to re-derive pooled. F.5 "about 20-23 us": one raw run's
  median is 27.4 (LP:data/w1_pickup_P.txt).
- POST-PUBLISH check of the published build 18:32:52-18:33:07 (public 18:35), Mon 18:55 (`date`): the 18:23 / 18:33 re-GOs
  added only wording and the machine line "MacBook Pro (M5 Pro, 18-core CPU, 20-core GPU, 64 GB, 2 TB SSD)", verified on the
  machine (hw.ncpu 18; system_profiler: Apple M5 Pro, 20 GPU cores; 64 GiB; disk0 2.0 TB). Five public.md = source minus
  dropped + the date line; PDF digits = public.md (unplaced page numbers only). PASS.

## 22. W24 / W24b (24 GB -> 120, Kamen's publish trigger), number check Mon 21:02 (`date`)
- w13t (the gate config, card_w24_env.txt, 8f76fc3) HumanEval: 116.45 tok/s pooled, 12 prompts 76.0-141.6, peak 23.915 GB =
  w24b_report "116.5 / 23.92". 3 % under 120.
- Two report rows are single-prompt stalls (both the arm's last prompt, 19:44 / 19:48, around Leap's chat smoke / Kamen's chat
  start; 'paused' empty): w13tm MBPP 29.55 = 11 prompts 72-117 + mbpp/319 at 4.8 tok/s (43 s) -> 90.5 without it; w12rh HE
  14.20 = HumanEval/63 at 1.6 tok/s (164 s) -> 110.4 without it. Do not quote w13t's MBPP without a re-run. Sent to BigMoE
  and EvalSuite. Machine at 21:02: Kamen's chat (Python 20 GB, 100 % CPU), swap 31 GB.
- CORRECTION 22:39 (`date`; BigMoE 22:37, verified): the screen DID flag both rows. Their runs.json 'bad' field reads 'PAUSED (161.7 s
  waited, 0 gaps)' (w12rh, HumanEval/63) and 'PAUSED (40.2 s waited, 0 gaps)' (w13tm, mbpp/319): PAUSE-file waits (heat), not
  engine stalls. The W24b report counts them as heat-freeze rows; its 'without them' 110.4 / 90.5 match my re-derivation.
  Never quote 29.6 / 14.2.

## 23. Fit12's S1u (12 GB one-call check, union-only cut; Kamen's go 22:3x), number check Mon 22:56 (`date`)
Source: ~/Fit12/data/scr/s1u (s1uh.txt, s1ue.txt, s1u_read.txt, s1u_rows.txt); script scripts/numcheck/s1u_check.py.
- CONFIRMED from the per-prompt lines (16 / 16, no PAUSE inside, rc 124 = the 15-min stop during the reference arm):
  whole decode 41.23 pooled (41.79 geo; the read-out's 41.22 is rounding), whole request 23.21, kept 10.32 per check
  (3353 tokens / 325 checks), check 250.3 ms (81.33 s / 325), 138 MB per kept token, peak 11.335 GB. It ran under a share
  lock beside Kamen's apps, so it is a rough run.
- [P], not timed: the split into SSD read 119.5 / non-read 130.8 ms, and the wide call's 108.2 ms outside reads, are
  GB x 83.7 ms/GB (the chw16 fit's marginal read cost; s1u_read.py l.49 says [P]).
- DIFF: "x3.235 pooled / x2.788 geo" (s1u_read.txt) and "S1u is x3.2" (BT:docs/PRICE_12GB_90_CODE.md l.201) use all 8
  reference prompts, including HumanEval/6 (VOID: a 40.4 s PAUSE inside), and unpaired sets (16 vs 8). Paired on the
  7 common prompts: S1u 42.65 vs exact 17.87 = x2.39 pooled (x2.42 geo). The reference over its 7 counted prompts: 17.87.
- DIFF: s1uh.txt's "= ex tokens True (first divergence None)" is vacuous. fit_onecall.py:301 compares an arm to itself
  when no ex arm ran in the same process. Real token equality vs tonight's exact arm: 4 / 7 (HumanEval/156, /91, /79
  differ), as expected for a lossy cut.
- First token 4.18 s median follows paper_rows.py's definition (prefill + check 0); prefill alone: median 3.87 s (3.56-5.16).
- Told the orchestrator, Fit12 and Breakthrough at 22:55. Verdict unchanged: outside reads FAIL (> 90), bytes PASS
  (<= 150), so re-price before S2.
- APPLIED 22:58 (`date`): Breakthrough 2b82985 (PRICE doc: x2.39 / x2.42 paired, marked CORRECTED; the split labelled [P]);
  Fit12 23:0x (s1u_read.py pairs PAIR-ok prompts: 42.64 vs 17.87 = x2.386 / x2.422, tokens 4 / 7; v1 kept as s1u_read_v1.txt;
  fit_onecall.py prints 'n/a' without an ex arm; [P] marked in its read-out, PLAN, TRIED, findings). 42.64 (sum n / sum
  decode s) is the precise figure; my 42.65 came from per-prompt tok/s rounded to 2 decimals. Fit12 notes the same vacuous
  field in fit_t3.py's single-arm lines (chw19c / h, ex16); its verdicts used chw13_read's cross-run comparisons.

## 24. W24c / W24d (24 GB -> 120: the gate's config), number check Tue 02:02 (`date`), after the fact
My account was at its 5-hour limit from ~00:0x to 02:00, so the orchestrator gave EvalSuite the final word at 01:33
(w13b, card_w24d_env.txt, blob e06f495e; EvalSuite amendment 12d). Checked now with scripts/numcheck/w24c_check.py
(criteria fixed at 23:08, before any result): NOTHING DISAGREES.
- W24d (same-run ABBA on HumanEval; 12 / 12 prompts clean in every arm):
  - w13b (w13e / w13f): 130.20 / 130.46 tok/s, pooled 130.33; peaks 23.819 / 23.781 GB.
  - 14-row (w14c / w14d): 126.35 / 125.48 tok/s, pooled 125.91; peaks 23.828 / 23.782 GB (controls 0.7 % apart).
  - w13b over the 14-row config: +3.51 %. Tokens identical within the pair (12 / 12) and by lh hash.
  - PASS on my 4 checks (>= 120.0, clean rows, peak < 24.000, same prompts).
- W24c: 14-row (w14a / w14b) 125.84 / 126.38, pooled 126.11; peaks 23.808 / 23.769; controls w13c / w13d 120.91 / 121.38
  (0.4 % apart). The gated w13t's w13d peaked 24.006 GB (HumanEval/68), over the line: the orchestrator's reason to switch.
- CAVEAT for every "under 24 GB" claim: the w13b gate's own lifetime peak reached 24.813 GB on a GPQA prompt's prefill
  (EvalSuite 01:58; LCB's longer prompts may go higher). "Peak < 24 GB" holds on HumanEval-length prompts only. SMALL_RAM
  needs the prompt-length limit, as the editor noted for 32 / 45 GB (34.0-35.5 GB at 32 GB on 10K / 30K prompts).
- S1p (Fit12, 00:01:48-00:04:14, before the gate), checked 02:06 (`date`) from s1ph.txt / s1p_prof.txt: 41.79 tok/s pooled
  (geo 42.44), request 23.85, peak 11.314 GB, tokens = S1u's on 16 / 16 (sha), so the run is deterministic. MEASURED split [K]: the wide
  call's loader idle (non-read) is 102.4 ms (median 103.2) of 217.4 ms; io 113.4 ms for 1.323 GB = 11.67 GB/s. S1u's 90 ms
  line fails on the measured split too (102.4 vs the [P] 106.9-108.2).

## 25. SMALL_RAM's 24 GB fill + Leap's pkg_w24c (Wednesday release), number check Tue 10:35 (`date`)
Scope (orchestrator 10:3x): SMALL_RAM's new 24 GB numbers (W24d, W24e, W24f, the gate's 25.112 GB peak, the ~560-token
label) and ~/Leap/release/out/pkg_w24c's docs. Base: ~/Papers/.SMALL_RAM_1110_pre24.md -> SMALL_RAM.md (10:30), 119 new or
changed numeric lines. Script: scripts/numcheck/sr24_check.py; GPQA / LCB re-grade with gpqa_recount.py / lcb_recount.py.
VERDICT: every number CHECKS OUT. Three wording points do not:
- CONFIRMED (STYLE 3 pooled):
  - W24d HE w13b: 130.33 (geo 130.3; passes 130.2 / 130.5), request 40.8, first token 3.54 (3.20-4.78), peak 23.82.
  - W24f MBPP: 91.25 (geo 93.7; passes 90.0 / 92.6), 26.1, 3.34 (3.09-4.62), 23.74; tokens 12 / 12 equal between passes.
  - W24f IFEval: 30.58 (geo 28.7), 18.1, 3.09 (2.45-4.16), 23.66. w13t controls: MBPP 92.62, IFEval 29.74.
  - W24e: 1024 peak 25.025, first token 5.7-14.6 s; 512 -> 24.345 GB at x1.31-1.62; 256 -> 23.954 GB at x1.95-2.89. Tokens
    equal to 1024 on 0 / 6 and 1 / 6; hashes on 0 / 6 for both. The 1024 repeat matches 6 / 6.
  - The gate: 40 processes, 764 prompt runs, 22 over 24.0 GB, max 25.112 GB at 2,801 tokens (gpqad/recnTTKdBzfuoZ7w7),
    shortest first-crossing prompt 561 tokens (lcb6/abc397_e), 14 IFBench-only processes all under (max 23.813, prompts
    <= 206). Scores re-graded: GPQA 47 vs 44 in both orders (6 / 3, p 0.5078); LCB fwd 29.8 vs 32.1 (2 / 5, p 0.4531), rev
    32.8 vs 32.1 (4 / 3, p 1.0); IFBench 40.7 vs 44.0 (from the counts).
  - Strata at 24 GB (SE:h2h/results_low24, B1 / D1; first token = wall TTFT): HE decode 26.14 / 26.22, request 21.97 /
    21.29, TTFT 1.52 / 1.66; MBPP 27.02 / 27.34, 20.36 / 20.29, 1.31 / 1.31; IFEval 28.36 / 26.87, 21.02 / 20.32, 1.65 /
    1.46; peaks 23.748 / 24.143 GB. The ratios in 6.5 (3.3-5.0x, 1.3-1.9x, 1.1-1.2x, 1.1x, 1.9-2.5x) follow.
  - pkg_w24c: RESULTS.md l.14, README l.24 (about 130 / 91 / 31; 560; 25.1), INSTALL l.6-8.
- NOT CHECKING OUT (sent to the orchestrator, the Editor and Leap):
  1. SMALL_RAM l.67-68 and l.683 say smaller chunks keep it under. Only 256-token chunks do; 512-token chunks reach 24.345 GB.
  2. SMALL_RAM l.677-680:
     - The '424 prompts ... stayed under' sentence is true by construction, since a lifetime peak only rises.
     - The 592-token prompt (lcb6/abc397_f) ran first in a fresh process both times (24.112 / 23.954 GB). That is
       run-to-run variation, not 'process state'.
  3. pkg_w24c LIMITS.md has no 24 GB line: the long-prompt peak and the first-token gap vs Strata are missing. Its l.12-13
     are superseded by the PF-D long-prompt results if the erratum ships with it.
