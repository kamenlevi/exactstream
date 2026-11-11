# Fit12 plan and pre-registrations (speed of the <12 GB setup)

Started Fri 2026-10-23 16:45 (account-2, Pro). Owner of B12 quality: Breakthrough; verdicts: Eval suite; chief: NewIdeas.

## Target (orchestrator 17:30): B12LL = B12 + B12_LATE3=2,10 + B12_NORM=mass + MTP 3:3:0
- Eval suite run 1: PASS 5/5 (~/EvalSuite/results/report_b12ll.txt). Speed (Breakthrough day_spd, 00:30 Fri):
  MTP code 21.2 tok/s (3.37 tok/pass, 159 ms/pass, 575 waited reads/pass, SSD 7.3 GB/s), prose 13.1; eager 14.2 / 13.9.
- Code reading (bt_b12.py block): under MTP every pass is a 4-row verify; FL_MR waits for ranks < NF = max(2, 10, 2) =
  10, so every verify row uses its exact top-10, all read and waited. The late path (S = 1) is unused under 3:3:0 with
  the guard off. So B12LL + MTP = v3 natural routing at 10 % residency (C_m 51 / C_l 20), HC8 mixers, decoupled
  prefill, no prefetch, no LFU. Its outputs do not depend on residency or read timing.
- Consequence: residency / timing levers (JIT prefetch, LFU, async submission) keep B12LL's outputs (check: same sha);
  the 24 GB per-pass budget (FM1 b16, judged 5/5 vs v3 natural) applies with the same routing semantics.
- Marginal cost per miss: 24 GB v3 natural ~0.19 ms (100 ms/pass, 332 misses, ~37 ms compute); B12LL ~0.21 ms
  (159 ms, 575). Time per pass ~ 40 ms + 0.2 ms x misses.

## F1 pre-registration (written 17:55, before any run). Same lock, code + prose, 128 tokens after a 16-token warm-up.
Baseline = B12LL as the suite runs it (MTP_C 64, MAXK 80, lookup off, guard off, decoupled prefill 512/4096) + cache 64 MB.
| arm | exact vs B12LL? | prediction (tok/s vs baseline) | kill / keep line |
|---|---|---|---|
| width k=2 / k=1 | outputs equal (natural routing) | k=2 -5..+10 %, k=1 -15..0 % | keep if >= +5 % on both |
| ASYNC_EVERY=1 | same sha | +0..+5 % | keep if >= +3 % |
| JIT W=12 (PF_JIT, QD 4, victim slots) | same sha | -5..+20 % (C_m 51 is small: victims hurt) | keep if >= +8 % |
| budget b16 | NO (FM1 semantics) | +25..+45 % (misses x0.56-0.68) | keep if >= +20 %; quality = suite |
| b16 + budget-aware JIT | NO | +30..+55 % | keep if >= b16 + 8 % |
Checks: baseline within 10 % of day_spd (21.2 / 13.1); shas equal across exact arms; footprint <= 12 GB.

## F1 RESULT (18:21-18:25, one quiet lock; ~/Fit12/data/f1/*.txt; 128 tokens; per pass = verify incl. commit)
| arm | code tok/s (tok/pass, verify ms, waited/pass) | prose tok/s (same) | sha vs base | verdict vs pre-reg |
|---|---|---|---|---|
| base = B12LL as judged | 20.5 (3.37, 156, 580) | 12.5 (2.17, 165, 623) | - | day_spd 21.2 / 13.1: within 3-5 % |
| k=2 | 20.8 (2.91, 135, 479) | 13.8 (2.06, 143, 519) | code same, prose DIFFERS | +1.5 / +10 %: below keep line |
| k=1 | 17.5 (1.91, 106, 330) | - | code differs | -15 %: dead |
| ASYNC_EVERY=1 | 21.3 (150 ms) | 13.0 (157 ms) | same / same | +4 / +4 %: KEEP (exact) |
| JIT W=12 | 19.9 (161, 649; pf 392 issued / 219 used) | 12.4 | same / same | -3 / -1 %: KILLED |
| JIT W=6 | 20.5 | 12.7 | same / same | 0 / +2 %: KILLED |
| budget b16 | 26.0 (3.20, 115, 341) | 17.6 (2.25, 118, 356) | differs (non-exact) | +27 / +41 %: KEEP, quality = suite |
| b16 + budget-aware JIT 12 | 25.3 | 17.3 | = b16's | below b16: KILLED |
| LFU 0.99 (v3g port) | 20.6 (154, 567) | 12.4 (166, 642) | same / same | +0.5 / -0.8 %: KILLED at C_m 51 |
| v3g, LFU off (control) | 20.4 (156, 580) | - | same | the copy = v3f |
- Widths are not output-invariant on this path (prose k=2 vs k=3, code k=1): batch-variant verify numerics somewhere.
- JIT at C_m 51: prefetch victims evict experts the pass needs (waited +12 %), SSD 9.8 GB/s mostly wasted reads.
- Peak footprint 12.85 GB in every arm: my env copied the suite's (no BIGMOE_PF_GROUP=64; Breakthrough: PF_GROUP takes
  B12 + MTP 12.9 -> 11.5 GB at the same sha). F2 measures with it.

## F2 pre-registration (18:30, before the run): async + PF_GROUP=64 in every arm
| arm | prediction | line |
|---|---|---|
| base (async) | code 21.3, prose 13.0 (= F1 async); peak footprint 11.2-11.6 GB | footprint > 12.0 = trim needed |
| b16 | code ~27, prose ~18.3 | |
| b18 / b20 | +20-30 % / +12-22 % vs base | menu for the suite (gentler cuts) |
| KC 10, code k=3 | sha = KC 20's (exact for B12LL) | differs = KC 10 is not B12LL |
| KC 10, k=7 (8 rows): natural | verify 220-250 ms/pass (misses ~1.6x of 4 rows) | |
| 8 rows + b32 / b24 / b16 | ~170 / ~150 / ~125 ms/pass | |
| b16 + MTP k=5 / k=7 (KC 10; added 18:45) | budget caps the pass set, so wider passes add compute, not reads: code k=5 +10-20 % over b16 k=3, prose 0-10 %, k=7 <= k=5 | keep if >= +8 % on both |
DFlash2 bound from these: tok/s ~ tau8 (code 6.15, x ~0.95 under a budget) / (verify8 + ~6 ms drafter). Build DFlash2 at 12 GB
only if that bound is >= 1.2x the best MTP config on code (b16 + async ~27 tok/s): needs verify8 <= ~170 ms.

## F2 RESULT (18:31-18:35, one quiet lock; ~/Fit12/data/f2/*.txt; async + PF_GROUP=64 everywhere; 128 tokens, one text each)
| arm | code tok/s (tok/pass, verify ms, draft ms, waited) | prose | footprint | sha |
|---|---|---|---|---|
| base (B12LL + async + PF_GROUP) | 21.1 (3.37, 151, 8, 580) | 12.3 (2.17, 167, 10, 623) | **11.28 GB** | = F1 base (both) |
| b16 | **27.8** (3.20, 107, 8, 341) | **18.9** (2.25, 109, 9, 356) | 11.35 | = F1 b16 (reproducible) |
| b18 | 26.3 (3.20, 113, 8, 375) | 16.4 (2.37, 134, 10, 416) | 11.35 | code = b16's |
| b20 | 22.8 (3.20, 132, 8, 482) | 17.0 (2.29, 125, 9, 445) | 11.35 | |
| KC 10, natural k=3 | 18.5 (3.20, 165, 8, 666) | - | 11.35 | DIFFERS from KC 20 |
| KC 10, natural k=7 (8 rows) | aborted: `bigmoe: 58 experts > C=51 at layer 0` | | | |
| KC 10 + b32 / b24 / b16, k=7 | 21.8 (4.41, 185, 18) / 25.1 (4.27, 152, 17) / 23.5 (3.76, 140, 20) | b16: 15.3 (2.42, 136, 22) | 11.35 | |
| KC 10 + b16, k=5 (6 rows) | **34.3** (4.57, 121, 13, 390) | 17.5 (2.33, 119, 14, 385) | 11.35 | |
Read-out against the pre-registration:
- PF_GROUP=64 takes B12LL from 12.85 to 11.28 GB at the same shas (prediction 11.2-11.6): B12LL fits 12 GB only with it.
- b16 +32 % code / +54 % prose over the same-env base (prediction ~27 / ~18.3: met). b18 +25 / +33 %, b20 +8 / +38 %.
- KC 10 is NOT B12LL (prediction wrong): other tokens and +15 % waited. With KC 20 the post lists ranks 10-19 too and the
  loader protects their resident copies from this call's evictions; those are often the next tokens' top-10. A free,
  exact-by-construction cache rule worth keeping in mind (protect near-miss candidates).
- Natural 8-row passes cannot run at C_m 51 (the union of 8 rows' top-10 exceeds the pool). Wide passes at 12 GB need a budget.
- 8-row verify: b16 140, b24 152, b32 185 ms (prediction ~125 / ~150 / ~170). MTP drafting costs ~2.5-3 ms per draft.
- Width under b16 (KC 10): code k=5 34.3 tok/s vs k=3 (KC 20) 27.8; prose k=5 -7 %, k=7 worse on both. Non-exact arms
  write different texts, so tok/pass mixes text and acceptance (one 128-token sample each): per-pass times are the firm part.
- DFlash2 bound (code): 6.15 x ~0.95 / (0.140-0.152 + 0.006) = 37-40 tok/s = 1.35-1.45x b16 k=3 but only ~1.1-1.15x the one
  b16 k=5 sample. Needs the 5-tap verify capture, the drafter loop, +0.35-1.2 GB. PARKED until the main track integrates
  DFlash2 (BigMoE / SpecDecode) and a budgeted adaptive MTP width is measured on several code prompts.

## F3 pre-registration (18:50, before the run): budgeted MTP width, 4 prompts (code, prose, math = GSM8K 0, write = an
IFEval-style instruction), b16 + async + PF_GROUP, 128 tokens each; spec125.AdaptiveWidth seeded with F2's 12 GB verify times
| arm | prediction vs b16 fixed k=3 (KC 20) | line |
|---|---|---|
| KC 20, AW max 3 | geomean 0..+8 % (narrower passes on uncertain prose / write) | |
| KC 15 (MAXK 92), fixed k=5 | code / math +10-20 %, prose / write -5..-10 % | |
| KC 15, AW max 5 | best: geomean +8-15 % | adopt a width policy only if geomean >= +8 % and no prompt < -5 % |
| KC 10, AW max 7 | <= KC 15 AW (KC 10 costs +15 % waited) | |
Natural (no budget) math / write rows give each prompt a b16 speed-up reference (code / prose from F2).

## F3 RESULT (19:21-19:26; ~/Fit12/data/f3/*.txt; async + PF_GROUP; tok/s, 128 tokens per prompt)
| arm | code | prose | math | write | geomean | vs b16 k=3 |
|---|---|---|---|---|---|---|
| natural B12LL k=3 | 21.2 | 11.0 | 16.8 | 14.5 | 15.4 | -32 % |
| **b16 k=3 (KC 20)** | **27.8** | **18.8** | **25.1** | **19.4** | **22.5** | - |
| b16 AW max 3 | 24.9 | 18.6 | 22.5 | 19.1 | 21.1 | -6 % |
| b16 KC 15 k=5 | 23.6 | 18.0 | 27.8 | 19.6 | 21.9 | -3 % |
| b16 KC 15 AW max 5 | 25.0 | 18.6 | 27.2 | 19.6 | 22.3 | -1 % |
| b16 KC 10 AW max 7 | 26.2 | 19.2 | 27.6 | 19.3 | 22.8 | +1 % |
- Pre-registered line (geomean >= +8 %, no prompt < -5 %): no width policy passes. Keep fixed 3:3:0 (the suite's config).
  F2's code k=5 34.3 tok/s was one lucky text (4.57 tok/pass; here 3.12-3.28). CORRECTION to F2's DFlash2 parking: the
  right MTP reference is b16 k=3, so the DFlash2 bound is 1.35-1.45x on code and ~1.35x on math (above the 1.2x line).
- b16 vs natural: code +31 %, prose +71 % (this run's natural prose was slow: 11.0 vs 12.3 in F2), math +49 %, write +34 %.
Per-pass breakdown of exact B12LL (natural, k=3; loader `slow_s` = time from seeing a gate row to releasing it, calls with misses):
| prompt | verify ms | loader stall ms | stall share | MB / pass | SSD during the stalls |
|---|---|---|---|---|---|
| code | 150 | 107 | 71 % | 1138 | 10.7 GB/s |
| prose | 185 | 143 | 77 % | 1213 | 8.6 GB/s |
| math | 180 | 137 | 76 % | 1516 | 11.1 GB/s |
| write | 161 | 118 | 73 % | 1265 | 10.8 GB/s |
- ~85 of 99 gated calls per pass stall; each moves ~7 reads (13.4 MB) in ~1.26 ms = 10.6 GB/s: ~0.25 ms latency + transfer near
  the SSD's rate at that queue depth. So bytes bind (~80 %), latency ~20 %. Non-overlapped compute ~43 ms/pass, MTP draft 8-11 ms.
- b16: verify 106-115 ms, stall 65-75 ms (62-66 %), ~9.5 GB/s during stalls.
- Exact ceiling at 12 GB (k=3): io fully overlapped with compute ~29 tok/s code; io at 14.6 GB/s ~26; both ~39 (today 21).

## DF1 pre-registration (19:45, before any run): DFlash2 on B12LL at 12 GB, 3 HumanEval+ + 3 GSM8K tf_gen items, 160 tokens
Prototype ~/Fit12/scripts/fit_dfl2.py: B12LL without the MTP head; DFlash2 (8-bit unless DQ=16) bound to B12's embedding and
lm_head; taps 5/15/25/35/45 (HC mean) from every forward; spec125's snapshot / capture / commit verify loop.
| arm | prediction | line |
|---|---|---|
| ref_nat: exact B12LL, MTP k=3 | code ~21, math ~17 tok/s | reference |
| ref_b16: B12LL-b16, MTP k=3 | code ~27, math ~25 | reference |
| df4nat: exact routing, DFlash2 3 drafts (S = 4) | +10-20 % code, +5-15 % math vs ref_nat (tau4 3.6-3.8) | exact-routing lever if >= +10 % |
| df8b16: b16, DFlash2 7 drafts (S = 8, KC 10) | code 36-42, math 30-36 (+35-45 % vs ref_b16) | |
| df8b24 | within +-10 % of df8b16 | |
| df4b16 | +10-20 % vs ref_b16 on code | |
| df8b16, bf16 drafter | acceptance within 3 % of 8-bit | |
- Footprint <= 12.0 GB with the 8-bit drafter (prediction 11.4-11.8 GB without the MTP head).
- KILL DFlash2 at 12 GB if df8b16 and df8b24 are both < 1.2x ref_b16 on code AND on math.

## DF1 RESULT (19:47-19:55; ~/Fit12/data/df1/*.txt; 160 tokens per item; tok/s of the decode loop)
| arm | tfc0 | tfc1 | tfc2 | tfg0 | tfg1 | tfg2 | GSM geomean | vs ref_b16 (GSM) |
|---|---|---|---|---|---|---|---|---|
| ref_nat (exact B12LL, MTP k=3) | 17.8 | 18.6 | 17.5 | 17.5 | 18.4 | 18.9 | 18.3 | 0.70x |
| ref_b16 (MTP k=3) | 29.7 | 31.5 | 28.7 | 22.4 | 27.8 | 28.8 | 26.2 | 1.00x |
| df4nat (exact routing, 4 rows) | 18.1 | 18.2 | 17.5 | 17.9 | 18.6 | 19.1 | 18.5 | 0.71x |
| **df8b16 (8 rows, 8-bit drafter)** | 52.1 | 48.2 | 47.1 | 24.1 | 37.3 | 40.6 | **33.2** | **1.27x** |
| df8b24 | 35.3 | 38.4 | 38.3 | 20.7 | 28.9 | 32.4 | 26.9 | 1.03x |
| df4b16 | 30.6 | 29.1 | 28.3 | 20.6 | 26.3 | 27.1 | 24.5 | 0.94x |
| df8b16, bf16 drafter | 51.7 | 47.5 | 46.7 | 23.2 | 38.6 | 40.4 | 32.9 | 1.26x |
- CODE ITEMS ARE COPY-DOMINATED: the first 160 tokens of these HumanEval+ answers re-type the prompt's signature and docstring
  (decoded starts in df1/ref_nat.txt), so drafters hit 7/7 (8.00 tok/pass) and MTP 3.9-4.0 of 4: code ratios (1.6x) are not
  representative. DF2 re-measures on new-code prompts at 256 tokens.
- GSM8K (real generation): DFlash2 + b16 at 8 rows = 1.27x B12LL-b16 MTP and 1.82x exact B12LL; tok/pass 3.3-6.4 vs MTP 1.9-3.6.
  At 4 rows DFlash2 = MTP (df4nat +1 %, df4b16 -6 %): the gain is all in the width. b24 at 8 rows reads too much (verify
  146-217 ms vs 130-152).
- Correctness: no refused commits; df8b16 outputs = exact B12LL's on 4 of 6 items (code x3, tfg2) at 160 tokens; df4nat's GSM
  outputs differ from ref_nat's (my loop prefills in 512-token chunks, fit_b12 uses the decoupled prefill: not bitwise).
- Footprint: 8-bit drafter 11.21-11.55 GB (prediction 11.4-11.8: met); bf16 11.67-12.01 GB. 8-bit vs bf16 tok/pass 5.85 vs 6.23
  (texts differ under the budget); speed equal -> 8-bit.
- Pre-registered kill (df8b16 and df8b24 < 1.2x ref_b16 on code AND math): NOT met; math 1.27x, code pending DF2.
- Quality caveat: b16 shared by 8 rows distorts routing more than b16 over 4 rows (FM1's verdict was 4-row passes).

## DF2 pre-registration (20:05): new-code prompts (code = BigMoE prompt_code, lru, csv) + tfg1 / tfg2, 256 tokens, outputs saved
| arm | prediction | line |
|---|---|---|
| df8b16 vs ref_b16 on new code | 1.3-1.5x (tok/pass ~5 vs ~3.2 at ~150 vs ~115 ms/pass) | DFlash2 code lead if >= 1.2x |
| df8b16 vs ref_b16 on GSM (tfg1/2) | 1.25-1.35x (as DF1) | |
| df8b20 | between df8b16 and df8b24: 1.1-1.25x | |
| token agreement with exact B12LL (first divergence) | df8b16 ~ ref_b16 (same order of distance) | if df8b16 diverges much earlier, its quality needs its own verdict |

## DF2 RESULT (20:41-20:46; ~/Fit12/data/df2/; 256 tokens; outputs in df2/<arm>_<prompt>.json)
| arm | code | lru | csv | tfg1 | tfg2 | new-code geomean | GSM geomean | all |
|---|---|---|---|---|---|---|---|---|
| ref_nat (exact B12LL, MTP 3:3:0) | 19.4 | 18.1 | 21.0 | 18.6 | 18.6 | 19.5 | 18.6 | 19.1 |
| ref_b16 (MTP 3:3:0) | 27.1 | 27.9 | 27.6 | 28.2 | 27.4 | 27.5 | 27.8 | 27.6 |
| df8b16 (DFlash2 8-bit, 8 rows) | 25.2 | 32.0 | 38.3 | 38.1 | 30.7 | 31.4 (1.14x) | 34.2 (1.23x) | 32.5 (1.18x) |
| df8b20 | 25.3 | 33.0 | 38.6 | 31.1 | 28.2 | 31.8 (1.16x) | 29.6 (1.07x) | 30.9 (1.12x) |
- tok/pass: DFlash2 at 8 rows 3.8-5.8 (free-form code 3.8-5.5) vs MTP at 4 rows 3.2-3.6; 8-row verify 118-157 ms vs 108-121 ms
  (+20-25 %: b16 over 8 rows still adds ~20 % waited reads, and KC 10 costs ~15 % more misses than KC 20).
- Against the pre-registration: new-code 1.14x vs predicted 1.3-1.5x: MISSED (below the 1.2x code-lead line); GSM 1.23x vs
  1.25-1.35x (just below the range); df8b20 1.12x (inside 1.1-1.25). Every non-exact arm differs from exact B12LL at 256 tokens
  on every prompt (first-divergence positions: CPU check pending, the Mac was locked).
- VERDICT: DFlash2 at 12 GB = +14 % (code) / +23 % (math) over B12LL-b16 with MTP, needs b16 shared by 8 rows (unjudged quality)
  and a drafter (+0.35 GB). Not worth a 12 GB engine build on its own: PARKED; the prototype stays for reuse if the main track
  integrates DFlash2. Cheap upgrades if revisited: KC 20 at 8 rows (a 160-id gate row, ext change), bf16 drafter (12.0 GB).

## P1 pre-registration (20:58): B12LL prefill inside 12 GB (~2.6K and ~8K-token prompts) + a cache probe
| arm | prediction | line |
|---|---|---|
| decoupled 512 / 4096 (shipping) | 2.6K ~280 tok/s, ~11.3-11.6 GB; 8K ~270 tok/s, ~11.8 GB | reference |
| decoupled 512 / 8192 | 8K +30-60 % (half the expert reads), +0.2-0.5 GB; 2.6K = 4096 | candidate if >= +25 % at <= 12.0 GB |
| streamed ring 4 (TrainLab, exact) | 340-430 tok/s, +0.4-0.6 GB | same line |
| streamed ring 4 + 8192 | 8K 450-550 tok/s | same line |
| streamed ring 6 | +5-10 % over ring 4, +0.3 GB (may break 12 GB) | same line |
| all modes | the same 32 decode tokens (s_moe and streaming are exact per Breakthrough / TrainLab) | a differing sha = not exact |
| KC 23 (MAXK 92): natural / b16, code + prose | waited -3..-8 % vs KC 20 (more near-miss residents protected) | keep if >= +3 % tok/s |

## P1 RESULT (20:54-20:59; ~/Fit12/data/p1/): decode shas identical across all prefill modes (exact)
| mode | 2.6K tok/s | 2.6K peak footprint | 8K tok/s | 8K peak footprint |
|---|---|---|---|---|
| decoupled 512 / 4096 (shipping) | 257 | 11.35 GB | 275 | 11.92 GB |
| decoupled 512 / 8192 | 267 | 11.35 | 346 (+26 %) | 12.73 |
| streamed ring 4 | **347 (+35 %)** | **11.83** | 335 (+22 %) | 12.41 |
| streamed ring 4 + 8192 | 348 | 11.84 | 377 (+37 %) | 13.40 |
| streamed ring 6 | 379 (+47 %) | 12.20 | 368 (+34 %) | 12.79 |
- Inside 12 GB only ring 4 on short prompts qualifies (+35 % at 2.6K, 11.83 GB; Breakthrough's opt-in). At 8K every faster mode
  breaks 12 GB, and the shipping mode is already at 11.92 GB: long-context memory, not prefill code, is the binding limit.
- KC 23 vs KC 20: waited misses equal (578 vs 580, 341 vs 341), same shas: KILLED (protection saturates by rank 20).
- Noise: identical reads gave 18.7 vs 21.2 tok/s (code natural) across runs: ~10 % SSD-throughput noise in this io-bound regime.
- DF2 first divergence from exact B12LL (tokens): ref_b16 24 / 55 / 97 / 1 / 210 (code / csv / lru / tfg1 / tfg2); df8b16 0 / 35 /
  21 / 1 / 218; df8b20 0 / 55 / 97 / 17 / 213 (df arms prefill in 512-token chunks, so token 0 can differ on code).

## J1 pre-registration (21:10): why JIT prefetch loses at C_m 51 (diagnostic, code + prose)
| arm | prediction | what it decides |
|---|---|---|
| JIT 12, natural (prefetch detail per pass) | of ~219 used prefetches, >= 40 % late (read still in flight at demand), victims >= 50 / pass | late-dominated: a staging ring will not help; victim-dominated: a small ring outside the pools could |
| JIT 12 + b16 + budget-aware JIT | late share similar; victims lower (pool headroom) | same |
| C_m 59 (C_l 20), no JIT: 8 more resident slots (+0.4 GB) | waited -5..-9 %, tok/s +3-6 % | the value of slots vs prefetch |

## J1 RESULT (21:44-21:46; ~/Fit12/data/j1/): JIT prefetch at 12 GB is victim- and waste-bound, not late
| arm | tok/s code / prose | waited / pass | prefetch per pass (code): issued, on time, late, victims | SSD MB / pass |
|---|---|---|---|---|
| JIT 12, natural | 20.8 / 13.0 (base 21.1 / 12.3) | 648 / 684 (base 580 / 623) | 390, 174, 45, **258** | 1659 (base 1138) |
| JIT 12 + b16 + budget JIT | 27.4 / 18.6 (b16 27.8 / 18.9) | 368 / 390 (b16 341 / 356) | 234, 128, 13, 140 | 884 (b16 622) |
| C_m 59 / C_l 20, no JIT (+0.8 GB) | 22.6 / 13.9 (+7 % / ~+10 %) | 531 / 570 (-8 %) | - | 1034 |
- Prefetch timing is fine (80 % of used prefetches on time; pf wait ~9 ms/pass). It loses because every prefetch takes a
  victim (258 / pass natural) whose loss adds ~80 demand misses, and 44 % of prefetches are wasted reads that compete for SSD
  bandwidth in a bandwidth-bound regime. A victim-free staging ring would remove the first cost, not the second: expected
  +5-10 % (exact). Not built: low value for the engine change.
- 8 more resident slots per layer = -8 % misses = +7 % tok/s at the same tokens, but 12.17 GB: worth doing only if ~0.8 GB of
  B12LL's non-MLX footprint (1.0 GB at 2.6K, 1.4 GB at 8K: footprint minus MLX peak) can be reclaimed. The n-gram hot cache
  is NOT it (decode-only LRU, small at these lengths). Source still unknown (needs a live vmmap).

## M1 pre-registration (21:55): B12LL memory breakdown (footprint + vmmap after load / 2.6K prefill / 64 decode tokens)
- Prediction: the ~1.0 GB of non-MLX footprint is MALLOC (Python, numpy, tokenizer, n-gram row copies) ~0.4-0.6 GB plus IOKit /
  Metal driver memory ~0.3-0.5 GB, plus small ext buffers. Line: if >= 0.5 GB sits in one reclaimable bucket (a cache, a
  staging buffer, a duplicate copy), reclaim it and spend it on C_m +5-8 (J1: +8 slots = +7 % tok/s, same tokens).

## M1 RESULT (22:01-22:02; ~/Fit12/data/m1/): B12LL's 11.35 GB "peak footprint" is a LOAD transient
- After load: footprint now 11.03, lifetime peak 11.35, MLX peak 10.80 / active 9.68 GB. After the 2.6K prefill: now 10.39 GB.
  After 64 decode tokens: now 10.67 GB, of which IOAccelerator (MLX GPU buffers) 9.6 GB, untagged 121 MB, MALLOC 191 MB.
- So B12LL RUNS at ~10.4-10.7 GB at short context (~1.3 GB below 12). The load peak comes from v3f/hc8.install, which
  evaluates every 8-bit mixer at once while all bf16 mixers (1.27 GB) are alive.

## M2 pre-registration (22:10): layer-wise hc8 install (~/Fit12/scripts/fit_hc8lw.py, bitwise the same weights) + C_m 59
| arm | prediction | line |
|---|---|---|
| C_m 51, layer-wise hc8 | after-load lifetime peak 10.2-10.7 GB (from 11.35); same shas as B12LL | load trim = pure win if >= 0.5 GB and same shas |
| C_m 59 / C_l 20, layer-wise | after-load peak 11.0-11.5; 2.6K prompt peak <= 12.0; 8K ~12.5 (over) | short-context option if <= 12.0 at 2.6K |
| C_m 59 speed (natural / b16) | +5-8 % / +4-7 % vs C_m 51 (J1: +7 %) | |

## M2 RESULT (22:29-22:32; ~/Fit12/data/m2/): load trim KILLED; C_m 59 exact +6-7 % but over 12 GB
| arm | after-load lifetime peak (MLX peak during load) | code / prose tok/s | 2.6K peak (now) | 8K peak (now) | shas |
|---|---|---|---|---|---|
| C_m 51, layer-wise hc8 | 11.25 GB (10.91) vs 11.35 (10.80) | 21.3 / 13.0 | 11.28 (10.74) | 11.92 (11.28) | = B12LL |
| C_m 59 / C_l 20, layer-wise | 12.08 (11.74) | 22.6 / 13.9 (+6 / +7 %) | 12.12 (11.57) | 12.75 (12.10) | = B12LL |
| C_m 59 + b16 | 12.08 | 28.5 / 20.1 (b16 at 51: 27.8 / 18.9) | - | - | = b16's |
- Prediction (load peak 10.2-10.7) MISSED: the layer-wise install moves the load peak by -0.1 GB only; MLX still peaks ~1.2 GB
  above active during load, so the bf16 mixers are already materialised by the loader before hc8 runs (or the transient is
  elsewhere in load_streamed). M1's "hc8.install is the cause" was wrong; the transient is real but its source is unpinned.
- B12LL runs at 10.7-11.3 GB (2.6K-8K, "now") but peaks at 11.25-11.92 GB (load / long prefill). C_m 59 (+0.82 GB) breaks
  12 GB at every point that matters. Not adoptable at 12 GB without trimming both the load and the prefill peaks.

## Eval suite verdict B12LL-b16 run 1 (22:33, report_b12llb16*.txt): FAIL 3/5 vs the 4-bit ref, PASS 5/5 vs B12LL
- vs ref: math 88 (+2), IFEval 83 vs 90 (-7.0 [-11.2, -2.8]) FAIL, HumanEval+ 91.7 = ref, truncations 15 / 2 vs 13 / 2 FAIL.
- vs B12LL: IFEval -4.0 [-8.6, +0.6] (n.s.), all 5 pass. As feared from FM1: b16 costs ~4 IFEval points, and B12LL is already
  -3 vs ref. Suite next: b12llb16.r2, then the b20 fallback.

## U1 pre-registration (22:45): uncertainty-gated budget (NewIdeas' UGB) for B12LL, speed side
Gate per pass: budget b16 on iff the MTP head's lowest draft-token probability in the pass >= theta; else natural routing
(B12LL exact). Rationale: the budget hurt IFEval (uncertain text); NewIdeas' strict-KLD analysis passes b22 with a confidence
gate. Same lock, 4 prompts (code, prose, math, write), 128 tokens.
| arm | prediction (budget-on share; tok/s vs natural) |
|---|---|
| natural / b16 / b20 | references (b16 ~+46 %, b20 +25-35 %) |
| b16, gate 0.8 | on: code 70-85 %, math 60-80 %, prose 30-50 %, write 20-40 %; geomean +25-35 % |
| b16, gate 0.5 | on: 10-20 points higher; geomean +32-42 % |
Line: a gate that keeps >= 70 % of b16's geomean gain and keeps the budget off on >= half of the write / prose passes is the
better verdict candidate than plain b20 (it protects the uncertain text by construction). Quality = Eval suite.

## U1 RESULT (23:28-23:33; ~/Fit12/data/u1/; tok/s, budget-on share of drafted passes)
| arm | code | prose | math | write | geomean | gain kept vs b16 |
|---|---|---|---|---|---|---|
| natural B12LL | 21.3 | 13.0 | 16.3 | 14.5 | 16.0 | - |
| b16 | 27.8 | 18.9 | 25.1 | 19.5 | 22.5 (+41 %) | 100 % |
| b20 | 22.9 | 17.0 | 20.7 | 19.1 | 19.8 (+24 %) | 58 % |
| b16, gate 0.5 | 25.9 (78 %) | 15.7 (53 %) | 22.7 (88 %) | 16.1 (39 %) | 19.6 (+23 %) | 55 % |
| b16, gate 0.8 | 25.0 (45 %) | 14.0 (22 %) | 20.4 (62 %) | 15.8 (20 %) | 18.3 (+15 %) | 35 % |
- Pre-registration: gate 0.8 coverage on code 45 % (predicted 70-85 %), geomean +15 % (predicted +25-35 %): MISSED; gate 0.5 +23 %
  (predicted +32-42 %): MISSED. Line (>= 70 % of b16's gain, budget off on >= half of prose / write passes): not met by either.
- The MTP head is less confident than the target (TrainLab's strict gate used the target's own top-1); at 12 GB a gated pass
  saves reads only when on, so speed tracks coverage.
- Speed-equal pair for a verdict: b20 (milder budget everywhere) vs b16-gate-0.5 (b16 on 78-88 % of code / math passes, off on
  47-61 % of prose / write passes = exact B12LL routing there). Hook: load_b12ll_budget("b16", ugb=0.5).

## Later candidates (pre-register before each run)
- LFU eviction (BigMoE EV1 port into a v3f copy, ~/Fit12/v3g): exact for B12LL; +5..+13 %; kill < +3 %.
- DFlash2 (code / math): exact for B12LL (greedy verify); needs 8-row passes = more misses per pass at 12 GB unless
  budgeted. Pre-register after F1's width and budget numbers.
- Prefill inside 12 GB (decoupled 280 tok/s; streamed ring 341-474): after decode.

## C1 (Sat 10-24 02:15, orchestrator 23:47 / 01:44): what breaks the 12 GB SSD-wait ceiling
Physics (B12LL + MTP, code): a pass waits on ~12 misses per layer (~26 MB, two gated calls of ~1.26 ms each), ~1.1-1.5 GB per
pass. Peak SSD rate is 14.6 GB/s (MoonShots ssd_qd: whole reads 14.0 GB/s already at QD 4), so the floor is 75-100 ms of
pure transfer per pass vs 107-143 ms of stalls today. Compute ~43 ms + draft 8-11 ms are serial with the reads.
Already ruled out by other sessions (do not repeat):
- Prefetch at equal RAM, every form: spare slots out of C, victim JIT, corrected d=1 predictor. Breakthrough SCOREBOARD
  455-460 and 609-619; my J1.
- Belady eviction is within 5 % of LRU (SCOREBOARD 234). LFU gave +0.5 % (mine).
- Multi-row phase B (verify rows add cold ranks half a layer late). Verify passes got 16 % faster, but acceptance fell and
  tok/s dropped 16-17 % (MoonShots HANDOVER, CLOSED 22:17 Thu).
- The QD is already at the SSD's rate.
So the only lever left is BYTES PER PASS: fewer bytes per expert, or more residents per GB. Options:
| option | kind | first test | prediction | kill line |
|---|---|---|---|---|
| O1 lossless entropy coding of the streamed MSB-3 part | exact | Q1 (CPU): order-0 entropy of m codes, s3/b3 byte-wise, zlib/lzma | m 2.6-2.85 bits of 3; s3/b3 20-30 % smaller; MSB part 8-14 % smaller in total | < 12 % -> kill. Below that the GPU decode cost and a 45 GB store are not worth <= +8 % |
| O2 rotated re-quantization: randomized block Hadamard on K, then RTN / Lloyd-Max levels | non-exact (new store) | Q1: rel MSE vs W4 (4-bit, = ref) for each format | MSB-3 3-6 %; RHT RTN-3 g128 (3.25 b) and RHT Lloyd-Max-3 per row (3.0 b) <= MSB-3; every 2.5-bit format >= 2x MSB-3 | no format at <= 3.0 bits/weight within 1.0x MSB-3's error -> kill (that would be <= 7 % fewer bytes) |
| O3 shared base: mean expert per layer resident, residuals streamed at fewer bits | non-exact | Q1: leave-one-out mean expert, share of each expert's energy it explains | < 5 % (independent experts) | < 15 % -> kill |
Q1 sample: layers 2, 12, 24, 36, 46, 8 random experts each (seed 0), all of gate / up / down (40 records, 117 MB read).
Bit layout checks before any number counts:
- rebuild q = 2m + l: every group must hold 0 and 15 (min-max 4-bit);
- b3 == bf16(b + s * mean(l)) on >= 99.9 % of groups.
Script ~/Fit12/scripts/fit_q.py, output ~/Fit12/data/c1/q1.txt.

## C1 / Q1 RESULT (02:20, CPU, 74 s; ~/Fit12/data/c1/q1.txt): all three byte options are dead at equal quality
Layout checks pass over 3,072,000 groups: q min 0 in 100 %, q max 15 in 85 % (MLX's edge-exact scale), b3 rebuild 100 %.
- O1 lossless: H(m) 2.72 bits of 3 (static Huffman 2.76); s3 / b3 8.8-9.1 bits of 16 (sign + exponent nearly constant);
  LSB plane incompressible (H(l | m) 0.987). MSB part 14.3 % smaller at the entropy bound, 13.1 % with static Huffman.
  That passes the 12 % line, but the decode kills it: ~480 MSB misses / pass x 4.9 M codes = 2.35 G symbols per pass. The
  saved bytes are worth 12-17 ms / pass, so the GPU decode must run at >= 150-200 G symbols/s just to break even; table
  Huffman / rANS on this GPU is plausibly 50-250. Even at break-even + 2x the net is <= +6 %, and it needs a new 45 GB store
  (disk: 126 GB free, floor 110). KILLED on decode economics. Exact scale-only repacking (sign + exponent index + 7-bit
  mantissa = 11 of 16 bits) is ~4.5 % of the bytes: not worth a store rewrite.
| format (rel. squared error vs W4) | bits / weight | error | vs MSB-3 |
|---|---|---|---|
| MSB-3 (today) | 3.5 | 0.0252 | 1.00x |
| RTN-3 g64 fresh / RHT + RTN-3 g64 | 3.5 | 0.0359 / 0.0369 | 1.43x / 1.46x |
| LM-3 g64 symmetric / RHT + LM-3 g128 | 3.25 / 3.125 | 0.0318 / 0.0318 | 1.26x |
| RHT + LM-3 per row | 3.0 | 0.0343 | 1.36x |
| MSB-2 truncation / RHT + LM-2 g64 | 2.5 / 2.25 | 0.1275 / 0.1113 | 5.1x / 4.4x |
- O2 KILLED: no format at fewer bits comes within 1.26x of MSB-3. MSB-3 shares W4's grid, so dropping the LSB costs
  s^2 / 4 per weight, against s^2 / 3 for any fresh rounding (0.70x measured, 0.75x predicted). The Hadamard rotation does
  not help: these experts are already Gaussian-like (RTN g64 1.43x without it, 1.46x with it).
- O3 KILLED: the cosine with the leave-one-out mean is +0.003 / +0.000 / -0.000 (gate / up / down); experts share nothing.
  This also explains why frugal substitution hurts: a resident stand-in is an unrelated expert.
Conclusion: at 12 GB the exact ceiling is physical. Bytes per pass = misses x 2.15 MB. MSB-3 is the most compact 3.5-bit
form of W4, and cache policy, prefetch and late-add are exhausted. What remains exact:
- reclaimed memory spent on slots (+8 slots = +7 %; Breakthrough's transient job);
- less compute per pass (43 ms; the 350 line's kernels, MoonShots / Breakthrough).
Together that is ~+15 % (code ~24-26 tok/s). Anything beyond that changes routing (budget, UGB-0.5) and is judged by the
suite.

## S1 pre-registration (02:35): robust speed table for the verdict candidate, B12LL vs B12LL-UGB-0.5
Launch only if the suite passes UGB-0.5 (report_b12llugb*.txt, ~02:45); otherwise park it. Runner ~/Fit12/scripts/s1.sh,
6 prompts (code, prose, math, write, lru, csv) x 256 tokens, MTP 3:3:0, async + PF_GROUP 64. Two fresh processes per
arm in ABAB order, so the SSD's ~10 % run-to-run noise shows up as a spread.
| arm | prediction (tok/s geomean, 6 prompts) |
|---|---|
| B12LL natural | 16-19 (U1 four-prompt 16.0; lru / csv are code-like) |
| UGB-0.5 | +18-25 % over natural; per-prompt gain tracks the budget-on share (code / math high, prose / write low) |
Line: a run pair whose spread is > 10 % on any prompt -> report the range, not a single number.

## X1 pre-registration (02:40; NewIdeas QUEUE L1-02 + L1-03): DFlash2's REAL 8-row pass with exact verify at 24-34 GB
Same B12LL frame at higher residency: load_b12(F12_FRAC), C_m 192 / C_l 77 (frac 0.375, ~26 GB) and C_m 256 / C_l 102
(frac 0.5, ~34 GB). The verify is natural and unbudgeted (FL_MR: every row waits for its top-10 = v3 natural routing).
MLX's default dense kernels are used, which are NOT batch-invariant at 6+ rows (MoonShots: DUPROW chunks cost 3.5-6.4x);
the strictly exact wide pass is L1-01's kernel. DFlash2 is 8-bit, bound to the target, chain of 7 drafts. The counting is
the official one (k accepted + 1 = up to 8 per pass). 256 tokens on 7 items:
- free-form code: code, lru, csv;
- HumanEval+: tfc0, tfc1 (copy-heavy starts, like Leap's replay set);
- GSM8K: tfg0, tfg1.
Runner ~/Fit12/scripts/x1.sh, one lock. A step is skipped if active + wired + its size > 56 GB; the run is killed if
swap grows > 2 GB.
| arm | prediction |
|---|---|
| x_mtp192: MTP k=3 at C_m 192 (reference) | pass 70-90 ms, code ~35-45 tok/s, 3.2-3.6 tok/pass |
| x_df192: DFlash2 8 rows at C_m 192 | pass incl. draft 90-130 ms (verify compute 50-65 ms + stalls); tok/pass: free-form code 3.8-5.5, HumanEval 6.3-7.5, GSM 5.0-6.0; tok/s 1.1-1.4x MTP |
| x_df256: same at C_m 256 | pass 70-100 ms |
Lines:
- L1-02: an 8-row pass of >= 60 ms (incl. draft) at the best residency means code-350 needs > 3x from kernels. Predicted:
  the line is hit.
- L1-03: tokens per pass < 0.9x Leap's replay (code 6.93 on HumanEval+, GSM 5.74) means the replay overstated. Predicted:
  HumanEval / GSM within 0.9x; free-form code 0.55-0.8x, so the replay's code number holds only on copy-heavy code.

## T1 pre-registration (03:05; QUEUE L2-02 + L2-03, handed over by Breakthrough 02:15): exact trims so C_m 59 fits 12 GB
Both trims are in ~/Fit12/scripts/fit_trim.py and hook ExpertStreamer.install, so they act before the loader's
mx.eval(model.parameters()):
- T1 hc_pre: hc8's 8-bit swap on the still-lazy bf16 mixers. Today load_b12 runs it after the full eval, so the bf16
  copies (~1.25 GB) coexist with the 8-bit ones; M1's MLX load peak is 10.80 vs 9.68 active.
- T2 embed: memmap row gather + the same mx.dequantize; the 358 MB table is never read.
The fit_pf.py sampler records the peak footprint per phase (load, then prefill and decode for p2k and p8k). Same lock as X1,
after it (x1.sh).
| arm | prediction (peak footprint load / p2k / p8k, GB) | sha (decode 32) |
|---|---|---|
| t_embcheck | bitwise equal on 4 id batches | - |
| t_base59 (C_m 59, no trims) | 12.08 / 12.12 / 12.75 (M2) | 82995cdb4e / 0b0d91cd43 (M2) |
| t_hc59 | load -0.6..-1.1 (11.0-11.5) / p2k ~12.1 / p8k ~12.7 | same |
| t_both59 | ~10.7-11.1 / ~11.7-11.8 / ~12.35-12.4 | same |
| t_both59s2k (s_moe 2048) | p8k a further -0.15..-0.35; prefill tok/s -10..-25 % | same |
| t_both51 (C_m 51 + trims) | p8k ~11.5 | same |
| t_spd59 vs t_spd51 (code / prose, 128 tokens) | +5-8 % tok/s | identical texts |
Lines:
- L2-02: C_m 59 qualifies if every phase peak is <= 11.95 GB at 2.6K and 8K. Otherwise report the remaining gap and
  what is left at 8K.
- L2-03: a gain < +2 % or a differing sha -> kill.

## B1 pre-registration (Sat 10-24 04:00, before any run): one read burst per layer (answers the orchestrator's C1 gap question)
Cause found in the engine (read-only reading of ~/Breakthrough/v3f/ext/bigmoe_ext.cpp loader_main + bt_b12 block): the two
gated calls per layer are NOT gate/up vs down (one record read already holds all three); they are the MSB-3 base row (L)
and the LSB sidecar row (49 + L). The single loader thread serves rows one at a time (ensure -> dispatch_apply reads ->
release), and MLX evaluates spin_m before post_l, so the sidecar's reads start only after the base row is released:
each layer pays the per-call cost (~0.25-0.35 ms: GCD wake, SSD first byte, burst tail + bounce memcpy, release pickup)
twice. ~85 stalled calls x ~0.34 ms = ~29 ms/pass = the 30-45 ms gap between 107-143 ms of stalls and pure transfer.
B1 (~/Fit12/v3j = v3g + BIGMOE_JOINT): gated.py makes each layer's first phase-0 wait depend on every row posted with it
(mx.depends, so post_l runs before spin_m); the loader, on seeing base row L, waits <= 30 us for sidecar row 49 + L of
the same pass and starts its missing waited ids reading in the background (slots chosen exactly as make_room would),
then serves the base row as before; the sidecar row is then served as before (waits for those reads, promotes them).
Same evictions, same ids per slot -> bitwise the same tokens. BIGMOE_CALLLOG logs every served row (fit_clog.py).
Runner ~/Fit12/scripts/b1.sh: b1_ref (Breakthrough v3f) + ABAB off/on on v3j, code/prose/math/write k=3, 128 tokens.
Predictions: all 20 shas = U1 u_nat (259e67193e, b5cc227c26, 6e3af723f9, 67746aae84). Off: base ~47 stalled/pass at
1.5-2.0 ms, sidecar 35-45 stalled/pass at 0.35-0.7 ms, sidecar seen < 20 us after its base release. On: >= 90 % paired;
sidecar stalled calls 0.02-0.15 ms; base calls +0.1-0.25 ms; stall -10..-20 ms/pass; tok/s +5..+12 % geomean.
Lines: any sha differs, or < +3 % geomean -> kill. >= +5 % and all equal -> hand to Breakthrough (exact, no RAM).

## X1 RESULT (04:00; ~/Fit12/data/x1/x_*.txt)
| arm | measured | prediction |
|---|---|---|
| x_mtp192 MTP k=3 | 33.0 tok/s (24.4-36.8; tfg0 1.45 tok/pass), pass 94-111 ms incl. draft, stall 46-62 ms | 70-90 ms, 35-45 tok/s: MISS (slower) |
| x_df192 DFlash2 8 rows | 32.6 tok/s, 5.08 tok/pass, pass 144-172 ms incl. draft (stall 64-95, compute ~70) | 90-130 ms, 1.1-1.4x MTP: MISS (~1.0x) |
| x_df256 | skipped by the memory gate (20 GB in use + 38 > 56) | - |
| x_or192f oracle T8 / T16 | T8 173 ms (frugal); T16 1.64-1.78 s on v3's prefill path (16 rows > 9 = 92 ids / top-10) | - |
L1-02: line hit (predicted). L1-03: HumanEval 5.33/6.24 = 0.84x of 6.93, GSM 3.76/5.33 = 0.79x of 5.74 (MISS: predicted
within 0.9x); free-form code 4.92-5.45 = 0.71-0.79x (hit). T16/T8 estimate 1.5-1.65 from the 4 -> 8 step. Oracle drafts
diverged from the DFlash2 run's tokens after 22-231 tokens (8-row kernels are not batch-invariant). x1.sh's rc column is
always 0 (bug: rc took the test's status, fixed in b1.sh); outputs were checked directly.

## T1 RESULT (04:00; ~/Fit12/data/x1/t_*.txt; shas 82995cdb4e / 0b0d91cd43 in every arm)
| arm (C_m 59 unless noted) | load | p2k prefill | p8k prefill (lifetime) |
|---|---|---|---|
| t_base59 | 12.02 (12.17) | 12.07 | 12.71 (12.76) |
| t_hc59 (hc8 on lazy bf16) | 10.84 | 12.07 | 12.67 (12.75) |
| t_both59 (+ streamed embed) | 10.48 | 11.71 | 12.35 (12.41) |
| t_both59s2k (+ s_moe 2048) | 10.48 | 11.59 | 11.95 (11.99); prefill 211 vs 281 tok/s |
| t_both51 (C_m 51) | 9.65 | 10.90 | 11.51 (11.59) |
Predictions hit (t_hc59 beat -0.6..-1.1 with -1.33). C_m 59 qualifies only up to ~2.6K prompts (line 11.95). Embed check
BITWISE EQUAL. t_snap59: the 8K peak is MLX buffers (10 GB IOAccelerator + 467 MB owned-unmapped), malloc+VM ~0.3 GB.
t_spd51 = U1 u_nat exactly. t_spd59 (fit_b12, C_m 59 + both) DIED in warm-up: 47 spin-gate timeouts -> isolation in b1.sh.

## B1 RESULT (04:45; ~/Fit12/data/b1): exact, but no gain -> v1 KILLED (line < +3 %)
All 20 shas = U1 u_nat (bitwise, JOINT on and off, v3f ref). tok/s off a/b vs on a/b: code 21.3/21.2 vs 21.4/21.4, prose
13.0/13.0 vs 13.1/13.1, math 16.8/16.8 vs 16.9/16.9, write 14.5/14.5 vs 14.6/14.5 (+0.5 %); stall -0.8 ms/pass.
Call log (code+prose+math+write, 217 passes): base 46.8 stalled calls/pass x 2.12 ms (11.2 misses = 24 MB, 11.4 GB/s)
= 99 ms; sidecar 40.7 x 0.38 ms (3.2 misses) = 15.5 ms; MTP head 2.8 x 1.6 ms = 4.6 ms. Off: the sidecar row is seen
> 50 us after its base release in 87 % of layers (posted after spin_m). On: paired 100 %, partner wait 0, but pf_used 0:
MLX ran post_l BEFORE post_m (mx.depends order), so the loader's scan met the sidecar row first and served it the old
way; the base row then found nothing left to read. The gap, corrected: per-call fixed cost ~0.47 ms per base call (24 MB
at 11.4 vs 14.6 GB/s) + ~0.21 ms per sidecar call = ~30 ms/pass; only the sidecar part (~8 ms) is mergeable.
## B2 pre-registration (04:50): B1 v2 + fixed fit_trim
v2: post_l's input depends on post_m (base row posted first) and the loader skips a sidecar row while its base row of the
same pass is posted and unserved. Predictions: shas = u_nat everywhere; on: pf_used ~3/layer-call, sidecar stalled call
<= 0.1 ms, base call +0.1-0.2 ms, stall -4..-9 ms/pass, tok/s +2..+5 % (line: < +2 % -> kill for good).
fit_trim fix (installs after load_b12's _env(); the early import left the ext's BIGMOE_DEFER at 0 while the gate
expected phase B -> 47 timeouts on the first 1-row pass): f59both / f59hc / f59emb / f51both run, shas = u_nat, tok/s
f59* within 2 % of i59plain (22.5 / 13.8); f59pf memory = t_both59 (10.48 / 11.71 / 12.35).
- X1 CORRECTION (05:00, Leap 04:44): fit_dfl2 ran to 256 tokens with no end-of-turn stop; tfg0 ends at token 101 and tfc0
  at 240 (verified in the saved outputs), so post-turn tokens diluted tok/pass (MTP's tfg0 1.45 too). On spans equal to
  the 4-bit reference, real/replay = 0.93-0.98 (tfc1 6.24 vs 6.40, tfg1 5.33 vs 5.75): L1-03's "replay overstates ~20%"
  is WRONG; the 0.71-0.79x on free-form code stands (content type). DFlash2 vs MTP on the 5 clean items: 1.04, 0.99, 1.01,
  1.10, 0.99 (geomean ~1.03): the verdict "= MTP at C192" stands. fit_dfl2 now stops at 248046/248044 (F12_STOP=1).
- B2 addendum (05:05): call-time fit on B1's logs: base call = 0.255 ms + 0.166 ms per miss (12.9 GB/s marginal),
  sidecar 0.152 + 0.072 / miss, MTP head 0.351 + 0.244 / miss. Fixed cost ~19 ms/pass + bandwidth shortfall ~10 ms.
  Each read goes through a per-thread bounce buffer, then 9 memcpys into the pool slots (~0.1-0.15 ms after the last
  read: a large part of the 0.255 ms). B2 adds BIGMOE_RECORDS_DIRECT=1 (the engine's preadv straight into the slots,
  same bytes; an old note says unaligned F_NOCACHE destinations are slow, and the s3/b3 slots are not 16 KB aligned):
  b2_dir_a/b vs b2_off_a/b, and b2_dirj (+ JOINT). Prediction: direct -0.05..-0.15 ms per base call, or slower if the
  old note holds; line: >= +2 % with shas equal -> keep.

## O1 pre-registration (06:30): out-of-order experts on B12LL's verify rows (exact)
B12 calls the fused 3c kernel after the loader spin, so on the ~47 stalled layers per pass the resident pairs' GLU and the
shared expert wait behind the SSD. B12LL's verify rows wait for all of ranks 0-9 (NF = 10), so the used set is fixed
before the spin, and the post kernel already returns the partial-hit row. fit_ooo.py (F12_OOO=1) patches bt_b12._install
in-process: resident pairs (ranks 0-1 need both record parts) + shared expert before the spin, the missed pairs after,
then tierglu.combine_phase (bitwise == tier_glu(..., scores=) for any hit mask: MoonShots, BigMoE OOO1 +3-8 % at 24 GB).
1-row calls (late parts) unchanged. Stock Breakthrough v3f. Arms (~/Fit12/data/o1): o_off_a, o_on_a, o_off_b, o_on_b (A4,
k=3, 128 tokens), o59on (C_m 59 + both trims + OoO, code + prose).
Prediction: all shas = u_nat; MoonShots' post-spin path at M=4 0.314 -> 0.06-0.12 ms/layer, x ~45 stalled layers = 9-12 ms
of a ~125-160 ms pass -> +4 to +8 % tok/s (A4 mean, on vs off). Kill: < +2 %, or any sha differs.

## B2 RESULT (06:35; ~/Fit12/data/b2, lock 06:19-06:29): B1 v2 KEEP (exact, +3-5 %), direct reads KILLED, trims fixed
All 36 decode shas = U1 u_nat (b2_* x 4 prompts, f* x 2); f59pf decode shas = T1's (82995cdb4e / 0b0d91cd43).
- B1 v2 (v3j, JOINT=1): tok/s on a/b code 22.4/22.3, prose 13.5/13.8, math 17.5/17.7, write 14.8/15.2. vs B1's off arms
  (v3j v1, JOINT 0; = U1): +5.2 / +5.0 / +4.8 / +3.4 % (mean +4.6 %, prediction +2..+5 %). Same-batch off a/b: code 21.2,
  prose 12.1/12.0 (7 % below B1 and U1 at equal misses and stalled calls, stall +10.7 ms/pass; math/write/code = B1),
  unexplained -> vs those +5.4 / +13 / +4.8 / +3.8 %; O1 repeats v2 beside stock-v3f off arms. Call log (on_a): paired
  100 %, sidecar service 0.426 -> 0.007 ms per stalled call (17.3 -> 0.3 ms/pass), base call 2.178 -> 2.326 ms (+6.9 ms/p):
  net -10.1 ms of loader time per pass.
- Direct reads (BIGMOE_RECORDS_DIRECT=1, preadv into the 9 pool slots): code 16.5, prose 10.0, math 13.3, write 11.2
  (-22 %); base call 2.756 ms, sidecar 0.825. With JOINT: 18.8 / 11.6 / 15.1 / 12.9. KILLED (the bounce read + memcpy
  is faster).
- fit_trim fix: f59both code 22.3 / prose 13.7 (i59plain 22.5 / 13.8: -0.9 / -0.7 %, within the 2 % line), f59hc 22.4 /
  13.7, f59emb 22.3 / 13.6, f51both 21.1 / 12.9 (U1 21.3 / 13.0); load peak f59both 10.48, f51both 9.66 GB.
- f59pf (C_m 59 + both trims) = T1 t_both59 within 0.02 GB: load 10.48, p2k prefill 11.73 sampled / 11.76 lifetime, p8k
  12.37 / 12.43 GB. C_m 59 is < 12 GB to ~2.6K-token prompts only.

## O1 RESULT (07:55; ~/Fit12/data/o1, lock 07:47-07:53): OoO exact but +0.6 % -> KILLED as built; cause found
All 32 decode shas = U1 u_nat (8 arms). OoO on vs off (stock v3f): code 21.3/21.4 vs 21.3/21.2, prose 13.1/13.1 vs 13.0/13.0,
math 16.9/15.5 vs 16.8/16.8 (on_b math: one slow item, stall +16 ms), write 14.6/14.6 vs 14.5/14.5: +0.5..+0.8 %, verify
time -0.5..-1.3 ms/pass (the shared expert's time only). B1 v2 same batch (o_v2 vs stock off): code 22.4 vs 21.25, prose
13.8 vs 13.0, math 17.6 vs 16.8, write 15.3 vs 14.5 = +5.4 / +6.2 / +4.8 / +5.5 % (mean +5.5 %): B2's off-arm prose dip
was that arm, v2 holds. Stack o59all (C_m 59 + trims + v2 + OoO): code 23.7, prose 14.5 (+11.3 / +11.5 % vs U1).
CAUSE of the OoO null: the streamed MTP head's block has first = its own layer, so each draft call runs new_pass(): seq
advances 4 per verify cycle (1 trunk + 3 head calls), while the loader publishes each layer's residency table for seq+1
only. A trunk layer's post never finds its table (tag mismatch) -> the partial-hit row is all -1 (phase 1 empty) and B12's
fast path never fires on trunk layers under MTP.

## O2 pre-registration (08:05): residency tables for seq+1..seq+4 (v3j ext, BIGMOE_PUB_AHEAD=4) + OoO
Ext change (~/Fit12/v3j/ext, backup bigmoe_ext.cpp.bak_b2): publish each served layer's table for passes seq+1..seq+G;
at the start of a layer's service, clear its tables for seq+1..seq+G-1 left by its previous service (before its pool
changes). Default G = 1 = unchanged. Arms (~/Fit12/data/o2), all v3j + JOINT=1: o2_v2 (G 1), o2_pub (G 4), o2_oo (G 4 + OoO),
a/b; o2_cnt (lazy phase-1 pair counts, code + prose); o2_59 (C_m 59 + trims + G 4 + OoO).
Predictions: every sha = u_nat. o2_cnt phase-1 share 45-70 % of verify pairs (11.2 misses of ~28 unique ids per layer;
ranks 0-1 need both parts). o2_pub vs o2_v2: 0..+2 % (fast path on all-resident rows). o2_oo vs o2_pub: +4..+8 %.
Kill: o2_oo < +2 % over o2_pub, phase-1 share < 20 %, or any sha differs (then G 4 is unsafe: do not fold).

## O2 RESULT (08:32; ~/Fit12/data/o2, lock 08:23-08:30): tables ahead exact but no gain; OoO +1.3 % -> KILLED
All 30 decode shas = U1 u_nat (8 arms; B12's fast path now fires on trunk layers and results stay bitwise).
- o2_cnt: phase-1 (resident at post) share 23.4 % of verify pairs (48994 / 209700; prediction 45-70 %, kill < 20 %).
- o2_pub vs o2_v2 (tables for seq+1..4 vs seq+1): code 22.4/22.5 vs 22.4/22.5, prose 13.8 = 13.8, math 17.7/17.7 vs
  17.6/17.7, write 15.3 = 15.3 (o2_v2_a write 13.7: one disturbed item, stall +17 ms): 0 %.
- o2_oo vs o2_pub: code +0.4 %, prose +2.5 %, math +0.3 %, write +2.0 %: mean +1.3 % < +2 % line -> OoO KILLED at 12 GB
  (at 23 % resident pairs the moved GLU work is ~3 ms/pass). C_m 59 stack o2_59: code 23.8, prose 15.0 (o59all 23.7 / 14.5).
- Kept as findings: under streamed MTP the trunk's residency tables never matched before (fixed by BIGMOE_PUB_AHEAD in
  ~/Fit12/v3j, exact); worth more where rows are often all-resident (>= 24 GB), not at 12 GB.

## O3 pre-registration (08:40): MoonShots' exact GDN on B12LL (F12_GDNX=1)
v3f is pinned at BigMoE 7b8c7b4, before 40ed49f (BIGMOE_GDN_EXACT: exact one-dispatch GDN conv+silu+l2norm + gated
norm, bitwise == MTPLX's eager chain at decode and 4-row verify; BigMoE H50 +1-4 %). B12LL runs without it. fit_b12
installs ~/MoonShots/patches/planes_engine/gdn_exact_patch after load_b12 (read-only import, no bytecode writes) and sets
MTPLX_FUSED_GDN_CONVNORM=1, MTPLX_FUSED_CONVNORM_VERIFY=1. Arms (~/Fit12/data/o3): o3_base (v3j + JOINT=1) vs o3_gdn
(+ GDNX), a/b, A4; o3_ref_gdn (stock v3f + GDNX, A4; fold check vs U1); o3_59 (C_m 59 + trims + v2 + GDNX, A2).
Prediction: all shas = u_nat; -3 to -4 ms per pass (13 chain kernels per GDN layer) -> +1.5..+3 % tok/s. Kill: < +1 % or
any sha differs.

## O3 RESULT (08:45; ~/Fit12/data/o3, lock 08:36-08:41): exact GDN bitwise but +0.8 % -> KILLED (line +1 %)
All 22 decode shas = U1 u_nat. Verify time -0.9..-1.6 ms/pass (predicted -3..-4). tok/s base a/b vs gdn a/b: code 22.4/22.5
vs 22.6/22.7, prose 13.8/13.4* vs 13.9/13.9, math 17.7/17.7 vs 17.9/15.3*, write 15.3/15.3 vs 15.4/15.5 (* = one item
with stall +4..+29 ms, the recurring single-item SSD disturbance; excluded): +0.8 % mean. Stock v3f + GDN (o3_ref_gdn):
code 21.5, math 16.9, write 14.6 vs U1 21.3 / 16.8 / 14.5 (+0.8 %), prose 12.1* (disturbed). Stack o3_59 (C_m 59 + trims +
v2 + GDN): code 23.9, prose 14.7.
NIGHT SUMMARY (line 2, exact): C_m 59 + trims (+5 %, Breakthrough folding) and B1 v2 (+5.5 %) -> code 23.8 / prose 14.7-15.0
tok/s (+12 / +13-15 % vs U1 21.3 / 13.0), bitwise U1 tokens, < 12 GB to ~2.6K-token prompts (C_m 51 + v2 for longer).
Killed tonight: direct reads (-22 %), OoO (+1.3 %), tables ahead alone (0 %), exact GDN (+0.8 %).

## O4 pre-registration (08:44): fold check of the B1 v2 patch on a clean v3f copy
/Users/kamen/Fit12/patches/b1v2_joint_v3f.diff (12 hunks: ext joint_side + v2 skip rule + CALLLOG, gated.py JOINT post /
spin ordering, stream.py BIGMOE_JOINT_SIDE0; no LFU / PUB_AHEAD / O-series code) applied to a sources-only copy of
Breakthrough's v3f = /Users/kamen/Fit12/v3k, configured + built in the batch. Arms (~/Fit12/data/o4): k_off a/b (JOINT 0),
k_on a/b (JOINT 1), k_59 (JOINT 1 + C_m 59 + trims). Prediction: all shas = u_nat; k_off = U1 (21.3 / 13.0 / 16.8 / 14.5)
within 1 %; k_on = O1's o_v2 (22.4 / 13.8 / 17.6 / 15.3) within 1 %; k_59 = o2_59 / o3_59 (code ~23.8). Kill (do not hand
over): any sha differs, a build failure, or k_on < +3 % over k_off.

## O4 RESULT (08:52; ~/Fit12/data/o4, lock 08:46-08:50; first try 08:45 stopped at configure: NB_CMAKE_DIR, no arm ran)
Fold check PASSED. v3k (clean v3f + b1v2_joint_v3f.diff) configured + built from scratch. All 18 decode shas = U1 u_nat.
k_off a/b: code 21.3/21.3, prose 12.0*/13.1, math 16.8/16.8, write 14.5/14.5 (= U1; * stall +13.5 ms, the single-item
disturbance). k_on a/b: code 22.5/22.4, prose 13.8/13.8, math 17.7/17.7, write 15.3/15.3 -> +5.4 % mean (+5.4 / +5.3 /
+5.4 / +5.5). k_59 (+ C_m 59 + trims): code 23.7, prose 14.6. Handed to Breakthrough for the ~12:04 fold.

## SB1 pre-registration (08:58): finer GPU spin back-off at 12 GB (env only; 'SB1', not the 02:35 S1)
The spin kernel polls one fresh line pair per poll and multiplies its busy-wait by SPIN_G2 = 1.1 per poll (PAIRS = 128
polls). Scout SC5 (BigMoE LEDGER): x1.1 wakes +35 / 40 / 130 us late after 0.6 / 1 / 2 ms stalls; their A/B (G1 1.05, KNEE
9000, G2 1.3, PAIRS 256) found no gain at 24 GB, but G2 1.3 after the knee makes waits > ~0.6 ms wake later. B12LL at 12 GB
waits ~2.3 ms on ~47 base calls per pass, where overshoot ~ (G2 - 1) x wait x ~0.5-0.65. Arms (~/Fit12/data/sb1, v3k + JOINT=1,
A4, a/b): s_def (defaults), s_g105 (G2 1.05, PAIRS 256), s_g103 (G2 1.03, PAIRS 384); both keep the timeout budget >= the
default's (cap 400K iterations per poll reached within the pairs). Prediction: shas = u_nat, no timeouts; verify time -2.5
(g105) / -3.8 (g103) ms per pass -> +1.5..+2.5 %. Kill: < +1 %, any timeout or sha change.

## SB1 RESULT (09:02; ~/Fit12/data/sb1, lock 08:55-09:00): G2 1.05 + PAIRS 256 KEEP (+1.8 %, exact); G2 1.03 + PAIRS 384 REJECTED
All 24 decode shas = U1 u_nat; no spin timeouts (any timeout raises in fit_b12; to = 0 in every arm).
- g105 vs def (v3k + JOINT=1): code 22.8/22.8 vs 22.4/22.4 (+1.8 %), prose 14.0/14.0 vs 13.8 (+1.4 %), math 18.0 vs 17.7/17.7
  (+1.7 %), write 15.6/15.5 vs 15.2 (+2.3 %); verify -2.3..-3.8 ms/pass (predicted -2.5). Disturbed single items excluded
  (def_b prose 13.5, def_b write 12.8, g105_b math 16.0: stall +3..+30 ms). Mean +1.8 % > +1 % line -> KEEP.
- g103 (G2 1.03, PAIRS 384): prose / math / write +2.2 / +2.3 / +3.3 %, but code 19.8 / 20.2 in BOTH repeats (-11 %, loader
  stall +20 ms/pass: not the single-item disturbance; mechanism unknown, maybe the 3x release writes) -> REJECTED.
- Stack v2 + g105 = code 22.8 / prose 14.0 / math 18.0 / write 15.6 (+7.0 / +7.7 / +7.1 / +7.6 % vs U1); + C_m 59 + trims
  estimated code ~24.1 / prose ~14.9 (not run). Env only: BIGMOE_SPIN_G2=1.05 BIGMOE_SPIN_PAIRS=256 (gated.py reads PAIRS at
  import: set them in _env()); region memory +6.3 MB (+6.3 MB more for the phase-B regions).

## GP1 pre-registration (09:50): GDN share of the compute per pass at 12 GB; the full exact stack once
Ask (orchestrator 09:23): Leap's 35B profile (~/Leap/TABLE_350.md) puts the biggest kernel gap in GDN (12.1 ms vs a 2 ms
floor per 16-row pass, 30 layers; 3.7 ms at 1 row, 6.6 at 8). How much of B12LL's ~43 ms compute per pass is GDN?
Arms (~/Fit12/data/gp1; v3k + JOINT=1 + SB1 env, A4): gp_prof = F12_PROF=1 (scripts/fit_prof.py: a sync before and after
every GDN / attention / PLE / decoder-layer call; gpu = eval minus a null round trip; profile only, its tok/s is not
comparable; shas must = U1 or the profile is void). gp_ref a/b (C_m 51) and gp_full a/b (+ C_m 59 + trims): the full stack
measured once, replacing the ~24.1 / 14.9 estimate. Prediction: GDN gpu 5-9 ms per 4-row verify pass (36 layers),
attention 1-3 ms, PLE < 1 ms. Decision: GDN >= 8 ms/pass -> a multi-row GDN kernel near its floor is worth >= +4 % at
12 GB (code pass ~130 ms): tell Leap / NewIdeas it is a line-2 lever too; <= 4 ms -> not a line-2 lever (< +2 %).
Full stack: shas = U1 on every arm; report tok/s vs gp_ref in the same batch.

## GP1 RESULT (~/Fit12/data/gp1, lock 10:30-10:35): full stack measured (+12-14 % vs U1); GDN ~12 ms per verify pass (synced)
All 20 decode shas = U1 u_nat (the profile arm too: the wrapper is math-neutral).
- Full stack (v2 + SB1 + C_m 59 + trims) a/b: code 24.0/24.0, prose 14.7/14.8, math 18.8/18.8, write 16.2/16.1. Ref (v2 + SB1,
  C_m 51) a/b: 22.8/22.8, 14.0/14.0, 18.0/17.9, 15.6/15.6 -> C_m 59 + trims +5.3 / +5.4 / +4.7 / +3.5 %; the stack vs U1
  +12.7 / +13.5 / +11.9 / +11.4 % (estimate was code 24.1 / prose 14.9). Lifetime peak footprint 11.22 -> 11.94 GB over the
  four prompts (the harness keeps all four prompt states; ref 11.36-11.57).
- Profile (gp_prof, per 4-row trunk verify pass, 37-57 passes per prompt, null round trip 0.17 ms subtracted): GDN x36 gpu
  11.8-12.3 ms (wall 19.2-19.7, build 1.3-1.5); attention x12 3.4-3.8; PLE x1 (layer 2 only) 0.4. MTP head = one attention
  layer, ~3.2 S=1 draft calls per pass at ~2.8-3.2 ms wall each (with syncs). Pipelined verify - loader stall = 39.6 / 39.4 /
  39.0 / 39.4 ms -> GDN ~30 % of the compute if the synced time is its pipelined cost.
- Floor (config: 16 k-heads x 128, 48 v-heads x 128, hidden 2560, 4-bit g64): 32.6 MB of weights per GDN layer = 1.17 GB per
  pass + 0.23 GB fp32 state read/write = ~1.4 GB -> ~4-5 ms at ~300-340 GB/s. Gap ~7-8 ms if the 12 ms holds.
- Decision (pre-registered): >= 8 ms -> line-2 lever too. 12 ms passes, but the sync also times Metal encoding, which the
  async decode overlaps with the stalls -> GP2 calibrates before Leap / NewIdeas are told.

## GP2 pre-registration (before the 10:38 lock): calibrate the synced GDN time
Arms (~/Fit12/data/gp2; v3k + JOINT=1 + SB1, A4): gp_prof2 (F12_PROF=1 with an encode / wait split: encode = mx.async_eval
returning), gp_prof2x (+ F12_GDNX=1), gp_ref_c / gp_x_c (pipelined, without / with GDNX). Shas = U1 in all (GDNX bitwise, O3).
Reading: r = pipelined GDNX saving (O3: -1.3 ms/pass) / synced GDNX saving; pipelined GDN ~ r x 12 ms. Prediction: encode
3-5 of the 12 ms, r 0.5-1. Decision: pipelined GDN >= 8 ms -> a line-2 lever (Leap / NewIdeas, with the floor); 4-8 ms ->
report as +2-4 % potential; <= 4 ms -> not a line-2 lever.

## GP2 RESULT (~/Fit12/data/gp2, lock 10:38-10:42): pipelined GDN ~8-12 ms per verify pass (central ~10) -> a modest line-2 lever
All 16 decode shas = U1 u_nat (profile arms and GDNX arms too).
- Synced GDN x36 (gp_prof2): gpu 11.7-12.3 ms = encode 3.2-3.5 + wait 14.4-14.7 - 36 null round trips (0.17 ms each).
- GDNX calibration: synced gpu -2.3..-2.6 (encode -1.0..-1.1, wait -1.3..-1.4); pipelined verify (gp_x_c vs gp_ref_c)
  -1.1 / -0.8 / -1.7 / -1.2 ms/pass (O3: -1.3), tok/s 23.0 / 14.1 / 18.1 / 15.7 vs 22.8 / 14.0 / 18.0 / 15.6. The pipelined
  saving tracks the synced WAIT (GPU work), not the encode (overlapped with the stalls by ASYNC_EVERY) -> pipelined GDN ~ wait
  minus per-call completion latency (0.05-0.17 ms) = ~8.4-12.7 ms, central ~10 ms per 4-row verify pass = ~20-30 % of the
  ~40 ms non-stall verify time. Attention x12: wait 3.4-3.7 -> pipelined ~1.5-3.4 ms (floor ~1-1.4): not a lever.
- Floor ~4-5 ms (GP1) -> a multi-row GDN kernel near its floor saves ~4-7 ms/pass = ~+3-5 % exact at 12 GB (code pass
  ~148 ms incl. draft). Decision (pre-registered >= 8 ms): a line-2 lever, modest; reported to Leap / NewIdeas / orchestrator.
- The remaining ~28 ms of non-stall compute is the MoE side (HC, router, shared expert, tier_glu over 4 rows x top-10) and the
  post / spin overheads: not split here.

## GP3 pre-registration (before the 10:52 lock): where the 0.89 ms between layers goes (pipelined, exact)
Call log (B2 on_a, v2 without SB1; scripts/fit_gp3.py): per verify cycle (median 171.6 ms) loader service 112.2 ms, gaps
between layers (release of L-1 -> post of L seen) 41.8 ms = 0.89 ms per layer, drafts ~10, tail 5.1 + head 3.6. The loader
busy-polls every row, so pickup costs microseconds; the gap is GPU work plus the spin wake (~0.06 ms). Mixers ~13 ms (GP2) ->
~27 ms per pass on the MoE side (tier_glu, HCs, router + top-20, post, selection chain), floor ~12 ms. The synced profile
cannot split it (the block's own async_eval overlaps the stall with the timing window), so: duplicate each piece on the verify
rows and make its consumer wait for the copy (scripts/fit_dup.py, F12_DUP; mx.depends keeps the values -> shas = U1).
Arms (~/Fit12/data/gp3; v3k + JOINT=1 + SB1, A4): g3_ref a/b (a with BIGMOE_CALLLOG), g3_tg, g3_rt, g3_sh, g3_sel, g3_k10.
Prediction (verify ms/pass added): tg +5-9, rt +2-4, sh ~0 (already behind the stall) or +1.5-2.5, sel +3-6, k10 +2-4
(5-8 us per tiny kernel). Decision: a piece that costs >= 4 ms/pass above its floor is a named kernel target (owner: BigMoE /
Leap); k10 >= 5 us per kernel -> fusing the ~25 small ops per layer (top-20 + selection) into one exact Metal kernel is worth
>= 6 ms/pass (+4 %) -> my next build. Shas must = U1 in every arm.

## GP3 RESULT (~/Fit12/data/gp3, lock 10:52-10:58): tier_glu = 12.4 ms per verify pass, at the bandwidth roofline
All 28 decode shas = U1 u_nat. Disturbed items excluded (stall +15..+21 ms): k10 math / write, ref_b write.
- Added verify ms per pass vs ref (a/b mean): tg +12.2 / +12.4 / +12.4 / +12.7 (code / prose / math / write); rt -0.05 /
  +0.15 / -0.05 / -0.5 (~0); sh +0.65 / +0.45 / +1.05 / +0.3 (~0.6: mostly behind the stall); sel +0.75 / +0.35 / +0.45 /
  +0.2 (~0.45); k10 +0.35 / +0.25 (480 tiny kernels: ~0.6 us each). Predictions: tg +5-9 (MISS: higher), rt +2-4 (MISS: ~0),
  sh ~0 (HIT), sel +3-6 (MISS: lower), k10 +2-4 (MISS: lower). Small kernels are nearly free: no fusion lever there.
- Why tier_glu costs 12.4: v3f's kernel (tierglu.py) runs one grid row per (row, rank) pair, p = m * R + r, so each of the
  40 pairs per layer reads its expert's weights: 4 x (2 x 2.21 + 8 x 1.72 MB) x 48 = 3.49 GB per pass = 281 GB/s in 12.4 ms
  = Leap's 280 GB/s floor. Routing traces (~/NewIdeas/data/samelayer, 4-token windows): 28.6 distinct experts of 40 (code,
  GSM), 25.3 (IFEval + multi) -> reading each distinct expert once for all its rows = 0.72x / 0.64x the bytes -> -3.5 / -4.5
  ms per pass (+2.4 / +3.0 % exact at 12 GB), if the SLC does not already absorb part of the repeats. Same per-row arithmetic
  order -> bit-identical (as Leap's GDN16). At 16 rows (Leap / DFlash) 69 distinct of 160 pairs (union_forecast): 0.43x.
- Call log of the current stack (clog_ref_a, fit_gp3.py): cycle 164.9 ms median, loader service 108.7, gaps 39.2 (0.83 ms
  per layer), tail 4.8, head 3.9, draft service 4.9. Gaps known so far: tier_glu 12.4 + GDN ~10 + attention ~2.5 + small
  ops ~1.5 + spin wake ~2-3 + PLE 0.4 = ~29-30 of 39.2 -> ~9-10 ms left, likely the three HC ops per layer -> GP4.

## GP4 pre-registration (before the 11:01 lock): the hyper-connection ops' pipelined cost
Arms (~/Fit12/data/gp4; v3k + JOINT=1 + SB1, A4): g4_ref a/b, g4_hca (attention HC read again), g4_hcm (MLP HC read with
the pending attention write again), g4_hcw (residual write again); each copy feeds the next consumer via mx.depends (exact).
Prediction: hca + hcm + hcw together +6-10 ms/pass (the unexplained ~9-10). Decision: >= 4 ms/pass above a ~1-2 ms floor ->
HC is a named kernel target (owner: MoonShots, hc_pack); < 2 ms in total -> the leftover is elsewhere (pass edges / wake).

## GP4 RESULT (~/Fit12/data/gp4, lock 11:01-11:06): hyper-connections ~2.25 ms per pass in total -> not the leftover
All 20 shas = U1. ref_b write disturbed (stall +15 ms; the last item of the last run, as in GP3) -> excluded.
- Added verify ms per pass vs ref: hca +1.5 / +0.85 / +1.05 / +0.8 (~1.05); hcm +1.2 / +1.35 / +0.85 / +1.1 (~1.1); hcw 0.0 /
  +0.25 / +0.25 / 0.0 (~0.1). Prediction +6-10 in total: MISS (2.25). Decision: < 4 ms above floor -> HC is not a target.
- Gaps (39.2 ms) known so far: tier_glu 12.4, GDN ~10 (8.4-12.7, estimate), attention ~2.5 (1.5-3.4), HC 2.25, small ops
  ~1.5, spin wake ~2-3, PLE 0.4 -> ~31-32; ~7-8 ms left (2-12 with the GDN / attention ranges).

## GP5 pre-registration (before the 11:09 lock): the GDN / attention projections in the pipelined decode; buffer boundaries; fast sync
MoonShots round 30 (IDEAS.md): raising MLX's buffer limits breaks the spin gate (no raise here). Arms (~/Fit12/data/gp5;
v3k + JOINT=1 + SB1, A4; all exact): g5_ref a/b; g5_gdnp (F12_DUPP=gdnp: in_proj_qkv / z / b / a + out_proj of the 36 GDN
layers twice on S > 1 calls, output waits for the copy); g5_attp (q / k / v / o of the 12 attention layers twice);
g5_ops20 (MLX_MAX_OPS_PER_BUFFER=20 vs the default: more boundaries -> the cost per boundary); g5_fs (MLX_METAL_FAST_SYNCH=1,
BigMoE LEDGER #20, never run). Prediction: gdnp +5-8 ms/pass (1.17 GB of projection weights = 4.2 ms at 280 GB/s, at
50-80 % of the floor), attp +1-2, ops20 +1-3, fs -0.5..-1 (+0.3-0.6 %). Decision: fs >= +1 % with shas = U1 -> a free
env keeper; ops20 >= +2 ms -> buffer boundaries are worth MoonShots' explicit-commit design (BigMoE's gate); gdnp fixes
GDN's pipelined share for Leap's 125B estimate.

## GP5 RESULT (~/Fit12/data/gp5, lock 11:09-11:15): GDN projections ~4.5 ms (near their 4.2 ms floor); boundaries and fast sync no lever
All 24 shas = U1. ref_b math disturbed (stall +5.5 ms) -> excluded.
- gdnp (180 modules = 36 x 5): +4.25 / +4.85 / +4.3 / +4.45 ms per pass (code / prose / math / write) ~ +4.5 vs the bytes
  floor 1.17 GB / 280 GB/s = 4.2 ms. attp (48 modules): +1.05 / +1.25 / +1.3 / +1.55 ~ +1.3 vs ~1.2 floor. Prediction gdnp
  +5-8: MISS (lower); attp +1-2: HIT. Caveat: the copy re-reads the same 9-15 MB right after the original, so it may hit the
  SLC and undercount -> GP6 (copy on the next layer's weights).
- ops20 (MLX_MAX_OPS_PER_BUFFER=20): +0.05 / +0.65 / +0.8 / +0.55 ~ +0.5 ms for ~150-200 extra boundaries -> ~3 us each:
  fewer boundaries (MoonShots' explicit-commit design) would save < 0.5 ms -> not a lever (prediction +1-3: MISS).
- fs (MLX_METAL_FAST_SYNCH=1): -0.25 / +0.05 / -0.1 / +0.05, tok/s identical -> 0 (< +1 %) -> not a keeper.

## GP6 pre-registration (before the 11:16 lock): SLC-proof projection copies
Arms (~/Fit12/data/gp6; v3k + JOINT=1 + SB1, A4): g6_ref a/b, g6_gdnpx / g6_attpx (the copy runs the next GDN / attention
layer's same-shape weights on this layer's input; those were last read ~a pass ago). Reading: gdnpx ~ gdnp (+4.5) ->
projections at their floor at 4 rows, GDN's kernel gap is the recurrence / conv / norms / capture part (~4-8 ms); gdnpx
>= +7 -> the SLC hid part of gdnp's copy and the projections are at <= 60 % of their floor (a multi-row matmul helps).

## GP6 RESULT (~/Fit12/data/gp6, lock 11:16-11:20): projections at their floor at 4 rows (SLC-proof)
All 16 shas = U1, no disturbed item. gdnpx (copy on the next GDN layer's weights) +4.8 / +4.6 / +4.4 / +4.85 ~ +4.65 ms
(gdnp +4.5); attpx +1.4 / +1.4 / +1.3 / +1.65 ~ +1.45 (attp +1.3). The SLC hid little: GDN projections ~4.6 ms = ~90 % of
their 4.2 ms bytes floor; attention ~1.45 vs ~1.2. -> GDN's kernel gap at 4 rows is the non-projection part (recurrence,
conv, norms, gate, capture / repair): ~5.4 ms (3.8-8.1) over a ~1 ms floor. A multi-row matmul does not help the 125B's
4-row verify; Leap's recurrence / elementwise work does.

## RH1 pre-registration (before the 11:23 lock): hybrid aligned record reads (loader, exact)
The v3k loader reads each missing record part with one aligned pread into a per-thread bounce, then memcpy's all 9 pieces
into the pools (B12 base part 1.72 MB, ~1.1 GB per pass of CPU copying after the reads, on the layer's critical path).
B2's BIGMOE_RECORDS_DIRECT (preadv into all 9 slots) lost 22 %: its 40 KB s / b destinations are not page-aligned. New
BIGMOE_RECORDS_HYBRID=1 (v3k ext, grep "Fit12 RH1"; backup bigmoe_ext.cpp.bak_rh1): one preadv with each 491,520-byte w
piece straight into its pool slot and each s + b pair (81,920 = 5 pages) into an aligned bounce; only s / b are copied
(0.25 of 1.72 MB). Layout check per layer (else the bounce path); the LSB part keeps the bounce path. Arms (~/Fit12/data/rh1,
v3k + JOINT=1 + SB1, A4, ABAB): rh_ref a/b, rh_hyb a/b. Prediction: loader stall -3..-6 ms/pass, +2-4 % tok/s, shas = U1.
Kill: < +1 %, any sha change or spin timeout. The ref arms also check the rebuilt ext's default path (= GP6 ref +- 0.5 ms).

## GP3 CORRECTION (after RH1, 11:28): the expert is moe_intermediate_size 640, not 512 (config; RH1's piece log: w3 614,400 B)
Per projection w3 614,400, s / b 51,200, l1 204,800 B: expert base 2.150 MB, ranks 0-1 (w3 + s3 + l1 + b4) 2.765 MB.
tier_glu's per-pair bytes = 4 x (2 x 2.765 + 8 x 2.150) x 48 = 4.36 GB per pass = 352 GB/s in 12.4 ms: above the ~280 GB/s
DRAM rate, so the SLC already serves part of the cross-row repeats (or the dup copy undercounts). Explicit dedup = 3.13 GB
(code) / 2.78 GB (IFEval + multi) -> 11.2 / 9.9 ms at 280 GB/s -> saves ~1.2 / ~2.5 ms per pass (+0.8 / +1.6 %), not the
3.5 / 4.5 ms (+2.4 / +3.0 %) above. tier_glu stays the largest GPU piece and at the roofline; its dedup is a small lever.

## RH1 RESULT (~/Fit12/data/rh1, lock 11:23-11:27): NOT APPLICABLE -> KILLED (no change measured, exact)
All 16 shas = U1. The layout check refused every layer (log: base 9 pieces 614,400 / 51,200 / 51,200 B; LSB 6 pieces 204,800
/ 51,200): no piece is a 16 KB multiple (614,400 = 37.5 pages), so page-aligned direct reads into per-slot pools cannot exist
without padding the pool slot strides (an engine change: tier pools + tier_glu addressing, ~+58 MB at C_m 51). hyb = ref:
code 22.8 / 22.8, verify 139.3 / 139.2 vs 139.3 / 139.4 -> the rebuilt v3k ext's default path is unchanged (= GP6 ref).
The bounce copy (2.15 MB per base record, ~35-70 us after the last read of a layer) is estimated at ~1.7-3.4 ms per pass:
the padded-stride + direct-read design would be worth <= +1-2 %, Breakthrough / BigMoE's call.

## Per-pass map at 12 GB (B12LL + v2 + SB1, C_m 51; GP1-GP6, all pipelined + exact unless noted)
Verify cycle (median, 4 prompts) ~165 ms = loader service ~108.7 (66 %) + gaps between layers ~39.2 (24 %) + drafts ~8.3
(incl. MTP-head stalls 4.9) + pass edges ~8.7 (tail 4.8 + head 3.9). In the gaps (per pass): tier_glu 12.4 (at the
roofline; dedup -1.2..-2.5), GDN ~10 (projections 4.65 at ~90 % of floor; the rest ~5.4 over a ~1 ms floor), attention
~2.5 (projections 1.45 ~ floor), hyper-connections 2.25, router + top-20 ~0, shared expert ~0.6, selection ~0.45, PLE 0.4,
spin wake + post visibility + misc ~5-7. Command-buffer boundaries ~3 us each; MLX_METAL_FAST_SYNCH 0.
Exact levers left: GDN recurrence / conv / norms (~-4 ms, +2.5 %), tier_glu dedup (+0.8-1.6 %), padded-stride direct reads
(<= +1-2 %), attention non-projection (~-1 ms). Together ~+4-6 %; the SSD service (66 %) has no cheap exact lever left.

## D2 pre-registration (before the 11:31 lock): the MTP head drafts with resident experts (BigMoE LEDGER D2; moot at 24 GB, live at 12 GB)
At 12 GB the MTP head streams (BIGMOE_MTP_STREAM=1, C 64 of 512): draft service 4.9 ms per cycle of ~165 (call log). Drafts
only set how many tokens a pass accepts; at the fixed 4-row verify every row's math is its own -> the output is the
verify's tokens whatever the drafts. F12_ND=n (scripts/fit_dup.py): the head's one-row calls (L == NL) wait only for ranks
< n, the other ranks take resident experts from its top-20 (B12's 'first resident of the rest'). Arms (~/Fit12/data/d2;
v3k + JOINT=1 + SB1, A4, ABAB): d2_ref, d2_nd2, d2_nd4. Prediction: draft ms -2..-4 per pass, tok/pass -0..-5 %, net
-1..+2 %. Keep: shas = U1 on every arm AND >= +1 % tok/s mean; kill: any sha change (would mean the verify is not
batch-invariant: a finding by itself), or < +1 %.

## D2 RESULT (~/Fit12/data/d2, lock 11:31-11:37): VOID -> the patch never ran (all 24 shas = U1, every number = ref)
nd2 / nd4 = ref in passes (38 / 59 / 40 / 52), tok/pass (3.37 / 2.17 / 3.20 / 2.46) and draft ms (~8.0): the MTP head is not a
B12 tier layer (stream.py:569: its _bm has no "tier") -> bt_b12's block returns orig() -> v3's plain streamed path
(route_and_gate, every rank waited), which F12_ND does not touch. A real D2 needs v3's non-tier path to post flag-mode rows
(ranks < ND waited) and mask the non-resident ranks' expert outputs (weight 0): ~30 lines in v3k's stream.py. Ceiling: the
draft service 4.9 ms per cycle (3 %); with ND = 2 maybe -3 ms, minus the acceptance lost to dropped experts -> <= +1-2 %
net, below or near the +1 % line. Parked (not built); the output stays exact either way (the verify decides).
Note (11:38): header times of GP2-GP6 / RH1 / D2 corrected from logs/summary.txt. Each pre-registration was appended
right before its lock request; the clock times first written in those headers were estimates, some later than the run.

## GR1 pre-registration (before its lock): Leap's exact GDN recurrence at the 125B's 4-row verify
Leap (lounge 11:39) offered ~/Leap/scripts/gdn35_exact.py for GDN's non-projection gap (~5.4 ms/pass at 4 rows, GP6):
recurrence_r4 = mlx_lm 0.31.3's gated_delta kernel expressions in the same order, 4 value rows per thread, 16-token chunks
in threadgroup memory (-24 % at T = 16 on the 35B). Shape-generic, and g / beta stay mlx_lm's ops (their gates variant uses
MLX 0.32.4's sigmoid; this venv is 0.32.2, so it is NOT used). scripts/fit_gdnr.py (F12_GDNR=1) swaps mlx_lm's
gated_delta_kernel for it on unmasked scalar-g calls (verify, commit repair, prefill). Pre-check (scripts/gr1_check.py):
bitwise outputs + states at Hk 16 / Hv 48 / 128 x 128, T = 1, 3, 4, 16, 512, else no decode arms. Arms (~/Fit12/data/gr1;
v3k + JOINT=1 + SB1, A4): gr_ref a/b, gr_r4 a/b, gr_x (GDNX alone), gr_r4x a/b (r4 + GDNX). Prediction: r4 -0.5..-1.0 ms
per pass (+0.3-0.7 %), r4x -1.7..-2.2 (+1.1-1.5 %). Keep (each): shas = U1 and >= +1 % mean; GDNX alone stays killed.

## GR1 RESULT (~/Fit12/data/gr1, lock 11:41-11:47): r4 alone ~0 (KILLED); r4 + GDNX +1.1 % (narrow keeper, mostly GDNX)
Pre-check BITWISE (y and state, T = 1 / 3 / 4 / 16 / 512 at Hk 16, Hv 48, 128 x 128). All 28 shas = U1; recurrence_r4
served all 12,564 calls (0 fallbacks). gr_r4x_b code / write disturbed (stall +21.5 / +13.5 ms) -> excluded.
- Compute per pass (verify - stall) vs ref a/b: r4 -0.15 / -0.3 / -0.05 / -0.2 (~-0.2 ms, ~+0.1 %; prediction -0.5..-1.0:
  MISS); GDNX alone (x_a) -1.3 / -1.45 / -1.25 / -1.45 (-1.36 ms; tok/s 23.0 / 14.1 / 18.1 / 15.7 vs 22.75 / 13.95 / 17.95 /
  15.5); r4 + GDNX -1.3 / -1.65 / -1.4 / -1.65 (-1.5 ms; tok/s 23.0 / 14.15 / 18.15 / 15.7 = +0.9 / +1.4 / +1.0 / +1.3 %,
  mean ~+1.1 %; prediction -1.7..-2.2: lower).
- Decision (pre-registered: each >= +1 % with shas = U1): r4 KILLED; r4 + GDNX passes narrowly (+1.1 %). Its gain is ~90 %
  GDNX, whose three pipelined measurements are -1.3 (O3), -1.2 (GP2), -1.36 (GR1) ms = ~+0.9 % mean: the pair is a
  ~+1.0-1.1 % exact item at the line, worth folding only if cheap (BigMoE 40ed49f already has BIGMOE_GDN_EXACT; r4 is
  Leap's file). At 4 rows the recurrence is small: GDN's remaining ~4 ms is spread over conv / norms / gate / repair.

## SS1 pre-registration (before its lock): SSD burst throughput vs read splitting (read-only microbenchmark, no GPU)
Loader service = 66 % of the cycle; reads run at ~10.5-11 GB/s within a layer (vs ~14.6 GB/s measured for the SSD before).
scripts/ssd_burst.py: bursts of 6 / 12 / 20 random experts' MSB parts (2,162,688 B) from records3, uncached, page-aligned,
as 1 / 2 / 4 / 6 reads per part, interleaved; median and p90 burst time. Decision: if 2-4 reads per part cut the 12-part burst
>= 8 % -> a split-read option in the v3k loader (same bytes, same slots: exact) is worth an A/B; < 4 % -> the loader's
single aligned read per part stays, and the SSD side is at its rate for this pattern.

## SS1 RESULT (logs/ss1.txt, quiet lock ~12:10, SSD idle): one aligned read per part is fastest; the loader is at the SSD's rate
Median burst ms (GB/s) by reads per part 1 / 2 / 4 / 6: 12 parts 2.16 (12.0) / 2.19 / 2.24 / 2.30; 6 parts 1.16 (11.2) / 1.17 /
1.19 / 1.22; 20 parts 3.49 (12.4) / 3.54 / 3.63 / 3.73 (p90 within 0.1 ms of the median everywhere). Fit: ~0.16 ms fixed per
burst + 2.16 MB at ~13 GB/s marginal per part. Splitting only adds cost (decision: < 4 % -> no split-read option). The decode's
~2.26 ms of service per layer (12 base parts + sidecars + bookkeeping) is within ~5 % of this burst floor: the loader term (66 %
of the cycle) is SSD-bound at its rate for random 2 MB parts (the earlier 14.6 GB/s was not this pattern). Exact levers there
= fewer bytes only (residency, cache policy: tested) or the ~0.16 ms x 48 burst latency (needs reads before the post).

## LK1 pre-registration (before its lock): spec125 prompt lookup on B12LL at 12 GB (never measured here; the suite runs 3:3:0)
spec125 (read-only) drafts the 3 MTP tokens, then uses its prompt-lookup draft (prompt + generated text) instead when one
exists and the MTP head agrees on its first token (xcheck). max_draft 3 keeps every verify at <= 4 rows (B12's SMAX), but
widths now vary 2-4 -> exactness needs B12LL's per-row math to be width-invariant (BigMoE: exact band S <= 5 at 24 GB).
No draft call is saved in this path; the gain can only come from better acceptance on copy-like spans and shorter (cheaper)
verifies. Arms (~/Fit12/data/lk1; v3k + JOINT=1 + SB1, A4, ABAB): lk_ref, lk_l1 (lookup 1: standard), lk_l2 (lookup 2: >= 8-
gram only). Prediction: shas = U1; lookup used on 5-25 % of passes; tok/s 0..+3 % (code most). Keep: shas = U1 on every arm
and >= +1 % mean with no prompt < -2 %; any sha change -> killed + reported (width-variance finding).

## LK1 RESULT (~/Fit12/data/lk1, lock 12:11-12:17): lookup 1 +1.2 % exact (narrow keeper); lookup 2 +0.6 % (killed)
All 24 shas = U1 with widths 2-4 (lookup 1 used in 13 / 0 / 5 / 11 passes of 38 / 59 / 40 / 52): B12LL's verify is
width-invariant over S 2-4 on these prompts. tok/s a / b: ref code 22.7 / 22.7, prose 13.9 / 13.9, math 17.8 / 17.9, write
15.4 / 15.5; lookup 1: 22.9 / 22.9, 13.9 / 13.9, 18.3 / 18.2, 15.7 / 15.7 (+0.9 / 0 / +2.2 / +1.6 %, mean +1.2 %); lookup 2:
22.7, 13.9, 17.9, 15.8 / 15.8 (mean +0.6 %). tok/pass and pass counts are UNCHANGED under lookup 1: the gain is shorter
verifies (2-token lookup drafts -> 3 rows -> fewer reads; math stall -2.4 ms/pass), not more accepted tokens. Draft ms is
+0.7 in every arm of this batch, ref included (Mac busier), so the A/B holds. Prediction (5-25 % use, 0..+3 %): HIT.
Decision: lookup 1 passes (+1.2 %, no prompt negative, exact) -> a candidate for the suite's _gen_mtp (3:3:1), prompt-
dependent (copy-heavy text gains most); the suite confirms token identity on its own prompts.

## WT1 pre-registration (before its lock): confidence width on NATURAL B12LL (AW was only ever run under b16, F3)
Each verify row costs its own top-10 at 12 GB (LK1: shorter verifies alone gave +1-2 %). F12_WTAU=tau (fit_b12.py width hook):
verify the MTP drafts while their cumulative head probability >= tau, at least 1 (a 1-row pass takes B12's late path, other
arithmetic); lookup drafts whole. Arms (~/Fit12/data/wt1; v3k + JOINT=1 + SB1, A4, ABAB): wt_ref, wt_t3 (0.3), wt_t5 (0.5).
Prediction: drafts per pass 3.0 -> 2.4-2.7; tok/pass -2..-5 %; verify -8..-15 ms per cut row; net +2..+6 % prose / write,
0..+2 % code / math. Keep: shas = U1, >= +1 % mean, no prompt < -2 %.

## WT1 RESULT (~/Fit12/data/wt1, lock 12:19-12:25): confidence width is fast, but B12LL's verify is NOT width-invariant
Both repeats identical (a = b to 0.1 tok/s and in every sha). tok/s (sha vs U1):
| arm | code | prose | math | write |
|---|---|---|---|---|
| ref (a/b) | 22.6 | 13.9 | 17.8 | 15.4 |
| tau 0.3 (drafts/pass 2.84 / 2.35 / 2.92 / 2.43) | 23.15 (= U1) | 14.9 (= U1) | 17.95 (= U1) | 18.4 (83ed65b6d3, CHANGED) |
| tau 0.5 (2.48 / 1.84 / 2.80 / 1.98) | 22.65 (06035b51da, CHANGED) | 16.65 (887629baa7, CHANGED) | 18.3 (= U1) | 18.85 (f28e8c2951, CHANGED) |
- Exact cells: tau 0.3 code +2.4 %, prose +7.2 % (verify 133 vs 146 ms for 0.65 fewer rows: ~20 ms per verified row), math
  +0.8 %; tau 0.5 math +2.8 %. Changed cells are NOT exact (the text differs, so their speed is confounded by the new text).
- Finding: with widths varying often (2-4 rows, never 1), tokens leave U1 -> some op's per-row result depends on the verify
  width (rounding only: same model, same math), flipping near-tie argmaxes. LK1's 29 narrower passes per run hit no near-tie;
  its exactness holds on these 4 prompts only. Prediction (exact, +2..+6 %): MISS on exactness.
- Decision: tau policies NOT adopted (non-exact). Next: find the width-variant op(s) (row-0 bitwise at M = 2 / 3 / 4 per op
  class) and make the verify width-invariant -> confidence width (and lookup) become exact; worth ~+3-8 % exact on uncertain
  prompts, more than tier_glu dedup / padded strides (~+1-2 % each).

## WV1 pre-registration (before its lock): find the width-variant op (diagnostic, no keep/kill)
scripts/fit_wv.py: per prompt, one prefilled state; spec125's verify forward on the same 4 tokens at widths 4 / 3 / 2 (fresh
restore_state each); records input + output rows of every PLE / GDN / attention / MoE block / decoder layer (trunk, S > 1);
reports the first module whose shared rows differ bitwise (an input-only difference = the compiled HC before it) and whether
the logits rows / argmax differ. Expectation: the first difference sits in a dense op with an M-dependent kernel choice
(quantized matmul qmv vs qmm, or a verify-width fused path), not in tier_glu (one grid row per pair).

## WV1 RESULT (~/Fit12/data/wv1, lock 12:31): one verify pass IS width-invariant
From one prefilled state, the same tokens at widths 4 / 3 / 2: all 145 recorded calls (48 x mixer / MoE / layer + PLE) equal
bitwise on the shared rows, and the logits rows too, on all 4 prompts. So WT1's token changes come from CHUNKING: a position
verified at row 2 of a 4-row pass vs row 0 of the next pass after a shorter one, i.e. from state carried across a commit.

## WV2 pre-registration (before its lock): chunk invariance across the commit (diagnostic)
scripts/fit_wv.py probe2: commit the same 4 tokens as one 4-row pass or as two 2-row passes (keep all rows, as generate on
full acceptance: commit_verified_window), compare every cache array bitwise (GDN recurrent + conv state, PLE conv state,
attention KV), then the logits of an identical next 4-row pass. Expectation: a carried state in reduced precision (e.g. a
bf16 cast at commit) or the capture-commit's re-run path differs from the in-kernel value.

## WV2 RESULT (~/Fit12/data/wv2, lock 12:33): full-accept chunking IS invariant
4 tokens committed as one 4-row pass or two 2-row passes (all kept): every cache array (ArraysCache, QSACache) bitwise equal
and the next 4-row pass's logits equal, on all 4 prompts. Left: partial accepts. commit_verified_window (MTPLX) then trims
attention, replays each pure-GDN layer's recurrence over the kept rows, and replays the PLE layer (layer 2) as a FULL layer
forward over the kept rows; with keep = 1 that is a 1-row forward (decode-width HC / GDN paths, B12's one-row block).

## WV3 pre-registration (before its lock): partial-accept replay vs full accepts (diagnostic)
Positions p0..p0+4 committed three ways: ref (2 full + 3 full), keep1 (4-row pass keeping 1, then 4 full), keep2 (4-row pass
keeping 2, then 3 full); compare every cache array with ref. Expectation: keep1 differs (width-1 replay of layer 2), keep2
equal (width-2 replay = verify width) -> the fix is to never replay at width 1.

## WV3 RESULT (~/Fit12/data/wv3, lock 12:37): the keep-1 commit replay is the policy-dependent step
p0..p0+4 committed with a partial accept keeping 2 of 4 = the full-accept reference bitwise (0 arrays differ, all 4 prompts);
keeping 1 of 4: 117-118 arrays differ, first in cache[1] = the PLE-carrying layer (layer index 1; ple_layer_ids [2] is
1-based): fp32 recurrent state |d| <= 4.4e-4, bf16 conv state 1 ulp, then every later layer through the next pass.
MTPLX's commit_verified_window replays that layer as a full forward over the kept rows; at keep 1 that is a 1-row forward
(decode-width arithmetic). U1's tokens are thus a property of B12LL + exactly 3:3:0's acceptance pattern.

## PR1 pre-registration (before its lock): 2-row replay for keep-1 commits (scripts/fit_pr1.py, F12_PR1=1)
Keep-1 commits replay the PLE layer on 2 rows (kept + first rejected, from the verify's captured inputs) inside MTPLX's
boundary_capture_scope([1]) and take its slots 0-3 (GDN conv tail + state, PLE conv window + n-gram tail) at offset 1.
Gate: WV3 under PR1 must show keep1 = ref (0 arrays differ) on all 4 prompts, else no decode arms. Decode (~/Fit12/data/pr1,
A4): pr1_ref (3:3:0) defines U2; pr1_t3 / pr1_t5 (tau 0.3 / 0.5) and pr1_l1 (lookup 1) must give U2's shas; pr1_u1 (no PR1)
must give U1's. Prediction: U2 differs from U1 on 0-3 prompts; every policy arm = U2 (policy-invariant); speed as WT1 / LK1's
exact cells (tau 0.3: +2-7 %, lookup +1 %). U2 replaces U1 only after an Eval suite verdict (Kamen's call: new tokens).
Note: since Breakthrough's fold (a50d103, 12:17) bt_b12 applies the memory trims by default (B12_TRIMS=1, bitwise); my
batches from WT1 on ran with them (refs = U1 shas throughout).

## PR1 RESULT (~/Fit12/data/pr1, lock 12:41-12:47): B12LL + PR1 is draft-policy-invariant; confidence width tau 0.5 = +5.3 % on it
- Gate: WV3 under PR1: keep1 = ref and keep2 = ref, 0 cache arrays differ, all 4 prompts.
- U2 (3:3:0 + PR1) shas: code fb1e61e413, prose 42eef85996, math 6e3af723f9 (= U1), write 92b930c6cc (3 of 4 differ from U1:
  rounding-level flips where U1 committed a keep-1 pass at a near-tie). Control pr1_u1 (no PR1) = U1 on all 4.
- Policy invariance: tau 0.3, tau 0.5 and lookup 1 give U2's shas on all 4 prompts (12 of 12; without PR1, WT1 changed 4 of 8).
- PR1's cost: math (same tokens in U1 and U2) 17.9 vs 17.9 tok/s, verify +0.3 ms: ~0. Keep-1 replays: 35-45 per run.
- Speed vs U2 (same tokens, clean): tau 0.3 code / prose / math / write 23.4 / 16.2 / 18.1 / 18.8 = +2.6 / +5.2 / +1.1 / +2.7
  (mean +2.9 %); tau 0.5 22.9 / 17.0 / 18.3 / 19.8 = +0.4 / +10.4 / +2.2 / +8.2 (mean +5.3 %, none negative); lookup 1 23.2 /
  15.4 / 17.3 / 18.5 = +1.8 / 0 / -3.4 / +1.1 (mean -0.1 %: dropped). U2 vs U1 speed differs on the changed texts only (text-
  dependent: prose 15.4 vs 13.9, write 18.3 vs 15.5); math (same text) is the clean PR1-cost read.
- Prediction: policy arms = U2: HIT; tau 0.3 +2-7 %: HIT; lookup +1 %: MISS.
- Decision: PR1 + tau 0.5 is the new exact-family candidate for line 2: exact w.r.t. U2 for ANY draft policy with verify width
  >= 2 (seen on 4 prompts x 4 policies). U2 replaces U1 only after an Eval suite verdict (new tokens at near-ties; same model
  math): Kamen's call. Caveat: a 1-row verify (no drafts: guard armed, or the last token at max_tokens) still takes B12's
  one-row late path; the suite's config has the guard off; drafting >= 1 at the end would remove the last case.

## LR1 pre-registration (before its lock): Leap's lazy-replay fix on B12LL + PR1 (scheduling only, exact)
Leap (12:50): their DFlash loop left the GDN rollback replay lazy inside the next verify graph; mx.async_eval of the rolled-
back states right after the rollback gave the 35B +12.9 %. MTPLX's commit_verified_window is lazy by design. scripts/fit_lr1.py
(F12_LR1=1): after a landed commit, mx.async_eval the ArraysCache arrays (GDN / PLE states). Arms (~/Fit12/data/lr1, all with
PR1, A4, ABAB): lr_ref / lr_on (3:3:0 without / with LR1), lr_t5 / lr_t5on (tau 0.5 without / with). Shas must = U2 in all.
Prediction: verify -0.5..-3 ms per pass (the replay leaves the next pass's critical path), draft time +0..+1 ms -> +0.5-2 %.
Keep: >= +1 % mean with shas = U2.

## LR1 RESULT (~/Fit12/data/lr1, lock 12:48-12:56): KILLED (no gain on B12LL); tau 0.5 on U2 re-confirmed
All 32 shas = U2 (scheduling only, as required; 208-225 commits kicked per run). LR1 vs off, a / b: code 22.8 / 22.8 vs
22.9 / 22.9, prose 15.4 / 15.3 vs 15.4 / 15.5, math 17.9 / 17.9 vs 17.9 / 17.9, write 18.2 / 18.2 vs 18.3 / 18.4 (verify
+0.2..+0.5 ms, draft +0.1..+0.4); with tau 0.5 the same (~-0.1 tok/s). At 12 GB the commit replay is small next to the
stalls and already hidden (Leap's 35B case was the replay inside the next verify graph in kernel mode). Prediction +0.5-2 %:
MISS. tau 0.5 on U2 (lr_t5 vs lr_ref): code -0.4 %, prose +10.4 %, math +2.0 %, write +7.9 % (mean ~+5.0 %; PR1: +5.3 %).
Eval suite asked for U2's core verdict (12:55): hook = fit_hook.load_b12ll_budget("", pr1=True) (fit_hook.py pr1 option,
fit_pr1.install idempotent); engine v3f / trims 0 for a clean attribution (PR1 is engine-independent).

## WT2 pre-registration (before its lock): confidence-width threshold sweep on U2 (exact w.r.t. U2)
Arms (~/Fit12/data/wt2; v3k + JOINT + SB1 + PR1, A4, ABAB): w2_ref (3:3:0), tau 0.5 / 0.6 / 0.7. Shas = U2 in every arm.
Prediction: drafts per pass fall further (prose ~1.8 -> ~1.4), prose / write gain a little more, code / math flat or lose
(fewer accepted tokens); best mean within 0.5-0.6. Pick the tau with the best mean and no prompt < -2 % vs U2 3:3:0.

## WT2 RESULT (~/Fit12/data/wt2, lock 12:57-13:04): tau 0.5-0.7 is a plateau; tau 0.7 best (+5.7 % on U2, exact w.r.t. U2)
All 32 shas = U2 (tau 0.6 / 0.7 policy-invariant too). Disturbed (stall +16 / +21 ms): ref_b math, t6_b write -> excluded.
tok/s vs U2 3:3:0 (code 22.9, prose 15.4, math 17.9, write 18.3): tau 0.5 22.85 / 17.1 / 18.2 / 19.8 (-0.2 / +11.0 / +1.7 /
+8.2 %, mean +5.2 %); tau 0.6 22.9 / 17.1 / 18.3 / 19.7 (0 / +11.0 / +2.2 / +7.7 %, +5.2 %); tau 0.7 22.95 / 17.25 / 18.55 /
19.55 (+0.2 / +12.0 / +3.6 / +6.8 %, +5.7 %). Drafts verified per pass at 0.7: 2.17 / 1.56 / 2.46 / 1.71. Prediction (best
within 0.5-0.6): MISS by a little (0.7 slightly ahead; write peaks at 0.5, prose / math still rising). Pick: tau 0.7.
With the folded stack (Breakthrough a50d103) and C_m 51, B12LL + PR1 + tau 0.7 on these prompts: code 22.95, prose 17.25, math
18.55, write 19.55 tok/s (U2 tokens).

## FS1 pre-registration (before its lock): the line-2 candidate measured once (no decision; numbers for Kamen)
Folded stack (bt_b12 a50d103 defaults: JOINT + SB1 + trims; engine v3k = v3g code) + PR1 + tau 0.7, at C_m 51 and at the
C_m 59 opt-in (B12_CM=59 B12_CL=20; trims by bt_b12 itself, no Fit12 trim switches), ABAB; fs_59ref = C_m 59 + PR1 at 3:3:0.
Shas must = U2 in all arms (C_m is cache size only). Report tok/s and lifetime peak footprint per arm.

## FS1 RESULT (~/Fit12/data/fs1, lock 13:06-13:11): the line-2 candidate (U2 tokens, all 20 shas = U2)
fs_51_a math disturbed (verify 186.7 vs 157.1) -> excluded. tok/s code / prose / math / write (lifetime peak footprint):
- C_m 51 + PR1 + tau 0.7: 22.95 / 17.25 / 18.6 / 19.7 (10.4-11.26 GB)
- C_m 59 + PR1 + tau 0.7: 24.3 / 18.25 / 19.5 / 20.45 (11.23-12.20 GB: the 4th prompt's lifetime peak passes 12 GB with the
  harness holding 4 prompt states; GP1's C_m 59 peaked 11.94) -> C_m 59 stays opt-in for short prompts, now at the edge.
- C_m 59 + PR1 at 3:3:0: 24.2 / 15.0 / 18.8 / 19.1 (11.22-12.13 GB).

## Q2 pre-registration (before its lock): OUTPUT error of 2-bit expert codes on real MoE inputs (bytes-per-token bet)
Kamen (13:1x via the orchestrator): line 2 needs a bytes-per-token breakthrough. C1/Q1 killed scalar re-quantization on
WEIGHT error (2-bit 4.4-5.1x MSB-3's); never tested: activation-aware codes. scripts/fit_wv.py capture (F12_CAP): the MoE
inputs of layers 2 / 12 / 24 / 36 / 46 over a ~4K-token prefill of local text -> ~/Fit12/data/q2/x_L*.npy; scripts/fit_q2.py
(CPU): Q1's 40 experts, output error tr(dW H dW^T) / tr(W4 H W4^T) for MSB-3, RTN-2 g64, GPTQ-2 g64 / g128, GPTQ-3 g64.
Decision (sent 13:12): GPTQ-2 <= 1.5x MSB-3 -> continue (VQ / trellis + block-wise distillation, then an Eval verdict);
> 3x -> kill the 2-bit bet at this level. Prediction: GPTQ-2 g64 ~2-3x MSB-3 (Hessian weighting usually halves RTN's
output error; RTN-2 ~4-5x).

## Q2 RESULT (data/q2/q2.txt; CPU 75 s under a share lock, 15:06): 2-bit per-expert codes KILLED at the GPTQ level
Output error tr(dW H dW^T) / tr(W4 H W4^T) on 8,336 captured MoE inputs per layer (layers 2 / 12 / 24 / 36 / 46, Q1's 40
experts; all tokens, not only routed ones), gate / up / down, mean vs MSB-3:
| format (bits incl. bf16 scale + bias) | gate | up | down | vs MSB-3 |
|---|---|---|---|---|
| MSB-3 (3.5, B12's base today) | 0.0165 | 0.0219 | 0.0260 | 1.00x |
| RTN-2 g64 (2.5) | 0.1295 | 0.1715 | 0.2029 | 7.81x |
| GPTQ-2 g64 (2.5) | 0.0452 | 0.0597 | 0.1432 | 3.65x |
| GPTQ-2 g128 (2.25) | 0.0589 | 0.0775 | 0.1812 | 4.69x |
| GPTQ-3 g64 (3.5) | 0.0073 | 0.0096 | 0.0248 | 0.61x |
- Decision (pre-registered: <= 1.5x go on, > 3x kill): GPTQ-2 3.65x -> KILLED at this level; down is the worst (5.5x).
  Activation-aware rounding halves RTN-2's error (7.81 -> 3.65x) but 2 bits stay far from 3. VQ / trellis typically gains
  30-40 % over scalar GPTQ at 2 bits (~2.3-2.6x): still > 1.5x; only distillation could close it (training, uncertain).
- Side finding: GPTQ-3 g64 = 0.61x MSB-3 at the same 3.5 bits (fitting ~6x error per bit, ~3.2 b would match MSB-3: -8 %
  bytes at equal error: a quality option for B12's base, not a bytes breakthrough). Prediction (2-3x): MISS (worse).
- Process note: the first run (14:11) sat under a 3 h quiet lock (my --queue 180m set the lock length) and was stopped when
  the orchestrator cut it at 14:24; CPU-only work now runs under 'peer lock share --gpu 0'.

## U2 VERDICT (Eval suite 15:05; ~/EvalSuite/results/report_b12llu2*.txt): 4/5, fails on truncations alone
vs 4-bit ref 4/5 and vs B12LL 4/5: truncations 16 (ref 13, B12LL 14, rule allows 14); task scores = B12LL's (math 86.0 %,
50/50 answers equal; IFEval 87.0 %, 2 items flip each way; HumanEval+ 91.7 %, 60/60 equal). The 2 extra capped rows are
real loops (ifeval/2275, ifeval/3326). PR1 moves tokens like a summation-order change (80/210 rows token-identical to
B12LL; floor vs ref 59/210). 8-row B12LL replay on today's bt_b12 = 8/8 identical (the diffs are PR1's). Adoption: Kamen.

## Q3 RESULT (data/q2/q3.txt; CPU 9 s, share lock ~15:15): KLT + water-filling KILLED for line 2 (MSB-3 parity at ~3.1 b)
Gate / up in the layer's H eigenbasis, Leap's kernel layout (64-column groups sorted by eigenvalue, one bit width per group,
bf16 scale + bias per row-group, 0-bit tail groups dropped), budgets incl. overhead:
| budget | gate | up | vs MSB-3 (3.5 b: 0.0165 / 0.0219) | coded columns of 2560 |
|---|---|---|---|---|
| 1.5 b | 0.2157 | 0.2891 | 13.14x | 1789 / 1802 |
| 2.0 b | 0.1271 | 0.1695 | 7.72x | 2211 / 2214 |
| 2.5 b | 0.0651 | 0.0864 | 3.95x | 2485 / 2485 |
| 3.0 b | 0.0252 | 0.0332 | 1.52x | 2560 / 2560 |
- Decision (with Leap: MSB-3's error at <= 2 b -> continue): 7.7x at 2 b -> KILLED. Per-group RTN at 1-2 bits wastes bits
  (min-max levels); a greedy operational allocation would gain some, not 7x.
- Synthesis Q1-Q3 + Leap's act16: every code reaches MSB-3's output error at ~3.1-3.2 b (GPTQ-3 0.61x at 3.5 b, KLT 1.52x
  at 3.0 b, top-r 4-bit 1.75x at 2.7 b), i.e. ~10-15 % fewer bytes at equal quality: the Gaussian-like 4-bit experts sit near
  the rate-distortion bound. At 12 GB the SSD term cannot halve this way. For line 1 it matters: at ~3.0 b the 125B's experts
  are ~48 GB (+ ~3 GB dense) -> the whole model RAM-resident on the 64 GB Mac (MSB-3 + LSB would be ~59 GB), no SSD.

## TK1 pre-registration (before its lock): 12 GB speed side of L1-17 (fewer experts per verify row; NON-EXACT)
Leap's L1-17 on Fit12's captures (MoE-block error vs MSB-3 on all 10): top-8 0.78x at -16 % union, adaptive p0.8 1.43x at
-24 %, p0.9 0.38x at -10 %. scripts/fit_topk.py (F12_TOPK / F12_TOPP; verify rows only, kept ranks waited, no substitutes,
renormalized over the kept set; prefill / one-row / MTP head keep top-10). Eval hook sent 15:40: fit_hook.load_b12ll_budget
("", topk=8) = b12llk8, ("", topp=0.8) = b12llka80, on U1 (B12LL). Arms (~/Fit12/data/tk1; v3k + JOINT + SB1, U1 base, 6
prompts code / prose / math / write / lru / csv, 128 tokens): tk_ref a/b, tk_k10 (no-op: shas must = tk_ref), tk_k8 a/b,
tk_p8 a/b, tk_p9 a/b. Tokens change (non-exact), so per-prompt speed is confounded by the text: report the 6-prompt geomean.
Prediction: k8 +12-18 %, p0.8 +18-25 %, p0.9 +6-10 % tok/s at 12 GB (SSD bytes bind). Quality: the Eval suite's verdict.

## RES1 PLAN (orchestrator 15:33): a fully resident 125B at ~3 b, MSB-3 rows from the existing store first (no new disk)
Sizes (records3 index: base part 2,162,688 B incl. s3 / b3 = 2.15 MB, LSB part 0.77 MB per expert; 48 x 512 trunk experts +
the MTP layer's 512): trunk MSB-3 = 52.8 GB, MTP head MSB-3 = 1.1 GB, dense + runtime ~4.5 GB (B12's measured footprint
minus its pools: 10.4-11.3 GB - 6.0 GB pools), load transient ~1 GB.
- Step A, all experts resident at MSB-3 (no LSB: ranks 0-1 lose their 4-bit -> NON-EXACT vs B12LL, needs a verdict):
  peak ~58-59 GB > the 45 GB rule by ~14 GB. With the LSB planes (exact B12LL math) +18.9 GB = ~78 GB: impossible on 64 GB.
  -> not runnable under the rule; only with Kamen's explicit OK to exceed 45 GB (and Chrome etc. closed).
- Step A' under the rule (recommended first): MSB-3 pools at ~39 GB = C_m ~378 of 512 per layer (74 %), LSB sidecar for
  ranks 0-1 streamed as today (exact B12LL math, U1/U2 tokens), loader as today. Peak ~45 GB. Expected misses ~5 % of the
  ~1,370 unique experts per 4-row pass (~70 x 2.15 MB = 0.15 GB, ~12 ms) + GPU ~30 ms + draft ~4 + edges ~9 -> ~55 ms per
  pass: code ~60 tok/s at 3:3:0 (3.37 tok/pass), ~55 with tau 0.7; prose ~45. Runs on the existing store: zero new disk.
  Guards: the batch framework's swap guard (abort if swap grows > 2 GB over the batch start), the 56 GB anon+wired+compressor
  gate, peak footprint printed per arm; first run C_m 192 (24 GB, known) then 288, then 378.
- Toward 350 (line 1): all-resident needs <= ~2.6 b / weight for the experts to fit 45 GB (every code measured is >= 1.75x
  MSB-3's error there: Q2 / Q3 / act16); with top-8 (L1-17, -16 % union) and DFlash 16-row verify (~9 tok / pass), a pass reads
  ~62 x 48 x 2.15 MB = 6.4 GB from RAM (~23 ms at 280 GB/s) + GPU kernels -> ~150-200 tok/s code at MSB-3 bytes; 350 needs
  ~2x fewer bytes per pass on top (codes near ~2.5 b at acceptable quality, or a bigger cut in experts per token).
- Disk for a GPTQ-3 store (0.61x MSB-3's error at the same 3.5 b): ~53.9 GB (trunk + MTP head); a ~3.0 b store ~46 GB.
  Free 124 GB vs the 110 GB floor -> either needs ~40-55 GB freed first: Kamen's call.

## TK1 RESULT (~/Fit12/data/tk1, lock 15:36-15:47): top-8 = +17.4 % at 12 GB with acceptance intact; p0.8 hurts acceptance
- No-op tk_k10 = tk_ref shas on all 6 prompts (code / prose / math / write / lru / csv): the patch is inert at k = 10.
- 6-prompt geomean tok/s (a / b): ref 18.75 / 18.90; top-8 22.12 / 22.11 (+17.4 %); adaptive p0.8 21.63 / 21.70 (+15.1 %);
  p0.9 20.05 / 19.36 (+4.3 %, noisier). top-8 verify -14..-28 ms per pass, almost all of it loader stall.
- Acceptance (tok / pass): top-8 keeps it (lru 3.46 vs 3.46, csv 3.66 vs 3.56, prose 2.37 vs 2.17); p0.8 collapses on code
  (lru 2.06, csv 1.29): the trunk moves far enough that the unchanged (top-10) MTP head drafts it badly -> a quality red flag,
  consistent with Leap's MoE-block errors (top-8 0.78x MSB-3, p0.8 1.43x). Prediction (k8 +12-18 %, p0.8 +18-25 %, p0.9
  +6-10 %): k8 HIT, p0.8 MISS (lower; acceptance), p0.9 MISS (lower). Tokens change in all three (non-exact): the Eval suite's
  b12llk8 / b12llka80 verdicts decide. If top-8 passes: it stacks with U2 + tau 0.7 (exact w.r.t. U2) and C_m 59.

## TK2 pre-registration (before its lock): the stack top-8 + U2 (PR1) + tau 0.7 (+ C_m 59), 6 prompts (NON-EXACT vs U1)
Arms (~/Fit12/data/tk2): t2_ref a/b (U1 B12LL), t2_k8 (PR1 + top-8, 3:3:0), t2_k8t7 a/b (+ tau 0.7), t2_k8t7c59 a/b
(+ C_m 59 / C_l 20). Check: t2_k8t7 shas = t2_k8 shas (PR1's policy invariance must survive top-8). Prediction: k8t7 ~+25 %
geomean vs U1, k8t7c59 ~+30 % (peak footprint <= 12.2 GB as FS1). Numbers for Kamen once b12llk8's verdict is in.
Queued behind the Eval suite's L1-17 verdicts (priority) and Leap's L1-18.

## TK2 RESULT (~/Fit12/data/tk2, lock 15:51-15:59): PR1 invariance holds under top-8; stack = +19.8 % (C_m 51, 11.72 GB)
- Shas: t2_k8 (3:3:0) = t2_k8t7 (tau 0.7) = t2_k8t7c59 on all 6 prompts (ea37e2 / a83ac5 / 8a62a3 / 0eb6a3 / bb3e36 /
  2e0e87): with PR1, draft policy and cache size do not move top-8's tokens either. t2_ref = U1 on all 6.
- 6-prompt geomean tok/s: U1 18.80 / 18.85; PR1 + top-8 21.41 (+13.7 %); + tau 0.7 22.60 / 22.50 (+19.8 %); + C_m 59 23.80 (a;
  b had 2 disturbed items: code 21.0, write 19.1) = +26.4 %. Per prompt (k8 + tau 0.7, C_m 51): code 22.5 (-0.9 %), prose 20.15
  (+45 %), math 20.85 (+17 %), write 20.4 (+32 %), lru 25.65 (+19 %), csv 26.6 (+12 %) (text-dependent: tokens differ).
- Peak footprint (harness holds all 6 prompt states): C_m 51 arms 11.71-11.72 GB; C_m 59 arms 12.55-12.56 GB (> 12 GB).
- Prediction (+25 % / +30 %): MISS (lower: +19.8 / +26.4 %). Adoption needs b12llk8's verdict (non-exact) + Kamen.

## RES1 pre-registration (before its lock): Step A' (C_m 192 / 288 / 378) then Step A (C_m 512), exact w.r.t. U2
Kamen (16:10 via the orchestrator) approved Step A once, watched. Design change vs the RES1 PLAN: Step A keeps B12LL's math
EXACT (C_m 512 = every MSB-3 base part resident, 52.8 GB; ranks 0-1's LSB parts stream as today with C_l 48: ~59 GB peak)
instead of dropping ranks 0-1 to 3 bits (that would need a B12 block patch and a verdict). All arms: v3k + JOINT + SB1 + PR1
+ tau 0.7, A4 twice per process (round 2 = warm pools). scripts/bigres.sh: free-memory gate per arm, swap abort +2 GB,
BIGTEST PID file for Step A (orchestrator's watcher: app freeze, swap / pressure / free-memory SIGTERM). Shas must = U2 in
every arm. Prediction (round 2, code / prose): c192 ~34 / 24, c288 ~45 / 32, c378 ~55 / 40, c512 ~60-65 / 45 tok/s; peak
footprint ~25 / 35 / 45 / 59 GB; swap delta < 1 GB except possibly c512.

## RES1 RESULT (~/Fit12/data/res1, lock 16:52-16:56): Step A' exact at 2x the 12 GB speed; Step A aborted by the watcher
All 24 Step A' prompt runs = U2 shas (exact). Round 2 (warm pools), tok/s code / prose / math / write (lifetime peak):
- C_m 192 / C_l 20: 35.8 / 28.8 / 33.6 / 29.1 (25.8 GB); verify 63-82 ms, stall 31-49 ms, 155-283 reads per pass.
- C_m 288 / C_l 24: 39.2 / 31.1 / 38.5 / 32.6 (35.9 GB); stall 23-37 ms.
- C_m 378 / C_l 32: 46.2 / 34.7 / 45.4 / 36.1 (45.5 GB); verify 49-58 ms, stall 17-25 ms, 76-135 reads per pass. vs C_m 51
  (U2 + tau 0.7: 22.95 / 17.25 / 18.6 / 19.7): 2.0x / 2.0x / 2.4x / 1.8x. Swap +0 throughout.
- Prediction (code / prose: c192 ~34 / 24, c288 ~45 / 32, c378 ~55 / 40): c192 HIT (above), c288 / c378 MISS (lower): at
  C_m 378 the 4 prompts still evict base parts (they touch > 378 experts per layer) and C_l 32 leaves LSB misses for ranks 0-1.
- Step A (C_m 512 / C_l 48): BIGTEST protocol worked; the watcher froze 45 app processes and SIGTERMed the test at 16:55:35
  (free memory 32-124 MB for its window while MLX allocated the 52.8 GB of pools; test RSS 0.8-1.4 GB: the pools are Metal
  buffers), thawed at 16:55:40; swap +0. With other sessions / apps at 17-24 GB, ~57 GB (pools + dense) cannot fit in 64 GB:
  it needs ~10 GB released elsewhere (Kamen's call) or a smaller set (C_m 448 = 87.5 %, pools 46.2 GB, ~51 GB total).

## RES2 pre-registration (before its lock): C_m 448 (87.5 %), then C_m 512 if 448 keeps > 6 GB available (Kamen 17:2x)
scripts/bigres2.sh: each arm a fresh process with the BIGTEST protocol (orchestrator's watcher, re-armed 17:21: abort if
available = free + inactive + speculative < 1 GB for 30 s, swap +2 GB, or critical pressure); my gate = available >= 34 GB
at start; my swap abort +2 GB; the minimum available memory per arm is logged; c512 runs only if c448 finished with
min available > 6 GB. Same exact config as RES1 (v3k + JOINT + SB1 + PR1 + tau 0.7, C_l 48, A4 twice): shas must = U2.
Prediction (round 2, code / prose / math / write): c448 ~50 / 37 / 48 / 39 tok/s at ~52 GB peak; c512 ~52 / 38 / 50 / 40 at
~58 GB, if it runs.

## RES2 RESULT (~/Fit12/data/res2, lock 17:49-18:02): C_m 448 collapses under compression (0.3-0.7 tok/s); c512 skipped
- c448 / C_l 48: after load 51.69 GB (MLX active 51.34). Shas = U2 on the 2 prompts that finished (code fb1e61e413, prose
  42eef85996): exact, but code 0.7 tok/s (verify 4,349 ms per pass), prose 0.3 (6,572 ms). macOS compressed the model's own
  pools: compressor 6.4 -> 29.5 GB while available memory stayed 11-18 GB (the watcher's < 1 GB rule never fired) and pressure
  went to 2; swap +1.94 GB at 18:02:22 -> my swap guard stopped it at 18:02:26 (+2 GB). 45 app processes frozen 17:49:16,
  thawed 18:02:27. c512 skipped by rule (c448 rc 127; min available 10.8 GB).
- Reading: with ~20 GB of other sessions / apps resident, ~45 GB (C_m 378: swap +0, 2x the 12 GB speed) is this Mac's
  practical ceiling today; beyond it the OS compresses the pools. Ways past it (Kamen's call): release ~10 GB elsewhere, or wire
  the pools (MLX wired limit) so macOS compresses others instead (risk: wired memory cannot be reclaimed).
- Prediction (c448 ~50 / 37 / 48 / 39 tok/s): MISS (compression).

## b12llk8 VERDICT (Eval suite 18:42; ~/EvalSuite/results/report_b12llk8*.txt): top-8 (U1-based) PASSES 5/5 and 5/5
vs the 4-bit ref: math 84.0 % (-2.0 [-5.3, +1.3]), IFEval 86.0 % (-4.0 [-7.2, -0.8]), HumanEval+ 91.7 % (+0.0), answer agreement
95.2 % (floor 97.1), truncations 14 vs 13 (the limit; B12LL also 14), loops 2 = 2. vs B12LL: math -2.0 (1 flip), IFEval -1.0
(4 / 3 flips), HumanEval+ 0, truncations 14 = 14; 38/210 rows token-identical. Harness speed 16.0 vs 13.3 tok/s core (+20 %).
Identity checks passed (topk=10 and topp=1.0 = B12LL, 6/6). Next at the Eval suite: b12llk10u2 identity, b12llk8u2 core (the
stack's verdict: PR1 + top-8), b12llka80. Adoption: Kamen.

## RES3 pre-registration (before its lock): C_m 448 with the pools WIRED (Kamen 18:2x via the orchestrator 18:33)
RES2 lost to macOS compressing the model's own pools. F12_WIRED_GB=49 (fit_b12.py _env wrapper: mx.set_wired_limit before
load_b12 allocates; pools 46.2 + 1.8 GB LSB + ~1 GB; under the default GPU wired cap 51.8 GB, no sudo), so macOS must compress /
swap the frozen apps instead. scripts/bigres3.sh: c448 only (no 512: its pools exceed the cap), BIGTEST protocol (watcher
pid 22544: swap +4 GB, compressor +12 GB, pressure warn 60 s / critical 10 s, available < 1 GB 30 s), my swap guard +4 GB,
wired / compressor / swap peaks logged every 5 s. Checks: wired memory must grow by ~46 GB during load (else the limit did
not take); shas = U2. Prediction (round 2): ~50 / 37 / 48 / 39 tok/s code / prose / math / write if the pools stay resident;
swap may grow 1-4 GB (the frozen apps' memory; Kamen accepts).

## RES3 RESULT (~/Fit12/data/res3, lock 19:13-19:15): wired C_m 448 = exact at 2.0-2.8x the 12 GB speed
mx.set_wired_limit(49 GB) took (previous 0); vm_stat wired 4.3 -> max 56.5 GB during the run; compressor max 6.6 GB (RES2:
29.5); swap 11,011 -> max 12,185 MB (+1.17 GB), 11,936 at the end; min available 2.8 GB; the watcher froze 46 apps 19:14:02,
thawed 19:14:52, no abort. After load 51.68 GB; lifetime peak 53.28 GB. All 8 prompt runs = U2 shas (exact).
Round 2 (warm), tok/s code / prose / math / write: 53.5 / 39.5 / 52.5 / 39.9 (verify 43-49 ms, stall 11-16 ms, 52-85 reads per
pass: LSB misses at C_l 48 + some base evictions); vs C_m 378 46.2 / 34.7 / 45.4 / 36.1 (+16 / +14 / +16 / +11 %); vs the
12 GB setup (U2 + tau 0.7, C_m 51) 2.33 / 2.29 / 2.82 / 2.03x. Prediction (~50 / 37 / 48 / 39): HIT. C_m 512 stays out (its pools
exceed the 51.8 GB wired cap; raising it needs Kamen's admin password and leaves macOS ~3 GB).

## B3 pre-registration (Sun 25 Oct 05:40; NewIdeas' bet B3): data-aware 2-bit TRELLIS codes vs MSB-3 output error (CPU)
Q2 killed GPTQ-2 g64 (3.65x MSB-3) and Q3 KLT + water-filling (7.7x at 2 b); a QTIP-class code was never tested here.
scripts/fit_b3.py (runner scripts/b3.sh, share lock, CPU only) on Q2's 40 experts + captures (data/q2): TR-k = Breakthrough's
D2 trellis (RHT + per-row RMS + bitshift trellis L = 12, exact Viterbi; encoder copied to scripts/fit_trellis_enc.c with a
worker cap), TRL-k = the same + BlockLDLQ (16-column blocks, 16 x 16 tiles), k = 2 / 3 (2.06 / 3.06 b incl. the initial
state + row scale); RD = ideal-Gaussian floor (reverse water-filling on H's eigenbasis) at 2.0 / 2.5 / 3.0 b.
Kill line (NewIdeas): TRL-2 > 1.5x MSB-3 (mean of the gate / up / down ratios, Q2's metric) -> B3 KILLED; <= 1.5x -> a
nested design + an Eval route. Sanity gates: TR-k on iid N(0,1) within 5 % of Breakthrough's 0.071 / 0.0191; TRL(H = I) ==
TR bitwise; MSB-3 recomputed = Q2's 0.0165 / 0.0219 / 0.0260. Prediction: TRL-2 ~2.0-2.6x (kill), TRL-3 ~0.4-0.6x, MSB-3
parity at ~2.5-2.7 b; the RD floor at 2.0 b below 1x (the gap is the practical code's).

## B4 pre-registration (Sun 25 Oct 05:55; NewIdeas' bet B4): expert read-path microbench (read-only on records3)
/Users/kamen/Fit12/b4/b4bench.swift (runner scripts/b4.sh, short quiet lock): calls of N = 1-24 cold MSB-3 base parts
(2,162,688 B; mincore-checked not cached, each read once) of one random layer, 4 paths interleaved over 50 reps:
A bounce = v3k's default (dispatch_apply, aligned F_NOCACHE pread into a 16 KB bounce, 9 memcpys into per-piece Metal
pools); B direct = one pread straight into a 16 KB-aligned Metal slot (the record image as the slot: what a kernel-native
store gives; the existing records are already 16 KB-aligned, only the pools + tier_glu addressing would change); C io1 =
MTLIOCommandQueue (concurrent, high priority), one IO command buffer with N loads; D ioN = N one-load IO command buffers.
Fit: median call ms = a + b N. Kill line (NewIdeas): the best of B / C / D has a > 0.7 x A's a (< 30 % less fixed cost per
call) -> B4 KILLED; a path that loses marginal GB/s must win it back at N ~ 11-12 (the engine's base call). Also logged: CPU
ms per call, MTLIO's page-cache side effect, content equality of every call's last slot vs the file. Prediction: B saves
only the memcpy tail (~0.03-0.06 ms, ~15-25 % of a) -> kill zone; C / D unknown (lower latency, or a higher per-buffer
cost); marginal GB/s ~equal (11-13). The engine's 0.255 ms per base call also holds GCD wake + gate pickup, outside this bench.

## TK3 pre-registration (Sun 25 Oct 06:05): the top-p NaN fix (Eval suite: b12llka80 = 42 / 210 rows flood with token 0)
Cause found by reading bt_b12's routing block (read-only): a verify row that keeps only rank 0 (p0.8's confident rows) no
longer waits for rank 1, so rank 1's LSB sidecar slot stays -1 and tier_glu reads outside the LSB pool for position 1 at
weight 0 (0 x Inf / NaN = NaN). The engine's own one-row path redirects a late rank 1 to rank 0's slot; fit_topk now does
the same for verify rows with k = 1 (TK3 line in fit_topk.py; a no-op for k >= 2; dry run: all 6 replacements apply,
the patched _install compiles). The harness now prints 'tok0 N' (count of token id 0) per item. Arms (data/tk3, TK1's
6 prompts, U1 base): tk3_k8 (shas must = TK1 tk_k8_a: no-op check), tk3_p8_a / b (fixed p0.8). Pass: tok0 = 0 on every
p8 item and code acceptance back near top-8's (lru >= 3.0, csv >= 3.0 tok / pass vs 2.06 / 1.29 before). Then the Eval
suite re-runs b12llka80 (non-exact; adoption is Kamen's).

## B3 RESULT, in-sample (data/b3/b3.txt; CPU 1073 s under a share lock 05:55-06:13): TRL-2 = 1.15x MSB-3 -> PASSES the line
| code (bits / weight) | gate | up | down | mean vs MSB-3 |
|---|---|---|---|---|
| MSB-3 (3.5, recomputed = Q2) | 0.0165 | 0.0219 | 0.0260 | 1.00x |
| TR-2 data-free (2.06) | 0.0480 | 0.0630 | 0.0742 | 2.88x |
| **TRL-2 BlockLDLQ (2.06)** | 0.0142 | 0.0189 | 0.0450 | **1.15x** |
| TR-3 data-free (3.06) | 0.0126 | 0.0169 | 0.0200 | 0.77x |
| TRL-3 BlockLDLQ (3.06) | 0.0037 | 0.0049 | 0.0119 | 0.30x |
| TRL-2 gate / up + TRL-3 down (2.39) | | | | 0.73x |
- vs Q2's GPTQ-2 g64 (2.5 b, 3.65x): the trellis gains ~2.7x over scalar codes (TR-2 2.88x vs RTN-2 7.81x) and LDLQ
  another ~2.5x (as GPTQ's 2.1x). MSB-3 parity at ~2.16 b (log-linear). Down is the hard projection (TRL-2 1.73x).
- Sanity: TR on iid N(0,1) = Breakthrough's 0.071 / 0.0191; TRL(H = I) == TR bitwise; MSB-3 = Q2's numbers.
- The RD floor rows were WRONG (bisection range too narrow; down's 'floor' sat above TRL-2): fixed, re-run queued.
- CAVEAT before any claim: in-sample (LDLQ fitted to the same 8,336 captured tokens it is judged on, N / K = 3.3 for
  gate / up). B3b pre-registration: fit on the first half of the tokens, judge on the second half (fit_b3.py oos): B3
  stays alive if the out-of-sample TRL-2 <= 1.5x MSB-3; the mixed 2.39 b design is strictly better per byte if <= 1.0x.
  Prediction: OOS TRL-2 1.3-1.6x (gate / up lose most), mixed 0.8-0.95x. Queued in the B4 / TK3 quiet window.

## B8 pre-registration (Sun 25 Oct 06:30; NewIdeas' bet B8, forecast-guided eviction; exact, a cache policy)
MoonShots' prior (FINDINGS item 3): Belady needs 8-16 passes of foresight; perfect 1-2-pass prediction got 7-13 %. B8's
forecast (drafted tokens + a same-layer router forecast) sees <= 1-2 passes ahead with recall 0.74-0.82, so the bound
for it is Belady with perfect foresight CAPPED at F passes. scripts/fit_b8.py: exact routing (all top-10 waited), per-layer
caches of C = 51 / 128 / 192, demand insertion, Breakthrough's expA 125B decode traces (4 prompts, 2,926 tokens), passes of
A = 3 (MTP-like) or 1 token; LRU vs OPT-1 / 2 / 4 / 8 / 16 / OPT. Kill line (NewIdeas): OPT-2 < 10 % fewer misses than
LRU at C 128 and 192 -> B8 KILLED (no real forecast beats perfect foresight). Prediction: OPT-1 -3..-6 %, OPT-2 -5..-9 %,
OPT -25..-40 % -> kill. Runs in the B4 / TK3 / B3b quiet window (CPU, ~2-3 min).

## B4 RESULT (data/b4/b4.txt; quiet lock 07:11, 1,400 calls in 5.3 s, every expert read once, all 24,576 cold): KILLED
| path | fixed a (ms / call) | marginal GB/s | median N = 1 / 12 / 24 (ms) | a vs today |
|---|---|---|---|---|
| A bounce (today: aligned pread -> bounce -> 9 memcpys) | 0.300 | 12.92 | 0.440 / 2.344 / 4.282 | 1.00x |
| B direct (pread straight into a 16 KB-aligned slot) | 0.240 | 12.48 | 0.387 / 2.339 / 4.366 | 0.80x |
| C MTLIOCommandQueue, 1 IO buffer per call | 0.369 | 12.52 | 0.513 / 2.479 / 4.468 | 1.23x |
| D MTLIOCommandQueue, 1 IO buffer per expert | 0.362 | 12.40 | 0.511 / 2.503 / 4.502 | 1.21x |
- Kill line (< 30 % less fixed cost): best = B at -20 % -> KILLED. B saves only the memcpy at small N and loses ~3 % of
  marginal bandwidth, so at the engine's ~11-12 misses per base call it equals today (2.339 vs 2.344 ms at N = 12).
- MTLIO is slower at every N (fixed +21-23 %), saves no CPU (6.0-7.0 vs 5.1 ms CPU at N = 24), and FILLS THE PAGE CACHE:
  all 3,370 experts it read were resident afterwards (7.3 GB of store pages; the F_NOCACHE paths left 0). Under a 12 GB
  budget that is a cost, not a gain. Content of every call's last slot = the file (all paths).
- Prediction (B ~-15..-25 %, kill zone; C / D unknown): HIT for B; MTLIO worse than expected. The ~19-22 ms of fixed
  read cost per pass is SSD / dispatch latency, not the bounce copy; no read-API lever left on this path.

## TK3 RESULT (data/tk3; quiet lock 07:08-07:11): the top-p fix works; top-8 unchanged
- tk3_k8 shas = TK1 tk_k8_a on all 6 prompts (the fix is a no-op for k >= 2). tk3_p8 a = b (deterministic).
- Fixed p0.8 vs TK1's p0.8: code / prose / math / write shas EQUAL (those never met a dropped rank 1 with an empty LSB
  slot); lru and csv CHANGE and recover: 2.06 -> 3.46 and 1.29 -> 3.46 tok / pass (= top-8's 3.46 / 3.66 range), 24.6 ->
  28.3 and 22.0 -> 29.2 tok/s; tok0 = 0 on every item. 6-prompt geomean 23.15 tok/s vs top-8 22.07 (+4.9 %) and U1 ref
  18.75-18.90 (TK1; ~+23 %). Non-exact: needs the Eval suite's re-run of b12llka80 (its hook imports fit_topk, so it
  picks the fix up as is).

## B10a pre-registration (Sun 25 Oct 07:30; NewIdeas' B10 part (a) + the v2 residual spec of 06:57)
Held-out everything: new capture (fit_wv.py, F12_CAP_TEXTS=A,B; layers 2 / 7 / 12 / 18 / 24 / 30 / 36 / 41 / 46; MoE inputs
+ each row's top-10 ids). Text A (FIT) = MoonShots FINDINGS.md + the 3 largest Swift files of ~/OjoX + Leap's qmm16.py (no
Eval-suite data); text B (JUDGE) = the 4-bit reference model's own prompts + completions (~/EvalSuite/cache/ref/tf_gen.jsonl
ids, round-robin over sets), ~4K tokens each, read-only. 4 new experts per layer (seed 2, none of Q2's 40). fit_b3.py b10a
(runner scripts/b10a.sh): LDLQ's H from A; errors on all of B's tokens AND on the tokens routed to each expert.
Numbers (NewIdeas): (1) TRL-2 base vs MSB-3 -> kill if > 1.5x (also gate/up-2 + down-3); (2) base + trellis residual (k_r = 1
/ 2, LDLQ) vs W4, scored against E4 = W4's own rounding error (s4^2 / 12 per weight, H-weighted; sanity: MSB-3 = ~3 E4) ->
kill if > 0.2 E4 (= total > 1.2x of 4-bit, independent errors; NewIdeas to confirm the reading); (3) encode s per expert.
Prediction: held-out TRL-2 1.2-1.6x on all tokens, worse on routed tokens; residual k_r = 2 ~0.15-0.3 E4 (at the line),
k_r = 1 > 0.5 E4; TRL-2 encode ~4-6 s per expert at 12 threads.

## B3b RESULT (data/b3/b3b_oos.txt; quiet window 07:11-07:18, CPU 392 s): OUT OF SAMPLE the 2-bit base FAILS the line
Rows are all unique (8,336 of 8,336: no token captured twice). LDLQ's H from the first 4,168 rows (N / K = 1.6 for gate /
up), judged on the other 4,168:
| code (bits) | gate | up | down | mean vs MSB-3 (test half) |
|---|---|---|---|---|
| MSB-3 (3.5) | 0.0167 | 0.0220 | 0.0261 | 1.00x |
| TR-2 data-free (2.06) | 0.0483 | 0.0632 | 0.0744 | 2.87x |
| TRL-2, judged on its own fit half (in-sample) | 0.0107 | 0.0142 | 0.0403 | 0.94x |
| **TRL-2, judged on the held-out half** | 0.0353 | 0.0469 | 0.0603 | **2.19x** |
| TRL-3 down, held-out | | | 0.0162 | 0.62x (down) |
| gate/up TRL-2 + down TRL-3 (2.39), held-out | | | | 1.62x |
- Pre-registered prediction (1.3-1.6x): MISS (worse). The in-sample 1.15x was mostly LDLQ fitting noise in a thin H: in-
  sample 0.94x at N = 4,168, 1.15x at 8,336; a 1/N model of both puts the true value near ~1.36x (estimate only).
- RD floor (bisection fixed; H = I check 0.0615 / 0.0308 / 0.0154 vs theory 0.0625 / 0.0312 / 0.0156): in-sample ideal
  Gaussian code at 2.0 / 2.5 / 3.0 b = 0.90x / 0.45x / 0.23x MSB-3 (row x col 0.89 / 0.45 / 0.22): even the ideal 2-bit
  code has only ~10 % headroom under MSB-3; in-sample TRL-2 sits at 1.21-1.34x of the floor.
- Decision: B3 as tested is KILLED out of sample (2.19x > 1.5x). Open: whether a LARGE calibration set brings TRL-2 to
  <= 1.5x held-out -> B10a (revised) measures it directly. Correction sent to NewIdeas 07:40.

## B8 first run (data/b8/b8.txt, 07:18): OPT-F rows INVALID (float64 key bug), unbounded OPT valid
Unknown-horizon eviction keys were 1e18 - last_use in float64 (spacing 128 there), so LRU order collapsed into ties and
OPT-1 / 2 came out worse than LRU (+10..+22 %). Fixed (1e15). Valid: unbounded Belady vs LRU at A = 3: C 51 -33.4 %, C 128
-43.7 %, C 192 -37.0 % (A = 1: -39.7 / -45.5 / -37.9 %); LRU misses / token 147 / 71.4 / 42.8 (A = 3). Re-run queued.

## B10a pre-registration REVISED (Sun 25 Oct 07:50, after B3b): large calibration, held-out text, routed tokens
The fit set becomes 8 different ~4K-token local documents (A0-A7: MoonShots / NewIdeas / Breakthrough / Leap notes, OjoX Swift,
Leap + MoonShots Python; ~32K tokens, N / K ~12.8 for gate / up, ~51 for down; stored fp16), with a trend arm fitted on the
first ~8K tokens only. Judge text and everything else as at 07:30 (text B = the reference model's own prompts + completions;
all tokens + each expert's routed tokens; 4 held-out experts x 9 layers; base TRL-2, gate/up-2 + down-3, base + residual k_r =
1 / 2 vs E4, encode seconds). Kill lines unchanged: held-out TRL-2 > 1.5x MSB-3 (routed and all tokens) -> B10 killed at
2.06 b; residual > 0.2 E4 -> the v2 rank 0-1 code fails at that k_r. Prediction (after B3b): TRL-2 held-out ~1.4-1.7x on
all tokens, worse on routed; the 8K-fit arm worse than the 32K one by ~0.2-0.4x. B8's fixed replay runs after it.

B10a run change (09:25, before any result): RULES.md's new 4-core cap -> analysis at B3_THREADS = 4 / BLAS 4 (~30 min), one
45 m quiet lock (capture + analysis + B8), the analysis bounded by the lock's end with per-layer partial results
(data/b10/b10a_partial.txt). Judge text B is 4,096 tokens (~80 routed rows per expert): experts with < 8 routed rows skip the
routed test (count printed). Arms, experts (seed 2), layers and kill lines unchanged.

## B10a RESULT (Sun 25 Oct 10:29; data/b10/b10a.txt, b10_cap.txt): KILLED. B8 RESULT (data/b8/b8_fixed.txt): KILLED
Ran: capture 09:59-10:03 (9 layers; fit A0..A7 = 31,217 unique rows, trend fit A0..A1 = 6,647; judge B = 4,095 rows), analysis
10:03-10:29 (36 held-out experts, seed 2; B3_THREADS 4), B8 10:29 (25 s). All rc 0. Sanity: MSB-3 / E4 = 2.97 / 2.97 (expect ~3);
MSB-3 errors 0.0173 / 0.0221 / 0.0261 (Q2: 0.0165 / 0.0219 / 0.0260).
| code (judged on held-out text B)          | all tokens      | routed tokens (30 of 36 experts, ~75 rows) | line          |
| TRL-2 base (2.06 b), fit on 31K tokens     | 2.49x MSB-3     | 2.59x                                      | > 1.5x: KILL  |
| TRL-2 base, fit on 6.6K tokens (trend)     | 3.03x           | 3.13x                                      |               |
| 1/N extrapolation to unlimited fit text    | ~2.35x (est.)   | ~2.44x (est.)                              |               |
| gate/up TRL-2 + down TRL-3 (2.40 b)        | 1.87x           | 1.91x                                      | > 1.5x        |
| base + residual k_r = 1 (~3.1 b), vs W4    | 2.43 E4         | 2.47 E4                                    | > 0.2 E4      |
| base + residual k_r = 2 (~4.1 b), vs W4    | 0.66 E4         | 0.67 E4                                    | > 0.2 (and > NewIdeas' soft 0.5) |
| TRL-3 down only (3.07 b)                   | 0.685x MSB-3    | 0.749x                                     | side finding  |
Encode (4 threads, quiet mode): TRL-2 3.33 s / matrix = 10.0 s / expert; residual k=2 10.0 s, k=1 8.3 s per expert.
Reading: B3's 1.15x was overfit; with 7x more fit text the held-out base is 2.49x and its 1/N asymptote ~2.35x, so more
calibration cannot reach 1.5x. Side finding: a 3-bit trellis beats MSB-3 held-out on down (0.69x error at 3.07 vs 3.5 b,
-12 % bytes); gate/up TRL-3 held-out not measured (B3b had only down).
B8 (fixed replay, 1e15 key): A = 3: OPT-2 vs LRU C 51 -15.8 % (steady -16.2), C 128 -8.2 % (-9.2), C 192 -4.7 % (-6.0); OPT-1
-9.1 / -4.5 / -2.6 %. A = 1: OPT-2 -6.8 / -3.3 / -1.9 %. Unbounded OPT -33.4 / -43.7 / -37.0 % (= the first run). Kill line
(OPT-2 < 10 % at C 128 and 192): KILLED. At C 51 (12 GB line) even perfect 2-pass foresight gives -16 %; a real forecast less.
LESSON (BigMoE 10:03, RULES.md 'CPU jobs vs the GPU queue'): I held the quiet lock 26 min through the CPU-only analysis while
MoonShots and Leap waited. From now on: quiet lock for the GPU part only, unlock, CPU part when no one waits (or < 80 % CPU).

## B10-r pre-registration (Sun 25 Oct 16:40; NewIdeas' final 2-bit test, ~/NewIdeas/QUEUE.md row B10-r; asked 10:34 / 13:17 / 16:35)
Hypothesis: B10a's held-out failure (TRL-2 2.49x MSB-3) was domain shift (fit on 8 technical documents, judged on eval text).
- Calibration (/Users/kamen/Fit12/scripts/fit_b10r_text.py -> data/b10r/texts): R0..R7 FIT (32,768 tokens), R8..R9 VALIDATION
  (8,192), each part a round-robin of 7 domains: math (GSM8K 0-599 + MATH-500 non-picks, with solutions), code (HumanEval+
  non-picks, canonical solutions), instructions (IFEval non-picks), multilingual (MGSM de/es/fr/ja/ru/sw/zh, ids dec_mgsm
  never uses), Bulgarian wiki (after the suite's 65,536 tokens), MMLU-Pro non-picks, Kamen's own Claude Code sessions after
  route4 was built (validation uses other sessions). The suite's picks are reproduced and asserted against its item files;
  0 of 4,065 32-grams per part also occur in judge B.
- Judges: B = B10a's capture (data/b10cap/B: all + routed tokens); U = English literary prose (Pride and Prejudice tokens
  70,000-74,095, absent from the calibration; the suite's tf_doc uses 0-65,535), all tokens (+ routed, information only).
- Same 36 held-out experts / 9 layers as B10a. TRL-2 with D2's LUT (as B10a) at H damping 0.01 / 0.1 / 0.3 x mean diag;
  d* = the damping with the lowest mean validation error (never chosen on B or U).
- KILL (QUEUE.md): TRL-2 at d* > 1.5x MSB-3 on B (all OR routed), or > 1.8x on U -> the data-aware 2-bit codes route CLOSES.
- Same batch, information for 'TRL-3 replaces MSB-3': TRL-3 and TRL-2 with Leap's 3INST table (QTIP's computed code,
  tabulated as the kernel computes it), each matrix at its validation-best damping; gate / up / down vs MSB-3 on B and U.
- Run: quiet lock for the ~4 min capture only (SSD <= 66 C before, stop at >= 73 C), then unlock; the CPU analysis runs
  without a lock at 4 cores and steps aside between experts while anyone holds or waits for quiet (RULES.md). Code check
  16:35: smoke run on stand-in captures (1 expert) end to end, 55 s.

## Item (a) RESULT (16:45): NewIdeas' v3 replay CONFIRMED (own code: /Users/kamen/Fit12/scripts/fit_v3replay.py, from the engine's
ensure / make_room / lfu_victim; data/b10r/v3replay.txt, v3replay_chunks.txt, v3replay_ram.txt). Warm from token 0, counted
from 400 (their protocol: cold at 400 gives 157-174, so not theirs), MB = 1e6, engine read sizes (MSB 2,162,688, LSB 770,048 B).
| 24 GB C_m 147 C_l 46            | mine  | NewIdeas | diff  |
| per token, LRU                  | 143.5 | 144.5    | -0.7 % |
| per token, LFU 0.99             | 127.3 | 125.8    | +1.2 % |
| 16-row / accept 10, LFU 0.99    | 115.9 | 116.0    | -0.1 % |
| 16-row / accept 10, LRU         | 132.0 | 144.3    | -8.5 % (the one miss; per-row calls give 141.0, LFU then 156.7) |
RAM curve (16-row, LFU): 28 GB 84.7 vs 84.3, 32 GB 62.2 vs 61.8, 36 GB 46.7 vs 46.5, 40 GB 35.4 vs 35.1, 45 GB 26.9 vs 26.6
(all within +1.0 %). Both replays leave out prefetch spare slots (C_main = C - spare) and in-flight reads.

## B10-r RESULT (Sun 25 Oct 17:11; data/b10r/b10r.txt): KILLED on all three lines -> the data-aware 2-bit codes route is CLOSED
Ran: capture 16:37-16:39 (99/99 files, SSD max 58 C), analysis 16:39-17:11 (36 experts, rc 0, no step-asides). Checks: MSB-3 /
E4 2.96-2.97 on every judge; MSB-3 errors on B = B10a's exactly (same experts, same capture); 3INST table mean / RMS -0.0002.
Validation (R8..R9): TRL-2 1.775x / 1.750x / 1.743x MSB-3 at damping 0.01 / 0.1 / 0.3 -> d* = 0.3 (edge of the grid, curve flat).
| TRL-2 (2.06 b) at d* 0.3 vs MSB-3 | B all | B routed | U all (unseen prose) | validation |
| decisive (D2 LUT)                 | 1.854x (line 1.5: KILL) | 1.957x (KILL) | 2.287x (line 1.8: KILL) | 1.743x |
| same, 3INST code (decodable)      | 1.849x | 1.954x | 2.287x | 1.736x |
| B10a, technical-notes fit (d 0.01)| 2.49x  | 2.59x  | -      | -      |
Reading: domain shift was real (B 2.49x -> 1.85x with the broad mix), but even in-distribution held-out text (validation) is
1.74x, so better calibration cannot reach 1.5x. A rough 1/N fit of B3b (2.19x at 4.2K rows) and validation (1.74x at 32.8K)
puts unlimited in-distribution calibration at ~1.67x (estimate only).
TRL-3 with 3INST (~3.07 b, decodable), each matrix at its validation-best damping, vs MSB-3 (3.5 b):
| judge      | gate   | up     | down   | mean   |
| validation | -      | -      | -      | 0.468x |
| B all      | 0.445x | 0.451x | 0.601x | 0.499x |
| B routed   | 0.456x | 0.461x | 0.653x | 0.524x |
| U all      | 0.586x | 0.596x | 0.676x | 0.620x |
| U routed   | 0.594x | 0.603x | 0.776x | 0.658x |
-> 'TRL-3 replaces MSB-3' is fully measured: about half MSB-3's error at ~13 % fewer bytes per base part, holding on the unseen
domain (0.62x). Mixed gate/up TRL-2 + down TRL-3 (2.40 b): B 1.31x / routed 1.37x / U 1.69x (information only).
Encode at 4 threads: TRL-3 3INST 2.71 s per matrix (8.1 s per expert -> ~55 h for all 24,576 experts on 4 cores: needs
the GPU encoder); TRL-2 3.2-3.3 s. Coupling to note: v3's LSB sidecar refines MSB-3's grid, so with a TRL-3 base the
rank-0-1 (4-bit) path needs W4 from elsewhere (a full 4-bit record, or a residual on the trellis base). Non-exact format: Kamen's call.

## T3R pre-registration (Sun 25 Oct 20:05; NewIdeas 19:58, QUEUE row T3R): TRL-3 3INST base for EVERY expert + a 1-bit
trellis residual (R1) for ranks 0-1 in place of v3's LSB sidecar. Non-exact opt-in mode: adopting it is Kamen's call.
Records per expert (16 KiB-padded like the engine's): base = 3 b/w + 12-bit initial state per 256-weight tile + bf16 row
scale = 1,879,680 B -> 1,884,160 (0.871x MSB 2,162,688); R1 = 1 b/w + 12/256 + bf16 row scale = 650,880 B -> 655,360
(0.851x LSB 770,048). Ranks 0-1 = base + R1 = 4.13 b/w padded (v3 MSB + LSB = W4: 4.77).
- K1 bytes (/Users/kamen/Fit12/scripts/fit_t3r_k1.py, 1 core): item (a)'s replay (engine LFU 0.99, 16-row checks accepting
  10, expA traces, warm from 0, counted from 400). Budgets per layer = v3 slots x records: 12 GB B12LL C_m 51 + C_l 20
  (125.7 MB, DECISIVE), 12 GB C_m 59 + C_l 20 (side), 24 GB 147 + 46 (353.3 MB). T3R: best C_r on a grid (C_l/4 .. 3 C_l),
  C_b = the rest / base record. Side: v3 re-split in the same budget, unpadded records, per-token protocols; checks:
  item (a)'s 115.9 and NewIdeas' b10_12gb.txt v3 rows. KILL: T3R < 20 % fewer bytes / token than v3 (51 / 20) at 12 GB.
  Prediction: -17..-21 % at 12 GB (at ~10 % residency +16 % slots buys only ~5-7 % fewer misses; the rest is the
  0.87 / 0.85 record sizes), so a near-line call; 24 GB -20..-26 %.
- K2 quality (/Users/kamen/Fit12/scripts/fit_t3r_k2.py, <= 2 threads: fanwatch down since 13:56): B10-r's 36 experts and
  captures. Per matrix: TRL-3 3INST at damping 0.01 / 0.1 / 0.3, the validation-best (R8..R9, the base's own error) kept;
  R1 = BlockLDLQ k = 1, 3INST table, own bf16 row scale, on r0 = W4' - base, at 0.01 / 0.1 / 0.3, validation-best kept.
  Judged held-out on B all, B routed, U all, U routed in E4 units as B10a (per matrix err / E4, mean over experts per
  projection, mean of the three). Side: R1 with D2's LUT at the same damping; everything vs MSB-3. KILL: base + R1 > 1.0
  E4 on B routed -> no v3-quality claim (report the level). Prediction: TRL-3 alone ~1.5 E4 (0.52x x 2.96); a 1-bit
  trellis keeps ~0.3 of it (B10a's k_r = 2 kept ~0.09) -> ~0.45-0.6 E4 -> pass.
- Reading the K2 line: errors are vs W4, where v3's ranks 0-1 are exact. Against the (unseen) bf16 weights both carry W4's
  own ~1 E4 and T3R adds its error on top (independent given W4), so "< 1 E4" means "within one W4 rounding error of v3",
  not "better than W4". By the same arithmetic vs bf16: ranks 2-9 v3 ~4 E4 vs T3R ~2.5 E4.
- The captures' top-10 ids are unsorted (argpartition), so no rank-0-1-only judge: B routed = rows routed at any rank.

## T3R K1 RESULT (20:13; K1 ran 20:05-20:09; /Users/kamen/Fit12/data/t3r/k1.txt + k1_side.txt): KILL at 12 GB on the pre-registered protocol
Checks: v3 24 GB = item (a)'s 115.9; NewIdeas' b10_12gb.txt v3 rows: LFU +0.2 / +0.8 %, LRU -1.5 / -5.5 % (their LRU bug,
which they found: a check's misses evicting the same check's needs).
| 16-row / accept 10, LFU 0.99 (MB/token) | v3 as configured | T3R, best split           | T3R cut               | v3 best split |
| 12 GB B12LL 51 / 20 (DECISIVE)          | 256.4            | 214.3 (C_b 46 / C_r 59)   | -16.4 % -> KILL (20 %) | 248.8 (36 / 60) |
| 12 GB 59 / 20                           | 250.6            | 208.1 (C_b 64 / C_r 34)   | -17.0 %               | 246.6 (45 / 59) |
| 24 GB 147 / 46                          | 115.9            | 82.0 (C_b 175 / C_r 36)   | -29.2 %               | 115.4 (150 / 37) |
Side, 12 GB 51 / 20 (C_r >= one check's rank-0-1 need): per token LRU -20.8 %, LFU -21.5 %; 4-row / accept 3 (MTP-like;
B12LL's engine runs LRU) LRU -20.6 %, LFU -21.4 %; 16-row LRU -16.3 %. Unpadded records: +0.3..0.5 points.
Why: at 12 GB one 16-row check's top-10 union (~100+ experts / layer) exceeds the 46-75 base slots, so more slots buy
almost nothing and the cut is the record sizes (0.871x / 0.851x); with fewer rows per check the cache works and T3R's
+16-25 % slots add ~5 points. At 24 GB the slots pay (base misses 47.2 -> 36.0 / token), as NewIdeas predicted (~85).
Replay note: a tier asked for more than its slots in one check keeps the overflow (like the engine's union scratch) until
the next check, so a C_r below one check's need drifts to C_r = 1 (unbudgeted RAM); the side numbers keep C_r >= that need
(fit_t3r_k1's own grid C_l/4..3 C_l found C_r 5 at the 59 / 20 budget: 203.9, -18.6 %, same artifact).
K2 runs anyway as information (cheap, CPU) unless NewIdeas stops it.
- K2 smoke (20:13-20:15, layer 2 expert 57, 113 s): end to end OK; MSB-3 / E4 2.99-3.02; base + R1 0.43 E4 (B all) / 0.51
  (B routed); TRL-3 base alone 1.37 / 1.63 E4; D2-LUT residual = 3INST within 0.005 E4. Encode at 2 threads: TRL-3 5.2 s,
  R1 5.0 s per matrix -> the full grid would take ~65 min and run into the 21:41 GPU-queue wakes. DEVIATION (before the
  full run): damping grid trimmed to 0.1 / 0.3 for base and R1 (B10-r chose 0.01 for 1 of 108 matrices) and the D2 side
  arm dropped (estimate ~40 min; it took 56). A coarser grid can only understate T3R (conservative for the line). Full run from 20:16.

## T3R K2 RESULT (21:12; /Users/kamen/Fit12/data/t3r/k2.txt): PASS on NewIdeas' line (information: K1 already failed)
Ran 20:16-21:12 (3356 s, 36 / 36 experts, rc 0, 2 threads, no step-asides). Checks: MSB-3 / E4 2.961-2.972 on every judge;
the TRL-3 base (its own val-best damping) = B10-r's TRL-3 within 0.001x everywhere (B all 0.499x, routed 0.524x, U 0.619x /
0.657x, val 0.468x); routed judges = B10-r's (B 30, U 21 experts with >= 8 rows).
| E4 units (B10a's way: per-matrix ratio, mean over experts, mean of 3) | val | B all | B routed | U all | U routed |
| MSB-3 (sanity ~3)                                                      | 2.966 | 2.966 | 2.972 | 2.961 | 2.969 |
| TRL-3 3INST base (= T3R ranks 2-9)                                     | 1.407 | 1.504 | 1.556 | 1.871 | 1.960 |
| base + R1 (= T3R ranks 0-1)                                            | 0.433 | 0.461 | 0.475 -> PASS (line 1.0) | 0.570 | 0.596 |
B routed per projection (base + R1): gate 0.419, up 0.423, down 0.583 E4; ratio-of-means variant 0.476. R1 keeps ~0.31 of
the base's error. Predictions hit (base ~1.5, base + R1 0.45-0.6).
R1 exact rate: 1 + 12/256 + 16/K b/w = gate / up 1.0531, down 1.0719 -> 650,880 B per expert (1.0594 b/w); 655,360 B as a
16 KiB record. Base: 1,879,680 B (3.0594 b/w), 1,884,160 padded. Val-picked damping (0.1 / 0.3): base gate 5/31, up 5/31,
down 16/20; R1 gate 5/31, up 4/32, down 17/19.
Encode at 2 threads: TRL-3 5.29 s, R1 5.09 s per matrix (k = 1 costs as much as k = 3 in this Viterbi) -> base + R1 =
~31 s per expert, ~210 h for all 24,576 experts on 2 cores: the GPU encoder is a must.
Reading: vs W4, T3R's ranks 0-1 carry 0.475 E4 where v3 has 0, and ranks 2-9 1.56 E4 where v3's MSB-3 has 2.97. Vs the
(unseen) bf16 weights, if the errors add: ranks 0-1 ~1.48 vs 1.0 E4 (+48 %), ranks 2-9 ~2.56 vs ~3.97 E4 (-36 %). So T3R
trades top-2 accuracy for ranks-2-9 accuracy; it is not "v3 or better on both classes". Only the KLD screen can price it.
T3R overall (Fit12's read): K1 fails on the pre-registered 16-row protocol at 12 GB (-16.4 %), passes by ~1 point with
per-token or 4-row checks, -29 % at 24 GB; K2 passes. Whether it lives = NewIdeas' call (protocol); adoption = Kamen's.

## PFR pre-registration (Sun 25 Oct ~22:00; NewIdeas 21:52: "propose your best line-2 lever + its cheapest kill test")
Line-2 anatomy (C_m 51 decode, per-pass map above): ~165 ms = SSD service 108.7 (66 %, at the burst floor: SS1) + compute
gaps 39.2 + drafts 8.3 + edges 8.7; exact decode levers left are small (GDN ~+2.5 %, tier_glu dedup ~+1 %, direct reads
<= +1-2 %, slots ~+0.7 % each). The unexploited slice is PREFILL-TIME MEMORY: the decode pools (MSB 51 x 2,162,688 x 48 =
5.29 GB + LSB 20 x 770,048 x 48 = 0.74 GB + the MTP head's) sit idle during prefill (tier._prefill_grouped reads records3
into its own 2 x 64-expert scratch) yet set the 12 GB peak, and the prefill is chunk-major (bt_b12.b12_prefill: for each
4096-token MoE chunk, all 48 layers), so every chunk re-reads every expert of every layer (~1.5 GB / layer, ~72 GB / chunk).
Measured cost per extra chunk-read: 5.4 s (Breakthrough L2-10 batch 2, 64K lane: s_moe 2048 298 s vs 4096 212 s, +16
chunks) and 6.3 s (my P1, 8,474 tokens: s_moe 8192 346 vs 4096 275 tok/s, 3 -> 2 chunks); s_moe 8192 broke 12 GB (12.73).
PFR = Prefill with the decode pools Released: before a prefill whose transient would not fit beside the pools, free the
pools (Python refs + mx.clear_cache; v3k ext: a reset_layer that swaps in the new buffers and empties table / LRU / free
slots / prefetch state / published tables), run b12_prefill with s_moe = min(P, cap), then re-create the pools sized for
DECODE (C_m by context length). Exact by construction: s_moe changes are bitwise (P1 incl. 512 / 8192; L2-10 2048 vs 4096)
and pool size / contents never change outputs (C_m 51 / 59 / 192-448 all = U2 shas).
Predictions (from the logged numbers, before any build):
- prefill 8,474 tokens: 30.8 s -> 18.2-20.0 s (one chunk: -2 chunk-reads) = +54..+69 % (424-466 tok/s); 2.6K: no change;
  20K: -22..-25 s of ~75 s; 64K plain-path lane at s_moe 16384: 212 -> ~148 s (1.44x).
- prefill-phase peak at 8.5K: 11.51 GB (t_both51) -> ~6.3 GB with s_moe 8474 (+0.84 transient, P1); 64K lane 12.69 ->
  ~6.8 GB at s_moe 4096 / ~9 GB at 16384 (the plain-path 64K blocker, L2-10, becomes a decode-only question: 11.56 GB).
- decode: the prefill no longer sets the lifetime peak, so C_m can use the decode headroom: 8.5K C_m 59 (11.00 + 0.83 =
  11.83 GB; ~+5 % decode, today impossible at 8K), 2.6K C_m 64 (10.44 + 1.35 = 11.79 GB; ~+2.8 % over the C_m 59 opt-in).
- cost: decode starts cold after a released prefill (~5.3 GB re-read over the first ~10-20 passes, ~0.4 s); only released
  when the transient would not fit (short turn prefills keep the pools).
T0 (CPU only, now, <= 30 min, 1 thread; NewIdeas' authorization): (a) read-only audit: every use of the pool memory on the
prefill path of the shipped env (TIER_WARM's adopt, RESCOUNT lookups, the MTP head's streamed pool, the GPU-visible slot
tables), and what reset_layer must clear; (b) the arithmetic above re-derived from the log files. KILL before any build if
the prefill path reads or computes through the decode pools (not only an optional warm copy), or if the logs imply < +25 %
prefill speed at 8,474 tokens or < 3 GB freed.
T1 (GPU, one quiet lock ~15 min; needs NewIdeas' OK; after a ~2-3 h prototype in my copies: v3k ext + fit_b12 harness,
bt_b12 imported read-only): 2.6K and 8,474-token prompts, ref (shipped) vs PFR (C_m 51) vs PFR + C_m 59 at 8.5K; prefill
s, phase peaks (fit_pf sampler), 32 decode tokens. KILL: any decode sha != ref; 8.5K prefill < +25 %; prefill-phase peak
not >= 3.0 GB below ref. The decode-headroom part is dropped (prefill part stands) if C_m 59 after an 8.5K PFR prefill
peaks > 11.95 GB.

## PFR T0 RESULT (Sun 25 Oct ~22:05; CPU only, read-only + the log numbers): PASS -> propose T1 (GPU, needs NewIdeas' OK)
(a) Audit (~/Breakthrough/v3f/bigmoe_v3f/tier.py prefill_call 211-247, _prefill_single 302-323, _prefill_grouped 524-583;
bt_b12.b12_prefill 428-562; shipped env = ~/EvalSuite/evalsuite/cli.py SHIP_AR + SHIP_MTP = fit_b12 ENV): every trunk
prefill branch (streamed ring, grouped G 64, single) reads records3 into its own scratch; the tier pools are touched only by
BIGMOE_TIER_WARM (adopt; NOT set in the shipped env) and BIGMOE_TIER_PF_RESCOUNT (lookups; not set). The MTP head (stream.py,
MTP_STREAM 1, MTP_C 64) DOES gather from its own pool in prefill (stream.py 291) -> PFR keeps that pool (~0.14 GB) and
releases only the 48 trunk base + sidecar pools (6.03 GB). Engine: ext Layer keeps raw pool pointers (Piece.pool, set in
add_layer, bigmoe_ext.cpp 1048-1053) plus table / slot_owner / lru / where / free_slots / pf / pf_order / ready / queued /
meta / pmark -> a v3k reset_layer(L, arrays) must swap the 9 (6) piece pointers and empty all of these; the published
tables come from Layer.table at the next publish (to check in the build).
(b) Logs: per extra chunk-read 5.375 s (64K lane: (298 - 212) / (32 - 16)) and 6.32 s (8,474 tokens: 30.81 - 24.49 s, P1);
8,474-token PFR prefill (one chunk) 30.81 - 2 x 5.375..6.32 = 18.2-20.1 s = +54..+69 % (line +25 %); freed 6.03 GB (line
3 GB); predicted prefill-phase peak 11.51 - 6.03 + 0.81 = ~6.3 GB.
Prior art: MoonShots IDEAS round 12 (PF2, layer-major prefill on the streamed 35B): 1.63x at 8K, all 80 cache arrays and 32
greedy tokens bit-identical; never ported to B12 at 12 GB, where its full-layer staging (~1.4 GB) did not fit beside the
pools. PFR's freed 6 GB makes it fit too (a follow-up: experts read once per prompt, no s_moe transient growth).
- (22:12) Clock note: the PFR stamps were first written as guesses (pre-registration ~22:05, T0 ~22:20, the message to
  NewIdeas "22:25"); file times put them at ~22:00 / ~22:05 / ~22:06 (v3p was created at 22:08). Corrected above.
- Prototype (22:08-22:12, CPU only, my copies): /Users/kamen/Fit12/v3p = v3k sources + ext.reset_layer / invalidate_tables
  (grep "Fit12 PFR"; not built yet: the Eval suite's docbench share lock holds the GPU to 22:48 and the hook asked to wait);
  harness /Users/kamen/Fit12/scripts/fit_pfr.py (release / restore, balanced chunks <= 16,384, tier.tier_call raises while
  the pools are gone); runner /Users/kamen/Fit12/scripts/pfr.sh (SSD gate from ~/Library/Logs/health.csv, fanwatch is down).
  T1 launches only after NewIdeas' GPU OK.
- (22:15) Added to T1's readout before any run: the "restore" phase peak and each arm's lifetime peak. Risk: re-creating 6 GB
  of pools while the prefill's freed-not-reclaimed GPU memory (Breakthrough: up to ~2.1 GB 'unmapped') is still owned
  could spike the footprint at restore. Not a kill line (the pre-registered ones stand); if pfr51's lifetime peak exceeds
  ref51's, PFR's memory claim is void until the restore is staged (e.g. per-layer allocation with clear_cache between).
- (22:27) v3p ext built (cmake, -j 2, rc 0; /Users/kamen/Fit12/v3p/ext/build/bigmoe_ext.cpython-312-darwin.so has
  reset_layer / invalidate_tables). T1 is ready to launch on NewIdeas' GPU OK (pfr.sh after a quiet-lock grant).

## PFR T1 RESULT (Sun 25 Oct 23:38; GPU 23:33:20-23:35:31, one quiet lock, NewIdeas GO 23:33): PASS on all three lines
Outputs /Users/kamen/Fit12/data/pfr/{ref51,pfr51,pfr59}.txt, log /Users/kamen/Fit12/logs/summary.txt; all arms rc 0, 2 prompts
each, no errors. Engine /Users/kamen/Fit12/v3p; harness /Users/kamen/Fit12/scripts/fit_pfr.py; runner /Users/kamen/Fit12/scripts/pfr.sh.
| 8,474-token prompt            | ref51 (shipped) | pfr51            | pfr59 (C_m 59)   |
| prefill                       | 29.59 s, 286/s  | 21.22 s, 399/s   | 21.14 s, 401/s   |
| prefill-phase peak footprint  | 11.62 GB        | 6.42 (-5.20)     | 7.90 (-3.72)     |
| MLX peak in prefill           | 10.16 GB        | 5.10             | 5.10             |
| process lifetime peak         | 11.62 GB        | 10.96            | 11.79            |
| time to first token           | 29.85 s         | 21.68 (-8.17)    | 21.61            |
| decode 32 (cold start)        | 13.9 tok/s      | 13.2             | 13.7             |
| sha (p2k / p8k)               | 82995cdb4e / 0b0d91cd43, identical in all three arms        |
- Kill lines: shas identical -> pass; 8.5K prefill +39.4 % (pfr59 +40.0 %) >= +25 % -> pass; prefill-phase peak 5.20 GB below
  ref (>= 3.0) -> pass. Decode headroom: C_m 59 lifetime peak 11.79 GB <= 11.95 -> the headroom part stands (about
  103 MB per C_m slot, so C_m 60 at 11.89 GB is the 8.5K ceiling). pfr51's lifetime peak is BELOW ref51's (10.96 vs 11.62),
  so the memory claim holds without staging the restore (restore phase peak 9.16 GB; release 0.03 s, restore 0.12-0.14 s).
- Prediction miss: +39.4 % vs +54..+69 % predicted (21.2 s vs 18.2-20.1). The two re-reads cut saved 8.37 s (4.19 s each),
  not 5.4-6.3 s: ref's third chunk is only 282 tokens, so its re-read covers fewer experts. The peak prediction held
  (11.51 -> ~6.3 GB predicted, 11.62 -> 6.42 measured).
- 2,617-token prompt (one chunk either way): no gain. Prefill 9.93 -> 10.10 s, time to first token 10.27 -> 10.60 s (+0.33 s:
  release + restore + 0.17 s slower prefill after clear_cache). A port should gate PFR on n > s_moe (4096).
- Cold start (NewIdeas' ask): at p8k, ref's pools hold p2k's decode residue; PFR starts empty. First pass 0.26 -> 0.31 s,
  tokens 1-32 2.05 -> 2.12 s: about +0.12 s, not the ~0.4 s estimate. At p2k (first prompt after load) the arms are equal:
  15.8 vs 15.9 tok/s, first pass 0.33 s in both. Not measured: a same-conversation follow-up, where the residue would help ref more.
- Swap (NewIdeas' ask): 7,017 MB used before, after each arm and at the end: unchanged. SSD 22 C -> 49 C (stop line 73).

## PFR T2 pre-registration (Sun 25 Oct 23:47; NewIdeas 23:39: GO for T2 at 64K, ~8 min GPU, one quiet lock)
Path: the AR path at N = 65,000, mirroring Breakthrough's pk64.py (read-only): B12LL loaded without the MTP head, the
first 65,000 tokens of ~/Breakthrough/data/prompts/ctx64k.txt, B.b12_prefill(model, ids[:-1], cache, 512, s_moe), then
greedy one-row decode (32 steps; the sha of the first 8 = pk64's token sha). This is the path where the prefill is the
12 GB blocker. pk64 at 64K: plain prefill peak 13.58 GB, then 11.64 GB after 8 decode steps; lane 12.69 / 11.56. The
MTP path cannot fit at 64K whatever the prefill does: pk64 g64_mtp is at 13.55 GB right after the prefill, pools
resident, and _gen_mtp also decodes on a deep copy of the prefill KV (spec125.restore_state). It is not in T2.
Arms, one quiet lock (--queue 40m; an arm is skipped if < 8 min of lock is left), SSD gate <= 66 C / stop 73 C:
  ref64 (s_moe 4096) | pfr64 (PFR cap 16384 -> 4 chunks of 16,250) | lane64 (L2-10 lane, s_moe 4096) | lanepfr64 (lane + PFR)
Harness /Users/kamen/Fit12/scripts/fit_pfr64.py (release/restore as in T1, plus gc + clear_cache before the restore);
runner /Users/kamen/Fit12/scripts/pfr64.sh; outputs /Users/kamen/Fit12/data/pfr/{ref64,pfr64,lane64,lanepfr64}.txt.
Prediction: 12 avoided chunk re-reads (16 -> 4) x 4.2-5.4 s = 50-65 s saved. ref64 ~306-312 s -> pfr64 ~241-262 s
(+20..+27 %); lane 212 s -> ~147-162 s. Prefill-phase peak 13.6 -> ~8.6-9.9 GB (6.0 GB of pools out, a 16K MoE chunk's
transient in). Lifetime peak: the decode-time state at 64K (pk64 11.64 GB) plus how fast the freed prefill memory is
reclaimed before the pools come back -> 11.6-12.3 GB, i.e. borderline on 12.0.
KILL lines:
  - any PFR arm's 32-token sha != its ref arm's (pfr64 vs ref64, lanepfr64 vs lane64) -> KILL
  - pfr64's prefill saving vs ref64 < 25 s (half the T1-calibrated low end, 12 x 2.1 s) -> KILL
  - pfr64's prefill-phase peak not >= 3.0 GB below ref64's -> KILL
Own pass/fail line (NewIdeas' question 2): pfr64's process lifetime peak < 12.0 GB -> "64K fits under 12 GB exact"; the
same line for lanepfr64 (the config that would ship if Kamen adopts the lane). Information: the 8-token shas vs pk64
(plain cfa3402e3e, lane 56c4e038e8), time to first token, decode tok/s, per-phase peaks, KV bytes, swap before/after.

## PFR T2 RESULT (Mon 26 Oct 01:19; GPU 23:48:10-00:01, then frozen by the silent night): KILL on the sha line
Outputs /Users/kamen/Fit12/data/pfr/{ref64,pfr64,lane64}.txt (rc 0, no errors); lanepfr64 frozen ~77 s in (SIGSTOP 00:01,
resumes 08:01-08:16; its prefill time will include the freeze, so it is invalid; its sha and peaks will still be valid).
| N = 65,000, AR path      | ref64 (plain)      | pfr64 (plain + PFR) | lane64 (lane)      |
| prefill                  | 259.9 s, 250/s     | 208.8 s, 311/s      | 201.7 s, 322/s     |
| prefill-phase peak       | 13.56 GB           | 9.39 (-4.17)        | 12.70              |
| process lifetime peak    | 13.72 GB           | 11.62 (decode)      | 12.78              |
| decode, steps 2-32       | 12.54 tok/s        | 13.71               | 13.33              |
| sha 8 / first logits     | cfa3402e3e / 4dcabb778f | 56c4e038e8 / 68ddb5344e | 56c4e038e8 / 1ccb5e8c59 |
| sha 32                   | dc67a9270b         | bba3d6e0f5          | aef9384fda         |
- ref64 and lane64 match Breakthrough's pk64 bit for bit (8-token sha and first-step logits hash: g64_base and g64_lane),
  so harness and engine reproduce the reference.
- KILL: pfr64 != ref64 (tokens AND first-step logits). The other lines would pass: saving 51.1 s (line 25 s; predicted
  50-65), prefill-phase peak -4.17 GB (line 3.0), lifetime peak 11.62 GB < 12.0. The 64K-under-12-GB memory result is real,
  but this PFR is not exact.
- Consequence for T1: its lines compared TOKEN shas only. Here pfr64's first 8 tokens equal the lane's while its logits
  differ from both, so equal token shas do not prove bit-exactness. T1's "exact" is UNPROVEN until a logits-hash check.
- Two candidate causes, not separated yet. (A) The MoE chunk size: 16,250 rows vs 4096 (tier.py 392/496: the sorted rhs
  gather and the expert grouping depend on row counts). Against: pk64 g64_lane_m2048 = g64_lane bit for bit, so 2048 vs
  4096 is invariant. (B) The decode starts from a reset tier state: restore() runs reset_layer, which clears pred/pf/lru,
  and invalidate_tables(). If the B12LL decode depends on that state (late MSB-3 ranks 2-10 vs residency, prefetch), the
  first step differs. Hint: pfr64 decodes 9 % faster than ref64.
- Next (GPU, needs NewIdeas' OK, after 08:01): the AR harness at 8,474 tokens, ~35 s per arm, logits hash per arm.
  ref (s_moe 4096) | s_moe 8474 with the pools resident (A only) | PFR cap 4096 (release + reset, same chunks: B only) |
  PFR cap 16384 (both). Then the same two isolating arms at 64K only if 8.5K is inconclusive.

## PFR A/B pre-registration (Mon 26 Oct 01:22; diagnostic, GPU after 08:01 with NewIdeas' OK)
Question: what broke the bits in T2: (A) the bigger MoE chunk, or (B) the release/restore/reset around it.
Equal = BOTH the first-step logits hash and the 32-token sha equal ref's (a token sha alone is not enough; see T2).
Harness /Users/kamen/Fit12/scripts/fit_pfr64.py (AR path; F12_PFR_BAL=0 gives chunks of exactly the cap); runner
/Users/kamen/Fit12/scripts/pfrab.sh (ARMS=...; outputs /Users/kamen/Fit12/data/pfr/ab_<arm>.txt).
Stage 1, N 8,474 (~4 min GPU, one quiet lock): r8 ref s_moe 4096 | s8 s_moe 8474 with the pools resident (A only) |
c8 PFR with chunks of exactly 4096 (B only) | p8 PFR cap 16384 (= 8474, both).
Stage 2, N 65,000, only if stage 1 is all-equal (~10 min): s64 s_moe 16250 with the pools resident | c64 PFR exactly 4096.
Reading:
- s8 != r8 (or s64 != ref64): (A). Chunks above 4096 change the bits, so PFR's speed part cannot be exact. If c8/c64 is
  equal, the exact PFR is the memory part alone (release with s_moe 4096: 64K under 12 GB, no prefill gain).
- c8 != r8 (or c64 != ref64): (B). The restore changes the decode. Fix: swap the pool pointers but keep pred/pf/lru, then
  re-test.
- All equal at 8.5K, and stage 2 shows (A) or (B): T1's 8.5K claim is confirmed bit-exact, and the 64K break is size-specific.
No kill lines: this decides whether an exact PFR exists, and which one.

## PFR A/B additions (Mon 26 Oct 08:09; NewIdeas GO 03:03, before the run)
1. Every arm hashes the KV cache at the end of the prefill, before any restore (all 48 layers' cache state; per-layer
   prefixes printed), so prefill math and decode state separate. Readings: KV(s8) != KV(r8): (A), the chunk size.
   KV(c8) != KV(r8): the prefill with the pools released itself changes the math. KV equal but the first-step logits
   differ: the decode depends on residency / tier state (then find which path: slot-order sums, HC / late selection, or
   the tables, before any pointer-swap fix).
2. Every arm whose RESULT (kv, logits, sha32, sha8) differs from r8's runs once more in a fresh process, then r8 does too.
   An arm that differs from itself means a race (e.g. a slot read before its load lands), a bug, not rounding order.
First-step logits saved per arm (/Users/kamen/Fit12/data/pfr/ab_<arm>_logits.npy, 0.6 MB each) to size any difference.
lanepfr64 (T2 arm 4, frozen 00:01-08:01, so its timings are void): sha8 56c4e038e8 = lane64's, but logits 6d30f0597c !=
lane64's 1ccb5e8c59 and sha32 3513442745 != aef9384fda. The lane + PFR is not exact either. Prefill-phase peak 9.73 GB,
lifetime peak 11.96 GB (decode MLX active 11.03 GB, as pfr64's 11.02).

## PFR A/B stage 1 RESULT (Mon 26 Oct 10:04:37-10:06:38, one quiet lock, GPU 0 % before r8; all 4 arms rc 0, no errors)
N 8,474 (first 8,475 tokens of ctx64k.txt), AR path, v3p, C_m 51 / C_l 20, 32 greedy steps:
| arm | prefill | KV after prefill | first-step logits | sha8 / sha32 | prefill-phase peak | lifetime peak |
| r8 ref s_moe 4096 | 27.4 s | ecd389371d | c57fe16fd1 | 578ced9ee1 / 142abd5d7e | 11.05 GB | 11.11 GB |
| s8 s_moe 8474, pools resident | 18.0 s | = | = | = | 12.00 GB | 12.03 GB |
| c8 PFR exactly 4096 | 27.9 s | = | = | = | 5.08 GB | 9.88 GB |
| p8 PFR cap 16384 (one chunk) | 18.8 s | = | = | = | 6.04 GB | 9.88 GB |
The four first-step logits files are byte-identical (md5 a68e370c0f94fe9d2b233b97f3a9f32e). No arm differed, so nothing
was repeated. Decode 12.71-12.78 tok/s in all four. Time to first token: r8 27.57 s, p8 19.15 s (-8.4 s, -31 %).
Reading (as pre-registered): at 8.5K PFR is bit-exact: neither the bigger MoE chunk (s8) nor the release/restore (c8)
changes a bit. So the 64K break is size-specific. T1 (MTP path, chat prompt) had no logits hash; this supports it but
does not re-prove it. Stage 2 runs.

## PFR A/B stage 2 pre-registration (Mon 26 Oct 10:08; stage 1 all-equal, so it runs as pre-registered at 01:22)
N 65,000, AR path, same harness and runner: r64 ref (added: ref64 had no KV hash, so a 64K KV reference is needed) |
s64 s_moe 16250, pools resident (A at 64K) | c64 PFR, chunks of exactly 4096 (B at 64K). One quiet lock (30 min). Every
arm that differs from r64 is repeated once; r64 itself is not (REPREF=0). Its logits / sha32 must equal T2's ref64
(4dcabb778f / dc67a9270b), which is the cross-process check.
Readings: KV(s64) != KV(r64): (A), a chunk-size threshold between 8,474 and 16,250 tokens. c64 equal on all fields: the
exact PFR at 64K is the release with s_moe 4096 (64K under 12 GB, no prefill gain beyond the memory). KV(c64) equal but
the logits differ: the decode state after the restore differs at 64K only (find the path). A repeat that differs from
its first run: a race.

## PFR A/B stage 2 RESULT (Mon 26 Oct 10:26:40-10:42:54, one quiet lock, GPU 0 % before each first-run arm; all rc 0)
N 65,000, AR path, v3p, C_m 51 / C_l 20, 32 greedy steps:
| arm | prefill | KV after prefill | logits | sha8 / sha32 | prefill-phase peak | lifetime peak |
| r64 ref s_moe 4096 | 261.1 s | 2227a92414 | 4dcabb778f | cfa3402e3e / dc67a9270b | 13.70 GB | 13.74 GB |
| s64 s_moe 16250, pools resident | 200.2 s | 11d802ef7b | 68ddb5344e | 56c4e038e8 / bba3d6e0f5 | 15.32 GB | 15.34 GB |
| s64b (repeat) | 200.6 s | 11d802ef7b | 68ddb5344e | 56c4e038e8 / bba3d6e0f5 | 15.34 GB | 15.36 GB |
| c64 PFR exactly 4096 | 271.5 s | 2227a92414 | 4dcabb778f | cfa3402e3e / dc67a9270b | 7.69 GB | 11.99 GB |
- r64 = T2's ref64 (logits, sha32): the reference is reproducible across processes.
- s64 reproduces T2's pfr64 exactly (logits 68ddb5344e, sha32 bba3d6e0f5). So T2's break came from the 16,250-token
  chunk alone; the release / restore added nothing. The repeat is identical, so it is deterministic, not a race.
- c64 = r64 on every field: the release / restore is bit-exact at 64K, as at 8.5K.
- s64's KV differs from r64's in all 48 layers from layer 0, i.e. before any MoE. So it is not the MoE arithmetic.
- CAUSE (code, ~/Breakthrough/scripts/bt_b12.py b12_prefill): each s_moe chunk is cut into s_attn = 512 query chunks
  counted from the chunk's own start: `range(m0, m1, s_attn)`. So with s_moe 16250 (not a multiple of 512), every query
  chunk after token 16,250 starts at a different place than in the reference, and the attention / linear-attention
  (gated DeltaNet) chunks round differently from layer 0 on. s_moe 4096, 2048 and 8474 (the last chunk at 8.5K) keep the
  512 grid, which is why stage 1 and pk64's m2048 = m4096 are bit-exact. The docstring's "equals chunk s_attn bit for
  bit" holds only when s_moe is a multiple of s_attn.
- Memory: the 64K decode itself peaks at 11.94-11.97 GB in every arm (pools 6.00 + KV 1.60 GB). The release takes the
  prefill phase far below that (7.69 GB), so 64K's lifetime peak becomes the decode's: 11.99 GB, at the line with no
  margin. (T2's pfr64 decode peak of 11.62 GB was the low outlier; MLX active in decode is 11.02-11.15 GB in all.)

## PFR-a pre-registration (Mon 26 Oct 10:46; one confirmation arm right after stage 2)
Prediction from the cause: PFR with 512-aligned chunks is bit-exact AND keeps the speed. Arm a64: F12_PFR=1,
F12_PFR_CAP=16384, F12_PFR_BAL=0 (chunks 16384 x 3 + 15848, all on the 512 grid), N 65,000, same harness.
PASS lines: (1) KV, logits, sha8 and sha32 all equal r64's (2227a92414 / 4dcabb778f / cfa3402e3e / dc67a9270b);
(2) prefill <= 215 s (r64 261.1 s); (3) prefill-phase peak below the decode-phase peak (the prefill no longer sets the
process peak). KILL: any of them fails. If it passes, PFR = release + chunks of 512 * ceil(n / (k * 512)),
k = ceil(n / 16384), last chunk > 8 tokens.

## PFR-a RESULT (Mon 26 Oct 11:26:30-11:29:55, one quiet lock, GPU 0 % before it, rc 0): PASS on all 3 lines
a64 = release + chunks of exactly 16384 (512-aligned: 16384 x 3 + 15848), N 65,000, AR path:
- (1) PASS: KV 2227a92414, logits 4dcabb778f, sha8 cfa3402e3e, sha32 dc67a9270b = r64 and c64. The r64 / c64 / a64
  logits files are byte-identical (md5 c0af4220af61c057f8249dc214178220).
- (2) PASS: prefill 196.4 s vs r64 261.1 s (-64.7 s, -25 %, 1.33x); TTFT 196.77 vs 261.29 s.
- (3) PASS: prefill-phase peak 9.32 GB < decode-phase peak 12.09 GB (r64's prefill phase 13.70 GB).
- Memory caveat: the decode phase peaked at 12.09 GB and the lifetime at 12.12 GB, over 12. The instrumented A/B arms
  (r64 11.97, c64 11.97, a64 12.09) all sit +0.35-0.47 GB above the T2 runs, which had no KV hash (ref64 and pfr64
  decode phase 11.62 GB). MLX active in decode is 11.15 GB in r64, c64 and a64 alike, so the extra is non-MLX memory.
  Likely the hash's ~3.2 GB of float32 numpy copies just before the decode (not shown).
So the 64K exact-PFR memory number is open; next, a64m (below).

## PFR-a memory pre-registration (Mon 26 Oct 11:30): a64m = a64 without the KV hash (F12_KVHASH=0, T2's conditions)
Lines: logits, sha8 and sha32 equal r64's (exactness carries over); lifetime peak <= 11.95 GB (T1's line).
Readings: <= 11.95: exact 64K under 12 GB, with the decode setting the peak. > 11.95: at C_m 51, 64K sits at the line
(margin needs fewer slots at long context, or the 8-bit KV of Breakthrough's dx line).

## PFR-a memory RESULT (Mon 26 Oct 12:25:54-12:29:20, one quiet lock, GPU 0 % before it, rc 0): PASS
a64m = a64 without the KV hash (T2's conditions):
- Exact: logits 4dcabb778f, sha8 cfa3402e3e, sha32 dc67a9270b = r64, and the logits file is byte-identical
  (md5 c0af4220af61c057f8249dc214178220).
- Lifetime peak 11.77 GB <= 11.95 (PASS). Phase peaks: prefill 9.32 GB, restore 11.10 GB, decode 11.75 GB.
  MLX active in decode is 11.15 GB, as in r64 / c64 / a64.
- The KV hash was the extra: a64's decode phase 12.09 GB vs a64m's 11.75 GB.
- Prefill 195.9 s; TTFT 196.25 s (vs ref64 / r64 260.0-261.3 s).
- So exact PFR-a at 64K: prefill -25 %, TTFT -64 s, lifetime peak 13.72 GB (T2 ref64, same conditions) -> 11.77 GB.
  The decode sets it, with 0.18 GB to T1's 11.95 line.
PF2 (layer-major, experts read once per prompt) sized by this: at 64K, each extra chunk-read costs ~5.4-6.3 s (P1 logs),
so going from 4 chunks to 1 would save <= ~16-19 s (~8-10 %) and needs the 65K-token MoE transient. At <= 16K, PFR
already uses one chunk. So PF2 is small now; it is not worth a build before the decode-side levers.

## Rapid screens, batch 1 (Mon 26 Oct 17:41; Kamen's rule, pacer RULES.md 'Rapid screening': one screen <= 10 min of machine time, line first)
- PLA (per-layer pool sizes; CPU replay, 1 core, ~5 min; /Users/kamen/Fit12/scripts/fit_pla.py, engine-faithful LRU as
  fit_v3replay.py; expA traces, MTP-like checks R 4 / A 3, counted from token 400): per-layer C_m / C_l chosen greedily on
  2 sets, judged on the other 2 (both folds), same pool bytes as uniform C_m 51 / C_l 20. Control: the best UNIFORM split of
  the same bytes. PASS: per-layer read MB / token >= 5 % below uniform 51 / 20 out of sample in both folds -> engine screen.
  KILL: < 3 % in either fold. (A uniform-split gain alone is reported separately: an env-only change.)
- CMP (per-prompt C_m; NewIdeas GO 13:46, QUEUE L2-11; GPU, one 10-min quiet lock; /Users/kamen/Fit12/scripts/fit_cmp.py =
  fit_pfr.py + 512-aligned PFR chunks + hashes, MTP 3:3:0, 96 greedy tokens per prompt, PFR on, B12_PFR=0 so bt_b12 adds
  nothing): arms cm51 / cm59 on p2k (2.6K) + p8k (8,474); then cm51 on p32k (peak only) if >= 200 s of lock is left.
  PASS: cm59's first-step logits hash, sha32, sha96 and decode-end state hashes (trunk + MTP head) = cm51's at both sizes,
  every lifetime peak <= 11.95 GB, warm decode (tokens 33-96) >= +4 % at 8.5K. KILL: any hash mismatch, a peak > 11.95 GB,
  or < +2 % at 8.5K. Between: no verdict, NewIdeas decides.
- PLA RESULT (17:47, CPU 8 s; /Users/kamen/Fit12/data/pla/pla.txt, curves data/pla_curves.npz): KILLED. Lagrangian per-layer
  C_m / C_l at the same 6.03 GB: out of sample +0.3 % (fit bulgarian + chat, judged on code + math: 374.6 -> 375.6 MB/token)
  and -1.3 % (fit code + math: 206.4 -> 203.8); in-sample only -1.7 %. The best uniform split (C_m 54 / C_l 10) is -0.7..-0.8 %:
  noise-level, not worth an env change. At 10 % residency every layer looks alike, so uniform pools are already near optimal.
  (The first allocator, a plain greedy, got stuck on the curves' flat parts: +0.1 / +6.0 %, and +1.9 % in-sample, worse than
  the uniform pools it could have chosen. That was an allocator bug, not a result, so it was replaced by the Lagrangian.)
- PRN (static expert pruning, REAP-style, bytes side; CPU replay ~2 min; /Users/kamen/Fit12/scripts/fit_prn.py): per layer,
  prune the experts with the least routed gate mass (fit on 2 expA sets), re-route the other 2 sets (top-10 = the first 10
  unpruned of each token's top-20), replay the engine's pools: 12 GB (C_m 51 / C_l 20, LRU, checks 4 / 3) and 24 GB (147 / 46,
  LFU 0.99, checks 16 / 10). Fractions 12.5 / 25 / 37.5 / 50 %. Also logged: the routed gate mass that moves to substitutes.
  PASS (bytes): at 25 % pruned, 12 GB read MB / token >= 15 % below unpruned out of sample in both folds -> a quality screen
  next (non-exact: task results, Kamen's call). KILL: < 8 %.
- CMP RAW (17:47:11-17:50:32, one quiet lock, GPU 0-3 % before each arm, no PAUSE; /Users/kamen/Fit12/data/scr/cmp/*.txt):
  cm51 reproduces T1 (sha32 82995cdb4e / 0b0d91cd43). 2.6K: cm59 = cm51 on every hash (logits bd6097bdf7, sha96 fae7ae2de6,
  kv e053ddce52, head 1541ecfe57); warm 14.67 -> 15.30 tok/s (+4.3 %); peaks 10.47 / 11.44 GB. 8.5K: logits 7d045d7b7d, sha32
  and sha96 6ca22291ed equal, but the decode-end state differs (kv f84117b336 vs 47f230fae6, head ea45ffea6e vs 12189f4c06);
  warm 16.44 -> 17.34 (+5.5 %); peaks 10.98 / 11.94 GB. 32K cm51: decode peak 12.51 GB (MLX active 11.96), over the line at
  C_m 51 already; prefill 86.0 s (31,653 tokens, chunks 15,872 x 2 - 91).
- CMPR (repeat, before any CMP verdict; ~3 min, one 10-min lock; p8k only, F12_LAYERS=1 per-layer prefixes): r51 / r59 on
  v3p, l51 / l59 on Breakthrough's v3l (read-only; v3i's gate stamps). Reading: r51 != cm51 -> the decode-end state hash is
  not reproducible, so CMP has no exactness verdict yet (find why). r51 = cm51 and r59 = cm59 -> pool size changes the MTP
  decode's bits at 8.5K -> CMP KILLED on the exact line (tokens equal: information), LAYERS names the first layer that differs.
- MOV (~2 min, one 10-min lock): the MTP path holds TWO copies of the prefill state during decode (bt_b12 and the suite keep
  `pre` while spec125.restore_state deep-copies it). F12_MOVE=1 hands the arrays over instead. p32k, C_m 51, v3p. PASS:
  lifetime peak <= 11.95 GB AND logits / sha32 / sha96 / kv / head = cm51_32k's. KILL: a peak > 11.95 GB or any hash differs.
- PRN RESULT (17:52, CPU 8 s; /Users/kamen/Fit12/data/prn/prn.txt): KILLED. At 25 % pruned, 12 GB read MB / token -1.9 %
  (fit bulgarian + chat -> code + math) and +5.8 % (fit code + math -> bulgarian + chat); line 15 %, kill 8 %. The quality
  side is worse: the pruned experts held 26-27 % of the judged sets' top-10 gate mass (13 % at 12.5 %, 53 % at 50 %), i.e.
  about what random pruning would remove: the 125B's routing has no rarely-used experts that transfer across domains. 24 GB
  bytes fall faster (-4 / -12 % at 25 %, -54 / -56 % at 50 %) but only by deleting half the routed mass. (24 GB baseline here
  99.1 MB / token, 4-set mean, LFU 0.99 16 / 10; the 12 GB baseline equals PLA's exactly.)
- AMENDMENT (17:56, before reading MOV's result): kv_hash's overall hash changed at 17:51 (it now hashes per-layer digests,
  for the LAYERS line), so cm*'s kv / head hashes cannot be compared with r* / l* / m*. Only the logits and token shas are
  comparable across that change. MOV's kv / head are therefore judged against movc = the same arm without F12_MOVE (new hash,
  fresh process, p32k, C_m 51); its logits / sha32 / sha96 still against cm51_32k too.
- CMP RESULT (17:56; data/scr/cmp/{cm51,cm59,r51,r59,l51,l59}.txt): KILLED on the pre-registered exact line. At 8.5K the
  decode-end state differs between C_m 51 / C_l 20 and C_m 59 / C_l 24 (r51 kv e7b11d54c0 head e98cc2d48c vs r59 0a68624c42 /
  50984187ae), deterministically (v3l gives the same pairs: l51 = r51, l59 = r59). Per layer: 0-6 equal (incl. the layer-3
  attention KV), layer 7 (attention) onward differ, and the MTP head. Tokens equal at both sizes (sha96 fae7ae2de6 /
  6ca22291ed); at 2.6K every hash is equal. Speed (warm, tokens 33-96): +4.3 % at 2.6K, +5.5 % / +5.1 % (repeat) at 8.5K.
  Peaks at 8.5K C_m 59: 11.91-11.94 GB (line 11.95). Also: the state's 'logits' are the MTP HEAD's draft logits, and they
  depend on the process's history (p8k after p2k 7d045d7b7d, fresh p8k 629a4e74db): the head's pool persists across prompts.
- CMP2 (pre-registered 17:58, ~1.5 min, one 10-min lock; p8k, v3p, LAYERS): is it C_l? x59l20 (C_m 59, C_l 20) vs r51
  (C_m 51, C_l 20); x51l24 (C_m 51, C_l 24) as diagnosis. PASS: x59l20 = r51 on logits, sha96, kv, head; warm >= +4 %;
  peak <= 11.95 GB. KILL: any hash differs, or < +2 %.
- MOV RESULT (17:54:36-17:58:13, two 10-min locks, GPU 0 % before each; data/scr/mov/{m51_32k,c51_32k}.txt): PASS, exact.
  Handing the prefill arrays to the decode instead of deep-copying them (F12_MOVE=1): every hash = the control's (logits
  cbc97c2aee, sha32 149c35ee95, sha96 ba0ccdbaac, kv 946633575a, head df07a098e9; logits and shas also = cm51_32k's).
  32K (31,653 tokens) MTP decode peak 12.48 -> 11.32 GB (-1.16 GB; MLX active 11.96 -> 10.80), warm 14.35 vs 14.32 tok/s.
  So the MTP path's 12 GB wall moves from < 32K to > 32K. Port (not Fit12's code): spec125.restore_state copies
  state['cache'] / ['mtp_cache'] while bt_b12 (pre / PRESTATE) and the suite's decode._gen_mtp keep the state alive; a
  consume flag (pop instead of copy) plus dropping the caller's reference does it, where the state is not reused.
- CMP2 RESULT (17:58-18:00; data/scr/cmp/{x59l20,x51l24}.txt): KILLED. It is C_m, not C_l: x59l20 = r59 (kv 0a68624c42, head
  50984187ae) and x51l24 = r51 (kv e7b11d54c0, head e98cc2d48c). C_l 24 buys nothing (warm 17.21 vs 17.20; 16.41 vs 16.36) and
  costs 0.14 GB (C_m 59 peak 11.77 GB with C_l 20, 11.91 with C_l 24).
- CMP3 (diagnosis, pre-registered 18:01; ~1.5 min): bt_b12 trunk layers use the verify rows' wait-all flags (FL_MR) only
  when S > 1; single-row passes use FL, where B12_LATE3=2,10 adds COLD ranks 2-9 half a layer late, which depends on C_m by
  construction. Test in a Fit12 copy (/Users/kamen/Fit12/btc/bt_b12.py, one change: F12_ALLWAIT=1 -> single-row trunk passes
  use FL_MR): w51 / w59 (C_l 20), p8k, LAYERS. w51 = w59 on kv and head -> that rule is the whole C_m dependence; else not.
- CMP3 RESULT (18:01:04-18:02:10; data/scr/cmp/{w51,w59}.txt, bt_b12 from /Users/kamen/Fit12/btc): CONFIRMED. With single-row
  trunk passes waiting for every rank (F12_ALLWAIT=1), C_m 51 = C_m 59 bitwise at 8.5K (kv 97da92bd47, head f91812db61, logits
  629a4e74db, sha96 6ca22291ed): bt_b12's single-row FL rule (B12_LATE3=2,10 adds cold ranks 2-9 half a layer late) is the
  whole C_m dependence of the shipped MTP path. Cost of the change: none seen (warm 16.36 -> 16.37 tok/s at C_m 51; tokens = the
  shipped path's, state bits not). On that path CMP passes its line at 8.5K: C_m 59 / C_l 20 = C_m 51 on every hash, peak
  11.77 GB, warm +4.6 % (16.37 -> 17.12). Not adopted: changing the single-row rule changes the shipped path's bits
  (Breakthrough's bt_b12; NewIdeas / Kamen decide), and a confirmation (more prompts and sizes, MOV for > 16K) comes first.
- MOV64 (pre-registered 18:06; ~4.5 min, one 10-min lock): the suite's MTP path + MOV at ~64K tokens (p64k = HEAD + the
  first 199,000 chars of ctx64k.txt), C_m 51 / C_l 20, v3p, shipped single-row rule. PASS: lifetime peak <= 11.95 GB (then
  the MTP path holds '64K within 12 GB' with MOV). KILL: > 11.95 GB; then long prompts need C_m below 51 (0.104 GB per slot,
  measured by CMP) and the smallest fitting C_m is logged. Decode tok/s and prefill time: information.
- AW (pre-registered 18:07; ~5 min, one 10-min lock): the ALLWAIT change itself (single-row trunk passes wait for every rank;
  /Users/kamen/Fit12/btc/bt_b12.py) on p2k + p8k, 256 greedy tokens, MOV on, C_l 20, v3p: s51 (shipped rule, C_m 51), a51
  (ALLWAIT, C_m 51), a59 (ALLWAIT, C_m 59); a65 (ALLWAIT, C_m 65, p2k only). PASS: a51's tokens (sha256) = s51's at both sizes,
  a51's warm tok/s >= s51's - 1 % (mean of the two prompts), and a59 = a51 on logits, sha256, kv and head at both sizes.
  KILL: any token difference vs s51, a cost > 2 %, or a59 != a51. a65: peak and warm speed at 2.6K (fits if <= 11.95 GB).
- MOV64 RESULT (18:06:55-18:10:56, one quiet lock, GPU 0 % before; data/scr/mov/m51_64k.txt): KILLED. 65,455 tokens, MTP
  path + MOV, C_m 51 / C_l 20: decode peak 12.88 GB (MLX active 12.24), prefill phase 9.85 GB; prefill 226.6 s (289 tok/s);
  decode 14.1 tok/s (96 tokens). From 32K to 64K the MTP path grows ~46 KB / token even with one state copy, vs ~25 KB / token
  for the AR path (11.77 GB at 64K): ~0.5 GB of MLX at 64K is unaccounted for (pools 6.03 + KV 1.6 + head KV 0.13 + weights).
  So a 64K prompt on the MTP path needs C_m <= 41 (computed at 0.104 GB / slot, not run); 32K fits at C_m 51 (11.32 GB).
- AW RESULT (18:11:31-18:15:27, one quiet lock, GPU 0 % before the first arm; data/scr/aw/*.txt): PASS on all three lines.
  Tokens: a51 = s51 (sha256 a0774ec635 at 2.6K, ed67050746 at 8.5K). Cost: warm 15.24 -> 15.13 (-0.7 %) and 17.16 -> 17.18
  (+0.1 %), mean -0.3 % (line -1 %). Bitwise: a59 = a51 on logits, sha256, kv and head at both sizes (2.6K kv 37028a6545,
  8.5K kv 5a760388d6), and so is a65 at 2.6K. In this history (p2k, then p8k, MOV, 256 tokens) the shipped rule s51 also
  equals a51 bitwise; it differed in the fresh-p8k runs (CMP3), so the late rule's effect depends on the run, ALLWAIT's
  invariance held in both. Speed (warm, tokens 33-256): C_m 59 +5.4 % (2.6K) / +4.8 % (8.5K); C_m 65 at 2.6K +8.5 % (16.42).
  Peaks (lifetime, MOV on): 2.6K 10.31 / 11.14 / 11.76 GB at C_m 51 / 59 / 65; 8.5K 10.59 / 11.41 at 51 / 59. (The runner's
  'GPU 98 %' before a65 was read 0 s after a59 exited: my own tail; GPU 0 % a minute later, a65's hashes = a51's.)
  Per-prompt rule, MTP path + MOV + ALLWAIT, C_l 20, 0.104 GB per C_m slot: C_m 65 at <= 2.6K (measured), ~63 at 8.5K
  (computed; 59 measured), ~56 at 32K (computed from 11.32 GB), ~41 at 64K (computed from 12.88 GB; below 51).
- HCB (pre-registered 18:18; ~2.5 min, one 10-min lock): the 12 GB line with bf16 hyper-connection mixers (B12_HC=0, BigMoE's
  compiled HC) instead of HC8 (8-bit; TrainLab turn21: x1.092 of v3's mean KLD, code x1.244: 'not free'). Stack = MOV +
  ALLWAIT, C_m 51 / C_l 20, p2k + p8k, 256 tokens: h8 (shipped HC8) vs h0 (bf16). PASS: h0's lifetime peak <= 11.95 GB at
  both sizes AND warm tok/s >= h8's - 3 % (mean) -> bf16 mixers are an affordable quality step for 12 GB prompts up to ~8.5K
  (adoption = Kamen's call after an Eval-suite verdict). KILL: a peak > 11.95 GB at 2.6K, or < -5 %. Tokens vs h8: information.
- PFM (pre-registered 18:18; ~2.5 min, one 10-min lock): PFR on the MTP path (bt_b12 skips it there: 'an MTP feed'). Fresh
  process, p2k + p8k, 256 tokens, MOV + ALLWAIT, C_m 51 / C_l 20: pf0 (F12_PFR=0: s_moe 4096, pools resident) vs pf1 (PFR: one
  8,474-token MoE chunk; the MTP head's history then arrives in one feed instead of two). PASS (MTP-path port GO): every hash
  equal at both sizes AND 8.5K prefill >= 20 % faster. KILL: any hash differs (MTP-path PFR changes bits: the head's feed
  chunking would need a fix first). Prefill-phase peaks: information.
- HCB RESULT (18:18:04-18:20:25, one quiet lock; data/scr/hcb/{h8,h0}.txt): KILLED on speed. bf16 mixers (B12_HC=0) fit: peaks
  10.94 / 11.21 GB at 2.6K / 8.5K (HC8: 10.31 / 10.60, so +0.63 GB as computed), but warm decode is 15.29 -> 13.33 (-12.8 %) and
  17.15 -> 15.10 (-12.0 %) tok/s. Tokens change (8.5K sha32 4f1259dff2 vs HC8's 0b0d91cd43), as expected for a different
  mixer precision. The 12 GB line keeps HC8; a quality step there would need bf16-speed HC (MoonShots / Leap kernels).
- PFM RESULT (18:20-18:23, one quiet lock; data/scr/pfm/{pf0,pf1}.txt): PASS. PFR on the MTP path = PFR off on every hash
  at both sizes (2.6K kv 37028a6545 / head 8563204e11; 8.5K logits 7d045d7b7d, sha256 ed67050746, kv 5a760388d6, head
  f59755cf27) although the head's history came in one feed instead of two. 8.5K prefill 30.13 -> 21.20 s (-29.6 %),
  prefill-phase peak 11.54 -> 7.32 GB, lifetime peak 11.63 -> 10.59 GB; 2.6K prefill 10.06 vs 10.11 s (one chunk either way).
  Caveat: in pf0 the 8.5K decode started with pools still warm from the 2.6K decode, in pf1 with empty ones; they agree
  only because ALLWAIT makes the decode residency-invariant. On residency-dependent paths (the shipped single-row late rule;
  the AR path's frugal rule by design) PFR in a MULTI-prompt process starts each later decode cold instead of warm, which
  can move bits (and, on the AR path, tokens) vs PFR off. One-prompt-per-process A/Bs (Fit12's, Breakthrough's) cannot see this.
- MEM2 (pre-registered 18:24; ~1.5 min, one 10-min lock; /Users/kamen/Fit12/scripts/fit_mem2.py): census of the 12 GB stack's
  resident memory (load exactly as fit_cmp.py; then after a p2k prefill + 64 tokens): MLX active vs trunk pools vs the
  model's parameters by module, and the rest. At 2.6K decode MLX holds ~3.77 GB beside the 6.03 GB pools, vs ~2.6-2.9 GB
  estimated for weights + caches + scratch. PASS: >= 0.3 GB found that the decode math does not need (a duplicate, a
  bf16 copy of quantized weights, an unused module or buffer) -> a removal screen next (+~3 C_m per 0.3 GB). KILL: all accounted.
- MEM2 RESULT (18:24-18:26, two runs; data/scr/mem2/census.txt): KILLED, nothing removable. After load: MLX active 9.33 GB =
  parameters 8.37 + 0.96 outside them. The parameters hold the trunk's MSB base pools (switch_mlp gate / up / down, 5.27 GB)
  and 3.11 GB of other weights: the MTP head's expert pool 0.20 and ~2.91 GB of dense weights, all used every pass (GDN
  projections 1.03, HC8 mixers 0.63, lm_head 0.36, attention q / o 0.28 + k / v, shared experts 0.13, routers 0.06, MTP head
  dense, PLE / indexer projections). Outside the parameters: the LSB sidecar pools (~0.74 of resident_gb's 6.20) + ~0.22 GB of
  loader buffers; no Python-held arrays. After a 2.6K prompt + 64 tokens: +0.31 GB (KV, GDN state, decode buffers). My
  '~1 GB unexplained' (MEM2's pre-registration) was an underestimate of the dense weights (I had ~2.0 GB), not a hidden buffer. (Corrected
  18:27: a first version of this entry said 2.17 GB of non-pool weights, which double-counted the LSB pools.)
- AW1K (pre-registered 18:26; ~4 min, one 10-min lock): ALLWAIT over a long decode: s51 vs a51, p8k only, 1,024 greedy tokens,
  MOV, C_m 51 / C_l 20. PASS: sha1024 equal and a51's tok/s >= s51's - 1 %. KILL: any token difference or a cost > 2 %.
- AW1K RESULT (18:30; data/scr/aw1k/{s51,a51}.txt): PASS. 1,024 tokens at 8.5K (fresh p8k, MOV): a51 = s51 bitwise (sha1024
  e86066f456, kv 18280bc84c, head bed235c6e9); warm 16.44 -> 16.50 tok/s. With MOV on, the shipped rule has never differed
  from ALLWAIT (AW, AW1K); without MOV it did (CMPR r51 != CMP3 w51). So the deep copy may be what triggers the late rule.
- MVC (pre-registered 18:30; ~1.5 min, one 10-min lock): MOV alone, shipped rule (no ALLWAIT), fresh p8k, 96 tokens, LAYERS:
  v51 / v59 (C_l 20). Reading: v51 = v59 -> with MOV the shipped MTP path is C_m-invariant in this test (the copy was the
  trigger; ALLWAIT stays the guarantee); v51 != v59 -> MOV does not remove it, ALLWAIT is needed.
- MVC RESULT (18:32; data/scr/cmp/{v51,v59}.txt): MOV is not the trigger: v51 = r51 (kv e7b11d54c0), v59 = r59 (0a68624c42), so
  with MOV the shipped rule is still C_m-dependent at 96 tokens. What separates the runs is the decode budget: fresh p8k,
  shipped vs ALLWAIT, differ at 96 tokens (r51 vs w51) and are bitwise equal at 1,024 (AW1K s51 = a51). Hypothesis (not
  instrumented): single-row passes occur where the remaining budget leaves no room for drafts, i.e. at the end of a
  generation, so the late rule touches only the final token's pass: its logits (rarely the last token) and the KV left for a
  next turn. ALLWAIT removes it at no cost either way.

## Rapid screens, batch 2 (NewIdeas 19:05, QUEUE 'RAPID SCREENS' R-01..R-04; picked up Mon 26 Oct 21:43, written before any run)
Context: NewIdeas' 19:05 message waited while I was idle (orchestrator nudge). Disk 97 GB < the 100 GB floor (health ACT):
the runner (/Users/kamen/Fit12/scripts/screens.sh, SCREENS="gap kvc warm conf") waits for ACT and PAUSE to clear and starts
an arm only if it ends by 23:59 or after 08:01 (silent night 00:01-08:01). No big files: text outputs plus a 1 MB prompt corpus.
Common stack unless stated: MTP path, engine /Users/kamen/Fit12/v3p, MOV + PFR (cap 16384) + ALLWAIT (F12_BTC=1), C_l 20.
Note for NewIdeas: AW's peaks (65 / ~63 / ~56 / ~41) were ALREADY measured with PFR + MOV on (screens.sh scr_aw: F12_PFR=1),
so R-01 should find about the same slots, not ~1 GB more.
- R-04 GAP [port of MoonShots gapclock.py]: /Users/kamen/Fit12/scripts/fit_gap.py, random 4-bit weights at the engine's shapes:
  'layer' = one MoE layer's routed experts for 1 row (top-10 of 16 slots, 2560 -> 640 -> 2560, gs 64, gather_qmm), 'token' = 48
  such layers in one graph; each timed after spin and sleep gaps of 0 / 0.5 / 2 / 5 / 10 ms, 60 shuffled rounds per gap.
  LEVER if the 'layer' median after a 2 or 10 ms gap >= 1.20x back-to-back; KILL if < 1.05x at every gap. ~1 min.
  RESULT (Tue 27 Oct 19:18, g1, under a quiet lock, GPU 0 % at the grant): NO LEVER (between the lines; not pursued). 'layer'
  back-to-back 1.102 ms; after spin 0.5/2/5/10 ms 0.912/0.926/1.048/1.061x, after sleep 1.046/0.975/1.055/1.127x (max
  +0.14 ms after a 10 ms idle). 'token' (48 layers) 6.763 ms; after 2-10 ms gaps 1.069-1.096x (+0.5-0.65 ms per token
  after a gap). Wide p10-p90 at the 1-ms scale. The clock penalty after an SSD stall is <= ~0.65 ms: < 1 % of a ~60 ms
  decode step per stall. Output /Users/kamen/Fit12/data/scr/gap/g1.txt.
- R-02 KVC [new]: mtplx qsa_rows_target(n + NDEC + 16) around prefill + decode (F12_TGT=16): every QSA buffer is allocated once,
  instead of stock KVCache growth (+256 rows by concatenate: a full copy, ~65 MB per cache at 32K, 13 caches). p32k, 96 tokens
  (decode crosses one growth at ~token 90), C_m 51; k0 stock vs k1 target. PASS: decode-phase peak footprint -0.30 GB or more,
  all hashes equal (logits, sha96, kv, head), capacity line shows the target used. KILL: < 0.10 GB or any hash differs. ~4 min.
- R-03 WARM [new], screened as a bound before building the fill: p8k twice in one process (16 tokens, then 256), C_m 51.
  w0 PFR on: the 2nd decode starts with empty pools (as PFR ships). w1 PFR off: the 2nd decode starts with the pools the 1st
  run's 16 decode tokens left, i.e. the experts this exact continuation uses first, warmer than any prompt-count fill.
  Metric: 2nd run's first-128 decode rate (128 / (t_token128 - decode start)). PASS (build the real last-64 fill): w1 >= +10 %
  over w0 with equal hashes. KILL: w1 < +3 % (no fill can beat the oracle by much). ~3 min.
- R-01 CONF (confirmation, not a screen): per length L in 8.5K, 2.6K, 16K, 32K (in that order), 5 prompts c<L>_1..5 = seeded
  shuffles of a frozen corpus (/Users/kamen/Fit12/data/conf_corpus.txt: Breakthrough/scripts, SpecDecode, MoonShots/scripts
  .py files), run as [1,2,3,4,5,1] in one process: every prompt runs after another prompt, and prompt 1 also runs fresh.
  256 tokens each. Reference arm r51_L (C_m 51), then test arm at C_m(L) = 51 + floor((11.85 - P51) / 0.104) (cap 80), P51 = the
  reference's highest per-prompt clean peak (sampled footprint over release/prefill/restore/decode, hashing excluded).
  PASS: every hash (logits, sha256, kv, head) of every run = the reference run at the same position; prompt 1 fresh = warm in
  both arms; every clean peak (and the first run's lifetime peak) <= 11.95 GB; mean warm decode >= +4 % at 8.5K.
  KILL: any mismatch or peak > 11.95. Locks: 10 min per process (32K: 20 min). ~50 min of GPU in all.
> (a quote from the project owner, kept private)
  the experts"; orchestrator 18:44). It stays killed and is out of scope: no pruning or merging screens from now on.
- R-14 HC12 [port of Leap's R-09; NewIdeas 22:50 message; written Mon 26 Oct 22:36, before any run]. Finding first: bt_b12 runs HC8
  (B12_HC=8), i.e. MTPLX's EAGER HC chain with 8-bit down / up mixers and a bf16 inject [4, 10240]. Leap's X5 merges down +
  inject into ONE bf16 GEMM, which can't be bitwise here (a qmv and a bf16 gemv can't share one matmul without changing
  bits). So the screen is the price, step (1): F12_HCX=1 makes every GatedResidual read + write run twice (the copy on
  hyper + 0, its output added x 0, so the tokens stay the same and the copy can't be merged away), trunk + MTP head.
  p8k, 256 tokens, C_m 51, the batch-2 stack; 3 paired rounds h1 / h2 (stock / doubled); a call counter proves the patch
  engaged. Price = mean decode-time increase over tokens 33-256. KILL R-14 if the price < 1 % (no HC speedup can reach +1 %).
  If >= 3 %: HC is a lever, and step (2) is a bitwise HC8 read (e.g. the inject on its own stream, or the eager chain
  compiled, both checked against these hashes) as the next screen. In between: report. ~5 min.

### Batch 2 results (Tue 27 Oct 20:19; each screen under its own quiet lock after soft_end's ramp, PAUSE seen 0; hashes by position)
- R-04 GAP: NO LEVER (logged above, 19:18).
- R-02 KVC (19:48-19:51; k0 stock, k1 F12_TGT=16; p32k, 96 tokens, C_m 51): NOT A PASS (between the lines). Decode-phase clean
  peak 11.35 -> 11.23 GB (-0.12; PASS needed -0.30, KILL under -0.10); MLX decode peak 10.80 -> 10.74. Exact: all hashes equal
  (logits cbc97c2aee, sha96 ba0ccdbaac, kv 946633575a, head df07a098e9). Target used: kv cap 31765/31749 (stock 31997/31749).
  Decode unchanged (warm 14.20 vs 14.26 tok/s). Free and exact, about one slot at 32K: a fold candidate, NewIdeas' call.
  (k0 under the resume ramp, data/scr/kvc/k0_ramp.txt: same hashes, but clean peak 11.80 GB and 4.15 tok/s.)
- R-03 WARM (19:59-20:01; w0 PFR on = the 2nd decode starts cold, w1 PFR off = warm from the 1st run's 16 tokens): KILL.
  2nd run first-128 decode 16.24 -> 16.64 tok/s (+2.5 %; PASS needed +10 %, KILL under +3 %); warm 17.62 vs 17.68, all 16.98
  vs 17.22. Exact: hashes equal by position. Even an oracle-warm start is worth 2.5 % over 128 tokens: no fill to build.
- R-14 HC price (20:08-20:12; 3 pairs h1 stock / h2 every HC read + write run twice; p8k, 256 tokens): price 3.2 % (warm 17.07
  -> 16.54 tok/s, +1.87 ms per token; pairs 2.8 / 3.6 / 3.1 %; all-token 3.1 %, first128 3.3 %). Exact: hashes equal in all 6
  arms; 'HC calls doubled 11393' in each h2. Just over the 3 % line, so step 2 (a bitwise faster HC8 read) is allowed, but its
  ceiling is the whole price: R-14's PASS (+3 %) needs the HC to become free. NewIdeas' call; Fit12's view: low odds.
- Observation (R-03): the repeated p8k's 2nd position has a new 'logits' hash (61b2520e58; 629a4e74db at pos 1 and in every
  fresh p8k run) in both arms, while sha256, kv and head equal the fresh run's. fit_cmp hashes state['logits'] after the
  decode; under MOV the decode's head holds that same array. The pos-2 check decides whether the hash or the prefill moved.
- pos-2 check [written Tue 27 Oct 20:19, before any run]: p8k twice in one process (16 + 16 tokens), the R-03 w0 stack, MOV on (m1) and
  off (m0). New RESULT field 'lpre' = the same logits hash taken right after the prefill (own phase 'lpre', outside the clean
  peak; old fields unchanged). ARTIFACT if lpre is equal at both positions in both arms; LEAK if lpre moves in either arm
  (then R-01 is skipped and the leak goes to Breakthrough). ~3 min, one 10m lock.
- R-01 amendment [same time, before any R-01 run]: exactness vs the reference at the same position = sha, kv, head, lpre and
  logits; prompt 1 fresh = warm (pos 1 vs pos 6) = sha, kv, head and lpre (the post-decode 'logits' only if pos2 shows it
  repeat-invariant). The runner skips R-01 unless pos2 says ARTIFACT. R-01's locks were already one per arm.
- R-28 LAYSZ-X [NewIdeas QUEUE R-28 + 20:23 message; written Tue 27 Oct 20:25, before any run]: /Users/kamen/Fit12/scripts/fit_laysz.py,
  PLA's replay (fit_v3replay.Tier) with R-27's wide grid (C_m 1-231 step 10, C_l 2-80 step 2; PLA's was 24-96 / 8-40, so it
  could not reach R-27's 231-or-1 allocations). Policies LRU (d 0: what bt_b12 ships; it sets no BIGMOE_EVICT_LFU, and Fit12's
  v3p engine copy, BigMoE 7b8c7b4, has no LFU at all) and LFU 0.99 (BigMoE fast24's). Check shapes 4 / 3 (bt_b12's MTP path)
  and 16 / 10 (K1's 256.4 and R-27's). Per shape: (a) LFU vs LRU at uniform 51 / 20; (b) the Lagrangian per-layer allocation
  under LFU and (c) under LRU, both held out (leave one set out, mean of 4; PLA's 2-set folds too); R-27's published C_m with
  fitted C_l, in-sample, for reference. PASS / KILL (NewIdeas' row) on (b) held out: at 16 / 10 vs 256.4 (if this replay's
  16 / 10 LFU uniform is within 5 % of 256.4, else vs that uniform): PASS <= -15 %, KILL > -5 %. At 4 / 3 the same thresholds
  vs 4 / 3 LRU uniform (today's engine), which is what an engine A/B would test. CPU 1 core, ~4 min, one 10m quiet lock.
- R-28 CANCELLED by NewIdeas (20:29), before any run: MoonShots' strict replay (pool trimmed to C after every check) puts
  12 GB per-layer pools at -2 %, in line with PLA; R-27's -20.7 % came from wall24's forced admission keeping staged experts
  as free cache, not from LRU vs LFU. fit_laysz.py stays unrun. (20:41) pos-2 check started 20:40:57, then R-01.
- pos-2 check RESULT (20:40:57-20:42:38, one lock, rc 0, PAUSE seen 0): LEAK by its line, so R-01 was skipped. In both arms
  (MOV on m1, off m0) the logits hash taken right after the 2nd prefill already moves: lpre 629a4e74db at pos 1, 61b2520e58 at
  pos 2 (= the post-decode 'logits' field; MOV plays no part). Tokens (sha16 98dc29da32), kv (fe5cacaa98) and head (b8c2f8c195)
  are equal at both positions. So a repeated prompt's prefill gives other final logits while its KV and tokens are identical.
- pos-2 follow-up [written Tue 27 Oct 20:44, before any run; one 10m lock, ~3 min]: the same p8k twice (16 + 16 tokens) with the prefill
  logits saved at each position (F12_SAVE_LOGITS) in s1 = the fold stack (BTC + ALLWAIT + MOV + PFR) and s0 = stock (F12_BTC=0:
  Breakthrough's bt_b12 read-only, no ALLWAIT / MOV / PFR); /Users/kamen/Fit12/scripts/fit_pos2.py compares them: shapes,
  values that differ, max |diff|, argmax per row, last row, top-10. Verdict PRE-EXISTING if s0 moves too (then it is not the
  fold's, and R-01's fresh = warm can rest on sha, kv and head: NewIdeas' call); FOLD if only s1 moves (bisect next).
- New queue runner /Users/kamen/Fit12/scripts/runq.sh: screens named in /Users/kamen/Fit12/data/queue.txt run one at a time,
  with screens.sh sourced afresh before each (screens.sh's own main loop runs only when SCREENS_LIB is unset).
- R-01 criteria per NewIdeas' advance call (20:47) [written Tue 27 Oct 21:01, before any R-01 run]: if pos2b says PRE-EXISTING (or NONE),
  R-01 runs and prompt 1's fresh = warm is judged on token sha, kv and head only; by position vs the reference, all fields
  (sha, kv, head, lpre, logits) must still be equal. The pos-2 prefill-logits drift is then logged as a separate pre-existing
  finding, not a fold blocker. If FOLD: bisect first, R-01 waits. The runner sets LEAK from pos2b's verdict line.
  Caveat (Breakthrough 20:47): v3p's PFR release does not wait for in-flight prefetch / joint reads (v3m drains first). The
  hashes have held so far; any R-01 mismatch would point there first.
- Runner: arm()'s stop now kills the arm's whole process tree (kill_tree; orchestrator rule 27 Oct).
- pos-2 follow-up RESULT (21:18:12-21:20:13, one lock, rc 0 both arms, PAUSE seen 0): PRE-EXISTING. Stock bt_b12 (s0: F12_BTC=0,
  no ALLWAIT / MOV / PFR) and the fold stack (s1) give the same hashes at both positions (lpre 629a4e74db -> 61b2520e58; sha16,
  kv, head equal), and pos 1 is bitwise equal between them. The repeated p8k's prefill logits differ from the first in exactly
  20 of 248,320 entries, each by ~1e30: 20 vocab ids are masked (-1e30) at one position and real at the other; argmax equal,
  last-row top-10 differs. A logits mask carried over from the previous request (outside the fold); sent to Breakthrough with
  the files (/Users/kamen/Fit12/data/scr/pos2b/*.npy). Not a fold blocker (NewIdeas 20:47): R-01 runs, fresh = warm on sha,
  kv, head. To pin: which ids and which direction (a 1-s numpy look, blocked by the peers hook while others measured).
- R-01 CONF RESULT (Tue 27 Oct 22:11; 21:26-22:09, 7 arms, one lock each, rc 0, PAUSE seen 0, 6 RESULT lines each): KILL by its line, on
  memory at 32K only. The C_m 51 reference at 32K peaks 12.24 GB (clean; pos 4-6: 12.04 / 12.24 / 12.23; lifetime 12.35), so no
  32K test arm ran (the formula gave 47; the runner's awk int() rounds negative steps toward zero and printed 48: a bug, no effect).
  Everything else passes:
  * Exact: every hash (logits, sha32, sha256, kv, head, lpre) of t60_8k / t64_2k / t56_16k = the reference at every position.
    Prompt 1 fresh = warm (pos 1 vs 6) on sha, kv, head in all 7 arms; its logits / lpre differ (FR-Spec draft subset, pos2b).
  * Peaks of the test arms: 8.5K C_m 60 max 11.76, 2.6K C_m 64 max 11.81, 16K C_m 56 max 11.77 GB; first-run lifetimes <= 11.65.
  * Warm decode (tokens 33-256, mean of 6): 8.5K 15.06 -> 15.78 tok/s (+4.8 %; +4.5..+5.3 per position) PASS line met;
    2.6K 16.47 -> 17.75 (+7.7 %); 16K 15.64 -> 16.07 (+2.7 %).
  * New: clean peaks grow with position in one process, more at longer prompts, flat from pos 5 to 6: 2.6K +0.15, 8.5K +0.29,
    16K +0.40, 32K +0.90 GB (11.34 -> 12.24). MLX decode peaks stay flat (32K 10.82..11.08), so the growth is outside MLX's
    active memory. Cause open (candidates: PFR's pool re-creation each prompt, incl. Breakthrough's in-flight-read race; the
    harness's per-position numpy hashing; Metal/allocator retention). Multi-prompt C_m points: 64 / 60 / 56 / <= 47.
- GROW [proposed to NewIdeas Tue 27 Oct 22:12, before any run; queued only on GO]: c32k_1, c32k_2 in one process, 32 tokens each, C_m 51,
  the R-01 stack. g2 = kv / head hashing off (F12_NOHASH=1, new; logits / lpre stay); g3 = PFR off (hashing on). R-01's
  r51_32k is the reference for growth: clean peak pos 1 -> pos 2 +0.22 GB (11.34 -> 11.56). HARNESS if g2 grows <= 0.05 GB;
  PFR if g3 grows <= 0.05 GB; ENGINE (or allocator) if both grow >= 0.15 GB; else between. One 10m lock, ~7 min.
- Breakthrough 21:31 / 22:12: pos-2 root cause = FR-Spec's draft head is rebuilt with the current prompt's ids only when
  generate starts, after b12_prefill_state; draft-only (tokens, kv, head cannot change). Fixed in their bt_b12 (07fec4a);
  Fit12's btc copy does not have it.
- GROW revised [Tue 27 Oct 22:13, before any run; Breakthrough 22:12: 'flat once prompt 1 repeats' smells shape-keyed (Metal pipelines /
  compiled graphs / a per-length bank), not a leak]: R-01's stack, C_m 51, 32 tokens per prompt, hashing on as in R-01;
  g1 = c32k_1 twice, g2 = c32k_1 then c32k_2; MLX active / cache and footprint logged after each prompt. Growth = clean
  peak pos 2 - pos 1. SHAPE-KEYED if g1 <= 0.05 GB and g2 >= 0.15 GB; LEAK if g1 >= 0.15 GB; else between. One 10m lock,
  ~7 min. (The hashing-off / PFR-off arms are dropped.) Run within Fit12's line (the 12 GB setup's memory); NewIdeas told.
- pos2c RESULT (22:11, CPU): of the 20 ids that differ at the repeated prompt's 2nd prefill, 12 are masked (-1e30) at pos 1 and
  real at pos 2 (prompt ids outside FR-Spec's base subset, unmasked once the previous prompt's ids join it), and 8 are real at
  both with tiny drifts (e.g. 0.06152 -> 0.06177; the draft head's subset matrix changed shape). Top-10 ids are the same set,
  two near-equal draft candidates swap order. Draft-only, as Breakthrough found; their fix (07fec4a) makes both positions equal.
- GROW RESULT (22:18-22:24, one lock, rc 0, PAUSE seen 0): LEAK by its line (not shape-keyed). The same prompt twice grows the
  clean (decode-phase) peak 11.30 -> 11.51 GB (+0.21); two different prompts 11.26 -> 11.51 (+0.25). Hashes as expected (g1
  pos 2 = pos 1 on sha, kv, head). MLX active (9.427 GB) and cache (0) are identical after each prompt. The pointer: PFR's
  release at prompt 2 leaves the footprint at 8.03 GB vs 6.69 GB at prompt 1 (MLX active 3.33 both), and the re-created
  6.00 GB of pools land at 11.23 vs 11.02 GB. So memory released with the pools is not all returned (engine pool buffers or
  the allocator, not MLX). After-prompt footprint: same prompt +0.04, different prompt +0.26 GB.
- GROW2 [written Tue 27 Oct 23:03, before any run; NewIdeas' 22:15 GO included a PFR-off arm]: c32k_1 twice, PFR off, else as g1. PFR if the
  decode-phase peak grows <= 0.05 GB from prompt 1 to 2; ENGINE if >= 0.15 GB; else between. One 10m lock, ~4 min.
- GROW2 RESULT (23:06-23:10, one lock, rc 0, PAUSE seen 0): ENGINE by its line. With PFR off the same 32K prompt twice still
  grows the decode-phase peak 11.22 -> 11.53 GB (+0.31; MLX decode 10.75 -> 10.85), after-prompt footprint 11.117 -> 11.282
  (MLX active 9.427 -> 9.522). Hashes as expected (sha, kv, head equal). So neither PFR's release nor the harness's hashing
  (same in GROW's g1 and g2) is the cause: about 0.2 GB per prompt at 32K grows outside MLX's active memory, plateauing after
  ~4-5 prompts (R-01). Suspects: the C++ engine's read staging / free lists or residency tables, or Metal-side memory.
  For the fold: the 32K C_m point stays at R-01's plateau value (<= 47) unless the engine side is found and fixed.

## R-42 GROWFIX (NewIdeas QUEUE R-42 + 18:43 message; written Wed 28 Oct 18:47, before any run)
- /Users/kamen/Fit12/scripts/fit_gfix.py: Breakthrough's bt_b12 at its 31b63af defaults (engine v3m, B12_FOLD=1 fold-exact,
  B12_PFR_GRID=keep), read-only and unpatched, driven as its spec CLI (reset_generation_state, b12_prefill_state(512, 4096),
  spec125.generate 3:3:0); fit_cmp's MTP environment without the CLI-only BIGMOE_ASYNC_EVERY and B12_PFR=0. c32k_1..c32k_4 in
  one process, 32 tokens each (4 x 32K fits a 10-min lock per arm; ~18 min of GPU for both). Arm s = stock (nothing between
  prompts); arm t = after each prompt mx.clear_cache() + malloc_zone_pressure_relief(NULL, 0). No arm 3: v3m's ext offers no
  trim (only drain(), which waits for reads). Per prompt: token sha, prompt peak footprint (20 ms samples), footprint / MLX
  active / cache / lifetime peak after it.
- Growth G = prompt peak at prompt 4 - at prompt 1 (the creep over prompts 2-4); also the after-prompt footprint growth.
  PASS: G cut >= 70 % (1 - G_t / G_s) with every prompt's sha equal in both arms. KILL: cut < 20 % (live memory: a leak hunt,
  not a trim). Between: NewIdeas decides. No verdict if G_s < 0.10 GB (the new defaults do not creep at 32K).
- R-42 RESULT (written Wed 28 Oct 19:48; arm s 18:49-18:57, arm t 19:06-19:15, one 10m lock each, GPU 0 % at both grants, rc 0, PAUSE
  seen 0, 4 RESULT lines each; /Users/kamen/Fit12/data/scr/gfix/{s,t}.txt): KILL by its line. G_s = 11.442 - 11.290 =
  0.152 GB (>= 0.10, so a verdict), G_t = 11.433 - 11.286 = 0.147 GB: cut 3 %. Every prompt's sha equal in both arms
  (22451f8c02, 3bf29a5043, a6b9583d0d, eff21a96ea; c32k_1 = Tue's 22451f8c02). Time per prompt equal (s 124/123/119/132 s,
  t 123/124/120/133 s). The trim empties MLX's cache (0.07-0.10 GB) and leaves the after-prompt footprint 0.08-0.22 GB lower
  (s 10.106/10.196/10.170/10.187, t 10.024/10.063/9.953/10.089 GB; MLX active 9.332 throughout; pos 1 -> 4 s +0.081, t
  +0.065), but the next prompt takes it back. malloc_zone_pressure_relief freed < 0.5 MB each time, so the 0.62-0.86 GB
  outside MLX after a prompt (0.34 after load) is not libmalloc slack (engine, Metal-driver or other live memory).
  Caveat on G: c32k_4 is 1,681 tokens longer than c32k_1. Its KV in the 12 full-attention layers (2 heads x 256, K+V, bf16:
  24.6 KB/token) is ~0.04 GB, and a fit over the 4 prompts (peak = a + b n + c pos, residuals < 2 MB) gives 38 KB/token and
  0.029 GB per prompt, so the creep proper is ~0.09-0.11 GB over 3 prompts, close to the no-verdict line. The new defaults
  creep little: c32k_1 -> c32k_2 +0.044 GB (Tue's old-stack reference +0.22 GB was a decode-phase peak in fit_cmp, not the
  same metric). Lifetime peaks 11.443 (s) / 11.433 GB (t), under 11.95. Not pursued (KILL); a same-prompt x4 run (one arm,
  ~9 min) would split length from creep if NewIdeas wants it.
- NewIdeas 21:2x (seen Wed 28 Oct 22:20): R-42 KILL logged in QUEUE; the optional same-prompt x4 run is skipped, since the Eval suite's
  R-41C A runs the 29 agent prompts in one process on the 31b63af defaults and measures the creep at scale.

## R-45 CREEPMAP (NewIdeas QUEUE R-45 + 23:2x message, design Breakthrough; written Wed 28 Oct 23:30, before any run)
- Question: which suspect holds the per-row footprint creep of EvalSuite's fold-exact math process (R-41C: row starts
  10.12 -> 10.95 GB over rows 2-9, then -0.90 within row 9)? (a) the n-gram hot-row LRU (mtplx qwen4_exp _SidecarGather:
  MTPLX_NGRAM_HOT_MB 1024 per sidecar, cached rows are views into their whole fetch batch); (b) cyclic garbage (gen-2 GC).
- /Users/kamen/Fit12/scripts/fit_creep.py: one process as the suite's b12llfx decode stage runs it, read-only and unpatched:
  bt_b12 c3f957cba6 (sha256[:10] checked at start), load_b12(0.10, mtp=True); the suite's process env (cli.gpu's SHIP_AR +
  B12_ENV['b12llfx'] + SHIP_MTP + guard off; evalsuite/env.py's SHIP defaults: MTPLX_NGRAM_PREWARM off, compile switches 0,
  BIGMOE_SPARE 0, BIGMOE_PF_NOPUBLISH 1; MTPLX_NGRAM_HOT_MB unset); rows as decode._gen_mtp (reset, STASH / PENDH cleared,
  b12_prefill_state 512 / 4096, spec125 3:3:0 with stop_ids EOS, cut at the first EOS). Rows: the first 8 math rows of
  R-41C's fold-exact process in its order (prealgebra/1640 ... prealgebra/1973; 8,596 tokens, 468 s there), prompt ids
  from ~/EvalSuite/data/items/dec_math.jsonl.
- Per row k: tokens (sha1 of the id list) and equality with results/b12llfx/decode.jsonl's row; footprint at the start, row
  max (20 ms samples) and after the row (F_pre); MLX active / cache; every sidecar's hot cache: rows, rows x row bytes (what
  the cap counts) and real bytes H = the unique fetch batches the cached rows are views into; then gc.collect() with a
  DEBUG_SAVEALL census (objects, numpy / mx bytes reachable from the garbage; information only) and the footprint again
  (F_post); swap and compressor size. vmmap --summary after rows 1 and 8 (region types of the creep; information).
- Creep the suite would see over row starts 2-9 without this gc: C = F_post(8) - F_post(1) + B, where B = sum over rows
  2-8 of D(k) = F_pre(k) - F_post(k) (suspect b) and A = H(8) - H(1) (suspect a).
- PASS: A or B >= 80 % of C (then Breakthrough's fix after the freeze). KILL: neither >= 80 % (C_m points budget the creep).
  If A + B >= 80 % of C but neither alone: BETWEEN, NewIdeas decides. No verdict if C < 0.40 GB (the run does not show the
  suite's +0.83 GB) or if any row's tokens differ from b12llfx's (then the harness is not the suite's).
- Run: share lock (memory only; NewIdeas), 15m --cpu 400 --gpu 80, first try 08:10 Thu (after Leap's 08:02 quiet A/B has
  its lock; peer wait lets others' quiet locks and queued tickets go first), >= 16 GB free at the grant; one arm, cap 600 s.
- R-45 AMENDMENT (written Wed 28 Oct 23:37, before the screen; after three smoke tests in the scratchpad, 2-3 rows x 8-24 tokens,
  which are checks, not results): (1) D(k) is the gc-only drop: the footprint is read after a 1 s settle (smoke rows settle
  0.14-0.15 GB with nothing to collect), then gc runs; S(k) = end - settled is logged apart. (2) A counts the hot cache's
  full memory (data + the dict, keys, tuples and views: its rows are 100 B, their Python objects ~5x). (3) The census names
  the garbage's functions. Lines unchanged. Smoke observation: from row 2 on, gc finds 44-55 objects with 8 mx arrays =
  0.189 GB, from mtplx.frspec_draft._full_vocab_head's local class _FullVocabDraftHead (MLX active 9.521 -> 9.332 GB after
  gc); the hot cache is ~1-2 MB at 24 tokens / row.
- R-45 run change (Thu 29 Oct 00:28, before the screen): Kamen (Thu 00:27, via the orchestrator 00:28) lifted tonight's silent night
  until his stop; the orchestrator asked for R-45 now on the free GPU under the usual quiet lock (one 15m lock, cap 600 s)
  instead of a share lock from 08:10. Lines and measurement unchanged.
- R-45 RESULT (written Thu 29 Oct 00:39; 00:28:55-00:37:02, one quiet lock, GPU 0 % at the grant, rc 0, 485 s, PAUSE seen 0, swap flat,
  8 RESULT lines + SUMMARY; /Users/kamen/Fit12/data/scr/creep/c1.txt, vmmap c1_vmmap_r1 / r8): PASS (b), cyclic garbage.
  Tokens = the suite's b12llfx rows on all 8 (904 / 1536 / 614 / 1536 / 1186 / 856 / 1536 / 428; decode 470 s vs 468 s).
  Creep C = F_post 9.851 -> 9.979 (+0.128) + B 0.757 = 0.885 GB vs the suite's 0.833 over the same row starts (10.12 -> 10.95).
  B = gc drops rows 2-8: 0.095 GB on rows 2-7, 0.189 on row 8 = 0.757 GB = 86 % (>= 80 %). A = the n-gram hot cache, full:
  0.006 -> 0.036 GB = 0.030 GB = 3 % (56K rows of 100 B; 6 MB data, the rest Python objects). Unexplained 0.098 GB (11 %).
  The garbage each row: one mtplx.frspec_draft._full_vocab_head.<locals>._FullVocabDraftHead (22 objects: the class, its 3
  methods, cells, 4 mx arrays = 0.095 GB; two on row 8). _full_vocab_head (mtplx/frspec_draft.py:42, called at :228 when
  the FR-Spec draft head is built) defines the class per call; its methods' closure cells hold the pruned head, and the class
  is in a cycle (super()'s __class__ cell), so each old head lives until a gen-2 GC. It is MLX memory: active 9.427 at a row
  end -> 9.332 GB after gc (row 8: 9.522). The suite's -0.90 GB within row 9 = ~9 heads x 0.095, one automatic gen-2 GC.
  Also: the footprint settles 0.14-0.22 GB within 1 s after every row (same each row; not creep). With gc between rows the
  row max stays 10.64-10.83 GB (suite 10.64 -> 11.46 by row 8); lifetime peak 10.862 GB. Exact fixes (Breakthrough, after
  the freeze): gc.collect() between requests, or build the head once / a module-level class (mtplx is not ours).
- R-45F (confirmation of R-45's PASS on the arm over the line; written Thu 29 Oct 00:41, before any run; NewIdeas told, may redirect):
  fold-fast (b12llff: + B12_FOLD_AW 1, B12_FOLD_CM 1, B12_PFR_GRID balanced; bt_b12 c3f957cba6), the same harness and 8 rows
  (R-41C B: 440 s), tokens vs results/b12llff. CONFIRMED if B >= 80 % of C with tokens = b12llff's rows (fold-fast creeps by
  the same garbage); NOT CONFIRMED if B < 80 % (another route: then PFR's resize / its gc are the suspects). Reported too:
  the lifetime peak with gc between rows vs 11.95 GB and the implied C_m budget ((peak - 11.95) / 0.104 GB per C_m step,
  R-01's rule). One 15m quiet lock, cap 600 s.
- R-45F RESULT (written Thu 29 Oct 00:52; 00:41:36-00:49:07, one quiet lock, GPU 0 % at the grant, rc 0, 450 s, PAUSE seen 0, swap flat,
  8 RESULT lines + SUMMARY; /Users/kamen/Fit12/data/scr/creep/f1.txt): CONFIRMED. Tokens = b12llff's rows on all 8 (the
  same shas as fold-exact's on these rows). C 0.885 GB (suite 0.821 over the same row starts), B 0.757 = 86 %, A 0.030 = 3 %:
  the same garbage (one _FullVocabDraftHead, 0.095 GB per row; here from row 1 on). PFR resized to C_m 64 on row 1 (MLX
  active 9.33 -> 10.67 GB after gc). Bounded peak with gc between rows: row maxima 11.55 / 12.14 / 12.17 / 11.99 / 12.02 /
  12.15 / 12.09 / 12.11, lifetime 12.212 GB (row 3): still 0.26 GB over 11.95, so C_m 64 needs ~3 steps on these rows
  ((12.212 - 11.95) / 0.104 = 2.5: C_m 61 with the gc fix, vs ~52 without it; R-41C's peak was 13.20 GB).
- Residual after gc (both runs): F_post +0.127 GB over rows 1 -> 8, of which the hot cache is 0.030. vmmap (dirty + swapped,
  row 1 -> 8): MALLOC_LARGE +95 (fx) / +99 MB (ff), VM_ALLOCATE +17.5 MB (Python object arenas: the hot cache's objects),
  MALLOC_SMALL +6-8 MB; total +123 MB. About 14 MB per row of large mallocs that no gc frees (~0.6 GB over a 45-row process).

## R-45R RESIDUAL (proposed 00:52, NewIdeas GO 01:4x; written Thu 29 Oct 01:45, before any run)
- Who owns the ~14 MB per row that no gc frees (R-45 / R-45F: vmmap MALLOC_LARGE +95-99 MB, VM_ALLOCATE +17.5 MB over rows
  1-8)? fit_creep.py with F12_TRACE=1: fold-exact (b12llfx), rows 1-3, the same harness (gc after each row); tracemalloc
  started after load with 1 frame (numpy reports its data buffers to it); after row 1 and row 3 (after the gc and before any
  snapshot) vmmap --summary, then a snapshot summarized by file:line (the snapshot is dropped before the next row).
- G_vm = the growth of MALLOC_LARGE + VM_ALLOCATE (dirty + swapped) from row 1 to row 3; T = tracemalloc's traced growth
  over the same span; L = the largest single file:line growth. PASS: T >= 80 % of G_vm and L >= 50 % of G_vm (a named site
  for Breakthrough). KILL: T < 20 % of G_vm (native memory: the C++ engine or MLX; then a MallocStackLogging hunt). Else
  BETWEEN with the top sites. Tokens must still equal b12llfx's rows. One 15m quiet lock, cap 600 s.
- R-45R RESULT (written Thu 29 Oct 01:54; 01:46:43-01:49:49, one quiet lock, GPU 0 % at the grant, rc 0, 185 s, PAUSE seen 0; 3 RESULT
  lines + TRACE; /Users/kamen/Fit12/data/scr/creep/r1.txt, r1_vmmap_r1 / r3): BETWEEN by its line. Tokens = b12llfx's rows
  on all 3 (tracemalloc cost ~2 %: 52 / 83 / 39 s vs 51 / 81 / 39). G_vm rows 1 -> 3 = MALLOC_LARGE +43.1 + VM_ALLOCATE +5.5
  = +48.6 MB (MALLOC_SMALL +10.2). T = +11.0 MB traced = 23 % of G_vm (>= 20 %, < 80 %); L = qwen4_exp.py:5570 (the hot
  cache's tuple insert) +8.5 MB = 17 %. Next: memmap.py:359 +1.8 MB (the cache's fetch batches), qwen4_exp.py:5560 +0.5 MB;
  everything else < 0.05 MB. So the Python-visible residual is the n-gram hot cache (98 % of T); tracemalloc's own memory
  grew 5.3 -> 12.4 MB (in G_vm: net of it T = 27 %). About 37 MB over 2 rows (~18 MB per row) is native (malloc from C/C++
  or Rust, invisible to tracemalloc): the v3m engine, MLX (e.g. per-shape compiled graphs / kernels, as each row's prompt
  length differs) or the tokenizer. The footprint after gc moved -0.003 GB over these rows (R-45: +0.050): at this size the
  footprint is noise and vmmap is the instrument. A native hunt would be MallocStackLogging=lite + malloc_history.

## R-45S SHAPESPLIT (proposed to NewIdeas Thu 29 Oct 15:43; Breakthrough's 01:58 split; written before any run; runs only on GO)
- Is R-45R's native residual after gc (~18 MB per row) shape-driven? fit_creep.py, fold-exact (b12llfx), gc after each row,
  vmmap after every row (F12_VMALL=1). Arm s3: suite math row 3 (algebra/694, 614 tokens) three times (F12_ROWS=3,3,3);
  arm d3: rows 3, 10, 12 (614 / 571 / 404 tokens; three prompt lengths). Fresh process per arm, one quiet lock for both.
- G = MALLOC_LARGE + VM_ALLOCATE growth (dirty + swapped) from after row 1 to after row 3, per arm.
- PASS (shape-driven): G_s3 <= 20 % of G_d3 (per-new-shape native caches, bounded by the distinct shapes; fix route: shape
  buckets or a cache cap). KILL: G_s3 >= 80 % of G_d3 (a per-request native leak). Between: NewIdeas decides. No verdict
  if G_d3 < 15 MB. Tokens = the suite's rows each time (s3: all three = row 3's).
- NewIdeas GO for R-45S (15:5x, seen Thu 29 Oct 15:48); queued. Their queue for Fit12 next: R-46 HOTCAP (MTPLX_NGRAM_HOT_MB 1024 / 256 / 64), then R-47 BIGCACHE12 (after the fix + cap).
- RESULT (Thu 29 Oct 17:30-17:34; one quiet lock, GPU 14 % at the grant, top OjoX; s3 125 s, d3 110 s, rc 0): KILL, a
  per-request native residue, not shape-driven. Tokens = the suite's rows in all 6 (s3 ebee9e94c5 x3; d3 ebee9e94c5,
  baf3a1a4a9, 1e1e65333f).
  | arm | MALLOC_LARGE dirty r1 / r2 / r3 (MB) | regions | VM_ALLOCATE r1 -> r3 | G r1 -> r3 | after-gc footprint r1 / r2 / r3 (GB) |
  | s3 (3,3,3)  | 131.5 / 151.8 / 169.9 | 33 / 40 / 47 | 122.3 -> 122.3 | +38.4 MB | 9.852 / 9.876 / 9.893 |
  | d3 (3,10,12)| 132.2 / 151.8 / 171.4 | 33 / 40 / 47 | 122.3 -> 123.4 | +40.3 MB | 9.853 / 9.873 / 9.896 |
  G_s3 = 95 % of G_d3 (KILL line >= 80 %; G_d3 >= 15 MB, so a verdict). Swapped 0 everywhere. Both arms add 7 MALLOC_LARGE
  regions per request (~2.8 MB each, ~19-20 MB); after gc the footprint grows ~+21 MB per row in both, while MLX active
  stays 9.332 GB, tracked Python objects stay 103113-4 and the n-gram hot cache stays flat in s3 (7688 rows, 0.005 GB; d3
  0.005 -> 0.010 GB). So: native malloc, per request, the same for a repeated row as for new rows. Shape buckets or a cache
  cap won't bound it. Unknown: whether it levels off (R-45 was linear over 8 rows): over 45 rows ~0.9 GB if linear.
  Next (proposal to NewIdeas): R-45T MALLOCSTACKS, the call stacks of the 14 new large blocks (macOS MallocStackLogging=lite +
  malloc_history), to name the owner and route the fix.

## R-45T MALLOCSTACKS (proposed to NewIdeas Thu 29 Oct 19:10, GO seen 19:11; written before any run)
- Q: who allocates the 7 large blocks (~2.8 MB = one expert record: MSB 2,162,688 + LSB 770,048 B, NewIdeas) each request
  keeps? fit_creep.py, fold-exact (b12llfx), rows 3,10,12, gc after each row, bt_b12 ce6d4c63d2 (R-45 fix on; a pinned
  read-only copy of ~/Breakthrough/scripts/_bt_b12_gc.py at /Users/kamen/Fit12/pinned/bt_b12.py, sha logged),
  MallocStackLogging=lite. After rows 1 and 3: malloc_history <pid> -allBySize -fullStacks -q to data/scr/creep/t3_mhist_r*.txt;
  diff the live >= 1 MB allocations by call stack.
- PASS: one call stack (top 8 frames) holds >= 70 % of the new large-block bytes r1 -> r3: name the owner and route the fix.
  KILL: no stack >= 30 %. Between: report the top 3. Guards: tokens = the suite's rows; footprint < 36 GB; one quiet lock
  <= 15 min, memory-only (GPU <= 30 % after the grant). ~8 min.
- Grep before the run (19:12): v3m ext bigmoe_ext.cpp:118-122 read_expert keeps a per-thread aligned bounce
  (thread_local uint8_t* buf, posix_memalign 16384, len = the record or part length), freed only when a longer one is needed;
  a thread that exits leaks it (raw pointer, no destructor). R-45S's vmmap: Stack regions stay 73 at every row, so threads are
  replaced, not added. Prediction: the stack is read_expert -> posix_memalign, ~7 per request (new GCD workers).
- RESULT (Thu 29 Oct 19:55-19:57; one quiet lock, GPU 26 % at the grant, top WindowServer; t3 120 s, rc 0; bt_b12 ce6d4c63d2
  pinned, MallocStackLogging lite): PASS. Tokens = the suite's rows 3/3 (ebee9e94c5, baf3a1a4a9, 1e1e65333f).
  malloc_history -allBySize after rows 1 and 3 (88.5 MB each, 1-2 s), diffed by top-8-frame stack
  (/Users/kamen/Fit12/scripts/mhist_diff.awk -> data/scr/creep/t3_mhist_diff.tsv): of 43.4 MiB grown in all stacks,
  37.91 MiB (87 %; 93 % of the grown >= 1 MiB stacks) is ONE stack, +14 calls (7 per request):
  _malloc_zone_memalign < _posix_memalign < bigmoe_ext (unsymbolized, +36232) < bigmoe_ext (+48292) <
  _dispatch_client_callout2 < _dispatch_apply_invoke3 < ... _dispatch_root_queue_drain, i.e. a dispatch_apply worker in the
  v3m ext (its only posix_memalign: read_expert's thread_local bounce, bigmoe_ext.cpp:122). vmmap MALLOC_LARGE dirty
  134.7 -> 172.6 MiB (+37.9 = the stack's growth); Stack regions 72 at every row (workers replaced, not added). The rest:
  2 + 1 MiB of pymalloc arenas, < 1 MiB numpy. Owner: Breakthrough (v3m / v3n ext); BigMoE found and fixed the same in
  its 24 GB ext (shared bounce pool, ~/BigMoE 9f6c4f5, exact).
  Side note: with the fix on, row 1's gc still frees one _FullVocabDraftHead (0.095 GB, 4 mx arrays, 5 cells; D +0.095)
  and rows 2-3 free none: one head (likely the load-time one, built before _frs_nocycle wraps) stays until a gen-2 GC.
  With logging on, from row 2 malloc kept freed memory dirty: MALLOC_LARGE (empty) 168.8 MiB and MALLOC_SMALL (empty)
  112.8 MiB (absent in R-45S d3, same rows, no logging), so after-gc footprints jump 9.880 -> 10.198 GB r1 -> r2; R-45T's
  footprints are not comparable to R-45S's (the stack diff is unaffected: "(empty)" is free space, not live blocks).

## R-46 HOTCAP: revision proposed to NewIdeas (Thu 29 Oct 20:46; written before any run; runs only on GO)
- Facts first (R-45S + mtplx qwen4_exp.py:5317-5333, 5565-5580): the hot-row LRU caps ROWS, at MTPLX_NGRAM_HOT_MB x 2^20 /
  100 B (the counted row bytes); default 1024 -> 10.7 M rows. A cached row really costs ~620-650 B (key, tuple, one numpy view
  per map, the OrderedDict entry, its share of the pinned fetch batch): R-45S d3 16168 rows = 0.010 GB full, s3 7688 = 0.005.
  So the default lets the cache reach ~6.6 GB real before it binds; it grows ~4.2 K rows (~2.6 MB) per new request, so it
  binds only after ~2500 requests and stays <= ~0.12 GB in a 45-row run.
- So NewIdeas' arms 256 / 64 would not bind in a 24-row screen (64 MB = 671 K rows ~ 160 requests): no difference by
  construction. Revision, a cap that binds inside the screen: HOT_MB 1 (10.5 K rows; binds in request 2) vs 1024.
  Fold-exact (b12llfx, live bt_b12 ce6d4c63d2), R-41C math rows 1-4, one process per arm, A1 B1 B2 A2, gc after each row,
  per row: tokens vs the suite's row, decode tok/s, hits / misses, rows, full bytes. Two quiet locks (~7 min each).
- PASS: tokens identical; HOT 1's decode tok/s over rows 2-4 >= 99 % of 1024's (mean of the A and B pairs); its full bytes
  <= 8 MB. Then any cap >= 1 MB costs nothing: propose HOT_MB 16 (168 K rows, ~0.1 GB real) as the 12 GB default.
  KILL: >= 3 % slower: a HOT 16 arm next. Between: NewIdeas decides.
- 20:46: PARKED by NewIdeas before this revision was sent (R-46 matters only past a few hundred rows in one process);
  the facts above stand (cap counts 100 B / row, a row really costs ~620 B: the default binds at ~6.6 GB real).

## R-53 SCRATCH12 audit (NewIdeas, Thu 29 Oct; CPU code read, sent 20:50)
- Decode: 0 GB fresh scratch per decoded token (MTP 3:3:0 verify = 40 ids <= MAXK 80 -> sync-free -> tier.tier_call: no
  pool takes, no clear_cache). Prefill per 4,096-row MoE chunk: ~50 GB (48 x tier._prefill_grouped: 0.35 GB group double
  buffers + 0.63 GB outs / concatenate / take, then clear_cache tier.py:584) + ~11 GB for the MTP head's prompt pass
  (stream._sync_grouped, 8 x 0.18 GB takes per 512-row call) = ~60 GB, ~1.5 s per chunk at Leap's ~24 ms / GB.
  Prior: BIGMOE_TIER_PF_PERSIST=1 (the buffers part) -2.4 % prefill at 64K, exact (Breakthrough SCOREBOARD ~1126); MLX cache
  1 GB -1.9 %, peak +1 GB. Ceiling ~5-9 % of prefill on long prompts, 0 on decode. NewIdeas decides: A/B or skip to R-47.

## R-53 A/B pre-registration (Fri 30 Oct 00:44; NewIdeas GO Thu 20:50; written before any build or run)
- Engine: the live 12 GB path is now bt_b12 23b42972ef on v3n (records3m). v3n's Python prefill code (tier._prefill_grouped,
  stream._sync_grouped) = v3m's (diff: records dir + slot sizes only), so the A/B is built on v3n, not v3m.
- A = ~/Breakthrough/v3n as is. B = my copy /Users/kamen/Fit12/v3n53 (v3n tree, built ext copied unchanged, Python-only patch):
  (a) tier._prefill_grouped: the two group buffers persist across layer calls (grow-only to the largest group; smaller
      groups use slices, as now); dropped + mx.clear_cache() at the first tier.tier_call (decode) after a prefill. (Breakthrough's
      PERSIST test, SCOREBOARD ~1126: -2.4 % at 64K, exact, but decode kept the 0.36 GB: this frees them.)
  (b) tier._prefill_grouped + stream._sync_grouped: each group's output is scattered straight into its pair rows of one
      output array (MLX donates the buffer, so in place) instead of outs + concatenate + take (2 x 210 MB copies per trunk
      layer at 4,096 rows). Pure data movement: exact by construction.
  (c) stream._sync_grouped (the MTP head's prompt pass): MLX's cache limit raised to 256 MB inside the call only, so each
      group's scratch takes reuse the previous group's freed buffers; the 64 MB cap + clear_cache at the call's end, as now.
- Both arms: a pinned copy of bt_b12 23b42972ef (read-only, its defaults; B.V3F set to the arm's tree before load_b12), the
  suite's b12llfx env, MTP on. Prompts p8k and one 16K (16,384 tokens of ctx64k.txt), each prefilled by the live
  b12_prefill_state, then 32 spec125 3:3:0 tokens. Hashes: KV after the prefill, first-step logits, 32-token sha.
  Order A B B A (one process each), one quiet lock per <= 15-min chunk, behind 45 GB work.
- Measures: prefill seconds per prompt (mean of an arm's two runs), lifetime peak footprint, footprint after decode.
- PASS: every hash equal AND prefill -5 % or better on both prompts AND lifetime peak <= A + 0.05 GB AND decode footprint
  <= A + 0.05 GB. KILL: gain < 2 % on both prompts, or any hash differs (exact stays exact). Between: report, no default.
- AMENDMENT (01:43, before any run; lines unchanged): (b) in tier._prefill_grouped is now "after the groups, one
  synced scatter per group into one output array" instead of an async scatter chain. A scratchpad test (40,960 x 2,560 bf16,
  8 groups, MLX 0.32.2) showed that MLX donates the output buffer reliably only to a scatter issued after the previous one
  finished: async chains copied all of Y per group (peak 0.68-1.63 GB vs 0.63 GB for concatenate + take), while the synced
  scatters after the loop peaked at 0.42 GB, ~13 ms vs ~15 ms, bitwise equal. stream._sync_grouped already syncs per group.
- Smoke first (share lock, B arm only, p8k): the patch loads, its counters move, no error. Then the timed A/B.
- 01:44: no separate smoke run (the queue has no gap for a share lock): B1, the first patched arm, is the smoke test; scr_r53 = A1 B1 | B2 A2 in two quiet locks.
- AMENDMENT 2 (01:46, before any arm started; speed lines unchanged): each process first runs p2k (PFR's p2k, ~2K tokens,
  frozen at /Users/kamen/Fit12/data/prompts/p2k_src.txt), then p8k, p16k. Why: p8k and p16k take PFR (pools released during the
  prefill), so their prefill transient never sets the 12 GB peak; at <= 4,096 tokens the pools stay resident and the post-MoE
  transient does (EvalSuite R-41D: write14's +0.7 GB at 1,945 tokens). p2k is judged on memory only: its prefill-phase peak
  must stay <= A + 0.05 GB (guard); a lower peak is reported as R-53's memory effect. Its hashes must be equal too.

## R-53 A/B result (Fri 30 Oct 02:31; ran 01:58-02:27, two quiet locks; data/scr/r53/{a1,b1,b2,a2}.txt)
- All four arms rc 0. Engines as meant (a1 / a2 ~/Breakthrough/v3n, patch absent; b1 / b2 /Users/kamen/Fit12/v3n53, a 1 b 1),
  bt_b12 23b42972ef, BIGMOE_M01 1. Patch engaged in B (p2k: 2 allocs + 46 reuses of the group buffers, 341 scatters, 6
  MTP-head calls; p16k cumulative 6 / 186 / 1446 / 32). PF grouped 48 / 192 / 384 in every arm (no fallback).
- EXACT: every hash equal in all four arms. p2k kv 27aed6f97f h 90c78090b9 logits 7204d9848c sha32 82995cdb4e; p8k kv
  a27eaa9ec1 h 9091c3cf4c logits 90651cfda6 sha32 0b0d91cd43; p16k kv 719c80792e h 730b7ce740 logits 4a21736a94 sha32 3f218067f8.
- Heat pauses (fanwatch, FAN_CAP 3500): 02:01:56-02:04:18 froze b1 in its p8k decode (144 s); 02:24:08-02:26:50 froze a2
  in its p16k prefill (208 s). So b1 and a2 are void by the runner's rule (PAUSE seen). Clean: a1 and b2 only.
- Prefill s (n = 2617 / 8474 / 16384): a1 10.21 / 32.44 / 69.04 | b2 9.90 / 30.11 / 54.45 -> -3.0 / -7.2 / -21.1 %.
  Outside the rule (no pause in those prompts): a2 9.53 / 32.79 (before its pause); b1 p16k 53.11 (after its pause, cool chip);
  b1 p2k / p8k 12.28 / 43.00 ran before its pause but slow (cause unknown).
- CONFOUND: a1 started hot (die 80-88 C, right after MTP Head's lock); b2 started cool (die 58 C after a heat-pause cool-down).
  At FAN_CAP 3500 the die heats ~25 C within 2 min of load. The clean pair favours B; R-53's mechanism predicts ~5 % at
  16K (~37 GB fresh avoided per 4K chunk at ~24 ms / GB), not 21 %. So speed: NOT settled.
- Memory: neutral. p2k prefill-phase peak A 10.95 / 11.05 vs B 10.96 / 10.94 GB; MLX active peak 10.47 in a1, b1, b2 (the p2k
  transient sits where R-53 does not reach); serving peak A 10.95 / 11.05, B 10.96 / 10.94; lifetime peak A 11.05 / 11.05,
  B 11.01 / 11.09 (+0.04, inside +0.05); footprint after decode within +0.04 GB on every prompt.
- Verdict: EXACT + memory guard met; speed PASS on the only clean pair (-7.2 % p8k, -21.1 % p16k) but thermally biased:
  needs a confirmation with matched starting temperatures (R-53C below) before any adoption.

## R-53C confirmation lines (02:31, before any run)
- 4 fresh processes A1 B1 | B2 A2 (two quiet locks), p8k + p16k only (p2k showed nothing), same harness and engines. Each arm
  starts only when fanwatch's die max <= 55 C at thermal state 0 (cool-start gate, <= 5 min wait), so every arm gets the same
  thermal run. The harness flags any prompt whose prefill saw PAUSE ('paused 1': void, that prompt only).
- PASS: hashes = R-53's on both prompts, and with all 8 prefills clean, mean B <= mean A -5 % on both. KILL: mean B > mean A
  -2 % on both. Between: report numbers, NewIdeas decides. Memory guards as in R-53.
- Amendment (Fri 30 Oct 19:2x, before any R-53C arm ran): moved to the live engine. A = ~/Breakthrough/v3o, B =
  /Users/kamen/Fit12/v3o53 (v3o + the same R-53 patch: v3n's and v3o's tier.py / stream.py are byte-identical, v3o differs
  only in ext/), bt_b12 = Breakthrough's live 677c6558b6 (pinned copy /Users/kamen/Fit12/pinned/b677/bt_b12.py, from commit
  c83eae5: 23b42972ef + default engine v3o + B12_PFR_MIN env, default 4096 = unchanged). Hashes must still equal R-53's
  (Breakthrough a763814: v3o hashes = v3n's). Lines otherwise unchanged; a pass applies directly to the live engine.

## R-46 HOTCAP GO (Fri 30 Oct 02:35; NewIdeas 02:3x: "yes, run it (exact, env-only); the safety default for long-lived
## servers, the cap in real bytes"; R-47 DROPPED as an exact item: on fold-exact C_m changes the MTP path's bits)
- The Thu 20:46 revision above, moved to the live engine: fold-exact (b12llfx env) on bt_b12 23b42972ef (pinned copy
  /Users/kamen/Fit12/pinned/b23/bt_b12.py, F12_ENGINE=v3n, so records3m), fit_creep.py with F12_HOT_MB (MTPLX_NGRAM_HOT_MB).
- Arms: A = HOT 1024 (the default, set explicitly), B = HOT 1 (10.5 K rows: binds from request 2). R-41C math rows 1-4 in one
  process per arm, gc after each row; A1 B1 | B2 A2 in two quiet locks; cool() (die <= 55 C, state 0) before each arm.
  Per row: tokens vs the suite's row, tok/s (tokens / row wall, 2 decimals now), hot hits / misses, rows, full bytes.
- PASS: tokens identical on every row of every arm, HOT 1's tok/s over rows 2-4 >= 99 % of HOT 1024's (mean of the pairs),
  HOT 1's full bytes <= 8 MB. Then a real-byte cap costs nothing: propose HOT_MB 16 (168 K rows, ~0.1 GB real) as the 12 GB
  safety default, which Breakthrough sets. KILL: >= 3 % slower (then a HOT 16 arm). Between: NewIdeas decides.
- Amendment (Fri 30 Oct 19:2x; no valid R-46 data yet: a1 16:04 was VOID): moved again to the live engine, bt_b12
  677c6558b6 (pinned copy /Users/kamen/Fit12/pinned/b677/bt_b12.py, commit c83eae5) with F12_ENGINE=v3o (records3m, M-01
  on as before; v3o = v3n + the bounce-leak fix). Lines unchanged.
- Queue rule (NewIdeas 02:3x): yield to 45 GB items: never hold a grant while a 45 GB ticket waits behind it (screens.sh
  lock() + q45 since 02:35). R-53C's first request (02:32) was withdrawn by hand at 02:34, with Leap's R-40 4c behind it.

## S45b-2 (45 GB item; NewIdeas 16:1x, Fri 30 Oct 16:20, written before any run): is the misses-vs-C curve still falling past C 224?
- S45b (Leap, runner 38, Fri 00:55-01:32): C 224 vs 192: misses -24 %, all-6 mean +8 % tok/s, peak 40.5 GB; C 256 / 288
  stopped by the swap guard (+2.8 / +4.3 GB) with only 44-48 GB free.
- Harness: a copy of Leap's scripts/r40s3k_screen.py (sha 2e44b42a18; only the output dir changed) at
  /Users/kamen/Fit12/leap_s45/r40s3k_screen_f12.py, run as "<C> 256 speed": the live REF2b loop (4c key buckets merged),
  DFlash2 L1-20 drafting, S45's 6 prompts, 256 tokens each. ref2 / dflash / evalsuite imported read-only.
- Points: C 224 (the same-code baseline: S45b's C 224 predates the REF2b merge), C 256, then C 288 if it fits. One quiet lock
  per point. Gates: memory_pressure's free share >= 79 % (~50 GB; 85 % for C 288) before the load and again 30 s after the
  grant; PAUSE honoured (a point that saw PAUSE is void for tok/s); the sweep STOPS at the first swap trip (+2 GB).
- Per C: misses per check, SSD ms per check, tok/s code + math (+ all-6 mean), median cycle, kept rows per check, lifetime
  peak; tokens: == EvalSuite's stored REF2 rows (pre-buckets: may differ past 512 keys) AND == the C 224 point's tokens
  (C does not change bits on this path; that is the exactness check here).
- Answer, not a pass / kill: the misses and tok/s per C, and whether misses still fall past 224 (e.g. 256 vs 224 < -10 %).
- Amendment (Fri 30 Oct 18:4x, before any point ran): the copy's per-prompt 'S45:' line also prints 'tokens sha <sha1[:10]>
  (<n>)' of the 256 tokens (print only; nothing else changed), so the C 224 / 256 / 288 runs, separate processes, can be
  compared token for token.

## R-59 DENSE16 (45 GB item; NewIdeas QUEUE R-59, method OK 16:2x; Fri 30 Oct 16:25, written before any run)
- Why: R-50c (C 192): in a warm exact 16-row check the non-expert GPU work took ~52 ms (64.6 with syncs) while its weights
  (3.44 GB: HC 1.28, GDN 1.18, LM head 0.36, attention 0.36) read in ~12 ms at 280 GB/s.
- Harness /Users/kamen/Fit12/scripts/fit_r59.py = R-50c's setup (EvalSuite's REF harness via EM.load('ref', C), ref2.apply(),
  SB, PF1 prefill of HumanEval/68, warm 16-row checks sliding by one token along its REF2 row, commit keep 1), C 224.
  (1) In place: DecoderLayer.__call__ patched (eager at 16 rows; the compiled run path is only for <= 4 rows) to eval after
  each stage, per layer: PLE, HC read (attention side), GDN or QSA attention (+ its cache writes), HC read (MLP side, with the
  pending write), MoE block (router + shared + routed experts via ref2's slot path), HC write; plus the head (norm + LM head).
  Stage ms per check = sum of its intervals - n_syncs x a bare sync's latency; median over 10 warm checks.
  (2) Op groups (Leap's gdn16_profile method): the stateless stages from inputs captured in one check (HC reads, HC write,
  router, shared expert), each run for all 48 layers in one eval under ref2.verify_scope, median of 15 minus one latency.
  Floors: the weights each stage reads (its modules' parameter bytes) at 280 GB/s, and its GFLOP at 16 rows.
- Line (NewIdeas): any sub-stage >= 8 ms AND >= 3x its bytes floor -> a kernel row (exact only if bit-identical to REF2b's
  kernels; otherwise it folds into the next reference change). A measurement: no pass / kill for R-59 itself.
- Gates: memory_pressure free >= 79 % before the load (C 224 ~40.5 GB peak); PAUSE voids a run; one quiet lock (~5 min).

## R-57b-R LOOKAHEAD-RESERVED replay (45 GB item; NewIdeas 19:3x; Fri 30 Oct 19:4x, written before any data)
- Why: SlotStream (published, same 125B): look-ahead prefetch 1.111x, identical output, -21 % records read. Its forecast is
  layer L+1's router on layer L's post-attention state, plus a per-layer linear correction of the router logits (35 MiB
  FP16, fit on traces, agreement 0.80). Its reads land in RESERVED slots, so wrong guesses never evict. Our R-57 (BigMoE,
  17:00) used the raw forecast (top-96 ranked, LRU-tail victims): demand SSD -20 % but misses +10 % at C 192 (597 -> 658,
  evictions) -> KILL.
- Data: BigMoE's cap45 (45 GB live REF2b loop, 16-row checks; asked 19:4x for x at all 48 layers, every router, miss.bin
  at the capture's C, the PF1 prefill's effect on the pools). b45 cannot serve: x rows 0 in every prompt. CPU only, no
  lock, <= 4 threads. Script /Users/kamen/Fit12/scripts/r57b_replay.py, outputs /Users/kamen/Fit12/data/r57b/.
- (1) Recall per pair L -> L+1 (L = 0..46), x_L = layer L's MoE input (the router's input). (a) raw = router_{L+1}(x_L).
  (b) corrected = A_L raw + b_L (512 x 512 + bias, least squares, ridge chosen inside the training half), 2-fold by
  prompt: fit on half the prompts, test on the other half, and the reverse. Primary recall = per row |top10(forecast) ∩
  top10(true)| / 10, mean over rows and pairs. Also the 16-row union recall: |U top10 forecast ∩ U top10 true| / |U true|.
- (2) Cache replay: per layer LRU at C 192 / 224, mirroring the loader's ensure / make_room (needed experts protected, tail
  eviction, needed touched in row order). Baseline demand misses per check must match miss.bin at the capture's C before
  anything is scored. Then N reserved slots per layer (N 8 / 16 / 32): each check, L+1's top-N ranked non-resident
  forecast experts (rank = best row score) are read into the reserved pool (FIFO, kept until overwritten). A demand hit
  there saves a miss and is promoted as the miss would be, so the main cache stays the baseline's. Out per check: misses
  saved, wasted reads (prefetched, dropped unused), for (a) and (b). In-time completion is not modelled: an upper bound.
- PASS (NewIdeas): corrected recall >= 0.75 AND demand misses -15 % or better at C 224 with wasted reads <= misses saved
  -> BigMoE builds R-57b. KILL: recall < 0.6 or misses saved < 5 %. Between: numbers to NewIdeas.
- Prediction (before data): raw recall 0.6-0.7, corrected 0.72-0.80; at C 224 N 16 saves 15-25 % of misses.
- Amendment (19:5x, before any data; BigMoE 19:24 on cap45): the capture records every gated call of the live loop (the
  16-row checks AND the commit replays, each its own pass, x at all 48 layers), not the prefill. Pools are warm at the loop
  start and carry over between prompts; each npz has res0 (the resident set at the loop start), not the LRU order. So the
  replay serves every pass (commit replays included: they are demand too) and reports misses per 16-row check as S45 counts
  them (all misses / 16-row passes). Initial state per prompt = res0's set; residents kept from the previous prompt keep
  their order, the prefill's adoptions go to the front (approximation). The baseline check therefore becomes: total misses
  within 2 % of miss.bin's at C 192, with the per-(pass, layer) equality rate reported; C 224 starts from C 192's res0
  (another approximation, stated). Correction = a ridge residual on the raw logits (lambda -> inf gives raw + mean
  residual), lambda by leave-one-prompt-out inside the training half. Self-test PASS 19:5x (LRU mirror == a literal
  transcription of ensure() / make_room(); synthetic capture: router check 1.0000, miss.bin 9216 / 9216 equal with res0
  and commit replays; a learnable synthetic layer map: recall 0.016 raw -> 0.679 corrected).
- Amendment (Fri 30 Oct 21:1x, before any point ran; Leap 21:03): memory gates move from memory_pressure's free share to
  vm_stat's available pages (free + inactive + speculative + purgeable) >= the expected peak + 8 GB: C 224 49 GB, C 256 53,
  C 288 57 (R-59 49), before the load and 30 s after the grant. The free share counted other MLX jobs' wired Metal buffers
  as free (Leap's load passed a 76 % gate with ~14.5 GB really available and tripped the swap guard).
- R-59 amendment (Fri 30 Oct 23:4x, before any run; orchestrator 23:3x: C >= 192 tickets held while Kamen's apps leave
  ~37 GB available): run at C 96 (~24 GB peak, gate 32 GB available). What R-59 judges, the non-expert stages in place and
  the op groups, does not depend on C. moe_block at C 96 carries more SSD waits than at C 224: reported, not judged.
  Line unchanged.
- R-59 amendment WITHDRAWN (23:4x, nothing ran): C 96 cannot carry a 16-row check. BigMoE's ext ensure() aborts when one
  call needs more experts than C, and 16 rows x top-10 need up to 160 per layer. R-59 stays at C 224, held with S45b-2
  until the orchestrator says memory is back.
- Gate amendment (Fri 30 Oct 23:5x; orchestrator standing rule): every >= 10 GB load gates on vm_stat NO-SWAP (free +
  speculative + purgeable + file-backed pages) >= peak + 4 GB under a quiet lock (after the grant, 30 s settle), + 8 under a
  shared one; AVAILABLE (with inactive) let a C 224 load trip the swap guard at 23:51. R-59 at C 224: >= 45 GB after its grant.

## R-53C RESULT (Sat 31 Oct 00:05): PASS (exact; prefill -5.9 % p8k, -6.4 % p16k on the live v3o engine)
- Arms (data/scr/r53c/, cool starts die <= 55 C, PAUSE 0 everywhere): A = ~/Breakthrough/v3o, B = /Users/kamen/Fit12/v3o53
  (v3o + the R-53 patch), bt_b12 677c6558b6 pinned; A1 B1 23:10-23:15, B2 A2 23:58-00:03.
- Exact: on p8k and p16k every hash (KV a27eaa9ec1 / 719c80792e, h, draft logits, sha32 of 32 tokens) is equal in all four
  arms and equal to R-53's (v3n, Fri 01:58).
- Prefill, mean of 2: p8k A 31.66 s (31.91, 31.40) vs B 29.77 s (29.73, 29.82) = -5.9 %; p16k A 60.66 s (60.91, 60.42) vs
  B 56.79 s (54.56, 59.02) = -6.4 %. Line (<= -5 % on both, all 8 prefills clean): PASS. Caveat: B2 and A2 started with
  Kamen's UI on the GPU (WindowServer; GPU 46 % / 25 % at the arm starts); B2's p16k (59.02) is the slow one.
- Memory: prefill peaks B - A -0.37 GB (p8k) / -0.28 GB (p16k); decode and lifetime peaks equal (10.75-10.76 GB).
- R-53's clean pair (-7.2 % / -21.1 %) overstated p16k: its A arm started hot. The cool-start effect is ~6 %.
- Patch for adoption: /Users/kamen/Fit12/patches/r53_scratch_v3o.diff (tier.py + stream.py against v3o; switches F12_R53A,
  F12_R53B default 1, F12_R53C_MB default 256). Adopting it into the live engine is Breakthrough's / NewIdeas' call.

## R-46 HOTCAP RESULT (Sat 31 Oct 00:4x): PASS -> propose MTPLX_NGRAM_HOT_MB 16 as the 12 GB safety default
- Arms (data/scr/r46/; fold-exact b12llfx env, bt_b12 677c6558b6 pinned, engine v3o, math rows 1-4, one process per arm):
  A1 HOT 1024 (00:09; PAUSE seen: its timing is VOID, row 4 12.12 tok/s), B1 HOT 1 (00:16), B2 HOT 1 (00:27), A2 HOT 1024
  (00:33). B1 and A2 started with Kamen's UI on the GPU (38 % / 31 % at the start); B2 at 2 %.
- Tokens: rows 1-4 identical in all four arms (sha ca35b3618d, 8e41e434f1, ebee9e94c5, a6c1f1c8a3), each == the suite's row.
- tok/s rows 1-4: A2 17.38 18.51 16.04 18.00 | B1 17.92 19.37 16.10 17.96 | B2 17.81 18.92 15.84 18.22. Rows 2-4 means:
  HOT 1 17.74 (B1 17.81, B2 17.66) vs HOT 1024 17.52 (A2; A1 void) = 101.2 % (B2 vs A2 alone 100.8 %): line >= 99 % PASS.
- Hot table: HOT 1 holds at 10,485 rows, 0.007 GB full (<= 8 MB: PASS); HOT 1024 reaches 35,088 rows / 0.022 GB by row 4.
  Hits at row 4: 55,992 (HOT 1) vs 59,288 (-5.6 %), with no tok/s cost.
- So a real-byte cap costs nothing on these rows: propose HOT_MB 16 (~168 K rows, ~0.1 GB real) as the 12 GB safety default
  for long-lived servers (the uncapped table keeps growing with requests). Breakthrough sets it; NewIdeas' call.

## R-45U V3O RESIDUE (NewIdeas GO Sat 31 Oct 00:5x, low priority, shared GPU; written before any run)
- Q: what holds the ~3-6 MB each request still keeps after gc on v3o (Breakthrough a763814: +3.6 MB / request on agent
  rows; R-46 A2: after-gc +4 / +5 MB on math rows 3 / 4; B2 (HOT 1): +12 / 0)? R-45T's bounce leak is fixed in v3o.
- fit_creep.py, fold-exact b12llfx env, bt_b12 677c6558b6 (pinned b677) on v3o, HOT_MB 16 (the new default), math rows
  2,3,4,2,3,4 in one process, gc after each row, MallocStackLogging=lite. malloc_history -allBySize -fullStacks after rows 3
  and 6 (F12_MHIST_ROWS; the second pass brings no new content), vmmap --summary every row. Diff by top-8-frame stack
  (/Users/kamen/Fit12/scripts/mhist_diff.awk). Outputs /Users/kamen/Fit12/data/scr/r45u/.
- PASS: one stack holds >= 50 % of the bytes grown r3 -> r6: name its owner and route the fix. KILL: no stack >= 20 %
  (diffuse: allocator / pymalloc fragmentation). Between: the top 3.
- Guards: tokens == the suite's rows; shared lock (memory only, nothing timed); NO-SWAP >= 22 GB before the load (peak ~11
  + 8 + the stack log); swap guard. Logging keeps freed memory dirty (R-45T), so footprints are not comparable; the diff is.

## R-57b-R RESULT (Sat 31 Oct 01:1x): BETWEEN. Recall 0.693 (< 0.75) but demand misses -44 % at C 224 N 16 (wasted < saved)
- Data: BigMoE's cap45 (~/BigMoE/data/cap45, 01:09): the live REF2b loop at C 160 (spare 0, plain LRU, gate MAXK 160,
  DFlash2), S45's 6 prompts in one process (68, 1640, 91, 694, 1282, 708), x at all 48 layers, res0, routers w1..w47,
  miss.bin; tokens == EvalSuite's REF2 rows. Replay: /Users/kamen/Fit12/scripts/r57b_replay.py (self-test PASS), output
  /Users/kamen/Fit12/data/r57b/r57b_cap45.txt (+ .json), 414 s on efficiency cores (taskpolicy -b, 4 threads).
- Checks: baseline vs miss.bin at C 160: total misses +0.30 % (141,883 vs 141,465; the 2 % bar: PASS), 90.6 % of
  (pass, layer) pairs exactly equal (LRU order at the loop start is not exported). Router check: CPU top10(router_L(x_L))
  matches the capture's ids in 98.15 % of set slots, below my own 0.99 bar: f16-dequantized routers in f32 on CPU vs the
  model's own gate numerics. Every recall below carries ~2 % of that noise.
- (1) Recall of layer T's top-10 from layer T-1's MoE input, mean of 47 pairs: raw 0.605 (union 0.665); corrected
  (ridge residual, 2-fold by prompt) 0.693 (union 0.748). Per pair corrected: 0.393 (T 39) to 0.826 (T 47); T 1 0.508.
- (2) Replay at C 224 (misses per 16-row check; baseline 424.0):
    raw       N 8 / 16 / 32: saved 20.2 / 40.4 / 61.2 %, wasted 32.5 / 157.6 / 549.4 per check
    corrected N 8 / 16 / 32: saved 19.6 / 44.1 / 69.4 %, wasted 11.6 / 98.2 / 505.7
    oracle    N 8 / 16 / 32: saved 20.9 / 54.9 / 88.5 %, wasted 1.2 / 39.4 / 448.1
  C 192 (baseline 544.9): corrected N 16 saves 43.9 %, wasted 125.0 (< 239.3 saved).
- Line: PASS needs corrected recall >= 0.75 AND misses -15 % at C 224 with wasted <= saved. Recall misses (0.693); the
  misses half passes widely (N 16: -44.1 %, wasted 98 <= saved 187). KILL (recall < 0.6 or saved < 5 %): no. -> BETWEEN,
  NewIdeas decides.
- Reading: the RESERVED slots are what fix R-57. The raw forecast alone saves 40.4 % at N 16 (wasted 157.6 <= 171.2),
  where R-57's raw forecast into LRU-tail victims raised misses +10 %. The correction mainly cuts wasted reads (158 -> 98)
  and reaches 80 % of the oracle's saving at N 16. Upper bound: in-time completion is not modelled. At N 16 that is
  ~285 prefetch reads per check, ~6 per layer, against ~2.5-4 ms of compute per layer.
- S45b-2's question from the same baseline (no prefetch, misses per 16-row check): C 160 713 (the capture; loader 711),
  192 545, 224 424, 256 345, 288 297, 320 268. Still falling past 224: 256 vs 224 -18.7 %, 288 -30.0 %, 320 -36.7 %.
  C > 160 starts from C 160's res0 (the real prefill adopts more): the large-C values are slightly pessimistic.
- Export for BigMoE's R-57b build (NewIdeas GO 01:2x on BETWEEN): /Users/kamen/Fit12/data/r57b/corr.npz (all 6 prompts;
  in-sample recall 0.79, held-out 0.69), corr_fit68_91_1282.npz and corr_fit1640_694_708.npz (the 2-fold maps: give each
  A/B prompt the map that did not see it). Keys W1..W47 [512,512] f16, b1..b47 [512] f32: corrected = raw_logits @ W + b,
  rank = max over rows of log_softmax(corrected). f16 vs f32 recall within 0.0004 per layer. r57b_replay.py --export.

## R-59 RESULT (Sat 31 Oct 02:22; one quiet lock, NO-SWAP >= 45 GB after the grant, GPU 0 %, PAUSE 0): 2 kernel rows
- /Users/kamen/Fit12/data/r59/r59_c224.txt (+ .json): EM.load('ref', 224), REF2 + SB (key buckets on), HumanEval/68, 10
  warm 16-row checks (36 GDN, 12 QSA, 1 PLE layers). Lifetime peak 39.0 GB. Bare sync latency 0.257 ms.
- (1) In place, ms per check (median; each stage's syncs x latency subtracted) | bytes floor at 280 GB/s:
  PLE 0.75 | 0.23 (3.2x); HC read attn 7.22 | 2.26 (3.2x); GDN 24.07 | 4.20 (5.7x); QSA attention 11.62 | 1.28 (9.1x);
  HC read mlp 7.95 | 2.26 (3.5x); MoE block 143.30 (routed experts + SSD waits at C 224; no floor); HC write -0.66 (~0);
  head 2.29 | 1.32 (1.7x). Non-expert sum 53.1 ms. Check 260.6 ms with 241 forced syncs (R-50c unforced: 181.5 ms at C 192).
  The stages add to 196.5 ms = 260.6 - 241 x 0.257 ms.
- (2) Op groups (48 layers in one eval): HC read attn 4.63 | 2.26 (2.0x); HC read mlp 4.53 (2.0x); HC write 0.36; router
  1.09 | 0.24 (4.6x); shared expert 2.48 | 0.47 (5.2x).
- Line (>= 8 ms AND >= 3x the floor -> a kernel row): GDN (24.1 ms, 5.7x) and QSA attention (11.6 ms, 9.1x; the 16-row
  SDPA runs head-chunked, 12 chunks of 2 heads) are kernel rows. The HC reads (7.2 / 8.0 ms in place, 4.6 as op groups)
  sit just under 8 ms. A kernel row is exact only if bit-identical to REF2b's kernels; otherwise it folds into the next
  reference change. Owner: Leap (KERNELS).

## R-45U RESULT (run Sat 31 Oct 08:01-08:08, analysed Sun 1 Nov 01:1x): PASS. The residue is v3o's per-worker bounce buffers
- u6 (data/scr/r45u/): rows 2,3,4,2,3,4, rc 0, PAUSE 0, every row's tokens == the suite's; malloc_history after rows 3, 6.
- Live allocations r3 -> r6 (3 requests, no new content; mhist_diff.awk -> u6_mhist_diff.tsv): +2.83 MiB in 62 growing
  stacks, nothing shrank. ONE stack holds 2,785,280 B = 93.9 %: _posix_memalign in a dispatch_apply worker of bigmoe_ext
  (R-45T's stack: read_expert's bounce), +1 block of 170 x 16 KiB = one 16 KiB-aligned expert record (MSB 2,048,000 + LSB
  720,896 B). 74.1 MiB of such blocks are live at row 6 = ~28 workers x 2.66 MiB. Next: GCD continuations +67 KB, the
  logging's own table +32 KB, Metal buffer objects +9 KB.
- vmmap r3 -> r6: MALLOC_LARGE 109.6 -> 112.3 MiB (+2.7 = that block), Stack +1 thread, IOAccelerator 8.8 G unchanged,
  MALLOC_SMALL +4.4 MiB mostly empty regions (39 -> 68; logging keeps freed memory dirty, R-45T).
- Reading: not a per-request leak. v3o frees a worker's bounce at thread exit, but every live GCD worker holds its own,
  so the memory follows the worker pool: +1 worker over 3 requests, ~74 MiB at ~28 workers.
- Fix (exact: the bounce only carries bytes in transit): a shared bounce pool sized to the read queue depth, as BigMoE's
  24 GB ext did (~/BigMoE 9f6c4f5). It caps the bounce memory at QD x 2.66 MiB (~11-21 MiB) and ends the growth: ~-55 to
  -63 MiB on the 12 GB line. Owner: Breakthrough (v3o / v3p ext).

## M-5 DECISION-5 MEMORY CHECK (NewIdeas Sun 1 Nov 01:0x, "FIRST"; written before any run)
- Q: does the live 12 GB default (bt_b12 cf8bd7840a: fold-fast by default, engine v3p, HOT_MB 16) stay <= 11.95 GB
  lifetime on R-41C's rows? Kamen adopted decision 5 on b12llff3's 11.51 GB (its 29 agent rows; the suite's fold-exact math
  rows reached 11.51). Breakthrough's R-60 stock arm on the live file reached 11.15 / 11.37 / 11.38 GB on p2k / p8k / write16,
  then 12.00 -> 12.31 -> 12.43 GB through long math decodes (600-1536 tokens), under heavy system swap.
- fit_creep.py F12_ARM=live: the suite's process env, as R-60 ran it, with engine, fold pieces and grid left to the live
  file's defaults (pinned copy /Users/kamen/Fit12/pinned/cf8b/bt_b12.py, git show eaff23d). Two processes:
  (a) R-41C's 16 math rows in order (refs: b12llfx = fold-exact, so token equality is information only),
  (b) its 29 agent rows in order (refs: b12llff3 = fold-fast; tokens should be equal).
  Per row: tokens, footprint start / max / end, lifetime peak.
- Gates: shared lock (memory only, nothing timed); NO-SWAP >= 20 GB and no other process > 8 GB at the start; the swap
  guard (+2 GB) stops an arm. A big process appearing mid-run is logged and voids the arm.
- PASS: both lifetime peaks <= 11.95 GB, so the claim holds. FAIL: either > 11.95 -> to the orchestrator as a correction of
  decision 5's memory number (a Kamen-facing fact).
- Prediction: agent ~11.5 GB (= b12llff3); math ~12.2-12.4 GB (R-60's long decodes) -> likely FAIL on math.

## M-5 RESULT (ran Sun 1 Nov 01:14-01:50, analysed 07:4x; /Users/kamen/Fit12/data/scr/m5/{math,agent}.txt): **FAIL (math)**
- Valid: both arms started with NO-SWAP >= 20 GB and no other process > 8 GB, none at the end, swap flat 8.29 GB (guard
  not tripped). Tokens 45/45 = the suite's (agent 29/29 = b12llff3; math 16/16 = b12llfx's fold-exact refs, info only).
  Peak = proc_pid_rusage lifetime max phys_footprint / 1e9, the suite's measure.
- (b) agent, 29 rows: lifetime **11.533 GB** (row maxima 11.26-11.53) = b12llff3's 11.51: holds.
- (a) math, 16 rows: lifetime **12.353 GB** > 11.95: FAIL. No creep (row starts 10.71-10.91 after row 1's 9.67, ends
  10.99-11.13): a transient inside rows, +0.3 to +1.5 GB above each row's start (>= +0.6 on 15 of 16), largest on the
  short prompts: n 50-98: maxima 12.13-12.34 on 11 rows, 11.89 (r4), 11.05 (r1, cold start 9.67); n 148 / 249 / 520:
  11.56 / 11.52 / 11.19. It does not scale with decode length (r8 428 tokens 12.13, r13 382 tokens 12.29; r9 1536 tokens
  11.19). Prediction was right (math ~12.2-12.4).
- vs earlier: b12llff2's math process 11.73 (R-41D), FFX 8 math rows 11.68, R-60's live arm 12.43-12.50 (under swap).
  So since b12llff2 something added ~0.6 GB on math rows. b12llff3's math rows were never measured (only its 29 agent rows),
  so the cause could be b12llff3 (677c6558b6), v3p (R-53) or HOT_MB 16. Next step if wanted: M-5b bisect, rows 3 + 13 on
  pinned b677 / live with B12_ENGINE=v3o / live with HOT_MB 1024 (~20 min shared).
- Decision 5's 11.51 GB holds for agent rows only; on math the live default is ~0.4 GB over the 11.95 line.

## R-45U correction (Breakthrough R-45V, Sun 05:24, SCOREBOARD 8b6bfd1)
- v3q = v3p + BigMoE's shared bounce pool: exact, but only -16 MB at request 6 (10.727 vs 10.743 GB), not my -55-63 MiB.
  The pool holds 26 buffers from request 1: the engine runs ~26 expert reads at once, not QD 4-8 as I assumed, so a pool
  holds about one buffer per live worker. Not adopted (a hard cap would serialize reads). v3p's MALLOC_LARGE stayed flat
  over 6 fast-mode requests, so its +4.3 MB per request residual isn't bounces.

## M-5b BISECT (NewIdeas 07:5x GO; written before any run)
- Q: which piece of decision 5's bundle (eaff23d = 677c6558b6 + fast-mode defaults + engine v3p + HOT_MB 16) raises the
  math rows' in-row peak? One process per arm, math rows 8 (warm-up: the first row takes the C_m-resize PFR path),
  then 3 and 13 (M-5's 12.340 / 12.288), fit_creep.py as M-5 plus a PEAKAT line per row (prefill-phase max, the row max's
  phase and time, MLX active / cache at that max, MLX peak). Arms: A live (cf8b) | B pinned b677 + b12llff3's env (ARM ff3)
  | C live + B12_ENGINE=v3o | D live + MTPLX_NGRAM_HOT_MB=1024 | E live + B12_MODE=exact. Shared locks, NO-SWAP >= 21 GB,
  no other process > 8 GB (checked before and after each arm), swap guard +2 GB. Out: /Users/kamen/Fit12/data/scr/m5b/.
- Line (NewIdeas): the arm whose rows 3 / 13 maxima drop >= 0.5 GB vs A is the cause. Only E drops -> fold-fast itself
  (AW / PFR_MIN 512 / C_m points) -> NewIdeas reopens decision 5's memory claim. Nothing drops -> A's peak didn't
  reproduce or the cause is outside the bundle -> report.
- Stage 2 (only if E drops >= 0.5 GB, to name the smallest fix): F live + B12_FOLD_AW=0, G live + B12_FOLD_CM=0 (fast
  mode otherwise). PFR_MIN 512 can't matter here: rows 8 / 3 / 13 are under 512 tokens (PFR only on a C_m resize).
- Prediction: A ~12.2-12.3 on rows 3 / 13, the max in the decode; B high too (b12llff3's math rows were never measured);
  E drops (fold-fast's ALLWAIT or C_m 60); C and D don't.

## M-5b RESULT, stages 1-2 (ran 09:28-10:39, analysed 11:4x; /Users/kamen/Fit12/data/scr/m5b/{A..G}.txt): **cause = v3p (my R-53)**
- Rows 3 / 13 maxima (GB): A live 12.327 / 12.231 (= M-5's 12.340 / 12.288) | B b677 + b12llff3 env (v3o) 11.645 / 11.610
  | C live + v3o 11.641 / 11.612 | D live + HOT 1024 12.331 / 12.198 | E live + exact 11.398 / 11.255 | F live + AW 0
  12.323 / 12.205 | G live + CM 0 11.403 / 11.270. Tokens: all 21 rows = the suite's refs (same shas in every arm).
- Two pieces: (1) R-53 (v3p vs v3o, B = C): +0.69 GB, inside the PREFILL of short prompts read with resident pools (the
  max at 2.2-3.2 s, before the decode; MLX active at the max 11.815 vs 11.130). (2) fold-fast's C_m 60 (E = G, CM 0):
  pools +0.93 GB from row 1 on (starts 10.71-10.81 vs 9.78-9.88), the C_m 51 -> 60 that FFX planned. HOT_MB and AW: none.
- With v3o, fast mode's rows 3 / 13 peak at 11.61-11.65 GB: under 11.95, so decision 5's config (b12llff3) was fine;
  the bundle's v3p broke it. My R-53C called R-53 "memory neutral": true on its p8k / p16k PFR prefills (pools
  released), wrong on short prompts read beside resident pools, which R-53C never ran.
- Caveat: A ran beside two other 11 GB Python processes and 16.5 GB of system swap; D and F (no relevant change)
  reproduce A within 0.03 GB, so A stands. get_peak_memory after reset_peak_memory reads ~0.5-1.2 GB below the
  sampler's get_active_memory in every arm (unexplained; the footprint is the measure).

## M-5b stage 3 (11:4x, before any run): which R-53 part
- Arms (same rows, same harness): H live + F12_R53A=0 (group buffers per layer again), I live + F12_R53B=0 (no scatter
  into Y), J live + F12_R53C_MB=0 (no 256 MB cache limit in _sync_grouped). The part whose arm drops rows 3 / 13's mean
  max >= 0.5 GB vs A is the cause; its switch at 0 is the smallest fix (R-53's speed claim was p8k / p16k: re-check there
  only if the fix touches that path). None drops -> the parts interact: report v3o (R-53 off) as the fix.
- Prediction: H drops (two G-expert group buffers held across the chunk's layers, ~0.5-0.6 GB at G 64), I and J don't.

## M-5b stage 3 RESULT (ran 11:40-12:38; data/scr/m5b/{H,I,J}.txt): **R-53 part a alone** (prediction right)
- Rows 3 / 13 maxima, lifetime (GB): H R53A=0 11.636 / 11.647, 11.659 (= v3o's 11.641 / 11.612; MLX active at the max
  11.130 / 11.063 = v3o's exactly) | J R53C_MB=0 12.324 / 12.190, 12.337 (= A) | I R53B=0: row maxima VOID (my sampler
  thread died at start: mlx.core was in sys.modules while still importing; fixed 12:4x, the other 9 arms' samplers ran),
  lifetime 12.341 (= A), so b is not it either. Tokens equal on all 9 rows. H and I ran beside a 9.3-9.6 GB process.
- Mechanism: part a keeps the two G-expert group buffers alive across all 48 layers of a MoE chunk (dropped after layer
  47). A short prompt is read with the pools resident (no PFR), so they stack on the late-prefill transient v3o also has
  (max at 2.2-3.8 s of the prefill). In R-53C's PFR prefills the pools were released, so the same buffers fit there.
- Smallest fix: F12_R53A=0 (exact), with parts b and c kept. Part a's share of R-53's prefill gain is unmeasured. My
  estimate is small: it saves zero-filling ~0.5 GB per layer, ~0.1 s per chunk, vs R-53C's -1.9 s (p8k) / -3.9 s (p16k).
  A finer fix: part a only when the pools are released (PFR on).

## M-5d (Breakthrough 11:5x ask; before any run): the new live default's math lifetime
- Live bt_b12 = 31e4333196 (db49989, Sun 11:43): cf8bd7840a with the default engine v3o again (diff: that line + docs),
  pinned /Users/kamen/Fit12/pinned/31e4/bt_b12.py. M-5's math arm unchanged otherwise (16 rows, one process, suite env,
  shared lock, NO-SWAP >= 21 GB, no other process > 8 GB). Out: /Users/kamen/Fit12/data/scr/m5d/math.txt.
- PASS: lifetime peak <= 11.95 GB, so the 12 GB line holds again on math. FAIL: > 11.95 -> report the rows over the line.
- Prediction: ~11.65 GB (M-5's rows minus R-53's ~0.69 GB; arm C's rows 3 / 13 11.64 / 11.61).
- Amendment (12:5x, before any run): Breakthrough shipped option (1) instead. M-5d runs on live 8db71cc537 (3676db3,
  pinned /Users/kamen/Fit12/pinned/8db7): default engine v3p + os.environ.setdefault("F12_R53A", "0") at line 108,
  before any engine import (+ the opt-in B12_PMIN, unset here). fit_creep now prints the engine's R-53 parts after load
  and its R-53 counters at the end, to confirm part a is off. Same lines and prediction (arm H: 11.64 / 11.65).

## M-5d RESULT (ran 12:42-13:00; /Users/kamen/Fit12/data/scr/m5d/math.txt): **PASS, lifetime 11.774 GB** (line 11.95)
- Live 8db71cc537 (v3p + F12_R53A=0): the engine ran with R-53 a False / b True, counters alloc 0 (part a never ran),
  scat 1529. Tokens 16/16 = the suite's. No other process > 8 GB at the end; swap 14.4 GB (system), guard not tripped.
- Row maxima 11.05-11.76 (M-5 on v3p: 11.05-12.34); 14 of 16 in the prefill, 2 in the decode. The highest, r10 11.757,
  comes 0.11 s into its prefill: MLX active +0.88 GB over the row start (11.141 vs 10.259). The same early transient
  set v3o's r13 max at 0.10 s. It is now the binding peak (0.19 GB under the line): the next 12 GB memory lever if one
  is needed. Prediction (~11.65) was 0.12 GB low.
- So the live 12 GB default meets the 11.95 GB line on math again (11.77). Agent: 11.53 in M-5, measured on v3p with
  part a ON; not re-run with it off (all agent prompts are > 512 tokens and take PFR, where part a fit in R-53C).

## R-53D (NewIdeas 12:4x: by day, shared lock, ABBA; written before any run): does v3p + R53A=0 keep R-53's prefill gain?
- A = ~/Breakthrough/v3o, B = ~/Breakthrough/v3p with F12_R53A=0 (the live default). fit_scr53.py (R-53C's harness) on
  the live file 8db71cc537 (pinned 8db7) with F12_LIVE=1, so its fast-mode defaults apply (fold pieces, grid balanced,
  PFR_MIN 512): p8k + p16k, each prefilled by b12_prefill_state (timed), hashed (KV, MTP cache, h, draft logits), then
  32 tokens. Order A1 B1 | B2 A2, cool starts (cool 55 300), two shared locks, NO-SWAP >= 21 GB. Timings "shared GPU, rough".
- Exact: B's hashes = A's on both prompts, else stop and report.
- Ship rule (Breakthrough's, via NewIdeas): B keeps >= 3 % of the gain (mean B <= mean A -3 % on both prompts) -> keep
  v3p + R53A=0; B >= A +3 % on both -> back to v3o; B within +-3 % of A on either prompt (too close for shared noise) ->
  a quiet ABBA decides, with the same >= 3 % rule.
- Prediction: B -4 to -6 % on both (part a only saves the group buffers' zero-fill, ~0.1 s per chunk).
- Round 1 (13:05-13:49; moved to /Users/kamen/Fit12/data/scr/r53d_void1): NO VERDICT. Exact: kv / h / logits / sha32
  equal in a1, b1, b2 on both prompts, and the KV hashes = R-53C's (p8k a27eaa9ec1, p16k 719c80792e). Timings (shared
  GPU, rough; 17 heat pauses machine-wide 12:03-13:41): a1 v3o p8k 32.53 s, p16k 184.91 s VOID (paused); b1 v3p + R53A=0
  28.09 / 51.56 s; b2 33.32 / 51.29 s (its decode spanned a pause; the prefills didn't); a2 skipped (the 15 min lock ran
  out after the cool-start waits). Re-queued 13:5x with 25 min locks.

## R-62P PRUNE-COLD (NewIdeas Sun 1 Nov 01:0x, QUEUE R-62P; CPU replay; written before any result)
- Data: cap45 (C 160 live REF2b loop, block 16, gate; S45's 6 prompts). Kept per pass k = the rows of its one commit
  replay at layer 1 (one per partly accepted pass), else the pass width; sum k = the tokens (708: 197 = 197).
- Replay at C 224 (and 192): r57b_replay.py's LRU mirror (+0.30 % vs miss.bin at C 160); each policy evolves its own cache.
  B16 = as captured. T03 = t 0.3 modelled, rows 0..r_t-1 with r_t = min(16, k + D_p), where D_p = Leap's t 0.3 rows -
  kept per check (s45c2_c160: 1640 2.81, 91 2.18, 694 2.14, 1282 2.39, 708 2.39; 68 never truncated: r_t = 16),
  stochastic rounding. (a) ORACLE = rows 0..k-1 (the kept prefix with its bonus row). (b) CUT on T03, layer by layer: the
  check ends before the first row j >= 1 that needs a non-resident expert at that layer (row 0 always runs); row j's
  non-resident experts there are read async (resident by the next pass, not demand). Kept' = min(k, the final width).
- Metric: demand misses per kept token (~ SSD ms per kept token); also passes per token, and a cycle model per kept token
  (150 ms per pass + 0.27 ms per demand miss: Leap's S45 at C 224, 115.7 ms for 432 misses).
- PASS: (b) saves >= 15 % of SSD ms per kept token vs T03 -> a live screen (Leap's loop). KILL: (a) saves < 10 % vs T03.
- Caveats: T03 is modelled (cap45 has no drafter confidences), so here it never cuts the kept prefix (Leap's t 0.3 lost
  0.2-1.4 kept per check). (b)'s lost tokens come back in later passes; the trajectory is not re-simulated.
- Prediction: (a) -20-30 % vs T03. (b): only row 0's misses stay, kept ~2-3 per pass, so SSD per kept token may PASS
  while passes per token rise x2-3 and the cycle model per token gets worse.

## R-62P RESULT (Sun 1 Nov 01:4x; /Users/kamen/Fit12/data/r62p/r62p.txt, scripts/r62p_replay.py, 5 s CPU)
- Check: B16 at C 160 = 141,883 demand misses = r57b_replay's baseline (itself +0.30 % vs miss.bin). T03 modelled rows
  per pass: 68 16.00, 1640 8.04, 91 9.72, 694 9.67, 1282 8.89, 708 8.47 (Leap measured 7.55-10.86 at t 0.3).
- C 224, demand misses per token (1483 tokens; T03 = the baseline):
  B16 56.9 | T03 45.0 | ORACLE 39.2 = -12.9 % vs T03 (KILL needed < 10 %: not killed) |
  CUT 37.1 = -17.6 % vs T03 (the line's >= 15 %: met), BUT 1483 passes for 1483 tokens (1284 redo passes):
  passes per token +645 %, cycle model per token (150 ms per pass + 0.27 ms per miss) 160.0 vs 32.3 ms = +396 %.
  C 192: ORACLE -13.7 %, CUT -15.5 % (passes +645 %, cycle +362 %).
- Verdict: PASS on the letter for (b), but the rule as written degenerates to one row per pass. Almost every row >= 1
  needs some non-resident expert at some layer (~9 misses per layer per 16-row check at C 224), so each pass is cut
  to its first row: AR decoding with prefetch. A live screen of CUT would lose badly on tok/s: not worth one.
  The ORACLE's -12.9 % is the real room beyond t 0.3 (the ~1.5-2.4 rows t 0.3 still checks past the kept prefix).
  Reaching it needs a rule that knows which rows will be rejected: drafter confidence x the row's non-resident reads
  (EcoSpec's marginal-read cost). cap45 has no drafter confidences, so that needs a capture with them.
- Side number: AR-like passes (row 0 only, with async prefetch) cost 37.1 misses per token at C 224 vs T03's 45.0, so
  16-row verification with ~7.5 kept per pass costs only ~21 % more SSD per token than AR: rows share their experts.

## SG12 short-prompt grouped prefill: pre-registration (Sun 1 Nov 15:05; CPU code read; written before any build or run)
- Cause of the 12 GB default's binding transient (M-5d: +0.57-0.88 GB MLX active in short-prompt prefills, row max 11.757
  GB, headroom 0.19): v3p tier.prefill_call (bigmoe_v3f/tier.py:242) takes the grouped path (G = BIGMOE_TIER_PF_GROUP 64,
  SHIP_AR env) only when ids.size >= 4 x len(uniq) (the sorted-rhs gather regime). Short prompts (10 n < 4 x uniq) go to
  _prefill_single: one scratch of every used expert, len(uniq) x 2.76 MB (~215-320 experts = 0.6-0.9 GB), beside the
  resident pools. Side find: tier._mem_in / _mem_out reset MLX's peak counter at every prefill MoE call, which explains
  why my harness's 'row MLX peak' reads below the sampled active memory (M-5b caveat): that field is void.
- Fix (copy /Users/kamen/Fit12/v3q = v3p + SG12, switch F12_SG12=1, default off): in that regime, run the used experts in
  groups of <= G (ascending ids), two G-row scratch buffers (group g + 1 read while group g computes), each group's
  (row, slot) pairs in one call, no row padding, and the group scratch's expert count padded with unused zero rows so
  that B_g < 4 x E_g: every call then stays on the gather kernel the single call uses (MLX picks the sorted-rhs kernel
  only for B >= 16, sorted, B / E >= 4; else the per-pair vector kernel). Outputs go back to their pair positions before
  the model's combine. Exact by construction iff the vector kernel's per-pair output does not depend on the batch.
- Step 0 (GPU micro-test, synthetic 4-bit weights in the tier's shapes, mlx 0.32.2 + mlx_lm SwitchGLU, share lock,
  < 2 min, ~1.5 GB): grouped vs single on 12 skewed random routings, n = 8-300 (only those in the single regime), G = 64
  and 32. PASS: bit-equal outputs in every case AND a control (the same group without E padding, in the rhs regime)
  differs somewhere, so the test can see a kernel change. KILL: any single-regime case differs. Control equal
  everywhere: NO VERDICT (test blind).
- Step 1 (model, after EvalSuite r37 ends; shared lock, peak + 8 GB): pinned 8db7 bt_b12 + v3q vs v3p on M-5d's 16 math
  rows (G 64). PASS: tokens equal, KV / logits hashes equal on p2k-short checks, max row footprint <= 11.757 - 0.30 GB,
  prefill time per row <= +5 %. KILL: any bit change, or max drops < 0.10 GB.
- Value if PASS: headroom 0.19 -> ~0.5-0.7 GB at the 12 GB default's short-prompt peak; C_m 60 -> ~64-67 at short
  contexts (Breakthrough's call, own exact check).
- **Step 0 RESULT (15:10; ran 15:1x, shared lock, data/sg12/step0.txt): PASS.** 11 single-regime routings (n 8-150, U 59-438,
  two skews) x G 64 / 32: 0 differing bits (up to 3 groups per call needed padding). Control: the same G 64 groups without
  padding (2 groups in the rhs regime) changed 1,000,993 of 2,560,000 output values, so the test does see a kernel switch,
  and the padding rule is required. Compute alone, no reads: grouped 1.0-1.6x the single call (n 100: 11.1 -> 14.0 ms per
  layer call at G 64); in the engine group g + 1's reads overlap group g, which the single path cannot. MLX 0.32.2.
- **Step 1a RESULT (ran 19:50:26-19:51:16, shared lock, PAUSE 0, rc 0; data/scr/sg1/{a,b}.txt): EXACT, but the memory gain is small.**
  a = pinned 8db7 + v3p, b = + v3q F12_SG12=1; prompts m10 / m13 / m3 / m15 (94 / 68 / 97 / 77 tokens), one process each.
  kv / h / logits / sha32 equal on all 4 (m10 9376798271 / 84ca67acf4 / c4ef0ad4ee / 6f8d4c33f7; m3 7b5c958ab9 / cc888992e3 /
  9e492370ae / dcd0ce692f). SG12 engaged: 31 calls (3 / 21 / 4 / 3 per prompt), 118 groups, 19 padded, rows_max 89,
  uniq_max 316, skip 0; b's remaining single calls are the 'fallback' ones (uniq <= 64).
  Prefill-phase peak footprint a / b: m10 10.67 / 10.71, m13 11.57 / 11.47, m3 11.65 / 11.53, m15 11.58 / 11.58 GB (MLX
  active 10.31 / 10.31, 11.06 / 10.97, 11.13 / 11.03, 11.06 / 11.06): -0.10 / -0.12 / 0.00 on the warm prompts; the max
  over the 4 prompts 11.65 -> 11.58 (-0.07). Prefill s (shared GPU, rough): 3.53 / 4.10 (first prompt), 3.10 / 3.03,
  3.47 / 3.41, 2.91 / 2.88.
- Reading: the padding rule needs B / 4 + 1 rows for popular groups (rows_max 89: 2 x 89 x 2.76 MB = 0.49 GB of double
  buffers), so SG12 swaps a 0.65-0.87 GB scratch for up to 0.49 GB. And m15's peak did not move although SG12 ran there, so
  part of the +0.7-0.8 GB transient sits outside the single path (the grouped path's 2 x 64-expert buffers, 0.35 GB, or
  non-MoE work). The pre-registered value (+0.4-0.6 GB headroom) looks wrong; step 1b (16 rows vs M-5d's 11.757) decides.
- **Step 1b RESULT (ran 19:56-20:22, shared lock, rc 0, PAUSE seen 1: memory only, so no effect; data/scr/sg1b/math.txt): KILL by
  the pre-registered line (exact, gain < 0.10 GB).** v3q + F12_SG12=1 on pinned 8db7, M-5d's 16 math rows: tokens = the
  suite's on 16/16; SG12 ran (209 calls, 677 groups, 115 padded, rows_max 89, skip 0; PF single 37, grouped 522).
  Lifetime peak 11.700 GB (M-5d 11.774: -0.074); row max 11.685 (row 15; M-5d's top row 10: 11.757 -> 11.602). PASS needed
  <= 11.457; KILL if the drop is < 0.10 -> KILL. The binding peak moved to LATE prefill (row 15 at 2.87 s of a 3.22 s
  prefill, MLX active 11.060 GB = +0.80 over the row start): the late-prefill transient v3o also has (M-5b). That is the
  HumanEval+ trace's target. v3q stays a copy; not proposed.

## R-53D RESULT (round 2 ran 20:27-20:40, two shared locks, cool starts, PAUSE 0 in all arms; data/scr/r53d/): EXACT, inside +-3 %
- Hashes (kv / h / logits / sha32) equal in all 4 arms on both prompts and = R-53C's (p8k a27eaa9ec1 / 9091c3cf4c / 90651cfda6
  / 0b0d91cd43; p16k 719c80792e / 730b7ce740 / 4a21736a94 / 3f218067f8).
- Prefill s (shared GPU, rough), A = v3o, B = v3p + R53A=0: p8k A 21.68 / 21.54 (21.61), B 21.27 / 21.03 (21.15) = -2.1 %;
  p16k A 40.86 / 40.36 (40.61), B 39.12 / 39.87 (39.50) = -2.7 %. Fast-mode defaults (PFR from 512 tokens) make these
  prefills ~1/3 faster than R-53C's.
- Ship rule: within +-3 % on both prompts -> a quiet ABBA decides (NewIdeas' call). B keeps part of R-53C's -5.9 / -6.4 %.

## HT1 HumanEval+ prefill memory trace (NewIdeas GO Sun 19:5x / 22:5x, orchestrator Mon 20:4x; written Mon 2 Nov 20:5x, before any run)
- Q: which allocation makes the +0.8-0.9 GB MLX-active transient in short-prompt prefills that sets the 12 GB default's peak?
  (EvalSuite PMIN12 control, live 8db71cc537: 11.95 GB on HumanEval/156, /45, /42, /8, prompts 79-150 tokens; SG12 1b: the
  math max moved to LATE prefill, +0.80 GB.)
- Harness /Users/kamen/Fit12/scripts/fit_creep.py (backup .bak_ht1) with F12_MEMTRACE=1, F12_ARM=live, pinned 8db7, refs =
  the control's rows (F12_REFARM=b12llff4), F12_KEYS = HumanEval/68 (the suite's first humaneval row: warm-up, takes the C_m
  resize) then /156, /45, /42, /8, one process. Markers read counters only (get_active / get_peak / get_cache memory; the
  peak counter restarts at each marker, as the engine's _mem_in / _mem_out already do at every trunk MoE call; no syncs, no
  extra evals): per trunk MoE call 'pre' (the interval since the last MoE call: PLE, HC, attention, router) and 'moe' (the
  call; uniq, ids, grouped or single); the MTP head's prompt pass (_MTPHead.append in / out; its non-tier MoE sync_call in /
  out); b12_prefill in / out; the state's return; the decode as one interval. Per row: intervals ranked by MLX peak, and the
  interval that holds the 20 ms sampler's row max. Events saved per row (data/scr/ht1/he_events_r<k>.json).
- Readings (a diagnostic, no keep / kill): NAMED if one interval kind holds the row's MLX peak on >= 3 of the 4 target rows
  with peak - row-start active >= 0.5 GB; then a trim screen is pre-registered for it (exact: KV / logits / sha32 = the
  control's). DIFFUSE otherwise: the top 3 per row go to NewIdeas.
- Validity: every row's tokens = b12llff4's (else no reading). The target rows' sampled max vs the suite's row 'mem' is
  information (my process history differs: 5 rows, not 24).
- Prediction: trunk 'moe' intervals on the single path (uniq x 2.76 MB scratch, 0.6-0.9 GB) early in the prefill, and the
  MTP head's sync_call late; on these 79-150-token prompts mostly the first.
- Gates: shared lock (memory only, nothing timed), NO-SWAP >= 20 GB, no other process > 8 GB, EvalSuite decode idle 5 min,
  swap guard +2 GB; one process, ~4 min, cap 900 s.

## T0 two-pass decode, engine side (orchestrator 21:0x via Breakthrough 20:55, ~/Breakthrough/PLAN_12GB_TWOPASS.md section 4; written Mon 2 Nov 21:0x, before any run)
- Code /Users/kamen/Fit12/scripts/t0_twopass.py. apply(B) patches bt_b12._install in-process (9 asserted string replacements;
  Breakthrough's file untouched). A one-row call built while T0's pass-1 flag is on: ranks 0-1 posted waited, ranks 2-19
  flag 0 (never read), LRU-touch < 2, only ranks 0-1 counted (no residents added), no late rank (late1 / l3 skipped, no STASH
  write), no pre-routed pair. Weights stay B12_NORM=mass (probability / the true top-10 mass), so ranks 2-9 add exactly 0
  (their positions read rank 0's slot at weight 0). generate(): exact AR prefill of prompt[:-1] (B.b12_prefill, as bt_b12's
  b12_generate); then per row: eager snapshot of the non-trimmable state, pass 1, p1 = max softmax (fp32); keep if p1 >=
  theta, else rollback_after_verify(1) and pass 2 under ALLWAIT (B.AW[0] on for the whole decode).
- Gates (/Users/kamen/Fit12/scripts/fit_t0.py, one process, load_b12(0.10) without the MTP head; HumanEval/156, the first
  IFEval row, math number_theory/459; 96 tokens; arms ex / g1 / g0 / t80 / t95 / t80b, each from its own exact prefill):
  - G1 PASS: g1 (theta 1.01) = ex on tokens AND on the sha1 of every step's chosen logits, on all 3 prompts. FAIL: any
    difference -> the rollback is not exact; T0 goes to EvalSuite only after a fix.
  - Determinism PASS: t80b = t80 (tokens + logits sha) on all 3.
  - Pass 1 is top-2: g0 has 0 ALLWAIT calls, <= 2 waited reads per row-layer, and logits sha != ex's.
  - Information: pass-2 rates at theta 0.8 / 0.95, first divergence from ex, decode tok/s (shared GPU, rough).
- Then: the hook for EvalSuite's same-scores runs (theta 0.8 code, 0.95 chat); T1 = per-row k inside multi-row checks
  (one-row pass 1 already reads ranks 0-1 only here).
- Gates for the run: shared lock 'MEM 10 GB', one > 5 GB job at a time (waits for other sessions' runs), NO-SWAP >= 20 GB,
  swap guard +2 GB, PAUSE honoured; ~6 min, cap 900 s.
- **HT1 RESULT (ran 21:12:14-21:13:44, shared lock, rc 0, PAUSE seen 1 (memory only); data/scr/ht1/he.txt + he_events_r*.json):
  NAMED: the trunk's LAYER-0 MoE call on the single path.** Tokens = b12llff4's on all 5 rows. Row maxima vs the suite's:
  /156 11.885 (11.927), /45 11.668 (11.714), /42 11.749 (11.762), /8 11.826 (11.854) GB; my row starts are ~0.24 GB lower.
  On all 4 target rows the row's MLX peak is layer 0's MoE call at 0.12-0.16 s: +1.116 / +0.890 / +0.964 / +1.053 GB over
  the row start (MLX active 10.26). Layer 0 takes _prefill_single on all 4 (uniq 390 / 314 / 335 / 367 used experts: a
  scratch of uniq x 2.76 MB = 0.87-1.08 GB); on /42 and /8 the call is in the sorted-rhs regime (n >= 4 x uniq) but
  _groups found no consecutive grouping, so it fell back to single. Layer 0 is the only single-path layer on 3 of the 4
  rows (/45, 79 tokens, has 13). Next highest: grouped calls at +0.57-0.83 GB (groups grow past G = 64 until rows >= 4 per
  expert, up to 2G = 128 experts: 2 buffers x 128 x 2.76 MB = 0.71 GB). Other intervals: MTP-head MoE <= +0.41, the head
  pass <= +0.33, attention / HC / router <= +0.24 GB.
- Reading: the transient is the prefill's expert scratch, not the decode or the head. An exact trim must cap the per-call
  scratch in BOTH paths (single and grouped) at <= 64 experts while every group keeps the whole call's MLX kernel. Proposal
  (next, pre-registered separately): balanced (non-consecutive) groups of <= 64 experts. Expected: the layer MoE peak ~+0.35-0.45
  GB, so the row max -0.5..-0.65 GB on these rows; then the head (+0.41) binds. Prediction (MoE single early): right.

## SGB balanced prefill groups (HT1's trim; written Mon 2 Nov 21:2x, before any build or run)
- Idea: every prefill MoE call with > G used experts (G 64) runs in groups of <= G experts chosen by pair count (serpentine:
  heavy and light experts mixed, ids ascending inside a group), so that every group keeps the WHOLE call's MLX kernel:
  sorted-rhs calls (B >= 16, B >= 4 E) -> each group rows >= 4 n and >= 64 (sorted), rows padded to a multiple of 64 as the
  grouped path does (BUG1); per-pair calls -> each group rows < 4 n, no padding. Two <= G-expert buffers (0.35 GB) replace
  both the single scratch (uniq x 2.76 MB, up to 1.08 GB on HT1's rows) and the consecutive groups (up to 2G = 128 experts).
  Engine copy /Users/kamen/Fit12/v3r = cp -R ~/Breakthrough/v3p + tier.py patch (switch F12_SGB=1, default off).
- Step 0 (/Users/kamen/Fit12/scripts/sgb_micro.py; synthetic 4-bit experts in the tier's shapes, mlx 0.32.2, share lock,
  < 5 GB, ~2 min): 20 routings (n 79-200, alpha 0.5-1.1), G 64 and 48, vs the single call. PASS: 0 differing bits in every
  per-pair case and in every rhs case with the pad-64 rule, AND the control (heaviest experts forced into one rhs group of a
  per-pair call) differs. KILL: any difference under the rule. Also reported: the unpadded rhs variant, and a BUG1 probe
  (each rhs single call vs the same pairs on the per-pair kernel: max |d|; wrong rows would be large).
- Step 1 (model, after step 0 PASS; share lock 'MEM 12 GB', one > 5 GB job at a time): pinned 8db7 on v3r + F12_SGB=1 vs
  v3p: fit_scr53-style hashes (KV, MTP cache, h, logits, sha32) on m10 / m13 / m3 / m15 and HumanEval/156, /45, /42, /8;
  then fit_creep's HT1 rows (68, 156, 45, 42, 8) for the row maxima. PASS: every hash equal AND the 4 HumanEval maxima each
  <= HT1's - 0.30 GB (11.885 / 11.668 / 11.749 / 11.826) AND prefill s per row <= +10 % (shared GPU, rough). KILL: any bit
  change, or the largest max drops < 0.10 GB.
- Prediction: step 0 PASS (rhs needs the pad-64 rule, as the live grouped path); step 1 maxima -0.5..-0.65 GB, the MTP
  head's MoE (+0.41) binding next; prefill +0-10 %.
- **T0 gates RESULT (ran 21:23:48-21:37:41, shared lock, rc 0, PAUSE seen 1: the slow g1 arms on IFEval / math (357 / 297 s)
  were frozen, so timings there are void; data/scr/t0g/g1.txt): G1 PASS, determinism PASS, pass 1 is top-2: PASS on all 3.**
  - G1: theta 1.01 (every row: pass 1, rollback, pass 2) = the exact AR decode on tokens AND on every step's logits sha
    (HumanEval/156 lsha 37ccd09ba2, ifeval/1220 77c1b4c831, math 459 e83a5dd87b).
  - Determinism: t80b = t80 (tokens + logits) on all 3.
  - Pass 1: 0 ALLWAIT calls at theta 0; waited reads per row-layer 0.74 / 0.40 / 0.62 vs exact 2.73 / 1.75 / 2.28; logits
    differ from exact on all 3.
  - Pass-2 rates: theta 0.8 = 0 % (code) / 68.8 % (IFEval) / 20.8 % (math); theta 0.95 = 1.0 / 81.2 / 31.2 %.
  - Tokens vs exact (96): code equal under every theta, even theta 0; IFEval diverges at token 0 (theta 0) or 20 (0.8 / 0.95);
    math at 6 (theta 0), 4 (0.8: a certified pass-1 flip) or 6 (0.95).
  - Speed (shared GPU, rough; information): exact AR 10.9-12.7 tok/s; pass 1 only 21-25; theta 0.8: code 20.7, math 15.5,
    IFEval 10.6 (pass 1 + pass 2 on 69 % of rows is slower than exact).
- Handed to EvalSuite: /Users/kamen/Fit12/scripts/t0_hook.py (load / gen / STATS).
- **SGB step 0 RESULT (21:37:42-21:37:47, shared lock, rc 0; data/scr/sgb0/s0.txt): PASS.** Every case with a balanced
  split: 0 differing bits (per-pair 12 cases x G 64 / 48; rhs 7 cases, both the pad-64 and the unpadded variants); controls
  13 of 13 differ (a heavy rhs group inside a per-pair call: 181,480-554,772 of 2.56 M values). BUG1 probe: rhs vs per-pair
  max |d| 0.19-0.25 of max |y| 13.8-15.8 (rounding between kernels, no wrong rows). MLX peak 4.21 GB. But the serpentine
  split found no grouping in 6 of 20 cases, all with rows per expert 3.6-4.7 (near the kernel line), as HT1's /8 layer 0
  (4.06). AMENDMENT (21:4x, before step 1's v3r arm started; lines unchanged): v3r's _sgb_groups (and the micro test's)
  now split greedily (largest |rows - 4| first, positives to the smallest-sum group, negatives to the largest, m = ceil(U / G)
  .. + 2); the same per-group kernel rule is asserted before a split is used, so step 0's exactness basis carries over; a
  re-run of step 0 on the new split follows step 1.
- **SGB step 1 RESULT (ran 21:52-22:01, shared lock, rc 0; data/scr/sgb1/{a,b,m}.txt): PASS on every line (exact).**
  v3p vs v3r + F12_SGB=1 (greedy split): KV / h / logits / sha32 equal on all 8 prompts (m10 m13 m3 m15 h156 h45 h42 h8).
  HT1 rows on v3r: tokens = b12llff4 on 5/5; maxima /156 11.885 -> 11.338 (-0.547), /45 11.668 -> 11.316 (-0.352), /42
  11.749 -> 11.374 (-0.375), /8 11.826 -> 11.407 (-0.419) GB (line <= HT1 - 0.30: met). SGB 240 calls, 1019 groups, none
  unsplit, gmax 64. Prefill (shared GPU, rough) +0..+3.3 % (line +10 %). The new top intervals: the MTP head pass (+1.27 on
  the PFR row 1) and late trunk MoE calls (+0.55-0.57). Headroom to 11.95 on these rows: 0.07 -> 0.54 GB.
  Patch: /Users/kamen/Fit12/v3r/bigmoe_v3f/tier.py (diff vs tier.py.orig = v3p's). Adoption: Breakthrough's / NewIdeas' call.

## T1 + T2 two-pass MTP check on B12 (orchestrator 22:1x: continue T1 / T2; written Mon 2 Nov 22:1x, before any run)
- Drafter: the MTP head B12 ships (spec125 k 3, max_draft 3: 4-row checks); Breakthrough's 16-row DFlash2 plan is asked,
  not answered. The check logic is drafter-agnostic.
- Code /Users/kamen/Fit12/scripts/t2_twopass.py (T0's file untouched: EvalSuite wires it). T1 = apply(B): T0's patch with
  pass 1 allowed at any check width (S 1-4; verify rows post ranks 0-1 waited, the rest flag 0). T2 = generate(): spec125.generate
  (lookup off, early off, guard off) with the check replaced: pass 1 on all rows; certify (max softmax >= theta) and walk on
  the certified argmax; row n certified -> commit pass 1's rows 0..n; uncertified -> keep pass 1's rows 0..n-1 (commit keep n;
  n = 0: rollback), then spec125's exact check on toks[n:] (all 10 ranks). KV: certified rows keep pass 1's entries; the MTP
  head gets the matching hidden rows. theta None = the exact loop (no pass 1).
- Gates (/Users/kamen/Fit12/scripts/fit_t2.py, one process, MTP env of the live file, load_b12(0.10, mtp=True); HumanEval/156,
  the first IFEval row, math 459; 128 tokens; arms ref (spec125.generate) / ex / g1 / g0 / t80 / t95 / t80b):
  - loop = spec125: ex tokens = ref tokens on all 3 (my loop mirrors the shipped decode). FAIL -> fix before anything else.
  - G1 PASS: g1 (theta 1.01) = ex on tokens AND the sha1 of the logits that decided every committed token, on all 3.
  - determinism: t80b = t80 (tokens + lsha). g0: 0 checks with pass 2.
  - information: tok/s and tok / check at theta 0.8 / 0.95 vs ref (shared GPU, rough), pass-2 share of checks, rows per pass.
- Run: share lock 'MEM 12 GB', one > 5 GB job at a time, PAUSE honoured; ~10 min, cap 1500 s.
- **SGB step 0b (greedy split; ran 22:09:28-22:09:33, shared, rc 0; data/scr/sgb0/s0b.txt): PASS** (information): 0
  differing bits, controls 13 / 13 differ, MLX peak 4.21 GB; 1 case without a split (the serpentine split missed 6).
- **T2 gates RESULT (ran 22:41:39-22:53:24, shared lock, rc 0, PAUSE seen 1: ifeval g1 (238 s) and math ex (250 s) were
  frozen, their timings void; data/scr/t2g/g1.txt): every gate PASS on all 3 prompts.**
  - loop = spec125: ex tokens = ref tokens (HumanEval/156 5e8fe529fd, ifeval/1220 8e025ddb6b, math 459 8c6bb18289).
  - G1: theta 1.01 = ex on tokens AND logits sha (ed44f4b311 / 9cfc1c82ae / 5a9148f14e). Determinism: t80b = t80. g0: 0
    checks with pass 2.
  - Speed (shared GPU, rough; 128 tokens; tok/s vs ref = spec125's shipped decode):
    | prompt | ref | theta 0.8 (pass-2 checks) | theta 0.95 | theta 0 (pass 1 only) | first divergence (0.8) |
    | HumanEval/156 | 19.92 | 37.74 = 1.89x (2 / 34) | 37.44 (2 / 34) | 38.00 | 118 |
    | ifeval/1220 | 12.30 | 9.95 = 0.81x (67 / 72) | 10.45 (64 / 64) | 21.61 | 6 |
    | math 459 | 19.29 | 23.18 = 1.20x (20 / 37) | 19.70 (26 / 38) | 37.07 | 60 |
    tok / check: code 3.76 vs 3.88 (ref); math 3.46 vs 3.56; IFEval 1.78 vs 1.88.
  - Reading: the two-pass check works end to end at 12 GB with MTP 4-row checks: code ~1.9x where pass 1 certifies almost
    every row; chat loses (pass 1 wasted on 93-100 % of checks). Quality is not judged here (EvalSuite). Breakthrough
    (22:19): T3 timing on DFlash2 L1-20 16-row checks at 4 bits (D7b); MTP checks were for T2's correctness.
- T2s (written Mon 2 Nov 23:0x, before any run; information, shared GPU, rough): the skip rule for chat (t2_twopass skip=(2, 8):
  after 2 checks in a row that needed pass 2, the next 8 checks run exactly, then pass 1 again). Arms ref (spec125) / t80 /
  t80s (0.8 + skip) / t95s (0.95 + skip) on HumanEval/156, /42, the first IFEval row, math 459, algebra/509; 128 tokens.
  Expectation: IFEval back to ~1.0x ref with skip, code / math within a few % of t80. If so, T2-MTP + skip is a FAST-track
  12 GB mode for EvalSuite's same-scores test (theta 0.8 code, 0.95 chat), until T3's 16-row engine exists.
- **T2s RESULT (ran 23:05:52-23:13:57, shared lock, rc 0, PAUSE seen 1: arms with equal work but 2x the time are void:
  HumanEval/42 t80s / t95s, algebra/509 ref; data/scr/t2s/s1.txt). tok/s, 128 tokens:**
  | prompt | ref | t80 | t80s (skip 2, 8) | t95s | first divergence (t80) |
  | HumanEval/156 | 20.67 | 40.62 (1.97x) | 39.65 (3 skipped) | 38.64 | 118 |
  | HumanEval/42 | 21.10 | 42.76 (2.03x; all 128 tokens = ref) | void | void | none |
  | ifeval/1220 | 10.35 | 10.67 | 13.38 (50 of 64 checks exact) | 13.91 | 6 (t80), 18 (skip) |
  | math 459 | 19.92 | 23.97 (1.20x) | 22.72 | 20.66 | 60 |
  | algebra/509 | void | 24.63 | 24.89 | 25.18 | 10 |
  Reading: the skip rule brings chat back to (or above) the shipped speed while code keeps ~2x. Breakthrough (23:03, map
  simulation): a drafter-cf gate (two-pass only if the first draft's cf >= 0.9) separates code from chat better (22 % of
  code checks exact holding 9 % of code's kept tokens; 60-70 % of chat checks exact), plus the history rule for chat's rest.

## T3 two-pass 16-row check with DFlash2 L1-20 (D7b) on B12 (Breakthrough 22:19 / 23:03; written Mon 2 Nov 23:4x, before any run)
- Code /Users/kamen/Fit12/scripts/t3_twopass.py: apply(B) = T1 + a WIDE pass-1 branch (5-16 rows post ranks 0-1 only, 2 ids
  per row: gated.MAXK <= 92 caps B12's 20-id post at 4 rows) + a prefill tap hook (layers 5 / 15 / 25 / 35 / 45, HC mean);
  load_draft = L1-20 + d7b_live.apply (4-bit layers), bound to B12; generate = per check: 15 DFlash2 drafts + cf; pass 1 on
  16 rows; certify / walk; pass 2 on rows n..j (pass 1's first predicted miss) in <= 4-row exact calls; the check ends there
  (option (a)). The exact check = 16 rows in consecutive <= 4-row calls, stopping at the first miss. cf gate (two-pass only
  if the first draft's cf >= cfmin) and the history skip rule are options. No MTP head (DFlash2 drafts).
- Gates (/Users/kamen/Fit12/scripts/fit_t3.py, one process; HumanEval/156, /42, the first IFEval row, math 459; 128 tokens):
  - G1: theta 1.01 + p2_all (pass 2 = the exact check of every row from n) = theta None on tokens AND lsha, on all 4.
  - determinism: t80b = t80. The run must load and finish under 11.95 GB lifetime (D7b drafter in the budget).
  - information: tok/s and kept per check for ex / t80 / t80c / t80cs (shared GPU, rough), pass-2 / exact / cf-gated shares.
  KILL for T3 as built: G1 fails (fix first), or the lifetime peak > 11.95 GB.
- Run: share lock 'MEM 12 GB', one > 5 GB job at a time, PAUSE honoured; cap 1500 s; before the night's 00:30 big tests or
  by day.
- T3 run 1 (23:24:50-23:25:10): VOID. The ex arm ran (HumanEval/156: 11 checks, 11.64 kept per check, 17.53 tok/s, lifetime
  11.04 GB; drafter 0.928 GB own params, after load 10.59 GB), then g1 aborted: "bigmoe: 26 experts > C=20 at layer 48": pass 1
  on 16 rows posts up to 32 rank-0-1 LSB parts to the sidecar pool (C_l 20). AMENDMENT (before run 2): pools C_m 47 / C_l 32
  (+12 x 0.77 MB - 4 x 2.16 MB per layer = +0.03 GB), the plan's rank-aware direction; lines unchanged.
- **T3 gates RESULT (run 2, 23:39:31-23:55:21, shared lock, rc 0, C_m 47 / C_l 32, PAUSE seen 1: ifeval ex (254 s) and t80b
  (421 s) were frozen, their timings void; data/scr/t3g/g1.txt): G1 PASS and determinism PASS on all 4 prompts; lifetime
  peak 11.355 GB over 28 arms (line 11.95).** tok/s (shared GPU, rough), 128 tokens, kept per check in brackets:
  | prompt | ex (exact 16-row, 4-row calls) | t80 | t80c (cf gate 0.9) | t80cs (+ skip) | g0 (pass 1 only) |
  | HumanEval/156 | 18.21 (11.64) | 50.49 (10.67) | 53.50 (12.80) | 52.70 | 53.76 |
  | HumanEval/42 | 18.96 (14.22) | 49.42 (14.22; tokens = ex) | 48.94 | 49.77 | 65.23 |
  | ifeval/1220 | void | 8.11 (1.91; pass 2 in 66 / 67) | 10.18 (48 exact by the cf gate) | 10.81 | 13.28 |
  | math 459 | 17.17 (5.82) | 16.72 (4.57) | 18.21 (5.57) | 16.67 | 26.09 |
  Reading: code ~50 tok/s at 12 GB (2.5x the shipped MTP decode's ~20), math ~ex, chat ~ the shipped decode with the gates.
  Per check ~210 ms on code (10.7 kept / 50 tok/s) vs the plan's projected 106-119 ms: T3's timing (Breakthrough's line,
  quiet lock, per-check components; PASS >= 90 tok/s, KILL < 70 at today's kept) is next, with per-phase timers.

## T3 timing (Breakthrough's T3 line; written Tue 3 Nov 00:1x, before any run; runs by day, quiet lock)
- fit_t3.py with per-check timers at the loop's own eval points (draft, pass 1, pass 2 / exact, commit) and loader misses per
  phase; C_m 47 / C_l 32; prompts HumanEval/156, /42, /68, /91, math 459, prealgebra/1640, ifeval/1220; 256 tokens; arms
  ex (exact 16-row), t80c (theta 0.8 + cf gate 0.9), t80cs (+ skip 2, 8); one quiet lock (an important number).
- Lines (Breakthrough's T3, on code = the 4 HumanEval rows' t80c): PASS >= 90 tok/s; KILL < 70 tok/s. Between: report the
  per-check components (where the ~210 ms of today's shared run goes) to Breakthrough for T4's levers.
- AMENDMENT (Tue 3 Nov 00:2x, before any timing run): pass 1's tier_glu on ranks 0-1 only (plan '44', switch F12_P1K2 /
  T3._K2; the 10-rank form computes 8 weight-0 pairs per row reading rank 0's slot: 16 rows x 10 pairs ~ 21 GB of expert
  reads per pass at the kernel's one-grid-row-per-pair layout, ~75 ms at 280 GB/s). Exact by arithmetic (x + 0 = x), so a
  gate: g0k2 = g0 on tokens AND lsha on every prompt (FAIL -> K2 is dropped and the timing uses the 10-rank form). Arms now
  ex, g0, g0k2, t80c, t80ck2, t80csk2; the PASS / KILL lines apply to t80ck2 if K2's gate passes, else t80c.
- AMENDMENT (Tue 3 Nov 01:5x, before any timing run): the slot moves to tonight (NewIdeas 01:4x: the night GPU is free under
  ~30 GB while BigMoE's 45 GB turn is memory-blocked). Order: after Breakthrough's decision map and Leap's small screens,
  before EvalSuite's IFX (05:30) and MTP Head's chain (06:00). No quiet ticket while the map waits or runs (its guard stops a
  running map for any other session's quiet window, held or waited for). No start or grant from 05:05 (else a day slot, cron
  15:07). NO-SWAP >= 16 GB before the ticket. Design, arms, prompts and lines unchanged.
- RESULT (Tue 3 Nov 02:29-02:38, quiet lock, PAUSE 0, GPU 0 % at the grant, Kamen's Telegram ~1.2 cores; lifetime 11.324 GB;
  /Users/kamen/Fit12/data/scr/t3t/q1.txt; totals = summed tokens / summed decode s):
  GATE K2 PASS 7/7 (g0k2 = g0 on tokens + lsha), so the lines use t80ck2.
  Code (4 HumanEval rows): t80ck2 56.2 tok/s (156: 52.8, 42: 55.0, 68: 67.2, 91: 49.9) -> KILL (< 70) on Breakthrough's line.
    Same rows: ex 18.9, t80c 49.0, g0k2 61.7 (pass 1 only, no pass 2: pass 1 alone keeps code under 70), t80csk2 48.3.
    Per check, t80ck2 (kept 11.9): draft 11.1 ms, pass 1 145.1, pass 2 / exact 38.2, commit 12.4; total 211.
    Pass 1: ~670 part loads per check. Pass 2 / exact calls (<= 4 rows): ~120 ms and ~430 part loads per call
    (ex: 184 ms, ~830 loads per call).
    Bytes per phase and the per-layer split were not logged (next run).
  K2 saves ~30 ms of pass 1 per check on code (+15 % tok/s); keep it. The skip rule (2, 8) costs 14-20 % with DFlash2
  (code 48.3, math 17.2, IFEval 10.6): drop it for T3.
  Math (459 + 1640): t80ck2 21.5 vs ex 17.3 (1.24x). IFEval: t80ck2 11.3 vs ex 13.3 (0.85x); g0k2 18.3.
  t80ck2 equals ex on HumanEval/42 and /68; it diverges on /156 (token 118) and /91 (117). FAST quality is EvalSuite's call.
  To reach 90 at today's kept: <= 132 ms per check, so pass 1 <= ~70 ms if the rest stays (<= ~110 ms for 70).

## T3P pass-1 per-layer trace (Breakthrough 01:55, items 1-3; written Tue 3 Nov 03:0x, before any run)
- Question: where do pass 1's 145 ms per check go (code, t80ck2)? Split each layer's time into the loader holding the GPU at
  its gate (reads + table work) and the GPU side between gates (tier_glu, attention + router, route copy, poll latency).
  Compare bytes per pass and per miss with Breakthrough's replay (0.71 GB per check at C 51, ~0.60 at C 64). Measure how
  much C_m 51 (vs 47) cuts pass 1's misses and time.
- Run: fit_t3.py with F12_PHLOG=1 (phase marks: time, gate seq, disk bytes, misses at each phase's start and end; the
  computation is unchanged) and the loader's own call log (BIGMOE_CALLLOG: per gate call L, seq, misses, t_seen, t_rel).
  Arms ex, t80ck2; HumanEval/156, /42, /68; 256 tokens; two processes, C_m 47 / C_l 32 then C_m 51 / C_l 32; one quiet
  lock (screen scr_t3p). Analysis: /Users/kamen/Fit12/scripts/t3p_layers.py (per gated pass: wait, gap, edge, misses,
  MB, GB/s over the wait; per 8-layer band).
- Checks: in the C 47 process, t80ck2's tokens and lsha equal t3t's (156 674de3d0b4, 42 eae0a8bcd9, 68 685b9746a8), so
  the marks change nothing. C 51's equality to C 47 is reported only (a 1-row late path may depend on residency).
  Lifetime <= 11.95 GB in both.
- Readout (a diagnosis for T4's levers; Breakthrough ranks them; no PASS / KILL):
  (a) Wait share of pass 1's gated time > 60 %: reads (bytes or queue depth) are the lever. Gap share > 40 %: the
      per-layer chain or the GPU is.
  (b) MB per pass 1 vs the replay's 0.71 GB.
  (c) C 51 vs C 47: misses and ms of pass 1 per check.

## T3P amendment: the P1-K1 timing arm (NewIdeas 10:24, orchestrator slot 10:3x; written Tue 3 Nov 10:3x, before any run)
- What: arm t80ck1u = t80ck2 with each WIDE pass-1 call (3-16 rows) changed to NewIdeas' P1-K1 (the map's oea_k1 rule,
  ~/NewIdeas/scripts/dm_oea_masks.py). U = the rows' rank-0 experts; only U is posted (MSB-3 + LSB parts, waited). Each
  row's MoE = its ranks 0-9 that are in U, the first KU = 4 of them in rank order, all at 4 bits (U's LSB parts are read
  anyway, and the map runs 4-bit experts), at the model's weights / top-10 mass (B12_NORM=mass, no renorm). Narrow pass-1
  calls (<= 2 rows, only a capped run's last check) stay top-2 (K2). The certificate (theta 0.8), the cf gate (0.9) and
  pass 2 are unchanged.
- KU = 4: on NewIdeas' code masks (oea_k1, 162,144 realized row-layers; mean 2.44 experts), 95.3 % of row-layers keep
  <= 4 experts (<= 5: 99.1 %). The other 4.7 % lose their lowest-ranked union members. The engine counts kept pairs and
  rows over KU per pass 1 (reported per prompt; no extra sync, the counters ride on pass 1's own eval).
- Code: /Users/kamen/Fit12/scripts/t3_twopass.py k1u_moe + the _WIDE branch (backup t3_twopass.py.bak_k1u); fit_t3.py arm
  t80ck1u, F12_ALT, TOTAL lines (backup fit_t3.py.bak_k1u). Unit tests passed (synthetic, real dims 2560 / 640, fake
  loader): k1u_test.py (mask = dm_oea_masks' rule, posted ids = U, slots exact, weights bitwise, counters exact, output =
  a 10-column 4-bit tier_glu reference with the dropped pairs at weight 0, to bf16 rounding (<= 2 % of max |y|; bitwise
  in 2 of 12 cases)); k1u_install_test.py (the patched bt_b12._install block = the direct call bitwise; posts 16 ids per
  pool; the K2 path still posts ranks 0-1, 32 ids). Both in the session scratchpad.
- Run: inside T3P's quiet lock (orchestrator: the whole lock <= ~25 min), both processes (C_m 47 / 51): arms ex, t80ck2,
  t80ck1u on HumanEval/156, /42, /68, /91 (/91 added for an even count), 256 tokens. F12_ALT=1: odd prompts run t80ck1u
  before t80ck2, so each arm goes first on 2 prompts (warm-cache order balanced).
- Line (NewIdeas', taken as is): code tok/s = summed tokens / summed decode s over the 4 prompts, in the C 47 process
  (T3t's config). t80ck1u >= 1.10x t80ck2 = PASS; <= 1.03x = KILL; else BETWEEN. C 51's ratio is a replicate, reported;
  if it falls on the other side of a line, I say so.
- Reported: pass-1 ms, MB and misses per check (PHASES + call log), the wait / gap split per arm (t3p_layers.py with the
  arm filter), pass-2 share, kept per check, the K1U counters, the first divergence from ex. P1-K1 is not bitwise; its
  quality is judged by the map's certified flips, Breakthrough's mixed runs and EvalSuite.
- Checks: t80ck2's tokens + lsha = T3t's on 156 / 42 / 68 / 91 (674de3d0b4, eae0a8bcd9, 685b9746a8, d6162e5e65; the
  K1U branch is off there); t80ck1u's K1U counters engaged (row-layers > 0); lifetime <= 11.95 GB in both processes.
- If c47 fails with a script error, c51 runs T3P's own arms (ex, t80ck2), so the trace still lands.
- (10:4x, before any run) INFO arms added, no line: t80ck2m3 = t80ck2 with the wide pass 1 at 3 bits (ranks 0-1 from the
  MSB-3 pool only, no sidecar post, plan '33': Breakthrough's item 4 / M-18; its certificate comes from Breakthrough's
  3-bit map arm), and t80ck1um3 = t80ck1u at 3 bits (U's MSB-3 parts only, plan '3' x 4). Tested like the others
  (scratchpad m3_test.py: installed block = direct reference bitwise, one post per layer to the base pool, K1U-M3 = a
  10-column 3-bit reference to bf16 rounding; the 4-bit paths unchanged). Arm order per prompt: ex, t80ck2, t80ck2m3,
  t80ck1um3, t80ck1u; odd prompts reverse the four after ex, so t80ck2 and t80ck1u each run 1st twice and 4th twice.
  P1-K1's line and checks stand as above. Expected lock ~10-12 min (two processes, ~4.5 min each).

## T4-PF2 pass-2 prefetch from pass 1's record (Breakthrough 03:10 item 2 + 10:40; DESIGN NOTES Tue 3 Nov 10:4x, not pre-registered)
- Idea: pass 1 has routed all 48 layers for the 16 rows; for the rows pass 2 will run, its recorded top-10 (ranks 2-9;
  ranks 0-1 were just read) predicts pass 2's routing. Issue those reads ahead instead of layer by layer at the gate; log
  the overlap of pass 1's ids with pass 2's true ids; the gate demand-loads the rest as now.
- The ext's mechanism (read-only ~/Breakthrough/v3p/ext/bigmoe_ext.cpp): a layer-L post carries predictions for L+1 / L+2
  (n1 | n2 << 8 in row[65], ids in the prediction lines; capacity 30 + 31 x BIGMOE_PRED_LINES). Non-JIT: after L's release,
  prefetch() reads them into SPARE slots (BIGMOE_SPARE, carved from C; B12 sets 0 unless B12_PF), or inside the gate into
  LRU victims (BIGMOE_PF_VICTIM > 0) but only at gates that had a real load (so a prefetched layer does not prefetch the
  next: coverage alternates). PF_NOPUBLISH=1 (B12): the demand path adopts ready prefetches (pf_used). JIT mode
  (BIGMOE_PF_JIT: reserve / schedule L+1..L+3 / pump with BIGMOE_PF_QD) victims from the LRU tail without that limit, but
  changes table publishing for the whole engine. All of these envs are read once at ext import: a separate process config.
- So the arm needs its own process (t80ck2 vs t80ck2pf in it), pass 1's ids stashed per layer, the block's <= 4-row exact
  path given the call's rows' recorded ranks 2-9 (union) for L+1 / L+2. Sized by T3P: the exact calls' wait share and
  misses per call (T3t: ~120 ms and ~432 part loads per <= 4-row call).

## T3P + P1-K1 RESULT (ran Tue 3 Nov 14:39:30-14:46:02, one quiet lock (390 s of the 1491 granted), rc 0 / 0, PAUSE 0, swap flat 5.69 GB, SSD 25 -> 59 C; /Users/kamen/Fit12/data/scr/t3p/c47.txt, c51.txt, *_calls.csv, *_ph.json)
- Checks: t80ck2 = T3t (tokens + lsha) on all 4 prompts in BOTH processes (so the marks, the call log and the K1U / M3
  code change nothing there); tokens and kept identical across C_m 47 / 51 for every arm (deterministic); K1U counters
  engaged (2.38-2.66 experts per row-layer, 4.5-8.4 % of row-layers over KU 4); lifetime 11.309 GB (C 47), 11.734 (C 51).
  A first start (14:33) waited on a false 'another run' match (bigrun matched ' run (' in BigMoE's status text); fixed to
  peer ls's 'shared run (others may work)' annotation and restarted 14:39 (nothing had run).
- T3P readout (pass 1 of t80ck2, per gated pass, C 47 | C 51):
  (a) 154.9 | 152.5 ms = wait 85.8 | 83.5 (57 % of the gated time) + gap 63.3 | 63.4 (43 %) + edge 5.7. Gap > 40 %: the GPU
      side is a lever as large as the reads. The wait is BANDWIDTH-bound, not a per-layer sync: 950 | 918 MB per pass at
      11.1 | 11.0 GB/s over the wait; zero-miss gate calls cost 12 us. Wait per 8-layer band 19.3 -> 11.0 ms (misses 163 ->
      88: early layers miss most). Gap ~1.3 ms per layer for 16 rows (~0.72 ms for the <= 4-row exact calls).
  (b) 950 MB per pass 1 at C 47, 918 at C 51 = 1.29x Breakthrough's replay (0.71 GB at C 51); 1.33 MB per miss (MSB-3
      and LSB parts).
  (c) C 51 vs C 47: misses -2 % (696 vs 711), bytes -3.4 %, pass 1 -2.4 ms, t80ck2 57.09 vs 56.25 tok/s (+1.5 %): small
      (the replay's C^-1.7 predicted ~-12 % bytes).
  Exact calls (t80ck2's pass 2 / exact, <= 4 rows): per pass wait 78.2 + gap 34.5 ms, 832 MB at 10.6 GB/s; also
  bandwidth-bound, so prefetch inside pass 2 can hide at most the gap (~35 ms per call, ~-11 ms per check at today's pass-2
  rate), unless pass 2's reads go out during pass 1's gap. ex arm: 516 ms per check, per pass wait 138 + gap 39.5 ms.
  Gap costs per arm: K1U's 4 columns +7.9 ms (71.2), M3 (one pool, 3-bit) -7.4 ms (55.9).
- P1-K1, AS REGISTERED: KILL. Summed tokens / summed decode s, C 47: t80ck1u 47.73 / t80ck2 56.25 = x0.849 (C 51 x0.848).
  Pass 1 per check 145.9 -> 103.9 ms (wait 85.8 -> 47.7 per pass, bytes 950 -> 461 MB = x0.485), but kept 11.88 -> 10.11
  and pass 2 / exact 37.7 -> 84.1 ms per check.
- WHY (post hoc, not registered): the registered metric is confounded by divergent continuations, a flaw of my design.
  Per prompt (C 47): /68 x1.31 (no pass 2 in either arm, same kept 14.22), /42 x1.02, /156 x0.96, /91 x0.60. On /91
  t80ck2's text diverged from ex at token 117 and ended (EOS) at 177; ex itself and t80ck1u continue to the 256 cap
  through a hard stretch (ex kept 4.72 per check after token 117, t80ck1u 6.00), so t80ck2 skipped 80 hard tokens.
  On the text both arms share (each prompt until t80ck2's divergence from ex: 630 tokens), the arms take the same checks
  and paths (kept 14.32 vs 13.91, same pass-2 / exact counts, same draft cf), and the time ratios are: t80ck1u x1.100
  (C 51 x1.098), t80ck2m3 x1.122 (x1.126), t80ck1um3 x1.195 (x1.156). Excluding /91 from the registered metric: x1.07.
  So P1-K1's pass 1 does NOT cost certification or drafting on shared text, and it sits AT NewIdeas' 1.10 line there.
- INFO arms (registered metric): t80ck2m3 x1.030 (C 51 x1.035; pass 1 113.8 ms, kept 11.39, misses 294 vs 670 per
  check); t80ck1um3 x0.935 (x0.905). Breakthrough's map cleared 3-bit pass 1 on code (11:10; certified flips 0.13 %).
- Next (proposed, not run): a P1-K1 / M3 re-test with a pre-registered SHARED-TEXT metric and more prompts (8+ HumanEval),
  or a forced-text mode; the GPU gap (63 ms per pass 1 = Leap's dense lever) and pass-2 reads during pass 1's gap are the
  levers this trace names. Open: the commit phase reads 49-137 MB from disk per check with 0 loader misses (9-18 ms).

## T3R P1-K1 / M3 re-test on a SHARED-TEXT metric (PROPOSED Tue 3 Nov 14:5x, before any run; queue only with NewIdeas' OK and the orchestrator's slot)
- Why: T3P's registered summed metric compared different texts after the arms diverged (/91: the control stopped at EOS
  on token 177). The shared-text metric compares the same tokens.
- Run: one process, C_m 47 / C_l 32 (C 51 gave +1.5 % only), arms ex, t80ck2, t80ck2m3, t80ck1um3, t80ck1u (F12_ALT=1
  order), 256 tokens, 8 HumanEval prompts = T3P's 4 (156, 42, 68, 91) + the first 4 unused code keys in b12llff4's
  decode.jsonl order (16, 6, 79, 19). fit_t3.py now saves each arm's tokens (*_toks.json). One quiet lock, ~8-10 min,
  MEM 12 GB (screen scr_t3r).
- Metric: /Users/kamen/Fit12/scripts/t3_shared.py DIR TAG t80ck2 t80ck1u t80ck2m3 t80ck1um3: per prompt the cut = the
  first token where the two arms' outputs differ; each arm's checks until its committed tokens reach the cut, timed by its
  phase durations; tok/s summed over prompts. (On T3P's data it gives t80ck1u x1.100, t80ck2m3 x1.122, t80ck1um3 x1.195.)
- Lines (NewIdeas', on the shared-text metric): t80ck1u >= 1.10x t80ck2 = PASS, <= 1.03x = KILL, else BETWEEN. t80ck2m3
  and t80ck1um3 INFO (their quality: Breakthrough's map; the 3-bit + P1-K1 stack is not map-tested). The summed metric is
  reported beside it.
- Checks: t80ck2 = T3t / T3P (tokens + lsha) on 156 / 42 / 68 / 91; K1U counters engaged; lifetime <= 11.95 GB; the
  shared text covers >= 50 % of each arm's tokens in total (else the metric reads too little text: report, no verdict).
- (Tue 14:5x, before any run) AGREED: NewIdeas 14:52 OK'd T3R exactly as written above (their QUEUE 6bc28ab, replacing
  their stricter "every arm equals ex" cut). The orchestrator's slot: Wed 4 Nov 00:30-00:45, first in line (BigMoE at
  00:45): scr_t3r requests no lock before 00:30, uses none granted after 00:36, caps the process at 780 s.
  THE NEW LINE for this re-test of T3P's registered KILL (written now, before the run): on the shared-text metric above,
  t80ck1u >= 1.10x t80ck2 = PASS, <= 1.03x = KILL, else BETWEEN; < 50 % shared coverage = no verdict. T3R's verdict
  replaces T3P's for P1-K1's 12 GB timing (T3P's summed metric compared different texts). t80ck1um3 stays INFO until
  Breakthrough's stacked map arm (P1-K1 with KU 4 at 3 bits; masks ~/NewIdeas/data/dm_masks/oea_k1_ku4) clears its
  certificate.

## T3R-PF P2-PREFETCH (NewIdeas 14:56; written Tue 3 Nov 15:0x, before any run; EXACT add-on)
- Idea: pass 1's GPU gap (43 % of pass 1, ~1.3 ms per layer) leaves the SSD idle. In each WIDE pass-1 call, the base-pool
  post also carries up to B = 6 ids flagged late (bit 21): the ranks 1-9, not already posted, of the rows whose draft cf
  < 0.99 (the check's 15 draft cf are known before pass 1), row then rank order, deduped. The loader (flag mode, BIGMOE_DEFER
  8 in T3's env) releases the GPU for the waited ids first, then reads the late ones; pass 1 never uses their slots, and pass
  2 then finds them resident. MSB-3 parts only (base pool; rank-1 LSB parts are not prefetched). Nothing computed changes.
- Code: t3_twopass.pf2_extra + the _WIDE and k1u_moe posts (PF2_ROWS set by generate() before pass 1), fit_t3 arms t80ck2pf /
  t80ck1upf (F12_PF2_B 6), a per-prompt GATE PF2 line, and the loader's defer_stats per arm (late reads, late ids forced to
  wait). Synthetic tests passed (scratchpad pf2_test.py: the late ids = the rule's reference, pads unflagged, side posts
  unchanged, block output with PF on = off bitwise, for K2 and K1U; edge cases: no uncertain row, one row, dedupe).
- Run: a second process p47 in T3R's lock, ONLY if T3R's r47 ends rc 0 by 00:38 (the slot ends 00:45): C_m 47 / C_l 32,
  arms t80ck2, t80ck2pf, t80ck1u, t80ck1upf (F12_ALT=1: each PF arm runs after its base on 2 prompts, before it on 2),
  HumanEval 156 / 42 / 68 / 91, 256 tokens, phase marks + call log, cap 240 s.
- Gate: tokens + lsha of each PF arm = its base arm's on every prompt (else the add-on is not exact: no timing verdict).
- Line (NewIdeas'): summed tokens / summed decode s (no text confound: the tokens are equal), t80ck1upf >= 1.05x t80ck1u =
  PASS, <= 1.01x = KILL, else BETWEEN; t80ck2pf vs t80ck2 on the same line, reported as the second verdict.
- Reported: pass-2 / exact ms and misses per check, pass-1 misses per check, late ids posted and read, late ids forced to
  wait (must be 0), and the per-layer wait / gap of the PF arms vs their bases (does a late read delay the next layer?).
- (Tue 15:0x, before any run; orchestrator: GPU clear by 00:44) Both processes end by 00:43:30: r47's cap = min(780 s,
  seconds left to 00:43:30); a grant leaving < 480 s (after ~00:35:30) is released (slot lost: tell the orchestrator); p47
  starts only with >= 250 s left (and rc 0, SSD <= 62 C, no PAUSE), cap 240 s. T3R-PF approved inside the slot as gated
  (orchestrator 15:0x); if a gate fails it is dropped for tonight.
- (Tue 16:2x, orchestrator's check: the 00:23 start was a session-only cron wake, i.e. a prompt) The start is now a real
  background process of this session: /Users/kamen/Fit12/scripts/at_t3r.sh, task begh7gwar, pid 3922 (zsh 3920 -> claude
  18778; attached, since a detached runner's quiet lock lapses after 8 min). At Wed 00:23 it queues t3r + stop and runs runq.sh
  as its child; backstops: 00:37 with r47 never started -> stop the runner tree + `peer unlock` (also cancels a queued ticket),
  00:43:40 with the runner alive -> the same; nothing starts if it wakes after 00:30 or another Fit12 runner runs. Dry tests
  of all six paths: /Users/kamen/Fit12/scripts/at_t3r_test.sh. The 00:26 wake (cron 31d2cd0a) only verifies (fallback start <= 00:33 if the
  process is gone). No change to the run, the arms or the lines.

## T3R NOW (Tue 3 Nov 19:0x, written before the run; the orchestrator relaying Kamen: small breakthrough tests run now)
- T3R (+ T3R-PF on its gates) runs this evening instead of the 00:30 slot, in the quiet FIFO after MoonShots' M-28 and Leap's
  WC2: screens.sh scr_t3rn, output /Users/kamen/Fit12/data/scr/t3rn (r47 = T3R, p47 = T3R-PF). The arms, prompts, metric and
  lines are exactly T3R's (AGREED amendment) and T3R-PF's above; read-outs with DIR = /Users/kamen/Fit12/data/scr/t3rn.
- Gates (orchestrator: "Your gates stay"): NO-SWAP >= 16 GB before the request (<= 6 min wait) and again after the grant +
  30 s settle; swap +2 GB guard; r47 cap 780 s; p47 only on r47 rc 0, SSD <= 62 C, no PAUSE, cap 240 s; quiet lock 20 min.
  Changed condition: Kamen's apps are open, so the grant accepts GPU <= 20 % (was 5 %); the GPU % at the grant is logged and
  reported beside the verdict (the arms share the process and alternate order, so desktop load is noise, not bias).
- If r47 ends rc 0, the 00:30 slot is disarmed (at_t3r.sh pid 3922 stopped by pid, cron 31d2cd0a deleted) so BigMoE can
  start at 00:30; if a gate fails, the slot stays armed as it is. The orchestrator is told either way.
- RESULT (Tue 3 Nov 19:1x; ran 19:01:36-19:08:51, quiet lock 19:00:5x-19:08:53, data /Users/kamen/Fit12/data/scr/t3rn, read-out
  /Users/kamen/Fit12/scripts/t3r_verdict.py): T3R on the registered shared-text metric: t80ck1u / t80ck2 x1.065, coverage
  88 % / 84 % -> BETWEEN (line >= 1.10 PASS, <= 1.03 KILL). Per prompt (cut): 156 (118) x1.27, 16 (81) x1.11, 19 (254) x0.93,
  42 (132) x1.02, 6 (208) x1.08, 68 (256) x1.30, 79 (225) x1.18, 91 (117) x0.79. Checks: t80ck2 = T3t tokens + lsha on
  156/42/68/91 (4/4, exact arm reproduced bitwise); K1U engaged (2.38-2.66 experts per row-layer, 4.5-8.5 % of row-layers
  over KU 4); lifetime 11.317 GB; '] done'; rc 0, 435 s. Summed (T3's totals, INFO): ex 18.42, t80ck2 46.78, t80ck1u 44.45
  (x0.950), t80ck1um3 48.03 (x1.027) tok/s.
  FLAG: a heat PAUSE 19:07:51-19:08:31 (fanwatch, SSD 72 C at 19:07:31, die 86 C; it froze no job) fell inside r47. The
  engine waits on PAUSE at pass boundaries (v3p gated.py new_pass), so 36.9 s landed in one pass-1 call of /19 t80ck2m3
  (INFO arm: x0.461 overall, x0.12 on /19 = void). It also split /19's verdict pair: t80ck1u ran at the SSD's peak (~72 C;
  implied pass-1 read ~4.8 GB/s vs ~7.2 on its other prompts, assuming T3P's ~63 ms gap) and t80ck2 after the cool-down
  (~10.2 GB/s, normal). By the runner's standing rule a PAUSE-flagged run's timings are void, so this BETWEEN is not a clean
  verdict. Post hoc, NOT the registered metric: without /19, t80ck1u x1.114, t80ck2m3 x1.130, t80ck1um3 x1.233.
  INFO (no line; quality pending Breakthrough's stacked map arm): t80ck1um3 (P1-K1 KU 4 at 3 bits) x1.227 on shared text
  (coverage 76 % / 73 %; per prompt 1.28 1.16 1.17 1.15 1.31 1.38 1.35 0.87): T3P's x1.195 (4 prompts) held on 8.
  T3R-PF (p47) did not run: SSD 63 C at r47's end (gate <= 62 C). Next (the orchestrator's / NewIdeas' call): a clean re-run
  of T3R (~8 min) and T3R-PF (~4 min) in a cool quiet slot. The 00:30 slot is disarmed as asked (at_t3r.sh pid 3922 stopped
  by pid 19:11, cron 31d2cd0a deleted): BigMoE starts at 00:30.

## T3R2 (NewIdeas 19:15, QUEUE 4dbe21a; orchestrator 19:2x: Wed ~15:00, ~20 min quiet, first in line or right after Breakthrough's
## 15:00 arms, T0 follows; this section written Tue 3 Nov 19:2x, before any run)
- T3R's x1.065 is void by the runner's PAUSE rule: neither BETWEEN nor a verdict. The post-hoc x1.114 is not used.
- Arms: t80ck2 vs t80ck1u (P1-K1, KU 4), + t80ck1um3 INFO (ex and t80ck2m3 dropped, as NewIdeas' arm list). Order t80ck2,
  t80ck1um3, t80ck1u; F12_ALT=1 reverses it on odd prompts, so each verdict arm runs first on 8 of the 16 prompts.
- Prompts (the rule, fixed now): T3R's 8 (HumanEval 156, 42, 68, 91, 16, 6, 79, 19) + the next 8 HumanEval rows by index in
  /Users/kamen/EvalSuite/results/b12llff4/decode.jsonl (T3R used its humaneval positions 0-5 plus T3T's 156 / 42; the next 8
  are positions 6-13): 161, 17, 56, 141, 101, 63, 114, 77.
- Metric and lines unchanged: shared text (per prompt the cut = the first differing token; each arm's checks to the cut timed
  by phase sums); t80ck1u / t80ck2 >= 1.10 PASS, <= 1.03 KILL, else BETWEEN; < 50 % coverage for either arm = no verdict.
  THE VERDICT = the POOLED ratio over every counted prompt (16 unless one is paused twice); T3R's original 8 beside it.
  Read-out: /Users/kamen/Fit12/scripts/t3r2_verdict.py /Users/kamen/Fit12/data/scr/t3r2.
- Runner rules: start only at SSD <= 55 C and no PAUSE (checked under the lock, <= 8 min, else released; again in the process
  after the load, <= 2 min). A PAUSE (watched every 0.25 s) inside a prompt's span (its 3 arms, so also between the verdict
  pair) voids that prompt: its files are kept as void1_*; it is re-run once at the end after waiting <= 2 min for SSD <= 55 C
  and no PAUSE; paused again = excluded (void2_*). SSD per arm (start -> end) and PAUSE seconds per prompt are logged (PAIR
  lines). Caps: r47 960 s, quiet lock 25 min; C_m 47 / C_l 32, 256 tokens, phase marks.
- Then T3R-PF exactly as registered (fd081ae): p47 on r47 rc 0, then SSD <= 62 C and no PAUSE (waits <= 3 min), cap 240 s;
  a PAUSE inside its prompts voids its timings (logged per prompt, no re-run).
- Harness: fit_t3.py F12_RERUN (PAUSE watcher, run_prompt(), totals from the runs that count), dry-tested without the model
  (F12_FAKE=1, /Users/kamen/Fit12/scripts/fit_t3_dry.sh: a brief pause -> 3 prompts voided, re-run, kept; a pause that never
  clears -> 12 excluded; read-out checked on both). Screen: screens.sh scr_t3r2. Start: /Users/kamen/Fit12/scripts/at_screen.sh
  (dry-tested, 6 paths: /Users/kamen/Fit12/scripts/at_screen_test.sh) at Wed 14:54 (es_wait 5 min); the lock request at
  15:01:00 (a Breakthrough ticket queued at 15:00 goes first); nothing starts if it wakes after 15:20; slot over if r47 has
  not started by 16:15; hard stop 16:45.
- Heat: T3R ran at FAN_CAP 5000 and reached die 86 C / SSD 72 C after ~6 min (heat pause); T3P at cap 7000 ran 6.5 min clean.
  A higher cap during the slot is asked of the orchestrator (the owner away): ARMED (wed_boost.sh Wed 14:55 if Kamen idle
  >= 10 min: base 6000 / top 7000 until 15:45).
- AMENDMENT (NewIdeas 19:27, QUEUE 1ba42c3, before any run): the verdict needs >= 12 of the 16 prompts counted (excluded =
  paused twice); with fewer it is no verdict, and the pooled ratio is still reported (t3r2_verdict.py checks it).
- MOVED (orchestrator Tue 22:3x, before any run; Kamen on the laptop from ~13:45 Wed): Wed 13:05, quiet, MEM 12 GB, HARD
  STOP 13:40 (FLOOR's deadline 13:00; fans re-timed by the orchestrator to 12:55-13:40, base 6000 / top 7000). Arms, prompts,
  metric, lines, start rules and the 12-of-16 amendment are unchanged. Runner (screens.sh scr_t3r2): the lock request at
  13:05:00 only with >= 15 min to a 13:39:00 deadline; r47 starts only with >= 10 min left (it takes ~7-9 min; the SSD
  <= 55 C wait ends early if needed); p47 only with >= 260 s left; each cap is cut to the deadline - 20 s. Backstops in
  at_screen.sh: STOP 13:30 (r47 never started -> runner stopped + `peer unlock`, which also cancels a queued ticket),
  BACK 13:39:30 (the runner still alive -> the same). Dry test of the deadline logic with stubs (nine cases):
  /Users/kamen/Fit12/scripts/scr_t3r2_dry.sh. ARMED: Bash task bvfa6kh7j, pid 25482 (start 12:58); cron 77ebff26 Wed 13:09
  verifies only. The 15:01 arm (b9wtrsdx3 / pid 32556) and cron 4cdcecb9 are cancelled.
- HARNESS FIX (Tue 22:3x, before any run; timing and metric unchanged): fit_t3.py moves a paused prompt's files to void1_*
  as soon as its PAIR line says VOID (was: at its re-run), and t3r2_verdict.py counts a prompt only if its last PAIR line
  (main or re-run) says ok. A process cut before its re-runs (cap, the hard stop) can no longer leave a paused prompt
  counted. Dry test /Users/kamen/Fit12/scripts/fit_t3_dry.sh now has 4 cases (o3: cut mid main pass, o4: cut mid re-runs;
  plus a copy of o4 with the void files put back under live names: the PAIR rule alone drops all 9).
- [K] CHECK (Tue 22:4x; Leap 50de28c S7B-PROBE: inside one command buffer a CPU rewrite of a slot the GPU already read is
  not seen by later dispatches): the T3 path is NOT in the at-risk pattern, by construction, read in v3p (read-only):
  (1) a sync separates pass 1 and pass 2 (pass 2's rows come from pass 1's certainties on the CPU: mx.eval + tolist) and
  ends every <= 4-row exact call; (2) inside a call each pool is posted once per layer (MAXK 40, SMAX 4; wide pass 1 posts
  base and LSB rows to different pools) and the call's ids fit the pool (pass 1 <= 32 + 6 PF2 vs C_m 47, <= 32 vs C_l 32,
  exact calls <= 40); (3) the loader (ext make_room / flag mode) protects the whole row and takes victims from the LRU tail,
  so phase-B late reads and PF2 prefetches land in slots last read a call earlier (an earlier command buffer, a full
  48-layer forward of traffic between); (4) the prefill takes the sync path (mx.eval before a scratch buffer is refilled).
  Data beyond run-to-run: G1 PASS at C_m 47 / C_l 32 (t3g, 4 prompts: the exact loop after pass 1's pool churn = the loop
  alone, tokens + logits), G1 PASS at C_m 51 (t0g, 3 prompts), T3R's t80ck2 = T3t's with other arms run before it, C_m
  51 / 59 / 65 bitwise on the MTP path. Offered: K12, a direct A/B (T3R2's first 4 prompts with a sync at every MoE layer,
  bitwise vs T3R2's own r47 outputs, same order and history), ~5 min, any lock; wired only on the orchestrator's yes.
- K12 (orchestrator 22:5x: yes; run right after T3R2 inside the 13:05 slot if >= 6 min remain, else first thing at 13:45,
  ~5 min, any lock; send the verdict at once). Written before any run. [K] A/B, bitwise only: T3R2's first 4 prompts
  (156 42 68 91), its arms (t80ck2, t80ck1um3, t80ck1u) and ALT order, CAP 256, C_m 47 / C_l 32, again in a fresh process
  with mx.eval at every MoE block's entry and exit (fit_t3.py F12_K12SYNC=1 -> /Users/kamen/Fit12/scripts/k12sync.py wraps
  mtplx qwen4_exp.SparseMoeBlock.__call__ after load_b12; no engine file changed): the same requests and evictions, but no
  command buffer holds two reads of a pool with a loader rewrite between them. Compared with T3R2's own MAIN-pass outputs
  (void1_* when a prompt was paused and re-run: a PAUSE changes timings, not bits). Read-out
  /Users/kamen/Fit12/scripts/k12_verdict.py: PASS = every comparable (prompt, arm) equal on tokens + logits hash + per-check
  log; FAIL = any pair differs ([K]-type stale reads in the unsynced run, or timing-dependent bits); VOID = the sync not
  confirmed (wrapper on, > 0 synced calls, still installed at the end) or nothing comparable. Runner: scr_t3r2 after p47
  when >= 360 s to the deadline, >= 315 s of lock, SSD <= 66 C, no PAUSE and r47's main pass reached 156 (cap <= 330 s,
  the deadline - 20 s, the lock - 15 s); else scr_k12 at 13:45 under a shared lock (NO-SWAP >= 20 GB): Bash task
  bc5k3e50x, pid 61695 (it returns at once when K12 already finished in the slot). Dry tests: k12_dry.sh (5 read-out cases),
  scr_t3r2_dry.sh cases J-N; k12sync.wrap unit-tested on a dummy block.

## T3R2 + T3R-PF + K12 RESULT (ran Wed 4 Nov 13:05:43-13:17:16 in one quiet lock, 13:05:12-13:17:16 = 724 s of the 1791 s
## granted; rc 0 / 0 / 0, PAUSE seen 0 in all three, swap flat 10.36 -> 10.28 GB, SSD 25 -> 62 C; GPU free 23 min before
## the 13:40 hard stop; data + read-outs /Users/kamen/Fit12/data/scr/t3r2: verdict_t3r2.txt, verdict_k12.txt, p47_layers.txt)
- T3R2 VERDICT (pre-registered, QUEUE 4dbe21a + amendment 1ba42c3; /Users/kamen/Fit12/scripts/t3r2_verdict.py): PASS.
  Pooled t80ck1u / t80ck2 on the shared text x1.148 over 16 of 16 counted prompts (each: live files of both arms + a clean
  last PAIR line; no PAUSE, no re-run, none excluded), coverage 80 % / 77 %: t80ck2 2639 tok / 52.86 s = 49.9 tok/s,
  t80ck1u 2638 tok / 46.01 s = 57.3 tok/s. Per prompt x0.78 (91) .. x1.46 (161); 12 of 16 at >= x1.10 (below: 42 x1.01,
  56 x1.02, 63 x1.03, 91 x0.78). Beside it, T3R's original 8: x1.138 (coverage 88 % / 84 %). INFO t80ck1um3 (P1-K1 at
  3 bits) / t80ck2: x1.268 on 16 (x1.254 on the original 8). Full outputs (texts differ after the cut; not the metric):
  t80ck2 40.77, t80ck1um3 45.40, t80ck1u 44.17 tok/s. Checks: '] done'; lifetime max 11.354 GB (p47 11.329, k12 11.311;
  line 11.95); t80ck2 = T3t on 156/42/68/91 (tokens + logits); K1U engaged (16 runs); 16 PAIR lines, all clean.
- T3R-PF VERDICT (P2-PREFETCH B 6, exact add-on, pre-registered): KILL on both. t80ck1upf / t80ck1u 41.40 / 45.94 tok/s =
  x0.901; t80ck2pf / t80ck2 48.52 / 52.36 = x0.927 (same tokens, so a clean speed ratio; arm order balanced by ALT, 2 / 2).
  Exact: GATE PF2 8 / 8 PASS (tokens + logits); no PAUSE inside its prompts. Why (t3p_layers.py on p47_calls.csv): PF2 posts
  ~6 extra experts on every pass-1 layer call, adding ~150-160 misses to every pass 1 (t80ck2 729 -> 891 misses, 970 -> 1309
  MB per pass, loader wait 96.8 -> 122.6 ms; t80ck1u 341 -> 492, 461 -> 768 MB, wait 53.1 -> 85.6 ms), but a pass 2 follows
  pass 1 on only 10 of 65 / 19 of 74 checks (the other exact checks, 4 / 15, skip pass 1), and the pass-2 / exact passes save
  only 15 / 9 misses each (exact phase 203.0 -> 183.8 / 229.0 -> 223.3 ms). Net misses saved = ~3-4 % of the extra pass-1
  reads; net +19 / +24 ms per check (totals +1.24 s / +2.15 s). Closed in this form: a prefetch pays only where pass 2 is
  likely and its experts predictable, and neither held here.
- K12 VERDICT ([K] A/B, orchestrator 22:5x; /Users/kamen/Fit12/scripts/k12_verdict.py): PASS 12 / 12. With mx.eval at every
  MoE block's entry and exit (15,964 synced block calls, wrapper still installed at the end), all 12 (prompt, arm) pairs on
  156/42/68/91 are bitwise equal to T3R2's own unsynced run (tokens + logits hash + per-check log). Ran inside the slot
  (13:15:10-13:17:16, 125 s); the 13:45 fallback (task bc5k3e50x, pid 61695) disarmed 13:19:21. No [K]-type stale reads on
  T3's path, as the [K] CHECK above said: no past PASS affected. Sent to the orchestrator + Breakthrough 13:1x.
- ADOPTION (orchestrator 13:2x, correcting my report): t80ck1u needs no separate Kamen call. Under the 11-02 two-track rule
  the FAST track on every line adopts on EvalSuite same-scores; the exact 12 GB default stays as it is. Next: EvalSuite
  same-scores for t80ck1u at 12 GB, queued behind tonight's RELAX + T0 (NewIdeas orders the 12 GB queue). No Fit12 GPU in
  Kamen's day (13:45-23:30) unless he routes it.

## TRL STORE + MAP SCREEN 7 (Wed 4 Nov 14:1x-14:4x; the orchestrator 14:3x relaying Kamen's direction: published methods
## that work must be tested in their FAST form; NewIdeas' pre-registration QUEUE 14:3x, commit 3de406e; written before any arm)
- Why: B3 / B10a / B10-r (25 Oct) killed the data-aware trellis codes on an EXACT-style bar (weight error vs MSB-3, line 1.5x).
  The FAST track is judged by task results, so the question is new: the decision map with sure rows served from TRL.
  NewIdeas' replays (QOM + DRAFT17, 24 GB): TRL-2 code 4.0 / the 2.40 b mix 7.2 / today's MSB-3 FAST 19.5 MB per kept [P].
- Encode (/Users/kamen/Fit12/scripts/fit_trl_store.py, CPU only): B10-r's data-aware recipe for EVERY expert of the chosen
  layers, from B10-r's stored capture (/Users/kamen/Fit12/data/b10rcap: layers 2 7 12 18 24 30 36 41 46, fit rows R0..R7,
  32,760 rows): RHT (layer seed for gate / up, 1000 L + e for down) + bf16 row RMS + BlockLDLQ + bitshift trellis L 12 on
  16 x 16 tiles, 3INST table. Deviations from B10-r: damping FIXED per projection at B10-r's majority validation-best
  (gate 31/36 and up 32/36 at 0.3, down 18/36 at 0.1) instead of a per-matrix search (3x the encode); down's Hessian built
  in fp32 (numerically negligible). Form = the 2.40 b mix (gate / up TRL-2, down TRL-3).
- Harness view: the decoded weight W = Q R^T per (layer, expert, projection) as an MLX 8-bit affine view (group 64, bf16
  scales / biases): /Users/kamen/Fit12/data/trl/L{L}/e{EEE}_{proj}_k{k}.safetensors, 1.74 MB per matrix, 5.22 MB per
  expert, 2.67 GB per layer (4 layers 10.7 GB). Reader for Breakthrough's hook: /Users/kamen/Fit12/scripts/trl_view.py
  (dm_t3's read / _proj contract, gather_qmm bits 8; layout test vs dense 2-4e-7). MLX's CPU qmm with bf16 activations is
  only bf16-accurate (~5 % here); the GPU kernels accumulate in fp32.
- Measured (layer 18, experts 0-6): in-sample error vs MSB-3 gate 1.32-1.36x, up 1.31-1.38x (TRL-2), down 0.54-0.65x
  (TRL-3), consistent with B10-r (held-out TRL-2 1.74-1.85x, TRL-3 ~0.5x); the 8-bit view adds +0.18-0.54 % of TRL's own
  error. Speed at 4 threads: 35-38 s per expert under taskpolicy -b (E-cores, 5.1-5.4 h per layer); 9.8 s at normal priority.
- Signal: the map's code set has ~3,300 rows (1 flip = 0.03 points) and C sits 9 flips under the 1.34 % line, so a 4-layer
  TRL effect is ~1 flip: flips cannot decide a layer subset. NewIdeas' map_kl.py (per-row KL vs REF2-T from the stored top-8
  log-probs; flips = a + b sqrt(KL), rms 0.07 points on code over ten full-model arms) makes the paired subset informative.
- Screen 7 (NewIdeas' registration): S = {7, 18, 30, 41}; arms on qom_0_live_ball_c239_t3 (C + DRAFT17 at the mix's
  239-slot pool): 7-ref (all MSB-3), 7-trl (S from TRL for every kept expert of gated rows), 7-cal (S at MSB-2), 7-cal-all
  (all 48 layers at MSB-2). est_KL = ref + (trl - ref) x (cal_all - ref) / (cal - ref) -> est flips by the fit; lines PASS <=
  1.34 / 5.42 / 3.69 %, BETWEEN <= 1.5x, KILL above. NewIdeas' prediction: code ~1.0 % PASS, IFEval ~7.3 %, multi ~6.0 %.
  Mine (before any arm): code ~1.0-1.2 % (PASS), IFEval ~8-9 % (BETWEEN / KILL), multi ~6.5 % (KILL): the mix's gate / up
  error is ~1.5x MSB-3's, so the precision cost of sure rows rises by ~half; the 239-slot pool helps 7-ref, not the delta.
- Timing (the orchestrator 14:3x): encode at NORMAL priority, 4 cores, in chunks of 4 experts that wait while any other
  session's quiet ticket is queued or held (/Users/kamen/Fit12/scripts/trl_chunks2.sh: peer wait + a share lock per chunk),
  layer by layer 18, 30, 7, 41; hard stop 21:30; ETA ~20:40; arms ~22:00 beside T0 (share lock). Layer 18 is kept as the
  bit-reference for a Metal encoder.
- Option B, the full-model verdict (Kamen's call after tonight): CPU is out of reach (24,576 experts: ~65 h at normal
  priority, ~250 h in the background). A Metal encoder (Leap): first-order 0.5-1 h of GPU for all experts, ~0.5-1 day to
  build, checked against layer 18; Hessians for all 48 layers from a ~5 min GPU capture (~8 GB, deleted after); codes on
  disk: mix 36.2 GB, TRL-2 31.4 GB, TRL-3 46.7 GB (an 8-bit view would be 128 GB); a decode path in the map (Leap's B10(b)
  TA / TB trellis-decode kernels exist for k 2 as a benchmark; k 3 and the map wiring are new).
- Progress (Wed 4 Nov 17:05 check; encoder log /Users/kamen/Fit12/data/trl/encode_mix.log): layer 18 whole at 16:26:10
  (1,536 files of 1,741,125-1,741,130 B; 2.6 GB with the 105 MB H_x cache; experts 0-1 from the 14:22 smoke, the rest in
  the runners' logs); layer 30 at 231 of 512 at 17:08; then 7, 41. In-sample error vs MSB-3, mean (min-max):
    layer 18, all 512: gate TRL-2 1.346x (0.950-1.426), up TRL-2 1.356x (1.020-1.424), down TRL-3 0.603x (0.358-0.702);
                       the 8-bit view adds +0.21 / +0.21 / +0.48 % of TRL's own error.
    layer 30, first 232: gate 1.435x (1.277-1.496), up 1.436x (1.270-1.507), down 0.627x (0.384-0.723); view +0.20 / +0.19
                       / +0.47 %. Layer 30's TRL-2 sits ~7 % above layer 18's.
  Rate: 10.16 s per undisturbed expert (~350 per hour with chunk overhead); 283 per hour over 14:32-17:05 (13 fanwatch heat
  freezes, each matching a 'freezing the research jobs' line in /Users/kamen/Library/Logs/fanwatch.log, cost 1,061 s; CPU
  contention from other sessions' shared runs 454 s; one wait on another session's quiet ticket 269 s); 326 per hour on
  layer 30. No stall, no encoder error, disk 148 GB free, swap +0.2 GB. ETA ~21:10 at 326 per hour; heat pauses under the
  20:00 map batch could push it to the 21:30 stop. Breakthrough's batch (/Users/kamen/Breakthrough/scripts/dm_s7/s67_slot.sh,
  from 20:00, share locks) runs s7_trl only once dm_t3.trl_gate sees all 512 experts whole at the four layers, re-checked
  every 2 min up to its 23:10 last start. The orchestrator 17:1x: if the store is not whole at 21:30, the same runner goes
  on to a 22:45 end (/Users/kamen/Fit12/scripts/trl_relay.sh re-starts it with its clock check at 22:41, so the last chunk
  ends by 22:45, before BigMoE's 23:00 ticket and RELAX at 23:30).
- Progress 19:04: layer 30 whole at 18:32:34; layer 7 at 141 of 512; 1,165 of 2,048 experts. 221 per hour since 17:08
  (the map batch started early and its heat brought 13 fanwatch freezes, 28 min, plus slower experts; undisturbed experts
  still 10.16 s). In-sample error vs MSB-3, mean (min-max):
    layer 30, all 512: gate 1.436x (1.277-1.513), up 1.439x (1.270-1.511), down 0.630x (0.384-0.723); view +0.19 / +0.19
                       / +0.46 %.
    layer 7, first 146: gate 1.300x (1.150-1.355), up 1.310x (1.157-1.375), down 0.583x (0.192-0.687); view +0.22 / +0.22
                       / +0.50 %. So far layer 7 < 18 < 30 on every projection.
  ETA ~22:10-22:25 if the heat eases after the batch's GPU arms.
- Progress 19:35: layer 7 at 275 of 512, layer 41 at 0 (1,299 of 2,048); 259 per hour since 19:04 (the map process takes
  ~2.3 cores beside the encoder; 3 heat freezes since 19:00); 256 per hour over the whole run since 14:32. Expected: layer 7
  whole ~20:25, layer 41 ~55 % at 21:30, all four ~22:15-22:30 (at the 22:41 cut: 100 % at >= 245 per hour, ~87 % at 220).
  No stall, no error, disk 144 GB free, swap below the start.
- The orchestrator 19:3x, if layer 41 is not whole at the 22:41 cut: no diluted run (no MSB-3 fallback for the missing
  experts). Layer 41 is finished Thu from 06:00 on CPU (/Users/kamen/Fit12/scripts/trl_thu.sh: only if the store is not
  whole and records3 is not yet converted; last chunk start 08:30). s7_trl then runs in a ~10-min gap after BigMoE's
  Thu ticket (~08:55), all before the ~10:55 records3 conversion.
- Reader guard (Leap's records3 reader triage, 18:5x): Thursday's in-place records3 conversion (native+p11+pad, a per-layer
  "layout" key in index.json) would make fit_q.read decode permuted codes and 11-bit-packed scales silently. read() now
  raises unless the layer's layout is row-major (19:03:59; scratch-copy test: 180 arrays identical on today's store, a
  converted entry refused, also under python -O).
- Hand-off (the orchestrator 21:18): layer 7 was whole at 20:50:41 (not ~20:25 as expected at 19:35); layer 41 had 105 of
  512 experts at 21:20 (~210 per hour). Its remaining experts go to Leap's batched Metal encoder, which is bit-exact with
  this C path (layer 18: 1,536 / 1,536 matrices; layer 41 experts 10 / 50 / 90: 9 files, re-checked here with cmp), so
  screen 7 can run tonight. My runner stops on its 21:30 stop time at a chunk boundary (the 21:30 re-start relay was
  cancelled at 21:21). Leap encodes experts N-2..511 into its own dir after MTP Head's run (~21:47). I import them only
  after checks (/Users/kamen/Fit12/scripts/trl_import.py):
  - every file whole, with my tensors and metadata;
  - every file both dirs hold byte-identical, including a 2-expert overlap of the production run;
  - experts 300 and 511, re-encoded here by the C path (/Users/kamen/Fit12/scripts/trl_spot.sh), byte-identical to Leap's.
  Each file is copied to a temp name and renamed, so Breakthrough's gate never sees a partial file. The import checks were
  unit-tested on Leap's 9 files and on truncated, wrong-expert, wrong-projection and missing files.
- STORE WHOLE 21:40:48 (Wed 4 Nov): 6,144 / 6,144 files, 4 x 1,536, 10 GiB, no temp files left.
  - Layer 41: experts 0-147 by the C path (runner ended on its 21:30 stop time, 21:30:12, contiguous, no partial expert);
    148-511 by Leap's Metal encoder. Leap ran 146-511, 366 experts, in 340.8 s (21:34:38-21:40:20, 0.931 s per expert,
    batch 8, beside MTP Head's run, no swap growth; /Users/kamen/Leap/data/trl41_finish.log).
  - Import (trl_import.py 148 300,511, 3 s): 1,092 source files whole with my tensors and metadata. The 15 files both dirs
    hold (experts 10 / 50 / 90 and the in-run overlap 146 / 147) are byte-identical. Experts 300 and 511, re-encoded by the
    C path at 21:30-21:32, are byte-identical to Leap's (6 files).
  - Breakthrough's gate, run read-only here: dm_t3.trl_gate([7,18,30,41], 'mix') reports nothing missing, 512 / 512 experts
    per layer, without DM_TRL_FALLBACK. trl_view.py test OK at layers 7 / 30 / 41 (max rel diff vs dense <= 6.3e-07).
  - In-sample error vs MSB-3, mean (min-max), gate / up (TRL-2) / down (TRL-3):
    | layer | experts | gate | up | down | 8-bit view adds |
    | 7 | all 512 | 1.299x (1.098-1.371) | 1.311x (1.157-1.377) | 0.593x (0.192-0.687) | +0.22 / +0.22 / +0.49 % |
    | 18 | 510 logged | 1.346x (0.950-1.426) | 1.356x (1.020-1.424) | 0.603x (0.358-0.702) | +0.21 / +0.21 / +0.48 % |
    | 30 | all 512 | 1.436x (1.277-1.513) | 1.439x (1.270-1.511) | 0.630x (0.384-0.723) | +0.19 / +0.19 / +0.46 % |
    | 41 | e0-147 (C path) | 1.508x (1.234-1.597) | 1.526x (1.394-1.590) | 0.634x (0.427-0.727) | +0.19 / +0.18 / +0.46 % |
    Layer 18 correction: the earlier "all 512" figures were over the 506 experts in encode_mix.log. With the first
    runner's e2-e5 added (510) the means are unchanged at 3 decimals. e0 / e1 came from the 14:22 smoke, whose lines are not
    in the encode logs. Leap's run logs no error figures, so layer 41's row covers e0-147 only.
  - Order: layer 7 < 18 < 30 < 41 on every projection, so deeper layers lose more to the trellis codes relative to MSB-3.
  - Encode cost: the C path did 1,678 experts in 6 h 58 min (14:32-21:30, ~241 per hour, with heat freezes and other
    sessions' load; ~10 s per clean expert, 4 threads). Leap's Metal encoder takes 0.931 s per expert, so the whole model's
    mix store is ~6.4 h vs ~65 h on CPU (option B's cost; Leap's log).
  - Fallbacks cancelled: Thu 06:00 finisher (task b9uxbzxp6) and the 22:44 branch cron. s7_trl runs in Breakthrough's map
    batch tonight. The screen's verdict is NewIdeas' map_kl read-out, to be recorded here and in TRIED when it lands.
- SCREEN 7 VERDICT (NewIdeas d37b3ef, 22:2x; s7_trl ran 22:15-22:21, TRL 1,022-1,047 experts per window, fallback 0): the
  mix = BETWEEN / KILL / KILL (code / IFEval / multi, in-frame 1.45 / 9.54 / 10.44 % flips). Its damage corresponds to an
  effective error ratio r_eff 2.72 / 2.55 / 3.15 vs MSB-3, against the 1.62-1.87 that the MSE ratios predicted.

### ROUTED JUDGE PER DOMAIN (Wed 4 Nov 22:39-22:46; NewIdeas 22:33 for the orchestrator's QUEUE 22:3x, pre-registered 5fa0f47)
- Question: is screen 7's miss the inputs' distribution? The store was fit on a broad 7-domain mix; the map runs code,
  IFEval and multilingual answers.
- Pre-registered reading (NewIdeas):
  - mix >= ~2.4 on code / multilingual inputs: distribution explains the miss (a GPU probe of TRL-3 gets built);
  - <= ~2.0: trellis errors cost more KL per unit error on this model, and the family stops;
  - in between: the orchestrator's call.
- Method (/Users/kamen/Fit12/scripts/fit_trl_judge.py, fit_trl_judge_map.py; CPU, 4 threads, share lock, 2.4 + 1.1 min):
  - For every expert of layers 7 / 18 / 30 / 41, on the rows routed to it, the error against W4 (the 4-bit model) of the
    store's 8-bit view and of MSB-3: gate / up ||(W - W4) x||^2; down ||(W - W4) h4||^2 on the expert's own W4 hidden;
    whole expert ||y(W) - y(W4)||^2 with all three projections replaced. Ratio = pooled store error / pooled MSB-3 error.
  - Map inputs: Breakthrough's xcapture (/Users/kamen/FableGod/decision_map/out/xcap, read-only). These are the 36 map
    windows, 10,316 answer rows (code 3,366 / IFEval 3,397 / multi 3,553), with rank-sorted router ids and weights. Ranks
    2-9 = the experts s7_trl served from the store; whole_w weights each pair by router weight squared.
  - B10-r capture (/Users/kamen/Fit12/data/b10rcap): held-out R8 + R9 (8,190 rows), the fit rows R0-R7 (in-sample) and U
    (English prose). Per-token domains come from the parts' round-robin layout (/Users/kamen/Fit12/scripts/fit_dom_labels.py),
    which matches tokens_by_domain in all 10 parts. Captured row r = token r (rows 0-2 are identical across the parts, as their
    first 3 tokens are).
  - Checks: the error code reproduces the encoder's own in-sample numbers to ~0.2 % (L41 e0, L7 e5, all fit rows); the
    view decode equals mx.dequantize exactly.
- Map inputs, ranks 2-9, L7 / L18 / L30 / L41:
  | domain | whole_w | gate / up (TRL-2) | down (TRL-3) |
  | code | 1.12 / 1.22 / 1.27 / 1.31 | 1.40 / 1.60 / 1.67 / 1.70-1.72 | 0.61-0.67 |
  | IFEval | 1.58 / 1.65 / 1.67 / 1.83 | 2.03 / 2.21 / 2.31 / 2.48 | 0.76-0.79 |
  | multi | 1.59 / 1.65 / 1.68 / 1.75 | 2.06 / 2.24 / 2.31 / 2.43 | 0.71-0.79 |
  Ranks 0-1 are within ~5 % of ranks 2-9. Router weighting moves ranks 2-9 by <= 1.1 %.
- B10-r held-out R8 + R9, routed (top-10 set), L7 / L18 / L30 / L41:
  | domain | whole | gate / up | down |
  | code | 1.00 / 1.06 / 1.12 / 1.25 | 1.25-1.64 | 0.55-0.60 |
  | multi | 1.29 / 1.18 / 1.23 / 1.38 | 1.56-1.86 | 0.62-0.67 |
  | instr (IFEval prompts only) | 1.24 / 1.13 / 1.18 / 1.40 | 1.47-1.85 | 0.62-0.66 |
  | all 7 domains | 1.21 / 1.20 / 1.25 / 1.42 | 1.53-1.87 | 0.63-0.65 |
  | U prose (out of domain) | 1.62 / 1.71 / 1.76 / 1.92 | 2.09-2.64 | 0.78-0.80 |
  In-sample (fit rows) whole is 0.90-1.30. Code is the store's best domain everywhere; ratios rise with depth.
- Reading:
  - The map's chat answers are harder for the store than the held-out parts: gate/up ~2.0-2.5 vs ~1.6, whole ~1.6-1.8 vs
    ~1.25. Code is not: whole ~1.1-1.3 either way. So the distribution explains part of the chat miss and none of the code miss.
  - Whole-expert ratio (comparable to a uniform MSB-family r, since down at TRL-3 is 0.6-0.8x): <= 2.0 in all three
    domains, so the pre-registered reading is that the family stops. The map damage is 2.1-2.4x this on code and 1.4-2.0x
    on chat.
  - TRL-2 gate/up ratio: code <= 2.0; IFEval / multi 2.0-2.5 (L41 at the 2.4 line), which is the orchestrator's call.
  - Which measure applies depends on how s7_cal's r is defined. NewIdeas and the orchestrator decide.
  - SETTLED (NewIdeas 22:49): the pre-registration (5fa0f47) names the whole-expert output error, because the SWEEP's r is
    a uniform per-expert error ratio (MSB-2 = 5x MSB-3). Whole-expert is <= 2.0 in all three domains, so the TRELLIS
    FAMILY STOPS. NewIdeas cites these files in its QUEUE 22:4x and the SWEEP (fb39925, 3d45584). It reads the leftover gap
    (KL implies 2.6-3.2 vs 1.1-1.8 here) as partly noise, and asked Breakthrough whether s7_trl and s7_cal touch the same reads.

## HALF-K CHECK (Thu 5 Nov 18:56-19:02; the orchestrator 18:5x: check Fable A's numbers independently)
- What: Fable A's HALF-K (~/FableGod/LOSSY350_A.md section 1). Per row, deflate the layer's 32 dominant input directions
  (their response exact); per 64-wide group keep the c_x largest |x_perp| for gate / up and the c_h largest |h| for down.
  Weights touched 0.68 / 0.59 / 0.51 = x48 h32 / x40 h32 / x32 h32. Fable A: 0.47 / 0.71 / 1.22 x MSB-3, on 40 experts with
  the basis and held-out rows from one capture.
- Here (/Users/kamen/Fit12/scripts/fit_halfk_check.py, CPU 5.3 min, swap +0):
  - All 512 experts at layers 2 / 12 / 24 / 36 / 46. V = top-32 eigenvectors of B10-r's fit rows R0-R7 (uncentered).
  - Judged on B10-r's held-out R8+R9 and on the map's own answer rows (Breakthrough's xcapture, read-only).
  - Experts come from my own loader (pread of ~/BigMoE/models/bare; MSB-3 rebuilt as records3 held it). The records3
    conversion at 10:28 makes fit_q.read refuse, by design. --validate reproduces Wed's records3-based judge ratios
    (L41 e0, L7 e5) to 4 digits.
  - Metric: whole-expert error x MSB-3 (MSB-3's own relative error 0.039-0.066, rising with depth).
- Results, layer mean (range over the 5 layers):
  | weights touched | arm | held-out R8+R9 | map answer rows | Fable A |
  | 0.83 | down alone h32 | 0.33 (0.21-0.47) | 0.34 (0.23-0.51) | 0.36 |
  | 0.68 | x48 h32 | 0.44 (0.28-0.57) | 0.46 (0.31-0.61) | 0.47 |
  | 0.59 | x40 h32 | 0.70 (0.45-0.80) | 0.75 (0.50-0.86) | 0.71 |
  | 0.51 | x32 h32 | 1.26 (0.88-1.43) | 1.40 (1.01-1.59) | 1.22 |
  Plain (no deflation): x40 h32 0.91 / 0.89 (Fable A 0.91); x32 h32 1.80 / 1.76 (1.70). Map ranks 2-9 sit within 0.03 of
  all ranks. Layers 12 and 24 are the hardest, layer 2 the easiest.
- Reading: Fable A's frontier reproduces on held-out text and is 5-15 % worse on the map's own answers. 0.59 of the weights
  at ~0.7-0.75x MSB-3 is the robust point. "Half the weights at ~1x MSB-3" is ~1.25-1.4x on these rows.

## 12 GB PLAN, LEVER 1 (Thu 5 Nov 19:4x-20:0x; Kamen 19:4x via the orchestrator: 12 GB LOSSY at 90-130 tok/s, then publish)
- Fit12's section of the 12 GB plan: /Users/kamen/Fit12/PLAN_12GB_90_fit12.md (for Breakthrough's PLAN_12GB_90.md).
- Measured [K, T3R2's logs]: t80ck1u's 261 ms check at 57 tok/s = draft 13 + pass 1 130 (wait 53, compute 71) + exact
  passes 105 (46 % of checks) + commit 12. By check mode on full outputs (44.9 tok/s): pass-1-certified checks are 55 %
  of checks, 41 % of the time, 72 % of the tokens; pass 1 + exact pass 2 28 % / 41 % / 22 %; cf-gated exact checks
  17 % / 19 % / 6 %.
- Lever 1 = a sharper certificate (more rows certified at the same divergence count vs REF2-T). Calibration arm g1k1u +
  F12_CAL=1 built in fit_t3.py / t3_twopass.py (env-gated; off = the unchanged path; backups *.bak_cal); waits for a GPU slot.
- LEVER 1 RESULT (19:52; /Users/kamen/Fit12/PLAN_12GB_90_fit12.md 'Lever 1 RESULT'): the calibration run (g1k1u + g1k2, 16
  HumanEval prompts, 640 s, all = ex) and the replay. The certificate is already token-exact (0-1 divergences in ~3,400
  tokens at every theta / cf), so lever 1 as framed is KILLED; the damage is pass 1's KV (Breakthrough). Calibrated speeds
  (full outputs, cf 0.9): P1-K1 theta 0.8 / 0.9 / 0.95 = 45.8 / 41.0 / 37.9 tok/s; top-2 = 40.2 / 37.6 / 35.9. The
  map's code-valid point (top-2 theta 0.95, 1.22 %) is ~36 tok/s on these prompts.
- LEVER 2 RESULT (20:0x; 'Lever 2 ... RESULT' in /Users/kamen/Fit12/PLAN_12GB_90_fit12.md): the top-k pass-1 form is built
  (_TOPK; BIGMOE_MAXK=80). top-4 / 5 cannot run as one 16-row pass at 12 GB (union 54-80 experts > C_m 47 slots per layer:
  engine abort). top-3 (16 / 16 = ex) has a pass 1 of ~207 vs ~170 ms; at theta 0.9 / 0.95, cf 0.9, it runs 34.5 / 33.1
  tok/s vs top-2 37.6 / 35.9 and P1-K1 41.0 / 37.9. So a wider pass 1 loses speed at 12 GB; the code-valid two-pass ceiling
  is ~34-38 tok/s on full outputs.

## 12 GB PLAN, LEVER 3 (Fri 6 Nov 09:4x; the orchestrator 09:3x: price Leap's Fri kernel switches on the 12 GB path)
- Code read: the 12 GB T3 path reads none of REF2_DN_KS / D7BNAX_32 / att_single / fs4 / FAST-NAX as they are (no ref2,
  d7b_nax or r57 in the process). Ported in a copy: the drafter on qmm16c + per-shape ks (exact), and pass 1's attention as
  one SDPA call + pass 1's 4-bit dense on qmm16c + per-shape ks (FAST, pass 1 only). FAST-NAX: no counterpart (v3p).
- Pre-registered lines, arms and the run state: /Users/kamen/Fit12/PLAN_12GB_90_fit12.md 'Lever 3'. Run: kp2 (share lock).
- RESULT (Fri 6 Nov 13:2x; kp2 12:59:03-13:17:31, share lock, rc 0, swap flat; 14 clean prompts, ROUGH; full table in
  /Users/kamen/Fit12/PLAN_12GB_90_fit12.md 'Lever 3 RESULT'): t80ck2 38.72 tok/s; + dn1 (pass 1's dense on qmm16c + ks)
  40.28 (x1.054, pass 1 -17.5 ms per run on 14 / 14) PAYS; + att1 39.16 (x1.013) BETWEEN; all three 40.10 (x1.047); the
  drafter port -3.2 ms per check (exact loop ex 18.58 -> exd 18.93, +0.5 % from its draft part). exd = ex on 14 / 16: the
  known keep-1 replay dependence (WV3; PR1 not in the pinned engine), not a port bug. Correction: t80ck2 (40.8 in T3R2)
  is FAST, not the exact mode; the 12 GB exact loop is ~18.5 tok/s on HumanEval.

## 12GB-CLEAN (pre-registered Fri 6 Nov 21:3x; the orchestrator 21:3x, NewIdeas QUEUE f2243e8)
- The 12 GB verified-greedy loop (arm ex, not exact) alone on all 16 HumanEval prompts, quiet, Saturday morning after tonight's
  chain: per-prompt decode tok/s + geometric mean, its own lifetime peak footprint, the engine's settings. Runner, gate,
  report and validity: /Users/kamen/Fit12/PLAN_12GB_90_fit12.md '12GB-CLEAN'. Armed 21:37 (ex16.sh).
- Re-armed Fri 23:34 for the first clear gap (the window opened at 23:30; ex16.sh task b6utph004; WAIT file = NewIdeas' hold).
- 12GB-CLEAN RESULT (Sat 00:3x): 18.42 tok/s decode, geometric mean over all 16 HumanEval prompts (pooled 18.47); lifetime
  peak 11.144 GB = 10.38 GiB; alone in its process, quiet lock, rc 0, no PAUSE, outputs = kp2's ex 16 / 16. Verified
  greedy, not exact. Details: /Users/kamen/Fit12/PLAN_12GB_90_fit12.md '12GB-CLEAN RESULT'.
- t80ck1u MEASURED (Sat 08:19-08:21, quiet, alone, 16 HumanEval, 12 GB FAST candidate): 46.33 tok/s whole decode geometric
  mean (pooled 44.34), full-period 49.14, end to end 24.49; lifetime 11.245 GB; x2.52 the verified-greedy loop. Details:
  /Users/kamen/Fit12/PLAN_12GB_90_fit12.md 't80ck1u MEASURED'. EvalSuite's entry point: fit_t3.py F12_IDS_OUT (same section).
- WIRED A/B (Sat 15:49-15:55): mx.set_wired_limit on the 12 GB loop (t80ck1u) = x1.004 whole decode, tokens W = B 16 / 16:
  KILL at 12 GB; the first-check excess (4.7 % of decode) stays: cold pools after prefill, not un-wiring. Details:
  /Users/kamen/Fit12/PLAN_12GB_90_fit12.md 'WIRED A/B RESULT'.
- TABLE 12 GB CELLS (pre-registered Sat 17:2x): 12E (ex) + 12F (t95ck1u, theta 0.95) on S5's 12 HE + MBPP12, CAP 256, quiet,
  each alone: /Users/kamen/Fit12/PLAN_12GB_90_fit12.md 'TABLE 12 GB CELLS'.
- TABLE 12 GB CELLS RESULT (Sat 17:36-17:46): 12E verified greedy 18.58 (12 HE) / 18.18 (MBPP12) whole decode geo, 11.14 GB;
  12F theta 0.95 40.59 / 18.74, 11.28 GB. Details: /Users/kamen/Fit12/PLAN_12GB_90_fit12.md 'TABLE 12 GB CELLS'.
- 12 GB CHAT + PREFILL (pre-registered Sat 23:5x; NewIdeas PROGRAM CHAT 150): 12 IFEval on the verified-greedy loop (MB per kept
  token) + prefill A/B (s_moe 4096 vs 8192 on 1.6K / 4.7K / 7.9K), after CERT: /Users/kamen/Fit12/PLAN_12GB_90_fit12.md
  '12 GB CHAT + PREFILL'. t95ck1u failed EvalSuite's code gate (-5.00): the 12 GB lossy cell fails quality.
- 12 GB CHAT + PREFILL RESULT (Sun 05:07-05:12): chat 12.88 tok/s whole decode geo (12 IFEval, verified greedy), 2.34 kept per
  check, 556 MB SSD per kept token, 10.96 GB; prefill 191 / 378 / 461 tok/s at 1.6K / 4.7K / 7.9K (TTFT 8.5 / 13.0 / 17.5 s),
  already expert-once (s_moe 8192 = no gain), <= 10.95 GB. Details: PLAN_12GB_90_fit12.md '12 GB CHAT + PREFILL'.
