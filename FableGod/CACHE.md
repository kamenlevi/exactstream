# CACHE — the cache that knows the prompt, measured on the CPU (FableGod, mission 20, Mon 2 Nov 2026, 18:0x-18:1x EET)
Marks: [K] measured here (source named), [P] arithmetic on measured numbers, [G] judgement. Rules kept: CPU only (cache_replay.py, 7 min 9 s on the 36-window routing capture), no GPU, no downloads, writes only in ~/FableGod; cache policies change no token (exact); the two-pass check is quality-defined and marked so. TRIED positions in section 6.
Method [K]: DAY9's routing capture (rank-sorted top-10 expert ids per position and layer, prompt + answer, 12 windows per domain) replayed as one session per domain with the HPD-0 walks' real check boundaries and kept counts (391 code checks / 3,388 kept tokens; 1,373 IFEval / 3,416; 1,845 multi / 3,573). Prompt rows read position by position (exact top-10); each check reads the union of its rows' experts per layer; per-layer caches of C slots; misses counted on answer checks; 2.2 MB per expert record. Caveat carried from FLOOR: this replay's absolute misses run 1.25-1.4× the engine's measured ones (prefill here is position by position, the engine's is 1,024-row chunks), so absolute caps are pessimistic by ~1.3× and ratios are what to use; 12 windows per domain.

## 0. Verdict in six lines
1. **Prompt protection is dead** [K]: evicting the prompt's experts last raises misses per kept token at C 160 by +14 / +42 / +63 % (N = 10 / 30 / 100 checks) and at C 256 by +23 / +72 / +99 % on code; chat the same. It helps only where nothing fits (C 51: −2 / −7 / −11 %). The safari's 84-91 % coverage was a false lead: the prompt's set is 253 of 512 experts per layer, so protecting it fills a 256-slot cache and the answer's new experts thrash; LRU's recency is the better predictor (the first 100 answer rows cover 87-93 % of what follows).
2. **Pinning is neutral** [K]: the C/4 most-used experts per layer, from the prompt or from a global prior, move misses by −1.5 to +1 % at C ≥ 160 and hurt at C 51; TRIED's pinning KILL and SEED45's BETWEEN stand.
3. **LRU is within 1.6-2.0× of Belady's MIN** [K]: code 65.5 vs 41.2 (C 160), 29.6 vs 15.1 (256), 15.3 vs 7.9 (320); chat 43.3 vs 23.1, 39.9 vs 22.7 at C 160. The whole headroom of any eviction policy at 45 GB is ~2× on misses, i.e. the SSD-exposed ~39 ms of a 107 ms code cycle; no realizable signal tested here takes more than a tenth of it.
4. **The start-up** [K]: pre-loading the prompt's experts by frequency at prefill end cuts the first three checks' misses by 13 % at C 256 (1,637 → 1,417 per window), 8 % at C 160, and adds 1 % at C 51; the replay's first check costs 2× a steady check (506 vs 257 at C 256), the engine's 3-5× (K-80), so the engine's burst comes from its chunked prefill's evictions, which A6 fixes (first check 240-1,179 → 77 ms today [K a6r]). Pre-loading is a small second step after A6.
5. **The two-pass check holds cloud Fable's low-churn claim** [K, quality-defined]: on code at C 51 (12 GB) the SSD reads per check are exact 6.9 GB → two-pass with top-5 certified at p1 ≥ 0.9: 2.2 GB → two-pass with top-2 certified at p1 ≥ 0.99: 1.25 GB (the 125B-sure oracle 1.66): the top-2 design sits under the 1.8 GB line. SSD caps on code: 12 GB 16 → 51 / 90 tok/s; 24 GB (C 160) 90 → 194 / 243; 45 GB (C 256) 200 → 451 / 534. Chat at 12 GB: 6 → 20 / 44.
6. **Tok/s, conservatively, and the one engine test** (section 5): 12 GB code ~60-80 with the top-2 two-pass (SSD cap 90-117 after the replay's pessimism, then the second dense pass binds), ~40-50 with top-5; 24 GB ~120-150 (GPU-bound after the SSD leaves); 45 GB: the SSD leaves the path with either. The exact engine test for BigMoE is the prefill warm-up flag (PASS ≤ −8 % misses in the first three checks at C 256 with every token equal; KILL > −3 %); the two-pass needs mission 7's emulation first (quality-defined).

## 1. Policies against LRU: misses per kept token [K]
| domain | C (line) | LRU | +prot N=10 | +prot N=30 | +prot N=100 | +pin C/4 prompt | +pin C/4 prior | prot 30 + pin | Belady MIN |
|---|---|---|---|---|---|---|---|---|---|
| code | 51 (12 GB) | 362.4 | 354.6 | 336.3 | 321.2 | 344.9 | 340.0 | 331.8 | 260.6 |
| code | 160 (24 GB) | 65.5 | 74.4 | 92.7 | 106.9 | 65.8 | 65.4 | 92.6 | 41.2 |
| code | 256 (45 GB) | 29.6 | 36.5 | 50.8 | 58.8 | 29.3 | 29.1 | 50.7 | 15.1 |
| code | 320 (~51 GB) | 15.3 | 17.4 | 23.2 | 25.4 | 15.3 | 15.1 | 23.2 | 7.9 |
| IFEval | 51 | 1,066.8 | 1,041.0 | 995.5 | 870.9 | 1,123.7 | 1,046.1 | 1,064.3 | 920.9 |
| IFEval | 160 | 43.3 | 64.4 | 119.2 | 286.7 | 48.8 | 44.5 | 124.0 | 23.1 |
| IFEval | 256 | 16.7 | 18.3 | 21.9 | 35.4 | 17.3 | 16.7 | 22.4 | — |
| IFEval | 320 | 9.7 | 9.8 | 10.0 | 10.7 | 9.8 | 9.6 | 10.0 | — |
| multi | 51 | 1,005.8 | 986.3 | 949.5 | 866.6 | 1,083.9 | 1,055.3 | 1,025.8 | 857.8 |
| multi | 160 | 39.9 | 45.3 | 59.7 | 118.4 | 43.7 | 42.0 | 63.1 | 22.7 |
| multi | 256 | 17.6 | 17.6 | 17.7 | 18.3 | 18.2 | 17.8 | 18.3 | — |
| multi | 320 | 10.7 | 10.7 | 10.7 | 10.8 | 10.8 | 10.6 | 10.8 | — |
Reading: at every line that fits part of the working set, protecting the prompt's experts costs misses, more with longer protection; at C 51 nothing fits and holding the prompt's set beats pure recency by 2-11 %. Pinning by the prompt or by a prior is noise. Belady's MIN is 1.4× (C 51) to 2.0× (C 256) below LRU, the same headroom TRIED measured (44-48 %); the policies tried here are not the signal that reaches it. Chat at C 51 (12 GB) is 1,000+ misses per kept token: the 12 GB line is a code line.

## 2. The start-up [K]
| domain | C | misses in the first 1 / 2 / 3 checks per window, LRU | with the prompt's experts pre-loaded by frequency | steady misses per check |
|---|---|---|---|---|
| code | 51 | 2,726 / 6,321 / 9,957 | 2,840 / 6,435 / 10,070 | 3,140 |
| code | 160 | 1,138 / 2,582 / 3,701 | 1,092 / 2,362 / 3,414 | 568 |
| code | 256 | 506 / 1,151 / 1,637 | 459 / 1,000 / 1,417 | 257 |
| IFEval | 160 / 256 | 1,082 / 1,358 / 1,529; 409 / 520 / 588 | 1,028 / 1,284 / 1,451; 405 / 513 / 580 | 108; 42 |
| multi | 160 / 256 | 819 / 989 / 1,122; 478 / 568 / 639 | 822 / 991 / 1,124; 478 / 568 / 639 | 77; 34 |
Pre-loading saves 8-13 % of the first three checks' misses on code at C 160-256 (one check's worth of misses per answer), nothing on chat, and costs 1 % at C 51. The replay's burst is 2× a steady check; the engine's K-80 burst was 856-1,202 misses against ~250 steady (3-5×) because its prefill streams in 1,024-row chunks that evict the decode set, which this replay does not model; A6 addresses that directly and its arm r brought the first check to 77 ms [K]. Pre-loading is the second-order step after it.

## 3. The two-pass check at 12 and 24 GB (quality-defined) [K]
Pass 1 reads top-k1 for every row of the check; a row is certified when its own p1 under that cut is ≥ θ (the map's topk2 / topk5 runs on the same windows: top-2 at θ 0.99 covers 73.4 % of code rows at 0 flips, top-5 at θ 0.9 89.8 % at 0 flips [K FLOOR]); pass 2 reads ranks k1+1..10 for uncertified rows up to and including the first exact miss. The oracle reads top-2 on 125B-sure rows and top-10 elsewhere.
| domain | scheme | C 51: misses / MB per check, misses per kept, SSD cap | C 160 | C 256 |
|---|---|---|---|---|
| code | exact | 3,140 / 6,908 MB, 362, 16 tok/s | 568 / 1,249, 65.5, 90 | 257 / 564, 29.6, 200 |
| code | two-pass top-5 (θ 0.9) | 1,001 / 2,203, 115.6, 51 | 263 / 580, 30.4, 194 | 113 / 250, 13.1, 451 |
| code | two-pass top-2 (θ 0.99) | 569 / 1,252, 65.7, 90 | 211 / 464, 24.3, 243 | 96 / 211, 11.1, 534 |
| code | oracle (125B-sure top-2) | 754 / 1,659, 87.0, 68 | 141 / 310, 16.2, 364 | 63 / 139, 7.3, 810 |
| IFEval | exact / top-5 / top-2 / oracle | 2,654 / 748 / 336 / 1,791 misses; caps 6 / 20 / 44 / 8 | 108 / 74 / 88 / 74; caps 136 / 198 / 167 / 200 | 42 / 30 / 35 / 30; caps 353 / 490 / 415 / 489 |
| multi | exact / top-5 / top-2 / oracle | 1,948 / 307 / 215 / 961; caps 6 / 37 / 53 / 12 | 77 / 50 / 68 / 50; caps 148 / 230 / 169 / 231 | 34 / 23 / 31 / 23; caps 335 / 500 / 374 / 500 |
Prompt protection (N 30) under every scheme is worse or equal (table in cache_replay.out), so the "best policy from section 1" is LRU itself.
Reading: cloud Fable's Test 1 passes on this data: the realizable two-pass with the top-2 certificate reads 1.25 GB per check on code at C 51, under the 1.8 GB line, and its churn (569 misses per check) is lower than the 125B-sure oracle's (754) because rows after the first miss stay at the cheap tier. Its SSD cap at 12 GB is 90 tok/s on code (117 after the replay's ~1.3× pessimism); the top-5 certificate's 2.2 GB per check caps at 51 (66). On chat the two-pass cuts SSD bytes 6-8× but the line stays SSD-bound near 20-44 (26-57) because chat keeps 2.5 per check.

## 4. Why these policies fail where SAFARI's coverage suggested they would [P on K]
The prompt's 253 experts per layer cover 84-91 % of the answer's slots, but the answer's working set is 292 and its recent experts cover more (87-93 %); at C 256 protecting 253 experts leaves 3 slots for everything else, at C 160 it leaves none. LRU already keeps whichever prompt experts the answer keeps touching; what it loses is the churn among the ~40-90 experts per layer the answer brings in, which no prompt signal predicts. Belady's 2× says the churn is predictable only with the future, i.e. with the drafted tokens' routing (the runahead, M-16) or a per-layer predictor with high recall (R-57b's 0.75 was not enough at 45 GB).

## 5. Tok/s at each line, conservatively, and the engine test
- **Policy side (exact):** nothing to ship beyond A6; the prefill warm-up is worth one check's misses per answer (8-13 % of the first three checks at C 160-256, i.e. ~1-2 % of a 50-check answer) [P on K].
- **Two-pass side (quality-defined):** 12 GB code: SSD cap 90-117 (top-2) / 51-66 (top-5) [K/P]; the second dense pass (3.3 GB, 11 ms at the floor, ~36 today) and pass 2's rows add GPU time, so the engine lands ~60-80 with the top-2 certificate and ~40-50 with top-5 [G] (cloud Fable's 70-80 / near 60 bracket). 24 GB code: SSD cap 194-243 → GPU-bound at ~120-150 [G]. 45 GB code: SSD caps 451-534: the SSD leaves the path; the gain is then the rows and the second pass's cost, priced in FLOOR_125B 3.3 (it loses to the single-pass gate there). Chat at 12 GB: 20-44 → ~25-40 [G].
- **The one engine test (BigMoE, exact, 10 min, after A6):** flag PREFILL_WARM: at prefill end, insert each layer's prompt experts into the pools in frequency order (most used = most recent), up to the pool size; 3 code + 3 agent prompts at C 160 and C 256, two separate passes per arm; PASS misses in the first three checks ≤ −8 % with every token equal to REF2-T; KILL > −3 %. Everything else in this document is either a null result or quality-defined (the two-pass gate: mission 7's emulation, then decision 4).

## 6. TRIED
LRU vs LFU (KILL at 45 GB, LFU ADOPTED at 24): LRU is 1.6-2.0× from MIN here, so the eviction question at 45 GB is closed again. Pinning the hottest experts (KILL): confirmed, −1.5 to +1 %. Belady-trained / popularity-prior eviction (BETWEEN, −10 %): the ceiling measured here is −37 to −49 %; those policies took a quarter of it. Context matching (KILL, −4-6 %): the prompt's own routing is a stronger signal and still loses, for the reason in section 4. SEED45's popularity prior (BETWEEN): the prior-pinned C/4 here is noise. A6 (in progress, PASS on the split today): the start-up is the engine's chunked prefill, not the policy; pre-loading is its small companion. The two-pass check is cloud Fable's design (FLOOR_125B 3.3 priced it out at 45 GB; this replay prices it at 12 and 24 GB, where it is the lever).

## 7. Reproducibility
cache_replay.py → cache_replay.out (7 min 9 s). Inputs: decision_map/out/routing/*.npz, baseline.json, topk2.json, topk5.json, floor_rows.json, ~/TrainLab/data/d02b_walk_hpd0_{code,ifeval,multi}.npz. Belady at C 256-320 was run on code only (time); the policies are per-layer LRU with a protection map, a pinned set, or MIN by next use.
