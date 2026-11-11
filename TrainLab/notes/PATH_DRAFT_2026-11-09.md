<!-- Draft of MAIN section 11, replacing "What we left on the table" (MAIN l896-931). For the editor to merge (Mon 9 Nov). Keeps the content of the old seven items: 1 first token -> 11.1 PF-D row + 11.2 prefill wall; 2 long prompts -> 11.1 prefill rows; 3 chat decode -> 11.2 fixed-cost and kept walls + 11.3 chat; 4 ungated -> 11.1 chat stack and L rows; 5 code -> 11.1 never-wait row + 11.2 SSD wall; 6 memory -> 11.2 memory wall; 7 not measured -> last paragraph. Delete this comment on merge. -->

## 11. The path to more speed

Everything we tried is in the public ledger (FableGod/TRIED): one line per attempt, with its verdict, its source and the
wall it hit. Below: the gain of each change outside the shipped configuration (11.1), the walls (11.2), and the path
for chat and code per memory line (11.3).
**Sure** means measured [K] on this engine, in a run that had the change in it; a lossy change is sure as speed, and
its quality only once its gate passes. **Bet** means priced [P] from measured parts, or a judgement [G]. Gains multiply
only where they were measured in one run.

Today, as shipped at 45 GB (whole decode, 12 prompts per set): HumanEval 164.0, MBPP 112.7, math 78.9 and chat 50.7
tok/s, against 101.4 / 70.4 / 77.9 / 37.1 for the exact default [K, Tables 6.1a-6.1c].

### 11.1 If this is done, this is the gain

| change | line, workload | gain (whole decode unless noted) | mark | state |
|---|---|---|---|---|
| L by itself (CERT + the exact-safe bundle, without COH) | 45 GB, HumanEval / MBPP | +6.0 % / +4.7 % over CERT, pooled, one series (181.3 / 119.8); the shipped 164.0 / 112.7 is another snapshot | [K, BM:data/s5p/s5p_report.txt] | built; not gated by itself |
| the chat stack (zs + bundle / HC8_G + rank cut 8 + R1) | 45 GB, 12 IFEval | 56.0, +12.8 % over the run's first arm; cycle 44.1 ms | [K, BM:data/chs/chs5_report.txt] | built; ungated |
| zs32 (M1-B + COH + SB1) | 32 GB, 12 IFEval | 51.33, against 33.81 for the exact e32 | [K, BM:r57 zs32_160_runs.json, e32_160_runs.json] | built; not gated at 32 GB |
| never waiting for cold experts | 45 GB, HumanEval | +6.2 % pooled, under its +8 % line (BETWEEN) | [K, BM:data/r57/s1b{b,n}{1,2}_256_runs.json] | built; not shipped |
| PF-D beyond chat; a better next-layer predictor | 45 GB, first token | chat 2.91 -> 2.26 s exact (shipped); lossy form 1.83 s; reading only the missing experts ~1.6 s; 19-31 % of what v4 stages goes unused | [K, Table 6.1c]; 1.6 s [P, BM:data/pfb/pfb_report.txt] | built for chat; not run on other configurations or at 32 GB |
| exact layer-major prefill | 45 GB, 1.6K-7.9K-token prompts | prefill 1.9-3.4x | [K, F:2026-11-08_0800] | built; needs a 0.1 GB memory trim |
| one MoE call per layer in prefill (PF-B) | same | a further 12-21 %, exact; peaks 45.8-46.2 GB | [K, BM:data/pfd/pfd.log part B] | over the line |
| prefill reads overlapped with compute | same | ~590 / 750 tok/s prefill at 4.7K / 7.9K | [P, F:2026-11-08_0800] | needs building |
| two-row verify calls on chat (C2c) | 12 GB, 12 IFEval | priced x1.10-1.15 over P2'; measured x1.053 pooled (14.46 vs 13.73, share lock); code x0.881 against A' of an earlier run | [P, F12C]; [K, F12:logs/chw19_run.log] | screened: chat MARGINAL, code KILL |
| less Python on the critical path | 12 GB | at most x1.09 | [P, F12P] | test designed, not run |
| a lossy chat route | 24 GB, IFEval | 81.7 MB per kept token, 61 % under the exact rule's 210.2; passes the map at 147 slots | [P, SMALL_RAM 5.3; K, 4.4] | needs an engine run and a gate |
| the prompt pass at verify-call bandwidth | 12 GB, first token | 2.906 s for 63 tokens (21,986 MB at 7.6 GB/s) -> ~1.8 s at ~12 GB/s | [K, F12H]; [P] | needs building |

### 11.2 The limits

Per wall: today's limit; the size of its ledger section (TRIED.md, one wall per line, read Mon 9 Nov) and the best
result; what breaking it unlocks, and which wall then binds; what could break it.

| wall | today's limit | ledger lines; best result | unlocks; then binds | what could break it |
|---|---|---|---|---|
| fixed cost of a check | chat ~51 ms exact / ~40 ms lossy at 1-2 rows, against a memory floor of ~15-20 ms [K; floor P; F:2026-11-07_2355]; the fixed part is ~75-85 % of a chat check [P, [DRAFTER] App. C row 23] | 315; warm check + commit 232.6 -> 117.3 ms (T489, T476); chat cycle 48.5 -> 44.1 ms (CHS5) [K] | ~2x in the low-row check alone [P]; then kept tokens per check | low-row kernels at bandwidth rather than latency; fewer dependent kernels per layer ([KERNELS]) [G] |
| kept tokens per check | chat 2.5 of ~3 rows; math 6.5 of ~9; code ~11 of 16 [K, F:2026-11-07_2355; section 4.3] | 218; 41 of [DRAFTER]'s 83 items killed; best: the MTP head on chat, +24.5 % [K] | a drafter that never missed: ~105-115 tok/s of chat at today's check cost [P, [DRAFTER]]; then the fixed cost | a chat head trained on ~1.3M tokens of the model's own answers (8 % of that gave x1.014, under its x1.03 line [K, [DRAFTER] 3.1]); Strata's MTP layer accepts 87 % of its drafts [K, F:2026-11-08_1630] |
| SSD bytes per kept token | 45 GB code: 18.5 ms cold-minus-warm per 16-row check [K, F:2026-11-07_0025]; chat misses ~4 ms at 1-2 rows [K, F:2026-11-07_2355]; 12 GB chat 449-491 MB per kept token [K, F12:logs/chw19_run.log] | 303 SSD + 168 bytes; per-layer pools -17.2 % bytes, +8.3 % tok/s, exact; rank tiers about -20 % expert bytes [K, section 7] | at 45 GB code, at most the 18.5 ms per check; then the fixed cost | eviction nearer the offline optimum (OPT: 38-55 % fewer exact bytes than LRU, 386.8 vs 621.6 MB per kept token at 12 GB [P]); next-layer read-ahead (31-40 ms loader-idle per 12 GB call [P, F12P]) |
| the memory line | rows at 41.7-43.7 GB, at least 1.3 GB under; C 224 -> C 256 +0.1 % [K, F:2026-11-07_0500] | 101 | little decode at 45 GB; PF-B, 0.8-1.2 GB over the line [K]; at 32 / 24 / 12 GB memory sets the cache, so bytes per kept token bind | trimming PF-B's peak; the lossy routes of 11.1 [G] |
| first token and prefill | chat as shipped 2.26 s, reading 9.6 GB per prompt with PF-D [K, BM:data/pfd/pfd_report_0009.txt]; without it a 37-101-token prompt reads 20.4 GB, though 65.5 % of the routed 3-bit parts are in memory [K, BM:data/pfb/pfb_report.txt]; long prompts compute-bound, SSD at 3.8-5.2 of 14 GB/s [K, F:2026-11-08_0800] | 41; PF-D 2.91 -> 2.26 s exact; layer-major 1.9-3.4x [K] | ~1.6 s [P]; Strata's 0.54-1.01 s on the same prompts shows more is reachable here [K, section 9] | PF-D everywhere, a better predictor, reads overlapped with compute [P] |
| quality gates | lossy parts need not compose: relaxed acceptance (IFEval 87) and 8-bit mixers (89) together scored 84, against 90 [K, 10.6]; CD16 failed MBPP; the 12 GB FAST mode failed same-scores [K, 10.3-10.4] | 74 quality + 74 exactness | the ungated rows of 11.1; at 12 GB the FAST mode's 44.34 on HumanEval only [K, F12R:c1u16] | lossy rules that pass the gates, gated together [G] |

### 11.3 The path

Today, then 11.1, then each wall. Measured steps are sure; priced steps are bets.

- **45 GB, code (HumanEval / MBPP).** Today 164.0 / 112.7 [K]. Measured, not gated: L by itself, 181.3 / 119.8
  [K]. A bet: never waiting for cold experts, +6.2 % on an older configuration (BETWEEN) [K], never run with L.
  Past that, the SSD wall's room is 18.5 ms per check [K]; nothing further is priced.
- **45 GB, chat.** Today 50.7 [K]. Measured, not gated: the chat stack, 56.0 [K]. Bets: the check at its floor,
  about 2x in the check alone [P]; a drafter that never missed, ~105-115 tok/s at today's check cost [P]. 150 tok/s
  needs both walls at once: a ~25 ms cycle and ~3.75 kept per check [P, F:2026-11-07_2355]. First token: 2.26 s today
  [K], 1.83 s with the lossy form [K], ~1.6 s priced [P].
- **32 GB, chat.** Today 33.81 exact [K]; the shipped lossy-32 (L + COH at C 160) runs 39.0, not gated [K, [RELEASE]
  row 32]. Measured, not gated: zs32, 51.33 [K].
- **24 GB.** Code today 41.07 / 37.43 exact (E24, E24m) [K, SMALL_RAM 6.1]; chat not measured. A bet: the lossy chat route, 61 %
  fewer bytes per kept token than the exact rule [P].
- **12 GB.** Code today 18.34 verified [K, F12R:tab12e]; the FAST mode's 44.34 (HumanEval only) failed its gate [K,
  F12R:c1u16], so it is a bet until a lossy rule passes. Chat today 12.76 verified, 13.97 with P2' (not gated) [K,
  F12R:chat12, chw14c2]; two-row calls measured
  x1.053 over P2' (priced x1.10-1.15). Bets, not multiplied: less Python, at most x1.09 [P]; the prompt pass, 2.9 ->
  ~1.8 s [P]; eviction, bounded by OPT [P].

**What is sure and what is a bet.** Sure: the shipped rows, and the ungated rows of 11.1 as speed. Every step past a
wall is a bet: no wall in 11.2 has been broken in a run, and on chat two must fall together for 150 tok/s. Effort so
far: 315 ledger lines on the fixed cost, 471 on SSD bytes, 41 on prefill.

**Not measured at all.** Concurrent requests (Splash reports 171 tok/s aggregate with 4 concurrent requests on this
machine class, self-reported), answers longer than 256 tokens, and prompts longer than 7.9K tokens.
