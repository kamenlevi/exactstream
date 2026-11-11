# Night Wed 4 Nov: one quiet ticket at C 160 (00:30-03:07), WALL 1 sprint lead's report

Written Wed 03:14 from data/s160, data/sc, data/cmb, data/pw, data/r57 (reports: data/r57/stall_report.txt,
data/r57/s0_report.txt, data/r57/pr_160_preroute.txt, data/cmb/combo_report.txt, data/pw/pw_report.txt).
Memory held the chain until 00:30 (NO-SWAP 31 -> 35 GB). Every arm ran at C 160 (peak ~29-34 GB). One heat pause at
01:37-01:41 hit cmb5 (see the combination).

## Scoreboard row (W1B = S45's six prompts, CAP 256; traced c45t_160, untraced ka0_160)

| Night | SSD busy | Exposed | Calls/check | MB per kept token | Tok/s | Exact? |
|---|---|---|---|---|---|---|
| Wed 4 Nov, C 160 | 110.4 ms | 112.0 ms | 76.5 | 169.4 | 42.0 | yes (every token == REF2-T) |

**Best measured number (exact): 42.0 tok/s steady on W1B at C 160.** At C 160 the wall is BYTES, not latency:
1,190 MB per check at 12.9 GB/s = 92 ms of the 110 ms SSD busy; the per-burst fixed part is 18.1 ms per check
(76.5 bursts x 0.237 ms). That 18.1 ms is lever 1's ceiling at C 160 (11 % of the 160 ms check).

## Lines, one per test

- **Stale-slot check: PASS** (00:34). Each pair ran with and without a command-buffer boundary before every MoE call: the
  default, sidecar C_l 32 and zero-copy direct. In all three pairs, 3 prompts x 128 tokens and all 127 per-check logits
  hashes were equal. Every 45 GB number stands.
- **S1 JOINT (traced pair): KILL at C 160 by its line; NOT a build or flag problem** (rechecked Wed 04:1x on the
  orchestrator's question, scripts/stall_report.py now prints a "JOINT read" line).
  - The pre-registered line was: PASS if busy <= -8 ms AND bursts <= 50 AND tokens identical; KILL if busy > -3 ms.
    Measured: SSD busy (union) 110.4 -> 110.1 ms, stall 107.9 -> 109.9 ms, tokens exact.
  - JOINT does not merge the two pools' reads into one preadv by design. It starts the sidecar row's missing reads in
    the background while the base row is served.
  - The flag and build worked: JOINT issued 9,742 of 21,665 sidecar reads early (45 %; 0 % with JOINT off).
  - The "76.5 bursts in both arms" was my report's count: it counts base and sidecar apart, so it could never drop.
    Counting an overlapping base + sidecar once gives 76.5 -> 68.1 per check (27 % of the 31.5 layers with both kinds
    overlap).
  - Why there is no gain: without JOINT the sidecar reads already start 17 us after the base burst ends, so there is
    no serial gap to remove. With JOINT the whole layer read span stays the same (2.486 -> 2.489 ms median): the bytes
    bind at C 160.
  - Unpaired base rows also pay the 30 us pairing wait (issue delay +~20 us per layer).
- **S2 keepalive 50 us (A-B-A-B, untraced): about -3 ms per check on clean prompts.**
  - The mean demand SSD drop of -10.9 ms per check comes from two one-off spikes in the A arms (gsm8k in ka0r,
    HumanEval/68 in ka0).
  - On the 4 clean prompts it is -2..-4 ms, matching MoonShots' -3.4 ms and the stall trace's 2.6 ms.
  - Tok/s geo-mean +3.9 % (noise about 10 %).
  - Pre-registered: "busy -1 ms" is met. "d -0.04 ms" is unmeasured (untraced), so Wed's traced w1tk arm decides.
- **Volatile tier 64/32 (build_vt; 7.79 GB purgeable, outside phys_footprint): SSD MB per kept token 169 -> 118
  (-30 %) on every prompt (-34..-63 MB), exact.**
  - Tok/s only +3.1 % geo (+6.7 % without the HumanEval/68 spike), and demand SSD ms only -1.5 %.
  - So copying out of the volatile slots costs nearly what the SSD read did.
  - NEXT: zero-copy vtier (the GPU reads the volatile slot through the slot table; no memcpy). This is a BYTES10 factor
    of x1.43 on bytes at C 160.
- **Pre-route recall (lever 2): KILL WITHDRAWN (Wed 05:5x): it measured the wrong read; re-run = WED_TESTS 3b.**
  - MTP Head's catch (Wed 05:3x), confirmed in the code: full / p3 / p6 scored L+1's router on L+1's ATTENTION-side
    hyper-connection read. qwen4_exp has no layernorms; each block reads the streams twice, with its own hc_norm and read
    mix per side, and the router sees the MLP-side read.
  - The tell I missed: "full" carries strictly more than r57 (L's whole MoE output added) yet scored lower (0.565 vs 0.634).
  - The night's numbers (p3 0.565 / p6 0.566 per row, R-57 0.634) are of that attention-side read only. NewIdeas' L2-08
    measured the MLP-side pre-attention read at 0.735 code / 0.820 IFEval + multi on its teacher-forced path.
  - Fixed in scripts/r57_screen.py (CPU logic test PASS): "full" = L+1's MLP-side read at its start (PLE included), p3 / p6
    on the MLP side, and "fullA" keeps the old read as the control. scripts/pr2_run.sh re-runs pr_160's 18 prompts (~4 min,
    shared slot, no timing) and saves MTP Head's SAMELAYER-T training rows (R57_PR_DUMP).
- **DAY9 chat switch (chA vs chB, 20 chat prompts): +21.6 % geo-mean (median +22.2 %), tokens 3902 / 3902 ==
  chat23 REF2-T: exact PASS at C 160.** chA's agent/text23 was excluded (paused 179 s).
- **U3: crashed at start** (u3n: the LP1 patch imported tierglu before MoonShots' planes_engine was on sys.path; U3C had
  the same path). Fixed in a2584ff; re-run = WED_TESTS 5a. u3c, the control, ran: its 12 HumanEval agree with pr_160
  token for token (1489 / 1489).
- **The combination at C 160** (one pass per arm; every exact arm 11 / 11 prompts token-equal; chat / agent rest on few
  prompts, so +-5-8 % is noise):
  - Exact levers:
    - chat switch: chat +10.8 %;
    - chat at 8 rows: chat +7.2 % vs cmb0;
    - truncation 0.4: no gain;
    - A6 keep: ~0;
    - COPYGATE g2: chat +6.4 % but agent -21 % (agent MB per kept 236 -> 292). g2t (on cmb4) brings agent back to -9 %;
    - D7b code: math -8 %, agent -16 %;
    - K-83: +0.5..+2.4 %.
  - cmb5 (JOINT) reads +5-6 % in every domain. But a 4-min heat pause froze it 7 s after its start, so its prompts ran on a
    cooled machine. Read it as a cool-down artifact: the traced pair above shows no JOINT gain.
  - 32-row checks: tokens differ on 3 / 11 (outside REF2-T's certified 1-16-row verify, as pre-declared) and code -62 %:
    KILL.
  - cmbsc (sidecar x1.5): not run, because there is no data/seed45/laysz45_160_sc15.json (only 224 / 256).
  - FAST (tokens differ by design; quality goes to EvalSuite's RELAX):
    - fastBEn code 92.0 tok/s (+94 % vs cmb0);
    - fastBEh +89 / +14 / +11 / +17 % (code / math / chat / agent);
    - code SSD MB per kept 173 -> 64-71.
- **Mission 20 PREFILL_WARM at C 160: misses -16 % in the first three checks, but slower end to end.**
  - Misses: code -17.5 %, agent -9.7 %; tokens 24 / 24 equal.
  - The 600-900 ms warm-up is not overlapped, so pre-loop + 3 cycles take longer (e.g. HumanEval/68 1221 -> 1859 ms).
  - That is a KILL on time at C 160. Its line is on C 256, which did not run (memory).
- **S0 (to Leap): the zero-hit class is empty** (0 of 10,224 served base calls; hits 9480, no-miss 744).
  - With misses, the routing post is seen 1.03 ms (p90 1.79) after the previous release; without misses, 0.83 ms (p90 1.08).
  - No ms-scale late posts show in the engine at C 160.
  - The S0 stage events (kinds 5 / 6) need build_w1 (Wed).

## Next steps (Wed)

1. Lever 1 at 13:45 (WED_TESTS 2): at C 160 the per-burst fixed part is 18.1 ms per check, so its PASS line (fixed <= 10
   ms AND steady >= +5 %) needs more than half of it removed. Keepalive is about -3 of it; zero-copy into aligned
   slots and the coherent pickup must give the rest.
2. Zero-copy vtier: spec + build (the -30 % bytes become time only without the copy); for NewIdeas' BYTES10 ledger.
3. The A/B night's reference: data/ab/ref_tokens.json does not exist yet (ab_records3.sh STOPs without it). REF2-T tokens
   of the 10 extra HumanEval at CAP 256, merged with chat23's 30 (shared lock, ~12 min).
4. U3 re-run (5a), then U3C (5b).
