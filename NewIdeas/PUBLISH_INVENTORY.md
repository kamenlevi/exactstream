# Publish inventory (DRAFT, NewIdeas, Fri 6 Nov 2026 20:4x; final by Sat ~12:00)

NOTE Mon 07:55 (NUMBER CHECK): the speed cells in this file's older tables quote the run reports' aggregates. Those are
mostly geometric means over prompts, or the steady metric (tpsw). STYLE section 3 and the papers use pooled whole decode.
Examples:
- S5 MBPP: 118.3 here is geo; pooled it is 111.1.
- The wired-limit gain: +27.4 % here is geo; pooled it is +25.1 %.
- E24 / E24m: 41.3 / 38.4 here are geo; pooled they are 41.1 / 37.4.
Quote the papers or ~/NewIdeas/NUMBER_CHECK.md, never these cells.

PREPARATION ONLY. Nothing gets published until Kamen confirms the date, scope and venue; the orchestrator has asked him.
The deadline (a replacement model is expected within ~1-2 weeks) outranks the 250-330 / 90-130 trigger: the release
carries the best measured, quality-gated numbers at the freeze. Tags: [K] measured, [P] priced or modelled.

## 1. What is ours (from ~/TRIED/NOVELTY.md; one line each, with evidence, tag, owner, and whether it carries over)

Compiled Fri 20:5x from the 29 NOVELTY rows, each checked against its latest TRIED line (Tn = line n of TRIED.md).
Carries: MD = any MoE + drafter, SD = any spec-decode LLM, MAC = Apple SSD / Mac, AGPU = MoE on Apple's GPU. No row is
specific to this model's GDN / HC layers, so all of it ports to a successor MoE.

LIVE in a release config:

| item | tag | where it is live | owner | evidence | carries |
|---|---|---|---|---|---|
| SSD wake penalty + keepalive 50 us | new finding (known mechanism) | 45 GB EXACT default (Item 5, T10, +12 % with the night's levers) | MoonShots -> BigMoE | BigMoE 0365ada | MAC |
| W1-PICKUP coherent spin handoff | adapted | 45 GB EXACT default (lever 1) | Leap -> BigMoE | 5565a56 | MAC / AGPU |
| PURGE-1 volatile expert tier 64/32 | adapted | 45 GB EXACT default | Leap -> BigMoE | 0365ada | MAC |
| M-29 aligned zero-copy reads + copy split | engineering | 45 GB store (native + p11 + pad) | MoonShots -> BigMoE | 055d827, 37b9f8b | MAC |
| chat switch (drafter by task) | adapted | 45 GB EXACT default | MTP Head | 0365ada | SD |
| QOM (sure rows skip cold ranks 2-9; at thr 0 every drafted row is sure) | adapted | 45 GB FAST stack (code sets) | NewIdeas + BigMoE | 5bb2743; r57_screen_fast.py:233-240 | MD |
| 3-bit kept experts on drafted rows (U3's t3, inside QOM at thr 0) | new signal (published mechanism) | 45 GB FAST stack | NewIdeas + Breakthrough | T103, T106; r57_screen_fast.py:233-240 | MD |
| Q1-UNION (sure rows keep the check's union) | adapted (OEA) | RELAX B, code + math answers, and inside QOM (q1u) on the code sets | NewIdeas | T993, T985 | MD |
| comment rule (relaxed acceptance only in comments) | adapted (relaxed acceptance is published; the comment-only span was found nowhere in 2 web checks) | RELAX B | MoonShots | T1350, BigMoE 5b1d00f | SD |
| B12LL cold ranks half a layer late | published (KTransformers' Expert Deferral) | 12 GB default | Breakthrough | T1057 | MD |
| 11-bit scale pack (from LOSSLESS45) | published | the p11 store | MoonShots | 055d827 | AGPU |

PASSED, not live (findings): NAX hybrid padded tiles (+9 / +23 %, published practice; Leap, n7_stack.txt); the
prompt footprint as the code path's COLD table (new signal, PASS on code, T312); the wake-ahead ping (adapted, T47 /
T98); the 12 GB two-pass certificate and P1-OEA (adapted; speed PASS, but the T0 same-scores is KILL, T1039); M-28 /
M-30 / M-29b SSD measurements (engineering: material for the SSD section).

KILLED (the negative-results section): P-frame MoE output reuse (the one fully new idea; union only x0.91), the
staged check, P2-PREFETCH, expert-major / co-compute (T746, T450), 3-anchor ANE drafting (T850), the M-31 head ping
(T202), LSB-on-miss (overtaken by U3), COPYGATE (BETWEEN; only its prompt-copy rule ships, inside RELAX B). Fri's
KILLs join them: never-wait in the engine, SPLIT-NAX, CD16 on MBPP quality.

To fix before the freeze (NewIdeas):
- Refresh NOVELTY.md (last edited 3 Nov; the comment rule is now adapted, keepalive is adopted) and tag the live
  kernels and drafts: FAST-NAX, DN-KS, D7BNAX_32 / QMM32, att_single, lazy commit, DRAFT17, HC8. The NAX6 matrix-unit
  form is the candidate "ours".
- QOM's drafter certainty barely acts as shipped: it runs at cf 0, and thresholds <= 0.3 are identical (T1011). Its
  live effect is "skip what would miss", and that cache-state dependence carries the agent blunders.
- Some TRIED lines lag their evidence: QOM still says IN FLIGHT (T467), and EvalSuite's Fri verdicts are not merged.

## 2. Results we can stand behind today, and what is missing

| line | mode | workload | tok/s | quality gate | status |
|---|---|---|---|---|---|
| 45 GB | EXACT (NAX6-N) | ref14, 13-14 prompts | 49.9 [K, quiet, C 192, NATIVE-FRI run 2] | certified exact reference (Part 1 + 2 PASS, Thu 5 Nov) | stands |
| 45 GB | FAST (FAST-NAX all-in) | 12 HumanEval | 113.9 [K, quiet, C 224, palindrome] | same-scores: tonight's run on the stack | pending tonight |
| 45 GB | FAST stack (+ DN-KS, lazy, D7BNAX_32, att_single; no CD16) | 12 HumanEval | 132.6 steady = 116.9 whole decode [K, quiet, C 224, palindrome, Sat 00:2x; whole decode from the orchestrator's 05:0x table] | same-scores (runner 49 gates stackb, below) | measured |
| 45 GB | FAST stackb (the stack + B's comment rule + g2 copy) at C 256 | 12 HumanEval | 144.9 [K, quiet, palindrome 145.5 / 144.3, lifetime peak 42.8 GB; +46 % vs qall 99.1]; 142.4 in the S2 A/B (~1.7 % run to run) | runner 49 at C 224 (EvalSuite bb7094d, 3581be6): same scores vs REF2-T FAIL at the bar (HumanEval+ 90.0 vs 95.0, -3 of 60, p 0.25); vs NAX6 PASS (90.0 vs 91.7); B top-up PASS (0.30 / 10K); long code PASS (0.18 / 10K); MBPP PASS (76 vs 79, p 0.45). S4 compares vs NAX6-N (the orchestrator's (b); Kamen may reverse) | code verdicts in; S4 running |
| 45 GB | stackb + S2 (SB 2, D7BNAX_32=2, DSUB / SVH) at C 256 | 12 HumanEval | 146.8 steady [K, quiet A/B vs 142.4: +3.1 %, peak 42.8 GB]; whole decode 119.5 -> 124.8 (+4.4 %) | S4's gate (QOM run-to-run tokens differ, so no token route) | pending S4 |
| 45 GB | stackb + S2 + REF2_KEEP_CACHES at C 256 | 12 HumanEval | 164.2 steady [K, quiet A/B vs 149.1: +10.1 %; cycle 66.9 ms; peak 43.1 GB]; whole decode 126.0 -> 135.3 (+7.4 %), full period 143.6 -> 157.3 (+9.6 %); in S1's B arm: 165.3 steady = 158.5 full period = 137.8 whole decode [K, re-derived] | KEEP: EXACT proof 3 / 3 (same bits); S4 (if NW leaves v1, the second S4 slot) | v1 if NW stays BETWEEN |
| 45 GB | v1 as written for S4: the row above + NW v3 (code sets only; NW's chat map FAILED, so math / chat run without it) | 12 HumanEval | 180.3 steady [K, BigMoE f9c9fff] = 165.4 full period = 142.1 whole decode [K, re-derived Sat 04:5x]; peak 43.9 GB | NW's code map PASS (0.71 / 0.86 %). **S4 PASS** vs NAX6-N (EvalSuite a1aac9b, 05:38), both orders: HumanEval+ 88.3 vs 91.7 (53 vs 55 of 60, p 0.5), math 86 vs 88 (43 vs 44 of 50, p 1.0), long code 0 blunders in 27,488 tokens. Information: vs REF2-T, HumanEval+ 88.3 vs 95.0 (-4 of 60) would FAIL the original line | NOT the release: S1b (counters off) = BETWEEN (+6.9 % full period, +5.8 % whole decode; 149.4 whole decode with NW), so NW v3 leaves v1 (Sat 05:47) |
| 45 GB | **S5 = THE HEADLINE: v1' + wired limit at C 256** (BigMoE d8eb2e4, Sat 14:25, quiet, CLEAN: 0 live outside modules; cert_env.txt written for CERT) | 12 HumanEval / MBPP12 | **175.9 / 118.3 whole decode** (passes 174.6 / 176.9 and 118.1 / 118.4); full period 170.5 / 115.4; vs qall 92.5 / 65.8 (+90.1 % / +79.7 %) [K, re-derived]; lifetime peak 42.9 GB | S4 PASS vs NAX6-N (v1'); exact-safe wired fix carries; CERT from Sun 01:30 on cert_env | the release number unless PLAN 220 lands more |
| 45 GB | **v1' + wired limit (into S5)** = v1' + R57_WIRED=1: one mx.set_wired_limit(55.66 GB) after model load, so the prefill no longer unwires the 32 GB pools (Leap's FCP fix of NewIdeas' FIRSTCHK find) | 12 HumanEval | **181.2 whole decode** (+27.4 % vs 142.2), 176.5 full period (+4.3 %) [K, BigMoE S6W 2525931, quiet palindrome, re-derived]; check 0 320 -> 65 ms; lifetime peak 42.9 GB | exact-safe (verify-logits hash 43 / 43, tokens 3 / 3 in the exact env), so v1's S4 PASS carries over | S5 tonight measures the release number |
| 45 GB | **v1' (the release candidate)** = stackb + S2 + KEEP at C 256 (final_env dd7ad0f, snapshot 03c7782) | 12 HumanEval | **141.2 whole decode**, 164.2 full period [K, S1b's B arm, quiet, Sat 05:4x]; lifetime peak 43.8 GB | **S4 PASS** (EvalSuite 013a4a6, 06:48) vs NAX6-N: forward HumanEval+ 88.3 vs 91.7 (53 vs 55 of 60, p 0.5), math 84 vs 88 (42 vs 44 of 50, p 0.5); reverse HumanEval+ 90.0 (54), math 86 (43); long code 0 blunders in 27,014 tokens. Every discordant pair goes against the arm (small, consistent losses; none significant). vs REF2-T: FAIL (information) | gated; S5 (the final number) may go; FREEZE ~20:00; CERT from Sun 01:30 |
| 45 GB | FAST + RELAX B (code and math answers) | HumanEval+ / math | x1.49 / x1.32 vs the exact control [K, paired, C 160] | decision 4 PASS; drift PASS as registered | stands; tonight's top-up can roll it back |
| 45 GB | chat | IFEval | E x1.46 [K, paired] | E PASS alone; E + HC8 KILL (84 vs 90) | ships EXACT until E runs without HC8 |
| 45 GB | agent turns | 17 long agent turns | - | AGENTFAST FAIL (85 blunders; QOM-KEEP carries it) | ships EXACT (stated limit) |
| 45 GB | FAST | 12 MBPP | 71.3 [K, qall, C 224] | none on MBPP yet | coverage check tonight |
| 12 GB | VERIFIED GREEDY (not exact: 2-4-row verify calls round differently from one-token decoding) | 16 HumanEval | 18.42 decode geo mean [K, 12GB-CLEAN Sat 00:28, alone, quiet; pooled 18.47; 13.66 with prefill; lifetime peak 11.144 GB] | deterministic (outputs = kp2's 16 / 16); not bit-exact; the ~12x comparison withdrawn | stands as verified greedy, with that label |
| 12 GB | FAST (top-2, theta 0.95) | HumanEval | ~36 [P, modelled] | map code PASS 1.22 %; the 12 GB FAST arm's T0 same-scores was KILL on 5 Nov (HumanEval+ 80.0 vs 91.7, IFEval 84 vs 90; T1039) | needs a measured run + a new same-scores (map STEP0-P1C queued Sat night) |
| 12 GB | FAST t80ck1u (P1-K1, θ 0.8) and t90ck1u (θ 0.9) | 16 HumanEval / HumanEval+ 60 | 46.33 whole decode [K, Fit12, 11.245 GB]; t90ck1u ~41 [rough, shared] | EvalSuite 84b3715 (Sat 12:50): FAIL both. HumanEval+ 50 vs 57 of 60 vs REF2-T (0 / 7, p 0.016); θ 0.9 52 vs 57 (p 0.063). The map's code line had PASSED | not releasable: no 12 GB FAST code mode |
| 12 GB | FAST (theta 0.8, t80ck2) | HumanEval | 38.7-40.3 [K] | map code KILL 1.63 % | not releasable |

How tok/s is measured: a geo mean over prompts. The screen's 'steady' (tpsw = kept / (c2 - c0), checks >= 2) leaves
out the host work between cycles (~2.35 ms per check, so ~4 % high) and any wait placed there (NW's counter readback,
~4.75 ms). The release reports two figures:
- Headline: whole decode (all time after prefill; prefill reported separately). The orchestrator's rule, 05:0x.
- The steady-state figure: full-period steady (all wall time after the first two checks).
'steady' rows above are labeled as such. The same-metric table for every arm is in findings note
2026-11-07_0500. On it, v1 = 142.1 whole decode (137.5 without NW v3), and 200 is x1.41 away.

Stated limits in any release: agent turns run EXACT; every code number names its workload (HumanEval is flattered by
prompt copying); FAST outputs can differ between runs at near-ties (BigMoE, Fri), so FAST quality is certified under 2
prompt orders.
- CORRECTED Sun 05:2x (`date`), from CERT (EvalSuite RESULT_cert 2cfed64, re-derived by NewIdeas):
  - Chat in the shipped 45 GB config is NOT exact. Only 1 of 12 IFEval rows is token-identical to NAX6's AR.
  - Its quality is certified as the same scores (IFEval PASS in both orders).
  - Every set runs FAST kernels in that config: token-identical rows to NAX6-N are HumanEval+ 62 / 56 of 164, math
    17 / 14 of 100, MBPP 34 / 32 of 100, IFEval 11 / 11 of 100.
  - The old line "chat runs EXACT" is withdrawn.
  - An exact chat mode is the E5 env (exact stack), at the exact chat speed below.

### CHAT per memory line (Kamen: the publish table leads with chat; 12 IFEval prompts, greedy, cap 256, whole decode)
| line | mode | chat tok/s (whole decode) | peak | source | quality |
|---|---|---|---|---|---|
| 45 GB | EXACT (E5 / N45 exact env) | 37.1 pooled, 2 passes (N45; its report's 39.9 is a geo mean); 38.3 (CHS4 yes1, E + SB1) [K] | 41.7 GB | BigMoE N45 / CHS4 runs.json, NewIdeas NUMBER_CHECK 0a-2 | exact (24 / 24 == AR) |
| 45 GB | shipped lossy config (L) | 43.4 pooled (N45; report geo 45.0); 44.2 (CHS4 y0 + y0b) [K] | 42.4 GB | BigMoE N45 / CHS4 runs.json, NUMBER_CHECK 0a-2 | CERT: lossy, same scores (HE+, math, MBPP, IFEval). Multilingual: never gated. The MGSM gate is HELD for Kamen: its prompts are typed math, and the card lists no natural-language multilingual benchmark (known benchmarks only, Kamen 09:3x) |
| 45 GB | L + M1-B (QOM on the MTP head's chat drafts, bonus gated) | 49.0 [K, one pass, mean of 2]; chs4 again 49.3 / 50.5 | 42.4 GB | BigMoE CHS2 87156ea + CHS4, re-derived | IFEval same scores PASS both orders vs NAX6-N (-1.00 / -4.00 pt; EvalSuite RESULT_chat_m1b 70b5cf8). Multilingual: gate HELD for Kamen (MGSM is typed math, so QOM never runs on it; the options are English-only with a language switch, or MGSM as chat). The multi blunder line, 334 per 10K, is information |
| 45 GB | L + M1-B + R57_SB1 (1-row checks overlap their reads; schedule only) | 52.0 [K, one pass; +4.2 % vs M1-B's mean] | 42.3 GB | BigMoE CHS4 ybs1, re-derived (1-row checks 35.4 vs 48.0 / 45.3 ms) | SB1 is BIT-EXACT: E vs E + SB1 logits hashes identical on 676 / 676 checks (CHS5). CHS5 repeat on identical tokens: 51.89 vs 49.67 / 49.94 (+4.2 %), so it shares M1-B's gate. NOT adopted (BigMoE / Kamen) |
| 45 GB | L + M1-B + COH + SB1 + FAST bundle / HC8_G + rank cut 8 + R1 (best measured stack) | 56.05 [K, one pass] | 42.4 GB | BigMoE CHS5 zsgcr, re-derived | LOSSY, UNGATED (FAST kernels + the R1 rule): needs its own IFEval + MGSM gate (Kamen's rule); HELD behind M1-B's MGSM gate (it carries M1-B's QOM) |
| ~32 GB | EXACT (E at C 160) | 33.8 [K, one pass; 12 / 12 == NAX6-N AR] | 30.6 GB | BigMoE c32 (Sun 16:2x, data/chs/c32_report.txt), re-derived from runs.json | exact |
| ~32 GB | L + M1-B + COH + SB1 (zs at C 160) | 51.3 [K, one pass] | 31.2 GB | BigMoE c32, re-derived | LOSSY, UNGATED at 32 GB: QOM skips cold ranks, and C 160 has more cold experts than the gated C 256 |
| 24 GB | - | not measured (Breakthrough priced bytes only: EXACT 210 / 183 MB per kept token) | - | Breakthrough 2bb8413 | - |
| 12 GB | VERIFIED GREEDY (ex, B12 + MTP) | 12.88 geo mean (pooled 12.76) [K, alone, quiet] | 10.96 GB | Fit12 chat12 (Sun 05:07-05:12), re-derived from chat12.txt | verified greedy |

- STRATA HEAD-TO-HEAD, DONE (the orchestrator's runs: 8 / 8 arms Sun 14:54-16:07 at the default arena, B and D again
  16:12-16:25 at a 33 GB arena; ~/StrataEval/h2h/results{,_arena33}/REPORT.md; every number re-derived by NewIdeas
  from the request rows) [K]:
  - Strata 0.1.58-metal on the same 12 IFEval prompts, greedy, cap 256, experts streamed from the SSD. It runs
    ISTA-DASLab IQ3_XXS (~3 bits), not the 4-bit weights, so its tokens differ from NAX6-N's (1-2 of 12 rows equal).
    Its 'exact' arms (C, D: STRATA_HC_Q8=0) are exact only to its own 3-bit weights.
  - EQUAL MEMORY, by peak footprint (pooled IFEval decode, tok/s):

| peak footprint | ours (config) | Strata (arm) | ratio |
|---|---|---|---|
| ~41-42 GB | lossy 51.9 (zs) / 53.8 (zsgcb), 42.4 GB; exact 38.3 (CHS4 yes1; N45 37.1 pooled) | lossy 41.8 (B, 33 GB arena, 40.8 GB); exact 40.6 (D, 41.2 GB) | lossy x1.24-1.29; exact 0.91-0.94x |
| ~31-35 GB | lossy 51.3 (zs at C 160, 31.2 GB, UNGATED at 32 GB); exact 33.8 (E at C 160, 30.6 GB, 12 / 12 == AR) | lossy 37.4 / 37.1 (B1 / B2, default arena, 33.5-33.6 GB); exact 36.1 / 35.6 (D1 / D2, 33.5-34.0 GB) | lossy x1.37; exact 0.94x |

  - Strata without its MTP head: 28.0 / 25.6 (A1 / A2), exact 25.5 / 25.3 (C1 / C2) at the default arena.
  - TTFT is where Strata wins: median 0.83-1.06 s vs ours 2.9-3.1 s on the same prompts. End to end (TTFT included)
    ours is 24.8 at 45 GB (zs) vs Strata's 29.7 (B, 33 GB arena). CT1 below is the reason and PF-D the fix.
- CHAT TTFT (time to the first token; the whole-decode chat column leaves it out), 12 IFEval prompts of 37-101
  tokens [K]:
  - 45 GB: 3.05 s mean (2.51-3.65), the same in exact and lossy configs (BigMoE CHS5 zs / yeh / zsgcr runs.json).
  - 12 GB: 2.73 s mean prefill (2.07-3.25), reading 15-26 GB from the SSD per prompt (Fit12 chat12).
  - So 45 GB's TTFT is 1.12x 12 GB's: its pools don't help the chat prefill yet.
  - CT1 MEASURED (BigMoE PFB 16:26-16:35, zs + R57_PFSTAT=1, acf8ac3; judged by NewIdeas' pre-registered script
    scripts/chat/ct1_judge.py, BigMoE's numbers identical): the 45 GB chat prefill reads 244.9 GB over the 12 prompts,
    100.9 % of the 12 GB line's 242.7 GB (20.4 GB per prompt = routed expert-layers x 2.79 MB: every routed record is
    re-read). 65.5 % of the routed base expert-layers (25.1 % of the sides) were already resident. Mean base
    residency at start 45.7 % (37-51 %). Verdict BETWEEN (pass line 50 %).
  - PF-D for chat (serve resident experts in the prefill, exact) prices TTFT 3.1 -> ~1.6 s [P] (TTFT = 0.32 s + SSD
    GB / 7.33, fitted by BigMoE; bytes x (1 - 0.75 x 0.655 - 0.25 x 0.251) = x0.45). End to end at 45 GB 24.8 ->
    ~33 tok/s [P]. Strata's ~0.9 s needs ~4 GB per prompt.
  - At ~52 tok/s and ~145-token answers, TTFT is about half of each chat answer's wall time.
- 12 GB PREFILL [K] (same run, s_moe 4096, today's): 1.6K 190.6 tok/s (TTFT 8.5 s), 4.7K 377.7 (13.0 s), 7.9K 461.3
  (17.5 s). Each prompt streams ~63-67 GB from the SSD. s_moe 8192 (one MoE call per prompt) is x0.99-1.01: no gain.
  Tokens are equal. CORRECTED Sun 08:0x: only SHORT prompts are SSD-bound (~7.5 GB/s at 1.6K). At 7.9K, ~67 GB in
  17.5 s is ~3.8 GB/s, so long prompts look compute-bound [P].
- 45 GB PREFILL, EXACT [K] (BigMoE PFX 07:57, snapshot 94a6f02; re-derived by NewIdeas from runs.json / tokens.json):
  PF-A layer-major prefill (R57_PFLM=1).
  - 1.6K 194.9 tok/s (TTFT 8.03 s), 4.7K 367.5 (12.87 s), 7.9K 440.5 (18.01 s).
  - vs the 1024-chunk exact arm (102.8 / 126.9 / 128.2): x1.90 / x2.90 / x3.44.
  - Logits-hash lists and tokens are identical to that arm's on all 3 prompts.
  - SSD per prompt 64-68 GB (vs 109 / 265 / 430).
  - Peak 45.10 GB: 0.1 over the 45.0 line, so a trim is needed before it ships. Not adopted yet (BigMoE).
  - Same speed as the 12 GB line: both now read each expert once.
  - Bigger chunks (R57_PFCHUNK 8192, lossy) reach x2.46 at 4.7K but change tokens and peak at 53.1 GB: BETWEEN on the
    pre-registered line.

Missing before a release table is complete:
- 45 GB EXACT quiet tok/s per workload (HumanEval, MBPP, math, IFEval) on the certified kernel at the release C.
- 45 GB FAST stack: MBPP, math and IFEval speed; same-scores on MBPP; the chat launch (E without HC8) and its gate.
- 12 GB: the clean verified-greedy run (alone, all 16 prompts, own footprint: Sat morning); the theta 0.95 point
  measured and gated; MBPP, math and chat numbers. No 12 GB exact claim.
- Lifetime phys_footprint for every released config (the RAM line is the lifetime peak, not MLX's peak).
- Everything at 2 prompt orders (FAST depends on cache state).

## 3. Write-ups (who writes, how long [P, the owner's estimate to confirm])

| piece | content | owners | time |
|---|---|---|---|
| Paper A (core) | exact 16-row speculative decoding of a 125B MoE on a laptop: batch-invariant kernels on the GPU and matrix units, streamed experts, the 45 GB exact line (12 GB: verified greedy, not exact) | BigMoE (engine), Leap (kernels), EvalSuite (certification), Fit12 + Breakthrough (12 GB) | ~2-3 days |
| Paper B (FAST) | drafter-certainty expert gating (Q1, Q1-UNION, QOM, U3), the comment rule, how lossy modes are gated (decision map + same-scores), and the negative results (agent turns, CD16, never-wait, split) | NewIdeas + Breakthrough + EvalSuite | ~2 days |
| Note C | SSD on Apple silicon: the wake penalty, wake-ahead ping, zero-copy reads, queue mixing | MoonShots | ~1 day |
| Note D | M5 kernels for MoE decode: NAX tiles, DN-KS (split-K per shape), att_single, QMM32 | Leap | ~1 day |
| Note E | the 12 GB two-pass certificate (P1-OEA) | Fit12 + Breakthrough | ~1 day |
| Appendix | the TRIED ledger (every idea, KILLs included), NOVELTY, PRIOR_ART | NewIdeas | ~0.5 day |

The public findings notes (~/FableGod/findings/) are the changelog; the 12 GB note already carries a dated correction.

## 4. Freeze and nights (UPDATED Fri 21:3x: Kamen wants everything finished by Sunday and published by Tuesday at the latest)

- Fri night (window open from 23:30, re-timed Fri 23:3x): the stack palindrome (+ T16 / B's rule / C 256 arms) ->
  Leap's 5-min idle check -> Fit12's clean 12 GB run (or earlier, if it fits first) -> EvalSuite's gate on the winner
  + both long-code top-ups + MBPP -> the map (nw3 first) and CERT's reference decodes.
- Saturday, each slot as soon as its input is in: S0 async post (if the idle check shows >= ~8 ms per check), S1 NW v3
  (map arm, then the engine A/B), S2 small built items, S3 leftovers, S4 short gate (by 16:00 at the latest), S5 the
  final number in the evening.
- FREEZE Saturday ~20:00: the S5 config is the release config. Anything after that is post-release research.
- Saturday night (Sun ~01:30-10:00): CERT, the release certificate on the frozen config. HumanEval+ 164, math 100 and
  MBPP 100 in 2 orders; IFEval only if a chat FAST profile ships. Plus the final quiet speed numbers per workload
  (45 GB EXACT and FAST, 12 GB verified greedy) and lifetime footprints, where the window allows.
- Sunday: fill the paper from the owners' report files. Monday night is the fallback for anything CERT misses.
- Publish by Tuesday at the latest, if Kamen confirms date, scope and venue.
