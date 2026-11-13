# Wednesday 4 Nov: BigMoE's test list for Kamen's research day (13:45-23:30)

BigMoE, lead of the WALL 1 SPRINT, Tue 3 Nov ~22:3x. Everything below is built and unit-tested tonight; none of it needs
code written during the day. Exact = every token equal to the stored REF2-T tokens. Times include model loads.

## Already running tonight (results ready by ~05:00, read in the morning)

One quiet ticket at C 160 (data/s160/night.log):
- the wall-1 sprint:
  - JOINT on / off, traced (the first scoreboard row: SSD busy, exposed SSD, calls per check);
  - keepalive A-B-A-B;
  - the volatile expert cache smoke;
  - the pre-route capture (lever 2's recall; INVALID, it read the attention side: re-run = 3b);
  - the chat-switch checks;
- U3: CRASHED at start 01:02 (u3n rc 1: the LP1 patch imported tierglu before the TIER_GLU loader put MoonShots'
  planes_engine on sys.path; u3c, the control, ran). Fixed in scripts/r57_screen.py (U3C had the same bug); re-run = 5a;
- the combination's exact levers at C 160 (each lever's own delta, tokens checked);
- the prefill-warm mission.

Morning reports:
- stall_report.py and s0_report.py on c45t_160 / c45tj_160 (S0's zero-hit split: is the late routing post real?);
- pr_report, combo_report, the U3 and vtier lines.
- DONE Wed 03:14: docs/NIGHT_1104.md (scoreboard row, every line, next steps).
- JOINT (S1) rechecked Wed 04:1x (orchestrator): KILL at C 160 by its line, not a build or flag problem. JOINT issued
  45 % of sidecar reads early. Overlap-merged bursts went 76.5 -> 68.1 per check, the layer read span was unchanged
  (2.486 -> 2.489 ms), and without JOINT the sidecar already starts 17 us after the base. stall_report.py now prints a
  'JOINT read' line; in item 2 keep JOINT in the stack only if w1tj / w1tp move the layer span (POST_SYS may raise the
  pairing).

## Wednesday morning, before 13:40 (CPU / builds, no GPU)

- The stale-slot check ran FIRST in tonight's chain: PASS 00:34 (data/sc/sc.log; all three pairs, tokens + 127 logits
  hashes equal with vs without the command-buffer boundary). No item 0b.
- The conversion format becomes "native+p11+pad" (agreed with Leap 22:4x on MoonShots' M-29):
  - MSB s3 / b3 padded to 36,864 B; LSB l1 planes first, b packed; msb_sz / lsb_sz / recsz unchanged.
  - Leap: kernel stride from the pool shape + the tool's layout.
  - BigMoE:
    - r3layout_ref.py "native+p11+pad" + regenerate the scratch copies for Leap's cmp;
    - tier.py builds the sidecar pool pieces in record order;
    - the direct path for aligned pieces with the b tail copied;
    - test_layout / test_layout_engine on the new copies.
- U3C is built (0ea5b8b). Its first run also writes the mask dump for Breakthrough (R57_U3C_DUMP, 3 prompts).
- **5b NARROWED (Breakthrough Wed 07:26, FLOOR's 42 held-out windows, SCOREBOARD 45fb226): the per-request table is
  CODE+, not GENERAL.** pmtc (the form built in 0ea5b8b) vs the u8+t3 live control: multi 6.70 vs 5.66 % (z +2.78, a
  cost); IFEval z +0.92 and code z -1.88 show none. len (warm = the prompt's footprint alone, only when the prompt has
  >= 128 tokens) holds on code: 0.81 vs 1.11 %, at cached bytes x0.196 vs x0.272 (-28 %) [P, 45 GB plain-LRU replay].
  So 5b = U3C in len's form on code requests only; chat keeps plain U3. No engine change is needed for the test:
  R57_FAST=U3C + R57_U3C_LEN=1 exist (0ea5b8b) and 5b runs code prompts only (scripts/u3c_wed.sh, e947f23; Breakthrough
  07:3x: "the line is fine as written"). Breakthrough's mask check (dm_u3c_replay.py) now also checks warm against the
  length rule: prompts >= 128 tokens must hold the footprint alone, < 128 the footprint + cap45 table; the dump's
  prompt_len is len(ids), the same count the rule uses in run(). DEPLOYMENT CAVEAT (Breakthrough 07:3x): code-only
  needs a request-level switch in the engine (the screen knows the domain from the prompt key, a server would not);
  until one exists, U3C is an opt-in mode for code requests, which is what 5b measures.

## The day's tests, in order (GPU: one quiet lock per block)

**PULLED FORWARD TO THE MORNING (the orchestrator Wed 06:2x: the GPU is BigMoE's from ~08:30, after FLOOR and
Breakthrough's map batch, to 12:50; Kamen wants the tests done and prepared before 13:45).** scripts/morning_1104.sh,
started on Breakthrough's "map batch done" (log data/morning_1104.log):
- A (quiet, C 160): 3b scripts/pr2_run.sh, then 5a scripts/u3_smoke.sh. 5a's fresh u3c re-checks item 3's 10 HumanEval at
  full length.
- 0 (~09:00, CPU): scripts/item0_live.sh. Fit12 06:2x: YES, its 13:05 runs load neither ~/BigMoE/bigmoe nor
  ~/BigMoE/ext/build; they need the .venv and models/bare unchanged until ~13:55, and nothing here touches either.
- B (quiet): 1 scripts/w1b_24.sh (build_m01 rebuilt 06:23 from the current source), then 2 scripts/w1_lever1.sh (C 224 if
  NO-SWAP >= 44).
- C (quiet, needs 0): 4 scripts/mix_e2e.sh, then 5 scripts/default45_wed.sh.
- Every runner stops starting arms at 12:30 (the orchestrator 07:2x: every arm must END by ~12:55; Fit12's T3R2 goes quiet
  at 13:05, hard stop 13:40, no slack).
- STARTED 07:28 (Breakthrough's "map batch done" 07:26; FLOOR night 1 ended 07:06), so item 0 runs at ~07:4x, after A.
- **3b DONE 07:31 (151 s, lifetime peak 29.4 GB): BETWEEN.** Per-row recall@10: p3 0.789, p6 0.792; full 0.794, R-57
  0.634, fullA 0.565 (== pr_160's full, as required). All 18 prompts' tokens == pr_160's. The partial reads lose almost
  nothing against the full read at L+1's start; the gap to 1 is L+1's own attention write. MTP Head's SAMELAYER-T rows:
  data/r57/pr_dump/pr2_160_00..17.npz (2.4 GB; X = L+1's MLP-side read at its start, Y = its true MoE input; sent 07:3x).
  Report: data/r57/pr2_160_preroute.txt.
- **5a PASS (07:40):** u3n == u3c on all 12 prompts (the copied OOO path is exact); u3 ran (93 s). One 282 s heat pause
  inside u3n (fanwatch at the 6000 cap) put one prompt at 0.8 tok/s: the item 5 / 5b reports now leave paused runs untimed.
  Item 3's second check (fresh u3c_160 vs data/ab/ref_tokens.json, 12 keys, full length): ALL EQUAL.
- **Item 0 LIVE 07:51 (d0030a1):** the 07:41 try tested on build_nat, which lacks set_direct_split (the patched tier.py's
  native+p11+pad path needs it) -> rc 3, files restored; item0_live.sh now tests on build_w1. ext/build sha256
  5333899cb514 (byte-identical to build_w1 and build_m01), Leap 7f43a3f; EvalSuite told.
- **Item 1 (07:56): W1B 24 GB = 33.2 tok/s mean, 383.8 SSD MB per token, lifetime footprint 23.86 GB; w24 == w24t.** The
  07:42 try crashed at the 12-token warm-up: tier.py's block (01f652c, 11-01) took 9-16-row calls that the 24 GB stack's
  MAXK 80 sends to the sync prefill path (per-expert output under TIER_COMBINE). Fixed in da8e7a6 (such calls take the
  stock block, as before 11-01; nothing that worked changes). vs REF2-T: 3 of 6 equal (HumanEval 68 / 91, algebra 1282);
  prealgebra 1640, algebra 694, gsm8k 708 differ, not because of the fix (their prefills are single >= 50-row calls):
  expected: REF2-T is the 45 GB reference only; the 24 GB line keeps its own stock-kernel reference (w24 == w24t PASS).
- **Item 2 (07:50): lever 1 BETWEEN by its line.** All 8 arms exact (6 / 6 == REF2-T each); steady tok/s (geo) w1u 60.3 ->
  w1uf 65.7 (+9.0 %, C 224); but the fixed part per check 19.2 -> 19.5 ms (line <= 10; data/w1/stall_w1t_vs_w1tp.txt):
  the stall falls 4.1 ms per check, not the issue / first-byte / wake cost the lever targets. JOINT's step line: KILL
  (busy -2.4 ms, needs <= -8). Traced steps (cumulative, single pass): JOINT +1.6 %, keepalive +2.7 %, copy split
  +0.5 %, coherent spin +1.9 %, system post -0.6 %.
- **Item 4 (08:05): PASS.** mix (layers 0 / 20 / 47 converted, native+p11+pad) == mixc (the live package on the new
  ext/build) on all 11 prompts, and both == REF2-T on every token (vs ref14, data/ab/ref_tokens.json and 5a's u3c_160).
  The script printed "every token equal False" only because its reference fell back to chat23, which has no
  HumanEval/16 or /6 ("REFTOK 0 / 1"); mix_e2e.sh now uses ref14. So the converted path is exact end to end, and the
  rebuilt ext/build + the layout patch + the tier fix keep the 45 GB route exact.
- **Item 5 (08:05-08:32, C 160: NO-SWAP 48 < 52): ADOPT + lever 1 ADD -> ADOPTED as R57_PRESET=45 (0365ada, 08:35).**
  A-B-C-C-B-A, all six arms 14 / 14 == REF2-T, no heat pauses (data/d45/d45_report.txt). dfb vs dfa (the night's levers:
  chat switch, keepalive 50 us, volatile tier 64/32, K-83): steady +12.1 % geo (chat +17.3, code +12.4, math +12.8,
  agent +0.5; worst prompt agent/bash18 -4.5 %, best multi/zh15 +29.4 %); SSD MB per kept token 172.5 -> 120.2;
  lifetime footprint 31.8 -> 33.6 GB. dfc vs dfb (lever 1's full stack: JOINT, copy split, coherent spin, system post):
  +6.4 % (code +11.7, math +5.4, chat +3.8, agent +3.2). Geo tok/s per arm: dfa 35.7 / dfa2 36.1, dfb 39.5 / dfb2 40.9,
  dfc 42.9 / dfc2 42.7: ~+19 % together. R57_D45=0 = the 11-02 default. NOT yet measured at C 256 (the 45 GB default's
  C, 41.7 GB lifetime on 11-02; +1.8 GB here at C 160 -> ~43.5 GB expected): first free quiet slot.
- **5b PASS (08:48-08:57, C 224, after Breakthrough's screen 4; on the new R57_PRESET=45):** U3C len vs U3 on 12 HumanEval,
  A-B-A-B: SSD MB per kept token x0.732 (geo; per prompt x0.56-0.91), steady tok/s +3.7 % (per prompt -5.5..+15.5 %),
  kept per check x0.977; no heat pauses. Absolute geo steady: U3 106.4 / 105.7 tok/s at 20.5 MB per kept; U3C 110.1 /
  110.0 at 15.0 MB; lifetime peak 39.4 GB. The u3n2 gate (U3's null control on the rebuilt ext/build + the new default)
  == 5a's u3c_160 (REF2-T) on all 12: 83.1 tok/s, an EXACT number for the new default on code at C 224 (the old default
  at C 160 this morning: 49.8). Dump for Breakthrough's mask check: data/u3c/dump_224/ (HumanEval 16 / 68 / 91; sent).
  U3C MASK CHECK PASS (Breakthrough 08:5x, dm_u3c_replay.py on dump_224: 1,248 records bitwise == the rule; warm follows
  the length rule: 16 = the < 128 fallback with the cap45 table, 68 / 91 footprint only) -> cleared for EvalSuite same-
  scores on code. FAST: those decide adoption; code-only needs a request-level switch before it ships.
- **The new default at C 224 (09:49-10:10, after the 09:15 C 256 try swapped +2.3 GB by day): BETWEEN.** All 4 arms
  14 / 14 == REF2-T. new vs old +7.2 % geo steady (51.0 vs 47.1 tok/s; lifetime 42.1 vs 40.3 GB); code +7.4, chat +12.6,
  math +11.1, agent -9.9 % (bash18 -16.6 %, 9.7K-token prompt). W1B's six: 64.9 vs 60.9. The default stays (pre-registered
  rule); per-lever agent arms + the C 256 headline go to tonight (data/d256/d256_report.txt).
- **Agent leave-one-out by day (13:49-14:02, C 224, in a test window): CUT by the 2 GB swap rule in arm 2, no verdict.**
  old (aold): 5 / 5 agent prompts, 28.7 tok/s geo steady (bash18 24.1, text23 36.0, bash20 27.8, edit03 45.1, write12
  17.8), 444 s per arm (not ~3 min), lifetime peak 40.4 GB, swap +0.32 GB. new (anew) finished bash18, text23 and bash20
  (bash20 24.6 vs 27.8 steady), then swap passed +2 GB from the run's start (10.01 -> 12.13 GB; NO-SWAP 51 GB at its
  start; lifetime peak 42.2 GB plus the 7.8 GB purgeable volatile tier): SWAP GUARD. The report now says INCOMPLETE when
  arms are missing (it printed INTERACTION) and prints swap per arm; the RAM levers run first (-vtier, -chat, -lever1,
  -K83, -keepalive). The full run goes to Thu ~06:40 with apps closed (scripts/night_thu.sh, ~95 min). data/aloo/.
- **K-83's drafter row fallback as the agent cause (MTP Head's DRAFT17 lead): does not fire here.** R57_ROWS is unset, so
  every block is 16 rows (aold's 308 agent checks max 16 rows; Leap's 05fafe7 changes nothing at 16). Drafter ms per
  check FELL under the new default (agent 12.5 -> 11.4, code / math 11.6 -> 10.3; data/r57/d256*_224_checks.json). The
  agent cost sits in the check's SSD service: bash18 per check misses 246 -> 251, SSD MB 417 -> 341-362, I/O 56 -> 63-71
  ms, check 110 -> 118-124 ms (HumanEval/68 the same shape: I/O 127 -> 134-142 ms). Fewer bytes, slower service per miss:
  the volatile tier / keepalive / lever 1's I/O path, which the Thu leave-one-out tests first.
- **Next 45 GB slot after Thu's (pre-registered Wed 14:2x-15:1x, ~32 min quiet, MEM ~44 GB, all at C 224, 14 prompts):**
  plain / DENSE-N / HC8 / HC8 / DENSE-N / plain. DENSE-N = Leap's ref2.dense_native_convert(model) (2190e10: 4-bit dense
  weights with N >= 1024 replaced in place by the native layout, one copy, exact by construction): ADOPT into the exact
  default if >= +1 % geo steady AND every token == REF2-T. HC8 = Leap's hc8r.hc8_install(model) + REF2_HC8_FAST=1
  (c805fd0: 8-bit trunk hyper-connection mixers read in one pass, -0.6 GB; FAST, tokens differ): KEEP for the FAST track
  if >= +2 % geo steady with no domain < -3 %, then EvalSuite same-scores decides; else KILL.
- **Lever 2, stage 1 BUILT (Wed 14:5x-15:0x, e8b7b6d; BUILD_DAY #2): forecast reads into the volatile tier.** Today's
  default posts L+1's forecast (PF_B 96, ranked list <= 61) but issued no reads (no spare or victim slots; R-57's victim
  prefetch was KILLED: wrong guesses evicted needed experts). Now, right after L's release, the loader reads the first
  F = 4 ids of that list that are neither resident nor tiered (FS = 2 for the sidecar) as whole records INTO THE PURGEABLE
  VOLATILE TIER (one GCD block each, trace tag 3): the GPU never reads the tier, so a wrong guess evicts nothing; a demand
  miss found there is copied back (waiting for a read in flight). CPU test scripts/test_fc_tier.py PASS on records3 and on
  the converted layer, direct reads on and off (32 loads, every piece == plain demand loads; in-flight, pressure, skip and
  purge cases); test_layout_engine.py PASS on ext/build_l2. Slot (scripts/l2_wed.sh, C 160 by day, ~25-30 min): smoke,
  then l2c / l2f / l2f2 / l2c2 (14 prompts) + traced l2ct / l2ft (W1B's six). Line (pre-registered ~15:0x, before any run):
  KEEP if every token == REF2-T AND stall per check <= -20 % AND >= +3 % geo steady; KILL if < +1 % or the next-demand
  penalty > 0.05 ms per burst; else BETWEEN. Stage 2 = SAMELAYER-T's trained forecast (weights asked of MTP Head).
> (a quote from the project owner, kept private)
  r57_screen switches R57_HC8 (Leap's hc8r.hc8_install + ref2 hc8_fast, v4), R57_DNR (ref2.dense_native_convert) and
  R57_HCB (ref2 hc_b), each off unless set. scripts/allin_wed.sh at C 224 on the 12 HumanEval, all arms on ext/build_l2:
  u3c (U3C len) / allin (+ D7b code drafter + HC8 + DENSE-N-RAM + hc_b) / allinf (allin + lever 2) / allin2 / u3c2, then a
  dump arm for one map screen. No pass line: the combined FAST number today's builds reach, and the cycle's parts per check
  (check, draft, SSD I/O, gate service, build) for the list toward ~50 ms per check. QOM at 45 GB is not in (not cheap to
  wire). Runs behind lever 2's slot when memory allows (Chrome holds 13 GB by day), else Thu ~13:10.
- **QOM IN THE ENGINE (FAST, Wed 15:0x-15:5x; 5bb2743 + c387c97, ext/build_qom; R57_FAST=QOM):** Breakthrough's keep-set
  rule (~/Breakthrough/keepset 6f1ca44, read-only) on U3 at thr 0: a sure row's rank 2-9 expert outside the check's loads
  is posted with bit 21 (optional); the loader never reads it on a miss (slot -2), qom_keep_mx keeps the resident ones after
  the spin, a dropped use reads the row's rank-0 slot at weight 0. CPU test scripts/test_qom_ext.py PASS. DRAFT17
  (R57_D17=1, 723ee9b, MTP Head's draft17.py): a 17-position draft gates the last row too (smoke: 78 / 78 rows gated).
- **By-day FAST numbers (shared GPU, apps open, single pass: a lead, not a verdict).** C 96 / C 128 runs were INVALID: a
  16-row check (160 ids) takes the sync-free FAST path only when it fits the pool (C >= 160 per layer), so at C 128 every
  check ran ref2's host-synced _tier_slots (no OOO, no 3-bit sure rows, no QOM). At C 160 uniform pools (R57_LAYSZ=0,
  C_l 64; data/bdq/bdq_report.txt), 12 HumanEval: U3C 66.9 tok/s (33.8 SSD MB per kept, check 116.5 ms); QOM 67.3
  (24.8 MB per kept = x0.734; 747K optional uses, 153K misses never read; its drafter ms rose 11.9 -> 15.7: GPU
  contention); QOM ALL-IN (+ D7b code + HC8 v4 + DENSE-N-RAM + hc_b) 85.8 tok/s = +26.3 % vs U3C, check 100.2 ms.
  Lever 2 read nothing in the earlier FAST arm: only arm gate+t+pf posts L+1's forecast (and needs BIGMOE_PRED_LINES=4).
  Smoke 2 (data/smk): the exact default with lever 2 on, 2 prompts: every token == REF2-T (32 / 32 each), 2,538
  forecast reads, 1,192 used; the QOM all-in + lever 2 runs. Tonight 23:00 (if the apps close): QOM-based all-in at C 224
  (u3c / qall / qallf / qall2 / u3c2) then lever 2 at C 160.
- **5b after the morning block, if it ends before 12:00:** scripts/u3c_wed.sh (U3C len on code vs U3, A-B-A-B, plus the
  u3n2 gate on today's ext/build and the mask-check dump; line in its header).
- **Item 3 is DONE on CPU (06:22):** data/ab/ref_tokens.json = chat23's 30 + the night's u3c_160 (CAP 256, REF2-T) for the
  10 HumanEval; chat23 == u3c_160 on /68 and /91, and u3c_160 == pr_160 on the first 128 of all 10
  (data/ab/ref_tokens.README.txt).
- **Item 5 defined (06:5x):** see the row below.

| # | When | Test | Why | Time | Memory | Line |
|---|---|---|---|---|---|---|
| 0 | 13:40 (Breakthrough's layout check PASSED Wed 03:28: go), after Fit12's T3R2 and before 1 (nothing may import bigmoe/ while it changes) | **conversion: bigmoe/ patch live + ext/build rebuilt; ext/build_m01 rebuilt too** (CPU, 10 min; build_m01 for item 1:
b48's stale-ext guard refuses a .so older than the source), commit + .so sha to EvalSuite | Thursday's conversion; EvalSuite's ifx checks need the snapshot | 10 min | - | test_layout_engine.py on the live package PASS |
| 1 | 13:45 | **W1B 24 GB baseline** (scripts/w1b_24.sh): today's shipped exact 24 GB stack (s10b, C 128, M-01, JOINT) on W1B's six prompts (S10B_KEYS), untraced + traced | the sprint's 24 GB row; the 1.5x target needs it | ~15 min | <= 24 GB | measurement |
| 2 | right after 1 | **Lever 1 at 45 GB** (scripts/w1_lever1.sh; frozen copies r57_screen_w1.py / c45_runner_w1.sh made at arming) (C 224 if NO-SWAP >= 44, else C 160), traced, then the winner untraced: default / + JOINT / + JOINT + keepalive / + direct reads into the slots (BIGMOE_RECORDS_DIRECT, no bounce copy) / + coherent one-row spin (BIGMOE_SPIN_COHERENT) / + system-scope routing post (BIGMOE_POST_SYS), on ext/build_w1 | wall 1's per-call overhead (~20 ms of a 97 ms check) | ~50 min | C 224: ~44 GB / C 160: ~33 GB | exact AND fixed part <= 10 ms per check AND steady >= +5 % = PASS; < +1 % = KILL. Per step: JOINT busy <= -8 ms; keepalive d -0.04 ms; pickup: cadence a - burst a falls >= 0.03 ms |
| 3 | any shared slot before the A/B night (exactness only) | **the A/B night's reference**: REF2-T tokens of the 10 extra HumanEval (16, 6, 79, 19, 161, 17, 56, 141, 101, 63) at CAP 256, merged with chat23's 30 into data/ab/ref_tokens.json (found missing Wed 03:1x; ab_records3.sh STOPs without it) | the A/B night's every-token check | ~12 min | ~33 GB (C 160) | the 10 runs agree with tonight's pr_160 / u3c_160 tokens on their first 128 |
| 3b | with 3 (same shared slot), or any earlier gap the orchestrator gives (no map process running) | **the pre-route re-run with the fixed read** (scripts/pr2_run.sh; added Wed 05:5x): pr_160's 18 prompts, CAP 128, C 160, R57_PREROUTE=1 + R57_PR_DUMP=1. The night's KILL is withdrawn: its full / p3 / p6 were L+1's attention-side read (MTP Head's catch); "fullA" keeps that read as the control | lever 2's real recall; MTP Head's SAMELAYER-T rows (~2.5 GB in data/r57/pr_dump) with no separate capture | ~4 min | ~30 GB (C 160) | unchanged: PASS p3 >= 0.85 AND p6 >= 0.90 per row; KILL p3 < 0.75; else BETWEEN. Every token == pr_160's, and fullA within 0.01 of pr_160's full (0.565), else look first. MTP Head predicts all 0.76-0.81 (BETWEEN) |
| 4 | after 2 | **conversion end-to-end on scratch/model_mix** (layers 0 / 20 / 47 native+p11, the rest row-major; scripts/mix_e2e.sh at C 160) | the converted path (Leap's router + p11 kernels + my prefill reader) exact end to end, before any byte of records3 changes | ~20 min | ~33 GB | every token == stored REF2-T on 11 prompts = PASS; any difference = no conversion until found |
| 5 | after 4 | **the combined exact default** (scripts/default45_wed.sh, defined Wed 06:5x): A-B-A-B dfa / dfb / dfa2 / dfb2 on W1B + the combination's prompts (14), CAP 256, C 224 if NO-SWAP >= 52 else 160; dfb = the default + the chat switch + keepalive 50 us + vtier 64/32 + K-83 (+ lever 1's winner via D45_W1). Left out by the night: chat 8 rows, t 0.4, A6, COPYGATE, D7b, JOINT, 32 rows. ADOPT if every token == REF2-T AND geo-mean steady >= +3 % AND no domain < -3 %; KEEP the default if a token differs or < +1 %; the preset is edited by hand | the integrator's Thu 15:00 deliverable, one exact 45 GB default | ~25 min | as 2 | exact; the new default's steady tok/s vs 63.6 (C 256) or tonight's C 160 baseline |
| 5a | after 5, before 5b | **U3 smoke re-run** (scripts/u3_smoke.sh at C 160, fresh frozen copies; tonight's run crashed at start on the tierglu sys.path bug, fixed Wed 01:4x) | U3 and U3C (5b) need the copied OOO path exact first | ~6 min | as 2 | u3n == u3c token for token = PASS (then u3's FAST numbers count); any difference = U3 / U3C stop until found |
| 5b | after 5 | **Breakthrough's P1: the per-request COLD table on top of U3** (FAST; spec ~/Breakthrough/WED_PLAN.md P1, b800a4b): on drafter-unsure rows keep ranks 0-3 plus ranks 4-9 that are warm for this request (its prefill routed to them, or prior45 share >= 0.05) or in the check's union; sure rows = U3. BigMoE builds it as R57_FAST=U3C in r57_screen (Wed morning); Breakthrough checks the kept masks against its map (36 windows) from a routing dump | replay: -27 / -36 / -38 % SSD bytes vs U3 on code / IFEval / multi (halved -14..-25 %); quality = u8+t3's, so EvalSuite's task results decide | ~25 min (U3 vs U3C, R57 prompts) | as 2 | pre-registered (orchestrator / Breakthrough): chat SSD MB per kept token <= 0.85x U3's AND tok/s >= U3's = PASS; then EvalSuite same-scores |
| 6 | evening | **FAST preset** on the combined stack: EvalSuite's RELAX arms B + E / C + E, + NAX6 (+ hybrid) | the second track (quality judged by EvalSuite's same-scores tests) | ~30 min | as 2 | tok/s and kept per check; quality by EvalSuite after Thu's RELAX |
| 7 | when Leap asks | Leap's F1 (ref2 fatt_verify, 0002e2e): verify-call attention on fs4 | an exact-at-layer-level attention lever; EvalSuite certifies after RELAX | ~10 min | as 2 | Leap's line |

Notes:
- Order rule: 0 goes first, because 1-2 import bigmoe/ live and an A/B must not span two code versions. 1-2 then
  run on the patched engine, which is a no-op on today's store; their token checks confirm that. 1-2 need the GPU
  quiet. 4-6 need 0.
- If memory allows only C 160 by day (Kamen's apps open), every relative line still holds; the absolute 45 GB numbers
  wait for a night at C 224 / 256.
- Schedule (the orchestrator, Tue 22:3x): every batch test ends before Kamen arrives (FLOOR deadline 13:00, Fit12's T3R2
  13:05-13:40), and T0 moves out of his day. So the GPU is ours from 13:45: 0 (CPU, 13:40) -> 1 -> 2 -> 4 -> 5 -> 6, ~3 h of GPU in
  all (0 is CPU), leaving the rest of the day for what the results open up (lever 2's forecast reads, the next build).

## Thursday after RELAX_DONE + T0_DONE (unchanged)

EvalSuite's ifx check #1 (row-major, patched engine) -> heads-up -> scripts/convert_records3.sh -> EvalSuite's check #2.
Then Fri 00:30, the A/B (scripts/ab_records3.sh: B, B2, inverse, A, re-convert, B).

## Build queue after the day's tests (Kamen's decision 2 = YES, Tue 23:1x: a FAST mode may depend on the cache state)

QOM's engine path. Spec: NewIdeas QUEUE.md Tue 17:01. Sure rows (cf >= theta) keep ranks 0-1, union members, and rank
2-9 experts that are resident after the check's loads; would-be misses are dropped (weight 0). It needs:
- the loader: "optional" ids served only from committed residency, never loaded or taken from in-flight prefetches;
- the kernels: a drop sentinel = weight 0 (ask Leap);
- the harness: R57_FAST=QOM, default off, and a suite-format decode entry with per-row counters.
EvalSuite's four requirements (Tue 23:22) are met by that design. It certifies under >= 2 prompt orders at C 256 +
laysz45_256. The off path is checked first against REF2-T's stored rows (ifeval/3409, ifeval/2439, HumanEval/27).
Determinism (EvalSuite 23:3x, agreed):
- promotion only inside ensure(), never on read completion;
- look-ahead prefetch OFF in QOM mode;
- within a call: the non-optional loads in id order, then the optional ids judged against that state, the served ones
  touched in id order;
- self-test: the same key list twice in fresh processes, quiet vs beside a GPU + SSD load; the per-check served-optional
  set hashes (logged by the loader) and the tokens must be equal. The
theta per domain comes from the map verdict (scripts/qom_read.py -> data/QOM_read.txt).


## Wed 21:2x: the evening 45 GB slot vs memory (the early ticket did not run)

- The EARLY ticket (20:35-21:15, eve_early.sh) never queued: NO-SWAP (free + speculative + purgeable + file-backed)
  stayed at 41 GB, below its WAITMEM 44, from P-CAP's end to 21:15, with Kamen's apps open and Chrome closed.
  Stopped by hand; it had queued nothing (data/turn45.log).
- Lifetime peaks measured on earlier runs (data/r57/*_runs.json, mem[1]):
  - U3C on the 12 HumanEval: 30.9 GB at C 160, 39.4 at C 224. That is 0.134 GB per row, i.e. recsz x 48 layers.
  - The QOM all-in adds +0.3 (qalq 31.2).
  - Lever 2's forecast reads (FC_F 4) add +2.6 to +3.3 (qaf / qfx 33.5-33.6 against qs / qo1 30.2-30.9, on 2 prompts).
  - The biggest all-in arm (qallf) therefore peaks at about 34.8 + 0.134 (C - 160) GB, with 0.5 GB spare.
  - Under RULES (NO-SWAP >= peak + 4 under a quiet lock), C 224 needs 47.4 GB. The old 23:00 ticket's PEAK 40
    understated that.
- The 23:00-23:15 ticket now runs scripts/eve_2300b.sh. Every all-in arm runs at one C: the largest that fits at the grant
  (224 if NO-SWAP >= 47.4, 208 if >= 45.2, 192 if >= 43.1, else 176). The ratios (all-in vs U3C, relaxed, + lever 2)
  stay clean; the absolute tok/s at C < 224 is a lower bound for the 45 GB route. Lever 2 stays at C 160.
- The C 224 headline needs Kamen's apps closed (the night) or ~6 GB more free.
- Thu (the orchestrator 21:3x): the C 224 trio (u3ct / qallt / qallft, scripts/c224_thu.sh) is step 2 of the morning
  ticket (night_thu.sh: C 256 headline -> C 224 trio -> agent LOO cut at 09:00 so runner 42 starts by ~09:10).
- Thu (the orchestrator 21:5x): no new agent-LOO arm after 08:35, so the ticket touches q42/GO by ~08:45 (runner 42's
  original start); night_thu.sh also caps steps 1-2 at 08:10 / 08:30 for a late grant.
- Thu (the orchestrator 22:1x): Leap's DKV is installed in scripts/r57_screen.py behind R57_DKV=1 (+ R57_DKV_ADD=1), default
  off, loading a pinned copy (scripts/dkv.py = Leap 23919fb) by path; CPU smoke: hooks in, uninstall restores. Its token
  check (scripts/dkv_thu.sh: 14 ref14 prompts at C 224 vs REF2-T, plus the copy-free path count) is step 3 of the morning
  ticket, before the agent LOO (cut 08:35). Line (pre-registered): PASS = every token == REF2-T AND fast > 0 -> R57_PRESET=45
  after a same-run speed A/B; FAIL -> bisect DKV vs + ADD; VOID -> re-run.
- Wed 22:5x: the 23:00 ticket was cancelled before it queued: NO-SWAP 36 GB at 22:51 with no big job running (the many
  sessions' anonymous memory), below the 41 GB its smallest rung needed. The relaxed 0.5 arm and lever 2's exact stage 1 move
  to the Fri 00:30 night (with FAR-G1 and Rule C); the C 224 trio runs in the Thu 07:05 ticket.

## Thu 5 Nov morning ticket (07:38-08:15, memory-adaptive night_thu.sh): results

The ticket got its quiet lock at 07:38, after RELAX. NO-SWAP was 47.3 GB at the start. The runner 42 GO was touched at
08:14:53, ~30 min before the planned ~08:45.

1. **C 256 headline: SKIPPED.** Its arms need NO-SWAP >= 52.7 GB; there was 47.3.
2. **FAST trio at C 208:** the largest C where qallft fit. 12 HumanEval, order u3ct / qallt / qallft (data/c224t/c224t_report.txt).

   | arm | geo steady tok/s | kept per check | SSD MB per kept | lifetime peak | ck ms per check |
   |---|---|---|---|---|---|
   | u3ct (U3C len) | 102.9 | 10.54 | 20.4 | 37.3 GB | 86.4 |
   | qallt (QOM all-in) | 97.6 | 10.33 | 8.1 | 37.6 GB | 91.2 |
   | qallft (+ lever 2) | 93.4 | 10.33 | 51.5 | 43.4 GB | 96.4 |

   - Ratios: qallt vs u3ct -5.1 %; qallft vs qallt -4.3 % (lever 2's forecast reads: 76,957 issued, 9,010 used).
   - **Confounded, no verdict.** The arms ran in a fixed order with no control arm (the order control was in the cancelled
     Wed 23:00 run). The die warmed 60 -> 84 C under the 5,500 rpm quiet fan cap, and ck rose 86 -> 91 -> 96 ms in arm
     order, although the all-in's levers should lower ck. The byte figures stand: the all-in reads 0.40x U3C's SSD bytes
     per kept token.
   - The by-day C 160 result (all-in +26.3 % vs U3C) is not contradicted until an A-B-A run says so. The Fri relaxed-copy
     run gets u3c at both ends.
   - Memory: qallft peaked at 43.4 GB at C 208, above the ladder's 41.2 estimate. Forecast reads add ~5.8 GB over qall at
     C 208, not 2.6-3.3. The Fri ladder uses the measured value.
3. **DKV token check at C 224: PASS** (pre-registered). Tokens == REF2-T on 14 / 14 prompts; the copy-free path was taken
   1,345 times (slow 180, adopt 125). Next: a same-run speed A/B (on / off), then R57_PRESET=45.
4. **Agent LOO: INCOMPLETE** (2 of 9 arms).
   - aold ran 444 s, lifetime peak 40.3 GB.
   - anew ran 718 s, lifetime peak 42.1 GB, and included a 182 s heat freeze at 08:06.
   - The SWAP GUARD stopped -vtier at 08:14:50: swap grew 8.39 -> 10.51 GB since the LOO started.

## Thu 5 Nov 10:20-10:28: the records3 in-place conversion (done)
- Ran by hand after runner 42's GO: NAX6_DONE 'D4X: PASS', CHECK1_DONE 'check 1 rc 0 (10:20)', RELAX_DONE written 08:55,
  heads-up 08:54:24 (86 min before), BigMoE's quiet lock, nothing open in records3.
- The read-only check passed; md5 of the 48 row-major files (data/records3_md5_rowmajor.txt); converted in 370 s;
  md5 of the converted files (data/records3_md5_p11.txt); index check: 48 layers native+p11+pad with their rowmajor
  entries kept, and load_index refuses row-major readers. Log: data/conversion.log.
- Overlays: scratch/model_mix's index renamed .rowmajor-stale; the decision map's overlay (already renamed .STALE-... by its
  owner) still has its own index and must be re-indexed or deleted. Back: convert_records_p11.py --inverse with the
  row-major md5s.

## Fri 6 Nov 00:52-01:23: the palindrome run (data/fri/fri_report.txt)

The quiet lock came at 00:52 with 41 GB available, which gave the C 192 rung. The frx0 gate passed: the relaxed copy with
its flags off == REF2-T (2 prompts, 128 tokens). Every arm ran both passes on all 12 HumanEval. A heat freeze hit the
first half at 01:00-01:02 (162 s; it was not flagged PAUSED).

| arm | pair-mean geo steady tok/s | first / second pass | kept per check |
|---|---|---|---|
| u3c | 82.5 | 89.8 / 75.1 | 10.61 |
| qall | 86.8 | 88.3 / 85.3 | 10.05 |
| s99 | 81.3 | 71.9 / 90.7 | 10.10 |
| s90 | 89.1 | 85.8 / 92.2 | 10.14 |
| qallr | 87.9 | 98.5 / 75.0 | 11.04 |
| span | 78.5 | 88.9 / 67.8 | 11.10 |

### Verdicts

**Split (s99 / s90): VOID, not NOT PASS.** The all-in arms set REF2_OOO_1C. On that one-call path, Leap's ref2 applies
QOM's drop inside the kernel (ooo1c.a3c_one_call_qom; t["qom"] is popped) and never calls qom_keep_mx. The split plumbing
(R57_QOM_SPLIT) and the new group counters live in r57's two-phase _ooo_call_t3, which these arms do not use:
- the counters are 0 on every check;
- the strict rows (7.1 / 8.9 per check) are counted on the host only.

So s99 / s90 ran qall. Their -6.3 % / +2.7 % against qall measure tonight's noise floor between identical configs: about
+-6 %.

**Rule C: KILL** (pre-registered).
- s per task 1.167x; tokens per task 1.000x.
- Kept per check is +10.4 %, but tok/s is -9.5 %: the per-check host work (token decodes for the tagger and a sync per
  check) costs more than the extra kept tokens.

**Relaxed 0.5:** tok/s +1.3 % (inside the noise) at kept +9.9 %: the same per-check cost eats the gain.

**C3-R8: unresolved.** On this path 'uses' is the loader's posted count, before the drop: qall 2,812 vs exact 3,708 per
check (0.759x). Groups really computed need the counter inside the one-call path.

### Next
- Test the split on the two-phase path, where it works: qall / s99 / s90 with REF2_OOO_1C off. No new code is needed.
- Or have Leap move the split into the one-call kernel's keep flags.

## Fri 6 Nov 03:23-03:39: the split A/B on the one-call path (Leap's QOM-SPLIT-1C; data/fsp/fsp_report.txt)
C 224 (46 GB NO-SWAP at the grant), 12 HumanEval, the palindrome qall / s99 / s90 / qallm / s90b / s99b / qallb, Leap's one-call
counters in every arm.

**The split ENGAGES:**
- groups computed per check: qall 2,359; s99 1,475 (0.625x); s90 1,165 (0.494x), on Breakthrough's 45 GB-pool replay
  (x0.628 / x0.508);
- split drops per check 1,917 / 2,468; strict rows per check 6.6 / 8.5.

**Verdicts (Thu 10:40 line): both NOT PASS.**
- s99: +6.2 % tok/s, kept -4.2 %. s90: -2.8 % tok/s, kept -1.7 %.
- The run was heat-disturbed: the middle qall pass ran -21.3 % vs its end passes, and a freeze dropped s99b.
- The clean early passes agree with the verdicts: qall 96.3 / 97.5, s99 95.8, s90 99.2 tok/s.
- So a 38-51 % cut in computed groups buys ~0 tok/s at C 224: the check is not bound by expert groups there (the fixed
  cost per check, FLAT-2's 53.7 ms, and the SSD dominate).

C3-R8: qall 2,359 vs exact's 3,708 (Fri's fexr; exact groups do not depend on C) = 0.636x (the audit asked >= 0.85x).

## Fri 6 Nov 08:04-08:10: FAST-NAX engine A/B (data/fnx/fnx_report.txt)
C 224 (NO-SWAP 49 GB), 12 HumanEval, A-B-B-A, no freezes, qall drift -1.2 %.

| arm | geo steady tok/s | passes | kept per check |
|---|---|---|---|
| qall | 97.5 | 98.1 / 96.9 | 10.55 |
| FAST-NAX (R57_NAX=1 REF2_NAX_1CQ=1) | 113.9 | 112.7 / 115.0 | 10.36 |

**Verdict (pre-registered): BETWEEN.** tok/s +16.8 %, kept -1.8 %; the line needed kept within 1 %.
- The kept change comes from the base: NAX6-N is a certified exact reference with its own numerics.
- tok/s already nets it.
- Proposed for tonight's stack, with EvalSuite's same-scores as the gate (the orchestrator decides).

### Fri 11:25: both pre-flights VOID (my own race), re-armed as one chain

The D7BNAX_32=1 re-run (pf3) and never-wait bit check 4 (pf4) waited separately for the same clear Mac. They passed their
3 clear polls together and started at 11:25:06: 2 x ~31 GB, swap 11.5 -> 14.6 GB. The swap guard stopped f3k and f4n after
~27 s, and f3r and f4k never started (0 / 2 each). No result. Fix: scripts/pf_chain.sh runs them in sequence, and both
scripts look again 20 s after their share lock (any > 5 GB job or NO-SWAP < 40 GB: release and wait). Tonight's env files
also take NewIdeas' gate rule (QUEUE 70a4951): R57_QOM_SETS=humaneval,mbpp.

### Fri 12:4x: pre-flights after the crash fix (1f16f71): D7BNAX_32 stays =1; never-wait check 4 CLEAN

Both chains ran 2 / 2 prompts per arm with no new error. The "error lines 2" in pf3.log / pf4.log are the 11:37 / 12:29
crash's lines left in the same per-arm log; the scripts now clear those logs at their start.

**D7BNAX_32=2 vs =1** (2 prompts, CAP 32, C 160; =2 ran 09:27, =1 12:46, fresh processes, same order):

| arm | HumanEval/68 (code: the D7b drafter, where D7BNAX_32 acts) | ifeval/1220 (chat: the MTP head drafts) |
|---|---|---|
| f3k (stack + CD16) | identical | differs within its first tokens |
| f3r (stack + relax v2) | identical | identical |

The rule (the orchestrator): use =2 only if it matches =1 token for token, so **=1 stays** (tonight's script unchanged).
D7BNAX_32 is read only by Leap's d7b_nax.py (the code drafter), so the chat difference is not =2's: the same FAST env
gave different chat tokens in two fresh processes 3 h apart. Some FAST lever depends on machine state or timing, not only on
in-process history (EvalSuite 12:44: QOM-KEEP off still leaves the two prompt orders different).

**Never-wait bit check 4** (R57_NEVERWAIT=1 + REF2_NEVERWAIT=1, Leap ref2; docs/NEVERWAIT_IFACE.md section 4): CLEAN on qall
(f4n) and on stack + CD16 (f4k); ref2's layer-47 assert held. ref2.NWCNT sums (fields: late groups, late uses, late mass,
last-layer late uses, stalls, forced groups):

| arm | checks | late groups (median / check) | late uses | late mass | last-layer late uses | stalls | forced |
|---|---|---|---|---|---|---|---|
| f4n qall + NW | 20 | 4363 (160) | 7982 | 901.6 | 99 | 24 | 0 |
| f4k stack + CD16 + NW | 19 | 3990 (128) | 7636 | 871.9 | 100 | 25 | 0 |

Indicative only (one cold pass each, CAP 32, C 160, 3 min apart, no palindrome): stack + CD16 HumanEval/68 70.7 tok/s
(f3k) vs 75.9 with never-wait (f4k), kept 15.0 both; ifeval/1220 22.2 vs 22.7. The pre-registered speed A/B is Saturday night.

### Fri 13:55: tonight's never-wait arm env pre-flight (pf5): CLEAN

f5v = the stack + CD16 + R57_NEVERWAIT=1 + REF2_NEVERWAIT=1 + R57_QCNT1C=1 (tonight's snv env; the counter + never-wait pair was
the one combination check 4 did not run), 2 prompts, CAP 32, C 160: ran 2 / 2, no error lines. Both per-check counters are
written for all 19 checks (q1c and nw). NWCNT sums: late groups 4094, late uses 7746, last-layer late uses 82, stalls 25.

### Fri 15:40: day run at C 224: NEVER-WAIT KILL, SPLIT-NAX KILL (scripts/fri_day.sh, data/fday)

Kamen extended the big-test window to 17:00; NewIdeas ordered this run first. Palindrome on the 12 HumanEval: stack / ssp / snv /
snvb / sspb / stsb, quiet lock, C 224 (NO-SWAP 44.9 GB; peaks 38.4-38.9 GB), ~92-97 s per pass, no pass overlapped a freeze.
Lines pre-registered by NewIdeas (Fri 10:1x / 13:5x): net pair-mean geo steady tok/s vs the stack, PASS >= +8 %, KILL < +3 %.

| arm | geo steady tok/s (passes) | vs stack | kept per check | verdict |
|---|---|---|---|---|
| stack | 157.9 (157.2, 158.1) | | 12.58 | |
| SPLIT-NAX (stack + R57_QOM_SPLIT=0.99) | 155.8 (156.4, 154.6) | -1.3 % | 11.87 (-5.7 %) | KILL |
| NEVER-WAIT (stack + R57_NEVERWAIT + REF2_NEVERWAIT) | 153.5 (153.2, 152.3) | -2.8 % | 13.16 (+4.6 %) | KILL |

Per check (checks >= 2, medians), stack vs never-wait: check 68.8 vs 75.3 ms, gate service 14.0 vs 15.6 ms, misses 37 vs 39,
stalled layer calls 17 vs 17, eval wait 7.5 vs 7.3 ms. Never-wait counters per check: late groups 24, late uses 57, late
mass 6.3, last-layer late uses 0, stalls 0. Read: at C 224 the stack misses only ~37 experts per check, the stalled layer
calls did not drop, and the late work added ~6.5 ms per check. NewIdeas' +10-15 % assumed ~13.5-15 ms of misses per check.
Both arms leave tonight's palindrome; never-wait does not join the gated stack. The MBPP12 pair failed at once (rc=127: the
c45 runner splits arm envs on commas, so R57_QOM_SETS=humaneval,mbpp broke). It was re-run with the sets in the environment
(scripts/fri_mbpp.sh).

### Fri 15:51: MBPP12 CD16 pair (GOAL's line): speed PASS, quality KILL (3 of 12 tasks flip to fail)

scripts/fri_mbpp.sh: qall vs qall + R57_INERT=CD16 on GOAL's 12 MBPP (mbpp12_1106/MBPP12.md), CAP 384, C 224, palindrome mq / mc /
mcb / mqb, R57_QOM_SETS=humaneval,mbpp, quiet lock, no cuts, no freeze.

| | qall | qall + CD16 | ratio | GOAL's line |
|---|---|---|---|---|
| geo steady tok/s | 71.3 (70.2, 72.5) | 88.2 (88.9, 85.7) | +23.7 % | |
| kept per check | 6.48 | 9.38 | K x1.449 | PASS >= 1.30 |
| tokens per task | | | x1.020 | PASS <= 1.05 |
| tests passed (EvalSuite mbpp_score.py) | 11 / 12, both passes | 8 / 12, both passes | 3 flips per pass pair | KILL at >= 2 flips |

Flips pass -> fail: pass 1 mbpp/16, 296, 319; pass 2 mbpp/16, 268, 319 (data/fday/mbpp12_verdicts.json). INERT counters: rows
open 1236 / 1096 (> 0, as GOAL's line asks). The answers show why: CD16 keeps comment rows whatever the 125B says, and the
drafter's comments loop ("the string, the string, ...") until the answer is spent (mbpp/16), or the code it writes after its
own comments differs (mbpp/319). **Verdict: KILL on quality** (speed PASS). CD16 is in tonight's STACK and in the env files
EvalSuite gates; that decision went to NewIdeas at once.

**Correction (Fri 21:3x, read with Leap's NW v3 question):** the per-check "stalled layer calls" field (C3-R8, the ext's
`stalled` stat) is NOT a wait. ensure() counts `stalled += !jobs.empty()` (bigmoe_ext.cpp:881): every layer call that issued any
SSD read. Never-wait's late reads are still issued in their layer's call, so 17 vs 17 says nothing about waits. The day run's
split per check (medians, checks >= 2): graph build (it includes the sync-free gate's per-layer waits for router ids) 59.6 ->
66.1 ms (+6.5), eval wait 7.5 -> 7.3, gate service 14.0 -> 15.6, SSD io 11.3 -> 12.8, misses 37 -> 39. No wait disappeared; the
per-layer pipeline got 6.5 ms slower. In the ext the release is per layer call, after all of that call's non-optional reads
land; an optional-only miss is never read (-2) but resolves with that release.

### Sat 7 Nov 23:50-00:23: the 200 STACK night (scripts/sat_night.sh, data/sat; quiet, C 224, no freeze)

Lines pre-registered in the script header before arming (NewIdeas 4b2c45c / 21:5x, the orchestrator 21:2x-21:4x).

| arm (12 HumanEval, pair-mean geo steady) | tok/s | vs the stack | kept / check | lifetime peak | verdict |
|---|---|---|---|---|---|
| qall | 99.1 | | 10.45 | 39.1 GB | (noise floor +1.0 %) |
| stack (FAST-NAX + DN-KS + GDN lazy + D7BNAX_32 + FAST-tax) | 132.6 | | 10.48 | 38.5 GB | headline PASS: +33.8 % vs qall |
| stackr (relax v2, DELTA) | 145.3 | +9.5 % | 11.64 | 38.7 GB | PASS on speed; SPEED ONLY (relaxes code tokens) |
| stackrm (stackr + MATCH g2 at 16 rows) | 144.6 | +9.0 % | 11.74 | 38.5 GB | PASS on speed; SPEED ONLY |
| stackb (stack + ship B span R2:2:0.25 + g2) | 140.1 | +5.6 % | 11.33 | 39.4 GB | PASS (>= +2 %): code_env |
| stack at C 256 | 136.5 | +2.9 % vs C 224 | 10.39 | 42.7 GB | C 256 KILL (< +3 %) |
| stackb at C 256 | 144.9 | +3.4 % vs C 224 | 11.34 | 42.8 GB | C 256 PASS (>= +3 % AND <= 45.0 GB) |

- **FINAL stacked number (quality gate pending: EvalSuite same-scores on stackb):** stackb at C 256, 144.9 tok/s geo steady on
  the 12 HumanEval, lifetime peak phys_footprint 42.8 GB.
- **MATCH g2 at 16 rows:** its first GPU run under R57_FAST, a 2-prompt check, was clean. stackb's gain is almost all g2: ship
  B's span rule relaxed only 21 rows of ~5,900 (span stats). Copy counters: positions ~5,400, used 3,044, accepted 2,888.
- **T16 MBPP12 (GOAL's line): BETWEEN.** Pair 1 had all 12 verdicts equal. Pair 2 flipped mbpp/319 to fail. Median tokens per
  task x1.088 (> 1.05). K ratio 1.035. Rows open 2,288, kept whole 0; the guard audit found every accept inside the rule. No stackt.
- **Never-wait ceiling (R57_WARMSPLIT=4): PASS.** Median cold16 - warm16 is 18.5 ms over 58 samples vs S 9.7 ms; sample misses 46
  vs live 28 per check.
- **A void first try, 23:42.** A ticket started at 23:42 ran 0 arms (c45_runner's deadline compared HHMM across midnight;
  fixed 4ef9bb2). It is recorded as VOID, and part B was stopped before it read the void flags.

### Sat 00:33-00:40: S2 A/B at C 256 (scripts/s2_ab.sh, data/s2; quiet, no freeze): PASS +3.1 %, tokens differ -> S4's gate

S2 = Leap's glue (R57_SBN=2: submit every 2 layers) + D7BNAX_32=2 + R57_DSUB=1 + R57_SVH=1 on stackb (code_env). Line (NewIdeas
Sat 00:4x, pre-registered): PASS >= +2 %, KILL < 0 %; tokens equal on all 12 -> no gate, else S4's gate.

| arm (12 HumanEval, C 256) | geo steady tok/s (passes) | kept / check | median cycle | lifetime peak |
|---|---|---|---|---|
| B = stackb | 142.4 (139.9, 144.6) | 11.28 | 79.2 ms | 42.5 GB |
| S2 | 146.8 (144.5, 148.8) | 11.38 | 77.9 ms | 42.8 GB |

- Verdict: **PASS, +3.1 %**. The cycle is 1.3 ms shorter, at the low end of Leap's -1.3..-3.6 ms.
- Tokens identical per prompt: S2 vs B 7 and 10 of 12. The floor is B1 vs B2 at 7 of 12 (S2 vs S2: 9), so under QOM at C 256
  the outputs differ run to run, and the token route cannot clear S2. By the rule, S2 joins S4's gate: env file
  data/stack/stackb_s2_env.txt.
- SESSION-VOCAB counters: chk_out 0, svh_extra 3 == dsub_extra 3; svh_calls 273 / 270 per pass.
- New best: **stackb + S2 at C 256, 146.8 tok/s** (peak 42.8 GB), pending EvalSuite's gate.

### Sat 01:3x: S2c REF2_KEEP_CACHES=1 (scripts/s2c_ab.sh, data/s2c): EXACT proof PASS + engine A/B PASS +10.2 % -> joins v1

Leap's fix: ref2.remove() before each draft no longer drops REF2's per-weight caches, so checks stop re-padding and re-packing them.
- **EXACT bitwise proof** (NAX6-N exact env = Leap's nfnc env, C 224, CAP 64): keep off vs on gives identical tokens on 3 of 3
  prompts (HumanEval/68, /16, ifeval/1220; 64 tokens each).
- **Engine A/B at C 256** (12 HumanEval palindrome, quiet, no freeze): stackb + S2 149.1 tok/s (148.1, 150.0) vs + KEEP 164.2
  (161.6, 166.5) = **+10.2 % (PASS >= +4 %)**. Median cycle 75.8 -> 66.9 ms; lifetime peak 43.1 GB (<= 45.0).
- **New best: stackb + S2 + KEEP at C 256, 164.2 tok/s, 43.1 GB** (gate: S4).

### Sat 01:46-01:53: S1 NW v3 engine A/B at C 256 (scripts/s1_ab.sh, data/s1; quiet, no freeze): CORRECTED Sat 05:1x: BETWEEN on the full period (+4.3-4.8 %), not PASS; the +9.0 % below is tpsw's window

> **CORRECTION (Sat 05:1x; NewIdeas 04:5x, re-derived by BigMoE from data/r57/s1{b1,b2,n1,n2}_256_{runs,checks}.json):** tpsw =
> sum(kept[2:]) / sum(c2 - c0) leaves out each check's c2 -> next c0, and NW spent more time there: 4.71 / 4.78 ms per check vs
> B's 2.37 / 2.33. Over all wall time of checks >= 2 (full period): B 153.4 / 164.4, NW 166.7 / 166.3 tok/s, **+4.8 %**
> (NewIdeas' cut +4.3 %); whole decode B 133.4 / 141.7, NW 143.7 / 140.5, **+3.3 %**. Both are BETWEEN on S1's line (PASS >= +8 %,
> KILL < +3 %). Cause in the code: with ref2.NWCNT a list, each layer call adds a sort + 8 sums (Leap ref2.py:935) that are not
> ancestors of the logits, so the whole counter graph runs at the post-c2 readback (screen line 1355) while the host waits. B
> synced there too (R57_QCNT1C=1 in both arms: the q1c readback, line 1353), so the extra ~2.4 ms is NW's counters, which only
> measure. Every tpsw headline sits ~4 % above the full period (B 165.3 vs 158.8). Re-test: S1b below (NW with R57_NWCNT=0, the
> exact full period from per-check c0). Until S1b, NW's place in v1 is open; S4 (4ef3734, 04:40) runs NW with counters.

Base B = v1 candidate stackb + S2 + KEEP; NW = B + R57_NEVERWAIT=1 + REF2_NEVERWAIT=1 (Leap's ref2 v3, ddbc0621; the r57 side
gated per set). The 2-prompt NW check was clean. The nw3 map had PASSed at both pools (Breakthrough 01:4x).

| arm (12 HumanEval, C 256) | geo steady tok/s (passes) | kept / check | median cycle | lifetime peak |
|---|---|---|---|---|
| B | 165.3 (159.3, 170.6) | 10.98 | 67.4 ms | 43.9 GB |
| NW v3 | 180.3 (180.4, 180.1) | 11.02 | 65.8 ms | 43.9 GB |

- **Verdict:** PASS, +9.0 % (line: >= +8 % AND peak <= 45.0 GB). Caveat: B's two passes differ by 7 %, so the margin is within
  the noise. The pre-registered line holds.
- **NWCNT per check (medians):** late groups 18, late uses 37.5, late mass 4.3, L47 drops 0, L47 stalls 0, optional dropped now
  496.5, sidecars 3-bit 0.
- **Tokens:** NW vs B identical on 6 and 6 of 12 (floor B1 vs B2: 8). NW v3 is lossy (its 3-bit sidecar / drop paths), so it goes
  to S4's gate.
- **NEW BEST: stackb + S2 + KEEP + NW v3 at C 256, 180.3 tok/s geo steady on the 12 HumanEval, peak 43.9 GB** (gate: S4).
  (CORRECTED: 180.3 is tpsw; on the full period the same arm is 166.5 tok/s and B 158.8. See the correction above.)

### Sat 05:35-05:47: S1b (scripts/s1b_ab.sh, data/s1b; quiet, no freeze in the run): BETWEEN, full period +6.9 % -> v1' = stackb + S2 + KEEP (no NW)

| arm (12 HumanEval, C 256) | full period tok/s (passes) | whole decode | tpsw | outside c0..c2 | median cycle | kept / check | lifetime peak |
|---|---|---|---|---|---|---|---|
| B = v1 minus NW (production env: no QCNT1C) | **164.2** (163.7, 164.3) | 141.2 | 164.7 | 0.17 ms | 65.2 ms | 10.38 | 43.8 GB |
| NW v3, counters off (R57_NWCNT=0) | **175.6** (175.5, 175.5) | 149.4 | 176.1 | 0.16 ms | 61.6 ms | 10.32 | 43.0 GB |
| NW v3, counters read after the clock (R57_NWCNT=2; 5th arm, information, outside the palindrome) | 182.6 | 154.7 | | | | | 42.9 GB |

- **Verdict (pre-registered):** BETWEEN: full period **+6.9 %** (whole decode +5.8 %, tpsw +6.9 %), NW peak 43.0 GB. By NewIdeas'
  rule (05:1x), v1' = stackb + S2 + KEEP, written 05:49: snapshot 03c7782 (data/r57/snap/s4b/ + frozen_s4b, so S4's PASSed s4/
  stays), final_env.txt dd7ad0f (26 K=V, no NW). Its S4 runs in the v2 slot by 16:00. **v1' measured: 164.2 tok/s full period,
  141.2 whole decode, 43.8 GB** (S1b's B arm).
- **S4 (EvalSuite 05:38) PASSED v1 with NW (counters on, 4ef3734):** same scores vs NAX6-N PASS both orders, long code 0
  blunders. So NW's quality passed; only its speed missed the +8 % line.
- **R57_QCNT1C's cost:** S1's B (QCNT1C on) spent 2.35 ms per check outside c0..c2, production B 0.17 ms: the q1c readback.
- **5th arm's +4 % is acceptance, not speed** (NewIdeas 06:0x, checked by BigMoE): its median cycle 61.5 ms (ck 56.4, drf 5.26)
  equals the counters-off passes (61.8 / 61.5); kept per check (checks >= 2, pooled) 10.52 vs 10.05 / 10.21 = x1.038, the whole
  gap. One pass of acceptance noise (B's own passes: 9.63 / 10.21). No palindrome needed. NW per cycle: -5.5 % (61.6 vs 65.2 ms).
  NW v3 is now a post-release lever (NewIdeas): -3.6 ms per cycle, lossy.
- **Late groups at ~counters-off timing (5th arm):** 20.0 per check vs S1's 18.0 (counters read per check); late uses 40.0 vs
  37.5; optional dropped now 507 vs 496.5. Without the counters' 2.4 ms between checks, ~11 % more groups run late.
- **NW GPU check, counters ON vs OFF** (scripts/nwcnt_check.py = Leap's nw_check.py with a counters switch, on S4's ref2
  ddbc0621): both PASS (w) warm == off, (f) FORCE == reference, (c) cold == reference, (l) layer 47. The counters touch no
  output. (nw_check had last run on the v3 candidate dcd5f4f8, so this also re-checks production ref2.)
- **Tokens (information):** NW vs B identical on 5 and 4 of 12; floor B1 vs B2 5.

Setup (as armed 05:13):
- **Arms:** B = final_env 1403cd6 without NW (production-like: no R57_QCNT1C, no R57_RELAX_LOG); NW = B + R57_NWCNT=0 (new
  knob, default on: 4ef3734's behaviour is unchanged). Palindrome b1 / n1 / n2 / b2 on S1's 12 HumanEval at C 256, after a
  2-prompt NW check (clean + the host-side nw_late_calls > 0 on both).
- **Screen change (default env unchanged):** per-check c0 (ms from the decode start) in the checks record, the run's host-side
  NW counters (nwk) in the runs record; the full period = sum(kept[2:]) / (decode wall - c0 of check 2).
- **Line (NewIdeas Sat 04:5x, in the header before arming):** full period, NW vs B: PASS >= +8 % AND peak <= 45.0 GB -> NW stays
  in v1 with R57_NWCNT=0 (EvalSuite says whether S4's verdict on 4ef3734 carries); KILL < +3 % or peak > 45.0 GB; else BETWEEN.
  Not PASS -> v1 = stackb + S2 + KEEP. Runs at S4_DONE (turn45 quiet ticket, ~12 min, window to 12:00).

### Sat 04:35: final v1 written for EvalSuite's S4 (scripts/s4_final.sh; snapshot 4ef3734, final_env.txt 1403cd6)
- **v1 = stackb + S2 + KEEP + NW v3 at C 256** (`S4_VARIANT=stackb_s2 S4_KEEP=1 S4_NW=1`): the stack + ship B span + MATCH g2 + S2
  (SBN 2, D7BNAX_32 2, DSUB, SVH) + REF2_KEEP_CACHES + R57/REF2_NEVERWAIT; QOM (and so NW) on humaneval + mbpp only.
  Engine number so far: 180.3 tok/s on the 12 HE prompts, lifetime peak 43.9 GB (S1). S5 re-measures it vs qall tonight.
  (CORRECTED Sat 05:1x: 180.3 is tpsw; full period 166.5 tok/s. NW's place in v1 now waits on S1b; see S1's correction.)
- **Why stackb, not the stack** (NewIdeas' rule: same-scores + both long-code top-ups; B's top-up is its rollback gate):
  | runner 49 check (EvalSuite) | verdict |
  |---|---|
  | same scores vs NAX6-N (S4's reference: orchestrator 03:59, option b; PREREG_s4 amendment 2 = EvalSuite 39c4462) | PASS both orders (HE 90.0 vs 91.7, p 1.0; math, IFEval PASS) |
  | same scores vs REF2-T (runner 49's own line; on record, S4 reports both) | FAIL both orders (HE 90.0 vs 95.0, -5.00 vs a bar of > -5) |
  | B's code-only drift top-up (51 rows >= 512 tokens) | PASS: 0.30 blunders/10K (<= 1), pooled q4 KLD 0.0067 |
  | the stack's CODE env on all 41 humanevalt rows (tfwin vs NAX6-N) | PASS: 1 blunder in 54561 tokens = 0.18/10K (<= 1), deviations 1.76 % |
  | MBPP 100 | PASS (76.0 vs 79.0, p 0.45) |
- **Checks before writing:** live Leap/scripts, Leap/third_party, TrainLab/sticky, Breakthrough/keepset and the screen are
  byte-identical to S1's tested frozen_s1 (frozen_s4 vs frozen_s1: 0 files differ; screen equal modulo the dir). The env
  equals S1's NW arm minus two logging knobs (R57_QCNT1C counters, R57_RELAX_LOG) plus R57_QOM_SETS=humaneval,mbpp.
  NW acts only where a FAST rule acts (screen `nwp_ = NW and fz is not None`; ref2 gates on NW_ON), so math / IFEval run
  exact with NW off: Breakthrough's nw3 CHAT FAIL (be2c175) does not reach v1. NW v3 is code-validated only.

### Sat 07:04-07:05: FIRSTCHK kill test (scripts/fc_kill.sh, data/fc; quiet ticket, runner 53 yielded): ONE-TIME, not host
Orchestrator / NewIdeas 07:0x (QUEUE db21669): in v1' each prompt's check 0 takes ~306 ms vs ~58 ms later (build 287 vs 53 ms),
15.8 % of the decode wall; ceiling +21 % whole decode [P]. v1' env (final_env dd7ad0f), 3 HumanEval, C 256, CAP 64, main-thread
CPU per check (R57_TCPU=1). fcrep = R57_A6=repf: copy 1 of check 0 under the real check's FAST gate, rolled back, then the real
check 0 (copy 2). ('rep' ran copy 1 ungated, which would leave the FAST path's first use in copy 2.)

| check (medians of 3 prompts) | wall ms (build + wait) | main-thread CPU ms | demand SSD ms (per prompt) | misses (per prompt) |
|---|---|---|---|---|
| control check 0 | 326 (292 + 34) | 34 | 112 / 56 / 41 | 385 / 131 / 76 |
| fcrep copy 1 | 312 (288 + 23) | 32 | 109 / 56 / 41 | 385 / 133 / 77 |
| fcrep copy 2 (the same check, again) | **61** (56 + 5) | 22 | 0 | 0 |
| later checks 1-3 | 79-90 | 23 | 14-52 | 66-291 |

- **Verdict (pre-registered): ONE-TIME, GPU:** copy 2 61 ms < 200; host CPU in the excess 10 ms of ~221 ms.
- The excess is not host Python (thread CPU +10 ms) and mostly not the SSD (copy 1's demand reads are 41-109 ms vs 14-52 ms in
  later checks that have MORE misses). ~150-200 ms per prompt is the main thread blocked with no CPU. The prefill is evaluated
  before the decode clock (mx.eval per chunk + .item()), so it is not deferred prefill work.
- Candidates (not tested): a GPU-side or external wait paid once per prompt. Metal pipeline builds (they run in
  MTLCompilerService, so they don't count as thread CPU), the pool / tier switch after a 4-bit prefill (pf1), first touch of
  freshly allocated buffers (A6 keep was ruled out at C 160 exact, 11-04). Next: Leap's scope (e.g. sample the main thread
  during check 0, or time each layer's router wait in check 0 vs copy 2).
- Caution for any fix: a dummy check before the decode clock (A6 dummy) only moves the cost out of the clock (time to answer
  unchanged). A lever must remove it or overlap it with existing waits.

### Sat 09:07-09:13: S6W wired-limit engine A/B on v1' (scripts/s6w_ab.sh, data/s6w; quiet): PASS, whole decode +27.4 %
Leap's FCP (038bfa1): check 0 re-wired the 32 GB expert pools (IOGPU's wired collector unwired them during each prefill);
mx.set_wired_limit(max_recommended_working_set_size) once after model load removes it, bitwise in the exact env (43/43 logits
hashes). W = v1' + R57_WIRED=1 (screen 39e21da). 12 HumanEval, C 256, palindrome b1 / w1 / w2 / b2.

| arm | whole decode tok/s (passes) | full period (passes) | check 0 (median) | median cycle | kept / check | lifetime peak |
|---|---|---|---|---|---|---|
| B = v1' | 142.2 (138.9, 145.5) | 169.3 (166.7, 171.7) | 320 ms | 64.9 ms | 10.75 | 42.9 GB |
| **W = v1' + wired** | **181.2** (183.9, 178.4) | **176.5** (179.9, 173.0) | **65 ms** | 63.5 ms | 11.03 | 42.9 GB |

- **Verdict (pre-registered): PASS:** whole decode **+27.4 %** (line >= +8 %), full period +4.3 % (>= -2 %), W peak 42.9 GB.
  NewIdeas' price was +18.8 % [P]. Part of the full-period gain is acceptance noise (kept +2.6 %); the cycle is -2 %.
- **Tokens:** W vs B identical on 7 and 9 of 12; the floor B1 vs B2 is 7 and W1 vs W2 is 7 (QOM varies run to run). The
  carry-over proof for S4's v1' verdict is Leap's exact-env bitwise check (residency changes no kernel, input or order).
- **VTIER:** its buffers are ext newBuffer allocations marked volatile (bigmoe_ext.cpp:2011), outside MLX's allocator, so
  the residency set does not wire them; the peak is unchanged (42.9 GB both).
- **v1' + wired = 181.2 tok/s whole decode, 176.5 full period, 42.9 GB at C 256** (12 HumanEval). It goes into S5 tonight
  (R57_WIRED=1, explicit in the env line).

### Sat 14:12-14:25: S5, the number for Kamen (scripts/s5_final.sh, data/s5; quiet; snapshot d8eb2e4): CLEAN -> cert_env.txt
- **Freeze:** every copied .py's code paths rewritten into frozen_s5 (344 files: ~/Leap/scripts, ~/Leap/third_party, ~/TrainLab/sticky,
  "~/TrainLab" -> TrainLab_root, Breakthrough's keepset). S4's snapshot had rewritten only the screen; ref2 / ooo1cq / tierglu_nax6n
  insert ~/Leap/scripts themselves, so 17 files loaded live. Module manifests (R57_MODMAN=1): 30 modules from the snapshot per
  process, **0 live**. Pre-flight clean (2 / 2).
- C 256, qall / v1' + wired / v1' + wired / qall, one process per arm, no freeze window in the run:

| set | arm | whole decode tok/s (passes) | full period (passes) | check 0 | cycle | kept / check | tokens / task | lifetime peak |
|---|---|---|---|---|---|---|---|---|
| 12 HumanEval | qall | 92.5 (92.0, 92.9) | 100.9 (100.4, 101.3) | 330 ms | 105.3 ms | 10.17 | 250 | 42.5 GB |
| 12 HumanEval | **v1' + wired** | **175.9** (174.6, 176.9) | **170.5** (169.1, 171.6) | 68 ms | 64.4 ms | 10.74 | 242 | **42.9 GB** |
| MBPP12 | qall | 65.8 (66.1, 65.5) | 72.8 (72.4, 73.1) | 307 ms | 85.4 ms | 6.25 | 123 | 42.2 GB |
| MBPP12 | **v1' + wired** | **118.3** (118.1, 118.4) | **115.4** (115.0, 115.7) | 60 ms | 52.2 ms | 6.21 | 123 | 42.6 GB |

- **v1' + wired vs qall:** 12 HumanEval whole decode **+90.1 %**, full period +68.9 %; MBPP12 whole decode **+79.7 %**, full
  period +58.5 %. Tokens identical pass to pass: HumanEval v 9 / 12, q 9 / 12; MBPP v 7 / 12, q 6 / 12 (QOM varies run to run).
- S6W measured the same config at 181.2 whole decode (passes 183.9 / 178.4) this morning; S5's 175.9 (174.6 / 176.9) is ~3 %
  lower, within session-to-session spread. HumanEval's whole decode includes check 0's copied rows; MBPP12 is the plainer number.
- **cert_env.txt = s5_env.txt** (8036f5e): v1' + wired, screen data/r57/snap/s5/r57_screen_s5.py, snapshot d8eb2e4, 27 K=V
  (R57_QOM_SETS=humaneval,mbpp). EvalSuite's CERT from Sun 01:30.

### Sat 14:31-14:37: T16-HE, T16's HumanEval engine A/B on v1' + wired (scripts/t16he_ab.sh, data/t16he; quiet; S5's snapshot d8eb2e4): KILL
MTP Head's line (~/TrainLab/REPORT.md 'T16-HE', bccd23c). B = v1' + wired; T = B - span B + R57_INERT=T16 (+ per-pass INERT log).
C 256, the 12 HumanEval, B / T / T / B, 0 live outside modules.

| arm | whole decode (passes) | full period (passes) | median cycle | kept / check | tokens / task | check 0 | lifetime peak |
|---|---|---|---|---|---|---|---|
| B = v1' + wired | 173.2 (172.3, 173.8) | 168.0 (167.7, 168.0) | 63.6 ms | 10.56 | 246 | 68 ms | 42.9 GB |
| T = + T16 (span B off) | 173.2 (169.2, 176.6) | 167.8 (163.6, 171.3) | 65.5 ms | 10.89 | 246 | 67 ms | 42.9 GB |

- **Verdict (pre-registered): KILL:** whole decode -0.0 % (line: PASS >= +3 %, KILL < +1 %); tokens per task T / B 1.000; no runaway.
- **Why:** T16 keeps +3.1 % more tokens per check (10.89 vs 10.56) but its cycle is +1.9 ms (+3.0 %) slower, so the net is zero.
  Span B left with T (the screen runs INERT and span as separate arms), so the extra cycle is most likely InertScan's per-check host
  work on v1''s host-bound path. T16 rule-kept rows 47 of 2912 rows in spans (INERT log: 47 lines).
- **Tokens:** T vs B identical on 3 and 4 of 12; the floor B1 vs B2 is 8 (T16 is lossy, as expected).
- B here (173.2) sits below S5's 175.9 and S6W's 181.2 for the same config: the day's spread; the A/B is within one run.

### Sat 14:43-14:54: GDNB, R57_GDNB=1 on v1' + wired (scripts/gdnb_ab.sh, data/gdnb; quiet; S5's snapshot d8eb2e4): G KILL, G' BETWEEN
NewIdeas (PLAN 220 #2 slot): v1' carries REF2_GDN_LAZY=1 but no R57_GDNB, so its lazy commit is inert. G = v1' + wired + R57_GDNB=1
(the lazy commit engages); G' = G without REF2_GDN_LAZY. Leap 14:4x (before the run): the body alone saves ~0.13 ms per check here.

- **Bits (re-run today, exact NAX6-N env, C 224, CAP 64, LOGHASH):** G form and G' form both == gx0 on all 43 checks' verify-logits
  hashes and 3 / 3 prompts' tokens (GDNB-2 PASS re-confirmed on today's ref2 / gdnbody).
- **Engine (C 256, 12 HumanEval, B / G / G' / G' / G / B):**

| arm | median cycle | vs B | whole decode | full period | kept / check | check 0 | peak |
|---|---|---|---|---|---|---|---|
| B = v1' + wired | 63.48 ms | | 169.7 | 164.5 | 10.10 | 67 ms | 42.9 GB |
| G = + GDNB (lazy commit on) | 63.99 ms | **+0.52 ms (+0.81 %)** | 174.9 | 169.8 | 10.59 | 65 ms | 42.9 GB |
| G' = GDNB, lazy commit off | 63.55 ms | +0.08 ms (+0.12 %) | 172.3 | 167.2 | 10.31 | 65 ms | 42.9 GB |

- **Verdicts (NewIdeas' exact-safe line: PASS cycle <= -1 % AND bits; KILL cycle >= +0.5 %): G KILL, G' BETWEEN.** Neither joins the
  Sunday bundle. The whole decode / full period gains (+3.1 % / +1.6 %) are acceptance (kept 10.59 / 10.31 vs 10.10: QOM's pass-to-pass
  noise), which is why the line judges on the cycle. Matches Leap's prior: the body saves ~nothing and the engaged lazy commit costs on
  full accepts more than it saves on partial ones.

### Sat 15:14-15:21: xsafe p1a = PLAN 220 1a (Leap eprep + GDN TG 2, ref2 d6f736fb) + BigMoE #4 (R57_DQ1) on v1' + wired: BETWEEN
scripts/xsafe_ab.sh (a317590; reusable: new deep-frozen snapshot, exact-env bits on LOGHASH, B / X / X / B at C 256, NewIdeas' cycle
line, ref2 COUNTS engagement). Snapshot eba45a4 (345 files frozen; frozen ref2 == live d6f736fb modulo paths). Data data/xs_p1a, 07dbca9.

| | median cycle | whole decode | full period | kept / check | check 0 | peak |
|---|---|---|---|---|---|---|
| B = v1' + wired | 63.66 ms | 176.6 | 171.3 | 10.69 | 65 ms | 42.9 GB |
| X = + REF2_EPREP + REF2_GDN_TG + R57_DQ1 | 63.82 ms (+0.16 ms, +0.26 %) | 169.8 | 163.4 | 10.30 | 63 ms | 42.9 GB |

- **Verdict (NewIdeas' exact-safe line): BETWEEN.** BITS PASS (43 / 43 verify-logits hashes, 3 / 3 tokens; gdn_tg engaged 3,348 calls
  in the exact arm; eprep is a QOM one-call path, so it engages only in FAST). Engaged in X only: gdn_tg 29,376, eprep 24,192 calls.
- Leap priced -0.5..-0.7 ms per check; the engine shows +0.16 ms: no cycle gain in v1' + wired. The whole decode / full period
  differences (-3.8 % / -4.6 %) are acceptance (kept 10.30 vs 10.69), QOM's pass-to-pass noise; the cycle is the judge.

### Sat 15:32-15:39: xsafe p4s = Leap's GPU-idle seams (a-e) + BigMoE #4 on v1' + wired: PASS, cycle -1.31 % (snapshot 3c962ca)
Seams located by Leap (TABLE_350 7fee5b8: a blocking eval / readback drains the queue; ~2.5 ms of GPU idle per cycle). X = v1' + wired +
BIGMOE_PAUSE_EVERY_MS=50 (a, bigmoe/gated.py) + R57_S0CARRY=1 (c, screen; 56f011f) + REF2_SB_FIRST=1 (b) + REF2_PLE_LATE=1 (d) +
REF2_COMMIT_SUB=9 (e) (Leap's ref2 1f85c231) + R57_DQ1=1 (#4). scripts/xsafe_ab.sh; data data/xs_p4s, b6cd335.

| | median cycle | whole decode | full period | kept / check | check 0 | peak |
|---|---|---|---|---|---|---|
| B = v1' + wired | 63.88 ms | 175.4 | 169.8 | 10.77 | 68 ms | 42.9 GB |
| **X = + seams a-e + DQ1** | **63.05 ms (-0.84 ms, -1.31 %)** | **178.8 (+2.0 %)** | 173.2 (+2.0 %) | 10.67 | 64 ms | 42.9 GB |

- **Verdict (NewIdeas' exact-safe line): PASS** (cycle <= -1 % AND bits). BITS PASS: 43 / 43 verify-logits hashes, 3 / 3 tokens; engaged in
  the exact arm sb_first 27, ple_late 27, commit_sub 104; in X (both passes) sb_first 481, ple_late 481, commit_sub 912. -> the
  Sunday-night exact-safe bundle (no new gate; NewIdeas 15:3x).
- **Per-check read-out (scripts/xs_strat.py, information; Leap 15:4x's method), X - B medians, checks >= 2:**

| class (share of checks) | p1a cycle | p1a check + commit | p4s cycle | p4s check + commit |
|---|---|---|---|---|
| full, misses <= 100 (~50 %) | -0.95 | -1.10 | -0.57 | -0.64 |
| full, misses > 100 (~7.5 %) | -0.07 | -0.65 | **-4.70** | -5.70 |
| partial, misses <= 100 (~41 %) | +0.10 | -0.13 | **-1.70** | -1.57 |
| partial, misses > 100 (1.4 %) | -0.45 | -0.61 | -4.16 | -4.41 |
| class-weighted cycle | -0.45 ms (-0.73 %) | | **-1.40 ms (-2.22 %)** | |

  p1a's flat partial class (Leap's question) is now -1.70 ms: (d) / (e) act on partials as designed. The high-miss checks gain most
  (-4.7 ms): with the first submit earlier and the stats / PAUSE reads off the path, the GPU starts before the demand reads land.
  The read-out's p1a partial row equals Leap's own (n 172 / 184, +0.10, -0.13).

### Sat 17:10-17:23: E5, the 45 GB EXACT number at C 256 (scripts/e5_exact.sh, data/e5; quiet; snapshot f451b54): EXACT PROOF PASS
E = the EXACT default: R57_PRESET=45 on the converted store + NAX6-N (now the preset's default for exact runs, 2a88770) + REF2_KEEP_CACHES=1
+ R57_WIRED=1. EB = E + the Sunday exact-safe bundle (seams a-e + DQ1) = the E5' arm. E / EB / EB / E on S5's 12 HumanEval, then MBPP12.

- **EXACT PROOF: PASS:** 96 / 96 runs (24 prompts x 4 passes, E and EB) token-identical to NAX6-N's one-token AR (EvalSuite results/ref2tnax
  for HumanEval, ref2tnaxn.r2 for MBPP; first min(n, AR length) tokens); prompt lengths match on all 24.

| set | arm | whole decode tok/s (passes) | full period | end to end (prefill median) | check 0 | cycle | kept / check | lifetime peak |
|---|---|---|---|---|---|---|---|---|
| 12 HumanEval | E | **99.4** (100.0, 98.6) | 99.7 | 30.8 (4.52 s) | 132 ms | 109.0 ms | 10.71 | 42.2 GB |
| 12 HumanEval | EB | **102.7** (102.3, 103.2) | 103.1 | 31.6 (4.39 s) | 129 ms | 104.6 ms | 10.71 | 42.2 GB |
| MBPP12 | E | **71.3** (69.9, 72.4) | 69.2 | 20.3 (3.68 s) | 107 ms | 82.6 ms | 5.93 | 41.9 GB |
| MBPP12 | EB | **73.3** (71.7, 74.7) | 71.1 | 20.3 (3.92 s) | 94 ms | 81.0 ms | 5.93 | 41.9 GB |

- EB vs E: whole decode +3.4 % (HumanEval), +2.7 % (MBPP12); median cycle -2.68 % / -3.80 %. The bundle carries to the exact mode.
- The previous exact number (NATIVE-FRI, 49.9 tok/s at C 192 on the steady metric, before KEEP / wired / C 256) is superseded.
- Exact prefill is slow: 3.7-4.5 s median per prompt, so end to end is ~31 (HumanEval) / ~20 (MBPP) tok/s. Prefill is the next exact lever.

### Sat 17:24-17:36: S5' = S5's config + the Sunday exact-safe bundle (scripts/s5p.sh, data/s5p; quiet; snapshot 6db4c24): CLEAN
B = v1' + wired (S5's config); X = B + seams a-e + DQ1 (p4s PASS, bits 43/43; S4 / CERT carry). C 256, B / X / X / B per set.

| set | arm | whole decode tok/s (passes) | full period | end to end | check 0 | cycle | kept / check | tokens / task | lifetime peak |
|---|---|---|---|---|---|---|---|---|---|
| 12 HumanEval | B | 174.4 (168.6, 179.6) | 168.8 | 34.8 | 65 ms | 63.5 ms | 10.42 | 246 | 42.9 GB |
| 12 HumanEval | **X = S5'** | **181.4** (181.7, 180.9) | **176.2** | 35.4 | 64 ms | 63.1 ms | 10.83 | 243 | 42.9 GB |
| MBPP12 | B | 120.2 (118.3, 121.9) | 117.4 | 23.6 | 59 ms | 53.5 ms | 6.34 | 120 | 42.6 GB |
| MBPP12 | **X = S5'** | **123.6** (122.5, 124.7) | **120.8** | 23.4 | 58 ms | 51.9 ms | 6.38 | 116 | 42.6 GB |

- X vs B: whole decode +4.0 % (HumanEval; cycle -0.16 %, acceptance +3.9 % this run), +2.9 % (MBPP12; cycle -2.50 %). 0 live modules.
- data/stack/s5p_env.txt = the S5' config line (the 45 GB lossy table cell; S4 / CERT carry the bundle by its bits).
- **End to end is prefill-bound:** ~35 (HumanEval) / ~23 (MBPP) tok/s, prefill ~5 s per prompt in both modes (E5: 3.7-4.5 s). A short
  prompt's prefill touches nearly all 512 experts per layer and the pools hold 256, so prefill streams ~the whole model from SSD
  (~68 GB at ~12 GB/s). The next big lever for time-to-answer is prefill, not decode.

### Sat 17:49-17:58: E24, the 24 GB EXACT number (scripts/e24.sh, data/e24; quiet): 41.3 / 41.7 tok/s, 23.86 GB
W1B's 24 GB stack (s10b_trace.py, stock kernels, v3 tiers 4433333333 / 3c + COMBINE + OOO, GDN_EXACT, MAXK 80 + DUPROW chunking, C 128,
M-01, JOINT, MTP stream, FR-Spec + Scout; Wed's ext/build_m01, the binary W1B ran) + BIGMOE_WIRED=1 (mx.set_wired_limit: the only one
of this week's exact-safe fixes that applies; KEEP / NAX / S2 / the seams are r57 / ref2-only). r57 cannot go below C ~160 exact
(16-row checks need up to 160 experts resident per layer call). 1-prompt pre-flight clean (wired engaged), then two passes.

| set | whole decode tok/s (passes) | end to end | SSD MB per token | tokens / task |
|---|---|---|---|---|
| 12 HumanEval | **41.3** (41.8, 40.8) | 26.5 | 355.1 | 256 |
| MBPP12 | **41.7** (42.5, 41.0) | 27.7 | 308.1 | 256 |

- **Lifetime peak 23.86 GB (<= 24.0).** Exact status (NewIdeas): exact vs its own stock-kernel AR (proven Wed, w24 == w24t); tonight
  pass 1 == pass 2 on 24 / 24 prompts. Agreement with NAX6-N's AR (sha over the run's n tokens, information): 6 / 24; stock kernels
  differ from NAX6-N's bitwise, and s10b's generate runs to N 256 without stopping at EOS, so MBPP rows (AR ends ~115) cannot match.
- s10b records no per-check clock: full period n/a. Its N 256 without an EOS stop doubles MBPP's tokens per task vs r57 (~120):
  the decode rate is comparable, the per-task work is not.
- vs Wed's W1B (33.2 tok/s on 2 HumanEval + 4 math, no wired limit): different prompts; the wired limit is the only change in the stack.

### Sat 17:59-18:00: F24 (24 GB lossy on r57): NOT MEASURABLE at the 24 GB budget tonight (engine limit)
scripts/f24.sh probed r57 v1' + wired + the bundle (S5' snapshot 6db4c24, R57_VTIER off) at C 112 and C 104 on 12 HumanEval. Both arms
stopped within ~10 s on the FIRST verify check of the first prompt: ref2's REF2-T tier path asserts ("more experts than base slots", 120
unique experts in one layer call > 110 base slots at C 112; ref2.py:1490 via _sync_slots). That check runs ungated (FG off: no draft
confidence yet), i.e. exact, and a 16-row exact call can need up to 160 experts resident at once. So r57 needs C >= ~160 (~30 GB) even in
FAST mode; QOM's smaller demand set does not help the checks that fall back to the exact path. -> The 24 GB lossy cell is n/a tonight.
Routes (not tonight): chunk a layer call whose expert union exceeds C into row groups in ref2's tier path (as the 24 GB stack's DUPROW
chunking does), or port QOM's FAST rules into the s10b 24 GB stack. (f24.sh's probe line said "aborts 0": it counted only "no free slot";
the arms' rc=1 is in data/f24/c45.log.)

### Sat 18:04-18:12: C30, the ~30 GB lossy CONTEXT point (scripts/c30.sh, data/c30; quiet; S5' snapshot 6db4c24): 153.5 / 114.8 tok/s, 31.8 GB
NewIdeas 18:1x: S5''s config (v1' + wired + the Sunday bundle) at r57's smallest workable C, 160 (a 16-row exact check can need 160 experts
resident), R57_VTIER off. Context for Kamen's 24 GB question, NOT a table cell; quality basis = runner 49's stackb gate at C 160 / 192
(PASS vs NAX6). A C 160 probe pass (12 / 12, no abort), then 12 HumanEval x 2 and MBPP12 x 2; 0 live outside modules.

| set | whole decode tok/s (passes) | full period | end to end | check 0 | cycle | kept / check | SSD MB per kept token | lifetime peak |
|---|---|---|---|---|---|---|---|---|
| 12 HumanEval | **153.5** (152.3, 154.7) | 149.1 | 34.9 | 74 ms | 70.0 ms | 9.77 | 18.1 | **31.80 GB** |
| MBPP12 | **114.8** (116.6, 112.9) | 111.9 | 25.0 | 60 ms | 53.6 ms | 6.08 | 15.9 | 31.40 GB |

- vs S5' at C 256 (181.4 / 123.6, 42.9 GB): -15.4 % / -7.1 % whole decode for -11.1 GB of lifetime peak.
- CORRECTION: commit ef64e24's message ("C30 (24 GB lossy, C 160): CLEAN, f24_env.txt written") is text inherited from f24.sh by the sed
  copy: C30 is the ~31.8 GB context point and its env-file block is disabled (no data/stack/f24_env.txt exists). Its log / report keep
  f24's file names: data/c30/f24.log, data/c30/f24_report.txt.

### Sat 21:00-21:04: E24m, E24's MBPP12 with the EOS stop (scripts/e24m.sh, data/e24m; quiet): 38.4 tok/s, comparable now
E24's stack (W1B's s10b 24 GB + wired, Wed's build_m01) with S10B_EOS=1 (spec125's stop_ids = r57's EOS set) and S10B_SAVE_IDS=1.
- **MBPP12: WHOLE DECODE 38.4 tok/s** (passes 38.2, 38.6); END TO END 18.6; SSD MB per token 494.0; tokens / task 115 (r57 rows: 115-123,
  so the cell is comparable now; E24's 41.7 was mostly post-answer text). Lifetime peak 23.52 GB (<= 24.0).
- pass 1 == pass 2 on 12 / 12 (ids); agreement with NAX6-N's AR 9 / 12 (ids over min(n, AR); information: stock kernels differ bitwise).
- **EvalSuite 20:47: C30's code-only gate PASS** (S5' at C 160, VTIER off, s5p 6db4c24): HE+ vs NAX6-N fwd 88.3 vs 91.7 (-3.33, p 0.63),
  rev 91.7 (+0.00); long code 21 rows, 0 blunders (RESULT_c30, EvalSuite). So the ~32 GB lossy point is gated on code.

### Sat 21:04-21:22: F24k, the 24 GB LOSSY number on r57 with Leap's row-chunking (scripts/f24k.sh, data/f24k; quiet; snapshot 4caa5a3)
Leap's REF2_TIER_CHUNK=1 (ref2 c176a5ad, built in ~25 min): a tier call whose expert union exceeds the pool is split into contiguous row
groups that fit, each through the unchanged body (bitwise per Leap, 43/43). F = S5''s config + R57_VTIER off + REF2_TIER_CHUNK=1.
- **Probes (12 HumanEval, CAP 256):** C 100 -> 12 / 12, lifetime peak 23.7 GB, no abort; C 96 -> 12 / 12, 23.2 GB. C24 = 100.
- **Exact check of the chunked path at C 100 (the official proof): PASS:** NAX6-N exact env + REF2_TIER_CHUNK=1, 3 HumanEval, CAP 64:
  3 / 3 prompts token-identical to NAX6-N's one-token AR; 88 chunked calls (176 groups). (The script's check line had an f-string
  syntax error, so its first run stopped before the passes; the result was evaluated from the saved tokens and the passes re-run on
  the same committed snapshot, 37f0d36.)

| set | whole decode tok/s (passes) | full period | end to end | check 0 | median cycle | kept / check | SSD MB per kept token | lifetime peak |
|---|---|---|---|---|---|---|---|---|
| 12 HumanEval | **27.7** (27.5, 27.9) | 28.0 | 17.0 | 520 ms | 484.2 ms | 11.07 | 231.4 | **23.70 GB** |
| MBPP12 | **35.8** (34.5, 36.8) | 36.4 | 16.4 | 435 ms | 70.6 ms | 6.04 | 162.4 | 23.40 GB |

- CLEAN (0 live modules, no abort, peak <= 24.0). data/stack/f24_env.txt (96a87cf) -> EvalSuite's code-only gate (Sat 100, MEM 27 GB).
- **Slower than E24's exact s10b stack** (41.3 HE / 38.4 MBPP with the EOS stop), as Leap predicted: at C 100 every 16-row HumanEval
  check (160 ids > 98 slots) takes the synchronous chunked tier path (~480 ms per check, SSD reads not overlapped). MBPP's drafts are
  shorter (truncation), so most of its checks still fit the gate path (median cycle 70.6 ms). At 24 GB the exact s10b stack is the
  faster engine; r57's lossy rules need C >= ~160 to pay off (C30: 153.5 at 31.8 GB).

### Sat 22:05-23:25: N45, the paper's missing 45 GB measurements (scripts/n45.sh, data/n45; quiet; S5' snapshot 6db4c24, C 256)
EB = the EXACT default (R57_PRESET=45 + NAX6-N) + KEEP + wired + the Sunday bundle; L = S5''s lossy config (v1' + wired + bundle; QOM on
humaneval / mbpp only). Prompts: the first 12 IFEval and 12 math keys with NAX6-N AR rows (EvalSuite results/ref2tnax), CAP 256,
EB / L / L / EB per set; prefill: agent/edit10 (1565 tokens), agent/write13 (4730), agent/write11 (7932), CAP 16, two passes per config.

| set | arm | whole decode tok/s (passes) | full period | end to end | tokens / task | lifetime peak | == NAX6-N AR |
|---|---|---|---|---|---|---|---|
| 12 IFEval | EB | **39.9** (39.9, 39.8) | 41.6 | 17.1 | 142 | 41.7 GB | **24 / 24 (EXACT PROOF PASS)** |
| 12 IFEval | L | **45.0** (45.2, 44.8) | 44.9 | 17.3 | 144 | 42.4 GB | 6 / 24 |
| 12 math | EB | **78.8** (79.4, 78.3) | 81.0 | 39.5 | 256 | 42.5 GB | **24 / 24 (EXACT PROOF PASS)** |
| 12 math | L | **87.9** (85.3, 90.4) | 90.5 | 40.3 | 256 | 43.1 GB | 7 / 24 |

| prefill (TTFT = prefill + cache / drafter setup + the first token) | P | EXACT TTFT (passes) | exact prefill tok/s | LOSSY TTFT | lossy prefill tok/s |
|---|---|---|---|---|---|
| agent/edit10 | 1565 | 15.90 s (15.92, 15.88) | **98** | 16.28 s | 96 |
| agent/write13 | 4730 | 40.11 s (40.70, 39.51) | **118** | 41.23 s | 115 |
| agent/write11 | 7932 | 64.07 s (63.78, 64.36) | **124** | 66.57 s | 119 |

Lifetime peaks over the prefill arms: exact 44.5 GB, **lossy 45.1 GB (> 45.0 on the 7.9K prompt)**.
- **FINDING 1 (claims): v1''s lossy config is NOT exact on chat / math.** v1' runs those sets with exact acceptance (no QOM rule), yet only
  6 / 24 (IFEval) and 7 / 24 (math) runs equal NAX6-N's AR: a non-FAST v1' lever changes their outputs (candidates: R57_FASTTAX, REF2_HC8_FAST,
  the S2 session-vocab head R57_SVH / DSUB). Their quality basis is S4's same-scores PASS vs NAX6-N (math / IFEval), so they are "lossy,
  same scores", not exact. Not tested which lever.
- **FINDING 2: the 45 GB lossy config breaches 45.0 GB on a 7.9K-token prompt** (45.1 GB lifetime); code / chat / math prompts stay
  at 42.4-43.1 GB; exact stays at 44.5 GB on 7.9K.
- **FINDING 3: prefill is the weak spot:** ~100-124 tok/s (exact and lossy alike: the prefill path streams experts from SSD per 1024-row
  chunk). For comparison, Strata (Q2_0, ~40 GB RSS, whole model in RAM) reports ~226 tok/s at 4K on an M1 Max and ~1430 on an M5 Max.
- Chat decode is slow vs code (IFEval 39.9 exact / 45.0 lossy vs HumanEval 99.4 / 181.4): HumanEval flatters (copying, long accepts).

### Sun 00:10-00:29: CHS1, the CHAT150 strike's first chat A/Bs (scripts/chs.sh, data/chs; quiet; snapshot 94f0aad)
> (a quote from the project owner, kept private)
12 IFEval (N45's keys), C 256, one pass per arm. L = S5''s lossy config (data/stack/s5p_env.txt); E = exact + KEEP + wired + bundle.
Report: data/chs/chs1_report.txt. "Whole decode" is pooled over the 12 runs (n / decode wall).

| arm | config | whole decode | kept / check | ck + drf ms | misses / svc ms | peak GB | == AR |
|---|---|---|---|---|---|---|---|
| c0 / c0b | L (first / last) | 41.5 / 42.7 | 2.47 / 2.45 | 50.2+8.9 / 48.2+8.7 | 90.7 / 16.4 | 42.4 | 3 / 2 of 12 |
| q1 | L + QOM_SETS incl. ifeval | 42.1 | 2.48 | 49.2+9.0 | 89.7 / 15.5 | 42.4 | 4 |
| fa1 | L - FASTTAX | 42.3 | 2.52 | 50.2+8.9 | 88.3 / 15.3 | 42.4 | 3 |
| fa2 | L - HC8 (R57_HC8 + REF2_HC8_FAST) | 40.3 | 2.46 | 52.4+8.5 | 86.5 / 15.6 | 43.6 | 4 |
| fa3 | L - SVH - DSUB | 42.4 | 2.58 | 51.0+8.7 | 96.2 / 16.8 | 42.1 | 3 |
| fa4 | L - DN_KS | 39.4 | 2.47 | 54.3+8.2 | 83.2 / 14.2 | 42.4 | 3 |
| fa5 | L - RELAX_SPAN (+ rule) | 42.9 | 2.47 | 48.8+8.4 | 88.8 / 15.9 | 42.4 | 3 |
| fa6 | L - all six | 36.7 | 2.60 | 61.0+8.7 | 92.3 / 15.6 | 43.3 | 4 |
| e0 | E (exact) | 36.4 | 2.53 | 59.6+8.7 | 89.8 / 15.3 | 41.7 | 12 |
| r1 | L + R1 on chat (R57_CHAT_RULE=R1) | 45.1 | 2.57 | 48.1+8.3 | 92.1 / 15.4 | 42.4 | 2 (62 relaxed accepts) |
| qr1 | r1 + QOM_SETS incl. ifeval | 44.6 | 2.53 | 47.7+8.5 | 89.0 / 14.9 | 42.4 | 2 (63) |

- **FINDING 1: QOM never engaged on chat (q1: qom_skip = qom_opt = 0, misses unchanged).** The FAST row gate (QOM / U3 sure rows)
  ran only in the DFlash2 drafting branch, and chat drafts with the MTP head. Fixed in ce94a45 with R57_QOM_MTP=1|2: the gate on the
  MTP head's drafts, with the bonus row exact (1, Breakthrough's M1-A) or gated (2, M1-B). It confirms Breakthrough's read:
  L's chat reads as many MB per kept token as E (41.8 vs 40.2). chs2 measures it.
- **FINDING 2: L's +15.6 % over E on chat (42.1 mean vs 36.4) comes from HC8 and DN_KS.** Dropping HC8 costs -4 % and DN_KS -6 %.
  FASTTAX, SVH + DSUB and RELAX_SPAN do nothing measurable on chat (RELAX_SPAN made 0 relaxed accepts on IFEval). Dropping all
  six gives E's speed (36.7). Note: dropping REF2_HC8_FAST alone (FU's old fa2) is a no-op, because R57_HC8=1 sets ref2's hc8_fast.
- **FINDING 3 (C4): L's chat non-exactness is NOT in the six known-lossy levers.** No single drop restores chat == AR (2-4 of 12),
  and neither does dropping all six (fa6: 4 / 12, first diffs at tokens 7-78). fa6 runs at E's speed, so the breaking lever adds no
  chat speed. Leap: L's chat is run-to-run nondeterministic (c0 3 / 12 vs c0b 2 / 12 fits). chs3 runs a forward bisect from E
  (+ OOO_1C + NAX_1CQ twice, + FAST=QOM + build_qom, + SMALL_NATIVE, + GDN_LAZY, + DNR, + HCB, + SBN 2, + COPY g2, + D7B).
- **K1: R1 on chat = +7.1 %** (45.1 vs 42.1): kept 2.57 vs 2.46, 62 relaxed accepts over 12 prompts. This is a speed reading only:
  with HC8 on, it is arm E8, which failed IFEval (84 vs 90) on 11-05. A gated K1 needs HC8 off on chat (per-set HC8 is not built).
- Noise between the two L runs: 41.5 vs 42.7 (+-1.5 %). Lifetime peaks 41.7-43.6 GB (line 45.0).

### Sun 00:30-00:49: CHS2, M1 / D1 / lever 2 on chat (scripts/chsb.sh, data/chs; quiet; snapshot 09ef078, ref2 388ca3f1)
12 IFEval at C 256, one pass per arm. Report data/chs/chs2_report.txt; per-check read-out scripts/chs_strat.py (median check ms, ck by rows).
L's chat whole decode varies run to run (41.5 / 42.7 / 44.8 / 43.7 over four L runs tonight: different outputs, different kept),
so the per-check columns are the reliable read. L baseline = mean of x2c0 / x2c0b.

| arm | config (on L unless E) | whole decode | kept | ck median / mean ms | drf ms | misses / check | svc ms | MB / check | peak GB |
|---|---|---|---|---|---|---|---|---|---|
| x2c0 / x2c0b | L | 44.8 / 43.7 | 2.62 / 2.50 | 44.0 / 48.9, 47.8 | 9.1 / 8.8 | 97.8 / 91.1 | 15.8 / 14.9 | 109.5 / - | 42.4 |
| x2qm1 | QOM on MTP chat drafts, bonus row exact (M1-A) | 45.3 | 2.46 | 42.5 / 45.2 | 8.7 | 42.8 | 10.1 | 56.2 | 42.4 |
| x2qm1s | M1-A + pool shift S16 | 45.8 | 2.41 | 41.8 / 44.0 | 8.4 | 38.3 | 9.2 | 52.1 | 42.4 |
| **x2qm2** | **QOM on MTP chat drafts, bonus row gated (M1-B)** | **49.0** | 2.41 | **37.4 / 40.5** | 8.3 | **25.8** | **6.3** | **37.4** | 42.4 |
| x2qm2s | M1-B + pool shift S32 | 52.0 | 2.72 | 39.3 / 42.4 | 9.3 | 28.2 | 7.9 | 40.9 | 42.3 |
| x2rc8 | ranks >= 8 dropped on chat checks | 45.1 | 2.48 | 42.4 / 46.1 | 8.6 | 76.5 | 12.9 | 85.6 | 42.4 |
| x2p4 | D1: MTP head Linears at 4-bit | 38.4 | 2.55 | 44.6 / 49.6 | **16.2** | 95.9 | 15.9 | 108.6 | 42.3 |
| x2l2c | look-ahead on (gate+t+pf), lever 2's control | 42.4 | 2.49 | 44.7 / 49.5 | 8.5 | 88.6 | 14.8 | 101.0 | 42.4 |
| x2l2f | lever 2 (forecast reads FC_F 4 / FS 2) | 40.7 | 2.53 | 49.9 / 52.7 | 8.7 | 92.2 | 15.0 | 463 | **48.8** |
| x2e / x2ep4 | E / E + D1 | 36.6 / 36.9 | 2.53 / 2.57 | - / 59.4, 59.9 | 8.7 / 8.4 | 89.8 / 94.0 | 15.1 / 15.5 | - | 41.7 / 41.6 |

- **M1 WINNER: M1-B (x2qm2, R57_QOM_SETS += ifeval, R57_QOM_MTP=2): +10.7 % whole decode (49.0 vs 44.3), check median 37.4 vs
  44.0 ms (-15 %), misses per check 25.8 vs ~94 (-73 %), MB per check -66 %, svc 6.3 vs ~15.4 ms.** Every row count gains (1-2 rows
  39.8 vs 42.5 ms, 3 rows 36.9 vs 45.6, 4 rows 39.9 vs 50.2, 5+ rows 49.0 vs 68.9). Lossy: chat == AR 2 / 12 (L's own is 2-3 / 12);
  EvalSuite's IFEval gate decides. M1-A (bonus row exact) helps only the larger checks (+1-2 %).
- Breakthrough's pool shift does not pay in the engine: its check medians are worse with it (S16 41.8 vs 42.5 for M1-A, S32 39.3
  vs 37.4 for M1-B). x2qm2s's 52.0 whole decode is kept noise (2.72 vs 2.41).
- **D1 KILL as implemented:** on L the 4-bit MTP head doubles the draft (16.2 vs 9.1 ms per check; 6.3 vs 4.0 ms per drafted row);
  on E it is neutral and exact (8.4 vs 8.7 ms, 12 / 12 == AR). So it interacts with an L lever (a guess: ref2's hc8_fast read on the
  head's own GatedResidual mixers, which nn.quantize turned 4-bit).
- **Lever 2 KILL on chat:** 4x the SSD MB per check (150,268 forecast reads, 18,327 used), check +5 ms, lifetime peak 48.8 GB.
  The look-ahead alone (x2l2c) costs ~+0.7 ms per check; L decodes without it (arm gate+t).
- Rank cut 8 on chat: misses -22 %, check -1.6 ms median, +2 % whole decode; Breakthrough's map failed TOP8 on multi.

### Sun 00:54-00:56: PF-A smoke, layer-major prefill (scripts/pfa_smoke.sh, data/pfa; quiet; snapshot 37f2114): WORKS, EXACT, x1.98
Exact E, agent/edit10 (1,565 tokens = 2 chunks of <= 1024), CAP 16, LOGHASH. scripts/pflm.py wraps tier.prefill_call (the 45 GB path).
| arm | prefill | TTFT | prefill tok/s | prefill SSD | tokens / logits hashes vs pa0 | lifetime peak |
|---|---|---|---|---|---|---|
| pa0 | chunk-major (today, no-logits) | 15.52 s | 101 | 109.4 GB | - | 43.4 GB |
| pa1 | R57_PFLM=1: layer-major, full-E scratch, original ids | **7.84 s** | **200** | 63.6 GB | == / 2 of 2 bitwise | 43.3 GB |
| pa2 | R57_PFLM=2: compact copy per call | 11.21 s | 140 | 63.6 GB | == / 2 of 2 bitwise | 44.9 GB |
- Each layer's experts are read once per prompt (22,845 reads + 16,210 reuses over 96 calls) instead of once per chunk. On 2 chunks that
  is -42 % SSD and x1.98 TTFT. Longer prompts should gain more (8 chunks for 7.9K); PFX (after CERT, chained after CHS4) measures them.
- PFLM=1 is bitwise on these checks despite the full-E weight array (the gather kernel at 1,024- and 541-row chunks); PFLM=2 costs
  +1.6 GB of peak and time for nothing.
- Next: overlap the reads with compute (prefetch layer L+1 into a second scratch, +1.6 GB) and PF-D (pool-resident MSB reuse).

### Sun 00:57-01:14: CHS3, C4's forward bisect from exact E (scripts/chsb.sh, data/chs; quiet; snapshot 5d92dcc): TWO "EXACT" LEVERS BREAK CHAT
12 IFEval at C 256, E + one of L's remaining levers per arm (chs1: none of the six known-lossy levers explains L's chat != AR).
| arm | E + | whole decode | check median ms | == AR (first diffs) |
|---|---|---|---|---|
| x3e | (nothing) | 37.0 | 54.6 | 12 / 12 |
| **x3oc / x3oc2** | **REF2_OOO_1C + REF2_NAX_1CQ** (two runs) | 37.4 / 36.2 | 54.1 / 53.6 | **3 / 12, 4 / 12** (diffs differ between the runs: 62,4,44,... vs 62,22,44,...) |
| **x3sn** | **REF2_SMALL_NATIVE** | 36.5 | 58.1 | **3 / 12** (78, 34, 44, ...) |
| x3q | R57_FAST=QOM + build_qom + D17 (QOM off on chat) | 36.8 | 55.2 | 12 / 12 |
| x3gl / x3dnr / x3hcb | REF2_GDN_LAZY / R57_DNR / R57_HCB | 36.8 / 35.6 / 36.6 | 55.1 / 57.0 / 54.6 | 12 / 12 each |
| x3sbn / x3cp / x3d7 | R57_SBN=2 / R57_COPY=g2 / D7B + D7BNAX (+32) | 37.9 / 36.7 / 36.9 | 53.1 / 54.8 / 54.7 | 12 / 12 each |
- **C4 ANSWER:** L's chat gain (+15 %) is HC8 + DN_KS, both lossy, so the IFEval gate decides them. L's chat NON-EXACTNESS comes from two
  levers that were believed exact: **the one-call OOO (REF2_OOO_1C + NAX_1CQ), which is run-to-run NONDETERMINISTIC** (the two runs
  diverge at different tokens), and **REF2_SMALL_NATIVE**. Neither speeds up chat (37.4 / 36.5 vs 37.0). M1-B's QOM needs the one-call on
  the NAX path, so OOO_1C stays in the lossy chat config; its exactness / determinism is a bug for Leap (owner).

## Sun 8 Nov 07:13-07:42: CHS4, the CHAT150 lever A/Bs on M1-B + C4's coherent-read test + SB1 (data/chs/chs4_report.txt)

12 IFEval prompts (N45's keys) at C 256, one process per arm under a quiet lock, snapshot 342f914. L = S5' (lossy), M1-B = L + QOM on
the MTP head's chat drafts (R57_QOM_SETS += ifeval, multi; R57_QOM_MTP=2), E = exact. L's chat is run-to-run nondeterministic (whole
decode moves about 4 %, kept about 0.1), so the per-check medians (scripts/chs_strat.py) are the steadier read. Percentages are against the
mean of the two M1-B runs (49.9 tok/s) unless marked.

| arm | config | whole decode tok/s | kept / check | check median ms | cycle median ms | 1-row check median ms | misses / check | == AR |
|---|---|---|---|---|---|---|---|---|
| y0 / y0b | L (two runs) | 43.8 / 44.5 | 2.54 / 2.54 | 43.3 / 43.1 | 50.4 / 50.2 | 42.0 / 42.7 | 93.6 / 90.4 | 3 / 12 |
| yb / ybb | M1-B (two runs) | 49.3 / 50.5 (+13.0 % vs L) | 2.41 / 2.47 | 37.4 / 37.8 | 45.9 / 46.0 | 48.0 / 45.3 | 25.4 / 23.8 | 3 / 12 |
| **ybs1** | M1-B + R57_SB1=1 (exact, schedule only) | **52.0 (+4.2 %)** | 2.42 | **35.4** | **42.9** | **35.4** | 25.3 | 2 / 12 |
| yblg | M1-B + Leap's exact-safe bundle (K1M) + REF2_HC8_G=2 | 53.1 (+6.4 %) | 2.58 | 37.0 | 44.2 | 43.4 | 28.2 | 3 / 12 |
| ybr | M1-B + R57_CHAT_RULE=R1 (73 relaxed accepts) | 52.9 (+6.0 %) | 2.61 | 37.7 | 45.6 | 46.7 | 28.4 | 2 / 12 |
| ybrc | M1-B + rank cut 8 on IFEval (R57_RCUT_SETS=ifeval:8) | 52.2 (+4.6 %) | 2.49 | 36.4 | 44.7 | 45.6 | 25.1 | 2 / 12 |
| ybl / ybn / ybln / yblf | M1-B + bundle / NW / bundle + NW / bundle + Leap's FAST low-row | 49.6 / 48.9 / 50.6 / 50.4 | 2.47-2.63 | (mean 40.8 / 41.8 / 39.2 / 40.1 vs 40.2) | | | 28-30 | 1-3 / 12 |
| y0l | L + bundle | 44.8 (+1.5 % vs L) | 2.48 | (mean 46.5 vs 48.3) | | | 88.5 | 3 / 12 |
| y0p4 | L + R57_MTP_P4 (D1 v1b, 4-bit MTP head) | 40.2 (-9.0 % vs L) | 2.57 | (draft 14.9 vs 8.8 ms) | | | 95.9 | 3 / 12 |
| ycoc / ycoc2 | E + REF2_OOO_1C + NAX_1CQ (two runs) | 37.7 / 38.0 | 2.54 / 2.57 | | | | 91 | **3 / 12, 3 / 12** |
| **ycoh / ycoh2** | the same + REF2_NAX_COH=1 (coherent pool reads) | 37.8 / 37.9 | 2.53 / 2.53 | 53.0 | 59.9 | 52.0 | 89.8 | **12 / 12, 12 / 12** |
| yes1 | E + R57_SB1=1 | 38.3 | 2.53 | 53.0 | 60.3 | 45.7 | 89.8 | 12 / 12 |

- **C4 SOLVED (Leap's hypothesis b, stale slot lines):** without COH the one-call diverges from AR at 9 of 12 prompts, and its two runs
  diverge from each other at different tokens (first diffs 9,20,60,.. vs 9,62,46,..). With REF2_NAX_COH=1 both runs match AR on all 12 and
  each other token for token, at no measurable cost (37.85 vs 37.85 tok/s). After this, L's chat non-exactness comes only from
  SMALL_NATIVE (FAST by design) and the lossy HC8 / DN_KS.
- **SB1 PASS** (NewIdeas' pre-registered 1ROW test): +4.2 % >= +3 %; 1-row check median 46.7 -> 35.4 ms (<= 40); yes1 == AR 12 / 12.
  The GPU tail per check drops from 9.5-9.8 to 3.1 ms. Exact, so it can go into E as well (1-row median 45.7 vs 53.8 in CHS2).
- Lossy candidates for the IFEval gate: R1 (+6.0 %, all from kept), rank cut 8 (+4.6 %; check median -3 %), bundle + HC8_G (+6.4 % whole
  decode, but only -1.6 % on the check median; part of the gain is kept noise).
- Washes on chat: NW, the bundle alone, Leap's FAST low-row. **D1 v1b KILL:** the 4-bit MTP head drafts slower (14.9 vs 8.8 ms per check).
- The best arm is still about 53 tok/s, 2.8x short of 150. Next: CHS5 stacks M1-B + COH + SB1 + bundle/HC8_G + rank cut 8 (+- R1), with
  repeat runs and a determinism check of M1-B + COH.

## Sun 8 Nov 07:43-07:57: PFX, prefill by chunk size + PF-A layer-major on the 1.6K / 4.7K / 7.9K agent prompts (data/pfx/pfx_report.txt)

Exact E at C 256 (CAP 64 decode tokens), quiet, snapshot 94a6f02. TTFT = prefill + setup + first token; tokens / logits hash compare
with the 1024-chunk arm (pe1024, today's path). Prefill SSD = bytes read during the prefill.

| arm | 1.6K edit10 | 4.7K write13 | 7.9K write11 | exact vs pe1024 | prefill SSD (1.6K / 4.7K / 7.9K) | lifetime peak |
|---|---|---|---|---|---|---|
| pe1024 (today) | 15.2 s = 103 tok/s | 37.3 s = 127 | 61.9 s = 128 | (reference) | 109 / 265 / 430 GB | 44.50 GB |
| **plm1 (R57_PFLM=1: layer-major, full-E scratch, original ids)** | **8.0 s = 195 (x1.90)** | **12.9 s = 368 (x2.90)** | **18.0 s = 441 (x3.44)** | **tokens ==, logits hash 8/8, 28/28, 22/22** | **64 / 67 / 68 GB** | **45.10 GB** |
| plm2 (compact copy) | 11.3 s = 138 | 21.1 s = 224 | 31.0 s = 256 | tokens ==, hash all == | 64 / 67 / 68 GB | 45.90 GB |
| pe2048 / pe4096 / pe8192 (bigger chunks) | 161 / 161 / 159 | 171 / 225 / 312 | 201 / 290 / 389 | logits hash 0 / n; tokens differ at 4.7K-7.9K | 64 / 67 / 68 GB (8192) | 46.5 / 49.9 / 53.1 GB |
| pl8192 (lossy L, chunk 8192) | 153 | 314 | 392 | (lossy) | | 53.80 GB |

- **PF-A WORKS, EXACT:** layer-major prefill reads every expert once per prompt (68 GB at 7.9K vs 430 GB) and keeps the logits bitwise.
  Prefill 103 / 127 / 128 -> 195 / 368 / 441 tok/s. The lifetime peak is 45.10 GB, 0.1 GB over the 45.0 line: trim before adopting
  (a smaller C or dropping the scratch earlier).
- **Bound:** at 1.6K plm1 reads 64 GB in 8.0 s (8 GB/s, near the SSD); at 4.7K / 7.9K, 5.2 / 3.8 GB/s, so compute bounds the long
  prompts. That clears Leap's PF-B (faster prefill compute) and an SSD-compute overlap (PF-A double buffer) as the next steps toward 1000.
- **PF-D price:** at the start of each prompt's prefill, 8.7-13.9K of the 24.6K base expert-layers are already resident (36-57 %).
  Reusing them would cut the 1.6K prompt's 64 GB.
- Bigger chunks alone (NewIdeas' pre-registered verdict: BETWEEN, x2.46 at 4.7K) break bitwise logits and peak at 46.5-53.8 GB:
  superseded by PF-A.

## Sun 8 Nov 07:58-08:12: CHS5, the M1-B lever stack with coherent one-call reads (data/chs/chs5_report.txt)

12 IFEval prompts at C 256, one process per arm under a quiet lock, snapshot 65cd673. Every M1-B arm carries REF2_NAX_COH=1. Each row adds
one lever to the row above it.

| arm | config | whole decode tok/s | kept / check | check median ms | cycle median ms | 1-row / 2-row check median ms | tokens |
|---|---|---|---|---|---|---|---|
| z0 / z0b | M1-B + COH (two runs) | 49.7 / 49.9 | 2.43 | 37.7 / 37.4 | 46.2 / 45.8 | 45.8 / 32.8 | **z0 == z0b on 12 / 12** |
| zs | + R57_SB1=1 | 51.9 (+4.2 %) | 2.43 | 36.1 | 43.7 | 35.9 / 32.7 | == z0 on 12 / 12 (schedule only) |
| zsg | + Leap's bundle (K1M) + REF2_HC8_G=2 | 54.7 (+9.8 %) | 2.52 | 34.7 | 41.8 | | new numerics |
| zsgc / zsgcb | + rank cut 8 on IFEval (two runs) | 53.5 / 53.8 (+7.7 %) | 2.35 | 33.3 / 33.1 | 40.5 / 40.2 | 31.4 / 30.8 | **zsgc == zsgcb on 12 / 12** |
| zsgcn | zsgc + REF2_NGEMV_ROWS=2 | 55.4 (+11.2 %) | 2.44 | 33.4 | 40.5 | 31.9 / 30.0 | differs from zsgc on 8 / 12 |
| **zsgcr** | zsgc + R57_CHAT_RULE=R1 (83 relaxed accepts) | **56.0 (+12.4 %)** | 2.50 | 33.8 | 40.6 | | |
| yeh / yesh | E +- SB1, with R57_LOGHASH=1 | 36.5 / 36.8 | 2.53 | 59.4 / 58.8 | | | == AR 12 / 12 both |

- **M1-B + COH is DETERMINISTIC:** repeat runs give the same tokens on all 12 prompts (z0 / z0b, zsgc / zsgcb), and whole decode moves
  0.4-0.6 %. Before COH, L and M1-B moved about 4 % run to run. Chat A/Bs now read cleanly on whole decode.
- **SB1 is EXACT at the logits level:** E vs E + SB1 has the same verify-logits hash on all 676 checks (NewIdeas' hash clause: closed,
  SB1 PASS in full). On M1-B it keeps every token and saves 2 ms per check (1-row checks 45.8 -> 35.9 ms).
- The stack's check falls from 37.6 to 33.2 ms and its cycle from 46.0 to 40.4 ms (-12 %). rank cut 8 trims the check by 1.4 ms but
  costs kept (2.52 -> 2.35). NGEMV_ROWS=2 gives no gain at 1 row and -0.7 ms at 2 rows; its +3.5 % whole decode is new tokens
  (kept 2.44), not speed. R1 gets back kept (2.50) for the best arm, 56.0 tok/s: +27 % over L (44.2) but 2.7x short of 150.
- **What 150 needs:** at the stack's ~40 ms cycle, 150 tok/s takes 6 tokens kept per check (today 2.4-2.5). At 2.5 kept it takes a 17 ms cycle.
  The drafter's chat acceptance and the ~33 ms check are the two walls.
- **CORRECTION (Sun 08:2x, Leap):** MLX 0.32.2's sorted NAX gather (affine_gather_qmm_rhs_nax, quantized_nax.h:1547) computes
  short(M - (y_row + tm)). When a SwitchGLU call has more than 32,767 sorted rows (tokens x 10) and the count is not a multiple of 64,
  the short overflows and those output rows are never written (deterministic, not a race). PFX arms it hits: pe4096 on write11
  (second chunk 38,360 rows) and pe8192 / pe8192nl / pl8192 on write13 (47,300) and write11 (79,320). Their "tokens DIFF" there is partly
  this bug. Safe: every 1024- and 2048-chunk call (<= 20,480 rows: pe1024, plm1, plm2, exact E) and 4096-token chunks (40,960 = 640 x 64).
  The big-chunk TTFTs are therefore not valid prefill numbers on those prompts. Exact fix: pad the sorted rows to a multiple of 64
  (Leap's ~/Leap/scripts/pfb_fix.py switch_glu_padded, bitwise == the 1024-token calls). PF-A's numbers stand.

### Sun 8 Nov 09:13: EvalSuite's chat gate on M1-B + COH (their docs/RESULT_chat_m1b_2026-11-08.md, 70b5cf8): IFEval PASS, multi FAIL

- IFEval vs NAX6-N: -1.00 forward (p 1.0), -4.00 reverse (p 0.39): PASS. Multi (24 rows): 334 blunders per 10K (bar 1), 15.3 % deviations,
  worst zh 686 / ar 475 / ru 441 / bg 400 / ja 330 per 10K. M1-B is deterministic on one history but history-dependent (the two orders
  agree on 9 / 100 IFEval and 0 / 24 multi rows).
- So M1-B, CHS5's stack (chat2_env, which inherits QOM on multi) and the 56.0 tok/s figure hold for English chat only, pending the split
  gate chat3_env.txt (m1bi, 2187975). That gate puts QOM on humaneval, mbpp and ifeval but not multi, so multi runs S5' + COH: the base
  config's own multi baseline, never gated before. Either QOM causes the blunders, or S5''s lossy levers (HC8 / DN_KS / SMALL_NATIVE) do.
- PFX correction addendum: the MLX sorted-gather bug is BigMoE LEDGER BUG1 (PrefillKernels' align_probe.py; fixed in MLX 0.32.4). The
  tier's grouped prefill already pads to x64. Fast24 (F24) prefills in 1,024-token calls (<= 10,240 rows): safe.

## Sun 8 Nov 09:24-09:42: FU part B, the 45.1 GB agent-prompt peak on L + PF-A on L (data/fu/fu_report.txt)

L = S5' (lossy), the 3 agent prompts (1.6K / 4.7K / 7.9K) at CAP 16, then 12 HumanEval for the decode cost. The orchestrator's publish
job ran beside FU's quiet lock 09:24:44-09:29 (fb0 and the start of fb1): fb0's TTFTs are re-run in PFB (fb0r).

| arm | config | lifetime peak | TTFT 1.6K / 4.7K / 7.9K s | 12 HumanEval whole decode |
|---|---|---|---|---|
| fb0 / fh0 | L at C 256 | 45.10 GB | 16.5 / 41.4 / 68.1 (perturbed) | 175.2 tok/s (42.9 GB) |
| fb1 / fh1 | L at C 248 | 45.70 GB | 16.9 / 42.1 / 67.4 | 172.3 (43.5 GB) |
| fb2 | L + R57_PFCHUNK=512 | **44.00 GB** | 22.8 / 63.1 / 109.0 | |
| fb3 / fh3 | L + R57_CACHE_GB=2 | 45.20 GB | 16.2 / 40.6 / 65.4 | 175.7 (42.9 GB) |
| **fb4** | **L + PF-A (R57_PFLM=1)** | 45.50 GB | **8.5 / 13.7 / 19.2 (x1.9 / x3.0 / x3.5)** | |
| fb5 | L + PF-A + CACHE_GB 2 | 46.40 GB | 8.5 / 13.6 / 19.2 | |

- PF-A carries over to the lossy config: x1.9 / x3.0 / x3.5 TTFT on the agent prompts (fb4 vs fb0; fb0r will firm up the reference).
- The 7.9K prompt's lifetime peak stays over the 45.0 line in every variant except 512-token chunks (44.0 GB, at 1.4-1.6x the TTFT).
  C 248 and the 2 GB cache limit do not lower it (45.7 / 45.2 / 46.4 GB): the lifetime peak moves about 0.5 GB with the
  allocation pattern, so a fix needs more than 0.5 GB of margin. Decode cost of the fixes is nil (175.2 / 172.3 / 175.7 tok/s).

## Sun 8 Nov 15:2x: chat TTFT on M1-B + COH, from CHS5's records (data/chs/chat_ttft.txt; the orchestrator's "how bad is our chat")
No new run: CHS5's runs (12 N45 IFEval prompts, 37-101 tokens, C 256, CAP 256, quiet) already log `pfs`, the wall from the
prefill call to the first token's readback (chat drafts with the MTP head, so no drafter-cache setup inside it: TTFT = prefill).
- **M1-B + COH (z0b): TTFT median 2.92 s (2.34-3.81 s); fit 1.7 s + 22 ms per prompt token.** z0 / zsgcr agree within ~0.2 s per prompt.
- **At CAP 256 the TTFT is half the wall: decode 49.9 tok/s, end to end 24.4 tok/s** (TTFT 36.6 s of 71.5 s; the stack zsgcr:
  decode 56.0, end to end 24.6). Strata's h2h (the orchestrator): median TTFT 0.97 s on these prompts.
- Not a fixed cost (an ESTIMATE until PFB's CT1 arm measures it): each token routes top-10 of 512 experts per layer, so 40-100
  tokens touch ~55-85 % of each layer's experts; base residency at a prompt's start was 36-57 % in the agent records (PFSTAT), so
  ~6-14K expert-layers are read at ~2.6-2.8 MB = ~16-35 GB at PF-A's ~8 GB/s = ~2-4 s, the measured range. It saturates at the full
  63.6 GB read (PF-A's ~5.5 s + 1.57 ms/token on long prompts). A persistent server repeats it per request (the screen already is
  one: model loaded once, warm-up discarded, caches kept). Levers: parallel reads (8 -> 12-14 GB/s), a resident-set prefill (PF-D,
  lossy; CT1 prices it). CT1 (in PFB's slot: zs = M1-B + COH + SB1 on the same 12 keys, CAP 16, R57_PFSTAT) logs per prompt TTFT,
  prefill SSD GB, residency at start, routed and resident expert-layers.

## Sun 8 Nov 16:18-16:21: C32, the 32 GB chat cell (data/chs/c32_report.txt; the orchestrator's OK of NewIdeas' 32 GB arm)
12 IFEval (N45's keys, CAP 256) at C 160 (C30's ~32 GB route), card snapshot e057abf (FU's code), one pass per arm, quiet
(a heat pause froze the runner 16:17:06-16:18:07, before the first arm: no arm timed through it).
| arm | config | whole decode | kept / check | cycle ms | misses / check | MB / kept | lifetime peak | == AR | TTFT median | end to end |
|---|---|---|---|---|---|---|---|---|---|---|
| e32 | E (exact) | 33.8 tok/s | 2.53 | 73.7 | 168.4 | 80.8 | 30.60 GB | 12/12 | 2.83 s | 19.8 tok/s |
| zs32 | M1-B + COH + SB1 | **51.3 tok/s** | 2.52 | 48.6 | 42.7 | 20.8 | 31.20 GB | 2/12 | 2.99 s | 22.5 tok/s |
- **zs32 is LOSSY and UNGATED at 32 GB**: M1-B's IFEval gate (EvalSuite runner 59) ran at C 256, and QOM skips more cold ranks at C 160.
  Its tokens match AR on 2/12 prompts, the same count as at C 256 (z0 / zs / z0b 2/12).
- vs C 256 on the same prompts: E 36.4 (CHS1 e0) -> 33.8 (-7 %); zs 51.9 (CHS5) -> 51.3 (-1 %): QOM's cold-rank skips absorb most of
  the smaller cache (misses 42.7 vs 24.6 per check; E's 168.4 vs 89.8).
- Against Strata (the orchestrator's h2h at its 33.6-34.9 GB peak): decode 37.4, TTFT 0.97 s, end to end 25.7 tok/s. At ~31 GB we lead
  on decode (51.3 lossy / 33.8 exact) but trail end to end (22.5 / 19.8) because TTFT is ~3 s here (the prompt's expert reads; see
  the chat-TTFT section above and PFB's CT1).

## Sun 8 Nov 16:26-16:35: PFB = PF-B vs PF-A + CT1 (chat prefill anatomy) + FU's fb0 re-run (data/pfb/pfb_report.txt, d44ec7a)
Snapshot b5e0d1f (ref2 d05063fa), quiet lock, no heat pause during the run; E (exact) at C 256, CAP 64, the 1.6K / 4.7K / 7.9K agent prompts.
**PF-B (R57_PFLM=3) is +15-22 % over PF-A but NOT bit-exact.**
| arm | 1.6K | 4.7K | 7.9K | vs PF-A logits hash | tokens | lifetime peak |
|---|---|---|---|---|---|---|
| pa1 PF-A (control) | 189 tok/s (8.3 s) | 360 (13.1 s) | 425 (18.7 s) | - | - | 45.10 GB |
| pb3 PF-B, TOK 4096 + read-ahead | 218 (x1.15) | 428 (x1.19) | 490 (x1.15) | 0/7, 0/24, 0/22 | ==, DIFF, DIFF | 46.00 GB |
| pb38 PF-B, TOK 8192 | 218 (x1.15) | 438 (x1.22) | **499 (x1.17)** | 0/7, 0/28, 0/21 | ==, DIFF, DIFF | 46.20 GB |
| pb3n PF-B, no read-ahead | 197 (x1.04) | 384 (x1.06) | 456 (x1.07) | 0/7, 0/24, 0/22 | ==, DIFF, DIFF | 45.80 GB |
| pb3c PF-B + clear_cache per layer | 197 (x1.04) | 404 (x1.12) | 483 (x1.14) | 0/7, 0/24, 0/22 | ==, DIFF, DIFF | 45.80 GB |
- The read-ahead (the CPU reads the layer's experts while the GPU runs all chunks' attention) is most of the gain (no read-ahead: +4-7 %).
- Not exact in any PF-B arm, the read-ahead-free one included: the logits differ from PF-A at every hashed decode step and the tokens
  diverge on the 4.7K / 7.9K prompts. Suspect: the MoE runs on concatenated, 64-padded chunk groups (M up to TOK) where PF-A calls it per
  1,024-row chunk, so the expert matmuls see a different M (tiling / accumulation order). Untested exact variant: groups of one chunk
  (TOK = PFCHUNK) and no padding while rows x top-k <= 32,767 (LEDGER BUG1 needs > 32,767), same M as PF-A, read-ahead kept.
- Peaks rise 0.7-1.1 GB over PF-A (45.8-46.2 GB, line 45.0).
**CT1 (NewIdeas' pre-registered chat prefill price; zs = M1-B + COH + SB1, the 12 N45 IFEval prompts, CAP 16, R57_PFSTAT):**
- TTFT mean 3.10 s; prefill SSD 20.4 GB per prompt (15.6-25.9); routed expert-layers 7,321 (of 24,576); base residency at start 45.7 %;
  of the routed expert-layers 65.5 % had their MSB-3 base and 25.1 % their LSB side resident.
- **The chat prefill re-reads every routed expert-layer, resident or not: SSD bytes = routed x 2.79 MB on all 12 prompts (2.78-2.79).**
- **TTFT = 0.32 s + bytes / 7.33 GB/s (r = 0.993): the chat TTFT is the SSD read.**
- PF-D (serve resident base / side parts from the pools; exact, same weights) would keep ~45 % of the bytes (1 - 0.75 x 0.655 - 0.25 x
  0.251) = ~9.1 GB: TTFT ~1.56 s at today's rate, ~1.1 s at 12 GB/s (estimates; Strata 0.97 s).
- Pre-registration (NewIdeas QUEUE): bytes vs 12 GB's prefill (15-26 GB per prompt) ~100 % (>= 80 %), mean base residency at start 45.7 %
  (< 50 %, >= 25 %): BETWEEN by its letter; the routed-resident share (65.5 % base) is the direct measure of what PF-D skips.
**fb0r:** FU's fb0 (L at C 256; the orchestrator's publish job overlapped it) re-run: TTFT 16.7 / 66.6 / 40.9 s vs 16.5 / 68.1 / 41.4 s,
peak 45.10 GB both: the overlap did not change fb0 beyond ~2 %; FU's numbers stand.

## Sun 8 Nov 17:23-17:40: PFD = PF-D v1 (chat prefill serves pool-resident experts) + PF-B with Leap's seg fix (data/pfd/pfd_report.txt, a6d5347)
Snapshot ba7dc54 (code 06ed3dd), quiet lock, swap flat (5.39-5.41 GB). Gate 0 passed first: scripts/pfd_unit.py, the GPU rebuild of 18 experts
(layers 0 / 23 / 47) from their MSB-3 / LSB-1 parts == the ext's read_rebuilt_layout on all 9 pieces, 0 differing words.
**PF-D v1 is EXACT but SLOWER (12 N45 IFEval prompts, C 256):**
| arm | TTFT mean (median) | prefill SSD per prompt | rows A / B / C (12 prompts) | vs control | == AR | lifetime peak |
|---|---|---|---|---|---|---|
| pde0 E (control), CAP 64 | 3.06 s (2.94) | 20.4 GB | - | - | 12/12 | 41.80 GB |
| pde1 E + PF-D | 7.91 s (7.85) | 7.2 GB | 27,632 / 39,274 / 21,006 | tokens 12/12, logits hash 256/256 | 12/12 | 46.20 GB |
| pde1k E + PF-D + CHECK (timing void), CAP 16 | - | - | 63,835 rows compared | 0 differing (A 0 / B 0) | 12/12 | 48.90 GB |
| pdz0 zs (M1-B + COH + SB1; control), CAP 256 | 3.20 s (3.11) | 20.4 GB | - | - | 2/12 | 42.40 GB |
| pdz1 zs + PF-D | 8.15 s (8.32) | 7.5 GB | 25,965 / 39,880 / 22,002 | tokens 12/12, logits hash 721/721 | 2/12 | 46.90 GB |
- Bitwise: every rebuilt row equals the ext's, and tokens + logits hashes equal the control in both configs. SSD bytes -64 % (better than
  CT1's 45 % estimate: rows B read only the 0.72 MB side part).
- Slower: the added time is linear in the rebuilt rows (A + B): ~0.85 ms per expert-row (ifeval/1220 +2.58 s for 3,087 rows; ifeval/374
  +6.5 s for 7,599 rows), i.e. the MLX-op rebuild (~100 elementwise kernels per projection) runs at ~3.3 GB/s, slower than reading the
  bytes. Its intermediates also add ~4.4 GB to the peak. Next: one fused Metal kernel (mx.fast.metal_kernel) per piece type.
**PF-B + seg per chunk is EXACT (3 PFX agent prompts, E, CAP 64; vs PF-A pdpa1 189 / 358 / 433 tok/s, peak 45.10 GB):**
- pdb38s (TOK 8192): 215 / 432 / **500 tok/s** (x1.14 / x1.21 / x1.16); tokens == and logits hash 8/8, 28/28, 22/22; peak 46.20 GB.
- pdb32s (TOK 2048): 212 / 418 / 486 tok/s (x1.12 / x1.17 / x1.12); all ==; peak 45.80 GB.
- Leap's find (6553fcb): the shared expert's q4 gate (N = 1) runs through qmm_splitk with split_k from M; per chunk slice it is bitwise.

## Sun 8 Nov 17:47-17:57: PFD v2 = PF-D with the rebuild as two fused Metal kernels (data/pfd/pfd_report.txt, snapshot 06c8bf9, code 08c90e7)
Unit test first (scripts/pfd_unit.py, no model): kernels == read_rebuilt_layout on 18 experts x 9 pieces; 0.050 ms per expert vs v1's MLX
ops 0.901 ms (96 experts of layer 23; kernel == ops on all 96).
| arm (12 N45 IFEval, C 256) | TTFT mean (median, min-max) | SSD per prompt | vs control | lifetime peak |
|---|---|---|---|---|
| pde0 E (control), CAP 64 | 3.01 s (2.89, 2.45-3.77) | 20.4 GB | == AR 12/12 | 41.80 GB |
| pde1 E + PF-D v2 | 3.01 s (2.97, 2.41-3.68) | 7.2 GB | tokens 12/12, logits hash 256/256, == AR 12/12 | 44.30 GB |
| pde1k + CHECK (timing void), CAP 16 | - | - | 63,835 rows compared, 0 differing | 47.20 GB |
| pdz0 zs (control), CAP 256 | 3.12 s (3.01, 2.45-3.81) | 20.4 GB | - | 42.40 GB |
| pdz1 zs + PF-D v2 | 3.30 s (3.11, 2.55-4.04) | 7.5 GB | tokens 12/12, logits hash 721/721 | 45.00 GB |
- **Exact, -64 % SSD bytes, but no TTFT gain yet** (E: equal; zs: +6 %). Per-prompt stage timers (pde1, runs.json "pfd"): side-part reads
  0.22 s, today's reads for absent bases 0.44 s (~11 GB/s on both), GPU rebuild + scatter 0.96 s (~20 ms per layer, ~3x the kernels
  alone), the MoE call 0.21 s, everything else 1.19 s. The step that should cost ~0.3 s costs ~1 s: next, time its parts (gathers,
  numpy -> MLX copies, scatter) on synthetic pools (scripts/pfd_bench.py), then fuse or overlap them.
- Peak: +2.5 GB over the control (44.3 / 45.0 GB vs 41.8 / 42.4): the gathered inputs, kernel outputs and the scattered scratch copy.

## Sun 8 Nov 18:10-18:27: PFD v3 = PF-D with one-pass assemble kernels: EXACT and FASTER (data/pfd/pfd_report.txt, snapshot 7d50712, code 4d08ac9; data 862660f)
Unit first (scripts/pfd_unit.py): the two assemble kernels (rowmap: A rows from the pools in place, B rows from the base pool + read side
parts, C rows copied from the ext's rebuild) == read_rebuilt_layout on 8 A / 8 B / 8 C experts in shuffled scratch rows; pfd_bench.py
4.65 ms per chat layer vs v2's 11.94 ms. In-run, 12 N45 IFEval prompts, C 256, quiet (EvalSuite's card reference yielded):
| arm | first token mean (median, min-max) | prefill SSD per prompt | vs control | lifetime peak |
|---|---|---|---|---|
| pde0 E (exact; control), CAP 64 | 3.04 s (2.89, 2.29-3.84) | 20.4 GB | == AR 12/12 | 41.80 GB |
| **pde1 E + PF-D v3** | **2.29 s (2.26, 1.72-2.70)** | 7.2 GB | tokens 12/12, logits hash 256/256, == AR 12/12 | 42.60 GB |
| pde1k + CHECK (timing void), CAP 16 | - | - | 87,912 rows (A 26,570 / B 37,265 / C 24,077) compared: 0 differing | 46.00 GB |
| pdz0 zs (M1-B + COH + SB1; control), CAP 256 | 3.14 s (3.02, 2.51-3.87) | 20.4 GB | - | 42.40 GB |
| **pdz1 zs + PF-D v3** | **2.67 s (2.54, 2.19-3.19)** | 7.5 GB | tokens 12/12, logits hash 721/721 | 43.40 GB |
| pdz1n zs + PF-D v3, MLX cache kept (R57_PFD_CC=0) | 2.58 s (2.48, 2.08-3.16) | 7.5 GB | tokens 12/12, logits hash 721/721 | 44.00 GB |
- First token -25 % on E (exact to NAX6-N) and -15 % / -18 % on the chat config, bitwise to the control in every run; SSD bytes -64 %.
- Where the 2.29 s goes now (pde1, per prompt, runs.json "pfd"): inside the MoE calls 2.03 s = side-part reads 0.21 + C-row scratch zeros
  0.15 + C-row reads (ext CPU rebuild) 0.44 + assemble 0.59 (~12 ms per layer in-run vs 4.65 in the bench) + the MoE call 0.25 + per-call
  Python / lookups / copies ~0.39; outside the MoE calls 0.26 s (zs: 0.55 s, the MTP head's prompt pass). Next levers: C rows as raw reads
  + GPU rebuild (no CPU rebuild, no zeros), reads overlapped with the assemble of A rows, and Strata's next-layer read-ahead.
- **CAVEAT (18:3x, same run): the chat DECODE got slower with PF-D**, with identical tokens and misses (18.2 per check). Pooled triples
  (whole decode / whole request / first token median): pdz0 49.6 / 23.9 tok/s / 3.02 s; pdz1 40.0 / 23.0 / 2.54 s; pdz1n 38.2 / 22.8 /
  2.48 s; E: pde0 35.8 / 12.5 / 2.89 s, pde1 35.1 / 14.7 / 2.26 s. The check's CPU build rose 34.2 -> 43.7 ms (zs), mostly after a prompt's
  16th check (38.4 -> 52.3 ms); E's checks +2 %. The PF-D arms ran after the control (order / heat not excluded); PF-D also leaves ~0.35 GB
  more footprint outside MLX. Until a diagnostic (order swapped, PF-D idle, no reader threads) explains it: no chat whole-request claim.

## Sun 8 Nov 23:49-23:55: PFD diagnostic = the chat decode slowdown did NOT come back (data/pfd/pfd_report_2355.txt, snapshot 366c452; data 3175ef3)
Same 12 N45 IFEval prompts, zs = M1-B + COH + SB1, C 256, CAP 256, quiet, PF-D v3 FIRST this time (the 18:2x run had it after the control):
| arm (run order) | whole decode | whole request | first token median (mean) | check ms (CPU build) | tokens / logits hash vs control | lifetime peak |
|---|---|---|---|---|---|---|
| pdz1 zs + PF-D v3 (1st) | 50.1 | **27.4** | **2.32 s** (2.39) | 39.4 (34.1) | 12/12, 721/721 | 43.40 GB |
| pdz0 zs control (2nd) | 50.3 | 24.6 | 2.91 s (3.02) | 39.2 (33.8) | - | 42.40 GB |
| pdz1i PF-D installed, idle (3rd) | 50.6 | 24.8 | 2.90 s (2.99) | 39.0 (33.6) | 12/12, 721/721 | 42.40 GB |
| pdz1p PF-D v3, reader threads shut after each call (4th) | 50.8 | **28.0** | **2.25 s** (2.33) | 38.8 (33.4) | 12/12, 721/721 | 43.40 GB |
- With PF-D first, the chat decode is unchanged (50.1-50.8 vs 50.3 tok/s; check CPU build 33.4-34.1 vs 33.8 ms, also after a prompt's
  16th check: 37.6-38.1 vs 37.9) and whole request rises 24.6 -> 27.4 / 28.0 tok/s (+11 / +14 %), first token median 2.91 -> 2.32 /
  2.25 s (-20 / -23 %), bitwise to the control (tokens 12/12, logits hash 721/721). SSD per prompt 20.4 -> 7.5 GB. Peak +1.0 GB.
- The 18:2x slowdown (49.6 -> 40.0 whole decode, CPU build +9.5 ms) is not reproduced; its cause is unknown. That run had PF-D after
  the control, fans held at 5000 (die 70-85 C, thermal state nominal; a heat pause had cleared 27 s before its start) and came right
  after EvalSuite's reference yielded; tonight: PF-D first, fans ~7000, die 73-81 C. Not PF-D's own cost: idle (installed, unused) and
  active arms decode alike tonight.
- The v4 (exact, raw C reads + read-ahead), FAST and v4-check arms died compiling in-run (MLX passes inputs of < 8 elements in Metal's
  'constant' space; the v4 kernels take 'device' pointers; the engine's placeholder had 1 element; the unit test only used real
  arrays). Fixed in 9aa398b (pflm.v4_placeholder, a 1,024-word array; pfd_unit.py now tests the placeholders: PASS). Re-run queued.

## Mon 9 Nov 00:02-00:09: PFD v4 + read-ahead (exact) and FAST (lossy) in-run, after the placeholder fix (data/pfd/pfd_report_0009.txt, snapshot 36ee120; data db922b9)
Same 12 N45 IFEval prompts, zs, C 256, CAP 256, quiet, no heat pause in the window; control = pdz0 of 23:5x (tokens are deterministic):
| arm (run order) | whole decode | whole request | first token median (mean) | SSD per prompt | vs the control | lifetime peak |
|---|---|---|---|---|---|---|
| pdz4a PF-D v4 exact + read-ahead (1st) | 50.7 | **28.8** | **2.26 s** (2.17) | 9.6 GB | tokens 12/12, logits hash 721/721 | 43.70 GB |
| pdz1q PF-D v3, POOL=call (2nd; this run's reference) | 50.4 | 27.4 | 2.41 s (2.41) | 7.5 GB | 12/12, 721/721 | 43.40 GB |
| pdzfa PF-D FAST + read-ahead (3rd; LOSSY) | 49.3 | 30.5 | 1.83 s (1.74) | 4.9 GB | tokens 2/12, logits hash 0/656 | 43.20 GB |
| pde4k E + v4 + read-ahead + CHECK (4th; timing void), CAP 16 | - | - | - | 30.4 GB | 87,912 rows == the ext's (A 26,570 / B 37,265 / C 24,077), == AR 12/12 | 46.40 GB |
- **v4 + read-ahead is exact and the fastest exact chat prefill:** on zs tokens and logits equal the control; on E every staged row
  equals the ext's rebuild. Same run vs v3: first token mean 2.17 vs 2.41 s (-10 %), median 2.26 vs 2.41, whole request 28.8 vs 27.4
  (+5 %). Vs the control: whole request 24.6 -> 28.8 (+17 %), first token median 2.91 -> 2.26 s (-22 %), whole decode 50.7 vs 50.3.
- Read-ahead (layer L+1's router on layer L's MoE input, top-10 union): of the rows needed from the SSD, 75 % (B) / 72 % (C) came
  from the stage; 19 % / 31 % of staged rows went unused (predicted B 37,068 / C 22,912, used 29,898 / 15,726, on demand 9,982 / 6,276),
  so SSD bytes per prompt rose 7.5 -> 9.6 GB. Left: better prediction (two layers' gates, or the union of the last tokens' choices).
- **FAST** (side-absent experts as 3-bit base-only pieces): first token median 1.83 s, whole request 30.5 (+24 % vs the control), but
  tokens differ on 10 / 12 prompts: LOSSY and ungated (no same-scores gate is scheduled before 16:00), so it ships off, labelled.
- Defaults (rel0): chat at 45 GB = PF-D v4 + read-ahead on (exact). E at 45 GB: off (its table rows were measured without PF-D; on
  is exact: v3 cut E's first token 2.89 -> 2.26 s at 64 tokens). 32 GB: off (not measured there; PF-D adds 1.0-1.3 GB of peak).

## Mon 9 Nov 09:47-09:57: REL ROWS = the shipped lossy config L + COH: rows, references, and a repeat (data/rel/rel_rows_report.txt, snapshot rel1 = 0b4df8c; data 53dfff5)
L + COH = data/stack/lcoh_env.txt (== the package's lossy-45 / lossy-32 line), from the frozen rel1 snapshot, quiet lock at
EvalSuite's lcoh process boundary (NewIdeas' slot), NO-SWAP 54.7 GB at the start, CAP 256, one process per arm, run order as listed.
| arm | set (12 prompts) | C | whole decode | whole request | first token median (mean) | lifetime peak | == AR | ended before the cap |
|---|---|---|---|---|---|---|---|---|
| rlh_256 | HumanEval (E5's e5h1 keys) | 256 | 164.0 | 36.2 | 4.60 s (4.70) | 42.94 GB | 5/12 | 7 |
| rlm_256 | MBPP (E5's e5m1 keys) | 256 | 112.7 | 25.3 | 3.72 s (3.88) | 42.60 GB | 5/12 | 11 |
| rli_256 | IFEval (data/n45/keys.txt line 1) | 256 | 45.4 | 22.3 | 3.12 s (3.16) | 42.42 GB | 2/12 | 10 |
| rlt_256 | math (line 2) | 256 | 78.9 | 37.9 | 3.20 s (3.52) | 43.17 GB | 5/12 | 0 |
| rli_160 | IFEval | 160 | 39.0 | 21.1 | 2.90 s (3.00) | 31.30 GB | 2/12 | 10 |
| rlr_256 | IFEval again (the repeat) | 256 | 44.9 | 22.5 | 2.91 s (3.06) | 42.42 GB | 2/12 | 10 |
- **REPEAT 12/12:** rlr_256 (its own process, same keys in the same order) gave rli_256's tokens on every prompt. L alone repeated on
  4/12 (CHS4 y0 / y0b); COH had made E's one-call path and M1-B repeat (CHS4 ycoh / ycoh2, CHS5 z0 / z0b); now L + COH too. It
  shows same-order repeatability only: the drafter's state carries across prompts, so a different history may still change tokens.
- **C 160 == C 256** on the IFEval tokens (12/12): one set of IFEval references serves lossy-45 and lossy-32.
- Pooled IFEval at 45 GB (rli_256 + rlr_256, 24 rows): 45.2 / 22.4 / 3.01 s, 42.42 GB (RELEASE row 27). All peaks are under their
  lines (45.0 at C 256, 32.0 at C 160).
- Against L's rows (RELEASE row 28, before COH): HumanEval 164.0 vs 171.0 (-4 %), MBPP 112.7 vs 114.5 (-2 %), IFEval 45.2 vs 43.4
  (+4 %), math 78.9 vs 86.9 (-9 %). This is not an A/B: one pass against two, different days, and COH changes tokens.
- Checks: cut rows 0 on all six arms (the screen's 600 s budget); fanwatch shows no PAUSE or freeze between the grant (09:47:01) and
  the end (09:56:39); the three planes_engine kernel files read live equal the copies frozen in rel1 before and after the run
  (md5 dc81729b / 3b835480 / 936c2635).
- The tokens (data/r57/rl*_256_tokens.json, rli_160_tokens.json) are the package's lossy references; paths and the REPEAT count went
  to Leap at 09:5x for its 12:05 final run. turn45 logged "done rc=1": the script's last log call returned 1 outside dry runs; the
  report and the data commit succeeded. Fixed in the next commit.
- **Correction (Mon 10:3x, after EvalSuite's naming note):** RELEASE row 28's HumanEval / MBPP cells (171.0 / 114.5) were S5' B = CERT
  (v1' + wired, no bundle), while its IFEval / math cells (N45 nil / nml) and this run's base are L = s5p_env = CERT + the 6-key bundle
  (scripts/n45.sh l30-31 read s5p_env; lcoh_env = s5p_env + REF2_NAX_COH=1, one key apart). Against L itself (S5' X: spx1-2, spmx1-2;
  pooled 181.3 / 36.6 / 4.51 s and 119.8 / 25.3 / 3.76 s) L + COH is HumanEval -9.5 %, MBPP -5.9 %, math -9.2 %, IFEval +4.1 % whole
  decode: still not an A/B (different days, one pass against two, tokens differ, so acceptance differs), cause not measured. RELEASE row
  28 now holds L's rows, with CERT's two rows and the certificate in its label.

## Mon 9 Nov 19:07-19:34: W24 = the shipped lossy config L + COH below 24 GB on the fast path (data/w24/w24_report.txt; snapshot w24 = 016cb7f; data 85ed4d3)
Kamen 14:5x / 18:5x via the orchestrator: 120 tok/s whole decode on HumanEval under 24 GB, gated before Wednesday's publish. A check
takes the sync-free gate only if rows x top-10 <= base slots (bigmoe/stream.py:76): F24k (Sat; 94 base slots, 16-row checks) ran on
the slow sync path at 27.7 tok/s. W24 moves the same ~13 GB of pools into the base pool and caps checks so they fit; R57_VTIER off
(no volatile tier beside the footprint). One pass per set, 12 prompts, CAP 256, quiet lock, fan cap 5000 (Kamen on the Mac).
Whole decode / whole request (tok/s) / first token median / lifetime peak:

| arm | slots (base / sidecar), rows | HumanEval | MBPP | IFEval (chat) |
|---|---|---|---|---|
| **w12** | 120 / 34, 12 | **116.5 / 33.3 / 4.51 s, 23.67 GB** | 84.1 / 24.8 / 3.86 s, 23.33 | 28.4 / 17.3 / 3.04 s, 23.16 |
| w16u (U3 rule, 16-row code checks) | 120 / 34, 16 on code | 39.7 (1 heat-paused row) | 68.6 | 17.1 (1 paused) |
| w10 | 104 / 76, 10 | 102.6 | 11.7 (1 paused: 117.9 s) | 28.3 |
| w8 | 100 / 90, 8 | 98.2 | 28.6 (1 paused) | 16.8 (1 paused) |
| w12pf (+ look-ahead prefetch) | 120 / 34, 12 | did not start: assert PF_B 96 <= 30 + 31 x PRED_LINES (1); needs BIGMOE_PRED_LINES=4 | | |

- **w12 = the pick** (pre-registered rule, docs/W24_DECISION.md): highest HumanEval whole decode; its sets had no paused rows.
  HumanEval kept 9.34 per check at a 79.9 ms median cycle (mean 80.2 ms: no slow checks), 29.1 MB SSD per kept token. **3 % short
  of 120.**
- **Quality risk:** QOM drops 30.7 % (HE) / 27.7 % (MBPP) of optional expert uses at 120 base slots, against 11.4 / 12.2 % at C 256:
  the C 256 card gate does not carry. The 24 GB config needs its own card gate (EvalSuite runner 61, ~10 h).
- **Heat pauses:** four rows in other arms waited 37-118 s on fanwatch's PAUSE (fan cap 5000); whole decode counts that wait, so those
  sets read low (e.g. w10 MBPP 11.7 with a 99.6 tok/s steady state on the paused prompt). w12's numbers are unaffected.
- **The report's 'NOT all fast'** comes from 241 sync-path tier calls per PROCESS, the same on every set (warm-up runs, not measured
  checks): to be fixed in the report (W24b).
- **w16u (U3, 16 rows on the gate via R57_GATEREQ, 742758e) is slower**, not faster: 2x the SSD bytes per kept token (57.5 MB) and long
  tails; dropped.
- **Next (W24b):** w13t = 130 base / 24 sidecar, rows 13, with the prefill transient trimmed (BIGMOE_TIER_PF_GROUP=64,
  R57_CACHE_GB=0.25; the lifetime peak is set in prefill: MLX peak prefill 22.2-22.7 vs decode 21.5-21.7 GB), a w12 repeat (noise),
  and w12pf with BIGMOE_PRED_LINES=4.

## Mon 9 Nov 19:40-19:59: W24b = arms toward 120 at 24 GB, HumanEval + MBPP (data/w24/w24b_report.txt; data 8f76fc3)
Same snapshot as W24 (016cb7f); one pass per set; quiet lock; fan cap 5000. Like for like on the prompts that no heat pause touched
(11 per set; HE/63 and MBPP/319 paused in one arm each), whole decode tok/s:

| arm | change from w12 | HumanEval | MBPP | first token median HE / MBPP | lifetime peak HE |
|---|---|---|---|---|---|
| w13t | 130 base / 26 sidecar, 13 rows, prefill trim | **115.4** (pooled 12: 116.5) | **90.5** | 3.66 / 3.37 s | 23.92 GB |
| w12r | none (w12 again) | 110.4 | 87.8 | 4.88 / 3.86 s | 23.67 GB |
| w12t | prefill trim (BIGMOE_TIER_PF_GROUP=64, R57_CACHE_GB=0.25) | 113.7 | 86.9 | 3.62 / 3.28 s | 23.20 GB |
| w12pf | + look-ahead prefetch (gate+t+pf, BIGMOE_PRED_LINES=4) | 109.3 | 87.2 | 4.38 / 3.67 s | 23.67 GB |
| (W24's w12, another run) | | 115.7 | 86.1 | 4.54 / 3.88 s | 23.67 GB |

- **No arm reaches 120.** w13t vs the same-run w12r: +4.5 % HE, +3.1 % MBPP; but w12 and w12r give the same tokens (12 / 12 on both
  sets) and differ by 4.6 % on HE, so the run-to-run noise is as large as the gain: not a demonstrated gain.
- **The prefill trim is token-exact** (w12t == w12r, 12 / 12 on both sets) and, in the same run, cuts the first token by ~25 % (HE
  4.88 -> 3.62 s, MBPP 3.86 -> 3.28 s) and the lifetime peak by 0.47 GB. A first-token lever to try at 32 and 45 GB too.
- **The look-ahead prefetch does not help at 24 GB** (109.3 vs 110.4 same run), as at 45 GB.
- **Gate pick** (docs/W24_DECISION.md, highest HE): w13t -> data/stack/card_w24_env.txt (8f76fc3) for EvalSuite's card gate from
  Tue 01:25 (~10 h). Its peak is 0.08 GB under 24.0; QOM drops 28.8 / 26.6 % of optional uses (w12: 30.7 / 27.7).
- Bookkeeping: the run's own data commit failed (bash 3.2 has no ${RUN^^}); committed by hand (8f76fc3), script fixed (ebaa58f).

## Mon 9 Nov 23:34-23:43: W24c = the drafter diet buys a 14th row: 126 tok/s on HumanEval under 24 GB (data/w24/w24c_report.txt, w24c2_report.txt, w24c_verdict.txt; data 71d239b, e7c6eb0)
Same snapshot as W24 (016cb7f), env only; quiet lock; Kamen off the Mac (apps closed). The gated w13t holds two DFlash2 drafters
(bf16 for math / agent-typed prompts, 4-bit for code); R57_D7B=all keeps only the 4-bit one (code drafts are unchanged), and the
freed ~1.1 GB pays for 140 base + 28 sidecar slots and 14 rows. ABBA on HumanEval; all 12 prompts clean in every arm (no pause,
no bad row). Whole decode tok/s:

| arm | config | HumanEval | whole request | first token | kept per check | lifetime peak |
|---|---|---|---|---|---|---|
| w13c | w13t as gated (130 / 26 slots, 13 rows, both drafters) | 120.9 | 39.7 | 3.50 s | 9.97 | 23.92 GB |
| w14a | 140 / 28 slots, 14 rows, R57_D7B=all | **125.8** | 40.2 | 3.58 s | 10.33 | 23.81 GB |
| w14b | w14a again | **126.4** | 40.4 | 3.56 s | 10.33 | 23.77 GB |
| w13d | w13t as gated, again | 121.4 | 39.0 | 3.62 s | 9.97 | **24.01 GB (over the line)** |

- **14 rows under 24 GB: 126.1 tok/s pooled** (w14a + w14b, all 12 prompts each), +4.1 % over the same-run controls (121.1). The
  controls agree within 0.4 %, the repeats within 0.4 %, and tokens are identical within each pair (12 / 12). w14a on MBPP:
  87.9 tok/s, 23.69 GB; on IFEval: 29.8 tok/s, 23.64 GB. Checks over 300 ms: 0 on the code sets; 3-4 of 633 on IFEval, all the
  first or last check of a prompt.
- **w13t is not reliably under 24 GB:** w13d peaked 24.006 GB (HumanEval/68) with the same tokens as w13c (23.916): the
  footprint moves ~0.1 GB between identical runs. The 14-row config peaks 0.19-0.37 GB under the line on all three sets.
- w13t itself reaches 121 on a quiet machine (W24b, with Kamen on the Mac: 116.5).
- **Gate rule** (docs/W24_DECISION.md 'W24c', pre-registered): conditions 1-3 PASS (126.11 >= 120.0; 0 bad rows; 23.808 <
  24.000), condition 4 FAILS (+4.1 % vs the > +4.6 % bar, the noise between two separate runs), so as written w13t stays. The
  orchestrator was asked at 23:4x whether to switch anyway (w13t's 24.006 breach; 0.4 % noise inside this run).
  **Decision (the orchestrator 23:4x): it overrode the +4.6 % bar and switched the 02:10 card gate to the 14-row config
  (data/stack/card_w24c_env.txt, EvalSuite's dry run passed), subject to NewIdeas' number check** (docs/W24_DECISION.md
  override note, bd719b4). Published numbers come from this same-run pair: 125.8 / 126.4 tok/s (pooled 126.1), peak 23.81 GB.
- **Diagnostics** (same run, after the decision passes, single samples; data/w24/w24c3_report.txt, w24c4_report.txt):
  - w13a = w13t + R57_D7B=all: 122.7 tok/s, lifetime peak **22.71 GB**: the drafter diet alone frees 1.2 GB.
  - w13b = the freed GB as 11 more base slots (141 / 26, 13 rows, R57_D7B=all): **129.9 tok/s at 23.84 GB**, the best arm:
    cycle 77.4 ms (14-row 80.5), SSD 24.2 MB per kept token (25.0), QOM drops 26.7 % (27.0; w13t 28.8), 0 slow checks.
    One sample after the ABBA block, so W24d re-tests it against the 14-row config in an ABBA (the orchestrator's option B).
  - **The 14-row gain is HumanEval-only:** w13t on MBPP 92.7 tok/s / IFEval 30.2 vs the 14-row 87.9 / 29.8 (-5.2 % / -1.5 %).
    MBPP keeps ~6.3 tokens per check, so a 14th row costs more misses than it returns.
  - Slow checks on IFEval: w13t 5, the 14-row 4 (of ~633), all a prompt's first or last check: normal for the chat path.

## Mon 9 Nov 23:51-23:56: W24d = w13b vs the 14-row config, same-run ABBA on HumanEval: w13b 130.3 tok/s under 24 GB (data/w24/w24d_report.txt, w24d_verdict.txt; data 86f0d13)
The orchestrator's option B: W24c's diagnostic w13b re-tested against the 14-row config, rule pre-registered (docs/W24_DECISION.md
'W24d', 498691a). Same snapshot (016cb7f), env only, quiet lock. All 12 prompts clean in every arm. Whole decode tok/s:

| arm | config | HumanEval | whole request | first token | kept per check | cycle | SSD MB / kept token | QOM drops | lifetime peak |
|---|---|---|---|---|---|---|---|---|---|
| w14c | 14-row: 140 base + 28 sidecar, 14 rows, R57_D7B=all | 126.4 | 40.1 | 3.53 s | 10.33 | 80.5 ms | 24.9 | 27.0 % | 23.83 GB |
| w13e | w13b: 141 base + 26 sidecar, 13 rows, R57_D7B=all | **130.2** | 40.7 | 3.54 s | 10.15 | 77.6 ms | 24.2 | 26.7 % | 23.82 GB |
| w13f | w13b again | **130.5** | 40.9 | 3.53 s | 10.15 | 78.1 ms | 24.2 | 26.7 % | 23.78 GB |
| w14d | 14-row again | 125.5 | 40.5 | 3.54 s | 10.33 | 78.9 ms | 24.9 | 27.0 % | 23.78 GB |

- **w13b: 130.3 tok/s pooled, +3.5 % over the 14-row config (125.9) in the same run**; peaks 23.82 / 23.78 GB; 0 bad rows.
  The rule's three measured conditions pass (> +1 %; w13b peaks < 24.000 GB; zero bad rows); EvalSuite's dry run of
  data/stack/card_w24d_env.txt passed (blob e06f495e). The switch to w13b waits only for NewIdeas' number check (by 01:30).
- Reproducible: tokens identical 12 / 12 within each pair and across runs (w13e == W24c's w13b; w14c == W24c's w14a); speeds
  repeat (w13b 129.9 -> 130.3; 14-row 126.1 -> 125.9). For reference, w13t as gated: 120.9 / 121.4, peak 23.92 / 24.01 GB.
- Why w13b beats a 14th row: with 141 slots for at most 130 ids per check, 11 slots keep experts resident across checks, so
  the cycle is ~3 ms shorter and the SSD reads ~3 % fewer; the 14th row gains kept tokens on HumanEval only.
- Not measured yet: w13b on MBPP and IFEval (the 14-row config was 5.2 % slower than w13t on MBPP).

## Tue 10 Nov 00:28-00:54: PF-D on multi-chunk prompts = EXACT (data/pfd_long/summary.txt; scripts/pfd_long.sh, bfcc256)
The orchestrator's question via Leap (PF-B, R57_PFLM=3, is not bitwise at 30K): is PF-D exact beyond the single-chunk IFEval
prompts it was checked on? The published chat-45 config (PF-D on: R57_PFD=1, V=4, AHEAD=1) through the release package's
run.py, on Leap's 9,988- and 29,988-token prompts (~/Leap/data/prefill_1109/prompt_{10k,30k}.txt), 64 new tokens,
R57_LOGHASH=1, one process per run, quiet lock:

| prompt | arm | first token | whole decode | lifetime peak | vs PF-D off | PF-D calls (fallbacks) | rows A / B / C | rows checked (differing) |
|---|---|---|---|---|---|---|---|---|
| 10K | PF-D off (R57_PFD=0) | 78.89 s | 35.5 | 45.17 GB | (reference) | - | - | - |
| 10K | PF-D as published | 76.56 s | 35.5 | 47.46 GB | tokens 64/64, logits hashes 25/25 | 480 (0) | 24,634 / 50,274 / 100,271 | - |
| 10K | + R57_PFD_CHECK=1 (timing void) | - | - | 52.66 GB | tokens + hashes equal | 480 (0) | the same | 175,179 (0) |
| 30K | PF-D off | 249.41 s | 24.5 | 46.65 GB | (reference) | - | - | - |
| 30K | PF-D as published | 238.39 s | 24.6 | 48.56 GB | tokens 64/64, logits hashes 27/27 | 1,440 (0) | 73,997 / 153,466 / 302,778 | - |
| 30K | + R57_PFD_CHECK=1 (timing void) | - | - | 52.84 GB | tokens + hashes equal | 1,440 (0) | the same | 530,241 (0) |

- **PF-D is exact on multi-chunk prompts:** at 10K and 30K, its tokens and logits hashes equal the stock chunk-major prefill's,
  and the check arms found all 705,420 assembled scratch rows equal to tier._read_rebuilt's. One PF-D call per 1,024-row
  chunk per layer (10 / 30 chunks x 48 layers), no fallback: the 45 GB chat config sets no grouped prefill, so long prompts
  DO take PF-D (the release text says they take the grouped path without it: wrong, sent to the Editor).
- **Its first-token gain is small on long prompts:** -3.0 % at 10K, -4.4 % at 30K (on short chat prompts: 2.91 -> 2.26 s).
  57 % of a long prompt's per-chunk unique experts have no resident base (rows C), so most bytes are read anyway.
- **It costs ~2 GB of lifetime peak on long prompts** (staging): 45.17 -> 47.46 GB at 10K, 46.65 -> 48.56 GB at 30K. Without
  PF-D the 45 GB chat config is already above its 45.0 line on these prompts (as on the 7.9K agent prompt).
- Decode is unchanged (35.5 / 24.5-24.6 tok/s). Swap grew 1.5 GB during the 10K check arm (52.7 GB peak), under the guard.

## Tue 10 Nov 02:0x: w13b's 24 GB line holds only on short prompts (EvalSuite's gate, fwd process 1; analysis here)
EvalSuite's card gate on w13b (card_w24d_env.txt) measured a lifetime peak of **24.81 GB** on gpqad/rec6sE2CRtD4drtHg, a
919-token prompt: MLX prefill peak 23.83 GB (EvalSuite results/card/es_cdw24_100136_fwd1_runs.json). Every W24 prompt was
short, which is why W24c / W24d stayed under the line. MLX prefill peak against prompt length over all 176 rows of the 24 GB
runs plus the gate's first 20 rows [K]: under 400 tokens (174 rows) median 22.8 GB, max 23.1; 481 tokens 23.03; 919 tokens
23.83. The lifetime footprint adds ~1.0 GB outside MLX. Mechanism [P]: the screen prefills in R57_PFCHUNK = 1,024-row chunks,
so a prompt up to 1,024 tokens is one chunk whose transients grow with its length; beyond that only the KV / recurrent state
grows (at 45 GB without PF-D: +1.8 GB at 10K, +3.3 GB at 30K, data/pfd_long). At 45 GB, R57_PFCHUNK=512 cut a 7.9K prompt's
peak by 1.1 GB at 1.4-1.6x the first token (FU table above); untested at 24 GB, and probably not token-identical (the MoE call
shapes change). Sent to the orchestrator with options: label the line as holding on benchmark-length prompts, and/or measure
w13b with smaller chunks on long prompts after the gate. The gate itself (scores) runs on.

## Tue 10 Nov 08:53-09:00: W24e = w13b with smaller prefill chunks on the gate's longest prompts: no smaller chunk keeps the tokens (data/w24/w24e_report.txt; data 8e515d5)
The orchestrator's option (B). w13b exactly as gated (data/stack/card_w24d_env.txt, blob e06f495e) + R57_PFCHUNK, on the card
gate's six long prompts (726-2,801 tokens; not its six longest), CAP 128, R57_LOGHASH=1, one process per arm, quiet; rule pre-registered
(docs/W24_DECISION.md 'W24e'). First token per prompt and the run's max lifetime peak:

| chunk | 2,801 (GPQA) | 1,465 (LCB) | 1,102 (LCB) | 942 (LCB) | 919 (GPQA) | 726 (LCB) | max lifetime peak | tokens / hashes vs 1024 |
|---|---|---|---|---|---|---|---|---|
| 1024 (as gated) | 14.57 s | 11.29 s | 10.02 s | 6.21 s | 5.72 s | 6.02 s | 25.025 GB | (reference) |
| 512 | 21.60 s | 14.79 s | 14.42 s | 9.90 s | 8.46 s | 9.75 s | 24.345 GB | 0/6, 0/6 |
| 256 | 31.20 s | 23.37 s | 19.60 s | 17.93 s | 13.46 s | 12.99 s | **23.954 GB** | 1/6, 0/6 |
| 1024 again | 14.32 s | 11.06 s | 9.77 s | 6.08 s | 5.58 s | 5.89 s | 24.961 GB | 6/6, 6/6 |

- **The chunk drives the long-prompt peak** (MLX prefill peak on the 2,801-token prompt: 24.12 / 23.50 / 23.13 GB at 1024 / 512
  / 256 rows), and only 256-row chunks stay under 24.0 GB on all six prompts, at 2.1-2.3x the first token.
- **But a smaller chunk changes the tokens** (512: 0 of 6 prompts identical, 256: 1 of 6; logits hashes 0 of 6 for both),
  while two 1024 runs agree exactly. By the rule neither carries the card gate's label: **Wednesday ships w13b as gated
  (R57_PFCHUNK = 1024) with the label "under 24 GB on prompts up to about 560 tokens, up to 25.1 GB on longer ones"** (the
  gate's recount over 40 processes: 22 over the line, max 25.112 GB, shortest prompt at a first crossing 561 tokens).
- A strictly-under-24 GB variant (256-row chunks) would need its own card gate.
- The card gate on w13b PASSED at 08:41 on all six lines vs NAX6-N (EvalSuite results/CARD_w24_DONE): IFBench 40.7 / 40.7 % vs
  44.0 % (p 0.30), GPQA-D 47.0 / 47.0 % vs 44.0 % (p 0.51), LCB v6 29.8 / 32.8 % vs 32.1 % (p 0.45 / 1.0).

## Tue 10 Nov 10:03-10:16: W24f = w13b vs w13t on MBPP and IFEval, same-run ABBA (data/w24/w24f_report.txt, w24f_verdict.txt; data 9e48f45)
The orchestrator's slot after Leap's package check. Arms tf1 (w13t, W24b's gated config: 130 base / 26 sidecar, 13 rows, both
drafters), bf1, bf2 (w13b, the card-gated 24 GB config: 141 / 26, 13 rows, R57_D7B=all), tf2 (w13t); 12 MBPP + 12 IFEval
prompts per arm, CAP 256, quiet; every prompt clean in all four arms; tokens identical within each config's pair (12 / 12).

| set | arm | whole decode | whole request | first token | tokens | kept per check | cycle | SSD MB / kept token | QOM drops | lifetime peak |
|---|---|---|---|---|---|---|---|---|---|---|
| MBPP | tf1 w13t | 92.6 | 30.1 | 3.33 s | 1,787 | 6.31 | 61.4 ms | 22.7 | 26.6 % | 23.84 GB |
| MBPP | bf1 w13b | 90.0 | 26.0 | 3.32 s | 1,517 | 6.23 | 63.8 ms | 22.4 | 24.5 % | 23.74 GB |
| MBPP | bf2 w13b | 92.6 | 26.2 | 3.37 s | 1,517 | 6.23 | 63.2 ms | 22.4 | 24.5 % | 23.72 GB |
| MBPP | tf2 w13t | 92.7 | 29.6 | 3.37 s | 1,787 | 6.31 | 62.3 ms | 22.5 | 26.6 % | 23.86 GB |
| IFEval | tf1 w13t | 29.6 | 17.8 | 3.11 s | 1,658 | 3.67 | 84.1 ms | 220.7 | - | 23.79 GB |
| IFEval | bf1 w13b | 30.1 | 17.9 | 3.10 s | 1,658 | 3.67 | 83.9 ms | 205.0 | - | 23.66 GB |
| IFEval | bf2 w13b | 31.1 | 18.4 | 3.07 s | 1,658 | 3.67 | 79.6 ms | 205.0 | - | 23.66 GB |
| IFEval | tf2 w13t | 29.9 | 18.1 | 3.06 s | 1,658 | 3.67 | 82.7 ms | 220.5 | - | 23.78 GB |

- **MBPP: no gain.** Pooled whole decode w13b 91.25 vs w13t 92.62 tok/s (-1.5 %), inside w13b's own spread between two
  token-identical runs (90.0 / 92.6, 2.9 %); the w13t controls agree within 0.1 %. w13b's HumanEval gain (+7.6 % over w13t, W24c /
  W24d) does not carry over to MBPP. Its MBPP outputs differ from w13t's (1,517 vs 1,787 tokens; the lossy rule serves what is
  resident, and w13b drops fewer optional uses, 24.5 vs 26.6 %), so its lower whole request (26.0-26.2 vs 29.6-30.1) is mostly
  fewer tokens over the same first-token cost.
- **IFEval: +2.8 %** pooled (30.58 vs 29.74), the same token counts (1,658), 7 % less SSD per kept token (205.0 vs 220.6 MB).
- All lifetime peaks under 24.0 GB on these sets (w13b 23.66-23.74, w13t 23.78-23.86 GB). Checks over 300 ms: 0 on MBPP; 3-6 of
  634 on IFEval in both configs (chat first / last checks, as in W24c).

## Tue 10 Nov 13:38-13:42: W24g = w13b's math speed, 12 math prompts, two passes (data/w24/w24g_report.txt; data 16ce2de)
The orchestrator's lean run for tomorrow's paper (the last unmeasured 24 GB cell): w13b (the card-gated 24 GB config), the 12
math prompts (data/n45/keys.txt line 2, MATH-500, 50-520 tokens), CAP 256, quiet, two passes; no control arm.

| pass | whole decode | whole request | first token | tokens | kept per check | cycle | SSD MB / kept token | lifetime peak |
|---|---|---|---|---|---|---|---|---|
| bg1 | 40.7 | 27.3 | 2.93 s | 3,072 | 6.41 | 155.4 ms | 195.9 | 24.014 GB |
| bg2 | 40.9 | 27.5 | 2.84 s | 3,072 | 6.41 | 154.9 ms | 195.7 | 24.013 GB |

- **Math: 40.8 tok/s whole decode** (pooled), 27.4 whole request, first token ~2.9 s; the two passes agree within 0.4 % and
  their tokens are identical (12 / 12); every prompt ran to the 256-token cap; 1 of 498 checks over 300 ms; no pause, abort or
  cut. Math drafts on the 4-bit DFlash2 drafter here (R57_D7B=all) and reads ~196 MB from the SSD per kept token.
- **The lifetime peak crossed 24.0 GB by 14 MB, on a 520-token prompt** (math500/test/geometry/702: MLX prefill 23.17 GB), so the
  line's prompt-length limit is about 500 tokens, not 560: the gate's 481-token row peaked 23.910 GB, this 520-token one 24.014.

## Tue 10 Nov 23:08: CONV B0 = conversation-cache state sizes + restore mechanics: PASS (data/conv/b0.json; scripts/conv_b0.py, 9f86270)

Kamen's goal via the orchestrator (23:0x): full Claude Code support by Thu 12 Nov 17:00, with the conversation cache on
the critical path. B0 ran one QSA layer and one GDN layer of qwen4_exp (random weights, the real cache classes) on the
recorded Claude Code shape:
- turn 2 restored at 12,288 on the live cache after decode-style writes (208 tokens prefilled);
- a new conversation restored at the stable edge 9,216 (3,084 tokens prefilled).

Both match a cold prefill bit for bit (chunk outputs + all 9 state arrays), and a wrong checkpoint is caught.

Sizes:
- 29,952 B/token logical (KV 24,576 + indexer 5,376);
- per checkpoint: GDN 115.46 MB plus the DFlash2 taps tail, 52.4 MB (the design said 13.1 MB: wrong width);
- clone / restore ~1.2 ms.

Details: docs/CONV_CACHE_DESIGN.md section 11. Next: B1 on the real engine (T1), Wednesday.

## Wed 11 Nov 01:01-01:13: CONV T1 = the conversation cache on the real engine: FAIL, 5 of 12 hits bitwise; cause of 4 found (data/conv/t1/verdict.json; conv/t1_verdict.py)

**Setup.** Leap ran serve.py `--mem 32` with the cache hook (conv/convcache.py, 7eb3401) and CONV_CACHE_CHECK=1. With the
check on, every hit is compared with a cold prefill in the same process. The load was the 45 recorded Claude Code and Codex
requests.

**The cache's choices matched the prediction exactly.**
- 12 hits, every one at its predicted restore point.
- 30.2 % of the prompt tokens were prefilled.
- Restore took 3-15 ms; the cache peaked at 2.07 GB.

**Only 5 of the 12 hits were bitwise equal to cold.**
- 4 FAILs had different logits. Each had a tail of 116-197 new tokens, which the module ran chunk-major, while the cold
  prompt ran it in PF-A. On short MoE calls the two pick different MLX gather kernels (E = 512 vs the call's unique
  experts).
  - Fix: choose PF-A by the whole prompt, as the loop does.
- 3 FAILs differ in one state array of 122 while the logits and taps are equal. The cause is not yet known; the rerun names
  the array.

Leap reruns with the fix at ~02:00. Details: docs/CONV_CACHE_DESIGN.md section 14.

## Wed 11 Nov 01:3x-01:45: CONV T1b = the kind A fix holds; the last kind is layer 35's attention keys after a copy (data/conv/t1b/verdict.json)

**Setup.** Leap reran T1 with the module choosing PF-A by the whole prompt (now conv/convcache.py, sha256 00488b23) and the
FAIL line naming each differing leaf.

**Results.**
- 12 hits, 7 PASS, 5 FAIL.
- The 4 logit FAILs of T1 are gone; all 6 Codex hits pass.
- The 5 FAILs are one array: layer 35's attention keys, on every Claude Code hit from the conversation's first branch
  copy on. Logits, taps and the other 121 arrays are equal.
- Warm prefill: 2.8 s for 116 new tokens, 3.8-5.6 s for 196-425, 12.1 s for 3,084 (a cold 12,302-token prompt: 28.0 s).

**Next.** A per-row drift diagnostic (docs/CONV_CACHE_DESIGN.md section 15) goes into T1c, after the night's long-context
runs, to show where those rows change.

## Wed 11 Nov 03:57-04:09: CONV T1c = the branch copy's clone rewrote one value (data/conv/t1c/verdict.json)

**Setup.** Leap reran T1 with the per-row drift diagnostic (conv/convcache.py 134e811, sha256 e3434b13).

**Results.**
- The same 12 hits, 7 PASS, 5 FAIL.
- No stored entry changed between requests (0 drift lines): the decode does not touch the stored rows.
- Both branch copies come out with row 0 of layer 35's keys changed, before any prefill. Every FAIL is that one row.

**Cause and fix.** The copy cloned arrays as `x + 0`, which turns -0.0 into +0.0 (and rewrites NaN bits). The values stay
equal, so outputs matched, but the bits did not. convcache now copies bits through an integer view, the GDN checkpoints
too (they used the same rule). A unit test with planted -0.0 and NaN values passes 10/10. Details:
docs/CONV_CACHE_DESIGN.md section 16. Next: T1e, the same run on the fix with branches on.

## Wed 11 Nov 04:10-04:23: CONV T1d = the fallback (a branch is a miss) on the real engine: PASS, 10 of 10 hits bitwise (data/conv/t1d/verdict.json)

**Setup.** Leap ran T1 on b855883 with CONV_CACHE_BRANCH=0: a request that would copy from another entry runs a cold
prefill instead.

**Results.**
- 10 hits as predicted (data/conv/t1_predicted_branch0.json), all 10 bitwise vs cold. 0 drift lines, 0 server errors.
- 41.3 % of the prompt tokens were prefilled (30.2 % with branches on): the 2 branches ran cold, 28 s each.
- Live hits: restore 58-93 ms, then 2.9-5.7 s for 116-425 new tokens.

**Next.** This is the fallback for the cached agent test. The copy fix (T1c above) needs its own T1 (T1e) before
branches come back. Details: docs/CONV_CACHE_DESIGN.md section 17.

## Wed 11 Nov 05:08-05:20: CONV T1e = the copy fix on the real engine: PASS, 12 of 12 hits bitwise with branches on (data/conv/t1e/verdict.json)

**Setup.** Leap ran T1 on 3c068d7 (bitwise clones), with branches on and the serve.py the cached agent test will use.

**Results.**
- 12 hits as predicted, all 12 bitwise vs cold, both branch copies included. 0 drift lines, 0 server errors.
- 30.2 % of the prompt tokens were prefilled (41.3 % with branches off).
- Every restore's census shows the cause: a single -0.0 in row 0 of layer 35's keys (bits 0x8000), which the old clone
  turned into +0.0.

**Next.** The cached 32 GB agent test runs on 3c068d7 with branches on, after Leap's 64K / 128K runs. Details:
docs/CONV_CACHE_DESIGN.md section 18.

## Wed 11 Nov 07:01-07:21: CONV agent32c = the cached 32 GB Claude Code agent run (Leap's 32c_1111): 117 of 118 requests hit, tasks 2.8x faster (data/conv/agent32c/summary.json)

**Setup.** Leap ran EvalSuite's agent test (30-case probe + 12 tasks) with the conversation cache on: 3c068d7,
branches on, a 2.5 GB budget, and no per-hit cold check.

**Results.**
- 117 of 118 requests hit: 106 live and 11 copies (each task's first request after t01). The only miss is the first
  request.
- 9.3 % of the prompt tokens were prefilled, a median of 664 new tokens per request.
- First token: median 6.3 s against 31.9 s cold, p90 12.7 s, max 36.4 s. Peak 37.07 GB.
- Probe 29/30, tasks 11/12, 0 server errors; the tasks took 916 s against 2,595 s cold (2.8x) [K Leap].

**Next.** T1-45 (the 45 GB chunk-major T1) on 3c068d7. This run's exactness rests on T1e (12/12 bitwise). Details:
docs/CONV_CACHE_DESIGN.md section 19.

## Wed 11 Nov 07:21-07:45: CONV T1-45 = T1 at 45 GB: PASS, 12 of 12 hits bitwise on the chunk-major path (data/conv/t1_45/verdict.json)

**Setup.** Leap ran T1 on 3c068d7 at --mem 45 (branches on, CHECK on, the same 45 requests). At 45 GB every prefill
takes convcache's chunk-major path over the engine's PF-D prefill, and convcache takes the GDN checkpoints itself.
T1 through T1e ran PF-A at 32 GB.

**Results.**
- 12 hits as predicted, all 12 bitwise vs cold. 0 drift lines, 0 server errors; 30.2 % of the prompt tokens
  prefilled.
- The census is the same as at 32 GB: one -0.0 in row 0 of layer 35's keys at every restore.
- Long prefills run at PF-D's ~138 tok/s (PF-A at 32 GB: 369-444), so the misses and copies took 1.9-3.2x longer.
  Short ones match. Peak 50.02 GB.

**Next.** Leap's cached 45 GB agent run. Details: docs/CONV_CACHE_DESIGN.md section 20.

## Wed 11 Nov 07:47-08:11: CONV agent45c = the cached 45 GB Claude Code agent run (Leap's 45c_1111): 117 of 118 requests hit (data/conv/agent45c/summary.json)

**Setup.** As agent32c, at --mem 45 (PF-D prefill): 3c068d7, branches on, a 2.5 GB budget, and no per-hit cold check.

**Results.**
- 117 of 118 requests hit: 107 live and 10 copies. The only miss is the first request, 72.5 s cold (32 GB: 20.4 s).
- 8.8 % of the prompt tokens were prefilled, a median of 746 new tokens per request.
- First token: median 6.8 s (32 GB: 6.3 s), p90 26.2 s (32 GB: 12.7 s, since PF-D is slower on big chunks), max
  72.5 s. Peak 49.43 GB, no swap growth.
- Probe 29/30. The task checks gave 12/12, but EvalSuite scores t10 a fail because it searched outside its folder,
  giving 11/12. 0 server errors; the tasks took 1,097 s (32 GB cached: 916 s) [K Leap].

**Next.** Exactness at 45 GB rests on T1-45. Details: docs/CONV_CACHE_DESIGN.md section 21.

## Wed 11 Nov 14:12-14:25: W24FT = first token at 24 GB: P2 v0 (LOSSY) 3.08 -> 1.12 s median, under Strata's 1.46 s; P3's fold exact (data/w24/w24ft_report.txt, w24ft2_report.txt)

**Setup.** An idle-GPU hour by day, with the owner's apps open, under a share lock. The shipped 24 GB card (w13b as gated)
ran on snapshot w24ft: the card's screen plus P3's fold (bc04e37) and P2 v0 (ff0215d), applied as patches. The 12 IFEval
prompts (P 37-101), CAP 256, one process per arm, in the order below.

| arm (order) | TTFT median | mean | range | prefill SSD per prompt | tokens / logits hash vs control | lifetime peak |
|---|---|---|---|---|---|---|
| control: the card as gated (1st) | 3.08 s | 3.12 | 2.53-3.64 | 20.4 GB | - | 23.80 GB |
| + PF-D v4 + read-ahead, exact (2nd) | 3.32 s | 3.34 | 2.71-3.81 | 19.4 GB | 12/12, 634/634 | 29.50 GB |
| + PF-D + fold (P3), exact (3rd) | 3.11 s | 3.20 | 2.76-3.73 | 19.4 GB | 12/12, 634/634 | 29.30 GB |
| + P2 v0, 16 rows, LOSSY (4th) | 1.66 s | 1.83 | 1.10-3.17 | 12.7 GB | 2/12, 0/584 | 22.90 GB |
| control again (5th) | 3.09 s | 3.09 | 2.58-3.51 | 20.4 GB | 12/12, 634/634 | 23.70 GB |
| + P2 v0, 13 rows, verify, LOSSY (6th) | 1.18 s | 1.30 | 0.87-1.93 | 12.9 GB | 2/12, 0/582 | 23.00 GB |
| + P2 v0, 13 rows, plain, LOSSY (7th) | **1.12 s** | 1.25 | 0.82-1.89 | 12.8 GB | 2/12, 0/583 | 23.00 GB |

Strata at 24 GB on the same 12 prompts (StrataEval results_low24 REPORT): median 1.65 s (arm B) and 1.46 s (arm D).

**Readings.**
- **P2 at the card's own row cap is the 24 GB first-token lever.**
  - At 13 rows a chunk is 130 ids, within the 141 base slots, so it stays on the sync-free gate.
  - The median falls from 3.08 to 1.12 s (-64 %), 23 % under Strata's better arm.
  - It stays under the 24 GB line (23.00 GB), and the decode is unchanged (whole decode 31.8-31.9 vs 31.0 tok/s).
  - At 16 rows a chunk is 160 ids, which leaves the sync-free gate: 1.66 s.
- **P2 is LOSSY:** tokens match the exact prefill on 2/12. It needs EvalSuite's same-scores gate before any chat config
  uses it.
- **P2 is now read-bound:** about 2.2 GB per 13-row chunk (8.5-18.8 GB per prompt) at about 11 GB/s. The next lever is
  bytes: keep a prompt's experts across its chunks, and MSB-only reads for the low ranks.
- **PF-D is not for 24 GB.** It is slower than the card's own prefill (3.32 vs 3.08 s) and adds 5.7 GB of peak (29.5 GB).
- **P3's fold is exact** (12/12 and 634/634 vs both the control and PF-D). It takes PF-D from 3.32 to 3.11 s (-6 %) in
  one order; tonight's 45 GB ABBA settles it.

**Next.** Tonight's 45 GB A/B (P3 ABBA, FAST without / with the fold, P2). Then the plan for EvalSuite's P2 gate at 24 GB.

## Wed 11 Nov 18:37-19:36: D2 = a 2-bit trunk copy + resident experts as a training-free chat drafter: KILL (data/d2/d2_score.json)

**Setup.** Kamen's ask (via the orchestrator): a better drafter without retraining, for any model, toward 98 %. D2 is
FableGod BREAK_THE_WALL section 4.4. Pre-registered before any run (docs/PREREG_D2_2026-11-11.md, with amendment 1 for a
definitive verdict).
- The swap: every trunk linear converted to 2 bits (532 linears: GDN, attention, indexer, HC mixers and inject, PLE;
  dequantize, then requantize, group 64). Router and shared expert unchanged.
- Experts: a fixed resident set by expert id (the 45 GB base pool's size per layer, by popularity), with drop semantics.
- Harness: teacher-forced on the decision map's windows (code 24, IFEval 30, multi 24, agent 12), against the 125B's
  exact top-1, with no 125B run. Breakthrough's map and scoring.

**Results.** Agreement on chat (IFEval + multi), kept per 16-row check, and agreement on code:

| arm | chat agreement | chat kept / check | code agreement |
|---|---|---|---|
| D2: 2-bit trunk + resident | **0.026** (IFEval 0.025, multi 0.027) | 1.03 | 0.066 |
| 2-bit trunk, all experts | 0.056 | 1.06 | 0.110 |
| 3-bit trunk + resident | 0.442 | 1.78 | 0.920 |
| 4-bit trunk + resident | 0.572 | 2.31 | 0.987 |
| store trunk + resident (control) | 0.580 | 2.35 | 0.990 |
| ops path, all experts (harness floor) | 1.000 | - | 1.000 |

**Validity.** The gate passed at 100 %, the swap census was exact, and the determinism repeat was bit-identical. The agent
set is excluded: its own gate failed at 83.4 % against the stored reference, because its windows start mid-conversation
and the map prefills only the window.

**Reading.**
- KILL: naive 2-bit conversion destroys the trunk (44-49 % weight error per matrix).
- 3 bits keeps code (0.92) but not chat (0.44).
- A popularity-based fixed expert set, drawn from code runs, caps chat at about 0.58 even with the full trunk.
- A training-free drafter from the 125B itself needs a trunk of 4 bits or more and a domain-matched expert set; that is
  the near-copy, whose cost per drafted token is the known wall (DRAFTER_98 section 3.4).
- Breakthrough's independent read (79cb1b5) matches every printed digit on all 8 arms. Its kept per check against the
  exact top-1, chat / code: resident 2.35 / 14.35, 4-bit 2.30 / 13.84, 3-bit 1.78 / 8.55, D2 1.03 / 1.07.

## Wed 11 Nov 14:41-19:51: EvalSuite's same-scores gate on P2 at 24 GB: PASS (EvalSuite 8adeb09, docs/RESULT_p2gate_2026-11-11.md)

**Setup.** The 24 GB card w13b as gated plus P2 (R57_P2=256, R57_P2_ROWS=13, R57_P2_VERIFY=0; data/stack/card_w24p2_env.txt
6ff0296f; screen w24ft), against the card as gated. EvalSuite's pre-registration (12007aa) set the scope: every card-gate
prompt of 256 tokens or fewer, that is IFBench 150 / 150 and GPQA Diamond 67 / 100. The controls were the 5 shortest GPQA
prompts above 256 tokens (258-269).

**Results.**

| set | P2 | card as gated | delta | McNemar p (arm-only / card-only) | line |
|---|---|---|---|---|---|
| IFBench (150) | 39.3 % | 40.7 % | -1.33 pt | 0.80 (7 / 9) | PASS |
| GPQA Diamond (67 short prompts) | 55.2 % | 58.2 % | -2.99 pt | 0.63 (1 / 3) | PASS |

- The gate: PASS (delta above -5 pt and p >= 0.05 on both sets).
- The 5 controls were token-identical, so P2 leaves longer prompts alone.
- runs.json p2 was above 0 on all 217 short prompts (median 6 chunks on IFBench, 16 on GPQA) and 0 on the controls.

**Information only.** The gate ran by day through 12 heat freezes.
- Median first token: IFBench 1.60 vs 3.31 s; GPQA 3.12 vs 3.75 s.
- Peak 23.25 vs 25.05 GB on the same GPQA keys.
- Decode 27.9 vs 29.3 tok/s, which is not a controlled comparison: the card's figure is from a night run. In W24FT's
  same-run arms, P2's decode was 31.8 against the control's 31.0.

**Next.** One clean ABBA timing at 24 GB under a quiet lock tonight (docs/PREREG_W24T_2026-11-11.md). If it holds near
1.1 s, P2 at 24 GB goes into the next package (1.2), on Kamen's word.

## Thu 12 Nov 00:31-00:37: W24T = the clean 24 GB first-token timing (PREREG 23af00c): P2 median 1.12 s vs the card's 3.11 s, under Strata's 1.46 s (data/w24/w24t_report.txt)
ABBA under one quiet lock (NO-SWAP 34 GB), the 12 IFEval prompts, CAP 256. The pooled card is 3.11 s median (mean 3.21,
range 2.56-4.41). The pooled P2 (13 rows, plain) is **1.12 s median (mean 1.25, range 0.82-1.91), 0.36x the card**.
Strata on the same prompts: 1.46 s (D) and 1.65 s (B). Validity: the controls are identical (12/12, 634/634), P2 is
deterministic (12/12, 639/639), and P2 engaged on every prompt. Decode 31.3-31.4 (P2) vs 30.3-31.2 tok/s (card). Peak
23.00 vs 23.70 GB. The full write-up waits for the Thu 17:00 reset.

## Thu 12 Nov 01:07-13:30: DEEPDRAFT capture night 1 (PREREG 52b8571 + amendments 754a8a8 / d24597a): 604,474 answer tokens on fresh Python prompts
2,277 Magicoder Python prompts (eval-overlap filtered) at C 160, picked from NO-SWAP 42 GB. 604,474 answer tokens, 32.80
GB of features (h, wd, top-32) in ~/FableGod/deepdraft/capture/n1112/; every prompt on-path 1.0000. Stopped by the 13:30
hard stop as planned (the prompt in progress is redone next night). 59 heat freezes, 12,711 s frozen (28 % of the run),
most after Kamen's 07:08 fan minimum (FAN_CAP 2000). Pace about 68K answer tokens per hour with normal fans and about
29-43K per hour at the minimum. Verdict lines only; the full write-up comes after the Thu 17:00 reset.

## Thu 12 Nov 15:0x-15:3x: DEEPDRAFT pre-proof (PREREG ef380e3): a learned per-token stand-in for the trunk state read by the unchanged MTP head b5: KILL (data/deepdraft/preproof/eval.json)
A 36M-parameter recurrent helper, trained on the capture's 604K tokens (1/3, 2/3 and all), predicts the trunk's widened
stream from the last exact state plus the drafted tokens. Fed to b5 at depths >= 2 on the 24 held-out HumanEval+ windows,
it gives 17.9 % at depths 23-30 at best (f2), against b5's own chain at 85.1 % (gap recovered -5.3 to -6.8; KILL below
0.15). The stale control gives 13.5 %. MBPP is the same. b5 cannot read approximate trunk states (cosine about 0.7). The
route needs the reader trained with its stand-in (PLAN step 3). The eval peaked at 10.5 GB, over the 8 GB day limit;
swap stayed flat.

## Thu 12 Nov 17:0x-19:0x: DEEPDRAFT step 3 first test (PREREG a73438e): b5 trained with D-22's chain to depth 30 on capture night 1: INCONCLUSIVE (data/deepdraft/ttt30/eval.json)
The MTP head b5 was trained for 600 more steps with D-22's training-time-test chain (KL to the 125B's top-32 at every depth,
accept-until-fail), now to depth 30, on night 1's Magicoder answers; the 125B is untouched. On the 24 held-out HumanEval+
windows, depths 23-30 rose from 85.1 % to 87.3 %, between KILL (< 87) and CONTINUE (>= 90). Depth 0 stayed (97.7 -> 97.6).
The gain grows with depth (7-10 +1.2, 15-22 +2.2, 23-30 +2.3 pt) and transfers: MBPP 23-30 rose from 63.3 % to 68.5 %. Kept
per 32-row check went from 12.55 to 13.12 (HumanEval+) and from 6.22 to 6.54 (MBPP). The run used about 19 % of night 1's
tokens, once. Next: about 3,000 steps (one full pass), pre-registered, with the curve at 600 / 1,500 / 3,000 steps.

## Fri 13 Nov 02:16-02:26: the 45 GB first-token A/B: P2 v0 (LOSSY) 0.47 s median vs the shipped chat's 2.44 s and Strata's 0.83 s; P3's fold exact, -2.9 % (data/p3/p3_report.txt)
ABCCBA under one quiet lock (NO-SWAP 50 GB), on the 12 IFEval prompts at CAP 256, one process per arm. The order: control
(the 45 GB chat as shipped: zs + PF-D v4 + read-ahead), P3 fold, P2 plain twice, fold, control. Then FAST without and with
the fold, and P2's verify scope (back to back, info only).

| arm | first token, median | mean | range | tokens / logits hash vs control | SSD per prompt | peak |
|---|---|---|---|---|---|---|
| control (1st + 6th) | 2.44 s | 2.38 | 1.71-2.91 | 12/12, 721/721 | 9.6 GB | 43.70 GB |
| + P3 fold, exact (2nd + 5th) | 2.37 s | 2.31 | 1.79-2.77 | 12/12, 721/721 | 9.6 GB | 43.70 GB |
| P2 v0 plain, 16 rows, LOSSY (3rd + 4th) | 0.47 s | 0.43 | 0.23-0.66 | 2/12, 0/581 | 2.1 GB | 42.40 GB |
| P2 v0 verify scope, LOSSY (9th) | 0.51 s | 0.47 | 0.25-0.67 | 2/12, 0/632 | 2.1 GB | 42.30 GB |
| FAST + read-ahead, LOSSY (7th) | 1.89 s | 1.81 | 1.27-2.12 | (the FAST pair's own control) | 4.9 GB | 43.30 GB |
| FAST + read-ahead + fold (8th) | 1.78 s | 1.72 | 1.25-2.06 | 12/12, 705/705 vs FAST | 4.9 GB | 43.10 GB |

- **P2 at 45 GB: 0.47 s median, 0.36 s (43 %) under Strata's 0.83 s on these prompts.** The exact path is still about 2.9x
  Strata's. P2 reads 2.1 GB per prompt instead of 9.6 GB.
- **P2 is LOSSY** (tokens 2/12, as at 24 GB). The 24 GB same-scores gate (13-row chunks, EvalSuite 8adeb09) does not
  cover the 45 GB setting (16-row chunks on the 45 GB chat), so it needs its own gate before it ships.
- **P3's fold is exact** in both fold arms but small here: -2.9 % median (-0.07 s), against the "up to -0.35 s" estimate.
  At 24 GB it gave -6 %. The FAST pair ran back to back, which favours the second arm, so its -5.8 % is not a clean figure.
- **Validity.** No heat pause or freeze during the arms (one fan curve step at 02:19). The two controls agree (12/12,
  721/721). Decode medians are 47.2-48.9 tok/s in all six alternated arms, but P2's answers differ (median 104 tokens
  vs 160), so decode is not a matched comparison. Peaks are 42.3-43.7 GB, under 45.

## Fri 13 Nov 02:29-10:38: DEEPDRAFT step 3 second test, TTT30B (PREREG f341ed9): one full pass over capture night 1: INCONCLUSIVE on the lines (89.7 % at depths 23-30; CONTINUE >= 90), more data PAYS (+2.3 pt over t30a) (data/deepdraft/ttt30b/eval.json)
The MTP head b5 was trained with t30a's recipe (D-22's chain to depth 30, KL to the 125B's top-32, accept-until-fail) for
one full pass over night 1's Magicoder answers: 4,206 steps, every anchor once, against t30a's 600 random windows (~19 %).
The 125B is untouched. Same held-out windows and harness as t30a; agreement with the 125B, in %:

| head | HumanEval+ depth 0 | 7-10 | 15-22 | 23-30 | kept / 32 | MBPP depth 0 | 23-30 | kept / 32 |
|---|---|---|---|---|---|---|---|---|
| b5 (start) | 97.7 | 90.5 | 86.8 | 85.1 | 12.55 | 93.5 | 63.3 | 6.22 |
| t30a (600 random-window steps, annealed) | 97.6 | 91.7 | 89.1 | 87.3 | 13.12 | 93.6 | 68.5 | 6.54 |
| t30b after 600 steps (mid-schedule) | 97.6 | 89.9 | 86.1 | 84.2 | 11.81 | 93.5 | 64.7 | 6.14 |
| t30b after 1,500 | 97.8 | 91.5 | 88.4 | 86.7 | 13.28 | 93.7 | 68.6 | 6.49 |
| t30b after 3,000 | 97.8 | 92.8 | 90.2 | 88.7 | 13.62 | 93.9 | 70.9 | 6.76 |
| **t30b, one pass (4,206)** | 97.9 | 93.1 | 91.0 | **89.7** | 13.83 | 94.0 | **72.1** | 6.92 |

- **Lines: INCONCLUSIVE**, 0.34 pt short of CONTINUE (>= 90 % at 23-30 with depth 0 kept; depth 0 moved +0.2 pt).
- **More data PAYS:** +2.3 pt over t30a at HumanEval+ 23-30 (line +1.0), and MBPP 23-30 72.1 % vs t30a's 68.5 %.
- **The gain grows with depth** (HumanEval+ vs b5: +2.6 at 7-10, +4.2 at 15-22, +4.6 at 23-30; MBPP +8.8 at 23-30), and
  kept per 32-row check rises 10-11 % on both sets.
- **The curve still climbs at the end of the pass** (+1.0 pt over the last 1,206 steps), but those steps also finish the
  anneal, so the end slope is not a clean data effect.
- **Scale, roughly:** ~5.3x the data for +2.3 pt is about +1 pt per doubling; at that rate data alone does not reach
  ~97 % at depth 30. Next: more capture nights, and the capacity side (PLAN step 3's carried state and the capacity
  sweep), pre-registered first.
- **Validity.** b5 and t30a re-score exactly as in t30a's eval. One resume at step 1,271 restored the weights, Adam's
  moments and the step. Pooled cells only, one run per arm: a direction, not a significance claim. Heat pauses
  throughout; no timing claims.

## Fri 13 Nov 15:13-: DEEPDRAFT step 3 third test, TTT30C (PREREG 22bf65a): smokes done, t30f training (in progress) (data/deepdraft/ttt30c/)
The arms are t30b's exact run plus one change each: t30f adds a feature loss, and t30x2 / t30x4 add a dense branch
(capacity x2.1 / x4.4). The slot opened at 15:08, when EvalSuite's 45 GB P2 gate ended.
- **t30x2 smoke:** both exact-equality checks EQUAL (the lambda-0 path and the zero-init branch each give b5's loss,
  0.3548715). The MLX peak was 22.4 GB at step 1, over its 20 GB guard, so the arm was skipped for a re-plan. The branch
  adds 6.8 GB to t30b's 15.6 GB.
- **t30f smoke:** EQUAL, lambda 0.09164, peak 15.6 GB. Training since 15:16, beside the capture.
- **Next:** t30x2, then t30x4 (a linear projection puts it near 36 GB), each with its guard re-set from its own smoke and
  run with the capture off. No lines are read yet.
