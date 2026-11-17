# WEEK98: the 98 % deep drafter, Sat 14 Nov -> Sat 21 Nov 2026

> (a quote from the project owner, kept private)
itself or the method for it being certified that it will work in a week, because I don't have more time than that to
spare to show this part of the project to my dad. I mean, so yeah, basically we must be able to find the thing which
unlocks 98% accuracy at 30 tokens deep. By then we must know what it is, and either I don't care if in a week we just
know the principles of how everything will work and it's certain, and we still have a week of building everything. I
don't care about that. I just need to get that."

## KAMEN, Tue 17 Nov 14:1x (after the deep pass): "You've found 600-800. Then do it ... make sure it's done for Saturday"
(He objects to every "physical limit" phrasing. The walls are the limits of the KNOWN methods, not of the Mac.)
- **Orchestrator 14:1x: BUILD THE STACK NOW, all four pieces, not after Saturday.**
  1. Draft fixed cost: MoonShots UNPARKS the persistent fused tree level for the Q8H32 head (0.55 -> <= 0.10 ms per pop,
     bitwise), by Friday. That's 211 -> ~261 on Saturday's 16-row config if it lands.
  2. Expert phase: Breakthrough's MoE block (megakernel decision at 18:00).
  3. Check: Leap's 32-row line + the 16-row modules (slots #2-#4).
  4. Lighter head: MTP Head starts S22 (the path-A QAT head, ~0.19 ms per level) NOW and asks for a GPU window.
  Every piece that lands is measured and goes into Saturday's number. No promise of 600 by Saturday.
- 14:3x S22 placement (MTP Head; pre-registered c9cdf7d, PASS kept >= 0.95x full t30b at tree-16 on both sets):
  - SMOKE (~15 min, <= 16 GB) from ~16:30, in any gap where Kamen has been away >= 15 min;
  - qa (~2.8 h, ~16 GB) chained on the Wed-night CERTIFY98 capture's end marker, latest start Thu 05:00;
  - qc not this week.
- 14:4x MoonShots' PERSISTENT draft kernel (PK; NewIdeas QUEUE M-46 e129db5; PK_PLAN.md 3720b44):
  - Lines: bit test PK == SF on all 120 real trees; quiet timing f <= 0.10 ms PASS, >= 0.25 KILL; a non-bitwise skeleton
    first (f <= 0.15 GO).
  - Flag R57_TREE_PK=1 (default off), bitwise, so it can go on after Thursday's key test with only the timing re-run.
  - ETA: skeleton Wed ~09:00; flag to BigMoE Thu evening for runner 69's timing; verdict Fri.
  - CORRECTED gain: at f 0.10 a Q8H32 pop is 0.66 ms (SF 1.11), a tree-16 block 9.9 ms, so HE+ 211 -> ~240 (x1.14). The
    deep pass's 261 used path A's 76.7 MB head, which failed C2. Confidence: skeleton ~60 %, bitwise by Fri ~50 %.
- 14:4x Exact headline: Leap is asked to confirm ref2_switches_v3 (REF2_BOF + REF2_HC_B, bitwise, ~-3..-5 ms per exact
  check). BigMoE adds its key test to tonight's chain after M4, ON in (ii) on PASS.
- 14:5x Leap:
  - ref2_switches_v3 is applied as b4331e4 (default OFF). Its in-loop ON / OFF bit test never ran, so BigMoE's key test
    must compare tokens + logits bitwise ON vs OFF.
  - CORRECTION: tgl16eu's -2.1 ms came from c25 arms modelling 1c tiers on every row (failed V2). Under g1c (comments
    only), code rows carry few tier-3 pairs, so ~-0.5..-0.9 ms [G]; GDN16 carries over. The deciding number comes from a
    g1c-shaped check (BigMoE's knob ON vs OFF, or a tiers-OFF 16-row arm in slot #2, once BigMoE names g1c's code-row
    expert path).
  - Saturday's gated lossy is therefore more likely ~205-215.
- 15:0x BigMoE + Leap: g1c's HE+ code rows DO run the 1c tiers (lambda 0.016, S 0.98; TIERS_SETS = humaneval). Only
  '#' rows run COMQOM; MBPP / HE-X / IFEval run the base path.
  - So tgl16eu's HE+ gain carries over. Leap withdrew most of his 14:5x correction; slot #2's F16g arms now run at 0.016
    (1f605c4); tiers-OFF arms F16n / F16nx go to slot #3.
  - **Orchestrator flag to EvalSuite (15:0x): V2 FAILED 1c-tiers on HE+ (lambda 0.018 setting), yet g1c carries 1c
    tiers at 0.016 on HE+.** Confirm g1c's tier setting is not the failed one, or re-pick / strip it before G1's B0.
- **14:49 EvalSuite CONFIRMED: g1c carries the IDENTICAL failed tier setting on HE+** (R57_TIERS=1c, LAM 0.016, S 0.98,
  SETS humaneval, + COMQOM). All 8 G1 envs carry 1c tiers on HE+. G1's chain was STOPPED at 14:49 (no GPU spent).
- **Orchestrator 15:0x: G1 SKIPPED this week** (a dated deviation from 5d, covering NewIdeas).
  - The tiers buy ~-0.3 ms of bytes, and the extra tgl16eu gain they enable is ~-1.5 ms (~3 %): not worth a likely
    stack tier-1 fail or ~5 h of 45 GB GPU.
  - **Saturday's stack = t30b Q8H32 + tree-16 + CC(1) on lossy-45's base path + sync-free + GDN16 + exact tgl16e, NO
    tiers.** Leap prices it on the tiers-OFF arms (F16n / F16nx, slot #3).
- **15:00 BigMoE V3KT PASS:** REF2_BOF=1 + REF2_HC_B=1 vs OFF (Leap b4331e4) are BITWISE identical (tokens + every
  check's verify logits; 4 code keys, 628 tokens, 90 checks; BOF 11,174 / HC_B 11,060 fires, drift 0). Scope is 16-row
  checks. **ON in Saturday's exact headline (ii).** Tonight's chain is M4 only.
> (a quote from the project owner, kept private)
  replies are BIG-only from now on.

## KAMEN, Tue 17 Nov 11:4x (verbatim, on the compiled check): MODEL-SPECIFIC WORK
"I don't know how I feel about building layers which are only model specific-there is no benefit from it for other
models"
- Orchestrator 11:4x:
  - Most of the gain is in GENERAL methods: stream / MMA overlap in the expert step, folding router / shared / combine,
    SLC prefetch, tree checks, sync-free drafting. They apply to any MoE model on Apple Silicon.
  - Only the final tuning is model-specific. So the MoE-block proof is built as a TEMPLATE (shapes, experts, top-k,
    bits, tiles as parameters + a per-model config), so a new MoE model (Qwen 4 next) needs a config, not new kernels.
  - Fixes that belong in MLX itself go upstream (as with the sorted-gather bug #4632).
  - Asked of Breakthrough 11:4x, with a heads-up if the template costs > ~half a day extra. Kamen can still say to drop
    model-specific work entirely.

## KAMEN, Tue 17 Nov 11:0x (verbatim, after the morning's closures and the ~200-220 Saturday range)
"Bro. I'm not accepting we can't do that. These are actual thinking problems. Which require more smart new ideas with
much less tests than anything"
- **Orchestrator 11:1x: BREAK98c** (~/FableGod/deepdraft/BREAK98c.md), a desk sprint to ~12:30. It attacks the
  frame's ASSUMPTIONS (not its walls): five thinkers + MoonShots. Tests only if <= 10 min and decisive.
- Same turn:
  - MoonShots' draft kernel line PARKED at sync-free (fusing nets ~-1.5..-2.2 ms per block for 2-3 days); MoonShots
    joins BREAK98c.
  - **Breakthrough's expert rewrite tgl16eu: YES for Saturday's stack** (0.843x / 0.830x of the expert one-call at
    32 / 16 rows, ~-3.0 ms on the 16-row check; the TIER-4 UPGRADE joins 3-bit pairs to their expert's 4-bit segment
    at no extra bytes; Leap reviewed it as acceptable on FAST). It enters as a flag before Thursday's tier 1, if Leap's
    c25 A/B confirms by Wed evening. Quality is judged inside the stack's gate; the first bisect turns it off.
    w4a8u: NO for Saturday (3 % expert-output error).
- 11:1x EvalSuite (with BigMoE, 10:56): the CERTIFY98 count check on t30b's STORED format keeps the +-3 % check valid
  with no separate amendment. The form is R57_T30B_FMT=stored, R57_TREE_K=1, 31 nodes, exact acceptance, TF32 default;
  identity is checked per window.
  - (b)'s lagging head-pass follower could queue up to ~4.4 GB of wd on disk. The orchestrator agreed to cap it at ~1 GB
    (the capture waits), to hold the 100 GiB floor.
  - G1's chain had a gate bug (it read V2's verdict in the single-order form). Fixed and re-armed 10:03. V2 resumed
    11:08; verdict ~12:00-12:30.
- **11:2x BREAK98c, Fable FAR (IDEAS_1117c_prefix_eyes.md): PREFIX EYES** (assumption 3).
  - Mechanism: the head cross-attends, at every level, into the 125B's EXACT prefix K/V at 2-3 late attention layers.
    Inside copied spans, the prefill state replaces the carried state. ~12 MB per pop, no 125B pass, exact (drafter
    input only).
  - Number: the drift gap is the lever (chain own 13.77 -> true 20.77; LAB + ties 26.79). G 0.8 + a hidden draft +
    a 40 ms check gives ~600; G 0.5 at K64 ~364-426.
  - Decisive test: Sandbox98 kvx vs ctl (a zero-init cross-attention over the frozen 2B trunk's layer-3 / 4 prefix
    K/V; ~40 min). PASS G >= 0.3 at bands 9-30 AND tree-32 kept >= +8 %; KILL G < 0.1. Dispatched 11:2x, to start when
    memory allows (V2 runs to ~12:30).
  - Also from FAR: assumption 4 is worth only ~x1.05-1.09 on HE+. Old-suffix reuse after a miss was TESTED: KILL (13 %
    right per row vs 58-63 % fresh).
- **11:2x BREAK98c, Fable B (IDEAS_1117c_carryover.md): assumptions 1 + 2 are THEOREMS of (greedy walk) x (unified
  memory), not choices.**
  - No on-chip store holds a pass's bytes (the SLC is tens of MB against a ~5 GB working set).
  - After the walk, every unreached row has a wrong token in its prefix: 70.7 % (HE+) / 91.0 % (MBPP) of a 127-row
    tree's rows are such bets [K].
  - The carry-overs that exist (prefix KV / GDN, the bonus token, 45 GB residency) are all in use.
  - The one missing row is the bonus-token row at the walk's end node: in the head's top-8 there 80.8 % HE+, but in
    the tree by construction 0 %. Expanding the top-8 at the ~4 lowest-margin chain nodes (+28 rows) gives ~+1.3 kept
    for ~4-5 ms: under the 1.8 ms-per-kept bar, and EAGLE-2 already approximates it. Not run (orchestrator: small).
  - Verdict: the frame breaks only at positions per weight read (the drafter, assumptions 3-6) or at the per-row price
    (7-8).
- **11:2x Kamen's hunch:** Taalas HC1 / ChatJimmy (Llama 3.1 8B hardwired into silicon; ~17K tok/s claimed, ~14.4K
  measured independently): could a SOFTWARE version of weight-etching work on this Mac? Forwarded to ZERO for its
  BREAK98c lens: SLC-resident drafter, ~2-bit codebook weights with in-register decode, a compiled check.
- **11:2x Leap: the c25 SMOKE of the new arms (11:18) = wiring PASS; real numbers come from slot #2 tonight.**
  - Bitwise as designed: GDN16, exact tgl16e, the 32-row one-launch HC read, the 32-row bitwise stack and its
    compositions.
  - ROUGH, not for quoting: F16gx (G1 + GDN16 + tgl16eu = Saturday's flags) 36.4 ms vs G1's 38.95; F32c6xu 51.6 vs
    F32c6i 56.4. tgl16eu: expert GB +0.01 at 16 rows, argmax 16 / 16.
  - Slot #2 (14 arms x 36 checks, with a determinism re-run) is armed after M4.
- **11:2x BREAK98c, Fable ZERO (IDEAS_1117c_slc.md + price_1117c_slc.py), answering Kamen's software-etching idea:**
  - DRAM's 280 GB/s is a wall, but "read by the GPU, once" is not. The SLC re-serves bytes at ~x2.5 [K: 6.6 MB, 38.9
    -> 13-18.7 us].
  - The CPU can fill the SLC with no dispatch and no sync inside the expert phase's DRAM slack (tue1: 175 GB/s, i.e.
    38 % idle, ~58 MB per layer).
  - Priced: -1..-4.5 ms per check (the next layer's dense into the SLC) and -1..-3.7 ms per tree-16 draft block (the
    head's 85 MB re-read part), x1.03-1.18 depending on the SLC's unknown size.
  - Decisive test: a 10-min Metal microbench (GPU re-read bandwidth vs buffer size 2-256 MB; a 21 MB buffer cold /
    CPU-touched / GPU-touched; PASS CPU ~= GPU). Assigned to MoonShots for the next quiet gap.
  - Second idea: expert widths by row survival (path_p x gate; sure rows keep C6), -0.5..-1.1 GB per check [P / G].
    ZERO runs its bytes screen on the dumps now; EvalSuite gets a quality arm if it's >= ~0.5 GB.
  - Priced DEAD on this chip: entropy coding at DRAM rates, texture decode, T-MAC LUT GEMM, pre-decoded int8.
  - Assumption 8's "compute-bound" is really serialization (bytes 16.4 + MMA 6.1 + decode 3.1 + loads 2.9 ~ 26.3 ms):
    Breakthrough's ground.
  - **ZERO's plain answer to Kamen's HC1 hunch (IDEAS_1117c_slc.md section 5): only a COMPILED CHECK moves 550.**
    - (i) On-chip: GPU-resident storage is a few MB; the SLC's usable reach is ~10-40 MB (x2.5 at 6.6 MB, x1.45 at
      14.75 [K]), against the drafter's 95-178 MB per pop (85 MB re-read every pop at Q8).
      - A drafter SHAPED for the SLC + a fused level kernel gives pops ~0.2-0.3 ms and a tree-16 block 3-4 ms:
        ~292 -> ~320 tok/s (+10 %), exact.
      - The check is 48 of 58 ms, so this alone can't reach 550.
    - (ii) 2-bit trellis / codebook on the check: byte floor 25.2 -> ~15.7 ms, but the decode makes the expert call
      1.15-1.21x SLOWER [K], and quality kills 2-bit (2.49x MSB-3 error). Only TRL-3 (3.07 b) passed: -1.4 ms, PARKED.
    - (iii) A COMPILED CHECK (fused per-sub-block kernels in this model's shapes, native tile layouts, one replayed
      command buffer) at 85-90 % of the byte floor: ~28-30 ms at 32 rows.
      - At 29 + 3 + 1.5 ms: kept 17 -> ~507, kept 19 -> ~567, K64's 20.8 -> ~620.
      - Leap prices it at weeks (A+B), then 4-6 weeks (D/E).
    - **So 550 = a compiled check + a 48-64-row tree + a <= 4 ms draft.** The "software HC1" works for the drafter; for
      the check, the 6.7 GB per block fits no cache.
  - **11:3x orchestrator -> Leap:** can ONE full layer, compiled that way, be shown at >= 85 % of its byte floor (quiet,
    bitwise or FAST-gated) by FRIDAY, without hurting Saturday's measured numbers? That would prove the engine half of
> (a quote from the project owner, kept private)
  - **11:3x Leap's answer:**
    - A FULL layer at >= 85 % by Friday: NO (~10-15 %). Today the 32-row check runs at ~46 % of the C6 + int8 bound
      (54.2 ms GPU vs 25.2).
    - Per class vs its floor:
      - experts ~62 % (26.3 vs 16.4; tgl16eu takes it to ~73 %);
      - dense ~52 % (14.7 vs 7.6, compute-bound);
      - cores + HC ~20-35 % (latency-bound).
      A whole layer needs ~1.85x on all of them at once: the 4-6 week D/E work.
    - **PLAUSIBLE by Friday (~50 %): one layer's MoE BLOCK (router -> experts -> shared -> combine) at >= 85 % of ITS
      byte floor.** It is ~48 % of the check's GPU time. Plain streaming of the real expert planes already runs at
      x0.92 of the 280 floor, and Breakthrough's X0 shows MMA and stream overlap.
    - **Orchestrator: GO.** Breakthrough builds it (pre-registered floor + 85 % line). Leap reviews, wires it as a c25 arm
      and measures it quiet in slot #4 (Thu). It is bitwise or FAST-gated; Saturday's path doesn't change.
    - MoonShots' dense sub-block (~70 % expected): not now (account-1 is ahead of its weekly pace).
    - Main account at ~76 %, where Leap and BigMoE stop at 85 %: the main-account thinkers pause after BREAK98c unless
      an idea earns a test, to protect Saturday's slots and BigMoE's plumbing.
- **11:3x ZERO's idea 2, bytes screen** (screen_1117c_pathp.py, CPU; chain-window proxies for tree rows):
  - Width-only (MSB-2 for rows with path_p < 0.2): -0.29 GB per 32-row check. Under the 0.5 GB bar.
  - DROP form (drop an expert used ONLY by one row with path_p < 0.2 at rank 2-9; gate renormalized): -19.8 % = -0.91 GB
    per 32-row check (-3.3 ms at the floor, -5.2 at tue1's rate); 24 % of rows demoted, 20 % of the union dropped.
    tau 0.3: -1.13 GB. At 16 rows: -0.25 GB only. Expected kept cost ~0.2-0.5 %.
  - Orchestrator: recorded as the top 32-row byte lever for the 550 check line. The EvalSuite quality arm comes AFTER
    Saturday (this week's 45 GB EvalSuite time goes to Saturday's tree-16 gates), re-screened on real tree rows first.
- **11:3x Breakthrough: the template costs ~2-3 h extra; proceeding as a TEMPLATE.** The kernels are already
  shape-templated; what's left is a per-model config + no 125B constants in the Python wrappers. The 85 % pre-registration
  stands (f6f806e). Odds: the expert call is at ~77 % of its floor today (tgl16eu), so 85 % for the whole block needs
  ~15 % more + folding. Leap's ~50 % stands.
- **11:3x BREAK98c, Fable C (IDEAS_1117c_walls.md; its file says 12:2x, but the date was 11:3x): AUDIT VERDICT: 550 = kept 16.5 at a 30 ms cycle = the check at its
  byte floor + the draft at its physics. NO new information is needed. 775 needs an information source not found.**
  - The kept, row and byte walls are physics for greedy + the gates: sampling KILLed 23 Oct; 72 % of shallow HE+ misses
    sit at 125B margins < 1 nat; the A_line is spent by CC(1).
  - The one wall made of assumptions is the DRAFT (0.55 ms fixed + 0.56 ms of bytes per pop, serial).
  - New fact: the HE+ tree-32 is a ~25-deep chain (24.9 distinct depths in 31 pops), so level batching buys ~0 on HE+
    (x1.9 MBPP). The draft floor is depth x bytes per forward: 14 ms with the Q8 head, 2.3 ms with a QAT head.
  - Best idea, the POOL DRAFTER: the head keeps a refilled pool ahead of the cursor, and checks take the best 31 nodes
    under it. The refill runs on the idle CPU during the check once the head is <= ~40 MB (after S22's QAT).
    - Measured lower bound: an unrefilled pool serves 19-25 % of HE+ checks with no draft, at x0.978-0.949 kept.
    - Net +5-12 % at the floor.
    - Decisive test (<= 10 min CPU, refill emulation; PASS fresh-draft share <= 0.25 at >= x0.95 kept): asked of
      Fable C 11:3x.
    - **RESULT 11:4x: KILL** (section 7; pool_cursor_1117c.py mode C).
      - The fresh-draft share passes (0.136 HE+ / 0.167 MBPP per check).
      - The kept bar fails: the pool-block factor measures x0.88 against the x0.943 needed.
      - Root cause: right after a correction, the drifted head is right 86.5 % vs 94.2 % for a head re-rooted at the
        125B's true state (MBPP 76.2 vs 91.1). A draft must be re-rooted there: the serial dependency the frame already
        pays. The only hybrid that could pass is a wash at the floor.
- **11:4x BREAK98c, Lunatic (IDEAS_1117c_channels.md): CODE || COMMENTARY** (assumptions 5 + 6). **The first idea on
  record past 775 at the bound (HE+).**
  - Mechanism: the code decodes as ONE causal channel that never reads the answer's comments / docstrings. Each comment
    is a branch forked at its slot and decoded in the same sweeps. Every token is still 125B-verified in its channel.
  - Why: comment and docstring wording holds 66 % (HE+) / 74 % (MBPP) of the exact check-ending misses, so ~2/3 of the
    sequential decisions leave the critical path.
  - Number (t30b cells, chain walks, CC(1)): x1.67-1.87 fewer sweeps on HE+ (x1.39-1.47 MBPP); kept per sweep 16.9 ->
    28-32 on HE+ at ~47 rows.
    - Full cycle at this week's costs: x1.41-1.54 HE+ / x1.21-1.33 MBPP.
    - At the C6 + int8 bound with a 3 ms draft: ~860-960 HE+ / ~395-410 MBPP, vs 568 / 288 coupled.
  - Decisive test: a paired plain-decode quality run of the comment-blind code with branch comments; a Sandbox98 2B
    pre-screen first (~30 min, queued after PREFIX EYES).
  - **NEEDS KAMEN'S CALL on the form:** the comments are written and verified, but the code doesn't read them. That is
    neither COMMENT-FREE (his "cheating", on the kill list) nor INERT. Asked 11:5x.
  - Lunatic also killed today: fresh-name and tokenization-equivalent acceptance (x1.002-1.027), and reusing rows after
    a miss.
- **11:5x GPU gap after V2 (orchestrator, covering for NewIdeas):** Leap's 3-min smoke -> Breakthrough's ~6-min quiet
  MoE-block bench (F1: the shared expert folded into the routed kernels, routed rows bitwise) -> MoonShots' SLC
  microbench (if ready) -> Sandbox98 PREFIX EYES -> Sandbox98 CODE || COMMENTARY pre-screen. EvalSuite holds G1 (filler)
  until ~13:30 at the latest.
- **12:14 EvalSuite: 1c-tiers V2 (both orders pooled) FAILS, so 1c-tiers is NOT adopted** (RESULT_t1c_2026-11-16.md,
  a3a7efc).
  - HE+ 89.6 vs 93.0 %, -3.35 pt, p 0.0074 (2 / 13).
  - Runaway guard: 23 vs 8, p 0.011.
  - Pooled code -2.84 pt, p 0.0041. MBPP passes.
  - G1's 5d pick (delegated, pick5d.py): branch C, comments-only, env g1c, RES off. G1_PICK is written by 13:30.
    G1's chain then runs on g1b by day / tonight after slot #2 / Wed by day; verdict ~Wed midday, before Thursday's
    stack tier 1.
  - **Saturday's stack lever is now g1c, if it passes its gate.**
- **12:4x Sandbox98: PREFIX EYES (B98C) = KILL, both arms** (e7d558a). HE G(9-30) +0.023 [-0.080, +0.135] at block lr
  3e-5 and -0.077 at 1e-4; every band's G < 0.10; tree-32 x0.985 / x0.974. The mid-layer prefix K/V add nothing to the
  head's own view of the final states. FAR: the copy-source half is untested.
- **11:55 Breakthrough: the MoE-block proof FAILS its first quiet read vs 85 %** (best F2 0.52 at L20/32, ~0.40
  elsewhere). The megakernel yes / no is a pre-registered 3-min quiet GPU split at 18:00: propose if the glue GPU is
  >= 150 us per block, close if <= 60.
- **13:xx Lunatic closes CODE || COMMENTARY: the faithful form is a KILL** (Sandbox98 CHAN pre-screen; addendum in
  IDEAS_1117c_channels.md).
  - Speed was confirmed (x1.72 fewer sweeps), but after the extra rows (48.6 per sweep) and 11 % shorter answers it is
    only ~x1.3 tok/s.
  - pass@1 -3.03 (mostly fixable parse failures).
  - The FORM fails: comment consistency 25.6 %, and the code matches the coupled answer in only 32 / 264. Comments steer
    ~3/4 of the next lines, so the split writes a different program.
  - The only rescue (FIM-style comments written after their code) is untested and would be Kamen's call. Kamen's form
    question is moot.
> (a quote from the project owner, kept private)
  - The GPU re-read knee is ~24-28 MB (x1.7 flat to 24 MB; DRAM speed by ~40-48 MB; 85 MB x1.00).
  - A 21 MB buffer: cold 76.8 us, GPU-touched 42.9 us (x1.79), CPU-touched 147 us (x0.52, i.e. 2x SLOWER). Bytes the
    CPU has read reach the GPU at half speed.
  - A GPU-resident re-read helps only for working sets <= ~24 MB, far under the drafter's 85-178 MB per pop.
- **14:xx A FRESH first-principles deep pass (an unanchored agent; ~/FableGod/deepdraft/GRANDMA_1117.md): NO >= 3x
  mechanism. 775 on code is out of reach for this drafter on this Mac (~95 % confidence).**
  - New measurement: the real 127-node t30b trees replayed for every size from 8 to 127 rows, counting kept AND
    sequential draft levels (it reproduces 16.30 / 9.05 / 20.77).
  - At 100 % of every measured peak with perfect overlap:
    - a draft at 0.19 ms per level: 553 HE+ / 336 MBPP;
    - + 64-row clamped trees: 664 / 351;
    - + a FREE draft: 808 / 397.
    For 775, MBPP needs 1.8-3.2x the head's kept under any engine.
  - Why nothing breaks the frame: a weight pass serves only rows whose tokens exist before it starts. The walk is
    two-sided: on HE+, 37 % of checks keep <= 8.
  - 3x over Saturday exists only as a STACK:
    - a check at 90 % of its byte floor (x1.44, 303);
    - a draft at 0.37 ms per level (x1.38, 418);
    - 64-row clamped trees (x1.27, 530);
    - a 0.19 ms-per-level head (x1.15, 612).
    Each is known and owned; none is ready by Saturday; the clamp needs a quality gate. MBPP on that stack is ~321.
  - Best levers: the expert phase (Breakthrough's MoE block) and the draft fixed cost per level (a persistent fused
    level; PASS <= 0.10 ms vs 0.55 today; at 0.37 ms per level the 16-row check goes 211 -> 261).
- **BREAK98c round CLOSED:** no idea breaks the kept wall. What stays open:
  - ZERO's SLC prefetch (+3-18 %; MoonShots' microbench);
  - the compiled check (the 550 path, weeks);
  - the expert-drop byte lever (after Saturday);
  - Breakthrough's megakernel split at 18:00.
## KAMEN, Tue 17 Nov 07:2x (verbatim, dictated): THE TWO BREAKTHROUGHS BY SATURDAY; LOSSY >= 550 OR NOTHING
"I'm missing the point of all of this. I'm not trying to make another 50% or it's even less improvement on code and
exact and lossy. I don't really care about whether we bump up speed from 100 and something tokens per second to 100 and
something tokens per second plus 25. The thing I care about is the breakthrough. You know it. You know how much I need
to get it by Saturday. And I think we already established that we need to find a way to find the cheapest possible way,
which hasn't been done by anybody before. The cheapest possible way to give the drafter information from the big model.
I thought we already got to that conclusion. So yes, the project, you will have to do that. It's going to be a big part
of it, and especially for Saturday. 1 hour minute. So by Saturday, the breakthrough you have made is create something
beyond the physical theoretical limit on how fast we can give the drafter information from the big model, as well as
getting the drafter to 97.8% accuracy at 30 tokens deep, which also hasn't been done by anybody and is also just as big
as a part of the project. So we have to accomplish these goals. Do you understand me? That's like we're going to play
tennis and not actually playing tennis and just warming up. I mean, yeah, so these are the goals and you should do them
by Saturday. As I told you, you already have all the needed information, so you want to use it. So yeah, both things
done by Saturday, and remember that, not some goofy little improvements of 5%. I'm not accepting an end result of lossy
under 550 tokens per second."
- **GOAL 1 (by Sat):** the cheapest way ever to give the drafter the 125B's information on the drafted tokens, beyond
  the believed limit (today that limit is "a 125B row per drafted token").
- **GOAL 2 (by Sat):** the drafter at >= 97.8 % per guess, 30 tokens deep.
- **END RESULT:** lossy >= 550 tok/s on code. Small levers (+5-50 %) are not the work.
- (CORRECTED 07:5x, Fable C's audit: "97.8 % flat to 30 with true states" is the marginal row. Walk-strict with true
  states: 93.2 / 95.3 / 96.4 / 97.8 / 98.8 / 99.2 / 99.3 % by band, so >= 97.8 % from band 7-10 to 30. See BREAK98.md.)
- On record, it is the same wall (Lunatic IDEAS_1116c_drift.md, HE+ kept per 32-row check):
  - own tree-32 16.26; ORACLE-31 18.48; true-state chain 20.77;
  - true-state + tie siblings 26.79, so drift fixed + a tree = ~24.5-26.8;
  - "the only source that delivers it at scale is a 125B row".
  So GOAL 1 (cheap true-state information) is what lifts GOAL 2's deep accuracy. 550 at kept ~24-26 needs a cycle of
  ~44-47 ms (today 56.4 check + 19.1 draft + ~4).
- The orchestrator's response (07:3x): BREAK98.md (the sprint brief) to the five thinkers + MoonShots + Breakthrough.
- **07:3x NewIdeas, ORACLE-31 NUMBER CHECK** (Lunatic's script re-run on a copy):
  - It reproduces 18.48 / 11.21, K127 23.70, LAB 20.77 and EAGLE-2-31 16.26.
  - REFINED: ORACLE-31 is a hindsight bound. Without new information, the head's own cap at 32 rows is 16.26 in
    expectation (calibrated head, best-first).
  - So kept 24 at 32 rows needs (1) true-state-class information (BREAK98; it also gives GOAL 2), or (2) ~128
    own-information rows (23.70) at the price of 32 (early death + compute; speed only).
  - The relay to all seven sessions is done. The GPU queue rule is set (a BREAK98 test always goes first). The stack
    chain is not armed. G1 is filler.
- **BREAK98 results as they land** (NewIdeas screens from ~09:00):
  - **Fable B (IDEAS_1117_partialwidth.md): partial-width 125B rows per drafted token are NOT route 1** [P, config +
    280 GB/s].
    - A 3 ms-per-block channel buys ~28 MB per token, about one 125B layer.
    - The candidates are all over that: dense + attention + shared 57x, shared + top-1 62x, 8 dense layers 10x, layer 0
      alone 2x (worthless for drift).
    - Batching across positions informs only the correct prefix, i.e. a fork re-anchor, measured +2.4 % kept: KILL.
    - Rows 33-64 inside the check: +15-18 ms compute, route 2 only.
    - Capture ask to NewIdeas: ablated forwards (shared + top-1 first) dumping the final state + top-8, to price what a
      learned proxy (FAR's lens) must reproduce.
  - **07:3x NewIdeas' pricing tool** (b98sim.py, HE+ chain-32, true state given at):
    - every 2nd depth: 19.20 (78 % of the own -> LAB gap);
    - every 4th: 17.18 (49 %);
    - one refresh per block: <= 15.32 (22 %);
    - depths 0-15 only: 18.64 (70 %).
    So a channel must be DENSE and EARLY.
  - **Fable FAR (IDEAS_1117_futurestates.md): FUTURE-STATE ROWS.**
    - k = 8 placeholder rows (<|fim_pad|> or the mean embedding) hang as chains in the check batch under the end row
      and the lowest-p rows. They give the next block's head the 125B's state at its first 8 drafted positions,
      conditioned on the true prefix plus an explicit unknown (so not SIB).
    - 2 chains cover 71 % of HE+ block ends (59 % MBPP; CPU screen endcover.out).
    - ~0.28 ms per marginal row [P, band 0.19-0.45; the placeholder routing is unknown]. 16 rows ~4.5 ms.
    - Pay line at the bound: +18 % kept.
    - Screen 1 (Sandbox98, 2B, <= 5 GB, ~30 min): dispatched by the orchestrator at 07:3x with FAR's lines.
    - Screen 2: TrainLab on tonight's capture.
    - Second idea: PREFIX-DEPTH INPUTS (EAGLE-3 fusion of layers 8 / 16 / 24 + a GLIDE read of the 125B's attention
      K/V).
    - Capture asks went to NewIdeas to consolidate for BigMoE by 18:00.
  - **Fable ZERO (IDEAS_1117_channel.md): CHECK-AS-TAP-PASS.**
    - Byte law: a 125B-class state per drafted token comes only from a BATCHED pass over the draft. Sequential or
      wavefront shapes re-read the dense weights (1.4 GB per block for ONE layer). The bar is 504 MB of reads per
      extra kept token.
    - Best (route 1): at layers 8-16 the check already holds the 125B's state at every drafted row. Tap the head with
      those states and re-draft the rows early death would drop, inside the same check: +0.7-1.1 GB, +2.6-3.8 ms,
      +1.5..+3 kept [P], exact acceptance.
    - Screen: TAP-LAB / FORK-TAP on tonight's mid_*.npz (a ridge map from layer L to the head's tap input, frozen
      head). PASS >= 40 % of the drift gap closed at layer 16, bands 3-14; KILL < 15 %.
    - Capture ask: the 10,240 pre-mixer stream at layers 8 / 16 / 24 at the same rows, plus a dense-only layer-8 state.
    - **Honest cycle:** route 1 alone gives kept ~18-19, ~420-500 tok/s at a 29.6 ms check. Kept 24 needs route 1 +
      route 2's rows, not either alone.
  - **07:3x NewIdeas: THE LEAD = IN-CHECK REPAIR** (= ZERO's CHECK-AS-TAP-PASS; research/week98/b98_repair.py / .txt, a
    CPU bound on the cells).
    - Mechanism: at layer L (8-16) the check holds the 125B's state at every drafted row. Locate the first error,
      re-predict it from the tap, refresh the head and re-draft the tail. The new rows catch up through layers 1..L,
      then rejoin the check.
    - Bound (perfect location, true-state repair, chain-32 HE+):
      - one round: 13.77 -> 18.97 (+38 %, 74 % of the own -> LAB gap);
      - two rounds: 20.83 (= LAB);
      - MBPP 6.92 -> 10.65 / 13.13.
    - Partial repair quality q, one round, HE+: q 0.5 gives +22 %, q 0.3 gives +13 %.
    - FAR's FUTURE-STATE ROWS bounded the same way: <= +14 % even with perfect states, +6 % at G 0.5. Its sandbox screen
      stays only as a probe.
    - Deciding unknowns: (a) the layer-L error location (false repairs break correct tokens); (b) q; (c) the cost per
      round (re-draft ~2-4 ms + catch-up ~0.7-1.1 GB, +2.6-3.8 ms, with the check paused). (a) and (b) are measured by
      TAP-LAB / FORK-TAP on the mid capture.
    - **Orchestrator 07:4x: its capture (L8 / 16 / 24 streams along the true text) is pulled forward to TODAY by day**
      if BigMoE is ready by ~13:00 (the owner away). The other capture asks stay tonight.
  - **07:4x NewIdeas, IN-CHECK REPAIR PRICED (BREAK98_SCREEN.md, 50eac1a): a big kept bound, a moderate net gain.**
    - Tree version, HE+, perfect location (q = the share of true-state-quality repairs):
      - one round: 22.06 (+36 %) at q 1 / 19.20 at q 0.5;
      - two rounds: 24.06 (+48 %, the 24 that 550 needs) / 19.98;
      - MBPP 9.03 -> 14.10 / 17.04 at q 1.
    - Cost per round, with the check paused: readout + tail re-draft (~12 levels) + catch-up of layers 1..L for ~16
      rows (~0.40 ms per layer at the bound).
    - Net vs tree-32 at the bound (551 tok/s), one round, q 1 / q 0.5: L4 +24 % / +8 %; L8 +18 % / +2 %; L16 +8 % /
      -6 %. Today's engine: at most +18 %. It pays only with a SHALLOW, high-quality tap.
    - Swap-only repair: KILL (the old tail is right after a wrong token only 16 % / 12 %).
    - The cheaper cousin is EARLY DEATH on a big tree: the same readout selects among pre-drafted K64 / K127 rows, with
      no pause and no catch-up. The day capture serves both; the spec is now layers 4 / 8 / 16 / 24.
  - **MoonShots (IDEAS_1117_valuepred.md): SESSION VALUE PREDICTION = KILL** (R2 <= 0.22; the hindsight-best earlier
    state is 0.67-0.76 rad off).
  - **Breakthrough (IDEAS_1117_landmark.md, f499735): LANDMARK = BETWEEN, KILL side.** Context-matched states are close
    at mid-depth (tap 15 cos 0.785 vs 0.405) but not at tap 45. Late states are history-specific, so no token-keyed
    source delivers them.
  - **NewIdeas' read for 550:** after costs, every channel found so far is a +10-25 % lever at the bound, and plain
    tree-32 already reaches ~551 there. So the gap from today's 205 is mostly the CHECK (56.4 -> ~25-30 ms) and the
    DRAFT (19.1 -> ~3 ms), with information levers multiplying on top.
  - **Orchestrator 07:4x: three parallel lines under BREAK98:**
    1. CHECK (Leap): -> ~28-32 ms.
    2. DRAFT: -> <= 4-5 ms. MTP Head (8-bit + screen head) + MoonShots (sync-free block drafting + a fused level kernel,
       pulled into this week). NewIdeas assigns.
    3. INFORMATION: repair / early death / FAR rows.
    NewIdeas sends owners, dates and a Saturday-realistic range by ~10:00. The orchestrator's rough range: ~300-500.
  - **07:4x Breakthrough's correction (036e34d) + NewIdeas' bound (b98_landmark.txt): LANDMARK-ANS = the best
    ECONOMICS so far.**
    - Mechanism: where a drafted position's last k tokens recur EARLIER IN THE ANSWER, feed the head that verified
      row's stored state (a state store + a GPU k-gram hash). Free and exact.
    - Coverage 38 % of HE answer positions (49 % past the first third). The late state is cos 0.795 to the true state,
      as close as the true state one step back (0.798); the stale anchor is 0.644.
    - ~~Bound (random refreshes at 38 %, the head treats a landmark like the true state): HE+ chain-32 13.77 -> 17.83
      (+29 %, 58 % of the drift gap); MBPP +40 %.~~ **CORRECTED 07:5x (NewIdeas 98cce14, b98_landmark_real.txt): with
      the REAL, clustered coverage (answer-only k-gram matches on the bank's ids):**
      - coverage 34 / 18 / 11 / 2 % at k = 2 / 3 / 4 / 8;
      - ceiling at true-state quality, HE+ chain: +24 / +13 / +5 / +2 %.
      - The 38 % is the k ~2 regime, where MoonShots' kill bites (retrieval = copy, R2 <= 0.22). The likely outcome is
        a copy-class gain (+5-10 %).
      - The audit addendum adds KILL-COPY (G_landmark <= G_copy + 0.10 -> KILL) and a stale-by-one reference (WARN if
        < 0.5 x G_stale1), per k.
      - **The information line's honest ceiling today: every channel nets ~+5-25 %. The 550 rests mostly on the check
        (Leap) and the draft (MTP Head + MoonShots).**
  - **07:5x Fable C (IDEAS_1117_repairshape.md): in-check repair TRANSFERS to trees, budget-correct on the real 127-node
    dumps** (NewIdeas' chain 18.97 reproduced).
    - Tree-32 16.30 -> 23.10 kept at one round (x1.42) / 25.81 at two (x1.58), with perfect location + LAB-quality
      guess, rows-neutral. At q 0.5: 20.0 / 20.6. MBPP 9.05 -> 15.01 / 19.45.
    - So kept 24-26 at 32 rows is inside route 1 alone if location x guess >= ~0.8 with two rounds.
    - A probe-free refresh at a fixed row is DEAD (x1.04-1.16). LOCATION is the mechanism, so the deciding number is
      the layer-L probe's first-miss RECALL.
    - Twist: repair as an ADDED branch (the original tail stays; false flags cost rows, never kept): 24.61 / 29.93
      perfect, ~22 / ~23.5 at q ~0.65 [P].
    - Screen: two lines added to the early-death run:
      - first-miss recall vs the false-flag curve at L 8 / 16 / 24;
      - P(correction in the head's top-2 from the true state, or in the lens top-2) at the last right row.
      PASS: recall >= 0.8 at <= 20 % false flags AND multi-guess >= 0.75 at L16.
    - Audit:
      - "97.8 % flat walk-strict" is the marginal row (corrected above and in BREAK98.md);
      - 26.79 is unbudgeted;
      - a marginal batched row costs ~0.11-0.18 ms at Leap's floor;
      - 550 needs the check's fall (x2.7) AND the channel (x1.5); each round adds +1.7-3.8 ms.
  - **07:5x MoonShots CORRECTION, the draft line (DRAFT_PLAN_1117.md, b8e4e76):** the planned head (8-bit Linears + 32K
    screen) reads 177.6 MB per level (dense 99.7 + head 47.2 + experts 30.7) [K p3_m012_stock.log], not ~42-73.
    - Byte floor: 0.607 ms per level = 12.6 ms per HE+ tree-32 block at zero fixed cost (today 20.0).
    - So <= 4-5 ms needs a 2.4-5x SMALLER head (MTP Head's lane), not kernels.
    - Kernels: fused2 18.2 ms now (bitwise); a persistent kernel ~14.5-16 ms by Saturday.
    - **The orchestrator's ~300-500 Saturday range was too high on the draft side; NewIdeas' 10:00 report re-prices
      it.**
  - **07:5x Lunatic (IDEAS_1117_floor.md): the floor is 1-3 located 125B DECISIONS per block, not states, and the 125B
    forms them LATE.**
    - Two decisions at the chain's first misses (the 125B's own token, no state) give HE+ 23.65 / MBPP 15.39 kept,
      against 20.77 / 12.83 for the true state at all 31 rows. The LAB head fixes only 68.5 % of first misses.
    - States must be dense: every 2nd row 78 %, every 8th 25 % of the LAB gain.
    - Late verdict: a tuned lens finds the head's misses at AUC 0.64-0.70 after layers 6-36 (the head's own margin
      0.954), 0.92 after 46, 0.99 at 48. As repair deciders, layers <= 36 LOWER kept (13.77 -> 10-13).
    - Screen: VERDICT-DEPTH on the capture (floor_lens.py, < 1 min). LIVE if a layer <= 24 gives decider phi >= 0.7 at
      psi <= 0.5 %; otherwise route 1 by partial depth is CLOSED, and the GPU goes to route 2 + the check + the draft.
  - **08:1x NewIdeas, BIG: VERDICT-DEPTH on BigMoE's day capture (HE+, 07:59-08:27, all self-checks pass; 2b30212):
    ROUTE 1 BY PARTIAL DEPTH IS CLOSED on the 125B.**
    - Location: the head's own margin finds its first misses at AUC 0.954. The 125B's lens: 0.69-0.73 at layers 2-32,
      0.756 at 40, 0.839 at 44, 0.97-0.999 only at the final layer.
    - Decision: the lens picks the right token at 20-29 % of first misses through layer 32.
    - Repair driven by any layer <= 44 LOWERS kept.
    - EARLYDEATH stage 2: KILL (the lens drops 3.0-4.6 % of wrong children at <= 2 % false drops; the head alone 31.4 %).
    - The 125B decides in its last ~4 layers. Logged by NewIdeas (TRIED VERDICT-DEPTH + EARLYDEATH-S2, 363ee3c).
  - **08:1x orchestrator: BREAK98b** (~/FableGod/deepdraft/BREAK98b.md) went to the five thinkers, until ~09:45.
    - The question: kept >= 24 for about a 32-row check's cost, via:
      - (a) nearly free extra rows (compute);
      - (b) two located decisions per block for <= ~1.3 passes;
      - (c) a smarter drafter at <= ~5 ms;
      - (d) anything else.
    - What is left this week for kept: LANDMARK-ANS (+5-10 % realistic), FAR's rows (<= +14 %), and route 2's per-row
      ALU cost (Leap).
  - **08:2x Fable ZERO, BREAK98b (IDEAS_1117b_cheapest_row.md): kept 24 for a 32-row price is OUT by arithmetic, but
    bigger trees get CLOSE to 550 at the bound.**
    - A row is 3.6 G dense MACs: 0.16 ms at the measured int8 peak (0.13 without lm_head), ~7 ms on the ANE, ~35 ms on
      the CPU, and all of them share the saturated DRAM bus. So 96 rows cost >= 12.5 ms.
    - Free per row: the experts (off-spine rows routed inside the loaded union ride in the one-call's 16-row MMA
      tiles, now 2/3 padding) and the GDN cores (a chunked TREE form reads the 226 MB state once: ~2 ms at 128 rows;
      lossy line).
    - Marginal row ~0.13 ms, so K64-96 gives ~580-610 tok/s at 100 % of every peak, ~530 at 85 % (K128 545 / 474) [P].
    - Screens: Leap's 10-min slice microbenches, sent to Leap 08:2x to run first.
      - S1: chunked tree GDN at 64 / 128 rows, PASS <= 3 ms per 36 layers.
      - S2: the one-call at 64 / 128 rows with off-spine routing clamped to the 32-row union (lossy), PASS <= 1.10x the
        32-row time.
  - **08:2x Fable B, BREAK98b (IDEAS_1117b_dense.md): route (a) on the dense pass is closed at the desk, unless the
    matrix rate is higher.**
    - A marginal row is 3.6 G dense MACs (layers 2.75 + shared 0.24 + lm_head 0.64): 0.26 ms bf16 / 0.15-0.17 ms int8
      AT the measured 14 T MAC/s matrix peak [K]. So 0.15 is the floor, and the 0.05 ms-per-row frame is 3x below it.
    - Activation precision maxes at the measured 1.55-1.73x (already counted). Sibling / prefix sharing is 0 on the
      dense pass.
    - Exact skips: lm_head at internal nodes only, -0.6..-1.3 ms per 128-row check. int8 on the head -2 ms (G1-gated).
    - The ONE deciding screen: is 14 T MAC/s the neural-accelerator rate? A Metal 4 tensor-op matmul on GDN qkv|z at
      M 128 vs dn32q; PASS >= 2.5x, KILL < 1.3x.
    - Leap's M27-P (data/nax_peak.txt) already calls 14.0 T MAC/s the matrix-unit peak (rough smoke, shared GPU).
      Asked of Leap 08:2x: answer from the record, or run it after S1 / S2.
  - **08:2x Fable FAR, BREAK98b (IDEAS_1117b_2bproposer.md): the 2B-INFORMED TREE, the first bound on record near 24.**
    Measured on the HE+ / MBPP cells + MoonShots' rows2b / rows4b (alignment self-check 99.4 %).
    - At the own chain's first misses: the frozen 2B is right 67 % (4B 72 %), the head's #2 67 %, #2|#3 82 %. They are
      complementary: head #2 OR 2B 86 %, #2 / #3 OR 2B 90 %.
    - As an inline DECIDER the 2B buys only 13.77 -> 15.1-15.5 (+10-13 %), capped by false overrides.
    - As a SIBLING PROPOSER (the 2B's token as a tree node beside the head's #2): unbudgeted bound 22.06 HE+ (23.61 with
      #3; MBPP 15.9 / 18.3) vs own information 19.14, at ~1.3 siblings per check.
    - Realistic tree-32: ~18.5-19.5 [G]. 22-24 needs the siblings' sub-chains budgeted (48-64 rows), i.e. this knowledge
      PLUS route (a)'s cheap rows (ZERO's K64-96).
    - Cost: one batched 2B pass per block (1.06 GB, ~4 ms), unless hidden on the ANE (WHOLE-CHIP measured +23-25 %).
    - Screens:
      - NewIdeas' tree join on the 127-node dumps + rows2b (PASS >= +12 % kept at <= 40 rows);
      - then a 10-min GPU 2B pass over the tree nodes.
      Second idea: a distilled chooser over the head's top-3 at flagged rows (TrainLab; Sandbox98 screen).
    - CORRECTION to FAR's IDEAS_1117: block-end coverage was overstated 2-3x by a grouping bug. Two placeholder chains
      cover 37 % of HE+ block ends (not 71 %); 10 chains ~84 %. The Sandbox98 G screen is unaffected; the price is
      2-3x worse.
  - **08:2x Leap on the route-(a) screens:**
    - S2 (the one-call at 64 rows, off-spine routing inside the 32-row union, real layer-20 routing): building, ETA
      ~09:10. 128 rows needs the grouping kernel widened (P <= 1024 pairs), so it will be projected.
    - M27-P's 14 T MAC/s was the bf16 NEURAL-ACCELERATOR path (Metal 4 MPP matmul2d, 13.7-15.0); INT8 was never
      measured. An int8 register-peak test follows S2 (~10 min). < 1.3x bf16 = KILL; >= 2.5x = Fable B's GEMM screen.
    - S1 (tree-chunk GDN) needs a new kernel (~2-4 h; ~1.5-3 ms at 128 rows [P]).
    - Orchestrator rule: S1 starts only if S2 holds at 64 rows (<= ~1.2x the 32-row time). Otherwise Leap returns to
      the 32-row check modules. tue1 runs at 09:30 as armed.
  - **08:25 BigMoE: the day mid capture LANDED, PASS.** mid_{heplus24,mbppbank}.npz, layers 2 / 4 / 8 / 16 / 24 / 32 /
    40 / 44, 22,016 answer rows each; 124 / 124 files bitwise equal to the bank; 26 min, peak 21.8 GB.
    - Disk 110 -> 105 GiB (floor ~100). BigMoE asks to delete ~/datasets/deepdraft/rawtaps (3.8 GB, superseded,
      re-creatable ~1 h) and its intermediate t30x2 / t30x4 step checkpoints (5.4 GB).
    - HELD: permanent deletes need Kamen's OK; asked at 08:2x.
    - **08:3x EvalSuite: the proofs need wd on the fresh set, so the capture is FULL** (dd_topk.py feeds the candidate
      z["wd"]; raw taps can't rebuild it; a re-prefill differs from the runtime path).
      - Slim works only as option (b): the candidate's TOPK + T8 dumps written during the capture, from the in-memory
        wd (~0.8 h of dump GPU). That needs the proof-run amendment (candidate sha256, commit, settings, scorer sha256)
        filed BEFORE the capture, i.e. the candidate pinned by Wednesday evening.
    - **Orchestrator plan (08:3x):**
      1. If Kamen approves both deletes by Wed ~15:00: FULL (~100-104 GiB after).
      2. Else (b), if t30b can be pinned by Wed ~18:00 (NewIdeas + MTP Head confirm today), with the 3.8 GB delete:
         ~100-102 GiB.
      3. Else the capture slips until the disk is cleared.
      EvalSuite prepares the (b) amendment variant now as the fallback.
    - **08:3x NewIdeas:** the candidate should be t30b AS IS (BREAK98 found no better drafter), pinnable today, so (b)
      is the safe route and needs only the 3.8 GB delete. Exception: MTP Head's 8-bit + screen head as the candidate
      would pin Wed afternoon. MTP Head answers ~09:45.
  - **08:3x NewIdeas: the 2B discrepancy is RECONCILED (b98_2b_reconcile.txt, ad06be2). FAR's 2B-informed tree: KILL.**
    - Lunatic's 52 / 58 % are the WALK's first misses (the decoding process, 337 HE+). FAR's 67 / 72 / 67 % count first
      misses of chains started at EVERY anchor: 3,401 misses at only 542 positions, each counted 6.3x (up to 30x).
    - Budgeted: 2B siblings keep 15.94 at 71.9 rows per check vs best-first K63 20.77. FAR's 22.06 was unbudgeted.
  - **BREAK98b OVERALL (NewIdeas):**
    - (a) closed: int8 = 1.4-1.7x bf16, not 2.5x;
    - (a') 64 cheap expert rows hold the 1.2x gate (S2);
    - (b) closed by arithmetic;
    - (c) KILL;
    - no new (d).
    Kept is exhausted near 17-19. **550 = check ~26.5-28 ms + draft <= 3-4 ms.** The 10:00 report follows after MTP
    Head's ~09:45 answer.
  - **08:4x EvalSuite: the (b) fallback amendment is DRAFTED, not filed** (86d1e55,
    results/certify98/proofrun_amendment_b_DRAFT_2026-11-17.md).
    - Per fresh file, in the capture sitting: write wd (fp16, hashed), read it back through the offline loader into
      dd_topk (TOPK) + dd_tree (T8, 32 nodes), write the slim file, delete the temporary wd. At most one wd on disk.
    - Owners:
      - NewIdeas / MTP Head: the candidate's pins by Wed ~18:00.
      - BigMoE: the sitting script; the head-only passes beside the capture with a NO-SWAP gate; and Wed by day a
        BITWISE equivalence check vs the offline dumps on the dev bank. Not bitwise means (b) is off.
      - EvalSuite: a (b) mode of certify98_dumpcheck.py.
      - NewIdeas: 7a's conditional HE+ T8-127 is dropped unless it is named and run in the sitting.
    - Filing conditions (orchestrator 08:4x): Kamen approves at least the 3.8 GB delete by Wed ~15:00 (without it, (b)
      ends at ~96-98 GiB), and the candidate is pinned by ~18:00. Otherwise the capture slips.
  - **08:4x Sandbox98, BREAK98 Screen 1 (FAR's future-state rows): variant A (<|fim_pad|>) = KILL and NEGATIVE**
    (LEVERS_A B98S1, af17982).
    - HE G(1-8) -0.105 [-0.145, -0.060], tree-32 x0.960; MBPP G -0.072, x0.970.
    - The frozen 2B's own top-1 at a placeholder row is right 26 % at offset 1 and 6-10 % at 2-8 (the head 93 % at
      depth 1).
    - Variants B (mean embedding) and C (<|vision_pad|>) are running; the screen is KILL only if all three are. ETA
      ~09:30, then LANDMARK-ANS (~15 min).
    - **09:2x FINAL: KILL on all three variants, each NEGATIVE** (LEVERS_A "B98S1 result", e68d9cc; TRIED SB-B98S1).
      - HE G(1-8): A -0.105, B -0.130, C -0.174. HE tree-32 x0.960 / 0.958 / 0.949. MBPP x0.970 / 0.971 / 0.958.
      - The frozen 2B's own top-1 at a placeholder row is right 21-26 % at offset 1 and 6-11 % at 2-8.
      - FAR's own kill condition holds (a frozen trunk carries no usable plan at a placeholder), so the 125B
        placeholder capture is NOT needed.
      - LANDMARK-ANS started 09:21 (~15 min).
- **Tue 17 09:3x (NewIdeas' 10:00 REPORT, BREAK98_SCREEN.md): Saturday ~270 tok/s realistic, stretch ~375 (HE+ lossy,
  ungated, tree-32 + CC(1) at kept 16.91). 550 = check <= ~28 ms + draft <= 3-4 ms, neither in reach this week.**
  - CHECK (Leap): 56.4 -> realistic ~45 (43-47); stretch ~38-40, or ~34 only with the expert MMA / stream overlap
    (-5.5 ms, a FAST rewrite).
    - Steps: Tue seams -2.0 (measured), gdn_fast -1.5, one-launch HC read -2.5..-3.5, int8fold -2..-3; Tue night
      module A/B #1; Wed folds -2..-5 + router / glue; Thu Friday's dry run + opt-in flags to BigMoE; Fri the
      measured check.
  - DRAFT (MoonShots kernels + MTP Head): realistic Saturday 14.5-16 ms per block (the stored head reads 177.6 MB per
    level; MTP Head withdrew 8-bit + screen as a byte lever). <= 4-5 ms needs <= ~34 MB per level (tree-32) or ~75 MB
    (tree-16):
    - Path A: a tree-16 stack (M3 pricing).
    - Path B: a distilled ~25-30 M student (Sandbox S20 today, ~13:30; if it holds, a 125B run Wed night / Thu).
  - INFORMATION: nearly all CLOSED on measurements. Open: LANDMARK-ANS and 64 cheap expert rows (~flat at the bound).
    Kept is exhausted near ~17-19 this week.
  - Range:
    - realistic: check 45 + draft 15 + 3 -> ~270 (~285-295 with LANDMARK);
    - stretch: check 34 + draft ~9 + 2 -> ~375 (~410 with LANDMARK).
  - **Orchestrator decisions (09:3x):**
    1. The expert rewrite: Breakthrough owns it, Leap reviews; FAST track via EvalSuite's gate; opt-in flag.
    2. A GATED, MEASURED lossy number by Saturday: YES. BigMoE plumbs t30b + tree-16 + CC(1) + G1's pick into r57 from
       Wed morning: key test Thu midday -> tier 1 Thu -> quiet timing Thu night -> card gate Fri night -> verdict ~Sat
       07:00. If a step slips, it reads "measured, card gate pending".
    3. The exact headline timing (ii) is back in for Thu night (t30b + tree-16 after M4, token identity 264 / 264).
    4. Wed night: CERTIFY98 SLIM (b), with t30b as is pinned by Wed 18:00. It needs Kamen's OK on the 3.8 GB delete.
    5. Tonight's capture is EMPTY: M4 -> Leap slot #2 -> G1 filler.
  - **09:4x MTP Head: the CERTIFY98 candidate is PINNED: t30b AS IS** (TrainLab data/t30b_pin.md, 82171b2:
    dd_t30b.safetensors sha256 790c42db...d502, code + scorer hashes, and an engine-form flag for EvalSuite). So
    Wednesday can go SLIM, pending Kamen's OK on the 3.8 GB delete.
    - CORRECTION from MTP Head: its 22:2x head plan saves NO bytes. 8-bit + 32K screen is the head as stored, 177.6 MB
      per level; the 42-73 MB figures were wrong. That also voids the orchestrator's earlier "draft ~8.8-11.2 ms,
      ~235-240 tok/s" relay.
    - The draft line: path A (Q4 + 8K screen + top-4 experts, tree-16, ~74 MB per level) on MoonShots' M3; path B = S20
      (dd16a7a; the sandbox head at 0.53x / 0.46x params, full budget), read-out ~13:30. If it holds, a 125B student
      trainer for a Wed / Thu night run.
  - **09:4x Sandbox98: LANDMARK-ANS = KILL** (LAB, frozen head). HE covered-cell G(1-8) -2.84 at w 1 / -2.56 at w 0.5.
    Even the TRUE state one position back hurts the frozen head (G -1.36), so the "stale by one" premise fails. **With
    this, every BREAK98 information lever tested this week is CLOSED.** Saturday's realistic range loses LANDMARK's
    +5-10 %: ~270 / stretch ~375.
    - Sandbox98 memory correction (LEVERS_A 1a21ed2): its read-out steps peaked at 6.4-7.5 GB lifetime footprint, above
      the 5 GB locks (swap flat, no harm). Its locks are now 8 GB for read-outs and 6 GB for 2-arm training, with the
      NO-SWAP gate at the lock + 8.
  - **09:4x Leap asked: build the folds for the 16-row check too?** Saturday's gated number runs tree-16, and c25's
    F16 lossy check is 38.6 ms with the same kernels. Estimate -5..-10 ms at 16 rows, +10-20 % on Saturday's number.
    **Orchestrator: YES (b).** Time the modules at 16 rows in slot #3 (Wed morning), including a 16-row one-launch HC
    read. The passing ones go to BigMoE as r57 flags by Wednesday evening, before Thursday's key test, with EvalSuite's
    gate deciding (FAST). The 32-row work stays first for the 550 path. tue1 is waiting for NO-SWAP >= 39 (latest
    start 11:30).
  - **09:3x NewIdeas: BREAK98's INFORMATION line is CLOSED for this week** (LANDMARK-ANS KILL, read by Breakthrough on its
    registered lines, 04a1f7b). What is left for kept: a better-trained head (Goal A training, after Saturday) and route
    2's per-row ALU cost (Leap).
    - GPU: tue1 waits behind MoonShots' C2 (path-A pricing, ~10:15) and MTP Head's S20 (~10:19), then ~10:20-13:20,
      then V2 (~14:30), then G1 filler. The orchestrator left the order as is.
    - MoonShots: the r57 path drafts a block per forward, so fused2 / sync-free have no engine target this week.
      MoonShots starts the persistent level kernel today. Breakthrough's expert rewrite started 09:3x under Leap's
      review.
    - **Orchestrator 09:4x:** BigMoE's Wednesday tree-16 loop IS per level, so sync-free (-28 % per level, exact) +
      fused2 (bitwise) become its DEFAULT if they bit-test equal by Wednesday evening: ~-5 ms of draft, ~+8-10 % on
      Saturday's measured number. **CORRECTED 09:4x (MoonShots / NewIdeas): fused2 is bitwise only for the stored MTP
      head on a chain, so it is OUT for this loop. Sync-free stays (~-3 ms per tree-16 block, bit-tested by Wed
      evening).**
  - **09:4x NewIdeas CORRECTION: Saturday's GATED numbers are LOWER than the 10:00 range.**
    - BigMoE's t30b tree loop (dd_tree.py) reads ~676 MB PER POP as stored (fp32 dense 277 MB, the full 248K lm_head
      358 MB, a host sync per pop): ~35-43 ms of draft per tree-16 block.
    - The format sets the draft: bf16 + 32K screen ~17 ms; Q8 + 32K ~14; 4-bit + 8K (C1) ~10 (kept x0.963 x 0.976). The
      32K screen is free in kept (MTP Head's sandbox).
    - Saturday's GATED config (t30b + tree-16 + CC(1) + G1, 16-row check), HE+ [P from parts; kept 11.26 x 1.04]:
      - check 38.6 ms (today): as stored ~145; bf16 + 32K ~200; Q8 + 32K ~209; C1 ~213;
      - check ~31 ms (Leap's 16-row modules): as stored ~160; bf16 + 32K ~230; Q8 + 32K ~242; C1 ~250.
    - The EXACT headline (ii), tree-16, exact check ~66.5 ms: as stored ~104; bf16 / Q8 + 32K ~130-134.
    - **So the gated, measured lossy number is ~200-250 and the exact one ~130. The 10:00 "~270 / ~375" was the
      UNGATED tree-32 projection.**
  - **Orchestrator 09:4x decision:**
    - t30b in r57 runs at Q8 + 32K if MoonShots' Wed ~07:30 Q8 read-out holds kept, else bf16 + 32K. NEVER as stored.
      (Drafter format: the gate covers it in lossy; speed only in exact.)
    - CERTIFY98's count check runs the PINNED stored format; EvalSuite + BigMoE confirm the +-3 % check stays valid, or
      file an amendment.
    - The corrected range went to Kamen at 09:4x.
  - **09:4x NewIdeas PAUSES for usage pacing until ~17:42** (the account-1 weekly pace line). Coverage:
    - G1's 5d pick at V2's verdict (~13:40-14:30) is EvalSuite's; it's mechanical.
    - GPU slot questions go to the orchestrator.
    - Standing order: Leap's tue1 after C2 / S20 -> V2 -> G1 filler. Tonight: M4 -> Leap's slot #2 -> G1's night
      window. Wed morning: MoonShots' cuts -> Leap's slot #3.
    - Results due go straight to the orchestrator: MoonShots' C2 (~10:15) and Q8 (~13:30-14:00), MTP Head's S20
      (~13:30), BigMoE's answers (tree flags + chain-32 path), EvalSuite's amendment drafts.
  - **09:5x Leap CORRECTION: the 16-row modules are -1..-4 ms of 38.6 [P], not -5..-10 (+3-8 % on Saturday's number).**
    - The 16-row one-launch HC read is built and bitwise (m16.py, 32 / 32) but 1.04-1.13x SLOWER: KILL at 16 rows. The
      32-row one-launch read stays.
    - GDN16: bitwise, ~-0.5..-1 ms [P], into slot #3.
    - t1cf (fused tier-glue): +0.49 ms slower at 32 rows already.
    - The launch-fold lever is unsized until a quiet boundary microbench runs after tue1. Fold estimates go down.
    - So most of what is left is Breakthrough's expert kernel (~-5.5 ms), if it lands. Per-class numbers after tue1
      (~12:00).
    - **The orchestrator's read: Saturday's gated lossy number is more likely ~200-220 (Q8 + 32K, check ~35-37 ms),
      ~240-250 only with the expert kernel. Exact ~130.**
  - **09:56 / 10:17 MoonShots: Q8 = PASS; the decided r57 config Q8 + 32K (Q8H32) costs ~2.4 % kept on HE+, ~0 on
    MBPP** (q8early_scores.txt, h32early_scores.txt).
    - Q8 alone: HE+ x0.995 / x0.998, MBPP x0.996 / x1.0035.
    - The 32K screen alone: HE+ x0.991 / x0.987. The screened kernel is bitwise on 124 / 124 windows.
    - Q8H32: HE+ tree-16 x0.9775, tree-32 x0.9751; MBPP x0.998 / x1.005.
    - 32K beats 8K by ~1.5-2.4 pt on HE+.
    - The stored format to price: t30b's 10 trained Linears + the head's bf16 HC mixes at mx.quantize(bf16, gs 64,
      8 bits), the router at 8 bits, routed experts as stored, top-10.
  - **10:2x MTP Head: S20 SMALLHEAD = KILL** (REPORT ec48479): a head MLP at 0.53x / 0.46x params keeps x0.69-0.80, so
    path B (a smaller student) is CLOSED.
    - The draft target rests on path A (MoonShots' C2, 76.7 MB per level; kept read-out running) + a f ~0.1 kernel:
      tree-16 4.6 ms, tree-32 7.5 ms.
    - **Orchestrator: GO on FAR's LEVEL PAIRING** (one level drafts 2 tokens where p1 >= 0.95; tree-32 7.5 -> ~5 ms):
      a sandbox screen (~2 h + ~30 min GPU, <= 4 GB) for the post-Saturday draft wall, pre-registered.
  - **11:0x MTP Head: LEVEL PAIRING = KILL at stage 0** (REPORT 'S21 PAIR', CPU, before any GPU).
    - Mechanism gap: a 2-ahead output saves only leaf levels. Continuing a path needs the skipped depth's state + K / V,
      i.e. a trained double-step transition (~1-2 days).
    - A perfect pairing saves 23-34 % of levels on HE+ and 12-27 % on MBPP (t30b TOPK dumps), under the 20 % line on
      both sets. Best case ~1-1.5 ms per tree-32 block.
    - **Orchestrator: MTP Head's next job is the Q8 + 32K t30b artifact for BigMoE's r57** (MoonShots' measured format,
      sha-pinned, verified vs Q8H32's kept), by Wednesday morning. Then, at the desk, a pre-registered path-A QAT
      recipe for after Saturday.
  - **10:2x MTP Head: the t30b Q8H32 engine artifact is BUILT, VERIFIED, PINNED and handed to BigMoE.**
    - ~/datasets/deepdraft/q8h32/t30b_q8h32.safetensors: 94.4 MB, sha256 b5388cc4...e9c4. The orchestrator re-hashed
      it: it matches.
    - All 16 modules, dequantized, equal MoonShots' apply_cut(Q8) BITWISE. So the expected kept is M3's Q8H32 (HE+ t16
      x0.9775).
    - Hand-off note: TrainLab data/q8h32/HANDOFF.md (7d6c47c), with the in-process recipe too. t30b as stored stays the
      CERTIFY98 candidate.
  - **11:0x Leap tue1 DONE** (re-armed with a NO-SWAP gate of 36 after the 39 gate held it idle 10:20-11:03; quiet run
    11:04-11:05:35, C 160, 36 checks per arm; data/week98/c25/tue1_run.txt):
    - Walls [K]: F16 38.92 ms; F32c6i 56.12; F32c6g 55.04. GPU-paced at both sizes (96-97 % busy; CPU 23-25 ms).
    - gdn_fast at 32 rows: -1.19 ms, bitwise.
    - int8 dense in the engine: +1.06 ms SLOWER, so G1 goes int8-free.
    - Per class, GPU ms (16-row / 32-row):
      - experts 17.44 / 26.31 (45-47 %, the lever);
      - gdn_proj 6.34 / 8.25;
      - hc_read 4.71 / 5.52;
      - gdn_core 2.10 / 4.25;
      - att_proj 1.94 / 2.62;
      - moe_shared 1.18 / 2.17;
      - router 1.20 / 1.66;
      - att_core 0.80 / 1.12;
      - ple 0.57 / 0.80;
      - hc_write 0.24 / 0.69.
    - Re-priced [P]:
      - 16 rows (Saturday): GDN16 -0.6 + Breakthrough's tgl16eu -2.1 (FAST; exact tgl16e -0.5) + G1 lambda bytes -0.3
        -> ~36 ms.
      - 32 rows: ~48 ms. 28-30 is out of reach without a replayed / compiled check.
      - att_core (module 5) is closed by size.
    - **Orchestrator: Saturday's gated lossy at a ~36 ms check + ~16.6 ms sync-free Q8H32 draft + ~3 ms is ~205 tok/s
      (HE+, kept 11.45).**
    - BigMoE 08:3x sizes:
      - tonight 0.1-1.5 GB;
      - Wednesday's CERTIFY98 capture ~10 GB (9-12) + TOPK / tree dumps ~1-1.5 + ~1 GB temporary parts = 11-13.5 GB,
        i.e. ~92-94 GiB free without deletes (below the floor).
      - Slim option: raw taps + tk only, ~5.5 GB (NewIdeas + EvalSuite decide whether the proofs need the final state).
      - Outcomes: slim + both deletes ~105 GiB; full + deletes ~101-103.
    - Leap S2 08:28: the one-call at 64 rows = 1.229x (COH off) / 1.285x (on) the 32-row call: BETWEEN. Pessimistic:
      chain rows keep only 5.64 of 10 uses inside the union. Fill + sibling-like variants are next (~10 min), then the
      S1 call.
  - **08:3x Lunatic, BREAK98b (IDEAS_1117b_limits.md): kept 24 for a 32-row price is beyond (a)-(c) on this Mac.**
    - (a) An exact extra row costs >= 0.25-0.37 ms (dense 3.73 G MAC = 0.27 ms bf16 at 14.0 T MAC/s, ~0.15 with int8
      activations; union +0.06-0.10; cores 0.04), so 96 rows cost >= 24-36 ms. Own-information trees peak at 48 rows
      (581 vs 566 at the bound, +3 %).
    - (b) A dependent decision needs a second weight stream (>= 0.57 pass), so two cost >= ~1.6 passes. Hindsight-
      perfect selection from the head's 127: 18.48 / 21.39 / 22.45 / 23.70 at 32 / 48 / 64 / 128 rows.
    - (c) At the chain's first misses the 2B / 4B top-1 is right 52 / 58 % (head #2 58 %, head + true state 68.5 %).
      As deciders <= 15.6 kept; as siblings they lose to best-first at equal rows. (DISCREPANCY with FAR's 67 / 72 %:
      NewIdeas reconciles.)
    - **(d), best: re-aim the week at the cycle.** Tree-32 + CC(1) (~17.0 kept) is 550 at a ~26.5 ms check (~95 % of the
      C6 + int8 bound) with a 3 ms draft; 523 at 90 %; 480 at a 7.4 ms draft. Line: check <= 28 ms + draft <= 3-4 ms
      keeps 550 alive; a check > 30 ms or a draft > 5 ms kills it for any kept lever within these limits.
  - **08:30 Leap S2 RESULT:** the one-call at 64 rows HOLDS with realistic placement: sib 1.19x, fill 1.15x (COH on
    1.13x); chain 1.23-1.32x. Projected: 96 rows ~1.46-1.57x, 128 rows ~1.70-1.87x. So the padding absorbs rows only up
    to ~64 (s2_bench.txt).
  - **08:3x orchestrator, REVISED: S1 (tree-chunk GDN) is DEFERRED despite S2 holding.** Lunatic's limits put big trees
    at +3 % at the bound (~+9 % at today's check; ZERO +5-10 %): not worth 2-4 h of Leap's time this week.
    - **Leap's whole line is the 32-row check, 56.4 -> <= 28-30 ms** (Lunatic's (d) line for 550).
    - The int8 NAX register peak (~10 min) still runs: today's experts are compute-bound, so the int8 matrix rate
      decides the FAST expert path for the 32-row check.
    - tue1 at 09:30 as armed.
  - **08:3x Fable C, BREAK98b (IDEAS_1117b_closure.md), AUDIT of VERDICT-DEPTH: the conclusion stands, but for COST,
    not readout.**
    - The readout was a ridge (3,200 rows per parity, 10,240 dims, lambda pinned at 0.1). So "the 125B decides in its
      last ~4 layers" is NOT established; what is established is "no cheap readout reads the decision before the last
      layers". A translator trained at scale would settle it but changes no Saturday decision.
    - It closes repair regardless: with the EXACT L48 verdict, FIX-r is flat at the bound (430-466 vs 462 tok/s). At
      L <= 24 the decision rate (24-29 %) would need ~70-80 %.
    - (b) is a KILL by arithmetic: a second pass is >= 0.66 of a pass, so 23.65 kept costs 1.66-1.78 passes = 13.3-14.2
      kept per pass < tree-32's 16.26. So (b) = (a): kept 24 at a 32-row price needs a marginal row <= ~0.08 ms (0.28
      today).
    - (d): the recorded levers stacked and halved give kept ~19-21 at ~1.1x cost. 550 on HE+ is then cycle arithmetic:
      kept 20 needs check 25-30 + draft 3-5 ms; kept 24 clears it at check 32 + draft 5.
  - **08:32 Leap, the int8 NAX register peak (nax_peak_i8.txt; rough, shared GPU):** int8 x int8 -> int32 on Metal 4
    matmul2d is 1.4-1.7x bf16 at 32x32x32 (15.2-20.4 vs 10.8-12.2 T MAC/s), 1.15-1.53x at 16x32x32, ~none at K = 16.
    Quiet int8 is ~20-25 T MAC/s [P].
    - Route (a)'s 2.5x premise FAILS (BETWEEN on Fable B's lines). Big trees stay a small lever.
    - For the 32-row check: the experts' MMA is ~150 of ~530 us busy per call, so W4A8 (int8 activations, lossy,
      EvalSuite-gated) would save ~-2 ms per check [P]. It goes on the check-module list as an option; it is not the
      28-30 ms lever.
    - Capture ask: layers 32 / 40 / 44. **The orchestrator put them into TODAY's capture (07:5x)**, with the two verdict
      screens run first.
    - It costs nothing in the check, so the gain holds at any check speed.
    - Risks: clustered real coverage; whether the head can use a state stale by one (needs training). The orchestrator
      asked that the pre-registration answer MoonShots' KILL of the close cousin (R2 <= 0.22, 0.67-0.76 rad).
    - Screens: Sandbox98's SIB harness after FAR's screen (PASS G >= 0.5 HE, ~30-50 min), then the 125B on the capture
      (ask: token id + head-input state per mid_* row, ~0.2 GB).
    - Information-line order (NewIdeas): LANDMARK-ANS > early death / in-check repair > FAR's rows.
    - The draft line went to MTP Head + MoonShots; they answer by ~09:45.
  The queue is re-pointed: no new small-lever work, and the GPU nights go to these tests first.

## KAMEN, Tue 17 Nov 07:1x (verbatim, dictated while walking, after the 06:4x lossy correction): 775 ON CODE, NO COMPROMISE
"Why so low? What are these numbers? They were at for projection of 536 tokens per second. I mean, 150 is lower than
what we have right now. Is this on chat or what is this on? But I want to reach at least 600 tokens per second from
Claude. Actually, no, as I told you, there is no room for compromise. We need to achieve 775 tokens per second on
Claude. I need this breakthrough, you know it, so do it."
- My reading: "Claude" = "code" (dictation). **Target unchanged: >= 775 tok/s lossy at quality on CODE.**
- The orchestrator's answer (07:1x):
  - All of these numbers are code.
  - The 536 was the ungated tree-32 lossy projection with the engine at its limit (~480-545), not a Saturday number.
  - Exact ~150 (replay, heplus24) is ~50 % above the published exact (101.4 on 12 copy-heavy HumanEval prompts).
  - B2 runs 117.6 on those same 12 prompts even on a busy Mac, and 80.6 on all 164.
  - 775 needs kept >= ~24 per check AND a ~25-29 ms cycle. Lunatic's ORACLE-31 caps the head's OWN information at
    18.48 kept on HE+ for a 32-row check (IDEAS_1116c_drift.md; K127 = 23.70 at 128 rows). So a better tree or
    training on the same information cannot reach it. The draft must get NEW 125B information cheaply, or a check
    must cover far more rows at the same cost.
  - No line was running on that wall at 07:1x; it restarts today (NewIdeas orders it).

## KAMEN, Mon 16 Nov 17:3x (verbatim, the target part): 775-800 LOSSY, AT QUALITY, BY SATURDAY
"We can't stop at what is the estimate you gave me for Saturday, 270 to 340 tokens per second or something like that.
It can't be that. It must be that 800 tokens per second we said. I don't care how you achieve it. I mean, it just has to
pass, and that's for Lossy, 400 and something exact." ... "I'll be fine with 800 or even 775, but not less, in lossy
mode, which passes the quality and everything like that." ... "if you want, don't run any big tests at all. Only go
through ideas and then build on their projections without having built large tests and try to break the walls" (his
guess, "not a suggestion on the way you should do it").
- **Targets now: >= 775 tok/s LOSSY (measured, whole decode, at quality) and 400+ EXACT, by Sat 21 Nov.**
- Plan and frame: PUSH800.md (17:4x). ~~At today's kept 16.91, even 100 % of every peak with free drafting gives
  707 / 779 / 821 ... kept 21-24 per 32-row check (x1.25-1.42).~~ **CORRECTED 17:5x (NewIdeas; I had used Leap's
  pre-13:45 table with the old expert sizes):** real bounds at 100 % of every peak are 27.7 (C5) / 25.2 (C6 + int8) /
  24.0 (C6 + V4 + int8) ms (dense int8 is byte-bound at 7.6 ms), so today's kept gives at most 610 / 671 / 705 with
  free drafting. NO byte plan reaches 775 at today's kept. 775 / 800 need kept 24.2-27.3 per 32-row check (x1.43-1.62,
  ~99 % effective acceptance per position) at 85-90 % with a 3 ms draft. Exact 400: kept >= ~25 at the 57.9 ms floor.
- Correction to the record: exact measures ~144 tok/s today (tree-16, measured engine). "450 exact" was the ceiling
  math at a 97.8-98 % drafter, not a measurement.
- 17:4x: ~~#8 (span close-call in comments) and #13 (inert comment / docstring acceptance) ON under "I don't care how"
  (Kamen may veto)~~ **WITHDRAWN 18:2x: #13's figures were the WHOLESALE form, a recorded quality KILL (CD16-DROP
  11-06: MBPP12 8/12 vs 11/12, comment loops + code drift); the guarded forms give x0.997-1.019; #8 is already in the
  shipped base. I turned it on without grepping TRIED.** THINK_1116b sprint to ~19:00; Leap asked for an engine module split.
> (a quote from the project owner, kept private)
  certain is a way to get it down to something like 1 ms or 2ms".** Context: in c25, the lossy 32-row check is build +
  submit 50.95 of 56.41 ms (GPU wait 5.05, commit 0.35); F16 34.90 of 38.88. ~~**The REPLAYED COMMAND PATH is RE-OPENED
  as the main engine bet**~~ **CORRECTED 18:0x (Leap 18:0x): "build + submit" is the forward call's WALL, and the CPU
  is blocked inside async_eval while the GPU runs; it is NOT 51 ms of CPU work. Evidence: CHECK55's profile on the same
  harness (exact C32one) had the GPU busy 89.3 of 90.5 ms (99 %) with only 26.3 ms of CPU, and +2 ms of CPU moved the
  wall +0.13. In c25, byte-only cuts move the wall (C6 -1.20, V3 + V1 -1.90) while a launch cut does not (t1cf +0.49).
  Prediction: F32c6i's GPU is busy ~50-54 of 56.4 ms, so a perfect replay wins ~2-5 ms today. Rule (pre-registered by
  Leap): GPU busy >= 90 % of the wall = GPU-paced (kernel modules lead); GPU idle >= 30 % = CPU-paced (replay leads).
  The CPU still binds at the target (~28 ms of CPU vs a ~29 ms check), so compile / specialize stays a module, and
  the replay one-layer prototype runs in parallel. The orchestrator relayed the wrong reading to Kamen at 17:5x and
  corrected it at 18:0x.**
  **REFINED 18:0x (Leap's 4-layer slice, thread CPU; data/week98/check25/cpusplit_slice.txt):** the CPU side IS large.
  At 32 rows the forward submits nothing until the logits' eval: Python builds the lazy graph at ~0.45 ms per layer and
  MLX encodes ~3 us per dispatch, so ~20 + ~12-14 = **~30-35 ms of CPU per 32-row check**. Still open:
  - if the engine's in-forward waits are GPU-paced (GPU busy ~50), a replay wins ~2-6 ms today;
  - if they serialize CPU with GPU (GPU busy ~35-40), it wins ~15-20 ms.
  Tonight's c25_cpusplit run decides (Python build / encode / blocked / GPU busy per check). Either way, compile /
  replay is REQUIRED for any check under ~35 ms, so it stays module 1. (record the check's GPU commands once, replay per check). This is a new mechanism vs the old
  KILL, which assumed the exact engine is GPU-bound. Leap = module #1 (MoonShots helps). First: split the 50.95 ms into
  Python build / MLX encode / blocking, plus the GPU busy time (libmtlprof). After the CPU part goes, the check falls
  to the GPU's own time (the bound is 25.2 ms at 100 % for C6 + int8). Kept still decides 775. The early app quit
  frees nothing (only Sleight, 35 MB, was open).
> (a quote from the project owner, kept private)
  sub 10ms everything. Not 30 and I'm pretty positive it's extremely possible and actually reletavely easy".** So his
  idle chats stay, and V2 moves to Tuesday (4d). Leap is asked to MEASURE the check's GPU busy time tonight, which
  settles it. c25's 5.05 ms is the wait after the last submit, while the GPU works underneath the 50.95 ms build:
  c25's own floor for F32c6i is 18.2 ms (5.09 GB at 280 GB/s). ~~the measured kernel speeds put GPU busy at
  ~30-45 ms [P]~~ Leap's prediction is ~50-54 ms (see the 18:0x correction above).

## KAMEN, Sun 15 Nov ~01:0x (verbatim): DON'T STOP AT WHAT PAPERS SAY
"Don't rely on the published papers only. Because one paper said that for example the extra layers suck, doesn't mean
that in our use case we can't spin things up so that they become really valuable. What was it "one man's trash is
another one's treasure". And also 400tpa exact is fine and also I know that the 1k would be for code specific tasks and
chat will be a bit less don't worry. So yeah. Just because one paper said something doesn't work doesn't mean we can't
get it to work."
- **Targets now: 400 tok/s EXACT; ~1,000 tok/s LOSSY on code tasks** (chat a bit less, accepted).
- **Rule:** a published negative is a hypothesis about THEIR setup, not a verdict for ours. Before closing a lever on a
  paper's word (or on one short screen of ours), ask what setup could make it work here (init, lr, budget, data,
  where it sits, what it reads) and screen the best version. Depth is re-opened under this rule: tonight's zero-start
  layer at 3e-5 barely grew in 400 steps, which is not a fair test of depth. Try a copied (non-zero) init, its own
  higher lr, the full budget, and borrowed pretrained depth.

## THE DRAFTER SPEC FOR 400 EXACT (orchestrator from Leap 01:1x)
- At tau 23.8 (98 % to 30), 400 tok/s needs a cycle <= 59.5 ms: the 52.0 ms check floor plus <= 7.5 ms for drafting AND
  overhead. Overhead is the commit replay (3.9-5.4 ms, ~-2.5 with a batched commit) plus seams (2.5-3.6).
- So the drafter gets ~2-4 ms per 32-token block, and it must be small in BYTES too, because the check is
  bandwidth-bound and any concurrent read adds in full (the ANE / second-stream overlap can't hide bytes).
- **Goal A, precisely:** >= 98 % per guess to depth 30, reading <= ~0.5 GB of weights per 32-token block (~2 ms at
  280 GB/s), with its compute hidden or trivial.
  - Reference points: DFlash2 drafts a 32-row block in 10.96 ms (0.52 GB, Leap R-10, 11-03); today's chain head reads
    5.4 GB per block (>= 19 ms).
  - C2 DENSE32 (Leap, tonight): if mr2 32-row stays bitwise at R <= 1.3, it becomes REF2's 17-32-row verify path (the
    build).

## THE TWO GOALS (Kamen, Sun 15 Nov ~00:4x, verbatim)
"so yeah, we must do that by the other saturday and havbe a 100% certain path ... so basically we have two main goals and
they dont differ from lossy or exact. the first one is simple - just find the perfect way to keep 98 % of tokens, 30
tokenms deep. thats it. then the second one is the check. it must happen in sub 25ms(for lossy but we generally have to
push it to its absolute limit regardless and reach that 49ms for exact). and we must do that by saturday."
- **GOAL A, the drafter:** >= 98 % per guess held to depth 30. The method gets proven by held-out tests by Sat 21 Nov.
- **GOAL B, the check:** a 32-row check under 25 ms (lossy, EvalSuite-gated), and the exact check pushed to its limit
  (floor 49.1 ms at 45 GB). Note: 100 % of memory bandwidth is not reachable by any kernel; dense decode reaches
  ~77-89 %, so ~55 ms is the realistic exact target.
- Tue 17 11:0x (MTP Head, REPORT 77090eb; CPU LAB): **LEVEL PAIRING KILL at stage 0.** A perfect pairing saves at
  most 12-27 % (MBPP) / 23-34 % (HE+) of levels at depth 20 on t30b's dumps (pessimistic-optimistic). FAR's
  mechanism also lacks the double-step state + K / V transition. HE+ best case is ~1-1.5 ms per tree-32 block on
  path A, not worth a 2-day build. Draft line from my side: bytes = path A + MoonShots' kernel.
- Tue 17 10:2x (MTP Head, REPORT ec48479; LAB sandbox): **S20 SMALLHEAD KILL: a 2x smaller head loses 20-30 % kept,
  so draft path B (a small student) closes.** The draft target rests on path A (MoonShots' M3 C2, 76.7 MB, tree-16
  ~4.6 ms at f 0.1) + the persistent kernel. t30b is PINNED for CERTIFY98 (TrainLab data/t30b_pin.md; EvalSuite
  confirmed the hashes).
- **Tue 09:4x NewIdeas: CORRECTION to the 10:00 report's Saturday range. The ~270 realistic / ~375 stretch was the UNGATED tree-32 projection.
  Saturday's GATED, MEASURED config runs TREE-16 (t30b + tree-16 + CC(1) + G1's pick, on the 16-row check): ~200-250 tok/s HE+.
  The EXACT headline (ii) is ~130.** [P from parts]
  - Kept: tree-16 11.26 x CC(1) ~1.04. Check: 38.6 ms today, ~31 with Leap's 16-row modules.
  - Draft: set by t30b's FORMAT in r57 (MoonShots' read of dd_tree.py):
    - as stored: fp32 dense + the full 248K lm_head + a host sync per pop, ~676 MB per pop, ~35-43 ms per tree-16 block -> ~145-160 tok/s;
    - bf16 + 32K screen: ~17 ms -> ~200-230;
    - Q8 + 32K: ~14 ms -> ~209-242;
    - 4-bit + 8K: ~10 ms, kept x0.94 -> ~213-250.
  - Recommended (to the orchestrator + BigMoE): Q8 + 32K if MoonShots' Wed ~07:30 Q8 read-out holds kept (the 32K screen is free in MTP
    Head's sandbox), else bf16 + 32K; never as stored.
  - fused2 is OUT for this loop (bitwise only for the stored MTP head on a chain). Sync-free is IN (MoonShots, bit-tested on BigMoE's
    ops by Wed evening, ~-3 ms per block).
  - CERTIFY98's chain-32 count check runs the PINNED stored format, or is amended.
- **Tue 09:3x NewIdeas: the three-line REPORT to the orchestrator (~/NewIdeas/research/week98/REPORT_1117_1000.txt). Saturday HE+ lossy,
  ungated, from parts: REALISTIC ~270 tok/s (check ~45 [Leap] + draft ~15 [the stored head, 177.6 MB per level] + 3; kept 16.91);
  STRETCH ~375-410.** 550 = check <= ~28 + draft <= 3-4 at kept ~17-19 (Lunatic / Fable C / NewIdeas agree that kept is exhausted near
  17-19 this week).
  - Information (BREAK98 + 98b) is closed on measurements: route 1 by partial depth on the 125B capture; early death stage 2; FAR's
    future-state rows (Sandbox98, negative); FAR's 2B tree (52 % at walk first misses; loses to best-first at equal rows); free rows and
    two-decisions-per-pass by arithmetic. OPEN: LANDMARK-ANS (free; Sandbox98) and 64 cheap expert rows (Leap S2: 1.15-1.19x).
  - Draft: MTP Head withdrew 8-bit + screen as a byte lever. Paths: A (a tree-16 ~74 MB stack, priced by MoonShots on M3) and B (a
    distilled ~25-30 M student; sandbox S20 today).
  - CERTIFY98 Wednesday: SLIM, option (b); the candidate = t30b AS IS, pinned by Wed 18:00 (MTP Head).
- **Tue 08:1x NewIdeas (BREAK98, on BigMoE's day capture, HE+; self-checks all pass): ROUTE 1 BY PARTIAL DEPTH IS CLOSED,
  AND EARLY DEATH IS KILLED, on the 125B itself.** (~/NewIdeas/research/week98/BREAK98_SCREEN.md, 2b30212 + this entry)
  - VERDICT-DEPTH (Lunatic's floor_lens, number-checked, adapted to the widened capture): the head's own margin locates its first
    misses at AUC 0.954. The 125B's lens: 0.69-0.73 at layers 2-32, 0.839 at 44, 0.97-0.999 only at the final layer. It picks the
    right token at 20-29 % of first misses by layer 32 and 37-42 % at 44. In-check repair driven by any layer <= 44 LOWERS kept
    (L16 10.5, L24 11.3, L44 13.8 vs 13.77). The decision forms in the last ~4 layers.
  - EARLY DEATH stage 2 (pre-registered): droppable children at <= 2 % false drops, P-lens / P-tuned 2-5 % at L8-24, P-draft (the
    head) 31.4 %, combined 24-26 %. KILL on Fable B's line (< 60 % at L16).
  - What stays open: LANDMARK-ANS (free; Sandbox98 ~10:15) and FAR's placeholder rows (Sandbox98 now), both ~+5-15 %
    realistic. Route 2 is left with the per-row ALU cost (Leap's check line). MBPP's capture (landing ~08:27) is run for completeness.
- **Tue 06:4x NewIdeas (slot order, on the orchestrator's 06:5x ask): Saturday's two headline measurements are PLACED (the
  owners are confirming harness and length).** Both: a quiet ticket, HE+ 164 + MBPP 100, one order, n / decode wall per set and
  pooled.
  - (i) LOSSY: the gated chain-16 stack on its pinned env line, the same one tier 1 + the card gate run (EvalSuite, or
    BigMoE).
  - (ii) EXACT: t30b + tree-16 fixed, after tonight's M4 PASS (BigMoE).
  - **Thu night order** (cool, after the 23:30 quit): ADAPTK's C T T C (only if its bit test passed) -> (i) -> (ii) -> the
    proofs' COUNT check (results/certify98/ENGINE_CHECK_END) -> the card gate's remainder. Fri night is the reserve.
  - (ii') EXACT + ADAPTIVE K: only if it passes all its gates by Fri 12:00; Fri ~12:00-15:00, else at the head of Fri
    night. A missed (i) / (ii) moves to the head of Fri night.
- **Tue 06:4x orchestrator (NewIdeas' correction below, checked: r57_screen_g1b.py lines 125 / 1236 and 0 tree
  references; ccx45cc1's own log reads "[code: dfl] ... 23 checks (kept 10.35)"): option (a) for Saturday. Gating the
  ladder config (b) is next week's FIRST lossy job.**
  1. **Saturday:**
     - LOSSY at quality = the gated chain-16 r57 stack (lossy-45 + CC(1) + G1's pick), measured by whole decode on
       its env line.
     - EXACT = t30b + tree-16 (+ ADAPTIVE K if it passes Fri 12:00), with no quality gate needed.
     - LADDER v4's lossy rows and every "lossy at quality" price for tree-32 (mine at Tue 00:5x, ~277-315, included)
       read "projected from measured parts, quality NOT gated".
  2. **Two clean measurements for Saturday** (NewIdeas places both; quiet, known benchmarks, n / decode wall):
     - the gated lossy stack on its env line;
     - exact t30b + tree-16 after M4's PASS (+ ADAPTIVE K's arm if it passes).
     **PLACED (NewIdeas 06:4x, approved by the orchestrator; owners confirm harness and length):**
     - (i) LOSSY: the gated stack, whole decode on its pinned env line (EvalSuite, or BigMoE if EvalSuite's runner
       cannot print n / decode wall).
     - (ii) EXACT: t30b + tree-16 fixed, after tonight's M4 PASS (BigMoE, the M4 harness).
     - Both: HE+ 164 + MBPP 100, one order, per set + pooled, quiet ticket, ~20-40 min each.
     - Thu night order: C T T C (if its bit test passed) -> (i) -> (ii) -> the proofs' count check -> the card gate's
       remainder.
     - (ii') EXACT + ADAPTIVE K, only if it passes all gates by Fri 12:00: Fri ~12:00-15:00 after Leap's Friday check
       (quiet, away >= 900), else at the head of Fri night. A missed (i) / (ii) also goes to the head of Fri night.
     - **The orchestrator's three conditions:**
       - No heat pause (fanwatch SIGSTOP) inside a timed decode. The fans are at Kamen's 5k line, so log fanwatch's
         pause lines for the window. Any pause inside a set re-runs that set.
       - Each number prints its config next to it: drafter, rows, tree or chain, acceptance rule, env blob.
       - (ii) and (ii') print token identity against the exact reference per prompt (264 / 264). Any mismatch means
         it is not exact.
     **Reference points for Saturday (06:5x), NOT measurements:**
     - B2 = the lossy-45 base, chain-16, dfl on all 264 code answers, share lock, not quiet. Whole decode n / decode
       wall, recomputed by the orchestrator from es_cdlx45t1c1_rev_170523_*_runs.json (it matches EvalSuite):
       - "humaneval" 60 keys 107.8; "humanevalx" 104 keys 71.5; so all 164 HE+ keys = 80.6;
       - MBPP 100 keys 75.5;
       - pooled code 78.6; IFEval 35.9.
     - EvalSuite expects the gated stack near 80-100 tok/s.
     - Exact t30b + tree-16 replay (E14's quiet costs, draft counted, decode only; dev sets heplus24 / mbppbank): HE+
       147.6, MBPP 100.0. So the EXACT headline may well beat the gated LOSSY one. Thursday's two runs settle it.
> (a quote from the project owner, kept private)
  3. **Card-gate scope (EvalSuite's ask): the FULL card gate (A), 5.8-6.5 h, with lcb6 in every lever's sets.**
     - lcb6 joins every lever's sets (BigMoE writes it), so LCB v6 tests the stack on long code. Without it the card
       gate tests no lever.
     - IFBench / GPQA-D stay in full. Tier 1 compares against lossy-45, not exact, so the card gate is the first and
       only comparison of base + stack against NAX6-N on those sets.
     - ~~Identity checks against cached base rows, ~3 h~~ (CORRECTED 06:5x, NewIdeas / EvalSuite: no base card rows
       are cached for lossy-45 / lx45g1b, so that would cost a base run first).
     - Fallback (ii) if tier 1's PASS or runner 69's dry passes are not in by Wed 21:00: LCB v6 both orders (~3 h) +
       a switch-off spot check (20 IFBench + 20 GPQA keys, stack vs base, ~40 min).
     - EvalSuite files the scope as a dated amendment before any card-gate data. Adoption stays Kamen's.
  4. **Sizing (EvalSuite's numbers; replaces items 1-2 of 06:2x):**
     - Stack tier 1: Wed ~10:30-13:30, after G1's verdict (~09:30-10:30).
     - The card gate: Wed night right after the capture (~2.1 h; ~5 h of the gate fits there), if tier 1 passed
       and runner 69 is ready.
     - Thu night, in this order:
       1. ADAPTK's C T T C first while cool (~30 min, only if its bit test passed);
       2. the proofs' engine check (COUNT, ~30 min; marker results/certify98/ENGINE_CHECK_END, named Wed night);
       3. the card gate's remainder.
     - Fri night: the reserve. Expected verdict: Fri morning.
     - No TREE16 arm is in EvalSuite's records. One goes in only if BigMoE or Leap name it.
- **Tue 06:4x NewIdeas: CORRECTED: the lossy stack being GATED is CHAIN-16 with the DFlash (dfl) drafter, not the ladder's
  config.** (EvalSuite found it; I verified it in BigMoE's data/r57/snap/g1b/r57_screen_g1b.py: line 125 R = R57_ROWS or 16;
  line 1236 allows 32 rows only where the copy runs; no tree path.)
  - Every gated env (lx45cc1, ccx45cc1, lx45t1c1, lx45g1b, the G1 envs) leaves R57_ROWS / R57_CHATROWS unset. Drafter:
    dfl, d7b on code, copy g2. Gated code kept per check is 7.0 / 7.4.
  - Reference point (EvalSuite 06:4x; share lock, NOT quiet, NOT a measurement): this morning's B2, the lossy-45 base on
    chain-16, decoded code at 78.6 tok/s whole decode pooled (humaneval, the 60-key subset, 107.8; humanevalx 71.5; mbpp
    75.5; 31.7 with prefill). All 164 HE+ keys pool to 80.6 (the orchestrator's recompute), so compare Thursday's (i) with 80.6. Expect Saturday's gated-stack number near 80-100 tok/s. For comparison, the exact t30b + tree-16 replay on E14
    costs gives HE+ 147.6 / MBPP 100.0 (dev sets heplus24 / mbppbank, decode only; not the 164 keys), so the EXACT headline may
    come out above the gated LOSSY one. (i) and (ii) must run the SAME 264 code keys (HE+ 164 = humaneval 60 + humanevalx 104,
    MBPP 100).
  - So LADDER v4's lossy rows (t30b, tree-32 + CC(1), kept 16.91, HE+ ~310-430) and ADAPTK-LOSSY's "serves the lossy
    headline" are UNGATED projections. Rows, trees and the drafter decide which rows exist, and with them close calls,
    1c tiers and COMQOM / RES rows, so this week's gate verdicts do not transfer to them.
  - Saturday's lossy number at quality = the gated chain-16 r57 stack (lossy-45 + CC(1) + G1's pick), measured on its
    env line. EXACT (t30b + tree-16, + ADAPTIVE K if it passes) needs no quality gate. Options are with the orchestrator:
    accept, or gate the ladder config (a t30b / tree path in the r57 screen; very tight).
- **Tue 06:2x orchestrator (NewIdeas' 06:2x slot ask; LOSSY_GATE section 4): the lossy STACK's quality gates run
  BEFORE Saturday; "after Saturday" is only the fallback.** (Items 1-2 SUPERSEDED by 06:4x's sizing and the ~3 h
  card-gate scope.) The stack = lossy-45 + CC(1) + G1's pick. G2 is out
> (a quote from the project owner, kept private)
  combination.
  1. **Tier 1, both orders (~2.5 h, 45 GB):** Wed by day in G2's freed slot, chained on G1's verdict + pick, ending by
     Wed 21:30 (>= 2 h before the capture). If G1 settles too late for that, Thu by day after the TOPK / T8-32 dumps
     and Leap's slot.
  2. **The card gate (IFBench 150 + GPQA-D 100 + LCB v6 131, both orders vs NAX6-N; 5-7 h):** chained on tier 1's
     PASS, split at process boundaries:
     - Wed by day to 21:30;
     - then Thu night AFTER the proofs' GPU steps, chained on their end marker;
     - then Fri night as the reserve.
     The capture, the proofs and Leap's Friday measured check keep priority.
  3. **A verdict by Sat 08:00 counts for Saturday.** Otherwise Saturday's lossy number reads "measured; card gate
     pending", and the gate finishes after.
  - (Moot for Saturday under 06:4x's option (a): the lossy headline is chain-16.)
  - **Asked of NewIdeas + EvalSuite (their call):** under CC(1) the tree can change WHICH tokens are kept (exact mode
    cannot), so ADAPTIVE K may change lossy outputs, not only speed. If Saturday's lossy headline uses ADAPTIVE K, the
    card gate's config must name the tree policy; otherwise the lossy headline stays on fixed tree-32.
  - **Verified in the records (orchestrator 06:2x):**
    - CC(1) PASS in both orders (n 528, -1.14 pt, p 0.26; EvalSuite 61f5f46);
    - d2 FAIL (MBPP -6.00 pt, pooled -4.17, p 0.013);
    - tier-0a lines ADOPTED (A 0.9775, F 0.00 / 1k; 7528c5f);
    - BigMoE's probe re-run PASS 3 / 3 at 05:21:59 (peaks 42.4-42.7 GB, swap 0, 0 swapouts; G1P_PROBE_DONE);
    - B2 (lx45t1c1 reverse) running since 05:23;
    - COMQOM 07:00, the band re-run and the Fit12 kill test are armed.
- **Tue 04:5x NewIdeas: ADAPTIVE K DOWN (verify 8 rows when the drafter is unsure) = a SMALL lever. It passes on CPU;
  the engine's 8-row cost decides.** (~/NewIdeas/research/week98/adaptk_k8.txt, adaptk_k8lossy.txt; ADAPTK.md)
  - PEEK rule: verify 8 rows iff the 8th pop's path_p < ~0.25 (cross-fit), else ADAPTIVE K as frozen. path_p is
    calibrated at 8 rows too (E[kept8] 7.17 vs real 7.20 on HE+).
  - Exact, E14 costs, L 0.66, over KERN2, at an 8-row check of 0.78 x the 16-row one: HE+ +1.6 % [+0.2, +3.0], MBPP
    +3.0 % [+1.8, +4.1]. At 0.71: +3.9 / +7.8 %. At 0.86: ~0.
  - 0.78-0.80 is today's engine [est]: 16-row tile padding leaves only the expert union and the cores to shrink.
  - Lossy CC(1) is similar: v4pass at 0.78 +0.9 / +2.4 %; byte floor +2.0 / +4.1 %.
  - Kill test: one 8-row timing in any E-run (~1 min). KILL if >= 0.82.
- **Tue 04:4x NewIdeas (number check of EvalSuite's night1117, results/night1117_chain.log): close-call d1 PASSES tier 1
  in BOTH orders; d2 FAILS. The close-call lever ends at d1.**
  - d1 (ccx45cc1 vs lx45cc1, code tier 1): forward n 264 -1.14 pt (McNemar p 0.55; arm-only 4 / reference-only 7);
    both orders pooled n 528 -1.14 pt (p 0.26; 7 / 13): PASS vs the -3 pt screening line. Speed +6.2 % forward /
    +4.5 % reverse on all 264 keys (+5.7 / +2.9 % on keys uncapped in both).
  - d2 (ccx45cc1d2): forward n 264 -4.17 pt, p 0.013 (3 / 14): FAIL (MBPP, pooled); reverse skipped. Speed +9.2 %
    vs lx45cc1, i.e. +2.8 % over d1's forward (7.661 vs 7.450), as the replay frontier said (R1(2) 17.323 vs R1(1)
    16.910 kept, +2.4 %). The 2B predicted the fail (-6.06 pooled).
  - Read: CC(1) is the lossy acceptance rung. LADDER v4 and ADAPTK-LOSSY already use CC(1), so neither moves. PUSH800
    item 4 loses "close-call d2" (it was x1.02-1.05 on trees). T0A (S3 p2x45rpl2, tier 0) started 04:36, cutoff 06:36.
- **Tue 03:0x NewIdeas: ADAPTIVE K also pays on the LOSSY line (ADAPTK-LOSSY, CPU replay, 20 s).** With close-call
  CC(1) acceptance (tree_cc32's rule) on LADDER v4's modules-PASS costs: HE+ +9.0 % [+4.3, +13.5] over the better
  fixed K at the frozen tau 0.9 (exact acceptance: +8.1 %); v4 stretch +9.0 %; on the 25 ms byte floor +4.8 %, because
  32 rows are relatively cheaper there. MBPP ~0, never a loss. So BigMoE's gated build serves both headlines, and
  PLAN900's conditional rung (lossy HE+ ~335-465) now rests on a measurement instead of an assumption. Self-checks
  reproduce kept 11.263 / 16.263 and 16.910 (v4's tree-32 + CC). NewIdeas research/week98/adaptk_lossy.py / .txt.
- Tue 17 01:0x (NewIdeas, ~/NewIdeas/research/week98/ADAPTK.md, f36cb3c): **ADAPTIVE K screen PASS (exact): choose 16
  or 32 rows PER CHECK by the drafter's own path_p. HE+ +8-10 % on today's kernels, MBPP ~0.**
  - The rule: best-first drafting pops while the next pop's path_p >= tau, and the rows round up to 16 / 32. Under
    calibration this is the optimal form.
  - path_p IS calibrated (expected kept 12.83 vs realised 12.82, HE+ 16 rows) and non-increasing in pop order.
  - Replay of the t30b 127-node dumps on E14 costs, draft cost per level counted (0.40-0.92 ms), tau from the other
    set:
    - KERN2 (16 / 32): HE+ +8.0-10.3 %, MBPP -0.2..+0.2 %;
    - one-pass up to 128 rows [est]: HE+ +10.3-15.6 %;
    - repeated passes for > 32 rows: +5.5-6.4 %;
    - hindsight ceiling: HE+ +17-34 %, MBPP +9.5-12.4 %.
  - Needs a 32-row TREE verify path. Next kill: fresh cells (Wed capture) at the frozen tau. Engine timing is the
    orchestrator's call.
- Tue 17 01:0x (NewIdeas; Leap's E14 00:54-00:55): **K* = tree-16, FINAL for the exact engine. CORRECTED: R32/R16 is
  ~1.44, not 1.480.** (~/NewIdeas/research/week98/KSTAR_E14.md, 1f0dd99)
  - E14: C32one / A16 = 1.433 paired median (E2's statistic), 1.437 ratio of means. The 1.480 quoted since Monday
    (CHECK55 a082f8c; Leap's costs v2 onepass 1.478) was a ratio of POOLED MEDIANS, a cross-window artifact. Monday's
    own records give 1.444 / 1.438.
  - K* (treek_final, E14 costs, drafting hidden): HE+ is a tie within +-1 % (plain 32 +0.9 %; BOF 16 +0.8 %;
    contract 16 +0.4 %). MBPP is tree-16 by 18.7-20.5 %. Draft cost and the flat lossy frontier only push further
    toward 16. Monday's "tree-16 on HE+ by ~4 %" (below) was really ~1-2 %.
  - Like for like, the levers RAISE R: BOF 1.46, contract 1.46 (they save the same ms on both shapes).
  - Lossy model (information): HE+ prefers tree-32 (+11.1 %), MBPP tree-16 (+7.7 %); pooled it is a tie.
  - CERTIFY98 amendment 8a: R = 1.433. T32/16's power was never 0.00: at 1.433 the bar is 1.535 vs dev 1.531.
- Tue 17 00:5x (NewIdeas; BigMoE's key test 00:38-00:48): **M4 -> TUESDAY NIGHT; G1 key test 7 / 9 PASS (the
  comments-only pair fails a probe gap, not a defect).**
  - M4 NOT RUN Monday: Kamen was on the Mac up to its 23:50 latest start (the away gate; NOMARK + m4_1116_AWAY_SKIP).
    New slot: Tue night FIRST, chained with BigMoE's capture in one script after the 23:30 quit. M4 latest start Wed
    01:30, then the capture by 04:00 (MoonShots' early start is 05:24). Not tonight: try B (00:49) -> E14 -> night1117
    -> T0A fill the night, and T0A gates Leap's slot #1, so a push costs B2 / V2 / G1 on Tuesday. TREE16's engine arm
    is Thu night, so the t16c freeze still has Wed by day.
  - g1b key test: 7 / 9 PASS. Scored keys exact vs lx45cc1 27 / 27, swap +0, peak 42.4 GB. g1c / g1cr fail only the
    COMQOM engagement probe (com_rows 0). The probe key mbpp/115 was picked on lx45cc1's answer (two '#' lines); g1's
    own answer leaves it at token 8 ('d' for 'dicts') and has no '#' line. So the '#' path is untested. D engaged
    through the mbpp/146 docstring (106 / 107 rows), which doesn't exercise '#' either.
  - Next: BigMoE re-runs g1c / g1cr on keys with early '#' lines (mbpp/303 + mbpp/237 proposed) in a race-free gap.
    5d fallback sent to EvalSuite: comments-only picked without a PASS -> D unless D's kill-test arm is KILL, else
    lambda 0.8e-2.
- Tue 17 01:3x (NewIdeas, ADAPTK.md f36cb3c; CPU replay of the t30b 127-node dumps on E14 costs): **LEAD, exact:
  ADAPTIVE K, i.e. choose the tree size per check.** Best-first drafting pops while the next node's path_p >= tau; the
  check takes 16 or 32 rows to fit.
  - With today's two kernels (16 / 32), at the other set's tau: HE+ +8.0-10.3 % over the best fixed K; MBPP +-0.2 %.
  - One-pass checks to 128 rows [est]: HE+ +10-16 %. Hindsight ceiling: HE+ +17-34 %, MBPP +10-12 %.
  - path_p is calibrated (expected kept 12.83 vs realised 12.82). HE+'s docstring stretches want big trees.
  - It needs a 32-row TREE verify path: TREE16 is 16 rows, and C32one exists only as a chain check.
  - **Orchestrator's call (01:4x): gated build.**
    1. The build starts on M4's PASS alone (Tue night: the 16-row tree verify exact on the full model). BigMoE (owner
       of the tree path, with Leap reviewing the C32one hook) builds a 32-row tree verify Wed-Thu as an opt-in switch,
       bit-tested against one-token decoding.
    2. Saturday's gate: the fresh-cell re-read at the frozen tau (Thu ~midday, after Wednesday night's capture + the
       127-node dumps; KILL if HE+ < +3 % at L 0.66, KERN2), the bit test, and the C T T C engine A/B, all by Fri 12:00.
       Otherwise it waits until after Saturday.
    (Amended 01:5x on NewIdeas' timing fix: fresh cells exist only on Thu. Interim at the frozen tau 0.9 (out of set
    for HE+): HE+ +8.6 % [95 % window bootstrap +6.6, +10.7], 24 of 24 windows up (min +2.4 %); MBPP +0.2 %;
    adaptk_windows.py b439d44.)
    Leap's GPU-side work (chain folds, modules) keeps priority, since it is worth more ms per check.
- Tue 17 00:5x (Leap, E14; data/week98/e14/e14_run.txt, verified by the orchestrator): **exact engine levers 5 of 6
  PASS, 1 BETWEEN; every bit as pre-registered.**
  - Times (48 checks per arm, quiet): A16 61.3 ms, C32 102.23, C32one 90.61.
  - E2 step 1: C32one / A16 = 1.433, BETWEEN (1.35-1.45).
  - **NewIdeas 01:0x CORRECTION + DECISION:**
    - The engine's R32 / R16 is ~1.44 (E14 paired median 1.433, ratio of means 1.437), not the 1.480 quoted since
      Monday. 1.480 was a ratio of pooled medians, a cross-window artifact; Monday's own records give 1.444 / 1.438.
    - **K* = tree-16 is final for the exact engine.** HE+ is a tie within +-1 % (plain, BOF, contract); MBPP picks
      tree-16 by 19-20 %.
    - Lossy (information): HE+ prefers tree-32 (+11 %), MBPP tree-16 (+8 %); pooled, a tie.
  - Exact levers, all PASS: BOF -2.36 ms (A16), BOF2-O -2.16 ms (C32one), hc_b -0.81 ms (A16). They go into ref2 behind
    switches for the exact-45 path after BigMoE's review.
  - **The reference contract** (CHECK55_LEVERS #6: fixed K-chunks and summation order for every dense GEMM, the HC read
    and the head, so the arithmetic doesn't depend on the row count) PASSES both lines: A16c - A16 -5.76 ms, C32c -
    C32one -7.04 ms (the estimate was -4). It changes the exact REFERENCE, so adopting it is KAMEN'S call (asked
    01:0x).
  - Orchestrator [P]: exact levers + contract take the exact 16-row check ~61 -> ~52-56 ms (overlap unknown: hc_b and
    the contract both touch the HC read), so exact tree-16 goes ~144 -> ~150-165 tok/s.
- Tue 17 00:5x (Leap, try B; data/week98/c25/gpubB_run.txt, verified by the orchestrator): **the lossy 32-row check is
  GPU-PACED: GPU busy 54.16 of 56.14 ms (96 %), idle 1.98.**
  - CPU thread 23.2 ms, fully hidden. Build + submit 50.72 = Python build 12.00 + MLX encode 11.17 + blocked in eval
    27.38.
  - 3,172 dispatches per check (7.7 us of CPU each) in 266 command buffers.
  - The other arms agree: F16 38.64 (97 %), F32s 67.74 (96 %), F32o 58.48 (97 %).
  - Per the pre-registered rule, the modules are now ranked by GPU ms. A perfect CPU replay (1a) wins <= ~2 ms today. It
    binds once the GPU side is under ~23 ms, so it is still needed for a ~25 ms check. 1a's SLOTS kernel also removes
    the GPU-side loader poll and ~480 dispatches.
> (a quote from the project owner, kept private)
  - **Module 2 (experts, MoonShots) bit-exact track = KILL:** 583 us per call vs the 480 line. The call is
    compute-bound, and the 8-row tiles run padded. Experts stay at ~23 ms of GPU per check, so every other module at
    PASS sums to ~46 ms of GPU, not ~40.
  - Left for module 2: a FAST small-segment path (SIMD FMAs for <= 8 pairs). Per Kamen's standing FAST-track rule
    (11-02; adoption on EvalSuite same-scores, orchestrator 11-04), it needs EvalSuite's gate, not a separate Kamen call.
  - The GPU-side lead: 54 ms over 3,172 dispatches is ~17 us of GPU per dispatch, and a grid barrier costs ~1.5 us.
    So fold dependent chains (GDN core, attention core, MoE tail) into one launch. The HC read in one launch already
    does it: 72/72 bitwise, -5.6..-6.4 ms per check (rough), 288 -> 96 launches.
  - Saturday, priced by the orchestrator [P] with ladder v4's other inputs; NewIdeas re-prices: at a ~46 ms check,
    HE+ lossy ~~at quality~~ (CORRECTED 06:4x: projected, quality NOT gated; the gated lossy stack is chain-16)
    is ~277-282 (f 0.269, overhead 4) / ~290-295 (overhead 1.5) / ~309-315 (f 0.10). Ladder
    v4's ~385-428 needs ~14 ms more off the check, from the chain folds and / or the FAST expert path.
- Tue 17 00:38-00:48 (BigMoE): **G1 key test (g1b, base + 8 int8-free envs) = 7 / 9 PASS.**
  - Scored keys exact 27 / 27, swap +0.00, 0 swapouts.
  - g1c / g1cr fail only the COMQOM probe: g1's mbpp/115 answer has no '#' line, so the comments-only path is untested,
    not failed. BigMoE re-runs it on keys with '#' lines; the 5d fallback went to EvalSuite.
  - It ran at gate 33 (orchestrator 00:3x). Chrome had been reopened at 23:06 after the 23:00 quit and was re-quit at
    00:33; NO-SWAP went 22.3 -> 34.2. One run compressed ~14 GB of idle pages (NO-SWAP 47-49 after it).
  - M4 moves to TUE NIGHT first (NewIdeas 00:5x): chained with BigMoE's capture after the 23:30 quit; M4 latest start
    Wed 01:30, capture by 04:00.
- Mon 16 22:2x (MTP Head, REPORT 288e8c7; LAB sandbox S16-S19): **smaller-drafter head plan = 8-bit Linears (free post
  hoc) + a ~32K vocab screen (kept >= x0.99); 4-bit fails even with QAT; training can't recover the screen (its loss
  is coverage).**
  - Kept at tree-16/32: 8-bit x0.995-1.001; screen 8K x0.968-0.983, 16K x0.982-0.989, 32K x0.995-0.999.
  - 125B estimate, with 8-bit Linears: 8K ~42, 16K ~52, 32K ~73 MB per level vs 177.6 today.
  - To confirm on the 125B: MoonShots' M3 Q8 + H16 / H32.
  - **CORRECTED 22:3x (MTP Head):** 32K over 8K costs +31 MB per level = +0.11 ms at 280 GB/s (not ~0.07 ms), so
    ~+2.3 ms per tree-32 draft. That is near-neutral in tok/s against its ~+2-3 % kept, with 8K slightly ahead.
    Wed 06:30: MoonShots runs the 125B's H32 (+ H16) beside Q8, and that curve picks N.
- Mon 16 18:2x (MTP Head, REPORT 2fed5f1; LAB sandbox): **S13 QAT screen FAIL (no anneal), S14 shallow tap COLLAPSES ->
  tapmlp closes; S16 QAT-with-anneal running.**
  - Post-hoc 4-bit costs x0.928 / x0.906 at tree-32 (MBPP / HE), which reproduces M3's 125B loss.
  - QAT at 2e-6 for 300 steps recovers ~60 % of it. The constant-lr controls drift, so S16 repeats the test with
    a cosine.
  - tap 2: MBPP +1.19 / HE +0.20 pt (0.25x / 0.04x tap 12's gain).
- Mon 16 18:2x (NewIdeas): **CORRECTED: the INERT numbers in the entry below are the WHOLESALE arm, a recorded quality KILL
  (CD16-DROP 11-06). The quality-safe guarded arms (T16 / TC16) add x1.00-1.02 over today's lossy line (HE+ tree-32 16.87
  vs 16.91; MBPP x1.005-1.019). So the frontier AT QUALITY is flat: kept x1.43 must come from the drafter. Honest
  Saturday HE+ lossy at quality: ~340-420 tok/s, not 400-600.** The INERT swap in night1117 is withdrawn; night1117 runs
  as filed. NewIdeas research/week98/tree_frontier_t16.txt.
- Mon 16 18:1x (NewIdeas, PUSH800 frontier; CORRECTED above): **no lossy acceptance rule reaches x1.4 kept on HE+ at tree-32 (best x1.11;
  code-safe x1.07). Chains gain more: chain-32 INERT + R1(1) x1.29, INERT + TOP2 x1.40, both optimistic (CPU replay;
  the 'stop' bound is 14.2-14.8 kept).** MBPP chain-32 INERT x1.60. With Leap's split sums (32-40 ms per check), the
  Saturday HE+ lossy best is ~400-600 tok/s. 775 needs kept 28-38, at or above a 32-row check's ceiling.
  Tonight night1117 gates INERT (CD16) in place of d2, if BigMoE and EvalSuite are ready; its lossy_speed read-out
  measures the real chain kept. NewIdeas research/week98/tree_frontier.txt (bde3761).
- Mon 16 17:2x (NewIdeas, slot order): **EvalSuite's B2 -> V2 (the pooled RUNAWAY verdict) has waited for memory
  since 15:04 (NO-SWAP 38-40 vs its 43), so the GPU sat idle.** Tonight's order:
  1. Breakthrough's map screen now (~40 min, 18 GB; it yields to B2);
  2. BigMoE's g1b key test (~12 min; its exit should open B2's 43, as at 03:27 / 06:55 / 11:24);
  3. B2 -> V2 (must start by ~19:45 to end by 22:30).
  If B2 hasn't started by 19:45, V2 moves to Tuesday by day (4d) and G1 slips ~2.5 h. The remaining lever then is
  memory (idle sessions/apps), which is the orchestrator's call.
- Mon 16 14:1x (NewIdeas): **RE-CORRECTED with the full V1 code rows (264 keys): 1c-only runaways 10 vs base-only 3
  (p 0.092); a pooled FAIL after V2 is plausible (~20 vs 6, p ~0.01), so G1's 5d branches are likely needed.**
- Mon 16 14:1x (NewIdeas): **CORRECTED: 1c-tiers' early runaway signal is 2 of 45 vs 0 (p 0.5), not 4.** mbpp/13
  and HumanEval/147 cap in the base too under the gate's per-item caps. A RUNAWAY FAIL is no longer likely; G1's
  branches (amendment 5d) stay armed as insurance.
- Mon 16 13:5x (NewIdeas, NUMBER CHECK of Leap): **CORRECTED: expert bytes were 23-30 % low in every check budget**
  (records3: 2.785 MB per 4-bit expert, not 2.135). A 25 ms lossy check needs >= 96 % of every peak even for C6 + V4 +
  int8 (bound 24.0 ms); the exact 32-row code floor is 57.9 ms, not 47.3, so the "~55 ms realistic exact target" above
  is below the real floor. Measured times are unchanged.
  **CORRECTED Mon 16 13:5x (Leap 73ddc5a; NewIdeas checked records3 L01.bin = 2.785 MB per 4-bit expert): the budgets
  priced experts 23-30 % low. So the exact 32-row floor is ~58 ms [est, Leap to re-price], not 47.3 / 49.1, and "~55 ms
  realistic" is below the real floor. The engine's 90.5 ms is ~1.6x the floor, not 1.9x.** Lossy side: 25 ms is at the
  physical edge. At 100 % of every peak only C6 + V4 + int8 gets under it (24.0); at a realistic 85 % it needs experts
  <= 3.83 GB per check, 17 % below C6. Measured times don't move.
- Mon 16 13:4x (NewIdeas): **The lossy 32-row check measures 56.4 ms on the real engine (2.26x from 25 ms); the
  1c-tiers runaway fix is comment rows on QOM.**
  - Leap run2b (quiet, every expert resident): C5 + C6 + int8 = 56.41 ms. The expert kernel is at ~2x its byte floor
    at 16 and 32 rows whatever the split (~28 ms of experts alone): the check pillar's main wall.
  - Fit12: 35 of 1c's 38 flips are on "#" comment rows (all 16 non-tie). Comment rows on QOM cost +3 ms (+5 with RES).
  - NewIdeas: one installer (comment-QOM + RES) built now; 4-key kill test and Breakthrough's Tue 07:00 map read
    pre-registered; G1's fix if tonight's RUNAWAY guard fails.
- Mon 16 13:1x (MTP Head, REPORT 'S12 FULLSTATE RESULT', 4ac7efb): **S12: tapmlp at full budget = PROMISING; carryac
  NO EFFECT** (LAB, sandbox 2B, 1,889 steps from native vs f30, which is bitwise s30).
  - tapmlp (the target's true layer-12 residual -> MLP -> added to the carried state):
    - MBPP 23-30 +4.78 pt (p 7e-13), kept +2.7 %; HE+ 23-30 +4.55 pt (p 2e-9), kept +5.1 %;
    - HE walk-strict: 7-14 86.3 -> 88.4, 15-22 85.2 -> 88.7, 23-30 89.3 -> 92.4.
  - Cost: a partial trunk pass per drafted token (13 of 24 layers on the 2B), sequential on the 125B. Price it
    before any 125B use.
- Mon 16 10:3x (NewIdeas): **CORRECTED: early death's ~410-455 tok/s was drafting-HIDDEN, which a bandwidth-bound check
  can't do.**
  - With the K128 tree's ~39.6 draft levels at the full head's 177.6 MB per level (26-36 ms per block), the like-for-like
    line (HE+, byte floor, probe untested) is: K128 + early death 251-303 tok/s vs tree-32 209-224 vs tree-16 204-216.
  - Early death still leads by ~+20-35 %, but it does not reach 450. 450 exact needs early death AND a smaller
    same-accuracy drafter AND a near-floor check.
  - The Tuesday-night dev pass stays (~35 min; best exact route). Number check by the orchestrator 10:4x.
- Mon 16 09:23 (MTP Head, REPORT 'S10 PASS2 RESULT'): **S10 PASS2 = FAIL, so no t30c** (LAB, sandbox 2B: MBPP-50 /
  HE-30 vs s30).
  - p2f (fresh pass, 1.365x): MBPP 23-30 +2.61 pt (p 0.002), kept +3.7 % = PASS; HE 23-30 +0.98 pt, kept -1.2 % = FAIL.
  - p2s (0.36x fresh with its own anneal, the analog of the only t30c the 125B's 755 unseen files allow): MBPP +0.2,
    HE -1.3 pt (p 0.001). So the A recipe is predicted null-to-harmful.
  - A mid-cosine copy is 7-9 pt worse: never pin one.
  - p2b (second epoch on seen data, information) is running.
  - The day's 16 GB slot is free; the t30c runner and manifests stay (pre-registered a43f0e0).
- Mon 16 08:3x (NewIdeas): **C3 lossy gate coordinated** (the orchestrator's YES 08:3x).
  - Chain: Leap's ref2 change (2-D nolsb + GPU dup pick), then BigMoE's wiring + counter + a NEW lossy-45 base, then
    EvalSuite's gate (tier 0 -> tier 1 CODE-ONLY, both orders, <= 16-row checks, ~4.6 h at 45 GB). Each comes after the
    owner's WEEK98 items; none delays E14 or M4.
  - Slot: Tue by day after night1117 and after BigMoE's 127-node dry run, else Wed by day before the capture. The
    drafter line wins any conflict.
  - Leap also prices a DISPATCH DIET (<= 30 min desk) for the 900 line's CPU third.
  - MoonShots gets a ~5-min quiet timing window after S10 PASS2 (Kamen away).
  - Number checks: engine R32 = 1.480 measured (not 1.445); LOSSY32 confirmed.
- Mon 16 09:2x (NewIdeas 56587b0; BigMoE's real dry-run dump tree_t30b_heplus24_n127k8.npz, nest check PASS): **REAL
  127-node trees keep far more than S11's proxy on HE+: 23.70 kept per check at 128 rows (proxy 19.14).** HE+ ONLY;
  MBPP's dump is still running.
  - Kept at 16 / 24 / 32 / 48 / 64 / 96 / 128 rows: 11.26 / 14.39 / 16.26 / 18.86 / 20.77 / 22.77 / 23.70. K64 equals
    the true-state chain-32 (20.77); K128 beats the flat-97.8 % line of a 32-row check (23.15).
  - The extra is LONGER RUNS, not better per-band coverage: walk-strict bands 3-14 stay 93-96 % at 127 nodes (K98
    holds); bands 0 / 1-2 reach 98.9 / 98.3. Mean tree depth ~40, many runs past 30 (per NewIdeas, mostly HE+'s
    docstring copies; check that MBPP shows it before leaning on it).
  - K* under Leap's per-row costs is unchanged (24-32 at the floor; the engine still picks K16).
  - **What changes: EARLY DEATH (THINK #5, Fable B).** A K128 check whose dead branches leave mid-pass is ~54 ms by
    bytes (probe at layer 16) / ~48 (layer 8). ~~With the real 23.7: ~410-455 tok/s exact AT THE BYTE FLOOR, vs ~277
    for K32. The first route in the arithmetic to 450 exact.~~ **CORRECTED 10:4x (orchestrator's number check, NewIdeas
    agreed, b04f5b6; see the top entry):** the 410-455 kept drafting hidden (+4 ms). The full head reads 177.6 MB per
    level and a K128 tree has 39.6 levels, so drafting is 26.2-36.3 ms per block (a lower bound). Like for like (HE+,
    byte floor, drafting NOT hidden, probe untested): **K128 + early death 251-303 tok/s; tree-32 209-224; tree-16
    204-216.** Early death leads by ~+20-35 % but does NOT reach 450; that needs it AND a smaller same-accuracy drafter
    AND the check near its floor.
  - The probe is UNTESTED (PASS >= 90 % of off-spine nodes droppable at <= 2 % false drops at layer 16; KILL < 60 %).
    Its data: the 125B's layer 8 / 16 / 24 states on the dev bank (HE+ 24 + MBPP 100), a teacher-forced 45 GB pass of
    ~20-40 min (BigMoE prices it). Proposed for Tuesday night (confirm by Tue 18:00); the probe test is CPU, Wed by day.
  - **10:1x MBPP's real 127-node dump (a5be808): the proxy was RIGHT on MBPP** (kept 9.03 / 10.19 / 11.29 at 32 / 64 /
    128 rows; K128 11.29 vs proxy 11.24). So HE+'s extra lift is HE+-specific (long confident runs, mostly copied
    docstrings). MBPP still gains +25 % kept from 32 to 128 rows: at early death's byte price K128 projects to
    ~195-217 tok/s on MBPP vs tree-16's 171 at the floor (+14-27 %), and ~410-455 vs 277 on HE+ (+48-64 %)
    [both drafting-hidden: SUPERSEDED by the 10:4x correction above]. Labels:
    at the byte floor, probe untested, the HE+ magnitude is copy-heavy code. MBPP walk-strict at K127: 99.6 / 97.2 /
    92.4 / 87.5 / 86.8 / 88.7 / 86.6 (knowledge-limited); K* on MBPP stays tree-16 under every cost.
  - **11:1x Thursday's window cut to TOPK + T8 at 32 nodes (~0.8 h) by the orchestrator** (EvalSuite 696a937: a
    127-node anchor costs ~10x a 32-node one, so TOPK + T8 + T16 would be ~4.3-5.5 h, not the approved 2.2-2.5 h). T16
    is information only: on dev, k16 = k8 at tree-16/32/64 on HE+ (x1.000) and within 1.1 % at 127; MBPP tree-16
    band-0 97.84 vs 97.79. No CERTIFY98 line reads trees beyond 32 rows. NewIdeas (11:1x, the same order) adds one
    CONDITIONAL, information only: a fresh T8 at 127 nodes on HE+ (~1.2-1.4 h), only if Wednesday's early-death probe
    PASSES and a by-day window remains after the gated work without touching the proofs. EvalSuite's independent dev
    scoring matches NewIdeas' K-curve (HE+ x1.277 / x1.457 at 64 / 128 rows; MBPP x1.128 / x1.250).
    BigMoE 11:2x: a 127-node T8 CONTAINS the 32-node T8 (first 32 pops bitwise on all 124 windows). If the
    conditional runs, HE+ T8(127) runs BESIDE HE+ T8(32) (+~8 min), not instead. The gated tree-32 read stays on
    T8(32) as pre-registered, and T8(127) is information, valid only if its first 32 pops match T8(32) bitwise
    (else VOID). The default stays TOPK + T8(32) on both sets (~0.7-0.9 h).
  - **11:03 BigMoE's 127-node dry run DONE** (rc 0; all 4 dumps: heplus24 24 windows + mbppbank 100 windows at k 8 and
    k 16, 6,375 + 15,641 anchors; 1 h 58 min; MLX peak 13.1 GB; swap +0 on every part). The first 32 pops equal the
    32-node dumps BITWISE on all 124 windows; k 16 adds <= 0.008 kept. Marker N127_DRY_1117_DONE. (Its "all anchors"
    kept, HE+ 38.19 / MBPP 12.45 at 127, is a different basis from the walk numbers quoted here.)
  - **CONFIRMED for Tuesday night (NewIdeas 10:1x):** BigMoE's teacher-forced dev pass, ~26 min + load (~35 min slot at
    45 GB), layers 8 / 16 / 24 for 22,016 answer rows (1.35 GB fp16), no token change, self-check final-layer stream ==
    stored wd bitwise; taps built Tue by day. **The Tue 23:30 app quit is armed** (apps_2330_1117.sh; Kamen's Tue-Fri
    yes). The probe test (CPU, Wed by day) is pre-registered Tuesday with Fable B's lines.
- Mon 16 09:2x (NewIdeas e65c19e; Leap cf70494): **day order + Saturday's candidate + the dispatch-diet price.**
  - Today's big slot: BigMoE's 127-node dry run + MoonShots' M3 -> BigMoE's 1c-tiers key test (~10 min, 43 GB) ->
    EvalSuite's 1c-tiers gate by day if armed by ~11:00 (yields ~15:00, resumes Tue after T0A_1117_END). MTP Head's
    item 5 (sandbox, <= 4 GB) beside it if memory allows.
  - **Saturday's drafter: t30b + EAGLE-2 trees + the frozen 2B audit** (pinned in amendment 8). K* = tree-16 unless
    E2's R32/R16 <= ~1.43 (final Tue evening).
  - Drafter-line GPU left this week: the capture (Wed night); BigMoE's post-capture TOPK + 127-node TREE dumps (~12 GB,
    ~1.8 h by day); the 2B (+ 4B info) audit rows on the fresh cells (Thu by day); the proofs (Thu-Fri nights).
    **Tuesday night is free.**
  - **Dispatch diet (Leap, desk):** C32one issues 4,022 dispatches per check (~3.0 us build + 3.5 us encode each,
    + 1.7 ms fixed); 52 % are MLX glue. Fusing the glue (-1,800 dispatches): CPU -11.7 ms, GPU -2-4 ms, 1.5-2.5 weeks,
    exact. Python specialization: -1.5-2 us per dispatch, 3-5 days. Together: CPU ~12 ms per check. The <= 5 ms
    that 900 needs takes a replayed command path (Metal indirect command buffers, ~1-2 ms CPU; 3-5 weeks, high risk)
    or megakernels (~5 ms; 4-6 weeks). First step: a 1-day command-buffer microbench. Side: 1c-tiers adds ~34 small ops
    per MoE layer (~+2-3 ms GPU unless fused into router_fused).
- Mon 16 09:23 (MTP Head ee806ed; LAB, sandbox 2B, MBPP-50 / HE-30 vs s30): **S10 PASS2 = FAIL, so t30c does NOT
  run** (S10's pre-registered line: both sets must pass).
  - p2f (s30 + a fresh-data pass at 1.365x the volume): MBPP 23-30 +2.61 pt (CI +1.81..+3.38, p 0.002), kept/32
    +3.7 %: PASS. HE 23-30 +0.98 pt (CI +0.07..+1.89, p 0.36), kept/32 -1.2 %, walk-strict depth 0 87.7 -> 84.9:
    FAIL.
  - p2s (680 steps, own anneal = the analog of the only t30c the 125B's 755 unseen files allow): MBPP +0.21 (n.s.),
    HE -1.28 pt (p 0.001), kept -1.4 / -3.1 %. So an unseen-files-only t30c is predicted null-to-harmful.
  - A mid-cosine copy (step 680) is 7-9 pt worse than s30: never pin a mid-run copy.
  - Saturday's drafter stays t30b (+ trees + the 2B audit). t30c's pre-registration (a43f0e0), manifests and gated
    runner stay ready; nothing is seeded. The day's big GPU slot is free.
- Mon 16 08:4x (MoonShots dfe0e4a; NewIdeas' calls): **PILLAR 3 M0-M2 (drafting off the critical path) = KILL for stock
  MLX: the best byte cut drafts an HE+ tree-32 block in 9.3 ms (line 8; PASS <= 4).**
  - What binds: ~0.31 ms of fixed GPU cost per level (~144 dependent dispatches at ~2.1 us), not the host sync (0.549
    ms synced vs 0.311 sync-free async). mx.compile and the engine's fusions don't remove it.
  - Per block, async (full head 95 MB / compiled 2-bit + top-2-expert cut 39.8 MB): HE+ tree-32 13.4 / 9.3 ms; HE+
    tree-16 (K*) 8.1 / 5.6; MBPP tree-16 6.6 / 4.6.
  - NewIdeas: a hand-fused level kernel is NO GO this week (no CERTIFY98 line needs it). It is next week's first
    PILLAR 3 build: <= ~35 dependent steps per level vs ~144, taking tree-16 drafting to ~2-3 ms (~7 % tok/s at the
    engine). M3 (do the byte-cut heads keep the same tokens? within 1 % kept on both sets) is GO today beside t30c at
    MEM ~12 GB.
  - Engine note for after Saturday (exact, cheap): a sync-free async_eval draft level costs 0.645 vs 0.901 ms (-28 %).
  - **M3 RESULT 08:45-10:17 (MoonShots; TRUE-STATE, rows 16 and 32 under certify98's walk): every byte cut FAILS the 1 %
    kept line, on both sets, at both sizes.** Gate PASS (--cut none reproduces BigMoE's tree_t30b dumps bitwise, 21 / 21
    arrays). Tree-32 kept vs the full head: H8 (8K screen only) x0.963 HE+ / x0.985 MBPP; Q4 x0.922 / x0.899; C1 x0.875 /
    x0.889; C3 x0.767 / x0.791; C8 x0.186 / x0.327. Weight cuts drift with depth (Q4 top-3 set 55 -> 23 % on HE+); the
    screen doesn't (H8 top-1 >= 99.3 % at every depth). So drafting stays ~8 ms per tree-16 block until next week's
    fused level kernel.
  - **10:3x NewIdeas: PILLAR 3 CLOSED this week.** The full head reads 177.6 MB per level (~293 GB/s), so bytes alone
    take ~7.6 ms per HE+ tree-16 block (12.6 levels) and ~12.6 ms per tree-32 block (20.8 levels). The <= 4 ms target
    is unreachable with this head. Draft per block: tree-16 11.6 ms sync-free today -> 8.3 with a hand-fused level;
    tree-32 19.1 -> 13.8. At the measured engine, HE+ tree-16 goes 144 -> 150 tok/s (+4 %; fully hidden would be 169).
    Drafting can't be hidden on a bandwidth-bound check (~2.2 GB per block adds in full). So the hand-fused level is a
    modest exact engine gain for next week; Pillar 3's real route is a SMALLER drafter with the same accuracy (a
    distillation / training project). Guard: certify the head as stored (177.6 MB per level, trained parts bf16); no
    4-bit draft cut (-8-10 % kept at tree-32), and an 8K screen costs 1.5-3.7 %.
- Mon 16 08:4x (EvalSuite): **CERTIFY98 amendment 8 filed (eb928cf, pre-capture): LAB, C1, C2 (2B audit, P = 2) and
  the tree-K line (T32/32 on HE+ AND MBPP; T32/16 on HE+); walk-strict bands count targets only; 8 gated claims, quoted
  together in the RESULT.** A new scorer (certify98_lines.py) reproduces every dev number.
  - Power at the registered bounds: LAB 0.71; C1 tree-16 1.00, tree-32 0.92; C2 0.97 at the dev truth, but 0.81 if
    the true deep bands are 98.0 and 0.60 at 97.9; T32/32 1.00; **T32/16 0.00 at the measured R32 1.480** (it passes
    only if E2 measures R32/R16 <= ~1.43). Amendment 8a pins c32 and R32/R16 from E2 Tue evening.
  - C2 now scores the 2B against the head's own label: dev 98.41 / 98.11 (35 HE+ near-tie rows had differing labels;
    on fresh files any such row VOIDs C2).
  - LAB's cross-check vs BigMoE's REPLACE oracle holds on MBPP only (HE+ 20.77 vs 20.90: DIFFERS).
  - A second pinned drafter candidate (t30c) gets the stricter 97.5 % bound on the same sets.
  - Tier-0a re-run armed LAST in night1117: follower pid 45029 (0ca9f08), only after night1117's "done:" marker; no
    start after Tue 13:15; ends by start + 2 h or 14:30. Next: the 1c-tiers lossy gate's pre-registration today.
- Mon 16 08:2x (Fit12 LOSSY32, desk, pre-registered; number-checked by NewIdeas): **the lossy 32-row check costs ~50-55
  ms realistic on the measured engine, not 21-25 ms. So the 900-lossy line is ~300 tok/s HE+ realistic, ~500 with best
  kernels, with today's drafter.** Record: ~/Fit12/data/lossy32/lossy32_price.txt, PLAN_12GB_90_fit12.md "LOSSY32 RESULT".
  - Self-checks: the 16-row model gives 61.4 ms vs lossy-45's measured 61.2; exact at 32 rows calibrates to C32one (89.28
    GPU / 90.46 wall).
  - The "5.86 GB = 20.9 / 24.9 ms" plan put one bandwidth efficiency on the whole check. Priced per part (C5, 6.12 GB
    on code) it is 49.8-53.3 ms:
    - FAST dense side 23.8-26.6 (3x its floor; ~5 ms of cores);
    - experts 20.7-21.4 (1.43x);
    - 45 GB misses 5.3;
    - the CPU build + encode + commit 27.9 ms (C32one) as a bound.
  - Its floor on code is 21.9 ms: Breakthrough's 99.9 / 36.4 were all-domain means (code 114.2 / 39.2).
  - At K* lossy (kept 16.91, drafting hidden): C5 295-314 HE+ / 166-177 MBPP. C3 (lossy-45 + 1c tiers CODE-ONLY) alone
    285-297, ~92 % of C5's gain. lossy-45 as shipped, run at 32 rows, is 90-92 ms (about exact's 90.5).
  - Best case, all at once (misses 0, dense at 1.2x floor, experts 1.1x, CPU < 25 ms): ~29.8 ms GPU -> ~500 tok/s. 900
    at kept 16.91 needs a ~15 ms check: nothing priced gets there. It needs fewer bytes (~4 GB per check), near-floor
    kernels, the CPU side under ~15 ms (a dispatch diet: C32one issues 4,022 dispatches), zero misses, AND kept x1.32.
  - **Exact side: REF2-T reads the LSB plane for only 31 of 114 union experts on code, so the exact 32-row byte floor on
    code is 47.3 ms, not 52-54** (tree-32 at the floor ~306 tok/s). The measured engine stays 90.5 ms.
  - Cheapest gateable lossy check this week: C3. It needs Leap's ref2 change (2-D nolsb + GPU dup pick), BigMoE's
    wiring + counter, and a NEW lossy-45 base on that snapshot: ~2.3 + 2.3 h of 45 GB gate runs (tier 0 -> tier 1
    CODE-ONLY, both orders; it can gate at <= 16-row checks). The orchestrator RESOURCED it 08:3x (build by day
    today / Tue, gate in a by-day 45 GB slot; the drafter line wins any clash). Spec: ~/Fit12/LOSSY_1C_TIERS_SPEC.md
    (the gate is named "1c-tiers"; at 16-row code checks 1c drops 61.9 % of the expert union, 6.54 -> 2.51 GB).
    **BUILT Mon 09:0x:** Leap's ref2 change b43efc4 (2-D nolsb, bitwise checks PASS) + BigMoE's snapshot t1c1
    (68274ff). Left before EvalSuite arms: a ~10-min 45 GB real-loop key test (2 envs x 3 keys). Today only if t30c
    doesn't start; otherwise it's a Tuesday follower after night1117 / tier-0a, before the gate.
    **ARMED Mon 09:3x (t30c is off): EvalSuite's 1c-tiers gate, today's phase pid 86890, Tuesday's resume pid 86926**
    (PREREG_lossygate amendments 4-4c, 7860280). It starts after BigMoE's dry-run marker and the t1c1 key test's PASS
    (peaks <= 42.5 GB), under the away rule, at a 43 GB gate. Today: base mini-panel -> variant mini-panel (KILL ends
    it) -> base forward -> variant forward (FAIL ends it), as far as the 14:30 cutoff allows. Tuesday resumes after
    T0A_1117_END. The reverse pair (B2 -> V2) moved to Tuesday by day: follower pid 95134 (amendment 4d, 4cabcf6)
    runs it only if V1 passes, after Tuesday's phase, cutoff 14:30. **11:24 BigMoE's t1c1 key test PASS** (started
    at gate 40, read 40.5, against peaks 42.3 base / 42.4 variant; swap +0 on both; all 3 keys equal lx45cc1's
    tokens). Its exit lifted NO-SWAP 40.5 -> 49.4, and **the 1c-tiers mon phase started B0 at 11:25:26 at 48 GB** (the
    43 GB gate; amendment 4e's 42 GB fallback, 9429e48, unused). Next: B0 -> V0 -> B1 -> V1 by 14:30.
    **11:31 B0 (the lossy-45 base on t1c1, tier 0 vs the exact refs): KILL, arm only 0 / reference only 4 on n 40
    (exactly the kill line).** **11:38 V0 (1c-tiers vs B0): GO, arm only 1 / reference only 0.** B1 base forward runs
    from 11:38. B0's KILL is lossy-45's own loss vs exact on the mini-panel, not 1c's (asked EvalSuite whether it is
    lossy-45's known tier-0 read or a snapshot effect).
    **13:05 B1 (base forward vs exact) PASS: pooled code -1.14 pt (p 0.58), exactly lx45cc1 forward's line; the
    unset-path check PASSES on all 364 keys** (tokens identical to lx45cc1 forward, QOM sets included). The chain then
    PAUSED at 13:06 (amendment 4g, e1a798d) to give the GPU to Sandbox98's SIB. **SIB's smoke check FAILED at 13:06:57
    (rc 5: "mbpp: true_r 89.18 vs true 90.38", a harness-control mismatch), so its main read-out didn't run.**
    Sandbox98 is diagnosing. EvalSuite resumes V1 at 13:30 unless SIB is running, and pauses again after V1 if SIB is
    ready (sib_READY). It runs on runner 65:
    runner 64 would have voided every 1c run, because 1c replaces QOM in the checks (BigMoE caught it, amendment 4b).
  - **This MONTH's lossy-check ladder** (Fit12 LOSSY32 ADDENDUM vs TRIED; NewIdeas 8245427): C5 60.9-63.5 ms today
    (two 16-row calls) -> 49.8-53.3 with a one-pass > 16-row FAST form (not built; the biggest item) -> ~45-48 with
    Leap's 1b list (~3 days) -> ~41-45 with a resident-only MSB-3 rule (quality unpriced). **~41-45 ms = ~325-375
    tok/s HE+ at K* lossy.** No recorded lever touches the expert kernels at 32 rows or the 27.9 ms CPU build; Leap
    prices a dispatch diet.
- Mon 16 08:1x (NewIdeas): **queue + amendment + THINK #7.**
  - **C2b DROPPED** after L1CF's KILL: the gate's deep bands already pass with the plain 2B at P=2.
  - **EvalSuite got the amendment set** (AMENDMENT_INPUT_1116.md, 64d8a29):
    - C2-4B (4B, P=2) as information: dev 98.3 / 98.3, kept +14.9 %.
    - Engine prior R32 1.445: the T32/16 bar is 1.548 vs dev 1.531, so K* = tree-16 until E2. **CORRECTED 08:2x
      (NewIdeas' number check of the orchestrator's prior): the measured engine R32 is 1.480 (a082f8c), so the bar is
      1.585 and tree-16 leads by ~4 %. 1.445 was an untested composition.** E2 pins the bar on Tuesday either way.
    - Tier-0a goes LAST in night1117 (agreed).
  - **Queue:** S10 PASS2 now -> t30c by day at 16 GB if p2f passes. MoonShots' M0-M2 (<= 4 GB) run beside it; its M3
    uses BigMoE's dd_tree.py (--run).
  - **THINK #2 sandbox analog -> Sandbox98:** the 125B has no rejected-row states; Wednesday's capture is plain
    decoding.
  - **THINK #7 SLOT PRIOR = PASS on HE+:** in-comment rows are 13 % of HE+ answer rows but hold 71 % of the 125B's
    near-ties (m < 1), lift x5.4. MBPP x1.96 (KILL). A second key, next to #1's hard-start key, for Tuesday's
    widen / TIE-STUBS screen.
- **Mon 16 11:4x KAMEN (verbatim): "We MUST get 900 lossy. We must get 97.8 percent in 30 tokens and the check at
  25ms. This should be 100 percent clear. I mean yeah - no compromise. And all until Saturday".** WEEK98 is re-aimed on
  three numbers, all needed at once for 900 tok/s lossy on code:
  - (A) acceptance >= 97.8 % per guess to 30 deep (flat 97.8 = 23.15 kept per 32-row check; lossy rules allowed,
    EvalSuite-gated). Today: tree-32 16.26 exact / 16.91 lossy K* (HE+); true state 20.77.
  - (B) the lossy 32-row check <= 25 ms. Today: ~61 ms (two 16-row calls); realistic 50-55 with this week's 1c-tiers.
  - (C) drafting ~<= 3-5 ms per block (900 at kept ~23-28 leaves a 26-31 ms cycle). Today: the t30b head reads 177.6 MB
    per level: 8.3-11.5 ms for tree-16, ~20 ms for a 31-deep chain.
  - Owners (orchestrator 11:5x): Leap the check, MTP Head a smaller same-accuracy drafter, NewIdeas the budget,
    ranking and every other assignment, by 14:00.
  - **12:0x Leap CHECK25 plan (79a850f): a 32-row lossy check at C5's bytes is bound at 23.9 ms on this Mac at 100 %
    of every peak**, so 25 ms needs 96 % of every peak. The check is layer-sequential, and at 32 rows the dense side is
    COMPUTE-bound: 3.73 B dense params x 32 rows = 119 G MAC = 8.5 ms at the measured 14.0 T MAC/s bf16 peak, above its
    7.6 ms byte floor. Experts 14.2 ms, cores ~1.2. Today's best single kernels reach 74-84 % of peak, whole exact
    checks 53-60 %. 25 ms needs fewer bytes AND int8 dense math AND ~80-85 % whole-check efficiency: C6 (lambda 1.8e-2,
    5.55 GB) + int8 needs 84 %; C6 + V4 (3-bit GDN / attention) + int8 needs 79 %. The efficiency needs megakernels or
    a replayed command path (3-6 weeks). **Leap commits to a MEASURED full check by Friday at ~33-38 ms GPU [est]:**
    - qmm32n, a true 32-row native 4-bit GEMM;
    - f32one, every FAST op at 17-32 rows in one pass;
    - the 1c glue fused into one kernel;
    - chunked GDN at T 32;
    - the expert one-call at M 32;
    - int8 dense as an option;
    - engine measurements Tue / Wed / Thu nights.
    - 14:1x Leap, two check leads:
      - The lossy expert one-call reads ~27 % more expert bytes than it needs: one group per (expert, tier), so on
        32-row blocks 60.4 groups for 46.5 distinct experts, and 13.9 experts per layer have their 2.06 MB MSB plane
        read twice (134.9 vs 106.1 MB). If those second reads hit DRAM, a merged-group kernel (MSB read once, both
        tiers computed, bitwise) saves ~21 % of expert time, ~-4 ms per check. Tue slot #1 measures it; the kernel
        is Wed. **CORRECTED 14:2x (Leap, rough A/B beside V1): the second reads mostly hit the SLC (~1/4 of a DRAM
        byte each), so the merged kernel saves only ~5-8 % of expert time (~-1..-1.5 ms); on hold. The real expert gap
        is the lossy one-call's base streaming rate: ~140 GB/s vs ~290 for the exact path. Tue slot #1 measures it.**
      - On real sizes, CHECK55's profile shows the EXACT expert kernels at ~290 GB/s (near the measured peak), so the
        scattered access pattern isn't the wall.
      - Built today, all bitwise: gdn_fast (the GDN core ~1.6x, ~-1.5 ms), a t1cf fix (5 fill dispatches per layer),
        the c25 profile pass + pass K (per-class GPU ms).
    - **14:0x BIG CORRECTION (Leap 73ddc5a; data/week98/check25_bound_real.txt): every check budget priced experts
      23-30 % LOW.** All the budgets (cheapcheck.md 11-14, the CHECK25 bound, Fit12's lossy32_price, Breakthrough's
      LOSSY_CHECK_PRICE) used 2.135 MB per 4-bit expert and 1.661 MB per MSB-3 plane. The engine reads records3
      pieces of 2.784 / 2.064 MB; records3/index.json, the c25 engine's pools (21.21 GB) and config.json agree.
      - C5 experts 3.99 -> 5.05 GB per 32-row check; C6 3.63 -> 4.60 GB.
      - The 25 ms bound at 100 % of EVERY peak: C5 27.7 bf16 / 26.8 int8 ms; C6 + int8 25.2; C6 + V4 + int8 24.0
        (needs 96 %). **So 25 ms is NOT reachable at C5 or C6 bytes even with perfect kernels.**
      - At a realistic 85 %, experts must be <= 3.83 GB per check (with V4 + int8 dense): ~17 % below C6, or a more
        compact expert record (the s3 / b3 / b scale pieces are 11-15 % of each part).
      - Restated: the expert one-call runs at 1.52-1.64x its floor (not 2x); today's best engine check (56.4 ms) is
        2.69x its real floor of 21.0 ms.
      - Decision owners: NewIdeas (bytes / lambda / format research); any loosening of the quality bar is Kamen's.
    - **13:4x c25 ENGINE A/B (Leap 619fc22; REAL engine, lab checks: 3 code windows, true-state tokens, C 160,
      every expert resident, misses 0; quiet, 36 paired rounds per arm).** Median ms per check:
      - F16 today's 16-row check 38.88;
      - F32s, 32 rows on today's split route, 68.18 (1.74x);
      - + f32one one pass 58.77 (-9.46);
      - + dn32 58.06;
      - + t1cf fused tiers 58.42 (slower, open);
      - + int8 dense 59.62 (slower in the engine, open);
      - + C6 58.32;
      - **C5 bytes 56.65 (floor 19.0 at 5.32 GB = 2.98x, 34 % of peak); C5 + C6 + int8 56.41 (floor 18.2 = 3.10x,
        32 %).**
      - Per class: dense 12.97 (dn32; floor 8.5 bf16 / 4.9 int8 compute); HC read 6.3-8.1; the experts' one-call is
        2x its byte floor at 32 AND 16 rows (structural, ~25-28 ms per check); GDN core 3.9. ~6,300 graph ops per
        32-row check (proxy).
      - Bits: in the engine F32o vs F32s is NOT bitwise (argmax 94 / 96). 1c gives each expert one tier per check, so
        tiny differences flip tiers. The slice is bitwise; an exact-mode A/B will settle it.
      - **Leap's plain answer: 56.4 vs 25 is 2.26x away. By Friday ~40-46 ms looks reachable** (expert kernel to
        ~1.3x its floor -7, GDN one-wave -2.5, the two engine regressions -1.6, dense int8 -1..2). **25 ms does NOT fit
        by Friday on this design: it needs ~85 % of peak in every class.** Next: a libmtlprof pass (Tue slot #1), an
        expert one-call redesign, the GDN core, the regressions.
    - **CORRECTED 12:2x (Leap, after a measured quiet slot, 1d4890d): the realistic MEASURED C5 check by Friday is
      ~40-46 ms GPU [est], not 33-38; C6 + int8 ~36-42 [est].** Fit12's ladder agrees (40.5-45.3).
      - [K] The FAST dense matmuls of a 32-row check, ms per check: 19.30 today (two 16-row calls) -> 15.27 qmm32n
        (bitwise) -> 13.20 dn32 -> 12.76 dn32q int8 (11.21 with the quantize fused). Floor 6.90. GDN qkv|z per call:
        213 -> 134 us (floors 84 bytes / 96 matrix). Small shapes run 3-34x their floors (fixed cost per dispatch), so
        they need folding, not faster kernels.
      - The dense side stays ~25-28 ms at 32 rows: one-pass matmuls ~13 + HC reads 4.7 / 7.5 + GDN recurrence ~4.3 +
        router ~1.7 + attention ~1 + glue ~3. Experts ~20.4 + 1c glue 1.8-3.4.
      - Left to cut: chunked GDN -2.8, fused 1c glue -1.8..-3.4, HC4 -1.2, row-dense tiles -1.5..-4 (unproven),
        dead-row drop -2.7..-5.6.
  - **THINK_1116 results (11:59-12:05; files in ~/FableGod/{lunatic,zero,far,fableb,fablec}/IDEAS_1116_*.md):**
    - LUNATIC: 900 allows <= 303-333 MB of DRAM per kept token; the lossy check takes 264 MB per kept token at kept
      23.15. So the head must read <= 47 MB per level (t30b reads 177.6): with t30b, 900 is out at any shape (flat 97.8 %
      gives 660 at the floor). **The three targets don't compose: 97.8 % + 25 ms + 3-5 ms draft = 772-827 tok/s; 900
      needs flat 98.4-98.9 % at 25 ms.** Best: deep runs off the head (head levels only on bands 0-6; copy / lookup for
      the rest, since 58 % of HE+ positions sit in >= 8-token seen spans, right 99.2 %): 902 at the floor on HE+, weaker
      on MBPP.
    - FABLE C: misses are TIES or DRIFT. Even perfect tie forgiveness leaves bands 1-10 under 97.8, so no acceptance
      rule on code rows certifies 97.8 per band. Best: COMMENT-FREE as an exact, drafter-independent policy (a
      no-comments instruction; 71 % of HE+ ties and 52 % of first misses never generated; ~97-98 at bands 1-6 with
      tree-32 + CC(1) [G]). It needs Kamen's OK as a product choice and one plain paired quality run.
    - FABLE B: realistic check by Friday ~35-40 ms; 25 needs ~400 launches per check (2-3 weeks). The 4-day diet is
      -1,500 launches (CPU 27.9 -> ~17).
    - ZERO: SPAN TREE / DRAFT DEATH / WIDE-SHALLOW, all CPU tests on the 127-node dumps.
  - FAR (THINK_1116): of the head's 177.6 MB per level, only ~63 MB are trained Linears; ~115 MB are the vocab side
    (10.5 MB with the 8K screen). A QAT-distilled t30b (4-bit + 8K screen, trained) is ~26 MB per level; 15-min screen
    proposed. ZERO: depth is the draft wall (marginal kept per level 0.76 -> 0.31 from level 6 to 30). 900 needs
    ~0.1 ms levels AND kept tokens that cost no levels (span copying).
  - **12:2x NewIdeas PLAN900 (914f322; PLAN900.md here):** today ~215-227 tok/s lossy (desk check) / ~144 exact
    (measured engine, tree-16) on HE+ with drafting not hidden. **Kamen's three targets together give 681-827; 900
    needs one of them beaten** (e.g. kept >= 26.6 = 98.8 % flat at 25 + 3 + 1.5 ms, or a ~20 ms check at 23.15).
    - Ties cap the bands even with zero drift: the true-state chain reads 93.2 / 95.3 / 96.4 / 97.8 at bands 0 /
      1-2 / 3-6 / 7-10. Trees cover ties: band 0 98.5; bands 1-2 96.6 at K32, 98.3 at K127; bands 3-14 stay 93-96
      even at K127.
    - So band 0 (trees) and bands 15-30 (the 2B audit, 98.3) are reachable. Bands 1-10 at 97.8 need trees AND a
      drift fix: SIBLING-STATE drafting (the rejected rows' exact 125B states as the head's inputs; ceiling +51 %
      kept) is the only untested true-state lever, and its sandbox analog goes to Sandbox98.
    - Ranked bets, each with an owner:
      1. SPAN TREE / copied deep runs (NewIdeas, CPU today);
      2. the one-pass > 16-row check (Leap);
      3. the CPU diet (Leap / Fable B);
      4. check BYTES (C6 + int8 dense, <= 5.5 GB at quality; Breakthrough + EvalSuite);
      5. the QAT small head (MTP Head, a COST lever);
      6. the FUSED LEVEL KERNEL (MoonShots, multi-day, from now);
      7. DRAFT DEATH / WIDE-SHALLOW (CPU);
      8. SIBLING-STATE (Sandbox98);
      9. close-call d1 / d2 (tonight);
      10. early death (Tue-night data);
      11. misses (Fit12).
    - GPU plan to Saturday: Leap's engine slots Tue / Wed / Thu, the capture Wed night and the proofs Thu-Fri nights
      untouched; Friday is Leap's MEASURED lossy check; Saturday reads out measured + projected from certified parts.
  - **12:3x NewIdeas: PLAN900 bets #1 SPAN TREE and #9 DRAFT DEATH / WIDE-SHALLOW are both KILLs (f9c20b2, 2b0f0a2);
    KEPT, not the draft, is the binding wall for 900.**
    - SPAN TREE on the real 127-node dumps: HE+ x0.797 of tree-32 kept at <= 7 levels (copy spans fire on 30 % of
      checks), MBPP x0.757 (line >= 0.90x). The deep half of tree-32's kept on HE+ is the head's own confident
      drafting, not copies. It cuts levels 20.7 -> 6.9: +9 % tok/s at a 25 ms check, -4 % at today's realistic check.
    - DRAFT DEATH / WIDE-SHALLOW: no depth cap (6-24) or confidence cut beats plain tree-32 on HE+; MBPP tree-16 wins
      by +11 %. EAGLE-2's path-probability order is near-optimal on depth vs width.
    - With today's kept (16.26 HE+), even ALL the draft fixes (the QAT 26 MB head + a fused level = 0.30 ms per level)
      give HE+ ~449 tok/s at a 25 ms check. 900 needs kept ~26-28: the drift + ties wall. The drift levers left are
      SIBLING-STATE (Sandbox98), S12 (MTP Head) and the audit (bands 15-30).
    - Lossy check-bytes gates filed (EvalSuite amendment 5, 1e8b9a5): G1 = C6 + int8 dense (Tue by day, after Leap's
      engine slot #1), G2 = G1 + V4 (Wed).
      - ~~"V4 was never flip-tested" (a 12:3x "misread" correction)~~ **WITHDRAWN by NewIdeas, 12:4x: V4 WAS
        flip-tested and FAILED.** Sun 15 Nov, 12 code windows: bt_ga3 (GDN + attention at 3-bit RTN) code flips 4.43 /
        3.62 (d9b2db7); bt_ga3awq (AWQ-style) 3.86 / 2.99 (43ef47c). Both are over the 1.34 % line. The chat arm's
        multi 3.86 % was a coincidence.
      - Only 11-04's DENSE-3 variant (MSB-3 midpoint, ~27 % lower MSE than RTN-3) never ran; its prior is FAIL
        (~3.5-4 %). G2 runs only on a DENSE-3 PASS.
      - Breakthrough's price (ddd7a0b): on code G1 = C6 + int8 is 5.77 GB (bound 21.8 ms, so 25 ms needs 87 % of every
        peak); G2 is 5.43 GB (82 %). The only no-V4 route to <= 5.5 GB is lambda 2.0e-2 (5.46 GB; predicted code flips
        ~1.34, right at the line).
      - Leap's measured price: int8 dense saves only 0.44 ms per 32-row check (2.0 fused); C6 alone carries most of
        G1's -2.9 ms. G1 + G2 together buy ~-3..-6 ms, and 25 ms needs ~-28 ms.
      - Slot (NewIdeas): Breakthrough's one-load 4-arm map screen (~25 min, ~14-16 GB, 24 code windows): C6 bytes, G1
        with int8 fake-quant, DENSE-3 alone, lambda 2.0e-2. It runs right after EvalSuite's 1c-tiers chain ends and
        finishes before 22:30 (fallback Tue ~07:00-08:30).
  - **22:2x (orchestrator; Breakthrough's 21:1x report reached me at 22:2x):**
    - **Breakthrough MODULE 1a milestone 1 MET (rough), two days early** (docs/MODULE1A.md; fac1f0f, c33be8b, 3a2fa85).
      - FAST layer 0 (GDN + MoE, 32 rows, real weights, the check's route) is recorded once and replayed as ONE Metal
        indirect command buffer (123 commands). It is bit for bit equal to MLX on new inputs: 16/16 windows, layer
        output + both GDN states.
      - CPU per replay: 45 us median (min 31) with an MTLResidencySet, vs 327 us for a plain re-encode. 36 us of it is
        per-check setup (command buffer + encoder); the layer's own part is ~10 us.
      - So a 48-layer check in one ICB projects to ~0.05-0.1 ms of CPU [P], vs ~30-35 ms today.
      - Rough: a share lock beside other load. The official number is Leap's Wed quiet bench.
      - Milestone 2 still needs: Fable B's SLOTS kernel + BigMoE's no-slot-reuse rule (the slice's MoE is the
        all-resident gather_qmm); the SDPA's KV length from a buffer; the tree mask, positions and SVH ids as input
        buffers; MLX's cache limit at 0 while a record lives. The record is one fixed 32-row shape.
      - What it buys depends on tonight's try B GPU-busy number (~01:15). GPU busy >= 90 % of the check: the replay
        saves only the gaps (~2-6 ms). GPU idle >= 30 %: ~15-20 ms.
    - MTP Head's head plan (22:2x, corrected 22:3x; entry under THE TWO GOALS). Orchestrator's check: draft per tree-32
      block ~8.8 ms with the 8K head, ~11.2 ms with 32K (f 0.269 ms + bytes at the 273 GB/s that today's 0.92 ms per
      level implies), vs 19.1 today. At today's 56.4 ms check, both give ~235-240 tok/s vs ~213.
  - **20:4x:**
    - MTP Head S17 p8 (post-hoc 8-bit Linears; TrainLab data/sb98/s17_p8.json) PASSES its line: vs s30, tree-16 x0.9968
      MBPP / x0.9948 HE, tree-32 x1.0005 / x1.0000 (CIs include 1). So 8-bit Linears are free. The head's remaining byte
      question is the 8K vocab screen (post-hoc x0.963 HE+ / x0.985 MBPP; S18 trains under it). 8-bit + a recovered
      screen is ~42 MB per level vs 177.6. p6 and qn6 are still running.
> (a quote from the project owner, kept private)
      it. Breakthrough is asked to PAUSE the band screen (24 GB). The g1b key test and Leap's gpub wait for the night
      window. The 23:00 app quit (Chrome included) is being confirmed with him.
  - **20:3x NewIdeas evening batch (no new BIG result):**
    - Drift screens: Fable B + C carried-state correction KILL (all 8 arms; the frozen head reads injected corrections
      badly, as with SIB). FAR's WALK-MIX BETWEEN, leaning KILL (tree-32 +1.3 % on MBPP only, HE flat: the ALLROWS
      pattern). ZERO's tail audit is a desk KILL (its own costs outweigh +3-4.6 %). Lunatic's line goes to Wednesday's
      early-death probe. Next: Sandbox98's tree-aware mask variant, Tue by day.
    - Draft bytes: the vocab side is 115 of 177.6 MB per level, so bits matter less than the 8K screen. S17
      (post-hoc 8 / 6-bit) runs now; screen-aware training is next. The Wed 06:30 125B QAT slot is held until S17.
    - Gates filed: G1 int8-free (5e). G2 (5f) = C6 + V3 + lambda 2.0e-2 with the full SVH head, int8-free (~6.44 GB per
      check; twin 1.22 / 0.72).
    - GPU: B2 -> V2 on Tue (4j), after Leap's slot #1 (cutoff 17:00), then 5d's pick -> G1 base -> G1 (verdict ~Tue
      18:00). Breakthrough's band-lambda screen runs tonight (from 20:27).
    - **OUTLOOK: no x1.43 drafter path is visible this week.** The best remaining drafter bet is the 125B's mid-layer
      states (Tue-night capture, then Wed: early death's tree lines + mid-tap screens). Honest Saturday stays ~340-420
      tok/s HE+ lossy at quality.
  - **20:2x (orchestrator read of the records):**
    - MTP Head S16 (QAT 4-bit WITH a cosine anneal; TrainLab data/sb98/s16_RESULT.txt) = FAIL vs its line (>= 0.99x).
      q16 vs its anneal control c16: tree-32 x0.9795 HE / x0.9705 MBPP. qn vs s30: x0.985 / x0.970. So a 4-bit head still
      costs ~2-3 % kept with the recipe fixed. S17 (pre-registered 19:5x) tests post-hoc 8 / 6-bit and 6-bit QAT: if
      8-bit Linears are free, 8-bit + a recovered screen (~42 MB per level) is the Saturday head (Lunatic's 900 bound
      was <= 47 MB). Due ~20:50.
    - Breakthrough's G1 / G2 map screen DONE 19:33 (d40a3b2). G1 as gated PASS (0.95 / 0.53). DENSE-3 FAIL (2.09), so no
      V4 (registered rule). G2 without V4 at V1-32k 1.36 / 0.93 is just OVER the line (2 / 4 rows). V3 is the costly
      lever (+0.28 pt for 0.34 GB). The band screen moves to tonight.
    - EvalSuite's B2 -> V2 moves to Tuesday (PREREG 4j). The g1b key test and Leap's GPU-busy run wait for NO-SWAP 40
      (35-36 now, with the sandbox and MTP Head jobs running); latest start 22:50.
  - **19:0x Sandbox98 drift screen 1 (frozen s30, HE decides; pre-registered): ASSIM (Fable C) and ERRMEM (Fable B) =
    KILL.** G(3-8) is negative with CIs below zero (assim -3.57, errmem -3.61), and the correction RAISES the direction
    error instead of lowering it. FINAL 19:1x: KILL on all 8 arms, MBPP too (Sandbox98 dd61b92). Screen 2 (FAR's WALK-MIX, 3 arms x 400
    steps, 5d1e7eb) is training; verdict ~20:15. MTP Head's S16 (QAT with an anneal) runs until ~19:15.
  - **18:3x-18:4x THINK_1116c, DRIFT only (all five in; brief THINK_1116c.md, its facts CORRECTED by Fable C):**
    - Fable C (audit): "97.8 % flat" and the 89.65 / 94.32 / 95.29 band 23-30 figures are MARGINAL rows. The walk-strict
      drift gap on code is 4.0 / 3.5 / 2.2 pt at bands 3-6 / 7-10 / 11-14. Drift is sequence-specific and persists across
      blocks on HE+ (92 % of the per-window variance; a deep non-tie miss recurs x3.2). Invention: ASSIMILATED DRAFTING
      (a per-sequence running mean of true - carried over the accepted rows, added at depths >= 1). Exact, 0 bytes.
      Sandbox98 sibstate.py "assim" ~5 min: PASS G(3-8) >= 0.25 with CI > 0, KILL < 0.10.
    - Fable B: MEANPULL / ERRMEM (a Kalman-style pull to the window mean + a per-depth error memory from every tree row).
      Exact, < 10 us per block. Sandbox98 <= 20 min, PASS G >= 0.25. TRUE-KEY RESYNC = already done (BigMoE's code read
      18:4x: the head's prefix K/V are rebuilt from the verify states every block).
    - FAR: WALK-MIX TTT (post-miss rows at weight 0.3-0.5, judged on TREE kept). 0 cost, exact. Sandbox98 400 steps,
      PASS tree-32 kept +3 % with strict d1-d4 within -0.2 pt.
    - ZERO: depth accuracy is bought with model layers run wide. The cheapest certified source (one 2B pass, 1.06 GB)
      equals the whole 775 draft budget, so no serial drift lever fits 775. Best fit: audit the tail under the check
      (+3..+4.6 % kept, needs the QAT head).
    - Lunatic: 775 at 32 rows needs ~90 % of the own -> true gap closed, not half. The head's OWN information caps at
      ORACLE-31 18.48 HE+ / 11.21 MBPP, so a fix must bring NEW information (the check's states, the 125B's rows). Kept
      is lost to silent drift in confident chains, not at forks. Best: buy drift with 125B rows (early death). Wed probe
      PASS = a 64-row check <= 1.15x a 32-row one.
  - **18:1x MTP Head (LAB, sandbox; REPORT 2fed5f1): S13 QAT screen FAILS its line (>= 0.99x control at tree-16 / 32)
    from the recipe's constant lr with no anneal, not from QAT.** Post-hoc 4-bit costs x0.928 / x0.906 at tree-32 (it
    reproduces M3's 125B loss). QAT at 2e-6 recovers ~60 % (x0.97 vs s30), and the constant-lr bf16 controls themselves
    drift (x0.84-0.89). tapmlp CLOSES: S14's shallow tap 2 collapses (MBPP +1.19 pt = 0.249x tap 12's gain), so the
    drift fix would need mid-depth taps, which the 125B can't give; S15 is moot. S16 (QAT WITH a cosine anneal, from
    native, <= 4 GB, ~45 min) runs now and decides NewIdeas' Wed 06:30 125B QAT slot.
  - **18:2x NewIdeas CORRECTION of the frontier below: its INERT rows were the WHOLESALE arm (CD), a recorded quality
    KILL (CD16-DROP 11-06). The quality-safe guarded forms (T16 / TC16; tree_frontier_t16.txt) are FLAT: HE+ tree-32
    T16 16.87 (x0.997), T16 + R1(1) 17.00 (x1.005); MBPP x1.005-1.019. Today's lossy line is already on the frontier.
    HONEST SATURDAY, HE+ lossy at quality: ~340-420 tok/s (kept ~16.3-16.8 with the QAT head, Leap's 32-40 ms check,
    ~6.4 ms draft), IF the engine modules and the QAT draft land. The kept x1.43-1.62 that 775 needs (and exact 400+)
    can come only from the DRAFTER (Goal A), not from acceptance rules. The night1117 INERT swap is withdrawn.**
  - ~~**18:1x NewIdeas LOSSY FRONTIER~~ (SUPERSEDED by the 18:2x correction above; BIG; CPU replay of the real t30b dumps, self-check exact; tree_frontier.txt
    bde3761): no acceptance rule reaches x1.4 kept on HE+ at tree-32, so 775 is NOT reachable by Saturday on any
    measured or priced path.**
    - tree-32: best (INERT #13 + code top-2) 18.75 = x1.11; code-safe INERT + R1(1) 18.11 = x1.07; d2 / d3 / typical /
      SPAN #8 x1.02-1.05.
    - chain-32 (OPTIMISTIC: it continues on the true text after a relaxed accept; 14.2-14.8 if each relaxed accept
      ended its check): INERT 20.50 (x1.21); INERT + R1(1) 21.76 (x1.29); + code top-2 23.61 (x1.40, 12.7 code
      relaxations per 1k tokens, high risk). MBPP chain-32 INERT x1.60, + R1(1) x1.73.
    - Saturday projection, HE+ lossy, at Leap's check ~40 (every module PASS) to ~32 ms (stretch):
      - today's rule: 346-413 at the QAT head's 7.4 ms draft, 380-463 at 3 ms;
      - best code-safe: 413-487 / 489-596;
      - aggressive: 448-528 / 531-647.
    - 775 at a 32-40 ms check needs kept 28-38 per check, at or above a 32-row check's ceiling. **Realistic Saturday
      best: ~~~400-600 tok/s HE+ lossy~~ (WITHDRAWN 18:2x: ~340-420 at quality), IF the QAT draft, the engine modules AND chain lossy kept all land.**
    - night1117 swaps its d2 arm for an INERT arm (d2 x1.02-1.08 here and -6.06 pooled on the 2B), if EvalSuite and
      BigMoE can pre-register and head-test it by ~03:00.
  - **18:0x-18:1x THINK_1116b + the engine split:**
    - Fable B: STATS-FORWARD (-2.5..-4.5 ms [P], partials bitwise).
    - ZERO: CONFIDENCE-SPLIT LAMBDA (-1.2..-1.5 GB per check [P]).
    - FAR: ADAPTIVE BLOCK LENGTH (64-96 rows only on sure spines: x1.21-1.23 tok/s HE+ on the real dump, exact; MBPP
      x1.00; needs the fast head).
    - Fable C: #13 was booked at chain prices (tree-32 x1.10 HE+); INERT-shaped drafting (chains in comment / docstring
      spans) as the invention.
    - Leap's SPLIT_1116.md (33a75e5): replay (Breakthrough + Fable B), compile / specialize (Fit12), expert one-call
      (MoonShots), dense / HC / GDN (Leap). GPU check ~40 ms at every PASS, ~32 at stretch, Thursday's measured number.
  - **16:1x NewIdeas SATURDAY LADDER v3 (measured anchors; HE+ lossy, 32-row tree check, kept 15.7-16.3; PLAN900
    "SATURDAY LADDER v3"):**
    - today 205-222 tok/s;
    - + C6 + RES 217-240;
    - + the QAT ~26 MB head [P, S13 tonight] 255-285;
    - + Leap's lossy expert one-call fix [P, Tue / Wed] 277-345.
    - **So the realistic Saturday lossy line is ~255-345.** 900 needs a check <= 18 ms at kept 26.6, and every priced
      byte plan's bound is >= 24 ms at 100 % of peak, so 900 is beyond the physical bound at 32 rows on today's byte
      plans. (Supersedes the orchestrator's 16:08 "~260-270 today / ~300-350 end of week" to Kamen.)
    - Also in the batch:
      - Leap's lead: the lossy one-call streams at 140-184 GB/s vs the exact path's ~290, a structure gap worth up to
        -5..-10 ms per check (Tue 09:30 slot).
      - BET 7 closed: fixed cost FAIL, overlap BETWEEN (0.699), prefetch KILL; the exact -8.7 % per level is banked.
      - G1 branches are ready: Fit12's installer passed every GPU bit test; BigMoE key-tests 9 envs tonight; the Tue
        07:00 chain is armed.
      - EARLY DEATH is pre-registered for Wed (EARLYDEATH_PREREG.md).
  - **15:03 SIB MAIN READ-OUT (Sandbox98; LAB, true-state, teacher-forced on the 2B) = NOT PASS on the frozen head,
    KILL side. The sibling states are WORSE stand-ins than the head's own carried state at every band, on both sets.**
    - Walk-strict own / true / sib, and G = (sib - own) / (true - own) with its 95 % CI:
      - HE 60 (2,181 post-miss starts): 1-2 83.4 / 91.8 / 57.1 (G -3.13); 3-6 81.6 / 93.8 / 70.0 (G -0.96); 7-8
        85.4 / 94.8 / 80.1 (G -0.56); 1-8 82.8 / 93.4 / 62.6, G -1.91 [-2.36, -1.48].
      - MBPP 100 (5,085 starts): 1-8 75.7 / 89.0 / 56.8, G -1.41 [-1.57, -1.28].
    - PASS needed G >= 0.50 (HE) / 0.25 (MBPP). The harness is exact: recomputed true rows bitwise; VOID checks
      clear.
    - Why: the state after a wrong token belongs to a different sequence. Its direction error to the true state is
      above the carried state's at every depth (HE 0.95-1.10 vs 0.85-0.99), worst right after the miss (depths 1-2,
      -26 pt).
    - The pre-registered final KILL also needs the fine-tune variant (ctl vs sibmix, ~50 min, <= 5 GB), armed at a
      NO-SWAP >= 13 gate, so after EvalSuite's chain (~17:30).
    - **15:2x SIB FINAL = KILL** (oracle-called by NewIdeas 15:1x; Sandbox98 ac91254). The best fixed own / sib mix (w
      0.2) closes only 5-7 % of the gap (G +0.05 / +0.07), and w 0.4 hurts. Phase 2 was disarmed before any step. The
      learned-mix fine-tune was NOT run; reopen only with a new mechanism that beats the fixed-mix oracle by 3x+.
      Parked micro-lever: the w 0.2 mix (+0.55 pt HE / +0.98 MBPP walk-strict 1-8, LAB, no CI), free on the 125B,
      goes on the drafter's final-config list. Sandbox98 is free.
  - **14:22 1c-tiers V1 (variant forward vs the base lx45t1c1): tier 1 PASS, pooled code -2.27 pt (McNemar p 0.11;
    arm only 2 / reference only 8; screening line > -3 pt); RUNAWAY 10 / 3 (p 0.092) PASS; IFEval identical**
    (EvalSuite 278fb02). Pause 2 (a) now: Leap's profile + SIB; resume on SIB's rc 0 or 15:22; the chain ends ~17:30
    (B2 -> V2).
  - **14:3x MoonShots BET 7 (fused level kernel) = FAIL on timing** (bitwise vs stock; a600df5). The best fused form's
    fixed cost is f = 0.269 ms per level (stock 0.353; PASS <= 0.056, BETWEEN <= 0.10); exact -8.7 % per level
    (0.960 -> 0.876 ms). Tree-32 block at the full head 20.0 -> 18.2 ms (correcting M3's ~13.8 projection); at ~26 MB
    [P] 7.4 ms (tree-16 4.5). Cause: ~0.29 ms per level lost between dependent kernels (~16 us each), not the dispatch
    count. So at 20.8 levels a tree-32 block's fixed cost alone is 5.6 ms, above the 3-5 ms window at any byte budget.
    Options with NewIdeas: a persistent kernel with weight prefetch, F > 1 rows per level, or stop at fused2.
  - 14:3x SIB: the second smoke failed (14:24; the recompute path is 1.2 % off the capture, from shape-dependent
    kernels). A capture-shaped forward is bitwise equal to the stored states (083fe8d). Re-armed (pid 94655); the main
    read-out ends ~15:00.
  - **RE-CORRECTED 14:1x (NewIdeas, on V1's FULL 264 code keys): under 3a's exact rule, 1c-only runaways 10,
    base-only 3, both 11** (exact McNemar two-sided p 0.092). So V1's single-order guard should PASS at 0.05 (the
    official read comes ~14:15-14:20). If V2 repeats it, the pooled guard is ~20 vs 6, p ~0.01: a RUNAWAY FAIL is
    plausible, even likely. 1c-tiers adds ~2.7 pt of runaway rate (3.8 % 1c-only vs 1.1 % base-only); by set MBPP 5,
    HE 1, HE-X 4. The 5d branches are likely NEEDED; everything is armed (9 key-test envs tonight on g1b, Tue 07:00
    map read, the kill test with up to 6 discordant keys, a band screen).
  - **CORRECTED 14:1x (Fit12 caught it): under amendment 3a's rule only 2 of the 4 keys are 1c-only runaways**
    (mbpp/122, mbpp/147: eos in the base, length in 1c). The gate cuts at MBPP 1024 / HE 768 tokens, not the screen's
    2,048. mbpp/13 and HumanEval/147 cap in the base too. So the early signal is 2 of 45 vs 0 (McNemar p 0.5), and
    "FAIL looks likely" was WRONG. The comment-spiral evidence rests on 2 keys. What stands: 35 of 1c's 38 map flips
    are on "#" comment rows, and comment rows flip 1.8-2.2 % under any change vs 0.24-0.36 % on code. The 5d branches
    (comment-QOM / lambda 0.8e-2) stay armed as insurance.
  - ~~13:3x EARLY WARNING ... 4 of 45~~ (superseded by the correction above) **13:3x EARLY WARNING on today's
    1c-tiers gate (Fit12 reading EvalSuite's rows; information, the verdict is EvalSuite's): RUNAWAY signal.** 4 of 45 code keys hit the ~2,048-token cap under 1c-tiers vs 0 in the base
    (mbpp/122 247 -> 2052, mbpp/147 292 -> 2049, mbpp/13 1292 -> 2056, HumanEval/147 1081 -> 2049; exact p 0.125 so
    far). The RUNAWAY guard (amendment 3a) decides at the end. Check ms still falls at 16 rows (65.2 -> 58.1 HE).
    - Teacher-forced map screens can't see runaways; only decode gates do. G1 (lambda 1.8e-2) is more aggressive.
    - If 1c-tiers fails on runaway, G1 is re-planned before Tuesday's base, by a dated amendment: lambda <= 1.6e-2, or
      a repetition guard, or run as registered. Fit12 is characterizing the 4 keys and holds the RES build until the
      verdict (~19-20 h).
    - **13:4x DIAGNOSED (Fit12, desk): near-tie flips (exact margins 0.12-0.62 nats) that open into COMMENT
      DELIBERATION spirals ("# Let me reconsider / # But wait") never closing before the cap.** They are not loops
      (no periodic tail) and not a dropped-expert spike. 3 of 4 divergences sit in docstring / comment spans. The fix
      candidate if the guard fails: keep COMMENT SPANS EXACT (no 1c tiers on comment rows, by lossy-45's causal
      tagger); Fit12 is pricing it (~13 % of HE+ rows, ~20 % of MBPP). This is evidence AGAINST relaxed acceptance in
      comments, and for exact comments on the lossy line too.
    - **13:4x Fit12 (1c-tiers flips by tag, 12 HE windows): 35 of 38 flips sit on "#" comment rows (6.9 % of them,
      all 16 non-tie flips); code rows flip at QOM's rate; 0 on docstrings.** Comment rows flip 1.8-2.2 % under ANY
      change, even path noise (code 0.24-0.36 %).
      - Cheapest fix: comment rows on the QOM path, +3 ms per 32-row check (+5 with RES), quality ~QOM's [P]. Exact
        comments would cost +14 ms (misses 60 -> 102).
      - NewIdeas' decision: Fit12 builds ONE installer now (comment-QOM + RES switches; MBPP docstrings a per-set
        option). A pre-registered 4-key free-decoding KILL test follows its bit tests: the fix dies if >= 3 of 4
        still cap. If RUNAWAY fails tonight: G1 = 1c + comment-QOM (+ RES), by a dated amendment before Tuesday's
        gate.
    - NewIdeas' number: real lossy today with 1c misses is ~56-60 ms per 32-row check. At K* lossy kept 16.9 +
      tree-16 draft 4.8 + 1.5 ms that is **~260-270 tok/s HE+**. The expert kernel at ~2x its byte floor (135-145
      GB/s) is ~28 ms of experts alone, above the whole 25 ms budget: the check pillar's main wall now.
  - **13:2x Fit12 CORRECTION (desk, pre-registered; PLAN_12GB_90_fit12.md "RES-1c PRICE RESULT"): 1c-tiers alone
    gives back QOM's miss protection.** Counted on Breakthrough's 45 GB residency masks, 1c alone leaves ~60 SSD misses
    per 32-row check (~15 ms), not ~5. On code, 1c replaces QOM, whose resident keep produced lossy-45's low miss count.
    - Corrected realistic: 1c-tiers 64.3-67.5 ms per 32-row check (was 54.5-57.7); the plan C5 61.1-65.5.
    - Fix: RES (the 3-bit tier read only when resident) takes misses 60 -> 8 (~2 ms) at a predicted +0.05 pt code
      flips (Breakthrough's calibrated law). It needs a real map arm. With RES: C5 47.9-52.3; the month ladder still
      ends ~42-44 ms (~355-365 tok/s HE+).
    - Falsifiable prediction registered for today's 1c-tiers gate rows: variant / base misses >= 0.8 on humaneval +
      mbpp, <= 0.25 on humanevalx.
  - **13:3x MoonShots BET 7 (fused level kernel): BITWISE equal to the stock level** (17 dependent kernels per level
    vs 144 stock primitives; 21 chained levels x 2 start tokens at 400 / 1,010 / 2,000 keys, every output and cache
    row; 4646311; a negative control fails at level 0). Timing waits for a quiet slot. M7a (quiet, afcc3f2): a
    dependent dispatch costs <= 1.3 us on the GPU, so only ~half of the stock level's 0.35 ms over its bytes is
    dispatches. **CORRECTION (f43949a): 4-bit + the 8K screen is C1 = 95.1 MB per level MEASURED, not FAR's ~26 MB.**
    With f = 0.056 that drafts tree-32 in 7.9 ms, not <= 3 ms; Lunatic's <= 47 MB per level is not met by QAT + screen.
  - 13:3x MTP Head: the tapmlp cost question is in REPORT for Leap ('S14 SHALLOWTAP', b7b0db1). A check-computed tap
    adds nothing at depth 0 (the head already reads the anchor's full true state), and drafted rows' taps need a
    partial pass per drafted token, except tree siblings (SIB). S14 = tapmlp at tap 2 (~ the 125B's tap 5) runs after
    S13. The 125B QAT runs only on an S13 PASS (Wed 06:30-08:00).
  - **13:1x MTP Head S12 FULLSTATE (4ac7efb; LAB, sandbox 2B, 1,889 steps vs f30 == s30 bitwise): tapmlp =
    PROMISING, the first state lever to pass.** The target's true layer-12 residual -> a zero-init MLP -> added to the
    carried state. MBPP 23-30 +4.78 pt (CI +4.18..+5.39), kept +2.7 %; HE 23-30 +4.55 pt (CI +3.87..+5.31), kept
    +5.1 %. HE walk-strict 7-14 86.3 -> 88.4, 15-22 85.2 -> 88.7, 23-30 89.3 -> 92.4; depth 0 doesn't gain. carryac:
    NO EFFECT. Cost caveat: a partial trunk pass per drafted token (13 of 24 layers on the 2B; sequential per depth on
    the 125B) likely breaks the draft budget, so it's not this week's path. A desk cost question goes to Leap.
    - Small drafter: S13 QATSCREEN pre-registered (96baf2c): fake 4-bit on the head's 8 Linears, QAT 300 steps (lr
      2e-6 / 1e-5) vs post-hoc Q4, kept at chain-32 / tree-16 / tree-32. PASS = >= 0.99x the control at tree-16 AND
      tree-32 on both sets; sandbox, <= 4 GB, ~30 min, when memory allows. Then the 125B QAT screen (~12 GB) this
      evening. LEVEL PAIRING after that.
> (a quote from the project owner, kept private)
    and worth it for speed but also comments are a core part of the code and I'm not sure. I feel like it would be
    cheating if we did".** So no no-comments policy: comments stay part of the task. Relaxed acceptance INSIDE comments
    on the lossy line (comments kept, wording may differ at near-ties, EvalSuite-gated) was offered to him as a separate
    question.
> (a quote from the project owner, kept private)
    you".** Day / night limits and the Kamen-at-the-Mac yield are lifted until he says otherwise. The memory gates,
    the +2 GB swap guard, the disk floor, the fan cap and lid = pause stay. A thinking round on the three walls
    (THINK_1116.md) starts 12:0x.
- Mon 16 07:1x (Sandbox98, 8d0c9da; pre-reg d8611a9): **PLAN4D item 2, L1CF (lever 1c as a full-data distilled 2B
  auditor) = KILL.** [LAB: teacher-forced agreement with the 125B's top-1, not real decoding.]
  - One epoch over 3,032 files (805,005 answer rows). Control: HE+ 94.42 / MBPP 85.67 / val 87.29.
  - HE+ +0.45 / +0.80 / +0.82 at steps 1,000 / 2,000 / 3,032 (end 95.23, CI +0.33..+1.37): under the +1.0 line at every
    read. MBPP +0.83, val +1.99 (rising at every read). HE+ went flat over the last third while the loss kept falling.
  - The extra data cured 1c's overfit (1c at matched steps: HE+ -0.09 / MBPP -0.81 / val +0.42).
  - For scale: the 4B auditor agrees 95.48 and this 2B 95.23. If the audit law holds, the audited HE+ 23-30 band is
    ~95.2 against the 97.8 target.
  - Ran 07:00:41-07:09:28 by day, peak 6.15 GB, swap unchanged. NewIdeas picks next: a bigger adapter, the 4B auditor,
    or dropping C2b.
- Mon 16 03:31 (Leap, 867b7b5): **at 45 GB a gated 32-row EXACT check is ~1.56-1.70x a 16-row check [est], MISSING
  the <= 1.35-1.4x line a 32-row tree needs.** The run was real decoding on the shipped exact-45 (C 256), 4 HE keys,
  tokens == REF2. Records: Leap data/week98/cc3/README.md "exact-45 (a45)".
  - 16 rows: the cold check is 110.7 ms = dense 30.2 + experts 40.1 + misses 40.4. The live 16-row check is 139.1.
  - 32 rows: the dense pass alone is 70.9 ms (2.35x 16 rows, GPU-bound). Today's slot path is 2.5 s (graph build). With
    a gate carrying 320 ids, ~188.4 ms [est, from the parts].
  - CORRECTED in the same commit: the printed 1.35x divided a forward-pass estimate by the whole live check. Like for
    like it is 1.70x forward / forward, or 1.56x with the live extra on both sides.
  - It agrees with the engine profile (one pass 1.48x at the profile's setting; REF2-T split 1.67x) and costs v2: on
    today's engine a 32-row tree does not pay. The levers are the 32-row dense pass (70.9 ms; superlinear vs 16 rows)
    and the 32-row misses (~60.6 ms).
- Sun 15 22:2x (NewIdeas): **THINK_1115 #1 MARGIN-KEYED SHAPE = KILL as a chain/tree switch; the key is real on HE+**
  (research/week98/margin_keyed.*).
  - Keyed (chain-32 after an easy start, tree-32 after a hard one) vs always tree-32: -3.9 % HE+ / -4.9 % MBPP (line +3 %).
    Even after easy starts the tree's siblings help: band 0-2 miss 0.6 % tree vs 2.1-2.7 % chain.
  - The key itself: P(m < 1) lift x4.6 / 4.3 / 4.1 at k 1-3 on HE+ (base 3.5 %), but only x1.6 on MBPP. Band 0-2 miss
    under tree-32: 5.4 % after hard starts vs 0.6 % after easy ones.
  - So the key's use is to widen MORE after hard starts, not to drop width after easy ones. It goes to Tuesday's
    TIE-STUBS / widen-after-hard screen on the real 127-node dumps.
  - THINK #4 (lookup as a charged node) was already run: leaf KILL (x1.005), expanded upper bound x1.057.
- Sun 15 22:0x-22:1x (NewIdeas, bigger-trees lead): **K* first cut = tree-32 on HE+, tree-16 on MBPP; 48-128 never
  pay** (BIGGER_TREES.md, e324a41 / 331675b).
  - Leap's R(K) (05eea11) x S11 kept: HE+ tok/s floor / realistic / lossy-25 = tree-16 257 / 167 / 493, tree-32
    277 / 174 / 542, K64 229 / 123 / 456. K64 would need kept 21.65 to tie K32 (proxy 17.91).
  - Tree-32 vs chain-32 at the same rows: x1.18 HE+ / x1.30 MBPP. Band 0 coverage 98.5 / 98.6.
  - EvalSuite gets T8 at 32 rows, two bars:
    - tree-32 / chain-32 (c32 x 1.03 ~ 1.071; dev 1.181 / 1.304);
    - tree-32 / chain-16 (R32/R16 x c32 x 1.03, R from Leap's measured E2): bar 1.454 floor / 1.489 contract /
      1.74 today, against HE+ 1.531.
  - **450 exact / 900 lossy framing** (TARGETS_450_900.md): 450 exact at 32 rows needs kept >= 23.9-25.2, i.e. flat
    98.0-98.4 %, above the true-state head (20.77 = 97.0 %), with the check AT its byte floor and drafting hidden. Flat
    97.8 % gives 413-436.
    - The only exact route the arithmetic leaves: 64-row checks at their byte floor with ~97.8 % to depth 63 (~470).
    - 900 lossy needs 22.4 kept at the 20.9 ms floor (tree-32 16.26 -> 653).
  - Instant information: context lookup + an n-gram table of the 125B's answers gives kept +8.7 % HE+ / +9.7 % MBPP
    (upper bound) but moves bands 1-14 only +0.2-0.8. Monday: a budget-correct tree-node screen.
- Sun 15 22:02: **MTP Head S11 KPRICE: kept per check vs tree size K (LAB: teacher-forced, exact, under the walk)**
  (pre-registered TrainLab dd0ef8e, result 3ad8eb8; data/sb98/tree_kprice.txt).
  - Real to K 32; beyond that, the proxy. Its gamma 1 is calibrated to ~1 % against the real trees at K 16 / 32.
  - kept/check, MBPP / HE+:
    - K16 7.49 / 11.26 and K32 9.03 / 16.26 (real);
    - K48 9.73 / 17.18; K64 10.16 / 17.91; K96 10.88 / 18.86; K128 11.24 / 19.14;
    - gamma 0.5-2 band at K128: 9.38-13.02 / 17.00-21.46.
  - vs chain-16: K64 x1.54 / 1.69; K128 x1.71 / 1.80. Diminishing: K32 -> K128 is +24 % / +18 %.
  - Shape: a wide band at depths 2-8 (MBPP K128 ~10 nodes per depth at depths 4-5) plus a long thin tail.
    - Median max depth 16 (MBPP) / 29 (HE+) at K128.
    - 82-86 % of nodes are off the #1 spine.
    - HE+ keeps ~3.3 nodes per depth down to depth 30.
  - The 127-node dump spec for Wednesday's capture is in REPORT S11: tree_t30b's fields, every anchor, prefix check vs
    the 32-node dump.
> (a quote from the project owner, kept private)
  PLAN4D_REAIM978.md. All results are desk joins with exact self-checks.
  - **The gate is walk-strict, and there the profile flips.** HE+ deep bands 15-30 are AT the line with the audit
    (2B P=2 98.3 / 98.3; P=1 97.6 / 98.2), and band 0 with the tree (98.6).
  - **Bands 1-14 stay 90.6-97.1 for every deployable lever.** Even the true-state head gives 95.3 / 96.4 at 1-2 / 3-6:
    check starts sit on the 125B's near-ties.
  - **LAB line, no build:** the true-state head = the depth-0 guess of anchor a+d. It reproduces BigMoE's REPLACE oracle
    (20.77 / 12.72 vs 20.90 / 12.76); marginal 98.0-97.7 flat.
  - **Chooser CLOSED with head-side features too:** +0.2-0.5 pt at 23-30 out of sample, against a 96.3 / 97.1 ceiling.
    Side result: the 2B audit gets +3.5-4.5 % kept/32 from it.
  - **Item 3 (tree + audit) KILL on HE+** (x0.991-1.016 vs tree-16).
  - **My 21:3x C2 line FAILED on dev** (band 3-6 +1.2 / +0.4). Replaced by walk-strict 15-30 >= 97.8 point, LB >= 97.0,
    kept co-primary.
- Sun 15 23:4x (orchestrator, both verified): **two night results.**
  - **Leap C7c MEETS step 1** (937d6c9). The one-pass exact 32-row check is 1.345 x A16 on the 12-layer slice
    (IQR 1.286-1.364; line <= 1.35), down from 1.632 split. The slice tracks the engine (split 1.632 vs G32T 1.625).
    - ~~With R32 ~1.35, tree-32 beats tree-16 on HE+ by ~+6 %~~ **WITHDRAWN 00:4x (Leap).** On the ENGINE the one-pass
      32-row check is 1.480 x A16 (C32one 90.46 / A16 61.11 ms, Breakthrough's profile). The slice compresses ratios:
      ~half of its check is Python build, and experts are a smaller share there.
      - HE+ tree-32 vs tree-16: one pass -1.7 % (166.0 vs 169.0 tok/s); today's split -12.5 %; the contract -3.7 %.
      - Break-even: C32 <= 88.7 ms. (~~the seams' -2.15 brings it to ~88.3, i.e. a tie~~ CORRECTED Mon 08:2x:
        v32one + seams was never measured together, so ~88.3 is untested; the measured one-pass check is 90.46 ms.)
      - The biggest 32-row excess is the experts: 50.65 ms vs a 40.2 ms byte floor.
      - Engine-calibrated costs v2 went to NewIdeas (305f315).
    - E2 (Tue 00:30) gives the engine number.
  - **BigMoE M1 PASS: tree DRAFTING on the FULL 125B** (t16a in real decoding with the live MTP head; night1115.log #3,
    a9fc848).
    - CORRECTED 23:4x: M1 checks that the 16-node best-first tree's main chain == the greedy 15-draft chain on every
      check: 65 / 65 (HumanEval/1 10 / 10, mbpp/16 21 / 21, ifeval/1219 34 / 34).
    - Tree drafting costs +3.1 ms per check (41.7 vs 38.6 ms).
    - Exactness of the tree VERIFY on the full model is M4's job (Monday 23:05). Monday's M4 takes its PASS branch.
    - ng1 PASS too (bitwise with RX=2 + PREFETCH=alt).
    - The marker was written 23:40:18; EvalSuite's chain started 23:41:03 at NO-SWAP 47 GB.
  - NO-SWAP rose 38 -> 44 GB during BigMoE's runs (the compressor effect), so EvalSuite's gate may open.
- Sun 15 22:3x: **KAMEN (verbatim): "yes the mac stays home with the lid open"** on school days, so the DAY is free
  compute while he's away.
  - Away signal: ~/.local/share/quiet/away_secs.sh >= 900 (no keyboard / mouse for 15 min). Memory gates still decide,
    and 45 GB runs need his apps closed. When he's back: >= 5 GB free, yield.
  - Moved: t30c (~8 h, 15.6 GB) to MONDAY BY DAY if MTP Head's morning pre-screen passes; Tuesday night is the fallback.
  - Also by day: Sandbox98 L1CF, BigMoE's dump dry run (Tue), screens.
  - Owners added for the two unowned pillars (NewIdeas briefed them):
    - MoonShots, drafting off the critical path (<= 4 ms per 32-token block or hidden);
    - Fit12, the 32-row lossy check (a desk table by Tue; 900 needs ~21 ms).
  - Screens already run tonight (NewIdeas):
    - #1 margin-keyed switch KILL (-3.9 / -4.9 %). The key is real on HE+ (lift x4.6) and moves to Tuesday's TIE-STUBS
      arm (widen MORE after hard starts) (239fe79);
    - #4 lookup as a charged node: leaf KILL (x1.005, 4c30a27).
- Sun 15 22:2x: **THINK_1115 DONE: all five thinkers in** (combined list: THINK_1115_RESULTS.md here).
  - The convergence: ZERO, Lunatic and Fable C independently found MARGIN-KEYED SHAPE. The last check's exact 125B
    margins pick the next check's shape. Free and exact.
  - Also FAR's SIBLING-STATE DRAFTING (the rejected rows' exact states as the head's inputs; 0 bytes, exact).
  - Ranked Monday screens went to NewIdeas to order and assign.
  - Settled: K* = 32; the 400-exact wall sits in bands 0-6 after a miss, and only trees pass it.
- Sun 15 22:1x: **Lossy K* = tree-32 + close-call** (NewIdeas 2aca5eb). The two overlap (additivity 0.57 HE+ / 0.70 MBPP).
  - Kept per 32-row check, HE+: chain-32 13.77; + CC 16.78; tree-32 16.26; tree-32 + CC 16.91. MBPP: 6.92 / 8.52 / 9.03 /
    9.50.
  - HE+ tok/s: 679 at the 20.9 ms lossy floor (+4 ms); 583 at 25 ms.
  - 900 lossy needs 22.4 kept: a x1.32 gap. MBPP keeps K 16 on the lossy side.
- Sun 15 22:1x: **NewIdeas TARGETS_450_900 (331675b, verified), by pillar:**
  - 450 EXACT at 32 rows needs a drafter ABOVE the true-state head (flat 98.0-98.4 %), the exact check AT its byte
    floor (49-52 ms), and drafting hidden. A flat-97.8 % drafter with everything else perfect gives 413-436.
  - 900 LOSSY needs kept 22.4 at the 20.9 ms lossy byte floor (97.6 % flat), vs tree-32's 16.26. The lossy check is
    not built.
  - The one exact route the arithmetic leaves: a 64-row check at its byte floor (~69 ms) with ~97.8 % to depth 63,
    giving ~470. The marginal row must cost bytes only.
  - Breakthroughs needed (none in hand):
    1. the exact check at its byte floor (one-pass K-row kernels);
    2. covering post-miss near-ties >= 98 % at depths 1-14 (the true state caps at 93.2 / 95.3 / 96.4; trees are the
       only cover);
    3. drafting off the critical path (tree-32 ~24 ms unhidden).
  - The PLAN4D levers each add ~5-9 % kept, about a sixth of the exact drafter gap each.
  - Saturday story: certified per-pillar bests + measured exact check + floor projections.
  - Lines proposed to EvalSuite:
    - (a) tree-32 / chain-32 at the same rows, bar ~1.071 (dev 1.181 / 1.304);
    - (b) tree-32 / chain-16, bar R32 / R16 x c32 x 1.03, with R from Leap's E2 (1.454 at the floor vs dev HE+ 1.531).
- Sun 15 22:1x: **EvalSuite CERTIFY98 amendment 7 (ccb5dc5): bigger trees.**
  - T8 = the gated line's top-8 tree; T16 = 127 nodes, top-16, information.
  - Slot approved by the orchestrator: TOPK + T8 + T16 (~2.2-2.5 h head-only, ~12.5 GB, --part slices) on THURSDAY BY
    DAY after the proof-run amendment, as the one job above 4 GB. Scoring lands Thursday evening.
  - T8 stays at 32 nodes (K* = 32) unless NewIdeas states k and K above 32.
- Sun 15 22:0x: **BIGGER TREES, first cut (NewIdeas e324a41, verified): K* = 32 nodes on HE+, 16 on MBPP. Trees of 48-128
  never pay under any cost model.**
  - HE+ tok/s (exact floor / exact realistic / lossy 25 ms): tree-32 277 / 174 / 542; tree-16 257 / 167 / 493;
    64 229 / 123 / 456.
  - MBPP: tree-16 171 / 111 / 327.
  - The certificate line T(32) = tree-32 / chain-16 kept, pay line ~1.45 (dev HE+ 1.531 passes; MBPP fails, information).
  - Instant-information screen Monday (NewIdeas, CPU): prompt lookup + an n-gram table as a charged node in tree-32
    (upper bound kept +8.7 % HE+).
  - Power: C2 P=2 1.00, C1 band 0 1.00, LAB marginal 0.71.
  - Steers relayed: Leap 32 vs 16 microbenches first; nothing beyond 32 for MTP Head; BigMoE's dry run on heplus24 +
    mbppbank; no T64.
  - **Fable B (problem 2), corrected 22:1x against Leap's measured unions:**
    - The union keeps growing (~1.2 experts per layer per marginal row at K96-128).
    - The MARGINAL ROW IS COMPUTE, not bytes: measured ~0.7 ms (bytes 0.44 ms, MACs ~0.8 at today's kernel efficiency).
    - Sibling reuse, union-aware choice and staged checks can't pay.
    - Only EARLY DEATH can (drop dead subtrees mid-pass on a probe of the 125B's own layer-L state; repair exactly if
      wrong), plus row-dense expert tiles. A tok/s win past K32 needs early death to drop >= 70 % of rows by mid-pass.
    - A Monday probe test (a layer-16 "child survives" probe on captured states) decides whether big trees exist.
    - Its DFS-GDN duplicates Leap's TREE16 L1 (66b4af8).
    - **Final 22:1x:**
      - one-pass byte floors on the measured unions: K16 39 / 32 52 / 48 62 / 64 70 / 96 81 / 128 90 ms;
      - kept per ms 0.288 / 0.310 / 0.277 / 0.257 / 0.233 / 0.213, so K32 is the optimum;
      - EARLY DEATH at layer 8 with <= 24 survivors is x1.16 at best (x1.08 halved): BETWEEN. Monday's line: >= 80 % of
        off-spine nodes droppable at layer 8 with <= 5 % false drops; KILL < 50 %;
      - row-dense expert tiles stay as the compute lever past K48;
      - K* = 32 holds this week.
  - Thinkers so far:
    - ZERO (problem 1): TIE-RUN (the last check's 125B margins predict the next near-ties), TIE-STUBS, SLOT PRIOR,
      DISAGREEMENT FLAG;
    - FAR (problem 3): SIBLING-STATE DRAFTING (feed the head the 125B's exact states from the rejected rows, a
      near-tie sibling prefix, instead of its own drifting state; 0 bytes), COPY-SOURCE STATES, TREE-STATE REUSE.
- Sun 15 22:1x: **KAMEN (verbatim): "i mean this will most porbably need a breakthrough. but i literary cant even tell you how
  much i need to do this. i need to achieve 900 tps in lossy and 450 in exact on code."** Targets on code: 450 EXACT / 900 LOSSY.
  - Arithmetic (kept per 32-row check -> the cycle needed):
    - 450 exact: tree-32 16.26 -> 36 ms; true state 20.77 -> 46 ms; flat 97.8 23.15 -> 51 ms. The exact byte floor is
      52 ms, so it needs ~97.8 %-flat drafting AND an exact check at or below today's byte floor, with drafting hidden.
    - 900 lossy: 18 / 23 / 26 ms cycles at those kept values.
  - Added to THINK_1115 for the thinkers; sent to NewIdeas for the K* / plan framing.
> (a quote from the project owner, kept private)
  an hour alongside the tests which are running to try and solve these issues in any way").
  - The brief is THINK_1115.md here. Five desk-only Fable thinkers, one problem each:
    - fable-zero: where to branch;
    - fable-b: cheap big trees;
    - fable-far: cheap information for the drafter;
    - fable-c: near-ties after a check;
    - Lunatic: question the limits.
  - Ideas files and summaries are due 22:45. They run on the unused Fable weekly allowance.
  - MTP Head S11 KPRICE (3ad8eb8), kept per check vs K (MBPP / HE+): K16 7.49 / 11.26; K32 9.03 / 16.26;
    K64 10.16 / 17.91; K128 11.24 / 19.14. Returns diminish (K32 -> K128 +24 % / +18 %). The 127-node dump spec
    went to BigMoE (top-16 children suggested).
- Sun 15 22:0x: **KAMEN DECIDED (verbatim): "ok go for the bigger trees then."** BIGGER-TREES push to Saturday, NewIdeas
  leads. Scope "just the tests": certify on the fresh capture; no tree-K engine build by Saturday.
  - Goal: the throughput-best exact tree size K* (32 / 48 / 64 / 96 / 128 nodes), certified.
  - Pieces:
    - MTP Head: kept(K) on HE+ / MBPP, real to K 32, proxy beyond, by Mon evening.
    - Leap: R(K) floor + realistic (REF2-T and the contract) and the tree overhead c(K), by Tue evening; R32 from
      tonight's C7c and Monday's E2.
    - BigMoE: a 127-node head-only best-first dump (top-8 children) in Wednesday's capture; dry run Tue by day.
    - EvalSuite: a tree-K* line (T = tree-K / chain-16, pay line R(K) x c(K)) in the pre-capture amendment, Wed ~18:00.
    - MoonShots: T32 already MATCHES SIM; a T64 on the 2B only if NewIdeas wants it.
  - K* with tok/s (exact floor / realistic; lossy 25 ms) is due Tue evening.
  - BigMoE 22:0x: the CERTIFY98 tree dump goes to 127 nodes per anchor [est until Tuesday's dry run].
    - Head-only + the 125B's lm_head, no 45 GB model.
    - ~55 ms per anchor x ~114K fresh anchors = ~1.8 h (+1.4 h vs 32 nodes), in --part slices.
    - Peak ~12-12.5 GB, so it can run by day as the one big job. Outputs ~0.6 GB.
    - Single look: it runs only after EvalSuite's amendment.
    - Tue dry run on the bank (heplus24). Gate: the first 32 pops == today's 32-node dump bitwise, plus TOPK
      exactness.
- Sun 15 21:5x (orchestrator): **NewIdeas' re-aim at 97.8 verified (8df555d) and its lines APPROVED** for EvalSuite's
  pre-capture amendment (Monday morning).
  - Lines: the LAB true-state line; C1 tree band 0 >= 97.8; C2 walk-strict 15-22 / 23-30 >= 97.8 with LB >= 97.0,
    plus kept/32.
> (a quote from the project owner, kept private)
    the information it needs literary instantly. thats all".
  - Answered with the TRUE STATE row: perfect information gives kept/32 20.77 on HE+, not the flat-97.8 23.15.
    Walk-strict bands 1-6 cap at 95.3 / 96.4 because of the 125B's near-ties; trees, not information, cover those.
  - Tok/s on HE+ (32-row check; exact floor 52 ms / lossy 25 ms):
    - tree-32 16.26: ~313 / ~650;
    - true state 20.77: ~400 / ~830;
    - flat 97.8 23.15: ~445 / ~925.
  - NewIdeas was asked for any unclosed "instant information" source as a Monday screen.
- Sun 15 21:50: **MoonShots TREE-E2E T32: tree-32 vs chain-16 at one 32-row shape on the 2B + s30 MATCHES SIM on both
  sets; EXACT 264 / 264; pass@1 identical** (9e41a54; verified from tree_result_t32_fixed_b32.json).
  - Full panel: x1.343 HE+ [1.329, 1.358] / x1.367 MBPP [1.345, 1.387].
  - Like for like: x1.360 / x1.359 vs the sim's x1.332 / x1.368, both inside the CIs.
  - So the sim behind the tree-32 line (S9c on the 125B: x1.371 / 1.531) holds in real decoding at 32 nodes too.
    Whether it pays still depends on R32 (tonight's C7c, Monday's E2).
- Sun 15 21:4x: **KAMEN (verbatim): "no no. 97.8 is absolutely good enough. dont push for 98. i just need that by
  saturday with everything done. i mean i cant explain how much we need to do it"**
  - The TARGET is now 97.8 %, not 98 %.
  - Careful: 97.8 % is the head's TRUE-STATE (no-drift) accuracy, flat to 30 on HE+, i.e. depth 0 at every anchor
    (97.94). Real drafting has no true state ahead, so real deep bands are lower:
    - t30b alone, 23-30: 89.65;
    - + 2B audit: 94.32; + 4B audit: 95.29; oracle head-or-2B: 96.29.
  - So "97.8 in real use" = removing almost all drift. No tested lever does that this week; it was told to Kamen
    honestly at 21:5x.
  - Plan: PLAN4D stays, re-aimed at 97.8 (each lever's pass line vs 97.8 per band). Also certify the true-state 97.8
    on the fresh set as a lab line (EvalSuite).
> (a quote from the project owner, kept private)
    at full speed (much more building)?
> (a quote from the project owner, kept private)
    - Saturday = everything CERTIFIED by tests. No full engine build is needed by then.
    - Tue-Fri nights: his apps are quit at 23:30 with tonight's procedure.
      - Tue: t30c, only if Monday's pre-screen passes.
      - Wed: the capture.
      - Thu-Fri: the proof runs.
  - His question "if we got 96 percent with the perfectly optimised check how many tps is that?" was answered by the
    orchestrator. A flat 96 % per guess means 18.2 tokens per 32-row check:
    - ~350-370 tok/s at the exact floor (52.0 / 49.1 ms) with drafting hidden;
    - ~310 at a realistic ~55 ms check + ~4 ms overhead;
    - ~240 if today's ~19 ms drafter isn't hidden.
    - (97.8 % gives 23.2 tokens: ~445-470 at the floor, ~390 realistic.)
  - Leap E14 amendment 1 (7d93cab): Breakthrough's lever 2 joins Monday's engine A/B as A16m / C32onem.
    - Line: PASS <= -2.4 ms, KILL > -0.5.
    - Fail-safe: any lever-2 error falls back to K-73.
    - Runner pid 9955 unchanged.
- Sun 15 21:3x (NewIdeas, Goal A lead): **PLAN4D for HE+'s deep bands** (the orchestrator's 21:2x ask;
  ~/NewIdeas/research/week98/PLAN4D_HEPLUS.md, mirror LEVERS_A).
  - **Honest ceiling:** 98 % per band to 30 is out of reach on HE+ this week. The true-state head is 97.76 % flat, and
    the best measured exact deep fix is the audit: 2B 94.3 / 4B 95.3 at 23-30. MBPP is knowledge-limited (oracle 93.6;
    audited 85.1-87.2).
  - **Ranked (all exact):**
    1. t30b + frozen 2B audit (C2): +4.7 at 23-30, certifiable offline on Wed's cells.
    2. NEW: a full-data distilled auditor. The audited level IS the auditor's agreement, and 1c used 360 files with a
       one-layer LoRA, so ~8.4x the rows is untested. Mon kill-test ~30 min at <= 7 GB.
    3. NEW: tree at depths 0-2 + audited chain (desk).
    4. t30c second pass (+0.5-1.5 unaudited; Tue night only, ~8 h at 15.6 GB, after MTP Head's sandbox pre-screen).
    5. Full-budget state arms (sandbox first).
  - **Pins by Wed ~18:00:** C1 tree (already pinned), C2 audit, C2b distilled auditor if it passes, C3 t30c if it PAYS.
- Sun 15 21:22: **MTP Head K98 = NO: no exact tree up to K 128 holds coverage >= 98 % in every band to depth 30. The
  head's own accuracy binds, not tree width** (CPU on BigMoE's TOPK + TREE; pre-registered TrainLab 5bb6024, result
  ad85682; data/sb98/tree_k98.txt).
  - Cross-check: K 16 depth 0 = 97.8 / 98.6, as above.
  - Best-first EAGLE-2, bands 0 | 1-2 | 3-6 | 7-14 | 15-22 | 23-30:
    - MBPP K 32 real: 98.6 | 94.3 | 89.2 | 83.9 | 84.9 | 84.2.
    - MBPP K 128 proxy: 99.6 | 97.4 | 92.7 | 87.3 | 87.6 | 87.5.
    - HE+ K 128 proxy: 99.1 | 98.6 | 96.0 | 95.6 | 97.3 | 97.7.
    - K >= 48 rests on proxied nodes (validated to ~1.7 pt per band at K 16 / 32). Even the optimistic proxy at K 128
      is 85-90 % in MBPP bands 7-30.
  - Exploratory per-depth beams do no better: K 125, and K 249 gives MBPP 88-91 % in bands 3-30.
  - The top-8 ceiling (98-100 % per band) does NOT bind.
    - Once the true path takes a non-#1 step it ranks below the main paths and gets #1 continuations only. So deep
      coverage tends to the head's per-position #1 accuracy: MBPP ~87-89 %, HE+ ~96-97 %.
    - Holding 98 % by width would need ~1e5 nodes.
  - So "98 % at every depth to 30, exact" needs the head's mid / deep accuracy near 98 % (the drift / state fixes).
    Trees deliver bands 0-2 and the kept gain (S9c).
  - No new dump changes this. A 127-node dump would only pin down HE+'s deep bands.
- Sun 15 21:2x: **KAMEN (verbatim): "i really really REALLY need to get it by saturday. you just dont understand how
  important this is".**
  - Orchestrator's reading: Goal A's literal claim, 98 % at EVERY depth to 30 (exact), is the open piece.
    - Tree-16 meets 98 % only at depth 0 (coverage HE+ 98.6 / MBPP 97.8).
    - Bands 3-14 stay at or below the chain (HE+ 93.7-96.2, MBPP 80.4-86.1).
    - The certificate as amended (6a) proves the tree gain, not "98 % to 30".
  - New task tonight, **K98**: the smallest exact tree (K nodes, best-first EAGLE-2 on t30b) that holds coverage
    >= 98 % in every band to 30.
    - MTP Head: CPU on BigMoE's dumps, K = 16-128; it marks the proxied bands and specs any new head-only dump.
    - MoonShots: the real-numerics twin on the 2B + s30 after T32, before 23:25.
  - If K is reachable:
    - BigMoE adds the K-node head-only dump to Wednesday's capture;
    - EvalSuite pre-registers "coverage >= 98 % per band (LB >= 97) with a K-node tree" (amendment 7) before the
      capture;
    - Leap prices a K-row check (the contract kernels at K rows).
  - Then Saturday's answer: 98 % at 30 deep, exact, with a K-node tree, certified on fresh data, at the measured check
    cost.
  - **K98 = NO** (MTP Head, pre-registered 5bb6024 21:19, result ad85682 21:22; verified).
    - No exact tree up to K 128 (beam K 249) holds 98 % per band. The head's own deep accuracy binds.
    - MBPP bands 3-30 are 84-90 % even at K 128; HE+ bands 3-14 are 93-96 %.
    - Once the true path leaves #1, it gets only #1 continuations, so deep coverage tends to the head's per-position
      accuracy. Holding 98 % by width would need ~1e5 nodes.
    - The true-state ceiling is HE+ 97.8 % flat and MBPP 93.6 %. So HE+ is a drift gap of ~1-4 pt; MBPP is also
      knowledge.
  - **21:2x: NewIdeas leads a 4-day plan** for t30b's own deep accuracy on HE+ (bands 3-30 toward 98 %), certifiable
    Thu-Fri. MTP Head and MoonShots feed it levers. A new candidate must be named and pre-registered before
    Wednesday's capture. Due ~23:15 or first thing Monday.
  - **21:4x PLAN4D APPROVED by the orchestrator** (NewIdeas 2d196fb; PLAN4D_HEPLUS.md here).
    - Honest top line: 98 % per band to 30 is out of reach this week even on HE+. The true-state ceiling is 97.76 %
      flat. The best exact deep fix is the 2B audit (HE+ 23-30 89.65 -> 94.32).
    - MBPP's exact bands 3-30 can reach ~85-88.
    - Assignments:
      - tonight, CPU: NewIdeas item 3 (tree 0-2 + audited chain); MoonShots item 1's read;
      - Mon by day: Sandbox98 item 2 (full-data distilled auditor, <= 7 GB, briefed); MTP Head item 4's pre-screen
        (<= 5 GB), then item 5.
    - Not tonight: free memory was ~11 GB at 21:39 (Kamen building in Xcode).
    - Pins by Wed ~18:00: C1 tree, C2 t30b + 2B audit, C2b / C3 if they pass.
    - Tue night (t30c, ~8 h at 15.6 GB) needs Kamen's app OK; asked 21:4x together with Wed-Fri.
  - Tonight's chain is re-armed and verified at 21:24:
    - apps quit 23:30 (pid 61214);
    - Leap a45 62263 -> C7c 64828;
    - BigMoE night1115e 66280;
    - EvalSuite 71926 (now stops cleanly on HOLD, with a marker);
    - Breakthrough's profile 76388 (replaced 62519 at 21:28: it now skips on HOLD and waits out PAUSE, 8d85240), then its lever-2 quiet bench (follower 88412, ca8f75f; latest start 06:45, done by 07:00);
    - Monday: m4_night 50884, E14 11089 (re-armed 00:3x with 8 arms, lever 2 dropped, amendment 2 26456b0);
    - no markers present.
- Sun 15 18:1x (orchestrator; Leap's three results, verified in commit order):
  - **BOF PASS with REF2-T's OWN bits** (pre-registered 4a4fc62, bits 9b5b19a, result b380726): lever (b) rebuilt
    as a blocked ordered fold. qmm16c's serial K loop is spread over simdgroups, and the per-group terms are added
    back in the original order.
    - Bitwise vs REF2-T on all 5 shapes x 4 configs.
    - -2.22 ms per 16-row check standalone; o_proj / out_proj 85.9 -> 48.6 us.
    - That is 78 % of the contract's dense gain on the same shapes, with no reference change.
    - Next: an opt-in installer, a slice A/B, then an E1 arm Monday night.
  - **C8: the contract carries to the 12-layer slice** (pre-registered b147947, PASS 5457853, run after the D8 fix):
    - -1.31 ms per 16-row check, -1.05 per 32-row; transfer 0.81 / 0.67 of CB1;
    - under the contract, a 32-row check == 2 x 16-row and 16 one-row == one 16-row, every logits row, so SD == AR
      holds end to end under the new reference;
    - the contract arms join E2.
  - **D8, FIXED 574060e:** the slice's one-call resident experts at 17-32 rows dropped every (row, expert) pair past
    16 per expert, leaving zeros.
    - The engine is not affected: NAX6's grouping caps at 16 pairs by design (969eac1).
    - Slice timings stand. Slice 32-row BIT verdicts made before the fix (C7 / v32one, pass S) get re-run on the
      fixed slice before E2, or each is shown unaffected (asked 18:1x).
    - **AUDIT DONE 18:16 (6a8a257, verified):**
      - C3b and C3-ONE were re-run on the fixed path at 4 and 12 layers: ALL BITWISE. Their old verdicts held only
        because no expert passed 16 of 32 rows on those inputs.
      - Cleared with reasons:
        - pass S (both arms share the expert path);
        - unit tests with no experts;
        - the engine runs (NAX6 groups at <= 16);
        - timing-only runs.
      - C8 v1 is void (v2 replaces it).
  - **C9, BOF in a real check on the 12-layer slice** (pre-registered 0a8c085, result 9007c6e):
    - exact: 32 / 32 logits rows == REF2-T;
    - the whole-wall pair is -0.30 ms, T 0.46 -> BETWEEN by its line (IQR +-5 ms; the Python build is half the
      slice's wall);
    - after the build, -0.49 ms with a tight IQR (info only).
    - The engine arm (E1, Monday night) decides.
  - Monday order, Leap's proposal: E2 (R32), then the contract arms (A16c / C32c), then E1 (hc_b + BOF), then E3.
  - **18:32 C9b + C10b PASS, exact with REF2-T's own bits** (re-tests pre-registered 2a7f0a8 with unchanged lines,
    result 136ddc1; verified in commit order):
    - BOF at 16 rows: -0.752 ms per slice check (IQR -0.85 / -0.65, T 1.14).
    - BOF2 on o_proj / out_proj at 32 rows: -0.497 ms (IQR -0.58 / -0.42, T 1.06).
    - Why the re-tests: in C9 / C10 the first arm of each round read the PLE n-gram rows cold, giving +-5-8 ms of
      whole-wall noise. The fix is one untimed warm check per round (IQR ~0.2 ms). C9 BETWEEN and C10 KILL stay in the
      record as registered.
    - Caution: BOF2-O is a post-hoc subset (the 5-shape BOF2 was a KILL, 578279c) and then a re-test. Treat it as a
      lead. Its engine arm needs its own pre-registered line.
    - Engine-scaled [est]: about -2.5 ms (16 rows) / -2 ms (32 rows). The REF2-T central is ~55-56 / ~74; Monday's
      engine arms decide.
  - Estimates: REF2-T 16 rows ~56-57 ms with BOF's expected engine share (~-1.5). The contract adds ~-2..-3 on the
    engine, attention later.
  - Monday slot: Leap's hard end moves 02:15 -> 03:00, ordered by value (E2 first: its R32 decides the tree-32 line).
    EvalSuite's fallback moves to ~03:15. Leftovers go to Tue night, which needs Kamen's OK for an app quit.
- Sun 15 17:38: **Leap CB1: the track-B REFERENCE CONTRACT passes both pre-registered lines, ALL BITWISE**
  (pre-registered 8746000 at 17:30, smoke 8aecdaa, run 17:35:57 on a quiet ticket, result 58c2439; verified from
  screens/cb1.txt).
  - What it measured: 7 dense four-bit kinds at the 125B's real shapes and counts, plus the HC down GEMM, each with one
    fixed K-chunk per shape, against REF2-T's forms. Cold weights.
  - Bits: every config bitwise across M = 1 / 16 / 32, nt and layout. So SD == AR holds by construction under the new
    reference. 1-row calls pad to 16, so AR speeds up the same way.
  - Per check over this set:
    - 16 rows 18.52 -> 12.54 ms (-5.98; line PASS <= -4.0);
    - 32 rows 22.52 -> 16.11 (-6.41; line PASS <= -6.0).
  - The biggest part is o_proj / out_proj at a 4-way K split, 87 -> 44 us per call (-2.1 ms); then the HC read at
    2,048-key partitions (-1.5), the shared cat (-0.9) and the in-proj (-0.65).
  - Standalone only. By the transfer rule (DN-KS kept ~55 % on the engine), expect ~-3 ms on the engine.
  - Next: a slice A/B via an opt-in installer, then a native-layout 32-row kernel.
  - So next week's one re-certification has a tested contract. HC up and attention by key lists come on top.
- Sun 15 17:37 (`date`) **NewIdeas: L3 HC SLC STAGING = KILL in substance** (NO JUDGEMENT by its pre-registered context
  gate; quiet 17:36-17:37; NewIdeas 471a068, research/week98/l3_slc_bench.txt).
  - D3-0 reproduced at 33.2 / 13.0 us (vs 38.9 / 18.7; the cached figure is outside +-25 %).
  - In situ (evict -> window -> o_proj -> S4), S4 is only 14 % byte-bound: cold 55.4 vs bytes-free 47.8 us. So the
    ceiling is <= 0.73 ms per 16-row check, and the touch was not hidden (+25.4 us per window). Closed by its ceiling.
  - Track A at 16 rows stays Leap's ~57-58 ms, minus lever 2 (the MoE dense side, -2.7 est) if it lands.
- Sun 15 17:3x: **TREE16 B3 DONE (Breakthrough), ahead of its Tue-Wed date. The exact tree check is now ALL BITWISE end
  to end above 2K context** (verified by the orchestrator from the logs).
  - Bit test (pre-registered cef6107, result bd0695c): 900 / 900 nodes' per-node QSA selection masks == their path
    chain rows, over 11 trees x 6 contexts (L 3000-3003, 2036, 2044). Today's stock node-order mask differs on 548 / 888,
    so the test can fail.
  - End to end (281470d): Leap's tree_verify + B1 + B2 + B3 on layers 0-3 at P 3000 / 3002. Logits + hidden of all 302
    nodes over 20 trees == their path chains.
  - Integration:
    - Leap wires in two small hooks;
    - B3 joins the NEXT snapshot after Monday, with Breakthrough's T2 probe at 3K on its own ticket;
    - Monday's M4 keeps its frozen t16b. Its keys (81-367-token prompts + 256 new) never pass a + T = 2044.
  - **17:45 BigMoE (648826b, verified):** TREE16 + B3 in the decode loop on both slices, ALL BITWISE.
    - 200-token decodes at P 1024 / 2030 / 3000; every kept row == REF2 AR.
    - B3 was live on 58 / 60 checks at P 2030 and 60 / 60 at P 3000; side paths 12; the chain control was clean.
    - t16b was untouched.
    - Next: t16c after Monday, then a pre-registered full-model long-context arm on its own ticket.
  - Breakthrough moves on to CHECK55 lever 2.
- Sun 15 16:4x: **MoonShots TREE-E2E: the exact tree check works in REAL decoding** (2B + s30; MoonShots 7471145;
  verified by the orchestrator from week98/tree/tree_result.json).
  - Setup: EAGLE-2 tree-16 vs chain-16, 16 rows per check, path-batched, one batch shape (B = 16) for every target
    step. 264 keys; pre-registered in LEVERS_A before any code ran.
  - EXACT 264 / 264 token for token, so pass@1 is identical (HE+ 43.29 / MBPP 29.00).
  - Kept per check:
    - HE+ x1.148 [1.135, 1.160]: 6.041 -> 6.933, the same 46,275 tokens in 7,660 -> 6,675 checks;
    - MBPP x1.188 [1.175, 1.201]: 4.935 -> 5.864.
  - Like for like with the sim (its keys, first 512 tokens): x1.166 / x1.216 vs the sim's x1.139 / x1.198. MATCHES
    SIM; real is slightly better on HE+.
  - So the sim method behind S9c and CERTIFY98's tree line holds in real decoding. The 2B's gain is larger than the
    125B's because its drafter is weaker. This checks the method, not the 125B's size.
  - Side finding: on stock MLX kernels the batch shape alone (B >= 2 vs B = 1) flips 142 / 264 answers, all at
    near-ties (top-2 gap <= 0.375). The 125B is not exposed:
    - REF2 pads every step to one 16-row shape, AR included (R-40);
    - R57_REPLAY and TREE16 are bitwise;
    - at 32 rows, C32 must stay == 2 x 16 (CHECK55's one-pass rule).
  - Next: T32 (tree-32 vs chain-16, both at one shape B = 32), approved for this evening, pre-registered first.
    - It is the real-decoding check for a possible tree-32 line: S9c gives tree-32 x1.531 kept on HE+ vs x1.060 for
      tree-16.
    - That line pays if tonight's C7c and Monday's E2 bring R32 near 1.3.
- Sun 15 16:22 (`date`) **NewIdeas: CHECK55 RESEARCH PASS for Leap** (Goal B; CPU / desk; NewIdeas
  research/week98/CHECK55_LEVERS.md, 09ee3d7). A ranked table: track A (REF2-T) / B (next week's reference), priced
  from Breakthrough's class times, TRIED-tagged, each lever with its kill-test. Top 5:
  1. DENSE-N re-test (A, bitwise, built): -1.9 ms at A16. ISO2's ~0 was measured while the check was CPU-bound.
     **CORRECTED 16:25 (c2620a9):** wrong. ISO2 was a rotated A/B on the engine with the check GPU-bound: +0.7 ms.
     Lever 1 and the GDNB re-test are CONDITIONAL on tonight's engine profile.
  2. MoE dense side as ~2 exact dispatches (A): -2.7 / -5 ms.
  3. **HC SLC staging (A, new):** stage the next HC read's ~6.65 MB mixers during the latency-bound kernels; ceiling
     -1.9 ms from D3-0's cold / cached gap.
  4. **Track B reference CONTRACT:** a fixed K-chunk independent of M, code-MMA order, an ordered fp32 reduce, so we
     re-certify once and later kernels stay bitwise. It enables projections and HC at ~1.3x their floors.
  5. The contract's bench (3 shapes x chunk x M = 1 / 16 / 32, the bit test across M).
  - Track A in total: ~52-55 / ~70-76 ms at 16 / 32 rows. **Re-totalled 16:25 without lever 1 / GDNB: ~53-56 /
    ~68-76.** Track B adds ~-11 / -16 (Leap's ~45 / ~62).
  - **CORRECTED 17:1x (Leap):** the re-total still counted Leap's levers (a) and (c), which Leap closed at 14:3x
    (QMVW16 KILL; ATT-HCHUNK parked).
    - 16 rows under REF2-T: **~57-58 ms**, from hc_b ~-1, lever (b) -1..-1.5 (not built yet) and step 2 ~-2.
    - **~53-55** only if NewIdeas' lever 2 (MoE dense side, -2.7) and lever 3 (HC SLC staging, -1..-1.9) pass
      paired engine A/Bs.
    - **17:37: lever 3 is KILLED in substance** (NewIdeas 471a068; quiet ticket 17:36-17:37, after MoonShots' T32;
      verified from l3_slc_bench.txt).
      - In situ, S4 is only 14 % byte-bound (cold 55.4 vs bytes-free 47.8 us), so the ceiling is <= 0.73 ms.
      - The touch was not hidden (+25.4 us per window; touched S4 77.3 us).
      - So the 16-row check this week is ~55 ms if lever 2 lands, else ~57-58.
    - The 32-row figure waits for C7c tonight.
  - Owners (17:2x):
    - lever 3 microbenches: NewIdeas tonight, one quiet ticket after MoonShots' T32 (~19:30), done by 22:30, Leap's
      kernels read-only;
    - the contract bench (7 + 8): Leap, Mon by day;
    - lever 2: Breakthrough after B3 (its piece 2), first read in tonight's engine profile; Leap integrates.
- Sun 15 16:1x (orchestrator): **NewIdeas takes a desk-only CHECK55 research pass** (its offer, accepted; Goal B has
  equal priority).
  - Two tracks:
    - (A) levers exact under REF2-T this week: same per-output arithmetic order, batch-invariant;
    - (B) the design of next week's re-certified bandwidth reference: fixed split-K partitions that don't depend on M
      and a fixed-order reduce, for the dense GEMMs and the HC read at 16-32 rows, so SD == AR stays bitwise. Get it
      right before re-certifying once.
  - Output: a ranked table (ms per engine check + the cheapest kill-test for each lever), top 3-5 to Leap by ~21:00.
    Passes can join Monday night's E1 (Tue 00:30).
  - Leap decides what to build. No GPU from 22:30.
  - **DONE 16:21** (NewIdeas 09ee3d7, ~/NewIdeas/research/week98/CHECK55_LEVERS.md; verified by the orchestrator):
    - Track A:
      - MoE dense side as ~2 exact dispatches: -2.7 / -5 ms [est];
      - NEW: HC SLC staging, where a touch kernel warms the next HC read's mixers during latency-bound windows:
        ceiling -1..-1.9 ms [est], from D3-0's cold / cached gap;
      - Leap's own list.
    - Track B: one REFERENCE CONTRACT for every dense GEMM, the HC read and the head:
      - a fixed K-chunk that doesn't depend on M;
      - the code-MMA order inside a chunk;
      - an ordered fp32 sum of the chunk partials;
      - one bf16 rounding.
      Any later kernel that honours it stays bitwise, so we re-certify once. Its bench (one afternoon) picks the chunk per
      shape. With it the dense side drops a further ~-11 / -16 ms (16 / 32 rows), which reaches Leap's ~45 / ~62.
    - **CORRECTED 16:3x by the orchestrator.** Lever 1 (a DENSE-N re-test, -1.9 ms [est]) rested on "ISO2 was measured
      while the check was CPU-bound". That is wrong:
      - ISO2 was a rotated A/B on the engine (BigMoE WED_TESTS.md:2042): DENSE-N +0.7 ms, GDNB +0.2;
      - the check was GPU-bound then too. The profile only corrected how C6 had been read.
      So lever 1 and the GDNB re-test wait for tonight's full-engine profile: either the engine is GPU-bound and they
      don't transfer, or the gate's CPU service hides every GPU lever. NewIdeas re-totals and re-sends to Leap.
    - **Fixed 16:25** (NewIdeas c2620a9, verified):
      - lever 1 and the GDNB re-test are CONDITIONAL;
      - a TRANSFER RULE: a standalone or slice gain counts only after a paired engine A/B;
      - SLC ~24 MB is now [est] (Leap's KNEE measured 2-8 MB re-reads at 2.3-3.2 TB/s, 12 MB 1.1 TB/s);
      - track A re-totalled to 16 rows ~53-56 ms, 32 rows ~68-76 (conservative: the listed items sum to ~51-56 /
        ~62-72).
      - Leap's new order: HC SLC staging, the MoE dense side, its own list without GDNB, then the track B contract.
- Sun 15 14:46 (`date`) **NewIdeas: DRAFT TREE LINE for CERTIFY98** (sent to EvalSuite; NewIdeas research/week98/TREE_LINE.md;
  dev numbers from tree_coverage.py on BigMoE's real t30b tree dumps).
  - **SUPERSEDED by CERTIFY98 amendments 6 / 6a** (EvalSuite c915da9, 598d5fb; agreed):
    - a delta-method LB (CR1, t(K-1));
    - each method under its own walk;
    - coverage in certify98's walk convention (HE+ depth 0 93.33 -> 98.23, MBPP 89.72 -> 97.61);
    - Leap's measured c16 1.025, so PASS = LB >= 1.025 AND T >= 1.05575 (strict 3 % net); HE+ power 0.72, MBPP 1.00;
    - candidate t30b CONFIRMED (dd_t30b.safetensors, sha256 790c42db...215d502).
  - **Candidate:** t30b + EAGLE-2-16. **Statistic:** T = tree-16 / chain-16 kept per check under the walk.
    **Uncertainty:** a window-clustered bootstrap.
  - **Line** (HE+ and MBPP; everyday reported only):
    - PASS: one-sided 95 % LB(T) >= 1.021 (Leap's c16) AND point T >= 1.052 (S9c's 3 % net).
    - BETWEEN: point >= 1.052 with LB < 1.021.
    - FAIL: point < 1.052.
  - **Power**, dev SE scaled to the fresh windows:
    - MBPP ~1.00 (T 1.136, SE 0.006).
    - HE+: 1.00 on the LB and 0.87 on the point (T 1.060, SE 0.0071). At dev's 95 % low end (1.027): 0.20 / 0.00. HE+'s
      net margin binds.
  - **"98 % exact" with trees** = walk-strict COVERAGE per position: the 125B's word is among the verified candidates. For
    a chain this nests the per-guess metric.
    - Dev, chain -> tree-16 at depth 0: HE+ 93.80 -> 98.58, MBPP 90.03 -> 97.79. Trees meet 98 % at the check-start wall.
    - Deeper (3-14), the 16-node budget leaves coverage at or below the chain's (HE+ 93.7-96.2; MBPP 80.4-86.1).
    - So the certificate states coverage per band + T. It makes no single "98 % to 30" claim.
- Sun 15 14:42 (`date`) **NewIdeas: CALIBRATED TREE SCORES = KILL** (CPU, 3.2 s; research/week98/calib_tree.py / .txt).
  - What: isotonic calibration of t30b's edge probabilities, per depth bucket (0 / 1-2 / 3-6 / 7-14 / 15+) x walk-start
    flag, CROSS-FIT on half the windows and scored on the other half, then EAGLE-2's top-15 by the calibrated path
    score.
  - Result: x0.996 (MBPP 7.455 vs 7.487) and x0.988 (HE+ 11.126 vs 11.263). The line was +2 % on both sets.
  - Read: the head's raw probabilities already order the nodes as well as any per-depth monotone map does, and a
    per-depth temperature table is a special case of one. ORACLE-15's headroom (x1.160 / x1.076) is information the head
    lacks (which sibling is right), not miscalibration. Nothing to add to BigMoE's M1 from this.
- Sun 15 14:40 (`date`) **NewIdeas: TREE SHAPE = EAGLE-2's order stands; TREES x CLOSE-CALL = OVERLAPPING** (CPU, 2.4 s, 0.14 GB;
  NewIdeas research/week98/tree_shape_cc.py / .txt / .json). Data: BigMoE's real t30b tree dumps; MTP Head's S9c walk,
  which it reproduces (EAGLE-2-16 7.487 / 11.263).
  - **Shape** (re-select 15 of the 32 dumped nodes, 16-row check):
    - Wider trees by a depth discount lose: x0.959-0.995 MBPP, x0.973-0.995 HE+.
    - Root tempering (more first-guess siblings) is noise: x0.992-0.998 / x1.002-1.004.
    - ORACLE-15 (the best reshaping of these nodes) is x1.160 / x1.076.
    - So no confidence heuristic beats EAGLE-2's path-probability order. The headroom needs better-calibrated node
      scores, not a shape rule. **Build EAGLE-2's order as is.**
  - **Trees x close-call (delta 1)**, kept/check, budget-correct, with the existing CC walks' approximation after a
    close-call:
    - MBPP: chain-16 6.588 | +CC 7.919 (x1.202) | tree-16 7.487 (x1.136) | tree+CC 7.968 (x1.209).
    - HE+: 10.625 | 12.097 (x1.139) | 11.263 (x1.060) | 11.762 (x1.107).
    - Additivity is 0.62 / 0.54: OVERLAPPING. On the lossy line trees add ~0 (MBPP) or lose (HE+), since a tree spends
      nodes on siblings that close-call already rescues.
    - First chain miss on the walk, rescued by: tree only 27 / 21 %, CC only 15 / 16 %, both 12 / 16 %, neither 45 / 48 %.
  - **Read:** trees are the EXACT line's lever (as ranked). If close-call ships on the lossy line, a chain + CC is as good as
    a tree + CC. A tree + CC policy that also close-calls selected siblings needs the 125B's lp for them (not in the dumps).
- Sun 15 11:46 (`date`) **NewIdeas: SHORT RANKING, exact first (Kamen ~10:4x).** Replaces my 11:46 entry, which ranked
  lossy first and kept the levers that are now closed.
  1. **Trees, exact: THE lead.** Tree-16 now (net ~+11-13 % MBPP / +4-5 % HE+, S9c); tree-32 once R32 is ~1.3. The #2
     branch at a check start is the only exact way to get "the 125B's own pass on the corrected token". It also fits
     S6: ~90 % of check-start misses sit on near-ties, which a branch covers and no reader can predict. Build split:
     Leap kernels, Breakthrough metadata, BigMoE the loop; lands ~Thu.
  2. **CHECK55 (Goal B, equal priority):** R32 to ~1.3 is also the tree's lever (tree-32 x1.371 / 1.531 kept).
  3. **S5b** (Sandbox98, running): the 2B reading the 125B's wd through gated cross-attention. Keep it as a screen only:
     near-ties cap any reader.
  4. **2B audit as a tree-branch proposer:** only if MoonShots' ANE route makes the 1.06 GB pass ~free; else parked.
  5. **Lossy side track:** close-call delta 1 / delta 2 + the RUNAWAY guard, Monday-night gate.
  - Closed: training levers (WALKW, ACCSET, margin loss), 1c / data scaling, chain depth, ALLROWS, kNN, a separate
    1-row bonus pass (trees do it inside the check). DFlash2 as a corrector is parked.
  - **CORRECTION:** my 03:0x root-tree numbers are VOID: roottree.py misread `rank` (BigMoE). The margin ceilings stand
    (they use right + m only).
  - **Queue flags:**
    - (i) Wednesday's CERTIFY98 capture measures per-guess 98 %, but the exact lead raises kept/32. Add a tree-16
      kept/32-under-the-walk read-out to the same capture, so it measures what ships.
    - (ii) Missing CPU screens on BigMoE's TREE dumps (analysis-only, fine during the pause): the tree shape at check
      starts (root width vs depth, by the head's margin), and trees x close-call, to see whether their gains add on the
      lossy line.
    - (iii) Leap's 32-node tree overhead is unpriced, and tree-32's pay line needs it.
- Sun 15 11:4x: **KAMEN'S ANSWERS** (question box he asked for):
  - **Exact reference switch: NEXT week**, after Saturday's certification. This week TREE16 uses attention route (b).
  - **45 GB nights: apps quit (force-quit OK) on both nights, but ONLY AFTER 23:00.**
    - Tonight's chain shifts ~2 h: a45 23:00 -> ~03:00, tier-0a ~03:00 -> 04:30, lx45cc1 ~04:30 -> 06:15, then
      Breakthrough's profile (deadline 09:00). Leap and EvalSuite were told to re-arm.
    - Monday night is unchanged (apps closed from 23:00, runs from Tue 00:30).
  - **Sandbox98 private GitHub backup: yes** (scripts + results only, scanned for private strings).
  - NewIdeas: Kamen pressed Esc (~11:37); it runs again and was briefed on today's state.
  - TREE16 build split approved (Leap L1-L3, Breakthrough B1-B3, BigMoE M1-M4; spec by Mon 10:00; engine arm Thu
    night).
- Sun 15 11:5x: **EvalSuite prices re-certifying a new exact reference** (e48b106; estimate, nothing run).
  - Cost: ~4 h to certify (+1.6-3.3 h if IFX stage 2 runs) plus ~6 h to rebase the pinned runs, i.e. ~10 h of
    big-test time (one weekday window or two shared nights), plus ~4-5 h of EvalSuite's time.
  - A Tue-Wed switch redoes tonight's tier-0a (1.5 h). The close-call verdicts stand. Wed's CERTIFY98 capture is
    unaffected, but the capture and the Thu-Fri engine check must share ONE engine build.
  - CAUTION: a switch moves every same-scores comparator. REF2-T -> NAX6 moved HE+ by -3.33 pt and flipped a verdict.
    Quote every "vs exact" with its reference.
  - Orchestrator's recommendation to Kamen: yes to the bandwidth reference (~45 / ~62 ms vs ~55 / ~76), but NEXT week,
    after Saturday's certification on NAX6-N.
    - This week: today's reference, Leap's steps 1-2, and trees exact via route (b) (the ~1-day P·V port).
    - Tonight's tier-0a stays as booked.
- Sun 15 11:28: **MTP Head S9c: REAL t30b EAGLE-2 trees confirm it. A 16-node tree is a NET SAME-COST WIN on the
  125B, exact: +13.6 % (MBPP) / +6.0 % (HE+) kept per check vs chain-16** (BigMoE's real tree dump b198498, every
  anchor; pre-registered TrainLab 06d09a6 + 48b0e47, result fa3ce76; data/sb98/tree_real125.txt).
  - Real tree walk: tree-16 x1.136 [1.115, 1.157] / x1.060 [1.025, 1.093]. This is above the net line 1.052 (3 % over
    Leap's tree overhead +1.3 ms on a 62 ms check). HE+ is narrow.
  - tree-32 x1.371 / 1.531: it does not pay at R32 1.625; it pays at R32 1.3 (no 32-node overhead priced yet).
  - Throughput:
    - R32 1.625: tree-16 is the best everywhere, at x1.113-1.130 (MBPP) / x1.038-1.054 (HE+).
    - R32 1.3: tree-16 stays the best on MBPP; tree-32 is the best on HE+ (x1.177).
  - S9b's off-path proxy was accurate: real / proxy is 1.005-1.018, slightly pessimistic.
  - All checks pass: the anchors, the walk starts, the roots (bitwise vs TOPK), and chain-32 = 6.92 / 13.77.
- Sun 15 11:4x (orchestrator):
  - **Leap's revised exact-check estimate:**
    - under today's reference REF2-T: ~55 ms (16 rows) / ~76 (32);
    - with a re-certified bandwidth reference: ~45 / ~62. That means HC split-K on more partitions, per-row-key
      attention and parallel small projections; SD == AR stays bitwise.
    - Kamen is asked; EvalSuite is pricing the re-certification.
  - **BigMoE on cold PLE:** the exact-45 loop does NOT prefetch drafted tokens' n-gram rows, so cold 3-grams pay
    their stage in the critical path.
    - MTPLX's prefetch_rows_np already does it, bitwise by construction.
    - Approved, opt-in and default off, on a NEW snapshot so tonight's pinned runs stay byte-identical: (a) a
      hot-hit / miss receipt in runs.json; (b) R57_NGRAM_PREFETCH=1, as a night arm beside Breakthrough's
      fresh-token profile.
- Sun 15 11:2x (orchestrator): **CHECK55 progress on the 12-layer slice, both verified from commits:**
  - Leap C7 (4817ad2): the one-dispatch 32-row check = 1.430 x A16 (v32's split 1.574), PARTIAL by its line (<= 1.35).
  - Breakthrough pass S (2757be6): SB_FIRST + PLE_LATE widened to 17-32 rows, bitwise, -2.0 ms per 32-row check
    (~0.085 x A16). Together the slice's 32-row check is near ~1.35x A16 (from 1.63).
  - Lead (open): cold PLE n-gram staging costs +5-8 / +8-12 ms per check when the tokens' rows aren't in the page
    cache.
    - BigMoE was asked whether the real loop prefetches drafted tokens' n-gram rows.
    - Breakthrough adds a fresh-token arm to tonight's ~04:15 full-engine profile.
    - If the loop pays it, prefetch during drafting hides ~5-10 % of a check, exactly.
- Sun 15 11:1x: **Breakthrough CHECK55 step 0: the exact check is GPU-BOUND, not dispatch / CPU-bound** (Leap's C6 slice,
  12 layers, exact-45 32-row config, quiet; in-process Metal hook, no xctrace; ~/Breakthrough/docs/CHECK55_PROFILE.md, f425a3b).
  - Measured: GPU busy 94 % / 91 % of wall at 16 / 32 rows; CPU 0.35 / 0.33, hidden behind it. A16 23.37 ms, C32 38.16
    (x1.633). C6's "build 0.54 / 0.62" was the forward call's wall time, which includes waiting on the GPU.
  - Causal checks: +2 ms of CPU moves the wall +0.32 / +0.80 ms; reading the LM head once instead of twice moves it
    -1.55 ms (GPU -1.60).
  - The 17-row step, +14.8 ms:
    - experts' union +4.5 (bytes);
    - 2 x 16 double reads: projections +3.1, head +1.9;
    - second-chunk cores +2.8;
    - a late GPU start: ref2's SB_FIRST / PLE_LATE fire only at 5-16 rows (an exact fix, under test).
  - The dense side runs 2-10x its weight floor; the experts 1.23-1.27x. Engine-scaled, that predicts ~60 / ~100 ms
    vs the measured 61-64 / 99-102.
  - What changes in CHECK55:
    - step 1's value is GPU time (one weight pass for projections + head first);
    - step 2 is re-priced to ~-2 ms;
    - dense-side kernel efficiency at 16 rows becomes the big lever (hc_b + GDNB first);
    - piece 2 (MoE dense side) does not qualify (5-6 % of GPU).
- Sun 15 11:00 (orchestrator): S9b verified (tree_eagle125.txt: self-checks pass; EAGLE-2-16 x1.131 [1.109, 1.152] /
  x1.053 [1.018, 1.087]). The tree check is now the exact drafter-side lead.
  - Leap's tree-verify estimate (TREE16, route (b)) is ~+0.4-1.3 ms on ~62 ms, inside the 1.053x break-even.
  - EAGLE-2-32 x1.363 / x1.504 pays once Leap's step 1 brings R32 to ~1.3.
  - Next:
    - BigMoE: a head-only t30b TREE dump (real EAGLE-2 drafting, off-path nodes' top-8 + probs, ~32 nodes per start),
      to settle the off-path confidence;
    - then MTP Head re-prices;
    - build order with Leap after its step 1.
  - Kamen's answer on the invariant attention reference would make the tree's attention exact with no P·V port.
- Sun 15 10:59: **MTP Head S9b: an EAGLE-2-style 16-node tree is a SAME-COST WIN on the 125B, exact: x1.131 (MBPP) /
  x1.053 (HE+) kept vs chain-16** (CPU on BigMoE's t30b top-8 dump 9f49e03; pre-registered TrainLab 74314c3 + c2bfe6d,
  result 1ead8da; data/sb98/tree_eagle125.txt).
  - EAGLE-2-16: CIs [1.109, 1.152] / [1.018, 1.087]. Nodes whose parent is on the true path are exact; only the
    descendants of a wrong sibling are proxied.
    - Proxy band x1.056-1.221 / x1.017-1.107. The hard worst case (no-decay off-path chains) is x0.91 / 0.93.
    - So HE+'s win is marginal under the pessimistic proxy. The head's off-path confidence is not in any dump yet.
  - EAGLE-2-32: x1.363 / 1.504. It does not pay at R32 1.625; it pays at Leap's target 1.3.
    - At R32 1.3 the best is still EAGLE-2-16 on MBPP (x1.13), and EAGLE-2-32 on HE+ (x1.16).
  - Break-even for Leap: a 16-node tree verify may cost up to 1.131x (MBPP) / 1.053x (HE+) a 16-row chain's.
  - Delta 1 (a sibling counts only when it is y, a lower bound): x1.036 / 1.002.
  - Other policies at 16 rows, now correct: static branches x1.003 / 0.936; confidence-gated single-level x1.056 /
    0.958; optimal static = the chain. Ceilings: single-level oracle x1.41 / 1.20, multi-level x1.76 / 1.32.
  - Settles it: (a) real trees drafted with the head, off-path nodes included (an s30 build, or a head-only tree dump
    for t30b); (b) Leap's tree-verify price vs the break-even.
- Sun 15 11:0x: **Leap CHECK55 plan** (76817d3, ~/Leap/data/week98/check55_plan.md).
  - Central targets: exact 32-row ~60 ms, 16-row ~45 (~55 / ~41 only if every step lands at its good end).
  - The check is bound by dispatches and latency, not bytes: graph build 0.54 / 0.62 of a check, ~2,000 kernels.
    The step past 16 rows is the second 2 x 16 chunk of every op.
  - Steps:
    - 0: profile (Breakthrough owns it, briefed 10:5x);
    - 1: one launch per op at 17-32 rows, giving 32 rows ~75-85 ms (R32 ~1.2-1.35), first drop by Tue;
    - 2: dispatch diet, -5..-10;
    - 3: one-dispatch attention, -4..-6;
    - 4: gate round trips + commit, -2..-4;
    - 5: bytes at the floor, -2..-5.
    About two weeks in all.
  - **For Kamen today: the invariant (per-row key list) attention reference.** It changes today's exact bits and needs
    a one-time re-certification. In return it gives step 3's -4..-5 ms more simply and makes trees exact for free
    (saves ~1 day of the P·V port).
- Sun 15 10:5x: **KAMEN CONFIRMS: "let's chase exact drafter", and the CHECK gets equal top priority with the drafter**
  ("from everything I've heard the check unlocks like 3x the tokens").
  - Honest size given to him: the exact 32-row check 99-102 -> ~55 ms is ~1.8x; ~3x+ is the lossy check (~21-25 ms).
  - Leap was asked for its ordered plan to ~55 ms (32 rows) / ~40 ms (16 rows) with ETAs, and whether Breakthrough
    (idle, fresh week) should own a non-colliding piece.
  - **Leap TREE16 price** (desk estimate, 91cc592, data/week98/tree_verify_price.md): a 16-node tree check costs about
    chain-16 except exact attention.
    - GDN: depth-first walk with state saved at forks, +0.1-0.5 ms, exact by construction.
    - Attention: a plain tree mask is NOT bitwise for side nodes. Exact routes:
      - (b) port P·V bitwise, +0.2-0.5 ms, ~1 day;
      - (a) no new kernel, +3-6 ms at 1-2K context;
      - (c) a new AR-equal reference: changes today's reference bits, Kamen's call.
    - Net vs S9's EAGLE-2 estimate: ~+15 % (MBPP) / +3-4 % (HE+) kept with (b). Build ~1.5-2 days, all exact.
    - Decision after MTP Head prices EAGLE-2 on the 125B from BigMoE's t30b top-3 dump; the check plan comes first.
- Sun 15 10:4x: **Sandbox98 QAT follow-up FLOOR (lr floor 30 %) = KILL; the three follow-ups are done, none passes
  both sets** (802d7d4; pre-registered d7e11e2).
  - FLOOR, kept/32 vs the 8-bit body, MBPP-50 / HE-30: -3.05 / -8.15 % (list -2.80 / -8.05 %). Against plain QAT:
    MBPP -0.011 kept, HE -0.370 (CI -0.605..-0.195). The halves agree (-5.04 / -7.02 %). The low end of the schedule
    matters: ending at 3e-6 instead of 1e-6 costs ~6 % on HE.
  - The three: MIX (44 MB) +1.42 / -1.79 %; LSQ (38 MB) -0.90 / -2.25 %; FLOOR KILL; plain QAT -2.80 / -2.25 %.
    MBPP-50 can be brought inside the 1 % line (MIX, LSQ); HE-30 does not move. Averaged over both halves, MIX sits
    within ~0.3 % of the bar on both sets.
  - Exact road: a 4-bit body changes only the drafter's guesses (kept/32); the check keeps every token exact.
  - Untested: MIX + LSQ together; a lower floor or a cool-down. Sandbox98 is idle.
- Sun 15 10:4x: **KAMEN'S DIRECTION: the drafter's 98 % stays EXACT** (his words: "that's the purpose of the drafter";
  lossy close-call only if it were clearly the better use of the week). Orchestrator's plan, given to him:
  - One guess at 98 % right after a miss is blocked: the 125B is near-tied there. Exact 98 % comes back as CANDIDATES.
    ~~At check starts the head's top-2 holds the 125B's word 96.36 % (MBPP) / 100 % (HE+ 24)~~ **CORRECTED 11:0x**
    (BigMoE TOPK 9f49e03; re-counted by the orchestrator from topk_t30b_*.npz at the depth-0 walk starts, n 461 / 2,253):
    - The HE+ 100 % was an artifact: rank = -1 on all wrong cells, counted as top-2 hits. The MBPP 96.36 % used the
      draft's rank in the 125B's list.
    - The head's own top-1 / 2 / 4 / 8: HE+ 91.97 / 96.1-96.5 / 98.48 / 99.35 %; MBPP 89.70 / 96.3-96.4 / 99.0 /
      99.69 %.
    - So a #2 branch recovers ~57-65 % of check-start misses and a top-4 fan ~81-91 %.
    - The head's p1 separates them: p1 >= 0.95 (61-70 % of starts) is right 99.5-100 %; p1 < 0.5 (~8 %) only 40-44 %.
      The confidence gate has real signal.
  - So the exact route for the week is TREES + CHEAPER ROWS, both exact:
    - verify the head's #2 / #3 at its unsure spots in the same check (MTP Head's S9 lead: EAGLE-2-style 16-node tree
      ~+17 % / ~+5 %; a perfect tree ~x2 at 32 rows);
    - make the extra rows cheap (Goal B: R32 1.625 -> ~1.3, Leap).
  - Asked:
    - BigMoE: t30b's top-3 tokens + probabilities along the walk, head-only recompute if possible; else tonight's
      45 GB block.
    - MTP Head: price EAGLE-2 on the 125B at R32 1.625 and ~1.3.
    - Leap: the tree verify cost on the hybrid GDN model.
  - Lossy close-call stays the side track: Monday night's gate (delta 1 + delta 2) is already registered.
- Sun 15 10:26: **MTP Head S9 TREE-DYN: no tree makes the 32-row check pay; the one lead is a 16-row EAGLE-2-style
  tree at ~+17 % (MBPP) / ~+5 % (HE+) kept, exact, estimated** (CPU on the 125B cells + s30; pre-registered TrainLab
  4dd7de9, result d0aafb2; data/sb98/tree_price.txt).
  - **CORRECTION (Sun 15 10:59): this bullet's 125B numbers are VOID.** The cells' 'rank' field is the draft's rank
    in the 125B's top-32, not y's rank in the head's list (BigMoE). NewIdeas' roottree read it the same way. S9b
    replaces these rows with BigMoE's top-8 dump; the s30 rows stand.
  - Below, kept/check is vs chain-16, as MBPP / HE+. The 125B cells hold where the true token sits in the head's list,
    but not the head's confidence. So on the 125B only no-confidence trees and oracles are priced.
  - 125B realizable:
    - Static branches: 16 rows x1.001 / 0.933; 32 rows x1.067 / 1.087.
    - The optimal static tree (Sequoia-style) never branches at 16 rows (x1.000); x1.05 / 1.20 at 32.
    - Delta 1 is similar.
  - 125B per-check oracle (single-level ceiling): 16 rows x1.40 / 1.18, so only the oracle passes the same-cost line.
    32 rows x1.54 / 1.69, which fails 1.625 on MBPP: the 32-row line closes for single-level trees.
    - Reference: chain-32 / chain-16 is only x1.05 / 1.30.
  - Added after the run:
    - The multi-level ceiling (a perfect #2 / #3 tree, paying the wrong siblings' rows) is x1.60 / 1.22 at 16 rows and
      x2.08 / 1.92 at 32.
    - An EAGLE-2-style tree on s30 (best-first by the head's path probability; off-path nodes proxied) gives x1.20 /
      1.14 at 16 rows. It captures 24-40 % of that ceiling's gain. Carried to the 125B: ~x1.17 / ~x1.05 at 16 rows,
      ~x1.43 / ~x1.31 at 32 (exact).
  - Next, if wanted:
    - The 125B head's own top-3 probabilities along the walk (a GPU dump on BigMoE's harness) would price EAGLE-2-16
      directly.
    - Leap's price for a 16-node tree verify on the hybrid model (GDN state forked per branch, tree mask) decides
      whether the +17 % is real at 61-64 ms.
- Sun 15 10:1x: **MoonShots J(CC): close-call acceptance end to end on the 2B + s30 (HE+ 164 + MBPP 100, paired):
  delta 1 HOLDS (+0.76 pt, p 0.88; numerics control +0.76), delta 2 FAILS (-6.06; HE+ -9.15, p 0.020).**
  - delta 2's loss is runaway answers (cap hits 26 -> 50) and docstring rewrites steering the code.
  - Real kept/check: +21 % at delta 1 (= the 125B's teacher-forced) and +69 % at delta 2. Without capped answers,
    delta 1 gives +9 % (HE+) / +19 % (MBPP).
  - At the 125B's delta-2 code dose the 2B holds on both sets, so delta 2 on the 125B is plausible but needs the gate,
    with cap-hit and loop rows (LEVERS_A_MOONSHOTS.md J(CC); 12f76d1).
  - Orchestrator, 10:3x: verified and sent to EvalSuite, with three suggestions:
    - cap-hit / loop rows and kept on uncapped answers for Monday's delta-1 arm;
    - a delta-2 arm when a 45 GB slot has room;
    - optionally, a comment / docstring guard.
- Sun 15 10:2x (orchestrator): S8 / S8b verified (rerank_gate.txt, margin_RESULT.txt).
  - Next for MTP Head: **TREE-DYN pricing** (CPU, the 125B cells + s30).
    - The best version of NewIdeas' root-only ROOT TREE: the head's #2 with its own sub-chain at the least-confident
      positions, under a fixed 16- or 32-row budget, exact and delta 1.
    - Line: tree-16 beats chain-16 (same cost), or tree-32 / chain-16 kept > 1.625 (R32).
  - Caveat: the hybrid GDN trunk needs its recurrent state forked per branch path; Leap prices that if it passes.
- Sun 15 10:1x: **Sandbox98 QAT follow-up LSQ (learned group scales) = BETWEEN** (a1c15c3; pre-registered d7e11e2):
  every Linear 4-bit g32 with each group's scale and offset learned; body 38 MB, plain QAT's size.
  - kept/32 vs the 8-bit body, MBPP-50 / HE-30: -0.90 / -2.25 % (list -0.81 / -1.84 %); plain QAT g32 -2.80 / -2.25 %.
    MBPP passes the 1 % line at no extra bytes (+0.083 kept over plain QAT, CI +0.017..+0.151); HE-30 does not move
    (+0.000). Second halves: MBPP -1.20, HE +0.20 %; averaged over both halves LSQ sits ~1 % under the bar.
  - So far both levers lift MBPP (MIX more, for +6 MB) and neither moves HE-30. FLOOR (QAT lr floor 30 %) runs next,
    ~1.5-2 h at this GPU share.
- Sun 15 10:08: **MTP Head S8 + S8b: the head's #2 holds the fix, but training cannot move it to #1; the
  train-for-the-close-call-rule family closes on the sandbox** (pre-registered TrainLab 4418e5b / f88e870; results
  5fa33f6 / 32689e5).
  - S8 gate (s30, read-out): at the delta-1 walk's misses, s30's #2 is IN (the true token or in the delta-1 set) at
    52 %, and #2 or #3 at 70 %.
    - Oracle kept/32 under delta 1: top-2 +57 %, top-3 +97..100 %. The exact top-2 oracle alone gives +55 %
      (MBPP 4.32 -> 6.68).
  - S8b: an argmax-margin loss (KL + relu(1 - (best IN logit - best OUT logit))), 400 steps. NO EFFECT on both arms:
    - Without close-call survival, vs n1: delta-1 walk MBPP -0.010 (p 0.65), HE -0.033.
    - With survival, vs S7's cs1: -0.073 (p 0.135), HE -0.124.
  - With S7's set-mass loss, both losses aimed at the rule are now NO EFFECT; the plain KL already gets what training
    can. The +55-57 % top-2 headroom needs more candidates checked per verify (trees or extra rows), or a ranker that
    reads information the head lacks.
  - Not tested: other loss weights, the 125B's head.
- Sun 15 09:0x (orchestrator): S7 ACCSET verified (TrainLab data/sb98/accset_RESULT.txt; confirm NO EFFECT, MBPP +0.105
  kept/32, p 0.77).
  - Next on the lossy road: **MoonShots J(CC)**, a 2B end-to-end quality pre-screen of close-call. pass@1 on
    HumanEval+ / MBPP tests, exact vs delta 1 vs delta 2, paired, plus changed tokens per 1K by span type.
  - If delta 2 holds on the 2B (a harsh proxy: more near-ties than the 125B), EvalSuite can add a delta 2 arm to
    Monday's gate. delta 2 is worth +32 % / +46 % kept vs +22 / +23 % at delta 1.
- Sun 15 09:00: **MTP Head S7 ACCSET = KILL at the confirm: training the head for the close-call accept set adds
  nothing measurable** (Sandbox98; pre-registered TrainLab af0ffee, result 153899d; data/sb98/accset_RESULT.txt).
  - Lever: per depth, KL + 1 x (-log of the head's mass in the target's delta set), optionally keeping rows alive while
    the head's #1 is in the set.
  - Screen (400 steps vs n1, kept/32 on the delta-1 walk):
    - cs1 (delta 1 + close-call survival): PROMISING, MBPP +0.115 (p 0.023). But its exact kept/32 rose +0.075 too.
    - cs1k (the loss alone): NO EFFECT, MBPP +0.017.
    - cs2 (delta 2): NO EFFECT, MBPP +0.110 (p 0.060).
  - Confirm (cs1 at s30's recipe, 1,889 steps) vs s30, delta-1 walk: MBPP 5.243 -> 5.348 (+2.0 %, p 0.77, CI incl. 0);
    HE +0.6 %; exact +0.073 / -0.039. Changed tokens per 1K: 52.4 -> 54.3 / 26.8 -> 28.3.
  - So the lossy gain belongs to the rule itself, and the exact-trained head is the right one to measure it with. On the
    sandbox the rule at delta 1 lifts s30 by +21 % (MBPP) / +15 % (HE), vs Breakthrough's +23 % / +22 % on the 125B.
  - Not tested: LAM other than 1, an argmax-margin loss, the 125B's head.
- Sun 15 08:2x: **Sandbox98 QAT follow-up MIX (4/8-bit by sensitivity) = BETWEEN** (c5b1ea1; pre-registered d7e11e2):
  fc + k_proj + o_proj at 8-bit, the rest QAT 4-bit g32; body 44 MB (~54 MB per step with the 8K context list).
  - kept/32 vs the 8-bit body, MBPP-50 / HE-30: +1.42 / -1.79 % (list +1.39 / -1.92 %); plain QAT g32 -2.80 / -2.25 %.
    MBPP passes, HE-30 misses the 1 % line by 0.8-0.9. The second halves flip (MBPP -0.90, HE +1.74 %): averaged over
    both halves MIX sits within ~0.3 % of the bar on both sets.
  - Sensitivity pass on s30: o_proj alone at 4-bit costs -2.77 % (7 % of the weights), down_proj -1.50, fc -0.97,
    k_proj -0.91, the rest -0.2..-0.8.
  - Next: LSQ (learned group scales, 38 MB) is running, then FLOOR.
- Sun 15 08:1x (orchestrator): S6 CSM verified from its files. In flight:
  - **MTP Head: ACCEPT-SET training** (its own R-61N follow-up from 11-01, never run; non-exact, FAST track).
    - The head is trained so its #1 lands in the close-call accept set at delta 1 (2 optional).
    - Line: kept/32 under the close-call walk vs s30 under that walk, with the changed-token rate beside it.
    - Why: ~90 % of check-start misses sit on the 125B's close calls (CSM), and the delta sweep above used an
      exact-trained drafter.
  - **Sandbox98: three single-arm QAT screens**: mixed 4/8-bit, learned group scales, higher lr floor.
  - Not now: MTP Head's depth-0-only extra step. Closing the whole ~3 pt residual at depth 0 gives ~+3 % kept against
    ~2 % cost, so the net is <= ~+1 %.
- Sun 15 08:02: **MTP Head S6 CSM: the check-start dip is mostly close calls** (CPU read-out of t30b's cells, no
  training; pre-registered TrainLab 968ccfc, result 67649e7; data/w98/csm_*.txt). Sets: HE+ 24, and BigMoE's MBPP bank
  100 (read-only).
  - Depth 0 at the walk's check starts vs other anchors: HE+ 91.97 vs 98.40 %, MBPP 89.70 vs 94.69 %.
  - Matching on the 125B's top-1 margin explains 54 % of the gap on HE+ (CI 38-76) and 40 % on MBPP (CI 28-54):
    BETWEEN by the rule.
    - Check starts land on close calls (m < 2 nats) more often than other anchors: 16.9 vs 5.4 % (HE+), 23.8 vs 16.9 %
      (MBPP).
    - The rest is a ~3 pt residual at equal margin, only at m < 4; at m >= 4, starts are within 1.1 pt of other anchors.
  - Exploratory cuts:
    - Of the misses at check starts, 92 % (HE+) and 88 % (MBPP) fall on rows where the 125B's margin is under 2 nats.
    - There is no dip after a full accept (99-100 %); the dip is all at after-miss starts (89.2-89.3 %).
    - The residual is largest (4.7-6.5 pt) when the missed row was wrong even from a fresh anchor: hard rows run in a row,
      so this is general drafter accuracy, not missing information at check starts.
  - Read: lossy close-call acceptance targets ~90 % of the check-start misses directly. Exact options stay as worded.
    The depth-0-only extra-step screen is ranked below the queue and was not run.
- Sun 15 07:2x: **Sandbox98 QAT 4-bit chain head at FULL budget, group 32 = BETWEEN** (7b14cd8; pre-registered a298c0d):
  from the native head over s30's 1,889 steps, ctl + qat matched arms (the ctl reproduces s30 exactly).
  - kept/32 vs the 8-bit body with the same list, MBPP-50 / HE-30: QAT 4-bit g32 -2.80 / -2.25 % (8K context list
    -2.35 / -1.77 %); post-training 4-bit g32 -4.15 / -6.59 %, g64 -8.28 / -10.88 %. QAT recovers 32 % / 66 % of the g32
    loss; the held-out second halves agree (-2.45 / -2.36 %). The line (within 1 % in all four cells) is missed by 0.8-1.8.
  - The ~48 MB form (38 MB body + 8K context list) keeps 4.143 / 6.076 vs the 8-bit full head's 4.350 / 6.269
    (-4.8 / -3.1 %); the 8-bit body + list (~74 MB) keeps 4.242 / 6.185 (-2.5 / -1.3 %).
  - At the halfway stop line the 4-bit head read level with the 8-bit body (MBPP -0.03 %, CI +-1.5); -2.8 % by the end,
    so the gap opens in the anneal, or the halfway read was lucky.
  - Open, each <= 20 min single-arm vs s30: mixed 4/8-bit by sensitivity, learned group scales, a higher QAT lr floor.
- Sun 15 06:3x (orchestrator): **WALKW = NO EFFECT** (MTP Head, verified from TrainLab data/sb98/walkw_compare.txt).
  - Check-start weighting (ww): kept/32 +0.019 MBPP-50 (p 0.31) / +0.069 HE-30 (p 0.09).
  - Margin weighting (mw): -0.084 / -0.100.
  - Depth 0 at check starts doesn't move: 84.15 -> 84.42 % MBPP, 87.26 -> 86.74 % HE.
  - **Arithmetic for the ranking (orchestrator):**
    - Every cheap source of the hard first guess is now closed: state refresh (LAYERS: the state lives near L45,
      ~a verify), the check's own rows (K-next), caches (L), 2B / 4B / audit (~87-89 %), training weight (WALKW),
      root trees at a fixed 32 (+4.5 % max).
    - The distillation route (1c) can't close it this week. The 2B agrees with the 125B on 57.2 % of MBPP's unsure rows
      (1c control); 98 % needs ~93 %; 1c's peak was +0.8 pt at ~1 epoch. At ~+1 pt per doubling of data that is out of
      reach, so no data-scaling probe was run.
    - So exact at check starts tops out near the margin ceilings (HE+ ~16-18 kept/32), and the 98 % / ~23.8 kept target
      runs through lossy close-call (gated). NewIdeas to confirm or overturn in its ranking. It is stuck on the
      usage-limit panel; Kamen can close it in the morning.
  - Mac idle at 06:3x with the lid open, so Sandbox98 was asked to start its planned full-budget QAT (group 32).
- Sun 15 06:0x (orchestrator): check-in, quiet.
  - Machine: no job running; swap 8.10 GB (flat since 04:05); disk 118 GiB; fans idle at 2,355 rpm, cap 5900.
  - Usage: sub2 open again after 06:00. NewIdeas woken for its 06:00 read and the Goal A re-rank (ranking of the open
    builds + owners + Sun-night 45 GB slots, to the top of this file).
  - S4 addendum (read from TrainLab data/sb98/mhd1_s380_compare.txt):
    - kept/32 vs the 1-layer control: c2 -0.244 MBPP / -0.404 HE; c2h -0.500 / -0.695. Both lose at depth 0 too.
    - Every init was tried: zero start (deep2 NO EFFECT), tied (tie2 HURTS), untied copy (c2 / c2h KILL).
    - On the 125B a full extra layer costs +56 % per drafted token (~40 ms per 32-block), so it would need kept/32 ~1.6x
      to pay. Closed for FULL layers; a small cache-resident block (<= ~12 MB, KNEE) stays open.
  - MTP Head next (06:1x, no objection from me): **WALK-WEIGHTED TTT**, "train where you draft". Training anchors at
    the walk's check starts get 3x KL weight. Sandbox, 400 steps from native, vs the deterministic control; line on
    kept/32 + depth 0 at check starts; pre-registered.
    - Its bound: BigMoE's 84-89 % true-state depth 0 after a miss. It can only win where the drafter is
      capacity-limited at close calls.
    - Optional arm: weight by the 125B's margin (< 1 nat at 3x) instead of walk position.
- Sun 15 04:10: **Breakthrough: AWQ-style 3-bit GDN + attention = CLOSED for scaling-only quantizers** (pre-registered
  544a1f1; 43ef47c; 12 code windows): activation-aware 3-bit code flips 3.86 / 2.99 % (all-row / non-tie) vs plain RTN
  3-bit 4.43 / 3.62 and the lines 1.34 / 0.87; scaling cuts RTN's flips only 13-17 %. The alpha-0 control reproduces
  RTN on 100 % of rows. Next for the dense wall: full-Hessian GPTQ or a trellis code. Until then GDN / attention stay at
  4 bits and the code plan stays at 5.86 GB (20.9 ms floor / 24.9 at 84 %).
- Sun 15 04:0x: **Sandbox98 QAT 4-bit chain head = BETWEEN** (0a83afe): kept/32 vs the 8-bit body at matched steps (200
  steps x depth 30 from s30o): MBPP post-training 4-bit -6.3 %, QAT -2.6 % (CI -4.5..-0.9), QAT + 8K context list -2.1 %;
  HE -10.1 / -5.8 / -5.3 %. QAT recovers 59 % / 43 % of the 4-bit loss (PASS needs within 1 % on both). The QAT weights
  read unquantized already sit -1.7 / -3.4 % below control, so the short run under-trains. Today's ~44 MB form (4-bit QAT
  body + 8K context list) keeps 4.031 / 5.482 vs the 8-bit full head's 4.213 / 5.886 (-4.3 / -6.9 %). For the day: QAT
  over a full budget (1,000+ steps, or 1,889 from native) and group size 32 (body ~38 MB).
  - Sandbox98's night: deep63 (for Leap); 1c KILL; CHAIN SIZE no; S5 / S5b VOID, S5c NO EFFECT; prompt-aware list
    PROMISING; QAT BETWEEN. Resting.
- Sun 15 04:05: check-in. account-1 (sub1) ended its week at 96 % at 03:59 (reset 04:00); sub2 11 % weekly, its 5-hour
  window full until ~06:00; main 26 %. Swap 8.15 GB (+1.1 since 01:09, under the +2 GB stop rule); disk 118 GiB; fans
  5900, die 62 C. No research job running; NO-SWAP 40.9 GB. MTP Head's "mhd1_train, MEM 9 GB" lock (taken 03:30, until
  05:00) has no process behind it after S4's KILL; its owner is blocked until ~06:00, and it lapses at 05:00.
- Sun 15 04:0x: **MoonShots' K-next lead FAILS its line** (BigMoE ran the dump; bank path, 2,433 branches, peak 28.6
  GB, 30 min): the 125B's verify row past a miss (computed on the wrong draft) predicts the next depth-0 target only
  16.6 % of the time on MBPP; where t30b is wrong (M1) 6.6 % (n 228; line >= 50 %); HE+ 2.9 %. The wrong token derails
  the next prediction; only a fresh pass on the bonus token would carry the 125B's information. With BigMoE's walk
  analysis this closes the cheap 125B-side route at the first guess.
  - MoonShots' final read (396d9bd): where the row past the miss disagrees with t30b's fresh guess it is right 0.9 % /
    0.4 %; even confident targets reach only 19 % (having seen the wrong token, the 125B continues that token). The
    drafter's own next token after its wrong one (d1) is NOT LIVE either (right 76-79 % overall but only 20.5 / 17.1 %
    where the fresh guess misses). One token later the 125B recovers (59 % MBPP / 70 % HE+). **The hard first guess's
    information is in no row the check already computes**: only a 125B pass on the bonus token carries it (the next
    check itself, or a partial trunk pass at ~0.6 ms per layer + head training). MoonShots' cheap-source search is closed.
- Sun 15 04:0x: **MTP Head (auto) S4 DEPTHC = KILL at its stop line** (step 380, MBPP-50 23-30 vs n1): c2 -2.52 pt
  (p 0.0007), c2h -7.91 pt (p 0.0000). Depth done fairly (copied init, own lr) hurts at matched steps on the sandbox.
- **State of GOAL A after the night (orchestrator):** the wall is the first guesses at hard spots after a miss, where
  the 125B itself is unsure (selection, not drafter state). Cheap routes closed tonight: learned chooser (bounded),
  the 4B auditor (+1 pt), 1c short form (overfit), S5 injection (no effect), DFlash2-TF (KILL), allrows (no kept
  effect), carry adapter (no effect), fair depth (KILL), correction cache (KILL), the verify row past a miss (FAIL),
  kNN (KILL). Partial gains: the 2B audit (+5-12 % kept under the walk), lossy close-call at 1 nat (+22 % kept, to the
  gate). Open, planned-day builds: a fresh pass on the bonus token, deep KV injection, full-data distillation.
- Sun 15 03:4x: **Sandbox98: a PROMPT-AWARE lm_head list removes most of the static list's loss** (69c93b2; s30,
  eval-only, kept/32 vs the full 248K vocab, MBPP-50 / HE-30): static 8K -8.5 / -18.1 %; 8K + the prompt's tokens -4.4 /
  -2.3 %; **8K + prompt + answer tokens so far -2.4 / -1.1 %** (depth 0 -0.89 / -0.38); 16K + context -1.5 / -1.2 %; 4K +
  context -3.5 / -3.3 %. The static list's loss was mostly the prompt's own identifiers. 8K + context = ~8.4K rows = ~9.7
  MB per step at 4-bit, gathered once per check.
  - The chain head's lm_head goes from 286 MB to ~10 MB per step for <= 2.5 % kept. The body is the last blocker (34 MB
    at 4-bit, where post-training 4-bit costs 6-11 % kept; 64.6 MB at 8-bit, free): a 4-bit QAT screen (~15 min) decides
    the cache-resident chain (~44 MB per step).
  - The same trick applies to Leap's P2 screened head and Breakthrough's 16K code shortlist in the lossy check (a context
    list could also serve chat, which a static code list can't).
  - Leap's pricing (fd1cc81): at ~44 MB per step the bytes cost ~0.09 ms, so launches dominate. In today's ~40-kernel
    MTP-step form (~0.54 ms fixed) 30 steps still cost ~15-17 ms; fused to ~5 dispatches with the token on the GPU it is
    ~0.15 ms per step = ~4.5 ms per 30 steps, inside the 400-exact budget. **The cheap chain needs both: <= ~48 MB per step
    AND a fused step** (Leap takes the kernel job once a head design holds its accuracy).
- Sun 15 03:3x: **Sandbox98 S5 on the REAL pair = NO EFFECT; family (c) doesn't pay in screen-testable forms** (8099ff3):
  the 2B (1c's LoRA form) + the 125B's own final states from the BANK (wd, 10240-d) at prefix rows u <= s; 360 files,
  1,792 steps. S5b (Flamingo gate) VOID (the gate stayed at 0.0003); S5c fixed it (zero-init outputs, adapter Adam 2e-5
  + clip; both injections engaged). MBPP kept run per 30-row block: none 7.95; cross-attention 8.13 (+0.18, CI
  -0.10..+0.48); state at the anchor 7.73 (-0.22). Agreement MBPP +0.37 (p 0.011) / -0.69; HE+ +0.01 / -0.32 (line +3).
  Not even lag 1 gains, though the 125B's own head turns that state + the token into 94.0 % on MBPP: an add-on after the
  2B's last layer can't compose it with the current token.
  - Left with a theory case: a deep KV injection into several of the 2B's attention layers (needs a GDN backward + ~10x
    the 125B's answers; 1c / S5b / S5c all overfit the 360 files in 1-2 epochs): a planned-day build.
  - Reference: the plain 2B vs the 125B teacher-forced, kept run per 30-row block HE+ 19.86 / MBPP 8.29 / val 9.95.
- Sun 15 03:2x: **BigMoE: positions right after a miss are INTRINSICALLY hard (selection, not drafter state)** (t30b's
  HE+ cells with the TRUE trunk state at every anchor, cells_t30b_heplus24, b316365): depth-0 accuracy after a depth-0
  miss 83.97 % (n 131) vs after a hit 98.23 % (n 6,220), overall 97.95. The 125B itself is far less sure there: median
  margin 3.87 vs 11.87 nats; near-ties (m < 0.5) 12.2 % vs 1.6 %. A miss marks a surprising place, where the next token
  is less predictable for the 125B too, so a fresh-state refresh can't close the check-start gap. For MoonShots' dump:
  the post-miss verify rows carry the 125B's information exactly there (good), but the ceiling is lower (3x thinner
  margins): log the margin per row. Extension offered: check starts after a miss at any depth + MBPP (~4 min at 10 GB).
  - **Extension (data/deepdraft/ceiling/checkstart_*.json): at the walk's check starts after a miss at ANY depth, with a
    FRESH state, depth 0 is ~89 % on both sets vs 99-100 % after a full accept.** HE+ 24: after a miss 89.29 % (n 336;
    125B margin median 6.44, near-ties 6.0 %), after a full 31-draft accept 99.01 % (n 101; margin 12.31), all anchors
    97.95. MBPP 100: after a miss 89.16 % (n 2,140; margin 6.00, near-ties 7.0 %), after a full accept 100 % (n 13), all
    94.01. Matches the 91.6 / 89.5 % at check starts: the wall is the positions' hardness (selection), not drafter state.
  - MoonShots' K-next dump: miss lists ready (326 HE+ / 2,107 MBPP with d_k recoverable); BigMoE builds and runs it
    around the 04:20 review if the slot allows (~25 min at ~30 GB).
- Sun 15 03:25: **BigMoE: the overflow guard is built and tested** (0231f64 / 2a91737: BIGMOE_OVERFLOW_FLAG=1 turns an
  over-pool gate call into slot 0 + a counter, plus a force hook; 9 / 9 16 + 16 redos bitwise). **Close-call acceptance
  is ready to gate:** R57_RELAX_DELTA=1, scoped to the code sets, with its own "close_call" counter (smoke: HumanEval
  accepted 2, IFEval absent), on snapshot cc1 with its lossy-45 base env (6bb40d6); EvalSuite has it. C32m (mr2) dropped
  (-0.002 x A16). Left for a 32-row loop: the redo inside the decode loop + the GATE_N / MAXK pins.
- Sun 15 03:3x: **MoonShots K + L on the hard-first-guess wall** (7a1735f; its walk scorer reproduces t30b exactly):
  - **K, the audit on the decode walk (P = 1): kept/32 MBPP 6.879 -> 2B 7.459 (+8.4 %) / 4B 7.702 (+12.0 %); HE+ 13.769 ->
    14.456 (+5.0 %) / 15.399 (+11.8 %)** (P = 2: 2B 7.739 / 15.00; 4B 8.172 / 15.82). But HE+ walk-strict depth 0 moves
    only 91.58 -> 92.74 / 93.00 (line 95.0): NOT LIVE. The audit's kept gain comes from deeper guesses.
  - **The hard first guess after a miss is hard for every cheap model** (HE+ depth 0 at check starts: head 91.6, 2B ~87,
    4B ~89): the information is specific to the 125B, right after it overrode the drafter.
  - L, a correction cache within the answer ((previous token, draft) -> the 125B's token): KILL (broke 893 / 893 and
    82 / 82 hits; fixed 3 / 274 and 2 / 42 check-start misses; kept/32 fell to 5.34 / 12.50).
  - What's left is 125B-side: (1) the check's own rows past the miss (free); (2) a 2-way tree at depth 0 after a miss
    (+1 row per check); (3) a partial trunk pass on the bonus token (~0.6 ms per layer + head training). **Cheapest test:
    a verify dump saving the 125B's argmax at the row after the miss (computed on the wrong token), counting how often
    it equals the token after the bonus.** If high, depth 0 after a miss gets a free, 125B-informed candidate. Needs
    BigMoE's verify path.
- Sun 15 02:41: **Fit12's S1 DF-TF ran (re-armed with an automated head test; rc 0, peak 10.5 GB; read by the
  orchestrator since Fit12 and NewIdeas are blocked): family (a), DFlash2 teacher-forced, is NOT a corrector base.**
  Self-check j=1 == native 626 / 626. DFlash2 l120_ft (the 12 GB engine's drafter), block 32, offline on the BANK.
  - STRICT conditional (every anchor), HE+ (186 anchors): 0 97.31 / 1-2 94.97 / 3-6 94.91 / 7-10 89.01 / 11-14 88.26 /
    15-22 83.54 / **23-30 88.14 (n 59)**; MBPP (440): 91.36 / 89.78 / 81.77 / 73.15 / 72.97 / 78.26 (n 46) / 23-30 no
    survivors.
  - TF kept/32 11.76 (HE+) / 6.25 (MBPP); native (masked) kept/32 11.46 / 6.62.
  - Line (strict): PASS needs HE+ 23-30 >= 95 and MBPP 23-30 >= 85; KILL < 90. The script printed NO DATA (MBPP had no
    23-30 survivors); **by the HE+ line, 88.14 < 90 = KILL.** Even teacher-forced, DFlash2's strict accuracy falls to
    ~88-89 % by depths 7-14 on HE+ and ~73 % on MBPP: far from a 98 % corrector.
- Sun 15 03:1x: **45 GB SLOTS for the week (orchestrator's draft; each needs ~47 GB NO-SWAP, i.e. Kamen closing apps):**
> (a quote from the project owner, kept private)
  - **23:3x, what actually ran:**
    - The app quit at 23:30 worked: 9 apps; PyCharm vetoed and was TERMed after 90 s.
    - But NO-SWAP was only 34.4 GB: 52 claude processes held ~14.6 GB. Closing the 5 idle Fable thinkers (orchestrator's
      own) brought it to 37.6 GB.
    - Leap TERMed the a45 runner (it held nothing). C7c started first at 23:35:42 (gate 26 GB). a45 is SKIPPED tonight:
      it needs a night with 47 GB free.
    - BigMoE M1 + ng1 follow on C7c's end (gate 36 GB). EvalSuite's chain needs ~need + 4 GB; big runs push idle pages
      into the compressor, so it may open later in the night.
    - Kamen's idle chats hold only ~4 GB resident, so they aren't worth closing for this.
    - 23:4x: NO-SWAP reached 46.5-47 GB after C7c + BigMoE (the compressor effect). EvalSuite started 23:41:03.
    - a45 is re-armed LAST (orchestrator's yes to Leap's offer):
      - gate: Breakthrough's lever-2 follower 88412 ends, plus 47 GB on two reads, its own quiet ticket;
      - every run ENDS by 06:45 (no start after ~06:00);
      - otherwise it skips to a school-day slot.
      - Armed 23:48: pid 29218 (a45_last.sh 06:00 / 06:45), marker A45_1116_DONE. Verified.
    - **00:2x Mon: EvalSuite's chain STALLED on memory.** Its tier-0a process has waited since ~00:01 for 49 GB NO-SWAP
      (need 45 + 4) at 40-45 GB.
      - The orchestrator wrote Breakthrough's manual GO (00:23). The engine profile (39 GB gate) took its quiet ticket
        at 00:24:32; then the lever-2 bench, then a45.
      - EvalSuite keeps waiting (cutoffs 08:00 / 09:30 / 14:00). The compressor effect may let it fit after.
      - If the close-call base can't run tonight, Monday's close-call arms slip (side track).
    - **00:2x-00:3x:**
      - Breakthrough's engine profile ABORTED 20 s in (rc 134, no data). Its warm-up forward on 32 prompt tokens needed
        179 experts at layer 0, over the tier's C 175, and BigMoE's ext aborts on that; the resident slice couldn't
        catch it.
        - Fix: warm on a span that fits the window's gate_req, plus BIGMOE_OVERFLOW_FLAG=1.
        - It re-runs tonight after its head test + stub sim (it fits at ~42.7 GB vs its 39 GB gate). Cutoffs: start
          by 06:00, end by 06:45. Else Monday by day while Kamen is away.
      - **CHECK55 LEVER 2 KILLED** by its own pre-registered quiet line: dense_pre is SLOWER than K-73's chain, +30.3
        us / layer at 16 rows (line <= -50), +9.8 at 32. Bits held. The pieces only reach parity.
        - E14's A16m / C32onem arms are to be dropped (Breakthrough's advice, the orchestrator agrees; Leap decides).
      - a45's runner (8663) waits on 47 GB and EvalSuite on 49, at ~42.7 GB NO-SWAP.
    - **00:42 Breakthrough's FULL-ENGINE CHECK55 profile DONE** (fixed d9be614; rc 0, peak 30.8 GB, bitwise, no check
      flagged):
      - GPU-bound on the engine: busy 97.8 % at A16 (59.75 of 61.11 ms), 96.6 % at C32 (98.57 of 102.09). +2 ms of CPU
        moves the wall +0.17 / +0.13.
      - Leap's v32one at 32 rows: -11.64 ms [IQR -11.85, -11.42]. Seams to 32: -2.15. Both bitwise. LM head as one call:
        -1.49 (not exact).
      - ~~So the engine C32 after step 1 is ~88.3 ms, R32 ~1.445~~ **CORRECTED Mon 08:2x (NewIdeas' number check):
        the measured engine R32 is 1.480** (C32one 90.46 / A16 61.11; a082f8c). The 88.3 ms subtracted v32one (-11.64)
        AND the seams (-2.15) from C32, but the profile has no v32one + seams arm, so it was an untested composition.
        The 17-row step's experts (+16.9 ms) are untouched by step 1.
      - ~~At R32 1.445, tree-32 ~ tree-16 on HE+~~ At the measured 1.480 the T32/16 bar is 1.585 (1.48 x 1.04 x 1.03)
        against dev 1.531, so K* = tree-16 on HE+ by ~4 % (10.99 vs 10.57 kept per A16-ms). Tree-32 needs R32
        <= ~1.43 (E2, or the contract). E2 (Tue) is the clean number.
      - A16 by class: experts 33.7 ms (52 %), HC read 9.4, GDN proj 8.3, attention proj 3.0, MoE dense side 3.8 (5.9 %).
      - Cold first check after a prefill: +74.9 ms (SSD 70.1, 364 misses). A cold 32-row check: 335 ms.
    - **03:0x Mon: the night STALLED from ~00:50; nothing ran for ~2 h.**
      - a45 (gate 47 = its peak 42.5 + 4) and EvalSuite's runner 64 (gate 49 = MEM 45 + 4; waiting since 23:41) are
        both exact-45 C 256 runs.
      - NO-SWAP read 45 at 00:51, right after the profile, then 42 from 01:51 on. GPU 0 %, swap 6.7 GB flat.
      - The orchestrator's watcher woke only on swap, heat, disk, fans and the 03:00 check-in, so the stall went unseen.
        Fixed: night_watch_1116.sh also wakes on GPU < 5 % with no live lock for 20 min.
      - 03:0x asks to the owners:
        - Leap re-armed a45 at gate 42 (03:14, 2f3edf7; wrapper 69340 -> runner 69443). On the orchestrator's word it
          moved to 41 at 03:16 (NO-SWAP 41.2; wrapper 22494 -> runner 22535). NO-SWAP slipped to 40.9 by 03:18, with
          anon up ~0.04 GB/min across idle sessions, so 03:2x moved it to 40 (the orchestrator's second exception). It
          goes first (~10 min). If it doesn't open by 03:40, EvalSuite goes first instead.
        - The only memory left to free is Kamen's own sessions and apps, which stay untouched without his word.
          - This is the FIRST start at a run's own peak (~42.5). ad5f655's 44 was a chain's lowest gate (its run's peak
            35.6 + 8), so it is no precedent for this. CORRECTED (Leap).
          - Pre-registered: a45_vm_1116.txt logs swap, NO-SWAP, the compressor and swapouts every 10 s. Swapouts inside
            the timed window label the misses / SSD-wait part "tight memory: may read high". The dense part stands
            either way.
        - EvalSuite re-armed 03:22 as chain pid 68891, behind A45_1116_DONE written after today's "run a45" line
          (fallback 03:45 if a45 hasn't started). The gate is 43 per process (47 for m1b), via CARD_MEM 39 / 43. Its
          calibration rows are outputs and margins, and the base's speed read-out is kept per check, so neither is
          timing-biased.
          - The away rule replaces the hard 08:00 no-start (the orchestrator agreed): no start while Kamen is at the Mac.
            If he comes back mid-run, it yields at once; S1, B1 and B2 resume with CARD_RESUME=1 once he's away.
            Cutoffs stay 09:30 / 14:00.
      - 03:23:44: a45 SKIPPED at gate 41 (read 2 was 40). 03:24:48: re-armed at 40 (runner 71188).
      - **03:26:10 a45 RAN at gate 40; it ended 03:27:11, rc 0 (61 s).**
        - 17 probe checks, 0 errors, tokens == REF2. Lifetime peak 42.71 GB.
        - Swap +0 MB (6,884 -> 6,884) and swapouts +0 in the run. The compressor went 3.0 -> 8.3 GB at its peak. So
          no "tight memory" label (pre-registered rule), and the split stands in full. Leap's write-up follows.
        - NO-SWAP 50 GB at exit: the compressor effect.
      - **03:28:10 EvalSuite's chain STARTED** on the a45 marker (NO-SWAP 49). S1 x45rpl1 process 1 runs under a
        share lock (MEM 39 = the gate shift; ~39 GB RSS). GPU 75 %, swap flat.
      - **06:28:43 EvalSuite's chain ENDED (rc 0; results/night1116_chain.log, CHAIN_1116_END).**
        - S1 x45rpl1 own rows: tier 0 GO. S2 x45rpl1 REPLAY=exact: GO, A 1.0000 (10,315 / 10,315 drafted rows),
          F_hi 0.00 per 1,000. S1's unset-path check: PASS (40 identical).
        - **Base lx45cc1 (the close-call arms' base for Monday night): tier 1 PASS both ways.** Forward: pooled code
          (n 264) delta -1.14 pt, McNemar p 0.58 (arm only 5 / reference only 8). Reverse: delta +0.00 pt, p 1.00
          (7 / 7). Line > -3 pt (a screening line).
        - **S3 calibration (p2x45rpl1, lossycalx45rpl1, m1bx45rpl1) did NOT run: every process exited rc 1 in ~12 s
          with no tokens, 30 tries each, then "runner stops" (rc 3).** A setup fault, not a verdict (EvalSuite
          def8934):
          - p2 and m1b: an IndexError in the rpl1 snapshot's REPLAY row gate. relax_core.gate_rows reads 15 rows, and
            a block cut by a key's end is shorter. Exact-45 never calls it, so S2's self-check passed.
          - lossycal: its env asks for QOM on the NAX path, which ref2 doesn't build.
          - Amendment 2a: no A / F_hi lines; tier 0 stays the mini-panel KILL line. A re-run needs BigMoE's fixed
            snapshot and a ~1.5 h 45 GB slot (low priority: nothing in WEEK98 waits on it; NewIdeas orders).
          - **06:55-06:58 BigMoE's real-loop key test of the fix (rpl2, 89a6798): 4 / 4 envs PASS**
            (data/r57/rpl2_keytest.log). Two MBPP keys, each replayed to the reference's end: x45 / p2 / m1b 16 / 16 +
            33 / 33; lossycal 16 / 16 + 31 / 33 (A 0.9394, lossy by design). Peaks 41.5-42.6 GB, swap +0.00 on each.
            It started on a 44 GB gate (read 44.2 / 44.2; then 48-50). The re-run is unblocked, but it is not in
            night1117 (EvalSuite pid 52881, armed 06:52); it stays low priority.
      - Lesson (one night's evidence): a +4 GB margin on a 45 GB run can't open on a weekday night (~40-42 GB NO-SWAP
        after the app quit). A start at the run's own peak worked: the compressor took ~2.7 GB of idle pages with zero
        swapouts, and the next run fit at once.
    - The app quit is armed for 23:30:00 (apps_2330_1115.sh, pid 61214).
    - Same order and markers, times +30 min:
      1. Leap a45 from 23:35 (every run ending by 01:30), then C7c;
      2. BigMoE M1 + ng1 (fallback 02:00);
      3. EvalSuite tier-0a + lx45cc1 (fallback 02:00, no start after 08:00);
      4. Breakthrough's profile (deadline Mon 15:00).
    - The 5 GB rule holds until 23:29:30 (mem_watch ends then).
> (a quote from the project owner, kept private)
> (a quote from the project owner, kept private)
    mem?"
    - HOLD is lifted. Free memory (NO-SWAP) must stay >= 5 GB; it was 24.4 GB at 20:54.
    - Research <= ~15 GB peak in total, share locks with MEM, one GPU job above 4 GB at a time.
    - No quiet tickets or timing verdicts while he uses the Mac.
    - mem_watch wakes the orchestrator under 6 GB.
    - MoonShots' T32 (3.8 GB) continues. Leap, Breakthrough and BigMoE continue at desk / slice scale; BigMoE may
      re-arm Monday's m4_night.
    - Tonight's 45 GB chain waits for Kamen's answer on the 23:00 app quit.
> (a quote from the project owner, kept private)
    continue them later".**
    - HOLD + hold_guard are set, so PAUSE is held.
    - The waiting runners were killed with kill -9, so no markers were written (checked):
      - Leap cc3_runner (12190) + c7c_night (65117);
      - BigMoE night1115d (22759) + m4_night (37703);
      - EvalSuite night1116_chain (71626);
      - Breakthrough run_c55_engine (51312).
    - The 23:00 apps quit is stopped (APPS_2300_STOP).
    - MoonShots' T32 is paused between keys by PAUSE (0 % CPU) and asked to stop at a key boundary.
    - On his "continue": rm HOLD, then each owner re-arms the same chain in the same order with new times. The 23:00
      apps quit only holds if he is back before 23:00.
  - **SUPERSEDED for Sun night, 15:0x (orchestrator):** apps quit at 23:00 (apps_2300_1115.sh, pid 33477; force-quit
    OK; Claude / Terminal / Finder / Spotify kept). Fans capped at 4,350 rpm, so heat stretches runs ~1.9x.
    ONE ORDER, so the memory-gated waiters don't start together:
    1. Leap a45 (~30 min);
    2. Leap C7c (end marker C7C_DONE);
    3. BigMoE's full-model M1 check, then the ng1 bit test (end marker NIGHT1115_BIGMOE_DONE);
       - **Now night1115d, pid 22759, armed 17:20:53.**
         - It fixes a launch bug in b / c: they launched jobs as "( env VARS exec python ... )". env cannot run the
           builtin exec, so every job would have hit rc 127 at once. The marker would then have released EvalSuite
           early, and M4 would have dropped.
         - Found by BigMoE's stub simulation of the whole runner (7626e74). Tonight's other runners were scanned by the
           orchestrator: no such line.
         - M4's runner (scripts/m4_night.sh, pid 37703) is armed for Monday 23:05.
           - It runs only on "#3 M1 rc 0: PASS".
           - Its marker is M4_1116_BIGMOE_DONE.
           - Pre-registered in docs/TREE16_M4_PREREG.md before any run.
       - (Earlier) Runner night1115c, pid 31180, armed 17:13. It has the same gates as night1115b; only ng1's verdict
         changed.
       - Before anything ran, two verdict bugs were fixed: the M1 and ng1 records never reached runs.json (72b5f46,
         8a9e240). M1 would have read FAIL even with every check matching.
       - The old runner was killed while waiting, so no marker was written. Verified 17:1x.
    4. EvalSuite's chain: tier-0a, then lx45cc1 both orders (~00:30 -> ~06:30; fallback start 01:30), writing
       CHAIN_1116_END;
    5. Breakthrough's engine profile (deadline Mon 15:00, heat episodes dropped).
  - **Mon 16 night, ADDED 17:1x (orchestrator): BigMoE's TREE16 M4 engine A/B goes FIRST**:
    - when: ~23:05, after the 23:00 apps quit (the Monday quit runner is still to be armed, Monday by day);
    - what: one quiet ticket, NO-SWAP >= 36 GB, t16b chain-16 vs tree-16, C T T C, 8 HumanEval + 8 MBPP keys, C 160,
      256 tokens, ~30 min; pass = tree tokens == chain tokens on every key, then tok/s and kept per check;
    - hard end 00:25 with an end marker;
    - **ORDER CHANGED 17:2x (orchestrator):**
      1. M4;
      2. Leap's E1-E4 from 00:30 (hard end ~02:15, an end marker on any end);
      3. EvalSuite's close-call arms, gated on Leap's marker.
      Why: the check has equal top priority and the close-call is the side track. Timing A/Bs need the cool early
      night; quality arms can spill into Tuesday morning. Leap, EvalSuite and BigMoE were told.
    - **18:1x: Leap's hard end is now 03:00** (more arms: BOF, the contract arms). EvalSuite's fallback is ~03:15.
    - Leap's slot (17:3x): 00:30 -> HARD END 02:15.
      - The end marker ~/Leap/data/week98/screens/E1E4_1117_DONE is written on any end; EvalSuite gates on it, with a
        fallback of its choice.
      - Order: E2 first (the 32-row engine number), then E1, then E3; E4 only if the one-pass fails.
      - It waits for M4's end marker and 00:30. If M4 is dropped (no marker and no BigMoE lock), it starts at 00:30.
      - BigMoE sends Leap its marker path when it arms.
    - it waits for tonight's full-model M1 check: if M1 fails, the slot is dropped.
    - Built: M4 loop test ALL BITWISE on both slices (BigMoE 7f4b4e0, verified: P 1024 and 1990, side paths, the
      chain fallback, 0 bad rows).
  - (Original draft) Sun 15 night: Leap's a45 (exact-45 timing) 21:00 -> ~01:00; then EvalSuite tier 0a REPLAY calibration (~1.5 h;
    PREREG_lossygate amendment 2, eba6d3c; BigMoE's four envs on rpl1 03be8b3, dry runs clean); then the close-call base
    lx45cc1 (lossy-45 as shipped, on snapshot cc1) cached in both orders (~1.7 h) -> done ~04:15 Mon.
  - Mon 16 night: the close-call variant ccx45cc1 (delta 1 nat, CODE-ONLY; PREREG_lossygate amendment 3, 763da1c):
    forward ~50 min, reverse ~50 min if the forward passes. The 1c check variants need their own base on their later
    snapshot (a separate base run). Accept rules use CARD_ENGAGE_MODE=any (> 0 summed per process on code, 0 on IFEval).
    - Added Sun 15 10:3x (orchestrator): the delta-2 arm ccx45cc1d2 (amendment 3b, 5d72210; ~1.7 h in both orders)
      runs right after delta 1 the same night, in the weekday big-test window (Tue 00:30 -> 15:00). Both arms carry
      the new RUNAWAY guard (amendment 3a). NewIdeas may reorder when it's back.
    - Added Sun 15 11:1x (orchestrator): Sunday night after the close-call base (~04:15 Mon): Breakthrough's
      full-engine check profile (~10 min, ~35 GB; gate + NAX6 / OOO covered). Leap was asked what exact engine A/Bs
      to book in Monday night's window after the close-call arms (hc_b + GDNB in the 45 preset; a clean C32m / mr2).
    - Booked Sun 15 11:3x: Leap's exact engine A/Bs, Monday night (Tue 00:30 -> 15:00), after the close-call arms.
      Exact-45 loop, C 160, quiet ticket, NO-SWAP 45 GB, ~50 min:
      - E1: 16-row levers (base / +hc_b / +GDNB / passed new levers; tokens + logits == base first);
      - E2: 32 rows (A16 / C32 split / C32 one-pass, bitwise then G32T timing);
      - E3: Breakthrough's hook profile on the E2 arms;
      - E4: C32m clean, only if the MLX-layout weight pass fails.
  - Wed 18 night: BigMoE's CERTIFY98 fresh-set capture (~3-4 h).
  - Thu 19 - Fri 20 nights: the proof runs.
  - To ask Kamen in the morning: close apps for Sunday night from 21:00.
- Sun 15 03:03: **BigMoE: R57_REPLAY CERTIFIED** (abe7772): exact-45 replays its own rows at 100 % (724 / 724 drafted, 0
  flips, a chat key included). The self-check found 2 start-up bugs (fixed d1fd8c5, e5d17c6); EvalSuite told to pin
  aadd08b. On REF2-T's rows it reads 525 / 527, both flips at REF2-T's own tie points. G32 PASS published (7dd40ff);
  G32T under Leap's PREREG (e9feac1). 1c's wiring needs a small change in Leap's ref2 one-call (2-D nolsb + a GPU dup
  pick): Sunday. BigMoE's Sunday daytime: the overflow guard (slot 0 at weight 0, counter, host redo as two 16-row
  checks, a force-flag test hook) and the loop's GATE_N / MAXK pins; the close-call flag spec is recorded.
- Sun 15 03:1x: **ENGINE TIMING of the exact 32-row check (G32T, Leap's pre-registered arm run by BigMoE on the exact-45
  loop, C 160, warm pool, quiet; 83679e7): 1.625x a 16-row check (IQR 1.60-1.68).** C32 == 2 x 16 bitwise again.
  - **Per check: 16 rows 61-64 ms; 24 rows 96-99; 32 rows 99-102; two 16-row checks 124-131.** One pass is ~20 %
    cheaper than two 16-row checks. Line: "needs kernel work" (1.40-1.70); the slice's 1.53 was a little optimistic.
  - Economics: it pays iff kept32 / kept16 > 1.625. Today's drafter 1.30: no; the goal drafter ~1.64: break-even. So
    the 32-row check needs its overhead cut first. The STEP at 17 rows (16 -> 24 +34.7 ms, 24 -> 32 +4) comes from the
    second, padded chunks of the ops still split 2 x 16 (latency-bound launches).
  - Leap's Sunday builds (exact by construction): one dispatch for both row tiles in the small-weight ops, a 32-row GDN
    core, attention projections at full width. Target R32 ~1.3 on the engine (~+25 % throughput at the goal drafter).
  - **Orchestrator's arithmetic vs the floor:** 16 rows 62 ms vs a 38.8 ms floor; 32 rows ~100 ms vs 52.0. With a goal
    drafter (kept/32 ~22.6), today's engine gives ~200 tok/s exact on HE+-like code; 400 exact needs the 32-row check
    near ~55 ms (~1.8x faster): GOAL B's exact track.
- Sun 15 03:0x: **Breakthrough: CLOSE-CALL acceptance at delta 1 nat = PASS under the walk** (pre-registered c8515d2;
  3c09867; BigMoE re-emitted the HE+ cells with the miss gaps, b316365):
  - **HE+ kept/32 13.77 -> 15.11 (+9.7 %) at delta 0.5, 16.78 (+21.8 %, [14.38, 19.44]) at delta 1, 18.21 (+32.3 %) at
    delta 2**; walk-strict 0-14 93.0 -> 95.1 % at delta 1. MBPP 6.92 -> 7.84 (+13.3 %) / 8.52 (+23.1 %) / 10.14 (+46.5 %).
  - Exposure at delta 1 (accepted non-top-1 per 1K kept): HE+ 15.4 (code 4.1, comment 11.0, string 0.3); MBPP 36.6 (code
    8.1, comment 12.7, string 15.9). Code changes stay under the pre-registered 10 per 1K.
  - PASS (both lifts >= +15 %, code <= 10 per 1K) -> worth EvalSuite's real-decoding CODE-ONLY gate as a lossy variant
    (delta 1) against the lossy-45 base. Teacher-forced upper bounds (J(A)). +22 % on HE+ is ~3 kept per check toward
    ~23.3: it helps, doesn't close the gap alone, and stacks with drafter gains and the lossy check.
- Sun 15 02:5x: **Leap C6: the exact 32-row check = 1.58x a 16-row check** (12-layer slice, quiet ticket 02:50, the
  engine's SB, equal contexts; the harness is fixed: two 16-row checks = 2.007x one).
  - Grid: 20 rows 1.387x, 24 rows 1.461x, 32 rows 1.579x. **Rows cost a STEP, not a slope:** crossing 16 rows starts a
    second, padded chunk of every op still run 2 x 16 (GDN / attention cores, the 8-bit router / indexer, the shared
    expert, small-N projections), which are latency-bound. So 17-31-row checks cost nearly a 32-row one; the useful
    lengths are 16 and 32.
  - Value of each build step at 32 rows (in 16-row checks): the expert union read once +0.26 (the main win); mr2 +0.06;
    GDN at full width 0; the one-pass HC read -0.05 (dropped -> ~1.53x).
  - The build share is 0.62 (line >= 0.5), so the next lever is fewer dispatches: one-launch 32-row forms for the
    remaining 2 x 16 pieces (a 32-row gdn_body, 32-row replicas of MLX's 16-row qmm for router / indexer / shared expert /
    small-N, attention projections at full width). Target R32 ~1.3. Engine-level timing (BigMoE's G32T time mode) is
    pre-registered: R32 <= 1.40 pays; 1.40-1.70 needs kernel work.
  - Leap 03:0x, the decision rule restated: 32 rows pay iff kept32 / kept16 > the cost ratio (~1.53 measured). Today's
    drafter 1.30 -> 16 rows. At the goal drafter (98 % at depths 0-14) ~1.64 -> the exact 32-row check pays ~+7 % now,
    ~+20 % once the second-chunk launches go (cost ratio -> ~1.35). The one-pass HC read lost because the small weights
    re-read from cache on the second chunk (KNEE): the remaining work must cut launches (one dispatch over both row tiles,
    like mg2), not bytes.
- Sun 15 02:5x: **Sandbox98 S5 INJECT (no-download analog) = VOID, not KILL** (838a8e0): agreement with the full 2B's
  top-1, MBPP-50 all lags: no injection 51.98 vs injection 33.63 (-18.4); HE-30 -20.9; kept run per 30-row block 1.66 vs
  0.71. VOID because the injection arm is worse even at lag 1, where it reads the full 2B's state one row back (a
  zero-init output projection under Adam 1e-4 swamped the residual stream), and the 12-layer early exit is not a
  competent small LM (52 % after 8 min). Measured cost: a 12-layer corrector pass over a 32-row block = 15.7 ms on the 2B.
  - **Next, S5b on the REAL pair (no download):** the competent 2B (85.67 % MBPP agreement with the 125B), LoRA on layer
    23 as in 1c, plus a Flamingo-style tanh-gated cross-attention (gate at 0) reading the 125B's own final state (the
    BANK's wd, 10240-dim) at prefix rows u <= s, block-causal; trained on the BANK's 360 files against the 125B's top-32;
    read kept-run and strict per-lag first. Line: injection >= +3 pt (p < 0.05) vs no injection at the same steps ->
    family (c) is worth a build.
- Sun 15 02:47: **GOAL B MILESTONE: G32 PASSES. The first single-pass EXACT 32-row check on the full streamed 125B.**
  (BigMoE; data/deepdraft/bank/g32loop.log "G32TEST verdict: ALL BITWISE ... peak 30.9 GB", rc 0, verified by the
  orchestrator.) With the exact-45 loop's own setup (NAX6, K-73 block, OOO; C 160, MAXK 320, ext build_g32, Leap's v32
  b0c99d5), one 32-row check is bitwise equal to <= 16-row checks on 3 windows: T1 at 32 / 17 / 24 / 31 rows and T2
  (commit k, then a 4-row probe) at k = 1 / 9 / 16 / 17 / 24 / 31, every row.
  - Two fixes got it there: Leap's v32 block path (b0c99d5; the loop's sorted-rank tier block) and BigMoE's gate
    (6169f13: the loader's per-call id buffer held 256 ids and a 32-row call posts 320; it overflowed silently).
  - Before a 32-row loop ships (sent to Leap to split): (1) the overflow guard (a window's distinct count isn't known
    ahead; max seen 242 fits C 256), (2) the loop's GATE_N / MAXK pins.
  - Timing context (C5, slice): one exact 32-row check = 1.63x a 16-row check, 18 % cheaper than two; it pays once the
    drafter's kept/32 / kept/16 passes ~1.6.
- Sun 15 03:0x: **Breakthrough: CLOSE-CALL acceptance under the real decode walk** (pre-registered c8515d2; 7ae4bb5;
  EvalSuite's kept_walk imported; delta 0 reproduces 13.77 / 6.92 exactly). Accept a guess if it is the 125B's top-1 or
  its log-prob is within delta nats.
  - **MBPP kept/32 (exact 6.92):** delta 0.1 7.04 (+2 %); 0.25 7.41 (+7 %); 0.5 7.84 (+13 %); **1 8.52 (+23 %, [8.14,
    9.00]); 2 10.14 (+47 %)**. Walk-strict 0-14: 85.9 -> 90.8 %. Accepted non-top-1 per 1K kept tokens at delta 1: 36.6
    (code 8.1, comment 12.7, string 15.9).
  - **Why modest: the drafter's MBPP misses are mostly FAR from the 125B's top-1 (median gap -4.5 nats; only ~10 % within
    0.625).** NewIdeas' margin ceilings assume the drafter is right on every confident row, a different lever from
    accepting near-ties.
  - HE+: NOT priceable from t30b's HE+ cells (dlp = NaN on all 14,671 misses). Verdict UNDETERMINED; MBPP alone meets
    both lines first at delta 1 (+23 % lift, code exposure <= 10 per 1K). Needs HE+ cells with rank / dlp on misses
    (BigMoE's dd_ceiling misses path, as for MBPP). Teacher-forced upper bounds (J(A)); EvalSuite's code gate decides.
- Sun 15 03:0x: **BIG (NewIdeas, CPU; NewIdeas research/week98/; logged here for NewIdeas, whose account is at its
  limit): the MARGIN CEILINGS under the walk.**
  - At the walk's check starts, unsure rows (125B margin < 4 nats) are **28 % on HE+ (vs 10 % overall) and 37.5 % on
    MBPP**; depth 0 there is 92.0 / 89.7. Walk-strict on HE+ is 90-94 at depths 0-9, then 96-99.
  - A root tree (a rank-1 branch at a fixed 32 rows) is not a lever: HE+ only loses, MBPP +4.5 %.
  - **Ceilings, making every row at or above a 125B margin right (kept/32 HE+ / MBPP, from 13.77 / 6.92):**
    - >= 4 nats: 15.86 / 8.06;
    - >= 2: 17.66 / 9.90;
    - >= 1: 20.70 / 13.64;
    - >= 0.25: 28.33 / 23.48.
  - **So fixing every miss on rows the 125B is sure of caps HE+ near ~16 kept. Reaching ~23.8 needs the 125B's 0.25-2
    nat rows right: a reader of its uncertain picks (exact), or lossy close-call acceptance there.** Deep-drift levers
    drop.
  - Orchestrator's arithmetic for Kamen: exact at ~16-21 kept/32 and a ~55-65 ms cycle is ~250-320 tok/s on HE+-like
    code; 400 exact needs ~23-24 kept, i.e. a drafter that matches the 125B's uncertain picks. Lossy close-call on the
    uncertain rows is the direct route there, gated by EvalSuite.
- Sun 15 02:4x: **Sandbox98 CHAIN SIZE: no chain head at <= 48 MB per step holds its accuracy** (d7d9319;
  results/chainsize.json; 200 matched steps x depth 30 from s30o; control MBPP-50 0 / 23-30 / kept 88.07 / 55.05 / 4.219,
  HE-30 93.14 / 66.68 / 5.921):
  - Best <= 48 MB form: full body 4-bit + 8K list, 43.7 MB per step: MBPP 23-30 -7.4 pt, **kept/32 -12.6 % (MBPP), -23 %
    (HE)**. FFN 1/4 at 8-bit + 8K list (44.0 MB): kept -21 %.
  - Cost per cut (kept/32 MBPP / HE): 8-bit body ~free (-0.1 / -0.6 %); 4-bit body post-training -6.4 / -10.6 %; static
    lm_head list 32K -0.8 / -4.2 %, 16K -4.0 / -10.7 %, 8K -8.1 / -17.4 % (16K covers only 97.7 / 96.4 %; depth 0 pays);
    FFN 1/2 -7.4 / -6.4 %, FFN 1/4 -15.9 / -15.0 % (small per-step errors compound over 30 steps).
  - Action: the cache-resident chain is NOT live as is (<= 12 MB is out of reach for this head); block drafting stays
    the route. Open levers, each its own screen (kept/32 first): 4-bit QAT; a static + prompt-token list; pruning
    recovered by full-budget distillation. For Leap: P2's "4-bit + head 8K" option for the 125B chain needs QAT and a
    better list.
  - S5 started 02:39 (NO-SWAP 35 GB); its read-out carries kept-run and strict per-lag accuracy.
- Sun 15 02:3x: **BIG (EvalSuite, CERTIFY98 amendment 4; 9480ba4, 9861b4f): kept/32 is lost in the FIRST ~15 GUESSES
  AFTER EACH CHECK, not at depth 23-30.**
  - Strict accuracy at EVERY anchor also overstates the engine. The engine drafts only from its own check starts, which
    come right after a miss, and misses cluster. t30b, depth 0: **HE+ 97.94 % at every anchor but 91.58 % at the walk's
    check starts; MBPP 93.97 vs 89.46.** Every-anchor strict implies kept/32 21.25 on HE+; the real walk gives 13.77.
  - The walk-strict version reproduces kept_walk exactly (1 + sum of prod c_j; 1.8e-15 on both sets; kept/16 too).
  - **Where kept/32 is lost** (t30b's walk-strict curve with some depths set to 98 %): HE+ 13.77 as is; depths 0-2 at
    98 % -> 16.53, 0-6 -> 19.83, **0-14 -> 22.63** (of 23.81 all-98); **15-30 alone -> 14.38 (+0.6)**. MBPP 6.92; 0-2 ->
    9.02, 0-6 -> 11.96, 0-14 -> 17.74; 15-30 alone -> 7.63 (+0.7). t30b's walk-strict by band on HE+: 91.4 / 91.9 / 93.8 /
    96.5 / 96.2 / 97.5, rising with depth (survivors). Caveat: the deep walk numbers are survivors, so the counterfactual
    is optimistic about the deep part; the direction is clear.
  - **CERTIFY98's registered rule now:** PRIMARY walk-strict P(right at k | right at 0..k-1) at the walk's check starts,
    bands 0-2 / 3-6 / 7-10 / 11-14 / 15-22 / 23-30, each band's CR1 lower bound >= HE_LINE 97.0, per-depth guard >= 96.5
    (>= 200 rows); CO-PRIMARY kept/32 lower bound >= 20.76 (flat 97 %) + the engine match within +-3 %; dd_ttt bands and
    every-anchor strict = information. Power ~0.7 at a true 98 % with 140 HE+ windows (0.93-0.94 with 300).
  - **What it means for GOAL A (orchestrator):** the deep-drift fixes (oracle 89.7 -> 97.8 at 23-30) mostly fix rows the
    engine never reaches. Depth 0 always has the 125B's TRUE state, yet reads 91.6 % at check starts, so the first
    guesses in hard stretches (right after a miss, where the 125B itself is unsure: WHERE THE ERRORS LIVE, MoonShots F)
    are the wall. Levers are judged on kept/32 and walk-strict 0-14 (certify98.py score <cells>).
  - For GOAL B: with today's drafter kept/32 / kept/16 = 13.77 / 10.63 = 1.30 < R32 1.63 (C5), so 16-row checks win
    today; 32 rows pay only near the goal (0-14 at 98 %: ratio ~1.64).
- Sun 15 02:2x: **CORRECTION (MTP Head): ALLROWS is NOT a GOAL A lever = NO EFFECT by the runtime metric.** kept/32 is
  flat: MBPP-50 4.076 -> 4.066 (CI -0.067..+0.054), HE-30 5.791 -> 5.687 (CI -0.228..+0.015). The band counts every row at
  every depth, including rows already past an earlier miss, which the engine never drafts; allrows trains exactly those
  rows, so it lifts the band, not acceptance. The pre-registered band line was the wrong metric for a lever that changes
  which rows train. Sandbox98's confirmation and any 125B transfer are CANCELLED.
  - **Method rule from here (all drafter read-outs): kept/32, or the STRICT conditional per-depth accuracy (right at k
    given right at 1..k-1), decides; the bands are information.** CERTIFY98's metric must say which it gates on.
  - Sandbox98's independent read of mhx.json: the strict conditional is slightly WORSE where most kept tokens are earned
    (MBPP d1-d4 0.881 / 0.811 / 0.775 / 0.752 -> 0.877 / 0.804 / 0.767 / 0.743; HE d2-d4 0.893 / 0.870 / 0.856 -> 0.885 /
    0.859 / 0.846), and better deep (HE d18-d24 ~0.83-0.84 vs ~0.78-0.81) where few chains arrive (cond_n < 400 past d17).
    It moves accuracy from shallow survivors to deep cells. For later: allrows, or a mix with accept-until-fail, may pay
    once chains survive deep (a near-98 % drafter).
  - NewIdeas re-checked the list under the strict rule (LEVERS_A.md "METRIC RULE", 9e459e4): band-only evidence = the 2B
    audit's "flat at the auditor's agreement" law (its deployed worth is x1.09 accepted), MoonShots' perfect-chooser
    bounds, the earlier sandbox depth verdicts (S4 must read kept/32), and S1 DF-TF (Fit12 asked at 02:25 to add strict
    per-depth + TF kept/32, or save per-anchor hit vectors; its line moves to strict). Unaffected: the fresh-state oracle
    (kept/32 13.77 -> 20.90 HE+) and WHERE THE ERRORS LIVE (depth 0, where strict = band). **GOAL A target restated:
    strict >= 98 % to depth 30, i.e. kept/32 ~23.8.**
  - Also: MTP Head's chain started S4 at 02:20:34 (5.8 GB) against the G32 slot; MTP Head stopped it, and it restarts at
    >= 03:30 with NO-SWAP >= 20 GB.
- Sun 15 02:2x: (superseded by the correction above) **MTP Head's CARRY400 2b ALLROWS read PROMISING by its band line** (auto, night chain v2; TrainLab
  data/sb98/mhx_compare.txt; pre-registered TrainLab REPORT "WEEK98 S2 CARRY400", which predicted "allrows HURTS 40 %").
  - The lever: train without accept-until-fail truncation. Failed rows stay in the chain, so the deep depths get full
    gradient instead of a survival-weighted share. 400 steps from the 2B's native head vs the same-recipe control
    (mhc2a_n1, which equals dn_n1 exactly).
  - **MBPP-50 23-30: 48.37 -> 56.95 (+8.58 pt, CI +7.59..+9.62; 48 / 48 windows better).** HE-30 23-30: 61.69 -> 67.44
    (+5.75, CI +4.79..+6.91; 29 / 30 better). Depth 0: MBPP -0.38 (CI -0.72..-0.06), HE -0.09 (n.s.). kept/32 flat on the
    sandbox (4.08 / 5.79), since its ~88 % depth 0 rarely reaches deep rows.
  - Line (>= +2.0 pt, p < 0.05) met: confirm at the full 1,889 steps (vs s30's 55.47), then the 125B transfer (t30b
    re-trained without truncation; a 15-min screen with a stop line first). On the 125B, with ~98 % at depth 0, deep-band
    gains feed kept/32 directly. Sandbox98 asked to run the confirm ahead of S5 only if it can use MTP Head's recipe
    faithfully; otherwise MTP Head at 06:00.
- Sun 15 02:19: **Breakthrough: the combined lossy code plan on 24 code windows stays within both lines** (f755d70,
  5cf4a34; 6,327 rows): **all-row 1.09 % [0.57, 1.73] (line 1.34), non-tie 0.62 % [0.32, 0.96] (line 0.87)**; ranges about
  halved. Windows 1-12 reproduce the first screen exactly (1.25 / 0.60); the new 13-24 give 0.91 / 0.65. Pre-registered
  action: points within, upper ends cross -> integrate; EvalSuite's code gate decides.
  - The AWQ-style 3-bit GDN + attention screen (dm_bt.py v2, per-linear scaling + an alpha-0 control; ~14 GB, ~12 min) is
    ready but DEFERRED behind G32's memory slot.
  - BigMoE's question answered: 1c's "4-bit" tier keeps REF2-T's base precision (4433333333) there. EvalSuite's three
    spec notes applied (R57_HC4_SETS; the -2.65 line restated as pooled -0.95).
- Sun 15 02:1x: **Leap C5 TIME32, first timing** (80c683b; 12-layer resident slice, quiet ticket, power mode 2): one
  16-row check 24.03 ms, two 16-row checks 58.81, **one exact 32-row check (all of tonight's builds) 48.00 ms = 18 %
  cheaper than two 16-row checks.**
  - The registered R32 read 1.91, but the harness was biased (the 16-row arm ran at ~1.3K context vs ~2.2K). The fair
    read on the same cache: R32 = 2 x 48.00 / 58.81 = **1.63**, in the "pays only with the kernel work" band (1.4-1.7),
    not <= 1.4. (At a 98 % drafter, kept/32 / kept/16 is ~1.69, so 32 rows pay only narrowly at 1.63.)
  - The fixed run (v2, equal contexts) skipped on memory and waits for G32; then v2 with the engine's async submit
    every 4 layers on both arms (c5_sb.py) gives the engine-like ratio. Leap is looking at where the extra ~0.6x goes
    (the 2 x 16 cores, router, attention, build time).
- Sun 15 02:0x: **MoonShots: result H DOWNGRADED** (b36253f):
  - H on MBPP with the start-AND-end guard (exact; relax_guard.py with the validated tracker), 23-30: R(2) 91.07
    (accepted 12.64), R(3) 93.10 (14.44), R(5) 94.85 (16.57; fixes 0.654 of the frozen rule's deep misses; 15.4 % of
    in-span deep rows deviate). The guard costs 0.2-0.4 pt.
  - **J(A), H's approximation tested by branching on the 2B** (pre-registered: >= 90 % unchanged holds, < 70 % not
    trustworthy): 50 MBPP answers, 144 guarded relaxation events (the 2B's rank-2 token in a comment / docstring), the 2B's
    greedy for 64 tokens after each. **The first two code lines after the span were unchanged in only 53 of 102 (52 %):
    NOT TRUSTWORTHY.** Comments steer code (" strip" -> " filter" turned word.strip() into an `if word` filter; "Calculate"
    in a docstring produced a gcd helper instead of edge-case checks).
  - So H's teacher-forced kept gains assume the original continuation, and the code gate will likely see flips (as T16
    and CD16 did). It stays a candidate only if EvalSuite's CODE-ONLY gate passes it in real decoding. Caveats: rank 2 is
    a stress choice; the target was the 2B; a changed line isn't necessarily a failed test.
- Sun 15 02:0x: **MTP Head (auto) CARRY400 2a = NO EFFECT** (TrainLab data/sb98/mhc2a_compare.txt): MBPP-50 23-30 vs
  mhc2a_n1: carry adapter (ca) -0.0104 (p 0.19), carry-only consistency (cac) +0.0070 (p 0.31); control = 0.0000.
- Sun 15 02:0x: **EvalSuite: CERTIFY98's everyday pool drawn** (amendment 3, c0be69c; local only): 29 agent windows
  (Claude Code-style turns, prompts 1.6K-12K tokens) + 60 of IFEval + 61 of IFBench = 150, all with exact 45 GB answers;
  CLEAN (0 / 150 vs n1112). The fresh manifest is complete (HE+ 140 / strict 80, MBPP 300, everyday 150) = BigMoE's Wed
  18 capture list, now **~3-4 h at 45 GB** (decode the fresh answers on the runtime path ~1 h, then the BANK pass ~2.1 h).
  Also: both orders pooled in the all scope give -1.10 pt, p 0.256, PASS.
- Sun 15 02:0x: **EvalSuite: the CERTIFY98 builds are done** (CPU; 4ed09fc, ed721e0):
  - tools/certify98.py reproduces t30b exactly on MBPP 100 and HE+ (all 7 bands to 0.0000 pt; kept/32 6.923860 /
    13.768898). t30b on HE+ dev: NOT CERTIFIED (band-0 lower bound 96.74; 23-30 89.65, lower 86.87; kept/32 13.77, CI
    10.38-17.16).
  - Contamination: dev sets CLEAN against all 3,072 n1112 captures (HE+ 0 / 24, MBPP 0 / 100).
  - **Fresh sets drawn before any candidate** (results/certify98/fresh_manifest_2026-11-15.json; amendment 2): HE+ 140
    gated (+ a strict-80 sensitivity line); MBPP fresh 300 from a pool of 310 (TrainLab had captured 375 MBPP problems,
    all excluded). These ids are BigMoE's Wed 18 capture list.
  - Pooled-orders mode reproduces CERT (code -0.95, p 0.383). Runner 64: CARD_ORDER=rev, engagement key subsets, tier 0a
    REPLAY (R57_REPLAY 6a0f70f read against section 9: matches).
  - Caught a mismatch in Breakthrough's spec: summing the whole lossy_check dict would read chat as engaged; the gate
    reads e3 + edrop + lmsl_rows only; R57_HC4 needs a set scope. Acceptance-rule variants (result H) are in LOSSY_GATE
    section 11 (own counter, pinned checkpoints, MBPP guard the line to watch).
  - Next GPU (not tonight): re-pin to BigMoE's trees, tier-0 identity 40 / 40, then a REPLAY calibration night.
- Sun 15 01:55: **BigMoE built R57_REPLAY and the 1c tier rule.**
  - R57_REPLAY (6a0f70f, scripts/r57_screen_fast.py; EvalSuite's LOSSY_GATE section 9): inert unless R57_REPLAY=
    <rows.jsonl>[:margins.json]. Outputs runs.json "replay" (A, flips binned by the exact margin, F_hi), {TAG}_replay.json
    and a margins file. Toy test + head-test pass. Its exact self-check (4 REF2-T rows, exact-45 flags at C 160; pass =
    100 % accepted) is chained after the G32 test. EvalSuite told to re-pin and confirm the unset path (tier-0 refs).
  - 1c tier rule (c1c217b, bigmoe/tiers1c.py): the float64 reference and the engine's MLX form match Breakthrough's
    tiers_1c masks bit for bit on the 12 code windows (1.63 M slots). Engine wiring + counter next.
  - **Memory slot (orchestrator 01:56): G32 goes first.** NO-SWAP has been 19-26 GB since 01:42 (G32 needs 33).
    peer say to all: from 02:20 until BigMoE says G32 has loaded, no new job above 5 GB. BigMoE takes its share lock (MEM
    26 GB) when Breakthrough's 14 GB lock ends (~02:26).
- Sun 15 01:5x: **Sandbox98 1c (distil the 2B toward the 125B), SHORT FORM = KILL** (3dc2d09; results/LEVERS_A.md
  "Lever 1c", results/lever1c_a.json): LoRA r 16 on layer 23 only, CE to the 125B's renormalized top-32 on 95,070 answer
  rows (360 BANK train files), lr 1e-4. Control reproduced exactly (94.42 / 85.67 / val 87.29). Agreement change HE+ /
  MBPP / val: 1 min (1.2 epochs) +0.27 / +0.52 / +1.43; 2 min +0.44 / +0.30 / +1.40; 4 min -0.03 / -0.52 / +0.96; 8 min
  (8.4 epochs) -0.09 / -0.81 / +0.42. OVERFITTING: the data caps the peak at ~1 epoch, under the +1 KILL line.
  - Sandbox98: a real 1c needs ~10x the 125B's answers plus LoRA through the trunk (a Gated-DeltaNet backward): a build,
    not a screen.
  - Per Kamen's rule, this kills the short form, not distillation. The best version needs far more 125B-labelled rows
    (e.g. teacher-forced 125B prefill over code corpora, much faster than decode), more layers adapted, early stopping
    on val, and weight on the unsure rows where the common misses live. On NewIdeas' 06:00 list.
  - Now running: CHAIN SIZE (pre-registered): 300 steps x 30 deep for the full head (control), FFN 1/2 and 1/4
    (energy-ranked neurons, ridge refit of down_proj), eval-only 4-bit and 32K / 16K / 8K / 4K lm_head shortlists, with MB
    per step at 4-bit. Arithmetic first: the 2B's full head at 4-bit + an 8K list is 43.6 MB per step (inside the 48 MB
    plateau); <= 12 MB needs a narrower head than a native-initialized cut reaches. S5 follows.
- Sun 15 01:5x: **MoonShots result H, the LOSSY comment / docstring relaxation on the frozen 2B audit** (pre-registered;
  6acc7ed). Rule R(k): a deep miss in a comment or docstring span counts as accepted when the drafted token's 125B rank
  is <= k. MBPP is exact; HE+ gives bounds (its cells have no rank on misses).
  - 23-30 (depth 0 holds or rises everywhere): **MBPP 85.14 -> R(2) 91.32 / R(3) 93.48 / R(5) 95.30; HE+ 94.32 -> R(2)
    95.70-96.84 / R(3) 96.20-97.35 / R(5) 96.67-97.82.**
  - Mean accepted with every row audited: HE+ 21.83 -> R(5) 22.73-24.52; **MBPP 9.34 -> 17.23 (x1.84).**
  - **Quality exposure at 23-30** (in-span rows whose token differs from the 125B's top-1): MBPP 9.7 / 13.1 / 16.0 %,
    HE+ 1.9-4.9 % (HE+ docstrings are mostly prompt copies). MBPP is where T16 flipped a task, so its 16 % is the real
    risk the CODE-ONLY gate must test.
  - Line (HE+ 23-30 >= 97.5, depth 0 held): R(2), R(3) fail; **R(5) OPEN (96.67-97.82)**. Deciding it needs HE+ cells
    with the drafted token per cell (MTP Head: one more evalrep pass with am saved); relax.py then gives the number.
  - Check-side cost: T16's +1.9 ms per cycle was a Python span scan (BigMoE's "InertScan host work likely", not
    profiled). An incremental span state + one top-5 on the check's logits should be ~0.2-0.3 ms per cycle. Profile first.
  - Approximation: after a relaxed token, later cells keep their teacher-forced outcomes (the gate measures the real
    effect).
  - **Result I (c21b049): the incremental span tracker passes all three lines.** week98/span_tracker.py, a per-token
    state machine carried across checks (``` fences, # comments, quotes split across tokens, escapes; docstrings only
    at line start after a ':' line or as a block's first statement). Against Python's tokenize + ast on every HE+ / MBPP
    BANK answer (124 blocks, 22,016 tokens): recall 99.32 % (comments 100, docstrings 97.7; misses are safe), precision
    100 %, 0 false in-span rows (of 7,691 code, 115 ordinary-string, 481 prose rows); 0.87 us per token, ~30 us per
    check (vs T16's 1.9 ms). Engine spec: ~/MoonShots/week98/ENGINE_SPEC_relax.md (one tracker per request, rank test
    on the check's own logits, relax_k = 0 = today's exact path; counters relaxed / relaxed_by_rank / span_rows /
    relaxed_pos).
  - New must-have guard: a relaxed token must start AND end inside the span. H's estimates lack it, so they read
    slightly high.
- Sun 15 01:4x: **MoonShots results F + G** (b52690a; LEVERS_A_MOONSHOTS.md):
  - **F, the COMMON MISSES at 23-30 (head AND 2B wrong) are the 125B's own uncertain picks:** only 12.4 % (HE+) / 7.4 %
    (MBPP) have a 125B margin >= 4 nats (vs 89 / 70 % of all deep rows). Near-ties (< 0.25 nat) are 9.4 / 11.1 %: no exact
    method fixes those. Token class: HE+ 74.2 % comment text, 11 % identifiers; MBPP 40 % comments + 37 % string / prose;
    81-83 % mid-line. Copy can't reach them (bigram 1.4 / 4.2 %; only 25-32 % appear in the last 64 tokens).
    - HE+ arithmetic: 98 needs +3.68 pt over the frozen rule; better choosing gives at most +1.97, so >= 1.71 must come
      from these misses (about half of them): 125B-specific knowledge (1c distillation, or the 125B's own state).
    - **Lossy track: 74 % of HE+'s remaining deep misses are COMMENT text**, where a gated relaxed acceptance applies.
      T16-HE (11-07, comment rows kept when the 125B ranks them top 3) was a KILL then (kept +3 %, cycle +1.9 ms, old
      drafter); at a ~94 % flat drafter the same rule is worth far more. Re-priced next (MoonShots).
  - **G, SIZE SCALING (Qwen3.5-4B, on disk, teacher-forced):** HE+ 94.42 -> 95.48, MBPP 85.67 -> 88.02. By the 125B's
    margin, 2B -> 4B: >= 4 nats HE+ 98.83 -> 99.46 / MBPP 97.30 -> 98.64; 1-4 nats 58.8 -> 67.2 / 65.0 -> 70.8; below 1 nat
    flat (43-47 %). **Size helps where the 125B is sure and does nothing on its uncertain picks; those need the 125B's own
    data (distillation). Size alone won't reach 98.**
    - The 4B as the AUDITOR (frozen rule, 0aa353e), 23-30: HE+ 89.65 -> 2B 94.32 -> 4B 95.29 (depth 0 97.98; oracle
      bound 97.09); MBPP 72.13 -> 85.14 -> 87.17 (bound 91.91). Accepted per anchor HE+ 19.33 -> 21.07 -> 21.49. Bytes per
      pass: 4B 2.37 GB, 2B 1.06 GB (spec 0.5). The flat-at-the-auditor's-level law holds a third time (95.29 vs the 4B's
      95.48 agreement).
- Sun 15 01:35: **BigMoE's g32 test FAILED AT LAUNCH** (g32.log: "room: NO-SWAP 40 GB -> run", then "G32-DONE rc 127" =
  command not found). Spotted by MoonShots; BigMoE asked at 01:40 to fix, dry-run against the real loop and re-arm.
  - **G32 LOADED 02:25:41 and running** (25.1 GB active, C 160, MAXK 320, 9 id lines, ext build_g32, V32_MOE1 1; the
    first window's gate_req 142). The > 5 GB hold lifted; big jobs follow their own peak + 8 GB rule.
  - **02:31: the full-path 32-row check is NOT bitwise yet** (g32.log): T1 at M 32: rows bitwise 0 / 32 (max |diff| 3);
    T2 keep 31: probe rows 0 / 4 (max |diff| 2.24); 0-bitwise at M 32 / 17 / 24 / 31 and at every keep on key 1. Leap's
    4-layer slice was bitwise, so the difference is on the full streamed 125B (tiers 4433333333, streamed SwitchGLU,
    the gate above 185 ids, or layers past the slice). BigMoE stopped the test after key 1 and started a BISECT at 02:32
    (B0: two prefills bitwise, taps bitwise, widened bitwise; then rollbacks).
  - **BISECT (02:32-02:33, peak 28.2 GB; the runner ended rc 139, a segfault at exit):**
    - B1 determinism: 16 rows twice bitwise, 32 rows twice bitwise.
    - B2 one 32-row call vs 16 + 16 on one prefill: 0 / 32 rows bitwise but **argmax equal 32 / 32**; per-row max |diff|
      0.9-3.1 (logits).
    - B3 with moe1 OFF (the MoE block split 2 x 16): still 0 / 32 bitwise, so the one-call experts are NOT the cause.
    - B4 layer taps: already non-bitwise at L5 (max |diff| 0.0072), growing to L15 0.0141 and L25 0.0404. **The split
      starts in layers 0-5 of the streamed path**, which Leap's 4-layer slice (exact-45's ref2 forms) didn't cover:
      e.g. the streamed path's 3-bit tiers (4433333333) or another op that takes a different kernel at > 16 rows.
  - **02:4x: CAUSE FOUND AND FIXED (Leap's bug in v32; b0c99d5).** At 17-32 rows, v32's one-call MoE path called MTPLX's
    stock block instead of the exact-45 loop's own block. The loop's block sorts the top-k into gate-rank order (REF2-T
    decodes ranks 0-1 / 2-9 by position at 4 / 3 bits) and hands the scores to the tier call, which combines inside it;
    the stock block does neither, so the right experts were decoded at the wrong bit widths and combined by MLX
    afterwards (small drift on every row, argmax intact). Leap's 4-layer slice had no tiers, so it couldn't show this.
    The fix replicates the installed block (K-73 or R-69) at 17-32 rows with one switch call, and refuses the 2 x 16
    block split on the streamed engine. BigMoE is re-running G32.
  - Leap prices the pipeline at 16 rows (today's operating point) and adds 17-24-row checks to the timing grid (C6,
    needs a ~23 GB slot, queued on its two-read gate).
- Sun 15 02:34: **Fit12's S1 DF-TF CRASHED at start** (DONE "rc 1"; s1tf.py:69 draft.hidden_states(blk, empty, cache, j):
  "[reshape] Cannot infer the shape of an empty array" in DFlash2's k_norm). Fit12 told (fix + head-test + re-arm if
  sub2's window allows, else first at 06:00).
  - Fixed 01:42: the wrapper was zsh, and dd_guard.sh needs bash (set -m); the runner is now a bash script. The
    head-test (G32_DRY=1, the exact launch, no model) found a second bug: g32_test.py asserted 4 extra id lines where G32
    has 9, which would have crashed after the model load (fixed d22f418). Dry run rc 0: MAXK 320, gate_req 142 / 154 / 156
    per window, all within C 160. Re-armed; starts when NO-SWAP reads >= 33 GB twice.
- Sun 15 01:3x: **GOAL B lossy: Breakthrough's map screens MEASURE the code plan within both lines** (01:19-01:35, the
  12 code windows; d9b2db7 / 09a6efd; pre-registered action 0d90b27):
  - **Combined (1c + 4-bit mixers + 16K lm_head shortlist): all-row 1.25 % [0.45, 2.30] (line 1.34), non-tie 0.60 %
    [0.19, 1.06] (line 0.87).** 5.86 GB per 32-row check = **20.9 ms at the floor, 24.9 ms at 84 %.**
  - 1c experts alone 1.13 / 0.48 (predicted 1.08 / 0.50: the expert calibration holds). 4-bit mixers alone 1.16 / 0.48,
    but on top of 1c only +0.09 / +0.09: flips are SUB-additive (both perturbations flip the same fragile rows), so the
    tie-aware ~1.23 was right.
  - GDN + attention at 3-bit (RTN): 4.43 / 3.62 = CLOSED until a better quantizer.
  - Caveat: 12 windows give wide ranges, and 84 % is an assumed kernel efficiency. EvalSuite's code gate against the
    lossy-45 base decides quality; the engine decides time.
  - EvalSuite's asks answered in ~/FableGod/deepdraft/LOSSY_CHECK_SPEC.md: base lossy-45 (pkg_120b, already HC8 + QOM
    on code, dense read ~2.81 GB); set-scoped flags R57_TIERS* / R57_LMSL* / R57_HC4; one runs.json "lossy_check"
    counter (e4 / e3 / edrop / lmsl_rows); patch notes + a CPU self-check. Integrate and gate at 45 GB first.
- Sun 15 01:35: **Leap KNEE probe = LINE MET: a cache-resident chain is a second drafting route** (4be8cf9):
  - Re-read bandwidth by weight-set size: 2-8 MB 2.3-3.2 TB/s (~10x DRAM); 12 MB 1.1 TB/s; 16-48 MB ~500 GB/s; 96 MB+
    DRAM ~276 GB/s.
  - An MLX chain of 32 dependent GEMVs (one eval, no host sync): 0.026 / 0.035 ms per step at 8 / 12 MB, 0.11 / 0.18 ms
    at 32 / 48 MB. **30 steps cost ~1 ms at <= 12 MB, ~3-5 ms at 32-48 MB: inside the 400-exact drafting budget.**
  - **Budget for the drafter owners:** <= ~8-12 MB of weights per step (or <= ~48 MB at ~500 GB/s), the token kept on
    the GPU (argmax -> embedding, no .item()), few kernels per step. Today's MTP head: 95-178 MB in ~40 kernels + a host
    sync. Example: ~24M parameters at 4-bit = 12 MB (an 8K-row screened head alone is 10.5 MB).
- Sun 15 03:4x: Breakthrough screened the context list for the lossy check's lm_head shortlist (offline, conservative;
  pre-registered 03c71ba; f1f6418): **keep the 16K static code list.** Code misses: 4K static 4.25 % -> with context
  1.04 %; 8K 1.49 -> 0.48 %; 16K 0.03 % both ways (it already covers the bank's 8,775 distinct code tokens). Combined code
  arm: 16K static or with context 1.09 / 0.62 (within); 8K + context 1.39 / 0.91 (just over). Bytes barely differ (<= 18
  MB). Chat: context cuts the code list's misses (IFEval 32 -> 26 %, multi 76 -> 51 %) but is nowhere near safe (line
  0.5 %); a chat shortlist needs its own chat-derived vocabulary. The shortlist stays code-scoped (R57_LMSL_SETS).
  - Open: can a drafter that small reach ~98 %? That is now the drafter owners' question; the speed side is measured.
- Sun 15 01:34: **EvalSuite: CERTIFY98 delivered** (~/FableGod/deepdraft/CERTIFY98.md + EvalSuite PREREG_certify98,
  ed4dee6):
  - **Measured clustering:** t30b's design effect is 15.7 at band 0 and 131 at 23-30 on HE+ (MBPP 5.1 / 27). With 140
    fresh HE+ windows, "lower bound >= 97.5 in every band" passes only 56 % of the time at a true flat 98 %; 80 % needs
    ~300 HE+-like windows.
  - **Registered: HE_LINE = 97.0** (94 % power at a true 98, 0.5 % false certification at a true 97); "98 met" is added
    when every band's point estimate is >= 98; per-depth guard >= 96.5. MBPP_LINE = none (one number to change).
    Everyday text reported. Margin bins (m >= 4 vs unsure) required.
  - Held-out: all sets screened this week are DEV. Fresh test sets: HE+ 140, MBPP 300 (from the test split, local),
    everyday 150; a 50-token-span contamination check first.
  - **METHOD CERTIFIED (HE+)** needs: (a) the 125B oracle at the line on HE+ fresh; (b) the 2B at full budget within
    1.0 pt of its own oracle at 23-30; (c) a 125B partial run at matched steps beating the baseline at 23-30 and closing
    >= 0.8x the 2B's share of the gap.
  - Kamen decisions: HE_LINE 97.0 or 97.5 (97.5 needs ~150 more HE+-like windows = a download); an MBPP line, if any.
  - Timeline: Mon-Tue tools (certify98.py, contamination checker, reproduce t30b to 0.01 pt); **Wed 18 night: capture
    the fresh sets on BigMoE's bank path (~590 files, ~2.1 h at 45 GB, BigMoE's GPU)**; Thu-Fri proof-run scoring.
  - Lossy gate amendment 1c: both orders -0.95 (McNemar p 0.383, 8 / 13); a forward PASS is provisional until the
    reverse order runs. v1' confirm planned (appendix A): MBPP-x 300 + LiveCodeBench v6 131, both orders, ~4.9 h at 45 GB.
- Sun 15 01:2x: **MoonShots result E, the LEARNED CHOOSER: the frozen rule stands** (8e4639c / a9a37eb; pre-registered;
  draft-time features only, since the cells' m / rank / dlp are the 125B's own answer, a leak):
  - Held-out 23-30, head / frozen / learned / oracle: MBPP files 50-99 72.06 / 85.02 / 84.93 / 89.96; HE+ 89.65 / 94.32 /
    94.34 / 96.29. Depth 0 within 0.05. Line (HE+ >= 96, MBPP >= 88) not met.
  - **The real oracle (head OR 2B right, per cell) is HE+ 96.29 / MBPP 89.96**, not the independence estimate's 99.4 /
    96.0: the head and the 2B miss the same hard rows. The frozen rule already recovers ~70 % of it. No chooser over
    today's 2B reaches 98; that takes a better auditor (1c).
- Sun 15 01:2x: **The v1' code finding, corrected** (MoonShots as editor; ~/Papers/CERT_CODE_CAVEAT_2026-11-15.md,
  private): EvalSuite's -2.65 is CERT's FORWARD order only. The reverse order is +0.76, and **over both orders the code
  sets come to -0.95 points (net -5 of 528 pairs)**. CERT's pre-registered lines (both orders) passed and stand.
  - Twelve published sentences checked (papers v1.1.0, package 1.2.0 docs, release notes). A caveat (not a correction)
    is drafted for six spots, for Kamen to approve; no public text changed.
- Sun 15 01:2x: **Leap build step 2 (mr2 at 17-32 rows) = DONE, exact** (6525ada; separate module v32mr2.py, so the v32
  BigMoE's test imports is unchanged): GDN projections (qkv | z, out_proj, the LM head) at full width with one weight
  read; only the recurrence core runs 2 x 16. Forward at 32 / 17 / 24 / 31 rows and commit at keep 1-31 bitwise.
  Worth ~-2.7 ms per 32-row check (from C2's per-shape times, not yet engine-timed); costs ~+1.5 GB of native copies.
  **The 17-32-row exact path now has every piece but the gate** (BigMoE's g32 test). Leap's KNEE probe is queued.
- Sun 15 01:2x: **Disk**: 121 GB free, down 17 GB since 22:20 (~6 GB/h), against Kamen's ~100 GB floor (RULES.md). Most
  of the last 3 h of writes are Sandbox98's (data/mc2 8.2 GB, data/knn 2.8 GB, 0.2-0.4 GB checkpoints per arm,
  including MTP Head's). Asked both to keep only needed checkpoints; no deletes tonight; the night watcher now wakes
  at < 108 GiB.
- Sun 15 01:2x: **BigMoE night status:**
  - **G32 gate BUILT** (488c428, ext/build_g32; inert for MAXK <= 185). V32_MOE1 sends Leap's one-call experts into
    BigMoE's streamed SwitchGLU.
  - The full-path bitwise test (g32_test.py, ab6573b: Leap's c3_v32 T1 / T2 on the streamed 125B at C 160, MAXK 320) is
    queued behind its NO-SWAP >= 33 GB guard (30 GB at 01:2x while Breakthrough's 10 GB job runs).
  - The overflow guard is designed, not built. First it must check whether every GPU path zero-weights a fully dropped
    layer. Tonight's test windows fit the pool.
  - R57_REPLAY is NOT built tonight, deliberately: it edits the decode loop EvalSuite's gates depend on, after ~22 h of
    session. The build plan is in HANDOVER 09c8f78, for a fresh build first thing Sunday.
- Sun 15 01:2x: **Sandbox98 depth-63 read-out (for Leap): 48 / 64-row checks don't pay** (54b4b10; results/deep63.json;
  the 2B's s30 head trained 30 deep, so 31-62 is extrapolation):
  - MBPP-50 chain, bands 23-30 / 31-38 / 39-46 / 47-54 / 55-62: 55.5 / 53.4 / 51.5 / 50.5 / 50.1; kept per 32 / 48 / 64
    rows 4.32 / 4.32 / 4.32. True state: flat ~87.7-87.8 to depth 62; kept 7.75 / 7.97 / 8.02.
  - HE-30 chain: 68.7 / 66.5 / 64.8 / 63.1 / 61.6; kept 6.24 / 6.26 / 6.26. True state 93.3 -> 92.5; kept 12.23 / 13.30
    / 13.81.
  - So even a perfect-state drafter gains only +3 % (MBPP) / +13 % (HE) kept at 64 rows: a longer check must cost less
    than that much more than a 32-row one. Chain drafting grows linearly (2.54 ms per token at 1 layer on the 2B).
- Sun 15 01:21: **NewIdeas: WHERE THE ERRORS LIVE** (CPU; BigMoE's CEILING cells + MoonShots' rows2b; NewIdeas
  research/week98/where_errors_live.py + .txt).
  - **MBPP.** 29 % of answer rows have the 125B's own margin under 4 nats. With the TRUE state, t30b's depth 0 gets
    99.89 % on the sure rows and only 79.6 % on those 29 % (45.7 / 63.1 / 82.6 / 95.1 % in the bins < 0.25 / 0.25-1 /
    1-2 / 2-4 nats). Overall that is 94.01.
  - **What 98 % on MBPP would take.** Keeping the sure rows at 99.89, it needs 93.4 % on the unsure rows: +14 pt over the
    125B's own head with a perfect state. Close-call acceptance adds only +0.2-1.7 pt (CEILING). So exact 98 % on
    MBPP-like text is out of reach for a cheap drafter unless the corrector learns the 125B's uncertain choices.
  - **HE+.** 90.2 % of rows are sure (>= 4 nats). The 2B gets 98.83 % on them and 53.9 % on the rest. With the head's
    sure-row level (~99.9 %), 98 % needs ~80.5 % on HE+'s unsure rows, against ~79.5 % implied by the oracle's 97.9. That
    is about +1 pt on 10 % of rows, the "+0.2-0.3 pt reader" in this log. **98 % is reachable on HumanEval-like code.**
  - **For the correctors (and CERTIFY98):**
    - The 2B's misses fall 9 / 30 / 25 / 22 / 13 % by margin bin on MBPP, so 87 % sit on rows the 125B itself is unsure
      of. Distillation (1c) targets exactly those rows.
    - Goal A's certification should be stated per text type: HE+-like code can carry 98 %, MBPP-like code tops out near
      ~94-95 % exact by this arithmetic.
  - Orchestrator's arithmetic: at 94 % flat, kept per 32-row check is ~14.4 (vs ~23.3 at 98 %), so MBPP-like code would
    run at ~60 % of the HE+-like speed. Tell Kamen in the morning: the 400 / 1k targets hold for HumanEval-like code.
- Sun 15 01:2x: **MoonShots: HE+ CONFIRMS the frozen 2B-audit rule** (tau 4 / 2 / 1 / 0.5 by depth, frozen before the
  run; LEVERS_A_MOONSHOTS.md result D; MTP Head's cells_t30b_heplus.npz, 186,465 cells, 0 margin mismatches):
  - HE+ head -> audited: depth 0 97.95 -> 97.90; 7-10 93.06 -> 94.76; 15-22 90.94 -> 94.52; **23-30 89.65 -> 94.32**.
  - Mean accepted per anchor 19.33 -> 21.07 at P = 1 (x1.09), 21.42 at P = 2, 21.83 with every row audited.
  - Line passed (depth 0 within 0.3, 23-30 +4.67 >= 3.0, accepted x1.09 >= 1.05). **The law "flat at the auditor's own
    level" holds on both sets** (flat at ~94.5 % from depth 7 = the 2B's own 94.42 %).
  - **ANE pricing: the Neural Engine can't hide this audit at the exact floor.** t30b drafts block n+1 from check n's
    final state, so block n+1 doesn't exist until check n ends; a stale speculative draft pays only on full accepts. The
    cost stays GPU-serial: 1.06 GB, ~4-9 ms per pass until timed, ~-4 % to +5 % net on MBPP at a 52-55 ms check. A win at
    the floor needs a ~0.4 GB auditor or an early-tap drafter.
  - 1c (distil the 2B toward the 125B) moved to **Sandbox98 now** (NewIdeas 1280261; script
    ~/MoonShots/week98/lever1c_distill.py). Lines: MBPP >= 89.0; 98-class HE+ >= 97.0 and MBPP >= 91.0.
- Sun 15 01:18-01:20: **Leap C3b = ALL BITWISE** (9e9f9ba; data/week98/screens/README.md "C3b"): v32 runs MTPLX's
  block VERIFY path at 17-32 rows (keeping the verify sigmoid lowering); router + shared expert split 2 x 16; **the
  experts as ONE call** through ref2's resident kernel, so the expert union is read ONCE (the bytes win of a 32-row
  check). Forward at 17 / 24 / 31 / 32 rows and commit at keep 1-31: bitwise on every row; the one-call path fired on
  4 / 4 layers (a first run at 01:17 missed the unfused class and was re-run).
  - Split agreed: the 32-row block override stays in Leap's ref2 / v32; **BigMoE builds the 32-row gate** (gate + stream
    + the overflow redo; G32 488c428: the sync-free gate carries up to 340 ids), tested by one 32-row verify through
    the wide gate vs 2 x 16 on the runtime path.
  - Leap next: build step 2, mr2 for the dense projections at 32 rows (dense R 1.21 vs 1.65, ~-5 ms per 32-row check),
    bit tests under a share lock, no timing until done.
- Sun 15 01:1x: **Leap P2 CHAIN-BYTES = CLOSED** (6e635f1; the 125B's MTP head standalone, quiet ticket):
  - MB / ms per step: as stored 177.6 / 1.147; bf16 Linears at 8-bit 146.9 / 1.065, at 4-bit 130.5 / 1.008; screened
    head 16K 154.0 / 1.052, 8K 142.2 / 1.009; 4-bit + head 8K 95.1 / 0.864.
  - Fit: each step is ~0.54 ms fixed (kernel launches) + bytes at ~290 GB/s. **A 32-step chain costs >= ~17 ms at ANY
    bytes in MLX's per-op form**; the line (<= 0.5 ms and <= 60 MB per step) is not met.
  - **Design constraint for GOAL A:** the 98 % method must run as a one-pass block drafter, a chain whose whole step is
    one fused kernel, or on the Neural Engine. Optional keep: 4-bit + head 8K cuts a 32-step chain 36.7 -> 27.6 ms if a
    bank replay shows acceptance holds.
- Sun 15 01:1x: **Sandbox98 lever 3 kNN STATE MEMORY = KILL** (f3335fd; results/LEVERS_A.md). Index: 1,040 Magicoder
  answers = 737,010 rows keyed by the last 16 / 12 / 8 / 6 / 4 tokens (no MBPP / HE rows).
  - MBPP-50 23-30 vs chain 55.47: nearest n >= 6 -0.07; mean of 8 +0.97 (p 0.11); nearest n >= 12 +0.21 (n.s.). Hit rate
    at 23-30: 17 % (n >= 6), 1.2 % (n >= 12).
  - On the hit cells the retrieved states are WORSE than the head's own carry (52.75 / 56.22 vs 65.24).
  - Reading: a real state from another context that shares the last 6+ tokens carries that OTHER context's
    composition. Freshness has to come from THIS context: a composer run over these tokens.
- Sun 15 01:17: **NewIdeas: the merged GOAL A list** (~/FableGod/deepdraft/LEVERS_A.md; NewIdeas research/LEVERS_A.md).
  - The spec forces a block drafter. A sequential chain head costs ~2.2 ms per step in the engine (TRIED HEADBENCH;
    Leap's standalone P2: 1.147), so ~37-66 ms per block. DFlash2's one pass reads 0.52 GB (10.96 ms; byte floor ~1.9 ms).
  - The structural point, now measured by MoonShots' audit: a corrector makes the drafter FLAT at its own teacher-forced
    agreement. So **GOAL A = a corrector at >= 98 % teacher-forced agreement, <= ~0.5 GB per pass** (or on the ANE: Leap
    P1, x1.07).
  - Three corrector families:
    - (a) target-fed block models: DFlash2. S1 DF-TF -> Fit12.
    - (b) distilled small LMs: the 2B and the 0.8B. S2 = lever 1c -> MoonShots (its line MBPP >= 89.0; 98-class
      information line HE+ >= 97.0 and MBPP >= 91.0; Sandbox98 is the backup).
    - (c) a small LM with the 125B's prefix features injected. S5 INJECT -> Sandbox98, after kNN, on the 0.8B.
  - S3 JACOBI-K (K refinement passes, kept/32) follows S1 / S2. S4 DEPTH-FAIR -> MTP Head, with each chain layer priced
    per step.
  - Prior art: xPress 2608.02438 (Jacobi on diffusion drafters), PARD 2504.18583, DFlare 2606.02091 (depth pays with
    per-layer target fusion and 3x data), SPIRe and 2505.15380 (copied target layers as the drafter's init), DistillSpec /
    AdaSPEC / Draft-OPD.
- Sun 15 01:1x: night tasks (orchestrator):
  - **BigMoE** (main; idle since 23:2x, main 21 % vs a ~33 % linear line): (1) the wider gate (320 ids) + a 32-row MoE
    block that keeps the verify lowering, bitwise vs 2 x 16, file boundaries agreed with Leap (Leap's build step 1
    waits on it); (2) R57_REPLAY mode (LOSSY_GATE section 9); (3) Breakthrough's 1c integration.
  - **Breakthrough:** the three map screens (the combined arm first), then EvalSuite's three asks as a spec for BigMoE.
  - **EvalSuite:** GOAL A's certification protocol (~/FableGod/deepdraft/CERTIFY98.md + a PREREG): metric, held-out
    discipline, sample size from the measured clustering, decision rule, and what certifies a METHOD.
  - **MoonShots:** the 2B audit on the Neural Engine (price), the frozen tau schedule on HE+, the 0.8B auditor screen.
  - **MTP Head:** sends cells_t30b_heplus.npz to MoonShots when written. **NewIdeas:** ranks lever 1c and the new results.
- Sun 15 01:1x: **MoonShots: the 2B AUDIT on MBPP** (a489724; week98/join_mbpp.py, CPU; all 438,850 CEILING cells
  matched):
  - Rule: where the 2B disagrees with the head and its own margin is >= tau, the 2B's token stands.
  - MBPP 100, depth 0 / 3-6 / 23-30:
    - head: 94.01 / 83.45 / 72.13 (7.38 accepted per anchor);
    - tau 2: 93.18 / 86.48 / 82.47;
    - tau 3: 93.84 / 85.61 / 80.33;
    - **tau schedule 4 / 2 / 1 / 0.5 by depth: 93.96 / 86.90 / 85.14 (9.34 accepted per anchor at all audit passes).**
  - No single tau passes the line (23-30 >= 82 with depth 0 within 0.3), so it is BETWEEN. The schedule passes, but it
    was picked after seeing the rows, so HE+ confirms or kills it (frozen first).
  - **What it means:** the audit flattens the drafter at the AUDITOR's own level (MBPP 85.7, HE+ 94.4). It can't pass
    that level, so it isn't 98 % by itself; 98 % needs an auditor that agrees ~98 % with the 125B.
  - Bytes: one 2B pass reads 1.06 GB, so 1 pass costs ~+7 % of a 49 ms check: +12 % accepted, ~+5 % net (2 passes ~+4 %
    net). On the Neural Engine in the check's shadow it could cost ~0 (being priced; ANE-1 measured the GPU check
    x1.07 beside back-to-back ANE drafts).
  - Next levers: 1c, distil the 2B auditor toward the 125B (LoRA KL on the 125B's top-32 from the BANK; line MBPP
    agreement >= 89.0); and Qwen3.5-0.8B as an auditor inside 0.5 GB (mlx-community/Qwen3.5-0.8B-MLX-4bit, ~0.6 GB,
    Apache-2.0). NOT fetched: MoonShots' rule needs Kamen's own OK, so the orchestrator asks him in the morning.
- Sun 15 01:1x: **Leap P1 OVERLAP = CLOSED** (02d58c6): a 32-step chain head on a second GPU stream beside a 48-layer
  check proxy. 16 rows: check 84.6 ms, chain 37.6, together 119.7; 32 rows: 110.8, 38.0, 145.3. kappa 0.93 at both (line
  >= 0.52): one GPU runs the two streams almost serially.
  - So drafting must get cheaper itself: P2 (the chain step's byte diet) or a one-pass block draft (R-10: 10.96 ms at 32
    rows, 0.52 GB, ~5x its byte floor). The Neural Engine is the one overlap route measured to work, and only for a
    dense drafter.
  - Found while building: MTPLX's MoE block takes its PREFILL combine at >= 32 rows even in verify (a different sigmoid
    lowering for the shared-expert gate). So one 32-row MoE block is NOT exact as is; v32 splits it, and BigMoE's wider
    gate must handle it.
- Sun 15 01:1x: **Leap C3 = ALL BITWISE** (08d4a80): REF2's 17-32-row verify scaffold matches <= 16-row verifies row for
  row on layers 0-3 (3 GDN incl. PLE, 1 attention), forward at 17 / 24 / 31 / 32 rows and commit at keep 1-31 (max
  |diff| 0). Left: (1) the experts as ONE 32-row call (needs the wider gate + the verify lowering), (2) mr2 for the dense
  projections (R 1.21 vs 1.65), (3) the exact-45 loop's K-73 block and the streamed tiers at 17-32 rows.
- Sun 15 01:1x: **Breakthrough's dense addendum** (278a404 / 28c8a3c; LOSSY_CHECK_PLAN.md, docs/LOSSY_CHECK_PRICE.md; a
  pricing, not a screen):
  - The dense 3.44 GB: HC mixers 1.28 (bf16), GDN 1.17, lm_head 0.36, attention 0.34; the rest <= 0.13 each.
  - Cuts:
    - Mixers + PLE to 8-bit: -0.63 GB. On the map (hc8w), code flips are at the numeric-noise level. This is the 12 GB
      line's HC8.
    - Mixers + PLE to 4-bit: -0.97 GB (+0.42 / +0.12 %, confounded).
    - lm_head 16K-row code shortlist: -0.33 GB (0.03 % code misses). Chat needs its own list.
    - GDN + attention at 3-bit: -0.34 GB, not measured alone.
  - **GOAL B lossy, on paper (code):**
    - lambda 1.2e-2 tiers + 4-bit mixers + shortlist: 6.61 GB, **23.6 ms at the floor / 28.1 at 84 %**, within both
      lines.
    - 1c + the same: 5.86 GB, **20.9 / 24.9 ms**. Non-tie flips are within (0.65 vs 0.87). All-row flips are over (1.53
      vs 1.34) unless tie flips are counted once (~1.23, Breakthrough's reading).
  - So sub-25 ms at a realistic 84 % rests on that tie-aware reading and on the dense and expert flips adding up. The
    map screens decide (~4 min each, GPU): the combined arm, mixers 4-bit with exact experts, and GDN + attention 3-bit
    alone.
  - Toward ~1k: that needs a ~16-17 ms check. The remaining dense (2.1 GB = 7.6 ms at the floor) is the next wall.
- Sun 15 01:1x: **EvalSuite's CODE-ONLY gate is in** (5147f0a / 6a5a053; LOSSY_GATE.md section 10, PREREG_lossygate
  amendment 1, runner 64, lossy_d4.py):
  - HE+ 164 + MBPP 100 pooled (-3 pt line, p >= 0.05), with an interim look at 132.
  - Then IFEval 100 in its own processes as the switch-off check: the counter must be 0, and tokens must be identical to a
    cached base run.
  - Verdicts carry "(CODE-ONLY mode)".
  - **Finding (a re-cut of CERT's 11-08 data; information):** the shipped 45 GB stack v1' is -2.65 pt vs exact on HE+ +
    MBPP pooled (McNemar p 0.039, 1 vs 8 discordant).
    - CERT's pre-registered verdict (all three sets pooled: -2.20, p 0.077) stays PASS.
    - It is a post-hoc subset with 9 discordant pairs: a lead to confirm on a larger code panel, not a verdict.
  - So code-scoped variants are gated against a cached base run (their own cost). Base + variant vs exact is tier 2's
    question.
  - Gap: a deployed code-only mode needs a request classifier, and its misroutes are not gated.
  - Needs from Breakthrough: a set-scoped switch, the counter field, and the base env.
- Sun 15 01:1x: usage. sub1 (account-1) is at 91 % at 01:08, +4 in 30 min. At this pace it is used up by ~02:15, ahead
  of its 04:00 reset. sub2 reset at 01:00 (2 %).
- Sun 15 01:1x: **MoonShots on the 125B BANK** (LEVERS_A_MOONSHOTS.md v1; ~/MoonShots/week98/):
  - The 2B as a STATE composer is KILLED: ridge 2B -> wd R2 0.588, predicted MBPP 23-30 67.1 (< 78).
  - **NEW LEAD:** the 2B as a TOKEN co-drafter. Teacher-forced, it matches the 125B's runtime top-1 at 94.42 % (HE+)
    and 85.67 % (MBPP), FLAT with depth and with no training. On sure rows (>= 4 nats): 98.8 / 97.3 %.
    - t30b at 23-30 is 89.65 / 72.13.
    - Perfect-chooser upper bound at 23-30: 99.4 (HE+) / 96.0 (MBPP).
    - Proposed design: the head drafts, the 2B audits the block in one pass and repairs from the first confident
      disagreement (published as staged / cascade SD; the flat-vs-deep numbers are ours).
    - Cost is the open question: the 2B's bytes per pass vs the <= ~0.5 GB / block spec.
    - Next: a tau sweep joined with t30b's MBPP cells (ceiling/cells_t30b_mbpp100.npz, CPU, now). HE+ cells come from
      MTP Head tonight.
    - Sun 15 01:2x, **HE+ cells DONE (MTP Head):** ceiling/cells_t30b_heplus.npz, sent to MoonShots.
      - t30b's own chain on the 24 HE+ windows with runtime-path (bank) context, via BigMoE's evalrep run unmodified
        (--set code --ctx bank --replace wd --refresh 1000, so nothing is replaced).
      - 186,465 cells. By band: 0 97.95, 7-10 93.06, 23-30 89.65; kept/32 13.77.
      - The MBPP cells' fields and order, plus fi / an / row. rank / dlp are unknown on misses.
      - Builder: TrainLab sticky/w98_heplus_cells.py.
    - Sun 15 02:2x, **CARRY400 on the 2B sandbox (MTP Head): no carry lever moves kept/32.** 400 steps from native,
      control = dn_n1 exactly.
      - Carry adapter (guess and carry split): -1.0 pt, NO EFFECT. Carry-only consistency loss: +0.7 pt, NO EFFECT.
      - **allrows (no accept-until-fail truncation) is NOT a lever:**
        - The band 23-30 is +8.6 (MBPP-50) / +5.8 (HE-30), but kept/32 is 4.076 -> 4.066 (CI -0.067..+0.054) and
          5.791 -> 5.687.
        - The band is unconditional: it counts rows already past an earlier miss, which the engine never drafts.
          allrows trains exactly those rows.
      - Method: judge row-set levers by kept/32 or the strict conditional, not the band.
      - TrainLab REPORT 'WEEK98 S2 CARRY400 RESULT'.
    - Sun 15 03:52, **S4 DEPTH-FAIR = KILL (MTP Head, 2B sandbox).**
      - Setup: an UNTIED copied (non-zero) second layer, from native on s30's recipe with the 1,889-step cosine. One arm
        in the main Adam, one on its own 3e-5.
      - Both lose to the 1-layer control at the step-380 stop line. MBPP-50 23-30: -2.5 / -7.9 pt. kept/32: -0.24 /
        -0.50 (HE-30 -0.40 / -0.70). Depth 0 is down too.
      - With deep2 (zero start) and tie2, every extra-layer variant trained on the chain loses or ties here.
      - Cost if it had worked: +56 % per drafted token (2.54 -> 3.96 ms on the 2B). On the 125B that is ~+1.2-1.4 ms
        per chain step, ~40 ms per 32-block, against the 2-4 ms spec.
      - **Published drift fixes (item 3), desk check:** TTT on the target's own outputs (AngelSpec, FastMTP, Badarayani)
        and the head attending to the target's true prefix states (DFlash's conditioning) are already the chain
        recipe. A target-fed second layer is what S4 tested. In-block dependency (DSpark / DSpine / DFlash2's two-tap
        conv) is a block-drafter feature, so it belongs to LEVERS_A #1 (S1). No new chain screen from item 3.
      - TrainLab REPORT 'WEEK98 S1 DEPTHC RESULT'.
    - Sun 15 06:24, **S5 WALKW = NO EFFECT (MTP Head, 2B sandbox):** training where you draft does not fix the first
      guess after a check.
      - 400 steps from native, vs the deterministic control. Two arms:
        - ww, the walk's check-start anchors at 3x: kept/32 +0.02 (MBPP-50) / +0.07 (HE-30).
        - mw, close calls (margin < 1 nat) at 3x: -0.08 / -0.10, leaning worse.
      - The sandbox shows EvalSuite's gap: depth 0 at check starts is 84.2 % vs 88.1 % over all anchors (MBPP), and
        87.3 vs 93.5 % (HE).
      - Training weight doesn't move the check-start gap (84.4 % with 3x weight).
        - Depth 0 already reads the 125B's true state. What it lacks is the 125B's own pass on the bonus token.
        - Exact options left there: a sequential 125B row on the bonus token (priced as a ~0.6 ms/layer partial pass),
          or more rows per check if a 64-row tree is cheap.
        - Lossy close-call acceptance is the direct one.
      - TrainLab REPORT 'WEEK98 S5 WALKW RESULT'.
- Sun 15 01:05: **Leap C2 DENSE32 PASS** (b7cd3e7). D32's 32-row kernels (mr2 / mg2) are bitwise with 2 x 16 for all 12
  shapes, padding-independent. Dense per check is 18.30 ms at 16 rows and 22.21 ms at 32 (R 1.21; floor 12.3). MLX's
  own 32-row call differs for small-N shapes, which stay 2 x 16. Action: build REF2's 17-32-row verify path (v32.py
  scaffold, bits only). Next: P1 OVERLAP.
- Sun 15 01:05: NewIdeas + MTP Head briefed for the night. NewIdeas: research pass + merged ranked list + screen
  hand-outs. MTP Head: depth done properly, the CARRY, the published drift fixes, t30b HE+ cells for MoonShots.
- Sun 15 01:0x: **Sandbox98 KEYFRAMES = BETWEEN** (86c8d56). MBPP-50 23-30: chain 55.42; refresh every row 80.61;
  k = 4 71.56 (64 % of the gain), k = 8 65.27 (39 %), k = 16 56.64 (5 %).
  - With the TRUE state as the keyframe the shares are the same, so **the limit is the head's CARRY**: a refreshed state
    decays within ~4 depths.
  - Refresh frequency alone can't make a stale-carrying head cheap. What's needed is a cheap fresh-state source per
    token, or a head whose carry holds 30 deep.
- **Breakthrough: GOAL B lossy check priced** (01:1x, LOSSY_CHECK_PLAN.md; Breakthrough f4d546b / 37229bd; CPU only).
  - Pre-registered pooled verdict (code + IFEval + multi): **KILL**.
  - **Code-scoped:** acceptance-weighted bit-plane tiers (4-bit / MSB-3 / drop by reach-weighted importance) give
    23.0 ms (today's drafter) / 24.6 ms (98 % drafter) at the bandwidth floor, and 27.3 / 29.2 ms at a realistic 84 %
    kernel efficiency. Predicted code flips 1.2-1.3 %. The pooled-safe version is 32.3 / 38.5 ms.
  - Byte model: an exact 32-row check is 13.68 GB / 48.9 ms. 25 ms means <= 7.0 GB, and the DENSE part (3.44 GB) is
    half the budget and stays. Experts must fit <= 24-35 4-bit-equivalents per layer.
  - Correction: MSB-3 behaves like an error of 0.08, not 0.02, so HOBBIT-style low bits are worth about half the naive
    estimate.
  - So 25 ms needs ~92-98 % of bandwidth or ~0.7-1.0 GB less dense reading. **Chat binds** (IFEval flips 5-10x more per
    unit of missed energy), which means a code-scoped gate or a code-only switch.
  - Prerequisite: the exact single-pass 32-row check at near-floor kernels first; the byte levers pay only after that.
  - Addendum (Breakthrough 4aa247c), leave-one-out bands: **screen 1c FIRST**. Three tiers (4-bit / MSB-3 / drop, s 0.98)
    give 25.6 ms at the floor / 30.5 at 84 %, with code flips 1.08 [0.99, 1.15] against the 1.34 level, inside in every
    refit, and it's the simplest build. 1b: 1.21 [1.06, 1.28]. 1a: borderline.
- EvalSuite's lossy gate is BUILT and validated (826a4c2): the scorer reproduces CERT's lines exactly. Discordance 4.4 %,
  so tier 1 catches -3 pt pooled 78 % of the time (-5 pt ~always) with ~3 % false FAILs. Runner 64 dry runs are OK.
  Asked for: a CODE-scoped tier 1 (HE+ + MBPP, plus an IFEval no-harm check with the switch off). For reference, the
  shipped lossy v1' stack scores -2.20 pt pooled (p 0.077) on this panel.
- **MoonShots' top 3 GOAL A levers (01:0x, LEVERS_A_MOONSHOTS.md v0; all training-free):**
  1. BORROWED DEPTH: Qwen3.5-2B / 4B (on disk, the 125B's exact tokenizer, 248,077 ids) as the composer. Ridge from
     its layers {6,12,18,24} -> wd on the 125B BANK, scored by bank_proxy (PASS >= 85 predicted at MBPP 23-30, then
     REPLACE). Owner: MTP Head at 01:05.
  2. KEYFRAMES: the composer runs every k tokens and the head carries in between (k = 4 / 8 / 16). Owner: Sandbox98
     now.
  3. kNN STATE MEMORY: real target states retrieved by suffix n-gram (one neighbour vs the mean of 8). Owner:
     Sandbox98 now.
  - Prior art: none found for the three. Closest: EAGLE / FastEagle, SPD, NEST / REST / SENSE, KVShot. Mirror-SD
    (Apple, 2510.13161) runs a composer on the NPU beside the GPU, which may be the way to make lever 1 cheap.
- **Leap's pillar 2 + 3 plan (00:5x, Leap 6300098; data/week98/plan_1115.md + pipe_price.txt; desk only):**
  - **The floors moved up** with BigMoE's measured union (72.5 / 108.5 distinct experts per layer): 38.8 ms (16 rows)
    and **52.0 ms (32 rows)** at 280 GB/s. The 16-row check today is 1.6-1.7x its floor.
  - At 98 % a 32-row check keeps 23.8, so 450 needs a cycle <= 52.9 ms. That is 0.9 ms of room at 280 GB/s, which must
    also hold the commit replay (3.9-5.4 ms) and idle seams (2.5-3.6).
  - **The max at the floor with free drafting is 458-491; the realistic EXACT ceiling is ~370-410 tok/s.**
  - **The drafter's own cost matters:** today's chain head reads 5.4 GB per 32-step block (>= 19 ms even perfectly
    overlapped).
    - Best case at a floor check: chain head ~241-247 tok/s; a ONE-PASS BLOCK DRAFT (~10 ms, 0.52 GB) ~355-374.
    - **So Goal A must deliver 98 % to 30 deep with a CHEAP drafter (one-pass / block, few bytes).**
  - Pipelining hides ~nothing on one GPU: two streams at 32 rows were 1-2 % faster; q = 0.98^32 = 52 % of early drafts
    are usable. It's second-order.
  - Deeper checks (48-64 rows) could reach 499-521 at the floor if the drafter holds 98 % that deep, but they need a
    pool > 256 (more than 45 GB).
  - Sunday screens: P1 OVERLAP, C2 DENSE32 (exact 32-row dense path), P2 CHAIN-BYTES, P3 BLOCK-COST, C3 GDN / ATT32;
    C4 COMMIT-B (a build).
  - a45 re-armed for Sun 21:00 -> Mon 01:00 (it needs 47 GB free).
  - **ASK for the drafter owners: the head's accuracy at depths 31-63**, if any read-out walks that deep.
- EvalSuite's LOSSY_GATE.md (00:5x; EvalSuite 658fca1 / 1f9e82b):
  - Tier 0: a 15-min kill-only pre-screen (REPLAY agreement + a 40-item panel).
  - Tier 1: a ~45-min fast gate at 45 GB: HE+ 164 + MBPP 100 + IFEval 100 paired; pooled delta > -3 pt as a SCREENING
    line only.
  - Tier 2: the card gate.
  - **BigMoE, tomorrow: build the R57_REPLAY screen mode per LOSSY_GATE.md section 9** (teacher-forced 32-row checks of
    reference rows; per-position accept / top-1 / margin).
- Sub1 (87 % at 00:38) must be used up by its 04:00 reset. From 00:4x: Breakthrough (lossy check pricing,
  LOSSY_CHECK_PLAN.md), MoonShots (outside-the-literature levers for A, LEVERS_A_MOONSHOTS.md), EvalSuite (fast
  lossy gate, LOSSY_GATE.md). Deliverables are due ~03:45.

## NEW SPEED TARGETS (Kamen, Sun 15 Nov ~00:3x, verbatim)
"we have to achieve 450 tps exact as thats the limit at 98 percent and also 1k tps seems extremely doable with
optimisations for lossy". These are at 45 GB on HumanEval-like code; chat accepts less.

| Pillar | Needed for 450 exact | Needed for 1,000 lossy | Owner |
|---|---|---|---|
| 1. Drafter | ~98 % flat to 30: close the drift gap 89.7 -> 97.8+ (HE+) | the same, trained against the lossy target's own picks | NewIdeas + MTP Head + Sandbox98 (screens), BigMoE (125B) |
| 2. Check | a single-pass exact 32-row check at ~49-52 ms (floor 49.1; today the 16-row check is 1.7-1.8x its floor): gate widened to 320 ids, a 17-32-row path, kernels | a ~23 ms check: read < half the exact bytes (low-bit weak experts, acceptance-weighted expert sets, union caps), EvalSuite same-scores gate | Leap (kernels / paths), BigMoE (32-row spec), Breakthrough / EvalSuite (lossy + gate) |
| 3. Drafting cost | ~3 ms left for 30 guesses: pipeline drafting with the check (draft block N+1 while block N verifies) or draft the block in one pass | the same | Leap (pipeline), drafter owners (block drafting) |

Drafter levers from the research report, on top of our TTT:
- target state in every draft layer (DFlash);
- in-block dependency heads (DSpark / DSpine / DFlash 2's two-tap convolution);
- rollouts / self-distillation (AngelSpec, FastMTP);
- training at the check length;
- far more training data / budget (the 2B gained +7 pt from budget).

## The goal
- By Sat 21 Nov, know the method that gets the drafter to >= 98 % right per guess at every depth up to 30. Held-out test
  numbers must back it; a small-scale proof counts. The full build can take the week after.
- The 125B stays untouched. Only the drafter learns.
- Measure first on held-out HumanEval+ windows (the 23-30 band), then on MBPP and everyday text.

## Where we start (t30b, per guess)
| | depth 0 | depths 23-30 |
|---|---|---|
| HumanEval+ | 97.90 | 89.66 |
| MBPP | 94.01 | 72.13 |

- Data: about +1 pt per doubling (t30a -> t30b). 98 % this way needs over 100x the data, so it isn't the path.
- Capacity: t30x2 NO EFFECT (-0.79 pt). t30x4 and TAPSQ (multi-layer taps) land Sat ~15:10 / ~15:25.
- 98 % at depth 30 means the deep guesses lose nothing against the first guess, and the first guess gains a little too.

## Why it matters (Kamen, Sat 14 Nov 15:2x, in short)
- Be the first: one proven recipe that keeps any drafter ~98 % right 30 tokens deep, usable on every model, exact
  output, ideally "a little engine package". It unlocks huge local models: the 125B in 30-45 GB at ~1,000 tok/s on code,
  and much bigger models streamed from SSD on a big Mac.
- 98 % per guess for 30 deep = about 23 tokens per check of the big model (sum of 0.98^k for k = 1..30, plus the bonus
  token), against 13.83 for t30b today. The other half of 1,000 tok/s is a fast check, about 23 ms per 32-row check.
- "No retraining": the big model is never retrained. Each model's drafter likely needs one short training, because
  drafterwall (11-11) showed training-free arrangements don't move kept by 20 %. Say so plainly.

## How we go fast, and how we show it's general
- Fast inner loop: try levers on a small local target model first, where a drafter trains in minutes, then confirm the
  winners on the 125B. Downloads follow the standing 11-07 rule (20 GB stays free for swap; log the name, source and
  size).
- Before calling it a method for any model, show it on a second model family as well as the 125B.

## Rules
- Every lever gets a screen of at most 15 min, with "result -> action" written first (pacer/RULES.md, 11-14).
- Long runs only confirm, with a stop line at 10-25 % of their steps against the baseline at the same steps.
- Results go to GitHub the same turn, with a one-line note to the orchestrator.
- A lever that changes the output (lossy) needs EvalSuite's same-scores gate before it counts, and Kamen's call to adopt.

> (a quote from the project owner, kept private)
The fresh state h_{s+j-1} gives the head two things its carried stand-in lacks:
- **(a) RETRIEVAL.** Information pulled from the far prefix by attention in all 48 layers, which becomes relevant only
  because of the new tokens (an identifier defined 200 tokens back, an open bracket, a call signature).
  - The fix is to give the drafter lookups: a copy pointer / prompt-lookup over the context, or attention over the
    target's KV cache of the verified prefix, which is computed anyway (prior art: GLIDE-with-CAPE).
- **(b) COMPOSITION.** Deep processing of the new tokens themselves.
  - The fix is a learned transition model that steps the state forward (ROLLOUT).
- **(c) DIMENSION.** Maybe only a few directions of the state matter to the head. Then a small predictor only has to
  regenerate k numbers per step.

Fast tests, mostly no training:
- **T1 COPYABILITY** (BigMoE, CPU): the share of oracle-fixable deep misses whose token appears earlier in the context
  or is predicted by a prompt-lookup n-gram.
- **T2 CONTEXT TRUNCATION** (Sandbox98, 2B): fresh states from the last K tokens only, accuracy vs K. Local -> (b);
  needs full -> (a).
- **T3 SUBSPACE** (BigMoE, REPLACE): true wd in its top-k PCs only, accuracy vs k -> (c).
- **T4 ROLLOUT** (both): is the state's evolution Markov in (h_t, x_{t+1})?
The method follows from which of (a) / (b) / (c) carries the gap.

> (a quote from the project owner, kept private)
- **Feature bank:** run the 125B once to store its states (taps {5,15,25,35,45} + wd, plus the top-32 log-probs on
  HumanEval+) for ~500 train answers + MBPP 100 + the HumanEval+ windows. Most screens then never touch the 125B, its
  heat or its SSD (BigMoE, after COMBO5).
- **Seconds-fast proxy:** score a candidate map or formula by held-out R2 / direction error against the true deep
  state. Day 1's six points are monotone (0.41 -> 41.7 ... 0.82 -> 91.6, threshold near 0.8). Only proxy winners get
  the ~5-min real eval. The proxy filters; it never decides.
- **Paired small evals for screens:** e.g. 30 MBPP windows with a per-window sign test. The full set only confirms.
- **Batches, not single tests:** write 5-10 screens at once, each with its result -> action line, and run them back to
  back in one runner. No agent turn between tests.
> (a quote from the project owner, kept private)
  the control measurably moves; below it, NO EFFECT means untested (Sandbox98's 300-step lesson).
  - Screen at depth ~12 instead of 30 (~2.5x cheaper per step); confirm winners at 30.
  - Train the control plus 2-3 levers in ONE process on shared slices.
  - Continue the base's saved Adam state at its own lr (no drift).
- **Parallel:** CPU screens across sessions while the GPU does one confirmation at a time. Fans up for SSD-heavy
  chains when Kamen OKs.

## Day 1 first: the CEILING test (BigMoE)
- How often is the 125B's own choice a near-tie (the top-1 vs top-2 margin, in bins)?
- What share of t30b's misses at depth 0 and at depths 23-30 sit on near-ties?
- Result -> action:
  - If near-ties alone block exact 98 %, the method must include close-call acceptance (lossy, quality-gated).
  - If the misses are mostly NOT near-ties, they are learnable: focus on what the drafter sees and how fresh it is.

## Owners
- **BigMoE:** the ceiling test, then the GPU screens of depth levers (taps, fresher features).
- **MTP Head (TrainLab, from Sun 01:00):** training levers (D98 owner).
- **NewIdeas (from Sun 01:00):** the lever list with a short prior-art pass (published / untested / new), handing out
  15-min screens.
- **Breakthrough:** close-call acceptance (its delta-0.05 pricing), carried to the 98 % line.
- **MoonShots:** levers from outside the drafting literature.
- **EvalSuite:** the quality gate for any lossy lever.
- **Leap:** keeps 1.2.1 ready for Kamen's word. Engine help when a lever needs it, e.g. a cheap partial trunk pass on
  drafted rows.
- **Orchestrator:** the daily plan, verification, reports to Kamen.

## Day plan
- Sat 14: the t30x4 + TAPSQ verdicts; the ceiling test.
- Sun 15 - Wed 18: breadth, with 15-min screens of every depth lever.
- Thu 19 - Fri 20: the best lever's proof run (with a stop line) and the method write-up.
- Sat 21: the answer for Kamen: the method and its numbers, or exactly what blocks it, with the numbers.

## Log
- Sat 14 15:0x: goal set; charter written (orchestrator).
- Sat 14 15:08: CEILING part 1 (BigMoE e4f9cd3): the 125B's near-ties (m < 0.25 nat) are about 2 % of answer rows,
  < 0.1 nat 0.65-0.82 %; the median margin is 8-9 nats. Ties cost at most ~2 pt, so 98 % is not ruled out. Part 2
  (t30b's misses by margin) decides LEARNABLE vs MIXED.
- Sat 14 15:08: TAPSQ amendment 1 (b44d00f): tap lr 1e-4 -> 1e-5. A zero-init wide input at 10x lr spikes the loss
  (0.35 -> 3.78). The same mechanism likely caused t30x2's / t30x4's early spikes, so the capacity verdicts carry that
  caveat. Candidate: a 15-min capacity retest at 1e-5 against t30b's saved per-step KL.
- Sat 14 15:15: t30x4 verdict HURTS: HumanEval+ 23-30 88.14 % vs t30b 89.66 % (-1.51 pt); MBPP 70.01 vs 72.13; kept
  13.20 vs 13.83. Both capacity arms lose about 0.7 pt per doubling. A clean 15-min retest at branch lr 1e-5 is queued
  after TAPSQ and CEILING part 2.
- Sat 14 15:20: **CEILING = LEARNABLE** (BigMoE 62f5b9e, t30b on MBPP 100, every cell).
  - Tie-limited ceiling: 98.87 % at depth 0, 98.76 % at 23-30, so ties don't block exact 98 %.
  - 75.8 % of the 23-30 misses (21.12 of 27.87 pt) are on rows the 125B was sure of (m >= 1 nat). On very sure rows
    (m >= 4) the first guess misses 0.1 % and the deep guesses miss 16.1 %.
  - 7.9 % of the deep misses fall outside the 125B's top-32: the drafter loses the thread with depth.
  - The tie tax is real: depth 0 misses 54 % of near-ties, about 1.1 pt. Exact 98 % leaves about 0.9 pt for everything
    else.
  - Close-call acceptance adds +0.2 / +0.9 / +1.7 pt at depth (delta 0.1 / 0.25 / 0.5): an end bonus that buys back the
    tie tax, not the week's lever.
  - Caveat: MBPP only (the HumanEval+ windows carry no log-probs).
  - **The week goes to what the drafter sees and how fresh its view is at depth.**
  - Breakthrough: close-call stays a small end piece; its main week job is a depth lever (NewIdeas hands it one).
- Sat 14 15:31: TAPSQ judged KILL (MBPP 23-30: C 58.26, T 57.34; ratio 1.364), but it is held as VOID pending a harness
  check.
  - The control arm itself collapsed: t30b 72.13 -> C 58.26 after 80 steps at lr 1e-5, while depth 0 only went
    94.0 -> 93.1.
  - Suspect: the quick trainer gives the deep depths no signal (the smoke's per-depth KL 24-30 reads 0.0000).
  - Line before any rerun: C within ~1 pt of t30b. The raw-taps test waits for that fix.
  - Note: T's deep-depth KL was 0.43-0.49x C's, which is worth a look once the harness is sound.
- Sat 14 15:32: fans ramp 5900 -> 3750 (-250 / 2 min, fans_down_1114.sh), as told to Kamen.
- Sat 14 15:5x: an outside review (pasted by Kamen) made two fair points:
  - The ceiling is per dataset and decoding mode. Ours is MBPP code at greedy, and it's an estimate from the measured
    near-tie miss rate, not a hard bound.
  - Chat and agent text will be harder than code.
  - So early this week (owner NewIdeas / BigMoE): one quick measurement of the same per-depth curve (conditional
    acceptance, depth 1..30) on chat and agent answers. That means a small capture of ~50 answers plus t30b's eval,
    within the 15-30 min rule.
  - The expert-union cost per check is the other half of the speed goal (BYTES10 / union pricing), not this week's
    target.
- Sat 14 15:5x: the outside review's second round. Its "oracle" test (give the drafter the 125B's true states at the
  deep positions) is drafterwall's Law 2 (RESULT.md, 11-11).
  - With the true state two tokens back, the reader holds 97.7 % on HumanEval+ at every depth (vs 86.6 % at 23-30
    without), and 93.5 % on MBPP (vs 76.2 % by 7-10).
  - So the deep drop is stale information that lives in the 125B's forward pass. The fix must be architectural: a
    learned stand-in carried through the draft, or a cheap refresh mid-draft.
  - On MBPP even fresh states top out at 93.5 %, so the reader must improve there too.
  - Contamination check (orchestrator): all 3,072 n1112 training captures are src dlog-magicoder, split train; none
    come from HumanEval or MBPP. TrainLab's notes list MBPP as held out. Whether the 125B saw the benchmarks in
    pretraining is unknowable: the chat/agent curve is the real test.
- Sat 14 15:53: TAPSQ2 harness valid: C vs t30b at MBPP 23-30 -0.07 pt, band 0 -0.04. T runs; verdict ~16:10.
- Sat 14 16:0x: **TAPSQ2 = KILL with a valid control** (BigMoE 068dc9b).
  - DFlash2's fused multi-layer taps at the context / anchor rows add noise, not information: MBPP 23-30 71.43 vs C
    72.06, KL 1.083x, windows 35 better / 37 worse.
  - A richer view of the verified context doesn't reach the deep rows, which fits Law 2.
- Sat 14 16:1x: the next screens, freshness first (orchestrator's call):
  1. FRESH-H (no re-prefill, after CAPQ): the true fused h fed at every drafted row on the correct chain, tq2_C as the
     control, MBPP 23-30. It goes on if it recovers >= half the gap toward MBPP's lag-2 93.5.
  2. Then a re-prefill with raw taps on every row: per-layer freshness (what a cheap partial trunk pass buys), plus the
     anchor raw-taps item from the same data.
- Sat 14 16:31: **CAPQ = CLOSED** (clean capacity retest). The x2 branch at lr 1e-5 has a clean start (steps 1-20 KL
  1.024x t30b, against the 10x-lr arm's +203 %). Over steps 101-200 its KL is 1.016x t30b's, lower on 46 % (REOPENS
  needed <= 0.95 on >= 60 %). Capacity is closed for the week. The lr confound explained the spikes, not the result.
- Sat 14 16:44: **FRESH-H = FLAT**. The true fused h at every drafted row through a zero-init additive adapter (lr
  1e-5, 80 steps) gives MBPP 23-30 70.09 vs C 72.06 (-1.97). WORKS needed 82.78.
  - Read narrowly: the short adapter didn't learn to use a new input. Law 2's 93.5 % used the reader's native trunk
    state (wd).
  - Next (orchestrator -> BigMoE, no head training):
    1. a TRUE-WD replacement eval, which should reproduce Law 2 in our harness;
    2. a ridge map h -> wd (and later layer-L taps -> wd) fed as a replacement, giving the per-layer freshness curve
       without adapter training.
- Sat 14 16:52: **LEAD: REPLACE-H = CARRIES** (BigMoE cc53cb5, t30b unchanged, no training, MBPP 100).
  - At 23-30, t30b's own chain gives 72.13. With the stand-in replaced by M(true h), a ridge from DFlash2's fused h to
    wd (R2 0.80), it gives **91.20**. With the true wd (the gate, which reproduces Law 2) it gives 93.63.
  - The M(h) bands are flat with depth: 94.01 / 91.78 / 91.60 / 91.46 / 91.34 / 91.28 / 91.20. Kept per 32 goes
    6.92 -> 10.55 (oracle 12.76).
  - Freshness is the lever, confirmed. Open: its cost, since h needs trunk layers up to 45 per drafted token.
  - Next: the re-prefill (raw taps) gives the per-layer freshness curve ({5}, {5,15}, {5,15,25}, all) (BigMoE).
    Leap prices a partial trunk pass per L at 45 / 24 GB, with the break-even kept per check.
- Sat 14 17:01: **Leap's trunk pricing** (5bc983d, data/week98/README.md, trunk_price.txt).
  - A fresh partial trunk pass PER drafted token doesn't pay: ~0.62 ms per layer even for 1 row (CPU graph build).
    45 GB L 5 adds ~101 ms per check and needs kept 27.8 HE+ / 13.9 MBPP to break even; L >= 15 is impossible.
  - BLOCK refresh does: a k-row pass costs about a 1-row pass. At 45 GB, k = 8: L5 +11 ms (break-even 15.3 / 7.7),
    L15 +27 ms (17.6 / 8.8), L25 +43 ms (19.8 / 9.9), L45 +76 ms (24.3 / 12.2). L45 at k = 16 adds +23 ms (17.0 / 8.5).
  - Side note: t30b at a 32-row check (~100 ms at 45 GB) gives ~138 / 69 tok/s, below the shipped 16-row lossy-45 loop's
    164 / 113. The deep drafter also needs the cheap check.
  - Next: the (L, k) block-refresh sweep in the REPLACE-H harness after LAYERS. The cheapest (L, k) over its break-even
    is the method's cost-feasible point.
- Sat 14 17:46: **LAYERS = COSTLY** (BigMoE a856e7e).
  - The gate is valid: the re-prefilled stream holds 93.64 at MBPP 23-30.
  - Ridge(taps) -> stream, replacing the stand-in, MBPP 23-30:
    - {5} 41.66 (R2 0.41);
    - {5,15} 47.47;
    - {5,15,25} 56.14 (R2 0.51);
    - {45} 91.61 (R2 0.82);
    - all five 91.52.
  - Early-layer maps are WORSE than the stale chain (72.13): a poor reconstruction is worse than the head's own guess.
  - The fresh state lives in the deep trunk, so a usable refresh is close to a full pass. Refreshing at L45 every k
    tokens is nearly the same as VERIFYING every k tokens.
  - BLOCK (L x k) is due ~18:15: does L45 at k = 16 clear 8.5 kept (45 GB)?
- Sat 14 17:5x: **Leap closes trunk refresh as a speed lever** (fce1ca7).
  - At L45, a refresh is a verify minus the bonus token and the early exit. MBPP at 45 GB: refreshing every 16 tops
    out at ~89 tok/s (every 8 at ~62), both at the per-row bound of 10.98 kept. Plain 16-row checks with t30b give ~96,
    and the shipped loop 112.7. Tying would need kept >= ~11.8.
  - So the 98 % method must make the drafter predict the deep state itself.
  - Leap's cheapest next screen (sent to BigMoE): ridge [the carried stand-in, the fresh tap {5}] -> wd. Refreshing L5
    every 8 costs +11 ms (break-even MBPP 7.7).
- Sat 14 18:0x: **COMBO5 = CLOSED** (BigMoE 7d6e935). Freshness from the trunk is closed for the week.
  - Ridge [carried stand-in, tap 5] -> wd: R2 0.448. Refreshing every 8 gives MBPP 23-30 65.85, kept/32 6.50
    (break-even 7.7); every depth 25.54.
  - t30b's exact kept per 16-row check is 6.59, so Leap's L45 verdict stands.
  - The head's carried stand-in, read as a wd estimate, has R2 -2.32. It lives in the head's own space, yet the head
    drafts 72 % at depth from it.
- **The week's method, from here: a learned better stand-in.** The drafter predicts the deep state itself. The
  oracle (a fresh wd, 93.6 % flat with depth) shows the head can use a good state.
  - The BANK (runtime-path taps + wd + top-32) and the PROXY (R2 -> predicted 23-30) are due ~22:00 (BigMoE).
  - The proxy filters only wd-space predictors. Learned stand-ins in the head's own space are judged by the drafting
    eval.
- Sat 14 18:03: Leap recorded the trunk-refresh closure (4448350) and withdrew its microbench. From 18:0x Leap has
  route 3, the cheap check: cutting the per-check graph-build overhead (0.62 ms per layer even at 1 row). Exact stays
  bit-exact; GPU timings after the BANK or overnight.
- Sat 14 18:1x: **the BANK is building** (BigMoE b889c90, 524 files, end ~20:05).
  - The runtime path reproduces the n1112 and MBPP captures bit for bit (prompt AND answer rows).
  - HumanEval+ warning: TrainLab's d16w windows do NOT match the runtime path (stored wd differs 10 % on prompt rows,
    21 % on answer rows). Every HE+ number so far (t30b 89.66 at 23-30, all TTT30C HE+ lines) came from d16w features.
    After the build, BigMoE re-scores t30b and b5 on HE+ with bank (runtime-path) features. Quote HE+ headlines with
    that caveat until then.
  - PROXY: bank_proxy.py (3e462ea). Its calibration is redone on the bank after the build.
- Sat 14 18:1x: **Leap cheap check, screen 1** (518732e, data/week98/cheapcheck.md).
  - The 16/32-row verify is GPU-bound (check ~ 30 + 2.47 ms per row at 45 GB), so graph caching saves ~0. Closed.
  - The 1-row 0.62 ms per layer was GPU time; the CPU build is ~0.21 ms per layer.
  - Exact bandwidth floors at 45 GB: 15.9 ms (1 row), 36.5 (16), 49.1 (32). A 32-row check today is ~90-97 ms
    (estimated).
  - The exact path to ~60 ms at 32 rows is GPU kernels: experts to floor, dense ~20 -> 13-14 ms. The ~23 ms target is
    lossy-only (FAST track).
  - **Rule of thumb:** a deep drafter beats plain 16-row checks only when kept/32 >= ~1.35-1.4 x kept/16. t30b on MBPP
    is at 1.05; a 98 %-per-guess drafter would be at 1.69. The drafter decides the 32-row question; a faster check
    speeds both loops.
  - Next: Leap measures a real 32-row check (dense / experts / misses) after the BANK, under a quiet lock (~20:30).
  - **Engine fact (Leap 18:3x): no exact single-pass 32-row check exists today.**
    - The sync-free gate carries at most 185 ids per routing row, i.e. 18 rows at top-10 (bigmoe/gated.py MAXK <= 185).
      Above 16 rows a check falls to the slot / prefill path (4-bit on every rank, outside REF2-T's certified 1-16 rows).
    - So an exact 32-row check today is 2 x 16 gated passes, about 2x a 16-row check. No drafter pays at 2x: 98 % gives
      1.69x.
    - The ~60 ms exact 32-row check and the 1.35-1.4x rule both assume a gate widened to 320 ids in one pass. That is
      item 1 of BigMoE's 32-row engine spec, ahead of the kernel work.
    - Tonight's screen times 16 rows, 2 x 16 and 1 x 32 (slot), cold and warm, plus a dense-only pass, on exact-45 via
      the R57_COPYPROBE hook (the loop stays byte-identical).
- Sat 14 18:5x: **method catch (Sandbox98, results/lr_screen2.md):** the accept-until-fail training loss is
  survival-weighted, so a better-chaining arm can show a HIGHER loss. Judge arms by read-outs (bands, kept/32) or
  per-depth KL on fixed rows, never by cross-arm AUF loss.
  - **CAPQ's CLOSED rested on the loss ratio alone, so it is not certain.** A read-out of t30x2c@200 vs a matched
    t30b@200 on MBPP 100 is queued with BigMoE after the BANK and Leap's screen.
  - The stop-line rule in RULES.md is fixed to use a quick read-out.
  - Sandbox98 status at 19:00: Qwen3.5-2B + its native MTP head (121.7 MB fetched). lr screen -> base lr 1e-5 (native
    kept/32 3.18, 23-30 only 1.7 %). The base drafter is training; the gate evals follow.
- Sat 14 19:51: **Sandbox98 validity gate PASS** (~/Sandbox98/results/gate_report.md). Qwen3.5-2B with its MTP head
  trained 30-deep (s30) reproduces the 125B's day-1 pattern.
  - MBPP 23-30: chain 57.32 vs depth 0 88.54; true W at every depth 88.35, flat. HE: chain 70.31; true W 93.71.
  - Maps from early taps are poor (L2 47.75, L12 59.19); late taps are good (L22 87.30, R2 0.93).
  - **New:** the head tolerates UNBIASED noise far better than map error at the same R2. True W + noise at R2 ~0.75
    gives 85.60, at ~0.51 gives 82.66; a ridge map at R2 0.44 gives 59.19.
  - So a learned stand-in needs noise-like errors, not exactness. The PROXY's R2 calibration must be per predictor
    family.
  - Screens now run on the sandbox (control + the t30f feature-loss transfer first), minutes each.
- Sun 15 00:2x: **RESEARCH REPORT** (~/FableGod/deepdraft/reports/Deep drafting and fast verification.md, 4 researchers
  + writer, cited).
  - **Check:** the accept rule is proven optimal (block verification +5-8 %). Nobody has pushed MoE check COST to a
    limit; the only floor is the expert union, verified in Lean (Lan 2026).
    - Cheapest published: 32 rows = 1.62x one row (3B-active MoE, A100). On llama.cpp Metal, speculation was a net
      slowdown.
    - Ours: the 16-row check is 1.7-1.8x its 36.5 ms floor.
  - **Depth:** depth is the weakest lever; drift training matters more. Published accuracy ends at guess 15 (~86 %, a
    vendor blog). OUR curve (97.9 -> 89.7 at 23-30 on HE+) is already deeper and higher than anything published (not a
    controlled comparison).
  - **The combination is new as a result, not exotic as a method.** On this Mac an exact 32-row check cannot beat
    49.1 ms. That caps a flat-98 % drafter at ~485 tok/s and a perfect one at ~650.
    - **Exact 1,000 tok/s is impossible on the 64 GB M5 Pro.** It needs lossy / low-bit checks reading < half the
      bytes (benchmark-gated), or more bandwidth (the report: Apple lists the M5 Ultra at 1.2 TB/s).
    - Exact ladder (HE+, check time only): ~160-170 today -> ~265-285 (drift closed + a single-pass 32-row check) ->
      ~365-420 (dense-class kernels) -> ~470 (the floor).
  - **Drafter levers to screen** (on top of our TTT):
    - DFlash-style target state in EVERY draft layer;
    - in-block dependency heads (DSpark, DSpine, DFlash 2's two-tap convolution, ~80 % of a 15-layer gain);
    - rollouts / self-distillation (AngelSpec, FastMTP);
    - training at the check length.
  - Publishable gaps we already fill: depth-30 curves with a true-state ceiling; 16-32-row check costs on a 512-expert
    MoE on Apple silicon.
- Sat 14 23:58: **Sandbox98 FAIR DEPTH = NO EFFECT** (70abb98 / 5da3dcc). 2-layer vs 1-layer heads, both from the
  native start, 400 steps x 30 deep: MBPP-50 23-30 48.42 vs 48.37 (+0.05, CI -1.04..+1.16); depth 0 -0.69; HE -1.14
  n.s.
  - 113.2M vs 60.8M params; 3.96 vs 2.54 ms per drafted token. The extra layer barely grew (|w| ~0.0025 at 3e-5).
  - **Training budget beats depth here:** the same 1-layer head gains +7.1 pt at 23-30 from budget alone (400 -> 1,889
    steps).
  - Reading: T2's +14 pt is PRETRAINED composition. A zero-start drafter layer trained on a few hundred answers doesn't
    grow into it (added late -2.5, from the start +0.05).
  - Caveats: one extra layer, one lr, a zero start, an under-trained budget.
  - **T30D (bolting a zero-init layer onto t30b) is ON HOLD.** The 2B says it hurts or does nothing at short budgets.
  - Routes left for the 01:00 sessions, after the prior-art pass:
    (a) pretrained composition: drafter layers from a pretrained model (the target's own top layers, Kangaroo /
        LayerSkip-style, which cost per token; or a small pretrained LM of the same family conditioned on the target's
        fresh state);
    (b) far more drafter training data / budget;
    (c) whatever the research report (~00:30) shows.
- Sat 14 23:4x: **Sandbox98 DEPTH (bolted onto the converged s30) HURTS** (d3b5a60; 400 steps, pool2, re-annealed; the
  control holds at -0.50).
  - deep2 (a 2nd layer, a no-op at step 0) -2.48 at MBPP 23-30 (significant), HE -2.04.
  - tie2 (one layer applied twice) -2.38.
  - **distill -0.48 n.s.** Its earlier +3.1 / +8.1 was drift protection only, so distill is CLOSED as a gain.
  - Reading: T2's gain is PRETRAINED 24-layer composition. A zero-start layer on a converged head only perturbs the
    chain; this was not a fair depth test.
  - Draft timing on the 2B (shared GPU): 2.54 ms / token with 1 layer (lm_head + argmax 1.23), +~1.4 ms per extra layer.
    The draft chain is a real part of each check's cost; the pricing tables must include it.
  - Next (~15 min): the FAIR depth test, 2-layer vs 1-layer heads both from the native start, same budget. Prior art:
    FlexDraft (depth helps), Mixture of Attentions (fixed-budget depth hurt).
> (a quote from the project owner, kept private)
  The prior-art pass we skipped today (orchestrator's miss against the research-first rule), quick search only:
  - The depth-wise decay and its cause ARE published: "compounding feature drift" (later draft steps rely on the
    drafter's own shallow states).
    - Speculative Pipeline Decoding, arXiv 2605.30852.
    - FastMTP 2509.18362: vanilla MTP trained single-step.
    - DSpark 2607.05147.
    - Adjacent Causal Injection 2609.36173.
  - Deeper drafters help: FlexDraft 2605.20022 defaults to 10 draft layers. Mixture of Attentions 2410.03804: EAGLE with
    more layers at a fixed parameter budget did worse. FastEagle 2509.20416: EAGLE-3 per-position acceptance ~0.78-0.81
    on MT-Bench.
  - Future Lens (2311.04897): a single hidden state anticipates t+2.. only partly (~48 %, GPT-J), the same as our
    STALE result.
  - NOT found: any acceptance curve out to 30 draft positions, or ~98 % per position deep.
  - **So "a deeper drafter" is known, not a breakthrough.** Tonight's findings are general (two models agree) but
    largely match the literature. The open part is still 98 % at depth 30 at a drafting cost that pays.
  - The check-cost side (the MoE expert union) is a busy 2025-26 field, from a quick search:
    - MoESD (Huang et al. 2025): expected distinct experts for n tokens.
    - "The Limits of Speculation" (arXiv 2609.22156).
    - "Expert Bytes, Kernel Bandwidth and the Best Draft Length" (2610.34440), the closest to our own pricing work.
    - Cost-Aware SD (2607.12696); AcceptMoE (2608.02989); Adaptive Verification (2605.00342); MoE-Spec / EcoSpec /
      EVICT.
    - Temporal correlation cuts unique experts 20-31 %.
    - Offloaded MoE + SD: Specter (2511.14102), SpecMoEOff, BigMoMo (2609.14643, mobile, <= 30B), MoE-SpeQ, 2508.21706.
  - Rough speed table (45 GB, HE+ code, per 32-row check incl. ~7 ms of drafting):

    | per-check cost | today's drafter (13.8 kept) | 98 % drafter (23.3 kept) |
    |---|---|---|
    | ~100 ms (today, est.) | ~130 tok/s | ~220 tok/s |
    | ~60 ms (exact plan) | ~205 tok/s | ~350 tok/s |
    | ~35 ms (lossy) | ~330 tok/s | ~550 tok/s |
    | ~23 ms (lossy) | ~460 tok/s | ~780 tok/s |

    The two halves multiply.
  - **01:00 order:** NewIdeas' prior-art pass FIRST (the best published per-position accuracy at depth; every method
    beyond depth: pipelined / parallel / causal-injection / tree / target-KV ones), then pick the bet. T30D still
    measures how far depth goes on the 125B and at what cost, but it's not the method by itself.
- Sat 14 23:2x: **T3 SUBSPACE (125B): no small-k shortcut** (BigMoE 7255e24). MBPP 23-30, the true wd kept in its top-k
  PCs (mean / stale fill):
  - k = 16: 29.9 / 27.6;
  - k = 64: 50.1 / 44.4;
  - k = 256: 69.9 / 64.8 (about the chain's level);
  - k = 1024 (87.7 % of the variance): 86.9 / 85.4;
  - full: 93.6.
  - A predictor must regenerate > 1,000 dimensions; the last 12 % of the variance still holds 6.7 pt.
- **THE NIGHT'S THEORY, all four tests agree (T1-T4):**
  - the head reads a fresh state well;
  - context-free steppers fail;
  - lookups aren't the gap;
  - few directions don't suffice;
  - a deep stack over a 64-256-token window does.
  - **The missing piece is deep composition of the drafted tokens with local context, regenerating a near-complete
    state: a DEEPER drafter.** Sandbox98's depth test (~23:50) is the first direct test; T30D is the 125B version.
- Sat 14 23:2x: **T30D handoff.** BigMoE wrote the build spec ("T30D BUILD SPEC" in ~/BigMoE/HANDOVER.md, committed)
  instead of building it tired. The spec:
  - per-extra-layer P_e copied from t30b, reusing the frozen experts and HC, each with its own context KV;
  - a ReZero gate a_e = 0, so the smoke is EQUAL to t30b;
  - three optimizer groups (the head on t30b's Adam re-annealed; P_e fresh; a_e ~1e-3, set by a smoke);
  - read-out stop lines at 420 / 1,050 steps.
  - **Owner: MTP Head (TrainLab) at 01:00 if Sandbox98's depth verdict (~23:50) is positive**, else BigMoE tomorrow.
- Sat 14 23:1x: **T1 COPYABILITY (125B) says it's composition, not lookups** (BigMoE; MBPP 100 per cell).
  - FIXED 23-30 (chain misses, oracle hits): 63.0 % seen earlier, 15.5 % far, prompt lookup predicts 12.2 %. HIT23
    (chain hits): 82.8 % seen, lookup 38.9 %.
  - The chain already gets the copyable tokens; what a fresh state fixes is less copyable than average. A perfect lookup
    adds at most ~2.6 pt at 23-30.
  - **Both models now agree: COMPOSITION DEPTH.**
  - Next on the 125B: BigMoE pre-registers t30d (t30b + 1-2 zero-init residual layers, continued Adam re-annealed,
    read-out stop line) to start when the sandbox's depth test is positive. T3 (subspace k) is rerunning after a bug,
    ~12 min.
- Sat 14 23:04: **T2 CONTEXT TRUNCATION (2B) = MIXED, and a DIRECTION** (Sandbox98 ed32994; MBPP-50, 3,857 anchors,
  s30 unchanged).
  - 23-30: chain 55.42 | the 2B over ONLY the drafted tokens (K = 0) 69.62 | K = 16 75.47 | K = 64 80.61 | K = 256
    86.25 | full 87.71.
  - Band 1-2: chain 78.1 | K = 0 17.5 | K = 64 79.4 | K = 256 86.8 | full 88.2.
  - **Theory: the deep gap is COMPOSITION DEPTH.** The head already attends to every true prefix state, yet its ONE
    layer composes the drafted tokens far worse than a 24-layer stack with no prefix at all (+14 pt at 23-30). Context
    adds ~+5 pt per 4x window; the knee is 64-256 tokens. Retrieval of the far prefix matters little (K = 256 is -1.5).
  - On the 125B, capacity tests added WIDTH (no effect / hurts). DEPTH is untested.
  - **Next: a DEEPER drafter** (Sandbox98, ~1 h): the head's layer applied 2x (tied), then 2-3 untied zero-init residual
    layers, TTT on pool2 re-annealed, against an in-batch control. Also time one drafter step at 1 / 2 / 3 layers.
- Sat 14 22:5x: **ROLLOUT on the 2B = KILL** (Sandbox98 053eeaf), transfer match #5. Linear 20.86, +MLP 22.32, vs
  chain 57.32 (stale 15.50, oracle 88.35). Even one step loses to the chain (66.8 vs 78.2 at band 1-2).
  - **Theory:** a "no change" step has direction error 1.005 there, so consecutive top states are nearly orthogonal. The
    next-token state is RECOMPUTED from context at every step, not a slowly drifting summary. No context-free stepper
    can track it.
  - Freshness must come from a context-aware computation: the drafter's own attention chain trained better (distill =
    the provisional lead), cheap lookups (T1), or the target's prefix KV.
  - Deep accuracy tends toward the drafter's standalone agreement with the target given true context.
- Sat 14 22:4x: **ROLLOUT = PARTIAL** (BigMoE a8adf45; MBPP 23-30, t30b with the stand-in replaced).
  - STALE anchor (wd_s at every depth): 16.00 (1-2: 54.08, 3-6: 25.15), kept 2.87.
  - A linear PCA-512 transition rolled out from the true anchor: 23.08, kept 2.97.
  - The chain gives 72.13; the oracle 93.63.
  - Direction error: rollout 0.59 (1 step) -> 0.79 (30), stale 0.85 -> 1.04. k = 1024 doesn't help, so the linear step
    is the limit.
  - Read: composition of the new tokens is most of what depth needs. The head's own learned carry (72 %) is far better
    than linear external dynamics (23 %). The 72 -> 94 gap is what the carry still misses. T1 / T3 / T2 decide whether
    that is retrieval, composition depth or dimension.
> (a quote from the project owner, kept private)
  achieve in principle what we are chasing, not lo0ng tests".
  - The sandbox's long-screen control drifted (-7.2 pt; overfitting a 120-answer pool), so its long screens are
    provisional and stopped.
  - **ROLLOUT** (125B on the BANK: BigMoE; 2B: Sandbox98, as a transfer pair; no head training): a learned
    state-transition f(W_t, emb(tok_{t+1})) -> W_{t+1} (linear PCA ridge, then a small MLP) rolled out from the fresh
    anchor along the drafted tokens, then REPLACE eval.
  - Baselines: the chain, the oracle, and a stale anchor. +5 pt over the chain at 23-30 = the principle works.
- Sat 14 22:1x: sandbox -> 125B transfer: 4 matches so far (the depth drop, the flat fresh-state oracle, early vs late
  maps, feat NO EFFECT). Transfer checks asked for (TRANSFER.md): capacity x2 at the head lr (the 125B's depth-growing
  +1.2 with a worse loss), and anchor fused taps.
  - Parallel side-by-side runs buy little: one sandbox process keeps the GPU ~84 % busy. Use multi-arm runs.
  - Rule: sandbox wins are filters. Confirm on the 125B before trusting them.
- Sat 14 22:0x: **CAPQ read-out** (BigMoE 24e7758): CLOSED stands by its line, but the loss-only reading was wrong in
  kind.
  - t30x2c@200 vs t30bc@200 on MBPP 100, by band: 0 -0.09, 3-6 +0.20, 7-10 +0.44, 11-14 +0.85, 15-22 +1.00, 23-30 +1.20
    (58.18 -> 59.39). Kept/32 +0.043; REOPENS needed +1.0 AND +0.1.
  - The AUF loss read 1.6 % WORSE while the deep read-out is better and the gain grows with depth: survival weighting,
    confirmed.
  - Capacity buys a small depth-growing gain (~+1 pt at 23-30 in 200 steps). It's tiny next to the oracle (+21.5 MBPP /
    +8.1 HE+); not the week's bet, though it may compound with stand-in work.
  - Harness: re-running t30b's first 200 steps today doesn't reproduce its saved losses bitwise (max abs 0.143). Paired
    arms must run on the same code and day.
  - BigMoE's night chain is done (BANK + PROXY, routing Step 0 to Leap 64c9d2d, CAPQ). The capture stays stopped.
- Sat 14 21:33: **Sandbox98's three learned-stand-in screens** (36e531d; 300 steps at lr 2e-6; MBPP-50 / HE-30 23-30
  against a control that held, 55.96 / 68.34):
  - distill (KL to the head's own true-state output): -0.13 / -0.15, nothing moved;
  - headfeat (stand-in pulled to the true state in head space): -1.50 (p 0.013) / -0.45, a real small LOSS;
  - noisehead: n.s. at depth; depth 0 +0.29 (p < 0.05).
  - **Budget caveat:** the control itself moved only +0.5 pt from s30 in 300 steps, so distill is UNTESTED, not
    refuted. Rule: training levers need a budget at which the control measurably moves.
  - Next (now, Sandbox98): re-train s30 with its Adam state saved, then 1,000-step screens continuing that state at the
    base lr (distill first).
- Sat 14 21:2x: **Leap screen 3, lossy-24 part** (4d9322e, data/week98/cc3/README.md; High Power Mode; within-run).
  - The live 13-row check is 81.5 ms. Its split at 13 rows: dense 23.9 + experts/misses 53.7 (130 misses, 23.9 ms SSD).
    Dense at 13 rows is CPU-bound (18 of 23 ms is graph build).
  - Dense-only 32 rows: 181 ms. CORRECTED (Leap 21:3x, 76edc07): this is lossy-24's DENSE-N-RAM unpacking every dense
    weight per call above 16 rows (counter: 2592 = 24 passes x 108 weights), not GDN / attention. Fix candidate: in
    verify scope, run 17-32 rows as two 16-row native calls (exact per row, no unpack); not built until the 24 GB gate
    carries 32 rows. exact-45 (no DNR) will show the real GDN / attention cost above 16 rows. A real 32-row pass on the
    slot path: 2.81 s, mostly graph build.
  - At 24 GB a 32-row check costs 34.5x a 13-row one. Even with a widened gate, the non-expert 32-row path alone is
    2.2x a whole 13-row check. The 24 GB line keeps <= 13-row checks.
  - Deep drafting needs real engine work for 17-32 rows: (1) a gate widened to 320 ids, (2) a GDN / attention path for
    17-32 rows (now a time cost, not only memory), (3) kernels. That's next week's build, BigMoE's spec.
  - exact-45 waits for 47 GB until 01:00.
- Sat 14 21:1x: **BANK + PROXY ready** (BigMoE 5786e5e; usage in ~/BigMoE/data/deepdraft/bank/PROXY.md).
  - Ridge calibration, MBPP 23-30 (real REPLACE): tap5 46.11, 5+15 51.16, 5+15+25 59.69, tap45 91.96, all 91.88,
    oracle 93.63. The direction error is the primary measure.
  - **HE+ on the runtime path (the headline set):**
    - t30b's own chain gives 89.65 at 23-30, so the headline holds (d16w gave 89.66).
    - **The true stream at every drafted row gives 97.95 at depth 0 and 97.76 at 23-30, FLAT; kept/32 goes 13.77 ->
      20.90.**
    - The tap45 ridge gives 96.40, kept 18.06.
  - **So the WEEK98 target on HE+ is concrete:** a learned stand-in that reaches the fresh-state oracle (~97.8 % flat),
    plus a slightly better reader (+0.2-0.3 pt), meets >= 98 % per guess at depth 30. The tie ceiling is ~98.8 %.
  - Caveat: HE+ ground truth is off the runtime path in 11 of 24 windows (0.3-2.7 % of rows).
- Sat 14 21:06: **Leap screen 3 got no numbers.**
  - lossy-24 exited rc 1 at start-up: the 24 GB loop refuses R57_ROWCAP with R57_WARMSPLIT. Leap's bug; the runner had
    been tested only with fakes.
  - exact-45 was skipped on its gate: NO-SWAP 34 vs 47 GB. The rest is anon ~29 GB of apps and sessions (Safari
    WebContent ~6 GB, PyCharm 2 GB, idle Claude sessions).
  - Earlier, the quiet ticket stalled 17 min (20:40-20:58) behind BigMoE's yielded-but-held share lock.
  - Fix: a 24 GB probe redesign, a dry run against both real loops, b24 queued after CALIBRATION-DONE, and a45 only when
    NO-SWAP >= 47 on two reads (window to 01:00, else tomorrow night).
- Sat 14 20:26: Sandbox98 tapmlp (item 1, a zero-init MLP from the true tap-12 residual added to the stand-in) = NO
  EFFECT by its +2 pt line. Small real gain: MBPP-50 23-30 +0.61 pt (p 0.002), HE-30 +0.82 (p 0.016). Tap-17 is a
  breadth item: it's the trunk-refresh route, closed on cost.
  - Sent to Sandbox98 for tonight (its lines): (1) ORACLE DISTILLATION, where the true-W drafter is the teacher and the
    chain's head output is matched to it at every depth; (2) a state target in the head's fused input space; (3) a
    noise-robust head + stand-in predictor.
- Sat 14 20:21:25: **BANK-DONE**: all 524 files built, rc 0. Every n1112 / MBPP file is bit-exact against the source;
  the 24 HE+ windows are on the runtime path (they differ from d16w by design). Calibration started 20:21:46 (proxy
  rows: 31,805 train / 9,984 val). The fans ramp back to 3750 from 20:21:42.
- Sat 14 20:13: **Sandbox98 ready** (git 6cb92ae; `./screen.sh NAME LEVER "LINE"`, 5-min paired screens, about +/-0.5 pt
  resolution).
  - First transfer check: feat (t30f's feature loss) = NO EFFECT, as on the 125B (MBPP 23-30 -0.26 pt, CI -0.71..+0.16).
  - Findings:
    1. The RMS-normalized direction error, not R2, orders all 7 predictors by read-out. It becomes the PROXY's
       primary measure.
    2. The AUF loss bias (above).
    3. A converged head continued at 10x its final lr with a fresh Adam drifts the control 8 pt in 300 steps (2e-6
       holds). Every screen checks its control.
    4. An exact two-pass TTT gradient cuts the MLX peak from 5.8-7.8 to 2.5-3.4 GB. It may shrink the 125B's 24-39 GB
       TTT runs; a 15-min screen for BigMoE / TrainLab.
  - Ready: levers/tapmlp.py (item 1, a nonlinear early-tap map).
  - To do (orchestrator): add ~/Sandbox98 to fanwatch's RES (needs a restart), and a private GitHub backup (asking
    Kamen).
- Sat 14 19:16: the BANK was restarted (fresh process, C 160, resuming at file 271), still bit-exact. Kamen turned on
  High Power Mode (~19:15). Pace ~25-30 ms/token, ETA ~20:40. No heat pauses 19:00-20:00.
- Sat 14 19:00: the BANK has 197 of 524 files, all bit-exact except the 24 HumanEval+ windows (expected, since d16w ≠ the
  runtime path). Greedy holds 97.3-100 % there. ETA ~21:00 (the sandbox shares the GPU). Swap 7.15 GB (start 6.51).
> (a quote from the project owner, kept private)
  second-model generality proof. It starts only if he starts it.
- **What that leaves for the week** (for NewIdeas / MTP Head at 01:00, as 15-min screens):
  1. A nonlinear or trained map from early taps to the deep stream. Linear explains only 41-51 %. If an MLP from
     <= 25 layers reaches R2 >= ~0.8 and REPLACE >= ~88, a half-trunk refresh is back in play. Taps for 100 train +
     MBPP 100 files are already re-prefilled (BigMoE).
  2. A learned refresher: train the drafter's own stand-in to track the true deep state (a feature loss against the
     true wd at depth), judged by the same REPLACE-style held-out eval. t30f's feature term was NO EFFECT, so find out
     why first (its weight, its target layer, which depths).
  3. The cheap check (Leap): a 32-row verify at ~100 ms (45 GB) is CPU-graph-bound (0.62 ms per layer even at 1 row).
     A compiled graph is the optimistic bound (0.39 ms per layer). A cheaper check makes frequent verification the
     refresh.
  4. Prior art for "a drafter that predicts the target's future hidden states" (NewIdeas' prior-art pass).
> (a quote from the project owner, kept private)
  generality added (orchestrator).
