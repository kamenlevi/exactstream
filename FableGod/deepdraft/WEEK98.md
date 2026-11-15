# WEEK98: the 98 % deep drafter, Sat 14 Nov -> Sat 21 Nov 2026

> (a quote from the project owner, kept private)
itself or the method for it being certified that it will work in a week, because I don't have more time than that to
spare to show this part of the project to my dad. I mean, so yeah, basically we must be able to find the thing which
unlocks 98% accuracy at 30 tokens deep. By then we must know what it is, and either I don't care if in a week we just
know the principles of how everything will work and it's certain, and we still have a week of building everything. I
don't care about that. I just need to get that."

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
  - Sun 15 night: Leap's a45 (exact-45 timing) 21:00 -> ~01:00; then EvalSuite tier 0a REPLAY calibration (~1.5 h;
    PREREG_lossygate amendment 2, eba6d3c; BigMoE's four envs on rpl1 03be8b3, dry runs clean); then the close-call base
    lx45cc1 (lossy-45 as shipped, on snapshot cc1) cached in both orders (~1.7 h) -> done ~04:15 Mon.
  - Mon 16 night: the close-call variant ccx45cc1 (delta 1 nat, CODE-ONLY; PREREG_lossygate amendment 3, 763da1c):
    forward ~50 min, reverse ~50 min if the forward passes. The 1c check variants need their own base on their later
    snapshot (a separate base run). Accept rules use CARD_ENGAGE_MODE=any (> 0 summed per process on code, 0 on IFEval).
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
