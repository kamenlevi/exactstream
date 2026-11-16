# PREREG: the lossy-check gate (WEEK98 goal B), a skeleton registered before any variant (Sun 2026-11-15 01:0x)

- **What it gates:** any lossy 32-row check, the variants that read under half the exact bytes (low-bit weak experts,
  acceptance-weighted expert sets, union caps; Breakthrough's LOSSY_CHECK_PLAN.md). The design is in
  ~/FableGod/deepdraft/LOSSY_GATE.md (copy: docs/LOSSY_GATE_2026-11-15.md).
- **Each variant gets a dated amendment before its run.** It names:
  - the env file (BigMoE data/stack/<variant>_env.txt) and its full git blob;
  - the snapshot's tree hashes (CARD_TREES), the screen's sha256 and the .so's;
  - the engagement field (CARD_ENGAGE);
  - the reference: exact, or a cached base run;
  - the tier and the window.
- **The rule from the orchestrator (00:5x):** tier 1's pooled line is a SCREENING line. Adopting any lossy check for a
  mode stays Kamen's call after a full verdict (tier 2, the card gate), as with P2.

## The panel (tools/lossy_panel.json, sha256 bb42228a500d029c...)
- **Tier 1:** HumanEval+ 164 (humaneval 60 + humanevalx 104), MBPP 100 and IFEval 100, so 364 keys.
  - They run in a proportional round-robin order, so any prefix is balanced: the first 182 keys are 82 / 50 / 50.
  - Scoring follows CERT's scorers: HumanEval+ through evalsuite.decode_report (humanevalx scored as HumanEval+), MBPP
    through tools/mbpp_score.py, IFEval through decode_report.
- **Tier 0:** the panel's first 20 HumanEval+ and first 20 IFEval keys (40).
- **Exact references at 45 GB:** results/ref2tnax (humaneval 60, ifeval 100), ref2tnaxn.r3 (humanevalx 104) and
  ref2tnaxn.r2 (mbpp 100), merged as CARD_REF=ref2tnax,ref2tnaxn.r3,ref2tnaxn.r2. They hold all 364 keys.
- **A cached base reference** (attribution against the shipping config): the base config's own tier-1 run, in the same
  order and process split. It is run once per base snapshot, under its own amendment.

## The lines (tools/lossy_d4.py; a run's verdict is the last line `LOSSY <mode>: ...`)
- **Tier 1, per set** (HumanEval+, MBPP, IFEval): decision 4. The paired delta must be above -5 points and the exact
  McNemar p at least 0.05; a gain with p < 0.05 is DIFFERS (HIGHER), which passes.
- **Tier 1, pooled** over all 364 pairs: delta above -3 points AND McNemar p >= 0.05; DIFFERS (HIGHER) passes. This is
  the screening line.
- **Tier 1 PASS:** the pooled line and the three per-set lines all pass, and the run is valid (below).
- **Interim**, after the first 182 keys: pooled z = (b - c) / sqrt(b + c). STOP-FAIL if z <= -2.8, else continue. There
  is no early PASS.
- **Tier 0** (kill only; it never adopts):
  - KILL if, on the 40 keys, the reference-only correct minus the arm-only correct is >= 4, or any arm row is empty.
  - Otherwise GO.
  - Once BigMoE's REPLAY mode exists (LOSSY_GATE.md section 9), two lines are added: agreement A and high-margin flips
    F_hi. Their thresholds come from the calibration below, registered in a dated amendment before their first use.

## Validity (otherwise VOID: fix, then a new run)
- **The pins:** the env blob, the snapshot trees, the screen and the .so equal the variant amendment's. Runner 64
  checks them at the start (a mismatch stops it, exit 7 / 2), and bm_pin plus the per-process manifests during the run.
- **Engagement:** the variant's engagement field is > 0 on every kept run. A base arm runs with no field.
- **QOM:** engaged on every kept run in the config's QOM sets, as at every gate.
- **Rows:** every panel key has exactly one uncut run.

## The calibration (before tier 0's REPLAY lines are used)
- **The configs:** tier 0 runs on configs whose gate outcome is known.
  - Passed: exact (A = 100 %), P2-24, P2-45, w13b (vs NAX6-N), m1b.
  - Failed: the 12 GB t80ck1u (vs REF2-T).
- **The thresholds,** to be registered in a dated amendment:
  - A below the lowest passing A minus 0.5 points;
  - F_hi above twice the highest passing F_hi.

## Checked before registration (01:0x)
- **The scorer reproduces CERT exactly.** `lossy_d4.py tier1 cert_fwd ref2tnax,ref2tnaxn.r3,ref2tnaxn.r2` matches
  CERT's published forward lines:
  - HumanEval+ 90.2 % vs 92.7 % (0 / 4);
  - MBPP 76.0 % vs 79.0 % (1 / 4);
  - IFEval 86.0 % vs 87.0 % (3 / 4).
  - Its pooled line is -2.20 points, p 0.0768 (4 / 12), so PASS: the shipped v1' stack sits 0.8 points above the
    screening line on this panel.
  - interim: z -2.00, CONTINUE. tier0: net loss 2, GO.
- **Power at this panel's measured discordance** (CERT's arm vs exact: 16 of 364 pairs, 4.4 %):
  - 78 % to FAIL a true -3 point pooled loss, and about 100 % for -5 points;
  - an equal variant FAILs about 3 % of the time.
  - At 12 % discordance (more lossy variants): 38 % / 79 % / >99 % for -3 / -5 / -8 points, and about 7 % false FAILs.
- **Runner 64** (tools/day_runner64.sh, sha256 4ef0308074c5fd52...; scorer tools/lossy_d4.py 8cd71dfd66697d20...: runner 63's checks, gates and yields with the panel, the interim look, a generic
  engagement field, merged references and start-time pins):
  - Dry runs at tiers 1 and 0 on chat45's env as a stand-in: blob, trees, screen and .so pinned, C 256, MEM 45, bm_pin
    OK, and the exact references holding 364 / 364 and 40 / 40.
  - A wrong tree pin stops it (exit 7), and so does a wrong blob (exit 2).

## Run times (measured throughput, process loads and heat included)
- **Tier 1 (about 92K tokens):** ~45 min at 45 GB with the fans up (2,045 tokens/min), ~85 min in heavy heat, and ~76
  min at 24 GB (1,216 tokens/min).
- **Tier 0 (about 11K tokens):** ~7 min at 45 GB and ~10 min at 24 GB, plus REPLAY's ~2-3 min once it exists.
- **Scoring:** about 75 s on the CPU.

## Amendment 1 (Sun 2026-11-15 01:0x, before any variant): the CODE-ONLY scope
- **Why (the orchestrator 01:2x; Breakthrough's LOSSY_CHECK_PLAN.md):** Breakthrough's pricing pre-registers a pooled
  verdict of KILL. Only a code-scoped lossy check reaches about 23-29 ms, because chat (IFEval, multi) flips 5-10 times
  more per unit of missed energy.
- **What it licenses:** a code-only mode only. The lossy check is on for code prompts, and chat stays on the base path.
  Every verdict line carries "(CODE-ONLY mode)". It never licenses a lossy check on chat.
- **Tier 1, code** (CARD_SCOPE=code, tools/lossy_panel.json sha256 8c0b23c6e74b01d3...):
  - HumanEval+ 164 and MBPP 100 (264, round-robin), with the interim look at 132.
  - Per-set decision-4 guards, plus the pooled code line: delta above -3 points and McNemar p >= 0.05 (a screening
    line).
  - Then IFEval 100, in its own processes with no code prompt in them, as the switch-off check. The engagement counter
    must be 0 on every IFEval run.
  - Against a cached base run (same snapshot, order and process split; CARD_IFEVAL_IDENT=1), IFEval tokens must be
    identical on 100 / 100.
  - Against exact rows instead, IFEval gets decision 4 ("no harm").
- **Tier 0, code:** the first 20 HumanEval+ and first 20 MBPP keys. KILL at a net loss of >= 4, or any empty row.
- **The reference for code-scoped variants is the cached base run.** That measures the variant's own cost; exact rows
  are information only.
  - The reason, measured at 01:0x [CORRECTED in amendment 1c: CERT's forward order only]: the shipped v1' stack itself
    (CERT's forward arm) is -2.65 points against exact on
    HumanEval+ + MBPP pooled, with McNemar p 0.0391 (1 / 8).
  - So it would FAIL the code screening line against exact before any lossy check is added. Gating variants on that
    base against exact would fail them for the base's own cost.
  - Whether base plus variant together hold against exact is tier 2's question, before any adoption.
- **The switch in the gate is by benchmark set:** the env scopes the lossy path to the code sets, the way R57_QOM_SETS
  does. A deployed code-only mode needs a request classifier, and this gate does not test its misroutes (chat sent down
  the lossy path).
- **Engagement:** CARD_ENGAGE=<runs.json field>, either a number or a dict of counts whose numbers are summed. For
  Breakthrough's counter, that is experts per check by tier (4-bit, MSB-3, dropped).
  - It must be > 0 on every code run and 0 on every IFEval run in the code scope, and > 0 on every run in the all scope.
  - It was unit-tested on synthetic runs (a non-engaged code run and an engaged chat run are both caught).
- **Power, code pooled:** the measured discordance is 9 / 264 = 3.4 % (CERT's arm against exact), giving about 76 % power
  against a -3 point loss and about 99 % against -5.
- **Checked (01:0x):** lossy_d4.py (sha256 d8cc7531a56b83d7...), code mode, on CERT's arm against exact.
  - HumanEval+ PASS and MBPP PASS, but pooled code -2.65 points with p 0.0391: FAIL. IFEval no-harm PASS. So the verdict
    is FAIL (pooled).
  - With identity against exact, IFEval is 11 / 100 identical, so FAIL: the stack's chat is not exact, as known.
  - The self-test (arm = reference) passes, with 100 / 100.
  - interim code: z -2.33, CONTINUE. tier0 code: net loss 3, GO.
  - Runner 64 (sha256 78c91f6288634b2b...) dry runs pass at tier 1 code (with and without identity), tier 0 code and tier 1 all.

## Amendment 1c (Sun 2026-11-15 01:3x, before any variant): amendment 1's reason restated, and the order rule
- **The correction** (the orchestrator 01:3x, from MoonShots' ~/Papers/CERT_CODE_CAVEAT_2026-11-15.md, reading CERT's
  published table): amendment 1's "-2.65 points against exact" is CERT's forward order only.
  - The reverse order is +0.76 points on the code sets (HumanEval+ 0.00, MBPP +2.00).
  - Both orders together are -0.95 points (net -5 of 528 pairs).
  - So "v1' fails the code screening line against exact" holds for the forward order only.
  - Checked here (01:3x) with `lossy_d4.py tier1 cert_rev <exact refs> code`:
    - HumanEval+ 0.00 (2 / 2), MBPP +2.00 (5 / 3), pooled code +0.76, p 0.7744 (7 / 5).
    - With the forward order's 1 / 8: both orders -0.95 points, McNemar p 0.3833 (8 / 13).
- **The reason for a cached base reference, restated:** gating against the base run measures the variant's own cost.
  The exact comparison stays information, and whether base plus variant hold against exact is tier 2's question.
- **The order rule:**
  - Each arm runs in its own processes from the same cold start (the same warm-up), in the same order and process split
    as its cached base run. It never runs back-to-back with another arm in one process: a warm cache changes QOM's
    tokens and favours the second arm (ab-arms-order-warm-cache).
  - The order's cache state therefore hits both arms alike and cancels in the pairs. QOM noise after the tokens diverge
    is discordance, which McNemar counts as noise.
  - One order is still one draw (CERT's v1' moved 3.4 points between orders). So:
    - **Forward order:** the tier-1 lines as registered. A FAIL or an interim STOP-FAIL ends the variant there.
    - **A forward PASS is provisional.** The variant then runs the reverse order against the base's reverse run (cached
      once per base snapshot).
    - **The tier-1 verdict that licenses tier 2** applies the same lines to both orders pooled: 528 code pairs, plus
      the IFEval switch-off check in both orders.
- **Cost:** one more ~50 min pass per forward-passing variant, plus the base's reverse run once.
- **Build before the first reverse pass:** a reverse-order switch for runner 64. CARD_ORDER=rev reverses the panel's
  keys and the IFEval block, and the interim look falls on the first half of the reversed order. It gets dry runs as
  runner 64's other settings did, and the variant's amendment names its sha256.
- **The scorer needs a pooled mode too:** lossy_d4.py merges runs by key, which would collapse two orders of the same
  keys into one row. A pooled mode pairs (arm forward, base forward) and (arm reverse, base reverse) separately, then
  pools the pairs, so each key counts once per order.
  - It is checked on CERT's own runs before use. It must reproduce the both-orders code numbers: -0.95 points, net -5 of
    528 pairs.
- **Runner 64 fix (01:3x, before any variant):**
  - Its dry runs had copied the lossy panel's 364 keys over results/card/card_ref_keys.txt, which is the card gate's
    reference-key file (381 keys). card_info.py and card_chain*.sh read that file.
  - The file is restored from git (381 keys). Runner 64 now writes its own results/card/lossy_ref_keys_<tag>_t<tier><scope>.txt
    (tools/day_runner64.sh sha256 0986fef7247e3f25...; bash -n OK).
  - Its dry runs repeat in the first variant's amendment, before any run.

## Amendment 1d (Sun 2026-11-15 01:5x, before any variant): engagement keys, HC4's scope, accept rules, reverse order built
- **Engagement takes key subsets:** CARD_ENGAGE=<field>:<key>+<key> sums only the listed counts of a dict field.
  - Breakthrough's counter (~/FableGod/deepdraft/LOSSY_CHECK_SPEC.md) is gated as lossy_check:e3+edrop+lmsl_rows:
    > 0 on every code run, 0 on every IFEval run.
  - Its `checks` and `e4` are nonzero on every prompt, so amendment 1's whole-dict sum would have read chat as
    engaged.
  - Unit-tested on six synthetic runs.json cases:
    - a correctly scoped variant passes;
    - the whole-dict sum flags chat (the old behaviour);
    - a code run with no lossy tier is caught;
    - global HC4 on chat passes without `hc4` in the sum and is caught with it;
    - a base arm (no field) passes.
- **V3's R57_HC4:** the spec gives it no set scope. As written it would run 4-bit mixers on chat too, so V3 gets a
  CODE-ONLY verdict only if:
  - HC4 is scoped to the code sets;
  - its amendment gates lossy_check:e3+edrop+lmsl_rows+hc4 (0 on IFEval).
  - Otherwise V3 is gated in the all scope.
- **Acceptance-rule variants** (the orchestrator 01:5x; MoonShots' result H) follow LOSSY_GATE.md section 11:
  - their own counter field of accepted non-top-1 tokens, > 0 on code and 0 on IFEval;
  - the drafter (and the 2B auditor's) checkpoints pinned;
  - MBPP's per-set guard is the line to watch;
  - tier 0 is the mini-panel's KILL line only, until an accept-rule calibration amendment.
- **The reverse order is built:** runner 64 CARD_ORDER=rev (tools/day_runner64.sh sha256 804247d9253fed67...).
  - The code keys and the IFEval block reverse apart, so IFEval still runs last in its own processes and the interim
    look falls on code keys.
  - The run, its key files, logs and DONE file get "_rev".
  - CARD_FWD_RUN + CARD_FWD_REF add the both-orders pooled verdict at the end, and that pooled line becomes the DONE
    verdict.
- **Dry runs (01:52, chat45's env as a stand-in; BigMoE's current trees):**
  - forward code with identity, reverse code with identity, reverse all scope, reverse tier 0 code: all rc 0;
  - the key orders and counts are as designed (364 / 364 / 364 / 40);
  - the reference holds every key;
  - the stale bigmoe tree from 01:0x (before BigMoE's c1c217b) stops it, exit 7;
  - the card gate's results/card/card_ref_keys.txt is untouched.
- **The pooled scorer:** lossy_d4.py joins the orders with '+' (arm `x_fwd+x_rev`, reference `b_fwd+b_rev`; one
  reference group serves every order). Each order is paired on its own, and the pairs pool with keys "o:key".
  - Single-order regression: `tier1 cert_fwd <exact> code` still gives -2.65 pt, p 0.0391 (1 / 8), FAIL.
  - **Both orders pooled reproduce CERT (02:00):** `tier1 cert_fwd+cert_rev <exact> code` (lossy_d4.py sha256 c41eb07c6df50f27...;
    results/lossygate/pooled_cert_2026-11-15.txt):
    - HumanEval+ (n 328): -1.22 pt, p 0.2891 (2 / 6);
    - MBPP (n 200): -0.50 pt (6 / 7);
    - **pooled code (n 528): -0.95 pt, McNemar p 0.3833 (8 / 13): PASS,** the required check;
    - IFEval no-harm (n 200): -1.50 pt (7 / 10), PASS.
- **Tier 0a, the REPLAY pass, is wired into runner 64** (sha256 a31b49a9ec74ccc9...). BigMoE built R57_REPLAY in
  scripts/r57_screen_fast.py, 6a0f70f, and its code matches LOSSY_GATE section 9.
  - With CARD_REPLAY set, tier 0 first writes the tier-0 keys' reference rows from CARD_REF (earlier references win,
    as in scoring). It runs them under R57_REPLAY as its own run (<run>_rpl), then the mini-panel as before.
  - **CARD_REPLAY=exact:** the exact config's self-check.
    - Each process writes its own margins (a margins path shared by two processes would make the second one read the
      first's file). They are merged into results/card/replay_margins_<tag>.json.
    - PASS needs every drafted and check row accepted, the prefill picks equal, and margins for every key. A FAIL
      VOIDs the run.
  - **CARD_REPLAY=<merged margins>:** a variant's pass. It reports the pooled A and F_hi (flips per 1,000 drafted rows
    at exact margin >= 2), plus the flips by margin bin.
    - It is INCOMPLETE if any key lacks a summary, or if the margins are not the exact run's.
    - It stays information until the calibration amendment registers the A / F_hi lines (the skeleton's rule: A below
      the lowest passing A minus 0.5 pt; F_hi above twice the highest passing F_hi).
  - **Checked offline on synthetic files:**
    - the rows writer (earlier reference wins; a missing key stops it);
    - the summary: exact all-accepted PASS rc 0; one rejected row FAIL rc 1; a variant on exact margins rc 0 (A 0.9500,
      F_hi 16.67); a variant on its own margins INCOMPLETE rc 1.
  - **The first GPU use is the calibration night,** under its own amendment:
    - exact (the self-check), then the known configs: passed P2-24, P2-45, w13b and m1b; failed t80ck1u.
    - Before it, re-pin the trees and the screen sha (BigMoE's HEAD moved: bigmoe d92cdb4, the screen 6a0f70f), and
      check that the unset path is unchanged: tier-0 tokens identical to the exact references on 40 / 40.
- **Tier 0a needs BigMoE's screen at d1fd8c5 or later** (02:5x; scripts/r57_screen_fast.py sha256 9abb2608fd8a...).
  - At 6a0f70f, R57_REPLAY asserted "no row for None" in the loop's warm-up decode, before the key loop. Runner 64 runs
    that warm-up, so its first REPLAY pass would have stopped at start-up.
  - d1fd8c5 engages REPLAY only once the loop sets a key. The diff was checked here; the unset path is unchanged.
  - The variant amendments and the calibration amendment pin CARD_SCREEN_SHA=9abb2608fd8a (or a later registered sha).
  - Tier 0's exact pass waits for BigMoE's own REPLAY self-check (4 REF2-T rows, exact-45, C 160, 100 % drafted rows),
    which is queued after the G32T timing run.
- **R57_REPLAY is certified by BigMoE (03:0x; abe7772):** exact-45 replays its own rows at 100 %: 724 / 724 drafted
  rows, 775 / 775 all rows, 0 flips; 4 keys including a chat key. Two consequences for runner 64 (sha256 f1281e81f008363c...):
  - **Replay the exact config's OWN rows.**
    - On REF2-T's rows the exact config accepts 525 / 527, with both flips at REF2-T tie points (margins 0.0 and
      0.125 nat).
    - EvalSuite's exact references (ref2tnax, ref2tnaxn.*) were made on the REF2 path; every row carries its ref2 call
      stats. So they would make the strict 100 % self-check fail at ties.
    - New: CARD_REPLAY_ROWS=<run[,run]> names the replayed rows (default CARD_REF, with a logged warning). Every tier-0a
      pass, the exact self-check and every variant, replays the same exact-45 own run at the registered C, with the
      margins from that exact pass.
  - **The screen must contain REPLAY with both start-up fixes:** d1fd8c5 (the warm-up has no key) and e5d17c6 (DFlash2
    carries the reference drafts on every set, chat included). BigMoE pins scripts/r57_screen_fast.py at aadd08b
    (sha256 7c08a4970679) or e5d17c6 (c91b1c0ae107).
    - Runner 64 runs the screen that the env comment names as a snapshot, or a freeze of BigMoE's live tree when none
      is named. chat45's P3 snapshot (r57_screen_p3.py) predates REPLAY.
    - New guard: with CARD_REPLAY set, the runner stops (exit 2) unless the screen holds both fixes. Dry runs: the P3
      snapshot without REPLAY gives rc 0; with CARD_REPLAY=exact it gives rc 2 and the message.
- **Tier 0a, in order (the calibration amendment pins each step):**
  1. A tier-0 decode on exact-45 flags (no REPLAY) with the certified screen at the registered C. Its rows are the own
     rows (CARD_REPLAY_ROWS).
     - This is also the unset-path check: compared with the REF2-path references, the token match share is reported,
       and every difference must sit at a REF2 margin < 0.5 nat.
     - Identity on 40 / 40 is not required: the exact paths differ at ties.
  2. CARD_REPLAY=exact with CARD_REPLAY_ROWS=<step 1's run>: must be 100 %; it writes the merged margins.
  3. The known configs and the variants: CARD_REPLAY=<merged margins>, with the same CARD_REPLAY_ROWS.
- **The REPLAY-capable snapshot (BigMoE 03:2x, 5cdbcab), verified here:**
  - data/r57/snap/rpl1/r57_screen_rpl1.py, sha256 0d3d45ac9b72. It is cut from the screen at aadd08b and every
    difference is a path rewrite to the frozen helpers.
  - Both fix greps match once each.
  - Its helpers are data/r57/snap/frozen_rpl1. Both directories are committed and clean.
  - Its pins: CARD_TREES=data/r57/snap/rpl1=0fb256faba39d467950e5c451b751c6d3b347786,
    data/r57/snap/frozen_rpl1=db72c3c1367e2c68f9a1365ed8d6137fce6c8af6, bigmoe=d92cdb4b9e81d386e4170323d50f11b34147aa5f;
    CARD_SCREEN_SHA=0d3d45ac9b72.
  - It serves the tier-0a calibration (the exact-45 own-rows run, the exact self-check, the known configs on REPLAY).
  - The variants V1-V4 need a newer snapshot carrying the 1c wiring. The cached lossy-45 base for their tier-1 gate is
    run on that same newer snapshot, so base and variant differ only by the variant's flags.

## Amendment 2 (Sun 2026-11-15 03:1x, before any REPLAY run): the tier-0a calibration plan
- **The configs** (BigMoE 03be8b3, ~/BigMoE/data/stack/; all on the rpl1 snapshot; dry runs rc 0 at 03:11):

| tag | what | env blob | .so | MEM | known |
|---|---|---|---|---|---|
| x45rpl1 | exact-45 as shipped (Leap's exact-45.env), C 256 | dcfe4196dec6663fe90988c06cd5a3fc2cd69b33 | 5333899cb514 | 45 | exact (A = 100 %) |
| p2x45rpl1 | chat45_p2's line verbatim | fb592bfa1f41cf76deadd587f9847ffb49941068 | ff69fa24a745 | 45 | PASS (the 45 GB P2 gate) |
| m1bx45rpl1 | card_m1b's line verbatim | 532f0c44eedd23c7f5e188267e11dd3b438c3f1b | ff69fa24a745 | 47 | PASS (the card gates) |
| lossycalx45rpl1 | exact-45 + QOM + each row's top-3 of 10 experts only (R57_RCUT_SETS ...:3), deliberately lossy | 8cb1a7ca3cd3334b70a8215840c59372f33dc4ca | ff69fa24a745 | 45 | untested; known-bad only if its own mini-panel KILLs |

- **Pins for every run:**
  - CARD_TREES=data/r57/snap/rpl1=0fb256faba39d467950e5c451b751c6d3b347786,data/r57/snap/frozen_rpl1=db72c3c1367e2c68f9a1365ed8d6137fce6c8af6,bigmoe=d92cdb4b9e81d386e4170323d50f11b34147aa5f
  - CARD_SCREEN_SHA=0d3d45ac9b72
  - CARD_SO_SHA per the table
  - CARD_SCOPE=code, CARD_TIER=0, CARD_REF=ref2tnax,ref2tnaxn.r3,ref2tnaxn.r2
- **Steps, one night at 45 GB under a share lock, ~1.5 h** (each tier-0 invocation ~10-20 min with the load):
  1. **x45rpl1 without CARD_REPLAY.** Its decode (card_x45rpl1_t0code) is the own rows.
     - The unset-path check: the token match share against the REF2-path references is reported, plus each
       divergence's position.
     - PASS needs every divergence to sit at a near-tie: the exact own margin (from step 2) < 0.5 nat there.
  2. **x45rpl1 with CARD_REPLAY=exact, CARD_REPLAY_ROWS=card_x45rpl1_t0code.**
     - It must give 100 % (the self-check), and it writes results/card/replay_margins_x45rpl1.json.
     - Corrected 03:2x: no new tag. Runner 64 requires the tag in the env comment. The replay pass runs as
       card_x45rpl1_t0code_rpl under fresh process tags, and the mini-panel decode is skipped because step 1's rows
       exist.
     - Step 1's check then runs (tools/replay_unset_check.py card_x45rpl1_t0code <CARD_REF> <the merged margins> <the
       tier-0 code keys>). PASS needs each key's FIRST divergence at an own margin < 0.5 nat. A divergence with no
       margin there is INCOMPLETE and is read by hand.
       - Tested on synthetic rows: PASS rc 0, a 2.7-nat divergence FAIL rc 1, a missing margin INCOMPLETE rc 2.
  3. **p2x45rpl1, m1bx45rpl1 (CARD_MEM=47) and lossycalx45rpl1,** each with CARD_REPLAY=<step 2's margins> and
     CARD_REPLAY_ROWS=card_x45rpl1_t0code.
     - Each gives A and F_hi, then its own mini-panel KILL / GO against the exact references.
- **The lines, computed after step 3 from these runs only:**
  - A_line = the lowest passing A (p2, m1b) minus 0.5 pt;
  - F_line = twice the highest passing F_hi.
  - A variant is KILLed at tier 0a if A < A_line OR F_hi > F_line.
  - **Adoption check:** lossycal, if its mini-panel KILLs, must fall outside the lines. If it doesn't, the REPLAY lines
    can't separate even a grossly lossy arm. They are then NOT adopted, and tier 0 stays the mini-panel only.
  - If lossycal does not KILL, it is reported, and the lines rest on the passes alone. The amendment that adopts them
    says so.
- **Caveats:**
  - Under REPLAY, chat keys draft on DFlash2's path (the MTP head isn't wired), so m1b's QOM-gated MTP row gate isn't
    what runs. The code keys (this tier 0's whole scope) are unaffected.
  - With one known-bad arm the bad side is a single point. A graded set (top-5, top-7) would show the curve between,
    but it is not in this plan.
- **The slot (the orchestrator 03:1x):** Sunday night after Leap's a45 run (21:00 -> ~01:00).
  - First this calibration (~1.5 h).
  - Then the lossy-45 base cached in both orders (~1.5 h), if BigMoE's snapshot with the 1c wiring exists by then.
    Otherwise the base goes first on Monday night, before the variant gates.
  - The runs need ~47 GB NO-SWAP; the orchestrator asks Kamen to close apps from 21:00.
  - They are chained on one gate behind Leap's run, with two reads 30 s apart.

## Amendment 3 (Sun 2026-11-15 03:2x, before its runs): variant CLOSE-CALL acceptance at delta 1 nat (CODE-ONLY)
- **The variant** (BigMoE 2ad2d67; priced by Breakthrough 3c09867, HE+ kept/32 +21.8 %, MBPP +23.1 % as upper bounds):
  R1(delta) accepts a draft whose logit is within 1 nat of the top-1, on the code sets only (R57_RELAX_DELTA=1,
  R57_RELAX_DELTA_SETS=humaneval,humanevalx,mbpp). It is an ACCEPT RULE (LOSSY_GATE section 11).
- **The snapshot cc1** (6bb40d6), verified here:
  - screen data/r57/snap/cc1/r57_screen_cc1.py, sha256 efc1f9b6f531;
  - both REPLAY fixes, the RELAX_DELTA code and the close_call counter are present;
  - committed and clean.
  - CARD_TREES=data/r57/snap/cc1=85a88a46a945b3a4d21cc2c6fb216372fa549eab,data/r57/snap/frozen_cc1=ec97bdbb7ac3ba7fdd1cdf845762e4957d6b4c06,bigmoe=d92cdb4b9e81d386e4170323d50f11b34147aa5f;
    CARD_SCREEN_SHA=efc1f9b6f531; CARD_SO_SHA=ff69fa24a745; C 256, MEM 45.
- **The arms, on the same snapshot** (BigMoE data/stack/):
  - **BASE lx45cc1** (env blob 216c8e20d35494ad32c855eb83c2712e86c94d18): lossy-45 as shipped, Leap's release line
    minus C=.
  - **VARIANT ccx45cc1** (env blob 46bdf06bc43d05bc768219c7298793013e846dab): the same line plus exactly the two flags
    (the env-line diff was checked here).
- **Engagement:** CARD_ENGAGE=close_call:accepted with CARD_ENGAGE_MODE=any (new, runner 64 sha256 5cab78942ae9501c...).
  - BigMoE's counter counts the committed non-top-1 drafts kept ONLY by R1; the base's relaxations never count. Outside
    the scope it is absent.
  - An accept rule engages only where a draft qualifies, about 1-2 tokens per answer by the pricing. So many code runs
    honestly read 0, and the every-run rule would VOID them.
  - The rule is therefore > 0 summed over each process's code runs, and absent or 0 on every IFEval run.
  - Unit-tested on synthetic runs.json:
    - every-run mode with a code run at 0: VOID (the old rule);
    - any mode: OK;
    - any mode with all code at 0: VOID;
    - a chat run engaged: VOID.
- **The runs** (tier 1, CARD_SCOPE=code; the IFEval block in its own processes):
  1. BASE forward and reverse (CARD_ORDER=rev), with no CARD_ENGAGE and CARD_REF = the exact refs (information only).
     These are the cached references: card_lx45cc1_t1code and card_lx45cc1_t1code_rev.
  2. VARIANT forward: CARD_REF=card_lx45cc1_t1code, CARD_IFEVAL_IDENT=1 (IFEval tokens identical to the base's on
     100 / 100).
     - FAIL or an interim STOP-FAIL ends it.
  3. If step 2 passes, VARIANT reverse: CARD_REF=card_lx45cc1_t1code_rev, CARD_FWD_RUN=card_ccx45cc1_t1code,
     CARD_FWD_REF=card_lx45cc1_t1code. The verdict is both orders pooled.
- **Lines:** as registered: per-set decision 4, the pooled code line (> -3 pt AND p >= 0.05), and the IFEval identity.
  MBPP's per-set guard is the line to watch.
- **Tier 0** for an accept rule is the mini-panel KILL line only. It is optional before step 2.
- **Dry runs (03:22):**
  - base forward and reverse: rc 0 (364 keys, QOM sets humaneval,mbpp);
  - the variant against its not-yet-made base: it stops ("the reference holds 0 of 364"), as it must;
  - the variant with the exact refs as a stand-in: rc 0.
- **Time:** ~50 min per arm per order at 45 GB (code ~61K tokens + IFEval). That is ~1.7 h for the base's two orders
  and ~1.7 h for the variant's.

## Amendment 3a (Sun 2026-11-15 10:3x, before any variant data): a RUNAWAY guard in tier 1, and the speed read-out
- **Why:** MoonShots' J(CC) (12f76d1; verified by the orchestrator 10:3x) ran end to end on the 2B, s30, 264 keys paired,
  with EvalSuite's scorers.
  - Close-call at delta 1 HOLDS: pooled +0.76 pt, p 0.88. A numerics-only control gives +0.76 too.
  - At delta 2 it FAILS: pooled -6.06 pt; HumanEval+ -9.15 pt, p 0.020.
  - The mechanism is runaway answers (cap hits 26 -> 50) plus docstring rewrites that steer the code.
  - Part of delta 1's speed lift was runaway answers: real kept per check +21 %, but only +9 % (HumanEval+) / +19 %
    (MBPP) without capped answers.
- **The guard, for every variant from now on** (tools/lossy_d4.py tier1, in both the single-order and pooled modes):
  - An answer is a RUNAWAY if it hit the token cap (finish "length") or loops (evalsuite.decode_report.looped: a
    20-token n-gram >= 3 times in the last 400 tokens).
  - The guard is paired over the scope's pooled sets: arm-only runaways b against reference-only c.
  - FAIL if b > c with exact McNemar p < 0.05. Cap hits and loops print per set either way.
  - Tier 1 PASS now also needs this guard.
- **The speed read-out (information beside the verdict, never a line):** tools/lossy_speed.py <arm TG> <base TG>.
  - It gives kept tokens per check, arm against base, per set and for code pooled.
  - Twice: on all answers, and on the answers uncapped and unlooped in BOTH arms.
  - It reads runner 64's per-process runs.json (the REPLAY processes are left out) and the runs' decode rows.
  - Tested on synthetic files: a capped fast answer reads +125 % on all answers, but +25 % on the clean ones.
  - Any speed claim quotes the clean number.

## Amendment 3b (Sun 2026-11-15 10:3x, before its data): a delta-2 close-call arm, if BigMoE writes its env
- **The arm:** ccx45cc1d2, lx45cc1's line + R57_RELAX_DELTA=2 + R57_RELAX_DELTA_SETS=humaneval,humanevalx,mbpp, on the
  same cc1 snapshot.
  - It is gated exactly as amendment 3 (CARD_ENGAGE=close_call:accepted, CARD_ENGAGE_MODE=any), against the same
    cached base, in both orders.
  - It runs after delta 1. Time: ~1.7 h at 45 GB; the night comes from the orchestrator and NewIdeas' queue.
- **What is known:**
  - On the 2B, delta 2 fails through runaway answers, which the new guard tests directly.
  - On the 125B, the teacher-forced delta-2 code dose (HumanEval+ 5.8 non-top-1 per 1K) is below the 2B's delta-1 dose
    (7.5), which held. So delta 2 is plausible on the 125B, not established.
- **Two arms against one base:**
  - Each gets its own verdict.
  - If both pass and the faster is preferred, that choice is a selection, and the tier-2 card gate must confirm it
    before any adoption.
- **Not registered:** delta 2 with a comment / docstring guard. That is Breakthrough's spec. Comments and docstrings
  carry ~75 % of delta 1's rule commits (24.9 of 32.9 per 1K), so the guard cuts most of the gain. It would come under
  its own amendment.
- **Checked (10:2x)** with lossy_d4.py sha256 f607a4c6a1f4fbde... (the speed tool is sha256 eab2ee49e4582f4f...): CERT's v1' in both orders
  against exact, code (results/lossygate/runaway_cert_2026-11-15.txt).
  - HumanEval+ cap hits 14 vs 8 (arm only 7 / reference only 1).
  - MBPP cap hits 14 vs 12, loops 1 vs 0 (8 / 5).
  - **Runaway guard: arm only 15 / reference only 6, McNemar p 0.0784: PASS.** CERT's tier-1 verdict is unchanged:
    PASS, both orders pooled.
  - Information: the shipped lossy stack itself trends toward more capped answers than exact. A variant is gated
    against the cached lossy-45 base, so the guard measures the variant's own increment over that base.
- **The delta-2 env is registered (amendment 3b; BigMoE e7ba034):** ccx45cc1d2_env.txt, blob
  a5cb28ef13ec3221d24655c199b8e65e6ffb2ad1, clean. The only difference from ccx45cc1 is R57_RELAX_DELTA 1 -> 2 (checked
  here). Same cc1 pins. Dry run with the exact refs as a stand-in: rc 0.

## Amendment 2a (Mon 2026-11-16 06:3x, after the calibration night): no tier-0a lines
- **What ran** (docs/RESULT_lossygate_2026-11-16.md):
  - steps 1 and 2 PASS: tier 0 GO, the unset path 40 / 40 token-identical, and the exact REPLAY self-check at A 1.0000
    on 10,315 drafted rows with 0 flips;
  - step 3 gave no rows. p2x45rpl1 and m1bx45rpl1 crashed in the rpl1 snapshot's REPLAY row gate (IndexError in
    relax_core.gate_rows, from r57_screen_rpl1.py line 1324, on a block cut by the reference's end).
    lossycalx45rpl1 crashed in ref2 ("QOM on the NAX path ... is not built"). Each runner stopped after 30 processes
    (exit 3).
- **So:**
  - A_line and F_line are not computed and not adopted, since amendment 2 computes them from step 3's passes only;
  - tier 0 stays the mini-panel KILL line, and a REPLAY pass stays information.
- **A re-run** is a new dated amendment, registered before its rows. It needs:
  - BigMoE's fix in a new snapshot, with its pins;
  - a deliberately lossy arm whose env runs on a QOM-capable path;
  - before arming, one scored key per env on the real loop. The head-test stops at the model load, and neither crash
    came before the first scored key.

## Amendment 2b (Mon 2026-11-16 07:0x, before its rows): the tier-0a re-run on snapshot rpl2, step 3 only
- **The snapshot rpl2** (BigMoE 89a6798), checked here:
  - screen data/r57/snap/rpl2/r57_screen_rpl2.py, sha256 d74d8b119e34; helpers data/r57/snap/frozen_rpl1, unchanged;
  - its only differences from rpl1's screen (the diff, read here):
    - a module-level assert: R57_FAST=QOM with R57_NAX=1 needs REF2_OOO_1C=1 and REF2_NAX_1CQ=1, and no R57_H1C;
    - `_gate_block`, in place of both `gate_rows` calls in R57_FAST's row gate. A full window takes the same
      `gate_rows` call as before. A block cut by the reference's end is padded, and every row from its end on is
      exact, as a full block's row 15 is.
  - With R57_FAST unset, neither change runs, so the exact path is rpl1's.
  - Pins: CARD_TREES=data/r57/snap/rpl2=7038e9a8ca1e98f16a0e2b88df75fd5bd3efc097,data/r57/snap/frozen_rpl1=db72c3c1367e2c68f9a1365ed8d6137fce6c8af6,bigmoe=d92cdb4b9e81d386e4170323d50f11b34147aa5f;
    CARD_SCREEN_SHA=d74d8b119e34; CARD_SO_SHA=ff69fa24a745.
- **Steps 1 and 2 carry over** from amendment 2a's night: the own rows card_x45rpl1_t0code and the merged margins
  results/card/replay_margins_x45rpl1.json.
  - The reason: the exact path is unchanged (above). BigMoE's key test also replayed those rows on x45rpl2 (exact-45
    on rpl2, env blob d2e1b8fa2e12e73979961c20ba0f8f91e1f6b332) at A 1.0000: mbpp/118 16 / 16, mbpp/128 33 / 33.
- **Step 3, re-run on rpl2.** Each arm uses CARD_TIER=0, CARD_SCOPE=code, CARD_REF=ref2tnax,ref2tnaxn.r3,ref2tnaxn.r2,
  CARD_REPLAY=results/card/replay_margins_x45rpl1.json and CARD_REPLAY_ROWS=card_x45rpl1_t0code:

| tag | what | env blob | NO-SWAP gate | known |
|---|---|---|---|---|
| p2x45rpl2 | p2x45rpl1's line verbatim (chat45_p2) | ff30bd2afd75c146b7676d2dfd15ba0d0eab9315 | 43 (CARD_MEM 39) | PASS (the 45 GB P2 gate) |
| m1bx45rpl2 | m1bx45rpl1's line verbatim (card_m1b) | a59e13e024504b7083cea083b30400d1d0e4a8e3 | 47 (CARD_MEM 43; its card env's peak ~47 GB) | PASS (the card gates) |
| lossycalx45rpl2 | m1bx45rpl2's line + R57_RCUT_SETS=humaneval:3,mbpp:3,ifeval:3 (each decode row of those sets keeps its top-3 of 10 experts, on top of QOM) | 741c4d2f0af44aca58c11304b61fd22963a0b4b1 | 47 (CARD_MEM 43) | untested; deliberately lossy, known-bad only if its own mini-panel KILLs |

  - lossycalx45rpl2 replaces lossycalx45rpl1, which the engine refused (QOM on the NAX path without the one-call).
  - Dry runs (07:01) on these pins: rc 0 on all three, 40 keys each, and the REPLAY guard passes.
- **The pre-arming check (amendment 2a):** BigMoE's real-loop key test (a16e652, data/r57/rpl2_keytest.json, Mon
  06:55-06:58).
  - All four rpl2 envs were built as runner 64 builds them, with R57_REPLAY on these rows and margins.
  - mbpp/118 and mbpp/128 were each replayed to the reference's end, through last checks of 1 and 3 drafts.
  - Every run ended rc 0, with the FAST gate engaged up to each key's last check.
  - p2 and m1b reached A 1.0000 with 0 flips. lossycal had mbpp/128 at 31 / 33 (A 0.9394; information).
  - Lifetime peaks were 41.7-42.6 GB on these short keys.
- **The lines and the adoption check: as in amendment 2,** from these three runs only.
  - A_line = the lowest passing A (p2, m1b) minus 0.5 pt; F_line = twice the highest passing F_hi.
  - If lossycal's mini-panel KILLs, lossycal must fall outside the lines, or the lines are not adopted. If it doesn't
    KILL, it is reported, and the lines rest on the passes alone.
  - A consequence, stated before the data: if both passes have F_hi 0.00, then F_line is 0. A single flip at an exact
    margin >= 2 nats then KILLs at tier 0a. Tier 0 never adopts, so this errs toward killing, and the rule stays as
    registered.
- **The slot:** low priority (the orchestrator 06:4x), and not in night1117, which was armed before rpl2 was verified.
  - Next is a free 45 GB slot later in the week: about 40 min for the three tier-0 passes, with each process gated as
    in the table.
  - The usual rules apply: HOLD, PAUSE and a shut lid hold it, and it yields to Kamen at the Mac.
  - A REPLAY pass does not resume (its merge is per runner start). An arm whose pass yields is re-run whole.

## Amendment 2c (Mon 2026-11-16 08:4x, before its rows): the tier-0a re-run goes LAST in night1117
- **Why:** amendment 2b put the re-run in a free 45 GB slot later in the week, because night1117 was armed before rpl2
  was verified.
  - The orchestrator's condition for night1117 (06:4x, 06:5x) was NewIdeas' agreement. NewIdeas agreed (08:2x).
  - Every later night this week is taken: Tue t30c, Wed the capture, Thu-Fri the proofs.
- **How: a separate follower,** tools/night1117_t0a.sh (sha256 b106d07f1583e9cc...). The armed chain (pid 52881,
  sha256 d434e7f7866d2bc2) is not touched.
  - **Start:** night1117's end marker (results/CHAIN_1117_END), written after the follower's arming and saying "done:",
    with the chain's process gone. It is read twice, 30 s apart.
  - Any other end (a HOLD, a signal, no run), or the chain gone without a new marker, means no run. The re-run then
    waits for another slot.
  - No start while HOLD, PAUSE, a shut lid or Kamen at the Mac, and none after Tue 13:15. The three arms need ~45-90 min.
  - **The cutoff** is 14:30, or 2 h after the start if that is earlier, so a memory wait cannot hold up Tuesday's day
    slot.
- **The runs are amendment 2b's, unchanged:** p2x45rpl2 (gate 43), m1bx45rpl2 (47), then lossycalx45rpl2 (47).
  - Each is its REPLAY pass on the merged margins, then its mini-panel.
  - The pins, env blobs, margins, rows and lines are as registered there.
  - An arm that ends without a verdict stops the follower, since the lines need all three.
- **A yield to Kamen:** once nothing holds, the arm is re-run whole, because a REPLAY pass does not resume (amendment
  2b). Its partial run dirs are moved aside first (<dir>.yield<N>).
- **Every end writes results/T0A_1117_END.** Stop: `kill -TERM <pid>` (never -9).
- **Checked before arming:**
  - **The dry run (08:38) on the real runner:** rc 0 on all three. Each had pins OK, 40 keys, the reference holding
    40 / 40, bm_pin OK and the REPLAY screen guard passing.
  - **A stub simulation** (results/lossygate/t0a_follower_sim_2026-11-16.txt), seven scenarios:
    - a stale marker is ignored while the chain runs;
    - a "done:" end starts the three arms at MEM 39 / 43 / 43 with the start + 2 h cutoff;
    - a HOLD end or the chain gone without a marker: no run (rc 5);
    - past the last start: rc 6;
    - a yield: the runner stopped, peer unlock, the dirs moved aside, the arm re-run whole;
    - a HOLD mid-run: rc 8 with the marker;
    - an arm crash: the follower stops there.

## Amendment 4 (Mon 2026-11-16 08:4x, before any 1c-tiers run): the plan for variant 1c-tiers (V2), CODE-ONLY; the pins follow in 4a
- **Why:** the orchestrator approved Fit12's cheapest gateable lossy check at 08:3x. That is the LOSSY32 price's
  candidate "C3": lossy-45 + V2, the 1c expert tiers. Its price [P] is 53-55 ms per 32-row check, against lossy-45's
  90-92 (/Users/kamen/Fit12/data/lossy32/lossy32_price.txt).
  - NewIdeas coordinates the owners and orders the slot.
  - The spec: /Users/kamen/Fit12/LOSSY_1C_TIERS_SPEC.md (Fit12 08:5x).
  - The name is "1c-tiers"; "C3" is CERTIFY98's t30c.
- **The variant (V2 only):** lossy-45 + R57_TIERS=1c R57_TIERS_LAM=0.016 R57_TIERS_S=0.98
  R57_TIERS_SETS=humaneval,humanevalx,mbpp.
  - It acts in verify checks only, in the check's MoE layers after the router.
  - Per check and layer: importance I_e = sum_d s^d w(d, e)^2. 4-bit if I_e > 2.775 lambda, MSB-3 if > 0.8457 lambda,
    else dropped (weight 0, no renormalisation). The reference code is BigMoE bigmoe/tiers1c.py (c1c217b).
  - On its sets it replaces QOM's keep tables in the check; they are never stacked.
  - Prefill, both drafters, the commit replay and the dense side are unchanged. No V1, V3, V4 or close-call.
- **The base:** lossy-45 on the same new snapshot, with R57_TIERS unset. It is a new cached base, not lx45cc1, so base
  and variant differ only by the four flags (amendment 1d).
- **Engagement:** CARD_ENGAGE=lossy_check:e3+edrop+lmsl_rows (amendment 1d), per run, not "any".
  - It must be > 0 on every code run, and 0 or absent on every IFEval run. The base arm has no field.
  - Information: Fit12's map windows put e3 + edrop at ~3,260 per full 16-row check (48 layers).
- **The steps** (runner 64, CARD_SCOPE=code). Each arm runs in its own processes from a cold start, in the base's order
  and process split (amendment 1c).
  1. **B0:** the base's mini-panel (tier 0, the 40 code keys), against the exact refs (information). It is V0's
     reference, with the same keys, order and split.
  2. **V0:** the variant's mini-panel against B0. A KILL (net loss >= 4 of 40, or an empty row) ends the variant.
  3. **B1:** the base forward (tier 1 code: 264 code keys + IFEval 100), against the exact refs (information).
  4. **V1:** the variant forward, with CARD_REF = B1 and CARD_IFEVAL_IDENT=1. A FAIL or an interim STOP-FAIL ends it.
  5. **B2:** the base reverse.
  6. **V2:** the variant reverse against B2, with CARD_FWD_RUN / CARD_FWD_REF. Both orders pooled give the tier-1
     verdict.
  - **Why this order:** a KILL or FAIL stops before the next base run. A bad variant then costs ~15 min or ~2.5 h, not
    ~5 h.
- **The lines, as registered:**
  - tier 0: the KILL line;
  - tier 1 CODE-ONLY: per-set decision 4; the pooled code line (> -3 pt AND McNemar p >= 0.05); the interim
    (STOP-FAIL at z <= -2.8 after 132 code keys); the RUNAWAY guard (amendment 3a);
  - IFEval: tokens identical to the base on 100 / 100, with engagement 0.
  - The speed read-outs (tools/lossy_speed.py) are information. The gate judges quality, never speed.
- **Two checks on the new snapshot:**
  - **The unset path (information; flagged if it fails):** B1 against lx45cc1's forward run (same order and split).
    Tokens must be identical on the sets outside QOM (humanevalx 104, IFEval 100), as lx45cc1's two orders were. The
    QOM sets are information.
  - **QOM's validity check:** runner 64 VOIDs a run in R57_QOM_SETS whose prefill shows no QOM use (pf.qom_opt > 0).
    1c replaces QOM in the check only, so it should hold. BigMoE's key test must show it on the variant's HumanEval key
    before arming.
- **Memory:** a 43 GB gate (CARD_MEM 39) for both arms, as for lx45cc1 (peak 42.2 GB). 1c reads fewer expert bytes.
  Amendment 4a moves the gate if the key test peaks above 42.5 GB.
- **Time:** B0 + V0 ~15 min; each tier-1 order ~70 min (lx45cc1's). That is ~5 h if every step passes.
- **The slot (NewIdeas orders it):**
  - Tue by day, after night1117 and its tier-0a (results/T0A_1117_END) and BigMoE's 127-node dry run;
  - or Wed by day before the capture, never touching its preparation.
  - The drafter line wins any collision. Kamen's away rule and the yields apply, and the cutoff is 14:30.
  - A tier-1 run cut by a window's end resumes in the next window (CARD_RESUME=1). A tier-0 run starts fresh.
- **What a PASS licenses:** the 16-row form in CODE-ONLY mode, switched by benchmark set. The package's `--kind code`
  routing is not tested.
  - The 32-row form's quality rests on the map's 32-row PASS until a 32-row lossy loop exists.
  - Adoption stays Kamen's call after tier 2 (the card gate).
- **Amendment 4a pins, before the first run:**
  - the snapshot trees, the screen's and the .so's sha256, both env blobs (each env diff checked: the variant is the
    base + exactly the four R57_TIERS flags), runner 64's sha256, and the dry runs;
  - BigMoE's section-6 checklist: the CPU self-check, the unset path, the scope, one scored HumanEval key and one IFEval
    key per env on the real loop (with the counter and pf.qom_opt), and the limits.

## Amendment 4b (Mon 2026-11-16 08:5x, before any 1c-tiers run): QOM's validity rule under 1c; amendment 4 corrected
- **The correction:** BigMoE pointed it out at 08:5x, and it was checked here in scripts/r57_screen_fast.py. Amendment 4
  said runner 64's QOM check reads the prefill and "should hold" under 1c. Both claims are wrong.
  - runs.json "pf" is the expert store's stats delta over the prompt (st.stats() after minus before), not the
    prefill's.
  - pf.qom_opt counts the loader's optional uses (bit 21). Only QOM's per-check mask in _fast_call creates them, in
    verify checks.
  - lossy-45 has no R57_QOM_MTP, so QOM engages only in the verify checks of R57_QOM_SETS (humaneval, mbpp).
  - 1c replaces QOM's keep in every check of R57_TIERS_SETS. So the variant's humaneval and mbpp runs read
    pf.qom_opt = 0, and runner 64's rule (> 0 on every run in R57_QOM_SETS) would VOID every one of them.
- **The rule from now on (runner 65; each arm's env decides it):**
  - **R57_TIERS unset** (the base, and every earlier config): unchanged, pf.qom_opt > 0 on every run in R57_QOM_SETS.
  - **R57_TIERS set:** on the sets in both R57_QOM_SETS and R57_TIERS_SETS, pf.qom_opt must be 0 on every run. That
    shows 1c replaced QOM rather than stacking on it, as the spec requires. A QOM set outside R57_TIERS_SETS keeps the
    > 0 rule.
  - The engagement rule is unchanged: lossy_check e3 + edrop + lmsl_rows > 0 on every code run, and 0 or absent on
    IFEval.
  - A breach VOIDs the run, as before. R57_TIERS without R57_TIERS_SETS stops the runner.
- **Runner 65** (tools/day_runner65.sh, sha256 9848a30c6a15d682...) is runner 64 with only this change; its logs are
  results/day65_*.
  - It is a new file because runner 64 (5cab78942ae9501c, unchanged) serves tonight's armed chains: night1117 and its
    tier-0a follower.
  - The 1c-tiers arms run on runner 65, both base and variant. Amendment 4a pins its sha256.
- **Checked** (results/lossygate/runner65_checks_2026-11-16.txt):
  - The QOM check, extracted from each runner, on 7 synthetic runs.json cases: PASS.
    - With no 1c sets, runner 65 == runner 64.
    - The replaced variant passes on 65 and VOIDs on 64. A stacked variant VOIDs on both.
    - A cut run is ignored. A chat-QOM env keeps its > 0 rule outside 1c's sets.
  - Dry runs of runner 65 (cc1 pins, tier 1 code):
    - lx45cc1's env: rc 0, 364 / 364, the rule logged as before;
    - a scratch copy with the four 1c flags (only to exercise the parse): rc 0, logging "1c sets
      humaneval,humanevalx,mbpp";
    - R57_TIERS without R57_TIERS_SETS: rc 2.
    - The base's key files were unchanged by these dry runs.

## Amendment 4a (Mon 2026-11-16 09:1x, before any 1c-tiers run): the pins, the dry runs and the slot rules for 1c-tiers
- **The snapshot t1c1** (BigMoE 68274ff; BigMoE's HEAD e6fcaea0 holds it, clean), checked here:
  - CARD_TREES=data/r57/snap/t1c1=04180c59e76e39b5deffd8e38c5ac277c179a005,data/r57/snap/frozen_t1c1=14bb1462b292d2cb366401cf4d30e8967e79a124,bigmoe=d92cdb4b9e81d386e4170323d50f11b34147aa5f
  - CARD_SCREEN_SHA=8d56cb717843 (data/r57/snap/t1c1/r57_screen_t1c1.py, sha256 8d56cb7178430bbb...); CARD_SO_SHA=ff69fa24a745
    (rpl2's .so).
- **The env files** (BigMoE data/stack, tracked and committed):
  - base lx45t1c1, blob c9ebc0cfe9cb6697aefa2cc44c08b30b035fb58e;
  - variant t1cx45t1c1, blob d4b1217db0234aff6848e820fe1199f68cb28211.
  - Checked here: the base line equals lx45cc1's verbatim (R57_QOM_SETS=humaneval,mbpp; no R57_TIERS). The variant
    line equals the base's plus exactly ",R57_TIERS=1c,R57_TIERS_LAM=0.016,R57_TIERS_S=0.98,R57_TIERS_SETS=humaneval,humanevalx,mbpp".
  - Each comment names its own tag and the snapshot's screen, with C 256 and MEM 45.
- **BigMoE's section-6 checks on t1c1 (08:5x-09:1x):**
  1. The CPU self-check passes: tiers_np against Breakthrough's masks on the 12 code windows, and the MLX form bit for
     bit.
  2. The wiring test passes (scripts/test_t1c_wiring.py on the t1c1 screen):
     - 0 mismatches over 3,095,040 slots, in the drop mask, the no-LSB marks and the counter;
     - e4 + e3 + edrop = the union on every check-layer;
     - means at 16 rows 76.64 = 8.65 + 20.52 + 47.47, against Fit12's reference 76.6 / 8.7 / 20.5 / 47.5.
  3. EvalSuite's head test passes on both envs.
  4. Limits: 1c never adds ids (a drop maps to the row's rank-0 id), so MAXK and C are unchanged.
- **Leap's ref2 change** (b43efc4), read by BigMoE for the R57_TIERS-unset path:
  - Every new branch is gated on a 2-D nolsb mask, which only the 1c wiring creates. The 1-D path keeps the old
    branches.
  - Leap's own check: the 1-D and no-nolsb paths give 168 arrays bit for bit, live against new.
  - The frozen copy equals b43efc4's ref2 modulo 11 path-rewrite lines.
  - So the base should be bitwise as before. The key test and the gate's own unset-path check (amendment 4: B1 against
    lx45cc1 on humanevalx + IFEval) test it.
- **Runner 65** (sha256 9848a30c6a15d682...), dry runs at 09:14 on these pins:
  - B0 (base tier 0, exact refs): rc 0, 40 / 40, the base's QOM rule.
  - V0 against B0 before B0 exists: rc 2, "holds 0 of 40", as it must.
  - V0 with the exact refs as a stand-in: rc 0, with the 1c QOM rule and the engagement field.
  - B1, B2 (base forward / reverse): rc 0, 364 / 364.
  - V1, V2 (the variant forward / reverse, stand-in refs, IFEval identity, V2 with the pooled settings): rc 0.
- **The steps' settings** (CARD_SCOPE=code, the pins above, CARD_MEM 39 so the gate is 43 GB):
  - B0: CARD_TIER=0, CARD_TAG=lx45t1c1, CARD_REF = the exact refs.
  - V0: CARD_TIER=0, CARD_TAG=t1cx45t1c1, CARD_REF=card_lx45t1c1_t0code, CARD_ENGAGE=lossy_check:e3+edrop+lmsl_rows.
  - B1: CARD_TIER=1, the base, exact refs. V1: the variant, CARD_REF=card_lx45t1c1_t1code, CARD_IFEVAL_IDENT=1, the
    engagement field.
  - B2 / V2: the same with CARD_ORDER=rev, CARD_REF=card_lx45t1c1_t1code_rev, and V2's CARD_FWD_RUN=card_t1cx45t1c1_t1code
    / CARD_FWD_REF=card_lx45t1c1_t1code.
- **Arming needs BigMoE's real-loop key test** (scripts/t1c_keytest.py --snap t1c1; it waits for a 45 GB slot).
  - It decodes humaneval/HumanEval/45, humanevalx/HumanEval/23 and ifeval/2239 end to end on both envs.
  - Base: tokens equal lx45cc1's rows on humanevalx and IFEval; pf.qom_opt > 0 on humaneval; no lossy_check.
  - Variant: e3 + edrop > 0 on humaneval and humanevalx, and lossy_check absent on IFEval; IFEval tokens equal the
    base's; pf.qom_opt == 0 on humaneval.
  - A per-key lifetime peak above 42.5 GB moves the gate before arming.
- **Start gates (NewIdeas 08:4x), all required before any step:**
  1. results/T0A_1117_END written after the follower's arming (Mon 08:40:20), on any end;
  2. BigMoE's dry-run marker ~/BigMoE/data/deepdraft/ceiling/N127_DRY_1117_DONE, with mtime >= Mon 09:05 (the dry
     run moved to today);
  3. this amendment and the key test above;
  4. the away rule, with no other session's run above 16 GB live (runner 65's per-process yield holds the rest).
- **Windows:**
  - **Tuesday by day:** B0 -> V0 -> B1 -> V1, as far as the window allows. The cutoff is 14:30.
  - **Wednesday by day:** the drafter line goes FIRST: t30c's dev eval, its TOPK / TREE dumps if t30c PAYS, and anything
    the Wednesday pins need. Only on NewIdeas' word does the rest resume (CARD_RESUME for a cut tier-1 run), then
    B2 -> V2.
    - The cutoff is 14:30. The capture starts no earlier than Wed 23:30 (BigMoE), so every step ends >= 2 h before
      it.
  - Anything left waits until after the proofs (Thu-Fri nights).
  - A KILL or FAIL stops the chain, as registered.

## Amendment 4c (Mon 2026-11-16 09:3x, before any 1c-tiers run): 1c-tiers starts TODAY; the key test becomes a start gate; the chain
- **Today's start** (NewIdeas 09:2x; the orchestrator agrees): t30c is off (S10 PASS2 FAIL), so the drafter line needs no
  GPU today. the owner away until ~15:00, with his apps closed.
  - Today's start gates: BigMoE's dry-run marker (mtime >= 09:05); the key test's PASS (below); this PREREG's 4a; and
    the away rule.
  - T0A_1117_END does not apply today. It is back in force on Tuesday.
- **The key test becomes a start gate rather than an arming condition** (BigMoE 09:1x, the orchestrator's arrangement).
  - The marker is ~/BigMoE/data/r57/T1C1_KEYTEST_DONE, written on any end by BigMoE scripts/t1c_follow.sh (c7cf64a8):
    "T1C1 KEYTEST END <time>: PASS|FAIL|STOPPED|NOT RUN (<per env verdict, peak, swap>)".
  - No step starts before a PASS dated >= 09:20 today.
  - A FAIL, or any env's "peak <x> GB" above 42.5, means no run (rc 5).
  - STOPPED or NOT RUN: the chain waits for a re-test.
  - Amendment 4a's criteria for the key test stand; BigMoE's PASS covers them.
- **The chain:** tools/t1c_chain.sh (sha256 d7d9001d6549811d...), runner 65, the settings of 4a. Its phases:
  - **mon (today) / tue:** B0 -> V0 -> B1 -> V1 as far as the day's 14:30 cutoff allows. It never starts B2 or V2. Tue
    adds the T0A gate.
  - **wed (armed on NewIdeas' word, after the drafter line):** the steps not done, then B2 -> V2.
  - A done step is skipped. A cut tier-0 run starts fresh; a cut tier-1 run resumes.
  - V0's KILL or V1's FAIL ends the variant. A run without a verdict, VOID or INCOMPLETE stops the chain.
  - Every end writes results/T1C_<PHASE>_END.
- **Checked:**
  - DRY passes of the mon and tue phases through runner 65: all six runner calls rc 0.
  - A stub simulation, 13 scenarios (results/lossygate/t1c_chain_sim_2026-11-16.txt):
    - stale gate markers ignored;
    - a full Tuesday that stops before B2;
    - V0 KILL;
    - yields (tier 0 fresh, tier 1 resumed);
    - a cut B1 that Wednesday resumes and completes;
    - V1 FAIL, after which Wednesday has nothing to run;
    - HOLD (rc 8); the last start missed (rc 6); INCOMPLETE (a stop, not a cut);
    - the key test FAIL (rc 5); STOPPED then PASS (it waits, then starts); a 43.1 GB peak (rc 5); a PASS without
      peaks (it starts and notes it);
    - the mon phase without the T0A gate.

## Amendment 4d (Mon 2026-11-16 09:3x, before any 1c-tiers run): if V1 passes, B2 -> V2 run on Tuesday by day
- **Why:** NewIdeas (09:3x): t30c is off, so Tuesday by day has no drafter-line GPU need. Finishing the gate on Tuesday
  keeps Wednesday clear for the capture's preparation.
- **The rule:** if V1 passes (today or Tuesday), B2 -> V2 run on Tuesday by day.
  - The usual gates apply: after T0A_1117_END, under the away rule, with the 14:30 cutoff and one > 16 GB job at a time.
  - Anything left after Tuesday's 14:30 goes to Wednesday by day after the drafter line, or after the proofs.
- **How: a follower,** tools/t1c_tue_follower.sh (sha256 e6ef42aaaba31a10...). The armed chain's file serves today's phase
  and must not change under it.
  - It starts when results/T1C_TUE_END (the tue phase's end, after T0A_1117_END) says "V1 PASS", with the tue phase gone,
    read twice 30 s apart. Any other tue end means no run.
  - It then runs t1c_chain.sh's wed phase with Tuesday's date: the done steps are skipped, then B2 -> V2, under that
    chain's own guards. Its log lines read "[wed]" with Tuesday's date, and its end is results/T1C_WED_END.
  - No start after Tue 14:15. Every end writes results/T1C_TUEF_END.
- **Tuesday's neighbours (NewIdeas):**
  - Leap's command-buffer microbench, a quiet window of <= 10 min, takes a gap between steps, never inside a running
    step. A yield is covered by CARD_RESUME.
  - A possible early-death dev pass is Tuesday NIGHT. It does not touch these runs.
- **Checked:**
  - The DRY pass ran the wed phase with Tuesday's cutoff: six runner calls rc 0.
  - A stub simulation, 5 scenarios (results/lossygate/t1c_tue_follower_sim_2026-11-16.txt):
    - a "V1 PASS" end runs B2 -> V2, with the done steps skipped;
    - a FAIL end, or the tue phase gone without a marker: no run (rc 5);
    - a stale marker is ignored;
    - past the last start: rc 6.

## Amendment 4e (Mon 2026-11-16 11:3x, before any 1c-tiers run): a memory-gate fallback for 1c-tiers, 43 -> 42 GB (the orchestrator 11:3x)
- **Why:** runner 65 starts each CHAT process only at NO-SWAP >= MEM + 4 = 43 GB (CARD_MEM 39, amendment 4a).
  - By day the Mac read 35.1-38.5 GB at 11:07-11:15, with MTP Head's S12 up, and 40.1-40.5 GB after S12 (11:18-11:21).
  - BigMoE lowered its key test's start gate from 42 to 41, then to 40, for the same reason.
- **Expected, with no change (the orchestrator):** the key test's ~43 GB process pushes idle memory into the compressor.
  Its exit should lift NO-SWAP to ~48-50 GB, and back-to-back processes keep it there.
  - The same happened at 06:55 (44.2 -> 48-50 after the first run).
  - It happened at 03:27 too: a45's exit lifted it to 50, and that chain started at 49.
- **The fallback** runs only if 43 has not opened 10 min after BigMoE's key-test PASS marker (the marker's mtime).
  - The per-process gate drops to 42 GB for the mon and tue phases and for the Tuesday follower's B2 -> V2.
  - The at-peak evidence (the orchestrator's figures):
    - lx45cc1 peaked at 42.2 GB;
    - a45 started at 40.7 GB against its 42.71 peak, with swap +0;
    - the rpl2 key test started at 44.2 against 42.6, with swap +0.
  - Unchanged:
    - the +2 GB swap guard and every stop;
    - CARD_MEM 39, so the share lock still says "MEM 39 GB";
    - the cutoffs, the pins and every scoring rule.
  - This is a gate change, not a scoring change.
- **How, with no armed file changed:** tools/day_runner66.sh (sha256 6032f95fe47e5f97...) is runner 65 with one change.
  - The CHAT step's gate is CARD_GATE, in whole GB from 41 to 49, inherited from the caller's environment. A value out
    of range, or not a number, stops the runner. Unset, the gate is MEM + 4 as before.
  - Its logs are results/day66_*.
  - **Today:** TERM the mon phase (pid 86890) while its B0 waits for memory, with no GPU process running. Then start
    t1c_chain.sh's mon phase again with `CHAIN_RUNNER=tools/day_runner66.sh CARD_GATE=42`. Its start gates already
    hold: the dry-run marker and the key-test PASS.
  - **Tuesday:**
    - TERM the tue phase (pid 86926). The follower (pid 95134) then ends with "no run".
    - Re-arm the tue phase the same way.
    - Re-arm the follower on the new tue pid (CHAIN_TUEPID) with the same two settings. The wed phase it starts
      inherits them.
  - The chain's start line still prints "runner 65; NO-SWAP gate 43 GB", because its file is unchanged. Runner 66's log
    names the gate it uses ("CHAT gate 42 GB NO-SWAP, CARD_GATE").
  - If 43 opens in time, nothing changes and runner 66 is not used.
- **Checked** (results/lossygate/runner66_checks_2026-11-16.txt):
  - The diff against runner 65 is the header, the log names, the gate line, the CARD_GATE check and the start / end
    labels. bash -n passes.
  - A stub test of runner 66's own gate and check:
    - unset gives 43; 42, 41 and 49 are accepted;
    - 40, 50 and "4x" stop the runner (rc 2);
    - the gate waits at 41 for 42 and at 42 for 43, passes at once at 42, and stops at the cutoff (rc 6).
  - A DRY chain pass through CHAIN_RUNNER runs right after the key test, before any use.

## Amendment 4f (Mon 2026-11-16 12:0x, during the 1c-tiers run, before B1's verdict): run the gate to completion today; operational only
> (a quote from the project owner, kept private)
  use its compute until I tell you". The day limits and the Kamen-at-the-Mac yield are lifted until he says otherwise.
  - The orchestrator: run the 1c-tiers gate to completion today, past the 14:30 cutoff: B1 -> V1, then B2 -> V2 if V1
    passes, with no at-Mac yield.
  - Keep the memory gate, the +2 GB swap guard and HOLD / PAUSE / lid.
- **The changes** (the cutoff and the yield only; NO scoring change):
  - The cutoff is 22:30 today, and the last start 22:15. That is before tonight's 45 GB order (M4 23:05 -> E14 00:30
    -> night1117).
  - There is no Kamen-at-the-Mac rule, neither as a start condition nor as a mid-run yield.
  - B2 -> V2 run today, right after V1, if V1 passes. Amendment 4d's Tuesday follower becomes the fallback.
  - Unchanged:
    - runner 65 and its 43 GB memory gate;
    - the +2 GB swap guard, and the per-process yield to another session's MEM run above 5 GB;
    - HOLD / PAUSE / lid;
    - every pin and every scoring rule.
- **How:** tools/t1c_chain2.sh (sha256 822c2fc2db41f880...) is t1c_chain.sh plus three settings and one fix.
  - The settings:
    - CHAIN_CUTOFF: the runner's cutoff;
    - CHAIN_NOYIELD=1: no at-Mac rule;
    - CHAIN_TAG: the log tag and the end marker.
  - The fix: a recorded VOID / INCOMPLETE tier-1 verdict now stops the chain (rc 9, a decision). t1c_chain.sh counts it
    as done.
  - **12:01:21-31:** the mon phase (pid 86890) was stopped by TERM after B1's 7th process. B1's first half (132 rows)
    is kept.
    - The runner was in B1's CPU-only interim scoring, so no GPU work was lost.
    - Its stack process and the share lock were released (checked). T1C_MON_END reads "stopped by a signal during: B1".
  - **12:01:46:** chain2 started (pid 60135): phase wed on 11-16, tag mon2 (log [mon2], end results/T1C_MON2_END),
    CHAIN_CUTOFF 22:30, CHAIN_NOYIELD=1.
    - B0 and V0 were already done.
    - B1 resumed at 12:02:16 (CARD_RESUME=1; 132 rows kept; the module manifest is checked against B0's).
  - The tue phase (pid 86926) and the Tuesday follower (pid 95134) stay armed, as the fallback if today's run stops.
    - Caveat: t1c_chain.sh counts a recorded VOID / INCOMPLETE tier-1 verdict as done.
    - If one is recorded today, EvalSuite stops both before Tuesday.
- **Checked:**
  - The diff against t1c_chain.sh is the header, the three settings and the fix. bash -n passes.
  - A stub simulation, 6 scenarios (results/lossygate/t1c_chain2_sim_2026-11-16.txt), from today's state (B0 and V0
    recorded, B1 partial):
    - N1: with Kamen at the Mac throughout and NOYIELD, B0 and V0 are skipped, B1 resumes, and V1, B2 and V2 run to
      "done", with the 23:30 test cutoff passed to every runner call.
    - N2: a V1 FAIL ends before B2.
    - N3: a recorded VOID B1 stops with rc 9 (the fix).
    - N4: without NOYIELD it waits while Kamen is at the Mac, then starts.
    - N5: a HOLD during B2 gives rc 8, with the runner stopped.
    - N6: a past cutoff means no start (rc 6).

## Amendment 5 (Mon 2026-11-16 12:3x, before any config, snapshot or run): the check-bytes quality gates G1 and G2 (PLAN900 bet 5), the frame; the pins follow in 5a
- **Why:** NewIdeas' 900 plan (~/NewIdeas/research/week98/PLAN900.md, 914f322), bet 5 "check BYTES".
  - Three changes aim together at a code-scoped check of <= ~5.5 GB at quality: C6 (1c lambda 1.8e-2), int8 dense,
    and V4 (GDN / attention at 3 bits).
  - Breakthrough brings the bytes and Leap the kernels; BigMoE builds the stack, as for t1c1.
  - NewIdeas 12:2x picked option A: two stacked gates, each against the base, bisecting only on a FAIL. That is
    Breakthrough's rule: one combined same-scores test per batch.
- **The arms:**
  - **G1 = C6 + int8 dense:** Leap's Friday measured config.
  - **G2 = G1 + V4:** GDN / attention at 3 bits.
  - **The base:** lossy-45 as shipped (lx45cc1's line verbatim) on the snapshot that holds the new code, with that
    code unset. It is the same line as 1c-tiers' base lx45t1c1.
    - A config with no new code (env only on t1c1) uses today's lx45t1c1 runs (B0 / B1 / B2) as its base.
- **The steps.** Once per snapshot, the base:
  - B0: tier 0 against the exact refs. This is information, and it is V0's reference.
  - B1: forward tier 1, with the unset-path check: tokens identical to lx45t1c1 forward on humanevalx 104 + IFEval 100
    (tools/t1c_unset_path.py). The QOM sets are information.
  - B2: reverse.
  - Then each gate in order, G1 then G2:
    - V0: tier 0 against B0;
    - V1: forward against B1;
    - V2: reverse against B2, with both orders pooled.
- **The rules,** as amendment 4, CODE-ONLY:
  - Tier 0: KILL if reference-only minus arm-only is >= 4 on the 40 code keys, or if any row is empty.
  - Tier 1:
    - each set: decision 4, i.e. delta > -5 pt AND exact McNemar p >= 0.05 (DIFFERS (HIGHER) passes);
    - the pooled code line: delta > -3 pt AND p >= 0.05;
    - the interim STOP-FAIL at z <= -2.8 after 132 code keys;
    - the RUNAWAY guard (amendment 3a).
  - IFEval: tokens identical to the base on 100 / 100 (the switch-off check), IF every change in the config engages
    only on the code sets. Otherwise IFEval is decision 4 against the base ("no harm"). 5a states which applies to each
    config.
  - A KILL or FAIL ends that gate.
  - **Bisect:**
    - If G1 fails, G2 does not run. Its slot goes to C6 alone, then int8 dense alone, each against the same base.
    - If G2 fails after G1 passed, V4 on top of G1 is the cause, and no further bisect is needed.
- **What a PASS licenses:** a screening line, as for every lossy gate. A code-scoped PASS licenses a code-only mode.
  Adoption stays Kamen's call after a full verdict (tier 2, the card gate).
  - The speed read-outs (tools/lossy_speed.py) are information. The gate judges quality, never speed.
- **Memory:** each process needs NO-SWAP >= MEM + 4 GB (runner 65's rule). The MEM comes from the config's key test,
  whose every peak must be <= the gate - 0.5 GB. The +2 GB swap guard, HOLD / PAUSE / lid, and one > 16 GB job at a
  time stay. Kamen 11:5x lifted the day limits and the at-Mac yield until he says otherwise.
- **The slots** (NewIdeas 12:2x):
  - **Tue by day:** Leap's engine slot #1 FIRST (~40 min, from ~09:30-10:00, after night1117). Then the new snapshot's
    base (~2.5 h), then G1 (~2.5 h), so Leap has G1's verdict by Tue evening.
  - **Wed by day:** G2 (~2.5 h) and Leap's engine slot #3. Everything ends >= 2 h before the capture (>= 23:30).
  - If the inputs aren't ready Tue morning, the base slides and G1 / G2 follow in order on Wednesday. Anything left
    after Wednesday waits until after the proofs (Thu-Fri nights).
- **The inputs,** by Tue morning (Breakthrough / Leap, with BigMoE building the stack; NewIdeas forwarded the list):
  - per config, BigMoE-format env file(s) pinned by blob;
  - the snapshot: its tree hashes and the ext .so;
  - MEM;
  - a real-loop key test PASS like t1c1's.
- **Amendment 5a** pins these, plus the runner, the chain and its dry runs, before the base starts.

## Amendment 5a (Mon 2026-11-16 12:4x, before any config, snapshot, screen or run): G1's IFEval check and memory; G2 runs only on a DENSE-3 PASS (NewIdeas 12:3x); the pins follow in 5b
- **G1's scope:**
  - int8 dense is code-scoped. Leap: `R57_INT8D_SETS=humaneval,humanevalx,mbpp`, toggled per item's set like
    R57_TIERS_SETS.
  - C6 is an R57_TIERS env change on t1c1, so it is set-scoped as well.
  - So G1's IFEval check is TOKEN IDENTITY against the base on 100 / 100 (amendment 5's switch-off check).
- **G1's memory:** int8 dense adds +1.7 GB resident for its own weight layout.
  - G1's MEM comes from G1's own key test, whose peak includes it. The gate is MEM + 4 GB NO-SWAP.
  - Example: at t1cx45t1c1's 42.38 GB + 1.7, MEM is ~41, so the gate is ~45 GB. That opens by day only right after
    another big process exits (the compressor effect, as at 11:24: 40.5 -> 49.4).
  - If it does not open, the 4e fallback is NOT carried over. A lower gate for G1 needs its own amendment with its
    at-peak evidence.
  - The base on the new snapshot leaves int8 unset. Its MEM is its own key test's.
- **G2 is CONDITIONAL. It runs only on a DENSE-3 PASS.**
  - The screen is Breakthrough's DENSE-3 map arm: 3-bit dense on the decision map, Wed ~07:00-09:00, ~1 h, ~14 GB.
  - G2 keeps its Wednesday slot only if Breakthrough records DENSE-3's verdict as PASS (<= 1.34 % code flips) before
    the slot. BETWEEN or FAIL drops G2.
  - Why (NewIdeas):
    - The "3.86 % code flips" that closed V4 in Fable B's notes was the cf 0.3 chat arm's MULTI flips.
    - DENSE-3's map arm never ran.
    - The only prior is 17-18 % relative rms per projection (Leap, 11-04).
  - **CORRECTED 12:4x (NewIdeas): the rationale above is wrong. The rule stands, and its prior is worse.**
    - 3-bit dense WAS flip-tested on Sun 15 Nov, and it FAILED against the 1.34 % line: Breakthrough's
      bt_ga3 (RTN) gave 4.43 % code flips and bt_ga3awq (AWQ) 3.86 %.
    - Only Leap's DENSE-3 midpoint variant never ran. Its prior is FAIL (~3.5-4 %).
  - If G2 drops, its slot goes to G1's bisect if G1 failed (amendment 5: C6 alone, then int8 dense alone). Otherwise it
    goes to a sibling-state follow-up, if Sandbox98's analog passes (NewIdeas' call, under its own registration).
  - EvalSuite reads only DENSE-3's recorded verdict word for this condition. The screen and its rules are Breakthrough's.
- **Amendment 5b** pins the env blobs, the snapshot (its tree hashes and ext .so), each env's MEM and key test, the
  runner, the chain and their dry runs, before the base starts.

## Amendment 4g (Mon 2026-11-16 12:5x, during the 1c-tiers run, before V1): a pause after B1 for Sandbox98's SIB test (the orchestrator 12:4x); operational only
- **Why:** the orchestrator 12:4x.
  - Sandbox98's SIB (sibling-state drafting, PLAN900 9b) is the one test aimed at the binding wall for Kamen's
    97.8 % / 900: the drafter's drift.
  - Its runner needs NO-SWAP >= 13 GB, which never opens beside the 1c-tiers processes (~42 GB).
  - MoonShots' quiet ticket may take a few minutes of the gap first.
- **The pause:** chain2 (pid 60135) is stopped by TERM at the clean boundary right after B1's verdict is recorded.
  - V1 is not started: its runner is in its CPU-only setup when the TERM lands.
  - T1C_MON2_END then reads "stopped by a signal during: V1". That marker is a pause, not the gate's end. Breakthrough
    was told (12:5x) to key tonight's map screen on T1C_MON3_END instead.
- **The resume:** tools/t1c_resume_follower.sh (sha256 9a1004e54a55d3fd...).
  - It waits until pid 60135 is gone; that moment is the pause's start.
  - Then it waits for Sandbox98's ~/Sandbox98/logs/sib_END, or for 15:30, whichever comes first.
  - The marker counts only if it was written after the pause's start AND its first line starts with
    "rc 0: main read-out done" (the orchestrator 12:5x).
    - A stale stop marker ("rc 143: stopped by a signal ... waiting for room", from a mistaken test arm) existed at
      12:53, so neither a stopped attempt nor plain existence counts.
    - SIB's fine-tune marker (logs/sib_ft_END) is ignored.
  - Then it runs t1c_chain2.sh: phase wed on 11-16, tag mon3 (log [mon3], end results/T1C_MON3_END), cutoff 22:30,
    no at-Mac rule.
  - B0 / V0 / B1 are skipped as done; a recorded VOID / INCOMPLETE stops it.
  - Then V1 -> B2 -> V2 run as planned (amendments 4d / 4f), inside the 22:30 cutoff.
- **Unchanged:** every pin, gate, guard and scoring rule; the 22:30 cutoff; no at-Mac rule (amendment 4f).
- **Checked** with a stub simulation, 7 scenarios (results/lossygate/t1c_resume_follower_sim_2026-11-16.txt):
  - R1: an "rc 0: main read-out done" mark written after the pause began starts it. It passes phase wed, day 11-16,
    cutoff "2026-11-16 23:59" (the space kept), NOYIELD 1 and tag mon3.
  - R2: the resume time already passed starts it.
  - R3: a stale "rc 0" mark (an old mtime) is ignored; it starts at the time.
  - R4: an "rc 0" mark written while the paused chain was alive (before the pause) is ignored; it starts at the time.
  - R5: past the last start, there is no start (rc 6).
  - R6: a TERM while it waits starts no chain; its END reads "stopped by a signal" (rc 143).
  - R7: an "rc 143" stop marker written after the pause is ignored; it starts at the time.
  - The simulation caught a quoting bug first: a quoted "CHAIN_CUTOFF=..." word was taken as the command (rc 127). It
    was fixed before arming.

## Amendment 5b (Mon 2026-11-16 13:0x, before any G1 run): G1's pins (BigMoE 12:5x), checked here; MEM, the key test and the chain follow in a dated addendum
- **Snapshot g1a** (BigMoE abffe6e1, clean against HEAD):

  | piece | pin |
  |---|---|
  | data/r57/snap/g1a tree | bd3d827025324c56445f6a79506fdf90ddfc9bcf |
  | data/r57/snap/frozen_g1a tree | f018c23c3747b7f05d9e91163846959637d7b23f |
  | bigmoe tree | d92cdb4b9e81d386e4170323d50f11b34147aa5f (t1c1's, unchanged) |
  | screen data/r57/snap/g1a/r57_screen_g1a.py | sha256 aa00c0bb204d5fc5... (CARD_SCREEN_SHA aa00c0bb204d) |
  | ext .so | ff69fa24a745 (unchanged) |

- **The envs** (BigMoE data/stack/; the blobs equal HEAD):
  - base **lx45g1a_env.txt**, blob 022d03e6128e38a014f11e22f3e5dd18543a96af: exactly lx45t1c1's 34 pairs (the same line,
    checked here).
  - G1 **g1x45g1a_env.txt**, blob 849a7c242aa1cf600ee7d0483414a7e4adb2cf32: the base plus exactly R57_TIERS=1c,
    R57_TIERS_LAM=0.018, R57_TIERS_S=0.98, R57_TIERS_SETS=humaneval,humanevalx,mbpp, R57_INT8D=1 and
    R57_INT8D_SETS=humaneval,humanevalx,mbpp (checked here).
- **What g1a changes,** checked here with the paths normalized:
  - frozen_g1a = frozen_t1c1 + Leap's week98/int8d.py and dn32.py (7536125). The 407 other changed files differ only in
    their own paths: 712 lines removed = 712 added once the paths are normalized.
  - In the working dir, pflm.py differs only in paths.
  - The screen adds 36 int8 lines, behind `I8 is not None` / R57_INT8D, and changes one line: lossy_check now merges
    the int8 counters. With R57_TIERS and R57_INT8D unset it is None, as before.
  - So by inspection the base path (both unset) is unchanged. B1's unset-path check on g1a tests it.
- **BigMoE's checks:** its head test PASSES on both envs, and the 1c wiring test on g1a PASSES. runs.json lossy_check
  gains "int8" / "int8_rows" on the code sets; they are absent on IFEval.
- **Pending,** for a dated addendum before the base starts:
  - each env's MEM, from BigMoE's key test (scripts/g1_keytest.py: section 6's criteria + int8 > 0 on the code keys +
    per-env at-peak margins and swapouts; data/r57/G1A_KEYTEST_DONE). It runs after the resumed 1c-tiers chain ends;
  - G1's engagement key (CARD_ENGAGE);
  - the runner and the chain (from t1c_chain2.sh), with their dry runs and a stub sim.

## Amendment 5c (Mon 2026-11-16 13:0x, before any G2 config, snapshot or data): G2's composition (NewIdeas 13:0x); it replaces 5a's G2 condition
- **Why:** Breakthrough found that G1 as built reads 6.22 GB per check, not 5.77. g1a is C6 + int8 on lossy-45's
  HC8 + SVH, and V1 (the 16K head list) and V3 (HC mixers / inject / PLE at 4 bits) aren't built.
- **G1 stays as built** (5b) for Tuesday, unchanged.
- **G2 is ONE stacked gate on Wednesday,** its make-up set by tonight's DENSE-3 map arm (in Breakthrough's 4-arm map
  screen):
  - on a PASS (<= 1.34 % code flips): G2 = G1 + V1 + V3 + V4;
  - on a FAIL: G2 = G1 + V1 + V3 + lambda 2.0e-2, the no-V4 route to <= 5.5 GB;
  - on a BETWEEN: the no-V4 route, as 5a kept V4 only on a PASS.
- **The rules** are amendment 5's: against the base, V0 -> V1 -> V2, CODE-ONLY, bisecting only on a FAIL.
  - IFEval is token identity only if every change in G2 is code-scoped; otherwise it is decision 4. The arming
    amendment states which.
- **Its snapshot** (V1 + V3 built by BigMoE or Breakthrough, wired by BigMoE) needs its own real-loop key test before
  arming.
- **G2's base.** Amendment 5 says the base runs once per snapshot. G2's snapshot may instead reuse g1a's base runs
  (B0 / B1 / B2) if two things hold:
  - by inspection, as for g1a, the snapshot adds only code that is inert when unset;
  - a base-path identity run on it gives tokens identical to g1a's B1 on all 204 non-QOM keys. That run uses the base
    env on humanevalx 104 + IFEval 100, ~35-40 min.
  - Otherwise G2's snapshot gets its own base (~2.5 h).
- **If V1 + V3 can't be built by Wednesday morning,** G2 is DROPPED this week. Its slot goes to G1's bisect if G1
  failed, otherwise to the sibling-state follow-up (NewIdeas' call, under its own registration).
- **Confirmed 13:1x (NewIdeas):**
  - BETWEEN takes the no-V4 route.
  - Aim for the base-path identity route. Breakthrough writes V1 / V3 as opt-in installers in int8d's form, and its
    own test shows the unset path bitwise.
  - **Wednesday's order,** by day ~08:00 -> 21:30, ending >= 2 h before the capture:
    1. BigMoE's g2a key test (Tue night or early Wed);
    2. ~08:00, the identity run (~35-40 min), or the full base (~2.5 h) if it FAILs;
    3. G2: V0 -> V1 -> V2 (~2.5 h);
    4. Leap's engine slot #3 (~40 min).
    - Worst case ~6 h, ending ~14:00-16:30.
  - **The drop deadline is ~10:00 Wed:** if V1 + V3 or the g2a key test aren't ready by then, G2 is dropped this week.
