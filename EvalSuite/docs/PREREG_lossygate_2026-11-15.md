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
