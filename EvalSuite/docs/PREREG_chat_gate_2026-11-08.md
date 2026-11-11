# PREREG: the CHAT150 chat gate, a ready same-scores line for lossy chat configs (Sun 2026-11-08 02:4x; before any row)

Written before any row of any chat candidate.
> (a quote from the project owner, kept private)
  a ready IFEval chat gate that can start within 1 h of a candidate. Candidates come from BigMoE as env files, like S4
  and C30. The first is M1-B: s5p_env + R57_QOM_SETS=humaneval,mbpp,ifeval,multi + R57_QOM_MTP=2, QOM gating the MTP
  head's chat drafts.
- **Why a gate is needed:** chat in v1' is already not exact. BigMoE's N45 found 6 of 24 IFEval runs equal to NAX6-N's
  AR; CERT amendment 3. Lossy chat levers need a same-scores line.

## Rows (RELAX arm E's form)
- IFEval, the 100 core prompts (dec_ifeval), and multi, the 24 behaviour prompts (dec_multi; max_tokens 384): 124 rows.
- Two prompt orders: forward = the keys sorted, then reverse. FAST may depend on cache state (QOM), so >= 2 histories.
- 10 prompts per process, at the C in the env file's comment.

## The config (the entry point)
- BigMoE writes ~/BigMoE/data/stack/chat_env.txt per candidate, in final_env / c30_env's format: 'Sat <C>',
  'MEM <n> GB' and 'screen data/r57/snap/<dir>/<file>.py' (a committed snapshot) in the comment.
- The runner applies S4's code checks unchanged:
  - every BigMoE module a process loads equals BigMoE's HEAD;
  - the manifests are equal across processes;
  - the store, swap, memory and the day rule.
- **Engagement:** if the env's R57_QOM_SETS holds ifeval / multi, every row of those sets must show QOM optional uses
  (else VOID).

## The lines (vs NAX6-N, the 45 GB rule)
1. **IFEval same scores:** decision 4 vs NAX6's IFEval rows (results/ref2tnax), per order: paired delta > -5 pt AND
   exact McNemar p >= 0.05, on the 100 core prompts, no stage 2. DIFFERS (HIGHER) passes. PASS = both orders.
2. **multi blunders:** the forward order's 24 multi rows, scored by tfwin against NAX6-N, as CERT and S4 score long
   code. Blunders (NAX6-N's p of a committed token < 0.01) <= 1 per 10K committed tokens.
- **Chat gate PASS = both lines.** A FAIL names its line.
- VOID: the code checks or engagement fail. INCOMPLETE: a "no free slot" abort or missing rows.

## Information
- IFEval vs REF2-T and REF.
- Rows token-identical to NAX6-N, per set and order.
- multi deviation rate.
- Token agreement between the orders.
- Cap hits.
- Rough tok/s from the runs (not an A/B).

## Runs
- Runner 59 (tools/day_runner59.sh, built from runners 54 / 58). Each candidate's outputs go to results/chat_<tag>_*,
  results/CHAT_<tag>_DONE.
- Cost: ~124 rows x 2 orders at ~45-50 tok/s ≈ 30-40 min, plus tfwin on 24 rows ≈ 10 min.
- It never runs during CERT; it starts after CERT's end, or when NewIdeas' queue says.

## Amendment 1 (Sun 09:2x, before any m1bi row): m1bi, the split of M1-B's multi FAIL
- **The candidate:** BigMoE's data/stack/chat3_env.txt (2187975), tag m1bi, dry run OK 09:19.
  - It is M1-B + COH with QOM on humaneval / mbpp / ifeval but NOT multi.
  - Same CHS4 snapshot (342f914), the same code pin as m1b (6b6a4fb64229), Sat 256, MEM 47 GB.
- **The lines are unchanged.** IFEval same scores in both orders, and multi blunders on the forward order.
- **What multi measures here:** S5' + COH with no QOM on multi. BigMoE: QOM_MTP only acts where QOM runs. That is the
  base config's first multilingual line; CERT had no multi line.
- **Reading, fixed now:**
  - multi PASS: QOM is the cause of m1b's multi FAIL.
  - multi FAIL: the cause is in S5''s own non-exact levers (the bisect is BigMoE's). The release config's multilingual
    chat would then be untested by CERT and failing.
- **Validity check (post hoc, from the screen's row stats):** every multi row must show no QOM optional uses
  (pf.qom_opt == 0). Otherwise the split is VOID.
- **A PASS is a diagnosis, not a shippable config.** In the engine, ifeval and multi are both type 'chat'. "QOM on
  English chat only" needs a request-level switch (e.g. by language or script) and its own gate.

## Amendment 2 (Sun 09:4x, before any MGSM row): Kamen's 11-08 rule — known benchmarks decide; MGSM replaces the multi line
> (a quote from the project owner, kept private)
  compare against known benchmarks of the model. Not something new".
- **Line 2 (multi blunders) is information from now on**, for every candidate, m1b included. So are the long-code
  blunder lines elsewhere. CERT, S4, C30 and STACK passed those lines, so their verdicts do not change.
- **Amendment 1 (m1bi) is withdrawn:** it decided on the multi line. As information it would still split QOM from S5''s
  levers, if NewIdeas queues it.
- **The new line 2: MGSM same scores.**
  - MGSM is the published multilingual grade-school math benchmark (Shi et al. 2022). EvalSuite holds 80 items:
    10 languages (bn de es fr ja ru sw te th zh) x 8, max 1,024 tokens, data/items/dec_mgsm.jsonl.
  - Scoring: decode_report's MGSM scorer (dec_math's final-answer check).
  - Decision 4 vs NAX6-N per order: delta > -5 pt AND exact McNemar p >= 0.05, no stage 2. DIFFERS (HIGHER) passes.
    Two orders (forward = the keys sorted, then reverse). PASS = both orders.
- **The chat gate = IFEval same scores (unchanged) + MGSM same scores.** m1b's IFEval PASS stands; it needs only MGSM.
- **NAX6-N's MGSM rows do not exist yet.** They come first, as ref2tnaxn.r5 (80 rows), on the same decode stage and
  environment as CERT's references ref2tnaxn.r2 / .r3.
  - Cost: NAX6-N ran mathx at 14.2 tok/s. REF's MGSM rows average 427 tokens, so ~45 min.
  - REF's MGSM rows (results/ref, 91.2 % in RESULT_fastmode_b16) are information.
- **The candidate's QOM must run on MGSM.** QOM is switched per eval set, and mgsm is in none of the env files so far.
  - BigMoE writes the env: M1-B + COH (chat_env 97e81b8) with R57_QOM_SETS adding mgsm, i.e. QOM on every chat set.
  - Engagement: every MGSM row must show QOM optional uses, else VOID.
  - The other checks are runner 59's: code pin, manifests, memory and swap gates.
  - Cost: 80 rows x 2 orders, ~45 min.
- **Information:** MGSM per language, rows token-identical to NAX6-N, order agreement, cap hits, MGSM vs REF.
- **Anchoring NAX6-N to the model card: not possible on MGSM.**
  - The 125B's card (models/bare/README-upstream-qwen.md) lists no natural-language multilingual benchmark. Its only
    "multilingual" entry is SWE-bench Multilingual, i.e. programming languages.
  - Its general benchmarks are IFBench 81.3, GPQA Diamond 91.7, HLE 35.9 and LiveCodeBench v6 91.9, in thinking mode
    with sampling.
  - EvalSuite holds GPQA Diamond's 198 items (data/items/gpqa.jsonl). An anchor there needs thinking-mode decoding.
    That is a separate decision.
- **The runner:** tools/day_runner60.sh (runner 59's checks; stage 1 the reference rows, stage 2 the candidate) ->
  results/mgsm_<tag>_{fwd,rev}, results/MGSM_<tag>_DONE. It starts when NewIdeas queues it.

## Amendment 3 (Sun 10:0x, before any MGSM row): the arms, and what MGSM can and cannot say
- **The arms (NewIdeas 09:2x; the orchestrator asked for L).** Each arm has its own verdict on amendment 2's line.
  - **L + COH (tag lcoh):** the certified config as it would ship. That is cert_env's levers + REF2_NAX_COH=1, on a
    snapshot whose frozen ref2 has COH; cert_env's s5 snapshot has none. BigMoE writes data/stack/lcoh_env.txt.
  - **M1-B (tag m1bg):** chat_env's config with mgsm in R57_QOM_SETS (BigMoE's chat4_env.txt). It waits for the
    orchestrator's decision below.
- **The screens.** chs4's and s5's SETOF has no mgsm, so R57_KEYS=mgsm/... raises a KeyError.
  - Each arm's env must name a snapshot whose screen maps "mgsm": "mgsm".
  - Runner 60 checks this and stops before any GPU step otherwise.
- **What MGSM tests in the engine.** The screen's own rule() types all 80 MGSM prompts "math": they carry an English
  instruction and \boxed. So they draft on DFlash, not MTP, and M1-B's MTP + QOM gating (its chat lever) does not act on
  them.
  - For L, MGSM tests the release config's multilingual math path: a valid release check.
  - For M1-B, an MGSM PASS would not speak to its multilingual chat lever. The orchestrator decides between:
    - (a) MGSM re-templated so the engine types it chat (native-language instruction, no \boxed), with the typing checked
      before any row;
    - (b) MMMLU, which needs a download;
    - (c) MGSM as registered.
- **Order:** stage 0 (NAX6-N's rows), then L + COH, then M1-B's arm or its replacement.
- **Runner 60** = tools/day_runner60.sh, one arm per invocation (MGSM_TAG, MGSM_FENV).
- **HOLD (the orchestrator, 09:3x):** both arms wait for Kamen's decision on how M1-B's multilingual chat is judged.
- **An accidental start, 09:43:18 to 09:45:41, during the hold.**
  - What happened: a dry run of runner 60 took the real path. An apostrophe in the ${MGSM_FENV:?...} message opened a
    quote that swallowed lines 23-~55, so DRY and several helpers were never set. The undefined `say` fell through to
    macOS's speech command.
  - Effect: it decoded 2 NAX6-N MGSM rows under a shared lock for 2.4 min and delayed BigMoE's PFB ticket. No arm row
    ran.
  - I stopped it by pid (4 processes) and ran peer unlock. Its rows and log are moved aside (results/ref2tnaxn.r5.void_0943,
    results/day60_lcoh_runner.void_0943.log).
  - Fix: no apostrophe inside ${} text. Re-tested: the head gives DRY=1 with the helpers defined, and the dry run stops
    at the SETOF check as it should (CHS4's screen).
