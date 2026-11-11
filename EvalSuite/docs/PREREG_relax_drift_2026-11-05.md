# PREREG: RELAX ship mode, the long-row drift re-check for B on code (Thu 2026-11-05 08:29; before any row)

Written before any re-check row.
> (a quote from the project owner, kept private)
  long-row drift re-check passes". Prepared now; the GPU runs only after Kamen's yes.
- **Why:** ship (B on code + E on chat) passed decision 4. B failed mission 7's drift bar on HumanEval+ at 6.04, but on
  only 4 rows of >= 512 tokens (KLD q1 0.00157, q4 0.00948). Math, with 23 long rows, passed at 0.89.
  - The question: does B's numerical drift against REF2-T really grow along long code answers, or did 4 rows mislead?

## What runs
- **Arm:** RELAX's B, exactly as the night ran it: Q1-UNION cf 0.9 + the copy gate + the comment rule (R2 in comments)
  + HC8, on the 16-row live loop, C 160 (or 192 when memory allows; tokens do not depend on C).
  - The code is the night's (snap/leap-faff8e5, snap/bigmoe-6c64148, relax.py). Only the set list grows, so B covers the
    two new sets with its HumanEval+ / agent settings.
- **Rows** (local, no download; tools/drift_build.py; keys results/drift/keys_B.txt, sha256 a568c71632b6):
  - `humanevalx`: HumanEval+'s 104 problems outside the core 60, dec_humaneval's template, max_tokens 768
    (data/items/dec_humanevalx.jsonl, 75e4b50ae57a).
  - `agentlong`: dec_agent's 17 write / edit turns (real code written into files, 1.6-12K-token contexts), max_tokens
    1024 instead of 128 (data/items/dec_agentlong.jsonl, ea6211b4933c).
- **The reference:** REF2-T's replay of B's logged checks along B's own prefix, as the night's bars used (replay.py).
  The bar compares REF2-T's and B's distributions token by token, so no REF2-T decode is needed.

## The line (mission 7's drift bar, as registered)
- **Rows:** every code row of >= 512 tokens, pooled:
  - RELAX's 4 long HumanEval+ rows;
  - every long `humanevalx` row;
  - every long `agentlong` row.
- **PASS:** the mean KLD(E||G) over the last quarter of positions is at most 1.5x the first quarter's, pooled over those
  rows. The first-quarter mean must be > 0.
- **FAIL:** the ratio is above 1.5.
- **Expected count:** ~21-23 long rows (B's long-row rate of 4/60 on HumanEval+ gives ~7; most of the 17 long agent
  turns should pass 512).
  - If fewer than 20 come out, the verdict is reported with that count beside it, and the orchestrator decides whether
    it is enough.
- **Information:**
  - the ratio per source (HumanEval+ old / new, agent);
  - per-row q1 / q4 KLD;
  - blunders (E's p of the committed token < 0.01) on the new rows, and on agent rows separately for the command text
    vs the description text (the orchestrator's bash22 question);
  - deviated tokens by cause.

## What a result means
- **PASS:** the 4-row drift was not borne out on >= 20 long code rows. Together with the night's other bars
  (math + code blunders 0.21 per 10K), ship's only failing bar would be the one agent-row blunder. Kamen decides.
- **FAIL:** B's drift grows along long code answers. Ship stays BETWEEN, and the drift is the reason.
- Decision 4's verdict on the core 210 does not change either way.

## Runs and cost
- **GPU:** B on the 121 keys (~16 min for humanevalx, ~35 min for agentlong, mostly their 8K prefills), then REF2-T's
  replay of those checks (~50 min). About 1.7 h on a shared lock, MEM ~30-34 GB, no timing line.
- **Slot:** Friday by day (the orchestrator, Thu morning), after the night batch, ending before 15:00 by END-bounded cutoffs.
  Only after Kamen's yes.
- **Code:** relax.py's set list gains humanevalx / agentlong, and B gains them with its settings. That is evalsuite/,
  which T0's runner pins until Fri 00:15, so the change waits in pending/ and is applied after T0 stops.
- **The report:** tools/relax_report.py gains a pooled long-code drift line, which is not pinned.

## Amendment 1 (Thu 2026-11-05 09:03; before any re-check row): the converted store, the bridge, E8
> (a quote from the project owner, kept private)
  the drift re-check first, then E8, then T0's leftover if time remains. Both checks run after Thursday's records3
  conversion (on NAX6 Part 1's PASS), so their REF2-T replays must read the store layout that is valid then. The night's
  code (snap/bigmoe-6c64148 on build_next, snap/leap-faff8e5) reads only the row-major store.
- **The code by store** (tools/day_runner45.sh picks it once, at GO, and stops if the store changes after that):
  - **Converted** (all 48 layers): BigMoE snap/bigmoe-da8e7a6 on ext/build (MAXK 160 is set at runtime there). REF2-T
    runs as `ref2tf0`: REF2-T on snap/leap-52cf2c9 with every switch off. Converted layers take Leap's 3c-N, the native
    twin of 3c, even with REF2 removed.
  - **Row-major** (no conversion happened): the night's code, as registered above.
  - The arm itself does not change: B's settings, the drafter, the 16-row loop and relax.py's logic. The code change
    is in pending/apply_drift_e8.py: the two sets, arm E8, and the REF2-T variant read from EVALSUITE_RELAX_VARIANT
    (default ref2t). The gate-ext rule takes build_next, or ext/build on a snapshot after Nov 4.
- **The bridge to the night's numbers** (converted mode). Each check is a stop rule:
  1. First, B's replay process runs on the night's rows. All of them are replayed already, so it runs only its step 0:
     REF2-T on the new code must replay the night control's gate-off rows bitwise (a hash of every logits row;
     replay.py's existing check). If not, the run is VOID: no Friday row runs, and it is reported as VOID.
  2. In every G process, step 0 runs HumanEval/27 with the gate and the rule off. It must equal results/ref2th8 (the
     HC8 arms' reference) token for token. If not, the run is VOID.
  3. Thursday's records3 check 2 (runner 43, six IFEval rows) is the token-level evidence that 3c-N == 3c. Here it is
     information only.
  - If the bridge passes, the night's 4 long HumanEval+ rows pool with Friday's as registered. If it fails, nothing is
    pooled, and the orchestrator decides what runs instead.
- **E8** (confirmed by the orchestrator on Thu 08:5x). NewIdeas: HC8 is process-global as built, so the shipped build
  runs HC8 on chat too. The night's E did not.
  - **Arm E8** = E + HC8: R1 on ifeval / multi, hc8=True. It runs on E's 124 keys (results/relax/keys_E8.txt is a copy
    of keys_E.txt), and REF2-T then replays it. E needed no replay, because it has REF2-T's numerics; E8 has HC8, so it
    needs one.
  - **Mode ship8** = B on math + HumanEval+, and E8 on IFEval. It gets the same verdict as `ship`
    (tools/relax_report.py --score):
    - decision 4 on the 210 core rows vs REF2-T (results/ref2t);
    - mission 7's bars;
    - the per-task line.
    - **KILL:** decision 4 fails, or the per-task line flags it.
    - **PASS:** decision 4 passes and the bars pass.
    - **BETWEEN:** anything else.
  - **What it means:** ship8 is the mode that ships, since HC8 is global as built. If its verdict differs from ship's,
    ship8's is the one that counts for the shipped build. E8's IFEval drift is information, as ship's is.
- **Order and time:**
  1. the bridge, ~5 min;
  2. B on the 121 keys plus their replay, ~1.7 h;
  3. E8 on 124 chat rows plus their replay, ~1.6 h (the night's E took 42 min; B's replay of 124 rows took 51 min).
  - The run is Friday by day, after the night batch (results/q45/GO), as the day's one > 5 GB job.
  - Chunks are END-bounded by 15:00, so the whole run fits if GO comes by about 11:00.
  - If E8 does not finish by 15:00, its rows resume in the next slot the orchestrator gives (the rows are resumable).
    ship8's verdict waits for all of them.

## Amendment 2 (Fri 2026-11-06 08:09; before any top-up row): a code-only confirmation
- **Why:**
  - The orchestrator (Fri 08:0x) accepted 16 long rows for the registered pooled line, so the drift PASS stands.
  - The pooled line cannot see code drift, because agentlong's ~200x larger KLD dominates it. The 10 HumanEval-type long
    rows alone grow ×4-6 from a tiny base (docs/RESULT_relax_drift_2026-11-06.md).
  - So: a code-only confirmation on >= 20 long HumanEval-type rows. If it fails, the RELAX code part rolls back before
    anything is published. The orchestrator accepted the line below (Fri 08:3x).
- **Rows:** HumanEval+ is used up (all 164 problems; 10 gave >= 512 tokens). Local, no download: 41 HumanEval+ problems
  (task ids 0, 4, 8, ... 160) with a longer answer:
  - the prompt asks for the function plus a pytest module of at least 15 tests;
  - max_tokens 1536;
  - set `humanevalt`, built by tools/drift_build.py in BigMoE's venv (the chat template), keys
    results/drift/keys_T.txt.
- **Arm:** RELAX's B exactly as Friday's run: the converted store, `ref2tf0`, snap/bigmoe-da8e7a6. B gains the set
  `humanevalt` with its HumanEval+ settings, including the comment rule. Then REF2-T's replay of B's checks.
- **The line,** on every code row of >= 512 tokens (Thursday's 10 + the new long rows):
  - **PASS:**
    - blunders (E's p of the committed token < 0.01) <= 1 per 10K tokens on those rows;
    - AND the pooled q4 KLD (the mean over the last quarter of positions) <= 0.02;
    - AND no row's q4 KLD > 0.05.
  - **FAIL:** any of the three fails.
  - 0.02 is about 2x Thursday's code q4 mean (0.0095). Math's q4 is 0.0066.
  - If fewer than 20 long code rows come out, the verdict is reported with that count beside it, and the orchestrator
    decides whether it is enough.
- **Information:**
  - the code-only q4 / q1 ratio (not in the line: with q1 near zero it fails on noise);
  - per-row q1 / q4;
  - deviations by cause;
  - blunders by position (in code vs in comments).
- **Slot:** tonight (Sat 00:30 to ~10:00), after the agent check, per the orchestrator.

## Amendment 3 (Fri 12:58; before any smoke row): the long-code smoke on tonight's CODE env (NewIdeas, QUEUE b6e90b5)
- **Why:** the agent bisect showed QOM-KEEP carries long agent outputs' blunders. Long code is the open question for the
  code headline, which tonight's stack (QOM on the code sets) carries.
- **The smoke:**
  - tonight's CODE env, built from BigMoE's sat_night.sh: QA + STACK, i.e. the QOM FAST all-in + NAX6-N + DN-KS + GDN
    lazy + D7BNAX_32 + CD16 + FAST-tax.
  - R57_QOM_SETS=humaneval,mbpp,humanevalt. humanevalt is a code set; BigMoE adds it to the screen's SETOF.
  - It runs on 6 humanevalt rows (HumanEval/0, 4, 8, 12, 16, 20), one order, one process on the live tree; the process's
    module manifest is the provenance.
  - It is scored by tfwin against NAX6-N (`ref2tnaxn`).
- **The line:** more than 1 blunder per 10K tokens means tonight's top-up ALSO runs the QOM-off CODE arm (the same env
  with R57_QOM_SETS without humanevalt). At or below, no extra arm. The smoke changes no registered verdict.
- **Runs:** runner 51, today after Breakthrough's never-wait map arm, ending by 15:00 (~20-30 min).

## Amendment 4 (Fri 13:32; before any row): tonight's top-up arms (NewIdeas Fri 13:3x)
- **Order:** the stack same-scores run, then the registered B top-up (relax.py; KLD + blunders; the rollback gate), then
  the stack's CODE env on all 41 humanevalt rows, then MBPP / CD16.
- **The stack on humanevalt:**
  - code_env.txt's env with R57_QOM_SETS=humaneval,mbpp,humanevalt, one order, scored by tfwin against NAX6-N.
  - Reported with amendment 2's blunder bar (<= 1 per 10K) as its line. It has no KLD, because the screen logs no
    distributions.
  - It is not RELAX B's rollback gate.
- **The QOM-off CODE arm:** the same env with R57_QOM_SETS=humaneval,mbpp, run only if the afternoon smoke said
  > 1/10K (amendment 3).
- **Report:** results/stack_ht_report.txt.
- **Status Fri 15:17: the smoke did not run.**
  - It waited from 13:51 for 50 GB NO-SWAP, which peaked at 47. NewIdeas stopped it after the big-test window closed at
    15:00. No row exists.
  - Tonight's two top-ups supersede it (amendment 4). The QOM-off CODE arm stays pre-registered as their fail arm,
    for Saturday by day.

