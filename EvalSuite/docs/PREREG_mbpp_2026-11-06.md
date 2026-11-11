# PREREG: CD16 on MBPP, decision 4 vs the exact reference (Fri 2026-11-06 10:53; before any row)

Written before any MBPP row.
- **Request:** the orchestrator, Fri 09:5x, on Fable GOAL's K-WALL (~/FableGod/goal/K_WALL_1106.md section 2 item 1).
  - CD16 (R57_INERT=CD16) keeps the drafter's comment and docstring text wholesale. On MBPP that is 57 % of answer tokens
    (K 5.6 -> 10.3-10.9 [P]).
  - But MBPP docstrings carry the 125B's own plan: 6 of 100 answers think inside one until the cap. There, the drafter's
    text replaces the model's reasoning, which HumanEval+ / math / IFEval rows cannot see.
  - CD16 does not ship beyond HumanEval-type answers without this test.
- **The orchestrator accepted the plan and the line (Fri 09:5x):** tonight, after the stack same-scores run and the drift
  top-up. F1 moves to Sunday night.

## Rows
- **dec_mbpp:** the 100 tasks of MTP Head's exact walk (task ids 2-321, 99 of them in MBPP's test split), from
  ~/TrainLab/data/mbpp/mbpp.jsonl (local; Kamen's 11-02 download OK).
  - MTP Head's template: the task text + its 3 asserts, which fix the function name.
  - max_tokens 1024. tools/mbpp_build.py; sha256 c700f2ac9a25.
- **pass@1** (tools/mbpp_score.py): the answer's longest ```python block, then the task's test_setup_code and its 3
  asserts, run in an isolated child process (python -I in a temp dir, evalplus's reliability_guard, a 10 s timeout).
  Pass = exit 0.
  - Self-test (Fri 10:53): the dataset's own solutions pass 100/100.

## Arms
- **The exact reference:** NAX6-N's AR (`ref2tnaxn.r2`: the decode stage on the converted store, runner 43's env), the 100
  rows.
- **The arm:** the stack with CD16. BigMoE's data/stack/stack_env.txt as written tonight (it carries R57_INERT=CD16)
  goes through tonight's frozen sat screen, with MBPP in QOM's FAST sets and the screen's set map, per BigMoE.
  - Two prompt orders (forward = the task ids sorted, reverse), chunks of 10 prompts per process. QOM depends on cache
    state (Kamen's >= 2 histories).
  - The C is that of BigMoE's run.
  - The pins and the manifest checks are runner 49's.

## The line
- **Decision 4** per order: MBPP pass@1, the arm vs the exact reference. The paired delta must be above -5 pt AND the
  exact McNemar p at least 0.05.
- **PASS:** both orders pass. **FAIL:** either fails. **DIFFERS (HIGHER):** a gain with p < 0.05, counted as PASS.
- **VOID:** the stack did not engage (its counters), or a loaded module is not the reference's.
- **What it means:** PASS -> CD16 may ship for general code answers (MBPP-type). FAIL -> CD16 stays on HumanEval-type
  answers.

## Information (no lines)
- Cap hits (rows that reach 1024 tokens) per arm, against the reference's, and those rows' pass / fail.
- Token-identical rows and the first divergence vs the reference.
- Agreement between the two orders.
- CD16's counters (inert rows) per run.

## Runs
- Tonight in runner 49, after the stack same-scores run and the drift top-up:
  - the reference ~40 min;
  - the arm ~30 min (two orders).
- Report: tools/mbpp_score.py d4 per order -> results/mbpp_{fwd,rev}_d4.txt, then a RESULT doc.

## Amendment 1 (Fri 10:59; before any row): the arm's env, per BigMoE (Fri 11:0x)
- **The set map:** 'mbpp' is in both screens' SETOF (committed before tonight's snapshot) and loads dec_mbpp.
- **QOM's FAST sets:** stack_env.txt's own R57_QOM_SETS (humaneval,mbpp,math,ifeval,multi) is used as written. Only if
  it lacks mbpp does the runner add R57_QOM_SETS=humaneval,ifeval,multi,mbpp, and it logs that.
- **CD16** keys on the code type and the answer text: INERT runs on every DFlash2-drafted prompt and acts only on
  comment / docstring spans of python fences. Fable GOAL checked that the type rule returns 'code' for all 100.
- **Caps:** the items' max_tokens stays 1024, as registered. GOAL's 384 is for its 12-prompt speed arm. The screen runs
  to CAP 1536, and each row is cut at 1024.

## Amendment 2 (Fri 15:36; before any row): the reference's afternoon slot (NewIdeas)
- Kamen extended the big-test window to 17:00. The exact reference (ref2tnaxn.r2 on dec_mbpp) decodes 16:15-16:55
  (runner 52): one chunk sized to end by 16:55, and the cli's deadline is 16:55.
- Rows left over resume tonight in runner 49's MBPP phase, which decodes only missing rows. The stack + CD16 arms stay
  tonight, after the top-ups. The line is unchanged.

## Amendment 3 (Fri 16:47; before any arm row): CD16 dropped, so the arm is a coverage check
- **NewIdeas, Fri 16:0x:** CD16 is KILLed on quality by GOAL's MBPP12 (BigMoE: 11/12 -> 8/12 per pass pair, from the
  drafter's comment loops). It is out of tonight's stack (BigMoE f27b132).
- **The arm:** tonight's stack_env.txt as written, now WITHOUT CD16. It runs on the 100, in two orders, last tonight, and
  yields if the night runs short (the END bound). It is a coverage check of the stack on MBPP, decision 4 per order as
  registered.
- **The reference is done** (runner 52, 16:15-16:46): ref2tnaxn.r2, 100/100 rows, pass@1 79.0 %, 6 cap hits.

