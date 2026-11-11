# PREREG: the FAST stack's same-scores run, its adoption gate (Fri 2026-11-06 08:21; before any row)

Written before any row of this run.
- **Request:** NewIdeas' GPU ORDER (Fri 08:1x, NewIdeas 3d82ace) and the orchestrator (Fri 08:3x). It is ONE same-scores
  run (decision 4 vs REF2-T) on exactly the stack that wins BigMoE's stacked palindrome (the Fri -> Sat night). That run
  is the stack's adoption gate.
- **RELAX as routed (NewIdeas):**
  - B ships for code and math answers;
  - agent turns stay EXACT until the agent check passes (PREREG_agentfast);
  - chat uses the EXACT launch.

## What runs
- **The stack:** BigMoE's env for the palindrome's winner, through scripts/r57_screen_fast.py. BigMoE writes it for this
  run. The env, the commit, the ext build and the C are added here under "Recipe" before any row. The runner pins
  BigMoE's tree as runner 47 does: tracked .py files and bigmoe/ equal to the commit, the .so sha, and a manifest of
  every module loaded.
- **Profiles:**
  - **math (50) and HumanEval+ (60), the core keys:** the CODE profile = the stack + B with HC8.
  - **IFEval (100):** the stack + E without HC8, only if BigMoE's HC8-off launch exists by then. Otherwise chat stays
    EXACT and its rows are NAX6's own (results/ref2tnax), which passed decision 4 (RESULT_nax6).
- **Two prompt orders,** forward (the keys sorted) and reverse, one process per profile and order:
  - Kamen's decision 2: a FAST mode that depends on the cache state certifies under >= 2 histories
    (docs/PREREG_TEMPLATE_cachedep.md).
  - QOM-KEEP keeps by cache state.
- **Rows:** tools/r57rows.py turns the tokens into decode rows, results/stack_fwd/ and results/stack_rev/. Each row is
  cut after its first EOS or at the item's max_tokens. CAP 1536 is the largest max_tokens.

## The line
- **Decision 4 vs REF2-T** (results/ref2t), per set, per order (tools/d4x.py; IFEval's stage 2 on IFX's 220 as usual,
  only if IFEval runs and fails alone).
- **PASS:** both orders pass every set.
- **FAIL:** either order fails a set. NewIdeas bisects only on a fail.
- **DIFFERS (HIGHER):** a gain with p < 0.05 and nothing fails, counted as PASS for adoption.
- **VOID:**
  - the stack did not engage (BigMoE's counters for its parts);
  - or a loaded BigMoE module differs from the commit.

## Information (no lines)
- Token agreement between the two orders (the cache dependence).
- The suite's standard table against REF2-T.
- Mission 7's blunder bar, where an exact replay exists (none planned: no logged distributions).
- Rough tok/s (not an A/B).

## Runs and cost
- **Time:** per order, ~110 code + math rows at FAST speed with their prefills, ~35-45 min. With IFEval, +~30 min per
  order. Two orders: ~1.5-2.5 h.
- **Slot:** the Fri -> Sat night after BigMoE's palindrome (NewIdeas). Then the code drift top-up (PREREG_relax_drift
  amendment 2), then F1's rest if it fits.

## Amendment 1 (Fri 09:03; before any row): the profiles as BigMoE builds them
- **BigMoE (Fri 09:0x):** tonight's stacks are QOM-based, not RELAX's B / E.
- **What BigMoE's scripts/sat_night.sh writes tonight:** each file is '#' comment lines plus one comma-separated K=V env
  line; the runner reads the last non-comment line.
  - data/stack/stack_env.txt: the FAST all-in (R57_FAST=QOM, R57_HC8=1 ...) + NAX6-N (R57_NAX=1 REF2_NAX_1CQ=1) +
    REF2_DN_KS + REF2_GDN_LAZY + D7BNAX_32 + R57_INERT=CD16 + R57_FASTTAX.
  - data/stack/stackr_env.txt: the same with MoonShots' relax v2 (R57_RELAX_DELTA=0.5, R57_RELAX_SPAN=1) in place of
    CD16.
  - data/stack/code_env.txt: **the code profile this run uses**, written after BigMoE's report. It is stackr if
    stackr's pre-registered line passes (>= the stack +3 % on code), else the stack. Its comment line says which.
- **Profiles here:**
  - **math (50) and HumanEval+ (60):** code_env.txt's env, as written.
  - **IFEval (100):** EXACT, because the stack without HC8 is not a tested launch (the orchestrator). Its rows are
    NAX6's own (results/ref2tnax), copied into each order's run.
  - **Agent turns:** exact, and outside decision 4.
- **The pin:** BigMoE's HEAD when code_env.txt is read, with tracked .py files and bigmoe/ equal to it and nothing
  untracked. The ext build named in the env, with its .so sha recorded at the start. The per-process module manifest
  is checked against that commit, as in the agent check.
- **Chunks:** 10 prompts per process (the screen's HARD = 600 s per process; since 9c82d1d the tokens are written after
  every run). A prompt that a process misses runs in the next process.
- **The two orders and the line are unchanged:** decision 4 vs REF2-T per set, per order, and PASS = both orders pass.

## Amendment 2 (Fri 10:59; before any row): QOM's sets and the env line
- **BigMoE (Fri 11:0x):** tonight's env files (stack_env / stackr_env / code_env) carry
  R57_QOM_SETS=humaneval,mbpp,math,ifeval,multi.
  - So the code profile's math rows run QOM's FAST rule too. That is new, and this run is its first check. The math
    line is decision 4's like the others.
  - If the orchestrator wants math exact, math leaves the list before the run, and this amendment says so.
- **The env line** is comma-separated. A token without '=' continues the previous value (R57_QOM_SETS's list); any
  other shape stops the run.

## Amendment 3 (Fri 11:28; before any row): NewIdeas' gate (QUEUE 70a4951, after the agent check's FAIL)
- **QOM on the code sets only:** R57_QOM_SETS=humaneval,mbpp. BigMoE was asked to write it into tonight's env files.
  - Math rows stay as certified, without QOM: they run the code profile's env with math outside QOM's FAST sets.
  - The env line as BigMoE writes it decides. The runner logs it, and amendment 2's math-on-QOM note applies only if
    the file still lists math.
- **Chat:** a separate launch (the stack minus HC8, plus E's R1 0.5, no QOM) only if BigMoE has it ready. Otherwise chat
  stays EXACT: NAX6's own IFEval rows, as amendment 1 has it. The runner has no chat launch, so chat stays EXACT tonight.
- **Verdicts:** per set, as registered. The code part's go / no-go reads HumanEval+ here, plus the drift top-up
  (PREREG_relax_drift amendment 2) and MBPP for CD16 (PREREG_mbpp).

