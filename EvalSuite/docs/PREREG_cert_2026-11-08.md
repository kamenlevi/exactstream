# PREREG: CERT, the release certificate for the frozen 45 GB FAST config (Fri 2026-11-06 21:25; before any row)

Written before any CERT row.
- **Request:** NewIdeas (QUEUE 07f18e5; the orchestrator approved). It certifies the config that ships.

## The config
- **The FROZEN config:** BigMoE's S5 config as of Saturday ~20:00 (data/stack/final_env.txt, or whatever BigMoE names the
  S5 env; the file and its commit are recorded here before any row). Nothing that lands after 20:00 goes in.
- **The code:** frozen at CERT's start with tools/freeze_r57.py (every outside tree copied, every .py path rewritten).
  - The first process's manifest is the reference.
  - Every process must load its outside modules from the freeze (else VOID).
- **Chat and agent:** EXACT (no FAST chat profile ships). IFEval joins only if the orchestrator says a chat FAST profile
  ships; that would be its own amendment before rows.

## Sets (each in two prompt orders: forward = the keys sorted, reverse)
| set | n | items | the exact reference |
|---|---|---|---|
| HumanEval+ (pass@1, evalplus base + plus) | 164 | dec_humaneval 60 + dec_humanevalx 104 | NAX6-N's AR: results/ref2tnax (60) + a new decode (104) |
| math (boxed / numeric) | 100 | dec_math 50 + dec_mathx 50 (sha256 4fffdcb2d283, tools/cert_build.py) | results/ref2tnax (50) + a new decode (50) |
| MBPP (pass@1, tools/mbpp_score.py) | 100 | dec_mbpp (MTP Head's walk) | results/ref2tnaxn.r2 (done Fri 16:46) |

- **The new reference decodes** (NAX6-N's exact AR on the 154 rows it lacks) run before CERT: Saturday by day if a slot
  allows, else at the start of CERT's night.
- **The reference is NAX6-N**, the certified 45 GB exact reference (RESULT_nax6) and the stack's own base. REF2-T is
  reported as information on the sets where its rows exist (the core 60 + 50).

## The lines (pre-registered)
1. **Per set and order (6 lines):** decision 4 vs NAX6-N's rows. The paired delta must be above -5 pt AND the exact
   McNemar p at least 0.05.
2. **Pooled, the larger n's tighter line:** over all three sets and both orders (728 pairs), the pooled delta must be
   above -2 pt AND the pooled McNemar p at least 0.05.
3. **Long code:** the 41 humanevalt rows, one order, scored by tfwin against the exact base. Blunders (the base's p of a
   committed token < 0.01) must be at most 1 per 10K tokens.
- **CERT PASS = all 8 lines pass.** A FAIL names its lines.
- **VOID:** the freeze check, or the config's engagement counters, fails.

## Reports (fixed paths, quoted by the paper)
- results/cert/CERT_humaneval.txt, CERT_math.txt, CERT_mbpp.txt (each: both orders' decision 4 lines);
- results/cert/CERT_pooled.txt, CERT_longcode.txt;
- results/cert/CERT_VERDICT.txt (the 8 lines and the verdict, one line each).

## Information (no lines)
- Token agreement between the orders.
- The standard table vs NAX6-N and vs REF2-T.
- Cap hits.
- Rough tok/s (not an A/B).

## Runs and cost
- **The reference decodes:** ~1.5 h (exact AR; independent of the config).
- **CERT itself, about 3.5 h:**
  - the config on 364 rows x 2 orders, in chunks of 10 per process (the screen's HARD 600 s): ~2.3 h;
  - the long-code rows + tfwin: ~1.2 h.
- **Slot:** Saturday night in the Sun ~01:30-10:00 window (NewIdeas), so the verdicts land before Sunday's paper fill.
  The fallback is Monday night, only if the config is not frozen by Saturday evening.

## Amendment 1 (Fri 21:25; before any row): QOM's sets for CERT's rows
- QOM's FAST rule keys on the screen's set list (R57_QOM_SETS), not on the type.
- So CERT runs the frozen env with R57_QOM_SETS extended by humanevalx (and humanevalt for the long-code line). Every
  HumanEval-type row then gets the shipped code rule.
- math and mathx stay as the env has them (out of QOM: the code-only gate).
- BigMoE adds 'humanevalx' and 'mathx' to the screens' SETOF before the Saturday freeze.


## Amendment 2 (Sat 2026-11-07 05:5x, before any CERT row): the runner, the chat check, FAST sets
- **The runner:** tools/day_runner55.sh. It uses runner 54's (S4's) gates, code checks and manifests unchanged.
- **The config:** BigMoE writes the FROZEN config at the ~20:00 freeze to data/stack/cert_env.txt, in final_env.txt's
  format: 'Sat <C>', 'MEM <n> GB' and 'screen data/r57/snap/<dir>/<file>.py' in its comment. CERT starts once that
  file has been unchanged for 2 min and runners 53 / 54 have ended. The file, its commit and BigMoE's HEAD go into the
  RESULT.
- **The code:**
  - If the file names a committed snapshot, CERT runs it. Every BigMoE module a process loads must then equal BigMoE's
    HEAD at CERT's start, and the snapshot's paths may hold nothing uncommitted.
  - Without a snapshot, tools/freeze_r57.py `now` freezes the live trees (as PREREG_s4 amendment 1).
  - The screen's SETOF must cover CERT's sets, checked before any row.
  - Live outside files (S4 found 17 Leap / TrainLab files loaded live) are pinned by sha at every gate. BigMoE was asked
    to freeze them into the CERT snapshot.
- **The reference rows:** the decode stage tops up ref2tnaxn.r3 (humanevalx + mathx, 154) first if runner 53 did not
  finish.
- **The chat check** (the orchestrator, Sat 05:5x: "chat EXACT in v1 is assumed, not measured"; Breakthrough flagged
  the same):
  - The rows: 12 IFEval prompts (the first 12 keys of dec_ifeval, sorted) interleaved with 12 humanevalx prompts, in one
    order, 8 prompts per process. The engine therefore swaps profiles at request boundaries, code <-> chat.
  - **CHAT-EXACT: YES** only if all 12 IFEval rows are token-identical to NAX6's AR rows (results/ref2tnax); otherwise
    NO, with the first divergences.
  - It is a check of its own (results/cert/CERT_chat.txt), outside CERT's 8 lines. The paper may say "chat runs EXACT
    in v1" only on YES.
- **FAST sets (information, for the paper):** CERT_info.txt lists, per set and order, how many rows are token-identical
  to NAX6-N.
  - S4 already showed math is NOT exact in the stack: 6 of 50 math rows identical in v1, 5-8 of 50 in stackb.
  - The math processes log relax-span engagement ("span 10 runs"), while QOM is on HumanEval / MBPP only.
  - The paper must say which sets run FAST kernels.

## Amendment 3 (Sat 2026-11-07 23:3x, before any CERT row): IFEval joins as a scored set
- **Why:** the PREREG said "IFEval joins only if the orchestrator says a chat FAST profile ships". The orchestrator
  (23:3x) says that in effect it ships.
  - BigMoE's N45 (23:26, data/n45; the orchestrator verified the raw runs): the shipped config (cert_env's v1' + wired)
    ran 12 IFEval prompts at exact acceptance, yet only 6 of 24 runs equal NAX6-N's AR (math 7 / 24).
  - Some non-QOM lever changes the chat and math bits (R57_FASTTAX, REF2_HC8_FAST, or S2's SVH / DSUB; untested which).
  - So chat in v1' is NOT exact, and no line covered its quality.
- **The new lines:** IFEval, the 100 core prompts (dec_ifeval), in both orders inside CERT's main run (CERT's rows
  become 464 per order). Decision 4 vs NAX6's IFEval rows (results/ref2tnax), per order, same bar as the other sets:
  delta > -5 pt AND McNemar p >= 0.05. No stage 2. That makes 2 more lines: CERT PASS = all 10 lines pass.
- **The pooled line keeps its pre-registered 3 sets (728 pairs).** A 4-set pool with IFEval is reported as information.
- **Unchanged:** the chat-exact check stays as written; it will most likely say NO. Then the paper says "chat: lossy,
  same scores".
- **Cost:** ~30-40 min more (100 IFEval rows x 2 orders at ~45 tok/s). CERT starts at 01:30 as planned.
