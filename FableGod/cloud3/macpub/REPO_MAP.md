# REPO_MAP: what to read in `~/350-tps` for this mission, in order (orchestrator, Thu 5 Nov ~00:15)

## A. Directly about your wall (read fully)
- `FableGod/FLOOR_125B.md`: mission 13, how much of the 125B each token really needs (the floor of experts per token).
- `FableGod/BREAK_THE_WALL.md`: mission 14, the engine from physics up, including the gate's union cut (pairs x0.48, bytes x0.58).
- `FableGod/FINDINGS.md`: summaries of missions 12-33, floors and verdicts.
- `FableGod/CLOUD_ANSWER_FASTMODE.md`, `CLOUD_ANSWER_BYTES33.md`, `CLOUD_ANSWER_SSD45.md`, `CLOUD_ANSWER_CLEANSHEET.md`, `CLOUD_ANSWER_NEWIDEAS.md`, `CLOUD_ANSWER_ROAD350.md`: earlier cloud Fable analyses. Their task files are `CLOUD_TASK_*.md`.
- `FableGod/ASSUMPTION_HUNT.md`: mission 31, fifteen assumptions measured.
- `FableGod/CACHE.md`: mission 20, the prompt-aware cache.
- `FableGod/BLUEPRINT.md`: an MoE designed for one user on unified memory.
- `FableGod/far/FAR.md`: foresight, the rejected-row share of reuse, the SSD wait law.
- `FableGod/goal/GOAL.md`, `FableGod/full/FULL.md`, `FableGod/zero/ZERO.md`: last night's three context-level answers.
- `FableGod/fablec/BYTES10.md` and `NewIdeas/research/SWEEP.md` + `TEN_X_LEDGER.md`: the 24 GB bytes ledger.

## B. Ground truth (grep, don't read whole)
- `FableGod/TRIED/TRIED.md`: every idea tried, with its verdict. `FableGod/TRIED/<session>.md`: each session's part. `FableGod/TRIED/NOVELTY.md`.
- `FableGod/KERNEL90.md`, `FableGod/EXPERT_KERNEL.md`: the kernel.
- `Leap/TABLE_350.md`: every kernel / dense timing.
- `FableGod/BUILD_DAY_1104.md`: what was built on Wed 4 Nov.
- `FableGod/findings/`: dated public notes, including the 12 GB exact run, the chat route, trellis, and near / far forecasting.

## C. How the data is made (read the code before using the data)
- `FableGod/decision_map/*.py`: the decision map. `dm_common.py` defines the capture formats; `dm_ablations.py` the arms.
- `BigMoE/scripts/` (r57 screen), `Leap/scripts/gmr24.py` (r24 routes), `MoonShots/` (replay frames).

## D. Neighbouring walls (skim, for context)
- `DRAFTER_98.md`, `DRAFTER_WALL.md`, `DEPTH_WALL.md`, `KNOWLEDGE_WALL.md`, `DRAFT_HIDDEN.md`: the drafter side; kept per check is capped ~12-13.
- `HC_READ.md`, `CLOUD_ANSWER_DENSE.md`: the dense pass.
- `COMMENT_LOOPHOLE.md`: relaxed acceptance in comments.
