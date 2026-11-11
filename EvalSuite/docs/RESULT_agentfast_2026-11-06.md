# The agent check: plain FAST on long agent turns vs the agent bar (Fri 2026-11-06; rows 08:26 – 11:24)

Pre-registration: `docs/PREREG_agentfast_2026-11-06.md` (08:09 + the recipe and three status notes, each before its rows).

## Verdict: FAIL (85 blunders; the line is 0). Agent turns run EXACT.
| order | rows | tokens | not the reference's argmax | blunders (the reference's p < 0.01) |
|---|---|---|---|---|
| forward | 17 | 9,631 | 780 (8.10 %) | 36 |
| reverse | 17 | 10,118 | 824 (8.14 %) | 49 |
| both | 34 | 19,749 | 1,604 (8.12 %) | **85** (~43 per 10K) |

- **The mode:** plain FAST (QOM-KEEP + DRAFT17 + the FAST all-in, no RELAX gate) on NAX6-N, at C 224, CAP 1024.
  - It ran on BigMoE's r57_screen_fast.py with frozen outside code: Leap 03a63c5, BigMoE fce0e55 (NEVER-WAIT inert),
    TrainLab and Breakthrough copies (snap/r57frz-af1). The forward write11-16 rows ran on the same versions live.
- **The scoring:** every committed token goes through NAX6-N's exact verify path along FAST's own prefix (stage `tfwin`).
  - Its self-check (exact rows of results/ref2tnax reproduced 100 %) passed in every process.
- **Where the blunders sit:** 75 of the 85 are inside the written code (the tool call's first parameter), 10 in
  descriptions.
- **By turn:**
  - Short edits are clean: edit00 / 01 / 04-06 / 10 have 0-3 % deviations and 0-1 blunders.
  - The long file writes carry the failure: write12-14 have 16-20 % deviations and up to 16 blunders in one turn.
- **Against B with the RELAX gate** on the same turns (RESULT_relax_drift): 22.6 blunders per 10K. Plain FAST is
  worse, not better, on agent code.
- **Cache dependence:** the two orders give token-identical rows on 3 of 17 turns; several diverge by token 1-6.

## What it means
- Agent turns run EXACT (the PREREG's FAIL branch) until a FAST form passes this bar.
- The two orders differ this much, so a certificate for any cache-dependent mode on agent turns needs both histories,
  as this one had.

## Runs and provenance
- **Runner 47 (08:19-08:57):** the smoke (tfwin's self-check + edit10: 0 deviations). FAST forward's first process kept
  edit00-05. FAST reverse's first process crashed in BigMoE's screen (a run cut by HARD with 0 checks); BigMoE fixed it
  in 9c82d1d.
- **Runner 48:**
  - 09:02-09:25: the forward order in chunks, stopped by a BigMoE commit.
  - 09:28: voided a chunk when Leap's ref2.py changed under it.
  - 09:40-11:24: on the freeze, the forward edits rerun, reverse in full, then tfwin on both.
- Forward rows from Leap 7ebf2d7 (edit00-10) are set aside as information: results/agentfast_fwd_v1/.
- **Data:**
  - results/agentfast_fwd/, results/agentfast_rev/ (rows + tfwin/);
  - results/agentfast/ (tokens, runs, per-process module manifests);
  - results/agentfast_report.txt;
  - snap/r57frz-af1/FREEZE.json.
