# The agent FAIL's cause bisect: one FAST lever off per arm (Fri 2026-11-06; rows 11:4x – 12:43)

Pre-registration: `docs/PREREG_agentbisect_2026-11-06.md` (11:28 + the recipe and a status note, each before its rows).
On agentlong/write12-14, the agent check's worst turns, plain FAST runs on the frozen outside code with one lever off per
arm. Each arm is scored against its own exact base (stage `tfwin`). The lines are NewIdeas'.

## Results (forward unless noted; base = plain FAST, rerun)
| arm | lever off | blunders | vs base (21) | deviations | verdict |
|---|---|---|---|---|---|
| base | none | 21 (reverse 18) | - | 10.75 % (rev 10.95 %) | - |
| qom | QOM-KEEP (R57_QOM_SETS without agentlong) | 4 (reverse **0**) | 19 % | 6.90 % (rev 7.82 %) | **carries it** |
| d17 | DRAFT17's bonus row | 9 | 43 % | 10.49 % | partial |
| hc8 | HC8 (both flags) | 20 | 95 % | 12.55 % | not the main cause |
| nax | FAST-NAX's base (REF2-T's reference) | 20 | 95 % | 12.85 % | not the main cause |
| small | SMALL-N | 21 | 100 % | 11.23 % | not the main cause |

- **The sharp test fails.** With QOM-KEEP off the two orders still differ on all three turns (first differences at tokens
  4, 64 and 4). Base differs on all three too (2, 9, 4).
  - So QOM-KEEP is not the only lever whose tokens depend on the cache's history.
  - The remaining dependence brings few blunders (qom: 4 and 0). Its 7-8 % deviations are mostly near-ties.
- base's 21 / 18 here vs the agent check's 16 / 27 on the same turns: the chunking (all three turns in one process) is a
  different history. That is why base was rerun.

## What it means
- **QOM-KEEP carries the agent turns' low-probability tokens.** Its sure rows (the resident keep and the 3-bit tails)
  account for ~80-100 % of the blunders on long agent writes.
- DRAFT17's bonus row adds part. HC8, the NAX6-N base and SMALL-N do not move the count.
- A second history-dependent lever exists. Finding it needs QOM off plus one more lever off, in two orders. That is
  BigMoE's / NewIdeas' call.
- BigMoE offered R57_QOM_NOKEEP (QOM's keep split from its 3-bit tails) for a split of QOM's own two parts.

## Runs
- **Runner 50:**
  - 11:36-12:27 on freeze af1: qom fwd / rev, base fwd / rev, d17, hc8.
  - 12:29-12:43 on af2: nax, small.
- The nax arm's first try voided on af1: p11 loaded through tierglu_3cn's unrewritten live path.
- af2 rewrites every frozen .py. Its 33 reference modules equal af1's modulo the directory name.
- **Data:** results/bisect_<arm>_<order>/ (rows + tfwin), results/agentbisect_report.txt, results/bisect/arms.txt.

## Addendum (Fri 12:53): the drafts carry the history (the arms' own check logs, no new rows)
- **BigMoE's mechanism (12:5x):** if any FAST kernel's bits depend on a check's row count or position, then even exact
  acceptance depends on the drafts. The drafts can carry history or timing. So arms with one lever off cannot isolate
  it.
- **The checks logs** (results/agentfast/es_bi_*_checks.json: per check, the drafted rows and the kept tokens), the two
  orders compared:

  | arm | turn | first (rows, kept) difference | committed position | first token difference |
  |---|---|---|---|---|
  | qom | write13 | check 9: rows 11 vs 10 (kept 2 = 2) | ~37 | 64 |
  | qom | write12 | check 2: rows 2 vs 4 | ~5 | 4 |
  | qom | write14 | check 2: rows 3 vs 2 | ~5 | 4 |
  | base | write12 / 13 / 14 | check 1-2 | ~3-9 | 2 / 9 / 4 |

- **What it means:**
  - On write13, the drafted row count differs about 27 tokens before the committed tokens do. The drafter's proposals
    or its confidence cut depend on more than the committed prefix.
  - Elsewhere the split is at the very first checks. The seed is at the start: the prefill's taps, or a drafter / n-gram
  cache that carries over (BigMoE's suspects).
- **The next tests** (BigMoE's (a) / (b); NewIdeas schedules them):
  - (a) the exact base in both orders, which must be identical;
  - (b) one lever on at a time from it, QOM off.

