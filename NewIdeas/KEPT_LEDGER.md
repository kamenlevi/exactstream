# KEPT_LEDGER: tokens kept per 16-row check, per lever and domain (NewIdeas; Fable's DRAFTER_WALL section 3 item 7)
Purpose: check the drafter stack's arithmetic against measured walks instead of assuming it. One row per measured lever.
"x" = ratio to the row's own base on the same prompts. Blank = not measured. Started Mon 2 Nov 2026 from the owners'
REPORT lines; NewIdeas adds each new walk at its wakes. Owners: report kept per check per domain and the base with every
drafter result.

## Ceilings and bases
| lever | base | code | math | IFEval | multi | agent | notes | source |
|---|---|---|---|---|---|---|---|---|
| full-size twin (ceiling, on the 125B's own answers) | - | 14.8 | 14.2 | 10.6-11.1 (chat) | | | upper bound, not a drafter | MTP Head, SUMMARY_FOR_KAMEN |
| DFlash2, no confidence cut | - | 10.44 | 6.6 | | | | old default, 16 rows | D-22 |
| MTP head, EAGLE-3 training-time test | DFlash2 | 10.98 | 6.86 | | | | exact | D-22 |
| DFlash2 L1-20 (E2's base) | - | 9.57 | 7.00 | | | | live 45 GB loop | E2 / D2, 1d8ca7f |
| DFlash2's top-16 candidates (tree upper bound) | DFlash2 picked 9.5 | 12.8 | | | | | trees KILL at equal cost | TREES, Breakthrough 91b0a56 |

Per-position accuracy (the drafter's horizon, Lunatic round 2, lu_alpha.out): on rows the 125B is sure of, DFlash2 is
99.6 % right at block position 0 and 88.4 % at positions 12-15. Deep positions are where kept per check is lost.

## Levers
| lever | base | code | math | IFEval | multi | agent | verdict | source |
|---|---|---|---|---|---|---|---|---|
| E2 / D2 x2: one doubling of fresh MBPP code from the 125B | L1-20 | 9.79 (+0.22) | 6.99 | | | | PASS (line +0.10 per doubling) | 1d8ca7f |
| E2 / D2 x4: two doublings | L1-20 | 9.85 (+0.06 over x2; slope +0.14 per doubling) | 7.06 | | | | PASS by the line; slow lever (code >= 13 needs ~23 more doublings) | c051d23 |
| HPD-1 H: 2 more epochs where L1-20 is unsure | L1-20 | 9.52 (vs 9.57) | | | | | KILL | 555690b |
| E1 coverage: more de / ru / ja answers | L1-20 | | | | 2.05 (vs 2.06) | | KILL | REPORT 'E1 COVERAGE RESULT' |
| E5 DFlash-X: borrow resident experts | L1-20 | | | -0.01 | | | KILL | 40c90a5 |
| MTP chain K15 + the default's cut, on agent | DFlash2 4.80 | | | | | 8.20 | kept up, tok/s flat (cold experts) | D-23d, d0a50cd |
| MTP chain alone vs DFlash2, chat | DFlash2 | | | x1.08 | x1.32 | | chat switch PASS on tok/s (D-23d) | D-17, D-23d |
| MATCH: session match override (Fable 2.1) | drafter | x1.038 (HumanEval+); x1.204 at 32 rows = 12.35 | | | | x1.047 | BETWEEN (MBPP x0.980: imprecise matches) | a2086f2 |

## Walks (several levers measured together; compare with the product of the rows above)
| stack | code | math | IFEval | multi | agent | product of rows | source |
|---|---|---|---|---|---|---|---|
| (none measured yet; DUALDRAFT Tue by day, MATCH + precision gate next) | | | | | | | |
