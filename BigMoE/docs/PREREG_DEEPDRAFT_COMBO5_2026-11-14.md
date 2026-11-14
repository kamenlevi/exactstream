# PREREG: DEEPDRAFT COMBO5, can cheap layer-5 taps repair the head's own stand-in? (Sat 14 Nov 2026, ~15 min)

Written before the run. Leap's screen (17:5x, Leap fce1ca7; ~/Leap/data/week98/README.md, addendum), after LAYERS =
COSTLY:
- Taps from early layers alone are worse than the stale stand-in ({5}: 41.66 at MBPP 23-30, kept 2.72).
- Layer 45 recovers 91 % of the gap, but a refresh at L 45 is a verify pass. No refresh schedule beats plain 16-row
  checks, so the BLOCK sweep was stopped after 2 of 20 evals; it couldn't change that verdict.
- What's left: give a cheap fresh signal to a map that also sees what the head already carries.

## Setup
- **Pairs:** t30b's own chain (no training) over the 100 re-prefilled training files (dd_taps.py collect).
  - At depths j = 2, 4, 8, 16, 24, 30, every 2nd anchor, it collects the carried stand-in (10240), the raw layer-5 tap
    of the same row (2560), and the re-prefilled wd of that row (the target, as in LAYERS).
- **Map:** ridge [stand-in, tap 5] (12800) -> wd. 90 files fit it; lambda is picked on the other 10.
- **Evals:** t30b on MBPP 100 (dd_taps.py evalrep --replace combo).
  - The stand-in is replaced by the map at every depth (k = 1, the upper bound) and every 8 depths (k = 8, Leap's cost
    point: L 5 refreshed every 8 drafted tokens, +11 ms per check at 45 GB).
  - Also t30b's own chain, to give its exact kept per 16-row check. Leap's verdict leans on a constant-p estimate,
    ~6.38.
- Kept is reported per 32-row check (the eval's metric) and per 16-row check.

## Lines (result -> action; MBPP 100, k = 8)
| Result | Line | Action |
|---|---|---|
| LIVE | kept per 32 >= 7.7 (Leap's L 5, k 8 break-even at 45 GB; 7.5 at 24 GB) | A cheap refresh repairs the stand-in. Leap prices it against plain 16-row checks; a trained version (the head learns to use the tap) is the next screen. |
| CLOSED | below 7.7 | Freshness from the trunk is closed for the week. The 98 % method has to make the drafter predict the deep state itself. |

- The k = 1 upper bound, every band and the R2 are reported with it.

## Result (Sat 14 Nov 18:02): CLOSED. Cheap layer-5 taps can't repair the stand-in; freshness from the trunk is closed for the week
- **Pairs:** 70,810 train rows from 90 files, 9,924 validation rows (t30b's own chain, depths 2/4/8/16/24/30, every 2nd
  anchor).
- **The map:** ridge [stand-in, tap 5] -> wd, best lambda 0.1 x mean diag: R2 0.448, normalized error 0.617.
  - The stand-in as is, read as a prediction of wd, has R2 -2.32. The head's carried state is far from the true stream
    in absolute terms, yet the head drafts 72 % at depth from it.
- **MBPP 100, t30b:**

  | arm | 23-30 | kept per 32 | kept per 16 |
  |---|---|---|---|
  | t30b's own chain (no replacement) | 72.13 | 6.92 | **6.59** |
  | the map every 8 depths (k = 8) | 65.85 | 6.50 | 6.31 |
  | the map every depth (k = 1) | 25.54 | 3.26 | 3.26 |

- **Verdict: CLOSED.** kept per 32 at k = 8 is 6.50, below Leap's 7.7 break-even and below t30b's own 6.92.
- **t30b's exact kept per 16-row check on MBPP is 6.59** (Leap's constant-p estimate was ~6.38). Leap's verdict (a
  refresh at L 45 loses to plain 16-row checks; it would flip only if t30b kept <= ~5.9 in 16 rows) stands, and more so.
- **What it means for WEEK98:**
  - Freshness from the trunk is closed: early taps don't carry the state, and layer 45 costs a verify.
  - The 98 % method has to make the drafter predict the deep state itself.
  - The fresh-state oracles (93.6 % at 23-30, flat with depth) show the head reads a fresh state well. The lever is
    a better stand-in, learned.
