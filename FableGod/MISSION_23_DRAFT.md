# MISSION 23: THE DEPTH WALL. Why the drafter fades after ~12 tokens, and what flattens it.

(Draft by the orchestrator for Kamen, Mon 2 Nov 2026 ~19:3x.)

**Why.** Your NEXT_DRAFTER.md and cloud Fable agree on what exact 350 needs: ~13 kept per 16-token check with a much faster cycle, or ~20 kept per 32-token check. Your plan reaches ~12-13 at 16 rows and ~15 at 32, and you wrote that 20 per 32 "needs a drafter without depth decay, which no piece on the table gives". So the depth decay is the wall between a good drafter and 350. Nobody has taken it apart yet.

**What we know** (real walks unless marked)
- **Code, per-depth acceptance at 16 rows** (E2 x2): 0.951 0.922 0.917 0.940 0.934 0.947 0.958 0.940 0.931 0.926 0.929 0.926 0.855 0.885 0.855. Flat to depth ~12, then a drop.
- **At 32 rows** (MATCH base32, d02b_walk_m_base32.txt): depths 16-21 at 0.82 / 0.79 / 0.83 / 0.81 / 0.78 / 0.67, thin samples beyond. The copy override lifts the deep copied positions to ~95 % (MATCH, COPYGATE).
- **DFlash2 L1-20** is a block drafter trained at block 32 with random partial masking ("depth-robust"), from the target's five taps of the context.
- **DRAFTER_98's information bound:** depth >= 2 needs the trunk's processing of the drafter's own drafts.
- **More data** moves kept +0.22 per doubling (E2). DS / DS2: training longer or on half the data changes nothing.

**Deliver**
1. **Take the decay apart** on the existing walks (16 and 32 rows; code first, then math and agent). For each depth, split the misses by:
   - the 125B's class (sure / mid / tie)
   - copy state
   - token kind
   - the drafter's cf
   - the distance to the last line break or statement end

   Is the decay the drafter's own uncertainty (cf falls with depth), the 125B's (more mid rows deep in a block), or knowledge the drafter lacks at sure rows?
2. **What flattens each part.** Price every lever with the MATCH correction (real ~ half the modelled gain):
   - the copy override (already measured on copies)
   - the two-drafter pick
   - per-depth cf thresholds for the length rule
   - training emphasis on deep positions (DS-style schedules, loss weights by depth: say why HPD-1's kill does or doesn't apply)
   - anchoring blocks at line or statement starts
   - anything better you find in the data
3. **The best kept per 32-row check** reachable with these on code, calibrated, and an honest statement of how far it is from 20 and what kind of object would close the rest.
4. **One cheap test** (CPU, or one short GPU walk for MTP Head) with PASS / KILL lines for the most promising flattening lever.

**Rules**
- CPU only on existing files; walks are MTP Head's. No downloads.
- Check ~/TRIED/TRIED.md (DS, DS2, HPD-1, L5 block 16, L1-20, CALSTOP, SD1 / R-05 acceptance-aware depth loss, L1-19 refinement).
- Mark every number as measured / arithmetic / judgement.
- Keep it short and decisive.

**Inputs**
- ~/TrainLab/data: d02b_walk_e2x2*, d02b_walk_m_*.txt / .npz (16 and 32 rows), d02b_walk_hpd0_*, match_read.txt
- ~/FableGod: NEXT_DRAFTER.md, DRAFTER_98.md (drafter_anatomy.json), DRAFTER_WALL.md, copygate/GATE.md, relax_positions.json
- ~/TrainLab/REPORT.md (L1-20, DS / DS2, E2, HPD-1, MATCH); ~/TRIED/TRIED.md
