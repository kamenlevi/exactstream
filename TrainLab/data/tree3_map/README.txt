TREE3 (MTP Head, Tue 11-03, for the Breakthrough Push): the top-3 draft tree of DFlash2 L1-20 and of the native MTP head
b5 on the 36 decision-map windows (code = 12 HumanEval+, ifeval = 12, multi = 12), ALIGNED to the map.

Positions are the map's: token index t in `tokens` (== the map's routing/<window>.npz tokens); map row t is the 125B's row
at position t and predicts token t + 1. A check at anchor A (the last known token) drafts depth j = 0..14 = the token at
A + 1 + j = map row A + j. Every position of the answer is an anchor here (`A`), so any loop can be replayed.

Per window, arrays over anchors (n) x depth (15) [x 3]:
  A [n] anchor positions; m [n] depths with a token (<= 15); true [n, 15] the 125B's token at A + 1 + j (-1 past the end)
  dfl_t3 / dfl_p3 [n, 15, 3]  DFlash2's top-3 at (A, j) by its selector softmax over its 16 candidates, teacher-forced
                              (the predecessor is the true token at j - 1), descending; dfl_rank [n, 15] the true token's
                              rank among the 16 (16 = not a candidate); dfl_own [n, 15] / dfl_own_cf the drafter's own
                              chain (what the live engine drafts; equal to the teacher-forced pick up to the first miss)
  mtp_t3 / mtp_p3 [n, 15, 3]  the MTP head's top-3 (f32 softmax over the full vocab), teacher-forced; its chain at a check
                              anchored at A starts from the trunk's hidden at A - 1 (D-24 row A - 1 - a0, depth j + 1);
                              mtp_rank [n, 15] the true token's rank capped at 3 (-1 = no data)
  t125_t3 / t125_p3 [n, 15, 3] the 125B's own top-3 at map row A + j (the map's exact run baseline.json; -1 outside its rows)
Teacher-forced = exact along the true path up to the first miss, which is all a check's kept length depends on.
Built by sticky/tree3_mtp.py, d02b_walk_match.py (TL_WALK_EVERY + TL_WALK_TOP16), sticky/tree3_export.py (turns/tree3_job.sh).
