# s98: Saturday's stack screen (work in progress)

`r57_screen_s98_X.py` is the screen template. Its paths use `frozen_X`; the snapshot builder replaces `X` with the
snapshot's name and freezes Leap / Breakthrough / TrainLab / Fit12 / MoonShots trees beside it.

- How it was made: a three-way merge (`git merge-file`) with base = the live screen at 162d2d9d (t16a's freeze), theirs
  = t16b's screen (TREE16) and ours = g1v3's screen (the base path, G1, Leap b4331e4's REF2_BOF / REF2_HC_B). Paths were
  normalized to `frozen_X` first. There were 3 conflicts, each resolved by keeping both sides. The three inputs are kept
  here (`merge_*.py`).
- Added on top of the merge:
  - CC(1) on the tree walk: `tree16_loop.accept_cc`. With delta 0 it equals TC.walk on 2,000 random trees. Chain checks
    skip the chain CC block.
  - The flags `R57_TREE_NODES` / `R57_TREE_K` and `R57_XRW` (Breakthrough's knob_v2, md5-pinned, ON for the whole run).
  - runs.json gets `tree` {nodes, k, checks, sib_accepted, close_calls} and `xrw` {mode, cand, orig, fallback_not_p11}.
- The tree path still asserts no FAST / SPAN / QOM_MTP / INERT (the orchestrator 08:3x: blob 1 = exact-safe verify
  levers + tree-16 + CC(1)).
- Pending: MoonShots' r57_draft (the Q8H32 / c2 / stored install and the sync-free draft; R57_DRAFT / R57_T30B_FMT /
  R57_T30B_W / R57_TREE_SF), Leap's GDN16 flag, the snapshot builder and the bit tests.
