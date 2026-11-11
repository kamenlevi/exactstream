# MISSION 28: THE TREE WALL. Make exact branching cheap, so the check can try the MTP head's top 3 at the hard positions.

> (a quote from the project owner, kept private)

**Why.**
- **Tokens kept per check multiplies every line.** 350 needs ~12.5-13 kept per 16-row check. Today it's ~10.5 on code and only 2.0-2.5 on chat.
- **The misses concentrate at "mid" positions, where the 125B itself hesitates:**
  - 12 % of code rows, 20 % of math and 55-58 % of chat (your RELAX_350)
  - 71 % of code's mid rows sit in '#' comments (your COMMENT_LOOPHOLE)
- **New data:** MTP Head's KWALL (b) (~/TrainLab/data/kwall_mtp.txt, commit 8d5e47f) measured the native MTP head at depth 1 on the code map windows:
  - mid rows: top-1 0.860, top-3 0.982
  - sure rows 0.985, tie rows 0.800
  - DFlash2 at the same rows, deeper on average: 0.659

  A check that tries the top 3 at hard positions would catch nearly all of them.
- **Trees have been killed before, mostly on cost:**
  - TREES: best tree vs best chain at equal cost -10.1 .. +2.1 %
  - L1-22: tree N 15 gets 10.43 tokens (+9 % at 16 rows), but the rows cost
  - exact tree checks need per-node GDN states (~50 MB per layer)
  - D-24 MTP leaf trees: KILL on a cost model, PASS on paper
  - M-10 SIBLINGS, the sibling-union wide trees, M-16b TREE-RUNAHEAD
- **The obstacle is exactness in the 36 GDN layers:** each branch needs its own recurrent state, evolved bitwise as the 1-row path would evolve it.

**Deliver**
1. **The exact cost of one branch.** Work from the code: the GDN state shapes from ~/BigMoE/models/bare/config.json, how the 16-row verify computes states and outputs (~/BigMoE/ref/qwen4_exp.py and the engine), attention's KV with masks, and the expert union that sibling rows add. Say what must be computed bitwise for a leaf sibling and for a deeper branch, in bytes and ms at the floor engine.
2. **At least three designs for cheap exact branching at hard positions only.** Starting points, not limits:
   - leaf-only siblings that share one read of the parent state, with a fused state-update-and-output that never stores the child state
   - branching only at the first hard position of a block
   - a small sibling pass for the first rejected position only
   - placing siblings so they share experts

   For each give:
   - extra rows, bytes and ms per check
   - the kept gain, using the 98 % top-3 at mid rows and the class mix
   - the exactness argument (bitwise = REF2-T's 1-row path)
   - the risk
   - what MTP Head or Leap would build
   - the cheapest test: a CPU replay on stored walks (MTP Head's D-24 ranks: ~/TrainLab/data/d24, d24_read.txt)
3. **Chat:** the same at 55-58 % hard rows. What does a tree do for chat's 2.0-2.5 kept?
4. **The bet,** and the kept and tok/s at the floor engine with it. Or the proof that exact branching can't be cheap in GDN, and why.

**Rules**
- CPU only; writes only in ~/FableGod; no downloads. GPU timing is Leap's and MTP Head's.
- Read TRIED's tree entries before writing (TREES, L1-22, D-24, M-10, sibling-union, M-16b, the exact-tree prior art TreeWY / Bole) and say why each design differs.
- Exact by construction, or say why not.
- Calibrate: our estimates have run ~2x high (Leap's D1-B gave -0.74 ms where -2.6 was predicted). Give upper bounds, then halve them.
- Mark every number as measured / arithmetic / judgement. Read in targeted pieces.
