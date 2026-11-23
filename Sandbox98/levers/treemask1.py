"""TREE-AWARE MASK, k = 1 (NewIdeas 20:3x: Sandbox98's follow-up to FAR's WALK-MIX; results/LEVERS_A.md section TREEMASK).
Training keeps accept-until-fail for the survivors (KL weight 1, the failing depth included, as t30b). A row that misses
stays in the chain only when its true token was in the head's top-1 at that depth (rank = tokens with a strictly higher
logit < 1), where a tree still walks on; it trains at KL weight w = 1.0 from the next depth on. A miss outside the
top-1 drops the row as today. sb/core.py ttt_step reads _treemask_k / _treemask_w off this hook; k = 1 is today's loss
exactly (the smoke's check arm). Training-only: the read-out chain is unchanged (the hook is an identity)."""
K, W = 1, 1.0


def standin(j, h, anc):
    return h


standin._treemask_k = K
standin._treemask_w = W
