"""FAR's WALK-MIX TTT (THINK_1116c; ~/FableGod/far/IDEAS_1116c_drift.md idea 1; NewIdeas 18:5x drift screen 2), w = 0.0.
Training keeps accept-until-fail for the survivors (KL weight 1, the failing depth included, as t30b) and keeps the
rows that missed in the chain, at KL weight w from the next depth on (sb/core.py ttt_step reads _walkmix_w off this
hook; the depth weight uses the weighted row count, so w = 0 is today's loss and w = 1 the 'allrows' loss).
Training-only: the read-out chain is unchanged (the hook is an identity).
w = 0 is the smoke's check arm: its loss must match the in-batch control's."""
W = 0.0


def standin(j, h, anc):
    return h


standin._walkmix_w = W
