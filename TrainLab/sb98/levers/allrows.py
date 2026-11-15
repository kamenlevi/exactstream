"""MTP Head lever 'allrows' (WEEK98 screen 2, THE CARRY; REPORT 'WEEK98 S2 CARRY400'): training keeps EVERY row in the
chain at every depth (no accept-until-fail truncation). The deep carry then trains on all continuations, not only on
rows whose earlier guesses were all right, and the deep depths weigh more (their counted rows no longer shrink with
survival; the KL mask and wdecay are unchanged). Training-only: the read-out chain is unchanged.

Mechanism, with Sandbox98's code unmodified: ttt_step's per-depth KL function (core._KLG['vg']) is wrapped once so that
it reports every row as alive only while THIS arm is stepping. train_multi sets core.REPL['fn'] to the stepping arm's
standin hook; this lever's hook (an identity) carries the marker. Other arms in the same process are unaffected."""
import mlx.core as mx


def standin(j, h, anc):
    return h


standin._mh_allrows = True


def setup(D, args):
    from sb import core as C
    if C._KLG.get("mh_allrows"):
        return
    base = C._kl_hit

    def kl_hit(hn, tk, lp, mask, tgt):
        l, hit = base(hn, tk, lp, mask, tgt)
        if getattr(C.REPL["fn"], "_mh_allrows", False):
            hit = mx.ones_like(hit)
        return l, hit
    C._KLG["vg"] = mx.value_and_grad(kl_hit)
    C._KLG["mh_allrows"] = True
