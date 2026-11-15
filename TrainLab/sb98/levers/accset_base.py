"""MTP Head lever family 'accset' (WEEK98 S7 ACCSET; REPORT 'WEEK98 S7 ACCSET'): train the chain head for close-call
acceptance. Per depth the loss becomes Sandbox98's KL_top (unchanged) + LAM x the masked row mean of
-log sum_{t in A} p_d(t), with A = {t in the target's stored top-32 : lp_top1 - lp_t <= DELTA} (the target's close-call
accept set; the top-1 is always in it). With SURV, a training row stays in the chain while the head's #1 equals the true
token or lies in A, so training follows the close-call walk the read-out scores; without it, rows leave at the first
exact miss as now. Training-only: the read-out chain is unchanged (the standin hook is the identity).

Mechanism, with Sandbox98's code unmodified (as levers/allrows.py): ttt_step's per-depth function core._KLG['vg'] is
wrapped once. train_multi sets core.REPL['fn'] to the stepping arm's standin hook; each arm's hook carries its own
parameters, so arms without the marker (e.g. a control in the same process) run core._kl_hit itself."""
import mlx.core as mx


def make(delta, lam, surv):
    def standin(j, h, anc):
        return h
    standin._mh_accset = {"delta": float(delta), "lam": float(lam), "surv": bool(surv)}

    def setup(D, args):
        from sb import core as C
        if C._KLG.get("mh_accset"):
            return
        base = C._kl_hit

        def kl_hit(hn, tk, lp, mask, tgt):
            P = getattr(C.REPL["fn"], "_mh_accset", None)
            if P is None:
                return base(hn, tk, lp, mask, tgt)
            lg = C.lm_logits_train(hn)[0]                              # as core._kl_hit
            lq = mx.take_along_axis(lg, tk, -1) - mx.logsumexp(lg, -1, keepdims=True)
            den = mx.maximum(mask.sum(), 1.0)
            kl = (C.kl_top(lq, lp) * mask).sum() / den
            inA = (lp[:, :1] - lp) <= P["delta"]                      # [n, 32]: the target's accept set
            las = -mx.logsumexp(mx.where(inA, lq, -1e30), -1)          # -log of the head's mass inside it
            am = mx.argmax(lg, -1)
            hit = am == tgt
            if P["surv"]:
                hit = mx.logical_or(hit, mx.logical_and(tk == am[:, None], inA).any(-1))
            return kl + P["lam"] * (las * mask).sum() / den, mx.stop_gradient(hit.astype(mx.float32))
        C._KLG["vg"] = mx.value_and_grad(kl_hit)
        C._KLG["mh_accset"] = True
    return standin, setup
