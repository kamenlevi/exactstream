"""MTP Head lever family 'margin' (WEEK98 S8b MARGIN; REPORT 'WEEK98 S8b MARGIN'): an argmax-margin loss for the
close-call rule. Per depth the loss becomes Sandbox98's KL_top (unchanged) + LAM x the masked row mean of
relu(GAMMA - (z_in - z_out)): z = the head's logits, IN = the true token or A = {t in the target's stored top-32 :
lp_top1 - lp_t <= DELTA}, z_in = the max over IN, z_out = the max over every other token (IN logits masked to
-1e30). Zero where the best IN logit already leads by GAMMA. With SURV, a training row stays in the chain while the
head's #1 is IN (as accset's SURV). Training-only: the standin hook is the identity.
Mechanism as levers/allrows.py / accset_base.py: core._KLG['vg'] is wrapped once; the stepping arm's hook carries its
parameters (train_multi sets core.REPL['fn'] per arm)."""
import mlx.core as mx


def make(delta, lam, gamma, surv):
    def standin(j, h, anc):
        return h
    standin._mh_margin = {"delta": float(delta), "lam": float(lam), "gamma": float(gamma), "surv": bool(surv)}

    def setup(D, args):
        from sb import core as C
        if C._KLG.get("mh_margin"):
            return
        base = C._kl_hit

        def kl_hit(hn, tk, lp, mask, tgt):
            P = getattr(C.REPL["fn"], "_mh_margin", None)
            if P is None:
                return base(hn, tk, lp, mask, tgt)
            lg = C.lm_logits_train(hn)[0]                              # as core._kl_hit
            lgk = mx.take_along_axis(lg, tk, -1)
            lq = lgk - mx.logsumexp(lg, -1, keepdims=True)
            den = mx.maximum(mask.sum(), 1.0)
            kl = (C.kl_top(lq, lp) * mask).sum() / den
            inA = (lp[:, :1] - lp) <= P["delta"]                      # [n, 32]: the target's accept set
            tg = tgt.astype(mx.int32)[:, None]
            z_in = mx.maximum(mx.max(mx.where(inA, lgk, -1e30), -1), mx.take_along_axis(lg, tg, -1)[:, 0])
            ins = mx.concatenate([mx.where(inA, tk.astype(mx.int32), tg), tg], -1)          # [n, 33] IN ids (pad = true)
            z_out = mx.max(mx.put_along_axis(lg, ins, mx.array(-1e30), axis=-1), -1)      # best OUT logit (5x faster
            #                                                  than a top-34 argpartition; same value, checked 09:4x)
            hinge = mx.maximum(P["gamma"] - (z_in - z_out), 0.0)
            am = mx.argmax(lg, -1)
            hit = am == tgt
            if P["surv"]:
                hit = mx.logical_or(hit, mx.logical_and(tk == am[:, None], inA).any(-1))
            return kl + P["lam"] * (hinge * mask).sum() / den, mx.stop_gradient(hit.astype(mx.float32))
        C._KLG["vg"] = mx.value_and_grad(kl_hit)
        C._KLG["mh_margin"] = True
    return standin, setup
