"""Lever 'distill' (orchestrator 20:2x, item 1: ORACLE DISTILLATION). Teacher = the same head reading the TRUE W at every
depth (the 'true' read-out: 88.35 flat on MBPP). With true W everywhere, depth j for anchor s IS depth 1 for anchor
s + j - 1, so the teacher's distribution at (s, j) is the current head's depth-1 output at row s + j - 1 (no extra chain).
At every depth j >= 2 the student (the real chain, reading its carried stand-in) gets LAM x KL(teacher top-32 + rest ||
student) on the same rows, on top of the usual TTT loss (the target's top-32), weighted like it. The teacher is the
current head under stop-gradient (pass 1 has no autograd), so it improves as the head does. Aim: behave as if fresh."""
import os
import mlx.core as mx

LAM = float(os.environ.get("DISTILL_LAM", "1.0"))


def aux(j, anc_np, ctx):
    if j < 2:
        return None
    from sb import core as C
    D, h1, a0 = ctx["D"], ctx["h1"], ctx["a0"]
    rows = mx.array(anc_np + j - 1 - a0)                     # depth-1 rows of anchors s + j - 1 (< h1's length)
    lg = C.lm_logits_train(D.norm(h1[:, rows]))[0]
    lp_all = lg - mx.logsumexp(lg, -1, keepdims=True)
    t32 = mx.argpartition(-lp_all, kth=31, axis=-1)[:, :32]
    v32 = mx.take_along_axis(lp_all, t32, -1)
    mx.eval(t32, v32)
    return t32.astype(mx.int32), v32, LAM
