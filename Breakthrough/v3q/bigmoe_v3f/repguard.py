"""Repetition guard for BigMoE's own decode loops (decode.greedy, SpecDecode's spec125.generate): MTPLX's LoopGuard
(loop-armed DRY steering, mtplx/loop_guard.py) on the committed tokens. Release must-have (Eval suite: the 3-bit plan
loops to the cap on ifeval/1219).

Exactness: while disarmed nothing changes (same graph, same argmax). It arms only when a 12-gram recurs 4x in the last
2,048 tokens past 256 tokens; while armed the loops decode 1-token steps and subtract the guard's penalties from that
step's logits before the argmax; it disarms after 256 tokens without a penalty. On by default; BIGMOE_LOOP_GUARD=0 off.
MTPLX knobs (MTPLX_LOOP_GUARD_*) apply."""
import os
import mlx.core as mx

STATS = {"armed": 0, "steered": 0, "penalized": 0}


def make(tokenizer=None):
    if os.environ.get("BIGMOE_LOOP_GUARD", "1") != "1":
        return None
    from mtplx.loop_guard import LoopGuard, loop_guard_config_from_env
    cfg = loop_guard_config_from_env(True, tokenizer=tokenizer)
    return LoopGuard(cfg) if cfg.enabled else None


def observe(guard, out):
    if guard is not None and guard.observe(out) == "armed":
        STATS["armed"] += 1


def steer(guard, working, logits):
    """logits [V] (any dtype) -> float32 logits with the guard's penalties for the position after `working`."""
    lg = logits.astype(mx.float32)
    pen = guard.penalties_for(working) if guard is not None and guard.armed else None
    STATS["steered"] += 1
    if pen:
        STATS["penalized"] += 1
        idx = mx.array(list(pen.keys()), dtype=mx.int32)
        lg = lg.at[idx].add(-mx.array(list(pen.values()), dtype=mx.float32))
    return lg
