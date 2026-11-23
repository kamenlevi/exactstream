"""DRAFT17 (MTP Head, Wed 11-04 13:58; pre-registered in REPORT 'DRAFT17'; routed by the orchestrator 13:5x for NewIdeas'
13:44 question, build #6's gated last row): the engine's DFlash2 block pass at 16 / 17 / 32 positions, in ms.
Path = BigMoE's r57_screen.py propose() on the 45 GB default (K83_EXACT: Leap's draft_fused with R-78's forms, chain_fused;
head2s with K 16), copied verbatim: hidden_states(blk, ctx, cache, 1) -> head2s -> hidden_projection -> chain_fused.
Arms: R16 (today); R17 / R32 as the code stands (R-78's forms are 16-row only: other row counts take the fallback spelling);
R17g / R32g with the forms opened to that row count (draft_fused._rows16 patched).
Weights: L1-20 (bf16, loaded over incoai's base as the engine does); the target's 4-bit embed + LM head (shard 16).
Context: per arm its own cache prefilled with CTX rows of bf16 noise, then +NEW rows per call (a check's accepted tokens);
the mask memo cleared before each call (the engine builds it once per check); arms interleaved, order reversed every other
rep; WARM untimed + REPS timed calls; median / p10 / p90 ms.
env: D17_CTX (600,1900), D17_NEW (10), D17_REPS (40), D17_WARM (5), D17_OUT (~/TrainLab/data/draft17_time.txt).
"""
import os, sys, time, resource
os.environ.setdefault("HF_HUB_OFFLINE", "1")             # the drafter's config / base weights are in the HF cache: no network
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("~/Leap/third_party"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
import dflash_model_mlx as DF
import draft_fused as DFU
from chain_fused import chain_fused

mx.set_cache_limit(256 << 20)                           # TTS1's lesson: keep MLX's buffer cache from growing the footprint
CTXS = [int(c) for c in os.environ.get("D17_CTX", "600,1900").split(",")]
NEW = int(os.environ.get("D17_NEW", "10"))
REPS = int(os.environ.get("D17_REPS", "40"))
WARM = int(os.environ.get("D17_WARM", "5"))
OUT = os.path.expanduser(os.environ.get("D17_OUT", "~/TrainLab/data/draft17_time.txt"))
K, TOK = 16, 9707
ARMS = (("R16", 16, False), ("R17", 17, False), ("R17g", 17, True), ("R32", 32, False), ("R32g", 32, True))
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


t0 = time.time()
draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.load_weights(os.path.expanduser("~/Leap/data/l120_ft.safetensors"), strict=False)   # the engine's drafter (bf16)
mx.eval(draft.parameters())
V = draft.config.vocab_size
dt_mlp = draft.layers[0].mlp.gate_proj.weight.dtype


class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(V, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, V, bias=False, group_size=64, bits=4)


w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
tgt = Target()
tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                 + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
del w
mx.eval(tgt.parameters())
draft.bind(tgt)
mx.eval(draft.parameters())
DFU.install(forms=True, draft=draft)                    # the 45 GB default: K83_EXACT with D7B off -> forms on
MASK = int(draft.config.mask_token_id)
W = draft.fc.weight.shape[1]
ROWS16 = DFU._rows16


def set_rows(R, gen):                                   # the forms' row test: 16 only (as shipped) or also R
    DFU._rows16 = (lambda x, R=R: x.ndim >= 2 and x.size // x.shape[-1] in (16, R)) if gen else ROWS16


def head2s(hid):                                        # r57_screen.py head2s, verbatim (draft -> this draft)
    lg = draft.compute_logits(hid)
    B, s = 128, lg.shape
    blk = lg.reshape(*s[:-1], s[-1] // B, B)
    tb = mx.argpartition(blk.max(axis=-1), -K, axis=-1)[..., -K:]
    sel = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K, B)), axis=-2).reshape(*s[:-1], K * B)
    j = mx.argpartition(sel, -K, axis=-1)[..., -K:]
    c = mx.take_along_axis(tb, j // B, axis=-1) * B + j % B
    return c, mx.take_along_axis(lg, c, axis=-1)


def propose(blk, ctx, dcache):                          # r57_screen.py propose with K83 (chain_fused)
    hid = draft.hidden_states(blk, ctx, dcache, 1)
    cand, unary = head2s(hid)
    hp = draft.candidate_selector.hidden_projection(hid)
    return chain_fused(blk, unary, cand, hp, draft.candidate_selector)


say(f"DRAFT17 {time.strftime('%m-%d %H:%M')}: DFlash2 block pass, the engine's K83 path (draft_fused forms + chain_fused, "
    f"head2s K {K}); drafter L1-20 ({dt_mlp}), fc in {W}, vocab {V}; loaded in {time.time() - t0:.0f} s; "
    f"mlx {mx.__version__}; cache limit 256 MB")
for ctx0 in CTXS:
    rng = np.random.default_rng(ctx0)
    pre = mx.array(rng.standard_normal((1, ctx0, W)).astype(np.float32)).astype(mx.bfloat16)
    caches = {}
    for name, R, gen in ARMS:                           # each arm: its own cache with the same prefill
        set_rows(R, gen)
        c = draft.make_cache()
        mx.eval(draft.hidden_states(mx.array([[TOK] + [MASK] * (R - 1)]), pre, c, 1))
        caches[name] = c
    T = {a[0]: [] for a in ARMS}
    P = {a[0]: [] for a in ARMS}
    for r in range(WARM + REPS):
        new = mx.array(rng.standard_normal((1, NEW, W)).astype(np.float32)).astype(mx.bfloat16)
        mx.eval(new)
        for name, R, gen in (ARMS if r % 2 == 0 else ARMS[::-1]):
            set_rows(R, gen)
            blk = mx.array([[TOK] + [MASK] * (R - 1)])
            mx.eval(blk)
            DFU._MASKS.clear()
            ts = time.perf_counter()
            pth, cf = propose(blk, new, caches[name])
            mx.eval(pth, cf)
            dt = (time.perf_counter() - ts) * 1e3
            assert pth.shape == (1, R - 1), (name, pth.shape)
            if r >= WARM:
                T[name].append(dt)
                P[name].append(np.array(pth))
    set_rows(16, False)
    base = float(np.median(T["R16"]))
    say(f"  context {ctx0} rows (+{NEW} per call; {REPS} timed calls per arm after {WARM} warm):")
    for name, R, gen in ARMS:
        t = np.array(T[name])
        say(f"    {name:5s} {R:2d} positions: median {np.median(t):6.2f} ms (p10 {np.percentile(t, 10):6.2f}, "
            f"p90 {np.percentile(t, 90):6.2f}); vs R16 {np.median(t) - base:+6.2f} ms")
    for a, b in (("R17", "R17g"), ("R32", "R32g")):
        eq = np.mean([np.array_equal(x, y) for x, y in zip(P[a], P[b])])
        say(f"    sanity: {a} and {b} draft the same tokens in {100 * eq:.0f} % of calls")
say(f"  mlx peak {mx.get_peak_memory() / 2**30:.2f} GB; max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB; "
    f"{time.time() - t0:.0f} s in all")
open(OUT, "a").write("\n".join(lines) + "\n\n")
