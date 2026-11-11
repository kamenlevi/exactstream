"""R-84 DRAFT-HEAD (Leap, Sun 2026-11-01; shared GPU, day-sized ~3 GB): DFlash2's head (the target's 4-bit lm_head,
248,320 x 2560, called with 15-16 rows) on qmm16c (REF2's 16-row NAX kernel; kern23: 1.53 vs 2.21 ms for MLX's 16-row
call) instead of MLX's call. The draft logits round differently, so drafts are NOT bitwise; the target still verifies
every token, so outputs stay exact and only kept-per-check can move.
1. ms of head2s (lm_head + the two-stage top-16), MLX vs qmm16c, 16 rows (shared GPU, rough).
2. Kept per check, teacher-forced along the 125B's own tokens (D-02b's replay: dfl2d_code 24 samples, dfl2d_math 60), the
   L1-20 drafter with draft_fused (R-76 / R-78) and chain_fused (R-77) in both arms: stock head vs qmm16c head; plus the
   share of checks whose drafts are identical.
Usage: [BigMoE venv] r84_draft_head.py -> ~/Leap/data/r84_draft_head.txt"""
import glob, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_third_party"))
import dflash_model_mlx as DF
import draft_fused as DFU
from chain_fused import chain_fused
from qmm16 import qmm16c

lines = []
def say(s):
    print(s, flush=True); lines.append(s)
D = os.path.expanduser("~/Leap/data")
V, K = 248320, 16
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
d = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
d.load_weights(os.path.join(D, "l120_ft.safetensors"), strict=False)
d.bind(tgt)
mx.eval(d.parameters())
DFU.install(forms=True, draft=d)
sel = d.candidate_selector
HEAD = ["mlx"]
def logits_of(hid):
    lead, rows = hid.shape[:-1], hid.size // hid.shape[-1]
    lm = d.lm_head
    if HEAD[0] == "q16" and rows <= 16:
        x = hid.reshape(rows, -1)
        x16 = x if rows == 16 else mx.concatenate([x, mx.zeros((16 - rows, x.shape[-1]), x.dtype)])
        lg = qmm16c(x16, lm.weight, lm.scales, lm.biases)[:rows].reshape(*lead, -1)
    else:
        lg = lm(hid)
    lg = lg * d.config.output_multiplier
    if d.config.final_logit_softcapping:
        cap = d.config.final_logit_softcapping
        lg = mx.tanh(lg / cap) * cap
    return lg
def head2s(hid):
    lg = logits_of(hid)
    B, s = 128, lg.shape
    blk = lg.reshape(*s[:-1], s[-1] // B, B)
    tb = mx.argpartition(blk.max(axis=-1), -K, axis=-1)[..., -K:]
    sel_ = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K, B)), axis=-2).reshape(*s[:-1], K * B)
    j = mx.argpartition(sel_, -K, axis=-1)[..., -K:]
    c = mx.take_along_axis(tb, j // B, axis=-1) * B + j % B
    return c, mx.take_along_axis(lg, c, axis=-1)
def fn(blk, ctx, cache):
    hid = d.hidden_states(blk, ctx, cache, 1)
    cand, unary = head2s(hid)
    return chain_fused(blk, unary, cand, sel.hidden_projection(hid), sel)[0]

# 1. head timing at 15 rows (the drafter's rows after logits_start 1)
hid = (mx.random.normal((1, 15, 2560)) * 0.05).astype(mx.bfloat16); mx.eval(hid)
for h in ("mlx", "q16", "mlx", "q16"):
    HEAD[0] = h
    for _ in range(5):
        mx.eval(head2s(hid))
    ts = []
    for _ in range(40):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(head2s(hid)); ts.append(time.perf_counter() - t0)
    say(f"head2s with the {h} head: {1e3 * float(np.median(ts)):.2f} ms (15 rows, shared GPU, rough)")

# 2. teacher-forced replay
def walk(sample):
    f, ids, off, gen = sample
    cache = d.make_cache()
    MASK = int(d.config.mask_token_id)
    start = max(off, gen - (d.config.sliding_window - 1))
    for c in cache:
        c.offset = start
    fed, a, out, drafts, n = start, gen, [], [], len(ids)
    while a < off + n - 1:
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        blk = mx.array([[int(ids[a - off])] + [MASK] * 15])
        dr = np.array(fn(blk, ctx, cache)[0]).tolist()
        drafts.append(tuple(dr))
        fed = a
        true = ids[a - off + 1:a - off + 16]
        k = 0
        while k < len(true) and dr[k] == true[k]:
            k += 1
        out.append(k + 1)
        if (trim := cache[0].offset - a) > 0:
            DF._trim_recent_cache(cache, trim)
        a = a + k + 1
    return out, drafts
for folder, name in (("dfl2d_code", "code (HumanEval+)"), ("dfl2d_math", "math (GSM8K CoT)")):
    smp = []
    for p in sorted(glob.glob(os.path.join(D, folder, "*.npz"))):
        z = np.load(p)
        smp.append((z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])))
    res = {}
    for h in ("mlx", "q16"):
        HEAD[0] = h
        kept, drafts = [], []
        for sm in smp:
            o, dr = walk(sm)
            kept += o; drafts.append(dr)
        res[h] = (kept, drafts)
    same = sum(int(a == b) for da, db in zip(res["mlx"][1], res["q16"][1]) for a, b in zip(da, db))
    tot = sum(min(len(da), len(db)) for da, db in zip(res["mlx"][1], res["q16"][1]))
    say(f"{name}: kept per check mlx head {np.mean(res['mlx'][0]):.3f} ({len(res['mlx'][0])} checks), qmm16c head "
        f"{np.mean(res['q16'][0]):.3f} ({len(res['q16'][0])} checks); identical drafts on {same} / {tot} aligned checks")
DFU.uninstall()
open(os.path.join(D, "r84_draft_head.txt"), "w").write("\n".join(lines) + "\n")
