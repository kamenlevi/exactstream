"""BP-9 (Leap, Wed 2026-11-04 17:0x; TABLE_350 'BP-9'): the screened draft head on qmm16c (d7b_nax.DraftSubset: FR-Spec
top-98,304 + session ids) vs the full head on qmm16c (BP-7), both with the D8-NAX drafter; the walk feeds session ids.
 (1) ms per propose (one eval + read-back), BP-0's cycles (1,000 context rows, 8 new + a 16-row block per cycle),
     3 interleaved rounds;
 (2) acceptance: MTP Head's engine-order walk (D7B-1 / BP-1's copy) on ~/Leap/data/dfl2d_{code,math}, cut t 0.3, kept
     per check per arm, paired per window: D7b-NAX vs D7b (same 4-bit weights; only fp32 accumulation order differs),
     with bf16 as the reference.
usage: [BigMoE venv] bp_d7b_nax_check.py -> ~/Leap/data/bp_d7b_nax_check.txt (GPU, MEM ~3.5 GB; share lock)"""
import os, sys, glob, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_third_party"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_scripts"))
import dflash_model_mlx as DF
import draft_fused as DFU
from chain_fused import chain_fused
import d7b_live
import d7b_nax

mx.set_cache_limit(256 << 20)
T_CUT, R, K = 0.3, 16, 16
OUT = os.path.expanduser("~/Leap/data/bp_sub_check.txt")
def say(s=""):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
t00 = time.time()


class Target(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(V, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, V, bias=False, group_size=64, bits=4)


def load(q4):
    d_ = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
    d_.load_weights(os.path.expanduser("~/Leap/data/l120_ft.safetensors"), strict=False)
    if q4:
        d7b_live.apply(d_)
    mx.eval(d_.parameters())
    return d_
d8 = load(False)
N8 = d7b_nax.apply8(d8)
d16 = d4 = d8
w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
tgt = Target(d16.config.vocab_size)
tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                 + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
del w
mx.eval(tgt.parameters())
for d_ in (d8,):
    d_.bind(tgt)
    mx.eval(d_.parameters())
DFU.install(forms=False, draft=d8)                      # R-76 conv kernel + mask memo; no forms on a quantized drafter
from qmm16 import qmm16c
HEADC = [False]
MASK, SW = int(d16.config.mask_token_id), d16.config.sliding_window
ARMS = {"D8 + head c": (d8, "c"), "D8 + sub c": (d8, "s")}
SUB = d7b_nax.DraftSubset(tgt.lm_head)
SUBC = [False]
NINST = [0]


def arm(name):
    dr, nax = ARMS[name]
    HEADC[0] = nax == "c"
    SUBC[0] = nax == "s"
    return dr


def logits(dr, hid):
    if not HEADC[0]:
        return dr.compute_logits(hid)
    lm = dr.lm_head
    lg = qmm16c(hid.reshape(-1, hid.shape[-1]), lm["weight"], lm["scales"], lm["biases"]).reshape(*hid.shape[:-1], -1)
    lg = lg * dr.config.output_multiplier
    cap = dr.config.final_logit_softcapping
    if cap is not None and cap > 0:
        lg = mx.tanh(lg / cap) * cap
    return lg


def head2s(dr, hid):
    if SUBC[0]:
        return d7b_nax.screened_head2s(dr, hid, SUB)
    lg = logits(dr, hid)
    B, s = 128, lg.shape
    blk = lg.reshape(*s[:-1], s[-1] // B, B)
    tb = mx.argpartition(blk.max(axis=-1), -K, axis=-1)[..., -K:]
    sel = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K, B)), axis=-2).reshape(*s[:-1], K * B)
    j = mx.argpartition(sel, -K, axis=-1)[..., -K:]
    c = mx.take_along_axis(tb, j // B, axis=-1) * B + j % B
    return c, mx.take_along_axis(lg, c, axis=-1)


def propose(dr, blk, ctx, dcache):
    hid = dr.hidden_states(blk, ctx, dcache, 1)
    cand, unary = head2s(dr, hid)
    hp = dr.candidate_selector.hidden_projection(hid)
    return chain_fused(blk, unary, cand, hp, dr.candidate_selector)


def timing(name, n=30, seed=7):
    dr = arm(name)
    rng = np.random.default_rng(seed)
    dc = dr.make_cache()
    ctx0 = mx.array((rng.standard_normal((1000, 5 * 2560)) * 0.5).astype(np.float16))
    mx.eval(dr.hidden_states(mx.array([[1] + [MASK] * (R - 1)]), ctx0[None].astype(mx.bfloat16), dc, 1))
    ts = []
    for c in range(n):
        a = dc[0].offset + 8
        ctx = mx.array((rng.standard_normal((8, 5 * 2560)) * 0.5).astype(np.float16)).astype(mx.bfloat16)[None]
        blk = mx.array([[int(rng.integers(0, 248000))] + [MASK] * (R - 1)])
        mx.eval(ctx, blk)
        t0 = time.perf_counter()
        pth, cf = propose(dr, blk, ctx, dc)
        np.array(pth); np.array(cf)
        ts.append(time.perf_counter() - t0)
        if (trim := dc[0].offset - a) > 0:
            DF._trim_recent_cache(dc, trim)
    return float(np.median(ts[4:]))


def walk(name, folder):
    dr = arm(name)
    ks, per = [], []
    for p in sorted(glob.glob(os.path.expanduser(f"~/Leap/data/{folder}/*.npz"))):
        z = np.load(p)
        f, ids, off, gen = z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])
        dcache = dr.make_cache()
        if SUBC[0]:
            SUB.reset(); SUB.add(ids[:max(0, gen - off)])
        start = max(off, gen - (SW - 1))
        for c in dcache:
            c.offset = start
        fed, a, kw = start, gen, []
        while a < off + len(ids) - 1:
            ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
            pth, cf = propose(dr, mx.array([[int(ids[a - off])] + [MASK] * (R - 1)]), ctx, dcache)
            mx.eval(pth, cf)
            path, cfl = np.array(pth[0]).tolist(), np.array(cf[0].astype(mx.float32)).tolist()
            c_ = int((np.cumprod(np.array(cfl)) >= T_CUT).sum())
            fed = a
            true = ids[a - off + 1:a - off + 1 + c_]
            k = 0
            while k < len(true) and path[k] == true[k]:
                k += 1
            kw.append(k + 1)
            if SUBC[0]:
                SUB.add(ids[a - off + 1:a - off + k + 2])
            if a + k + 1 - off >= len(ids):
                break
            a = a + k + 1
        ks += kw; per.append((sum(kw), len(kw)))
    return float(np.mean(ks)), np.array(per, float)


say(f"\nBP-9 {time.strftime('%a %H:%M:%S')}: D8-NAX ({N8} NaxQL8): full head on qmm16c vs DraftSubset (98,304 + session) on qmm16c; loaded in {time.time() - t00:.0f} s")
res = {k: [] for k in ARMS}
for rnd in range(3):
    for name in (list(ARMS) if rnd % 2 == 0 else list(ARMS)[::-1]):
        res[name].append(timing(name, seed=7 + rnd))
say("(1) ms per propose (median per round; 3 interleaved rounds): " + ", ".join(
    f"{k} {[round(1e3 * v, 2) for v in vs]}" for k, vs in res.items()))
c0 = d7b_nax.COUNTS.copy()
rng = np.random.default_rng(0)
for folder in ("dfl2d_code", "dfl2d_math"):
    out = {k: walk(k, folder) for k in ARMS}
    base, nax = out["D8 + head c"][1], out["D8 + sub c"][1]
    bs = [(nax[i, 0].sum() / nax[i, 1].sum()) / (base[i, 0].sum() / base[i, 1].sum())
          for i in (rng.integers(0, len(base), len(base)) for _ in range(2000))]
    say(f"(2) {folder}: kept per check " + ", ".join(f"{k} {v[0]:.3f}" for k, v in out.items())
        + f"; sub / full x{out['D8 + sub c'][0] / out['D8 + head c'][0]:.4f} (90 % CI x{np.percentile(bs, 5):.3f}-x{np.percentile(bs, 95):.3f}); "
        f"lower in {int(np.sum(nax[:, 0] / nax[:, 1] < base[:, 0] / base[:, 1]))}, higher in {int(np.sum(nax[:, 0] / nax[:, 1] > base[:, 0] / base[:, 1]))} of {len(base)}; "
        "")
say(f"  NaxQL modules {NINST[0]}; qmm16c calls {d7b_nax.COUNTS['nax']}, MLX calls {d7b_nax.COUNTS['mlx']} (NaxQL); mlx peak {mx.get_peak_memory() / 2**30:.2f} GB; {time.time() - t00:.0f} s")
