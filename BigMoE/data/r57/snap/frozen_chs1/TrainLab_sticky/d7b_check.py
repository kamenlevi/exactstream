"""D7B-1 (MTP Head, Wed 11-04 15:5x; the orchestrator's D7b discrepancy: BigMoE's live all-in arms at C 128 saw R57_D7B=code
cut code kept 10.35 -> 9.91, Leap's R-85b replay said +0.00 %): bf16 L1-20 (forms on) vs D7b arm E (d7b_live.apply: the 5
layers 4-bit RTN g64, no forms), both on draft_fused's conv kernel + mask memo and chain_fused, as r57_screen.py wires
D7B=code; the engine-order walk with the t 0.3 cut on ~/Leap/data/dfl2d_{code,math}; paired per window.
Output: data/d7b_check.txt (appended).

Setup copied from DA-1:'s offline same-drafts check (MTP Head, Wed 11-04; the orchestrator's item 2 'minimal test'; pre-registered
in REPORT 'DA-1'): the engine's propose path (r57_screen.py K83: draft_fused forms + chain_fused, head2s K 16; L1-20 bf16)
replayed on the walk's code windows (~/Leap/data/dfl2d_code: the 125B's taps f and tokens), in the engine's order: propose ->
cut at t 0.3 -> check against the 125B's tokens -> commit (k, next anchor) -> next propose on the taps of rows 0..k.
After each check's draft, sticky/draft_ahead.py pre-drafts the planned prefixes on cloned caches. At the next check, on a
hit, the pre-drafted block is compared with the serial one (path and cf bitwise, and the drafter cache state), and the serial
cache is checked untouched by the pre-drafts. Reports: hit rate (3 prefixes; + 'end2'), identical on hits (must be 100 %),
cache untouched (must be 100 %), ms per propose. env: DA_WIN (24 windows). Output: data/da_check.txt (appended)."""
import os, sys, glob, time, resource
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_third_party"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/TrainLab_sticky"))
import dflash_model_mlx as DF
import draft_fused as DFU
from chain_fused import chain_fused
import draft_ahead as DA

mx.set_cache_limit(256 << 20)
T_CUT, R, K = 0.3, 16, 16
N_WIN = int(os.environ.get("DA_WIN", "24"))
OUT = os.path.expanduser("~/TrainLab/data/da_check.txt")
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


t00 = time.time()
draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.load_weights(os.path.expanduser("~/Leap/data/l120_ft.safetensors"), strict=False)
mx.eval(draft.parameters())
V = draft.config.vocab_size


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
d4 = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
d4.load_weights(os.path.expanduser("~/Leap/data/l120_ft.safetensors"), strict=False)
import d7b_live
d7b_live.apply(d4, group_size=int(os.environ.get("D7B_GS", "64")), bits=int(os.environ.get("D7B_BITS", "4")))
d4.bind(tgt)
mx.eval(d4.parameters())
DFU.install(forms=True, draft=draft)                    # as r57_screen.py with D7B=code: forms on the bf16 drafter only
MASK, SW = int(draft.config.mask_token_id), draft.config.sliding_window


def head2s(hid):                                        # r57_screen.py head2s, verbatim (both drafters share the 4-bit head)
    lg = draft.compute_logits(hid)
    B, s = 128, lg.shape
    blk = lg.reshape(*s[:-1], s[-1] // B, B)
    tb = mx.argpartition(blk.max(axis=-1), -K, axis=-1)[..., -K:]
    sel = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K, B)), axis=-2).reshape(*s[:-1], K * B)
    j = mx.argpartition(sel, -K, axis=-1)[..., -K:]
    c = mx.take_along_axis(tb, j // B, axis=-1) * B + j % B
    return c, mx.take_along_axis(lg, c, axis=-1)


def propose_x(blk, ctx, dcache, draft=draft):            # r57_screen.py propose (K83)
    hid = draft.hidden_states(blk, ctx, dcache, 1)
    cand, unary = head2s(hid)
    hp = draft.candidate_selector.hidden_projection(hid)
    pth, cf = chain_fused(blk, unary, cand, hp, draft.candidate_selector)
    return pth, cf, cand, unary, hp


def state(dc):                                          # a cache's state, bit for bit
    return [(c.offset, getattr(c, "_idx", None),
             None if c.keys is None else np.array(c.keys.view(mx.uint16)),
             None if c.values is None else np.array(c.values.view(mx.uint16))) for c in dc]


def same(s1, s2):
    return all(a[0] == b[0] and a[1] == b[1] and all((x is None and y is None) or (x is not None and y is not None and
               x.shape == y.shape and np.array_equal(x, y)) for x, y in ((a[2], b[2]), (a[3], b[3]))) for a, b in zip(s1, s2))




def walk(dr_, folder):
    ks, per = [], []
    for p in sorted(glob.glob(os.path.expanduser(f"~/Leap/data/{folder}/*.npz"))):
        z = np.load(p)
        f, ids, off, gen = z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])
        dcache = dr_.make_cache()
        start = max(off, gen - (SW - 1))
        for c in dcache:
            c.offset = start
        fed, a, kw = start, gen, []
        while a < off + len(ids) - 1:
            ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
            pth, cf, *_ = propose_x(mx.array([[int(ids[a - off])] + [MASK] * (R - 1)]), ctx, dcache, dr_)
            mx.eval(pth, cf)
            path, cfl = np.array(pth[0]).tolist(), np.array(cf[0].astype(mx.float32)).tolist()
            c_ = int((np.cumprod(np.array(cfl)) >= T_CUT).sum())
            fed = a
            true = ids[a - off + 1:a - off + 1 + c_]
            k = 0
            while k < len(true) and path[k] == true[k]:
                k += 1
            kw.append(k + 1)
            if a + k + 1 - off >= len(ids):
                break
            a = a + k + 1
        ks += kw; per.append((sum(kw), len(kw)))
    return np.mean(ks), per


say(f"D7B-1 {time.strftime('%m-%d %H:%M')}: bf16 L1-20 (forms) vs D7b arm E ({os.environ.get('D7B_BITS', '4')}-bit RTN g{os.environ.get('D7B_GS', '64')} layers, no forms), engine-order walk, "
    f"cut t {T_CUT}, block {R}; loaded in {time.time() - t00:.0f} s")
rng = np.random.default_rng(0)
for folder in ("dfl2d_code", "dfl2d_math"):
    k16, p16 = walk(draft, folder)
    k4, p4 = walk(d4, folder)
    a16, a4 = np.array(p16, float), np.array(p4, float)       # per window: tokens kept, checks
    bs = [ (a4[i, 0].sum() / a4[i, 1].sum()) / (a16[i, 0].sum() / a16[i, 1].sum())
           for i in (rng.integers(0, len(a16), len(a16)) for _ in range(2000)) ]
    say(f"  {folder}: {len(p16)} windows; kept per check bf16 {k16:.2f}, D7b {k4:.2f} -> x{k4 / k16:.4f} "
        f"(90 % CI over windows x{np.percentile(bs, 5):.3f}-x{np.percentile(bs, 95):.3f}); tokens per window paired, "
        f"D7b lower in {int(np.sum(a4[:, 0] / a4[:, 1] < a16[:, 0] / a16[:, 1]))} of {len(p16)}")
say(f"  mlx peak {mx.get_peak_memory() / 2**30:.2f} GB; {time.time() - t00:.0f} s")
open(os.path.expanduser("~/TrainLab/data/d7b_check.txt"), "a").write("\n".join(lines) + "\n\n")
