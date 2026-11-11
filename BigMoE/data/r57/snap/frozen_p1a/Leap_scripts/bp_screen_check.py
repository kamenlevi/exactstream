"""BP-1 (Leap, Wed 2026-11-04 16:2x; TABLE_350 'BP-1'): DFlash2's draft head screened to FR-Spec's top-32K tokens + the
session's own tokens (screened_head.ScreenedHead: 2.15 -> 0.45 ms at 16 rows, BP-0), against the full 248K head.
Drafts are verified, so the output stays exact; the question is acceptance.
Replay = MTP Head's D7B-1 / DA-1 engine-order walk (/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/TrainLab_sticky/d7b_check.py, copied): the bf16 drafter with
r57_screen's K83 wiring (draft_fused forms + chain_fused, head2s K 16), ~/Leap/data/dfl2d_{code,math} windows (the 125B's
taps + tokens): propose -> cut at t 0.3 -> check against the 125B's tokens -> commit -> next propose. The screened arm
adds the window's prompt ids at its start and each check's committed ids after it (what the engine knows).
Output: kept per check per arm, paired per window (bootstrap 90 % CI of the ratio), and ms per propose.
usage: [BigMoE venv] bp_screen_check.py -> ~/Leap/data/bp_screen_check.txt (GPU, MEM ~3.5 GB; share lock)"""
import os, sys, glob, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_third_party"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_scripts"))
import dflash_model_mlx as DF
import draft_fused as DFU
from chain_fused import chain_fused
from screened_head import ScreenedHead

mx.set_cache_limit(256 << 20)
T_CUT, R, K = 0.3, 16, 16
OUT = os.path.expanduser("~/Leap/data/bp_screen_check.txt")
def say(s=""):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
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
DFU.install(forms=True, draft=draft)                    # r57_screen K83: forms on the bf16 drafter
MASK, SW = int(draft.config.mask_token_id), draft.config.sliding_window
NBASE = int(os.environ.get("BP_BASE", "32768"))
RANK = np.load(os.path.expanduser("~/Leap/data/frspec_rank.npy"))[:NBASE]
OM = draft.config.output_multiplier
CAP = draft.config.final_logit_softcapping


def topk2(lg):                                          # r57_screen.py head2s's two-stage top-K, verbatim
    B, s = 128, lg.shape
    blk = lg.reshape(*s[:-1], s[-1] // B, B)
    tb = mx.argpartition(blk.max(axis=-1), -K, axis=-1)[..., -K:]
    sel = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K, B)), axis=-2).reshape(*s[:-1], K * B)
    j = mx.argpartition(sel, -K, axis=-1)[..., -K:]
    c = mx.take_along_axis(tb, j // B, axis=-1) * B + j % B
    return c, mx.take_along_axis(lg, c, axis=-1)


def head2s(hid, sh=None):
    if sh is None:
        return topk2(draft.compute_logits(hid))
    lg = sh.logits(hid).astype(hid.dtype).reshape(*hid.shape[:-1], -1) * OM   # compute_logits on the subset rows: the
    #                                                    lm_head's bf16 outputs (exact round trip via fp32), same ops
    if CAP is not None and CAP > 0:
        lg = mx.tanh(lg / CAP) * CAP
    c, u = topk2(lg)
    ids = mx.concatenate([sh.ids, sh.xids]) if sh.n_extra else sh.ids
    return mx.take(ids, c), u


def propose(blk, ctx, dcache, sh=None):
    hid = draft.hidden_states(blk, ctx, dcache, 1)
    cand, unary = head2s(hid, sh)
    hp = draft.candidate_selector.hidden_projection(hid)
    return chain_fused(blk, unary, cand, hp, draft.candidate_selector)


def walk(screened, folder):
    ks, per, ts = [], [], []
    for p in sorted(glob.glob(os.path.expanduser(f"~/Leap/data/{folder}/*.npz"))):
        z = np.load(p)
        f, ids, off, gen = z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])
        sh = None
        if screened:
            sh = ScreenedHead(tgt.lm_head, RANK, extra_cap=4096)
            sh.add(ids[:max(0, gen - off)].tolist())
        dcache = draft.make_cache()
        start = max(off, gen - (SW - 1))
        for c in dcache:
            c.offset = start
        fed, a, kw = start, gen, []
        while a < off + len(ids) - 1:
            ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
            t0 = time.perf_counter()
            pth, cf = propose(mx.array([[int(ids[a - off])] + [MASK] * (R - 1)]), ctx, dcache, sh)
            mx.eval(pth, cf)
            path, cfl = np.array(pth[0]).tolist(), np.array(cf[0].astype(mx.float32)).tolist()
            ts.append(time.perf_counter() - t0)
            c_ = int((np.cumprod(np.array(cfl)) >= T_CUT).sum())
            fed = a
            true = ids[a - off + 1:a - off + 1 + c_]
            k = 0
            while k < len(true) and path[k] == true[k]:
                k += 1
            kw.append(k + 1)
            if sh is not None:                           # the committed ids: the k accepted + the target's next
                sh.add(ids[a - off + 1:a - off + k + 2].tolist())
            if a + k + 1 - off >= len(ids):
                break
            a = a + k + 1
        ks += kw; per.append((sum(kw), len(kw)))
    return np.mean(ks), per, float(np.median(ts[5:]))


say(f"\nBP-1 {time.strftime('%a %H:%M:%S')}: full vs screened draft head (FR-Spec top-{NBASE} + session, cap 4096), engine-order walk, "
    f"cut t {T_CUT}, block {R}; loaded in {time.time() - t00:.0f} s")
rng = np.random.default_rng(0)
for folder in ("dfl2d_code", "dfl2d_math"):
    kf, pf, tf = walk(False, folder)
    ks_, ps, tsc = walk(True, folder)
    af, as_ = np.array(pf, float), np.array(ps, float)
    bs = [(as_[i, 0].sum() / as_[i, 1].sum()) / (af[i, 0].sum() / af[i, 1].sum())
          for i in (rng.integers(0, len(af), len(af)) for _ in range(2000))]
    say(f"  {folder}: {len(pf)} windows; kept per check full {kf:.3f}, screened {ks_:.3f} -> x{ks_ / kf:.4f} "
        f"(90 % CI over windows x{np.percentile(bs, 5):.3f}-x{np.percentile(bs, 95):.3f}); screened lower in "
        f"{int(np.sum(as_[:, 0] / as_[:, 1] < af[:, 0] / af[:, 1]))}, higher in {int(np.sum(as_[:, 0] / as_[:, 1] > af[:, 0] / af[:, 1]))} "
        f"of {len(pf)}; ms per propose (median, one eval + read-back) full {1e3 * tf:.2f}, screened {1e3 * tsc:.2f}")
say(f"  mlx peak {mx.get_peak_memory() / 2**30:.2f} GB; {time.time() - t00:.0f} s")
