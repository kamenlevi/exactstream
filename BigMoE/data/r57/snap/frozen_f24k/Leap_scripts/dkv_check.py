"""DKV check (Leap, Wed 2026-11-04 18:0x; TABLE_350 'DKV'): scripts/dkv.py (copy-free drafter window cache) vs the
reference (mlx_lm RotatingKVCache + draft_fused's memoized-mask attention), on the real D8-NAX drafter.
 (a) stress, one real attention layer (layer 0; NaxQL8 projections): 6 prompts x 121 steps; each prompt a fresh cache pair
     with r57's start offset (0 for short prompts; a random prompt end for full windows), a first update of 2,100 / 2,047 / 1,000 / 300 / 40 / 2 rows, then S = 1 (15 %) or
     2..16; ~8 % of steps a DF._trim_recent_cache of 1-5 rows; odd prompts a small buffer (cap = window + 200: many
     compactions). Every step's output == the reference's, bitwise, and the offsets equal; all paths counted;
 (b) BP-9's walk (dfl2d_code / math, D8-NAX + the screened head): every propose's path and confidences bitwise ==
     (hashed), kept per check equal: dkv, dkv + additive mask;
 (c) ms per propose, BP-0's cycle (8 new rows + a 16-row block) at 1,000 and 2,047 context rows: ref / dkv / dkv+add,
     3 interleaved rounds (GPU shared with Breakthrough's map batch: relative numbers).
Pre-registered: (a) and (b) bitwise everywhere (else a bug); (c) dkv <= ref - 0.25 ms at 2,047 rows -> hand to BigMoE
as an exact lever; else KILL.
usage: [BigMoE venv] dkv_check.py -> ~/Leap/data/dkv_check.txt (GPU, MEM ~3 GB; share lock)"""
import os, sys, glob, time, hashlib
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/Leap_third_party"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/Leap_scripts"))
import dflash_model_mlx as DF
import draft_fused as DFU
from chain_fused import chain_fused
import d7b_nax
import dkv as DKV
from mlx_lm.models.cache import RotatingKVCache

mx.set_cache_limit(256 << 20)
T_CUT, R = 0.3, 16
OUT = os.path.expanduser("~/Leap/data/dkv_check.txt")
def say(s=""):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
t00 = time.time()


class Target(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(V, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, V, bias=False, group_size=64, bits=4)


d8 = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
d8.load_weights(os.path.expanduser("~/Leap/data/l120_ft.safetensors"), strict=False)
N8 = d7b_nax.apply8(d8)
w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
V = d8.config.vocab_size
tgt = Target(V)
tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                 + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
del w
mx.eval(tgt.parameters())
d8.bind(tgt)
mx.eval(d8.parameters())
DFU.install(forms=False, draft=d8)
SUB = d7b_nax.DraftSubset(tgt.lm_head)
MASK, SW = int(d8.config.mask_token_id), d8.config.sliding_window
say(f"\nDKV check {time.strftime('%a %H:%M:%S')}: D8-NAX ({N8} NaxQL8), MLX {mx.__version__}; loaded in {time.time() - t00:.0f} s")

ONLY_C = os.environ.get("DKV_ONLY_C") == "1"           # (21:1x) re-time only: (a) / (b) passed at 19:0x (2f726fb)
# ---- (a) stress on one real attention layer ----
DKV.install()
layer, rope = d8.layers[0].self_attn, d8.rope
rng = np.random.default_rng(0)
rnd = lambda shape: mx.array((rng.standard_normal(shape)).astype(np.float16)).astype(mx.bfloat16)
c0 = dict(DKV.COUNTS)
steps = bad = trims = 0
for pr, S0 in enumerate(() if ONLY_C else (2100, 2047, 1000, 300, 40, 2)):
    ref = RotatingKVCache(SW - 1, 0)
    new = DKV.LinRotCache(SW - 1, 0, cap=(SW - 1 + 200) if pr % 2 else None)
    P0 = S0 + (int(rng.integers(0, 3000)) if S0 >= SW - 1 else 0)    # r57's start: max(0, P - 2047) (deviation 19:0x:
    ref.offset = new.offset = P0 - S0                                 # random preset offsets hit the reference's own crash)
    for S in [S0] + [1 if rng.random() < 0.15 else int(rng.integers(2, 17)) for _ in range(120)]:
        if S == 1 and ref.keys is not None and ref.keys.shape[2] < ref.max_size and ref.offset >= ref.keys.shape[2] \
                and ref.offset > ref.max_size:
            S = 2                                                     # a state where the reference's 1-token update cannot grow
        x, xc = rnd((1, R, 2560)), rnd((1, S, 2560))
        o_r, o_n = layer(x, xc, rope, ref), layer(x, xc, rope, new)
        mx.eval(o_r, o_n)
        bad += int(not (bool(mx.array_equal(o_r.view(mx.uint16), o_n.view(mx.uint16)).item()) and ref.offset == new.offset))
        steps += 1
        if rng.random() < 0.08:
            n = int(rng.integers(1, 6))
            DF._trim_recent_cache([ref], n); DF._trim_recent_cache([new], n)
            trims += 1
dc_ = {k: DKV.COUNTS[k] - c0[k] for k in DKV.COUNTS}
ok_a = None if ONLY_C else (bad == 0 and all(dc_[k] > 0 for k in ("fast", "slow", "adopt", "compact", "trim_fast")))
if not ONLY_C:
    say(f"(a) stress: {steps} steps (6 prompts, {trims} trims): outputs != reference in {bad}; paths taken {dc_} -> "
        f"{'PASS' if ok_a else 'FAIL'}")
DKV.uninstall()


# ---- (b) BP-9's walk, bitwise per propose ----
def propose(blk, ctx, dc):
    hid = d8.hidden_states(blk, ctx, dc, 1)
    cand, unary = d7b_nax.screened_head2s(d8, hid, SUB)
    return chain_fused(blk, unary, cand, d8.candidate_selector.hidden_projection(hid), d8.candidate_selector)


def walk(folder):
    ks, h = [], hashlib.sha256()
    for p in sorted(glob.glob(os.path.expanduser(f"~/Leap/data/{folder}/*.npz"))):
        z = np.load(p)
        f, ids, off, gen = z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])
        dcache = d8.make_cache()
        SUB.reset(); SUB.add(ids[:max(0, gen - off)])
        start = max(off, gen - (SW - 1))
        for c in dcache:
            c.offset = start
        fed, a = start, gen
        while a < off + len(ids) - 1:
            ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
            pth, cf = propose(mx.array([[int(ids[a - off])] + [MASK] * (R - 1)]), ctx, dcache)
            mx.eval(pth, cf)
            path, cfa = np.array(pth[0]), np.array(cf[0].astype(mx.float32))
            h.update(path.tobytes()); h.update(cfa.tobytes())
            c_ = int((np.cumprod(cfa) >= T_CUT).sum())
            fed = a
            true = ids[a - off + 1:a - off + 1 + c_]
            k = 0
            while k < len(true) and path[k] == true[k]:
                k += 1
            ks.append(k + 1)
            SUB.add(ids[a - off + 1:a - off + k + 2])
            if a + k + 1 - off >= len(ids):
                break
            a = a + k + 1
    return float(np.mean(ks)), len(ks), h.hexdigest()[:16]


ARMS = {"ref": None, "dkv": False, "dkv+add": True}
def arm(name):
    DKV.uninstall()
    if ARMS[name] is not None:
        DKV.install(addmask=ARMS[name])


ok_b, lines = (None if ONLY_C else True), []
for folder in (() if ONLY_C else ("dfl2d_code", "dfl2d_math")):
    o = {}
    for name in ARMS:
        arm(name)
        o[name] = walk(folder)
    eq = all(o[k] == o["ref"] for k in ARMS)
    ok_b &= eq
    say(f"(b) {folder}: " + ", ".join(f"{k} kept {v[0]:.3f} / {v[1]} proposes / hash {v[2]}" for k, v in o.items())
        + f" -> {'bitwise ==' if eq else 'DIFFER'}")
arm("ref")


# ---- (c) propose timing, BP-0's cycle ----
def timing(name, ctx_rows, n=30, seed=7):
    arm(name)
    rng_ = np.random.default_rng(seed)
    dc = d8.make_cache()
    ctx0 = mx.array((rng_.standard_normal((ctx_rows, 5 * 2560)) * 0.5).astype(np.float16))
    mx.eval(propose(mx.array([[1] + [MASK] * (R - 1)]), ctx0[None].astype(mx.bfloat16), dc))
    ts = []
    for c in range(n):
        a = dc[0].offset + 8
        ctx = mx.array((rng_.standard_normal((8, 5 * 2560)) * 0.5).astype(np.float16)).astype(mx.bfloat16)[None]
        blk = mx.array([[int(rng_.integers(0, 248000))] + [MASK] * (R - 1)])
        mx.eval(ctx, blk)
        t0 = time.perf_counter()
        pth, cf = propose(blk, ctx, dc)
        np.array(pth); np.array(cf)
        ts.append(time.perf_counter() - t0)
        if (trim := dc[0].offset - a) > 0:
            DF._trim_recent_cache(dc, trim)
    return float(np.median(ts[4:]))


for rows in (1000, 2047):
    res = {k: [] for k in ARMS}
    for rnd_ in range(3):
        for name in (list(ARMS) if rnd_ % 2 == 0 else list(ARMS)[::-1]):
            res[name].append(timing(name, rows, seed=7 + rnd_))
    med = {k: 1e3 * float(np.median(v)) for k, v in res.items()}
    say(f"(c) {rows} context rows: ms per propose (median of 3 interleaved rounds): " + ", ".join(
        f"{k} {v:.2f} ({v - med['ref']:+.2f})" for k, v in med.items()) + f"; rounds {[[round(1e3 * x, 2) for x in v] for v in res.values()]}")
arm("ref")
DKV.uninstall()
say(f"DKV check: (a) {'skipped' if ok_a is None else 'PASS' if ok_a else 'FAIL'}, (b) {'skipped' if ok_b is None else 'PASS' if ok_b else 'FAIL'}; COUNTS {DKV.COUNTS}; mlx peak "
    f"{mx.get_peak_memory() / 2**30:.2f} GB; {time.time() - t00:.0f} s")
