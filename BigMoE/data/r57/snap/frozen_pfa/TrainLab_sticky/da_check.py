"""DA-1, DRAFT-AHEAD's offline same-drafts check (MTP Head, Wed 11-04; the orchestrator's item 2 'minimal test'; pre-registered
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
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_third_party"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/TrainLab_sticky"))
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
DFU.install(forms=True, draft=draft)                    # the 45 GB default's drafter path (K83, D7B off)
MASK, SW = int(draft.config.mask_token_id), draft.config.sliding_window


def head2s(hid):                                        # r57_screen.py head2s, verbatim
    lg = draft.compute_logits(hid)
    B, s = 128, lg.shape
    blk = lg.reshape(*s[:-1], s[-1] // B, B)
    tb = mx.argpartition(blk.max(axis=-1), -K, axis=-1)[..., -K:]
    sel = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K, B)), axis=-2).reshape(*s[:-1], K * B)
    j = mx.argpartition(sel, -K, axis=-1)[..., -K:]
    c = mx.take_along_axis(tb, j // B, axis=-1) * B + j % B
    return c, mx.take_along_axis(lg, c, axis=-1)


def propose_x(blk, ctx, dcache):                        # r57_screen.py propose (K83) + its candidates for the guesses
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


say(f"DA-1 {time.strftime('%m-%d %H:%M')}: DRAFT-AHEAD offline same-drafts check, the engine's K83 drafter path (L1-20 bf16), "
    f"cut t {T_CUT}, block {R}, code windows ~/Leap/data/dfl2d_code (first {N_WIN}); loaded in {time.time() - t00:.0f} s")
C = dict(checks=0, hit3=0, hit4=0, same=0, diff=0, untouched=0, touched=0, npre=0)
TAGS3 = ("end", "low1", "low2")
tser, tpre, kept = [], [], []
for p in sorted(glob.glob(os.path.expanduser("~/Leap/data/dfl2d_code/*.npz")))[:N_WIN]:
    z = np.load(p)
    f, ids, off, gen = z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])
    dcache = draft.make_cache()
    start = max(off, gen - (SW - 1))
    for c in dcache:
        c.offset = start
    fed, a, pend = start, gen, None
    while a < off + len(ids) - 1:
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        blk = mx.array([[int(ids[a - off])] + [MASK] * (R - 1)])
        t0 = time.perf_counter()
        pth, cf, cand, unary, hp = propose_x(blk, ctx, dcache)
        mx.eval(pth, cf, cand, unary, hp)
        tser.append(time.perf_counter() - t0)
        if pend is not None:                            # a pre-drafted block for exactly this check: must be the serial one
            _, pp, pc, pd = pend
            if (np.array_equal(np.array(pp), np.array(pth)) and np.array_equal(np.array(pc.view(mx.uint32)),
                                                                             np.array(cf.view(mx.uint32)))
                    and same(state(pd), state(dcache))):
                C["same"] += 1
            else:
                C["diff"] += 1
        fed = a
        if (trim := dcache[0].offset - a) > 0:          # as the engine (a no-op here)
            DF._trim_recent_cache(dcache, trim)
        path, cfl = np.array(pth[0]).tolist(), np.array(cf[0]).tolist()
        c_ = int((np.cumprod(np.array(cfl)) >= T_CUT).sum())
        pre = DA.plan(path, cfl, c_, cand, unary, hp, draft.candidate_selector, int(ids[a - off]), seg2=True)
        s0 = state(dcache)
        t1 = time.perf_counter()
        hs = DA.launch(propose_x, lambda k_: mx.array(f[a - off:a - off + k_ + 1][None]).astype(mx.bfloat16), pre, R, MASK,
                       dcache)
        mx.eval(*[x for h in hs for x in h[3:5]])
        tpre.append((time.perf_counter() - t1, len(hs)))
        C["npre"] += len(hs)
        if same(s0, state(dcache)):
            C["untouched"] += 1
        else:
            C["touched"] += 1
        true = ids[a - off + 1:a - off + 1 + c_]
        k = 0
        while k < len(true) and path[k] == true[k]:
            k += 1
        C["checks"] += 1
        kept.append(k + 1)
        if a + k + 1 - off >= len(ids):
            break
        nxt = int(ids[a + k + 1 - off])
        h3, h4 = DA.resolve(hs, k, nxt, TAGS3), DA.resolve(hs, k, nxt)
        C["hit3"] += h3 is not None
        C["hit4"] += h4 is not None
        pend = h4
        a = a + k + 1
n = C["checks"]
ts = 1e3 * np.median(tser)
tp1 = 1e3 * np.median([t / max(1, m) for t, m in tpre if m])
say(f"  {n} checks over {N_WIN} windows, kept {np.mean(kept):.2f} per check; pre-drafts {C['npre'] / n:.2f} per check")
say(f"  HIT: 3 prefixes (end, low1, low2) {100 * C['hit3'] / n:.1f} % of checks; + end2 (2nd-ranked past the cut) "
    f"{100 * C['hit4'] / n:.1f} % (DA-0's replay: 34.2 %)")
say(f"  same drafts on hits: {C['same']} identical, {C['diff']} different (path, cf and drafter cache, bit for bit); "
    f"serial cache untouched by the pre-drafts: {C['untouched']} of {C['untouched'] + C['touched']} checks")
say(f"  ms: serial propose median {ts:.2f}; per pre-draft {tp1:.2f} (sequential here; in the engine they run inside the check). "
    f"If the pre-drafts were free, the drafter's critical-path time falls by the hit rate: {ts:.1f} -> "
    f"{ts * (1 - C['hit3'] / n):.1f} ms (3 prefixes) / {ts * (1 - C['hit4'] / n):.1f} ms (4)")
say(f"  mlx peak {mx.get_peak_memory() / 2**30:.2f} GB, max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB; "
    f"{time.time() - t00:.0f} s in all")
open(OUT, "a").write("\n".join(lines) + "\n\n")
