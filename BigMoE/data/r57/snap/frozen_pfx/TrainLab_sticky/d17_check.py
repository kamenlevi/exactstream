"""D17-1 (MTP Head, Wed 11-04 15:4x; BigMoE's ask for R57_D17 + QOM): the 17-position drafter pass vs today's 16 on the
engine's K83 path (forms open 16..64 rows, Leap 05fafe7). Per check in the engine order (16-position chain as today): both
passes on copies of the same drafter cache; drafts 1-15 and their cf compared (bit for bit, and token agreement); the 16th
draft's accuracy where all 15 checked drafts are right (a full block: QOM's gate on row 15, DA's full-block anchor guess).
Output: data/d17_check.txt (appended).

Copied setup from DA-1:'s offline same-drafts check (MTP Head, Wed 11-04; the orchestrator's item 2 'minimal test'; pre-registered
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
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_third_party"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/TrainLab_sticky"))
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


say(f"D17-1 {time.strftime('%m-%d %H:%M')}: 17- vs 16-position drafter pass, K83 path (L1-20 bf16), forms default "
    f"{getattr(DFU, 'FORM_ROWS', '16..64?')}, code windows ~/Leap/data/dfl2d_code (first {N_WIN})")
C = dict(n=0, bit=0, tok15=0, tokall=0, full=0, d16=0, cut_same=0)
t16, t17 = [], []
for p in sorted(glob.glob(os.path.expanduser("~/Leap/data/dfl2d_code/*.npz")))[:N_WIN]:
    z = np.load(p)
    f, ids, off, gen = z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])
    dcache = draft.make_cache()
    start = max(off, gen - (SW - 1))
    for c in dcache:
        c.offset = start
    fed, a = start, gen
    while a < off + len(ids) - 1:
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        tok = int(ids[a - off])
        dc17 = DA.clone_cache(dcache)
        t0 = time.perf_counter()
        p16, c16, *_ = propose_x(mx.array([[tok] + [MASK] * 15]), ctx, dcache)
        mx.eval(p16, c16); t16.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        p17, c17, *_ = propose_x(mx.array([[tok] + [MASK] * 16]), ctx, dc17)
        mx.eval(p17, c17); t17.append(time.perf_counter() - t0)
        a16, a17 = np.array(p16[0]), np.array(p17[0])
        f16, f17 = np.array(c16[0].astype(mx.float32)), np.array(c17[0].astype(mx.float32))
        C["n"] += 1
        C["bit"] += int(np.array_equal(a16, a17[:15]) and np.array_equal(f16.view(np.uint32), f17[:15].view(np.uint32)))
        C["tok15"] += int(np.array_equal(a16, a17[:15])); C["tokall"] += int(np.sum(a16 == a17[:15]))
        cut16 = int((np.cumprod(f16) >= 0.3).sum()); cut17 = int((np.cumprod(f17[:15]) >= 0.3).sum())
        C["cut_same"] += int(cut16 == cut17)
        fed = a
        true = ids[a - off + 1:a - off + 1 + cut16]
        path = a16.tolist()
        k = 0
        while k < len(true) and path[k] == true[k]:
            k += 1
        if cut16 == 15 and k == 15 and a + 16 - off < len(ids):      # a full block, all right: the 16th draft vs the 125B
            C["full"] += 1; C["d16"] += int(a17[15] == ids[a + 16 - off])
        if a + k + 1 - off >= len(ids):
            break
        a = a + k + 1
n = C["n"]
say(f"  {n} checks: drafts 1-15 bit for bit equal (tokens and cf) on {100 * C['bit'] / n:.1f} %, same 15 tokens on "
    f"{100 * C['tok15'] / n:.1f} %, per-draft token agreement {100 * C['tokall'] / (15 * n):.2f} %; the t 0.3 cut lands at the "
    f"same depth on {100 * C['cut_same'] / n:.1f} %")
say(f"  full blocks (15 checked drafts, all right): {C['full']} ({100 * C['full'] / n:.1f} % of checks); the 16th draft = the "
    f"125B's token on {100 * C['d16'] / max(1, C['full']):.1f} % of them")
say(f"  ms: 16-position pass median {1e3 * np.median(t16):.2f}, 17-position {1e3 * np.median(t17):.2f}; mlx peak "
    f"{mx.get_peak_memory() / 2**30:.2f} GB; {time.time() - t00:.0f} s")
open(os.path.expanduser("~/TrainLab/data/d17_check.txt"), "a").write("\n".join(lines) + "\n\n")
