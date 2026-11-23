"""P1 OVERLAP (WEEK98 pillar 3; Leap, Sun 15 Nov 2026 01:3x; data/week98/plan_1115.md): does the MTP chain head's draft
hide behind a check on the same GPU? kappa = (check + chain together - check alone) / chain alone.
Pre-registered (plan_1115.md, P1, written Sun 00:4x before any run):
  kappa < 0.25          -> prototype the draft-ahead loop in a copy (P1b);
  0.25 <= kappa < 0.52  -> park (<= +5 % today);
  kappa >= 0.52 (= q at a flat 98 %) -> pipelining on one GPU is closed (it loses); drafting must get cheaper itself.
  Prior (NAX-OCC: two NAX6 calls on two streams only 1.2-2.4 % faster than serial at S 32; R-11 D: a second-stream
  draft hid nothing): kappa ~0.8-1.0, KILL 65 %.
Check proxy: 48 chained "layers", each = a share of one check's dense set (D32's 17 shapes at their per-check counts,
random weights, one distinct weight per call so the bytes come from DRAM: 4-bit big shapes as qmm16n at 16 rows / D32's
mr2 at 32 rows, small 4-bit, 8-bit and bf16 shapes as MLX calls) + one NAX6 expert call on three random 512-expert pools
in NAX6's tier layout (4433333333), with real top-10 routing of layers 6 / 20 / 40 from the code / chat / math traces
(ane_probe's step-2 proxy; its records3 loader no longer matches the store, 01:07). S = 16 and 32 rows.
Chain: the 125B's MTP head built standalone as mtp_step_profile.py builds it (Qwen4ExpMTP + 4-bit embed + FR-Spec 32K
screened head, 400-position QSA cache), 32 steps, one sync per step (as the engine's chain runs), on GPU stream 2.
Arms per round, order rotated: A check alone; B chain alone; C check sent (async_eval) then the chain on stream 2, then
the check waited. kappa = (C - A) / B per round; medians over rounds. Also the chain's own time inside C vs B, and the
tokens of the chain on stream 2 == on the default stream (sanity).
usage: [venv] p1_overlap.py [smoke] (qscreen.sh puts stdout in data/week98/screens/p1_overlap.txt); MEM ~14 GB"""
import os, sys, json, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98a/Leap_scripts"))
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
t00 = time.time()


def say(s):
    print(s, flush=True)


say(f"P1 OVERLAP {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}{' (SMOKE)' if SMOKE else ''}")

# ---- the chain: the MTP head, standalone (mtp_step_profile.py's build)
from mtplx.models import qwen4_exp as Q
from mlx.utils import tree_flatten
from screened_head import ScreenedHead
D = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(f"{D}/config.json"))
args = Q.TextArgs.from_dict(cfg.get("text_config", cfg))
mtp = Q.Qwen4ExpMTP(args)
raw = mx.load(f"{D}/mtp.safetensors")
st = {k[len("mtp."):]: v for k, v in raw.items() if k.startswith("mtp.")}
for n in ("pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"):
    if n in st and st[n].ndim == 1:
        st[n] = (st[n].astype(mx.float32) + 1.0).astype(st[n].dtype)
module_in = {p[:-len(".weight")]: int(a.shape[-1]) for p, a in tree_flatten(mtp.parameters()) if p.endswith(".weight") and a.ndim >= 2}
qmap = {}
for mod, in_f in module_in.items():
    w, s = st.get(f"{mod}.weight"), st.get(f"{mod}.scales")
    if w is None or s is None or w.dtype != mx.uint32:
        continue
    qmap[mod] = {"bits": int(w.shape[-1]) * 32 // in_f, "group_size": in_f // int(s.shape[-1])}
nn.quantize(mtp, group_size=64, bits=8, class_predicate=lambda p, m: qmap.get(p, False))
mtp.load_weights(list(st.items()), strict=True)
mtp.eval()
mx.eval(mtp.parameters())
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
def get(k):
    return mx.load(f"{D}/{wmap[k]}")[k]
V = args.vocab_size
emb = nn.QuantizedEmbedding(V, args.hidden_size, group_size=64, bits=4)
head = nn.QuantizedLinear(args.hidden_size, V, bias=False, group_size=64, bits=4)
for m, p in ((emb, "language_model.model.embed_tokens"), (head, "language_model.lm_head")):
    m.weight, m.scales, m.biases = get(p + ".weight"), get(p + ".scales"), get(p + ".biases")
mx.eval(emb.parameters(), head.parameters())
sh = ScreenedHead(head, np.load(os.path.expanduser("~/Leap/data/frspec_rank.npy"))[:32768], extra_cap=4096)
HCW = args.hc_count * args.hidden_size
rng = np.random.default_rng(5)
cache = [Q.QSACache(args.indexer_compress_ratio or 4)]
NH = 400
Wh = mx.array((rng.standard_normal((1, NH, HCW)) * 0.6).astype(np.float32)).astype(mx.bfloat16)
ids0 = mx.array(rng.integers(0, 200000, (1, NH)).astype(np.int32))
mx.eval(mtp.fuse_and_run(Wh, emb(ids0), cache))
mx.random.seed(0)
h0 = (mx.random.normal((1, 1, HCW)) * 0.6).astype(mx.bfloat16)
mx.eval(h0)
allids = sh.ids
NSTEP = 32


def step(h, tok):
    hh = mtp.fuse_and_run(h, emb(mx.array([[tok]])), cache)
    lg = sh.logits(mtp.hyper_connection_mixer(hh))[0]
    j = mx.argmax(lg)
    return hh, mx.take(allids, j), mx.exp(lg[j] - mx.logsumexp(lg))


def chain(stream=None, tok0=1000):
    """32 steps, one sync per step; returns (seconds, tokens)"""
    h, tok, toks = h0, tok0, []
    t0 = time.perf_counter()
    for d in range(NSTEP):
        if stream is None:
            h, tk, p = step(h, tok)
        else:
            with mx.stream(stream):
                h, tk, p = step(h, tok)
        mx.eval(tk, p)
        tok = int(tk.item())
        toks.append(tok)
    dt = time.perf_counter() - t0
    cache[0].trim(NSTEP)
    return dt, toks


say(f"chain built in {time.time() - t00:.0f} s: MTP head {sum(a.nbytes for _, a in tree_flatten(mtp.parameters())) / 1e9:.2f} GB, "
    f"{NSTEP} steps per chain")

# ---- the check proxy
sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/planes_engine"))
import tierglu as TG
import tierglu_nax6 as N6
from qmm26 import qmm16n
import d32
PLAN, RK, R4 = "4433333333", 10, 2
H = 2560
RD = os.path.expanduser("~/BigMoE/models/bare/records3")
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}


def load(layer):                                    # ane_probe / k62p_floor.py's loader (records3 pools, slot = expert id)
    idx = json.load(open(f"{RD}/index.json"))["layers"][str(layer)]
    E, REC = idx["E"], idx["recsz"]
    raw_ = np.fromfile(f"{RD}/{idx['file']}", dtype=np.uint8).reshape(E, REC)
    msb, lsb = {}, {}
    for p_, pr in enumerate(TG.PROJ):
        o, i = SH[pr]
        ow, os_, ob = idx["msb_piece_off"][3 * p_:3 * p_ + 3]
        msb[pr] = {"w3": mx.array(np.ascontiguousarray(raw_[:, ow:ow + o * i * 3 // 8]).view(np.uint32).reshape(E, o, i * 3 // 32)),
                   "s3": mx.array(np.ascontiguousarray(raw_[:, os_:os_ + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16),
                   "b3": mx.array(np.ascontiguousarray(raw_[:, ob:ob + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
        lw, lb = idx["lsb_piece_off"][2 * p_:2 * p_ + 2]
        lsb[pr] = {"l1": mx.array(np.ascontiguousarray(raw_[:, lw:lw + o * i // 8]).view(np.uint32).reshape(E, o, i // 32)),
                   "b": mx.array(np.ascontiguousarray(raw_[:, lb:lb + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
    del raw_
    mx.eval([v for dd in (msb, lsb) for pr in dd for v in dd[pr].values()])
    return msb, lsb


def rand_pool(E=512):
    """random pools in the shapes NAX6 reads (the records3 store changed layout since ane_probe's loader was written,
    01:07 smoke: a reshape error); the bytes and the real routing are what this proxy needs, not the weights' values"""
    msb, lsb = {}, {}
    for pr in TG.PROJ:
        o, i = SH[pr]
        sc = mx.random.uniform(0.001, 0.003, (E, o, i // 64)).astype(mx.bfloat16)
        msb[pr] = {"w3": mx.random.randint(0, 2 ** 31, (E, o, i * 3 // 32)).astype(mx.uint32), "s3": sc,
                   "b3": (-4.0 * sc.astype(mx.float32)).astype(mx.bfloat16)}
        lsb[pr] = {"l1": mx.random.randint(0, 2 ** 31, (E, o, i // 32)).astype(mx.uint32),
                   "b": (mx.random.uniform(-1e-3, 1e-3, (E, o, i // 64))).astype(mx.bfloat16)}
    mx.eval([v for dd in (msb, lsb) for pr in dd for v in dd[pr].values()])
    return msb, lsb


LAY = (6, 20, 40) if not SMOKE else (6,)
POOLS = {l: rand_pool() for l in LAY}
TR = {l: [np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{t}_800_g.npz"))[f"L{l}"].astype(np.int32)
          for t in ("code", "chat", "math")] for l in LAY}
say(f"pools loaded ({len(LAY)} layers) at {time.time() - t00:.0f} s; active {mx.get_active_memory() / 1e9:.1f} GB")

# the dense set of one check, one distinct random weight per call (D32's shapes and counts)
SHAPES = d32.SHAPES if not SMOKE else d32.SHAPES[:3]
DENSE = []                                          # (kind, N, K, weights)
for name, kind, N, K, cnt in SHAPES:
    for c in range(cnt if not SMOKE else 1):
        if kind == "q4" and N % 32 == 0 and N >= 512:
            KG = K // 64
            wn = mx.random.randint(0, 2 ** 31, (N // 32 * KG * 32 * 8,)).astype(mx.uint32)
            s_ = (mx.random.uniform(0.001, 0.003, (N // 32, KG, 32))).astype(mx.bfloat16)
            b_ = (-8 * s_.astype(mx.float32)).astype(mx.bfloat16)
            DENSE.append(("q4n", N, K, (wn, s_, b_)))
        elif kind in ("q4", "q8"):
            bits = 4 if kind == "q4" else 8
            w_ = mx.random.randint(0, 2 ** 31, (N, K * bits // 32)).astype(mx.uint32)
            s_ = mx.random.uniform(0.001, 0.003, (N, K // 64)).astype(mx.bfloat16)
            b_ = (-(2 ** (bits - 1)) * s_.astype(mx.float32)).astype(mx.bfloat16)
            DENSE.append(("mlxq", N, K, (w_, s_, b_, bits)))
        else:
            DENSE.append(("bf16", N, K, ((mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16),)))
    mx.eval([a for d_ in DENSE[-cnt:] for a in d_[3] if isinstance(a, mx.array)])
dense_gb = sum(a.nbytes for d_ in DENSE for a in d_[3] if isinstance(a, mx.array)) / 1e9
say(f"dense set: {len(DENSE)} calls per check, {dense_gb:.2f} GB of distinct weights; active {mx.get_active_memory() / 1e9:.1f} GB")
ORDER = list(range(len(DENSE)))
np.random.default_rng(7).shuffle(ORDER)
PER = [ORDER[c::48] for c in range(48)]              # the dense calls spread over the 48 layers


def dense_call(i, xin, S):
    kind, N, K, W = DENSE[i]
    x_ = xin[:, :K] if K <= H else mx.tile(xin, (1, (K + H - 1) // H))[:, :K]
    if kind == "q4n":
        ks = 1
        if S == 16:
            return qmm16n(x_, *W, N, ks, 1)
        return d32.mm("q4", x_, W, N, ks, 1, 2, 1)
    if kind == "mlxq":
        w_, s_, b_, bits = W
        return mx.quantized_matmul(x_, w_, s_, b_, transpose=True, group_size=64, bits=bits)
    return x_ @ W[0].T


rng2 = np.random.default_rng(2)


def plans(S, n=4):
    out = []
    for _ in range(n):
        pl = []
        for c in range(48):
            l = LAY[c % len(LAY)]
            tr = TR[l][int(rng2.integers(0, len(TR[l])))]
            t0 = int(rng2.integers(0, tr.shape[0] - S))
            ids = tr[t0:t0 + S, :RK]
            pl.append((l, mx.array(ids), mx.array(ids[:, :R4].copy()),
                       mx.softmax(mx.random.normal((S, RK)), axis=-1).astype(mx.bfloat16)))
        out.append(pl)
    mx.eval([a for pl in out for (_, sm, sl, sc) in pl for a in (sm, sl, sc)])
    return out


def check_graph(pl, S, X0, ONES, ONESW):
    x = X0
    for c, (l, sm, sl, sc) in enumerate(pl):
        dep = None
        for i in PER[c]:
            y = dense_call(i, x, S)
            t_ = y[:, :1].astype(mx.float32).sum()
            dep = t_ if dep is None else dep + t_
        msb, lsb = POOLS[l]
        y = N6.tier_glu_nax6(x, msb, lsb, sm, sl, plan=PLAN, grp=N6.gpu_grp(sm, sl, PLAN)).reshape(S * RK, -1)
        x = mx.fast.rms_norm(TG.combine_phase(y, y, ONES, sc), ONESW, 1e-6)
        if dep is not None:
            x = x + (dep * 0.0).astype(x.dtype)
    return x


S2 = mx.new_stream(mx.gpu)
_, ref_toks = chain()
_, s2_toks = chain(S2)
say(f"sanity: the chain on stream 2 gives the default stream's tokens: {ref_toks == s2_toks} ({ref_toks[:6]} ...)")

RES = {}
for S in (16, 32):
    PL = plans(S)
    X0 = (mx.random.normal((S, H)) * 0.5).astype(mx.bfloat16)
    ONES, ONESW = mx.ones((S, RK), mx.int32), mx.ones((H,), mx.bfloat16)
    mx.eval(X0, ONES, ONESW)

    def A(i):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(check_graph(PL[i % 4], S, X0, ONES, ONESW))
        return time.perf_counter() - t0, None

    def B(i):
        mx.synchronize()
        dt, _ = chain(S2, tok0=1000 + i)
        return dt, None

    def C(i):
        mx.synchronize()
        t0 = time.perf_counter()
        out = check_graph(PL[i % 4], S, X0, ONES, ONESW)
        mx.async_eval(out)
        dt_chain, _ = chain(S2, tok0=1000 + i)
        mx.eval(out)
        return time.perf_counter() - t0, dt_chain

    for i in range(3):                               # warm-up
        A(i); B(i); C(i)
    rounds = 10 if not SMOKE else 2
    a_, b_, c_, cc_, kap = [], [], [], [], []
    for r in range(rounds):
        order = [(A, a_), (B, b_), (C, c_)]
        order = order[r % 3:] + order[:r % 3]
        res = {}
        for fn, lst in order:
            v, extra = fn(r)
            lst.append(v)
            res[fn.__name__] = (v, extra)
        cc_.append(res["C"][1])
        kap.append((res["C"][0] - res["A"][0]) / res["B"][0])
    m = lambda xs: 1e3 * float(np.median(xs))
    k_med = float(np.median(kap))
    RES[S] = dict(A=m(a_), B=m(b_), C=m(c_), chain_in_C=m(cc_), kappa=k_med, kappas=[round(k, 3) for k in kap])
    say(f"S {S}: check alone {m(a_):.1f} ms, chain alone {m(b_):.1f} ms ({m(b_) / NSTEP:.3f} per step), together "
        f"{m(c_):.1f} ms (the chain inside it {m(cc_):.1f} ms, x{np.median(cc_) / np.median(b_):.2f}); kappa median "
        f"{k_med:.3f} (rounds {', '.join(f'{k:.2f}' for k in kap)}); serial sum {m(a_) + m(b_):.1f}")

say("")
for S, r in RES.items():
    k = r["kappa"]
    verdict = ("PROTOTYPE (kappa < 0.25)" if k < 0.25 else "PARK (0.25 <= kappa < 0.52)" if k < 0.52
               else "CLOSED on one GPU (kappa >= 0.52 = q: pipelining loses)")
    say(f"P1 verdict at S {S}: kappa {k:.3f} -> {verdict}")
json.dump(RES, open(os.path.expanduser("~/Leap/data/week98/screens/p1_overlap.json"), "w"), indent=1)
say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.1f} GB")
