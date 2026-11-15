"""TREE16 M4 slice loop test (BigMoE, Sun 15 Nov 2026 17:0x): the t16b loop's tree check (scripts/tree16_loop.py: the
same verify / accept / commit / pairs the screen calls) over a MULTI-CHECK decode on a slice, bitwise against REF2's AR.
- R: REF2's AR reference (ref2.ar_step: a 1-row verify + commit keep 1, here with return_hidden), N tokens after a P-token
  prefill; per position the sha1 of its logits row and of its widened stream (the MTP head's input).
- C: a chain control in the loop's own form (a 16-row verify of [tok] + 15 drafts, commit_verified_window keep k + 1):
  drafts = R's next tokens, one made wrong at a random depth (or none). Every kept row's logits / stream == R's.
- T: the tree loop. Per check a random tree (chain, bushy, random; T <= 16, DFS preorder) carrying R's continuation on a
  root-to-node path (the main chain or a side branch), cut at a random depth by a wrong token; siblings never carry the
  token the walk wants, so the walk's path is planned. tree16_loop.verify -> accept -> commit -> pairs; every path node's
  logits / stream == R's at its position, the path == the planned one, the tokens == R's, pairs == R's streams.
  Past the dense budget (a + T > 2044) the check runs as C does (the loop's chain fallback).
Contexts: P 1024 (dense throughout) and P 1990 (crosses 2,044: tree checks right below it, then the fallback; R / C there
also cross 2,048, where QSA selects blocks).
usage: [BigMoE venv] [TREE16_SLICE=0-3|4-7] python scripts/tree16_m4loop.py -> data/deepdraft/tree16/m4loop_<slice>.json"""
import contextlib, glob, hashlib, json, os, sys, time
for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1")):
    os.environ.setdefault(k_, v_)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
import numpy as np
import mlx.core as mx
import mlx_lm.utils as mu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
t00 = time.time()
OUT = os.path.expanduser("~/BigMoE/data/deepdraft/tree16")
SL = os.environ.get("TREE16_SLICE", "0-3")
SA, SB = (int(x) for x in SL.split("-")); SB += 1
N = int(os.environ.get("M4L_N", "200"))
B3 = os.environ.get("M4L_B3") == "1"                    # B3 in Leap's QSA_TREE slot (d6b5884): no fallback past 2,044
PS = [int(x) for x in os.environ.get("M4L_P", "1024,1990").split(",")]


def say(s):
    print(s, flush=True)


MD = os.path.expanduser("~/BigMoE/models/bare")
orig = mu.load_model


def lazy_load_model(path, *a, **kw):
    a = list(a)
    if a:
        a[0] = True
    else:
        kw["lazy"] = True
    model, cfg = orig(path, *a, **kw)
    if hasattr(model, "language_model"):
        inner = model.language_model.model
        inner.layers = inner.layers[SA:SB]
        inner._ple_stage_idx = next((i for i, l in enumerate(inner.layers) if getattr(l, "ple", None) is not None), None)
        mx.eval(model.parameters())
    return model, cfg


mu.load_model = lazy_load_model
try:
    from mtplx.runtime import load
    rt = load(MD, mtp=False)
finally:
    mu.load_model = orig
from mtplx import cache_state as CS
from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind
model = rt.model
inner = model.language_model.model
KINDS = "".join("G" if l.is_linear else "A" for l in inner.layers)
HAS_PLE = inner._ple_stage_idx is not None
import ref2
ref2.apply()
for k_ in ("att_rope", "idx_prep", "hc_write", "hc_gate", "gdn_inproj_x", "gdn_body", "hc_b"):
    ref2.CFG[k_] = True
import tree16_loop as T16L
T16L.install(ref2, ple=HAS_PLE, qsa=B3)
files = sorted(glob.glob(os.path.expanduser("~/Leap/scripts/*.py")))
ids = list(rt.tokenizer.encode("".join(open(f).read() for f in files)))
V = [None]                                               # the logits width, set by the first prefill
say(f"TREE16-M4LOOP {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; slice {SL} ({KINDS}; PLE "
    f"{'at ' + str(inner._ple_stage_idx) if HAS_PLE else 'none'}); B3 {'ON (QSA_TREE slot)' if B3 else 'off'}; P {PS}; N {N}; "
    f"{mx.get_active_memory() / 1e9:.1f} GB active")


@contextlib.contextmanager
def verify_scope():
    with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True), \
            model.verify_capture_scope():
        yield


def states(cache):
    return [c.state for c in cache if hasattr(c, "state")]


def hsh(x):
    return hashlib.sha1(np.array(x.astype(mx.float32)).tobytes()).hexdigest()[:16]


def prefill(P):
    cache = model.make_cache()
    lg = model(mx.array([ids[0:P]]), cache=cache)
    mx.eval(lg, states(cache))
    V[0] = int(lg.shape[-1])
    return cache, int(mx.argmax(lg[0, -1]).item())


def ar_ref(P):
    cache, tok = prefill(P)
    toks, LH, HH = [tok], [], []
    for _ in range(N):
        snap = CS.snapshot_untrimmable_cache_lazy(cache)
        with verify_scope():
            lg, hw = model(mx.array([[tok]]), cache=cache, return_hidden=True)
        model.commit_verified_window(cache, snap.states, keep_tokens=1, verified_tokens=1)
        mx.eval(lg, hw, states(cache))
        LH.append(hsh(lg[0, -1])); HH.append(hsh(hw[0, -1]))
        tok = int(mx.argmax(lg[0, -1]).item())
        toks.append(tok)
    return toks, LH, HH


def wrong(t, rng):
    return int((t + 1 + rng.integers(0, 5000)) % V[0])


def chain_step(cache, a, pos, R, rng, n=15):
    """the loop's chain check with planned drafts; returns (k, new, bad rows)"""
    toks, LH, HH = R
    cut = int(rng.integers(0, n + 1))
    dr = [toks[pos + 1 + j] for j in range(n)]
    if cut < n:
        dr[cut] = wrong(dr[cut], rng)
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    with verify_scope():
        lg = model(mx.array([[toks[pos]] + dr]), cache=cache)
    wv = inner._last_widened
    am = mx.argmax(lg[0], axis=-1)
    mx.eval(am, wv)
    am = am.tolist()
    k = 0
    while k < n and dr[k] == am[k]:
        k += 1
    bad = sum(int(hsh(lg[0, d]) != LH[pos + d]) + int(hsh(wv[0, d]) != HH[pos + d]) for d in range(k + 1))
    model.commit_verified_window(cache, snap.states, keep_tokens=k + 1, verified_tokens=n + 1)
    mx.eval(states(cache))
    new = dr[:k] + [am[k]]
    bad += int(k != cut) + int(new != toks[pos + 1:pos + k + 2])
    return k, new, bad


def dfs_relabel(parents):
    n = len(parents)
    kids = [[] for _ in range(n)]
    for c in range(1, n):
        kids[parents[c]].append(c)
    order, stack = [], [0]
    while stack:
        u = stack.pop()
        order.append(u)
        stack.extend(reversed(kids[u]))
    new = {u: i for i, u in enumerate(order)}
    return [-1] + [new[parents[u]] for u in order[1:]]


def rand_tree(rng):
    s = int(rng.integers(0, 4))
    T = 16 if rng.random() < 0.75 else int(rng.integers(2, 16))
    if s == 0:
        par = [-1] + list(range(T - 1))                                   # a chain
    elif s == 1:
        par = [-1] + [int(rng.integers(0, max(1, c // 3 + 1))) for c in range(1, T)]   # bushy
    else:
        par = [-1] + [int(rng.integers(max(0, c - 4), c)) for c in range(1, T)]       # random, deep-ish
    return dfs_relabel(par)


def tree_step(cache, a, pos, R, rng, S):
    toks, LH, HH = R
    par = rand_tree(rng)
    T = len(par)
    kids = [[] for _ in range(T)]
    for c in range(1, T):
        kids[par[c]].append(c)
    D = 1
    while D < T and par[D] == D - 1:
        D += 1

    def root_path(u):
        p = []
        while u >= 0:
            p.append(u); u = par[u]
        return p[::-1]
    u = D - 1 if rng.random() < 0.4 else int(rng.integers(0, T))        # the main chain's end, or any node
    full = root_path(u)
    plan = full
    if len(full) > 1 and rng.random() < 0.5:
        c = int(rng.integers(1, len(full)))                              # a wrong token at depth c: the walk stops at c - 1
        plan = full[:c]
    want = {plan[d - 1]: toks[pos + d] for d in range(1, len(plan) + 1)}  # node -> the token the walk asks of its children
    for _try in range(50):
        tk = [int(t) for t in rng.integers(1000, V[0] - 1, T)]
        tk[0] = toks[pos]
        for d, nd in enumerate(full[1:], 1):
            tk[nd] = toks[pos + d] if d < len(plan) else wrong(toks[pos + d], rng)
        valid = all(len({tk[c] for c in kids[p_]}) == len(kids[p_]) for p_ in range(T))   # siblings distinct
        valid &= all(tk[plan[d]] == toks[pos + d] for d in range(1, len(plan)))         # the plan carries R's tokens
        valid &= all(tk[c] != w_ for p_, w_ in want.items() for c in kids[p_] if c not in plan)   # off-plan never matches
        if valid:
            break
    else:
        raise AssertionError("no valid token assignment in 50 tries")
    tr = dict(tokens=np.array(tk, np.int32), parents=np.array(par, np.int32), T=T, main_chain_len=D)
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    prev = (list(ids[:SPLIT[0]]) + toks[:pos])[-T16L.PREV:]             # the committed tokens at a - 9 .. a - 1
    with verify_scope():
        lg, hw = T16L.verify(model, cache, tr, a, prev)
    am = mx.argmax(lg, axis=-1)
    mx.eval(am, hw)
    am = am.tolist()
    path, bonus, k, new = T16L.accept(tr, am)
    bad = int(path != plan)
    bad += sum(int(hsh(lg[nd]) != LH[pos + d]) + int(hsh(hw[nd]) != HH[pos + d]) for d, nd in enumerate(path))
    kind = T16L.commit(model, cache, snap.states, path, tr, a, inner, ref2)
    hp, nx = T16L.pairs(hw, tr, path, bonus, len(new))
    mx.eval(hp, states(cache))
    bad += sum(int(hsh(hp[0, d]) != HH[pos + d]) for d in range(len(new)))
    bad += int(new != toks[pos + 1:pos + k + 2]) + int(nx != new) + int(tuple(hp.shape[:2]) != (1, len(new)))
    S["kinds"][kind] += 1
    S["side"] += int(not all(int(n_) == i for i, n_ in enumerate(path)))
    S["T"].append(T); S["k"].append(k)
    return k, new, bad


SPLIT = [0]
R_ALL = {"slice": SL, "kinds": KINDS, "ple": HAS_PLE, "b3": B3, "N": N, "ctx": {}}
ok_all = True
for P in PS:
    SPLIT[0] = P
    t0 = time.time()
    R = ar_ref(P)
    t_r = time.time() - t0
    res = {}
    for mode in ("chain", "tree"):
        rng = np.random.default_rng(23 if mode == "chain" else 29)
        cache, tok0 = prefill(P)
        assert tok0 == R[0][0]
        a, pos, bad, checks, fb = P, 0, 0, 0, 0
        S = {"kinds": {"chain": 0, "tree": 0}, "side": 0, "T": [], "k": []}
        b3c0 = (T16L.TV.CNT.get("b3_live", 0), T16L.TV.CNT.get("b3_dense", 0))
        bad_lo = bad_hi = 0
        while pos + 16 < N:
            if mode == "tree" and T16L.fits(a, 16):
                k, new, b = tree_step(cache, a, pos, R, rng, S)
            else:
                fb += int(mode == "tree")
                k, new, b = chain_step(cache, a, pos, R, rng)
            if a + 16 <= 2048:
                bad_lo += b
            else:
                bad_hi += b
            checks += 1; pos += k + 1; a += k + 1
        res[mode] = dict(checks=checks, bad_rows_dense=bad_lo, bad_rows_past2048=bad_hi, fallback=fb, end=a,
                         **({"kinds": S["kinds"], "side_paths": S["side"], "T_mean": float(np.mean(S["T"])) if S["T"] else None,
                             "k_mean": float(np.mean(S["k"])) if S["k"] else None,
                             "b3_live": T16L.TV.CNT.get("b3_live", 0) - b3c0[0], "b3_dense": T16L.TV.CNT.get("b3_dense", 0) - b3c0[1]}
                            if mode == "tree" else {}))
    ok = res["tree"]["bad_rows_dense"] == 0 and res["chain"]["bad_rows_dense"] == 0
    if B3:                                               # B3 live: the rows past 2,048 count too (tree and chain control)
        ok = ok and res["tree"]["bad_rows_past2048"] == 0 and res["chain"]["bad_rows_past2048"] == 0
    ok_all &= ok
    R_ALL["ctx"][str(P)] = dict(res, ar_s=round(t_r, 1))
    say(f"  P {P}: AR {N} tokens ({t_r:.0f} s); chain control {res['chain']['checks']} checks, bad rows "
        f"{res['chain']['bad_rows_dense']} dense / {res['chain']['bad_rows_past2048']} past 2048; TREE "
        f"{res['tree']['checks']} checks ({res['tree']['kinds']}, side paths {res['tree']['side_paths']}, fallback "
        f"{res['tree']['fallback']}; B3 live / dense {res['tree']['b3_live']} / {res['tree']['b3_dense']}), bad rows {res['tree']['bad_rows_dense']} dense / {res['tree']['bad_rows_past2048']} "
        f"past 2048 -> {'BITWISE' if ok else 'NOT BITWISE'}")
R_ALL.update(verdict=ok_all, wall_s=round(time.time() - t00, 1), peak_gb=round(mx.get_peak_memory() / 1e9, 2),
             prov=T16L.provenance())
os.makedirs(OUT, exist_ok=True)
json.dump(R_ALL, open(os.path.join(OUT, f"m4loop_{SL}{'_b3' if B3 else ''}.json"), "w"), indent=1)
say(f"TREE16-M4LOOP verdict: {'ALL BITWISE' if ok_all else 'NOT BITWISE'} (tree loop == REF2 AR on every kept row, "
    f"{R_ALL['wall_s']:.0f} s, peak {R_ALL['peak_gb']:.1f} GB)")
