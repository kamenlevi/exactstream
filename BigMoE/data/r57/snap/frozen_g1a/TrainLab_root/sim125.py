"""TrainLab: tau(K) of the 125B MTP head (stock or trained runs) by SIMULATED passes on stored greedy text of the 125B
(NewIdeas 02:30: "no class worse" must also hold on general text). Items: ~/EvalSuite/cache/ref/tf_gen.jsonl (the
125B's own greedy continuations of 138 prompts: GSM8K 60, IFEval 30, HumanEval 24, 8 languages 24), taken round-robin
over the sets so that any prefix is balanced.
One trunk prefill per item (widened stream kept in memory, ~7 MB an item), then every head variant on the same items:
  stock            MTPLX's head as loaded (must come first: the others rewrite the head in place)
  NAME             a trained native run as trained (fp32 projections)
  NAME@q           the same weights stored as the stock head stores them (4-bit g64 attention / shared expert, 8-bit
                   router, bf16 fc): what a deployed head would be; changes smaller than half a 4-bit step round away
  NAME@q8          the same with the 4-bit layers stored at 8 bits g64 (~29 MB more than stock)
  NAME@bf16        the trained projections in bf16 (no quantization; ~80 MB more head weights than stock)
  stock@fp32 / stock@bf16 / stock@q   controls: the stock weights through the same conversions (stock@q ~ stock)
Passes as in Chat1000's protocol: a pass at q drafts a chain from the true pair (W_q, S[q+1]) with the head's cache
holding the true pairs before q; a = leading matches with S[q+2 ..]; the pass emits a + 1 tokens; drafts are trimmed
from the cache and the accepted positions' true pairs appended (= his max_draft=0 + k_mtp chain, walked by passes).
K = drafts per pass (K 15 = a 16-row pass, Chat1000's "mtp 15" row). Ratios vs the first variant are paired (same
items); 90 % intervals by bootstrap over items.
usage: sim125.py [--runs stock,NAME,NAME@q,NAME@bf16] [--K 15] [--per-set 0 (all)] [--budget S]
       -> data/sim125_results.txt
"""
import argparse, itertools, json, os, sys, time
os.environ["TL_INFER"] = "1"                                  # mtp125: keep MTPLX's fused (bit-identical) prefill paths
import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtp125 as M

ITEMS = os.path.expanduser("~/EvalSuite/cache/ref/tf_gen.jsonl")
RES = os.path.join(M.TL, "data/sim125_results.txt")


def widened(lm, ids):
    """Trunk prefill over ids (chunks of M.CHUNK): the widened stream [1, S, Wd] only (no lm_head)."""
    cache = lm.make_cache()
    Ws = []
    for a in range(0, len(ids), M.CHUNK):
        _, W = lm(mx.array([ids[a:a + M.CHUNK]]), cache=cache, return_hidden=True, emit_logits=False)
        Ws.append(W)
        mx.eval(W, [c.state for c in cache if hasattr(c, "state")])
    return mx.concatenate(Ws, 1) if len(Ws) > 1 else Ws[0]


def simulate(lm, W, S, P, K):
    n = len(S)
    q0 = P - 1
    cache = lm.make_mtp_cache()
    if q0 > 0:
        lm.mtp_update_cache(W[:, :q0], mx.array([S[1:q0 + 1]]), mtp_cache=cache)
    q, toks, passes, first = q0, 0, 0, []
    while q + 1 < n - 1:
        logits, h = lm.mtp_forward(W[:, q:q + 1], mx.array([[S[q + 1]]]), mtp_cache=cache, return_hidden=True)
        a, added = 0, 0
        while a < K and q + 2 + a < n:
            d = int(mx.argmax(logits[0, -1]))
            if d != S[q + 2 + a]:
                break
            a += 1
            if a < K and q + 2 + a < n:
                logits, h = lm.mtp_forward(h, mx.array([[d]]), mtp_cache=cache, return_hidden=True)
                added += 1
        if added:
            cache[0].trim(added)
        if a > 0:
            lm.mtp_update_cache(W[:, q + 1:q + 1 + a], mx.array([S[q + 2:q + 2 + a]]), mtp_cache=cache)
        first.append(a > 0)
        toks += a + 1; passes += 1; q += a + 1
    return toks, passes, float(np.mean(first)) if first else 0.0


def trained(name):
    return any(name == t or name.startswith(t + ".") for t in M.TRAIN_MODULES)


def formats(head):
    """How the stock head stores its trained Linear layers: path -> ('q', bits, group) | ('f', dtype)."""
    f = {}
    for name, m in head.named_modules():
        if isinstance(m, nn.QuantizedLinear):
            f[name] = ("q", m.bits, m.group_size)
        elif isinstance(m, nn.Linear):
            f[name] = ("f", m.weight.dtype)
        elif hasattr(m, "gu_weight"):                          # MTPLX's fused gate+up shared expert
            f[name + ".gate_proj"] = f[name + ".up_proj"] = ("q", m.bits, m.group_size)
    return {k: v for k, v in f.items() if trained(k)}


def set_path(root, path, mod):
    *parents, last = path.split(".")
    for p in parents:
        root = root[int(p)] if p.isdigit() else getattr(root, p)
    setattr(root, last, mod)


def to_format(head, fmt, how):
    """The head's trained fp32 Linear layers -> bf16 Linear ('bf16') or each layer's stock format ('q'). For 'q' also
    reports how much of the trained change survives: <dq, d> / <d, d>, d = trained - stock, dq = stored - stock."""
    n, num, den = 0, 0.0, 0.0
    for name, m in list(head.named_modules()):
        if not isinstance(m, nn.Linear) or not trained(name):
            continue
        w = m.weight.astype(mx.bfloat16)
        k = fmt.get(name, ("f", mx.bfloat16))
        if how in ("q", "q8") and k[0] == "q":
            bits = max(8, k[1]) if how == "q8" else k[1]
            new = nn.QuantizedLinear(w.shape[1], w.shape[0], bias=False, group_size=k[2], bits=bits)
            new.weight, new.scales, new.biases = mx.quantize(w, group_size=k[2], bits=bits)
            if name in STOCK:
                d = m.weight - STOCK[name]
                dq = mx.dequantize(new.weight, new.scales, new.biases, group_size=k[2], bits=bits).astype(mx.float32) - STOCK[name]
                num += float((dq * d).sum()); den += float((d * d).sum())
        else:
            new = nn.Linear(w.shape[1], w.shape[0], bias=False)
            new.weight = w
        set_path(head, name, new); n += 1
    head.freeze()
    mx.eval(head.parameters())
    if how in ("q", "q8") and den > 0:
        print(f"  {how}: share of the trained change kept in the quantized layers {num / den:.3f}", flush=True)
    return n


STOCK = {}                                                     # stock head's trained layers, dequantized to fp32


def prepare(head, run, fmt):
    """stock: the head as loaded (first variant only). NAME[@how]: a native run's weights; stock@how: the stock weights
    (snapshot taken at the first rewrite, i.e. before any checkpoint) in the format how."""
    if run == "stock":
        return "as loaded"
    name, _, how = run.partition("@")
    if name != "stock":
        ck = os.path.join(M.TL, "ckpt", name + ".mtp125")
        if not json.load(open(ck + ".json")).get("native"):
            raise SystemExit(f"{name}: not a native run")
    M.native_prepare(head)                                     # fp32 trainable structure (from whatever the head holds)
    lin = lambda: {k: m for k, m in head.named_modules() if isinstance(m, nn.Linear) and trained(k)}
    if not STOCK:
        STOCK.update({k: m.weight for k, m in lin().items()})
    if name == "stock":
        head.load_weights([(k + ".weight", v) for k, v in STOCK.items()], strict=False)
        mx.eval(head.parameters())
    else:
        head.load_weights(ck + ".safetensors", strict=False)
        mx.eval(head.parameters())
        ch = {k: float(mx.sqrt(((m.weight - STOCK[k]) ** 2).mean() / (STOCK[k] ** 2).mean())) for k, m in lin().items() if k in STOCK}
        if not ch or max(ch.values()) == 0:
            raise SystemExit(f"{name}: the checkpoint changed nothing in the head (key mismatch?)")
        print(f"  {name}: rms change / rms weight " + " ".join(f"{k.split('.')[-1] if 'shared' not in k else 'se.' + k.split('.')[-1]} "
              f"{v:.4f}" for k, v in ch.items()), flush=True)
    return f"{how}: {to_format(head, fmt, how)} layers converted" if how in ("q", "q8", "bf16") else "fp32 as trained"


def main():
    M.limit_memory()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="stock"); ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--per-set", type=int, default=0); ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--budget", type=float, default=1e9, help="seconds after the load for prefill + every variant")
    ap.add_argument("--prefill-share", type=float, default=0.3)
    a = ap.parse_args()
    ov = os.path.join(M.TL, "data/sim125_runs.txt")            # override for a queued turn (its script is not edited)
    runs = (open(ov).read().strip() if os.path.exists(ov) else a.runs).split(",")
    assert "stock" not in runs[1:], "stock must come first"
    rt, lm, head = M.load(a.frac)
    fmt = formats(head)
    print("stock formats:", {k: (v[0], v[1], v[2]) if v[0] == "q" else (v[0], str(v[1])) for k, v in fmt.items()}, flush=True)
    by = {}
    for l in open(ITEMS):
        it = json.loads(l); by.setdefault(it["set"], []).append(it)
    if a.per_set:
        by = {s: v[:a.per_set] for s, v in by.items()}
    order = [it for grp in itertools.zip_longest(*by.values()) for it in grp if it is not None]
    tb = time.time()
    store = []
    for it in order:
        if time.time() - tb > a.budget * a.prefill_share:
            break
        S = list(it["prompt_ids"]) + list(it["completion_ids"])
        store.append((it["set"], S, len(it["prompt_ids"]), widened(lm, S)))
    print(f"prefilled {len(store)} of {len(order)} items in {time.time() - tb:.0f}s", flush=True)
    res = {}
    for vi, run in enumerate(runs):
        note = prepare(head, run, fmt)
        share = (a.budget - (time.time() - tb)) / (len(runs) - vi)
        lim = len(res[runs[0]]) if vi else len(store)
        t1, out = time.time(), []
        for j in range(lim):
            if time.time() - t1 > share:
                break
            _, S, P, W = store[j]
            out.append(simulate(lm, W, S, P, a.K))
        res[run] = out
        print(f"{run} ({note}): {len(out)} items in {time.time() - t1:.0f}s", flush=True)
    n = min(len(v) for v in res.values())
    sets = sorted({store[j][0] for j in range(n)})
    rng = np.random.default_rng(0)
    base = runs[0]
    for run in runs:
        for s in sets + ["ALL"]:
            idx = [j for j in range(n) if s == "ALL" or store[j][0] == s]
            tk = np.array([res[run][j][0] for j in idx]); ps = np.array([res[run][j][1] for j in idx])
            hit = np.mean([res[run][j][2] for j in idx])
            line = (f"{time.strftime('%m-%d %H:%M')} sim125 {run:22s} K{a.K} {s:10s} tau {tk.sum() / ps.sum():.3f} "
                    f"(items {len(idx)}, passes {ps.sum()}, depth-1 hit {hit:.3f})")
            refs = [base] if run != base else []
            fm = run.partition("@")[2] or "fp32"
            if not run.startswith("stock") and f"stock@{fm}" in res:
                refs.append(f"stock@{fm}")                     # the training effect at equal storage
            for ref in refs:
                tk0 = np.array([res[ref][j][0] for j in idx]); ps0 = np.array([res[ref][j][1] for j in idx])
                r = (tk.sum() / ps.sum()) / (tk0.sum() / ps0.sum())
                bs = []
                for _ in range(2000):
                    b = rng.integers(0, len(idx), len(idx))
                    bs.append((tk[b].sum() / ps[b].sum()) / (tk0[b].sum() / ps0[b].sum()))
                lo, hi = np.percentile(bs, [5, 95])
                line += f"  vs {ref} {100 * (r - 1):+.1f} % [90 % {100 * (lo - 1):+.1f}, {100 * (hi - 1):+.1f}]"
            print(line, flush=True); open(RES, "a").write(line + "\n")
    print(f"sim125 done: {n} items x {len(runs)} variants, {time.time() - tb:.0f}s after the load", flush=True)


if __name__ == "__main__":
    main()
