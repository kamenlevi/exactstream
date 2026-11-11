"""Split one S=3 verify pass vs one plain step (S=1) on the 125B: wall time, misses, stalled layers, SSD/loader time.
Each pass runs twice at the same position: cold (as in decode) and warm (its experts were just loaded, ~0 misses),
so warm = compute only and cold - warm = streaming cost. The cache is rolled back between runs (MTPLX capture-commit
snapshot). Run twice: pass 1 saves the greedy continuation, pass 2 (fresh pools) measures every GAP-th position, S=3 and S=1 alternating.
Usage: verify_split.py [frac=0.50] [positions=12]   (reference env: BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1)"""
import os, sys, time
for k, v in (("MTPLX_NGRAM_PREWARM", "off"), ("MTPLX_COMPILED_VERIFY", "0"), ("MTPLX_QWEN4EXP_COMPILE", "0"),
             ("MTPLX_COMPILE_AR_FORWARD", "0"), ("MTPLX_COMPILED_TARGET_PREFIX", "0"), ("MTPLX_FAMILY_CAPTURE_COMMIT", "1")):
    os.environ.setdefault(k, v)
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import numpy as np, mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe.gated import ext
from mtplx.cache_state import rollback_after_verify, snapshot_untrimmable_cache_lazy
from mtplx.graphbank import ensure_eager_window_capacity
frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.50
npos = int(sys.argv[2]) if len(sys.argv) > 2 else 18
rt, st = load_streamed(os.path.expanduser("~/BigMoE/models/bare"), round(frac * 512))
model = rt.model
ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read()}],
                                       add_generation_prompt=True, enable_thinking=False)
ids = ids if isinstance(ids[0], int) else ids["input_ids"]
L0 = lambda o: o[0] if isinstance(o, tuple) else o

# reference continuation (plain greedy) so the S=3 rows are the tokens a perfect draft would propose
GAP = 4                                   # measured positions are GAP apart: an S=3 pass never pre-loads a later measured row
TOK = os.path.expanduser("~/BigMoE/data/verify_split_cont.npy")
if not os.path.exists(TOK):               # pass 1 (own process): the continuation; pass 2 measures with pools fresh from prefill
    from bigmoe.decode import greedy
    np.save(TOK, np.array(greedy(model, ids, GAP * npos + 8))); print("saved continuation; run again"); sys.exit(0)
cont = [int(t) for t in np.load(TOK)]
cache = model.make_cache()
lg = L0(model(mx.array([ids[:-1]]), cache=cache)); mx.eval(lg)
seq = [ids[-1]] + cont


def run(toks):
    snap = snapshot_untrimmable_cache_lazy(cache); ensure_eager_window_capacity(cache, len(toks))
    a = ext.stats(); t = time.perf_counter()
    with model.verify_capture_scope():
        lg = L0(model(mx.array([toks]), cache=cache))
        mx.eval(mx.argmax(lg[0].astype(mx.float32), axis=-1))
    dt = time.perf_counter() - t; b = ext.stats()
    rollback_after_verify(cache, snap, verified_tokens=len(toks)); model.clear_verify_capture(cache)
    d = {k: b[k] - a[k] for k in ("misses", "stalled", "io_s", "slow_s", "slow_calls", "cpu_wait_s", "proc_s")}
    d["ms"] = dt * 1e3
    return d


R = {k: [] for k in ("1c", "1w", "3c", "3w")}
for i in range(GAP * npos):
    if i % GAP == 0:
        k = "3" if (i // GAP) % 2 == 0 else "1"
        toks = seq[i:i + int(k)]
        R[k + "c"].append(run(toks)); R[k + "w"].append(run(toks))
    lg = L0(model(mx.array([seq[i:i + 1]]), cache=cache)); mx.eval(lg)       # advance one real token
st.gate.check()
R = {k: v[1:] for k, v in R.items()}           # first pass of each width traces/compiles: dropped
for k, rows in R.items():
    m = {f: float(np.median([r[f] for r in rows])) for f in rows[0]}
    print(f"S={k[0]} {'cold' if k[1] == 'c' else 'warm'} n={len(rows)}: " + " ".join(
        f"{f}={m[f]:.1f}" if f in ("ms", "misses", "stalled", "slow_calls") else f"{f}={m[f] * 1e3:.1f}ms" for f in m), flush=True)
