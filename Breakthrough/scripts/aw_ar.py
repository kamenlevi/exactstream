"""Breakthrough AW-AR (NewIdeas GO Tue 10-27 ~20:1x; lines written before any run in SCOREBOARD 'AW-AR'): Fit12's
ALLWAIT on the AR path, where EVERY decode step is a single-row pass, vs the shipped late rule (B12_LATE3=2,10 adds cold
ranks 2-9 half a layer late). One process, v3l, shipped env (B12LL), PFR on. Per length (2.6K, 8.5K tokens of ctx64k)
two ABBA rounds (s a, a s); every run starts from empty pools (bt_b12's PFR release + re-create) and decodes 256 greedy
tokens. Warm rate = tokens 33-256 (Fit12's convention). PASS (fold ALLWAIT into AR too): ALLWAIT's mean warm decode >=
shipped's - 2 % at both lengths. KILL for AR: > 2 % slower (then ALLWAIT is folded on the MTP path only). Token hashes
both ways (ALLWAIT is expected to change AR bits; adopting it on AR is Kamen's call); s = s and a = a across rounds
(same start state) are checked; B.AW_N shows the switch engaged in the 'a' runs only.
Usage: [BigMoE venv] aw_ar.py TAG"""
import hashlib, json, os, sys, time

ENV = {"B12_CACHE_MB": "64", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_PF_GROUP": "64", "BIGMOE_MTP_C": "32",
       "BIGMOE_ASYNC_EVERY": "1", "B12_LATE3": "2,10", "B12_NORM": "mass"}          # pk64.py's shipped env
for k, v in ENV.items():
    os.environ.setdefault(k, v)
TAG = sys.argv[1] if len(sys.argv) > 1 else "aw_ar"
DEC, W0 = int(os.environ.get("AW_DEC", "256")), 32
LENS = [int(v) for v in os.environ.get("AW_LENS", "2600,8474").split(",")]

import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bt_b12 as B
assert hasattr(B, "AW"), "this bt_b12 has no B12_ALLWAIT switch"
rt, st = B.load_b12(0.10, mtp=False)
from bigmoe_v3f.fuse import reset_generation_state
model, tok = rt.model, rt.tokenizer
ALL = tok.encode(open(os.path.expanduser("~/Breakthrough/data/prompts/ctx64k.txt")).read())
print(f"[{TAG}] engine {B.ENGINE}, AR path, C_m {B.CFG['cm']} C_l {B.CFG['cl']}, late3 {B.CFG['l3']}, B12_PFR "
      f"{os.environ.get('B12_PFR', '1')}, {DEC} tokens per run, warm = tokens {W0 + 1}-{DEC}", flush=True)


def empty_pools():                                   # as kv8_long's dx fix: every run starts from the same empty pools
    B.PFR_LOG.append(("reset",))
    with B._PFR():
        pass


def run(p, arm, rnd):
    B.AW[0] = arm == "a"
    empty_pools()
    reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
    cache = model.make_cache()
    t0 = time.perf_counter()
    B.b12_prefill(model, mx.array([p[:-1]]), cache, 512, 4096)
    mx.eval([a for c in cache for a in (c.state if hasattr(c, "state") else []) if isinstance(a, mx.array)])
    tp = time.perf_counter() - t0
    n0 = B.AW_N[0]
    y, out, TT, lh = p[-1], [], [], None
    for i in range(DEC):
        lg = model(mx.array([[y]]), cache=cache)
        lg = (lg[0] if isinstance(lg, tuple) else lg)[0, -1]
        if i == 0:
            lh = hashlib.sha1(np.array(lg.astype(mx.float32)).tobytes()).hexdigest()[:10]
        y = int(mx.argmax(lg).item())
        out.append(y)
        TT.append(time.perf_counter())
    st.gate.check()
    r = dict(arm=arm, rnd=rnd, warm=(DEC - W0) / (TT[-1] - TT[W0 - 1]), full=(DEC - 1) / (TT[-1] - TT[0]), lh=lh,
             sha=hashlib.sha1(json.dumps(out).encode()).hexdigest()[:10],
             sha32=hashlib.sha1(json.dumps(out[:32]).encode()).hexdigest()[:10], awn=B.AW_N[0] - n0)
    print(f"[{TAG}] RESULT n {len(p) - 1} {arm}{rnd} | prefill {tp:.1f} s | warm {r['warm']:.2f} tok/s, steps 2-{DEC} "
          f"{r['full']:.2f} tok/s | logits {lh} sha32 {r['sha32']} sha{DEC} {r['sha']} | ALLWAIT layer calls {r['awn']}",
          flush=True)
    del cache
    return r


B.AW[0] = False
for L in LENS:
    p = ALL[:L + 1]
    assert len(p) == L + 1, len(p)
    R = [run(p, arm, rnd) for arm, rnd in (("s", 1), ("a", 1), ("a", 2), ("s", 2))]
    s = [r for r in R if r["arm"] == "s"]
    a = [r for r in R if r["arm"] == "a"]
    ms, ma = sum(r["warm"] for r in s) / 2, sum(r["warm"] for r in a) / 2
    print(f"[{TAG}] SUMMARY n {L}: warm shipped {ms:.2f} ({s[0]['warm']:.2f} / {s[1]['warm']:.2f}), ALLWAIT {ma:.2f} "
          f"({a[0]['warm']:.2f} / {a[1]['warm']:.2f}) tok/s -> ALLWAIT {100 * (ma / ms - 1):+.1f} % | s1 = s2 "
          f"{(s[0]['lh'], s[0]['sha']) == (s[1]['lh'], s[1]['sha'])}, a1 = a2 {(a[0]['lh'], a[0]['sha']) == (a[1]['lh'], a[1]['sha'])}"
          f" | tokens a = s: sha{DEC} {a[0]['sha'] == s[0]['sha']}, sha32 {a[0]['sha32'] == s[0]['sha32']}, first-step "
          f"logits {a[0]['lh'] == s[0]['lh']} | ALLWAIT layer calls s {[r['awn'] for r in s]} a {[r['awn'] for r in a]}",
          flush=True)
B.AW[0] = False
print(f"[{TAG}] done", flush=True)
