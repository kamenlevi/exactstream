> (a quote from the project owner, kept private)
Leap, pre-registered Thu 21:00 EET in ~/Leap/TABLE_350.md '## R-50 SLOTCALL' / R-50b): 16-row check time, gather vs
slot, at the 45 GB line's REF2 configuration (C 192) with the needed experts resident (warm checks).
Harness: EvalSuite's REF harness (load_streamed + its install), C = 192, PF1 for prefills only, REF2 + SB on, no drafter.
Windows: 16-row checks sliding by one token along HumanEval/68's REF2 row (EvalSuite results/ref2), commit keep 1 (3z's
Part W), so each check's experts are almost all resident. Four passes over the same N windows, each from a fresh
prefill, in the order gather, slot, slot, gather. Per window the check's logits are compared across the arms (bitwise);
time = median over warm checks (window >= 2) per arm.
  gather: BigMoE's _sync_call as shipped (scratch gather per layer)
  slot:   R-50's slot call (REF2 on, <= 0.9 C distinct experts: the experts kernel on pool slots, no gather)
Lines (pre-registered, R-50's): PASS = every window's logits equal across the arms AND >= 40 ms saved per warm check
(median gather - median slot); KILL = < 15 ms.
Usage: [BigMoE venv] r50b_screen.py [C=192] [N=14] -> ~/Leap/data/r50b_screen.txt"""
import hashlib, json, os, sys, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
import numpy as np
import mlx.core as mx

D = os.path.expanduser("~/Leap/data")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 192
N = int(sys.argv[2]) if len(sys.argv) > 2 else 14
R = 16
KEY = "humaneval/HumanEval/68"
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v)) if len(v) else float("nan")

t0 = time.time()
rt, st = EM.load(EM.variant("ref"), C=C)
model = rt.model
import ref2
import bigmoe.stream as BS
from mtplx import cache_state as CS
Q = ref2.Q
ref2.apply()
Q_mid = Q._prefill_midloop_eval_layers
Q._prefill_midloop_eval_layers = lambda rows: Q_mid(rows) or (4 if 2 <= int(rows) <= 16 else 0)   # SB
say(f"R-50b: full 125B via load_streamed (C = {C}), loaded in {time.time() - t0:.0f} s, "
    f"{mx.get_active_memory() / 1e9:.1f} GB active; REF2 + SB on; {KEY}; {N} sliding windows per pass")

ref2.CFG["slot_call"] = False                           # ref2.py has R-50 built in since 22:1x: keep the
_orig_sync = BS._sync_call                               # gather arm on BigMoE's shipped path
SLOT = {"on": False, "calls": 0, "fallback": 0}
def _slot_call(cls, self, x, indices, bm, *a, **kw):     # = R-50's (scripts/r50_screen.py)
    if not SLOT["on"] or "tier" in bm or BS._PF_PREFILL or bm.get("mtp") or not ref2.enabled():
        return _orig_sync(cls, self, x, indices, bm, *a, **kw)
    L, Cb = bm["L"], bm["C"]
    ids = np.array(indices)
    uniq = np.unique(ids)
    if len(uniq) > int(Cb * 0.9):
        SLOT["fallback"] += 1
        return _orig_sync(cls, self, x, indices, bm, *a, **kw)
    BS.SYNC["calls"] += 1; BS.SYNC["rows"] += ids.size // ids.shape[-1]
    slots = np.array(BS.ext.ensure(L, mx.array(uniq.astype(np.int32))))
    sl = slots[np.searchsorted(uniq, ids)].astype(np.int32)
    y = cls.__call__(self, x, mx.array(sl).astype(indices.dtype), *a, **kw)
    mx.eval(y)
    SLOT["calls"] += 1
    return y
BS._sync_call = _slot_call

def pf1(on):
    if not on and BS._STAGE["bufs"] is not None:
        BS._stage_drop()
    BS._PF_PREFILL = on

def states(cache):
    return [c.state for c in cache if c is not None and hasattr(c, "state")]

def misses():
    try:
        return int(st.stats()["misses"])
    except Exception:
        return 0

items = {it["key"]: it for it in ED.load("dec_humaneval") if it["key"] == KEY}
ids = list(items[KEY]["prompt_ids"])
seq = list({d["key"]: d for d in map(json.loads, open(os.path.expanduser("~/EvalSuite/results/ref2/decode.jsonl")))}[KEY]["completion_ids"])

def one_pass(slot):
    SLOT["on"] = False
    pf1(True)
    cache = model.make_cache()
    lg = model(mx.array([ids]), cache=cache)
    mx.eval(lg, states(cache))
    pf1(False)
    SLOT["on"] = slot
    ts, hs, ms = [], [], []
    for i in range(N):
        m0 = misses()
        mx.synchronize()
        t = time.perf_counter()
        snap = CS.snapshot_untrimmable_cache_lazy(cache)
        with ref2.verify_scope(model):
            lg = model(mx.array([seq[i:i + R]]), cache=cache)
        mx.eval(lg)
        model.commit_verified_window(cache, snap.states, keep_tokens=1, verified_tokens=R)
        mx.async_eval(states(cache))                     # SB: the commit replay submitted alone
        mx.eval(states(cache))
        ts.append(1e3 * (time.perf_counter() - t))
        hs.append(hashlib.sha1(np.array(mx.view(lg[0].reshape(-1), mx.uint8)).tobytes()).hexdigest()[:16])
        ms.append(misses() - m0)
    SLOT["on"] = False
    del cache
    mx.clear_cache()
    return ts, hs, ms

P = {}
for n, slot in enumerate((False, True, True, False)):
    while os.path.exists(os.path.expanduser("~/.local/share/quiet/PAUSE")):
        time.sleep(5)
    c0, f0 = SLOT["calls"], SLOT["fallback"]
    ts, hs, ms = one_pass(slot)
    P.setdefault(slot, []).append((ts, hs, ms))
    say(f"  pass {n + 1} [{'slot' if slot else 'gather'}]: check + commit ms per window {[round(x) for x in ts]}; "
        f"misses per check {ms}; slot calls {SLOT['calls'] - c0}, fallbacks {SLOT['fallback'] - f0}")
hashes = [h for runs in P.values() for (_, h, _) in runs]
equal = all(h == hashes[0] for h in hashes)
warm = {s: [t for (ts, _, _) in runs for t in ts[2:]] for s, runs in P.items()}
save = med(warm[False]) - med(warm[True])
v = "FAIL (arms not bit-equal)" if not equal else "PASS" if save >= 40 else "KILL" if save < 15 else "BETWEEN"
say(f"  warm checks (windows >= 2, 2 passes each): gather median {med(warm[False]):.1f} ms, slot median {med(warm[True]):.1f} ms; "
    f"saved {save:.1f} ms per 48-layer check; logits equal across all 4 passes for every window: {equal}; "
    f"misses per warm check gather {np.mean([m for (_, _, ms) in P[False] for m in ms[2:]]):.1f}, slot "
    f"{np.mean([m for (_, _, ms) in P[True] for m in ms[2:]]):.1f}")
say(f"R-50b verdict (PASS: bit-equal AND >= 40 ms saved per warm check at C {C}; KILL: < 15 ms): {v}; {time.time() - t0:.0f} s")
open(os.path.join(D, "r50b_screen.txt"), "w").write("\n".join(lines) + "\n")
