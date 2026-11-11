"""R-40 STEP 4b (Leap, pre-registered Fri 2026-10-30 02:30 EET in ~/Leap/TABLE_350.md 'R-40 STEP 4b'): the class structure of
the 16-row attention call past 512 keys, on real data. STEP 4a: past ~512 keys a check row j >= 2 leaves REF2's AR at a
QSA attention layer (rows 0-1 equal), on 702 at 536 keys and on HumanEval/68 at 516 / 536. For q_len 16 MLX 0.32.2 takes its
full attention kernel (NAX on M5: bq 64, bk 32, split head dim), not the vector kernel.
Capture (geometry/702, P 520): layer 35's _verify_sdpa inputs (q, k, v, mask) in the 16-row check at 520 whose drafts are
the AR's tokens (4a's case A: row 2's first difference is layer 35), and in AR step 2 (position 522, its padded call).
Replays, through MTPLX's head-chunked SDPA:
  a. both calls reproduce their captured outputs;
  b. check row 2 at key length N' (the check's k / v zero-padded, mask False past 536) for N' = 536..1100: which N'
     give the AR's bits, which the check's: the class map past 512;
  c. the AR's call with the real row moved from q index 0 to index 2 (same N): does the row's index matter?
  d. the bucket fix: the check's call and the AR's call both padded to N' = 1024 (and 1536): is row 2 equal?
A measurement; no lines.
Usage: [BigMoE venv] r40s4b_screen.py [C=144] -> ~/Leap/data/r40s4b_screen.txt"""
import os, sys, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
import mlx.core as mx

D = os.path.expanduser("~/Leap/data")
C = min(int(sys.argv[1]) if len(sys.argv) > 1 else 144, 144)
R, LT, ROW = 16, 35, 2
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

t0 = time.time()
rt, st = EM.load(EM.variant("ref"), C=C)
model = rt.model
import ref2
import bigmoe.stream as BS
Q = ref2.Q
ref2.apply()
say(f"R-40 STEP 4b: full 125B via load_streamed (C = {C}), loaded in {time.time() - t0:.0f} s; layer {LT}, row {ROW}")
_mid = Q._prefill_midloop_eval_layers
Q._prefill_midloop_eval_layers = lambda rows: _mid(rows) or (4 if 2 <= int(rows) <= 16 else 0)   # SB, as the loop

def pf1(on):
    if not on and BS._STAGE["bufs"] is not None:
        BS._stage_drop()
    BS._PF_PREFILL = on

def states(cache):
    return [c.state for c in cache if c is not None and hasattr(c, "state")]

lay = model.language_model.model.layers
IDX = {id(l): i for i, l in enumerate(lay)}
CUR = {"L": None, "on": False, "calls": []}
_layer_call = Q.DecoderLayer.__call__
def _layer(self, *a, **kw):
    prev, CUR["L"] = CUR["L"], IDX[id(self)]
    try:
        return _layer_call(self, *a, **kw)
    finally:
        CUR["L"] = prev
Q.DecoderLayer.__call__ = _layer
_vs = Q._verify_sdpa                                      # ref2's wrapper (it calls MTPLX's own underneath)
def _cap(q, k, v, *, scale, mask):
    out = _vs(q, k, v, scale=scale, mask=mask)
    if CUR["on"] and CUR["L"] == LT:
        CUR["calls"].append((q, k, v, mask, scale, out))
    return out
Q._verify_sdpa = _cap

def prefill(ids, chunk=1024):
    pf1(True)
    cache, lg = model.make_cache(), None
    for a in range(0, len(ids), chunk):
        lg = model(mx.array([ids[a:a + chunk]]), cache=cache)
        mx.eval(lg, states(cache))
    pf1(False)
    return cache, lg[0, -1]

items = {it["key"]: it for it in ED.load("dec_math")}
ids = list(items["math500/test/geometry/702.json"]["prompt_ids"])
P = len(ids)

# the AR: steps 0..R-1, capture step ROW's layer-LT call
cache, last = prefill(ids)
toks = [int(mx.argmax(last).item())]
for s_ in range(R):
    CUR["calls"], CUR["on"] = [], s_ == ROW
    lg = ref2.ar_step(model, cache, toks[-1])
    mx.eval(lg)
    if s_ == ROW:
        ar_call = CUR["calls"][0]
        mx.eval(*[x for x in ar_call if isinstance(x, mx.array)])
    CUR["on"] = False
    toks.append(int(mx.argmax(lg).item()))
del cache
mx.clear_cache()
# the check: drafts = the AR's tokens
cache, _ = prefill(ids)
CUR["calls"], CUR["on"] = [], True
with ref2.verify_scope(model):
    lg = model(mx.array([toks[:R]]), cache=cache)
mx.eval(lg)
CUR["on"] = False
ck_call = CUR["calls"][0]
mx.eval(*[x for x in ck_call if isinstance(x, mx.array)])
del cache
mx.clear_cache()

eqb = lambda a, b: a.shape == b.shape and bool(mx.all(mx.view(a.reshape(-1), mx.uint8) == mx.view(b.reshape(-1), mx.uint8)).item())
qa, ka, va, ma, sc, oa = ar_call
qc, kc, vc, mc, _, oc = ck_call
say(f"  AR step {ROW} call: q {tuple(qa.shape)}, k {tuple(ka.shape)}, mask {None if ma is None else (tuple(ma.shape), ma.dtype)}")
say(f"  check call:  q {tuple(qc.shape)}, k {tuple(kc.shape)}, mask {None if mc is None else (tuple(mc.shape), mc.dtype)}")
say(f"  row {ROW}: q equal {eqb(qa[:, :, 0], qc[:, :, ROW])}; keys 0..{P + ROW} equal {eqb(ka[:, :, :P + ROW + 1], kc[:, :, :P + ROW + 1])}; "
    f"values equal {eqb(va[:, :, :P + ROW + 1], vc[:, :, :P + ROW + 1])}; output equal {eqb(oa[:, :, 0], oc[:, :, ROW])}")

sdpa = Q._ORIG_VERIFY_SDPA if hasattr(Q, "_ORIG_VERIFY_SDPA") else ref2._ORIG["vsdpa"]     # MTPLX's own (no REF2 split)
def run(q, k, v, mask, n):
    """the call with keys zero-padded to n and the mask extended with False"""
    T = k.shape[2]
    if n > T:
        z = mx.zeros((k.shape[0], k.shape[1], n - T, k.shape[3]), k.dtype)
        k, v = mx.concatenate([k, z], axis=2), mx.concatenate([v, z], axis=2)
        pad = (mx.zeros(mask.shape[:-1] + (n - T,), mx.bool_) if mask.dtype == mx.bool_
               else mx.full(mask.shape[:-1] + (n - T,), -mx.inf, mask.dtype))
        mask = mx.concatenate([mask, pad], axis=-1)
    o = sdpa(q, k, v, scale=sc, mask=mask)
    mx.eval(o)
    return o
say(f"  a. replays reproduce: AR {eqb(run(qa, ka, va, ma, ka.shape[2]), oa)}; check {eqb(run(qc, kc, vc, mc, kc.shape[2]), oc)}")

Tc = kc.shape[2]
cls = []
for n in range(Tc, 1101):
    o = run(qc, kc, vc, mc, n)[:, :, ROW]
    cls.append((n, "AR" if eqb(o, oa[:, :, 0]) else "check" if eqb(o, oc[:, :, ROW]) else "other"))
runs, cur = [], None
for n, c in cls:
    if cur is None or c != cur[2]:
        cur = [n, n, c]; runs.append(cur)
    else:
        cur[1] = n
say(f"  b. check row {ROW} at key length N' = {Tc}..1100 (the AR's own call has N = {ka.shape[2]}): "
    + "; ".join(f"{a}-{b}: {c}" for a, b, c in runs[:40]) + (" ..." if len(runs) > 40 else ""))
say(f"     distinct runs {len(runs)}; N' giving the AR's bits: {sum(c == 'AR' for _, c in cls)} of {len(cls)}")

q2 = mx.concatenate([qa[:, :, 1:3], qa[:, :, 0:1], qa[:, :, 3:]], axis=2)    # the real row at index 2, copies elsewhere
m2 = mx.concatenate([ma[:, :, 1:3], ma[:, :, 0:1], ma[:, :, 3:]], axis=2)
say(f"  c. AR call with the real row at q index 2 (its mask row moved with it): equals the AR's row "
    f"{eqb(run(q2, ka, va, m2, ka.shape[2])[:, :, 2], oa[:, :, 0])}")

for n in (1024, 1536):
    a_ = run(qa, ka, va, ma, n)[:, :, 0]
    c_ = run(qc, kc, vc, mc, n)[:, :, ROW]
    say(f"  d. both padded to N' = {n}: check row {ROW} == AR row: {eqb(a_, c_)}")
say(f"R-40 STEP 4b done in {time.time() - t0:.0f} s")
open(os.path.join(D, "r40s4b_screen.txt"), "w").write("\n".join(lines) + "\n")
