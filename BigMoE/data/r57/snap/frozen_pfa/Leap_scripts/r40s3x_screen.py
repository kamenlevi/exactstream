"""R-40 STEP 3x (Leap, pre-registered Thu 2026-10-29 02:15 EET in ~/Leap/TABLE_350.md 'R-40 STEP 3x'): why STEP 3's REF2
spec checks were not bit-identical to REF2's AR (0 / 48 logits rows, math tokens off at 199), and what a 16-row check
costs in BigMoE's streamed engine once its experts are resident.
Harness as STEP 3: EvalSuite's REF harness (load 'ref', C = 176), HumanEval/68 (P 367); tokens = the first 16 of
EvalSuite's REF2 AR row.
  Part A (localize): AR = 16 ref2.ar_step on those tokens (logits). Each variant prefills its own cache (prefill last
  logits compared with the AR cache's: prefill determinism), then one 16-row verify of the same tokens; rows vs AR,
  bitwise:
    V0  REF2 on (= the 16:52 stream check, here at C = 176 and P = 367)      V0b  V0 again (verify determinism)
    V1  V0 + SB's mid-loop async every 4 layers                               V2   V0 + the drafter-tap hook on
    V3  drafter loaded; one draft (REF2 off) between prefill and verify      V4   all of STEP 3's loop (SB + taps + draft)
    V5  V0 after a stock 16-row check + commit on another cache (STEP 3 ran today's arm before its REF2 run)
  Part B (warm checks): one cache; 16-row verifies sliding by one token along the REF2 row (commit keep 1), so each
  check's experts were mostly loaded by the previous one; today vs REF2 + SB in alternating blocks of 4, 5 rounds.
  Median check (verify + commit) per arm, expert misses per check (ext.stats). Line (NewIdeas' step-3 line on warm
  checks): >= 40 ms saved PASS, < 20 KILL, else BETWEEN.
Usage: [BigMoE venv] r40s3x_screen.py [C=176] -> ~/Leap/data/r40s3x_screen.txt"""
import json, os, sys, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_third_party"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
import numpy as np
import mlx.core as mx

D = os.path.expanduser("~/Leap/data")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 176
R = 16
KEY = "humaneval/HumanEval/68"
TAPS = (5, 15, 25, 35, 45)
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v)) if len(v) else float("nan")
eqb = lambda a, b: a.shape == b.shape and bool(mx.all(mx.view(a.reshape(-1), mx.uint8) == mx.view(b.reshape(-1), mx.uint8)).item())

t0 = time.time()
rt, st = EM.load(EM.variant("ref"), C=C)
model = rt.model
import ref2
from mtplx import cache_state as CS
Q = ref2.Q
ref2.apply()
say(f"R-40 STEP 3x: full 125B via load_streamed (C = {C}), {mx.get_active_memory() / 1e9:.1f} GB active; self-test "
    f"{ref2.self_test()}")
ref2.reset_counts()

TAP = {"on": False, "x": [[] for _ in TAPS]}
lay = model.language_model.model.layers
TAP_OF = {id(lay[L]): i for i, L in enumerate(TAPS)}
_layer_call = Q.DecoderLayer.__call__
def _layer(self, *a, **kw):
    out = _layer_call(self, *a, **kw)
    i = TAP_OF.get(id(self))
    if TAP["on"] and i is not None:
        o = out if isinstance(out, mx.array) else out[0]
        B, L, W = o.shape
        TAP["x"][i].append(o.reshape(B, L, 4, W // 4).astype(mx.float32).mean(axis=2)[0].astype(mx.float16))
    return out
Q.DecoderLayer.__call__ = _layer
def taps():
    f = [mx.concatenate(x) if len(x) > 1 else x[0] for x in TAP["x"]]
    return mx.stack(f, axis=1).reshape(f[0].shape[0], -1)

_midloop = Q._prefill_midloop_eval_layers
def sb(on):
    Q._prefill_midloop_eval_layers = (lambda rows: _midloop(rows) or (4 if 2 <= int(rows) <= 16 else 0)) if on else _midloop

def states(cache):
    return [c.state for c in cache if c is not None and hasattr(c, "state")]

def prefill(ids, tap=False):
    """ref2.prefill's calls (chunks of 1024, eval per chunk); tap=True also keeps the drafter taps"""
    cache, lg = model.make_cache(), None
    TAP["x"] = [[] for _ in TAPS]; TAP["on"] = tap
    for a in range(0, len(ids), 1024):
        lg = model(mx.array([ids[a:a + 1024]]), cache=cache)
        mx.eval(lg, states(cache), [x[-1] for x in TAP["x"]] if tap else [])
    TAP["on"] = False
    return cache, lg[0, -1]

def verify(cache, toks, tap=False):
    TAP["x"] = [[] for _ in TAPS]; TAP["on"] = tap
    with ref2.verify_scope(model):
        lg = model(mx.array([toks]), cache=cache)
    TAP["on"] = False
    mx.eval(lg, taps() if tap else [])
    return lg

items = {it["key"]: it for it in ED.load("dec_humaneval") if it["key"] == KEY}
ids = list(items[KEY]["prompt_ids"])
row = {d["key"]: d for d in map(json.loads, open(os.path.expanduser("~/EvalSuite/results/ref2/decode.jsonl")))}[KEY]
toks = list(row["completion_ids"][:R])

# ---------------------------------------------------------------- Part A: localize
ref2.apply(); sb(False)
t1 = time.time()
cache, last_ar = prefill(ids)
AR = []
for j in range(R):
    lg = ref2.ar_step(model, cache, toks[j])
    mx.eval(lg)
    AR.append(lg)
del cache
mx.clear_cache()
say(f"  AR: prefill + 16 ar_step in {time.time() - t1:.0f} s; AR argmax == REF2 row: "
    f"{sum(int(mx.argmax(AR[j]).item()) == row['completion_ids'][j + 1] for j in range(R - 1))} / {R - 1}")

draft = None
def variant(name, sb_on=False, tap=False, use_draft=False):
    t = time.time()
    ref2.apply(); sb(sb_on)
    cache, last = prefill(ids, tap=tap or use_draft)
    if use_draft:
        T = taps()
        ref2.remove()
        dr = propose(mx.array([[toks[0]] + [MASK] * (R - 1)]), T[max(0, len(ids) - (SW - 1)):][None].astype(mx.bfloat16),
                     draft.make_cache())
        mx.eval(dr)
        ref2.apply()
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    lg = verify(cache, toks, tap=tap)
    eq = [eqb(lg[0, j], AR[j]) for j in range(R)]
    am = mx.argmax(lg[0], axis=-1).tolist()
    say(f"  {name}: prefill last logits == AR cache's: {eqb(last, last_ar)}; verify rows bitwise == AR: {sum(eq)} / {R}"
        f" (rows equal: {''.join('=' if e else 'x' for e in eq)}); argmax == AR argmax: "
        f"{sum(am[j] == int(mx.argmax(AR[j]).item()) for j in range(R))} / {R}; {time.time() - t:.0f} s")
    sb(False)
    out = lg
    del cache, snap
    mx.clear_cache()
    return out

V0 = variant("V0  REF2 on")
V0b = variant("V0b REF2 on, again")
say(f"  V0 rows == V0b rows (verify determinism): {sum(eqb(V0[0, j], V0b[0, j]) for j in range(R))} / {R}")
variant("V1  + SB mid-loop async", sb_on=True)
variant("V2  + drafter taps", tap=True)

sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_third_party"))
import dflash_model_mlx as DF
draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.load_weights(os.path.join(D, "l120_ft.safetensors"), strict=False)
draft.bind(model)
mx.eval(draft.parameters())
MASK, SW, K = int(draft.config.mask_token_id), int(draft.config.sliding_window), 16
def propose(blk, ctx, dcache):
    hid = draft.hidden_states(blk, ctx, dcache, 1)
    return draft.compute_logits(hid).argmax(axis=-1)      # the forward through the 125B's head is what matters here
variant("V3  drafter loaded + one draft before the verify", use_draft=True)
variant("V4  STEP 3's loop: SB + taps + draft", sb_on=True, tap=True, use_draft=True)

ref2.remove(); sb(False)                                 # V5: STEP 3 ran today's arm first; a stock 16-row check +
c5, _ = prefill(ids)                                     # commit on its own cache, then V0 again
snap5 = CS.snapshot_untrimmable_cache_lazy(c5)
lg5 = verify(c5, toks)
model.commit_verified_window(c5, snap5.states, keep_tokens=5, verified_tokens=R)
mx.eval(states(c5))
del c5, snap5, lg5
mx.clear_cache()
variant("V5  REF2 on after a stock 16-row check + commit (STEP 3's order)")

# ---------------------------------------------------------------- Part B: warm checks, today vs REF2 + SB
ref2.remove(); sb(False)
cache, _ = prefill(ids)
seq = list(row["completion_ids"])
pos = [0]
def check(ref):
    if ref:
        ref2.apply(); sb(True)
    i = pos[0]
    m0 = misses()
    mx.synchronize()
    t = time.perf_counter()
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    with ref2.verify_scope(model):
        lg = model(mx.array([seq[i:i + R]]), cache=cache)
    mx.argmax(lg[0], axis=-1).tolist()
    model.commit_verified_window(cache, snap.states, keep_tokens=1, verified_tokens=R)
    if ref:
        mx.async_eval(states(cache))
    mx.eval(states(cache))
    dt = time.perf_counter() - t
    pos[0] += 1
    if ref:
        ref2.remove(); sb(False)
    return 1e3 * dt, misses() - m0

def misses():
    try:
        return int(st.stats()["misses"])
    except Exception:
        return 0

W = {False: [], True: []}
M = {False: [], True: []}
check(False); check(True)                                # cold
for r in range(5):
    while os.path.exists(PAUSE):
        time.sleep(5)
    for ref in ((False, True) if r % 2 == 0 else (True, False)):
        for _ in range(4):
            dt, ms = check(ref)
            W[ref].append(dt); M[ref].append(ms)
save = med(W[False]) - med(W[True])
say(f"  Part B (warm 16-row checks, commit keep 1, sliding along the REF2 row): today {med(W[False]):.1f} ms "
    f"(misses per check {np.mean(M[False]):.1f}), REF2 + SB {med(W[True]):.1f} ms (misses {np.mean(M[True]):.1f}); "
    f"saved {save:.1f} ms per 48-layer check; spread today {min(W[False]):.0f}-{max(W[False]):.0f}, REF2 "
    f"{min(W[True]):.0f}-{max(W[True]):.0f}")
say(f"R-40 STEP 3x Part B (PASS >= 40 ms saved on warm checks, KILL < 20): "
    f"{'PASS' if save >= 40 else 'KILL' if save < 20 else 'BETWEEN'}")
open(os.path.join(D, "r40s3x_screen.txt"), "w").write("\n".join(lines) + "\n")
