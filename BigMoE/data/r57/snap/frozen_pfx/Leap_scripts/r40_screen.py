"""R-40 REF2 screen (Kamen approved decision 4; NewIdeas' lines; Leap, pre-registered Wed 2026-10-28 16:10 EET in
~/Leap/TABLE_350.md 'R-40'): the REF2 module (scripts/ref2.py) on CA's resident 12-layer slice, loaded with
MTPLX_FUSED_GATE_UP=1 so the experts run experts_nax on the fused gate | up weights.
  Part 1, batch invariance: 16 one-token AR steps (the engine's verify path, commit keep 1) vs one 16-row verify, with
          REF2 on, at 1K: every hooked record (GDN, PLE, attention, MoE, layer outputs) and every logits row, bitwise.
  Part 2, speed: per 12-layer check, today's kernels (eager check) vs REF2 + SB (check async every 4 layers + the
          commit replay submitted alone); two identically prefilled lanes, the same tokens, alternating 8-check
          blocks, 5 rounds, at 1K and 8K. (The lanes' bits differ by design: REF2 is the new reference.)
  Plus ref2.self_test().
Lines (NewIdeas): PASS = bitwise batch-invariant (Part 1: all records and logits of all 16 rows) AND REF2 + SB <= today
- 40 ms per 48-layer check (>= 10 ms saved per 12-layer check) at 1K and 8K. KILL = not invariant.
Usage: [BigMoE venv] r40_screen.py -> ~/Leap/data/r40_screen.txt"""
import os, sys, time
os.environ["MTPLX_FUSED_GATE_UP"] = "1"                 # before the model loads: experts_nax needs the fused weights
sys.argv = [sys.argv[0], "r40"]
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_scripts"))
import ca_anat as CA
import ref2
import numpy as np
import mlx.core as mx

Q, CS, model, D = CA.Q, CA.CS, CA.model, CA.D
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v))
R, K = 16, CA.KEEP[16]
ref2.apply(); ref2.remove()                             # installed, off until a lane turns it on
fused = sum(isinstance(l.mlp.switch_mlp, Q._FusedGateUpSwitchGLU) for l in CA.inner.layers)
say(f"R-40: CA's 12-layer slice, MTPLX_FUSED_GATE_UP=1 ({fused} of {len(CA.inner.layers)} MoE blocks fused); self-test "
    f"{ref2.self_test()}")

REC = []
def hook(klass, part):
    orig = klass.__call__
    def call(self, *a, **kw):
        y = orig(self, *a, **kw)
        REC.append((part, y if isinstance(y, mx.array) else y[0]))
        return y
    klass.__call__ = call
for klass, part in ((Q.GatedDeltaNet, "gdn"), (Q.Attention, "attn"), (Q.SparseMoeBlock, "moe"), (Q.PLELayer, "ple"),
                    (Q.DecoderLayer, "layer")):
    hook(klass, part)
eqb = lambda a, b: a.shape == b.shape and bool(mx.all(mx.view(a.reshape(-1), mx.uint8) == mx.view(b.reshape(-1), mx.uint8)).item())

# ---------------------------------------------------------------- Part 1: batch invariance with REF2 on
ref2.apply()
P = 1024
toks = CA.ids[P:P + R]
cv, ca = CA.prefill(P), CA.prefill(P)
REC.clear()
snap = CS.snapshot_untrimmable_cache_lazy(cv)
lgv = CA.forward("A", mx.array([toks]), cv)
mx.eval(lgv, [r for _, r in REC])
recv = list(REC)
reca, lga = [], []
for j in range(R):
    REC.clear()
    snap = CS.snapshot_untrimmable_cache_lazy(ca)
    lg = CA.forward("A", mx.array([[toks[j]]]), ca)
    mx.eval(lg, [r for _, r in REC])
    reca.append(list(REC)); lga.append(lg)
    model.commit_verified_window(ca, snap.states, keep_tokens=1, verified_tokens=1)
    mx.eval(CA.states(ca))
ref2.remove()
bad = [(j, e, recv[e][0]) for j in range(R) for e in range(len(recv))
       if not eqb(recv[e][1][:, j:j + 1], reca[j][e][1])]
nlog = sum(eqb(lgv[:, j:j + 1], lga[j]) for j in range(R))
invariant = not bad and nlog == R
say(f"  Part 1 (1K): logits rows equal {nlog} / {R}; differing (row, record) pairs {len(bad)}"
    + (f"; first {bad[0]}" if bad else "") + f" -> batch-invariant: {invariant}")
del cv, ca
REC.clear()
mx.clear_cache()
for klass in (Q.GatedDeltaNet, Q.Attention, Q.SparseMoeBlock, Q.PLELayer, Q.DecoderLayer):
    pass                                                  # hooks stay (cheap); REC is cleared per block below

# ---------------------------------------------------------------- Part 2: speed, today vs REF2 + SB
class Lane:
    def __init__(self, P_):
        self.cache, self.pos = CA.prefill(P_), P_
    def states(self):
        return CA.leaves(CA.states(self.cache))

def check(L, ref):
    if ref:
        ref2.apply()
    x = mx.array([CA.ids[L.pos:L.pos + R]])
    snap = CS.snapshot_untrimmable_cache_lazy(L.cache)
    lg = CA.forward("M" if ref else "A", x, L.cache)
    mx.argmax(lg[0], axis=-1).tolist()
    model.commit_verified_window(L.cache, snap.states, keep_tokens=K, verified_tokens=R)
    if ref:
        mx.async_eval(*L.states())
        ref2.remove()
    L.pos += K

def block(L, ref, n):
    while os.path.exists(PAUSE):
        time.sleep(5)
    paused = os.path.exists(PAUSE)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        check(L, ref)
        REC.clear()
    mx.eval(*L.states())
    mx.synchronize()
    return 1e3 * (time.perf_counter() - t0) / n, paused or os.path.exists(PAUSE)

SAVE = {}
for P_ in (1024, 8192):
    lanes = {"today": Lane(P_), "REF2": Lane(P_)}
    for a, L in lanes.items():
        block(L, a == "REF2", 3)
    W = {a: [] for a in lanes}
    for r in range(5):
        order = list(lanes) if r % 2 == 0 else list(lanes)[::-1]
        got = {a: block(lanes[a], a == "REF2", 8) for a in order}
        if any(p for _, p in got.values()):
            say(f"  ctx {P_}: round {r} had a PAUSE, dropped")
            continue
        for a, (w, _) in got.items():
            W[a].append(w)
    sv = med([b - c for b, c in zip(W["today"], W["REF2"])])
    SAVE[P_] = sv
    say(f"  Part 2 ctx {P_}: per 12-layer check today {med(W['today']):.2f}, REF2 + SB {med(W['REF2']):.2f} ms; saved "
        f"(paired) {sv:.2f} -> x4 = {4 * sv:.1f} ms per 48-layer check")
    del lanes
    mx.clear_cache()
worst = min(SAVE.values())
v = "KILL (not batch-invariant)" if not invariant else "PASS" if worst >= 10 else "BETWEEN"
say(f"R-40 verdict (PASS: bitwise batch-invariant AND >= 10 ms saved per 12-layer check (40 per 48) at 1K and 8K; "
    f"KILL: not invariant): {v} (worst {worst:.2f} ms per 12-layer check = {4 * worst:.1f} per 48)")
open(os.path.join(D, "r40_screen.txt"), "w").write("\n".join(lines) + "\n")
