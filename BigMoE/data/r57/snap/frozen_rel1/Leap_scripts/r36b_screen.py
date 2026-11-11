"""R-36b = R-39 STACK (NewIdeas GO ~22:15, lines below; Leap, pre-registered Tue 2026-10-27 22:13 EET in ~/Leap/TABLE_350.md 'R-36b'): do the three bit-identical
levers stack inside CA's real 12-layer check?
  base  the engine as shipped (eager 16-row check, acceptance read, lazy commit, keep 10)
  SB    the check built async every 4 layers (CA's M) + the commit replay submitted alone after the acceptance read
  S4D2  R-36's both: GatedResidual's 16-row read -> S4, split-K matmuls -> D2
  all3  SB + S4 + D2
Four lanes, identically prefilled, the same tokens, alternating 8-check blocks (5 rounds after 3), at 1K and 8K.
Bitwise: every check's logits and all cache states vs base.
Lines (NewIdeas, R-39): PASS = every bit equal AND all3's paired saving >= 6.5 ms per 12-layer check at 1K and 8K.
KILL = any bit differs. (Also reported: the stacking ratio all3 / (SB + S4D2).)
Usage: [BigMoE venv] r36b_screen.py -> ~/Leap/data/r36b_screen.txt"""
import os, sys, time
sys.argv = [sys.argv[0], "r36b"]                          # ca_anat's MODE: load only, no run
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_scripts"))
import ca_anat as CA
import leapk as LK
import numpy as np
import mlx.core as mx
from mtplx.attention_context import current_attention_phase

Q, CS, model, D = CA.Q, CA.CS, CA.model, CA.D
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v))
R, K = 16, CA.KEEP[16]
S4_ON, D2_ON = [False], [False]
COUNT = {"s4": 0, "d2": 0}

# ---------------------------------------------------------------- the swaps
WCAT = {}
ORIG_GR = Q.GatedResidual.__call__
def gr_call(self, hyper_input, pending=None):
    rows = 1
    for s_ in hyper_input.shape[:-1]:
        rows *= s_
    if S4_ON[0] and rows == R and "block_inject_weight" in self and current_attention_phase() == "decode_verify":
        if id(self) not in WCAT:
            WCAT[id(self)] = mx.concatenate([self.input_mix_weight_down.weight, self.block_inject_weight.weight], axis=0)
            mx.eval(WCAT[id(self)])
        COUNT["s4"] += 1
        return LK.hc_read(self, WCAT[id(self)], hyper_input, pending, rows=R)
    return ORIG_GR(self, hyper_input, pending)
Q.GatedResidual.__call__ = gr_call

ORIG_QMM = mx.quantized_matmul
def qmm(x, w, *args, **kw):
    scales = args[0] if len(args) > 0 else kw.get("scales")
    biases = args[1] if len(args) > 1 else kw.get("biases")
    transpose = args[2] if len(args) > 2 else kw.get("transpose", True)
    group_size = args[3] if len(args) > 3 else kw.get("group_size")
    bits = args[4] if len(args) > 4 else kw.get("bits")
    mode = args[5] if len(args) > 5 else kw.get("mode", "affine")
    if (D2_ON[0] and transpose and bits == 4 and group_size == 64 and mode == "affine" and biases is not None
            and x.dtype == mx.bfloat16 and x.ndim in (2, 3) and (x.ndim == 2 or x.shape[0] == 1)):
        M, Kd, N = x.shape[-2], x.shape[-1], w.shape[0]
        if M == R and LK.routes_splitk(M, N, Kd):
            COUNT["d2"] += 1
            return LK.splitk_d2(x.reshape(M, Kd), (w, scales, biases)).reshape(*x.shape[:-1], N)
    return ORIG_QMM(x, w, *args, **kw)
mx.quantized_matmul = qmm

ARMS = {"base": (False, False, False), "SB": (False, False, True), "S4D2": (True, True, False), "all3": (True, True, True)}

class Lane:
    def __init__(self, P):
        self.cache, self.pos, self.logits = CA.prefill(P), P, []
    def states(self):
        return CA.leaves(CA.states(self.cache))

def check(L, arm, keep_logits):
    S4_ON[0], D2_ON[0], sb = ARMS[arm]
    x = mx.array([CA.ids[L.pos:L.pos + R]])
    snap = CS.snapshot_untrimmable_cache_lazy(L.cache)
    lg = CA.forward("M" if sb else "A", x, L.cache)
    g = mx.argmax(lg[0], axis=-1).tolist()               # the engine's acceptance read (sync)
    model.commit_verified_window(L.cache, snap.states, keep_tokens=K, verified_tokens=R)
    if sb:
        mx.async_eval(*L.states())                       # the commit replay goes out on its own (A7)
    S4_ON[0] = D2_ON[0] = False
    L.pos += K
    if keep_logits:
        L.logits.append(lg[0])
    return g

def block(L, arm, n):
    while os.path.exists(PAUSE):
        time.sleep(5)
    paused = os.path.exists(PAUSE)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        check(L, arm, True)
    mx.eval(*L.states())
    S4_ON[0] = D2_ON[0] = False
    mx.synchronize()
    return 1e3 * (time.perf_counter() - t0) / n, paused or os.path.exists(PAUSE)

eq = lambda a, b: a.shape == b.shape and bool(mx.all(mx.view(a.reshape(-1), mx.uint8) == mx.view(b.reshape(-1), mx.uint8)).item())
RES = {}
for P in (1024, 8192):
    lanes = {a: Lane(P) for a in ARMS}
    for a, L in lanes.items():                           # warm-up (kept in the bitwise record)
        block(L, a, 3)
    W = {a: [] for a in ARMS}
    for r in range(5):
        order = list(ARMS)[r % 4:] + list(ARMS)[:r % 4]
        got = {a: block(lanes[a], a, 8) for a in order}
        if any(p for _, p in got.values()):
            say(f"  ctx {P}: round {r} had a PAUSE, dropped")
            continue
        for a, (wall, _) in got.items():
            W[a].append(wall)
    base = lanes["base"]
    bit = {}
    for a in ("SB", "S4D2", "all3"):
        L = lanes[a]
        lg_ok = len(L.logits) == len(base.logits) and all(eq(x, y) for x, y in zip(L.logits, base.logits))
        st_ok = all(eq(x, y) for x, y in zip(L.states(), base.states()))
        bit[a] = (lg_ok, st_ok)
    m = {a: med(W[a]) for a in ARMS}
    sv = {a: med([b - c for b, c in zip(W["base"], W[a])]) for a in ("SB", "S4D2", "all3")}
    RES[P] = (bit, sv, m)
    say(f"  ctx {P}: per 12-layer check base {m['base']:.2f}, SB {m['SB']:.2f}, S4D2 {m['S4D2']:.2f}, all3 {m['all3']:.2f} ms; "
        f"saved (paired) SB {sv['SB']:.2f}, S4D2 {sv['S4D2']:.2f}, all3 {sv['all3']:.2f} (sum of parts {sv['SB'] + sv['S4D2']:.2f}); "
        f"x4 for 48 layers: all3 {4 * sv['all3']:.2f} ms; bitwise vs base (logits of {len(base.logits)} checks / all states): "
        + ", ".join(f"{a} {bit[a][0]}/{bit[a][1]}" for a in bit))
    for L in lanes.values():
        L.logits.clear()
    del lanes
    mx.clear_cache()
say(f"  swaps engaged: S4 {COUNT['s4']} reads, D2 {COUNT['d2']} matmuls")
allbit = all(all(v) for bit, _, _ in RES.values() for v in bit.values())
stack = min(sv["all3"] / max(sv["SB"] + sv["S4D2"], 1e-9) for _, sv, _ in RES.values())
gain = min(sv["all3"] for _, sv, _ in RES.values())
v = "KILL (a bit differs)" if not allbit else "PASS" if gain >= 6.5 else "BETWEEN"
say(f"R-39 STACK verdict (NewIdeas: PASS: every bit equal AND all3 saves >= 6.5 ms per 12-layer check at 1K and 8K; "
    f"KILL: any bit differs): {v} (worst all3 saving {gain:.2f} ms; "
    f"stacking ratio {stack:.2f}; all3 per 48-layer check: " + ", ".join(f"{P // 1024}K {4 * sv['all3']:.1f} ms" for P, (_, sv, _) in RES.items()) + ")")
open(os.path.join(D, "r36b_screen.txt"), "w").write("\n".join(lines) + "\n")
raise SystemExit(0 if allbit else 1)
