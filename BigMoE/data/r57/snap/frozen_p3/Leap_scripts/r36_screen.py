"""R-36 INCHECK (NewIdeas GO Tue ~20:59; Leap, pre-registered Tue 2026-10-27 21:01 EET in ~/Leap/TABLE_350.md
'R-36'): the bit-identical kernels inside CA's real 12-layer check (layers 0-11 of the 125B, MTPLX's verify forward
read-only, the engine's eager 16-row check with its acceptance read and lazy commit, keep 10):
  base  the engine as shipped
  S4    GatedResidual's 16-row read (decode_verify, with block inject) -> leapk.hc_read (R-09b's S4)
  D2    mx.quantized_matmul calls MLX would serve with split-K at M = 16 -> leapk.splitk_d2 (R-13d's D2)
  both  S4 + D2
Four lanes, each with its own identically prefilled cache, the same text tokens; blocks of 8 checks alternate across
lanes in rotating order (5 rounds after 3 warm checks). Bitwise: every check's logits vs base's, and every cache
state at the end. At 1K and 8K.
Lines (NewIdeas): PASS = every bit equal (logits + all cache states) AND the in-check saving of 'both' >= 80 % of the
standalone sum (14.68 + 4.52 ms per 48 layers -> >= 3.84 ms per 12-layer check). KILL = any bit differs.
Also reported: each swap's in-check factor = 4 x its 12-layer saving / its standalone number (S4: the x1.3 question).
Usage: [BigMoE venv] r36_screen.py -> ~/Leap/data/r36_screen.txt"""
import os, sys, time
sys.argv = [sys.argv[0], "r36"]                          # ca_anat's MODE: load only, no run
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_scripts"))
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

ARMS = {"base": (False, False), "S4": (True, False), "D2": (False, True), "both": (True, True)}

class Lane:
    def __init__(self, P):
        self.cache, self.pos, self.logits = CA.prefill(P), P, []
    def states(self):
        return CA.leaves(CA.states(self.cache))

def check(L, arm, keep_logits):
    S4_ON[0], D2_ON[0] = ARMS[arm]
    x = mx.array([CA.ids[L.pos:L.pos + R]])
    snap = CS.snapshot_untrimmable_cache_lazy(L.cache)
    lg = CA.forward("A", x, L.cache)
    g = mx.argmax(lg[0], axis=-1).tolist()               # the engine's acceptance read (sync)
    model.commit_verified_window(L.cache, snap.states, keep_tokens=K, verified_tokens=R)
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
    S4_ON[0], D2_ON[0] = ARMS[arm]                       # the last commit's replay runs with the arm's kernels
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
    for a in ("S4", "D2", "both"):
        L = lanes[a]
        lg_ok = len(L.logits) == len(base.logits) and all(eq(x, y) for x, y in zip(L.logits, base.logits))
        st_ok = all(eq(x, y) for x, y in zip(L.states(), base.states()))
        bit[a] = (lg_ok, st_ok)
    m = {a: med(W[a]) for a in ARMS}
    sv = {a: med([b - c for b, c in zip(W["base"], W[a])]) for a in ("S4", "D2", "both")}
    RES[P] = (bit, sv, m)
    say(f"  ctx {P}: per 12-layer check base {m['base']:.2f}, S4 {m['S4']:.2f}, D2 {m['D2']:.2f}, both {m['both']:.2f} ms; "
        f"saved (paired) S4 {sv['S4']:.2f}, D2 {sv['D2']:.2f}, both {sv['both']:.2f}; bitwise vs base (logits of "
        f"{len(base.logits)} checks / all states): " + ", ".join(f"{a} {bit[a][0]}/{bit[a][1]}" for a in bit))
    say(f"    in-check factors (x4 for 48 layers / standalone): S4 {4 * sv['S4'] / 14.68:.2f} (the x1.3 question), "
        f"D2 {4 * sv['D2'] / 4.52:.2f}; both {4 * sv['both']:.2f} ms per 48-layer check vs 19.20 standalone")
    for L in lanes.values():
        L.logits.clear()
    del lanes
    mx.clear_cache()
say(f"  swaps engaged: S4 {COUNT['s4']} reads, D2 {COUNT['d2']} matmuls")
allbit = all(all(v) for bit, _, _ in RES.values() for v in bit.values())
worst = min(4 * sv["both"] for _, sv, _ in RES.values())
v = "KILL (a bit differs)" if not allbit else "PASS" if worst >= 15.36 else "BETWEEN"
say(f"R-36 verdict (PASS: every bit equal AND both >= 15.4 ms per 48 layers at 1K and 8K; KILL: any bit differs): {v} "
    f"(worst context: both {worst:.2f} ms per 48-layer check)")
open(os.path.join(D, "r36_screen.txt"), "w").write("\n".join(lines) + "\n")
raise SystemExit(0 if allbit else 1)
