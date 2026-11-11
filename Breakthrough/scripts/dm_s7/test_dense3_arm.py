"""CPU wiring test of NewIdeas' DENSE-3 map arm (Breakthrough, Wed 2026-11-04 17:0x; NewIdeas 16:5x, the arm code is
dense3_arm.py = ~/NewIdeas/data/arms/dense3_arm.py at 8fefaa3; Leap's ~/Leap/scripts/dense3.py read-only). Runs against
the dm_ablations.py first on PYTHONPATH (install_dense3.sh: the scratch copy; later: the installed one), with a mock
model in place of RT["rt"] (no GPU, no weights loaded; bf16 scales / biases as the real model's):
  T1  the stack "qom_0_live_ball_c147_t3+dense3" resolves; its PERT equals screen 6's (dense3 adds no PERT key);
  T2  the setup rewrites exactly the targets (4-bit g64 GDN in / out and attention q / k / v / o projections) to
      w & 0xEEEEEEEE, biases bf16(b + s / 2), scales untouched; decoys (router, an 8-bit projection, an MTP projection,
      a 4-bit g32 projection) untouched; one TEARDOWN entry; the "[dm] dense3: <n> modules" line, n = 120 here;
  T3  clear() puts back the very same arrays (weight, scales, biases) and empties PERT and TEARDOWN;
  T4  the env guard: with BIGMOE_PKD1=1 (and with BIGMOE_SCOUT_BUNDLE=1) the setup raises before touching a weight.
Usage (BigMoE venv): PYTHONPATH=<dir with dm_ablations.py>:~/FableGod/decision_map python test_dense3_arm.py"""
import os, sys
import mlx.core as mx
import mlx.nn as nn
mx.set_default_device(mx.cpu)
import dm_ablations as A
import dm_common as C
ARM, TWIN = "qom_0_live_ball_c147_t3+dense3", "qom_0_live_ball_c147_t3"
print(f"[t] dm_ablations from {A.__file__}", flush=True)
bad = []
def ok(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond: bad.append(msg)

def QL(bits=4, gs=64):
    m = nn.QuantizedLinear(128, 64, bias=False, group_size=gs, bits=bits)
    m.scales = m.scales.astype(mx.bfloat16); m.biases = m.biases.astype(mx.bfloat16)
    return m
class Proj(nn.Module):
    def __init__(self, names, **kw):
        super().__init__()
        for n in names: setattr(self, n, QL(**kw))
class Wrap(nn.Module):
    def __init__(self, **sub):
        super().__init__()
        for k, v in sub.items(): setattr(self, k, v)
class Block(nn.Module):
    def __init__(self, attn):
        super().__init__()
        if attn: self.self_attn = Proj(("q_proj", "k_proj", "v_proj", "o_proj"))
        else: self.linear_attn = Proj(("in_proj", "out_proj"))
        self.mlp = Proj(("gate",))                                                   # decoy: the router
class Mock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [Block(attn=(i % 4 == 3)) for i in range(48)]   # 36 GDN x 2 + 12 attention x 4 = 120 targets
        self.mtp = Wrap(self_attn=Proj(("q_proj",)))                               # decoy: an MTP projection
        self.extra = Wrap(self_attn=Proj(("q_proj",), bits=8))                     # decoy: 8-bit
        self.extra2 = Wrap(linear_attn=Proj(("in_proj",), gs=32))                  # decoy: 4-bit g32
model = Mock(); mx.eval(model.parameters())
mods = {p: m for p, m in model.named_modules() if isinstance(m, nn.QuantizedLinear)}
TGT = {p for p in mods if p.startswith("layers.") and (".self_attn." in p or ".linear_attn." in p)}
KEYS = ("weight", "scales", "biases")
orig = {p: tuple(m[k] for k in KEYS) for p, m in mods.items()}                      # references (arrays are immutable)
print(f"[t] mock: {len(mods)} quantized linears, {len(TGT)} targets, {len(mods) - len(TGT)} decoys; "
      f"biases {orig[next(iter(TGT))][2].dtype}", flush=True)
C.RT["rt"] = model
eq = lambda a, b: a.dtype == b.dtype and a.shape == b.shape and bool(mx.array_equal(a, b))
same_obj = lambda p: all(mods[p][k] is o for k, o in zip(KEYS, orig[p]))

# T1
C.clear(); A.ablation(TWIN)(); p6 = dict(C.PERT); C.clear()
f = A.ablation(ARM)
ok(callable(f) and "dense3" in A.ABLATIONS, "T1 the stack resolves; ABLATIONS['dense3'] is registered")
f(); pd = dict(C.PERT)
ok(pd == p6 and p6.get("gate") == ("t3mask",), f"T1 the stack's PERT == screen 6's ({sorted(p6)})")
# T2
n_t = n_d = 0
for p, (w, s, b) in orig.items():
    w2, s2, b2 = (mods[p][k] for k in KEYS)
    if p in TGT:
        wx = mx.bitwise_and(w, mx.array(0xEEEEEEEE, mx.uint32))
        bx = (b.astype(mx.float32) + 0.5 * s.astype(mx.float32)).astype(b.dtype)
        n_t += eq(w2, wx) and eq(s2, s) and eq(b2, bx) and not eq(w2, w) and not eq(b2, b)
    else:
        n_d += same_obj(p)
ok(n_t == len(TGT), f"T2 {n_t}/{len(TGT)} targets at w & 0xEEEEEEEE, biases bf16(b + s/2), scales unchanged")
ok(n_d == len(mods) - len(TGT), f"T2 {n_d}/{len(mods) - len(TGT)} decoys untouched (router, MTP, 8-bit, g32)")
ok(len(C.TEARDOWN) == 1, f"T2 one TEARDOWN entry ({len(C.TEARDOWN)})")
# T3
C.clear()
back = sum(same_obj(p) for p in mods)
ok(back == len(mods), f"T3 clear() puts back the original arrays in {back}/{len(mods)} modules")
ok(not C.PERT and not C.TEARDOWN, "T3 PERT and TEARDOWN empty after clear()")
# T4
for var in ("BIGMOE_PKD1", "BIGMOE_SCOUT_BUNDLE"):
    os.environ[var] = "1"
    try: A.ablation(ARM)(); raised = False
    except AssertionError: raised = True
    finally: del os.environ[var]
    untouched = all(same_obj(p) for p in mods); n_td = len(C.TEARDOWN); C.clear()
    ok(raised and untouched and n_td == 0, f"T4 {var}=1: the setup raises, no weight touched")
print("ALL OK" if not bad else f"FAILED: {len(bad)}", flush=True)
sys.exit(1 if bad else 0)
