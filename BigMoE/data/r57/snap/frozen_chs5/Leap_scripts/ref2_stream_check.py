"""REF2 under BigMoE's streamed engine (EvalSuite's question, Wed 16:3x; Leap, pre-registered Wed 2026-10-28 16:35 EET
in ~/Leap/TABLE_350.md 'R-40'): is REF2 batch-invariant on the FULL 125B in EvalSuite's REF harness (evalsuite.env + evalsuite.model.load('ref', C = 192):
load_streamed, its SparseMoeBlock combine patch, no fused gate | up)? The experts take ref2's split path
(qmmg on BigMoE's gate / up / down pools).
Test: ref2.self_test(); then after the same 1K plain prefill on two caches, 16 one-token AR steps (ref2.ar_step: a
1-row verify + commit keep 1) vs one 16-row verify of the same tokens. Every layer's records (GDN, PLE, attention, MoE,
layer output) and every logits row, bitwise. All 48 layers.
Pass: self-test all True AND every record and logits row equal (then EvalSuite may run ref2.generate).
Usage: [BigMoE venv] ref2_stream_check.py [C=192] -> ~/Leap/data/ref2_stream_check.txt"""
import glob, os, sys, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)            # the streamed engine has no fused gate | up
os.environ.pop("BIGMOE_TIER", None)
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs5/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM
import numpy as np
import mlx.core as mx

MD = os.path.expanduser("~/BigMoE/models/bare")
D = os.path.expanduser("~/Leap/data")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 192
R, P = 16, 1024
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

t0 = time.time()
rt, st = EM.load(EM.variant("ref"), C=C)                # EvalSuite's REF arm: load_streamed + its install + configure
model = rt.model
import ref2
from mtplx import cache_state as CS
Q = ref2.Q
say(f"REF2 stream check: full 125B via load_streamed (C = {C}), loaded in {time.time() - t0:.0f} s, "
    f"{mx.get_active_memory() / 1e9:.1f} GB active; switch class {type(model.language_model.model.layers[0].mlp.switch_mlp).__name__}")
ref2.apply()
st_ok = ref2.self_test()
say(f"  self-test: {st_ok}")

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

files = sorted(glob.glob(os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs5/Leap_scripts/*.py")))
ids = list(rt.tokenizer.encode("".join(open(f).read() for f in files)))[:P + R]
t1 = time.time()
cv, _ = ref2.prefill(model, ids[:P])
ca, _ = ref2.prefill(model, ids[:P])
say(f"  two 1K prefills in {time.time() - t1:.0f} s")
toks = ids[P:P + R]
ref2.reset_counts()
REC.clear()
snap = CS.snapshot_untrimmable_cache_lazy(cv)
with ref2.verify_scope(model):
    lgv = model(mx.array([toks]), cache=cv)
mx.eval(lgv, [r for _, r in REC])
recv = list(REC)
cnt_v = ref2.counts(); ref2.reset_counts()
reca, lga = [], []
t2 = time.time()
for j in range(R):
    REC.clear()
    lg = ref2.ar_step(model, ca, toks[j])
    mx.eval(lg, [r for _, r in REC])
    reca.append(list(REC)); lga.append(lg)
cnt_a = ref2.counts()
say(f"  16 AR steps in {time.time() - t2:.0f} s; {len(recv)} records per forward")
say(f"  REF2 calls: 16-row verify {cnt_v}; 16 AR steps {cnt_a} (experts per AR token {cnt_a['experts'] / R:.1f})")
bad = [(j, e, recv[e][0]) for j in range(R) for e in range(min(len(recv), len(reca[j])))
       if not eqb(recv[e][1][:, j:j + 1], reca[j][e][1])]
nlog = sum(eqb(lgv[0, j], lga[j]) for j in range(R))
ok = all(st_ok.values()) and not bad and nlog == R and all(len(r) >= len(recv) for r in reca)   # AR steps add the commit's replay records after the forward's
first = min(bad, key=lambda b: b[1]) if bad else None
say(f"  logits rows equal {nlog} / {R}; differing (row, record) pairs {len(bad)}"
    + (f"; earliest record {first[1]} ({first[2]}) at row {first[0]}" if first else ""))
say(f"REF2 stream check verdict: {'PASS: batch-invariant on the streamed full model' if ok else 'FAIL'}")
open(os.path.join(D, "ref2_stream_check.txt"), "w").write("\n".join(lines) + "\n")
raise SystemExit(0 if ok else 1)
