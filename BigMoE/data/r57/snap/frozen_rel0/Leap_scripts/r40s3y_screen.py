"""R-40 STEP 3y (Leap, pre-registered Thu 2026-10-29 16:25 EET in ~/Leap/TABLE_350.md 'R-40 STEP 3y'): STEP 3x's V5 showed
that after a stock 16-row check + commit, a fresh prefill of the same prompt gives other bits. Which call leaves the
state, and which component carries it?
Harness as STEP 3x (EvalSuite's REF harness, C = 176, HumanEval/68, P 367; tokens = EvalSuite's REF2 AR row). Prefill A
with per-layer records (GDN, attention, MoE, PLE, HC read, layer output: the last row and a checksum of all rows; the
last logits). Then, in this order, each followed by a fresh prefill compared with A record by record:
  t1  REF2 on: a 16-row check + commit keep 5 (what REF2's own spec loop does)
  t2  stock: a 16-row check, no commit
  t3  stock: commit keep 5 of t2's check
  t4  stock: one 1-row AR step (verify scopes + commit keep 1)
The first trigger that changes prefill A, and its earliest differing record, localize the leak. A measurement; no lines.
Usage: [BigMoE venv] r40s3y_screen.py [C=176] [TAG] -> ~/Leap/data/r40s3y_screen[_TAG].txt
(TAG "fix": STEP 3y-fix, the same bisection on BigMoE 7ac3192's PF1 fix; BigMoE asked Thu 17:51)"""
import json, os, sys, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel0/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
import mlx.core as mx

D = os.path.expanduser("~/Leap/data")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 176
C = min(C, 128)   # Thu 20:08: two loads at C 176 tripped run_locked's +2 GB swap guard (13 GB compressor,
                      # 19:15 and 20:06); a smaller pool cuts the footprint ~6.5 GB (numerics do not depend on C)
TAG = ("_" + sys.argv[2]) if len(sys.argv) > 2 else ""
R = 16
KEY = "humaneval/HumanEval/68"
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
eqb = lambda a, b: a.shape == b.shape and bool(mx.all(mx.view(a.reshape(-1), mx.uint8) == mx.view(b.reshape(-1), mx.uint8)).item())

rt, st = EM.load(EM.variant("ref"), C=C)
model = rt.model
import ref2
from mtplx import cache_state as CS
Q = ref2.Q
ref2.apply(); ref2.remove(); ref2.reset_counts()          # installed, off (as STEP 3)

REC = {"on": False, "r": []}
def hook(klass, part):
    orig = klass.__call__
    def call(self, *a, **kw):
        y = orig(self, *a, **kw)
        if REC["on"]:
            o = y if isinstance(y, mx.array) else y[0]
            o2 = o.reshape(-1, o.shape[-1])
            dig = mx.sum(mx.view(o2.reshape(-1), mx.uint16).astype(mx.uint32))   # checksum of every row's bits
            REC["r"].append((part, o2[-1], dig))
            mx.async_eval(o2[-1], dig)
        return y
    klass.__call__ = call
for klass, part in ((Q.GatedDeltaNet, "gdn"), (Q.Attention, "attn"), (Q.SparseMoeBlock, "moe"), (Q.PLELayer, "ple"),
                    (Q.GatedResidual, "hc"), (Q.DecoderLayer, "layer")):
    hook(klass, part)

def states(cache):
    return [c.state for c in cache if c is not None and hasattr(c, "state")]

def prefill(ids, rec=False):
    cache, lg = model.make_cache(), None
    REC["r"] = []; REC["on"] = rec
    for a in range(0, len(ids), 1024):
        lg = model(mx.array([ids[a:a + 1024]]), cache=cache)
        mx.eval(lg, states(cache))
    REC["on"] = False
    r = REC["r"]; REC["r"] = []
    return cache, lg[0, -1], r

def compare(name, A, B):
    (la, ra), (lb, rb) = A, B
    n = min(len(ra), len(rb))
    diff = [e for e in range(n) if ra[e][0] != rb[e][0] or not eqb(ra[e][1], rb[e][1]) or not eqb(ra[e][2], rb[e][2])]
    layer = 0
    first = None
    if diff:
        e0 = diff[0]
        layer = sum(1 for e in range(e0) if ra[e][0] == "layer")
        first = f"record {e0} ({ra[e0][0]}, layer {layer})"
    say(f"  after {name}: prefill last logits == A: {eqb(la, lb)}; records {len(ra)} vs {len(rb)}; differing "
        f"{len(diff)}" + (f"; earliest {first}" if first else ""))
    return not diff and eqb(la, lb)

items = {it["key"]: it for it in ED.load("dec_humaneval") if it["key"] == KEY}
ids = list(items[KEY]["prompt_ids"])
row = {d["key"]: d for d in map(json.loads, open(os.path.expanduser("~/EvalSuite/results/ref2/decode.jsonl")))}[KEY]
toks = list(row["completion_ids"][:R])
import subprocess
rev = subprocess.run(["git", "-C", os.path.expanduser("~/BigMoE"), "log", "-1", "--format=%h %s"], capture_output=True,
                     text=True).stdout.strip()[:90]
say(f"R-40 STEP 3y{TAG}: full 125B via load_streamed (C = {C}), {mx.get_active_memory() / 1e9:.1f} GB active; {KEY}, P {len(ids)}; "
    f"BigMoE {rev}")

_, la, ra = prefill(ids, rec=True)
A = (la, ra)
_, lb, rb = prefill(ids, rec=True)
compare("nothing (prefill determinism)", A, (lb, rb))

import bigmoe.stream as BS
def pending(name):                                       # amended prediction (17:36): a stale PF1 staging entry
    say(f"  PF1 staging pending after {name}: layers {sorted(BS._STAGE['pending'])} (safety-net restages so far: "
        f"{BS._STAGE['t'].get('restaged', 0)})")

def check(ref, keep):
    if ref:
        ref2.apply()
    cache, _, _ = prefill(ids)
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    with ref2.verify_scope(model):
        lg = model(mx.array([toks]), cache=cache)
    mx.eval(lg)
    pending("the 16-row check")
    if keep:
        model.commit_verified_window(cache, snap.states, keep_tokens=keep, verified_tokens=R)
        mx.eval(states(cache))
        pending(f"its commit keep {keep}")
    ref2.remove()
    return cache, snap

c1, _ = check(True, 5)
del c1
_, l1, r1 = prefill(ids, rec=True)
ok = compare("t1 REF2 16-row check + commit keep 5", A, (l1, r1))
c2, s2 = check(False, 0)
_, l2, r2 = prefill(ids, rec=True)
ok2 = compare("t2 stock 16-row check, no commit", A, (l2, r2))
model.commit_verified_window(c2, s2.states, keep_tokens=5, verified_tokens=R)
mx.eval(states(c2))
pending("t3's commit keep 5")
del c2, s2
_, l3, r3 = prefill(ids, rec=True)
compare("t3 stock commit keep 5 of t2's check", A, (l3, r3))
c4, _, _ = prefill(ids)
snap4 = CS.snapshot_untrimmable_cache_lazy(c4)
with ref2.verify_scope(model):
    lg4 = model(mx.array([[toks[0]]]), cache=c4)
mx.eval(lg4)
model.commit_verified_window(c4, snap4.states, keep_tokens=1, verified_tokens=1)
mx.eval(states(c4))
pending("t4's 1-row step (a gated call: _stage_drop)")
del c4, snap4
_, l4, r4 = prefill(ids, rec=True)
compare("t4 stock 1-row AR step", A, (l4, r4))
open(os.path.join(D, f"r40s3y_screen{TAG}.txt"), "w").write("\n".join(lines) + "\n")
