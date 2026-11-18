"""rec1a_gateview_test.py (Breakthrough, Wed 18 Nov 2026 15:2x): the test of rec1a_c25.GateView, the fix for the first
engine run's stop (rec1a_1119: BG.seq on the bigmoe.gated MODULE). Fake objects shaped like the engine's, no GPU, no
model. Checks:
  (a) the engine's shape: G["BG"] = a module with ext + MAXK, the Gate on every streamed switch layer as _bm["gate"]
      (a dense layer without switch_mlp in between) -> the view finds the one Gate; seq, nl, tables, status and
      new_pass() are the Gate's, read live; ext / maxk come from the module; families() keeps the ring family;
      counters() reads the module's ext and the Gate's status;
  (b) the stub's shape: G["BG"] = the Gate itself (has seq and ext) -> used as is;
  (c) two different Gates on the switch layers -> AssertionError;
  (d) no Gate on any switch layer -> AssertionError.
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/m1/rec1a/rec1a_gateview_test.py   (prints GATEVIEW TEST: PASS / FAIL)"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rec1a_c25 as RC  # noqa: E402


class FakeGate:                                          # the fields of bigmoe.gated.Gate that rec1a_c25 reads
    def __init__(self, ext=None):
        self.nl, self.ntab, self.nreg, self.seq = 97, 1024, 128, 5
        self.tables = np.zeros(1024 * 17 * 32, np.int32)
        self.region = np.zeros(128 * 4 * 32 * 11, np.int32)
        self.ring = np.zeros(4 << 20, np.int32)
        self.status = np.zeros(4096, np.int32)
        if ext is not None:
            self.ext = ext

    def new_pass(self):
        self.seq += 1
        return self.seq


def ns(**k):
    return types.SimpleNamespace(**k)


def model_with(gates):                                   # gates: one per layer; None = a dense layer (no switch_mlp)
    layers = [ns(mlp=ns()) if g is None else ns(mlp=ns(switch_mlp=ns(_bm=dict(L=i, gate=g)))) for i, g in enumerate(gates)]
    return ns(language_model=ns(model=ns(layers=layers)))


def raises(f):
    try:
        f()
    except AssertionError:
        return True
    return False


mod = types.ModuleType("fake_gated")
mod.ext = ns(stats=lambda: {"gated": 7})
mod.MAXK = 320
g = FakeGate()
g.status[0], g.status[16] = 2, 3
v = RC.GateView(dict(BG=mod, model=model_with([g, None, g, g])))
s0 = int(v.seq)
g.new_pass()
live = int(v.seq) == s0 + 1
s2 = int(v.new_pass())
nl, fams = RC.families(v)
names = sorted({f[0].split(".")[0] for f in fams})
a = dict(found=v.g is g and v.src.startswith("the switch layers"), live_seq=live and s2 == g.seq == s0 + 2,
         ext=v.ext is mod.ext, maxk=v.maxk == 320, nl=nl == 97, families=names == ["reg", "ring", "tab"],
         counters=RC.counters(v) == (7, 2, 3), tables=v.tables is g.tables)
ext2 = ns(stats=lambda: {"gated": 1})
g2 = FakeGate(ext=ext2)
v2 = RC.GateView(dict(BG=g2))
b = dict(used_as_is=v2.g is g2, ext=v2.ext is ext2, live_seq=int(v2.seq) == 5)
c = raises(lambda: RC.GateView(dict(BG=mod, model=model_with([FakeGate(), FakeGate()]))))
d = raises(lambda: RC.GateView(dict(BG=mod, model=model_with([None, None]))))
ok = all(a.values()) and all(b.values()) and c and d
print(f"  (a) engine shape: {a}")
print(f"  (b) stub shape: {b}")
print(f"  (c) two Gates -> AssertionError: {c}")
print(f"  (d) no Gate -> AssertionError: {d}")
print(f"GATEVIEW TEST: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
