"""R-40 STEP 4a (Leap, pre-registered Fri 2026-10-30 02:10 EET in ~/Leap/TABLE_350.md 'R-40 STEP 4a'): where does REF2's
live loop leave REF2's AR past ~512 keys? EvalSuite's R-56 (01:40): on geometry/702 (P 520) only 2/48 kept rows are
bitwise equal to the in-process REF2 AR, and all 8 full-length divergences sit at 498-1124 total keys. Below that (3k,
and R-56's first 48-97 rows on every other prompt) every row is exact.
Each part: REF2's AR from the prefill (per step: its logits, and every DecoderLayer's output for the real row), then,
from a fresh prefill, the same AR steps up to the check's start, and one 16-row check whose drafts are the AR's own
next tokens. Every row of that check is then an AR row. Per row j: logits bitwise vs AR step start + j; the first layer
whose output row differs. Per layer: the rows that differ there.
  A: geometry/702 (P 520), the check right after the prefill (keys 536).
  B: HumanEval/68 (P 367), the check right after the prefill (keys 383; 3k: exact): the control.
  C: HumanEval/68, checks starting at absolute 500 (its rows' AR steps straddle 512 keys) and 520 (all past 512).
     C against A separates 'past 512 keys' from 'a prompt longer than 512'.
Harness = 3k's (EvalSuite's REF harness, PF1 for prefills only, REF2 + SB with ref2's defaults), no drafter. C <= 144.
A measurement (a bisection); no lines.
Usage: [BigMoE venv] r40s4a_screen.py [C=144] -> ~/Leap/data/r40s4a_screen.txt"""
import os, sys, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs3/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
import numpy as np
import mlx.core as mx

D = os.path.expanduser("~/Leap/data")
C = min(int(sys.argv[1]) if len(sys.argv) > 1 else 144, 144)
R = 16
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
say(f"R-40 STEP 4a: full 125B via load_streamed (C = {C}), loaded in {time.time() - t0:.0f} s; REF2 defaults {ref2.CFG}")
say(f"  self-test: {ref2.self_test()}")
_mid = Q._prefill_midloop_eval_layers
Q._prefill_midloop_eval_layers = lambda rows: _mid(rows) or (4 if 2 <= int(rows) <= 16 else 0)   # SB, as the loop

def pf1(on):
    if not on and BS._STAGE["bufs"] is not None:
        BS._stage_drop()
    BS._PF_PREFILL = on

def states(cache):
    return [c.state for c in cache if c is not None and hasattr(c, "state")]

lay = model.language_model.model.layers
NL = len(lay)
IDX = {id(l): i for i, l in enumerate(lay)}
kind = lambda L: "QSA attention" if L % 4 == 3 else "PLE" if L == 1 else "GDN"
CAP = {"on": False, "out": {}}
_layer_call = Q.DecoderLayer.__call__
def _layer(self, *a, **kw):
    out = _layer_call(self, *a, **kw)
    if CAP["on"]:
        o = out if isinstance(out, mx.array) else out[0]
        CAP["out"].setdefault(IDX[id(self)], o)        # the forward's own call (a replay would come later)
    return out
Q.DecoderLayer.__call__ = _layer

def prefill(ids, chunk=1024):
    pf1(True)
    cache, lg = model.make_cache(), None
    for a in range(0, len(ids), chunk):
        lg = model(mx.array([ids[a:a + chunk]]), cache=cache)
        mx.eval(lg, states(cache))
    pf1(False)
    return cache, lg[0, -1]

def ar(ids, n, cap):
    """REF2's AR for n steps from the prefill: tokens, per-step logits, per-layer outputs of the steps in cap"""
    cache, last = prefill(ids)
    tok = int(mx.argmax(last).item())
    toks, lgs, lays = [tok], [], {}
    for s_ in range(n):
        CAP["out"], CAP["on"] = {}, s_ in cap
        lg = ref2.ar_step(model, cache, tok)
        mx.eval(lg, list(CAP["out"].values()))
        CAP["on"] = False
        if s_ in cap:
            lays[s_] = {L: o.reshape(-1, o.shape[-1])[0] for L, o in CAP["out"].items()}
        lgs.append(lg)
        tok = int(mx.argmax(lg).item())
        toks.append(tok)
    del cache
    mx.clear_cache()
    return toks, lgs, lays

eqb = lambda a, b: a.size == b.size and bool(mx.all(mx.view(a.reshape(-1), mx.uint8) == mx.view(b.reshape(-1), mx.uint8)).item())

def check(ids, toks, m, lgs, lays, tag):
    """AR steps 0..m-1 from a fresh prefill, then one 16-row check of toks[m .. m+15]: row j == AR step m + j"""
    cache, _ = prefill(ids)
    for s_ in range(m):
        mx.eval(ref2.ar_step(model, cache, toks[s_]))
    CAP["out"], CAP["on"] = {}, True
    with ref2.verify_scope(model):
        lg = model(mx.array([toks[m:m + R]]), cache=cache)
    mx.eval(lg, list(CAP["out"].values()))
    CAP["on"] = False
    out = {L: o.reshape(-1, o.shape[-1]) for L, o in CAP["out"].items()}
    P = len(ids)
    rows_ok, first = [], {}
    for j in range(R):
        ok = eqb(lg[0, j], lgs[m + j])
        rows_ok.append(ok)
        first[j] = next((L for L in range(NL) if not eqb(out[L][j], lays[m + j][L])), None)
    by_layer = {}
    for j, L in first.items():
        if L is not None:
            by_layer.setdefault(L, []).append(j)
    say(f"  {tag}: check at absolute {P + m} (keys {P + m + R}), rows' AR steps at keys {P + m + R}..{P + m + 2 * R - 1}: "
        f"logits rows bitwise {sum(rows_ok)} / {R}: {''.join('=' if x else 'x' for x in rows_ok)}")
    for L in sorted(by_layer):
        say(f"    first differing layer {L} ({kind(L)}): rows {by_layer[L]}")
    if not by_layer:
        say("    every layer output of every row equal")
    d = [float(mx.max(mx.abs(lg[0, j].astype(mx.float32) - lgs[m + j].reshape(-1).astype(mx.float32))).item())
         for j in range(R)]
    say(f"    max |logit delta| per row: {[round(x, 4) for x in d]}")
    del cache
    mx.clear_cache()

items = {}
for s_ in ("math", "humaneval"):
    items.update({it["key"]: it for it in ED.load("dec_" + s_)})
K702, K68 = "math500/test/geometry/702.json", "humaneval/HumanEval/68"

ids = list(items[K702]["prompt_ids"])
toks, lgs, lays = ar(ids, R, set(range(R)))
check(ids, toks, 0, lgs, lays, f"A {K702} (P {len(ids)})")

ids = list(items[K68]["prompt_ids"])
P = len(ids)
m1, m2 = 500 - P, 520 - P
toks, lgs, lays = ar(ids, m2 + R, set(range(R)) | set(range(m1, m1 + R)) | set(range(m2, m2 + R)))
check(ids, toks, 0, lgs, lays, f"B {K68} (P {P}, control)")
check(ids, toks, m1, lgs, lays, f"C1 {K68}")
check(ids, toks, m2, lgs, lays, f"C2 {K68}")
say(f"R-40 STEP 4a done in {time.time() - t0:.0f} s")
open(os.path.join(D, "r40s4a_screen.txt"), "w").write("\n".join(lines) + "\n")
