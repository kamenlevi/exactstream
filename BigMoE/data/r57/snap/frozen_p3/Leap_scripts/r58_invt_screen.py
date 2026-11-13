"""R-58 INV-T (Leap, pre-registered Fri 2026-10-30 16:20 EET in ~/Leap/TABLE_350.md 'R-58 REF2-T'; NewIdeas GO 10:20): is REF2
on BigMoE's rank-tier engine (BIGMOE_TIER=4433333333, TIER_GLU 3c; the 45 GB line's Kamen-adopted plan) batch-invariant?
ref2.py's REF2-T branch runs 5-16-row calls on the tier pools with MoonShots' fused 3c kernel (base-pool slots for every
rank, LSB-sidecar slots for ranks 0-1); 1-4-row calls (the AR) take BigMoE's gated tier_call with the same kernel.
STEP 4a's method: REF2's AR, then from a fresh prefill one 16-row check whose drafts are the AR's own tokens; per row
logits bitwise and the first differing DecoderLayer.
  B: HumanEval/68 (P 367), the check right after the prefill (keys 383).
  A: geometry/702 (P 520), the check right after the prefill (keys 536, REF2b's key buckets).
  control: B with CFG["tiers"] off (the check's experts then run BigMoE's exact 4-bit prefill path): must differ.
PASS = 16/16 rows and every layer equal in A and B, tier calls counted on both paths, AND the control differs.
KILL = any row differs in A or B. EvalSuite's REF harness at C 192 (the 4-bit slot budget, split into tier pools).
Usage: [BigMoE venv] r58_invt_screen.py [C=192] -> ~/Leap/data/r58_invt_screen.txt"""
import os, sys, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)
os.environ["BIGMOE_TIER"] = "4433333333"                   # R-58: the 45 GB line's tier plan
os.environ["BIGMOE_TIER_GLU"] = "3c"
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
import numpy as np
import mlx.core as mx

D = os.path.expanduser("~/Leap/data")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 192
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
from bigmoe import tier as T
say(f"R-58 INV-T: full 125B via load_streamed with BIGMOE_TIER {T.PLAN} / GLU {T._GLU} (C = {C}), loaded in "
    f"{time.time() - t0:.0f} s; REF2 defaults {ref2.CFG}")
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
    return sum(rows_ok), len(by_layer)

items = {}
for s_ in ("math", "humaneval"):
    items.update({it["key"]: it for it in ED.load("dec_" + s_)})
K702, K68 = "math500/test/geometry/702.json", "humaneval/HumanEval/68"
RES = {}
def case(key, tag):
    ids = list(items[key]["prompt_ids"])
    ref2.reset_counts(); T.STATS["calls"] = 0
    toks, lgs, lays = ar(ids, R, set(range(R)))
    ar_tier = T.STATS["calls"]
    ref2.reset_counts()
    ok_rows, n_bad_layers = check(ids, toks, 0, lgs, lays, f"{tag} {key} (P {len(ids)})")
    say(f"    tier calls: AR (gated tier_call) {ar_tier}; check (REF2-T slot branch) {ref2.COUNTS['tier_calls']}")
    RES[tag] = (ok_rows, n_bad_layers, ar_tier, ref2.COUNTS["tier_calls"])
case(K68, "B")
case(K702, "A")
ref2.CFG["tiers"] = False
case(K68, "control (tiers branch off)")
ref2.CFG["tiers"] = True
exact = all(RES[t_][0] == R and RES[t_][1] == 0 and RES[t_][2] > 0 and RES[t_][3] > 0 for t_ in ("A", "B"))
ctrl = RES["control (tiers branch off)"]
v = "KILL" if not exact else "VOID (the control did not differ)" if ctrl[0] == R and ctrl[1] == 0 else "PASS"
say(f"R-58 INV-T verdict (PASS: A and B 16/16 rows + every layer equal on tiers, and the control differs): {v}; {RES}")
say(f"R-58 INV-T done in {time.time() - t0:.0f} s")
open(os.path.join(D, "r58_invt_screen.txt"), "w").write("\n".join(lines) + "\n")
