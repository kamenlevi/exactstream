"""R-40 STEP 3t (Leap, pre-registered Thu 2026-10-29 22:15 EET in ~/Leap/TABLE_350.md 'R-40 STEP 3t'): where does REF2's
live loop leave its AR? STEP 3u: with or without the drafter, the loop's kept rows leave REF2's AR bit-identity at the
check that starts at relative position ~40-41 on prealgebra/1640 (U0: check #7 after a keep-15 commit; U3: check #10
after a keep-2 commit), and every later check differs.
Harness: EvalSuite's REF harness, C 128, PF1 for prefills only, REF2 + SB on, no drafter, no taps; U3's synthetic
drafts (m correct AR tokens, the rest shifted), greedy accept, commit keep accepted + 1.
  AR: prefill + 64 REF2 AR steps. After each step, per-layer records of its row (GDN, attention, MoE, PLE, HC read,
      layer output: sha1 of the row's bytes); at every position where the loop will start a check, a digest of all
      48 layers' cache state (3w's).
  Loop: at each check start, the same cache digest vs the AR's at that position (the first check whose START state
      differs, and its layers / slots); each check's per-layer records for every row; at the first check with a
      kept row that differs from the AR logits, the earliest differing record of that row vs the AR step's.
A measurement; no lines.
Usage: [BigMoE venv] r40s3t_screen.py [C=128] -> ~/Leap/data/r40s3t_screen.txt"""
import hashlib, os, sys, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
import numpy as np
import mlx.core as mx

D = os.path.expanduser("~/Leap/data")
C = min(int(sys.argv[1]) if len(sys.argv) > 1 else 128, 128)   # memory (3y-fix's second attempt); C-independent
R, NAR = 16, 64
MS = (3, 0, 6, 1, 9, 2, 4, 0, 5, 1, 7, 2, 3, 6)          # = STEP 3u's U2 / U3
KEY = "math500/test/prealgebra/1640.json"
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
eqb = lambda a, b: a.shape == b.shape and bool(mx.all(mx.view(a.reshape(-1), mx.uint8) == mx.view(b.reshape(-1), mx.uint8)).item())
sha = lambda x: hashlib.sha1(np.array(mx.view(mx.contiguous(x).reshape(-1), mx.uint8)).tobytes()).hexdigest()[:16]

t0 = time.time()
rt, st = EM.load(EM.variant("ref"), C=C)
model = rt.model
import ref2
import bigmoe.stream as BS
from mtplx import cache_state as CS
Q = ref2.Q
ref2.apply()
ref2.CFG["slot_call"] = False   # 3u's harness: BigMoE's gather path (R-50 is bit-equal, but keep it fixed)
Q_mid = Q._prefill_midloop_eval_layers
Q._prefill_midloop_eval_layers = lambda rows: Q_mid(rows) or (4 if 2 <= int(rows) <= 16 else 0)   # SB
lay = model.language_model.model.layers
KIND = ["ple" if "ple" in l else ("gdn" if l.is_linear else "qsa") for l in lay]
say(f"R-40 STEP 3t: full 125B via load_streamed (C = {C}), loaded in {time.time() - t0:.0f} s; REF2 + SB on; {KEY}")

REC = {"on": False, "r": []}
def hook(klass, part):
    orig = klass.__call__
    def call(self, *a, **kw):
        y = orig(self, *a, **kw)
        if REC["on"]:
            o = y if isinstance(y, mx.array) else y[0]
            REC["r"].append((part, o.reshape(-1, o.shape[-1])))
        return y
    klass.__call__ = call
for klass, part in ((Q.GatedDeltaNet, "gdn"), (Q.Attention, "attn"), (Q.SparseMoeBlock, "moe"), (Q.PLELayer, "ple"),
                    (Q.GatedResidual, "hc"), (Q.DecoderLayer, "layer")):
    hook(klass, part)

def pf1(on):
    if not on and BS._STAGE["bufs"] is not None:
        BS._stage_drop()
    BS._PF_PREFILL = on

def states(cache):
    return [c.state for c in cache if c is not None and hasattr(c, "state")]

def prefill(ids):
    pf1(True)
    cache = model.make_cache()
    lg = model(mx.array([ids]), cache=cache)
    mx.eval(lg, states(cache))
    pf1(False)
    return cache, lg[0, -1]

def digest(cache):
    out = []
    for c in cache:
        row = []
        for x in (c.state if c is not None and hasattr(c, "state") else ()):
            row.append(None if x is None else (tuple(x.shape), sha(x)))
        out.append(row)
    return out

def diff(a, b):
    res = []
    for i, (ra, rb) in enumerate(zip(a, b)):
        bad = [j for j in range(max(len(ra), len(rb))) if (ra[j] if j < len(ra) else "-") != (rb[j] if j < len(rb) else "-")]
        if bad:
            res.append((i, KIND[i], bad))
    return res

def rows_records(rec, row):
    """per record: (part, sha of the given row) -- records hold every row of the call"""
    return [(p, sha(o[row])) for p, o in rec]

items = {it["key"]: it for it in ED.load("dec_math") if it["key"] == KEY}
ids = list(items[KEY]["prompt_ids"])

starts, a = [], 0                                        # the loop's check starts if it stays exact
for m in MS:
    starts.append(a)
    a += m + 1

# ---------------------------------------------------------------- the AR: logits, per-step records, digests at starts
cache, last = prefill(ids)
toks, ar_lg, ar_rec, ar_dig = [int(mx.argmax(last).item())], [], [], {}
if 0 in starts:
    ar_dig[0] = digest(cache)
for s in range(NAR):
    REC["r"] = []; REC["on"] = True
    lg = ref2.ar_step(model, cache, toks[-1])
    mx.eval(lg)
    REC["on"] = False
    ar_rec.append(rows_records(REC["r"], 0))             # the step's one real row
    ar_lg.append(lg.reshape(-1))
    toks.append(int(mx.argmax(lg).item()))
    if s + 1 in starts:
        ar_dig[s + 1] = digest(cache)                     # state after toks[0 .. s]
REC["r"] = []
del cache
mx.clear_cache()
say(f"  AR: {NAR} steps; {len(ar_rec[0])} records per step; digests at the loop's check starts {sorted(ar_dig)}")

# ---------------------------------------------------------------- the loop
cache, last = prefill(ids)
a, tok, first_state, first_row = 0, toks[0], None, None
for n_c, m in enumerate(MS):
    if a + R + 1 > NAR:
        break
    if a in ar_dig and first_state is None:
        d = diff(ar_dig[a], digest(cache))
        if d:
            first_state = (n_c, a, d[:10])
    drafts = toks[a + 1:a + 1 + m] + [toks[a + 1 + j + 1] for j in range(m, R - 1)]
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    REC["r"] = []; REC["on"] = True
    with ref2.verify_scope(model):
        lg = model(mx.array([[tok] + drafts]), cache=cache)
    am = mx.argmax(lg[0], axis=-1)
    mx.eval(lg, am)
    REC["on"] = False
    am = am.tolist()
    k = 0
    while k < R - 1 and drafts[k] == am[k]:
        k += 1
    bad = [j for j in range(k + 1) if not eqb(lg[0, j], ar_lg[a + j])]
    if bad and first_row is None:
        r = bad[0]
        mine, ref = rows_records(REC["r"], r), ar_rec[a + r]
        e = next((i for i in range(min(len(mine), len(ref))) if mine[i] != ref[i]), None)
        where = None if e is None else (e, mine[e][0], sum(1 for x in range(e) if mine[x][0] == "layer"))
        first_row = (n_c, a, k + 1, bad, where, len(mine), len(ref))
    REC["r"] = []
    model.commit_verified_window(cache, snap.states, keep_tokens=k + 1, verified_tokens=R)
    mx.async_eval(states(cache))
    say(f"  check #{n_c}: start {a} (absolute {len(ids) + a}), kept {k + 1}, differing kept rows {bad}")
    a += k + 1
    tok = am[k]
    if tok != toks[a]:
        say(f"  the loop's next token left the AR's at relative position {a}")
        break
say("  first check whose START state differs from the AR's: "
    + (f"#{first_state[0]} at relative {first_state[1]}: layers {first_state[2]}" if first_state else "none"))
say("  first check with a differing kept row: "
    + (f"#{first_row[0]} at relative {first_row[1]} (kept {first_row[2]}, rows {first_row[3]}); its row {first_row[3][0]}'s "
       f"earliest differing record vs the AR step: "
       + (f"record {first_row[4][0]} ({first_row[4][1]}, layer {first_row[4][2]})" if first_row[4] else "none (all equal)")
       + f"; records {first_row[5]} vs {first_row[6]}" if first_row else "none"))
say(f"R-40 STEP 3t done in {time.time() - t0:.0f} s")
open(os.path.join(D, "r40s3t_screen.txt"), "w").write("\n".join(lines) + "\n")
