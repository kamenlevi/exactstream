"""R-40 STEP 3v (Leap, pre-registered Thu 2026-10-29 20:05 EET in ~/Leap/TABLE_350.md 'R-40 STEP 3v'): does a partial
commit leak the REJECTED rows' content into the next forward? STEP 3w: one check + commit keep k is exact when rows
k..15 hold the AR's own next tokens; STEP 3z (wrong drafts there) is not. Stale buffer data past the trimmed frontier is
outside the cache state 3w digested, but the next forward may read it.
Harness as STEP 3w (EvalSuite's REF harness, C = 176, PF1 for prefills only, REF2 + SB on throughout, no drafter);
prealgebra/1640 (P 62); REF2's AR for 48 steps from one prefill (tokens, logits).
  Part 1, one cycle: for k in (2, 5, 8, 12):
    clean = prefill + k AR steps; dirty(g) = prefill + ONE 16-row check of toks[:k] + garbage g in rows k..15 + commit
    keep k, g = 'shift' (each row holds the AR token one position later) or 'const' (a fixed token).
    From each state the same 16-row probe check (the AR's tokens k..k+15) with per-layer records (GDN, attention, MoE,
    PLE, HC read, layer output: last row + a checksum of all rows): dirty vs clean rows and records, earliest differing
    record; clean rows vs the AR logits.
  Part 2, many cycles, no drafter: 12 checks; row 0 = the last kept token, then m correct AR tokens and 15 - m wrong
    ones (shift), m from a fixed list; greedy accept, commit keep accepted + 1 (as STEP 3z); kept rows vs the AR logits.
A measurement; no lines.
Usage: [BigMoE venv] r40s3v_screen.py [C=176] -> ~/Leap/data/r40s3v_screen.txt"""
import json, os, sys, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel0/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
import numpy as np
import mlx.core as mx

D = os.path.expanduser("~/Leap/data")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 176
C = min(C, 128)   # Thu 20:08: two loads at C 176 tripped run_locked's +2 GB swap guard (13 GB compressor,
                      # 19:15 and 20:06); a smaller pool cuts the footprint ~6.5 GB (numerics do not depend on C)
R, NAR = 16, 48
KS = (2, 5, 8, 12)
CONST = 1000
MS = (3, 0, 6, 1, 9, 2, 4, 0, 5, 1, 7, 2)                # correct drafts per check in Part 2
KEY = "math500/test/prealgebra/1640.json"
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
eqb = lambda a, b: a.shape == b.shape and bool(mx.all(mx.view(a.reshape(-1), mx.uint8) == mx.view(b.reshape(-1), mx.uint8)).item())

t0 = time.time()
rt, st = EM.load(EM.variant("ref"), C=C)
model = rt.model
import ref2
import bigmoe.stream as BS
from mtplx import cache_state as CS
Q = ref2.Q
ref2.apply()                                             # REF2's patches go in before the record hooks wrap them
Q_mid = Q._prefill_midloop_eval_layers
Q._prefill_midloop_eval_layers = lambda rows: Q_mid(rows) or (4 if 2 <= int(rows) <= 16 else 0)   # SB, as STEP 3z
say(f"R-40 STEP 3v: full 125B via load_streamed (C = {C}), loaded in {time.time() - t0:.0f} s; REF2 + SB on")

REC = {"on": False, "r": []}
def hook(klass, part):
    orig = klass.__call__
    def call(self, *a, **kw):
        y = orig(self, *a, **kw)
        if REC["on"]:
            o = y if isinstance(y, mx.array) else y[0]
            o2 = o.reshape(-1, o.shape[-1])
            dig = mx.sum(mx.view(o2.reshape(-1), mx.uint16).astype(mx.uint32))
            REC["r"].append((part, o2[-1], dig))
            mx.async_eval(o2[-1], dig)
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

def check(cache, rows, rec=False):
    REC["r"] = []; REC["on"] = rec
    with ref2.verify_scope(model):
        lg = model(mx.array([rows]), cache=cache)
    mx.eval(lg)
    REC["on"] = False
    r = REC["r"]; REC["r"] = []
    return lg[0], r

def first_diff(ra, rb):
    for e in range(min(len(ra), len(rb))):
        if ra[e][0] != rb[e][0] or not eqb(ra[e][1], rb[e][1]) or not eqb(ra[e][2], rb[e][2]):
            layer = sum(1 for x in range(e) if ra[x][0] == "layer")
            return f"record {e} ({ra[e][0]}, layer {layer})", sum(1 for x in range(e, min(len(ra), len(rb)))
                                                                 if not (eqb(ra[x][1], rb[x][1]) and eqb(ra[x][2], rb[x][2])))
    return None, 0

items = {it["key"]: it for it in ED.load("dec_math") if it["key"] == KEY}
ids = list(items[KEY]["prompt_ids"])

# ---------------------------------------------------------------- REF2's AR: tokens and logits
cache, last = prefill(ids)
toks, ar_lg = [int(mx.argmax(last).item())], []
for _ in range(NAR):
    lg = ref2.ar_step(model, cache, toks[-1])
    mx.eval(lg)
    ar_lg.append(lg.reshape(-1))
    toks.append(int(mx.argmax(lg).item()))
del cache
mx.clear_cache()
say(f"  {KEY}, P {len(ids)}: REF2 AR {NAR} steps (ar_lg[j] = the logits after toks[j])")

# ---------------------------------------------------------------- Part 1: one cycle, garbage in the rejected rows
for k in KS:
    cache, _ = prefill(ids)
    for j in range(k):
        mx.eval(ref2.ar_step(model, cache, toks[j]))
    lc, rc = check(cache, toks[k:k + R], rec=True)          # clean probe
    clean_vs_ar = sum(eqb(lc[j], ar_lg[k + j]) for j in range(R))
    del cache
    mx.clear_cache()
    for g in ("shift", "const"):
        cache, _ = prefill(ids)
        junk = [toks[j + 1] for j in range(k, R)] if g == "shift" else [CONST] * (R - k)
        snap = CS.snapshot_untrimmable_cache_lazy(cache)
        check(cache, toks[:k] + junk)
        ok = model.commit_verified_window(cache, snap.states, keep_tokens=k, verified_tokens=R)
        mx.eval(states(cache))
        ld, rd = check(cache, toks[k:k + R], rec=True)      # dirty probe
        rows_eq = sum(eqb(ld[j], lc[j]) for j in range(R))
        where, n = first_diff(rc, rd)
        say(f"  k {k:2d} {g:5s}: commit {'landed' if ok else 'REFUSED'}; probe rows == clean {rows_eq} / {R} (clean == AR "
            f"{clean_vs_ar} / {R}); records {len(rd)} vs {len(rc)}" + (f"; earliest differing {where}, {n} differ"
                                                                   if where else "; all records equal"))
        del cache, snap
        mx.clear_cache()

# ---------------------------------------------------------------- Part 2: many cycles, synthetic drafts, no drafter
cache, last = prefill(ids)
a, tok, eq_rows, n_rows, kept = 0, toks[0], 0, 0, []
for m in MS:
    if a + R + 1 > NAR:
        break
    drafts = toks[a + 1:a + 1 + m] + [toks[a + 1 + j + 1] for j in range(m, R - 1)]
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    lg, _ = check(cache, [tok] + drafts)
    am = mx.argmax(lg, axis=-1).tolist()
    k = 0
    while k < R - 1 and drafts[k] == am[k]:
        k += 1
    for j in range(k + 1):                               # rows 0..k: inputs toks[a + j]
        n_rows += 1
        eq_rows += eqb(lg[j], ar_lg[a + j])
    model.commit_verified_window(cache, snap.states, keep_tokens=k + 1, verified_tokens=R)
    mx.async_eval(states(cache))
    kept.append(k + 1)
    a += k + 1
    tok = am[k]
    if tok != toks[a]:
        say(f"  Part 2: the loop's token {tok} left the AR's {toks[a]} at position {a}")
        break
say(f"  Part 2 ({len(kept)} checks, kept {kept}): kept rows bitwise == AR {eq_rows} / {n_rows}")
say(f"R-40 STEP 3v done in {time.time() - t0:.0f} s")
open(os.path.join(D, "r40s3v_screen.txt"), "w").write("\n".join(lines) + "\n")
