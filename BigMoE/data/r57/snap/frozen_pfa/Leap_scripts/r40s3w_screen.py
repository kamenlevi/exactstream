"""R-40 STEP 3w (Leap, pre-registered Thu 2026-10-29 18:55 EET in ~/Leap/TABLE_350.md 'R-40 STEP 3w'): which part of a
PARTIAL-accept commit leaves state that REF2's AR does not? STEP 3z: REF2 checks are bit-identical to REF2's AR after a
prefill or full accepts, but not after partial commits (prealgebra/1640: 40 / 48 rows); REF2's AR never runs a replay
(its 1-row commit is a full accept).
Harness as STEP 3z (EvalSuite's REF harness, C = 176, PF1 for prefills only, REF2 on throughout, no drafter).
prealgebra/1640 (P 62): REF2 AR for 17 steps from one prefill, every layer's cache state digested after k steps. Then,
for each k in K, a fresh prefill, ONE 16-row REF2 check of the AR's own first 16 tokens (so every row is 'accepted'),
commit keep k, and the same digests; plus one ar_step probe from that state vs the AR's step k + 1 logits.
Per layer the state is (conv, recurrent) for GDN layers, all slots for the PLE layer (zero-based 1), (keys, values,
raw indexer keys, pooled blocks) for QSA layers; each array is compared bitwise (sha1 of its bytes).
A measurement; no lines. k = 16 (a full accept, no replay) is the control.
Usage: [BigMoE venv] r40s3w_screen.py [C=176] -> ~/Leap/data/r40s3w_screen.txt"""
import hashlib, json, os, sys, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
import numpy as np
import mlx.core as mx

D = os.path.expanduser("~/Leap/data")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 176
R = 16
K = (2, 5, 8, 12, 15, 16)
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
lay = model.language_model.model.layers
KIND = ["ple" if "ple" in l else ("gdn" if l.is_linear else "qsa") for l in lay]
say(f"R-40 STEP 3w: full 125B via load_streamed (C = {C}), loaded in {time.time() - t0:.0f} s; layers: "
    f"{KIND.count('gdn')} GDN, {KIND.count('ple')} PLE (zero-based {KIND.index('ple')}), {KIND.count('qsa')} QSA")

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
    """per layer: a sha1 per state slot (None stays None), from the bytes as they are now"""
    out = []
    for c in cache:
        st_ = c.state if c is not None and hasattr(c, "state") else ()
        row = []
        for x in st_:
            if x is None:
                row.append(None)
                continue
            mx.eval(x)
            b = np.array(mx.view(mx.contiguous(x).reshape(-1), mx.uint8))
            row.append((tuple(x.shape), hashlib.sha1(b.tobytes()).hexdigest()[:16]))
        out.append(row)
    return out

def diff(a, b):
    """[(layer, kind, slots that differ)] between two digests"""
    res = []
    for i, (ra, rb) in enumerate(zip(a, b)):
        bad = [j for j in range(max(len(ra), len(rb))) if (ra[j] if j < len(ra) else "-") != (rb[j] if j < len(rb) else "-")]
        if bad:
            res.append((i, KIND[i], bad))
    return res

items = {it["key"]: it for it in ED.load("dec_math") if it["key"] == KEY}
ids = list(items[KEY]["prompt_ids"])
ref2.apply()
Q_mid = Q._prefill_midloop_eval_layers
Q._prefill_midloop_eval_layers = lambda rows: Q_mid(rows) or (4 if 2 <= int(rows) <= 16 else 0)   # SB, as STEP 3z

# ---------------------------------------------------------------- REF2's AR: digests after k steps, logits per step
cache, ar_last = prefill(ids)
toks, ar_lg, ar_dig = [int(mx.argmax(ar_last).item())], [], {}
for s in range(1, R + 2):
    lg = ref2.ar_step(model, cache, toks[-1])
    mx.eval(lg)
    ar_lg.append(lg.reshape(-1))
    toks.append(int(mx.argmax(lg).item()))
    if s in K:
        ar_dig[s] = digest(cache)                    # state after feeding toks[0 .. s-1]
del cache
mx.clear_cache()
row = {d["key"]: d for d in map(json.loads, open(os.path.expanduser("~/EvalSuite/results/ref2/decode.jsonl")))}[KEY]
same_row = sum(1 for a, b in zip(toks, row["completion_ids"]) if a == b)
say(f"  {KEY}, P {len(ids)}: REF2 AR {R + 1} steps; tokens == EvalSuite's REF2 row for {same_row} / {len(toks)} (in order "
    f"up to {next((i for i, (a, b) in enumerate(zip(toks, row['completion_ids'])) if a != b), len(toks))})")

# ---------------------------------------------------------------- one 16-row check + commit keep k, per k
import contextlib
from mtplx.attention_context import attention_phase, model_forward_kind
@contextlib.contextmanager
def verify_phase():      # arm B (added before any run): the commit's replays in the verify forward's phase, which MTPLX's
    with attention_phase("decode_verify"), model_forward_kind("target_verify"):   # HC inject and MoE combine branch on
        yield
for arm, scope in (("A commit as STEP 3z (outside the verify scopes)", contextlib.nullcontext),
                   ("B commit inside attention_phase('decode_verify')", verify_phase)):
    say(f"  arm {arm}:")
    for k in K:
        cache, last2 = prefill(ids)
        pf_same = eqb(last2, ar_last)
        snap = CS.snapshot_untrimmable_cache_lazy(cache)
        with ref2.verify_scope(model):
            lg = model(mx.array([toks[:R]]), cache=cache)
        mx.eval(lg)
        rows_ok = sum(eqb(lg[0, j], ar_lg[j]) for j in range(R))   # row j: the AR's logits after toks[j]
        with scope():
            ok = model.commit_verified_window(cache, snap.states, keep_tokens=k, verified_tokens=R)
            mx.eval(states(cache))
        dg = digest(cache)
        d = diff(ar_dig[k], dg)
        pr = ref2.ar_step(model, cache, toks[k])                    # probe: the next AR step from this state
        mx.eval(pr)
        probe = eqb(pr.reshape(-1), ar_lg[k])
        say(f"    k {k:2d}: prefill == the AR's {pf_same}; check rows == AR {rows_ok} / {R}; commit "
            f"{'landed' if ok else 'REFUSED'}; layers whose state differs from the AR's after {k} steps: {len(d)}"
            + (f" {[(i, kd, sl) for i, kd, sl in d][:12]}" if d else "") + f"; probe step logits == AR: {probe}")
        del cache, snap, lg
        mx.clear_cache()
say(f"R-40 STEP 3w done in {time.time() - t0:.0f} s")
open(os.path.join(D, "r40s3w_screen.txt"), "w").write("\n".join(lines) + "\n")
