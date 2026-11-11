"""Checks of the NEVER-WAIT masks (Fable FAR, Thu 5 Nov 2026; CPU, numpy, seconds; reads only).
1. Twins: this build's twin masks (nw_masks.py without a _nw suffix) equal, bit for bit, NewIdeas' own builder's output
   (masks_orig/, run unmodified) and, at the published pool, the snapshots Breakthrough ran on Thursday (out/ni/).
2. Each never-wait arm against its twin, per window:
   shapes = the routing capture's; keep == the twin's keep everywhere but the tail layers (>= 48 - d), where it only
   drops ranks >= 1; late is a subset of keep; no late flag at a tail layer or on a prompt row (rows < a; the first check's anchor is row a - 1, which a split
   arm may cut but which is never late: the prefill has just touched its experts); r1: no late rank 0; nw0: keep == the twin's exactly.
3. Family relations at one pool: nw1's late == nw0's late below layer 47; nw2's == nw0's below 46; nw1r1's and nw1s's
   late are subsets of nw1's.
Writes nothing. Usage: python -I -B check_masks.py [<masks dir>]   (exit 0 = ALL OK)"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "masks")
CK = os.path.join(HERE, "masks_check"); ORIG = os.path.join(HERE, "masks_orig")
DMO = os.path.expanduser("~/FableGod/decision_map/out"); NI = os.path.expanduser("~/NewIdeas/data/dm_masks")
base = json.load(open(os.path.join(DMO, "baseline.json")))
bad = []
def ok(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c: bad.append(m)
def wins(d): return sorted(f for f in os.listdir(d) if f.endswith(".npz"))
def same_keep(a, b):
    na, nb = wins(a), wins(b)
    if na != nb or len(na) != 36: return False, len(na)
    return all(np.array_equal(np.load(os.path.join(a, n))["keep"], np.load(os.path.join(b, n))["keep"]) for n in na), len(na)

print("1. twins against their references")
for arm, refs in (("qom_0_live_ball_c208", [(ORIG, "NewIdeas' builder, unmodified")]),
                  ("qom_0_live_ball_c266", [(ORIG, "NewIdeas' builder, unmodified")]),
                  ("qom_0_split0.99_live_ball_c208", [(ORIG, "NewIdeas' builder, unmodified")]),
                  ("qom_0_live_ball", [(os.path.join(DMO, "ni"), "the snapshot run Thu"), (NI, "NewIdeas' masks")]),
                  ("qom_0_split0.99_live_ball", [(os.path.join(DMO, "ni"), "the snapshot run Thu"), (NI, "NewIdeas' masks")]),
                  ("qom_0_split0.9_live_ball", [(os.path.join(DMO, "ni"), "the snapshot run Thu"), (NI, "NewIdeas' masks")])):
    mine = os.path.join(MD, arm) if os.path.isdir(os.path.join(MD, arm)) else os.path.join(CK, arm)
    if not os.path.isdir(mine): print(f"  --   {arm}: not built here"); continue
    for rd, what in refs:
        r = os.path.join(rd, arm)
        if not os.path.isdir(r): print(f"  --   {arm}: no reference at {r}"); continue
        e, n = same_keep(mine, r); ok(e, f"{arm}: keep == {what} on {n}/36 windows, bit for bit")

ARMS = [("qom_0_live_ball_c208_nw1", "qom_0_live_ball_c208", 1, ""), ("qom_0_live_ball_c208_nw1r1", "qom_0_live_ball_c208", 1, "r1"),
        ("qom_0_live_ball_c208_nw0", "qom_0_live_ball_c208", 0, ""), ("qom_0_live_ball_c208_nw2", "qom_0_live_ball_c208", 2, ""),
        ("qom_0_live_ball_c208_nw1s", "qom_0_live_ball_c208", 1, "s"), ("qom_0_live_ball_nw1", "qom_0_live_ball", 1, ""),
        ("qom_0_live_ball_c266_nw1", "qom_0_live_ball_c266", 1, ""),
        ("qom_0_split0.99_live_ball_nw1", "qom_0_split0.99_live_ball", 1, ""),
        ("qom_0_split0.99_live_ball_c208_nw1", "qom_0_split0.99_live_ball_c208", 1, ""),
        ("qom_0_split0.9_live_ball_nw1", "qom_0_split0.9_live_ball", 1, "")]
def twin_dir(t):
    for d in (os.path.join(MD, t), os.path.join(DMO, "ni", t), os.path.join(CK, t)):
        if os.path.isdir(d): return d
print("2. each never-wait arm against its twin")
for arm, twin, d, var in ARMS:
    A = os.path.join(MD, arm)
    if not os.path.isdir(A): print(f"  --   {arm}: not built"); continue
    T = twin_dir(twin); names = wins(A)
    c = dict(shape=0, body=0, tail=0, sub=0, notail=0, prompt=0, r0=0, exact=0); nl = nk = nd = 0; dom = {}
    for n in names:
        z = np.load(os.path.join(A, n)); kp, lt = z["keep"], z["late"]; tk = np.load(os.path.join(T, n))["keep"]
        rt = np.load(os.path.join(DMO, "routing", n))["ids"]; a = base[n]["a"]; t0 = 48 - d
        c["shape"] += kp.shape == lt.shape == tk.shape == rt.shape and kp.dtype == lt.dtype == np.bool_
        c["body"] += np.array_equal(kp[:, :t0], tk[:, :t0])
        c["tail"] += (not (kp[:, t0:] & ~tk[:, t0:]).any()) and np.array_equal(kp[:, t0:, 0], tk[:, t0:, 0])
        c["sub"] += not (lt & ~kp).any()
        c["notail"] += not lt[:, t0:].any()
        c["prompt"] += (not lt[:a].any()) and bool(kp[:max(a - 1, 0)].all())
        c["r0"] += not lt[:, :, 0].any()
        c["exact"] += np.array_equal(kp, tk)
        nl += int(lt.sum()); nk += int(kp.sum()); nd += int((tk & ~kp).sum())
        x = dom.setdefault(base[n]["dom"], [0, 0]); x[0] += int(lt[a:].sum()); x[1] += kp.shape[0] - a
    N = len(names)
    ok(N == 36 and c["shape"] == N, f"{arm}: {N}/36 windows, {c['shape']} with the routing's shape (keep and late bool [rows, 48, 10])")
    ok(c["body"] == N and c["tail"] == N, f"{arm}: keep == {twin}'s below layer {48 - d} ({c['body']}/{N}); at the tail only ranks >= 1 "
       f"are dropped ({c['tail']}/{N}; {nd} terms dropped in all)")
    ok(c["sub"] == N and c["notail"] == N and c["prompt"] == N, f"{arm}: late within keep ({c['sub']}), none at the tail ({c['notail']}), "
       f"none on prompt rows and the rows before the first check's anchor keep all ({c['prompt']}); {nl} late flags")
    if var == "r1": ok(c["r0"] == N, f"{arm}: no late rank 0 ({c['r0']}/{N})")
    if d == 0: ok(c["exact"] == N, f"{arm}: keep == the twin's exactly ({c['exact']}/{N})")
    print("       late terms per answer row (answer rows = the measured positions): " +
          ", ".join(f"{k} {v[0] / max(v[1], 1):.2f}" for k, v in dom.items()))

print("3. family relations at the 208-slot pool")
F = {k: os.path.join(MD, "qom_0_live_ball_c208_" + k) for k in ("nw0", "nw1", "nw2", "nw1r1", "nw1s")}
if all(os.path.isdir(v) for v in F.values()):
    r = dict(a=0, b=0, c=0, d=0); names = wins(F["nw0"])
    for n in names:
        L = {k: np.load(os.path.join(v, n))["late"] for k, v in F.items()}
        r["a"] += np.array_equal(L["nw1"][:, :47], L["nw0"][:, :47]); r["b"] += np.array_equal(L["nw2"][:, :46], L["nw0"][:, :46])
        r["c"] += not (L["nw1r1"] & ~L["nw1"]).any(); r["d"] += not (L["nw1s"] & ~L["nw1"]).any()
    ok(r["a"] == 36 and r["b"] == 36, f"nw1's late == nw0's below layer 47 ({r['a']}/36); nw2's == nw0's below layer 46 ({r['b']}/36)")
    ok(r["c"] == 36 and r["d"] == 36, f"nw1r1's late within nw1's ({r['c']}/36); nw1s's late within nw1's ({r['d']}/36)")
else: print("  --   the nw0 / nw1 / nw2 / nw1r1 / nw1s family is not complete here")
print("ALL OK" if not bad else f"FAILED: {len(bad)}", flush=True)
sys.exit(1 if bad else 0)
