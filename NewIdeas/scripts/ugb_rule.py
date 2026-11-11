"""NEW bet UGB (uncertainty-gated budget), kill test with the Eval suite's OWN strict rules (CPU; read-only imports of
evalsuite.report as TrainLab's sticky/rule125.py does; run with PYTHONDONTWRITEBYTECODE=1). Input: TrainLab turn15's per-position
arrays (v3 tiers, 4-row tf passes): arms exact (= v3 tiered floor), b22, b20, b18. A gated arm uses the budget arm's KLD / agree on
passes the gate calls confident and the exact arm's on the others. Passes = consecutive chunks of 4 positions per sequence (the tf
runner's R=4 chunks). Gate signals: ORACLE = the exact arm's own min top-1 log-prob over the pass's rows (bound only);
POST = the budget arm's own min top-1 log-prob (realizable as 'rerun the pass exactly if any row is unsure');
LAG = the exact arm's min top-1 log-prob over the PREVIOUS pass (known before the pass: realizable without a rerun);
DRAFT / DRAFT_EX = the MTP drafter's depth-1 top-1 log-prob per position (TrainLab turn17, UGB_NPZ=..._mtp_pos.npz): known
before the pass, the realizable gate. usage: UGB_NPZ=<npz> ugb_rule.py [ORACLE POST LAG DRAFT DRAFT_EX]
Rules per part vs ar4 at the same positions (90 % cluster CI): mean KLD ratio <= 1.10, P99.9 ratio <= 1.10, top-1 drop >= -0.5 pt.
Pre-registered kill: no threshold passes all four parts with the budget on >= 80 % of code positions -> KILL."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
from evalsuite.report import cluster_boot, quantile_ratio, MARGINS  # noqa: E402
_files = os.environ.get("UGB_NPZ", "~/TrainLab/data/budget125_tf_R4_plan4433333333_pos.npz").split(",")
z = {}                                                       # several npz files (e.g. turn18's v3 arm + turn18c128's arms) merged
for _f in _files:
    _d = np.load(os.path.expanduser(_f))
    for _k in _d.files:
        z.setdefault(_k, _d[_k])
class _Z(dict):
    files = property(lambda self: list(self.keys()))
z = _Z(z)
BASE = os.environ.get("UGB_BASE", "exact")                    # B2 (turn18): UGB_BASE=p4433333333
ARMS = os.environ.get("UGB_ARMS", "b22,b20,b18").split(",")   # B2: UGB_ARMS=p4433222222,p4422222222
HAS_MTP = f"{BASE}__mtp_lp1" in z.files                       # TrainLab turn17 (..._mtp_pos.npz): the drafter's depth-1 top-1 log-prob
a = np.load(os.path.expanduser("~/EvalSuite/results/ar4/tf.npz"))
akey = {(int(s), int(p)): j for j, (s, p) in enumerate(zip(a["seq"], a["pos"]))}
bt, nm = MARGINS["bootstrap"], MARGINS["noninferiority"]
parts = [str(x) for x in z["parts"]]
ex = {f: z[f"{BASE}__{f}"] for f in ("kld", "agree", "lp1", "seq", "pos", "part")}
eidx = {(int(s), int(p)): i for i, (s, p) in enumerate(zip(ex["seq"], ex["pos"]))}

def passes(seq, pos):
    """pass id per position: chunks of 4 consecutive positions within each sequence, from its first scored position"""
    pid = np.empty(len(seq), np.int64)
    for s in np.unique(seq):
        m = np.where(seq == s)[0]
        o = m[np.argsort(pos[m])]
        pid[o] = s * 100000 + (np.arange(len(o)) // 4)
    return pid

def rules(kv, gv, seq, part, pos):
    idx = np.array([akey.get((int(s), int(p)), -1) for s, p in zip(seq, pos)])
    res, ok_all = {}, True
    for pi, pn in enumerate(parts):
        m = (idx >= 0) & (part == pi)
        if not m.any():
            continue
        ka, ga, cl = a["kld"][idx[m]].astype(np.float64), a["agree"][idx[m]].astype(np.float64), seq[m]
        e, lo, hi = cluster_boot(kv[m], ka, cl, bt["resamples"], bt["seed"])
        q, qlo, qhi = quantile_ratio(kv[m], ka, cl, 0.999, bt["quantile_resamples"], bt["seed"])
        d, dlo, dhi = cluster_boot(gv[m], ga, cl, bt["resamples"], bt["seed"], stat="diff")
        ok = hi <= nm["kld_mean_ratio_max"] and qhi <= nm["kld_p999_ratio_max"] and dlo * 100 >= -nm["top1_agree_drop_max_points"]
        ok_all &= ok
        res[pn[3:]] = (round(hi, 2), round(qhi, 2), round(dlo * 100, 1), ok)
    return ok_all, res

for arm in ARMS:
    seq, pos, part = z[f"{arm}__seq"], z[f"{arm}__pos"], z[f"{arm}__part"]
    j = np.array([eidx.get((int(s), int(p)), -1) for s, p in zip(seq, pos)])
    keep = j >= 0
    seq, pos, part, j = seq[keep], pos[keep], part[keep], j[keep]
    kb, gb, lb = z[f"{arm}__kld"][keep].astype(np.float64), z[f"{arm}__agree"][keep].astype(np.float64), z[f"{arm}__lp1"][keep]
    ke, ge, le = ex["kld"][j].astype(np.float64), ex["agree"][j].astype(np.float64), ex["lp1"][j]
    pid = passes(seq, pos)
    _, inv = np.unique(pid, return_inverse=True)
    sel = sys.argv[1:] or ["ORACLE", "POST", "LAG"] + (["DRAFT", "DRAFT_EX"] if HAS_MTP else [])
    sigs = [("ORACLE", le), ("POST", lb), ("LAG", le)]
    if HAS_MTP:                                 # DRAFT: the budget arm's own drafter (budgeted hidden states, as deployed);
        md = z[f"{arm}__mtp_lp1"][keep].astype(np.float64)          # DRAFT_EX: the exact arm's drafter
        mx_ = z[f"{BASE}__mtp_lp1"][j].astype(np.float64)
        sigs += [("DRAFT", np.where(np.isfinite(md), md, -np.inf)), ("DRAFT_EX", np.where(np.isfinite(mx_), mx_, -np.inf))]
    for sig_name, sig in sigs:
        if sig_name not in sel:
            continue
        pmin = np.full(inv.max() + 1, np.inf)
        np.minimum.at(pmin, inv, sig)
        if sig_name == "LAG":                       # realizable before the pass: the PREVIOUS pass's min (same sequence)
            ids = np.unique(pid)                    # sorted pass ids; seq * 100000 + chunk
            prev = np.full(len(ids), -np.inf)       # first pass of a sequence: no history -> natural routing
            same = (ids[1:] // 100000) == (ids[:-1] // 100000)
            prev[1:][same] = pmin[:-1][same]
            pmin = prev
        gate_pos = pmin[inv]                        # each position gets its pass's (or previous pass's) min top-1 log-prob
        print(f"\n== {arm}, gate {sig_name} (per 4-row pass, min top-1 log-prob)")
        best = None
        for thr in [-np.inf] + list(np.quantile(gate_pos, [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])):
            conf = gate_pos >= thr
            ok, res = rules(np.where(conf, kb, ke), np.where(conf, gb, ge), seq, part, pos)
            cov = {pn[3:]: float(conf[part == pi].mean()) for pi, pn in enumerate(parts) if (part == pi).any()}
            tag = "PASS" if ok else "fail"
            print(f"  thr {thr:8.3f} {tag} | coverage " + " ".join(f"{k} {v:.0%}" for k, v in cov.items()) + " | "
                  + " | ".join(f"{k}: mean<= {v[0]} p999<= {v[1]} top1>= {v[2]}" for k, v in res.items()))
            if ok and cov.get("code", 0) >= 0.8 and best is None:
                best = (thr, cov)
        print(f"  -> {'SURVIVES: ' + str(best) if best else 'KILL for this arm / gate (no pass with >= 80 % code coverage)'}")
