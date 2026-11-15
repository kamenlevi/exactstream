"""WEEK98 TREE-E2E scorer (MoonShots, Sun 15 Nov 2026), as pre-registered in LEVERS_A_MOONSHOTS.md 'TREE-E2E'.
Reads tree/<arm>_fixed.jsonl (tree_decode.py) and jcc/arm0.jsonl (J(CC)'s batch-1 exact arm). Reports:
  (1) kept/check per set (committed tokens per check, every check incl. the last one cut by the stop, as the sim's walk),
      the T16 / C16 ratio with a paired answer bootstrap (10,000 draws, seed 1115), beside the sim's x1.198 / x1.139;
      like for like with the sim: its keys (MBPP-50 / HE-30, from TrainLab data/sb98/tree_s30_*.npz) and checks that
      start within the first 512 answer tokens;
  (2) exactness: T16 vs C16 token ids per answer (EXACT line), both vs arm0 (the batch-shape noise floor), first
      differing positions (the 2B's top-2 gaps there come from tree_decode.py --gaps, if any);
  (3) pass@1 per arm with EvalSuite's scorers (read-only, via jcc_score.score_all; cached in tree/scores.json), McNemar.
Self-checks: (a) C16 like for like within +-10 % of the sim's chain-16 (MBPP 4.280, HE 5.971); (c) at every check no
root-to-leaf path matches the committed text further than the accepted count, and the accepted path matches exactly
that far. Also tree shape, sibling rescues, accepted-depth histogram, steps / head calls per check, wall time.
T32 run (LEVERS_A 'TREE-E2E T32'): arms c16_fixed_b32,t32_fixed_b32 (both at 32 rows); the sim's line is eagle2-32 vs
chain-16 (MBPP x1.368, HE x1.332); context: T32@32 vs T16@16 kept/check (paired by key) and the 16-vs-32 batch-shape
floor (C16@32 vs C16@16). Writes tree/tree_result_<tree arm>.json for non-default arms.
usage: ~/EvalSuite/.venv/bin/python -I -B tree_score.py [c16_fixed,t16_fixed] [--no-pass]"""
import collections, json, os, sys
import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/MoonShots/week98"))
import jcc_score as JS                                    # noqa: E402  (score_all, line, grp: EvalSuite read-only)

T = H("~/MoonShots/week98/tree")
SIM = {"mbpp": dict(chain16=4.280, tree16=5.128, ratio=1.198, ci=(1.176, 1.219), npz="mbpp"),
       "humaneval": dict(chain16=5.971, tree16=6.802, ratio=1.139, ci=(1.104, 1.175), npz="he")}
SIM32 = {"mbpp": dict(chain16=4.280, tree32=5.856, ratio=1.368, ci=(1.346, 1.390), npz="mbpp"),     # tree_price.txt
         "humaneval": dict(chain16=5.971, tree32=7.951, ratio=1.332, ci=(1.300, 1.357), npz="he")}
EOS = None


def load(path):
    return {r["key"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else {}


def sim_keys(g):
    z = np.load(H(f"~/TrainLab/data/sb98/tree_s30_{SIM[g]['npz']}.npz"))
    out = []
    for f in z["files"]:
        f = str(f)[:-4]
        if f.startswith("mbpp_"):
            out.append("mbpp/" + f[5:])
        else:                                             # humaneval_HumanEval_12 -> humaneval/HumanEval/12
            s, rest = f.split("_", 1)
            out.append(s + "/" + rest.replace("_", "/", 1))
    return out


def kpc(rows, keys, s_max=None):
    ks = nc = 0
    for k in keys:
        for c in rows[k]["checks"]:
            if s_max is None or c["s"] < s_max:
                ks += c["kept"]; nc += 1
    return ks, nc


def boot_ratio(A, Bm, keys, s_max=None, n=10000, seed=1115):
    """paired answer bootstrap of (sum kept / checks of arm A) / (same of arm B) over keys"""
    a = np.array([kpc(A, [k], s_max) for k in keys], float)
    b = np.array([kpc(Bm, [k], s_max) for k in keys], float)
    r0 = (a[:, 0].sum() / a[:, 1].sum()) / (b[:, 0].sum() / b[:, 1].sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n, len(keys)))
    sa, sb = a[idx].sum(1), b[idx].sum(1)
    rs = (sa[:, 0] / sa[:, 1]) / (sb[:, 0] / sb[:, 1])
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return float(r0), float(lo), float(hi)


def first_diff(a, b):
    fd = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if fd is None and len(a) != len(b):
        fd = min(len(a), len(b))
    return fd


def selfcheck_c(r):
    """every check: max over root-to-leaf paths of the prefix matching the committed text == the accepted count"""
    ans, bad = r["ids"], []
    for c in r["checks"]:
        toks, par = c["tree"]
        kids = set(p for p in par if p >= 0)
        best = 0
        for v in range(len(toks)):
            if v in kids:
                continue
            seq, u = [], v
            while u >= 0:
                seq.append(toks[u]); u = par[u]
            seq.reverse()
            mt = 0
            for i, x in enumerate(seq):
                pos = c["s"] + 1 + i
                if pos < len(ans) and ans[pos] == x:
                    mt += 1
                else:
                    break
            best = max(best, mt)
        avail = len(ans) - (c["s"] + 1)
        want = min(c["m"], avail) if c["cut"] else c["m"]
        if best != want:
            bad.append((c["s"], c["m"], best))
    return bad


def main():
    arms = (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "c16_fixed,t16_fixed").split(",")
    do_pass = "--no-pass" not in sys.argv
    A = {a: load(os.path.join(T, a + ".jsonl")) for a in arms}
    a0 = load(H("~/MoonShots/week98/jcc/arm0.jsonl"))
    cn, tn = arms[0], arms[1] if len(arms) > 1 else None
    C, Tt = A[cn], (A[tn] if tn else {})
    keys = sorted(set(C) & set(Tt)) if tn else sorted(C)
    res = {"arms": arms, "keys": len(keys), "kept": {}, "exact": {}, "selfcheck": {}, "shape": {}, "pass": {}}
    grp = lambda k: "mbpp" if C[k]["set"] == "mbpp" else "humaneval"
    big = bool(tn and tn.startswith("t32"))
    SIMX = SIM32 if big else SIM
    # (1) kept/check
    for g in ("humaneval", "mbpp"):
        ks = [k for k in keys if grp(k) == g]
        if not ks:                                        # a partial run (the stop rule) may not reach a set yet
            continue
        sk = [k for k in sim_keys(g) if k in set(ks)]
        row = {"answers": len(ks)}
        for a in arms:
            s_, n_ = kpc(A[a], ks)
            row[a] = dict(kept=s_, checks=n_, kpc=round(s_ / max(n_, 1), 3))
            s5, n5 = kpc(A[a], sk, 512)
            row[a + "_sim_keys_512"] = dict(keys=len(sk), kept=s5, checks=n5, kpc=round(s5 / max(n5, 1), 3))
        if tn:
            r0, lo, hi = boot_ratio(Tt, C, ks)
            row["ratio"] = dict(value=round(r0, 3), ci95=[round(lo, 3), round(hi, 3)])
            r5, lo5, hi5 = boot_ratio(Tt, C, sk, 512)
            row["ratio_sim_keys_512"] = dict(value=round(r5, 3), ci95=[round(lo5, 3), round(hi5, 3)], keys=len(sk))
            row["sim"] = SIMX[g]
            row["gain_survives"] = bool(r0 >= 1.03 and lo > 1.0)
            row["sim_vs_real"] = ("MATCHES SIM" if (lo5 <= SIMX[g]["ratio"] <= hi5 or SIMX[g]["ratio"] < lo5)
                                  else "BELOW SIM")
            row["ci_half_width_pct"] = round(100 * (hi - lo) / 2 / r0, 2)
        c5 = row[cn + "_sim_keys_512"]["kpc"]
        row["selfcheck_a"] = dict(c16_sim_keys_512=c5, sim_chain16=SIM[g]["chain16"],
                                  ratio=round(c5 / SIM[g]["chain16"], 3),
                                  verdict="PASS" if abs(c5 / SIM[g]["chain16"] - 1) <= 0.10 else "FAIL")
        if big:                                           # context: T32@32 vs T16@16, paired by key (no line)
            t16 = load(os.path.join(T, "t16_fixed.jsonl"))
            kk = [k for k in ks if k in t16]
            if kk:
                r1, l1, h1 = boot_ratio(Tt, t16, kk)
                row["t32_vs_t16"] = dict(value=round(r1, 3), ci95=[round(l1, 3), round(h1, 3)], keys=len(kk),
                                         t16_kpc=round(kpc(t16, kk)[0] / max(kpc(t16, kk)[1], 1), 3))
        res["kept"][g] = row
    # (2) exactness
    def cmp(X, Y, ks):
        same = [k for k in ks if X[k]["ids"] == Y[k]["ids"]]
        diff = [dict(key=k, first=first_diff(X[k]["ids"], Y[k]["ids"]), n=(len(X[k]["ids"]), len(Y[k]["ids"])))
                for k in ks if X[k]["ids"] != Y[k]["ids"]]
        return dict(equal=len(same), of=len(ks), differ=diff)
    if tn:
        tvc = "t16_vs_c16" if arms == ["c16_fixed", "t16_fixed"] else f"{tn}_vs_{cn}"   # the tree vs its chain
        res["exact"][tvc] = cmp(Tt, C, keys)
        res["exact"]["EXACT"] = res["exact"][tvc]["equal"] == len(keys)
    ka = [k for k in keys if k in a0]
    for a in arms:
        res["exact"][f"{a}_vs_arm0"] = cmp(A[a], a0, ka)
    if big:                                               # the 16-vs-32 batch-shape floor (no line)
        c16 = load(os.path.join(T, "c16_fixed.jsonl"))
        res["exact"]["c16_32_vs_c16_16"] = cmp(C, c16, [k for k in keys if k in c16])
    if tn:
        d1 = {x["key"] for x in res["exact"][f"{cn}_vs_arm0"]["differ"]}
        d2 = {x["key"] for x in res["exact"][f"{tn}_vs_arm0"]["differ"]}
        res["exact"]["arm0_floor_same_keys"] = d1 == d2
    # self-check (c) and the shape
    for a in arms:
        bad = {k: selfcheck_c(A[a][k]) for k in keys}
        bad = {k: v for k, v in bad.items() if v}
        res["selfcheck"][f"c_{a}"] = dict(checks=sum(len(A[a][k]["checks"]) for k in keys),
                                         bad_checks=sum(len(v) for v in bad.values()), bad_keys=list(bad)[:10])
        ch = [c for k in keys for c in A[a][k]["checks"]]
        sib = [c for c in ch if any(s != 0 for s in c["slots"])]
        hist = collections.Counter(c["m"] for c in ch)
        res["shape"][a] = dict(checks=len(ch), paths=round(float(np.mean([c["paths"] for c in ch])), 2),
                               nodes=round(float(np.mean([c["nodes"] for c in ch])), 2),
                               depth=round(float(np.mean([c["depth"] for c in ch])), 2),
                               steps_per_check=round(float(np.mean([c["steps"] for c in ch])), 2),
                               head_calls_per_check=round(sum(A[a][k]["head_calls"] for k in keys) / len(ch), 2),
                               sibling_rescue_share=round(len(sib) / len(ch), 4),
                               kept_on_sibling_checks=round(float(np.mean([c["kept"] for c in sib])), 2) if sib else None,
                               accepted_hist={int(m): hist[m] for m in sorted(hist)},
                               tokens=sum(A[a][k]["n"] for k in keys), secs=round(sum(A[a][k]["secs"] for k in keys)),
                               cap_hits=sum(A[a][k]["stop"] == "cap" for k in keys))
    # (3) pass@1
    if do_pass:
        p = os.path.join(T, "scores.json")
        sc = json.load(open(p)) if os.path.exists(p) else {}
        for a in arms:
            todo = {k: A[a][k] for k in keys if k not in sc.get(a, {})}
            if todo:
                sc.setdefault(a, {}).update(JS.score_all(todo))
                json.dump(sc, open(p, "w"), indent=0)
        ref0 = json.load(open(H("~/MoonShots/week98/jcc/scores.json")))["arm0"]
        for a in arms:
            res["pass"][a] = {g: round(100 * sum(sc[a][k] for k in keys if grp(k) == g) /
                                       max(1, sum(1 for k in keys if grp(k) == g)), 2) for g in ("humaneval", "mbpp")}
            res["pass"][a]["pooled"] = round(100 * sum(sc[a][k] for k in keys) / len(keys), 2)
            res["pass"][a]["vs_arm0"] = JS.line(ref0, sc[a], ka)
        if tn:
            res["pass"][tvc] = JS.line(sc[cn], sc[tn], keys)
    out = "tree_result.json" if arms == ["c16_fixed", "t16_fixed"] else f"tree_result_{tn or cn}.json"
    json.dump(res, open(os.path.join(T, out), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "exact"}, indent=1))
    ex = res["exact"]
    print("exactness:", {k: (v if isinstance(v, bool) else f"{v['equal']} / {v['of']}") for k, v in ex.items()})
    for k, v in ex.items():
        if isinstance(v, dict) and v["differ"]:
            print(f"  {k}: first diffs (key, position): {[(x['key'], x['first']) for x in v['differ'][:12]]}")


if __name__ == "__main__":
    main()
