"""K* FINAL (Tuesday evening deliverable; NewIdeas, written Sun 15 Nov 22:2x and tested on the 32-node dumps).
Inputs:
  - tree dumps in S9c's layout (fi, an, node = best-first pop order, depth, on_true), any number of pops per anchor:
    tonight's ceiling/tree_t30b_{heplus24,mbppbank}.npz (32 pops), Tuesday's 127-node dry run (pass --dump-glob);
  - a costs JSON: {"A16_ms": 62.0, "overhead_ms": 4.0, "lossy32_ms": 25.0, "lossy_floor32_ms": 20.9,
    "R": {"floor": {"16": .., "32": .., ...}, "contract": {...}, "today": {...}}, "c": {"16": 1.025, "32": .., ...}}
    (R = check / A16 for K rows; c = the tree overhead) from Leap's treek_price.md (refined Tue evening).
Output: kept(K) under certify98's walk (K = rows - 1 best-first nodes + the bonus), T vs chain-16 / chain-32, tree depth,
and tok/s = kept / (R x c x A16 + overhead) per cost model, plus lossy (lossy32 x R_floor(K) / R_floor(32) x c), drafting
hidden; then K* per set and cost model, and the margin over the runner-up (a K above 32 is chosen only if >= 3 % better).
usage: ~/BigMoE/.venv/bin/python -I -B treek_final.py --costs costs.json [--dump-dir DIR --dump-fmt tree_t30b_{tag}.npz]"""
import os, sys, json, argparse
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/NewIdeas/research/week98"))
import tree_audit as TA


def kept_by_rows(W, dump, rows_list):
    fi, an, nd, ot = (dump[k] for k in ("fi", "an", "node", "on_true"))
    dp = dump["depth"]
    npops = int(nd.max()) + 1
    out = {}
    for rows in rows_list:
        K = rows - 1
        if K > npops:
            continue
        hit = ot & (nd < K); acc = {}
        for f, a_, d in zip(fi[hit], an[hit], dp[hit]):
            acc[(int(f), int(a_))] = max(acc.get((int(f), int(a_)), 0), int(d) + 1)
        sel = nd < K; dep = {}
        for f, a_, d in zip(fi[sel], an[sel], dp[sel]):
            dep[(int(f), int(a_))] = max(dep.get((int(f), int(a_)), 0), int(d) + 1)
        ks, ds = [], []
        for X in W:
            p = 1
            while p < X["n"]:
                key = (X["fi"], int(X["an"][p])) if p < len(X["an"]) else None
                a = min(acc.get(key, 0), K) if key else 0
                ks.append(a + 1)
                if key:
                    ds.append(dep.get(key, 0))
                p += a + 1
        out[rows] = {"kept": float(np.mean(ks)), "depth_mean": float(np.mean(ds)), "npops": npops}
    return out


def chain_kept(W, cap):
    ks = []
    for X in W:
        p = 1
        while p < X["n"]:
            a = TA.lead_right(X["rt"][p], 0, cap); ks.append(a + 1); p += a + 1
    return float(np.mean(ks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--costs", required=True)
    ap.add_argument("--dump-dir", default=TA.CE)
    ap.add_argument("--dump-fmt", default="tree_t30b_{tag}.npz")
    ap.add_argument("--rows", default="16,24,32,48,64,96,128")
    ap.add_argument("--out", default=os.path.expanduser("~/NewIdeas/research/week98/treek_final"))
    a = ap.parse_args()
    C = json.load(open(a.costs)); A16, OV = C["A16_ms"], C["overhead_ms"]
    rows_list = [int(x) for x in a.rows.split(",")]
    TA.say = lambda s="": None
    lines, res = [], {}
    def say(s=""):
        print(s, flush=True); lines.append(s)
    say(f"costs: {a.costs}; dumps: {a.dump_dir}/{a.dump_fmt}; drafting hidden, overhead {OV} ms")
    for s, (tag, rtag, pub32, pub16) in TA.SETS.items():
        W = TA.load_windows(tag, pub32)
        dump = np.load(os.path.join(a.dump_dir, a.dump_fmt.format(tag=tag)), allow_pickle=True)
        kk = kept_by_rows(W, dump, rows_list)
        c16, c32 = chain_kept(W, 15), chain_kept(W, 31)
        if 16 in kk:
            assert abs(kk[16]["kept"] - pub16) < 0.01, f"{s}: 16-row tree {kk[16]['kept']:.3f} != S9c {pub16}"
        say(f"\n=== {s} ({tag}) === chain-16 {c16:.3f}, chain-32 {c32:.3f}; pops per anchor in the dump: {next(iter(kk.values()))['npops']}")
        tab = {}
        for rows, v in sorted(kk.items()):
            k = v["kept"]; r = str(rows); row = {"kept": round(k, 3), "T16": round(k / c16, 4), "T32": round(k / c32, 4),
                                                   "depth": round(v["depth_mean"], 2)}
            for m, R in C["R"].items():
                if r in R and r in C["c"]:
                    row[m] = round(1000 * k / (R[r] * C["c"][r] * A16 + OV), 1)
            if r in C["R"].get("floor", {}) and "32" in C["R"]["floor"] and r in C["c"]:
                row["lossy"] = round(1000 * k / (C["lossy32_ms"] * C["R"]["floor"][r] / C["R"]["floor"]["32"] * C["c"][r] + OV), 1)
            tab[rows] = row
            say(f"  rows {rows:4d}: kept {k:7.3f}  T16 {k / c16:5.3f}  T32 {k / c32:5.3f}  depth {v['depth_mean']:5.2f}  "
                + "  ".join(f"{m} {row[m]:6.1f}" for m in list(C["R"]) + ["lossy"] if m in row))
        for m in list(C["R"]) + ["lossy"]:
            cand = sorted(((row[m], rows) for rows, row in tab.items() if m in row), reverse=True)
            if not cand:
                continue
            (b, kb), rest = cand[0], cand[1:]
            k32 = tab.get(32, {}).get(m)
            pick = kb if (kb <= 32 or k32 is None or b >= 1.03 * k32) else 32
            say(f"  K* [{m}]: {pick} rows ({tab[pick][m]:.1f} tok/s)" + (f"; runner-up {rest[0][1]} rows {rest[0][0]:.1f} ({100 * (tab[pick][m] / rest[0][0] - 1):+.1f} %)" if rest else ""))
        res[s] = {"chain16": c16, "chain32": c32, "table": tab}
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    open(a.out + ".txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
