"""Fit12 R-62P PRUNE-COLD replay (Sun 1 Nov 2026; NewIdeas QUEUE R-62P; PLAN.md 'R-62P', written before any result).
CPU only (<= 4 threads), on BigMoE's cap45 (the C 160 live REF2b loop at block 16, gate, S45's 6 prompts). Each pass's
kept count k = the rows of its one commit replay at layer 1, else the pass width. Policies, each with its own per-layer
LRU (r57b_replay.serve, validated against miss.bin):
  B16    the passes as captured
  T03    t 0.3 truncation modelled: rows 0..r_t-1, r_t = min(width, k + D_p) (D_p = Leap's t 0.3 rows - kept per check,
         s45c2_c160; HumanEval/68 is never truncated), stochastic rounding with a fixed seed
  ORACLE rows 0..k-1: the kept prefix with its bonus row
  CUT    T03, then layer by layer the pass ends before the first row j >= 1 that needs a non-resident expert there (row 0
         always runs) and row j's non-resident experts at that layer are read async (resident before the next pass, not
         demand). Cut rows < k are true tokens: a redo pass re-checks them next (same routing: same tokens, same prefix),
         under the same rule, so every policy yields the same tokens; only misses and passes differ.
Out: demand misses, async reads, passes, misses per token, a cycle model per token (150 ms per pass + 0.27 ms per demand
miss), at C 192 / 224 (B16 also at C 160 against r57b's 141,883). Usage: [BigMoE venv] r62p_replay.py [CAPDIR]"""
import os
for _k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "4"
import glob, json, sys, time
from collections import OrderedDict
import numpy as np
sys.path.insert(0, "/Users/kamen/Fit12/scripts")
import r57b_replay as R                                  # need_of, serve (the loader's ensure / make_room mirror)

CAP = sys.argv[1] if len(sys.argv) > 1 else "/Users/kamen/BigMoE/data/cap45"
OUT = "/Users/kamen/Fit12/data/r62p"
NL, PASS_MS, MISS_MS = 48, 150.0, 0.27
DP = {"HumanEval_68": None, "prealgebra_1640": 2.81, "HumanEval_91": 2.18, "algebra_694": 2.14, "algebra_1282": 2.39,
      "gsm8k_708": 2.39}                                 # Leap s45c2_c160: t 0.3 rows - kept per check
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def load(cap):
    """ids only (no x): per prompt in process order, the full 48-layer passes with their kept counts, and res0"""
    out = []
    files = glob.glob(os.path.join(cap, "*_gate_*.npz"))
    first = {p: int(np.load(p)["seq"].min()) for p in files}
    for p in sorted(files, key=first.get):
        d = np.load(p)
        seq, L, rows, ro, ids = (d[k] for k in ("seq", "L", "rows", "row_off", "ids"))
        firstc, kept = {}, {}
        for i in range(len(seq)):
            key = (int(seq[i]), int(L[i]))
            if key in firstc:
                if int(L[i]) == 1:
                    kept[int(seq[i])] = int(rows[i])     # the commit's layer-1 replay over the kept rows
            else:
                firstc[key] = i
        by = {}
        for (s, l), i in firstc.items():
            by.setdefault(s, {})[l] = i
        passes = []
        for s in sorted(by):
            ls = by[s]
            if len(ls) != NL or len({int(rows[i]) for i in ls.values()}) != 1:
                continue
            w = int(rows[ls[0]])
            passes.append({"seq": s, "w": w, "k": kept.get(s, w),
                           "ids": [ids[ro[ls[l]]:ro[ls[l]] + w].astype(np.int64) for l in range(NL)]})
        name = os.path.basename(p)
        dp = next((v for key, v in DP.items() if key in name), None)
        out.append({"name": name.replace("_gate_0.npz", ""), "passes": passes, "res0": d["res0"], "dp": dp,
                    "tokens": len(d["tokens"])})
    return out


def init_pools(main, P, C):
    r0 = P["res0"]                                       # as r57b_replay: kept residents keep their order, adoptions front
    for l in range(NL):
        od = OrderedDict((e, True) for e in main[l] if r0[l][e] >= 0)
        for e in np.argsort(r0[l], kind="stable").tolist():
            if r0[l][e] >= 0 and e not in od:
                od[e] = True
        while len(od) > C:
            od.popitem(last=False)
        main[l] = od


def widths(prompts, seed=62):
    """T03's r_t per pass, fixed once (CUT uses the same)"""
    rng = np.random.default_rng(seed)
    for P in prompts:
        for c in P["passes"]:
            if P["dp"] is None or c["w"] < 2:
                c["rt"] = c["w"]
            else:
                d = P["dp"]; extra = int(np.floor(d)) + (1 if rng.random() < d - np.floor(d) else 0)
                c["rt"] = min(c["w"], c["k"] + extra)


def run(prompts, C, policy):
    t = {"passes": 0, "redo": 0, "tokens": 0, "miss": 0, "async": 0, "need": 0}
    per = {}
    main = [OrderedDict() for _ in range(NL)]
    for P in prompts:
        init_pools(main, P, C)
        pend = [set() for _ in range(NL)]
        pm = pt = pp = 0
        for c in P["passes"]:
            if policy == "B16":
                todo = [(0, c["w"], c["k"])]
            elif policy == "ORACLE":
                todo = [(0, min(c["w"], c["k"]), c["k"])]
            else:
                todo = [(0, c["rt"], c["k"])]            # (first row, rows to run, rows of true tokens kept)
            while todo:
                lo, m, k = todo.pop(0)
                for l in range(NL):                      # async reads land before the pass
                    if pend[l]:
                        R.serve(main[l], sorted(pend[l]), C); pend[l] = set()
                rows_ids = [ids[lo:lo + m] for ids in c["ids"]]
                for l in range(NL):
                    if policy == "CUT" and m > 1:
                        od = main[l]
                        for j in range(1, m):
                            nr = [e for e in rows_ids[l][j].tolist() if e not in od]
                            if nr:
                                m = j; pend[l].update(nr); t["async"] += len(nr); break
                    need = R.need_of(rows_ids[l][:m])
                    miss = R.serve(main[l], need, C)
                    t["miss"] += len(miss); t["need"] += len(need); pm += len(miss)
                t["passes"] += 1; pp += 1
                got = min(k, m)
                t["tokens"] += got; pt += got
                if got < k:                              # cut true tokens: re-checked next, same routing
                    t["redo"] += 1
                    todo.insert(0, (lo + got, k - got, k - got))
        per[P["name"]] = (pp, pt, pm)
    return t, per


def main():
    t0 = time.time()
    prompts = load(CAP)
    widths(prompts)
    np_ = sum(len(P["passes"]) for P in prompts)
    say(f"R-62P replay of {CAP}: {len(prompts)} prompts, {np_} passes, kept sum {sum(c['k'] for P in prompts for c in P['passes'])} "
        f"(tokens {sum(P['tokens'] for P in prompts)}); T03 rows per pass by prompt: " + ", ".join(
            f"{P['name'].split('_')[-1]} {np.mean([c['rt'] for c in P['passes']]):.2f} (k {np.mean([c['k'] for c in P['passes']]):.2f})"
            for P in prompts))
    b160, _ = run(prompts, 160, "B16")
    say(f"  check: B16 at C 160 = {b160['miss']} demand misses (r57b_replay's baseline: 141,883)")
    res = {}
    for C in (192, 224):
        say(f"  C {C}:")
        base = None
        for pol in ("B16", "T03", "ORACLE", "CUT"):
            r, per = run(prompts, C, pol)
            res[(C, pol)] = (r, per)
            mpt = r["miss"] / r["tokens"]; cyc = (PASS_MS * r["passes"] + MISS_MS * r["miss"]) / r["tokens"]
            if pol == "T03":
                base = (mpt, cyc, r["passes"] / r["tokens"])
            rel = "" if base is None or pol in ("B16", "T03") else (
                f" | vs T03: misses per token {100 * (mpt / base[0] - 1):+.1f} %, passes per token "
                f"{100 * (r['passes'] / r['tokens'] / base[2] - 1):+.1f} %, cycle per token {100 * (cyc / base[1] - 1):+.1f} %")
            say(f"    {pol:6s}: {r['passes']} passes ({r['redo']} redo), {r['tokens']} tokens, demand misses {r['miss']} "
                f"= {mpt:.1f} per token, async reads {r['async']}, cycle model {cyc:.1f} ms per token{rel}")
        for pol in ("T03", "ORACLE", "CUT"):
            per = res[(C, pol)][1]
            say(f"      {pol:6s} per prompt (passes, tokens, misses per token): " + "; ".join(
                f"{k.split('_')[-1]} {v[0]}, {v[1]}, {v[2] / max(1, v[1]):.1f}" for k, v in per.items()))
    os.makedirs(OUT, exist_ok=True)
    json.dump({f"{C}_{p}": v[0] for (C, p), v in res.items()}, open(os.path.join(OUT, "r62p.json"), "w"))
    open(os.path.join(OUT, "r62p.txt"), "w").write("\n".join(lines) + "\n")
    say(f"  done in {time.time() - t0:.0f} s -> {OUT}/r62p.txt")


if __name__ == "__main__":
    main()
