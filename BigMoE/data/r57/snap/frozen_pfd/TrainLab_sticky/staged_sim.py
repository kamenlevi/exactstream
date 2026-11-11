"""STAGED (MTP Head, Tue 11-03; the Breakthrough Push, for Breakthrough's staged verification), CPU: kept per check and
exact rows per check when a CHEAP pass picks the path out of DFlash2's top-3 tree and the EXACT pass verifies only that
path. Inputs: data/tree3_map/<set>/<window>.npz (aligned, teacher-forced: exact along the true path up to the first
miss) and one of Breakthrough's decision-map runs as the cheap pass (~/FableGod/decision_map/out/<run>.json: per window
the cheap configuration's top-8 tokens / log-probs at map rows a .. a + R - 1, teacher-forced on the true prefix).
Arms, each its own official loop (next anchor = A + kept):
  chain     DFlash2's own chain, the exact pass verifies all 16 rows
  cut       the shipped default: the own chain cut where the running product of its cf < 0.3
  mcut      the shipped chat drafter: the MTP head's chain cut where the running product of its top-1 p < 0.3
  cat       the cheap pass sees DFlash2's top-1 chain + the top-2 / top-3 siblings at every depth (one tree pass, 46 rows);
            the path follows the top-1 while the cheap pass prefers it among the three; at the first depth where it
            prefers a sibling the path takes it and ends (a sibling has no children in this tree)
  beam      the cheap pass picks among the three at every depth and the path goes on (needs the drafter's candidates
            below the picked token: one cheap step per depth, 3 rows each)
  oracle    the 125B's token whenever it is in the top-3 (the tree's ceiling)
  m*        the same arms on the MTP head's top-3 tree (mtp_t3; the MTP chain is the shipped chat drafter)
  u*        the union of both drafters' top-3 (up to 6 candidates per depth; the top-1 = DFlash2's)
  The cheap pass's pick among the three = the candidate with the highest cheap log-prob (not in its top-8: -inf; none of
  the three in its top-8: DFlash2's top-1). Stop rule (cat / beam): end the path where the cheap pass's best candidate has
  p < theta, so the exact pass verifies fewer rows. Exact rows = the path's drafts + 1.
Usage: [BigMoE venv] staged_sim.py <run> [<run> ...]   (e.g. baseline cert_top2_p09 att_q4). Output: data/staged_sim.txt."""
import os, sys, json, glob
import numpy as np
TD = os.path.expanduser("~/TrainLab/data")
MAPO = os.path.expanduser("~/FableGod/decision_map/out")
THETAS = (0.0, 0.3, 0.5)
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

def loop(win, pick_fn):
    """the official loop over a window's anchors; pick_fn(c) -> (kept, exact_rows, cheap_rows)"""
    A, at = win["A"], {int(a): c for c, a in enumerate(win["A"])}
    a, out = int(A[0]), []
    while a in at:
        c = at[a]
        if int(win["m"][c]) <= 0:
            break
        k, rows, cheap = pick_fn(c)
        out.append((k, rows, cheap))
        a += k
    return out

def run_set(st, run):
    R = json.load(open(os.path.join(MAPO, run + ".json"))) if run else None
    res = {}
    for f in sorted(glob.glob(os.path.join(TD, "tree3_map", st, "*.npz"))):
        key = os.path.basename(f)
        z = dict(np.load(f))
        true, m = z["true"], z["m"]
        own, cf = z["dfl_own"], z["dfl_own_cf"].astype(np.float64)
        TREES = {"": z["dfl_t3"], "m": z["mtp_t3"], "u": np.concatenate([z["dfl_t3"], z["mtp_t3"]], axis=-1)}
        if R is not None:
            rr = R[key]; a0r = int(rr["a"]); tk = np.array(rr["tk"]); lp = np.array(rr["lp"], np.float64)
        def cheap_lp(c, j, d3):
            """the cheap pass's log-prob of the candidates at map row A + j (None if the run has no such row)"""
            row = int(z["A"][c]) + j - a0r
            if row < 0 or row >= len(tk):
                return None
            d = dict(zip(tk[row].tolist(), lp[row].tolist()))
            return np.array([d.get(int(t), -np.inf) if int(t) >= 0 else -np.inf for t in d3[c, j]])
        def first_wrong(path, c):
            k = 0
            while k < len(path) and path[k] == int(true[c, k]):
                k += 1
            return k
        arms = {}
        def chain(c):
            mm = int(m[c]); k = first_wrong([int(x) for x in own[c, :mm]], c); return k + 1, 16, 0
        def cut(c):
            mm = int(m[c]); rows = 1 + int((np.cumprod(cf[c, :mm]) >= 0.3).sum())
            k = min(first_wrong([int(x) for x in own[c, :mm]], c), rows - 1); return k + 1, rows, 0
        def mcut(c):                                         # the shipped chat drafter: the MTP chain cut at running p1 < 0.3
            mm = int(m[c]); p1 = z["mtp_p3"][c, :mm, 0].astype(np.float64)
            rows = 1 + int((np.cumprod(p1) >= 0.3).sum())
            k = min(first_wrong([int(x) for x in z["mtp_t3"][c, :mm, 0]], c), rows - 1); return k + 1, rows, 0
        arms["chain"], arms["cut"], arms["mcut"] = chain, cut, mcut
        for tn, d3 in TREES.items():
            def oracle(c, d3=d3):
                mm = int(m[c]); k = 0
                while k < mm and int(true[c, k]) in d3[c, k].tolist():
                    k += 1
                return k + 1, k + 1, 0
            arms[f"{tn}oracle"] = oracle
            if R is not None:
                for th in THETAS:
                    def cat(c, th=th, d3=d3):
                        mm, path = int(m[c]), []
                        for j in range(mm):
                            l = cheap_lp(c, j, d3)
                            if l is None:
                                break
                            b = int(np.argmax(l)) if np.isfinite(l).any() else 0
                            if th > 0 and np.exp(l[b]) < th:
                                break
                            path.append(int(d3[c, j, b]))
                            if b != 0:
                                break
                        k = first_wrong(path, c); return k + 1, len(path) + 1, 1 + mm * d3.shape[-1]
                    def beam(c, th=th, d3=d3):
                        mm, path = int(m[c]), []
                        for j in range(mm):
                            l = cheap_lp(c, j, d3)
                            if l is None:
                                break
                            b = int(np.argmax(l)) if np.isfinite(l).any() else 0
                            if th > 0 and np.exp(l[b]) < th:
                                break
                            path.append(int(d3[c, j, b]))
                            if path[-1] != int(true[c, j]):
                                break                            # past the first wrong pick nothing is kept
                        k = first_wrong(path, c); return k + 1, len(path) + 1, d3.shape[-1] * len(path) + 1
                    arms[f"{tn}cat t{th}"], arms[f"{tn}beam t{th}"] = cat, beam
        for nm, fn in arms.items():
            v = loop(z, fn)
            r = res.setdefault(nm, [0, 0, 0, 0])
            r[0] += sum(x[0] for x in v); r[1] += sum(x[1] for x in v); r[2] += sum(x[2] for x in v); r[3] += len(v)
    return res

runs = sys.argv[1:] or ["baseline"]
say("STAGED: kept per check / exact rows per check (cheap tree rows) on the tree3 map windows, teacher-forced, each arm its own loop")
for run in runs:
    for st in ("code", "ifeval", "multi"):
        r = run_set(st, run)
        if not r:
            say(f"  {run} {st}: no tree3 files"); continue
        say(f"  cheap = {run}, {st}: " + "; ".join(f"{nm} {v[0] / v[3]:.2f} / {v[1] / v[3]:.1f}" + (f" ({v[2] / v[3]:.0f})" if v[2] else "")
                                                 for nm, v in r.items()))
open(os.path.join(TD, "staged_sim.txt"), "a").write("\n".join(lines) + "\n\n")
