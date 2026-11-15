"""INSTANT-INFORMATION SCREEN: context lookup as a free extra proposal (NewIdeas, Sun 15 Nov 22:0x; the orchestrator's
> (a quote from the project owner, kept private)
Prior art (TRIED): D-06 PTR (prompt lookup recovers 10.6 % of first misses on agent text), M-05 SUFX (datastore KILL on
math), LEVER1-COPY-REPLAY (KILL at 16 rows beside DFlash2), L1-10 (PASS, small, agentic). Never measured against t30b's
walk-strict bands on code, which is what the re-aimed 97.8 line gates.
Proposers (draft-time only: the context = the prompt + the answer up to the cell's row, all true tokens on the walk):
  CTX  prompt lookup (PLD): the most recent earlier occurrence of the last n tokens (n = 4, 3, 2, longest first) ->
       the token that followed it;
  CTX1 the same with n = 1 allowed as a last resort;
  DS   the 125B's own past outputs: an n-gram -> most-frequent-next table over the answer tokens of the n1112 captures
       t30b trained on (2,164 files; none of the dev windows), n = 4, 3, 2 longest first;
  CTX+DS  CTX, else DS.
Each cell: 1 if the proposal == the 125B's runtime top-1 at that row (bank tk[row, 0]), 0 if it differs, -1 if none.
ARMS (under certify98's walk, R0 = 1, 32-row checks, the target-less end cells not counted, as tree_audit.py):
  head; head OR proposer (unlimited: any depth, the head's chain continuing from the true prefix: an UPPER BOUND that
  ignores the extra rows); head OR proposer, P = 1 (one rescue per check).
LINE (written before the run): KILL if the unlimited upper bound adds < +1.0 pt walk-strict at EVERY band 1-2 / 3-6 /
7-10 / 11-14 on HE+ AND < +3 % kept/32; else PASS -> a budget-correct Monday screen (the proposal as one extra tree
row).
usage: ~/BigMoE/.venv/bin/python -I -B lookup_screen.py"""
import os, sys, json, glob
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/NewIdeas/research/week98"))
import tree_audit as TA

H = os.path.expanduser("~")
BANK = f"{H}/datasets/deepdraft/bank"
CAP = f"{H}/FableGod/deepdraft/capture/n1112"
SPLIT = f"{H}/BigMoE/data/deepdraft/ttt30b/split.json"
OUT = os.path.expanduser("~/NewIdeas/research/week98/lookup_screen")
BANDS = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]
NS = (4, 3, 2)
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


def ctx_props(ids, n_list):
    """prop[r] = the PLD proposal for position r + 1 from ids[0..r] (-1 if none)"""
    L = len(ids); prop = np.full(L, -1, np.int64); last = {n: {} for n in n_list}
    for r in range(L):
        for n in n_list:                                  # query before inserting the n-gram ending at r
            if r - n + 1 < 0:
                continue
            key = tuple(ids[r - n + 1:r + 1])
            j = last[n].get(key)
            if j is not None and prop[r] < 0:
                prop[r] = ids[j + 1]
        for n in n_list:
            if r - n + 1 >= 0 and r + 1 < L:
                last[n][tuple(ids[r - n + 1:r + 1])] = r
    return prop


def build_ds():
    import json as js
    tr = js.load(open(SPLIT))["train"]
    cnt = {n: {} for n in NS}
    for f in tr:
        z = np.load(os.path.join(CAP, f), allow_pickle=True)
        ids = np.asarray(z["ids"], np.int64); gen = int(z["gen"])
        for n in NS:
            d = cnt[n]
            for r in range(max(gen - 1, n - 1), len(ids) - 1):
                key = tuple(ids[r - n + 1:r + 1]); nx = int(ids[r + 1])
                e = d.get(key)
                if e is None:
                    d[key] = {nx: 1}
                else:
                    e[nx] = e.get(nx, 0) + 1
    best = {n: {k: max(v.items(), key=lambda kv: kv[1])[0] for k, v in cnt[n].items()} for n in NS}
    say(f"  datastore: {len(tr)} files; n-grams {', '.join(f'{n}:{len(best[n])}' for n in NS)}")
    return best


def ds_props(ids, best):
    L = len(ids); prop = np.full(L, -1, np.int64)
    for r in range(L):
        for n in NS:
            if r - n + 1 < 0:
                continue
            v = best[n].get(tuple(ids[r - n + 1:r + 1]))
            if v is not None:
                prop[r] = v; break
    return prop


def walk(X, okfn, P):
    """kept and walk-strict counts; okfn(p, k) -> (head_right, prop_right) for the cell"""
    ks, num, den = [], np.zeros(31), np.zeros(31); p = 1
    while p < X["n"]:
        k = used = 0
        while k < 31:
            if X["rt"][p, k] < 0:
                break
            h, q = okfn(p, k)
            if h:
                ok = True
            elif q and used < P:
                ok = True; used += 1
            else:
                ok = False
            if not ok:
                den[k] += 1; break
            num[k] += 1; den[k] += 1; k += 1
        ks.append(k + 1); p += k + 1
    return ks, num, den


def main():
    TA.say = lambda s="": None
    res = {}
    best = build_ds()
    for s, (tag, rtag, pub32, pub16) in TA.SETS.items():
        W = TA.load_windows(tag, pub32)
        fp = [0, 0]
        for X in W:
            z = np.load(os.path.join(BANK, X["file"]), allow_pickle=True)
            ids = np.asarray(z["ids"], np.int64); tk0 = np.asarray(z["tk"][:, 0], np.int64)
            props = {"CTX": ctx_props(ids, NS), "CTX1": ctx_props(ids, NS + (1,)), "DS": ds_props(ids, best)}
            props["CTX+DS"] = np.where(props["CTX"] >= 0, props["CTX"], props["DS"])
            for name, pr in props.items():
                M = np.full(X["rt"].shape, -1, np.int8)
                for r in range(X["n"] - 1):
                    for d in range(31):
                        if X["rt"][r, d] < 0:
                            continue
                        row = int(X["RW"][r, d])
                        if pr[row] >= 0:
                            M[r, d] = int(pr[row] == tk0[row])
                X[f"LK_{name}"] = M
            for r in range(X["n"] - 1):                    # fingerprint: the head's right flag must agree with tk0 rows
                row = int(X["RW"][r, 0])
                if row + 1 < len(ids):
                    fp[0] += 1; fp[1] += int(ids[row + 1] == tk0[row])
        say(f"\n=== {s} ({tag}) ===  text == the 125B's top-1 at depth-0 rows: {fp[1]} / {fp[0]}")
        r_ = {}
        def arm(name, okfn, P):
            ks, num, den = [], np.zeros(31), np.zeros(31)
            for X in W:
                k, nu, de = walk(X, lambda p, k_, X=X: okfn(X, p, k_), P); ks += k; num += nu; den += de
            ws = [round(100 * num[a:b + 1].sum() / den[a:b + 1].sum(), 2) if den[a:b + 1].sum() else None for a, b in BANDS]
            r_[name] = {"kept32": round(float(np.mean(ks)), 3), "walk_strict": ws}
            f = " ".join(f"{x:5.1f}" for x in ws)
            say(f"  {name:28s} kept/32 {np.mean(ks):6.2f}  walk-strict 0|1-2|3-6|7-10|11-14|15-22|23-30: {f}")
        arm("head", lambda X, p, k: (X["rt"][p, k] == 1, False), 0)
        for nm in ("CTX", "CTX1", "DS", "CTX+DS"):
            cov = []
            for X in W:
                v = X["rt"][:-1] >= 0
                cov.append(((X[f"LK_{nm}"][:-1] >= 0) & v).sum() / max(v.sum(), 1))
            hm = [((X["rt"][:-1] == 0) & (X[f"LK_{nm}"][:-1] == 1)).sum() for X in W]; hw = [(X["rt"][:-1] == 0).sum() for X in W]
            say(f"  [{nm}] proposal on {100 * np.mean(cov):.1f} % of cells; right where the head is wrong: {100 * sum(hm) / max(sum(hw), 1):.1f} % of head misses (marginal)")
            arm(f"head OR {nm} (unlimited, UB)", lambda X, p, k, nm=nm: (X["rt"][p, k] == 1, X[f"LK_{nm}"][p, k] == 1), 99)
            arm(f"head OR {nm} (P=1)", lambda X, p, k, nm=nm: (X["rt"][p, k] == 1, X[f"LK_{nm}"][p, k] == 1), 1)
        res[s] = r_
    say("\nLINE (HE+): KILL if the unlimited UB adds < +1.0 pt walk-strict at every band 1-2 / 3-6 / 7-10 / 11-14 AND < +3 % kept/32")
    hb = res["he"]["head"]
    for nm in ("CTX", "CTX1", "DS", "CTX+DS"):
        u = res["he"][f"head OR {nm} (unlimited, UB)"]
        gains = [u["walk_strict"][i] - hb["walk_strict"][i] for i in (1, 2, 3, 4)]
        kg = u["kept32"] / hb["kept32"] - 1
        v = "PASS" if (max(gains) >= 1.0 or kg >= 0.03) else "KILL"
        say(f"  {nm}: band gains 1-2 / 3-6 / 7-10 / 11-14 {' / '.join(f'{g:+.2f}' for g in gains)}; kept {100 * kg:+.1f} % -> {v}")
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    open(OUT + ".txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
