"""KEYED SHAPE AT 16 ROWS (NewIdeas, Mon 16 Nov 08:2x). K* is tree-16 at the engine prior (R32 ~1.445), so the two
keys found by THINK_1115 #1 (hard start: lift x4.6 on HE+) and #7 (in comment: 71 % of HE+ near-ties) are tested where
they would be used now: re-selecting EAGLE-2-16's 15 nodes from BigMoE's real 32-pop dumps (TREE-SHAPE's framework,
whose ALWAYS-ON wide policies were a KILL at x0.959-0.995).
Keys at each check start p (draft-time only):
  HARD    = the previous check missed at a row whose 125B margin < 2 nats, or one of its last 3 verified rows had m < 2
            (Fable C's rule; the first check of a window counts as HARD);
  COMMENT = the lexer class of the prefix's last token at the start (slot_prior.py v2's lexer) is "in comment".
Shapes (15 of the 32 pops; every score is monotone along a path, so the top 15 form a closed subtree):
  EAGLE-2 (pop order); DISC l = path_p x l^depth (l 0.8 / 0.6); TEMPER T = the depth-0 edge p0 -> p0^(1/T) (T 2 / 3).
Arms: EAGLE-2 always; each wide shape always; each wide shape keyed on HARD, on COMMENT, on HARD or COMMENT (EAGLE-2
otherwise). Each arm under its own certify98 walk (R0 = 1), 16-row checks.
LINE (written before the run): PASS if the best keyed arm >= EAGLE-2-16 x 1.02 kept on HE+ (MBPP reported).
usage: ~/BigMoE/.venv/bin/python -I -B keyed16.py"""
import os, sys, json
import numpy as np
from tokenizers import Tokenizer
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/NewIdeas/research/week98"))
import tree_audit as TA
import slot_prior as SP

OUT = os.path.expanduser("~/NewIdeas/research/week98/keyed16")
NODES = 15
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


def main():
    TA.say = lambda s="": None
    tok = Tokenizer.from_file(SP.TOKJ)
    res = {}
    for s, (tag, rtag, pub32, pub16) in TA.SETS.items():
        W = TA.load_windows(tag, pub32)
        tz = np.load(f"{TA.CE}/tree_t30b_{tag}.npz", allow_pickle=True)
        fi, an, nd, dp, ot, pp, pa = (tz[k] for k in ("fi", "an", "node", "depth", "on_true", "path_p", "parent"))
        order = np.lexsort((nd, an, fi)); fi, an, nd, dp, ot, pp, pa = (x[order] for x in (fi, an, nd, dp, ot, pp, pa))
        b = np.flatnonzero(np.append(True, (np.diff(fi) != 0) | (np.diff(an) != 0))); b = np.append(b, len(fi))
        POL = {"EAGLE-2": None, "DISC 0.8": ("l", 0.8), "DISC 0.6": ("l", 0.6), "TEMPER 2": ("T", 2.0), "TEMPER 3": ("T", 3.0)}
        acc = {k: {} for k in POL}
        for b0, b1 in zip(b[:-1], b[1:]):
            key = (int(fi[b0]), int(an[b0]))
            n_, d_, o_, p_, a_ = nd[b0:b1], dp[b0:b1], ot[b0:b1].astype(bool), pp[b0:b1].astype(np.float64), pa[b0:b1]
            idx = {int(x): i for i, x in enumerate(n_)}
            p0 = np.empty(len(n_))
            for i in range(len(n_)):
                j = i
                while d_[j] > 0 and int(a_[j]) in idx:
                    j = idx[int(a_[j])]
                p0[i] = p_[j]
            for name, pol in POL.items():
                if pol is None:
                    sel = n_ < NODES
                elif pol[0] == "l":
                    sel = np.zeros(len(n_), bool); sel[np.argsort(-(p_ * pol[1] ** d_), kind="stable")[:NODES]] = True
                else:
                    sel = np.zeros(len(n_), bool)
                    sel[np.argsort(-(p_ * (p0 ** (1 / pol[1])) / np.maximum(p0, 1e-30)), kind="stable")[:NODES]] = True
                hit = o_ & sel
                acc[name][key] = int(d_[hit].max()) + 1 if hit.any() else 0
        # the COMMENT key per anchor row
        com = {}
        for X in W:
            z = np.load(os.path.join(SP.BANK, X["file"]), allow_pickle=True)
            ids = [int(x) for x in z["ids"]]; gen = int(z["gen"]); st = max(gen - 1, 0)
            cls = [None] * st + SP.lex_classes([tok.decode([i]) for i in ids[st:]])
            com[X["fi"]] = {int(a_): (cls[int(a_) + 1] == "in comment") if 0 <= int(a_) + 1 < len(cls) else False for a_ in X["an"]}
        assert abs(np.mean([1]) - 1) < 1e-9
        def run(choose):
            ks = []; nh = nc = 0
            for X in W:
                p, prev = 1, None
                while p < X["n"]:
                    if prev is None:
                        hard = True
                    else:
                        q, qa = prev
                        mm = X["M"][q, qa] if qa < NODES and X["rt"][q, qa] >= 0 else np.inf
                        last = [X["M"][q, j] for j in range(max(0, qa - 3), qa)]
                        hard = (mm < 2) or any((not np.isnan(v)) and v < 2 for v in last)
                    if p < len(X["an"]):
                        a0 = int(X["an"][p]); cm = com[X["fi"]].get(a0, False)
                        pol = choose(hard, cm); a = acc[pol].get((X["fi"], a0), 0)
                        nh += hard; nc += cm
                    else:
                        a = 0
                    ks.append(a + 1); prev = (p, a); p += a + 1
            return float(np.mean(ks)), nh, nc
        say(f"\n=== {s} ({tag}) ===")
        r = {}
        base, nh, nc = run(lambda h, c: "EAGLE-2")
        say(f"  EAGLE-2-16 kept {base:.3f} (S9c {pub16}: {'PASS' if abs(base - pub16) < 0.01 else 'FAIL'}); HARD starts {nh}, COMMENT starts {nc}")
        r["EAGLE-2"] = base
        for w in ("DISC 0.8", "DISC 0.6", "TEMPER 2", "TEMPER 3"):
            for kname, keyf in (("always", lambda h, c: True), ("HARD", lambda h, c: h), ("COMMENT", lambda h, c: c),
                                 ("HARD or COMMENT", lambda h, c: h or c)):
                k, _, _ = run(lambda h, c, w=w, keyf=keyf: w if keyf(h, c) else "EAGLE-2")
                r[f"{w} {kname}"] = k
                say(f"  {w:9s} {kname:16s} kept {k:7.3f}  x{k / base:.4f}")
        best = max((v, k) for k, v in r.items() if k != "EAGLE-2" and not k.endswith("always"))
        say(f"  best keyed: {best[1]} x{best[0] / base:.4f}")
        res[s] = {"kept": r, "best_keyed": best[1], "best_ratio": best[0] / base}
    v = "PASS" if res["he"]["best_ratio"] >= 1.02 else "KILL"
    say(f"\nLINE (HE+ best keyed >= EAGLE-2-16 x 1.02): {v} (HE+ x{res['he']['best_ratio']:.4f} {res['he']['best_keyed']}; "
        f"MBPP x{res['mbpp']['best_ratio']:.4f} {res['mbpp']['best_keyed']})")
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    open(OUT + ".txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
