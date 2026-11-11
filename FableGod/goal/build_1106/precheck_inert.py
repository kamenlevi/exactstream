"""precheck_inert.py (GOAL session, Thu 2026-11-05): the INERT replay through the ENGINE PATH. CPU only, python -I.
The same functions r57 will call (inert_span.pre_block / classify / inert_k), the engine's own tagger (BigMoE's vendored
relax_core.CodeScan inside InertScan), BigMoE's tokenizer with every token decoded alone, the scan fed the answer from the
first generated token the way r57 feeds it, a copy fed the block's drafts. Data: MTP Head's every-position DFlash2 walks
(~/TrainLab/data/d02b_walk_d23h_dfl_<set>.npz, the 125B's own answers) and Breakthrough's top-8 rows
(~/Breakthrough/data/rulec) for the rows a likelihood rule decides.

What is not on disk is the 125B's output after an accepted deviation. Convention (stated in LOSSY350_GOAL.md 3.3, test A):
the 125B agrees with drafts that stay inside the span and with the draft that ends the span; nothing after that is kept in
the same check; a row a likelihood rule would decide after a deviation is not kept. The walk then resumes at the 125B's
own end of that span. Cost model: 62 ms per check + 2.8 ms per verified row (AUDIT_1105 F2).

usage: python -I precheck_inert.py            the pre-check: CD16 against the lexer replay's x1.17 / x1.63 (+- 0.03)
       python -I precheck_inert.py all        every arm, with and without the doctest guard"""
import importlib.util
import json
import math
import os
import sys

import numpy as np
from tokenizers import Tokenizer

H = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import inert_span as INM                                             # noqa: E402

F0, R0, CUT = 62.0, 2.8, 0.3
TD = f"{H}/TrainLab/data"
TOP8 = {"he12": f"{H}/Breakthrough/data/rulec/rulec_he12.json", "mbpp": f"{H}/Breakthrough/data/rulec/rulec_mbpp.json"}


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


RXC = load_mod("relax_core_bigmoe", f"{H}/BigMoE/scripts/relax_core.py")   # the engine's copy (r57 loads this one)
TKZ = Tokenizer.from_file(f"{H}/BigMoE/models/bare/tokenizer.json")
_dc = {}


def dec(t):
    t = int(t)
    if t not in _dc:
        _dc[t] = TKZ.decode([t])
    return _dc[t]


def samples(dom, top8=None, only=None):
    z = np.load(f"{TD}/d02b_walk_d23h_dfl_{dom}.npz", allow_pickle=False)
    rc = json.load(open(top8)) if top8 else {}
    out = []
    for s in range(len(z["names"])):
        nm = str(z["names"][s])
        if only is not None and (nm in only[0]) != only[1]:
            continue
        out.append(dict(name=nm, ids=[int(t) for t in z[f"{s}_ids"]], gen=int(z[f"{s}_gen"]), ch=z[f"{s}_checks"],
                        dt=z[f"{s}_dtok"], cf=z[f"{s}_dcf"], rc=rc.get(nm)))
    return out


def top8(x, pos, d):
    """(dlm, rank) of token d on the row that predicts position pos, from the 125B's top-8; (-inf, 99) when unknown"""
    r = x["rc"]
    if r is None:
        return -math.inf, 99
    i = pos - int(r["a"]) - 1
    if i < 0 or i >= len(r["tk"]):
        return -math.inf, 99
    tk, lp = r["tk"][i], r["lp"][i]
    if d not in tk:
        return -math.inf, 99
    j = tk.index(d)
    return lp[j] - lp[0], sum(1 for v in lp if v > lp[j])


def replay(S, arm):
    """the official loop over the samples under one arm (None = exact). -> totals"""
    E = T = rows_all = checks = whole = ruled = 0
    by = {"comment": 0, "docstring": 0}
    for x in S:
        ids, gen, ch = x["ids"], x["gen"], x["ch"]
        n = len(ids)
        idx = {int(a): i for i, (k, m, a) in enumerate(ch)}
        a = max(int(ch[:, 2].min()), gen)
        a_first = a
        # the scan after every answer token (r57 feeds the first generated token, then each check's committed tokens)
        sc = INM.InertScan(RXC)
        at = {}
        for p in range(gen, n):
            sc.feed(dec(ids[p]))
            at[p] = sc.copy()
        # the 125B's own text: kind of every position, and whether its token ends that kind's span
        kt = {}
        if arm is not None:
            txt = [dec(t) for t in ids[gen + 1:]]
            for j, (kind, mode, _) in enumerate(INM.classify(RXC, at[gen], txt, txt, dict(arm, com="whole" if arm["com"] else None,
                                                                                       doc="whole" if arm["doc"] else None))):
                kt[gen + 1 + j] = (kind, mode == "whole")
        while a in idx:
            i = idx[a]; m = int(ch[i, 1])
            if m <= 0:
                break
            dr = [int(t) for t in x["dt"][i][:m]]
            cfl = [float(c) for c in x["cf"][i][:m]]
            truth = ids[a + 1:a + 1 + m]
            dtx = [dec(t) for t in dr]
            scn = at[a]
            if arm is not None:
                nk, cfl, _ = INM.pre_block(RXC, scn, dr, cfl, dtx, arm)
                dr, cfl, dtx = dr[:nk], cfl[:nk], dtx[:nk]
            dr = dr[:int((np.cumprod(np.array(cfl)) >= CUT).sum())] if dr else dr     # r57: the running-confidence cut
            dtx = dtx[:len(dr)]
            R = len(dr)
            if arm is None:
                k = 0
                while k < R and dr[k] == truth[k]:
                    k += 1
                got = adv = k + 1
            else:
                pre = INM.classify(RXC, scn, dtx, None, arm)      # the draft side: what the rows are in the draft's own text
                am, dlm, drk = [], [], []
                j0 = je = None
                for j in range(R):
                    if j0 is None:                               # still on the 125B's own text
                        am.append(truth[j])
                        if dr[j] == truth[j]:
                            dlm.append(0.0); drk.append(0)
                        else:
                            dl, rk = top8(x, a + 1 + j, dr[j])
                            dlm.append(dl); drk.append(rk)
                            j0 = j
                    else:
                        dlm.append(-math.inf); drk.append(99)
                        if je is not None:
                            am.append(-1)                        # after the span's end: not on disk
                        elif pre[j][1] == "whole":
                            am.append(dr[j])                     # inside the span: the 125B goes along
                        elif pre[j][0] is not None and pre[j][0] == pre[j0][0] and (
                                "\n" in dtx[j] if pre[j][0] == "comment" else ('"' in dtx[j] or "'" in dtx[j])):
                            am.append(dr[j]); je = j             # the draft ends the span and the 125B agrees
                        else:
                            am.append(-1)
                atx = [dec(t) if t >= 0 else "" for t in am]
                cls = INM.classify(RXC, scn, dtx, atx, arm)
                k, rel = INM.inert_k(dr, am, dlm, drk, cls)
                got = k + 1
                if not rel:
                    adv = got
                    if j0 is not None and k > j0:                 # cannot happen: a deviation is kept only through rel
                        raise AssertionError("kept past an unrelaxed deviation")
                else:
                    r0, kind0, _ = rel[0]
                    by[kind0] += 1
                    ruled += sum(1 for r_ in rel if r_[2] == "rule")
                    whole += sum(1 for j in range(r0, k) if pre[j][1] == "whole")
                    p = a + 1 + r0                                # the 125B's own end of this span
                    while p < n and kt.get(p, (None, False))[0] == kind0 and kt[p][1]:
                        p += 1
                    if je is not None and k > je:
                        adv = max(p + 1 - a, 1)
                    else:
                        adv = min(got, max(p - a, 1))
            E += got; rows_all += R + 1; checks += 1
            a += adv
        T += a - a_first
    ms = checks * F0 + rows_all * R0
    return dict(E=E, T=T, rows=rows_all, checks=checks, ms=ms, whole=whole, ruled=ruled, by=by,
                kept=E / max(checks, 1), tps=E / ms * 1000)


def line(name, x, base):
    return (f"  {name:34s} kept {x['kept']:6.2f}  rows {x['rows'] / x['checks']:5.2f}  tok/s x{x['tps'] / base['tps']:.3f}  "
            f"length T {x['E'] / x['T']:.3f}  kept whole {100 * x['whole'] / x['E']:5.1f} %  by a rule {x['ruled']}  "
            f"deviating checks: comment {x['by']['comment']}, docstring {x['by']['docstring']}")


def main():
    fresh = set(json.load(open(TOP8["he12"])).keys())
    sets = [("HumanEval+ fresh 12", samples("code", TOP8["he12"], (fresh, True)), 1.17),
            ("MBPP 100", samples("mbpp", TOP8["mbpp"]), 1.63),
            ("HumanEval+ all 24 (the 12 map windows have no top-8 here)", samples("code", TOP8["he12"]), None)]
    every = len(sys.argv) > 1 and sys.argv[1] == "all"
    fails = []
    for label, S, want in sets:
        base = replay(S, None)
        print(f"\n=== {label}: {len(S)} answers; exact (0.3 cut): kept {base['kept']:.2f}, rows {base['rows'] / base['checks']:.2f}, "
              f"{base['tps']:.0f} tok/s at 62 + 2.8 per row")
        cd = replay(S, INM.parse_arm("CD16"))
        print(line("CD16 (the engine path)", cd, base))
        if want is not None:
            r = cd["tps"] / base["tps"]
            ok = abs(r - want) <= 0.03
            print(f"  pre-check: x{r:.3f} vs the lexer replay's x{want:.2f} -> {'PASS' if ok else 'OUTSIDE'} (+-0.03)")
            if not ok:
                fails.append((label, round(r, 3), want))
        if every:
            for nm in ("D16", "G16", "C16"):
                print(line(nm, replay(S, INM.parse_arm(nm)), base))
            for nm in ("CD16", "D16"):
                print(line(nm + ", doctest guard off", replay(S, INM.parse_arm(nm, "0")), base))
    print("\n" + ("PRE-CHECK PASS" if not fails else f"PRE-CHECK OUTSIDE THE LINE: {fails}"))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
