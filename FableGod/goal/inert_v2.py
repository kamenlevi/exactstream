"""INERT v2 (GOAL, Thu 5 Nov 2026): stress tests of the INERT replay. CPU only, python -I, numpy + tokenizers.
  A  draft-faithful 16-row replay: the wholesale run inside a span ends where the DRAFT's own chain (dtok) ends the span
     (the 125B is assumed to agree with that end), not where the 125B's own text ends it; emitted tokens vs the 125B's
     answer length = the length ratio T.
  B  scratchpad guard: comment tokens by the length of their block of consecutive comment lines; INERT without blocks of
     >= 3 lines.
  C  where the checks and the time go, by the region a check starts in (base vs INERT).
  D  graded rules for in-span rows, profile on the joined windows (the 125B's top-8): rule C (top-3 at >= 0.1 p1),
     top-8, p >= 0.01, wholesale; no cut inside a span.
"""
import json
import math
import sys
import numpy as np
from tokenizers import Tokenizer

G = {}
exec(open("/Users/kamen/FableGod/goal/inert_classes.py").read().split("def main")[0], G)
lex, rows_cut = G["lex"], G["rows_cut"]
D = "/Users/kamen/TrainLab/data/"
R = "/Users/kamen/Breakthrough/data/rulec/"
TOK = sys.argv[1]
F0, R0 = 62.0, 2.8
tok = Tokenizer.from_file(TOK)
_c = {}


def piece(t):
    t = int(t)
    if t not in _c:
        _c[t] = tok.decode([t], skip_special_tokens=False)
    return _c[t]


def ender(p, kind):
    return ("\n" in p) if kind == 1 else any(c in p for c in "\"'")


def load(dom, rcfile=None, only=None):
    z = np.load(D + f"d02b_walk_d23h_dfl_{dom}.npz", allow_pickle=False)
    rc = json.load(open(R + rcfile)) if rcfile else None
    S = []
    for s in range(len(z["names"])):
        nm = str(z["names"][s])
        if only is not None and (nm in only[0]) != only[1]:
            continue
        ids = z[f"{s}_ids"]; gen = int(z[f"{s}_gen"])
        pcs = [piece(t) for t in ids]
        st = lex(pcs, gen)
        n = len(ids)
        kind = np.zeros(n, np.int8)
        for i in range(gen, n):
            kind[i] = 1 if st[i] == "comment" else 2 if st[i] == "docstring" else 0
        # comment blocks: consecutive lines whose text starts with '#'
        line = np.zeros(n, np.int32); ln = 0
        for i in range(n):
            line[i] = ln
            ln += pcs[i].count("\n")
        text_lines = "".join(pcs).split("\n")
        isc = [t.strip().startswith("#") for t in text_lines]
        blen = [0] * len(text_lines)
        i = 0
        while i < len(isc):
            if isc[i]:
                j = i
                while j < len(isc) and isc[j]:
                    j += 1
                for q in range(i, j):
                    blen[q] = j - i
                i = j
            else:
                i += 1
        cb = np.array([blen[min(line[i], len(blen) - 1)] if kind[i] == 1 else 0 for i in range(n)], np.int32)
        ok = np.array([kind[i] > 0 and not ender(pcs[i], int(kind[i])) for i in range(n)], bool)
        r = rc.get(nm) if rc else None
        S.append(dict(ids=ids, ch=z[f"{s}_checks"], dt=z[f"{s}_dtok"], cf=z[f"{s}_dcf"], tt=z[f"{s}_ttok"], kind=kind, ok=ok,
                      cb=cb, gen=gen, pcs=pcs, st=st, rc=r, name=nm))
    return S


def sim(S, spans, end="truth", max_block=None, nmax=15):
    """INERT replay. spans: kinds accepted wholesale. end: 'truth' (v1) or 'draft' (the draft's own span end, 16 rows only).
    max_block: comments in a block of more than this many consecutive comment lines are NOT wholesale."""
    E = T = rows = checks = whole = 0
    reg = {}
    for x in S:
        ids, ch, dt, cf, tt, kind, ok, cb = x["ids"], x["ch"], x["dt"], x["cf"], x["tt"], x["kind"], x["ok"], x["cb"]
        n = len(ids)
        idx = {int(a): i for i, (k, m, a) in enumerate(ch)}
        a = int(ch[:, 2].min()); a_first = a

        def okp(pos):
            if pos >= n or not ok[pos] or int(kind[pos]) not in spans:
                return False
            if max_block is not None and kind[pos] == 1 and cb[pos] > max_block:
                return False
            return True

        while a in idx:
            i = idx[a]; m = int(ch[i, 1])
            if m <= 0:
                break
            Rw = rows_cut(cf[i], m)
            j0 = None; k = 0
            for j in range(min(m, 15)):
                pos = a + 1 + j
                if j >= Rw:
                    if okp(pos):
                        j0 = j
                    break
                if tt[i][j] == ids[pos]:
                    k += 1
                    continue
                if okp(pos):
                    j0 = j
                break
            else:
                j = min(m, 15)
                if nmax > 15 and okp(a + 1 + j):
                    j0 = j
            start_reg = int(kind[a + 1]) if a + 1 < n else 0
            if j0 is None:
                got = k + 1; r = min(Rw, k + 1) + 1 if k < Rw else Rw + 1
                r = Rw + 1
                adv = got
            else:
                kd = int(kind[a + 1 + j0])
                # the 125B's own span end from pos0
                p = a + 1 + j0
                while p < n and okp(p):
                    p += 1
                e_truth = p                                   # first position that is not wholesale (the span's ender, or code)
                if end == "truth" or nmax > 15:
                    L = min(e_truth - (a + 1 + j0), nmax - j0)
                    got = j0 + L + 1; adv = got; r = j0 + L + 1
                    whole += L
                else:
                    L = None
                    for jj in range(j0, 15):
                        if ender(piece(dt[i][jj]), kd):
                            L = jj - j0
                            break
                    if L is None:                             # the draft keeps the span open to the block's end
                        L = 15 - j0
                        got = j0 + L + 1; r = 16
                        adv = min(got, max(e_truth - a, 1))   # the 125B ends the span at its own place at the latest
                    else:                                     # the draft ends the span at depth j0 + L: ender verified, + bonus
                        got = j0 + L + 2; r = j0 + L + 2
                        adv = max(e_truth + 1 - a, 1) if e_truth < n else got
                    whole += L
            y = reg.setdefault(start_reg, [0, 0.0, 0]); y[0] += 1; y[1] += F0 + R0 * r; y[2] += got
            E += got; rows += r; checks += 1
            a += adv
        T += a - a_first
    return dict(E=E, T=T, rows=rows, checks=checks, whole=whole, reg=reg, ms=checks * F0 + rows * R0)


def show(name, x, base):
    t = x["E"] / x["ms"] * 1000; tb = base["E"] / base["ms"] * 1000
    print(f"  {name:58s} kept {x['E'] / x['checks']:6.2f}  tok/s x{t / tb:.3f}  length ratio T {x['E'] / x['T']:.3f}  "
          f"time per answer x{(x['ms'] / base['ms']):.3f}  wholesale {100 * x['whole'] / x['E']:.1f} %")


def study_abc():
    fresh = set(json.load(open(R + "rulec_he12.json")).keys())
    for label, dom, only in (("HumanEval+ all 24", "code", None), ("HumanEval+ fresh 12", "code", (fresh, True)), ("MBPP 100", "mbpp", None)):
        S = load(dom, only=only)
        base = sim(S, set())
        print(f"\n=== {label}: base kept {base['E'] / base['checks']:.2f}, {base['E'] / base['ms'] * 1000:.0f} tok/s")
        # B census
        ct = np.concatenate([x["cb"][x["kind"] == 1] for x in S]); tot = sum(len(x["ids"]) - x["gen"] for x in S)
        print(f"  comment tokens {100 * len(ct) / tot:.1f} % of answer tokens; by block of consecutive comment lines: trailing {100 * np.mean(ct == 0):.1f} %, "
              f"1 line {100 * np.mean(ct == 1):.1f} %, 2 lines {100 * np.mean(ct == 2):.1f} %, 3+ lines {100 * np.mean(ct >= 3):.1f} % (of comment tokens)")
        for name, kw in (
            ("A  comments + docstrings, 16 rows, the 125B's span ends (v1)", dict(spans={1, 2})),
            ("A  comments + docstrings, 16 rows, the DRAFT's span ends", dict(spans={1, 2}, end="draft")),
            ("A  comments only, the DRAFT's span ends", dict(spans={1}, end="draft")),
            ("A  docstrings only, the DRAFT's span ends", dict(spans={2}, end="draft")),
            ("B  docstrings + comments in blocks <= 2 lines, draft ends", dict(spans={1, 2}, end="draft", max_block=2)),
            ("B  docstrings + comments in blocks <= 2 lines, 32 rows (v1 ends)", dict(spans={1, 2}, max_block=2, nmax=31)),
            ("   comments + docstrings, 32 rows (v1 ends)", dict(spans={1, 2}, nmax=31)),
            ("   docstrings only, 32 rows (v1 ends)", dict(spans={2}, nmax=31)),
        ):
            show(name, sim(S, **kw), base)
        # C regions
        nm = {0: "code", 1: "comment", 2: "docstring"}
        for tag, x in (("base", base), ("INERT 16 (draft ends)", sim(S, {1, 2}, end="draft")), ("INERT 32", sim(S, {1, 2}, nmax=31))):
            tot_ms = sum(v[1] for v in x["reg"].values())
            print(f"  C  {tag:22s}: " + "; ".join(f"checks starting in {nm[k]} {100 * v[0] / x['checks']:.0f} % of checks, {100 * v[1] / tot_ms:.0f} % of time, kept {v[2] / v[0]:.1f}"
                                                 for k, v in sorted(x["reg"].items())))


def study_d():
    for label, dom, rf in (("HumanEval+ fresh 12", "code", "rulec_he12.json"), ("MBPP 100", "mbpp", "rulec_mbpp.json")):
        S = [x for x in load(dom, rcfile=rf) if x["rc"] is not None]

        def info(x, pos, t):
            r = x["rc"]; i = pos - int(r["a"]) - 1
            if i < 0 or i >= len(r["tk"]):
                return None
            row = r["tk"][i]; lp = r["lp"][i]
            if t in row:
                j = row.index(t)
                return math.exp(lp[0]), j + 1, lp[0] - lp[j]
            return math.exp(lp[0]), 99, 1e9

        RULES = [("exact everywhere", None),
                 ("rule C form: top-3 at >= 0.1 p1", lambda p1, rk, gap: rk <= 3 and gap <= math.log(10.0)),
                 ("the 125B's top-8", lambda p1, rk, gap: rk <= 8),
                 ("p(draft) >= 0.01", lambda p1, rk, gap: p1 * math.exp(-gap) >= 0.01),
                 ("wholesale", lambda p1, rk, gap: True)]
        print(f"\n=== D  graded in-span rules, profile, 16 rows, no cut inside a span: {label} ({len(S)} windows)")
        base = None
        for spname, spans in (("comments", {1}), ("docstrings", {2}), ("comments + docstrings", {1, 2})):
            for rname, rule in RULES:
                if rule is None and base is not None:
                    continue
                kept = rows = checks = dev = 0
                for x in S:
                    ids, ch, cf, tt, kind, ok = x["ids"], x["ch"], x["cf"], x["tt"], x["kind"], x["ok"]
                    n = len(ids)
                    idx = {int(a): i for i, (k, m, a) in enumerate(ch)}
                    a = max(int(ch[:, 2].min()), int(x["rc"]["a"]))
                    while a in idx:
                        i = idx[a]; m = int(ch[i, 1])
                        if m <= 0:
                            break
                        Rw = rows_cut(cf[i], m)
                        k = 0; r = 1; off = False
                        for j in range(min(m, 15)):
                            pos = a + 1 + j
                            inok = rule is not None and bool(ok[pos]) and int(kind[pos]) in spans
                            if j >= Rw and not inok:
                                break
                            if off and not inok:
                                break
                            r += 1
                            if tt[i][j] == ids[pos]:
                                k += 1
                                continue
                            y = info(x, pos, int(tt[i][j])) if inok else None
                            if inok and (y is not None and rule(*y) or (y is None and rname == "wholesale")):
                                k += 1; dev += 1; off = True
                                continue
                            break
                        kept += k + 1; rows += r; checks += 1
                        a += k + 1
                t = kept / (checks * F0 + rows * R0) * 1000
                if base is None:
                    base = (kept / checks, t)
                    print(f"  exact everywhere: kept {base[0]:.2f}")
                    continue
                print(f"  {spname:22s} {rname:34s} kept {kept / checks:6.2f} (x{kept / checks / base[0]:.3f})  tok/s x{t / base[1]:.3f}  deviating {100 * dev / kept:5.2f} % of kept tokens")


if __name__ == "__main__":
    study_abc()
    study_d()
