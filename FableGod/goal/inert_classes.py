"""INERT-CLASSES (GOAL, Thu 5 Nov 2026): which draft misses on code sit where a different token cannot change what
the program does? CPU only, python -I, numpy + tokenizers (the cached Qwen3.6 tokenizer.json decodes the 125B's ids).
Data: ~/TrainLab/data/d02b_walk_d23h_dfl_<dom>.npz (DFlash2 L1-20 drafts at every position along the 125B's own answers).

A causal lexer over the answer text gives each token's state at its start:
  prose      outside the ```python fence (chat text around the code)
  comment    after '#' on the line
  docstring  inside a triple-quoted string
  string     inside a '...' / "..." literal
  code       everything else; sub-classes for a miss in code:
     blank     both the 125B's token and the draft are whitespace-only and neither changes the line's indentation
     binding   both are identifiers, the position binds a NEW name (assignment target, for / lambda / as / def target)
               and the 125B's name has not appeared before in prompt + answer
     semantic  the rest
Then the official loop (0.3 cut), profile acceptance: a miss in a forgiven class is treated as a hit (teacher-forced walk),
only if neither token ends the span (no newline in a comment, no quote in a string / docstring).
"""
import re
import sys
import numpy as np
from tokenizers import Tokenizer

D = "/Users/kamen/TrainLab/data/"
TOK = sys.argv[1]
DOMS = sys.argv[2:] or ["code", "mbpp"]
CUT = 0.3
ID = re.compile(r"\s*[A-Za-z_][A-Za-z0-9_]*\Z")


def rows_cut(cf, n, t=CUT):
    run = np.cumprod(cf[:n].astype(np.float64))
    bad = np.nonzero(run < t)[0]
    return max(int(bad[0]) if len(bad) else n, 1)


def lex(pieces, start):
    """state of every token at its start, from position `start` (the answer); earlier tokens are 'prompt'"""
    st = ["prompt"] * len(pieces)
    fence = False; mode = "code"; q = ""
    buf = ""
    for i in range(start, len(pieces)):
        if not fence:
            st[i] = "prose"
        else:
            st[i] = mode
        for ch in pieces[i]:
            buf = (buf + ch)[-3:]
            if not fence:
                if buf == "```":
                    fence = True; mode = "code"; buf = ""
                continue
            if mode == "code":
                if buf == "```":
                    fence = False; buf = ""
                elif ch == "#":
                    mode = "comment"
                elif buf in ('"""', "'''"):
                    mode = "docstring"; q = buf; buf = ""
                elif ch in "\"'":
                    mode = "string"; q = ch
            elif mode == "comment":
                if ch == "\n":
                    mode = "code"
            elif mode == "string":
                if buf[-2:] in ('""', "''") and buf[-1] == q and len(buf) >= 2 and buf[-2] == q:
                    # '' or "" : either an empty string or the start of a triple quote; decide on the next char
                    mode = "maybe3"
                elif ch == q or ch == "\n":
                    mode = "code"
            elif mode == "maybe3":
                if ch == q:
                    mode = "docstring"; q = q * 3; buf = ""
                else:
                    mode = "code"
                    if ch == "#":
                        mode = "comment"
            elif mode == "docstring":
                if buf == q:
                    mode = "code"; buf = ""
    return st


def main():
    tok = Tokenizer.from_file(TOK)
    cache = {}

    def piece(t):
        t = int(t)
        if t not in cache:
            cache[t] = tok.decode([t], skip_special_tokens=False)
        return cache[t]

    for dom in DOMS:
        z = np.load(D + f"d02b_walk_d23h_dfl_{dom}.npz", allow_pickle=False)
        S = []
        share = {}
        for s in range(len(z["names"])):
            ids = z[f"{s}_ids"]; gen = int(z[f"{s}_gen"]); ch = z[f"{s}_checks"]
            pcs = [piece(t) for t in ids]
            st = lex(pcs, gen)
            for x in st[gen:]:
                share[x] = share.get(x, 0) + 1
            S.append((ids, ch, z[f"{s}_dtok"], z[f"{s}_dcf"], z[f"{s}_ttok"], pcs, st, gen))
        tot = sum(share.values())
        print(f"\n=== {dom}: {len(S)} answers, {tot} answer tokens; by lexer state: " +
              ", ".join(f"{k} {100 * v / tot:.1f} %" for k, v in sorted(share.items(), key=lambda kv: -kv[1])))

        def cls(smp, pos, dtokid):
            """class of a miss at token position pos (the 125B's token ids[pos], the draft dtokid)"""
            ids, ch, dt, cf, tt, pcs, st, gen = smp
            s0 = st[pos]
            tp = pcs[pos]; dp = piece(dtokid)
            if s0 == "comment":
                return "comment" if ("\n" not in tp and "\n" not in dp) else "comment-end"
            if s0 in ("docstring", "string"):
                if any(c in tp or c in dp for c in "\"'") or (s0 == "string" and ("\n" in tp or "\n" in dp)):
                    return s0 + "-end"
                return s0
            if s0 in ("prose", "prompt", "maybe3"):
                return "prose"
            # code
            if tp.strip() == "" and dp.strip() == "":
                # whitespace both: inert only if the indentation that follows the last newline is the same
                ti = tp.rsplit("\n", 1)[-1] if "\n" in tp else None
                di = dp.rsplit("\n", 1)[-1] if "\n" in dp else None
                if ti == di:
                    return "blank"
                return "semantic"
            if ID.match(tp) and ID.match(dp):
                before = "".join(pcs[:pos])
                line = before.rsplit("\n", 1)[-1]
                nxt = pcs[pos + 1] if pos + 1 < len(pcs) else ""
                name = tp.strip()
                new = re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", before) is None
                dnew = re.search(r"(?<![A-Za-z0-9_])" + re.escape(dp.strip()) + r"(?![A-Za-z0-9_])", before) is None
                n2 = nxt.strip()
                assign = (n2 == "=" or (n2.startswith("=") and not n2.startswith("=="))) and line.strip() == ""
                fort = re.search(r"\bfor\s+([A-Za-z_0-9, ]*)$", line) is not None
                lam = re.search(r"\blambda\s+[A-Za-z_0-9, ]*$", line) is not None
                as_ = re.search(r"\bas\s+$", line) is not None
                deff = re.search(r"^\s*def\s+$", line) is not None
                if new and dnew and (assign or fort or lam or as_ or deff):
                    return "binding"
            return "semantic"

        # ---- first misses of the exact loop by class
        cnt = {}; n_miss = 0
        for smp in S:
            ids, ch, dt, cf, tt, pcs, st, gen = smp
            idx = {int(a): i for i, (k, m, a) in enumerate(ch)}
            a = int(ch[:, 2].min())
            while a in idx:
                i = idx[a]; m = int(ch[i, 1])
                if m <= 0:
                    break
                true = ids[a + 1:a + 1 + m]
                Rw = rows_cut(cf[i], m)
                miss = np.nonzero(dt[i][:Rw] != true[:Rw])[0]
                k = int(miss[0]) if len(miss) else Rw
                if len(miss):
                    c = cls(smp, a + 1 + k, int(dt[i][k]))
                    cnt[c] = cnt.get(c, 0) + 1; n_miss += 1
                a += k + 1
        print(f"  first misses of the exact loop: {n_miss}; by class: " +
              ", ".join(f"{k} {100 * v / n_miss:.1f} %" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])))

        # ---- profile kept when a class set is forgiven
        SETS = [("exact", set()),
                ("comments", {"comment"}),
                ("comments + docstrings", {"comment", "docstring"}),
                ("+ strings", {"comment", "docstring", "string"}),
                ("+ blank lines", {"comment", "docstring", "string", "blank"}),
                ("+ new names at binding sites", {"comment", "docstring", "string", "blank", "binding"}),
                ("+ prose around the code", {"comment", "docstring", "string", "blank", "binding", "prose"})]
        base = None
        for name, F in SETS:
            for t in (0.3, 0.1):
                kept = rows = checks = dev = 0
                for smp in S:
                    ids, ch, dt, cf, tt, pcs, st, gen = smp
                    idx = {int(a): i for i, (k, m, a) in enumerate(ch)}
                    a = int(ch[:, 2].min())
                    while a in idx:
                        i = idx[a]; m = int(ch[i, 1])
                        if m <= 0:
                            break
                        true = ids[a + 1:a + 1 + m]
                        Rw = rows_cut(cf[i], m, t)
                        tk = tt[i]; k = Rw
                        for j in range(Rw):
                            if tk[j] != true[j]:
                                if cls(smp, a + 1 + j, int(tk[j])) in F:
                                    dev += 1
                                    continue
                                k = j
                                break
                        kept += k + 1; rows += Rw + 1; checks += 1
                        a += k + 1
                tps = kept / (checks * 62.0 + rows * 2.8) * 1000
                if base is None:
                    base = (kept / checks, tps)
                print(f"  forgive {name:32s} cut {t}: kept {kept / checks:6.2f} (x{kept / checks / base[0]:.3f})  tok/s x{tps / base[1]:.3f}  forgiven {100 * dev / kept:5.2f} % of kept tokens")


def study2():
    """INERT mode: inside a comment / docstring the draft is accepted wholesale (no verify outcome matters), the span's end
    (newline of a comment, the closing quotes) stays exact. Walk along the 125B's own answers: a check verifies drafts exactly
    (0.3 cut) until the first miss; if that miss, or the cut, or the 15-draft end falls on a forgivable in-span position, the
    check goes on wholesale to the end of the span or of the block (15 / 31 / 63 drafts). Conservative: the check stops at the
    span's end. Assumes the drafter's inert text is as long as the 125B's."""
    tok = Tokenizer.from_file(TOK)
    cache = {}

    def piece(t):
        t = int(t)
        if t not in cache:
            cache[t] = tok.decode([t], skip_special_tokens=False)
        return cache[t]

    for dom in DOMS:
        z = np.load(D + f"d02b_walk_d23h_dfl_{dom}.npz", allow_pickle=False)
        S = []
        import os, json
        SUB = os.environ.get("GOAL_SUBSET")          # 'fresh' / 'seen': HumanEval+ windows outside / inside the decision map
        FRESH = set(json.load(open("/Users/kamen/Breakthrough/data/rulec/rulec_he12.json")).keys()) if SUB else set()
        acc = {}                                      # per lexer state: [teacher-forced right, n] over depths 1-15 of the official loop
        for s in range(len(z["names"])):
            if SUB and ((str(z["names"][s]) in FRESH) != (SUB == "fresh")):
                continue
            ids = z[f"{s}_ids"]; gen = int(z[f"{s}_gen"]); ch = z[f"{s}_checks"]
            pcs = [piece(t) for t in ids]
            st = lex(pcs, gen)
            tt_ = z[f"{s}_ttok"]
            for (k, m, a), t in zip(ch, tt_):
                for j in range(int(m)):
                    y = acc.setdefault(st[a + 1 + j], [0, 0]); y[0] += int(t[j] == ids[a + 1 + j]); y[1] += 1
            forg = np.zeros(len(ids), bool); kind = np.zeros(len(ids), np.int8)
            for i in range(gen, len(ids)):
                if st[i] == "comment" and "\n" not in pcs[i]:
                    forg[i] = True; kind[i] = 1
                elif st[i] == "docstring" and not any(c in pcs[i] for c in "\"'"):
                    forg[i] = True; kind[i] = 2
            S.append((ids, ch, z[f"{s}_dcf"], z[f"{s}_ttok"], forg, kind))

        def run(nmax, spans):
            """spans: set of kinds accepted wholesale (1 comment, 2 docstring)"""
            kept = rows = rin = checks = whole = 0
            for ids, ch, cf, tt, forg, kind in S:
                idx = {int(a): i for i, (k, m, a) in enumerate(ch)}
                a = int(ch[:, 2].min())
                n = len(ids)
                while a in idx:
                    i = idx[a]; m = int(ch[i, 1])
                    if m <= 0:
                        break
                    Rw = rows_cut(cf[i], m)
                    k = 0; r = 1; ri = 0; inspan = False
                    for j in range(nmax):
                        pos = a + 1 + j
                        if pos >= n:
                            break
                        ok = bool(forg[pos]) and int(kind[pos]) in spans
                        if inspan:
                            if not ok:
                                break
                            k += 1; r += 1; ri += 1; whole += 1
                            continue
                        if j >= Rw:
                            if ok:
                                inspan = True; k += 1; r += 1; ri += 1; whole += 1
                                continue
                            break
                        r += 1
                        if tt[i][j] == ids[pos]:
                            k += 1
                        elif ok:
                            inspan = True; k += 1; ri += 1; whole += 1
                        else:
                            break
                    kept += k + 1; rows += r; rin += ri; checks += 1
                    a += k + 1
            return kept, rows, rin, checks, whole

        print(f"\n--- {dom}{' (' + SUB + ')' if SUB else ''}: DFlash2 per-token accuracy (teacher-forced, all depths) by span: " +
              ", ".join(f"{k} {v[0] / v[1]:.3f}" for k, v in sorted(acc.items(), key=lambda kv: -kv[1][1]) if v[1] > 200))
        base = run(15, set())
        def tps(x, F=62.0, r=2.8, r_in=None, d=0.0):
            kept, rows, rin, checks, whole = x
            r_in = r if r_in is None else r_in
            return kept / (checks * (F + d) + (rows - rin) * r + rin * r_in) * 1000
        print(f"\n=== INERT mode, {dom}: base kept {base[0] / base[3]:.2f}, rows {base[1] / base[3]:.2f}, {tps(base):.0f} tok/s at 62 + 2.8/row")
        for name, spans in (("comments", {1}), ("docstrings", {2}), ("comments + docstrings", {1, 2})):
            for nmax in (15, 31, 63):
                x = run(nmax, spans)
                d = 0.0 if nmax == 15 else 3.0 if nmax == 31 else 9.0   # extra drafter time for longer blocks [G]
                print(f"  wholesale in {name:22s} block {nmax + 1:2d}: kept {x[0] / x[3]:6.2f} rows {x[1] / x[3]:6.2f}  "
                      f"tok/s x{tps(x, d=d) / tps(base):.3f} (rows 2.8 ms)  x{tps(x, r_in=1.5, d=d) / tps(base):.3f} (in-span rows 1.5 ms)  "
                      f"x{tps(x, F=30.0, r=1.5, r_in=1.0, d=d / 2) / tps(base, F=30.0, r=1.5):.3f} (floor engine 30 + 1.5, in-span 1.0)  "
                      f"drafter-written {100 * x[4] / x[0]:.1f} % of tokens, checks x{x[3] / base[3]:.3f}")


if __name__ == "__main__":
    if len(DOMS) and DOMS[0] == "inert":
        DOMS.pop(0)
        if not DOMS:
            DOMS.extend(["code", "mbpp"])
        study2()
    else:
        main()
