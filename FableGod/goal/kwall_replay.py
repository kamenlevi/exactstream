"""kwall_replay.py (GOAL session, Fri 2026-11-06): the K wall on MBPP and docstring-free HumanEval. CPU only, python -I.
What ends a 16-row check, exact and with comments + docstrings kept wholesale (INERT CD16, the engine's own tagger), by
the Python class of the token the check stops on; and the replay of each K_WALL idea (lines in K_WALL_1106.md, written
before its run). Data: MTP Head's every-position DFlash2 walks (d02b_walk_d23h_dfl_<set>.npz, the 125B's own answers),
Breakthrough's top-8 rows (rulec) for the rows a likelihood decides.

Two pointers per check: row j's draft is compared with the 125B's own token at position q. A row kept wholesale inside a
span (the draft's text differs from the 125B's) puts the check on the draft's own text until the draft ends the span;
then, convention A (precheck_inert.py): the check stops there; convention B: the 125B's code after a drafter-written
comment is the code it wrote after its own comment, so the rows after the span are compared with the 125B's own tokens
after its own end of that span. B is the convention under which INERT keeps the task's result at all.

usage: python -I kwall_replay.py census | split | ideas | tail | all"""
import ast
import builtins
import importlib.util
import io
import json
import keyword
import math
import os
import re
import sys
import tokenize

import numpy as np

H = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


PI = load_mod("precheck_inert", f"{HERE}/build_1105/precheck_inert.py")
INM, RXC, dec = PI.INM, PI.RXC, PI.dec
CUT = 0.3
CD16 = INM.parse_arm("CD16")


# ---------------------------------------------------------------------------------------------- the Python classes
def fine(x):
    """per answer position p >= gen: (class, detail). Classes: comment, docstring, string, ident-new, ident-seen, kw,
    builtin, num, op, ws-nl, ws-sp, prose, fence. detail: the Python token's text (idents: + ':start' / ':cont')"""
    ids, gen = x["ids"], x["gen"]
    pieces = [dec(t) for t in ids[gen:]]
    st = np.cumsum([0] + [len(s) for s in pieces])
    text = "".join(pieces)
    prompt = "".join(dec(t) for t in ids[:gen])
    cc = ["prose"] * len(text)
    det = [""] * len(text)
    inprompt = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", prompt))
    seen = set()
    for m in re.finditer(r"```(?:python|py|python3)[ \t]*\n", text):
        for i in range(m.start(), m.end()):
            cc[i] = "fence"
        c0 = m.end(); c1 = text.find("\n```", c0)
        if c1 >= 0:
            for i in range(c1 + 1, min(c1 + 4, len(text))):
                cc[i] = "fence"
        c1 = len(text) if c1 < 0 else c1 + 1
        code = text[c0:c1]
        for i in range(c0, c1):
            cc[i] = "ws"
        lo = np.cumsum([0] + [len(l) + 1 for l in code.split("\n")])
        toks = []
        try:
            for t in tokenize.generate_tokens(io.StringIO(code).readline):
                toks.append(t)
        except (tokenize.TokenError, IndentationError, SyntaxError):
            pass
        sig = [t for t in toks if t.type not in (tokenize.NL, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT)]
        stmt = set()                                           # string statements (docstrings): a STRING alone on its line
        for i, t in enumerate(sig):
            if t.type == tokenize.STRING and (i == 0 or sig[i - 1].type == tokenize.NEWLINE) and (
                    i + 1 == len(sig) or sig[i + 1].type in (tokenize.NEWLINE, tokenize.ENDMARKER)):
                stmt.add((t.start, t.end))
        for t in toks:
            a_, b_ = int(lo[t.start[0] - 1]) + t.start[1], int(lo[t.end[0] - 1]) + t.end[1]
            if t.type == tokenize.COMMENT:
                c, d = "comment", ""
            elif t.type == tokenize.STRING:
                c, d = ("docstring" if (t.start, t.end) in stmt else "string"), ""
            elif t.type == tokenize.NAME:
                if keyword.iskeyword(t.string):
                    c, d = "kw", t.string
                elif hasattr(builtins, t.string):
                    c, d = "builtin", t.string
                else:
                    c = "ident-prompt" if t.string in inprompt else "ident-seen" if t.string in seen else "ident-new"
                    d = t.string
                    seen.add(t.string)
            elif t.type == tokenize.NUMBER:
                c, d = "num", t.string
            elif t.type == tokenize.OP:
                c, d = "op", t.string
            elif tokenize.tok_name[t.type].startswith("FSTRING"):
                c, d = "string", ""
            else:
                continue
            for i in range(a_, min(b_, len(code))):
                cc[c0 + i] = c
                det[c0 + i] = d + (":start" if i == a_ else ":cont")
    out = {}
    for i, s in enumerate(pieces):
        c, d = None, ""
        for j in range(int(st[i]), int(st[i + 1])):
            if cc[j] != "ws":
                c, d = cc[j], det[j]
                break
        if c is None:
            c = "ws-nl" if "\n" in s else "ws-sp"
        out[gen + i] = (c, d)
    x["cc"], x["st0"] = cc, st
    return out


def dclass(s):
    t = s.strip()
    if not t:
        return "ws-nl" if "\n" in s else "ws-sp"
    if t.startswith("#"):
        return "comment-open"
    if '"""' in t or "'''" in t:
        return "doc-quote"
    if re.match(r"[A-Za-z_]", t):
        return "kw" if keyword.iskeyword(t) else "name"
    if t[0].isdigit():
        return "num"
    if t[0] in "\"'":
        return "quote"
    return "op"


def p1_of(x, pos):
    r = x["rc"]
    if r is None:
        return None
    i = pos - int(r["a"]) - 1
    if i < 0 or i >= len(r["lp"]):
        return None
    return math.exp(r["lp"][i][0])


# ---------------------------------------------------------------------------------------------- one sample's tables
def prep(x):
    """the tagger's state after every answer token; the 125B's own text by kind; its own end of the span at each position"""
    ids, gen, n = x["ids"], x["gen"], len(x["ids"])
    sc = INM.InertScan(RXC)
    at = {}
    for p in range(gen, n):
        sc.feed(dec(ids[p]))
        at[p] = sc.copy()
    txt = [dec(t) for t in ids[gen + 1:]]
    kt = {}
    for j, (kind, mode, _) in enumerate(INM.classify(RXC, at[gen], txt, txt, CD16)):
        kt[gen + 1 + j] = (kind, mode == "whole")
    fn = fine(x)
    for p in range(gen + 1, n):                           # an answer cut inside a docstring: tokenize reads nothing there
        if fn[p][0] in ("ws-sp", "ws-nl") and at[p - 1].in_doc() and at[p].in_doc():
            fn[p] = ("docstring", "")
    x["at"], x["kt"], x["fine"] = at, kt, fn
    x["idx"] = {int(a): i for i, (k, m, a) in enumerate(x["ch"])}
    # regions: sig (up to the first docstring or the end of the def line), doc (the first docstring), body (the rest)
    reg, cur, seen_doc = {}, "sig", False
    for p in range(gen, n):
        c = fn[p][0]
        if cur == "sig":
            if c == "docstring":
                cur, seen_doc = "doc", True
            elif c == "ws-nl" and any(fn[r][0] == "kw" and fn[r][1].startswith("def") for r in range(gen, p)) \
                    and not any(fn[r][0] == "docstring" for r in range(p + 1, min(p + 4, n))):
                reg[p] = "sig"; cur = "body"; continue
        elif cur == "doc" and c != "docstring":
            cur = "body"
        reg[p] = cur
    x["reg"] = reg
    x["capped"] = (n - gen) >= 500 and at[n - 1].in_doc()


def own_end(x, q, kind):
    """the 125B's own end of the span it is in at position q: the first position >= q that is not a whole-open row of
    this kind (its newline / its closing quote)"""
    kt, n = x["kt"], len(x["ids"])
    p = q
    while p < n and kt.get(p, (None, False))[0] == kind and kt[p][1]:
        p += 1
    return p


def ends_span(kind, s):
    return "\n" in s if kind == "comment" else ('"' in s or "'" in s)


# ---------------------------------------------------------------------------------------------- the loop
def run(S, arm=None, conv="B", ext=(), forgive=None, census=None, cut=CUT, over=None, refresh=0, bern=None, seed=0, d0=None,
        pi_dev=1.0, lift=1.0):
    """the official loop. arm: an INERT arm or None (exact). ext: extra acceptance rules (K_WALL ideas). over: a draft
    override (an idea that changes the draft before the check): over(x, a, j, q, s_, d) -> token or None. forgive: a set
    of stop classes an oracle drafter gets right (ceilings). After an overridden or forgiven row the chain is the
    drafter's own pick on the right prefix (the walk's teacher-forced picks: the first-order selector re-walked), while
    the row still sits at its own position. census: a list that collects every check's stop. -> totals"""
    E = T = rows_all = checks = 0
    ext = set(ext)
    nover = nover_ok = nover_bad = 0
    rng = np.random.default_rng(seed)
    fgv = forgive if (forgive is None or callable(forgive)) else (lambda why, x, q, _f=frozenset(forgive): why[0] in _f)
    for x in S:
        ids, gen, ch, idx = x["ids"], x["gen"], x["ch"], x["idx"]
        n = len(ids)
        a = max(int(ch[:, 2].min()), gen)
        while a in idx:
            i = idx[a]; m = int(ch[i, 1])
            if m <= 0:
                break
            dt, cf, tt, tc = x["dt"][i], x["cf"][i], x["tt"][i], x["tc"][i]
            scn = x["at"][a]
            s_ = scn.copy()
            q, k, state, stop = a + 1, 0, None, None           # state: None = on the 125B's own text, else [kind, q_end, q_dev, c]
            use_tf, rc, R = False, 1.0, 0
            for j in range(m):
                tf = use_tf and state is None and q == a + 1 + j
                d, cfj = (int(tt[j]), float(tc[j])) if tf else (int(dt[j]), float(cf[j]))
                if state is None and q < n:
                    if refresh and j + 1 > (refresh if d0 is None else d0) and (q - refresh) in idx:   # R3 / R7: a fresh anchor D back
                        i2 = idx[q - refresh]
                        if int(ch[i2, 1]) >= refresh:
                            d, cfj = int(x["tt"][i2][refresh - 1]), float(x["tc"][i2][refresh - 1])
                    if bern is not None and x["fine"][q][0] not in ("comment", "docstring"):   # R4: code rows right with p = bern
                        d, cfj = (ids[q] if rng.random() < bern else -7), 1.0
                if over is not None and state is None and q < n:
                    o = over(x, a, j, q, s_, d)
                    if o is not None and o != d:
                        nover += 1
                        nover_ok += o == ids[q]; nover_bad += d == ids[q]
                        d, cfj = o, 1.0
                        if q == a + 1 + j:
                            use_tf = True
                ds = dec(d) if d >= 0 else " \x00"
                if arm is not None:
                    m_ = INM.classify(RXC, s_, [ds], None, arm)[0][1]
                    if m_ == "whole":
                        cfj = 1.0                               # pre_block: a row inside a span is sure in the cut
                    elif isinstance(m_, tuple) and len(m_) > 3:
                        cfj = max(cfj, lift)                    # a guarded row: its confidence floor in the cut
                rc *= cfj
                if rc < cut:
                    stop = ("cut", None); break
                R += 1
                if state is None:
                    if q >= n:
                        stop = ("end", None); break
                    t = ids[q]
                    if d == t:
                        q += 1
                    else:
                        ts = dec(t)
                        kind, mode, _ = INM.classify(RXC, s_, [ds], [ts], arm)[0] if arm is not None else (None, None, "")
                        ok = False
                        if mode == "whole":
                            state = [kind, own_end(x, q, kind), q, 0]; ok = True
                        elif mode is not None:
                            dl, rk = PI.top8(x, q, d)                   # the 125B's own rank / margin for the draft, on its own text
                            if INM.rule_ok(mode, dl, rk):
                                if ends_span(kind, ds):                 # the draft ends the span sooner than the 125B
                                    q = own_end(x, q, kind) + 1
                                else:                                   # a guarded row kept inside the span
                                    state = [kind, own_end(x, q, kind), q, 0]
                                ok = True
                        if not ok:
                            why = stop_class(x, q, s_, ds, ts, arm, kind)
                            ok = rule_ext(x, a, j, q, s_, d, ds, t, ts, why, ext, arm) if ext else None
                            if ok:                                       # an idea's rule kept the row: (new q, new state)
                                q, state = ok
                            elif fgv is not None and fgv(why, x, q):
                                if q == a + 1 + j:
                                    use_tf = True
                                q += 1; ds = ts                          # an oracle drafter here: the 125B's token
                            else:
                                stop = why; break
                else:
                    kind, mode, _ = INM.classify(RXC, s_, [ds], None, arm)[0]
                    guarded = arm.get("com") not in (None, "whole") or arm.get("doc") not in (None, "whole")
                    if kind == state[0] and mode == "whole":
                        state[3] += 1
                    elif kind == state[0] and isinstance(mode, tuple):  # a guarded row on the drafter's own text: the 125B's
                        if rng.random() < (pi_dev(kind, j) if callable(pi_dev) else pi_dev):   # not on disk: kept with these odds
                            state[3] += 1
                        else:
                            stop = ("dev-guard", None); break
                    elif kind == state[0] and ends_span(kind, ds) and guarded and rng.random() >= (
                            pi_dev(kind, j) if callable(pi_dev) else pi_dev):
                        stop = ("dev-guard", None); break               # a guarded arm: the span's end is exact, same odds
                    elif kind == state[0] and ends_span(kind, ds):
                        if conv == "A":
                            s_.feed(ds); k += 1
                            q, state, stop = state[1] + 1, None, ("conv-A", None)
                            break
                        q, state = state[1] + 1, None
                    else:
                        stop = ("dev-other", None); break
                s_.feed(ds); k += 1
            if stop is None:
                stop = ("full", None)
            elif stop[0] != "cut":                             # the engine verifies every row up to the cut, kept or not
                for j2 in range(R, m):
                    rc *= float(cf[j2])
                    if rc < cut:
                        break
                    R += 1
            got = k + 1
            if state is None:
                a2 = q if stop[0] != "end" else n
            else:
                a2 = min(state[2] + state[3] + 1, state[1])
            a2 = max(a2, a + 1)
            if census is not None:
                census.append(dict(stop=stop, k=k, R=R, a=a, q=q, x=x, dev=state is not None))
            E += got; rows_all += R + 1; checks += 1; T += a2 - a
            a = a2
    return dict(E=E, T=T, rows=rows_all, checks=checks, kept=E / max(checks, 1), rowsper=rows_all / max(checks, 1),
                lenr=E / max(T, 1), over=(nover, nover_ok, nover_bad))


def stop_class(x, q, s_, ds, ts, arm, kind):
    """why an unequal row on the 125B's own text was not kept. -> (class, detail)"""
    fc, fd = x["fine"].get(q, ("prose", ""))
    prev = x["fine"].get(q - 1, ("prose", ""))[0]
    c = 0
    while c < len(ts) and c < len(ds) and ts[c] == ds[c]:
        c += 1
    if 0 < c < len(ts) and fc not in ("comment", "docstring", "prose", "fence"):
        j = int(x["st0"][q - x["gen"]]) + c
        while j < int(x["st0"][q - x["gen"] + 1]) and x["cc"][j] == "ws":
            j += 1
        if j < int(x["st0"][q - x["gen"] + 1]) and x["cc"][j] not in ("ws", "prose", "fence"):
            fc = x["cc"][j]
    din = "comment" if s_.in_comment() else "docstring" if s_.in_doc() else None    # where the draft's row sits
    if din is not None:                                    # the row sits in a span the tagger sees
        if ends_span(din, ts) and not ends_span(din, ds):
            return (f"{din}-end: 125B ends, draft goes on", fd)
        if ends_span(din, ds) and not ends_span(din, ts):
            return (f"{din}-end: draft ends, 125B goes on", fd)
        if ends_span(din, ds) and ends_span(din, ts):
            return (f"{din}-end: both end, other token", fd)
        return (f"{din}: closed by a guard", fd)
    if fc == "comment":
        if prev != "comment":
            return ("comment-open: 125B opens, draft " + dclass(ds), fd)
        return ("comment: not tagged", fd)
    if fc == "docstring":
        if prev != "docstring":
            return ("docstring-open: 125B opens, draft " + dclass(ds), fd)
        return ("docstring: not tagged", fd)
    if dclass(ds) == "comment-open" and fc not in ("prose", "fence", "string"):
        return ("comment-open: draft opens, 125B " + fc, fd)
    if dclass(ds) == "doc-quote" and fc not in ("prose", "fence", "string"):
        return ("docstring-open: draft opens, 125B " + fc, fd)
    return ("code: " + fc if fc not in ("prose", "fence") else fc, fd)


def rule_ext(x, a, j, q, s_, d, ds, t, ts, why, ext, arm):
    return None


# ---------------------------------------------------------------------------------------------- read-outs
def samples(dom, top8=None, only=None):
    """precheck_inert's samples + the walk's teacher-forced picks (the drafter's pick at each depth on the right prefix)"""
    S = PI.samples(dom, top8, only)
    z = np.load(f"{PI.TD}/d02b_walk_d23h_dfl_{dom}.npz", allow_pickle=False)
    at = {str(z["names"][i]): i for i in range(len(z["names"]))}
    for x in S:
        i = at[x["name"]]
        x["tt"], x["tc"] = z[f"{i}_ttok"], z[f"{i}_tcf"]
    return S


def load_sets():
    fresh = set(json.load(open(PI.TOP8["he12"])).keys())
    sets = [("MBPP 100", samples("mbpp", PI.TOP8["mbpp"])),
            ("HumanEval+ fresh 12", samples("code", PI.TOP8["he12"], (fresh, True))),
            ("HumanEval+ all 24", samples("code", PI.TOP8["he12"]))]
    for _, S in sets:
        for x in S:
            prep(x)
    return sets


def share(cen, key):
    out = {}
    for c in cen:
        out[key(c)] = out.get(key(c), 0) + 1
    return sorted(out.items(), key=lambda kv: -kv[1])


def census_main(sets):
    for label, S in sets:
        ntok = sum(len(x["ids"]) - x["gen"] for x in S)
        fc = {}
        for x in S:
            for p, (c, _) in x["fine"].items():
                fc[c] = fc.get(c, 0) + 1
        print(f"\n=== {label}: {len(S)} answers, {ntok} answer tokens; by Python class: "
              + ", ".join(f"{c} {100 * v / ntok:.1f} %" for c, v in sorted(fc.items(), key=lambda kv: -kv[1])))
        for name, arm, conv in (("exact", None, "B"), ("CD16, convention A (precheck)", CD16, "A"), ("CD16, convention B", CD16, "B")):
            cen = []
            r = run(S, arm, conv, census=cen)
            print(f"\n  -- {name}: kept {r['kept']:.2f} per check, rows {r['rowsper']:.2f}, {r['checks']} checks, length {r['lenr']:.3f}")
            n = len(cen)
            grp = share(cen, lambda c: c["stop"][0].split(":")[0])
            print("     stops by group: " + ", ".join(f"{g} {100 * v / n:.1f} %" for g, v in grp))
            for g, v in share(cen, lambda c: c["stop"][0])[:28]:
                ks = [c["k"] for c in cen if c["stop"][0] == g]
                print(f"       {100 * v / n:5.1f} %  {g:54s} mean drafts kept before it {np.mean(ks):5.2f}")
            # the code misses by the 125B's p1 on that row
            cm = [c for c in cen if c["stop"][0].startswith("code")]
            ps = [p1_of(c["x"], c["q"]) for c in cm]
            ps = [p for p in ps if p is not None]
            if ps:
                ps = np.array(ps)
                print(f"     code stops by the 125B's p1 on the row ({len(ps)} with top-8 data): >= 0.99 {100 * np.mean(ps >= 0.99):.1f} %, "
                      f"0.9-0.99 {100 * np.mean((ps >= 0.9) & (ps < 0.99)):.1f} %, 0.5-0.9 {100 * np.mean((ps >= 0.5) & (ps < 0.9)):.1f} %, "
                      f"< 0.5 {100 * np.mean(ps < 0.5):.1f} %")
            if arm is not None and conv == "B":
                print("     ceilings (an oracle drafter on one group of stops; the walk goes on):")
                base = r["kept"]
                groups = sorted({c["stop"][0].split(":")[0] for c in cen} - {"full", "cut", "end", "dev-other", "conv-A"})
                for g in groups:
                    fg = {c["stop"][0] for c in cen if c["stop"][0].split(":")[0] == g}
                    r2 = run(S, arm, conv, forgive=fg)
                    print(f"       no stop on {g:22s} kept {r2['kept']:6.2f} (x{r2['kept'] / base:.3f})")
                fg = {c["stop"][0] for c in cen if not c["stop"][0].startswith("code")}
                r2 = run(S, arm, conv, forgive=fg)
                print(f"       no stop outside code tokens     kept {r2['kept']:6.2f} (x{r2['kept'] / base:.3f})")
                fg = {c["stop"][0] for c in cen}
                r2 = run(S, arm, conv, forgive=fg)
                print(f"       no stop at all (the cut only)   kept {r2['kept']:6.2f} (x{r2['kept'] / base:.3f})")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "census"
    sets = load_sets()
    if what in ("census", "all"):
        census_main(sets)


def split_main(sets):
    """K by region of the check's first draft (sig / doc / body), the cut sweep, and the ceilings by code class"""
    more = [("MBPP 94 (without the 6 answers cut inside a docstring)", [x for x in sets[0][1] if not x["capped"]])]
    for label, S in sets + more:
        ntok = sum(len(x["ids"]) - x["gen"] for x in S)
        rg = {}
        for x in S:
            for p, r in x["reg"].items():
                rg[r] = rg.get(r, 0) + 1
        print(f"\n=== {label}: {len(S)} answers, {ntok} tokens; regions: " + ", ".join(f"{r} {100 * v / ntok:.1f} %" for r, v in sorted(rg.items())))
        for name, arm in (("exact", None), ("CD16-B", CD16)):
            cen = []
            r = run(S, arm, "B", census=cen)
            parts = []
            for g in ("sig", "doc", "body"):
                ks = [c["k"] + 1 for c in cen if c["x"]["reg"].get(c["a"] + 1) == g]
                if ks:
                    parts.append(f"{g} {np.mean(ks):.2f} ({len(ks)} checks, {100 * sum(ks) / max(r['E'], 1):.0f} % of tokens)")
            print(f"  {name:7s} kept {r['kept']:.2f} per check ({r['checks']} checks, rows {r['rowsper']:.2f}); by the region the check starts in: " + "; ".join(parts))
            # body checks by where they stop
            body = [c for c in cen if c["x"]["reg"].get(c["a"] + 1) == "body"]
            if body:
                grp = share(body, lambda c: c["stop"][0].split(":")[0])
                print(f"          body checks stop on: " + ", ".join(f"{g} {100 * v / len(body):.0f} %" for g, v in grp[:7]))
        print("  the cut (running confidence), kept per check / rows per check / tok/s at 62 + 2.8 per row | at 30 + 2.4 per row | at 50 + 1.1 per row:")
        for name, arm in (("exact", None), ("CD16-B", CD16)):
            base = None
            for cut in (0.3, 0.2, 0.1, 0.03, 0.0):
                r = run(S, arm, "B", cut=cut)
                t = [1000 * r["E"] / (r["checks"] * f0 + r["rows"] * r0) for f0, r0 in ((62, 2.8), (30, 2.4), (50, 1.1))]
                base = base or t
                print(f"    {name:7s} cut {cut:4.2f}: kept {r['kept']:6.2f}  rows {r['rowsper']:5.2f}  tok/s {t[0]:5.1f} (x{t[0] / base[0]:.3f}) | "
                      f"{t[1]:5.1f} (x{t[1] / base[1]:.3f}) | {t[2]:5.1f} (x{t[2] / base[2]:.3f})")
        cen = []
        r = run(S, CD16, "B", census=cen)
        print(f"  ceilings under CD16-B (kept {r['kept']:.2f}): an oracle drafter on one class of code stops, the walk goes on:")
        labs = sorted({c["stop"][0] for c in cen if c["stop"][0].startswith("code")})
        for g in labs:
            r2 = run(S, CD16, "B", forgive={g})
            nst = sum(1 for c in cen if c["stop"][0] == g)
            print(f"    {g:22s} {100 * nst / len(cen):5.1f} % of checks   kept {r2['kept']:6.2f} (x{r2['kept'] / r['kept']:.3f})")
        for cut in (0.3, 0.0):
            r0_ = run(S, CD16, "B", cut=cut)
            r2 = run(S, CD16, "B", forgive={c["stop"][0] for c in cen if c["stop"][0].startswith("code")}, cut=cut)
            r3 = run(S, CD16, "B", forgive={c["stop"][0] for c in cen}, cut=cut)
            print(f"    cut {cut:.1f}: CD16-B {r0_['kept']:.2f}; no code stop {r2['kept']:.2f}; no stop at all {r3['kept']:.2f}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("split", "all"):
    split_main(sets)


def ideas_main(sets):
    """R2-R6 (lines in K_WALL_lines_1106.md, written 10:38 before this ran)"""
    mb94 = ("MBPP 94 (without the 6 answers cut inside a docstring)", [x for x in sets[0][1] if not x["capped"]])
    INERT = ("comment", "docstring")
    # ---- R2 ALIGN
    print("=== R2 ALIGN: teacher-forced accuracy on code tokens the 125B is sure of (p1 >= 0.99), by depth and by the number of")
    print("    comment / docstring tokens between the anchor and the row")
    for label, S in (sets[0], mb94, sets[1]):
        acc = {b: [np.zeros(4), np.zeros(4)] for b in ("0", "1-4", ">=5")}
        for x in S:
            ids, gen = x["ids"], x["gen"]
            fn = x["fine"]
            ine = np.array([1 if fn.get(p, ("", ""))[0] in INERT else 0 for p in range(len(ids))])
            cs = np.concatenate([[0], np.cumsum(ine)])
            for i, (k, m, a) in enumerate(x["ch"]):
                m, a = int(m), int(a)
                if a < gen:
                    continue
                for j in range(m):
                    p = a + 1 + j
                    if fn[p][0] in INERT or fn[p][0] in ("prose", "fence"):
                        continue
                    p1 = p1_of(x, p)
                    if p1 is None or p1 < 0.99:
                        continue
                    nb = int(cs[p] - cs[a + 1])
                    b = "0" if nb == 0 else "1-4" if nb <= 4 else ">=5"
                    g = 0 if j < 3 else 1 if j < 7 else 2 if j < 11 else 3
                    acc[b][0][g] += int(x["tt"][i][j]) == ids[p]; acc[b][1][g] += 1
        print(f"  {label}")
        for b in acc:
            print(f"    inert tokens between {b:4s}: depth 1-3 {acc[b][0][0] / max(acc[b][1][0], 1):.3f} (n {int(acc[b][1][0])}), 4-7 "
                  f"{acc[b][0][1] / max(acc[b][1][1], 1):.3f} ({int(acc[b][1][1])}), 8-11 {acc[b][0][2] / max(acc[b][1][2], 1):.3f} ({int(acc[b][1][2])}), "
                  f"12-15 {acc[b][0][3] / max(acc[b][1][3], 1):.3f} ({int(acc[b][1][3])})")
        d0 = (acc["0"][0][2] + acc["0"][0][3]) / max(acc["0"][1][2] + acc["0"][1][3], 1)
        d5 = (acc[">=5"][0][2] + acc[">=5"][0][3]) / max(acc[">=5"][1][2] + acc[">=5"][1][3], 1)
        print(f"    depths 8-15: none between {d0:.3f}, >= 5 between {d5:.3f}, gap {d0 - d5:+.3f}")
    # ---- R3 REFRESH, R4, R5
    for label, S in sets + [mb94]:
        base = run(S, CD16, "B")
        ex = run(S, None, "B")
        print(f"\n=== {label}: exact {ex['kept']:.2f}; CD16-B kept {base['kept']:.2f}, rows {base['rowsper']:.2f}")
        print("  R3 REFRESH ceiling: every row deeper than D picked as a fresh anchor D tokens back would pick it")
        for D in (8, 4, 2, 1):
            r = run(S, CD16, "B", refresh=D)
            r0 = run(S, CD16, "B", refresh=D, cut=0.0)
            e = run(S, None, "B", refresh=D)
            print(f"    D = {D}: CD16-B kept {r['kept']:6.2f} (x{r['kept'] / base['kept']:.3f}), rows {r['rowsper']:5.2f}; without the cut {r0['kept']:6.2f}; "
                  f"exact acceptance {e['kept']:6.2f} (x{e['kept'] / ex['kept']:.3f})")
        print("  R4 pricing table: code rows right with probability r at every depth (no cut; 3 seeds)")
        for rr in (0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0):
            ks = [run(S, CD16, "B", bern=rr, seed=sd, cut=0.0)["kept"] for sd in (0, 1, 2)]
            ke = [run(S, None, "B", bern=rr, seed=sd, cut=0.0)["kept"] for sd in (0, 1, 2)]
            print(f"    r = {rr:.2f}: CD16-B kept {np.mean(ks):6.2f}; exact acceptance (inert text as the drafter has it) {np.mean(ke):6.2f}")
        print("  R5 ceilings (an oracle drafter on one class; the walk goes on with the drafter's own picks on the right prefix)")
        cen = []
        run(S, CD16, "B", census=cen)
        for nm, f in (("R5a PROMPT-NAME: identifiers that occur in the prompt", lambda why, x, q: why[0] == "code: ident-prompt"),
                      ("R5b ALPHA: new identifiers", lambda why, x, q: why[0] == "code: ident-new"),
                      ("     identifiers seen earlier in the answer", lambda why, x, q: why[0] == "code: ident-seen"),
                      ("     operators and punctuation", lambda why, x, q: why[0] == "code: op"),
                      ("     keywords + builtins", lambda why, x, q: why[0] in ("code: kw", "code: builtin")),
                      ("     numbers + strings", lambda why, x, q: why[0] in ("code: num", "code: string")),
                      ("     whitespace (newlines, indentation)", lambda why, x, q: why[0] in ("code: ws-nl", "code: ws-sp")),
                      ("     every stop inside the def line (region sig)", lambda why, x, q: x["reg"].get(q) == "sig"),
                      ("     code stops on rows the 125B is sure of (p1 >= 0.99)", lambda why, x, q: why[0].startswith("code") and (p1_of(x, q) or 0) >= 0.99),
                      ("     code stops on rows with p1 < 0.99", lambda why, x, q: why[0].startswith("code") and (p1_of(x, q) or 1) < 0.99),
                      ("     every code stop", lambda why, x, q: why[0].startswith("code"))):
            r = run(S, CD16, "B", forgive=f)
            r0 = run(S, CD16, "B", forgive=f, cut=0.0)
            print(f"    {nm:58s} kept {r['kept']:6.2f} (x{r['kept'] / base['kept']:.3f}); without the cut {r0['kept']:6.2f}")
    # ---- R6 CANDIDATES
    z = np.load(f"{PI.TD}/d02b_walk_kw_code.npz", allow_pickle=False)
    byname = {x["name"]: x for x in sets[2][1]}
    rk = {g: np.zeros(18, np.int64) for g in ("code d1-7", "code d8-15", "code d8-15, 125B sure", "inert d8-15")}
    for w in range(len(z["names"])):
        x = byname.get(str(z["names"][w]))
        if x is None:
            continue
        ids, off = z[f"{w}_ids"], int(z[f"{w}_off"])
        assert list(ids) == list(x["ids"])
        ck, cand, tsm = z[f"{w}_checks"], z[f"{w}_cand"], z[f"{w}_tsm"].astype(np.float32)
        for c, (k, m, A) in enumerate(ck):
            m, A = int(m), int(A)
            D = min(15, m, len(ids) - (A - off) - 1)
            order = np.argsort(-tsm[c, :D], axis=1, kind="stable")
            ranked = np.take_along_axis(cand[c, :D], order, axis=1)
            for d in range(D):
                p = A - off + 1 + d
                t = int(ids[p])
                if int(ranked[d, 0]) == t:
                    continue
                hit = np.nonzero(ranked[d] == t)[0]
                r_ = int(hit[0]) + 1 if len(hit) else 17
                fc = x["fine"].get(p, ("prose", ""))[0]
                if fc in INERT:
                    if d >= 7:
                        rk["inert d8-15"][r_] += 1
                elif fc not in ("prose", "fence"):
                    rk["code d1-7" if d < 7 else "code d8-15"][r_] += 1
                    if d >= 7 and (p1_of(x, p) or 0) >= 0.99:
                        rk["code d8-15, 125B sure"][r_] += 1
    print("\n=== R6 CANDIDATES (HumanEval+ 24, the candidate walk): rank of the 125B's token among DFlash2's 16 candidates where the")
    print("    teacher-forced pick is wrong")
    for g, v in rk.items():
        n = max(v.sum(), 1)
        print(f"  {g:24s} n {int(v.sum()):4d}: rank 2 {100 * v[2] / n:5.1f} %, rank 3 {100 * v[3] / n:5.1f} %, 4-16 {100 * v[4:17].sum() / n:5.1f} %, "
              f"not a candidate {100 * v[17] / n:5.1f} %")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("ideas", "all"):
    ideas_main(sets)


def tail_main(sets):
    """R7 AR-TAIL ceiling, R8 ALPHA's realistic subset (lines written 10:47)"""
    import keyword as kwd
    mb94 = ("MBPP 94", [x for x in sets[0][1] if not x["capped"]])
    for label, S in [mb94, sets[0], sets[1], sets[2]]:
        base = run(S, CD16, "B"); ex = run(S, None, "B")
        print(f"\n=== {label}: exact {ex['kept']:.2f}; CD16-B kept {base['kept']:.2f}")
        for D0 in (8, 4, 2, 0):
            r = run(S, CD16, "B", refresh=1, d0=D0)
            e = run(S, None, "B", refresh=1, d0=D0)
            print(f"  R7 tail past depth {D0}: CD16-B kept {r['kept']:6.2f} (x{r['kept'] / base['kept']:.3f}), rows {r['rowsper']:5.2f}; exact acceptance {e['kept']:6.2f} (x{e['kept'] / ex['kept']:.3f})")
        cen = []
        run(S, CD16, "B", census=cen)
        def real(why, x, q, a_ci=None):
            return why[0] == "code: ident-new" and why[1].endswith(":start")
        names = {}
        def f(why, x, q):
            return why[0] == "code: ident-new" and why[1].endswith(":start")
        r = run(S, CD16, "B", forgive=f)
        n1 = sum(1 for c in cen if c["stop"][0] == "code: ident-new")
        n2 = sum(1 for c in cen if c["stop"][0] == "code: ident-new" and c["stop"][1].endswith(":start"))
        nn = 0
        for c in cen:
            if c["stop"][0] == "code: ident-new" and c["stop"][1].endswith(":start"):
                x = c["x"]; i = x["idx"][c["a"]]
                # the draft's token at the stop row: on the own chain when nothing was forgiven before it
                d = dec(int(x["dt"][i][c["k"]])) if c["q"] == c["a"] + 1 + c["k"] else ""
                t = d.strip().lstrip("([{,*")
                nn += bool(t) and (t[0].isalpha() or t[0] == "_") and not kwd.iskeyword(t)
        print(f"  R8 ALPHA: stops on a new identifier {n1} of {len(cen)} checks; at its first token {n2}; with a name-like draft there {nn}")
        print(f"     an oracle on 'first token of a new identifier': kept {r['kept']:.2f} (x{r['kept'] / base['kept']:.3f}); scaled to the name-like share: x{1 + (r['kept'] / base['kept'] - 1) * nn / max(n2, 1):.3f}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("tail", "all"):
    tail_main(sets)


def guard_main(sets):
    """Fri 21:2x, the guarded INERT forms (build_1106/inert_span.py): K by replay. pi = the odds that a guarded row on the
    drafter's own text is kept (the 125B's view of that text is not on disk); the engine's failed CD16 run gives 0.85-0.87
    for rule C's form on polluted text (data/fday logs), so 0.85 is the centre, 0.7 and 1.0 the range"""
    G = load_mod("inert_span_1106", f"{HERE}/build_1106/inert_span.py")
    keys = set(open(f"{HERE}/mbpp12_1106/mbpp12_keys.txt").read().strip().split(","))
    mb = sets[0][1]
    m12 = [x for x in mb if "mbpp/" + re.search(r"mbpp(\d+)", x["name"]).group(1) in keys]
    m90 = [x for x in mb if len(x["ids"]) - x["gen"] < 500]
    arms = [("CD16 (wholesale; failed MBPP quality)", G.parse_arm("CD16")), ("T16 comments + docstrings, top-3 at 0.1 p1", G.parse_arm("T16")),
            ("TC16 comments only, the same guard", G.parse_arm("TC16")), ("U16 top-8 at 0.01 p1", G.parse_arm("U16")),
            ("rule C as shipped (comments, no cut lift)", dict(doc=None, com="ruleC", rows=16, name="RC", doctest=True))]
    global INM
    INM = G
    for label, S in (("HumanEval+ fresh 12", sets[1][1]), ("MBPP12 (the engine set)", m12), ("MBPP 90 (answers that finish)", m90)):
        ex = run(S, None, "B")
        print(f"\n=== {label}: exact kept {ex['kept']:.2f}, rows {ex['rowsper']:.2f}")
        for nm, arm in arms:
            out = []
            for pi in ((1.0,) if arm["name"] == "CD16" else (0.7, 0.85, 1.0)):
                rs = [run(S, arm, "B", pi_dev=pi, seed=sd) for sd in ((0,) if pi == 1.0 else (0, 1, 2, 3, 4))]
                k = np.mean([r["kept"] for r in rs]); rw = np.mean([r["rowsper"] for r in rs]); ln = np.mean([r["lenr"] for r in rs])
                t = np.mean([1000 * r["E"] / (r["checks"] * 50 + r["rows"] * 1.1) for r in rs]) / (1000 * ex["E"] / (ex["checks"] * 50 + ex["rows"] * 1.1))
                out.append(f"pi {pi:.2f}: kept {k:5.2f} (x{k / ex['kept']:.3f}) rows {rw:5.2f} tok/s x{t:.3f} len {ln:.3f}")
            print(f"  {nm:44s} " + " | ".join(out))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "guard":
    guard_main(sets)


# the odds a guarded row is kept, by depth, measured on the 125B's own text (guard_pi.py, Fri 21:14): T3 on comment / docstring rows
PI_T3 = {"he": {"comment": [0.98, 0.95, 0.89, 0.85, 0.79, 0.75, 0.69, 0.64, 0.58, 0.51, 0.49, 0.44, 0.41, 0.38, 0.38],
                "docstring": [1.00, 1.00, 1.00, 1.00, 1.00, 0.99, 0.99, 0.97, 0.96, 0.93, 0.90, 0.87, 0.85, 0.81, 0.78]},
         "mbpp": {"comment": [0.96, 0.87, 0.77, 0.69, 0.61, 0.55, 0.48, 0.43, 0.37, 0.33, 0.31, 0.27, 0.25, 0.23, 0.22],
                  "docstring": [0.98, 0.94, 0.89, 0.83, 0.78, 0.73, 0.69, 0.63, 0.61, 0.57, 0.53, 0.51, 0.48, 0.44, 0.41]}}


def lift_main(sets):
    """T16 with the cut's floor for guarded rows at 1.0 (sure) / 0.9 / 0.8 / none, the odds after a deviation by depth"""
    G = load_mod("inert_span_1106", f"{HERE}/build_1106/inert_span.py")
    keys = set(open(f"{HERE}/mbpp12_1106/mbpp12_keys.txt").read().strip().split(","))
    mb = sets[0][1]
    m12 = [x for x in mb if "mbpp/" + re.search(r"mbpp(\d+)", x["name"]).group(1) in keys]
    m90 = [x for x in mb if len(x["ids"]) - x["gen"] < 500]
    global INM
    INM = G
    costs = ((62, 2.8), (51, 1.1))
    for label, S, dom in (("HumanEval+ fresh 12", sets[1][1], "he"), ("MBPP12 (the engine set)", m12, "mbpp"), ("MBPP 90 (answers that finish)", m90, "mbpp")):
        ex = run(S, None, "B")
        te = [1000 * ex["E"] / (ex["checks"] * f0 + ex["rows"] * r0) for f0, r0 in costs]
        print(f"\n=== {label}: exact kept {ex['kept']:.2f}, rows {ex['rowsper']:.2f}; tok/s at 62 + 2.8 per row {te[0]:.1f}, at 51 + 1.1 per row {te[1]:.1f}")
        pf = lambda kind, j, _d=dom: PI_T3[_d][kind][min(j, 14)]
        for nm in ("T16", "TC16"):
            for lf in (1.0, 0.9, 0.8, 0.0):
                rs = [run(S, G.parse_arm(nm), "B", pi_dev=pf, seed=sd, lift=lf) for sd in range(8)]
                k = np.mean([r["kept"] for r in rs]); rw = np.mean([r["rowsper"] for r in rs]); ln = np.mean([r["lenr"] for r in rs])
                t = [np.mean([1000 * r["E"] / (r["checks"] * f0 + r["rows"] * r0) for r in rs]) for f0, r0 in costs]
                print(f"  {nm:5s} cut floor {lf:.1f}: kept {k:5.2f} (x{k / ex['kept']:.3f})  rows {rw:5.2f}  tok/s x{t[0] / te[0]:.3f} | x{t[1] / te[1]:.3f}  tokens per task {ln:.3f}")
        r = run(S, G.parse_arm("CD16"), "B")
        t = [1000 * r["E"] / (r["checks"] * f0 + r["rows"] * r0) for f0, r0 in costs]
        print(f"  CD16 (reference): kept {r['kept']:5.2f} (x{r['kept'] / ex['kept']:.3f})  rows {r['rowsper']:5.2f}  tok/s x{t[0] / te[0]:.3f} | x{t[1] / te[1]:.3f}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "lift":
    lift_main(sets)
