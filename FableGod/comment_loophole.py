#!/usr/bin/env python3 -I
"""comment_loophole.py (FableGod, mission 25). CPU, no numpy. Regions from the token stream as the engine could see them
at draft time (code / '#' comment / docstring prose / doctest lines inside docstrings / string literals), per tested
position of DFlash2's walks (HumanEval+ 24 windows at 16 rows; MBPP 100 and agent 12 from MATCH's base16 walks),
with the 125B's class and top-8 on the 12 map windows. Per region: share, class mix, drafter accuracy, share of
misses. Then relaxed acceptance restricted to '#' comments and docstring prose (never code, doctests, strings):
R1 at 0.5 nat, R2 top-2 at >= 0.25 p1, R2 top-3 at >= 0.25 p1: per-token rate, deviation (all inside prose), and the
kept ratio from the per-depth profile (to be halved by the MATCH correction)."""
import zipfile, struct, array, json, os, re, math
from collections import defaultdict, Counter
TD = os.path.expanduser("~/TrainLab/data"); OUT = os.path.expanduser("~/FableGod/decision_map/out")
def npz(path):
    z = zipfile.ZipFile(path); out = {}
    for i in z.infolist():
        d = z.read(i.filename); hl = struct.unpack("<H", d[8:10])[0]; hdr = d[10:10 + hl].decode("latin1"); body = d[10 + hl:]
        descr = re.search(r"'descr': '([^']+)'", hdr).group(1)
        shape = tuple(int(x) for x in re.findall(r"\d+", re.search(r"'shape': \(([^)]*)\)", hdr).group(1)))
        nm = i.filename[:-4]
        if descr.startswith("<U"):
            w = int(descr[2:]); n = shape[0] if shape else 1
            out[nm] = [body[j * w * 4:(j + 1) * w * 4].decode("utf-32-le").rstrip("\x00") for j in range(n)]
        elif descr == "<f2": n = len(body) // 2; out[nm] = (list(struct.unpack(f"<{n}e", body)), shape)
        else:
            a = array.array({"<i2": "h", "<i4": "i", "<i8": "q", "<f4": "f", "<f8": "d"}[descr]); a.frombytes(body); out[nm] = (a, shape)
    return out
vocab = json.load(open(os.path.expanduser("~/BigMoE/models/bare/vocab.json"))); id2tok = {v: k for k, v in vocab.items()}
def dec(t): return id2tok.get(t, "?").replace("Ġ", " ").replace("Ċ", "\n")
base = json.load(open(f"{OUT}/baseline.json"))

def regions(ids):
    """region per token index: 'code' | 'comment' | 'docstring' | 'doctest' | 'string'; a line-based scan over the decoded text."""
    pieces = [dec(t) for t in ids]; text = "".join(pieces); starts = []; o = 0
    for s in pieces: starts.append(o); o += len(s)
    reg = ["code"] * len(text); in_doc = False; doc_q = None; last_doctest = False
    for line in text.split("\n"):
        # 'line' positions
        ls = text.find(line, 0) if False else None
    # character scan with explicit line handling
    i = 0; n = len(text); in_str = None; in_comment = False; line_start = 0; doctest_line = False
    while i < n:
        c = text[i]
        if c == "\n":
            in_str = None; in_comment = False; line_start = i + 1
            # decide the next line's doctest state lazily
            nxt = text[i + 1:text.find("\n", i + 1) if text.find("\n", i + 1) != -1 else n]
            stripped = nxt.lstrip()
            if in_doc:
                if stripped.startswith(">>>") or stripped.startswith("..."): doctest_line = True; last_doctest = True
                elif stripped == "": doctest_line = False; last_doctest = False
                else: doctest_line = last_doctest            # an output line after a >>> line
            else: doctest_line = False
            reg[i] = "doctest" if (in_doc and doctest_line) else ("docstring" if in_doc else ("comment" if False else "code")); i += 1; continue
        if in_doc:
            if text.startswith(doc_q, i):
                for k in range(3): reg[i + k] = "docstring"
                i += 3; in_doc = False; doc_q = None; doctest_line = False; last_doctest = False; continue
            reg[i] = "doctest" if doctest_line else "docstring"; i += 1; continue
        if in_comment: reg[i] = "comment"; i += 1; continue
        if in_str:
            reg[i] = "string"
            if c == "\\":
                if i + 1 < n: reg[i + 1] = "string"
                i += 2; continue
            if c == in_str: in_str = None
            i += 1; continue
        if text.startswith('"""', i) or text.startswith("'''", i):
            doc_q = text[i:i + 3]; in_doc = True
            for k in range(3): reg[i + k] = "docstring"
            i += 3
            # the rest of the opening line is docstring prose unless it is a doctest line
            continue
        if c == "#": in_comment = True; reg[i] = "comment"; i += 1; continue
        if c in "\"'": in_str = c; reg[i] = "string"; i += 1; continue
        reg[i] = "code"; i += 1
    out = []
    for ti, st in enumerate(starts):
        s = pieces[ti]; k = 0
        while k < len(s) - 1 and s[k] in " \t": k += 1            # classify by the first non-blank char of the token
        out.append(reg[min(st + k, n - 1)] if n else "code")
    return out

def walk_positions(path, with_base):
    W = npz(path); P = []
    for i, nm in enumerate(W["names"]):
        if f"{i}_dtok" not in W: continue
        ids = list(W[f"{i}_ids"][0]); checks, (nc, _) = W[f"{i}_checks"]; dtok = W[f"{i}_dtok"][0]; dcf = W[f"{i}_dcf"][0] if f"{i}_dcf" in W else None
        per = len(dtok) // max(nc, 1); R = regions(ids); b = base.get(nm) if with_base else None
        for c in range(nc):
            k, m, A = checks[c * 3], checks[c * 3 + 1], checks[c * 3 + 2]
            for j in range(min(k + 1, m)):
                p = A + j
                if p + 1 >= len(ids): break
                q = dict(reg=R[p + 1], right=j < k, d=j + 1, cf=dcf[c * per + j] if dcf else float("nan"), cls=None, lp=None, tk=None, draft=dtok[c * per + j])
                if b and 0 <= p - b["a"] < len(b["lp"]):
                    lp = b["lp"][p - b["a"]]; p1, p2 = math.exp(lp[0]), math.exp(lp[1])
                    q["cls"] = "sure" if p1 >= 0.99 else ("tie" if p1 - p2 <= 0.1 else "mid"); q["lp"] = lp; q["tk"] = b["tk"][p - b["a"]]
                P.append(q)
    return P
def pct(n, d): return 100.0 * n / d if d else float("nan")
SETS = [("HumanEval+ (24 windows, 12 with classes)", f"{TD}/d02b_walk_hpd0_code.npz", True), ("MBPP (100)", f"{TD}/d02b_walk_m_base16_mbpp.npz", False), ("agent (12)", f"{TD}/d02b_walk_m_base16_agent.npz", False)]
REGS = ["code", "comment", "docstring", "doctest", "string"]
for title, path, wb in SETS:
    if not os.path.exists(path): print(f"\n### {title}: file absent"); continue
    P = walk_positions(path, wb); n = len(P); miss = [q for q in P if not q["right"]]
    print(f"\n### {title}: {n} tested positions, {len(miss)} misses, drafter right {pct(n - len(miss), n):.1f} %")
    print("| region | share of positions | drafter right % | share of misses | class mix sure / mid / tie (classed rows) | mid rows: drafter right % |")
    print("|---|---|---|---|---|---|")
    for r in REGS:
        S = [q for q in P if q["reg"] == r]
        if not S: continue
        C = [q for q in S if q["cls"]]; mid = [q for q in C if q["cls"] == "mid"]
        mix = f"{pct(sum(q['cls'] == 'sure' for q in C), len(C)):.0f} / {pct(sum(q['cls'] == 'mid' for q in C), len(C)):.0f} / {pct(sum(q['cls'] == 'tie' for q in C), len(C)):.0f}" if C else "—"
        print(f"| {r} | {pct(len(S), n):.1f} % | {pct(sum(q['right'] for q in S), len(S)):.1f} | {pct(sum(not q['right'] for q in S), len(miss)):.1f} % | {mix} | {pct(sum(q['right'] for q in mid), len(mid)):.1f} ({len(mid)}) |")
    if wb:
        C = [q for q in P if q["cls"]]
        print(f"mid rows by region (classed rows): " + ", ".join(f"{r} {sum(1 for q in C if q['cls'] == 'mid' and q['reg'] == r)}" for r in REGS) + f"; misses at classed rows by region: " + ", ".join(f"{r} {sum(1 for q in C if not q['right'] and q['reg'] == r)}" for r in REGS))
        # relaxed acceptance restricted to prose (comment + docstring), on classed rows
        def accept(q, rule):
            if q["right"]: return True
            if q["reg"] not in ("comment", "docstring") or q["lp"] is None or q["draft"] not in q["tk"]: return False
            lpd = q["lp"][q["tk"].index(q["draft"])]; p1 = math.exp(q["lp"][0]); pd = math.exp(lpd); rank = sum(1 for v in q["lp"] if v > lpd)
            if rule == "R1 0.5 nat": return lpd >= q["lp"][0] - 0.5
            if rule == "R2 top-2 0.25": return rank < 2 and pd >= 0.25 * p1
            if rule == "R2 top-3 0.25": return rank < 3 and pd >= 0.25 * p1
        def kept(prof, B):
            a = dict(prof); g = min(1.0, (a[15] / a[11]) ** 0.25) if a.get(11) and a.get(15) else 1.0
            tot = 0.0; surv = 1.0; last = None
            for j in range(1, B):
                if j in a: aj = a[j]; last = aj
                elif j <= 15 and last is not None: aj = last
                else: last = (last if last is not None else 0.0) * g; aj = last
                surv *= aj; tot += surv
            return tot
        print("| rule (prose only) | per-token accept % | deviation % of positions (all in prose) | deviation: comment / docstring | kept ratio 16 / 32 (profile model; halve for real) |")
        print("|---|---|---|---|---|")
        for rule in ("exact", "R1 0.5 nat", "R2 top-2 0.25", "R2 top-3 0.25"):
            acc = [q["right"] if rule == "exact" else accept(q, rule) for q in C]
            dev = [(a and not q["right"]) for a, q in zip(acc, C)]
            prof = {d: pct(sum(a for a, q in zip(acc, C) if q["d"] == d), sum(1 for q in C if q["d"] == d)) / 100 for d in range(1, 16)}
            if rule == "exact": base_k = (kept(prof, 16), kept(prof, 32))
            print(f"| {rule} | {pct(sum(acc), len(C)):.2f} | {pct(sum(dev), len(C)):.2f} | {sum(1 for dv, q in zip(dev, C) if dv and q['reg'] == 'comment')} / {sum(1 for dv, q in zip(dev, C) if dv and q['reg'] == 'docstring')} | {kept(prof, 16) / base_k[0]:.3f} / {kept(prof, 32) / base_k[1]:.3f} |")
