#!/usr/bin/env python3
"""copygate/gate_sim.py (FableGod copygate, Mon 2 Nov 2026). CPU only; reads TrainLab's MATCH walks read-only.

1. Replays the session match override of ~/TrainLab/sticky/d02b_walk_match.py (its Matcher's lookup, k 16..8, most
   recent occurrence, sources committed up to the check's anchor) teacher-forced at every position a real walk checked.
   At reached positions (p <= k, every earlier draft right) the walk's suffix IS the true text, so the replay is exact
   there: the override's precision where it decides acceptance is measured, not modelled.
2. A drafter model for arms nobody walked: the drafter is right at a token iff the base walk with the same rows was right
   there (token hardness; wrong at the base walk's first misses; tokens it never drafted count as right). Checked per
   check against the real match16 / match32 walks and, on code at 16 rows, against the drafter's own stored blocks.
3. Simulates the official loop (anchors move with the accepted count) per gate: kept per check vs the real base walk.
Usage: ~/BigMoE/.venv/bin/python -I gate_sim.py [valid] [loss] [ex] [gates] [cf] [hyb]   (all of them: gate_sim.out)
Gate names (G()): all, L<k>, ext<n>, prompt, amb8, sh/cl/du:<a>:<b>:<th> (trackers), cf:<th>, cfsh:<a>:<b>, BG<n>,
PX<n>, PBD<a>:<b> (= copygate.py's 'copygate' mode for PBD1:1), joined with & or |
"""
import os, sys
import numpy as np

TD = os.path.expanduser("~/TrainLab/data")
SETS = ("code", "mbpp", "agent")
KS = range(16, 7, -1)


def load(arm, s):
    z = np.load(f"{TD}/d02b_walk_m_{arm}_{s}.npz")
    out = []
    for i in range(len(z["names"])):
        off = int(z[f"{i}_off"])
        out.append(([int(x) for x in z[f"{i}_ids"]], int(z[f"{i}_gen"]) - off,
                    [(int(k), int(m), int(a) - off) for k, m, a in z[f"{i}_checks"]]))
    return out


class Matcher:
    """d02b_walk_match.py's table (k 8-16 -> next index of the most recent occurrence) plus, per k-gram, the count of
    each continuation seen (for ambiguity gates); look(x) = the walk's lookup for ids[x] from the suffix ids[x-k:x]"""
    def __init__(self, ids):
        self.ids, self.upto = ids, 0
        self.T = {k: {} for k in range(8, 17)}
        self.C = {k: {} for k in range(8, 17)}

    def commit(self, r):
        ids = self.ids
        for e in range(self.upto, r):
            nt = ids[e + 1]
            for k in range(8, 17):
                if e - k + 1 < 0:
                    break
                key = tuple(ids[e - k + 1:e + 1])
                self.T[k][key] = e + 1
                c = self.C[k].get(key)
                if c is None:
                    self.C[k][key] = {nt: 1}
                else:
                    c[nt] = c.get(nt, 0) + 1
        self.upto = max(self.upto, r)

    def look(self, x):
        ids = self.ids
        for k in KS:
            if x - k < 0:
                continue
            key = tuple(ids[x - k:x])
            j = self.T[k].get(key)
            if j is not None:
                o, c = ids[j], self.C[k][key]
                n = sum(c.values())
                m = 0
                while m < 256 and j - 1 - m >= 0 and ids[x - 1 - m] == ids[j - 1 - m]:
                    m += 1
                c8 = self.C[8].get(tuple(ids[x - 8:x]), {})
                NL = nl_ids()
                return {"o": o, "L": k, "j": j, "ext": m, "amb": len(c), "cnt": n, "agree": c[o] / n,
                        "bnd": o in NL or ids[j - 1] in NL,
                        "amb8": len(c8), "dist": x - j}
        return None


def hardness(base, other=None):
    """per sample: 1 = the base walk drafted this token right, 0 = its first miss, -1 = never drafted (filled from
    `other`, the base walk at the other row count, where it drafted it)"""
    out = []
    for si, (ids, g, checks) in enumerate(base):
        h = np.full(len(ids), -1, np.int8)
        for k, m, a in checks:
            h[a + 1:a + 1 + k] = 1
            if k < m:
                h[a + 1 + k] = 0
        out.append(h)
    if other is not None:
        for h, h2 in zip(out, hardness(other)):
            u = h < 0
            h[u] = h2[u]
    return out


class State:
    """per-session trackers fed by labels the engine has after each check: cl = overrides (gated or not) at checked
    positions p <= k; sh = the shadow predictor (the walk's lookup, teacher-forced) at every committed answer token"""
    def __init__(self):
        self.cl = [0, 0]
        self.sh = [0, 0]
        self.shp = [0, 0]
        self.shL = {}
        self.duel = [0, 0]
        self.duelb = [0, 0]


_NL = None


def nl_ids():
    """token ids whose text holds a newline (vocab.json, BPE: Ċ = newline)"""
    global _NL
    if _NL is None:
        import json
        v = json.load(open(os.path.expanduser("~/BigMoE/models/bare/vocab.json")))
        _NL = {i for t, i in v.items() if "Ċ" in t}
    return _NL


def gate_all(f, st):
    return True


def simulate(samples, hard, rows, gate, prompt_scan=False, rec=None):
    """the official loop under the drafter model; returns (kept per check, checks, override stats at reached positions)"""
    K, n_chk = 0, 0
    S = {"reach": 0, "hit": 0, "used": 0, "used_ok": 0, "fix": 0, "harm": 0}
    for si, (ids, g, _) in enumerate(samples):
        n, h, mt, st = len(ids), hard[si], Matcher(ids), State()
        if prompt_scan:
            for x in range(1, g + 1):
                mt.commit(x - 1)
                f = mt.look(x)
                if f is not None:
                    st.shp[0 if f["o"] == ids[x] else 1] += 1
        a = g
        while a < n - 1:
            mt.commit(a)
            mm = min(rows - 1, n - 1 - a)
            k, hits = 0, []
            for p in range(mm):
                x = a + 1 + p
                f = mt.look(x)
                dr_ok = h[x] != 0
                S["reach"] += 1
                if f is not None:
                    f["p"] = p
                    f["src"] = "prompt" if f["j"] <= g else "answer"
                    use = gate(f, st)
                    ok_o = f["o"] == ids[x]
                    hits.append((p, f, ok_o, dr_ok))
                    f["_b"] = f["bnd"] and f["ext"] < 24
                    S["hit"] += 1
                    if use:
                        S["used"] += 1
                        S["used_ok"] += ok_o
                        S["fix"] += ok_o and not dr_ok
                        S["harm"] += (not ok_o) and dr_ok
                        ok = ok_o
                    else:
                        ok = dr_ok
                else:
                    ok = dr_ok
                if not ok:
                    break
                k += 1
            K += k + 1
            n_chk += 1
            if rec is not None:
                rec.append((si, a, k, mm))
            for p, f, ok_o, d_ok in hits:                # the check labels every lookup at p <= k
                if p <= k:
                    st.cl[0 if ok_o else 1] += 1
                    if ok_o != d_ok:
                        st.duel[0 if ok_o else 1] += 1
                        if f["_b"]:
                            st.duelb[0 if ok_o else 1] += 1
            for x in range(a + 1, a + k + 2):            # shadow labels on the newly committed tokens
                if x >= n:
                    break
                mt.commit(x - 1)
                f = mt.look(x)
                if f is not None:
                    r = f["o"] == ids[x]
                    st.sh[0 if r else 1] += 1
                    b = st.shL.setdefault(min(f["L"], 16), [0, 0])
                    b[0 if r else 1] += 1
            a += k + 1
    return K / max(n_chk, 1), n_chk, S


def kept(samples):
    ks = [k + 1 for _, _, c in samples for k, m, a in c]
    return float(np.mean(ks)), len(ks)


def valid():
    """V1: the real match walks, exact at reached positions; V2: the model's per-check k vs the real k; V3: G = all
    simulated vs the real walk"""
    print("## V1 / V2: the override at reached positions of the REAL match walks (exact replay) and the model per check")
    for s in SETS:
        b16, b32 = load("base16", s), load("base32", s)
        hard = {16: hardness(b16, b32), 32: hardness(b32, b16)}
        for rows in (16, 32):
            real = load(f"match{rows}", s)
            H = hard[rows]
            reach = hit = hit_ok = bad_replay = 0
            agree = n = 0
            dk = []
            for si, (ids, g, checks) in enumerate(real):
                mt = Matcher(ids)
                h = H[si]
                for k, m, a in checks:
                    mt.commit(a)
                    km = None
                    for p in range(m):
                        x = a + 1 + p
                        f = mt.look(x)
                        if p <= k:
                            reach += 1
                            if f is not None:
                                hit += 1
                                okk = f["o"] == ids[x]
                                hit_ok += okk
                                if p < k and not okk:
                                    bad_replay += 1
                        ok = (f["o"] == ids[x]) if f is not None else (h[x] != 0)
                        if not ok and km is None:
                            km = p
                    km = m if km is None else km
                    agree += km == k
                    n += 1
                    dk.append(km - k)
            kr, nc = kept(real)
            kb, _ = kept(b16 if rows == 16 else b32)
            sim, nsim, _ = simulate(b16 if rows == 16 else b32, H, rows, gate_all)
            print(f"  {s} {rows}: reached positions {reach}, with a match {100 * hit / reach:.1f} %, right "
                  f"{100 * hit_ok / max(hit, 1):.1f} % (replay errors {bad_replay}); model k = real k at "
                  f"{100 * agree / n:.1f} % of {n} checks, mean (model - real) {np.mean(dk):+.3f}; kept real "
                  f"{kr:.3f} ({kr / kb:.3f}x base {kb:.3f}) vs simulated {sim:.3f} ({sim / kb:.3f}x)")



def bucket(name, v):
    edges = {"L": (8, 10, 12, 16, 17), "ext": (8, 12, 16, 24, 32, 64, 999), "cnt": (1, 2, 4, 999),
             "dist": (0, 17, 65, 257, 99999), "p": (0, 4, 8, 16, 32), "amb": (1, 2, 999), "amb8": (0, 1, 2, 999)}[name]
    for lo, hi in zip(edges, edges[1:]):
        if lo <= v < hi:
            return f"{lo}-{hi - 1}" if hi - 1 > lo else f"{lo}"
    return "?"


def loss():
    """at the BASE walk's anchors the drafter is known exactly at p <= k0 (right before its first miss, wrong at it), and
    the override replay is exact there: harm = override wrong where the drafter was right (p < k0); fix = override right
    at the drafter's first miss (p = k0); by feature bucket"""
    print("## LOSS: first-order harm / fix of the override at the base walk's anchors (exact), 16 rows")
    for s in SETS:
        base = load("base16", s)
        tab = {}
        tot = {"hit": 0, "harm": 0, "fix": 0, "both_wrong": 0, "same": 0, "reach": 0}
        for ids, g, checks in base:
            mt = Matcher(ids)
            for k0, m, a in checks:
                mt.commit(a)
                for p in range(min(k0 + 1, m)):
                    x = a + 1 + p
                    tot["reach"] += 1
                    f = mt.look(x)
                    if f is None:
                        continue
                    f["p"] = p
                    f["src"] = "prompt" if f["j"] <= g else "answer"
                    ok = f["o"] == ids[x]
                    cat = ("same" if ok else "harm") if p < k0 else ("fix" if ok else "both_wrong")
                    tot["hit"] += 1
                    tot[cat] += 1
                    for nm in ("L", "ext", "amb", "amb8", "cnt", "dist", "p", "src"):
                        key = (nm, f[nm] if nm == "src" else bucket(nm, f[nm]))
                        t = tab.setdefault(key, {"hit": 0, "harm": 0, "fix": 0, "both_wrong": 0, "same": 0})
                        t["hit"] += 1
                        t[cat] += 1
        print(f"  {s}: reached {tot['reach']}, hits {tot['hit']} ({100 * tot['hit'] / tot['reach']:.1f} %): same {tot['same']}, "
              f"fix {tot['fix']}, harm {tot['harm']}, both wrong {tot['both_wrong']}")
        print("    feature bucket: hits / fix / harm / both-wrong")
        for nm in ("L", "ext", "amb", "amb8", "cnt", "dist", "src", "p"):
            row = [f"{b}: {t['hit']}/{t['fix']}/{t['harm']}/{t['both_wrong']}" for (n_, b), t in
                   sorted(tab.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))) if n_ == nm]
            print(f"    {nm:5s} " + "; ".join(row))


def post(c, a, b):
    return (c[0] + a) / (c[0] + c[1] + a + b)


def G(name):
    """gates by name: 'all', 'L12', 'ext24', 'prompt', 'amb8', 'sh:a:b:th' (shadow precision, Beta(a, b) prior >= th),
    'cl:a:b:th' (check labels), 'du:a:b:th' (duel: override-only-right share among disagreements), '&' / '|' combine"""
    if "|" in name:
        gs = [G(x) for x in name.split("|")]
        return lambda f, st: any(g(f, st) for g in gs)
    if "&" in name:
        gs = [G(x) for x in name.split("&")]
        return lambda f, st: all(g(f, st) for g in gs)
    if name == "all":
        return gate_all
    if name.startswith("BG"):                            # BG<t>: no line-boundary copy (the copy's token or the token
        t = int(name[2:]); NL = nl_ids()                 # it follows in the source holds a newline) unless ext >= t
        return lambda f, st: f["ext"] >= t or not f["bnd"]
    if name.startswith("PBD"):                           # PBD<a>:<b>: prompt & duel(3:1, 0.6) & short line-boundary
        a, b = (float(v) for v in name[3:].split(":")); du = G("du:3:1:0.6")   # copies need their own duel >= 0.6
        return lambda f, st: (f["src"] == "prompt" and du(f, st)
                              and (f["ext"] >= 24 or not f["bnd"] or post(st.duelb, a, b) >= 0.6))
    if name.startswith("PX"):                            # PX<t>: prompt source and (ext >= t or the session's duel says so)
        t = int(name[2:]); du = G("du:3:1:0.6")
        return lambda f, st: f["src"] == "prompt" and (f["ext"] >= t or du(f, st))
    if name == "none":
        return lambda f, st: False
    if name.startswith("L"):
        t = int(name[1:]); return lambda f, st: f["L"] >= t
    if name.startswith("ext"):
        t = int(name[3:]); return lambda f, st: f["ext"] >= t
    if name.startswith("dist"):
        t = int(name[4:]); return lambda f, st: f["dist"] >= t
    if name == "prompt":
        return lambda f, st: f["src"] == "prompt"
    if name == "amb8":
        return lambda f, st: f["amb8"] <= 1
    if name == "amb":
        return lambda f, st: f["amb"] <= 1
    if name.startswith("cfsh:"):                         # the copy when the session's copy precision beats the drafter's cf
        _, a, b = name.split(":"); a, b = float(a), float(b)
        return lambda f, st: post(st.sh, a, b) > f.get("cf", 0.0)
    if name.startswith("cf:"):
        t = float(name[3:]); return lambda f, st: f.get("cf", 0.0) < t
    kind, a, b, th = name.split(":")
    a, b, th = float(a), float(b), float(th)
    attr = {"sh": "sh", "cl": "cl", "du": "duel"}[kind]
    return lambda f, st: post(getattr(st, attr), a, b) >= th


REAL = {}


def gates(names=None, rows_list=(16, 32), prompt_scan=False):
    names = names or ["all", "L12", "L16", "ext24", "ext32", "amb8", "sh:19:1:0.93", "sh:19:1:0.95", "cl:9:1:0.9",
                      "du:3:1:0.6", "BG32", "prompt", "prompt&amb8", "prompt&L10", "prompt&ext12", "prompt&sh:9:1:0.9",
                      "prompt&du:3:1:0.6", "prompt&du:4:1:0.7", "PX24", "prompt&BG24", "prompt&BG32&du:3:1:0.6",
                      "PBD1:1", "PBD2:1"]
    print("## GATES: simulated kept per check / base (the drafter model), 3 sets x rows; the real walk's 'all' beside")
    base = {s: {16: load("base16", s), 32: load("base32", s)} for s in SETS}
    hard = {s: {16: hardness(base[s][16], base[s][32]), 32: hardness(base[s][32], base[s][16])} for s in SETS}
    real = {s: {r: kept(load(f"match{r}", s))[0] / kept(base[s][r])[0] for r in rows_list} for s in SETS}
    simall = {}
    hdr = "  gate".ljust(26) + "".join(f"{s}{r}".rjust(16) for s in SETS for r in rows_list)
    print(hdr)
    print("  real walk (all)".ljust(26) + "".join(f"{real[s][r]:.3f}".rjust(16) for s in SETS for r in rows_list))
    for nm in names:
        g = G(nm)
        cells = []
        for s in SETS:
            for r in rows_list:
                kb = kept(base[s][r])[0]
                sim, _, S = simulate(base[s][r], hard[s][r], r, g, prompt_scan)
                ratio = sim / kb
                if nm == "all":
                    simall[(s, r)] = ratio
                corr = ratio + real[s][r] - simall.get((s, r), ratio)
                cells.append(f"{ratio:.3f}/{corr:.3f}" + f"({100 * S['used'] / max(S['hit'], 1):.0f}%)".rjust(5))
        print(f"  {nm}".ljust(26) + "".join(c.rjust(16) for c in cells))
    print("  cells: simulated ratio / offset-corrected (+ real all - simulated all) (share of reached matches used)")


def load_hpd(dom):
    """HPD-0's walks (= base16 on code, same checks): the drafter's own block at every check (dtok) and its cf (dcf)"""
    z = np.load(f"{TD}/d02b_walk_hpd0_{dom}.npz")
    out = []
    for i in range(len(z["names"])):
        off = int(z[f"{i}_off"])
        out.append(([int(x) for x in z[f"{i}_ids"]], int(z[f"{i}_gen"]) - off,
                    [(int(k), int(m), int(a) - off) for k, m, a in z[f"{i}_checks"]],
                    z[f"{i}_dtok"].astype(np.int64), z[f"{i}_dcf"].astype(np.float64)))
    return out


def per_anchor(samples, gate):
    """exact per anchor: at each base anchor the drafter's real block and cf; the gated block's accepted count there
    (the anchors stay the base walk's: no trajectory change); trackers fed along it"""
    Kb = Kg = 0
    S = {"hit": 0, "used": 0, "fix": 0, "harm": 0, "post_ok": 0, "post_n": 0, "post_model": 0}
    for ids, g, checks, dtok, dcf in samples:
        mt, st, n = Matcher(ids), State(), len(ids)
        for c, (k0, m, a) in enumerate(checks):
            mt.commit(a)
            kg, hits = None, []
            for p in range(m):
                x = a + 1 + p
                d0 = int(dtok[c, p])
                tok = d0
                f = mt.look(x)
                if f is not None:
                    f.update(p=p, src="prompt" if f["j"] <= g else "answer", cf=float(dcf[c, p]), d0=d0)
                    S["hit"] += kg is None
                    if gate(f, st):
                        tok = f["o"]
                        if kg is None:
                            S["used"] += 1
                            S["fix"] += tok == ids[x] and d0 != ids[x]
                            S["harm"] += tok != ids[x] and d0 == ids[x]
                    if kg is None:
                        hits.append((p, f["o"] == ids[x], d0 == ids[x]))
                if kg is None and tok != ids[x]:
                    kg = p
            kg = m if kg is None else kg
            Kb += k0 + 1
            Kg += kg + 1
            for p, ok_o, d_ok in hits:
                st.cl[0 if ok_o else 1] += 1
                if ok_o != d_ok:
                    st.duel[0 if ok_o else 1] += 1
            for x in range(a + 1, a + k0 + 2):
                if x >= n:
                    break
                mt.commit(x - 1)
                f = mt.look(x)
                if f is not None:
                    st.sh[0 if f["o"] == ids[x] else 1] += 1
    return Kg / Kb, S


def cf():
    """the drafter's cf at the override's positions (code, math, IFEval, multi: HPD-0's stored blocks), exact per anchor"""
    print("## CF: exact per-anchor ratios with the drafter's real blocks (HPD-0 = base16 trajectory), 16 rows")
    names = ["all", "prompt", "L12", "prompt&L12", "cf:0.99", "cf:0.9", "cf:0.8", "cfsh:19:1", "cfsh:9:1",
             "prompt&cfsh:19:1", "prompt&cfsh:9:1", "prompt&cf:0.9", "sh:19:1:0.93", "prompt&du:3:1:0.6"]
    sets = {d: load_hpd(d) for d in ("code", "math", "ifeval", "multi")}
    print("  gate".ljust(22) + "".join(d.rjust(22) for d in sets))
    for nm in names:
        g = G(nm)
        cells = []
        for d, smp in sets.items():
            r, S = per_anchor(smp, g)
            cells.append(f"{r:.3f} f{S['fix']}/h{S['harm']} ({100 * S['used'] / max(S['hit'], 1):.0f}%)")
        print(f"  {nm}".ljust(22) + "".join(c.rjust(22) for c in cells))
    print("  cells: ratio over base kept; first-order fixes / harms at reached positions; share of reached matches used")
    # the drafter after its own first miss (what a fix continues into): its real tokens vs the token-hardness model
    for d, smp in sets.items():
        nxt = [0, 0]; dep = {}
        for ids, g, checks, dtok, dcf_ in smp:
            for c, (k0, m, a) in enumerate(checks):
                if k0 + 1 < m:
                    nxt[0] += int(dtok[c, k0 + 1]) == ids[a + k0 + 2]; nxt[1] += 1
        print(f"  {d}: the drafter right at depth k0+2 (just after its first miss) {100 * nxt[0] / max(nxt[1], 1):.1f} % of {nxt[1]}")


def load_hpd_t(dom):
    z = np.load(f"{TD}/d02b_walk_hpd0_{dom}.npz")
    return [z[f"{i}_ttok"].astype(np.int64) for i in range(len(z["names"]))]


def hybrid(samples, ttoks, gate, rechain=False, q_post=0.2, ret_sync=False):
    """the official loop at 16 rows; at a base anchor the drafter's real block (dtok; ttok = the chain re-run with the
    override as predecessor when rechain), elsewhere the token-hardness model, where after a fix the drafter's own next
    pick is right with q_post (measured on code: 20.5 %) unless rechain"""
    Kt = C = sync = 0
    S = {"hit": 0, "used": 0}
    for (ids, g, checks, dtok, dcf), ttok in zip(samples, ttoks):
        n = len(ids)
        bat = {a: c for c, (k0, m, a) in enumerate(checks)}
        h = np.full(n, -1, np.int8)
        for k0, m, a in checks:
            h[a + 1:a + 1 + k0] = 1
            if k0 < m:
                h[a + 1 + k0] = 0
        mt, st, a = Matcher(ids), State(), g
        while a < n - 1:
            mt.commit(a)
            mm = min(15, n - 1 - a)
            c = bat.get(a)
            sync += c is not None
            k, fixed, hits = 0, False, []
            for p in range(mm):
                x = a + 1 + p
                f = mt.look(x)
                if c is not None:
                    dt = int(ttok[c, p] if rechain else dtok[c, p])
                    d_ok = dt == ids[x]
                    cfp = float(dcf[c, p])
                else:
                    d_ok = h[x] != 0
                    if fixed and not rechain:
                        d_ok = d_ok and ((x * 2654435761 + a) % 1000) < q_post * 1000
                    cfp = 0.0
                if f is not None:
                    f.update(p=p, src="prompt" if f["j"] <= g else "answer", cf=cfp)
                    S["hit"] += 1
                    ok_o = f["o"] == ids[x]
                    hits.append((p, ok_o, d_ok))
                    if gate(f, st):
                        S["used"] += 1
                        if ok_o and not d_ok:
                            fixed = True
                        ok = ok_o
                    else:
                        ok = d_ok
                else:
                    ok = d_ok
                if not ok:
                    break
                k += 1
            Kt += k + 1
            C += 1
            for p, ok_o, d_ok in hits:
                if p <= k:
                    st.cl[0 if ok_o else 1] += 1
                    if ok_o != d_ok:
                        st.duel[0 if ok_o else 1] += 1
            for x in range(a + 1, a + k + 2):
                if x >= n:
                    break
                mt.commit(x - 1)
                f = mt.look(x)
                if f is not None:
                    st.sh[0 if f["o"] == ids[x] else 1] += 1
            a += k + 1
    return Kt / C, C, sync / C, S


def hyb(names=None):
    names = names or ["all", "prompt", "L12", "prompt&L12", "prompt&du:3:1:0.6", "prompt&BG24", "PBD1:1"]
    sets = {d: load_hpd(d) for d in ("code", "math", "ifeval", "multi")}
    tt = {d: load_hpd_t(d) for d in sets}
    for d, smp in sets.items():                        # sanity: before the first miss the two chains agree
        bad = sum(int(dk[c, p] != tk[c, p]) for (ids, g, ch, dk, cf_), tk in zip(smp, tt[d]) for c, (k0, m, a) in
                  enumerate(ch) for p in range(min(k0, m)))
        print(f"  {d}: dtok != ttok before the first miss: {bad}")
    print("## HYBRID: 16 rows, real blocks at base anchors, the model elsewhere; ratio over base (share of checks at base anchors)")
    print("  gate".ljust(26) + "".join((d + " " + v).rjust(20) for d in sets for v in ("own", "rechain")))
    for nm in names:
        g = G(nm)
        cells = []
        for d, smp in sets.items():
            kb = np.mean([k0 + 1 for ids, g_, ch, dk, cf_ in smp for k0, m, a in ch])
            for rc in (False, True):
                kt, C, sy, S = hybrid(smp, tt[d], g, rechain=rc)
                cells.append(f"{kt / kb:.3f} ({100 * sy:.0f}%)")
        print(f"  {nm}".ljust(26) + "".join(c.rjust(20) for c in cells))


def examples(s="mbpp", nmax=8):
    import json
    vocab = json.load(open(os.path.expanduser("~/BigMoE/models/bare/vocab.json")))
    inv = {v: k for k, v in vocab.items()}
    dec = lambda t: "".join(inv.get(i, "?") for i in t).replace("\u0120", " ").replace("\u010a", "\\n").replace("Ġ", " ").replace("Ċ", "\\n").replace("ĉ", "\\t")
    base = load("base16", s)
    per = []
    shown = 0
    for si, (ids, g, checks) in enumerate(base):
        mt, fx, hm = Matcher(ids), 0, 0
        for k0, m, a in checks:
            mt.commit(a)
            for p in range(min(k0 + 1, m)):
                x = a + 1 + p
                f = mt.look(x)
                if f is None:
                    continue
                ok = f["o"] == ids[x]
                if p < k0 and not ok:
                    hm += 1
                    if shown < nmax:
                        shown += 1
                        src = "prompt" if f["j"] <= g else "answer"
                        print(f"  harm s{si} L{f['L']} ext{f['ext']} {src} dist {f['dist']}: ...{dec(ids[x - 12:x])!r} -> copy {dec([f['o']])!r} / true {dec([ids[x]])!r}; source ctx {dec(ids[f['j'] - 4:f['j'] + 3])!r}")
                if p == k0 and ok:
                    fx += 1
        per.append((fx, hm))
    hs = sorted((h for f_, h in per), reverse=True)
    print(f"  {s}: sessions with harm {sum(h > 0 for f_, h in per)} of {len(per)}; with a fix {sum(f_ > 0 for f_, h in per)}; "
          f"top-10 sessions hold {sum(hs[:10])} of {sum(hs)} harms; sessions with harm and no fix {sum(h > 0 and f_ == 0 for f_, h in per)}")


def ex():
    print("## EXAMPLES: MBPP's first-order harms (the copy wrong where the drafter was right), and where harms sit")
    examples("mbpp", 10)
    examples("agent", 0)
    examples("code", 0)


if __name__ == "__main__":
    for sec in sys.argv[1:] or ["valid"]:
        globals()[sec]()
