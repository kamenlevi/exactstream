"""Read T1's map arms by the 125B's own row class, in fablec/wall_class_flips.py's convention (Fable A, Thu 5 Nov 2026).
CPU, seconds, read-only. A flip = the arm's top-1 differs from the baseline's at an answer row; classes from
~/FableGod/floor_rows.json (sure p1 >= 0.99 / tie p1 - p2 <= 0.1 / mid); non-tie = sure + mid.

Usage: python read_xs.py [--out DIR] [--base FILE] <arm> [<arm> ...]
  --out   the map's out folder (default ~/FableGod/decision_map/out, or $DM_OUT)
  --base  the baseline json (default <out>/baseline.json)
An arm named "<parent>+xs<c>[+code]" is also compared with its parent's file (<parent>.json, else <parent>+code.json),
on the windows both have: the ADDED flips are what T1's lines are written on.
Lines (pre-registered in INSTALL.md). Standalone arms are read on NON-TIE rows: fits if within the room above today's
preset (0.66 / 1.03 / 0.72); the lever is dead if xs32's non-tie code flips exceed 1.03 % or xs48's exceed 0.21 %.
Stacked arms: PASS = all-row <= 1.34 / 5.42 / 3.69 % AND non-tie <= 0.87 / 2.22 / 1.63 %; dead if xs32 adds more than
+1.2 all-row code points over its parent, or xs48 more than +0.25. xs64 must show 0 flips.
"""
import ast, json, os, re, sys
args = sys.argv[1:]
OUT = os.path.expanduser(os.environ.get("DM_OUT", "~/FableGod/decision_map/out")); BASE = None
while args and args[0].startswith("--"):
    if args[0] == "--out": OUT = os.path.expanduser(args[1]); args = args[2:]
    elif args[0] == "--base": BASE = os.path.expanduser(args[1]); args = args[2:]
    else: raise SystemExit(__doc__)
if not args: raise SystemExit(__doc__)
fr = json.load(open(os.path.expanduser("~/FableGod/floor_rows.json")))
CLS = {(r["nm"], int(r["i"])): r["cls"] for r in fr}; DOM = {r["nm"]: r["dom"] for r in fr}
base = json.load(open(BASE or os.path.join(OUT, "baseline.json")))
DOMS = ("code", "ifeval", "multi"); KEYS = ("all", "nontie", "tie", "sure", "mid")
LINE = {"all": (1.34, 5.42, 3.69), "nontie": (0.87, 2.22, 1.63)}
ROOM = (0.66, 1.03, 0.72)          # non-tie room above today's preset (bar minus the twin's 0.21 on code, QOM-BQ's 1.19 / 0.91 on chat)
PRED = {48: {"code": (0.05, 0.07), "ifeval": (0.55, 0.66), "multi": (0.48, 0.54)},      # predict.py -> PREDICT.out
        40: {"code": (0.17, 0.24), "ifeval": (1.82, 2.18), "multi": (1.59, 1.77)},
        32: {"code": (0.42, 0.60), "ifeval": (4.49, 5.38), "multi": (3.93, 4.37)}}
DEAD_NT = {48: 0.21, 32: 1.03}
DEAD_WHY = {48: "top_k8's own cost, at 3x this arm's error", 32: "topk5's own cost, at 1.6x this arm's error"}

def read(path, only=None):
    """{(dom, key): [flips, rows]} over the arm's windows (restricted to `only` when given); plus notes"""
    A = json.load(open(path)); acc = {}; notes = []; wins = set()
    for w, v in A.items():
        if w not in base or w not in DOM or not isinstance(v, dict) or "tk" not in v: continue
        if only is not None and w not in only: continue
        ta, tb = v["tk"], base[w]["tk"]
        if len(ta) != len(tb): notes.append(f"{w}: {len(ta)} rows vs the baseline's {len(tb)} (skipped)"); continue
        if v.get("warn"): notes.append(f"{w}: {v['warn']}")
        wins.add(w); a0 = int(base[w]["a"])
        for j in range(len(ta)):
            c = CLS.get((w, a0 + j))
            if c is None: continue
            fl = int(ta[j][0] != tb[j][0])
            for key in ("all", c, "nontie" if c != "tie" else None):
                if key: x = acc.setdefault((DOM[w], key), [0, 0]); x[0] += fl; x[1] += 1
    xs = sorted({(v["xs"]["c"]) for v in A.values() if isinstance(v, dict) and "xs" in v})
    return acc, notes, wins, xs

def pct(acc, dom, key):
    return 100.0 * acc[(dom, key)][0] / acc[(dom, key)][1] if (dom, key) in acc and acc[(dom, key)][1] else None

def row(acc, key):
    return " / ".join("-" if pct(acc, d, key) is None else f"{pct(acc, d, key):.2f}" for d in DOMS)

print("| arm | rows code / IFEval / multi | all-row flips % | non-tie | tie | sure | mid |"); print("|---|---|---|---|---|---|---|")
verdicts = []
for arm in args:
    p = os.path.join(OUT, arm + ".json")
    if not os.path.exists(p): print(f"| {arm} | (no file {p}) |"); continue
    acc, notes, wins, xs = read(p)
    n = " / ".join(str(acc.get((d, "all"), [0, 0])[1]) for d in DOMS)
    print(f"| {arm} | {n} | " + " | ".join(row(acc, k) for k in KEYS) + " |")
    m = re.fullmatch(r"(.*?)\+?xs(\d+)((?:\+[a-z]+)*)", arm)
    c = int(m.group(2)) if m else None; parent = m.group(1) if m else None
    add = None
    if parent:
        for cand in (parent, parent + "+code"):
            pp = os.path.join(OUT, cand + ".json")
            if os.path.exists(pp):
                pacc, _, _, _ = read(pp, only=wins); cacc, _, _, _ = read(p, only=set(json.load(open(pp)).keys()))
                print(f"| ^ parent {cand}, same windows | | " + " | ".join(row(pacc, k) for k in KEYS) + " |")
                add = {(d, k): (pct(cacc, d, k) - pct(pacc, d, k)) for d in DOMS for k in KEYS if pct(cacc, d, k) is not None and pct(pacc, d, k) is not None}
                print(f"| ^ ADDED by xs{c} | | " + " | ".join(" / ".join("-" if (d, k) not in add else f"{add[(d, k)]:+.2f}" for d in DOMS) for k in KEYS) + " |")
                break
    else:
        add = {(d, k): pct(acc, d, k) for d in DOMS for k in KEYS if pct(acc, d, k) is not None}      # standalone: added over the exact model
    if m:
        if xs and xs != [c]: notes.append(f"the file's xs record says c = {xs}, the name says {c}")
        if c is not None and c < 64 and not xs: notes.append("no xs record in the file: was DM_XS=1 set, and is this the patched harness?")
        if c == 64:
            n_fl = sum(acc.get((d, "all"), [0, 0])[0] for d in DOMS)
            if not parent: verdicts.append(f"{arm}: identity arm, {n_fl} flips: " + ("OK, the wrapper is inert" if n_fl == 0 else "NOT INERT: stop, do not trust the other arms"))
        elif not parent:                                   # standalone: read on non-tie rows (all-row carries the tie floor)
            for i, d in enumerate(DOMS):
                nt = pct(acc, d, "nontie")
                if nt is None: continue
                pr = PRED.get(c, {}).get(d)
                verdicts.append(f"{arm} {d}: non-tie {nt:.2f} %" + (f" (predicted {pr[0]:.2f}-{pr[1]:.2f})" if pr else "") +
                                f"; room above today's preset {ROOM[i]:.2f}: {'FITS' if nt <= ROOM[i] else 'OVER'}")
            nt = pct(acc, "code", "nontie"); lim = DEAD_NT.get(c)
            if nt is not None and lim is not None:
                verdicts.append(f"{arm}: lever check, non-tie code {nt:.2f} % vs the kill level {lim} ({DEAD_WHY[c]}): {'LEVER DEAD' if nt > lim else 'alive'}")
        else:                                              # stacked on a preset: WALL_1105's PASS convention
            for i, d in enumerate(DOMS):
                a, nt = pct(acc, d, "all"), pct(acc, d, "nontie")
                if a is None: continue
                ok = a <= LINE["all"][i] and nt <= LINE["nontie"][i]
                verdicts.append(f"{arm} {d}: all-row {a:.2f} % (line {LINE['all'][i]}), non-tie {nt:.2f} % (bar {LINE['nontie'][i]}): {'PASS' if ok else 'OVER'}")
            lim = {32: 1.2, 48: 0.25}.get(c)
            if add and ("code", "all") in add and lim is not None:
                verdicts.append(f"{arm}: lever check, added all-row code flips {add[('code', 'all')]:+.2f} vs the kill limit +{lim}: {'LEVER DEAD' if add[('code', 'all')] > lim else 'alive'}")
    for t in notes: print(f"  note: {t}")
print(); [print(v) for v in verdicts]
