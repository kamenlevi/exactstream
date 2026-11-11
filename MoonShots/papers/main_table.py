"""MAIN paper result table, re-derived from the engine's raw run records (read-only): per config and prompt set,
whole decode (pooled = total tokens / total decode wall; and the reports' geo = geomean over prompts per pass, then
over passes), whole request (pooled tokens / (TTFT + decode)), first token (median TTFT over all prompts and passes,
min-max). TTFT = the record's pfs (prefill + setup + the first token). No GPU, no new runs.
usage: python3 papers/main_table.py > papers/main_table.txt"""
import json, math, os, statistics as st

R = os.path.expanduser("~/BigMoE/data/r57")
GROUPS = [  # (label, tags = one per pass)
    ("45 EXACT E (NAX6-N + levers + KEEP + wired), E5", ["e5h1", "e5h2", "e5m1", "e5m2"]),
    ("45 EXACT EB (E + bitwise Sunday bundle), E5", ["e5hb1", "e5hb2", "e5mb1", "e5mb2"]),
    ("45 LOSSY S5 v1'+wired (= CERT config), S5", ["s5v1", "s5v2", "s5mv1", "s5mv2"]),
    ("45 LOSSY S5 qall (FAST baseline), S5", ["s5q1", "s5q2", "s5mq1", "s5mq2"]),
    ("45 LOSSY S5' B (CERT config), S5'", ["spb1", "spb2", "spmb1", "spmb2"]),
    ("45 LOSSY S5' L = CERT + the bundle (formerly X), S5'", ["spx1", "spx2", "spmx1", "spmx2"]),
    ("45 EXACT EB chat/math, N45", ["nie1", "nie2", "nme1", "nme2"]),
    ("45 LOSSY L = CERT + the bundle, chat/math, N45", ["nil1", "nil2", "nml1", "nml2"]),
    ("45 EXACT E + COH (chat), CHS4 ycoh / ycoh2", ["ycoh", "ycoh2"]),
    ("45 EXACT E + COH + SB1 (chat), CHS4 yes1", ["yes1"]),
    ("45 LOSSY M1-B + COH (chat), CHS5 z0 / z0b", ["z0", "z0b"]),
    ("45 LOSSY M1-B + COH + SB1, CHS5 zs", ["zs"]),
    ("45 LOSSY stack M1-B+COH+SB1+bundle/HC8_G+rank cut 8+R1 (ungated), CHS5 zsgcr", ["zsgcr"]),
    ("45 LOSSY chat as shipped: zs + PF-D v4 + read-ahead, pdz4a", ["pdz4a"]),
    ("45 LOSSY zs, control without PF-D, pdz0", ["pdz0"]),
    ("45 LOSSY L + COH as shipped (snapshot rel1), REL", ["rlh", "rlm", "rli", "rlr", "rlt"]),
    ("32 EXACT e32 (C 160), C32", ["e32"]),
    ("32 LOSSY zs32 (C 160), C32", ["zs32"]),
    ("32 LOSSY L + COH as shipped at C 160 (snapshot rel1), REL", ["rli_160"]),
    ("24 LOSSY w13b as gated (L + COH, 141 + 26 slots, rows <= 13), W24d w13e / w13f", ["w13eh_100", "w13fh_100"]),
    ("24 LOSSY 14-row alternative, W24d w14c / w14d (information)", ["w14ch_100", "w14dh_100"]),
    ("24 LOSSY w13b as gated, W24f bf1 / bf2 (MBPP, IFEval)", ["bf1m_100", "bf2m_100", "bf1i_100", "bf2i_100"]),
    ("24 LOSSY w13b as gated, W24g bg1 / bg2 (math)", ["bg1a_100", "bg2a_100"]),
]


def load(tag):
    ps = [os.path.join(R, f"{tag}_runs.json")] if tag.endswith(("_256", "_160", "_100")) else []   # explicit stem, e.g. rli_160, w13eh_100
    ps += [os.path.join(R, f"{tag}_{c}_runs.json") for c in (256, 160)]   # C 256 (45 GB), C 160 (the 32 GB runs)
    p = next((x for x in ps if os.path.exists(x)), None)
    if p is None:
        return None
    d = json.load(open(p))
    return d if isinstance(d, list) else d.get("runs", [])


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def row(label, s, passes):
    allr = [r for _, rs in passes for r in rs]
    n = sum(r["n"] for r in allr)
    dec = sum(r["n"] / r["tps"] for r in allr)
    ttfts = [r["pfs"] for r in allr if r.get("pfs") is not None]
    geos = [geo([r["tps"] for r in rs]) for _, rs in passes]
    out = (f"{label} | {s} | prompts {len(passes[0][1])} x {len(passes)} pass(es) [{','.join(t for t, _ in passes)}] | "
           f"whole decode pooled {n / dec:.1f} geo {geo(geos):.1f} (passes {' / '.join(f'{g:.1f}' for g in geos)}) | ")
    if ttfts and len(ttfts) == len(allr):
        out += (f"whole request pooled {n / (dec + sum(ttfts)):.1f} | first token median {st.median(ttfts):.2f} s "
                f"({min(ttfts):.2f}-{max(ttfts):.2f}) | ")
    else:
        out += f"whole request / first token: no pfs in {len(allr) - len(ttfts)} of {len(allr)} records | "
    return out + f"tokens/task {n / len(allr):.0f}"


print(__doc__.split("\n")[0])
for label, tags in GROUPS:
    by = {}                                                     # set -> list of passes (tag, runs)
    for t in tags:
        runs = load(t)
        if runs is None:
            print(f"{label}: {t} missing")
            continue
        sets = {}
        for r in runs:
            if r.get("tps") and r.get("n"):
                sets.setdefault(r["key"].split("/")[0], []).append(r)
        for s, rs in sets.items():
            by.setdefault(s, []).append((t, rs))
    for s, passes in sorted(by.items()):
        print(row(label, s, passes))
