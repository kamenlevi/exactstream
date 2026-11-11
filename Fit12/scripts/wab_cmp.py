"""Fit12 (Sat 7 Nov 10:5x): compare the wired-limit A/B of /Users/kamen/Fit12/scripts/wab.sh. W = F12_WIRED=1 (mx.set_wired_limit
once after load), B = the base; both arm t80ck1u on c1u16's footing, each in its own process. Reads <tag>_report.json (written
by c1u16_read.py) and the per-prompt phase logs; prints whole decode / full-period / end to end (geometric mean, pooled), the
W / B ratio (geometric mean of per-prompt ratios), each run's check-0 excess over later same-mode checks, lifetime peaks, and
tokens W vs B. Usage: python -I -B wab_cmp.py [/Users/kamen/Fit12/data/scr/wab] [wabw] [wabb]"""
import json, math, os, statistics as S, sys

D = sys.argv[1] if len(sys.argv) > 1 else "/Users/kamen/Fit12/data/scr/wab"
TW = sys.argv[2] if len(sys.argv) > 2 else "wabw"
TB = sys.argv[3] if len(sys.argv) > 3 else "wabb"
ARM = "t80ck1u"
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def load(tag):
    rep = json.load(open(os.path.join(D, f"{tag}_report.json")))
    per, ex_ms, dec_ms = {}, 0.0, 0.0
    for r in rep["prompts"]:
        kf = r["key"].replace("/", "_")
        lg = json.load(open(os.path.join(D, f"{tag}_{kf}_{ARM}.json")))
        ph = json.load(open(os.path.join(D, f"{tag}_{kf}_{ARM}_ph.json")))
        by = {}
        for p in ph:
            a = by.setdefault(p[1], [p[2], p[6]])
            a[0], a[1] = min(a[0], p[2]), max(a[1], p[6])
        d = {c: 1e3 * (e - s) for c, (s, e) in by.items()}
        same = [d[c] for c in d if c > 0 and lg[c][1] == lg[0][1]]
        ex = max(0.0, d[0] - S.median(same)) if same else 0.0
        ex_ms += ex
        dec_ms += 1e3 * r["dec"]
        toks = json.load(open(os.path.join(D, f"{tag}_{kf}_{ARM}_toks.json")))
        per[r["key"]] = dict(r, c0=d[0], c0_ex=ex, toks=toks)
    return rep, per, ex_ms, dec_ms


rw, pw, exw, decw = load(TW)
rb, pb, exb, decb = load(TB)
keys = [k for k in pw if k in pb]
for name, rep, ex, dec in (("W (wired)", rw, exw, decw), ("B (base)", rb, exb, decb)):
    print(f"{name:10s}: whole decode geo {rep['whole_geo']:.2f} / pooled {rep['whole_pooled']:.2f} | full-period geo "
          f"{rep['full_geo']:.2f} | e2e geo {rep['e2e_geo']:.2f} | lifetime {rep['lifetime_gb']:.3f} GB | check-0 excess "
          f"{ex:.0f} ms = {100 * ex / dec:.1f} % of decode | {len(rep['prompts'])} prompts, PAUSE {rep['pause']}")
for m in ("whole", "full", "e2e"):
    print(f"W / B {m:5s}: x{geo([pw[k][m] / pb[k][m] for k in keys if pw[k][m] and pb[k][m]]):.4f} (geometric mean of "
          f"per-prompt ratios, {len(keys)} prompts)")
print("check 0 ms W / B per prompt: " + ", ".join(f"{k.split('/')[-1]} {pw[k]['c0']:.0f}/{pb[k]['c0']:.0f}" for k in keys))
same = [k for k in keys if pw[k]["toks"] == pb[k]["toks"]]
print(f"tokens W = B on {len(same)} / {len(keys)} prompts" + ("" if len(same) == len(keys) else "; differ: " + ", ".join(
    f"{k.split('/')[-1]} @{next((i for i, (a, b) in enumerate(zip(pw[k]['toks'], pb[k]['toks'])) if a != b), None)}"
    for k in keys if k not in same)))
json.dump(dict(w=dict(whole_geo=rw["whole_geo"], whole_pooled=rw["whole_pooled"], full_geo=rw["full_geo"], e2e_geo=rw["e2e_geo"],
                      lifetime_gb=rw["lifetime_gb"], c0_excess_ms=exw),
               b=dict(whole_geo=rb["whole_geo"], whole_pooled=rb["whole_pooled"], full_geo=rb["full_geo"], e2e_geo=rb["e2e_geo"],
                      lifetime_gb=rb["lifetime_gb"], c0_excess_ms=exb),
               ratio_whole=geo([pw[k]["whole"] / pb[k]["whole"] for k in keys]), same_tokens=len(same), prompts=len(keys)),
          open(os.path.join(D, "wab_cmp.json"), "w"), indent=1)
