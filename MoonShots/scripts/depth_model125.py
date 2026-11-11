"""Round 32i: optimal MTP draft depth in today's engine vs a levered one (pass model, scripts/pass_model125.py).
Per-depth conditional acceptance from the trace's MTP stats (verify_trace125.json meta: accepted/drafted by depth, 5
prompts): pooled d1 0.854, d2 0.796, d3 0.825 (3.09 tokens/pass); beyond depth 3 extrapolated at a(d) = 0.80 - 0.02 (d - 3).
Pass cost at depth D: rows = D + 1, expert union and misses x ((D + 1) / 4)^0.67, plus D draft steps x c_d ms (unknown
until the DRAFT_SYNC profile: c_d = 1.5 and 3 ms).   usage: depth_model125.py   (CPU, instant)"""
import json, os, sys
sys.argv = [sys.argv[0]]
HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location("pm", os.path.join(HERE, "pass_model125.py"))
src = open(os.path.join(HERE, "pass_model125.py")).read().split("STACKS = [")[0]
pm = {}; exec(src, pm)
meta = json.load(open("/Users/kamen/MoonShots/data/verify_trace125.json"))["meta"]
acc = [sum(m["accepted_by_depth"][d] for m in meta) for d in range(3)]
drf = [sum(m["drafted_by_depth"][d] for m in meta) for d in range(3)]
a = [acc[0] / drf[0], acc[1] / acc[0], acc[2] / acc[1]]
print("pooled conditional acceptance d1..d3:", [round(x, 3) for x in a])
def cond(d):
    return a[d - 1] if d <= 3 else max(0.4, 0.80 - 0.02 * (d - 3))
def tokens(D):
    t, p = 1.0, 1.0
    for d in range(1, D + 1):
        p *= cond(d); t += p
    return t
REG = {"today 17 GB": (17.0, set()), "today 34 GB": (34.0, set()),
       "levered 17 GB": (17.0, {"3c", "t3", "eff", "thin", "hide", "opt"}),
       "levered 34 GB": (34.0, {"3c", "t3", "eff", "thin", "hide", "opt"})}
for cd in (1.5, 3.0):
    print(f"\ndraft step cost {cd} ms (the base pass model already includes 3 steps in 'between'; extra steps beyond 3 cost {cd} ms each)")
    for name, (gb, lv) in REG.items():
        row = []
        for D in range(1, 9):
            sc = ((D + 1) / 4) ** 0.67
            _, T = pm["tps"](gb, lv, tok=1.0, scale=sc)
            T += (D - 3) * cd
            row.append((1000 * tokens(D) / T, D))
        best = max(row)
        print(f"  {name:14s} " + " ".join(f"D{D}:{v:5.1f}" for v, D in row) + f"   best D={best[1]} ({best[0]:.1f} tok/s; tokens/pass {tokens(best[1]):.2f})")
