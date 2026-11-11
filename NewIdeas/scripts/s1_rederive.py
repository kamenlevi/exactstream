"""Re-derive BigMoE's S1 NW v3 A/B (data/r57/s1{b1,n1,n2,b2}_256_*.json) independently.

Prints per pass: geo mean of steady tok/s (tpsw) and raw tok/s (tps) over the valid prompts, pooled
tokens / pooled time, mean and median check cycle from the per-check rows, and kept per check.
Then the NW / B ratios: as registered (geo over both passes), per pass pair, and per prompt.
"""
import json, math, statistics as st, sys

D = "/Users/kamen/BigMoE/data/r57/"
ARMS = {"B": ["s1b1_256", "s1b2_256"], "NW": ["s1n1_256", "s1n2_256"]}


def load(tag):
    runs = json.load(open(D + tag + "_runs.json"))
    checks = json.load(open(D + tag + "_checks.json"))
    return runs, checks


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


per = {}
for arm, tags in ARMS.items():
    for tag in tags:
        runs, checks = load(tag)
        ok = [r for r in runs if not r.get("bad")]
        tpsw = {r["key"]: r["tpsw"] for r in ok}
        tps = {r["key"]: r["tps"] for r in ok}
        t_tot = sum(r["n"] / r["tps"] for r in ok)
        n_tot = sum(r["n"] for r in ok)
        cyc_all, kept_all = [], []
        for c in checks:
            if c.get("paused"):
                continue
            ck, drf, kept = c["ck"], c["drf"], c["kept"]
            for i in range(1, len(ck)):  # skip each prompt's first check (prefill-adjacent)
                cyc_all.append(ck[i] + drf[i])
                kept_all.append(kept[i])
        per[tag] = dict(tpsw=tpsw, tps=tps)
        print(f"{arm:2s} {tag}: valid {len(ok)}/{len(runs)}  geo tpsw {geo(tpsw.values()):6.1f}  "
              f"geo tps {geo(tps.values()):6.1f}  pooled {n_tot / t_tot:6.1f}  "
              f"ck+drf mean {st.mean(cyc_all):5.1f} median {st.median(cyc_all):5.1f} ms  "
              f"kept/check {sum(kept_all) / len(kept_all):5.2f}  "
              f"kept/(ck+drf) {1000 * sum(kept_all) / sum(cyc_all):6.1f} tok/s")

keys = sorted(set.intersection(*[set(per[t]["tpsw"]) for t in per]))
print(f"\nprompts valid in all 4 passes: {len(keys)}")


def arm_geo(arm, field):
    # registered form: geo over prompts of the per-prompt mean over the arm's 2 passes
    return geo([st.mean(per[t][field][k] for t in ARMS[arm]) for k in keys])


for field in ("tpsw", "tps"):
    b, n = arm_geo("B", field), arm_geo("NW", field)
    print(f"{field}: B {b:6.1f}  NW {n:6.1f}  NW/B {n / b:6.4f}")

b1, b2 = (geo([per[t]["tpsw"][k] for k in keys]) for t in ARMS["B"])
n1, n2 = (geo([per[t]["tpsw"][k] for k in keys]) for t in ARMS["NW"])
print(f"pairs (tpsw geo): n1/b1 {n1 / b1:.4f}  n2/b2 {n2 / b2:.4f}  n1/b2 {n1 / b2:.4f}  n2/b1 {n2 / b1:.4f}")
print(f"B pass spread b2/b1 {b2 / b1:.4f}; NW pass spread n1/n2 {n1 / n2:.4f}")

rat = []
for k in keys:
    b = st.mean(per[t]["tpsw"][k] for t in ARMS["B"])
    n = st.mean(per[t]["tpsw"][k] for t in ARMS["NW"])
    rat.append((n / b, k, b, n))
rat.sort()
wins = sum(1 for r in rat if r[0] > 1)
print(f"\nper prompt NW/B (tpsw, mean of passes): wins {wins}/{len(rat)}; "
      f"median {st.median(r[0] for r in rat):.4f}")
for r, k, b, n in rat:
    print(f"  {k:28s} B {b:6.1f} NW {n:6.1f}  x{r:.3f}")
# leave-one-out spread of the geo ratio
loo = []
for i in range(len(rat)):
    rest = [r[0] for j, r in enumerate(rat) if j != i]
    loo.append(geo(rest))
print(f"leave-one-prompt-out geo NW/B: min {min(loo):.4f} max {max(loo):.4f}")

print("\nfirst check of each prompt vs the rest (means over the 12 prompts):")
for arm, tags in ARMS.items():
    for tag in tags:
        runs, checks = load(tag)
        rk = {r["key"]: r for r in runs}
        f_ck, f_drf, f_kept, rest_t, tot_t, wall_t, n_t = [], [], [], 0.0, 0.0, 0.0, 0
        for c in checks:
            r = rk[c["key"]]
            f_ck.append(c["ck"][0]); f_drf.append(c["drf"][0]); f_kept.append(c["kept"][0])
            s = [a + b for a, b in zip(c["ck"], c["drf"])]
            rest_t += sum(s[1:]); tot_t += sum(s)
            wall_t += r["n"] / r["tps"] * 1000; n_t += r["n"]
        print(f"  {arm:2s} {tag}: first ck {st.mean(f_ck):6.1f} ms drf {st.mean(f_drf):5.1f} kept {st.mean(f_kept):5.2f} | "
              f"first-check share of ck+drf {100 * (tot_t - rest_t) / tot_t:4.1f} % | "
              f"ck+drf covers {100 * tot_t / wall_t:5.1f} % of n/tps wall")
