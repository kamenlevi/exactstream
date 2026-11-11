"""v1' + wired (S6W's W passes): where the decode wall goes now, and how heavy the slow-check tail is."""
import json, statistics as st
D = "/Users/kamen/BigMoE/data/r57/"
for tags in (("s6w1_256", "s6w2_256"),):
    cks, rows = [], []
    tot = dict(ck=0.0, drf=0.0, other=0.0, wall=0.0)
    comp = dict(io=0.0, svc=0.0, bld=0.0, wt=0.0)
    kept_full, kept_part, n_chk = 0, 0, 0
    for t in tags:
        runs = {r["key"]: r for r in json.load(open(D + t + "_runs.json"))}
        for c in json.load(open(D + t + "_checks.json")):
            r = runs[c["key"]]
            wall = r["n"] / r["tps"] * 1e3
            n = len(c["ck"])
            tot["wall"] += wall; tot["ck"] += sum(c["ck"]); tot["drf"] += sum(c["drf"][:n])
            tot["other"] += wall - sum(c["ck"]) - sum(c["drf"][:n])
            for f in comp:
                v = c.get(f) or []
                comp[f] += sum(v[:n]) if len(v) >= n else 0.0
            for i in range(1, n):
                cks.append(c["ck"][i]); rows.append((c["ck"][i], c["miss"][i], c["io"][i], c["kept"][i], c["rows"][i], c["key"], i))
            n_chk += n
            kept_full += sum(1 for k, rw in zip(c["kept"], c["rows"]) if k == rw)
            kept_part += sum(1 for k, rw in zip(c["kept"], c["rows"]) if k < rw)
    W = tot["wall"]
    print(f"decode wall split: checks {100*tot['ck']/W:.1f} %, drafts {100*tot['drf']/W:.1f} %, other {100*tot['other']/W:.1f} %")
    print("inside checks (sum over checks, % of check time): " + ", ".join(f"{f} {100*v/tot['ck']:.1f} %" for f, v in comp.items()))
    print(f"checks: {n_chk}; full accepts {kept_full} ({100*kept_full/n_chk:.0f} %), partial {kept_part}")
    q = sorted(cks)
    p = lambda x: q[min(len(q) - 1, int(x * len(q)))]
    med = st.median(q)
    print(f"check time (checks >= 1): p10 {p(.1):.1f} p50 {med:.1f} p90 {p(.9):.1f} p99 {p(.99):.1f} max {q[-1]:.1f} ms")
    for k in (1.25, 1.5, 2.0):
        sl = [x for x in q if x > k * med]
        print(f"  checks > {k}x median: {len(sl)} ({100*len(sl)/len(q):.1f} %), excess over median {sum(x - med for x in sl):.0f} ms "
              f"= {100*sum(x - med for x in sl)/W:.1f} % of decode wall")
    rows.sort(reverse=True)
    print("slowest 8 checks (ck ms, misses, io ms, kept/rows, prompt, index):")
    for r in rows[:8]:
        print(f"  {r[0]:6.1f} {r[1]:5.0f} {r[2]:6.1f} {r[3]:2d}/{r[4]:2d} {r[5]} #{r[6]}")
    lo = [r for r in rows if r[1] <= st.median([x[1] for x in rows])]
    hi = [r for r in rows if r[1] > st.median([x[1] for x in rows])]
    print(f"misses <= median: ck median {st.median(r[0] for r in lo):.1f}; misses > median: {st.median(r[0] for r in hi):.1f} ms")
