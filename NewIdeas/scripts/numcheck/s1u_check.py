"""NUMBER CHECK of Fit12's S1u (Mon 9 Nov 22:37 run): re-derive the headline numbers from s1uh.txt / s1ue.txt per-prompt lines."""
import re, statistics as st, sys, json, glob, os
D = os.path.expanduser("~/Fit12/data/scr/s1u")
def rows(fn, arm):
    out = {}
    for l in open(os.path.join(D, fn)):
        m = re.search(r"\] (\S+) " + arm + r" theta .*?\| tok (\d+) sha (\w+) .*?checks (\d+), kept ([\d.]+).*?prefill ([\d.]+) s, decode ([\d.]+) s, ([\d.]+) tok/s \| lifetime ([\d.]+) GB", l)
        if m:
            k = m.group(1); out[k] = dict(n=int(m.group(2)), sha=m.group(3), checks=int(m.group(4)), kept=float(m.group(5)),
                                         pf=float(m.group(6)), dec=float(m.group(7)), tps=float(m.group(8)), mem=float(m.group(9)))
        m = re.search(r"PAIR (\S+): .*PAUSE ([\d.]+) s inside -> (\w+)", l)
        if m and k in out: out[m.group(1)]["pair"] = (float(m.group(2)), m.group(3))
    return out
def summ(R, label):
    n = sum(r["n"] for r in R.values()); dec = sum(r["n"] / r["tps"] for r in R.values()); pf = sum(r["pf"] for r in R.values())
    geo = st.geometric_mean([r["tps"] for r in R.values()])
    pfs = sorted(r["pf"] for r in R.values())
    print(f"{label}: prompts {len(R)}, tokens {n}, checks {sum(r['checks'] for r in R.values())}, kept/check {n / sum(r['checks'] for r in R.values()):.2f} | "
          f"whole decode pooled {n / dec:.2f} (geo {geo:.2f}) | whole request pooled {n / (dec + pf):.2f} | prefill median {st.median(pfs):.2f} s "
          f"({pfs[0]:.2f}-{pfs[-1]:.2f}) | peak {max(r['mem'] for r in R.values()):.3f} GB | sum decode {dec:.2f} s")
U = rows("s1uh.txt", "u01b"); E = rows("s1ue.txt", "ex")
summ(U, "s1uh u01b (all 16)")
Ev = {k: v for k, v in E.items() if v.get("pair", (0, "ok"))[1] == "ok"}
summ(E, "s1ue ex (all ran)"); summ(Ev, "s1ue ex (VOID dropped)")
common = [k for k in Ev if k in U]
summ({k: U[k] for k in common}, f"s1uh on the {len(common)} common prompts"); summ({k: Ev[k] for k in common}, f"s1ue on the {len(common)} common prompts")
same = sum(U[k]["sha"] == Ev[k]["sha"] for k in common)
print(f"tokens equal (sha) on {same} / {len(common)} common prompts: " + ", ".join(f"{k.split('/')[-1]}{'=' if U[k]['sha'] == Ev[k]['sha'] else '!'}" for k in common))
