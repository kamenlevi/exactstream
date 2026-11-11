"""b45 scorer (R-44 b, NewIdeas 10-29): A = the 24 GB default (M-01), P = + BIGMOE_POM=4, L = + BIGMOE_POM_LP1, runs
A1 P1 L1 L2 P2 A2 (s10b_trace.py, prompt 4 of each domain, 400 tok). Per arm: whole-process rusage SSD bytes / token, tok/s
(mean of the prompts), gate wait, and the shares the quality run must match (ext.pom_stats per prompt): uses at gate rank
>= 4 served by a 2-plane slot (replay: 20 %) and rank-1 uses served without the LSB (replay: 26 %).
Lines: P vs A PASS bytes/tok <= -12 % with tok/s not worse (>= -1 %, the b44 A1-A2 noise was 0.6 %), KILL a saving under
6 %; L vs P PASS a further <= -6 % with tok/s not worse, KILL a saving under 3 %; else BETWEEN. Both arms are non-exact
(on a PASS -> the Eval suite's strict verdict, then Kamen); their shas are listed, not required to match A.
usage: r44b_ab.py DIR   (PEAK_GB = L's lifetime peak footprint from the runner's probe)"""
import json, os, sys
import numpy as np

D = sys.argv[1]
RUNS = ("A1", "P1", "L1", "L2", "P2", "A2")


def load(r):
    m = json.load(open(os.path.join(D, r, "meta.json")))
    P = m["prompts"]
    pom = np.sum([p["pom"] for p in P if p.get("pom")], axis=0) if any(p.get("pom") for p in P) else np.zeros(9)
    return dict(tok=sum(p["n_tok"] for p in P), disk=sum(p["disk_read"] for p in P),
                gw=sum(p["trace"].get("gate_wait_s", 0) for p in P), tps={p["name"]: p["tok_s"] for p in P},
                sha={p["name"]: p["sha"] for p in P}, pom=pom)


R = {r: load(r) for r in RUNS}


def arm(rs):
    tok = sum(R[r]["tok"] for r in rs)
    pom = np.sum([R[r]["pom"] for r in rs], axis=0)
    return dict(mb=sum(R[r]["disk"] for r in rs) / tok / 1e6, gw=1e3 * sum(R[r]["gw"] for r in rs) / tok,
                tps=float(np.mean([np.mean(list(R[r]["tps"].values())) for r in rs])), pom=pom,
                two=pom[4] / max(pom[3], 1), lp=pom[6] / max(pom[5], 1))


A, Pm, Lm = arm(("A1", "A2")), arm(("P1", "P2")), arm(("L1", "L2"))
print(__doc__.split("\n")[4].strip()); print(__doc__.split("\n")[5].strip())
names = list(R["A1"]["sha"])
print("run   disk MB/tok  gate wait ms/tok  tok/s   2-plane reads  full reads  upgrades  rank>=4 uses at 2 planes  "
      "rank-1 uses w/o LSB  shas")
for r in RUNS:
    q = arm((r,))
    p = q["pom"]
    print(f"{r:4s}  {q['mb']:11.1f}  {q['gw']:16.2f}  {q['tps']:6.2f}  {int(p[0]):13d}  {int(p[1]):10d}  {int(p[2]):8d}  "
          f"{100 * q['two']:23.1f} %  {100 * q['lp']:17.1f} %  " + " ".join(R[r]["sha"][n] for n in names))
noise = {k: 100 * abs(arm(("A2",))[k] / arm(("A1",))[k] - 1) for k in ("mb", "tps")}
print("A1 vs A2 noise: " + ", ".join(f"{k} {v:.1f} %" for k, v in noise.items()))
det = {a: all(R[a + "1"]["sha"][n] == R[a + "2"]["sha"][n] for n in names) for a in "APL"}
print("sha equal across an arm's two runs: " + ", ".join(f"{a} {'yes' if v else 'NO'}" for a, v in det.items())
      + " (P / L depend on residency, so their tokens may move between runs)")
dP = {k: 100 * (Pm[k] / A[k] - 1) for k in ("mb", "gw", "tps")}
dL = {k: 100 * (Lm[k] / Pm[k] - 1) for k in ("mb", "gw", "tps")}
vP = "PASS" if dP["mb"] <= -12 and dP["tps"] >= -1 else "KILL" if dP["mb"] > -6 else "BETWEEN"
vL = "PASS" if dL["mb"] <= -6 and dL["tps"] >= -1 else "KILL" if dL["mb"] > -3 else "BETWEEN"
pk = os.environ.get("PEAK_GB")
print(f"P vs A: bytes/tok {dP['mb']:+.1f} %, gate wait {dP['gw']:+.1f} %, tok/s {dP['tps']:+.1f} %; rank >= 4 uses at 2 "
      f"planes {100 * Pm['two']:.1f} % (replay 20 %)  -> {vP}")
print(f"L vs P: bytes/tok {dL['mb']:+.1f} %, gate wait {dL['gw']:+.1f} %, tok/s {dL['tps']:+.1f} %; rank-1 uses without "
      f"the LSB {100 * Lm['lp']:.1f} % (replay 26 %), rank >= 4 at 2 planes {100 * Lm['two']:.1f} %  -> {vL}")
print(f"L vs A: bytes/tok {100 * (Lm['mb'] / A['mb'] - 1):+.1f} %, tok/s {100 * (Lm['tps'] / A['tps'] - 1):+.1f} %; "
      f"L lifetime peak footprint {pk} GB")
for n in names:
    a, p_, l_ = (np.mean([R[r]["tps"][n] for r in rs]) for rs in (("A1", "A2"), ("P1", "P2"), ("L1", "L2")))
    print(f"    {n}: tok/s A {a:.2f}  P {p_:.2f} ({100 * (p_ / a - 1):+.1f} %)  L {l_:.2f} ({100 * (l_ / a - 1):+.1f} %)")
json.dump(dict(A=A, P=Pm, L=Lm, dP=dP, dL=dL, vP=vP, vL=vL, noise=noise, det=det, peak=pk),
          open(os.path.join(D, "ab.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
