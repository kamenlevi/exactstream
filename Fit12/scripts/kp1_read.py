"""Fit12 (Fri 6 Nov 09:5x; 12 GB plan lever 3): read /Users/kamen/Fit12/scripts/kp1.sh's run (fit_t3.py stdout): per arm the
tokens = ex, decode tok/s (pooled, and the per-prompt geometric-mean ratio against its base arm: exd vs ex, t80ck2* vs
t80ck2), kept per check, ms per check by part (draft / pass 1 / exact or pass 2 / commit / rest, weighted by checks), each
phase's ms and MB per run (PHASES lines), lifetime peak and the ports' engagement counters (KP).
Usage: python -I -B kp1_read.py [/Users/kamen/Fit12/data/scr/kp1/kp1.txt] [TAG]  -> prints a table, writes kp1_read.json"""
import json, math, os, re, sys

F = sys.argv[1] if len(sys.argv) > 1 else "/Users/kamen/Fit12/data/scr/kp1/kp1.txt"
TAG = sys.argv[2] if len(sys.argv) > 2 else "kp1"
BASE = {"exd": "ex", "t80ck2a": "t80ck2", "t80ck2n": "t80ck2", "t80ck2f": "t80ck2"}
RX = re.compile(rf"^\[{TAG}\] (\S+) (\S+) theta .*?\| tok (\d+) .*?\| = ex tokens (True|False) \(first divergence (\S+)\) "
                r"\| checks (\d+), kept ([\d.]+) \| pass2 (\d+) exact (\d+) cfgate (\d+) .*?\| prefill ([\d.]+) s, decode "
                r"([\d.]+) s, ([\d.]+) tok/s \| lifetime ([\d.]+) GB \| per check ms: draft ([\d.]+) pass 1 ([\d.]+) pass 2 / "
                r"exact ([\d.]+) commit ([\d.]+);(.*)$")
RP = re.compile(rf"^\[{TAG}\] (\S+) (\S+) PHASES (.*)$")
RPH = re.compile(r"(\w+): (\d+) runs, ([\d.]+) ms, (\d+) MB")
SKIP = set(os.environ.get("KP_SKIP", "").split(",")) - {""}   # whole prompts left out (e.g. disturbed by a neighbour)
rows, ph = [], {}
for ln in open(F):
    if any(f" {k} " in ln for k in SKIP):
        continue
    m = RX.match(ln.strip())
    if m:
        g = m.groups()
        kp = {}
        if " | KP " in g[18]:
            for k, v in re.findall(r"(\w+) (\d+)(?:,|$)", g[18].split(" | KP ", 1)[1].split("): ", 1)[-1]):
                kp[k] = int(v)
        rows.append(dict(key=g[0], arm=g[1], tok=int(g[2]), eq=g[3] == "True", div=g[4], checks=int(g[5]), kept=float(g[6]),
                         pass2=int(g[7]), exact=int(g[8]), cfgate=int(g[9]), pf=float(g[10]), dec=float(g[11]),
                         tps=float(g[12]), life=float(g[13]), draft=float(g[14]), p1=float(g[15]), p2=float(g[16]),
                         commit=float(g[17]), kp=kp))
        continue
    m = RP.match(ln.strip())
    if m:
        for name, n, ms, mb in RPH.findall(m.group(3)):
            a = ph.setdefault(m.group(2), {}).setdefault(name, [0, 0.0, 0.0])
            a[0] += int(n); a[1] += int(n) * float(ms); a[2] += int(n) * float(mb)
arms = list(dict.fromkeys(r["arm"] for r in rows))
by = {a: {r["key"]: r for r in rows if r["arm"] == a} for a in arms}
out = {}
print(f"{F}: {len(rows)} arm-prompt rows, arms {arms}")
print("arm      | n  | =ex   | tok/s pooled | x base (geo, per prompt) | kept | ms/check: draft  pass1  exact  commit  rest  "
      "total | lifetime GB")
for a in arms:
    R = list(by[a].values())
    tok, dec, ch = sum(r["tok"] for r in R), sum(r["dec"] for r in R), sum(r["checks"] for r in R)
    part = {p: sum(r[p] * r["checks"] for r in R) / max(1, ch) for p in ("draft", "p1", "p2", "commit")}
    tot = 1e3 * dec / max(1, ch)
    rest = tot - sum(part.values())
    b = BASE.get(a)
    ratio = None
    if b in by:
        ks = [k for k in by[a] if k in by[b]]
        ratio = math.exp(sum(math.log(by[a][k]["tps"] / by[b][k]["tps"]) for k in ks) / max(1, len(ks)))
    kp = {}
    for r in R:
        for k, v in r["kp"].items():
            kp[k] = kp.get(k, 0) + v
    o = dict(n=len(R), eq=sum(r["eq"] for r in R), tok=tok, dec_s=dec, tps=tok / max(1e-9, dec), ratio=ratio,
             base=b, checks=ch, kept=tok / max(1, ch), ms=dict(part, rest=rest, total=tot), life=max(r["life"] for r in R),
             modes=dict(pass2=sum(r["pass2"] for r in R), exact=sum(r["exact"] for r in R), cfgate=sum(r["cfgate"] for r in R)),
             phases={k: dict(runs=v[0], ms=v[1] / max(1, v[0]), mb=v[2] / max(1, v[0])) for k, v in ph.get(a, {}).items()},
             kp=kp)
    out[a] = o
    rs = f"x{ratio:.3f} vs {b}" if ratio else "-"
    print(f"{a:8s} | {o['n']:2d} | {o['eq']:2d}/{o['n']:<2d} | {o['tps']:12.2f} | {rs:24s} | {o['kept']:4.2f} | "
          f"{part['draft']:13.1f} {part['p1']:6.1f} {part['p2']:6.1f} {part['commit']:7.1f} {rest:5.1f} {tot:6.1f} | {o['life']:.2f}")
print("phases per run (ms, MB read):")
for a in arms:
    print(f"  {a:8s} " + " | ".join(f"{k} {v['runs']} runs {v['ms']:.1f} ms {v['mb']:.0f} MB" for k, v in out[a]["phases"].items())
          + (f" || KP {out[a]['kp']}" if out[a]["kp"] else ""))
json.dump(dict(skip=sorted(SKIP), arms=out), open(os.path.join(os.path.dirname(F), f"{TAG}_read{'_skip' if SKIP else ''}.json"), "w"), indent=1)
