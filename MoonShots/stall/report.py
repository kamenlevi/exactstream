"""Stall anatomy report: per prompt, per pass decomposition of BigMoE's real streamed MTP decode (data/stall/anatomy_*.json)
against the all-hits compute baseline per verify width S (data/stall/base_exact_S*.txt). Per pass: verify ms, compute(S)
(interpolated), exposed stall = verify - compute, demand I/O ms (loader time), overlap = io - exposed stall, misses, stalled
layer-events, SSD bytes (misses x record size), floor = max(compute, bytes / BW). Prints a table per prompt + the totals.
usage: report.py [BW_GBs=14] [RECORD_MB=2.933] [tag, default exact] [baseline tag, default = tag]
       (B12: report.py 14 2.16 b12 tier)"""
import glob, json, os, re, sys
import numpy as np

BW = float(sys.argv[1]) if len(sys.argv) > 1 else 14.0
REC = float(sys.argv[2]) if len(sys.argv) > 2 else 2.932736
TAG = sys.argv[3] if len(sys.argv) > 3 else "exact"
BTAG = sys.argv[4] if len(sys.argv) > 4 else TAG                # baseline env tag (b12 uses the tier baselines)
D = os.path.expanduser("~/MoonShots/data/stall")
base = {}
for f in glob.glob(f"{D}/base_{BTAG}_S*.txt"):
    m = re.search(r"verify pass median ([\d.]+) ms", open(f).read())
    if m:
        base[int(re.search(r"_S(\d+)\.txt$", f).group(1))] = float(m.group(1))
Ss = sorted(base)
comp = lambda S: float(np.interp(S, Ss, [base[s] for s in Ss])) if Ss else float("nan")
print(f"all-hits compute per pass by verify width S ({BTAG} env): " + ", ".join(f"S={s} {base[s]:.1f} ms" for s in Ss))
print(f"assumptions: SSD bandwidth ceiling {BW} GB/s, bytes per miss {REC:.3f} MB\n")
for f in sorted(glob.glob(f"{D}/anatomy_*_{TAG}_f*.json")):
    r = json.load(open(f))
    P = r["passes"]
    if not P:
        continue
    S = np.array([p["S"] for p in P]); ver = np.array([p["t_verify"] for p in P]) * 1e3
    mis = np.array([p["x_misses"] for p in P]); stl = np.array([p["x_stalled"] for p in P])
    io = np.array([p["x_io_s"] for p in P]) * 1e3; acc = np.array([p.get("n", 0) for p in P])
    cmp_ = np.array([comp(s) for s in S]); stall = ver - cmp_
    mb = mis * REC; ssd = mb / BW                                  # ms at the ceiling (MB / (GB/s) = ms)
    floor = np.maximum(cmp_, ssd)
    sm = r["summary"]
    tok = sm.get("tok_per_pass", float("nan"))
    print(f"== {r['prompt']} ({TAG}, C={r['C']}): {sm.get('tok_s', 0):.1f} tok/s, {tok:.2f} tok/pass, {len(P)} passes, mean S {S.mean():.1f}")
    print(f"   per pass (mean): verify {ver.mean():.1f} ms = compute {cmp_.mean():.1f} + exposed stall {stall.mean():.1f}; demand I/O "
          f"{io.mean():.1f} ms -> overlap {io.mean() - stall.mean():.1f} ms; misses {mis.mean():.1f} ({mb.mean():.0f} MB); stalled "
          f"layer-events {stl.mean():.1f} ({mis.sum() / max(1, stl.sum()):.1f} reads each, exposed wait "
          f"{stall.sum() / max(1, stl.sum()):.2f} ms each)")
    print(f"   SSD: achieved {mb.sum() / max(1e-9, io.sum()):.1f} GB/s during demand I/O; bytes/BW floor {ssd.mean():.1f} ms; "
          f"floor max(compute, bytes/BW) {floor.mean():.1f} ms -> gap {ver.mean() - floor.mean():.1f} ms "
          f"({100 * (ver.mean() / floor.mean() - 1):.0f} % above); draft {1e3 * sm.get('t_draft', 0) / len(P):.1f} ms/pass")
    lim = "SSD BYTES" if ssd.mean() > cmp_.mean() else "COMPUTE"
    print(f"   binding at the floor: {lim}; if overlap were perfect (verify = max(compute, io)): "
          f"{np.maximum(cmp_, io).mean():.1f} ms per pass\n")

# ---- compact table across modes (exact / tier = v3 / b12), one line per prompt; lever estimates per pass
if os.environ.get("STALL_TABLE", "1") == "1":
    rows = []
    for tag, btag, rec in (("exact", "exact", 2.932736), ("tier", "tier", 2.162688), ("b12", "tier10", 2.162688)):
        bb = {}
        for f in glob.glob(f"{D}/base_{btag}_S*.txt"):
            m = re.search(r"verify pass median ([\d.]+) ms", open(f).read())
            if m:
                bb[int(re.search(r"_S(\d+)\.txt$", f).group(1))] = float(m.group(1))
        if not bb:
            continue
        ks = sorted(bb)
        cf = lambda S: float(np.interp(S, ks, [bb[k] for k in ks]))
        for f in sorted(glob.glob(f"{D}/anatomy_*_{tag}_f*.json")):
            r = json.load(open(f))
            P = r["passes"]
            if not P:
                continue
            S = np.array([p["S"] for p in P]); ver = np.array([p["t_verify"] for p in P]) * 1e3
            io = np.array([p["x_io_s"] for p in P]) * 1e3; mb = np.array([p["x_misses"] for p in P]) * rec
            cm = np.array([cf(s) for s in S]); ssd = mb / BW
            sm = r["summary"]
            rows.append((tag, r["prompt"], sm.get("tok_s", 0), sm.get("tok_per_pass", 0), S.mean(), ver.mean(), cm.mean(),
                         (ver - cm).mean(), io.mean(), (io - (ver - cm)).mean(), mb.mean(), mb.sum() / max(1e-9, io.sum()),
                         np.maximum(cm, ssd).mean(), np.maximum(cm, io).mean(), np.maximum(cm, 0.55 * ssd).mean()))
    if rows:
        print("mode   prompt  tok/s tok/pass  S   pass  compute stall   I/O  overlap  MB/pass  GB/s  floor  perfect-overlap  floor@0.55 bytes")
        for t in rows:
            print(f"{t[0]:6s} {t[1]:6s} {t[2]:5.1f} {t[3]:6.2f} {t[4]:4.1f} {t[5]:6.1f} {t[6]:6.1f} {t[7]:6.1f} {t[8]:6.1f} {t[9]:7.1f} "
                  f"{t[10]:7.0f} {t[11]:5.1f} {t[12]:6.1f} {t[13]:10.1f} {t[14]:12.1f}")
