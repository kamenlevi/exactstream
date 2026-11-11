"""M-31 HEADPING summary (MoonShots, Tue 2026-11-03; pre-registered in ~/NewIdeas/QUEUE.md M-31 before any timing).
Line: H1 (a 4 KB ping of the record's head, keepalive on) at Delta = 100 / 300 / 1,000 us cuts the full read by >= 30 us
vs H0 (no ping): PASS; every Delta < 10 us: KILL; else BETWEEN. H2-H5 and the 2-record call tell the mechanism.
usage: python ssdmicro4_sum.py raw.tsv [out.txt] -> data/ssdmicro4.txt"""
import os, sys, time
import numpy as np

OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/MoonShots/data/ssdmicro4.txt")
rows, meta = {}, []
for ln in open(sys.argv[1]):
    f = ln.rstrip("\n").split("\t")
    if ln.startswith("#"):
        meta.append(ln[2:].strip())
    elif f[0] != "round":
        rows.setdefault((f[2], int(f[4]), int(f[5]), int(f[6])), []).append((float(f[7]), float(f[8])))
R = {k: np.array(v) for k, v in rows.items()}
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def med(k, c=0):
    return float(np.median(R[k][:, c]))


H0, H0off = ("H0", 0, 50, 1), ("H0", 0, 0, 1)
say(f"== M-31 HEADPING ({time.strftime('%a %H:%M')}): {len(R)} points x {len(R[H0])} rounds; {'; '.join(meta)}")
say(f"  full base-part read (2,162,688 B, QD 1) after 0.8 ms idle: keepalive on {med(H0):.1f} us, off {med(H0off):.1f}")
desc = {"H1": "4 KB of its head", "H2": "the part's last 4 KB", "H3": "the next record's head", "H4": "a far record's head",
        "H5": "64 KB of its head", "G1": "both heads (2-record GCD call)"}
for k in sorted(R, key=lambda k: (k[2] == 0, k[0], k[1])):
    if k[0] in ("H0", "G0"):
        continue
    base = H0 if k[3] == 1 and k[2] else H0off if k[3] == 1 else ("G0", 0, 50, 2)
    say(f"    {k[0]} ping {desc[k[0]]:31s} Delta {k[1]:5d} us, keepalive {'on ' if k[2] else 'off'}: {med(k):7.1f} us "
        f"({med(k) - med(base):+6.1f} vs no ping); ping {med(k, 1):6.1f} us")
cut = {d: med(H0) - med(("H1", d, 50, 1)) for d in (100, 300, 1000)}
say(f"  HEAD line: head ping cuts the full read by " + " / ".join(f"{cut[d]:.1f}" for d in cut) + " us at Delta 100 / 300 / "
    "1,000 -> " + ("PASS" if max(cut.values()) >= 30 else "KILL" if max(cut.values()) < 10 else "BETWEEN") + " (line 30 / 10 us)")
open(OUT, "a").write("\n".join(lines) + "\n")
