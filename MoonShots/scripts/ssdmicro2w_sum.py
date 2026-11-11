"""Wed 09:0x SSD ticket summary (MoonShots; the orchestrator pulled it forward): the M-29b repeat (the registered verdict
stays the 22:46 run's) and the wake-ahead ping at the engine's 1.2 / 2.0 ms gaps between bursts (M-29's P line:
removal >= 70 % at some T <= 200 us).
usage: python ssdmicro2w_sum.py raw.tsv -> appended to data/ssdmicro2.txt"""
import os, sys, time
import numpy as np

rows, meta, cols = {}, [], None
for ln in open(sys.argv[1]):
    f = ln.rstrip("\n").split("\t")
    if ln.startswith("#"):
        meta.append(ln[2:].strip())
    elif f[0] == "round":
        cols = f[8:]
    else:
        rows.setdefault((f[2], f[3], int(f[4]), f[5], f[7]), []).append([float(x) for x in f[8:]])
R = {k: np.array(v) for k, v in rows.items()}
C = {c: i for i, c in enumerate(cols)}
med = lambda k, c="T_us": float(np.median(R[k][:, C[c]]))
L = [f"== Wed SSD ticket ({time.strftime('%a %H:%M')}): {len(R)} points x {len(next(iter(R.values())))} rounds; {'; '.join(meta)}",
     "  M-29b repeat: records per call -> call us (critical read + copy): Z0m today / Z0n no copy / Z3bm zero-copy"]
for n in (4, 8, 15):
    k = {a: ("Zb", a, n, "g800", "p0") for a in ("Z0m", "Z0n", "Z3bm")}
    L.append(f"    {n:2d}: " + " / ".join(f"{a} {med(k[a]):7.1f} ({med(k[a], 'read_us'):6.1f} + {med(k[a], 'copy_us'):5.1f})" for a in k)
             + f" | zero-copy saves {med(k['Z0m']) - med(k['Z3bm']):.1f} us")
b2b = med(("Pw", "R0", 2, "g0", "p0"))
L.append(f"  wake-ahead ping at the engine's gaps (R0 = 2 records, bounce): back to back {b2b:.1f} us")
for g in (1200, 2000):
    off = med(("Pw", "R0", 2, f"g{g}", "p0"))
    L.append(f"    gap {g / 1e3:.1f} ms: no ping {off:.1f} (+{off - b2b:.1f})" + "".join(
        f"; ping {T} us ahead {med(('Pw', 'R0', 2, f'g{g}', f'p{T}')):.1f} (removes "
        f"{100 * (1 - (med(('Pw', 'R0', 2, f'g{g}', f'p{T}')) - b2b) / (off - b2b)):.0f} %)" for T in (150, 250)))
print("\n".join(L))
open(os.path.expanduser("~/MoonShots/data/ssdmicro2.txt"), "a").write("\n".join(L) + "\n")
