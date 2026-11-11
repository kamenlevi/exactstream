"""M-29b summary (MoonShots, Tue 2026-11-03; pre-registered in ~/NewIdeas/QUEUE.md M-29b before any timing): zero-copy at
the 24 GB line's burst sizes. Line: Z3bm (zero-copy into 4 KB-aligned Metal slots) saves >= 50 us per call vs Z0m (bounce
+ copy into Metal slots) at both 8 and 15 records per call: PASS; < 20 us at both: KILL; else BETWEEN.
usage: python ssdmicro2b_sum.py raw.tsv [out.txt] -> data/ssdmicro2.txt (appended)"""
import os, sys, time
import numpy as np

OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/MoonShots/data/ssdmicro2.txt")
rows, meta, cols = {}, [], None
for ln in open(sys.argv[1]):
    f = ln.rstrip("\n").split("\t")
    if ln.startswith("#"):
        meta.append(ln[2:].strip())
    elif f[0] == "round":
        cols = f[8:]
    else:
        rows.setdefault((f[3], int(f[4])), []).append([float(x) for x in f[8:]])
R = {k: np.array(v) for k, v in rows.items()}
C = {c: i for i, c in enumerate(cols)}
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def med(k, c="T_us"):
    return float(np.median(R[k][:, C[c]]))


say(f"== M-29b SSDMICRO2 big bursts ({time.strftime('%a %H:%M')}): {len(R)} points x {len(next(iter(R.values())))} rounds; {'; '.join(meta)}")
say("  records per call: call us (critical job: read + copy) for Z0m today / Z0pm copy split in 3 / Z0n no copy / Z3bm zero-copy")
sv = {}
for n in (4, 8, 15):
    say(f"    {n:2d}: " + " / ".join(f"{a} {med((a, n)):7.1f} ({med((a, n), 'read_us'):6.1f} + {med((a, n), 'copy_us'):5.1f})"
                              for a in ("Z0m", "Z0pm", "Z0n", "Z3bm", "Z1m") if (a, n) in R))
    sv[n] = med(("Z0m", n)) - med(("Z3bm", n))
    if ("Z1m", n) in R:
        say(f"        exploratory: contiguous zero-copy (one read per record) saves {med(('Z0m', n)) - med(('Z1m', n)):.1f} us")
say(f"  M-29b line: zero-copy saves {sv[4]:.1f} / {sv[8]:.1f} / {sv[15]:.1f} us per call at 4 / 8 / 15 records -> "
    + ("PASS" if min(sv[8], sv[15]) >= 50 else "KILL" if max(sv[8], sv[15]) < 20 else "BETWEEN") + " (line 50 / 20 at 8 and 15)")
open(OUT, "a").write("\n".join(lines) + "\n")
