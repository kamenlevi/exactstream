"""Fit12 PFR A/B compare (2026-10-26): for each arm vs the reference arm, from /Users/kamen/Fit12/data/pfr/ab_<arm>.txt and
ab_<arm>_logits.npy: RESULT fields equal or not, the first cache layer whose KV hash differs (and how many differ), and the
first-step logits' max |diff|, number of differing entries and whether the argmax agrees. CPU only, tiny files.
Usage: [BigMoE venv] -I ab_cmp.py REF ARM [ARM ...]"""
import re, sys
import numpy as np

D = "/Users/kamen/Fit12/data/pfr"


def read(arm):
    txt = open(f"{D}/ab_{arm}.txt").read()
    m = re.search(r" RESULT (.*)", txt)
    res = dict(zip(*[iter(m.group(1).split())] * 2)) if m else {}
    m = re.search(r"per layer (.*)", txt)
    per = m.group(1).split() if m else []
    try:
        lg = np.load(f"{D}/ab_{arm}_logits.npy")
    except OSError:
        lg = None
    return res, per, lg


ref = sys.argv[1]
r_res, r_per, r_lg = read(ref)
print(f"{ref}: {' '.join(f'{k} {v}' for k, v in r_res.items())}")
for arm in sys.argv[2:]:
    res, per, lg = read(arm)
    same = [k for k in r_res if res.get(k) == r_res[k]]
    diff = [k for k in r_res if res.get(k) != r_res[k]]
    dl = [i for i, (a, b) in enumerate(zip(per, r_per)) if a != b]
    kv = f"KV layers differ: {len(dl)}/{len(per)}" + (f", first {dl[0]}" if dl else "")
    if lg is not None and r_lg is not None:
        d = np.abs(lg.astype(np.float64) - r_lg.astype(np.float64))
        lgs = (f"logits max|d| {d.max():.4g}, {int((d > 0).sum())} of {d.size} differ, argmax {int(lg.argmax())} vs "
               f"{int(r_lg.argmax())}")
    else:
        lgs = "logits: missing"
    print(f"{arm}: equal {','.join(same) or '-'} | differ {','.join(diff) or '-'} | {kv} | {lgs}")
