"""Frugal routing x bit-plane expert cache (Breakthrough D1 x planes, 2026-10-20): trace replay on the 125B routing
traces (MoEPrefetch traces125: router top-20 ids per decode token and layer; code / chat / bulgarian / math, 800 tokens).

Expert = 4.9152M weights: 4 code planes of 0.614 MB + scales/biases 0.307 MB (2.765 MB at 4 bits). A plane cache keeps
per layer an H tier of MSB-k parts (k planes + scales: k=2 1.536 MB, k=3 2.150 MB) for C_H experts and an F tier of
the remaining 4-k planes (LSB parts) for C_F experts whose MSB part is resident; both LRU, fixed byte budgets.
Frugal rule per token (eager, one row) with plan bits b(r) per router rank (e.g. 4433332222):
  ranks 0-1: full 4 bits always: missing MSB part -> read MSB + LSB (waited); MSB resident, LSB not -> read LSB (waited)
  ranks 2-9: resident MSB -> use at min(b(r), planes resident) bits (b(r) = 4 needs the LSB part too, else MSB-k);
             missing -> the best-ranked resident expert among the row's ranks 10-19 (at min(b(r), resident) bits),
             else the rank is dropped. Nothing is read for ranks >= 2.
Baseline 'full': today's frugal (one tier of whole 4-bit experts, C = budget / 2.765 MB).
Prints per budget and layout: waited SSD MB/token, reads/token, dropped / substituted ranks per token, mean bits of the
used ranks 2-9, and the slot counts (the GPU emulation runs frugal with C = C_H slots + the plan's bits by position).
usage: plane_sim.py [FRAC ...]   (default 0.05 0.10; budget = FRAC x 512 x 2.765 MB per layer)"""
import sys
from collections import OrderedDict
import numpy as np

T = "/Users/kamen/MoEPrefetch/data/traces125/"; SETS = ["code", "chat", "bulgarian", "math"]; NL = 48; E = 512
PL, SC = 0.6144, 0.3072                       # MB per code plane, MB of scales + biases per expert
FULL = 4 * PL + SC
trs = {s: np.stack([np.load(T + f"{s}_800.npz")[f"L{l}"] for l in range(NL)], 1).astype(np.int32) for s in SETS}


def run(tr, k, CH, CF, plan):
    """one prompt's trace -> per-token reads MB, reads, drops, subs, mean bits of used ranks 2-9"""
    mb = reads = drops = subs = 0.0
    bits_sum = bits_n = 0
    sM, sL = k * PL + SC, (4 - k) * PL
    for l in range(NL):
        H, F = OrderedDict(), OrderedDict()
        for t in range(tr.shape[0]):
            row = [int(x) for x in tr[t, l]]
            used = set()
            for r in range(2):                                  # waited ranks: full 4 bits
                e = row[r]
                if e not in H:
                    mb += sM; reads += 1
                    H[e] = 1
                if k < 4 and e not in F:
                    mb += sL; reads += 1
                    F[e] = 1
                H.move_to_end(e)
                if k < 4: F.move_to_end(e)
                used.add(e)
            alt = [e for e in row[10:20] if e in H]
            for r in range(2, 10):
                e, b = row[r], int(plan[r])
                if e in H and e not in used:
                    x = e
                else:
                    x = next((y for y in alt if y not in used), None)
                    if x is None:
                        drops += 1
                        continue
                    subs += 1
                used.add(x)
                have = 4 if (k == 4 or x in F) else k
                bits_sum += min(b, have); bits_n += 1
                H.move_to_end(x)
                if b > k and x in F:
                    F.move_to_end(x)
            while len(H) > CH:
                v, _ = H.popitem(last=False)
                F.pop(v, None)
            while len(F) > CF:
                F.popitem(last=False)
    n = tr.shape[0]
    return mb / n, reads / n, drops / n, subs / n, bits_sum / max(1, bits_n)


fracs = [float(a) for a in sys.argv[1:]] or [0.05, 0.10]
print("waited SSD MB/token (reads/token) | dropped / substituted ranks 2-9 per token | mean bits of used ranks 2-9; mean of 4 prompts")
for frac in fracs:
    B = frac * E * FULL                                          # MB per layer
    print(f"\nbudget {frac:.0%} of the 4-bit experts = {B:.0f} MB per layer, {B * NL / 1000:.1f} GB of expert cache")
    rows = [("full 4-bit (today's frugal)", 4, int(B / FULL), 0, "4444444444")]
    for k, plan in ((3, "4433333333"), (3, "4433332222"), (2, "4422222222"), (2, "4444222222")):
        sM, sL = k * PL + SC, (4 - k) * PL
        for fshare in (0.0, 0.1, 0.2, 0.35, 0.5):               # share of the budget given to the LSB tier
            CF = int(B * fshare / sL)
            CH = int((B - CF * sL) / sM)
            if CF > CH: continue
            rows.append((f"MSB-{k} H={CH} F={CF} plan {plan}", k, CH, CF, plan))
    for label, k, CH, CF, plan in rows:
        mb, rd, dr, su, bt = np.mean([run(trs[s], k, CH, CF, plan) for s in SETS], 0)
        print(f"  {label:44s} {mb:6.1f} MB ({rd:5.1f}) | drop {dr:5.1f} sub {su:5.1f} | bits {bt:.2f}", flush=True)
