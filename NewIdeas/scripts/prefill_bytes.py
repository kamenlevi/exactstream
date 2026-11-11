"""PREFILL vs DECODE bytes under the FAST stack (NewIdeas, Wed 4 Nov 2026 14:5x; CPU only). The ledger replays count
decode only (misses in live checks per kept token). Here the same replay also counts the prompt's prefill reads (every
prompt row's top-10 through the same pool, as the warm-up loop does), per kept token of the window's answer:
  EXACT-style prefill reads 4-bit experts (MSB + LSB parts); FAST 3-bit prefill reads the MSB part only.
Pool: the 24 GB FAST pool (11-bit packed parts, POOLSPLIT 171 / 0); decode = C's rule with DRAFT17 (every row gated).
Writes ~/NewIdeas/data/ledger/prefill_bytes.txt."""
import os, sys, time
import multiprocessing as mp
from collections import OrderedDict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ten_x_ledger as T  # noqa: E402
import ledger_lastrow as LL  # noqa: E402
MR11, LR11, C = 2064384, 720896, 171


def run(dom):
    bp = [OrderedDict() for _ in range(48)]
    pf = dc = kept = ptoks = 0
    for w in T.W[dom]:
        ids = w["ids"]
        for p in range(w["a"]):
            for l in range(48):
                for e in ids[p, l].tolist():
                    if e not in bp[l]: pf += 1
                    T.touch(bp[l], e, C)
        ptoks += w["a"]
        for rows, cfs, kp in LL.blocks_conv(w, "all"):
            for l in range(48):
                U = set()
                for r in rows: U.update(ids[r, l, :2].tolist())
                dc += sum(1 for e in U if e not in bp[l])
                for e in U: T.touch(bp[l], e, C)
            kept += kp
    return dom, (pf, dc, kept, ptoks, len(T.W[dom]))


def main():
    t0 = time.time(); T.W = T.windows()
    with mp.get_context("fork").Pool(3) as P:
        R = dict(P.map(run, list(T.W)))
    L = ["PREFILL vs DECODE (24 GB FAST pool 171 / 0, 11-bit parts; decode = C + DRAFT17), MB per kept token of the answers",
         "| domain | windows | prompt tokens per window | kept per window | decode | prefill 4-bit | prefill FAST 3-bit | prefill share (4-bit / 3-bit) |",
         "|---|---|---|---|---|---|---|---|"]
    for d, (pf, dc, kept, pt, nw) in R.items():
        dmb = dc * MR11 / kept / 1e6; p4 = pf * (MR11 + LR11) / kept / 1e6; p3 = pf * MR11 / kept / 1e6
        L.append(f"| {d} | {nw} | {pt / nw:.0f} | {kept / nw:.0f} | {dmb:.1f} | {p4:.1f} | {p3:.1f} | "
                 f"{100 * p4 / (p4 + dmb):.0f} / {100 * p3 / (p3 + dmb):.0f} % |")
    L.append(f"({time.time() - t0:.0f} s; decode must equal trellis_draft17.txt's 24 GB MSB-3 11-bit row 19.5 / 14.3 / 10.6; "
             f"the map windows' answers are short, so prefill per kept token here is an upper end for long answers)")
    open(os.path.join(T.DST, "prefill_bytes.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
