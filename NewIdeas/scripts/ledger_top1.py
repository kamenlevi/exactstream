"""TOP-1 PRICES (NewIdeas, Wed 4 Nov 2026 07:3x; CPU only). Screen 2b (qom_0_k1_live_t3: QOM cf 0, sure rows keep rank 0
at 3 bits + union members + resident ranks 1-9) PASSed x3 on the map (Breakthrough 07:27). This prices it, and the stack
it opens (+ the gated last row, QOM form), in SSD MB per kept token with ten_x_ledger's replay (plain LRU per layer,
prompt warm-up, one session per domain, misses counted before each block) and ledger_lastrow's conventions:
  map     the last row of a live segment exact (what the map's _live arms test; 2b as tested)
  engine  the last row gated by the draft it verifies (the map's _bq arms; exact when no draft is left)
Sure rows read rank 0's base record only (ranks 0..ks-1 for top-2), no LSB part under 3 bits. Two LSB variants for
top-1 + 3 bits, because the map served a RESIDENT rank 1 at 4 bits (the harness keeps it at the model's precision):
  b  no LSB on sure rows (an engine serving a resident rank 1 from its 3-bit base record; not what the map tested)
  a  a sure row's resident rank 1 also needs its LSB part (read on an LSB-pool miss): what the map tested
Writes ~/NewIdeas/data/ledger/ledger_top1.txt. Usage (BigMoE venv): python ledger_top1.py
"""
import os, sys, time
import multiprocessing as mp
from collections import OrderedDict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ten_x_ledger as T  # noqa: E402
import ledger_lastrow as LL  # noqa: E402

RULES = [("QOM cf 0 + 3b (top-2)", 0.0, 2, None), ("top-1 + 3b, LSB b", 0.0, 1, "b"), ("top-1 + 3b, LSB a", 0.0, 1, "a")]
CONV = ("map", "engine")


def replay(task):
    dom, pool, rule, conv = task
    _, thr, ks, var = rule
    Cm, Cl = T.POOLS[pool]
    bp = [OrderedDict() for _ in range(48)]; lp = [OrderedDict() for _ in range(48)]
    mb = ml = kept = 0
    for w in T.W[dom]:
        ids = w["ids"]
        for p in range(w["a"]):                                  # the prompt's prefill warms both pools
            for l in range(48):
                for e in ids[p, l].tolist(): T.touch(bp[l], e, Cm[l])
                for e in ids[p, l, :2].tolist(): T.touch(lp[l], e, Cl[l])
        for rows, cfs, kp in LL.blocks_conv(w, conv):
            sure = [c >= thr for c in cfs]
            for l in range(48):
                Ub, Ul = set(), set()
                for r, s_ in zip(rows, sure):
                    Ub.update(ids[r, l, :ks if s_ else 10].tolist())
                    if not s_: Ul.update(ids[r, l, :2].tolist())   # 3 bits on sure rows: no LSB part there
                mb += sum(1 for e in Ub if e not in bp[l])
                for e in Ub: T.touch(bp[l], e, Cm[l])
                if var == "a":                                   # a resident rank 1 on a sure row, served at 4 bits
                    for r, s_ in zip(rows, sure):
                        if s_ and int(ids[r, l, 1]) in bp[l]: Ul.add(int(ids[r, l, 1]))
                ml += sum(1 for e in Ul if e not in lp[l])
                for e in Ul: T.touch(lp[l], e, Cl[l])
            kept += kp
    return task, (mb * T.MR + ml * T.LR) / max(kept, 1) / 1e6


def main():
    t0 = time.time(); T.W = T.windows()
    tasks = [(d, p, r, c) for d in T.W for p in T.POOLS for r in RULES for c in CONV]
    with mp.get_context("fork").Pool(int(os.environ.get("TXL_PROCS", "3"))) as P:
        res = dict(P.map(replay, tasks, chunksize=1))
    L = ["TOP-1 PRICES: SSD MB per kept token on today's live checks (ten_x_ledger replay, before-block misses); "
         "map = last row exact (2b as tested), engine = last row gated by its draft (the _bq arms)",
         "| domain | pool | rule | map | engine |", "|---|---|---|---|---|"]
    for d in T.W:
        for p in T.POOLS:
            for r in RULES:
                L.append(f"| {d} | {p} | {r[0]} | " + " | ".join(f"{res[(d, p, r, c)]:.1f}" for c in CONV) + " |")
    L.append(f"({time.time() - t0:.0f} s)")
    os.makedirs(T.DST, exist_ok=True)
    open(os.path.join(T.DST, "ledger_top1.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
