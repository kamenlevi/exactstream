"""LAST-ROW reconciliation (NewIdeas, Wed 4 Nov 2026 03:4x; CPU only). Why my capture price for QOM (band45_capture_qom.txt,
the numbers behind Kamen's decision 2) and the ledger's part C differ on chat.

Two conventions for the last row of a live check (the row after the last verified draft):
  map     cf -1 = the bonus row stays exact (10 ranks, both parts). The map's live arms (qom_*_live, q1u_*_live) and
          ten_x_ledger.py part C use it: this is what the map's PASS verdicts cover.
  engine  the last row of EVERY live segment (a realized bonus row, or the discarded last row of a segment that ended in
          a miss) is gated by the cf of the draft it verifies: the walk's next draft after the cut (dd[L]); exact only when
          no such draft exists (a segment that used all its drafts). band45_oea.py's capture mode uses this ("cf per row =
          the draft that row verifies"; no draft only on a full check's 16th row, 1.9 % of code rows, 0 % on chat): my
          decision-2 prices were on this convention. (03:5x v2: v1 gated only realized bonus rows, by g[p].)
  all     the last row of every segment gated (cf 1.0): Fable C's '+bonus all' bound.
  disc    (03:5x) the engine's gate on DISCARDED last rows only (a segment that ended in a miss: its last row's output is
          thrown away and, under causal attention, cannot change a kept token); realized bonus rows stay exact. Its flips
          equal the map convention's (the map masks only realized rows), so the map's PASS already covers it.
Same replay as ten_x_ledger.py (plain LRU per layer, prompt warm-up, one session per domain, the three pool lines).
Writes ~/NewIdeas/data/ledger/ledger_lastrow.txt. Usage (BigMoE venv): python ledger_lastrow.py
"""
import os, sys, time
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ten_x_ledger as T  # noqa: E402
from dm_oea_masks import live_segments  # noqa: E402
import numpy as np  # noqa: E402
import dm_oea_masks  # noqa: E402
LIVE_T = 0.3   # the live cut, as dm_oea_masks.live_segments (asserted equal per check below)

RULES = [("EXACT", (None, False)), ("QOM cf 0.5", (0.5, False)), ("QOM cf 0.5 + 3b", (0.5, True)),
         ("QOM cf 0", (0.0, False)), ("QOM cf 0 + 3b", (0.0, True))]
CONV = ("map", "disc", "engine", "all")


def segments_nxt(k, m, A, dc):
    """dm_oea_masks.live_segments plus, per segment, the cf of the draft its last row verifies (-1: none left)"""
    segs, a0, kk, dd = [], A, k, [float(x) for x in dc[:m]]
    while True:
        L = int((np.cumprod(dd) >= LIVE_T).sum()) if dd else 0
        if L >= kk or L == len(dd):
            segs.append((kk, L, a0, dd, dd[L] if L < len(dd) else -1.0)); return segs
        segs.append((L, L, a0, dd[:L], dd[L]))
        a0, kk, dd = a0 + L + 1, kk - L - 1, dd[L + 1:]


def blocks_conv(w, conv):
    ids, g = w["ids"], w["g"]; N = len(ids); bl = []
    for (k0_, m0_, A0_), dc0_ in zip(w["checks"], w["dcf"]):
        segs = segments_nxt(k0_, m0_, A0_, dc0_)
        assert [x[:4] for x in segs] == live_segments(k0_, m0_, A0_, dc0_)   # the same cut as every live arm
        for (k, m, A, dc, nxt) in segs:
            last, fm = min(A + m, N - 1), min(A + k, N - 1)
            rows, cfs = [], []
            for p in range(A, last + 1):
                is_last = p - A == m
                if p <= fm:
                    rows.append(p); c = -1.0 if (k == m and is_last) else g[p]
                else:
                    rows.append((p + 97) % N); c = dc[p - A] if p - A < m else -1.0
                if is_last and conv == "engine": c = nxt                  # the draft this last row verifies, if any
                if is_last and conv == "all": c = 1.0
                if is_last and conv == "disc" and p > fm: c = nxt         # a discarded last row (after the first miss)
                cfs.append(c)
            if rows: bl.append((rows, cfs, min(k, fm - A) + 1))
    return bl


def run(task):
    dom, pool, rule, conv = task
    T.live_blocks = lambda w: blocks_conv(w, conv)               # replay() calls live_blocks(w) for case 'live'
    _, mbk, _, _, _ = T.replay((dom, pool, "live", rule[1]))
    return task, mbk


def main():
    t0 = time.time(); T.W = T.windows()
    nb = {}
    for dom, ws in T.W.items():                                   # segments, realized bonus rows, last rows with a draft
        n = rb = wd = 0
        for w in ws:
            for (k0_, m0_, A0_), dc0_ in zip(w["checks"], w["dcf"]):
                for (k, m, A, dc, nxt) in segments_nxt(k0_, m0_, A0_, dc0_):
                    n += 1; rb += int(k == m); wd += int(nxt >= 0)
        nb[dom] = (n, rb, wd)
    tasks = [(d, p, r, c) for d in T.W for p in T.POOLS for r in RULES for c in CONV]
    with mp.get_context("fork").Pool(int(os.environ.get("TXL_PROCS", "3"))) as P:
        res = dict(P.map(run, tasks))
    L = ["LAST-ROW reconciliation: SSD MB per kept token on today's live checks (plain LRU; ten_x_ledger conventions); "
         "map = bonus row exact (what the map tested), engine = bonus row gated by its own draft's cf (my capture price's "
         "convention), all = every bonus row gated (Fable C's bound); disc = the engine's gate on discarded last rows only "
         "(quality-free: the map's PASS covers it)"]
    for dom, (n, rb, wd) in nb.items():
        L.append(f"  {dom}: live segments {n}; ending in a realized bonus row {rb}; last row has a draft to verify (engine gates it) {wd}")
    L += ["| domain | pool | rule | map | disc | engine | all |", "|---|---|---|---|---|---|---|"]
    for d in T.W:
        for p in T.POOLS:
            e = res[(d, p, RULES[0], "map")]                          # EXACT is the same under every convention
            for r in RULES:
                cells = []
                for c in CONV:
                    v = res[(d, p, r, c)]
                    cells.append(f"{v:.1f}" + ("" if r[0] == "EXACT" else f" ({100 * (v / e - 1):+.0f} %)"))
                L.append(f"| {d} | {p} | {r[0]} | " + " | ".join(cells) + " |")
    L.append(f"({time.time() - t0:.0f} s; percentages vs EXACT under the map convention)")
    os.makedirs(T.DST, exist_ok=True)
    open(os.path.join(T.DST, "ledger_lastrow.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
