"""AUDIT 1105, test 3a (Fable C, Thu 5 Nov 2026, 07:1x; CPU only, one core, ~minutes). Bytes AND tiles per kept token on the 36
decision-map windows at the 45 GB (laysz45_256 per-layer MSB sizes) and 24 GB (147) pools, live-cut blocks, the ledger's
conventions as in bytes10_replay.py (plain LRU per layer, prompt rows warm the pool, one session per domain in walk order,
wrong-draft stand-ins p + 97, kept = min(k, fm - A) + 1; the bonus row gated by the next segment's first draft cf = the
QOM-BQ / DRAFT17 form). The LSB sidecar is not replayed (cloud3's convention, REPORT_P3 section 8).
Rules (k0 = the ranks a row LOADS; keep = the ranks it COMPUTES; 'members' = experts in the block's loads at that layer):
  EXACT      every row loads and keeps 10.
  QOM0BQ     today's 45 GB FAST preset (qom_keep.py: THR 0, KS 2): drafted rows load ranks 0-1; keep ranks 0-1 + any rank 2-9
             that is a member or resident after the block's loads; rows without a draft load and keep 10.
  B2D        cloud3's strict three-band table: k0 = 2 at cf >= 0.99, 5 on [0.9, 0.99), 8 on every other drafted row, 10 with no
             draft; keep = ranks < k0 + members; no resident keep.
  B2D-TWIN   the same k0 with the resident keep (cloud3 section 20's twin).
  SPLIT99    this audit's split form: loads as QOM0BQ (every drafted row ranks 0-1); keep: rows with cf >= 0.99 strict (ranks
             0-1 + members), every other drafted row as QOM (ranks 0-1 + members + residents), no-draft rows 10.
  SPLIT90    as SPLIT99 with the strict band at cf >= 0.9.
Misses are counted before the block (loads not resident), the engine-reachable convention. LRU order: the loads are inserted
first (evicting LRU victims), then every kept expert is touched row by row in rank order. Tiles = sum over layers of the
distinct kept experts per block, per kept token. Output: audit_split_replay.out / .json."""
import json, os, sys, time
from collections import OrderedDict
import numpy as np
sys.path.insert(0, os.path.expanduser("~/FableGod/fablec"))
from bytes10_replay import windows, live_blocks, POOLS, MR  # noqa: E402

RULES = ["EXACT", "QOM0BQ", "B2D", "B2D-TWIN", "SPLIT99", "SPLIT90"]
POOL_LINES = ["45", "24"] if "quick" not in sys.argv else ["45"]


def k0_of(rule, c):
    if rule == "EXACT" or c < 0: return 10
    if rule in ("QOM0BQ", "SPLIT99", "SPLIT90"): return 2
    if c >= 0.99: return 2
    if c >= 0.9: return 5
    return 8


def strict_row(rule, c):
    """True = no resident keep on this row"""
    if rule in ("EXACT", "QOM0BQ", "B2D-TWIN"): return False
    if rule == "B2D": return True
    if rule == "SPLIT99": return c >= 0.99
    if rule == "SPLIT90": return c >= 0.9
    return False


def run(ws, rule, Cm):
    pools = [OrderedDict() for _ in range(48)]
    miss = tiles = 0; kept = 0; nblk = 0
    for w in ws:
        ids = w["ids"]; a = w["a"]
        for l in range(48):                                   # warm: the prompt rows' experts in order
            c = pools[l]
            for e in ids[:a, l].ravel().tolist():
                if e in c: c.move_to_end(e)
                else:
                    c[e] = 1
                    if len(c) > Cm[l]: c.popitem(last=False)
        for rows, cfs, kp in live_blocks(w, bonus="next"):
            E = ids[rows]; R = len(rows); kept += kp; nblk += 1
            k0 = [k0_of(rule, c) for c in cfs]; st = [strict_row(rule, c) for c in cfs]
            for l in range(48):
                c = pools[l]
                loads = set()
                for r in range(R): loads.update(E[r, l, :k0[r]].tolist())
                for e in loads:                               # demand reads; misses before the block
                    if e in c: c.move_to_end(e)
                    else:
                        miss += 1; c[e] = 1
                        if len(c) > Cm[l]: c.popitem(last=False)
                keepset = set()
                for r in range(R):
                    row = E[r, l].tolist(); kk = k0[r]
                    keepset.update(row[:kk])
                    for e in row[kk:]:
                        if e in loads or (not st[r] and e in c): keepset.add(e)
                for e in keepset:
                    if e in c: c.move_to_end(e)
                tiles += len(keepset)
    return dict(mb=miss * MR / 1e6 / max(kept, 1), tiles=tiles / max(kept, 1), kept=kept, blocks=nblk)


W = windows(); res = {}; lines = []
t0 = time.time()
lines.append("| domain | pool | rule | SSD MB per kept token | tiles per kept token | tiles vs EXACT | tiles vs QOM0BQ | MB vs QOM0BQ |")
lines.append("|---|---|---|---|---|---|---|---|")
for pool in POOL_LINES:
    Cm = POOLS[pool][0]
    for dom in ("code", "ifeval", "multi"):
        for rule in RULES:
            r = run(W[dom], rule, Cm); res[f"{dom}|{pool}|{rule}"] = r
            ex = res[f"{dom}|{pool}|EXACT"]; q = res.get(f"{dom}|{pool}|QOM0BQ", r)
            lines.append(f"| {dom} | {pool} GB | {rule} | {r['mb']:.1f} | {r['tiles']:.0f} | x{r['tiles'] / ex['tiles']:.3f} | x{r['tiles'] / q['tiles']:.3f} | x{r['mb'] / max(q['mb'], 1e-9):.2f} |")
            print(lines[-1], flush=True)
lines.append(f"({time.time() - t0:.0f} s; blocks per domain " + ", ".join(f"{d} {res[f'{d}|45|EXACT']['blocks']} / kept {res[f'{d}|45|EXACT']['kept']}" for d in ("code", "ifeval", "multi")) + ")")
out = "\n".join(lines); print(out)
D = os.path.expanduser("~/FableGod/fablec")
open(os.path.join(D, "audit_split_replay.out"), "w").write(out + "\n"); json.dump(res, open(os.path.join(D, "audit_split_replay.json"), "w"), indent=1)
