"""price_split_nocom.py (Fable FULL, LOSSY350, for the orchestrator's 16:4x ask): the comment-free mode beside the split
form, in the ENGINE's convention (live-cut blocks at t 0.3, the 45 GB pools), on the 12 code map windows.
Machinery: Fable C's bytes10_replay (windows, live_blocks, POOLS: plain LRU per layer, prompt rows warm the pool, the
bonus row gated by the next segment's first draft cf) and audit_split_replay's rules, copied here unchanged:
  EXACT    every row loads and keeps 10.
  QOM0BQ   today's preset: drafted rows load ranks 0-1; keep ranks 0-1 + members + residents; no-draft rows 10.
  SPLIT99  loads as QOM; rows at cf >= 0.99 strict (ranks 0-1 + members), the others as QOM.
  SPLIT90  the same with the strict band at cf >= 0.9.
Variants of the block list:
  all      the live blocks as recorded.
  nocom-L  comment-free, lower bound: rows whose input token is a comment token (or a comment-ending newline) are dropped,
           kept tokens that are comment tokens are not counted, blocks left with no kept token are dropped; the blocks are
           NOT re-formed (a real engine would draft longer blocks once the comments are gone).
  pure     the blocks with no comment token among their rows and kept tokens, read on their own (the pools still see every
           block): what a block looks like on comment-free text; biased up (the hard lines after comments are mostly not in).
Tiles = distinct kept experts per block summed over layers. Run: nice -n 15 ~/BigMoE/.venv/bin/python -I price_split_nocom.py"""
import importlib.util, os, sys
from collections import OrderedDict
import numpy as np
H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/FableGod/fablec")
from bytes10_replay import windows, live_blocks, POOLS, MR  # noqa: E402
from tokenizers import Tokenizer
TOK = Tokenizer.from_file(f"{H}/BigMoE/models/bare/tokenizer.json")
spec = importlib.util.spec_from_file_location("relax_core", f"{H}/EvalSuite/evalsuite/relax_core.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
WK = np.load(f"{H}/TrainLab/data/d02b_walk_hpd0_code.npz", allow_pickle=True)   # the project's own walk; pickle for names only
NAMES = [str(x) for x in WK["names"]]
RULES = ["EXACT", "QOM0BQ", "SPLIT99", "SPLIT90"]
def k0_of(rule, c): return 10 if (rule == "EXACT" or c < 0) else 2
def strict_row(rule, c): return (rule == "SPLIT99" and c >= 0.99) or (rule == "SPLIT90" and c >= 0.9)
def flags_of(nm):
    ids = WK[f"{NAMES.index(nm)}_ids"].astype(int).tolist(); sc = rc.CodeScan(); fl = []
    for i in ids:
        s = TOK.decode([i]); before = bool(sc.code and sc.com); anyc = False
        for ch in s:
            sc._ch(ch); anyc = anyc or bool(sc.code and sc.com)
        fl.append(1 if anyc else (2 if before and s.startswith("\n") else (1 if before else 0)))
    return fl
def run(ws, rule, Cm, variant):
    pools = [OrderedDict() for _ in range(48)]
    acc = dict(miss=0, tiles=0, kept=0, blocks=0, rows=0, sure=0, real=0)
    for w in ws:
        ids = w["ids"]; a = w["a"]; fl = w["fl"]; N = len(ids)
        for l in range(48):
            c = pools[l]
            for e in ids[:a, l].ravel().tolist():
                if e in c: c.move_to_end(e)
                else:
                    c[e] = 1
                    if len(c) > Cm[l]: c.popitem(last=False)
        for rows, cfs, kp in live_blocks(w, bonus="next"):
            A = rows[0]
            real = list(range(min(kp, len(rows))))                              # the first kp rows are real, the rest stand-ins
            ktok = [p for p in range(A + 1, A + kp + 1) if p < len(fl)]
            com_rows = [i for i in real if rows[i] < len(fl) and fl[rows[i]] != 0]
            com_kept = [p for p in ktok if fl[p] != 0]
            count = True; keep_idx = list(range(len(rows))); kept = kp
            if variant == "nocom-L":
                keep_idx = [i for i in range(len(rows)) if i not in set(com_rows)]
                kept = kp - len(com_kept)
                if kept <= 0 or not [i for i in keep_idx if i < kp]: continue    # the block is gone
            elif variant == "pure":
                count = not com_rows and not com_kept
            E = ids[[rows[i] for i in keep_idx]]; cf = [cfs[i] for i in keep_idx]; R = len(keep_idx)
            k0 = [k0_of(rule, c) for c in cf]; st = [strict_row(rule, c) for c in cf]
            t_blk = 0; m_blk = 0
            for l in range(48):
                c = pools[l]; loads = set()
                for r in range(R): loads.update(E[r, l, :k0[r]].tolist())
                for e in loads:
                    if e in c: c.move_to_end(e)
                    else:
                        m_blk += 1; c[e] = 1
                        if len(c) > Cm[l]: c.popitem(last=False)
                keepset = set()
                for r in range(R):
                    row = E[r, l].tolist(); kk = k0[r]
                    keepset.update(row[:kk])
                    for e in row[kk:]:
                        if e in loads or (not st[r] and e in c): keepset.add(e)
                for e in keepset:
                    if e in c: c.move_to_end(e)
                t_blk += len(keepset)
            if count:
                acc["miss"] += m_blk; acc["tiles"] += t_blk; acc["kept"] += kept; acc["blocks"] += 1; acc["rows"] += R
                acc["sure"] += sum(1 for x in cf if x >= 0.9); acc["real"] += sum(1 for i in keep_idx if i < kp)
    return acc
W = windows()["code"]
for w in W: w["fl"] = flags_of(w["nm"])
Cm = POOLS["45"][0]
print("12 code map windows, live-cut blocks, 45 GB pools. Per block: kept tokens, rows computed, tiles; per kept token: tiles, SSD MB.")
res = {}
for variant in ("all", "nocom-L", "pure"):
    for rule in RULES:
        r = run(W, rule, Cm, variant); res[(variant, rule)] = r
        b = max(1, r["blocks"])
        print(f"{variant:8s} {rule:8s}: blocks {r['blocks']:4d} | kept per block {r['kept'] / b:5.2f} | rows per block {r['rows'] / b:5.2f} (cf >= 0.9: {100 * r['sure'] / max(1, r['rows']):.0f} %) | "
              f"tiles per block {r['tiles'] / b:6.0f} | tiles per kept token {r['tiles'] / max(1, r['kept']):6.1f} | SSD MB per kept token {r['miss'] * MR / 1e6 / max(1, r['kept']):.1f}")
q = res[("all", "QOM0BQ")]; tq = q["tiles"] / q["blocks"]; kq = q["kept"] / q["blocks"]
print("\nPriced [P]: ms per check = (95 - E) + E x (tiles per block / today's), E = today's expert term 22 .. 27 ms [K AUDIT_1105 F2]; tok/s relative to today's QOM = 1.00")
for variant in ("all", "nocom-L", "pure"):
    for rule in ("QOM0BQ", "SPLIT99", "SPLIT90"):
        r = res[(variant, rule)]; b = r["blocks"]; out = []
        for E in (22.0, 27.0):
            ms = (95 - E) + E * (r["tiles"] / b) / tq
            out.append((ms, (r["kept"] / b) / ms / (kq / 95.0)))
        print(f"  {variant:8s} {rule:8s}: check {out[0][0]:.1f} .. {out[1][0]:.1f} ms, kept {r['kept'] / b:.2f} -> tok/s x{out[0][1]:.3f} .. x{out[1][1]:.3f}")
