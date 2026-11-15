"""WEEK98 H rerun on MBPP with the engine's start-AND-end guard (ENGINE_SPEC_relax.md), MoonShots Sun 15 Nov 2026.
The span test is now the validated SpanTracker (span_tracker.py): span_before = in span after the true prefix; span_after =
still in span after stepping the tracker over the FINAL drafted token itself. The token is known for every candidate: an
override's token is the 2B's top-1; the head's token is the 125B's top-32 entry at the cell's stored rank (candidates have
rank 1..k-1 <= 31). MBPP only: MTP Head's HE+ cells hold no rank on misses (they wait for the 06:00 dump).
usage: ~/BigMoE/.venv/bin/python -I relax_guard.py"""
import json, os, sys
import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/MoonShots/week98"))
import chooser as CH                                      # noqa: E402
from span_tracker import SpanTracker                      # noqa: E402

BANK = H("~/datasets/deepdraft/bank")
SHOW = [(0, 0), (7, 10), (15, 22), (23, 30)]
SLOTS = SpanTracker.__slots__[1:]


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(CH.TOKJ)

    class Lazy(dict):
        def __missing__(self, k):
            v = tok.decode([int(k)]); self[k] = v; return v
    table = Lazy()
    c = CH.attach_2b(CH.mbpp_cells(), H("~/MoonShots/week98/rows2b_mbpp.npz"), lambda t: 5)
    rank_head = np.load(H("~/BigMoE/data/deepdraft/ceiling/cells_t30b_mbpp100.npz"))["rank"].astype(np.int64)
    R = np.load(H("~/MoonShots/week98/rows2b_mbpp.npz"))
    top1 = {(os.path.basename(str(n)), int(r)): int(t) for n, r, t in zip(R["fname"], R["row"], R["top1"])}
    tau = np.array([CH.SCHED[int(d)] for d in c["depth"]], np.float32)
    ov = c["m2b"] >= tau
    audit_ok = np.where(ov, c["r2b"], c["right"])
    N = len(c["depth"])
    span_before = np.zeros(N, bool); final_rank = np.full(N, 99, np.int64); after_ok = {k: np.zeros(N, bool) for k in (2, 3, 5)}
    for k, nm in enumerate(c["files"]):
        sel = np.flatnonzero(c["fi"] == k)
        if not len(sel):
            continue
        z = np.load(os.path.join(BANK, nm)); ids = [int(x) for x in np.asarray(z["ids"])]; gen = int(z["gen"]); tk = np.asarray(z["tk"])
        tr = SpanTracker(table); snaps = {}
        for t in range(gen, len(ids)):                    # the state BEFORE token t (after the true prefix)
            snaps[t] = tuple(getattr(tr, s) for s in SLOTS); tr.update(ids[t])
        for i in sel:
            r = int(c["row"][i]); t = r + 1
            if t not in snaps:
                continue
            probe = SpanTracker(table)
            for s, v in zip(SLOTS, snaps[t]):
                setattr(probe, s, v)
            span_before[i] = probe.in_span()
            if audit_ok[i] or not span_before[i]:
                continue
            if ov[i]:
                d = top1.get((nm, r), -1); hit = np.flatnonzero(tk[r] == d); rk = int(hit[0]) if len(hit) else 99
            else:
                rk = int(rank_head[i]); d = int(tk[r][rk]) if 0 <= rk < 32 else -1
            final_rank[i] = rk
            if d < 0 or not (1 <= rk <= 4):
                continue
            probe.update(d)
            for kk in (2, 3, 5):
                after_ok[kk][i] = rk <= kk - 1 and probe.in_span()
    deep = (c["depth"] >= 23) & (c["depth"] <= 30)

    def bands(ok):
        return {(f"{a}-{b}" if a != b else str(a)): round(100 * float(ok[(c["depth"] >= a) & (c["depth"] <= b)].mean()), 2) for a, b in SHOW}

    def walk_all(ok):
        o = np.lexsort((c["depth"], c["an"], c["fi"])); f_, a_, k_ = c["fi"][o], c["an"][o], ok[o]
        st = np.r_[0, np.flatnonzero((np.diff(f_) != 0) | (np.diff(a_) != 0)) + 1, len(f_)]
        acc = [int(np.flatnonzero(~k_[s0:s1])[0]) if (~k_[s0:s1]).any() else s1 - s0 for s0, s1 in zip(st[:-1], st[1:])]
        return round(float(np.mean(acc)), 3)

    res = {"in_span_share_23_30_tracker": round(100 * float(span_before[deep].mean()), 1),
           "frozen": dict(bands=bands(audit_ok), accepted=walk_all(audit_ok))}
    miss23 = int((~audit_ok & deep).sum())
    for kk in (2, 3, 5):
        base = (~audit_ok) & span_before & (final_rank >= 1) & (final_rank <= kk - 1)
        for tag, relaxed in (("tracker, no guard", base), ("tracker + start-AND-end guard", after_ok[kk])):
            ok = audit_ok | relaxed
            res[f"R{kk} {tag}"] = dict(bands=bands(ok), accepted=walk_all(ok),
                                      fixed_share_23_30=round(float((relaxed & deep).sum()) / max(miss23, 1), 3),
                                      in_span_rows_deviating_23_30=round(100 * float((relaxed & deep).sum()) / max(int((deep & span_before).sum()), 1), 2))
    for k_, v in res.items():
        print(k_, v)
    json.dump(res, open(H("~/MoonShots/week98/relax_guard_mbpp.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
