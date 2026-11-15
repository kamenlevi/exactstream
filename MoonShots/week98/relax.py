"""WEEK98 H (MoonShots, Sun 15 Nov 2026), exactly as pre-registered in LEVERS_A_MOONSHOTS.md: LOSSY relaxed acceptance in
comments / docstrings on top of the frozen 2B audit (T16-HE's rule).
R(k): a wrong final token (after the audit) counts as accepted if the true prefix ends inside a '#' comment or a
triple-quoted docstring (ordinary strings excluded) AND the final token's rank in the 125B's top-32 at that row is <= k.
MBPP: the head's token rank from BigMoE's cells. HE+: MTP Head's cells hold no rank on misses -> lower / upper bounds.
Inputs read-only: the cells, the bank files (ids, gen, tk), rows2b_*.npz, the 2B tokenizer.
usage: ~/BigMoE/.venv/bin/python -I relax.py"""
import json, os, sys
import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/MoonShots/week98"))
import chooser as CH                                      # noqa: E402

BANK = H("~/datasets/deepdraft/bank")
SHOW = [(0, 0), (7, 10), (15, 22), (23, 30)]


def scan_after(text):
    """the mode AFTER each character: 0 code, 1 string, 2 comment, 3 triple-quoted docstring"""
    n = len(text); after = np.zeros(n, np.int8); i, mode, q = 0, 0, ""
    while i < n:
        ch = text[i]
        if mode == 0:
            if ch == "#":
                mode = 2; after[i] = 2; i += 1; continue
            if text.startswith('"""', i) or text.startswith("'''", i):
                q = text[i:i + 3]; mode = 3; after[i:i + 3] = 3; i += 3; continue
            if ch in "'\"":
                q = ch; mode = 1; after[i] = 1; i += 1; continue
            i += 1; continue
        if mode == 1:
            if ch == "\\":
                after[i:i + 2] = 1; i += 2; continue
            if ch == q or ch == "\n":
                mode = 0
            after[i] = mode; i += 1; continue
        if mode == 3:
            if ch == "\\":
                after[i:i + 2] = 3; i += 2; continue
            if text.startswith(q, i):
                mode = 0; i += 3; continue              # closing quote: after = 0
            after[i] = 3; i += 1; continue
        if ch == "\n":                                    # mode 2 (comment) ends at the newline
            mode = 0
        after[i] = mode; i += 1
    return after


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(CH.TOKJ)
    out = {}
    for name, cells, cf, r2f in (("mbpp", CH.mbpp_cells(), "cells_t30b_mbpp100.npz", "rows2b_mbpp.npz"),
                                 ("heplus", CH.heplus_cells(), "cells_t30b_heplus.npz", "rows2b_heplus.npz")):
        c = CH.attach_2b(cells, H(f"~/MoonShots/week98/{r2f}"), lambda t: 5)
        rank_head = np.load(H(f"~/BigMoE/data/deepdraft/ceiling/{cf}"), allow_pickle=True)["rank"].astype(np.int64)
        R = np.load(H(f"~/MoonShots/week98/{r2f}"))
        top1 = {(os.path.basename(str(n)), int(r)): int(t) for n, r, t in zip(R["fname"], R["row"], R["top1"])}
        N = len(c["depth"]); in_span = np.zeros(N, bool); rank2b = np.full(N, 99, np.int64)
        for k, nm in enumerate(c["files"]):
            sel = np.flatnonzero(c["fi"] == k)
            if not len(sel):
                continue
            z = np.load(os.path.join(BANK, nm)); ids = np.asarray(z["ids"]); gen = int(z["gen"]); tk = np.asarray(z["tk"])
            pieces = [tok.decode([int(i)]) for i in ids[gen:]]
            off = np.concatenate([[0], np.cumsum([len(p) for p in pieces])])   # char offset of token gen + j
            after = scan_after("".join(pieces))
            for i in sel:
                r = int(c["row"][i]); t = r + 1; j = t - gen
                in_span[i] = j > 0 and off[j] > 0 and after[off[j] - 1] in (2, 3)
                hit = np.flatnonzero(tk[r] == top1.get((nm, r), -1))
                rank2b[i] = int(hit[0]) if len(hit) else 99
        tau = np.array([CH.SCHED[int(d)] for d in c["depth"]], np.float32)
        ov = c["m2b"] >= tau
        audit_ok = np.where(ov, c["r2b"], c["right"])
        unknown = (~ov) & (~c["right"]) & (rank_head < 0)        # HE+: the head's missed token, rank not in the dump
        final_rank = np.where(ov, rank2b, rank_head)
        deep = (c["depth"] >= 23) & (c["depth"] <= 30)
        res = {"in_span_share_23_30": round(100 * float(in_span[deep].mean()), 1)}

        def bands(ok):
            return {(f"{a}-{b}" if a != b else str(a)): round(100 * float(ok[(c["depth"] >= a) & (c["depth"] <= b)].mean()), 2) for a, b in SHOW}

        def walk_all(ok):                                         # every row audited: consecutive accepted rows
            o = np.lexsort((c["depth"], c["an"], c["fi"])); f_, a_, k_ = c["fi"][o], c["an"][o], ok[o]
            st = np.r_[0, np.flatnonzero((np.diff(f_) != 0) | (np.diff(a_) != 0)) + 1, len(f_)]
            acc = []
            for s0, s1 in zip(st[:-1], st[1:]):
                z_ = np.flatnonzero(~k_[s0:s1]); acc.append(int(z_[0]) if len(z_) else s1 - s0)
            return round(float(np.mean(acc)), 3)

        res["head"] = dict(bands=bands(c["right"]), accepted=walk_all(c["right"]))
        res["frozen"] = dict(bands=bands(audit_ok), accepted=walk_all(audit_ok))
        for kk in (2, 3, 5):
            base = (~audit_ok) & in_span & (final_rank >= 1) & (final_rank <= kk - 1)
            for bound, relaxed in (("lower", base & ~unknown), ("upper", (base & ~unknown) | (unknown & in_span))):
                if name == "mbpp" and bound == "upper":
                    continue
                ok = audit_ok | relaxed
                miss23 = int((~audit_ok & deep).sum())
                tag = f"R{kk}" + ("" if name == "mbpp" else f"_{bound}")
                res[tag] = dict(bands=bands(ok), accepted=walk_all(ok),
                                fixed_share_of_frozen_misses_23_30=round(float((relaxed & deep).sum()) / max(miss23, 1), 3),
                                relaxed_per_100_cells_23_30=round(100 * float((relaxed & deep).mean()), 2),
                                in_span_rows_deviating_23_30=round(100 * float((relaxed & deep & in_span).sum()) / max(int((deep & in_span).sum()), 1), 2))
        out[name] = res
        print(f"===== {name}  (in-span rows at 23-30: {res['in_span_share_23_30']} %)")
        for kname, v in res.items():
            if isinstance(v, dict):
                extra = {k2: v[k2] for k2 in v if k2 not in ("bands", "accepted")}
                print(f"  {kname:9s} {v['bands']}  accepted(all audited) {v['accepted']}  {extra if extra else ''}")
    json.dump(out, open(H("~/MoonShots/week98/relax.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
