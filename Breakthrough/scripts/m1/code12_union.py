#!/usr/bin/env python3
"""12 GB one-call fit (Fit12's answer, Mon 9 Nov 2026 21:0x): does a code check's REQUIRED expert set fit Fit12's one-call
pool (C_m 47 base slots, C_l 32 sidecar slots per layer) under each byte rule, at the decision map's live-cut blocks and at
k-row blocks (k 4 / 8 / 12 / 16)? Plus each rule's bytes under the replay's other pool policies (RCP, OPT) at the 12 GB
pools (51 + 32): the lossless policy lever. Breakthrough, Mon 9 Nov 2026 21:1x. CPU only, one core.

Core: bytes10_core.py as a library (its own main() writes into NewIdeas' folder and is NOT called). Rules: code12_replay's.
Required set per (check, layer): Ub = the experts whose MSB records the rule must hold (the ranks it keeps), Ul = the ranks
0-1 LSB sidecars it must hold. QOM's optional ranks 2-9 (used only when resident) are not required: a one-call check can drop
them when the pool is full, so they never block the fit (whether the map's c51 arm drops the same uses is a separate check).
k-row blocks: bytes10_core.synth_blocks(w, k, 0, 'walk'): k consecutive answer rows, the last one the bonus row (exact; for
'+ball' rules it takes cf 1.0 as in live_blocks(bonus='all')), drafted rows carry the walk's own cf.
Output: ~/Breakthrough/data/m1/code12_union.out and .json.
Usage: nice -n 15 ~/BigMoE/.venv/bin/python -I -B code12_union.py
"""
import json, os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bytes10_core as B  # noqa: E402

DST = os.path.expanduser("~/Breakthrough/data/m1")
B.RULES.update({
    "QOM0": (0.0, 2, 10, False, None, False, None),
    "QOM0+ball": (0.0, 2, 10, False, None, False, "all"),
})
FIT_RULES = ["EXACT", "Q1 0.5", "QOM0", "QOM0+ball", "Q1 0 t3 +bonus all"]
POLICY_RULES = ["EXACT", "QOM0", "QOM0+ball"]
KS = (4, 8, 12, 16)
CM_CALL, CL_CALL = 47, 32


def blocks_for(w, rule, how):
    bm = B.RULES[rule][6]
    if how == "live":
        return B.live_blocks(w, bm)
    bl = B.synth_blocks(w, int(how[1:]), 0, "walk")
    if bm == "all":
        bl = [(rows, list(cfs[:-1]) + [1.0], kp) for rows, cfs, kp in bl]
    return bl


def fit_stats(W, rule, how):
    nb, nl, rows, kept, blk_over_b, blk_over_b51, blk_over_l = [], [], [], [], 0, 0, 0
    for w in W:
        for Ub, Ul, Ua, kp, nr in B.block_sets(w, blocks_for(w, rule, how), rule, None):
            b = np.array([len(x) for x in Ub]); l = np.array([len(x) for x in Ul])
            nb.append(b); nl.append(l); rows.append(nr); kept.append(kp)
            blk_over_b += int((b > CM_CALL).any()); blk_over_b51 += int((b > 51).any()); blk_over_l += int((l > CL_CALL).any())
    nb = np.concatenate(nb); nl = np.concatenate(nl); n = len(rows)
    q = lambda a, p: float(np.percentile(a, p))
    return dict(blocks=n, rows=float(np.mean(rows)), kept=float(np.mean(kept)),
                b_mean=float(nb.mean()), b_p50=q(nb, 50), b_p90=q(nb, 90), b_p99=q(nb, 99), b_max=int(nb.max()),
                b_over47_layers=float((nb > CM_CALL).mean()), b_over47_blocks=blk_over_b / n,
                b_over51_blocks=blk_over_b51 / n,
                l_mean=float(nl.mean()), l_p90=q(nl, 90), l_max=int(nl.max()),
                l_over32_layers=float((nl > CL_CALL).mean()), l_over32_blocks=blk_over_l / n)


def main():
    t0 = time.time(); lines = []; res = {"fit": {}, "policy": {}}
    def say(s=""): print(s, flush=True); lines.append(s)
    W = B.windows()["code"]
    say(f"code12 union fit (one call per check; call pool C_m {CM_CALL} base / C_l {CL_CALL} sidecar per layer; "
        f"code windows {len(W)}; required sets only: QOM's resident-only ranks 2-9 are optional)")
    say("| rule | blocks | rows | kept | base ids per layer: mean / p50 / p90 / p99 / max | layers > 47 | checks with a layer > 47 "
        "(> 51) | sidecar ids: mean / p90 / max | checks with a layer > 32 |")
    say("|---|---|---|---|---|---|---|---|---|")
    for rule in FIT_RULES:
        for how in ("live",) + tuple(f"k{k}" for k in KS):
            s = fit_stats(W, rule, how); res["fit"][f"{rule}|{how}"] = s
            say(f"| {rule} {how} | {s['blocks']} | {s['rows']:.2f} | {s['kept']:.2f} | {s['b_mean']:.1f} / {s['b_p50']:.0f} / "
                f"{s['b_p90']:.0f} / {s['b_p99']:.0f} / {s['b_max']} | {s['b_over47_layers']:.1%} | {s['b_over47_blocks']:.1%} "
                f"({s['b_over51_blocks']:.1%}) | {s['l_mean']:.1f} / {s['l_p90']:.0f} / {s['l_max']} | {s['l_over32_blocks']:.1%} |")
        print(f"  ({rule}: {time.time() - t0:.0f} s)", flush=True)
    say("")
    Cm0, Cl0 = B.POOLS["12"]
    say(f"Pool policy at the 12 GB pools (C_m {Cm0[0]} / C_l {Cl0[0]}), live blocks: MB per kept token (GB per check)")
    say("| rule | LRU | RCP (rank-class pools, A = 0.7 C) | OPT (Belady bound) |")
    say("|---|---|---|---|")
    for rule in POLICY_RULES:
        seqs = [(B.warm_sets(w), B.block_sets(w, blocks_for(w, rule, "live"), rule, None)) for w in W]
        cells = []
        for pol in ("LRU", "RCP", "OPT"):
            r = B.simulate(seqs, Cm0, Cl0, pol)
            mbk = B.mbk(r); gbc = (r["mb"] * B.MR + r["ml"] * B.LR) / r["nchk"] / 1e9
            res["policy"][f"{rule}|{pol}"] = dict(mbk=mbk, gbc=gbc); cells.append(f"{mbk:.1f} ({gbc:.2f})")
        say(f"| {rule} | " + " | ".join(cells) + " |")
        print(f"  (policy {rule}: {time.time() - t0:.0f} s)", flush=True)
    say(f"done in {time.time() - t0:.0f} s")
    os.makedirs(DST, exist_ok=True)
    open(os.path.join(DST, "code12_union.out"), "w").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(DST, "code12_union.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
