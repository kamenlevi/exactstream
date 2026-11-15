"""THINK_1115 #1 MARGIN-KEYED SHAPE (ZERO TIE-RUN + Lunatic + Fable C idea 1), NewIdeas, Sun 15 Nov 22:2x. CPU only.
Pre-registered (the orchestrator 22:2x): PASS = a lift >= 2x at k <= 3 AND kept/32 >= +3 % (keyed vs always tree-32 at
equal 32 rows) on HE+; MBPP reported; Fable C's line (+2 % on both sets) reported too.
(a) LIFT: along each window's rows, m(t) = the 125B's top-1 - top-2 margin (cells m at depth 0). P(m(t+k) < 1 |
    min m(t-2..t) < 2) vs P(m < 1), k = 1..6. Plus the walk-strict band 0-2 miss rate of tree-32's walk, bucketed by
    the start (hard / easy).
(b) KEYED SHAPE at 32 rows: at each check start, HARD if the previous check ended in a miss whose 125B margin is < 2
    nats, or any of its last 3 verified rows had margin < 2 (Fable C's rule); else EASY (incl. a full accept). Keyed =
    chain-32 after EASY, EAGLE-2 tree-32 (31 best-first pops) after HARD. Arms under certify98's walk: always chain-32,
    always tree-32, keyed, and the inverse (sanity). The first check of a window counts as HARD.
    Limitation: the HARD arm is EAGLE-2's own tree-32 (the 32-pop dumps hold no other shape); Fable C's wider shallow
    fork shape needs Tuesday's 127-node dumps.
usage: ~/BigMoE/.venv/bin/python -I -B margin_keyed.py"""
import os, sys, json
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/NewIdeas/research/week98"))
import tree_audit as TA

OUT = os.path.expanduser("~/NewIdeas/research/week98/margin_keyed")
K = 31
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


def main():
    TA.say = lambda s="": None
    res = {}
    for s, (tag, rtag, pub32, pub16) in TA.SETS.items():
        W = TA.load_windows(tag, pub32)
        tz = np.load(f"{TA.CE}/tree_t30b_{tag}.npz", allow_pickle=True)
        fi, an, nd, dp, ot = (tz[k] for k in ("fi", "an", "node", "depth", "on_true"))
        hit = ot & (nd < K); acc = {}
        for f, a_, d in zip(fi[hit], an[hit], dp[hit]):
            acc[(int(f), int(a_))] = max(acc.get((int(f), int(a_)), 0), int(d) + 1)
        # (a) lift along rows
        base_n = base_d = 0; lift = {k: [0, 0] for k in range(1, 7)}
        for X in W:
            assert (np.diff(X["an"]) == 1).all()
            m = X["M"][:-1, 0]                                   # margin at each anchor's depth-0 row, rows consecutive
            ok = ~np.isnan(m); base_n += int(((m < 1) & ok).sum()); base_d += int(ok.sum())
            for t in range(2, len(m)):
                w = m[t - 2:t + 1]
                if np.isnan(w).any() or w.min() >= 2:
                    continue
                for k in range(1, 7):
                    if t + k < len(m) and not np.isnan(m[t + k]):
                        lift[k][0] += int(m[t + k] < 1); lift[k][1] += 1
        base = base_n / base_d
        say(f"\n=== {s} ({tag}) ===  base P(m < 1) = {100 * base:.1f} %")
        say("  lift P(m(t+k) < 1 | min m(t-2..t) < 2) / base: " + "  ".join(
            f"k{k} {100 * v[0] / max(v[1], 1):.1f} % (x{v[0] / max(v[1], 1) / base:.2f})" for k, v in lift.items()))
        # (b) the keyed walk
        def run(policy):
            ks, buckets = [], {"HARD": [0, 0, 0], "EASY": [0, 0, 0]}   # [checks, band 0-2 misses, band 0-2 reached]
            for X in W:
                p, prev = 1, None
                while p < X["n"]:
                    if prev is None:
                        hard = True
                    else:
                        pp, pa = prev
                        miss_m = X["M"][pp, pa] if pa < K and X["rt"][pp, pa] >= 0 else np.inf
                        last = [X["M"][pp, j] for j in range(max(0, pa - 3), pa)]
                        hard = (miss_m < 2) or any((not np.isnan(v)) and v < 2 for v in last)
                    use_tree = policy(hard)
                    if p < len(X["an"]):
                        a = min(acc.get((X["fi"], int(X["an"][p])), 0), K) if use_tree else TA.lead_right(X["rt"][p], 0, K)
                    else:
                        a = 0
                    b = buckets["HARD" if hard else "EASY"]; b[0] += 1
                    for d in range(3):                           # walk-strict reach / miss at depths 0-2
                        if d < a:
                            b[2] += 1
                        elif d == a and a < K and p < len(X["an"]) and X["rt"][p, a] >= 0:
                            b[2] += 1; b[1] += 1; break
                        else:
                            break
                    ks.append(a + 1); prev = (p, a); p += a + 1
            return float(np.mean(ks)), buckets
        arms = {"always chain-32": lambda h: False, "always tree-32": lambda h: True,
                "KEYED (chain easy / tree hard)": lambda h: h, "inverse (tree easy / chain hard)": lambda h: not h}
        r = {}
        for name, pol in arms.items():
            k, bk = run(pol); r[name] = k
            sh = bk["HARD"][0] / max(bk["HARD"][0] + bk["EASY"][0], 1)
            say(f"  {name:34s} kept/32 {k:7.3f}  hard starts {100 * sh:4.1f} %  band 0-2 miss: hard "
                f"{100 * bk['HARD'][1] / max(bk['HARD'][2], 1):5.2f} %  easy {100 * bk['EASY'][1] / max(bk['EASY'][2], 1):5.2f} %")
        g = r["KEYED (chain easy / tree hard)"] / r["always tree-32"] - 1
        l3 = max(lift[k][0] / max(lift[k][1], 1) / base for k in (1, 2, 3))
        say(f"  keyed vs always tree-32: {100 * g:+.2f} %; max lift at k <= 3: x{l3:.2f}")
        res[s] = {"base": base, "lift": {k: v[0] / max(v[1], 1) / base for k, v in lift.items()}, "kept": r, "gain": g, "lift3": l3}
    he, mb = res["he"], res["mbpp"]
    v = "PASS" if (he["lift3"] >= 2 and he["gain"] >= 0.03) else "KILL"
    say(f"\nLINE (orchestrator: HE+ lift >= 2x at k <= 3 AND kept/32 >= +3 %): {v}  (HE+ lift x{he['lift3']:.2f}, gain {100 * he['gain']:+.2f} %)")
    say(f"Fable C's line (+2 % on both sets): {'PASS' if he['gain'] >= 0.02 and mb['gain'] >= 0.02 else 'KILL'} "
        f"(HE+ {100 * he['gain']:+.2f} %, MBPP {100 * mb['gain']:+.2f} %)")
    json.dump(res, open(OUT + ".json", "w"), indent=1, default=float)
    open(OUT + ".txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
