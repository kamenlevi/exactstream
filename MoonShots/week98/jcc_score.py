"""WEEK98 J(CC) scorer (MoonShots, Sun 15 Nov 2026), as pre-registered in LEVERS_A_MOONSHOTS.md: pass@1 of the 2B's
close-call arms (jcc_decode.py: jcc/arm0.jsonl exact, arm1.jsonl, arm2.jsonl) against the exact arm, paired by key.
Scoring = EvalSuite's own scorers, imported read-only (run with -I -B; nothing is written under ~/EvalSuite):
HE+ = evalsuite/decode_report.score_humaneval (evalplus base + plus; humanevalx scored as HumanEval+),
MBPP = tools/mbpp_score.score (the 3 asserts in an isolated child, 10 s).
Lines (per delta, pooled over the keys both arms hold): HOLDS = drop <= 1.0 pt AND exact McNemar p > 0.2 AND neither set
loses with p < 0.05; FAILS = lower with p < 0.05 pooled OR drop > 3 pt; BETWEEN otherwise.
Also: R1-only commits per 1K generated tokens by class, answers changed, length ratio, cap hits, real kept per 32-row
check and its lift, discordant keys with the first changed token's class, the first-divergence check (the arm's ids
equal exact's up to its first R1 commit), self-check (a) (exact kept per check vs Sandbox98's s30 chain read-out).
usage: ~/EvalSuite/.venv/bin/python -I -B jcc_score.py [arms, default 1,2]"""
import json, os, sys
import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/EvalSuite")); sys.path.insert(0, H("~/EvalSuite/tools"))
J = H("~/MoonShots/week98/jcc")
CLASSES = ("code", "comment", "docstring", "string", "prose")
TF125 = {"humaneval": {1: 4.1, 2: 5.8}, "mbpp": {1: 8.1, 2: 12.8}}          # Breakthrough: 125B code changes per 1K kept
TF125_LIFT = {"humaneval": {1: 21.8, 2: 32.3}, "mbpp": {1: 23.1, 2: 46.5}}  # teacher-forced kept/32 lift, %
SB_READOUT = {"humaneval": 6.43, "mbpp": 4.42}                            # Sandbox98 gate.json s30 chain kept per 32


def load(name):
    p = os.path.join(J, name + ".jsonl")
    return {r["key"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}


def grp(r):
    return "mbpp" if r["set"] == "mbpp" else "humaneval"


def score_all(rows):
    from evalsuite import decode_report as DR
    import mbpp_score as MS
    its = {}
    for f in ("dec_humaneval", "dec_humanevalx"):
        for l in open(H(f"~/EvalSuite/data/items/{f}.jsonl")):
            it = json.loads(l); it = dict(it, set="humaneval"); its[it["key"]] = it
    he = DR.score_humaneval({k: r for k, r in rows.items() if grp(r) == "humaneval"}, its)
    mb = MS.score({k: r for k, r in rows.items() if grp(r) == "mbpp"}, MS.items())
    return {k: bool(v[0]) for k, v in list(he.items()) + list(mb.items())}


def mcnemar(b, c):
    from evalsuite.report import mcnemar_p
    return mcnemar_p(b, c)


def line(ref, arm, keys):
    n = len(keys)
    b = sum(1 for k in keys if arm[k] and not ref[k])     # arm-only passes
    c = sum(1 for k in keys if ref[k] and not arm[k])     # exact-only passes
    pr, pa = 100 * sum(ref[k] for k in keys) / max(n, 1), 100 * sum(arm[k] for k in keys) / max(n, 1)
    return dict(n=n, exact=round(pr, 2), arm=round(pa, 2), delta_pt=round(pa - pr, 2), arm_only=b, exact_only=c,
                p=round(mcnemar(b, c), 4))


def main():
    arms = [x for x in (sys.argv[1] if len(sys.argv) > 1 else "1,2,B").split(",")]
    ex = load("arm0")
    res = {"exact": {}, "arms": {}}
    cache_p = os.path.join(J, "scores.json")
    sc = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    for name in ["arm0"] + [f"arm{d}" for d in arms]:
        rows = load(name)
        todo = {k: r for k, r in rows.items() if k not in sc.get(name, {})}
        if todo:
            sc.setdefault(name, {}).update(score_all(todo))
    json.dump(sc, open(cache_p, "w"), indent=0)
    ref = sc["arm0"]
    # self-check (a) + the exact arm's own numbers
    for g in ("humaneval", "mbpp"):
        ch = [x for r in ex.values() if grp(r) == g for x in r["checks"]]
        k = float(np.mean(ch)) if ch else float("nan")
        res["exact"][g] = dict(keys=sum(1 for r in ex.values() if grp(r) == g), kept_per_check=round(k, 3),
                               sandbox98_readout=SB_READOUT[g], ratio=round(k / SB_READOUT[g], 3),
                               selfcheck_a="PASS" if abs(k / SB_READOUT[g] - 1) <= 0.15 else "FAIL",
                               cap_hits=sum(1 for r in ex.values() if grp(r) == g and r["stop"] == "cap"),
                               tokens=sum(r["n"] for r in ex.values() if grp(r) == g))
    # self-check (a), like for like (diagnostic): Sandbox98's own capture (the same 2B greedy, BatchGenerator, cap 512) on
    # its keys (MBPP 100, humaneval 60): answer identity, and the exact arm's kept per check over checks ending <= 512
    for g, sub, pre in (("humaneval", "he", "humaneval_"), ("mbpp", "mbpp", "mbpp_")):
        same = n = 0; ch = []
        for k, r in ex.items():
            if r["set"] != ("mbpp" if g == "mbpp" else "humaneval"):
                continue
            f = H(f"~/Sandbox98/data/{sub}/{k.replace('/', '_')}.npz")
            if not os.path.exists(f):
                continue
            z = np.load(f); sb = [int(x) for x in z["ids"][int(z["gen"]):]]
            m = min(len(sb), len(r["ids"]), 512)
            n += 1; same += int(sb[:m] == r["ids"][:m])
            pos = 1
            for c in r["checks"]:
                pos += c
                if pos > 512:
                    break
                ch.append(c)
        res["exact"][g]["like_for_like"] = dict(keys=n, same_answer_first512=same,
                                                kept_per_check_first512=round(float(np.mean(ch)), 3) if ch else None)
    for d in arms:
        arm = load(f"arm{d}")
        keys = sorted(set(arm) & set(ex) & set(sc[f"arm{d}"]) & set(ref))
        A = sc[f"arm{d}"]
        out = {"pooled": line(ref, A, keys)}
        for g in ("humaneval", "mbpp"):
            out[g] = line(ref, A, [k for k in keys if grp(ex[k]) == g])
        po = out["pooled"]
        set_loss = any(out[g]["delta_pt"] < 0 and out[g]["p"] < 0.05 for g in ("humaneval", "mbpp"))
        if (po["delta_pt"] < 0 and po["p"] < 0.05) or po["delta_pt"] < -3:
            out["verdict"] = "FAILS"
        elif po["delta_pt"] >= -1.0 and po["p"] > 0.2 and not set_loss:
            out["verdict"] = "HOLDS"
        else:
            out["verdict"] = "BETWEEN"
        # exposure, lengths, kept per check, divergence
        for g in ("humaneval", "mbpp", "pooled"):
            ks = [k for k in keys if g == "pooled" or grp(ex[k]) == g]
            ntok = sum(arm[k]["n"] for k in ks)
            ev = [e for k in ks for e in arm[k]["relaxed"]]
            by = {c: sum(1 for e in ev if e["cls"] == c) for c in CLASSES}
            tok_by = {c: sum(arm[k]["cls"][c] for k in ks) for c in CLASSES}
            chk_a = [x for k in ks for x in arm[k]["checks"]]
            chk_e = [x for k in ks for x in ex[k]["checks"]]
            ka, ke = (float(np.mean(chk_a)) if chk_a else float("nan")), float(np.mean(chk_e))
            out[g + "_behaviour"] = dict(
                tokens=ntok, exact_tokens=sum(ex[k]["n"] for k in ks), length_ratio=round(ntok / max(sum(ex[k]["n"] for k in ks), 1), 3),
                r1_per_1k=round(1000 * len(ev) / max(ntok, 1), 2),
                r1_per_1k_by_class={c: round(1000 * by[c] / max(ntok, 1), 2) for c in CLASSES},
                r1_per_1k_class_tokens={c: round(1000 * by[c] / max(tok_by[c], 1), 2) for c in CLASSES},
                token_share_by_class={c: round(tok_by[c] / max(ntok, 1), 3) for c in CLASSES},
                answers_changed=sum(1 for k in ks if arm[k]["ids"] != ex[k]["ids"]), answers=len(ks),
                cap_hits=sum(1 for k in ks if arm[k]["stop"] == "cap"), exact_cap_hits=sum(1 for k in ks if ex[k]["stop"] == "cap"),
                kept_per_check=round(ka, 3), exact_kept_per_check=round(ke, 3), kept_lift_pct=round(100 * (ka / ke - 1), 1),
                **({"tf125_code_per_1k": TF125[g][int(d)], "tf125_kept_lift_pct": TF125_LIFT[g][int(d)]} if g in TF125 and d.isdigit() and int(d) in TF125[g] else {}))
        # first divergence = the first R1 commit (identical loop otherwise)
        bad = []
        for k in keys:
            a_, e_ = arm[k]["ids"], ex[k]["ids"]
            fd = next((i for i, (x, y) in enumerate(zip(a_, e_)) if x != y), None if len(a_) == len(e_) else min(len(a_), len(e_)))
            fr = arm[k]["relaxed"][0]["pos"] if arm[k]["relaxed"] else None
            if fd != fr and not (fd is None and fr is None):
                bad.append((k, fd, fr))
        out["divergence_check"] = dict(ok=len(keys) - len(bad), bad=bad[:10])
        out["discordant"] = [dict(key=k, exact=ref[k], arm=A[k], r1=len(arm[k]["relaxed"]),
                                  first_cls=arm[k]["relaxed"][0]["cls"] if arm[k]["relaxed"] else None,
                                  first_pos=arm[k]["relaxed"][0]["pos"] if arm[k]["relaxed"] else None,
                                  first_dlm=arm[k]["relaxed"][0]["dlm"] if arm[k]["relaxed"] else None)
                             for k in keys if ref[k] != A[k]]
        res["arms"][f"delta{d}" if d.isdigit() else f"control_{d}"] = out
    json.dump(res, open(os.path.join(J, "jcc_result.json"), "w"), indent=1)
    print(json.dumps(res["exact"], indent=1))
    for d, out in res["arms"].items():
        print(f"== {d}: {out['verdict']}")
        for g in ("pooled", "humaneval", "mbpp"):
            print(f"  {g}: {out[g]}")
        for g in ("humaneval", "mbpp", "pooled"):
            print(f"  {g} behaviour: {out[g + '_behaviour']}")
        print(f"  divergence check: {out['divergence_check']}")
        print(f"  discordant: {len(out['discordant'])}")
        for x in out["discordant"]:
            print(f"    {x}")


if __name__ == "__main__":
    main()
