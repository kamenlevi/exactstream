"""REF2c-T detail (docs/RESULT_ref2ct_2026-11-02.md): discordant items per set, ref2ct vs ref2t (decision 4's pair), with
ref2b, ref.r2 and the floor beside them; per row: finish, tokens, the first token where ref2ct leaves ref2t, and for
IFEval the instructions each arm missed. Then how often each same-weights arm pair disagrees on IFEval (the noise
around the verdict). Usage: .venv/bin/python tools/ref2ct_detail.py"""
import itertools, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evalsuite import decode_report as dr

ARMS = ("ref2ct", "ref2t", "ref2b", "ref.r2", "floor")


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    its = dr.items()
    V = {n: dr.load(n) for n in ARMS}
    keys = sorted(set.intersection(*(set(d) for d in V.values())))
    print("rows:", {n: len(d) for n, d in V.items()}, "common", len(keys))
    sc = {}
    for s, fn in dr.SCORERS.items():
        ks = [k for k in keys if its.get(k, {}).get("set") == s]
        if not ks:
            continue
        for n, d in V.items():
            sc.setdefault(n, {}).update(fn({k: d[k] for k in ks}, its))
        tot = {n: sum(sc[n][k][1] for k in ks) for n in ARMS}
        print(f"\n== {s}: {len(ks)} rows; correct: " + ", ".join(f"{n} {tot[n]:.0f}" for n in ARMS))
        print(f"   discordant ref2ct vs ref2t (1 = correct); first diff = token where ref2ct leaves ref2t")
        for k in ks:
            a, b = sc["ref2ct"][k][1], sc["ref2t"][k][1]
            if a == b:
                continue
            c, t = V["ref2ct"][k], V["ref2t"][k]
            print(f"  {k:44s} ref2ct {a:.0f} ({c['finish']}, {len(c['completion_ids'])} tok)  ref2t {b:.0f} "
                  f"({t['finish']}, {len(t['completion_ids'])} tok)  first diff {first_diff(c['completion_ids'], t['completion_ids'])}  "
                  + "  ".join(f"{n} {sc[n][k][1]:.0f}" for n in ("ref2b", "ref.r2", "floor")))
            if s == "ifeval":
                ids = its[k]["instruction_id_list"]
                for n in ARMS:
                    miss = [i for i, ok in zip(ids, sc[n][k][0]) if not ok]
                    print(f"      {n:7s} missed: {', '.join(miss) if miss else '-'}")
    ks = [k for k in keys if its.get(k, {}).get("set") == "ifeval"]
    print("\n== IFEval: items scored differently by each pair of arms (b/c = first arm correct only / second arm correct only)")
    for x, y in itertools.combinations(ARMS, 2):
        bx = sum(1 for k in ks if sc[x][k][1] > sc[y][k][1])
        by = sum(1 for k in ks if sc[y][k][1] > sc[x][k][1])
        same = sum(1 for k in ks if V[x][k]["completion_ids"] == V[y][k]["completion_ids"])
        print(f"  {x:7s} vs {y:7s}: {bx}/{by}  (token-identical rows {same}/{len(ks)})")
    print("\n== IFEval items one arm alone gets wrong (the four others right) / alone right (the four others wrong)")
    for n in ARMS:
        o = [m for m in ARMS if m != n]
        wrong = [k for k in ks if sc[n][k][1] == 0 and all(sc[m][k][1] == 1 for m in o)]
        right = [k for k in ks if sc[n][k][1] == 1 and all(sc[m][k][1] == 0 for m in o)]
        print(f"  {n:7s} alone wrong {len(wrong)} {wrong}, alone right {len(right)} {right}")
    # paused_s is all time with PAUSE set (decode waits there); a freeze happens inside a PAUSE and is in it already
    print("\n== rough AR speed per set (tokens / (row seconds - heat pause)); shared GPU, not an A/B)")
    for s in ("math", "ifeval", "humaneval"):
        ks = [k for k in keys if its.get(k, {}).get("set") == s]
        out = []
        for n in ("ref2ct", "ref2t"):
            tok = sum(len(V[n][k]["completion_ids"]) for k in ks)
            sec = sum(V[n][k]["seconds"] - V[n][k].get("mem", {}).get("paused_s", 0) for k in ks)
            out.append(f"{n} {tok / sec:.2f} tok/s ({tok} tok)")
        print(f"  {s:9s} " + ", ".join(out))


if __name__ == "__main__":
    main()
