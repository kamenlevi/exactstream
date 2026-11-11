"""A reference candidate's detail (RESULT_nax6; tools/hc8_detail.py with the arm as an argument): per set, token-identical
rows and where the arm first leaves ref2t and ref2b; discordant items vs ref2t (decision 4's pair) with the other
reference arms beside them, and for IFEval the instructions each arm missed; the items one arm alone gets wrong / right
(the noise around the verdict); rough AR speed.
Usage: .venv/bin/python tools/ref_detail.py [arm, default ref2tnax] [reference, default ref2t] [others, comma-separated]
  e.g. T0: tools/ref_detail.py t0_mode_T0 ref b12llff,floor,ref.r2,ref2t"""
import os, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evalsuite import decode_report as dr

OTHERS = ("ref2b", "ref2ct", "ref2th8", "ref.r2", "floor", "ref")


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main(arm, ref="ref2t", others=OTHERS):
    others = tuple(o for o in others if o not in (arm, ref))
    ARMS = (arm, ref) + others
    its = dr.items()
    V = {n: dr.load(n) for n in ARMS}
    keys = sorted(set.intersection(*(set(d) for d in V.values())))
    print("rows:", {n: len(d) for n, d in V.items()}, "common", len(keys))
    sc = {}
    for s in ("math", "ifeval", "humaneval"):
        ks = [k for k in keys if its.get(k, {}).get("set") == s]
        for n, d in V.items():
            sc.setdefault(n, {}).update(dr.SCORERS[s]({k: d[k] for k in ks}, its))
        tot = {n: 100 * sum(sc[n][k][1] for k in ks) / len(ks) for n in ARMS}
        print(f"\n== {s}: {len(ks)} rows; % correct: " + ", ".join(f"{n} {tot[n]:.1f}" for n in ARMS))
        for r in [x for x in dict.fromkeys((ref, "ref2b")) if x in V]:
            fd = [first_diff(V[arm][k]["completion_ids"], V[r][k]["completion_ids"]) for k in ks]
            dv = [x for x in fd if x is not None]
            print(f"   vs {r}: token-identical {len(fd) - len(dv)}/{len(ks)}; first divergence median "
                  f"{statistics.median(dv) if dv else '-'}, min {min(dv) if dv else '-'}")
        print(f"   discordant {arm} vs {ref} (1 = correct); first diff = token where {arm} leaves {ref}")
        for k in ks:
            a, b = sc[arm][k][1], sc[ref][k][1]
            if a == b:
                continue
            c, t = V[arm][k], V[ref][k]
            print(f"  {k:40s} {arm} {a:.0f} ({c['finish']}, {len(c['completion_ids'])} tok)  {ref} {b:.0f} "
                  f"({t['finish']}, {len(t['completion_ids'])} tok, cap {its[k]['max_tokens']})  first diff "
                  f"{first_diff(c['completion_ids'], t['completion_ids'])}  "
                  + "  ".join(f"{n} {sc[n][k][1]:.0f}" for n in others))
            if s == "ifeval":
                ids = its[k]["instruction_id_list"]
                for n in ARMS:
                    miss = [i for i, ok in zip(ids, sc[n][k][0]) if not ok]
                    print(f"      {n:8s} missed: {', '.join(miss) if miss else '-'}")
        print(f"   items one arm alone gets wrong (all {len(ARMS) - 1} others right) / alone right (all others wrong)")
        for n in ARMS:
            o = [m for m in ARMS if m != n]
            wrong = [k for k in ks if sc[n][k][1] == 0 and all(sc[m][k][1] == 1 for m in o)]
            right = [k for k in ks if sc[n][k][1] == 1 and all(sc[m][k][1] == 0 for m in o)]
            print(f"  {n:8s} alone wrong {len(wrong)} {wrong}, alone right {len(right)} {right}")
    # paused_s is all time with PAUSE set (decode waits there); a freeze happens inside a PAUSE and is in it already
    print("\n== rough AR speed per set (tokens / (row seconds - heat pause)); shared GPU, not an A/B")
    for s in ("math", "ifeval", "humaneval"):
        ks = [k for k in keys if its.get(k, {}).get("set") == s]
        out = []
        for n in (arm, ref):
            tok = sum(len(V[n][k]["completion_ids"]) for k in ks)
            sec = sum(V[n][k]["seconds"] - V[n][k].get("mem", {}).get("paused_s", 0) for k in ks)
            out.append(f"{n} {tok / sec:.2f} tok/s ({tok} tok)")
        print(f"  {s:9s} " + ", ".join(out))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "ref2tnax", a[1] if len(a) > 1 else "ref2t", tuple(a[2].split(",")) if len(a) > 2 else OTHERS)
